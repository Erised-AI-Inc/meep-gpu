#!/usr/bin/env python
"""DEVICE gate for the SCRATCH-OUTPUT off-diagonal D/E welds.

THIS GATE EXISTS TO SUPERSEDE ONE ARTIFACT, ON ITS OWN TERMS.

``parity/meep_gpu/results/triton_fused_offdiag_electric_2026-08-20`` is this
backend's own recorded refusal of the D->E off-diagonal pair. It measured, on an
RTX A6000 under both float32 subnormal policies, that a one-ordinary-launch weld
of the two SHIPPED IN-PLACE halves disagrees with its two-launch reference on
**42 of 60** subject cases at up to 6.58e-02, and that the disagreement is
SCHEDULE DEPENDENT — 4384 differing words at BLOCK=64 falling to 0 at BLOCK=1024
on a 4096-cell grid. Its four control legs (pointwise C1, zero-coefficient C2,
curl-is-identity C3, planted-pointwise P) all came back bit-identical, which is
what made the STENCIL, and nothing else in that weld, the defect.

That measurement is not disputed. What it measured is a weld in which ``step_D``
writes D while the off-diagonal constitutive arm reads its neighbours' D out of
the same volume. The products this gate certifies remove both premises: the curl
half writes to LAUNCH-LOCAL SCRATCH, the constitutive half RE-DERIVES every
foreign tap from pre-launch state, and the plan rotates the buffers afterwards.

So the two numbers this gate has to beat are the two that artifact recorded:

    S1 must be 60/60 IDENTICAL, where it was 42/60 DIVERGENT.
    S2 must be IDENTICAL AT EVERY BLOCK SIZE, where its differing-word count
       moved with the block size on every fixture that could arm the window.

Everything else here is the standing byte-identity standard: per COMPLETE driver
step as uint32 over every stored volume, against BOTH the array path AND the
separately certified kernels the weld replaces, both float32 subnormal policies,
launch counts by two independent counters, null controls that MUST diverge, and
mutations armed through the shipped launcher's own ``kernel=`` door.

Progress reporting: one flushed line per leg and one fsynced JSON write per leg group, on the
machine that owns the run.

    CUDA_VISIBLE_DEVICES=<n> TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub \\
      LIBRARY_PATH=$HOME/triton_libcuda_stub \\
      CUPY_CACHE_DIR=<fresh>/cupy_cache/ftz_stripped PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_offdiag_stencil_welds.py \\
      --subnormal-policy keep --out-root <fresh campaign>
    # -> <campaign>/offdiag_fused_electric_pair/gate.json and
    #    <campaign>/folded_offdiag_fused_electric_pair/gate.json, the layout
    #    seed_triton_welds.py derives each ledger key from

    # laptop: the structural legs only
    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_triton_offdiag_stencil_welds.py --no-device \\
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
import importlib
import json
import os
import socket
import sys
import textwrap
import time
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

GATE = "triton_offdiag_stencil_welds"

#: The artifact this gate supersedes, and the two numbers it has to beat.
SUPERSEDES = {
    "artifact": "results/triton_fused_offdiag_electric_2026-08-20",
    "its_S1": "42 of 60 DIVERGENT, max abs difference 6.58e-02 (keep cut)",
    "its_S2": "differing words moved with BLOCK: 4384 at 64 -> 0 at 1024",
    "what_it_measured": (
        "a one-ordinary-launch weld of the SHIPPED IN-PLACE halves, in which "
        "step_D writes D while the off-diagonal constitutive arm reads its "
        "neighbours' D out of the same volume"),
    "what_changed": (
        "the curl half writes launch-local SCRATCH and the constitutive half "
        "re-derives every foreign tap from pre-launch state; the plan rotates "
        "the D/fu references after the launch returns"),
}

#: Every module whose bytes this gate's verdict depends on.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/offdiag_fused_electric_pair.py",
    "meep_gpu/triton_kernels/folded_offdiag_fused_electric_pair.py",
    "meep_gpu/triton_kernels/offdiag_update_e.py",
    "meep_gpu/triton_kernels/folded_offdiag_update_e.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/symmetry.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/stepping.py",
    # THE DRIVER IS BOUND ON PURPOSE. This weld's whole `REPLACES` claim is about
    # WHICH passes run between `step_D` and `update_E`, and the reference this
    # gate compares against is that call order. A driver that reordered its seam
    # would leave every clause here reading correctly while the bytes described a
    # step nothing performs, so a driver move must withhold this credit and force
    # a re-run rather than be invisible.
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    # THE WARM LEGS DRIVE ``fastpath.warm_plan``, the plan-time path dispatch takes
    # before step 1 on a slot these products fill, so from 2026-09-15 its bytes are
    # part of this verdict; and the identity helper writes the ``environment`` block
    # the ledger seed reads.
    "meep_gpu/fastpath.py",
    "parity/meep_gpu/gate_triton_offdiag_stencil_welds.py",
    "parity/meep_gpu/probe_triton_offdiag_scratch_weld.py",
    "parity/meep_gpu/triton_device_identity.py",
)

#: Every stored volume the comparison walks, per COMPLETE driver step. A weld
#: that agreed on E while leaving fu_D wrong fails on step 2, so the census is
#: over the whole state and not over the sub-step's nominal outputs.
STATE_VOLUMES: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "Bx", "By", "Bz", "Hx", "Hy", "Hz",
)

VALUE_CLASSES: Tuple[str, ...] = ("normal", "signed_zero_lattice",
                                  "subnormal_band")

#: The block sizes S2 sweeps. 1024 is where the 2026-08-20 subject's race window
#: CLOSED (four programs on a 4096-cell grid) and 64 is where it was widest, so a
#: sweep that omitted either end would omit the measurement.
BLOCKS: Tuple[int, ...] = (64, 128, 256, 512, 1024)

#: Repeats per S2 point. The 2026-08-20 table is six repeats of one fixture at a
#: fixed block; a race is not repeatable, so a single run cannot distinguish
#: "identical" from "identical this time".
S2_REPEATS = 6

#: The blocks S1 runs at, and the split between the two legs is deliberate.
#: S1 measures BREADTH — five fixtures x three value classes x BOTH references —
#: and S2 measures the SCHEDULE, which is the axis the 2026-08-20 refusal moved
#: along: every block size, six repeats, on the grid that refusal used. Running
#: the whole block sweep in both legs would add nothing but compile time (this
#: kernel inlines the certified curl at sixteen cells, and a BLOCK=1024 compile
#: of it is minutes on a contended host), and it is the S2 leg that carries the
#: claim.
S1_BLOCKS: Tuple[int, ...] = (64, 256)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

FAMILIES: Dict[str, Dict[str, Any]] = {
    "offdiag_fused_electric_pair": {
        "module": "meep_gpu.triton_kernels.offdiag_fused_electric_pair",
        "kernel": "offdiag_fused_curl_constitutive_D",
        "planner": "plan_offdiag_fused_electric_pair",
        "predicate": "offdiag_fused_electric_pair_coverage",
        "carried_passes": ("zero_metal_D",),
        # The certified two-launch composition this weld replaces.
        "reference_launches": 2,
        "fixtures": (
            {"label": "walled_xy", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}},
            {"label": "periodic", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "periodic", "y": "periodic", "z": "periodic"}},
            {"label": "walled_3d", "cell": (0.6, 0.6, 0.6), "dimensions": 3,
             "boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"}},
            {"label": "walled_x", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "metallic", "y": "periodic", "z": "periodic"}},
            {"label": "walled_xy_4096", "resolution": 32.0,
             "cell": (2.0, 2.0, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}},
        ),
        # EXACTLY THE 2026-08-20 GRID. That artifact's schedule-dependence table
        # is 4096 cells: 64 programs at BLOCK=64 and FOUR at BLOCK=1024, which is
        # where it recorded the race window closing "entirely". A sweep on a
        # smaller grid would reach one program at the top and prove nothing
        # there, which is the vacuity this fixture exists to avoid.
        "s2_fixture": {"label": "walled_xy_4096", "resolution": 32.0,
                       "cell": (2.0, 2.0, 0.0),
                       "boundaries": {"x": "metallic", "y": "metallic",
                                      "z": "periodic"}},
    },
    "folded_offdiag_fused_electric_pair": {
        "module": "meep_gpu.triton_kernels.folded_offdiag_fused_electric_pair",
        "kernel": "folded_offdiag_fused_curl_constitutive_D",
        "planner": "plan_folded_offdiag_fused_electric_pair",
        "predicate": "folded_offdiag_fused_electric_pair_coverage",
        "carried_passes": ("fill_symmetry_bc_D", "zero_metal_D"),
        # Curl + one mirror_ghost_fill launch per folded axis + constitutive.
        "reference_launches": None,   # measured, not declared
        "fixtures": (
            {"label": "fold_y_even_walled", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
             "symmetry": (("Y", 1),)},
            {"label": "fold_y_odd_walled", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
             "symmetry": (("Y", -1),)},
            {"label": "fold_xy_mixed_walled", "cell": (1.2, 1.2, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
             "symmetry": (("X", -1), ("Y", 1))},
            {"label": "fold_x_walled_3d", "cell": (0.6, 0.6, 0.6),
             "dimensions": 3,
             "boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"},
             "symmetry": (("X", 1),)},
            {"label": "fold_y_odd_walled_4096", "resolution": 32.0,
             "cell": (2.0, 4.0, 0.0),
             "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
             "symmetry": (("Y", -1),)},
        ),
        "s2_fixture": {"label": "fold_y_odd_walled_4096", "resolution": 32.0,
                       "cell": (2.0, 4.0, 0.0),
                       "boundaries": {"x": "metallic", "y": "metallic",
                                      "z": "periodic"},
                       "symmetry": (("Y", -1),)},
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], out: Optional[str]) -> None:
    if not out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    return {name: sha256(os.path.join(API_ROOT, name))
            for name in SOURCES if os.path.exists(os.path.join(API_ROOT, name))}


def imported_source_sha256() -> Dict[str, str]:
    """Every ``meep_gpu`` / ``parity`` module the PROCESS actually imported.

    Not the declared list: a gate that hashed only what it meant to import
    cannot notice that it imported something else.
    """
    out: Dict[str, str] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not os.path.isfile(path):
            continue
        relative = os.path.relpath(os.path.abspath(path), API_ROOT)
        if relative.startswith("meep_gpu") or relative.startswith(
                os.path.join("parity", "meep_gpu")):
            out[relative] = sha256(path)
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seed_class(shape: Sequence[int], rng: np.random.Generator,
                value_class: str) -> np.ndarray:
    values = rng.standard_normal(tuple(shape)).astype(np.float32)
    if value_class == "normal":
        return values
    if value_class == "signed_zero_lattice":
        # Normal values with a +/-0 lattice over about a third of the cells. An
        # array of nothing but +/-0 leaves every route unmoved and makes the leg
        # VACUOUS, which the 2026-08-20 round measured and refused by name; the
        # lattice is what makes a sign observable without that.
        picks = rng.random(tuple(shape))
        values = np.where(picks < 0.18, np.float32(0.0), values)
        values = np.where((picks >= 0.18) & (picks < 0.36),
                          np.float32(-0.0), values)
        return values.astype(np.float32)
    if value_class == "subnormal_band":
        return (values * np.float32(1e-40)).astype(np.float32)
    raise ValueError(value_class)


def build_fixture(fixture: Dict[str, Any], value_class: str, seed: int,
                  xp: Any) -> Tuple[Any, Any]:
    """A real ``Grid``/``Fields``/``PML`` on ``xp``, off-diagonal rows live."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    rng = np.random.default_rng(seed)
    grid = Grid(resolution=float(fixture.get("resolution", 10.0)),
                cell_size=fixture["cell"],
                dimensions=fixture.get("dimensions", 2),
                boundaries=fixture["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in fixture.get("symmetry", ())),
                xp=xp)
    fields = Fields(grid=grid)
    shape = tuple(grid.shape)
    diagonal = {name: xp.asarray(np.full(shape, 2.25, dtype=np.float32))
                for name in ("Ex", "Ey", "Ez")}
    inverse = {name: xp.asarray(np.full(shape, 1.0 / 2.25, dtype=np.float32))
               for name in ("Ex", "Ey", "Ez")}
    # SPATIALLY VARYING rows: a uniform coefficient makes the Yee REGISTRATION of
    # the off-diagonal entry invisible, which the certified off-diagonal gate
    # records as its own measured refinement.
    rows = {
        "Ex": {"Ey": 0.05, "Ez": 0.03},
        "Ey": {"Ez": 0.02, "Ex": 0.04},
        "Ez": {"Ex": 0.01, "Ey": 0.06},
    }
    rows = {row: {partner: xp.asarray(
        (base + 0.01 * rng.random(shape)).astype(np.float32))
        for partner, base in entries.items()}
        for row, entries in rows.items()}
    fields.set_epsilon_volumes(diagonal, inverse, rows)
    fields.enable_pml_storage()
    folded = {axis for axis, _phase in fixture.get("symmetry", ())}
    thickness: Dict[str, Any] = {}
    for axis, kind in fixture["boundaries"].items():
        if axis.upper() in folded:
            # cell 0 lies ON the mirror plane, which is a boundary condition and
            # not an absorber; `PML` refuses the low face there by name.
            thickness[axis] = {"high": 0.2}
        elif kind == "metallic":
            thickness[axis] = 0.2
    pml = PML(grid=grid, thickness=thickness or {"x": 0.2})
    for name in STATE_VOLUMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = xp.asarray(_seed_class(array.shape, rng, value_class))
    return fields, pml


def _words(array: Any) -> np.ndarray:
    host = array.get() if hasattr(array, "get") else np.asarray(array)
    return np.ascontiguousarray(host).view(np.uint32).reshape(-1)


def snapshot(fields: Any) -> Dict[str, np.ndarray]:
    return {name: _words(getattr(fields, name))
            for name in STATE_VOLUMES if getattr(fields, name, None) is not None}


def compare(left: Dict[str, np.ndarray],
            right: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Differing uint32 words per volume, the total, and the words compared.

    UINT32 AND NOT FLOAT: ``-0.0 != 0.0`` is False, so a float comparison is
    blind exactly where the signed-zero class matters.
    """
    out: Dict[str, Any] = {}
    total = 0
    compared = 0
    if set(left) != set(right):
        raise AssertionError("the two states hold different volumes")
    for name in sorted(left):
        differ = int(np.count_nonzero(left[name] != right[name]))
        compared += int(left[name].size)
        if differ:
            out[name] = differ
        total += differ
    out["_total"] = total
    out["_compared"] = compared
    return out


def moved(before: Dict[str, np.ndarray],
          after: Dict[str, np.ndarray]) -> int:
    """How many words the reference itself moved. THE NON-VACUITY FLOOR.

    A leg on a state nothing changed reports identity for free. Compared as
    uint32 words for the same reason the verdict is: a float floor is blind to a
    ``+0.0 -> -0.0`` move, which is exactly the class the signed-zero fixture is
    built to produce.
    """
    return compare(before, after)["_total"]


# ---------------------------------------------------------------------------
# The two routes
# ---------------------------------------------------------------------------

def array_path_step(fields: Any, pml: Any) -> None:
    """The driver's own D-seam order, on the ARRAY PATH, no kernel involved.

    ``driver.py:3302-3313`` with no electric source (every row these cells serve
    declares none): ``step_D``; ``fill_symmetry_bc_D``; ``zero_metal_D``;
    ``fill_folded_far_ghosts_D``; ``update_E``. All three passes are called,
    including ones a family refuses — a reference that dropped a pass would agree
    with a weld that dropped it too.
    """
    from meep_gpu import stepping

    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


class _CountingKernel:
    """A launch counter that wraps a Triton kernel without changing its bytes.

    ONE OF THE TWO INDEPENDENT COUNTERS. It sits at the ``kernel[grid](...)``
    boundary and counts dispatches; the plan's own ``launches`` counts calls to
    ``run``. The two disagree exactly when a plan launches more than once per
    run, which is the thing a fused product's whole claim denies.
    """

    __slots__ = ("inner", "count", "grids")

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.count = 0
        # Every grid the plan subscripted the kernel with, in order. The warm leg
        # reads it: a warm is ONE subscript at ``(0,)``, a step one at the plan's
        # own grid.
        self.grids: List[Tuple[int, ...]] = []

    def __getitem__(self, grid: Any) -> Any:
        self.grids.append(tuple(int(n) for n in grid))
        launcher = self.inner[grid]

        def run(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return launcher(*args, **kwargs)

        return run


def certified_composition(fields: Any, pml: Any, family: str) -> Tuple[Any, int]:
    """The separately certified kernels this weld replaces, composed.

    Curl arm, then the family's carried in-seam passes as their own certified
    kernels where one exists (``symmetry.mirror_ghost_fill`` for the mirror
    fill), then the constitutive arm. Returns a callable and the launch count one
    complete step costs.
    """
    from meep_gpu import stepping
    from meep_gpu.triton_kernels import launch as launch_module
    from meep_gpu.triton_kernels import offdiag_update_e as offdiag
    from meep_gpu.triton_kernels import symmetry as symmetry_module
    from meep_gpu.triton_kernels import folded_offdiag_update_e as folded_offdiag

    folded = family.startswith("folded_")
    if folded:
        curl = symmetry_module.plan_folded_pml_curl(fields, pml, "step_D")
        constitutive = folded_offdiag.plan_folded_offdiagonal_constitutive(
            fields, pml)
        fill = symmetry_module.plan_mirror_ghost_fill(fields, "D")
    else:
        curl = launch_module.plan_pml_curl(fields, pml, "step_D")
        constitutive = offdiag.plan_offdiagonal_constitutive(fields, pml)
        fill = None
    if curl is None or constitutive is None:
        raise AssertionError(
            f"the certified {family} composition refused this fixture; the weld "
            f"cannot be compared against a reference that does not exist")

    launches = 2 + (len(getattr(fill, "axes", ())) if fill is not None else 0)

    def run() -> None:
        curl.run()
        if fill is not None:
            fill.run()
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        constitutive.run()

    return run, launches


# ---------------------------------------------------------------------------
# Mutations — armed through the shipped launcher's own kernel= door
# ---------------------------------------------------------------------------

#: Per family: ``(tag, target, old, new, expected, fixture)``. The last field is
#: the fixture the mutation is SCORED ON, and it is not decoration — the first
#: folded run of this gate declared three mutations CAUGHT and measured two of
#: them NULL, because both are only observable on an ODD fold plane and the
#: default fixture is an even one. ``None`` means the family's first fixture.
#: A mutation scored where it cannot fire is a disarmed leg wearing a pass.
MUTATIONS: Dict[str, Tuple[Tuple[str, str, str, str, str, Optional[str]], ...]] = {
    "offdiag_fused_electric_pair": (
        ("m_flatten_curl_parens", "kernel",
         "curl0 = dtdx * ((c_y - c) + (b - b_z))",
         "curl0 = dtdx * (((c_y - c) + b) - b_z)",
         "CAUGHT", None),
        ("m_mispair_row_coefficients", "kernel",
         "u01, idx,\n                ui * nyz + j * nz + k,\n"
         "                live, uvx)",
         "u02, idx,\n                ui * nyz + j * nz + k,\n"
         "                live, uvx)",
         "CAUGHT", None),
        ("m_seam_clear_dropped", "kernel",
         "        if ZM_X:\n            v1 = tl.where(at_x, 0.0, v1)",
         "        if ZM_X:\n            v1 = v1",
         "CAUGHT", None),
        ("m_tap_collapsed_to_own_cell", "kernel",
         "tap_yd_1 = _tap(1, i, dj, k, dvy,",
         "tap_yd_1 = _tap(1, i, j, k, live,",
         "CAUGHT", None),
        ("m_tap_reads_the_up_face", "kernel",
         "tap_xu_1 = _tap(1, ui, j, k, uvx,",
         "tap_xu_1 = _tap(1, di, j, k, dvx,",
         "CAUGHT", None),
        ("m_aux_takes_the_resolved_value", "kernel",
         "tl.store(a0s + idx, aux0, mask=live)",
         "tl.store(a0s + idx, own0, mask=live)",
         "CAUGHT", None),
        ("m_reload_own_D_from_memory", "kernel",
         "        gs0 = dfin0\n",
         "        gs0 = tl.load(d0s + idx, mask=live, other=0.0)\n",
         "NULL", None),
    ),
    "folded_offdiag_fused_electric_pair": (
        ("m_flatten_curl_parens", "kernel",
         "curl0 = dtdx * ((c_y - c) + (b - b_z))",
         "curl0 = dtdx * (((c_y - c) + b) - b_z)",
         "CAUGHT", None),
        ("m_mispair_row_coefficients", "kernel",
         "u01, idx,\n                ui * nyz + j * nz + k,\n"
         "                live, uvx, wy, MG_Y)",
         "u02, idx,\n                ui * nyz + j * nz + k,\n"
         "                live, uvx, wy, MG_Y)",
         "CAUGHT", None),
        # SCORED ON THE ODD PLANE, and the first run of this gate is why. On an
        # EVEN plane `PH_Y` is +1 and `PH_Y * value` IS `value`, so dropping the
        # multiply moves no byte: declared CAUGHT, measured NULL, on the even
        # fixture. The parity is only observable where the parity is not the
        # identity.
        ("m_fill_phase_dropped", "kernel",
         "                value = tl.where(near_y, PH_Y * value, value)",
         "                value = tl.where(near_y, value, value)",
         "CAUGHT", "fold_y_odd_walled"),
        ("m_fill_row_is_not_the_mirror_row", "kernel",
         "                j = tl.where(near_y, MIRROR_ROW, j)",
         "                j = tl.where(near_y, 1, j)",
         "CAUGHT", None),
        # THE ORDER OF THE TWO CARRIED PASSES, as a real swap of the two blocks.
        # The first version of this mutation neutralised the clear's guard under
        # `if ZM_Y:` and measured NULL — correctly, and for a reason worth
        # keeping: `zero_metal_axes` excludes a MIRRORED axis, so `ZM_Y` is 0 on
        # every fixture this family serves and that block never compiles. The
        # order is observable only where the two passes meet on DIFFERENT axes:
        # at the corner (i=0, j=0) of an x-walled, y-folded grid, component Dz has
        # Yee shift 0 on both, so the fill's parity and the wall's zero both fire.
        # Fill-then-clear gives +0.0; clear-then-fill gives `PH_Y * (+0.0)`, which
        # is -0.0 on an ODD plane and +0.0 on an even one — so this too is scored
        # on the odd fixture, and what it moves is signed-zero WORDS, which is
        # exactly why this gate compares uint32 and not floats.
        ("m_clear_before_fill", "kernel",
         "        if COMP != 0:\n            if MG_X:\n"
         "                value = tl.where(near_x, PH_X * value, value)\n"
         "        if COMP != 1:\n            if MG_Y:\n"
         "                value = tl.where(near_y, PH_Y * value, value)\n"
         "        if COMP != 2:\n            if MG_Z:\n"
         "                value = tl.where(near_z, PH_Z * value, value)\n"
         "\n"
         "        # --- the wall clear, AFTER the fill (the driver's order).\n"
         "        if COMP != 0:\n            if ZM_X:\n"
         "                value = tl.where(near_x, 0.0, value)\n"
         "        if COMP != 1:\n            if ZM_Y:\n"
         "                value = tl.where(near_y, 0.0, value)\n"
         "        if COMP != 2:\n            if ZM_Z:\n"
         "                value = tl.where(near_z, 0.0, value)\n",
         "        if COMP != 0:\n            if ZM_X:\n"
         "                value = tl.where(near_x, 0.0, value)\n"
         "        if COMP != 1:\n            if ZM_Y:\n"
         "                value = tl.where(near_y, 0.0, value)\n"
         "        if COMP != 2:\n            if ZM_Z:\n"
         "                value = tl.where(near_z, 0.0, value)\n"
         "\n"
         "        # --- the fill's parity, MOVED BELOW the clear (the mutation).\n"
         "        if COMP != 0:\n            if MG_X:\n"
         "                value = tl.where(near_x, PH_X * value, value)\n"
         "        if COMP != 1:\n            if MG_Y:\n"
         "                value = tl.where(near_y, PH_Y * value, value)\n"
         "        if COMP != 2:\n            if MG_Z:\n"
         "                value = tl.where(near_z, PH_Z * value, value)\n",
         "CAUGHT", "fold_y_odd_walled"),
        ("m_tap_collapsed_to_own_cell", "kernel",
         "tap_yd_1 = _d_final(1, i, dj, k, dvy,",
         "tap_yd_1 = _d_final(1, i, j, k, live,",
         "CAUGHT", None),
        ("m_ghost_weight_wrong_axis", "kernel",
         "live, uvx, wy, MG_Y)", "live, uvx, wx, MG_Y)",
         "CAUGHT", None),
        # `own0`, NOT `dfin0`. The first run declared this against `dfin0`, which
        # in THIS kernel is bound three lines BELOW the auxiliary store, so the
        # mutant died with a NameError and the leg reported REFUSED TO COMPILE —
        # a mutation that cannot build measures nothing.
        ("m_aux_takes_the_resolved_value", "kernel",
         "tl.store(a0s + idx, aux0, mask=live)",
         "tl.store(a0s + idx, own0, mask=live)",
         "CAUGHT", None),
        ("m_reload_own_D_from_memory", "kernel",
         "        gs0 = dfin0\n",
         "        gs0 = tl.load(d0s + idx, mask=live, other=0.0)\n",
         "NULL", None),
    ),
}

#: Why each mutation is a defect worth arming against, and what it would look
#: like in a run if it shipped.
MUTATION_REASONS: Dict[str, str] = {
    "m_flatten_curl_parens":
        "the curl's association is re-parsed; `(a + b) - c` and `a + (b - c)` "
        "round differently in float32. The 2026-08-20 round recorded this "
        "mutation as a FALSE NULL on its first device run because the "
        "replacement it used re-parsed to the original — the spelling here is "
        "the one that mutated",
    "m_mispair_row_coefficients":
        "the row coefficient is bound to the other partner; the certified "
        "off-diagonal gate's own m3",
    "m_seam_clear_dropped":
        "the in-seam wall pass stops writing one component, so a wall cell "
        "keeps its stepped value and every stencil tap that reads it moves",
    "m_tap_collapsed_to_own_cell":
        "the foreign tap becomes a second read of this cell — the POINTWISE "
        "plant. In the 2026-08-20 in-place weld this made the disagreement "
        "vanish; here it must MAKE one, which is what proves the stencil is "
        "live in the shipped kernel rather than compiled away",
    "m_tap_reads_the_up_face":
        "the up-face tap steps DOWN instead: the half-cell registration error, "
        "in the direction the certified body's docstring point 2 names",
    "m_aux_takes_the_resolved_value":
        "the split-field auxiliary takes the SEAM-RESOLVED displacement rather "
        "than the raw stepped one. Neither in-seam pass walks fu_*, so this is "
        "a pass applied to a volume the driver never touches — invisible in E "
        "on step 1 and wrong from step 2 on",
    "m_reload_own_D_from_memory":
        "the constitutive half re-loads its own cell's displacement from the "
        "scratch this program just wrote instead of using the register. "
        "PREDICTED NULL: one program's store and load of one address are "
        "ordered, so this is inert — and confirming it is what shows the "
        "correctness comes from the FOREIGN taps and not from the register",
    "m_fill_phase_dropped":
        "the mirror fill's parity is dropped, so an odd plane images the ghost "
        "with the wrong sign",
    "m_fill_row_is_not_the_mirror_row":
        "the near redirect reads stored row 1 instead of MIRROR_ROW=2; MEEP's "
        "little_owned_corner0 puts the image at 2 and row 1 is a real cell",
    "m_clear_before_fill":
        "the two carried passes run in the driver's order REVERSED. Observable "
        "only at a corner on a fold plane of one axis and a wall of another, and "
        "only as a signed zero: fill-then-clear leaves +0.0, clear-then-fill "
        "leaves PH * (+0.0), which is -0.0 on an odd plane",
    "m_ghost_weight_wrong_axis":
        "the mirror ghost weight of one axis is applied to another axis's "
        "ghost lane",
}


def mutated_module(family: str, old: str, new: str, tag: str) -> Any:
    """Import a COPY of the family module with one source edit applied.

    The mutation reaches the launch through the shipped planner's ``kernel=``
    door, which is the door the product itself documents; a harness that
    launched the shipped kernel instead would report a pass for a defect it
    never introduced, and that has happened in this campaign before.
    """
    import linecache
    import types

    source_path = os.path.join(
        API_ROOT, FAMILIES[family]["module"].replace(".", os.sep) + ".py")
    text = open(source_path, encoding="utf-8").read()
    hits = text.count(old)
    if hits != 1:
        raise AssertionError(
            f"{family}/{tag}: the mutation needle matches {hits} times, not "
            f"once; a mutation that matched nothing is a disarmed leg and one "
            f"that matched twice is two defects")
    mutated_text = text.replace(old, new, 1)
    if mutated_text == text:
        raise AssertionError(f"{family}/{tag}: the mutation changed nothing")
    name = f"{FAMILIES[family]['module']}__mut_{tag}"
    filename = source_path + f"#{tag}"
    # TRITON READS THE SOURCE BACK. `JITFunction.__init__` calls
    # `inspect.getsourcelines(fn)` (and `getsource` for the body it compiles), and
    # a function created by `exec` has no file for `inspect` to find — the run
    # that discovered this died with `OSError: source code not available` at the
    # FIRST mutation, after S1, S2 and both nulls had already passed. Registering
    # the mutated text in `linecache` under this module's own fake filename is
    # what makes the mutant compile, and it also keeps Triton's JIT cache key
    # distinct: the key is over the SOURCE, and the mutated source differs from
    # the shipped one by construction, so a warm cache cannot serve the shipped
    # binary for a mutant.
    linecache.cache[filename] = (len(mutated_text), None,
                                 mutated_text.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "meep_gpu.triton_kernels"
    sys.modules[name] = module
    code = compile(mutated_text, filename, "exec")
    exec(code, module.__dict__)  # noqa: S102 - a deliberate mutation copy
    return module


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def _plan(family: str, fields: Any, pml: Any, block: Optional[int] = None,
          kernel: Any = None) -> Any:
    module = importlib.import_module(FAMILIES[family]["module"])
    planner = getattr(module, FAMILIES[family]["planner"])
    return planner(fields, pml, sources=(), block=block, kernel=kernel)


def one_case(family: str, fixture: Dict[str, Any], value_class: str,
             block: int, steps: int, seed: int, xp: Any,
             kernel: Any = None,
             reference: str = "array_path") -> Dict[str, Any]:
    """One (fixture, class, block) case: the weld against one reference.

    Two complete driver steps, every stored volume, uint32 words. Step 2 is not
    decoration: the weld ROTATES buffers, and a rotation that is subtly wrong is
    invisible on step 1 and wrong from step 2 on.
    """
    reference_fields, reference_pml = build_fixture(fixture, value_class, seed, xp)
    weld_fields, weld_pml = build_fixture(fixture, value_class, seed, xp)
    before = snapshot(reference_fields)
    start = compare(before, snapshot(weld_fields))
    if start["_total"]:
        raise AssertionError("the two fixtures did not start identical")

    counter: Optional[_CountingKernel] = None
    if kernel is not None:
        counter = _CountingKernel(kernel)
    plan = _plan(family, weld_fields, weld_pml, block=block,
                 kernel=counter if counter is not None else None)
    if plan is None:
        return {"case": fixture["label"], "value_class": value_class,
                "block": block, "planned": False,
                "reason": "the shipped predicate refused this fixture",
                "passed": False}

    if counter is None:
        module = importlib.import_module(FAMILIES[family]["module"])
        counter = _CountingKernel(getattr(module, FAMILIES[family]["kernel"]))
        plan._kernel = counter

    reference_launches = 0
    if reference == "certified_kernels":
        run_reference, reference_launches = certified_composition(
            reference_fields, reference_pml, family)
    else:
        def run_reference() -> None:
            array_path_step(reference_fields, reference_pml)

    per_step: List[Dict[str, Any]] = []
    for _ in range(steps):
        run_reference()
        plan.run()
        per_step.append(compare(snapshot(reference_fields),
                                snapshot(weld_fields)))
    after = snapshot(reference_fields)
    return {
        "case": fixture["label"], "value_class": value_class, "block": block,
        "planned": True, "reference": reference,
        "steps": steps,
        "differing_words_per_step": [row["_total"] for row in per_step],
        "compared_words": per_step[-1]["_compared"],
        "detail": {key: value for key, value in per_step[-1].items()
                   if not key.startswith("_")},
        "reference_moved_words": moved(before, after),
        "weld_launches_kernel_counter": counter.count,
        "weld_launches_plan_counter": plan.launches,
        "reference_launches": reference_launches,
        "passed": (all(row["_total"] == 0 for row in per_step)
                   and moved(before, after) > 0
                   and counter.count == plan.launches == steps),
    }


def s1_leg(family: str, steps: int, seed: int, xp: Any) -> List[Dict[str, Any]]:
    """THE SUBJECT. Every fixture x value class x block, both references.

    This is the leg the 2026-08-20 artifact reported 42/60 DIVERGENT on. It must
    come back 60/60 identical, and it must do so against BOTH references: the
    array path (no kernel anywhere) and the certified kernels the weld replaces.
    """
    spec = FAMILIES[family]
    rows: List[Dict[str, Any]] = []
    for fixture in spec["fixtures"]:
        for value_class in VALUE_CLASSES:
            for block in S1_BLOCKS:
                for reference in ("array_path", "certified_kernels"):
                    # A LEG THAT RAISES IS A RESULT, NOT A CRASH. The certified
                    # composition can refuse a fixture (its own predicates are
                    # narrower than this weld's in places), and a gate that died
                    # there would lose every leg after it — which is the reason
                    # the runner is `set -uo pipefail` and not `-e`.
                    try:
                        row = one_case(family, fixture, value_class, block,
                                       steps, seed, xp, reference=reference)
                    except Exception as error:  # noqa: BLE001
                        row = {"case": fixture["label"],
                               "value_class": value_class, "block": block,
                               "reference": reference, "planned": False,
                               "raised": str(error)[:600], "passed": False}
                    row["leg"] = "S1"
                    row["family"] = family
                    rows.append(row)
    return rows


def s2_leg(family: str, steps: int, seed: int, xp: Any) -> List[Dict[str, Any]]:
    """THE BLOCK SCHEDULE. One fixture, every block size, repeated.

    The direct rebuttal of the 2026-08-20 table, which moved from 4384 differing
    words at BLOCK=64 to 0 at BLOCK=1024 on a 4096-cell grid. A race is not
    repeatable, so one run per point cannot distinguish "identical" from
    "identical this time" — hence the repeats.
    """
    spec = FAMILIES[family]
    fixture = spec["s2_fixture"]
    cells = _cell_count(fixture)
    rows: List[Dict[str, Any]] = []
    for block in BLOCKS:
        counts: List[int] = []
        for repeat in range(S2_REPEATS):
            row = one_case(family, fixture, "normal", block, steps,
                           seed + repeat, xp)
            counts.append(sum(row["differing_words_per_step"]))
        certified = one_case(family, fixture, "normal", block, steps, seed, xp,
                             reference="certified_kernels")
        counts.append(sum(certified["differing_words_per_step"]))
        programs = (cells + block - 1) // block
        # A POINT WITH ONE PROGRAM MEASURES NOTHING. The whole claim is about
        # cross-program reads; at one program there are none, so "identical" is
        # free. Recorded per point rather than assumed, and a clause requires
        # every point to have carried at least two.
        rows.append({"leg": "S2", "family": family, "fixture": fixture["label"],
                     "block": block, "repeats": S2_REPEATS,
                     "cells": cells, "programs": programs,
                     "references": ["array_path"] * S2_REPEATS
                                   + ["certified_kernels"],
                     "differing_words": counts,
                     "passed": all(count == 0 for count in counts)
                               and programs >= 2})
    return rows


def _cell_count(fixture: Dict[str, Any]) -> int:
    """The stored cell count of a fixture, from the engine's own Grid."""
    from meep_gpu.grid import Grid, Mirror

    grid = Grid(resolution=float(fixture.get("resolution", 10.0)),
                cell_size=fixture["cell"],
                dimensions=fixture.get("dimensions", 2),
                boundaries=fixture["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in fixture.get("symmetry", ())))
    shape = tuple(grid.shape)
    return shape[0] * shape[1] * shape[2]


def null_leg(family: str, steps: int, seed: int, xp: Any) -> List[Dict[str, Any]]:
    """Controls that MUST diverge, on the device, through the shipped plan.

    ``rotation_skipped`` is the design's own load-bearing step: without it the
    engine keeps reading the buffer the launch did not write. ``scratch_aliased``
    is the plan's refusal — binding a twin to its own live volume would be the
    2026-08-20 in-place weld wearing this plan's name, and the plan must REFUSE
    rather than launch.
    """
    spec = FAMILIES[family]
    fixture = spec["fixtures"][0]
    rows: List[Dict[str, Any]] = []

    # 1. rotation skipped
    reference_fields, reference_pml = build_fixture(fixture, "normal", seed, xp)
    weld_fields, weld_pml = build_fixture(fixture, "normal", seed, xp)
    plan = _plan(family, weld_fields, weld_pml, block=256)
    array_path_step(reference_fields, reference_pml)
    writes, reads = plan._resolve()
    plan._launch(writes, reads, None)           # launch WITHOUT the rotation
    difference = compare(snapshot(reference_fields), snapshot(weld_fields))
    rows.append({"leg": "null", "family": family, "null": "rotation_skipped",
                 "differing_words": difference["_total"],
                 "reason": "the engine keeps its pre-launch references, so every "
                           "later read sees a displacement the launch did not "
                           "write",
                 "expected": "DIVERGE",
                 "passed": difference["_total"] > 0})

    # 2. the plan refuses an aliased twin
    aliased_fields, aliased_pml = build_fixture(fixture, "normal", seed, xp)
    refused = None
    try:
        plan = _plan(family, aliased_fields, aliased_pml, block=256)
        plan.rotated["Dx"] = getattr(aliased_fields, "Dx")
        plan.run()
    except RuntimeError as error:
        refused = str(error)
    rows.append({"leg": "null", "family": family, "null": "scratch_aliased",
                 "refusal": refused,
                 "reason": "an aliased twin is the in-place weld again; the plan "
                           "must refuse rather than launch",
                 "expected": "REFUSE",
                 "passed": refused is not None and "aliased" in refused})
    return rows


def mutation_leg(family: str, steps: int, seed: int,
                 xp: Any) -> List[Dict[str, Any]]:
    """Every declared mutation, armed through the shipped planner's kernel= door.

    Scored on a leg that CLAIMS IDENTITY — the subject at BLOCK=256 on a walled
    fixture, whose own verdict is 0 differing words — so a mutation that changes
    a byte is caught by the same comparison the release rests on. Nothing is
    scored on a leg that already fails.
    """
    spec = FAMILIES[family]
    by_label = {entry["label"]: entry for entry in spec["fixtures"]}
    rows: List[Dict[str, Any]] = []
    for tag, target, old, new, expected, scored_on in MUTATIONS[family]:
        fixture = (by_label[scored_on] if scored_on else spec["fixtures"][0])
        try:
            module = mutated_module(family, old, new, tag)
        except Exception as error:  # noqa: BLE001 - a build failure is a result
            rows.append({"leg": "mutation", "family": family, "mutation": tag,
                         "expected": expected, "outcome": "NEEDLE FAILED",
                         "detail": str(error), "passed": False})
            continue
        kernel = getattr(module, spec["kernel"])
        try:
            row = one_case(family, fixture, "normal", 256, steps, seed, xp,
                           kernel=kernel)
            differing = sum(row["differing_words_per_step"])
            outcome = "CAUGHT" if differing else "NULL"
            detail = f"{differing} differing words of {row['compared_words']}"
        except Exception as error:  # noqa: BLE001 - a compile failure is a result
            differing = None
            outcome = "REFUSED TO COMPILE"
            detail = str(error)[:400]
        rows.append({"leg": "mutation", "family": family, "mutation": tag,
                     "target": target, "expected": expected, "outcome": outcome,
                     "scored_on": fixture["label"],
                     "differing_words": differing, "detail": detail,
                     "reason": MUTATION_REASONS.get(tag, ""),
                     "passed": outcome == expected})
    return rows


def warm_leg(family: str, steps: int, seed: int, xp: Any) -> List[Dict[str, Any]]:
    """THE PLAN-TIME WARM, ON THE DEVICE, through ``fastpath.warm_plan``.

    What dispatch does on a slot this product fills before step 1, and the one
    thing this gate's 2026-09-13 batch never ran: it drove ``run`` alone, because
    nothing warmed these plans while no composer routed them. On the family's
    first fixture ``warm_plan`` must answer ``None`` (warmed); the counted kernel
    must be subscripted ONCE, at grid ``(0,)``, and launched once; the plan's
    ``launches`` must stay 0; ``_grid`` must come back; every rotating reference --
    the engine's and the twin table's -- must be the object it was; and no stored
    word may move. Then ``steps`` complete steps against the array path must be
    identical, so the warm is shown to leave a plan that steps correctly rather
    than one that was merely left alone.

    WHAT IT MEASURES THAT NOTHING ELSE HERE DOES: whether Triton compiles this
    kernel and enqueues nothing at grid ``(0,)``, which ``fastpath.warm_plan``'s
    docstring records as unmeasured in general.
    """
    from meep_gpu import fastpath

    spec = FAMILIES[family]
    fixture = spec["fixtures"][0]
    module = importlib.import_module(spec["module"])
    row: Dict[str, Any] = {"leg": "warm", "family": family,
                           "case": fixture["label"], "block": 256}
    reference_fields, reference_pml = build_fixture(fixture, "normal", seed, xp)
    weld_fields, weld_pml = build_fixture(fixture, "normal", seed, xp)
    counter = _CountingKernel(getattr(module, spec["kernel"]))
    plan = _plan(family, weld_fields, weld_pml, block=256, kernel=counter)
    if plan is None:
        row.update(planned=False, passed=False,
                   reason="the shipped predicate refused this fixture")
        return [row]
    live = {name: getattr(weld_fields, name) for name in module.ROTATED}
    twins = dict(plan.rotated)
    grid = tuple(plan._grid)
    before = snapshot(weld_fields)
    started_identical = compare(snapshot(reference_fields), before)["_total"] == 0

    raised: Optional[str] = None
    answer: Any = "<not called>"
    try:
        answer = fastpath.warm_plan(plan)
    except Exception as error:  # noqa: BLE001 - a warm that raises is a result
        raised = str(error)[:600]
    identity = all(getattr(weld_fields, name) is live[name]
                   and plan.rotated[name] is twins[name]
                   for name in module.ROTATED)
    warm_subscripts = list(counter.grids)
    warm_launches = counter.count
    row.update({
        "planned": True,
        "started_identical": started_identical,
        "plan_grid": list(grid),
        "warm_plan_answered": answer,
        "raised": raised,
        "warm_kernel_subscripts": [list(g) for g in warm_subscripts],
        "warm_kernel_launches": warm_launches,
        "warm_plan_launches": plan.launches,
        "grid_restored": tuple(plan._grid) == grid,
        "rotation_identity_preserved": identity,
        "words_the_warm_moved": compare(before, snapshot(weld_fields))["_total"],
    })
    warm_clean = (raised is None and answer is None
                  and warm_subscripts == [(0,)] and warm_launches == 1
                  and plan.launches == 0 and row["grid_restored"] and identity
                  and row["words_the_warm_moved"] == 0)

    per_step: List[int] = []
    if raised is None:
        for _ in range(steps):
            array_path_step(reference_fields, reference_pml)
            plan.run()
            per_step.append(compare(snapshot(reference_fields),
                                    snapshot(weld_fields))["_total"])
    row["differing_words_per_step_after_the_warm"] = per_step
    row["reference_moved_words"] = moved(before, snapshot(reference_fields))
    row["step_kernel_subscripts"] = [list(g) for g in counter.grids[1:]]
    row["passed"] = bool(
        started_identical and warm_clean
        and len(per_step) == steps and all(n == 0 for n in per_step)
        and row["reference_moved_words"] > 0
        and plan.launches == steps and counter.count == 1 + steps
        and counter.grids[1:] == [grid] * steps)
    return [row]


# ---------------------------------------------------------------------------
# The no-device legs
# ---------------------------------------------------------------------------

class _RecordingStub:
    """The ``kernel[grid](...)`` boundary on a host with no Triton: recorded, inert."""

    __slots__ = ("grids", "count")

    def __init__(self) -> None:
        self.grids: List[Tuple[int, ...]] = []
        self.count = 0

    def __getitem__(self, grid: Any) -> Any:
        self.grids.append(tuple(int(n) for n in grid))

        def launch(*args: Any, **kwargs: Any) -> None:
            self.count += 1

        return launch


def _host_plan(family: str, kernel: Any) -> Tuple[Any, Any, List[str]]:
    """The SHIPPED builder on a NumPy host, past the one clause the host fails.

    Returns ``(fields, plan, structural_reasons)``. The family's own predicate is
    asked first, on the family's first fixture, and every reason it gives other
    than the backend clause is returned; the plan is built only when there is none,
    with the predicate replaced for that one call and restored in a ``finally``. So
    the replacement changes which BACKEND the builder believes it is on and nothing
    about which configuration it builds for.
    """
    from meep_gpu.triton_kernels.coverage import Coverage

    spec = FAMILIES[family]
    module = importlib.import_module(spec["module"])
    fields, pml = build_fixture(spec["fixtures"][0], "normal", 20260902, np)
    predicate = getattr(module, spec["predicate"])
    structural = [reason for reason in predicate(fields, pml, ()).reasons
                  if "not cupy" not in reason]
    if structural:
        return fields, None, structural
    setattr(module, spec["predicate"], lambda *a, **k: Coverage(True, ()))
    try:
        plan = getattr(module, spec["planner"])(fields, pml, sources=(),
                                                block=256, kernel=kernel)
    finally:
        setattr(module, spec["predicate"], predicate)
    return fields, plan, []


@contextlib.contextmanager
def _kernels_on_this_host() -> Any:
    """``triton_kernels.kernels`` as the launch reaches it, on a host that may lack Triton.

    Both plans' ``_launch`` import ``ENABLE_FP_FUSION`` from ``kernels`` before they
    subscript the kernel, and that module imports Triton at its top. Where it
    imports, it is used as it is and this yields ``"imported"``. Where it does not,
    a stand-in carrying that ONE name at its shipped value is installed for the
    duration of the leg and removed afterwards, yielding ``"stand-in"`` -- the
    technique ``test_triton_complex_fused_electric_pair``'s launch-signature test
    uses -- so a launch body that started reading a second name from ``kernels``
    fails here rather than being accommodated.
    """
    name = "meep_gpu.triton_kernels.kernels"
    try:
        importlib.import_module(name)
    except ImportError:
        stand_in = types.ModuleType(name)
        stand_in.ENABLE_FP_FUSION = False  # type: ignore[attr-defined]
        sys.modules[name] = stand_in
        try:
            yield "stand-in"
        finally:
            if sys.modules.get(name) is stand_in:
                del sys.modules[name]
        return
    yield "imported"


def no_device_warm_leg(family: str) -> Dict[str, Any]:
    """The host warm leg, with ``kernels`` resolved for this host; see the body."""
    with _kernels_on_this_host() as kernels_module:
        row = _no_device_warm_leg_body(family)
    row["kernels_module"] = kernels_module
    return row


def _no_device_warm_leg_body(family: str) -> Dict[str, Any]:
    """THE WARM'S STRUCTURE on the host: the shipped plan, a recording stub kernel.

    Through ``fastpath.warm_plan``, exactly as dispatch reaches it: the stub must be
    subscripted once at ``(0,)`` and launched once, ``launches`` must stay 0,
    ``_grid`` must come back, and every rotating reference must be the object it
    was. The ARMED NULL rides beside it: the same plan through
    ``fastpath._warm_with_empty_grid`` -- the path ``warm_plan`` took before
    ``ScratchWeldPairPlan.warm`` existed -- must COUNT a launch and ROTATE, or this
    leg could not see the defect it exists to exclude.
    """
    from meep_gpu import fastpath

    module = importlib.import_module(FAMILIES[family]["module"])
    findings: List[str] = []
    stub = _RecordingStub()
    fields, plan, structural = _host_plan(family, stub)
    row: Dict[str, Any] = {"leg": "no_device:warm", "family": family,
                           "structural_refusals": structural}
    if plan is None:
        findings.append(f"the shipped builder refused the host fixture: {structural}")
        row.update(findings=findings, passed=False)
        return row
    live = {name: getattr(fields, name) for name in module.ROTATED}
    twins = dict(plan.rotated)
    grid = tuple(plan._grid)
    answer = fastpath.warm_plan(plan)
    identity = all(getattr(fields, name) is live[name]
                   and plan.rotated[name] is twins[name]
                   for name in module.ROTATED)
    if answer is not None:
        findings.append(f"warm_plan answered {answer!r} rather than warming")
    if stub.grids != [(0,)] or stub.count != 1:
        findings.append(f"the warm subscripted {stub.grids} and launched "
                        f"{stub.count} times; one launch at (0,) was required")
    if plan.launches != 0:
        findings.append(f"the warm counted {plan.launches} launches")
    if tuple(plan._grid) != grid:
        findings.append(f"the grid came back as {plan._grid}, not {grid}")
    if not identity:
        findings.append("the warm moved a rotating reference")

    null_stub = _RecordingStub()
    null_fields, null_plan, _ = _host_plan(family, null_stub)
    null_live = {name: getattr(null_fields, name) for name in module.ROTATED}
    fastpath._warm_with_empty_grid(null_plan, null_plan.run)
    null_rotated = all(getattr(null_fields, name) is not null_live[name]
                       for name in module.ROTATED)
    if not (null_plan.launches == 1 and null_rotated):
        findings.append(
            f"the empty-grid fallback did not show the defect (launches "
            f"{null_plan.launches}, rotated {null_rotated}); the leg is disarmed")
    row.update({
        "plan_grid": list(grid), "warm_plan_answered": answer,
        "warm_kernel_subscripts": [list(g) for g in stub.grids],
        "warm_kernel_launches": stub.count, "warm_plan_launches": plan.launches,
        "rotation_identity_preserved": identity,
        "null_empty_grid_run": {"plan_launches": null_plan.launches,
                                "rotated": null_rotated},
        "findings": findings, "passed": not findings,
    })
    return row

def no_device_legs(families: Sequence[str]) -> List[Dict[str, Any]]:
    """Everything establishable without a GPU, run here so the merge bar has it.

    The host probe's whole verdict, plus the two structural facts this gate
    depends on: that every declared mutation needle still matches its family's
    source EXACTLY ONCE (a needle that stopped matching is a disarmed leg, and it
    would report as a pass), and that the two products declare the seam they
    claim to replace.
    """
    import probe_triton_offdiag_scratch_weld as probe

    rows: List[Dict[str, Any]] = []
    for family in families:
        for name, leg in (("transcription", probe.transcription_leg),
                          ("tap_binding", probe.tap_binding_leg),
                          ("arming", probe.arming_leg)):
            row = leg(family)
            row["leg"] = f"no_device:{name}"
            rows.append(row)

        source_path = os.path.join(
            API_ROOT, FAMILIES[family]["module"].replace(".", os.sep) + ".py")
        text = open(source_path, encoding="utf-8").read()
        findings: List[str] = []
        for tag, _target, old, _new, _expected, _scored in MUTATIONS[family]:
            hits = text.count(old)
            if hits != 1:
                findings.append(
                    f"{tag}: needle matches {hits} times, not once — the "
                    f"mutation is disarmed and would report a pass")
        rows.append({"leg": "no_device:mutation_needles", "family": family,
                     "mutations": len(MUTATIONS[family]),
                     "findings": findings, "passed": not findings})

        module = importlib.import_module(FAMILIES[family]["module"])
        declared = tuple(module.REPLACES)
        carried = FAMILIES[family]["carried_passes"]
        expected = ("step_D",) + carried + ("update_E",)
        rows.append({"leg": "no_device:replaces", "family": family,
                     "declared": declared, "expected": expected,
                     "carries_deposit_repair": module.CARRIES_DEPOSIT_REPAIR,
                     "passed": declared == expected
                               and module.CARRIES_DEPOSIT_REPAIR is False})
        rows.append(no_device_warm_leg(family))
    return rows


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def verdict(rows: Sequence[Dict[str, Any]], families: Sequence[str],
            device: bool, policy_attained: bool = True) -> Dict[str, Any]:
    clauses: Dict[str, Any] = {}
    s1 = [row for row in rows if row.get("leg") == "S1"]
    s2 = [row for row in rows if row.get("leg") == "S2"]
    nulls = [row for row in rows if row.get("leg") == "null"]
    mutations = [row for row in rows if row.get("leg") == "mutation"]
    warms = [row for row in rows if row.get("leg") == "warm"]
    structural = [row for row in rows
                  if str(row.get("leg", "")).startswith("no_device:")]

    clauses["every_structural_leg_passes"] = bool(structural) and all(
        row["passed"] for row in structural)
    if not device:
        return {"clauses": clauses, "released": all(clauses.values()),
                "device_legs_ran": False,
                "counts": {"structural": len(structural)}}

    clauses["S1_is_identical_on_every_case"] = bool(s1) and all(
        row["passed"] for row in s1)
    clauses["S1_ran_against_both_references"] = (
        {row["reference"] for row in s1 if row.get("planned")}
        == {"array_path", "certified_kernels"})
    clauses["S1_swept_its_declared_blocks"] = (
        {row["block"] for row in s1} == set(S1_BLOCKS))
    clauses["S2_carried_both_oracles_at_every_block"] = bool(s2) and all(
        set(row.get("references", ())) == {"array_path", "certified_kernels"}
        for row in s2)
    clauses["S1_ran_on_all_three_value_classes"] = (
        {row["value_class"] for row in s1} == set(VALUE_CLASSES))
    clauses["S1_every_case_moved_the_reference"] = all(
        row.get("reference_moved_words", 0) > 0 for row in s1)
    clauses["S2_is_identical_at_every_block_size"] = bool(s2) and all(
        row["passed"] for row in s2)
    clauses["S2_swept_the_block_sizes_the_2026_08_20_table_did"] = (
        {row["block"] for row in s2} == set(BLOCKS))
    clauses["every_S2_point_carried_at_least_two_programs"] = bool(s2) and all(
        row.get("programs", 0) >= 2 for row in s2)
    clauses["S2_ran_on_a_grid_at_least_as_large_as_the_2026_08_20_one"] = (
        bool(s2) and min(row.get("cells", 0) for row in s2) >= 4096)
    clauses["the_weld_launches_exactly_once_per_step"] = all(
        row.get("weld_launches_kernel_counter")
        == row.get("weld_launches_plan_counter")
        == row.get("steps") for row in s1 if row.get("planned"))
    clauses["every_null_behaves_as_declared"] = bool(nulls) and all(
        row["passed"] for row in nulls)
    clauses["every_mutation_lands_as_declared"] = bool(mutations) and all(
        row["passed"] for row in mutations)
    clauses["the_pointwise_plant_is_CAUGHT_here"] = any(
        row["mutation"] == "m_tap_collapsed_to_own_cell"
        and row["outcome"] == "CAUGHT" for row in mutations)
    clauses["every_family_ran"] = (
        {row["family"] for row in s1} == set(families))
    clauses["the_policy_installed_is_the_one_requested"] = bool(policy_attained)
    clauses["the_plan_time_warm_rotates_nothing_and_leaves_a_plan_that_steps"] = (
        bool(warms) and all(row["passed"] for row in warms))
    clauses["every_family_warmed"] = (
        {row["family"] for row in warms} == set(families))
    return {"clauses": clauses, "released": all(clauses.values()),
            "device_legs_ran": True,
            "counts": {"S1": len(s1), "S2": len(s2), "nulls": len(nulls),
                       "warm": len(warms),
                       "mutations": len(mutations),
                       "structural": len(structural)},
            "S1_identical": sum(1 for row in s1 if row["passed"]),
            "S1_total": len(s1)}


def environment(xp: Any) -> Dict[str, Any]:
    """What ran this, in the shape the weld tools read.

    ``hostname``, ``device``, ``compute_capability``, ``cupy``, ``triton`` and
    ``cuda_visible_devices`` are the keys ``rebind_triton_welds._host_line``
    composes a weld's ``host`` from, and ``seed_triton_welds`` refuses an artifact
    with no ``environment.device`` by name. This gate's 2026-09-13 batch artifacts
    carry no such block, which is one of three reasons neither can seed the two
    products' ledger entries (the others: directories named ``plain``/``folded``,
    and digests the routing batch moves). The block is handed to
    ``triton_device_identity.record``, which fills what this function could not read
    and normalises the capability spelling. On ``--no-device`` it records the host
    and no device, so a host-only artifact can never be seeded.
    """
    record: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    if xp is not None and xp is not np:
        record["cupy"] = getattr(xp, "__version__", "?")
        try:
            import triton  # noqa: PLC0415

            record["triton"] = getattr(triton, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - recorded, and the seed then refuses
            record["triton"] = None
            record["triton_error"] = repr(exc)
        try:
            index = int(xp.cuda.runtime.getDevice())
            properties = xp.cuda.runtime.getDeviceProperties(index)
            name = properties["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            record["device_index"] = index
            record["device"] = name
            record["device_name"] = name
            record["compute_capability"] = (
                f"{properties['major']}.{properties['minor']}")
        except Exception as exc:  # noqa: BLE001 - recorded, and the seed then refuses
            record["device_error"] = repr(exc)
    import triton_device_identity  # noqa: PLC0415

    return triton_device_identity.record(record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=None,
        help="ONE artifact over every family this process runs; host legs only")
    parser.add_argument(
        "--out-root", default=None,
        help="a campaign directory: each family writes <root>/<family>/gate.json, "
             "the layout seed_triton_welds.py derives ledger keys from")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--family", default=None,
                        choices=(None, *sorted(FAMILIES)))
    parser.add_argument("--subnormal-policy", default=None,
                        choices=(None, "keep", "flush"))
    args = parser.parse_args(argv)

    families = ([args.family] if args.family else sorted(FAMILIES))
    if args.out and args.out_root:
        parser.error("--out and --out-root name two layouts; pass one")
    # A DEVICE ARTIFACT IS WRITTEN ONLY WHERE A SEED CAN READ IT RIGHT. The ledger
    # key is derived from the directory name, so a device run that wrote one
    # gate.json over both families -- or per-family directories under any other
    # names -- would certify bytes under a key no ARM_CERTIFICATION row names.
    if not args.no_device and not args.out_root:
        parser.error(
            "a device run writes one artifact PER FAMILY under --out-root, because "
            "seed_triton_welds.py derives each ledger key from the directory name; "
            "--out is for the host legs (--no-device)")

    device: Optional[Dict[str, Any]] = None
    if not args.no_device:
        from meep_gpu.triton_kernels import launch as launch_module

        launch_module.require_triton()
        import cupy

        from meep_gpu import subnormal_policy

        # BEFORE THE FIRST DEVICE COMPILE, and ONCE per process however many
        # families follow. strict=True: a process that asked to keep and quietly
        # did not is a process whose bytes mean nothing, and the 2026-08-20
        # artifact this gate supersedes reports its two cuts separately for exactly
        # that reason — `keep` and `flush` are different platforms as far as these
        # bytes are concerned and neither transfers.
        subnormal_policy.install_subnormal_policy(
            args.subnormal_policy, cupy=cupy, strict=True)
        import triton

        device = {
            "xp": cupy,
            "subnormal_policy": subnormal_policy.policy_stamp(),
            "cupy_version": cupy.__version__,
            "triton_version": triton.__version__,
            "device": str(cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())["name"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
            "nvidia_smi_compute_apps_at_start": _compute_apps(),
        }
        log(f"subnormal policy installed: "
            f"{device['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}, "
            f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    if args.out_root:
        worst = 0
        for family in families:
            out = os.path.join(args.out_root, family, "gate.json")
            log(f"===== {family} -> {out}")
            worst |= run_gate([family], args, out, device)
        return worst
    return run_gate(families, args, args.out, device)


def run_gate(families: Sequence[str], args: Any, out: Optional[str],
             device: Optional[Dict[str, Any]]) -> int:
    """Every leg for ``families`` into ONE artifact at ``out``; 0 when released."""
    started = time.time()
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "gate": GATE,
        "supersedes": SUPERSEDES,
        "families": list(families),
        "seed": args.seed,
        "steps": args.steps,
        "numpy_version": np.__version__,
        "subnormal_policy_requested": args.subnormal_policy,
        "blocks": list(BLOCKS),
        "s2_repeats": S2_REPEATS,
        "source_sha256": source_hashes(),
        "rows": rows,
    }

    xp: Any = None
    if device is not None:
        xp = device["xp"]
        payload.update({key: value for key, value in device.items() if key != "xp"})
    payload["environment"] = environment(xp)
    log(f"environment: {json.dumps(payload['environment'], sort_keys=True, default=str)}")

    for row in no_device_legs(families):
        rows.append(row)
        log(f"{row['leg']:34s} {row.get('family', ''):38s} "
            f"{'PASS' if row['passed'] else 'FAIL'} "
            f"({time.time() - started:.1f} s)")
        for finding in row.get("findings", []):
            log(f"    ! {finding}")
    save(payload, out)

    if device is not None:
        for family in families:
            for row in warm_leg(family, args.steps, args.seed, xp):
                rows.append(row)
                log(f"WARM {family:38s} {row.get('case', '-'):22s} "
                    f"subscripts={row.get('warm_kernel_subscripts')} "
                    f"plan_launches={row.get('warm_plan_launches')} "
                    f"identity={row.get('rotation_identity_preserved')} "
                    f"after={row.get('differing_words_per_step_after_the_warm')} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
            save(payload, out)

            for row in s1_leg(family, args.steps, args.seed, xp):
                rows.append(row)
                log(f"S1 {family:38s} {row['case']:22s} "
                    f"{row['value_class']:20s} block={row['block']:5d} "
                    f"ref={row.get('reference', '-'):18s} "
                    f"differing={row.get('differing_words_per_step')} "
                    f"moved={row.get('reference_moved_words')} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
            save(payload, out)

            for row in s2_leg(family, args.steps, args.seed, xp):
                rows.append(row)
                log(f"S2 {family:38s} block={row['block']:5d} "
                    f"differing={row['differing_words']} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
            save(payload, out)

            for row in null_leg(family, args.steps, args.seed, xp):
                rows.append(row)
                log(f"NULL {family:38s} {row['null']:22s} "
                    f"expected={row['expected']} "
                    f"moved={row.get('differing_words')} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
            save(payload, out)

            for row in mutation_leg(family, args.steps, args.seed, xp):
                rows.append(row)
                log(f"MUT {family:38s} {row['mutation']:34s} "
                    f"expected={row['expected']:8s} outcome={row['outcome']:20s} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
                if not row["passed"]:
                    log(f"    ! {row.get('detail')}")
            save(payload, out)

    payload["imported_source_sha256"] = imported_source_sha256()
    payload["verdict"] = verdict(
        rows, families, device is not None,
        policy_attained=bool((payload.get("subnormal_policy") or {}).get(
            "attained", True)))
    payload["elapsed_s"] = round(time.time() - started, 2)
    # THE RELEASE SHAPE THE BOARD'S BINDING CHECK READS, and it is deliberately
    # four separate keys rather than one. `build_triton_fusion_matrix`'s
    # GATE_BOUND arm refuses a gate unless device_status is RUN, release.released
    # is True, passed is True and no planted defect is recorded — so a run that
    # measured no bytes on any device cannot be read as a release, which has
    # happened on this board before and cost it a re-cut.
    payload["device_status"] = "SKIPPED" if device is None else "RUN"
    payload["passed"] = bool(payload["verdict"]["released"])
    payload["planted_defect"] = None
    payload["release"] = {
        "released": bool(payload["verdict"]["released"]
                         and device is not None),
        "families": list(families),
        "supersedes": SUPERSEDES["artifact"],
        "S1_identical": payload["verdict"].get("S1_identical"),
        "S1_total": payload["verdict"].get("S1_total"),
        "reason_if_not": (
            None if payload["verdict"]["released"] and device is not None
            else ("no device legs ran" if device is None
                  else [name for name, value
                        in payload["verdict"]["clauses"].items() if not value])),
    }
    log(f"VERDICT released={payload['verdict']['released']}")
    for name, value in payload["verdict"]["clauses"].items():
        log(f"    {name:60s} {value}")
    save(payload, out)
    if out:
        log(f"wrote {out}")
    return 0 if payload["verdict"]["released"] else 1


def _compute_apps() -> List[str]:
    """What else is on this device. A bit-identity verdict cannot be changed by
    contention — this gate times nothing — but contention CAN change how often a
    race window is hit, so a run that found none is worth distinguishing from one
    that shared the card."""
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
        return [line for line in out.stdout.splitlines() if line.strip()]
    except Exception as error:  # noqa: BLE001
        return [f"unreadable: {error}"]


if __name__ == "__main__":
    raise SystemExit(main())
