"""Byte-identity gate for the CUDA IN-SEAM PASSES, one pass at a time.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/in_seam_passes.py`` ships six kernels
-- a B and a D variant of each of the three passes the driver runs between a curl
and the constitutive update that closes it::

    step_B -> magnetic sources -> fill_symmetry_bc_B     (driver.py:3284)
                               -> zero_metal_B           (driver.py:3285)
                               -> fill_folded_far_ghosts_B (driver.py:3286) -> update_H

Each pass is gated SEPARATELY (``--pass zero_metal|fill_symmetry|fill_folded_far``)
against the ``stepping`` function it transcribes, from one frozen state, as raw
uint32 words. A pass with a verdict earns a record block and moves from
``UNCERTIFIED_KERNELS`` to ``CERTIFIED_KERNELS``; a pass without one does not.

THIS GATE CERTIFIES NO FUSION AND MEASURES NO THROUGHPUT. It measures three
passes against three array-path functions. What that buys is the PARTS a fused
CUDA pair would have to absorb -- nothing more, and the summary's
``does_not_claim`` says so.

=============================================================================
WHY EACH PASS IS ITS OWN GATE AND THE THREE ARE NEVER RUN AS ONE
=============================================================================

They share planes. On a grid with a mirrored X and a metallic Y, ``Dz`` is
near-filled on X (the plane ``Dz[0, :, :]``) and wall-wiped on Y (the plane
``Dz[:, 0, :]``), and the two meet on the edge ``Dz[0, 0, :]``. The driver runs
the fill first and the wipe second, so that edge ends at zero -- an ORDER, not a
partition. A gate that ran the three together would be measuring the composition
and could not say which pass owned a divergence; one that fused them into a
launch would be racing on that edge. The fixture carries exactly that
configuration (``fold_X_wall_Y``) so the sharing is exercised even though each
pass is measured alone.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of the family's words the ARRAY PATH changed
  from the frozen input. Every one of these passes is a plane write, so a state
  that already satisfies the boundary condition is a fixed point and a case that
  moved nothing certifies nothing. Zero-init is exactly such a state, which is
  why the fixture never uses it.
* ``planes_touched`` -- how many launches the pass actually issued. A pass with
  no live axis on this grid issues none, and a leg that launched nothing and
  compared equal is the purest vacuous pass there is. Cases below the floor are
  SKIPPED with the reason recorded, never scored.
* ``operand_census`` -- how many subnormals and signed zeros the operands really
  hold. The ``subnormal_band`` class exists to make the policy legs mean
  something, and ``uniform(-1, 1)`` provably draws neither.
* The mutation battery, including mutations that MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.

=============================================================================
THE MUTATIONS THIS GATE OWNS
=============================================================================

None of the shared probe's mutations edit these kernels -- they were written for
the curl and constitutive bodies -- so the battery here is this gate's own,
spelled as source rewrites with a site count so a rewrite that stops matching is
reported as NOT ARMED rather than passing silently.

Two are worth naming:

* ``reflect_row_n_minus_two`` (host) bakes the fixed ``stored - 2`` image row.
  MEASURED on this laptop before the gate ran: the real row is ``stored - 2`` at
  an EVEN full count and ``stored - 3`` at an ODD one, over eight counts 12..19.
  So this mutation is INERT at even counts by construction and is scored only on
  odd-count specs -- a battery that mixed them would report PARTIAL for a reason
  about the fixture.
* ``wall_folded_metallic_axis`` (host) is ``stepping._zero_metal``'s
  ``not is_mirrored`` term armed as a defect: it adds a folded METALLIC axis to
  the walled set. The engine's own docstring measures that at 1.28e+00 complex
  relative L2 against CPU MEEP with the fold's ghost zeroed, against 2.0e-07 with
  the skip in place, so it must be caught -- and it can only be scored on a grid
  that HAS a folded metallic axis.

``copy_instead_of_multiply`` is POLICY-DEPENDENT and is reported rather than
scored. The array path writes ``phase * field[...]``, a float32 multiply; under
``"flush"`` CuPy compiles with ``-ftz=true`` and that multiply flushes a
subnormal operand, while a copy would preserve it. Its verdict is therefore
expected to be UNCAUGHT under ``keep`` and CAUGHT under ``flush`` on the
subnormal-band class, and the record carries both readings.

=============================================================================
RUNNING IT
=============================================================================

Device (ONE verified-empty GPU, pinned by UUID; the cache dir MUST carry the
policy token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_in_seam_passes.py --pass zero_metal \\
        --subnormal-policy keep --out $OUT/zero_metal/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the shipped
device tree. IT COMPILES NOTHING AND CERTIFIES NOTHING -- what it settles is
whether the transcription reproduces ``stepping`` at all, and whether the harness
can fail::

    python -u gate_cuda_in_seam_passes.py --pass zero_metal --backend numpy \\
        --out /tmp/in_seam_zero_metal.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
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
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS, mirror_parity  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

#: The per-case RNG is seeded from a sha256 of the case's own label, NOT from
#: ``hash()``. ``PYTHONHASHSEED`` salts ``hash()`` of a tuple of strings, so a
#: hash-seeded case cannot be replayed from its own record -- the artifact would
#: name a case whose state nobody can rebuild.
SEED = 20260821

#: Consecutive launches in the multi-step leg. 60 is the budget every hand-CUDA
#: record on this track is cut at. "Identical for N launches" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps moving. THESE PASSES CARRY NO STATE -- the output
#: of a plane write depends only on the input -- so held fixed, launches 2..60
#: would re-derive launch 1's answer and the leg would be a slow single launch.
#: The same exact float32 scale is applied on both paths, so it cancels out of the
#: comparison; ``multi_step_oracle_moved`` records whether it kept the leg live.
_ADVANCE = np.float32(0.97)

PASSES: Tuple[str, ...] = ("zero_metal", "fill_symmetry", "fill_folded_far")
FAMILIES: Tuple[str, ...] = ("B", "D")

#: The array-path function each (pass, family) is measured against.
ORACLES: Dict[Tuple[str, str], Callable[[Any], None]] = {
    ("zero_metal", "B"): stepping.zero_metal_B,
    ("zero_metal", "D"): stepping.zero_metal_D,
    ("fill_symmetry", "B"): stepping.fill_symmetry_bc_B,
    ("fill_symmetry", "D"): stepping.fill_symmetry_bc_D,
    ("fill_folded_far", "B"): stepping.fill_folded_far_ghosts_B,
    ("fill_folded_far", "D"): stepping.fill_folded_far_ghosts_D,
}

#: The device-source attribute each (pass, family) kernel lives in.
SOURCE_ATTRIBUTE: Dict[Tuple[str, str], str] = {
    ("zero_metal", "B"): "_zero_metal_B_kernel_code",
    ("zero_metal", "D"): "_zero_metal_D_kernel_code",
    ("fill_symmetry", "B"): "_fill_symmetry_B_kernel_code",
    ("fill_symmetry", "D"): "_fill_symmetry_D_kernel_code",
    ("fill_folded_far", "B"): "_fill_folded_far_B_kernel_code",
    ("fill_folded_far", "D"): "_fill_folded_far_D_kernel_code",
}

FAMILY_COMPONENTS: Dict[str, Tuple[str, str, str]] = {
    "B": ("Bx", "By", "Bz"),
    "D": ("Dx", "Dy", "Dz"),
}


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------
#
# EVERY AXIS IS FOLDED SOMEWHERE, EVERY AXIS IS WALLED SOMEWHERE, and both
# full-count parities appear on a folded PERIODIC axis. A sweep carrying one
# folded axis proves nothing about the other two -- the three have the slowest,
# middle and fastest stride respectively, and a face decomposition that confuses
# them is bit-identical on a cube.
#
# THE EXTENTS ARE DELIBERATELY UNEQUAL. A cube lets an index decomposition swap i
# and k and stay identical, which is what ``transpose_face_strides`` plants.
#
# ``cell`` is in units of 1/resolution and the fixture runs at resolution 1.0, so
# the numbers ARE the full cell counts.

SPECS: Tuple[Dict[str, Any], ...] = (
    # --- walls only, no fold ------------------------------------------------
    {"label": "wall_XYZ", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_X_only", "axes": "", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_Z_only", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (9.0, 10.0, 11.0)},
    {"label": "no_wall_no_fold", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 10.0, 11.0)},

    # --- one fold, both terminations, both plane phases ----------------------
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 10.0, 11.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 10.0, 11.0)},
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 16.0, 11.0)},
    {"label": "fold_Y_periodic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 16.0, 11.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (9.0, 16.0, 11.0)},
    {"label": "fold_Z_periodic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 10.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (9.0, 10.0, 16.0)},

    # --- ODD full count on the folded PERIODIC axis --------------------------
    # MEASURED: the far ghost's image row is ``stored - 2`` at an even full count
    # and ``stored - 3`` at an odd one (eight counts, 12..19, on this laptop).
    # A sweep carrying one parity cannot see a baked ``n - 2``.
    {"label": "fold_X_periodic_odd_count", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (17.0, 10.0, 11.0)},
    {"label": "fold_Y_periodic_odd_count", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 17.0, 11.0)},
    {"label": "fold_Z_periodic_odd_count", "axes": "Z", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (9.0, 10.0, 17.0)},

    # --- the SHARED-PLANE configuration -------------------------------------
    # A mirrored X and a metallic Y: Dz is near-filled on the x = 0 plane and
    # wall-wiped on the y = 0 plane, and the two meet on one edge. Carried so the
    # sharing is exercised even though each pass is measured alone.
    {"label": "fold_X_wall_Y", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 10.0, 11.0)},

    # A FOLDED METALLIC AXIS BESIDE A GENUINE WALL. ``_zero_metal`` clears the y
    # plane and must NOT clear the x one, whose stored cell 0 is the fold's
    # parity-weighted ghost rather than a wall. This is the only spec on which
    # ``wall_folded_metallic_axis`` is scored against an oracle that does REAL
    # work, so the mutation is caught by a wrong extra plane rather than only by
    # the difference between doing something and doing nothing.
    {"label": "fold_X_metallic_wall_Y", "axes": "X", "phase": -1,
     "boundaries": ("metallic", "metallic", "periodic"), "cell": (16.0, 10.0, 11.0)},

    # --- two and three planes at once ---------------------------------------
    {"label": "fold_XY_periodic", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 18.0, 11.0)},
    {"label": "fold_XY_mixed", "axes": "XY", "phase": -1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 18.0, 11.0)},
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_periodic_odd", "axes": "XYZ", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (17.0, 19.0, 21.0)},
    {"label": "fold_XYZ_metallic", "axes": "XYZ", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_mixed", "axes": "XYZ", "phase": -1,
     "boundaries": ("metallic", "periodic", "metallic"), "cell": (16.0, 18.0, 20.0)},
)

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: ``--fmad=false`` is carried for consistency with the track. NOTHING in these
#: six device strings contracts -- there is no ``a*b + c`` anywhere -- so the
#: unguarded control is EXPECTED to be identical, and the summary reports that
#: reading instead of implying the guard bought an answer.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fmad_false", ("--fmad=false",)),
    ("default_no_options", ()),
)


class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else the fixture exercises -- the fold, the stored extent, the
    dtype, the contiguity -- is a real object either way. Same shim the sibling
    gates use, so the three tracks' laptop legs are commensurable.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any]):
    """A frozen ``(fields, grid)`` pair for one spec. No PML: no pass reads one."""
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=0.5)
    fields = Fields(grid=grid)
    return fields, grid


def case_rng(label: str) -> np.random.Generator:
    """A per-case generator seeded from a sha256 of the case's own name.

    NOT ``hash()``: ``PYTHONHASHSEED`` salts the hash of a tuple of strings, so a
    hash-seeded case could not be replayed from the record that names it. The
    digest is stable across processes, machines and interpreter versions.
    """
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def seed_state(fields, grid, family: str, value_class: str,
               rng) -> Dict[str, np.ndarray]:
    """Fill the family's three arrays; return the host draw.

    NEVER ZERO-INIT. Every one of these passes writes a plane, and a state that
    already satisfies the boundary condition is a fixed point: the array path
    would move nothing and the case would certify nothing. ``oracle_moved``
    refuses such a case, but the fixture should not be manufacturing them.
    """
    xp = grid.xp
    names = FAMILY_COMPONENTS[family]
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(names, tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, family: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy()
            for name in FAMILY_COMPONENTS[family]}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance(fields, family: str) -> None:
    """Move the state between launches, exactly the same way on both paths."""
    for name in FAMILY_COMPONENTS[family]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The NumPy backend: a SECOND transcription of the device tree
# ---------------------------------------------------------------------------
#
# Written from the device strings' own index arithmetic -- a flat view of the
# volume, ``face_geometry``'s base/stride, the same component selections -- rather
# than from ``stepping``. Two independent transcriptions agreeing with each other
# AND with the array path is what makes "bit-identical" mean the contract rather
# than "the kernel reproduces whatever I wrote twice".

def _face_geometry(shape: Tuple[int, int, int], axis: int) -> Tuple[np.ndarray, int, int]:
    """``face_geometry``'s (base vector, stride, plane count) for one axis."""
    nx, ny, nz = shape
    if axis == 0:
        t = np.arange(ny * nz, dtype=np.int64)
        return t, ny * nz, ny * nz
    if axis == 1:
        t = np.arange(nx * nz, dtype=np.int64)
        return (t // nz) * (ny * nz) + (t % nz), nz, nx * nz
    t = np.arange(nx * ny, dtype=np.int64)
    return (t // ny) * (ny * nz) + (t % ny) * nz, 1, nx * ny


def _flat(fields, family: str, component: int) -> np.ndarray:
    return getattr(fields, FAMILY_COMPONENTS[family][component]).reshape(-1)


def numpy_zero_metal(fields, family: str, walls: Sequence[int]) -> None:
    """``zero_metal_B``/``_D``'s device tree in NumPy."""
    shape = tuple(int(n) for n in getattr(fields, FAMILY_COMPONENTS[family][0]).shape)
    for axis in range(3):
        if not int(bool(walls[axis])):
            continue
        base, _stride, _plane = _face_geometry(shape, axis)
        for component in in_seam_coverage.shift_zero_components(family, axis):
            _flat(fields, family, component)[base] = np.float32(0.0)


def numpy_fill_symmetry(fields, family: str, axis: int, phase: int) -> None:
    """``fill_symmetry_B``/``_D``'s device tree in NumPy: cell 0 = phase * cell 2."""
    shape = tuple(int(n) for n in getattr(fields, FAMILY_COMPONENTS[family][0]).shape)
    base, stride, _plane = _face_geometry(shape, axis)
    scale = np.float32(phase)
    for component in in_seam_coverage.shift_zero_components(family, axis):
        flat = _flat(fields, family, component)
        flat[base] = (scale * flat[base + 2 * stride]).astype(np.float32)


def numpy_fill_folded_far(fields, family: str, axis: int, phase: int,
                          reflect_row: int) -> None:
    """``fill_folded_far_B``/``_D``'s device tree in NumPy: last = -phase * row."""
    shape = tuple(int(n) for n in getattr(fields, FAMILY_COMPONENTS[family][0]).shape)
    base, stride, _plane = _face_geometry(shape, axis)
    last = shape[axis] - 1
    parity = np.float32(-np.float32(phase))
    for component in in_seam_coverage.shift_one_components(family, axis):
        flat = _flat(fields, family, component)
        flat[base + last * stride] = (
            parity * flat[base + reflect_row * stride]).astype(np.float32)


def run_numpy(pass_name: str, fields, family: str,
              launches: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    out = []
    for entry in launches:
        if pass_name == "zero_metal":
            numpy_zero_metal(fields, family, entry["axes"])
            out.append({"launched": True, "axes": list(entry["axes"])})
        elif pass_name == "fill_symmetry":
            numpy_fill_symmetry(fields, family, entry["axis"], entry["phase"])
            out.append({"launched": True, "axis": entry["axis"]})
        else:
            numpy_fill_folded_far(fields, family, entry["axis"], entry["phase"],
                                  entry["reflect_row"])
            out.append({"launched": True, "axis": entry["axis"]})
    return tuple(out)


def run_cuda(pass_name: str, fields, family: str,
             launches: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """The SHIPPED kernels, through their own public entry point.

    ``launches`` is passed explicitly -- the entry point's keyword-only gate door,
    which exists so a harness can hand in deliberately wrong plans. Passing the
    grid instead would derive them correctly and disarm every host mutation.
    """
    from meep_gpu.cuda_kernels import in_seam_passes  # noqa: PLC0415
    out = in_seam_passes.run_pass(pass_name, fields, family, launches=launches)
    cp.cuda.runtime.deviceSynchronize()
    return out


# ---------------------------------------------------------------------------
# The host mutations: they corrupt the PLAN, so they run on both backends
# ---------------------------------------------------------------------------

def mutate_plan(name: str, pass_name: str, launches: Tuple[Dict[str, Any], ...],
                grid) -> Tuple[Dict[str, Any], ...]:
    """One corrupted launch list. Every entry stays IN BOUNDS: a leg must measure a
    wrong ANSWER, not a memory fault."""
    entries = [dict(entry) for entry in launches]
    if name == "reverse_axis_order":
        return tuple(reversed(entries))
    if name == "drop_last_launch":
        return tuple(entries[:-1])
    if name == "flip_plan_phase":
        for entry in entries:
            if "phase" in entry:
                entry["phase"] = -int(entry["phase"])
        return tuple(entries)
    if name == "reflect_row_n_minus_two":
        for entry in entries:
            stored = int(grid.stored_cells(int(entry["axis"])))
            entry["reflect_row"] = stored - 2
        return tuple(entries)
    if name == "wall_every_axis":
        return tuple({"axes": (1, 1, 1)} for _ in entries) or ({"axes": (1, 1, 1)},)
    if name == "wall_folded_metallic_axis":
        # stepping._zero_metal's ``not is_mirrored`` term, armed. A folded
        # metallic axis's stored cell 0 is the fold's parity-weighted ghost, not a
        # wall; clearing it destroys the fold.
        walls = list(in_seam_coverage.zero_metal_axes(grid))
        for axis in range(3):
            if grid.is_mirrored(axis) and grid.is_metallic(axis):
                walls[axis] = True
        return ({"axes": tuple(int(bool(w)) for w in walls)},)
    raise KeyError(f"unknown host mutation {name!r}")


#: Host mutations per pass, and which must be caught. ``reverse_axis_order`` is
#: declared a NULL on the strength of the sibling Triton gate's own measurement
#: (14/14 identical, recorded as a null) and the arithmetic reason behind it:
#: every fill is a multiply by exactly +/-1 from a plane no other axis's fill
#: writes, so two axes' fills commute bitwise. It is carried anyway, paired with
#: ``drop_last_launch`` on the same plan, because "the order does not matter" is
#: worth more measured than argued -- and because a null with no must-be-caught
#: partner on the same object proves only that the leg does nothing.
HOST_MUTATIONS: Dict[str, Tuple[Tuple[str, bool], ...]] = {
    # (name, must_be_caught)
    "zero_metal": (("wall_every_axis", True),
                   ("wall_folded_metallic_axis", True)),
    "fill_symmetry": (("drop_last_launch", True),
                      ("flip_plan_phase", True),
                      ("reverse_axis_order", False)),
    "fill_folded_far": (("drop_last_launch", True),
                        ("flip_plan_phase", True),
                        ("reflect_row_n_minus_two", True),
                        ("reverse_axis_order", False)),
}

#: A host mutation that is a NO-OP on some specs is scored only where it is live.
#: Collapsing "unasked" into "inert" is how a gate acquires a silent hole.
HOST_MUTATION_APPLIES: Dict[str, Callable[[Dict[str, Any], Any], bool]] = {
    # A plan with one entry has nothing to reverse and nothing to drop.
    "reverse_axis_order": lambda spec, grid: len(spec["axes"]) >= 2,
    "drop_last_launch": lambda spec, grid: len(spec["axes"]) >= 1,
    # Baking ``stored - 2`` IS the right row at an even full count. MEASURED over
    # counts 12..19: even -> stored - 2, odd -> stored - 3.
    "reflect_row_n_minus_two":
        lambda spec, grid: any(int(grid.shape_full[a]) % 2 == 1
                               for a in range(3) if grid.is_mirrored(a)
                               and not grid.is_metallic(a)),
    # Only bites where some axis is NOT already walled.
    "wall_every_axis":
        lambda spec, grid: not all(in_seam_coverage.zero_metal_axes(grid)),
    # Only exists on a grid with a folded METALLIC axis.
    "wall_folded_metallic_axis":
        lambda spec, grid: any(grid.is_mirrored(a) and grid.is_metallic(a)
                               for a in range(3)),
    "flip_plan_phase": lambda spec, grid: len(spec["axes"]) >= 1,
}


# ---------------------------------------------------------------------------
# The source mutations: device-text rewrites, CUDA backend only
# ---------------------------------------------------------------------------
#
# NONE OF THE SHARED PROBE'S MUTATIONS EDIT THESE KERNELS -- they were written for
# the curl and constitutive bodies -- so the battery is this gate's own. Each
# returns ``(mutated, sites)``; a rewrite that matches nothing comes back NOT
# ARMED rather than passing silently.

def _sub(source: str, pattern: str, replacement: str) -> Tuple[str, int]:
    mutated, sites = re.subn(pattern, replacement, source)
    return mutated, sites


def _negative_zero(source: str) -> Tuple[str, int]:
    """``+0.0f`` becomes ``-0.0f``. Identical under ``==``, a different word."""
    return _sub(source, r"= 0\.0f;", "= -0.0f;")


def _wipe_row_one(source: str) -> Tuple[str, int]:
    """The wipe lands on stored row 1 instead of the wall at row 0."""
    return _sub(source, r"\[base\] = 0\.0f;", "[base + stride] = 0.0f;")


def _drop_x_wall(source: str) -> Tuple[str, int]:
    """The x wall is never cleared."""
    return _sub(source, r"if \(wall_x\) \{", "if (0) {")


def _transpose_face_strides(source: str) -> Tuple[str, int]:
    """``face_geometry``'s y branch takes the z branch's ROW STRIDE.

    MEASURED INERT ON ``zero_metal`` AND ONLY THERE, which is why it is carried as
    that pass's null: the wipe stores a constant at ``base`` and never steps a row,
    so the stride is dead code in it. On either fill it moves which plane the ghost
    images and must be caught.
    """
    return _sub(source, r"\*stride = nz;\n        \*base = \(t / nz\)",
                "*stride = 1;\n        *base = (t / nz)")


def _transpose_face_base(source: str) -> Tuple[str, int]:
    """``face_geometry``'s y branch takes the z branch's BASE: i and k confused.

    The wipe's counterpart to the stride defect above -- ``base`` is the one thing
    every pass uses, so this is the leg that shows the face decomposition is being
    measured at all on ``zero_metal``. Only live where the y face is.
    """
    return _sub(source,
                r"\*base = \(t / nz\) \* \(ny \* nz\) \+ \(t % nz\);",
                "*base = (t / ny) * (ny * nz) + (t % ny) * nz;")


def _mirror_source_row_one(source: str) -> Tuple[str, int]:
    """MIRROR_SOURCE_INDEX 2 becomes 1: the ghost images the wrong plane."""
    return _sub(source, r"base \+ 2 \* stride", "base + 1 * stride")


def _flip_near_parity(source: str) -> Tuple[str, int]:
    """The near fill's ``+phase`` becomes ``-phase``."""
    return _sub(source, r"= phase \* ", "= -phase * ")


def _flip_far_parity(source: str) -> Tuple[str, int]:
    """The far fill's ``-phase`` becomes ``+phase``: the shift-1 rule dropped."""
    return _sub(source, r"float parity = -phase;", "float parity = phase;")


def _far_writes_first_plane(source: str) -> Tuple[str, int]:
    """The far ghost lands on stored row 0 instead of the last row."""
    return _sub(source, r"\[base \+ last \* stride\] =", "[base] =")


def _far_reads_last_plane(source: str) -> Tuple[str, int]:
    """The far ghost images the row it is about to overwrite."""
    return _sub(source, r"\[base \+ reflect_row \* stride\]",
                "[base + last * stride]")


def _drop_second_component(source: str) -> Tuple[str, int]:
    """The two-component branch writes only one: the family inversion, half-done."""
    return _sub(source, r"\n    g1\[base[^\n]*\n", "\n")


def _swap_component_selection(source: str) -> Tuple[str, int]:
    """The single-component branch picks the neighbour's array."""
    mutated, sites = _sub(
        source, r"\(axis == 0\) \? Bx : \(\(axis == 1\) \? By : Bz\)",
        "(axis == 0) ? By : ((axis == 1) ? Bx : Bz)")
    if sites:
        return mutated, sites
    return _sub(source, r"\(axis == 0\) \? Dx : \(\(axis == 1\) \? Dy : Dz\)",
                "(axis == 0) ? Dy : ((axis == 1) ? Dx : Dz)")


def _commute_parity_scale(source: str) -> Tuple[str, int]:
    """``phase * f[...]`` becomes ``f[...] * phase``. A NULL: float32 multiply is
    bitwise commutative, so this must come back UNCAUGHT."""
    mutated, first = _sub(source, r"= phase \* (\w+)\[([^\]]+)\];",
                          r"= \1[\2] * phase;")
    mutated, second = _sub(mutated, r"= parity \* (\w+)\[([^\]]+)\];",
                           r"= \1[\2] * parity;")
    return mutated, first + second


def _reload_through_register(source: str) -> Tuple[str, int]:
    """The value round-trips through a named register before the store. A NULL: a
    float32 in a 32-bit register is the identity on the bits."""
    mutated, first = _sub(
        source, r"(\w+)\[base\] = phase \* (\w+)\[base \+ 2 \* stride\];",
        r"{ float v = phase * \2[base + 2 * stride]; \1[base] = v; }")
    mutated, second = _sub(
        mutated,
        r"(\w+)\[base \+ last \* stride\] = parity \* (\w+)\[base \+ reflect_row \* stride\];",
        r"{ float v = parity * \2[base + reflect_row * stride]; "
        r"\1[base + last * stride] = v; }")
    return mutated, first + second


def _copy_instead_of_multiply(source: str) -> Tuple[str, int]:
    """The parity multiply becomes a bare copy. POLICY-DEPENDENT, not scored.

    Under ``keep`` the multiply by exactly +1.0f is the identity on the bits and
    this is inert wherever the plane phase is +1 (and caught wherever it is -1).
    Under ``flush`` the multiply flushes a subnormal operand and the copy does
    not, so it is caught on the band class regardless of phase. Both readings are
    recorded; neither is a release clause.
    """
    mutated, first = _sub(source, r"= phase \* (\w+)\[([^\]]+)\];", r"= \1[\2];")
    mutated, second = _sub(mutated, r"= parity \* (\w+)\[([^\]]+)\];", r"= \1[\2];")
    return mutated, first + second


#: (name, transform, must_be_caught_or_None). ``None`` means POLICY-DEPENDENT:
#: recorded, reported, and NOT a release clause.
SOURCE_MUTATIONS: Dict[str, Tuple[Tuple[str, Callable[[str], Tuple[str, int]], Optional[bool]], ...]] = {
    "zero_metal": (
        ("write_negative_zero", _negative_zero, True),
        ("wipe_row_one", _wipe_row_one, True),
        ("drop_x_wall", _drop_x_wall, True),
        ("transpose_face_base", _transpose_face_base, True),
        # A NULL, and a MEASURED one rather than an argued one: the wipe stores a
        # constant at ``base`` and never steps a row, so the row stride is dead
        # code in it. Paired with ``transpose_face_base``, which edits the SAME
        # helper and must be caught -- which is what stops "inert" from being
        # indistinguishable from "never asked".
        ("transpose_face_strides", _transpose_face_strides, False),
    ),
    "fill_symmetry": (
        ("mirror_source_row_one", _mirror_source_row_one, True),
        ("flip_near_parity", _flip_near_parity, True),
        ("transpose_face_strides", _transpose_face_strides, True),
        ("transpose_face_base", _transpose_face_base, True),
        ("drop_second_component", _drop_second_component, True),
        ("swap_component_selection", _swap_component_selection, True),
        ("commute_parity_scale", _commute_parity_scale, False),
        ("reload_through_register", _reload_through_register, False),
        ("copy_instead_of_multiply", _copy_instead_of_multiply, None),
    ),
    "fill_folded_far": (
        ("flip_far_parity", _flip_far_parity, True),
        ("far_writes_first_plane", _far_writes_first_plane, True),
        ("far_reads_last_plane", _far_reads_last_plane, True),
        ("transpose_face_strides", _transpose_face_strides, True),
        ("transpose_face_base", _transpose_face_base, True),
        ("drop_second_component", _drop_second_component, True),
        ("swap_component_selection", _swap_component_selection, True),
        ("commute_parity_scale", _commute_parity_scale, False),
        ("reload_through_register", _reload_through_register, False),
        ("copy_instead_of_multiply", _copy_instead_of_multiply, None),
    ),
}

#: WHICH SPECS A SOURCE MUTATION IS LIVE ON. A rewrite that edits ONE axis branch
#: of the face decomposition is a NO-OP on a grid where that axis never launches,
#: and scoring it there reports PARTIAL for a reason about the case table rather
#: than about the kernel. Absent from this map means "live wherever the pass runs".
#:
#: Collapsing "not applicable" into "uncaught" is the same silent hole the host
#: battery's ``not_applicable_on`` exists to avoid, one level down.
SOURCE_MUTATION_APPLIES: Dict[str, Callable[[str, Any], bool]] = {
    # Both edit face_geometry's Y branch.
    "transpose_face_strides": lambda p, g: 1 in live_axes(p, g),
    "transpose_face_base": lambda p, g: 1 in live_axes(p, g),
    # Swaps the x and y arms of the single-component selector.
    "swap_component_selection": lambda p, g: bool({0, 1} & live_axes(p, g)),
    # Deletes the x wall.
    "drop_x_wall": lambda p, g: 0 in live_axes(p, g),
}


def live_axes(pass_name: str, grid) -> set:
    """Which axes this pass actually launches on this grid."""
    axes: set = set()
    for entry in in_seam_coverage.plan(pass_name, grid):
        if "axis" in entry:
            axes.add(int(entry["axis"]))
        else:
            axes.update(a for a, w in enumerate(entry["axes"]) if w)
    return axes

#: Mutations that only exist in one family's device string. ``drop_second_component``
#: needs a two-component branch (D on the near fill, B on the far one);
#: ``swap_component_selection`` needs a one-component branch. A leg that could not
#: match reports NOT ARMED, which is correct and is why they are still listed for
#: both families.
_FAMILY_ONLY = ("drop_second_component", "swap_component_selection")


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 family: str) -> float:
    """Fraction of the family's WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately.
    """
    moved = total = 0
    for name in FAMILY_COMPONENTS[family]:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "full_count_parity": ["even" if int(grid.shape_full[a]) % 2 == 0 else "odd"
                              for a in range(3)],
        "zero_metal_axes": [bool(w) for w in in_seam_coverage.zero_metal_axes(grid)],
        "mirror_phases": [None if p is None else int(p)
                          for p in in_seam_coverage.mirror_fill_phases(grid)],
        "far_reflect_rows": [None if r is None else int(r)
                             for r in in_seam_coverage.folded_far_rows(grid)],
    }


def cross_check_derivations(grid) -> Dict[str, Any]:
    """Every host derivation, checked against ``stepping``'s own on this grid.

    The predicate module restates ``_mirror_phases``, ``_stored_past_owned`` and
    ``_far_reflect_rows`` rather than importing them (it has to stay
    engine-import-free), and a restatement that has drifted is a silent wrong
    answer on one plane. This is the drift check, run per case and recorded.
    """
    engine_phases = stepping._mirror_phases(grid)
    engine_rows = stepping._far_reflect_rows(grid)
    ours_phases = in_seam_coverage.mirror_fill_phases(grid)
    ours_rows = in_seam_coverage.folded_far_rows(grid)
    ours_past = in_seam_coverage.stored_past_owned(grid)
    engine_past = tuple(stepping._stored_past_owned(grid, a) for a in range(3))
    # ``_far_reflect_rows`` reports a row on every folded non-metallic axis;
    # ``_fill_folded_far_ghosts`` only VISITS ``_stored_past_owned`` axes, so ours
    # is the engine's masked by that visit list. Compare on the visited axes.
    rows_agree = all(
        (ours_rows[a] == engine_rows[a]) if engine_past[a] else (ours_rows[a] is None)
        for a in range(3))
    return {
        "mirror_phases_agree": all(
            (ours_phases[a] is None and engine_phases[a] is None)
            or (engine_phases[a] is not None and ours_phases[a] == engine_phases[a])
            for a in range(3)),
        "stored_past_owned_agree": all(bool(ours_past[a]) == bool(engine_past[a])
                                       for a in range(3)),
        "far_reflect_rows_agree": rows_agree,
        "engine_far_reflect_rows": [None if r is None else int(r) for r in engine_rows],
    }


def parity_identity_check() -> Dict[str, Any]:
    """``mirror_parity(c, a, ph) == ph * (1 - 2*iyee[c][a])`` over every combination.

    The whole parity input to these kernels is one signed float per launch, and
    that closed form is why. Re-measured inside the gate rather than trusted from
    a comment, and recorded in the artifact.
    """
    checked = mismatches = 0
    for component, shifts in IYEE_SHIFTS.items():
        for axis in range(3):
            for phase in (1, -1):
                checked += 1
                if mirror_parity(component, axis, phase) != phase * (1 - 2 * shifts[axis]):
                    mismatches += 1
    return {"combinations": checked, "mismatches": mismatches,
            "holds": mismatches == 0}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, pass_name: str, spec: Dict[str, Any], family: str,
             value_class: str, guard: str,
             host_mutation: Optional[str] = None,
             multi_step: bool = True) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel."""
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    label = f"{pass_name}|{spec['label']}|{family}|{value_class}"
    rng = case_rng(label)

    fields, grid = build(xp, spec)
    host = seed_state(fields, grid, family, value_class, rng)

    case: Dict[str, Any] = {
        "pass": pass_name, "label": spec["label"], "family": family,
        "value_class": value_class, "guard": guard, "backend": backend,
        "host_mutation": host_mutation,
        "case_seed_label": label,
        "fold_axes": spec["axes"], "mirror_phase": spec["phase"],
        "boundaries": list(spec["boundaries"]),
        "structure": structure_facts(grid),
        "derivations": cross_check_derivations(grid),
        "operand_census": operand_census(host),
    }

    covered, reason = in_seam_coverage.covers(pass_name, fields, grid, family)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    launches = in_seam_coverage.plan(pass_name, grid)
    case["planned_launches"] = [dict(entry) for entry in launches]
    used = (launches if host_mutation is None
            else mutate_plan(host_mutation, pass_name, launches, grid))
    case["launches_used"] = [dict(entry) for entry in used]

    # THE TWO FLOORS ARE FOR AN UNMUTATED LEG, and they are relaxed for a mutated
    # one DELIBERATELY. "The array path does nothing here and the kernel writes a
    # plane" is not a vacuous case -- it is the strongest form of a catch, and it
    # is exactly what ``wall_folded_metallic_axis`` plants: on a grid whose every
    # metallic axis is folded, ``_zero_metal`` touches nothing, and a walled set
    # that included the fold would destroy it. Skipping that leg for "the oracle
    # moved nothing" would score the gate's sharpest mutation as NO LEGS.
    vacuous_ok = host_mutation is not None and bool(used)
    if not launches and not vacuous_ok:
        case["skipped"] = ("this grid has no live axis for this pass; a leg that "
                           "launched nothing and compared equal is vacuous")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, family)

    # Leg 1: the oracle -- stepping itself, on real Grid/Fields objects.
    ORACLES[(pass_name, family)](fields)
    reference = snapshot(fields, family)

    moved = oracle_moved(frozen, reference, family)
    case["oracle_moved"] = moved
    if moved == 0.0 and not vacuous_ok:
        case["skipped"] = ("the array path changed no word from the frozen input; "
                           "a state already satisfying this boundary condition is "
                           "a fixed point and certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    runner = run_cuda if backend == "cuda" else run_numpy
    case["launch_report"] = list(runner(pass_name, fields, family, used))

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in FAMILY_COMPONENTS[family]}
    case["single_launch"] = combine(parts)

    # Leg 3: MULTI_STEP_BUDGET consecutive launches, advanced between them.
    if multi_step and host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            ORACLES[(pass_name, family)](fields)
            advance(fields, family)
        oracle_multi = snapshot(fields, family)
        multi_moved = oracle_moved(frozen, oracle_multi, family)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(pass_name, fields, family, used)
            advance(fields, family)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in FAMILY_COMPONENTS[family]}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET
        case["multi_step"]["oracle_moved"] = multi_moved

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(pass_name: str, product: str) -> List[Tuple[Dict[str, Any], str, str]]:
    specs = SPECS if product == "full" else SPECS[:8]
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, family, value_class)
            for spec in specs for family in FAMILIES for value_class in classes]


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              pass_name: str, product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan_list = case_product(pass_name, product)
    for index, (spec, family, value_class) in enumerate(plan_list, start=1):
        case = one_case(backend, pass_name, spec, family, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] {index}/{len(plan_list)} {spec['label']} {family} "
                f"{value_class} SKIPPED: {case['skipped'][:58]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] {index}/{len(plan_list)} {spec['label']} {family} "
            f"{value_class} shape={case['structure']['shape']} "
            f"launches={len(case['launches_used'])} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.4f} ({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: sweep: a leg taken from the head of the case product would land entirely on the
#: unfolded controls, and every fold-only mutation would score 0/0 UNCAUGHT -- a
#: harness defect that reads exactly like a kernel defect.
MUTATION_SPEC_LABELS: Dict[str, Tuple[str, ...]] = {
    "zero_metal": ("wall_XYZ",                 # all three walls live
                   "wall_X_only",              # the slowest-stride wall alone
                   "wall_Z_only",              # the fastest-stride wall alone
                   "fold_X_metallic_wall_Y",   # a folded metallic axis beside a wall
                   "fold_X_wall_Y",            # a wall beside a folded periodic axis
                   "fold_XYZ_metallic"),       # three folded metallic axes: no wall
    "fill_symmetry": ("fold_X_periodic", "fold_Y_metallic", "fold_Z_periodic",
                      "fold_XY_mixed", "fold_XYZ_periodic", "fold_XYZ_metallic"),
    "fill_folded_far": ("fold_X_periodic", "fold_Y_periodic",
                        "fold_Z_periodic_odd_count", "fold_X_periodic_odd_count",
                        "fold_XY_periodic", "fold_XYZ_periodic",
                        "fold_XYZ_periodic_odd"),
}


def mutation_plan(pass_name: str, product: str) -> List[Tuple[Dict[str, Any], str]]:
    """One case per (spec, family) for the mutation legs, on the uniform class.

    Held to one value class deliberately: a mutation leg answers "can this gate
    see this defect at all", and multiplying it by the whole sweep buys
    repetitions of that answer rather than a second question.
    """
    by_label = {spec["label"]: spec for spec in SPECS}
    labels = MUTATION_SPEC_LABELS[pass_name]
    if product != "full":
        labels = labels[:3]
    return [(by_label[label], family) for label in labels for family in FAMILIES]


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _verdict(caught: int, scored: int, must_be_caught: Optional[bool]) -> str:
    if not scored:
        return "NO LEGS"
    if must_be_caught is None:
        return "POLICY DEPENDENT: CAUGHT" if caught else "POLICY DEPENDENT: UNCAUGHT"
    if must_be_caught:
        return "CAUGHT" if caught == scored else ("PARTIAL" if caught else "UNCAUGHT")
    return "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       pass_name: str, product: str) -> Dict[str, Any]:
    """Every plan-level defect, on both backends. A gate that cannot fail certifies nothing."""
    out: Dict[str, Any] = {}
    plan_list = mutation_plan(pass_name, product)
    for name, must_be_caught in HOST_MUTATIONS[pass_name]:
        legs: List[Dict[str, Any]] = []
        skipped_specs: List[str] = []
        for spec, family in plan_list:
            # The applicability question is about the GRID, which is
            # backend-independent, so it is asked on a NumPy grid even on the CUDA
            # leg: building a CuPy Fields per check would allocate device memory
            # to answer a question about cell counts.
            _fields, grid = build(_NumpyWearingCupysName(), spec)
            if not HOST_MUTATION_APPLIES[name](spec, grid):
                skipped_specs.append(f"{spec['label']}:{family}")
                continue
            legs.append(one_case(backend, pass_name, spec, family, "uniform",
                                 "fmad_false", host_mutation=name,
                                 multi_step=False))
        scored = [c for c in legs if not c.get("skipped")]
        caught = sum(1 for c in scored if leg_caught(c))
        out[name] = {
            "ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught,
            "not_applicable_on": skipped_specs,
            "verdict": _verdict(caught, len(scored), must_be_caught),
            "cases": legs,
        }
        log(f"[host-mut] {name}: caught {caught}/{len(scored)} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results: Dict[str, Any], out_path: str, pass_name: str,
                         product: str) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE (``compile_cache.kernel_cache_key``),
    so a mutated body is a miss and reaches NVRTC. Every leg records how many
    kernel constructions came from the mutated bytes: a leg reporting a pass for a
    mutation it never applied is worse than no leg.
    """
    from meep_gpu.cuda_kernels import compile_cache, in_seam_passes  # noqa: PLC0415

    originals = {family: getattr(in_seam_passes, SOURCE_ATTRIBUTE[(pass_name, family)])
                 for family in FAMILIES}
    out: Dict[str, Any] = {}
    plan_list = mutation_plan(pass_name, product)

    try:
        for family in FAMILIES:
            attribute = SOURCE_ATTRIBUTE[(pass_name, family)]
            for name, transform, must_be_caught in SOURCE_MUTATIONS[pass_name]:
                key = f"{family}:{name}"
                mutated, sites = transform(originals[family])
                if sites == 0 or mutated == originals[family]:
                    out[key] = {
                        "armed": False,
                        "why": (f"matched {sites} site(s) and changed nothing"
                                + (" (this branch exists only in the other family's "
                                   "device string)" if name in _FAMILY_ONLY else
                                   "; the mutation and the kernel have drifted apart")),
                    }
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                setattr(in_seam_passes, attribute, mutated)
                in_seam_passes._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                applies = SOURCE_MUTATION_APPLIES.get(name, lambda p, g: True)
                legs = []
                inert_on: List[str] = []
                classes = (("uniform",) if must_be_caught is not None
                           # The band class is what makes the policy question
                           # visible at all, and copy_instead_of_multiply is the
                           # leg it exists for.
                           else ("uniform", "subnormal_band"))
                for value_class in classes:
                    for spec, leg_family in plan_list:
                        if leg_family != family:
                            continue
                        _fields, leg_grid = build(_NumpyWearingCupysName(), spec)
                        if not applies(pass_name, leg_grid):
                            if value_class == "uniform":
                                inert_on.append(f"{spec['label']}:{family}")
                            continue
                        legs.append(one_case("cuda", pass_name, spec, family,
                                             value_class, "fmad_false",
                                             multi_step=False))
                setattr(in_seam_passes, attribute, originals[family])
                in_seam_passes._clear_kernel_cache()
                scored = [c for c in legs if not c.get("skipped")]
                caught = sum(1 for c in scored if leg_caught(c))
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                out[key] = {
                    "armed": True, "sites": sites,
                    "mutated_source_sha256": digest,
                    "kernel_constructions_from_mutated_bytes": from_mutated,
                    "ran": len(scored), "caught": caught,
                    "caught_by_value_class": {
                        cls: sum(1 for c in scored
                                 if c["value_class"] == cls and leg_caught(c))
                        for cls in VALUE_CLASSES},
                    "scored_by_value_class": {
                        cls: sum(1 for c in scored if c["value_class"] == cls)
                        for cls in VALUE_CLASSES},
                    "must_be_caught": must_be_caught,
                    "not_applicable_on": inert_on,
                    "verdict": _verdict(caught, len(scored), must_be_caught),
                    "cases": legs,
                }
                if from_mutated == 0 and scored:
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {caught}/{len(scored)} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for family in FAMILIES:
            setattr(in_seam_passes, SOURCE_ATTRIBUTE[(pass_name, family)],
                    originals[family])
        in_seam_passes._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any], pass_name: str) -> Dict[str, Any]:
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    single_ok = [c for c in scored if c["single_launch"]["bit_identical"]]
    multi_cases = [c for c in scored if "multi_step" in c]
    multi_ok = [c for c in multi_cases if c["multi_step"]["bit_identical"]]

    if not scored:
        reasons.append("no case was scored at all")
    if len(single_ok) != len(scored):
        reasons.append(f"single-launch divergence on {len(scored) - len(single_ok)} "
                       f"of {len(scored)} cases")
    if len(multi_ok) != len(multi_cases):
        reasons.append(f"multi-launch divergence on {len(multi_cases) - len(multi_ok)} "
                       f"of {len(multi_cases)} cases")

    # THE HOST DERIVATIONS MUST NOT HAVE DRIFTED from stepping's own, on any case.
    drifted = [c["label"] for c in scored
               if not all(c["derivations"][k] for k in
                          ("mirror_phases_agree", "stored_past_owned_agree",
                           "far_reflect_rows_agree"))]
    if drifted:
        reasons.append(f"in_seam_coverage's restatement of stepping's derivations "
                       f"disagreed with stepping on {sorted(set(drifted))}")

    identity = results.get("parity_identity", {})
    if not identity.get("holds"):
        reasons.append("mirror_parity(c, a, ph) != ph * (1 - 2*iyee[c][a]) on some "
                       "combination; the kernels' single signed parity is wrong")

    # EVERY AXIS MUST HAVE BEEN LIVE somewhere in the sweep. The three have the
    # slowest, middle and fastest stride, and a face decomposition that confuses
    # them is bit-identical on a sweep that only ever exercises one.
    live_axes = set()
    for case in scored:
        for entry in case["launches_used"]:
            if "axis" in entry:
                live_axes.add(int(entry["axis"]))
            else:
                live_axes.update(a for a, w in enumerate(entry["axes"]) if w)
    if live_axes != {0, 1, 2}:
        reasons.append(f"only axes {sorted(live_axes)} were ever live in this "
                       f"sweep; a face decomposition defect is invisible on the "
                       f"axes that never ran")

    # BOTH VALUE CLASSES, and the band class must really contain the band.
    classes = {c["value_class"] for c in scored}
    if classes != set(VALUE_CLASSES):
        reasons.append(f"value classes scored were {sorted(classes)}, not "
                       f"{sorted(VALUE_CLASSES)}")
    band = [c for c in scored if c["value_class"] == "subnormal_band"]
    if band and not any(c["operand_census"]["subnormals"] > 0 for c in band):
        reasons.append("no subnormal-band case actually held a subnormal operand; "
                       "the policy legs measured the fixture, not the policy")

    # BOTH FULL-COUNT PARITIES on a live folded axis, for the far fill only --
    # that is where the image row differs between them.
    if pass_name == "fill_folded_far":
        parities = set()
        for case in scored:
            for entry in case["launches_used"]:
                axis = int(entry["axis"])
                parities.add(case["structure"]["full_count_parity"][axis])
        if parities != {"even", "odd"}:
            reasons.append(f"full-count parities exercised on a live folded axis "
                           f"were {sorted(parities)}, not both; the image row is "
                           f"stored-2 at one and stored-3 at the other")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not "
                           f"inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            # A mutation whose branch exists only in the other family's device
            # string is correctly NOT ARMED and is not a defect.
            if key.split(":", 1)[1] in _FAMILY_ONLY:
                continue
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"].startswith("POLICY DEPENDENT"):
            continue
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    control = [c for c in sweep.get("default_no_options", []) if not c.get("skipped")]
    control_diverged = [c for c in control if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored": len(control),
        "diverged": len(control_diverged),
        "reading": (
            "NOT MEASURED on this run: no unguarded leg was scored"
            if not control else
            "the contraction guard changed an answer here, which would be a "
            "surprise: no device string in this module contains a contractible "
            "a*b + c"
            if control_diverged else
            "MEASURED DECORATIVE, as expected: none of these device strings has a "
            "contractible expression, so --fmad=false has nothing to disable and "
            "this run carries no evidence that it changed an answer"),
    }

    return {
        "released": not reasons,
        "pass": pass_name,
        "reasons": reasons,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "live_axes": sorted(live_axes),
        "value_classes": sorted(classes),
        "claim": (f"the shipped CUDA {pass_name} kernels (B and D) are "
                  f"byte-identical to stepping's own pass, per launch and over "
                  f"{MULTI_STEP_BUDGET} consecutive launches, on every "
                  f"configuration this sweep scored"),
        "does_not_claim": [
            "NO FUSION. This gate measures one pass alone against one array-path "
            "function. It licenses no fused product and does not shorten any seam "
            "by itself; what it supplies is a part such a product would have to "
            "absorb",
            "the other two in-seam passes: each is gated separately and a verdict "
            "here says nothing about them",
            "COMPLEX STORAGE. A Bloch run stores complex64 and these kernels index "
            "float32; the predicate refuses it by name and no complex case was run",
            "the cylindrical radial axis: refused by name, never swept",
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all",
            "no throughput claim of any kind is made or measurable from this run",
        ],
    }


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass", dest="pass_name", required=True, choices=PASSES)
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_in_seam_passes",
        "pass": args.pass_name,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": (f"is the shipped CUDA {args.pass_name} pair byte-identical to "
                     f"stepping's own {args.pass_name} pass?"),
        "parity_identity": parity_identity_check(),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip
        # wraps it, so it records the option tuple NVRTC was really given.
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}
    save(results, args.out)

    for guard, options in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue  # there is no compiler on this leg to guard
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import in_seam_passes  # noqa: PLC0415
            in_seam_passes._COMPILE_OPTIONS = tuple(options)
            in_seam_passes._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.pass_name, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import in_seam_passes  # noqa: PLC0415
        in_seam_passes._COMPILE_OPTIONS = ("--fmad=false",)
        in_seam_passes._clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.backend, args.pass_name,
                           args.product)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.pass_name, args.product)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results, args.pass_name)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] pass={args.pass_name} released={verdict['released']} "
        f"scored={verdict['scored_cases']} "
        f"single_identical={verdict['single_launch_identical']} "
        f"multi_identical={verdict['multi_step_identical']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
