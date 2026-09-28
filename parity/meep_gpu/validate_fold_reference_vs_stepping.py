"""Pin the folded-grid gate's REFERENCE against ``stepping.py``, on NumPy, no GPU.

The twin of ``validate_pml_reference_vs_stepping.py``, for the fold. The device
gate (``gate_triton_symmetry.py``) compares the KERNEL against a reference
transcribed by hand from ``stepping.py``; this compares that REFERENCE against
the array path itself, which is what stops "bit-identical" from meaning "the
kernel reproduces whatever I wrote twice".

It runs anywhere: NumPy stands in for CuPy at import (the gate module imports
``cupy`` at module scope and never needs a device for its reference half).

    python -u validate_fold_reference_vs_stepping.py

Eight configurations, chosen for what each one is the only witness to: both
terminations (folded PERIODIC, folded METALLIC), both full-count parities (the
reflect row is ``stored - 2`` at an even count and ``stored - 3`` at an odd one),
both plane phases, a fold on each of X, Y and Z, and two planes at once. Exit
status is the verdict.
"""
import os
import sys
import types

import numpy as np

API = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, API)
sys.path.insert(0, os.path.join(API, "parity", "meep_gpu"))
sys.modules.setdefault("cupy", np)          # the gate imports cupy at module scope

import gate_triton_symmetry as gate          # noqa: E402
from meep_gpu import stepping                # noqa: E402
from meep_gpu.fields import Fields           # noqa: E402
from meep_gpu.grid import Grid, Mirror       # noqa: E402
from meep_gpu.pml import PML                 # noqa: E402
from meep_gpu.triton_kernels import symmetry  # noqa: E402

NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "By", "Bz", "Dx", "Dy", "Dz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")

CASES = (
    ("Y", "periodic", (1.6, 2.0, 1.2), 1),
    ("Y", "periodic", (1.6, 2.1, 1.2), 1),     # odd full count -> reflect row n-3
    ("Y", "periodic", (1.6, 2.0, 1.2), -1),    # odd plane
    ("Y", "metallic", (1.6, 2.0, 1.2), 1),
    ("X", "periodic", (2.1, 1.2, 1.2), 1),
    ("Z", "metallic", (1.2, 1.2, 2.0), 1),
    ("XY", "periodic", (2.0, 2.0, 1.2), 1),
    ("XY", "periodic", (2.1, 2.0, 1.2), -1),
)

BOUNDARY_STRING = {"periodic": gate.PERIODIC, "metallic": gate.METALLIC}


def boundaries_of(grid, pml):
    kinds = stepping._boundary_kinds(grid, pml)
    out = []
    for axis, kind in enumerate(kinds):
        if kind != "mirror":
            out.append(kind)
        elif stepping._stored_past_owned(grid, axis):
            out.append(gate.MIRROR_PERIODIC)
        else:
            out.append(gate.MIRROR_METALLIC)
    return tuple(out)


def coefficients_of(pml, half_integer):
    suffix = "_h" if half_integer else ""
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


failures = 0
for axes, declaration, cell, phase in CASES:
    planes = tuple(Mirror(name, phase) for name in axes)
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=declaration,
                symmetry=planes, xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    rng = np.random.default_rng(gate.SEED + 31)
    for name in NAMES:
        getattr(fields, name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(np.float32)

    mine = {name: getattr(fields, name).copy() for name in NAMES}
    boundaries = boundaries_of(grid, pml)
    phases = tuple(grid.mirror_phase(a) or 1 for a in range(3))
    rows = stepping._far_reflect_rows(grid)

    worst = []
    for _ in range(3):
        for sub_step in ("step_B", "step_D"):
            getattr(stepping, sub_step)(fields, pml)
            gate.reference_pml_step(
                np, mine, coefficients_of(pml, sub_step == "step_B"),
                grid.dt / grid.dx, sub_step, boundaries, phases, rows,
                "array_order")
            for name in NAMES:
                a = np.ascontiguousarray(getattr(fields, name)).view(np.uint32)
                b = np.ascontiguousarray(mine[name]).view(np.uint32)
                if not np.array_equal(a, b):
                    worst.append((sub_step, name, int((a != b).sum())))
    ok = not worst
    failures += 0 if ok else 1
    print(f"{axes}/{declaration}/phase{phase:+d} shape={tuple(grid.shape)} "
          f"stored={[grid.stored_cells(i) for i in range(3)]} "
          f"owned={[grid.owned_cells(i) for i in range(3)]} "
          f"codes={symmetry.folded_axis_kinds(grid, pml)[0]} rows={rows} "
          f"boundaries={boundaries}: reference==stepping -> {ok}"
          + ("" if ok else f"  DIVERGED {worst[:6]}"), flush=True)

print(f"\n{len(CASES) - failures}/{len(CASES)} cases bit-identical to stepping.py")
sys.exit(1 if failures else 0)
