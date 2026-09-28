"""Can a fused curl->constitutive pair carry a source deposit, byte-identically?

THE QUESTION, as it stood on 2026-08-22. Every fusion board then excluded 215 of the 387
seam-instances priced on the 186-row basis because the driver injects a source BETWEEN
the curl and the constitutive half -- magnetic at ``driver.py:3283-3284``, electric at
``driver.py:3294-3299``. That exclusion had been read as structural. It is not: it is a
property of the shipped product's SHAPE, and the repair this probe licensed has since
crossed it. Measured
separately, the deposit touches a median 0.016% of a real grid and at most 3.45%, so a
whole-grid fused launch plus a REPAIR over the deposit points is arithmetically
admissible and cheap.

This probe asks whether it is EXACT, on real ``Grid``/``Fields``/``PML`` objects and the
shipped ``stepping`` functions, comparing uint32 words. No device and no Triton: what is
being tested is the ARCHITECTURE, not a kernel. A device gate would re-run this shape
with the real fused kernel in place of leg B's emulated pair.

THE DEVICE LEG (``--backend triton``). Leg B above emulates the fused product with the
shipped array functions, which tests the ARCHITECTURE and not a kernel. With
``--backend triton`` leg B instead builds the real product through
``triton_kernels.plan_fused_pair(fields, pml, "D", sources=...)`` and launches
``fused_curl_constitutive_D`` -- the same kernel the certified welds cover.

THE PREDICATE IS DELIBERATELY BYPASSED THERE, and only there. ``fused_pair_coverage``
refuses any row carrying an in-seam source (``triton_kernels/coverage.py:477-481``), which
is the clause the deposit work proposes to change; a probe that respected it could never
ask the question. So the device leg passes ``sources=()`` to obtain the plan and then
injects for real. That is NOT a claim the predicate should admit blindly -- it is the
measurement that would license changing it, and the repair is what makes the answer right.

THE TWO LEGS, per step.

  A -- THE REFERENCE, the driver's own order (``driver.py:3291-3304``)::

        step_D ; inject ; fill_symmetry_bc_D ; zero_metal_D ;
        fill_folded_far_ghosts_D ; update_E

  B -- FUSED PLUS REPAIR. The fused product runs at the ``step_D`` consult and does both
    halves against an UNINJECTED D, carrying the wall clear inline the way the shipped
    kernel does (``triton_kernels/kernels.py:474-479`` applies ZM_* to the curl output
    before storing). The driver then injects and runs its own unconditional
    ``zero_metal_D``, and the ``update_E`` consult -- which the fused arm also owns --
    spends itself on the repair::

        save fw,E at the deposit points ; step_D ; zero_metal_D ; update_E ;
        inject ; fill_symmetry_bc_D ; zero_metal_D ; fill_folded_far_ghosts_D ;
        repair(deposit points)

WHY THE REPAIR IS A REPAIR AND NOT A SECOND INJECTION. The constitutive half is STATEFUL:
``_apply_constitutive_pml`` (``stepping.py:2130-2134``) reads ``fw`` BEFORE overwriting
it and then ACCUMULATES into the field::

        fw_previous = fw.copy() ; fw[...] = source
        field += kps * fw ; field -= kms * fw_previous

Correcting that after the fact by adding a delta is not float-exact. The repair instead
restores the two arrays at the deposit points from values saved BEFORE the fused launch
and recomputes them, in the same operand order, from the injected D. Every other cell is
already correct, because the deposit changed nothing there.

WHY NO WALL CLEAR IS NEEDED IN THE REPAIR. ``driver.py:3301`` calls ``zero_metal_D``
unconditionally, after the injection -- it is not behind a dispatch consult. So by the
time the ``update_E`` consult is reached the D array is already injected, filled and
cleared. A deposit that lands on a metallic wall cell is zero in both legs.

WHAT A PASS DOES NOT LICENSE. Nothing about a device, nothing about Triton, nothing about
throughput, and nothing about rows whose seam also carries a live ``fill_symmetry_bc_D``
or ``fill_folded_far_ghosts_D`` pass -- those clauses stay in the predicate.

Progress: one flushed line per case, and the artifact is written whether the verdict
releases or not (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent.parent
for entry in (str(API), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy  # noqa: E402

from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

ELECTRIC = ("Ex", "Ey", "Ez")
DISPLACEMENT = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}


def _words(array) -> numpy.ndarray:
    """Raw bits, so a comparison is bytes and not a tolerance."""
    if hasattr(array, "get"):          # a CuPy array: compare on the host
        array = array.get()
    flat = numpy.ascontiguousarray(array).ravel()
    if numpy.iscomplexobj(flat):
        flat = flat.view(numpy.float32 if flat.dtype == numpy.complex64 else numpy.float64)
    if flat.dtype == numpy.float32:
        return flat.view(numpy.uint32)
    if flat.dtype == numpy.float64:
        return flat.view(numpy.uint64)
    return flat


def _snapshot(fields) -> dict:
    out = {}
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz",
                 "Bx", "By", "Bz", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "fu_Dx", "fu_Dy", "fu_Dz", "fu_Bx", "fu_By", "fu_Bz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        out[name] = numpy.array(array.get() if hasattr(array, "get") else array, copy=True)
    return out


def _differing(left: dict, right: dict) -> dict:
    out = {}
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            out[name] = "present on one side only"
            continue
        wa, wb = _words(a), _words(b)
        if wa.shape != wb.shape:
            out[name] = f"shape {wa.shape} vs {wb.shape}"
            continue
        n = int((wa != wb).sum())
        if n:
            out[name] = n
    return out


def _array_module(backend: str):
    """The array module the fields live on. A Triton plan binds DEVICE pointers, so a
    NumPy-backed Fields yields no plan at all -- which the first device run reported as
    'dispatched none' rather than passing on the emulation's result."""
    if backend != "triton":
        return None
    import cupy  # noqa: PLC0415
    return cupy


def _build(cell=(1.2, 1.2, 0.0), resolution=12.0, thickness=2, seed=20260822,
           complex_fields=False, backend="numpy"):
    # A zero-extent axis is a REDUCED-DIMENSION run, and Grid wants that declared
    # rather than inferred: cell_size components must be positive on every axis the
    # run resolves, so a 2-D case must say dimensions=2.
    dimensions = sum(1 for extent in cell if extent > 0.0)
    xp = _array_module(backend)
    grid = (Grid(resolution=resolution, cell_size=cell, dimensions=dimensions, xp=xp)
            if xp is not None else
            Grid(resolution=resolution, cell_size=cell, dimensions=dimensions))
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    # PER AXIS, because an axis the run does not resolve is one cell wide and a PML
    # shell does not fit on it. The absorber goes on the resolved axes only.
    per_axis = tuple(thickness if extent > 0.0 else 0 for extent in cell)
    pml = PML(grid=grid, thickness=per_axis)
    # BEFORE SEEDING, so the split-field and auxiliary volumes the absorber needs exist
    # and get seeded too. Seeding only the primary arrays would leave the auxiliaries at
    # zero, and zero-init is a FIXED POINT of the constitutive recurrence -- the case
    # would then pass without exercising the state the repair restores.
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz",
                 "Bx", "By", "Bz", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "fu_Dx", "fu_Dy", "fu_Dz", "fu_Bx", "fu_By", "fu_Bz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        values = generator.standard_normal(array.shape)
        if numpy.iscomplexobj(array):
            filled = (values + 1j * generator.standard_normal(array.shape)).astype(
                numpy.complex64 if array.dtype == numpy.complex64 else numpy.complex128)
        else:
            filled = values.astype(numpy.float32 if array.dtype == numpy.float32
                                   else numpy.float64)
        array[...] = xp.asarray(filled) if xp is not None else filled
    return grid, fields, pml


def _make_source(grid, component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)):
    return GaussianPulsedSource(grid=grid, component=component, center=center,
                                size=size, frequency=1.0, fwidth=0.2, amplitude=1.0)


def _deposit_index(source):
    """The (ix, iy, iz) the injection actually writes, as plain integer arrays."""
    ix, iy, iz = source._point_ix, source._point_iy, source._point_iz
    if ix is None:
        return None
    return (ix, iy, iz)      # already on the grid's array module; do not move them


def _leg_a(fields, pml, sources, when, dt):
    """The driver's own order, verbatim."""
    stepping.step_D(fields, pml)
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


def _fused_pair_D(fields, pml):
    """The real product, or None where it cannot be built on this host."""
    try:
        from meep_gpu import triton_kernels
    except Exception:      # noqa: BLE001 - no Triton on this host
        return None
    try:
        # sources=() ON PURPOSE: see the module docstring. The predicate's in-seam-source
        # clause is the thing under test, so it is stepped around here and nowhere else.
        return triton_kernels.plan_fused_pair(fields, pml, "D", sources=())
    except Exception:      # noqa: BLE001
        return None


class _HostPair:
    """The fused product's two halves, for a host leg with no Triton.

    Stands in for the kernel at the FIRST consult only. The repair under test is the
    shipped component either way, so what this changes is whether the arithmetic it
    repairs came from one launch or from three array calls -- which is exactly the
    difference the device leg exists to close.
    """

    def __init__(self, fields, pml):
        self._fields, self._pml = fields, pml

    def run(self, *_args, **_kwargs):
        stepping.step_D(self._fields, self._pml)
        stepping.zero_metal_D(self._fields)   # the kernel's inline ZM_* (kernels.py:474-479)
        stepping.update_E(self._fields, self._pml)


def _leg_b(fields, pml, sources, when, dt, repair=True, plan=None):
    """Fused pair at the step_D consult, repair at the update_E consult.

    THIS DRIVES THE SHIPPED OBJECTS, not a copy of their logic. Earlier revisions of this
    probe carried their own save/restore inline, which established the SHAPE and is what
    the 2026-08-22 records measured -- but a probe that reimplements the thing it certifies
    cannot license the thing that ships. It now runs
    ``deposit_repair.LeadingRepairPlan`` and ``TrailingRepairPlan``, the exact pair
    ``triton_kernels.launch._install_fused_pair`` puts in a fused pair's two slots.
    """
    inner = plan if plan is not None else _HostPair(fields, pml)

    if not repair:
        # THE NULL CONTROL. The fused halves run, the driver's own passes run, and
        # nothing is saved or restored. If this leg also matches the reference then the
        # case cannot tell a working repair from a missing one, and its PASS is worth
        # nothing.
        inner.run()
        for source in sources:
            source.inject(fields, when + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        return

    leading = deposit_repair.LeadingRepairPlan(inner, fields, pml, sources, "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, fields, pml)

    leading.run()          # saves the deposit points, then ONE launch: curl + ZM + constitutive

    # --- the driver's own unconditional passes, untouched ---
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)

    trailing.run()         # the repair, at the deposit points only
    if not leading.repairs:
        raise AssertionError(
            "the repair reported touching no point: a leg that repairs nothing and still "
            "matches the reference is measuring the reference against itself")


def run_case(name: str, *, steps: int, cell, resolution, thickness, component,
             center, size, complex_fields, progress, backend="numpy") -> dict:
    record = {"case": name, "steps": steps, "cell": list(cell),
              "resolution": resolution, "pml_thickness": thickness,
              "component": component, "center": list(center), "size": list(size),
              "complex_fields": complex_fields}
    started = time.time()
    try:
        grid_a, fields_a, pml_a = _build(cell, resolution, thickness, complex_fields=complex_fields, backend=backend)
        grid_b, fields_b, pml_b = _build(cell, resolution, thickness, complex_fields=complex_fields, backend=backend)
        pre = _differing(_snapshot(fields_a), _snapshot(fields_b))
        record["precondition_identical"] = not pre
        if pre:
            record["precondition_differs"] = pre
            return record
        src_a = [_make_source(grid_a, component, center, size)]
        src_b = [_make_source(grid_b, component, center, size)]
        idx = _deposit_index(src_a[0])
        record["deposit_points"] = 0 if idx is None else int(len(idx[0]))
        record["grid_cells"] = int(grid_a.nx * grid_a.ny * grid_a.nz)
        record["deposit_fraction"] = (record["deposit_points"] / record["grid_cells"]
                                      if record["grid_cells"] else None)
        if not record["deposit_points"]:
            record["note"] = "no deposit points: this case would not discriminate"
            return record
        dt = grid_a.dt
        plan = _fused_pair_D(fields_b, pml_b) if backend == "triton" else None
        record["backend"] = backend
        record["real_kernel_dispatched"] = plan is not None
        if backend == "triton" and plan is None:
            record["note"] = ("the fused pair could not be built on this host; a device "
                              "leg that silently fell back to the array emulation would "
                              "report the architecture's result as the kernel's")
            return record
        for step in range(steps):
            when = step * dt
            _leg_a(fields_a, pml_a, src_a, when, dt)
            _leg_b(fields_b, pml_b, src_b, when, dt, plan=plan)
        record["differing"] = _differing(_snapshot(fields_a), _snapshot(fields_b))
        record["identical"] = not record["differing"]

        # THE NULL CONTROL, on its own pair of fields: the same protocol with the repair
        # withheld. It MUST diverge, or this case is not measuring the repair.
        grid_c, fields_c, pml_c = _build(cell, resolution, thickness, complex_fields=complex_fields, backend=backend)
        grid_d, fields_d, pml_d = _build(cell, resolution, thickness, complex_fields=complex_fields, backend=backend)
        src_c = [_make_source(grid_c, component, center, size)]
        src_d = [_make_source(grid_d, component, center, size)]
        control_plan = _fused_pair_D(fields_d, pml_d) if backend == "triton" else None
        for step in range(steps):
            when = step * dt
            _leg_a(fields_c, pml_c, src_c, when, dt)
            _leg_b(fields_d, pml_d, src_d, when, dt, repair=False, plan=control_plan)
        control = _differing(_snapshot(fields_c), _snapshot(fields_d))
        record["control_differing"] = control
        record["control_diverged"] = bool(control)
        record["control_words"] = sum(v for v in control.values() if isinstance(v, int))
    except BaseException as exc:  # noqa: BLE001
        import traceback
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        record["traceback"] = traceback.format_exc()[-900:]
    finally:
        record["seconds"] = round(time.time() - started, 3)
    progress(f"{name}: identical={record.get('identical')} "
             f"control_diverged={record.get('control_diverged')} "
             f"control_words={record.get('control_words')} "
             f"pts={record.get('deposit_points')} "
             f"frac={record.get('deposit_fraction')} "
             f"{record.get('error', '')}")
    return record


CASES = (
    # name,                      steps, cell,            res,  pml, comp, center,            size,             complex
    ("point_ez_pml",                 6, (1.2, 1.2, 0.0), 12.0, 2, "Ez", (0.0, 0.0, 0.0),  (0.0, 0.0, 0.0), False),
    ("point_ez_pml_multistep",      24, (1.2, 1.2, 0.0), 12.0, 2, "Ez", (0.0, 0.0, 0.0),  (0.0, 0.0, 0.0), False),
    ("point_ex_pml",                 6, (1.2, 1.2, 0.0), 12.0, 2, "Ex", (0.1, -0.2, 0.0), (0.0, 0.0, 0.0), False),
    ("line_ez_pml",                  6, (1.2, 1.2, 0.0), 12.0, 2, "Ez", (0.0, 0.0, 0.0),  (0.6, 0.0, 0.0), False),
    ("point_ez_pml_complex",         6, (1.2, 1.2, 0.0), 12.0, 2, "Ez", (0.0, 0.0, 0.0),  (0.0, 0.0, 0.0), True),
    ("point_ez_3d",                  4, (0.8, 0.8, 0.8), 10.0, 2, "Ez", (0.0, 0.0, 0.0),  (0.0, 0.0, 0.0), False),
    # DEPOSIT ON THE PML SHELL: the wall/absorber interaction the repair must not disturb.
    ("point_ez_in_pml",              6, (1.2, 1.2, 0.0), 12.0, 2, "Ez", (0.5, 0.5, 0.0),  (0.0, 0.0, 0.0), False),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None)
    parser.add_argument("--backend", default="numpy", choices=("numpy", "triton"),
                        help="'numpy' emulates the fused pair with the shipped array "
                             "functions and tests the ARCHITECTURE; 'triton' builds the "
                             "real product and launches fused_curl_constitutive_D, which "
                             "needs a CUDA host.")
    args = parser.parse_args()

    def progress(message: str) -> None:
        print(f"[deposit-repair] {message}", flush=True)

    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    rows = []
    for (name, steps, cell, res, pml, comp, center, size, cplx) in CASES:
        if wanted and name not in wanted:
            continue
        rows.append(run_case(name, steps=steps, cell=cell, resolution=res,
                             thickness=pml, component=comp, center=center,
                             size=size, complex_fields=cplx, progress=progress,
                             backend=args.backend))

    scored = [r for r in rows if "identical" in r]
    verdict = {
        "cases": len(rows),
        "scored": len(scored),
        "identical": sum(1 for r in scored if r["identical"]),
        "errors": sum(1 for r in rows if r.get("error")),
        "control_diverged": sum(1 for r in scored if r.get("control_diverged")),
        # A CASE RELEASES ONLY IF BOTH HOLD: the repaired leg matches AND the unrepaired
        # one does not. The second clause is what stops a vacuous pass.
        "release": bool(scored)
                   and all(r["identical"] and r.get("control_diverged") for r in scored)
                   and not any(r.get("error") for r in rows),
    }
    verdict["backend"] = args.backend
    verdict["real_kernel_dispatched"] = sum(1 for r in rows if r.get("real_kernel_dispatched"))
    if args.backend == "triton" and not verdict["real_kernel_dispatched"]:
        verdict["release"] = False
        verdict["why_not"] = ("asked for the real kernel and dispatched none; a pass here "
                              "would be the array emulation wearing the device leg's name")
    payload = {"probe": "deposit_repair", "rows": rows, "verdict": verdict,
               "what_it_does_not_license": (
                   "nothing about a device, Triton, throughput, or rows whose seam carries a "
                   "live fill_symmetry_bc_D or fill_folded_far_ghosts_D pass")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    progress(f"verdict={json.dumps(verdict)} -> {out}")
    return 0 if verdict["release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
