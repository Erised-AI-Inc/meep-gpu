"""Survey MEEP's own example scripts through ``meep_gpu.gpu_compatibility``.

The coverage question this answers: of the simulations MEEP users actually write,
what fraction can :func:`meep_gpu.lift_simulation` run, and for the rest, is every
refusal a *specific named reason* rather than a silent approximation? The corpus is
the upstream ``meep/python/examples`` tree — third-party code neither this package
nor its tests were written against.

Each script is executed in its own subprocess with MEEP's terminal calls stubbed
(``Simulation.run``, ``init_sim``, ``solve_cw``, ``run_k_points``, and MPB's
``ModeSolver.init_params``), so the ``mp.Simulation`` is captured **at the moment
the script asks for it to run** — after any ``add_flux`` monitors, after any
``change_sources`` — without a single time step or grid allocation. Nothing under
``examples/`` is written to: the child runs in a scratch directory with the corpus's
data files symlinked in.

Progress reporting: one flushed line per example as it is attempted, and each result is appended
to the JSONL as it lands, so the run is readable while it is still going.

Usage (from ``the repository root``)::

    python -m parity.meep_gpu.survey_meep_examples \\
        --examples /path/to/meep/python/examples --out parity/meep_gpu/results/survey

``--one <script>`` is the child mode and is not meant to be called by hand.

The capture itself is factored out as ``_capture`` inside :data:`CHILD_PREAMBLE`
and is shared: ``sweep_corpus_lift_parity.py`` execs the same preamble to obtain
the same ``mp.Simulation``, then LIFTS and STEPS it. There is one capture in this
directory, not two, so "the object the survey scored" and "the object the sweep
lifted" cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Keyword -> short tag for ranking refusals by frequency. Ordered: the first match
# wins, so the more specific patterns come first. Reason text is the contract
# GpuCompatibility publishes; this table is only the survey's histogram key.
REASON_TAGS: tuple[tuple[str, str], ...] = (
    ("cylindrical coordinates", "cylindrical"),
    ("-D simulation is not implemented", "reduced_dimension"),
    ("special_kz", "special_kz"),
    ("azimuthal mode number", "cylindrical_m"),
    ("k_point is unset", "k_point_unset"),
    ("differ in more than their permittivity", "media_differ_beyond_epsilon"),
    ("not normal to a coordinate axis", "curved_or_rotated_geometry"),
    ("and a PML absorbs on the same axis", "bloch_on_a_pml_axis"),
    ("already been initialized", "post_init_mutation"),
    ("distinct media", "structured_epsilon"),
    ("material_function", "material_function"),
    ("epsilon_func", "epsilon_func"),
    ("epsilon_input_file", "epsilon_input_file"),
    ("rather than plain mp.Medium", "material_not_medium"),
    # add_flux / add_dft_fields are migrated onto the driver now, so the only monitor
    # refusal left is the kind whose accumulation this engine does not implement.
    ("this engine cannot rebuild", "unmigratable_monitor"),
    ("not a plain mp.Source", "special_source"),
    ("declares no sources", "no_sources"),
    ("time dependence", "unsupported_src_time"),
    ("amp_func_file", "amp_func_file"),
    ("amp_data", "amp_data"),
    ("amp_func on a zero-size", "amp_func_on_point"),
    ("mp.Absorber", "absorber"),
    ("custom pml_profile", "pml_profile"),
    ("R_asymptotic", "pml_r_asymptotic"),
    ("mean_stretch", "pml_mean_stretch"),
    # `pml_fractional_cells` used to be labelled from "not a whole number". That
    # refusal is gone: a PML that is not a whole number of cells is what MEEP builds
    # and what meep_gpu.pml now builds (pml.half_cell_extent). The three scripts it
    # blocked — planar_cavity_ldos.py (35.5 cells), cavity-farfield.py (66.666667),
    # cylinder_cross_section.py (27.488936) — now all report LIFTABLE here. Read that
    # for what it is: this survey scores gpu_compatibility and never calls
    # lift_simulation, and measured, only planar_cavity_ldos actually builds a driver
    # (grid 462 x 1 x 497, faces ((0, 35.5), (0, 0), (35.5, 35.5))). See the parity
    # matrix's PML row for where the other two stop.
    ("is not an mp.PML", "boundary_not_pml"),
    ("is not implemented: this engine folds mirror", "non_mirror_symmetry"),
    ("phase that is not +1 or -1", "mirror_phase"),
    ("geometry_center", "geometry_center"),
    ("Courant=", "courant"),
    ("resolution=", "resolution"),
    ("MPI ranks", "mpi"),
    ("already been stepped", "already_stepped"),
    ("epsilon_offdiag", "epsilon_offdiag"),
    ("mu_diag", "mu"),
    ("H_susceptibilities", "magnetic_dispersion"),
    ("B_conductivity", "b_conductivity"),
    ("magnetic nonlinearity", "magnetic_nonlinearity"),
    ("D_conductivity_offdiag", "conductivity_offdiag"),
    ("differs between components", "anisotropic_conductivity"),
    ("sigma_offdiag", "sigma_offdiag"),
    ("this engine implements the plain Lorentzian", "exotic_susceptibility"),
    ("must be finite and strictly positive", "bad_susceptibility"),
    # Cylindrical runs drive mp.Er / mp.Ep and place PML on mp.R / mp.P; those are
    # follow-on noise from the cylindrical refusal, not separate blockers.
    ("which is not one of mp.Ex/Ey/Ez", "cylindrical_component"),
    ("which is not mp.X, mp.Y or mp.Z", "cylindrical_pml_direction"),
)


def tag_reason(reason: str) -> str:  # Histogram key for one refusal sentence.
    for needle, tag in REASON_TAGS:
        if needle in reason:
            return tag
    return "other:" + reason[:48]


# --- child: build one example's Simulation and ask gpu_compatibility ----------------


CHILD_PREAMBLE = r"""
import json, os, sys, types

os.environ.setdefault("MPLBACKEND", "Agg")


class _StopExample(Exception):
    "Raised in place of the script's first terminal call, to capture the Simulation."


# Scripts whose own argparse interface REQUIRES positional arguments: executed bare
# they die at SystemExit(2) before building a Simulation — a fact about this harness,
# not about the engine, and the reason three dipole scripts sat unassessed through
# the first corpus survey. None of them declares a default for these positionals, so
# the values here are the tutorial's own invocation where one is documented
# (`Near_to_Far_Field_Spectra.md`: "Ex and Ey dipoles at r = 0.1 um" for the
# off-axis case) and the first declared `choices` entry otherwise. The argv used is
# written onto the row as `argv`.
_SCRIPT_ARGV = {
    "dipole_in_vacuum_1D.py": ["x"],
    "dipole_in_vacuum_cyl_on_axis.py": ["x"],
    "dipole_in_vacuum_cyl_off_axis.py": ["x", "0.1"],
}


def _capture(script):
    # Execute one example with MEEP's terminal calls stubbed and hand back its
    # Simulation. Returns (record, sim, restore): the record up to and including
    # `facts`, the captured mp.Simulation (None when the script never built one),
    # and a callable that puts every patched MEEP method back.
    #
    # The survey itself never calls restore -- it only asks gpu_compatibility,
    # which steps nothing -- so this split leaves the survey's behaviour exactly
    # what it was. A caller that means to LIFT and STEP the captured object MUST
    # call it, because `run` and `init_sim` are still stubs that raise until it does.
    import meep as mp

    state = {"constructed": [], "captured": None, "capture_point": None}
    originals = []  # (owner, attribute, value) for restore(), in patch order.

    original_init = mp.Simulation.__init__
    originals.append((mp.Simulation, "__init__", original_init))

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        state["constructed"].append(self)

    mp.Simulation.__init__ = patched_init

    def make_stub(name):
        def stub(self, *args, **kwargs):
            state["captured"] = self
            state["capture_point"] = name
            raise _StopExample(name)
        return stub

    # Every route into MEEP's own stepper or grid build. Capturing at the FIRST of
    # them means monitors added between construction and run are already attached,
    # which is what a user's script actually hands to the engine.
    for name in ("run", "init_sim", "solve_cw", "run_k_points", "_run_until",
                 "get_eigenmode_coefficients", "load_minus_flux"):
        if hasattr(mp.Simulation, name):
            originals.append((mp.Simulation, name, getattr(mp.Simulation, name)))
            setattr(mp.Simulation, name, make_stub(name))

    # set_boundary is the one mutation that is invisible after the fact: it writes
    # a PEC/PMC wall into fields.boundaries, which MEEP exposes only as an opaque
    # pointer. It also forces init_sim, so the stub above already stops the script
    # there — recording the call is how the survey can still report it.
    state["set_boundary"] = []
    original_set_boundary = mp.Simulation.set_boundary
    originals.append((mp.Simulation, "set_boundary", original_set_boundary))

    def patched_set_boundary(self, side, direction, condition):
        state["set_boundary"].append((int(side), int(direction), int(condition)))
        state["captured"] = self
        state["capture_point"] = "set_boundary"
        raise _StopExample("set_boundary")

    mp.Simulation.set_boundary = patched_set_boundary
    try:
        from meep import mpb
        originals.append((mpb.ModeSolver, "init_params", mpb.ModeSolver.init_params))
        mpb.ModeSolver.init_params = make_stub("mpb.init_params")
    except Exception:
        pass

    def restore():
        for owner, attribute, value in reversed(originals):
            setattr(owner, attribute, value)

    record = {"script": os.path.basename(script)}
    # Which interpreter (and therefore which MEEP build) produced this row. The
    # corpus is now surveyed from more than one env — the pristine reference MEEP
    # 1.33.0 and the sigma-reader env that carries the optional packages — and a
    # number that does not name its producer is not a record.
    record["python"] = sys.executable
    record["meep_version"] = str(getattr(mp, "__version__", None))
    # PyMieScatt (the analytic Mie reference two scripts import) still does
    # `from scipy.integrate import trapz`, an alias scipy >= 1.14 removed in favour
    # of the identical `trapezoid`. Restoring the alias here is an exact rename, not
    # a numerical change, and it keeps the fix in the harness instead of pinning the
    # env's scipy back. Applied only when the alias is actually missing.
    try:
        import scipy.integrate as _scipy_integrate
        if not hasattr(_scipy_integrate, "trapz"):
            _scipy_integrate.trapz = _scipy_integrate.trapezoid
            record["scipy_trapz_shim"] = True
    except Exception:
        pass
    argv_extra = list(_SCRIPT_ARGV.get(os.path.basename(script), []))
    if argv_extra:
        record["argv"] = argv_extra
    try:
        import runpy
        sys.argv = [script] + argv_extra
        runpy.run_path(script, run_name="__main__")
        record["outcome"] = "ran_to_end"
    except _StopExample as exc:
        record["outcome"] = "captured"
        record["capture_point"] = str(exc)
    except SystemExit as exc:
        record["outcome"] = "sysexit"
        record["error"] = repr(exc)
    except BaseException as exc:
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]

    record["n_simulations_constructed"] = len(state["constructed"])
    record["set_boundary_calls"] = state.get("set_boundary", [])
    sim = state["captured"]
    if sim is None and state["constructed"]:
        sim = state["constructed"][-1]
        record["capture_point"] = "constructed_only"
    if sim is None:
        record["has_simulation"] = False
        return record, None, restore
    record["has_simulation"] = True
    if record["outcome"] not in ("captured",):
        record["outcome"] = record["outcome"] + "+simulation"

    record["facts"] = _facts(mp, sim)
    return record, sim, restore


def _survey(script, out_path):
    import meep_gpu

    record, sim, _restore = _capture(script)  # The survey steps nothing, so no restore.
    if sim is None:
        with open(out_path, "w") as handle:
            json.dump(record, handle)
        return
    try:
        verdict = meep_gpu.gpu_compatibility(sim)
        record["supported"] = bool(verdict.supported)
        record["reasons"] = list(verdict.reasons)
    except BaseException as exc:
        record["supported"] = None
        record["compat_error"] = f"{type(exc).__name__}: {exc}"[:400]
    with open(out_path, "w") as handle:
        json.dump(record, handle)


def _facts(mp, sim):
    def v3(value):
        try:
            return [float(value.x), float(value.y), float(value.z)]
        except Exception:
            return None
    out = {}
    out["cell_size"] = v3(sim.cell_size)
    out["dimensions_attr"] = getattr(sim, "dimensions", None)
    out["is_cylindrical"] = bool(getattr(sim, "is_cylindrical", False))
    cell = out["cell_size"] or [0, 0, 0]
    zero_axes = sum(1 for value in cell if value == 0.0)
    out["zero_extent_axes"] = zero_axes
    out["effective_dims"] = ("cylindrical" if out["is_cylindrical"]
                             else 3 - zero_axes)
    try:
        out["resolution"] = float(sim.resolution)
    except Exception:
        out["resolution"] = repr(sim.resolution)
    out["courant"] = float(sim.Courant)
    out["k_point"] = v3(sim.k_point) if sim.k_point else None
    out["n_geometry"] = len(getattr(sim, "geometry", ()) or ())
    out["n_sources"] = len(getattr(sim, "sources", ()) or ())
    out["source_kinds"] = sorted({type(s).__name__ for s in (getattr(sim, "sources", ()) or ())})
    out["src_time_kinds"] = sorted({type(getattr(s, "src", None)).__name__
                                    for s in (getattr(sim, "sources", ()) or ())})
    out["boundary_kinds"] = sorted({type(b).__name__
                                    for b in (getattr(sim, "boundary_layers", ()) or ())})
    out["symmetry_kinds"] = sorted({type(s).__name__
                                    for s in (getattr(sim, "symmetries", ()) or ())})
    out["n_dft_objects"] = len(getattr(sim, "dft_objects", ()) or ())
    out["dft_kinds"] = sorted({type(d).__name__ for d in (getattr(sim, "dft_objects", ()) or ())})
    out["has_material_function"] = getattr(sim, "material_function", None) is not None
    out["has_epsilon_func"] = getattr(sim, "epsilon_func", None) is not None
    out["default_material"] = type(getattr(sim, "default_material", None)).__name__
    out["geometry_materials"] = sorted({type(getattr(o, "material", None)).__name__
                                        for o in (getattr(sim, "geometry", ()) or ())})
    return out
"""


def _as_text(stream: bytes | str | None) -> str:  # subprocess hands back bytes without text=True.
    if stream is None:
        return ""
    return stream.decode("utf8", "replace") if isinstance(stream, bytes) else stream


def run_child(script: str, out_path: str) -> None:  # In-process: the --one mode's body.
    namespace: dict = {}
    exec(compile(CHILD_PREAMBLE, "<survey-child>", "exec"), namespace)  # noqa: S102
    namespace["_survey"](script, out_path)


# --- parent: one subprocess per example, partial results as they land ---------------


def scratch_workdir(examples: Path, root: Path) -> Path:
    """A writable cwd with the corpus's data files symlinked in.

    Examples write HDF5, PNG and log files next to themselves; the corpus is a
    read-only checkout as far as this survey is concerned.
    """
    work = root / "workdir"
    work.mkdir(parents=True, exist_ok=True)
    for entry in examples.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            try:
                link.symlink_to(entry)
            except OSError:
                pass
    return work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=os.path.join(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")), "python", "examples"))
    parser.add_argument("--out", default="parity/meep_gpu/results/survey")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--one", default=None, help="child mode: survey a single script")
    parser.add_argument("--out-json", default=None, help="child mode: where to write the record")
    parser.add_argument("--only", default=None, help="comma-separated script names to survey")
    args = parser.parse_args()

    if args.one:
        run_child(args.one, args.out_json)
        return 0

    examples = Path(args.examples)
    scripts = sorted(examples.glob("*.py"))
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        scripts = [path for path in scripts if path.name in wanted]
    # RESOLVED, not as given. Each child runs with cwd=workdir (so an example script
    # finds its own data files), and it is handed --out-json built from this path. A
    # relative --out therefore resolves against the workdir in the child, where it does
    # not exist, and EVERY child dies with FileNotFoundError writing its record — which
    # the parent then reports as an indistinguishable "child_died". The default --out is
    # relative, so the plain documented invocation was the broken one.
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    per_example = out_root / "per_example"
    per_example.mkdir(exist_ok=True)
    jsonl = out_root / "survey.jsonl"
    jsonl.write_text("", encoding="utf-8")
    work = scratch_workdir(examples, out_root)

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    print(f"surveying {len(scripts)} example scripts from {examples}", flush=True)
    started = time.time()
    records = []
    for index, script in enumerate(scripts, start=1):
        record_path = per_example / f"{script.stem}.json"
        if record_path.exists():
            record_path.unlink()
        case_start = time.time()
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--one", str(script), "--out-json", str(record_path),
        ]
        status = "ok"
        stderr_text, returncode = "", None
        try:
            completed = subprocess.run(
                command, cwd=str(work), env=environment, timeout=args.timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
            )
            stderr_text = _as_text(completed.stderr)
            returncode = completed.returncode
        except subprocess.TimeoutExpired as expired:
            status = "timeout"
            stderr_text = _as_text(expired.stderr)
        elapsed = time.time() - case_start
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            # The child's stderr is the ONLY account of why it died, and discarding it
            # (as this did) turns every failure into an indistinguishable "child_died"
            # — including a harness bug that kills all 80 at once, which is exactly what
            # it hid. Keep the tail on the record and the full text beside it.
            record = {"script": script.name, "outcome": status if status != "ok" else "child_died",
                      "has_simulation": False, "returncode": returncode,
                      "stderr_tail": stderr_text[-2000:]}
            if stderr_text:
                (per_example / f"{script.stem}.stderr.txt").write_text(stderr_text, encoding="utf-8")
        record["wall_s"] = round(elapsed, 2)
        records.append(record)
        with jsonl.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        verdict = (
            "LIFTABLE" if record.get("supported") is True
            else f"refused[{len(record.get('reasons', []))}]" if record.get("supported") is False
            else record.get("outcome", "?")
        )
        dims = (record.get("facts") or {}).get("effective_dims", "-")
        print(
            f"  {index:>2}/{len(scripts)} {script.name:<42} {str(dims):>11}D  {verdict:<14} "
            f"({elapsed:5.1f} s)",
            flush=True,
        )
    summarize(records, out_root)
    print(f"survey complete in {time.time() - started:.1f} s -> {out_root}", flush=True)
    return 0


def summarize(records: list[dict], out_root: Path) -> None:  # Coverage table + refusal histogram.
    from collections import Counter

    with_sim = [r for r in records if r.get("has_simulation")]
    liftable = [r for r in with_sim if r.get("supported") is True]
    refused = [r for r in with_sim if r.get("supported") is False]
    tag_counts: Counter = Counter()
    sole_tag_counts: Counter = Counter()
    for record in refused:
        tags = {tag_reason(reason) for reason in record["reasons"]}
        tag_counts.update(tags)
        if len(tags) == 1:
            sole_tag_counts.update(tags)
    summary = {
        "n_scripts": len(records),
        "n_with_simulation": len(with_sim),
        "n_liftable": len(liftable),
        "n_refused": len(refused),
        "liftable_scripts": sorted(r["script"] for r in liftable),
        "reason_frequency": tag_counts.most_common(),
        "sole_reason_frequency": sole_tag_counts.most_common(),
        "dims_histogram": Counter(
            str((r.get("facts") or {}).get("effective_dims")) for r in with_sim
        ).most_common(),
        "no_simulation": sorted(
            (r["script"], r.get("outcome"), r.get("error", "")[:120])
            for r in records if not r.get("has_simulation")
        ),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("", flush=True)
    print(f"scripts surveyed        : {summary['n_scripts']}", flush=True)
    print(f"built an mp.Simulation  : {summary['n_with_simulation']}", flush=True)
    print(f"LIFTABLE                : {summary['n_liftable']}", flush=True)
    print(f"refused (with reasons)  : {summary['n_refused']}", flush=True)
    print("dimensionality          : " + ", ".join(f"{k}D={v}" for k, v in summary["dims_histogram"]), flush=True)
    print("refusal reasons ranked  :", flush=True)
    for tag, count in summary["reason_frequency"]:
        print(f"    {count:>3}  {tag}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
