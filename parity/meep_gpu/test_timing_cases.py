"""The timing-cases registry: sizes, the one-builder rule, the GPU specs, the seams.

Tests that build a simulation run it in a child process, as the root conftest's own
single-precision probe does: this test process may hold PyTorch, and MEEP is not
imported beside it here.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_cases as tc  # noqa: E402

#: The 2026-09-28 digest of record of ``pml_3d`` at resolution 12: the A6000 host's
#: ``digests/pml_3d_res12.sha256`` and this laptop's ``harness_lift_digest_res12.json``
#: agree on it. It pins the ported digest function and the builder together.
PML_3D_RES12_DIGEST = "01b48b81433dc893662c826d1b5ec3b78c5d6ec21b1190405b82179940a7ca4e"

requires_meep = pytest.mark.requires_resource("meep")
requires_single = pytest.mark.requires_resource("single_precision_meep")


@functools.lru_cache(maxsize=1)
def meep_precision() -> str:
    if importlib.util.find_spec("meep") is None:
        return "absent"
    done = subprocess.run([sys.executable, "-c", "import meep; print('SINGLE' if "
                           "meep.is_single_precision() else 'DOUBLE')"],
                          capture_output=True, text=True, timeout=300)
    return "single" if "SINGLE" in done.stdout.split() else "double"


def need_meep(single: bool = False) -> None:
    precision = meep_precision()
    if precision == "absent":
        pytest.skip("[requires_resource][meep] MEEP is not installed")
    if single and precision != "single":
        pytest.skip("[requires_resource][single_precision_meep] this MEEP is a "
                    "double-precision build")


def child(code: str, timeout: float = 300.0) -> dict:
    """Run ``code`` in a fresh interpreter; its last stdout line is JSON."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO, HERE]),
               MEEP_GPU_DISPATCH="0", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=HERE)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


def test_importing_the_registry_loads_no_heavy_module():
    code = ("import sys, json; sys.path.insert(0, %r); import timing_cases; "
            "print(json.dumps(sorted(n for n in ('meep', 'meep_gpu', 'cupy', 'triton', "
            "'torch', 'numpy') if n in sys.modules)))" % HERE)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize("name, res, cells", [
    ("pml_3d", 12, 110_592), ("pml_3d", 48, 7_077_888),
    ("thin_pml_3d", 6, 110_592), ("thin_pml_3d", 10, 512_000),
    ("thin_pml_3d_hz", 16, 2_097_152), ("thin_pml_3d_diagonal", 28, 11_239_424),
    ("pml_3d_lorentz", 20, 512_000), ("pml_3d_monitors", 32, 2_097_152),
    ("pml_3d_hz", 48, 7_077_888),
])
def test_cells_and_resolution_round_trip(name, res, cells):
    assert tc.cells(name, res) == cells
    assert tc.res_for_cells(name, cells) == res


def test_every_2026_09_28_size_maps_to_half_the_resolution_on_the_thin_cases():
    for res in (12, 16, 20, 26, 32, 40, 48, 56):
        assert tc.res_for_cells("thin_pml_3d", tc.cells("pml_3d", res)) == res // 2


def test_an_unreachable_size_is_refused_by_name():
    with pytest.raises(SystemExit, match="no integer resolution gives pml_3d exactly"):
        tc.res_for_cells("pml_3d", 500_000)
    with pytest.raises(SystemExit, match="is not a timing case"):
        tc.cells("no_such_case", 4)


def test_declared_pml_fractions():
    assert tc.declared("pml_3d").pml_fraction_nominal == pytest.approx(0.784)
    for name, case in tc.NEW_CASES.items():
        expected = 0.330078 if name.startswith("thin_") else 0.784
        assert case.pml_fraction_nominal == pytest.approx(expected, abs=1e-6), name


def test_the_extended_ladder_cases_hold_the_sizes_of_record():
    """Dispersive, monitor-heavy, source on a fused seam: each on the pml_3d cell, so
    resolutions 20 / 32 / 48 hold the 1.0 ladder's 512,000 / 2,097,152 / 7,077,888."""
    assert tc.EXTENDED_LADDER_CASES == ("pml_3d_lorentz", "pml_3d_monitors", "pml_3d_hz")
    for name in tc.EXTENDED_LADDER_CASES:
        case = tc.NEW_CASES[name]
        assert case.cell == (4.0, 4.0, 4.0) and case.pml == 0.8, name
        assert case.base in ("pml_3d", "pml_3d_diagonal"), name
        assert [tc.res_for_cells(name, n) for n in (512_000, 2_097_152, 7_077_888)] == [
            20, 32, 48], name
    assert tc.NEW_CASES["pml_3d_lorentz"].dispersive is True
    assert tc.NEW_CASES["pml_3d_lorentz"].expected_repair is None
    assert tc.NEW_CASES["pml_3d_monitors"].dft_objects == 8
    assert tc.NEW_CASES["pml_3d_monitors"].expected_repair is None
    assert tc.NEW_CASES["pml_3d_hz"].expected_repair == "B"
    assert tc.expected_repair("pml_3d_hz", "triton") == "B"


def test_a_table_override_changes_one_tables_spec_and_bracket(monkeypatch):
    """A measured per-table entry: its spec fields are copied, its ``expected_repair``
    answers for that table's rows only, and neither it nor ``measured`` enters the spec."""
    import gate_dispatch_fused_route as route  # noqa: PLC0415
    monkeypatch.setitem(tc.TABLE_OVERRIDES, "cuda", {"pml_3d_lorentz": {
        "pairs": 2, "expected_repair": "D", "measured": "a stand-in smoke row"}})
    assert tc.expected_repair("pml_3d_lorentz", "cuda") == "D"
    assert tc.expected_repair("pml_3d_lorentz", "triton") is None
    assert tc.expected_repair("pml_3d_lorentz") is None
    spec = tc.drive_spec("cuda", "pml_3d_lorentz", route.DRIVE_CUDA)
    assert spec["pairs"] == 2
    assert "expected_repair" not in spec and "measured" not in spec
    # A case built on a route-gate builder records that builder's source digest too.
    identity = tc.case_identity("pml_3d_hz")
    assert identity["base"] == "pml_3d" and identity["base_builder_source_sha256"]


def test_resolve_returns_the_route_gates_own_builder_for_a_builtin_case():
    builder, facts = tc.resolve("pml_3d", HERE)
    import gate_dispatch_fused_route as route  # noqa: PLC0415
    assert builder is route.CASES["pml_3d"]
    assert facts["builder_is_end_to_end_case"] is True
    new, new_facts = tc.resolve("thin_pml_3d_hz", HERE)
    assert new is tc.case_thin_pml_3d_hz
    assert new_facts["template"] == "pml_3d" and new_facts["expected_repair"] == "B"
    seam, seam_facts = tc.resolve("pml_3d_hz", HERE)
    assert seam is tc.case_pml_3d_hz and seam_facts["base"] == "pml_3d"
    with pytest.raises(SystemExit, match="neither a case"):
        tc.resolve("magnetic_seam_3d", HERE)


def test_drive_spec_is_a_deep_copy_and_leaves_the_tables_untouched():
    import gate_dispatch_fused_route as route  # noqa: PLC0415
    before = (copy.deepcopy(route.DRIVE), copy.deepcopy(route.DRIVE_CUDA))
    for table, rows in (("triton", route.DRIVE), ("cuda", route.DRIVE_CUDA)):
        for name, case in tc.NEW_CASES.items():
            assert case.template in rows, (table, case.template)
            spec = tc.drive_spec(table, name, rows)
            assert spec["arms"] == rows[case.template]["arms"]
            assert spec["arms"] is not rows[case.template]["arms"]
            spec["arms"].append("mutated")
            assert "timing case" in spec["why"]
    assert tc.drive_spec("cuda", "thin_pml_3d_diagonal", route.DRIVE_CUDA)[
        "unfused_baseline"] == "cuda"
    assert (route.DRIVE, route.DRIVE_CUDA) == before
    with pytest.raises(SystemExit, match="a built-in case"):
        tc.drive_spec("triton", "pml_3d", route.DRIVE)


def test_templates_exist_in_the_metal_table():
    """Read from the gate's source, not imported: importing it loads the Metal launcher,
    and this check needs only the keys of its DRIVE literal."""
    import ast  # noqa: PLC0415
    with open(os.path.join(HERE, "gate_dispatch_metal_route.py"), "r",
              encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    keys = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DRIVE":
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert keys, "gate_dispatch_metal_route.DRIVE is no longer a dict literal"
    for name, case in tc.NEW_CASES.items():
        assert case.template in keys, (name, case.template)


@requires_single
def test_new_cases_build_and_initialise_as_declared():
    need_meep(single=True)
    code = """
import json
import meep as mp
mp.verbosity(0)
import timing_cases as tc
import bench_meep_identical_case as bench
out = {}
for name in tc.NEW_CASES:
    builder, _ = tc.resolve(name)
    sim, monitors, until = builder(mp, 4)
    sim.init_sim()
    digest = bench.geometry_digest(mp, sim, sample_epsilon=False)
    out[name] = {"cells": digest["facts"]["cells"], "nominal": digest["facts"]["cells_nominal"],
                 "dft": len(sim.dft_objects), "monitors": len(monitors), "until": until,
                 "pml": digest["facts"]["pml_cell_fraction_nominal"],
                 "source": digest["facts"]["sources"][0]["component"],
                 "susceptibilities": sum(len(g["material"].get("E_susceptibilities") or [])
                                         for g in digest["facts"]["geometry"]),
                 "structural": len(digest["structural_digest"]) == 64}
print(json.dumps(out))
"""
    found = child(code)
    for name, case in tc.NEW_CASES.items():
        facts = found[name]
        assert facts["cells"] == facts["nominal"] == tc.cells(name, 4), name
        assert facts["dft"] >= 1 and facts["monitors"] >= 1, name
        assert facts["pml"] == pytest.approx(case.pml_fraction_nominal, abs=1e-6), name
        assert facts["source"].lower() == case.source_component.lower(), name
        assert facts["dft"] == case.dft_objects, name
        assert (facts["susceptibilities"] > 0) is case.dispersive, name
        assert facts["structural"], name


@requires_single
def test_ported_digest_of_pml_3d_at_res_12_is_the_2026_09_28_digest():
    need_meep(single=True)
    code = """
import json
import meep as mp
mp.verbosity(0)
import timing_cases as tc
import bench_meep_identical_case as bench
builder, _ = tc.resolve("pml_3d")
sim, _monitors, _until = builder(mp, 12)
sim.init_sim()
print(json.dumps({"digest": bench.geometry_digest(mp, sim)["digest"]}))
"""
    assert child(code)["digest"] == PML_3D_RES12_DIGEST


@requires_meep
def test_seams_and_off_diagonal_flag_on_a_numpy_lift():
    """What each case declares, read off the package's own reference lift: the grid's
    off-diagonal flag (``fastpath._run_shape``) and the seam the source is injected in
    (``deposit_repair.in_seam_sources``) -- the facts the route gates cite as measured
    on NumPy lifts."""
    need_meep()
    code = """
import json
import meep as mp
mp.verbosity(0)
import meep_gpu
from meep_gpu import deposit_repair, fastpath
import timing_cases as tc
out = {}
for name in list(tc.NEW_CASES) + ["pml_3d"]:
    builder, _ = tc.resolve(name)
    sim, _m, _u = builder(mp, 4)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
    try:
        shape = fastpath._run_shape(driver.fields, driver.pml, driver.grid)
        from meep_gpu.triton_kernels import launch as triton_launch
        plan = triton_launch.plan_step(driver.fields, driver.pml, fuse=True,
                                       sources=driver._sources)
        # A FUSED PAIR THE TRITON TABLE WOULD ADMIT ON A CUPY HOST: on this NumPy lift
        # its only refusals name the array module (the route gates' own reading).
        admitted = sorted(key for key, why in plan.reasons.items()
                          if key.startswith("fused_pair") and why
                          and all("array module" in str(w) for w in why))
        out[name] = {"off_diagonal": shape.get("off_diagonal_epsilon"),
                     "B": len(deposit_repair.in_seam_sources(driver._sources, "B")),
                     "D": len(deposit_repair.in_seam_sources(driver._sources, "D")),
                     "monitors": len(driver._flux_monitors) + len(driver._dft_monitors),
                     "dft_objects": len(sim.dft_objects),
                     "triton_pairs": admitted}
    finally:
        driver.close()
print(json.dumps(out))
"""
    found = child(code, timeout=600)
    for name in list(tc.NEW_CASES) + ["pml_3d"]:
        case = tc.declared(name)
        facts = found[name]
        assert facts["off_diagonal"] is case.off_diagonal_epsilon, (name, facts)
        seam = "B" if case.source_component.startswith("H") else "D"
        assert (facts["B"], facts["D"]) == ((1, 0) if seam == "B" else (0, 1)), (name, facts)
        if case.expected_repair is not None:
            assert case.expected_repair == seam, name
        # The lift keeps one monitor per MEEP DFT object (digest_harness_lift's
        # driver_monitor_count_equals_meep). The fused pairs the Triton predicates admit
        # here are recorded, not pinned: a NumPy lift reads predicate admission only, and
        # the release envelope decides what composes on a device (pml_3d_hz admits the
        # off-diagonal electric weld here and composes fused pair B alone on both NVIDIA
        # tables: the 2026-10-17 smoke), which TABLE_OVERRIDES and the preflight hold.
        assert facts["monitors"] == facts["dft_objects"], (name, facts)
        assert "fused_pair_B" in facts["triton_pairs"], (name, facts["triton_pairs"])


def test_a_name_the_metal_gate_rebuilds_needs_its_table():
    root = tc.find_harness_root(HERE)
    overridden = tc.metal_overrides(root)
    assert "complex_3d" in overridden and "pml_3d" not in overridden
    with pytest.raises(SystemExit, match="assigns 'complex_3d' a builder of its own"):
        tc.resolve("complex_3d", root)
    _builder, facts = tc.resolve("pml_3d", root, "metal")
    assert facts["source"] == "gate_dispatch_fused_route.CASES" and facts["table"] == "metal"
    with pytest.raises(SystemExit, match="table 'opencl'"):
        tc.resolve("pml_3d", root, "opencl")
