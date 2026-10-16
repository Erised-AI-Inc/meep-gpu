"""A gate's import record is its own dependency set: the dispatch layer only where declared.

WHAT THIS HOLDS (owner ruling R17, bit_parity_0_10 DESIGN.md section 8). A gate artifact's
import record is every repository module its process held when it wrote the artifact, and the
rebind tools, the family and composition re-cuts, the bit-identity re-cut, the Metal runner's
manifest and the three board builders check that WHOLE record against the checkout. When the
package root imported the driver, the dispatch planners and the MEEP lift eagerly, every
gate's record held them, so a dispatch-only edit voided every kernel's evidence. The root now
binds its names on first use (``meep_gpu/test_lazy_package.py``), and this file keeps every
gate that does not dispatch, route, compose or build a driver free of :data:`NOT_KERNEL`:

* by the IMPORT closure (importing the gate, never running it), measured in a fresh
  interpreter on a sample and read off the source for every gate;
* by the RUN closure, the upper bound ``import_closure.Resolver.run_closure`` computes on what
  a run of the gate can import (its ``main`` and every function it reaches).

A gate that needs the dispatch layer is DECLARED below, by category; an undeclared gate whose
closure reaches it fails, and so does a declared gate whose closure no longer does (the list
cannot go stale). The planted controls put an eager import back -- in the package root and in
a kernel module, read off the source, and in a small package measured in a fresh interpreter
-- and each must fail the check.

The whole-import checks of the tools are not relaxed by any of this; the records shrink.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import import_closure as ic  # noqa: E402

#: Gate and probe scripts whose run closure reaches the dispatch layer or the lift, with why.
#: One of four categories each; everything else is a kernel gate and must reach neither.
DECLARED: Mapping[str, str] = {
    # ROUTE: the dispatch route and licence gates; they step drivers through the planners.
    "gate_dispatch_end_to_end.py": "route",
    "gate_dispatch_fused_route.py": "route",
    "gate_dispatch_metal_route.py": "route",
    "gate_driver_route_fused.py": "route",
    # DISPATCH: read the planners' own tables or decisions (slot owners, the per-table
    # subnormal policy, dispatch reachability) or drive the planner directly.
    "gate_dispatch_absent_components.py": "dispatch",
    "gate_triton_complex_ade.py": "dispatch",
    "gate_triton_complex_no_pml_stored_e.py": "dispatch",
    "gate_triton_offdiag_stencil_welds.py": "dispatch",
    "probe_cuda_hd_lift_autopsy.py": "dispatch",
    "probe_cuda_source_seam_blockage.py": "dispatch",
    "probe_metal_dispatch_dryrun.py": "dispatch",
    # COMPOSITION: compose a table's products in a driver's step and compare it with the
    # reference driver.
    "probe_triton_bfast_composition.py": "composition",
    "probe_triton_complex_composition.py": "composition",
    "probe_triton_conductivity_composition.py": "composition",
    "probe_triton_cylindrical_complex_composition.py": "composition",
    "probe_triton_cylindrical_composition.py": "composition",
    "probe_triton_engine_route.py": "composition",
    "probe_triton_folded_complex_composition.py": "composition",
    "probe_triton_folded_deposit_closure.py": "composition",
    "probe_triton_folded_offdiag_composition.py": "composition",
    "probe_triton_no_pml_constitutive_composition.py": "composition",
    "probe_triton_nonlinear_composition.py": "composition",
    "probe_triton_offdiag_composition.py": "composition",
    "probe_triton_source_seams.py": "composition",
    "probe_triton_special_kz_composition.py": "composition",
    "probe_triton_symmetry_composition.py": "composition",
    # DRIVER: a leg builds an FdtdDriver (the H->D seams' sync and order legs, the fused
    # pairs' composed legs, the stencil welds' driver-order leg, a lift of a MEEP row).
    "gate_complex_beta_fused_hd_pair.py": "driver",
    "gate_cuda_complex_fused_hd_pair.py": "driver",
    "gate_cuda_complex_offdiag_stencil_welds.py": "driver",
    "gate_cuda_conductive_bfast_fused_hd_pair.py": "driver",
    "gate_cuda_cylindrical_fused_hd_pair.py": "driver",
    "gate_cuda_cylindrical_real_fused_hd_pair.py": "driver",
    "gate_cuda_dispersive_offdiag_polarization_pair.py": "driver",
    "gate_cuda_fused_hd_pair.py": "driver",
    "gate_cuda_fused_magnetic_pair.py": "driver",
    "gate_cuda_no_pml_null_constitutive.py": "driver",
    "gate_cuda_offdiag_stencil_welds.py": "driver",
    "gate_metal_beta_complex_fused_hd_pair.py": "driver",
    "gate_metal_beta_real_fused_hd_pair.py": "driver",
    "gate_metal_bfast_fused_hd_pair.py": "driver",
    "gate_metal_complex_conductive_fused_pair.py": "driver",
    "gate_metal_complex_fused_hd_pair.py": "driver",
    "gate_metal_conductive_fused_hd_pair.py": "driver",
    "gate_metal_cylindrical_real_fused_hd_pair.py": "driver",
    "gate_metal_folded_complex_fused_hd_pair.py": "driver",
    "gate_metal_folded_fused_hd_pair.py": "driver",
    "gate_metal_fused_hd_pair.py": "driver",
    "gate_metal_fused_hd_pair_nonlinear.py": "driver",
    "gate_metal_fused_magnetic_pair.py": "driver",
    "gate_special_kz_fused_hd_pair.py": "driver",
    "gate_triton_beta_complex_fused_hd_pair.py": "driver",
    "gate_triton_beta_real_fused_hd_pair.py": "driver",
    "gate_triton_complex_fused_hd_pair.py": "driver",
    "gate_triton_conductive_bfast_fused_hd_pair.py": "driver",
    "gate_triton_cylindrical_fused_hd_pair.py": "driver",
    "gate_triton_cylindrical_real_fused_hd_pair.py": "driver",
    "gate_triton_folded_complex_fused_hd_pair.py": "driver",
    "gate_triton_folded_fused_hd_pair.py": "driver",
    # Its optional corpus-lift leg (not release-gating) lifts a MEEP row through the package
    # root's lazy ``lift_simulation``.
    "gate_triton_folded_offdiag_fused_ade_chain.py": "driver",
    "gate_triton_fused_hd_pair.py": "driver",
    "gate_triton_no_pml.py": "driver",
    "gate_triton_no_pml_constitutive.py": "driver",
    "gate_triton_nonlinear_fused_hd_pair.py": "driver",
    "probe_constitutive_absorber_reach.py": "driver",
    "probe_dispatch_corpus_byteparity.py": "driver",
    "probe_dispatch_policy_gate.py": "driver",
    "probe_dispatch_seam_sweep.py": "driver",
    "probe_ftz_ptx_census.py": "driver",
    "probe_fused_kernel_bit_identity.py": "driver",
    "probe_metal_complex_driver_step.py": "driver",
    "probe_metal_special_kz_driver_step.py": "driver",
    # Lifts a route case twice on the array path (gate_dispatch_metal_route._lift).
    "probe_pml_intermediate_flush.py": "driver",
    "probe_sync_half_step.py": "driver",
    "probe_triton_beta_fused_electric_pair.py": "driver",
    "probe_triton_beta_fused_magnetic_pair.py": "driver",
    "probe_triton_bfast_fused_electric_pair.py": "driver",
    "probe_triton_bfast_fused_magnetic_pair.py": "driver",
    "probe_triton_complex_beta_fused_electric_pair.py": "driver",
    "probe_triton_complex_beta_fused_magnetic_pair.py": "driver",
    "probe_triton_complex_conductive_fused_pair.py": "driver",
    "probe_triton_complex_fused_electric_pair.py": "driver",
    "probe_triton_complex_fused_magnetic_pair.py": "driver",
    "probe_triton_conductive_fused_electric_pair.py": "driver",
    "probe_triton_cylindrical_fused_electric_pair.py": "driver",
    "probe_triton_cylindrical_real_fused_electric_pair.py": "driver",
    "probe_triton_cylindrical_real_fused_magnetic_pair.py": "driver",
    "probe_triton_dispersive_fused_pair.py": "driver",
    "probe_triton_folded_beta_complex_fused_magnetic_pair.py": "driver",
    "probe_triton_folded_beta_complex_fused_pair.py": "driver",
    "probe_triton_folded_beta_fused_electric_pair.py": "driver",
    "probe_triton_folded_beta_fused_magnetic_pair.py": "driver",
    "probe_triton_folded_complex_fused_magnetic_pair.py": "driver",
    "probe_triton_folded_complex_fused_pair.py": "driver",
    "probe_triton_folded_dispersive_fused_pair.py": "driver",
    "probe_triton_folded_fused_magnetic_pair.py": "driver",
    "probe_triton_folded_fused_pair.py": "driver",
    "probe_triton_fused_ade_state.py": "driver",
    "probe_triton_fused_dispersive_chain.py": "driver",
    "probe_triton_no_pml_fused_electric_pair.py": "driver",
    "probe_triton_nonlinear_fused_magnetic_pair.py": "driver",
}
CATEGORIES = ("route", "dispatch", "composition", "driver")

#: Measured in a fresh interpreter: kernel gates of each table, the ADE gate (whose
#: interpreter import used to bring the driver in), and two declared gates that hold the
#: planner at import. Each imports with NumPy alone, on every host the CI view runs on.
MEASURED_SAMPLE = (
    "parity/meep_gpu/gate_metal_pml.py",
    "parity/meep_gpu/gate_metal_no_pml_curl.py",
    "parity/meep_gpu/gate_cuda_offdiag.py",
    "parity/meep_gpu/gate_cuda_ade.py",
    "parity/meep_gpu/gate_triton_complex.py",
    "parity/meep_gpu/gate_triton_no_pml_conductive.py",
    "parity/meep_gpu/gate_metal_fused_hd_pair.py",
    "parity/meep_gpu/gate_triton_no_pml_constitutive.py",
)


def _offenders(rows: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Undeclared scripts whose closure reaches :data:`ic.NOT_KERNEL`, with the witnesses."""
    return {script: row["run_witness"] for script, row in rows.items()
            if row["run_hits"] and Path(script).name not in DECLARED}


@pytest.fixture(scope="module")
def resolver():
    return ic.Resolver(ic.REPO_ROOT)


@pytest.fixture(scope="module")
def rows(resolver):
    started = time.time()
    out = {}
    scripts = ic.gate_scripts(ic.REPO_ROOT)
    for index, script in enumerate(scripts, 1):
        imported = resolver.import_closure(script)
        ran, why = resolver.run_closure_with_witness(script)
        out[script] = {"import_hits": sorted(set(imported) & set(ic.NOT_KERNEL)),
                       "run_hits": sorted(set(ran) & set(ic.NOT_KERNEL)),
                       "run_witness": {h: why[h] for h in sorted(set(ran) & set(ic.NOT_KERNEL))},
                       "import_closure": imported}
        if index % 50 == 0 or index == len(scripts):
            print(f"closure {index}/{len(scripts)} ({time.time() - started:.1f} s)", flush=True)
    return out


def test_the_declared_list_names_real_scripts_in_the_four_categories():
    names = {Path(s).name for s in ic.gate_scripts(ic.REPO_ROOT)}
    assert sorted(set(DECLARED) - names) == []
    assert sorted({c for c in DECLARED.values()} - set(CATEGORIES)) == []


def test_no_undeclared_gate_reaches_the_dispatch_layer(rows):
    offenders = _offenders(rows)
    assert offenders == {}, (
        "gate scripts whose import record can hold the dispatch layer or the lift, and are "
        "not declared (script -> {module: (file, function) that imports it}):\n"
        + "\n".join(f"{s}: {w}" for s, w in sorted(offenders.items())))


def test_every_declared_gate_still_reaches_it(rows):
    by_name = {Path(s).name: row for s, row in rows.items()}
    stale = sorted(name for name in DECLARED if not by_name[name]["run_hits"])
    assert stale == [], f"declared but no longer reaching the dispatch layer: {stale}"


def test_the_import_closure_is_inside_the_run_closure(rows, resolver):
    for script in MEASURED_SAMPLE:
        assert rows[script]["import_closure"] <= resolver.run_closure(script), script


def test_the_package_function_level_edges_into_the_dispatch_layer_are_the_listed_ones(resolver):
    """A NEW lazy import of the planner inside the package is reviewed, never inherited."""
    assert ic.package_lazy_edges(resolver) == sorted(ic.PACKAGE_LAZY_EDGES)


@pytest.mark.parametrize("script", MEASURED_SAMPLE)
def test_the_measured_import_closure_agrees_with_the_source(script, resolver):
    """Importing the gate in a fresh interpreter loads what the source says, and no more.

    The measured closure is INSIDE the static one: the static one also follows imports a
    host without CuPy or Triton skips (a guarded ``import triton`` above them), so it is the
    closure on the fullest host. The dispatch-layer hits are the same in both.
    """
    started = time.time()
    measured = ic.measured_import_closure(script)
    print(f"measured {script}: {len(measured['files'])} files ({time.time() - started:.1f} s)",
          flush=True)
    assert measured["error"] is None, measured["error"]
    files = set(measured["files"])
    static = resolver.import_closure(script)
    assert files <= static, sorted(files - static)
    assert sorted(files & set(ic.NOT_KERNEL)) == sorted(static & set(ic.NOT_KERNEL))
    if Path(script).name not in DECLARED:
        assert files & set(ic.NOT_KERNEL) == set(), sorted(files & set(ic.NOT_KERNEL))


# --------------------------------------------------------------------------------- controls --

PLANTED_KERNEL_GATES = ("parity/meep_gpu/gate_metal_pml.py", "parity/meep_gpu/gate_cuda_offdiag.py",
                        "parity/meep_gpu/gate_triton_complex.py")


def _planted(relative: str, line: str) -> ic.Resolver:
    text = (ic.REPO_ROOT / relative).read_text(encoding="utf-8") + "\n" + line + "\n"
    return ic.Resolver(ic.REPO_ROOT, overrides={relative: text})


@pytest.mark.parametrize("script", PLANTED_KERNEL_GATES)
def test_an_eager_import_put_back_in_the_package_root_fails_the_check(script, resolver):
    planted = _planted("meep_gpu/__init__.py", "from .driver import FdtdDriver")
    assert not set(resolver.import_closure(script)) & set(ic.NOT_KERNEL)
    hits = sorted(set(planted.import_closure(script)) & set(ic.NOT_KERNEL))
    assert {"meep_gpu/driver.py", "meep_gpu/fastpath.py"} <= set(hits)
    assert _offenders({script: {"run_hits": hits, "run_witness": {}}}), (
        f"{script}: a planted eager import was not caught")


def test_an_eager_import_put_in_a_kernel_module_fails_the_check():
    script = "parity/meep_gpu/gate_metal_pml.py"
    planted = _planted("meep_gpu/metal_kernels/launch.py", "from .. import fastpath")
    hits = sorted(set(planted.import_closure(script)) & set(ic.NOT_KERNEL))
    assert "meep_gpu/fastpath.py" in hits
    assert _offenders({script: {"run_hits": hits, "run_witness": {}}})


def test_a_planted_eager_import_is_measured_in_a_fresh_interpreter(tmp_path):
    """The measuring child, end to end, on a small package: lazy root clean, eager root not."""
    def package(root: Path, eager: bool) -> None:
        (root / "meep_gpu").mkdir(parents=True)
        (root / "parity" / "meep_gpu").mkdir(parents=True)
        root_text = (ic.REPO_ROOT / "meep_gpu" / "__init__.py").read_text(encoding="utf-8")
        if eager:
            root_text += "\nfrom .driver import FdtdDriver\n"
        (root / "meep_gpu" / "__init__.py").write_text(root_text, encoding="utf-8")
        (root / "meep_gpu" / "driver.py").write_text(
            "from . import fastpath\n\n\nclass FdtdDriver:\n    pass\n", encoding="utf-8")
        (root / "meep_gpu" / "fastpath.py").write_text("", encoding="utf-8")
        (root / "meep_gpu" / "stepping.py").write_text("", encoding="utf-8")
        (root / "parity" / "meep_gpu" / "gate_kernel.py").write_text(
            "from meep_gpu import stepping\n", encoding="utf-8")

    seen = {}
    for eager in (False, True):
        root = tmp_path / ("eager" if eager else "lazy")
        package(root, eager)
        measured = ic.measured_import_closure("parity/meep_gpu/gate_kernel.py", root=root)
        assert measured["error"] is None, measured["error"]
        seen[eager] = sorted(set(measured["files"]) & set(ic.NOT_KERNEL))
    assert seen == {False: [], True: ["meep_gpu/driver.py", "meep_gpu/fastpath.py"]}


# A name the package root binds on first use (``meep_gpu._LAZY``) imports the submodule that
# defines it, at import (``from meep_gpu import NAME`` runs the root's ``__getattr__``) and when
# a reached function names it through the package (``meep_gpu.NAME(...)``). The resolver
# follows both, so a gate that lifts through the root carries the lift in its closure.
LAZY_PROBE = "parity/meep_gpu/gate_lazy_name_probe.py"


def _planted_gate(text: str) -> ic.Resolver:
    return ic.Resolver(ic.REPO_ROOT, overrides={LAZY_PROBE: text})


def test_the_package_roots_lazy_names_resolve_to_their_defining_modules(resolver):
    lazy = resolver.lazy_names("meep_gpu/__init__.py")
    assert lazy["lift_simulation"] == ("meep_gpu/from_meep.py", "lift_simulation")
    assert lazy["FdtdDriver"] == ("meep_gpu/driver.py", "FdtdDriver")
    assert resolver.lazy_names("meep_gpu/from_meep.py") == {}, "only a package __init__ has lazy names"


def test_a_from_import_of_a_lazy_name_loads_its_module_at_import():
    planted = _planted_gate("from meep_gpu import lift_simulation\n")
    assert "meep_gpu/from_meep.py" in planted.import_closure(LAZY_PROBE)
    clean = _planted_gate("from meep_gpu import stepping\n")
    assert not set(clean.import_closure(LAZY_PROBE)) & set(ic.NOT_KERNEL)


def test_a_lazy_name_called_through_the_package_is_in_the_run_closure():
    planted = _planted_gate("import meep_gpu\n\n\ndef main():\n"
                            "    return meep_gpu.lift_simulation(None)\n")
    assert not set(planted.import_closure(LAZY_PROBE)) & set(ic.NOT_KERNEL)
    ran = planted.run_closure(LAZY_PROBE)
    assert {"meep_gpu/from_meep.py", "meep_gpu/driver.py"} <= set(ran), sorted(ran)
    unreached = _planted_gate("import meep_gpu\n\n\ndef never_called():\n"
                              "    return meep_gpu.lift_simulation(None)\n")
    assert "meep_gpu/from_meep.py" not in unreached.run_closure(LAZY_PROBE)
