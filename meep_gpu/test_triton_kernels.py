"""Tests for the Triton real-field PML curl track.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware
— bit-identity, the PTX census, throughput — lives in the shared gate
(``parity/meep_gpu/probe_fused_kernel_bit_identity.py --track triton``); the D/E
pair's dedicated exact-state and mutation product lives in
``parity/meep_gpu/gate_triton_fused_electric.py``. A byte comparison against the
array path is only meaningful on the device that runs it.

What is pinned here instead is everything that decides whether the kernel is ever
ALLOWED to run: the coverage predicate (with the two mutations that prove it is
load-bearing), the term/coefficient tables the kernel hard-codes, and the rule
that a missing optional dependency must not break the engine.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import pathlib
import sys

import pytest

from meep_gpu import stepping
from meep_gpu.device_identity import weld_survives_edit
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.coverage import COVERED_BOUNDARIES, pml_curl_coverage

PACKAGE_DIR = pathlib.Path(launch_module.__file__).parent


def build(cell_size=(0.8, 0.8, 0.8), **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------

def test_the_package_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """A missing optional dependency must not break the engine — or this package.

    ``triton`` is an optional dependency OF an optional package. Importing
    ``meep_gpu.triton_kernels``, asking it whether Triton is available, and asking
    it whether a configuration is covered must all work on a machine that has
    never heard of Triton; only building or launching a plan may fail.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels", raising=False)
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.kernels", raising=False)

    package = importlib.import_module("meep_gpu.triton_kernels")
    assert package.triton_available() is False
    fields, pml = build()
    assert package.explain(fields, pml).reasons  # answers, rather than raising
    with pytest.raises(ImportError, match="triton"):
        package.require_triton()


def code_of(path: pathlib.Path) -> str:
    """A module's source with comments and docstrings removed.

    The prose in this package NAMES the modules it must not touch (that is how a
    reader learns dispatch is deferred), so an ownership check that greps the raw
    file would fire on its own documentation. Stripping to executable text is what
    makes the check about behaviour.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_dependency_between_dispatch_and_this_package_runs_ONE_WAY():
    """File ownership, enforced rather than agreed — and the direction is the point.

    ``fastpath.py`` now dispatches this track, so it names it; that is the round
    that wired the driver seam. What must NOT happen is the reverse. Nothing in
    this package may know a dispatcher exists, for two reasons that both bite:

    * every file here is sha256-welded to the device gate that certified it
      (``host_sha256``), so a symbol added for a dispatcher's convenience is a
      device-gate event, and a dispatcher that needed one would be pushing its
      own churn into a certified record;
    * a predicate that could see who is asking is a predicate that could answer
      differently for the driver than for the gate.

    So the dispatch vocabulary — the runnable slot names, the null and fused arm
    labels, the warm helper — lives in ``fastpath.py``, and its agreement with
    what this package actually writes is MEASURED against a real composition in
    ``test_dispatch_contract`` rather than asserted by a shared symbol.

    ``stepping.py`` keeps the stricter rule unchanged: it is the ORACLE every gate
    compares against, and it must stay free of the track that is measured against
    it. The hand-CUDA track stays walled off from both.

    THE SEAM NOW COMPOSES ALL THREE TABLES AND THE RULE IS NO LONGER "WHICH ONE IS
    NAMED". ``fastpath.py`` composes the Triton table here, the hand-CUDA table
    through the one function-local import measured below, and the Metal table
    through the sibling ``meep_gpu/metal_dispatch.py``. So "``cuda_kernels`` is not
    named in ``fastpath.py``" stopped being the boundary on the day Phase 2 landed,
    and what replaced it is a STRUCTURAL clause rather than a weaker one: the import
    must be function-local, and it must sit inside ``_decide`` — below the backend
    rung, which is what keeps a NumPy step off a package that needs a device.
    ``test_package_boundary`` measures the runtime half of the same rule.
    """
    for path in PACKAGE_DIR.glob("*.py"):
        code = code_of(path)
        assert "fastpath" not in code, f"{path.name} references fastpath"
        assert "cuda_kernels" not in code, f"{path.name} references the hand-CUDA track"
    engine = pathlib.Path(launch_module.__file__).parents[1]
    assert "triton" not in (engine / "stepping.py").read_text(encoding="utf-8").lower(), (
        "stepping.py is the oracle this track is measured against and must stay "
        "free of it")
    _assert_the_cuda_import_is_function_local_and_below_the_backend_rung(engine)


def _assert_the_cuda_import_is_function_local_and_below_the_backend_rung(engine):
    """Every ``cuda_kernels`` import in ``fastpath.py``, measured off the parse tree.

    TWO CLAUSES, AND THE SECOND IS THE ONE THAT MATTERS AT RUNTIME:

    * MODULE LEVEL IS FORBIDDEN. ``cuda_kernels`` reaches modules that import cupy
      at module scope, so a module-level import here is an ImportError on every
      NumPy host and a needless NVRTC pull-in on the rest.
    * IT MUST BE INSIDE ``_decide``. That is where the backend rung lives; an import
      in any other function could be reached from a NumPy step, which is exactly the
      pull-in ``test_package_boundary`` measures the consequence of.
    """
    import ast as _ast

    tree = _ast.parse((engine / "fastpath.py").read_text(encoding="utf-8"))
    inside_decide = set()
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "_decide":
            inside_decide = {id(sub) for sub in _ast.walk(node)}
    found = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and "cuda_kernels" in (node.module or ""):
            found.append(node)
        elif isinstance(node, _ast.Import) and any(
                "cuda_kernels" in alias.name for alias in node.names):
            found.append(node)
    assert found, ("fastpath.py composes the hand-CUDA table and must import it; "
                   "an absent import means the second NVIDIA table is unreachable")
    for node in found:
        assert node not in tree.body, (
            f"fastpath.py:{node.lineno} imports cuda_kernels at MODULE level")
        assert id(node) in inside_decide, (
            f"fastpath.py:{node.lineno} imports cuda_kernels outside _decide, so a "
            "step that never reached the backend rung could pull in the package")


# ---------------------------------------------------------------------------
# The contraction guard is one constant, not a per-launch habit
# ---------------------------------------------------------------------------

#: The one spelling any launch site may use. Every ``run`` in the package must
#: pass the shared constant (with the gate's guard override), never a literal.
GUARD_SPELLING = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"


def test_every_launch_site_passes_the_shared_guard_constant():
    """``enable_fp_fusion=False`` is a per-CALL keyword, and that is the hazard.

    The hand-CUDA track pins ``--fmad=false`` once, in ``_COMPILE_OPTIONS``, and
    every RawKernel inherits it. Triton's equivalent has to be repeated at every
    launch site, and a site that omits it compiles a kernel that is bit-wrong but
    numerically plausible — one that passes a carelessly written curl gate outright
    at dtdx = 0.5.

    The central ``launch.py`` has five launch sites now that both cross-sub-step
    fused pairs have landed, and sibling modules add more specialized sites. Merely
    counting the central file is therefore not the whole check. What holds instead
    is stronger: EVERY occurrence of the keyword in this directory is the shared
    constant, so a reviewer never has to notice a literal, and a later kernel added
    without the guard fails here rather than in the gate.
    """
    sites = {path.name: code_of(path).count("enable_fp_fusion=")
             for path in PACKAGE_DIR.glob("*.py")}
    assert sites["launch.py"] == 5, sites
    for name in ("__init__.py", "coverage.py", "kernels.py"):
        assert sites[name] == 0, sites
    # EVERY occurrence in the package is the shared constant, counted over the
    # whole directory rather than against a fixed file list: a module added later
    # (a folded-grid kernel, a dispersive one) must be held to the same rule, and
    # a fixed list would have failed on its arrival instead of checking it.
    #
    # TWO SITES ARE EXEMPT, AND THE EXEMPTION IS MEASURED, NOT ASSERTED.
    #
    # The shared spelling needs ``ENABLE_FP_FUSION`` in scope, which means
    # importing kernels.py — and that import RAISES ModuleNotFoundError on a
    # host without Triton (measured 2026-08-17 on arm64: `from
    # meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION` fails). Every
    # conforming module imports it unconditionally at the top of ``run``, which
    # is fine for them because their ``run`` cannot execute without Triton
    # anyway. These two families are different: they are exercised through the
    # explicit-guard route ON THIS HOST — test_triton_complex_no_pml_stored_e.py
    # calls ``plan.run(guard=False)`` at :284 and :302, and those tests pass
    # here — so they bind the guard through a local under ``if guard is None``
    # and never touch kernels.py on the explicit route.
    #
    # The invariant's PURPOSE is preserved: no site may spell a literal. What is
    # allowed is one alternative form, at named modules, whose own structure is
    # checked below — so a third module cannot quietly adopt it, and neither of
    # these two can quietly replace the deferral with a literal.
    DEFERRED_GUARD_MODULES = {
        "complex_ade.py": "explicit-guard route must not import kernels.py",
        "complex_no_pml_stored_e.py": "same; run(guard=False) is exercised here",
    }
    # No trailing comma: code_of() round-trips through ast.unparse, which
    # normalises call punctuation away. Matching the raw source text here would
    # silently count zero and turn this exemption into a rubber stamp.
    DEFERRED_FORM = "enable_fp_fusion=fusion"
    DEFERRED_BIND = "fusion = ENABLE_FP_FUSION"

    shared = sum(code_of(path).count(GUARD_SPELLING)
                 for path in PACKAGE_DIR.glob("*.py"))
    deferred = 0
    for name in DEFERRED_GUARD_MODULES:
        source = code_of(PACKAGE_DIR / name)
        count = source.count(DEFERRED_FORM)
        assert count, f"{name} is exempted but no longer uses the deferred form"
        # The exemption buys the deferral, not a literal: the local must still be
        # bound FROM the shared constant, under the guard-is-None branch.
        assert DEFERRED_BIND in source, (
            f"{name} uses the deferred form without binding it from "
            f"ENABLE_FP_FUSION — that is a literal wearing the exemption")
        assert "bool(guard)" in source, f"{name} drops the explicit-guard branch"
        deferred += count

    assert shared + deferred == sum(sites.values()), (
        "a launch site spells the guard differently from the shared constant "
        f"(shared={shared}, deferred={deferred}, total={sum(sites.values())})")
    # Read the definition off the source rather than importing it, so this holds
    # on a machine with no Triton — where importing kernels.py is impossible.
    assert "\nENABLE_FP_FUSION = False\n" in (PACKAGE_DIR / "kernels.py").read_text(
        encoding="utf-8")


def test_the_guard_is_a_launch_keyword_and_never_an_options_dict():
    """MEASURED asymmetry: the two routes fail differently and only one fails loudly.

    Passed as a JIT launch keyword, an unrecognised name RAISES. Passed inside a
    ``options={...}`` dict, an unrecognised name is SILENTLY DROPPED and the kernel
    compiles contracted — bit-wrong, no error, no signal. So the options route is
    banned by test rather than by convention, and this is the test.
    """
    for path in PACKAGE_DIR.glob("*.py"):
        code = code_of(path)
        assert "options=" not in code, f"{path.name} builds an options dict"
    assert "enable_fp_fusion=" in code_of(PACKAGE_DIR / "launch.py")


# ---------------------------------------------------------------------------
# The version guard — the whole of it runs here, with no GPU and no Triton
# ---------------------------------------------------------------------------

FINGERPRINTS = PACKAGE_DIR / "fingerprints.json"


def fingerprints() -> dict:
    import json

    return json.loads(FINGERPRINTS.read_text(encoding="utf-8"))


def test_the_fingerprint_record_matches_the_shipped_kernel_source():
    """THE load-bearing one. Editing the kernel invalidates the record.

    These kernels are bit-identical to ``stepping.py`` because their expression
    grouping survives Triton's MLIR pipeline, and that is held EMPIRICALLY by the
    byte gate rather than by construction. So the record carries the gate's verdict
    AND the hash of the source that verdict was measured on, and this test welds
    the two: the kernel cannot be edited without failing here, and the record
    cannot be re-cut without re-running the gate.

    It runs on a laptop, with no GPU and no Triton, which is the point — the pin
    has to bite at the merge bar, not on the one machine that can launch a kernel.
    """
    import hashlib

    record = fingerprints()
    path = PACKAGE_DIR / "kernels.py"
    live = hashlib.sha256(path.read_bytes()).hexdigest()
    # ONE HOME FOR THE RULE (device_identity.py:209). What the gate measured is
    # the compiled kernels, and for Triton those ARE the ``@triton.jit`` bodies
    # plus the constants they read -- so an edit the shared rule proves left
    # those untouched cannot move the verdict. It returns False for everything
    # it cannot establish, so the byte rule below is what applies until a
    # device_sha256 is backfilled beside this digest.
    if record["kernel_source_sha256"] != live and not weld_survives_edit(
            path, record, "kernels.py"):
        assert record["kernel_source_sha256"] == live, (
            "kernels.py changed since the bit-identity gate certified it, and "
            "the change reaches executable code. Re-run "
            "parity/meep_gpu/probe_fused_kernel_bit_identity.py --track triton on "
            "a CUDA host and re-cut fingerprints.json with the new hash and "
            "verdict.")


def test_the_fingerprint_record_matches_the_shipped_host_side():
    """The kernel is not the whole of what the gate certifies.

    ``launch.py`` picks the Yee sub-lattice, binds the coefficient pointers and
    decides ``SIGMA_IS_VOLUME``; predicate and specialized modules decide whether
    and how a kernel runs. Each is a silent wrong answer if it changes, and this
    round supplied the
    proof: ``plan_constitutive_from_arrays`` dropped its ``kernel=`` argument on
    the way to the plan object and every source-mutation leg quietly launched the
    SHIPPED kernel — 4/4 real defects reported back as 120/120 identical, with
    kernels.py byte-perfect throughout.
    """
    import hashlib

    record = fingerprints()
    recorded = record["host_sha256"]
    for name, digest in recorded.items():
        path = PACKAGE_DIR / name
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == live:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209), and the SAME call
        # test_triton_planner_composition's C19 drift test makes, deliberately:
        # two spellings of "this host file moved" is two places to disagree.
        # These are host files, so only comments and docstrings are free here --
        # a renamed variable or a reordered argument reaches the device through
        # host logic and is refused, as is anything the helper cannot establish.
        if weld_survives_edit(path, record, name):
            continue
        assert digest == live, (
            f"{name} changed since the bit-identity gate certified it, in a way "
            f"that reaches executable code. Re-run the gate on a CUDA host and "
            f"re-cut fingerprints.json.")


def test_the_fingerprint_record_names_every_shipped_kernel():
    """A kernel added without a gate verdict must fail here, not ship unmeasured."""
    source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    shipped = {line.split("def ", 1)[1].split("(")[0]
               for line in source.splitlines() if line.startswith("def ")}
    assert set(fingerprints()["kernels"]) == shipped, (
        "fingerprints.json and kernels.py disagree about which kernels exist")


def test_the_recorded_guard_is_the_launch_route_and_the_value_is_false():
    record = fingerprints()["guard"]
    assert record["value"] is False
    assert record["how_it_is_passed"] == "JIT launch keyword"
    assert "\nENABLE_FP_FUSION = False\n" in (PACKAGE_DIR / "kernels.py").read_text(
        encoding="utf-8")


def test_the_recorded_gate_verdicts_are_pass_shaped():
    """A record that degenerated into "whatever we measured" is not a record.

    Every guarded leg must be N/N and every unguarded control 0/N. The unguarded
    half is what stops the guarded half from being a comparison of two identical
    things: if the contraction guard did nothing, both columns would read N/N.
    """
    legs = fingerprints()["bit_identity_gate"]["legs"]
    for name in ("pml_curl_step", "constitutive_step"):
        guarded = legs[name]["single_launch_guarded"]
        ran, total = guarded.split("/")
        assert ran == total and int(total) > 0, (name, guarded)
        assert legs[name]["single_launch_unguarded"].startswith("0/"), name
    assert fingerprints()["bit_identity_gate"]["harness_blindness_controls"]


def test_the_fused_electric_record_is_welded_to_both_oracles_and_the_route():
    """The D/E product and the later composition recut form one evidence chain.

    This is intentionally more than a source digest. A record which retained the
    new hash but lost the separate-kernel control, mutation product, actual driver
    route, or explicit no-throughput status would turn a narrow correctness result
    into a much broader-looking claim.  The D/E gate owns the unchanged kernel
    product; the gate named by ``current_host_gate`` owns the latest additive host
    composition and its exact launch/predicate/module hashes.
    """
    import hashlib

    record = fingerprints()
    gate = record["fused_electric_gate"]
    assert gate["kernel"] == "fused_curl_constitutive_D"
    assert gate["single_launch"]["bit_identical_vs_array_path"] == "56/56"
    assert gate["single_launch"]["bit_identical_vs_two_separate_triton_kernels"] == "56/56"
    assert gate["single_launch"]["structurally_invalid_skipped"] == 8
    assert gate["mutations"]["expectations_satisfied"] == "10/10"
    assert gate["num_warps"] == launch_module.FUSED_DEFAULT_NUM_WARPS == 1
    assert gate["numpy_reference_weld"]["valid_cases_bit_identical"] == "308/308"
    assert gate["numpy_reference_weld"]["structurally_invalid_skipped"] == 44
    assert gate["engine_route"]["nondispersive"].startswith("10/10 steps byte-identical")
    assert gate["engine_route"]["dispersive"].startswith("10/10 steps byte-identical")
    assert gate["throughput"].startswith("MEASURED for the two-pair")
    throughput = gate["throughput_gate"]
    assert throughput["fused_warps"] == [1, 1]
    assert throughput["independently_tuned_unfused_warps"] == [8, 1, 2, 1]
    assert throughput["policy_raw_bit_identical"] == "12/12 through every timing window"
    assert throughput["policy_positive_cases"] == "11/12"
    assert throughput["policy_median_ratio"] > 1.07
    assert throughput["policy_worst_ratio"] > 0.999
    assert gate["dispatch"].startswith("DISABLED")

    assert gate["source_sha256"]["kernels.py"] == record["kernel_source_sha256"]
    composition = record[record["current_host_gate"]]
    # A gate only certifies the dependency closure it actually executes.  The
    # current cylindrical gate cannot honestly stand in for specialized Cartesian
    # modules that it deliberately refuses, but it must pin every central byte it
    # does exercise.
    for name in ("launch.py", "coverage.py", "cylindrical_triton.py", "__init__.py"):
        assert composition["source_sha256"][name] == record["host_sha256"][name]
    assert composition["source_sha256"]["kernels.py"] == record["kernel_source_sha256"]

    api_dir = PACKAGE_DIR.parents[1]
    for key in ("probe", "numpy_reference_weld"):
        item = gate if key == "probe" else gate[key]
        path = api_dir / item["probe"]
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["probe_sha256"] == live, (
            f"{path.name} changed since its recorded D/E evidence was produced")
    throughput_probe = api_dir / throughput["probe"]
    assert throughput["probe_sha256"] == hashlib.sha256(
        throughput_probe.read_bytes()).hexdigest()


def test_the_installed_triton_version_is_one_the_gate_certified():
    """FAILS, does not skip, on an uncertified Triton. A no-op where Triton is absent.

    On a machine with no Triton this is a no-op BY CONSTRUCTION rather than a skip:
    there is no version to disagree with, and the version guard's job there is done
    by the source-hash test above. Where Triton IS importable, an unlisted version
    fails before anything is launched.
    """
    try:
        import triton  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - absent Triton is not a failure, it is the norm
        return
    validated = fingerprints()["validated_triton_versions"]
    assert triton.__version__ in validated, (
        f"Triton {triton.__version__} is outside the validated set {validated}. "
        "The kernels' expression grouping is held empirically by the byte gate, so "
        "a version bump is a correctness event. To clear it: re-run the byte gate "
        "on a CUDA host, then append the version and the new verdict to "
        "fingerprints.json. Nothing re-records that file automatically.")


def test_the_packaging_extra_pins_a_range_and_is_a_no_op_off_linux_x86_64():
    """A RANGE, not an exact pin, and a platform marker.

    torch 2.5.x pins ``triton==3.1.0`` on linux-x86_64, so an exact pin here would
    be an unsolvable conflict against any environment that also wants torch. And
    Triton has no Metal backend, so the marker is not "unvalidated on Darwin", it
    is "unavailable" — the extra has to be installable-and-inert on the laptop that
    is the merge bar.
    """
    text = (pathlib.Path(launch_module.__file__).parents[2] /
            "pyproject.toml").read_text(encoding="utf-8")
    assert "fdtd-gpu-triton = [" in text
    # Split on the closing bracket at column 0: the entries themselves contain
    # "[fdtd-gpu]", so splitting on a bare "]" truncates after the first line.
    extra = text.split("fdtd-gpu-triton = [", 1)[1].split("\n]", 1)[0]
    assert "triton>=3.1,<3.2" in extra, extra
    assert "triton==" not in extra, "an exact pin conflicts with torch's own"
    assert "sys_platform == 'linux' and platform_machine == 'x86_64'" in extra


def test_the_unbuilt_tier_is_recorded_rather_than_omitted():
    """The PTX-census tier is not built, and the record says so with its reason.

    A guard with a silent hole reads exactly like a complete one. This asserts the
    hole is written down — so the next reader learns that a Triton bump costs a
    full CUDA-host gate run today, rather than discovering it under time pressure.
    """
    missing = fingerprints()["not_built_this_round"]
    assert "ptx_census" in missing and missing["ptx_census"]


# ---------------------------------------------------------------------------
# The tables the kernel hard-codes, checked against the array path's own
# ---------------------------------------------------------------------------

def test_dsig_axes_match_the_curl_term_tables():
    """The recurrence's (dsig, dsigu) pairing, per target, on both sides.

    The kernel hard-codes one triple for both sub-steps because
    ``B_CURL_TERMS`` and ``D_CURL_TERMS`` happen to agree on it. That agreement is
    a fact about vec.hpp's cycle_direction, not a coincidence to rely on silently.
    """
    axis_index = {"x": 0, "y": 1, "z": 2}
    for terms in (stepping.B_CURL_TERMS, stepping.D_CURL_TERMS):
        found = tuple((axis_index[t.dsig], axis_index[t.dsigu]) for t in terms)
        assert found == launch_module.DSIG_AXES


def test_the_curl_term_axes_match_what_the_kernel_differences():
    """(first, first_axis, second, second_axis) per target, both sub-steps.

    The kernel loads six shifted operands and assembles three curls from a fixed
    pattern: target 0 differences source 2 on y against source 1 on z, target 1
    source 0 on z against source 2 on x, target 2 source 1 on x against source 0
    on y. If ``stepping``'s table ever moves, this is where it is noticed.
    """
    axis_index = {"x": 0, "y": 1, "z": 2}
    expected = ((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))
    for terms, sources in ((stepping.B_CURL_TERMS, ("Ex", "Ey", "Ez")),
                           (stepping.D_CURL_TERMS, ("Hx", "Hy", "Hz"))):
        found = tuple((sources.index(t.first), t.first_axis,
                       sources.index(t.second), t.second_axis) for t in terms)
        assert found == expected
        # And the target order the kernel assumes.
        assert tuple(axis_index[t.dsig] for t in terms) == (1, 2, 0)


def test_the_metallic_mask_axes_match_the_yee_shift_table():
    """Which cell-0 plane each target's curl loses on a metallic axis.

    ``_mask_non_owned_cells`` zeroes cell 0 on every axis where the TARGET's Yee
    shift is 0. The kernel writes that out per component as a compile-time
    constant: one axis for each B target, two for each D target. A silently
    swapped triple here is a half-cell error at the wall, not a crash.
    """
    b_masked = tuple(tuple(a for a in range(3) if IYEE_SHIFTS[c][a] == 0)
                     for c in ("Bx", "By", "Bz"))
    d_masked = tuple(tuple(a for a in range(3) if IYEE_SHIFTS[c][a] == 0)
                     for c in ("Dx", "Dy", "Dz"))
    assert b_masked == ((0,), (1,), (2,))
    assert d_masked == ((1, 2), (0, 2), (0, 1))


def test_the_half_integer_pairing_is_the_one_stepping_uses():
    """B reads the half-integer coefficient set, D the integer one.

    Getting this backwards is a half-cell error, not a crash — so the suffix the
    plan appends is compared against the attribute ``_curl_coefficients`` actually
    fetches, rather than against the comment that says which is which.
    """
    _, pml = build()
    for sub_step, half_integer in (("step_B", True), ("step_D", False)):
        suffix = launch_module.SUB_STEPS[sub_step]["suffix"]
        for axis in "xyz":
            kms, sinv = stepping._curl_coefficients(pml, axis, half_integer=half_integer)
            assert kms is getattr(pml, f"kms_{axis}{suffix}")
            assert sinv is getattr(pml, f"sinv_{axis}{suffix}")
    # And the two sets are genuinely different arrays, so a swap would show.
    assert not (pml.kms_y == pml.kms_y_h).all()


# ---------------------------------------------------------------------------
# Coverage: the positive verdict
# ---------------------------------------------------------------------------

def test_the_target_configuration_is_refused_only_for_the_backend():
    """The real-field PML case is covered but for CuPy, which a laptop has not.

    This is the positive half of the predicate: on the exact configuration the
    kernel is FOR, every clause but the backend must pass. A predicate that no
    configuration satisfies is as useless as one that everything satisfies.
    """
    fields, pml = build()
    verdict = pml_curl_coverage(fields, pml)
    assert verdict.covered is False
    assert len(verdict.reasons) == 1, verdict.reasons
    assert "cupy" in verdict.reasons[0]


def test_metallic_boundaries_are_covered_too():
    fields, pml = build(boundaries="metallic")
    verdict = pml_curl_coverage(fields, pml)
    assert [r for r in verdict.reasons if "cupy" not in r] == []


# ---------------------------------------------------------------------------
# Coverage: the refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def _reasons_without_backend(fields, pml):
    return [r for r in pml_curl_coverage(fields, pml).reasons if "cupy" not in r]


def test_complex_storage_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    reasons = _reasons_without_backend(fields, PML(grid=grid, thickness=2))
    assert any("complex" in r for r in reasons), reasons


def test_a_mirror_plane_is_refused():
    fields, pml = build(symmetry=("X",))
    reasons = _reasons_without_backend(fields, pml)
    assert any("mirror" in r for r in reasons), reasons


def test_a_nonzero_k_point_is_refused():
    fields, pml = build(k_point=(0.3, 0.0, 0.0))
    reasons = _reasons_without_backend(fields, pml)
    assert any("k_point" in r for r in reasons), reasons


def test_no_active_layer_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = _reasons_without_backend(fields, PML(grid=grid, thickness=0))
    assert any("no active PML" in r for r in reasons), reasons


def test_a_conductivity_on_a_curl_target_is_refused():
    fields, pml = build()
    import numpy
    fields.set_d_conductivity(
        numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32))
    reasons = _reasons_without_backend(fields, pml)
    assert any("conductivity" in r for r in reasons), reasons


def test_a_magnetic_susceptibility_is_refused():
    """Re-aimed from ``test_a_susceptibility_is_refused`` by the composability pass.

    An ELECTRIC pole is now admitted by the curl predicate (see
    :func:`test_an_electric_lorentz_pole_is_admitted_by_the_curl_predicate`), because
    the curl differences the STORED E and H arrays and never reads a polarization.
    A MAGNETIC one is a different claim: it would drive B through a sub-step that
    does not exist in this engine, and the D curl would difference an H array
    something else was writing. Nothing constructs one today — ``PolarizationState``
    allocates over E components only and ``from_meep`` refuses ``H_susceptibilities``
    — and the clause is written and tested anyway, because "another module guards
    it" is the reasoning this predicate exists to refuse.
    """
    fields, pml = build()
    fields.polarizations.append(_MagneticPole())
    reasons = _reasons_without_backend(fields, pml)
    assert any("magnetic component" in r for r in reasons), reasons


def test_an_unreadable_susceptibility_is_refused():
    """A registered object that cannot answer ``driven()`` is not a covered one."""
    fields, pml = build()
    fields.polarizations.append(object())
    reasons = _reasons_without_backend(fields, pml)
    assert any("does not report driven()" in r for r in reasons), reasons


def test_a_susceptibility_kind_outside_lorentzian_drude_is_refused():
    fields, pml = build()
    fields.polarizations.append(_StubPole(("Ez",), kind="noisy-lorentzian"))
    reasons = _reasons_without_backend(fields, pml)
    assert any("is outside" in r and "noisy" in r for r in reasons), reasons


def test_covered_susceptibility_kinds_do_not_drift_from_the_engines():
    """The literal in ``coverage`` must equal ``dispersion.SUSCEPTIBILITY_KINDS``.

    The predicate module deliberately imports nothing from the engine, so the kind
    list is a literal. A literal that silently falls behind the engine is a
    predicate that refuses a kind the ADE kernel would have handled, or — much
    worse, if the engine ever GAINS a kind — admits one it would not.
    """
    from meep_gpu.dispersion import SUSCEPTIBILITY_KINDS

    assert coverage_module.COVERED_SUSCEPTIBILITY_KINDS == SUSCEPTIBILITY_KINDS


def test_an_instantaneous_nonlinearity_is_refused():
    fields, pml = build()
    fields._chi2_components = {"Ez": 1.0}
    fields._chi3_components = {"Ez": 1.0}
    reasons = _reasons_without_backend(fields, pml)
    assert any("chi2" in r for r in reasons), reasons


def test_bfast_and_special_kz_are_refused():
    fields, pml = build()
    fields.grid.beta = 0.25
    assert any("beta" in r for r in _reasons_without_backend(fields, pml))
    fields.grid.beta = 0.0
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        assert any("BFAST" in r for r in _reasons_without_backend(fields, pml))


def test_missing_pml_storage_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)  # no enable_pml_storage()
    reasons = _reasons_without_backend(fields, PML(grid=grid, thickness=2))
    assert any("fu_Bx is not allocated" in r for r in reasons), reasons


def test_a_non_contiguous_field_volume_is_refused():
    fields, pml = build()
    fields.Bx = fields.Bx[::-1]  # A reversed view: same shape, wrong strides.
    reasons = _reasons_without_backend(fields, pml)
    assert any("C-contiguous" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Predicate mutations — the part that proves the predicate is load-bearing
# ---------------------------------------------------------------------------

def test_mutation_widening_the_boundary_set_by_one_axis_is_caught(monkeypatch):
    """MUTATION 1: admit ``mirror`` as a covered ghost rule.

    A predicate no mutation exercises is indistinguishable from one that returns
    True. This is the widen-by-one-axis mutation: a folded grid whose boundary
    kind is ``mirror`` must be refused, and if the covered set is widened to admit
    it, the ONLY thing standing between the kernel and a wrong answer is the
    separate ``is_mirrored`` clause — so this asserts both halves.
    """
    fields, pml = build(symmetry=("X",))
    assert any("mirror" in r for r in _reasons_without_backend(fields, pml))

    monkeypatch.setattr(coverage_module, "COVERED_BOUNDARIES",
                        COVERED_BOUNDARIES + ("mirror",))
    still_refused = _reasons_without_backend(fields, pml)
    assert any("folded by a mirror plane" in r for r in still_refused), still_refused
    # And with BOTH the boundary set widened and the fold clause gone, the
    # predicate accepts a configuration the kernel cannot step. That is the
    # failure this pair of clauses exists to prevent.
    monkeypatch.setattr(coverage_module, "_call",
                        lambda obj, name, *a, **kw: kw.get("default"))
    assert _reasons_without_backend(fields, pml) == []


def _mutate(function, needle: str, replacement: str):
    """Recompile one predicate with one clause rewritten, in the module's namespace."""
    source = inspect.getsource(function)
    mutated_source = source.replace(needle, replacement)
    assert mutated_source != source, f"the clause moved; update this mutation: {needle!r}"
    namespace = dict(coverage_module.__dict__)
    exec(compile(mutated_source, "<mutated coverage>", "exec"), namespace)  # noqa: S102
    return namespace[function.__name__]


def test_mutation_dropping_the_stored_E_check_is_caught():
    """MUTATION 2, re-aimed by the composability pass: drop clause 9c.

    It used to drop the blanket dispersion refusal. That refusal is gone — the curl
    admits an electric pole — and what replaced it as the load-bearing invariant is
    ``stores_E``. The curl may difference E while a pole is live ONLY because E is a
    stored array ``update_E`` wrote; if it were recomputed as ``D * inv_eps`` inside
    the sub-step, the pole's ``- sum P`` would be missing from what the curl reads.
    An active PML forces stored E today, so the clause looks redundant — this
    mutation is what shows it is not: with the clause gone, a configuration that is
    NOT storing E is admitted.
    """
    fields, pml = build()
    fields._stored_E = False  # What a future "PML without stored E" would look like.
    assert any("recomputed from D" in r for r in _reasons_without_backend(fields, pml))

    mutated = _mutate(coverage_module.pml_curl_coverage,
                      'if not getattr(fields, "stores_E", False):',
                      "if False:")
    reasons = [r for r in mutated(fields, pml).reasons if "cupy" not in r]
    assert reasons == [], reasons  # UNCAUGHT once the clause is gone: it is load-bearing.


def test_mutation_dropping_the_magnetic_target_check_is_caught(monkeypatch):
    """MUTATION 2b: drop clause 9a, the electric-targets-only check.

    The narrowed predicate admits dispersion, so what still has to refuse is a pole
    with a MAGNETIC target. This proves clause 9a is the only thing doing it — the
    engine's own guards are elsewhere and this module may not lean on them.
    """
    fields, pml = build()
    fields.polarizations.append(_MagneticPole())
    assert any("magnetic component" in r for r in _reasons_without_backend(fields, pml))

    mutated = _mutate(coverage_module._susceptibility_reasons,
                      'if _call(state, "drives", name, default=False):',
                      "if False:")
    monkeypatch.setattr(coverage_module, "_susceptibility_reasons", mutated)
    reasons = _reasons_without_backend(fields, pml)
    assert reasons == [], reasons  # UNCAUGHT once the clause is gone.


def test_mutation_admitting_a_conductivity_is_caught(monkeypatch):
    """MUTATION 9 (spec §G.m9): the widening went one clause too far.

    Dispersion was admitted because the curl provably does not depend on it. A
    conductivity is the control: the curl genuinely DOES depend on it — a conductive
    target routes to ``_apply_conductive_pml_update``'s three-history recurrence
    (stepping.py:2001), a different recurrence entirely. Widening the predicate the
    same way for conductivity produces an admitted configuration the kernel would
    step wrongly, which is the failure this test exhibits deliberately. It is the
    control that says the composability widening was reasoned, not permissive.
    """
    import numpy

    fields, pml = build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32))
    assert any("conductivity" in r for r in _reasons_without_backend(fields, pml))

    mutated = _mutate(coverage_module.pml_curl_coverage,
                      "if conductive:", "if False:")
    reasons = [r for r in mutated(fields, pml, "step_D").reasons if "cupy" not in r]
    assert reasons == [], reasons


# ---------------------------------------------------------------------------
# The composability fix: dispersion is admitted for the curl, refused for update_E
# ---------------------------------------------------------------------------

class _StubPole:
    """The smallest object the predicate can read as a susceptibility.

    Deliberately not a ``PolarizationState``: these tests are about what the
    PREDICATE refuses, and a stub is how a target set the engine cannot currently
    construct (a magnetic pole) gets exercised at all.
    """

    def __init__(self, driven=("Ez",), kind="lorentzian", magnetic=()):
        self._driven = tuple(driven)
        self._magnetic = tuple(magnetic)
        self.susceptibility = type("S", (), {"kind": kind})()

    def driven(self):
        return self._driven

    def drives(self, component):
        return component in self._driven or component in self._magnetic


def _MagneticPole():  # noqa: N802 - reads as a constructor at the call sites
    return _StubPole(driven=("Ez",), magnetic=("Bz",))


def _with_pole(component="Ez", sigma=0.3, kind="lorentzian"):
    """A real ``PolarizationState`` on a real dispersive Fields, NumPy backend."""
    import numpy

    from meep_gpu.dispersion import PolarizationState, Susceptibility

    fields, pml = build()
    fields.enable_field_storage()
    susceptibility = Susceptibility(frequency=1.0, gamma=0.1, kind=kind)
    volume = numpy.zeros(fields.grid.shape, dtype=numpy.float32)
    volume[...] = 0.0
    volume[1:-1, 1:-1, 1:-1] = sigma
    sigmas = {name: (volume if name == component else 0.0)
              for name in ("Ex", "Ey", "Ez")}
    state = PolarizationState(susceptibility, sigmas, fields.grid, numpy.float32)
    fields.polarizations.append(state)
    return fields, pml, state


def test_an_electric_lorentz_pole_is_admitted_by_the_curl_predicate():
    """THE COMPOSABILITY FIX, stated as a test.

    Before this pass the curl refused any registered susceptibility, which meant the
    curl kernel and the ADE kernel could never be used in the same run — the ADE
    kernel is needed exactly when the curl refused. The curl reads ``fields.Ex/Ey/Ez``
    and ``fields.Hx/Hy/Hz``, both stored under PML, and no function on its path
    (``step_B``, ``step_D``, ``_apply_curl``, ``_curl_operands``,
    ``_curl_from_operands``, ``_mask_non_owned_cells``, ``_apply_pml_update``)
    mentions a polarization. So the refusal was over-broad and this is the widening.
    """
    fields, pml, _ = _with_pole()
    assert fields.has_polarizations
    assert _reasons_without_backend(fields, pml) == [], _reasons_without_backend(fields, pml)


def test_an_inert_zero_sigma_pole_is_admitted_too():
    """A pole whose sigma is identically zero drives nothing and must not cost the path.

    ``sigma_is_trivial`` (dispersion.py:600-612) means such a state allocates no P
    and is provably 0.0 in every cell for every step, so the run is byte-identical to
    the non-dispersive engine — yet the old clause's second disjunct
    (``or fields.polarizations``) refused the kernel for it.
    """
    fields, pml, state = _with_pole(sigma=0.0)
    assert state.driven() == ()
    assert fields.has_polarizations is False
    assert _reasons_without_backend(fields, pml) == []


def test_the_curl_predicate_admits_off_diagonal_epsilon_and_says_so():
    """The inconsistency that showed the dispersion refusal was over-broad.

    Off-diagonal chi1inv is a constitutive-only feature with the same shape as
    dispersion — it forces stored E and its whole effect is one extra term inside
    ``update_E`` — and the predicate always ADMITTED it silently while refusing
    dispersion on exactly the argument it declined to apply here. One of the two was
    wrong; the source said it was the refusal. Now both are admitted for the curl
    and both are named in the module's docstring, so the file states a decision
    instead of being silent.
    """
    source = pathlib.Path(coverage_module.__file__).read_text(encoding="utf-8")
    assert "off-diagonal" in source.lower()
    assert "has_offdiagonal_epsilon" in source  # refused on the E side, by name


# ---------------------------------------------------------------------------
# The constitutive sub-step (update_H / update_E)
# ---------------------------------------------------------------------------

def _constitutive_reasons(fields, pml, side):
    return [r for r in coverage_module.constitutive_coverage(fields, pml, side).reasons
            if "cupy" not in r]


def test_both_constitutive_sides_are_covered_but_for_the_backend():
    fields, pml = build()
    for side in ("H", "E"):
        assert _constitutive_reasons(fields, pml, side) == [], side


def test_the_constitutive_coefficient_pairing_is_the_one_stepping_uses():
    """H reads the INTEGER coefficient set, E the half-integer one.

    ``stepping.py:948`` passes ``half_integer=False`` for H and ``:986``
    ``half_integer=True`` for E. Swapped, it is a half-cell error in the absorber
    profile: converged, smooth, and off by a fraction of a cell. The plan chooses the
    sub-lattice on the host and the kernel cannot check it, so the choice is compared
    here against what ``_constitutive_coefficients`` actually fetches.
    """
    _, pml = build()
    for side, half_integer in (("H", False), ("E", True)):
        assert coverage_module.CONSTITUTIVE_SIDES[side]["half_integer"] is half_integer
        suffix = "_h" if half_integer else ""
        for axis in "xyz":
            kps, kms = stepping._constitutive_coefficients(pml, axis,
                                                           half_integer=half_integer)
            assert kps is getattr(pml, f"kps_{axis}{suffix}")
            assert kms is getattr(pml, f"kms_{axis}{suffix}")
    assert not (pml.kps_y == pml.kps_y_h).all()  # a swap would genuinely show


def test_the_constitutive_terms_match_steppings_own_tables():
    """Targets, sources and the OWN-AXIS coefficient index, per component.

    The coefficient index is MEEP's ``dsigw`` — the absorption a component
    accumulates along the direction it points in — not the ``dsig``/``dsigu`` cycle
    the curl recurrence uses. Component 0 takes axis x, 1 takes y, 2 takes z.
    """
    for side, terms in (("H", stepping.H_CONSTITUTIVE_TERMS),
                        ("E", stepping.E_CONSTITUTIVE_TERMS)):
        spec = coverage_module.CONSTITUTIVE_SIDES[side]
        assert tuple(t[0] for t in terms) == spec["targets"]
        assert tuple(t[1] for t in terms) == spec["sources"]
        assert tuple(t[2] for t in terms) == ("x", "y", "z")
        assert spec["aux"] == tuple("f_w_" + t[0] for t in terms)


def test_update_E_refuses_a_susceptibility_and_the_curl_does_not():
    """The two predicates disagree on ONE configuration, and that is the design.

    With a pole registered ``update_E``'s source is ``D - sum P`` and ``update_P``
    closes the step, so ``update_E`` falls out of the covered set — while the curls
    and ``update_H``, which the pole does not touch, stay in. Coverage is a set PER
    SUB-STEP; that is what makes the two kernels co-exist.
    """
    fields, pml, _ = _with_pole()
    assert _reasons_without_backend(fields, pml) == []          # curl: covered
    assert _constitutive_reasons(fields, pml, "H") == []        # update_H: covered
    reasons = _constitutive_reasons(fields, pml, "E")
    assert any("(D - sum P)" in r for r in reasons), reasons    # update_E: refused


def test_update_E_refuses_off_diagonal_epsilon():
    fields, pml = build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": 0.1}}
    assert fields.has_offdiagonal_epsilon
    reasons = _constitutive_reasons(fields, pml, "E")
    assert any("off-diagonal" in r for r in reasons), reasons
    assert _reasons_without_backend(fields, pml) == []  # the curl still covers it


def test_the_constitutive_predicate_refuses_a_fold_even_though_it_reads_no_neighbour():
    """RISK 2, pinned: no stencil does NOT mean boundaries do not matter.

    ``update_H``/``update_E`` read no neighbour, which invites the reasoning "no
    stencil, so accept mirror and cylindrical". Wrong: a fold changes the STORED
    EXTENT, hence ``n_a``, hence the coefficient index on the component's own axis,
    hence every cell's coefficient — a smooth, plausible, entirely wrong field.
    """
    fields, pml = build(symmetry=("X",))
    for side in ("H", "E"):
        reasons = _constitutive_reasons(fields, pml, side)
        assert any("mirror" in r for r in reasons), (side, reasons)


def test_the_constitutive_predicate_refuses_a_scalar_inverse_epsilon():
    fields, pml = build()
    fields._inv_eps_components = {"Ex": 1.0, "Ey": 1.0, "Ez": 1.0}
    reasons = _constitutive_reasons(fields, pml, "E")
    assert any("is a scalar, not a volume" in r for r in reasons), reasons


def test_the_constitutive_predicate_refuses_an_unnamed_side():
    fields, pml = build()
    with pytest.raises(ValueError, match="side must be one of"):
        coverage_module.constitutive_coverage(fields, pml, "B")


# ---------------------------------------------------------------------------
# The ADE sub-step (update_P)
# ---------------------------------------------------------------------------

def _ade_reasons(fields, state, component):
    return [r for r in coverage_module.ade_update_p_coverage(fields, state, component).reasons
            if "cupy" not in r]


def test_a_real_lorentz_pole_is_covered_but_for_the_backend():
    fields, _, state = _with_pole()
    assert state.driven() == ("Ez",)
    assert _ade_reasons(fields, state, "Ez") == []


def test_a_drude_pole_is_covered_too():
    fields, _, state = _with_pole(kind="drude")
    assert _ade_reasons(fields, state, "Ez") == []


def test_the_ade_predicate_refuses_a_component_the_pole_does_not_drive():
    fields, _, state = _with_pole(component="Ez")
    reasons = _ade_reasons(fields, state, "Ex")
    assert any("does not drive Ex" in r for r in reasons), reasons


def test_the_ade_predicate_refuses_a_magnetic_component_by_name():
    fields, _, state = _with_pole()
    reasons = _ade_reasons(fields, state, "Bz")
    assert any("outside" in r for r in reasons), reasons


def test_the_ade_predicate_requires_the_pml_drive_field():
    """RISK/mutation m11: P must be driven from ``f_w_*``, never the stored E.

    ``Fields.drive_field`` returns ``f_w_*`` under PML and the stored E without
    (fields.py:1140-1163). The two agree exactly OUTSIDE the absorber, so driving
    from E passes every no-PML case and is wrong only inside the layer — smoothly,
    plausibly, and in the region a transmission spectrum is normalised against. The
    predicate pins that the auxiliary exists so the launcher's read cannot fall back.
    """
    fields, _, state = _with_pole()
    assert fields.drive_field("Ez") is fields.f_w_Ez
    fields.f_w_Ez = None
    reasons = _ade_reasons(fields, state, "Ez")
    assert any("f_w_Ez is not allocated" in r for r in reasons), reasons


def test_sigma_is_volume_agrees_with_what_the_predicate_checked():
    """Mutation m13's target: the constexpr and the clause are ONE function.

    Choosing ``SIGMA_IS_VOLUME`` wrongly reads a scalar as a pointer or a pointer as
    a scalar — a wrong answer, not a crash — so the check and the compile-time
    choice may not be two expressions that happen to agree.
    """
    fields, _, state = _with_pole()
    assert coverage_module.sigma_is_volume(state, "Ez") is True
    assert _ade_reasons(fields, state, "Ez") == []
    state.sigma["Ez"] = 0.5  # the uniform case
    assert coverage_module.sigma_is_volume(state, "Ez") is False
    assert _ade_reasons(fields, state, "Ez") == []


def test_the_ade_predicate_refuses_a_mis_shaped_history():
    import numpy

    fields, _, state = _with_pole()
    state.P_prev["Ez"] = numpy.zeros((2, 2, 2), dtype=numpy.float32)
    reasons = _ade_reasons(fields, state, "Ez")
    assert any("P_prev[Ez] shape" in r for r in reasons), reasons


def test_the_ade_predicate_refuses_a_missing_scratch():
    fields, _, state = _with_pole()
    state._scratch = None
    reasons = _ade_reasons(fields, state, "Ez")
    assert any("_scratch[Ez] is not allocated" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Composition — the covered SUBSET, and the sub-steps that fall out of it
# ---------------------------------------------------------------------------

def test_the_step_plan_replaces_nothing_on_a_numpy_host_and_says_why():
    """Refusals return a plan that replaces nothing, never an exception.

    The array path is always a correct answer; a predicate that raises into a caller
    that would otherwise have stepped correctly is not.
    """
    fields, pml = build()
    plan = launch_module.plan_step(fields, pml)
    assert plan.replaces == ()
    assert set(plan.reasons) == {"step_B", "step_D", "update_H", "update_E"}
    for reasons in plan.reasons.values():
        assert any("cupy" in r for r in reasons), reasons


def test_the_step_order_matches_the_drivers_own():
    """``STEP_ORDER`` is the driver's sub-step sequence, not an alphabetical list."""
    assert launch_module.STEP_ORDER == (
        "step_B", "fill_B", "update_H",
        "step_D", "fill_D", "update_E", "update_P",
    )


def test_the_step_plan_reports_disjoint_products_for_a_dispersive_run():
    """The composed verdict for the now-covered dispersive configuration.

    On a CuPy host this configuration selects all five products: PML curls,
    ordinary H, specialized pole-aware E, and ADE P. On NumPy every predicate
    refuses for the backend. The two curl products are still evaluated
    independently, so their refusal combines a backend reason from each product
    with the no-PML product's active-PML reason.
    """
    fields, pml, _ = _with_pole()
    plan = launch_module.plan_step(fields, pml)
    assert plan.replaces == ()  # NumPy host
    backend_only = {name for name, reasons in plan.reasons.items()
                    if all("cupy" in r for r in reasons)}
    assert backend_only == {"update_H", "update_P"}
    for name in ("step_B", "step_D"):
        assert any(reason.startswith("PML: ") and "cupy" in reason
                   for reason in plan.reasons[name])
        assert any(reason.startswith("no-PML: ") and "cupy" in reason
                   for reason in plan.reasons[name])
        assert any("no-PML: an active PML" in reason
                   for reason in plan.reasons[name])
    assert any("(D - sum P)" in r for r in plan.reasons["update_E"])


def test_a_refused_polarization_keeps_the_whole_update_P_substep_on_the_array_path():
    """Halves of one sub-step do not compose, and the plan says so.

    ``PolarizationState.update`` rotates a SHARED scratch buffer from component to
    component inside one call. Covering two of three components and leaving the
    third to the array path would interleave two rotations over one buffer set —
    which is why :func:`plan_ade_update_p` is all-or-nothing per state.
    """
    fields, pml, state = _with_pole()
    state.susceptibility = type("S", (), {"kind": "gyrotropic"})()
    plan = launch_module.plan_step(fields, pml)
    assert "update_P" in plan.reasons
    assert any("gyrotropic" in r for r in plan.reasons["update_P"])


# ---------------------------------------------------------------------------
# What the kernel source says, read on a machine that cannot compile it
# ---------------------------------------------------------------------------

def kernel_source() -> str:
    return (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")


def test_the_constitutive_kernel_reads_the_auxiliary_before_it_writes_it():
    """RISK 4: ``_apply_constitutive_pml`` copies ``fw`` BEFORE overwriting it.

    A kernel that stores ``src`` into ``f_w`` before loading ``prev`` reads its own
    write. On the array path the copy makes that impossible; in a fused kernel it is
    one line's ordering, and it produces a wrong answer only where ``kms != 0`` —
    i.e. inside the PML only — which looks like a slightly worse absorber, not like
    a bug. The gate carries the mutation; this is the cheap local check that the
    shipped order is right, on a machine that cannot compile the kernel at all.
    """
    body = kernel_source().split("def constitutive_step(")[1].split(
        "def fused_curl_constitutive_B")[0]
    for index in (0, 1, 2):
        load = body.index(f"prev{index} = tl.load(w{index} + idx")
        store = body.index(f"tl.store(w{index} + idx, src{index}")
        assert load < store, f"component {index} writes f_w before reading fw_previous"


def test_the_constitutive_kernel_keeps_the_two_accumulations_separate():
    """RISK/mutation: the grouping is ``((f + kps*src) - kms*prev)``.

    Flattened to ``f + (kps*src - kms*prev)`` it is a different float32 number. The
    array path spells it as two statements (``field += kps * fw`` then
    ``field -= kms * fw_previous``) and so does the kernel; a single expression here
    would be the regroup mutation shipped by accident.
    """
    body = kernel_source().split("def constitutive_step(")[1].split(
        "def fused_curl_constitutive_B")[0]
    for index in (0, 1, 2):
        assert f"a{index} = a{index} + kp_{index} * src{index}" in body
        assert f"a{index} = a{index} - km_{index} * prev{index}" in body


def test_the_constitutive_kernel_indexes_coefficients_on_the_components_own_axis():
    """Component 0 reads the coefficient at ``i``, 1 at ``j``, 2 at ``k``.

    That is MEEP's ``dsigw``. Using the curl's ``dsig``/``dsigu`` cycle here instead
    would absorb each component along the wrong direction — again converged and
    smooth and wrong.
    """
    body = kernel_source().split("def constitutive_step(")[1].split(
        "def fused_curl_constitutive_B")[0]
    for index, axis_variable in enumerate("ijk"):
        assert f"kp_{index} = tl.load(kp{index} + {axis_variable}," in body
        assert f"km_{index} = tl.load(km{index} + {axis_variable}," in body


# ---------------------------------------------------------------------------
# The cross-sub-step fused pair — step_B + zero_metal_B + update_H (F1)
# ---------------------------------------------------------------------------
#
# The fusion is legal because the producer writes B at ``idx`` and the consumer
# reads B at ``idx``: same index, no halo. What runs BETWEEN them in the driver is
# the whole of the risk, and it is exactly two things — the magnetic source inject
# (refused) and ``zero_metal_B`` (carried inline). The tests below pin both, plus
# the register substitution itself, on a machine that cannot compile the kernel.


def fused_body(pair: str = "B") -> str:
    """One fused kernel's executable text — its docstring quotes the code it replaces."""
    if pair == "B":
        body = kernel_source().split("def fused_curl_constitutive_B(")[1].split(
            "def fused_curl_constitutive_D")[0]
    elif pair == "D":
        body = kernel_source().split("def fused_curl_constitutive_D(")[1].split(
            "def ade_update_p")[0]
    else:  # pragma: no cover - a test helper, not package behaviour
        raise ValueError(pair)
    return body.split('"""', 2)[2]


def test_the_fused_pair_is_covered_but_for_the_backend_when_sources_are_declared():
    """The positive half: on the configuration the fusion is FOR, only CuPy is missing."""
    fields, pml = build()
    verdict = coverage_module.fused_pair_coverage(fields, pml, "B", sources=())
    assert verdict.covered is False
    assert all("cupy" in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("pair", ("B", "D"))
def test_an_undeclared_source_set_is_a_refusal_not_an_empty_one(pair):
    """``sources=None`` must REFUSE. ``Fields`` does not hold the source list.

    A predicate that reads "no magnetic source" off its own ignorance is the
    over-covering failure the plan's §10.4 names as the dangerous one: it would
    dispatch a fused pair on a run whose magnetic current lands inside the seam,
    and the deposit would vanish silently.
    """
    fields, pml = build()
    reasons = coverage_module.fused_pair_coverage(fields, pml, pair).reasons
    assert any("source set was not declared" in reason for reason in reasons), reasons


class _Source:
    """A source that declares its family and nothing else.

    It publishes no ``_point_ix``, which is deliberate: it is the case
    ``deposit_repair`` cannot save and restore, so it stays refused under the shipped
    declaration and can still be used to ask which seam a family lands in.
    """

    def __init__(self, field_type):
        self.field_type = field_type


def test_a_magnetic_source_lands_in_the_B_seam_and_an_electric_one_does_not(monkeypatch):
    """The seam's only real occupant. Named, with the driver line that puts it there.

    Asked with the deposit repair NOT declared, which isolates the SEAM ASSIGNMENT from
    what the product does about it. A wrongly assigned seam would refuse the magnetic
    source in the D pair and admit it here — the over-covering
    ``test_each_predicate_asks_about_the_seam_its_own_refusal_names`` caught once
    already on the Metal side.
    """
    fields, pml = build()
    monkeypatch.setattr(coverage_module, "CARRIES_DEPOSIT_REPAIR", False)

    magnetic = coverage_module.fused_pair_coverage(
        fields, pml, "B", sources=(_Source("B"),)).reasons
    assert any("is magnetic" in reason for reason in magnetic), magnetic
    electric = coverage_module.fused_pair_coverage(
        fields, pml, "B", sources=(_Source("D"),)).reasons
    assert not any("is magnetic" in reason for reason in electric), electric


def test_an_electric_source_lands_in_the_D_seam_and_a_magnetic_one_does_not(monkeypatch):
    """The other seam: electric injection sits between step_D and update_E."""
    fields, pml = build()
    monkeypatch.setattr(coverage_module, "CARRIES_DEPOSIT_REPAIR", False)

    electric = coverage_module.fused_pair_coverage(
        fields, pml, "D", sources=(_Source("D"),)).reasons
    assert any("is electric" in reason for reason in electric), electric
    magnetic = coverage_module.fused_pair_coverage(
        fields, pml, "D", sources=(_Source("B"),)).reasons
    assert not any("is electric" in reason for reason in magnetic), magnetic


def test_the_shipped_predicate_carries_the_in_seam_deposit_instead_of_refusing_it():
    """What the flag bought, at the predicate that ships it.

    The refusal that used to fall on every in-seam source is gone; what is left is the
    narrower one the repair genuinely cannot carry — a source that never says which
    index it writes. Both halves matter: the first is the 215-instance admission, the
    second is the fail-closed edge that keeps it from being an unconditional yes.
    """
    fields, pml = build()
    assert coverage_module.CARRIES_DEPOSIT_REPAIR is True

    for pair, family in (("B", "B"), ("D", "D")):
        reasons = coverage_module.fused_pair_coverage(
            fields, pml, pair, sources=(_Source(family),)).reasons
        assert not any("driver injects it BETWEEN" in reason
                       for reason in reasons), reasons
        assert any("does not publish the index it writes" in reason
                   for reason in reasons), reasons


def test_the_D_pair_inherits_update_Es_dispersion_refusal():
    """D cannot feed E directly when the constitutive source is D - sum(P)."""
    fields, pml, _ = _with_pole()
    reasons = coverage_module.fused_pair_coverage(
        fields, pml, "D", sources=()).reasons
    assert any(
        reason.startswith("constitutive half: ") and "(D - sum P)" in reason
        for reason in reasons
    ), reasons


def test_the_magnetic_field_type_literal_does_not_drift_from_sources_py():
    from meep_gpu.sources import FIELD_TYPE_B

    assert coverage_module.MAGNETIC_FIELD_TYPE == FIELD_TYPE_B


def test_the_fused_pair_inherits_both_halves_refusals():
    """It is the conjunction of two predicates, and it says which half objected.

    The prefixes matter: one launch now carries two sub-steps' coverage surfaces,
    and a refusal that did not say which of them objected would send a reader to
    the wrong predicate.
    """
    fields, pml = build()
    reasons = coverage_module.fused_pair_coverage(fields, pml, "B", sources=()).reasons
    assert any(reason.startswith("curl half: ") for reason in reasons), reasons
    assert any(reason.startswith("constitutive half: ") for reason in reasons), reasons
    # A nonlinearity is a shared clause; it must still arrive labelled by half.
    lossy, lossy_pml = build()
    lossy.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.2})
    labelled = coverage_module.fused_pair_coverage(
        lossy, lossy_pml, "B", sources=()).reasons
    assert any("curl half: " in r and "chi2/chi3" in r for r in labelled), labelled
    assert any("constitutive half: " in r and "chi2/chi3" in r for r in labelled), labelled


def test_zero_metal_axes_answers_what_zero_metal_itself_asks():
    """``ZM_X/Y/Z`` is the grid's own metallic declaration, not the ghost rule.

    ``stepping._zero_metal`` asks ``grid.is_metallic(axis) and not
    grid.is_mirrored(axis)`` behind ``grid.has_metallic``;
    ``stepping._boundary_kinds`` answers a different question (a fold outranks the
    declaration, an invariant axis softens to periodic). Binding the constexprs
    from the second would clear the wrong planes.
    """
    periodic, _ = build()
    assert coverage_module.zero_metal_axes(periodic.grid) == (False, False, False)
    walled, _ = build(boundaries="metallic")
    assert coverage_module.zero_metal_axes(walled.grid) == (True, True, True)


def test_zero_metal_axes_matches_the_components_the_array_path_clears():
    """Axis ``d`` clears the B component whose Yee shift on ``d`` is 0 — one each.

    That one-to-one mapping is what lets the kernel spell the wipe as
    ``ZM_X -> v0``, ``ZM_Y -> v1``, ``ZM_Z -> v2``. If any axis cleared two B
    components the constexprs would be under-specified and the inline wipe wrong.
    """
    for axis, component in enumerate(("Bx", "By", "Bz")):
        cleared = [name for name in ("Bx", "By", "Bz")
                   if IYEE_SHIFTS[name][axis] == 0]
        assert cleared == [component], (axis, cleared)


def test_zero_metal_axes_clear_two_D_components_per_wall():
    """A PEC wall clears tangential D: two components, unlike the B pair's one."""
    for axis in range(3):
        cleared = [name for name in ("Dx", "Dy", "Dz")
                   if IYEE_SHIFTS[name][axis] == 0]
        assert len(cleared) == 2, (axis, cleared)
        assert ("Dx", "Dy", "Dz")[axis] not in cleared


def test_the_fused_kernel_applies_the_wall_wipe_to_the_curl_result_before_the_consumer():
    """The seam's ordering, read off the source.

    ``zero_metal_B`` runs between the two sub-steps (driver.py:3167 vs :3169), so
    the constitutive half must read the ZEROED B. Applying it after — or only to
    the stored copy — computes H from a value the array path never had, and only
    on the wall plane of a walled run, which no benchmark case is.
    """
    body = fused_body()
    for index, axis in enumerate("xyz"):
        wipe = body.index(f"v{index} = tl.where(at_{axis}, 0.0, v{index})")
        store = body.index(f"tl.store(f{index} + idx, v{index}, mask=live)")
        consume = body.index(f"src{index} = v{index}")
        assert wipe < store, f"component {index}: B is stored before the wall wipe"
        assert wipe < consume, f"component {index}: update_H reads an unwiped B"


def test_the_D_fused_kernel_applies_both_tangential_wall_wipes_before_update_E():
    """Each D register is cleared on the two wall axes where its Yee shift is zero."""
    body = fused_body("D")
    for component_index, component in enumerate(("Dx", "Dy", "Dz")):
        store = body.index(f"tl.store(f{component_index} + idx, v{component_index}, mask=live)")
        consume = body.index(f"src{component_index} = v{component_index}")
        for axis, axis_name in enumerate("xyz"):
            snippet = (
                f"v{component_index} = tl.where(at_{axis_name}, 0.0, v{component_index})"
            )
            if IYEE_SHIFTS[component][axis] == 0:
                wipe = body.index(snippet)
                assert wipe < store
                assert wipe < consume
            else:
                assert snippet not in body


def test_the_D_fused_kernel_scales_each_register_by_its_inverse_epsilon():
    """update_E consumes D*chi1inv; using the raw D register is silently wrong."""
    body = fused_body("D")
    for index in range(3):
        assert f"src{index} = v{index} * tl.load(ie{index} + idx" in body


def test_the_fused_kernel_substitutes_the_register_for_the_reload():
    """The fusion itself: no ``tl.load`` of the curl target feeds the constitutive half.

    ``constitutive_step`` reads ``src0 = tl.load(g0 + idx)``; here ``g0`` is the
    CURL's source (Ex), and the constitutive source is the register. A reload would
    be bit-identical — it is a float32 register into a float32 array and back — but
    it is the whole of the traffic this round is trying to remove, so its absence
    is asserted rather than assumed.
    """
    body = fused_body()
    for index in range(3):
        assert f"src{index} = v{index}" in body
        assert f"src{index} = tl.load(g{index} + idx" not in body


def test_the_fused_kernel_keeps_both_halves_groupings_verbatim():
    """Every parenthesisation that decides a float32 value, in one place.

    The curl's ``dtdx * ((a - b) + (c - d))`` and the constitutive side's two
    separate accumulations. A fused body is a copy, and a copy is exactly where a
    silent re-association gets introduced.
    """
    body = fused_body()
    assert "curl0 = dtdx * ((c_y - c) + (b - b_z))" in body
    assert "curl1 = dtdx * ((a_z - a) + (c - c_x))" in body
    assert "curl2 = dtdx * ((b_x - b) + (a - a_y))" in body
    for index in range(3):
        assert f"a{index} = a{index} + kp_{index} * src{index}" in body
        assert f"a{index} = a{index} - km_{index} * prev{index}" in body
        assert body.index(f"prev{index} = tl.load(w{index} + idx") < body.index(
            f"tl.store(w{index} + idx, src{index}")


def test_the_fused_kernel_carries_the_curls_recurrence_unchanged():
    """The split-field recurrence is the curl's, character for character."""
    curl = kernel_source().split("def pml_curl_step(")[1].split(
        "def constitutive_step")[0]
    body = fused_body()
    for index, (dsig, dsigu) in enumerate((("y", "z"), ("z", "x"), ("x", "y"))):
        line = (f"n{index} = ((p{index} * km_{dsig}) - curl{index}) * si_{dsig}")
        assert line in curl and line in body


def test_the_fused_plan_binds_both_pairs_sub_lattices_the_way_stepping_does():
    """B/H binds half→integer and D/E binds integer→half, each in one launch.

    Both tables now ride the same plan, which is a new way to get the pairing
    wrong: one swap would be a half-cell error in the absorber profile, converged
    and smooth. The builder is read off the source because a CuPy-free host cannot
    construct the plan.
    """
    source = inspect.getsource(launch_module.plan_fused_pair)
    assert 'f"{stem}_{axis}{curl_spec[\'suffix\']}"' in source
    assert '("kms", "sinv")' in source
    assert 'side_spec[\'half_integer\']' in source
    assert '("kps", "kms")' in source
    assert launch_module.SUB_STEPS["step_B"]["suffix"] == "_h"
    assert launch_module.SUB_STEPS["step_D"]["suffix"] == ""
    assert coverage_module.CONSTITUTIVE_SIDES["H"]["half_integer"] is False
    assert coverage_module.CONSTITUTIVE_SIDES["E"]["half_integer"] is True


def test_both_cross_sub_step_pairs_are_explicitly_enumerated():
    """The D/E pair is a separate admitted product, never inferred from BACKWARD."""
    assert tuple(coverage_module.FUSED_PAIRS) == ("B", "D")
    assert coverage_module.FUSED_PAIRS["B"]["curl"] == "step_B"
    assert coverage_module.FUSED_PAIRS["B"]["constitutive"] == "H"
    assert coverage_module.FUSED_PAIRS["D"]["curl"] == "step_D"
    assert coverage_module.FUSED_PAIRS["D"]["constitutive"] == "E"


def test_plan_step_leaves_the_unfused_composition_as_the_default():
    """The control this round is measured against must stay the default path."""
    assert inspect.signature(launch_module.plan_step).parameters["fuse"].default is False


def test_fused_plans_default_to_the_measured_one_warp_policy():
    """The opt-in composer must launch the setting the policy sweep measured."""
    assert launch_module.FUSED_DEFAULT_NUM_WARPS == 1
    for builder in (
        launch_module.FusedPairPlan,
        launch_module.plan_fused_pair,
        launch_module.plan_fused_pair_from_arrays,
        launch_module.plan_step,
    ):
        assert inspect.signature(builder).parameters["num_warps"].default == 1
    gate_source = (PACKAGE_DIR.parents[1]
                   / "parity/meep_gpu/gate_triton_fused_electric.py").read_text(
                       encoding="utf-8")
    assert 'parser.add_argument("--num-warps", type=int, default=1)' in gate_source
    assert "kernel=kernel, num_warps=num_warps" in gate_source


def test_separate_plans_expose_the_same_opt_in_warp_control_as_fusion():
    """A fused tuning result must be comparable to a tuned four-kernel control.

    Defaults stay delegated to Triton; merely adding the control must not silently
    change the compiled kernel used by existing gates or callers.
    """
    for builder in (
        launch_module.plan_pml_curl,
        launch_module.plan_from_arrays,
        launch_module.plan_constitutive,
        launch_module.plan_constitutive_from_arrays,
    ):
        parameter = inspect.signature(builder).parameters["num_warps"]
        assert parameter.default is None
    for plan_class in (launch_module.PmlCurlPlan, launch_module.ConstitutivePlan):
        run_source = inspect.getsource(plan_class.run)
        assert '{"num_warps": self.num_warps}' in run_source
        assert "**extra" in run_source


def test_the_noop_sentinel_still_reports_update_H_as_replaced():
    """``replaces`` must name BOTH sub-steps: both really are off the array path.

    A plan that reported only ``step_B`` would tell a reader — and the whole-step
    regression test — that ``update_H`` ran on the array path, which it did not.
    """
    pair = object()
    noop = launch_module.NoopPlan("update_H", pair)
    plan = launch_module.TritonStepPlan({"step_B": object(), "update_H": noop}, (), {})
    assert plan.replaces == ("step_B", "update_H")
    assert noop.absorbed_by is pair
    assert noop.run() is None


def test_the_fused_kernel_is_an_addition_and_the_separate_kernels_remain():
    """The composed-but-unfused path is the control, so it has to stay runnable."""
    source = kernel_source()
    for name in ("pml_curl_step", "constitutive_step", "ade_update_p",
                 "fused_curl_constitutive_B", "fused_curl_constitutive_D"):
        assert f"def {name}(" in source
    launch_source = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    for name in ("class PmlCurlPlan", "class ConstitutivePlan", "class FusedPairPlan"):
        assert name in launch_source
