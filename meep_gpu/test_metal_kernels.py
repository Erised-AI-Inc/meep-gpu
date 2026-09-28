"""Merge-bar tests for the Metal kernel package.

Four things are pinned here that no gate can pin, because a gate needs a GPU and
this file must pass on any host:

1. **The optional-dependency contract.** :mod:`meep_gpu.metal_kernels`, its
   coverage predicates and its shader sources import with no torch anywhere near
   them. A predicate that could only be read on a Mac would be a predicate nobody
   reviews.
2. **The clause-list equivalence pin.** ``metal_kernels.coverage._grid_reasons``
   is a DUPLICATE of the Triton one with clause 1 replaced, and this file runs
   both over a shared configuration matrix and asserts the reason sets are equal
   once each side's own backend clause is removed. That is what makes the
   duplication a stated debt rather than a drift generator: a clause added to
   either list and not to the other fails here, which a string filter over the
   Triton list would never have caught.
3. **The single contraction-pragma spelling.** The directive that decides
   bit-identity is spelled once in the package; a second spelling, or any math-mode
   pragma, fails the build. Mirrors the Triton package's ``ENABLE_FP_FUSION`` test.
4. **The fingerprints.** The checked-in file must match the tree.

The device-touching tests below are skipped without MPS, and they are the ones
that check the arithmetic against ``stepping.py`` in process. The full case matrix
lives in ``parity/meep_gpu/gate_metal_pml.py``; what is here is the smallest
non-vacuous version of it, so a red merge bar names the defect without a GPU run.
"""

from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
import pytest

from meep_gpu.metal_kernels import coverage as metal_coverage
from meep_gpu.metal_kernels import shaders, subnormal
from meep_gpu.triton_kernels import coverage as triton_coverage

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_kernels")


def _package_sources():
    for name in sorted(os.listdir(PACKAGE_DIR)):
        if name.endswith(".py"):
            with open(os.path.join(PACKAGE_DIR, name), "r", encoding="utf-8") as handle:
                yield name, handle.read()


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


# ---------------------------------------------------------------------------
# 1. The optional-dependency contract
# ---------------------------------------------------------------------------

def test_no_module_level_torch_import_anywhere_in_the_package():
    """torch is imported inside the functions that launch, and nowhere else.

    A module-level import would make the coverage predicates unreadable on any
    host without a Mac GPU, which is most of them, and would drag a heavy optional
    dependency into ordinary engine startup.
    """
    offenders = []
    for name, source in _package_sources():
        tree = ast.parse(source)
        for node in tree.body:  # module level only; nested imports are the pattern
            if isinstance(node, ast.Import):
                offenders.extend(f"{name}: import {alias.name}"
                                 for alias in node.names
                                 if alias.name.split(".")[0] == "torch")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
                offenders.append(f"{name}: from {node.module} import ...")
    assert not offenders, f"module-level torch imports: {offenders}"


def test_coverage_and_shaders_import_without_touching_torch():
    """Importing the predicate side must not pull torch in as a side effect."""
    import subprocess
    import sys

    code = ("import sys;"
            "import meep_gpu.metal_kernels.coverage;"
            "import meep_gpu.metal_kernels.shaders;"
            "print('torch' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=root, check=True)
    assert result.stdout.strip() == "False", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# 2. The clause-list equivalence pin
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **kwargs):
        self.xp = np
        self.shape = (4, 3, 2)
        self.dt = 0.05
        self.dx = 0.1
        self.k_point = (0.0, 0.0, 0.0)
        self.has_bloch = False
        self.cylindrical = False
        self.bfast_active = False
        self.beta = 0.0
        self._symmetry = False
        self._mirrored = ()
        self._axis = ()
        self.__dict__.update(kwargs)

    def has_symmetry(self):
        return self._symmetry

    def is_mirrored(self, axis):
        return axis in self._mirrored

    def is_axis(self, axis):
        return axis in self._axis

    def has_metallic(self):
        return False

    def is_metallic(self, axis):
        return False


class _PML:
    def __init__(self, active=True):
        self.is_active = active


class _Fields:
    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.force_complex_fields = False
        self.has_nonlinearity = False
        self.stores_E = True
        self.polarizations = ()
        self.has_polarizations = False
        self.has_offdiagonal_epsilon = False
        self.__dict__.update(kwargs)


#: The configuration matrix both clause lists are run over. Each entry flips ONE
#: thing a clause reads, so a clause missing from either list shows up as a reason
#: present on one side only.
_MATRIX = (
    ("plain", {}, {}),
    ("complex_storage", {}, {"force_complex_fields": True}),
    ("nonlinear", {}, {"has_nonlinearity": True}),
    ("bloch_flag", {"has_bloch": True}, {}),
    ("bloch_kpoint", {"k_point": (0.3, 0.0, 0.0)}, {}),
    ("cylindrical", {"cylindrical": True}, {}),
    ("cylindrical_axis", {"_axis": (0,)}, {}),
    ("symmetry", {"_symmetry": True}, {}),
    ("mirrored_axis", {"_mirrored": (1,)}, {}),
    ("bfast", {"bfast_active": True}, {}),
    ("beta", {"beta": 0.7}, {}),
)


def _triton_backend_clause(reason: str) -> bool:
    """Triton's clause 1, and ONLY it.

    A prefix/suffix match, used HERE and nowhere else. In the implementation a
    match like this would be an inference from a literal and would silently miss a
    clause added later; in the test it is the subtraction the comparison needs,
    and any clause it fails to match stays in the set and fails the assertion —
    which is the behaviour that makes it safe here and unsafe there.
    """
    return reason.startswith("array module is ") and reason.endswith(", not cupy")


@pytest.mark.parametrize("label,grid_kwargs,fields_kwargs", _MATRIX,
                         ids=[entry[0] for entry in _MATRIX])
def test_grid_reasons_match_the_triton_clause_list(label, grid_kwargs, fields_kwargs):
    grid = _Grid(**grid_kwargs)
    fields = _Fields(grid, **fields_kwargs)
    pml = _PML()

    triton_reasons = set(triton_coverage._grid_reasons(fields, pml, grid))
    metal_reasons = set(metal_coverage._grid_reasons(fields, pml, grid))

    triton_rest = {r for r in triton_reasons if not _triton_backend_clause(r)}
    metal_backend = set(metal_coverage._metal_backend_reasons(grid))
    metal_rest = metal_reasons - metal_backend

    assert triton_rest == metal_rest, (
        f"{label}: the two clause lists disagree.\n"
        f"  only Triton has: {sorted(triton_rest - metal_rest)}\n"
        f"  only Metal has:  {sorted(metal_rest - triton_rest)}\n"
        "A clause added to one list and not the other is exactly the drift this "
        "pin exists to catch; hoist the shared body into a backend-neutral module "
        "rather than patching one side.")


def test_the_metal_backend_clause_is_the_only_difference_on_a_numpy_grid():
    """The subtraction the previous test performs must not be vacuous.

    On a NumPy grid Triton refuses ("not cupy") and Metal must refuse for its own
    named reasons — if both sides happened to produce identical strings the
    equivalence test would pass without ever exercising the replacement.
    """
    grid = _Grid()
    fields = _Fields(grid)
    triton_reasons = set(triton_coverage._grid_reasons(fields, _PML(), grid))
    assert any(_triton_backend_clause(r) for r in triton_reasons)
    metal_backend = metal_coverage._metal_backend_reasons(grid)
    # On this host torch and MPS are present, so the only backend clause left is
    # the subnormal-policy one (default resolves to keep on arm64) — or none, if
    # the caller requested flush. Either way it must not be Triton's string.
    assert not any(_triton_backend_clause(r) for r in metal_backend)


# ---------------------------------------------------------------------------
# 3. The single contraction-pragma spelling
# ---------------------------------------------------------------------------

def test_the_contraction_directive_is_spelled_exactly_once():
    """One spelling, in ``shaders``, and no second one anywhere in the package.

    A launch site — or a second source template — that quietly omitted it would
    produce a kernel that is bit-wrong and numerically plausible, which is the
    failure this package's whole gate exists to make impossible to ship.
    """
    occurrences = []
    for name, source in _package_sources():
        for number, line in enumerate(source.splitlines(), start=1):
            if "pragma clang fp" in line:
                occurrences.append(f"{name}:{number}")
    assert occurrences == ["shaders.py:%d" % _pragma_line()], (
        f"the contraction directive is spelled at {occurrences}; it must appear "
        "exactly once, in shaders.py, so no source can be emitted without it")


def _pragma_line() -> int:
    with open(os.path.join(PACKAGE_DIR, "shaders.py"), "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle.read().splitlines(), start=1):
            if "pragma clang fp" in line:
                return number
    raise AssertionError("the contraction directive is not spelled at all")


def test_no_math_mode_pragma_anywhere_in_the_package():
    """A math-mode pragma would re-enable exactly what the contraction guard removes."""
    banned = re.compile(r"pragma\s+(METAL|clang)\s+fp\s+(math_mode|denorm)")
    for name, source in _package_sources():
        assert not banned.search(source), f"{name} carries a math-mode pragma"


def test_every_shipped_source_carries_the_guard_in_off_mode():
    for label, source in shaders.enumerate_sources(shaders.CONTRACT_OFF).items():
        assert shaders.contraction_pragma(shaders.CONTRACT_OFF) in source, label


def test_the_fast_variant_really_differs_from_the_guarded_one():
    """The guard selector must select something. A no-op selector is a vacuous leg."""
    off = shaders.curl_source((0, 0, 0), False, shaders.CONTRACT_OFF)
    fast = shaders.curl_source((0, 0, 0), False, shaders.CONTRACT_FAST)
    assert off != fast
    assert shaders.source_sha256(off) != shaders.source_sha256(fast)


# ---------------------------------------------------------------------------
# The shader specialisation
# ---------------------------------------------------------------------------

def test_curl_source_bakes_the_direction_and_the_ghost_rule():
    forward = shaders.curl_source((shaders.PERIODIC,) * 3, False)
    backward = shaders.curl_source((shaders.PERIODIC,) * 3, True)
    assert "int si = i + 1" in forward and "int si = i - 1" in backward
    assert "(si == nxi) ? 0" in forward and "(si < 0) ? (nxi - 1)" in backward
    walled = shaders.curl_source((shaders.METALLIC, 0, 0), False)
    assert "vx = (si >= 0) && (si < nxi);" in walled
    assert "vx = (si >= 0)" not in forward


def test_the_ownership_mask_follows_the_yee_shift_table():
    """B masks one axis per target; D masks two. Getting this backwards is a plane
    of wrong values at every metallic wall, not a crash."""
    b_mask = shaders.curl_source((shaders.METALLIC,) * 3, False)
    d_mask = shaders.curl_source((shaders.METALLIC,) * 3, True)
    assert b_mask.count("? 0.0f : curl") == 3
    assert d_mask.count("? 0.0f : curl") == 6


def test_the_constitutive_product_puts_d_on_the_left():
    """stepping.py:1011-1013 writes ``source * inverse_epsilon``; float multiply is
    bitwise commutative and a transcription is not a place to rely on that."""
    assert "float src0 = g0[ii] * e0[ii];" in shaders.constitutive_source("E")
    assert "float src0 = g0[ii];" in shaders.constitutive_source("H")


def test_the_two_accumulations_are_never_flattened():
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        assert "a0 = a0 + kp_0 * src0;" in source
        assert "a0 = a0 - km_0 * prev0;" in source
        assert "kp_0 * src0 - km_0 * prev0" not in source


def test_the_curl_grouping_is_never_flattened():
    source = shaders.curl_source((0, 0, 0), False)
    assert "dtdx * ((c_y - c) + (b - b_z))" in source
    assert "dtdx * (c_y - c + b - b_z)" not in source


def test_prev_is_read_before_fw_is_written():
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        assert source.index("float prev0 = w0[ii];") < source.index("w0[ii] = src0;")


def test_an_unknown_boundary_code_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        shaders.curl_source((2, 0, 0), False)
    with pytest.raises(ValueError):
        shaders.curl_source((0, 0), False)
    with pytest.raises(ValueError):
        shaders.constitutive_source("B")
    with pytest.raises(ValueError):
        shaders.contraction_pragma("relaxed")


# ---------------------------------------------------------------------------
# The residency invariant
# ---------------------------------------------------------------------------

_ALL_FIELD_VOLUMES = tuple(sorted({
    name for spec in metal_coverage.SUB_STEP_VOLUMES.values()
    for name in tuple(spec["writes"]) + tuple(spec["reads"])}))


def test_residency_admits_a_fully_planned_step():
    planned = ("step_B", "update_H", "step_D", "update_E")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, planned)
    assert verdict.covered, verdict.reasons


def test_residency_refuses_an_array_path_wall_clear_between_launches():
    """The defect with no Triton analogue: ``zero_metal_B`` writes the HOST Bx
    while the mirror holds the value the curl just computed."""
    planned = ("step_B", "update_H", "step_D", "update_E")
    live = planned + ("zero_metal_B", "zero_metal_D")
    verdict = metal_coverage.residency_coverage(_ALL_FIELD_VOLUMES, planned, live)
    assert not verdict.covered
    assert any("zero_metal_B" in reason for reason in verdict.reasons)


def test_residency_admits_the_same_wall_clear_once_it_is_declared_synced():
    planned = ("step_B", "update_H", "step_D", "update_E")
    live = planned + ("zero_metal_B", "zero_metal_D")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, live, ("zero_metal_B", "zero_metal_D"))
    assert verdict.covered, verdict.reasons


def test_residency_refuses_an_array_path_update_p_reading_the_drive_field():
    """``update_P`` reads ``f_w_*`` under PML (fields.py:1140-1163), which the
    constitutive mirror writes — the one dispersive read that touches it."""
    planned = ("step_B", "update_H", "step_D", "update_E")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, planned + ("update_P",))
    assert not verdict.covered
    assert any("update_P" in reason for reason in verdict.reasons)


def test_residency_refuses_an_undeclared_set_rather_than_assuming_an_empty_one():
    for mirrored, planned, live in ((None, (), ()), ((), None, ()), ((), (), None)):
        verdict = metal_coverage.residency_coverage(mirrored, planned, live)
        assert not verdict.covered
        assert any("not declared" in reason for reason in verdict.reasons)


def test_residency_refuses_a_planned_sub_step_whose_volumes_are_not_mirrored():
    verdict = metal_coverage.residency_coverage(("Bx",), ("step_B",), ("step_B",))
    assert not verdict.covered
    assert any("carry no mirror" in reason for reason in verdict.reasons)


def test_residency_order_covers_every_sub_step_with_a_volume_table():
    assert set(metal_coverage.RESIDENCY_ORDER) == set(metal_coverage.SUB_STEP_VOLUMES)


# ---------------------------------------------------------------------------
# Per-clause refusals on constructed objects
# ---------------------------------------------------------------------------

def test_a_plan_with_no_declared_residency_is_refused_by_name():
    grid = _Grid()
    fields = _Fields(grid)
    verdict = metal_coverage.pml_curl_coverage(fields, _PML(), "step_B", None)
    assert not verdict.covered
    assert any("residency was not declared" in reason for reason in verdict.reasons)


def test_an_inactive_absorber_is_refused_on_both_predicates():
    grid = _Grid()
    fields = _Fields(grid)
    for verdict in (metal_coverage.pml_curl_coverage(fields, _PML(False), "step_B"),
                    metal_coverage.constitutive_coverage(fields, _PML(False), "H")):
        assert any("no active PML layer" in reason for reason in verdict.reasons)


def test_the_e_side_refuses_a_registered_polarization():
    grid = _Grid()
    fields = _Fields(grid, has_polarizations=True)
    verdict = metal_coverage.constitutive_coverage(fields, _PML(), "E")
    assert any("D - sum P" in reason for reason in verdict.reasons)


def test_an_unknown_sub_step_or_side_raises_rather_than_refusing_quietly():
    grid = _Grid()
    fields = _Fields(grid)
    with pytest.raises(ValueError):
        metal_coverage.pml_curl_coverage(fields, _PML(), "step_Q")
    with pytest.raises(ValueError):
        metal_coverage.constitutive_coverage(fields, _PML(), "Q")


# ---------------------------------------------------------------------------
# The subnormal executor column
# ---------------------------------------------------------------------------

def test_keep_is_a_refusal_and_flush_is_admitted(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    reasons = subnormal.mps_policy_reasons()
    assert reasons and "cannot honour" in reasons[0]
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    assert subnormal.mps_policy_reasons() == []


def test_the_policy_report_records_the_absent_disassembly_audit(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    report = subnormal.mps_policy_report()
    assert report["resolved"] == subnormal.FLUSH
    assert report["admitted"] is True
    assert report["ptx_equivalent_audit"] is None
    assert "no disassembly" in report["ptx_equivalent_audit_note"]


def test_the_subnormal_census_counts_bits_not_values():
    band = np.array([1.401298464324817e-45, 0.0, -0.0, 1.0], dtype=np.float32)
    assert subnormal.census(band) == 1
    zeros = subnormal.signed_zero_census(band)
    assert zeros == {"negative_zero": 1, "positive_zero": 1}


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def test_the_checked_in_fingerprints_match_the_tree():
    from meep_gpu.metal_kernels import launch

    stored = launch.load_fingerprints()["metal_kernels"]
    computed = launch.compute_fingerprints()["metal_kernels"]
    assert stored["kernel_source_sha256"] == computed["kernel_source_sha256"], (
        "a shipped kernel source changed without the fingerprint being re-cut; "
        "that is a correctness event, not a formatting one — re-run "
        "launch.write_fingerprints() DELIBERATELY and re-cut the gate")
    assert stored["host_sha256"] == computed["host_sha256"], (
        "a host module changed: it chooses the Yee sub-lattice, binds the "
        "coefficient pointers and decides the specialisation constants")


def test_the_fingerprint_file_pins_a_toolchain_it_measured():
    from meep_gpu.metal_kernels import launch

    with open(launch.fingerprints_path(), "r", encoding="utf-8") as handle:
        stored = json.load(handle)["metal_kernels"]
    assert stored["validated_torch_versions"], "no torch version was pinned"
    frontend = stored["validated_metal_frontend"]
    assert frontend is None or frontend.startswith("metalfe-")


# ---------------------------------------------------------------------------
# The arithmetic, in process, against stepping.py
# ---------------------------------------------------------------------------

def _engine_case(boundaries=("periodic", "periodic", "periodic"), courant=0.35,
                 seed=20260814):
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=courant, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    eps = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(eps, (np.float32(1.0) / eps).astype(np.float32))
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    rng = np.random.default_rng(seed)
    for name in _STATE:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
    return grid, fields, pml


_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(array):
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


@requires_mps
def test_one_complete_cycle_is_byte_identical_to_stepping(monkeypatch):
    """Four sub-steps on the device against four on the array path, in one process.

    uint32 word equality on the targets AND the auxiliaries, and an assertion that
    the step MOVED STATE — a no-op agreeing with a no-op is trivially identical.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu import stepping
    from meep_gpu.metal_kernels import launch

    grid, fields, pml = _engine_case()
    before = {name: np.array(getattr(fields, name), copy=True) for name in _STATE}

    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=())
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.residency.covered, plan.residency.reasons
    for name in plan.replaces:
        plan.plans[name].run()
    residency.sync_out()
    device = {name: np.array(getattr(fields, name), copy=True) for name in _STATE}

    for name, value in before.items():
        getattr(fields, name)[...] = value
    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.update_E(fields, pml)

    moved = sum(int(np.count_nonzero(_words(getattr(fields, n)) != _words(before[n])))
                for n in _STATE)
    assert moved > 0, "VACUOUS: the array path moved no state"
    differing = {n: int(np.count_nonzero(_words(device[n]) != _words(getattr(fields, n))))
                 for n in _STATE}
    assert not {n: c for n, c in differing.items() if c}, differing


@requires_mps
def test_the_residency_mirrors_match_the_host_after_a_synced_step(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=())
    for name in plan.replaces:
        plan.plans[name].run()
    residency.sync_out()
    assert residency.names, "VACUOUS: an empty mirror registry verifies trivially"
    assert residency.verify() == {}


@requires_mps
def test_asking_for_an_unbuilt_contract_variant_raises(monkeypatch):
    """A guard selector that silently fell back to the pinned source would make the
    gate's guard leg certify nothing."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    plan = launch.plan_pml_curl(fields, pml, "step_B", residency)
    with pytest.raises(KeyError):
        plan.run(contract=shaders.CONTRACT_FAST)


@requires_mps
def test_two_plans_share_one_mirror_rather_than_holding_private_copies(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    curl = launch.plan_pml_curl(fields, pml, "step_B", residency)
    constitutive = launch.plan_constitutive(fields, pml, "H", residency)
    assert curl is not None and constitutive is not None
    assert residency.tensor("Bx").data_ptr() == residency.tensor("Bx").data_ptr()
    # step_B writes Bx; update_H reads it. Same allocation or the second launch
    # reads the first one's stale bytes.
    assert any(t.data_ptr() == residency.tensor("Bx").data_ptr()
               for t in curl._args if hasattr(t, "data_ptr"))
    assert any(t.data_ptr() == residency.tensor("Bx").data_ptr()
               for t in constitutive._args if hasattr(t, "data_ptr"))


@requires_mps
def test_the_keep_policy_refuses_a_plan_at_build_time(monkeypatch):
    """The awkward default, pinned: this arm64 host resolves to keep, which MPS
    cannot honour, so a plan built without an explicit flush request is refused."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    verdict = metal_coverage.pml_curl_coverage(fields, pml, "step_B",
                                               launch.Residency())
    assert not verdict.covered
    assert any("cannot honour" in reason for reason in verdict.reasons)
