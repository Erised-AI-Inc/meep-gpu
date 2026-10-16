"""Tests for the conductivity + PML Triton kernel.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware
— bit-identity, the mutation legs, the step budget — lives in the gate
(``parity/meep_gpu/gate_triton_conductivity.py``), because a byte comparison
against the array path is only meaningful on the device that runs it.

What is pinned here instead is everything that decides whether the kernel is ever
ALLOWED to run, plus the two things this kernel is most likely to get wrong in a
way no benchmark would show:

* the coverage predicate, with the mutations that prove it is load-bearing —
  including the over-covering one (a mixed-conductivity run told the kernel is
  uniformly lossy), which is a silent wrong answer and not a crash;
* the FOUR-CASE recurrence itself, checked against
  ``stepping._apply_conductive_pml_update`` bytewise on NumPy. That check is the
  reason "the kernel matches the gate's reference" means "the kernel matches the
  array path", and it belongs at the merge bar rather than on the one machine
  that can launch a kernel;
* the duplicated curl body. This file may not edit ``kernels.py``, so
  ``conductivity.py`` carries its own copy of the ghost rule, the stencil
  grouping and the ownership mask. The two copies have no mechanical link, so a
  test diffs them textually.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import conductivity as conductivity_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.conductivity import (
    CONDUCTIVE_SUB_STEPS,
    DSIG_AXES,
    conductive_pml_curl_coverage,
    conductive_sub_steps,
    conductive_targets,
    plan_conductive_pml_curl,
)

PACKAGE_DIR = pathlib.Path(launch_module.__file__).parent
GATE_PATH = (pathlib.Path(launch_module.__file__).parents[2] / "parity" /
             "meep_gpu" / "gate_triton_conductivity.py")


def build(cell_size=(0.8, 0.8, 0.8), sigma=0.4, side="D", components=None,
          **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy with a conductivity installed.

    ``components`` installs a MAPPING rather than one volume, which is how a
    graded or per-component sigma arrives (``Fields._set_conductivity_side``) and
    the only way to build the mixed configuration the kernel's ``COND`` constexprs
    exist for.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    if sigma is not None:
        volume = np.full(grid.shape, sigma, dtype=np.float32)
        payload = ({name: volume for name in components} if components is not None
                   else volume)
        if side == "D":
            fields.set_d_conductivity(payload)
        else:
            fields.set_b_conductivity(payload)
    return fields, PML(grid=grid, thickness=2)


def reasons_without_backend(fields, pml, sub_step="step_D"):
    return [r for r in conductive_pml_curl_coverage(fields, pml, sub_step).reasons
            if "cupy" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """A missing optional dependency must not break the engine — or this module.

    Importing it, asking it whether a configuration is covered and asking it to
    plan must all work on a machine that has never heard of Triton; only LAUNCHING
    may fail. Planning a refused configuration must return None without ever
    reaching for the kernel module.
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
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.conductivity",
                        raising=False)
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.kernels",
                        raising=False)

    module = importlib.import_module("meep_gpu.triton_kernels.conductivity")
    assert module.triton is None
    fields, pml = build()
    assert module.conductive_pml_curl_coverage(fields, pml, "step_D").reasons
    assert module.plan_conductive_pml_curl(fields, pml, "step_D") is None


# ---------------------------------------------------------------------------
# The tables this module restates, against the ones it must agree with
# ---------------------------------------------------------------------------

def test_the_sub_step_targets_match_the_shipped_launchers():
    """A plan built here and a plan built there must name the same components."""
    for sub_step, targets in CONDUCTIVE_SUB_STEPS.items():
        assert tuple(launch_module.SUB_STEPS[sub_step]["targets"]) == targets


def test_the_dsig_axes_match_the_shipped_ones():
    assert DSIG_AXES == launch_module.DSIG_AXES


def test_the_boundary_codes_match_the_shipped_kernels():
    """Read off the SOURCE, so this holds where Triton cannot be imported."""
    shipped = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    mine = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    for line in ("PERIODIC = tl.constexpr(0)", "METALLIC = tl.constexpr(1)"):
        assert line in shipped and line in mine, line


def _kernel_body(text: str, function: str, start: str, end: str) -> str:
    """The region of one kernel's source between two markers, whitespace-normalised."""
    body = text.split(f"def {function}(", 1)[1]
    body = body.split(start, 1)[1]
    body = body.split(end, 1)[0]
    return "\n".join(line.strip() for line in body.splitlines()
                     if line.strip() and not line.strip().startswith("#"))


def test_the_duplicated_curl_body_still_matches_the_shipped_one():
    """Two transcriptions of the ghost rule, the stencil and the mask, diffed.

    ``conductivity.py`` carries its own copy of ``kernels.pml_curl_step``'s head --
    the ghost rule, the stencil through ``forms.curl_stencil`` with the record's
    presence word, and the ownership mask, the three things most likely to drift.
    Until the two are folded into one kernel behind ``COND`` constexprs, this diff
    is the guard: an edit to either curl body that is not made to both fails here.
    """
    shipped = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    mine = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    a = _kernel_body(shipped, "pml_curl_step", "idx = tl.program_id(0)",
                     "--- split-field recurrence")
    b = _kernel_body(mine, "conductive_pml_curl_step", "idx = tl.program_id(0)",
                     "--- the forms, per component")
    assert a == b, "the conductive kernel's curl body has drifted from kernels.py"


def test_the_coefficient_loads_match_the_shipped_ones():
    """The six per-plane loads, and the axis each one is indexed by."""
    shipped = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    mine = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    for axis, index in (("x", "i"), ("y", "j"), ("z", "k")):
        for stem, table in (("km", "km"), ("si", "sinv")):
            line = (f"{stem}_{axis} = tl.load({table}{axis} + {index}, "
                    f"mask=live, other=0.0)")
            assert line in shipped and line in mine, line


# ---------------------------------------------------------------------------
# The recurrence itself, against stepping.py, on NumPy
# ---------------------------------------------------------------------------

def load_gate():
    """The gate module, by path: it is a dev script, not an installed package."""
    spec = importlib.util.spec_from_file_location("gate_triton_conductivity",
                                                  GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(7, 5, 3), (5, 5, 5)])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_the_gates_reference_is_bit_identical_to_stepping(shape, target):
    """The conductive forms, evaluated as selections, against the array path's.

    The array path computes every form over the whole volume and selects with
    ``copyto``; the kernel evaluates one form per cell. They are the same bits only
    because the forms are mutually independent, and that is a claim to MEASURE
    rather than assert. The memberships are the gate's ``split`` record's lines on
    the component's lattice, with every chunk keeping the conductivity and with the
    ``interior`` conductive-chunk volume (an outer shell of chunks that keeps none).
    Bytewise, never allclose, and at a non-power-of-two Courant as well as 0.5.
    """
    from meep_gpu.stepping import _apply_conductive_pml_update

    gate = load_gate()
    a1, a2 = DSIG_AXES[target]
    axis_names = ("x", "y", "z")
    half = target == 0
    record = gate.split_record(shape)
    coefficients = gate.fit_coefficients(
        np, gate.conductive_coefficients(np, shape, half), record, half, shape)
    pairs = (coefficients["kms_" + axis_names[a1]],
             coefficients["sinv_" + axis_names[a1]],
             coefficients["kms_" + axis_names[a2]],
             coefficients["sinv_" + axis_names[a2]])
    dsig = gate.membership(np, record, half, a1, shape)
    dsigu = gate.membership(np, record, half, a2, shape)
    for dtdx in (0.5, 0.35):
        for kept in (None, gate.interior_volume(shape)):
            rng = np.random.default_rng(20260810 + target)
            field = (rng.standard_normal(shape) * 0.5).astype(np.float32)
            field.reshape(-1)[::7] = np.float32(-0.0)
            fu = (rng.standard_normal(shape) * 0.5).astype(np.float32)
            f_cond = (rng.standard_normal(shape) * 0.5).astype(np.float32)
            curl = ((rng.standard_normal(shape) * 0.5) * dtdx).astype(np.float32)
            curl.reshape(-1)[::5] = np.float32(0.0)
            sigma = (0.4 * (0.3 + rng.random(shape))).astype(np.float32)
            half_dt = np.float32(0.00625)
            condfac = (np.float32(1.0) - sigma * half_dt).astype(np.float32)
            condinv = (np.float32(1.0) / (np.float32(1.0) + sigma * half_dt)
                       ).astype(np.float32)

            got = [field.copy(), fu.copy(), f_cond.copy()]
            _apply_conductive_pml_update(np, got[0], curl, condfac, condinv, *pairs,
                                         got[1], got[2], dsig_active=dsig,
                                         dsigu_active=dsigu, conductive=kept,
                                         scratch=None)
            want = [field.copy(), fu.copy(), f_cond.copy()]
            gate.reference_conductive_recurrence(np, want[0], want[1], want[2], curl,
                                                 condfac, condinv, *pairs, dsig=dsig,
                                                 dsigu=dsigu, kept=kept)
            for name, a, b in zip(("field", "fu", "f_cond"), got, want):
                assert a.tobytes() == b.tobytes(), (
                    f"{name} differs at dtdx={dtdx} kept={kept is not None}: "
                    f"{int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))} "
                    f"of {a.size} floats")
            # More than one form must be reachable here, or the check above is a
            # check of one form wearing several names.
            mix = gate.case_mix(record, shape, "step_B" if half else "step_D",
                                axis_names[a1], axis_names[a2], kept)
            assert sum(1 for count in mix.values() if count) >= 2, mix


def test_the_gates_reference_bites_when_the_grouping_is_flattened():
    """The negative control for the test above, on NumPy alone.

    The increment added in two operations (``((u*km1) + c_new) - c``, the grouping
    the kernel carried before the alignment) and ``(((f*km2) + u_new) - u)``
    flattened to ``((f*km2) + (u_new - u))`` are each a different float32 number. If
    either matched, the byte comparison above would be comparing two spellings of
    the same expression and proving nothing.
    """
    from meep_gpu.stepping import _apply_conductive_pml_update

    gate = load_gate()
    shape = (7, 5, 3)
    record = gate.split_record(shape)
    coefficients = gate.fit_coefficients(
        np, gate.conductive_coefficients(np, shape, False), record, False, shape)
    rng = np.random.default_rng(20260810)
    field = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    fu = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    f_cond = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    curl = ((rng.standard_normal(shape) * 0.5) * 0.35).astype(np.float32)
    sigma = (0.4 * (0.3 + rng.random(shape))).astype(np.float32)
    half_dt = np.float32(0.00625)
    condfac = (np.float32(1.0) - sigma * half_dt).astype(np.float32)
    condinv = (np.float32(1.0) / (np.float32(1.0) + sigma * half_dt)).astype(np.float32)
    kms, sinv = coefficients["kms_y"], coefficients["sinv_y"]
    kms_u, sinv_u = coefficients["kms_z"], coefficients["sinv_z"]
    dsig = gate.membership(np, record, False, 1, shape)
    dsigu = gate.membership(np, record, False, 2, shape)

    got = [field.copy(), fu.copy(), f_cond.copy()]
    _apply_conductive_pml_update(np, got[0], curl, condfac, condinv, kms, sinv, kms_u,
                                 sinv_u, got[1], got[2], dsig_active=dsig,
                                 dsigu_active=dsigu, conductive=None, scratch=None)

    c_new = ((f_cond * condfac) - curl) * condinv
    u_cond = ((fu * condfac) - curl) * condinv
    previous = np.where(dsig, (((fu * kms) + c_new) - f_cond) * sinv, u_cond)
    f_first = ((field * kms) + (c_new - f_cond)) * sinv
    f_direct = ((field * condfac) - curl) * condinv
    wrong_u = np.where(dsigu, (((field * kms_u) + previous) - fu) * sinv_u,
                       np.where(dsig, f_first, f_direct))
    assert got[0].tobytes() != wrong_u.tobytes()

    u_new = np.where(dsig, ((fu * kms) + (c_new - f_cond)) * sinv, u_cond)
    flattened = ((field * kms_u) + (u_new - fu)) * sinv_u   # THE MUTATION
    wrong_f = np.where(dsigu, flattened, np.where(dsig, f_first, f_direct))
    assert got[0].tobytes() != wrong_f.tobytes()


def test_the_auxiliary_loads_sit_under_their_guards():
    """The two declared nulls of KERNELS.md section 7.2, pinned by needle instead.

    An unconditional ``fu`` or ``f_cond`` load moves no stored word (the loaded value
    reaches a store only on the cells its guard admits), so no byte gate can see one;
    it costs the traffic the auxiliary contract exists to save. The same lines in the
    fused electric pair are its helper's, pinned by that module's transcription test.
    """
    text = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    assert text.count("u = tl.load(u_ptr + idx, mask=live & du, other=0.0)") == 1
    assert text.count("c = tl.load(c_ptr + idx, mask=live & ds & kept, other=0.0)") == 1
    assert "tl.load(u_ptr + idx, mask=live, other=0.0)" not in text
    assert "tl.load(c_ptr + idx, mask=live, other=0.0)" not in text


def test_the_forms_are_keyed_on_the_record_never_on_a_coefficient():
    """No form of the kernel tests a coefficient: membership is the record's.

    A cell inside a layer chunk whose own sigma is exactly zero takes its chunk's
    form, which a coefficient test would not give it.
    """
    text = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    source = text[text.index("    def _conductive_component("):
                  text.index("# Coverage\n")]
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "!= 1.0" not in code
    for name in ("ds0", "du0", "ds1", "du1", "ds2", "du2"):
        assert code.count(f"{name} = record_in(cb, lattice + ") == 1, name


# ---------------------------------------------------------------------------
# The plan's bindings: the record and the conductive-chunk volumes
# ---------------------------------------------------------------------------

def _interior_fields(sigma_kind):
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.4, 1.2))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    sigma = np.full(grid.shape, 0.4, dtype=np.float32)
    if sigma_kind == "interior":
        inside = np.zeros(grid.shape, dtype=bool)
        inside[5:-5, 5:-5, 5:-5] = True
        sigma = np.where(inside, sigma, np.float32(0.0)).astype(np.float32)
    fields.set_d_conductivity({"Dx": sigma, "Dz": sigma})
    return fields, PML(grid=grid, thickness=2)


def test_the_conductive_chunk_volumes_follow_the_array_paths_chunks():
    """The plan's third test is the region plan's, per component, or nothing.

    With sigma on an interior block the layer chunks keep none and each conductive
    target binds ``ChunkLayout.conductive_mask`` of the division; with sigma
    everywhere every chunk keeps it and nothing is bound (the array path's
    ``conductive=None``); a lossless target binds nothing either way.
    """
    from meep_gpu.chunk_map import ChunkLayout
    from meep_gpu.fields import IYEE_SHIFTS

    for kind in ("whole", "interior"):
        fields, pml = _interior_fields(kind)
        layout = ChunkLayout.one_rank(fields.grid, pml)
        volumes = conductivity_module.conductive_chunk_volumes(fields, "step_D", None, pml)
        assert volumes[1] is None
        for index, target in ((0, "Dx"), (2, "Dz")):
            if kind == "whole":
                assert volumes[index] is None, target
                continue
            want = layout.conductive_mask(fields.conductivity_for(target),
                                          IYEE_SHIFTS[target])
            assert volumes[index] is not None and np.array_equal(volumes[index], want)
            assert want.any() and not want.all()


def test_the_plan_reads_a_volume_only_for_a_conductive_target_that_carries_one():
    """``CMASK`` is 1 exactly where a conductive target binds a volume."""
    shape = (6, 5, 4)
    arrays = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    volume = np.ones(shape, dtype=bool)
    plan = conductivity_module.ConductivePmlCurlPlan(
        "step_D", shape, 0.5, (0, 0, 0), (1, 1, 0), 256, arrays, arrays,
        [arrays[0], arrays[1], None], [arrays[0], arrays[1], None],
        [arrays[0], arrays[1], None], arrays, [np.ones(n, np.float32)
                                               for n in shape for _ in range(2)],
        chunk_bounds=_every_cell(shape), conductive_chunks=(volume, None, volume))
    assert plan.cmask == (1, 0, 0)
    assert plan._chunks[0].array.dtype == np.uint8  # noqa: SLF001


def _every_cell(shape):
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, RECORD_WORDS, record_index

    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    for axis, n in enumerate(shape):
        if n > 1:
            record[RECORD_PRESENCE_WORD] |= 1 << axis
            for half in (False, True):
                record[record_index(half, axis, 1)] = n
    return record


def test_a_plan_without_a_record_is_refused_by_name():
    """The harness route has no default division (``launch.plan_from_arrays``'s rule)."""
    from meep_gpu.chunk_map import ChunkLayoutRefusal

    shape = (6, 5, 4)
    arrays = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    with pytest.raises(ChunkLayoutRefusal, match="ConductivePmlCurlPlan was given no chunk"):
        conductivity_module.ConductivePmlCurlPlan(
            "step_D", shape, 0.5, (0, 0, 0), (1, 1, 1), 256, arrays, arrays, arrays,
            arrays, arrays, arrays, [np.ones(n, np.float32)
                                     for n in shape for _ in range(2)],
            chunk_bounds=None)
    with pytest.raises(TypeError):
        # chunk_bounds deliberately missing (the expansion keeps the record-binding
        # scan of test_triton_chunk_bounds from counting this refusal as a plan).
        conductivity_module.plan_conductive_from_arrays(
            "step_D", {}, {}, (0, 0, 0), (1, 1, 1), 0.5, **{})


def test_a_conductive_target_whose_volume_cannot_be_read_is_refused(monkeypatch):
    """Which chunks keep a conductivity is read off its volume; no volume, no claim."""
    fields, pml = build()
    monkeypatch.setattr(fields, "conductivity_for", lambda component: None)
    reasons = reasons_without_backend(fields, pml)
    assert any("conductivity_for" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Coverage: the positive half
# ---------------------------------------------------------------------------

def test_a_conductive_D_run_is_covered_for_step_D_only_but_for_the_backend():
    """The configuration the kernel is FOR, and the one it must decline.

    ``2d_cond_pml``'s own shape: ``mp.Medium(epsilon=2.0, D_conductivity=0.4)``,
    no absorber, no ``B_conductivity``. Its ``step_D`` is this kernel's; its
    ``step_B`` is an ordinary lossless PML curl that ``kernels.pml_curl_step``
    already computes, and two kernels claiming one slot is how a composer ends up
    building both.
    """
    fields, pml = build()
    verdict = conductive_pml_curl_coverage(fields, pml, "step_D")
    assert verdict.covered is False
    assert len(verdict.reasons) == 1 and "cupy" in verdict.reasons[0], verdict.reasons

    refused = conductive_pml_curl_coverage(fields, pml, "step_B")
    assert any("no step_B target carries a conductivity" in r
               for r in refused.reasons), refused.reasons


def test_an_absorber_shaped_run_is_covered_on_both_sides():
    """An ``mp.Absorber`` composes a ramp onto BOTH sides, so one body serves both."""
    fields, pml = build()
    fields.set_b_conductivity(np.full(fields.grid.shape, 0.4, dtype=np.float32))
    for sub_step in ("step_B", "step_D"):
        assert reasons_without_backend(fields, pml, sub_step) == []


def test_a_lossless_run_is_refused_by_name():
    fields, pml = build(sigma=None)
    for sub_step in ("step_B", "step_D"):
        reasons = reasons_without_backend(fields, pml, sub_step)
        assert any("kernels.pml_curl_step's" in r for r in reasons), reasons


def test_metallic_boundaries_are_covered_too():
    fields, pml = build(boundaries="metallic")
    assert reasons_without_backend(fields, pml) == []


def test_conductive_sub_steps_reports_the_subset():
    """The shape ``plan_step`` will read: a subset, not a yes/no about the run."""
    fields, pml = build()
    assert conductive_sub_steps(fields, pml) == ()   # NumPy host: nothing covered
    assert conductive_targets(fields, "step_D") == (True, True, True)
    assert conductive_targets(fields, "step_B") == (False, False, False)


def test_a_mixed_per_component_conductivity_is_admitted_and_reported_per_component():
    """RISK 4 in its positive form. ``Dy`` lossless beside two lossy neighbours.

    ``_apply_curl`` reads ``condfac_for`` PER COMPONENT (stepping.py:508-537), so
    this configuration steps ``Dy`` through the plain recurrence and its
    neighbours through the conductive one. A predicate that answered "the run is
    conductive" and a kernel that then applied the conductive branch to all three
    would give ``Dy`` a spurious ``f_cond`` history — smooth, plausible, wrong,
    and no exception.
    """
    fields, pml = build(components=("Dx", "Dz"))
    assert conductive_targets(fields, "step_D") == (True, False, True)
    assert reasons_without_backend(fields, pml) == []
    assert fields.f_cond_Dy is None


def test_the_constexpr_and_the_clause_read_the_same_function(monkeypatch):
    """``conductive_targets`` is the single place the answer is decided.

    Getting the predicate and the compile-time choice out of step is the
    over-covering failure; one function is what makes it impossible.
    """
    fields, pml = build(components=("Dx", "Dz"))
    plan_flags = []
    real = conductivity_module.conductive_targets
    monkeypatch.setattr(conductivity_module, "conductive_targets",
                        lambda f, s: plan_flags.append(real(f, s)) or real(f, s))
    conductive_pml_curl_coverage(fields, pml, "step_D")
    assert plan_flags == [(True, False, True)]


# ---------------------------------------------------------------------------
# Coverage: the refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def test_complex_storage_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    fields.set_d_conductivity(np.full(grid.shape, 0.4, dtype=np.float32))
    reasons = reasons_without_backend(fields, PML(grid=grid, thickness=2))
    assert any("complex" in r for r in reasons), reasons


def test_a_mirror_plane_is_refused():
    fields, pml = build(symmetry=("X",))
    reasons = reasons_without_backend(fields, pml)
    assert any("mirror" in r for r in reasons), reasons


def test_a_nonzero_k_point_is_refused():
    fields, pml = build(k_point=(0.3, 0.0, 0.0))
    reasons = reasons_without_backend(fields, pml)
    assert any("k_point" in r for r in reasons), reasons


def test_no_active_layer_is_refused_because_the_array_path_takes_another_recurrence():
    """Without PML the array path takes ``_apply_conductive_update`` — case D alone.

    Different recurrence, no ``fu``, no ``f_cond``. Claiming it would be a kernel
    stepping histories the engine does not maintain.
    """
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    fields.set_d_conductivity(np.full(grid.shape, 0.4, dtype=np.float32))
    reasons = reasons_without_backend(fields, PML(grid=grid, thickness=0))
    assert any("no active PML" in r for r in reasons), reasons


def test_a_missing_f_cond_is_refused_by_name_rather_than_raised():
    """``_apply_curl`` RAISES on this (stepping.py:516-520); a predicate must not.

    A refusal that arrives as a traceback out of a step the array path would have
    completed is not a refusal, it is a regression.
    """
    fields, pml = build()
    fields.f_cond_Dy = None
    reasons = reasons_without_backend(fields, pml)
    assert any("f_cond_Dy is not allocated" in r for r in reasons), reasons


def test_condfac_without_condinv_is_refused(monkeypatch):
    """Half a coefficient pair is not a covered configuration."""
    fields, pml = build()
    monkeypatch.setattr(fields, "condinv_for",
                        lambda name: None if name == "Dz" else fields._condinv.get(name))
    reasons = reasons_without_backend(fields, pml)
    assert any("both or neither" in r for r in reasons), reasons


def test_a_non_contiguous_conductivity_volume_is_refused():
    fields, pml = build()
    fields._condfac["Dx"] = fields._condfac["Dx"][::-1]
    reasons = reasons_without_backend(fields, pml)
    assert any("condfac[Dx]" in r for r in reasons), reasons


def test_an_unnamed_sub_step_raises():
    fields, pml = build()
    with pytest.raises(ValueError, match="sub_step"):
        conductive_pml_curl_coverage(fields, pml, "update_E")
    with pytest.raises(ValueError, match="sub_step"):
        conductive_targets(fields, "update_E")


def test_bfast_and_special_kz_are_refused():
    """Both add a term to the curl this kernel does not carry — silently."""
    fields, pml = build()
    fields.grid.beta = 0.25
    assert any("beta" in r for r in reasons_without_backend(fields, pml))
    fields.grid.beta = 0.0
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        assert any("BFAST" in r for r in reasons_without_backend(fields, pml))


# ---------------------------------------------------------------------------
# Predicate mutations — the clauses have to be load-bearing
# ---------------------------------------------------------------------------

def test_mutation_admitting_a_lossless_run_is_caught(monkeypatch):
    """Drop the "at least one conductive target" clause and the two kernels collide.

    With it gone a lossless run is covered by BOTH predicates, and a composer that
    asks this one first would step every plain PML run through the conductive
    kernel — which would dereference the placeholder pointers it binds for a
    lossless component.
    """
    fields, pml = build(sigma=None)
    assert reasons_without_backend(fields, pml) != []
    monkeypatch.setattr(conductivity_module, "conductive_targets",
                        lambda f, s: (True, True, True))
    # The mutated predicate no longer refuses for that reason; the f_cond clause
    # catches it instead, which is the SECOND line of defence and is why both exist.
    reasons = reasons_without_backend(fields, pml)
    assert any("f_cond_" in r for r in reasons), reasons


def test_mutation_dropping_the_stored_E_check_is_caught():
    """The invariant behind admitting dispersion, checked rather than inherited."""
    fields, pml = build()
    fields._stored_E = False
    reasons = reasons_without_backend(fields, pml)
    assert any("stored" in r for r in reasons), reasons


def test_mutation_widening_the_boundary_set_by_one_axis_is_caught(monkeypatch):
    """A ghost rule outside the two the kernel writes must disqualify the run."""
    fields, pml = build()
    monkeypatch.setattr(conductivity_module, "_boundary_kinds",
                        lambda grid, layer: ("periodic", "bloch", "metallic"))
    reasons = reasons_without_backend(fields, pml)
    assert any("bloch" in r for r in reasons), reasons


def test_the_predicate_does_not_import_triton_or_cupy():
    """The part that has to be readable on any machine reaches for neither."""
    tree = ast.parse((PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_level if isinstance(node, ast.Import)
             for alias in node.names}
    assert "cupy" not in names
    # Triton is imported at module scope behind a try/except, deliberately: the
    # `@triton.jit` decorator has to run somewhere. It must be the guarded one.
    text = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    assert "except Exception:  # noqa: BLE001 - a missing optional dependency" in text


def test_plan_refuses_before_it_reaches_the_kernel_module(monkeypatch):
    """A refused configuration must not need Triton to be refused."""
    fields, pml = build(sigma=None)

    def explode():
        raise AssertionError("the plan reached for the kernel module on a refusal")

    monkeypatch.setattr(conductivity_module, "_kernel_module", explode)
    assert plan_conductive_pml_curl(fields, pml, "step_D") is None


# ---------------------------------------------------------------------------
# Public experimental composition — ordinary and conductive curls are disjoint
# ---------------------------------------------------------------------------

def test_shared_curl_and_constitutive_coverage_are_split_by_the_conductive_side():
    """D conductivity changes step_D only; the other three ordinary products remain.

    NumPy is the only expected objection to the B curl, H and E. The ordinary D
    curl must refuse by name while the specialized D curl is otherwise admitted.
    """
    fields, pml = build(side="D")
    ordinary_b = coverage_module.pml_curl_coverage(fields, pml, "step_B")
    ordinary_d = coverage_module.pml_curl_coverage(fields, pml, "step_D")
    conductive_d = conductive_pml_curl_coverage(fields, pml, "step_D")
    assert all("cupy" in reason for reason in ordinary_b.reasons), ordinary_b.reasons
    assert any("conductivity" in reason for reason in ordinary_d.reasons), ordinary_d.reasons
    assert all("cupy" in reason for reason in conductive_d.reasons), conductive_d.reasons
    for side in ("H", "E"):
        verdict = coverage_module.constitutive_coverage(fields, pml, side)
        assert all("cupy" in reason for reason in verdict.reasons), verdict.reasons


def _composition_refusal(reason="outside conductivity composition test"):
    return coverage_module.Coverage(False, (reason,))


def test_plan_step_selects_one_ordinary_and_one_conductive_curl(monkeypatch):
    ordinary = {
        "step_B": coverage_module.Coverage(True, ()),
        "step_D": _composition_refusal("ordinary D refuses conductivity"),
    }
    conductive = {
        "step_B": _composition_refusal("conductive B has no lossy target"),
        "step_D": coverage_module.Coverage(True, ()),
    }
    monkeypatch.setattr(
        launch_module, "pml_curl_coverage",
        lambda fields, pml, sub_step: ordinary[sub_step])
    monkeypatch.setattr(
        launch_module, "conductive_pml_curl_coverage",
        lambda fields, pml, sub_step: conductive[sub_step], raising=False)
    monkeypatch.setattr(
        launch_module, "plain_curl_coverage",
        lambda fields, pml, sub_step: _composition_refusal())
    monkeypatch.setattr(
        launch_module, "plan_pml_curl",
        lambda fields, pml, sub_step, block=None, chunk_layout=None:
            f"ordinary:{sub_step}")
    monkeypatch.setattr(
        launch_module, "plan_conductive_pml_curl",
        lambda fields, pml, sub_step, block=None, chunk_layout=None:
            f"conductive:{sub_step}",
        raising=False)
    monkeypatch.setattr(
        launch_module, "constitutive_coverage",
        lambda fields, pml, side: _composition_refusal())
    monkeypatch.setattr(
        launch_module, "dispersive_constitutive_coverage",
        lambda fields, pml: _composition_refusal())

    plan = launch_module.plan_step(type("Fields", (), {"polarizations": ()})(), object())
    assert plan.plans["step_B"] == "ordinary:step_B"
    assert plan.plans["step_D"] == "conductive:step_D"


def test_plan_step_refuses_an_ordinary_conductive_overlap(monkeypatch):
    admitted = coverage_module.Coverage(True, ())
    refused = _composition_refusal()
    monkeypatch.setattr(launch_module, "pml_curl_coverage",
                        lambda fields, pml, sub_step: admitted)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda fields, pml, sub_step: admitted, raising=False)
    monkeypatch.setattr(launch_module, "plain_curl_coverage",
                        lambda fields, pml, sub_step: refused)
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: refused)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda fields, pml: refused)

    plan = launch_module.plan_step(type("Fields", (), {"polarizations": ()})(), object())
    assert "step_B" not in plan.plans and "step_D" not in plan.plans
    assert all(any("ambiguous" in reason for reason in plan.reasons[name])
               for name in ("step_B", "step_D"))


def test_the_public_package_exports_the_conductive_predicate_and_builder_lazily():
    import meep_gpu.triton_kernels as public

    for name in ("conductive_pml_curl_coverage", "plan_conductive_pml_curl"):
        assert name in public.__all__
        assert callable(getattr(public, name))


def test_the_conductivity_composition_fingerprint_names_the_exact_gated_bytes():
    """The standalone kernel product and its public composition form one record."""
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    specialized = record["specialized_kernel_sources"]["conductivity.py"]
    assert specialized["kernel"] == "conductive_pml_curl_step"
    assert specialized["sha256"] == hashlib.sha256(
        pathlib.Path(conductivity_module.__file__).read_bytes()).hexdigest()

    gate = record["conductivity_composition_gate"]
    assert gate["standalone_product_gate"]["single_launch_guarded"].startswith(
        "240/240")
    assert gate["standalone_product_gate"]["mutations"] == "11/11 caught"
    assert "12/12 complete steps" in gate["real_engine_route"]["conductive"]
    assert gate["real_engine_route"]["complete_steps_exact"] == "30/30"
    assert gate["dispatch"].startswith("DISABLED")
