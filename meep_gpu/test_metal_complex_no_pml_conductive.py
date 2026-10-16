"""Host contracts for Metal's complex/Bloch no-PML conductive curl family."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import (
    RECORD_PRESENCE_WORD,
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
)
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, chunk_bounds, shaders
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.metal_kernels import complex_no_pml_conductive as family
from meep_gpu.pml import PML


class HostResidency:
    #: Where a plan uploads its chunk record (``chunk_bounds.tensor``): the host.
    device = "cpu"

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, sides=("B", "D"), complex_storage=True, bloch=True, seed=57):
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries="periodic", k_point=(0.2, -0.125, 0.0) if bloch
                else (0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            real = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            imag = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            array[...] = real + 1j * imag if complex_storage else real
    sigma = np.full(grid.shape, np.float32(0.18))
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    return fields, PML(grid=grid, thickness=0)


def record_of(fields, pml):
    """The record the composer hands a plan for this run: the one-rank division."""
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, expected_targets):
    def run(*args):
        targets, condfac, condinv = args[:3], args[6:9], args[9:12]
        assert tuple(targets) == tuple(expected_targets)
        for index, name in enumerate(family.SUB_STEPS[sub_step]["targets"]):
            if actual.condfac_for(name) is None:
                assert condfac[index] is targets[index]
                assert condinv[index] is targets[index]
            else:
                assert condfac[index] is actual.condfac_for(name)
                assert condinv[index] is actual.condinv_for(name)
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, family.SUB_STEPS[sub_step]["targets"]):
            target[...] = getattr(reference, name)
    return run


def _unresolvable(grid, pml):  # noqa: ARG001
    """What the array path does with a grid it will not resolve.

    ``coverage._boundary_kinds`` turns exactly this into ``None`` (coverage.py:105-121),
    which is the input the predicate has to REFUSE rather than crash on.
    """
    raise RuntimeError("the array path will not resolve this grid")


def test_source_keeps_bloch_rotation_and_three_ordered_complex_passes():
    """The plane-0 connection of the unshifted partners, and the three ordered passes.

    ``stepping._bloch_ghost_operand`` replaces the unshifted partner on plane 0 of a
    phased axis (kind 2, ``CONNECT_PHASE``: MEEP's unfused product; kind 1,
    ``CONNECT_NEGATE``: a sign flip per part), and the conductive tail is
    ``_apply_conductive_update``'s three passes, one product per part.
    """
    source = family.complex_conductive_no_pml_curl_source(
        (0, 0, 0), False, (2, 1, 0), (True, False, True))
    assert "device float2*" in source
    assert "    if (i == 0) {\n        c_dx = c_mul(c, px);\n        b_dx = c_mul(b, px);\n    }" in source
    assert "    if (j == 0) {\n        c_dy = -c;\n        a_dy = -a;\n    }" in source
    assert "    // axis z: no connection" in source
    assert "float2 value0 = f0[ii];" in source
    assert "value0 = c_mul_field_left(value0, cf0[ii]);" in source
    assert "value0 = value0 - curl0;" in source
    assert "value0 = c_mul_field_left(value0, ci0[ii]);" in source
    assert "f1[ii] = f1[ii] - curl1;" in source
    assert "[[buffer(20)]]" in source and "[[buffer(21)]]" not in source


@pytest.mark.parametrize(("sides", "expected"), [
    (("B",), {"step_B": (True, True, True), "step_D": (False, False, False)}),
    (("D",), {"step_B": (False, False, False), "step_D": (True, True, True)}),
])
def test_complex_per_target_conductivity_is_split_by_curl_side(
        backend_ready, sides, expected):
    fields, pml = build(sides=sides)
    for slot in family.SUB_STEPS:
        plan = family.plan_metal_complex_conductive_no_pml_curl(
            fields, pml, slot, HostResidency(),
            functions={shaders.CONTRACT_OFF: lambda *args: None},
            chunk_bounds=record_of(fields, pml))
        assert plan is not None
        assert plan.conductive == expected[slot]


def test_complex_bloch_two_curl_cycles_match_the_array_path(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    plan_b = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_B", residency,
        chunk_bounds=record_of(actual, actual_pml), functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", [actual.Bx, actual.By, actual.Bz])})
    plan_d = family.plan_metal_complex_conductive_no_pml_curl(
        actual, actual_pml, "step_D", residency,
        chunk_bounds=record_of(actual, actual_pml), functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", [actual.Dx, actual.Dy, actual.Dz])})
    assert plan_b is not None and plan_d is not None
    for _ in range(2):
        plan_b.run()
        plan_d.run()
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert np.array_equal(getattr(actual, name).view(np.uint32),
                              getattr(reference, name).view(np.uint32))
    assert plan_b.launches == plan_d.launches == 2


def test_conductive_step_d_binds_derived_h_through_the_live_b_mirrors(backend_ready):
    fields, pml = build()
    residency = HostResidency()
    plan = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_D", residency,
        functions={shaders.CONTRACT_OFF: lambda *args: None},
            chunk_bounds=record_of(fields, pml))
    assert plan is not None
    assert {"Bx", "By", "Bz"}.issubset(residency.arrays)
    assert not {"Hx", "Hy", "Hz"}.intersection(residency.arrays)
    assert plan.volumes == ("Dx", "Dy", "Dz", "Bx", "By", "Bz")


def test_complex_mixed_target_flags_bind_only_the_lossy_components(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.18))
    fields.set_b_conductivity({"Bx": None, "By": sigma, "Bz": None})
    fields.set_d_conductivity({"Dx": sigma, "Dy": None, "Dz": None})
    b = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
            chunk_bounds=record_of(fields, pml))
    d = family.plan_metal_complex_conductive_no_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
            chunk_bounds=record_of(fields, pml))
    assert b is not None and d is not None
    assert b.conductive == (False, True, False)
    assert d.conductive == (True, False, False)


def test_an_unreadable_boundary_rule_is_a_refusal_and_not_a_crash(monkeypatch):
    """UNREADABLE IS A REFUSAL, NEVER A CRASH — measured on an otherwise REAL grid.

    ``coverage._boundary_kinds`` answers ``None`` for a grid the array path will not
    resolve, and ``complex_no_pml_conductive._base_reasons`` — the clause list this
    family shares with the other complex no-PML curls, the off-diagonal E and the
    fused conductive pair — reads that triple TWICE: once for the covered ghost rule
    and once for per-axis Bloch legality. The grid here is real, so ``bloch_phase``
    is callable and the SECOND read is genuinely reached; a guard on the first read
    alone would still raise here.
    """
    fields, pml = build()
    monkeypatch.setattr(stepping, "_boundary_kinds", _unresolvable)
    verdict = family.metal_complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency())
    assert verdict.covered is False
    assert any("the per-axis boundary rule is unreadable" in reason
               for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "complex64 storage"), ("active_pml", "active PML"),
    ("lossless", "no curl target"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(sides=(() if mutation == "lossless" else ("B", "D")))
    if mutation == "real":
        fields, pml = build(complex_storage=False, bloch=False)
    elif mutation == "active_pml":
        pml = PML(grid=fields.grid, thickness=1)
    verdict = family.metal_complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_B", HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_after_direct_and_complete_step_evidence():
    for slot in family.SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == "complex conductive no-PML curl"
        assert arm.wired


def test_native_gate_is_incremental_and_arms_conductive_failure_modes():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_complex_no_pml_conductive.py")
    # The stencil, connection and presence rows live in the module both complex
    # no-PML curl gates import, so the gate's armed text is the two files together.
    source = gate.read_text(encoding="utf-8") + gate.with_name(
        "complex_no_pml_gate_common.py").read_text(encoding="utf-8")
    assert "flush=True" in source and "os.fsync" in source
    assert "condfac_condinv_swapped" in source
    assert "target_store_dropped" in source
    assert "previous_curl_grouping" in source
    assert "bloch_x_connection_dropped" in source


# ---------------------------------------------------------------------------
# MEEP's stencil, the plane-0 connection and the record (bit_parity 0.10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward", (False, True))
def test_the_stencil_and_connection_are_the_complex_pml_curls_own_lines(backward):
    """One stencil in the package: each statement is a line the PML curl emits."""
    reference = shared_predicate.bloch_curl_source((0, 0, 0), backward, (2, 1, 0))
    source = family.complex_conductive_no_pml_curl_source(
        (0, 0, 0), backward, (2, 1, 0), (False, False, False))
    assert family.CONNECTED_REGISTERS in reference
    assert family.CONNECTED_REGISTERS in source
    for statement in family.CURL_STATEMENTS:
        assert statement in reference
        assert statement in source
    for axis in "xyz":
        for kind in shared_predicate.CONNECT_KINDS:
            block = shared_predicate._phase_block(axis, kind)
            if kind == shared_predicate.CONNECT_NONE:
                continue
            emitted = family.complex_conductive_no_pml_curl_source(
                (0, 0, 0), backward,
                tuple(kind if a == axis else 0 for a in "xyz"),
                (False, False, False))
            assert block in emitted
    # The previous spellings are gone: the pairwise bracket and the shifted-read
    # rotation at the wrap.
    assert "((c_y - c) + (b - b_z))" not in source
    assert "wx ?" not in source and "wy ?" not in source


def test_the_record_binds_last_with_a_presence_only_bounds_block():
    source = family.complex_conductive_no_pml_curl_source(
        (1, 0, 1), True, (0, 2, 0), (True, True, False))
    assert chunk_bounds.binding(family.RECORD_BUFFER) + "," in source
    assert chunk_bounds.struct_source() in source
    assert source.count(chunk_bounds.BLOCK_BEGIN) == 1
    block = source.split(chunk_bounds.BLOCK_BEGIN, 1)[1].split(chunk_bounds.BLOCK_END, 1)[0]
    for axis in "xyz":
        assert f"const bool has_{axis} = " in block
    assert "in_i_" not in block and "in_h_" not in block
    # Below the decode prologue, so a lift cut there carries the block with it.
    assert source.index("    int nyz = nyi * nzi;\n") < source.index(chunk_bounds.BLOCK_BEGIN)


def test_an_unknown_connection_kind_is_refused_at_emit_time():
    with pytest.raises(ValueError, match="connection kind 3"):
        family.complex_conductive_no_pml_curl_source(
            (0, 0, 0), False, (3, 0, 0), (False, False, False))
    with pytest.raises(ValueError, match="carries a phase but is not periodic"):
        family.complex_conductive_no_pml_curl_source(
            (1, 0, 0), False, (1, 0, 0), (False, False, False))


def test_the_builder_takes_the_record_with_no_default_and_the_arm_hands_it_over(
        monkeypatch):
    import inspect

    parameter = inspect.signature(
        family.plan_metal_complex_conductive_no_pml_curl).parameters["chunk_bounds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    seen = {}

    def builder(fields, pml, slot, residency, variants, *, chunk_bounds):
        seen["chunk_bounds"] = chunk_bounds

    monkeypatch.setattr(family, "plan_metal_complex_conductive_no_pml_curl", builder)

    class _Context:
        fields = pml = residency = None
        contract_variants = (shaders.CONTRACT_OFF,)
        extra = {}
        chunk_bounds = object()

    family._arm_plan(_Context, "step_D")
    assert seen["chunk_bounds"] is _Context.chunk_bounds


@pytest.mark.parametrize("defect", ("missing", "contradicting"))
def test_a_bad_record_is_refused_by_name_before_anything_compiles(
        backend_ready, monkeypatch, defect):
    def refuse(*_args, **_kwargs):
        raise AssertionError("compiled before the record was checked")

    monkeypatch.setattr(family, "compile_source", refuse)
    fields, pml = build()
    record = None
    match = "no chunk record"
    if defect == "contradicting":
        record = np.array(record_of(fields, pml), copy=True)
        record[RECORD_PRESENCE_WORD] &= ~4  # z marked invariant on a 3-D grid
        match = "marks z invariant"
    residency = HostResidency()
    with pytest.raises(ChunkLayoutRefusal, match=match):
        family.plan_metal_complex_conductive_no_pml_curl(
            fields, pml, "step_B", residency, chunk_bounds=record)
    assert residency.arrays == {}


def test_the_no_absorber_record_is_presence_alone():
    for cell, dimensions, presence in (((1.0, 0.875, 0.75), 3, 7),
                                       ((1.0, 0.875, 0.0), 2, 3),
                                       ((0.0, 0.0, 2.0), 1, 4)):
        grid = Grid(resolution=8.0, cell_size=cell, courant=0.3, xp=np,
                    dimensions=dimensions)
        words = chunk_bounds.plan_words(
            chunk_bounds_record(ChunkLayout.one_rank(grid, PML(grid=grid, thickness=0))),
            grid.shape, "test")
        assert int(words[RECORD_PRESENCE_WORD]) == presence
        assert not words[:RECORD_PRESENCE_WORD].any()


# --- the kernel's per-cell text, transcribed on the host, against the array path ---

_F32 = np.float32


def _parts(value):
    return (np.ascontiguousarray(value.real, dtype=_F32),
            np.ascontiguousarray(value.imag, dtype=_F32))


def _shifted(part, axis, backward, metallic):
    if not metallic:
        return np.roll(part, 1 if backward else -1, axis)
    out = np.zeros_like(part)
    n = part.shape[axis]
    to, take = [slice(None)] * 3, [slice(None)] * 3
    to[axis], take[axis] = ((slice(1, n), slice(0, n - 1)) if backward
                            else (slice(0, n - 1), slice(1, n)))
    out[tuple(to)] = part[tuple(take)]
    return out


def _kernel_text_on_host(sources, targets, codes, kinds, phases, backward, dtdx,
                         presence, condfac, condinv, previous_grouping=False):
    """The emitted statements, per part, in float32: ghost reads, the plane-0
    connection, MEEP's bracket and its absent-partner forms, the mask, the tails."""
    a, b, c = sources
    shifted = {name: tuple(_shifted(p, axis, backward, codes[axis] == 1) for p in value)
               for name, value, axis in (("a_y", a, 1), ("a_z", a, 2), ("b_x", b, 0),
                                         ("b_z", b, 2), ("c_x", c, 0), ("c_y", c, 1))}
    connected = {"c_dy": c, "b_dz": b, "a_dz": a, "c_dx": c, "b_dx": b, "a_dy": a}
    by_axis = {0: (("c_dx", c), ("b_dx", b)), 1: (("c_dy", c), ("a_dy", a)),
               2: (("b_dz", b), ("a_dz", a))}
    for axis, kind in enumerate(kinds):
        if kind == shared_predicate.CONNECT_NONE:
            continue
        plane = tuple(0 if a_ == axis else slice(None) for a_ in range(3))
        pr, pi = _F32(phases[axis][0]), _F32(phases[axis][1])
        for name, (zr, zi) in by_axis[axis]:
            nr, ni = zr.copy(), zi.copy()
            if kind == shared_predicate.CONNECT_NEGATE:
                nr[plane], ni[plane] = -zr[plane], -zi[plane]
            else:
                nr[plane] = (zr[plane] * pr) - (zi[plane] * pi)
                ni[plane] = (zr[plane] * pi) + (zi[plane] * pr)
            connected[name] = (nr, ni)
    d = _F32(dtdx)

    def form(g1s, g1, g2, g2s, first, second):
        out = []
        for k in range(2):
            if first and second:
                total = ((g1s[k] - g1[k]) + (g2[k] - g2s[k]) if previous_grouping
                         else ((g1s[k] - g1[k]) + g2[k]) - g2s[k])
                out.append(d * total)
            elif first:
                out.append(d * (g1s[k] - g1[k]))
            elif second:
                out.append((-d) * (g2s[k] - g2[k]))
            else:
                out.append(np.zeros_like(g1[k]))
        return out

    hx, hy, hz = presence
    curls = [form(shifted["c_y"], connected["c_dy"], connected["b_dz"], shifted["b_z"],
                  hy, hz),
             form(shifted["a_z"], connected["a_dz"], connected["c_dx"], shifted["c_x"],
                  hz, hx),
             form(shifted["b_x"], connected["b_dx"], connected["a_dy"], shifted["a_y"],
                  hx, hy)]
    pairs = (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)) if backward
             else ((0, 0), (1, 1), (2, 2)))
    for target, axis in pairs:
        if codes[axis] == 1:
            plane = tuple(0 if a_ == axis else slice(None) for a_ in range(3))
            for k in range(2):
                curls[target][k][plane] = _F32(0.0)
    stepped = []
    for target in range(3):
        fr, fi = targets[target]
        if condfac[target] is None:
            stepped.append((fr - curls[target][0], fi - curls[target][1]))
            continue
        vr, vi = fr * condfac[target], fi * condfac[target]
        vr, vi = vr - curls[target][0], vi - curls[target][1]
        stepped.append((vr * condinv[target], vi * condinv[target]))
    return stepped


_HOST_CASES = (
    ("3d_phase_xy", (1.0, 0.875, 0.75), 3, (0.2, -0.125, 0.0), None, ()),
    ("3d_edge_x", (1.0, 0.875, 0.75), 3, (0.5, 0.0, 0.0), None, ("B", "D")),
    ("3d_metallic_xz", (1.0, 0.875, 0.75), 3, (0.0, 0.125, 0.0),
     {"x": "metallic", "z": "metallic"}, ("B", "D")),
    ("3d_one_cell_y", (1.25, 0.0625, 0.75), 3, (0.2, 0.0, 0.0), None, ()),
    ("2d_phase", (1.0, 0.875, 0.0), 2, (0.2, -0.125, 0.0), None, ("D",)),
    ("2d_edge_y", (1.0, 1.0, 0.0), 2, (0.0, 0.5, 0.0), None, ()),
    ("2d_metallic_x", (1.0, 0.875, 0.0), 2, (0.0, 0.3, 0.0), {"x": "metallic"}, ("B",)),
    ("1d_phase_z", (0.0, 0.0, 2.0), 1, (0.0, 0.0, 0.3), None, ()),
)


def _host_case(cell, dimensions, k_point, boundaries, sides, planted):
    extra = {} if boundaries is None else {"boundaries": boundaries}
    grid = Grid(resolution=8.0, cell_size=cell, courant=0.3, k_point=k_point, xp=np,
                dimensions=dimensions, **extra)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    sigma = np.broadcast_to(np.linspace(_F32(0.06), _F32(0.22), grid.shape[0],
                                        dtype=_F32)[:, None, None], grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": sigma})
    if "D" in sides:
        fields.set_d_conductivity({"Dx": None, "Dy": sigma, "Dz": sigma})
    rng = np.random.default_rng(29)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        if planted and name[0] in "BD":
            array.real[...] = -0.0   # f = -0 per part, equal stencil neighbours
            array.imag[...] = -0.0
        elif planted:
            array[...] = np.complex64(0.25 - 0.125j)
        else:
            array[...] = (rng.uniform(-0.3, 0.3, grid.shape).astype(_F32)
                          + 1j * rng.uniform(-0.3, 0.3, grid.shape).astype(_F32))
    return fields, PML(grid=grid, thickness=0)


def _host_compare(fields, pml, previous_grouping=False):
    grid = fields.grid
    resolved = stepping._boundary_kinds(grid, None)
    codes = tuple(1 if kind == "metallic" else 0 for kind in resolved)
    presence = tuple(not grid.is_invariant(axis) for axis in range(3))
    differing = words = 0
    for sub_step in ("step_B", "step_D"):
        spec = family.SUB_STEPS[sub_step]
        backward = bool(spec["backward"])
        kinds, phases = shared_predicate.phase_arguments(
            shared_predicate.bloch_phase_table(grid, resolved), backward=backward)
        accessor = getattr(fields, family.SOURCE_ACCESSOR[sub_step])
        sources = [_parts(np.array(accessor(name))) for name in spec["sources"]]
        targets = [_parts(np.array(getattr(fields, name))) for name in spec["targets"]]
        flags = family.conductive_no_pml_targets(fields, sub_step)
        condfac = [np.asarray(fields.condfac_for(name), dtype=_F32) if flags[index]
                   else None for index, name in enumerate(spec["targets"])]
        condinv = [np.asarray(fields.condinv_for(name), dtype=_F32) if flags[index]
                   else None for index, name in enumerate(spec["targets"])]
        host = _kernel_text_on_host(sources, targets, codes, kinds, phases, backward,
                                    grid.dt / grid.dx, presence, condfac, condinv,
                                    previous_grouping)
        getattr(stepping, sub_step)(fields, pml)
        for index, name in enumerate(spec["targets"]):
            for reference, emulated in zip(_parts(getattr(fields, name)), host[index]):
                differing += int(np.count_nonzero(
                    reference.view(np.uint32) != emulated.view(np.uint32)))
                words += reference.size
    return differing, words


@pytest.mark.parametrize("planted", (False, True))
@pytest.mark.parametrize("case", _HOST_CASES, ids=[case[0] for case in _HOST_CASES])
def test_the_kernel_text_steps_the_array_paths_words_on_the_host(case, planted):
    """The emitted forms, per part, reproduce stepping.step_B / step_D word for word.

    3-D, 2-D and 1-D grids (presence from invariance; a 3-D grid storing one cell on
    a live axis), unphased, phased (``CONNECT_PHASE``) and Brillouin-edge
    (``CONNECT_NEGATE``) axes, metallic walls, per-target conductivity, and a planted
    state with ``f = -0`` per part and equal stencil neighbours.
    """
    _, cell, dimensions, k_point, boundaries, sides = case
    fields, pml = _host_case(cell, dimensions, k_point, boundaries, sides, planted)
    differing, words = _host_compare(fields, pml)
    assert words > 0
    assert differing == 0, (differing, words)


def test_the_previous_pairwise_grouping_is_a_different_word_on_the_host():
    """The inverted mutation (KERNELS.md §7.1) is visible to this comparison."""
    _, cell, dimensions, k_point, boundaries, sides = _HOST_CASES[0]
    fields, pml = _host_case(cell, dimensions, k_point, boundaries, sides, False)
    differing, words = _host_compare(fields, pml, previous_grouping=True)
    assert differing > 0, (differing, words)
