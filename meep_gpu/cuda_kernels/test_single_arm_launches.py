"""The hand-CUDA arms that carry a launch, and the fill twins the merge adopts.

WHAT THIS FILE SETTLES. On 2026-09-27 four families gained a launch-argument
resolution and a launcher -- the three off-diagonal ``update_E`` singles and the
mirror fill over the certified in-seam kernels -- so the CUDA table can compose a
folded run and an off-diagonal D/E seam without a second table. Four questions decide
whether that can be trusted, and each has a section below:

1. **Registry and certification agree.** Every launchable arm names a
   ``CUDA_ARM_CERTIFICATION`` row for its own registry family, the row's ledger key
   resolves, and the entry points the launcher compiles are the kernels that ledger
   entry certified -- and every single-arm row names a launchable arm.
2. **The resolved arguments bind the certified signature and reproduce the array
   path.** The launch goes through the family's REAL wrapper with the device kernel
   replaced by a capture; the captured argument tuple is bound, parameter by
   parameter, to the signature parsed out of the emitted device text, and the bound
   arrays are then run through that family's own NumPy evaluator of the same text
   and byte-compared against ``stepping.update_E``. The fill is checked the same way
   against the three passes the driver runs.
3. **The fill plan is two passes.** Near and far are separate calls around the wall
   wipe, the launch grid is the near pass's, and the pass count agrees with the
   seam's own bookkeeping.
4. **Folded rows are no longer refused whole.** On a real mirror-folded grid the
   composer fills both fill slots, the merge adopts them as a twin pair, and the
   condition rung 6b refuses on (a fill slot with no kernel) no longer holds. The
   twin is never split across two tables.

NOTHING HERE NEEDS A DEVICE. The kernel modules that import CuPy at scope are loaded
against NumPy wearing CuPy's name for the duration of one test and REMOVED from
``sys.modules`` afterwards (the ``meep_gpu/conftest.py`` guard fails a test that
leaves a stand-in behind, and a kernel module bound to the stand-in would be the same
leak one level down). The device kernel itself is replaced by a capture, which is
what makes the argument tuple observable at all.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy
import pytest

from .. import fastpath, fastpath_cuda, stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import arms, dispersive_offdiag_update_e, folded_offdiag_kernels
from . import in_seam_coverage, offdiag_emitter, registry
from .test_dispersive_offdiag_update_e import (
    evaluate_dispersive_offdiag_source, register_poles)
from .test_folded_offdiag import _fold_cell, evaluate_folded_source
from .test_offdiag_constitutive_pml_real import (
    _bits, build, evaluate_emitted_source, install_rows, seed_state)

#: The (family, slot) arms with an established launch, nine of fifty-one.
LAUNCHABLE = {
    ("cuda_curl", "step_B"), ("cuda_curl", "step_D"),
    ("cuda_constitutive", "update_H"), ("cuda_constitutive", "update_E"),
    ("cuda_offdiag", "update_E"), ("cuda_folded_offdiag", "update_E"),
    ("cuda_dispersive_offdiag", "update_E"),
    ("cuda_mirror_fill", "fill_B"), ("cuda_mirror_fill", "fill_D"),
}

#: The row masks the corpus drives (``test_offdiag_constitutive_pml_real``), plus the
#: all-six mask, which is the widest signature the emitter writes.
ROW_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1))


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


@pytest.fixture
def device_modules(monkeypatch):
    """Import CuPy-at-scope kernel modules against the stand-in, then remove them.

    ``monkeypatch.setitem`` puts ``cupy`` back at teardown; this fixture's own
    teardown (which runs FIRST, because it requested ``monkeypatch``) pops every
    module the test imported and every attribute those imports set on the package,
    so no later test can find a kernel module whose ``cp`` is NumPy.
    """
    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    package = sys.modules["meep_gpu.cuda_kernels"]
    modules_before = set(sys.modules)
    attributes_before = set(vars(package))

    def load(stem: str) -> Any:
        return importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")

    yield load
    for name in sorted(set(sys.modules) - modules_before):
        sys.modules.pop(name, None)
    for attribute in sorted(set(vars(package)) - attributes_before):
        delattr(package, attribute)


class _Capture:
    """Stands in for ``_get_kernel(...)``'s ``RawKernel``: records, launches nothing."""

    def __init__(self):
        self.keys: List[Tuple[Any, ...]] = []
        self.calls: List[Tuple[Any, Any, Tuple[Any, ...]]] = []
        self.compiled: List[Tuple[Any, ...]] = []

    def get_kernel(self, *key):
        self.keys.append(key)
        capture = self

        class _Kernel:
            def __call__(self, grid, block, args):
                capture.calls.append((grid, block, tuple(args)))

            def compile(self):
                capture.compiled.append(key)

        return _Kernel()


# ---------------------------------------------------------------------------
# Signature binding
# ---------------------------------------------------------------------------

_PARAMETER = re.compile(r"^(?:const\s+)?(?P<type>float|int)\s*(?P<pointer>\*)?\s*"
                        r"(?:__restrict__\s+)?(?P<name>\w+)$")


def parameters(source: str, kernel: str) -> List[Tuple[str, str]]:
    """``[(name, 'pointer' | 'int' | 'float'), ...]`` in the emitted declaration's order."""
    head = f'extern "C" __global__ void {kernel}('
    start = source.index(head) + len(head)
    text = source[start:source.index(") {", start)]
    out = []
    for raw in text.split(","):
        match = _PARAMETER.match(" ".join(raw.split()))
        assert match, f"unreadable parameter {raw!r} in {kernel}"
        out.append((match["name"], "pointer" if match["pointer"] else match["type"]))
    return out


def bind(signature: List[Tuple[str, str]], args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Bind a captured argument tuple to a parsed signature, checking every type."""
    assert len(args) == len(signature), (
        f"the launch passed {len(args)} arguments to a {len(signature)}-parameter "
        f"signature {[name for name, _ in signature]}")
    bound = {}
    for (name, kind), value in zip(signature, args):
        if kind == "pointer":
            assert isinstance(value, numpy.ndarray), (name, type(value))
            assert value.dtype == numpy.float32, (name, value.dtype)
        elif kind == "int":
            assert isinstance(value, numpy.int32), (name, type(value))
        else:
            assert isinstance(value, numpy.float32), (name, type(value))
        bound[name] = value
    return bound


def registered_arm(family: str, slot: str) -> arms.ArmSpec:
    matches = [spec for spec in arms.registered(slot) if spec.family == family]
    assert len(matches) == 1, (family, slot, matches)
    return matches[0]


# ---------------------------------------------------------------------------
# 1. REGISTRY AND CERTIFICATION
# ---------------------------------------------------------------------------

def test_the_launchable_arms_are_exactly_the_nine():
    context = arms.StepContext(fields=None, pml=None, grid=None)
    launchable = {(spec.family, spec.slot) for spec in arms.registered()
                  if spec.plan(context, spec.slot).launchable}
    assert launchable == LAUNCHABLE


def test_every_launchable_arm_has_a_certification_row_for_its_own_family():
    """Launchable, certified, and certified FOR THE KERNELS THE LAUNCHER COMPILES."""
    context = arms.StepContext(fields=None, pml=None, grid=None)
    ledger = fastpath_cuda.fingerprints()
    blocks = fastpath_cuda.certification()
    for spec in arms.registered():
        plan = spec.plan(context, spec.slot)
        if not plan.launchable:
            continue
        label = fastpath_cuda.namespaced(spec.label)
        assert label in fastpath_cuda.CUDA_ARM_CERTIFICATION, label
        family, key = fastpath_cuda.CUDA_ARM_CERTIFICATION[label]
        assert family == spec.family, (label, family, spec.family)
        entry = ledger.get(key)
        assert isinstance(entry, dict), f"{label} names {key}, not in fingerprints"
        assert entry.get("status") == "PASS", (key, entry.get("status"))
        narrative = str((entry.get("_notes") or {}).get("_narrative_lives_in") or "")
        assert narrative.rsplit(":", 1)[-1] in blocks, (key, narrative)
        compiles = plan.launch_kernel._ENTRY_POINTS[spec.slot]  # noqa: SLF001
        compiles = (compiles,) if isinstance(compiles, str) else tuple(compiles)
        missing = set(compiles) - set(entry.get("kernels") or ())
        assert not missing, f"{label} launches {sorted(missing)}, which {key} " \
                            f"did not certify ({entry.get('kernels')})"


def test_every_single_arm_certification_row_names_a_launchable_registered_arm():
    """The converse: a single-arm row that no launchable arm reads is a row for a
    launch that does not exist, and the dispatch record would quote it anyway."""
    context = arms.StepContext(fields=None, pml=None, grid=None)
    launchable = {(spec.family, fastpath_cuda.namespaced(spec.label))
                  for spec in arms.registered()
                  if spec.plan(context, spec.slot).launchable}
    for label, (family, _key) in fastpath_cuda.CUDA_ARM_CERTIFICATION.items():
        if fastpath_cuda.is_fused(label):
            continue
        assert (family, label) in launchable, (label, family)


def test_the_new_certification_rows_point_at_their_own_byte_gates():
    rows = fastpath_cuda.CUDA_ARM_CERTIFICATION
    assert rows["cuda:off-diagonal"] == ("cuda_offdiag", "offdiag_2026-08-16")
    assert rows["cuda:folded off-diagonal"] == (
        "cuda_folded_offdiag", "cuda_folded_offdiag_2026-08-21")
    assert rows["cuda:dispersive off-diagonal"] == (
        "cuda_dispersive_offdiag", "cuda_dispersive_offdiag_2026-08-20")
    assert rows["cuda:mirror fill"] == ("cuda_mirror_fill", "in_seam_2026-08-21")
    # A certification row is what the device-identity rung reads its architecture
    # from, so a new row must not widen the validated list.
    assert fastpath_cuda.validated_compute_capabilities() == ("8.6",)


def test_the_launcher_entry_points_are_the_modules_own_kernel_names():
    """Compiled BY ENTRY POINT, and each entry point is the module's own name."""
    assert arms.launch_offdiag._ENTRY_POINTS == {  # noqa: SLF001
        "update_E": offdiag_emitter.KERNEL_NAME}
    assert arms.launch_folded_offdiag._ENTRY_POINTS == {  # noqa: SLF001
        "update_E": folded_offdiag_kernels.KERNEL_NAME}
    assert arms.launch_dispersive_offdiag._ENTRY_POINTS == {  # noqa: SLF001
        "update_E": dispersive_offdiag_update_e.KERNEL_NAME}
    text = (pathlib.Path(arms.__file__).parent / "in_seam_passes.py").read_text(
        encoding="utf-8")
    declared = set(re.findall(r'extern "C" __global__ void (\w+)\(', text))
    for slot, pair in arms.launch_mirror_fill._ENTRY_POINTS.items():  # noqa: SLF001
        assert set(pair) <= declared, (slot, pair)
        assert pair == (f"fill_symmetry_{slot[-1]}", f"fill_folded_far_{slot[-1]}")


def test_the_fill_arm_is_the_seventh_shape_and_the_split_plan():
    for slot in ("fill_B", "fill_D"):
        spec = registered_arm("cuda_mirror_fill", slot)
        assert spec.shape == "fill_by_family"
        assert spec.predicate_name == "cuda_kernels.arms.covers_mirror_fill"
        plan = spec.plan(arms.StepContext(fields=None, pml=None, grid=None), slot)
        assert isinstance(plan, arms.CudaMirrorFillPlan)
        assert callable(plan.run_near) and callable(plan.run_far)
        # A LABEL, not an entry point: the slot is two kernels.
        assert plan.kernel_label not in arms.launch_mirror_fill._ENTRY_POINTS[slot]  # noqa: SLF001


def test_no_driver_slot_is_left_without_an_arm():
    assert arms.NO_ARM_REASONS == {}
    for slot in arms.STEP_ORDER:
        assert arms.registered(slot), f"{slot} has no registered arm"


# ---------------------------------------------------------------------------
# 2. THE OFF-DIAGONAL LAUNCHES, BOUND AND RUN
# ---------------------------------------------------------------------------

#: family -> (kernel module stem, kernel name, emitted-source builder, evaluator)
_OFFDIAG = {
    "cuda_offdiag": ("offdiag_constitutive_kernels", offdiag_emitter.KERNEL_NAME),
    "cuda_folded_offdiag": ("folded_offdiag_kernels",
                            folded_offdiag_kernels.KERNEL_NAME),
    "cuda_dispersive_offdiag": ("dispersive_offdiag_update_e",
                                dispersive_offdiag_update_e.KERNEL_NAME),
}


def _offdiag_configuration(family: str, mask, symmetry: Tuple[str, ...]):
    """A plain-NumPy frozen state the family's wrapper accepts, seeded away from zero."""
    planes = tuple(Mirror(name.upper(), 1) for name in symmetry)
    fields, layer, grid = build(numpy, boundaries=("periodic",) * 3,
                                symmetry=planes, cell=_fold_cell(symmetry))
    install_rows(fields, grid, mask, seed=20260927)
    if family == "cuda_dispersive_offdiag":
        register_poles(fields, grid, (1, 1, 1), seed=20260927)
    seed_state(fields, grid, seed=27)
    return fields, layer, grid


def _emitted(family: str, key: Tuple[Any, ...]) -> str:
    if family == "cuda_offdiag":
        return offdiag_emitter.offdiag_source(key[0])
    if family == "cuda_folded_offdiag":
        return folded_offdiag_kernels.folded_offdiag_source(key[0])
    return dispersive_offdiag_update_e.dispersive_offdiag_source(key[0], key[1])


def _evaluate(family: str, source: str, bound: Dict[str, Any], shape) -> None:
    arrays = {name: value for name, value in bound.items()
              if isinstance(value, numpy.ndarray)
              and not name.startswith(("kps_", "kms_"))}
    tables = {name: value for name, value in bound.items()
              if name.startswith(("kps_", "kms_"))}
    codes = tuple(int(bound[f"bc_{axis}"]) for axis in "xyz")
    walls = tuple(int(bound[f"wm_{axis}"]) for axis in "xyz")
    if family == "cuda_offdiag":
        evaluate_emitted_source(source, arrays, tables, codes, walls, shape)
        return
    weights = tuple(float(bound[f"gw_{axis}"]) for axis in "xyz")
    if family == "cuda_folded_offdiag":
        evaluate_folded_source(source, arrays, tables, codes, walls, weights, shape)
    else:
        evaluate_dispersive_offdiag_source(source, arrays, tables, codes, walls,
                                           weights, shape)


_OFFDIAG_CASES = [
    ("cuda_offdiag", mask, ()) for mask in ROW_MASKS
] + [
    ("cuda_folded_offdiag", mask, ("y",)) for mask in ROW_MASKS
] + [
    ("cuda_folded_offdiag", ROW_MASKS[0], ("x", "z")),
    ("cuda_dispersive_offdiag", ROW_MASKS[0], ("y",)),
    ("cuda_dispersive_offdiag", ROW_MASKS[0], ()),
]


@pytest.mark.parametrize("family,mask,symmetry", _OFFDIAG_CASES,
                         ids=[f"{f}-{''.join(map(str, m))}-{''.join(s) or 'unfolded'}"
                              for f, m, s in _OFFDIAG_CASES])
def test_the_resolved_launch_binds_the_signature_and_reproduces_update_E(
        family, mask, symmetry, device_modules, monkeypatch):
    """The registered plan's ONE launch, through the family's real wrapper.

    The argument tuple is bound to the emitted declaration parameter by parameter;
    the bound arrays are the ``Fields``' own objects; and the certified source, run
    over exactly those bound arguments by the family's own evaluator, leaves the
    same bytes ``stepping.update_E`` does from the same frozen state.
    """
    fields, layer, grid = _offdiag_configuration(family, mask, symmetry)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    stepping.update_E(fields, layer)
    oracle = _bits(fields)
    assert any(oracle[name] != frozen[name].tobytes() for name in oracle), \
        "the oracle moved nothing, so every tree would agree with it"
    for name, values in frozen.items():
        getattr(fields, name)[...] = values

    stem, kernel = _OFFDIAG[family]
    module = device_modules(stem)
    capture = _Capture()
    monkeypatch.setattr(module, "_get_kernel", capture.get_kernel)

    context = arms.StepContext(fields, layer, grid)
    plan = registered_arm(family, "update_E").plan(context, "update_E")
    assert plan.launchable
    report = plan.run()
    assert len(capture.calls) == 1
    launch_grid, block, args = capture.calls[0]
    source = _emitted(family, capture.keys[0])
    bound = bind(parameters(source, kernel), args)

    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez", "Dx", "Dy", "Dz"):
        assert bound[name] is getattr(fields, name), name
    count = int(fields.Ex.size)
    threads = int(block[0])
    assert launch_grid == ((count + threads - 1) // threads,)
    assert report["blocks"] == launch_grid[0] and plan.launch_grid == launch_grid

    _evaluate(family, source, bound, tuple(grid.shape))
    assert _bits(fields) == oracle


@pytest.mark.parametrize("family,mask,symmetry", _OFFDIAG_CASES[:1] + _OFFDIAG_CASES[2:3]
                         + _OFFDIAG_CASES[5:6],
                         ids=["unfolded", "folded", "dispersive"])
def test_the_resolution_is_the_wrappers_own_pml_door(family, mask, symmetry,
                                                     device_modules, monkeypatch):
    """The resolver calls the derivations the wrapper calls when handed ``pml``, so
    the two launches bind the same values in every parameter."""
    fields, layer, grid = _offdiag_configuration(family, mask, symmetry)
    stem, kernel = _OFFDIAG[family]
    module = device_modules(stem)
    capture = _Capture()
    monkeypatch.setattr(module, "_get_kernel", capture.get_kernel)
    launcher = {"cuda_offdiag": arms.launch_offdiag,
                "cuda_folded_offdiag": arms.launch_folded_offdiag,
                "cuda_dispersive_offdiag": arms.launch_dispersive_offdiag}[family]
    wrapper = getattr(module, launcher._WRAPPER)  # noqa: SLF001

    wrapper(fields, layer)
    registered_arm(family, "update_E").plan(
        arms.StepContext(fields, layer, grid), "update_E").run()
    assert len(capture.calls) == 2 and capture.keys[0] == capture.keys[1]
    signature = parameters(_emitted(family, capture.keys[0]), kernel)
    through_pml = bind(signature, capture.calls[0][2])
    resolved = bind(signature, capture.calls[1][2])
    for name, _kind in signature:
        left, right = through_pml[name], resolved[name]
        if isinstance(left, numpy.ndarray):
            assert left.shape == right.shape and left.tobytes() == right.tobytes(), name
        else:
            assert left == right, (name, left, right)


def test_the_arguments_are_resolved_once_and_the_pole_chain_on_every_launch(
        device_modules, monkeypatch):
    """Tables, codes, walls and weights are frozen configuration; the pole chain is
    not -- ``PolarizationState.update`` rotates its buffers every step."""
    fields, layer, grid = _offdiag_configuration("cuda_dispersive_offdiag",
                                                 ROW_MASKS[0], ("y",))
    module = device_modules("dispersive_offdiag_update_e")
    capture = _Capture()
    monkeypatch.setattr(module, "_get_kernel", capture.get_kernel)
    resolutions = []
    spec = registered_arm("cuda_dispersive_offdiag", "update_E")
    plan = spec.plan(arms.StepContext(fields, layer, grid), "update_E")
    real = plan.resolve_launch_args

    def counting(context, slot):
        resolutions.append(slot)
        return real(context, slot)

    plan.resolve_launch_args = counting
    plan.run()
    state = fields.polarizations[0]
    rotated = numpy.array(state.P["Ex"], copy=True)
    state.P["Ex"] = rotated
    plan.run()
    assert resolutions == ["update_E"], "the configuration was resolved more than once"
    signature = parameters(_emitted("cuda_dispersive_offdiag", capture.keys[1]),
                           dispersive_offdiag_update_e.KERNEL_NAME)
    second = bind(signature, capture.calls[1][2])
    assert second["P_Ex_0"] is rotated, "the launch bound last step's pole buffer"


def test_the_dispersive_launcher_refuses_a_resolved_pole_chain():
    with pytest.raises(ValueError, match="resolved per launch"):
        arms.launch_dispersive_offdiag(object(), "update_E", {"plan": {}})


def test_the_off_diagonal_warm_compiles_the_specialisation_the_launch_uses(
        device_modules, monkeypatch):
    fields, layer, grid = _offdiag_configuration("cuda_offdiag", ROW_MASKS[1], ())
    module = device_modules("offdiag_constitutive_kernels")
    capture = _Capture()
    monkeypatch.setattr(module, "_get_kernel", capture.get_kernel)
    plan = registered_arm("cuda_offdiag", "update_E").plan(
        arms.StepContext(fields, layer, grid), "update_E")
    assert plan.warm() is None
    plan.run()
    assert capture.compiled == [capture.keys[-1]]


def test_a_launcher_that_warms_from_the_context_is_handed_it():
    seen = []

    class Launcher:
        WARMS_FROM_CONTEXT = True

        def __call__(self, fields, slot, arguments):
            return {"launched": True, "blocks": 1}

        def warm(self, slot, context):
            seen.append((slot, context.fields))
            return None

    context = arms.StepContext(fields="F", pml=None, grid=None)
    plan = arms.plan_factory("probe", "x", "k", lambda c, s: {}, Launcher())(
        context, "update_E")
    assert plan.warm() is None and seen == [("update_E", "F")]
    bare = arms.CudaSlotPlan("probe", "update_E", "x", "k", lambda c, s: {},
                             launch=Launcher())
    assert "no composition context" in bare.warm()


# ---------------------------------------------------------------------------
# 3. THE FILL: TWO PASSES, BOUND AND RUN AROUND THE WALL WIPE
# ---------------------------------------------------------------------------

def _folded(axes: str = "", phases=(), boundaries=None, extent: float = 2.0,
            xp_module: Any = numpy):
    """A small folded grid, as ``test_fused_magnetic_pair_fill_carry.build`` makes it."""
    size = [2.4, 2.4, 1.6]
    for name in axes:
        size["XYZ".index(name)] = float(extent)
    grid = Grid(resolution=4.0, cell_size=tuple(size), dimensions=3, courant=0.35,
                symmetry=tuple(Mirror(name, int(phase))
                               for name, phase in zip(axes, phases)),
                boundaries=boundaries, xp=xp_module)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    folded = {"XYZ".index(name) for name in axes}
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if axis in folded else (2, 2)
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness), grid


FOLDS = (
    ("y_periodic_even", {"axes": "Y", "phases": (1,)}),
    ("y_periodic_odd", {"axes": "Y", "phases": (-1,), "extent": 2.25}),
    ("y_metallic", {"axes": "Y", "phases": (1,), "boundaries": {"y": "metallic"}}),
    ("y_periodic_wall_z", {"axes": "Y", "phases": (1,),
                           "boundaries": {"z": "metallic"}}),
    ("xy_periodic_mixed_phase", {"axes": "XY", "phases": (1, -1)}),
    ("xy_mixed_termination", {"axes": "XY", "phases": (1, 1),
                              "boundaries": {"y": "metallic"}}),
    ("xyz_periodic", {"axes": "XYZ", "phases": (1, -1, 1)}),
)

#: The statements the emulator below transcribes, pinned against the device text so
#: the emulator cannot drift from what the certified kernel does.
_PINNED = {
    "fill_symmetry_B": ("f[base] = phase * f[base + 2 * stride];",),
    "fill_symmetry_D": ("g0[base] = phase * g0[base + 2 * stride];",
                        "g1[base] = phase * g1[base + 2 * stride];"),
    "fill_folded_far_B": ("float parity = -phase;",
                          "g0[base + last * stride] = parity * g0[base + reflect_row * stride];",
                          "g1[base + last * stride] = parity * g1[base + reflect_row * stride];"),
    "fill_folded_far_D": ("float parity = -phase;",
                          "f[base + last * stride] = parity * f[base + reflect_row * stride];"),
}


def _face(axis: int, row: int):
    index = [slice(None)] * 3
    index[axis] = row
    return tuple(index)


def _emulate(kernel: str, bound: Dict[str, Any]) -> None:
    """One in-seam launch in float32 NumPy, from the BOUND arguments alone."""
    family = kernel[-1]
    arrays = [bound[f"{family}{c}"] for c in "xyz"]
    axis = int(bound["axis"])
    phase = numpy.float32(bound["phase"])
    if kernel.startswith("fill_symmetry"):
        targets = ([arrays[axis]] if family == "B"
                   else [arrays[c] for c in range(3) if c != axis])
        for target in targets:
            target[_face(axis, 0)] = phase * target[_face(axis, 2)]
        return
    last = (int(bound["nx"]), int(bound["ny"]), int(bound["nz"]))[axis] - 1
    parity = numpy.float32(-phase)
    row = int(bound["reflect_row"])
    targets = ([arrays[c] for c in range(3) if c != axis] if family == "B"
               else [arrays[axis]])
    for target in targets:
        target[_face(axis, last)] = parity * target[_face(axis, row)]


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
@pytest.mark.parametrize("slot", ("fill_B", "fill_D"))
def test_the_fill_launch_reproduces_the_drivers_three_passes(
        label, kwargs, slot, device_modules, monkeypatch):
    """near fill (kernel) -> wall wipe (array path) -> far fill (kernel), against
    ``fill_symmetry_bc_*`` -> ``zero_metal_*`` -> ``fill_folded_far_ghosts_*``."""
    family = slot[-1]
    names = tuple(f"{family}{c}" for c in "xyz")
    fields, layer, grid = _folded(**kwargs)
    rng = numpy.random.default_rng(927)
    for name in names:
        getattr(fields, name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(numpy.float32)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy() for name in names}
    getattr(stepping, f"fill_symmetry_bc_{family}")(fields)
    getattr(stepping, f"zero_metal_{family}")(fields)
    getattr(stepping, f"fill_folded_far_ghosts_{family}")(fields)
    oracle = {name: numpy.asarray(getattr(fields, name)).tobytes() for name in names}
    assert any(oracle[name] != frozen[name].tobytes() for name in names)
    for name, values in frozen.items():
        getattr(fields, name)[...] = values

    passes = device_modules("in_seam_passes")
    sources = passes.device_sources()
    for kernel, statements in _PINNED.items():
        for statement in statements:
            assert statement in sources[kernel], (kernel, statement)
    launched: List[str] = []

    def get_kernel(name):
        signature = parameters(sources[name], name)

        def kernel(grid_, block, args):
            launched.append(name)
            _emulate(name, bind(signature, tuple(args)))

        return kernel

    monkeypatch.setattr(passes, "_get_kernel", get_kernel)
    plan = registered_arm("cuda_mirror_fill", slot).plan(
        arms.StepContext(fields, layer, grid), slot)
    assert plan.launchable
    near = plan.run_near()
    getattr(stepping, f"zero_metal_{family}")(fields)
    far = plan.run_far()

    produced = {name: numpy.asarray(getattr(fields, name)).tobytes() for name in names}
    assert produced == oracle
    near_list = in_seam_coverage.plan("fill_symmetry", grid)
    far_list = in_seam_coverage.plan("fill_folded_far", grid)
    assert launched == ([f"fill_symmetry_{family}"] * len(near_list)
                        + [f"fill_folded_far_{family}"] * len(far_list))
    assert near["launches"] == len(near_list) >= 1
    assert far["launches"] == len(far_list)
    assert plan.launches == 2 and plan.launch_grid == (near["blocks"],)


def test_the_fill_plans_launch_grid_is_the_near_pass_and_run_is_near_then_far():
    calls = []

    class Launcher:
        def run_half(self, fields, slot, arguments, half):
            calls.append(half)
            return {"launched": half == "near", "launches": int(half == "near"),
                    "blocks": 5 if half == "near" else 0}

        def __call__(self, fields, slot, arguments):
            raise AssertionError("a split plan is never launched whole")

    plan = arms.plan_factory("probe", "mirror fill", "k", lambda c, s: {"x": 1},
                             Launcher(), plan_class=arms.CudaMirrorFillPlan)(
        arms.StepContext(fields="F", pml=None, grid=None), "fill_B")
    assert plan.launch_grid is None
    plan.run_near()
    plan.run_far()
    assert plan.launch_grid == (5,), "a far pass that launched nothing cleared it"
    assert plan.launches == 2
    plan.run()
    assert calls == ["near", "far", "near", "far"]


def test_the_fill_arm_refuses_an_unfolded_grid_by_name(xp):
    fields, layer, grid = _folded(xp_module=xp)
    plan = arms.plan_step(fields, layer, grid)
    for slot in ("fill_B", "fill_D"):
        assert slot not in plan.plans
        reason, = plan.reasons[slot]
        assert "not mirror-folded" in reason, reason


def test_the_fill_arm_refuses_complex_storage_for_its_own_reason(xp):
    """No standalone complex fill kernel exists; the in-seam predicate names why."""
    size = (2.4, 2.0, 1.6)
    grid = Grid(resolution=4.0, cell_size=size, dimensions=3, courant=0.35,
                symmetry=(Mirror("Y", 1),), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = PML(grid=grid, thickness=tuple((0, 0) if grid.shape[a] < 6
                                           else (0, 2) if a == 1 else (2, 2)
                                           for a in range(3)))
    plan = arms.plan_step(fields, layer, grid)
    for slot in ("fill_B", "fill_D"):
        assert slot not in plan.plans
        assert any("complex64" in reason for reason in plan.reasons[slot]), \
            plan.reasons[slot]
        assert not any("not mirror-folded" in reason
                       for reason in plan.reasons[slot])


# ---------------------------------------------------------------------------
# 4. THE MERGE, AND THE FOLD RUNG'S CONDITION
# ---------------------------------------------------------------------------

def _merged_alone(plan) -> Tuple[Any, Dict[str, Any]]:
    record: Dict[str, Any] = {}
    merged = fastpath_cuda.merge_tables(
        [("cuda", plan)], pending_of={"cuda": {}}, record=record)
    return merged, record["arbitration"]["refused"]["cuda"]


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_a_folded_real_run_composed_by_this_table_alone_covers_both_fills(
        label, kwargs, xp):
    """What rung 6b refused whole until 2026-09-27: a fold whose curls and
    constitutives are on this table's kernels and whose fills had no CUDA arm."""
    fields, layer, grid = _folded(xp_module=xp, **kwargs)
    plan = arms.plan_step(fields, layer, grid)
    assert plan.selected.get("fill_B") == plan.selected.get("fill_D") == "mirror fill"
    assert {"fill_B", "fill_D"} <= set(plan.launchable)
    merged, refused = _merged_alone(plan)
    assert merged.selected["fill_B"] == merged.selected["fill_D"] == "cuda:mirror fill"
    # THE RUNG'S OWN CONDITION (fastpath.py rung 6b): a fold is refused whole when
    # any far-fill-owning slot is not dispatchable while another slot is.
    assert fastpath._fold_description(grid) is not None  # noqa: SLF001
    assert not set(fastpath.FAR_FILL_PASSES) - set(merged.plans)
    assert not {slot for slot in refused if slot in fastpath_cuda.FILL_ARM_TWINS}
    # And the curl/constitutive seams went with it, whole.
    assert {"step_B", "update_H", "step_D", "update_E"} <= set(merged.plans)


def test_a_folded_off_diagonal_run_adopts_its_whole_d_seam_and_both_fills(xp):
    fields, layer, grid = build(xp, symmetry=(Mirror("Y", 1),),
                                cell=_fold_cell(("y",)))
    install_rows(fields, grid, ROW_MASKS[0], seed=20260927)
    plan = arms.plan_step(fields, layer, grid)
    assert plan.selected["update_E"] == "folded off-diagonal"
    merged, refused = _merged_alone(plan)
    assert merged.selected["step_D"] == "cuda:PML"
    assert merged.selected["update_E"] == "cuda:folded off-diagonal"
    assert merged.selected["fill_B"] == merged.selected["fill_D"] == "cuda:mirror fill"
    assert refused == {}


def test_an_unfolded_off_diagonal_run_adopts_its_whole_d_seam(xp):
    fields, layer, grid = build(xp)
    install_rows(fields, grid, ROW_MASKS[1], seed=20260927)
    plan = arms.plan_step(fields, layer, grid)
    assert plan.selected["update_E"] == "off-diagonal"
    assert "update_E" in plan.launchable
    merged, refused = _merged_alone(plan)
    assert merged.selected["step_D"] == "cuda:PML"
    assert merged.selected["update_E"] == "cuda:off-diagonal"
    assert refused == {}


class _Fill:
    """A fill plan as the composer installs one: launchable, split, runnable."""

    launchable = True

    def __init__(self, far: bool = True):
        self.run_near = lambda *a, **k: {"launched": True, "blocks": 1}
        if far:
            self.run_far = lambda *a, **k: {"launched": True, "blocks": 1}
        self.run = lambda *a, **k: None


def _cuda_fills(fill_B=None, fill_D=None):
    plans = {slot: plan for slot, plan in (("fill_B", fill_B), ("fill_D", fill_D))
             if plan is not None}
    return types.SimpleNamespace(plans=plans,
                                 selected={slot: "mirror fill" for slot in plans},
                                 reasons={})


def _triton_holding(*slots):
    return types.SimpleNamespace(
        plans={slot: types.SimpleNamespace(run=lambda *a, **k: None) for slot in slots},
        selected={slot: "mirror fill" for slot in slots}, reasons={})


def _merge(primary, secondary):
    record: Dict[str, Any] = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", primary), ("cuda", secondary)],
        pending_of={"triton": {}, "cuda": {}}, record=record)
    return merged, record["arbitration"]["refused"]["cuda"]


def test_both_twins_are_adopted_when_no_other_table_holds_either():
    merged, refused = _merge(_triton_holding(), _cuda_fills(_Fill(), _Fill()))
    assert merged.selected == {"fill_B": "cuda:mirror fill",
                               "fill_D": "cuda:mirror fill"}
    assert merged.backends == {"fill_B": "cuda", "fill_D": "cuda"}
    assert refused == {}


def test_a_twin_whose_partner_another_table_holds_is_refused_by_name():
    """The two-table fold, refused even though this half's own slot is free."""
    merged, refused = _merge(_triton_holding("fill_B"), _cuda_fills(_Fill(), _Fill()))
    assert merged.selected == {"fill_B": "mirror fill"}
    assert merged.backends == {"fill_B": "triton"}
    assert "fill_D" not in merged.plans
    reason = refused["cuda:mirror fill (fill_D)"]
    assert "its twin fill_B is held by triton:mirror fill" in reason


def test_twins_both_held_by_the_incumbent_stay_with_it():
    merged, refused = _merge(_triton_holding("fill_B", "fill_D"),
                             _cuda_fills(_Fill(), _Fill()))
    assert merged.backends == {"fill_B": "triton", "fill_D": "triton"}
    assert "held by triton:mirror fill" in refused["cuda:mirror fill"]


def test_a_lone_fill_is_never_adopted_without_its_twin():
    merged, refused = _merge(_triton_holding(), _cuda_fills(fill_B=_Fill()))
    assert merged.plans == {}
    assert "not an adoptable twin" in refused["fill_B"]


def test_a_fill_without_a_far_half_is_not_an_adoptable_twin():
    """``run`` alone would be driven as one launch, in front of the wall wipe."""
    merged, refused = _merge(_triton_holding(),
                             _cuda_fills(_Fill(far=False), _Fill()))
    assert merged.plans == {}
    assert set(refused) == {"fill_B", "fill_D"}


def test_a_fill_label_without_a_certification_row_is_not_adoptable():
    plan = _cuda_fills(_Fill(), _Fill())
    plan.selected = {"fill_B": "uncertified fill", "fill_D": "uncertified fill"}
    merged, refused = _merge(_triton_holding(), plan)
    assert merged.plans == {}
    assert set(refused) == {"fill_B", "fill_D"}


def test_the_seam_rule_is_unchanged_for_the_off_diagonal_singles():
    """A lone launchable update_E single is still refused without its step_D."""
    single = types.SimpleNamespace(launchable=True, run=lambda *a, **k: None)
    plan = types.SimpleNamespace(plans={"update_E": single},
                                 selected={"update_E": "off-diagonal"}, reasons={})
    merged, refused = _merge(_triton_holding(), plan)
    assert merged.plans == {}
    assert "seam partner" in refused["update_E"]
