"""Host checks of the off-diagonal E->P gate's structural nulls (no device).

The gate's ``drive_from_the_stored_E`` mutation drives the recurrence from the stored
E instead of the drive register. It can only differ on cells whose chunk carries the
fused component's OWN-axis layer: everywhere else ``update_E`` stores the drive itself.
On a 2-D grid no chunk carries a z layer, so on the fixture whose weld carries Ez alone
the mutation is the identity, and the full product scored it PARTIAL. The gate now
declares that arm a structural null from the array path's region plan, runs it and
requires it identical, and carries a 3-D fixture where the same single-component weld
has its layer and the mutation must be caught. Which arms are declared, and how the
arms are scored, are host decisions, so both are pinned here.

The fixtures' DOMAIN is a host decision too: every fixture must be a chunk division the
composer can hand the weld (the composer's chunk clause admits it), because the weld's
predicate ends on that clause and a device gate whose fixtures it refuses cannot launch at
all. Each fixture's division, the wrapping division the refusal leg measures, and the
fold's termination sweep are pinned here, without a device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

MUTATION = "drive_from_the_stored_E"


def _load_gate():
    path = HERE / "gate_cuda_dispersive_offdiag_polarization_pair.py"
    spec = importlib.util.spec_from_file_location(
        "gate_cuda_dispersive_offdiag_polarization_pair_host", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def _fused(spec) -> str:
    """The weld's rule (``fused_component``): the FIRST driven component, x, y, z order."""
    return next(name for name in ("Ex", "Ey", "Ez") if name in spec["driven"])


def _declared(gate, spec, component):
    fields, _grid, pml = gate.frame(spec, np)
    return gate.own_layer_absent(fields, pml, component)


def test_the_stored_E_null_is_declared_exactly_where_no_chunk_carries_the_layer(gate):
    declared = {spec["label"]: _declared(gate, spec, _fused(spec)) is not None
                for spec in gate.SPECS}
    assert declared == {spec["label"]: spec["label"] == "one_state_one_component"
                        for spec in gate.SPECS}, declared


def test_a_2d_grid_carries_no_z_layer_and_a_3d_grid_carries_all_three(gate):
    by_label = {spec["label"]: spec for spec in gate.SPECS}
    flat = by_label["one_state_one_component"]
    solid = by_label["one_state_one_component_three_d"]
    assert _declared(gate, flat, "Ez") is not None
    assert _declared(gate, flat, "Ex") is None and _declared(gate, flat, "Ey") is None
    assert all(_declared(gate, solid, name) is None for name in ("Ex", "Ey", "Ez"))


def test_every_fused_component_has_a_fixture_that_arms_the_mutation(gate):
    armed = {}
    for spec in gate.SPECS:
        target = _fused(spec)
        armed[target] = armed.get(target, False) or _declared(gate, spec, target) is None
    assert armed and all(armed.values()), armed


def test_the_structural_null_table_names_must_catch_source_mutations(gate):
    expected = {name: want for name, _rewrite, want, _why in gate.SOURCE_MUTATIONS}
    assert set(gate.STRUCTURAL_NULLS) == {MUTATION}
    assert all(expected[name] == "CAUGHT" for name in gate.STRUCTURAL_NULLS)


def _arm(spec, caught, declared=False):
    arm = {"spec": spec, "caught": caught, "first_divergence": None}
    if declared:
        arm["declared_null"] = "no chunk carries the layer"
    return arm


def test_a_confirmed_declared_null_leaves_the_mutation_caught(gate):
    arms = [_arm(f"live{i}", True) for i in range(7)] + [_arm("flat", False, True)]
    score = gate.score_source_arms(arms)
    assert score["outcome"] == "CAUGHT" and score["armed"] == 7
    assert score["declared_nulls"] == ["flat"] and score["declared_nulls_diverged"] == []


def test_a_declared_null_that_diverges_fails_whatever_the_other_arms_caught(gate):
    arms = [_arm(f"live{i}", True) for i in range(7)] + [_arm("flat", True, True)]
    assert gate.score_source_arms(arms)["outcome"] == "DECLARED NULL DIVERGED"


def test_an_undeclared_uncaught_arm_is_still_partial(gate):
    arms = [_arm(f"live{i}", True) for i in range(6)] + [_arm("flat", False)]
    assert gate.score_source_arms(arms)["outcome"] == "PARTIAL"


def test_a_mutation_whose_every_arm_is_declared_is_not_armed(gate):
    arms = [_arm("flat", False, True),
            {"spec": "absent", "predicted_null": True, "reason": "anchor"}]
    assert gate.score_source_arms(arms)["outcome"] == "NOT ARMED"


# ---------------------------------------------------------------------------
# The domain: every fixture is a division the composer can hand the weld
# ---------------------------------------------------------------------------

def _composer_refusal(gate, spec):
    from meep_gpu.cuda_kernels.arms import StepContext
    from meep_gpu.cuda_kernels.chunk_coverage import CHUNK_AXES_OFFDIAG

    fields, grid, pml = gate.frame(spec, np)
    return StepContext(fields, pml, grid).chunk_division_refusal(CHUNK_AXES_OFFDIAG)


def test_every_fixture_is_a_division_the_composers_chunk_clause_admits(gate):
    refused = {spec["label"]: why for spec in gate.SPECS
               if (why := _composer_refusal(gate, spec)) is not None}
    assert not refused, refused


def test_the_wrapping_division_the_refusal_leg_measures_is_refused_along_x(gate):
    wrapping = dict(gate.SPECS[0], label="refusal_wrapping", boundaries=gate.WRAPPING)
    why = _composer_refusal(gate, wrapping)
    assert why is not None and why.startswith(
        "the chunk division is not a {low, interior, high} product along x"), why


def test_a_wrapping_axis_carries_no_layer_and_the_fold_termination_is_swept(gate):
    """A layered axis is walled; an axis that wraps is in the sweep only bare. The
    folded y both wraps (the periodic termination) and is walled (the metallic one)."""
    terminations = set()
    for spec in gate.SPECS:
        _fields, grid, _pml = gate.frame(spec, np)
        bare = {"xyz".index(name) for name in spec.get("bare", ())}
        for axis in range(3):
            wraps = "periodic" in grid.boundaries[axis]
            if wraps and grid.shape[axis] >= 6:
                assert axis in bare, (spec["label"], axis)
        if grid.is_mirrored(1):
            terminations.add("periodic" if "periodic" in grid.boundaries[1] else "wall")
    assert terminations == {"periodic", "wall"}, terminations
