"""MEEP's per-chunk forms through the DISPERSIVE fused electric pair's emitted text.

The real twin's word-level test (``test_fused_pair_chunk_forms.py``) runs the fused
electric text through the root interpreter on its divisions -- walls, a slab, a split
division, folds on one, two and three axes, a deep fold one-rank and lifted -- against
the array path's ``step_D -> fill_symmetry_bc_D -> zero_metal_D ->
fill_folded_far_ghosts_D -> update_E``. This runs the DISPERSIVE family's text on the
same cases with live susceptibilities: the twin's curl and fill carry, and the
constitutive half's source ``(D - sum P) * inv_eps`` formed at each cell the
constitutive helper writes (the own cell, and every imaged ghost with the DESTINATION's
poles), with MEEP's form chosen per chunk from the record.

Every stored word of the state is compared, the pole volumes are required unchanged
(the E half reads them and must not write them), and a component with no contributor
(the ``(2, 0, 3)`` arity) reads D directly. Planted defects -- the 0.9.2 grouping, the
select removed, the auxiliary stored on a plain cell, the integer record -- must
each change a word; the commuted product must not. (The plain form spelt as the
accumulation with unit coefficients differs only on a ``-0`` source, which this text
forms from the curl's register and a fixture cannot plant; ``test_dispersive_chunk_forms``
plants it through the stand-alone kernel, whose constitutive helper this text splices.)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy
import pytest

from ..dispersion import PolarizationState, Susceptibility
from . import dispersive_fused_electric_pair as dfep
from . import dispersive_kernels
from .test_fused_pair_chunk_forms import (  # noqa: F401 - ``certified`` is a fixture
    CASES, STATE, F32, ScatterInterpreter, _Installed, _bits, _restore, _snapshot,
    arguments, build, certified, oracle)

COMPONENTS = ("Ex", "Ey", "Ez")

#: Every component driven twice; one component with no contributor (it reads D
#: directly); one pole each, where every chain order agrees.
ARITIES: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2), (2, 0, 3), (1, 1, 1))


class _InstalledDispersive(_Installed):
    """The twin's certified modules, plus the constitutive module this family checks."""

    def __enter__(self):
        super().__enter__()
        self.held.append((dfep, "constitutive_kernels", dfep.constitutive_kernels))
        dfep.constitutive_kernels = self.certified["constitutive_kernels"]
        return self


def _text(certified, arity, edit=None) -> str:
    with _InstalledDispersive(certified):
        text = dfep.dispersive_fused_electric_pair_source(arity)
    if edit is not None:
        old, new = edit
        assert text.count(old) >= 1, old
        text = text.replace(old, new)
    return text


def _with_poles(case: str, plant: bool, arity, seed: int = 11):
    """The twin's seeded state, with ``max(arity)`` live susceptibilities registered.

    A per-component sigma of zero is the engine's own spelling of "this term does not
    drive that component" (``PolarizationState`` filters it), so the arity is built
    from real objects. The pole volumes are seeded nonzero -- zero is a fixed point of
    ``s - P`` -- and carry planted ``-0`` words when the case plants them in the fields.
    """
    fields, grid, pml = build(case, plant, True)
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    kinds = ("lorentzian", "drude")
    for index in range(max(arity)):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(COMPONENTS)}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    for state in fields.polarizations:
        for store in (state.P, state.P_prev):
            for array in store.values():
                values = rng.uniform(-0.6, 0.6, size=shape).astype(F32)
                if plant:
                    values[rng.uniform(size=shape) < 0.25] = F32(-0.0)
                array[...] = values
    return fields, grid, pml


def _poles(fields) -> Dict[str, numpy.ndarray]:
    return {f"P{index}_{component}": numpy.array(state.P[component], copy=True)
            for index, state in enumerate(fields.polarizations)
            for component in state.driven()}


def run(certified, case: str, plant: bool, arity, edit=None):
    """``(start, oracle, kernel, poles before, poles after, form counts)``."""
    fields, grid, pml = _with_poles(case, plant, arity)
    start = _snapshot(fields)
    poles_before = _poles(fields)
    oracle("D", fields, pml)
    want = _snapshot(fields)
    _restore(fields, start)
    plan = fields.region_plan
    named = arguments("D", certified, fields, grid, pml)
    chain = dispersive_kernels.resolve_pole_plan(fields)
    for component in COMPONENTS:
        for index, array in enumerate(chain[component]):
            named[f"P_{component}_{index}"] = numpy.ascontiguousarray(array).reshape(-1)
    ScatterInterpreter(_text(certified, arity, edit)).launch(
        named, int(numpy.prod(grid.shape)))
    got = _snapshot(fields)
    shape = tuple(int(n) for n in grid.shape)
    counts = {}
    for component in COMPONENTS:
        inside = numpy.broadcast_to(plan.mask("constitutive", component, numpy), shape)
        counts[component] = (int(inside.sum()), int((~inside).sum()))
    return start, want, got, poles_before, _poles(fields), counts


def _differing(want, got) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(_bits(want[name]) != _bits(got[name])))
            for name in want}


PARAMETERS = ([(case, plant, (2, 2, 2)) for case in CASES for plant in (False, True)]
              + [(case, plant, arity) for case in ("walls3d", "fold_xy_mixed",
                                                   "split_y3d")
                 for plant in (False, True) for arity in ARITIES[1:]])


@pytest.mark.parametrize("case,plant,arity", PARAMETERS)
def test_the_dispersive_fused_text_is_the_array_path_word_for_word(case, plant, arity,
                                                                    certified):
    start, want, got, before, after, counts = run(certified, case, plant, arity)
    differing = {name: count for name, count in _differing(want, got).items() if count}
    assert not differing, (case, plant, arity, differing)
    assert not {name: count for name, count in _differing(before, after).items()
                if count}, "the E half wrote a pole volume it only reads"
    moved = sum(_differing(start, want)[name] for name in COMPONENTS)
    assert moved > 0, "update_E moved no E word: the case measured nothing"
    assert all(inside + outside > 0 for inside, outside in counts.values())


def test_the_cases_carry_both_forms_on_every_component(certified):
    """The sweep reaches both of MEEP's forms on each component somewhere, so a select
    defect has a cell to show on."""
    reached = {component: [False, False] for component in COMPONENTS}
    for case in CASES:
        *_unused, counts = run(certified, case, False, (1, 1, 1))
        for component, (inside, outside) in counts.items():
            reached[component][0] |= inside > 0
            reached[component][1] |= outside > 0
    assert all(all(pair) for pair in reached.values()), reached


#: name -> (edit, case, plant, arity, must change a word)
DEFECTS: Dict[str, Tuple[Tuple[str, str], str, bool, Tuple[int, int, int], bool]] = {
    # THE INVERTED MUTATION: the 0.9.2 package grouping where MEEP's ships.
    "package_grouping": (("(f[idx] + ((kps * src) - (kms * prev)))",
                          "((f[idx] + (kps * src)) - (kms * prev))"),
                         "walls3d", False, (2, 2, 2), True),
    # The select removed on Ex's own cell: the accumulation everywhere (0.9.2).
    "select_removed": (("kms_half_x[i], in_h_x);", "kms_half_x[i], 1);"),
                       "walls3d", False, (2, 2, 2), True),
    # The auxiliary stored on a plain cell (C2's auxiliary contract).
    "auxiliary_on_a_plain_cell": (("    if (dsigw) fw[idx] = src;\n",
                                   "    fw[idx] = src;\n"),
                                  "walls3d", False, (2, 2, 2), True),
    # update_E's membership read on the integer lattice along x: the boundary plane
    # takes the other form.
    "integer_record_for_update_E": (("kms_half_x[i], in_h_x);", "kms_half_x[i], in_i_x);"),
                                    "walls3d", False, (2, 2, 2), True),
    # A NULL: float32 multiply commutes bitwise.
    "product_commuted": (("(kps * src)", "(src * kps)"), "walls3d", True, (2, 2, 2),
                         False),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_defect_changes_a_word_and_the_null_does_not(defect, certified):
    edit, case, plant, arity, must_change = DEFECTS[defect]
    _start, want, got, *_rest = run(certified, case, plant, arity, edit)
    changed = sum(_differing(want, got).values())
    if must_change:
        assert changed > 0, f"{defect} left every word unchanged"
    else:
        assert changed == 0, f"the null {defect} changed {changed} words"
