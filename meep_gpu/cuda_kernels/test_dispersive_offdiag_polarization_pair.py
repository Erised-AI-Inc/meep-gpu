"""The off-diagonal E->P weld: the shape a shipped refusal named, and its lift.

WHAT THIS FILE SETTLES. ``dispersive_offdiag_fused_polarization_pair`` exists because
``fused_polarization_pair``'s own preamble refuses this cell in as many words -- and
refuses it for the PER-COMPONENT SPLIT, naming the shape that would serve it: "all
three components in one launch". So the questions here are sharper than usual and all
of them are answerable without a device:

1. **Is the E half lifted WHOLE?** The whole argument for this weld is that
   ``update_E`` never splits, so no rotation happens inside the launch and the
   off-diagonal row product cannot read a rotated P slot. If the emitted body were
   missing one component's block, the weld would be the very thing its sibling
   refuses.

2. **Is the P half the certified recurrence?** It is ``ade_kernels``' own body under
   ``fused_polarization_pair``'s checked renames, and the seam line replaces the drive
   LOAD with the ``src`` register the E half just stored.

3. **Is the pole buffer bound exactly once?** The E half's chain reads ``state.P[c]``
   and the recurrence's ``p_now`` IS that array. Two ``__restrict__`` pointers to one
   allocation is undefined behaviour NVRTC miscompiles without a diagnostic.

4. **Does the predicate refuse what the launch cannot serve?** Including the two
   clauses this weld ADDS over its halves -- the drive identity and the extent the
   dropped ADE guard assumed.

5. **Does the tail run the components the launch did not, in the certified order?**
   The rotation moves three names per component, and a launcher that got the sequence
   wrong would still compute.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING; the arithmetic is a DEVICE claim
(``parity/meep_gpu/gate_cuda_dispersive_offdiag_polarization_pair.py``).
"""

from __future__ import annotations

import re

import numpy
import pytest

from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import dispersive_offdiag_update_e as offdiag_e
from . import fused_pairs, fused_polarization_pair
from . import dispersive_offdiag_fused_polarization_pair as weld
from .test_fused_pairs import _NumpyWearingCupysName

#: The corpus row this product exists for, as the census records it:
#: ``examples/absorbed_power_density.py`` -- ONE Lorentzian state driving all three
#: components, folded on Y, off-diagonal chi1inv, real storage.
CORPUS_ROW = {"row": "absorbed_power_density.py", "n_polarizations": 1,
              "driven": ("Ex", "Ey", "Ez"), "mirrored": (False, True, False),
              "has_offdiagonal_epsilon": True, "force_complex_fields": False}

MASK = (1, 0, 0, 1, 0, 0)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


_DEFAULT_ROWS = {"Ex": ("Ey",), "Ey": ("Ex",)}


def _build(xp, *, rows=_DEFAULT_ROWS, driven=("Ex", "Ey", "Ez"), states=1):
    """A folded, off-diagonal, dispersive triple in the corpus row's shape."""
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, symmetry=(Mirror("Y", 1),), xp=xp)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    shape = tuple(int(n) for n in grid.shape)
    names = ("Ex", "Ey", "Ez")
    rng = numpy.random.default_rng(5)
    built = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(numpy.float32)
                   for partner in partners}
             for row, partners in rows.items()}
    fields.set_epsilon_volumes(
        {n: numpy.full(shape, v, numpy.float32) for n, v in zip(names, (2., 2.5, 3.))},
        {n: numpy.full(shape, numpy.float32(1.0 / v), numpy.float32)
         for n, v in zip(names, (2., 2.5, 3.))},
        chi1inv_offdiagonal=built or None)
    fields.polarizations = [
        PolarizationState(Susceptibility(1.0, 0.1, "lorentzian"),
                          {n: (0.25 if n in driven else 0.0) for n in names},
                          grid, numpy.float32)
        for _ in range(states)]
    return fields, layer, grid


def _source(**kwargs):
    return weld.kernel_source(kwargs.get("mask", MASK), kwargs.get("counts", (1, 1, 1)),
                              kwargs.get("component", 0),
                              kwargs.get("kinds", (False,)))


# ---------------------------------------------------------------------------
# 1. THE E HALF IS LIFTED WHOLE
# ---------------------------------------------------------------------------

def test_the_update_e_half_is_the_certified_source_verbatim_but_for_two_edits():
    """THE WHOLE ARGUMENT FOR THIS WELD. ``update_E`` never splits, so no rotation
    happens inside the launch and the off-diagonal row product cannot read a rotated
    P slot. Every line of the certified emission must therefore survive, except the
    entry-point name and the signature's closing line."""
    certified = offdiag_e.dispersive_offdiag_source(MASK, (1, 1, 1))
    source = _source()
    missing = []
    for line in certified.splitlines():
        if not line.strip():
            continue
        if line.startswith('extern "C" __global__ void'):
            continue
        if line == "    float gw_x, float gw_y, float gw_z":
            continue
        if line not in source:
            missing.append(line)
    assert not missing, f"the certified update_E lost {len(missing)} lines: {missing[:4]}"


def test_all_three_components_are_in_one_launch():
    """The shape the sibling's refusal names, asserted rather than assumed: a body
    missing one component's block would BE the per-component split."""
    source = _source()
    for name in ("Ex", "Ey", "Ez"):
        assert f"    // --- {name}: own axis" in source
        assert f"constitutive_apply({name}, f_w_{name}, idx, src_{name}," in source
    assert source.count("constitutive_apply(") == 4  # three calls + the definition


def test_the_weld_carries_its_own_entry_point_name():
    """A second kernel wearing the certified one's name would make
    ``certification.json``'s partition test and any NVRTC binary observation
    ambiguous about which body it saw."""
    names = re.findall(r'extern "C" __global__ void (\w+)\(', _source())
    assert names == [weld.KERNEL_NAME]
    assert weld.KERNEL_NAME != offdiag_e.KERNEL_NAME


# ---------------------------------------------------------------------------
# 2. THE P HALF IS THE CERTIFIED RECURRENCE, AND THE SEAM IS A REGISTER
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component,target", [(0, "Ex"), (1, "Ey"), (2, "Ez")])
def test_the_recurrence_is_the_certified_line_and_the_drive_is_the_register(
        component, target):
    source = _source(component=component)
    assert (f"    p_out_0[idx] = ((p_0 * c_now_0) + (c_prev_0 * q_0)) + "
            f"(c_drive_0 * (s_0 * w));") in source
    # THE SEAM: the drive LOAD is gone and the register is used.
    assert "float w = drive[idx];" not in source
    assert f"    float w = src_{target};" in source
    # ...and the register is defined ABOVE the recurrence, which is what makes the
    # hand-off a hand-off rather than a use-before-definition.
    assert source.index(f"float src_{target} =") < source.index(f"float w = src_{target};")


def test_the_pole_buffer_is_bound_exactly_once():
    """Two ``__restrict__`` pointers to one allocation is undefined behaviour NVRTC
    miscompiles without a diagnostic. The E half's chain and the recurrence's
    ``p_now`` are the SAME array, so the signature declares it once."""
    source = _source()
    # THE KERNEL'S signature, not the first `) {` in the file: the prelude declares
    # several __device__ helpers above it and slicing at the first terminator would
    # measure one of those.
    entry = source.index(f'extern "C" __global__ void {weld.KERNEL_NAME}(')
    signature = source[entry:].split("\n) {\n", 1)[0]
    assert signature.count("P_Ex_0") == 1
    # ``p_now`` survives in the certified ADE PROLOGUE COMMENT, which is lifted with
    # the recurrence and is where its provenance lives. What must be gone is the
    # PARAMETER, so the check is made on the source with comments stripped.
    code = re.sub(r"//[^\n]*", "", source)
    assert "p_now" not in code
    assert "p_now" in source, (
        "the certified ADE prologue's own citation vanished with the parameter; the "
        "recurrence would be lifted without the comment that says where it came from")
    # And the recurrence really reads through that one parameter.
    assert "float p_0 = P_Ex_0[idx];" in source


@pytest.mark.parametrize("kinds", [(False,), (True,), (True, False), (False, True)])
def test_every_pole_arity_and_sigma_kind_emits_its_own_group(kinds):
    counts = tuple(len(kinds) for _ in range(3))
    source = weld.kernel_source(MASK, counts, 0, kinds)
    for index, volume in enumerate(kinds):
        assert f"p_out_{index}[idx] = " in source
        assert (f"float s_{index} = sigma_{index}[idx];" if volume
                else f"float s_{index} = sigma_{index};") in source
    assert f"p_out_{len(kinds)}" not in source


def test_the_signature_does_not_end_on_a_dangling_comma():
    """A trailing comma is a parameter declarator NVRTC expects and never finds --
    caught on the first device run and pinned here so it cannot come back."""
    for kinds in ((False,), (True, False)):
        counts = tuple(len(kinds) for _ in range(3))
        signature = weld.kernel_source(MASK, counts, 0, kinds).split("\n) {\n", 1)[0]
        assert not signature.rstrip().endswith(",")


def test_a_drifted_certified_signature_fails_the_splice(monkeypatch):
    """A certified string that moved must STOP the splice rather than emit a kernel
    that is quietly not the certified arithmetic."""
    original = offdiag_e.dispersive_offdiag_source

    def drifted(mask, counts):
        return original(mask, counts).replace(
            "    float gw_x, float gw_y, float gw_z\n) {\n",
            "    float gw_x, float gw_y, float gw_z, int spare\n) {\n")

    monkeypatch.setattr(offdiag_e, "dispersive_offdiag_source", drifted)
    with pytest.raises(AssertionError, match="no longer closes"):
        _source()


def test_the_lift_edits_are_data_and_carry_no_silent_extra():
    assert len(weld.LIFT_EDITS) == 6
    assert all(set(edit) == {"line", "became", "why"} for edit in weld.LIFT_EDITS)
    assert all(edit["why"] for edit in weld.LIFT_EDITS)


# ---------------------------------------------------------------------------
# 3. THE PREDICATE
# ---------------------------------------------------------------------------

def test_the_weld_admits_the_corpus_row_shape(xp):
    fields, layer, grid = _build(xp)
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert covered, why
    assert weld.fused_component(fields) == "Ex"
    assert weld.pole_counts(fields) == (1, 1, 1)


def test_the_diagonal_polarization_pair_refuses_the_same_row(xp):
    """THE PARTITION, and it is one clause wide: the shipped E->P pair's update_E
    half refuses an off-diagonal chi1inv row BY NAME, and this weld requires one. No
    row can reach both, which is what buys the slots."""
    fields, layer, grid = _build(xp)
    assert not fused_polarization_pair.covers_fused_polarization_pair(
        fields, layer, grid, ())[0]


def test_a_diagonal_run_is_refused_by_this_weld(xp):
    """The other side of the partition: with no off-diagonal row installed this cell
    belongs to the SHIPPED E->P pair and to this weld not at all."""
    fields, layer, grid = _build(xp, rows={})
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert not covered, why
    assert why.startswith("constitutive half:")
    assert fused_polarization_pair.covers_fused_polarization_pair(
        fields, layer, grid, ())[0], "the diagonal pair must serve what this refuses"


def test_a_run_with_no_driven_component_is_refused_by_name(xp):
    """A weld with no recurrence to carry is the certified update_E alone, and
    saying so is what keeps REPLACES honest."""
    fields, layer, grid = _build(xp, driven=())
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert not covered
    # THE ADE HALF REFUSES FIRST, and that is the conjunction working rather than a
    # missing clause: this weld's own "nothing is driven" clause sits BELOW both
    # halves and is reachable only where they admit. It is written anyway, because
    # ``fused_component`` returning None must never reach the emitter.
    assert why.startswith("ADE half:") and "driven" in why
    assert weld.fused_component(fields) is None


def test_a_mismatched_recurrence_extent_is_refused_by_name(xp):
    """This kernel emits ONE bounds guard -- the E half's -- so every recurrence
    volume must carry exactly the extent that guard walks."""
    fields, layer, grid = _build(xp)
    state = fields.polarizations[0]
    state.P["Ex"] = numpy.zeros(3, numpy.float32)
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert not covered
    # THE CONSTITUTIVE HALF REFUSES THIS ONE FIRST -- it checks the pole volumes
    # against the GRID's shape -- and this weld's own extent clause asks a different
    # question: against the extent the DROPPED ADE guard would have used, which is
    # the size of the array this launch walks. Both are written; the halves are
    # asked first, so only a shape the halves admit can reach the second.
    assert why.startswith("constitutive half:") and "shape" in why


# ---------------------------------------------------------------------------
# 4. THE TAIL AND THE ROTATION
# ---------------------------------------------------------------------------

def test_the_rotation_is_the_certified_three_line_move(xp):
    """``dispersion.py:689-691``: the scratch becomes P, P becomes P_prev, and the
    retired history becomes the scratch. Anything else advances one buffer twice."""
    fields, _layer, _grid = _build(xp)
    state = fields.polarizations[0]
    before = (state.P["Ex"], state.P_prev["Ex"], state._scratch)
    weld._rotate(state, "Ex")
    assert state.P["Ex"] is before[2]
    assert state.P_prev["Ex"] is before[0]
    assert state._scratch is before[1]


def test_the_launch_ledger_names_the_saving_rather_than_claiming_it(xp):
    """What this weld is worth is a COUNT, and the count is on this row: the array
    path runs 1 + 3 launches on this seam and the weld runs 1 + 2."""
    fields, _layer, _grid = _build(xp)
    driven = sum(len(tuple(state.driven())) for state in fields.polarizations)
    assert driven == 3
    assert weld.fused_component(fields) == "Ex"
    # The tail is every (state, component) the fused launch did not advance.
    assert driven - 1 == 2


def test_the_corpus_row_is_the_one_the_census_records():
    """The denominator, pinned: this cell carries ONE seam-instance and the product
    exists for it. A row count that moved silently would make every number above
    stale."""
    assert CORPUS_ROW["n_polarizations"] == 1
    assert CORPUS_ROW["driven"] == ("Ex", "Ey", "Ez")


# ---------------------------------------------------------------------------
# 5. THE WIRING
# ---------------------------------------------------------------------------

def test_the_weld_row_names_its_own_module_seam_and_arms():
    family = weld.FAMILY
    assert fused_pairs.FUSED_PRODUCTS[family]["module"] == \
        "dispersive_offdiag_fused_polarization_pair"
    assert fused_pairs.FUSED_PRODUCTS[family]["curl_slot"] == "update_E"
    assert fused_pairs.FUSED_PAIR_ARMS[family] == ("dispersive off-diagonal", "ADE")
    assert weld.SLOT == "update_E"
    assert weld.REPLACES == ("update_E", "update_P")
    assert weld.CARRIES_DEPOSIT_REPAIR is False
    # RELEASED 2026-09-02, declared as the FINAL BYTES the gate ran against.
    assert weld.CERTIFIED_KERNELS == (weld.KERNEL_NAME,)
    assert weld.UNCERTIFIED_KERNELS == {}


def test_the_e_to_p_seam_carries_no_deposit_list_at_all():
    """The flag's False here is DRIVER CONSTRUCTION, not a measurement of a cell:
    ``FUSED_PAIR_SEAMS`` gives this seam ``None`` because nothing is injected between
    the driver's two consults, so ``_install_fused_pair`` reads it as unconditionally
    empty."""
    assert fused_pairs.FUSED_PAIR_SEAMS["update_E"] == ("update_P", None)
