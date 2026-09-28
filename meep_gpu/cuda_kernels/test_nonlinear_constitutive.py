"""The nonlinear (chi2/chi3) constitutive slice, on a laptop with no GPU.

WHAT THIS FILE CAN AND CANNOT SETTLE. It compiles nothing and launches nothing,
so it certifies nothing. What it does settle, before a device slot is spent:

* that the ARITHMETIC transcribed into the device string reproduces
  ``stepping.update_E`` BIT FOR BIT under a nonlinearity -- the whole device tree,
  in float32 NumPy, on periodic, metallic and mixed grids, on a partly nonlinear
  run, on volume chi as well as scalar chi, and on the 1-D-like shape the two
  corpus rows carry (:func:`test_the_device_tree_reproduces_stepping_bit_for_bit`);
* that the reference can FAIL -- every arithmetic defect the gate will arm is
  shown here to be visible against that same reference
  (:func:`test_each_planted_defect_is_visible_against_the_reference`), so a device
  leg reporting a defect UNCAUGHT is a statement about the kernel rather than
  about the fixture;
* that the device string still carries the exact lines the gate's source
  mutations key on, so a mutation cannot silently stop arming
  (:func:`test_the_device_string_carries_the_lines_the_mutations_key_on`);
* that the H-side NULL is a reading of the SOURCE and not a memory: the claim
  "``stepping.update_H`` does not read the nonlinearity" is checked by parsing
  ``stepping.py`` (:func:`test_update_H_names_nothing_nonlinear`);
* that this predicate and every sibling constitutive predicate PARTITION the
  chi2/chi3 clause rather than overlapping on it.

THE NUMPY TRANSCRIPTION BELOW IS A SECOND READER OF THE DEVICE STRING, and it is
deliberately not shared with the gate's, which has one of its own. Two
independent transcriptions of the same CUDA text that agree with ``stepping`` are
worth more than one factored out and imported twice, because a misreading has to
be made twice to survive.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import types

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.cuda_kernels import coverage
from meep_gpu.cuda_kernels import nonlinear_constitutive as nlc

HERE = pathlib.Path(__file__).resolve().parent

BC_PERIODIC, BC_METALLIC = 0, 1

#: Every array this sub-step writes, plus the three volumes it reads as
#: neighbours. ``f_w`` IS STATE and is compared on every case: a tree that gets
#: the field right and the auxiliary wrong is correct for exactly one launch.
STATE = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez", "Dx", "Dy", "Dz")
OUTPUTS = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(np, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, boundaries=("periodic", "periodic", "periodic"),
          cell=(8.0, 10.0, 12.0), chi2=None, chi3=None, seed=11,
          courant=0.35, symmetry=(), epsilon="volumes",
          force_complex_fields=False, **grid_kwargs):
    """A frozen ``(fields, layer, grid)`` triple carrying a live nonlinearity.

    ``epsilon="volumes"`` draws THREE INDEPENDENT inverse-permittivity volumes,
    away from 1.0, for the reason the certified fixture states: against a table
    of ones ``drop_inverse_epsilon`` is bit-identical and ``bind_Ez_inv_eps_for_all_three``
    is invisible.
    """
    planes = tuple(Mirror(name, 1) for name in symmetry)
    grid = Grid(resolution=1.0, cell_size=tuple(cell), boundaries=tuple(boundaries),
                symmetry=planes, xp=xp, courant=courant, **grid_kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    if force_complex_fields:
        # A complex fixture exists only so the refusal can fire; the arithmetic
        # legs never reach it, and seeding complex storage from a float32 draw
        # would be a second fixture nothing measures.
        return fields, layer, grid
    if epsilon == "volumes":
        forward, inverse = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
            forward[component] = xp.asarray(values)
            inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
        fields.set_epsilon_volumes(forward, inverse)
    if chi2 is not None or chi3 is not None:
        fields.set_nonlinear_volumes(chi2 or {}, chi3 or {})
    for name in STATE:
        getattr(fields, name)[...] = xp.asarray(
            rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32))
    return fields, layer, grid


def snapshot(fields):
    return {name: np.asarray(getattr(fields, name)).copy() for name in STATE}


def restore(fields, frozen):
    for name, values in frozen.items():
        getattr(fields, name)[...] = values


def tables_for_E(layer):
    """The HALF-INTEGER views ``update_E`` reads (stepping.py:1015)."""
    suffix = "_h" if coverage.constitutive_sub_lattice("E") else ""
    return {f"{stem}_{axis}": np.asarray(getattr(layer, f"{stem}_{axis}{suffix}")).reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


# ---------------------------------------------------------------------------
# The NumPy reader of the device string
# ---------------------------------------------------------------------------

def _up(n, code):
    index = np.arange(n) + 1
    if code == BC_METALLIC:
        valid = index < n
        return np.where(valid, index, 0), valid
    return np.where(index == n, 0, index), np.ones(n, bool)


def _down(n, code):
    index = np.arange(n) - 1
    if code == BC_METALLIC:
        valid = index >= 0
        return np.where(valid, index, 0), valid
    return np.where(index < 0, n - 1, index), np.ones(n, bool)


def four_point_sum(g, own_axis, partner_axis, codes, shape, defect=None):
    """``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])`` -- the SHIFTED-PAIR association.

    ``defect`` plants one of the gate's arithmetic mutations so this reference can
    be shown to see it: ``"left_to_right"`` is MEEP C's summation order
    (step_generic.cpp:646-648) and ``"same_direction"`` takes both shifts the same
    way round, the half-cell registration error.
    """
    identity = [np.arange(shape[axis]) for axis in range(3)]
    up_index, up_valid = _up(shape[own_axis], codes[own_axis])
    if defect == "same_direction":
        down_index, down_valid = _up(shape[partner_axis], codes[partner_axis])
    else:
        down_index, down_valid = _down(shape[partner_axis], codes[partner_axis])

    def corner(own_up, partner_down):
        picks = list(identity)
        if own_up:
            picks[own_axis] = up_index
        if partner_down:
            picks[partner_axis] = down_index
        gathered = g[np.ix_(*picks)]
        mask = np.ones(shape, bool)
        if own_up:
            bshape = [1, 1, 1]
            bshape[own_axis] = shape[own_axis]
            mask = mask & up_valid.reshape(bshape)
        if partner_down:
            bshape = [1, 1, 1]
            bshape[partner_axis] = shape[partner_axis]
            mask = mask & down_valid.reshape(bshape)
        return np.where(mask, gathered, np.float32(0.0)).astype(np.float32)

    here, down, up, both = (corner(False, False), corner(False, True),
                            corner(True, False), corner(True, True))
    if defect == "left_to_right":
        return (((here + up) + down) + both).astype(np.float32)
    return ((here + down) + (up + both)).astype(np.float32)


def device_tree(fields, tables, codes, defect=None):
    """``update_E_pml_real_nonlinear``, transcribed in float32 NumPy.

    Line for line the CUDA body: three sources formed first, then the certified
    two-accumulation tail with ``prev`` read before the store. ``defect`` plants
    exactly one of the gate's mutations.
    """
    shape = tuple(fields.grid.shape)
    displacement = {0: np.asarray(fields.Dx), 1: np.asarray(fields.Dy),
                    2: np.asarray(fields.Dz)}
    one, two, three = np.float32(1.0), np.float32(2.0), np.float32(3.0)
    sources = []
    for component, _source, axis in nlc.E_TERMS:
        gs = displacement[axis]
        us = np.asarray(fields.inverse_epsilon_for(component))
        src = (gs * us).astype(np.float32)
        if fields.is_nonlinear(component):
            first, second = nlc.TRANSVERSE_PARTNERS[axis]
            if defect == "swap_transverse_partners":
                first, second = second, first
            sub = defect if defect in ("left_to_right", "same_direction") else None
            g1s = four_point_sum(displacement[first], axis, first, codes, shape, sub)
            g2s = four_point_sum(displacement[second], axis, second, codes, shape, sub)
            transverse = np.float32(0.0625) * (g1s * g1s + g2s * g2s)
            if defect == "drop_transverse":
                transverse = np.float32(0.0)
            dsqr = (gs * gs + transverse).astype(np.float32)
            chi2 = fields.chi2_for(component)
            chi3 = fields.chi3_for(component)
            if defect == "swap_chi2_chi3":
                chi2, chi3 = chi3, chi2
            chi2 = np.float32(chi2) if np.ndim(chi2) == 0 else np.asarray(chi2)
            chi3 = np.float32(chi3) if np.ndim(chi3) == 0 else np.asarray(chi3)
            us_sq = (us * us).astype(np.float32)
            us_cu = (us_sq * us).astype(np.float32)
            if defect == "swap_epsilon_powers":
                us_sq, us_cu = us_cu, us_sq  # c2 gets chi1inv^3, c3 gets chi1inv^2
            c2 = ((gs * chi2) * us_sq).astype(np.float32)
            c3 = ((dsqr * chi3) * us_cu).astype(np.float32)
            if defect == "regroup_pade_numerator":
                num = (one + (c2 + two * c3)).astype(np.float32)
            else:
                num = ((one + c2) + two * c3).astype(np.float32)
            den = ((one + two * c2) + three * c3).astype(np.float32)
            factor = (num / den).astype(np.float32)
            if defect == "drop_pade_factor":
                factor = np.float32(1.0)
            if defect == "scale_row_before_product":
                src = (gs * (us * factor)).astype(np.float32)
            elif defect == "commute_pade_scale":
                src = (factor * src).astype(np.float32)
            else:
                src = (src * factor).astype(np.float32)
        sources.append(src)
    for (component, _source, axis), src in zip(nlc.E_TERMS, sources):
        target = np.asarray(getattr(fields, component))
        auxiliary = np.asarray(getattr(fields, "f_w_" + component))
        bshape = [1, 1, 1]
        bshape[axis] = shape[axis]
        kps = tables[f"kps_{'xyz'[axis]}"].reshape(bshape)
        kms = tables[f"kms_{'xyz'[axis]}"].reshape(bshape)
        previous = auxiliary.copy()
        auxiliary[...] = src
        accumulated = (target + kps * src).astype(np.float32)
        target[...] = (accumulated - kms * previous).astype(np.float32)


def differing_words(reference, fields):
    total = 0
    for name in OUTPUTS:
        a = np.ascontiguousarray(reference[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(np.asarray(getattr(fields, name)),
                                 dtype=np.float32).ravel().view(np.uint32)
        total += int(np.count_nonzero(a != b))
    return total


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

CASES = {
    # All three axes wrap: the ghost rule the two corpus rows take.
    "all_periodic": dict(boundaries=("periodic",) * 3,
                         chi2={"Ez": 0.0}, chi3={"Ez": 0.05}),
    # All three walled: every four-point corner loses a term at a face.
    "all_metallic": dict(boundaries=("metallic",) * 3,
                         chi2={"Ez": 0.0}, chi3={"Ez": 0.05}),
    # One of each, so a swapped axis code cannot hide.
    "mixed": dict(boundaries=("periodic", "metallic", "periodic"),
                  chi2={"Ex": 0.02, "Ey": -0.01, "Ez": 0.0},
                  chi3={"Ex": 0.03, "Ey": 0.02, "Ez": 0.05}),
    # A PARTLY nonlinear run: Ey takes MEEP's ``else if (u)`` branch, which must
    # compile to the certified plain body.
    "partly_nonlinear": dict(boundaries=("periodic", "periodic", "metallic"),
                             chi2={"Ex": 0.0, "Ez": 0.0},
                             chi3={"Ex": 0.04, "Ez": 0.05}),
    # THE CORPUS SHAPE: 3rd-harm-1d.py is (1, 1, nz). Two invariant axes whose
    # wrap returns the same cell, which is MEEP's stride(d) = 0.
    "one_dimensional": dict(boundaries=("periodic",) * 3, cell=(1.0, 1.0, 40.0),
                            chi2={"Ex": 0.0}, chi3={"Ex": 0.06}),
    # A negative chi2 is an ordinary material (a reversed Pockels coefficient),
    # and it is the only class in which c2 subtracts from the denominator.
    "negative_chi2": dict(boundaries=("periodic", "periodic", "metallic"),
                          chi2={"Ez": -0.05}, chi3={"Ez": 0.03}),
}

#: Every arithmetic defect the gate arms in the device text, as this reference
#: can plant it. ``True`` means the reference must SEE it; ``False`` means it is
#: a NULL that must leave the bytes alone. A battery of only-must-be-caught legs
#: scores identically whether the comparator works or fails everything.
PLANTED_DEFECTS = {
    "left_to_right": True,
    "same_direction": True,
    "drop_transverse": True,
    "drop_pade_factor": True,
    "regroup_pade_numerator": True,
    "swap_epsilon_powers": True,
    "scale_row_before_product": True,
    "swap_chi2_chi3": True,
    "swap_transverse_partners": False,   # Dsqr's sum commutes, bitwise
    "commute_pade_scale": False,         # IEEE multiply commutes, bitwise
}


def _volume_chi(grid, value, seed):
    """A SPATIALLY VARYING chi volume.

    A uniform volume cannot distinguish a chi lookup that reads the wrong cell
    from one that reads the right one, so the volume class is drawn to vary.
    """
    rng = np.random.default_rng(seed)
    return (np.float32(value)
            * rng.uniform(0.4, 1.6, size=grid.shape).astype(np.float32))


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", sorted(CASES))
def test_the_device_tree_reproduces_stepping_bit_for_bit(label, xp):
    """The whole kernel body against ``stepping.update_E``, as uint32 words.

    NON-VACUITY IS ASSERTED, NOT ASSUMED: the array path must have moved output
    words (zero-init is a fixed point of this recurrence), the Pade factor must
    differ from 1.0 somewhere (an identity factor makes every nonlinear mutation
    invisible), and the run must sit inside the pole guard's admitted domain.
    """
    fields, layer, grid = build(xp, **CASES[label])
    codes, refusal = nlc.boundary_codes(grid)
    assert refusal is None, refusal

    frozen = snapshot(fields)
    stepping.update_E(fields, layer)
    reference = snapshot(fields)

    moved = sum(
        int(np.count_nonzero(
            np.ascontiguousarray(frozen[name]).ravel().view(np.uint32)
            != np.ascontiguousarray(reference[name]).ravel().view(np.uint32)))
        for name in OUTPUTS)
    assert moved > 0, "the array path moved no output word; this case certifies nothing"

    margin = stepping.nonlinear_margin(fields, layer)
    assert margin is not None, "the fixture installed no live nonlinearity"
    assert 0.0 < margin.expansion < nlc.POLE_EXPANSION_BOUND, (
        f"expansion {margin.expansion} is outside (0, 1/3); a zero expansion makes "
        f"the Pade factor the identity and a large one leaves the guard's domain")

    restore(fields, frozen)
    device_tree(fields, tables_for_E(layer), codes)
    assert differing_words(reference, fields) == 0, (
        f"{label}: the device tree diverged from stepping.update_E in "
        f"{differing_words(reference, fields)} words")


@pytest.mark.parametrize("defect", sorted(PLANTED_DEFECTS))
def test_each_planted_defect_is_visible_against_the_reference(defect, xp):
    """Can this reference fail? Each defect is scored CAUGHT or NULL, never assumed.

    The two NULLs are the control. Without them a battery in which everything is
    caught scores identically whether the comparator works or has degenerated
    into failing everything.
    """
    fields, layer, grid = build(xp, **CASES["mixed"])
    codes, refusal = nlc.boundary_codes(grid)
    assert refusal is None, refusal
    frozen = snapshot(fields)
    stepping.update_E(fields, layer)
    reference = snapshot(fields)

    restore(fields, frozen)
    device_tree(fields, tables_for_E(layer), codes, defect=defect)
    seen = differing_words(reference, fields)
    if PLANTED_DEFECTS[defect]:
        assert seen > 0, f"{defect} left every word identical; the reference cannot see it"
    else:
        assert seen == 0, (
            f"{defect} is claimed inert and changed {seen} words; either IEEE-754 "
            f"does not say what this test says it does, or the plant is wrong")


def test_a_volume_chi_and_a_scalar_chi_take_the_same_reference(xp):
    """A spatially varying chi2/chi3 volume, against ``stepping``.

    ``Fields._coerce_nonlinear_value`` (fields.py:929-945) accepts a float or a
    float32 volume, and the kernel selects between them per component with a
    runtime flag. The volume arm is what makes a chi lookup that reads the wrong
    cell visible at all -- a uniform chi forgives it exactly.
    """
    fields, layer, grid = build(xp, boundaries=("periodic", "metallic", "periodic"))
    fields.set_nonlinear_volumes(
        {"Ex": 0.0, "Ez": _volume_chi(grid, 0.02, 3)},
        {"Ex": _volume_chi(grid, 0.04, 1), "Ez": _volume_chi(grid, 0.03, 2)})
    codes, refusal = nlc.boundary_codes(grid)
    assert refusal is None, refusal

    bindings = nlc.nonlinear_bindings(fields)
    assert bindings["nonlinear"] == [True, False, True]
    assert bindings["chi3_is_volume"] == [True, False, True]
    assert bindings["chi2_is_volume"] == [False, False, True], (
        "Ex's chi2 is the scalar 0.0 MEEP stores for the missing half of the pair")

    frozen = snapshot(fields)
    stepping.update_E(fields, layer)
    reference = snapshot(fields)
    restore(fields, frozen)
    device_tree(fields, tables_for_E(layer), codes)
    assert differing_words(reference, fields) == 0


def test_a_linear_component_compiles_to_the_certified_plain_body(xp):
    """NL = 0 must be ``src = D * inv_eps`` -- MEEP's ``else if (u)`` branch.

    Checked as a property of the shipped device string rather than of the
    reference: the plain product is formed unconditionally and the Pade factor is
    applied inside ``if (nl_*)``, so a component with no chi runs exactly the
    expression ``update_E_pml_real`` runs.
    """
    code = nlc._update_E_pml_real_nonlinear_kernel_code
    for axis, name in enumerate("xyz"):
        plain = f"float src_{name} = gs_{name} * us_{name};"
        assert code.count(plain) == 1, f"{plain!r} is not the unconditional source"
        scaled = f"src_{name} = src_{name} * pade_u("
        assert code.count(scaled) == 1
        assert code.index(plain) < code.index(scaled), (
            "the Pade factor must scale an already-formed plain product")


# ---------------------------------------------------------------------------
# The H-side null, read off stepping.py
# ---------------------------------------------------------------------------

NONLINEAR_NAMES = (
    "_chi2_components", "_chi3_components", "has_nonlinearity", "is_nonlinear",
    "chi2_for", "chi3_for", "_nonlinear_constitutive", "calc_nonlinear_u",
    "_nonlinear_displacement", "_nonlinear_transverse_sums", "_nonlinear_dsqr",
)


def _function_source(name: str) -> str:
    tree = ast.parse(pathlib.Path(stepping.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"stepping.py has no top-level function {name!r}")


def test_update_H_names_nothing_nonlinear():
    """THE H-SIDE NULL'S WHOLE PREMISE, read off the file rather than remembered.

    ``covers_real_pml_constitutive`` refuses a nonlinearity on BOTH sides
    (coverage.py:1056-1057). That is correct on ``update_E`` and inherited on
    ``update_H``: not one of the engine's nonlinear names occurs anywhere in
    ``stepping.update_H``. The gate MEASURES the consequence on a device; this
    pins the premise on a laptop, so a future edit that gives ``update_H`` a
    nonlinear term fails here first.
    """
    body = _function_source("update_H")
    present = [name for name in NONLINEAR_NAMES if name in body]
    assert not present, (
        f"stepping.update_H now reads {present}; the H-side null admission in "
        f"nonlinear_constitutive.covers_real_pml_nonlinear_constitutive is no "
        f"longer a reading of this function and must be withdrawn")
    # And the E side must still read them, or the pin above is vacuous.
    electric = _function_source("update_E")
    assert any(name in electric for name in NONLINEAR_NAMES), (
        "stepping.update_E reads no nonlinear name either; the scan is broken")


def test_the_H_side_kernel_is_the_certified_one_and_not_a_new_string():
    """Arm H ships NO kernel. ``SIDE_KERNEL['H']`` is None, and the file has one ``__global__``."""
    assert nlc.SIDE_KERNEL["H"] is None
    assert nlc.SIDE_KERNEL["E"] == "update_E_pml_real_nonlinear"
    source = (HERE / "nonlinear_constitutive.py").read_text(encoding="utf-8")
    assert source.count('extern "C" __global__ void') == 1, (
        "this family declares exactly one kernel; the H side reuses the certified "
        "update_H_pml_real and must not grow a second copy of it")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_admits_the_reference_configuration(xp):
    fields, layer, grid = build(xp, **CASES["all_periodic"])
    for side in nlc.NONLINEAR_SIDES:
        covered, reason = nlc.covers_real_pml_nonlinear_constitutive(
            fields, layer, grid, side)
        assert covered, f"side {side} refused the reference configuration: {reason}"
        assert reason is None


def test_the_predicate_and_the_sibling_partition_the_nonlinearity(xp):
    """DISJOINTNESS, measured on both sides and in both directions.

    The census's disjointness check is the property this protects: two families
    admitting one slot is not extra coverage, it is one of them widened past its
    evidence. Here the two predicates are complementary ON THE CHI CLAUSE, so no
    coordinating edit to ``coverage.py`` is needed for them to partition.
    """
    for side in ("H", "E"):
        nonlinear, layer_nl, grid_nl = build(xp, **CASES["all_periodic"])
        linear, layer_lin, grid_lin = build(xp, boundaries=("periodic",) * 3)

        mine_nl, _reason = nlc.covers_real_pml_nonlinear_constitutive(
            nonlinear, layer_nl, grid_nl, side)
        theirs_nl, why_nl = coverage.covers_real_pml_constitutive(
            nonlinear, layer_nl, grid_nl, side)
        assert mine_nl and not theirs_nl
        assert "chi2/chi3" in why_nl

        mine_lin, why_lin = nlc.covers_real_pml_nonlinear_constitutive(
            linear, layer_lin, grid_lin, side)
        theirs_lin, _ = coverage.covers_real_pml_constitutive(
            linear, layer_lin, grid_lin, side)
        assert theirs_lin and not mine_lin
        assert "no instantaneous chi2/chi3" in why_lin


@pytest.mark.parametrize("family,arguments", [
    ("offdiag", ()),
    ("no_pml", ()),
])
def test_every_sibling_constitutive_predicate_still_refuses_the_nonlinearity(
        family, arguments, xp):
    """The other two constitutive arms must keep their chi clause, or we overlap."""
    fields, layer, grid = build(xp, **CASES["all_periodic"])
    if family == "offdiag":
        covered, reason = coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid)
    else:
        from meep_gpu.cuda_kernels import no_pml_constitutive
        covered, reason = no_pml_constitutive.covers_no_pml_null_constitutive(
            fields, layer, grid, "E")
    assert not covered
    assert reason is not None


REFUSALS = {
    "complex_storage": (dict(force_complex_fields=True), "complex64 storage"),
    "fold": (dict(symmetry="Y", cell=(8.0, 16.0, 12.0)), "mirror symmetry"),
    "bloch": (dict(k_point=(0.1, 0.0, 0.0)), "nonzero Bloch k"),
}


@pytest.mark.parametrize("label", sorted(REFUSALS))
@pytest.mark.parametrize("side", ("H", "E"))
def test_the_predicate_refuses_by_name(label, side, xp):
    """Each refusal fires, and fires with the words the census will print."""
    overrides, needle = REFUSALS[label]
    kwargs = dict(CASES["all_periodic"])
    kwargs.update(overrides)
    fields, layer, grid = build(xp, **kwargs)
    covered, reason = nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, side)
    assert not covered, f"{label} was admitted on side {side}"
    assert needle in reason, f"{label} on side {side} refused with {reason!r}"


def test_a_polarization_and_an_offdiagonal_row_are_refused_on_E_only(xp):
    """Both change what the E source IS, and neither touches ``update_H``."""
    fields, layer, grid = build(xp, **CASES["all_periodic"])

    class _State:
        def drives(self, component):
            return True

    fields.polarizations.append(_State())
    covered_e, reason_e = nlc.covers_real_pml_nonlinear_constitutive(
        fields, layer, grid, "E")
    covered_h, _ = nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, "H")
    assert not covered_e and "dispersion" in reason_e
    assert covered_h, "a polarization does not reach update_H"


def test_an_unknown_side_raises_rather_than_refusing(xp):
    fields, layer, grid = build(xp, **CASES["all_periodic"])
    with pytest.raises(ValueError):
        nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, "P")


def test_the_predicate_refuses_rather_than_raising_on_an_object_that_cannot_answer(xp):
    """A raise escaping a predicate is a crashed run where a fail-closed no was right."""

    class _Hostile:
        xp = _NumpyWearingCupysName()

        def __getattr__(self, item):
            raise RuntimeError(f"cannot answer {item}")

    fields, layer, _grid = build(xp, **CASES["all_periodic"])
    covered, reason = nlc.covers_real_pml_nonlinear_constitutive(
        fields, layer, _Hostile(), "E")
    assert not covered and reason


def test_the_backend_clause_fires_first_on_a_plain_numpy_grid():
    """The census factors its coverage number on the backend clause firing first."""
    fields, layer, grid = build(np, **CASES["all_periodic"])
    covered, reason = nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, "E")
    assert not covered and reason == "backend is not CuPy"


def test_a_chi3_only_object_is_still_seen_as_nonlinear(xp):
    """``has_nonlinearity`` reads ``_chi2_components`` alone (fields.py:966-967).

    A predicate that asked the property instead of the two maps would call a
    hand-assembled chi3-only object linear and hand it to a kernel that multiplies
    by a Pade factor of exactly 1.
    """
    fields, layer, grid = build(xp, **CASES["all_periodic"])
    fields._chi2_components = {}
    assert not fields.has_nonlinearity
    covered, reason = nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, "E")
    assert covered, reason
    assert nlc._nonlinearity_installed(fields)


# ---------------------------------------------------------------------------
# The transcription pins
# ---------------------------------------------------------------------------

def test_the_term_table_matches_steppings():
    """Same components, same sources, same own axis -- spelled as an index here.

    ``stepping.E_CONSTITUTIVE_TERMS`` (stepping.py:228) names the axis ("x"),
    which is what ``_constitutive_coefficients`` takes; the kernel indexes a
    table, so this module carries the index. The mapping is pinned rather than
    assumed, because the own axis is the coefficient index and getting it wrong
    is a smooth, converged, entirely wrong absorber.
    """
    names = ("x", "y", "z")
    assert tuple((component, source, names[axis])
                 for component, source, axis in nlc.E_TERMS) == \
        tuple(stepping.E_CONSTITUTIVE_TERMS)


def test_the_transverse_partners_are_cycle_direction():
    """X -> Y -> Z: own axis + 1 first, own axis + 2 second (vec.hpp:586)."""
    assert nlc.TRANSVERSE_PARTNERS == tuple(
        ((axis + 1) % 3, (axis + 2) % 3) for axis in range(3))


def test_the_sub_lattice_is_the_half_integer_one(xp):
    """``update_E`` reads the HALF-INTEGER coefficients (stepping.py:1015).

    Bound backwards it is a half-cell error in the absorber profile: converged,
    smooth and wrong. The launcher asks ``constitutive_sub_lattice`` -- the same
    function the predicate asks -- so the two cannot disagree, and this pins the
    answer against the layer's own two table sets.
    """
    assert coverage.constitutive_sub_lattice("E") is True
    _fields, layer, _grid = build(xp, **CASES["all_periodic"])
    half = tables_for_E(layer)
    integer = {f"{stem}_{axis}": np.asarray(getattr(layer, f"{stem}_{axis}")).reshape(-1)
               for axis in "xyz" for stem in ("kps", "kms")}
    assert any(not np.array_equal(half[key], integer[key]) for key in half), (
        "the two sub-lattices hold the same numbers on this layer, so a swapped "
        "pairing would be invisible and swap_constitutive_sublattice is decorative")


def test_the_pole_bound_is_the_one_stepping_derives():
    assert nlc.POLE_EXPANSION_BOUND == pytest.approx(1.0 / 3.0)
    assert "1/3" in stepping.nonlinear_margin.__doc__


def test_the_compile_options_match_the_certified_pair():
    """``--fmad=false`` is CORRECTNESS on this sub-step; both modules must spell it."""
    sibling = (HERE / "constitutive_kernels.py").read_text(encoding="utf-8")
    for node in ast.parse(sibling).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_COMPILE_OPTIONS"):
            assert nlc._COMPILE_OPTIONS == ast.literal_eval(node.value)
            return
    raise AssertionError("constitutive_kernels.py no longer defines _COMPILE_OPTIONS")


def test_the_tail_is_the_certified_pairs_tail_character_for_character():
    """The shared source mutations arm through this text; a second spelling disarms them.

    ``regroup_constitutive``, ``drop_fw_store``, ``store_fw_before_reading_prev``
    and ``commute_constitutive_scale`` in ``probe_fused_kernel_bit_identity`` all
    key on the four lines of ``constitutive_apply``. If this module ever spells
    them differently the gate's battery goes quietly NOT ARMED.
    """
    sibling = (HERE / "constitutive_kernels.py").read_text(encoding="utf-8")
    tail = ("    float prev = fw[idx];\n"
            "    fw[idx] = src;\n"
            "    float a = f[idx] + kps * src;\n"
            "    f[idx] = a - kms * prev;\n")
    assert tail in sibling, "the certified pair's tail moved; re-derive this pin"
    assert tail in nlc._NONLINEAR_PRELUDE


def test_the_device_string_carries_the_lines_the_mutations_key_on():
    """Every needle the gate's own source mutations rewrite, pinned on a laptop.

    A mutation that stops matching comes back NOT ARMED on the device, hours
    later; this fails in milliseconds. The counts are exact, so a needle that
    matches somewhere unintended fails too.
    """
    code = nlc._update_E_pml_real_nonlinear_kernel_code
    exact = {
        "    float near_pair = g[o_c] + (v_d ? g[o_d] : 0.0f);\n": 1,
        "    float far_pair = (v_u ? g[o_u] : 0.0f) + (v_ud ? g[o_ud] : 0.0f);\n": 1,
        "    return near_pair + far_pair;\n": 1,
        "    float c2 = (gs * chi2) * us_sq;\n": 1,
        "    float c3 = (dsqr * chi3) * us_cu;\n": 1,
        "    float num = (1.0f + c2) + 2.0f * c3;\n": 1,
        "    float den = (1.0f + 2.0f * c2) + 3.0f * c3;\n": 1,
        "    return __fdiv_rn(num, den);\n": 1,
    }
    for needle, count in exact.items():
        assert code.count(needle) == count, (
            f"{needle!r} occurs {code.count(needle)} times, not {count}; a gate "
            f"mutation keys on this line")
    assert code.count("0.0625f * (g1s * g1s + g2s * g2s)") == 3
    for axis in "xyz":
        assert code.count(f"float us_cu = us_sq * us_{axis};") == 1


def test_the_device_string_names_the_partner_volumes_the_table_names():
    """THE ONE ERROR THE NUMPY READER CANNOT SEE.

    The reader above builds its four-point sums from ``TRANSVERSE_PARTNERS``, so
    it agrees with ``stepping`` even if the CUDA text names the wrong D array.
    This reads the partner off the device string itself: per component, the first
    ``four_point_sum(`` argument must be the offset-1 partner's volume and the
    second the offset-2 partner's (MEEP's ``cycle_direction``, vec.hpp:586).

    ``Dsqr``'s sum commutes, so swapping the two is a NULL on the bits -- which is
    exactly why it has to be pinned as TEXT and cannot be measured on a device.
    """
    code = nlc._update_E_pml_real_nonlinear_kernel_code
    volumes = ("Dx", "Dy", "Dz")
    for component, _source, axis in nlc.E_TERMS:
        marker = f"// --- component {axis}: {component} "
        assert marker in code, f"the body no longer labels component {axis}"
        block = code[code.index(marker):]
        end = code.find("// --- component", code.index(marker) + len(marker))
        if end != -1:
            block = code[code.index(marker):end]
        called = re.findall(r"four_point_sum\(\s*\n?\s*(D[xyz]),", block)
        expected = [volumes[index] for index in nlc.TRANSVERSE_PARTNERS[axis]]
        assert called == expected, (
            f"{component} sums {called} where cycle_direction gives {expected}")


def test_the_device_string_is_pure_ascii_and_encodes_under_a_C_locale():
    """MEASURED 2026-08-15: two em-dashes killed a kernel at its first launch.

    ``cupy.cuda.compiler`` writes the source with a bare ``open(..., 'w')``, so
    the bytes go through the interpreter's LOCALE encoding -- ASCII under C/POSIX,
    which is what a non-interactive shell on the validation host gets. The encode
    is PERFORMED rather than a character range inspected.
    """
    code = nlc._update_E_pml_real_nonlinear_kernel_code
    code.encode("ascii")  # raises UnicodeEncodeError on the defect this pins
    assert code.isascii()


def test_the_division_is_the_round_to_nearest_intrinsic():
    """``__fdiv_rn``, not ``/``. The sibling Triton track measured that difference.

    On that platform the plain operator lowered to ``div.full.f32`` (~2 ulp) and
    every nonlinear sweep case diverged. CUDA's default is ``-prec-div=true``, so
    the two spellings are expected to AGREE here -- which is a claim about this
    platform, and the gate probes it rather than this test asserting it.
    """
    code = nlc._update_E_pml_real_nonlinear_kernel_code
    assert "__fdiv_rn(" in code
    body = code[code.index("__device__ __forceinline__ float pade_u"):]
    body = body[:body.index("\n}")]
    assert "/" not in body.replace("//", ""), (
        "pade_u contains a bare division; the transcription divides once and it "
        "must be the round-to-nearest intrinsic")


# ---------------------------------------------------------------------------
# The certification record, against the artifact it names
# ---------------------------------------------------------------------------

RECORD = nlc.NONLINEAR_CONSTITUTIVE_CERTIFICATION
ARTIFACTS = HERE.parents[1] / RECORD["artifacts"].split("apps/api/")[-1]


def _gate(policy: str) -> dict:
    path = ARTIFACTS / policy / "gate.json"
    assert path.exists(), (
        f"{path} is missing; the record names an artifact directory that is not "
        f"in the tree, so nothing here is checkable")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("policy", ("keep", "flush"))
def test_the_record_matches_the_artifact_it_names(policy):
    """Every number in the record, read back off the gate's own JSON.

    A record transcribed by hand is a record that drifts. These are the fields a
    reader would act on -- did it pass, on how many cases, with how many differing
    words -- and they are pinned against the file rather than trusted.
    """
    gate = _gate(policy)
    summary = gate["summary"]
    assert summary["released"] is True, summary["reasons"]
    assert summary["scored_cases"] == RECORD["scored_cases"][policy]
    assert summary["electric_cases"] == RECORD["electric_cases"][policy]
    assert summary["magnetic_cases"] == RECORD["magnetic_cases"][policy]
    assert summary["single_launch_identical"] == RECORD["single_launch_bit_identical"][policy]
    assert summary["multi_step_identical"] == RECORD["multi_step_bit_identical"][policy]
    assert summary["multi_step_cases"] == summary["multi_step_identical"]
    scored = [c for c in gate["sweep"]["fmad_false"] if not c.get("skipped")]
    words = sum(c["single_launch"]["total_floats"] for c in scored)
    words += sum(c["multi_step"]["total_floats"] for c in scored if "multi_step" in c)
    differing = sum(c["single_launch"]["differing_floats"] for c in scored)
    differing += sum(c["multi_step"]["differing_floats"] for c in scored
                     if "multi_step" in c)
    assert words == RECORD["words_compared"][policy]
    assert differing == RECORD["differing_words"][policy] == 0
    assert all(c["multi_step"]["launches"] == RECORD["multi_step_launches"]
               for c in scored if "multi_step" in c)


@pytest.mark.parametrize("policy", ("keep", "flush"))
def test_the_two_policies_compiled_distinct_binaries(policy):
    """The two legs must be two PROGRAMS, not one program measured twice.

    Under ``keep`` the strip must have removed ``-ftz=true`` from every NVRTC
    call; under ``flush`` every call must have carried it. Without that the two
    "policies" are one binary and the sweep is a repetition.
    """
    report = _gate(policy)["nvrtc_binary_report"]
    assert report["nvrtc_calls_observed"] > 0
    assert report["all_ftz_true_reached_nvrtc"] is (policy == "flush")
    assert report["any_ftz_true_reached_nvrtc"] is (policy == "flush")


@pytest.mark.parametrize("policy", ("keep", "flush"))
def test_every_mutation_leg_reached_its_verdict(policy):
    """No leg may be NOT ARMED, NO LEGS, UNACCOUNTED, PARTIAL or NULL VIOLATED.

    ``UNACCOUNTED`` is the one worth naming: it means the rewrite matched, the
    memo is keyed through the source, and yet no kernel was built from the mutated
    bytes -- a leg reporting a pass for a mutation it never applied.
    """
    gate = _gate(policy)
    allowed = {"CAUGHT", "NULL CONFIRMED"}
    bad = {key: leg.get("verdict", leg.get("why"))
           for key, leg in gate["source_mutations"].items()
           if not leg.get("armed") or leg["verdict"] not in allowed}
    bad.update({key: leg["verdict"] for key, leg in gate["host_mutations"].items()
                if leg["verdict"] not in allowed})
    assert not bad, bad
    caught = sum(1 for leg in gate["source_mutations"].values()
                 if leg.get("verdict") == "CAUGHT")
    caught += sum(1 for leg in gate["host_mutations"].values()
                  if leg["verdict"] == "CAUGHT")
    nulls = sum(1 for leg in gate["source_mutations"].values()
                if leg.get("verdict") == "NULL CONFIRMED")
    assert caught == RECORD["mutations_caught"]
    assert nulls == RECORD["mutations_null_confirmed"]
    assert nulls > 0, ("a battery of only-must-be-caught legs scores identically "
                       "whether the comparator works or fails everything")


@pytest.mark.parametrize("policy", ("keep", "flush"))
def test_the_H_arms_refusal_premise_was_armed_and_caught(policy):
    """Arm H is a NULL, and a null with no armed premise is a blind spot.

    ``pade_scale_the_H_source`` rewrites the CERTIFIED H device string so that it
    really does apply a Pade factor. If the comparator could not see that, the H
    arm's 100% pass rate would be consistent with a comparator that sees nothing.
    """
    leg = _gate(policy)["source_mutations"]["H:pade_scale_the_H_source"]
    assert leg["armed"] and leg["sites"] == 3
    assert leg["verdict"] == "CAUGHT" and leg["caught"] == leg["ran"] > 0
    assert leg["kernel_constructions_from_mutated_bytes"] > 0


def test_the_gate_bound_the_device_bytes_that_ship_today():
    """THE LOAD-BEARING PIN, and it is deliberately narrower than the file hash.

    ``certified_module_sha256`` stops matching the moment anyone fixes a typo in a
    docstring, which makes it useless as a pin on the thing that decides the
    verdict. What NVRTC compiled is the kernel source STRING, so that is what is
    checked here -- and the module hash is checked too, against either the hash
    the gate saw or a DECLARED post-gate edit that says why the verdict survives.
    A device-code edit therefore fails on a laptop rather than being discovered
    hours into a device run.
    """
    for name, attribute in (
            ("_NONLINEAR_PRELUDE", nlc._NONLINEAR_PRELUDE),
            ("update_E_pml_real_nonlinear",
             nlc._update_E_pml_real_nonlinear_kernel_code)):
        digest = hashlib.sha256(attribute.encode("utf-8")).hexdigest()
        assert RECORD["device_source_sha256"][name] == digest, (
            f"{name} has changed since the gate ran; NVRTC would compile "
            f"different bytes and the verdict has to be re-cut on a CUDA host")

    module = hashlib.sha256(
        (HERE / "nonlinear_constitutive.py").read_bytes()).hexdigest()
    declared = {RECORD["certified_module_sha256"]}
    declared.update(edit["revision_sha256"] for edit in RECORD["post_certification_edits"]
                    if edit["revision_sha256"])
    if module not in declared:
        edits = [edit for edit in RECORD["post_certification_edits"]
                 if not edit["touches_device_code"]]
        assert edits, (
            "nonlinear_constitutive.py has changed since the gate ran and "
            "post_certification_edits says nothing about it; declare the edit and "
            "say why the verdict survives, or re-cut on a CUDA host")

    for policy in ("keep", "flush"):
        stamped = _gate(policy)["imported_source_sha256"]
        key = "meep_gpu/cuda_kernels/nonlinear_constitutive.py"
        assert key in stamped, sorted(stamped)[:5]
        assert stamped[key] == RECORD["certified_module_sha256"], (
            f"the {policy} artifact was cut against a module hash the record does "
            f"not name; the record and the run disagree about which program ran")


def test_a_device_edit_would_fail_this_pin_on_a_laptop():
    """The pin above must be able to FAIL, or it is a decoration.

    Checked by digesting a one-character edit of the shipped string rather than by
    trusting that sha256 is sensitive: the thing being verified is that the TEST
    reads the shipped attribute and not a constant beside it.
    """
    mutated = nlc._update_E_pml_real_nonlinear_kernel_code.replace(
        "0.0625f", "0.0624f", 1)
    assert mutated != nlc._update_E_pml_real_nonlinear_kernel_code
    digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
    assert digest != RECORD["device_source_sha256"]["update_E_pml_real_nonlinear"]


def test_the_contraction_guard_control_is_reported_and_load_bearing():
    """``--fmad=false`` is claimed to be CORRECTNESS here; the control measures it."""
    for policy in ("keep", "flush"):
        control = _gate(policy)["summary"]["guard_control"]
        assert control["scored_at_inexact_courant"] > 0
        assert control["diverged"] > 0, (
            "the unguarded leg was bit-identical too, so this run carries no "
            "evidence that --fmad=false changed an answer; the record's "
            "contraction_guard_control field must say so instead")


def test_the_division_spelling_probe_confirmed_the_null():
    """The record's claim about ``/`` vs ``__fdiv_rn`` on this platform, checked."""
    for policy in ("keep", "flush"):
        probe = _gate(policy)["division_spelling_probe"]
        assert probe["armed"] and probe["sites"] == 1
        assert probe["verdict"] == "NULL CONFIRMED"
        assert probe["kernel_constructions_from_mutated_bytes"] > 0


def test_the_subnormal_class_reached_the_pade_terms_and_not_the_divide():
    """The ``div.rn`` FTZ question, answered with a census rather than an argument."""
    reach = _gate("keep")["summary"]["subnormal_reach_into_the_divide"]
    assert reach["cases"] > 0
    assert reach["chi_terms_subnormal"] > 0, (
        "no subnormal reached c2/c3, so this class measured nothing about the "
        "policy and the chi magnitudes need re-choosing")
    assert reach["quotient_operands_subnormal"] == 0, (
        "a subnormal reached the quotient itself; ptxas's FTZ-carrying div.rn "
        "range checks are LIVE on this expression and the kernel's note is wrong")


def test_every_shipped_kernel_is_in_exactly_one_of_the_two_sets():
    """A kernel added without a gate verdict fails here rather than shipping unmeasured."""
    source = (HERE / "nonlinear_constitutive.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    certified = uncertified = None
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id == "UNCERTIFIED_KERNELS":
            uncertified = set(ast.literal_eval(node.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "CERTIFIED_KERNELS"):
            certified = set(ast.literal_eval(node.value))
    assert certified is not None and uncertified is not None
    shipped = set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
    assert shipped, "no kernel declaration found; the scan is broken"
    assert certified.isdisjoint(uncertified)
    assert shipped == certified | uncertified, (
        f"shipped {sorted(shipped)} but the sets name {sorted(certified | uncertified)}")


def test_the_module_imports_without_a_device_library():
    """The predicate, the plan and the device text must answer on the merge bar."""
    assert nlc.cp is None or nlc.cp.__name__ == "cupy"
    assert callable(nlc.covers_real_pml_nonlinear_constitutive)
    assert isinstance(nlc._update_E_pml_real_nonlinear_kernel_code, str)


def test_the_launcher_refuses_both_or_neither_of_pml_and_tables(xp):
    fields, layer, _grid = build(xp, **CASES["all_periodic"])
    with pytest.raises(ValueError, match="exactly one"):
        nlc.update_E_fused_pml_real_nonlinear(fields, layer, tables={})
    with pytest.raises(ValueError, match="exactly one"):
        nlc.update_E_fused_pml_real_nonlinear(fields)


def test_the_boundary_codes_refuse_a_kind_this_kernel_has_no_ghost_rule_for(xp):
    fields, layer, grid = build(xp, symmetry="Y", cell=(8.0, 16.0, 12.0),
                                **CASES["all_periodic"])
    del fields, layer
    codes, refusal = nlc.boundary_codes(grid)
    assert codes is None and "mirror" in refusal

