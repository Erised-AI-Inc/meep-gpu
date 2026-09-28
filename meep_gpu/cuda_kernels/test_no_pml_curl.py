"""The hand-CUDA NO-ABSORBER curl family: what the array path does, and what the predicate says.

WHAT THIS FILE CAN SETTLE ON A LAPTOP AND WHAT IT CANNOT. The bit-identity claim
for ``step_{B,D}_no_pml_real`` and ``step_B_no_pml_real_derived``
belongs to ``parity/meep_gpu/gate_cuda_no_pml_curl.py``, on a device, under both
float32 subnormal policies and with the NVRTC contraction guard as an
experimental variable. NOTHING HERE COMPILES ANYTHING. What runs here is
everything else, and it is most of what would be wrong:

1. **THE ORACLE'S SHAPE, MEASURED.** The three kernels exist in the shape they do
   because of three facts about ``stepping`` that were measured rather than read:
   without an absorber ``step_B`` differences ``D * inv_eps`` when nothing stores
   E, ``step_D`` differences the B ARRAYS, and an INERT ``PML`` object steps
   identically to no layer at all. Each is asserted here against the real
   ``stepping`` on real ``Grid``/``Fields`` objects, as uint32 words. If any of
   them stopped being true, the kernels would be solving the wrong problem and no
   amount of device green would say so.

2. **THE DEVICE TEXT'S STRUCTURE.** The two verbatim kernels must differ from the
   certified ``step_{B,D}_pml_real`` in the TAIL AND NOWHERE ELSE, and the
   derived kernel must differ from the stored one only by the documented
   substitution. That is the file's whole safety argument, and it is checkable
   rather than reviewable.

3. **THE PREDICATE, CLAUSE BY CLAUSE**, pinned to exact refusal strings -- in a
   first-refusal predicate the clause ORDER is part of the contract -- plus the
   two structural properties the delegation rests on: that the ONE forgiven
   refusal is the sibling's LAST clause (so forgiving it skips nothing), and that
   the clauses which follow ``stores_E`` are still reached.

4. **THE GATE'S NUMPY BACKEND, END TO END.** A handful of the gate's own cases are
   run in process, so the transcription the device leg's laptop half depends on is
   exercised on every laptop rather than for the first time on the one host that
   runs the gate.

5. **MUTATIONS OF THE PREDICATE AND OF THE DEVICE TEXT.** Every clause must change
   a verdict when it is deleted from the REAL source, and every mutation the gate
   names must ARM on the kernel it claims -- a mutation silently absent from a
   battery and a mutation that cannot bite look identical in a verdict.
"""

from __future__ import annotations

import difflib
import importlib.util
import pathlib
import re
import sys
import types

import numpy
import pytest

from .. import stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import coverage
from . import no_pml_curl
from .no_pml_curl import covers_no_pml_curl as covers

HERE = pathlib.Path(__file__).parent
MODULE_PATH = HERE / "no_pml_curl.py"
SIBLING_PATH = HERE / "step_curl_kernels.py"
STEPPING_PATH = HERE.parent / "stepping.py"
GATE_PATH = (HERE.parent.parent / "parity" / "meep_gpu"
             / "gate_cuda_no_pml_curl.py")

SUB_STEPS = ("step_B", "step_D")


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in.

    The delegated predicate's first question is whether the backend is CuPy at
    all, and that is the one thing about the device library a laptop cannot
    supply. Everything else -- the fold, the stored extent, the dtype, the
    contiguity, the inverse-epsilon volumes -- is a real object either way.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def make(xp, *, boundaries=("periodic", "metallic", "periodic"), cell=(8.0, 10.0, 12.0),
         axes="", phase=1, stored=False, dimensions=3, courant=0.5,
         inhomogeneous_epsilon=False, seed=3):
    """A real ``(fields, grid)`` pair with no absorber."""
    planes = tuple(Mirror(name, phase) for name in axes)
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=planes, dimensions=dimensions, xp=xp, courant=courant)
    fields = Fields(grid=grid)
    if stored:
        fields.enable_field_storage()
    elif inhomogeneous_epsilon:
        rng = numpy.random.default_rng(seed)
        inverse, epsilon = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(0.2, 0.9, size=grid.shape).astype(numpy.float32)
            inverse[component] = values
            epsilon[component] = (1.0 / values).astype(numpy.float32)
        fields.set_epsilon_volumes(epsilon, inverse)
    return fields, grid


def seed_arrays(fields, grid, names, seed=11):
    rng = numpy.random.default_rng(seed)
    for name in names:
        getattr(fields, name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(numpy.float32)


def words(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(numpy.asarray(array),
                                   dtype=numpy.float32).ravel().view(numpy.uint32)


# ---------------------------------------------------------------------------
# 1. WHAT THE ARRAY PATH ACTUALLY DOES WITHOUT AN ABSORBER
#
# Every one of these is the reason a kernel here has the signature it has. They
# are measured against ``stepping`` itself, as uint32 words, with a nonzero
# oracle-moved floor -- an identity assertion over a sub-step that moved nothing
# would pass for a reason about the fixture.
# ---------------------------------------------------------------------------

def _shift(field, axis, boundary, up):
    shifted = numpy.roll(field, -1 if up else 1, axis=axis)
    if boundary != "periodic":
        selector = [slice(None)] * 3
        selector[axis] = -1 if up else 0
        shifted[tuple(selector)] = numpy.float32(0.0)
    return shifted


def _hand_curl_step(fields, grid, boundaries, terms, operands, targets, backward):
    """The kernels' arithmetic, by hand, in float32: ``target -= dtdx*((sf-f1)+(f2-ss))``."""
    dtdx = numpy.float32(grid.dt / grid.dx)
    out = {}
    for target, first, first_axis, second, second_axis in terms:
        f1, f2 = operands[first], operands[second]
        sf = _shift(f1, first_axis, boundaries[first_axis], not backward)
        ss = _shift(f2, second_axis, boundaries[second_axis], not backward)
        curl = (dtdx * ((sf - f1) + (f2 - ss))).astype(numpy.float32)
        from ..fields import IYEE_SHIFTS
        for axis in range(3):
            if IYEE_SHIFTS[target][axis] == 0 and boundaries[axis] == "metallic":
                selector = [slice(None)] * 3
                selector[axis] = 0
                curl[tuple(selector)] = numpy.float32(0.0)
        out[target] = (targets[target] - curl).astype(numpy.float32)
    return out


def test_without_stored_E_step_B_differences_D_times_inverse_epsilon(xp):
    """``stepping.step_B`` derives its operands; ``fields.Ex`` does not even exist.

    THIS IS WHY THE DERIVED KERNEL EXISTS. ``_read_component`` (stepping.py:2438)
    routes E through ``get_E``'s un-stored branch, ``D * inv_eps``
    (fields.py:1035), whenever ``fields.stores_E`` is False -- which, with no
    absorber and no susceptibility, is the normal case. A kernel binding
    Ex/Ey/Ez cannot serve it: the pointers are None.
    """
    boundaries = ("periodic", "metallic", "periodic")
    fields, grid = make(xp, boundaries=boundaries, inhomogeneous_epsilon=True)
    assert not fields.stores_E
    assert fields.Ex is None, "a run with no absorber and no pole stores no E"
    seed_arrays(fields, grid, ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))

    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Bx", "By", "Bz")}
    operands = {
        "E" + axis: (numpy.asarray(getattr(fields, "D" + axis))
                     * numpy.asarray(fields.inverse_epsilon_for("E" + axis))
                     ).astype(numpy.float32)
        for axis in "xyz"}
    expected = _hand_curl_step(
        fields, grid, boundaries,
        (("Bx", "Ez", 1, "Ey", 2), ("By", "Ex", 2, "Ez", 0), ("Bz", "Ey", 0, "Ex", 1)),
        operands, frozen, backward=False)

    stepping.step_B(fields, None)

    moved = sum(int(numpy.count_nonzero(words(frozen[n]) != words(getattr(fields, n))))
                for n in ("Bx", "By", "Bz"))
    assert moved > 0, "the sub-step moved nothing; this case would certify nothing"
    for name in ("Bx", "By", "Bz"):
        assert numpy.array_equal(words(expected[name]), words(getattr(fields, name))), (
            f"{name}: stepping.step_B is not the curl of D * inv_eps; the derived "
            f"kernel is solving a different problem than the array path")


def test_step_D_differences_the_B_arrays_because_get_H_returns_them(xp):
    """``fields.Hx`` is None and ``get_H`` hands back B. THE TRAP THIS FAMILY EXISTS FOR.

    A predicate that delegates its storage inventory to the certified curl
    predicate refuses every no-absorber run for "Hx is not allocated" -- and the
    refusal is FALSE, because the array path differences Bx/By/Bz directly.
    """
    boundaries = ("periodic", "metallic", "periodic")
    fields, grid = make(xp, boundaries=boundaries)
    assert fields.Hx is None, "H is never stored without an absorber"
    for name in ("Hx", "Hy", "Hz"):
        assert fields.get_H(name) is getattr(fields, "B" + name[1]), (
            "get_H's no-absorber branch must alias the B array itself")
    seed_arrays(fields, grid, ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))

    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz")}
    operands = {"H" + axis: numpy.asarray(getattr(fields, "B" + axis)).copy()
                for axis in "xyz"}
    expected = _hand_curl_step(
        fields, grid, boundaries,
        (("Dx", "Hz", 1, "Hy", 2), ("Dy", "Hx", 2, "Hz", 0), ("Dz", "Hy", 0, "Hx", 1)),
        operands, frozen, backward=True)

    stepping.step_D(fields, None)

    moved = sum(int(numpy.count_nonzero(words(frozen[n]) != words(getattr(fields, n))))
                for n in ("Dx", "Dy", "Dz"))
    assert moved > 0
    for name in ("Dx", "Dy", "Dz"):
        assert numpy.array_equal(words(expected[name]), words(getattr(fields, name))), (
            f"{name}: stepping.step_D is not the backward curl of the B arrays")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_an_inert_layer_steps_identically_to_no_layer(sub_step, xp):
    """A zero-thickness ``PML`` object is INACTIVE, and this family admits it.

    ``_pml_is_active`` is ``pml is not None and pml.is_active`` and
    ``PML.is_active`` is ``any(low > 0 or high > 0)`` -- so a layer present but
    inert takes the same tail. The predicate admits it on that reading; this is
    the measurement behind the reading, and without it "inert is the same as
    absent" would be an inference about ``pml.py``.
    """
    boundaries = ("periodic", "metallic", "metallic")
    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
    outcome = {}
    for absorber in ("none", "inert"):
        fields, grid = make(xp, boundaries=boundaries, stored=True)
        seed_arrays(fields, grid, names + ("Ex", "Ey", "Ez"))
        layer = (PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
                 if absorber == "inert" else None)
        if absorber == "inert":
            assert not layer.is_active
        getattr(stepping, sub_step)(fields, layer)
        outcome[absorber] = {n: words(getattr(fields, n)).copy() for n in names}
    for name in names:
        assert numpy.array_equal(outcome["none"][name], outcome["inert"][name]), (
            f"{name}: an inert layer changed the step; the predicate admits both "
            f"and one of them is not what it thinks it is")


# ---------------------------------------------------------------------------
# 2. THE DEVICE TEXT'S STRUCTURE
# ---------------------------------------------------------------------------

def _kernel_body(source: str, entry: str) -> str:
    """One kernel's ``extern "C"`` block, prelude stripped."""
    marker = f'extern "C" __global__ void {entry}'
    index = source.index(marker)
    return source[index:]


def test_the_prelude_is_the_siblings_own_bytes():
    """Read out of ``step_curl_kernels.py``, not copied into this file.

    Two copies of ``shift_up`` are equal only until someone edits one, and a
    silent divergence there is a wrong answer no diff of ``no_pml_curl.py`` would
    show. The ``ast`` route is checked here against a crude textual one so a
    change of shape in the sibling fails loudly rather than yielding a stale
    string.
    """
    sibling = SIBLING_PATH.read_text(encoding="utf-8")
    start = sibling.index("_REAL_PML_PRELUDE = r'''") + len("_REAL_PML_PRELUDE = r'''")
    end = sibling.index("'''", start)
    assert no_pml_curl._REAL_PML_PRELUDE == sibling[start:end]
    assert "shift_up" in no_pml_curl._REAL_PML_PRELUDE
    assert "BC_MIRROR_PERIODIC" in no_pml_curl._REAL_PML_PRELUDE


@pytest.mark.parametrize("sub_step,mine,theirs", [
    ("step_B", "step_B_no_pml_real", "step_B_pml_real"),
    ("step_D", "step_D_no_pml_real", "step_D_pml_real"),
])
def test_the_verbatim_kernels_differ_from_the_certified_pair_only_in_the_tail(
        sub_step, mine, theirs):
    """Line for line, the ONLY changes are the signature and the three tails.

    THE WHOLE SAFETY ARGUMENT OF THIS FAMILY is that the curl -- the term table,
    the ghost rule, the grouping and the six wall-mask lines -- is the certified
    pair's, unedited. That is a property of the bytes, so it is measured on the
    bytes: every changed line must be a ``pml_apply``/``fu``/coefficient line
    (removed) or a ``target = target - curl`` line (added). A changed mask line,
    or a re-grouped curl, fails here.
    """
    sibling = SIBLING_PATH.read_text(encoding="utf-8")
    start = sibling.index(f'extern "C" __global__ void {theirs}')
    end = sibling.index("'''", start)
    certified = sibling[start:end]
    ours = _kernel_body(no_pml_curl.kernel_source(mine), mine)

    changed = [line for line in difflib.unified_diff(
        certified.splitlines(), ours.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    allowed = re.compile(
        r"^[+-]\s*(//.*"                       # a comment
        r"|extern \"C\".*|.*step_._(?:no_pml_)?(?:pml_)?real\("  # the signature
        r"|.*float\* __restrict__ fu_.*"       # the auxiliary parameters
        r"|.*const float\* __restrict__ (?:kms|sinv)_.*"  # the coefficient parameters
        r"|.*pml_apply\(.*"                    # the recurrence call
        r"|\s*[BD][xyz]\[idx\] = [BD][xyz]\[idx\] - curl;"  # the new tail
        r")\s*$")
    unexplained = [line for line in changed if not allowed.match(line)]
    assert not unexplained, (
        f"{mine} differs from the certified {theirs} outside the tail:\n"
        + "\n".join(unexplained))
    assert sum(1 for line in changed
               if line.startswith("+") and "- curl;" in line) == 3, (
        "each of the three components must carry exactly one plain tail")


def test_the_derived_kernel_is_the_stored_one_with_the_derive_substituted():
    """``Ec[idx]`` -> ``derived_at(Dc, iec, idx)``, ``shift_up(Ec,`` -> ``derived_up(Dc, iec,``.

    Mechanically re-deriving the derived kernel from the stored one and comparing
    is what stops the two drifting: a mask line changed on one and not the other
    would be a defect on exactly the arm the corpus needs most.
    """
    stored = _kernel_body(no_pml_curl.kernel_source("step_B_no_pml_real"),
                          "step_B_no_pml_real")
    derived = _kernel_body(
        no_pml_curl.kernel_source("step_B_no_pml_real_derived"),
        "step_B_no_pml_real_derived")
    rebuilt = stored
    rebuilt = rebuilt.replace("step_B_no_pml_real(",
                              "step_B_no_pml_real_derived(")
    rebuilt = re.sub(r"const float\* __restrict__ Ex, const float\* __restrict__ Ey,\n"
                     r"    const float\* __restrict__ Ez,",
                     "const float* __restrict__ Dx, const float* __restrict__ Dy,\n"
                     "    const float* __restrict__ Dz,\n"
                     "    const float* __restrict__ iex, const float* __restrict__ iey,\n"
                     "    const float* __restrict__ iez,", rebuilt)
    rebuilt = re.sub(r"shift_up\(E([xyz]), (idx, \w+, n[xyz], s[xyz], bc_[xyz])\)",
                     lambda m: (f"derived_up(D{m.group(1)}, ie{m.group(1)}, "
                                f"{m.group(2)})"), rebuilt)
    rebuilt = re.sub(r"\bE([xyz])\[idx\]",
                     lambda m: f"derived_at(D{m.group(1)}, ie{m.group(1)}, idx)",
                     rebuilt)

    def normalise(text: str) -> list:
        return [line.rstrip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith("//")]

    assert normalise(rebuilt) == normalise(derived), (
        "the derived kernel is not the stored kernel with the derive substituted:\n"
        + "\n".join(difflib.unified_diff(normalise(rebuilt), normalise(derived),
                                         lineterm="", n=1)))


@pytest.mark.parametrize("name", no_pml_curl.NO_PML_KERNELS)
def test_the_shared_preludes_recurrence_helper_is_compiled_and_never_called(name):
    """``pml_apply`` is DEAD CODE in every kernel here, and that is the family.

    It is carried because the prelude is shared VERBATIM and forking it to drop
    three lines is the exact silent divergence the shared read exists to prevent.
    The gate turns the liability into evidence -- the four ``pml_apply`` mutations
    are scored as MUST-BE-UNCAUGHT nulls -- and this test is what makes "never
    called" a property of the bytes rather than of the reader's attention.
    """
    source = no_pml_curl.kernel_source(name)
    assert "pml_apply" in source, "the shared prelude is expected to define it"
    body = _kernel_body(source, name)
    assert "pml_apply" not in body, (
        f"{name} CALLS the split-field recurrence; this family's tail is "
        f"target -= curl and nothing else")
    for forbidden in ("fu_", "kms", "sinv"):
        assert forbidden not in body, (
            f"{name} still mentions {forbidden!r}; no auxiliary and no coefficient "
            f"vector exists on a run with no absorber")


def test_every_kernel_the_arm_table_names_exists_and_vice_versa():
    assert set(no_pml_curl.ARM_KERNELS.values()) == set(no_pml_curl.NO_PML_KERNELS)
    for name in no_pml_curl.NO_PML_KERNELS:
        assert f'void {name}(' in no_pml_curl.kernel_source(name)


# ---------------------------------------------------------------------------
# 3. THE PREDICATE
# ---------------------------------------------------------------------------

def test_the_arm_branches_on_the_flag_stepping_branches_on():
    """``no_pml_curl_arm`` reads ``stores_E``; so does ``stepping._read_component``.

    An arm chosen off anything else -- the presence of ``fields.Ex``, a
    polarization count -- would be right on today's corpus and wrong on the first
    configuration that separates them, and the failure would be a wrong kernel
    silently bound to the wrong pointers.
    """
    import inspect
    assert "fields.stores_E" in inspect.getsource(stepping._read_component)
    assert "stores_E" in inspect.getsource(no_pml_curl.no_pml_curl_arm)


def test_the_arm_follows_the_run_rather_than_the_caller(xp):
    stored, _ = make(xp, stored=True)
    derived, _ = make(xp, stored=False)
    assert no_pml_curl.no_pml_curl_arm(stored, "step_B") == "stored_E"
    assert no_pml_curl.no_pml_curl_arm(derived, "step_B") == "derived_E"
    for fields in (stored, derived):
        assert no_pml_curl.no_pml_curl_arm(fields, "step_D") == "magnetic"


def test_an_unnamed_sub_step_raises_rather_than_refusing(xp):
    """A typo must not read as "this configuration is not covered"."""
    fields, grid = make(xp)
    with pytest.raises(ValueError, match="step_B"):
        covers(fields, None, grid, "update_H")
    with pytest.raises(ValueError, match="step_B"):
        no_pml_curl.no_pml_curl_arm(fields, "update_E")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("stored", [False, True], ids=["derived_E", "stored_E"])
def test_a_plain_no_absorber_run_is_admitted_at_both_sub_steps(sub_step, stored, xp):
    """The case the family exists for, and the one the first version refused.

    Before the storage inventory was this file's own, ``step_B`` refused with "Ex
    is not allocated" on every derived run and ``step_D`` refused with "E is
    recomputed from D rather than stored" on every one of them -- a clause about
    an operand ``step_D`` does not read. Between them that was every real
    no-absorber row in the corpus except one.
    """
    fields, grid = make(xp, stored=stored, inhomogeneous_epsilon=not stored)
    assert covers(fields, None, grid, sub_step) == (True, "")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_susceptibility_puts_the_run_on_the_stored_arm_and_is_admitted(sub_step, xp):
    """The ONE corpus row the stored-E arm serves, and why it is stored.

    Of the eight real-storage no-absorber rows in the 2026-08-20 census, exactly
    one (``material-dispersion.py``) has ``stores_E`` True with no conductivity --
    because a registered susceptibility makes ``update_E`` write E
    (``enable_field_storage``, fields.py:653-676) so ``get_E`` cannot recompute a
    field one polarization out of date. The curl then differences the stored array
    and the plain kernel serves it. Dispersion is NOT a curl fact -- neither
    ``step_B`` nor ``_apply_curl`` mentions a polarization -- and the certified
    predicate stopped refusing it for exactly that reason; what this pins is that
    the ARM follows.
    """
    from ..dispersion import PolarizationState, Susceptibility

    fields, grid = make(xp)
    assert not fields.stores_E
    fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.1, gamma=0.05),
        grid.xp.full(grid.shape, numpy.float32(0.35)), grid, numpy.float32))
    fields.enable_field_storage()
    assert fields.has_polarizations and fields.stores_E
    assert no_pml_curl.no_pml_curl_arm(fields, "step_B") == "stored_E"
    assert covers(fields, None, grid, sub_step) == (True, "")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_an_inert_layer_object_is_admitted_and_an_active_one_is_not(sub_step, xp):
    fields, grid = make(xp, stored=True)
    inert = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
    assert covers(fields, inert, grid, sub_step) == (True, "")

    active_fields, active_grid = make(xp, stored=True)
    active = PML(grid=active_grid, thickness=((2, 2), (2, 2), (2, 2)))
    covered, reason = covers(active_fields, active, active_grid, sub_step)
    assert covered is False
    assert reason.startswith("an active absorber is installed")


def test_a_conductivity_reaches_one_curl_and_not_the_other(xp):
    """Per TARGET COMPONENT, exactly as ``_apply_curl`` reads it (stepping.py:508).

    MEEP's ``Absorber`` boundary layer IS a conductivity, and three of the eight
    real-storage no-absorber corpus rows carry one. Refusing both sub-steps would
    give up a slot that is served; admitting both would serve one that is not.
    """
    fields, grid = make(xp, stored=True)
    sigma = numpy.full(grid.shape, 0.3, dtype=numpy.float32)
    fields.set_d_conductivity({name: sigma for name in ("Dx", "Dy", "Dz")})
    assert covers(fields, None, grid, "step_B") == (True, "")
    covered, reason = covers(fields, None, grid, "step_D")
    assert covered is False and "conductivity" in reason

    magnetic, magnetic_grid = make(xp, stored=True)
    magnetic.set_b_conductivity({name: sigma for name in ("Bx", "By", "Bz")})
    covered, reason = covers(magnetic, None, magnetic_grid, "step_B")
    assert covered is False and "conductivity" in reason
    assert covers(magnetic, None, magnetic_grid, "step_D") == (True, "")


def test_an_allocated_auxiliary_is_refused_by_name(xp):
    """This family writes no ``fu``; one that exists would be left stale.

    Wrong on the NEXT sub-step rather than this one, which is why it is a refusal
    and not a silent overwrite.
    """
    fields, grid = make(xp, stored=True)
    fields.fu_Bx = grid.xp.zeros(grid.shape, dtype=numpy.float32)
    covered, reason = covers(fields, None, grid, "step_B")
    assert covered is False
    assert reason.startswith("fu_Bx is allocated while the layer is inactive")
    # ...and it is a step_B fact, not a whole-run one.
    assert covers(fields, None, grid, "step_D") == (True, "")


def test_the_launcher_raises_on_the_one_configuration_get_H_would_lie_about(xp):
    """A ``Fields`` left in PML storage mode makes ``get_H`` return the STORED H.

    The predicate refuses it (``fu_Dx`` is allocated, because
    ``enable_pml_storage`` allocates the auxiliaries alongside H), so the launcher
    raises rather than binding the wrong pointers -- the same split
    ``real_curl_boundary_codes`` takes. Without the guard the launcher and the
    predicate would be talking about different arrays on a configuration neither
    of them names.
    """
    fields, grid = make(xp, stored=True)
    fields.enable_pml_storage()
    assert fields.get_H("Hx") is fields.Hx is not fields.Bx
    covered, reason = covers(fields, None, grid, "step_D")
    assert covered is False and reason.startswith("fu_Dx is allocated")
    with pytest.raises(ValueError, match="PML storage mode"):
        no_pml_curl.step_no_pml_curl(fields, "step_D", (0, 0, 0), 0.5,
                                     arm="magnetic")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_non_float32_or_non_contiguous_operand_is_refused(sub_step, xp):
    fields, grid = make(xp, stored=True)
    target = "Bx" if sub_step == "step_B" else "Dx"
    saved = getattr(fields, target)
    setattr(fields, target, saved.astype(numpy.float64))
    covered, reason = covers(fields, None, grid, sub_step)
    assert covered is False and reason == f"{target} is float64, not float32"
    setattr(fields, target, saved)

    wide = numpy.zeros((grid.shape[0], grid.shape[1], grid.shape[2] * 2),
                       dtype=numpy.float32)[..., ::2]
    assert not wide.flags.c_contiguous
    setattr(fields, target, wide)
    covered, reason = covers(fields, None, grid, sub_step)
    assert covered is False and reason == f"{target} is not C-contiguous"


def test_a_float64_inverse_epsilon_is_refused_on_the_derived_arm_only(xp):
    """The dtype decides the ARITHMETIC, not just the launch.

    The array path's derived buffer is ``result_type(D, inv_eps)``
    (stepping.py:2449), so a float64 inverse epsilon makes the whole curl a
    float64 curl -- a different computation, not a different rounding. ``step_D``
    never forms the product and is unaffected, which is the asymmetry this test
    exists to pin.
    """
    fields, grid = make(xp, inhomogeneous_epsilon=True)
    fields._inv_eps_components = {
        component: numpy.ones(grid.shape, dtype=numpy.float64)
        for component in ("Ex", "Ey", "Ez")}
    covered, reason = covers(fields, None, grid, "step_B")
    assert covered is False
    assert reason == "inverse epsilon for Ex is float64, not float32"
    assert covers(fields, None, grid, "step_D") == (True, "")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_fold_is_admitted_at_both_terminations(sub_step, xp):
    for boundaries in (("periodic", "periodic", "periodic"),
                       ("periodic", "metallic", "periodic")):
        fields, grid = make(xp, boundaries=boundaries, cell=(8.0, 16.0, 12.0),
                            axes="Y", stored=True)
        assert covers(fields, None, grid, sub_step) == (True, "")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_third_simultaneous_fold_plane_is_refused(sub_step, xp):
    """THIS FAMILY'S OWN CAP since 2026-08-20, and the reason it stopped being
    inherited.

    It WAS inherited: the sibling refused a third plane and delegation carried the
    refusal here for free. On 2026-08-20 the sibling's gate swept a third plane on
    the PML kernels and ``coverage.CURL_FOLD_ADMISSION`` went to 3 -- which would
    have widened this family on a measurement of DIFFERENT KERNELS.
    ``gate_cuda_no_pml_curl.py``'s case table stops at ``"XY"``, so the cap is now
    stated here, read from
    :data:`no_pml_curl.NO_PML_CURL_NARROWER_THAN_THE_SIBLING`, and it costs zero
    slots: the corpus's triply-folded row is PML-active.
    """
    fields, grid = make(xp, boundaries=("periodic",) * 3, cell=(16.0, 16.0, 16.0),
                        axes="XYZ", stored=True)
    covered, reason = covers(fields, None, grid, sub_step)
    assert covered is False and "3 mirror planes at once" in reason
    assert "this family's gate swept 2" in reason, reason
    # AND THE SIBLING MUST DISAGREE, or the clause is not this family's and the
    # test would pass on inheritance again without saying so.
    assert coverage.covers_real_pml_curl(fields, None, grid, sub_step)[1] != reason


# THE CLAUSES BEHIND THE FORGIVEN ONE. Each of these lives AFTER ``stores_E`` in
# ``covers_real_pml_curl``, so each of them was silently skipped while the
# delegation forgave that refusal by string instead of answering it at the input.
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_clauses_that_follow_stores_E_are_still_reached(sub_step, xp):
    """chi2/chi3 on a derived-E run, and where the clause now lives.

    A STRING FILTER OVER A SHORT-CIRCUITING PREDICATE SKIPS EVERYTHING BEHIND THE
    REFUSAL IT FORGIVES, and ``stores_E`` is clause 12 of 16; this case is the arm
    such a filter would have waved through. The clause itself moved on 2026-08-20:
    the sibling retired its chi2/chi3 refusal on a device round that ran the PML
    kernels (``coverage.CURL_NONLINEAR_ADMISSION``), and this family's gate
    installs no chi anywhere, so the refusal is now stated HERE and asked before
    delegation. It costs zero slots -- both corpus nonlinear rows are PML-active.
    """
    fields, grid = make(xp, inhomogeneous_epsilon=True)
    zero = numpy.zeros(grid.shape, dtype=numpy.float32)
    chi3 = numpy.full(grid.shape, 0.1, dtype=numpy.float32)
    fields.set_nonlinear_volumes({c: zero for c in ("Ex", "Ey", "Ez")},
                                 {c: chi3 for c in ("Ex", "Ey", "Ez")})
    covered, reason = covers(fields, None, grid, sub_step)
    assert covered is False and "chi2/chi3" in reason


def test_the_forgiven_refusal_is_the_siblings_last_clause(xp):
    """The ONE refusal this file forgives must be the last one the sibling can state.

    Forgiving an EARLY refusal from a short-circuiting predicate admits everything
    behind it. The coefficient-vector clause is the sibling's last, so forgiving
    it skips nothing -- and that ordering is a property of ``coverage.py`` that
    this file depends on and does not own. Measured here rather than trusted: the
    source is read and the coefficient loop is required to be the final statement
    before the ``return True``.
    """
    import inspect
    source = inspect.getsource(coverage.covers_real_pml_curl)
    coefficient = source.index("_coefficient_vector_problem")
    array_loop = source.index('for name in ("Bx", "By", "Bz"')
    stores = source.index('"E is recomputed from D rather than stored"')
    assert stores < array_loop < coefficient, (
        "covers_real_pml_curl reordered its clauses; the no-PML delegation "
        "forgives the coefficient refusal ONLY because nothing follows it")
    tail = source[coefficient:]
    assert tail.count("return False") == 1, (
        "another refusal now follows the coefficient loop and would be forgiven "
        "with it")

    # ...and the forgiven prefix is really the one that function produces.
    problem = coverage._coefficient_vector_problem("kms_x", None, xp, 0, (4, 4, 4))
    assert problem.startswith(no_pml_curl._FORGIVEN_REFUSAL_PREFIX)


# ---------------------------------------------------------------------------
# 4. THE GATE'S NUMPY BACKEND, IN PROCESS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gate():
    """The device gate, imported by path. It compiles nothing on this backend."""
    if not GATE_PATH.exists():  # pragma: no cover - only if the tree is partial
        pytest.skip("the gate is not present in this checkout")
    parity_dir = str(GATE_PATH.parent)
    if parity_dir not in sys.path:
        sys.path.insert(0, parity_dir)
    spec = importlib.util.spec_from_file_location("gate_cuda_no_pml_curl_test",
                                                  GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - a partial parity tree
        pytest.skip(f"the gate's dependencies are not importable here: {exc}")
    return module


@pytest.mark.parametrize("label", ["3d_mixed", "fold_Y_periodic", "2d_metallic_walls",
                                   "1d_metallic"])
@pytest.mark.parametrize("sub_step,e_storage", [("step_B", "stored"),
                                                ("step_B", "derived"),
                                                ("step_D", "derived")])
def test_the_gate_numpy_backend_reproduces_stepping(gate, label, sub_step, e_storage):
    """The transcription the gate's laptop half rests on, exercised on every laptop.

    IT CERTIFIES NOTHING ABOUT THE DEVICE -- it compiles nothing and is not the
    shipped bytes. What it settles is that the harness agrees with ``stepping``
    where it should, so a divergence on the device leg is about the compiler or
    the kernel and not about the fixture.
    """
    spec = next(s for s in gate.SPECS if s["label"] == label)
    case = gate.one_case("numpy", spec, sub_step, e_storage,
                         gate.INEXACT_COURANT, "subnormal_band", "fmad_false",
                         gate.LICENSED_SUBSTITUTION)
    assert not case.get("skipped"), case.get("skipped")
    assert case["oracle_moved"] > 0.0
    assert case["single_launch"]["bit_identical"], case["localization_summary"]
    assert case["multi_step"]["bit_identical"], "60 launches diverged"
    assert case["predicate_today"]["covered"], case["predicate_today"]["reason"]


def test_the_gate_can_fail(gate):
    """A deliberately misdeclared code triple must diverge, or the gate measures nothing."""
    spec = next(s for s in gate.SPECS if s["label"] == "fold_Y_periodic")
    case = gate.one_case("numpy", spec, "step_B", "derived", gate.INEXACT_COURANT,
                         "uniform", "fmad_false", "mirror_as_metallic")
    assert not case.get("skipped")
    assert not case["single_launch"]["bit_identical"]
    assert case["localization_summary"]["differing_words"] > 0
    assert case["localization_summary"]["differing_elsewhere"] == 0, (
        "BC_METALLIC and BC_MIRROR_PERIODIC share a ghost rule, so the only "
        "difference between them is the top-plane mask; a differing word off the "
        "mask-delta planes means the control moved something else")


def test_a_one_dimensional_grid_masks_a_plane_that_was_never_live(gate):
    """The floor's real case, and why it gates the MUTATION legs and not the sweep.

    On a 1-D z grid ``curl_z = dEy/dx - dEx/dy`` is identically zero -- both
    transverse axes are invariant, so both differences are exact zeros -- and Bz's
    metallic cell-0 mask therefore masks nothing that was ever nonzero. That is a
    true fact about 1-D and not a degenerate fixture, so the case is still SCORED
    for identity; what it is excluded from is mutation scoring, where
    ``drop_metallic_mask`` would come back UNCAUGHT for a reason about the shape.
    """
    spec = next(s for s in gate.SPECS if s["label"] == "1d_metallic")
    case = gate.one_case("numpy", spec, "step_B", "stored", gate.INEXACT_COURANT,
                         "uniform", "fmad_false", gate.LICENSED_SUBSTITUTION)
    assert not case.get("skipped"), "the 1-D identity evidence must still be scored"
    assert case["single_launch"]["bit_identical"]
    assert not case["masked_planes_are_live"]["meets_floor"]
    dead = [entry for entry in case["masked_planes_are_live"]["planes"]
            if entry["nonzero_words"] == 0]
    assert [entry["target"] for entry in dead] == ["Bz"]

    scorable, excluded = gate.scorable_baselines(
        {"sweep": {"fmad_false": [case]}})
    assert not scorable, "a dead-mask triple must not be scored for mutations"
    assert any("identically zero unmasked curl" in entry["why"]
               for entry in excluded), "the exclusion must say why"


def test_the_gates_floors_can_refuse_a_case(gate):
    """A degenerate inverse-epsilon fixture is SKIPPED, not passed.

    Against a table of ones ``drop_derived_inverse_epsilon`` is invisible; a gate
    that scored such a case would be reporting a property of the fixture.
    """
    fields, grid = make(_NumpyWearingCupysName(), inhomogeneous_epsilon=False)
    profile = gate.epsilon_profile(fields, "derived_E")
    assert profile["applies"] and not profile["meets_floor"]
    good, _ = make(_NumpyWearingCupysName(), inhomogeneous_epsilon=True)
    assert gate.epsilon_profile(good, "derived_E")["meets_floor"]


def test_the_gate_terms_agree_with_stepping(gate):
    assert gate.check_terms_against_stepping()["agreed"]
    assert gate.check_arm_selection_against_stepping()["agreed"]


# ---------------------------------------------------------------------------
# 5. MUTATIONS
# ---------------------------------------------------------------------------

def test_every_source_mutation_the_gate_names_arms_on_the_kernel_it_claims(gate):
    """A mutation silently absent from a battery looks exactly like one that cannot bite.

    Each entry is applied to the kernel the gate would apply it to, and required
    to match at least one site and to change the text. The derive mutations are
    required to match on the DERIVED kernel and to match NOWHERE on the other two,
    because a leg scored on a kernel the defect cannot reach is a free pass.
    """
    for name in gate.SOURCE_MUTATIONS:
        transform = gate.SOURCE_MUTATIONS_ALL[name]
        needs_arm = gate.SOURCE_MUTATION_REQUIRES_ARM.get(name)
        for (sub_step, arm), kernel in no_pml_curl.ARM_KERNELS.items():
            source = no_pml_curl.kernel_source(kernel)
            mutated, sites = transform(source)
            if needs_arm is not None and arm != needs_arm:
                assert sites == 0, (
                    f"{name} matched {sites} site(s) in {kernel}, which the gate "
                    f"scores it on only for arm {needs_arm}")
                continue
            assert sites > 0 and mutated != source, (
                f"{name} matched {sites} site(s) in {kernel} and changed nothing")


def test_every_leg_filter_names_a_mutation_the_gate_actually_runs(gate):
    """A filter on a mutation nobody runs is dead configuration that reads as a rule."""
    for table in (gate.SOURCE_MUTATION_LEG_FILTER,
                  gate.SOURCE_MUTATION_LEG_REQUIREMENT,
                  gate.SOURCE_MUTATION_REQUIRES_ARM):
        assert set(table) <= set(gate.SOURCE_MUTATIONS), sorted(
            set(table) - set(gate.SOURCE_MUTATIONS))
    assert set(gate.SOURCE_MUTATION_LEG_FILTER) == set(
        gate.SOURCE_MUTATION_LEG_REQUIREMENT), (
        "every filter must be stated in words in the artifact too")
    assert set(gate.NULL_SOURCE_MUTATIONS) <= set(gate.SOURCE_MUTATIONS)


def test_the_index_decomposition_twin_is_computed_rather_than_argued(gate):
    """Which shapes can see the Fortran-order swap is EVALUATED, not reasoned about.

    Measured 2026-08-20 on the GPU host (keep leg, GPU 6): the mutation was caught on
    16 legs and identical on the two `1d_metallic` ones, shape (1, 1, 24), where
    the C-order and Fortran-order decompositions agree cell for cell. The filter
    evaluates both decompositions over every index rather than encoding a rule
    about shapes, so a spec added to this gate cannot land on the wrong side of a
    rule nobody re-derived.
    """
    observable = gate.index_decomposition_is_observable
    assert not observable((1, 1, 24)), "the measured escape must read as inert"
    assert not observable((7, 1, 1))
    assert not observable((1, 9, 1))
    assert observable((8, 10, 12))
    assert observable((12, 14, 1))
    seen = {spec["label"]: observable(gate.spec_shape(spec)) for spec in gate.SPECS}
    assert any(seen.values()) and not all(seen.values()), (
        f"this gate's specs must contain both kinds or one leg never runs: {seen}")


def test_the_folded_far_ghost_leg_is_a_null_and_the_metallic_one_is_not(gate):
    """The same transform, two names, DISJOINT grids -- and only one may be inert.

    MEASURED on the GPU host (GPU 6, smoke leg, 2026-08-20): dropping the zero far
    ghost from ``derived_up`` was caught on ``3d_metallic`` (505 differing words)
    and ``3d_mixed`` (370) and came back IDENTICAL on ``fold_Y_periodic``. The
    fold's top-plane mask drops exactly the plane that consumes the changed ghost,
    which is the folded sibling's "the folded ghost values are dead" measured on
    this family. Two names keep that a measurement instead of an escaped leg -- and
    the two filters must not overlap, or one grid would be required to both catch
    and not catch the same edit.
    """
    for base, twin, witness in (
            ("derived_far_ghost_wraps",
             "derived_far_ghost_wraps_on_a_folded_axis", "fold_Y_periodic"),
            ("derived_wrap_uses_own_inv_eps",
             "derived_wrap_uses_own_inv_eps_on_an_invariant_axis", "1d_metallic"),
            ("fortran_order_index_decomposition",
             "fortran_order_index_decomposition_on_a_flat_grid", "1d_metallic")):
        caught = gate.SOURCE_MUTATION_LEG_FILTER[base]
        null = gate.SOURCE_MUTATION_LEG_FILTER[twin]
        assert gate.SOURCE_MUTATIONS_ALL[base] is gate.SOURCE_MUTATIONS_ALL[twin], (
            f"{base} and {twin} must be the SAME transform or they measure two "
            f"different things")
        assert twin in gate.NULL_SOURCE_MUTATIONS
        assert base not in gate.NULL_SOURCE_MUTATIONS

        both, either = [], []
        for spec in gate.SPECS:
            codes, shape = gate.spec_boundary_codes(spec), gate.spec_shape(spec)
            if caught(codes, shape) and null(codes, shape):
                both.append(spec["label"])
            if caught(codes, shape) or null(codes, shape):
                either.append(spec["label"])
        assert not both, f"{base}: these grids are in both legs: {both}"
        assert either, f"{base}: neither leg would ever run"
        # ...and the grid the measurement was taken on is in the NULL leg.
        spec = next(s for s in gate.SPECS if s["label"] == witness)
        assert null(gate.spec_boundary_codes(spec), gate.spec_shape(spec)), (
            f"{witness} is the grid the device leg measured {base} inert on; it "
            f"must be scored under {twin}")


@pytest.mark.parametrize("name", ["drop_fu_store", "read_fprev_after_store",
                                  "swap_dsig_dsigu", "reload_fu_from_memory"])
@pytest.mark.parametrize("kernel", no_pml_curl.NO_PML_KERNELS)
def test_the_four_dead_prelude_nulls_edit_only_the_helper_nothing_calls(name, kernel,
                                                                       gate):
    """Their being UNCAUGHT on the device is evidence, and this is why.

    Each of these four edits ``pml_apply`` and nothing else. In the certified pair
    that helper IS the tail; here it is compiled and never called, so the device
    leg scores them as MUST-BE-UNCAUGHT nulls and their silence measures that this
    family really does not route through the split-field recurrence. That claim is
    only worth anything if the edit is confined to the dead helper -- which is a
    property of the bytes, checked here.
    """
    source = no_pml_curl.kernel_source(kernel)
    mutated, sites = gate.SOURCE_MUTATIONS_ALL[name](source)
    assert sites > 0 and mutated != source
    body_before = _kernel_body(source, kernel)
    body_after = _kernel_body(mutated, kernel)
    assert body_before == body_after, (
        f"{name} changed the body of {kernel}; it is scored as a NULL on the "
        f"device leg and would be a free pass")
    helper_before = source[:source.index('extern "C"')]
    helper_after = mutated[:mutated.index('extern "C"')]
    assert helper_before != helper_after


def test_mutation_dropping_the_conductivity_delegation_is_caught(xp):
    """Delete the clause from the REAL source and the verdict must move.

    The delegated predicate is what refuses a conductivity, so the mutation is on
    the DELEGATION: a version that answered ``covers_real_pml_curl``'s question
    itself, or skipped it, would admit MEEP's Absorber rows into a kernel that
    implements the wrong recurrence.
    """
    fields, grid = make(xp, stored=True)
    sigma = numpy.full(grid.shape, 0.3, dtype=numpy.float32)
    fields.set_b_conductivity({name: sigma for name in ("Bx", "By", "Bz")})
    assert covers(fields, None, grid, "step_B")[0] is False

    source = MODULE_PATH.read_text(encoding="utf-8")
    needle = ("    covered, reason = _coverage.covers_real_pml_curl(\n"
              "        _NoAbsorberFieldsProxy(fields), _ActiveLayerProxy(pml), "
              "grid, sub_step)\n")
    assert needle in source, "the delegation call has been reworded; re-pin this test"
    mutated = source.replace(needle, "    covered, reason = True, ''\n")
    module = _load(mutated, "no_pml_curl_without_delegation")
    assert module.covers_no_pml_curl(fields, None, grid, "step_B")[0] is True, (
        "deleting the delegation did not change the verdict; the clause it "
        "carries is not load-bearing and this test measures nothing")


def test_mutation_forgiving_every_refusal_instead_of_the_coefficient_one_is_caught(xp):
    """The forgiveness must be a PREFIX MATCH, not a blanket pass.

    THE FIXTURE IS A DELEGATED REFUSAL, and which one matters. It was a chi3 grid
    until 2026-08-20, when the sibling retired its chi2/chi3 clause and this file
    took the refusal back as its OWN (it is asked before delegation, so a blanket
    forgiveness of the DELEGATED answer cannot reach it) -- and the mutation went
    green for a reason that said nothing about the prefix match. A grid the
    sibling refuses and this file does not own is what the test needs, so it uses a
    NONZERO BLOCH k: ``covers_real_pml_curl`` refuses it with "nonzero Bloch k",
    which has no ``pml.`` prefix, no clause in this file and no entry in the array
    inventory -- complex storage does not qualify, because the inventory catches
    it first with "Bx is complex64, not float32".
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 10.0, 12.0),
                boundaries=("periodic", "periodic", "periodic"),
                k_point=(0.3, 0.0, 0.0), xp=xp, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    covered, reason = covers(fields, None, grid, "step_B")
    assert covered is False and "nonzero Bloch k" in reason, reason
    assert not reason.startswith("pml."), (
        "the fixture's refusal is the forgiven one, so forgiving everything "
        "cannot be distinguished from forgiving the coefficient clause")

    source = MODULE_PATH.read_text(encoding="utf-8")
    needle = '_FORGIVEN_REFUSAL_PREFIX = "pml."'
    assert needle in source
    mutated = source.replace(needle, '_FORGIVEN_REFUSAL_PREFIX = ""')
    module = _load(mutated, "no_pml_curl_forgiving_everything")
    assert module.covers_no_pml_curl(fields, None, grid, "step_B")[0] is True


def test_mutation_dropping_the_auxiliary_clause_is_caught(xp):
    fields, grid = make(xp, stored=True)
    fields.fu_Bx = grid.xp.zeros(grid.shape, dtype=numpy.float32)
    assert covers(fields, None, grid, "step_B")[0] is False

    source = MODULE_PATH.read_text(encoding="utf-8")
    needle = '    for name in (f"fu_{stem}" for stem in targets):\n'
    assert needle in source
    mutated = source.replace(needle, "    for name in ():\n")
    module = _load(mutated, "no_pml_curl_without_aux_clause")
    assert module.covers_no_pml_curl(fields, None, grid, "step_B")[0] is True


def _load(source: str, name: str):
    """Execute a mutated copy of the module, package-relative imports intact."""
    # Named INSIDE the package so ``ModuleSpec.parent`` -- which is derived from
    # the name and is read-only -- agrees with ``__package__``; the relative
    # imports at the top of the module resolve off that, and a mismatch is a
    # DeprecationWarning today and an ImportError in a later Python.
    spec = importlib.util.spec_from_loader(f"{no_pml_curl.__package__}.{name}",
                                           loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(MODULE_PATH)
    module.__package__ = no_pml_curl.__package__
    exec(compile(source, str(MODULE_PATH), "exec"), module.__dict__)
    return module


# ---------------------------------------------------------------------------
# 6. THE ADMISSION RECORD
# ---------------------------------------------------------------------------

def test_admission_record_is_all_or_nothing():
    """A half-filled device record is worse than an empty one.

    Every field that only a device run can fill must be filled together or not at
    all, so a green laptop suite can never be mistaken for a device verdict. Same
    rule, same reason, as ``no_pml_constitutive.NULL_CONSTITUTIVE_CONFIRMATION``.
    """
    record = no_pml_curl.NO_PML_CURL_ADMISSION
    device_fields = ("artifacts", "recorded_utc", "host", "device", "cases_scored",
                     "single_launch_identical", "multi_step_identical",
                     "kernel_source_sha256", "source_mutations_caught",
                     "source_mutations_null_confirmed", "host_mutations_caught",
                     "contraction_guard")
    filled = [name for name in device_fields if record[name] is not None]
    assert len(filled) in (0, len(device_fields)), (
        f"the admission record is half filled: {sorted(filled)}. Either a device "
        f"leg ran and every field is a number, or it did not and none of them is.")
    assert record["gate"].endswith("gate_cuda_no_pml_curl.py")
    assert set(record["arms_swept"]) == {arm for _, arm in no_pml_curl.ARM_KERNELS}


def test_the_recorded_device_digests_are_the_device_text_that_ships():
    """The record pins the THREE DEVICE STRINGS, and they are re-measured here.

    NO SKIP AND NO BRANCH ON EMPTINESS: an empty record has an empty digest map
    and the loop below runs zero times, which is the right amount of checking for
    a family no device has seen. Once a leg HAS run, all three digests are
    recomputed from :func:`kernel_source`, so editing a kernel without a re-run
    fails here.

    THE RECORD PINS THE DEVICE TEXT AND NOT THE FILE, deliberately. A record
    written after a gate ran cannot be inside the bytes that gate stamped, so a
    whole-file digest would make every comment edit read as an un-gated kernel
    change; the device strings are the thing that must not have moved.
    """
    import hashlib

    recorded = no_pml_curl.NO_PML_CURL_ADMISSION["kernel_source_sha256"] or {}
    assert set(recorded) in ({}, set(), set(no_pml_curl.NO_PML_KERNELS)), (
        f"the digest map names {sorted(recorded)}, which is neither empty nor "
        f"every kernel this module ships")
    for name, digest in recorded.items():
        live = hashlib.sha256(
            no_pml_curl.kernel_source(name).encode("utf-8")).hexdigest()
        assert live == digest, (
            f"{name}'s device text has changed since the gate ran; the admission "
            f"record describes bytes that no longer ship")


def test_the_admission_record_names_the_conductivity_it_does_not_serve():
    """The largest refused slice must be named, not left to be discovered.

    Three of the eight real-storage no-absorber corpus rows are MEEP ``Absorber``
    runs, whose absorber IS a conductivity. A record that listed only what it
    covered would leave the reader to work out why the corpus delta is smaller
    than the row count.
    """
    text = " ".join(no_pml_curl.NO_PML_CURL_ADMISSION["what_it_does_not_license"])
    assert "conductivity" in text
    assert "Absorber" in text
    assert "complex" in text


def test_every_cited_stepping_line_is_read_back_out_of_stepping():
    """A citation that has drifted into fiction is what the validate-against-the-reference rule forbids."""
    stepping_source = STEPPING_PATH.read_text(encoding="utf-8").splitlines()
    module_source = MODULE_PATH.read_text(encoding="utf-8")
    for line_number, needle in ((539, "target -= curl"),
                                (537, "_apply_conductive_update(target, curl"),
                                (2440, "fields.stores_E or fields.scratch is None"),
                                (2450, "multiply(displacement, inverse, out=target)")):
        assert f"stepping.py:{line_number}" in module_source or str(line_number) in module_source
        assert needle in stepping_source[line_number - 1], (
            f"stepping.py:{line_number} no longer reads {needle!r}; the citation "
            f"in no_pml_curl.py has drifted")
