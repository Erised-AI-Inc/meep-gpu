"""Laptop contracts for the CONDUCTIVE no-absorber curl — residual groups (E) and (H).

Closes 6 slots on 3 rows: ``TestAbsorber.test_absorber_2d`` (both curls, group E),
``TestAbsorber.test_absorber`` and ``absorber-1d.py`` (both curls each, group H).

Everything here runs on the NumPy merge-bar machine. Triton does not build on
arm64 macOS, so the kernel's BYTES are the owed device gate's
(``parity/meep_gpu/gate_triton_no_pml_conductive.py``); what a laptop can carry —
and what this file carries — is the whole arithmetic, because the tail is three
in-place float32 passes and there is nothing about them a NumPy host cannot round
exactly the same way.

THE THREE LOAD-BEARING LEGS, and each one is written so it CAN fail:

1. **The transcription.** :func:`reference_curl_step` is the kernel body written
   in NumPy from the same lines, and it is pinned uint32-for-uint32 against the
   real ``stepping.step_B``/``step_D``. A mutation battery drives eight edits
   through it and every one must be caught, so a passing transcription test is
   evidence rather than a tautology.
2. **The divergence.** The incumbent plain tail must DIVERGE here, or this module
   is closing a gap that does not exist. Measured against the real array path with
   the one reader that selects the tail cleared, with two controls that must come
   back IDENTICAL.
3. **The vacuity floor.** Every identity and divergence case asserts the array
   path MOVED words. Zero-init is a fixed point of a curl on a uniform field and a
   frozen state agreeing with a frozen state is not a measurement.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module

MODULE_NAME = "meep_gpu.triton_kernels.no_pml_conductive"
PACKAGE_DIR = pathlib.Path(coverage_module.__file__).parent
MODULE_PATH = PACKAGE_DIR / "no_pml_conductive.py"

TARGETS = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz")}
STORED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")


@pytest.fixture(scope="module")
def npc():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


# ---------------------------------------------------------------------------
# Fixtures — the four corpus rows' own configurations, by name
# ---------------------------------------------------------------------------

def build(*, poles=0, conductive=True, cell=(0.8, 0.8, 0.8), dimensions=3,
          seed=17, sigma=0.2, sides=("D", "B"), storage=None, pml_thickness=0,
          complex_storage=False, **grid_kwargs):
    """A Grid/Fields/PML triple with NO active split-field layer.

    ``sides`` selects which of MEEP's two conductivity setters runs, so the
    per-target reading can be exercised on a run that is lossy on ONE side only —
    the configuration ``_apply_curl``'s per-term ``condfac_for`` (stepping.py:508)
    exists for and the one a per-run clause would get wrong.
    """
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                courant=0.35, xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    components = ("Ex", "Ey", "Ez")
    eps = numpy.full(grid.shape, 2.25, dtype=numpy.float32)
    fields.set_epsilon_volumes({c: eps for c in components},
                               {c: (1.0 / eps).astype(numpy.float32)
                                for c in components})
    if conductive:
        volume = numpy.full(grid.shape, sigma, dtype=numpy.float32)
        if "D" in sides:
            fields.set_d_conductivity(volume)
        if "B" in sides:
            fields.set_b_conductivity(volume)
    for index in range(poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian"),
            {name: 0.3 + 0.05 * index for name in components}, grid, numpy.float32))
    if storage == "pml":
        fields.enable_pml_storage()
    elif storage == "field" or (storage is None and fields.polarizations):
        fields.enable_field_storage()
    rng = numpy.random.default_rng(seed)
    for name in STORED:
        array = getattr(fields, name, None)
        if array is not None:
            # A PHYSICAL BAND, not zeros: zero-init is a fixed point of every one
            # of these updates and a no-op agreeing with a no-op proves nothing.
            array[...] = rng.uniform(-0.4, 0.4, size=grid.shape).astype(array.dtype)
    for state in fields.polarizations:
        for component in state.driven():
            for slot in ("P", "P_prev"):
                getattr(state, slot)[component][...] = rng.uniform(
                    -0.2, 0.2, size=grid.shape).astype(numpy.float32)
    return fields, PML(grid=grid, thickness=pml_thickness)


def reasons(npc, fields, pml, sub_step="step_B"):
    """The refusal list with the ONE clause a NumPy host can never satisfy removed."""
    return [r for r in npc.conductive_plain_curl_coverage(fields, pml, sub_step).reasons
            if "not cupy" not in r]


def snapshot(fields, names=STORED):
    return {name: getattr(fields, name).copy() for name in names
            if getattr(fields, name, None) is not None}


def differing(left, right):
    """Words differing, compared as uint32: -0.0 != +0.0 and NaN never matches."""
    total = compared = 0
    for name, array in left.items():
        a = array.view(numpy.uint32).ravel()
        b = right[name].view(numpy.uint32).ravel()
        total += int((a != b).sum())
        compared += int(a.size)
    return total, compared


# ---------------------------------------------------------------------------
# 1. The optional dependency stays optional
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules, (
        "importing the package pulled in the conductive no-PML module, which imports "
        "Triton when it is present; the host predicate route must stay free of it")


def test_the_predicate_answers_without_triton_and_the_kernel_accessor_explains(npc):
    fields, pml = build()
    assert isinstance(npc.conductive_plain_curl_coverage(fields, pml, "step_B").covered,
                      bool)
    if npc.conductive_plain_curl_step is None:  # the laptop path
        with pytest.raises(ImportError, match="triton"):
            npc.conductive_plain_curl_kernel()


def test_the_module_file_is_readable_as_utf8_without_a_locale():
    """A CUDA context resets LC_CTYPE to ASCII; a locale-dependent open then dies."""
    assert MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. The transcription — a NumPy reference, pinned against stepping itself
# ---------------------------------------------------------------------------

def reference_curl_step(fields, sub_step, *, cond=None, tail="conductive",
                        group="paired", mask=True, derive=None):
    """The kernel body, in NumPy, from the same lines the Triton body cites.

    Written here rather than imported so the test has something to MUTATE: a
    transcription test that can only compare the shipped code to itself measures
    nothing. Every keyword below is a mutation lever and each one is driven.

    Transcribed from:
      * term table        stepping.B_CURL_TERMS (:213) / D_CURL_TERMS (:218)
      * ghost rule        stepping._shift_up (:1723) / _shift_down (:1787)
      * curl grouping     stepping._curl_from_operands (:1601)
      * ownership mask    stepping._mask_non_owned_cells (:1865)
      * E derivation      stepping._read_component (:2383-2396)
      * lossless tail     stepping._apply_curl (:509-510)
      * lossy tail        stepping._apply_conductive_update (:1938-1951)
    """
    grid = fields.grid
    nx, ny, nz = grid.shape
    dtdx = numpy.float32(grid.dt / grid.dx)
    kinds = stepping._boundary_kinds(grid, None)
    backward = sub_step == "step_D"

    if sub_step == "step_B":
        if derive is None:
            derive = not fields.stores_E
        if derive:
            source = {("Ex", "Ey", "Ez")[i]:
                      (getattr(fields, ("Dx", "Dy", "Dz")[i])
                       * fields.inverse_epsilon_for(("Ex", "Ey", "Ez")[i])
                       ).astype(numpy.float32)
                      for i in range(3)}
        else:
            source = {n: getattr(fields, n) for n in ("Ex", "Ey", "Ez")}
        operands = [source["Ex"], source["Ey"], source["Ez"]]
    else:
        operands = [getattr(fields, n) for n in ("Bx", "By", "Bz")]

    def shifted(volume, axis):
        """One neighbour read, per the axis's own ghost rule."""
        out = numpy.empty_like(volume)
        step = -1 if backward else +1
        rolled = numpy.roll(volume, -step, axis=axis)
        out[...] = rolled
        if kinds[axis] == "metallic":
            index = [slice(None)] * 3
            index[axis] = 0 if backward else volume.shape[axis] - 1
            out[tuple(index)] = numpy.float32(0.0)
        return out

    a, b, c = operands
    a_y, a_z = shifted(a, 1), shifted(a, 2)
    b_x, b_z = shifted(b, 0), shifted(b, 2)
    c_x, c_y = shifted(c, 0), shifted(c, 1)

    if group == "paired":  # stepping._curl_from_operands, DO NOT FLATTEN
        curls = [dtdx * ((c_y - c) + (b - b_z)),
                 dtdx * ((a_z - a) + (c - c_x)),
                 dtdx * ((b_x - b) + (a - a_y))]
    elif group == "flat":  # THE MUTATION
        curls = [dtdx * (c_y - c + b - b_z),
                 dtdx * (a_z - a + c - c_x),
                 dtdx * (b_x - b + a - a_y)]
    else:
        raise ValueError(group)

    def zero_face(volume, axis):
        index = [slice(None)] * 3
        index[axis] = 0
        volume[tuple(index)] = numpy.float32(0.0)

    if mask:
        pairs = (((1, 2), (0, 2), (0, 1)) if backward else ((0,), (1,), (2,)))
        for component, axes in enumerate(pairs):
            for axis in axes:
                if kinds[axis] == "metallic":
                    zero_face(curls[component], axis)

    names = TARGETS[sub_step]
    if cond is None:
        cond = tuple(fields.condfac_for(n) is not None for n in names)
    out = {}
    for index, name in enumerate(names):
        field = getattr(fields, name).astype(numpy.float32)
        if not cond[index]:
            out[name] = (field - curls[index]).astype(numpy.float32)
            continue
        cf = fields.condfac_for(name)
        ci = fields.condinv_for(name)
        if tail == "conductive":       # stepping.py:1996, :1997, :1998 IN ORDER
            value = field * cf
            value = value - curls[index]
            value = value * ci
        elif tail == "swapped":        # THE MUTATION: condfac/condinv exchanged
            value = ((field * ci) - curls[index]) * cf
        elif tail == "plain":          # THE INCUMBENT's tail — the gap itself
            value = field - curls[index]
        elif tail == "distributed":    # THE MUTATION: condinv distributed over the
            # subtraction. Algebraically the same and a different float32 number,
            # because each product rounds before the subtraction sees it — the
            # exact class ENABLE_FP_FUSION=False exists to stop a compiler doing.
            value = ((field * cf) * ci) - (curls[index] * ci)
        else:
            raise ValueError(tail)
        out[name] = value.astype(numpy.float32)
    return out


CASES = {
    # name: (build kwargs, sub_step)
    "E_2d_cond_no_poles": (dict(poles=0, cell=(2.0, 2.0, 0.0), dimensions=2), None),
    "E_3d_cond_no_poles": (dict(poles=0), None),
    "H_1d_cond_5_poles": (dict(poles=5, cell=(0.0, 0.0, 4.0), dimensions=1), None),
    "H_3d_cond_2_poles": (dict(poles=2), None),
    "MIXED_3d_D_side_only": (dict(poles=0, sides=("D",)), None),
    "MIXED_3d_B_side_only": (dict(poles=0, sides=("B",)), None),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_reference_body_is_byte_identical_to_the_array_path(case, sub_step):
    """The whole arithmetic claim, uint32-compared, with a vacuity floor."""
    kwargs, _ = CASES[case]
    fields, pml = build(**kwargs)
    before = snapshot(fields)
    expected = reference_curl_step(fields, sub_step)
    getattr(stepping, sub_step)(fields, pml)
    after = snapshot(fields, TARGETS[sub_step])

    moved, _ = differing({n: after[n] for n in TARGETS[sub_step]},
                         {n: before[n] for n in TARGETS[sub_step]})
    assert moved > 0, (
        f"{case}/{sub_step}: the array path moved NO words, so an identity here "
        f"would be two frozen states agreeing")
    bad, compared = differing(after, expected)
    assert bad == 0, (
        f"{case}/{sub_step}: {bad} of {compared} words differ between the "
        f"transcription and stepping.{sub_step}")


#: mutation -> (reference kwargs, the fixture it can actually bite on).
#:
#: TWO OF THESE ARE SHAPE-DEPENDENT AND SAY SO, because a battery run only on the
#: shape being unblocked certifies a mutated body:
#:
#: * ``drop_ownership_mask`` edits cells the mask zeroes, and the mask only fires
#:   on a METALLIC axis. On the all-periodic fixture it was measured UNCAUGHT
#:   (0 of 1536 words), which is why it carries its own metallic fixture;
#: * ``flatten_grouping`` reassociates ``dtdx * ((c_y - c) + (b - b_z))``. On a
#:   one-cell axis ``(c_y - c)`` is an EXACT zero and the reassociation is exact —
#:   the same 0/18-on-(64,1,64) result ``no_pml.py`` records — so it runs on the
#:   3-D shape and never on the 1-D corpus shape.
MUTATIONS = {
    "flatten_grouping": (dict(group="flat"), dict(poles=2)),
    "drop_ownership_mask": (dict(mask=False),
                            dict(poles=2, cell=(0.8, 0.8, 0.8), dimensions=3,
                                 boundaries=("metallic", "metallic", "metallic"))),
    "swap_condfac_condinv": (dict(tail="swapped"), dict(poles=2)),
    "plain_tail_the_incumbent": (dict(tail="plain"), dict(poles=2)),
    "distribute_condinv_over_the_subtraction": (dict(tail="distributed"),
                                                dict(poles=2)),
    "tell_every_target_it_is_lossless": (dict(cond=(False, False, False)),
                                         dict(poles=2)),
    "tell_one_target_it_is_lossless": (dict(cond=(False, True, True)),
                                       dict(poles=2)),
    "force_derive_against_stores_E": (dict(derive=True), dict(poles=2)),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_every_mutation_of_the_reference_body_is_caught(mutation):
    """A transcription test that catches nothing pins nothing."""
    kwargs, fixture = MUTATIONS[mutation]
    sub_step = "step_B"
    fields, pml = build(**fixture)
    mutated = reference_curl_step(fields, sub_step, **kwargs)
    getattr(stepping, sub_step)(fields, pml)
    after = snapshot(fields, TARGETS[sub_step])
    bad, compared = differing(after, mutated)
    assert bad > 0, (
        f"{mutation}: 0 of {compared} words differ, so the reference body agrees "
        f"with the array path even with this defect injected — the identity test "
        f"above is not measuring what it claims")


# ---------------------------------------------------------------------------
# 3. The divergence — the gap this module closes, measured, with its controls
# ---------------------------------------------------------------------------

def measure_incumbent_divergence(fields, pml, sub_step):
    """What the LOSSLESS tail would have left, against what the array path leaves.

    ``_apply_curl`` reads ``fields.condfac_for(term.target)`` at stepping.py:508 and
    takes ``target -= curl`` at :539 when it answers None, so clearing that one
    reader turns the REAL array path into exactly ``no_pml.plain_curl_step``'s
    product — real curl, real ghost rule, real ownership mask, real E source.
    """
    names = TARGETS[sub_step]
    seed = snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    array_path = snapshot(fields, names)

    for name, value in seed.items():
        getattr(fields, name)[...] = value
    reader = fields.condfac_for
    fields.condfac_for = lambda component: None
    try:
        getattr(stepping, sub_step)(fields, pml)
    finally:
        fields.condfac_for = reader
    plain = snapshot(fields, names)

    bad, compared = differing(array_path, plain)
    moved, _ = differing(array_path, {n: seed[n] for n in names})
    return bad, compared, moved


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("case,kwargs", [
    ("E_group_no_poles", dict(poles=0)),
    ("H_group_2_poles", dict(poles=2)),
    ("H_group_5_poles_1d", dict(poles=5, cell=(0.0, 0.0, 4.0), dimensions=1)),
])
def test_the_incumbent_plain_tail_diverges_on_every_conductive_case(case, kwargs,
                                                                   sub_step):
    """GROUP (H)'s CURLS, SETTLED WITHOUT A DEVICE.

    The closure round measured group (E) DIVERGENT at both curls and ASSUMED group
    (H) diverges for the same reason; it never measured (H)'s curls directly. This
    is that measurement, and it needs no GPU because the tail is three float32
    passes. If it ever comes back IDENTICAL, ``no_pml_conductive`` is closing a gap
    that does not exist and must be deleted rather than wired.
    """
    fields, pml = build(**kwargs)
    bad, compared, moved = measure_incumbent_divergence(fields, pml, sub_step)
    assert moved > 0, f"{case}/{sub_step}: the array path moved nothing"
    assert bad > 0, (
        f"{case}/{sub_step}: the lossless tail agrees with the array path on "
        f"{compared} words, so there is no conductive gap here to close")


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_control_with_no_conductivity_anywhere_is_identical(sub_step):
    """The load-bearing half of the divergence leg.

    With no sigma the two tails are the same branch of the same function, so a
    DIVERGENT verdict here would mean the harness is measuring something other than
    the conductive tail — a stale source, a reordered pass, a copied-out snapshot.
    """
    fields, pml = build(poles=2, conductive=False)
    bad, compared, moved = measure_incumbent_divergence(fields, pml, sub_step)
    assert moved > 0, f"{sub_step}: the control moved nothing and measures nothing"
    assert bad == 0, (
        f"{sub_step}: {bad} of {compared} words differ with NO conductivity "
        f"installed, so the divergence leg is measuring something else")


def test_the_module_records_the_divergence_it_claims_to_close(npc):
    """The record must state a DIVERGENT verdict and a non-zero vacuity floor."""
    record = npc.LOCAL_DIVERGENCE
    cases = {k: v for k, v in record.items() if not k.startswith("_")}
    assert cases, "LOCAL_DIVERGENCE records nothing"
    for name, entry in cases.items():
        assert entry["moved"] > 0, f"{name} records a vacuous case"
        if name.startswith("CONTROL"):
            assert entry["verdict"] == "IDENTICAL" and entry["differing"] == 0, name
        else:
            assert entry["verdict"] == "DIVERGENT" and entry["differing"] > 0, name
    assert any(k.startswith("H_") for k in cases), (
        "no group (H) case is recorded, so the assumption the closure round left "
        "open is still an assumption")


# ---------------------------------------------------------------------------
# 4. Per-term conductivity — the error that cost the hand-CUDA predicate 7 slots
# ---------------------------------------------------------------------------

def test_a_D_sigma_cannot_reach_step_B_and_a_B_sigma_cannot_reach_step_D(npc):
    """stepping.py:508 is read inside a loop over ONE side's terms."""
    fields, _ = build(poles=0, sides=("D",))
    assert npc.conductive_no_pml_targets(fields, "step_D") == (True, True, True)
    assert npc.conductive_no_pml_targets(fields, "step_B") == (False, False, False)

    fields, _ = build(poles=0, sides=("B",))
    assert npc.conductive_no_pml_targets(fields, "step_B") == (True, True, True)
    assert npc.conductive_no_pml_targets(fields, "step_D") == (False, False, False)


def test_a_one_sided_conductivity_is_admitted_on_both_curls_not_refused(npc):
    """The OVER-REFUSAL this predicate exists not to make.

    A D-only sigma leaves ``step_B``'s three targets lossless. The incumbent
    refuses BOTH curls (its clause 8 is per RUN), so if this predicate asked the
    narrow per-target question in its ADMISSION, ``step_B`` would be refused by
    both and fall silently to the array path. It asks the incumbent's own question
    instead, and reads per target only for the COMPILE-TIME flags.
    """
    for sides in (("D",), ("B",)):
        fields, pml = build(poles=0, sides=sides)
        for sub_step in ("step_B", "step_D"):
            assert not reasons(npc, fields, pml, sub_step), (
                f"sides={sides} {sub_step} refused: {reasons(npc, fields, pml, sub_step)}")


def test_the_partially_lossless_plan_compiles_the_plain_tail_for_that_component(npc):
    """The COND flags must be the per-target answer, not the per-run one."""
    fields, _ = build(poles=0, sides=("D",))
    assert npc.conductive_no_pml_targets(fields, "step_B") == (False, False, False)
    # ...and the reference body agrees: with no B sigma, step_B's three targets take
    # the lossless line, which is exactly what the array path does.
    fields, pml = build(poles=0, sides=("D",))
    expected = reference_curl_step(fields, "step_B")
    plain_only = reference_curl_step(fields, "step_B", cond=(False, False, False))
    stepping.step_B(fields, pml)
    after = snapshot(fields, TARGETS["step_B"])
    assert differing(after, expected)[0] == 0
    assert differing(after, plain_only)[0] == 0


def test_a_plan_with_no_conductive_target_builds_the_owned_lossless_slot(npc):
    """A one-sided conductive run still needs a plan for its unaffected curl.

    The incumbent predicate refuses both curls per run, while this plan's flags
    follow stepping._apply_curl per target.  Consequently all-zero flags mean
    "compile the plain tail in the slot this family owns", not "refuse".
    """
    volume = numpy.zeros((2, 2, 2), numpy.float32)
    plan = npc.ConductivePlainCurlPlan(
        "step_B", volume.shape, 0.5, (0, 0, 0), (0, 0, 0), 64,
        [volume.copy() for _ in range(3)],
        [volume.copy() for _ in range(3)],
        [None, None, None], [None, None, None])
    assert plan.cond == (0, 0, 0)


# ---------------------------------------------------------------------------
# 5. Disjointness — the numbered inverted clauses, each measured
# ---------------------------------------------------------------------------

def test_the_two_no_pml_curl_predicates_are_a_partition(npc):
    """Clause 8 inverted: exactly one of the two admits any curl slot."""
    from meep_gpu.triton_kernels.no_pml import plain_curl_coverage

    configurations = [
        dict(poles=0, conductive=False),
        dict(poles=0, conductive=True),
        dict(poles=2, conductive=False),
        dict(poles=2, conductive=True),
        dict(poles=0, conductive=True, sides=("D",)),
        dict(poles=0, conductive=True, sides=("B",)),
        dict(poles=5, conductive=True, cell=(0.0, 0.0, 4.0), dimensions=1),
    ]
    for kwargs in configurations:
        fields, pml = build(**kwargs)
        for sub_step in ("step_B", "step_D"):
            mine = not reasons(npc, fields, pml, sub_step)
            theirs = not [r for r in plain_curl_coverage(fields, pml, sub_step).reasons
                          if "not cupy" not in r]
            assert mine != theirs, (
                f"{kwargs} {sub_step}: conductive={mine} plain={theirs} — the two "
                f"predicates must partition the conductive axis, not overlap or "
                f"leave a hole")


def test_an_active_layer_is_refused_and_names_the_other_kernel(npc):
    """Clause 3, NOT inverted: the split-field conductive curl owns that slot."""
    from meep_gpu.triton_kernels.conductivity import conductive_pml_curl_coverage

    fields, pml = build(poles=0, storage="pml", pml_thickness=2)
    text = " ".join(reasons(npc, fields, pml, "step_B"))
    assert "active PML layer is installed" in text
    assert "split-field" in text
    # ...and the incumbent admits it, modulo the backend, so the slot is not lost.
    theirs = [r for r in conductive_pml_curl_coverage(fields, pml, "step_B").reasons
              if "not cupy" not in r]
    assert not theirs, f"the conductive PML predicate refused its own slot: {theirs}"


def test_an_inert_layer_object_lands_here_and_not_on_the_pml_kernel(npc):
    """PML(thickness=0) is INACTIVE; stepping._pml_is_active is the shared test."""
    from meep_gpu.triton_kernels.conductivity import conductive_pml_curl_coverage

    fields, pml = build(poles=0, pml_thickness=0)
    assert pml is not None and not pml.is_active
    assert not reasons(npc, fields, pml, "step_B")
    theirs = [r for r in conductive_pml_curl_coverage(fields, pml, "step_B").reasons
              if "not cupy" not in r]
    assert any("no active PML" in r for r in theirs)


def test_f_cond_is_not_required_and_that_is_the_whole_point(npc):
    """The incumbent's refusal of these rows is TRUE; requiring it here would
    reproduce it."""
    fields, pml = build(poles=2)
    for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert getattr(fields, "f_cond_" + target) is None, (
            "no no-PML run allocates f_cond (fields.py:732-733); the fixture is wrong")
    assert not reasons(npc, fields, pml, "step_B")
    assert not any("f_cond" in r for r in reasons(npc, fields, pml, "step_B"))


# ---------------------------------------------------------------------------
# 6. The remaining clauses, each refused BY NAME
# ---------------------------------------------------------------------------

def test_pml_storage_behind_an_inert_layer_is_refused(npc):
    fields, pml = build(poles=0, storage="pml", pml_thickness=0)
    assert any("PML storage enabled while the layer is inert" in r
               for r in reasons(npc, fields, pml, "step_B"))


def test_the_inert_storage_reason_does_not_fire_where_the_layer_is_active(npc):
    """A right verdict with a false reason is still a defect in a reasons list."""
    fields, pml = build(poles=0, storage="pml", pml_thickness=2)
    assert not any("while the layer is inert" in r
                   for r in reasons(npc, fields, pml, "step_B"))


def test_complex_storage_is_refused(npc):
    fields, pml = build(poles=0, complex_storage=True)
    assert any("force_complex_fields" in r for r in reasons(npc, fields, pml, "step_B"))


def test_a_mirror_plane_is_refused(npc):
    from meep_gpu.grid import Mirror

    fields, pml = build(poles=0, symmetry=(Mirror(axis="X", phase=1),))
    text = " ".join(reasons(npc, fields, pml, "step_B"))
    assert "mirror plane" in text or "folded by a mirror plane" in text


def test_a_nonzero_k_point_is_refused(npc):
    fields, pml = build(poles=0, k_point=(0.25, 0.0, 0.0))
    assert any("k_point" in r for r in reasons(npc, fields, pml, "step_B"))


def test_chi2_chi3_is_refused(npc):
    fields, pml = build(poles=0)
    ones = numpy.ones(fields.grid.shape, dtype=numpy.float32)
    fields.set_nonlinear_volumes({"Ex": ones}, {"Ex": ones})
    assert any("chi2/chi3" in r for r in reasons(npc, fields, pml, "step_B"))


def test_an_unknown_sub_step_raises_rather_than_refusing(npc):
    fields, pml = build(poles=0)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        npc.conductive_plain_curl_coverage(fields, pml, "update_E")
    with pytest.raises(ValueError, match="sub_step must be one of"):
        npc.conductive_no_pml_targets(fields, "update_E")


def test_a_half_installed_conductivity_is_refused_by_name(npc):
    """condfac without condinv is not a covered configuration."""
    fields, pml = build(poles=0)
    real = fields.condinv_for
    fields.condinv_for = lambda component: (None if component == "Bx"
                                            else real(component))
    assert any("both or neither" in r for r in reasons(npc, fields, pml, "step_B"))


def test_a_non_contiguous_condfac_is_refused(npc):
    fields, pml = build(poles=0)
    volume = fields.condfac_for("Bx")
    reversed_view = volume[::-1]
    real = fields.condfac_for
    fields.condfac_for = lambda component: (reversed_view if component == "Bx"
                                            else real(component))
    assert any("condfac[Bx] is not C-contiguous" in r
               for r in reasons(npc, fields, pml, "step_B"))


def test_derive_tracks_stores_E_rather_than_being_chosen(npc):
    """With poles the run stores E and the curl differences the STORED array."""
    fields, _ = build(poles=2)
    assert fields.stores_E
    fields_no_poles, _ = build(poles=0)
    assert not fields_no_poles.stores_E
    # The group (H) rows take DERIVE = 0 and the group (E) row takes DERIVE = 1;
    # the reference body's default agrees with stepping._read_component either way,
    # which the identity leg above measures on both.


# ---------------------------------------------------------------------------
# 7. The transcription is not allowed to drift from the lossless kernel's
# ---------------------------------------------------------------------------

def test_the_curl_half_is_byte_identical_to_the_lossless_no_pml_kernels(npc):
    """Two copies of one transcription must stay two copies of ONE transcription.

    The merge that would delete the duplication is a ``COND`` constexpr on
    ``no_pml.plain_curl_step``, which is that module's own gate event. Until then
    the only thing standing between the two bodies is this diff.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    other = (PACKAGE_DIR / "no_pml.py").read_text(encoding="utf-8")

    def segment(text, start, end):
        head = text.index(start)
        return text[head:text.index(end, head)]

    mine = segment(source, "# --- the ghost rule, per axis",
                   "# --- the tail, PER COMPONENT")
    theirs = segment(other, "# --- the ghost rule, per axis",
                     "# --- the plain update")

    def code_lines(text):
        """Executable lines, dedented to a common margin and stripped of comments.

        This module's copy sits inside ``if triton is not None:`` and carries four
        more spaces than ``no_pml``'s, and each file states the metallic-ghost note
        in a different place — neither is a divergence in what the body COMPUTES,
        which is what this test is about.
        """
        lines = [line for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        margin = min(len(line) - len(line.lstrip()) for line in lines)
        return [line[margin:].rstrip() for line in lines]

    assert code_lines(mine) == code_lines(theirs), (
        "the conductive curl's stencil half has drifted from no_pml.plain_curl_step's; "
        "conductivity enters _apply_curl only AFTER the curl is formed "
        "(stepping.py:508, :536), so the two must stay identical above the tail")
