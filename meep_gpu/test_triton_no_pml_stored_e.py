"""Laptop contracts for ARM S — ``update_E`` with stored E and no absorber.

Closes residual group (F) and the ``update_E`` third of group (H): 3 slots on
``TestAbsorber.test_absorber``, ``absorber-1d.py`` and ``material-dispersion.py``.

Everything here runs on the NumPy merge-bar machine. Triton does not build on
arm64 macOS, so the kernel's BYTES are the owed device gate's
(``parity/meep_gpu/gate_triton_no_pml_stored_e.py``); the ARITHMETIC is entirely
carryable here, because the sub-step is a subtraction chain and a multiply.

THE THREE THINGS THIS FILE EXISTS TO PIN, in order of how easy each is to get
wrong:

1. **The poles are not pre-summed and their ORDER is load-bearing.**
   ``(D - P0) - P1`` and ``D - (P0 + P1)`` are different float32 numbers, and so
   are ``((D-P0)-P1)`` and ``((D-P1)-P0)``. Both are driven as mutations and both
   must be caught, on a TWO-POLE fixture — at one pole neither can bite, which is
   the split that tells a real multi-pole defect from a harness artifact.
2. **The body is a STORE, not the certified accumulation.** Binding ``kps=1``,
   ``kms=0`` to ``kernels.constitutive_step`` does not recover it, and the
   signed-zero half of that claim is measured here rather than quoted.
3. **The record it supersedes.** ``no_pml_constitutive.STORED_E_ARM`` declined
   this arm on a demand figure that went stale when ``update_P`` was wired to the
   same three rows. The record's corrected fields are asserted, and so is the
   presence of the superseded text — a costed decision that quietly changes value
   is how an arm outlives its cost.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module

MODULE_NAME = "meep_gpu.triton_kernels.no_pml_stored_e"
PACKAGE_DIR = pathlib.Path(coverage_module.__file__).parent
MODULE_PATH = PACKAGE_DIR / "no_pml_stored_e.py"

E_COMPONENTS = ("Ex", "Ey", "Ez")
D_COMPONENTS = ("Dx", "Dy", "Dz")


@pytest.fixture(scope="module")
def arm_s():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


@pytest.fixture(scope="module")
def family():
    return importlib.import_module("meep_gpu.triton_kernels.no_pml_constitutive")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build(*, counts=(2, 2, 2), conductive=False, cell=(0.8, 0.8, 0.8),
          dimensions=3, seed=23, storage="field", pml_thickness=0,
          complex_storage=False, kind="lorentzian", **grid_kwargs):
    """A Grid/Fields/PML triple with NO active layer and STORED E.

    ``counts`` is the per-component pole count, so a fixture can carry a DIFFERENT
    subset on each component — ``PolarizationState._driven`` drops any component
    whose sigma is identically zero (dispersion.py:640-642), which is what makes
    the pole set per component rather than per run.
    """
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                courant=0.35, xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    eps = numpy.full(grid.shape, 2.25, dtype=numpy.float32)
    # A NON-UNIFORM inverse epsilon: a uniform one makes `s * inv_eps` a scaling
    # that many wrong transcriptions reproduce.
    rng = numpy.random.default_rng(seed)
    inverse = (1.0 / eps * rng.uniform(0.7, 1.3, size=grid.shape)).astype(numpy.float32)
    fields.set_epsilon_volumes({c: eps for c in E_COMPONENTS},
                               {c: inverse for c in E_COMPONENTS})
    if conductive:
        volume = numpy.full(grid.shape, 0.2, dtype=numpy.float32)
        fields.set_d_conductivity(volume)
        fields.set_b_conductivity(volume)
    for index in range(max(counts) if counts else 0):
        sigmas = {name: (0.3 + 0.05 * index) if counts[axis] > index else 0.0
                  for axis, name in enumerate(E_COMPONENTS)}
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.0, gamma=0.1, kind=kind),
            sigmas, grid, numpy.float32))
    if storage == "pml":
        fields.enable_pml_storage()
    elif storage == "field":
        fields.enable_field_storage()
    for name in D_COMPONENTS + E_COMPONENTS:
        array = getattr(fields, name, None)
        if array is not None:
            # A PHYSICAL BAND, not zeros. Zero-init is a fixed point of this
            # sub-step and a no-op agreeing with a no-op is trivially identical.
            array[...] = rng.uniform(-0.4, 0.4, size=grid.shape).astype(array.dtype)
    for state in fields.polarizations:
        for component in state.driven():
            for slot in ("P", "P_prev"):
                getattr(state, slot)[component][...] = rng.uniform(
                    -0.2, 0.2, size=grid.shape).astype(numpy.float32)
    return fields, PML(grid=grid, thickness=pml_thickness)


def reasons(arm_s, fields, pml):
    return [r for r in arm_s.stored_e_constitutive_coverage(fields, pml).reasons
            if "not cupy" not in r]


def snapshot(fields, names=E_COMPONENTS):
    return {name: getattr(fields, name).copy() for name in names}


def differing(left, right):
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
    assert MODULE_NAME not in sys.modules


def test_the_predicate_answers_without_triton_and_the_kernel_accessor_explains(arm_s):
    fields, pml = build()
    assert isinstance(arm_s.stored_e_constitutive_coverage(fields, pml).covered, bool)
    if arm_s.stored_e_constitutive_step is None:
        with pytest.raises(ImportError, match="triton"):
            arm_s.stored_e_constitutive_kernel()


def test_the_module_file_is_readable_as_utf8_without_a_locale():
    assert MODULE_PATH.read_text(encoding="utf-8")


def test_the_shared_clauses_it_names_still_exist(arm_s):
    """A rename in coverage.py fails HERE rather than dropping a clause silently."""
    for name in arm_s.SHARED_CLAUSES:
        assert hasattr(coverage_module, name), name


def test_the_pole_slot_count_matches_the_kernel_body_and_its_sibling(arm_s):
    """MAX_POLES is the length of the unrolled chain; a mismatch truncates silently."""
    from meep_gpu.triton_kernels import dispersive_update_e

    assert arm_s.MAX_POLES == dispersive_update_e.MAX_POLES
    source = MODULE_PATH.read_text(encoding="utf-8")
    for prefix in ("NP0", "NP1", "NP2"):
        chain = [line for line in source.splitlines()
                 if line.strip().startswith(f"if {prefix} > ")]
        assert len(chain) == arm_s.MAX_POLES, (
            f"{prefix}'s chain has {len(chain)} links but MAX_POLES is "
            f"{arm_s.MAX_POLES}; a pole past the chain is silently dropped")


# ---------------------------------------------------------------------------
# 2. The transcription — a NumPy reference, pinned against stepping.update_E
# ---------------------------------------------------------------------------

def reference_update_E(fields, *, presum=False, reverse=False, drop=0,
                       operand_swap=False, accumulate=False):
    """The kernel body, in NumPy, from stepping.py:998-1022 + fields.py:1079-1105.

    Written here rather than imported so the test has something to MUTATE. Every
    keyword is a mutation lever and each one is driven below.

        s   = ((D_c - P_0[c]) - P_1[c]) - ...     LEFT TO RIGHT, polarizations order
        E_c = s * inv_eps_c                        stored, stepping.py:1022
    """
    out = {}
    for component, source_name in (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz")):
        contributors = [state for state in fields.polarizations
                        if state.drives(component)]
        if reverse:
            contributors = list(reversed(contributors))
        if drop:
            contributors = contributors[:-drop] if drop <= len(contributors) else []
        source = getattr(fields, source_name).astype(numpy.float32)
        if presum and contributors:                      # THE MUTATION
            total = contributors[0].P[component].astype(numpy.float32)
            for state in contributors[1:]:
                total = total + state.P[component]
            value = source - total
        else:
            value = source
            for state in contributors:                   # dispersion.py:693-695
                value = value - state.P[component]
        inverse = fields.inverse_epsilon_for(component)
        if accumulate:                                   # THE MUTATION: the
            # certified PML body with kps=1, kms=0 and a zeroed f_w — the shape
            # STORED_E_ARM says does NOT recover a store.
            product = (value * inverse).astype(numpy.float32)
            zero = numpy.zeros_like(product)
            value = (zero + numpy.float32(1.0) * product) - numpy.float32(0.0) * zero
        elif operand_swap:                               # THE NULL (must be uncaught)
            value = inverse * value
        else:
            value = value * inverse                      # stepping.py:1011-1013
        out[component] = value.astype(numpy.float32)
    return out


CASES = {
    "material_dispersion_1voxel_2poles": dict(counts=(2, 2, 2), cell=(0.1, 0.1, 0.1)),
    "absorber_1d_5poles": dict(counts=(5, 5, 5), cell=(0.0, 0.0, 4.0), dimensions=1,
                               conductive=True),
    "three_d_2poles": dict(counts=(2, 2, 2)),
    "three_d_anisotropic_pole_sets": dict(counts=(3, 1, 2)),
    "three_d_no_poles_degenerate": dict(counts=(0, 0, 0), storage="field"),
    "three_d_drude": dict(counts=(2, 2, 2), kind="drude"),
    "three_d_conductive": dict(counts=(2, 2, 2), conductive=True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_reference_body_is_byte_identical_to_update_E(case):
    """The whole arithmetic claim, uint32-compared, with a vacuity floor."""
    fields, pml = build(**CASES[case])
    before = snapshot(fields)
    expected = reference_update_E(fields)
    stepping.update_E(fields, pml)
    after = snapshot(fields)

    moved, total = differing(after, before)
    assert moved > 0, (
        f"{case}: update_E moved NO words, so an identity here is two frozen "
        f"states agreeing (zero-init is a fixed point of this sub-step)")
    bad, compared = differing(after, expected)
    assert bad == 0, (
        f"{case}: {bad} of {compared} words differ between the transcription and "
        f"stepping.update_E (moved {moved} of {total})")


MUTATIONS = {
    # name: (reference kwargs, fixture, must_be_caught)
    "pre_sum_the_poles": (dict(presum=True), dict(counts=(2, 2, 2)), True),
    "reverse_the_pole_order": (dict(reverse=True), dict(counts=(2, 2, 2)), True),
    "drop_the_last_pole": (dict(drop=1), dict(counts=(2, 2, 2)), True),
    "use_the_certified_accumulation": (dict(accumulate=True),
                                       dict(counts=(2, 2, 2)), True),
    # THE EXPECTED-UNCAUGHT CONTROL. float32 multiplication is bitwise
    # commutative; a gate leg reporting this "caught" is comparing something other
    # than bytes. It is in the battery so that claim keeps being measured.
    "swap_the_multiply_operands": (dict(operand_swap=True),
                                   dict(counts=(2, 2, 2)), False),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_the_mutation_battery_splits_the_way_the_arithmetic_says_it_must(mutation):
    kwargs, fixture, must_be_caught = MUTATIONS[mutation]
    fields, pml = build(**fixture)
    if mutation == "use_the_certified_accumulation":
        # The two bodies differ only when their source carries a signed zero:
        # ``store(-0.0)`` preserves it, while ``+0.0 + -0.0`` canonicalises it.
        # The ordinary physical-band fixture deliberately contains no fabricated
        # signed zeros, so it cannot discriminate this particular substitution.
        # Seed the exact cancellation on one component before both references run.
        fields.Dx.fill(numpy.float32(-0.0))
        for state in fields.polarizations:
            state.P["Ex"].fill(numpy.float32(0.0))
    mutated = reference_update_E(fields, **kwargs)
    stepping.update_E(fields, pml)
    after = snapshot(fields)
    bad, compared = differing(after, mutated)
    if must_be_caught:
        assert bad > 0, (
            f"{mutation}: 0 of {compared} words differ, so the reference body "
            f"agrees with update_E with this defect injected")
    else:
        assert bad == 0, (
            f"{mutation}: {bad} of {compared} words differ, but float32 multiply "
            f"is bitwise commutative — this control is measuring something else")


@pytest.mark.parametrize("mutation", ("pre_sum_the_poles", "reverse_the_pole_order"))
def test_the_multi_pole_mutations_cannot_bite_at_one_pole(mutation):
    """The split that tells a real multi-pole defect from a harness artifact."""
    kwargs = MUTATIONS[mutation][0]
    fields, pml = build(counts=(1, 1, 1))
    mutated = reference_update_E(fields, **kwargs)
    stepping.update_E(fields, pml)
    bad, _ = differing(snapshot(fields), mutated)
    assert bad == 0, (
        f"{mutation} changed the answer at ONE pole, where reordering and "
        f"pre-summing a single term are both identities — the battery above is "
        f"catching a harness artifact rather than the defect it names")


def test_the_store_is_not_the_certified_accumulation_to_the_word():
    """``0.0 + (-0.0)`` is ``+0.0``: the accumulation canonicalises a signed zero.

    ``STORED_E_ARM['differs_from_certified_body']`` states this and cites a
    four-word measurement. Re-measured here so the record is not a quotation.
    """
    values = numpy.array([-0.0, 0.0, 1.5, -2.5], dtype=numpy.float32)
    accumulated = (numpy.zeros_like(values) + numpy.float32(1.0) * values
                   - numpy.float32(0.0) * numpy.zeros_like(values))
    stored = values.copy()
    bad = int((accumulated.view(numpy.uint32) != stored.view(numpy.uint32)).sum())
    assert bad == 1, (
        f"{bad} of 4 words differ; the claim that kps=1/kms=0/f=0 does not recover "
        f"a plain store rests on exactly the negative zero")
    assert numpy.signbit(stored[0]) and not numpy.signbit(accumulated[0])


# ---------------------------------------------------------------------------
# 3. Disjointness — the numbered inverted clauses, each measured
# ---------------------------------------------------------------------------

def test_arm_s_and_the_null_arm_partition_the_stores_E_axis(arm_s, family):
    """Inverted clause 1: the same ``fields.stores_E``, opposite signs."""
    for counts, storage in (((2, 2, 2), "field"), ((0, 0, 0), "field"),
                            ((0, 0, 0), None)):
        fields, pml = build(counts=counts, storage=storage)
        mine = not reasons(arm_s, fields, pml)
        theirs = not [r for r in family.null_constitutive_coverage(
            fields, pml, "E").reasons if "not cupy" not in r]
        assert mine != theirs, (
            f"counts={counts} storage={storage}: stored_e={mine} null={theirs} — "
            f"the pair must partition stepping.py:983's own test")


def test_arm_s_and_every_active_layer_arm_partition_the_absorber_axis(arm_s):
    """Inverted clause 2: the same ``pml.is_active``, opposite signs.

    Driven against the four ``update_E`` arms a dispersive real-f32 run can reach.
    """
    from meep_gpu.triton_kernels.coverage import constitutive_coverage
    from meep_gpu.triton_kernels.dispersive_update_e import (
        dispersive_constitutive_coverage)

    incumbents = {
        "ordinary": lambda f, p: constitutive_coverage(f, p, "E"),
        "dispersive": dispersive_constitutive_coverage,
    }
    for storage, thickness in (("field", 0), ("pml", 2)):
        fields, pml = build(counts=(1, 1, 1), storage=storage,
                            pml_thickness=thickness)
        mine = not reasons(arm_s, fields, pml)
        assert mine is (thickness == 0), (
            f"Arm S admitted={mine} at pml thickness {thickness}")
        for name, fn in incumbents.items():
            theirs = not [r for r in fn(fields, pml).reasons if "not cupy" not in r]
            assert not (mine and theirs), (
                f"Arm S and {name} both admit update_E at thickness {thickness}; "
                f"two admitters leave the slot UNSELECTED, which is a silent "
                f"coverage loss rather than an error")


def test_an_inert_layer_object_lands_on_arm_s_and_not_on_the_pml_arms(arm_s):
    """PML(thickness=0) is INACTIVE; stepping._pml_is_active is the shared test."""
    from meep_gpu.triton_kernels.dispersive_update_e import (
        dispersive_constitutive_coverage)

    fields, pml = build(counts=(1, 1, 1), pml_thickness=0)
    assert pml is not None and not pml.is_active
    assert not reasons(arm_s, fields, pml)
    theirs = dispersive_constitutive_coverage(fields, pml).reasons
    assert any("no active PML" in r for r in theirs)


def test_the_off_diagonal_and_nonlinear_products_are_refused_by_name(arm_s):
    fields, pml = build(counts=(1, 1, 1))
    ones = numpy.ones(fields.grid.shape, dtype=numpy.float32)
    fields.set_nonlinear_volumes({"Ex": ones}, {"Ex": ones})
    text = " ".join(reasons(arm_s, fields, pml))
    assert "chi2/chi3" in text and "nonlinear_update_e" in text


def test_pml_storage_mode_behind_an_inert_layer_is_refused(arm_s):
    """The frozen-drive hazard, refused from this side as no_pml_ade refuses it
    from the other."""
    fields, pml = build(counts=(1, 1, 1), storage="pml", pml_thickness=0)
    assert any("PML storage mode" in r for r in reasons(arm_s, fields, pml))


# ---------------------------------------------------------------------------
# 4. The remaining clauses
# ---------------------------------------------------------------------------

def test_complex_storage_is_refused(arm_s):
    fields, pml = build(counts=(1, 1, 1), complex_storage=True)
    assert any("force_complex_fields" in r for r in reasons(arm_s, fields, pml))


def test_an_unstored_E_is_refused_and_names_the_null_arm(arm_s):
    fields, pml = build(counts=(0, 0, 0), storage=None)
    text = " ".join(reasons(arm_s, fields, pml))
    assert "recomputed from D rather than stored" in text
    assert "null_constitutive_coverage" in text


def test_too_many_poles_is_refused_by_name_rather_than_truncated(arm_s):
    fields, pml = build(counts=(9, 9, 9))
    assert any("MAX_POLES" in r for r in reasons(arm_s, fields, pml))


def test_an_uncovered_susceptibility_kind_is_refused(arm_s):
    fields, pml = build(counts=(1, 1, 1))
    state = fields.polarizations[0]
    # ``Susceptibility`` rejects unsupported kinds at construction.  The gate still
    # needs to reject a foreign/lifted descriptor carrying one, rather than relying
    # on every caller having taken the engine's constructor path.
    state.susceptibility = SimpleNamespace(kind="noise")
    assert any("noise" in r for r in reasons(arm_s, fields, pml))


def test_a_malformed_P_prev_is_refused_even_though_this_kernel_never_reads_it(arm_s):
    """update_P rotates the three buffers as a set (dispersion.py:687-691)."""
    fields, pml = build(counts=(1, 1, 1))
    state = fields.polarizations[0]
    state.P_prev["Ez"] = state.P_prev["Ez"].astype(numpy.float64)
    assert any("P_prev[Ez]" in r for r in reasons(arm_s, fields, pml))


def test_a_fold_is_refused_and_says_it_is_conservative(arm_s):
    from meep_gpu.grid import Mirror

    fields, pml = build(counts=(1, 1, 1), symmetry=(Mirror(axis="X", phase=1),))
    text = " ".join(reasons(arm_s, fields, pml))
    assert "mirror plane" in text and "CONSERVATIVE" in text


def test_the_degenerate_no_pole_run_is_admitted_and_stores_D_times_inv_eps(arm_s):
    """Nothing else covers it, and the array path aliases D straight through."""
    fields, pml = build(counts=(0, 0, 0), storage="field")
    assert fields.stores_E and not fields.polarizations
    assert not reasons(arm_s, fields, pml)
    expected = reference_update_E(fields)
    before = snapshot(fields)
    stepping.update_E(fields, pml)
    assert differing(snapshot(fields), before)[0] > 0
    assert differing(snapshot(fields), expected)[0] == 0


def test_the_builder_refuses_out_of_coverage_and_needs_no_triton(arm_s):
    fields, pml = build(counts=(1, 1, 1), storage="pml", pml_thickness=2)
    assert arm_s.plan_stored_e_constitutive(fields, pml) is None


def test_the_pole_order_helper_matches_fields_own_contributor_list(arm_s):
    """A plan that caches a different order is a different float32 number."""
    fields, _ = build(counts=(3, 1, 2))
    order = arm_s.poles_per_component(fields)
    for component in E_COMPONENTS:
        expected = [s for s in fields.polarizations if s.drives(component)]
        assert list(order[component]) == expected, component


# ---------------------------------------------------------------------------
# 5. The record this arm supersedes
# ---------------------------------------------------------------------------

def test_the_stored_e_arm_record_says_built_and_names_where(family):
    record = family.STORED_E_ARM
    assert record["built"] is True
    assert "no_pml_stored_e" in record["built_as"]
    for key in ("array_path", "displacement_minus_polarization",
                "differs_from_certified_body", "compose_from",
                "drive_field_consequence", "measured_demand",
                "superseded_demand", "device_bytes_owed"):
        assert record[key], key


def test_the_demand_figure_is_current_and_the_stale_one_is_kept_as_such(family):
    """A costed decision that quietly changes value outlives its cost."""
    record = family.STORED_E_ARM
    assert "ZERO additional whole-step rows" not in record["measured_demand"], (
        "measured_demand still carries the figure that went stale when update_P "
        "was wired to these three rows")
    assert "ZERO additional whole-step rows" in record["superseded_demand"], (
        "the superseded figure is not kept, so a reader cannot see what changed")
    for row in ("material-dispersion.py", "absorber-1d.py",
                "TestAbsorber.test_absorber"):
        assert row in record["measured_demand"], row


def test_the_refusal_helper_delegates_and_is_empty_where_arm_s_covers(family, arm_s):
    """Its contract changed with the build: it is no longer always non-empty."""
    fields, pml = build(counts=(2, 2, 2))
    assert not [r for r in family.stored_e_constitutive_reasons(fields, pml)
                if "not cupy" not in r], (
        "the helper still refuses a configuration Arm S covers, so it is reporting "
        "a product that does not exist")
    assert (tuple(family.stored_e_constitutive_reasons(fields, pml))
            == arm_s.stored_e_constitutive_coverage(fields, pml).reasons)

    active, layer = build(counts=(2, 2, 2), storage="pml", pml_thickness=2)
    assert family.stored_e_constitutive_reasons(active, layer), (
        "an active layer must still be refused; that slot is dispersive_update_e's")
