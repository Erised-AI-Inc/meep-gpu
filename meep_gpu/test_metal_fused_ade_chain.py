"""The fused E->P chain, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_fused_ade_chain.py`` steps two engines side
by side for sixty complete seam steps and compares uint32 words — and no assertion
here duplicates that. What lives here is everything true about the family WITHOUT
a device: what the emitted source says, which configurations the predicate refuses
and by what name, that the three arms are registered UNWIRED, that the signature
respects the platform ceiling by ARITHMETIC, and — the claim this family exists
for — that the rotation the product leaves on the host can never put a launch's
output on its own input.

THE TRANSCRIPTION TESTS PARSE, THEY DO NOT MIRROR. Each one CALLS the certified
emitter and searches its output; none re-implements a line. A test that re-derived
the arithmetic would mirror a defect instead of executing it, which is a failure
mode this project has measured (three planted assembly defects, 86 of 87 tests
still green).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_fused_ade_chain.py"
PROBE = REPO / "parity" / "meep_gpu" / "probe_metal_ade_rotation_seam.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, arms, dispersive_update_e, fused_ade_chain as family,
    no_pml_stored_e, registry,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The source is a transcription of two certified bodies, and it PARSES as one
# ---------------------------------------------------------------------------

def test_the_ade_recurrence_is_the_certified_bodys_own_line():
    """The one line that carries the polarization arithmetic, character for character.

    ``ade_update_p.ade_source`` is CALLED; the line is searched for in its output
    and in this family's, with the single documented pointer rename applied. If
    either emitter's spelling moves this fails naming the line rather than quietly
    certifying a re-derivation.
    """
    certified = ade_update_p.ade_source("float32", True)
    line = "p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));"
    assert line in certified, "the certified ADE body's arithmetic line moved"
    fused = family.fused_ade_chain_source("no_pml", 0, 2, (True, True))
    for index in (0, 1):
        renamed = line.replace("p_out", f"o{index}")
        assert renamed in fused, (
            f"pole {index}'s recurrence is not the certified line with the "
            f"documented rename; it is a re-derivation")


def test_the_pole_chain_is_the_certified_stored_e_bodys_own_lines():
    certified = no_pml_stored_e.stored_e_source(3)
    fused = family.fused_ade_chain_source("no_pml", 0, 3, (False, False, False))
    for index in range(3):
        line = f"    source = source - p{index}[idx];"
        assert line in certified and line in fused


@pytest.mark.parametrize("axis,coordinate", ((0, "i"), (1, "j"), (2, "k")))
def test_the_pml_half_is_the_certified_dispersive_bodys_own_lines(axis, coordinate):
    """The dispersive arm's seam is a PURE ADDITION — nothing is rewritten.

    ``dispersive_update_e``'s body already names the drive ``src`` one line before
    it stores it into ``f_w`` (dispersive_update_e.py:79-80), so the fused body
    keeps every line and only ADDS ``float w = src;`` inside the ADE arms.
    """
    certified = dispersive_update_e.dispersive_e_source(2, axis)
    fused = family.fused_ade_chain_source("dispersive", axis, 2, (True, True))
    for line in ("    float prev = fw[idx];",
                 "    float source = d_in[idx];",
                 "    float src = source * inv_e[idx];",
                 "    fw[idx] = src;",
                 "    float value = e_out[idx];",
                 f"    value = value + kps[{coordinate}] * src;",
                 f"    value = value - kms[{coordinate}] * prev;",
                 "    e_out[idx] = value;"):
        assert line in certified, f"the certified dispersive body changed: {line!r}"
        assert line in fused, f"the fused body dropped a certified line: {line!r}"
    assert "float w = src;" in fused, "the seam register is not read by the ADE arm"


def test_the_no_pml_seam_is_the_one_rewritten_line_and_it_is_declared():
    """``e_out[idx] = source * inv_e[idx];`` becomes two lines, and only that.

    A float32 value stored to a ``device float*`` and reloaded is bit-identical to
    the register, so naming the product is byte-neutral BY CONSTRUCTION. This test
    pins that the rewrite is the ONE difference; the gate measures the bytes.
    """
    certified = no_pml_stored_e.stored_e_source(1)
    assert "    e_out[idx] = source * inv_e[idx];" in certified
    fused = family.fused_ade_chain_source("no_pml", 0, 1, (True,))
    assert "    float e = source * inv_e[idx];" in fused
    assert "    e_out[idx] = e;" in fused
    assert "    e_out[idx] = source * inv_e[idx];" not in fused
    assert "float w = e;" in fused


def test_the_certified_ade_drive_line_is_the_only_thing_the_seam_replaces():
    """``float w = drive[idx];`` is what the fusion removes, and it is gone."""
    certified = ade_update_p.ade_source("float32", True)
    assert "    float w = drive[idx];" in certified
    for arm, register in (("no_pml", "e"), ("dispersive", "src")):
        fused = family.fused_ade_chain_source(arm, 0, 1, (True,))
        assert "float w = drive[idx];" not in fused
        assert f"float w = {register};" in fused


# ---------------------------------------------------------------------------
# 2. The kernel is STRAIGHT LINE, which is what makes a mutation undeadable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ("no_pml", "dispersive"))
@pytest.mark.parametrize("poles", (0, 1, 6))
def test_the_body_has_exactly_one_branch(arm, poles):
    """Every emitted line but the guard is reached by every in-range thread.

    THE DEAD-BRANCH MUTATION — an edit to a line the scored grid never enters,
    reported UNCAUGHT while measuring nothing — cannot exist in a kernel with one
    branch. That property is asserted from the emitted TEXT rather than argued
    from the emitter's layout.
    """
    flags = tuple((index % 2) == 0 for index in range(poles))
    source = family.fused_ade_chain_source(arm, 0, poles, flags)
    assert len(re.findall(r"\bif\s*\(", source)) == 1
    assert "?" not in source
    assert not re.findall(r"\b(for|while)\s*\(", source)


# ---------------------------------------------------------------------------
# 3. The binding ceiling decides the product's SHAPE, by arithmetic
# ---------------------------------------------------------------------------

def test_the_binding_count_agrees_with_the_emitted_signature():
    """``binding_count`` is the predicate's arithmetic; the emitter must match it."""
    for arm in ("no_pml", "dispersive"):
        for poles in range(0, 7):
            for volumes in (0, poles):
                bindings = family.binding_count(arm, poles, volumes)
                if bindings > MAX_BUFFER_BINDINGS:
                    continue
                flags = tuple(index < volumes for index in range(poles))
                source = family.fused_ade_chain_source(arm, 0, poles, flags)
                highest = max(int(chunk.split(")")[0])
                              for chunk in source.split("[[buffer(")[1:])
                assert highest + 1 == bindings, (arm, poles, volumes, highest)


def test_the_corpus_worst_row_fits_and_one_pole_more_does_not():
    """Six poles with six sigma volumes is 31 against a ceiling of 31.

    The corpus's worst E->P row is ``stochastic_emitter*.py`` — SIX poles on every
    component — and that is the number the packed ``Params&`` exists for. Spelled
    here as arithmetic; the gate COMPILES both signatures and requires the second
    to fail.
    """
    assert family.binding_count("dispersive", 6, 6) == MAX_BUFFER_BINDINGS == 31
    assert family.binding_count("dispersive", 7, 7) > MAX_BUFFER_BINDINGS
    assert family.binding_count("no_pml", 5, 5) <= MAX_BUFFER_BINDINGS


def test_an_over_ceiling_signature_is_refused_by_name_not_left_to_the_compiler():
    with pytest.raises(ValueError) as excinfo:
        family.fused_ade_chain_source("dispersive", 0, 7, (True,) * 7)
    message = str(excinfo.value)
    assert "bindings" in message and "device.py:72" in message


def test_the_params_blob_is_a_flat_word_sequence_the_host_can_pack():
    """Every struct member is 4 bytes, so the layout has no padding to guess at."""
    blob = family.pack_params([(1.5, 2.5, 3.5), (4.5, 5.5, 6.5)],
                              [0.25, 0.75], 7, 9, 630)
    assert blob.dtype == np.float32
    assert blob.size == family.params_words(2) == 4 * 2 + 3
    assert blob[0] == np.float32(1.5) and blob[1] == np.float32(4.5)
    assert blob[2] == np.float32(2.5) and blob[3] == np.float32(5.5)
    assert blob[4] == np.float32(3.5) and blob[5] == np.float32(6.5)
    assert blob[6] == np.float32(0.25) and blob[7] == np.float32(0.75)
    tail = np.frombuffer(blob[8:11].tobytes(), dtype=np.uint32)
    assert tuple(int(value) for value in tail) == (7, 9, 630)


def test_the_signature_declares_exactly_the_poles_it_binds_not_eight():
    """The certified bodies pad to eight; this one cannot afford to.

    Eight declared pole slots is eight pointers the fused signature does not have,
    which is the whole reason the pad is dropped. A regression to the pad would be
    invisible except as a ceiling failure at six poles, so it is pinned here.
    """
    source = family.fused_ade_chain_source("dispersive", 0, 2, (True, True))
    assert "p2" not in source and "o2" not in source and "q2" not in source
    assert "device const float* p1" in source


# ---------------------------------------------------------------------------
# 4. THE ROTATION. The claim this family exists for, checked by enumeration.
# ---------------------------------------------------------------------------

def _rotate(configuration, driven):
    """``dispersion.PolarizationState.update``'s three assignments (:689-691),
    over buffer NAMES — which is the only thing the aliasing question depends on."""
    state = {"P": dict(configuration["P"]), "P_prev": dict(configuration["P_prev"]),
             "scratch": configuration["scratch"]}
    launches = []
    for component in driven:
        p, p_prev, scratch = (state["P"][component], state["P_prev"][component],
                              state["scratch"])
        launches.append({"component": component, "out": scratch,
                         "p_now": p, "p_prev": p_prev})
        state["P"][component] = scratch
        state["P_prev"][component] = p
        state["scratch"] = p_prev
    return state, launches


@pytest.mark.parametrize("driven_count", (1, 2, 3))
def test_a_per_component_launch_never_writes_a_buffer_it_reads(driven_count):
    """Walked to CLOSURE, not sampled.

    The rotation is a permutation of ``2d + 1`` buffers, so its orbit is finite and
    this enumerates it until it repeats. At EVERY configuration in the orbit a
    per-component launch's write set is disjoint from its read set — ``out`` is
    either the scratch or an EARLIER component's ``P_prev``, and neither is read by
    this component. That is the licence for leaving the rotation on the host, and
    it is what the plan's own launch-time check enforces.
    """
    driven = ("Ex", "Ey", "Ez")[:driven_count]
    configuration = {"P": {c: f"A[{c}]" for c in driven},
                     "P_prev": {c: f"B[{c}]" for c in driven}, "scratch": "S"}
    seen = set()
    positions = 0
    while True:
        key = (tuple(configuration["P"][c] for c in driven),
               tuple(configuration["P_prev"][c] for c in driven),
               configuration["scratch"])
        if key in seen:
            break
        seen.add(key)
        positions += 1
        _, launches = _rotate(configuration, driven)
        for row in launches:
            assert row["out"] not in (row["p_now"], row["p_prev"]), (
                positions, row)
        configuration, _ = _rotate(configuration, driven)
    assert positions >= driven_count + 1, positions


def test_the_all_component_shape_DOES_alias_and_that_is_why_this_one_is_per_component():
    """The contrast that makes the finding above load-bearing.

    ``triton_kernels.fused_ade_state`` puts all of one susceptibility's components
    in ONE launch, and its own note records that arm 1's output IS arm 0's
    ``p_prev``. Reproduced here so this family's shape is a CHOICE with a measured
    reason rather than an accident.
    """
    driven = ("Ex", "Ey", "Ez")
    configuration = {"P": {c: f"A[{c}]" for c in driven},
                     "P_prev": {c: f"B[{c}]" for c in driven}, "scratch": "S"}
    _, launches = _rotate(configuration, driven)
    reads = {row["p_now"] for row in launches} | {row["p_prev"] for row in launches}
    writes = {row["out"] for row in launches}
    assert reads & writes == {"B[Ex]", "B[Ey]"}


# ---------------------------------------------------------------------------
# 5. The predicate: what it refuses, and by what name
# ---------------------------------------------------------------------------

class _Fields:
    """A degenerate engine object. The predicate must REFUSE it, never raise."""

    grid = None
    polarizations = ()
    has_offdiagonal_epsilon = False
    has_nonlinearity = False


def _reasons(fields, pml, arm="no_pml", folded=False):
    return family.fused_ade_chain_coverage(fields, pml, arm, folded).reasons


def test_the_predicate_never_raises_on_a_degenerate_object():
    for arm, folded in (("no_pml", False), ("dispersive", False),
                        ("dispersive", True)):
        verdict = family.fused_ade_chain_coverage(object(), None, arm, folded)
        assert verdict.covered is False and verdict.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A refusal from either half must arrive NAMED, with the half that said it."""
    reasons = _reasons(_Fields(), None)
    assert any(reason.startswith("E half: ") for reason in reasons)
    assert any(reason.startswith("ADE half: ") for reason in reasons)


def test_an_unregistered_arm_is_refused_by_name():
    verdict = family.fused_ade_chain_coverage(_Fields(), None, "no_such_arm")
    assert verdict.covered is False
    assert any("no E-half predicate is registered" in reason
               for reason in verdict.reasons)


@pytest.mark.parametrize("flag", ("has_offdiagonal_epsilon", "has_nonlinearity"))
def test_the_interleave_clause_refuses_anything_that_couples_components(flag):
    """The seam clause is about the INTERLEAVE, not about the arm's arithmetic.

    Both features are already refused by the E half, and both are named again
    because the fact asserted here is different: the per-component interleave
    equals the driver's order only while ``update_E(c)`` reads component ``c``
    alone.
    """
    fields = _Fields()
    setattr(fields, flag, True)
    reasons = _reasons(fields, None)
    assert any("per-component interleave is not the driver's order" in reason
               for reason in reasons)


# ---------------------------------------------------------------------------
# 6. The table: registered, enumerable, and INVISIBLE to plan_step
# ---------------------------------------------------------------------------

def test_three_arms_are_registered_unwired_under_three_family_names():
    """Three cells, three FAMILIES, one module — and the split is not cosmetic.

    ``arms`` holds one row per (family, label) and the table's invariant is that no
    slot carries two arms from one family: two rows from one family on one slot
    would leave that slot permanently UNSELECTED. Three products sharing one
    builder under separate family names is exactly what
    ``folded_dispersive_update_e`` already is to ``dispersive_update_e``.
    """
    families = {name for name, _arm, _folded in family.REGISTERED_ARMS.values()}
    assert families == {family.FAMILY, family.DISPERSIVE_FAMILY,
                        family.FOLDED_FAMILY}
    assert len(families) == len(family.REGISTERED_ARMS) == 3
    rows = [spec for spec in arms.registered() if spec.family in families]
    assert len(rows) == 3, rows
    for spec in rows:
        assert spec.slot == family.SLOT == "update_E"
        assert spec.wired is False, (
            "this product spans update_E and update_P; wiring it would hand "
            "plan_step a slot assignment no composition rule has measured")
    assert {spec.label for spec in rows} == set(family.REGISTERED_ARMS)
    # The invariant this split exists for, asserted here as well as in the
    # composition suite: one family, one arm, per slot.
    per_slot = [spec.family for spec in arms.registered(family.SLOT)]
    assert len(per_slot) == len(set(per_slot))


def test_plan_step_cannot_select_any_of_them():
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    registered = [spec.label for spec in arms.registered(family.SLOT)]
    for label in family.REGISTERED_ARMS:
        assert label not in labels
        assert label in registered


def test_the_family_is_in_family_modules():
    assert family.FAMILY in registry.FAMILY_MODULES


def test_the_plan_declares_every_pass_it_performs():
    """``replaces_sub_steps`` is READ by the composer and by the whole-step walk."""
    assert family.REPLACES == ("update_E", "update_P")
    assert family.MetalFusedAdeChainPlan.replaces_sub_steps == family.REPLACES
    assert family.MetalFusedAdeChainPlan.launches_per_run == 3
    assert family.MetalFusedAdeChainPlan.performs_device_work is True


def test_the_seam_carries_no_source_clause_and_the_absence_is_documented():
    """Every other fused product on this backend needs a source inventory.

    This one does not, because the driver injects nothing between ``update_E`` and
    ``update_P``. That is a deliberate DIFFERENCE from the sibling products, so it
    is pinned: a future edit that added a source parameter here would be reversing
    a measured fact, not tightening a predicate.
    """
    import inspect

    signature = inspect.signature(family.fused_ade_chain_coverage)
    assert "sources" not in signature.parameters
    # Whitespace-normalised: the reason is a wrapped sentence and a raw substring
    # test would be pinning the line wrap rather than the claim.
    prose = " ".join(family.fused_ade_chain_coverage.__doc__.split())
    assert "THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE" in prose
    assert "This seam has no injection in it at all" in prose


# ---------------------------------------------------------------------------
# 7. The gate and the probe exist, and route through the runner
# ---------------------------------------------------------------------------

def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records "
        "the sources the PROCESS imported")
    assert "gate_provenance" in text


def test_the_rotation_probe_is_present_and_stamps_its_provenance():
    assert PROBE.is_file()
    text = PROBE.read_text(encoding="utf-8")
    assert "gate_provenance" in text
    assert "hashlib.sha256" in text, (
        "a per-case seed must come from a digest, not hash(): PYTHONHASHSEED "
        "salts hash() of a string and a failing case could not be replayed")
