"""The folded COMPLEX off-diagonal ``update_E`` arm is SHIPPED, not missing.

WHY THIS FILE EXISTS.  The 2026-09-02 Triton fusion board reports
``constitutive_admitters: []`` on three D->E seam-instances that carry the
``folded complex off-diagonal PML`` curl arm::

    examples:solve-cw.py
    tests:TestArrayMetadata.test_array_metadata
    tests:TestHoleyWvgBands.test_fields_at_kx

Read as a product statement that is false, and a build round was opened on it.
The arm exists (``complex_offdiag_update_e``'s folded/active-PML tail), is
registered at ``update_E`` in ``launch.py``'s arm table, and its device byte gate
is RELEASED (``triton_kernels/fingerprints.json`` ::
``triton_complex_offdiag_device_gate``, ``status: PASS``).

WHAT IS ACTUALLY TRUE is narrower and mechanical: the census that board reads was
cut under an expansion probe artifact classifying the BASE FOUR patterns, and
this tranche binds :data:`folded_complex.PARITY_PROBE_PATTERNS` — the base four
PLUS ``'c8_mul_c8_parity_coefficient_left'``.  The arm therefore refuses with two
clauses that are about the ARTIFACT and never about the configuration.

These tests pin the BEHAVIOUR, so the reading cannot be made again from a board
column: the arm admits every one of the three configurations modulo the CuPy
clause under an artifact that classifies the parity pattern; it is the SOLE
``update_E`` admitter there, so it would be selected rather than lost to
ambiguity; five controls that must refuse do refuse; and the parity arithmetic
the fifth pattern is about is measured, with a bare sign flip caught.

The two probe artifacts are read off the results tree.  Every test that needs one
SKIPS when it is absent rather than passing vacuously.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

API = Path(__file__).resolve().parents[1]

#: The artifact the 2026-09-02 census was cut under — base four patterns only.
CENSUS_PROBE = (API / "parity/meep_gpu/results"
                / "complex_expansion_beta_extension_2026-09-01/probe_keep/probe.json")

#: The artifact the FOLDED COMPLEX families are gated under — base four plus the
#: parity pattern.
FOLDED_PROBE = (API / "parity/meep_gpu/results"
                / "triton_regate_2026-08-31_final3/folded_complex/probe.json")

#: All SEVEN patterns.  This is the record the 2026-09-02 board already names in
#: ``credited_under_a_different_expansion_licence`` on the B->H twin of the three
#: rows above, so no NEW artifact has to be cut for the E-side arm to be credited
#: — the one the H-side arm is already credited under licenses it too.
UNIFIED_PROBE = (API / "parity/meep_gpu/results"
                 / "unified_expansion_2026-08-27/keep/gate.json")

#: The census declared this on every leg (``--run-policy keep``); the expansion
#: clauses are conditional on it, so every comparison here is made under it.
RUN_POLICY = "keep"

BACKEND_HEAD = "array module is "
BACKEND_TAIL = ", not cupy"

#: One fixture per corpus row's own configuration.  These kwargs are the ones
#: ``parity/meep_gpu/gate_triton_complex_offdiag.py`` builds its three FOLDED
#: product rows from — the independent check that they are the corpus's shapes
#: and not shapes chosen to pass.
FIXTURES: Dict[str, Dict[str, Any]] = {
    # examples:solve-cw.py and tests:TestArrayMetadata.test_array_metadata:
    # boundary_kinds ['mirror','mirror','periodic'], metallic [T,T,F], k = 0.
    "folded_twofold": dict(
        cell=(2.4, 2.4, 0.0), dimensions=2,
        boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
        symmetry=("X", "Y"), complex_storage=True, offdiag=True,
        pml_thickness=2),
    # tests:TestHoleyWvgBands.test_fields_at_kx: boundary_kinds
    # ['periodic','mirror','periodic'], no metallic, Bloch k = (3.5, 0, 0).
    "folded_bloch": dict(
        cell=(2.4, 2.4, 0.0), dimensions=2, boundaries="periodic",
        symmetry=("Y",), k_point=(3.5, 0.0, 0.0), complex_storage=True,
        offdiag=True, pml_thickness=2),
    # The 3-D folded shape the released gate's third product row carries.
    "folded_3d": dict(
        cell=(1.6, 1.6, 1.6), symmetry=("X",), complex_storage=True,
        offdiag=True, pml_thickness=2),
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _probe(path: Path) -> Any:
    """One expansion artifact, or a DECLARED skip naming the absent resource.

    ``parity/meep_gpu/results/`` is gitignored in full, so these records
    are present only on a machine that cut or fetched them.  The skip is
    sanctioned rather than bare so a run that could not ask the question says so
    in its own summary instead of reporting a pass.
    """
    if not path.exists():  # pragma: no cover - artifact-absent checkouts
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "expansion_probe_artifact",
            f"this record is under the gitignored results tree and is absent "
            f"here: {path}")
    return json.loads(path.read_text())


def _build(**kwargs: Any) -> Tuple[Any, Any]:
    import sys

    parity = str(API / "parity" / "meep_gpu")
    if parity not in sys.path:
        sys.path.insert(0, parity)
    import probe_residual_group_bodies as bodies  # noqa: PLC0415

    fields, pml, _notes = bodies.build(np, **kwargs)
    return fields, pml


def _residual(reasons: Any) -> Tuple[str, ...]:
    """Every refusal except the one clause a NumPy host cannot satisfy."""
    return tuple(str(reason) for reason in reasons
                 if not (str(reason).startswith(BACKEND_HEAD)
                         and str(reason).endswith(BACKEND_TAIL)))


def _coverage(fields: Any, pml: Any, probe: Any) -> Any:
    from meep_gpu.expansion_refusal import declaring_run_policy
    from meep_gpu.triton_kernels import complex_offdiag_update_e as arm

    with declaring_run_policy(RUN_POLICY):
        return arm.complex_folded_offdiag_update_e_coverage(fields, pml, probe)


# ---------------------------------------------------------------------------
# The arm is registered, and it is the SOLE admitter
# ---------------------------------------------------------------------------

def _consulted_arms(fields: Any, pml: Any, probe: Any,
                    slot: str) -> List[Dict[str, Any]]:
    """Every arm ``launch.plan_step`` consults for one slot, with its verdict.

    ``launch._select_slot`` already receives the arm table and already evaluates
    every gated arm's predicate before selecting; it is wrapped to record that
    and then delegates.  Nothing about the selection rule is re-implemented, and
    the ONE CuPy clause is factored at the forwarder layer with exactly the
    filter :func:`_residual` applies — the same seam
    ``parity/meep_gpu/triton_predicate_battery.py`` uses.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy
    from meep_gpu.triton_kernels import launch
    from meep_gpu.triton_kernels.coverage import Coverage

    def factored(function: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            verdict = function(*args, **kwargs)
            if getattr(verdict, "covered", False):
                return verdict
            residual = _residual(getattr(verdict, "reasons", ()) or ())
            return Coverage(not residual, residual)
        wrapped.__name__ = getattr(function, "__name__", "wrapped")
        return wrapped

    recorded: List[Dict[str, Any]] = []
    original_select = launch._select_slot
    originals: Dict[str, Any] = {}
    for name in dir(launch):
        if name.endswith("_coverage"):
            attribute = getattr(launch, name)
            if callable(attribute):
                originals[name] = attribute
                setattr(launch, name, factored(attribute))

    def recording_select(which: str, arms: Any, *rest: Any) -> Any:
        arms = tuple(arms)
        if which == slot:
            for arm in arms:
                if not arm.gate:
                    recorded.append({"arm": arm.label, "gate": False,
                                     "covered": None, "reasons": ()})
                    continue
                try:
                    verdict = arm.predicate()
                    recorded.append({
                        "arm": arm.label, "gate": True,
                        "covered": bool(verdict.covered),
                        "reasons": tuple(str(r) for r in verdict.reasons)})
                except Exception as error:  # noqa: BLE001 - a raise is a refusal
                    recorded.append({"arm": arm.label, "gate": True,
                                     "covered": False,
                                     "reasons": (f"raised {error!r}",)})
        return original_select(which, arms, *rest)

    launch._select_slot = recording_select
    try:
        with declaring_run_policy(RUN_POLICY):
            launch.plan_step(fields, pml, probe=probe)
    finally:
        launch._select_slot = original_select
        for name, attribute in originals.items():
            setattr(launch, name, attribute)
    return recorded


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_the_folded_complex_offdiag_arm_is_consulted_at_update_e(fixture: str) -> None:
    """The arm the board reports as absent is in the shipped ``update_E`` table."""
    fields, pml = _build(**FIXTURES[fixture])
    arms = _consulted_arms(fields, pml, _probe(FOLDED_PROBE), "update_E")
    labels = [entry["arm"] for entry in arms]
    assert "complex folded off-diagonal" in labels, (
        f"{fixture}: update_E consulted {labels}; the folded complex "
        f"off-diagonal arm is not in the shipped table")
    entry = next(e for e in arms if e["arm"] == "complex folded off-diagonal")
    assert entry["gate"] is True, f"{fixture}: the arm is gated off"


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_it_is_the_sole_update_e_admitter_under_a_parity_carrying_artifact(
        fixture: str) -> None:
    """Sole, so the slot is FILLED rather than emptied for ambiguity.

    ``_select_slot`` leaves a slot unselected when two or more arms admit, so
    "an arm admits" is not the same claim as "the slot is served".  This asserts
    the stronger one.
    """
    fields, pml = _build(**FIXTURES[fixture])
    arms = _consulted_arms(fields, pml, _probe(FOLDED_PROBE), "update_E")
    admitted = sorted(entry["arm"] for entry in arms if entry["covered"])
    assert admitted == ["complex folded off-diagonal"], (
        f"{fixture}: update_E admitters are {admitted}; expected exactly the "
        f"folded complex off-diagonal arm")


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_the_census_artifact_refuses_it_for_the_artifact_and_nothing_else(
        fixture: str) -> None:
    """Under the census's own artifact the arm refuses — and ONLY about the artifact.

    This is the whole of the board's empty ``constitutive_admitters`` column,
    stated as a set difference rather than as a verdict: swapping the artifact
    must remove exactly the expansion clauses and add nothing.
    """
    from meep_gpu.triton_kernels.folded_complex import PARITY_PROBE_PATTERN

    fields, pml = _build(**FIXTURES[fixture])
    census = _residual(_coverage(fields, pml, _probe(CENSUS_PROBE)).reasons)
    folded = _residual(_coverage(fields, pml, _probe(FOLDED_PROBE)).reasons)

    assert folded == (), (
        f"{fixture}: the arm still refuses under the parity-carrying artifact: "
        f"{folded}")
    assert census, f"{fixture}: the census artifact was expected to refuse"
    assert all(PARITY_PROBE_PATTERN in reason or "EXPANSION" in reason
               for reason in census), (
        f"{fixture}: the census artifact refuses for a reason that is NOT about "
        f"the artifact, which would make this a real product gap: {census}")
    assert set(folded) - set(census) == set(), (
        f"{fixture}: swapping the artifact ADDED refusals: "
        f"{sorted(set(folded) - set(census))}")


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_the_board_s_own_credited_artifact_already_licenses_this_arm(
        fixture: str) -> None:
    """No NEW artifact is needed to credit the arm — an existing one licenses it.

    ``unified_expansion_2026-08-27/keep/gate.json`` is the record the board names
    in ``credited_under_a_different_expansion_licence`` for the ``B->H`` twin of
    these three rows.  It classifies all seven patterns, the parity one included.
    """
    fields, pml = _build(**FIXTURES[fixture])
    residual = _residual(_coverage(fields, pml, _probe(UNIFIED_PROBE)).reasons)
    assert residual == (), (
        f"{fixture}: the arm refuses under the unified record the board already "
        f"credits its B->H twin under: {residual}")


def test_the_census_artifact_does_not_classify_the_pattern_this_tranche_binds() -> None:
    """The mechanism, named at the artifacts rather than inferred from a refusal."""
    from meep_gpu.triton_kernels.folded_complex import (
        PARITY_PROBE_PATTERN, PARITY_PROBE_PATTERNS)

    census = _probe(CENSUS_PROBE).get("patterns", {})
    folded = _probe(FOLDED_PROBE).get("patterns", {})
    assert PARITY_PROBE_PATTERN in PARITY_PROBE_PATTERNS
    assert PARITY_PROBE_PATTERN not in census, (
        "the census artifact now classifies the parity pattern; if that is "
        "deliberate, this file's premise has changed and the board should be "
        "re-cut rather than this test relaxed")
    assert PARITY_PROBE_PATTERN in folded, (
        f"{FOLDED_PROBE} no longer classifies {PARITY_PROBE_PATTERN!r}")
    unified = _probe(UNIFIED_PROBE).get("patterns", {})
    assert PARITY_PROBE_PATTERN in unified, (
        f"{UNIFIED_PROBE} no longer classifies {PARITY_PROBE_PATTERN!r}; the "
        f"board's existing credit for the B->H twin rests on it")


# ---------------------------------------------------------------------------
# The controls that must refuse
# ---------------------------------------------------------------------------

def _without_parity(record: Any) -> Any:
    from meep_gpu.triton_kernels.folded_complex import PARITY_PROBE_PATTERN

    stripped = copy.deepcopy(record)
    stripped.get("detail", {}).pop(PARITY_PROBE_PATTERN, None)
    stripped.get("patterns", {}).pop(PARITY_PROBE_PATTERN, None)
    return stripped


def _parity_unclassifiable(record: Any) -> Any:
    from meep_gpu.triton_kernels.folded_complex import PARITY_PROBE_PATTERN

    broken = copy.deepcopy(record)
    entry = broken.get("detail", {}).get(PARITY_PROBE_PATTERN)
    if isinstance(entry, dict):
        entry["classified"] = "NEITHER"
        entry["matches"] = {key: False for key in entry.get("matches", {})}
        entry["mismatch_words"] = {key: 7 for key in entry.get("mismatch_words", {})}
    broken.get("patterns", {})[PARITY_PROBE_PATTERN] = "NEITHER"
    return broken


@pytest.mark.parametrize("name,overrides,artifact", [
    ("no_offdiagonal_row", {"offdiag": False}, "folded"),
    ("unfolded", {"symmetry": ()}, "folded"),
    ("real_storage", {"complex_storage": False}, "folded"),
    ("parity_pattern_deleted", {}, "deleted"),
    ("parity_pattern_unclassifiable", {}, "unclassifiable"),
])
def test_the_arm_refuses_every_control(name: str, overrides: Dict[str, Any],
                                       artifact: str) -> None:
    """Five controls.  An admission on ANY of them voids the admissions above."""
    folded = _probe(FOLDED_PROBE)
    record = {"folded": folded, "deleted": _without_parity(folded),
              "unclassifiable": _parity_unclassifiable(folded)}[artifact]
    fields, pml = _build(**dict(FIXTURES["folded_twofold"], **overrides))
    residual = _residual(_coverage(fields, pml, record).reasons)
    assert residual, (
        f"control {name} was ADMITTED modulo backend; the predicate is broader "
        f"than its product")


# ---------------------------------------------------------------------------
# The arithmetic the fifth pattern is about
# ---------------------------------------------------------------------------

def _words(array: Any) -> Any:
    return np.ascontiguousarray(array, dtype=np.complex64).view(
        np.float32).view(np.uint32)


def _assemble(out_re: Any, out_im: Any) -> Any:
    """Two float32 word planes into one complex64 plane, SIGNS INTACT.

    ``re + 1j * im`` is not this: that spelling is a complex multiply and a
    complex add, and both rewrite the zero signs these tests are about.
    """
    out = np.empty(np.shape(out_re), dtype=np.complex64)
    out.real = np.asarray(out_re, dtype=np.float32)
    out.imag = np.asarray(out_im, dtype=np.float32)
    return out


def _needles() -> Any:
    parts = np.array([0.0, -0.0, 1.0, -1.0, 3.5, -3.5,
                      np.float32(1e-45), np.float32(-1e-45)], dtype=np.float32)
    return _assemble(np.repeat(parts, len(parts)), np.tile(parts, len(parts)))


def _mul_coefficient_left_naive(c: float, z: Any) -> Any:
    """``complex_fields._mul_coefficient_left``'s NAIVE arm, in float32."""
    z_re, z_im = z.real.astype(np.float32), z.imag.astype(np.float32)
    c32, zero = np.float32(c), np.float32(0.0)
    return _assemble((c32 * z_re) - (zero * z_im), (c32 * z_im) + (zero * z_re))


@pytest.mark.parametrize("phase", [1, -1])
def test_the_array_paths_mirror_parity_is_a_full_complex_multiply(phase: int) -> None:
    """``_shift_down``'s MIRROR arm is ``int * complex64 plane`` and NOT a sign flip.

    The shipped Triton helper reproduces it word for word; negating both stored
    words does not.  Measured as differing uint32 words, so the claim is a count
    and not an adjective.
    """
    plane = _needles()
    reference = (phase * plane).astype(np.complex64)   # stepping.py's own spelling
    shipped = _mul_coefficient_left_naive(float(phase), plane)
    flipped = (plane.copy() if phase == 1
               else _assemble(-plane.real.astype(np.float32),
                              -plane.imag.astype(np.float32)))

    reference_words = _words(reference)
    assert int(np.count_nonzero(_words(shipped) != reference_words)) == 0, (
        "the shipped _mul_coefficient_left spelling no longer reproduces the "
        "array path's mirror parity")
    differing = int(np.count_nonzero(_words(flipped) != reference_words))
    assert differing > 0, (
        f"phase {phase:+d}: the bare sign flip matched the array path on all "
        f"{reference_words.size} needle words, so this fixture arms no "
        f"discriminator and the pin is vacuous")


def test_an_even_mirror_is_not_a_no_op_on_complex_storage() -> None:
    """``1 * z`` is not ``z``: the control that stops a ``+1`` short circuit.

    Where ``re`` is ``-0.0`` and ``im`` is negative, the full multiply's
    ``a*c - b*d`` returns ``+0.0``.  A kernel that skipped the multiply at an
    even mirror plane would be byte-wrong on exactly those words.
    """
    plane = _needles()
    reference = (1 * plane).astype(np.complex64)
    differing = int(np.count_nonzero(_words(plane) != _words(reference)))
    assert differing > 0, (
        "an even mirror parity is now a no-op on this NumPy; the +1 short "
        "circuit this test forbids would have become legal, which is a "
        "platform change and not a licence to relax the kernel")


def test_the_flip_and_the_full_multiply_agree_where_no_zero_is_signed() -> None:
    """Specificity: the divergence above is the zero-sign class, not noise."""
    rng = np.random.default_rng(20260902)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), 512)
    other = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), 512)
    plane = _assemble(rng.uniform(0.25, 4.0, 512).astype(np.float32) * signs,
                      rng.uniform(0.25, 4.0, 512).astype(np.float32) * other)
    for phase in (1, -1):
        reference = (phase * plane).astype(np.complex64)
        flipped = (plane.copy() if phase == 1
                   else _assemble(-plane.real.astype(np.float32),
                                  -plane.imag.astype(np.float32)))
        assert int(np.count_nonzero(_words(flipped) != _words(reference))) == 0, (
            f"phase {phase:+d}: the flip diverged on an all-normal plane, so the "
            f"signed-zero measurement above is not specific to signed zeros")


# ---------------------------------------------------------------------------
# Is re-cutting the census under the unified record safe for every family?
# ---------------------------------------------------------------------------

#: A spread wide enough that a regression would show: real and complex storage,
#: folded and unfolded, with and without an off-diagonal row, a pole, a
#: nonlinearity, a conductivity, and with the absorber off.
SWAP_CASES: Dict[str, Dict[str, Any]] = dict(
    FIXTURES,
    real_pml_3d=dict(),
    real_offdiag=dict(offdiag=True),
    complex_pml=dict(complex_storage=True),
    complex_offdiag_unfolded=dict(complex_storage=True, offdiag=True,
                                  boundaries="periodic",
                                  k_point=(0.4, 0.0, 0.0)),
    folded_real=dict(symmetry=("X",)),
    folded_real_offdiag=dict(symmetry=("X",), offdiag=True),
    folded_complex=dict(symmetry=("X",), complex_storage=True),
    folded_complex_offdiag=dict(symmetry=("X",), complex_storage=True,
                                offdiag=True),
    dispersive=dict(poles=1),
    nonlinear=dict(nonlinear=True),
    conductive=dict(conductivity=0.2),
    no_pml_real=dict(pml_thickness=0),
    no_pml_complex=dict(pml_thickness=0, complex_storage=True),
)


def _slot_admitters(fields: Any, pml: Any, probe: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for slot in ("step_B", "step_D", "fill_B", "fill_D", "update_H", "update_E"):
        recorded = _consulted_arms(fields, pml, probe, slot)
        if recorded:
            out[slot] = sorted(entry["arm"] for entry in recorded
                               if entry["covered"])
    return out


@pytest.mark.parametrize("case", sorted(SWAP_CASES))
def test_the_unified_record_never_costs_an_admitter(case: str) -> None:
    """The unified record is a STRICT SUPERSET, measured rather than argued.

    Same five shared patterns, same classifications, same backend, same device,
    same ``keep`` policy — so re-cutting the census under it should only ever ADD
    admitters.  A single LOST admitter would make that re-cut unsafe, which is
    the thing a reader needs to know before ordering one.
    """
    fields, pml = _build(**SWAP_CASES[case])
    before = _slot_admitters(fields, pml, _probe(CENSUS_PROBE))
    after = _slot_admitters(fields, pml, _probe(UNIFIED_PROBE))
    lost = {slot: sorted(set(before.get(slot, [])) - set(after.get(slot, [])))
            for slot in sorted(set(before) | set(after))
            if set(before.get(slot, [])) - set(after.get(slot, []))}
    assert lost == {}, (
        f"{case}: swapping the census artifact for the unified record LOST "
        f"admitters {lost}; the re-cut is not safe as stated")


def test_the_census_artifact_also_hides_the_folded_complex_fill_arms() -> None:
    """The blast radius is wider than the three D->E rows, and it is measured.

    The parity pattern is bound by the mirror-ghost FILL as well as by the
    off-diagonal ``update_E`` stencil, so the census artifact suppresses the
    ``folded complex fill`` arm at ``fill_B``/``fill_D`` on EVERY folded complex
    configuration — not only the off-diagonal ones.  Recorded here so a re-cut is
    scoped to what it actually changes.
    """
    fields, pml = _build(symmetry=("X",), complex_storage=True)
    before = _slot_admitters(fields, pml, _probe(CENSUS_PROBE))
    after = _slot_admitters(fields, pml, _probe(UNIFIED_PROBE))
    for slot in ("fill_B", "fill_D"):
        gained = set(after.get(slot, [])) - set(before.get(slot, []))
        assert "folded complex fill" in gained, (
            f"{slot}: the census artifact no longer hides the folded complex "
            f"fill arm (gained {sorted(gained)}); if a re-cut has landed, this "
            f"file's premise has changed")


# ---------------------------------------------------------------------------
# The released gate, named rather than assumed
# ---------------------------------------------------------------------------

def test_the_family_carries_a_released_device_byte_gate() -> None:
    """The arm is CERTIFIED, not merely present, and the ledger says so."""
    ledger = json.loads(
        (API / "meep_gpu/triton_kernels/fingerprints.json").read_text())
    entry = ledger.get("triton_complex_offdiag_device_gate")
    assert entry is not None, (
        "the complex off-diagonal device gate entry is gone from the ledger")
    assert entry.get("status") == "PASS", (
        f"the complex off-diagonal device gate is {entry.get('status')!r}")
    devices = entry.get("device_sha256", {})
    assert any("complex_offdiag_update_e" in path for path in devices), (
        "the released gate no longer welds the complex off-diagonal kernel")
