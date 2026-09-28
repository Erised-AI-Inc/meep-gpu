"""The COMPLEX CONDUCTIVE fused Triton pair, as a laptop can check it.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/probe_triton_complex_conductive_fused_pair.py`` runs
three routes in lockstep on CuPy and compares uint32 words PER COMPLETE DRIVER
STEP, and no assertion here duplicates that. What lives here is everything true
about the family WITHOUT a device: what the shipped kernel TEXT says, which
configurations the predicate refuses and by what name, that the product is not an
arm, and the choices a byte gate structurally cannot hold.

THE FIVE CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. **The product is not an arm.** Nothing in ``launch.py`` names this module and
   ``fastpath`` is untouched, so no default run can reach it.
2. **``REPLACES`` is TWO call sites, not five, and that is a claim about what the
   launch performs.** The three passes in between are REFUSED — by a named clause
   each — rather than carried, and a product that claimed five would be licensing
   an inline pass it does not do.
3. **``CARRIES_DEPOSIT_REPAIR`` is False, and that is not a limitation here.** All
   four corpus rows of this cell declare a MAGNETIC source only, so the seam clause
   refuses nothing the corpus drives. The flag is a claim about the PLAN; this
   suite asserts it against ``test_fused_pair_deposit_wiring``'s own sets rather
   than restating it.
4. **The wall refusal is PRICED, not accidental.** A metallic axis buys zero
   seam-instances on the measured corpus, so ``zero_metal_D`` is declined by name
   instead of carried — and the refusal must be a REFUSAL, not a silent skip.
5. **The pole chain is the certified helper with exactly two lines removed.** The
   order is bit-load-bearing; the weld may not pre-sum or reorder.

THE ARM IS NOT ASSERTED HERE. Which expansion arm the platform takes is a measured
fact carried in a probe artifact, and a unit test that pinned one would be a guess
wearing a test's clothes.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
MODULE = PACKAGE_DIR / "triton_kernels" / "complex_conductive_fused_pair.py"
CURL = PACKAGE_DIR / "triton_kernels" / "complex_no_pml_conductive.py"
STORED = PACKAGE_DIR / "triton_kernels" / "complex_no_pml_stored_e.py"
GATE = (REPO / "parity" / "meep_gpu"
        / "probe_triton_complex_conductive_fused_pair.py")

family = pytest.importorskip(
    "meep_gpu.triton_kernels.complex_conductive_fused_pair")


def body() -> str:
    """The shipped kernel definitions, read out of the file rather than re-typed."""
    text = MODULE.read_text(encoding="utf-8")
    start = text.index("@triton.jit\ndef _subtract_complex_poles_in_registers(")
    end = text.index("def complex_conductive_fused_curl_constitutive_D_kernel(")
    return text[start:end]


def statements(text: str, start: str, end: str) -> list:
    """LOGICAL statements, not physical lines: continuation lines joined by bracket
    depth and whitespace collapsed, so a REFLOW — which changes no token and no
    float32 bit — does not fire, while any changed token or paren does."""
    block = text[text.index(start):text.index(end)]
    out, pending, depth = [], "", 0
    for line in block.splitlines():
        stripped = line.strip()
        if not depth and (not stripped or stripped.startswith("#")
                          or stripped.startswith('"')):
            continue
        pending = (pending + " " + stripped).strip() if pending else stripped
        depth += stripped.count("(") - stripped.count(")")
        if depth <= 0:
            out.append(" ".join(pending.split()))
            pending, depth = "", 0
    if pending:
        out.append(" ".join(pending.split()))
    return out


def fixture():
    """A live complex conductive no-PML ``(fields, pml)`` from the shared matrix."""
    if str(REPO / "parity" / "meep_gpu") not in sys.path:
        sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    return mx.complex_conductive_stored_e_no_pml()


# ---------------------------------------------------------------------------
# The seam and the wiring
# ---------------------------------------------------------------------------

def test_replaces_is_TWO_call_sites_and_the_other_three_are_refused_not_carried():
    """A launch may only claim to replace what it actually performs. The electric
    injection, ``zero_metal_D`` and the two symmetry fills are each refused by a
    named clause; declaring them here would license an inline pass this kernel does
    not do."""
    assert family.REPLACES == ("step_D", "update_E")
    assert family.CURL_SUB_STEP == "step_D"
    # There is no CONSTITUTIVE_SIDE: the stored-E half is the no-PML branch of
    # ``update_E``, which is not one of ``coverage.CONSTITUTIVE_SIDES``.
    assert not hasattr(family, "CONSTITUTIVE_SIDE")


def test_backward_matches_the_sub_step_table_and_is_one():
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert family.BACKWARD == 1 == int(SUB_STEPS["step_D"]["backward"])


def test_the_composer_routes_this_product_and_the_release_names_its_one_case():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-15 on one case — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    RELEASED ON ONE CASE SINCE 2026-09-15, and the second half is that sentence made
    checkable from the PRODUCT's side rather than from the composer's. Until then
    this test asserted the label was NOT in ``fastpath.RELEASED_FUSED_ARMS``, which
    was true while the pair weld had no ledger entry. The release names exactly
    ``complex_no_pml_3d``, the Triton route case restored in the same batch, and the
    label now sits in ``ARM_CERTIFICATION`` (the pair weld ``seed_triton_welds.py``
    cuts from the re-run pair gate) and NOT in ``PENDING_DEVICE_GATE_ARMS``; the
    exactly-one-of rule between those two maps is unchanged.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_complex_conductive_fused_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["complex_conductive_fused_pair"]
    assert row["module"] == "complex_conductive_fused_pair"
    assert row["builder"] == "plan_complex_conductive_fused_pair"
    label = row["label"]
    assert label == 'fused pair D (complex conductive no-PML)'
    assert _fastpath.arm_is_fused(label)
    assert _fastpath.RELEASED_FUSED_ARMS[label] == ("complex_no_pml_3d",)
    assert label in _fastpath.ARM_CERTIFICATION
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS


def test_the_carry_flag_is_False_and_the_wiring_sets_say_so():
    """FLAG AND WIRING MOVE TOGETHER, in both directions.

    This family is served in full WITHOUT the repair — all four corpus rows declare
    a magnetic source only — so the flag is False. Until 2026-09-02 that meant the
    module appeared in NEITHER wiring set, because no composer could install it.
    The installer wave routed it, so the pair of declarations it now owes is
    "installable, deliberately unbracketed": absent from ``WIRED_FOR_THE_REPAIR``
    and PRESENT in ``ROUTED_WITHOUT_THE_REPAIR``, which is the same pair the Metal
    off-diagonal welds carry.
    """
    from meep_gpu.test_fused_pair_deposit_wiring import (  # noqa: PLC0415
        DECLARED_BUT_NOT_ROUTED, ROUTED_WITHOUT_THE_REPAIR, WIRED_FOR_THE_REPAIR)

    key = "triton_kernels/complex_conductive_fused_pair.py"
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert key not in WIRED_FOR_THE_REPAIR
    assert key not in DECLARED_BUT_NOT_ROUTED
    assert key in ROUTED_WITHOUT_THE_REPAIR


def test_the_module_makes_no_identity_claim_it_has_not_earned():
    text = MODULE.read_text(encoding="utf-8")
    assert "bit-identical" not in text.lower()


# ---------------------------------------------------------------------------
# The multiply helpers — the probe-pattern claim
# ---------------------------------------------------------------------------

def test_the_kernel_calls_no_multiply_helper_outside_the_three_it_declares():
    tree = ast.parse(body())
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    multiplies = {name for name in called if "_mul" in name}
    assert multiplies <= set(family.LICENSED_MULTIPLY_HELPERS), multiplies
    assert multiplies, "the scan found NO multiply helper; it is measuring nothing"


def test_the_probe_pattern_set_is_the_base_four_unextended():
    """The seam adds NO multiply at all — it removes two loads — so this product
    launches no operand orientation its halves do not."""
    from meep_gpu.triton_kernels.complex_fields import PROBE_PATTERNS

    assert family.PRODUCT_PROBE_PATTERNS == tuple(PROBE_PATTERNS)
    assert len(family.PRODUCT_PROBE_PATTERNS) == 4


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def test_every_certified_curl_statement_appears_in_the_fused_body():
    """PARSED OUT OF BOTH FILES. A suite that re-implemented the kernel's arithmetic
    would MIRROR a planted defect instead of executing it."""
    certified = statements(
        CURL.read_text(encoding="utf-8"),
        "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "def complex_conductive_no_pml_curl_coverage(")
    fused = set(statements(
        body(),
        "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    # ==================== the constitutive half ===="))
    missing = [line for line in certified if line not in fused]
    assert len(certified) > 100, "the certified statement scan found almost nothing"
    assert missing == []


def test_the_pole_helper_is_the_certified_one_with_exactly_its_two_loads_removed():
    """:func:`_subtract_complex_poles_in_registers` is
    ``complex_no_pml_stored_e._subtract_complex_poles`` with its two OPENING LOADS
    replaced by the two arguments and NOTHING else changed. The removed lines are
    NAMED rather than tolerated: a helper missing a subtraction arm would also show
    as "missing", and a check that shrugged at both could not tell them apart."""
    certified = statements(
        STORED.read_text(encoding="utf-8"),
        "def _subtract_complex_poles(",
        "@triton.jit\ndef complex_stored_e_step(")
    weld = set(statements(
        body(), "def _subtract_complex_poles_in_registers(",
        "@triton.jit\ndef complex_conductive_fused_curl_constitutive_D("))
    missing = [line for line in certified if line not in weld]
    removed = {
        "real = tl.load(source + word, mask=live, other=0.0)",
        "imag = tl.load(source + word + 1, mask=live, other=0.0)",
    }
    signature = [line for line in missing
                 if line.startswith("def _subtract_complex_poles(")]
    assert len(signature) == 1, "the certified signature line was not found"
    assert set(missing) - set(signature) == removed, sorted(missing)
    # ...and all eight arms survive, in order.
    for pole in range(8):
        assert f"real = real - tl.load(p{pole} + word, mask=live, other=0.0)" in weld
        assert (f"imag = imag - tl.load(p{pole} + word + 1, mask=live, other=0.0)"
                in weld)


def test_the_pole_arms_appear_in_registration_order_and_are_never_pre_summed():
    """THE ORDER IS BIT-LOAD-BEARING: pre-summing or reordering changes float32
    rounding at two or more poles, which is why the certified body spells eight
    separate arms and why the gate arms a mutation that reverses them."""
    text = body()
    positions = [text.index(f"real = real - tl.load(p{pole} + word,")
                 for pole in range(8)]
    assert positions == sorted(positions)


def test_the_seam_takes_the_register_not_a_reload_of_D():
    text = body()
    for register, poles in ((0, "a"), (1, "b"), (2, "q")):
        assert (f"v{register}_re, v{register}_im, {poles}0," in text)
    # The certified helper's own source argument is gone from the call sites.
    assert "_subtract_complex_poles_in_registers(\n        tl.load(f" not in text


def test_the_D_store_is_KEPT_and_precedes_the_constitutive_half():
    """The array path leaves the stepped displacement in D and the NEXT timestep's
    ``update_P`` reads it, so a weld that dropped the store would fuse away a value
    the run still needs — a defect no per-step E comparison would see."""
    text = body()
    for target in range(3):
        assert f"tl.store(f{target} + 2 * idx, v{target}_re, mask=live)" in text
    assert (text.index("tl.store(f2 + 2 * idx + 1, v2_im, mask=live)")
            < text.index("word = 2 * idx"))


def test_the_inv_eps_load_is_indexed_UNDOUBLED_and_applied_after_the_poles():
    """``inv_eps`` is a float32 volume at the COMPLEX cell index — ``+ idx``, not
    ``+ word``. And the multiply comes AFTER the pole chain: ``(D - sum P) *
    inv_eps``, not ``(D * inv_eps) - sum P``."""
    text = body()
    for target in range(3):
        assert f"tl.load(iv{target} + idx, mask=live, other=0.0)" in text
        assert f"tl.load(iv{target} + word" not in text
        chain = text.index(f"s{target}_re, s{target}_im = "
                           f"_subtract_complex_poles_in_registers(")
        multiply = text.index(f"o{target}_re, o{target}_im = _mul_field_left(")
        assert chain < multiply


def test_the_conductive_tail_is_three_rounded_operations_in_order():
    """``D *= condfac; D -= curl; D *= condinv`` — each in-place complex64 operation
    rounds before the next begins, so the three may not be flattened and the factor
    may not move across the subtraction."""
    text = body()
    for target in range(3):
        factor = text.index(f"factor = tl.load(cf{target} + idx,")
        subtract = text.index(f"v{target}_re = v{target}_re - curl{target}_re")
        inverse = text.index(f"inverse = tl.load(ci{target} + idx,")
        assert factor < subtract < inverse


# ---------------------------------------------------------------------------
# Refusals — each one PRICED
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = fixture()
    verdict = family.complex_conductive_fused_pair_coverage(fields, pml, None)
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_an_electric_source_is_refused_by_name_citing_the_conductive_injection():
    """The injection lands between the halves, and on a conductive row it takes the
    ``condinv``-scaled path. THE CONDUCTIVE INJECTION IS THE SAME CLAUSE, not a
    second one: ``driver.step`` guards it with ``if electric and
    has_conductivity``, so with no electric source it cannot run at all."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = fixture()
    electric = VolumeSource(grid=fields.grid, component="Ez",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    verdict = family.complex_conductive_fused_pair_coverage(
        fields, pml, (electric,))
    assert not verdict.covered
    assert any("is electric" in reason and "driver.py:3294" in reason
               for reason in verdict.reasons)
    assert family.plan_complex_conductive_fused_pair(
        fields, pml, sources=(electric,)) is None


def test_a_magnetic_source_does_NOT_trip_the_source_clause():
    """THE POLARITY THAT MAKES THIS CELL REACHABLE AT ALL with the flag at False:
    all four corpus rows declare a magnetic source, injected in the B/H seam."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = fixture()
    magnetic = VolumeSource(grid=fields.grid, component="Hy",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    verdict = family.complex_conductive_fused_pair_coverage(
        fields, pml, (magnetic,))
    assert [r for r in verdict.reasons if "is electric" in r] == []


def test_a_metallic_axis_is_refused_by_name_and_the_refusal_is_priced():
    """``zero_metal_D`` runs inside this seam and this family does not carry it. The
    refusal is a DECISION with a measured price — a metallic axis buys zero
    seam-instances on the corpus — and the clause is not a comment: an unreadable
    boundary table is a refusal too."""
    fields, pml = fixture()
    grid = fields.grid

    class _Walled:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def has_metallic(self, *args, **kwargs):
            return True

        def is_metallic(self, axis):
            return axis == 0

    class _Fields:
        def __init__(self, inner, grid):
            self._inner = inner
            self.grid = grid

        def __getattr__(self, name):
            return getattr(self._inner, name)

    verdict = family.complex_conductive_fused_pair_coverage(
        _Fields(fields, _Walled(grid)), pml, ())
    assert not verdict.covered
    assert any("zero_metal_D" in reason for reason in verdict.reasons)


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A configuration either half refuses is refused here WITH that half's reasons,
    prefixed so a reader can tell which side said it."""
    text = MODULE.read_text(encoding="utf-8")
    assert 'f"curl half: {reason}"' in text
    assert 'f"E half: {reason}"' in text


def test_the_plan_refuses_to_none_rather_than_raising():
    fields, pml = fixture()
    assert family.plan_complex_conductive_fused_pair(fields, pml, None) is None


def test_a_missing_expansion_probe_is_a_refusal_not_a_guessed_arm():
    fields, pml = fixture()
    verdict = family.complex_conductive_fused_pair_coverage(
        fields, pml, (), probe=None)
    assert not verdict.covered
    assert any("expansion" in reason.lower() for reason in verdict.reasons)


def test_the_predicate_never_raises_on_a_degenerate_object():
    """``None`` is the only refusal, and a caller handing this predicate something
    it does not understand must get a refusal rather than a traceback."""

    class _Nothing:
        pass

    verdict = family.complex_conductive_fused_pair_coverage(_Nothing(), None, ())
    assert not verdict.covered
    assert verdict.reasons


# ---------------------------------------------------------------------------
# The gate exists and carries the machinery this suite refers to
# ---------------------------------------------------------------------------

def test_the_device_gate_exists_and_carries_its_guard_evaluator():
    text = GATE.read_text(encoding="utf-8")
    assert "GUARD_RULES" in text and "guard_chain_holds" in text
    assert "polarization_snapshot" in text
    assert "ABSORBED" in text


def test_the_gate_arms_a_needle_for_every_seam_choice_this_suite_pins():
    """The two halves of the same claim: this suite pins the TEXT, the gate EXECUTES
    a defect against it. A choice pinned here with no needle there is a choice
    nothing measures on a device."""
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import probe_triton_complex_conductive_fused_pair as gate  # noqa: PLC0415

    for needle in ("seam_takes_the_pre_conductive_curl",
                   "pole_chain_order_reversed",
                   "pole_chain_drops_the_last_arm",
                   "inv_eps_word_doubled",
                   "inv_eps_applied_before_the_poles",
                   "conductive_factor_and_inverse_swapped",
                   "conductive_factor_applied_after_the_subtraction",
                   "displacement_store_dropped"):
        assert needle in gate.MUTATIONS, needle
    block = gate.shipped_block()
    absent = [name for name, (_case, old, _new, _exp) in gate.MUTATIONS.items()
              if old not in block]
    assert absent == []


def test_the_gate_scores_every_mutation_on_a_case_that_compiles_its_guards():
    """The dead-branch class, checked from here as well as inside the gate: the gate
    fails the leg on a device, and this fails the merge on a laptop."""
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import probe_triton_complex_conductive_fused_pair as gate  # noqa: PLC0415

    leg = gate.leg_needle_reachability()
    unreachable = [row["mutation"] for row in leg["rows"]
                   if row["case_compiles_every_guard"] is not True]
    assert unreachable == []
    assert leg["passed"]


def test_the_module_is_welded_to_the_bytes_its_gate_executed():
    """THE WELD LANDED 2026-09-17, and this test turned over with it.

    It used to assert the ABSENCE of a ``fingerprints.json`` entry, with a
    ``pytest.skip`` as its own escape hatch for the day the gate ran. That day came:
    ``seed_triton_welds.py`` cut ``triton_complex_conductive_fused_pair_device_gate``
    from the re-run that released this product, and ``fused pair D (complex conductive
    no-PML)`` now dispatches on ``complex_no_pml_3d``.

    The skip could not survive either way — this suite's conftest refuses an
    unsanctioned skip and fails it by name, so a test that skips itself once its
    premise expires is a test that goes red rather than quiet. What replaces it is the
    assertion the landing actually earns, and it is strictly stronger than the
    ``assert True`` the old body fell through to: the entry EXISTS, it records a PASS,
    and every source digest it pins still equals the tree. That last clause is the one
    with teeth — it is what a later edit to the kernel, the probe or the composer would
    break, and it is checked here rather than assumed.
    """
    import hashlib  # noqa: PLC0415

    fingerprints = json.loads(
        (PACKAGE_DIR / "triton_kernels" / "fingerprints.json").read_text(
            encoding="utf-8"))
    entry = fingerprints.get("triton_complex_conductive_fused_pair_device_gate")
    assert entry, ("the weld is released but its ledger entry is missing; "
                   "seed_triton_welds.py cuts it from the pair gate's own artifact")
    assert entry.get("status") == "PASS", entry.get("status")

    recorded = entry.get("source_sha256") or {}
    assert recorded, "the entry records no source digests"
    api_root = PACKAGE_DIR.parent
    moved = []
    for relative, want in sorted(recorded.items()):
        target = api_root / relative
        got = (hashlib.sha256(target.read_bytes()).hexdigest()
               if target.is_file() else "MISSING")
        if got != want:
            moved.append(relative)
    assert not moved, f"the weld no longer binds the tree on {moved}"

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415

    label = "fused pair D (complex conductive no-PML)"
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
