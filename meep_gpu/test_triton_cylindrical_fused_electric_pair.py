"""The CYLINDRICAL COMPLEX fused ELECTRIC Triton pair, as a laptop can check it.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/probe_triton_cylindrical_fused_electric_pair.py`` runs
three routes in lockstep on CuPy and compares uint32 words PER COMPLETE DRIVER
STEP, with and without a real electric deposit in the seam, and no assertion here
duplicates that. What lives here is everything true about the family WITHOUT a
device: what the shipped kernel TEXT says, which configurations the predicate
refuses and by what name, that the product is not an arm, and the choices a byte
gate structurally cannot hold.

THE SIX CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. **The product is not an arm.** Nothing in ``launch.py`` names this module and
   ``fastpath`` is untouched, so no default run can reach it. That is a property of
   the TREE, and it is what lets this family land beside the wired ones.
2. **The kernel calls no multiply helper outside the three it declares.** A fourth
   would be a fourth operand orientation, which would need its own probe pattern
   before it could be licensed. The gate binds an arm; only a source scan says the
   arm covers every call.
3. **``BACKWARD`` is carried and bound to 1, and ``SCALE`` to 1**, so the curl half
   is a VERBATIM copy of the certified cylindrical curl and the constitutive half a
   verbatim copy of the Cartesian D->E weld — which is the whole reason this
   product's transcription risk is confined to two certified bodies plus a register
   clear.
4. **``ZM_X`` and ``ZM_Y`` are dead on every admitted grid**, and the predicate
   REFUSES a grid whose wall table says otherwise. A byte gate cannot assert that a
   constexpr branch is unreachable; the predicate and this suite can.
5. **The wall clear takes the D-SIDE component map**, which is the exact complement
   of the magnetic twin's — a z wall clears ``Dx`` and ``Dy`` and leaves ``Dz``
   alone. The predicate re-derives it from the shipped ``IYEE_SHIFTS`` and refuses
   on a disagreement; this suite pins that the refusal is real by feeding it a grid
   whose map it can no longer satisfy.
6. **The deposit-repair flag and the wiring it claims move together.** A product
   that passed ``True`` without the bracket would compute the constitutive half
   against a pre-injection field and report success. The flag is asserted against
   ``test_fused_pair_deposit_wiring``'s own sets rather than restated here.

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
MODULE = PACKAGE_DIR / "triton_kernels" / "cylindrical_fused_electric_pair.py"
CURL = PACKAGE_DIR / "triton_kernels" / "cylindrical_complex.py"
TWIN = PACKAGE_DIR / "triton_kernels" / "cylindrical_fused_magnetic_pair.py"
CARTESIAN = PACKAGE_DIR / "triton_kernels" / "complex_fused_electric_pair.py"
GATE = (REPO / "parity" / "meep_gpu"
        / "probe_triton_cylindrical_fused_electric_pair.py")

family = pytest.importorskip(
    "meep_gpu.triton_kernels.cylindrical_fused_electric_pair")


def body() -> str:
    """The shipped kernel definition, read out of the file rather than re-typed."""
    text = MODULE.read_text(encoding="utf-8")
    start = text.index("    def cyl_complex_fused_curl_constitutive_D(")
    end = text.index("else:  # pragma: no cover - laptop path")
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


def cylindrical(**kwargs):
    """A live Dcyl ``(fields, pml)`` from the shared composition fixture."""
    if str(REPO / "parity" / "meep_gpu") not in sys.path:
        sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    return mx.cylindrical(**kwargs)


# ---------------------------------------------------------------------------
# The seam and the wiring
# ---------------------------------------------------------------------------

def test_replaces_is_the_five_driver_call_sites_in_driver_order():
    assert family.REPLACES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                               "fill_folded_far_ghosts_D", "update_E")
    assert family.CURL_SUB_STEP == "step_D"
    assert family.CONSTITUTIVE_SIDE == "E"


def test_backward_matches_the_sub_step_table_and_is_one():
    """``BACKWARD`` is restated in this module so the kernel's one legal binding is
    visible without reading ``launch``; the two must be the same number, and it is
    the OPPOSITE of the magnetic twin's."""
    from meep_gpu.triton_kernels.launch import SUB_STEPS
    from meep_gpu.triton_kernels import cylindrical_fused_magnetic_pair as twin

    assert family.BACKWARD == 1 == int(SUB_STEPS["step_D"]["backward"])
    assert twin.BACKWARD == 0


def test_the_scale_constexpr_is_the_E_sides_and_the_twin_carries_none():
    """``SCALE = 1`` is the E-side arm: the source is multiplied by ``inv_eps``
    BEFORE the ``w`` store. The H-side twin's arm reads ``B`` directly, which is why
    it binds the source views as placeholders and never loads them."""
    from meep_gpu.triton_kernels import cylindrical_fused_magnetic_pair as twin

    assert family.SCALE == 1
    assert not hasattr(twin, "SCALE") or twin.SCALE == 0


def test_the_composer_routes_this_product_and_dispatch_admits_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch — both halves.

    This test replaces the refusal it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    The 2026-09-13 phase-B batch RELEASED it, and the second half is that sentence
    made checkable from the PRODUCT's side rather than from the composer's: the
    label this product writes is now in ``fastpath.RELEASED_FUSED_ARMS``, so clause
    (8) admits the whole plan; and it carries an ``ARM_CERTIFICATION`` row (its gate
    has a tracked ledger entry) while having LEFT ``PENDING_DEVICE_GATE_ARMS`` (the
    pending rung no longer refuses it).
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_cylindrical_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["cylindrical_fused_electric_pair"]
    assert row["module"] == "cylindrical_fused_electric_pair"
    assert row["builder"] == "plan_cylindrical_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (cylindrical complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_makes_no_identity_claim_it_has_not_earned():
    """No byte-identity claim is made in the module, so no checked-in hash may imply
    one; the gate hashes what it actually ran, into its own artifact."""
    text = MODULE.read_text(encoding="utf-8")
    assert "NOT RELEASED" in text or "not wired" in text.lower()
    assert "bit-identical" not in text.lower()


# ---------------------------------------------------------------------------
# The deposit repair — the clause this product turns on
# ---------------------------------------------------------------------------

def test_the_carry_flag_is_true_and_the_wiring_sets_agree_with_it():
    """FLAG AND WIRING MOVE TOGETHER. A product that declares the flag without the
    ``LeadingRepairPlan``/``TrailingRepairPlan`` bracket computes the constitutive
    half against a pre-injection field and reports success — the exact failure
    ``deposit_repair`` exists to prevent. The wiring sets live in
    ``test_fused_pair_deposit_wiring``; this asserts against THEM rather than
    restating the claim."""
    from meep_gpu.test_fused_pair_deposit_wiring import WIRED_FOR_THE_REPAIR

    key = "triton_kernels/cylindrical_fused_electric_pair.py"
    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert key in WIRED_FOR_THE_REPAIR


def test_an_electric_deposit_is_CARRIED_rather_than_refused():
    """THE WHOLE PRODUCT. All sixteen Dcyl corpus rows declare an electric source,
    so with ``carries_repair=False`` this arm admits ZERO rows. The clause still
    consults ``deposit_repair.repairable`` per run — it is a carry, not a waiver."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = cylindrical(m=1)
    electric = VolumeSource(grid=fields.grid, component="Ez",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    quiet = family.cylindrical_fused_electric_pair_coverage(fields, pml, ())
    carried = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (electric,))
    # The two verdicts must agree: the deposit is carried, not merely tolerated on
    # a configuration that was already refused for another reason.
    assert [r for r in carried.reasons if "is electric" in r] == []
    assert carried.covered == quiet.covered


def test_holding_the_carry_flag_False_puts_the_electric_refusal_straight_back(
        monkeypatch):
    """The flag is LOAD-BEARING, and this is the measurement of that. With it False
    the shipped clause refuses an electric seam BY NAME — which is what this family
    would serve on the corpus without the repair: nothing."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = cylindrical(m=1)
    electric = VolumeSource(grid=fields.grid, component="Ez",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    verdict = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (electric,))
    assert not verdict.covered
    assert any("is electric" in reason and "driver.py:3296" in reason
               for reason in verdict.reasons)


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so
    a predicate that inferred "no electric source" from not being told would be the
    over-covering refusal this clause exists to prevent."""
    fields, pml = cylindrical(m=1)
    verdict = family.cylindrical_fused_electric_pair_coverage(fields, pml, None)
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_magnetic_source_is_irrelevant_to_THIS_seam():
    """A magnetic source is injected in the B/H seam (driver.py:3283) and does not
    reach this one. The twin refuses it; this product must not."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = cylindrical(m=1)
    magnetic = VolumeSource(grid=fields.grid, component="Hy",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    verdict = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (magnetic,))
    assert [r for r in verdict.reasons if "is magnetic" in r] == []


# ---------------------------------------------------------------------------
# The multiply helpers — the probe-pattern claim
# ---------------------------------------------------------------------------

def test_the_kernel_calls_no_multiply_helper_outside_the_three_it_declares():
    """A fourth helper would be a fourth operand orientation, which would need its
    own probe pattern before it could be licensed. Scanned from the AST of the
    shipped body, so a call added anywhere in it is caught."""
    tree = ast.parse("if True:\n" + body())
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    multiplies = {name for name in called if "_mul" in name}
    assert multiplies <= set(family.LICENSED_MULTIPLY_HELPERS), multiplies
    assert multiplies, "the scan found NO multiply helper; it is measuring nothing"


def test_the_probe_pattern_set_is_the_base_four_unextended():
    """The cylindrical-only orientations are ARM-DEGENERATE — their coefficient's
    real word is a zero — and the E-side constitutive's ``inv_eps`` multiply is
    ``_mul_field_left``, an orientation the recurrence above it already launches six
    times. So this product launches no orientation its halves do not."""
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
        "    # --- stores: u then f (kernels.py:193-198 order), both planes ---")
    fused = set(statements(
        body(),
        "        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "        # --- the seam: stepping.zero_metal_D (driver.py:3301) ---"))
    missing = [line for line in certified if line not in fused]
    assert len(certified) > 100, "the certified statement scan found almost nothing"
    assert missing == []


def test_every_constitutive_statement_of_the_cartesian_weld_appears_here():
    """``update_E`` carries NO cylindrical branch — the cylindrical tranche measured
    480/480 rows, 0 differing words — so the constitutive half IS the Cartesian
    D->E weld's, character for character."""
    certified = statements(
        CARTESIAN.read_text(encoding="utf-8"),
        "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
        "else:  # pragma: no cover - laptop path")
    fused = set(statements(
        body(),
        "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
        "        tl.store(h2 + 2 * idx + 1, a_im, mask=live)"))
    fused.add("tl.store(h2 + 2 * idx + 1, a_im, mask=live)")
    missing = [line for line in certified if line not in fused]
    assert len(certified) > 30
    assert missing == []


def test_the_seam_takes_the_register_not_a_reload():
    text = body()
    for target in range(3):
        assert f"src_re = v{target}_re" in text
        assert f"src_im = v{target}_im" in text
    assert "src_re = tl.load(g0" not in text


def test_the_inv_eps_load_is_guarded_on_SCALE_and_indexed_UNDOUBLED():
    """``inv_eps`` is a float32 VOLUME at the COMPLEX cell index — ``+ idx``, not
    ``+ 2 * idx``. Word-doubling it would read the imaginary neighbour's coefficient
    into the real plane, which is smooth, converged and wrong."""
    text = body()
    for target in range(3):
        assert f"ie = tl.load(e{target} + idx, mask=live, other=0.0)" in text
        assert f"tl.load(e{target} + 2 * idx" not in text
    assert text.count("if SCALE:") == 3


def test_the_inv_eps_multiply_precedes_the_w_store_on_every_component():
    """``f_w_E*`` holds ``D * inv_eps`` and not ``D`` (complex_fields.py:783-786).
    Moving the multiply after the store is a history one step stale, which the gate
    arms as a catch."""
    text = body()
    multiply = "src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)"
    for target in range(3):
        # The component's OWN region: from its `prev` read to its `w` store. A
        # positional compare against the whole body would be satisfied by any
        # earlier component's multiply and would measure nothing.
        opens = text.index(f"prev_re = tl.load(w{target} + 2 * idx,")
        closes = text.index(f"tl.store(w{target} + 2 * idx, src_re, mask=live)")
        region = text[opens:closes]
        assert multiply in region, target
        assert f"ie = tl.load(e{target} + idx" in region, target
    # ...and the multiply appears EXACTLY three times, once per component: a fourth
    # would be an unlicensed orientation and a second in one region would double it.
    assert text.count(multiply) == 3


# ---------------------------------------------------------------------------
# The wall table and the D-side component map
# ---------------------------------------------------------------------------

def test_the_forbidden_wall_axes_are_r_and_phi():
    assert family.FORBIDDEN_WALL_AXES == (0, 1)


def test_all_three_zero_metal_branches_are_written_even_though_two_are_dead():
    """The r and phi rows are TRANSCRIBED so ``_zero_metal``'s diagonal table is
    legible, not because either can fire. The predicate refuses a grid whose table
    says otherwise, so a needle inside either would be a dead-branch mutation —
    which is why the gate's guard evaluator models both as permanently False and
    FAILS on a needle placed inside one."""
    text = body()
    for flag in ("ZM_X", "ZM_Y", "ZM_Z"):
        assert f"if {flag}:" in text


def test_the_wall_clear_takes_the_D_SIDE_map_and_not_the_magnetic_twins():
    """``_zero_metal`` clears every component whose Yee shift on the WALLED axis is
    ZERO. For B that is the component on its own axis; for D it is the OTHER TWO —
    a z wall clears ``Dx`` and ``Dy`` and LEAVES ``Dz`` ALONE. This is the one place
    a transcriber gets this seam wrong, and the Cartesian twin shipped the magnetic
    map to its first device run and came back wrong in both directions at once."""
    from meep_gpu.fields import IYEE_SHIFTS

    derived = tuple(
        tuple(index for index, name in enumerate(family.CURL_TARGETS)
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    assert family.WALL_CLEARED_COMPONENTS == derived == ((1, 2), (0, 2), (0, 1))
    text = body()
    # The z branch — the only one that can fire on a Dcyl grid — clears v0 and v1.
    z_branch = text[text.index("if ZM_Z:"):]
    assert "v0_re = tl.where(at_z, 0.0, v0_re)" in z_branch
    assert "v1_re = tl.where(at_z, 0.0, v1_re)" in z_branch
    assert "v2_re = tl.where(at_z, 0.0, v2_re)" not in z_branch


def test_the_predicate_refuses_when_the_baked_wall_map_leaves_the_engines():
    """THE CLAUSE THAT WOULD HAVE CAUGHT THE CARTESIAN TWIN'S FIRST DEVICE RUN. The
    map is re-derived from the shipped ``IYEE_SHIFTS`` on every call; a kernel whose
    baked map disagrees is refused rather than launched."""
    fields, pml = cylindrical(m=1)
    original = family.WALL_CLEARED_COMPONENTS
    try:
        family.WALL_CLEARED_COMPONENTS = ((0,), (1,), (2,))  # the magnetic map
        verdict = family.cylindrical_fused_electric_pair_coverage(fields, pml, ())
        assert not verdict.covered
        assert any("would zero one plane and leave another live" in reason
                   for reason in verdict.reasons)
    finally:
        family.WALL_CLEARED_COMPONENTS = original


def test_the_wall_clear_is_applied_to_both_planes():
    """``array[_face(axis, 0)] = 0`` on a complex64 volume writes COMPLEX zero
    (stepping._zero_metal :2246); clearing only the real plane is the plane-wise
    slip in its wall-clear disguise, and the gate catches it."""
    text = body()
    z_branch = text[text.index("if ZM_Z:"):]
    assert "v0_im = tl.where(at_z, 0.0, v0_im)" in z_branch
    assert "v1_im = tl.where(at_z, 0.0, v1_im)" in z_branch


def test_the_wall_clear_precedes_both_the_store_and_the_constitutive_read():
    """Applied to the REGISTERS, so the two consumers see the one value the array
    path leaves in D. A clear placed after the store would leave the constitutive
    half reading an uncleared displacement."""
    text = body()
    clear = text.index("if ZM_Z:")
    store = text.index("tl.store(f0 + 2 * idx, v0_re, mask=live)")
    constitutive = text.index("kp_0 = tl.load(kp0 + i")
    assert clear < store < constitutive


# ---------------------------------------------------------------------------
# The ordering fact this product's docstring names
# ---------------------------------------------------------------------------

def test_the_seam_reads_the_value_the_axis_rules_left_and_not_the_recurrences():
    """``_cylindrical_axis_zero_D`` runs AFTER ``_apply_pml_update``, so the value
    ``update_E`` must consume is the one BELOW the per-|m| block, not the
    recurrence's output. A weld that read ``v`` above it would hand ``update_E`` a
    displacement the array path never leaves in D — which is exactly the subtlety
    the CUDA track measured on this same family."""
    text = body()
    recurrence = text.index("v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)")
    axis_rules = text.index("if M_CLASS == M_ONE:\n            if BACKWARD:\n"
                            "                v2_re = tl.where(at_r, 0.0, v2_re)")
    seam = text.index("src_re = v0_re")
    assert recurrence < axis_rules < seam


def test_the_m_many_hold_zeroes_the_auxiliaries_as_well_as_the_fields():
    """|m| >= 2 holds all three components AND their ``fu`` on every row within
    ``ZERO_ROWS`` of the axis (stepping :565-567, :663-671)."""
    text = body()
    hold = text[text.index("            near = i < ZERO_ROWS"):]
    for register in ("v0", "v1", "v2", "n0", "n1", "n2"):
        assert f"{register}_re = tl.where(near, 0.0, {register}_re)" in hold
        assert f"{register}_im = tl.where(near, 0.0, {register}_im)" in hold


def test_the_axis_increment_targets_Dy_and_not_the_B_sides_component():
    """``AXIS_INCREMENT_TARGET['step_D'] == 1`` against ``step_B``'s 0. A
    transcriber carrying the B side's target here writes a silent wrong-component
    increment, which the gate arms as a catch."""
    from meep_gpu.triton_kernels.cylindrical_complex import (
        AXIS_INCREMENT_TARGET)

    assert AXIS_INCREMENT_TARGET["step_D"] == 1
    text = body()
    backward_arm = text[text.index("        if M_CLASS == M_ONE:"):]
    assert "curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)" in backward_arm
    assert "curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)" in backward_arm  # forward arm
    assert backward_arm.index("curl1_re = tl.where(at_r, inc_re * -1.0") < \
        backward_arm.index("curl0_re = tl.where(at_r, inc_re * -1.0")


def test_the_axis_increment_replaces_the_row_and_negates_with_a_literal():
    """REPLACE, never accumulate — and the negation is ``* -1.0``, never unary minus
    (Triton lowers ``-x`` as ``0.0 - x`` and canonicalizes +-0)."""
    text = body()
    assert "curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)" in text
    assert "tl.where(at_r, -inc_re" not in text


def test_the_curl_grouping_is_the_array_paths_and_is_not_flattened():
    """``stepping._curl_from_operands`` groups ``((A) + (B))``."""
    text = body()
    assert "t0_re = ((c_p_re - c_re) + (b_re - b_z_re))" in text
    assert "t1_re = ((a_z_re - a_re) + (c_re - c_r_re))" in text
    assert "t2_re = ((b_r_re - b_re) + (a_re - a_p_re))" in text


def test_the_D_side_prefix_substitution_is_the_backward_arm_and_not_the_B_sides():
    """The D-side prefix is ``(nr, ny, nz)`` from ``Hy`` at ``ir0 = 0.5`` with no
    extended wall row, and its substitution is a BACKWARD difference of the prefix
    against the same ``(a - a_p)`` term. The B side's is the forward difference of
    the EXTENDED prefix — one subtract, one multiply — and taking it here is a
    silently different curl."""
    from meep_gpu.triton_kernels.cylindrical_complex import PREFIX

    assert PREFIX["step_D"] != PREFIX["step_B"]
    text = body()
    assert "t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))" in text
    assert "p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)" in text


def test_the_imr_fold_is_a_subtract_and_the_coefficient_is_on_the_left():
    text = body()
    assert "_mul_general_coefficient_left(q0_re, q0_im, c_re, c_im" in text
    assert "_mul_general_coefficient_left(q2_re, q2_im, a_re, a_im" in text
    assert "curl0_re = curl0_re - m0_re" in text
    assert "curl2_re = curl2_re - m2_re" in text


# ---------------------------------------------------------------------------
# Refusals and the host route
# ---------------------------------------------------------------------------

def test_the_kernel_refuses_by_name_without_triton():
    """The module must be importable on the laptop that is the merge bar, and the
    refusal must NAME the missing package rather than raising an AttributeError."""
    if family.cyl_complex_fused_curl_constitutive_D is not None:
        pytest.skip("this host has Triton; the refusal path is not exercised")
    with pytest.raises(ImportError, match="triton"):
        family.cyl_complex_fused_curl_constitutive_D_kernel()


def test_real_storage_m_zero_is_refused_by_name_it_is_the_other_products_row():
    """Since 2026-09-04 m = 0 is admitted UNDER COMPLEX STORAGE (the M_ZERO arm);
    the REAL-storage m = 0 run is still ``cylindrical_triton``'s and the refusal
    names the storage flag, not m."""
    fields, pml = cylindrical(m=0, complex_storage=False)
    verdict = family.cylindrical_fused_electric_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert any("force_complex_fields is not set" in reason
               for reason in verdict.reasons), verdict.reasons
    assert not any("grid.m = 0" in reason for reason in verdict.reasons)


def test_the_predicate_prefixes_each_half_so_a_reader_can_tell_which_said_it():
    fields, pml = cylindrical(m=0, complex_storage=False)
    verdict = family.cylindrical_fused_electric_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert any(reason.startswith("cylindrical curl half: ")
               for reason in verdict.reasons)
    assert any(reason.startswith("constitutive half: ")
               for reason in verdict.reasons)


def test_a_registered_susceptibility_is_refused_by_this_products_own_clause():
    """A susceptibility makes the constitutive source ``(D - sum P)`` rather than
    ``D``. The E half already refuses it; restated because THIS kernel bakes the
    plain product. This is a D->E-only clause with no counterpart on the twin."""

    class _Polarized:
        def __init__(self, inner):
            self._inner = inner
            self.polarizations = ("a susceptibility",)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    fields, pml = cylindrical(m=1)
    verdict = family.cylindrical_fused_electric_pair_coverage(
        _Polarized(fields), pml, ())
    assert not verdict.covered
    assert any("susceptibility is registered" in reason
               for reason in verdict.reasons)


def test_a_fields_that_cannot_hand_over_inv_eps_is_refused_not_crashed():
    """An absent accessor would be a ``TypeError`` inside the builder rather than a
    refusal, which is the wrong way for an uncovered configuration to fail."""

    class _NoEpsilon:
        def __init__(self, inner):
            self._inner = inner
            self.inverse_epsilon_for = None

        def __getattr__(self, name):
            return getattr(self._inner, name)

    fields, pml = cylindrical(m=1)
    verdict = family.cylindrical_fused_electric_pair_coverage(
        _NoEpsilon(fields), pml, ())
    assert not verdict.covered
    assert any("inverse_epsilon_for" in reason for reason in verdict.reasons)


def test_the_plan_refuses_to_none_rather_than_raising():
    """``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly."""
    fields, pml = cylindrical(m=1)
    assert family.plan_cylindrical_fused_electric_pair(fields, pml, ()) is None


def test_a_missing_expansion_probe_is_a_refusal_not_a_guessed_arm():
    """Which arm the platform takes is a MEASURED fact. Without a probe the halves
    refuse, and the builder must return ``None`` rather than picking one."""
    fields, pml = cylindrical(m=1)
    verdict = family.cylindrical_fused_electric_pair_coverage(
        fields, pml, (), probe=None)
    assert not verdict.covered
    assert any("expansion" in reason.lower() for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The gate exists and carries the machinery this suite refers to
# ---------------------------------------------------------------------------

def test_the_device_gate_exists_and_carries_its_guard_evaluator():
    text = GATE.read_text(encoding="utf-8")
    assert "GUARD_RULES" in text and "guard_chain_holds" in text
    assert "CARRY_CASES" in text and "carry_null_control" in text
    assert "LeadingRepairPlan" in text and "TrailingRepairPlan" in text
    assert "require_repairs" in text


def test_the_gate_arms_a_needle_for_every_seam_choice_this_suite_pins():
    """The two halves of the same claim: this suite pins the TEXT, the gate EXECUTES
    a defect against it. A choice pinned here with no needle there is a choice
    nothing measures on a device."""
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import probe_triton_cylindrical_fused_electric_pair as gate  # noqa: PLC0415

    for needle in ("seam_takes_the_pre_recurrence_curl",
                   "seam_reads_before_the_axis_rules",
                   "zero_metal_takes_the_magnetic_twins_component_map",
                   "axis_increment_writes_the_B_sides_target",
                   "inv_eps_word_doubled",
                   "inv_eps_multiply_moved_after_the_w_store",
                   "prefix_substitution_dropped"):
        assert needle in gate.MUTATIONS, needle
    # And every needle must still be findable in the shipped source: a mutation
    # whose needle has drifted arms NOTHING, and the gate would report it as an
    # absent needle only when it is next run on a device.
    block = gate.shipped_block()
    absent = [name for name, (_case, old, _new, _exp) in gate.MUTATIONS.items()
              if old not in block]
    assert absent == []


def test_the_module_has_no_fingerprints_entry_until_its_gate_has_run():
    """WHICH BRANCH THIS TAKES IS ASKED STRUCTURALLY, NOT AS A SUBSTRING.

    ``product in json.dumps(record)`` also matches PROSE, and on 2026-09-04 it
    did: ``family_recert_2026-08-14``'s ``source_drift_since_recert``
    declaration for ``cylindrical_complex.py`` names
    ``probe_triton_cylindrical_fused_electric_pair.py`` in its ``re_cert_owed``
    line. That flipped this test into its "the weld is bound" branch — where it
    PASSED, over a sentence, while no ledger entry for this product exists. A
    branch chosen by prose is worse than a red: it stops asking. The two shapes
    that can carry a byte-identity claim are a ledger ENTRY keyed on the product
    and a DIGEST pinned against its module, both at any depth; see
    ``test_triton_cylindrical_fused_magnetic_pair._ledger_claims`` for the same
    rule on the sibling product.
    """
    import re

    record = json.loads(
        (PACKAGE_DIR / "triton_kernels" / "fingerprints.json").read_text(
            encoding="utf-8"))
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    product = "cylindrical_fused_electric_pair"
    claims: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                where = f"{path}.{key}"
                if product in str(key) or (
                        isinstance(value, str) and hex64.match(value)
                        and str(key).rsplit("/", 1)[-1] == f"{product}.py"):
                    claims.append(where)
                walk(value, where)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(record, "")
    text = MODULE.read_text(encoding="utf-8")
    if claims:
        # Once the gate has run and the weld is bound, the module must say so
        # rather than keeping its unreleased notice.
        assert "NOT RELEASED" not in text, claims
    else:
        assert "NOT RELEASED" in text or "not wired" in text.lower()
