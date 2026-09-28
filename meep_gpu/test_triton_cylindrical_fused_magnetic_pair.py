"""The CYLINDRICAL COMPLEX fused magnetic Triton pair, as a laptop can check it.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/probe_triton_cylindrical_fused_magnetic_pair.py`` runs
three routes in lockstep on CuPy and compares uint32 words at one launch and sixty,
under BOTH float32 subnormal policies, and no assertion here duplicates that. What
lives here is everything true about the family WITHOUT a device: what the shipped
kernel TEXT says, which configurations the predicate refuses and by what name, that
the product is not an arm, and the choices a byte gate structurally cannot hold.

THE FIVE CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. **The product is not an arm.** Nothing in ``launch.py`` names this module and
   ``fastpath`` is untouched, so no default run can reach it. That is a property of
   the TREE, and it is what lets this family land beside the wired one.
2. **The kernel calls no multiply helper outside the three it declares.** A fourth
   would be a fourth operand orientation, which would need its own probe pattern
   before it could be licensed. The gate binds an arm; only a source scan says the
   arm covers every call.
3. **``BACKWARD`` is carried and bound to 0**, so the curl half is a VERBATIM copy of
   the certified cylindrical curl rather than a hand-specialised one — which is the
   whole reason this product's transcription risk is confined to two certified
   bodies plus a register clear.
4. **``ZM_X`` and ``ZM_Y`` are dead on every admitted grid**, and the predicate
   REFUSES a grid whose wall table says otherwise. A byte gate cannot assert that a
   constexpr branch is unreachable; the predicate and this suite can.
5. **The transcription is checkable by construction.** Every arithmetic statement of
   the certified cylindrical curl and of the shipped complex constitutive weld
   appears in this body — parsed out of all three files, so a planted defect is
   EXECUTED against the certified text rather than mirrored by a re-implementation.

THE ARM IS NOT ASSERTED HERE. Which expansion arm the platform takes is a measured
fact carried in a probe artifact, and a unit test that pinned one would be a guess
wearing a test's clothes.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
MODULE = PACKAGE_DIR / "triton_kernels" / "cylindrical_fused_magnetic_pair.py"
CURL = PACKAGE_DIR / "triton_kernels" / "cylindrical_complex.py"
TWIN = PACKAGE_DIR / "triton_kernels" / "complex_fused_magnetic_pair.py"
GATE = (REPO / "parity" / "meep_gpu"
        / "probe_triton_cylindrical_fused_magnetic_pair.py")

family = pytest.importorskip(
    "meep_gpu.triton_kernels.cylindrical_fused_magnetic_pair")


def body() -> str:
    """The shipped kernel definition, read out of the file rather than re-typed."""
    text = MODULE.read_text(encoding="utf-8")
    start = text.index("    def cyl_complex_fused_curl_constitutive_B(")
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


# ---------------------------------------------------------------------------
# The seam and the wiring
# ---------------------------------------------------------------------------

def test_replaces_is_the_five_driver_call_sites_in_driver_order():
    assert family.REPLACES == ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                               "fill_folded_far_ghosts_B", "update_H")
    assert family.CURL_SUB_STEP == "step_B"
    assert family.CONSTITUTIVE_SIDE == "H"


def test_backward_matches_the_sub_step_table_and_is_zero():
    """``BACKWARD`` is restated in this module so the kernel's one legal binding is
    visible without reading ``launch``; the two must be the same number."""
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert family.BACKWARD == 0 == int(SUB_STEPS["step_B"]["backward"])


def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-13 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; the 2026-09-13 phase B batch
    (``dispatch_fused_route_2026-09-13_phaseB``) drove it through the driver's own
    consults on the two cylindrical-complex arms this product covers, so the label
    now names them in ``RELEASED_FUSED_ARMS`` and the pending rung no longer holds
    it.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its ledger entry was cut from the phase B fleet
    artifact by ``seed_triton_welds.py``) and out of ``PENDING_DEVICE_GATE_ARMS``.
    Both are asserted rather than one, because a label released while still pending
    would be a plan claiming a certification the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_cylindrical_fused_magnetic_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["cylindrical_fused_magnetic_pair"]
    assert row["module"] == "cylindrical_fused_magnetic_pair"
    assert row["builder"] == "plan_cylindrical_fused_magnetic_pair"
    label = row["label"]
    assert label == 'fused pair B (cylindrical complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert _fastpath.RELEASED_FUSED_ARMS[label] == (
        'cylindrical_m1', 'cylindrical_m0_complex')
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def _ledger_claims(product: str) -> tuple[list[str], list[str]]:
    """Where the Triton ledger makes a claim ABOUT ``product``: keys, and digests.

    ASKED STRUCTURALLY, BECAUSE A SUBSTRING ALSO MATCHES PROSE. Until 2026-09-04
    this was ``product not in json.dumps(record)``, and on that day it went red
    over a sentence: ``family_recert_2026-08-14``'s ``source_drift_since_recert``
    declaration for ``cylindrical_complex.py`` names
    ``probe_triton_cylindrical_fused_magnetic_pair.py`` in its ``re_cert_owed``
    line, which says which gate OWES a re-run and hashes nothing. The claim the
    docstring below cares about is a checked-in HASH, so the two shapes that can
    carry one are what is looked for: a ledger ENTRY keyed on the product, and a
    DIGEST pinned against its module. Both are recursive — a digest is checked
    wherever it lives, the same rule ``weld_record_walk`` applies.
    """
    import json
    import re

    record = json.loads(
        (PACKAGE_DIR / "triton_kernels" / "fingerprints.json").read_text(
            encoding="utf-8"))
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    keys: list[str] = []
    digests: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                where = f"{path}.{key}"
                if product in str(key):
                    keys.append(where)
                if isinstance(value, str) and hex64.match(value) \
                        and str(key).rsplit("/", 1)[-1] == f"{product}.py":
                    digests.append(where)
                walk(value, where)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(record, "")
    return keys, digests


def test_the_module_has_a_fingerprints_weld():
    """RELEASED 2026-09-13, so a byte-identity claim IS now made: the phase B batch
    ran this family's device gate to a release verdict and ``seed_triton_welds.py``
    cut its weld from the 2026-09-12 identity fleet, keyed on the gate and pinning
    the module's checked-in digests. What used to be "no checked-in hash may imply
    a claim" is now the positive contract — the ledger carries the weld and pins
    the module, each named exactly so a stray prose mention cannot satisfy it."""
    keys, digests = _ledger_claims("cylindrical_fused_magnetic_pair")
    assert ".triton_cylindrical_fused_magnetic_pair_device_gate" in keys, (
        f"the ledger carries no weld entry for this product: {keys}")
    assert (".triton_cylindrical_fused_magnetic_pair_device_gate.code_sha256"
            ".meep_gpu/triton_kernels/cylindrical_fused_magnetic_pair.py"
            in digests), (
        f"the ledger pins no code digest for this module: {digests}")
    assert (".triton_cylindrical_fused_magnetic_pair_device_gate.source_sha256"
            ".meep_gpu/triton_kernels/cylindrical_fused_magnetic_pair.py"
            in digests), (
        f"the ledger pins no source digest for this module: {digests}")


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
    """The two cylindrical-only orientations are ARM-DEGENERATE — their coefficient's
    real word is a zero, so ``fma(+-0, z, -(c*z))`` and ``(+-0*z) - (c*z)`` are the
    same single-rounding operation — and the H-side constitutive adds none. So this
    product launches no orientation its halves do not."""
    from meep_gpu.triton_kernels.complex_fields import PROBE_PATTERNS

    assert family.PRODUCT_PROBE_PATTERNS == tuple(PROBE_PATTERNS)
    assert len(family.PRODUCT_PROBE_PATTERNS) == 4


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def test_every_certified_curl_statement_appears_in_the_fused_body():
    """PARSED OUT OF BOTH FILES. A suite that re-implemented the kernel's arithmetic
    would MIRROR a planted defect instead of executing it — measured on this project
    2026-08-20, where three planted assembly defects each left 86 of 87 mirrored
    tests passing."""
    certified = statements(
        CURL.read_text(encoding="utf-8"),
        "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    # --- stores: u then f (kernels.py:193-198 order), both planes ---")
    fused = set(statements(
        body(),
        "        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "        # --- the seam: stepping.zero_metal_B (driver.py:3286) ---"))
    missing = [line for line in certified if line not in fused]
    assert len(certified) > 100, "the certified statement scan found almost nothing"
    assert missing == []


def test_every_constitutive_statement_of_the_twin_appears_in_the_fused_body():
    """``update_H`` carries NO cylindrical branch — the cylindrical tranche measured
    480/480 rows, 0 differing words — so the constitutive half IS the complex twin's
    weld, character for character."""
    certified = statements(
        TWIN.read_text(encoding="utf-8"),
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


# ---------------------------------------------------------------------------
# The wall table
# ---------------------------------------------------------------------------

def test_the_forbidden_wall_axes_are_r_and_phi():
    assert family.FORBIDDEN_WALL_AXES == (0, 1)


def test_all_three_zero_metal_branches_are_written_even_though_two_are_dead():
    """The r and phi rows are TRANSCRIBED so ``_zero_metal``'s diagonal table is
    legible and half-written, not because either can fire. The predicate refuses a
    grid whose table says otherwise, so a needle inside either would be a
    dead-branch mutation — which is why the gate's guard evaluator models both as
    permanently False and FAILS on a needle placed inside one."""
    text = body()
    for flag in ("ZM_X", "ZM_Y", "ZM_Z"):
        assert f"if {flag}:" in text


def test_the_wall_clear_is_applied_to_both_planes():
    """``array[_face(axis, 0)] = 0`` on a complex64 volume writes COMPLEX zero
    (stepping._zero_metal :2246); clearing only the real plane is the plane-wise
    slip in its wall-clear disguise, and the gate catches it."""
    text = body()
    assert "v2_re = tl.where(at_z, 0.0, v2_re)" in text
    assert "v2_im = tl.where(at_z, 0.0, v2_im)" in text


def test_the_wall_clear_precedes_both_the_store_and_the_constitutive_read():
    """Applied to the REGISTERS, so the two consumers see the one value the array
    path leaves in B. A clear placed after the store would leave the constitutive
    half reading an uncleared Bz — which is the defect the release verdict was shown
    to flip against on both backends."""
    text = body()
    clear = text.index("if ZM_Z:")
    store = text.index("tl.store(f2 + 2 * idx, v2_re, mask=live)")
    constitutive = text.index("kp_0 = tl.load(kp0 + i")
    assert clear < store < constitutive


# ---------------------------------------------------------------------------
# The byte-invisible choice a gate cannot hold
# ---------------------------------------------------------------------------

def test_the_curl_grouping_keeps_the_invariant_axis_difference():
    """``stepping._curl_from_operands`` groups ``((A) + (B))``. ``t0`` leads with the
    PHI SELF-DIFFERENCE — an exact ``+0.0`` on a one-cell axis — so flattening its
    parens is BYTE-INVISIBLE (measured on the GPU host 2026-08-20: 0 differing words
    over 8 launches, against a catch for the identical edit on ``t1``). Kept because
    it is what the array path computes, and pinned here because the gate records it
    as a null rather than holding it."""
    text = body()
    assert "t0_re = ((c_p_re - c_re) + (b_re - b_z_re))" in text
    assert "t1_re = ((a_z_re - a_re) + (c_re - c_r_re))" in text
    assert "t2_re = ((b_r_re - b_re) + (a_re - a_p_re))" in text


def test_the_axis_increment_replaces_the_row_and_negates_with_a_literal():
    """REPLACE, never accumulate — and the negation is ``* -1.0``, never unary minus
    (Triton lowers ``-x`` as ``0.0 - x`` and canonicalizes +-0).

    MEASURED ON THE GPU HOST 2026-08-20 AND THIS IS WHY THE GATE SWEEPS VALUE CLASSES:
    the accumulate spelling is a CATCH on the +-0 lattice and a NULL under random
    seeding and under zero-init. A gate that swept uniform seeds alone would have
    reported the defect uncaught and called the kernel certified."""
    text = body()
    assert "curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)" in text
    assert "curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)" in text
    assert "tl.where(at_r, -inc_re" not in text


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
    if family.cyl_complex_fused_curl_constitutive_B is not None:
        pytest.skip("this host has Triton; the refusal path is not exercised")
    with pytest.raises(ImportError, match="triton"):
        family.cyl_complex_fused_curl_constitutive_B_kernel()


def test_an_undeclared_source_list_is_refused():
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    fields, pml = mx.cylindrical(m=1)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, None)
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_magnetic_source_is_refused_by_name():
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = mx.cylindrical(m=1)
    magnetic = VolumeSource(grid=fields.grid, component="Hy",
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))
    verdict = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (magnetic,))
    assert not verdict.covered
    assert any("is magnetic" in reason and "driver.py:3283" in reason
               for reason in verdict.reasons)


def test_the_predicate_prefixes_each_half_so_a_reader_can_tell_which_said_it():
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    fields, pml = mx.cylindrical(m=0, complex_storage=False)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert any(reason.startswith("cylindrical curl half: ")
               for reason in verdict.reasons)
    assert any(reason.startswith("constitutive half: ")
               for reason in verdict.reasons)


def test_the_plan_refuses_to_none_rather_than_raising():
    """``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly."""
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    fields, pml = mx.cylindrical(m=1)
    assert family.plan_cylindrical_fused_magnetic_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The gate exists and carries the machinery this suite refers to
# ---------------------------------------------------------------------------

def test_the_device_gate_carries_the_guard_evaluator_and_the_value_classes():
    text = GATE.read_text(encoding="utf-8")
    assert "GUARD_RULES" in text and "guard_chain_holds" in text
    assert '"signed_zero"' in text and '"subnormal_band"' in text
    assert 'POLICIES: Tuple[str, ...] = ("keep", "flush")' in text
    assert "case_rng" in text and "hashlib.sha256" in text
