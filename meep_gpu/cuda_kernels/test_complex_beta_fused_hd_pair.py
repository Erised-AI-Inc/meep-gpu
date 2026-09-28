"""The COMPLEX beta H->D weld, read on a host with no device.

EVERY CLAUSE HERE RUNS ON A LAPTOP and none of them is skipped, which is a property of
the product rather than of the test: this family's whole transform chain
(``complex_emitter`` -> ``complex_folded_kernels`` -> ``complex_beta_kernels`` -> this
weld) is CuPy-free at module scope, so the device text can be emitted and compared
where there is no device. The byte identity itself is
``parity/meep_gpu/gate_complex_beta_fused_hd_pair.py``'s and lives in its artifacts.

THE TWO CENTRAL CLAIMS ARE EQUALITIES:

1. :func:`complex_beta_fused_hd_pair.curl_pieces` applied to the CERTIFIED PLAIN
   complex ``step_D`` text returns exactly what
   ``complex_fused_hd_pair.curl_pieces("plain", arm)`` returns, and applied to the
   CERTIFIED FOLDED text exactly what its ``"folded"`` variant returns -- byte for
   byte, on BOTH arms. The beta weld is therefore the RELEASED, GATED transform plus
   the beta insert.
2. The two cells this ROUND targeted -- complex beta and real beta -- are TWO
   PRODUCTS and not one, and that was measured rather than assumed: the clause below
   compares the two cells' certified ``update_H`` bodies and their two curl preludes
   and requires all four to be different strings.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from . import complex_beta_fused_hd_pair as family
from . import complex_beta_kernels, complex_emitter, complex_folded_kernels
from . import complex_fused_hd_pair as released

HERE = pathlib.Path(__file__).parent
ARMS = tuple(sorted(complex_emitter.EXPANSIONS))


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_module_declares_the_seam_it_spans():
    assert family.SLOT == "update_H"
    assert family.REPLACES == ("update_H", "step_D")
    from .. import withdraw_hoist
    assert family.SEAM == withdraw_hoist.SEAM
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.INSTALLABLE is False
    assert "slot arbitration" in family.INSTALLABLE_REASON


def test_the_kernel_is_in_exactly_one_half_of_the_partition():
    assert family.KERNEL_NAME not in family.CERTIFIED_KERNELS
    assert family.KERNEL_NAME in family.UNCERTIFIED_KERNELS
    assert not set(family.CERTIFIED_KERNELS) & set(family.UNCERTIFIED_KERNELS)
    assert "gate_complex_beta_fused_hd_pair.py" in \
        family.UNCERTIFIED_KERNELS[family.KERNEL_NAME]


def test_the_declaration_is_a_literal_this_files_own_readers_can_see():
    """``test_kernel_partition.py`` scans this file's TEXT for the declaration.

    The parameter list is derived from the released sibling's plus
    ``complex_beta_kernels``' own delta, so the NAME has to be spelled here as a
    literal or the package-wide partition walk cannot see the kernel this module ships.
    """
    text = (HERE / "complex_beta_fused_hd_pair.py").read_text(encoding="utf-8")
    assert f'extern "C" __global__ void {family.KERNEL_NAME}(' in text
    literals = {}
    for node in ast.parse(text).body:
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else (node.targets[0] if isinstance(node, ast.Assign)
                        and len(node.targets) == 1 else None))
        if isinstance(target, ast.Name) and target.id in (
                "CERTIFIED_KERNELS", "UNCERTIFIED_KERNELS"):
            literals[target.id] = ast.literal_eval(node.value)
    assert set(literals) == {"CERTIFIED_KERNELS", "UNCERTIFIED_KERNELS"}
    assert literals["UNCERTIFIED_KERNELS"] == dict(family.UNCERTIFIED_KERNELS)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_the_transform_on_the_certified_plain_text_is_the_released_transform(arm):
    """THE FIRST EQUALITY, byte for byte, on each arm."""
    assert family.curl_pieces(
        arm, complex_emitter.complex_source("step_D", arm)) == \
        released.curl_pieces("plain", arm)


@pytest.mark.parametrize("arm", ARMS)
def test_the_transform_on_the_certified_folded_text_is_the_released_transform(arm):
    """THE SECOND, on the text this family's beta source is actually derived FROM."""
    assert family.curl_pieces(
        arm, complex_folded_kernels.folded_source("step_D", arm)) == \
        released.curl_pieces("folded", arm)


@pytest.mark.parametrize("arm", ARMS)
def test_the_weld_redirects_every_magnetic_read_in_the_beta_body(arm):
    head, tail = family.curl_pieces(arm)
    assert tail.count("cshift_dn_recompute(") == family.HALO_TAPS
    assert tail.count("own_h[") == family.OWN_LOAD_EDITS
    for pointer in family.CURL_SOURCE_NAMES:
        assert f"cf_load({pointer}," not in tail
    assert "cshift_dn(" not in tail
    # The head keeps the certified preamble, the strides, the decode, the three phase
    # constants AND the two beta coefficient registers, verbatim.
    assert "cf bp; bp.re = bp_re; bp.im = bp_im;" in head
    assert "cf bm; bm.re = bm_re; bm.im = bm_im;" in head


def test_both_beta_partners_are_the_welds_own_register():
    """The clause this family owns and no sibling H->D weld has."""
    _head, tail = family.curl_pieces("FMA_V1")
    assert "curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));" in tail
    assert "curl = cf_sub(curl, mul_imag_coefficient_left(bm, f_1));" in tail
    assert "cf f_2 = own_h[1];" in tail      # target 0's Hy centre
    assert "cf f_1 = own_h[0];" in tail      # target 1's Hx centre


def test_a_beta_partner_the_weld_did_not_redirect_is_refused():
    """The armed control for the clause above: it must BITE.

    ``ss`` is the SHIFTED companion, which welded is a recompute rather than a weld
    register -- the risk ``complex_beta_kernels`` names when it says the register must
    be the UNSHIFTED centre.
    """
    corrupted = family._curl_source("FMA_V1").replace(  # noqa: SLF001
        "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));",
        "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, ss));", 1)
    with pytest.raises(AssertionError, match="beta partner"):
        family.curl_pieces("FMA_V1", corrupted)


def test_a_beta_partner_declared_from_the_pre_launch_volume_is_refused():
    """The OTHER branch of the pairing check, driven on a synthetic block."""
    block = ("    // Target 0\n"
             "    {\n"
             "        cf f_1 = own_h[2];\n"
             "        cf f_2 = cf_load(g1, idx);\n"
             "        cf curl = cf_zero();\n"
             "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n"
             "    }\n")
    with pytest.raises(AssertionError, match="PRE-launch magnetic field"):
        family._assert_the_beta_partner_is_the_weld_register(block, True)  # noqa: SLF001


def test_a_body_that_lost_a_beta_increment_is_refused():
    corrupted = family._curl_source("FMA_V1").replace(  # noqa: SLF001
        "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n", "", 1)
    with pytest.raises(AssertionError, match="beta increments"):
        family.curl_pieces("FMA_V1", corrupted)


# ---------------------------------------------------------------------------
# ONE PRODUCT PER STORAGE CLASS -- measured, not assumed
# ---------------------------------------------------------------------------

def test_the_two_beta_cells_share_no_text_and_are_therefore_two_products():
    """WHY THIS ROUND SHIPPED TWO MODULES RATHER THAN ONE WITH TWO VARIANTS.

    The released complex H->D product covers TWO cells with ONE transform because its
    two cells share their ``update_H`` half EXACTLY and its two ``step_D`` halves are
    one text through three anchored deltas. Across the storage boundary neither holds,
    and this clause is the measurement rather than the assertion: the two certified
    ``update_H`` bodies are different strings, the two curl preludes are different
    strings, and the complex one carries a scalar type the real one does not.

    Read WITHOUT importing the real halves' modules, which need CuPy: the real family's
    text is taken off its own source with the AST reader ``special_kz_curl`` already
    uses for the same reason.
    """
    real_curl = _real_literal("step_curl_kernels.py", "_REAL_PML_PRELUDE")
    real_constitutive = _real_literal("constitutive_kernels.py",
                                      "_REAL_CONSTITUTIVE_PRELUDE")
    complex_curl, _ = family._curl_source("FMA_V1").split(  # noqa: SLF001
        '\nextern "C" __global__ void ', 1)
    complex_update_h = complex_emitter.complex_source("update_H", "FMA_V1")
    real_update_h = _real_literal("constitutive_kernels.py",
                                  "_update_H_pml_real_kernel_code")
    assert complex_curl != real_curl
    assert complex_update_h != real_update_h
    assert real_constitutive not in complex_curl
    assert "cf " in complex_curl and "cf " not in real_curl
    # And the two products' kernels are different names, so no record can confuse them.
    from . import special_kz_fused_hd_pair as real_family
    assert family.KERNEL_NAME != real_family.KERNEL_NAME


def _real_literal(module: str, name: str) -> str:
    """One string literal off a sibling's SOURCE, without importing it.

    The two certified real modules import CuPy at scope; parsing their source for the
    literal is the technique ``special_kz_curl._sibling_prelude`` uses and is what lets
    this comparison run on a laptop.
    """
    source = (HERE / module).read_text(encoding="utf-8")
    literals: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals[target.id] = value.value
        elif (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
              and isinstance(value.left, ast.Name)
              and isinstance(value.right, ast.Constant)
              and value.left.id in literals):
            literals[target.id] = literals[value.left.id] + value.right.value
    return literals[name]


# ---------------------------------------------------------------------------
# The measured absence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_the_emitted_text_performs_no_floating_point_division(arm):
    source = family.kernel_source(arm)      # raises if the absence is violated
    divides = [line for line in source.splitlines()
               if "/" in line.split("//", 1)[0]]
    assert len(divides) == family._PERMITTED_DIVIDES  # noqa: SLF001
    assert all(line.rstrip() in family._PERMITTED_DIVIDE_LINES  # noqa: SLF001
               for line in divides)


def test_the_absence_assertion_bites():
    """A zero beside a control that moves. CuPy's ``complex64 / float32`` is the
    SCALED complex/complex algorithm and has no site here; a planted divide must be
    refused rather than emitted."""
    source = family.kernel_source("FMA_V1")
    with pytest.raises(AssertionError, match="divides outside"):
        family._assert_the_measured_absence_of_a_divide(  # noqa: SLF001
            source.replace("    cf own_h[3];", "    float bad = 1.0f / dtdx;", 1))


def test_a_dropped_index_decode_is_refused():
    """The decode is what indexes the absorber profile at the RECOMPUTED cell.

    Dropping one copy is the half-cell class of error -- converged, smooth and wrong --
    and the divide-count weld catches it on the host as well as the mutation battery
    catching it on a device.
    """
    source = family.kernel_source("FMA_V1")
    with pytest.raises(AssertionError, match="index-decode divide lines"):
        family._assert_the_measured_absence_of_a_divide(  # noqa: SLF001
            source.replace("    int j = (idx / nz) % ny;\n", "    int j = 0;\n", 1))


# ---------------------------------------------------------------------------
# The signature and the digest
# ---------------------------------------------------------------------------

def test_the_signature_is_the_released_ones_with_the_certified_beta_delta():
    """Neither half is retyped, so neither can drift."""
    old, new = family._beta_argument_delta()  # noqa: SLF001
    assert old in released._SIGNATURE_BODY   # noqa: SLF001
    assert "float bp_re, float bp_im, float bm_re, float bm_im" in new
    signature = family.signature()
    assert signature.startswith(
        f'\nextern "C" __global__ void {family.KERNEL_NAME}(\n')
    assert signature.endswith("\n) {\n")
    # The released body, minus its own declaration line, is carried whole.
    assert released._SIGNATURE_BODY.split(  # noqa: SLF001
        "    float pxr,")[0] in signature


def test_the_digest_moves_when_any_input_text_moves():
    """One sha256 over both arms; a changed character anywhere upstream moves it."""
    before = family.source_digest()
    original = complex_beta_kernels._MUL_IMAG_HELPER  # noqa: SLF001
    try:
        complex_beta_kernels._MUL_IMAG_HELPER = original.replace(  # noqa: SLF001
            "    return rotate_field_left(c, z);",
            "    return rotate_field_left(c, z);  // moved")
        assert family.source_digest() != before
    finally:
        complex_beta_kernels._MUL_IMAG_HELPER = original  # noqa: SLF001
    assert family.source_digest() == before


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _Grid:
    beta = 0.0
    dimensions = 2
    cylindrical = False

    @staticmethod
    def has_symmetry():
        return False

    @staticmethod
    def is_mirrored(axis):  # noqa: ARG004
        return False


class _Fields:
    force_complex_fields = True


@pytest.mark.parametrize("fields,grid,names", [
    (_Fields(), _Grid(), "grid.beta is zero"),
    (type("F", (), {"force_complex_fields": False})(),
     type("G", (_Grid,), {"beta": 0.25})(), "special_kz_curl"),
    (_Fields(), type("G", (_Grid,), {"beta": 0.25, "cylindrical": True})(),
     "cylindrical"),
    (_Fields(), type("G", (_Grid,), {"beta": 0.25, "dimensions": 3})(), "dimensions"),
])
def test_the_predicate_refuses_by_name(fields, grid, names):
    licence = {"arm": "FMA_V1", "refusals": [], "basis": "probe", "policy": "keep"}
    covered, reason = family.covers_complex_beta_fused_hd_pair(
        fields, None, grid, (), licence, "keep")
    assert covered is False
    assert names.lower() in reason.lower()


def test_the_arm_is_required_and_never_defaulted():
    with pytest.raises(ValueError):
        family.kernel_source("SOMETHING")


def test_an_undeclared_source_set_is_never_an_empty_one():
    from .. import withdraw_hoist

    reasons = withdraw_hoist.seam_withdraw_reasons(
        _Fields(), None, undeclared="the source set was not declared",
        refusal=lambda index, source: "standing",
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
    assert reasons and "not declared" in reasons[0]
