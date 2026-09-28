"""The REAL beta H->D weld, read on a host with no device.

WHAT THIS FILE PINS is what a device gate cannot: that the TRANSFORM is the released
one, that the beta partner is the weld's own register, that the two measured absences
are asserted rather than described, and that the module's declarations agree with each
other. The byte identity itself is
``parity/meep_gpu/gate_special_kz_fused_hd_pair.py``'s and lives in its artifacts.

THE CENTRAL CLAIM HERE IS AN EQUALITY, and it is the reason this product is small:
:func:`special_kz_fused_hd_pair.welded_curl_tail` applied to the CERTIFIED (beta-free)
``step_D_pml_real`` text returns exactly what the RELEASED
``fused_hd_pair.welded_curl_tail`` returns -- so the beta weld is the released, gated
transform plus the beta insert and one new assertion, measured rather than asserted in
prose. That clause needs the two certified curl modules, which import CuPy at scope, so
it declares itself skipped where there is none rather than passing quietly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from . import fused_hd_pair
from . import special_kz_curl
from . import special_kz_fused_hd_pair as family

HERE = pathlib.Path(__file__).parent


#: THE CERTIFIED DEVICE TEXT IS A GENUINELY ABSENT RESOURCE ON A HOST WITHOUT CuPy,
#: and it is declared as one rather than skipped quietly: ``step_curl_kernels`` and
#: ``constitutive_kernels`` import CuPy at scope, so on a laptop there is no text to
#: splice and the emitter's own guard raises by name. On the validation host these
#: run, and the gate's ``transcription`` leg measures the same properties over the
#: whole emitted string.
_HAS_CERTIFIED_TEXT = not (fused_hd_pair.step_curl_kernels is None
                           or fused_hd_pair.constitutive_kernels is None)


def _cupy_or_skip() -> None:
    if not _HAS_CERTIFIED_TEXT:
        pytest.skip("[requires_resource][cuda-certified-device-text] "
                    "step_curl_kernels / constitutive_kernels import CuPy at module "
                    "scope, so the certified strings this weld splices are not "
                    "loadable here")


_emitter = pytest.mark.requires_resource("cuda-certified-device-text")


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_module_declares_the_seam_it_spans():
    assert family.SLOT == "update_H"
    assert family.REPLACES == ("update_H", "step_D")
    from .. import withdraw_hoist
    assert family.SEAM == withdraw_hoist.SEAM
    # Nothing is INJECTED between the two consults, so there is no deposit to bracket;
    # and the hoist is declined in this round, together with INSTALLABLE.
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.INSTALLABLE is False
    assert "slot arbitration" in family.INSTALLABLE_REASON


def test_the_kernel_is_in_exactly_one_half_of_the_partition():
    """And the dead half NAMES the gate that would move it."""
    assert family.KERNEL_NAME not in family.CERTIFIED_KERNELS
    assert family.KERNEL_NAME in family.UNCERTIFIED_KERNELS
    assert not set(family.CERTIFIED_KERNELS) & set(family.UNCERTIFIED_KERNELS)
    assert "gate_special_kz_fused_hd_pair.py" in \
        family.UNCERTIFIED_KERNELS[family.KERNEL_NAME]


def test_the_declaration_is_a_literal_this_files_own_readers_can_see():
    """``test_kernel_partition.py`` scans this file's TEXT for the declaration.

    The parameter list is derived from the released sibling's, so the NAME has to be
    spelled here as a literal or the package-wide partition walk cannot see the kernel
    this module ships.
    """
    text = (HERE / "special_kz_fused_hd_pair.py").read_text(encoding="utf-8")
    assert f'extern "C" __global__ void {family.KERNEL_NAME}(' in text
    # BOTH SYNTAX FORMS, as ``test_kernel_partition.partition_sets`` reads them: an
    # annotated assignment is an ``ast.AnnAssign``, and a reader that matched only
    # ``ast.Assign`` would miss ``CERTIFIED_KERNELS: Tuple[str, ...] = ()`` and report
    # half the partition as absent.
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

@_emitter
def test_the_transform_on_the_certified_text_is_the_released_transform():
    """THE EQUALITY THIS PRODUCT RESTS ON, byte for byte.

    Applied to the certified beta-FREE ``step_D_pml_real`` text, this family's curl
    transform must return exactly what the released real H->D weld's does. What the
    beta weld adds is then the beta text and the beta assertions, and nothing else.
    """
    _cupy_or_skip()
    from . import fused_hd_pair, step_curl_kernels

    mine = family.welded_curl_tail(
        step_curl_kernels._step_D_pml_real_kernel_code)  # noqa: SLF001
    assert mine == fused_hd_pair.welded_curl_tail()


def test_the_weld_redirects_every_magnetic_read_in_the_beta_body():
    head, tail = family.welded_curl_tail()
    assert tail.count("shift_dn_recompute(") == family.HALO_TAPS
    assert tail.count("own_h[") == family.OWN_LOAD_EDITS
    for target in family.H_TARGETS:
        assert f"{target}[" not in tail, (
            f"{target} is the PRE-LAUNCH magnetic field in this signature and every "
            f"read of it must be a register or a recompute")
    assert "shift_dn(" not in tail
    # The head is the certified preamble, the strides and the decode, kept verbatim.
    assert "int idx = blockIdx.x * blockDim.x + threadIdx.x;" in head
    assert "int i = idx / (ny * nz);" in head


def test_both_beta_partners_are_the_welds_own_register():
    """The clause this family owns and no sibling H->D weld has."""
    _head, tail = family.welded_curl_tail()
    beta_lines = [line.strip() for line in tail.splitlines()
                  if line.strip().startswith("curl = curl - (beta_")]
    assert len(beta_lines) == family.BETA_INSERT_LINES
    for line, register in zip(beta_lines, ("f2", "f1")):
        assert line.endswith(f"* {register});"), line
        assert f"float {register} = own_h[" in tail


def test_a_beta_partner_the_weld_did_not_redirect_is_refused():
    """The armed control for the clause above: it must BITE."""
    # THE SHIFTED OPERAND, which is the risk special_kz_curl names: "the register
    # must be the UNSHIFTED centre, never an sf/ss shifted operand". Welded, ``ss`` is
    # a recompute rather than a weld register, so the pairing check must refuse it.
    corrupted = family._certified_curl_source().replace(  # noqa: SLF001
        "        curl = curl - (beta_plus * f2);",
        "        curl = curl - (beta_plus * ss);", 1)
    with pytest.raises(AssertionError, match="beta partner"):
        family.welded_curl_tail(corrupted)


def test_a_beta_partner_declared_from_the_pre_launch_volume_is_refused():
    """The OTHER branch of the pairing check, driven on a synthetic block.

    The branch above fires when the beta line names a register the block does not
    declare; this one fires when it declares it from something that is not a weld
    register. Both are wrong answers rather than crashes, so both are controlled.
    """
    block = ("    // Target 0\n"
             "    {\n"
             "        float f1 = own_h[2];\n"
             "        float f2 = Hy[idx];\n"
             "        float curl = 0.0f;\n"
             "        curl = curl - (beta_plus * f2);\n"
             "    }\n")
    with pytest.raises(AssertionError, match="PRE-launch magnetic field"):
        family._assert_the_beta_partner_is_the_weld_register(block, True)  # noqa: SLF001


def test_a_body_that_lost_a_beta_increment_is_refused():
    corrupted = family._certified_curl_source().replace(  # noqa: SLF001
        "        curl = curl - (beta_plus * f2);\n", "", 1)
    with pytest.raises(AssertionError, match="beta increments"):
        family.welded_curl_tail(corrupted)


@_emitter
def test_the_certified_special_kz_body_is_the_certified_real_one_plus_the_insert():
    """WHY ONE TRANSFORM SERVES BOTH, measured rather than asserted.

    The two ``step_D`` texts differ in the kernel NAME, two parameters, two beta
    statements and three comments -- and in nothing structural. That is what lets this
    family reuse the released transform whole.
    """
    _cupy_or_skip()
    from . import step_curl_kernels

    prelude = special_kz_curl._REAL_PML_PRELUDE  # noqa: SLF001
    certified = step_curl_kernels._step_D_pml_real_kernel_code[len(prelude):]  # noqa: SLF001
    beta = family._certified_curl_source()[len(prelude):]  # noqa: SLF001
    def statements(text):
        return [line.strip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith("//")]
    added = [line for line in statements(beta) if line not in statements(certified)]
    assert sorted(added) == sorted([
        'extern "C" __global__ void step_D_special_kz_real(',
        "float beta_plus, float beta_minus",
        "curl = curl - (beta_plus * f2);",
        "curl = curl - (beta_minus * f1);",
        "int bc_x, int bc_y, int bc_z,",
    ]), added


# ---------------------------------------------------------------------------
# The measured absences
# ---------------------------------------------------------------------------

@_emitter
def test_the_emitted_text_carries_no_complex_type_and_no_float_divide():
    _cupy_or_skip()
    source = family.kernel_source()          # raises if either absence is violated
    assert "cf " not in source and "cf_load" not in source
    divides = [line for line in source.splitlines()
               if "/" in line.split("//", 1)[0]]
    assert len(divides) == family._PERMITTED_DIVIDES  # noqa: SLF001
    assert all(line.rstrip() in family._PERMITTED_DIVIDE_LINES  # noqa: SLF001
               for line in divides)


@pytest.mark.parametrize("planted,expected", [
    ("        float curl = ((sf - f1) + (f2 - ss)) / dtdx;", "divides outside"),
    ("    cf own_h[3];", "complex type"),
])
@_emitter
def test_the_absence_assertions_bite(planted, expected):
    """Each absence has a control that must be REFUSED -- a zero beside a bite."""
    _cupy_or_skip()
    source = family.kernel_source()
    needle = ("        float curl = dtdx * ((sf - f1) + (f2 - ss));"
              if "/" in planted else "    float own_h[3];")
    with pytest.raises(AssertionError, match=expected):
        family._assert_the_measured_absences(  # noqa: SLF001
            source.replace(needle, planted, 1))


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _Grid:
    beta = 0.0
    dimensions = 2
    cylindrical = False


class _Fields:
    pass


@pytest.mark.parametrize("fields,grid,names", [
    (_Fields(), _Grid(), "grid.beta is zero"),
    (type("F", (), {"force_complex_fields": True})(),
     type("G", (_Grid,), {"beta": 0.25})(), "complex64 storage with beta"),
    (type("F", (), {"force_complex_fields": False,
                    "has_offdiagonal_epsilon": True})(),
     type("G", (_Grid,), {"beta": 0.25})(), "off-diagonal epsilon"),
    (_Fields(), type("G", (_Grid,), {"beta": 0.25, "cylindrical": True})(),
     "cylindrical"),
    (_Fields(), type("G", (_Grid,), {"beta": 0.25, "dimensions": 3})(), "dimensions"),
])
def test_the_predicate_refuses_by_name(fields, grid, names):
    covered, reason = family.covers_special_kz_fused_hd_pair(fields, None, grid, ())
    assert covered is False
    assert names.lower() in reason.lower()


def test_an_undeclared_source_set_is_never_an_empty_one():
    """IGNORANCE IS NOT AN ABSENCE. ``Fields`` does not hold the source list."""
    from .. import withdraw_hoist

    reasons = withdraw_hoist.seam_withdraw_reasons(
        _Fields(), None, undeclared="the source set was not declared",
        refusal=lambda index, source: "standing",
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
    assert reasons and "not declared" in reasons[0]
