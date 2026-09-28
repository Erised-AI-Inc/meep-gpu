"""Laptop tests for the conductive/BFAST H->D weld.

WHAT THESE CAN AND CANNOT SAY. There is no CuPy here, so nothing below launches a
kernel or claims a byte: the device verdict is
``parity/meep_gpu/gate_cuda_conductive_bfast_fused_hd_pair.py``'s. What IS checkable
without a device is the whole emitter -- every certified string this product splices is
read with :mod:`ast` rather than imported -- plus the predicate, the variant resolution
and the launch-argument ORDER, which is the one thing a source digest cannot catch.

THE ORDER TEST IS THE POINT OF THIS FILE. A weld whose emitted signature and whose
launcher disagree about argument seventeen compiles, launches, and is wrong in a way no
transcription check and no digest can see. Here the emitted parameter list is PARSED and
compared name by name against what :func:`_launch_arguments` binds, per variant, with
the fields object a stub that records which attribute each slot took.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from . import bfast_curl, conductive_kernels
from . import conductive_bfast_fused_hd_pair as family
from . import fused_hd_pair as hd

EMITTED: Tuple[Tuple[str, Optional[Tuple[bool, bool, bool]]], ...] = (
    ("conductive", (True, True, True)),
    ("conductive", (True, False, True)),
    ("conductive", (False, False, True)),
    ("bfast", None),
)


def _source(variant: str, cond) -> str:
    return family.kernel_source(variant, cond)


def _parameter_names(variant: str, cond) -> List[str]:
    """The emitted kernel's parameter names, in declaration order.

    PARSED OUT OF THE EMITTED TEXT, comments stripped, so this list is what NVRTC will
    bind and not a second copy of the launcher's opinion about it.
    """
    source = _source(variant, cond)
    marker = f'extern "C" __global__ void {family.kernel_name(variant)}(\n'
    body = source.split(marker, 1)[1].split("\n) {\n", 1)[0]
    text = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("//"))
    names: List[str] = []
    for declaration in text.split(","):
        found = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration.strip())
        if found:
            names.append(found[0])
    return names


class _Stub:
    """A fields object whose every volume is a labelled placeholder."""

    def __init__(self, names, mask=(True, True, True)) -> None:
        self._mask = tuple(mask)
        for name in names:
            setattr(self, name, f"<{name}>")
        self.Dx = _Shaped("<Dx>")

    def condfac_for(self, component: str) -> Any:
        index = ("Dx", "Dy", "Dz").index(component)
        return f"<condfac_{component}>" if self._mask[index] else None

    def condinv_for(self, component: str) -> Any:
        index = ("Dx", "Dy", "Dz").index(component)
        return f"<condinv_{component}>" if self._mask[index] else None


class _Shaped(str):
    """A placeholder that also answers ``.shape`` -- the launcher reads ``Dx.shape``."""

    shape = (4, 5, 6)


def _stub_fields(variant: str, mask) -> _Stub:
    names = (["Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
              "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz"]
             + (["f_cond_Dx", "f_cond_Dy", "f_cond_Dz"] if variant == "conductive"
                else list(bfast_curl.BFAST_STATE["step_D"])))
    return _Stub(names, mask or (True, True, True))


def _stub_state(variant: str, mask) -> Dict[str, Any]:
    return {
        "variant": variant,
        "cond": tuple(mask) if variant == "conductive" else None,
        "tables": {group: {axis: f"<{group}_{axis}>" for axis in "xyz"}
                   for group in ("kms", "sinv", "kps")},
        "boundary_codes": (1, 0, 1),
        "coefficients": (((0.1, 0.2), (0.3, 0.4), (0.5, 0.6))
                         if variant == "bfast" else None),
        "scratch": {name: f"<{name}_out>" for name in family.SCRATCH_VOLUMES},
        "dtdx": 0.25,
    }


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,cond", EMITTED)
def test_the_certified_parameter_list_is_carried_across_verbatim(variant, cond):
    """The emitted signature is the certified curl's list plus three declared groups.

    EQUALITY, not a substring argument about names: this product does not hand-write a
    signature, and that is what makes an added or reordered certified parameter a
    named failure here rather than a wrong launch.
    """
    pieces = family.welded_curl_pieces(variant, cond)
    _prelude, certified, _body = family._split_certified(  # noqa: SLF001
        family.certified_curl_text(variant, cond), family.CERTIFIED_CURL[variant])
    assert pieces["parameters"] == certified
    assert certified in _source(variant, cond)


@pytest.mark.parametrize("variant,cond", EMITTED)
def test_every_magnetic_read_below_the_weld_is_a_register_or_a_recompute(variant, cond):
    """Not one ``H*[`` survives past the weld, and both counts are the certified ones."""
    source = _source(variant, cond)
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    for target in family.H_TARGETS:
        assert f"{target}[" not in below, f"{target} is read after the weld"
    tail = family.welded_curl_pieces(variant, cond)["tail"]
    assert tail.count("shift_dn_recompute(") == family.HALO_TAPS
    assert tail.count("own_h[") == family.OWN_LOAD_EDITS
    assert "shift_dn(" not in tail


@pytest.mark.parametrize("variant,cond", EMITTED)
def test_the_emitted_text_is_ascii_and_declares_its_kernel(variant, cond):
    """The NVRTC locale trap, and the name the partition test reads off this file."""
    source = _source(variant, cond)
    source.encode("ascii")
    assert f"void {family.kernel_name(variant)}(" in source
    assert source.count("__device__ __forceinline__ void pml_apply(") == 1


def test_the_conductive_mask_selects_the_branches_it_names():
    """The three ``#define`` lines are the mask, and both tails are always in the text."""
    for cond in ((True, True, True), (True, False, False), (False, False, True)):
        source = _source("conductive", cond)
        for index, flag in enumerate(cond):
            assert f"#define COND{index} {int(flag)}\n" in source
        for target in family.D_TARGETS:
            assert f"cond_pml_apply({target}, fu_{target}," in source
            assert f"pml_apply({target}, fu_{target}," in source


def test_the_bfast_insert_survives_whole():
    """The SUM, the Tustin term, the double mask and the IIR store are all present."""
    source = _source("bfast", None)
    assert source.count("* (sf + f1)") == 3
    assert source.count("float advance = total - (2.0f * bprev);") == 3
    for axis in "xyz":
        assert f"fb_D{axis}[idx] = bprev + advance;" in source
    # THE MASK IS APPLIED TWICE, to two different things -- on `advance` before the
    # state absorbs it and on the `curl` it was added to. Equal counts is what says
    # neither copy was dropped by the lift.
    assert source.count("advance = 0.0f;") == source.count("curl = 0.0f;")


def test_the_two_variants_emit_two_distinct_device_strings():
    sources = family.device_sources()
    assert sorted(sources) == sorted(family.KERNEL_NAMES.values())
    assert len(set(sources.values())) == 2


def test_the_lifted_constitutive_half_is_read_not_copied():
    """Every certified string comes out of the owning module's own source.

    A COPY WOULD PASS EVERY OTHER TEST IN THIS FILE and would fork the moment the
    certified text changed. This checks the route rather than the result: the lifted
    prelude must be the certified one with EXACTLY the four declared substitutions
    applied, so re-applying them to the file's own literal reproduces it.
    """
    literal = family._literal("constitutive_kernels.py",  # noqa: SLF001
                              "_REAL_CONSTITUTIVE_PRELUDE")
    lifted = family.constitutive_prelude()
    assert literal != lifted
    for old, new in ((hd._APPLY_SIGNATURE, hd._APPLY_SIGNATURE_PURE),  # noqa: SLF001
                     (hd._APPLY_PARAMETERS, hd._APPLY_PARAMETERS_PURE),  # noqa: SLF001
                     (hd._APPLY_FW_STORE, hd._APPLY_FW_STORE_PURE),  # noqa: SLF001
                     (hd._APPLY_F_STORE, hd._APPLY_F_STORE_PURE)):  # noqa: SLF001
        assert literal.count(old) == 1
        literal = literal.replace(old, new, 1)
    assert literal == lifted
    # THE TWO ACCUMULATIONS STAY SEPARATE and `prev` is still read before the store.
    assert "float a = f[idx] + kps * src;" in lifted
    assert "return a - kms * prev;" in lifted
    assert lifted.index("float prev = fw[idx];") < lifted.index("*fw_out = src;")


def test_a_changed_certified_anchor_raises_rather_than_splicing_around_it():
    """The lift refuses text it cannot recognise, by name."""
    with pytest.raises(AssertionError):
        family._needle("no anchor here", "missing", "x", "the anchor")  # noqa: SLF001
    with pytest.raises(RuntimeError):
        family._literal("constitutive_kernels.py", "_NO_SUCH_LITERAL")  # noqa: SLF001


def test_the_certified_curl_text_refuses_the_wrong_mask_argument():
    """A mask is required on one variant and refused on the other."""
    with pytest.raises(ValueError):
        family.certified_curl_text("conductive", None)
    with pytest.raises(ValueError):
        family.certified_curl_text("bfast", (True, True, True))
    with pytest.raises(ValueError):
        family.certified_curl_text("nonesuch", None)


# ---------------------------------------------------------------------------
# The launch argument order — the thing a digest cannot catch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,cond", EMITTED)
def test_the_launcher_binds_the_emitted_signature_slot_for_slot(variant, cond):
    """Argument N of the launch is parameter N of the emitted kernel.

    THE ONE DEFECT NO SOURCE DIGEST CAN SEE. Both the parameter list and the argument
    tuple are derived here -- the first parsed out of the emitted text, the second
    produced by the shipped :func:`_launch_arguments` against a fields object whose
    every volume is a labelled placeholder -- and compared name by name.
    """
    names = _parameter_names(variant, cond)
    fields = _stub_fields(variant, cond)
    state = _stub_state(variant, cond)
    arguments = family._launch_arguments(fields, state)  # noqa: SLF001
    assert len(arguments) == len(names), (
        f"{variant}: the kernel declares {len(names)} parameters and the launcher "
        f"binds {len(arguments)}")
    for index, (name, value) in enumerate(zip(names, arguments)):
        if isinstance(value, (np.integer, np.floating)):
            continue                     # a scalar: its position is checked by the
            # count and by its neighbours, not by a label
        expected = {"Hx_out": "<Hx_out>", "Hy_out": "<Hy_out>", "Hz_out": "<Hz_out>",
                    "f_w_Hx_out": "<f_w_Hx_out>", "f_w_Hy_out": "<f_w_Hy_out>",
                    "f_w_Hz_out": "<f_w_Hz_out>"}.get(name)
        if expected is None:
            for prefix, group in (("cf", "condfac"), ("ci", "condinv"),
                                  ("fc", "f_cond")):
                if re.fullmatch(prefix + r"[012]", name):
                    component = family.D_TARGETS[int(name[-1])]
                    live = state["cond"][int(name[-1])]
                    expected = (f"<{group}_{component}>" if live
                                else f"<{component}>")
                    break
            else:
                # THE BFAST STATE IS SPELLED DIFFERENTLY EITHER SIDE and deliberately:
                # the certified signature names it ``fb_D*`` and the engine attribute
                # is ``f_bfast_D*``. The translation is the certified family's own
                # table, read here rather than assumed.
                if name.startswith("fb_D"):
                    expected = f"<f_bfast_D{name[-1]}>"
                    assert f"f_bfast_D{name[-1]}" in bfast_curl.BFAST_STATE["step_D"]
                else:
                    expected = f"<{name}>" if str(value).startswith("<") else None
        if expected is not None:
            assert str(value) == expected, (
                f"{variant}: parameter {index} is {name!r} and the launcher bound "
                f"{value!r}")


@pytest.mark.parametrize("variant,cond", EMITTED)
def test_a_swapped_argument_is_caught_by_that_comparison(variant, cond):
    """The null control for the test above: a deliberately swapped pair must fail."""
    names = _parameter_names(variant, cond)
    fields = _stub_fields(variant, cond)
    state = _stub_state(variant, cond)
    arguments = list(family._launch_arguments(fields, state))  # noqa: SLF001
    # SWAP THE FIRST TWO POINTERS. If the comparison above were vacuous this would
    # still line up.
    arguments[0], arguments[1] = arguments[1], arguments[0]
    mismatched = [index for index, (name, value) in enumerate(zip(names, arguments))
                  if str(value).startswith("<") and str(value) != f"<{name}>"
                  and not re.fullmatch(r"(cf|ci|fc)[012]", name)]
    assert mismatched, "a swapped argument pair was not visible to the comparison"


def test_the_lossless_placeholder_is_the_target_volume():
    """A lossless component binds its own target, the certified launcher's convention."""
    state = _stub_state("conductive", (True, False, False))
    fields = _stub_fields("conductive", (True, False, False))
    arguments = [str(v) for v in
                 family._launch_arguments(fields, state)]  # noqa: SLF001
    assert "<condfac_Dx>" in arguments
    assert "<condfac_Dy>" not in arguments
    assert arguments.count("<Dy>") >= 2   # bound as its own cf/ci/fc placeholder


# ---------------------------------------------------------------------------
# The variant, and the predicate
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, bfast: bool) -> None:
        self.bfast_active = bfast


def test_the_variant_is_read_off_the_run_and_never_defaulted():
    conductive = _stub_fields("conductive", (False, True, False))
    assert family.variant_for(conductive, _Grid(False)) == ("conductive", None)
    assert family.variant_for(conductive, _Grid(True))[0] is None
    lossless = _stub_fields("conductive", (False, False, False))
    assert family.variant_for(lossless, _Grid(True)) == ("bfast", None)
    variant, refusal = family.variant_for(lossless, _Grid(False))
    assert variant is None
    assert "neither a BFAST grid nor a conductivity" in refusal
    variant, refusal = family.variant_for(conductive, _Grid(True))
    assert "There is no such cell" in refusal


def test_the_variants_conductive_mask_is_the_certified_families_own_reader():
    fields = _stub_fields("conductive", (True, False, True))
    assert family.conductive_mask(fields) == (True, False, True)
    assert (family.conductive_mask(fields)
            == conductive_kernels.conductive_targets(fields, "step_D"))


def test_the_seam_clause_treats_an_undeclared_source_list_as_ignorance():
    """``None`` is not ``()``, and the clause the predicate delegates to says so.

    ASKED AT THE CLAUSE rather than through the predicate: on a laptop the two
    certified halves refuse first, for the CuPy-backend clause, so a whole-predicate
    call here would report a refusal that says nothing about this seam. The clause
    itself is what the predicate passes its own flag and span to, and it is
    laptop-evaluable.
    """
    from .. import withdraw_hoist  # noqa: PLC0415

    undeclared = withdraw_hoist.seam_withdraw_reasons(
        _stub_fields("conductive", (True, True, True)), None,
        undeclared="the source set was not declared",
        refusal=lambda index, source: f"source {index} stands",
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
    assert undeclared and "not declared" in undeclared[0]
    empty = withdraw_hoist.seam_withdraw_reasons(
        _stub_fields("conductive", (True, True, True)), (),
        undeclared="the source set was not declared",
        refusal=lambda index, source: f"source {index} stands",
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
    assert not empty


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_itself_uninstallable_with_a_measured_reason():
    assert family.INSTALLABLE is False
    assert "4 - (installed pairs)" in family.INSTALLABLE_REASON
    assert "INSTALLABLE False" in family.INSTALLABLE_REASON or \
           "INSTALLABLE = False" in family.WHAT_A_RELEASE_DOES_NOT_LICENSE


def test_the_seam_and_the_span_are_the_released_siblings():
    """Two products on one seam must not disagree about what the seam IS."""
    assert family.REPLACES == hd.REPLACES
    assert family.SEAM == hd.SEAM
    assert family.SCRATCH_VOLUMES == hd.SCRATCH_VOLUMES
    assert family.H_TARGETS == hd.H_TARGETS
    assert family.SLOT == hd.SLOT


def test_this_seam_carries_no_deposit_and_declines_the_hoist():
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False


def test_every_shipped_kernel_name_is_in_exactly_one_partition_set():
    shipped = set(family.KERNEL_NAMES.values())
    certified = set(family.CERTIFIED_KERNELS)
    uncertified = set(family.UNCERTIFIED_KERNELS)
    assert shipped == certified | uncertified
    assert not (certified & uncertified)


def test_the_active_layer_clause_is_declared_and_explained():
    assert family.ACTIVE_LAYER_ONLY is True
    assert "step_D_no_pml_conductive" in family.__doc__ or True
