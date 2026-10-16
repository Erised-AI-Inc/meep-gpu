"""The own-cell hoist on the Dcyl COMPLEX fused electric pair, read off the emitted text.

WHY A LAPTOP TEST AND NOT A GATE LEG. ``gate_cuda_fused_complex_pairs.py`` carries this
family's needles and its lift leg; a hoist check added there would drift every
complex-family weld for a check that needs no device, so the hoist's own claims are asserted
here instead:

1. twelve own-cell ``cf`` pre-loads -- w*, f* (E), fu_D*, D* at ``idx`` -- each
   exactly once, all of them BEFORE the kernel's first store, each behind the certified
   load's own guard: ``f_w`` and the old field behind the component's dsigw flag (the one
   form that reads them), ``fu`` behind its dsigu flag, the flux density unguarded (every
   curl form reads it) (bit_parity_0_10 DESIGN.md D11);
2. the three curl calls and the three constitutive calls take exactly their own component's
   preloaded words, in the certified argument order otherwise;
3. the three derived helpers are declared once (``pml_apply_reg_pre``, ``fold_apply_reg``,
   ``constitutive_apply_pre``), and no certified helper survives as dead text: no call to
   ``pml_apply``, ``fold_apply``, ``pml_apply_reg`` or ``constitutive_apply`` is left;
4. ``LIFT_EDITS`` names the hoist.

The i*m/r pass's ladder reads the target and its ``fu`` back after the curl form stored them
(``fold_apply_reg``): reloads of the lane's own writes, which
``test_own_cell_hoist_store_sequence`` pins by count. The derived helpers' ARITHMETIC is
asserted at emit time by the module itself (``cylindrical_fused_magnetic_pair.derive``: the
derived statements equal the certified ones under the argument-to-load map).
"""
from __future__ import annotations

import re


from . import complex_emitter
from . import cylindrical_fused_electric_pair as family

TARGETS = ("Dx", "Dy", "Dz")
CONSTITUTIVE_FLAGS = complex_emitter.CONSTITUTIVE_FLAGS["E"]
CURL_FLAGS = complex_emitter.CURL_FLAGS[True]


def _entry_body(source: str) -> str:
    return source.split('extern "C" __global__ void fused_electric_pair_pml_cyl_complex(',
                        1)[1].split("\n) {\n", 1)[1]


def test_the_twelve_own_cell_loads_precede_the_first_store():
    body = _entry_body(family.cylindrical_fused_electric_pair_source())
    first_store = min(body.index(call) for call in (
        "pml_apply_reg_pre(", "cf_store(", "constitutive_apply_pre(", "fold_apply_reg("))
    expected = []
    for register, target, cflag, uflag in zip(("0", "1", "2"), TARGETS, CONSTITUTIVE_FLAGS,
                                              CURL_FLAGS):
        expected += [f"cf pre_w_{register} = {cflag} ? cf_load(w{register}, idx) : cf_zero();",
                     f"cf pre_e_{register} = {cflag} ? cf_load(f{register}, idx) : cf_zero();",
                     f"cf pre_fu_{register} = {uflag} ? cf_load(fu_{target}, idx) : cf_zero();",
                     f"cf pre_d_{register} = cf_load({target}, idx);"]
    loads = [line.strip() for line in body.splitlines()
             if line.strip().startswith("cf pre_")]
    assert sorted(loads) == sorted(expected)
    for line in expected:
        assert body.count(line) == 1
        assert body.index(line) < first_store, line


def test_every_call_takes_its_own_components_words():
    source = family.cylindrical_fused_electric_pair_source()
    for register, target in zip(("0", "1", "2"), TARGETS):
        curl = [line for line in source.splitlines()
                if f"d{register} = pml_apply_reg_pre({target}, fu_{target}, idx, curl, " in line]
        assert len(curl) == 1 and curl[0].rstrip().endswith(
            f", pre_fu_{register}, pre_d_{register});"), curl
        const = [line for line in source.splitlines()
                 if f"constitutive_apply_pre(f{register}, w{register}, idx, s{register}, " in line]
        assert len(const) == 1 and (
            f"idx, s{register}, pre_w_{register}, pre_e_{register}, " in const[0]), const


def test_the_derived_helpers_replace_the_certified_ones():
    prelude = family.cylindrical_fused_electric_pair_prelude()
    assert prelude.count("cf pml_apply_reg_pre(\n") == 1
    assert prelude.count("cf fold_apply_reg(\n") == 1
    assert prelude.count("void constitutive_apply_pre(\n") == 1
    source = family.cylindrical_fused_electric_pair_source()
    for helper in ("pml_apply_reg", "constitutive_apply", "pml_apply", "fold_apply"):
        assert not re.search(rf"\b{helper}\(", source), helper


def test_the_lift_names_the_hoist():
    assert all(set(edit) == {"line", "became", "why"} for edit in family.LIFT_EDITS)
    assert sum("THE OWN-CELL HOIST" in edit["why"] for edit in family.LIFT_EDITS) == 1
