"""The own-cell hoist on the Dcyl COMPLEX fused magnetic pair, read off the emitted text.

WHY A LAPTOP TEST AND NOT A GATE LEG. ``gate_cuda_fused_complex_pairs.py`` carries this
family's needles and its lift leg, and 15 of the 55 CUDA fingerprint entries pin that file:
a hoist check added there would drift every complex-family weld for a check that needs no
device. None of that gate's six needles on this family moves under the hoist (every site
keeps its live function, count and call sites), and its ``leg_lift`` passes unchanged, so the
hoist's own claims are asserted here instead:

1. twelve own-cell ``cf`` pre-loads -- w*, f* (H), fu_B*, B* at ``idx`` -- each exactly once,
   all of them BEFORE the kernel's first store (the first ``cf_store`` call in the entry
   body), and unguarded as the certified loads are (this family refuses every folded grid);
2. the three curl calls and the three constitutive calls take exactly their own component's
   preloaded words, in the certified argument order otherwise;
3. both derived helpers are declared once, and neither certified helper survives as dead
   text (``pml_apply_reg`` / ``constitutive_apply``): no call to either is left;
4. ``LIFT_EDITS`` is thirteen rows, the last five being the hoist's.

The derived helpers' ARITHMETIC is asserted at emit time by the module itself
(``_hoist_derive``: the derived statements equal the certified ones under the
argument-to-load map); a certified edit that moves it refuses there by name.
"""
from __future__ import annotations

import re

import pytest

from . import cylindrical_fused_magnetic_pair as family
from . import complex_emitter

pytestmark = pytest.mark.skipif(
    family.cylindrical_complex_kernels is None,
    reason="the certified cylindrical text imports CuPy at module scope")

ARMS = sorted(complex_emitter.EXPANSIONS)


def _entry_body(source: str) -> str:
    return source.split("extern \"C\" __global__ void fused_magnetic_pair_pml_cyl_complex(",
                        1)[1].split("\n) {\n", 1)[1]


@pytest.mark.parametrize("arm", ARMS)
def test_the_twelve_own_cell_loads_precede_the_first_store(arm):
    body = _entry_body(family.cylindrical_fused_magnetic_pair_source(arm))
    first_store = min(body.index(call) for call in (
        "pml_apply_reg_pre(", "cf_store(", "constitutive_apply_pre("))
    expected = []
    for register, target in zip(("0", "1", "2"), ("Bx", "By", "Bz")):
        expected += [f"cf pre_w_{register} = cf_load(w{register}, idx);",
                     f"cf pre_h_{register} = cf_load(f{register}, idx);",
                     f"cf pre_fu_{register} = cf_load(fu_{target}, idx);",
                     f"cf pre_b_{register} = cf_load({target}, idx);"]
    loads = [line.strip() for line in body.splitlines()
             if line.strip().startswith("cf pre_")]
    assert sorted(loads) == sorted(expected)
    for line in expected:
        assert body.count(line) == 1
        assert body.index(line) < first_store, line


@pytest.mark.parametrize("arm", ARMS)
def test_every_call_takes_its_own_components_words(arm):
    source = family.cylindrical_fused_magnetic_pair_source(arm)
    for register, target in zip(("0", "1", "2"), ("Bx", "By", "Bz")):
        curl = [line for line in source.splitlines()
                if f"b{register} = pml_apply_reg_pre({target}, fu_{target}, idx, curl, " in line]
        assert len(curl) == 1 and curl[0].rstrip().endswith(
            f", pre_fu_{register}, pre_b_{register});"), curl
        const = [line for line in source.splitlines()
                 if f"constitutive_apply_pre(f{register}, w{register}, idx, s{register}, " in line]
        assert len(const) == 1 and (
            f"idx, s{register}, pre_w_{register}, pre_h_{register}, " in const[0]), const


@pytest.mark.parametrize("arm", ARMS)
def test_the_derived_helpers_replace_the_certified_ones(arm):
    prelude = family.cylindrical_fused_magnetic_pair_prelude(arm)
    assert prelude.count("cf pml_apply_reg_pre(\n") == 1
    assert prelude.count("void constitutive_apply_pre(\n") == 1
    source = family.cylindrical_fused_magnetic_pair_source(arm)
    assert not re.search(r"\bpml_apply_reg\(", source)
    assert not re.search(r"\bconstitutive_apply\(", source)
    assert not re.search(r"\bpml_apply\(", source)


def test_the_lift_is_thirteen_edits():
    assert len(family.LIFT_EDITS) == 13
    assert all(set(edit) == {"line", "became", "why"} for edit in family.LIFT_EDITS)
    assert "THE OWN-CELL HOIST" in family.LIFT_EDITS[8]["why"]
