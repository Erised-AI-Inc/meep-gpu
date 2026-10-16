"""The folded fused DISPERSIVE Metal pair, as claims a laptop can check without a GPU.

The BYTES are the device gate's (``parity/meep_gpu/gate_metal_tranche7_fused_pairs.py``,
the ``folded_fused_dispersive_pair`` family row). What lives here is what the emitted
source says about the per-chunk forms this family takes from the array path:

1. the curl per component is LIFTED from the certified folded D curl — its own
   statement, absent-partner forms included, and exactly the operand loads it reads;
2. the curl forms are the certified ``p``/``n``/``v`` statements through
   :data:`~meep_gpu.metal_kernels.folded_fused_pair.LIFT_EDITS`, the same text the
   three-component pair emits, with ``fu`` touched only on the recurrence's cells;
3. the pole-aware E half selects per chunk — the accumulation where the
   component's own layer holds, the output select ``e = src`` elsewhere, ``f_w``
   untouched there — and a FAR ghost evaluates its own flag at its destination;
4. the chunk record rides in ``Params`` and costs no binding;
5. an admitted plan given no record is refused by name.
"""

from __future__ import annotations

import re

import pytest

from meep_gpu.metal_kernels import chunk_bounds, shaders, symmetry  # noqa: E402
from meep_gpu.metal_kernels import folded_fused_dispersive_pair as family  # noqa: E402
from meep_gpu.metal_kernels import folded_fused_pair as sibling  # noqa: E402
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
)

#: A folded PERIODIC Y axis between two periodic ones: component 1 (Dy) takes the FAR
#: ghost on y, components 0 and 2 the NEAR one — the matrix's 2-D folded shape.
CODES = (CODE_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)
PHASES = (0, 1, 0)
WALLS = (False, False, False)
POLES = 2


def source(target: int, codes=CODES, phases=PHASES, walls=WALLS,
           contract: str = shaders.CONTRACT_OFF) -> str:
    return family.folded_fused_dispersive_pair_source(
        codes, target, POLES, phases, walls, contract)


@pytest.mark.parametrize("target", (0, 1, 2))
def test_the_curl_is_the_certified_folded_d_curls_own_statement(target):
    """The statement and its loads are lines of the certified body, not a retyping,
    and every operand the statement names is loaded exactly once."""
    loads, curl = family.certified_target_curl(CODES, target)
    certified = symmetry.folded_curl_source(CODES, sibling.BACKWARD).splitlines()
    assert all(line in certified for line in curl + loads)
    assert curl[0].startswith(f"    float curl{target} = ")
    assert curl[-1].rstrip().endswith(";")
    text = " ".join(curl)
    named = set(re.findall(r"\b[abc](?:_[xyz])?\b", text))
    declared = {line.split()[1] for line in loads}
    assert named == declared, (named, declared)
    emitted = source(target)
    assert all(line in emitted for line in curl + loads)


@pytest.mark.parametrize("target", (0, 1, 2))
def test_the_curl_forms_are_the_three_component_pairs_text(target):
    """``p``/``n``/``v`` and the guarded ``fu`` store are the sibling's lifted lines,
    and the shared prologue never loads the ``f`` a source thread writes."""
    statements = sibling.certified_curl_statements(CODES)
    previous, stage, stepped = statements["recurrence"][target]
    text = source(target)
    assert previous in text and stage in text and stepped.strip() in text
    assert statements["aux_stores"][target] in text
    shared = text.split("    if (!(", 1)[0]
    assert f"f{target}[ii]" not in shared


def test_the_e_half_selects_per_chunk_and_the_far_ghost_keys_at_its_destination():
    text = source(1)
    assert "    const bool dw1 = in_h_y;" in text
    assert "float oprev = dw1 ? fw[ii] : 0.0f;" in text
    assert "if (dw1) { fw[ii] = osrc; }" in text
    assert "float oacc = osrc;" in text
    assert ("if (dw1) { oacc = e_out[ii] + ((kp1[j] * osrc) - (km1[j] * oprev)); }"
            in text)
    # the far ghost's own flag, at stored `last` on its own axis
    assert ("const bool in_h_y_gf = (((nyi - 1)) >= cbh_y_a0 && ((nyi - 1)) < "
            "cbh_y_a1) || (((nyi - 1)) >= cbh_y_b0 && ((nyi - 1)) < cbh_y_b1);") in text
    assert "if (in_h_y_gf) { fw[gf_i] = gf_src; }" in text
    # the pole chain stays the array path's, in pole order, before the product
    assert text.index("osource = osource - q0[ii];") < text.index(
        "osource = osource - q1[ii];") < text.index("float osrc = osource * ie1[ii];")


@pytest.mark.parametrize("target", (0, 1, 2))
def test_each_component_reads_its_two_record_entries(target):
    """The curl's dsigu axis on the D curl's lattice, the E half's own axis on its."""
    text = source(target)
    curl_half = shaders.CURL_HALF_INTEGER[sibling.BACKWARD]
    dsigu = shaders.CURL_CYCLE[target][1]
    du = chunk_bounds.membership_name(curl_half, dsigu)
    dw = chunk_bounds.membership_name(sibling.CONSTITUTIVE_HALF_INTEGER, target)
    assert f"    const bool du{target} = {du};" in text
    assert f"    const bool dw{target} = {dw};" in text
    assert text.count(chunk_bounds.BLOCK_BEGIN) == 1


def test_the_record_rides_in_params_and_costs_no_binding():
    dtype = family.params_record_dtype()
    assert dtype.itemsize == 132
    assert dtype.fields["cb"][1] == 32
    text = source(0)
    assert text.count(chunk_bounds.struct_source()) == 1
    assert chunk_bounds.params_alias() in text
    highest = max(int(chunk.split(")")[0]) for chunk in text.split("[[buffer(")[1:])
    assert highest + 1 == family.PACKED_BINDINGS


def test_every_folded_specialisation_emits():
    """Every folded quadruple, both modes, all three components."""
    import itertools

    count = 0
    for codes in itertools.product((CODE_PERIODIC, CODE_METALLIC,
                                    CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC),
                                   repeat=3):
        if not any(code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
                   for code in codes):
            continue
        phases = tuple(1 if code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
                       else 0 for code in codes)
        walls = tuple(code == CODE_METALLIC for code in codes)
        for target in range(3):
            assert source(target, codes, phases, walls)
            count += 1
    assert count == 56 * 3


def test_an_admitted_plan_without_a_record_is_refused_by_name(monkeypatch):
    from meep_gpu.chunk_map import ChunkLayoutRefusal
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML
    from meep_gpu.triton_kernels.coverage import Coverage

    grid = Grid(resolution=10.0, cell_size=(1.6, 3.0, 1.0), boundaries="metallic",
                symmetry=(Mirror("Y", 1),))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    monkeypatch.setattr(family, "metal_folded_fused_dispersive_pair_coverage",
                        lambda *args, **kwargs: Coverage(True, ()))
    with pytest.raises(ChunkLayoutRefusal, match="MetalFoldedFusedDispersivePairPlan"):
        family.plan_metal_folded_fused_dispersive_pair(
            fields, PML(grid=grid, thickness=0.2), (), None, chunk_bounds=None)
