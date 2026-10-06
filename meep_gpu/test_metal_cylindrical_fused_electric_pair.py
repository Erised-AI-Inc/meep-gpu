"""The Metal Dcyl complex-storage fused ELECTRIC weld (m = 0 and |m| >= 1), as
claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_cylindrical_fused_electric_pair.py`` steps
engines side by side for twelve complete steps and compares uint32 words. No assertion
here duplicates that. What lives here is everything true about this family WITHOUT a
device.

THE CELL IS THE LARGEST ONE ON THE METAL BOARD — 16 reachable D->E seam-instances,
more than any other gap — and until this round it carried the verdict ``UNFUSABLE ON
METAL``: 33 pointers against a 30-pointer ceiling, over by THREE. Its magnetic twin
sits at EXACTLY 31 bindings with zero headroom, and the E side's three inverse-epsilon
volumes are what push it over. What cleared it is
:mod:`meep_gpu.metal_kernels.coefficient_pack` — the SAME technique, applied to the
same group, that cleared the m = 0 sibling's over-by-one. One tool, two cells.
"""

from __future__ import annotations

import re as _re

import ast
import json
import pathlib

import numpy as np
import pytest

from .device_identity import weld_survives_edit

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = (REPO / "parity" / "meep_gpu"
        / "gate_metal_cylindrical_fused_electric_pair.py")
#: REPOINTED 2026-09-04 from ``..._2026-08-31_pack``: the ``M_ZERO`` arm changed the
#: emitted device source (a third arm, a ninth ``Params`` member), so the earlier
#: artifact no longer describes the bytes that ship. The ``_pack`` artifact stays
#: where it is as the evidence for what the coefficient-pack cut credited.
#: REPOINTED 2026-09-25 to the ``_2026-09-25_night`` fleet's re-run (released, PASS, 253
#: recorded digests, ZERO disagreeing with the tree, checked before this line moved): the
#: citation re-point edited this module's comments, and the 09-04 artifact records no
#: device or code digest ``weld_survives_edit`` could clear that edit through.
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_cylindrical_fused_electric_pair_2026-10-02_arch3" / "gate.json")
CENSUS = (REPO / "parity" / "meep_gpu" / "results"
          / "metal_coverage_tranche6_2026-08-19")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, coefficient_pack,
    complex_fused_electric_pair as cartesian,
    cylindrical_complex,
    cylindrical_fused_electric_pair as family,
    cylindrical_fused_magnetic_pair as twin,
    launch as metal_launch,
    registry, templates,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402

#: The arm every emitted-source assertion below specialises at. ``FMA_V1`` is the
#: expansion the shipped probe classifies on this host; the SOURCE assertions are
#: arm-independent (they compare emitted text against the same emitters at the same
#: arm), so pinning one here keeps them deterministic without a device.
EXPANSION = "FMA_V1"
WALLED_Z = (False, False, True)
UNWALLED = (False, False, False)


def emit(bcz=templates.METALLIC, arm=cylindrical_complex.M_ONE, walls=WALLED_Z):
    return family.cylindrical_fused_electric_pair_source(bcz, arm, walls, EXPANSION)


def census_rows():
    def load(name):
        path = CENSUS / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load("tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def covered(row, key):
    entry = row["predicates"].get(key, {})
    return bool(entry.get("covered_modulo_backend", entry.get("covered")))


# ---------------------------------------------------------------------------
# What the corpus says this is worth
# ---------------------------------------------------------------------------

def test_the_cell_is_sixteen_seam_instances_and_every_one_carries_a_deposit():
    """16 -> 16 -> 16, and the third step is the one the flag pays for."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    curl = [r for r in rows if covered(r, "cylindrical_complex_pml_curl@step_D")]
    both = [r for r in curl
            if covered(r, "cylindrical_complex_constitutive@update_E")]
    assert (len(curl), len(both)) == (16, 16)
    for row in both:
        kinds = row["configuration"]["source_field_types"] or []
        # EVERY source on every row is ELECTRIC, and one row carries TWO of them
        # (examples:cylinder_cross_section.py) — so the assertion is per source
        # rather than on the list, which would read as "one source" and be wrong.
        assert kinds and all(str(kind) == "D" for kind in kinds), row["row"]
    assert family.CARRIES_DEPOSIT_REPAIR is True


def test_it_is_the_largest_cell_on_the_board_and_disjoint_from_the_m0_sibling():
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    complex_arm = {f"{r['leg']}:{r['row']}" for r in rows
                   if covered(r, "cylindrical_complex_pml_curl@step_D")}
    real = {f"{r['leg']}:{r['row']}" for r in rows
            if covered(r, "cylindrical_real_curl@step_D")}
    assert len(complex_arm) == 16 and len(real) == 3
    assert complex_arm & real == set()


# ---------------------------------------------------------------------------
# The construction
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_emitters_body_on_both_sides_of_the_splice():
    source = emit()
    curl = family.certified_cylindrical_curl_body(
        templates.METALLIC, cylindrical_complex.M_ONE, EXPANSION)
    head, tail = curl.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_curl_half_is_the_BACKWARD_arm_and_never_the_magnetic_one():
    """``backward`` is pinned True: ``step_B``'s forward strides are the twin's."""
    d_side = family.certified_cylindrical_curl_body(
        templates.METALLIC, cylindrical_complex.M_ONE, EXPANSION)
    b_side = twin.certified_cylindrical_curl_body(
        templates.METALLIC, cylindrical_complex.M_ONE, EXPANSION)
    assert d_side != b_side
    assert d_side == cylindrical_complex.cylindrical_curl_source(
        templates.METALLIC, True, cylindrical_complex.M_ONE, EXPANSION
    ).split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1][: -len("}\n")]


def test_the_constitutive_half_differs_in_exactly_the_three_seam_lines():
    source = emit()
    certified = cartesian.certified_constitutive_body(EXPANSION).splitlines()
    marker = "    // --- update_E (stepping.update_E"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    while spliced and (spliced[0].endswith("--")
                       or spliced[0].strip().startswith("//")):
        spliced.pop(0)
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(certified) == len(spliced)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [
        (f"    float2 src{t} = c_mul_field_left(g{t}[ii], e{t}[ii]);",
         f"    float2 src{t} = c_mul_field_left(v{t}, e{t}[ii]);") for t in range(3)]


def test_the_electric_half_reads_no_g_pointer_at_all():
    source = emit()
    electric = source.split("// --- update_E (stepping.update_E", 1)[1]
    for target in range(3):
        assert f"g{target}[" not in electric


def test_the_wall_clear_is_the_D_off_diagonal_and_only_z_can_be_walled():
    walled, periodic = emit(), emit(templates.PERIODIC, walls=UNWALLED)
    assert "    v0 = at_z ? " in walled and "    v1 = at_z ? " in walled
    assert "    v2 = at_z ? " not in walled  # Dz is the DIAGONAL one
    assert "no walled axis clears a D component" in periodic
    for axis in family.FORBIDDEN_WALL_AXES:
        walls = [False, False, False]
        walls[axis] = True
        with pytest.raises(ValueError, match="reported walled"):
            emit(walls=tuple(walls))


def test_the_wall_clear_sits_before_the_store_and_the_store_before_the_seam():
    source = emit()
    clear = source.index("    v0 = at_z ? ")
    store = source.index(family._CURL_STORE)
    seam = source.index("float2 src0 = c_mul_field_left(v0, e0[ii]);")
    assert clear < store < seam


# ---------------------------------------------------------------------------
# The signature, and the pack that makes it fit
# ---------------------------------------------------------------------------

def test_the_binding_count_the_module_spells_is_the_count_its_emitter_produces():
    source = emit()
    assert source.count("[[buffer(") == family.PACKED_BINDINGS
    pointers = [line for line in source.splitlines()
                if "[[buffer(" in line and "device" in line]
    scalars = [line for line in source.splitlines()
               if "[[buffer(" in line and "constant" in line]
    assert len(pointers) == family.PACKED_POINTERS == 28
    assert len(scalars) == 1
    assert family.PACKED_BINDINGS == family.PACKED_POINTERS + 1 <= MAX_BUFFER_BINDINGS


def test_every_specialisation_a_shipped_plan_can_emit_binds_the_same_count():
    """SIX sources — two z terminations x three m arms — and the count is a property
    of the SIGNATURE, so a specialisation that changed it would be a different kernel
    the module's own arithmetic guard never saw."""
    sources = family.enumerate_sources(EXPANSION)
    assert sorted(sources) == ["metallic_z:m0", "metallic_z:m1", "metallic_z:mmany",
                               "periodic_z:m0", "periodic_z:m1", "periodic_z:mmany"]
    assert len(set(sources.values())) == 6
    for label, text in sources.items():
        assert text.count("[[buffer(") == family.PACKED_BINDINGS, label


def test_the_unpacked_shape_is_over_the_ceiling_by_EXACTLY_THREE():
    """THE MEASUREMENT THE PACK EXISTS FOR, taken off the refuted emitters' own text.

    33 pointers plus one packed ``Params&`` is 34 bindings against a ceiling of 31.
    That is the board's ``UNFUSABLE ON METAL`` verdict for this cell restated as
    arithmetic; the gate's ``binding_ceiling`` leg turns it into a COMPILE.
    """
    refuted = family.refuted_unpacked_pointer_source()
    assert refuted.count("[[buffer(") == family.UNPACKED_POINTER_BINDINGS == 34
    assert family.UNPACKED_POINTERS == 33
    assert family.UNPACKED_POINTER_BINDINGS == MAX_BUFFER_BINDINGS + 3
    assert (family.refuted_separate_scalar_source().count("[[buffer(")
            == family.SEPARATE_SCALAR_BINDINGS > family.UNPACKED_POINTER_BINDINGS)
    assert (family.refuted_thirty_second_binding().count("[[buffer(")
            == family.OVER_CEILING_BINDINGS == MAX_BUFFER_BINDINGS + 1)


def test_the_pack_saves_five_pointers_and_the_cell_needed_three():
    """The margin is deliberate rather than lucky. The unit packed is the curl half's
    per-axis coefficient GROUP — six vectors in, one pointer out — not a subset chosen
    to land on the ceiling."""
    assert family.UNPACKED_POINTERS - family.PACKED_POINTERS == 5
    assert len(family.PACKED_VECTORS) == 6
    assert family.UNPACKED_POINTERS - (MAX_BUFFER_BINDINGS - 1) == 3


def test_the_same_pack_clears_the_m0_sibling_which_was_over_by_one():
    """ONE TECHNIQUE, TWO CELLS — which is what makes packing a tool here rather than
    a rescue for one signature."""
    from meep_gpu.metal_kernels import cylindrical_real_fused_electric_pair as m0

    assert m0.PACKED_VECTORS == family.PACKED_VECTORS
    assert m0.UNPACKED_POINTERS - (MAX_BUFFER_BINDINGS - 1) == 1
    assert m0.UNPACKED_POINTERS - m0.PACKED_POINTERS == 5


def test_the_prefix_is_NOT_in_the_pack_and_keeps_its_own_binding():
    """``pfx`` is the one member of the curl half's read-only set the plan REFRESHES
    every step. Folding a per-step write in beside constants nothing re-uploads would
    make a wrong offset a permanent, silent absorber corruption."""
    assert "pfx" not in family.PACKED_VECTORS
    source = emit()
    assert "device const float2* pfx     [[buffer(9)]]," in source
    assert "device const float*  cpml" in source


def test_the_pack_did_not_reach_the_arithmetic():
    source = emit()
    prologue = coefficient_pack.prologue("cpml", family.PACKED_VECTORS)
    assert prologue in source
    assert [line for line in source.splitlines()
            if "prm.off_" in line] == prologue.splitlines()
    curl = family.certified_cylindrical_curl_body(
        templates.METALLIC, cylindrical_complex.M_ONE, EXPANSION)
    head = curl.split(family._CURL_STORE, 1)[0]
    assert source.index(prologue) < source.index(head)


def test_the_params_record_and_the_struct_are_generated_from_one_tuple():
    """The kernel decodes by OFFSET, so a reordered record reinterprets every field."""
    source = emit()
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    assert struct.splitlines()[1].strip() == "float2 inc_b;"
    for name in family.PACKED_VECTORS:
        assert f"uint off_{name};" in struct
    dtype = family.params_record_dtype()
    assert dtype.itemsize == family.PARAMS_ITEMSIZE == 64
    assert list(dtype.names) == [
        "inc_b", "nx", "ny", "nz", "n_elem", "zrows", "dtdx", "minus_dtdx",
        "axis_coef"] + [f"off_{name}" for name in family.PACKED_VECTORS]
    # The six offsets are contiguous 4-byte members after the 40-byte prefix, which
    # is what the MSL struct's natural layout gives (36 before the m = 0 arm's
    # `axis_coef` became the ninth scalar; the itemsize did not move).
    offsets = [dtype.fields[name][1] for name in dtype.names]
    assert offsets == [0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    assert "float dtdx; float minus_dtdx; float axis_coef;" in struct


def test_the_eight_scalars_are_unpacked_into_the_certified_bodies_own_names():
    source = emit()
    for name in ("nx", "ny", "nz", "n_elem", "zrows", "dtdx", "minus_dtdx",
                 "axis_coef", "inc_b"):
        assert f"prm.{name}" in source
    lifted = source.split(
        coefficient_pack.prologue("cpml", family.PACKED_VECTORS), 1)[1]
    assert "prm." not in lifted


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_UNWIRED_and_carries_its_absorb_row():
    rows = [row for row in arms.registered() if row.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_D"
    assert rows[0].wired is False
    # THE MODULE NAME, NOT THE FAMILY NAME. They differ on this one product — the
    # module is `cylindrical_fused_electric_pair.py` and the arm family it registers
    # is `cylindrical_complex_fused_electric_pair`, exactly as on its magnetic twin.
    assert "cylindrical_fused_electric_pair" in registry.FAMILY_MODULES
    assert family.FAMILY not in registry.FAMILY_MODULES
    assert metal_launch.FUSED_PAIR_ARMS[family.FAMILY] == ("cylindrical complex",
                                                           "cylindrical complex")
    assert metal_launch.FUSED_PAIR_SEAMS[family.SLOT] == ("update_E", "D")


def test_the_absorb_row_names_the_arms_this_predicate_is_literally_built_out_of():
    source = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_fused_electric_pair.py").read_text(encoding="utf-8")
    body = source.split("def cylindrical_fused_electric_pair_coverage", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "cylindrical_complex_pml_curl_coverage(" in body
    assert "cylindrical_complex_constitutive_coverage(" in body


def test_the_deposit_flag_and_its_wiring_move_together():
    source = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_fused_electric_pair.py").read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source
    assert "carries_repair=True" not in source
    from .test_fused_pair_deposit_wiring import (
        METAL_ABSORB_DECLARATIONS, ROUTED_WITHOUT_THE_REPAIR, WIRED_FOR_THE_REPAIR)
    assert "metal_kernels/cylindrical_fused_electric_pair.py" in WIRED_FOR_THE_REPAIR
    assert family.FAMILY in METAL_ABSORB_DECLARATIONS
    # ...and the MAGNETIC twin still declines the REPAIR, on the same sixteen rows.
    # It no longer declines the ROW: since 2026-09-17 it holds an absorb row
    # (metal_kernels/launch.py:1234-1235) and is routed unbracketed, its source
    # clause refusing any in-seam magnetic deposit by name
    # (cylindrical_fused_magnetic_pair.py:710-721). So it is declared in
    # ROUTED_WITHOUT_THE_REPAIR -- keyed by its MODULE path, since its family name
    # diverges from its stem exactly as this module's does -- rather than absent.
    twin_source = (PACKAGE_DIR / "metal_kernels"
                   / "cylindrical_fused_magnetic_pair.py").read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = False" in twin_source
    twin_path = "metal_kernels/cylindrical_fused_magnetic_pair.py"
    assert twin_path not in WIRED_FOR_THE_REPAIR
    assert twin_path in ROUTED_WITHOUT_THE_REPAIR
    assert metal_launch.FUSED_PAIR_ARMS[twin.FAMILY] == ("cylindrical complex",
                                                         "cylindrical complex")
    assert twin.FAMILY in METAL_ABSORB_DECLARATIONS


def test_REPLACES_names_three_passes_and_the_injection_is_not_one_of_them():
    assert family.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert "sources" not in family.REPLACES


def _executable_text(path) -> str:
    """A module's source with comments and docstrings removed.

    ONE HOME FOR A RULE THIS PACKAGE ALREADY WROTE DOWN
    (``test_triton_kernels.code_of``): the prose in these modules NAMES the things
    they must not touch — that is how a reader learns which track is wired and
    which is not — so an ownership check that greps the RAW file fires on its own
    documentation. Measured 2026-09-02: correcting the four kernel-package
    docstrings that still claimed ``plan_fast_path`` returns None on every branch,
    and naming a gate ARTIFACT PATH in a refusal reason, turned four of these
    checks red without a single executable reference moving. Stripping to
    executable text is what makes the check about behaviour, and it is exactly
    what the companion test below this one has always done.
    """
    import ast as _ast

    from meep_gpu.code_identity import strip_docstrings

    return _ast.unparse(strip_docstrings(_ast.parse(
        pathlib.Path(path).read_text(encoding="utf-8"))))



#: The ONE way ``fastpath.py`` may name the Metal kernel package for itself: as a
#: path segment of the LEDGER it reads per table
#: (``os.path.join(..., "metal_kernels", "fingerprints.json")``).
#:
#: THE BARE "metal_kernels IS ABSENT" ASSERTION RETIRED WITH THE SECOND KERNEL
#: TABLE, and flipping it to "is present" would have been the wrong repair: "the
#: file mentions the package somewhere" says nothing about whether THIS module is
#: reachable, which is the whole subject. What replaced it is stronger than either
#: spelling. ``fastpath`` reaches the Metal table through the sibling
#: ``meep_gpu/metal_dispatch.py`` — the release rows live there precisely so a Metal
#: release edit never re-drifts the Triton ``driver_dispatch`` record — so an IMPORT
#: of the package from ``fastpath.py``, or a reference to any module inside it, is
#: still exactly the boundary violation the old assertion caught, on every module
#: rather than on this one.
_METAL_KERNELS_IMPORT = _re.compile(
    r"(?:^|\n)\s*(?:from\s+\.*metal_kernels|import\s+\.*metal_kernels"
    r"|from\s+[.\w]*\bmetal_kernels\b)")
#: A submodule reference: ``metal_kernels.launch`` or ``metal_kernels/launch.py``.
_METAL_KERNELS_MODULE = _re.compile(r"\bmetal_kernels[./](?!fingerprints\.json)[\w./]+")


def _metal_kernels_modules_named_in(text):
    """Every ``metal_kernels`` reference in ``text`` that is not the ledger path.

    Two shapes, because they fail differently: an IMPORT pulls the package into
    every process that touches the fast path, and a MODULE reference means the
    table's own vocabulary has leaked into the file the two tables share.
    """
    found = [match.group(0).strip()
             for match in _METAL_KERNELS_IMPORT.finditer(text)]
    found += [match.group(0) for match in _METAL_KERNELS_MODULE.finditer(text)]
    return sorted(set(found))

def _fastpath_code_outside_the_pending_reasons() -> str:
    """``fastpath.py``'s executable text with ``PENDING_DEVICE_GATE_ARMS`` blanked.

    WHY THE CARVE-OUT IS STRUCTURAL AND NOT A SUBSTRING ALLOWANCE. This check asks
    whether DISPATCH names a METAL module. Since the 2026-09-02 product wave,
    ``PENDING_DEVICE_GATE_ARMS`` carries, per refused label, the path of the TRITON
    gate artifact whose release has no ledger entry yet — and several of those
    Triton products share a bare name with a Metal one
    (``cylindrical_real_fused_magnetic_pair`` exists on both tracks). A raw
    substring check therefore fired on the OTHER track's artifact path while no
    executable reference to this package had moved at all.
    
    So the reason strings are removed by their OWN ASSIGNMENT NODE rather than by
    matching their text, and everything else in the file is still searched: an
    import, a table row, a call — any real naming of a Metal module — fails
    exactly as before.
    """
    import ast as _ast

    from meep_gpu.code_identity import strip_docstrings

    tree = strip_docstrings(_ast.parse(
        (PACKAGE_DIR / "fastpath.py").read_text(encoding="utf-8")))
    for node in _ast.walk(tree):
        targets = getattr(node, "targets", None) or (
            [node.target] if isinstance(node, _ast.AnnAssign) else [])
        named = {t.id for t in targets if isinstance(t, _ast.Name)}
        if "PENDING_DEVICE_GATE_ARMS" in named:
            node.value = _ast.Constant(value="<pending-gate reasons elided>")
    return _ast.unparse(tree)


def test_fastpath_does_not_name_this_module():
    fastpath = _fastpath_code_outside_the_pending_reasons()
    assert _metal_kernels_modules_named_in(fastpath) == [], (
        "fastpath.py names a metal_kernels MODULE; it may reach the Metal table "
        "only through the meep_gpu.metal_dispatch sibling and may read only that "
        "package's fingerprints.json")
    assert "metal_kernels/cylindrical_fused_electric_pair" not in fastpath
    assert "cylindrical_fused_electric_pair_step" not in fastpath


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    tree = ast.parse((PACKAGE_DIR / "metal_kernels"
                      / "cylindrical_fused_electric_pair.py").read_text(
                          encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code


# ---------------------------------------------------------------------------
# The gate, and the record it left
# ---------------------------------------------------------------------------

def test_the_gate_exists_and_names_the_product():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_cylindrical_fused_electric_pair" in source
    assert "cylindrical_fused_electric_pair_step" in source
    assert "refuted_unpacked_pointer_source" in source


def test_the_released_artifact_is_in_the_tree_and_says_it_released():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["release"]["released"] is True, payload["release"]
    assert payload["steps"] == 12
    assert all(row["passed"] for row in payload["rows"])
    products = [row for row in payload["rows"] if row["leg"] == "product"]
    assert len(products) == 9
    assert all(row["differing_words"] == 0 for row in products)
    # THE PREFIX PRE-PASS IS PART OF THE CONTRACT, asserted per case.
    assert all(row["prefix_syncs"] == 12 for row in products)
    # THE m = 0 ARM IS DRIVEN, not inferred from the |m| >= 1 rows.
    m_zero = [row for row in products if row["m_arm"] == cylindrical_complex.M_ZERO]
    assert len(m_zero) == 3 and all(row["zero_rows"] == 0 for row in m_zero)


def test_the_artifact_measured_the_binding_ceiling_in_every_direction():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    row = next(r for r in payload["rows"] if r["leg"] == "binding_ceiling")
    assert row["unpacked_pointer_compiled"] is False
    assert row["unpacked_pointer_refused_for_the_right_reason"] is True
    assert row["separate_scalar_compiled"] is False
    assert row["thirty_second_compiled"] is False
    assert row["shipped_compiled"] is True
    assert row["largest_pointer_count_that_compiles_with_one_struct"] == 30
    assert row["smallest_pointer_count_refused_with_one_struct"] == 31
    assert row["ceiling_measured_equals_declared"] is True
    assert row["over_the_ceiling_by"] == 3
    assert row["pack_saves_pointers"] == 5
    assert row["packed_fields_correct"] is True
    assert row["params_itemsize"] == family.PARAMS_ITEMSIZE


def test_the_artifact_measured_the_pack_against_six_separate_pointers():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    row = next(r for r in payload["rows"] if r["leg"] == "pack_identity")
    assert row["read_differing_words"] == 0
    assert row["nonzero_outputs"] > 0
    assert not any(row["bytes_differing_per_vector"].values())


def test_the_artifact_carried_a_real_deposit_and_its_null_control_diverged():
    """On BOTH arms that reach a corpus row with an in-seam electric source: the
    |m| = 1 row and, since 2026-09-04, the m = 0 row (``dipole_in_vacuum_cyl_off_
    axis.py`` declares two electric sources)."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    deposits = [r for r in payload["rows"] if r["leg"] == "deposit"]
    nulls = [r for r in payload["rows"] if r["leg"] == "deposit_null_control"]
    assert len(deposits) == 2 and len(nulls) == 2
    assert {r["m_arm"] for r in deposits} == {cylindrical_complex.M_ONE,
                                              cylindrical_complex.M_ZERO}
    for deposit in deposits:
        assert deposit["installed_plans"] == ["LeadingRepairPlan", "TrailingRepairPlan"]
        assert deposit["deposit_points_repaired"] > 0
        assert deposit["words_the_injection_moved"] > 0
        assert deposit["differing_words"] == 0
    for null in nulls:
        assert null["diverged"] is True
        assert null["words_the_injection_moved"] > 0


def test_the_artifact_armed_the_offset_defects_and_declared_only_the_phi_nulls():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    offsets = [r for r in payload["rows"] if r.get("offset_defect")]
    assert len(offsets) == 11
    nulls = sorted(r["label"] for r in offsets if r["expectation"] == "null")
    assert nulls == ["offset_kmy_plus_one_reads_sinvy",
                     "offset_sinvy_minus_one_reads_kmy"]
    assert all(r["caught"] is False for r in offsets if r["expectation"] == "null")
    assert all(r["caught"] for r in offsets if r["expectation"] == "caught")
    siblings = {r["label"]: r["caught"] for r in offsets}
    assert siblings["offset_kmy_minus_one_reads_sinvx"] is True
    assert siblings["offset_sinvy_plus_one_reads_kmz"] is True


def test_the_artifact_measured_the_fusion_and_caught_every_other_defect():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    controls = [r for r in payload["rows"] if r["leg"] == "separate_control"]
    assert controls and all(r["dispatches_removed_per_step"] == 1 for r in controls)
    # AND THE SCAN IS NOT WHAT THE FUSION REMOVED: both sides pay it every step.
    assert all(r["separate_prefix_syncs"] == r["fused_prefix_syncs"] == 12
               for r in controls)
    walled = [r for r in controls if r["label"] == "m1_z_metallic"]
    assert walled and walled[0][
        "host_passes_in_seam_on_the_separate_side"] == ["zero_metal_D"]
    assert walled[0]["host_passes_in_seam_on_the_fused_side"] == []
    assert any(r["label"] == "m0_z_metallic" for r in controls)
    mutations = [r for r in payload["rows"] if r["leg"] == "mutation"]
    assert all(r["caught"] for r in mutations
               if r.get("expectation", "caught") == "caught")
    assert all(r["caught"] is False for r in mutations
               if r.get("expectation") == "null")
    # THE m = 0 ARM'S OWN DEFECTS are armed on the m = 0 case, and its one declared
    # null — the in-kernel `4.0f * dtdx` spelling — is CONFIRMED null, not assumed.
    m_zero = [r for r in mutations if r.get("case") == "m0_z_metallic"]
    assert len(m_zero) >= 5
    assert [r["label"] for r in m_zero if r["expectation"] == "null"] == [
        "m_zero_axis_coef_formed_in_kernel"]
    assert payload["expansion"] == "FMA_V1"


def test_the_module_claims_identity_ONLY_through_the_artifact_that_measured_it():
    import hashlib

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for stem in ("cylindrical_fused_electric_pair.py", "coefficient_pack.py"):
        module = PACKAGE_DIR / "metal_kernels" / stem
        live = hashlib.sha256(module.read_bytes()).hexdigest()
        recorded = {key: value for key, value in payload["source_sha256"].items()
                    if key.endswith(f"metal_kernels/{stem}")}
        assert recorded, sorted(payload["source_sha256"])[:5]
        for key, digest in sorted(recorded.items()):
            if live == digest or weld_survives_edit(module, payload, key):
                continue
            assert live == digest, (
                f"{stem} has changed since the gate ran in a way that reaches "
                f"executable code")
