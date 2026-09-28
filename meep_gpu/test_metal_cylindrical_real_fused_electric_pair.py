"""The Metal Dcyl m = 0 fused ELECTRIC weld, as claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_cylindrical_real_fused_electric_pair.py`` steps
engines side by side for twelve complete steps and compares uint32 words. No assertion
here duplicates that. What lives here is everything true about this family WITHOUT a
device:

* that the emitted source is the two CERTIFIED emitters' own bytes with exactly one
  substitution per component — the ``float srcN = gN[ii] * ieN[ii];`` reload becoming
  the register — plus the spliced wall clear, and nothing else;
* that the wall clear it splices is the D family's OFF-DIAGONAL table and not the B
  side's complement, which is the single most likely porting slip;
* **that the PACK did not reach the arithmetic**: the six aliases are declared once,
  before the lifted text, and no ``prm.off_`` read appears anywhere else;
* that the binding count the module spells is the count its emitter produces, that the
  UNPACKED shape is over the platform ceiling by exactly one binding, and that both
  numbers come from counting the emitters' own output rather than from a docstring;
* that the arm is registered UNWIRED but IS in the absorb table, which is the pairing
  ``CARRIES_DEPOSIT_REPAIR`` is only allowed to move with;
* that the two driver passes :data:`REPLACES` omits really cannot execute on a grid
  this predicate admits — executed, not read off another module's guard;
* that its cell is DISJOINT from the |m| >= 1 complex sibling's, measured off the
  census rather than argued from the names.

THE CORPUS DEMAND, so the suite states what this is worth rather than leaving it to a
docstring: 3 reachable D->E seam-instances of the 387 priced on the 2026-08-19 census,
and the cell has NO attrition — 3 rows drive it, 3 are admitted by both halves, and
all 3 carry an ELECTRIC deposit inside the seam. With ``CARRIES_DEPOSIT_REPAIR`` at
False the same product would serve ZERO, which is what makes the flag the product here
rather than one clause of it.

THE CELL'S HISTORY, because it is the only one on this board recovered from the
BINDING ceiling rather than from a clause. ``fusion_matrix_metal_2026-08-30_lastcells``
scored it ``UNFUSABLE ON METAL — even at the largest sharing measured on this seam at
this arity (3) the fused signature needs 31 pointers, over the 30 the platform
allows``. Over by exactly one, and the one is the radial prefix. What cleared it is
:mod:`meep_gpu.metal_kernels.coefficient_pack` — NOT by folding the prefix (that
volume is device-written and packing it beside read-only coefficients would let the
scan corrupt an absorber silently) but by folding the curl half's six read-only PML
coefficient vectors into one buffer with six element offsets in the ``Params`` struct
the scalars already ride in.
"""

from __future__ import annotations

import re as _re

import ast
import json
import os
import pathlib

import numpy as np
import pytest

# THIS MODULE'S GREEN MAY NOT DEPEND ON A SIBLING'S IMPORT ORDER, which it did until
# 2026-08-31. `conftest.py`'s header names this hazard exactly: about twenty Metal
# parity modules open with this same `setdefault` at module scope, pytest imports
# every test module during COLLECTION, and so a full serial run happened to leave
# `flush` in force before this file's tests ran. MEASURED, three ways: standalone
# this module was 3 failed / 41 passed; preceded by
# `test_metal_below_the_cut_fused_pairs.py` (which declares the policy itself) all 44
# passed; with `MEEP_GPU_SUBNORMAL_POLICY=flush` exported all 44 passed. The three
# that failed are the ones that ASK the coverage predicate, and under the default
# `keep` it refuses by name — "the resolved float32 subnormal policy is 'keep' ... the
# MPS executor cannot honour it" — so the suite measured nothing and read as a red.
#
# `flush` is not a convenience here, it is the policy this family is certified under:
# the weld records `flush — native and uncontrollable on MPS; the oracle flushes too`,
# and it is the only policy under which this weld's coverage question has an answer.
# `setdefault`, so an explicit export from a campaign runner still wins. The same
# one-line declaration `test_metal_below_the_cut_fused_pairs.py` carries, for the same
# reason, and it is what makes the suite shardable rather than serial-only.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from .device_identity import weld_survives_edit  # noqa: E402

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = (REPO / "parity" / "meep_gpu"
        / "gate_metal_cylindrical_real_fused_electric_pair.py")
#: REPOINTED 2026-09-25 to the ``_2026-09-25_night`` fleet's re-run (released, PASS, 253
#: recorded digests, ZERO disagreeing with the tree, checked before this line moved): the
#: citation re-point edited this module's comments, and the 08-31 artifact records no
#: device or code digest ``weld_survives_edit`` could clear that edit through.
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_cylindrical_real_fused_electric_pair_2026-09-25_night"
            / "gate.json")
CENSUS = (REPO / "parity" / "meep_gpu" / "results"
          / "metal_coverage_tranche6_2026-08-19")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, coefficient_pack,
    cylindrical_real as cyl,
    cylindrical_real_fused_electric_pair as family,
    cylindrical_real_fused_magnetic_pair as twin,
    fused_electric_pair as cartesian,
    launch as metal_launch,
    registry,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS, Residency  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The Dcyl specialisation every admitted grid resolves the r axis to. The r axis is
#: pinned METALLIC (CYL_AXIS shares ``_shift_up``'s metallic zero ghost); z is the one
#: axis this family parameterises.
CODES_METALLIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.METALLIC)
CODES_PERIODIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.PERIODIC)
WALLED_Z = (False, False, True)
UNWALLED = (False, False, False)


def emit(codes=CODES_METALLIC_Z, walls=WALLED_Z):
    return family.cylindrical_real_fused_electric_pair_source(codes, walls)


def build(z_kind: str = "metallic", shape=(20, 1, 20), courant: float = 0.5,
          m: int = 0, complex_storage: bool = False):
    """A real Dcyl Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": z_kind},
                courant=float(courant), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness={"x": (0, max(2, shape[0] // 4)),
                                             "z": max(2, shape[2] // 4)})


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def _real_source(fields, component: str):
    """A REAL engine source, so ``field_type`` and the deposit index are the
    engine's own answers rather than a stub's."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.5, 0.0, 0.5), size=(0.0, 0.0, 0.0),
                        amplitude=1j,
                        envelope=ContinuousEnvelope(frequency=1.0))


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

def test_the_cell_is_three_seam_instances_and_every_one_carries_a_deposit():
    """3 -> 3 -> 3, and the third step is the one the flag pays for.

    All three rows declare ``source_field_types == ['D']``, so the count with
    ``CARRIES_DEPOSIT_REPAIR`` at False would be ZERO. That is the reverse of the
    magnetic twin's situation on the same three rows, where the seam is empty.
    """
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    curl = [r for r in rows if covered(r, "cylindrical_real_curl@step_D")]
    both = [r for r in curl if covered(r, "cylindrical_real_constitutive@update_E")]
    assert (len(curl), len(both)) == (3, 3)
    for row in both:
        configuration = row["configuration"]
        assert configuration["source_field_types"] == ["D"], configuration
        assert configuration["metallic"] == [False, False, True], configuration
    without_the_repair = [r for r in both
                          if not any(str(kind) == "D" for kind
                                     in (r["configuration"]["source_field_types"]
                                         or []))]
    assert without_the_repair == []
    assert family.CARRIES_DEPOSIT_REPAIR is True


def test_this_cell_is_disjoint_from_the_complex_cylindrical_sibling():
    """3 rows against 16, intersection empty — measured, not argued from the names."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    real = {f"{r['leg']}:{r['row']}" for r in rows
            if covered(r, "cylindrical_real_curl@step_D")}
    complex_arm = {f"{r['leg']}:{r['row']}" for r in rows
                   if covered(r, "cylindrical_complex_pml_curl@step_D")}
    assert len(real) == 3 and len(complex_arm) == 16
    assert real & complex_arm == set()


def test_the_two_seams_of_this_cell_serve_the_SAME_three_rows():
    """The magnetic twin and this product are the same two halves on two seams."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    magnetic = {f"{r['leg']}:{r['row']}" for r in rows
                if covered(r, "cylindrical_real_curl@step_B")}
    electric = {f"{r['leg']}:{r['row']}" for r in rows
                if covered(r, "cylindrical_real_curl@step_D")}
    assert magnetic == electric and len(electric) == 3


# ---------------------------------------------------------------------------
# The construction: both halves are the certified emitters' bytes
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_emitters_body_on_both_sides_of_the_splice():
    source = emit()
    curl = family.certified_cyl_curl_body(CODES_METALLIC_Z)
    head, tail = curl.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_constitutive_half_differs_in_exactly_the_three_seam_lines():
    source = emit()
    certified = family.certified_constitutive_body().splitlines()
    marker = "    // --- update_E (stepping.update_E"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(certified) == len(spliced)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float src{t} = g{t}[ii] * ie{t}[ii];",
                        f"    float src{t} = v{t} * ie{t}[ii];") for t in range(3)]


def test_the_inverse_permittivity_stays_on_the_right_of_the_seam_product():
    """``D * inv_eps``, not ``inv_eps * D``. float32 multiply is commutative to the
    bit, but the array path's ORDER is what the transcription rule protects and a
    reader must be able to see it unchanged."""
    source = emit()
    for target in range(3):
        assert f"float src{target} = v{target} * ie{target}[ii];" in source


def test_the_electric_half_reads_no_g_pointer_at_all():
    """The curl's ``g`` is H here and the constitutive's ``g`` was D. One surviving
    read would take the magnetic field for a displacement — smooth and wrong."""
    source = emit()
    electric = source.split("// --- update_E (stepping.update_E", 1)[1]
    for target in range(3):
        assert f"g{target}[" not in electric


def test_the_wall_clear_is_the_D_familys_off_diagonal_and_not_the_B_sides():
    from meep_gpu.fields import IYEE_SHIFTS
    from meep_gpu.metal_kernels.fused_dispersive_pair import _ZERO_METAL_ROWS as D_ROWS

    d_rows = {(target, axis) for target, axis, _flag in D_ROWS}
    b_rows = {(target, axis) for target, axis, _flag in twin._ZERO_METAL_ROWS}
    assert d_rows == {(1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2)}
    assert d_rows.isdisjoint(b_rows)
    for target, axis in d_rows:
        assert IYEE_SHIFTS[("Dx", "Dy", "Dz")[target]][axis] == 0


def test_a_z_walled_source_clears_the_two_tangential_D_components_and_a_periodic_none():
    walled, periodic = emit(), emit(CODES_PERIODIC_Z, UNWALLED)
    assert "    v0 = at_z ? 0.0f : v0;" in walled
    assert "    v1 = at_z ? 0.0f : v1;" in walled
    assert "    v2 = at_z ? 0.0f : v2;" not in walled  # Dz is the DIAGONAL one
    assert "at_z ? 0.0f" not in periodic
    assert "no walled axis clears a D component" in periodic


def test_the_wall_clear_sits_after_the_axis_rules_and_before_the_store():
    """The driver's own order: step_D ends with the axis rules, then zero_metal_D."""
    source = emit()
    axis_increment = source.index("    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;")
    axis_zero = source.index("    v1 = at_x ? 0.0f : v1;")
    clear = source.index("    v0 = at_z ? 0.0f : v0;")
    store = source.index(family._CURL_STORE)
    seam = source.index("float src0 = v0 * ie0[ii];")
    assert axis_increment < axis_zero < clear < store < seam


def test_the_m0_axis_increment_is_a_POST_ADD_and_is_not_folded_into_the_curl():
    """MEASURED in ``stepping.py``: folding it breaks m = 0 under PML (Er 2.7e-01
    against 3.6e-07), because Dz's dsig is R and sigma is zero on the axis row."""
    source = emit()
    increment = source.index("v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;")
    recurrence = source.index("float v2 = (((f2[ii] * km_y) + n2) - p2) * si_y;")
    assert recurrence < increment
    assert "curl2 = at_x ? (curl2 + (axis_coef" not in source


def test_a_periodic_r_axis_is_refused_by_name_before_any_splice():
    with pytest.raises(ValueError, match="must compile as METALLIC"):
        family.cylindrical_real_fused_electric_pair_source(
            (cyl.PERIODIC, cyl.PERIODIC, cyl.METALLIC), WALLED_Z)


# ---------------------------------------------------------------------------
# The signature, and the pack that makes it fit
# ---------------------------------------------------------------------------

def test_the_binding_count_the_module_spells_is_the_count_its_emitter_produces():
    source = emit()
    assert source.count("[[buffer(") == family.PACKED_BINDINGS
    # Counted off the SIGNATURE lines only: the pack prologue also declares
    # `device const float*` names, and those are aliases rather than bindings.
    pointers = [line for line in source.splitlines()
                if "[[buffer(" in line and "device" in line]
    scalars = [line for line in source.splitlines()
               if "[[buffer(" in line and "constant" in line]
    assert len(pointers) == family.PACKED_POINTERS
    assert len(scalars) == 1, scalars
    assert family.PACKED_BINDINGS == family.PACKED_POINTERS + 1
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_unpacked_shape_is_over_the_ceiling_by_EXACTLY_ONE_binding():
    """THE MEASUREMENT THE PACK EXISTS FOR, taken off the refuted emitters' own text.

    31 pointers plus one packed ``Params&`` is 32 bindings against a ceiling of 31.
    That is the board's ``UNFUSABLE ON METAL`` verdict for this cell restated as
    arithmetic; the gate's ``binding_ceiling`` leg is what turns it into a COMPILE.
    """
    refuted = family.refuted_unpacked_pointer_source()
    assert refuted.count("[[buffer(") == family.UNPACKED_POINTER_BINDINGS
    assert family.UNPACKED_POINTER_BINDINGS == family.UNPACKED_POINTERS + 1
    assert family.UNPACKED_POINTER_BINDINGS == MAX_BUFFER_BINDINGS + 1
    # ...and with the scalars unpacked too it is further over, which is the shape the
    # scalar packing every shipped pair already does exists to avoid.
    assert (family.refuted_separate_scalar_source().count("[[buffer(")
            == family.SEPARATE_SCALAR_BINDINGS > family.UNPACKED_POINTER_BINDINGS)


def test_the_pack_saves_exactly_five_pointers_and_the_cell_needed_one():
    """The margin is deliberate rather than lucky, and the suite says which is which.

    The cell was over by ONE. The pack saves FIVE, because the unit packed is the curl
    half's per-axis coefficient GROUP and not an arbitrary subset chosen to land on
    the ceiling — six vectors in, one pointer out.
    """
    assert family.UNPACKED_POINTERS - family.PACKED_POINTERS == 5
    assert len(family.PACKED_VECTORS) == 6
    assert family.UNPACKED_POINTERS - (MAX_BUFFER_BINDINGS - 1) == 1


def test_the_prefix_is_NOT_in_the_pack_and_keeps_its_own_binding():
    """THE ONE DESIGN DECISION HERE. ``pfx`` is the only member of the curl half's
    read-only set that the DEVICE writes — the radial scan fills it in the launch
    before this one. Folding it into the same allocation as the coefficients would
    have saved a seventh pointer and let a wrong offset in the SCAN overwrite a PML
    coefficient vector permanently (a constant mirror is uploaded once) and silently
    (an absorber that leaks a little is a plausible field)."""
    assert "pfx" not in family.PACKED_VECTORS
    source = emit()
    assert "device const float* pfx     [[buffer(9)]]," in source
    assert "device const float* cpml" in source


def test_the_packed_vectors_are_the_curl_halfs_six_and_in_the_emitters_order():
    assert family.PACKED_VECTORS == ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz")
    certified = cyl.cylindrical_curl_source("step_D", CODES_METALLIC_Z)
    order = [name for name in family.PACKED_VECTORS]
    positions = [certified.index(f"device const float* {name}") for name in order]
    assert positions == sorted(positions)


def test_the_pack_did_not_reach_the_arithmetic():
    """THE INVARIANT THAT KEEPS THE LIFT A LIFT. The six aliases are declared ONCE,
    before any lifted text, and no ``prm.off_`` read appears anywhere else — so the
    certified bodies go on reading ``kmx[i]`` character for character."""
    source = emit()
    prologue = coefficient_pack.prologue("cpml", family.PACKED_VECTORS)
    assert prologue in source
    offset_lines = [line for line in source.splitlines() if "prm.off_" in line]
    assert offset_lines == prologue.splitlines()
    curl = family.certified_cyl_curl_body(CODES_METALLIC_Z)
    first_lifted = curl.splitlines()[1]
    assert source.index(prologue) < source.index(first_lifted)


def test_the_struct_carries_the_six_scalars_and_the_six_offsets_in_one_order():
    source = emit()
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    assert "uint nx; uint ny; uint nz; uint n_elem; float dtdx; float axis_coef;" \
        in struct
    for name in family.PACKED_VECTORS:
        assert f"uint off_{name};" in struct
    fields = coefficient_pack.record_dtype_fields(family.PACKED_VECTORS)
    record = np.zeros(1, dtype=np.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
         ("dtdx", "<f4"), ("axis_coef", "<f4")] + fields))
    assert record.itemsize == 4 * (6 + len(family.PACKED_VECTORS))


def test_the_scalars_are_unpacked_into_the_certified_bodies_own_names():
    """A struct field read inline would fork the lifted text; it must not."""
    source = emit()
    for name in ("nx", "ny", "nz", "n_elem", "dtdx", "axis_coef"):
        assert f"prm.{name}" in source
    lifted = source.split(
        coefficient_pack.prologue("cpml", family.PACKED_VECTORS), 1)[1]
    assert "prm." not in lifted


def test_the_signature_group_order_is_the_cartesian_D_E_pairs():
    """One layout across the two D/E pairs, so a reader comparing them is comparing
    arithmetic and not bookkeeping. The two differences are named and no others:
    the radial prefix, and the six coefficient pointers become one pack."""
    def groups(text):
        return [line.split("*")[1].split("[[")[0].strip()
                for line in text.splitlines() if "[[buffer(" in line
                and "device" in line]

    mine = groups(emit())
    theirs = groups(cartesian.fused_electric_pair_source((1, 1, 1), UNWALLED))
    assert [n for n in mine if n not in ("pfx", "cpml")] == [
        n for n in theirs if n not in ("kmx", "sinvx", "kmy", "sinvy",
                                       "kmz", "sinvz")]


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def residual(verdict):
    return [reason for reason in verdict.reasons]


def test_a_dcyl_m0_grid_is_admitted_on_both_z_declarations():
    for z_kind in ("metallic", "periodic"):
        fields, pml = build(z_kind)
        verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
            fields, pml, (), Residency())
        assert verdict.covered, (z_kind, residual(verdict))


def test_an_electric_source_is_admitted_and_a_magnetic_one_does_not_disqualify():
    """The mirror image of the twin's clause, and the two are asked TOGETHER.

    If both families refused both polarities one of them would have copied the
    other's clause rather than read the driver.
    """
    fields, pml = build()
    residency = Residency()
    # A REAL engine source, because with the repair carried the electric clause now
    # consults `deposit_repair` and a stub that publishes no deposit index is refused
    # BY NAME — which is the next test, and a different question from this one.
    electric_source = _real_source(fields, "Ez")
    magnetic_source = _real_source(fields, "Hy")
    electric = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (electric_source,), residency)
    magnetic = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (magnetic_source,), residency)
    assert str(electric_source.field_type) == "D"
    assert str(magnetic_source.field_type) == "B"
    assert electric.covered, residual(electric)
    assert magnetic.covered, residual(magnetic)
    # ...and the twin answers the other way on the same two sources.
    twin_electric = twin.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (electric_source,), Residency())
    twin_magnetic = twin.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (magnetic_source,), Residency())
    assert twin_electric.covered
    assert not twin_magnetic.covered
    assert any("driver.py:3283-3284" in r for r in twin_magnetic.reasons)


def test_an_electric_source_publishing_no_deposit_index_is_REFUSED_by_name():
    """FAIL CLOSED. The flag carries deposits this module can reconstruct, not every
    deposit — so a source whose index the engine never resolved is refused, and by the
    PLAN as well as the predicate."""
    fields, pml = build()
    indexless = _real_source(fields, "Ez")
    indexless._point_ix = indexless._point_iy = indexless._point_iz = None
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (indexless,), Residency())
    assert not verdict.covered
    assert any("does not publish the index it writes" in r for r in verdict.reasons)
    assert family.plan_metal_cylindrical_real_fused_electric_pair(
        fields, pml, (indexless,), Residency()) is None


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = build()
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, None, Residency())
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)
    assert family.plan_metal_cylindrical_real_fused_electric_pair(
        fields, pml, None, Residency()) is None


def test_m_not_zero_and_a_cartesian_grid_are_both_refused_by_name():
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    fields, pml = build(m=1, complex_storage=True)
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (), Residency())
    assert not verdict.covered
    assert any("grid.m = 1" in reason for reason in verdict.reasons)

    grid = Grid(resolution=1.0, cell_size=(6.0, 6.0, 6.0), xp=np)
    cart = Fields(grid=grid)
    cart.enable_pml_storage()
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        cart, PML(grid=grid, thickness=1.0), (), Residency())
    assert not verdict.covered
    assert any("cylindrical" in reason for reason in verdict.reasons)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(6.0, 6.0, 6.0), xp=np)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, PML(grid=grid, thickness=1.0), (), Residency())
    assert any(r.startswith("cylindrical curl half:") for r in verdict.reasons)
    assert any(r.startswith("cylindrical constitutive half:") for r in verdict.reasons)


def test_a_registered_polarization_is_refused_because_the_source_becomes_D_minus_P():
    fields, pml = build()
    fields.polarizations = ("a susceptibility",)
    verdict = family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (), Residency())
    assert not verdict.covered
    assert any("D - sum P" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The seam this product claims
# ---------------------------------------------------------------------------

def test_the_two_omitted_passes_are_inert_on_an_admitted_grid():
    """EXECUTED, not read off another module's guard, with a control that must move."""
    from meep_gpu import stepping

    fields, pml = build()
    rng = np.random.default_rng(11)
    names = [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    for name in names:
        array = getattr(fields, name)
        array[...] = rng.uniform(-1.0, 1.0, size=array.shape).astype(array.dtype)
    assert family.metal_cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (), Residency()).covered
    before = {name: np.array(getattr(fields, name), copy=True) for name in names}
    for pass_name in family.INERT_PASSES:
        getattr(stepping, pass_name)(fields)
    for name in names:
        assert np.array_equal(before[name].view(np.uint32),
                              np.asarray(getattr(fields, name)).view(np.uint32)), name
    # THE CONTROL: zero_metal_D is not inert on this walled grid.
    stepping.zero_metal_D(fields)
    moved = [name for name in names
             if not np.array_equal(before[name].view(np.uint32),
                                   np.asarray(getattr(fields, name)).view(np.uint32))]
    assert moved, "zero_metal_D moved nothing on a WALLED grid"
    assert zero_metal_axes(fields.grid) == (False, False, True)


def test_REPLACES_and_INERT_PASSES_partition_the_seam():
    """Three carried plus two inert is the driver's five, with nothing unaccounted."""
    assert family.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert set(family.INERT_PASSES) == {"fill_symmetry_bc_D",
                                        "fill_folded_far_ghosts_D"}
    assert set(family.REPLACES).isdisjoint(family.INERT_PASSES)
    # The injection is the FIFTH pass and it is deliberately in neither set: it is
    # carried by the repair bracket AROUND the launch, not by the launch.
    assert "sources" not in family.REPLACES


def test_the_plan_is_two_dispatches_and_counts_them_apart():
    assert family.MetalCylindricalRealFusedElectricPairPlan.launches_per_run == 2
    slots = family.MetalCylindricalRealFusedElectricPairPlan.__slots__
    assert "prefix_launches" in slots and "fused_launches" in slots


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_UNWIRED_and_carries_its_absorb_row():
    """THE PAIRING ``CARRIES_DEPOSIT_REPAIR`` MAY ONLY MOVE WITH.

    ``wired=False`` keeps the arm out of ``_select_slot``; the absorb row is what lets
    ``_install_fused_pairs`` reach it and bracket it. A Metal family that flipped the
    flag without the row would be claiming a bracket the seam loop refuses to build.
    """
    rows = [row for row in arms.registered() if row.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_D"
    assert rows[0].wired is False
    assert family.FAMILY in registry.FAMILY_MODULES
    assert metal_launch.FUSED_PAIR_ARMS[family.FAMILY] == ("cylindrical m=0",
                                                           "cylindrical m=0")
    assert metal_launch.FUSED_PAIR_SEAMS[family.SLOT] == ("update_E", "D")


def test_the_absorb_row_names_the_arms_this_predicate_is_literally_built_out_of():
    """Read off the module, not guessed: the coverage function calls exactly the two
    ``cylindrical m=0`` arm predicates and nothing else."""
    source = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_real_fused_electric_pair.py").read_text(encoding="utf-8")
    body = source.split("def metal_cylindrical_real_fused_electric_pair_coverage",
                        1)[1].split("\ndef ", 1)[0]
    assert "cylindrical_real_curl_coverage(fields, pml, CURL_SUB_STEP, residency)" \
        in body
    assert "cylindrical_real_constitutive_coverage(" in body


def test_the_deposit_flag_and_its_wiring_move_together():
    source = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_real_fused_electric_pair.py").read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source
    assert "carries_repair=True" not in source
    from .test_fused_pair_deposit_wiring import (
        METAL_ABSORB_DECLARATIONS, WIRED_FOR_THE_REPAIR)
    key = "metal_kernels/cylindrical_real_fused_electric_pair.py"
    assert key in WIRED_FOR_THE_REPAIR
    assert family.FAMILY in METAL_ABSORB_DECLARATIONS


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
    """DISPATCH must not reach this package — asked in fastpath's OWN vocabulary.

    THE BARE NAME STOPPED SEPARATING THE TRACKS, and pretending it still does
    would be the wrong kind of green. The Triton track ships a product with the
    SAME name: since 2026-09-11 ``fastpath.ARM_CERTIFICATION`` carries
    ``'fused pair D (cylindrical)': ('cylindrical_real_fused_electric_pair',
    'triton_cylindrical_real_fused_electric_pair_device_gate')`` — a Triton family
    and its Triton ledger key — and ``PENDING_DEVICE_GATE_ARMS`` has carried
    Triton gate ARTIFACT PATHS with the same spelling since the 2026-09-02 wave.
    A bare-substring check over ``fastpath.py`` therefore reports the OTHER
    track's certification table as if this Metal module had been wired, which is
    a failure about nothing.

    So the question is asked the way the magnetic sibling asks it, and the
    replacement is STRONGER than the bare name rather than weaker: ``fastpath.py``
    must not name the METAL PACKAGE at all, so an import, a table row or a call
    reaching ANY Metal module fails here, not just this one. It is the same fact
    ``dispatch_reachability.backend_reaches_the_dispatch_seam`` measures off the
    parse tree on every board cut, and it is why every Metal board reports
    ``served_in_dispatch: 0``. The Metal-qualified path is checked on its own line
    so a future ``fastpath`` importing this module by path fails there rather than
    only through the package clause, and the Metal-only kernel entry point is
    checked because no Triton name collides with it.
    """
    fastpath = _fastpath_code_outside_the_pending_reasons()
    assert _metal_kernels_modules_named_in(fastpath) == [], (
        "fastpath.py names a metal_kernels MODULE; it may reach the Metal table "
        "only through the meep_gpu.metal_dispatch sibling and may read only that "
        "package's fingerprints.json")
    assert "metal_kernels/cylindrical_real_fused_electric_pair" not in fastpath
    assert "cyl_real_fused_electric_pair_step" not in fastpath


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    tree = ast.parse((PACKAGE_DIR / "metal_kernels"
                      / "cylindrical_real_fused_electric_pair.py").read_text(
                          encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "triton_kernels.launch" not in code or "SUB_STEPS" in code


# ---------------------------------------------------------------------------
# The gate, and the record it left
# ---------------------------------------------------------------------------

def test_the_gate_exists_and_names_the_product():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_metal_cylindrical_real_fused_electric_pair" in source
    assert "cyl_real_fused_electric_pair_step" in source
    assert "refuted_unpacked_pointer_source" in source


def test_the_released_artifact_is_in_the_tree_and_says_it_released():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["release"]["released"] is True, payload["release"]
    assert payload["steps"] == 12
    assert all(row["passed"] for row in payload["rows"])
    products = [row for row in payload["rows"] if row["leg"] == "product"]
    assert len(products) == 6
    assert all(row["differing_words"] == 0 for row in products)


def test_the_artifact_measured_the_binding_ceiling_in_BOTH_directions():
    """The leg that licenses the pack, read back from the artifact.

    The UNPACKED signature must have been REFUSED and the shipped one must have
    COMPILED — and the ceiling must have been located by bisection rather than read
    off ``MAX_BUFFER_BINDINGS``.
    """
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    row = next(r for r in payload["rows"] if r["leg"] == "binding_ceiling")
    assert row["unpacked_pointer_compiled"] is False
    assert row["unpacked_refused_for_the_right_reason"] is True
    assert row["separate_scalar_compiled"] is False
    assert row["shipped_compiled"] is True
    assert row["largest_pointer_count_that_compiles_with_one_struct"] == 30
    assert row["smallest_pointer_count_refused_with_one_struct"] == 31
    assert row["ceiling_measured_equals_declared"] is True
    assert row["over_the_ceiling_by"] == 1
    assert row["pack_saves_pointers"] == 5
    assert row["packed_fields_correct"] is True


def test_the_artifact_measured_the_pack_against_six_separate_pointers():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    row = next(r for r in payload["rows"] if r["leg"] == "pack_identity")
    assert row["read_differing_words"] == 0
    assert row["nonzero_outputs"] > 0
    assert not any(row["bytes_differing_per_vector"].values())
    assert row["lengths_agree"] is True


def test_the_artifact_carried_a_real_deposit_and_its_null_control_diverged():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    deposit = next(r for r in payload["rows"] if r["leg"] == "deposit")
    assert deposit["installed_plans"] == ["LeadingRepairPlan", "TrailingRepairPlan"]
    assert deposit["deposit_points_repaired"] > 0
    assert deposit["words_the_injection_moved"] > 0
    assert deposit["differing_words"] == 0
    null = next(r for r in payload["rows"] if r["leg"] == "deposit_null_control")
    assert null["diverged"] is True
    assert null["words_the_injection_moved"] > 0


def test_the_artifact_armed_the_offset_defects_and_declared_only_the_phi_nulls():
    """THE HAZARD THE PACK INTRODUCES, MEASURED. Six vectors in one allocation are
    told apart by six uints and nothing else."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    offsets = [r for r in payload["rows"] if r.get("offset_defect")]
    assert len(offsets) == 11
    nulls = sorted(r["label"] for r in offsets if r["expectation"] == "null")
    assert nulls == ["offset_kmy_plus_one_reads_sinvy",
                     "offset_sinvy_minus_one_reads_kmy"]
    assert all(r["caught"] is False for r in offsets if r["expectation"] == "null")
    assert all(r["caught"] for r in offsets if r["expectation"] == "caught")
    # ...and the two nulls' SIBLINGS — the same offsets moved the other way — are
    # caught, which is what stops the nulls reading as "the offsets do not matter".
    siblings = {r["label"]: r["caught"] for r in offsets}
    assert siblings["offset_kmy_minus_one_reads_sinvx"] is True
    assert siblings["offset_sinvy_plus_one_reads_kmz"] is True


def test_the_artifact_measured_the_fusion_and_every_other_mutation_was_caught():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    controls = [r for r in payload["rows"] if r["leg"] == "separate_control"]
    assert controls and all(r["separate_dispatches_per_step"] == 3
                            and r["fused_dispatches_per_step"] == 2
                            for r in controls)
    walled = [r for r in controls if r["label"] == "square_metallic"]
    assert walled and walled[0]["seam_host_passes_for_separate"] == ["zero_metal_D"]
    assert walled[0]["seam_host_passes_for_fused"] == []
    mutations = [r for r in payload["rows"] if r["leg"] == "mutation"]
    # FOUR DECLARED NULLS AND NO OTHERS. Each is a measurement with a live sibling in
    # the same set: the phi-invariant parenthesisation against its (z, r) twin, the
    # two phi offset swaps against the same offsets moved the other way, and the Dz
    # r ownership mask against Dy's.
    nulls = sorted(r["label"] for r in mutations if r.get("expectation") == "null")
    assert nulls == ["curl_parens_flattened_on_the_invariant_pair",
                     "offset_kmy_plus_one_reads_sinvy",
                     "offset_sinvy_minus_one_reads_kmy",
                     "ownership_mask_dropped_on_r_for_dz"]
    caught = {r["label"]: r["caught"] for r in mutations}
    assert caught["curl_parens_flattened_on_a_live_pair"] is True
    assert caught["ownership_mask_dropped_on_r_for_dy"] is True
    assert all(r["caught"] for r in mutations
               if r.get("expectation", "caught") == "caught")
    assert all(r["caught"] is False for r in mutations
               if r.get("expectation") == "null")


def test_the_module_claims_identity_ONLY_through_the_artifact_that_measured_it():
    """A released gate licenses the claim, and the digest is what binds it."""
    import hashlib

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for stem in ("cylindrical_real_fused_electric_pair.py", "coefficient_pack.py"):
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
                f"executable code; the identity claim no longer describes the bytes "
                f"that were measured")
