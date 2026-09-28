"""The Metal Dcyl m = 0 fused magnetic weld, as claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_cylindrical_real_fused_magnetic_pair.py`` steps
engines side by side for twelve complete steps and compares uint32 words. No
assertion here duplicates that. What lives here is everything true about this family
WITHOUT a device:

* that the emitted source is the two CERTIFIED emitters' own bytes with exactly one
  substitution per component — the ``float srcN = gN[ii];`` reload becoming the
  register — plus the spliced wall clear, and nothing else;
* that the wall clear it splices is the B family's DIAGONAL table and not the D
  side's complement, which is the single most likely porting slip;
* that the arm is registered UNWIRED and carries its absorb row — since 2026-09-17
  the Metal composer asks this product through its ``launch.FUSED_PAIR_ARMS`` row
  and through nothing else in ``launch`` — and that ``fastpath`` names no Metal
  module;
* that the binding count the module spells is the count its emitter produces, and
  that it sits under the platform ceiling with margin;
* that the two driver passes :data:`REPLACES` omits really cannot execute on a grid
  this predicate admits — executed, not read off another module's guard;
* that its cell is DISJOINT from the |m| >= 1 complex sibling's, measured off the
  census rather than argued from the names.

THE CORPUS DEMAND, so the suite states what this is worth rather than leaving it to a
docstring: 3 reachable B->H seam-instances of the 387 priced on the 2026-08-19 census,
and the cell has NO attrition — 3 rows drive it, 3 clear the magnetic seam, 3 are
admitted.
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
        / "gate_metal_cylindrical_real_fused_magnetic_pair.py")
#: THE ``b`` DIRECTORY IS THE RELEASE. Its sibling without the suffix is the same
#: gate against the module's PRE-AMENDMENT bytes, kept as the record of that run; a
#: weld binds bytes, so amending the module's docstring moved the file and required a
#: fresh artifact rather than an overwrite.
#: RE-CUT 2026-08-23, fresh directory, by the same rule the 08-20b cut established.
#: The in-seam deposit round delegated this module's source-presence clause to
#: `deposit_repair.seam_source_reasons`, which moved the file, so the 08-20b artifact
#: stopped describing it. The gate was re-run on this host and released PASS against
#: the current bytes. The module's docstring keeps RELEASED 2026-08-20: that is when
#: the product was first certified, and this is the evidence that it still holds.
#: RE-CUT 2026-08-26. The 08-23 artifact stopped binding when this module gained its
#: ``replaces=REPLACES`` registration argument — a HOST-side edit that reaches
#: executable code, so ``weld_survives_edit`` correctly refuses to wave it through.
#: The credit was re-earned by RE-RUNNING THE GATE, not by relaxing the rule: 42 legs
#: PASS, released=True, and the artifact's recorded digest for this module IS the
#: worktree's (d5f7656b2779). The emitted shader text is unchanged by that edit —
#: ``metal_kernels/fingerprints.json``'s ``kernel_source_sha256`` is byte-identical
#: across it — so nothing this artifact measures on the device moved.
#: REPOINTED 2026-09-25 to the ``_2026-09-25_night`` fleet's re-run (released, PASS, 96
#: recorded digests, ZERO disagreeing with the tree, checked before this line moved): the
#: citation re-point edited this module's comments, and the 08-26 artifact records no
#: device or code digest ``weld_survives_edit`` could clear that edit through. The note
#: above describes the 08-26 artifact.
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_cylindrical_real_fused_magnetic_pair_2026-09-25_night"
            / "gate.json")
CENSUS = (REPO / "parity" / "meep_gpu" / "results"
          / "metal_coverage_tranche6_2026-08-19")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, cylindrical_real as cyl,
    cylindrical_real_fused_magnetic_pair as family,
    fused_magnetic_pair as cartesian,
    launch as metal_launch,
    registry, shaders,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS, Residency  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The Dcyl specialisation every admitted grid resolves the r axis to. The r axis is
#: pinned METALLIC (CYL_AXIS shares ``_shift_up``'s metallic zero ghost); z is the one
#: axis this family parameterises.
CODES_METALLIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.METALLIC)
CODES_PERIODIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.PERIODIC)


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

def test_the_cell_has_no_attrition_and_is_worth_three_seam_instances():
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    curl = [r for r in rows if covered(r, "cylindrical_real_curl@step_B")]
    both = [r for r in curl if covered(r, "cylindrical_real_constitutive@update_H")]
    clear = [r for r in both
             if not any(str(kind) == "B"
                        for kind in (r["configuration"].get("source_field_types")
                                     or []))]
    assert (len(curl), len(both), len(clear)) == (3, 3, 3)
    for row in clear:
        configuration = row["configuration"]
        assert configuration["source_field_types"] == ["D"], configuration
        assert configuration["metallic"] == [False, False, True], configuration


def test_this_cell_is_disjoint_from_the_complex_cylindrical_siblings():
    """3 rows against 16, intersection empty — measured, not argued from the names."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the Metal census record is not present ({len(rows)} rows)")
    real = {f"{r['leg']}:{r['row']}" for r in rows
            if covered(r, "cylindrical_real_curl@step_B")}
    complex_arm = {f"{r['leg']}:{r['row']}" for r in rows
                   if covered(r, "cylindrical_complex_pml_curl@step_B")}
    assert len(real) == 3 and len(complex_arm) == 16
    assert real & complex_arm == set()


# ---------------------------------------------------------------------------
# The construction: both halves are the certified emitters' bytes
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_emitters_body_on_both_sides_of_the_splice():
    source = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    curl = family.certified_cyl_curl_body(CODES_METALLIC_Z)
    head, tail = curl.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_constitutive_half_differs_in_exactly_the_three_seam_lines():
    source = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    certified = family.certified_constitutive_body().splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(certified) == len(spliced)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float src{t} = g{t}[ii];", f"    float src{t} = v{t};")
                       for t in range(3)]


def test_the_wall_clear_is_the_B_familys_diagonal_and_not_the_D_sides_complement():
    """Bx clears on an x wall, By on y, Bz on z. Read off the imported table."""
    from meep_gpu.fields import IYEE_SHIFTS
    from meep_gpu.metal_kernels.fused_dispersive_pair import (
        _ZERO_METAL_ROWS as D_ROWS)

    b_rows = {(target, axis) for target, axis, _flag in family._ZERO_METAL_ROWS}
    d_rows = {(target, axis) for target, axis, _flag in D_ROWS}
    assert b_rows == {(0, 0), (1, 1), (2, 2)}
    assert b_rows.isdisjoint(d_rows)
    for target, axis in b_rows:
        assert IYEE_SHIFTS[("Bx", "By", "Bz")[target]][axis] == 0


def test_a_z_walled_source_emits_the_Bz_clear_and_a_periodic_one_emits_none():
    walled = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    periodic = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_PERIODIC_Z, (False, False, False))
    assert "    v2 = at_z ? 0.0f : v2;" in walled
    assert "    v2 = at_z ? 0.0f : v2;" not in periodic
    assert "no walled axis clears a B component" in periodic


def test_the_wall_clear_sits_after_the_axis_rule_and_before_the_store():
    """The driver's own order: step_B ends with the axis rules, then zero_metal_B."""
    source = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    axis_rule = source.index("    v0 = at_x ? 0.0f : v0;")
    clear = source.index("    v2 = at_z ? 0.0f : v2;")
    store = source.index(family._CURL_STORE)
    seam = source.index("float src0 = v0;")
    assert axis_rule < clear < store < seam


def test_a_periodic_r_axis_is_refused_by_name_before_any_splice():
    with pytest.raises(ValueError, match="must compile as METALLIC"):
        family.cylindrical_real_fused_magnetic_pair_source(
            (cyl.PERIODIC, cyl.PERIODIC, cyl.METALLIC), (False, False, True))


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------

def test_the_binding_count_the_module_spells_is_the_count_its_emitter_produces():
    source = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    declared = source.count("[[buffer(")
    assert declared == family.PACKED_BINDINGS
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS
    assert family.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS
    # The refuted signature really carries the count the module argues from.
    assert (family.refuted_separate_scalar_source().count("[[buffer(")
            == family.SEPARATE_SCALAR_BINDINGS)


def test_the_six_scalars_are_unpacked_into_the_certified_bodies_own_names():
    """A struct field read inline would fork the lifted text; it must not."""
    source = family.cylindrical_real_fused_magnetic_pair_source(
        CODES_METALLIC_Z, (False, False, True))
    body = source.split("__BODY__", 1)[0] if "__BODY__" in source else source
    for name in ("nx", "ny", "nz", "n_elem", "dtdx", "axis_coef"):
        assert f"prm.{name}" in source
    # ...and the lifted body uses the bare names, never the struct.
    lifted = source.split("float axis_coef = prm.axis_coef;", 1)[1]
    assert "prm." not in lifted


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def residual(verdict):
    return [reason for reason in verdict.reasons
            if "subnormal policy" not in reason]


def test_a_dcyl_m0_grid_is_admitted_on_both_z_declarations(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    for z_kind, walls in (("metallic", (False, False, True)),
                          ("periodic", (False, False, False))):
        fields, pml = build(z_kind=z_kind)
        verdict = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
            fields, pml, (), Residency())
        assert verdict.covered, (z_kind, list(verdict.reasons))
        assert zero_metal_axes(fields.grid) == walls


def test_a_magnetic_source_is_refused_and_an_electric_one_is_admitted(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    fields, pml = build()
    residency = Residency()
    refused = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (Source("B"),), residency)
    assert not refused.covered
    assert any("is magnetic" in r and "driver.py:3283-3284" in r
               for r in refused.reasons)
    admitted = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"),), residency)
    assert admitted.covered, list(admitted.reasons)


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    fields, pml = build()
    residency = Residency()
    verdict = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, None, residency)
    assert not verdict.covered
    assert any("was not declared" in r for r in verdict.reasons)
    assert family.plan_metal_cylindrical_real_fused_magnetic_pair(
        fields, pml, None, residency) is None


def test_m_not_zero_and_a_cartesian_grid_are_both_refused_by_name(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    fields, pml = build(m=1, complex_storage=True)
    verdict = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (), Residency())
    assert not verdict.covered
    assert any("grid.m = 1" in r for r in verdict.reasons)

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(8.0, 9.0, 6.0), xp=np)
    cart = Fields(grid=grid, force_complex_fields=False)
    cart.enable_pml_storage()
    flat = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        cart, PML(grid=grid, thickness=2), (), Residency())
    assert not flat.covered
    assert any("cylindrical" in r for r in flat.reasons)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    fields, pml = build(m=1, complex_storage=True)
    reasons = family.metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (), Residency()).reasons
    assert any(r.startswith("cylindrical curl half: ") for r in reasons)
    assert any(r.startswith("cylindrical constitutive half: ") for r in reasons)


# ---------------------------------------------------------------------------
# The two omitted passes, EXECUTED
# ---------------------------------------------------------------------------

def test_the_two_omitted_passes_are_inert_on_an_admitted_grid():
    """Executed, not read off another module's guard, with a control that must move."""
    from meep_gpu import stepping

    fields, _pml = build()
    rng = np.random.default_rng(4242)
    names = [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    names += [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    names += [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"]
    for name in names:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-1.0, 1.0, size=array.shape).astype(array.dtype)

    def snapshot():
        return {name: np.array(getattr(fields, name), copy=True) for name in names
                if getattr(fields, name, None) is not None}

    before = snapshot()
    for pass_name in family.INERT_PASSES:
        getattr(stepping, pass_name)(fields)
    after = snapshot()
    assert all(np.array_equal(before[name], after[name]) for name in before), (
        sorted(name for name in before
               if not np.array_equal(before[name], after[name])))
    # THE CONTROL: the pass this family DOES carry must move something on the same
    # grid, or the assertion above is about a frozen state.
    stepping.zero_metal_B(fields)
    control = snapshot()
    assert any(not np.array_equal(after[name], control[name]) for name in after)


def test_REPLACES_and_INERT_PASSES_partition_the_seam():
    assert tuple(family.REPLACES) == ("step_B", "zero_metal_B", "update_H")
    assert set(family.INERT_PASSES) == {"fill_symmetry_bc_B",
                                        "fill_folded_far_ghosts_B"}
    assert set(family.REPLACES).isdisjoint(family.INERT_PASSES)
    assert tuple(family.REPLACES) == tuple(cartesian.REPLACES)


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_UNWIRED_and_carries_its_absorb_row():
    """``wired=False`` and the absorb row answer different questions, and both hold.

    ``wired=False`` keeps the arm out of ``_select_slot``; the absorb row is what lets
    ``_install_fused_pairs`` ASK this product. The row landed 2026-09-17
    (``metal_kernels/launch.py:1236``, in the block headed "THE SIX B->H AND D->E
    ROWS THE COMPOSER WAS STILL MISSING"): before it the family was certified with a
    released weld and ``dispatch_reachability.metal_certified_but_not_installed``
    reported it by name, because the seam loop had no arm pair to absorb. The pair
    is the ``cylindrical m=0`` arm's own two predicates, which is what
    ``metal_cylindrical_real_fused_magnetic_pair_coverage`` calls — the electric
    twin's row, on the B->H seam.

    ASKED, NOT BRACKETED. ``CARRIES_DEPOSIT_REPAIR`` stays False (module :227) and
    the source clause refuses an in-seam magnetic deposit by name
    (``test_a_magnetic_source_is_refused_and_an_electric_one_is_admitted``), so the
    family is declared in ``ROUTED_WITHOUT_THE_REPAIR``
    (``test_metal_fused_pair_deposit_wiring.py:254-276``) rather than absent.
    """
    rows = [row for row in arms.registered() if row.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_B"
    assert rows[0].wired is False
    assert family.FAMILY in registry.FAMILY_MODULES
    assert metal_launch.FUSED_PAIR_ARMS[family.FAMILY] == ("cylindrical m=0",
                                                           "cylindrical m=0")
    assert family.FAMILY not in metal_launch.FUSED_PAIR_EXTRA_ARMS
    assert metal_launch.FUSED_PAIR_SEAMS[family.SLOT] == ("update_H", "B")


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


def _metal_launch_code_outside_the_absorb_rows() -> str:
    """``metal_kernels/launch.py``'s executable text with ``FUSED_PAIR_ARMS`` blanked.

    THE SAME STRUCTURAL CARVE-OUT AS :func:`_fastpath_code_outside_the_pending_reasons`,
    for the same reason. Since 2026-09-17 the absorb table names this family as a
    KEY (``launch.py:1236``) by design, so "launch never names the module" stopped
    being the test of "launch never reaches the module". The table is removed by its
    OWN ASSIGNMENT NODE rather than by matching its text, and everything else in the
    composer is still searched: an import of this module, a call into it, or a second
    table naming it fails exactly as the retired bare-name clause did. Measured
    2026-09-19: the family name occurs once in the stripped text and zero times once
    the one ``FUSED_PAIR_ARMS`` assignment is blanked.
    """
    import ast as _ast

    from meep_gpu.code_identity import strip_docstrings

    tree = strip_docstrings(_ast.parse(
        (PACKAGE_DIR / "metal_kernels" / "launch.py").read_text(encoding="utf-8")))
    blanked = 0
    for node in _ast.walk(tree):
        targets = getattr(node, "targets", None) or (
            [node.target] if isinstance(node, _ast.AnnAssign) else [])
        named = {t.id for t in targets if isinstance(t, _ast.Name)}
        if "FUSED_PAIR_ARMS" in named:
            node.value = _ast.Constant(value="<absorb rows elided>")
            blanked += 1
    # NON-VACUITY: the carve-out must have removed exactly the one table it names,
    # or a renamed or duplicated table would leave the check asking the wrong text.
    assert blanked == 1, f"blanked {blanked} FUSED_PAIR_ARMS assignments in launch.py"
    return _ast.unparse(tree)


def test_fastpath_does_not_name_this_module():
    """DISPATCH must not reach this package — asked per track, not by a name.

    THE BARE NAME STOPPED SEPARATING THE TRACKS ON 2026-09-02, and pretending it
    still does would be the wrong kind of green. The Triton track ships a product
    with the SAME name: ``fastpath.ARM_CERTIFICATION`` now carries
    ``'fused pair B (cylindrical)': ('cylindrical_real_fused_magnetic_pair',
    'triton_cylindrical_real_fused_magnetic_pair_device_gate')``, which is a
    Triton family and its Triton ledger key, and ``PENDING_DEVICE_GATE_ARMS``
    carries Triton gate ARTIFACT PATHS with the same spelling. A substring check
    over ``fastpath.py`` therefore reports the other track's certification table as
    if this Metal module had been wired.

    SO EACH FILE IS ASKED IN ITS OWN VOCABULARY, and neither question is weaker:

    * ``fastpath.py`` must not name the METAL PACKAGE, which is the whole of the
      boundary and is STRONGER than the bare name it replaces: an import, a table
      row or a call reaching any Metal module fails here, not just this one. It is
      the same fact ``dispatch_reachability.backend_reaches_the_dispatch_seam``
      measures off the parse tree on every board cut. Since 2026-09-17 this label
      is RELEASED on the Metal dispatch route — ``metal_dispatch.ARM_CERTIFICATION``
      (``metal_dispatch.py:310-312``), ``METAL_RELEASED_FUSED_ARMS`` (``:645``) and
      ``METAL_FUSED_RELEASE_ARM_AXES`` (``:970``) — and that route runs through the
      ``meep_gpu.metal_dispatch`` sibling, so ``fastpath`` still names no Metal
      module.
    * The Metal-qualified path is checked too, so a future ``fastpath`` that
      imported this module by path fails on its own line rather than only through
      the package clause.
    * ``metal_kernels/launch.py`` — the METAL composer — is checked here too,
      STRUCTURALLY. Since 2026-09-17 it names this family as the key of its absorb
      row (``launch.py:1236``; pinned positively by
      ``test_the_arm_is_registered_UNWIRED_and_carries_its_absorb_row``), which is
      how ``_install_fused_pairs`` asks the product while the arm stays
      ``wired=False``. With that one assignment blanked by its own node, nothing
      else in ``launch.py`` may name the module, and the Metal-only kernel entry
      point may not appear anywhere in it.
    """
    launch = _executable_text(PACKAGE_DIR / "metal_kernels" / "launch.py")
    fastpath = _fastpath_code_outside_the_pending_reasons()
    assert "cylindrical_real_fused_magnetic_pair" not in (
        _metal_launch_code_outside_the_absorb_rows()), (
        "launch.py names this module outside its FUSED_PAIR_ARMS row")
    assert "cyl_real_fused_magnetic_pair_step" not in launch, "launch.py"
    assert _metal_kernels_modules_named_in(fastpath) == [], (
        "fastpath.py names a metal_kernels MODULE; it may reach the Metal table "
        "only through the meep_gpu.metal_dispatch sibling and may read only that "
        "package's fingerprints.json")
    assert "metal_kernels/cylindrical_real_fused_magnetic_pair" not in fastpath
    assert "cyl_real_fused_magnetic_pair_step" not in fastpath, "fastpath.py"


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    tree = ast.parse((PACKAGE_DIR / "metal_kernels"
                      / "cylindrical_real_fused_magnetic_pair.py").read_text(
                          encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "fastpath" not in code
    assert "triton_kernels.launch" not in code or "SUB_STEPS" in code
    assert "cuda_kernels" not in code


# ---------------------------------------------------------------------------
# The gate, and the record it left
# ---------------------------------------------------------------------------

def test_the_gate_exists_and_names_the_product():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_metal_cylindrical_real_fused_magnetic_pair" in source
    assert "cyl_real_fused_magnetic_pair_step" in source


def test_the_released_artifact_is_in_the_tree_and_says_it_released():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["release"]["released"] is True, payload["release"]
    assert payload["steps"] == 12
    assert all(row["passed"] for row in payload["rows"])
    products = [row for row in payload["rows"] if row["leg"] == "product"]
    assert len(products) == 6
    assert all(row["differing_words"] == 0 for row in products)
    # THE FUSION, MEASURED: three dispatches become two, and on a walled run the
    # separate side leaves the wall clear on the host where the fused side does not.
    controls = [row for row in payload["rows"] if row["leg"] == "separate_control"]
    assert controls and all(row["separate_dispatches_per_step"] == 3
                            and row["fused_dispatches_per_step"] == 2
                            for row in controls)
    walled = [row for row in controls if row["label"] == "square_metallic"]
    assert walled and walled[0]["seam_host_passes_for_separate"] == ["zero_metal_B"]
    assert walled[0]["seam_host_passes_for_fused"] == []
    mutations = [row for row in payload["rows"] if row["leg"] == "mutation"]
    assert len(mutations) == 21
    nulls = [row for row in mutations if row.get("expectation") == "null"]
    assert [row["label"] for row in nulls] == [
        "curl_parens_flattened_on_the_invariant_pair"]
    assert nulls[0]["caught"] is False
    assert all(row["caught"] for row in mutations
               if row.get("expectation", "caught") == "caught")


def test_the_module_claims_identity_ONLY_through_the_artifact_that_measured_it():
    """A released gate licenses the claim, and the digest is what binds it."""
    import hashlib

    source = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_real_fused_magnetic_pair.py").read_text(encoding="utf-8")
    assert "RELEASED 2026-08-20" in source
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    module = (PACKAGE_DIR / "metal_kernels"
              / "cylindrical_real_fused_magnetic_pair.py")
    live = hashlib.sha256(module.read_bytes()).hexdigest()
    # ``metal_gate_runner`` REPLACES the gate's own ``source_sha256`` with the map of
    # every repo module the process imported, keyed by ABSOLUTE path. That is the
    # stronger record — it binds the whole imported tree and not a hand-picked pair —
    # so the lookup is by path rather than by the gate's own "family" label.
    recorded = {key: value for key, value in payload["source_sha256"].items()
                if key.endswith("metal_kernels/cylindrical_real_fused_magnetic_pair.py")}
    assert recorded, sorted(payload["source_sha256"])[:5]
    for key, digest in sorted(recorded.items()):
        if live == digest:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209). This module emits its
        # shader text, so its device identity is the emitted corpus rather than
        # its bytes: a prose edit here cannot change a character of what the MPS
        # compiler received. The helper returns False for anything it cannot
        # establish, so the byte rule below stands unless the artifact carries a
        # device_sha256/code_sha256 map that says otherwise.
        if weld_survives_edit(module, payload, key):
            continue
        assert live == digest, (
            "the module has changed since the gate ran in a way that reaches "
            "executable code; the identity claim in its docstring no longer "
            "describes the bytes that were measured")
    assert "NOT WIRED" in source
