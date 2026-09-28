"""The two COMPLEX off-diagonal SCRATCH-OUTPUT welds, at the merge bar.

WHAT THIS SUITE IS FOR. :mod:`.folded_complex_offdiag_fused_electric_pair` and
:mod:`.complex_no_pml_offdiag_fused_electric_pair` occupy the last two board cells
the hand-CUDA fusion matrix recorded UNBUILDABLE. Neither can be launched here --
there is no CuPy on the merge-bar host -- so what is asserted is everything that
decides a float BEFORE a launch: the lift's anchors, the emitted text's shape, the
predicate's clauses, the scratch discipline, the rotation and the composer wiring.
The device verdict is ``gate_cuda_complex_offdiag_stencil_welds.py``'s and the
arithmetic proof is ``probe_cuda_complex_offdiag_scratch_weld.py``'s; this suite is
what stops either from being reached with a module that already disagrees with
itself.

THE THREE THINGS MOST WORTH PINNING HERE, because each fails SILENTLY:

1. **the constitutive body must carry no volume name this launch does not bind.**
   The flux density is bound ONCE, as the curl's own ``f0``/``f1``/``f2``; a
   surviving ``Dx`` would either fail to compile or -- worse, after a later
   signature edit -- bind a volume the launch is not writing;
2. **the folded family reads its OWN boundary triple and the unfolded one reads a
   cross-checked single triple.** On a fold the curl and the constitutive read the
   same axis differently by design, and a body left reading the curl's would apply
   the wrong ghost rule to a whole plane;
3. **the certified samplers are carried and NOT called.** ``up_sample`` and
   ``down_sample`` ride through the prelude as dead code on purpose -- forking a
   certified prelude to drop them is the silent divergence indirection exists to
   prevent -- but "dead" is only evidence while nothing calls them, so it is
   measured rather than asserted in prose.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Tuple

import pytest

from . import complex_emitter
from . import complex_offdiag_stencil_weld as weld
from . import complex_offdiag_update_e as offdiag
from . import complex_no_pml_offdiag_fused_electric_pair as no_pml_pair
from . import folded_complex_offdiag_fused_electric_pair as folded_pair
from .. import deposit_repair

#: The two families, with the fixed facts each declares. Parametrised as a table so
#: a third arm cannot be added without deciding every one of these.
FAMILIES: Tuple[Tuple[str, Any, str, str, int, int], ...] = (
    ("folded", folded_pair, "pml", "cbc_", 6, 5),
    ("no_pml", no_pml_pair, "no_pml", "bc_", 3, 3),
)

_ARMS = tuple(sorted(complex_emitter.EXPANSIONS))
_MASKS = ((1, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 1, 0),
          (0, 0, 0, 0, 0, 1))


def code_only(source: str) -> str:
    """``source`` with every ``//`` comment removed.

    The certified emitters label every helper and every component block with the
    name and the axis they transcribe, and those comments are the audit trail this
    track most wants to keep. A token check that read them would either fire on
    every emission or force the comments out; this asks the question it means --
    what does the COMPILED text say.
    """
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def _sources(module: Any) -> Dict[Tuple[Tuple[int, ...], str], str]:
    return {(mask, arm): module.kernel_source(mask, arm)
            for mask in _MASKS for arm in _ARMS}


# ---------------------------------------------------------------------------
# The emitted text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_declared_facts_agree_with_the_module(label, module, arm, prefix,
                                                  scratch, replaces):
    """The six constants a reader and the board both rely on, checked against each
    other rather than against this file's memory of them."""
    assert module.ARM == arm
    assert len(module.SCRATCH_VOLUMES) == scratch
    assert len(module.REPLACES) == replaces
    assert module.SLOT == "step_D"
    assert module.REPLACES[0] == "step_D" and module.REPLACES[-1] == "update_E"
    assert module.CARRIES_DEPOSIT_REPAIR is False
    # The kernel name is the ONE symbol NVRTC is asked for and the one the
    # partition tests key on; it must be this module's own and nobody else's.
    #
    # RELEASED 2026-09-02: both names moved to CERTIFIED_KERNELS when
    # ``cuda_complex_offdiag_stencil_welds_2026-09-02`` landed in
    # ``certification.json`` with both policy legs released, and this assertion
    # moved with them in the same edit. ``test_kernel_partition.py`` is what holds
    # the declaration to the record; what this pins is that the partition names
    # THIS module's kernel and nothing else.
    assert module.KERNEL_NAME in module.CERTIFIED_KERNELS
    assert module.UNCERTIFIED_KERNELS == {}
    assert set(module.CERTIFIED_KERNELS) == {module.KERNEL_NAME}


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_every_emitted_source_is_one_pure_ascii_kernel(label, module, arm, prefix,
                                                       scratch, replaces):
    """One entry point, this family's name, pure ASCII, no placeholder left.

    ASCII IS A COMPILE REQUIREMENT rather than a style rule: CuPy writes the NVRTC
    source through a bare ``open(..., 'w')``, so the bytes go through the
    interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
    non-interactive shell on the validation host gets. Two em-dashes killed a
    sibling kernel at its first launch on 2026-08-15.
    """
    for (mask, expansion), source in _sources(module).items():
        entries = re.findall(r'extern "C" __global__ void (\w+)\(', source)
        assert entries == [module.KERNEL_NAME], (mask, expansion, entries)
        source.encode("ascii")
        stripped = source
        for token in ("__restrict__", "__global__", "__device__", "__forceinline__",
                      "__fmaf_rn"):
            stripped = stripped.replace(token, "")
        assert "__" not in stripped, (mask, expansion)


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_no_device_helper_or_define_is_emitted_twice(label, module, arm, prefix,
                                                     scratch, replaces):
    """The whole reason the two certified preludes are spliced in pieces.

    Both halves are built on ``complex_emitter``'s shared head, so concatenating the
    two preludes whole would redefine every ``cf`` helper; and the constitutive's own
    prelude repeats the curl's two boundary defines. A duplicate ``__device__``
    definition does not compile and a duplicate ``#define`` is diagnosed only as a
    warning, so both are checked here where the failure is a named assertion.
    """
    for (mask, expansion), source in _sources(module).items():
        helpers = re.findall(r'__device__ __forceinline__ \w+ (\w+)\(', source)
        repeated = [name for name, count in collections.Counter(helpers).items()
                    if count > 1]
        assert not repeated, (mask, expansion, repeated)
        defines = re.findall(r"#define (\w+)", source)
        repeated = [name for name, count in collections.Counter(defines).items()
                    if count > 1]
        assert not repeated, (mask, expansion, repeated)
        # BC_MIRROR and MIRROR_ROW are the constitutive's alone and must survive;
        # the curl's two must appear exactly once between the two preludes.
        assert set(defines) >= {"BC_PERIODIC", "BC_METALLIC", "BC_MIRROR",
                                "MIRROR_ROW", "NEAR_SOURCE_ROW"}, (mask, expansion)


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_splice_order_puts_every_definition_before_its_first_use(
        label, module, arm, prefix, scratch, replaces):
    """THE ORDER IS FORCED, and this is where that is a measurement.

    ``resolved_down_sample`` reads ``BC_MIRROR``; ``offdiag_term`` calls
    ``resolved_down_sample``; ``resolve_D_at`` calls ``raw_step_D_cell``, which needs
    ``WeldArgs``. A splice that put the constitutive prelude whole after the
    resolution would compile the first two backwards and fail at NVRTC on the
    validation host rather than here.
    """
    for (mask, expansion), source in _sources(module).items():
        def at(needle: str) -> int:
            index = source.find(needle)
            assert index >= 0, (mask, expansion, needle)
            return index

        assert at("#define BC_MIRROR") < at(
            "__device__ __forceinline__ cf resolved_down_sample(")
        assert at("struct WeldArgs {") < at(
            "__device__ __forceinline__ void raw_step_D_cell(")
        assert at("__device__ __forceinline__ void raw_step_D_cell(") < at(
            "__device__ __forceinline__ cf resolve_Dx(")
        assert at("__device__ __forceinline__ cf resolved_down_sample(") < at(
            "__device__ __forceinline__ cf offdiag_term(")
        assert at("__device__ __forceinline__ cf offdiag_term(") < at(
            'extern "C" __global__ void ')


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_certified_samplers_are_carried_and_never_called(
        label, module, arm, prefix, scratch, replaces):
    """Dead certified code is EVIDENCE only while nothing calls it.

    ``up_sample`` and ``down_sample`` are lifted whole from the certified
    constitutive prelude rather than dropped, on ``complex_offdiag_update_e``'s own
    precedent: forking a prelude to remove dead helpers is the silent divergence
    indirection exists to prevent. What makes carrying them safe is that the weld
    routes every sample through the RESOLVED forms instead, and the device gate arms
    a mutation of each as a MUST-BE-UNCAUGHT null -- which is only a valid null while
    this holds.
    """
    for (mask, expansion), source in _sources(module).items():
        for name in ("up_sample", "down_sample"):
            declaration = f"__device__ __forceinline__ cf {name}("
            assert source.count(declaration) == 1, (mask, expansion, name)
            calls = re.findall(rf"(?<![A-Za-z0-9_]){name}\(", code_only(source))
            # ONE occurrence: the declaration itself. Every other spelling in the
            # compiled text is `resolved_{name}(`, which the negative lookbehind on
            # `_` excludes; the certified comments that NAME the helper are stripped
            # by code_only, because the claim is about calls and not about prose.
            assert len(calls) == 1, (mask, expansion, name, len(calls))


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_constitutive_body_binds_no_volume_the_launch_does_not(
        label, module, arm, prefix, scratch, replaces):
    """No ``Dx``/``Dy``/``Dz`` survives outside a comment, on any row mask.

    THE SEAM IS EXACTLY THIS. The certified constitutive opens each component with
    ``cf_load(D*, idx)`` and hands each term its partner VOLUME; the weld replaces
    the first with the register this thread resolved and the second with a component
    TAG plus that partner's register. A surviving name is an unbound identifier at
    best and a second binding of an unwritten volume at worst.
    """
    for mask in _MASKS:
        for expansion in _ARMS:
            body = weld.constitutive_body(module.ARM, mask, expansion,
                                          boundary_prefix=prefix)
            code = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
            assert not re.search(r"(?<![A-Za-z0-9_])D[xyz](?![A-Za-z0-9_])", code), (
                mask, expansion)
            assert re.search(r"(?<![A-Za-z0-9_])v_x(?![A-Za-z0-9_])", code)
            assert "weld);" in body, (mask, expansion)


def test_the_folded_family_reads_its_own_boundary_triple():
    """THE ONE EDIT THE UNFOLDED SIBLING DOES NOT MAKE, and the count is pinned.

    Every read of the boundary code moves together -- both neighbour-coordinate
    lines, both up-wrap predicates and every term's ``dbc`` argument -- because the
    certified body derives the ghost lane and the parity weight from ONE triple on
    purpose. Renaming some and not others would be exactly the disagreement that
    comment exists to prevent.
    """
    for mask in _MASKS:
        for expansion in _ARMS:
            plain = weld.constitutive_body("pml", mask, expansion,
                                           boundary_prefix="bc_")
            renamed = weld.constitutive_body("pml", mask, expansion,
                                             boundary_prefix="cbc_")
            before = len(re.findall(r"(?<![A-Za-z0-9_])bc_[xyz](?![A-Za-z0-9_])",
                                    plain))
            after = len(re.findall(r"(?<![A-Za-z0-9_])cbc_[xyz](?![A-Za-z0-9_])",
                                   renamed))
            assert before > 0 and before == after, (mask, expansion, before, after)
            assert not re.search(r"(?<![A-Za-z0-9_])bc_[xyz](?![A-Za-z0-9_])",
                                 "\n".join(line.split("//", 1)[0]
                                           for line in renamed.splitlines()))


def test_the_pml_arm_renames_only_the_half_integer_kms():
    """ONE RENAME, NO ARITHMETIC -- and the integer triple is left alone.

    ``kms_*`` is the INTEGER sub-lattice for the D curl and the HALF-INTEGER one for
    ``update_E``. In one scope the bare name collides, and letting one shadow the
    other is half a cell in the absorber profile rather than a compile failure.
    """
    body = weld.constitutive_body("pml", (1, 1, 1, 1, 1, 1), "NAIVE",
                                  boundary_prefix="cbc_")
    assert body.count("kms_half_x[i]") == 1
    assert body.count("kms_half_y[j]") == 1
    assert body.count("kms_half_z[k]") == 1
    assert not re.search(r"(?<![A-Za-z0-9_])kms_[xyz]\[", body)
    # The store arm has no tail coefficients at all, which is a predicate clause
    # rather than an omission.
    store = weld.constitutive_body("no_pml", (1, 1, 1, 1, 1, 1), "NAIVE")
    assert "constitutive_apply(" not in store
    assert not re.search(r"(?<![A-Za-z0-9_])k[mp]s_", store)


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_scratch_group_is_a_separate_binding_group(label, module, arm, prefix,
                                                       scratch, replaces):
    """The design, read off the signature: an ``_out`` per scratch volume, and the
    pre-launch group bound ``const``.

    If a scratch output were the storage the launch would be the IN-PLACE weld the
    board refused, and every foreign recompute would read words other blocks had
    already overwritten. ``assert_scratch_is_disjoint`` checks it at run time by base
    address; this checks that the SIGNATURE has somewhere to put a separate one.
    """
    source = module.kernel_source((1, 1, 1, 1, 1, 1), "NAIVE")
    signature = code_only(
        source.split('extern "C" __global__ void ', 1)[1].split("\n) {\n")[0])
    outs = re.findall(r"float\* __restrict__ (\w+_out)", signature)
    assert len(outs) == scratch, (outs, scratch)
    for name in ("f0", "f1", "f2", "g0", "g1", "g2"):
        assert f"const float* __restrict__ {name}," in signature or \
               f"const float* __restrict__ {name}\n" in signature, name
    if arm == "pml":
        assert set(outs) == {"f0_out", "f1_out", "f2_out",
                             "u0_out", "u1_out", "u2_out"}
        for name in ("u0", "u1", "u2"):
            assert f"const float* __restrict__ {name}" in signature
    else:
        assert set(outs) == {"f0_out", "f1_out", "f2_out"}
        # NO auxiliary and NO constitutive history on this arm.
        assert "f_w_E" not in signature
        assert not re.search(r"(?<![A-Za-z0-9_])u[012](?![A-Za-z0-9_])", signature)


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_rotation_swaps_exactly_the_scratch_volumes(label, module, arm, prefix,
                                                        scratch, replaces):
    """``rotate_into_fields`` hands back the RETIRED volumes and installs the new.

    A rotation that returned the new ones would hand the caller the live buffers to
    overwrite on the next launch, which is the in-place weld arriving by the back
    door.
    """
    class Bag:
        pass

    fields = Bag()
    before = {name: object() for name in module.SCRATCH_VOLUMES}
    fresh = {name: object() for name in module.SCRATCH_VOLUMES}
    for name, value in before.items():
        setattr(fields, name, value)
    retired = module.rotate_into_fields(fields, fresh)
    assert retired == before
    assert {name: getattr(fields, name) for name in module.SCRATCH_VOLUMES} == fresh


# ---------------------------------------------------------------------------
# The lift's own declarations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_lift_edits_are_this_arm_s_and_are_shaped(label, module, arm, prefix,
                                                  scratch, replaces):
    """Every edit carries a line, what it became and WHY, and the arm's own set is
    the one declared.

    The list is DATA rather than prose precisely so a gate and this suite can assert
    it; an edit list that drifted from the lift it describes would be a module
    documenting a kernel it does not emit.
    """
    declared = module.LIFT_EDITS
    assert len(declared) >= len(weld.curl_lift_edits(arm))
    for entry in declared:
        assert set(entry) == {"line", "became", "why"}, entry
        for key, value in entry.items():
            assert isinstance(value, str) and value.strip(), (key, entry)
    for entry in weld.curl_lift_edits(arm):
        assert entry in declared, entry
    for entry in weld.RESOLUTION_LIFT_EDITS:
        assert entry in declared, entry


def test_the_two_arms_declare_different_curl_edits():
    """The split-field lift and the plain subtraction are DIFFERENT edits.

    A shared table that gave both arms the same list would mean one of them was
    describing the other's certified helper, which is the kind of drift the whole
    edit-list mechanism exists to make visible.
    """
    pml = weld.CURL_LIFT_EDITS["pml"]
    no_pml = weld.CURL_LIFT_EDITS["no_pml"]
    assert pml != no_pml
    assert any("pml_apply_pure" in entry["became"] for entry in pml)
    assert any("no_pml_apply_pure" in entry["became"] for entry in no_pml)
    assert not any("fu_out" in entry["became"] for entry in no_pml)
    with pytest.raises(ValueError):
        weld.normalized_arm("split_field")


def test_the_shared_lift_refuses_a_moved_anchor_by_name():
    """A certified line that moved is a NAMED failure, never a silent splice.

    Measured by handing the splicer text whose anchor is absent: the whole value of
    anchoring on certified spellings is that an upstream edit stops this build rather
    than quietly producing a kernel that is not the certified arithmetic.
    """
    with pytest.raises(AssertionError):
        weld.the_line_starting("int x = 1;\n", "    cf gs_Ex = ", "a moved anchor")
    with pytest.raises(AssertionError):
        weld.close_call_with_the_pack("nothing here\n", "    cf term_Ex_0 = ",
                                      "a moved call")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def _folded_complex_fixture():
    """A folded complex grid with an off-diagonal row -- the folded cell's shape."""
    import numpy as np  # noqa: PLC0415 - fixture only

    from ..fields import Fields  # noqa: PLC0415
    from ..grid import Grid, Mirror  # noqa: PLC0415
    from ..pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, symmetry=(Mirror("Y", 1),), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = tuple(grid.shape)
    names = ("Ex", "Ey", "Ez")
    fields.set_epsilon_volumes(
        {n: np.full(shape, v, np.float32) for n, v in zip(names, (2.0, 2.5, 3.0))},
        {n: np.full(shape, np.float32(1.0 / v), np.float32)
         for n, v in zip(names, (2.0, 2.5, 3.0))},
        chi1inv_offdiagonal={"Ex": {"Ey": np.full(shape, np.float32(0.25))}})
    fields.enable_pml_storage()
    # PER AXIS, and per FACE. A two-cell layer does not fit on a zero-extent z, and
    # the folded Y's LOW face is the mirror plane -- a boundary condition, not an
    # absorber -- so the engine refuses a layer there by name.
    return fields, PML(grid=grid,
                       thickness={"x": 2, "y": {"high": 2}, "z": 0})


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_predicate_refuses_a_missing_licence_by_name(
        label, module, arm, prefix, scratch, replaces):
    """No licence is a REFUSAL, not a default -- asked on a grid that gets that far.

    The complex multiply's arm is a MEASURED PLATFORM FACT; a predicate that
    defaulted one would license a binary nobody classified. Asked on a real folded
    complex grid rather than a bare object, so the refusal that comes back is the
    licence clause and not an earlier structural one -- which is what makes this a
    statement about the licence rather than about the fixture.
    """
    from ..cuda_kernels.coverage import (  # noqa: PLC0415
        complex_expansion_refusal,
    )

    refusal = complex_expansion_refusal(None, "keep")
    assert refusal and ("licence" in refusal or "license" in refusal)
    predicate = getattr(module, f"covers_{module.FAMILY[5:]}")
    fields, pml = _folded_complex_fixture()
    covered, reason = predicate(fields, pml, fields.grid, ())
    assert covered is False
    if label == "folded":
        assert "licence" in reason or "license" in reason, reason
    else:
        # The no-absorber family refuses this fixture on its ACTIVE LAYER first,
        # which is the earlier and more fundamental clause; its licence clause is
        # the same shared one asserted above.
        assert reason.startswith("curl half: "), reason


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_predicate_is_a_conjunction_that_opens_on_its_curl_half(
        label, module, arm, prefix, scratch, replaces):
    """The first refusal is the CURL half's, prefixed, and nothing is weakened.

    Asked with a licence-shaped object that is not a verdict, so the refusal comes
    from the certified half rather than from this product's own clauses -- which is
    what makes the prefix a proof that the half was asked at all.
    """
    predicate = getattr(module, f"covers_{module.FAMILY[5:]}")
    covered, reason = predicate(object(), None, object(), (), "not-a-verdict",
                                "keep")
    assert covered is False
    assert reason.startswith("curl half: "), reason


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_carries_deposit_repair_is_the_repair_module_s_own_answer(
        label, module, arm, prefix, scratch, replaces):
    """FALSE, and MEASURED rather than chosen.

    ``deposit_repair.repairable`` refuses an off-diagonal chi1inv row BY NAME: a
    point repair recomputes E at the deposit cell from THAT cell's displacement,
    while this constitutive reads its NEIGHBOURS'. Every row of both cells carries
    one by construction, because the constitutive arms REQUIRE a surviving
    off-diagonal row -- so the flag is not a policy choice and this asserts the
    refusal exists rather than asserting the constant against itself.
    """
    assert module.CARRIES_DEPOSIT_REPAIR is False
    text = deposit_repair.__doc__ or ""
    source = deposit_repair.repairable.__doc__ or ""
    assert "off-diagonal" in (text + source), (
        "deposit_repair no longer names the off-diagonal refusal this flag rests on")


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_an_undeclared_source_set_is_refused_rather_than_assumed_empty(
        label, module, arm, prefix, scratch, replaces):
    """The seam clause's own refusal string, checked for the word that matters.

    A predicate that inferred an empty electric source seam from ``Fields`` would
    serve rows whose deposit lands between the two halves.
    """
    reasons = deposit_repair.seam_source_reasons(
        object(), None, "D", undeclared="the source set was not declared",
        refusal=lambda index, source: "electric", carries_repair=False)
    assert reasons and "declared" in reasons[0]


# ---------------------------------------------------------------------------
# The composer wiring
# ---------------------------------------------------------------------------

def test_both_products_are_installed_with_the_arms_their_predicates_conjoin():
    """The arm-table row, the product row and the board's cell key must agree.

    ``fused_pairs.FUSED_PAIR_ARMS`` is what lets a pair ABSORB two slots, and the row
    must name the labels the composer already selected -- so a row that named
    anything else would be a product that can never install.
    """
    from . import fused_pairs
    from .registry import _TABLE

    labels = {(row["family"], row["label"]) for row in _TABLE}
    expected = {
        "cuda_folded_complex_offdiag_fused_electric_pair":
            (("cuda_complex_folded", "folded complex"),
             ("cuda_complex_offdiag", "complex off-diagonal PML")),
        "cuda_complex_no_pml_offdiag_fused_electric_pair":
            (("cuda_complex_no_pml", "complex no-PML curl"),
             ("cuda_complex_offdiag", "complex off-diagonal no-PML")),
    }
    for family, (curl, constitutive) in expected.items():
        assert family in fused_pairs.FUSED_PAIR_ARMS, family
        assert fused_pairs.FUSED_PAIR_ARMS[family] == (curl[1], constitutive[1])
        assert curl in labels and constitutive in labels, family
        assert family in fused_pairs.FUSED_PRODUCTS, family
        assert fused_pairs.FUSED_PRODUCTS[family]["curl_slot"] == "step_D"


def test_neither_product_bids_at_a_slot():
    """A fused pair holds two slots and bids at none.

    ``registry`` carries an explicit exclusion entry per fused predicate; a product
    missing from it would be read as a slot arm and double-counted against the two
    arms it absorbs.
    """
    from .registry import NOT_REGISTERED

    for key in ("folded_complex_offdiag_fused_electric_pair."
                "covers_folded_complex_offdiag_fused_electric_pair",
                "complex_no_pml_offdiag_fused_electric_pair."
                "covers_complex_no_pml_offdiag_fused_electric_pair"):
        assert key in NOT_REGISTERED, key
        assert NOT_REGISTERED[key].strip()


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_the_corpus_digest_is_stable_and_covers_both_expansion_arms(
        label, module, arm, prefix, scratch, replaces):
    """One sha256 over every source this family can emit.

    Stability is the point: the digest is what a certification record binds, so a
    value that moved between two calls in one process would make every record
    unfalsifiable.
    """
    first = module.corpus_digest()
    assert first == module.corpus_digest()
    assert len(first) == 64
    naive = module.kernel_source((1, 1, 1, 1, 1, 1), "NAIVE")
    fma = module.kernel_source((1, 1, 1, 1, 1, 1), "FMA_V1")
    assert naive != fma, "the two expansion arms emit the same text"


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_every_live_row_mask_emits(label, module, arm, prefix, scratch, replaces):
    """All 63 masks, one arm, so a mask that lost its anchor is caught here.

    The certified emitter binds only the LIVE coefficients, so each mask is a
    different signature and a different set of term call sites for the seam rewrite
    to find.
    """
    for mask in offdiag.LIVE_ROW_MASKS:
        source = module.kernel_source(mask, "NAIVE")
        live = sum(1 for flag in mask if flag)
        bound = len(re.findall(r"const float\* chi1inv_E[xyz]_E[xyz],", source))
        assert bound == live, (mask, bound, live)
        assert source.count("cf term_") == live, mask


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_an_unknown_row_mask_or_arm_is_refused_by_name(label, module, arm, prefix,
                                                       scratch, replaces):
    """A wrong arm is a wrong ANSWER rather than a crash, so it is never defaulted."""
    with pytest.raises(ValueError):
        module.kernel_source((0, 0, 0, 0, 0, 0), "NAIVE")
    with pytest.raises((ValueError, KeyError)):
        module.kernel_source((1, 1, 1, 1, 1, 1), "NOT_AN_ARM")


# ---------------------------------------------------------------------------
# The bindings, on a grid each family's OWN predicate gets structurally past
# ---------------------------------------------------------------------------

def _no_pml_complex_fixture():
    """A complex no-absorber grid with an off-diagonal row -- the no-PML cell's shape.

    NO ``enable_pml_storage``, and that is the family's own clause rather than a
    fixture preference: the complex no-absorber curl refuses a ``Fields`` in PML
    storage mode beside an inert layer BY NAME. What it leaves is the storage
    :data:`complex_no_pml_offdiag_fused_electric_pair._FIELD_BINDINGS` must be
    spelled against -- E and B allocated, H and every auxiliary ``None``.
    """
    import numpy as np  # noqa: PLC0415 - fixture only

    from ..fields import Fields  # noqa: PLC0415
    from ..grid import Grid  # noqa: PLC0415
    from ..pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                courant=0.35, xp=np, k_point=(0.3892, 0.1597, 0.0))
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = tuple(grid.shape)
    names = ("Ex", "Ey", "Ez")
    fields.set_epsilon_volumes(
        {n: np.full(shape, v, np.float32) for n, v in zip(names, (2.0, 2.5, 3.0))},
        {n: np.full(shape, np.float32(1.0 / v), np.float32)
         for n, v in zip(names, (2.0, 2.5, 3.0))},
        chi1inv_offdiagonal={"Ex": {"Ey": np.full(shape, np.float32(0.25))}})
    fields.enable_field_storage()
    return fields, PML(grid=grid, thickness=0)


_ADMITTED = {"folded": _folded_complex_fixture, "no_pml": _no_pml_complex_fixture}


@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_every_bound_volume_is_allocated_on_a_grid_this_family_serves(
        label, module, arm, prefix, scratch, replaces):
    """THE LAUNCH'S NINE-OR-FIFTEEN POINTERS RESOLVE, on this family's own storage.

    THIS IS THE TEST THAT WAS MISSING. ``_FIELD_BINDINGS`` is a list of ATTRIBUTE
    NAMES handed to ``word_view`` one by one, and every other check in this file
    reads the emitted TEXT -- which is identical whether the name resolves to an
    array or to ``None``. The no-absorber family had ``Hx``/``Hy``/``Hz`` here,
    and ``Fields.enable_field_storage`` deliberately does not allocate H without an
    absorber ("update_H writes nothing without PML, so a stored H would sit at zero
    for the whole run while get_H returned it", fields.py:653-668) -- so all three
    were ``None`` on every row the predicate admits and the first launch would have
    died inside ``word_view``. A device gate cannot report that as anything but a
    crash, and this file is where it is cheap to see.

    Asked on a fixture the family's own predicate gets STRUCTURALLY past -- the only
    refusal left is the expansion licence, which is a platform verdict this suite
    has no device to cut -- so the storage under test is the storage a real launch
    would meet.
    """
    fields, pml = _ADMITTED[label]()
    predicate = getattr(module, f"covers_{module.FAMILY[5:]}")
    covered, reason = predicate(fields, pml, fields.grid, ())
    assert covered is False and ("licence" in reason or "license" in reason), (
        f"{label}: this fixture is meant to clear every structural clause and stop "
        f"at the licence; it stopped at {reason!r}")

    shape = tuple(int(n) for n in fields.grid.shape)
    missing = [name for name in module._FIELD_BINDINGS  # noqa: SLF001
               if getattr(fields, name, None) is None]
    assert not missing, (
        f"{label} binds {missing} but this family's own storage leaves them "
        f"unallocated; word_view would be handed None at the first launch")
    for name in module._FIELD_BINDINGS:  # noqa: SLF001
        volume = getattr(fields, name)
        assert tuple(int(n) for n in volume.shape) == shape, (label, name)
        assert volume.dtype.kind == "c", (
            f"{label}/{name} is {volume.dtype}; this launch binds complex64 volumes "
            f"as their float32 word views and a real volume would halve every index")
    assert len(module._FIELD_BINDINGS) == 15 if label == "folded" else 9  # noqa: SLF001


@pytest.mark.requires_resource("cupy")
@pytest.mark.parametrize("label,module,arm,prefix,scratch,replaces", FAMILIES)
def test_a_fresh_scratch_is_disjoint_from_every_binding(label, module, arm, prefix,
                                                        scratch, replaces):
    """The scratch builder's volumes alias nothing the launch binds.

    The same nine-or-fifteen names, asked the way the launcher asks them: a
    ``ValueError`` here is the in-place weld being refused, and a pass is the
    scratch discipline holding on a real allocation rather than on the docstring.

    DEVICE-RESOURCED, and only this one: the scratch builder allocates through CuPy
    and ``assert_scratch_is_disjoint`` compares ``.data.ptr``, which NumPy has no
    answer for. The BINDING test above needs no device and is where the storage
    question is settled on the laptop.
    """
    pytest.importorskip("cupy",
                        reason="[requires_resource][cupy] the scratch builder "
                               "allocates on the device and the disjointness check "
                               "compares device base addresses")
    fields, pml = _ADMITTED[label]()
    fresh = getattr(module, f"{module.FAMILY[5:]}_scratch")(fields)
    assert sorted(fresh) == sorted(module.SCRATCH_VOLUMES)
    for name, volume in fresh.items():
        assert volume is not getattr(fields, name)
        assert tuple(int(n) for n in volume.shape) == tuple(
            int(n) for n in getattr(fields, name).shape)
    if label == "folded":
        tables = module.folded_complex_offdiag_fused_electric_pair_tables(pml)
        module.assert_scratch_is_disjoint(fields, fresh, tables)
    else:
        module.assert_scratch_is_disjoint(fields, fresh)
    aliased = {name: getattr(fields, name) for name in module.SCRATCH_VOLUMES}
    with pytest.raises(ValueError):
        if label == "folded":
            module.assert_scratch_is_disjoint(fields, aliased, tables)
        else:
            module.assert_scratch_is_disjoint(fields, aliased)
