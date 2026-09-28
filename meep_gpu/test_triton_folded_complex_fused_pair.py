"""Tests for the FOLDED COMPLEX fused ELECTRIC pair — folded Bloch ``step_D`` into
``update_E``, with both mirror fills and the wall clear carried inline.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the separately certified products
this launch replaces — lives in
``parity/meep_gpu/probe_triton_folded_complex_fused_pair.py``. Whether that gate has
RUN is a fact this suite reads out of the module's own
:data:`~.folded_complex_fused_pair.DEVICE_STATUS` rather than assuming: a green laptop
suite over an unrun device gate must never be mistaken for a certification.

WHAT IS PINNED HERE, and the three things that are genuinely NEW against the two
products this one sits between:

* **THE DEPOSIT FLAG IS THE WHOLE CELL.** Both corpus rows declare an ELECTRIC
  source, so :data:`~.folded_complex_fused_pair.CARRIES_DEPOSIT_REPAIR` at False
  would take this product from two seam-instances to zero. That is asserted
  DIRECTLY, by flipping the flag and re-asking the predicate, rather than restated
  in prose;
* **THE D FILL GEOMETRY**, which is the exact inverse of the magnetic twin's: the
  near fill takes the TWO axes that are not the component's own and the far fill
  takes its own, so the coefficient index moves for the FAR half here and for the
  NEAR half there;
* **THE WALL CLEAR SITS INSIDE THE PARITY CHAIN**, which it never does on the B
  side, because on the D family ``_zero_metal``'s rows ARE the near fill's axes.

and, carried over from the magnetic twin because they are equally load-bearing here:

* the parity chain: one ``_mul_imag_coefficient_left`` per pass, in the driver's own
  order, read OFF THE SHIPPED KERNEL SOURCE and compared against
  :func:`~.folded_complex_fused_pair.parity_chain`;
* the ownership enumeration: every emitted block's guard, destination offset, mask,
  inverse-epsilon binding and coefficient row, against
  :func:`~.folded_complex_fused_pair.carried_destinations`;
* every clause of the seam predicate, in both directions;
* the transcription: each region of the kernel against the certified module it names,
  by source text rather than by re-derivation;
* the declaration/kernel agreement — ``REPLACES`` against what the body actually
  carries, which is the failure a sibling module shipped for a day;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu.fields import IYEE_SHIFTS, Fields, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import folded_complex
from meep_gpu.triton_kernels import folded_complex_fused_pair as product
from meep_gpu.triton_kernels.coverage import zero_metal_axes
from meep_gpu.test_triton_complex_fields import stamp_probe_record

# Every probe record this module builds is stamped 'keep'; the complex families'
# policy clause fails closed when the run policy can be neither read nor declared,
# so the premise is declared rather than left implicit.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")

MODULE_PATH = pathlib.Path(product.__file__)
PACKAGE_DIR = MODULE_PATH.parent
API_ROOT = PACKAGE_DIR.parents[1]
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_folded_complex_fused_pair.py"

KERNEL_NAME = "folded_complex_fused_curl_constitutive_D"
CARRY_HELPER = "_carry_ghost_complex_E"

#: Which kernel argument carries which (axis, pass) coefficient. This table is the
#: kernel SIGNATURE's own naming and is asserted against it below, so a renamed
#: argument fails here rather than silently making the chain tests vacuous.
COEFFICIENT_ARGUMENTS = {
    "n0r": (0, "near"), "n1r": (1, "near"), "n2r": (2, "near"),
    "d0r": (0, "far"), "d1r": (1, "far"), "d2r": (2, "far"),
}

#: Per component: the offset name, lane predicate and clear flag each axis uses.
DN = {0: "dn_x", 1: "dn_y", 2: "dn_z"}
DF = {0: "df_x", 1: "df_y", 2: "df_z"}
NEAR_LANE = {0: "near_i", 1: "near_j", 2: "near_k"}
FAR_LANE = {0: "far_i", 1: "far_j", 2: "far_k"}
ZM_FLAG = {0: "ZM_X", 1: "ZM_Y", 2: "ZM_Z"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _probe(value: str = "FMA_V1", backend: str = "cupy", patterns=None):
    """A probe artifact carrying this product's EXTENDED pattern set."""
    names = product.PRODUCT_PROBE_PATTERNS if patterns is None else patterns
    return stamp_probe_record(
        {"backend": backend, "patterns": {name: value for name in names}})


def fold(axis: str = "Y", extent: float = 2.0, phase: int = 1,
         boundaries: str = "periodic", complex_storage: bool = True,
         k_point=(0.0, 0.0, 0.0), other: float = 1.6, dimensions: int = 2,
         pml=None, **kwargs):
    """A folded Grid/Fields/PML triple on NumPy, complex storage by default.

    Default: the ``special_kz_2_21_2`` class — a folded PERIODIC axis at an even full
    count with an even plane, complex64 storage, an active PML on the folded axis's
    HIGH face only (the low face is refused by ``stepping._require_consistent_pml``).
    ``boundaries='metallic'`` gives the MIRROR_METALLIC class instead, where no far
    ghost exists. Deliberately IDENTICAL in shape to the magnetic twin's fixture, so
    a difference between the two suites is a difference between the two products.
    """
    size = [other, other, 0.0] if dimensions == 2 else [other, other, other]
    size["XYZ".index(axis)] = extent
    declared = ({axis.lower(): boundaries}
                if dimensions == 2 and boundaries != "periodic" else boundaries)
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=dimensions,
                courant=0.35, boundaries=declared,
                symmetry=(Mirror(axis, phase),), k_point=k_point, xp=np, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    if pml is None:
        thickness = []
        for index in range(3):
            if grid.shape[index] < 6:
                thickness.append((0, 0))
            elif index == "XYZ".index(axis):
                thickness.append((0, 2))
            else:
                thickness.append((2, 2))
        pml = tuple(thickness)
    return fields, PML(grid=grid, thickness=pml)


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause."""
    return [reason for reason in verdict.reasons if "array module" not in reason]


class Source:
    """A source that declares a field type AND publishes the index it writes.

    BOTH halves matter to this product: the field type decides whether it is in the
    D seam at all, and the deposit index is what ``deposit_repair.save`` needs. A
    source that declares the first and not the second is refused BY NAME, which is
    its own test below.
    """

    def __init__(self, field_type: str, index=(1, 1, 0)) -> None:
        self.field_type = field_type
        self._point_ix, self._point_iy, self._point_iz = index


class SourceWithoutIndex:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


# ---------------------------------------------------------------------------
# Reading the SHIPPED kernel — parsed, never re-implemented
# ---------------------------------------------------------------------------

def kernel_node():
    """The shipped kernel's AST, read from the FILE.

    From the file rather than through ``inspect`` because with Triton absent the
    kernel degrades to ``None`` — and these assertions must hold on exactly the
    machine that has no Triton.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME:
            return node
    raise AssertionError(f"{KERNEL_NAME} is not defined in {MODULE_PATH.name}")


def kernel_source_no_docstring() -> str:
    """The kernel's statements as source text, docstring dropped.

    The docstring is dropped because it QUOTES spellings these tests forbid ("NOT
    folded into one coefficient"), and a prose mention is not a transcription.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    node = kernel_node()
    body = node.body[1:] if ast.get_docstring(node) else node.body
    return "\n".join(segment for segment in
                     (ast.get_source_segment(text, statement) for statement in body)
                     if segment)


def carry_blocks():
    """Every emitted carry block, read off the shipped body.

    Returns per block the guard, the target volume, the inverse-epsilon binding, the
    destination offset, the coefficient pair, the mask, the ORDERED ``(axis, pass)``
    chain taken from the ``_mul_imag_coefficient_left`` calls' first argument, and
    where in that chain the wall clear lands. Nothing here computes a parity or a
    ghost value; a test that re-implemented the kernel would mirror its defects
    instead of executing them.
    """
    out = []
    for statement in kernel_node().body:
        if not isinstance(statement, ast.If):
            continue
        calls = [node for node in ast.walk(statement)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == CARRY_HELPER]
        if not calls:
            continue
        assert len(calls) == 1, ast.unparse(statement.test)
        chain, clears = [], []
        for inner in statement.body:
            if (isinstance(inner, ast.Assign)
                    and isinstance(inner.value, ast.Call)
                    and getattr(inner.value.func, "id", "")
                    == "_mul_imag_coefficient_left"):
                chain.append(COEFFICIENT_ARGUMENTS[inner.value.args[0].id])
            if isinstance(inner, ast.If) and ast.unparse(inner.test).startswith("ZM_"):
                clears.append((len(chain), ast.unparse(inner.test)))
        call = calls[0]
        out.append({
            "guard": ast.unparse(statement.test),
            "target": call.args[0].id,
            "inverse": call.args[3].id,
            "destination": ast.unparse(call.args[4]),
            "coefficient": (f"{ast.unparse(call.args[7])}, "
                            f"{ast.unparse(call.args[8])}"),
            "mask": ast.unparse(call.args[9]),
            "chain": tuple(chain),
            "clears": tuple(clears),
        })
    return out


def destinations(component: int):
    """``(near subset, far subset)`` for one D component, from the shipped tables."""
    return product.carried_destinations(product.NEAR_FILL_AXES[component],
                                        product.FAR_FILL_AXES[component])


# ---------------------------------------------------------------------------
# The geometry: this is the D family and not the B family
# ---------------------------------------------------------------------------

def test_the_two_fills_reach_complementary_axis_sets_for_the_D_family():
    """The structural fact the whole carry rests on, read off ``IYEE_SHIFTS``.

    ``_fill_symmetry_ghost_cells`` (stepping.py:1453) fills component ``m`` on axis
    ``a`` exactly when ``iyee[m][a] == 0``; ``_fill_folded_far_ghosts`` (:1565) fills
    the complement. For D the near set is the TWO axes that are not the component's
    own, and the far set is its own axis and only it — the exact inverse of the B
    family, which is what makes this a different kernel and not a flag.
    """
    near = {index: tuple(axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
            for index, name in enumerate(("Dx", "Dy", "Dz"))}
    far = {index: tuple(axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 1)
           for index, name in enumerate(("Dx", "Dy", "Dz"))}
    assert near == {0: (1, 2), 1: (0, 2), 2: (0, 1)}, near
    assert far == {0: (0,), 1: (1,), 2: (2,)}, far
    assert product.NEAR_FILL_AXES == ((1, 2), (0, 2), (0, 1))
    assert product.FAR_FILL_AXES == ((0,), (1,), (2,))
    for component in range(3):
        assert not (set(product.NEAR_FILL_AXES[component])
                    & set(product.FAR_FILL_AXES[component]))


def test_the_D_geometry_is_the_inverse_of_the_shipped_B_familys():
    """Read against the MAGNETIC twin's own tables rather than restated.

    If the two families ever agreed on a component's fill axes, one of them would be
    transcribing the wrong Yee row.
    """
    twin = importlib.import_module(
        "meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair")
    # The twin indexes BY AXIS (which components an axis fills); this module indexes
    # BY COMPONENT. Transposing one gives the other, and the two must be disjoint.
    twin_near_by_component = {
        component: tuple(axis for axis in range(3)
                         if component in twin.NEAR_FILL_COMPONENTS[axis])
        for component in range(3)}
    for component in range(3):
        assert twin_near_by_component[component] == (component,), component
        assert product.NEAR_FILL_AXES[component] != (component,)
        assert product.FAR_FILL_AXES[component] == (component,)


def test_a_component_owns_seven_ghosts_when_every_one_of_its_axes_is_folded():
    """Two near planes, their corner, the far plane, and every composition.

    The magnetic twin's count is the same seven for the mirror-image reason; what
    differs is WHICH cells, and that is the enumeration below.
    """
    for component in range(3):
        assert len(destinations(component)) == 7, component
    assert destinations(0) == (
        ((), (0,)), ((1,), ()), ((2,), ()),
        ((1,), (0,)), ((1, 2), ()), ((2,), (0,)),
        ((1, 2), (0,)))


def test_the_near_parity_is_plus_the_plane_phase_and_the_far_parity_is_minus_it():
    """``fields.mirror_parity`` itself, over every component, axis and phase.

    ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])``
    (fields.py:117), so a NEAR destination (shift 0) takes ``+phase`` and a FAR one
    (shift 1) takes ``-phase``. That is the ONLY parity input, and it is what
    :func:`.folded_complex.mirror_parity_coefficients` rounds to the two word pairs
    this kernel is passed. The near/far split is about WHICH AXIS, never about which
    word — which is why the D family reuses the B family's coefficient function
    unchanged.
    """
    for phase in (1, -1):
        near_words, far_words = folded_complex.mirror_parity_coefficients(phase)
        assert near_words == (float(phase), 0.0), (phase, near_words)
        assert far_words == (float(-phase), 0.0), (phase, far_words)
        for component, name in enumerate(("Dx", "Dy", "Dz")):
            for axis in product.NEAR_FILL_AXES[component]:
                assert mirror_parity(name, axis, phase) == phase, (name, axis)
            for axis in product.FAR_FILL_AXES[component]:
                assert mirror_parity(name, axis, phase) == -phase, (name, axis)


def test_the_parity_coefficient_words_imaginary_half_is_bitwise_zero():
    """Both words come from ``numpy.complex64`` and the imaginary one is ``+0.0``.

    Not merely equal to zero: the exact pattern a literal ``+0.0`` produces.
    """
    for phase in (1, -1):
        for words in folded_complex.mirror_parity_coefficients(phase):
            assert np.float32(words[1]).view(np.uint32) == np.uint32(0), words


def test_the_word_table_the_plan_passes_is_zero_only_on_an_unfolded_axis():
    fields, _pml = fold()
    words = product.parity_coefficient_words(fields.grid)
    assert len(words) == 6
    for axis in range(3):
        folded = bool(fields.grid.is_mirrored(axis))
        assert (words[axis] != (0.0, 0.0)) is folded, (axis, words)
        assert (words[axis + 3] != (0.0, 0.0)) is folded, (axis, words)


# ---------------------------------------------------------------------------
# THE PARITY CHAIN, and the clear that sits inside it
# ---------------------------------------------------------------------------

def test_every_carry_block_applies_its_parities_in_the_drivers_own_order():
    """The chain read off the SHIPPED kernel equals :func:`parity_chain`'s answer.

    ``driver.step`` runs ``fill_symmetry_bc_D`` (:3300) to completion before
    ``fill_folded_far_ghosts_D`` (:3302), and each pass applies its axes in ASCENDING
    order (stepping.py:1441-1447, :1567). So a back-substituted corner carries the
    near applications INNERMOST and ascending, then the far one. This test PARSES the
    kernel's calls; it computes no arithmetic.
    """
    blocks = {(block["target"], block["guard"]): block for block in carry_blocks()}
    assert len(blocks) == 21, sorted(blocks)
    seen = set()
    for component in range(3):
        target = f"f{component}"
        for near_subset, far_subset in destinations(component):
            expected = product.parity_chain(near_subset, far_subset)
            matches = [block for (tgt, _), block in blocks.items()
                       if tgt == target and block["chain"] == expected]
            assert len(matches) == 1, (target, near_subset, far_subset, expected)
            seen.add((target, matches[0]["guard"]))
    assert seen == set(blocks), sorted(set(blocks) - seen)


def test_no_carry_block_folds_its_chain_into_one_coefficient():
    """The refuted spelling, asserted ABSENT from the shipped body.

    Folding a two-pass chain into one coefficient moved up to 23 of 128 uint32 words
    on the Metal twin's exhaustive table. Every block applies exactly as many
    multiplies as its destination carries passes.
    """
    for block in carry_blocks():
        assert len(block["chain"]) == len(set(block["chain"])), block
        near = [entry for entry in block["chain"] if entry[1] == "near"]
        far = [entry for entry in block["chain"] if entry[1] == "far"]
        assert len(far) <= 1, block
        assert block["chain"] == tuple(near + far), block


def test_the_near_applications_are_innermost_and_ascending_in_every_mixed_chain():
    for block in carry_blocks():
        passes = [entry[1] for entry in block["chain"]]
        assert passes == sorted(passes, key=lambda name: 0 if name == "near" else 1)
        near_axes = [axis for axis, name in block["chain"] if name == "near"]
        assert near_axes == sorted(near_axes), block


def test_the_wall_clear_lands_between_the_near_parities_and_the_far_one():
    """:data:`CLEAR_AFTER_NEAR`, asserted against the emitted body.

    On the D family the clear's rows ARE the near fill's axes, and the driver clears
    BETWEEN the two fills, so a near ghost takes the clear after its parity and a far
    parity lands after the clear. A far-ONLY ghost takes no clear of its own: the
    value it images is the owned register, which has already taken it.
    """
    assert product.CLEAR_AFTER_NEAR is True
    for block in carry_blocks():
        near = [entry for entry in block["chain"] if entry[1] == "near"]
        if not near:
            assert block["clears"] == (), block
            continue
        assert len(block["clears"]) == 2, block
        for position, _flag in block["clears"]:
            assert position == len(near), block


def test_each_near_chain_clears_exactly_the_two_rows_zero_metal_writes():
    """``_zero_metal`` clears component ``m`` on the two axes that are not ``m``.

    That is the SAME table as :data:`NEAR_FILL_AXES`, which is precisely why the
    clear can meet a fill on this family and cannot on the magnetic one.
    """
    for block in carry_blocks():
        if not any(entry[1] == "near" for entry in block["chain"]):
            continue
        component = int(block["target"][1])
        flags = {flag for _position, flag in block["clears"]}
        expected = {f"{ZM_FLAG[axis]}" for axis in product.NEAR_FILL_AXES[component]}
        assert flags == expected, (block, expected)


def test_the_parity_multiply_is_the_fills_own_device_function_and_nothing_else():
    """Every complex multiply in the body is one of the four licensed helpers."""
    called = {getattr(node.func, "id", "") for node in ast.walk(kernel_node())
              if isinstance(node, ast.Call)}
    helpers = {name for name in called if name.startswith("_mul")
               or name.startswith("_rotate")}
    assert helpers <= set(product.LICENSED_MULTIPLY_HELPERS), helpers
    assert "_mul_imag_coefficient_left" in helpers


def test_the_products_pattern_set_is_the_extended_one_because_of_that_call():
    assert tuple(product.PRODUCT_PROBE_PATTERNS) == tuple(
        folded_complex.PARITY_PROBE_PATTERNS)
    assert folded_complex.PARITY_PROBE_PATTERN in product.PRODUCT_PROBE_PATTERNS


# ---------------------------------------------------------------------------
# The ownership enumeration
# ---------------------------------------------------------------------------

def test_every_destination_the_enumeration_names_is_emitted_once():
    blocks = carry_blocks()
    assert len(blocks) == 21
    for component in range(3):
        target = f"f{component}"
        emitted = {block["destination"] for block in blocks
                   if block["target"] == target}
        expected = set()
        for near_subset, far_subset in destinations(component):
            expected.add(" + ".join(["idx"] + [DF[axis] for axis in far_subset]
                                    + [DN[axis] for axis in near_subset]))
        assert emitted == expected, (component, emitted ^ expected)


def test_every_carry_mask_is_ANDed_with_the_components_ownership_mask():
    """The defect the real board's device gate FOUND, asserted structurally.

    With two fills a lane can be the SOURCE of one and the DESTINATION of the other;
    without the ownership conjunction it would write the composite cell from a ``v``
    built on a load its own mask zeroed.
    """
    for component in range(3):
        target = f"f{component}"
        for block in carry_blocks():
            if block["target"] != target:
                continue
            assert block["mask"].startswith(f"own{component} & "), block


def test_the_far_destination_takes_the_moved_coefficient_and_the_near_one_does_not():
    """The D-side coefficient rule, and the exact reverse of the twin's.

    ``update_E`` indexes component ``m`` on axis ``m``; the FAR fill images along
    exactly that axis, so its destination reads ``kps``/``kms`` at the TOP row, while
    a near destination sits at the same index as the lane that owns it.
    """
    for block in carry_blocks():
        component = int(block["target"][1])
        far = [entry for entry in block["chain"] if entry[1] == "far"]
        if far:
            assert block["coefficient"] == f"kp_f{component}, km_f{component}", block
        else:
            assert block["coefficient"] == f"kp_{component}, km_{component}", block


def test_every_block_binds_its_own_components_inverse_epsilon():
    for block in carry_blocks():
        component = int(block["target"][1])
        assert block["inverse"] == f"ie{component}", block


def test_the_kernels_literal_near_source_index_is_the_named_constant():
    """The near source lane is stored cell 2, spelled as a literal in the body.

    A ``triton.jit`` body that closed over a module-level Python int is a
    Triton-version question this file cannot measure without a device, so the literal
    is used and pinned to the name HERE.
    """
    assert folded_complex.MIRROR_SOURCE_INDEX == 2
    source = kernel_source_no_docstring()
    for lane in ("near_i = live & (i == 2)", "near_j = live & (j == 2)",
                 "near_k = live & (k == 2)"):
        assert lane in source, lane
    for offset in ("dn_x = -2 * nyz", "dn_y = -2 * nz", "dn_z = -2"):
        assert offset in source, offset


def test_the_far_reflect_row_is_a_RUNTIME_argument_and_never_baked():
    """``n_full - stored + 2`` is ``stored - 3`` at an ODD full count.

    Baking ``n - 2`` reflects about the window top rather than the second mirror and
    is a whole cell wrong on every odd-count run, which is why ``rx``/``ry``/``rz``
    are runtime scalars.
    """
    node = kernel_node()
    names = [argument.arg for argument in node.args.args]
    constexprs = {argument.arg for argument in node.args.args
                  if argument.annotation is not None}
    assert {"rx", "ry", "rz"} <= set(names)
    # RUNTIME means NOT a ``tl.constexpr``: a constexpr reflect row would be baked
    # into the compiled specialization, which is the defect this argument exists to
    # avoid.
    assert not ({"rx", "ry", "rz"} & constexprs), constexprs
    source = kernel_source_no_docstring()
    for lane in ("far_i = live & (i == rx)", "far_j = live & (j == ry)",
                 "far_k = live & (k == rz)"):
        assert lane in source, lane
    for offset in ("df_x = (nx - 1 - rx) * nyz", "df_y = (ny - 1 - ry) * nz",
                   "df_z = (nz - 1 - rz)"):
        assert offset in source, offset


# ---------------------------------------------------------------------------
# Declaration against body
# ---------------------------------------------------------------------------

def test_REPLACES_names_every_pass_the_kernel_actually_carries():
    """The tuple against what the body does, in both directions."""
    assert product.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")
    source = kernel_source_no_docstring()
    # step_D: the split-field recurrence and the fu store.
    assert "tl.store(u0 + 2 * idx, n0_re, mask=live)" in source
    # fill_D: the near carry.
    assert any(entry[1] == "near" for block in carry_blocks()
               for entry in block["chain"])
    # zero_metal_D: the register clear, six rows.
    assert source.count("if ZM_X:") >= 1
    # fill_folded_far_ghosts_D: the far carry.
    assert any(entry[1] == "far" for block in carry_blocks()
               for entry in block["chain"])
    # update_E: the constitutive accumulation with the inv_eps multiply. The
    # accumulator is spelled `a_re` because that is `complex_fused_electric_pair`'s
    # own spelling and the transcription diffs against its text.
    assert "a_re = a_re + t_re" in source
    assert "tl.load(ie0 + idx, mask=own0, other=0.0)" in source


def test_the_wall_clear_is_the_D_familys_six_rows_and_not_the_Bs_three():
    """``_zero_metal`` writes component ``m`` at cell 0 of every axis but ``m``.

    Six register lines for D against the magnetic twin's three, on both word planes.
    """
    source = kernel_source_no_docstring()
    expected = {
        "ZM_X": ("v1_re", "v1_im", "v2_re", "v2_im"),
        "ZM_Y": ("v0_re", "v0_im", "v2_re", "v2_im"),
        "ZM_Z": ("v0_re", "v0_im", "v1_re", "v1_im"),
    }
    block = source[source.index("if ZM_X:"):source.index("tl.store(u0 + 2 * idx")]
    for flag, names in expected.items():
        assert f"if {flag}:" in block, flag
        for name in names:
            assert f"{name} = tl.where(at_" in block, (flag, name)
    # THREE components x TWO planes x TWO walls each = twelve register lines.
    assert block.count("= tl.where(at_") == 12, block


def test_the_far_carry_brings_the_top_plane_mask_with_it():
    """``MIRROR_PERIODIC`` in the curl's mask block and the far carry are one edit.

    The BACKWARD == 1 arm is THREE lines per plane — the D shifts put a 1 on the
    component's own axis only — against the B arm's six.
    """
    source = kernel_source_no_docstring()
    assert "if BCX == MIRROR_PERIODIC:" in source
    head = source[source.index("last_x, last_y, last_z"):
                  source.index("own0, own1, own2")]
    backward = head[head.index("if BACKWARD:"):head.index("else:")]
    assert backward.count("MIRROR_PERIODIC") == 3, backward
    assert "curl0_re = tl.where(last_x, 0.0, curl0_re)" in backward
    assert "curl1_re = tl.where(last_y, 0.0, curl1_re)" in backward
    assert "curl2_re = tl.where(last_z, 0.0, curl2_re)" in backward


def test_the_backward_constexpr_is_step_Ds_and_only_step_Ds():
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.CURL_SUB_STEP == "step_D"
    assert product.BACKWARD == int(SUB_STEPS["step_D"]["backward"]) == 1
    assert product.CONSTITUTIVE_SIDE == "E"


def test_the_fill_family_key_is_the_ghost_fill_tables_own_key():
    assert product.FILL_FAMILY in folded_complex.GHOST_FILL_FAMILIES
    assert (folded_complex.GHOST_FILL_FAMILIES[product.FILL_FAMILY]["targets"]
            == ("Dx", "Dy", "Dz"))


# ---------------------------------------------------------------------------
# The transcription: each region against the module it names
# ---------------------------------------------------------------------------

def certified_source(module_file: str, name: str) -> str:
    path = PACKAGE_DIR / module_file
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if ast.get_docstring(node) else node.body
            return "\n".join(
                segment for segment in
                (ast.get_source_segment(text, statement) for statement in body)
                if segment)
    raise AssertionError(f"{name} is not defined in {module_file}")


CURL_LINES = (
    "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))",
    "t1_re = ((a_z_re - a_re) + (c_re - c_x_re))",
    "t2_re = ((b_x_re - b_re) + (a_re - a_y_re))",
    "curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)",
    "q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)",
    "n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)",
    "v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)",
    "r_re = (r_re + n0_re) - p0_re",
)


def test_every_curl_line_is_the_certified_folded_complex_curls_own_line():
    """The curl half diffs clean against ``folded_bloch_pml_curl_step``.

    Not a re-derivation: each statement below appears CHARACTER FOR CHARACTER in the
    certified emitter, and the certified emitter is read here rather than quoted.
    The curl arithmetic is the same on both seams — only ``BACKWARD`` differs — so
    these are the twin's lines too, and that is the point.
    """
    certified = certified_source("folded_complex.py", "folded_bloch_pml_curl_step")
    ours = kernel_source_no_docstring()
    for line in CURL_LINES:
        assert line in certified, ("moved in folded_complex.py", line)
        assert line in ours, ("moved here", line)


def test_the_only_curl_edit_is_the_ownership_mask_on_the_D_load():
    """A destination lane never READS ``D`` — masked off, not merely unused."""
    certified = certified_source("folded_complex.py", "folded_bloch_pml_curl_step")
    ours = kernel_source_no_docstring()
    for component, own in ((0, "own0"), (1, "own1"), (2, "own2")):
        assert f"e_re = tl.load(f{component} + 2 * idx, mask=live, other=0.0)" \
            in certified
        assert f"e_re = tl.load(f{component} + 2 * idx, mask={own}, other=0.0)" \
            in ours
        assert f"e_re = tl.load(f{component} + 2 * idx, mask=live, other=0.0)" \
            not in ours


def test_the_constitutive_half_is_the_certified_complex_electric_pairs_own_body():
    """Read against :mod:`.complex_fused_electric_pair`, which is itself the
    certified ``bloch_constitutive_step`` ``SCALE = 1`` arm with ``src`` bound to a
    register. The only edits here are the mask and the register the multiply reads.
    """
    certified = certified_source("complex_fused_electric_pair.py",
                                 "complex_fused_curl_constitutive_D")
    ours = kernel_source_no_docstring()
    for line in ("t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
                 "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)"):
        assert line in certified, line
        assert line in ours, line
    # `prev` is read BEFORE the `w` store — the one ordering the constitutive cannot
    # survive being wrong about — and the inv_eps multiply sits between them.
    for component in range(3):
        prev = ours.index(f"prev_re = tl.load(w{component} + 2 * idx")
        scale = ours.index(f"tl.load(ie{component} + idx, mask=own{component}")
        store = ours.index(f"tl.store(w{component} + 2 * idx, src_re")
        assert prev < scale < store, component


def test_the_inverse_epsilon_is_read_at_the_COMPLEX_cell_index():
    """``+ idx``, never ``+ 2 * idx``: inv_eps stays float32 under complex storage.

    One real coefficient per complex cell (stepping.py:37-38, fields.py:1203-1204).
    Reading it at the word index would take the next cell's coefficient on every
    other component.
    """
    ours = kernel_source_no_docstring()
    for component in range(3):
        assert f"tl.load(ie{component} + idx, mask=own{component}, other=0.0)" in ours
        assert f"ie{component} + 2 * idx" not in ours
    helper = certified_source("folded_complex_fused_pair.py", CARRY_HELPER)
    assert "tl.load(ie + dst, mask=mask, other=0.0)" in helper
    assert "ie + 2 * dst" not in helper


def test_the_ghost_constitutive_repeats_the_same_statements_in_the_same_order():
    """``_carry_ghost_complex_E`` is the owned cell's own sequence, moved.

    Twenty-one inlined copies would be twenty-one places for one of them to drift,
    which is why it is a device function; that it IS the same sequence is asserted
    here. The D store comes FIRST — the array path's fill writes ``D`` and
    ``update_E`` then reads it.
    """
    helper = certified_source("folded_complex_fused_pair.py", CARRY_HELPER)
    order = [
        "tl.store(f + 2 * dst, ghost_re",
        "prev_re = tl.load(w + 2 * dst",
        "src_re, src_im = _mul_field_left(",
        "tl.store(w + 2 * dst, src_re",
        "acc_re = tl.load(e + 2 * dst",
        "_mul_coefficient_left(kp_d, src_re, src_im, EXPANSION)",
        "_mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)",
        "tl.store(e + 2 * dst, acc_re",
    ]
    positions = []
    for fragment in order:
        assert fragment in helper, fragment
        positions.append(helper.index(fragment))
    assert positions == sorted(positions), positions


def test_the_wall_clear_is_applied_to_the_registers_before_every_consumer():
    """``zero_metal_D`` runs BETWEEN the two fills (driver.py:3301), so the far carry
    must read a CLEARED ``v`` and the constitutive must too. Applying it to the
    registers before the store is what makes both true at once.
    """
    ours = kernel_source_no_docstring()
    clear = ours.index("if ZM_X:")
    store = ours.index("tl.store(f0 + 2 * idx, v0_re")
    constitutive = ours.index("prev_re = tl.load(w0 + 2 * idx")
    carry = ours.index(f"{CARRY_HELPER}(f0, w0, h0, ie0,")
    assert clear < store < constitutive < carry, (clear, store, constitutive, carry)


def test_the_fu_store_is_unmasked_because_the_fills_never_touch_it():
    """``step_D`` writes ``fu`` at EVERY cell and the fills write ``field`` only
    (stepping.py:1497-1498), so a destination lane still owns its ``fu`` word.
    """
    ours = kernel_source_no_docstring()
    for component in range(3):
        assert (f"tl.store(u{component} + 2 * idx, n{component}_re, mask=live)"
                in ours), component
        assert (f"tl.store(f{component} + 2 * idx, v{component}_re, "
                f"mask=own{component})") in ours, component


# ---------------------------------------------------------------------------
# The seam predicate, in both directions
# ---------------------------------------------------------------------------

def test_a_folded_periodic_complex_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = fold()
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert residual(verdict) == [], residual(verdict)


def test_a_folded_metallic_complex_grid_is_admitted_too():
    fields, pml = fold(boundaries="metallic")
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert residual(verdict) == [], residual(verdict)


def test_an_electric_source_is_CARRIED_and_a_magnetic_one_never_reaches_this_seam():
    fields, pml = fold()
    electric = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    magnetic = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("B")], probe=_probe())
    assert residual(electric) == []
    assert residual(magnetic) == []


def test_an_electric_source_that_publishes_no_deposit_index_is_REFUSED():
    """Ignorance is never an empty set, and neither is an unsaveable deposit."""
    fields, pml = fold()
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [SourceWithoutIndex("D")], probe=_probe())
    assert any("does not publish the index it writes" in reason
               for reason in residual(verdict)), residual(verdict)


def test_the_flag_at_False_would_cost_the_whole_cell(monkeypatch):
    """The measurement the module's docstring makes, executed rather than restated.

    BOTH corpus rows declare an electric source. With
    ``CARRIES_DEPOSIT_REPAIR`` False the seam clause refuses every one of them
    outright, so the cell is worth ZERO — which is why the flag is declared in the
    same change as the wiring that brackets the launch.
    """
    fields, pml = fold()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert any("is electric" in reason for reason in residual(verdict)), \
        residual(verdict)


def test_the_flag_is_passed_to_the_clause_that_reads_it_and_not_merely_declared():
    """A flag nothing forwards is a comment. Read off the source."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "seam_source_reasons"]
    assert len(calls) == 1, calls
    keywords = {keyword.arg: ast.unparse(keyword.value)
                for keyword in calls[0].keywords}
    assert keywords["carries_repair"] == "CARRIES_DEPOSIT_REPAIR", keywords


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = fold()
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, None, probe=_probe())
    assert any("source set was not declared" in reason
               for reason in residual(verdict)), residual(verdict)


def test_real_storage_is_refused_because_the_real_folded_pair_owns_it():
    fields, pml = fold(complex_storage=False)
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert residual(verdict), "a real-storage grid must be refused"


def test_an_unfolded_grid_is_refused_because_this_is_a_composition_product():
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic", xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert any("mirror" in reason.lower() for reason in residual(verdict)), \
        residual(verdict)


class _OffdiagonalFields:
    """``Fields`` with the off-diagonal flag answered True.

    ``Fields.has_offdiagonal_epsilon`` is a read-only PROPERTY derived from the
    installed ``chi1inv`` rows, so a test cannot assign it. Wrapping is the honest
    substitute: every other attribute is the real object's, and the predicate reads
    the flag through ``getattr`` exactly as it would on a run that installed a row.
    The GATE installs a real row instead, on the device's own objects.
    """

    def __init__(self, fields) -> None:
        object.__setattr__(self, "_fields", fields)

    has_offdiagonal_epsilon = True

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fields"), name)


def test_an_offdiagonal_row_is_refused_BY_NAME_because_update_E_is_a_stencil():
    """The clause that makes this cell buildable at all, asserted in its own right.

    The magnetic twin ADMITS an off-diagonal row through a second arm label, because
    on B->H the two arms are the same kernel. On D->E it is the stencil, and this
    weld may not claim it.
    """
    fields, pml = fold()
    verdict = product.folded_complex_fused_pair_coverage(
        _OffdiagonalFields(fields), pml, [Source("D")], probe=_probe())
    assert any("STENCIL" in reason for reason in residual(verdict)), \
        residual(verdict)


def test_a_probe_without_the_parity_pattern_refuses_this_product():
    fields, pml = fold()
    base = tuple(name for name in product.PRODUCT_PROBE_PATTERNS
                 if name != folded_complex.PARITY_PROBE_PATTERN)
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe(patterns=base))
    assert residual(verdict), "the parity pattern is not optional for this product"


def test_the_predicate_reports_every_halfs_reasons_with_its_side_named():
    fields, pml = fold(complex_storage=False)
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    prefixes = {reason.split(":")[0] for reason in verdict.reasons}
    assert "folded complex curl half" in prefixes, prefixes
    assert "folded complex constitutive half" in prefixes, prefixes
    assert "folded complex fill half" in prefixes, prefixes


def test_no_active_pml_is_refused_because_this_is_the_split_field_product():
    fields, pml = fold(pml=((0, 0), (0, 0), (0, 0)))
    verdict = product.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert residual(verdict), "an inactive layer must be refused"


def test_the_builder_returns_None_for_every_configuration_the_predicate_refuses():
    for kwargs in ({"complex_storage": False},
                   {"pml": ((0, 0), (0, 0), (0, 0))}):
        fields, pml = fold(**kwargs)
        assert product.plan_folded_complex_fused_pair(
            fields, pml, [Source("D")], probe=_probe()) is None, kwargs
    fields, pml = fold()
    assert product.plan_folded_complex_fused_pair(
        fields, pml, None, probe=_probe()) is None


def test_the_wall_clear_and_the_fold_are_disjoint_on_every_admitted_grid():
    """``_zero_metal`` skips a folded axis, and on the D family that is the ONLY
    thing keeping the clear and the near fill apart. The predicate checks it; this
    asserts the check is not vacuous on a grid that does carry a wall.
    """
    fields, pml = fold(boundaries="metallic")
    walls = zero_metal_axes(fields.grid)
    for axis in range(3):
        if fields.grid.is_mirrored(axis):
            assert not walls[axis], axis


# ---------------------------------------------------------------------------
# The plan's own refusals
# ---------------------------------------------------------------------------

def arrays_plan(**overrides):
    """A from-arrays plan with every argument defaulted to a legal one."""
    shape = (8, 8, 1)
    kwargs = dict(
        arrays={name: np.zeros(shape, dtype=np.complex64)
                for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
                             "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
                             "f_w_Ex", "f_w_Ey", "f_w_Ez")},
        curl_flat={f"{stem}_{axis}": np.zeros(shape["xyz".index(axis)],
                                              dtype=np.float32)
                   for axis in "xyz" for stem in ("kms", "sinv")},
        constitutive_flat={f"{stem}_{axis}": np.zeros(shape["xyz".index(axis)],
                                                      dtype=np.float32)
                           for axis in "xyz" for stem in ("kps", "kms")},
        codes=(folded_complex.CODE_MIRROR_PERIODIC, folded_complex.CODE_PERIODIC,
               folded_complex.CODE_PERIODIC),
        zero_metal=(False, False, False),
        phases=(None, None, None),
        parity=((1.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        dtdx=0.35, expansion=1, reflect=(4, None, None),
    )
    kwargs["arrays"].update(
        {f"inv_eps_{name}": np.zeros(shape, dtype=np.float32)
         for name in ("Ex", "Ey", "Ez")})
    kwargs.update(overrides)
    return product.plan_folded_complex_fused_pair_from_arrays(**kwargs)


def test_the_arrays_route_builds_a_plan_from_a_legal_declaration():
    plan = arrays_plan()
    assert plan.near == (True, False, False)
    assert plan.far == (True, False, False)
    assert plan.replaces == product.REPLACES


def test_the_plan_refuses_a_reflect_row_it_cannot_own():
    with pytest.raises(ValueError, match="reflect row"):
        arrays_plan(reflect=(7, None, None))


def test_the_plan_refuses_a_reflect_row_on_an_axis_that_has_no_far_ghost():
    with pytest.raises(ValueError, match="not folded PERIODIC"):
        arrays_plan(reflect=(4, 3, None))


def test_the_plan_refuses_a_fold_that_also_carries_a_wall_clear():
    with pytest.raises(ValueError, match="fold and a wall clear"):
        arrays_plan(zero_metal=(True, False, False))


def test_the_plan_refuses_a_fold_that_also_carries_a_bloch_phase():
    with pytest.raises(ValueError, match="fold and a Bloch phase"):
        arrays_plan(phases=(1j, None, None))


def test_the_plan_refuses_a_zero_parity_word_pair_on_a_folded_axis():
    with pytest.raises(ValueError, match="NEAR parity words"):
        arrays_plan(parity=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                            (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="FAR parity words"):
        arrays_plan(parity=((1.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                            (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))


def test_the_plan_refuses_a_parity_table_that_is_not_six_pairs():
    with pytest.raises(ValueError, match="six"):
        arrays_plan(parity=((1.0, 0.0), (-1.0, 0.0)))


def test_the_plan_derives_NEAR_and_FAR_from_the_codes_and_never_takes_them():
    """A kernel that masked the top plane on one axis set and imaged the far ghost
    on another is a plane of wrong values rather than a crash.
    """
    signature = product.FoldedComplexFusedPairPlan.__init__.__code__.co_varnames
    assert "near" not in signature and "far" not in signature
    plan = arrays_plan(codes=(folded_complex.CODE_MIRROR_METALLIC,
                              folded_complex.CODE_PERIODIC,
                              folded_complex.CODE_PERIODIC),
                       reflect=(None, None, None))
    assert plan.near == (True, False, False)
    assert plan.far == (False, False, False)


def test_the_plan_refuses_a_placeholder_inverse_epsilon_binding():
    """The kernel loads ``inv_eps`` on every launch and at every ghost; a ``None``
    binding would be read as a coefficient rather than refused.
    """
    with pytest.raises(ValueError, match="inv_eps"):
        product.FoldedComplexFusedPairPlan(
            (8, 8, 1), 0.35,
            (folded_complex.CODE_PERIODIC,) * 3, (False,) * 3,
            (0, 0, 0), (0.0,) * 6, ((0.0, 0.0),) * 6, 1, 256,
            [np.zeros((8, 8, 1), np.complex64)] * 3,
            [np.zeros((8, 8, 1), np.complex64)] * 3,
            [np.zeros((8, 8, 1), np.complex64)] * 3,
            [np.zeros(8, np.float32)] * 6,
            [np.zeros((8, 8, 1), np.complex64)] * 3,
            [np.zeros((8, 8, 1), np.complex64)] * 3,
            None,
            [np.zeros(8, np.float32)] * 6,
            reflect=(-1, -1, -1))


def test_the_engine_route_and_the_arrays_route_agree_on_the_parity_table():
    fields, pml = fold()
    words = product.parity_coefficient_words(fields.grid)
    plan = product.plan_folded_complex_fused_pair(
        fields, pml, [Source("D")], probe=_probe())
    if plan is None:  # the NumPy host clause refuses; compare the table alone
        assert len(words) == 6
        return
    assert plan.parity == tuple(words)  # pragma: no cover - needs a CuPy host


# ---------------------------------------------------------------------------
# The deferral, the wiring records and the import contract
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_dispatch_now_admits_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch by the
    2026-09-13 Phase B batch — both halves.

    Until 2026-09-02 this test asserted that ``launch.py`` and ``fastpath.py`` named
    this module NOWHERE, the seam that kept a certified-but-unrouted product deferred.
    The wave routed it: ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    The 2026-09-13 Phase B batch then RELEASED it, and the second half is that fact
    made checkable from the PRODUCT's side rather than from the composer's: the label
    this product writes now sits in ``fastpath.RELEASED_FUSED_ARMS``, so clause (8)
    admits the whole plan; and it still sits in exactly one of ``ARM_CERTIFICATION``
    (its gate has a tracked ledger entry) and ``PENDING_DEVICE_GATE_ARMS`` (it does
    not — the release moved it out of the pending rung).
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_complex_fused_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_complex_fused_pair"]
    assert row["module"] == "folded_complex_fused_pair"
    assert row["builder"] == "plan_folded_complex_fused_pair"
    label = row["label"]
    assert label == 'fused pair D (folded complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_package_does_not_import_the_module_eagerly():
    text = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert MODULE_PATH.stem not in text


def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    """A product that declares the flag must appear in the wiring test's own list.

    ``test_fused_pair_deposit_wiring.WIRED_FOR_THE_REPAIR`` is an equality in both
    directions, so a module that declared the flag and was never registered there
    fails that suite; this asserts the registration from THIS side too, so the pair
    cannot drift apart silently.
    """
    text = (PACKAGE_DIR.parent
            / "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert f"triton_kernels/{MODULE_PATH.name}" in text


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in text
    assert text.count("kernel[self._grid](") == 1


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """The merge bar is a laptop. The predicate and the builder must answer there."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in [key for key in list(sys.modules) if key.startswith("triton.")
                 or key == "triton"
                 or key.endswith("folded_complex_fused_pair")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # RELOADED THROUGH THE PACKAGE, not from a bare file spec: this module's body is
    # a chain of RELATIVE imports, and a spec with no parent package fails on the
    # first one with an ImportError that says nothing about Triton.
    reloaded = importlib.import_module(
        "meep_gpu.triton_kernels.folded_complex_fused_pair")
    reloaded = importlib.reload(reloaded)
    assert reloaded.folded_complex_fused_curl_constitutive_D is None
    with pytest.raises(ImportError, match="needs the optional `triton` package"):
        reloaded.folded_complex_fused_curl_constitutive_D_kernel()
    fields, pml = fold()
    verdict = reloaded.folded_complex_fused_pair_coverage(
        fields, pml, [Source("D")], probe=_probe())
    assert verdict.covered is False
    assert reloaded.plan_folded_complex_fused_pair(
        fields, pml, [Source("D")], probe=_probe()) is None


def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it():
    """No ``fingerprints.json`` entry, and ``DEVICE_STATUS`` says what has run.

    A green laptop suite over an unrun device gate must never be mistaken for a
    certification, so this reads the module's own declaration rather than assuming.
    """
    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    blob = json.dumps(fingerprints)
    status = product.DEVICE_STATUS
    if "NOT RELEASED" in status:
        assert MODULE_PATH.stem not in blob, \
            "an unreleased product may not carry a checked-in fingerprint"
    else:  # pragma: no cover - reached once the gate has released
        assert "gate.json" in status


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_complex_fused_pair" in text
    assert KERNEL_NAME in text


def test_the_weld_driver_knows_how_to_run_this_gate():
    text = (PARITY_DIR / "drive_triton_weld_gates.py").read_text(encoding="utf-8")
    assert "probe_triton_folded_complex_fused_pair.py" in text


def test_the_board_and_the_battery_both_know_this_product():
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR
               / "triton_predicate_battery.py").read_text(encoding="utf-8")
    assert '"folded_complex_fused_pair"' in board
    assert "folded_complex_fused_pair_coverage" in battery


# ---------------------------------------------------------------------------
# The gate's own declarations, read on the laptop
# ---------------------------------------------------------------------------

def load_gate():
    spec = importlib.util.spec_from_file_location("_fcfp_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    """EVERY no-device leg needs no GPU, so every one of them runs here.

    Read off the gate's own ``NO_DEVICE_LEGS`` tuple rather than listed: a leg added
    there and not here would be a leg the merge bar never runs.
    """
    gate = load_gate()
    for leg in gate.NO_DEVICE_LEGS:
        row = leg()
        assert row.get("passed") is True, (leg.__name__, row.get("findings"))


def test_every_mutation_declares_one_of_the_permitted_expectations():
    """``caught``, ``null`` or ``unreached``, and the last two must be EARNED.

    THE THREE ARE NOT INTERCHANGEABLE. ``null`` says the rewrite changes nothing —
    the defect is not a defect — and its reason lives in the rewrite's own
    docstring. ``unreached`` says the defect IS real, names what proves it in
    ``MUTATION_EVIDENCE``, and records that this gate's device legs do not reach it;
    collapsing it into ``null`` would retire a hazard without carrying it, and
    calling it ``caught`` would be false. A stubbornly uncaught mutation moved to
    either spelling without that record would be silenced rather than explained.
    """
    gate = load_gate()
    for name, why, expectation, rewrite in gate.mutation_table():
        assert expectation in ("caught", "null", "unreached"), (name, expectation)
        assert why, name
        if expectation == "null":
            assert "DECLARED NULL" in (rewrite.__doc__ or ""), name
            assert not gate.MUTATION_EVIDENCE.get(name), name
        if expectation == "unreached":
            evidence = gate.MUTATION_EVIDENCE.get(name)
            assert evidence, name
            # The evidence must name what MEASURED the defect real, not merely
            # assert that it is.
            assert "parity_chain_associativity_leg" in evidence \
                or "the same sweep" in evidence, name


def test_the_word_layer_sweep_says_the_unreached_defects_are_real():
    """An ``unreached`` declaration rests on this count, so the count is asserted.

    The leg sweeps the shipped multiply's two arms over the signed-zero / subnormal
    word table and reports how many words a re-order and a fold move. A ZERO there
    would mean the defect cannot be seen at any layer, and ``unreached`` would be a
    null with a longer word.
    """
    gate = load_gate()
    row = gate.parity_chain_associativity_leg()
    assert row["passed"] is True, row["findings"]
    assert row["rows"], row
    for entry in row["rows"]:
        assert entry["reorder_differing_words"] > 0, entry
        assert entry["fold_differing_words"] > 0, entry


def test_the_mutation_tables_name_no_key_twice():
    """A dict literal takes the LAST value, so a repeated key silently discards the
    earlier one — which is how ``m_fu_store_masked`` was put back on a case where its
    rewrite is a no-op on 2026-08-31. The gate's own pairing leg measures it; this
    pins that the leg is wired.
    """
    gate = load_gate()
    row = gate.mutation_pairing_leg()
    assert row["passed"] is True, row["findings"]


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    """A mutation whose rewrite does not apply is a leg that reports 'caught' after
    launching the shipped kernel. Each one is applied here, on the laptop.
    """
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name


def test_every_mutation_case_names_a_real_case():
    gate = load_gate()
    names = {case[0] for case in gate.CASES}
    for name, case in gate.MUTATION_CASE.items():
        assert case in names, (name, case)
    assert gate.DEFAULT_MUTATION_CASE in names


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    """TWO VOCABULARIES, and they are not the same list.

    The driver binds ``fill_symmetry_bc_D``; the RESIDENCY model spells that slot
    ``fill_D``, and ``REPLACES`` is declared in the residency spelling because that
    is what a composer reads. The gate maps them, and the map must be a bijection
    that lands exactly on this module's declaration.
    """
    gate = load_gate()
    assert sorted(gate.RESIDENCY_NAME) == sorted(gate.SEAM_PASSES)
    assert len(set(gate.RESIDENCY_NAME.values())) == len(gate.RESIDENCY_NAME)
    assert tuple(gate.RESIDENCY_NAME[name]
                 for name in gate.SEAM_PASSES) == product.REPLACES


def test_the_gate_carries_a_carry_family_and_a_null_control():
    """Every carry case needs a bracket-removed control that MUST diverge; a bracket
    that changes nothing is not load-bearing and the case is vacuous.
    """
    gate = load_gate()
    assert gate.CARRY_CASES, "the deposit carry is this cell's whole value"
    text = GATE.read_text(encoding="utf-8")
    assert "bracket" in text and "null" in text.lower()


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in ("meep_gpu/triton_kernels/folded_complex_fused_pair.py",
                     "meep_gpu/triton_kernels/folded_complex.py",
                     "meep_gpu/triton_kernels/complex_fields.py",
                     "meep_gpu/triton_kernels/special_kz.py",
                     "meep_gpu/deposit_repair.py",
                     "meep_gpu/stepping.py",
                     "meep_gpu/driver.py",
                     "meep_gpu/test_triton_folded_complex_fused_pair.py"):
        assert required in names, required


def test_the_gate_has_not_been_run_or_names_the_run_it_took():
    status = product.DEVICE_STATUS
    assert ("NOT RELEASED" in status) or ("gate.json" in status), status


def test_the_case_table_reaches_the_deepest_composition_the_kernel_emits():
    """A kernel emitting seven ghosts per component whose gate only ever folded one
    axis would leave fourteen blocks per component unexecuted.
    """
    gate = load_gate()
    folds = max(len(case[4]) for case in gate.CASES)
    assert folds == 3, ("no case folds three axes; a component with all three of its "
                        "axes folded owns SEVEN ghosts and the triple-composite "
                        "blocks are otherwise never compiled")


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "folded complex PML", "folded complex")
