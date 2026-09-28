"""Tests for the FOLDED COMPLEX fused magnetic pair — folded Bloch ``step_B`` into
``update_H``, with both mirror fills and the wall clear carried inline.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the separately certified products
this launch replaces — lives in
``parity/meep_gpu/probe_triton_folded_complex_fused_magnetic_pair.py``. Whether that
gate has RUN is a fact this suite reads out of the module's own
:data:`~.folded_complex_fused_magnetic_pair.DEVICE_STATUS` rather than assuming: a
green laptop suite over an unrun device gate must never be mistaken for a
certification.

WHAT IS PINNED HERE, and the one thing that is genuinely new on this board:

* **THE PARITY CHAIN.** The real-storage folded pair composes a destination's
  parities into ONE compile-time sign and states the result is order-independent.
  Under complex storage that is FALSE — the parity is a full complex multiply by
  ``(±1.0, +0.0)``, and complex float multiplication is not associative. So this
  kernel applies one ``_mul_imag_coefficient_left`` PER PASS in the driver's own
  order, and these tests read that order OFF THE SHIPPED KERNEL SOURCE and compare
  it against :func:`~.folded_complex_fused_magnetic_pair.parity_chain` — which is
  not a re-implementation of the arithmetic, it is a statement about which
  coefficient argument each call takes and in what sequence;
* the ownership enumeration: every emitted block's guard, destination offset, mask
  and coefficient index, against :func:`carried_destinations`;
* the geometry that makes this the B family and not the D one, read off
  ``fields.IYEE_SHIFTS``;
* every clause of the seam predicate, in both directions;
* the transcription: each region of the kernel against the certified module it
  names, by source text rather than by re-derivation;
* the declaration/kernel agreement — ``REPLACES`` against what the body actually
  carries, which is the failure a sibling module shipped for a day;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module.
"""

from __future__ import annotations

import ast
import hashlib
import builtins
import importlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu.fields import IYEE_SHIFTS, Fields, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import folded_complex
from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as product
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
GATE = PARITY_DIR / "probe_triton_folded_complex_fused_magnetic_pair.py"

KERNEL_NAME = "folded_complex_fused_curl_constitutive_B"

#: Which kernel argument carries which (axis, pass) coefficient. This table is the
#: kernel SIGNATURE's own naming and is asserted against it below, so a renamed
#: argument fails here rather than silently making the chain tests vacuous.
COEFFICIENT_ARGUMENTS = {
    "n0r": (0, "near"), "n1r": (1, "near"), "n2r": (2, "near"),
    "d0r": (0, "far"), "d1r": (1, "far"), "d2r": (2, "far"),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _probe(value: str = "FMA_V1", backend: str = "cupy",
           patterns=None):
    """A probe artifact carrying this product's EXTENDED pattern set."""
    names = product.PRODUCT_PROBE_PATTERNS if patterns is None else patterns
    return stamp_probe_record(
        {"backend": backend, "patterns": {name: value for name in names}})


def fold(axis: str = "Y", extent: float = 2.0, phase: int = 1,
         boundaries: str = "periodic", complex_storage: bool = True,
         k_point=(0.0, 0.0, 0.0), other: float = 1.6, dimensions: int = 2,
         pml=None, **kwargs):
    """A folded Grid/Fields/PML triple on NumPy, complex storage by default.

    Default: the ``special_kz_2_21_2`` class — a folded PERIODIC axis at an even
    full count with an even plane, complex64 storage, an active PML on the folded
    axis's HIGH face only (the low face is refused by
    ``stepping._require_consistent_pml``). ``boundaries='metallic'`` gives the
    MIRROR_METALLIC class instead, where no far ghost exists.
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

    The docstring is dropped because it QUOTES spellings these tests forbid
    ("NOT folded into one coefficient"), and a prose mention is not a
    transcription.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    node = kernel_node()
    body = node.body[1:] if ast.get_docstring(node) else node.body
    return "\n".join(segment for segment in
                     (ast.get_source_segment(text, statement) for statement in body)
                     if segment)


def carry_blocks():
    """Every emitted carry block, read off the shipped body.

    Returns ``(guard, target, destination, coefficient, mask, chain)`` per block,
    where ``chain`` is the ORDERED ``(axis, pass)`` list taken from the
    ``_mul_imag_coefficient_left`` calls' first argument — the kernel's own
    sequence, parsed. Nothing here computes a parity or a ghost value; a test that
    re-implemented the kernel would mirror its defects instead of executing them.
    """
    out = []
    for statement in kernel_node().body:
        if not isinstance(statement, ast.If):
            continue
        calls = [node for node in ast.walk(statement)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == "_carry_ghost_complex"]
        if not calls:
            continue
        assert len(calls) == 1, ast.unparse(statement.test)
        chain = []
        for inner in statement.body:
            if not isinstance(inner, ast.Assign):
                continue
            value = inner.value
            if (isinstance(value, ast.Call)
                    and getattr(value.func, "id", "") == "_mul_imag_coefficient_left"):
                chain.append(COEFFICIENT_ARGUMENTS[value.args[0].id])
        call = calls[0]
        out.append({
            "guard": ast.unparse(statement.test),
            "target": call.args[0].id,
            "destination": ast.unparse(call.args[3]),
            "coefficient": ast.unparse(call.args[6]),
            "mask": ast.unparse(call.args[8]),
            "chain": tuple(chain),
        })
    return out


# ---------------------------------------------------------------------------
# The geometry: this is the B family and not the D family
# ---------------------------------------------------------------------------

def test_the_two_fills_reach_complementary_axis_sets_for_the_B_family():
    """The structural fact the whole carry rests on, read off ``IYEE_SHIFTS``.

    ``_fill_symmetry_ghost_cells`` (stepping.py:1453) fills component ``m`` on axis
    ``a`` exactly when ``iyee[m][a] == 0``; ``_fill_folded_far_ghosts`` (:1565)
    fills the complement. For B the near set is ``a == m`` — the component's OWN
    axis, and only it — and the far set is the two that are not.
    """
    near = {axis: tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
                        if IYEE_SHIFTS[name][axis] == 0) for axis in range(3)}
    far = {axis: tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
                       if IYEE_SHIFTS[name][axis] == 1) for axis in range(3)}
    assert near == {0: (0,), 1: (1,), 2: (2,)}, near
    assert far == {0: (1, 2), 1: (0, 2), 2: (0, 1)}, far
    assert product.NEAR_FILL_COMPONENTS == ((0,), (1,), (2,))
    assert product.FAR_FILL_COMPONENTS == ((1, 2), (0, 2), (0, 1))
    for axis in range(3):
        assert not (set(product.NEAR_FILL_COMPONENTS[axis])
                    & set(product.FAR_FILL_COMPONENTS[axis]))


def test_the_near_parity_is_plus_the_plane_phase_and_the_far_parity_is_minus_it():
    """``fields.mirror_parity`` itself, over every component, axis and phase.

    ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])`` (fields.py:117),
    so a NEAR destination (shift 0) takes ``+phase`` and a FAR one (shift 1) takes
    ``-phase``. That is the ONLY parity input, and it is what
    :func:`.folded_complex.mirror_parity_coefficients` rounds to the two word pairs
    this kernel is passed.
    """
    for phase in (1, -1):
        near_words, far_words = folded_complex.mirror_parity_coefficients(phase)
        assert near_words == (float(phase), 0.0), (phase, near_words)
        assert far_words == (float(-phase), 0.0), (phase, far_words)
        for axis in range(3):
            for index, name in enumerate(("Bx", "By", "Bz")):
                expected = mirror_parity(name, axis, phase)
                if index in product.NEAR_FILL_COMPONENTS[axis]:
                    assert expected == phase, (name, axis, phase)
                    assert near_words[0] == float(expected)
                else:
                    assert index in product.FAR_FILL_COMPONENTS[axis]
                    assert expected == -phase, (name, axis, phase)
                    assert far_words[0] == float(expected)


def test_the_parity_coefficient_words_imaginary_half_is_bitwise_zero():
    """Both words come from ``numpy.complex64`` and the imaginary one is ``+0.0``.

    Not merely equal to zero: the exact pattern a literal ``+0.0`` produces. The
    gate's ``synthesize_zero_imag`` source mutation is VACUOUS BY CONSTRUCTION
    because of this, and is recorded there as a structural null — a fact that stops
    being true the moment this does.
    """
    for phase in (1, -1):
        for words in folded_complex.mirror_parity_coefficients(phase):
            assert np.float32(words[1]).view(np.uint32) == np.uint32(0), words


# ---------------------------------------------------------------------------
# THE PARITY CHAIN — the one thing that does not transfer from the real board
# ---------------------------------------------------------------------------

def test_every_carry_block_applies_its_parities_in_the_drivers_own_order():
    """The chain read off the SHIPPED kernel equals :func:`parity_chain`'s answer.

    ``driver.step`` runs ``fill_symmetry_bc_B`` (:3285) to completion before
    ``fill_folded_far_ghosts_B`` (:3287), and the far pass applies its axes in
    ASCENDING order (``stepping._fill_folded_far_ghosts``:1518, :1528). So a
    back-substituted corner carries the near application INNERMOST and then the far
    axes ascending. This test PARSES the kernel's calls; it computes no arithmetic.
    """
    blocks = {(block["target"], block["guard"]): block for block in carry_blocks()}
    assert len(blocks) == 21, sorted(blocks)
    seen = set()
    for component in range(3):
        near = (component,)
        far = tuple(axis for axis in range(3) if axis != component)
        target = f"f{component}"
        for subset, carries_near in product.carried_destinations(near, far):
            expected = product.parity_chain(near, subset, carries_near)
            matches = [block for (tgt, _), block in blocks.items()
                       if tgt == target and block["chain"] == expected]
            assert len(matches) == 1, (target, subset, carries_near, expected)
            seen.add((target, matches[0]["guard"]))
    assert seen == set(blocks), sorted(set(blocks) - seen)


def test_no_carry_block_folds_its_chain_into_one_coefficient():
    """The refuted spelling, asserted ABSENT from the shipped body.

    The real-storage twin writes ``PHX * PHY * v0`` — one compile-time sign. Under
    complex storage that folds two complex multiplies into one and moves bytes (the
    Metal twin measured up to 23 of 128 uint32 words). Every block must therefore
    carry exactly ``len(chain)`` multiply calls and no arithmetic between the
    coefficient words.
    """
    for block in carry_blocks():
        assert len(block["chain"]) >= 1, block
    source = kernel_source_no_docstring()
    for left in COEFFICIENT_ARGUMENTS:
        for right in COEFFICIENT_ARGUMENTS:
            for spelling in (f"{left} * {right}", f"{left}*{right}"):
                assert spelling not in source, spelling


def test_the_chain_length_matches_the_number_of_passes_the_destination_carries():
    """One multiply per pass — never more, never fewer."""
    for block in carry_blocks():
        guard = block["guard"]
        passes = guard.replace("(", "").replace(")", "").split(" and ")
        assert len(block["chain"]) == len(passes), block


def test_the_near_application_is_innermost_in_every_mixed_chain():
    """The driver's order, stated as a property of every emitted block."""
    for block in carry_blocks():
        kinds = [kind for _axis, kind in block["chain"]]
        assert kinds.count("near") <= 1, block
        if "near" in kinds:
            assert kinds[0] == "near", block
        far_axes = [axis for axis, kind in block["chain"] if kind == "far"]
        assert far_axes == sorted(far_axes), block


def test_the_parity_multiply_is_the_fills_own_device_function_and_nothing_else():
    """One spelling of the parity product in this package, not two.

    ``_mul_imag_coefficient_left`` is what
    :func:`.folded_complex.folded_mirror_ghost_fill_complex` calls; a second
    spelling here would be a second place for the coefficient-left orientation to be
    subtly wrong, and it would need its own probe pattern.
    """
    source = kernel_source_no_docstring()
    called = {node.func.id for node in ast.walk(kernel_node())
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    multiplies = {name for name in called if name.startswith("_mul")
                  or name.startswith("_rotate")}
    assert multiplies <= set(product.LICENSED_MULTIPLY_HELPERS), multiplies
    assert "_mul_imag_coefficient_left" in multiplies
    assert "_mul_imag_coefficient_left" in source


def test_the_products_pattern_set_is_the_extended_one_because_of_that_call():
    """The parity orientation is a NEW probe pattern and this product launches it."""
    assert (product.PRODUCT_PROBE_PATTERNS
            == tuple(folded_complex.PARITY_PROBE_PATTERNS))
    assert folded_complex.PARITY_PROBE_PATTERN in product.PRODUCT_PROBE_PATTERNS
    # And it is strictly bigger than the unfolded complex pair's, which does not
    # launch the parity orientation at all.
    complex_pair = importlib.import_module(
        "meep_gpu.triton_kernels.complex_fused_magnetic_pair")
    assert set(complex_pair.PRODUCT_PROBE_PATTERNS) < set(
        product.PRODUCT_PROBE_PATTERNS)


# ---------------------------------------------------------------------------
# The ownership enumeration, against the emitted blocks
# ---------------------------------------------------------------------------

def test_every_destination_the_enumeration_names_is_emitted_once():
    """Seven ghosts per component, twenty-one blocks, no duplicates."""
    blocks = carry_blocks()
    assert len(blocks) == 21
    destinations = [(block["target"], block["destination"]) for block in blocks]
    assert len(set(destinations)) == 21, destinations
    for component in range(3):
        near = (component,)
        far = tuple(axis for axis in range(3) if axis != component)
        assert len(product.carried_destinations(near, far)) == 7


def test_every_carry_mask_is_ANDed_with_the_components_ownership_mask():
    """The defect the real board's DEVICE run found, asserted structurally.

    With two fills a lane can be the SOURCE of one and the DESTINATION of the other,
    and it would then write the composite cell from a ``v`` built on a load its own
    ownership mask zeroed. Every mask must therefore start ``own?``.
    """
    for block in carry_blocks():
        component = block["target"][1]
        assert block["mask"].startswith(f"own{component} &"), block


def test_the_near_destination_takes_the_moved_coefficient_and_the_far_one_does_not():
    """``update_H`` indexes component ``m`` on axis ``m``; the near fill images
    along that same axis, so its destination's ``kps``/``kms`` pair is at index 0
    and is NOT the source lane's. A FAR destination does not move the indexed
    coordinate and reuses the lane's own pair.
    """
    for block in carry_blocks():
        component = block["target"][1]
        carries_near = any(kind == "near" for _axis, kind in block["chain"])
        expected = f"kp_d{component}" if carries_near else f"kp_{component}"
        assert block["coefficient"] == expected, block


def test_the_kernels_literal_near_source_index_is_the_named_constant():
    """``i == 2`` in the body, pinned to ``MIRROR_SOURCE_INDEX``.

    A jit body that closed over a module-level Python int would be a Triton-version
    question this file cannot measure without a device, so the literal is written
    and pinned here instead.
    """
    assert folded_complex.MIRROR_SOURCE_INDEX == 2
    source = kernel_source_no_docstring()
    for lane in ("near_i", "near_j", "near_k"):
        assert f"{lane} = live & (" in source
    for coordinate in ("i", "j", "k"):
        assert (f"({coordinate} == {folded_complex.MIRROR_SOURCE_INDEX})"
                in source), coordinate
    for offset, stride in (("dn_x", "-2 * nyz"), ("dn_y", "-2 * nz"), ("dn_z", "-2")):
        assert f"{offset} = {stride}" in source, offset


def test_the_far_reflect_row_is_a_RUNTIME_argument_and_never_baked():
    """``n_full - stored + 2`` is ``stored - 2`` at an even full count and
    ``stored - 3`` at an odd one, so baking ``n - 2`` is a whole cell wrong on every
    odd-count run. The kernel takes ``rx``/``ry``/``rz`` as runtime scalars.
    """
    names = [argument.arg for argument in kernel_node().args.args]
    for name in ("rx", "ry", "rz"):
        assert name in names
        assert name not in [a.arg for a in kernel_node().args.args
                            if a.annotation is not None]
    source = kernel_source_no_docstring()
    assert "df_x = (nx - 1 - rx) * nyz" in source
    assert "df_y = (ny - 1 - ry) * nz" in source
    assert "df_z = nz - 1 - rz" in source or "df_z = (nz - 1 - rz)" in source
    for baked in ("nx - 2", "ny - 2", "nz - 2"):
        assert f"df_x = ({baked}" not in source


# ---------------------------------------------------------------------------
# The declaration must agree with the kernel — the sibling module's own defect
# ---------------------------------------------------------------------------

def test_REPLACES_names_every_pass_the_kernel_actually_carries():
    """TRAP: a module once carried ``fill_folded_far_ghosts_B`` while its
    ``REPLACES`` tuple said it did not, and a passing test pinned the stale tuple.
    So the tuple is checked AGAINST THE BODY here, not against a copy of itself.
    """
    source = kernel_source_no_docstring()
    assert product.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                                "fill_folded_far_ghosts_B", "update_H")
    # step_B: the curl and the split-field recurrence.
    assert "curl0_re, curl0_im = _mul_coefficient_left(dtdx" in source
    # fill_B: the NEAR carry, on every component.
    assert all(f"if NEAR_{axis}:" in source for axis in "XYZ")
    # zero_metal_B: the register clear, both planes.
    assert all(f"if ZM_{axis}:" in source for axis in "XYZ")
    # fill_folded_far_ghosts_B: the FAR carry, on every component.
    assert all(f"if FAR_{axis}:" in source for axis in "XYZ")
    # update_H: the constitutive accumulation at the owned cell AND at each ghost.
    assert "_carry_ghost_complex" in source
    assert source.count("_mul_coefficient_left(kp_") == 3


def test_the_far_carry_brings_the_top_plane_mask_with_it():
    """Carrying ``fill_folded_far_ghosts_B`` means admitting ``MIRROR_PERIODIC``,
    and the folded complex curl masks the TOP plane there (its DELTA 3). A carry
    that took the fill and left that block out would leave the plane unmasked.
    """
    source = kernel_source_no_docstring()
    certified = certified_source("folded_complex.py", "folded_bloch_pml_curl_step")
    # THREE guards on the BACKWARD arm and SIX on the forward one — the D family's
    # shift is 1 on its own axis alone and the B family's on the two others. Counted
    # against the certified emitter rather than written as a number here.
    assert certified.count("MIRROR_PERIODIC") == 9, certified.count("MIRROR_PERIODIC")
    assert source.count("MIRROR_PERIODIC") == 9, source.count("MIRROR_PERIODIC")
    for last in ("last_x", "last_y", "last_z"):
        assert f"{last}, 0.0" in source
        assert f"{last}, 0.0" in certified
    assert "last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1" in source


def test_the_backward_constexpr_is_step_Bs_and_only_step_Bs():
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.BACKWARD == int(SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 0
    assert product.CURL_SUB_STEP == "step_B"
    assert product.CONSTITUTIVE_SIDE == "H"


def test_the_fill_family_key_is_the_ghost_fill_tables_own_key():
    """Spelled out rather than reusing ``MAGNETIC_FIELD_TYPE``, which happens to be
    the same string for an unrelated reason."""
    assert product.FILL_FAMILY in folded_complex.GHOST_FILL_FAMILIES
    assert (folded_complex.GHOST_FILL_FAMILIES[product.FILL_FAMILY]["targets"]
            == ("Bx", "By", "Bz"))


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
    """
    certified = certified_source("folded_complex.py", "folded_bloch_pml_curl_step")
    ours = kernel_source_no_docstring()
    for line in CURL_LINES:
        assert line in certified, ("moved in folded_complex.py", line)
        assert line in ours, ("moved here", line)


def test_the_only_curl_edit_is_the_ownership_mask_on_the_B_load():
    """A destination lane never READS ``B`` — masked off, not merely unused.

    The certified curl loads ``f?`` under ``live``; this kernel loads it under
    ``own?``. That is the whole of the edit and it is asserted in both directions.
    """
    certified = certified_source("folded_complex.py", "folded_bloch_pml_curl_step")
    ours = kernel_source_no_docstring()
    for component, own in ((0, "own0"), (1, "own1"), (2, "own2")):
        assert f"e_re = tl.load(f{component} + 2 * idx, mask=live, other=0.0)" \
            in certified
        assert f"e_re = tl.load(f{component} + 2 * idx, mask={own}, other=0.0)" \
            in ours
        assert f"e_re = tl.load(f{component} + 2 * idx, mask=live, other=0.0)" \
            not in ours


def test_the_constitutive_half_is_the_certified_complex_pairs_own_body():
    """Read against :mod:`.complex_fused_magnetic_pair`, which is itself the
    certified ``bloch_constitutive_step`` ``SCALE = 0`` arm with ``src`` bound to a
    register. The only edit here is the mask.
    """
    certified = certified_source("complex_fused_magnetic_pair.py",
                                 "complex_fused_curl_constitutive_B")
    ours = kernel_source_no_docstring()
    for line in ("t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
                 "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)"):
        assert line in certified, line
        assert line in ours, line
    # `prev` is read BEFORE the `w` store — the one ordering the constitutive
    # cannot survive being wrong about.
    for component in range(3):
        prev = ours.index(f"prev_re = tl.load(w{component} + 2 * idx")
        store = ours.index(f"tl.store(w{component} + 2 * idx, src_re")
        assert prev < store, component


def test_the_ghost_constitutive_repeats_the_same_statements_in_the_same_order():
    """``_carry_ghost_complex`` is the owned cell's own sequence, moved.

    Seven inlined copies of the same statements would be seven places for one of
    them to drift, which is why it is a device function; that it IS the same
    sequence is asserted here.
    """
    helper = certified_source("folded_complex_fused_magnetic_pair.py",
                              "_carry_ghost_complex")
    order = [
        "prev_re = tl.load(w + 2 * dst",
        "tl.store(w + 2 * dst, ghost_re",
        "acc_re = tl.load(h + 2 * dst",
        "_mul_coefficient_left(kp_d, ghost_re, ghost_im, EXPANSION)",
        "_mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)",
        "tl.store(h + 2 * dst, acc_re",
        "tl.store(f + 2 * dst, ghost_re",
    ]
    positions = []
    for fragment in order:
        assert fragment in helper, fragment
        positions.append(helper.index(fragment))
    assert positions == sorted(positions), positions


def test_the_wall_clear_is_applied_to_the_registers_before_every_consumer():
    """``zero_metal_B`` runs BETWEEN the near and far fills (driver.py:3286), so the
    far carry must read a CLEARED ``v`` and the constitutive must too. Applying it
    to the registers before the store is what makes both true at once.
    """
    ours = kernel_source_no_docstring()
    clear = ours.index("if ZM_X:")
    store = ours.index("tl.store(f0 + 2 * idx, v0_re")
    constitutive = ours.index("src_re = v0_re")
    carry = ours.index("_carry_ghost_complex(f0, w0, h0, idx + df_y")
    assert clear < store < constitutive < carry, (clear, store, constitutive, carry)


# ---------------------------------------------------------------------------
# The seam predicate, in both directions
# ---------------------------------------------------------------------------

def test_a_folded_periodic_complex_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = fold()
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert residual(verdict) == [], residual(verdict)


def test_a_folded_metallic_complex_grid_is_admitted_too():
    """MIRROR_METALLIC: the near fill runs, the far one does not, and the top-plane
    mask is not emitted. Both codes are admitted."""
    fields, pml = fold(boundaries="metallic")
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert residual(verdict) == [], residual(verdict)


def test_a_magnetic_deposit_is_carried_and_an_index_less_one_is_refused_by_name():
    """THE 2026-08-31 FLIP, pinned in the direction it now points.

    ``CARRIES_DEPOSIT_REPAIR`` became True on this module, so "a magnetic
    source in this seam" stopped being a refusal class: one that PUBLISHES the
    index the injection writes is ADMITTED (the shipped repair carries it —
    the gate's carry legs measure that in bytes), and one that cannot is
    refused by name. Before the flip this test pinned the blanket refusal;
    keeping that pin would have frozen the board's one clause-refusal on this
    family (tests:TestHoleyWvgBands.test_fields_at_kx) into the suite.
    """
    fields, pml = fold()
    electric = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(Source("D"),), probe=_probe())
    assert residual(electric) == [], residual(electric)

    class _Indexed:
        field_type = "B"
        _point_ix, _point_iy, _point_iz = 2, 2, 0

    carried = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(_Indexed(),), probe=_probe())
    assert residual(carried) == [], residual(carried)
    index_less = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(Source("B"),), probe=_probe())
    assert any("does not publish the index" in reason
               for reason in index_less.reasons), index_less.reasons


def test_the_flag_at_False_would_cost_the_whole_cell(monkeypatch):
    """The measurement behind the flip, executed rather than restated: with the
    flag held False the magnetic deposit is refused outright, which is what
    priced tests:TestHoleyWvgBands.test_fields_at_kx at zero until 2026-08-31."""
    fields, pml = fold()

    class _Indexed:
        field_type = "B"
        _point_ix, _point_iy, _point_iz = 2, 2, 0

    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    refused = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(_Indexed(),), probe=_probe())
    assert any("is magnetic" in reason for reason in refused.reasons), (
        refused.reasons)


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = fold()
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=None, probe=_probe())
    assert any("cannot infer an empty" in reason for reason in verdict.reasons)


def test_real_storage_is_refused_because_the_real_folded_pair_owns_it():
    fields, pml = fold(complex_storage=False)
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert any("storage is real float32" in reason for reason in verdict.reasons)


def test_an_unfolded_grid_is_refused_because_this_is_a_composition_product():
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries="periodic", k_point=(0.1, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), sources=(), probe=_probe())
    assert any("no mirror plane is active" in reason for reason in verdict.reasons)


def test_a_probe_without_the_parity_pattern_refuses_this_product():
    """The base four license the two curl/constitutive halves and NOT the fill, and
    the carry calls the fill's own multiply — so a base-only record must refuse."""
    base_only = _probe(patterns=folded_complex.PROBE_PATTERNS)
    fields, pml = fold()
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=base_only)
    assert not verdict.covered
    assert any(folded_complex.PARITY_PROBE_PATTERN in reason
               for reason in verdict.reasons), verdict.reasons


def test_the_predicate_reports_every_halfs_reasons_with_its_side_named():
    fields, pml = fold(complex_storage=False)
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    prefixes = {reason.split(":")[0] for reason in verdict.reasons}
    assert "folded complex curl half" in prefixes
    assert "folded complex constitutive half" in prefixes
    assert "folded complex fill half" in prefixes


def test_no_active_pml_is_refused_because_this_is_the_split_field_product():
    fields, pml = fold(pml=0.0)
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert any("no active PML" in reason for reason in verdict.reasons)


def test_the_builder_returns_None_for_every_configuration_the_predicate_refuses():
    """``None`` is the only refusal: a caller that would otherwise have stepped
    correctly must never see an exception."""
    for kwargs in ({"complex_storage": False}, {"pml": 0.0}):
        fields, pml = fold(**kwargs)
        assert product.plan_folded_complex_fused_magnetic_pair(
            fields, pml, sources=(), probe=_probe()) is None
    fields, pml = fold()
    assert product.plan_folded_complex_fused_magnetic_pair(
        fields, pml, sources=(Source("B"),), probe=_probe()) is None


# ---------------------------------------------------------------------------
# The plan's own refusals
# ---------------------------------------------------------------------------

def _plan_kwargs(shape=(8, 8, 1), codes=(0, 3, 0),
                 zero_metal=(False, False, False),
                 phases=(None, None, None), reflect=(None, 4, None),
                 parity=None, volume=None):
    arrays = {}
    for name in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        arrays[name] = (np.zeros(shape, dtype=np.complex64) if volume is None
                        else volume)
    flat = {f"{stem}_{axis}": np.zeros(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz")
            for stem in ("kms", "sinv", "kps")}
    if parity is None:
        parity = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                  (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))
    return dict(arrays=arrays, curl_flat=flat, constitutive_flat=flat, codes=codes,
                zero_metal=zero_metal, phases=phases, parity=parity, dtdx=0.35,
                expansion=1, reflect=reflect)


def test_the_plan_refuses_a_reflect_row_it_cannot_own():
    for reflect in ((None, 7, None), (None, -1, None), (None, 99, None)):
        with pytest.raises(ValueError, match="reflect row"):
            product.plan_folded_complex_fused_magnetic_pair_from_arrays(
                **_plan_kwargs(reflect=reflect))


def test_the_plan_refuses_a_reflect_row_on_an_axis_that_has_no_far_ghost():
    with pytest.raises(ValueError, match="not folded PERIODIC but carries"):
        product.plan_folded_complex_fused_magnetic_pair_from_arrays(
            **_plan_kwargs(codes=(0, 2, 0), reflect=(None, 4, None)))


def test_the_plan_refuses_a_fold_that_also_carries_a_wall_clear():
    with pytest.raises(ValueError, match="fold and a wall clear"):
        product.plan_folded_complex_fused_magnetic_pair_from_arrays(
            **_plan_kwargs(zero_metal=(False, True, False)))


def test_the_plan_refuses_a_fold_that_also_carries_a_bloch_phase():
    with pytest.raises(ValueError, match="fold and a Bloch phase"):
        product.plan_folded_complex_fused_magnetic_pair_from_arrays(
            **_plan_kwargs(phases=(None, 1j, None)))


def test_the_plan_refuses_a_zero_parity_word_pair_on_a_folded_axis():
    parity = ((1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
              (-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0))
    with pytest.raises(ValueError, match="NEAR parity words"):
        product.plan_folded_complex_fused_magnetic_pair_from_arrays(
            **_plan_kwargs(parity=parity))
    parity = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
              (-1.0, 0.0), (0.0, 0.0), (-1.0, 0.0))
    with pytest.raises(ValueError, match="FAR parity words"):
        product.plan_folded_complex_fused_magnetic_pair_from_arrays(
            **_plan_kwargs(parity=parity))


def test_the_plan_derives_NEAR_and_FAR_from_the_codes_and_never_takes_them():
    plan = product.plan_folded_complex_fused_magnetic_pair_from_arrays(
        **_plan_kwargs(codes=(2, 3, 0)))
    assert plan.near == (True, True, False)
    assert plan.far == (False, True, False)
    assert "near" not in {name for name in
                          product.plan_folded_complex_fused_magnetic_pair_from_arrays
                          .__code__.co_varnames}


def test_the_plan_refuses_a_cell_count_that_overflows_the_word_addressing(tmp_path):
    # 2**31 complex cells is 16 GiB a volume. The plan reads only the shape and the
    # word views before it refuses, so every volume is ONE sparse file-backed map that
    # nothing reads: a Linux host refuses an anonymous 16 GiB allocation outright when
    # its memory is smaller than the request.
    shape = (2 ** 16, 2 ** 15, 1)
    volume = np.memmap(tmp_path / "volume.bin", dtype=np.complex64, mode="w+",
                       shape=shape)
    try:
        with pytest.raises(ValueError, match="int32 word"):
            product.plan_folded_complex_fused_magnetic_pair_from_arrays(
                **_plan_kwargs(shape=shape, codes=(0, 0, 0),
                               reflect=(None, None, None), volume=volume))
    finally:
        del volume


class _NamedCupy:
    """NumPy under CuPy's name — the ONE attribute the backend clause reads.

    The predicate asks ``getattr(grid.xp, "__name__", "")`` and nothing else, so
    this makes the engine route reachable on a laptop without pretending anything
    about device memory. It is used ONLY to build a plan whose HOST-side table is
    then inspected; nothing here launches.
    """

    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(np, name)


def test_the_engine_route_and_the_arrays_route_agree_on_the_parity_table():
    fields, pml = fold()
    assert product.plan_folded_complex_fused_magnetic_pair(
        fields, pml, sources=(), probe=_probe()) is None, \
        "a NumPy host must be refused; the engine route is device-only"
    object.__setattr__(fields.grid, "xp", _NamedCupy())
    plan = product.plan_folded_complex_fused_magnetic_pair(
        fields, pml, sources=(), probe=_probe())
    assert plan is not None
    assert plan.parity == product.parity_coefficient_words(fields.grid)
    assert plan.replaces == product.REPLACES
    # The folded axis is Y at phase +1: near (+1, +0), far (-1, +0).
    assert plan.parity[1] == (1.0, 0.0)
    assert plan.parity[4] == (-1.0, 0.0)
    # And nothing on the unfolded axes.
    assert plan.parity[0] == plan.parity[2] == (0.0, 0.0)
    # The reflect row is the ENGINE's, never recomputed here.
    from meep_gpu.stepping import _far_reflect_rows

    assert plan.reflect == tuple(-1 if row is None else int(row)
                                 for row in _far_reflect_rows(fields.grid))
    assert plan.near == (False, True, False)
    assert plan.far == (False, True, False)


def test_the_wall_clear_and_the_fold_are_disjoint_on_every_admitted_grid():
    """``_zero_metal`` skips a folded axis, so the two passes cannot meet. CHECKED
    on the grid rather than inferred from the construction."""
    for boundaries in ("periodic", "metallic"):
        fields, _pml = fold(boundaries=boundaries)
        walls = zero_metal_axes(fields.grid)
        for axis in range(3):
            if fields.grid.is_mirrored(axis):
                assert not walls[axis], (boundaries, axis)


# ---------------------------------------------------------------------------
# Deferral and hygiene
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-13 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; the 2026-09-13 phase B batch
    (``dispatch_fused_route_2026-09-13_phaseB``) drove it through the driver's own
    consults, so the label now names it in ``RELEASED_FUSED_ARMS``. The partition
    is still exactly one of two, in the other direction: the label is in
    ``ARM_CERTIFICATION`` (its ledger entry was cut from the phase B fleet artifact
    by ``seed_triton_welds.py``) and out of ``PENDING_DEVICE_GATE_ARMS``.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_complex_fused_magnetic_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_complex_fused_magnetic_pair"]
    assert row["module"] == "folded_complex_fused_magnetic_pair"
    assert row["builder"] == "plan_folded_complex_fused_magnetic_pair"
    label = row["label"]
    assert label == 'fused pair B (folded complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_this_module_touches_neither_dispatch_nor_the_other_track():
    text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for banned in ("meep_gpu.metal_kernels", "meep_gpu.cuda_kernels",
                   "..metal_kernels", "..cuda_kernels", "..fastpath"):
        assert not any(name.startswith(banned) for name in imported), imported


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert text.count("enable_fp_fusion=") == 1
    assert "ENABLE_FP_FUSION if guard is None else bool(guard)" in text


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """The laptop path: no Triton, a named ImportError from the kernel accessor,
    and a coverage verdict that still answers."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in [key for key in sys.modules
                 if key.endswith("folded_complex_fused_magnetic_pair")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    reloaded = importlib.import_module(
        "meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair")
    reloaded = importlib.reload(reloaded)
    assert reloaded.folded_complex_fused_curl_constitutive_B is None
    with pytest.raises(ImportError, match="needs the optional `triton` package"):
        reloaded.folded_complex_fused_curl_constitutive_B_kernel()
    fields, pml = fold()
    verdict = reloaded.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert isinstance(verdict.covered, bool)


def test_the_module_has_a_fingerprints_weld_that_pins_these_bytes():
    """RELEASED 2026-09-13, so a byte-identity claim IS now made about this module.

    Until the phase B batch this module carried NO ``fingerprints.json`` entry on
    purpose: its licensed bytes are a function of a PROBE-BOUND expansion arm, so a
    checked-in hash would have recorded a choice rather than a measurement, and the
    binding stood in the gate artifact's own ``source_sha256`` map instead. The
    batch ran this family's device gate to a release verdict and
    ``seed_triton_welds.py`` cut its weld from the 2026-09-12 identity fleet, keyed
    on the gate and pinning the module's checked-in digests. So "no checked-in hash
    may imply a claim" is now the POSITIVE contract: the ledger carries exactly this
    product's weld, pins the module by both its code and source digest, and pins the
    probe that measured it. Every recorded SOURCE digest is re-hashed against the
    checkout here, so a weld cut from bytes the tree has since moved past FAILS
    rather than passing on a stale record. (``code_sha256`` is an AST digest and is
    checked for PRESENCE, not re-hashed raw.)
    """
    import hashlib
    import json

    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    entries = [key for key in record
               if "folded_complex_fused_magnetic_pair" in key]
    assert entries == ["triton_folded_complex_fused_magnetic_pair_device_gate"], (
        "the ledger must carry exactly this product's weld", entries)
    weld = record["triton_folded_complex_fused_magnetic_pair_device_gate"]
    assert weld["status"] == "PASS", weld["status"]
    module_key = "meep_gpu/triton_kernels/folded_complex_fused_magnetic_pair.py"
    probe_key = "parity/meep_gpu/probe_triton_folded_complex_fused_magnetic_pair.py"
    assert module_key in weld["code_sha256"], sorted(weld["code_sha256"])
    assert module_key in weld["source_sha256"], sorted(weld["source_sha256"])
    assert probe_key in weld["source_sha256"], sorted(weld["source_sha256"])
    # The weld must describe THESE bytes: every recorded source digest re-hashed
    # against the tree, so a released weld cut from a checkout the tree has since
    # moved past fails here rather than passing on a stale record.
    drift = []
    for relative, want in sorted(weld["source_sha256"].items()):
        target = API_ROOT / relative
        got = (hashlib.sha256(target.read_bytes()).hexdigest()
               if target.exists() else "MISSING")
        if got != want:
            drift.append(relative)
    assert drift == [], ("the weld does not describe these bytes", drift)

    # AND THE RELEASE MEASURED SOMETHING. The weld pins the bytes; the phase-B
    # fleet artifact those bytes ran through is what proves the run was clean, and
    # that verification is kept rather than reduced to the weld's one-word status.
    # RE-POINTED 2026-09-13 to the phase-B fleet (shard C re-gated this probe after
    # it learned to record the device identity); the run is device_status RUN,
    # passed, released, and every BYTE leg is bit-identical with the fused kernel
    # launched once per complete step.
    gate_artifact = (PARITY_DIR / "results"
                     / "triton_fleet_2026-09-13_batch_C"
                     / "probe_triton_folded_complex_fused_magnetic_pair" / "gate.json")
    assert gate_artifact.exists(), gate_artifact
    gate = json.loads(gate_artifact.read_text(encoding="utf-8"))
    assert gate["device_status"] == "RUN", gate["device_status"]
    assert gate["passed"] is True
    assert gate["release"]["released"] is True, gate["release"]
    byte_legs = [row for row in gate["device_legs"] if not row.get("armed")]
    assert byte_legs
    for row in byte_legs:
        assert row["first_divergence"] is None, row["leg"]
        assert row["control_divergence"] is None, row["leg"]
        assert row["fused_kernel_launches"] == len(row["per_step"]), row["leg"]


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_complex_fused_magnetic_pair" in text
    assert "gate_provenance" in text


# ---------------------------------------------------------------------------
# The gate's own design, checked at the merge bar
# ---------------------------------------------------------------------------

def load_gate():
    spec = importlib.util.spec_from_file_location("probe_folded_complex_pair", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    """Seven legs, on this machine, with no CuPy and no Triton.

    They are the design's own checks — the transcription, the parity chain, its
    reachability, the seam binding, the expansion licence, the predicate battery and
    branch reachability — and running them at the merge bar is what stops a kernel
    edit from waiting for a device to be refused.
    """
    gate = load_gate()
    failures = {}
    for leg in gate.NO_DEVICE_LEGS:
        row = leg()
        if not row["passed"]:
            failures[row["leg"]] = row.get("findings")
    assert failures == {}, failures


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    """A rewrite that matches nothing measures nothing and reports UNCAUGHT."""
    gate = load_gate()
    source = gate.shipped_source_from_file()
    unarmed = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(source)
        if hits == 0 or mutated == source:
            unarmed.append(name)
    assert unarmed == [], unarmed


def test_every_mutation_is_scored_on_a_case_that_ENTERS_the_branch_it_rewrites():
    """TRAP, paid for on this track: a rewrite can hit REAL lines under a guard the
    scored case does not enter, and report UNCAUGHT while measuring nothing.

    For each mutation this resolves the constexpr guards enclosing the lines the
    rewrite touches, and evaluates them against the constexprs the scored case
    really compiles — read off a real grid by the gate's own ``case_constexprs``,
    not modelled here.
    """
    gate = load_gate()
    source = gate.shipped_source_from_file()
    environments = {case[0]: gate.case_constexprs(case) for case in gate.CASES}
    failures = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, _hits = rewrite(source)
        case_name, _case = gate.mutation_case_for(name)
        environment = dict(environments[case_name])
        for index in gate.rewritten_lines(source, mutated):
            for guard in gate.enclosing_constexpr_guards(source, index):
                if guard == "<else>":
                    continue
                try:
                    entered = bool(eval(guard, {"__builtins__": {}}, environment))
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{name}: guard {guard!r} unevaluable: {exc!r}")
                    continue
                if not entered:
                    failures.append(
                        f"{name} is scored on {case_name}, which does NOT enter "
                        f"`if {guard}:` — the rewrite would be dead code there and "
                        f"the row would read as an uncaught mutation")
    assert failures == [], failures


def test_every_mutation_case_names_a_real_case():
    gate = load_gate()
    declared = {name for name, _w, _e, _r in gate.mutation_table()}
    assert set(gate.MUTATION_CASE) <= declared, set(gate.MUTATION_CASE) - declared
    for mutation, case_name in gate.MUTATION_CASE.items():
        assert case_name in gate.CASES_BY_NAME, (mutation, case_name)
    assert gate.DEFAULT_MUTATION_CASE in gate.CASES_BY_NAME


def test_the_gate_declares_a_value_class_floor_for_every_class_it_runs():
    """A leg whose operands cannot contain the class it claims to test measures
    nothing, and reports a pass while doing it."""
    gate = load_gate()
    assert set(gate.CLASS_FLOOR) == set(gate.VALUE_CLASSES)
    rng = np.random.default_rng(gate.case_seed("floor-check", "x"))
    shape = (6, 6, 2)
    for value_class in gate.VALUE_CLASSES:
        hosts = gate.value_class_hosts(("a", "b"), shape, rng, value_class)
        census = gate.operand_census(hosts)
        for key, minimum in gate.CLASS_FLOOR[value_class].items():
            assert census[key] >= minimum, (value_class, key, census)
    # And the uniform class provably carries NEITHER, which is the whole reason the
    # other two exist.
    uniform = gate.operand_census(
        gate.value_class_hosts(("a", "b"), shape, rng, "uniform"))
    assert uniform["subnormals"] == 0, uniform
    assert uniform["negative_zeros"] == 0, uniform


def test_the_gates_seeds_are_sha256_digests_and_not_process_salted():
    """``hash()`` of a str is salted per process, so a record naming a seeded state
    would not identify the state it measured."""
    gate = load_gate()
    first = gate.case_seed("y_fold_periodic", "uniform")
    assert first == gate.case_seed("y_fold_periodic", "uniform")
    assert first != gate.case_seed("y_fold_periodic", "subnormal_band")
    assert first != gate.case_seed("y_fold_metallic", "uniform")
    expected = int.from_bytes(hashlib.sha256(
        f"{gate.SEED_ROOT}/y_fold_periodic/uniform".encode()).digest()[:8],
        "big") >> 1
    assert first == expected
    # `hash(` must not appear in the gate's CODE. Read with comments and docstrings
    # removed: the gate's own prose explains why `hash()` is refused, and a text
    # search over the raw source finds the explanation and calls it a use.
    code = ast.unparse(_strip_docstrings(ast.parse(
        GATE.read_text(encoding="utf-8"))))
    assert "hash(" not in code.replace("sha256(", "").replace("source_hashes(", "")


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return tree


def test_the_ordering_mutations_are_scored_under_a_class_that_can_SEPARATE_them():
    """A mutation scored on operands that cannot show it reports a NULL saying
    nothing — and here the class is decided by a MEASUREMENT, not by argument.

    ``parity_chain_associativity_leg`` sweeps the shipped parity multiply's two arms
    and counts 32 of 2048 words moved by a re-order and 34 by a fold. Reading which
    words move settles the class: with ``c_im`` bitwise ``+0.0`` the addend
    ``(c_im * z_im) * -1.0`` is a SIGNED ZERO whose sign follows ``z_im``, and
    ``fma(c_re, z_re, that)`` canonicalizes ``-0 + +0`` to ``+0``. On every normal
    operand the multiply is an exact sign flip and the composition commutes to the
    bit. So the defect lives at signed zeros, and the class these mutations are
    scored under must carry them.
    """
    gate = load_gate()
    rng = np.random.default_rng(gate.case_seed("class-check", "x"))
    shape = (6, 6, 2)
    for name in ("m1_parity_chain_reordered", "m2_parity_chain_folded"):
        value_class = gate.MUTATION_VALUE_CLASS.get(
            name, gate.DEFAULT_MUTATION_VALUE_CLASS)
        census = gate.operand_census(
            gate.value_class_hosts(("a",), shape, rng, value_class))
        assert census["negative_zeros"] > 0, (name, value_class, census)
    # And the sweep itself: the defect must be REAL at the word layer, or the
    # mutations could not be caught anywhere and "caught" would be the wrong
    # declaration.
    leg = gate.parity_chain_associativity_leg()
    assert leg["passed"], leg["findings"]
    assert all(row["reorder_differing_words"] > 0 for row in leg["rows"]), leg["rows"]
    assert all(row["fold_differing_words"] > 0 for row in leg["rows"]), leg["rows"]


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    gate = load_gate()
    driver_text = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    for name in gate.SEAM_PASSES:
        assert f"{name}(" in driver_text, name
    assert set(gate.RESIDENCY_NAME) == set(gate.SEAM_PASSES)
    assert tuple(gate.RESIDENCY_NAME[name] for name in gate.SEAM_PASSES) \
        == product.REPLACES


def test_the_gates_case_table_reaches_the_deepest_composition_the_kernel_emits():
    """Seven ghost cells owned by one source lane. Below that the triple-composite
    blocks are shipped and unexecuted."""
    gate = load_gate()
    depths = []
    for case in gate.CASES:
        environment = gate.case_constexprs(case)
        depths.append(max(environment["ghost_destinations"]))
    assert max(depths) == 7, depths


def test_the_gate_stamps_provenance_before_every_json_dump():
    text = GATE.read_text(encoding="utf-8")
    assert text.count("json.dump(") == 1
    assert "from gate_provenance import stamp as _stamp_provenance" in text
    stamp = text.index("_stamp_provenance(payload)")
    dump = text.index("json.dump(")
    assert stamp < dump


def test_the_gate_reports_no_release_without_a_device():
    """A laptop leg is evidence about the design; it is not a release."""
    text = GATE.read_text(encoding="utf-8")
    assert '"released": False' in text
    assert "this artifact releases nothing" in text


# ---------------------------------------------------------------------------
# The second arm label — the off-diagonal admission, and the hazard it rests on
# ---------------------------------------------------------------------------

def _offdiagonal(fields, grid):
    """Install an off-diagonal ``chi1inv`` row on a NumPy Fields."""
    epsilon = np.full(grid.shape, 2.0, dtype=np.float32)
    inverse = np.full(grid.shape, 0.5, dtype=np.float32)
    partner = np.full(grid.shape, 0.04, dtype=np.float32)
    fields.set_epsilon_volumes(
        {name: epsilon for name in ("Ex", "Ey", "Ez")},
        {name: inverse for name in ("Ex", "Ey", "Ez")},
        chi1inv_offdiagonal={"Ex": {"Ey": partner}, "Ey": {"Ex": partner}})
    assert fields.has_offdiagonal_epsilon
    return fields


def test_an_offdiagonal_chi1inv_row_is_admitted_through_the_second_arm_label():
    """TWO of the four corpus rows this product serves carry one.

    ``folded_complex_composition_curl_coverage`` refuses an off-diagonal row BY NAME
    — the clause is ``update_E``'s, where the row product reads neighbours through a
    transverse Yee average (stepping.py:1219-1254). The Triton tranche registers a
    SECOND family over the SAME kernel and the SAME plan class for exactly those
    rows, and this product takes either admission as a matched pair.
    """
    fields, pml = fold()
    _offdiagonal(fields, fields.grid)
    plain = folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_B", probe=_probe())
    assert any("off-diagonal chi1inv row is installed" in reason
               for reason in plain.reasons), plain.reasons
    offdiag = folded_complex.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe=_probe())
    assert residual(offdiag) == [], residual(offdiag)
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert residual(verdict) == [], residual(verdict)


def test_the_two_admissions_are_disjoint_so_the_pair_is_never_mixed():
    """Each refuses the other's configuration by name, so "either matched pair" is a
    choice between two labels for one kernel and not a widening."""
    for offdiagonal in (False, True):
        fields, pml = fold()
        if offdiagonal:
            _offdiagonal(fields, fields.grid)
        plain = folded_complex.folded_complex_composition_curl_coverage(
            fields, pml, "step_B", probe=_probe())
        other = folded_complex.folded_complex_offdiag_pml_curl_coverage(
            fields, pml, "step_B", probe=_probe())
        assert bool(residual(plain)) != bool(residual(other)), offdiagonal


def test_the_offdiagonal_admission_rests_on_update_H_reading_no_inverse_epsilon():
    """The hazard, CARRIED rather than waived — and checked in three places."""
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

    # 1. the constitutive side table binds no inverse epsilon at all
    assert "inverse" not in CONSTITUTIVE_SIDES["H"]
    assert set(CONSTITUTIVE_SIDES["H"]) == {"targets", "aux", "sources",
                                            "half_integer"}
    # 2. this module never asks Fields for one — checked on the CODE, with prose
    #    stripped: the docstrings say why an inverse epsilon is absent and a text
    #    search over the raw source finds the explanation and calls it a call.
    code = ast.unparse(_strip_docstrings(
        ast.parse(MODULE_PATH.read_text(encoding="utf-8"))))
    code = "\n".join(line for line in code.splitlines()
                     if not line.strip().startswith("#"))
    assert "inverse_epsilon_for" not in code
    assert "inv_eps" not in code
    # 3. the kernel binds no such pointer. The signature is enumerated in full and
    #    compared against the ONE list this family declares, so a new pointer of any
    #    kind fails here — `sinv*` is the PML's split-field sinv vector, not an
    #    inverse permittivity, and is named as such.
    names = [argument.arg for argument in kernel_node().args.args]
    pointers = names[:names.index("nx")]
    assert pointers == [
        "f0", "f1", "f2", "u0", "u1", "u2", "g0", "g1", "g2",
        "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
        "h0", "h1", "h2", "w0", "w1", "w2",
        "kp0", "km0", "kp1", "km1", "kp2", "km2",
    ], pointers
    # And the constitutive half's coefficient arguments are the INTEGER-lattice
    # kps/kms alone: no fourth volume rides in beside them.
    assert [name for name in pointers if name.startswith(("kp", "km"))] == [
        "kmx", "kmy", "kmz", "kp0", "km0", "kp1", "km1", "kp2", "km2"]


def test_the_predicate_reports_the_admission_that_got_furthest():
    """A reader must see the clause that actually bound, not the other label's
    'no off-diagonal row is installed'."""
    fields, pml = fold(complex_storage=False)      # refused by BOTH admissions
    verdict = product.folded_complex_fused_magnetic_pair_coverage(
        fields, pml, sources=(), probe=_probe())
    assert any("storage is real float32" in reason for reason in verdict.reasons)
    assert not any("must not overlap them" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_the_gates_case_table_carries_a_row_with_an_offdiagonal_chi1inv():
    """If the off-diagonal admission were wrong, this is the case it would show on."""
    gate = load_gate()
    assert gate.OFFDIAGONAL_CASES
    for name in gate.OFFDIAGONAL_CASES:
        assert name in gate.CASES_BY_NAME, name


def test_the_gate_names_evidence_for_every_unreached_mutation():
    """``unreached`` is a real defect this gate does not reach. Without evidence
    that the defect IS real it is a null with a longer word."""
    gate = load_gate()
    row = gate.mutation_vocabulary_leg()
    assert row["passed"], row["findings"]
    unreached = [entry for entry in row["rows"]
                 if entry["expectation"] == "unreached"]
    for entry in unreached:
        assert entry["evidence"], entry
        # The evidence must NAME what proves the defect real and NAME what holds the
        # property instead. A sentence that does neither is a null with a longer word.
        assert "parity_chain" in entry["evidence"], entry
        assert "words" in entry["evidence"], entry
