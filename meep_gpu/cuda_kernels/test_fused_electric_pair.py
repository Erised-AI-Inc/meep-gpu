"""The hand-CUDA fused pair on the ELECTRIC seam: predicate, lift and protocol.

WHAT THIS FILE SETTLES. ``fused_electric_pair`` is the first product this track has
put on ``step_D -> zero_metal_D -> update_E``, and the three questions that decide
whether it is real are all answerable without a device:

1. **Does the predicate refuse everything the launch cannot serve, BY NAME?** This is
   the only failure mode on this track that nothing else catches: a bit-comparison on
   an admitted row says nothing about a row that should never have been admitted. The
   fold refusal is the one that matters most here, because BOTH halves' own certified
   predicates ADMIT a fold -- so unlike the complex siblings this clause is reachable,
   and it is the only thing standing between the product and a run whose two mirror
   fills would be inside the launch and unperformed.

2. **Is the lift the certified text?** Every anchor the splice takes is asserted
   against the certified strings, and the three edits that are NOT verbatim -- the
   register capture, the seam, the sub-lattice rename -- are asserted to have landed
   where they were meant to and nowhere else. The sub-lattice rename is the one whose
   failure is silent: the two vectors are both real, both indexed the same way, and
   half a cell apart in the absorber profile.

3. **Does the two-consult protocol reproduce the driver's own order, bit for bit?**
   Driven the way ``FdtdDriver.step`` drives the electric seam, against the array
   path, on NumPy, with a null control that must diverge.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. The fused kernel's arithmetic is a DEVICE
claim and no test on this host may speak to it (``parity/meep_gpu/
gate_cuda_fused_electric_pair.py`` is where that is measured); what is settled here is
the predicate and the protocol, which is exactly the half a byte comparison is blind
to.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from .. import deposit_repair, stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from ..triton_kernels.launch import NoopPlan
from . import fused_pairs, in_seam_coverage
from . import fused_electric_pair as family
from ..test_deposit_repair import _build, _differing, _reference, _words
from .test_fused_pairs import _NumpyWearingCupysName, build


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def electric_source(grid, component="Ez"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def magnetic_source(grid, component="Hz"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


class _ArrayPair:
    """Stands in for the fused kernel: curl + wall clear + constitutive in one object.

    THE THING UNDER TEST IS THE PROTOCOL, NOT THE KERNEL -- the same stand-in
    ``test_fused_pairs`` uses on the magnetic seam, with :data:`family.REPLACES` read
    from the module rather than transcribed, so a pass added to or dropped from that
    tuple changes this stand-in without an edit here. A pass with no ``stepping``
    function is an ``AttributeError`` here rather than a silently shorter seam.
    """

    #: Which of ``REPLACES``' passes take the absorber layer, read off ``stepping``'s
    #: signatures rather than guessed.
    TAKES_PML = frozenset({"step_D", "update_E"})

    def __init__(self, fields, pml):
        self._fields, self._pml = fields, pml
        self.runs = 0

    def run(self, *_args, **_kwargs):
        self.runs += 1
        for name in family.REPLACES:
            pass_ = getattr(stepping, name)
            if name in self.TAKES_PML:
                pass_(self._fields, self._pml)
            else:
                pass_(self._fields)


# --------------------------------------------------------------------------
# 1. THE DECLARATIONS, AND THE WIRING THEY MAY NOT DRIFT FROM
# --------------------------------------------------------------------------

def test_the_product_declares_the_repair_and_the_block_that_brackets_it_exists():
    """The flag and the wiring change together or not at all.

    A product that declared ``CARRIES_DEPOSIT_REPAIR`` without the two slots would
    compute ``update_E`` against a pre-injection D and report success -- the exact
    failure ``deposit_repair`` exists to make impossible.
    """
    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert fused_pairs.FUSED_PRODUCTS[family.FAMILY]["curl_slot"] == family.SLOT
    assert family.SLOT == "step_D"
    assert fused_pairs.FUSED_PAIR_SEAMS["step_D"] == ("update_E", "D")
    assert family.FAMILY in fused_pairs.FUSED_PAIR_ARMS


def test_the_replaced_passes_are_a_contiguous_run_of_the_drivers_own_order():
    """``REPLACES`` is a claim about what one launch performed; it is checked against
    ``FdtdDriver.step``'s SOURCE rather than against a list retyped here.

    SINCE 2026-08-31 THE RUN IS GAPLESS. The two mirror fills sit between ``step_D``
    and ``update_E`` in the driver and are now CARRIED by the ownership inversion, so
    ``REPLACES`` is the whole contiguous run rather than a subsequence of it -- and
    that is asserted, because a gap here would be a pass the launch skipped while the
    array path performed it.
    """
    driver = pathlib.Path(stepping.__file__).with_name("driver.py").read_text(
        encoding="utf-8")
    step = driver.split("def step(", 1)[1].split("\n    def ", 1)[0]
    order = [name for name in ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                               "fill_folded_far_ghosts_D", "update_E")
             if f"{name}(self.fields" in step]
    assert order == ["step_D", "fill_symmetry_bc_D", "zero_metal_D",
                     "fill_folded_far_ghosts_D", "update_E"]
    assert list(family.REPLACES) == order, family.REPLACES


def test_the_absorb_declaration_matches_the_arms_the_predicate_conjoins():
    """``FUSED_PAIR_ARMS`` is READ FROM THE PREDICATE, not assigned to it.

    If ``covers_fused_electric_pair`` ever conjoined different arms, absorbing those
    two slots would substitute a numerical product no arm admitted.
    """
    from . import registry  # noqa: PLC0415 - the table, read for its labels

    tree = ast.parse(pathlib.Path(family.__file__).read_text(encoding="utf-8"))
    predicate = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "covers_fused_electric_pair")
    called = {node.func.id for node in ast.walk(predicate)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"covers_real_pml_curl", "covers_real_pml_constitutive"} <= called

    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == (
        by_predicate["covers_real_pml_curl"],
        by_predicate["covers_real_pml_constitutive"])


def test_the_two_seam_products_named_the_same_arms_sit_on_different_seams():
    """The one thing that would turn this product into a lost seam rather than a gain.

    ``install_fused_pairs`` leaves a seam UNFUSED when two products claim it, so a
    second product sharing the real magnetic pair's two arm LABELS is safe only
    because the two are keyed on different curl slots. Asserted rather than argued:
    the labels really are the same strings.
    """
    assert (fused_pairs.FUSED_PAIR_ARMS["cuda_fused_electric_pair"]
            == fused_pairs.FUSED_PAIR_ARMS["cuda_fused_magnetic_pair"])
    slots = {name: product["curl_slot"]
             for name, product in fused_pairs.FUSED_PRODUCTS.items()}
    assert slots["cuda_fused_electric_pair"] != slots["cuda_fused_magnetic_pair"]


# --------------------------------------------------------------------------
# 2. THE PREDICATE — WHAT IT ADMITS, AND WHAT IT REFUSES BY NAME
# --------------------------------------------------------------------------

def test_an_electric_deposit_is_admitted_now_and_would_be_refused_without_the_flag(
        xp, monkeypatch):
    """The whole point of the product, measured against the SHIPPED flag.

    75 of the 79 corpus rows on this cell carry an electric source in this seam. The
    same configuration is asked twice -- once with ``CARRIES_DEPOSIT_REPAIR`` held at
    ``False`` -- so the admission is attributed to the bracket rather than assumed.
    """
    fields, layer, grid = build(xp)
    sources = [electric_source(grid)]
    covered, reason = family.covers_fused_electric_pair(fields, layer, grid, sources)
    assert covered, reason

    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    refused, why = family.covers_fused_electric_pair(fields, layer, grid, sources)
    assert not refused
    assert "is electric" in why and "driver.py:3308" in why


def test_a_magnetic_source_never_disqualifies_this_seam(xp):
    """The mirror image of the magnetic pair's own clause, and the reason the shared
    ``deposit_repair`` helper is imported rather than re-spelled: a product that asked
    about the wrong seam would refuse the harmless source and admit the fatal one."""
    fields, layer, grid = build(xp)
    covered, reason = family.covers_fused_electric_pair(
        fields, layer, grid, [magnetic_source(grid)])
    assert covered, reason


def test_a_source_that_hides_the_index_it_writes_is_refused_by_name(xp):
    """The repair saves and restores the cells a deposit writes; a source that does
    not publish them leaves the fused launch's pre-injection ``update_E`` standing."""
    fields, layer, grid = build(xp)

    class _Opaque:
        """Electric by field_type, and publishing no deposit index at all."""

        field_type = electric_source(grid).field_type

    covered, why = family.covers_fused_electric_pair(fields, layer, grid, [_Opaque()])
    assert not covered
    assert "does not publish the index it writes" in why


def test_an_undeclared_source_set_is_still_a_refusal(xp):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so a
    predicate that inferred "no sources" from not being told would over-cover every
    row in the corpus that has one -- which on this seam is almost all of them."""
    fields, layer, grid = build(xp)
    covered, why = family.covers_fused_electric_pair(fields, layer, grid, None)
    assert not covered
    assert "was not declared" in why


@pytest.mark.parametrize("axis,name", ((0, "X"), (1, "Y"), (2, "Z")))
def test_a_folded_axis_is_admitted_on_every_axis_since_the_carry_landed(xp, axis, name):
    """THE REACHABLE WIDENING, and the 53 corpus rows it is worth.

    Until 2026-08-31 this product refused every folded grid by name, and that refusal
    was the ONLY thing standing between it and 53 of the 79 rows in its cell -- the
    largest single gap the hand-CUDA board carried. The refusal is gone because the
    carry is here, not because the clause was relaxed: what the predicate asks now is
    the two fills' OWN predicates, delegated to ``in_seam_coverage``.
    """
    fields, layer, grid = build(xp, boundaries=("periodic", "periodic", "periodic"),
                                symmetry=(Mirror(name),))
    assert grid.is_mirrored(axis), "the fixture did not fold the axis under test"
    covered, why = family.covers_fused_electric_pair(fields, layer, grid, ())
    assert covered, why


def test_the_fold_admission_is_this_products_own_and_not_a_half_predicates(xp):
    """The carry is only load-bearing if neither half already refused a fold.

    Measured rather than argued: both halves are asked directly on the same folded
    configuration and both ADMIT it. If either ever started refusing, the fold rows
    would be served by a conjunction that never reached the carry, and this test says
    so rather than leaving the reason to a docstring.
    """
    from .coverage import (covers_real_pml_constitutive,  # noqa: PLC0415
                           covers_real_pml_curl)

    fields, layer, grid = build(xp, symmetry=(Mirror("Y"),))
    curl_covered, curl_reason = covers_real_pml_curl(fields, layer, grid, "step_D")
    const_covered, const_reason = covers_real_pml_constitutive(
        fields, layer, grid, "E")
    assert curl_covered, curl_reason
    assert const_covered, const_reason


def _only_this_products_clauses(monkeypatch):
    """Both halves and both fill predicates answered "covered", so the clauses this
    MODULE adds are the ones under test.

    Without it the three clauses below sit behind a conjunction that refuses the same
    configurations one frame earlier -- for a DIFFERENT reason -- and they would rot
    unexercised while the test still passed. Only what this file asserts is stubbed;
    the shipped conjunction is unchanged and is pinned by the tests above.
    """
    monkeypatch.setattr(family, "covers_real_pml_curl",
                        lambda *a, **k: (True, "covered"))
    monkeypatch.setattr(family, "covers_real_pml_constitutive",
                        lambda *a, **k: (True, "covered"))
    monkeypatch.setattr(family, "covers_fill_symmetry",
                        lambda *a, **k: (True, "covered"))
    monkeypatch.setattr(family, "covers_fill_folded_far",
                        lambda *a, **k: (True, "covered"))


def test_a_folded_axis_too_short_for_the_near_source_row_is_refused_by_name(
        xp, monkeypatch):
    """``stepping._mirror_source`` raises below three stored cells; this kernel would
    index outside the volume instead, so the clause is stated in the predicate."""
    fields, layer, grid = build(xp, symmetry=(Mirror("Y"),))
    _only_this_products_clauses(monkeypatch)

    class _Short:
        """The folded grid, reporting two stored cells on the folded axis."""

        def __getattr__(self, item):
            return getattr(grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 1 else grid.stored_cells(axis)

    class _ShortFields:
        """...and a D volume the same two cells deep, so the two agree."""

        class _Dx:
            shape = None

        def __getattr__(self, item):
            return getattr(fields, item)

    short = _ShortFields()
    short.__dict__["Dx"] = type("_A", (), {
        "shape": tuple(2 if axis == 1 else int(fields.Dx.shape[axis])
                       for axis in range(3))})()
    covered, why = family.covers_fused_electric_pair(short, layer, _Short(), ())
    assert not covered
    assert "source row 2 does not exist" in why


def test_an_axis_reported_both_folded_and_walled_is_refused_by_name(xp, monkeypatch):
    """The carry RESTS on the two sets being disjoint: a D component's near axes ARE
    the axes ``zero_metal_D`` clears it on, so a folded-and-walled axis would put a
    near ghost in a plane the clear also owns and the kernel writes no wall line at a
    ghost. Unreachable from a real ``Grid`` (``zero_metal_axes`` is ``is_metallic and
    not is_mirrored``) and refused anyway."""
    fields, layer, grid = build(xp, symmetry=(Mirror("Y"),))
    _only_this_products_clauses(monkeypatch)
    monkeypatch.setattr(family, "zero_metal_axes",
                        lambda _grid: (False, True, False))

    covered, why = family.covers_fused_electric_pair(fields, layer, grid, ())
    assert not covered
    assert "both folded and walled" in why


def test_a_conductivity_is_refused_because_the_driver_deposits_it_differently(xp):
    """``fields.has_conductivity`` is the WHOLE ENGINE'S, not this sub-step's.

    The curl half refuses a conductivity on a D TARGET; a magnetic-side sigma alone
    leaves ``step_D`` admissible while still routing the electric deposit through
    ``_inject_electric_through_conductivity`` (driver.py:3305), which rescales the
    increment by ``condinv``. The repair has no verdict on that route, so the product
    refuses it here by name rather than reconstructing an unscaled increment.
    """
    fields, layer, grid = build(xp)

    class _Conductive:
        def __getattr__(self, item):
            return getattr(fields, item)

        has_conductivity = True

    covered, why = family.covers_fused_electric_pair(_Conductive(), layer, grid, ())
    assert not covered
    assert "condinv" in why and "driver.py:3305" in why


def test_a_grid_that_cannot_say_which_axes_are_walled_is_refused(xp):
    """``zero_metal_D`` is CARRIED, so an unanswerable grid would silently be treated
    as unwalled -- two planes of wrong values per walled axis, not a crash."""
    fields, layer, grid = build(xp)

    class _Mute:
        def __getattr__(self, item):
            return getattr(grid, item)

        has_metallic = None

    covered, why = family.covers_fused_electric_pair(fields, layer, _Mute(), ())
    assert not covered
    assert "zero_metal_D cannot be carried" in why


def test_a_stored_extent_that_disagrees_with_the_walked_shape_is_refused(xp):
    """The wall plane is derived from ``grid.stored_cells`` and indexed into
    ``Dx.shape``; a disagreement writes the wipe into an array this launch is not
    walking."""
    fields, layer, grid = build(xp)

    class _Shrunk:
        def __getattr__(self, item):
            return getattr(grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 0 else grid.stored_cells(axis)

    covered, why = family.covers_fused_electric_pair(fields, layer, _Shrunk(), ())
    assert not covered
    assert "stored_cells" in why


def test_an_inverse_permittivity_volume_of_the_wrong_extent_is_refused(xp):
    """The three inv_eps volumes are indexed with THIS launch's flat index, so a
    component whose volume is a different shape reads outside it."""
    import numpy  # noqa: PLC0415

    fields, layer, grid = build(xp)

    class _Ragged:
        def __getattr__(self, item):
            return getattr(fields, item)

        def inverse_epsilon_for(self, component):
            if component == "Ez":
                return numpy.zeros((2, 2, 2), dtype=numpy.float32)
            return fields.inverse_epsilon_for(component)

    covered, why = family.covers_fused_electric_pair(_Ragged(), layer, grid, ())
    assert not covered
    assert "inverse_epsilon_for" in why


# --------------------------------------------------------------------------
# 3. THE LIFT — THE CERTIFIED TEXT, AND THE THREE EDITS THAT ARE NOT VERBATIM
# --------------------------------------------------------------------------

def _string_constants(module_name):
    """Every module-level ``name = <string expression>`` in one sibling, by ``ast``.

    THE CERTIFIED TEXT WITHOUT THE DEVICE LIBRARY, AND WITHOUT A SKIP.
    ``step_curl_kernels`` and ``constitutive_kernels`` import CuPy at module scope, so
    on the merge-bar host they cannot be imported -- and a test that skipped for that
    reason would leave the SPLICE, the part of this product that can be silently
    wrong, unchecked everywhere a laptop runs. The device strings are plain
    module-level assignments, so they are read from the SOURCE instead, the same way
    ``registry``'s arm table is read by ``build_cuda_fusion_matrix``.

    Only ``str`` constants, ``Name`` references to ones already bound, and ``+`` over
    those are evaluated. Anything else is skipped rather than guessed: this is a
    reader for a known shape, not an interpreter.
    """
    path = pathlib.Path(family.__file__).with_name(f"{module_name}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = value(node.left), value(node.right)
            return None if left is None or right is None else left + right
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = value(node.value)
        if text is not None:
            bound[target.id] = text
    return bound


class _CertifiedText:
    """A stand-in for one CuPy-importing sibling, holding only its device strings."""

    def __init__(self, module_name):
        for name, text in _string_constants(module_name).items():
            setattr(self, name, text)


@pytest.fixture
def certified(monkeypatch):
    """The emitter, with both certified halves supplied from source."""
    monkeypatch.setattr(family, "step_curl_kernels",
                        family.step_curl_kernels or _CertifiedText("step_curl_kernels"))
    monkeypatch.setattr(family, "constitutive_kernels",
                        family.constitutive_kernels
                        or _CertifiedText("constitutive_kernels"))
    return family.fused_electric_pair_source()


def test_the_emitter_refuses_by_name_where_the_certified_text_is_unreachable():
    """THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is
    nothing to emit. The refusal has to name the modules, one frame from the caller,
    rather than surface as an AttributeError inside a string operation."""
    from . import fused_electric_pair as module  # noqa: PLC0415

    saved = (module.step_curl_kernels, module.constitutive_kernels)
    try:
        module.step_curl_kernels = None
        module.constitutive_kernels = None
        with pytest.raises(RuntimeError) as raised:
            module.fused_electric_pair_source()
        assert "step_curl_kernels" in str(raised.value)
        assert "constitutive_kernels" in str(raised.value)
        # And the predicate still answers, which is the reason those imports are
        # defensive in the first place.
        assert module.covers_fused_electric_pair(None, None, None, ())[0] is False
    finally:
        module.step_curl_kernels, module.constitutive_kernels = saved


def test_the_lift_is_nine_edits_and_no_silent_tenth():
    """``LIFT_EDITS`` is DATA so a gate can assert it. A tenth edit that appeared in
    the emitter without a row here would be certified text changed with no record.

    SIX UNTIL 2026-08-31; the fill carry added three -- the ownership guard inside
    ``pml_apply``, and the two mirror fills whose destination thread became their
    source thread.
    """
    assert len(family.LIFT_EDITS) == 9
    assert all(set(edit) == {"line", "became", "why"} for edit in family.LIFT_EDITS)


def test_the_shared_flux_density_is_bound_exactly_once(certified):
    """THE ALIASING HAZARD. Two ``__restrict__`` pointers to one allocation is UB that
    NVRTC miscompiles without a diagnostic, so the constitutive half must have no D
    source pointers at all -- which is only true if the seam removed all three reads."""
    source = certified
    signature = source.split("fused_electric_pair_pml_real(", 1)[1].split("\n) {", 1)[0]
    for target in ("Dx", "Dy", "Dz"):
        assert signature.count(f" {target},") + signature.count(f" {target}\n") == 1, (
            f"{target} is bound {signature.count(target)} times in the signature")
    body = source.split("\n) {", 1)[1]
    for target in ("Dx", "Dy", "Dz"):
        assert f"{target}[idx] *" not in body, (
            f"a constitutive read of {target} survived the seam rewrite")


def test_the_seam_reads_the_register_and_keeps_the_certified_operand_order(certified):
    """``D * inv_eps`` with D ON THE LEFT (stepping.py:1011). A float32 multiply is
    commutative on the bits, but the certified text is what this family lifts and a
    reordered operand is an edit with no row in ``LIFT_EDITS``."""
    source = certified
    for axis in ("x", "y", "z"):
        assert f"float src_{axis} = d_{axis} * inv_eps_E{axis}[idx];" in source


def test_the_constitutive_half_reads_the_half_integer_sub_lattice(certified):
    """THE COLLISION THAT MATTERS. ``kms_a`` is the INTEGER vector for the D curl and
    the HALF-INTEGER one for ``update_E``; letting one shadow the other compiles
    perfectly and is half a cell wrong in the absorber profile."""
    source = certified
    for axis, coordinate in (("x", "i"), ("y", "j"), ("z", "k")):
        assert (f"constitutive_apply(E{axis}, f_w_E{axis}, idx, src_{axis}, "
                f"kps_{axis}[{coordinate}], kms_half_{axis}[{coordinate}]);") in source
        # The curl half keeps the bare name, so both groups are present and distinct.
        assert f"kms_{axis}[" in source and f"kms_half_{axis}[" in source


def test_the_sub_lattice_pairing_is_the_mirror_image_of_the_magnetic_seams():
    """Read off the two table builders rather than asserted in prose: the D curl takes
    ``half_integer=False`` and ``update_E`` takes ``True``, which is the opposite of
    ``fused_magnetic_pair``'s pairing. A swap is a half-cell error, not a crash."""
    from .coverage import constitutive_sub_lattice  # noqa: PLC0415

    tree = ast.parse(pathlib.Path(family.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "fused_electric_pair_tables")
    literals = [node.value for node in ast.walk(function)
                if isinstance(node, ast.Constant) and isinstance(node.value, bool)]
    assert literals == [False, True], literals
    assert constitutive_sub_lattice("E") is True
    assert constitutive_sub_lattice("H") is False


def test_the_carried_wall_clear_is_the_off_diagonal_table_and_a_plus_zero():
    """``zero_metal_D`` wipes TWO components per wall -- the complement of the B
    family's diagonal. A pair that reused the diagonal would clear one wrong component
    and leave two right ones standing on every walled run."""
    carry = family.zero_metal_carry()
    assert "if (wall_x && i == 0) { clr_y = 1; clr_z = 1; }" in carry
    assert "if (wall_y && j == 0) { clr_x = 1; clr_z = 1; }" in carry
    assert "if (wall_z && k == 0) { clr_x = 1; clr_y = 1; }" in carry
    # THE STORE IS GUARDED ON OWNERSHIP. A cell one of the two fills images belongs to
    # its source thread; a second thread clearing the same word would be a race, and
    # the array path agrees -- the fill writes that cell at :3309 or :3311.
    for axis in ("x", "y", "z"):
        assert (f"if (own_{axis} && clr_{axis}) "
                f"{{ d_{axis} = 0.0f; D{axis}[idx] = d_{axis}; }}") in carry
    assert "-0.0f" not in carry, "the array path assigns the Python int 0, never -0.0"
    # The register is cleared BESIDE the store, because the constitutive half reads
    # the register and not the volume. One clear per component.
    assert carry.count("= 0.0f;") == 3


def test_the_near_carry_reads_the_pre_clear_register_and_the_far_carry_the_cleared_one():
    """THE DRIVER'S ORDER, AND IT IS MEASURABLY NOT A DETAIL.

    ``fill_symmetry_bc_D`` (driver.py:3309) runs BEFORE ``zero_metal_D`` (:3310) and
    ``fill_folded_far_ghosts_D`` (:3311) AFTER it. On the B family that ordering is
    unobservable: a B component's one near axis IS its own axis and the only axis
    ``zero_metal_B`` clears it on, and folded excludes walled, so a B near ghost never
    lands in a cleared plane. On the D family the near axes are the OTHER TWO --
    exactly the axes ``zero_metal_D`` clears the component on -- so a component can be
    folded on one and walled on the other and its near ghost lands in the cleared
    plane. The array path leaves ``+0.0f`` there; multiplying the POST-clear register
    by an ODD plane's parity leaves ``-0.0f``, a different word.

    Measured on the closed form, not asserted from this docstring:
    ``probe_cuda_electric_fill_carry_order.py`` runs both orderings against the array
    path's own three passes and records 0 differing words for the shipped one against
    9 and 11 for the naive one on the two odd-phase-fold-plus-wall configurations.
    What is pinned HERE is that the emitted text is the shipped ordering.
    """
    blocks = "\n".join(family.fill_carry_blocks(0))
    # The near-only ghost: pre-clear register, then the clear at the destination.
    assert "float gx_ny_v = phase_y * pre_x;" in blocks
    assert "if (clr_x) gx_ny_v = 0.0f;" in blocks
    # The far-only ghost: the post-clear register, and no further clear.
    assert "float gx_x_v = d_x;" in blocks
    assert "gx_x_v = (-phase_x) * gx_x_v;" in blocks
    assert "if (clr_x) gx_x_v" not in blocks
    # The composed ghost: near product, clear, then the far weight, in driver order.
    composed = blocks.split("if (reflect_x >= 0 && i == reflect_x && near_y && j == 2)",
                            1)[1]
    order = [composed.index("float gx_xny_v = phase_y * pre_x;"),
             composed.index("if (clr_x) gx_xny_v = 0.0f;"),
             composed.index("gx_xny_v = (-phase_x) * gx_xny_v;")]
    assert order == sorted(order), "the three steps are not in driver order"


def test_every_ghost_block_is_nested_inside_its_components_ownership_guard(certified):
    """A thread can be the SOURCE of one fill and the DESTINATION of the other -- at
    ``j == 2`` on a folded y while standing on ``i == nx - 1`` of a folded periodic x
    -- and unguarded it would write a ghost from a register that was never formed. The
    destination is already owned by the fully back-substituted thread."""
    source = certified
    for target, axis in enumerate(("x", "y", "z")):
        block = source.split(f"    if (own_{axis}) {{", 1)[1].split("\n    }", 1)[0]
        # Every carry block for this component lives inside the guard: its own cell
        # plus one per destination the closed form names.
        assert block.count(f"constitutive_apply(E{axis},") == 1 + len(
            family.carried_destinations(family.near_fill_axes(target),
                                        family.far_fill_axes(target)))
    # ...and NO constitutive_apply on E lives outside a guard.
    assert source.count("constitutive_apply(E") == sum(
        1 + len(family.carried_destinations(family.near_fill_axes(target),
                                            family.far_fill_axes(target)))
        for target in range(3))


def test_the_near_and_far_axis_sets_are_the_mirror_image_of_the_magnetic_familys():
    """READ OFF ``IYEE_SHIFTS``, both families, in one place. D takes TWO near planes
    and ONE far; B takes one near and two far. A carry that copied the B sets would
    image every component on the wrong faces -- and would still compile."""
    from . import fused_magnetic_pair as magnetic  # noqa: PLC0415

    for target in range(3):
        assert family.far_fill_axes(target) == (target,)
        assert set(family.near_fill_axes(target)) == {0, 1, 2} - {target}
        assert magnetic.near_fill_axes(target) == (target,)
        assert set(magnetic.far_fill_axes(target)) == {0, 1, 2} - {target}
    # Seven destinations per component on both, and they are not the same seven.
    assert len(family.carried_destinations((1, 2), (0,))) == 7
    assert len(magnetic.carried_destinations((0,), (1, 2))) == 7


def test_the_fill_plan_refuses_a_grid_the_predicate_would_have(xp, monkeypatch):
    """The launcher does not build a plan the predicate would have refused.

    A gate hands ``launch_fused_electric_pair`` its own arguments, so the plan builder
    is reachable WITHOUT the predicate and must refuse on its own. Every fact the carry
    rests on is asserted twice for that reason.
    """
    _fields, _layer, grid = build(xp, symmetry=(Mirror("Y"),))
    assert family.fused_electric_pair_fills(grid)["near"] == (0, 1, 0)

    monkeypatch.setattr(family, "zero_metal_axes", lambda _grid: (False, True, False))
    with pytest.raises(ValueError, match="both folded and walled"):
        family.fused_electric_pair_fills(grid)

    monkeypatch.setattr(family, "zero_metal_axes", lambda _grid: (False, False, False))
    monkeypatch.setattr(family, "mirror_fill_phases", lambda _grid: (None, 0, None))
    with pytest.raises(ValueError, match="mirror phase"):
        family.fused_electric_pair_fills(grid)

    monkeypatch.setattr(family, "mirror_fill_phases", lambda _grid: (None, None, None))
    monkeypatch.setattr(family, "folded_far_rows", lambda _grid: (None, 4, None))
    with pytest.raises(ValueError, match="far reflect row"):
        family.fused_electric_pair_fills(grid)


def test_the_fill_plan_is_the_same_reading_in_seam_coverage_gives(xp):
    """ONE SOURCE OF TRUTH for which axes each fill visits. The carry and the certified
    stand-alone passes read the same three functions, so they cannot disagree about
    which axes a pass visits, what parity each carries, or which row the far ghost
    images -- and the `-1` in ``reflect`` is a SENTINEL the kernel never reads (the
    guard that would read it is ``reflect_a >= 0``)."""
    for name in ("X", "Y", "Z"):
        _fields, _layer, grid = build(xp, symmetry=(Mirror(name),))
        plan = family.fused_electric_pair_fills(grid)
        phases = in_seam_coverage.mirror_fill_phases(grid)
        rows = in_seam_coverage.folded_far_rows(grid)
        assert plan["near"] == tuple(int(p is not None) for p in phases)
        assert plan["reflect"] == tuple(-1 if r is None else int(r) for r in rows)
        assert plan["phase"] == tuple(0.0 if p is None else float(p) for p in phases)


def test_every_ghost_stores_the_imaged_flux_density_as_well_as_running_update_E():
    """THE STORE TO D IS NOT OPTIONAL, and this test exists because it was missing.

    The first device run of the carry reported 297 differing words on a 9x7x11
    single-fold row -- exactly three ghost planes of 99 cells -- because the carry ran
    ``update_E`` at each ghost and never wrote the imaged displacement back. ``D`` at a
    ghost is what the NEXT timestep's curl differences, so leaving it holding the
    un-imaged value the curl wrote is a plane of wrong operands one step later, not a
    dead register.
    """
    for target, axis in enumerate(("x", "y", "z")):
        blocks = "\n".join(family.fill_carry_blocks(target))
        stores = [line.strip() for line in blocks.splitlines()
                  if line.strip().startswith(f"D{axis}[")]
        applies = [line for line in blocks.splitlines()
                   if "constitutive_apply(" in line]
        assert len(stores) == len(applies) == len(family.carried_destinations(
            family.near_fill_axes(target), family.far_fill_axes(target)))
        for store in stores:
            tag = store.split("[", 1)[1].split("]", 1)[0]
            assert store == f"D{axis}[{tag}] = {tag[:-2]}_v;", store


def test_the_ghost_takes_the_destinations_coefficient_pair_on_the_far_plane_only():
    """``update_E`` indexes ``E{m}`` on the component's OWN axis, and on this family
    the FAR fill is the one that moves it -- the mirror image of the B side, where the
    NEAR fill did. A far ghost reusing the source's pair would apply the reflect row's
    absorber profile to the top plane."""
    blocks = "\n".join(family.fill_carry_blocks(0))
    assert "kps_x[nx - 1], kms_half_x[nx - 1]);" in blocks
    assert "kps_x[i], kms_half_x[i]);" in blocks
    for line in blocks.splitlines():
        if "constitutive_apply(Ex," not in line:
            continue
        tag = line.split("f_w_Ex, ", 1)[1].split("_i,", 1)[0]
        # A tag naming the far axis takes nx - 1; every other tag takes i.
        far = tag.startswith("gx_x")
        assert ("kps_x[nx - 1]" in line) == far, line


def test_the_carried_wall_table_agrees_with_the_certified_pass():
    """The table is spelled in this module; the certified pass is the authority.

    Read from ``in_seam_passes``' own device text so a change there is a red test
    rather than a divergence: x wall -> Dy and Dz, y -> Dx and Dz, z -> Dx and Dy.
    """
    text = _string_constants("in_seam_passes")["_zero_metal_D_kernel_code"]
    for flag, _coordinate, axes in family._ZERO_METAL_ROWS:
        block = text.split(f"if ({flag})", 1)[1].split("}", 1)[0]
        for axis in ("x", "y", "z"):
            written = f"D{axis}[base] = 0.0f;" in block
            assert written == (axis in axes), (
                f"{flag}: the certified pass writes D{axis}={written} but this "
                f"module's table says {axis in axes}")


def test_the_walls_the_predicate_admits_are_the_walls_the_launch_binds():
    """One function decides which axes are walled, so the carry and the certified pass
    cannot disagree -- and a folded metallic axis is excluded by BOTH by construction,
    which is why this product's fold refusal makes the question moot rather than
    dangerous."""
    xp_ = _NumpyWearingCupysName()
    fields, _layer, grid = build(xp_, boundaries=("metallic", "periodic", "periodic"))
    assert in_seam_coverage.zero_metal_axes(grid) == (True, False, False)
    assert fields.Dx.shape == tuple(grid.stored_cells(axis) for axis in range(3))


# --------------------------------------------------------------------------
# 4. THE TWO-CONSULT PROTOCOL AGAINST THE DRIVER'S OWN ORDER
# --------------------------------------------------------------------------

#: The two grids the numeric leg runs on: the 2-D shape the sibling deposit tests use
#: and a 3-D one, because the deposit is a POINT and a 2-D grid leaves one axis with
#: no interior for an index error to land in.
GRIDS = ((1.2, 1.2, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_two_slots_consulted_as_the_driver_consults_them_match_the_array_path(cell):
    """THE ONE THAT MATTERS, and the only numeric claim this file makes.

    Two independently seeded copies of the same state. One is stepped in the DRIVER'S
    order -- curl, inject, fill, clear, far fill, constitutive. The other is stepped
    the way a fused pair owning both consults is: the whole seam closed against an
    UNINJECTED field at the first consult, then the driver's own inject/fill/clear,
    then the repair at the second. Every float32 word of all 24 arrays is compared as
    a uint32, so "close" is not a passing answer.

    THE ELECTRIC SEAM'S INJECTION IS AT ``when + 0.5*dt`` (driver.py:3303), not at
    ``when``, and ``_reference`` is the shared helper that knows it -- a protocol test
    that injected at the wrong instant would compare two different runs and could pass
    only by cancelling its own error.
    """
    grid_a, fields_a, pml_a = _build(cell=cell)
    grid_b, fields_b, pml_b = _build(cell=cell)
    src_a = [electric_source(grid_a)]
    src_b = [electric_source(grid_b)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"

    plans, selected = {}, {}
    inner = _ArrayPair(fields_b, pml_b)
    fused_pairs._install_fused_pair(plans, selected, fields_b, pml_b, src_b, "D",
                                    "step_D", "update_E", "fused electric pair", inner)
    assert isinstance(plans["step_D"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plans["update_E"], deposit_repair.TrailingRepairPlan)

    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        plans["step_D"].run()
        for source in src_b:
            source.inject(fields_b, step * dt + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
        plans["update_E"].run()

    differing = _differing(_words(fields_a), _words(fields_b))
    assert not differing, differing
    assert inner.runs == 8
    assert plans["step_D"].repairs > 0, "the repair reported touching no point"


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_same_protocol_without_the_repair_diverges(cell):
    """The null control. Without it the case above proves only that both routes ran."""
    grid_a, fields_a, pml_a = _build(cell=cell)
    grid_b, fields_b, pml_b = _build(cell=cell)
    src_a = [electric_source(grid_a)]
    src_b = [electric_source(grid_b)]
    inner = _ArrayPair(fields_b, pml_b)
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        inner.run()
        for source in src_b:
            source.inject(fields_b, step * dt + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
    differing = _differing(_words(fields_a), _words(fields_b))
    assert differing, ("the unrepaired protocol matched the reference, so the case "
                       "above proves nothing")
    # WHERE it lands is the discriminating half: the constitutive target the deposit
    # feeds and its PML auxiliary, which are exactly the two arrays `apply` restores.
    assert set(differing) == {"Ez", "f_w_Ez"}, differing


def test_a_clean_seam_keeps_the_cheap_sentinel_in_the_second_slot():
    """No deposit, no repair: the second consult must stay the no-op it always was."""
    _grid, fields, pml = _build()
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(plans, selected, fields, pml, (), "D", "step_D",
                                    "update_E", "fused electric pair",
                                    _ArrayPair(fields, pml))
    assert isinstance(plans["update_E"], NoopPlan)
    assert selected["step_D"] == selected["update_E"] == "fused electric pair"


def test_the_second_slot_refuses_when_the_first_never_ran():
    """``dispatch`` contracts that True means the whole sub-step ran; a silent no-op
    here would leave the constitutive half undone and report success."""
    grid, fields, pml = _build()
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(plans, selected, fields, pml,
                                    [electric_source(grid)], "D", "step_D",
                                    "update_E", "fused electric pair",
                                    _ArrayPair(fields, pml))
    with pytest.raises(deposit_repair.DepositNotRepairable):
        plans["update_E"].run()


# --------------------------------------------------------------------------
# 5. THE BLOCK, DRIVEN THROUGH THE COMPOSER
# --------------------------------------------------------------------------

def test_the_electric_seam_fuses_and_the_pair_reports_the_pass_it_carries(xp):
    """Through ``plan_step(fuse=True)``, which is the only route a caller has.

    ``zero_metal_D`` is in neither slot's name, so a composition that reported only
    the slots would say it ran on the array path when the kernel performed it.
    """
    from . import arms  # noqa: PLC0415

    fields, layer, grid = build(xp, boundaries=("metallic", "periodic", "periodic"))
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert plan.selected.get("step_D") == "fused electric pair"
    assert plan.selected.get("update_E") == "fused electric pair"
    assert "zero_metal_D" in fused_pairs.replaced_sub_steps(plan.plans)


def test_fusion_is_opt_in_on_this_seam_too(xp):
    """``fuse`` defaults to off, so the composition the coverage census measured is
    unchanged and this block adds nothing nobody asked for."""
    from . import arms  # noqa: PLC0415

    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=())
    assert plan.selected.get("step_D") != "fused electric pair"
