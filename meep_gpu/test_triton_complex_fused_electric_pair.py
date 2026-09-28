"""Tests for the COMPLEX fused ELECTRIC pair — complex ``step_D`` into ``update_E``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the two separately certified
complex products this launch replaces — lives in
``parity/meep_gpu/probe_triton_complex_fused_electric_pair.py``. A green suite here
is NOT a certification; the module under test says so and so does this file.

THE TWIN IS THE BASELINE, AND THE DIFFERENCES ARE THE POINT. This product is
:mod:`meep_gpu.triton_kernels.complex_fused_magnetic_pair` on the other seam, so
every test that would merely re-run the twin's is written instead as a comparison
against it — the two predicates must agree everywhere except where the seam differs,
and a test that asserted the D side alone could not tell "correct" from "identical
to a product that answers a different question".

What IS pinned here:

* the four things this body carries that the twin's does not — ``BACKWARD = 1``,
  ``SCALE = 1``, the ``inv_eps`` binding (RAW, indexed at ``+ idx``), and the
  MIRRORED coefficient lattices — each against the shipped source rather than a
  restatement of it;
* the seam — that ``REPLACES`` names the driver's own D-side call sites, that the
  two symmetry fills are inert on every configuration the halves admit, and that
  the wall clear is carried on BOTH word planes;
* the source clause in all three directions: an electric deposit CARRIED, a
  magnetic one irrelevant, and the carry gated on ``CARRIES_DEPOSIT_REPAIR`` —
  which is the clause the whole product turns on;
* every other clause of the seam predicate, in both directions, and that the plan
  builder refuses exactly where the predicate does;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module, and
  it makes no identity claim without a weld.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import pathlib
import json
import sys

import numpy as np
import pytest

from meep_gpu import expansion_refusal
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_fields
from meep_gpu.triton_kernels import complex_fused_electric_pair as product
from meep_gpu.triton_kernels import complex_fused_magnetic_pair as twin

PACKAGE_DIR = pathlib.Path(product.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
KEEP_PROBE = (API_ROOT / "parity" / "meep_gpu" / "results"
              / "expansion_probe_2026-08-17" / "expansion_probe_keep.json")
#: The board that priced this cell, and the line it priced it on. Read rather than
#: quoted: the module docstring names a ceiling of 16/16, and a docstring number
#: nothing re-derives is a number that goes stale silently.
BOARD = (API_ROOT / "parity" / "meep_gpu" / "results"
         / "fusion_matrix_triton_2026-08-30_before_electric" / "fusion_matrix.json")


def build(cell=(3.0, 3.0, 0.0), boundaries="periodic", k_point=(0.23, 0.0, 0.0),
          mirrors=(), thickness=0.4, complex_storage=True, dimensions=2):
    """A real complex Grid/Fields/PML triple on NumPy, with PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def probe_record():
    return json.loads(KEEP_PROBE.read_text(encoding="utf-8"))


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause.

    MATCHED ON THE CLAUSE, NOT ON THE WORD 'cupy'. The complex predicates' EXPANSION
    refusal names the backend the probe must be for, so a filter on the word swallows
    it — and a refusal that vanishes from the residual is a refusal the caller reads
    as an admission.
    """
    return [reason for reason in verdict.reasons
            if not ("array module is" in reason and "not cupy" in reason)]


def electric_deposit(fields, component="Ey"):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one — the repair refuses
    it by name for publishing no deposit index. A case about the carry needs the
    engine's own source, and one that deposits nothing would make the admitting
    assertion pass by measuring an empty scatter, so both are checked here rather
    than assumed at each call site.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component=component,
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def _kernel_ast(module, name):
    """The shipped kernel's parse tree, read off the FILE.

    Not off the object: ``triton`` is absent on this host, so the JIT kernel is
    ``None`` and ``inspect.getsource`` has nothing to read. The file is the thing
    the gate hashes anyway.
    """
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {module.__file__}")


# ---------------------------------------------------------------------------
# The two constexprs this body carries that the twin's does not
# ---------------------------------------------------------------------------

def test_the_backward_constexpr_is_step_Ds_and_only_step_Ds():
    """1, read off ``SUB_STEPS`` rather than spelled — and NOT the twin's 0.

    A backward sub-step differences the cell BELOW; bound to 0 this kernel would
    compute a forward curl that converges to a smooth, wrong field.
    """
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.BACKWARD == SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 1
    assert product.BACKWARD != twin.BACKWARD
    arrays, curl, constitutive = _bare_arrays()
    plan = product.plan_complex_fused_electric_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (None, None, None), 0.35, 1)
    assert plan.backward == 1


def test_the_scale_constexpr_is_the_E_sides_and_the_twin_carries_none():
    """``SCALE`` is 1 here, and the H-side twin does not have the argument at all.

    Read off :class:`.complex_fields.ComplexConstitutivePlan`'s own rule
    (``1 if side == "E" else 0``) rather than asserted as a literal, so the two
    cannot drift apart while both keep passing.
    """
    source = inspect.getsource(complex_fields.ComplexConstitutivePlan.__init__)
    assert '1 if side == "E" else 0' in source, source
    assert product.SCALE == 1
    assert not hasattr(twin, "SCALE"), (
        "the magnetic twin has gained a SCALE; this test's contrast is gone")
    arrays, curl, constitutive = _bare_arrays()
    plan = product.plan_complex_fused_electric_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (None, None, None), 0.35, 1)
    assert plan.scale == 1


def test_the_kernel_guards_the_inv_eps_load_on_SCALE_and_indexes_it_UNDOUBLED():
    """The one arithmetic difference, read off the shipped body.

    ``inv_eps`` is one REAL coefficient per COMPLEX cell (fields.py:1203-1204), so
    the load is ``e0 + idx``. Word-doubling it to ``2 * idx`` would read the
    imaginary neighbour's coefficient into the real plane — smooth, converged and
    wrong — and is exactly the kind of defect no shape check catches.
    """
    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    loads = [n for n in ast.walk(node)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "load"]
    inv_eps = [n for n in loads
               if isinstance(n.args[0], ast.BinOp)
               and getattr(n.args[0].left, "id", None) in {"e0", "e1", "e2"}]
    assert len(inv_eps) == 3, [ast.unparse(n) for n in loads]
    for call in inv_eps:
        assert ast.unparse(call.args[0]).replace(" ", "").endswith("+idx"), (
            ast.unparse(call))
        assert "2 *" not in ast.unparse(call.args[0]), ast.unparse(call)
    # ...and each of the three sits under an ``if SCALE:``, not in the open.
    guarded = [n for n in ast.walk(node)
               if isinstance(n, ast.If) and getattr(n.test, "id", None) == "SCALE"]
    assert len(guarded) == 3, ast.dump(node)
    for branch in guarded:
        assert any(call in ast.walk(branch) for call in inv_eps)


def test_the_inv_eps_multiply_is_field_LEFT_and_adds_no_fourth_orientation():
    """The SCALE arm's multiply is ``_mul_field_left``, which the curl half already
    calls six times — which is why :data:`PRODUCT_PROBE_PATTERNS` stays the BASE set.

    A different helper here would be a fourth operand orientation and would need its
    own probe pattern before it could be licensed under any arm.
    """
    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    guarded = [n for n in ast.walk(node)
               if isinstance(n, ast.If) and getattr(n.test, "id", None) == "SCALE"]
    for branch in guarded:
        called = {getattr(n.func, "id", None) for n in ast.walk(branch)
                  if isinstance(n, ast.Call)}
        assert "_mul_field_left" in called, ast.unparse(branch)
    assert product.PRODUCT_PROBE_PATTERNS == tuple(complex_fields.PROBE_PATTERNS)
    assert product.PRODUCT_PROBE_PATTERNS == twin.PRODUCT_PROBE_PATTERNS


def test_the_kernel_body_calls_only_the_three_licensed_multiply_helpers():
    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    called = {getattr(n.func, "id", None) for n in ast.walk(node)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    multiplies = {name for name in called
                  if name and (name.startswith("_mul_") or name.startswith("_rotate_"))}
    assert multiplies <= set(product.LICENSED_MULTIPLY_HELPERS), multiplies
    assert multiplies == set(product.LICENSED_MULTIPLY_HELPERS), (
        "a licensed helper is declared but never called; the declaration is wider "
        "than the body")
    assert set(product.LICENSED_MULTIPLY_HELPERS) == set(twin.LICENSED_MULTIPLY_HELPERS)


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_passes_the_driver_actually_calls():
    """``REPLACES`` is the driver's D-side call list, read from ``driver.step``.

    Not compared against a remembered list: the source is parsed and the five names
    are required to appear, in order, between ``step_D`` and ``update_E``.
    """
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    source = pathlib.Path(driver_module.__file__).read_text(encoding="utf-8")
    body = source.split("step_D(self.fields, self.pml)", 1)[1]
    body = body.split("update_P", 1)[0]
    for name in ("fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
                 "update_E"):
        assert name in body, name
    assert product.REPLACES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")
    # The two products own DISJOINT seams: a name in both would mean one of them
    # replaces a pass the other also claims, which the composer could not resolve.
    assert not set(product.REPLACES) & set(twin.REPLACES)
    assert product.REPLACES == product.ComplexFusedElectricPairPlan.replaces


def test_both_symmetry_fills_return_on_the_has_symmetry_guard():
    """The two passes this weld does NOT carry are inert on every admitted
    configuration — measured on the shipped functions, not assumed."""
    from meep_gpu import stepping  # noqa: PLC0415

    fields, _pml = build()
    assert not fields.grid.has_symmetry()
    before = {name: np.array(getattr(fields, name)) for name in ("Dx", "Dy", "Dz")}
    stepping.fill_symmetry_bc_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    for name, was in before.items():
        assert np.array_equal(np.asarray(getattr(fields, name)), was), name


def test_the_wall_clear_is_carried_on_BOTH_word_planes():
    """``zero_metal_D`` writes complex zero, so the register clear must touch the
    imaginary plane too. A clear applied to the real words alone leaves a wall
    plane holding a live imaginary field, which is invisible in any real-valued
    check."""
    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    zm = [n for n in ast.walk(node)
          if isinstance(n, ast.If) and getattr(n.test, "id", "").startswith("ZM_")]
    assert len(zm) == 3, ast.dump(node)
    for branch in zm:
        targets = {ast.unparse(stmt.targets[0]) for stmt in branch.body
                   if isinstance(stmt, ast.Assign)}
        assert any(name.endswith("_re") for name in targets), targets
        assert any(name.endswith("_im") for name in targets), targets


def test_the_wall_clear_takes_the_D_SIDE_component_map_and_not_the_twins():
    """THE DEFECT THIS MODULE SHIPPED TO ITS FIRST DEVICE RUN, pinned so it cannot
    come back.

    ``stepping._zero_metal`` clears every component whose Yee shift on the WALLED
    axis is zero. For B that is the component on its own axis — an x wall clears
    ``Bx`` alone. For D it is the two TANGENTIAL ones — an x wall clears ``Dy`` and
    ``Dz`` and leaves ``Dx``. The two maps are complements, and carrying the twin's
    writes zero into a plane the array path keeps AND leaves live a plane the array
    path clears: two errors in opposite directions on the same grid, confined to one
    cell of one plane per axis, invisible on every periodic case.

    Derived from ``IYEE_SHIFTS`` here rather than typed, so this test measures the
    engine's rule and not a second copy of the module's own claim.
    """
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, name in enumerate(product.CURL_TARGETS)
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    assert derived == product.WALL_CLEARED_COMPONENTS == ((1, 2), (0, 2), (0, 1))
    # ...and it really is the twin's complement, not the same map by luck.
    magnetic = tuple(
        tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    assert magnetic == ((0,), (1,), (2,))
    for axis in range(3):
        assert not set(derived[axis]) & set(magnetic[axis]), axis

    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    body = ast.unparse(node)
    for axis, coordinate in enumerate(("at_x", "at_y", "at_z")):
        for component in range(3):
            cleared = any(
                f"v{component}_{plane} = tl.where({coordinate}, 0.0, "
                f"v{component}_{plane})" in body for plane in ("re", "im"))
            assert cleared is (component in derived[axis]), (axis, component)


def test_the_predicate_refuses_when_the_baked_wall_map_leaves_the_engines():
    """The clause is only worth having if it fires. ``IYEE_SHIFTS`` is the engine's
    and cannot be edited from here, so the module's baked copy is moved instead —
    the same disagreement, reached from the other side."""
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        record = probe_record()
        before = product.complex_fused_electric_pair_coverage(fields, pml, (),
                                                              probe=record)
        saved = product.WALL_CLEARED_COMPONENTS
        product.WALL_CLEARED_COMPONENTS = ((0,), (1,), (2,))  # the twin's map
        try:
            after = product.complex_fused_electric_pair_coverage(fields, pml, (),
                                                                 probe=record)
        finally:
            product.WALL_CLEARED_COMPONENTS = saved
    assert residual(before) == []
    assert not after.covered
    assert any("zero one plane and leave another live" in reason
               for reason in after.reasons), after.reasons


def test_the_zero_metal_axes_rule_is_imported_and_not_respelled():
    from meep_gpu.triton_kernels import coverage as coverage_module  # noqa: PLC0415

    assert product.zero_metal_axes is coverage_module.zero_metal_axes


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_a_complex_bloch_grid_is_admitted_modulo_the_numpy_host():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert residual(verdict) == []


def test_the_seam_clauses_add_nothing_the_twin_does_not_also_refuse():
    """The two predicates must agree on every configuration where the seam is not
    the difference. A D-side clause that fired on a plain complex Bloch grid would
    be a refusal the board could not see and nobody would look for."""
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        record = probe_record()
        mine = product.complex_fused_electric_pair_coverage(fields, pml, (),
                                                            probe=record)
        theirs = twin.complex_fused_magnetic_pair_coverage(fields, pml, (),
                                                           probe=record)
    assert residual(mine) == residual(theirs) == []


def test_an_electric_deposit_is_CARRIED_and_a_magnetic_source_is_irrelevant():
    """THE CLAUSE THE WHOLE PRODUCT TURNS ON, in all three directions.

    The driver injects an electric source between the two halves (driver.py:3296)
    and a magnetic one in the B/H seam. A predicate that refused both would be
    correct and useless; one that refused neither would drop a deposit. Fifteen of
    the sixteen complex corpus rows declare an electric source, so this is the
    clause that decides whether the product is worth 1 row or 16.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        record = probe_record()
        # (1) the OTHER seam's source: not this pair's business at all.
        magnetic = product.complex_fused_electric_pair_coverage(
            fields, pml, (Source("B"),), probe=record)
        # (2) a REAL electric deposit in this seam: CARRIED by the repair.
        carried = product.complex_fused_electric_pair_coverage(
            fields, pml, (electric_deposit(fields),), probe=record)
        # (3) an electric source that publishes no index: STILL REFUSED BY NAME.
        opaque = product.complex_fused_electric_pair_coverage(
            fields, pml, (Source("D"),), probe=record)
    assert residual(magnetic) == []
    assert residual(carried) == [], residual(carried)
    assert not opaque.covered
    assert any("does not publish the index it writes" in reason
               for reason in opaque.reasons), opaque.reasons


def test_holding_the_carry_flag_False_puts_the_electric_seam_refusal_straight_back(
        monkeypatch):
    """The admission above is the FLAG's doing, not a clause that quietly went away.

    The refusal prose names the driver line, because that is what a reader needs
    to check the claim against.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
        held = product.complex_fused_electric_pair_coverage(
            fields, pml, (electric_deposit(fields),), probe=probe_record())
    assert not held.covered
    assert any("is electric" in reason and "BETWEEN step_D and update_E" in reason
               for reason in held.reasons), held.reasons


def test_the_declared_flag_and_the_wiring_argument_move_together():
    """``CARRIES_DEPOSIT_REPAIR`` is only honest while the seam clause is passed it.

    A module that declared True and then passed a literal to
    ``seam_source_reasons`` would admit every deposit and bracket none.
    """
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source
    assert "carries_repair=True" not in source


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(
            fields, pml, None, probe=probe_record())
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_registered_susceptibility_is_refused_by_this_products_own_clause():
    """A D->E-only clause with no counterpart on the magnetic twin.

    With a pole registered the constitutive source is ``(D - sum P)``
    (fields.py:1096-1105) and this kernel bakes the plain product. The E-side half
    already refuses it; the clause is restated here because a reader must not have
    to chase another predicate to learn what this kernel computes.
    """
    class _Pole:
        pass

    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        object.__setattr__(fields, "polarizations", (_Pole(),))
        verdict = product.complex_fused_electric_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert not verdict.covered
    assert any("D and not (D - sum P)" in reason for reason in verdict.reasons), (
        verdict.reasons)


def test_a_fields_that_cannot_hand_over_inv_eps_is_refused_not_crashed():
    """The SCALE=1 arm loads three volumes the H side never binds. An absent
    accessor must be a REFUSAL, not a TypeError raised inside the builder after
    the predicate said yes."""
    class _NoEps:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        inverse_epsilon_for = None

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(
            _NoEps(fields), pml, (), probe=probe_record())
    assert not verdict.covered
    assert any("inverse_epsilon_for" in reason for reason in verdict.reasons), (
        verdict.reasons)


def test_a_fold_is_refused_by_this_products_own_clause_and_not_only_by_the_halves():
    """The seam clause must name the two D-side passes, not lean on the halves'
    wording. If the complex tranche ever admits a fold, the halves' refusal
    disappears and this weld would silently swallow two passes that had started
    doing work."""
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build(boundaries=("metallic", "metallic", "periodic"),
                            k_point=(0.0, 0.0, 0.0), mirrors=(("Y", 1),))
        verdict = product.complex_fused_electric_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert not verdict.covered
    seam = [reason for reason in verdict.reasons
            if "fill_symmetry_bc_D" in reason or "symmetry passes in this seam" in reason]
    assert seam, verdict.reasons


def test_a_missing_expansion_probe_is_a_refusal_not_a_guessed_arm():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(fields, pml, (),
                                                               probe={})
    assert not verdict.covered
    assert any("expansion probe artifact" in reason for reason in verdict.reasons)


def test_no_pml_is_refused_because_update_E_is_a_no_op_without_one():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(
            fields, None, (), probe=probe_record())
    assert not verdict.covered
    assert any("PML" in reason for reason in verdict.reasons)


def test_real_storage_is_refused_this_is_the_complex_arm():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build(complex_storage=False)
        verdict = product.complex_fused_electric_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert not verdict.covered


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_electric_pair_coverage(fields, pml, ())
    prefixes = {reason.split(":")[0] for reason in verdict.reasons}
    assert "complex curl half" in prefixes, verdict.reasons
    assert "complex constitutive half" in prefixes, verdict.reasons


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    """``None`` is the only refusal, and it must track the predicate exactly.

    A licence refused at the coverage seam but still bound at the plan seam is how
    an unlicensed arm reaches a kernel.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        folded_fields, folded_pml = build(
            boundaries=("metallic", "metallic", "periodic"),
            k_point=(0.0, 0.0, 0.0), mirrors=(("Y", 1),))
        record = probe_record()
        for args, kwargs in (
            ((fields, pml, (Source("D"),)), {"probe": record}),
            ((fields, pml, None), {"probe": record}),
            ((fields, pml, ()), {"probe": {}}),
            ((fields, None, ()), {"probe": record}),
            ((folded_fields, folded_pml, ()), {"probe": record}),
        ):
            assert product.plan_complex_fused_electric_pair(*args, **kwargs) is None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _bare_arrays(shape=(4, 5, 6)):
    names = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
             "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    arrays = {name: np.ascontiguousarray(np.zeros(shape, dtype=np.complex64))
              for name in names}
    # inv_eps is float32 with ONE coefficient per complex cell — same shape, not
    # the doubled word shape.
    arrays.update({f"inv_eps_{name}": np.ones(shape, dtype=np.float32)
                   for name in ("Ex", "Ey", "Ez")})
    curl = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    constitutive = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
                    for index, axis in enumerate("xyz") for stem in ("kps", "kms")}
    return arrays, curl, constitutive


def test_the_phase_table_is_rounded_to_complex64_AND_conjugated_for_this_backward_step():
    """Backward sub-step: the imaginary part is NEGATED, the conjugate of
    ``_shift_down`` (complex_fields:1818-1822).

    This is the twin's test with the sign flipped, and the flip is the assertion:
    a forward-convention phase on a backward curl is a wrong Bloch factor on
    exactly one wrapped plane per axis — a boundary-only error that a bulk field
    comparison can miss entirely.
    """
    arrays, curl, constitutive = _bare_arrays()
    phase = complex(0.5, 0.25)
    plan = product.plan_complex_fused_electric_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (phase, None, None), 0.35, 1)
    assert plan.phased == (1, 0, 0)
    assert plan.phase_values[:2] == (float(np.float32(phase.real)),
                                     -float(np.float32(phase.imag)))
    # ...and the twin, on the same phase, does NOT negate — which is what makes
    # the line above a statement about this sub-step and not about the helper.
    twin_arrays = {name.replace("D", "B").replace("Ex", "Hx").replace("Ey", "Hy")
                   .replace("Ez", "Hz"): value for name, value in arrays.items()}
    twin_arrays.update({name: arrays[f"inv_eps_E{axis}"].astype(np.complex64)
                        for name, axis in (("Ex", "x"), ("Ey", "y"), ("Ez", "z"))})
    twin_arrays.update({f"f_w_H{axis}": np.zeros((4, 5, 6), dtype=np.complex64)
                        for axis in "xyz"})
    twin_plan = twin.plan_complex_fused_magnetic_pair_from_arrays(
        twin_arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (phase, None, None), 0.35, 1)
    assert twin_plan.phase_values[1] == float(np.float32(phase.imag))
    # An unphased axis passes (1.0, 0.0) under PH = 0 and emits no multiply.
    assert plan.phase_values[2:] == (1.0, 0.0, 1.0, 0.0)


def test_the_engine_route_binds_the_MIRRORED_lattices_and_not_the_twins():
    """The curl takes the INTEGER lattice on ``step_D`` and the constitutive the
    HALF-INTEGER one on E — the opposite pair from the magnetic weld.

    Read off the shipped builders' source rather than run, because the plan builder
    refuses on a NumPy host. Swapped, this is a half-cell error in the absorber
    profile: converged, smooth, and wrong.
    """
    from meep_gpu.triton_kernels.launch import SUB_STEPS  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: PLC0415

    assert SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["E"]["half_integer"] is True
    mine = inspect.getsource(product.plan_complex_fused_electric_pair)
    theirs = inspect.getsource(twin.plan_complex_fused_magnetic_pair)
    # The curl half interpolates the sub-step's own suffix in BOTH modules...
    assert "{curl_spec['suffix']}" in mine and "{curl_spec['suffix']}" in theirs
    # ...and the constitutive half is where they differ: `_h` here, bare there.
    assert 'f"{stem}_{axis}_h"' in mine, mine
    assert 'f"{stem}_{axis}"' in theirs and 'f"{stem}_{axis}_h"' not in theirs


def test_the_plan_refuses_a_placeholder_inv_eps_binding():
    """``ComplexConstitutivePlan`` may bind the source views as inv_eps placeholders
    because its ``SCALE`` constexpr compiles the loads away. THIS product is E-only,
    so a placeholder would be read as an epsilon on every launch."""
    arrays, curl, constitutive = _bare_arrays()
    shape = arrays["Dx"].shape
    with pytest.raises(ValueError, match="placeholder"):
        product.ComplexFusedElectricPairPlan(
            shape, 0.35, (0, 0, 0), (False, False, False), (0, 0, 0),
            (1.0, 0.0, 1.0, 0.0, 1.0, 0.0), 1, 256,
            [arrays[n] for n in ("Dx", "Dy", "Dz")],
            [arrays[n] for n in ("fu_Dx", "fu_Dy", "fu_Dz")],
            [arrays[n] for n in ("Hx", "Hy", "Hz")],
            [curl[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
            [arrays[n] for n in ("Ex", "Ey", "Ez")],
            [arrays[n] for n in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
            None,
            [constitutive[f"{stem}_{axis}"] for axis in "xyz"
             for stem in ("kps", "kms")])


def test_the_plan_refuses_a_cell_count_whose_word_index_overflows_int32():
    """``2 * idx`` is computed in int32; the bound is halved to compensate.

    The from-arrays route runs no predicate at all, so the bound is checked in the
    plan rather than only in coverage — an overflowed word index is a wild write,
    not a wrong number.
    """
    arrays, curl, constitutive = _bare_arrays()

    class _Huge:
        shape = (2 ** 11, 2 ** 11, 2 ** 10)  # 2**32 cells
        dtype = np.dtype("complex64")

        class flags:
            c_contiguous = True

        def view(self, _dtype):
            return self

    arrays = dict(arrays)
    arrays["Dx"] = _Huge()
    with pytest.raises(ValueError, match="overflows"):
        product.plan_complex_fused_electric_pair_from_arrays(
            arrays, curl, constitutive, (0, 0, 0), (False, False, False),
            (None, None, None), 0.35, 1)


def test_a_real_storage_volume_is_refused_at_the_word_view():
    arrays, curl, constitutive = _bare_arrays()
    arrays = dict(arrays)
    arrays["Ex"] = np.zeros((4, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="complex64"):
        product.plan_complex_fused_electric_pair_from_arrays(
            arrays, curl, constitutive, (0, 0, 0), (False, False, False),
            (None, None, None), 0.35, 1)


def test_the_run_passes_both_constexprs_and_every_pointer_group_in_order(monkeypatch):
    """The launch signature, checked without a device.

    A pointer group passed in the wrong order types fine and computes a different
    field; this is the one place the order is stated in one line each and can be
    compared against the kernel's own parameter list.

    ``kernels`` is stubbed because ``run`` imports ``ENABLE_FP_FUSION`` from it and
    that module imports Triton at its top. The stub supplies the ONE name and
    nothing else, so a body that started reading a second thing from ``kernels``
    fails here rather than being silently accommodated.
    """
    import types  # noqa: PLC0415

    stub = types.ModuleType("meep_gpu.triton_kernels.kernels")
    stub.ENABLE_FP_FUSION = False
    monkeypatch.setitem(sys.modules, "meep_gpu.triton_kernels.kernels", stub)
    arrays, curl, constitutive = _bare_arrays()
    plan = product.plan_complex_fused_electric_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (None, None, None), 0.35, 1)
    seen = {}

    class _Recorder:
        def __getitem__(self, _grid):
            def launch(*args, **kwargs):
                seen["args"] = args
                seen["kwargs"] = kwargs
            return launch

    plan._kernel = _Recorder()
    plan.run()
    node = _kernel_ast(product, "complex_fused_curl_constitutive_D")
    # The constexprs go by keyword; everything without a ``tl.constexpr``
    # annotation is positional, and the two lists must line up exactly.
    positional = [arg.arg for arg in node.args.args
                  if arg.annotation is None
                  or "constexpr" not in ast.unparse(arg.annotation)]
    constexprs = {arg.arg for arg in node.args.args
                  if arg.arg not in positional}
    assert len(seen["args"]) == len(positional), (len(seen["args"]), positional)
    assert set(seen["kwargs"]) - {"enable_fp_fusion", "num_warps"} == constexprs
    assert seen["kwargs"]["SCALE"] == 1
    assert seen["kwargs"]["BACKWARD"] == 1
    # The three inv_eps pointers sit between the constitutive aux and the
    # coefficients, exactly where the signature puts e0/e1/e2.
    first_eps = positional.index("e0")
    for offset, name in enumerate(("Ex", "Ey", "Ez")):
        bound = seen["args"][first_eps + offset]
        assert bound.array is arrays[f"inv_eps_{name}"], name
        assert bound.array.dtype == np.float32


# ---------------------------------------------------------------------------
# The deferral, and the absence of Triton
# ---------------------------------------------------------------------------

def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """The merge bar is a laptop, so the predicates must answer without Triton."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(
        importlib.import_module(
            "meep_gpu.triton_kernels.complex_fused_electric_pair"))
    try:
        assert reloaded.complex_fused_curl_constitutive_D is None
        with pytest.raises(ImportError, match="triton"):
            reloaded.complex_fused_curl_constitutive_D_kernel()
        with expansion_refusal.declaring_run_policy("keep"):
            fields, pml = build()
            verdict = reloaded.complex_fused_electric_pair_coverage(
                fields, pml, (), probe=probe_record())
        assert residual(verdict) == []
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(reloaded)


def test_the_composer_routes_this_product_and_dispatch_admits_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch 2026-09-13.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; on 2026-09-13
    ``dispatch_fused_route_2026-09-13_phaseB`` drove it through the driver's own
    consults on ``bloch_2d`` and ``complex_nobloch_2d``, so the label now names
    those cases in ``RELEASED_FUSED_ARMS`` and clause (8) admits the plan.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its gate has a tracked ledger entry, cut from this
    campaign's fleet artifact by ``seed_triton_welds.py``) and out of
    ``PENDING_DEVICE_GATE_ARMS`` — the certified-and-released state, not a refusal.
    The two are asserted as exactly one of the pair, because a label released while
    still pending would be a plan claiming a certification the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_complex_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["complex_fused_electric_pair"]
    assert row["module"] == "complex_fused_electric_pair"
    assert row["builder"] == "plan_complex_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_makes_no_identity_claim_it_has_not_earned():
    """No weld, no byte-identity claim. The words are the tell, so they are what is
    checked: a module that says RELEASED while ``fingerprints.json`` has no entry
    for it is the one failure mode a green suite cannot otherwise catch."""
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    welded = any("complex_fused_electric_pair" in key for key in record)
    if not welded:
        assert "NOT RELEASED" in source, (
            "the module claims a device status with no fingerprints.json entry")
        assert "bit-identical" not in source and "byte-identical" not in source


def test_the_docstrings_ceiling_is_the_boards_own_number():
    """The module quotes 16 of 16 for this cell. Re-derived from the board that
    priced it, so the docstring cannot outlive the measurement."""
    if not BOARD.exists():  # pragma: no cover - the board is cut per round
        pytest.skip(f"{BOARD} has not been cut on this host")
    board = json.loads(BOARD.read_text(encoding="utf-8"))
    rows = [row for row in board["aggregate"]["cells"]
            if row["seam"] == "D->E" and row["curl_arm"] == "complex PML"
            and row["constitutive_arm"] == "complex"]
    assert len(rows) == 1, rows
    assert rows[0]["ceiling_no_in_seam_source"] == 16, rows[0]
    assert rows[0]["rows_driving_it"] == 16, rows[0]
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "ceiling 16" in source
    # ...and the board that priced it recorded NO product at this cell, which is
    # the fact the module's docstring rests on. A board cut after this product is
    # wired into PRODUCTS answers differently, and the docstring cites THIS one.
    assert rows[0]["product"] is None, rows[0]
