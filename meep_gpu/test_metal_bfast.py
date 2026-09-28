"""Merge-bar tests for the Metal BFAST curl family.

Three kinds of thing are pinned here, and the first kind is the one that carries
the most weight because NO GATE CAN CARRY IT.

**1. THE PREDICTED NULLS.** Five properties of the transcription are byte-INVISIBLE
in float32: no seed can distinguish the shipped spelling from its alternative,
because the two compute the same words on every input. A mutation leg for any of
them would report "not caught" and be indistinguishable from a real miss. They are
therefore pinned by SOURCE-TEXT ASSERTION here, and named in the gate's
``summary.predicted_nulls`` with the reason each is unmeasurable, so the record
shows what was measured and what was not.

This matters more on Metal than on Triton. The Triton track can read every
generated PTX instruction and REFUSE A COMPILE on a policy violation;
``torch.mps.compile_shader`` exposes no disassembly at all, so for this family the
source text is the only place these five can be held.

**2. THE DUPLICATE-PLUS-PIN AGREEMENTS.** Two things are deliberately restated
rather than imported — the advance's ownership mask (because ``shaders``'
generalises on the zero literal, not on the masked variable) and the clause list
(because the shipped reason list is built inside a function whose BFAST clause
cannot be subtracted from outside). Each restatement is asserted equal to its
originator over a full cross product, so drift fails a test in either direction
instead of producing a family whose wall is subtly not the certified wall.

**3. THE STRUCTURAL CONTRACTS** that would otherwise be discovered at compile time
or, worse, at run time: the 31-binding ceiling, the sub-step table's agreement with
the Triton port's, and the no-module-level-torch rule that keeps the predicate
readable on a host with no GPU.

The device-touching tests are skipped without MPS. The full case matrix lives in
``parity/meep_gpu/gate_metal_bfast.py``; what is here is the smallest non-vacuous
version, so a red merge bar names the defect without a GPU run.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from meep_gpu.metal_kernels import bfast_curl as bfast
from meep_gpu.metal_kernels import shaders
from meep_gpu.triton_kernels import bfast_curl as triton_bfast
from meep_gpu.triton_kernels.launch import SUB_STEPS as TRITON_SUB_STEPS

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "metal_kernels", "bfast_curl.py")

#: One representative walled configuration; the tail and both masks are live.
CODES = (shaders.METALLIC, shaders.METALLIC, shaders.PERIODIC)


def _source(backward: bool = False, has_bfast: bool = True) -> str:
    return bfast.bfast_curl_source(CODES, backward, has_bfast=has_bfast)


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


# ---------------------------------------------------------------------------
# 1. The predicted nulls — byte-invisible, so pinned by source text
# ---------------------------------------------------------------------------

def test_the_tail_sits_between_the_curl_and_the_mask():
    """PREDICTED NULL. The array path folds BFAST at stepping.py:392-396, AFTER
    the curl (:370) and BEFORE ``_mask_non_owned_cells`` (:397).

    WHY NO SEED CAN CATCH A MOVE. Both orders write an exact ``0.0`` into the same
    cells: the advance is masked by the same predicate the curl is, so masking
    before or after the fold produces identical words on every input. The ORDER is
    still the array path's and is held here rather than left to drift.
    """
    source = _source()
    curl = source.index("float curl0 = dtdx *")
    tail = source.index("float total0 =")
    mask = source.index("curl0 = at_x ? 0.0f : curl0;")
    assert curl < tail < mask, (
        "the BFAST tail must sit between the curl and the ownership mask, which "
        "is where stepping.py:392-396 folds it")


def test_the_sums_put_the_shifted_operand_first():
    """PREDICTED NULL. stepping.py:923-924 writes ``shifted + centre``.

    WHY NO SEED CAN CATCH A SWAP. IEEE addition is commutative IN BITS, including
    for signed zeros, so ``(c_y + c)`` and ``(c + c_y)`` are the same float32 word
    for every input. The operand order is the array path's and is pinned literally.
    """
    source = _source()
    for line in ("float total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b));",
                 "float total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c));",
                 "float total2 = (k1_2 * (b_x + b)) - (k2_2 * (a_y + a));"):
        assert line in source, line


def test_the_curl_fold_is_spelled_as_a_subtraction():
    """PREDICTED NULL. ``curl - adv`` carries stepping.py's ``curl + (-advance)``.

    WHY NO SEED CAN CATCH THE DIFFERENCE. IEEE-754 defines subtraction AS addition
    of the negation, so the two agree on every input including signed zeros. The
    subtraction spelling is used because it introduces NO unary minus — on Triton
    that mattered because unary minus lowers as ``0.0 - x``; on Metal ``-x`` was
    measured to be a sign-bit operation, so the workaround does NOT transfer and
    the spelling is kept for literalness rather than for the Triton reason.
    """
    source = _source()
    for target in range(3):
        assert f"curl{target} = curl{target} - adv{target};" in source
    # No unary minus anywhere in the emitted CODE. Comments are stripped first:
    # the tail's commentary quotes stepping.py's `-advance`, and a text search
    # that matched prose would fail for the wrong reason.
    code = "\n".join(line.split("//")[0] for line in source.splitlines())
    assert "-adv" not in code, (
        "no unary minus belongs in the tail; the fold is a subtraction")


def test_the_d_side_negates_before_rounding():
    """PREDICTED NULL. stepping.py:913-914 negates both k in HOST FLOAT64, and
    :923-924 rounds to float32 once, at use.

    WHY NO SEED CAN CATCH THE ORDER. f64 negation and f32 rounding commute exactly
    — negation is a sign-bit flip at both widths — so binding ``float32(-k)`` and
    ``-float32(k)`` are the same word. The order is held because the transcription
    is literal per call site, not because it is observable.
    """
    b = bfast.bfast_curl_coefficients((0.31, -0.17, 0.23), (False,) * 3,
                                      magnetic=True)
    d = bfast.bfast_curl_coefficients((0.31, -0.17, 0.23), (False,) * 3,
                                      magnetic=False)
    assert all(np.float32(x) == np.float32(-y) for x, y in zip(b, d))
    # And the rounding happens exactly once: the bits survive float().
    for value in b + d:
        assert np.float32(value) == np.float32(np.float32(value))


def test_the_advance_uses_the_two_times_spelling():
    """PREDICTED NULL. ``advance = total - 2.0*state`` is stepping.py:928 literally.

    WHY NO SEED CAN CATCH IT. ``2.0f * x`` and ``x + x`` are exactly equal in
    binary floating point for every finite input — multiplying by a power of two is
    exact — so the literal spelling is unobservable.
    """
    source = _source()
    for target in range(3):
        assert f"float adv{target} = total{target} - (2.0f * st{target});" in source


# ---------------------------------------------------------------------------
# 2. Duplicate-plus-pin agreements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward", (False, True), ids=("step_B", "step_D"))
def test_the_advance_mask_is_the_curl_mask(backward):
    """The advance carries the SAME ownership predicate the summed curl does.

    ``shaders.ownership_mask`` generalises on the ZERO LITERAL (for the complex
    family) and not on the masked VARIABLE, so this family restates the pair table.
    A restatement that masked different cells would leave an unowned cell's
    ``f_bfast`` holding a value MEEP's owned-cell loop never writes — invisible in
    the field, visible only in a diagnostic read of the state.
    """
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                codes = (cx, cy, cz)
                mine = bfast.advance_mask(codes, backward)
                theirs = shaders.ownership_mask(codes, backward)
                if theirs.lstrip().startswith("//"):
                    assert mine.lstrip().startswith("//"), codes
                    continue
                assert mine == theirs.replace("curl", "adv"), codes


def test_the_clause_list_matches_the_triton_bfast_predicate():
    """The Metal and Triton BFAST clause lists agree once each backend clause goes.

    The restatement exists because the shipped reason list is built inside a
    function whose BFAST clause cannot be subtracted from outside. A clause added
    to one list and not the other is exactly the over-covering drift this pin
    exists to catch.

    THE MATRIX CARRIES ITS OWN ABSORBER, and that is a repair rather than a detail.
    Every row used to build an ACTIVE ``_PML()``, so clause 3 — whose Metal wording
    is NOT the Triton module's, deliberately (see :data:`PORT_SPECIFIC_CLAUSES`) —
    was never compared, and the pin's "exact set equality" claim held only because
    it never reached the one clause that diverges. Measured on this tree before the
    repair: adding a ``_PML(active=False)`` row made the two sets disagree.
    """
    from meep_gpu.metal_kernels import coverage as metal_coverage

    compared = 0
    for label, grid_kwargs, fields_kwargs, pml in _MATRIX:
        grid = _Grid(**grid_kwargs)
        fields = _Fields(grid, **fields_kwargs)
        triton_reasons = set(triton_bfast._bfast_grid_reasons(fields, pml, grid))
        metal_reasons = set(bfast._bfast_grid_reasons(fields, pml, grid))
        triton_rest = {r for r in triton_reasons
                       if not (r.startswith("array module is ")
                               and r.endswith(", not cupy"))}
        metal_rest = metal_reasons - set(
            metal_coverage._metal_backend_reasons(grid))
        triton_rest, metal_rest = _subtract_declared(triton_rest, metal_rest)
        assert triton_rest == metal_rest, (
            f"{label}: the two BFAST clause lists disagree.\n"
            f"  only Triton has: {sorted(triton_rest - metal_rest)}\n"
            f"  only Metal has:  {sorted(metal_rest - triton_rest)}")
        compared += len(metal_rest)
    # A set equality between two EMPTY sets holds for any pair of clause lists, so
    # the pin needs a floor: the matrix must have produced clauses to compare.
    assert compared >= len(_MATRIX), (
        f"VACUOUS clause-list pin: {compared} clauses over {len(_MATRIX)} "
        f"configurations, so the two sets agreed largely by being empty")


def test_the_conductivity_clause_matches_the_triton_bfast_predicate():
    """The OTHER restated list, which nothing compared at all until now.

    ``_curl_conductivity_reasons`` is restated on this side exactly as the grid
    clauses are, and its wording ALSO diverges from the Triton module's — measured,
    on the half-sentence naming which package ships a conductive product. Nothing
    compared it, so the divergence was invisible; it is now declared and the rest
    is exact set equality, per target and for a missing reader.
    """
    conductive_cases = [("one target", ("Bx",)), ("two targets", ("Bx", "Dz")),
                        ("all six", tuple(bfast.CURL_TARGETS))]
    compared = 0
    for label, live in conductive_cases:
        fields = _Fields(_Grid())
        fields.condfac_for = (lambda target, _live=live:
                              np.zeros(3, np.float32) if target in _live else None)
        triton_rest, metal_rest = _subtract_declared(
            set(triton_bfast._curl_conductivity_reasons(fields)),
            set(bfast._curl_conductivity_reasons(fields)))
        assert triton_rest == metal_rest, (
            f"{label}: the two conductivity clause lists disagree.\n"
            f"  only Triton has: {sorted(triton_rest - metal_rest)}\n"
            f"  only Metal has:  {sorted(metal_rest - triton_rest)}")
        compared += len(live)
    # The unreadable-reader clause, which must be identical on both sides.
    blind = _Fields(_Grid())
    blind.condfac_for = None
    assert (set(triton_bfast._curl_conductivity_reasons(blind))
            == set(bfast._curl_conductivity_reasons(blind)))
    assert compared == 1 + 2 + 6, compared


def test_the_sub_step_table_is_the_triton_one():
    """This module BINDS the Triton table — object identity, not a value compare.

    Comparing ``bfast.SUB_STEPS`` field by field against ``TRITON_SUB_STEPS`` is
    ``x == x``: the module imports the Triton object, so no edit anywhere can make
    that compare fail and it measures nothing. The claim worth holding is the
    IDENTITY, which is one line, plus the compare that CAN fail — see the test
    below.
    """
    assert bfast.SUB_STEPS is TRITON_SUB_STEPS, (
        "this family must bind the Triton sub-step table itself, not a copy of it")


def test_the_metal_composer_table_matches_the_one_this_family_binds():
    """THE COMPARE THAT CAN ACTUALLY FAIL, and the drift it fences.

    ``metal_kernels.launch.SUB_STEPS`` is a SEPARATE literal — the table the shipped
    Metal composer, ``plan_from_arrays`` and the certified curl plan bind — while
    this family binds ``triton_kernels.launch.SUB_STEPS``. Two literals, two ports,
    one meaning; a ``suffix`` that drifted on one side would bind the curl's PML
    coefficients from the other Yee sub-lattice, which is a converged, smooth,
    half-cell-wrong absorber rather than a crash, and the gate's identity leg
    compares exactly across that seam.
    """
    from meep_gpu.metal_kernels.launch import SUB_STEPS as METAL_SUB_STEPS

    assert METAL_SUB_STEPS is not TRITON_SUB_STEPS, (
        "the two tables are the same object, so this test would be vacuous; if the "
        "composer has been re-pointed at the Triton table, delete this test rather "
        "than leaving a compare that cannot fail")
    assert set(METAL_SUB_STEPS) == set(TRITON_SUB_STEPS)
    for name, spec in METAL_SUB_STEPS.items():
        other = TRITON_SUB_STEPS[name]
        assert tuple(spec["targets"]) == tuple(other["targets"]), name
        assert tuple(spec["sources"]) == tuple(other["sources"]), name
        assert bool(spec["backward"]) == bool(other["backward"]), name
        assert spec["suffix"] == other["suffix"], name


def test_the_k_assignment_is_cross_gated_the_way_stepping_writes_it():
    """The k-indexing trap, pinned against ``stepping``'s own call site.

    ``k1`` is indexed by the SECOND partner's own direction but multiplies the
    FIRST's sum, and the flags are CROSS-gated: ``have_m`` gates ``k1``. Getting
    this backwards is silent — on a k along one axis it merely moves the term to
    the wrong pair of components — which is why stepping.py:818-821 calls it the
    single easiest mistake in the whole pass.
    """
    # Bx: first = Ez (own axis z), second = Ey (own axis y).
    # k1 = bfast[own(second)] = k_y ; k2 = bfast[own(first)] = k_z.
    ks = bfast.bfast_curl_coefficients((0.0, 5.0, 7.0), (False,) * 3, magnetic=True)
    assert ks[0] == pytest.approx(5.0), "k1 on target 0 must be k_y"
    assert ks[1] == pytest.approx(7.0), "k2 on target 0 must be k_z"

    # have_m gates k1: making the SECOND partner's derivative axis invariant
    # (z for Bx) must zero k1, not k2.
    invariant = [False, False, True]
    guarded = bfast.bfast_curl_coefficients((0.0, 5.0, 7.0), invariant,
                                            magnetic=True)
    assert guarded[0] == 0.0, "an invariant second-partner axis must zero k1"
    assert guarded[1] == pytest.approx(7.0), "k2 must survive: have_p gates it"


@pytest.mark.parametrize("bfast_k", ((0.31, 0.17, 0.23), (-0.42, 0.19, -0.28),
                                     (0.816958, 0.0, 0.0)))
@pytest.mark.parametrize("invariant", ((False, False, False), (False, False, True),
                                       (True, False, False)))
@pytest.mark.parametrize("magnetic", (True, False), ids=("step_B", "step_D"))
def test_the_import_route_answers_what_steppings_own_call_site_answers(
        bfast_k, invariant, magnetic):
    """The imported coefficients, against ``stepping._bfast_term``'s OWN call site.

    The module argues that importing ``bfast_curl_coefficients`` beats a third
    transcription of the cross-assigned k indices. That argument is only worth
    anything if the imported answer is pinned against the ORACLE rather than against
    a hand-derived expectation — the previous test derives ``k1 = k_y`` for Bx by
    hand, which pins the same reading twice. Here the expectation is rebuilt from
    ``stepping``'s own tables and its own ``_bfast_axis``, in ``stepping``'s own
    order, including the D side's host-float64 negation and the single float32
    rounding at use.
    """
    from meep_gpu import stepping

    terms = stepping.B_CURL_TERMS if magnetic else stepping.D_CURL_TERMS
    expected = []
    for term in terms:
        have_p = not invariant[term.first_axis]          # stepping.py:909
        have_m = not invariant[term.second_axis]         # stepping.py:910
        k1 = bfast_k[stepping._bfast_axis(term.second)] if have_m else 0.0   # :911
        k2 = bfast_k[stepping._bfast_axis(term.first)] if have_p else 0.0    # :885
        if not magnetic:
            k1, k2 = -k1, -k2                            # :886-887, host float64
        expected.extend([float(np.float32(k1)), float(np.float32(k2))])  # :896-897

    got = bfast.bfast_curl_coefficients(bfast_k, invariant, magnetic=magnetic)
    assert [np.float32(v).tobytes() for v in got] == \
           [np.float32(v).tobytes() for v in expected], (bfast_k, invariant, got,
                                                         expected)
    # Non-vacuity: a case whose six scalars are all zero would pass whatever the
    # indexing did.
    assert any(v != 0.0 for v in got), (bfast_k, invariant)


# ---------------------------------------------------------------------------
# 3. Structural contracts
# ---------------------------------------------------------------------------

def test_the_binding_count_stays_under_the_metal_ceiling():
    """29 bindings, and the Metal argument-table ceiling is 31.

    Asserted rather than left for a later edit to discover as a compile error on
    someone else's machine. A complex build of this family would double the volume
    bindings and overrun, which is why the complex composition is refused by name.
    """
    for backward in (False, True):
        for has_bfast in (True, False):
            source = bfast.bfast_curl_source(CODES, backward,
                                             has_bfast=has_bfast)
            count = source.count("[[buffer(")
            assert count == 29, (backward, has_bfast, count)
            assert count <= 31


def test_the_has_bfast_off_build_carries_no_tail():
    off = _source(has_bfast=False)
    for token in ("total0", "adv0", "st0"):
        assert token not in off, token
    # ...and still binds the states, so the identity leg proves they are inert.
    assert "device float*       s0" in off


def test_every_shipped_source_carries_the_contraction_guard():
    """The tail's ``k1*(sum) - k2*(sum)`` is a multiply-subtract and would fuse."""
    guard = shaders.contraction_pragma(shaders.CONTRACT_OFF)
    for label, source in bfast.enumerate_bfast_sources().items():
        assert guard in source, label


def test_the_fast_variant_really_differs():
    """A guard selector that selected nothing would make the gate's leg vacuous."""
    off = bfast.bfast_curl_source(CODES, False, shaders.CONTRACT_OFF)
    fast = bfast.bfast_curl_source(CODES, False, shaders.CONTRACT_FAST)
    assert off != fast


def test_no_module_level_torch_import():
    """The predicate must be readable on a host with no Mac GPU."""
    with open(MODULE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "torch" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")


def test_the_family_registers_four_wired_arms():
    """WIRED as of tranche 2: two curls and the certified constitutive pair.

    The constitutive arms carry NO NEW KERNEL. ``update_H`` and ``update_E`` read
    nothing BFAST-dependent — the second additive term and its IIR state live in
    ``step_db`` alone — so the pair is the certified constitutive plan under a
    restated predicate, which is why the arm count is four rather than two and why
    a BFAST run composes on all four slots instead of syncing twice per step.
    """
    from meep_gpu.metal_kernels import arms

    mine = [spec for spec in arms.registered() if spec.family == bfast.FAMILY]
    assert len(mine) == 4, mine
    assert all(spec.wired for spec in mine)
    assert {spec.slot for spec in mine} == {"step_B", "step_D",
                                            "update_H", "update_E"}


def test_the_constitutive_companion_delegates_to_the_certified_plan_class(
        monkeypatch):
    """"Same kernel" is checked as an identity, not asserted in a comment.

    The BFAST constitutive arm's whole claim is that it changes the ADMISSION and
    nothing else. If it ever built its own plan class, that claim would be false and
    the family's byte gate would not cover the sub-step it composes.

    The flush policy is set through ``monkeypatch`` for the reason the byte test
    below records at length: this arm64 host's default resolves to ``keep``, which
    the MPS executor cannot deliver, so the predicate refuses without it — and
    setting it on the raw environment leaks into every later test in the process.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    import numpy as _np

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.metal_kernels.device import Residency
    from meep_gpu.metal_kernels.launch import ConstitutivePlan
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), courant=0.35,
                bfast_scaled_k=(0.2, 0.0, 0.0), xp=_np)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = tuple(grid.shape)
    epsilon = _np.full(shape, _np.float32(2.0))
    fields.set_isotropic_epsilon_volume(epsilon,
                                        (_np.float32(1.0) / epsilon).astype(_np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    for side in ("H", "E"):
        assert bfast.bfast_run_constitutive_coverage(fields, pml, side,
                                                     Residency()).covered
        plan = bfast.plan_bfast_run_constitutive(fields, pml, side, Residency())
        assert type(plan) is ConstitutivePlan, type(plan)


def test_the_curl_arms_do_not_claim_a_constitutive_kernel_of_their_own():
    """The four arms come from two products, and the split is by slot."""
    from meep_gpu.metal_kernels import arms

    curls = [s for slot in ("step_B", "step_D")
             for s in arms.registered(slot) if s.family == bfast.FAMILY]
    constitutive = [s for slot in ("update_H", "update_E")
                    for s in arms.registered(slot) if s.family == bfast.FAMILY]
    assert len(curls) == 2 and len(constitutive) == 2
    assert {s.noun for s in curls} == {"BFAST PML curl"}
    assert {s.noun for s in constitutive} == {"BFAST constitutive"}
    # The module reaches `arms` from INSIDE `register_arms`, never at module scope.
    # `arms` imports the fail-closed machinery from `triton_kernels.launch`, and a
    # module-scope edge here would put that import on the path of anyone who merely
    # reads a BFAST source string.
    with open(MODULE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    top_level = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert not any((n.module or "").endswith("arms") or
                   any(alias.name == "arms" for alias in n.names)
                   for n in top_level), "arms is imported at module scope"


@requires_mps
def test_a_mirror_width_no_kernel_can_bind_is_refused_by_name():
    """The residency layer must refuse an element width that SILENTLY REINTERPRETS.

    Lives beside this family because this family's plan builders are what register
    mirrors, and because the defect was found by auditing the tranche's own claim
    that complex128 is "refused by name". It was not: measured on this host, a
    complex128 host array produced a ``torch.complex128`` tensor on ``mps:0``, bound
    to a ``device float2*`` signature, and the LAUNCH SUCCEEDED — one 128-bit cell
    read as two 64-bit ``float2`` cells, garbage written back, nothing raised. That
    is the silent-wrong-answer class every predicate in this package exists to
    refuse, and it belongs at registration rather than at the first wrong number.

    float64 is refused too. It happened to fail already inside ``Tensor.to`` because
    the MPS framework has no float64 — but that is the platform refusing an
    unrelated thing, and a claim that rests on which error the platform happens to
    raise first is not a claim.
    """
    from meep_gpu.metal_kernels import device

    residency = device.Residency()
    # The two widths that DO bind still bind: the refusal must not be a blanket one.
    residency.mirror("real", np.zeros(8, np.float32))
    residency.mirror("complex", np.zeros(8, np.complex64), dtype=np.complex64)
    assert set(residency.names) == {"real", "complex"}

    for dtype in (np.complex128, np.float64, np.int32):
        with pytest.raises(ValueError, match="can bind"):
            residency.mirror(f"bad_{np.dtype(dtype).name}", np.zeros(8, dtype),
                             dtype=dtype)
    # A refused registration must leave no half-built mirror behind.
    assert set(residency.names) == {"real", "complex"}


def test_an_unknown_boundary_code_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        bfast.bfast_curl_source((2, 0, 0), False)
    with pytest.raises(ValueError):
        bfast.bfast_curl_source((0, 0), False)


# ---------------------------------------------------------------------------
# Coverage refusals on constructed objects
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **kwargs):
        self.xp = np
        self.shape = (4, 3, 2)
        self.dt = 0.05
        self.dx = 0.1
        self.k_point = (0.0, 0.0, 0.0)
        self.has_bloch = False
        self.cylindrical = False
        self.bfast_active = True
        self.bfast_scaled_k = (0.3, 0.0, 0.0)
        self.beta = 0.0
        self._symmetry = False
        self._mirrored = ()
        self._axis = ()
        self._invariant = ()
        self.__dict__.update(kwargs)

    def has_symmetry(self):
        return self._symmetry

    def is_mirrored(self, axis):
        return axis in self._mirrored

    def is_axis(self, axis):
        return axis in self._axis

    def is_invariant(self, axis):
        return axis in self._invariant


class _PML:
    def __init__(self, active=True):
        self.is_active = active


class _Fields:
    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.force_complex_fields = False
        self.has_nonlinearity = False
        self.stores_E = True
        self.polarizations = ()
        self.has_polarizations = False
        self.has_offdiagonal_epsilon = False
        self.__dict__.update(kwargs)


#: Each row carries its OWN absorber. Every row used to share one active ``_PML()``,
#: which is what hid the clause-3 divergence: an inactive absorber is the only way to
#: reach that clause, and the pin never built one.
_MATRIX = (
    ("plain", {}, {}, _PML()),
    ("complex_storage", {}, {"force_complex_fields": True}, _PML()),
    ("nonlinear", {}, {"has_nonlinearity": True}, _PML()),
    ("bloch_flag", {"has_bloch": True}, {}, _PML()),
    ("bloch_kpoint", {"k_point": (0.3, 0.0, 0.0)}, {}, _PML()),
    ("cylindrical", {"cylindrical": True}, {}, _PML()),
    ("symmetry", {"_symmetry": True}, {}, _PML()),
    ("mirrored_axis", {"_mirrored": (1,)}, {}, _PML()),
    ("beta", {"beta": 0.7}, {}, _PML()),
    ("bfast_off", {"bfast_active": False}, {}, _PML()),
    ("invariant_axis_k", {"bfast_scaled_k": (0.4, 0.0, 0.3),
                          "_invariant": (2,)}, {}, _PML()),
    ("axis_r0", {"_axis": (0,)}, {}, _PML()),
    ("stores_E_off", {}, {"stores_E": False}, _PML()),
    ("pml_inactive", {}, {}, _PML(active=False)),
    ("pml_none", {}, {}, None),
)

#: THE DECLARED PORT-SPECIFIC DIVERGENCES, as ``(triton_string, metal_string)``
#: pairs. Everything else is compared as EXACT SET EQUALITY, which is this package's
#: doctrine; a normalised or fuzzy key comparison would let a real clause edit
#: through. Each pair is asserted OBSERVED on its own side over the matrix by
#: :func:`test_every_declared_divergence_is_actually_reached`, so a pair left behind
#: after a wording change fails a test instead of silently widening the exemption.
PORT_SPECIFIC_CLAUSES = (
    # Clause 3. The Triton tail asserts `no_pml.py refuses bfast too` — a statement
    # about the TRITON package that is MEASURABLY FALSE of this one:
    # `no_pml_constitutive.metal_null_constitutive_coverage` admits a BFAST run
    # under an inactive absorber (a different slot, and correct there).
    ("no active PML layer (this product implements the split-field path only; "
     "no-PML BFAST is refused by name — zero demand, and no_pml.py refuses bfast "
     "too)",
     "no active PML layer (this product implements the split-field path only; "
     "no-PML BFAST is refused by name)"),
)

#: Per-target conductivity clauses diverge the same way and for the same kind of
#: reason: which package ships a conductive product. Spelled as a template because
#: the clause names the target.
PORT_SPECIFIC_CONDUCTIVITY = tuple(
    (f"a conductivity is installed on {target}: this kernel transcribes the plain "
     f"split-field recurrence only, and the conductive family's own predicate "
     f"refuses bfast (conductivity.py:515-518) — the composition is a named "
     f"follow-up, not a silent overlap",
     f"a conductivity is installed on {target}: this kernel transcribes the plain "
     f"split-field recurrence only, this package ships no conductive product, and "
     f"the Triton conductive family's own predicate refuses bfast in return "
     f"(triton_kernels/conductivity.py:515-518) — a named follow-up, not a silent "
     f"overlap")
    for target in bfast.CURL_TARGETS)

_DECLARED = PORT_SPECIFIC_CLAUSES + PORT_SPECIFIC_CONDUCTIVITY

#: Every declared string that was actually seen, filled in as the pins run.
_OBSERVED: dict = {}


def _subtract_declared(triton_rest: set, metal_rest: set):
    """Remove the declared divergences from each side, recording what was seen."""
    for left, right in _DECLARED:
        if left in triton_rest and right in metal_rest:
            _OBSERVED[left] = _OBSERVED.get(left, 0) + 1
            triton_rest = triton_rest - {left}
            metal_rest = metal_rest - {right}
    return triton_rest, metal_rest


def test_every_declared_divergence_is_actually_reached():
    """A declared exemption nobody reaches is a permanently widened pin.

    Runs the two clause pins first (pytest gives no ordering guarantee, so this
    calls them rather than relying on having run after them), then requires every
    declared pair to have been observed on BOTH sides at least once.
    """
    _OBSERVED.clear()
    test_the_clause_list_matches_the_triton_bfast_predicate()
    test_the_conductivity_clause_matches_the_triton_bfast_predicate()
    unreached = [left for left, _ in _DECLARED if left not in _OBSERVED]
    assert not unreached, (
        f"{len(unreached)} declared port-specific clause(s) were never reached by "
        f"the matrix, so the exemption widens the pin without being exercised: "
        f"{unreached}")
    assert len(_OBSERVED) == len(_DECLARED)


def test_clause_11a_refuses_k_on_a_declared_invariant_axis():
    """THE OVER-COVERAGE DEFECT, refused by name.

    MEEP nulls the partner OPERAND as well as zeroing the coefficient, so its
    increment on a guarded target is ``F_new = -F_prev`` with the other term
    dropped; stepping.py zeroes only the coefficient and keeps the other product.
    They agree only while the surviving k is itself zero. The gate re-measures the
    divergence on the device; this pins that the predicate refuses it at all.
    """
    grid = _Grid(bfast_scaled_k=(0.4, 0.0, 0.3), _invariant=(2,))
    reasons = bfast._bfast_grid_reasons(_Fields(grid), _PML(), grid)
    assert any("DECLARED-invariant axis 2" in r for r in reasons), reasons


def test_clause_11a_admits_k_that_avoids_the_invariant_axis():
    """The refusal must be SPECIFIC, or it sends correct runs to the array path."""
    grid = _Grid(bfast_scaled_k=(0.4, 0.0, 0.0), _invariant=(2,))
    reasons = bfast._bfast_grid_reasons(_Fields(grid), _PML(), grid)
    assert not any("DECLARED-invariant" in r for r in reasons), reasons


def test_an_unanswerable_invariance_question_is_refused_not_admitted():
    """Inferring admission from an absent reader is admission by attribute absence."""
    class _NoReader(_Grid):
        is_invariant = None

    grid = _NoReader()
    reasons = bfast._bfast_grid_reasons(_Fields(grid), _PML(), grid)
    assert any("cannot be answered" in r for r in reasons), reasons


def test_a_non_bfast_run_belongs_to_the_certified_plain_kernels():
    grid = _Grid(bfast_active=False)
    reasons = bfast._bfast_grid_reasons(_Fields(grid), _PML(), grid)
    assert any("bfast_active is False" in r for r in reasons), reasons


def test_the_sub_step_predicate_requires_the_iir_states():
    """A missing state RAISES in the array path; here it is a refusal by name."""
    grid = _Grid()
    fields = _Fields(grid)
    verdict = bfast.bfast_pml_curl_coverage(fields, _PML(), "step_B")
    assert not verdict.covered
    assert any("f_bfast_Bx is not allocated" in r for r in verdict.reasons), \
        verdict.reasons


def test_an_unknown_sub_step_raises_rather_than_refusing_quietly():
    with pytest.raises(ValueError):
        bfast.bfast_pml_curl_coverage(_Fields(_Grid()), _PML(), "update_H")


# ---------------------------------------------------------------------------
# Device: the smallest non-vacuous byte comparison
# ---------------------------------------------------------------------------

@requires_mps
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_kernel_is_byte_identical_to_stepping(sub_step, monkeypatch):
    """The whole claim, in its smallest form: one launch, one seed, uint32 words.

    THE POLICY IS SET THROUGH ``monkeypatch``, NOT ``os.environ`` DIRECTLY, and the
    difference is not cosmetic. The MPS executor delivers ``flush`` natively and
    cannot deliver ``keep``, and this arm64 host's DEFAULT resolves to keep — so
    this test needs an explicit flush to be admitted at all. Setting it on the raw
    environment leaks into every later test in the same process: measured here, it
    turned ``test_subnormal_policy.py`` red purely by running after this file.
    ``monkeypatch`` scopes the change to this test and restores it afterwards.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    from meep_gpu import stepping
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.metal_kernels import device
    from meep_gpu.pml import PML

    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
             "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
             "fu_Dx", "fu_Dy", "fu_Dz",
             "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
             "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9),
                boundaries=("metallic", "metallic", "periodic"), dimensions=3,
                courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np,
                bfast_scaled_k=(0.31, 0.17, 0.23))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    rng = np.random.default_rng(20260815)
    for name in names:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)

    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    getattr(stepping, sub_step)(fields, pml)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}

    for name, value in before.items():
        getattr(fields, name)[...] = value
    residency = device.Residency()
    plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step, residency)
    assert plan is not None, bfast.bfast_pml_curl_coverage(
        fields, pml, sub_step, residency).reasons
    plan.run()
    residency.sync_out()

    spec = bfast.SUB_STEPS[sub_step]
    compared = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
                + bfast.BFAST_STATE_NAMES[sub_step])

    def words(array):
        return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)

    moved = sum(int(np.count_nonzero(words(after[n]) != words(before[n])))
                for n in compared)
    assert moved > 0, "VACUOUS: the oracle step moved no state"
    for name in compared:
        differing = int(np.count_nonzero(
            words(getattr(fields, name)) != words(after[name])))
        assert differing == 0, f"{name}: {differing} differing words"
