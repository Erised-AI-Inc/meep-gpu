"""Laptop contracts for the CONDUCTIVE PML fused ELECTRIC D/E pair on Triton.

Everything here runs on the NumPy merge-bar machine. Three checks are load-bearing:

* the TRANSCRIPTION leg -- this product claims to be
  ``conductivity.conductive_pml_curl_step``'s head, three calls to
  ``conductivity._conductive_component`` with ONE statement moved, and
  ``kernels.fused_curl_constitutive_D``'s wall clear and constitutive tail. That
  claim is checked here as FOUR exact statement-list equalities against the shipped
  sources rather than by reading the docstring;
* the MOVED STATEMENT -- ``_conductive_component_registers`` must be
  ``_conductive_component`` minus exactly ``tl.store(f_ptr + idx, v, mask=live)``
  and plus exactly ``return v``. Nothing else may differ, and the two masked stores
  must keep their masks;
* the DEPOSIT FLAG -- the single corpus row of this cell declares an ELECTRIC
  source, so :data:`~.conductive_fused_electric_pair.CARRIES_DEPOSIT_REPAIR` at
  False takes the product from one seam-instance to zero. That is asserted DIRECTLY,
  by flipping the flag and re-asking the predicate, rather than restated in prose.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_conductive_fused_electric_pair.py``); nothing here
launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from typing import List

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.conductive_fused_electric_pair"
MODULE_PATH = PACKAGE_DIR / "conductive_fused_electric_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_conductive_fused_electric_pair.py"

#: 2d_cond_pml / TestAdjointSolver.test_damping's own sigma scale. Any nonzero value
#: exercises the same four branches; this one is the corpus's.
SIGMA_CORPUS = 0.4

#: The ONE statement ``_conductive_component_registers`` may drop, and the ONE it may
#: add. Written out so a second deletion cannot hide behind a set difference.
MOVED_STORE = "tl.store(f_ptr + idx, v, mask=live)"
ADDED_RETURN = "return v"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes.

    ``is_integrated`` is the second half of what this seam asks, and it defaults to
    MEEP's own default (False). On a conductive run that is the case the predicate
    refuses: the driver rescales the WHOLE volume rather than the deposit.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = False

    def __init__(self, index=(1, 1, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _IntegratedElectric(_Electric):
    """The electric source a conductive run CAN carry -- driver.py:3355-3356."""

    is_integrated = True


class _ElectricWithoutIndex:
    field_type = "D"


def _statements(text: str) -> List[str]:
    """Executable lines: comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _node(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    assert node is not None, f"{name} is not defined in {path}"
    return node


def _shipped_body(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed.

    Read from the FILE rather than imported: the merge bar has no Triton, so
    importing ``kernels.py`` raises and this check would only ever bite on a device
    run.

    The text is EXACT -- never ``ast.unparse``d. What is being compared includes the
    PARENTHESISATION, and unparsing re-derives minimal parentheses, which would
    silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different float32
    grouping.
    """
    text = path.read_text(encoding="utf-8")
    node = _node(path, name)
    body = node.body
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    segments = [ast.get_source_segment(text, statement) for statement in body]
    assert all(segment is not None for segment in segments), name
    return _statements("\n".join(segments))


@pytest.fixture(scope="module")
def bodies():
    return {
        "fused": _shipped_body(MODULE_PATH,
                               "conductive_fused_curl_constitutive_D"),
        "helper": _shipped_body(MODULE_PATH, "_conductive_component_registers"),
        "shipped_helper": _shipped_body(PACKAGE_DIR / "conductivity.py",
                                        "_conductive_component"),
        "conductive_curl": _shipped_body(PACKAGE_DIR / "conductivity.py",
                                         "conductive_pml_curl_step"),
        "ordinary_fused": _shipped_body(PACKAGE_DIR / "kernels.py",
                                        "fused_curl_constitutive_D"),
        "ordinary_curl": _shipped_body(PACKAGE_DIR / "kernels.py",
                                       "pml_curl_step"),
    }


def _head(statements: List[str]) -> List[str]:
    index = next(n for n, s in enumerate(statements) if s.startswith("si_z = tl.load"))
    return statements[:index + 1]


def _tail(statements: List[str]) -> List[str]:
    index = next(n for n, s in enumerate(statements) if s.startswith("if ZM_X"))
    return statements[index:]


def _middle(statements: List[str]) -> List[str]:
    low = next(n for n, s in enumerate(statements) if s.startswith("si_z = tl.load"))
    high = next(n for n, s in enumerate(statements) if s.startswith("if ZM_X"))
    return statements[low + 1:high]


def _build(cell_size=(1.2, 1.0, 0.0), dimensions=2, boundaries=("metallic",
                                                                "metallic",
                                                                "periodic"),
           pml_thickness=2, complex_storage=False, sigma=SIGMA_CORPUS,
           components=None, seed=17, thickness=None, **grid_kwargs):
    """A conductive Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family -- 2-D Cartesian, real storage, an active PML,
    metallic in x and y, no fold, a D-side conductivity on every component. Every
    refusal test perturbs exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    if sigma is not None:
        volume = numpy.full(grid.shape, sigma, dtype=numpy.float32)
        payload = ({name: volume for name in components} if components is not None
                   else volume)
        fields.set_d_conductivity(payload)
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.conductive_fused_electric_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION - the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_the_head_is_the_shipped_conductive_curls_head(bodies):
    """Everything down to the six coefficient loads is the certified curl's."""
    assert _head(bodies["fused"]) == _head(bodies["conductive_curl"])


def test_the_shipped_conductive_curls_head_is_the_plain_curls(bodies):
    """The chain the docstring claims, checked rather than asserted.

    ``conductive_pml_curl_step``'s head is ``pml_curl_step``'s because conductivity
    enters ``_apply_curl`` only after the curl is formed. If that stops being true
    upstream, this product's head stops being a transcription of BOTH and the reader
    of the docstring should learn it here.
    """
    plain = _head(bodies["ordinary_curl"])
    assert _head(bodies["conductive_curl"])[:len(plain)] == plain


def test_the_tail_is_the_shipped_fused_pairs_tail(bodies):
    """The wall clear, the D store and the constitutive half are the fused pair's.

    Minus its three ``fu_D`` stores, which did not disappear: they moved INTO
    ``_conductive_component_registers``, where the shipped helper already performs
    them under the masks the four-case table demands. The next test pins that.
    """
    u_stores = {f"tl.store(u{n} + idx, n{n}, mask=live)" for n in (0, 1, 2)}
    expected = [s for s in _tail(bodies["ordinary_fused"]) if s not in u_stores]
    assert _tail(bodies["fused"]) == expected


def _calls(path: pathlib.Path, function: str, callee: str):
    """Every call to ``callee`` inside ``function``, as argument-name lists.

    Read through the AST rather than off the text, because a call wrapped across
    two source lines is one statement and a line-based reader would see two.
    """
    found = []
    for node in ast.walk(_node(path, function)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == callee:
            found.append([ast.unparse(a) for a in node.args])
    return found


def test_the_middle_is_exactly_three_calls_with_the_shipped_coefficient_pairs():
    """The three recurrences, and the (km, si) pairs are the certified curl's.

    A rotated pair here is a wrong absorber profile that no shape check would see,
    so the pairs are read out of ``conductive_pml_curl_step``'s own three calls
    rather than restated.
    """
    mine = _calls(MODULE_PATH, "conductive_fused_curl_constitutive_D",
                  "_conductive_component_registers")
    theirs = _calls(PACKAGE_DIR / "conductivity.py", "conductive_pml_curl_step",
                    "_conductive_component")
    assert len(mine) == len(theirs) == 3

    # The four coefficient names sit between `curl` and the COND constexpr in both
    # signatures, so the same slice reads both.
    assert [c[-5:-1] for c in mine] == [c[-5:-1] for c in theirs] == [
        ["km_y", "si_y", "km_z", "si_z"],
        ["km_z", "si_z", "km_x", "si_x"],
        ["km_x", "si_x", "km_y", "si_y"]]
    # And the CURL each call consumes is its own component's, not a neighbour's.
    assert [c[-6] for c in mine] == ["curl0", "curl1", "curl2"]
    # The five pointer slots are the component's own row of every volume.
    assert [c[:5] for c in mine] == [["f0", "u0", "c0", "cf0", "ci0"],
                                     ["f1", "u1", "c1", "cf1", "ci1"],
                                     ["f2", "u2", "c2", "cf2", "ci2"]]


def test_the_helper_is_the_shipped_one_with_exactly_one_statement_moved(bodies):
    """``_conductive_component`` minus one store, plus one return. Nothing else."""
    mine = [s for s in bodies["helper"] if s != ADDED_RETURN]
    theirs = [s for s in bodies["shipped_helper"] if s != MOVED_STORE]
    assert mine == theirs
    assert bodies["helper"].count(ADDED_RETURN) == 1
    assert MOVED_STORE not in bodies["helper"]
    assert MOVED_STORE in bodies["shipped_helper"]


def test_the_helper_keeps_both_masked_stores_and_their_masks(bodies):
    """``live & dsigu`` and ``live & dsig`` are the case table's "unchanged" column.

    Widening either one writes a component the array path leaves alone, which is a
    silent divergence in a volume no E comparison reads on the step it happens.
    """
    assert "tl.store(u_ptr + idx, u_new, mask=live & dsigu)" in bodies["helper"]
    assert "tl.store(c_ptr + idx, c_new, mask=live & dsig)" in bodies["helper"]
    assert "tl.store(u_ptr + idx, n, mask=live)" in bodies["helper"]


def test_every_statement_comes_from_a_shipped_body(bodies):
    """No line in this kernel is this module's own invention."""
    shipped = set(bodies["conductive_curl"]) | set(bodies["ordinary_fused"])
    mine = set(bodies["fused"])
    invented = {s for s in mine - shipped
                if "_conductive_component_registers" not in s}
    assert not invented, sorted(invented)


def test_the_weld_is_not_the_complex_conductive_one():
    """The no-PML conductive weld is a DIFFERENT recurrence and may not be read for it.

    ``complex_conductive_fused_pair`` carries ``_apply_conductive_update`` -- case D
    alone, no ``f_cond``, no ``fu``. Sharing a body between the two would be the
    over-covering failure both predicates exist to prevent.
    """
    other = _shipped_body(PACKAGE_DIR / "complex_conductive_fused_pair.py",
                          "complex_conductive_fused_curl_constitutive_D")
    mine = _shipped_body(MODULE_PATH, "conductive_fused_curl_constitutive_D")
    assert mine != other
    assert not any("f_cond" in s or "c_new" in s for s in other)


# ---------------------------------------------------------------------------
# The optional dependency, and the declared constants
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    """A Triton-free host must still import the package."""
    source = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "conductive_fused_electric_pair" not in source


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(
        product):
    fields, pml = _build()
    assert product.conductive_fused_electric_pair_coverage(
        fields, pml, ()) is not None
    if product.conductive_fused_curl_constitutive_D is None:
        with pytest.raises(ImportError, match="triton"):
            product.conductive_fused_curl_constitutive_D_kernel()


def test_backward_matches_the_sub_step_table(product):
    """``BACKWARD`` is restated in this module and must equal the table's."""
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 1
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"
    assert product.CONSTITUTIVE_SIDE in CONSTITUTIVE_SIDES


def test_replaces_is_the_drivers_own_call_order(product):
    """The five call sites, in the order driver.step makes them."""
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in product.REPLACES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), product.REPLACES
    assert product.REPLACES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")


# ---------------------------------------------------------------------------
# Coverage - what it admits and what it refuses BY NAME
# ---------------------------------------------------------------------------

def test_an_integrated_electric_deposit_is_admitted(product):
    """Conductive, walled, and an electric source in the seam that IS carried.

    ``is_integrated`` is what separates it from the corpus row: driver.py:3355-3356
    injects an integrated source point-wise, where :3363-3370 rescales the whole
    volume for a scaled one.
    """
    fields, pml = _build()
    assert _reasons(product, fields, pml,
                    (_IntegratedElectric(), _Magnetic())) == []


def test_the_corpus_rows_own_source_is_ADMITTED_after_the_lift(product):
    """The 2026-09-01 lift, asserted in the direction that serves the cell's row.

    ``tests:TestAdjointSolver.test_damping`` declares a NON-integrated electric
    source that publishes its deposit table. The driver now replays its condinv
    rescale sparsely at exactly those cells, the clause that refused it was lifted
    on the re-measured signed-zero walk (the gate's ``lifted_refusal`` leg), and
    the predicate carries the source.
    """
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(), _Magnetic())) == []


def test_a_scaled_source_publishing_no_deposit_table_is_still_refused(product):
    """What survives the lift: the driver's dense whole-volume fallback, by name.

    A scaled source with no ``_point_ix`` attribute cannot name its deposit, so
    the driver must still rescale the whole component — the retired composition,
    whose signed-zero canonicalisation no point repair can reconstruct. No
    in-tree source class is one; the clause fails CLOSED on the stranger.
    """
    fields, pml = _build()
    refused = _reasons(product, fields, pml, (_ElectricWithoutIndex(),))
    assert any("publishes NO deposit "
               "table" in r for r in refused), refused


def test_the_drivers_OWN_partition_is_the_one_this_clause_mirrors():
    """THE TRIPWIRE UNDER THE LIFT, and it is a measurement rather than a reading.

    The lifted clause is a claim about a file this module does not own: that
    ``FdtdDriver._inject_electric_through_conductivity`` replays the condinv
    rescale SPARSELY for a source that publishes a deposit table, and falls back
    to the WHOLE-VOLUME difference passes only for one that does not. If the
    driver ever loses the sparse branch, every reason string in this suite would
    still read correctly while the predicate silently admitted this cell's only
    corpus row on bytes that no longer match — rule 2's one silent failure, on a
    partition nothing else here can see.

    So this drives the driver's own method on both sides of its partition and
    reads the WORD, on the value class the whole lift is about: with a table
    published a ``-0.0`` FAR from the deposit survives as ``-0.0``
    (``0x80000000``) — which is what makes a point repair sufficient — and with
    NO table that same word comes back ``+0.0``, the canonicalisation the
    surviving clause refuses by name. The deposit cell is byte-identical under
    both, so the sparse replay reproduces the retired composition's arithmetic
    exactly where it still applies.

    The dense leg is not a hypothetical: it IS the retired driver, so the
    mutation this tripwire must catch is armed by construction rather than
    declared.
    """
    import struct
    import types

    from meep_gpu.driver import FdtdDriver

    def word(value) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    class _Deposits:
        """A scaled electric source that deposits at ONE cell, table optional."""

        field_type, component, is_integrated = "D", "Ez", False

        def __init__(self, table: bool):
            if table:
                self._point_ix = numpy.array([1])
                self._point_iy = numpy.array([1])
                self._point_iz = numpy.array([0])

        def inject(self, fields, _time):
            fields.Dz[1, 1, 0] += numpy.float32(1.0)

    seen = {}
    for table in (True, False):
        fields, _pml = _build()
        fields.Dz[...] = numpy.float32(0.0)
        fields.Dz[3, 3, 0] = numpy.float32(-0.0)      # FAR from the deposit.
        FdtdDriver._inject_electric_through_conductivity.__get__(
            types.SimpleNamespace(fields=fields), types.SimpleNamespace)(
                [_Deposits(table)], 0.0)
        seen[table] = (word(fields.Dz[3, 3, 0]), word(fields.Dz[1, 1, 0]))
    assert seen[True][0] == 0x80000000, seen      # sparse: the far -0.0 survives
    assert seen[False][0] == 0x00000000, seen     # dense: it is canonicalised
    assert seen[True][1] == seen[False][1], seen  # the deposit itself is unmoved


def test_the_clause_is_conditional_on_the_conductivity_not_on_the_seam(product):
    """A scaled electric source is only refused BECAUSE the run is conductive.

    Without a conductivity the driver injects point-wise (driver.py:3307-3308) and
    the ordinary fused pairs carry exactly this source; the clause must not be a
    blanket refusal wearing a conductivity's name. Measured through the shipped
    ordinary predicate, which admits the same source list on a lossless grid.
    """
    from meep_gpu.triton_kernels.coverage import fused_pair_coverage  # noqa: PLC0415

    fields, pml = _build(sigma=None)
    verdict = fused_pair_coverage(fields, pml, "D", (_Electric(),))
    assert not [r for r in verdict.reasons
                if "array module" not in r and "NON-INTEGRATED" in r]


def test_a_magnetic_source_never_reaches_this_seam(product):
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Magnetic(),)) == []


@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_ElectricWithoutIndex(),), "index"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    assert any(needle in r for r in _reasons(product, fields, pml, sources))


def test_the_flag_at_False_would_cost_every_integrated_electric_run(product,
                                                                   monkeypatch):
    """The deposit flag is measured, not declared.

    It is NOT what makes this cell reachable -- the corpus row is refused for the
    whole-volume rescale, whatever the flag says. What it decides is every conductive
    run whose electric source IS integrated, and that is asserted here by flipping it
    and re-asking rather than restated in prose.
    """
    fields, pml = _build()
    sources = (_IntegratedElectric(), _Magnetic())
    assert _reasons(product, fields, pml, sources) == []
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    refused = _reasons(product, fields, pml, sources)
    assert any("is electric" in r for r in refused), refused


def test_the_flag_is_passed_to_the_clause_that_reads_it_and_not_merely_declared():
    """``carries_repair=CARRIES_DEPOSIT_REPAIR`` reaches ``seam_source_reasons``."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text


def test_an_all_lossless_grid_is_refused_by_name(product):
    """That configuration is ``kernels.fused_curl_constitutive_D``'s own."""
    fields, pml = _build(sigma=None)
    assert any("no D component carries a conductivity" in r
               for r in _reasons(product, fields, pml, ()))


def test_a_mixed_per_component_conductivity_is_admitted(product):
    """One lossy component is enough; the other two compile to ``COND == 0``."""
    fields, pml = _build(components=("Dz",))
    assert _reasons(product, fields, pml, ()) == []
    from meep_gpu.triton_kernels.conductivity import conductive_targets
    assert conductive_targets(fields, "step_D") == (False, False, True)


def test_complex_storage_is_refused(product):
    fields, pml = _build(complex_storage=True)
    assert any("complex" in r for r in _reasons(product, fields, pml, ()))


def test_an_inactive_absorber_is_refused(product):
    """Without a layer the array path takes a DIFFERENT recurrence entirely."""
    fields, pml = _build(thickness=0)
    assert any("PML" in r or "pml" in r for r in _reasons(product, fields, pml, ()))


def test_a_fold_is_refused_by_name(product):
    """Both symmetry passes stop being no-ops and this weld carries neither."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields, pml = _build(symmetry=(Mirror("Y", 1),),
                         boundaries=("metallic", "periodic", "periodic"),
                         thickness=((2, 2), (0, 2), (0, 0)))
    refused = _reasons(product, fields, pml, ())
    assert any("mirror plane" in r for r in refused), refused


# ---------------------------------------------------------------------------
# The plan's bindings
# ---------------------------------------------------------------------------

def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_conductive_fused_electric_pair(fields, pml, ()) is None


def test_the_two_lattices_are_bound_the_way_the_array_path_reads_them(product):
    """Integer for the curl on ``step_D``, HALF-INTEGER for the constitutive on E.

    A swap is a silent half-cell error in the absorber profile, so the binding is
    read out of the builder's source rather than trusted.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "curl_spec['suffix']" in text
    assert "'_h' if side_spec['half_integer'] else ''" in text


def test_a_lossless_component_binds_its_own_target_as_the_placeholder(product):
    """Never a null: a pointer argument still has to type."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert text.count("CupyPointer(a if a is not None else b)") == 3


# ---------------------------------------------------------------------------
# Registration - the product is known to every record that must know it
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-11 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; on 2026-09-11
    ``dispatch_fused_route_2026-09-11_realarms`` drove it through the driver's own
    consults on ``conductive_2d`` — a case that already dispatched, whose D seam
    this product was SELECTED for and withheld from by ``fuse_labels`` alone
    (measured on the 2026-09-08 artifacts, whose ``fusion.not_installed`` reason
    for it reads "offered to run") — so the release names that case and the
    pending rung no longer holds it. The fourteen sibling products that were NOT
    driven keep the old name
    and the old assertion, which is what makes the two states legible per file.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its ledger entry was cut from this campaign's
    fleet artifact by ``seed_triton_welds.py``) and out of
    ``PENDING_DEVICE_GATE_ARMS``. Both are asserted rather than one, because a
    label released while still pending would be a plan claiming a certification
    the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_conductive_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["conductive_fused_electric_pair"]
    assert row["module"] == "conductive_fused_electric_pair"
    assert row["builder"] == "plan_conductive_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (conductive)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert _fastpath.RELEASED_FUSED_ARMS[label] == ("conductive_2d",)
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    text = (API_ROOT / "meep_gpu" /
            "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert "conductive_fused_electric_pair" in text


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists()
    assert "conductive_fused_electric_pair" in GATE.read_text(encoding="utf-8")


def test_the_weld_driver_the_board_and_the_battery_all_know_this_product():
    for name in ("drive_triton_weld_gates.py", "build_triton_fusion_matrix.py",
                 "triton_predicate_battery.py"):
        text = (PARITY_DIR / name).read_text(encoding="utf-8")
        assert "conductive_fused_electric_pair" in text, name


def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "conductive_fused_electric_pair" not in NOT_AN_ARM
    assert "conductive_fused_electric_pair" in _launch.SUPPORT_MODULES
def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it(product):
    """No byte-identity claim without a released gate record naming its directory."""
    status = product.DEVICE_STATUS
    if status.startswith("NOT RELEASED"):
        assert "gate.json" not in status
    else:
        assert "gate.json" in status and "results/" in status


# ---------------------------------------------------------------------------
# The gate - what it measures, checked on the merge bar
# ---------------------------------------------------------------------------

def load_gate():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_cfep_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    gate = load_gate()
    for name in ("transcription_leg", "equivalence_leg", "corpus_admission_leg",
                 "mutation_arming_leg", "coefficient_predicates_leg",
                 "whole_volume_rescale_leg"):
        leg = getattr(gate, name)()
        assert leg.get("passed") is True, (name, leg)


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name


def test_the_shipped_source_carries_the_helper_the_kernel_calls():
    """A mutant module without the helper would not import.

    And a mutation table that could not reach the helper could not arm a single
    rewrite of the four-case recurrence, which is this product's only new
    arithmetic.
    """
    gate = load_gate()
    source = gate.shipped_source()
    assert "def _conductive_component_registers" in source
    assert "def conductive_fused_curl_constitutive_D" in source
    assert source.count("@triton.jit") == 2


def test_every_mutation_is_scored_on_a_case_that_enters_the_branch_it_rewrites():
    gate = load_gate()
    for name, _kind, _reason, _rewrite in gate.mutation_table():
        gate.mutation_case_for(name)   # raises if the pairing is dead


def test_every_non_caught_mutation_carries_its_evidence():
    """`null` and `unreached` cost an entry in MUTATION_EVIDENCE.

    Without this a rewrite that stopped biting could be excused by relabelling it,
    which is the one failure in this whole gate that nothing else would turn red.
    """
    gate = load_gate()
    for name, _why, expectation, _rewrite in gate.mutation_table():
        assert expectation in ("caught", "null", "unreached"), (name, expectation)
        if expectation != "caught":
            assert gate.MUTATION_EVIDENCE.get(name), name


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls(product):
    gate = load_gate()
    assert tuple(gate.SEAM_PASSES) == tuple(product.REPLACES)


def test_the_gate_carries_a_carry_family_and_a_null_control():
    """Every carry case needs a bracket-removed control that MUST diverge."""
    gate = load_gate()
    assert gate.CARRY_CASES, "the integrated deposit is what the bracket carries"
    text = GATE.read_text(encoding="utf-8")
    assert "bracket" in text and "null" in text.lower()


def test_the_gate_carries_the_certified_kernel_identity_arm():
    """``COND = (0, 0, 0)`` must reproduce ``kernels.fused_curl_constitutive_D``."""
    text = GATE.read_text(encoding="utf-8")
    assert "cond_all_zero" in text
    assert "fused_curl_constitutive_D" in text


def test_the_gate_carries_a_MIXED_case(product):
    """One lossy component beside two lossless ones.

    The only shape that measures ``COND == 0`` beside a live ``COND == 1`` in the
    SAME launch, and the only case where the rewrite of the lossless arm is not a
    dead branch.
    """
    gate = load_gate()
    mixed = [case for case in gate.CASES if case[4]]
    assert mixed, "no case installs a per-component conductivity"
    assert gate.MUTATION_CASE.get(
        "m_lossless_arm_takes_the_conductive_one") in {c[0] for c in mixed}


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in ("meep_gpu/triton_kernels/conductive_fused_electric_pair.py",
                     "meep_gpu/triton_kernels/conductivity.py",
                     "meep_gpu/triton_kernels/kernels.py",
                     "meep_gpu/deposit_repair.py",
                     "meep_gpu/stepping.py",
                     "meep_gpu/driver.py",
                     "meep_gpu/test_triton_conductive_fused_electric_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "conductive PML", "ordinary")
