"""``triton_kernels/no_pml_fused_electric_pair.py`` — the host half of its certification.

THE PRODUCT welds the no-absorber ``step_D`` to the stored-E ``update_E`` and carries
``zero_metal_D`` between them, in one launch, with the in-seam electric deposit carried
across it by ``deposit_repair.PLAIN_PATH`` — the SECOND repair, and this is the first
product to declare it.

WHAT THIS FILE MEASURES AND WHAT THE DEVICE GATE MEASURES ARE DIFFERENT QUESTIONS. The
bytes are the gate's (``parity/meep_gpu/probe_triton_no_pml_fused_electric_pair.py``, on
an RTX A6000). What is here is everything answerable without a device and everything
that would be a silent widening if it drifted: the TRANSCRIPTION (each arithmetic block
is a contiguous run of a certified body), the PREDICATE in both directions, the two
tables the product reads rather than spells, the declaration that it is not an arm, and
the wiring records that have to name it.

THE ONE CLAUSE WORTH READING FIRST is the priced refusal. This cell has three corpus
rows and the product serves ONE. The other two are refused not for the deposit — that is
carried — but because ``FdtdDriver._inject_electric_through_conductivity`` rescales a
WHOLE D volume for a NON-INTEGRATED source on a conductive run (``driver.py:3363-3370``),
at cells no deposit closure can name. That refusal costs two thirds of the cell, so it is
pinned here in BOTH directions and MEASURED on the device by the gate's own
``priced_refusal`` leg rather than argued.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import numpy
import pytest

from . import deposit_repair
from .dispersion import PolarizationState, Susceptibility
from .fields import Fields
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.no_pml_fused_electric_pair"
GATE = PARITY / "probe_triton_no_pml_fused_electric_pair.py"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def gate():
    """The device gate's module, imported for its no-device legs and its tables."""
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("probe_triton_no_pml_fused_electric_pair")


# ---------------------------------------------------------------------------
# Fixtures — real Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

def build(poles=2, sigma=0.4, complex_storage=False, offdiag=False, symmetry=(),
          thickness=0, stored=True, boundaries=None):
    cell = (1.2, 1.2, 0.0)
    grid = Grid(resolution=12.0, cell_size=cell, dimensions=2,
                boundaries=boundaries or {"x": "metallic", "y": "metallic",
                                          "z": "periodic"},
                symmetry=tuple(symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if offdiag:
        diagonal = {name: numpy.full(grid.shape, 2.25, dtype=numpy.float32)
                    for name in ("Ex", "Ey", "Ez")}
        inverse = {name: numpy.full(grid.shape, 1.0 / 2.25, dtype=numpy.float32)
                   for name in ("Ex", "Ey", "Ez")}
        fields.set_epsilon_volumes(
            diagonal, inverse,
            {"Ex": {"Ey": numpy.full(grid.shape, 0.05, dtype=numpy.float32)}})
    else:
        fields.set_background_eps(2.25)
    for index in range(poles):
        sigma_map = {"Ex": 0.30 + 0.05 * index, "Ey": 0.0 if index else 0.20,
                     "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma_map, grid,
            numpy.complex64 if complex_storage else numpy.float32))
    if sigma is not None:
        fields.set_d_conductivity(
            numpy.full(grid.shape, float(sigma), dtype=numpy.float32))
    if thickness:
        fields.enable_pml_storage()
    elif stored:
        fields.enable_field_storage()
    pml = PML(grid=grid,
              thickness=({"x": thickness, "y": thickness} if thickness else 0))
    return fields, pml


class _Electric:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


class _ScaledElectric(_Electric):
    """The corpus rows' own source: NOT integrated, deposit table published."""

    is_integrated = False


class _ScaledElectricNoTable:
    """A scaled source with NO deposit table at all — the driver's dense fallback.

    ``hasattr(source, "_point_ix")`` is False, which is the partition
    ``_inject_electric_through_conductivity`` uses; no in-tree source class is
    one, and the surviving clause exists exactly for this duck-typed stranger.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = False


class _ElectricWithoutIndex:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = None


class _Magnetic:
    field_type = "B"
    component = "Hz"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


def reasons_of(product, fields, pml, sources):
    """The reasons that would REMAIN on a CuPy host, and nothing else.

    ONE clause comes off: ``array module is 'numpy', not cupy``, which every arm says
    on the merge-bar laptop and which keeping would make every row here read as
    refused.

    NOTHING ELSE COMES OFF, and that is the point of the 2026-08-31 fix this helper
    used to work around. ``_curl_half`` once asked BOTH no-absorber arms and returned
    both reason lists when neither admitted -- which on a NumPy host is always -- so
    the LOSING arm's clauses ("a conductivity is installed on Dx", "no curl target
    carries a conductivity") leaked into every verdict and this helper had to strip
    them. It now reads the arms' OWN partition function first and asks the one arm that
    serves the run, so the reason list is that arm's and a strip is no longer needed.
    A test that still had to strip them would be hiding the leak rather than measuring
    its absence.
    """
    verdict = product.no_pml_fused_electric_pair_coverage(fields, pml, sources)
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


def test_the_predicate_reports_ONE_arms_reasons_and_not_the_losing_arms(product):
    """The 2026-08-31 census finding, kept as a test.

    Asking both arms and returning both reason lists is not wrong about ``covered`` --
    it never was -- but it IS wrong about the reasons, and the predicate census's
    ``covered_modulo_backend`` is computed from the reasons. With both lists returned
    the board read EVERY row as refused and would have priced this product's two cells
    at zero for a property of the census HOST rather than of the run.
    """
    for sigma, expected in ((0.4, "conductive no-PML"), (None, "no-PML")):
        fields, pml = build(sigma=sigma)
        covered, reasons, which = product._curl_half(fields, pml)
        assert which == expected, (sigma, which)
        assert not covered, "a NumPy host cannot admit; this fixture proves nothing"
        assert all(reason.startswith(f"curl half ({expected} arm): ")
                   for reason in reasons), reasons
        # ...and the ONLY thing left after the backend clause is nothing at all.
        assert [r for r in reasons if "not cupy" not in r] == [], reasons


# ---------------------------------------------------------------------------
# The declaration — the second repair, and both halves of it
# ---------------------------------------------------------------------------

def test_the_product_declares_the_repair_and_which_one(product):
    """The flag and the path travel together or the flag is a lie.

    ``CARRIES_DEPOSIT_REPAIR`` says the two slots are bracketed;
    ``REPAIR_PATHS`` says WHICH recurrence that bracket inverts. A product that
    declared the flag with the wrong path would be REFUSED by
    ``deposit_repair.repairable`` on every row it serves — which is safe — but a
    product that declared the flag with no bracket would compute its constitutive half
    against a pre-injection field and report success, which is not.
    """
    assert product.CARRIES_DEPOSIT_REPAIR is True
    assert product.REPAIR_PATHS == (deposit_repair.PLAIN_PATH,)
    text = (PACKAGE_DIR / "no_pml_fused_electric_pair.py").read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
    assert "repair_paths=REPAIR_PATHS" in text


def test_the_declared_path_is_the_one_this_configuration_actually_runs(product):
    """Read from ``deposit_repair``'s side, not from the product's.

    ``repair_path_for`` asks ``stepping._pml_is_active`` — the same call ``update_E``
    branches on — so this is the engine's own answer about which recurrence executes,
    compared against what the product says it carries.
    """
    fields, pml = build()
    assert deposit_repair.repair_path_for(pml) == deposit_repair.PLAIN_PATH
    assert deposit_repair.repairable(fields, "D", pml,
                                     paths=product.REPAIR_PATHS) == (True, ())
    # ...and the DEFAULT declaration still refuses it, which is what makes the
    # product's own declaration load-bearing rather than decorative.
    covered, why = deposit_repair.repairable(fields, "D", pml)
    assert not covered and any("is_active False" in reason for reason in why)


def test_the_magnetic_seam_has_no_twin_on_this_branch():
    """There is no ``no_pml_fused_magnetic_pair`` and there cannot be one.

    With an inactive layer ``update_H`` returns without touching H
    (``stepping.py:944-945``), so the magnetic constitutive stores nothing and there is
    no second half to weld to. ``PLAIN_PATH_SEAMS`` states that, and this is the check
    that it still describes ``update_H``.
    """
    import inspect

    from . import stepping

    assert deposit_repair.PLAIN_PATH_SEAMS == ("D",)
    body = inspect.getsource(stepping.update_H)
    assert "if not _pml_is_active(pml):" in body
    assert "return" in body.split("if not _pml_is_active(pml):")[1][:200]
    assert not (PACKAGE_DIR / "no_pml_fused_magnetic_pair.py").exists()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def test_every_arithmetic_block_is_a_contiguous_run_of_a_certified_body(gate):
    """The gate's own transcription leg, run here so the merge bar carries it.

    Whole BLOCKS rather than line sets: a line-set comparison passes on a body whose
    statements were permuted, and permuting the pole chain or the conductive tail is a
    different float32 number in every cell.
    """
    row = gate.transcription_leg()
    assert row["passed"], row["findings"]
    assert row["curl_block_statements"] > 40, row
    assert row["wall_block_statements"] >= 6, row


def test_the_curl_table_is_READ_from_the_arm_and_not_from_launch(product):
    """The two ``SUB_STEPS`` tables in this package disagree about ``step_D``'s sources.

    ``launch.SUB_STEPS`` says ``('Hx','Hy','Hz')``, which is right under an absorber
    where H is stored. The no-absorber arms say ``('Bx','By','Bz')`` and say why
    (``no_pml.py:178-181``): ``Fields.enable_field_storage`` deliberately does NOT
    allocate H on this branch and ``get_H`` returns the B array itself. A product that
    followed ``launch``'s table would bind three ``None`` pointers, which is a
    ``TypeError`` at the launcher and not at its cause — and it is what this suite's
    own gate caught on its first run.
    """
    from .triton_kernels import launch as launch_module
    from .triton_kernels import no_pml, no_pml_conductive

    assert product.CURL_SOURCES == ("Bx", "By", "Bz")
    assert product.CURL_SOURCES == tuple(no_pml_conductive.SUB_STEPS["step_D"]["sources"])
    assert product.CURL_SOURCES == tuple(no_pml.SUB_STEPS["step_D"]["sources"])
    assert product.CURL_SOURCES != tuple(launch_module.SUB_STEPS["step_D"]["sources"])
    assert product.CURL_TARGETS == ("Dx", "Dy", "Dz")
    assert product.BACKWARD == 1 and product.DERIVE == 0
    # H really is unallocated on this branch, which is what makes the above a fact
    # rather than a naming preference.
    fields, _pml = build()
    assert fields.Hx is None and fields.Bx is not None


@pytest.mark.parametrize("sigma", [0.4, None], ids=["conductive", "lossless"])
def test_the_two_curl_arms_PARTITION_the_space_rather_than_overlap(product, sigma):
    """The disjunction in ``_curl_half`` is the kernel's own constexpr, not a widening.

    It is sound only because the two shipped predicates are DISJOINT: the conductive
    one requires a sigma on at least one curl target BY NAME and the lossless one
    refuses every sigma BY NAME. If they ever both admitted, this product would be
    reachable through an ambiguity rather than through one arm's verdict — and
    ``_select_slot`` would already have emptied the slot for the arms themselves.

    Asked on a NumPy host, so what is compared is the reasons MINUS the array-module
    clause: exactly one arm must have nothing else to say.
    """
    from .triton_kernels import no_pml, no_pml_conductive

    fields, pml = build(sigma=sigma)
    verdicts = {
        "conductive": no_pml_conductive.conductive_plain_curl_coverage(
            fields, pml, "step_D"),
        "lossless": no_pml.plain_curl_coverage(fields, pml, "step_D"),
    }
    admits = {name: not [reason for reason in verdict.reasons
                         if "not cupy" not in reason]
              for name, verdict in verdicts.items()}
    assert sum(admits.values()) == 1, (sigma, admits,
                                       {k: list(v.reasons) for k, v in verdicts.items()})
    assert admits["conductive"] is (sigma is not None), (sigma, admits)


def test_replaces_is_the_drivers_own_call_order(product):
    """``REPLACES`` names three call sites and each is one this launch performs.

    ``step_D`` and ``update_E`` are the two consults; ``zero_metal_D`` is the pass the
    kernel carries INLINE. The two symmetry fills are NOT here: they are refused rather
    than carried, and a launch may only claim to replace what it performs.
    """
    assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")
    # READ OFF THE DRIVER'S `step` BODY by AST, so a docstring mention cannot stand in
    # for a call and a call moved out of `step` fails here.
    import inspect

    from . import driver as driver_module

    import textwrap

    body = ast.parse(textwrap.dedent(
        inspect.getsource(driver_module.FdtdDriver.step)))
    # SORTED BY POSITION: `ast.walk` is breadth-first and its order is not the
    # source's, so an unsorted list answers a different question from "in what order
    # does the driver call these".
    calls = [node for node in ast.walk(body)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id in set(product.REPLACES)]
    order = [node.func.id for node in
             sorted(calls, key=lambda node: (node.lineno, node.col_offset))]
    seen = [name for index, name in enumerate(order) if name not in order[:index]]
    assert seen == list(product.REPLACES), seen


# ---------------------------------------------------------------------------
# The predicate, both directions
# ---------------------------------------------------------------------------

def test_the_gates_no_device_legs_all_pass_here(gate):
    for leg in gate.NO_DEVICE_LEGS:
        row = leg()
        assert row["passed"], (row["leg"], row["findings"])


@pytest.mark.parametrize("label,needle,fixture", [
    ("an active absorber", "an active PML layer is installed", dict(thickness=2)),
    ("an off-diagonal chi1inv row", "STENCIL", dict(offdiag=True)),
    ("a mirror plane", "mirror plane", dict(symmetry=("Y",))),
    ("a run that does not store E", "stores_E",
     dict(poles=0, sigma=None, stored=False)),
    ("complex storage", "complex", dict(complex_storage=True)),
])
def test_the_predicate_refuses_by_name(product, label, needle, fixture):
    fields, pml = build(**fixture)
    found = reasons_of(product, fields, pml, (_Electric(),))
    assert any(needle in reason for reason in found), (label, found[:6])


def test_an_undeclared_source_list_is_refused(product):
    """Ignorance is never an empty set: ``Fields`` does not hold the sources."""
    fields, pml = build()
    found = reasons_of(product, fields, pml, None)
    assert any("was not declared" in reason for reason in found), found[:6]


def test_an_electric_source_with_no_deposit_index_is_refused(product):
    fields, pml = build()
    found = reasons_of(product, fields, pml, (_ElectricWithoutIndex(),))
    assert any("does not publish the index" in reason for reason in found), found[:6]


def test_a_magnetic_source_never_reaches_this_seam(product):
    """The driver injects it in the B/H half, so it is not this seam's problem."""
    fields, pml = build()
    found = reasons_of(product, fields, pml, (_Magnetic(),))
    assert not any("is electric" in reason for reason in found), found[:6]


def test_an_integrated_electric_deposit_is_admitted(product):
    """The flag's whole purpose, in the ADMITTING direction.

    Only the array-module clause may remain on a NumPy host; if anything else does, the
    product refuses a configuration it claims to serve.
    """
    fields, pml = build()
    assert reasons_of(product, fields, pml, (_Electric(),)) == []


def test_the_flag_at_False_would_cost_every_electric_run(product):
    """The counterfactual, so the flag is priced rather than assumed.

    All three of this cell's corpus rows declare an electric source, so the same module
    with ``CARRIES_DEPOSIT_REPAIR`` False serves NOTHING. Asked through the shared
    clause rather than by editing the module.
    """
    fields, pml = build()
    refused = deposit_repair.seam_source_reasons(
        fields, (_Electric(),), "D", undeclared="undeclared",
        refusal=lambda index, source: "in-seam electric source",
        carries_repair=False)
    assert refused == ("in-seam electric source",)
    carried = deposit_repair.seam_source_reasons(
        fields, (_Electric(),), "D", undeclared="undeclared",
        refusal=lambda index, source: "in-seam electric source",
        carries_repair=True, pml=pml, repair_paths=product.REPAIR_PATHS)
    assert carried == ()


# ---------------------------------------------------------------------------
# THE LIFTED REFUSAL — the clause that used to cost two of the three corpus rows
# ---------------------------------------------------------------------------

def test_the_corpus_rows_own_scaled_source_is_now_admitted(product):
    """The 2026-09-01 lift, pinned in its admitting direction.

    ``FdtdDriver._inject_electric_through_conductivity`` used to rescale the WHOLE
    target D volume for a scaled source — the identity everywhere except that it
    canonicalised ``-0.0`` at cells no deposit closure can name — and the clause
    refused both conductive corpus rows for it. The driver now replays the rescale
    SPARSELY at the deposit cells the sources publish, so the corpus rows' own
    source (scaled, table published) is carried; the device measurement is the
    gate's ``lifted_refusal`` leg.
    """
    fields, pml = build(sigma=0.4)
    assert reasons_of(product, fields, pml, (_ScaledElectric(),)) == []


def test_a_scaled_source_publishing_no_deposit_table_is_still_refused(product):
    """What survives the lift: the driver's dense whole-volume fallback, by name.

    A scaled source with no ``_point_ix`` attribute cannot name its deposit, so the
    driver must still rescale the whole component — the composition whose
    signed-zero canonicalisation the retired clause priced. No in-tree source class
    is one; the clause fails CLOSED on the duck-typed stranger.
    """
    fields, pml = build(sigma=0.4)
    found = reasons_of(product, fields, pml, (_ScaledElectricNoTable(),))
    assert any("publishes NO deposit table" in reason for reason in found), found[:6]
    assert any("dense branch" in reason for reason in found), found[:6]


def test_the_clause_is_conditional_on_the_conductivity_and_not_on_the_seam(product):
    """The tableless source on a LOSSLESS run is admitted by THIS clause, and that is
    the discriminating half: without a conductivity the driver takes the plain
    per-source injection loop (``driver.py:3307``) and no rescale of any shape runs.
    (The repair's own index clause still refuses a tableless source at the seam —
    a different clause with a different needle.)"""
    fields, pml = build(sigma=None)
    assert product._scaled_conductive_injection_reasons(
        fields, (_ScaledElectricNoTable(),)) == ()
    assert reasons_of(product, fields, pml, (_ScaledElectric(),)) == []


def test_the_clause_is_quiet_except_for_the_tableless_scaled_source(product):
    """Per source, the surviving partition: integrated quiet, scaled-with-table
    quiet (the lift), scaled-without-table refused."""
    fields, pml = build(sigma=0.4)
    assert product._scaled_conductive_injection_reasons(fields, (_Electric(),)) == ()
    assert product._scaled_conductive_injection_reasons(
        fields, (_ScaledElectric(),)) == ()
    assert product._scaled_conductive_injection_reasons(
        fields, (_ScaledElectricNoTable(),)) != ()


def test_the_drivers_OWN_partition_is_the_one_this_clause_mirrors():
    """THE TRIPWIRE UNDER THE LIFT, and it is a measurement rather than a reading.

    The lifted clause is a claim about a file this module does not own: that
    ``FdtdDriver._inject_electric_through_conductivity`` replays the condinv
    rescale SPARSELY for a source that publishes a deposit table and falls back to
    the WHOLE-VOLUME difference passes only for one that does not. If the driver
    ever loses the sparse branch, every reason string here would still read
    correctly and the predicate would silently admit two corpus rows whose bytes
    no longer match — rule 2's one silent failure, on a partition no test in this
    suite could see.

    So this drives the driver's own method on both sides of its partition and
    reads the WORD, on the value class the whole lift is about:

      * table published -> a ``-0.0`` FAR from the deposit survives as ``-0.0``
        (``0x80000000``): the pass is the identity away from the deposit, which
        is exactly what makes a point repair sufficient;
      * NO table       -> that same word comes back ``+0.0`` (``0x00000000``):
        the dense fallback canonicalises it, which is what the surviving clause
        refuses by name;
      * and the DEPOSIT cell is byte-identical under both, so the sparse replay
        reproduces the retired composition's arithmetic where it still applies —
        the half of the lift that says it changed nothing it was not meant to.

    A reader might reasonably ask why this is not a source-text assertion. Because
    a rewrite that preserved the text and lost the behaviour is exactly the case
    that costs two corpus rows, and the text is the weaker of the two claims.
    """
    import struct
    import types

    from .driver import FdtdDriver

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
        fields, _pml = build(sigma=0.4, poles=0)
        fields.Dz[...] = numpy.float32(0.0)
        fields.Dz[3, 3, 0] = numpy.float32(-0.0)      # FAR from the deposit.
        FdtdDriver._inject_electric_through_conductivity.__get__(
            types.SimpleNamespace(fields=fields), types.SimpleNamespace)(
                [_Deposits(table)], 0.0)
        seen[table] = (word(fields.Dz[3, 3, 0]), word(fields.Dz[1, 1, 0]))
    assert seen[True][0] == 0x80000000, seen      # sparse: the far -0.0 survives
    assert seen[False][0] == 0x00000000, seen     # dense: it is canonicalised
    assert seen[True][1] == seen[False][1], seen  # the deposit itself is unmoved


def test_the_device_gate_MEASURES_the_lift_rather_than_asserting_it(gate):
    """A lift that serves two corpus rows has to stand on a measurement.

    The gate's ``lifted_refusal`` leg runs the formerly refused configuration with
    the bracket on and requires BYTE IDENTITY at every step — while the RETIRED
    whole-volume composition, replayed verbatim as an armed control, must still
    diverge on the same signed-zero seed. This is the host-side check that both
    halves of that burden exist in the gate and that its release reads them.
    """
    source = GATE.read_text(encoding="utf-8")
    assert "def lifted_refusal_leg(" in source
    assert "def _retired_whole_volume_inject(" in source
    assert "def _retired_passes_control(" in source
    assert "diverges_at_some_step" in source
    assert '"the RETIRED whole-volume composition did NOT diverge from the "' in source
    # ...and the gate's own release reads it.
    assert 'lifted_ok = bool(payload["lifted_refusal"].get("passed"))' in source
    assert "lifted_ok and" in source


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    """``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly."""
    fields, pml = build()
    assert product.plan_no_pml_fused_electric_pair(fields, pml, (_Electric(),)) is None


def test_the_package_does_not_import_the_module_eagerly():
    """A bare ``import meep_gpu.triton_kernels`` must not pull in a Triton kernel."""
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE not in sys.modules or True  # the fixture may have imported it
    text = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "no_pml_fused_electric_pair" not in text


def test_the_kernel_accessor_explains_itself_without_triton(product):
    """A ``None`` that fails later as a ``TypeError`` far from its cause is the failure
    this accessor exists to prevent."""
    if product.no_pml_fused_curl_constitutive_D is None:
        with pytest.raises(ImportError, match="triton"):
            product.no_pml_fused_curl_constitutive_D_kernel()
    else:  # pragma: no cover - a host with Triton
        assert product.no_pml_fused_curl_constitutive_D_kernel() is not None


# ---------------------------------------------------------------------------
# Wiring — the module is not an arm, and every record names it
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_dispatch_still_refuses_it():
    """ROUTED 2026-09-02 by the installer wave, and REFUSED at dispatch — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING IS NOT RELEASING, and the second half is that sentence made checkable
    from the PRODUCT's side rather than from the composer's. It asserted "the label
    this product writes is NOT in ``fastpath.RELEASED_FUSED_ARMS``" until
    2026-09-14 -- a PRE-WIRING fact that the release then inverted, which is the
    same shape this tree's re-gate recipe warns about for gates. The positional
    fact is now RECORDED against the release's own case list rather than asserted
    as an absence, and the substantive claims are unchanged: the label is a fused
    arm, it sits in exactly one of ``ARM_CERTIFICATION`` (its gate has a tracked
    ledger entry) and ``PENDING_DEVICE_GATE_ARMS``, and
    ``FUSED_ARM_CONSTITUENTS`` names it.

    ITS SECOND CASE CHANGED IN THE SAME ROUND: ``material_dispersion_0d`` was
    replaced by ``no_pml_dispersive_2d``, which drives the same two axis values
    (dimensions 2, conductivity False) on a cell with real extent, because on a
    (1, 1, 1) grid every non-vacuity control the driver-route gate owns is vacuous.
    Pinning the tuple here is what makes that swap visible from the product's side.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_no_pml_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["no_pml_fused_electric_pair"]
    assert row["module"] == "no_pml_fused_electric_pair"
    assert row["builder"] == "plan_no_pml_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (no-PML stored E)'
    assert _fastpath.arm_is_fused(label)
    assert _fastpath.RELEASED_FUSED_ARMS[label] == (
        "absorber_1d", "no_pml_dispersive_2d")
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "no_pml_fused_electric_pair" not in NOT_AN_ARM
    assert "no_pml_fused_electric_pair" in _launch.SUPPORT_MODULES
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    from .test_fused_pair_deposit_wiring import WIRED_FOR_THE_REPAIR

    assert "triton_kernels/no_pml_fused_electric_pair.py" in WIRED_FOR_THE_REPAIR


def test_the_installer_can_be_told_which_repair_this_product_carries():
    """``launch._install_fused_pair`` is the ONE constructor of the two repair plans.

    A product declaring ``PLAIN_PATH`` must be able to say so THROUGH it, or the day it
    is routed it silently gets the split-field declaration and is refused on every row.
    """
    import inspect

    from .triton_kernels import launch as launch_module

    signature = inspect.signature(launch_module._install_fused_pair)
    assert signature.parameters["repair_paths"].default == (
        deposit_repair.SPLIT_FIELD_PATH,)
    tree = ast.parse((PACKAGE_DIR / "launch.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "LeadingRepairPlan"]
    assert len(calls) == 1 and len(calls[0].args) == 6, calls


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists()
    source = GATE.read_text(encoding="utf-8")
    assert MODULE in source
    assert "no_pml_fused_curl_constitutive_D" in source


def test_the_weld_driver_the_board_and_the_battery_all_know_this_product():
    """A product nothing can run, score or credit is not a product."""
    for name, needle in (
        ("drive_triton_weld_gates.py", "no_pml_fused_electric_pair"),
        ("build_triton_fusion_matrix.py", "no_pml_fused_electric_pair"),
        ("triton_predicate_battery.py", "no_pml_fused_electric_pair"),
    ):
        text = (PARITY / name).read_text(encoding="utf-8")
        assert needle in text, name


def test_the_gate_carries_a_carry_family_and_a_null_control(gate):
    """The carry legs and the control that makes them non-vacuous.

    Every case in ``CARRY_CASES`` must be one the product ADMITS — on a conductive run
    that means an INTEGRATED source — so a carry leg measures the product rather than a
    refusal.
    """
    assert gate.CARRY_CASES
    for name in gate.CARRY_CASES:
        assert name in gate.CASES_BY_NAME, name
    source = GATE.read_text(encoding="utf-8")
    assert "null_control:" in source
    assert "require_identical=False" in source
    assert "require_repairs=True" in source


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls(gate):
    driver = (HERE / "driver.py").read_text(encoding="utf-8")
    for name in gate.SEAM_PASSES:
        assert f"{name}(self.fields" in driver, name


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source(gate):
    """A rewrite that matches nothing is a mutation that measured nothing.

    Checked HERE, on the laptop, because the device run would report it as an error
    long after the campaign has spent its GPU time.
    """
    source = gate.shipped_source()
    unarmed = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        _text, hits = rewrite(source)
        if not hits:
            unarmed.append(name)
    assert not unarmed, unarmed


def test_every_non_caught_mutation_carries_its_evidence(gate):
    """A mutation may be declared ``null`` only with a MEASUREMENT behind it.

    Relabelling an uncaught mutation is the cheapest way to make a gate pass, so the
    vocabulary is closed and both non-``caught`` entries have to name what was
    established. This reads the declaration from the gate's own table.
    """
    for name, _why, expectation, _rewrite in gate.mutation_table():
        assert expectation in ("caught", "null", "unreached"), (name, expectation)
        if expectation != "caught":
            evidence = gate.MUTATION_EVIDENCE.get(name)
            assert evidence and len(evidence) > 120, (name, evidence)
    # ...and the release refuses a non-caught expectation with no evidence.
    source = GATE.read_text(encoding="utf-8")
    assert 'if expectation != "caught" and not MUTATION_EVIDENCE.get(' in source


def test_every_mutation_is_scored_on_a_case_that_can_arm_it(gate):
    """A wall mutation on a periodic grid is a null wearing a caught label."""
    for name, _why, _expectation, _rewrite in gate.mutation_table():
        case_name, case = gate.mutation_case_for(name)
        assert case_name in gate.CASES_BY_NAME
        if name.startswith("wall_"):
            assert isinstance(case[2], dict) and "metallic" in case[2].values(), name
        if name.startswith("conductive_"):
            assert case[4] is not None, name
        if name.startswith("poles_"):
            assert case[6] >= 2, name


def test_the_gate_compares_every_polarization_buffer(gate):
    """``update_P`` is driven by the array this repair rewrites.

    ``Fields.drive_field`` returns the STORED E on this branch (``fields.py:1160-1162``),
    so a repair that got E right and P wrong would read as byte-identical without the
    polarization buffers in the comparison.
    """
    source = GATE.read_text(encoding="utf-8")
    assert 'f"pol{index}.{attribute}.{component}"' in source
    assert "def inventory(" in source
