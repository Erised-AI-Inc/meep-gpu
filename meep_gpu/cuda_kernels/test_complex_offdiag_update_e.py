"""The complex x off-diagonal ``update_E`` family: its bytes, clauses and hazards.

WHAT RUNS HERE AND WHAT IS OWED. A kernel has to be launched to be compared and
this host has no CuPy, so the BIT-IDENTITY question belongs to
``parity/meep_gpu/gate_cuda_complex_offdiag_update_e.py`` and its numbers land in
:data:`complex_offdiag_update_e.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION`.
:func:`test_admission_record_is_all_or_nothing` refuses a half-filled record, so a
green suite here can never be mistaken for a device verdict.

WHAT DOES RUN HERE is everything whose failure mode is a SILENT WRONG ANSWER that
no device would report:

1. **THE ARITHMETIC, WORD FOR WORD AGAINST THE ORACLE.** The EMITTED text is
   executed in float32 NumPy by the interpreter of ``test_complex_chunk_forms.py``
   through the family's own launcher (the run's chunk record, the presence words,
   both connection tables), and :func:`test_structure_matches_the_oracle` compares
   one launch against ``stepping.update_E`` WORD FOR WORD on real
   ``Grid``/``Fields``/``PML`` objects over folds on every axis, both terminations,
   both plane parities, Bloch phases, metallic walls, 3-D / 2-D / 1-D shapes, four
   row masks, two courants and both tails. There is no second transcription to
   drift: the interpreter executes the characters the device compiles, and a
   planted defect in them must change a word
   (:func:`test_each_planted_defect_changes_a_word`).

2. **THE TWO-KERNEL DECISION, MEASURED.** The module claims the split-field tail
   and the direct store are different COMPUTATIONS rather than one computation at
   different coefficients. :func:`test_the_two_tails_are_different_computations`
   measures that on the ARRAY PATH -- one fixture stepped with an inert layer and
   with a synthetic unit-coefficient layer -- and prints the differing word count.

3. **THREE FLOAT32 HAZARDS, MEASURED ON THIS HOST**, each printed with its count.
   A claim about rounding that no test evaluates is a claim, not a measurement.

4. **THE PREDICATE VERDICT TABLE**, on REAL engine objects rather than
   dictionaries, because a dictionary answers the way the test author expected and
   the engine answers the way the engine does.

5. **THE PARTITION** -- no slot may be admitted by this family and by a shipped
   sibling at once. Two admitters leave a slot unselected and it falls back to the
   array path: a silent coverage LOSS, not an error.

6. **THE PLATFORM REQUIREMENTS** -- ASCII, no division, no unary minus on a float
   path, no call into the sibling prelude's dead helpers -- which killed sibling
   kernels at their first launch and are cheap to pin.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

import numpy
import pytest

from .. import stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import complex_emitter, coverage
from . import complex_offdiag_update_e as family
from . import folded_offdiag_kernels as folded
from . import offdiag_emitter
from .test_complex_chunk_forms import launcher  # noqa: F401 - the NumPy-loaded launcher

HERE = pathlib.Path(__file__).parent
MODULE_PATH = HERE / "complex_offdiag_update_e.py"
STEPPING_PATH = HERE.parent / "stepping.py"

TAILS = ("pml", "no_pml")
MASKS = ((1, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 0), (1, 1, 0, 0, 0, 0),
         (0, 0, 1, 0, 0, 0))


class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``, with NumPy's own dtype objects.

    The dtype identity matters: ``xp.complex64`` has to be the object the arrays
    actually carry, or a dtype clause tests the stand-in instead of the array.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(numpy, item)


XP = _NumpyWearingCupysName()


# ---------------------------------------------------------------------------
# Fixtures: REAL engine objects
# ---------------------------------------------------------------------------

def build(*, cell=(8.0, 10.0, 12.0), boundaries=("periodic",) * 3, symmetry=(),
          k_point=(0.0, 0.0, 0.0), dimensions=3, courant=0.35,
          mask=(1, 0, 0, 1, 0, 0), active=True, complex_storage=True,
          seed=20260820, xp=XP, rows_uniform=False):
    """A frozen ``(fields, layer, grid)`` triple in this family's shape."""
    rng = numpy.random.default_rng(seed)
    grid = Grid(resolution=1.0, cell_size=tuple(cell),
                boundaries=tuple(boundaries),
                symmetry=tuple(Mirror(name, phase) for name, phase in symmetry),
                xp=xp, courant=courant, k_point=tuple(k_point),
                dimensions=dimensions)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    if active:
        fields.enable_pml_storage()
    shape = tuple(grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        # THREE DISTINCT, INHOMOGENEOUS volumes drawn away from 1.0. Against ones
        # the multiply is invisible; against one shared volume a component-aliasing
        # defect is invisible; against a constant volume a coefficient-INDEX error
        # is invisible.
        values = rng.uniform(1.2, 3.4, size=shape).astype(numpy.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    rows = {}
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        if rows_uniform:
            values = numpy.full(shape, numpy.float32(0.3125), dtype=numpy.float32)
        else:
            # STRADDLING ZERO: the inverse of a real tensor carries NEGATIVE
            # off-diagonals, and a sign error that only shows on one side of zero
            # is exactly the kind a positive-only draw hides.
            values = rng.uniform(-0.45, 0.45, size=shape).astype(numpy.float32)
        rows.setdefault(row, {})[partner] = xp.asarray(values)
    fields.set_epsilon_volumes(epsilon, inverse,
                              chi1inv_offdiagonal=rows or None)
    names = ["Dx", "Dy", "Dz", "Ex", "Ey", "Ez"]
    if active:
        names += ["f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for name in names:
        real = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
        if not complex_storage:
            getattr(fields, name)[...] = xp.asarray(numpy.ascontiguousarray(real))
            continue
        host = numpy.empty(shape, dtype=numpy.complex64)
        # THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``: ``1j*im``
        # carries a real part of ``0.0 * im``, so the sum computes
        # ``(-0.0) + (+0.0) = +0.0`` and destroys every negative zero.
        host.real = real
        host.imag = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
        getattr(fields, name)[...] = xp.asarray(numpy.ascontiguousarray(host))
    layer = None
    if active:
        # THE THICKNESS RULE IS THE SIBLING FOLD GATES': skip an axis too thin to
        # hold a layer, and on a MIRRORED axis ask for the HIGH face only, because
        # ``PML._resolve_mirror_faces`` refuses a named low face on a folded axis
        # (cell 0 is the mirror plane, a boundary condition rather than a wall).
        thickness = tuple(
            (0, 0) if grid.shape[axis] < 6
            else (0, 2) if grid.is_mirrored(axis)
            else (2, 2)
            for axis in range(3))
        layer = PML(grid=grid, thickness=thickness)
    return fields, layer, grid


#: Every structural configuration the evaluator leg sweeps. EVERY AXIS IS FOLDED
#: SOMEWHERE, both plane parities appear, both declared terminations appear, a
#: Bloch phase appears on an unfolded axis BESIDE a fold, metallic walls appear,
#: and 3-D / 2-D / 1-D shapes all appear.
SPECS = (
    {"label": "pml_unfolded_periodic_phased", "active": True,
     "cell": (8.0, 10.0, 12.0), "boundaries": ("periodic",) * 3,
     "symmetry": (), "k_point": (0.2, -0.35, 0.1)},
    {"label": "pml_unfolded_walls_phased", "active": True,
     "cell": (9.0, 10.0, 11.0), "boundaries": ("metallic", "periodic", "metallic"),
     "symmetry": (), "k_point": (0.0, 0.2, 0.0)},
    {"label": "pml_fold_X_metallic_even", "active": True,
     "cell": (16.0, 10.0, 12.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_X_periodic_odd", "active": True,
     "cell": (16.0, 10.0, 12.0), "boundaries": ("periodic",) * 3,
     "symmetry": (("X", -1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_Y_phased_X", "active": True,
     "cell": (10.0, 16.0, 12.0), "boundaries": ("periodic",) * 3,
     "symmetry": (("Y", 1),), "k_point": (0.31, 0.0, 0.0)},
    {"label": "pml_fold_Z_metallic_odd", "active": True,
     "cell": (8.0, 12.0, 16.0),
     "boundaries": ("periodic", "periodic", "metallic"),
     "symmetry": (("Z", -1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_X_wall_Y", "active": True,
     "cell": (16.0, 10.0, 12.0),
     "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_XY_mixed", "active": True,
     "cell": (16.0, 16.0, 10.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1)), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_XYZ", "active": True,
     "cell": (16.0, 16.0, 16.0), "boundaries": ("periodic",) * 3,
     "symmetry": (("X", 1), ("Y", 1), ("Z", 1)), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_2d_phased", "active": True, "dimensions": 2,
     "cell": (16.0, 14.0, 0.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.27, 0.0)},
    {"label": "store_3d_phased", "active": False,
     "cell": (8.0, 10.0, 12.0), "boundaries": ("periodic",) * 3,
     "symmetry": (), "k_point": (0.23, -0.17, 0.35)},
    {"label": "store_2d_phased", "active": False, "dimensions": 2,
     "cell": (12.0, 14.0, 0.0), "boundaries": ("periodic",) * 3,
     "symmetry": (), "k_point": (0.3892, 0.1597, 0.0)},
    {"label": "store_1d_phased", "active": False, "dimensions": 1,
     "cell": (0.0, 0.0, 24.0), "boundaries": ("periodic",) * 3,
     "symmetry": (), "k_point": (0.0, 0.0, 0.35)},
    {"label": "store_metallic_walls", "active": False,
     "cell": (9.0, 10.0, 11.0),
     "boundaries": ("metallic", "metallic", "periodic"),
     "symmetry": (), "k_point": (0.0, 0.0, 0.4)},
)


#: The interpreter of ``test_complex_chunk_forms.py`` executes the emitted text; the
#: regexes below read the emitted term table, wall mask and tail back out of it.
_TERM_CALL = re.compile(
    r"    cf term_(?P<comp>E[xyz])_(?P<offset>[01]) = offdiag_term\(\n"
    r"        (?P<volume>\w+), (?P<coefficient>\w+), idx,\n"
    r"        flat\((?P<down>[^)]*), nyz, nz\),\n"
    r"        flat\((?P<up>[^)]*), nyz, nz\),\n"
    r"        flat\((?P<corner>[^)]*), nyz, nz\),\n"
    r"        (?P<pa>\w+), (?P<dgl>\w+), (?P<dbc>\w+), (?P<w>\w+), (?P<pph>\w+), "
    r"(?P<pphase>\w+),\n"
    r"        (?P<oa>\w+), (?P<oph>\w+), (?P<ophase>\w+)\);")
_WALL = re.compile(r"    term_(?P<comp>E[xyz])_(?P<offset>[01]) = \((?P<flag>wm_[xyz]) && "
                   r"at_[xyz]\) \? cf_zero\(\) : term_(?P=comp)_(?P=offset);")
_TAIL_PML_CALL = re.compile(
    r"    constitutive_apply\((?P<comp>E[xyz]), f_w_(?P=comp), idx, src_(?P=comp), "
    r"kps_(?P<axis>[xyz])\[(?P<index>[ijk])\], kms_(?P=axis)\[(?P=index)\], "
    r"in_h_(?P=axis)\);")


def run_interpreted(launcher, monkeypatch, fields, layer, grid, text_edit=None,
                    connections=None, kept=None):
    """One launch of the family's own launcher with its compiled kernel replaced by the
    interpreter (``text_edit`` rewrites the emitted text first; ``connections`` and
    ``kept`` replace the derived arguments -- the defect doors)."""
    import sys  # noqa: PLC0415

    from .test_complex_chunk_forms import InterpretedKernel, _NumpyWearingCupysName  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)

    def kernel(arm, mask):
        text = family.kernel_source(arm, mask)
        if text_edit is not None:
            edited = text_edit(text)
            assert edited != text, "the planted defect matched nothing"
            text = edited
        return InterpretedKernel(text)

    monkeypatch.setattr(family, "_get_kernel", kernel)
    codes = family.complex_offdiag_boundary_codes(grid)
    record, derived = family.chunk_arguments(fields, layer, codes)
    overrides = dict(codes=codes, walls=coverage.offdiag_wall_mask_flags(grid),
                     weights=family.mirror_ghost_weights(grid),
                     connections=(family.bloch_connection_table(grid)
                                  if connections is None else connections),
                     chunk_bounds=numpy.asarray(record, dtype=numpy.int32),
                     kept=derived if kept is None else kept)
    if layer is not None and family.covers_complex_offdiag_pml_update_e is not None \
            and fields.f_w_Ex is not None:
        family.update_E_complex_offdiag_fused_pml(
            fields, tables=launcher.complex_constitutive_tables(layer, True),
            **overrides)
    else:
        family.update_E_complex_no_pml_offdiag(fields, **overrides)


_STATE = ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words_of(fields):
    return {name: numpy.array(getattr(fields, name), copy=True) for name in _STATE
            if getattr(fields, name, None) is not None}


def _differing(want, got):
    return {name: int(numpy.count_nonzero(
        numpy.ascontiguousarray(want[name]).view(numpy.uint32)
        != numpy.ascontiguousarray(got[name]).view(numpy.uint32))) for name in want}


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["label"])
def test_structure_matches_the_oracle(spec, launcher, monkeypatch):
    """One launch of the emitted kernel against ``stepping.update_E``, WORD FOR WORD.

    Every row mask of :data:`MASKS` and both courants, on the spec's grid. What it
    settles is the whole silent class -- a partner read at the wrong offset, a ghost
    taken in the wrong direction, a parity sign, a connection on the wrong axis or
    table, a mask on the wrong plane, a chunk form or a presence bit misread -- and
    the signed zeros the part-wise products decide.
    """
    scored = 0
    for mask in MASKS:
        for courant in (0.5, 0.35):
            fields, layer, grid = build(
                cell=spec["cell"], boundaries=spec["boundaries"],
                symmetry=spec["symmetry"], k_point=spec["k_point"],
                dimensions=spec.get("dimensions", 3), courant=courant,
                mask=mask, active=spec["active"], xp=numpy)
            start = _words_of(fields)
            stepping.update_E(fields, layer)
            want = _words_of(fields)
            for name, values in start.items():
                getattr(fields, name)[...] = values
            run_interpreted(launcher, monkeypatch, fields, layer, grid)
            got = _words_of(fields)
            diff = _differing(want, got)
            assert all(count == 0 for count in diff.values()), (
                spec["label"], mask, courant, {k: v for k, v in diff.items() if v})
            assert any(int(numpy.count_nonzero(
                numpy.ascontiguousarray(want[name]).view(numpy.uint32)
                != numpy.ascontiguousarray(start[name]).view(numpy.uint32))) > 0
                for name in ("Ex", "Ey", "Ez")), "the array path moved nothing"
            scored += 1
    print(f"{spec['label']}: {scored} configurations word for word against "
          f"stepping.update_E")


#: defect -> (spec label, row mask, text rewrite or None, argument override or None).
#: Each is a rewrite of the emitted text or of one launch argument, and each must
#: change a word on its fixture: the floor that makes the leg above a measurement.
_DEFECTS = {
    "down_index_on_the_own_coordinate": (
        "pml_fold_X_metallic_even", (1, 0, 0, 1, 0, 0),
        lambda text: text.replace("flat(i, dj, k, nyz, nz)", "flat(i, j, k, nyz, nz)"),
        None),
    # The previous package grouping of the tail (KERNELS.md section 7.1).
    "tail_as_two_accumulations": (
        "pml_unfolded_walls_phased", (1, 1, 1, 1, 1, 1),
        lambda text: text.replace(
            "cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), "
            "mul_coefficient_left(kms, prev)))",
            "cf_sub(cf_add(cf_load(f, idx), mul_coefficient_left(kps, src)), "
            "mul_coefficient_left(kms, prev))"),
        None),
    "row_summed_in_float32": (
        "pml_unfolded_walls_phased", (1, 1, 1, 1, 1, 1),
        lambda text: re.sub(r"double (row_E[xyz]_(?:re|im)) = \(double\)\(",
                            r"float \1 = (float)(", text)
        .replace("(0.25 * (double)(term_", "(0.25f * (term_"),
        None),
    "presence_select_removed": (
        "od_block3d", (1, 1, 1, 1, 1, 1), None, "kept_none"),
    "connection_tables_swapped": (
        "pml_unfolded_periodic_phased", (1, 1, 1, 1, 1, 1),
        None, "swap_connections"),
    "near_product_unconnected": (
        "pml_unfolded_periodic_phased", (1, 1, 1, 1, 1, 1),
        lambda text: text.replace(
            "cf near_term = cconnect(mul_field_left(near_pair, u[home]), oa, oph, ophase);",
            "cf near_term = mul_field_left(near_pair, u[home]);"),
        None),
    "mirror_weight_dropped": (
        "pml_fold_X_metallic_even", (1, 1, 1, 1, 1, 1),
        lambda text: text.replace(
            "    if (gl && bc == BC_MIRROR) return mul_coefficient_left(w, z);\n", ""),
        None),
    "wall_mask_on_the_real_word_only": (
        "store_metallic_walls", (1, 1, 1, 1, 1, 1),
        lambda text: re.sub(r"\? cf_zero\(\) : (term_E[xyz]_[01]);",
                            r"? real_zero(\1) : \1;", text)
        .replace("__device__ __forceinline__ cf cf_zero() {",
                 "__device__ __forceinline__ cf real_zero(cf a) {\n"
                 "    a.re = 0.0f;\n    return a;\n}\n\n"
                 "__device__ __forceinline__ cf cf_zero() {", 1), None),
}


@pytest.mark.parametrize("defect", sorted(_DEFECTS))
def test_each_planted_defect_changes_a_word(defect, launcher, monkeypatch):
    label, mask, edit, override = _DEFECTS[defect]
    if label == "od_block3d":
        # Rows on an interior block, so the chunks that carry a layer delete every
        # row's array (test_complex_chunk_forms' block case).
        from .test_complex_chunk_forms import build_offdiag  # noqa: PLC0415

        fields, grid, layer, _names = build_offdiag("od_block3d", True)
    else:
        spec = next(item for item in SPECS if item["label"] == label)
        fields, layer, grid = build(
            cell=spec["cell"], boundaries=spec["boundaries"], symmetry=spec["symmetry"],
            k_point=spec["k_point"], dimensions=spec.get("dimensions", 3), mask=mask,
            active=spec["active"], xp=numpy, seed=7)
    start = _words_of(fields)
    stepping.update_E(fields, layer)
    want = _words_of(fields)
    for name, values in start.items():
        getattr(fields, name)[...] = values
    kwargs = {}
    if override == "kept_none":
        from .offdiag_presence import ALL_CHUNKS_KEPT  # noqa: PLC0415

        kwargs["kept"] = (ALL_CHUNKS_KEPT,) * 6
    elif override == "swap_connections":
        import sys  # noqa: PLC0415

        monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels",
                            launcher)
        pair_kinds, pair, own_kinds, own = family.bloch_connection_table(grid)
        kwargs["connections"] = (own_kinds, own, pair_kinds, pair)
    run_interpreted(launcher, monkeypatch, fields, layer, grid, text_edit=edit, **kwargs)
    got = _words_of(fields)
    assert sum(_differing(want, got).values()) > 0, defect


# ---------------------------------------------------------------------------
# 2. THE TWO-KERNEL DECISION, MEASURED
# ---------------------------------------------------------------------------

def test_the_two_tails_are_different_computations():
    """The split-field tail cannot be degenerated into the direct store.

    The module's whole reason for shipping TWO kernels. Measured on the ARRAY PATH:
    the split-field recurrence with the identity coefficients ``kps = 1``,
    ``kms = 0`` computes ``E_old + constitutive``, not ``constitutive``, and
    ``update_E``'s input ``E`` is the field the previous step wrote.
    """
    fields, layer, grid = build(cell=(8.0, 8.0, 8.0), active=True)
    constitutive = numpy.asarray(fields.Dx, dtype=numpy.complex128) * numpy.asarray(
        fields.inverse_epsilon_for("Ex"), dtype=numpy.float64)
    before = numpy.asarray(fields.Ex, dtype=numpy.complex128).copy()
    ones = numpy.ones(grid.shape[0], dtype=numpy.float32)
    zeros = numpy.zeros(grid.shape[0], dtype=numpy.float32)
    recurrence = before + ones[:, None, None] * constitutive \
        - zeros[:, None, None] * numpy.asarray(fields.f_w_Ex,
                                               dtype=numpy.complex128)
    store = constitutive
    differing = int(numpy.count_nonzero(
        numpy.asarray(recurrence, dtype=numpy.complex64).view(numpy.float32)
        .view(numpy.uint32)
        != numpy.asarray(store, dtype=numpy.complex64).view(numpy.float32)
        .view(numpy.uint32)))
    total = int(numpy.asarray(store, dtype=numpy.complex64).view(
        numpy.float32).size)
    print(f"split-field tail at kps=1, kms=0 against the direct store: "
          f"{differing} of {total} float32 words differ")
    assert differing > 0.9 * total, (
        "the identity-coefficient recurrence agreed with the direct store on "
        f"{total - differing} of {total} words; if that were the whole volume the "
        f"two tails would be one kernel and this family would ship one")


def test_the_array_path_takes_two_different_branches():
    """``update_E`` accumulates where the chunk carries the own layer and overwrites
    elsewhere -- and without a layer overwrites everywhere.

    The reading behind the split, read off the ENGINE: the same frozen state stepped
    with an active layer and with an inert one leaves ``Ex`` different on the cells
    whose chunk carries the x layer (``stepping._apply_constitutive_regions``: the
    accumulation) and IDENTICAL everywhere else (``E = src`` in both runs), and the
    inert run equals the constitutive product exactly.
    """
    from ..chunk_map import chunk_bounds_record  # noqa: PLC0415

    # A ROW MASK WITH NO Ex SLOT, so Ex's constitutive value IS the bare diagonal
    # and the residual below is about the TAIL rather than about the coupling.
    mask = (0, 0, 1, 0, 0, 0)
    active_fields, layer, _ = build(cell=(8.0, 8.0, 8.0), active=True, seed=5,
                                    mask=mask, xp=numpy)
    inert_fields, _, _ = build(cell=(8.0, 8.0, 8.0), active=False, seed=5,
                               mask=mask, xp=numpy)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        assert numpy.array_equal(numpy.asarray(getattr(active_fields, name)),
                                 numpy.asarray(getattr(inert_fields, name))), name
    expected = (numpy.asarray(inert_fields.Dx, dtype=numpy.complex128)
                * numpy.asarray(inert_fields.inverse_epsilon_for("Ex"),
                                dtype=numpy.float64))
    stepping.update_E(active_fields, layer)
    stepping.update_E(inert_fields, None)
    record = chunk_bounds_record(active_fields.region_plan.layout)
    a0, a1, b0, b1 = (int(v) for v in record[12:16])  # the half-integer x record
    rows = numpy.arange(active_fields.Ex.shape[0])
    inside = ((rows >= a0) & (rows < a1)) | ((rows >= b0) & (rows < b1))
    moved = numpy.any(
        numpy.asarray(active_fields.Ex).view(numpy.uint32).reshape(
            active_fields.Ex.shape + (2,))
        != numpy.asarray(inert_fields.Ex).view(numpy.uint32).reshape(
            inert_fields.Ex.shape + (2,)), axis=-1)
    print(f"stepping.update_E active vs inert: {int(moved[inside].sum())} of "
          f"{int(moved[inside].size)} cells differ inside the x layer's chunks, "
          f"{int(moved[~inside].sum())} of {int(moved[~inside].size)} outside")
    assert inside.any() and (~inside).any()
    assert int(moved[~inside].sum()) == 0
    assert int(moved[inside].sum()) > 0.9 * moved[inside].size
    # And the inert branch really is the plain store, not a recurrence at unity.
    scale = float(numpy.max(numpy.abs(expected)))
    residual = float(numpy.max(numpy.abs(
        numpy.asarray(inert_fields.Ex, dtype=numpy.complex128) - expected)))
    print(f"inert branch against D * inv_eps: max residual {residual / scale:.3e} "
          f"relative")
    assert residual / scale < 1e-6


# ---------------------------------------------------------------------------
# 3. THE FLOAT32 HAZARDS, MEASURED ON THIS HOST
# ---------------------------------------------------------------------------

def test_the_ghost_coefficient_must_be_zero_not_the_wrapped_value():
    """``(+0.0f) * c`` is ``-0.0`` for negative ``c``, per part.

    The array path's ``_shift_up`` writes an exact ``(+0.0, +0.0)`` at a zero-ghost
    plane of the PRODUCT; the kernel reaches that only if the coefficient it
    multiplies the already-zero far pair by is itself ``+0.0f``. Measured here in
    plain float32 arithmetic, which is exact for these operands -- no rounding is
    involved, only the sign of a product with zero.
    """
    bad = good = 0
    for coefficient in (-0.75, -0.5, -1e-30, 0.0, 1e-30, 0.5, 0.75):
        real = numpy.float32(0.0) * numpy.float32(coefficient)
        if coefficient < 0.0:
            bad += int(numpy.signbit(real))
        else:
            good += int(not numpy.signbit(real))
    print(f"(+0) * c: {bad} of 3 negative coefficients produce -0.0, {good} of 4 "
          f"non-negative produce +0.0")
    assert bad == 3, ("if a negative coefficient did NOT produce -0.0 the ghosted "
                      "coefficient read would be unnecessary")
    for tail in TAILS:
        source = family.complex_offdiag_source(tail, MASKS[0])
        assert "cf far_term = mul_field_left(far_pair, ghosted(u, up));" in source, (
            "the emitted term no longer ghosts the up-plane coefficient")


def test_the_parity_multiply_is_the_part_wise_scale():
    """``parity * plane`` is ``part_products.scale``: one product per part.

    The mirror ghost's parity multiply (``host_writes.copy_face`` and the
    off-diagonal ``_shift_down``) scales each stored part by the parity. The full
    complex product against ``(parity + 0j)`` -- the previous family's form -- differs
    from it on signed zeros, measured here over the four signed-zero patterns.
    """
    from .. import part_products  # noqa: PLC0415

    patterns = [complex(re, im) for re in (0.0, -0.0) for im in (0.0, -0.0)]
    values = numpy.array(patterns, dtype=numpy.complex64)
    differing = 0
    for parity in (1, -1):
        full = numpy.complex64(parity) * values
        planewise = part_products.scale(numpy.float32(parity), values)
        differing += int(numpy.count_nonzero(
            full.view(numpy.float32).view(numpy.uint32)
            != planewise.view(numpy.float32).view(numpy.uint32)))
    total = 2 * values.view(numpy.float32).size
    print(f"parity * plane, full complex product against part-wise: {differing} "
          f"of {total} float32 words differ over the four signed-zero patterns")
    assert differing > 0
    for tail in TAILS:
        source = family.complex_offdiag_source(tail, MASKS[0])
        assert "if (gl && bc == BC_MIRROR) return mul_coefficient_left(w, z);" in source
        assert "    o.re = c * z.re;\n    o.im = c * z.im;" in source


def test_the_wall_mask_clears_both_words():
    """A clear that touched only the real plane is a known defect class here.

    The array path assigns the INTEGER ``0`` to the complex64 term, which is the pair
    ``(+0.0f, +0.0f)``. Measured: a real-only clear leaves the imaginary coupling
    alive on every masked cell.
    """
    values = numpy.array([1.5 + 2.5j, -0.25 - 3.0j], dtype=numpy.complex64)
    both = numpy.zeros_like(values)
    real_only = values.copy()
    real_only.real = numpy.float32(0.0)
    differing = int(numpy.count_nonzero(
        both.view(numpy.float32).view(numpy.uint32)
        != real_only.view(numpy.float32).view(numpy.uint32)))
    print(f"cf_zero() against a real-plane-only clear: {differing} of "
          f"{both.view(numpy.float32).size} float32 words differ")
    assert differing == values.size
    for tail in TAILS:
        source = family.complex_offdiag_source(tail, (1, 1, 1, 1, 1, 1))
        assert "? cf_zero() : term_" in source
        assert re.search(r"\? 0\.0f : term_", source) is None, (
            "the wall mask writes a bare float zero somewhere; complex64 is a "
            "word PAIR and a real-plane-only clear leaves the imaginary coupling "
            "alive on a metallic wall")


# ---------------------------------------------------------------------------
# 4. THE EMITTED TEXT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tail", TAILS)
def test_every_emitted_source_is_pure_ascii(tail):
    """A compile requirement, not a style rule: NVRTC writes the source through a
    bare ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE
    encoding. Two em-dashes killed a sibling kernel at its first launch."""
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        source = family.complex_offdiag_source(tail, mask)
        source.encode("ascii")  # raises if a non-ASCII byte survived
    print(f"{tail}: all {len(offdiag_emitter.LIVE_ROW_MASKS)} row masks "
          f"encode as ASCII")


@pytest.mark.parametrize("tail", TAILS)
def test_no_division_and_no_unary_minus_on_a_float_path(tail):
    """Both are platform requirements this track has adjudicated.

    Division is refused because the one PTX exception on record is ptxas expanding
    ``div.rn.f32`` into a sequence whose range checks carry ``.FTZ`` in SASS
    regardless of the PTX modifier. Unary minus is refused so a future edit cannot
    port the sibling Triton track's ``* -1.0`` reasoning backwards into a platform
    where it does not hold.
    """
    for mask in MASKS:
        source = family.complex_offdiag_source(tail, mask)
        code = "\n".join(line.split("//")[0] for line in source.splitlines())
        # Every division that survives must be INTEGER index arithmetic. The one
        # PTX exception on record is ptxas expanding div.rn.f32 into a sequence
        # whose range checks carry .FTZ in SASS whatever the PTX modifier says;
        # int division cannot reach it, float division must not appear.
        divisions = re.findall(r"[^\s;{}()]+\s*/\s*[^\s;{}()]+", code)
        assert set(divisions) <= {"idx / nz", "idx /", "(idx / nz) % ny",
                                  "idx / (ny"}, (tail, sorted(set(divisions)))
        assert re.search(r"[=(,]\s*-[A-Za-z_]", code) is None, (tail, mask)


@pytest.mark.parametrize("tail", TAILS)
def test_the_prelude_is_the_certified_siblings_bytes(tail):
    """``complex_emitter``'s head, part-wise products and tail are taken WHOLE, not
    respelled.

    The three multiply orientations, the word-pair addressing and the certified
    ``constitutive_apply`` are that module's bytes. Forking them to drop the dead
    helpers is the silent divergence ``no_pml_curl._sibling_prelude`` exists to
    prevent; carrying them turns the liability into evidence, because the gate arms
    a mutation of each as a MUST-BE-UNCAUGHT null.
    """
    source = family.complex_offdiag_source(tail, MASKS[0])
    for block in (complex_emitter._HEAD, complex_emitter._PART_PRODUCTS,
                  complex_emitter._TAIL):
        assert block in source, "a sibling prelude block was not carried verbatim"


@pytest.mark.parametrize("tail", TAILS)
def test_no_body_calls_the_dead_prelude_helpers(tail):
    """``cshift_up``/``cshift_dn``/``pml_apply`` are compiled in and never called.

    That is what makes carrying them safe, and it is a property of the build rather
    than of care: :func:`complex_offdiag_update_e._assert_body_is_clean` raises at
    emit. This asserts the same thing from outside, on every row mask.
    """
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        source = family.complex_offdiag_source(tail, mask)
        body = source[source.index('extern "C" __global__ void'):]
        for token in ("cshift_up(", "cshift_dn(", "pml_apply("):
            assert token not in body, (token, tail, mask)
    print(f"{tail}: no body of {len(offdiag_emitter.LIVE_ROW_MASKS)} row masks "
          f"calls a dead prelude helper")


def test_the_store_arm_carries_no_split_field_recurrence():
    """No auxiliary, no coefficient vector, no ``constitutive_apply``.

    The stored E IS the constitutive product without a layer (fields.py:1150), and
    an auxiliary in this arm would be a recurrence the array path does not run.
    """
    for mask in MASKS:
        source = family.complex_offdiag_source("no_pml", mask)
        body = source[source.index('extern "C" __global__ void'):]
        # COMMENTS ARE STRIPPED FIRST. The shared core's comments NAME the
        # split-field arm to say what does not belong here, and a token scan that
        # counted a comment would make the prose unwritable rather than the code
        # safe. The module's own emit-time check is regexed for the same reason.
        code = "\n".join(line.split("//")[0] for line in body.splitlines())
        for token in ("f_w_E", "kps_", "kms_", "constitutive_apply("):
            assert token not in code, (token, mask)
        assert code.count("cf_store(E") == 3
    print("the direct-store arm carries no f_w, no kps/kms and no "
          "constitutive_apply on any swept mask")


def test_the_pml_arm_carries_the_certified_tail():
    """Three ``constitutive_apply`` calls, each on its OWN axis's HALF-INTEGER pair.

    MEEP's ``dsigw``: component 0 takes x, 1 takes y, 2 takes z, the index is that
    axis's loop variable, and the per-chunk flag is that axis's half-integer record
    (``in_h_*``). Swapping the pair for the
    dsig/dsigu cycle the curl uses is a smooth, converged, entirely wrong absorber.
    """
    source = family.complex_offdiag_source("pml", MASKS[0])
    body = source[source.index('extern "C" __global__ void'):]
    seen = [(m["comp"], m["axis"], m["index"])
            for m in _TAIL_PML_CALL.finditer(body)]
    assert seen == [("Ex", "x", "i"), ("Ey", "y", "j"), ("Ez", "z", "k")], seen


@pytest.mark.parametrize("mask", offdiag_emitter.LIVE_ROW_MASKS[:16] +
                         offdiag_emitter.LIVE_ROW_MASKS[-1:])
def test_only_the_live_row_slots_reach_the_signature(mask):
    """A dead slot has NO parameter, so no pointer is bound to an array the kernel
    must never touch. ``offdiag_emitter``'s choice, for its reason."""
    for tail in TAILS:
        source = family.complex_offdiag_source(tail, mask)
        head = source.index('extern "C" __global__ void')
        signature = source[head:source.index(") {", head)]
        for slot, name in enumerate(family.ROW_PARAMETERS):
            assert (f"const float* {name}," in signature) == bool(mask[slot]), (
                name, mask[slot])


def test_the_term_table_is_the_transcribed_partner_cycle():
    """Every emitted term reads the partner MEEP's ``cycle_direction`` names.

    ``offset`` 0 is ``cycle_direction(dim, d_ec, 1)`` and 1 is ``(..., 2)``, and the
    partner axis, the coefficient slot, the partner volume, the ghost lane, the
    boundary code, the parity weight and both connections (the pair's on the partner
    axis, the near product's on the own axis) all follow from that one number. This
    re-derives the whole table from ``coverage``'s own tables and asserts the emitted
    text carries exactly it.
    """
    source = family.complex_offdiag_source("pml", (1, 1, 1, 1, 1, 1))
    body = source[source.index('extern "C" __global__ void'):]
    calls = {(m["comp"], int(m["offset"])): m.groupdict()
             for m in _TERM_CALL.finditer(body)}
    assert len(calls) == 6, sorted(calls)
    home = ["i", "j", "k"]
    for component, (name, _source_name, own_axis) in enumerate(family.E_TERMS):
        for offset in (0, 1):
            partner = coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
            call = calls[(name, offset)]
            letter, own = "xyz"[partner], "xyz"[own_axis]
            assert call["volume"] == family.PARTNER_VOLUMES[partner]
            assert call["coefficient"] == family.ROW_PARAMETERS[2 * component + offset]
            assert call["pa"] == home[partner]
            assert call["dgl"] == f"at_{letter}"
            assert call["dbc"] == f"bc_{letter}"
            assert call["w"] == f"gw_{letter}"
            assert call["pph"] == f"ph_{letter}"
            assert call["pphase"] == f"pp{letter}"
            assert call["oa"] == home[own_axis]
            assert call["oph"] == f"oh_{own}"
            assert call["ophase"] == f"op{own}"
            # The three shifted index expressions: DOWN on the partner axis, UP on
            # the own axis, and the corner carrying both.
            down = list(home)
            down[partner] = "d" + home[partner]
            up = list(home)
            up[own_axis] = "u" + home[own_axis]
            corner = list(up)
            corner[partner] = "d" + home[partner]
            assert call["down"] == ", ".join(down), (name, offset)
            assert call["up"] == ", ".join(up), (name, offset)
            assert call["corner"] == ", ".join(corner), (name, offset)


def test_the_wall_mask_axes_are_the_transcribed_ones():
    """Face 0 of every axis whose Yee shift is 0, per term, in the mask's loop order."""
    source = family.complex_offdiag_source("pml", (1, 1, 1, 1, 1, 1))
    body = source[source.index('extern "C" __global__ void'):]
    seen = {}
    for match in _WALL.finditer(body):
        seen.setdefault((match["comp"], int(match["offset"])), []).append(match["flag"])
    for component, (name, _s, _a) in enumerate(family.E_TERMS):
        expected = [f"wm_{'xyz'[axis]}"
                    for axis in coverage.OFFDIAG_WALL_MASK_AXES[component]]
        for offset in (0, 1):
            assert seen.get((name, offset), []) == expected, (name, offset, seen)


def test_an_all_dead_row_mask_is_refused():
    """That configuration belongs to the element-wise complex constitutive family,
    and emitting this family's source for it would overlap the two."""
    with pytest.raises(ValueError, match="no row slot survives"):
        family.complex_offdiag_source("pml", (0, 0, 0, 0, 0, 0))


def test_an_unknown_tail_is_refused_not_defaulted():
    """A wrong tail is a WRONG ANSWER, not a crash."""
    with pytest.raises(ValueError, match="arm must be one of"):
        family.complex_offdiag_source("split_field", MASKS[0])


# ---------------------------------------------------------------------------
# 5. THE CONSTANTS, PINNED AGAINST THEIR ONE HOME
# ---------------------------------------------------------------------------

def test_mirror_source_index_equals_steppings():
    assert family.MIRROR_SOURCE_INDEX == int(stepping.MIRROR_SOURCE_INDEX)
    assert family.MIRROR_SOURCE_INDEX == folded.MIRROR_SOURCE_INDEX
    assert "#define MIRROR_ROW 2" in family.complex_offdiag_source(
        "pml", MASKS[0])


def test_boundary_codes_equal_the_real_folded_families():
    """One renumbering must not happen on one track only."""
    assert family.FOLDED_BC_CODES == folded.FOLDED_BC_CODES
    assert family.BC_MIRROR_CODE == folded.BC_MIRROR_CODE
    source = family.complex_offdiag_source("pml", MASKS[0])
    for name, code in family.FOLDED_BC_CODES.items():
        assert f"#define BC_{name.upper()} {code}" in source


def test_compile_options_equal_every_siblings():
    from . import complex_no_pml_kernels  # noqa: PLC0415

    assert family._COMPILE_OPTIONS == ('--fmad=false',)
    assert family._COMPILE_OPTIONS == complex_no_pml_kernels._COMPILE_OPTIONS
    assert family._COMPILE_OPTIONS == folded._COMPILE_OPTIONS
    # The cupy-importing siblings are read off their TEXT, because this host has
    # no CuPy and "the tuple is the same" is a claim about the bytes a gate
    # compiled, not about an object this process can hold.
    for name in ("offdiag_constitutive_kernels", "complex_pml_kernels",
                 "constitutive_kernels"):
        text = (HERE / f"{name}.py").read_text(encoding="utf-8")
        found = re.findall(r"^_COMPILE_OPTIONS[^=]*= (.+)$", text, re.M)
        assert found, name
        assert all("'--fmad=false'" in line or '"--fmad=false"' in line
                   for line in found), (name, found)


def test_the_row_tables_are_the_certified_emitters():
    assert family.E_TERMS is offdiag_emitter.E_TERMS
    assert family.PARTNER_VOLUMES is offdiag_emitter.PARTNER_VOLUMES
    assert family.ROW_PARAMETERS is offdiag_emitter.ROW_PARAMETERS
    assert family.LIVE_ROW_MASKS is offdiag_emitter.LIVE_ROW_MASKS
    assert len(family.LIVE_ROW_MASKS) == 63


# ---------------------------------------------------------------------------
# 6. THE PHASE TABLE
# ---------------------------------------------------------------------------

def test_bloch_connection_table_takes_eikna_on_the_pair_and_its_conjugate_on_the_near_product(
        launcher, monkeypatch):
    """The pair's same-index operand reads the word ABOVE the stored window
    (``eikna``) and the near product the word BELOW it (``conj(eikna)``).

    Taking the same factor for both is the classic sign error: every magnitude stays
    plausible and only the phase moves, which is what a band structure is made of.
    """
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    _fields, _layer, grid = build(k_point=(0.23, -0.17, 0.35),
                                  boundaries=("periodic",) * 3, active=False)
    pair_kinds, pair, own_kinds, own = family.bloch_connection_table(grid)
    assert pair_kinds == own_kinds == (complex_emitter.CONNECT_PHASE,) * 3
    for axis in range(3):
        rounded = numpy.complex64(grid.bloch_phase(axis))
        assert pair[2 * axis] == own[2 * axis] == float(numpy.float32(rounded.real))
        assert pair[2 * axis + 1] == float(numpy.float32(rounded.imag))
        assert own[2 * axis + 1] == -pair[2 * axis + 1] != 0.0


def test_bloch_connection_table_skips_the_connection_at_k_zero(launcher, monkeypatch):
    """No phase means no connection at all -- kind ``CONNECT_NONE`` on both tables,
    never a multiply against ``1 + 0j``."""
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    _fields, _layer, grid = build(k_point=(0.0, 0.0, 0.0), active=False)
    pair_kinds, pair, own_kinds, own = family.bloch_connection_table(grid)
    assert pair_kinds == own_kinds == (complex_emitter.CONNECT_NONE,) * 3
    assert pair == own == (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)


def test_bloch_connection_table_refuses_a_phase_on_a_non_periodic_axis(launcher,
                                                                      monkeypatch):
    """``stepping._bloch_phases`` raises on the pairing; a caller who skipped the
    predicate is refused loudly rather than handed a mis-bound kind."""
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)

    class _PhasedMetallic:
        def __init__(self, grid):
            object.__setattr__(self, "_grid", grid)

        has_bloch = True

        def bloch_phase(self, axis):  # noqa: ARG002
            return complex(0.3, 0.4)

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_grid"), item)

    _fields, _layer, grid = build(boundaries=("metallic",) * 3, active=False,
                                  k_point=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="only a periodic wrap can carry a phase"):
        family.bloch_connection_table(_PhasedMetallic(grid))


def test_mirror_ghost_weights_are_the_real_familys():
    """One derivation of the parity, through ``fields.mirror_parity`` itself."""
    _fields, _layer, grid = build(
        cell=(16.0, 16.0, 12.0), boundaries=("metallic", "periodic", "periodic"),
        symmetry=(("X", 1), ("Y", -1)), active=True)
    assert family.mirror_ghost_weights(grid) == folded.mirror_ghost_weights(grid)
    weights = family.mirror_ghost_weights(grid)
    assert weights[0] == -1.0 and weights[1] == 1.0 and weights[2] == 1.0, weights


# ---------------------------------------------------------------------------
# 7. THE PREDICATES
# ---------------------------------------------------------------------------

def verdicts(fields, layer, grid):
    return (family.covers_complex_offdiag_pml_update_e(
                fields, layer, grid),
            family.covers_complex_no_pml_offdiag_update_e(
                fields, layer, grid))


def test_the_five_corpus_shapes_are_admitted():
    """The rows this family exists for, rebuilt from the census's own facts."""
    rows = (
        ("solve-cw / array_metadata", dict(
            cell=(16.0, 16.0, 12.0),
            boundaries=("metallic", "metallic", "periodic"),
            symmetry=(("X", 1), ("Y", 1)), k_point=(0.0, 0.0, 0.0), active=True),
         "pml"),
        ("holey_wvg_bands (k on an unfolded axis)", dict(
            cell=(10.0, 16.0, 12.0), boundaries=("periodic",) * 3,
            symmetry=(("Y", 1),), k_point=(0.35, 0.0, 0.0), active=True),
         "pml"),
        ("matgrid_3d", dict(
            cell=(12.0, 12.0, 12.0), boundaries=("periodic",) * 3,
            k_point=(0.23, -0.17, 0.35), active=False), "no_pml"),
        ("subpixel_smoothing", dict(
            cell=(12.0, 12.0, 0.0), dimensions=2, boundaries=("periodic",) * 3,
            k_point=(0.3892, 0.1597, 0.0), active=False), "no_pml"),
    )
    for label, kwargs, expected in rows:
        fields, layer, grid = build(**kwargs)
        pml_verdict, store_verdict = verdicts(fields, layer, grid)
        chosen = "pml" if pml_verdict[0] else "no_pml" if store_verdict[0] else None
        assert chosen == expected, (label, pml_verdict, store_verdict)
        assert not (pml_verdict[0] and store_verdict[0]), label
    print(f"all {len(rows)} corpus shapes admitted, by the expected arm, with the "
          f"two arms disjoint on every one")


def test_the_arms_are_disjoint_on_every_shape():
    """At most one tail may ever admit a run -- the ``_pml_is_active`` switch the
    array path itself branches on."""
    for spec in SPECS:
        fields, layer, grid = build(
            cell=spec["cell"], boundaries=spec["boundaries"],
            symmetry=spec["symmetry"], k_point=spec["k_point"],
            dimensions=spec.get("dimensions", 3), active=spec["active"])
        pml_verdict, store_verdict = verdicts(fields, layer, grid)
        assert not (pml_verdict[0] and store_verdict[0]), spec["label"]
        assert (pml_verdict[0] or store_verdict[0]), (spec["label"], pml_verdict,
                                                      store_verdict)


def test_the_partition_against_every_shipped_sibling():
    """No slot may be admitted by this family AND by a shipped sibling at once.

    Two admitters leave a slot unselected and it falls back to the array path: a
    silent coverage LOSS, not an error.
    """
    from . import complex_no_pml_kernels, complex_folded_kernels  # noqa: PLC0415

    for spec in SPECS:
        fields, layer, grid = build(
            cell=spec["cell"], boundaries=spec["boundaries"],
            symmetry=spec["symmetry"], k_point=spec["k_point"],
            dimensions=spec.get("dimensions", 3), active=spec["active"])
        pml_verdict, store_verdict = verdicts(fields, layer, grid)
        assert pml_verdict[0] or store_verdict[0], spec["label"]
        others = {
            "complex_constitutive_E": coverage.covers_real_pml_complex_constitutive(
                fields, layer, grid, "E"),
            "complex_folded_constitutive_E":
                complex_folded_kernels.covers_complex_folded_constitutive(
                    fields, layer, grid, "E"),
            "complex_no_pml_stored_e":
                complex_no_pml_kernels.covers_complex_no_pml_stored_e(
                    fields, layer, grid),
            "real_offdiag": coverage.covers_real_pml_offdiag_constitutive(
                fields, layer, grid),
            "folded_offdiag":
                folded.covers_folded_offdiag_composition(fields, layer, grid),
        }
        for name, (covered, reason) in others.items():
            assert not covered, (spec["label"], name, reason)


def test_real_storage_is_refused_by_both_arms():
    fields, layer, grid = build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    pml_verdict, store_verdict = verdicts(fields, layer, grid)
    assert not pml_verdict[0] and "real float32 storage" in pml_verdict[1]
    assert not store_verdict[0]


def test_a_diagonal_run_is_refused_by_both_arms():
    """The element-wise families own that configuration and this must not overlap."""
    fields, layer, grid = build(mask=(0, 0, 0, 0, 0, 0))
    pml_verdict, _store = verdicts(fields, layer, grid)
    assert not pml_verdict[0]
    assert "no off-diagonal chi1inv row survived" in pml_verdict[1]


def test_dispersion_and_nonlinearity_are_refused():
    """``update_E``'s source here is D and never ``(D - sum P)``; a Pade factor
    would scale the whole row product."""
    from ..dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    fields, layer, grid = build()
    fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.1, gamma=0.05),
        XP.full(grid.shape, numpy.float32(0.3)), grid, fields._field_dtype()))
    pml_verdict, _store = verdicts(fields, layer, grid)
    assert not pml_verdict[0] and "susceptibility" in pml_verdict[1]


def test_a_thin_folded_axis_is_refused():
    """``coord_dn``'s mirror arm returns stored row 2 unconditionally, and
    ``stepping._mirror_source`` RAISES below that many cells."""

    class _ThinFold:
        def __init__(self, grid):
            object.__setattr__(self, "_grid", grid)

        @property
        def shape(self):
            return (2,) + tuple(object.__getattribute__(self, "_grid").shape[1:])

        def is_mirrored(self, axis):
            return axis == 0

        def mirror_phase(self, axis):  # noqa: ARG002
            return 1

        def has_symmetry(self):
            return True

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_grid"), item)

    fields, layer, grid = build(
        cell=(16.0, 10.0, 12.0), boundaries=("metallic", "periodic", "periodic"),
        symmetry=(("X", 1),), active=True)
    covered, reason = family.covers_complex_offdiag_pml_update_e(
        fields, layer, _ThinFold(grid))
    assert not covered
    assert "stored cells" in reason or "shape" in reason, reason


def test_an_allocated_auxiliary_is_refused_by_the_store_arm():
    """An allocated ``f_w`` under an inert layer means something else built it and
    this step would leave it stale."""
    fields, _layer, grid = build(active=False)
    fields.enable_pml_storage()
    covered, reason = family.covers_complex_no_pml_offdiag_update_e(
        fields, None, grid)
    assert not covered
    assert "PML storage mode" in reason or "f_w_" in reason, reason


def test_a_cylindrical_grid_is_refused_by_name():
    """``admit_cylindrical=True`` is passed to the shared clause set only to admit
    the MIRROR boundary kind; Dcyl itself is refused HERE, first, with this
    family's own reason."""

    class _Cylindrical:
        def __init__(self, grid):
            object.__setattr__(self, "_grid", grid)

        @property
        def cylindrical(self):
            """``coverage._grid_facts`` reads ``grid.cylindrical``, so that is what
            a stand-in has to answer; a shim that answered a different spelling
            would test the shim."""
            return True

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_grid"), item)

    fields, layer, grid = build()
    covered, reason = family.covers_complex_offdiag_pml_update_e(
        fields, layer, _Cylindrical(grid))
    assert not covered
    assert "cylindrical" in reason, reason


# ---------------------------------------------------------------------------
# 8. THE LAUNCH-ARGUMENT VALIDATION
# ---------------------------------------------------------------------------

def test_launch_validation_refuses_the_four_silent_defects():
    """A NaN parity, a wall-masked fold, a thin fold, a phase off a periodic axis.

    Each is a WRONG ANSWER rather than a crash if it is not checked, the predicates
    name all four, and a gate can reach a launcher without the predicate -- which is
    why they are checked twice.
    """
    good = dict(shape=(8, 8, 8), codes=(2, 0, 0), walls=(0, 0, 0),
                weights=(-1.0, 1.0, 1.0), flags=(0, 0, 0))
    family._validate_launch_arguments(**good)  # the control: this one passes
    with pytest.raises(ValueError, match="ghost weight"):
        family._validate_launch_arguments(**dict(good, weights=(float("nan"),
                                                                1.0, 1.0)))
    with pytest.raises(ValueError, match="folded AND wall-masked"):
        family._validate_launch_arguments(**dict(good, walls=(1, 0, 0)))
    with pytest.raises(ValueError, match="stored cells"):
        family._validate_launch_arguments(**dict(good, shape=(2, 8, 8)))
    with pytest.raises(ValueError, match="only a periodic wrap can carry a phase"):
        family._validate_launch_arguments(**dict(good, flags=(1, 0, 0)))
    with pytest.raises(ValueError, match="boundary codes must be three of"):
        family._validate_launch_arguments(**dict(good, codes=(3, 0, 0)))


def test_the_launchers_refuse_two_answers_to_one_question():
    """Passing ``pml`` together with an override, or neither, is a bug either way."""
    fields, layer, _grid = build()
    with pytest.raises(ValueError, match="both supplied"):
        family.update_E_complex_offdiag_fused_pml(
            fields, layer, codes=(0, 0, 0))
    with pytest.raises(ValueError, match="pass exactly one of pml"):
        family.update_E_complex_offdiag_fused_pml(fields)
    with pytest.raises(ValueError, match="pass exactly one of pml"):
        family.update_E_complex_no_pml_offdiag(fields)


# ---------------------------------------------------------------------------
# 9. THE RECORD
# ---------------------------------------------------------------------------

def test_admission_record_is_all_or_nothing():
    """A ``host: None`` record may not carry device numbers, and a filled one may
    not omit them. The two states must not blur."""
    record = family.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION
    device_keys = ("host", "artifacts", "policies")
    filled = [key for key in device_keys if record.get(key) is not None]
    assert filled in ([], list(device_keys)), filled
    if record.get("host") is None:
        assert record.get("certified") is False
        assert not family.CERTIFIED_KERNELS, (
            "a kernel is listed as certified while the admission record carries "
            "no host; a predicate returning True licenses a MEASUREMENT and "
            "nothing else")


def test_every_shipped_kernel_is_described_by_exactly_one_set():
    shipped = family.shipped_kernel_names()
    assert shipped == set(family.KERNEL_NAMES.values()), shipped
    described = set(family.CERTIFIED_KERNELS) | set(family.UNCERTIFIED_KERNELS)
    assert shipped == described, shipped ^ described
    assert not (set(family.CERTIFIED_KERNELS)
                & set(family.UNCERTIFIED_KERNELS))


def test_no_kernel_wears_a_certified_siblings_name():
    """A second body under a certified name would make ``certification.json``'s
    partition test and any NVRTC binary observation ambiguous."""
    import json  # noqa: PLC0415

    record = json.loads((HERE / "certification.json").read_text(encoding="utf-8"))
    # THIS FAMILY'S OWN BLOCK IS NOT A COLLISION. The clause guards against a SECOND
    # BODY wearing a name a DIFFERENT certified family already owns; while this family
    # was uncertified, "appears nowhere in the record" and "is owned by nobody else"
    # were the same set, and the cheap spelling was exact. Since
    # complex_offdiag_update_e_2026-08-21 landed, its own certified_kernels list names
    # these two legitimately, so the sweep skips that one block and still reads every
    # other. Widening to "skip any block naming them" would be the wrong fix: it would
    # excuse precisely the sibling collision this exists to catch.
    record = {key: value for key, value in record.items()
              if key != "complex_offdiag_update_e_2026-08-21"}
    taken = set()
    stack = [record]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            taken |= {key for key in node if isinstance(key, str)}
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, str):
            taken.add(node)
    for name in family.KERNEL_NAMES.values():
        assert name not in taken, name
    assert set(family.KERNEL_NAMES.values()).isdisjoint(
        {complex_emitter.KERNELS[key][0] for key in complex_emitter.KERNELS})
    assert folded.KERNEL_NAME not in family.KERNEL_NAMES.values()
    assert offdiag_emitter.KERNEL_NAME not in family.KERNEL_NAMES.values()


def test_the_corpus_digest_moves_on_one_changed_character():
    """252 sources hash to one value, and it is worthless if it does not move."""
    baseline = family.corpus_digest()
    assert re.fullmatch(r"[0-9a-f]{64}", baseline)
    original = family._OWN_PRELUDE
    try:
        family._OWN_PRELUDE = original.replace("#define MIRROR_ROW 2",
                                               "#define MIRROR_ROW  2")
        moved = family.corpus_digest()
    finally:
        family._OWN_PRELUDE = original
    assert moved != baseline
    assert family.corpus_digest() == baseline, "the digest did not restore"
    print(f"corpus digest over {2 * 63 * 2} sources: {baseline}")


def test_the_module_source_is_pure_ascii():
    MODULE_PATH.read_text(encoding="utf-8").encode("ascii")


def test_the_module_does_not_import_cupy_at_scope():
    """The predicates are consumed by the coverage census and the census runs on a
    laptop."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("import cupy") or line.startswith("from cupy"):
            raise AssertionError(f"cupy imported at module scope: {line!r}")
    assert "import cupy as cp  # noqa: PLC0415" in text


def test_source_digests_are_stable_across_calls():
    """A gate records per-source digests; they have to be a function of the inputs
    alone."""
    for tail in TAILS:
        first = hashlib.sha256(family.complex_offdiag_source(
            tail, MASKS[1]).encode("utf-8")).hexdigest()
        second = hashlib.sha256(family.complex_offdiag_source(
            tail, MASKS[1]).encode("utf-8")).hexdigest()
        assert first == second


def test_the_admission_record_is_bound_to_the_emitter_it_describes():
    """``bound_by`` is a promise and this is the check behind it.

    The record cannot bind itself with an artifact hash -- both artifacts carry
    THIS FILE's bytes in ``imported_source_sha256``, so writing an artifact digest
    into the module would be circular the moment the block was added. The
    non-circular binding is the EMITTER CORPUS DIGEST: one sha256 over all 252
    sources this family can emit, carried in both artifacts and asserted equal to
    the shipped emitter here. A docstring edit moves the file hash and not the
    arithmetic; a changed device string moves this.
    """
    record = family.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION
    if record.get("host") is None:
        pytest.skip("no device leg has run; the digest clause has nothing to bind")
    assert record["emitter_corpus_digest"] == family.corpus_digest(), (
        "the shipped emitter no longer produces the digest the admission record "
        "was cut at: the record describes device code this module no longer emits")


def test_the_named_artifacts_carry_the_same_digest_and_verdict():
    """The record's numbers are read back off the artifacts it names.

    A record whose counts were typed rather than transcribed is a record; these are
    read from the JSON the device wrote, per policy, so a copied-in number that
    drifted from the run fails here.
    """
    import json  # noqa: PLC0415

    record = family.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION
    if record.get("host") is None:
        pytest.skip("no device leg has run")
    root = HERE.parent.parent
    for relative in record["artifacts"]:
        path = root / relative
        assert path.exists(), f"the record names an artifact that is not here: {relative}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["emitter_corpus_digest"] == record["emitter_corpus_digest"]
        assert payload["verdict"]["release"] is True, relative
        assert payload["sweep"]["summary"]["cases"] == record["cases_scored"]
        assert payload["sweep"]["summary"]["refused_as_vacuous"] == (
            record["cases_refused_as_vacuous"])
        assert payload["mutations"]["summary"]["escaped"] == []
        assert payload["mutations"]["summary"]["false_positives"] == []
        assert payload["mutations"]["summary"]["unarmed"] == 0
        assert payload["falsification"]["all_flip"] is True
        # THE GUARD FINDING, read back rather than restated: this family's control
        # did NOT diverge and the record says so, which is the opposite of every
        # certified sibling's clause and must not be quietly inherited.
        assert payload["guard_binary_probe"]["licensed_arm_ptx_distinct"] is (
            record["guard_ptx_distinct_on_the_licensed_arm"])
        assert payload["guard_binary_probe"]["naive_arm_ptx_distinct"] is (
            record["guard_ptx_distinct_on_the_naive_arm"])
    policies = {json.loads((root / relative).read_text(encoding="utf-8"))[
        "subnormal_policy"]["policy"] for relative in record["artifacts"]}
    assert policies == set(record["policies"]), policies
