"""The MIRROR FOLD under complex64 storage: its deltas, its predicate, its behaviour.

WHAT IS PINNED HERE AND WHAT IS NOT. The BYTE-IDENTITY VERDICT needs a device and
lives in ``parity/meep_gpu/gate_cuda_complex_folded.py``. What this file pins is
the half of the family's silent-failure surface a laptop can reach, and every one
of these failures is a plausible, smooth, WRONG field rather than a crash:

* THE DELTAS. The device source is ``complex_emitter``'s transformed, so the whole
  claim "no arithmetic change at all" is checkable by diff -- and is checked, line
  by line, against the certified source. A delta whose anchor has moved must FAIL
  here, on a laptop, rather than emit a kernel with one fold branch missing.
* THE MASK BLOCK, character for character against the REAL curl pair's, which is
  where the fold's device verdict already lives. A mask is the whole of what this
  family adds; a transcription of it that is "equivalent" rather than equal is a
  second spelling of the one rule the real gate measured.
* THE PREDICATE, whose admissions must not overlap the certified complex pair's --
  two families on one slot is a widening the union census reports as a FINDING --
  and whose refusals (a phase on a folded axis, an undecidable fold termination)
  are configurations the array path does not express.
* THE BEHAVIOUR. The transcribed device tree is stepped against ``stepping.step_B``
  / ``step_D`` on real FOLDED complex grids and compared as uint32 words, with the
  two fold masks each shown to MATTER: dropped, the answer diverges. A test that
  only checks verdict strings cannot tell a working kernel from one that returns
  its input.

WHAT THE NUMPY LEG CANNOT SEE, stated rather than papered over: it compiles
nothing, so it cannot see an NVRTC contraction, a lost signed zero, or the
expansion arm -- and on NumPy the complex multiply is bitwise commutative
(``ar*br - ai*bi`` against ``br*ar - bi*ai``, and ``x + y`` against ``y + x``), so
the operand ORIENTATION that ``FMA_V1`` makes normative is invisible here. Those
are the device gate's questions. What this leg settles is the STRUCTURE: which
ghost each face takes, which planes the masks drop, and that the fold's parity
ghost never reaches an output word.
"""

from __future__ import annotations

import pathlib
import re

import numpy
import pytest

from .. import stepping
from ..fields import Fields, IYEE_SHIFTS
from ..grid import Grid, Mirror
from ..pml import PML
from . import complex_emitter, complex_folded_kernels as folded, coverage

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

HERE = pathlib.Path(__file__).resolve().parent

#: A licence shaped exactly as ``expansion_license`` returns one, so the
#: predicate's licence clause is satisfied and the rest of it can be asked. It is
#: NOT a claim about any platform: no kernel is compiled in this file.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "basis": "measured",
           "refusals": [], "policy_resolved": "keep"}
POLICY = "keep"


# ---------------------------------------------------------------------------
# Fixtures, shared with test_complex_beta.py
# ---------------------------------------------------------------------------

def build(cell, boundaries, planes, k_point=(0.0, 0.0, 0.0), courant=0.5,
          beta=0.0, dimensions=None):
    """A ``(fields, layer, grid)`` triple with COMPLEX storage and a live PML.

    ``force_complex_fields=True`` always, so a k = 0 fold is a real fixture rather
    than an unreachable branch -- ``solve-cw.py`` and ``TestArrayMetadata`` are
    exactly that class. The thickness rule is every certified slice's: skip an axis
    too thin to hold a layer, and put the layer on the FAR face only of a folded
    axis, whose near face is the mirror plane.
    """
    kwargs = {}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    if beta:
        kwargs["beta"] = beta
    grid = Grid(resolution=1.0, cell_size=tuple(cell), boundaries=tuple(boundaries),
                symmetry=tuple(planes), xp=numpy, courant=courant,
                k_point=tuple(k_point), **kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


CURL_ARRAYS = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "sources": ("Ex", "Ey", "Ez"), "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "sources": ("Hx", "Hy", "Hz"), "half_integer": False,
               "backward": True},
}

#: ``(target, aux, first, first_axis, second, second_axis, dsig, dsigu)`` -- the
#: six curl terms as the EMITTED KERNEL spells them (``g2``/``g1`` for target 0 and
#: so on), transcribed from the device text rather than derived from ``stepping``:
#: deriving it from the oracle would make the behaviour leg compare the oracle with
#: itself. :func:`test_the_transcribed_term_table_is_steppings` pins it.
CURL_TERMS = {
    "step_B": (
        ("Bx", "fu_Bx", "Ez", 1, "Ey", 2, 1, 2),
        ("By", "fu_By", "Ex", 2, "Ez", 0, 2, 0),
        ("Bz", "fu_Bz", "Ey", 0, "Ex", 1, 0, 1),
    ),
    "step_D": (
        ("Dx", "fu_Dx", "Hz", 1, "Hy", 2, 1, 2),
        ("Dy", "fu_Dy", "Hx", 2, "Hz", 0, 2, 0),
        ("Dz", "fu_Dz", "Hy", 0, "Hx", 1, 0, 1),
    ),
}


def seed(fields, grid, names, rng):
    """Physical-band complex values in every array the sub-step touches.

    THE AUXILIARIES START NONZERO: a zero ``fu`` makes ``kms * prev`` exactly zero
    on the first launch whatever ``kms`` holds, which would hide a mis-indexed
    coefficient until step two.
    """
    for name in names:
        real = rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
        imag = rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
        getattr(fields, name)[...] = (real + 1j * imag).astype(numpy.complex64)


def snapshot(fields, names):
    return {name: numpy.array(getattr(fields, name), copy=True) for name in names}


def restore(fields, frozen):
    for name, values in frozen.items():
        getattr(fields, name)[...] = values


def words(array):
    """One complex64 volume as its float32 WORD PAIRS, viewed as uint32.

    The comparison unit everywhere in this family: a signed zero and a NaN payload
    are both invisible to ``==`` on floats and both decide bit-identity.
    """
    return numpy.ascontiguousarray(array).view(numpy.float32).ravel().view(
        numpy.uint32)


def differing_words(left, right, names):
    return sum(int(numpy.count_nonzero(words(left[name]) != words(right[name])))
               for name in names)


def boundary_codes_for(grid):
    """The SHIPPED code triple, from the function the predicate itself consults."""
    codes, refusal = folded.folded_complex_boundary_codes(grid)
    assert refusal is None, refusal
    return tuple(int(code) for code in codes)


def tables_for(sub_step, layer):
    suffix = "_h" if CURL_ARRAYS[sub_step]["half_integer"] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def phase_arguments(grid, backward):
    """``(flags, complex64 phases)`` -- ``stepping._bloch_phases``, transcribed.

    A phase of None is flag 0 and NO multiply, never a multiply against ``1+0j``:
    that skip is the bit-identity of k = 0. The BACKWARD sub-step takes the
    CONJUGATE (stepping.py:1865-1869), and negation is exact so the order of
    conjugation and rounding is immaterial.
    """
    flags, values = [], []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            flags.append(0)
            values.append(numpy.complex64(1.0))
            continue
        rounded = numpy.complex64(phase)
        flags.append(1)
        values.append(numpy.complex64(rounded.conjugate() if backward else rounded))
    return tuple(flags), tuple(values)


def _plane(axis, index, shape):
    key = [slice(None)] * len(shape)
    key[axis] = index
    return tuple(key)


def _cshift_up(field, axis, bc, flag, phase):
    """``cshift_up`` from the emitted text: wrap, phase the wrapped lane, else zero.

    ``BC_MIRROR_PERIODIC`` takes the metallic zero -- the fold delta -- and the
    zero is ``(+0.0f, +0.0f)``, which is what an integer 0 assigned into a complex64
    array is.
    """
    shifted = numpy.roll(field, -1, axis=axis)
    if bc == BC_PERIODIC:
        if flag:
            shifted[_plane(axis, -1, field.shape)] = (
                shifted[_plane(axis, -1, field.shape)] * phase)
        return shifted
    shifted[_plane(axis, -1, field.shape)] = 0
    return shifted


def _cshift_dn(field, axis, bc, flag, phase):
    """``cshift_dn`` from the emitted text -- the mirror of :func:`_cshift_up`."""
    shifted = numpy.roll(field, 1, axis=axis)
    if bc == BC_PERIODIC:
        if flag:
            shifted[_plane(axis, 0, field.shape)] = (
                shifted[_plane(axis, 0, field.shape)] * phase)
        return shifted
    shifted[_plane(axis, 0, field.shape)] = 0
    return shifted


def _broadcast(vector, axis):
    shape = [1, 1, 1]
    shape[axis] = int(numpy.asarray(vector).size)
    return numpy.asarray(vector, dtype=numpy.float32).reshape(shape)


def run_kernel_numpy(sub_step, fields, tables, codes, dtdx, flags, phases,
                     *, mask_arms=("metallic0", "mirror0", "mirror_top"),
                     beta=None, beta_after_mask=False,
                     beta_partner_shifted=False):
    """The emitted device tree, transcribed, in complex64 -- the laptop backend.

    Transcribed from the EMITTED SOURCE, statement for statement::

        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1),
                                                    cf_sub(f_2, ss)));
        [beta insert, targets 0 and 1 only]        # complex_beta_kernels
        if (bc_a == BC_METALLIC        && a == 0)      curl = cf_zero();
        if (bc_a == BC_MIRROR_PERIODIC && a == 0)      curl = cf_zero();
        if (bc_b == BC_MIRROR_PERIODIC && b == nb - 1) curl = cf_zero();
        pml_apply(...)

    THE MUTATION SEAMS are ``mask_arms``, ``beta_after_mask`` and
    ``beta_partner_shifted``, and they exist for the same reason the gate's do: a
    branch that is never shown to change an answer has not been tested, only
    executed. Dropping a mask arm is what makes the two fold masks measurable;
    moving the beta insert BELOW the masks is the composition needle a fold makes
    reachable (a fold widens the mask from METALLIC-only at cell 0 to
    non-PERIODIC at cell 0 PLUS MIRROR_PERIODIC at the top plane, and both arms
    reach a beta target); taking the SHIFTED companion instead of the centre is the
    partner defect.

    ``beta`` is ``((plus_re, plus_im), (minus_re, minus_im))`` or None, and lets
    ``test_complex_beta.py`` reuse this one transcription instead of keeping a
    second copy in step with it.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING.
    """
    spec = CURL_ARRAYS[sub_step]
    shift = _cshift_dn if spec["backward"] else _cshift_up
    scale = numpy.float32(dtdx)
    names = "xyz"
    coefficients = None
    if beta is not None:
        coefficients = (numpy.complex64(complex(*beta[0])),
                        numpy.complex64(complex(*beta[1])))
    for target_index, (target, aux, first, first_axis, second, second_axis,
                       dsig, dsigu) in enumerate(CURL_TERMS[sub_step]):
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        f_1 = getattr(fields, first)
        f_2 = getattr(fields, second)
        sf = shift(f_1, first_axis, codes[first_axis],
                   flags[first_axis], phases[first_axis])
        ss = shift(f_2, second_axis, codes[second_axis],
                   flags[second_axis], phases[second_axis])
        curl = (scale * ((sf - f_1) + (f_2 - ss))).astype(numpy.complex64)

        def _beta(current):
            partner = f_2 if target_index == 0 else f_1
            if beta_partner_shifted:
                partner = ss if target_index == 0 else sf
            return (current - (coefficients[target_index] * partner)).astype(
                numpy.complex64)

        live_beta = coefficients is not None and target_index in (0, 1)
        if live_beta and not beta_after_mask:
            curl = _beta(curl)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0:
                if codes[axis] == BC_METALLIC and "metallic0" in mask_arms:
                    curl[_plane(axis, 0, curl.shape)] = 0
                if codes[axis] == BC_MIRROR_PERIODIC and "mirror0" in mask_arms:
                    curl[_plane(axis, 0, curl.shape)] = 0
            elif codes[axis] == BC_MIRROR_PERIODIC and "mirror_top" in mask_arms:
                curl[_plane(axis, -1, curl.shape)] = 0
        if live_beta and beta_after_mask:
            curl = _beta(curl)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        fprev = fu.copy()
        fu_new = (((fprev * kms).astype(numpy.complex64) - curl).astype(
            numpy.complex64) * sinv).astype(numpy.complex64)
        fu[...] = fu_new
        a = ((f * kms_u).astype(numpy.complex64) + fu_new).astype(numpy.complex64)
        f[...] = ((a - fprev).astype(numpy.complex64) * sinv_u).astype(
            numpy.complex64)


#: The fixtures the behaviour legs step. Two of them are the CORPUS's own shapes in
#: kind -- a Mirror(Y) with a PERIODIC termination (``test_binary_grating_special_kz``,
#: ``TestHoleyWvgBands``) and a two-plane METALLIC fold (``solve-cw.py``,
#: ``TestArrayMetadata``) -- and the rest exist because a family must not release on
#: the half of a split it happens to need.
FOLD_SPECS = (
    {"label": "fold_Y_periodic", "cell": (8.0, 16.0, 0.0), "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "k": (0.0, 0.0, 0.0)},
    {"label": "fold_Y_metallic", "cell": (8.0, 16.0, 0.0), "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "k": (0.0, 0.0, 0.0)},
    {"label": "fold_Y_periodic_odd_plane", "cell": (8.0, 16.0, 0.0), "axes": "Y",
     "phase": -1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # AN ODD FULL COUNT on the folded axis. ``_stored_past_owned`` holds at BOTH
    # parities and what moves is the IMAGE ROW -- n_full 16 reflects stored-2 and
    # n_full 15 reflects stored-3 (measured on real Grid objects) -- so a family
    # that swept only the even arm would have baked ``n - 2`` without knowing it.
    {"label": "fold_Y_periodic_odd_count", "cell": (8.0, 15.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # A PHASE ON AN UNFOLDED AXIS, beside a fold on another: the corpus's
    # TestHoleyWvgBands / binary-grating class, and the case where a kernel that
    # applied the wrap factor on the folded axis would show.
    {"label": "fold_Y_periodic_kx", "cell": (8.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.31, 0.0, 0.0)},
    # THE SLOWEST-STRIDE AXIS, where an index defect that confused a plane with its
    # neighbour would move with the stride.
    {"label": "fold_X_periodic", "cell": (16.0, 8.0, 0.0), "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "k": (0.0, 0.0, 0.0)},
    # TWO PLANES AT ONCE, METALLIC -- solve-cw.py's class.
    {"label": "fold_XY_metallic", "cell": (16.0, 16.0, 0.0), "axes": "XY",
     "phase": 1, "boundaries": ("metallic", "metallic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # A 3-D fold with phases on the two unfolded axes -- the
    # TestModeDecomposition.test_triangular_lattice_oblique class.
    {"label": "fold_X_periodic_3d", "cell": (12.0, 8.0, 10.0), "axes": "X",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.19, 0.27)},
)


def build_spec(spec, courant=0.5, beta=0.0):
    """One fixture from its spec. A z extent of 0 means an effective 2-D grid, which
    is what ``Grid`` requires there and what every beta grid is by construction."""
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    dimensions = 2 if float(spec["cell"][2]) == 0.0 else None
    return build(spec["cell"], spec["boundaries"], planes, k_point=spec["k"],
                 courant=courant, beta=beta, dimensions=dimensions)


# ---------------------------------------------------------------------------
# THE DELTAS
# ---------------------------------------------------------------------------

def _real_kernel_mask_blocks():
    """The REAL curl pair's mask lines, per (sub_step, target), read from its source.

    Read out of ``step_curl_kernels.py``'s text rather than restated, because the
    whole point of the comparison is that the two are the same characters and a
    restatement here would compare this file with itself.
    """
    source = (HERE / "step_curl_kernels.py").read_text(encoding="utf-8")
    blocks = {}
    for sub_step, name in (("step_B", "step_B_pml_real"),
                           ("step_D", "step_D_pml_real")):
        start = source.index(f'extern "C" __global__ void {name}(')
        body = source[start:source.index("\n}\n", start)]
        targets = body.split("    {\n")[1:]
        assert len(targets) == 3, (name, len(targets))
        for index, block in enumerate(targets):
            blocks[(sub_step, index)] = "\n".join(
                line for line in block.split("\n") if "curl = 0.0f;" in line)
    return blocks


def test_the_mask_block_is_the_real_curl_pairs_character_for_character():
    """The fold IS a mask, so the mask must be the one the real gate measured.

    ``gate_cuda_folded_curl.py`` measured the METALLIC/PERIODIC split on the REAL
    pair: with the folded axis handed ``BC_METALLIC``, a folded METALLIC axis was
    already exact 48/48 and a folded PERIODIC one diverged 0/64 with all 19,649
    differing words on the mask-delta plane. That verdict is about THESE SIX
    BLOCKS. A complex family that spelled an "equivalent" mask would be claiming a
    verdict measured on different characters, so equality is the test -- with
    ``0.0f`` read as ``cf_zero()``, which is the storage change and the only one.
    """
    real = _real_kernel_mask_blocks()
    assert len(real) == 6
    for sub_step in ("step_B", "step_D"):
        backward = complex_emitter.KERNELS[sub_step][1]
        for target, axes in enumerate(complex_emitter._MASK_AXES[backward]):
            expected = real[(sub_step, target)].replace(
                "curl = 0.0f;", "curl = cf_zero();")
            assert expected, (sub_step, target, "no mask lines found in the real kernel")
            assert folded.fold_mask_lines(axes, target) == expected, (
                sub_step, target)


def test_the_mask_block_carries_all_three_arms_and_no_axis_carries_two():
    """The two cell-0 arms and the top-plane arm, and the per-axis complement.

    A component's shift-0 and shift-1 axes partition ``{x, y, z}``, so no component
    can ever carry a cell-0 mask AND a top-plane mask on the SAME axis. That is why
    the branch is three statements rather than one clamped index, and getting it
    wrong would drop a plane MEEP steps.
    """
    for sub_step in ("step_B", "step_D"):
        backward = complex_emitter.KERNELS[sub_step][1]
        for target, axes in enumerate(complex_emitter._MASK_AXES[backward]):
            block = folded.fold_mask_lines(axes, target)
            cell_zero_axes = set(re.findall(r"bc_(\w) == BC_METALLIC", block))
            mirror_zero = set(re.findall(
                r"bc_(\w) == BC_MIRROR_PERIODIC && \w == 0\)", block))
            top = set(re.findall(
                r"bc_(\w) == BC_MIRROR_PERIODIC && \w == n\w - 1\)", block))
            assert cell_zero_axes == set(axes), (sub_step, target)
            assert mirror_zero == set(axes), (sub_step, target)
            assert top == {"x", "y", "z"} - set(axes), (sub_step, target)
            assert not (mirror_zero & top), (sub_step, target)


def test_the_only_difference_from_the_certified_complex_source_is_the_deltas():
    """"No arithmetic change at all" is a claim a diff can check, so it is checked.

    Every line the transform ADDS must come from one of the declared deltas, and
    every line it REMOVES must be one a delta replaced. A stray edit -- a reordered
    multiply, a changed grouping, a dropped zero cross term -- would show here as a
    line in neither set, on a laptop, instead of as a byte divergence hours later
    on a device.
    """
    for sub_step in ("step_B", "step_D"):
        for arm in sorted(complex_emitter.EXPANSIONS):
            base = complex_emitter.complex_source(sub_step, arm).split("\n")
            new = folded.folded_source(sub_step, arm).split("\n")
            added_ok, removed_ok = set(), set()
            for _label, old, replacement, _sites in folded._deltas_for(sub_step):
                added_ok.update(replacement.split("\n"))
                removed_ok.update(old.split("\n"))
            added = [line for line in new if line not in base]
            removed = [line for line in base if line not in new]
            assert all(line in added_ok for line in added), (
                sub_step, arm, [line for line in added if line not in added_ok])
            assert all(line in removed_ok for line in removed), (
                sub_step, arm, [line for line in removed if line not in removed_ok])
            # And the transform must have DONE something: an empty diff would
            # satisfy both assertions above and would mean the fold never landed.
            assert added and removed, (sub_step, arm)


def test_a_moved_anchor_fails_the_transform_instead_of_emitting_a_short_kernel():
    """A site count is a WELD, and a weld has to be shown to hold.

    ``complex_emitter`` is a sibling under active development. If one of these
    anchors moves, the transform must RAISE -- because the alternative is a kernel
    with one fold branch missing, which compiles, runs, and is wrong on one plane
    of one component.
    """
    original = complex_emitter.complex_source
    for label, old, _new, _sites in folded._deltas_for("step_B"):
        def broken(sub_step, expansion, _old=old):
            return original(sub_step, expansion).replace(_old, "/* moved */")

        complex_emitter.complex_source = broken
        try:
            with pytest.raises(RuntimeError, match=r"matched 0 sites"):
                folded.folded_source("step_B", "FMA_V1")
        finally:
            complex_emitter.complex_source = original
        assert label  # every delta was exercised, by name


def test_the_fold_code_is_the_real_pairs_and_stays_out_of_the_shared_map():
    """Three codes for this family, two for the certified complex one.

    ``coverage.BC_CODES`` is SHARED with ``complex_pml_kernels``, whose kernels have
    no fold branch. Adding the third code there would hand a folded axis to a
    kernel that cannot serve one -- silently, because a kernel reads an ``int``.
    """
    assert (folded.BC_MIRROR_PERIODIC_CODE
            == coverage.REAL_CURL_BC_CODES[coverage.MIRROR_PERIODIC] == 2)
    assert coverage.MIRROR_PERIODIC not in coverage.BC_CODES
    assert set(coverage.BC_CODES) == {"periodic", "metallic"}
    assert folded._AXIS_INDEX == complex_emitter._AXIS_INDEX


def test_the_folded_ghost_takes_the_metallic_zero_on_both_shift_helpers():
    """Both faces, or the fold is only half applied.

    ``cshift_up`` serves ``step_B`` and ``cshift_dn`` serves ``step_D``; the delta
    is written once and lands on both, and the count of 2 in :data:`DELTAS` is what
    says so. This reads it back out of the emitted text, because a count is a claim
    about the transform and this is a claim about the kernel.
    """
    for sub_step in ("step_B", "step_D"):
        source = folded.folded_source(sub_step, "FMA_V1")
        assert source.count(
            "if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) return cf_zero();"
        ) == 2, sub_step
        assert "if (bc == BC_METALLIC) return cf_zero();" not in source
        assert "#define BC_MIRROR_PERIODIC 2" in source
        assert folded.FOLDED_KERNELS[sub_step] in source
        assert complex_emitter.KERNELS[sub_step][0] not in source


def test_the_transcribed_term_table_is_steppings():
    """The behaviour leg's term table against ``stepping``'s, so it is not a fiction.

    The leg consumes :data:`CURL_TERMS`; deriving it from ``stepping`` would make
    the leg compare the oracle with itself, and NOT pinning it would let a
    transcription error read as a passing test.
    """
    for sub_step, terms in CURL_TERMS.items():
        reference = (stepping.B_CURL_TERMS if sub_step == "step_B"
                     else stepping.D_CURL_TERMS)
        assert len(terms) == len(reference) == 3
        for (target, aux, first, first_axis, second, second_axis, dsig,
             dsigu), term in zip(terms, reference):
            assert target == term.target
            assert aux == "fu_" + term.target
            assert (first, first_axis) == (term.first, term.first_axis)
            assert (second, second_axis) == (term.second, term.second_axis)
            assert (dsig, dsigu) == ("xyz".index(term.dsig), "xyz".index(term.dsigu))


# ---------------------------------------------------------------------------
# THE BEHAVIOUR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", FOLD_SPECS, ids=lambda s: s["label"])
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_folded_curl_reproduces_stepping_word_for_word(spec, sub_step):
    """THE leg. The transcribed device tree against ``stepping`` on a folded grid.

    NON-VACUITY FLOORS, both asserted rather than hoped for:

    * the array path must MOVE state -- a comparison of two unchanged volumes
      passes for a kernel that does nothing;
    * the absorber profile must not be the identity -- with every ``kms`` at 1 and
      every ``sinv`` at 1 a coefficient-index error is invisible, so the case would
      measure the stencil and nothing else.
    """
    fields, layer, grid = build_spec(spec)
    rng = numpy.random.default_rng(20260820)
    arrays = (tuple(CURL_ARRAYS[sub_step]["targets"])
              + tuple(CURL_ARRAYS[sub_step]["aux"])
              + tuple(CURL_ARRAYS[sub_step]["sources"]))
    seed(fields, grid, arrays, rng)
    frozen = snapshot(fields, arrays)

    profile = numpy.concatenate([
        numpy.asarray(getattr(layer, f"{stem}_{axis}{suffix}")).ravel()
        for axis in "xyz" for stem in ("kms", "sinv") for suffix in ("", "_h")])
    assert float(numpy.max(numpy.abs(profile - 1.0))) > 1e-3, (
        "the absorber is the identity on every axis; this case cannot see a "
        "coefficient-index error")

    getattr(stepping, sub_step)(fields, layer)
    oracle = snapshot(fields, arrays)
    outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
               + tuple(CURL_ARRAYS[sub_step]["aux"]))
    assert differing_words(frozen, oracle, outputs) > 0, "the array path moved nothing"

    restore(fields, frozen)
    codes = boundary_codes_for(grid)
    assert BC_MIRROR_PERIODIC in codes or BC_METALLIC in codes, codes
    flags, phases = phase_arguments(grid, CURL_ARRAYS[sub_step]["backward"])
    run_kernel_numpy(sub_step, fields, tables_for(sub_step, layer), codes,
                     grid.dt / grid.dx, flags, phases)
    mine = snapshot(fields, arrays)
    assert differing_words(oracle, mine, outputs) == 0, (
        f"{spec['label']} {sub_step}: "
        f"{differing_words(oracle, mine, outputs)} words differ")


#: WHICH MASK ARM IS LIVE ON WHICH TERMINATION -- the split the real gate measured,
#: restated as an expectation this file checks rather than as prose. A folded
#: PERIODIC axis takes ``BC_MIRROR_PERIODIC`` and needs BOTH mirror arms; a folded
#: METALLIC axis takes ``BC_METALLIC``, whose cell-0 arm it already had, and must
#: NOT get a top-plane mask because MEEP steps that plane.
LIVE_ARMS = {
    BC_MIRROR_PERIODIC: ("mirror0", "mirror_top"),
    BC_METALLIC: ("metallic0",),
}


@pytest.mark.parametrize("spec", FOLD_SPECS, ids=lambda s: s["label"])
def test_the_fold_mask_arms_are_live_exactly_where_the_termination_says(spec):
    """Each arm is shown to reach a word -- and to reach NONE where it must not.

    THE HALF THAT IS EASY TO FORGET is the second one. A family that masked the top
    plane on a folded METALLIC axis too would still reproduce every folded-PERIODIC
    fixture, and would silently drop a plane MEEP steps. So this asserts the FULL
    matrix: dropping a live arm must change words, and dropping a dead one must
    change none.

    Without it, :func:`test_the_folded_curl_reproduces_stepping_word_for_word` would
    pass just as happily for a kernel whose fold branches were dead code on every
    fixture -- which is exactly the state the certified complex family is in and the
    reason this module exists.
    """
    fold_code = None
    for sub_step in ("step_B", "step_D"):
        for arm in ("metallic0", "mirror0", "mirror_top"):
            fields, layer, grid = build_spec(spec)
            rng = numpy.random.default_rng(20260821)
            arrays = (tuple(CURL_ARRAYS[sub_step]["targets"])
                      + tuple(CURL_ARRAYS[sub_step]["aux"])
                      + tuple(CURL_ARRAYS[sub_step]["sources"]))
            seed(fields, grid, arrays, rng)
            frozen = snapshot(fields, arrays)
            getattr(stepping, sub_step)(fields, layer)
            oracle = snapshot(fields, arrays)
            restore(fields, frozen)
            codes = boundary_codes_for(grid)
            folded_axes = [a for a in range(3) if grid.is_mirrored(a)]
            fold_code = codes[folded_axes[0]]
            assert all(codes[a] == fold_code for a in folded_axes), codes
            flags, phases = phase_arguments(
                grid, CURL_ARRAYS[sub_step]["backward"])
            kept = tuple(a for a in ("metallic0", "mirror0", "mirror_top")
                         if a != arm)
            run_kernel_numpy(sub_step, fields, tables_for(sub_step, layer), codes,
                             grid.dt / grid.dx, flags, phases, mask_arms=kept)
            mine = snapshot(fields, arrays)
            outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
                       + tuple(CURL_ARRAYS[sub_step]["aux"]))
            moved = differing_words(oracle, mine, outputs)
            if arm in LIVE_ARMS[fold_code]:
                assert moved > 0, (
                    f"{spec['label']} {sub_step}: dropping the {arm!r} arm changed "
                    f"nothing, so this fixture cannot see that arm at all")
            else:
                assert moved == 0, (
                    f"{spec['label']} {sub_step}: the {arm!r} arm moved {moved} "
                    f"words on a fold this termination does not give it")


@pytest.mark.parametrize("spec", [s for s in FOLD_SPECS
                                  if "metallic" not in s["label"]],
                         ids=lambda s: s["label"])
def test_the_mirror_ghost_the_array_path_writes_is_nonzero_and_still_dead(spec):
    """THE fold argument, measured on the array path rather than asserted.

    The claim this family rests on is that ``stepping``'s MIRROR ghost -- a
    parity-weighted image of a stored interior plane, not a zero -- never reaches
    an output word, because its one consumer plane is masked. Two halves, and both
    are checked:

    * the ghost is NOT zero. ``_shift_up``'s folded-PERIODIC arm writes
      ``parity * field[reflect_row]`` into the last stored slot; if that plane
      happened to be zero the whole argument would be vacuous and
      :func:`test_the_folded_curl_reproduces_stepping_word_for_word` would be
      passing for the wrong reason.
    * it is still dead. That is the other test's verdict, and this one only has to
      establish that it was not free.
    """
    fields, layer, grid = build_spec(spec)
    rng = numpy.random.default_rng(20260822)
    seed(fields, grid, ("Ex", "Ey", "Ez"), rng)
    axis = next(a for a in range(3) if grid.is_mirrored(a))
    reflect_rows = stepping._far_reflect_rows(grid)
    assert reflect_rows[axis] is not None, (
        "this fixture's folded axis stores no far ghost slot; it cannot exercise "
        "the top-plane arm")
    parities = stepping._mirror_phases(grid)
    shifted = stepping._shift_up(
        numpy, fields.Ez, axis, stepping.MIRROR, component="Ez",
        mirror_phase=parities[axis], reflect_row=reflect_rows[axis])
    ghost = numpy.asarray(shifted[_plane(axis, -1, fields.Ez.shape)])
    assert int(numpy.count_nonzero(words(ghost))) > 0, (
        f"{spec['label']}: the array path's mirror ghost is all zero words, so "
        f"'the kernel writes a zero instead' is not a claim this fixture tests")


# ---------------------------------------------------------------------------
# THE PREDICATE
# ---------------------------------------------------------------------------

def _ask(predicate, fields, layer, grid, key):
    return predicate(fields, layer, grid, key, license=LICENCE,
                     subnormal_policy=POLICY)


class _GridWithCupysName:
    """The run's grid behind a backend whose ``__name__`` is ``cupy``.

    The one clause a laptop cannot satisfy, satisfied at the INPUT rather than
    filtered out of a refusal string -- the technique the census battery uses, and
    necessary for the same reason: these predicates short-circuit.
    """

    class _Xp:
        __name__ = "cupy"

        def __init__(self, real):
            self._real = real

        def __getattr__(self, item):
            return getattr(self._real, item)

    def __init__(self, grid):
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", self._Xp(grid.xp))

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


@pytest.mark.parametrize("spec", FOLD_SPECS, ids=lambda s: s["label"])
def test_the_families_partition_a_folded_complex_row(spec):
    """Disjointness, per slot: EXACTLY ONE family may serve it -- whichever one.

    THE TEST IS THE PARTITION, NOT THE OWNERSHIP, and that distinction was bought
    on 2026-08-20, when ``coverage.covers_real_pml_complex_constitutive``
    gained ``admit_fold=True`` on a device verdict of its own, and every folded
    ``update_H`` slot became a slot two predicates admitted. A test that asserted
    "mine admits it" would have gone red for a sibling's correct widening; a test
    that asserts "exactly one admits it" goes red only for the thing that is
    actually wrong. The CURL is still this family's -- the certified complex curl
    takes ``bc_x/bc_y/bc_z`` with no code for a mirror -- and that is asserted
    separately below rather than folded into the count.
    """
    fields, layer, grid = build_spec(spec)
    proxy = _GridWithCupysName(grid)
    for sub_step in ("step_B", "step_D"):
        mine, reason = _ask(folded.covers_complex_folded_curl,
                            fields, layer, proxy, sub_step)
        theirs, base_reason = _ask(coverage.covers_real_pml_complex_curl,
                                   fields, layer, proxy, sub_step)
        assert int(mine) + int(theirs) == 1, (
            spec["label"], sub_step, reason, base_reason)
        assert mine, (
            spec["label"], sub_step,
            "the folded curl is this family's: the certified complex curl emits "
            "bc_x/bc_y/bc_z with no code for a mirror")
        assert "mirror symmetry" in base_reason
    for side in ("H", "E"):
        mine, reason = _ask(folded.covers_complex_folded_constitutive,
                            fields, layer, proxy, side)
        theirs, _ = _ask(coverage.covers_real_pml_complex_constitutive,
                         fields, layer, proxy, side)
        served = int(mine) + int(theirs)
        if side == "E" and not theirs and not mine:
            # An off-diagonal chi1inv row is refused by BOTH, which is a gap and
            # not an overlap. No fixture here carries one, so this is unreachable
            # today and is written so that adding one does not read as a failure.
            continue
        assert served == 1, (spec["label"], side, reason)
        if theirs:
            assert "already admits this configuration directly" in reason, (
                spec["label"], side,
                "the certified pair serves this slot but this family did not "
                "defer by name; the partition is not being enforced")


def test_an_unfolded_complex_run_is_refused_by_name():
    """The other half of the partition: this family may not take the base's rows."""
    fields, layer, grid = build(
        (12.0, 8.0, 0.0), ("periodic", "periodic", "periodic"), (),
        k_point=(0.21, 0.0, 0.0), dimensions=2)
    proxy = _GridWithCupysName(grid)
    for sub_step in ("step_B", "step_D"):
        covered, reason = _ask(folded.covers_complex_folded_curl,
                               fields, layer, proxy, sub_step)
        assert not covered
        assert "no mirror fold is active" in reason
        base, _ = _ask(coverage.covers_real_pml_complex_curl,
                       fields, layer, proxy, sub_step)
        assert base, "the certified pair should serve this row"


def test_a_real_storage_fold_is_refused_for_storage_not_for_the_fold():
    """A folded REAL run belongs to the certified real curl pair, which serves it.

    The refusal has to come from the STORAGE clause, not from a fold clause: this
    family inverts the fold and inherits everything else, so a real-storage grid
    must fall through to the base predicate's own storage refusal.
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 16.0, 0.0),
                boundaries=("periodic", "periodic", "periodic"),
                symmetry=(Mirror("Y", 1),), xp=numpy, courant=0.5, dimensions=2)
    layer = PML(grid=grid, thickness=((2, 2), (0, 2), (0, 0)))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = _ask(folded.covers_complex_folded_curl,
                           fields, layer, _GridWithCupysName(grid), "step_B")
    assert not covered
    assert "real float32 storage" in reason


class _StubGrid:
    """A grid that answers exactly what a clause asks -- for configurations a real
    ``Grid`` refuses to build.

    A phase on a folded axis is one of those: ``driver._require_bloch_is_
    representable`` refuses it, which is correct and is also why the predicate's own
    refusal cannot be exercised with a real object. The stub is how a fail-closed
    clause gets tested at all.
    """

    class _Xp:
        __name__ = "cupy"
        float32 = numpy.float32
        complex64 = numpy.complex64

    xp = _Xp()
    cylindrical = False
    bfast_active = False
    beta = 0.0
    dimensions = 2

    def __init__(self, mirrored=(False, False, False), metallic=(False,) * 3,
                 k_point=(0.0, 0.0, 0.0), stored_past_owned=(False,) * 3,
                 phases=(None, None, None)):
        self.shape = (8, 8, 1)
        self._mirrored = mirrored
        self._metallic = metallic
        self.k_point = k_point
        self._past = stored_past_owned
        self._phases = phases
        self.has_bloch = any(p is not None for p in phases)

    def has_symmetry(self):
        return any(self._mirrored)

    def is_mirrored(self, axis):
        return bool(self._mirrored[axis])

    def is_metallic(self, axis):
        return bool(self._metallic[axis])

    def is_axis(self, axis):  # noqa: ARG002
        return False

    def bloch_phase(self, axis):
        return self._phases[axis]

    def stored_cells(self, axis):
        return 8 + (1 if self._past[axis] else 0)

    def owned_cells(self, axis):  # noqa: ARG002
        return 8


def test_a_phase_on_a_folded_axis_is_refused_by_name():
    """The clause the fold-free proxy makes mandatory, and the one it costs 0 slots.

    ``_FoldFreeGrid`` makes a folded PERIODIC axis read back as an ordinary
    periodic wrap, so the certified predicate's "only a periodic wrap can carry a
    phase" clause would ADMIT a phase on it. The array path does not express that
    configuration -- ``_shift_up``'s PERIODIC and MIRROR arms are exclusive -- so
    the leak would be a rotation applied to a plane the mask is about to drop:
    silent on the interior, and wrong on the boundary of a band structure.
    """
    grid = _StubGrid(mirrored=(False, True, False), stored_past_owned=(False, True, False),
                     k_point=(0.0, 0.37, 0.0), phases=(None, complex(0.5, 0.5), None))
    reason = folded._fold_reasons(object(), grid)
    assert reason is not None
    assert "is folded AND carries k component" in reason


def test_a_fold_whose_two_terminations_disagree_is_refused():
    """Two routes to one fact, and a disagreement means NEITHER may be trusted.

    ``stored_cells > owned_cells`` and ``is_metallic`` are derived from the same
    split (``stepping._stored_past_owned``). If they disagree, the top-plane mask is
    applied where MEEP steps the plane or withheld where it does not -- a wrong
    answer on one plane of one component, not a crash.
    """
    grid = _StubGrid(mirrored=(False, True, False), metallic=(False, True, False),
                     stored_past_owned=(False, True, False))
    reason = folded._fold_reasons(object(), grid)
    assert reason is not None and "disagree" in reason


def test_a_grid_that_cannot_answer_is_refused_rather_than_raising():
    """FAIL-CLOSED. A raise escaping a predicate is a CRASHED RUN where "no" was right."""

    class _Mute:
        xp = _StubGrid._Xp()

        def has_symmetry(self):
            raise AttributeError("no symmetry table on this object")

    reason = folded._fold_reasons(object(), _Mute())
    assert reason is not None and "could not be asked whether it is folded" in reason


def test_the_admission_record_claims_nothing_it_has_not_measured():
    """A family record that says "passed" before its gate ran is worse than none.

    RELEASED ON DEVICE 2026-09-01, and the pins below moved WITH the release --
    the two-part-edit discipline this test exists for. Until that run this test
    pinned ``released`` False and empty ``artifacts``; now it pins the device
    leg's own numbers, their internal consistency, and the partition move that
    landed in the same edit (``CERTIFIED_KERNELS`` non-empty iff released).
    """
    record = folded.FOLDED_COMPLEX_ADMISSION
    assert record["released"] is True, (
        "released became True on the 2026-09-01 device leg; if it is False again "
        "the record was rolled back without rolling back this pin")
    assert record["artifacts"] == (
        "parity/meep_gpu/results/cuda_complex_folded_2026-09-01/keep/gate.json",
        "parity/meep_gpu/results/cuda_complex_folded_2026-09-01/flush/gate.json",
    )
    assert record["gate"].endswith("gate_cuda_complex_folded.py")
    assert any("nothing in meep_gpu imports cuda_kernels" in line
               for line in record["what_it_does_not_license"])
    # THE PARTITION MOVED WITH THE RELEASE, or "certified" is a word someone typed.
    assert folded.CERTIFIED_KERNELS == ("step_B_pml_complex_folded",
                                        "step_D_pml_complex_folded")
    assert folded.UNCERTIFIED_KERNELS == {}
    # THE DEVICE LEG'S NUMBERS MUST BE INTERNALLY CONSISTENT: a record whose arms
    # do not add up to its case count is a record nobody re-derived.
    device = record["device_leg"]
    assert (device["folded_periodic_cases"] + device["folded_metallic_cases"]
            == device["curl_cases_scored"]
            == device["curl_single_launch_identical"]
            == device["curl_multi_step_identical"] == 64)
    assert (device["mutation_legs_caught"] + device["mutation_legs_null_confirmed"]
            + device["mutation_legs_uncaught_or_unarmed"]
            == device["mutation_legs"])
    assert device["policies"] == ("ieee_keep_ftz_stripped", "meep_x86_flush"), (
        "a family released on one policy and consumed under the other is exactly "
        "the transfer POLICY_CONDITIONAL_LICENCE forbids")
    assert device["constitutive_cases_deferred"] == {"H": 32, "E": 32}, (
        "the constitutive sides defer BY NAME to the certified complex pair's "
        "fold admission; a scored count here would mean this family measured a "
        "slot it does not own")
    # THE HOST LEG STAYS, as the transcription record it always was, and its
    # numbers must still be internally consistent.
    host = record["host_leg"]
    assert host["artifact"].endswith("gate.json")
    assert "certifies no NVRTC" in host["backend"]
    assert (host["folded_periodic_cases"] + host["folded_metallic_cases"]
            == host["curl_cases_scored"] == host["curl_single_launch_identical"]
            == host["curl_multi_step_identical"])
    assert (host["mutation_legs_caught"] + host["mutation_legs_null_confirmed"]
            + host["mutation_legs_uncaught_or_unarmed"] == host["mutation_legs"])
    assert host["fold_ghost_wraps"].startswith("NULL CONFIRMED"), (
        "the fold argument is that the parity ghost never reaches a surviving "
        "word; that leg coming back CAUGHT would mean it does")
    assert record["slots"] == 10 and record["rows"] == 5
    assert record["sub_steps"] == ("step_B", "step_D")
