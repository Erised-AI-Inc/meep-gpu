"""What the COMPLEX-storage PML family must hold, on a machine with no GPU.

THE MERGE BAR FOR A SLICE WHOSE FAILURE MODE IS A SILENT WRONG ANSWER. Every claim
here is checkable without CuPy, without Triton and without a device: the emitter
imports nothing, the predicate imports nothing, and the CuPy half is read off its
syntax tree. What CANNOT be checked here is the only thing a device is for --
whether ``--fmad=false`` keeps every part-wise product and sum separately rounded
under NVRTC, whether the compiled bytes match the array path's. That is
``parity/meep_gpu/gate_cuda_complex.py`` and nothing in this file substitutes for
it; ``test_complex_chunk_forms.py`` executes the emitted text on the host.

The transcription tests read the reference out of ``stepping.py``,
``fields.py`` and the certified Triton sibling rather than restating it, so a
change to any of them that this family did not follow is a test failure here
rather than a divergence found on a device three weeks later.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from .. import fields as fields_module
from .. import stepping
from ..device_identity import weld_survives_edit
from . import complex_emitter, coverage

HERE = pathlib.Path(__file__).parent
EMITTER_MODULE = HERE / "complex_emitter.py"
KERNEL_MODULE = HERE / "complex_pml_kernels.py"
TRITON_COMPLEX = HERE.parent / "triton_kernels" / "complex_fields.py"

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')
SIGNATURE = re.compile(r'extern "C" __global__ void \w+\(([^)]*)\)', re.S)

SUB_STEPS = ("step_B", "step_D", "update_H", "update_E")


def module_level_literal(source: str, name: str):
    """A module-level constant, read off the syntax tree rather than imported.

    BOTH ``Assign`` AND ``AnnAssign`` are matched here, unlike the reader the
    PARTITION test uses. That asymmetry is deliberate and is the sibling families'
    own: ``UNCERTIFIED_KERNELS`` must stay UNANNOTATED because the partition is
    read by a checker that matches ``Assign`` only, and annotating it once made the
    name invisible and the partition unenforced. Everything else may be annotated,
    so this reader takes both.
    """
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            return ast.literal_eval(node.value)
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == name and node.value is not None):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level")


def unannotated_module_level_literal(source: str, name: str):
    """``Assign`` only -- the reader the partition depends on staying blind to."""
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise AssertionError(
        f"{name} is not an unannotated module-level assignment; the partition "
        f"reader matches ast.Assign only and an AnnAssign is invisible to it")


def every_source():
    return {sub_step: complex_emitter.complex_source(sub_step)
            for sub_step in SUB_STEPS}


# --------------------------------------------------------------------------
# THE TRANSCRIPTION
# --------------------------------------------------------------------------

def test_every_device_string_is_pure_ascii():
    """A COMPILE REQUIREMENT, not a style rule.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source to a ``.cu`` file
    through a bare ``open(..., 'w')``, so the bytes go through the interpreter's
    LOCALE encoding -- ASCII under C/POSIX, which is what a non-interactive shell
    on the validation host gets. Two em-dashes in a comment made a sibling kernel
    uncompilable on 2026-08-15, and it was found on device.
    """
    for key, source in every_source().items():
        source.encode("ascii")  # raises on the first non-ASCII byte
        assert KERNEL_DECLARATION.findall(source), key


def test_no_unary_minus_and_no_division_on_any_float_path():
    """Both are requirements rather than accidents, and each has its own reason.

    NEGATION is spelled ``* -1.0f``. On CUDA that is belt and braces -- ``neg.f32``
    does not canonicalize signed zeros -- but the SIBLING transcription this file
    copies is Triton's, where ``-x`` lowers as ``0.0 - x`` and turns every -0.0
    addend into +0.0 before the fma. Keeping the two bodies the same spelling is
    what makes them comparable; this test is what stops a future edit importing
    the Triton hazard by copying it back.

    DIVISION: the one PTX exception on record is ptxas expanding ``div.rn.f32``
    into a sequence whose range checks carry ``.FTZ`` in SASS regardless of the
    PTX modifier. It cannot arise in a divide-free family, and the reciprocal the
    recurrence needs is ``sinv``, computed host-side by ``PML``.
    """
    for key, source in every_source().items():
        for number, line in enumerate(source.splitlines(), 1):
            code = line.split("//", 1)[0]
            # The INTEGER index decomposition divides and that is not a float
            # operation: ``div.rn.f32`` is what the FTZ hazard rides on, and an
            # int divide cannot produce one. Lines declaring an int are skipped by
            # their declaration rather than by matching the expression, so a float
            # divide can never hide behind the exemption.
            if re.match(r"\s*(const\s+)?int\s", code):
                continue
            assert "/" not in code, f"{key} line {number}: {line}"
            assert not re.search(r"[=(,]\s*-[A-Za-z_(]", code), (
                f"{key} line {number} carries a unary minus on a value: {line}")


def test_the_helpers_are_the_part_wise_products_and_the_unfused_connection():
    """``part_products.scale``: one float32 product per part, and ``connect_parts``'
    unfused complex product for a phase -- read off the emitted helper bodies.

    The coefficient helpers carry no zero cross term and no fused multiply-add; the
    connection's four products and two sums are each rounded on their own, the phase
    on the left as the array path forms ``thephase * word``.
    """
    source = complex_emitter.complex_source("step_B")
    assert "    o.re = z.re * c;\n    o.im = z.im * c;" in source
    assert "    o.re = c * z.re;\n    o.im = c * z.im;" in source
    assert ("    o.re = (p.re * g.re) - (p.im * g.im);\n"
            "    o.im = (p.re * g.im) + (p.im * g.re);") in source
    assert "    o.re = a.re * -1.0f;\n    o.im = a.im * -1.0f;" in source
    part_products = (HERE.parent / "part_products.py").read_text(encoding="utf-8")
    assert "xp.multiply(coeff, values.real, out=out.real)" in part_products
    assert "xp.multiply(coeff, values.imag, out=out.imag)" in part_products


def test_no_zero_cross_term_and_no_fused_multiply_survives():
    """The previous family's two clauses, inverted.

    A literal zero cross term (``z.im * 0.0f``) or an explicit ``__fmaf_rn`` is the
    full complex multiply against ``(c + 0j)`` the array path no longer forms, and
    differs from the part-wise product on signed zeros -- so neither may reach any
    emitted source.
    """
    for sub_step, source in every_source().items():
        for needle in ("* 0.0f", "0.0f *", "__fmaf_rn", "fmaf("):
            assert needle not in source, (sub_step, needle)


def test_the_curl_stencil_is_meeps_bracket_with_the_presence_forms():
    """``step_curl``'s C-order bracket and its two partner-absent forms.

    ``stepping._curl_from_operands`` forms ``dtdx * (((g1s - g1) + g2) - g2s)`` with
    both partners present, ``dtdx * (g1s - g1)`` with the second absent and
    ``(-dtdx) * (g2s - g2)`` with the first absent; the previous package grouping
    ``(g1s - g1) + (g2 - g2s)`` is a different float32 number and appears nowhere.
    The array path's own lines are read out of ``stepping.py`` so the two cannot
    drift.
    """
    array_path = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    assert ("part_products.scale(dtdx, ((operands.shifted_first - first) + second)"
            in array_path)
    assert "part_products.scale(dtdx, operands.shifted_first - first)" in array_path
    assert "part_products.scale(-dtdx, operands.shifted_second - second)" in array_path
    for sub_step in ("step_B", "step_D"):
        source = complex_emitter.complex_source(sub_step)
        assert ("if (pr1 && pr2) return mul_coefficient_left(dtdx, "
                "cf_sub(cf_add(cf_sub(sf, f1), f2), ss));") in source
        assert "if (pr1) return mul_coefficient_left(dtdx, cf_sub(sf, f1));" in source
        assert ("if (pr2) return mul_coefficient_left(dtdx * -1.0f, cf_sub(ss, f2));"
                in source)
        assert source.count("cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, ") == 3
        assert "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss))" not in source


def test_the_constitutive_tail_is_meeps_grouping_and_not_two_accumulations():
    """``f + ((kps*src) - (kms*prev))`` -- the right-hand side formed first.

    ``stepping._apply_constitutive_pml`` transcribes ``step_update_EDHB``'s
    ``f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev``. The previous
    package grouping -- two accumulations, ``(f + kps*src) - kms*prev`` -- is a
    different float32 number and is the gate's inverted mutation now.
    """
    array_path = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    assert ("field += part_products.scale(kps, fw) - part_products.scale(kms, fw_previous)"
            in array_path)
    source = complex_emitter.complex_source("update_H")
    assert ("cf updated = dsigw ? cf_add(cf_load(f, idx), "
            "cf_sub(mul_coefficient_left(kps, src), "
            "mul_coefficient_left(kms, prev)))") in source
    assert "a = cf_add(a, mul_coefficient_left(kps, src));" not in source
    assert "a = cf_sub(a, mul_coefficient_left(kms, prev));" not in source


def test_prev_is_read_before_the_fw_store():
    """Storing first makes ``prev`` the value just written.

    Wrong only where ``kms != 0``, i.e. INSIDE THE PML ONLY, which reads as a
    slightly weaker absorber rather than as a bug -- the reason it is pinned here
    and carried as a gate mutation rather than left to review. Both sit under the
    chunk's ``dsigw`` flag (the auxiliary contract).
    """
    source = complex_emitter.complex_source("update_E")
    body = source.split("void constitutive_apply", 1)[1]
    assert body.index("cf prev = dsigw ? cf_load(fw, idx) : cf_zero();") < body.index(
        "if (dsigw) cf_store(fw, idx, src);")


def test_the_ownership_mask_axes_are_the_engines_yee_shifts():
    """``_mask_non_owned_cells`` zeroes cell 0 of every axis whose Yee shift is 0.

    Read off ``fields.IYEE_SHIFTS`` (fields.py:214-219) rather than restated: B
    targets have ONE zero-shift axis each and D targets have TWO, which is the
    asymmetry the emitted source has to show.
    """
    for sub_step, targets in (("step_B", ("Bx", "By", "Bz")),
                              ("step_D", ("Dx", "Dy", "Dz"))):
        backward = complex_emitter.KERNELS[sub_step][1]
        table = complex_emitter._MASK_AXES[backward]
        for index, target in enumerate(targets):
            shifts = fields_module.IYEE_SHIFTS[target]
            expected = tuple("xyz"[axis] for axis in range(3) if shifts[axis] == 0)
            assert table[index] == expected, (sub_step, target, shifts)
        source = complex_emitter.complex_source(sub_step)
        assert source.count("curl = cf_zero();") == sum(
            len(axes) for axes in table)


def test_the_curl_term_table_matches_the_engines():
    """Target 0 reads g2 windowed on y and g1 windowed on z, and so on.

    ``B_CURL_TERMS``/``D_CURL_TERMS`` name the partners and their axes; the emitted
    source names them positionally, and this is where the two are compared. Each
    partner is read at the cell itself through ``cwindow`` on ITS derivative axis --
    the connection across a phased window lives on that axis -- and shifted on the
    same axis. The two tables are IDENTICAL in shape between B and D -- only the
    shift direction, the connection's table and the sub-lattice differ -- which is
    why one template serves both.
    """
    axis_of = {0: "x", 1: "y", 2: "z"}
    coord_of = {0: "i", 1: "j", 2: "k"}
    for terms, sub_step, sources in (
            (stepping.B_CURL_TERMS, "step_B", ("Ex", "Ey", "Ez")),
            (stepping.D_CURL_TERMS, "step_D", ("Hx", "Hy", "Hz"))):
        source = complex_emitter.complex_source(sub_step)
        for index, term in enumerate(terms):
            first_slot = sources.index(term.first)
            second_slot = sources.index(term.second)
            a1, a2 = term.first_axis, term.second_axis
            block = source.split(f"// Target {index}:", 1)[1].split("}", 1)[0]
            assert (f"cf f_1 = cwindow(g{first_slot}, idx, {coord_of[a1]}, "
                    f"ph_{axis_of[a1]}, p{axis_of[a1]});") in block
            assert (f"cf f_2 = cwindow(g{second_slot}, idx, {coord_of[a2]}, "
                    f"ph_{axis_of[a2]}, p{axis_of[a2]});") in block
            assert f"g{first_slot}, idx, " in block
            assert f"bc_{axis_of[a1]}" in block
            assert f"bc_{axis_of[a2]}" in block
            assert f"pr_{axis_of[a1]}, pr_{axis_of[a2]});" in block


def test_the_sub_lattice_table_matches_the_engines_pairing():
    """B half-integer, D integer; H integer, E half-integer.

    ``_curl_coefficients`` (stepping.py:2473) and ``_constitutive_coefficients``
    (:2483) decide it; getting either backwards is a half-cell error in the
    absorber profile -- converged, smooth and wrong.
    """
    assert complex_emitter.HALF_INTEGER == {
        "step_B": True, "step_D": False, "H": False, "E": True}
    array_path = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    assert "half_integer=True" in array_path and "half_integer=False" in array_path


def test_the_boundary_and_phase_axes_are_runtime_arguments():
    """The design decision, enforced.

    A BRANCH axis (which index, which predicated zero, which connection one plane
    takes, which chunk form a cell takes) is a runtime argument: the boundary codes,
    the connection kinds and parts, and the run's chunk record ``cb``. Six binary
    axes as constexprs would be 64 sources for a family the corpus drives with a
    handful.
    """
    for sub_step in ("step_B", "step_D"):
        params = SIGNATURE.search(
            complex_emitter.complex_source(sub_step)).group(1)
        for name in ("int bc_x", "int bc_y", "int bc_z",
                     "int ph_x", "int ph_y", "int ph_z",
                     "float pxr", "float pxi", "float pyr", "float pyi",
                     "float pzr", "float pzi", "const int* __restrict__ cb"):
            assert name in params, (sub_step, name)


def test_the_inverse_epsilon_pointers_are_not_restrict_and_only_on_the_E_side():
    """An isotropic run hands the SAME device pointer three times.

    ``fields.py:1323-1326`` builds three references to one volume, and
    ``__restrict__`` on mutually aliasing arguments is a promise the caller cannot
    keep. They are read-only, so nothing is lost but the promise. The H side has
    no such parameter at all: its source is B directly, mu = 1 being already baked
    into ``H_CONSTITUTIVE_TERMS`` (stepping.py:949).
    """
    electric = complex_emitter.complex_source("update_E")
    magnetic = complex_emitter.complex_source("update_H")
    assert "const float* inv_eps_0" in electric
    assert "const float* __restrict__ inv_eps" not in electric
    assert "mul_field_left(s0, inv_eps_0[idx])" in electric
    # THE SIGNATURE, not the whole text: the shared prelude's comments name
    # ``inv_eps`` as one of the field-left orientations, so a substring search over
    # the file would fail for a reason that is documentation rather than code.
    assert "inv_eps" not in SIGNATURE.search(magnetic).group(1)
    assert "mul_field_left" not in magnetic.split("__global__", 1)[1]


def test_inverse_epsilon_is_indexed_by_the_complex_cell_and_never_word_doubled():
    """One REAL coefficient per cell, shared by both planes.

    ``inv_eps`` stays float32 under complex storage (stepping.py:41-50 against
    fields.py:1203-1204). Indexing it at ``2*idx`` would read the next cell's
    value into the imaginary plane -- a smooth, wrong permittivity.
    """
    source = complex_emitter.complex_source("update_E")
    for slot in range(3):
        assert f"inv_eps_{slot}[idx]" in source
        assert f"inv_eps_{slot}[2 * idx]" not in source


# --------------------------------------------------------------------------
# THE PARTITION AND THE RECORD
# --------------------------------------------------------------------------

def test_the_family_ships_four_kernels_and_it_is_partitioned():
    """A kernel cannot ship unmeasured: the two sets are disjoint and exhaustive.

    Read off the syntax tree rather than by importing, because importing
    ``complex_pml_kernels`` needs CuPy -- which is half the reason this file
    exists. ``UNCERTIFIED_KERNELS`` is spelled as an unannotated dict for the same
    reason the siblings spell theirs that way: an ``ast.AnnAssign`` is invisible to
    this reader, and an invisible partition is an unenforced one.
    """
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    certified = module_level_literal(source, "CERTIFIED_KERNELS")
    dead = unannotated_module_level_literal(source, "UNCERTIFIED_KERNELS")
    shipped = {complex_emitter.KERNELS[s][0] for s in SUB_STEPS}
    assert shipped == {"step_B_pml_complex_bloch",
                       "step_D_pml_complex_bloch",
                       "update_H_pml_complex_bloch",
                       "update_E_pml_complex_bloch"}
    assert not (set(certified) & set(dead)), "the partition overlaps"
    assert set(certified) | set(dead) == shipped, (
        f"every shipped kernel must be certified or named dead: "
        f"{shipped ^ (set(certified) | set(dead))}")


def test_the_names_do_not_collide_with_the_twelve_uncertified_complex_kernels():
    """``step_curl_kernels.py`` already ships ``step_B_pml_complex``.

    Those twelve are the ORIGINAL bundle's, measured 0/16 against the array path
    and carrying six named defects. A name collision would put one of them and one
    of these behind the same string in the compile memo.
    """
    old = KERNEL_DECLARATION.findall(
        (HERE / "step_curl_kernels.py").read_text(encoding="utf-8"))
    new = {complex_emitter.KERNELS[s][0] for s in SUB_STEPS}
    assert not (set(old) & new), sorted(set(old) & new)


def test_the_compile_options_match_all_four_siblings():
    """``--fmad=false``, spelled in the file rather than imported.

    A bit-identity probe loads these modules BY PATH, outside the package, where
    an import could pick up a different tuple than the one a gate compiled.
    """
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert module_level_literal(source, "_COMPILE_OPTIONS") == ("--fmad=false",)
    for sibling in ("constitutive_kernels.py", "ade_kernels.py",
                    "offdiag_constitutive_kernels.py"):
        text = (HERE / sibling).read_text(encoding="utf-8")
        assert module_level_literal(text, "_COMPILE_OPTIONS") == ("--fmad=false",)


def test_the_emitter_digest_moves_when_any_source_moves():
    """One sha256 over all eight sources; a changed character anywhere moves it."""
    before = complex_emitter.corpus_digest()
    original = complex_emitter.complex_source
    try:
        complex_emitter.complex_source = (
            lambda sub_step: original(sub_step) + "\n// moved\n")
        assert complex_emitter.corpus_digest() != before
    finally:
        complex_emitter.complex_source = original
    assert complex_emitter.corpus_digest() == before


# --------------------------------------------------------------------------
# THE PREDICATE
# --------------------------------------------------------------------------

class _Array:
    def __init__(self, shape, dtype="complex64", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.flags = type("F", (), {"c_contiguous": contiguous})()
        self.size = int(shape[0]) * int(shape[1]) * int(shape[2])


class _Vector:
    def __init__(self, axis, shape, dtype="float32"):
        self.shape = tuple(shape[a] if a == axis else 1 for a in range(3))
        self.size = shape[axis]
        self.dtype = dtype
        self.flags = type("F", (), {"c_contiguous": True})()


class _Xp:
    __name__ = "cupy"
    complex64 = "complex64"
    float32 = "float32"


SHAPE = (6, 7, 8)
FIELD_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
               "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
               "fu_Dx", "fu_Dy", "fu_Dz",
               "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def _grid(**over):
    facts = {"has_symmetry": False, "cylindrical": False,
             "mirrored": (False,) * 3, "is_axis": (False,) * 3,
             "metallic": (False,) * 3, "has_bloch": True,
             "bfast_active": False, "beta": 0.0, "shape": SHAPE,
             "k_point": (0.3, 0.0, 0.0),
             "phases": (complex(0.5, 0.5), None, None)}
    facts.update(over)

    class G:
        xp = _Xp()
        shape = facts["shape"]
        cylindrical = facts["cylindrical"]
        has_bloch = facts["has_bloch"]
        bfast_active = facts["bfast_active"]
        beta = facts["beta"]
        k_point = facts["k_point"]

        @staticmethod
        def has_symmetry():
            return facts["has_symmetry"]

        @staticmethod
        def is_mirrored(axis):
            return facts["mirrored"][axis]

        @staticmethod
        def is_axis(axis):
            return facts["is_axis"][axis]

        @staticmethod
        def is_metallic(axis):
            return facts["metallic"][axis]

        @staticmethod
        def bloch_phase(axis):
            return facts["phases"][axis]

    return G()


def _fields(grid, **over):
    facts = {"force_complex_fields": True, "stores_E": True,
             "polarizations": (), "has_polarizations": False,
             "has_offdiagonal_epsilon": False, "conductive": ()}
    facts.update(over)

    class F:
        force_complex_fields = facts["force_complex_fields"]
        stores_E = facts["stores_E"]
        polarizations = facts["polarizations"]
        has_polarizations = facts["has_polarizations"]
        has_offdiagonal_epsilon = facts["has_offdiagonal_epsilon"]
        _chi2_components = None
        _chi3_components = None

        @staticmethod
        def condfac_for(component):
            return object() if component in facts["conductive"] else None

        @staticmethod
        def inverse_epsilon_for(component):
            return _Array(SHAPE, dtype="float32")

    for name in FIELD_NAMES:
        setattr(F, name, _Array(SHAPE))
    return F()


def _pml():
    class P:
        is_active = True

    for axis, name in enumerate("xyz"):
        for stem in ("kms", "sinv", "kps"):
            for suffix in ("", "_h"):
                setattr(P, f"{stem}_{name}{suffix}", _Vector(axis, SHAPE))
    return P()


def test_a_well_formed_complex_configuration_is_admitted_at_every_sub_step():
    grid = _grid()
    for sub_step in ("step_B", "step_D"):
        assert coverage.covers_real_pml_complex_curl(
            _fields(grid), _pml(), grid, sub_step) == (
                True, "covered")
    for side in ("H", "E"):
        assert coverage.covers_real_pml_complex_constitutive(
            _fields(grid), _pml(), grid, side) == (
                True, "covered")


def test_the_storage_clause_is_a_partition_with_the_real_predicates():
    """No configuration is admitted by both families, and none falls between.

    The real predicates refuse ``force_complex_fields`` by name and this one
    REQUIRES it (or a nonzero k). A slot admitted by two families would be taken
    over twice or, worse, planned twice with different arithmetic.
    """
    grid = _grid(has_bloch=False, k_point=(0.0, 0.0, 0.0),
                 phases=(None, None, None))
    complex_fields_on = _fields(grid)
    covered, _ = coverage.covers_real_pml_complex_curl(
        complex_fields_on, _pml(), grid, "step_B")
    assert covered
    real = _fields(grid, force_complex_fields=False)
    covered, reason = coverage.covers_real_pml_complex_curl(
        real, _pml(), grid, "step_B")
    assert not covered and "real float32 storage" in reason


@pytest.mark.parametrize("over,needle", [
    ({"has_symmetry": True, "mirrored": (True, False, False)}, "mirror symmetry"),
    ({"cylindrical": True}, "cylindrical (Dcyl)"),
    ({"is_axis": (False, True, False)}, "cylindrical r = 0 axis"),
    ({"bfast_active": True}, "BFAST"),
    ({"beta": 0.5}, "special_kz"),
    ({"metallic": (True, False, False)}, "only a periodic wrap can carry a phase"),
])
def test_every_grid_clause_refuses_with_its_own_string(over, needle):
    grid = _grid(**over)
    covered, reason = coverage.covers_real_pml_complex_curl(
        _fields(grid), _pml(), grid, "step_B")
    assert not covered and needle in reason, reason


def test_a_metallic_axis_with_a_nonzero_k_is_refused_even_with_no_phase_object():
    """``_bloch_phases`` raises on that pairing, so the array path never runs it."""
    grid = _grid(metallic=(False, True, False), k_point=(0.0, 0.4, 0.0),
                 phases=(complex(0.5, 0.5), None, None))
    covered, reason = coverage.covers_real_pml_complex_curl(
        _fields(grid), _pml(), grid, "step_B")
    assert not covered and "no lattice vector for the phase" in reason


def test_an_unreadable_phase_is_not_an_unphased_one():
    class Boom:
        def __call__(self, axis):
            raise RuntimeError("no")

    grid = _grid()
    grid.__class__.bloch_phase = staticmethod(lambda axis: (_ for _ in ()).throw(
        RuntimeError("unreadable")))
    covered, reason = coverage.covers_real_pml_complex_curl(
        _fields(grid), _pml(), grid, "step_B")
    assert not covered and "unreadable phase is not an unphased one" in reason


def test_a_conductivity_is_charged_to_the_curl_that_writes_the_component():
    """Per TERM, as ``_apply_curl`` reads it (stepping.py:508).

    A D conductivity routes ``step_D`` to the three-history recurrence and leaves
    ``step_B`` an ordinary curl. Both constitutive sides stay admitted: nothing in
    ``update_H``/``update_E``/``_apply_constitutive_pml`` mentions ``condfac_for``.
    """
    grid = _grid()
    fields = _fields(grid, conductive=("Dx",))
    covered, _ = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_B")
    assert covered
    covered, reason = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_D")
    assert not covered and "carries a conductivity" in reason
    for side in ("H", "E"):
        covered, _ = coverage.covers_real_pml_complex_constitutive(
            fields, _pml(), grid, side)
        assert covered, side


def test_the_offdiagonal_row_is_refused_on_E_and_admitted_on_the_curl():
    grid = _grid()
    fields = _fields(grid, has_offdiagonal_epsilon=True)
    covered, _ = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_B")
    assert covered
    covered, _ = coverage.covers_real_pml_complex_constitutive(
        fields, _pml(), grid, "H")
    assert covered
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        fields, _pml(), grid, "E")
    assert not covered and "off-diagonal chi1inv" in reason


def test_real_float32_volumes_are_refused_by_dtype():
    """Real storage read as word pairs is half a volume of made-up complex cells."""
    grid = _grid()
    fields = _fields(grid)
    fields.Bx = _Array(SHAPE, dtype="float32")
    covered, reason = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_B")
    assert not covered and "not complex64" in reason


def test_a_strided_complex_volume_is_refused():
    grid = _grid()
    fields = _fields(grid)
    fields.Ex = _Array(SHAPE, contiguous=False)
    covered, reason = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_B")
    assert not covered and "not C-contiguous" in reason


def test_the_int32_bound_is_halved_against_the_real_paths():
    """The kernels address ``2*idx``, so ``2*ncells`` is what must stay below 2**31.

    A volume between the two bounds is admitted by the REAL predicate and must be
    refused here; that gap is the whole reason the clause is not shared.
    """
    cells = 2 ** 30 + 8
    grid = _grid(shape=(cells, 1, 1))
    fields = _fields(grid)
    for name in FIELD_NAMES:
        setattr(fields.__class__, name, _Array((cells, 1, 1)))
    covered, reason = coverage.covers_real_pml_complex_curl(
        fields, _pml(), grid, "step_B")
    assert not covered and "int32 word-offset range" in reason


def test_the_sub_step_argument_is_required_and_validated():
    grid = _grid()
    with pytest.raises(ValueError):
        coverage.covers_real_pml_complex_curl(
            _fields(grid), _pml(), grid, "update_H")
    with pytest.raises(ValueError):
        coverage.covers_real_pml_complex_constitutive(
            _fields(grid), _pml(), grid, "step_B")


def test_every_uncertified_name_has_no_record_block_and_the_reverse():
    """THE WELD BETWEEN THE PARTITION AND THE RECORD, in both directions.

    ``certification.json`` is the sibling of ``triton_kernels/fingerprints.json``
    and the thing a reader is entitled to check a claim against. "Certified" must
    therefore be a VERDICT someone can read, never a word someone typed -- so:

    * a name in ``CERTIFIED_KERNELS`` must have a record block that CLAIMS it and
      reports a pass;
    * a name in ``UNCERTIFIED_KERNELS`` must have NO record block.

    The second half is what makes the two edits atomic. All four kernels have
    passed their gate (``results/cuda_complex_release_2026-08-19/``) and sit in
    ``UNCERTIFIED_KERNELS`` because the record has not been written yet; moving a
    name across without adding its block fails here, and adding the block without
    moving the name fails here too. Neither half of that change is valid alone,
    which is the property this test exists to hold rather than a state it is
    tolerating.
    """
    import json  # noqa: PLC0415

    source = KERNEL_MODULE.read_text(encoding="utf-8")
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    dead = set(unannotated_module_level_literal(source, "UNCERTIFIED_KERNELS"))
    record = json.loads((HERE / "certification.json").read_text(encoding="utf-8"))

    claimed = set()
    for block in record.values():
        if not isinstance(block, dict):
            continue
        for key in ("kernels", "kernel", "certified_kernels"):
            value = block.get(key)
            if isinstance(value, str):
                claimed.add(value)
            elif isinstance(value, (list, tuple)):
                claimed.update(value)

    ours = {complex_emitter.KERNELS[s][0] for s in SUB_STEPS}
    assert certified <= claimed, (
        f"certified without a certification.json block: {sorted(certified - claimed)}")
    assert not (dead & claimed), (
        f"a record block claims a kernel this file still lists as uncertified: "
        f"{sorted(dead & claimed)}; the partition move and the record entry are "
        f"one change and must land together")
    assert (certified | dead) == ours


# --------------------------------------------------------------------------
# THE Dcyl CLAUSE — refused for the curl, admitted for the constitutive pair
# --------------------------------------------------------------------------
#
# WHY THE ASYMMETRY IS A TEST AND NOT A COMMENT. ``_complex_grid_refusal`` is one
# function with two callers, and the whole of the difference between them is one
# argument. A change that flipped the default, or that passed it from the curl,
# would widen a family that has never been measured on a Dcyl grid and nothing
# else in this file would notice. What licenses the constitutive half is
# :data:`coverage.COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION` and nothing else.


def _dcyl_grid(z_metallic=True, **over):
    """A Dcyl stand-in shaped like every Dcyl row the corpus carries.

    r resolves to the AXIS, phi is the invariant periodic axis MEEP collapses to
    one cell, and z is the only chosen termination (grid.py:498-508). All sixteen
    complex Dcyl rows in the 2026-08-20 census record exactly this pair of
    boundary kinds and ``has_bloch`` FALSE.
    """
    facts = {"cylindrical": True, "is_axis": (True, False, False),
             "metallic": (False, False, bool(z_metallic)),
             "has_bloch": False, "k_point": (0.0, 0.0, 0.0),
             "phases": (None, None, None), "shape": SHAPE}
    facts.update(over)
    return _grid(**facts)


@pytest.mark.parametrize("z_metallic", [True, False])
@pytest.mark.parametrize("side", ["H", "E"])
def test_a_dcyl_grid_is_admitted_by_the_constitutive_pair(z_metallic, side):
    """BOTH z TERMINATIONS, because the termination is what moves the stored extent.

    A periodic axis carries one stored cell past the owned window and a metallic
    one does not (grid.py:570-575), and the deepest absorber coefficient lands in
    that extra slot. The device gate swept both and was bit-identical on both.
    """
    grid = _dcyl_grid(z_metallic=z_metallic)
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        _fields(grid), _pml(), grid, side)
    assert covered, reason


@pytest.mark.parametrize("z_metallic", [True, False])
@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_a_dcyl_grid_is_still_refused_by_the_complex_curl(z_metallic, sub_step):
    """THE HALF THAT MUST NOT MOVE.

    ``cshift_up``/``cshift_dn`` are a neighbour stencil and Dcyl replaces the
    radial derivative with a prefix sum and adds the axis-row rules. Nothing has
    measured that, and this is the test that says so.
    """
    grid = _dcyl_grid(z_metallic=z_metallic)
    covered, reason = coverage.covers_real_pml_complex_curl(
        _fields(grid), _pml(), grid, sub_step)
    assert not covered and "cylindrical (Dcyl)" in reason, reason


@pytest.mark.parametrize("side", ["H", "E"])
def test_a_phased_dcyl_grid_is_refused_by_name_on_the_constitutive_pair(side):
    """UNMEASURED IS UNMEASURED, and it is refused rather than argued about.

    Dcyl Bloch is z-only in MEEP (grid.py:654-657), no corpus row carries one, and
    the constitutive kernel takes no phase argument at all -- so "the kernel cannot
    see it" is available as an argument and is deliberately not used. The gate
    swept only unphased Dcyl grids and the clause says so; it costs 0 slots.
    """
    grid = _dcyl_grid(z_metallic=False, has_bloch=True, k_point=(0.0, 0.0, 0.3),
                      phases=(None, None, complex(-1.0, 0.0)))
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        _fields(grid), _pml(), grid, side)
    assert not covered and "Bloch phase on a cylindrical grid" in reason, reason


@pytest.mark.parametrize("side", ["H", "E"])
def test_an_axis_flag_without_the_cylindrical_flag_is_refused_on_both_families(side):
    """FAIL-CLOSED, and NOT inherited from the retired clause.

    A grid whose r = 0 flag is set while it denies being cylindrical disagrees
    with itself, and neither predicate can answer for one. The admission reaches
    the axis row only through ``facts["cylindrical"]``; a grid that sets one flag
    and not the other must keep the old refusal on BOTH callers.
    """
    grid = _grid(cylindrical=False, is_axis=(True, False, False),
                 has_bloch=False, k_point=(0.0, 0.0, 0.0),
                 phases=(None, None, None))
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        _fields(grid), _pml(), grid, side)
    assert not covered and "cylindrical r = 0 axis" in reason, reason
    covered, reason = coverage.covers_real_pml_complex_curl(
        _fields(grid), _pml(), grid, "step_B")
    assert not covered and "cylindrical r = 0 axis" in reason, reason


def test_the_axis_boundary_kind_is_admitted_by_the_constitutive_list_only():
    """``CYL_AXIS`` has no ``BC_CODES`` entry, and the two lists say why.

    The curl's emitted source takes ``bc_x``/``bc_y``/``bc_z`` as runtime arguments
    and there is no code for the axis to pass; the constitutive template takes no
    boundary argument at all, so the kind it never reads cannot be the wrong one.
    """
    assert coverage.CYL_AXIS not in coverage.BC_CODES
    assert coverage.CYL_AXIS in coverage.COMPLEX_CONSTITUTIVE_BOUNDARY_KINDS
    assert set(coverage.COMPLEX_CONSTITUTIVE_BOUNDARY_KINDS) >= set(coverage.BC_CODES)


def test_only_the_constitutive_predicate_asks_to_admit_a_dcyl_grid():
    """EXACTLY ONE CALL SITE passes ``admit_cylindrical`` at all, and it is the pair.

    Read off the syntax tree rather than by grep: the parameter defaults to the
    refusal, so a new caller inherits the "no", and this is what keeps an added
    caller from acquiring the admission by copying a line.
    """
    source = (HERE / "coverage.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            target = inner.func
            if not (isinstance(target, ast.Name)
                    and target.id == "_complex_grid_refusal"):
                continue
            passes = any(keyword.arg == "admit_cylindrical"
                         and getattr(keyword.value, "value", None) is True
                         for keyword in inner.keywords)
            callers.append((node.name, passes))
    assert callers, "no call to _complex_grid_refusal was found at all"
    admitting = [name for name, passes in callers if passes]
    assert admitting == ["covers_real_pml_complex_constitutive"], callers
    signature = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "_complex_grid_refusal")
    default = signature.args.defaults[-1]
    assert getattr(default, "value", None) is False, (
        "admit_cylindrical must default to the REFUSAL; a caller that has not "
        "been measured on a Dcyl grid has to inherit the no")


def test_the_dcyl_admission_names_a_verdict_a_reader_can_check():
    """The admission is a RECORD, not a sentence: every field a reader needs.

    Both float32 subnormal policies, both z terminations, all three |m| classes and
    a negative m, the refusal's own premise armed and caught, the release verdict
    shown to flip, and the slot arithmetic recomputed on a stated denominator.
    """
    record = coverage.COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION
    assert record["identical_single_launch"] == record["cases_per_policy"]
    assert record["identical_at_sixty_launches"] == record["cases_per_policy"]
    assert record["dcyl_cases_per_policy"] < record["cases_per_policy"], (
        "the Cartesian controls are what make a Dcyl divergence attributable to "
        "Dcyl rather than to the fixture; a record with none has no control")
    assert set(record["policies"]) == {"ieee_keep_ftz_stripped", "meep_x86_flush"}
    assert set(record["z_terminations_swept"]) == {"metallic", "periodic"}
    assert set(record["m_classes_swept"]) == {"m0", "m1", "m2plus"}
    assert min(record["m_values_swept"]) < 0, "no negative m was swept"
    assert "signed_zero" in record["value_classes_swept"]
    assert record["mutation_legs_as_required"] == record["mutation_legs"]
    assert record["verdict_shown_to_flip"]
    assert record["phased_dcyl_swept"] is False
    assert record["slots_after"] - record["slots_before"] == record["slots_gained"]
    assert record["slots_gained"] == 2 * record["rows_gained"], (
        "each gained row is worth exactly its two constitutive sub-steps; the "
        "curl still refuses every one of them")
    assert (record["rows_covered_at_every_sub_step_after"]
            == record["rows_covered_at_every_sub_step_before"]), (
        "the curl still refuses these rows, so not one of them becomes a row the "
        "hand-CUDA track can step end to end -- a record that claimed otherwise "
        "would be claiming the curl moved")
    assert record["slots_after"] <= record["denominator"]


def test_the_dcyl_admission_matches_the_artifact_when_it_is_present():
    """If the gate's artifact is on disk, the record must be its summary.

    ``parity/meep_gpu/results/`` is gitignored, so a fresh checkout has nothing to
    check against and this test skips -- which is why the digests are in the record
    as well. Where the bytes ARE present, a record that drifted from them is caught
    here rather than by a reader who trusted the prose.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    record = coverage.COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION
    root = HERE.parent.parent / record["artifacts"]
    if not root.is_dir():
        pytest.skip(f"{record['artifacts']} is not on this checkout (gitignored)")
    for relative, digest in record["artifact_sha256"].items():
        path = root / relative
        assert path.is_file(), f"{relative} is missing from {root}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"{relative} is not the bytes the record names"
    for policy_leg in ("keep", "flush"):
        summary = json.loads((root / policy_leg / "gate.json").read_text(
            encoding="utf-8"))["summary"]
        assert summary["released"], (policy_leg, summary["reasons"])
        assert summary["scored_cases"] == record["cases_per_policy"]
        assert summary["cylindrical_cases"] == record["dcyl_cases_per_policy"]
        assert summary["single_launch_identical"] == record["identical_single_launch"]
        assert (summary["multi_step_identical"]
                == record["identical_at_sixty_launches"])
        assert sorted(summary["m_values"]) == sorted(record["m_values_swept"])
    falsified = json.loads((root / "falsify" / "gate.json").read_text(
        encoding="utf-8"))["summary"]
    assert falsified["released"] is False, (
        "the release verdict must be shown to FLIP against a planted defect; a "
        "gate whose legs can fail but whose verdict cannot is not a gate")


# --------------------------------------------------------------------------
# THE MIRROR FOLD, admitted for the CONSTITUTIVE pair on 2026-08-20 and still
# refused for the CURL. Both halves are pinned, because the asymmetry IS the
# claim: one emitted template takes bc_x/bc_y/bc_z and the other takes no
# boundary argument at all.
# --------------------------------------------------------------------------

_FOLD_SHAPES = (
    # (mirrored axes, metallic axes, what it is)
    ((True, False, False), (False, False, False), "one plane, PERIODIC-terminated"),
    ((False, True, False), (False, True, False), "one plane, METALLIC-terminated"),
    ((True, True, False), (True, True, False), "two planes, the corpus's own shape"),
    ((True, True, True), (True, True, True), "three planes"),
)


@pytest.mark.parametrize("mirrored,metallic,what", _FOLD_SHAPES,
                         ids=lambda value: str(value)[:28])
@pytest.mark.parametrize("side", ("H", "E"))
def test_the_constitutive_pair_admits_a_fold_and_the_curl_still_refuses(
        side, mirrored, metallic, what):
    """The one asymmetry ``admit_fold`` exists for, measured on both sides of it.

    The CONSTITUTIVE pair admits, on a device verdict of its own
    (:data:`coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION`: 180/180 per policy on
    156 folded cases, both terminations, all three axes, up to three simultaneous
    planes, the refusal's own premise armed and caught, the release verdict shown
    to flip). The CURL refuses, and its refusal is real: ``cshift_up``/
    ``cshift_dn`` are a neighbour stencil and the emitted curl takes
    ``bc_x/bc_y/bc_z`` with no code for a mirror.

    A TEST THAT CHECKED ONLY THE ADMISSION would pass just as well if the fold
    clause had been deleted from the shared helper, which would widen the curl
    too -- the exact leak ``test_the_sibling_predicates_did_not_widen_with_this_one``
    caught on the real track.
    """
    # A FOLDED AXIS CARRIES NO PHASE: MEEP forces a mirror plane's own axis to
    # k = 0 (Grid._resolve_bloch raises otherwise), so the phase goes on an
    # unfolded axis -- which is what all eight folded complex corpus rows record.
    k_point = tuple(0.0 if mirrored[axis] else 0.3 for axis in range(3))
    phases = tuple(None if (mirrored[axis] or metallic[axis] or k_point[axis] == 0.0)
                   else complex(0.5, 0.5) for axis in range(3))
    grid = _grid(has_symmetry=True, mirrored=mirrored, metallic=metallic,
                 k_point=k_point, phases=phases,
                 has_bloch=any(value is not None for value in phases))
    fields = _fields(grid)

    covered, reason = coverage.covers_real_pml_complex_constitutive(
        fields, _pml(), grid, side)
    assert covered, (what, side, reason)

    for sub_step in ("step_B", "step_D"):
        covered, reason = coverage.covers_real_pml_complex_curl(
            fields, _pml(), grid, sub_step)
        assert not covered, (what, sub_step)
        assert "mirror symmetry" in reason, reason


def test_admit_fold_defaults_to_the_refusal():
    """A caller that has not been measured inherits the "no".

    The parameter is what keeps ONE helper answering for TWO families, and its
    DEFAULT is the load-bearing half: a third caller added later gets the refusal
    unless somebody passes True, which is a decision with a record behind it.
    """
    grid = _grid(has_symmetry=True, mirrored=(False, True, False),
                 metallic=(False, True, False), k_point=(0.0, 0.0, 0.0),
                 phases=(None, None, None), has_bloch=False)
    fields = _fields(grid)
    assert coverage._complex_grid_refusal(fields, _pml(), grid) is not None
    assert coverage._complex_grid_refusal(
        fields, _pml(), grid, admit_cylindrical=True) is not None, (
        "admit_cylindrical alone admitted a fold, so the two parameters are not "
        "independent and the Dcyl verdict is carrying the fold one")
    assert coverage._complex_grid_refusal(
        fields, _pml(), grid, admit_cylindrical=True, admit_fold=True) is None


def test_a_grid_reporting_symmetry_with_no_folded_axis_fails_closed():
    """Unbuildable from a real ``Grid``, and refused rather than guessed.

    Same clause, same reason and same wording as ``covers_real_pml_curl``'s. It
    exists because keying the admission on ``mirrored`` alone would silently
    admit a ghost rule this module has never resolved.
    """
    grid = _grid(has_symmetry=True, mirrored=(False, False, False),
                 k_point=(0.0, 0.0, 0.0), phases=(None, None, None),
                 has_bloch=False)
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        _fields(grid), _pml(), grid, "H")
    assert not covered
    assert "symmetry with no folded axis" in reason, reason


def test_the_complex_fold_plane_cap_is_the_record_s_number():
    """Lower the record and the refusal comes back -- the clause reads the record.

    With the cap at 3 and a grid at three axes the clause cannot fire, so what is
    measurable is that the number DECIDING a three-plane grid is the record's.
    Without this, "the cap is a read of the table" would rest on a string search.
    """
    grid = _grid(has_symmetry=True, mirrored=(True, True, True),
                 metallic=(True, True, True), k_point=(0.0, 0.0, 0.0),
                 phases=(None, None, None), has_bloch=False)
    fields = _fields(grid)
    table = coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION
    assert table["folded_planes_swept"] == 3
    assert coverage.covers_real_pml_complex_constitutive(
        fields, _pml(), grid, "H")[0]
    shipped = table["folded_planes_swept"]
    try:
        table["folded_planes_swept"] = 2
        covered, reason = coverage.covers_real_pml_complex_constitutive(
            fields, _pml(), grid, "H")
        assert not covered, (
            "lowering the record's plane count did not change the verdict, so the "
            "predicate is not reading the record and the cap is unpinned")
        assert "3 mirror planes at once" in reason, reason
    finally:
        table["folded_planes_swept"] = shipped
    assert coverage.covers_real_pml_complex_constitutive(
        fields, _pml(), grid, "H")[0]


def test_the_complex_fold_admission_is_the_run_that_produced_it():
    """The record's numbers must be the artifact's, and the verdict must flip.

    Same shape as the Dcyl admission's test above and for the same reason: a
    number in a table nothing recomputes is a number that outlives its
    measurement. ``results/`` is gitignored, so this is a declared skip where the
    bytes are absent -- and where they are present, a record that drifted from
    them is caught here rather than by a reader who trusted the prose.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    record = coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION
    root = HERE.parent.parent / record["artifacts"]
    if not root.is_dir():
        pytest.skip(f"{record['artifacts']} is not on this checkout (gitignored)")
    for relative, digest in record["artifact_sha256"].items():
        path = root / relative
        assert path.is_file(), f"{relative} is missing from {root}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"{relative} is not the bytes the record names"
    # THE SUBJECT WAS NOT TOUCHED: a record whose kernel bytes moved describes a
    # kernel that no longer exists.
    for relative, digest in record["subject_sha256"].items():
        path = HERE.parent.parent / relative
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live == digest:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209): an edit that provably
        # cannot reach the compiled program leaves the kernel this record
        # describes intact. Anything the helper cannot establish -- including a
        # record carrying no device_sha256/code_sha256, which is this one today
        # -- comes back False and falls through to the byte rule below.
        if weld_survives_edit(path, record, relative):
            continue
        assert live == digest, f"{relative} has moved since the gate ran"
    for policy_leg in ("keep", "flush"):
        summary = json.loads((root / policy_leg / "gate.json").read_text(
            encoding="utf-8"))["summary"]
        assert summary["released"], (policy_leg, summary["reasons"])
        assert summary["scored_cases"] == record["cases_per_policy"]
        assert summary["folded_cases"] == record["folded_cases_per_policy"]
        assert summary["single_launch_identical"] == record["identical_single_launch"]
        assert (summary["multi_step_identical"]
                == record["identical_at_sixty_launches"])
        # THE CAP IS THE RUN'S OWN MAXIMUM, derived by the gate from the cases it
        # scored rather than from its spec tuple.
        assert (summary["max_folded_planes_scored"]
                == record["folded_planes_swept"]), policy_leg
        assert sorted(summary["folded_terminations"]) == sorted(
            record["folded_terminations_swept"])
        assert sorted(summary["folded_axes"]) == sorted(record["folded_axes_swept"])
        # AND THE PHASED ARM, which is the corpus's own shape: five of the eight
        # rows carry a Bloch phase on an unfolded axis.
        assert summary["phased_folded_cases"] == record["phased_folded_cases"]
    falsified = json.loads((root / "falsify" / "gate.json").read_text(
        encoding="utf-8"))["summary"]
    assert falsified["released"] is False, (
        "the release verdict must be shown to FLIP against a planted defect; a "
        "gate whose legs can fail but whose verdict cannot is not a gate")
    assert falsified["falsification"]["verdict_flipped"] is True
