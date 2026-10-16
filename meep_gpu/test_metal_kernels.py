"""Merge-bar tests for the Metal kernel package.

Four things are pinned here that no gate can pin, because a gate needs a GPU and
this file must pass on any host:

1. **The optional-dependency contract.** :mod:`meep_gpu.metal_kernels`, its
   coverage predicates and its shader sources import with no torch anywhere near
   them. A predicate that could only be read on a Mac would be a predicate nobody
   reviews.
2. **The clause-list equivalence pin.** ``metal_kernels.coverage._grid_reasons``
   is a DUPLICATE of the Triton one with clause 1 replaced, and this file runs
   both over a shared configuration matrix and asserts the reason sets are equal
   once each side's own backend clause is removed. That is what makes the
   duplication a stated debt rather than a drift generator: a clause added to
   either list and not to the other fails here, which a string filter over the
   Triton list would never have caught.
3. **The single contraction-pragma spelling.** The directive that decides
   bit-identity is spelled once in the package; a second spelling, or any math-mode
   pragma, fails the build. Mirrors the Triton package's ``ENABLE_FP_FUSION`` test.
4. **The fingerprints.** The checked-in file must match the tree.
5. **The chunk record and the per-chunk forms**, read off the emitted text: the
   record binds last; the anchored bounds block names one membership per target on
   the region plan's own (axis, lattice); every auxiliary load and store sits behind
   its form's membership; both plain forms are output selects; the curl is MEEP's
   left-to-right bracket with the absent-partner forms; the shared splice anchors
   survive, so a family that lifts a body needs only ``chunk_bounds.params_alias()``
   to bind it; and no plan route takes a default record.

The device-touching tests below are skipped without MPS, and they are the ones
that check the arithmetic against ``stepping.py`` in process. The full case matrix
lives in ``parity/meep_gpu/gate_metal_pml.py``; what is here is the smallest
non-vacuous version of it, so a red merge bar names the defect without a GPU run.
"""

from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
import pytest

from meep_gpu.metal_kernels import chunk_bounds
from meep_gpu.metal_kernels import coverage as metal_coverage
from meep_gpu.metal_kernels import shaders, subnormal
from meep_gpu.triton_kernels import coverage as triton_coverage

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_kernels")


def _package_sources():
    for name in sorted(os.listdir(PACKAGE_DIR)):
        if name.endswith(".py"):
            with open(os.path.join(PACKAGE_DIR, name), "r", encoding="utf-8") as handle:
                yield name, handle.read()


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
# 1. The optional-dependency contract
# ---------------------------------------------------------------------------

def test_no_module_level_torch_import_anywhere_in_the_package():
    """torch is imported inside the functions that launch, and nowhere else.

    A module-level import would make the coverage predicates unreadable on any
    host without a Mac GPU, which is most of them, and would drag a heavy optional
    dependency into ordinary engine startup.
    """
    offenders = []
    for name, source in _package_sources():
        tree = ast.parse(source)
        for node in tree.body:  # module level only; nested imports are the pattern
            if isinstance(node, ast.Import):
                offenders.extend(f"{name}: import {alias.name}"
                                 for alias in node.names
                                 if alias.name.split(".")[0] == "torch")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
                offenders.append(f"{name}: from {node.module} import ...")
    assert not offenders, f"module-level torch imports: {offenders}"


def test_coverage_and_shaders_import_without_touching_torch():
    """Importing the predicate side must not pull torch in as a side effect."""
    import subprocess
    import sys

    code = ("import sys;"
            "import meep_gpu.metal_kernels.coverage;"
            "import meep_gpu.metal_kernels.shaders;"
            "print('torch' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=root, check=True)
    assert result.stdout.strip() == "False", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# 2. The clause-list equivalence pin
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
        self.bfast_active = False
        self.beta = 0.0
        self._symmetry = False
        self._mirrored = ()
        self._axis = ()
        self.__dict__.update(kwargs)

    def has_symmetry(self):
        return self._symmetry

    def is_mirrored(self, axis):
        return axis in self._mirrored

    def is_axis(self, axis):
        return axis in self._axis

    def has_metallic(self):
        return False

    def is_metallic(self, axis):
        return False


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


#: The configuration matrix both clause lists are run over. Each entry flips ONE
#: thing a clause reads, so a clause missing from either list shows up as a reason
#: present on one side only.
_MATRIX = (
    ("plain", {}, {}),
    ("complex_storage", {}, {"force_complex_fields": True}),
    ("nonlinear", {}, {"has_nonlinearity": True}),
    ("bloch_flag", {"has_bloch": True}, {}),
    ("bloch_kpoint", {"k_point": (0.3, 0.0, 0.0)}, {}),
    ("cylindrical", {"cylindrical": True}, {}),
    ("cylindrical_axis", {"_axis": (0,)}, {}),
    ("symmetry", {"_symmetry": True}, {}),
    ("mirrored_axis", {"_mirrored": (1,)}, {}),
    ("bfast", {"bfast_active": True}, {}),
    ("beta", {"beta": 0.7}, {}),
)


def _triton_backend_clause(reason: str) -> bool:
    """Triton's clause 1, and ONLY it.

    A prefix/suffix match, used HERE and nowhere else. In the implementation a
    match like this would be an inference from a literal and would silently miss a
    clause added later; in the test it is the subtraction the comparison needs,
    and any clause it fails to match stays in the set and fails the assertion —
    which is the behaviour that makes it safe here and unsafe there.
    """
    return reason.startswith("array module is ") and reason.endswith(", not cupy")


@pytest.mark.parametrize("label,grid_kwargs,fields_kwargs", _MATRIX,
                         ids=[entry[0] for entry in _MATRIX])
def test_grid_reasons_match_the_triton_clause_list(label, grid_kwargs, fields_kwargs):
    grid = _Grid(**grid_kwargs)
    fields = _Fields(grid, **fields_kwargs)
    pml = _PML()

    triton_reasons = set(triton_coverage._grid_reasons(fields, pml, grid))
    metal_reasons = set(metal_coverage._grid_reasons(fields, pml, grid))

    triton_rest = {r for r in triton_reasons if not _triton_backend_clause(r)}
    metal_backend = set(metal_coverage._metal_backend_reasons(grid))
    metal_rest = metal_reasons - metal_backend

    assert triton_rest == metal_rest, (
        f"{label}: the two clause lists disagree.\n"
        f"  only Triton has: {sorted(triton_rest - metal_rest)}\n"
        f"  only Metal has:  {sorted(metal_rest - triton_rest)}\n"
        "A clause added to one list and not the other is exactly the drift this "
        "pin exists to catch; hoist the shared body into a backend-neutral module "
        "rather than patching one side.")


def test_the_metal_backend_clause_is_the_only_difference_on_a_numpy_grid():
    """The subtraction the previous test performs must not be vacuous.

    On a NumPy grid Triton refuses ("not cupy") and Metal must refuse for its own
    named reasons — if both sides happened to produce identical strings the
    equivalence test would pass without ever exercising the replacement.
    """
    grid = _Grid()
    fields = _Fields(grid)
    triton_reasons = set(triton_coverage._grid_reasons(fields, _PML(), grid))
    assert any(_triton_backend_clause(r) for r in triton_reasons)
    metal_backend = metal_coverage._metal_backend_reasons(grid)
    # On this host torch and MPS are present, so the only backend clause left is
    # the subnormal-policy one (default resolves to keep on arm64) — or none, if
    # the caller requested flush. Either way it must not be Triton's string.
    assert not any(_triton_backend_clause(r) for r in metal_backend)


# ---------------------------------------------------------------------------
# 3. The single contraction-pragma spelling
# ---------------------------------------------------------------------------

def test_the_contraction_directive_is_spelled_exactly_once():
    """One spelling, in ``shaders``, and no second one anywhere in the package.

    A launch site — or a second source template — that quietly omitted it would
    produce a kernel that is bit-wrong and numerically plausible, which is the
    failure this package's whole gate exists to make impossible to ship.
    """
    occurrences = []
    for name, source in _package_sources():
        for number, line in enumerate(source.splitlines(), start=1):
            if "pragma clang fp" in line:
                occurrences.append(f"{name}:{number}")
    assert occurrences == ["shaders.py:%d" % _pragma_line()], (
        f"the contraction directive is spelled at {occurrences}; it must appear "
        "exactly once, in shaders.py, so no source can be emitted without it")


def _pragma_line() -> int:
    with open(os.path.join(PACKAGE_DIR, "shaders.py"), "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle.read().splitlines(), start=1):
            if "pragma clang fp" in line:
                return number
    raise AssertionError("the contraction directive is not spelled at all")


def test_no_math_mode_pragma_anywhere_in_the_package():
    """A math-mode pragma would re-enable exactly what the contraction guard removes."""
    banned = re.compile(r"pragma\s+(METAL|clang)\s+fp\s+(math_mode|denorm)")
    for name, source in _package_sources():
        assert not banned.search(source), f"{name} carries a math-mode pragma"


def test_every_shipped_source_carries_the_guard_in_off_mode():
    for label, source in shaders.enumerate_sources(shaders.CONTRACT_OFF).items():
        assert shaders.contraction_pragma(shaders.CONTRACT_OFF) in source, label


def test_the_fast_variant_really_differs_from_the_guarded_one():
    """The guard selector must select something. A no-op selector is a vacuous leg."""
    off = shaders.curl_source((0, 0, 0), False, shaders.CONTRACT_OFF)
    fast = shaders.curl_source((0, 0, 0), False, shaders.CONTRACT_FAST)
    assert off != fast
    assert shaders.source_sha256(off) != shaders.source_sha256(fast)


# ---------------------------------------------------------------------------
# The shader specialisation
# ---------------------------------------------------------------------------

def test_curl_source_bakes_the_direction_and_the_ghost_rule():
    forward = shaders.curl_source((shaders.PERIODIC,) * 3, False)
    backward = shaders.curl_source((shaders.PERIODIC,) * 3, True)
    assert "int si = i + 1" in forward and "int si = i - 1" in backward
    assert "(si == nxi) ? 0" in forward and "(si < 0) ? (nxi - 1)" in backward
    walled = shaders.curl_source((shaders.METALLIC, 0, 0), False)
    assert "vx = (si >= 0) && (si < nxi);" in walled
    assert "vx = (si >= 0)" not in forward


def test_the_ownership_mask_follows_the_yee_shift_table():
    """B masks one axis per target; D masks two. Getting this backwards is a plane
    of wrong values at every metallic wall, not a crash."""
    b_mask = shaders.curl_source((shaders.METALLIC,) * 3, False)
    d_mask = shaders.curl_source((shaders.METALLIC,) * 3, True)
    assert b_mask.count("? 0.0f : curl") == 3
    assert d_mask.count("? 0.0f : curl") == 6


def test_the_constitutive_product_puts_d_on_the_left():
    """stepping.py:1011-1013 writes ``source * inverse_epsilon``; float multiply is
    bitwise commutative and a transcription is not a place to rely on that."""
    assert "float src0 = g0[ii] * e0[ii];" in shaders.constitutive_source("E")
    assert "float src0 = g0[ii];" in shaders.constitutive_source("H")


def test_the_accumulation_forms_its_right_hand_side_first():
    """MEEP's ``f + ((kps*src) - (kms*prev))`` (step_generic.cpp:689, :697): the two
    products, their difference, then one add. The package's previous grouping,
    ``(f + kps*src) - kms*prev``, is a different float32 number and is gone."""
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        for t in range(3):
            assert (f"if (dw{t}) {{ a{t} = f{t}[ii] + ((kp_{t} * src{t}) - "
                    f"(km_{t} * prev{t})); }}") in source, (side, t)
            assert f"a{t} = a{t} + kp_{t} * src{t};" not in source
            assert f"a{t} = a{t} - km_{t} * prev{t};" not in source
        assert "kp_0 * src0 - km_0 * prev0" not in source


def test_the_curl_is_meeps_left_to_right_bracket_with_the_absent_partner_forms():
    """``stepping._curl_from_operands``: ``((g1s - g1) + g2) - g2s`` with both
    partners, ``dtdx * (g1s - g1)`` without g2, ``(-dtdx) * (g2s - g2)`` without g1,
    and an exact zero with neither, chosen on the record's presence bits."""
    for backward in (False, True):
        source = shaders.curl_source((0, 0, 0), backward)
        assert ("float curl0 = has_y ? (has_z ? (dtdx * (((c_y - c) + b) - b_z)) : "
                "(dtdx * (c_y - c)))\n                        : (has_z ? ((-dtdx) * "
                "(b_z - b)) : 0.0f);") in source
        assert ("float curl1 = has_z ? (has_x ? (dtdx * (((a_z - a) + c) - c_x)) : "
                "(dtdx * (a_z - a)))\n                        : (has_x ? ((-dtdx) * "
                "(c_x - c)) : 0.0f);") in source
        assert ("float curl2 = has_x ? (has_y ? (dtdx * (((b_x - b) + a) - a_y)) : "
                "(dtdx * (b_x - b)))\n                        : (has_y ? ((-dtdx) * "
                "(a_y - a)) : 0.0f);") in source
        # The package's previous pairwise grouping is gone, flattened or not.
        assert "dtdx * ((c_y - c) + (b - b_z))" not in source
        assert "dtdx * (c_y - c + b - b_z)" not in source
        for axis in "xyz":
            assert (f"const bool has_{axis} = (cb.cb_presence & "
                    f"{1 << 'xyz'.index(axis)}u) != 0u;") in source


def test_the_curl_terms_pair_the_operands_steppings_terms_name():
    """The kernel's operand and cycle tables are ``stepping``'s term tables, and its
    membership lattices are the Yee table's."""
    from meep_gpu import stepping
    from meep_gpu.fields import IYEE_SHIFTS

    for terms, sources, backward in ((stepping.B_CURL_TERMS, ("Ex", "Ey", "Ez"), False),
                                     (stepping.D_CURL_TERMS, ("Hx", "Hy", "Hz"), True)):
        for (first, first_axis, second, second_axis), (dsig, dsigu), term in zip(
                shaders.CURL_OPERANDS, shaders.CURL_CYCLE, terms):
            assert (sources[first], first_axis, sources[second], second_axis) == (
                term.first, term.first_axis, term.second, term.second_axis), term
            assert ("xyz"[dsig], "xyz"[dsigu]) == (term.dsig, term.dsigu), term
            assert IYEE_SHIFTS[term.target][dsigu] == int(
                shaders.CURL_HALF_INTEGER[backward]), term
    for side, terms in (("H", stepping.H_CONSTITUTIVE_TERMS),
                        ("E", stepping.E_CONSTITUTIVE_TERMS)):
        for name, _source, axis in terms:
            assert IYEE_SHIFTS[name]["xyz".index(axis)] == int(
                shaders.CONSTITUTIVE_HALF_INTEGER[side]), name


def test_prev_is_read_before_fw_is_written():
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        for t in range(3):
            assert (source.index(f"float prev{t} = dw{t} ? w{t}[ii] : 0.0f;")
                    < source.index(f"if (dw{t}) {{ w{t}[ii] = src{t}; }}"))


def _block(source: str) -> str:
    begin = source.index(chunk_bounds.BLOCK_BEGIN)
    end = source.index(chunk_bounds.BLOCK_END) + len(chunk_bounds.BLOCK_END)
    return source[source.rindex("\n", 0, begin) + 1:end]


def _flag_line(source: str, flag: str) -> str:
    lines = [line.strip() for line in source.splitlines()
             if line.strip().startswith(f"const bool {flag}0 = ")]
    assert len(lines) == 1, (flag, lines)
    return lines[0]


@pytest.mark.parametrize("codes", [(0, 0, 0), (1, 0, 1), (1, 1, 1)])
def test_the_bounds_block_is_the_anchored_emitter_and_names_one_flag_per_target(codes):
    for backward in (False, True):
        source = shaders.curl_source(codes, backward)
        assert source.count(chunk_bounds.BLOCK_BEGIN) == 1
        assert source.count(chunk_bounds.BLOCK_END) == 1
        assert _block(source) == shaders.curl_bounds_block(backward)
        tag = "h" if not backward else "i"
        assert _flag_line(source, "du") == (
            f"const bool du0 = in_{tag}_z, du1 = in_{tag}_x, du2 = in_{tag}_y;")
    for side, tag in (("H", "i"), ("E", "h")):
        source = shaders.constitutive_source(side)
        assert source.count(chunk_bounds.BLOCK_BEGIN) == 1
        assert _block(source) == shaders.constitutive_bounds_block(side)
        assert _flag_line(source, "dw") == (
            f"const bool dw0 = in_{tag}_x, dw1 = in_{tag}_y, dw2 = in_{tag}_z;")


def test_the_membership_each_form_reads_is_the_region_plans_own_line():
    """Each target's flag is the record entry on the (axis, lattice) the array path's
    region plan splits that component on: the dsigu axis for a curl target, the own
    axis for a constitutive component, on the component's Yee shift there."""
    from meep_gpu.chunk_map import ChunkLayout, RegionPlan
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    plan = RegionPlan(ChunkLayout.one_rank(grid, pml), grid, pml)
    for backward, names in ((False, ("Bx", "By", "Bz")), (True, ("Dx", "Dy", "Dz"))):
        flags = shaders.curl_membership_flags(backward)
        for target, name in enumerate(names):
            spec = plan.spec("curl", name)
            assert flags[target] == chunk_bounds.membership_name(
                spec.half_integer, spec.axis), name
    for side, names in (("H", ("Hx", "Hy", "Hz")), ("E", ("Ex", "Ey", "Ez"))):
        flags = shaders.constitutive_membership_flags(side)
        for target, name in enumerate(names):
            spec = plan.spec("constitutive", name)
            assert flags[target] == chunk_bounds.membership_name(
                spec.half_integer, spec.axis), name


def test_every_auxiliary_load_and_store_sits_behind_its_forms_membership():
    """The auxiliary contract (DESIGN D3) on the device text: ``fu`` is read and
    written only under ``du``, ``f_w`` only under ``dw``. A load outside the guard
    changes no word (a declared null on the gate), so this static needle is what
    pins it."""
    for backward in (False, True):
        source = shaders.curl_source((0, 0, 0), backward)
        for t in range(3):
            assert source.count(f"u{t}[ii]") == 2, (backward, t)
            assert f"float p{t} = du{t} ? u{t}[ii] : f{t}[ii];" in source
            assert f"if (du{t}) {{ u{t}[ii] = n{t}; }}" in source
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        for t in range(3):
            assert source.count(f"w{t}[ii]") == 2, (side, t)
            assert f"float prev{t} = dw{t} ? w{t}[ii] : 0.0f;" in source
            assert f"if (dw{t}) {{ w{t}[ii] = src{t}; }}" in source


def test_the_plain_forms_are_output_selects():
    """Outside the dsigu run the curl's word is the dsig-only form, the recurrence's
    first stage with ``fu := f`` (not ``p := 0``, which differs on one signed-zero
    pair); outside the dsigw run the constitutive's word is ``src`` itself (not the
    accumulation with ``kps = 1, kms = 0``, which turns every -0 into +0)."""
    for backward in (False, True):
        source = shaders.curl_source((0, 0, 0), backward)
        for t, (dsig, dsigu) in enumerate(shaders.CURL_CYCLE):
            a, b = "xyz"[dsig], "xyz"[dsigu]
            assert f"float n{t} = ((p{t} * km_{a}) - curl{t}) * si_{a};" in source
            assert (f"float v{t} = du{t} ? ((((f{t}[ii] * km_{b}) + n{t}) - p{t}) * "
                    f"si_{b}) : n{t};") in source
            assert f"du{t} ? u{t}[ii] : 0.0f" not in source
    for side in ("H", "E"):
        source = shaders.constitutive_source(side)
        for t in range(3):
            assert f"float a{t} = src{t};" in source
            assert f"f{t}[ii] = a{t};" in source
        assert "1.0f * src" not in source and "0.0f * prev" not in source


def test_the_record_binds_last_and_under_the_ceiling():
    from meep_gpu.metal_kernels import device

    for source, index in (
            [(shaders.curl_source((0, 0, 0), backward), shaders.CURL_RECORD_BUFFER)
             for backward in (False, True)]
            + [(shaders.constitutive_source(side), shaders.CONSTITUTIVE_RECORD_BUFFER)
               for side in ("H", "E")]):
        binding = chunk_bounds.binding(index)
        assert f"    {binding},\n    uint idx [[thread_position_in_grid]])" in source
        assert source.count("[[buffer(") == index + 1
        assert source.count("[[buffer(") <= device.MAX_BUFFER_BINDINGS
        assert chunk_bounds.struct_source() in source
        assert source.index(chunk_bounds.struct_source()) < source.index("kernel void")
    assert shaders.CURL_RECORD_BUFFER == 20 and shaders.CONSTITUTIVE_RECORD_BUFFER == 22


def test_the_shared_splice_anchors_survive_byte_for_byte():
    """The lines the lifting families splice on stay character for character, and the
    lifted body reads the record ONLY through the anchored bounds block's ``cb.``
    reads. So a family that lifts a body and carries its scalars in a ``Params``
    needs exactly the one-liner ``chunk_bounds.params_alias()`` before ``__BODY__``
    (plus the struct and the ``Params`` member) to bind it."""
    from meep_gpu.metal_kernels.fused_magnetic_pair import (
        _BODY_ANCHOR, _CONSTITUTIVE_PROLOGUE_END, _CURL_STORE)

    sources = ([shaders.curl_source(codes, backward)
                for codes in ((0, 0, 0), (1, 1, 1)) for backward in (False, True)]
               + [shaders.constitutive_source(side) for side in ("H", "E")])
    for source in sources:
        assert source.count(_BODY_ANCHOR) == 1
        body = source.split(_BODY_ANCHOR, 1)[1]
        block = _block(body)
        outside = body.replace(block, "")
        assert "cb." not in outside and "cb_presence" not in outside
        assert all(line.strip().startswith("const ") or line.strip().startswith("//")
                   for line in block.splitlines())
    for source in sources[:4]:
        assert source.count(_CURL_STORE) == 1
    for side, product in (("H", "g{t}[ii]"), ("E", "g{t}[ii] * e{t}[ii]")):
        source = shaders.constitutive_source(side)
        assert source.count(_CONSTITUTIVE_PROLOGUE_END) == 1
        assert (source.index(_CONSTITUTIVE_PROLOGUE_END)
                < source.index(chunk_bounds.BLOCK_BEGIN))
        for t in range(3):
            assert f"    float src{t} = {product.format(t=t)};\n" in source
            assert f"f{t}[ii]" in source
    assert chunk_bounds.params_alias() == "    constant ChunkBounds& cb = prm.cb;"


def test_the_specialisation_set_does_not_grow_with_the_division():
    """The record is runtime data: the labelled set is the same eighteen per mode, and
    no emitter takes a division argument."""
    import inspect

    for mode in shaders.CONTRACT_MODES:
        labels = set(shaders.enumerate_sources(mode))
        assert len(labels) == 18
        assert sum(label.startswith("pml_curl_step/") for label in labels) == 16
    for emitter in (shaders.curl_source, shaders.constitutive_source):
        assert set(inspect.signature(emitter).parameters) <= {
            "codes", "backward", "side", "contract"}


def test_the_harness_routes_take_no_default_record():
    """Every route that builds a base plan states which division it binds: the
    record is a keyword with no default, and a missing one is refused by name."""
    import inspect

    from meep_gpu.chunk_map import ChunkLayoutRefusal
    from meep_gpu.metal_kernels import launch, nonlinear_update_e, symmetry

    for builder in (launch.plan_from_arrays, launch.plan_constitutive_from_arrays,
                    launch.plan_pml_curl, launch.plan_constitutive,
                    launch.PmlCurlPlan.__init__, launch.ConstitutivePlan.__init__,
                    symmetry.plan_folded_pml_curl,
                    symmetry.plan_folded_pml_curl_from_arrays,
                    symmetry.plan_folded_constitutive,
                    symmetry.FoldedPmlCurlPlan.__init__,
                    nonlinear_update_e.plan_nonlinear_run_pml_curl,
                    nonlinear_update_e.plan_nonlinear_run_constitutive):
        parameter = inspect.signature(builder).parameters["chunk_bounds"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, builder
        assert parameter.default is inspect.Parameter.empty, builder
    with pytest.raises(ChunkLayoutRefusal, match="PmlCurlPlan.*no chunk record"):
        chunk_bounds.plan_words(None, (4, 3, 2), "PmlCurlPlan(step_B)")


def test_a_plan_record_that_disagrees_with_its_shape_is_refused_by_name():
    from meep_gpu.chunk_map import (RECORD_PRESENCE_WORD, RECORD_WORDS, ChunkLayoutRefusal,
                                    record_index)

    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    record[RECORD_PRESENCE_WORD] = 0b011
    record[record_index(False, 0, 1)] = 2
    assert chunk_bounds.plan_words(record, (4, 3, 1), "owner").tolist() == record.tolist()
    with pytest.raises(ChunkLayoutRefusal, match="owner: the chunk record marks z invariant"):
        chunk_bounds.plan_words(record, (4, 3, 2), "owner")
    with pytest.raises(ChunkLayoutRefusal, match="owner: .*reach row 2"):
        chunk_bounds.plan_words(record, (1, 3, 1), "owner")
    stray = record.copy()
    stray[record_index(True, 2, 1)] = 1
    with pytest.raises(ChunkLayoutRefusal, match="invariant but holds a run"):
        chunk_bounds.plan_words(stray, (4, 3, 1), "owner")


def test_a_plan_binds_the_division_the_array_path_steps_the_same_fields_with():
    """The composer's record is the record of the region plan ``stepping`` steps the
    same ``Fields`` with: the one-rank division when none is installed, and the
    installed one otherwise."""
    from meep_gpu import stepping
    from meep_gpu.chunk_map import ChunkLayout, RegionPlan, chunk_bounds_record
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    layout, words, refused = launch.chunk_division(fields, pml)
    assert not refused
    plan = stepping._region_plan(fields, pml)
    assert np.array_equal(words, chunk_bounds_record(plan.layout))
    assert np.array_equal(words, chunk_bounds.for_layout(layout))

    # A division the driver installed is the one both read.
    installed = ChunkLayout.one_rank(fields.grid, pml)
    fields.install_region_plan(RegionPlan(installed, fields.grid, pml))
    _, words, refused = launch.chunk_division(fields, pml, installed)
    assert not refused
    assert np.array_equal(words, chunk_bounds_record(stepping._region_plan(fields,
                                                                         pml).layout))


class _Source:
    def __init__(self, field_type):
        self.field_type = field_type


def test_the_first_update_h_is_refused_where_the_h_start_could_move_a_word():
    """``stepping.separate_h_from_b`` starts H and f_w from B at the run's first
    update_H and the base kernel does not; the composer leaves that update to the
    array path unless B is +0 on every cell at that moment."""
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    for name in launch.H_START_VOLUMES:
        getattr(fields, name)[...] = 0.0
    fields.h_separated = False
    assert launch.h_start_reasons(fields, ()) == ()
    assert launch.h_start_reasons(fields, (_Source("E"),)) == ()
    assert any("magnetic source(s) [1]" in r
               for r in launch.h_start_reasons(fields, (_Source("E"), _Source("B"))))
    assert any("not declared" in r for r in launch.h_start_reasons(fields, None))
    fields.Ey[1, 1, 1] = np.float32(-0.0)
    assert any("Ey holds a word other than +0" in r
               for r in launch.h_start_reasons(fields, ()))
    fields.h_separated = True
    assert launch.h_start_reasons(fields, (_Source("B"),)) == ()


class _MagneticSource:
    def __init__(self, component):
        self.field_type = "B"
        self.component = component


def _started_2d_case():
    """A 2-D xy engine case (PML on x and y, z invariant) at a run's first update_H."""
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case(cell=(1.2, 1.0, 0.0))
    for name in launch.H_START_VOLUMES + ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy",
                                          "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = 0.0
    fields.h_separated = False
    return fields, pml


def test_the_start_writes_nothing_on_a_component_whose_layer_no_chunk_carries():
    """The premise the narrowed clause rests on, measured on the array path's own start.

    ``stepping.separate_h_from_b`` copies B into H and ``f_w`` only on the cells whose
    chunk carries H's own layer. On a 2-D xy run z carries none, so a deposit into Bz
    moves no started word; a deposit into Bx inside x's layer does (the planted
    negative the clause must still refuse).
    """
    from meep_gpu import stepping

    fields, pml = _started_2d_case()
    region = stepping._region_plan(fields, pml)  # noqa: SLF001
    assert region.spec("constitutive", "Hz").mode == "outside_only"
    assert region.spec("constitutive", "Hx").mode != "outside_only"
    inside = np.broadcast_to(region.mask("constitutive", "Hx", np), fields.Hx.shape)
    cell = tuple(int(axis[0]) for axis in np.nonzero(inside))
    fields.Bz[3, 3, 0] = np.float32(0.5)
    fields.Bx[cell] = np.float32(0.5)
    names = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    before = {name: np.array(getattr(fields, name), copy=True) for name in names
              if getattr(fields, name, None) is not None}
    stepping.separate_h_from_b(fields, pml)
    moved = {name: int(np.count_nonzero(getattr(fields, name).view(np.uint32)
                                        != value.view(np.uint32)))
             for name, value in before.items()}
    assert moved.get("Hz", 0) == 0 and moved.get("f_w_Hz", 0) == 0, moved
    assert moved["Hx"] == 1 and moved["f_w_Hx"] == 1, moved


def test_the_layer_narrows_the_magnetic_clause_to_the_components_the_start_copies():
    """Given the layer, a magnetic source refuses only where its B component's H
    carries cells in its own layer. The composer passes it for the base update_H
    kernel, whose refusal is run-long; the products that ask without it refuse every
    magnetic source, as before."""
    import inspect

    from meep_gpu.metal_kernels import launch

    fields, pml = _started_2d_case()
    assert launch.h_start_reasons(fields, (_MagneticSource("Hz"),), pml) == ()
    assert launch.h_start_reasons(fields, (_MagneticSource("Bz"),), pml) == ()
    refused = launch.h_start_reasons(
        fields, (_Source("E"), _MagneticSource("Hx")), pml)
    assert any("magnetic source(s) [1]" in r and "Hx's own layer" in r
               for r in refused), refused
    assert any("magnetic source(s) [0]" in r for r in launch.h_start_reasons(
        fields, (_MagneticSource("Hz"),)))
    # A source that names no magnetic component cannot be cleared: it refuses.
    assert launch.h_start_reasons(fields, (_Source("B"),), pml)
    # The volumes clause still stands under the layer.
    fields.Ey[1, 1, 0] = np.float32(-0.0)
    assert any("Ey holds a word other than +0" in r for r in launch.h_start_reasons(
        fields, (_MagneticSource("Hz"),), pml))
    assert "h_start_reasons(fields, sources, pml)" in inspect.getsource(launch.plan_step)


def test_an_unknown_boundary_code_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        shaders.curl_source((2, 0, 0), False)
    with pytest.raises(ValueError):
        shaders.curl_source((0, 0), False)
    with pytest.raises(ValueError):
        shaders.constitutive_source("B")
    with pytest.raises(ValueError):
        shaders.contraction_pragma("relaxed")


# ---------------------------------------------------------------------------
# The residency invariant
# ---------------------------------------------------------------------------

_ALL_FIELD_VOLUMES = tuple(sorted({
    name for spec in metal_coverage.SUB_STEP_VOLUMES.values()
    for name in tuple(spec["writes"]) + tuple(spec["reads"])}))


def test_residency_admits_a_fully_planned_step():
    planned = ("step_B", "update_H", "step_D", "update_E")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, planned)
    assert verdict.covered, verdict.reasons


def test_residency_refuses_an_array_path_wall_clear_between_launches():
    """The defect with no Triton analogue: ``zero_metal_B`` writes the HOST Bx
    while the mirror holds the value the curl just computed."""
    planned = ("step_B", "update_H", "step_D", "update_E")
    live = planned + ("zero_metal_B", "zero_metal_D")
    verdict = metal_coverage.residency_coverage(_ALL_FIELD_VOLUMES, planned, live)
    assert not verdict.covered
    assert any("zero_metal_B" in reason for reason in verdict.reasons)


def test_residency_admits_the_same_wall_clear_once_it_is_declared_synced():
    planned = ("step_B", "update_H", "step_D", "update_E")
    live = planned + ("zero_metal_B", "zero_metal_D")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, live, ("zero_metal_B", "zero_metal_D"))
    assert verdict.covered, verdict.reasons


def test_residency_refuses_an_array_path_update_p_reading_the_drive_field():
    """``update_P`` reads ``f_w_*`` under PML (fields.py:1140-1163), which the
    constitutive mirror writes — the one dispersive read that touches it."""
    planned = ("step_B", "update_H", "step_D", "update_E")
    verdict = metal_coverage.residency_coverage(
        _ALL_FIELD_VOLUMES, planned, planned + ("update_P",))
    assert not verdict.covered
    assert any("update_P" in reason for reason in verdict.reasons)


def test_residency_refuses_an_undeclared_set_rather_than_assuming_an_empty_one():
    for mirrored, planned, live in ((None, (), ()), ((), None, ()), ((), (), None)):
        verdict = metal_coverage.residency_coverage(mirrored, planned, live)
        assert not verdict.covered
        assert any("not declared" in reason for reason in verdict.reasons)


def test_residency_refuses_a_planned_sub_step_whose_volumes_are_not_mirrored():
    verdict = metal_coverage.residency_coverage(("Bx",), ("step_B",), ("step_B",))
    assert not verdict.covered
    assert any("carry no mirror" in reason for reason in verdict.reasons)


def test_residency_order_covers_every_sub_step_with_a_volume_table():
    assert set(metal_coverage.RESIDENCY_ORDER) == set(metal_coverage.SUB_STEP_VOLUMES)


# ---------------------------------------------------------------------------
# Per-clause refusals on constructed objects
# ---------------------------------------------------------------------------

def test_a_plan_with_no_declared_residency_is_refused_by_name():
    grid = _Grid()
    fields = _Fields(grid)
    verdict = metal_coverage.pml_curl_coverage(fields, _PML(), "step_B", None)
    assert not verdict.covered
    assert any("residency was not declared" in reason for reason in verdict.reasons)


def test_an_inactive_absorber_is_refused_on_both_predicates():
    grid = _Grid()
    fields = _Fields(grid)
    for verdict in (metal_coverage.pml_curl_coverage(fields, _PML(False), "step_B"),
                    metal_coverage.constitutive_coverage(fields, _PML(False), "H")):
        assert any("no active PML layer" in reason for reason in verdict.reasons)


def test_the_e_side_refuses_a_registered_polarization():
    grid = _Grid()
    fields = _Fields(grid, has_polarizations=True)
    verdict = metal_coverage.constitutive_coverage(fields, _PML(), "E")
    assert any("D - sum P" in reason for reason in verdict.reasons)


def test_an_unknown_sub_step_or_side_raises_rather_than_refusing_quietly():
    grid = _Grid()
    fields = _Fields(grid)
    with pytest.raises(ValueError):
        metal_coverage.pml_curl_coverage(fields, _PML(), "step_Q")
    with pytest.raises(ValueError):
        metal_coverage.constitutive_coverage(fields, _PML(), "Q")


# ---------------------------------------------------------------------------
# The subnormal executor column
# ---------------------------------------------------------------------------

def test_keep_is_a_refusal_and_flush_is_admitted(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    reasons = subnormal.mps_policy_reasons()
    assert reasons and "cannot honour" in reasons[0]
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    assert subnormal.mps_policy_reasons() == []


def test_the_policy_report_records_the_absent_disassembly_audit(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    report = subnormal.mps_policy_report()
    assert report["resolved"] == subnormal.FLUSH
    assert report["admitted"] is True
    assert report["ptx_equivalent_audit"] is None
    assert "no disassembly" in report["ptx_equivalent_audit_note"]


@pytest.mark.usefixtures("host_keeps_subnormals")  # host words in the subnormal band
def test_the_subnormal_census_counts_bits_not_values():
    band = np.array([1.401298464324817e-45, 0.0, -0.0, 1.0], dtype=np.float32)
    assert subnormal.census(band) == 1
    zeros = subnormal.signed_zero_census(band)
    assert zeros == {"negative_zero": 1, "positive_zero": 1}


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def test_the_checked_in_fingerprints_match_the_tree():
    from meep_gpu.metal_kernels import launch

    stored = launch.load_fingerprints()["metal_kernels"]
    computed = launch.compute_fingerprints()["metal_kernels"]
    assert stored["kernel_source_sha256"] == computed["kernel_source_sha256"], (
        "a shipped kernel source changed without the fingerprint being re-cut; "
        "that is a correctness event, not a formatting one — re-run "
        "launch.write_fingerprints() DELIBERATELY and re-cut the gate")
    assert stored["host_sha256"] == computed["host_sha256"], (
        "a host module changed: it chooses the Yee sub-lattice, binds the "
        "coefficient pointers and decides the specialisation constants")


def test_the_fingerprint_file_pins_a_toolchain_it_measured():
    from meep_gpu.metal_kernels import launch

    with open(launch.fingerprints_path(), "r", encoding="utf-8") as handle:
        stored = json.load(handle)["metal_kernels"]
    assert stored["validated_torch_versions"], "no torch version was pinned"
    frontend = stored["validated_metal_frontend"]
    assert frontend is None or frontend.startswith("metalfe-")


# ---------------------------------------------------------------------------
# The arithmetic, in process, against stepping.py
# ---------------------------------------------------------------------------

def _engine_case(boundaries=("periodic", "periodic", "periodic"), courant=0.35,
                 seed=20260814, cell=(1.2, 1.0, 0.9)):
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    dimensions = 3 if cell[2] else 2
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=courant, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    eps = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(eps, (np.float32(1.0) / eps).astype(np.float32))
    pml = PML(grid=grid, thickness=tuple((2, 2) if cell[axis] else (0, 0)
                                         for axis in range(3)))
    rng = np.random.default_rng(seed)
    for name in _STATE:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
    # A SEEDED STATE IS A MID-RUN ONE: H was started from B long ago.
    fields.h_separated = True
    return grid, fields, pml


def _one_rank_record(fields, pml):
    """The record of the one-rank division ``stepping`` steps these ``Fields`` with."""
    from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record

    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(array):
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


@requires_mps
@pytest.mark.parametrize("cell", [(1.2, 1.0, 0.9), (1.2, 1.0, 0.0)],
                         ids=["3d", "2d_absent_partner"])
def test_one_complete_cycle_is_byte_identical_to_stepping(monkeypatch, cell):
    """Four sub-steps on the device against four on the array path, in one process.

    uint32 word equality on the targets AND the auxiliaries, and an assertion that
    the step MOVED STATE — a no-op agreeing with a no-op is trivially identical. The
    2-D cell carries an invariant z, so the curl takes its absent-partner forms.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu import stepping
    from meep_gpu.metal_kernels import launch

    grid, fields, pml = _engine_case(cell=cell)
    before = {name: np.array(getattr(fields, name), copy=True) for name in _STATE}

    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=())
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.residency.covered, plan.residency.reasons
    for name in plan.replaces:
        plan.plans[name].run()
    residency.sync_out()
    device = {name: np.array(getattr(fields, name), copy=True) for name in _STATE}

    for name, value in before.items():
        getattr(fields, name)[...] = value
    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.update_E(fields, pml)

    moved = sum(int(np.count_nonzero(_words(getattr(fields, n)) != _words(before[n])))
                for n in _STATE)
    assert moved > 0, "VACUOUS: the array path moved no state"
    differing = {n: int(np.count_nonzero(_words(device[n]) != _words(getattr(fields, n))))
                 for n in _STATE}
    assert not {n: c for n, c in differing.items() if c}, differing


@requires_mps
def test_the_residency_mirrors_match_the_host_after_a_synced_step(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=())
    for name in plan.replaces:
        plan.plans[name].run()
    residency.sync_out()
    assert residency.names, "VACUOUS: an empty mirror registry verifies trivially"
    assert residency.verify() == {}


@requires_mps
def test_asking_for_an_unbuilt_contract_variant_raises(monkeypatch):
    """A guard selector that silently fell back to the pinned source would make the
    gate's guard leg certify nothing."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    plan = launch.plan_pml_curl(fields, pml, "step_B", residency,
                                chunk_bounds=_one_rank_record(fields, pml))
    with pytest.raises(KeyError):
        plan.run(contract=shaders.CONTRACT_FAST)


@requires_mps
def test_two_plans_share_one_mirror_rather_than_holding_private_copies(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    residency = launch.Residency()
    record = _one_rank_record(fields, pml)
    curl = launch.plan_pml_curl(fields, pml, "step_B", residency, chunk_bounds=record)
    constitutive = launch.plan_constitutive(fields, pml, "H", residency,
                                            chunk_bounds=record)
    assert curl is not None and constitutive is not None
    assert residency.tensor("Bx").data_ptr() == residency.tensor("Bx").data_ptr()
    # step_B writes Bx; update_H reads it. Same allocation or the second launch
    # reads the first one's stale bytes.
    assert any(t.data_ptr() == residency.tensor("Bx").data_ptr()
               for t in curl._args if hasattr(t, "data_ptr"))
    assert any(t.data_ptr() == residency.tensor("Bx").data_ptr()
               for t in constitutive._args if hasattr(t, "data_ptr"))


@requires_mps
def test_the_keep_policy_refuses_a_plan_at_build_time(monkeypatch):
    """The awkward default, pinned: this arm64 host resolves to keep, which MPS
    cannot honour, so a plan built without an explicit flush request is refused."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    from meep_gpu.metal_kernels import launch

    _, fields, pml = _engine_case()
    verdict = metal_coverage.pml_curl_coverage(fields, pml, "step_B",
                                               launch.Residency())
    assert not verdict.covered
    assert any("cannot honour" in reason for reason in verdict.reasons)
