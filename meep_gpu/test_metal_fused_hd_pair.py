"""Laptop tests for the Metal H->D weld (``update_H`` welded into ``step_D``).

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim owed by this family is a device
gate that runs complete driver steps and compares uint32 words; these tests are what
must stay green on every change, and they are chosen for the defects a byte gate would
catch LATE, or on a configuration nobody sweeps, or not at all:

* **the lift.** Both halves are SPLICED from the certified emitters' own output rather
  than retyped, and "spliced" is a hypothesis until something compares the strings. The
  tests below reconstruct the certified bodies independently and assert that the ONLY
  lines that moved are the ones :data:`.fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` names,
  and that every arithmetic line -- the two accumulations, the curl's parenthesisation,
  the split-field recurrence -- survives character for character;
* **the redirect.** Six magnetic loads become recomputes, and a redirect that landed on
  the wrong cell is a smooth, plausible, entirely wrong answer. The coordinates are
  derived from the emitter's own index lines, so the test asserts the derivation rather
  than a table;
* **the ghost.** The branch and the exact ``0.0f`` past a metallic wall are what a
  redirect is most likely to eat, and a walled run that quietly recomputed the ghost
  cell converges to a slightly different absorber;
* **the binding ceiling**, which is an EQUALITY here: 31 of 31, no headroom. A future
  edit that adds a volume discovers it in this file rather than at compile time;
* **the predicate's refusals**, every one of which is answerable without a GPU;
* **the composition.** This product is registered UNWIRED, holds a ``FUSED_PAIR_ARMS``
  row read off its two predicates, and declares ``INSTALLABLE = False`` on a measured
  verdict. Until 2026-09-05 that flag was load-bearing for a RELEASED product's slots:
  without it, registering this row made ``launch._neighbouring_seam_claimant`` refuse
  ``fused_magnetic_pair`` on every row it reached. It is not any more -- the B->H span
  is an END EDGE of the seam table and is never asked -- and that is driven here from
  BOTH sides: the released pair keeps its slot whatever this flag says, and this
  product is refused by ``_pair_may_absorb`` when the flag is out of the way;
* **the nonlinear widening.** The chi2/chi3 cell is admitted through the spine arms
  of ``nonlinear_update_e``, which build the ordinary kernels; the text identity is
  measured on device and the delegation is read off the source;
* **the scratch and the rotation**, both of which are pure host logic and are driven
  against a fake residency, so the disjointness refusal and the buffer swap are measured
  on a machine with no device at all.

Everything that needs a device is guarded and skipped, so this file is the merge bar on
a host with no MPS as well as on this one.

WHAT THIS FILE DOES NOT CLAIM. Nothing here is a byte-identity measurement, and nothing
here licenses installing the product. The seam's own numbers -- 49 rows in the
``(update_H ordinary -> step_D PML)`` cell, of which this predicate reaches 47 and
refuses 2 by name -- come from
``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl`` and are asserted
against that record where it is present rather than restated.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from meep_gpu import withdraw_hoist
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import (
    arms,
    fused_hd_pair as hd,
    launch as metal_launch,
    offdiag_weld_common as weld,
    shaders,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.pml import PML
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES


def _torch_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


needs_mps = pytest.mark.skipif(not _torch_mps(),
                               reason="no MPS device on this host")

#: Every specialisation the shipped plan can emit: eight boundary triples in two
#: contraction modes. The string tests run over ALL of them rather than a
#: representative one, because the ghost rule is what specialisation changes.
CODES = tuple(itertools.product((0, 1), repeat=3))
MODES = shaders.CONTRACT_MODES


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_span_is_the_seam_the_withdraw_module_owns():
    """``REPLACES`` and ``SEAM`` are read from :mod:`meep_gpu.withdraw_hoist`, not typed.

    A product whose span disagreed with that module's ``SEAM_SPAN`` would be refused by
    ``hoistable``'s span clause the moment it ever declared the hoist -- but only then.
    Pinning the equality here means the disagreement is a red now rather than a silent
    admission later.
    """
    assert hd.REPLACES == withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")
    assert hd.SEAM == withdraw_hoist.SEAM == "H_to_D"
    assert hd.SLOT == "update_H", "the arm sits on the slot the driver reaches first"
    assert hd.SLOT == hd.REPLACES[0]


def test_the_two_seam_declarations_are_both_false_and_say_why():
    """``CARRIES_DEPOSIT_REPAIR`` and ``HOISTS_THE_WITHDRAW`` are separate claims.

    Keeping them apart is the whole reason ``withdraw_hoist`` is a sibling of
    ``deposit_repair`` rather than a branch of it: a product that declared one must not
    thereby acquire the other's licence. Both are False here, for two DIFFERENT reasons
    -- the first because nothing is injected in this seam at all, the second because the
    wiring that would perform the hoist is unreachable for an uninstallable product.
    """
    assert hd.CARRIES_DEPOSIT_REPAIR is False
    assert hd.HOISTS_THE_WITHDRAW is False
    assert hd.INSTALLABLE is False
    assert len(hd.INSTALLABLE_REASON) > 200, (
        "the uninstallable declaration is reported on every configuration; it has to "
        "carry the measurement, not a shrug")
    for needle in ("4 - (installed pairs)", "133 rows", "22 rows", "TIE",
                   "h_to_d_seam_2026-09-04", "_pair_may_absorb", "3rd-harm-1d",
                   "0 rows on which displacing the released B->H pair is a gain"):
        assert needle in hd.INSTALLABLE_REASON, needle
    # The deposit clause is not merely answered False, it is not ASKED: there is no
    # injection between the two consults, so consulting `deposit_repair` here would be
    # asking a question about the NEXT seam, whose electric list `_in_seam_indexed`
    # would select for any seam name that is not the letter 'B'. Read as CALLS rather
    # than as a substring, so the module may still explain in prose why it does not.
    import ast
    called = {node.func.attr for node in ast.walk(ast.parse(
        open(hd.__file__, "r", encoding="utf-8").read()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "seam_source_reasons" not in called, (
        "this seam carries no injection; deposit_repair's clause would select the "
        "ELECTRIC list, which the driver injects one seam later")
    assert "seam_withdraw_reasons" in called


def test_the_family_is_welded_and_says_so_in_the_one_place_that_partitions_the_fleet():
    """WELD_OWED is EMPTY, and the weld entry it is the complement of exists.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the
    ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires the welded set
    and the ``WELD_OWED`` set to be disjoint and to exhaust it. So the two halves of a
    release move together, and the failure each half prevents is different:

    * emptying ``WELD_OWED`` without the ledger entry claims a released gate this tree
      cannot show -- the family would simply vanish from both sides of the partition;
    * cutting the ledger entry without emptying ``WELD_OWED`` leaves a welded family
      still declaring its weld is owed, which is the state that makes an honest
      withdrawal indistinguishable from a stale one.

    Both are checked here, off the tree, rather than either being taken on trust. The
    withdrawal this replaced (2026-09-05 through 2026-09-06) named the artifact that
    refused, the leg that failed and the two rows whose divergence was then
    unattributed; the module docstring now carries the attribution instead.
    """
    assert hd.WELD_OWED == "", (
        "the family holds a released device gate; WELD_OWED must be empty, and the "
        "weld contract's partition is what reads it")
    ledger = json.loads((Path(hd.__file__).with_name("fingerprints.json"))
                        .read_text(encoding="utf-8"))
    assert "metal_fused_hd_pair_device_gate" in ledger, (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_fused_hd_pair_device_gate: one of the two is false")
    # AND THE COMPOSER'S OWN REFUSAL STRING MUST NOT STILL POINT AT A WITHDRAWAL. It is
    # where a reader meets this family on every run, and it is the string the board
    # quotes; a live pointer to a constant that is now empty reads as a family whose
    # certification is still owed.
    assert "WELD_OWED" not in hd.INSTALLABLE_REASON
    assert "gate certifies" in hd.INSTALLABLE_REASON
    assert "metal_fused_hd_pair_2026-09-06_hdland" in hd.INSTALLABLE_REASON


# ---------------------------------------------------------------------------
# The lift: the constitutive half
# ---------------------------------------------------------------------------

def _certified_constitutive_tail(mode):
    """The certified ``update_H`` body below the decode prologue, UNEDITED."""
    body = shaders.constitutive_source("H", mode).split(
        "uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
    return body[: -len("}\n")].split(weld.DECODE_END, 1)[1]


@pytest.mark.parametrize("mode", MODES)
def test_the_constitutive_lift_moves_exactly_the_declared_lines(mode):
    """Line for line: what left, what arrived, and nothing else.

    The certified body and the lifted one are compared as MULTISETS OF LINES. Every
    line that disappeared must be a store or a rename source named in
    :data:`.fused_hd_pair.CONSTITUTIVE_LIFT_EDITS`, and every line that arrived must be
    its stated replacement. An arithmetic line that changed would show up in both sets
    and fail, which is the property this test exists for -- a splice is only a splice if
    the strings agree.
    """
    certified = _certified_constitutive_tail(mode).splitlines()
    lifted = hd.certified_constitutive_tail(mode).splitlines()
    removed = [line for line in certified if line not in lifted]
    added = [line for line in lifted if line not in certified]

    expected_removed = []
    expected_added = []
    for target, axis, coefficient in ((0, "i", "kmx"), (1, "j", "kmy"), (2, "k", "kmz")):
        expected_removed.append(
            f"    float kp_{target} = kp{target}[{axis}], "
            f"km_{target} = km{target}[{axis}];")
        expected_added.append(
            f"    float kp_{target} = kp{target}[{axis}], "
            f"km_{target} = {coefficient}[{axis}];")
        expected_removed += [f"    float prev{target} = w{target}[ii];",
                             f"    float src{target} = g{target}[ii];",
                             f"    w{target}[ii] = src{target};",
                             f"    float a{target} = f{target}[ii];",
                             f"    f{target}[ii] = a{target};"]
        expected_added += [f"    float prev{target} = wi{target}[ii];",
                           f"    float src{target} = b{target}[ii];",
                           f"    float a{target} = hi{target}[ii];"]
    assert sorted(removed) == sorted(expected_removed), sorted(removed)
    assert sorted(added) == sorted(expected_added), sorted(added)


@pytest.mark.parametrize("mode", MODES)
def test_the_two_accumulations_survive_the_lift_character_for_character(mode):
    """``((f + kps*src) - kms*prev)`` as two separate statements, left to right.

    Flattening it is a different float32 number and neither multiply may contract into
    an fma; the pragma stops the second and this stops the first. Asserted on the LIFTED
    text, where a rename could plausibly have reached the right-hand side.
    """
    lifted = hd.certified_constitutive_tail(mode)
    for target in range(3):
        assert f"    a{target} = a{target} + kp_{target} * src{target};\n" in lifted
        assert f"    a{target} = a{target} - km_{target} * prev{target};\n" in lifted
    assert lifted.count(" = a0 + ") == 1 and lifted.count(" = a0 - ") == 1


@pytest.mark.parametrize("mode", MODES)
def test_the_lifted_body_stores_nothing_and_reads_only_unwritten_state(mode):
    """THE PROPERTY THE WHOLE DESIGN RESTS ON.

    A recompute is a pure function of memory this launch does not write, so the lifted
    body must contain no store at all and must index only the three ``const`` groups.
    A single surviving store would make the foreign tap a read of another thread's
    output -- the hazard the scratch output removes.
    """
    lifted = hd.certified_constitutive_tail(mode)
    for line in lifted.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or not stripped:
            continue
        if "[ii]" in stripped and "=" in stripped:
            left = stripped.split("=", 1)[0]
            assert "[ii]" not in left, (
                f"the lifted update_H body still stores through {left.strip()!r}; a "
                f"foreign recompute must write nothing")
    for stem in ("f", "w", "g"):
        for target in range(3):
            assert f"{stem}{target}[" not in lifted, (stem, target)
    for stem in ("hi", "wi", "b"):
        assert all(f"{stem}{target}[ii]" in lifted for target in range(3)), stem


# ---------------------------------------------------------------------------
# The lift: the curl half and the redirect
# ---------------------------------------------------------------------------

def _certified_curl_tail(codes, mode):
    body = shaders.curl_source(codes, True, mode).split(
        "uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
    return body[: -len("}\n")].split(weld.DECODE_END, 1)[1]


@pytest.mark.parametrize("codes", CODES)
def test_the_curl_lift_moves_exactly_the_nine_magnetic_loads(codes):
    """Three own-cell loads become registers, six shifted loads become recomputes.

    Nothing else in ``step_D``'s body may move: the ghost rule, the index composition,
    the curl expression, the ownership mask and the split-field recurrence are all the
    certified emitter's, and a lift that touched one of them would be new arithmetic
    wearing a transcription's argument.
    """
    certified = _certified_curl_tail(codes, shaders.CONTRACT_OFF).splitlines()
    welded = hd.welded_curl_tail(codes, shaders.CONTRACT_OFF).splitlines()
    assert len(certified) == len(welded), "the lift added or dropped a line"
    moved = [(a, b) for a, b in zip(certified, welded) if a != b]
    assert len(moved) == 9, [a for a, _ in moved]
    for before, after in moved:
        assert "g0[" in before or "g1[" in before or "g2[" in before, before
        assert "own.a" in after or "h_cell(" in after, after


@pytest.mark.parametrize("codes", CODES)
def test_every_recompute_lands_on_the_cell_the_certified_index_composed(codes):
    """The redirect's coordinates are DERIVED from the emitter, and this checks it.

    ``h_cell`` composes ``ii = i*nyz + j*nzi + k`` from its three coordinate parameters;
    the certified curl composes ``ox = si*nyz + j*nzi + k`` for the same layout. So a
    tap that read ``g1[ox]`` must become ``h_cell(si, j, k, ...)``, and this reconstructs
    that pairing from both sources rather than trusting either.
    """
    certified = _certified_curl_tail(codes, shaders.CONTRACT_OFF)
    welded = hd.welded_curl_tail(codes, shaders.CONTRACT_OFF)
    offsets = hd.offset_coordinates(certified)
    assert offsets == {"ox": ("si", "j", "k"), "oy": ("i", "sj", "k"),
                       "oz": ("i", "j", "sk")}, offsets
    # And the helper composes its index by the same layout, spelled once in the
    # shared emitter.
    assert "    int ii = i * nyz + j * nzi + k;" in hd.h_cell_function()
    for line in certified.splitlines():
        if " ? g" not in line:
            continue
        register = line.split("float ", 1)[1].split(" =", 1)[0]
        target = int(line.split(" ? g", 1)[1][0])
        offset = line.split(f" ? g{target}[", 1)[1].split("]", 1)[0]
        expected = (f"h_cell({', '.join(offsets[offset])}, {hd.H_CELL_TAIL_ARGS})"
                    f".a{target}")
        replacement = next(other for other in welded.splitlines()
                           if other.startswith(f"    float {register} = "))
        assert expected in replacement, (register, replacement)
        # The guard is the emitter's own, unchanged, and so is the exact zero.
        assert replacement.split(" ? ", 1)[0] == line.split(" ? ", 1)[0]
        assert replacement.endswith(" : 0.0f;")


@pytest.mark.parametrize("codes", CODES)
@pytest.mark.parametrize("mode", MODES)
def test_no_magnetic_pointer_survives_anywhere_in_the_fused_kernel(codes, mode):
    """``g0``/``g1``/``g2`` do not exist in this signature at all.

    One missed redirect would not be a compile error in a kernel that HAD those
    pointers; here it is, which is a weaker guarantee than it looks -- the emitted text
    is what the gate hashes, so it is asserted directly.
    """
    source = hd.fused_hd_pair_source(codes, mode)
    for target in range(3):
        assert f"g{target}[" not in source
    assert source.count("h_cell(") == 1 + 6 + 1, (
        "one own-cell call, six halo recomputes, and the function's own definition")
    for target, register in enumerate(("a", "b", "c")):
        assert f"    float {register}   = own.a{target};\n" in source


@pytest.mark.parametrize("codes", CODES)
def test_the_ghost_and_the_wrap_are_the_emitters_own(codes):
    """A metallic axis still serves an exact ``0.0f``; a periodic one still wraps.

    Specialised per boundary triple, so this runs on all eight rather than on a
    representative one: the ghost rule is exactly what specialisation changes, and a
    redirect that ate the branch would show only on the walled configurations.
    """
    certified = _certified_curl_tail(codes, shaders.CONTRACT_OFF)
    welded = hd.welded_curl_tail(codes, shaders.CONTRACT_OFF)
    for axis, letter in enumerate("xyz"):
        line = shaders.ghost(letter, codes[axis], True)
        assert line in certified and line in welded, (axis, line)
    assert welded.count(" : 0.0f;") == 6, (
        "six shifted taps, each still guarded by the emitter's own validity flag")


# ---------------------------------------------------------------------------
# The two Yee sub-lattices, and why the signature fits
# ---------------------------------------------------------------------------

def test_both_halves_read_one_sub_lattice_which_is_what_makes_thirty_pointers():
    """The premise behind the shared ``kms`` group, read off the shipped tables.

    ``step_D``'s suffix and ``update_H``'s half-integer flag are what decide it, and if
    either moved, sharing the group would bind one half's coefficients on the other
    half's lattice -- a converged, smooth, half-cell-wrong absorber rather than a
    failure. The plan builder raises on the same comparison; this is where it is a red
    at the merge bar instead of at plan time.
    """
    assert metal_launch.SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False
    source = hd.fused_hd_pair_source((0, 0, 0))
    signature = source.split("kernel void fused_hd_pair_step(", 1)[1].split(
        "uint idx [[thread_position_in_grid]])", 1)[0]
    assert signature.count("* kmx") + signature.count("* kmy") \
        + signature.count("* kmz") == 3, "one kms group, shared by both halves"
    assert sum(signature.count(f"* kp{n}") for n in range(3)) == 3
    assert "km0" not in signature and "km1" not in signature


def test_a_curl_never_differences_a_component_along_its_own_axis():
    """Why every halo recompute reads a coefficient row that is already in range.

    Not a premise this weld rests on -- :func:`.fused_hd_pair.h_cell_function` indexes
    the coefficients at the RECOMPUTED cell's coordinates, so the arithmetic is the
    certified body's wherever it lands. It is asserted because it is the reason the
    recompute costs no extra coefficient binding, and because a curl that started
    differencing a component along its own axis would change what this kernel is.
    """
    own_axis = {0: "i", 1: "j", 2: "k"}
    certified = _certified_curl_tail((0, 0, 0), shaders.CONTRACT_OFF)
    offsets = hd.offset_coordinates(certified)
    shifted = {"ox": 0, "oy": 1, "oz": 2}
    for line in certified.splitlines():
        if " ? g" not in line:
            continue
        target = int(line.split(" ? g", 1)[1][0])
        offset = line.split(f" ? g{target}[", 1)[1].split("]", 1)[0]
        axis = shifted[offset]
        assert own_axis[target] == offsets[offset][target] or axis != target, (
            line, offsets[offset])
        assert axis != target, (
            f"{line.strip()} differences component {target} along its own axis; the "
            f"coefficient row this weld recomputes would then move too")


# ---------------------------------------------------------------------------
# The binding ceiling, as an EQUALITY
# ---------------------------------------------------------------------------

def test_the_shipped_signature_is_the_platform_ceiling_exactly():
    """31 of 31, counted off the emitted text rather than off the constant."""
    assert hd.shipped_signature_bindings() == hd.PACKED_BINDINGS
    assert hd.PACKED_BINDINGS == MAX_BUFFER_BINDINGS == 31
    source = hd.fused_hd_pair_source((0, 0, 0))
    for index in range(hd.PACKED_BINDINGS):
        assert f"[[buffer({index})]]" in source, index
    assert f"[[buffer({hd.PACKED_BINDINGS})]]" not in source


@pytest.mark.parametrize("builder,count", [
    (hd.refuted_one_more_pointer_source, hd.ONE_MORE_POINTER_BINDINGS),
    (hd.refuted_unshared_kms_source, hd.UNSHARED_KMS_BINDINGS),
    (hd.refuted_separate_scalar_source, hd.SEPARATE_SCALAR_BINDINGS),
])
def test_each_refuted_signature_declares_the_count_it_argues_from(builder, count):
    """The three shapes over the ceiling, each emitted with the binding model it claims.

    A refuted source that bound its scalars differently from the shipped kernel would
    be over the ceiling for a reason other than the one it is offered as evidence for;
    each of these is the shipped model plus exactly the one change named.
    """
    source = builder()
    assert source.count("[[buffer(") == count > MAX_BUFFER_BINDINGS
    assert f"[[buffer({count - 1})]]" in source


@pytest.mark.parametrize("codes", CODES)
@pytest.mark.parametrize("mode", MODES)
def test_the_source_is_ascii_and_carries_exactly_one_contraction_pragma(codes, mode):
    """A non-ASCII character kills the kernel at its first launch, measured 2026-08-15.

    And the contraction guard is a property of the SOURCE on this platform, so a second
    (or a differing) pragma would decide the arithmetic silently.
    """
    source = hd.fused_hd_pair_source(codes, mode)
    source.encode("ascii")
    assert source.count("#pragma") == 1
    assert shaders.contraction_pragma(mode) in source


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _Residency:
    """The minimum a predicate needs to see: something exposing ``mirror``."""

    def mirror(self, name, host, constant=False):  # pragma: no cover - never called
        raise AssertionError("the predicate must not mirror anything")


def _grid(**overrides):
    keywords = dict(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                    dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    keywords.update(overrides)
    return Grid(**keywords)


@pytest.fixture()
def flush_policy(monkeypatch):
    """ONE CANONICAL POLICY, and it is a precondition rather than a default.

    ``coverage._metal_backend_reasons`` delegates to ``subnormal.mps_policy_reasons``
    and a resolved ``keep`` is REFUSED by name rather than silently run: Metal flushes
    denormals natively and exposes no lever, so certifying under ``keep`` would be
    certifying bytes under a policy the device was never in. Every test below that
    expects an ADMITTED verdict therefore states the precondition, and
    :func:`test_a_keep_policy_run_is_refused_by_name` is what keeps the refusal armed.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")


def _fields(grid):
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, {})
    return fields


def _covered_case():
    grid = _grid()
    return _fields(grid), PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))


@needs_mps
def test_the_predicate_admits_the_cell_this_product_was_built_for(flush_policy):
    """A Cartesian, real, unfolded, k=0 PML run with no in-seam withdraw."""
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert verdict.covered, verdict.reasons


def test_an_undeclared_source_list_is_refused_by_name():
    """Ignorance is never an empty set: ``Fields`` does not hold the source list, so a
    predicate that inferred "no withdraw stands" from not being told would admit a row
    whose withdraw does work on every step from the second on."""
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, None, _Residency())
    assert not verdict.covered
    assert any("the source set was not declared" in reason
               for reason in verdict.reasons), verdict.reasons


def test_a_plan_built_with_no_residency_is_refused_by_both_halves():
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), None)
    assert not verdict.covered
    assert any("residency" in reason and reason.startswith("constitutive half:")
               for reason in verdict.reasons), verdict.reasons
    assert any("residency" in reason and reason.startswith("curl half:")
               for reason in verdict.reasons), verdict.reasons


def test_a_keep_policy_run_is_refused_by_name_rather_than_run(monkeypatch):
    """THE ARMED CONTROL BEHIND THE FLUSH PRECONDITION.

    Metal flushes denormals natively and exposes no lever -- both denormal pragma
    spellings are compile errors on this toolchain -- so a run whose resolved float32
    policy is ``keep`` would be certified under a policy the device was never in. It is
    refused by name, and this is what says the ``flush_policy`` fixture is a
    precondition the predicate enforces rather than a convenience the tests take.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "keep")
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert not verdict.covered
    assert any("subnormal policy is 'keep'" in reason
               for reason in verdict.reasons), verdict.reasons


def test_the_constitutive_half_is_asked_first_so_the_null_update_H_names_itself():
    """An inactive absorber is the board's single largest non-fusion reason -- 14 of the
    194 rows -- and ``update_H`` returns before its first statement there. The driver
    reaches the constitutive half first, so the first refusal a reader sees must be that
    half's."""
    grid = _grid()
    fields = _fields(grid)
    inert = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
    verdict = hd.metal_fused_hd_pair_coverage(fields, inert, (), _Residency())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("constitutive half:"), verdict.reasons
    assert any("no active PML layer" in reason for reason in verdict.reasons)


def _folded_case():
    """A folded grid and an absorber on the faces that can carry one.

    Cell 0 of a folded axis lies on the mirror plane, so ``PML`` refuses a low face
    there by name; the high face is the one an absorber can hold.
    """
    from meep_gpu.grid import Mirror

    grid = _grid(symmetry=(Mirror("Y", 1),))
    thickness = ((2, 2), (0, 2), (2, 2))
    return _fields(grid), PML(grid=grid, thickness=thickness)


@pytest.mark.parametrize("overrides,needle", [
    ({"k_point": (0.1, 0.0, 0.0)}, "k_point"),
])
def test_the_grid_refusals_name_the_configuration(overrides, needle, flush_policy):
    grid = _grid(**overrides)
    fields = _fields(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_the_fold_refusal_names_the_second_product_and_not_a_fill(flush_policy):
    """THE REASON IS DIFFERENT ON THIS SEAM, and saying the other one would be wrong.

    The B->H and D->E pairs refuse a fold because ``fill_symmetry_bc_*`` and
    ``fill_folded_far_ghosts_*`` run INSIDE their seam. Neither runs inside this one --
    ``fill_B`` closes the seam before it and ``fill_D`` opens the seam after it -- so
    this seam is fill-free on a folded grid, and what refuses a fold here is the ARM
    PAIRING: a folded run selects the ``folded`` arm on both slots, which is the 78-row
    cell and a different product.
    """
    fields, pml = _folded_case()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    folded = [reason for reason in verdict.reasons if reason.endswith(
        "own ghost-map measurement")]
    assert folded, verdict.reasons
    assert all("fill_symmetry_bc" not in reason for reason in folded), folded
    assert all("`folded` arm on both slots" in reason for reason in folded)


class _Source:
    """A source shaped like the driver's own list entries, and nothing more."""

    def __init__(self, field_type="D", is_integrated=False, points=1,
                 component="Ex"):
        self.field_type = field_type
        self.is_integrated = is_integrated
        self._n_source_points = points
        self.component = component
        self._applied_dipole = 0j

    def withdraw(self, fields):  # pragma: no cover - never called by a predicate
        raise AssertionError("a predicate must not withdraw anything")


def test_a_magnetic_source_is_not_this_seams_business(flush_policy):
    """The driver injects it one seam EARLIER (driver.py:3283-3284) and withdraws it
    over ``step_B`` (:3289-3290). A product that refused it would be refusing a row for
    a pass that is not in its span."""
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(
        fields, pml, (_Source(field_type="B", is_integrated=True, component="Hx"),),
        _Residency())
    assert verdict.covered, verdict.reasons


def test_a_non_integrated_electric_source_is_not_refused(flush_policy):
    """Its ``withdraw`` returns before its first array write, so nothing stands in the
    seam. Refusing it would cost the cell 40 of its 49 rows for a pass that no-ops."""
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(
        fields, pml, (_Source(is_integrated=False),), _Residency())
    assert verdict.covered, verdict.reasons


def test_a_standing_electric_withdraw_is_refused_by_name_while_the_flag_is_false():
    """THE 2 ROWS OF THE CELL'S 49 THIS PRODUCT DOES NOT REACH.

    The refusal names the flag, the installability declaration and the driver line,
    because the remedy is a package of two things (the absorb row landed 2026-09-05)
    and a reader has to be able to see which one is missing.
    """
    fields, pml = _covered_case()
    verdict = hd.metal_fused_hd_pair_coverage(
        fields, pml, (_Source(is_integrated=True),), _Residency())
    assert not verdict.covered
    named = [reason for reason in verdict.reasons if "HOISTS_THE_WITHDRAW" in reason]
    assert named, verdict.reasons
    assert "driver.py:3313-3314" in named[0]
    assert "INSTALLABLE = False" in named[0]


def test_the_withdraw_clause_is_the_shared_modules_and_not_a_second_copy():
    """A second implementation of "does a withdraw stand in this seam" is a second thing
    to get wrong, and the failure it would produce is a fused launch reading a D that
    still holds the previous step's standing dipole."""
    fields, pml = _covered_case()
    standing = _Source(is_integrated=True)
    assert withdraw_hoist.standing_withdraws((standing,)) == ((0, standing),)
    assert withdraw_hoist.standing_withdraws((_Source(is_integrated=False),)) == ()
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (standing,), _Residency())
    assert not verdict.covered


# ---------------------------------------------------------------------------
# The composition: registered, unwired, uninstallable
# ---------------------------------------------------------------------------

def _spec():
    matches = [spec for spec in arms.registered("update_H")
               if spec.family == hd.FAMILY]
    assert len(matches) == 1, [spec.family for spec in arms.registered("update_H")]
    return matches[0]


def test_the_arm_is_registered_unwired_and_is_a_weld():
    """Enumerable so the disjointness sweep can measure this predicate; unwired so
    ``plan_step`` cannot select it. ``is_weld`` is the distinction the sweep needs and
    ``wired`` is not it -- a weld always co-admits with the arm it welds."""
    spec = _spec()
    assert spec.wired is False
    assert spec.is_weld is True
    assert spec.replaces == hd.REPLACES
    assert hd.FAMILY not in {other.family for other in arms.registered("update_H")
                             if other.wired}


def test_the_builder_refuses_a_single_cell_where_the_composer_absorbs_a_second():
    """The table floor holds in the direction the artifact test cannot reach.

    `_wiring` compares the cells a family prices with the pairs the composer absorbs.
    Measured 2026-09-10 before this floor read `FUSED_PAIR_EXTRA_ARMS`: dropping the
    nonlinear cell from `fused_hd_pair`'s table raised NOTHING, because a single-cell
    entry skipped the comparison as "not multi-cell". A board cut on that table would
    have published 525 with a clean conscience. So the floor is pinned from the
    builder side: a family with an extra absorb row and one declared cell refuses.
    """
    import contextlib
    import copy
    import io
    import sys

    parity = Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
    if str(parity) not in sys.path:
        sys.path.insert(0, str(parity))
    import build_fusion_matrix as board  # noqa: PLC0415

    table = copy.deepcopy(board.PRODUCTS)
    table["fused_hd_pair"]["cells"] = table["fused_hd_pair"]["cells"][:1]
    original = board.PRODUCTS
    board.PRODUCTS = table
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(SystemExit, match="composer absorbs"):
                board._wiring()
    finally:
        board.PRODUCTS = original


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_board_prices_every_absorb_row_as_a_cell_and_no_other():
    """The table and the board move together, pinned from the artifact side.

    The builder asserts it from the table side (`_wiring`: a `cells` entry must equal
    `FUSED_PAIR_ARMS[family]` + `FUSED_PAIR_EXTRA_ARMS[family]`). This reads the cut
    board and asks the same question of what was actually published, so a board cut
    on a stale table cannot pass here on the strength of the builder's own floor.
    """
    board = (Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
             / "results" / "fusion_matrix_metal_2026-09-11_dispatch"
             / "fusion_matrix.json")
    if not board.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{board} is not in this checkout")
    products = json.loads(board.read_text(encoding="utf-8"))["products"]
    for family_name, extra in metal_launch.FUSED_PAIR_EXTRA_ARMS.items():
        priced = {tuple(cell["cell"])
                  for cell in products[family_name]["cells"].values()}
        absorbed = {tuple(metal_launch.FUSED_PAIR_ARMS[family_name])}
        absorbed |= {tuple(pair) for pair in extra}
        assert priced == absorbed, (family_name, priced, absorbed)


def test_the_absorb_row_is_read_off_the_two_predicates_this_module_conjoins():
    """``("ordinary", "PML")`` in the SEAM'S slot order -- constitutive, then curl --
    because the predicate opens with ``constitutive_coverage(..., "H")`` (the
    ``ordinary`` arm's own predicate on ``update_H``) and ``pml_curl_coverage(...,
    "step_D")`` (the ``PML`` arm's on ``step_D``). Both labels are checked against the
    registry rather than trusted, and so are the extra row's two spine-arm labels."""
    assert metal_launch.FUSED_PAIR_ARMS[hd.FAMILY] == ("ordinary", "PML")
    assert "ordinary" in {spec.label for spec in arms.registered("update_H")}
    assert "PML" in {spec.label for spec in arms.registered("step_D")}
    extra = metal_launch.FUSED_PAIR_EXTRA_ARMS[hd.FAMILY]
    assert extra == (("nonlinear PML magnetic", "nonlinear PML curl"),)
    assert "nonlinear PML magnetic" in {spec.label for spec in arms.registered("update_H")}
    assert "nonlinear PML curl" in {spec.label for spec in arms.registered("step_D")}
    # THREE FAMILIES CARRY AN EXTRA ROW SINCE THE BETA H->D PRODUCTS WIRED, and the
    # two that joined are a DIFFERENT KIND of extra from this family's: each resolves
    # its curl emitter from the grid's fold and covers a SECOND cell -- the FOLDED one
    # -- through one extra row apiece, declared beside them in
    # ``metal_kernels/launch.py``'s own comment on the table and driven in both
    # directions by each family's tests. The property this line was really keeping is that
    # nothing on this backend absorbs on a second cell without a test naming it, so
    # the membership is pinned AND every extra row's two labels are required to be
    # registered arms on the seam's two slots, exactly as this family's own row is
    # above. A family gaining an extra row with an unregistered label fails here.
    assert set(metal_launch.FUSED_PAIR_EXTRA_ARMS) == {
        hd.FAMILY, "beta_real_fused_hd_pair", "beta_complex_fused_hd_pair"}
    update_h_labels = {spec.label for spec in arms.registered("update_H")}
    step_d_labels = {spec.label for spec in arms.registered("step_D")}
    for family, rows in metal_launch.FUSED_PAIR_EXTRA_ARMS.items():
        for constitutive, curl in rows:
            assert constitutive in update_h_labels, (family, constitutive)
            assert curl in step_d_labels, (family, curl)


def test_the_uninstallable_declaration_is_read_off_this_module():
    """The second brake. ``_declared_uninstallable`` resolves the module through the
    coverage callable's ``__module__``, so a product whose predicate moved would fail
    open -- this pins that the resolution actually reaches this module's flag."""
    refusal = metal_launch._declared_uninstallable(_spec())
    assert refusal is not None
    assert refusal.startswith(f"{hd.FAMILY} declares INSTALLABLE = False: ")
    assert hd.INSTALLABLE_REASON in refusal


@needs_mps
def test_the_seam_loop_refuses_this_product_on_its_flag_and_then_on_the_released_slot(
        flush_policy, monkeypatch):
    """DRIVEN THROUGH THE COMPOSER, not read off the tables, from both sides.

    ``plan_step(fuse=True)`` is the opt-in path every Metal weld reaches, and on a
    configuration this predicate ADMITS the seam loop must still leave ``update_H`` and
    ``step_D`` to the pairs that already hold them. Shipped, the reason it records is
    the product's own ``INSTALLABLE = False`` -- a fact about the product, reported on
    every configuration. With that flag out of the way it is STILL refused, by
    ``_pair_may_absorb`` naming the released B->H pair that installed first, and the
    composition does not move. That second brake is what the 2026-09-04 tie
    measurement found missing, and it is asserted here rather than inferred.
    """
    fields, pml = _covered_case()
    assert hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency()).covered
    plan = metal_launch.plan_step(fields, pml, residency=metal_launch.Residency(),
                                  sources=(), fuse=True)
    reason = plan.reasons.get(f"fused_pair_{hd.FAMILY}")
    assert reason is not None, sorted(plan.reasons)
    assert reason[0].startswith(f"{hd.FAMILY} declares INSTALLABLE = False: "), reason
    # AND THE ARBITRATION FINDING, IN LIVE FORM. On this configuration BOTH released
    # incumbents install, so all four slots of the step_B - update_H - step_D -
    # update_E path are held by two pairs in two launches. That is the 133-row column
    # of the measurement `INSTALLABLE_REASON` cites: installing this product here would
    # take one slot from each of them and leave one seam served in three launches.
    assert plan.selected["update_H"] == plan.selected["step_B"], plan.selected
    assert plan.selected["step_D"] == plan.selected["update_E"], plan.selected
    assert plan.selected["update_H"] != plan.selected["step_D"], plan.selected

    monkeypatch.setattr(hd, "INSTALLABLE", True)
    fields, pml = _covered_case()
    unflagged = metal_launch.plan_step(fields, pml, residency=metal_launch.Residency(),
                                       sources=(), fuse=True)
    assert unflagged.selected == plan.selected
    reason = unflagged.reasons.get(f"fused_pair_{hd.FAMILY}")
    assert reason is not None and reason[0].startswith(
        f"update_H was selected by the {plan.selected['update_H']!r} arm"), reason
    assert hd.FAMILY not in {label for label in plan.selected.values()}


def test_the_h_to_d_seam_row_routes_to_the_withdraw_hoist_and_is_offered_last():
    """Key order is load-bearing: ``_install_fused_pairs`` walks ``.items()`` in dict
    order against the LIVE ``selected``, so this seam placed first would be offered
    ``update_H`` before ``fused_magnetic_pair`` and could displace a released product."""
    seams = metal_launch.FUSED_PAIR_SEAMS
    assert list(seams)[-1] == "update_H"
    assert seams["update_H"] == ("step_D", withdraw_hoist.SEAM)
    assert seams["update_H"][1] not in ("B", "D"), (
        "the seam name is not a deposit_repair field letter; any other string would "
        "select the ELECTRIC list, which the driver injects one seam later")


class _Context:
    def __init__(self, fields, pml, residency, sources=()):
        self.fields = fields
        self.pml = pml
        self.residency = residency
        self.sources = sources
        self.contract_variants = (shaders.CONTRACT_OFF,)
        self.synced = ()
        self.extra = {}


@needs_mps
def test_the_released_magnetic_pair_keeps_its_slot_whatever_this_flag_says(
        flush_policy, monkeypatch):
    """THE 2026-09-04 HAZARD, CLOSED FROM THE COMPOSER'S SIDE.

    ``_neighbouring_seam_claimant`` walks ``arms.registered("update_H")`` for welds that
    also replace ``step_D`` when an INTERIOR span asks who claims the seam it opens.
    The B->H span is an END EDGE of ``FUSED_PAIR_SEAMS`` and is never asked, so
    ``fused_magnetic_pair`` keeps both of its slots whether this product's flag is
    True or False. Until 2026-09-05 the same call returned THIS family as the claimant
    the moment the flag was flipped -- the flag was the only thing between a released,
    gate-passing product and a coverage regression on every row this predicate
    reaches. The armed control is this product's OWN span, which is interior and is
    asked: on this configuration it yields to the D->E pair.
    """
    fields, pml = _covered_case()
    context = _Context(fields, pml, _Residency())
    for flag in (False, True):
        monkeypatch.setattr(hd, "INSTALLABLE", flag)
        assert metal_launch._neighbouring_seam_claimant(
            context, "fused_magnetic_pair", "step_B", "update_H",
            ("step_B", "update_H")) is None, flag
    asked = metal_launch._neighbouring_seam_claimant(
        context, hd.FAMILY, "update_H", "step_D", ("update_H", "step_D"))
    assert asked is not None and asked[0] == "fused_electric_pair", asked
    assert "slot arbitration" in asked[1]


# ---------------------------------------------------------------------------
# The nonlinear widening, 2026-09-05 -- a text identity, not a third kernel
# ---------------------------------------------------------------------------

def _nonlinear(fields):
    shape = tuple(fields.grid.shape)
    fields.set_nonlinear_volumes({"Ez": np.full(shape, np.float32(0.1))},
                                 {"Ez": np.full(shape, np.float32(0.2))})
    return fields


@needs_mps
def test_a_chi2_chi3_run_is_admitted_through_the_spine_arms_and_refused_by_the_ordinary_halves(
        flush_policy):
    """The ordinary predicates refuse a nonlinear run by name (coverage clause 10) --
    asserted, so the widening is shown to be doing the admitting -- and this predicate
    admits it, because the two halves' spine twins do."""
    from meep_gpu.metal_kernels.coverage import constitutive_coverage, pml_curl_coverage

    fields, pml = _covered_case()
    _nonlinear(fields)
    ordinary_h = constitutive_coverage(fields, pml, "H", _Residency())
    ordinary_d = pml_curl_coverage(fields, pml, "step_D", _Residency())
    assert not ordinary_h.covered and any("chi2/chi3" in r for r in ordinary_h.reasons)
    assert not ordinary_d.covered and any("chi2/chi3" in r for r in ordinary_d.reasons)
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert verdict.covered, verdict.reasons


def test_a_double_refusal_reports_both_halves_and_both_widenings():
    """A configuration BOTH the ordinary half and its spine twin refuse names both,
    prefixed, so a reader can tell which side said what -- the CUDA precedent's
    wording. A k_point is refused by every one of the four."""
    grid = _grid(k_point=(0.1, 0.0, 0.0))
    fields, pml = _fields(grid), PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = hd.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert not verdict.covered
    prefixes = {reason.split(":")[0] for reason in verdict.reasons}
    assert {"constitutive half", "constitutive half, nonlinear-run widening",
            "curl half", "curl half, nonlinear-run widening"} <= prefixes, prefixes


def test_the_spine_arms_delegate_to_the_ordinary_builders_and_the_extra_row_names_them():
    """READ OFF THE SOURCE: the two spine arms build the ORDINARY kernels
    (``launch.plan_constitutive`` / ``launch.plan_pml_curl`` under a scope view), so
    the extra absorb row is a text identity rather than a widening of the weld; and
    the row's two labels are exactly the spine arms' registered labels on the two
    slots this product spans."""
    import ast
    import inspect

    from meep_gpu.metal_kernels import nonlinear_update_e as nl

    def calls_of(function):
        tree = ast.parse(inspect.getsource(function))
        return {f"{node.func.value.id}.{node.func.attr}" for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)}

    assert "launch.plan_constitutive" in calls_of(nl.plan_nonlinear_run_constitutive)
    assert "launch.plan_pml_curl" in calls_of(nl.plan_nonlinear_run_pml_curl)
    spine = {(spec.slot, spec.label) for spec in nl.SPINE_ARMS}
    (magnetic, curl), = metal_launch.FUSED_PAIR_EXTRA_ARMS[hd.FAMILY]
    assert ("update_H", magnetic) in spine and ("step_D", curl) in spine, spine


@needs_mps
def test_the_nonlinear_run_kernels_are_the_ordinary_text_byte_for_byte(
        flush_policy, monkeypatch):
    """MEASURED, NOT ARGUED: capture every source handed to ``compile_source`` while
    building the spine arms on a chi2/chi3 run and the ordinary arms on the same grid
    without it, and require the two captures equal. This is the fact the extra absorb
    row rests on."""
    from meep_gpu.metal_kernels import nonlinear_update_e as nl

    captured = {}
    # THE REAL COMPILER IS BOUND ONCE, BEFORE ANY PATCHING. Reading it inside
    # ``capturing`` binds whatever the PREVIOUS patch installed, so each new tag
    # chains onto the last one and every later compile is appended to every earlier
    # tag's list as well -- which makes the equality below compare two aggregates
    # rather than the two sources it claims to compare.
    real = metal_launch.compile_source

    def capturing(tag):
        def fake(source, *args, **kwargs):
            captured.setdefault(tag, []).append(source)
            return real(source, *args, **kwargs)
        return fake

    for tag, nonlinear, plan_h, plan_d in (
            ("ordinary", False,
             lambda f, p, r: metal_launch.plan_constitutive(f, p, "H", r),
             lambda f, p, r: metal_launch.plan_pml_curl(f, p, "step_D", r)),
            ("spine", True,
             lambda f, p, r: nl.plan_nonlinear_run_constitutive(f, p, "H", r),
             lambda f, p, r: nl.plan_nonlinear_run_pml_curl(f, p, "step_D", r))):
        fields, pml = _covered_case()
        if nonlinear:
            _nonlinear(fields)
        residency = metal_launch.Residency()
        monkeypatch.setattr(metal_launch, "compile_source", capturing(f"{tag}/update_H"))
        assert plan_h(fields, pml, residency) is not None, tag
        monkeypatch.setattr(metal_launch, "compile_source", capturing(f"{tag}/step_D"))
        assert plan_d(fields, pml, residency) is not None, tag
    for half in ("update_H", "step_D"):
        assert captured[f"ordinary/{half}"], half
        assert captured[f"ordinary/{half}"] == captured[f"spine/{half}"], half


# ---------------------------------------------------------------------------
# The scratch and the rotation — host logic, driven without a device
# ---------------------------------------------------------------------------

class _FakeResidency:
    """Resolves host arrays to distinct sentinels, by identity, like the real one."""

    def __init__(self, mapping):
        self._mapping = mapping

    def tensor_for_host(self, host):
        for candidate, tensor in self._mapping:
            if candidate is host:
                return tensor
        return None


class _FakeFields:
    pass


def _rotating_plan(alias=False):
    fields = _FakeFields()
    live, twins, mapping = {}, {}, []
    for index, name in enumerate(hd.ROTATED_NAMES):
        current = np.zeros(2, dtype=np.float32)
        twin = current if alias else np.zeros(2, dtype=np.float32)
        setattr(fields, name, current)
        live[name] = current
        twins[name] = twin
        mapping.append((current, f"read:{index}"))
        if not alias:
            mapping.append((twin, f"write:{index}"))
    launched = []
    plan = weld.plan_scratch_weld(
        hd.FAMILY, _FakeResidency(mapping), fields,
        rotated_names=hd.ROTATED_NAMES,
        static_args=("static-0", "static-1"),
        functions={shaders.CONTRACT_OFF: lambda *args: launched.append(args)},
        volumes=hd.ROTATED_NAMES, shape=(1, 1, 2), codes=(0, 0, 0),
        zero_metal=(0, 0, 0), row_mask=(), replaces=hd.REPLACES, twins=twins)
    return plan, fields, live, twins, launched


def test_the_launch_binds_every_scratch_first_then_every_pre_launch_buffer():
    """The signature order IS the argument order, and nothing else keeps them in step.

    Buffers 0-5 are the scratch and 6-11 the pre-launch reads, in ``ROTATED_NAMES``
    order, then the static pointers; a kernel that interleaved a static pointer between
    two rotating ones would bind the wrong buffer to every argument after it.
    """
    plan, _fields, _live, _twins, launched = _rotating_plan()
    plan.run()
    args = launched[0]
    assert args[:6] == tuple(f"write:{n}" for n in range(6))
    assert args[6:12] == tuple(f"read:{n}" for n in range(6))
    assert args[12:] == ("static-0", "static-1")
    assert len(args) == hd.PACKED_BINDINGS - 17, (
        "the fake carries two static arguments rather than the shipped nineteen")


def test_the_rotation_moves_the_engines_references_only_after_the_launch():
    """H and f_w_H swap with the plan-owned twins, so every later pass, sync and
    read-back sees the freshly written buffer under the engine's own name."""
    plan, fields, live, twins, _launched = _rotating_plan()
    before_twins = dict(twins)
    plan.run()
    for name in hd.ROTATED_NAMES:
        assert getattr(fields, name) is before_twins[name], name
        assert plan.rotated[name] is live[name], name
    plan.run()
    for name in hd.ROTATED_NAMES:
        assert getattr(fields, name) is live[name], "a second run rotates back"


def test_an_aliased_scratch_is_refused_before_the_launch():
    """THE WHOLE DESIGN IS THAT NOTHING WRITTEN IS READ. A twin resolving to the same
    tensor as its live buffer would make every foreign recompute a read of another
    thread's output, silently, so it is refused rather than launched."""
    plan, _fields, _live, _twins, launched = _rotating_plan(alias=True)
    with pytest.raises(RuntimeError, match="ONE tensor"):
        plan.run()
    assert not launched


def test_the_rotated_set_is_exactly_what_the_constitutive_half_writes():
    """``update_H`` writes ``H*`` and ``f_w_H*`` and nothing else (the residency model's
    own entry), so those six are exactly what has to double-buffer. D and fu_D are
    absent deliberately: the curl reads and writes them at the thread's own cell only,
    which is why they update in place."""
    from meep_gpu.metal_kernels.coverage import SUB_STEP_VOLUMES

    assert set(hd.ROTATED_NAMES) == set(SUB_STEP_VOLUMES["update_H"]["writes"])
    assert set(SUB_STEP_VOLUMES["step_D"]["writes"]).isdisjoint(hd.ROTATED_NAMES)
    assert set(SUB_STEP_VOLUMES["step_D"]["reads"]) <= set(hd.ROTATED_NAMES)


# ---------------------------------------------------------------------------
# The withdraw hoist's wiring, and what would have to change to install
# ---------------------------------------------------------------------------

def test_the_installer_is_the_only_thing_that_would_perform_the_hoist():
    """``_install_fused_pair`` routes this seam name to a ``LeadingWithdrawPlan`` and
    leaves the second slot a ``NoopPlan`` -- one slot, not two; a hoist, not a bracket.

    Driven here with a stub pair so the branch is EXERCISED rather than read, and it is
    what says the flag this family declares False is a statement about wiring that
    exists: the wiring is real, the product simply cannot reach it.
    """
    plans, selected = {}, {}
    pair = object()
    standing = _Source(is_integrated=True)
    metal_launch._install_fused_pair(
        plans, selected, _FakeFields(), None, (standing,), withdraw_hoist.SEAM,
        "update_H", "step_D", "fused H/D pair", pair)
    leading = plans["update_H"]
    assert isinstance(leading, withdraw_hoist.LeadingWithdrawPlan)
    assert leading.inner is pair and leading.absorbed_by is pair
    assert leading.span == hd.REPLACES
    assert leading.placement == withdraw_hoist.BEFORE_UPDATE_H
    assert plans["step_D"].absorbed_by is pair
    assert selected == {"update_H": "fused H/D pair", "step_D": "fused H/D pair"}


def test_the_licensed_placement_is_the_one_the_campaign_measured():
    """And the two it is not are refused by name, each with its own null control."""
    assert withdraw_hoist.LICENSED_PLACEMENTS == (withdraw_hoist.BEFORE_UPDATE_H,)
    ok, reasons = withdraw_hoist.hoistable(
        None, (_Source(is_integrated=True),), span=hd.REPLACES,
        placement=withdraw_hoist.AFTER_STEP_D)
    assert not ok and any("after_step_D" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# The cell this product reaches, against the record that priced it
# ---------------------------------------------------------------------------

_SEAM_RECORD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "parity", "meep_gpu", "results", "h_to_d_seam_2026-09-04", "h_to_d_seam.jsonl")


@pytest.mark.skipif(not os.path.exists(_SEAM_RECORD),
                    reason="the H_to_D seam record is not in this checkout")
def test_the_cell_is_49_rows_of_which_this_predicate_declines_2():
    """READ FROM THE RECORD, not restated. The docstring's 47/49 is a claim about a
    measurement, so it fails here if the measurement moves rather than going quietly
    stale in prose."""
    rows = [json.loads(line) for line in open(_SEAM_RECORD, "r", encoding="utf-8")]
    cell = [row for row in rows
            if row["arms"]["metal"]["update_H"] == "ordinary"
            and row["arms"]["metal"]["step_D"] == "PML"]
    withdrawing = [row for row in cell if row["h_to_d_seam"]["withdraw_in_seam"]]
    assert len(cell) == 49, len(cell)
    assert len(withdrawing) == 2, [row["label"] for row in withdrawing]
    assert {row["arms"]["metal"]["would_be_bucket"] for row in withdrawing} == {
        "withdraw_seam"}
    assert len(cell) - len(withdrawing) == 47


# ---------------------------------------------------------------------------
# The device gate: present, routed, and armed with needles that still resolve
# ---------------------------------------------------------------------------
#
# THE GATE IS THE CERTIFICATION AND THIS SECTION IS NOT. What it pins is the part
# of the gate that can go stale WITHOUT a device: a mutation needle that stopped
# matching arms nothing, a leg table that stopped naming a leg, a case that stopped
# walling the axis its defects are armed on, a census cell that moved. Every one of
# those turns a green gate into a green gate that measured less than it says, and
# none of them needs an MPS device to catch.

_GATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "parity", "meep_gpu", "gate_metal_fused_hd_pair.py")


def _gate_module():
    """The gate, imported as a module. Text-only legs; no device is touched."""
    import importlib.util
    import sys

    here = os.path.dirname(_GATE)
    if here not in sys.path:
        sys.path.insert(0, here)
    specification = importlib.util.spec_from_file_location(
        "_hd_gate_under_test", _GATE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert os.path.isfile(_GATE)
    text = open(_GATE, "r", encoding="utf-8").read()
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records "
        "the sources the PROCESS imported")
    assert "gate_provenance" in text, (
        "the artifact must carry imported_source_sha256 and canonical_verdict")


def test_the_gate_declares_every_leg_it_runs_and_groups_all_of_them():
    """A leg missing from the group table is a leg ``--legs all`` never runs, and the
    verdict would still read PASS because ``complete`` is computed from the same
    table. The two are checked against each other here instead."""
    gate = _gate_module()
    grouped = [leg for group in gate.LEG_GROUPS.values() for leg in group]
    assert sorted(grouped) == sorted(gate.ALL_LEGS)
    assert len(grouped) == len(set(grouped)), "a leg is in two groups"
    text = open(_GATE, "r", encoding="utf-8").read()
    for leg in gate.ALL_LEGS:
        assert f'"{leg}" in legs' in text, f"{leg} is declared but never dispatched"
    # SIX, not five: the subject is driven twice -- once in a fully dispatched
    # composition and once with every non-seam slot on the array path. The second is
    # what attributes a divergence when a REFERENCE composition is the thing that
    # moved, because the first shares its non-seam plans with one.
    assert set(gate.MODES) == {"array", "singles", "composition_today", "unfused",
                               "weld", "weld_seam_only"}


def test_every_mutation_needle_still_resolves_exactly_once():
    """The builders raise unless each needle matched exactly once, so calling them IS
    the assertion. A needle that stopped matching would arm nothing and the gate would
    report the defect as caught by a kernel that was never mutated."""
    gate = _gate_module()
    armed = gate.shader_mutations(gate.MUTATION_CODES)
    inert = gate.inert_mutations(gate.MUTATION_CODES)
    periodic = gate.periodic_wrap_mutations(gate.PERIODIC_CODES)
    assert armed and inert and periodic
    base = hd.fused_hd_pair_source(gate.MUTATION_CODES)
    for name, (source, _why) in armed.items():
        assert source != base, name
    for name, (source, _reason, _control) in inert.items():
        assert source != base, name
    periodic_base = hd.fused_hd_pair_source(gate.PERIODIC_CODES)
    for name, (source, _why) in periodic.items():
        assert source != periodic_base, name
    assert gate.byte_neutral_source(gate.MUTATION_CODES) != base
    names = set(armed) | set(inert) | set(periodic)
    assert len(names) == len(armed) + len(inert) + len(periodic), "a name is reused"


def test_every_defect_that_cannot_fire_names_a_control_that_can():
    """An inert or unarmable defect is only honest if the rule it cannot reach is
    reached somewhere that DOES diverge. Each one names that row, and the name has to
    resolve to an armed, firing mutation rather than to prose."""
    gate = _gate_module()
    firing = set(gate.shader_mutations(gate.MUTATION_CODES))
    firing |= set(gate.periodic_wrap_mutations(gate.PERIODIC_CODES))
    firing |= set(gate.HOST_MUTATIONS)
    for name, (_source, reason, control) in gate.inert_mutations(
            gate.MUTATION_CODES).items():
        assert control in firing, (name, control)
        assert reason.strip(), name
    for entry in gate.PREDICTED_NULL:
        assert entry["control"] in firing, entry
        assert entry["reason"].strip(), entry
        assert entry["name"] not in firing, (
            f"{entry['name']} is declared unarmable and is also armed")


def test_the_two_mutation_cases_are_the_specialisations_their_defects_need():
    """``bc_mmm`` must wall every axis or the wall lines are not emitted; ``bc_ppp``
    must wall none or the wrap lines are not emitted. Both are read off the case
    table, so a renamed or reordered case fails here rather than silently arming
    defects on text the shipped kernel does not carry."""
    gate = _gate_module()
    assert gate.MUTATION_CODES == (1, 1, 1)
    assert gate.PERIODIC_CODES == (0, 0, 0)
    names = [name for name, _ in gate.CASES]
    assert gate.MUTATION_CASE in names and gate.PERIODIC_MUTATION_CASE in names
    assert len(gate.CASES) == 8, "the eight boundary specialisations"
    metallic = hd.welded_curl_tail(gate.MUTATION_CODES)
    periodic = hd.welded_curl_tail(gate.PERIODIC_CODES)
    assert "sj = (sj < 0) ? (nyi - 1) : sj;" in periodic
    assert "sj = (sj < 0) ? (nyi - 1) : sj;" not in metallic
    assert "vy = (sj >= 0) && (sj < nyi);" in metallic


def test_the_budget_covers_the_state_the_weld_carries_between_steps():
    """Sixty complete steps, and the sync leg's accessor calls inside them. A budget
    that no longer contained the sync steps would fire the flux accessor after the
    walk had finished and measure nothing."""
    gate = _gate_module()
    assert gate.STEPS == 60
    assert gate.SYNC_STEPS and max(gate.SYNC_STEPS) < gate.STEPS
    assert len(set(gate.SYNC_STEPS)) == len(gate.SYNC_STEPS)


def test_the_gate_lifts_the_cell_the_seam_record_priced():
    """The census, the seam record and the cell's two arms are the ones the family's
    own numbers come from. A gate pointed at a different census would report a
    coverage number against a denominator nothing else in the tree uses."""
    gate = _gate_module()
    assert gate.CELL_ARMS == metal_launch.FUSED_PAIR_ARMS[hd.FAMILY]
    assert gate.SEAM_RECORD == "h_to_d_seam_2026-09-04"
    assert gate.CENSUS.startswith("metal_coverage_")


@pytest.mark.skipif(not os.path.exists(_SEAM_RECORD),
                    reason="the H_to_D seam record is not in this checkout")
def test_the_gates_lift_basis_is_the_records_own_forty_nine_rows():
    """DERIVED, not listed: the basis is cut from the census's own ``plan_step``
    selection and joined to the seam record. Both halves are checked, because a join
    that silently matched nothing would hand the lift leg an empty row list and the
    leg would then be scored against a denominator of zero."""
    gate = _gate_module()
    results = os.path.join(os.path.dirname(_GATE), "results")
    if not os.path.isdir(os.path.join(results, gate.CENSUS)):
        pytest.skip("the standing census is not in this checkout")
    rows, facts = gate.lift_basis(__import__("pathlib").Path(results))
    assert facts["rows_in_cell"] == len(rows) == 49, facts
    assert sorted(facts["rows_with_a_standing_withdraw"]) == [
        "examples:differential_cross_section.py",
        "tests:TestIntegratedSource.test_integrated_source"]
    assert all(row["interpreter"] for row in rows)
    assert all(row["leg"] == "examples" or row["module"] for row in rows), (
        "a tests row with no module cannot be re-lifted")


def test_the_seed_scale_is_a_power_of_two_and_the_source_carries_the_same_one():
    """Both halves of the fixture's amplitude, and they have to move together.

    The seed is scaled so the 60-step budget clears the float32 denormal band. A
    POWER OF TWO is what makes that a shift of every exponent rather than a change of
    any mantissa. And the withdraw leg's source must carry the SAME factor: an O(1)
    dipole added to a 2^80 field rounds away entirely, which silently disarms both of
    that leg's null controls while leaving its hoisted arrangement green.
    """
    gate = _gate_module()
    assert gate.SEED_SCALE_BITS > 0
    assert float(gate.INTEGRATED_SOURCE["amplitude"]) == 2.0 ** gate.SEED_SCALE_BITS
    scale = np.float32(2.0) ** gate.SEED_SCALE_BITS
    assert np.isfinite(scale) and scale > 0
    # The scaled peak must stay finite in float32 with room to spare: the fixture's
    # own amplitude is 0.37, so the headroom is what stops the other end of the range
    # becoming the problem the band was.
    assert float(scale) * 0.37 < float(np.finfo(np.float32).max) / 1e6


def test_a_lifted_row_answers_to_a_floor_and_the_synthetic_fixture_to_the_budget():
    """Two different budget rules, and the difference is whose state it is.

    The gate chooses the synthetic fixture's amplitude, so that fixture must clear the
    whole budget. It does not choose a lifted corpus row's: that state starts at the
    engine's zeros and is driven by the row's own sources, so it enters the band on
    its own schedule and the honest budget is every step the precondition holds.
    """
    gate = _gate_module()
    assert gate.LIFT_CLEAN_STEP_FLOOR >= 1
    assert gate.LIFT_CLEAN_STEP_FLOOR < gate.STEPS
    source = open(_GATE, "r", encoding="utf-8").read()
    assert "require_full_budget=False" in source, (
        "the lift's battery hook must not demand the full budget of a row whose "
        "state it does not choose")
    # AND THE FLOOR IS APPLIED ONCE, IN THE LEG. A row answers for what it measured;
    # whether that was enough is a judgement about the CELL, made where the rows below
    # the floor can be named together instead of each reporting as a defect.
    assert "rows_below_the_clean_step_floor" in source
    assert "rows_that_diverged" in source
    assert "majority_cleared_the_floor" in source


# ---------------------------------------------------------------------------
# The gate's own bookkeeping: an arrangement must give the engine its arrays back
# ---------------------------------------------------------------------------
#
# THE DEFECT THESE PIN IS MEASURED, NOT IMAGINED. Until 2026-09-05 the gate named
# its rotating plans by hand and only the weld arrangement named any, so
# ``composition_today`` -- which installs the off-diagonal fused electric pair on a
# row with an off-diagonal epsilon, and that product rotates ``D``/``fu_D`` -- left
# ``fields.D*`` naming the pair's twins for the rest of the run. Every arrangement
# stepped after it wrote the ORIGINAL arrays through mirrors bound by identity while
# ``capture`` read the twins, so three of five arrangements came back 4 words short
# in ``D``/``fu_D`` on ``examples:cavity_arrayslice.py``. It reads as a weld defect
# and is not one; worse, the same decoupling can manufacture an AGREEMENT between two
# arrangements that both wrote an array nobody read.

class _RotatingStub:
    """A plan that swaps one engine array for its twin at every launch."""

    def __init__(self, fields, name, declares):
        self.fields = fields
        self.name = name
        self.twin = np.zeros_like(getattr(fields, name))
        self.rotated = {name: self.twin}
        self.rotated_names = (name,) if declares else ()
        self.launches = 0
        self.swaps = 0

    def run(self):
        current = getattr(self.fields, self.name)
        setattr(self.fields, self.name, self.rotated[self.name])
        self.rotated[self.name] = current
        self.launches += 1
        self.swaps += 1


class _StubDriver:
    def __init__(self, gate):
        self.fields = _FakeFields()
        for name in gate.reset_declared_volumes():
            setattr(self.fields, name, np.zeros(4, dtype=np.float32))
        self._sources = ()
        self.step_count = 0
        self._fast_path = None
        self._fast_path_stale = False

    def step(self):
        fast = self._fast_path
        if fast is None or not fast.dispatch("update_H", self.fields):
            pass
        self.step_count += 1


class _MirrorRegistry:
    """A residency BOUND BY IDENTITY to one host array, the way the real one is.

    ``orphaned_mirrors`` asks a residency two questions -- which names it binds, and
    which host array each name binds -- and answers the hazard from those. Both are
    pure host bookkeeping (``device.Residency.names`` / ``.host``), so the arming can
    be driven on a machine with no MPS, which is what the rest of this file requires.
    """

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    @property
    def names(self):
        return tuple(sorted(self._mapping))

    def host(self, name):
        return self._mapping.get(name)


def _rotating_arrangements(gate, declares):
    driver = _StubDriver(gate)
    plan = _RotatingStub(driver.fields, "Dx", declares)
    shim = gate.Shim(driver.fields, {"update_H": plan})
    # THE ROTATOR'S MIRROR IS BOUND TO THE ARRAY THE ENGINE NAMES RIGHT NOW, which is
    # what makes the swap below an ORPHAN rather than a rename. Without it this
    # control arms nothing: a rotation of a volume no residency mirrors is the
    # engine's own polarization recurrence, which the guard is required to ignore.
    residency = _MirrorRegistry({"Dx": driver.fields.Dx})
    return driver, plan, {"array": gate.Arrangement("array", None, None),
                          "rotator": gate.Arrangement("rotator", shim, residency)}


def test_an_arrangement_finds_its_rotating_plans_without_being_told():
    """DERIVED FROM THE SHIM. The weld is not the only product that rotates, and a
    list the caller maintains cannot see one the composer chose."""
    gate = _gate_module()
    driver, plan, arrangements = _rotating_arrangements(gate, declares=True)
    rotator = arrangements["rotator"]
    assert rotator.rotating == (plan,)
    assert rotator.originals["Dx"] is driver.fields.Dx
    assert gate.Arrangement("array", None, None).rotating == ()
    # ONCE PER OWNER, not once per slot: a fused pair holds both slots of its seam.
    both = gate.Shim(driver.fields, {"update_H": plan, "step_D": plan})
    assert gate.Arrangement("pair", both, None).rotating == (plan,)


def test_a_rotation_left_standing_fails_the_run_and_settling_it_does_not():
    """THE ARMED CONTROL AND ITS NULL, on the same stub and the same driver.

    Armed: the plan rotates and declares nothing, so nothing can settle it and the
    engine ends the step naming the twin. Null: the same plan declaring the volume it
    rotates, which the arrangement discovers and settles. The swap is asserted to have
    happened in BOTH, so the null is not passing by never rotating at all.

    AND THE VERDICT IS THE ORPHAN, NOT THE RENAME, from 2026-09-06. The run's pass
    condition used to read "did any stored name's array object change", which the
    ENGINE'S OWN three-buffer polarization recurrence answers yes to on every
    dispersive row (``dispersion.PolarizationState.step``) while no residency mirrors
    a single one of those buffers -- so the check was failing runs over a rename that
    could orphan nothing. It now asks the mirrors themselves. Both facts are still
    recorded and both are asserted here, because they are different facts and the
    control has to arm the one the verdict reads.
    """
    gate = _gate_module()

    driver, plan, arrangements = _rotating_arrangements(gate, declares=False)
    before = driver.fields.Dx
    armed = gate.drive(driver, arrangements, steps=2)
    assert plan.swaps == 2, "the armed control never rotated; it proves nothing"
    # OBSERVED AFTER EACH ARRANGEMENT'S STEP, WHICH IS NOT THE SAME AS "CAUSED BY IT",
    # and the spread is the point: an unsettled rotation contaminates every
    # arrangement that steps after it, the reference included -- here the array path
    # is stepping through the twin by step 2.
    assert armed["engine_volumes_left_rotated"] == {"array": ["Dx"], "rotator": ["Dx"]}
    assert armed["mirrored_volumes_left_orphaned"] == {"array": ["Dx"],
                                                       "rotator": ["Dx"]}
    assert armed["every_arrangement_gave_the_engine_its_volumes_back"] is False
    assert armed["arrangements"]["rotator"]["engine_volumes_left_rotated"] == ["Dx"]
    # AND THE END STATE IS NOT THE TEST. Two swaps put the original array back under
    # the engine's name, which is exactly why an end-of-run identity check would miss
    # this: the decoupling is per step, so it is measured per step.
    assert driver.fields.Dx is before

    driver, plan, arrangements = _rotating_arrangements(gate, declares=True)
    before = driver.fields.Dx
    null = gate.drive(driver, arrangements, steps=2)
    assert plan.swaps == 2, "the null control never rotated; it proves nothing"
    assert null["engine_volumes_left_rotated"] == {}
    assert null["mirrored_volumes_left_orphaned"] == {}
    assert null["every_arrangement_gave_the_engine_its_volumes_back"] is True
    assert null["arrangements"]["rotator"]["engine_volumes_left_rotated"] == []
    assert driver.fields.Dx is before, "settling puts the engine's own array back"


def test_a_rename_no_residency_mirrors_is_not_an_orphan():
    """THE SECOND HALF OF THE 2026-09-06 REPAIR, and it is what narrowed the verdict.

    The same stub, the same two swaps, the same reported rename -- with the mirror
    registry bound to a DIFFERENT volume, which is the engine's own polarization
    recurrence in miniature: it renames buffers no residency mirrors, every step, by
    design (``dispersion.PolarizationState.step`` rotates three per driven component).
    The run must not fail on it, and the rename must still be REPORTED, because a
    guard that stopped recording the fact would have nothing left to argue from.
    """
    gate = _gate_module()
    driver = _StubDriver(gate)
    plan = _RotatingStub(driver.fields, "Dx", declares=False)
    shim = gate.Shim(driver.fields, {"update_H": plan})
    unrelated = _MirrorRegistry({"Dy": driver.fields.Dy})
    arrangements = {"array": gate.Arrangement("array", None, None),
                    "rotator": gate.Arrangement("rotator", shim, unrelated)}
    result = gate.drive(driver, arrangements, steps=2)
    assert plan.swaps == 2, "this control never rotated; it proves nothing"
    assert result["engine_volumes_left_rotated"] == {"array": ["Dx"],
                                                    "rotator": ["Dx"]}
    assert result["mirrored_volumes_left_orphaned"] == {}
    assert result["every_arrangement_gave_the_engine_its_volumes_back"] is True


def test_the_product_leg_scores_the_settled_rotation():
    """A byte comparison across an orphaned volume is not a measurement, so the
    run's pass condition carries the bookkeeping rather than inferring it."""
    gate = _gate_module()
    source = open(_GATE, "r", encoding="utf-8").read()
    assert "every_arrangement_gave_the_engine_its_volumes_back" in source
    assert "and settled and result[\"step_error\"] is None" in source
