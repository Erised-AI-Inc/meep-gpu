"""The hand-CUDA ADE ``update_P`` family, checked with no GPU in the room.

WHAT THIS FILE CAN AND CANNOT SAY. ``ade_kernels.py`` imports CuPy at module
scope, so nothing here imports it; the device strings are read off the SYNTAX
TREE and the predicate is imported from the CuPy-free ``coverage.py``. That
splits the family's claims cleanly in two:

  * the ADMISSION -- which configurations are refused, and by which clause -- is
    pure decision logic whose failure mode is a SILENT WRONG ANSWER, and it is
    fully exercised and mutated here, at the merge bar. MEASURED 2026-08-19 by
    an adversarial sweep that disables one clause at a time: all 38 clauses of
    ``covers_real_pml_ade_component`` and ``covers_real_pml_ade_update_p`` are
    refused by the case that names them, and by no other. Before the sweep, 19
    of the 38 were pinned by nothing at all and 2 of the 19 that had a case were
    caught by a NEIGHBOURING clause whose text happened to contain the fragment
    being matched;
  * the ARITHMETIC is asserted only as a TRANSCRIPTION: the expression this file
    ships is compared character for character against the two certified siblings
    it was copied from. That is not a bit-identity claim. Nothing in this
    directory has launched this kernel, ``CERTIFIED_KERNELS`` is empty, and the
    module docstring names what a device gate still owes.

The shape is the one ``test_offdiag_constitutive_pml_real.py`` converged on,
ported to a family whose device code is two fixed strings rather than an emitter.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import types

import numpy as np
import pytest

from .. import dispersion
from ..triton_kernels import coverage as triton_coverage
from . import coverage

HERE = pathlib.Path(__file__).parent
#: the repository root, from which the record's subject paths are written.
REPO = HERE.parents[1]
RECORD = HERE / "certification.json"

#: THE BLOCK THAT DESCRIBES THE TREE AS IT SHIPS, which is not the same block as the
#: one the original release was cut from. ``ade_2026-08-19`` is the release record and
#: stays exactly as it was; it carried the tree between runs through
#: ``post_certification_edits``, each entry an argument for why an edited subject
#: could not reach the verdict. That mechanism weakens with every entry, and on
#: 2026-09-11 ``meep_gpu/subnormal_policy.py`` grew the arm64 fenv lever -- a subject
#: whose whole business is the float32 policy the two legs are cut under, which is the
#: one kind of edit no prose should be asked to excuse. So the gate was RE-CUT on
#: device against these bytes (``record_cuda_ade_recut.py``), and this constant names
#: the block that run produced. It moves with each re-cut, the same way
#: ``metal_dispatch.METAL_DRIVER_ROUTE_GATE`` does, and for the same reason: the name
#: of the run a claim rests on belongs in the source, not in a reader's memory.
SHIPPING_RECORD_BLOCK = "cuda_ade_recut_2026-10-05_cc86_residue"
KERNEL_MODULE = HERE / "ade_kernels.py"
CONSTITUTIVE_MODULE = HERE / "constitutive_kernels.py"
OFFDIAG_MODULE = HERE / "offdiag_constitutive_kernels.py"
CURL_MODULE = HERE / "step_curl_kernels.py"

TRITON_KERNELS = HERE.parent / "triton_kernels" / "kernels.py"
METAL_ADE = HERE.parent / "metal_kernels" / "ade_update_p.py"

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')

#: The one expression that decides bit-identity, as it must appear in all three
#: tracks. Kept as data so a reader sees the claim without running anything.
ADE_EXPRESSION = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"


def module_source() -> str:
    return KERNEL_MODULE.read_text(encoding="utf-8")


def module_level_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level")


def device_sources(source: str) -> dict:
    """The exact CUDA strings ``_get_kernel`` hands ``cp.RawKernel``.

    Evaluated from the syntax tree rather than by importing the module, because
    importing it needs CuPy -- which is half the reason this file exists.
    ``PRELUDE + r'''...'''`` concatenations are folded against the constants
    already seen, so each string is assembled exactly as it is at compile time.
    """
    seen, out = {}, {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        value = node.value
        text = None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            text = value.value
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) \
                and isinstance(value.left, ast.Name) \
                and isinstance(value.right, ast.Constant):
            text = seen.get(value.left.id, "") + value.right.value
        if text is None:
            continue
        seen[name] = text
        for kernel in KERNEL_DECLARATION.findall(text):
            out[kernel] = text
    return out


# --------------------------------------------------------------------------
# THE TRANSCRIPTION
# --------------------------------------------------------------------------

def test_the_arithmetic_is_character_identical_to_both_certified_siblings():
    """The whole correctness claim available without a device.

    ``ade_kernels`` did not derive this expression; it copied one that two
    independently certified tracks already ship. Triton's
    ``kernels.ade_update_p`` is pinned by ``triton_kernels/fingerprints.json``
    and Metal's ``ade_source`` by ``metal_kernels/fingerprints.json``. If either
    reference moves, the transcription stops being one and this fails.

    float32 addition is not associative, so the PARENTHESES are the claim -- not
    the operand names, which differ across the three languages only in spelling.
    """
    triton = TRITON_KERNELS.read_text(encoding="utf-8")
    metal = METAL_ADE.read_text(encoding="utf-8")
    cuda = device_sources(module_source())

    assert ADE_EXPRESSION in triton, (
        "triton_kernels/kernels.py no longer contains the reference expression; "
        "the CUDA family was transcribed from something that has moved")
    assert ADE_EXPRESSION in metal, (
        "metal_kernels/ade_update_p.py no longer contains the reference "
        "expression")
    for name, source in cuda.items():
        found = [line.strip() for line in source.splitlines()
                 if "p_out[idx] =" in line]
        assert found == [f"p_out[idx] = {ADE_EXPRESSION};"], (
            f"{name} does not write the reference expression: {found}")


def test_the_two_variants_differ_only_in_the_sigma_binding():
    """Two device strings, and the diff has to be exactly the specialization.

    The sigma axis is compile-time because a scalar has no array to bind. What
    must NOT come with it is any drift in the arithmetic, the guard, the index
    arithmetic or the loads -- so the two sources are compared line by line and
    the only permitted differences are the parameter declaration and the ``s =``
    load.
    """
    sources = device_sources(module_source())
    volume = sources["update_P_pml_real"].splitlines()
    uniform = sources["update_P_pml_real_uniform"].splitlines()
    assert len(volume) == len(uniform)
    differing = [(a.strip(), b.strip()) for a, b in zip(volume, uniform)
                 if a != b]
    assert differing == [
        ("extern \"C\" __global__ void update_P_pml_real(",
         "extern \"C\" __global__ void update_P_pml_real_uniform("),
        ("const float* __restrict__ sigma,", "float sigma,"),
        ("float s = sigma[idx];", "float s = sigma;"),
    ], differing


def test_every_device_string_is_pure_ascii():
    """A COMPILE REQUIREMENT, not a style rule.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE
    encoding -- ASCII under C/POSIX, which is what a non-interactive shell on the
    validation host gets. Two em-dashes in a comment killed a sibling kernel at
    its first launch on 2026-08-15. Encoded, not merely scanned.
    """
    for name, source in device_sources(module_source()).items():
        source.encode("ascii")  # raises UnicodeEncodeError with the offset
        assert "—" not in source and "–" not in source, name


def test_no_unary_minus_and_no_division_on_any_float_path():
    """The two platform facts this family leans on, checked rather than recalled.

    CUDA lowers ``-x`` to ``neg.f32`` and does NOT canonicalize signed zeros, so
    the sibling track's ``(a*b)*-1.0`` idiom is unsafe here and is absent. And a
    divide-free family cannot meet the one PTX exception on record, where ptxas
    expands ``div.rn.f32`` into a sequence carrying ``.FTZ`` in SASS whatever the
    PTX modifier said.
    """
    for name, source in device_sources(module_source()).items():
        body = "\n".join(line.split("//")[0] for line in source.splitlines())
        assert "/" not in body.replace("//", ""), f"{name} contains a division"
        assert not re.search(r"[-]\s*[a-zA-Z0-9_(]", body.replace("->", "")
                             .replace("--", "")), f"{name} contains a negation"


# --------------------------------------------------------------------------
# THE PARTITION AND THE COMPILE OPTIONS
# --------------------------------------------------------------------------

def test_the_family_ships_two_kernels_and_it_is_partitioned():
    """The partition is the file's promise that a kernel cannot ship unmeasured.

    Until 2026-08-19 this asserted the opposite -- ``CERTIFIED_KERNELS`` empty
    and both names dead -- because nothing here had been near a GPU. The gate ran
    (``certification.json``, block ``ade_2026-08-19``) and the names moved. What
    does NOT change is the shape: the two sets stay disjoint and must exhaust the
    shipped declarations, so a third kernel added here fails until it is placed
    on one side or the other.
    """
    source = module_source()
    certified = module_level_literal(source, "CERTIFIED_KERNELS")
    dead = module_level_literal(source, "UNCERTIFIED_KERNELS")
    shipped = set(KERNEL_DECLARATION.findall(source))

    assert shipped == {"update_P_pml_real",
                       "update_P_pml_real_uniform"}
    assert set(certified) == shipped, (
        f"every shipped kernel must be certified or named dead: "
        f"{shipped ^ set(certified)}")
    assert dead == {}, (
        f"UNCERTIFIED_KERNELS is not empty: {sorted(dead)}; a name there needs "
        f"the gate it owes beside it")
    assert not (set(certified) & set(dead)), "the partition overlaps"
    assert set(certified) | set(dead) == shipped


def test_every_certified_name_has_a_record_block_behind_it():
    """"Certified" must be a VERDICT, never a word someone typed.

    Both directions, the shape the off-diagonal family's record test uses: a name
    in ``CERTIFIED_KERNELS`` needs a ``certification.json`` block that claims it
    and reports a pass, and the block must carry the legs that make the claim
    mean something. For THIS family that means, specifically:

    * BOTH sigma specializations certified rather than one and a projection --
      the compile-time ``SIGMA_IS_VOLUME`` axis decides whether the device
      signature declares a pointer or a scalar, so a verdict on one says nothing
      about the other;
    * a MULTI-STEP budget, because the ROTATION is what a single launch cannot
      exercise, plus the stale-pointer leg that must be caught ONLY there -- that
      is the leg proving the budget is the instrument and not decoration;
    * the WRONG-DRIVE CONTROL caught, with its own null uncaught;
    * a diverging unguarded control, so ``--fmad=false`` is measured rather than
      asserted.
    """
    import json  # noqa: PLC0415

    source = module_source()
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    claimed = {name: key for key, block in record.items()
               if isinstance(block, dict)
               for name in block.get("certified_kernels", ())}
    for name in certified:
        assert name in claimed, (
            f"{name} is claimed certified with no certification.json block "
            f"behind it")

    block = record["ade_2026-08-19"]
    assert block["passed"] is True
    assert set(block["certified_kernels"]) == certified
    assert sorted(block["policies_cut_under"]) == ["ieee_keep_ftz_stripped",
                                                   "meep_x86_flush"]
    assert block["multi_step_budget"] == 60
    for policy in ("keep", "flush"):
        leg = block["per_policy"][policy]
        assert leg["released"] is True, policy
        assert leg["both_sigma_variants_certified"] is True, policy
        assert leg["mutation_legs_not_measurable"] == 0, policy
        assert leg["every_leg_disarmed"] is True, policy
        assert leg["subnormal_band_is_non_vacuous"] is True, policy
        assert leg["every_case_had_a_live_drive_term"] is True, policy
        assert leg["every_case_had_a_live_history_term"] is True, policy
        assert leg["every_case_admitted_by_the_predicate"] is True, policy

        identical, ran = leg["single_launch"].split("/")
        assert identical == ran and int(ran) > 0, (policy, leg["single_launch"])
        for form in ("volume", "uniform"):
            counts = leg["by_sigma_form"][form]
            assert counts["identical"] == counts["ran"] > 0, (policy, form)
        identical, ran = leg["multi_step"].split("/")
        assert identical == ran and int(ran) > 0, (policy, leg["multi_step"])

        # THE GUARD MUST DIVERGE or --fmad=false is decorative here.
        identical, ran = leg["guard_control_identical"].split("/")
        assert int(identical) == 0 < int(ran), (policy, leg["guard_control_identical"])

        legs = leg["mutation_legs"]
        assert legs["h1_wrong_drive_stored_E"] == "caught", policy
        assert legs["h2_wrong_drive_is_null_when_E_equals_f_w"] == "uncaught", policy
        assert legs["h3_stale_pointers"] == "caught_only_multi_step", policy
        assert legs["h4_no_rotation"] == "caught", policy
        # A battery whose every leg must be caught scores the same whether the
        # comparator works or has degenerated into failing everything.
        assert [name for name, verdict in legs.items()
                if verdict == "uncaught"], policy
        as_required, total = leg["mutation_legs_as_required"].split("/")
        assert as_required == total and int(total) == len(legs), policy


def test_the_record_block_pins_the_device_strings_that_ship_today():
    """A device string edited after the gate leaves a record describing a kernel
    that no longer exists. The digests are recomputed from the syntax tree, so a
    changed comment inside a kernel fails here rather than in a device queue.

    READ OFF :data:`SHIPPING_RECORD_BLOCK`, which is the re-cut rather than the dated
    release. The subject half below is the reason: a subject that has moved since its
    gate ran must be DECLARED, and a declaration is an argument where a run would be a
    measurement. Pointing this at the freshest run keeps the declarations for the case
    where re-running is genuinely not available, instead of making them the ordinary
    way the record stays green.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    block = json.loads(RECORD.read_text(encoding="utf-8"))[SHIPPING_RECORD_BLOCK]
    sources = device_sources(module_source())
    recorded = block["device_source_sha256"]
    for name, text in sources.items():
        assert recorded[name] == hashlib.sha256(text.encode("utf-8")).hexdigest(), (
            f"{name}'s device string is not the one the gate compiled")
    # MEMBERSHIP AND ORDER, which per-string digests alone do not pin. The dated
    # block spelled this as a hash of the two strings concatenated in
    # CERTIFIED_KERNELS order; a gate payload cannot produce that digest, and a
    # recorder that computed it from the tree would be pinning the tree against
    # itself. The same property, stated directly: the block's certified list IS the
    # module's literal, in order, and the digest table covers exactly those names.
    certified = list(module_level_literal(module_source(), "CERTIFIED_KERNELS"))
    assert sorted(block["certified_kernels"]) == sorted(certified), (
        f"the record certifies {sorted(block['certified_kernels'])} and the module "
        f"declares {sorted(certified)}")
    assert set(recorded) == set(certified), (
        f"the device-digest table covers {sorted(set(recorded))}, not the certified "
        f"set {sorted(certified)}")
    assert len(sources) == len(certified), (
        f"{len(sources)} device strings in the module against {len(certified)} "
        f"certified names")

    # And any subject the gate read that has moved since must be declared.
    revisions = {name: digest for name, digest
                 in (block.get("revision_sha256") or {}).items()
                 if not name.startswith("_")}
    edits = block["post_certification_edits"]
    assert (revisions and edits) or not (revisions or edits), (
        "a declared revision with no edit entry beside it, or an edit entry naming "
        "no revised subject, is half a declaration")
    assert block["subject_sha256"], "a block that names no subject pins no bytes"
    for name, digest in block["subject_sha256"].items():
        live = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        if live == digest:
            continue
        assert name in revisions, (
            f"{name} has changed since the gate ran and no post_certification_edits "
            f"entry says why")
        assert revisions[name] == live, (
            f"{name} has changed AGAIN since the recorded post-gate edit")
    for entry in edits:
        assert entry["touches_device_code"] is False, (
            "a post-gate edit that touched device code invalidates the verdict")


def test_the_shipping_block_is_a_re_cut_of_the_dated_release_and_agrees_with_it():
    """A re-cut that quietly certified DIFFERENT kernels would be a new release.

    The two blocks have to agree on what was measured -- the module, the certified
    names, both policies and the device strings -- or the newer one is not a re-cut of
    the older and :data:`SHIPPING_RECORD_BLOCK` is pointing at the wrong evidence. The
    dated block itself is never edited: it records what the original release was cut
    against, including the post-gate edits that carried it, and that is its whole
    value.
    """
    import json  # noqa: PLC0415

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    shipping = record[SHIPPING_RECORD_BLOCK]
    dated = record[shipping["re_cuts"]]
    assert shipping["passed"] is True
    assert shipping["kernel_module"] == dated["kernel_module"]
    assert set(shipping["certified_kernels"]) == set(dated["certified_kernels"])
    assert sorted(shipping["policies_cut_under"]) == \
        sorted(dated["policies_cut_under"])
    assert shipping["multi_step_budget"] == dated["multi_step_budget"]
    for name, digest in shipping["device_source_sha256"].items():
        assert dated["device_source_sha256"][name] == digest, (
            f"{name}'s device string moved between the release and the re-cut, so "
            f"this is a new certification rather than a re-measurement")
    # Both legs released, and the unguarded control diverged on both -- without that
    # the guarded 242/242 is a number with nothing to be measured against.
    for policy in ("keep", "flush"):
        leg = shipping["per_policy"][policy]
        assert leg["released"] is True, policy
        assert leg["guard_control_diverged"] is True, policy
        identical, ran = leg["single_launch"].split("/")
        assert identical == ran and int(ran) > 0, policy
        as_required, total = leg["mutation_legs_as_required"].split("/")
        assert as_required == total and int(total) > 0, policy


def test_the_docstring_records_the_verdict_and_what_it_does_not_cover():
    """A reader who reaches the kernel before the record must still be told.

    Four things by name: that it is certified and by what, the WRONG-DRIVE
    control (the one leg a no-PML test can never supply), the multi-step budget
    that carries the rotation, and -- because certification is a statement about
    bytes and not about wiring -- that nothing dispatches it.

    AND THE THIN LEG, WITH ITS COUNT. The gate's fold coverage is two guarded
    single-launch cases on one X plane, with nothing folded in the multi-step leg
    or the mutation battery. The first transcription of this said the gate swept
    unfolded grids ONLY, which the case records disprove -- so the docstring is
    required to carry the NUMBER rather than an adjective, and the number is
    checked against ``certification.json``'s ``fold_coverage`` block so the two
    cannot drift into two different stories.
    """
    import json  # noqa: PLC0415

    doc = ast.get_docstring(ast.parse(module_source()))
    for phrase in ("CERTIFIED 2026-08-19", "ade_2026-08-19", "WRONG-DRIVE CONTROL",
                   "multi-step", "Nothing dispatches this",
                   "stays a PROJECTION"):
        assert phrase in doc, f"the docstring does not say {phrase!r}"

    fold = json.loads(RECORD.read_text(encoding="utf-8"))["ade_2026-08-19"][
        "fold_coverage"]
    assert fold["folded_cases_in_the_multi_step_leg"] == 0
    assert fold["folded_cases_in_the_mutation_battery"] == 0
    assert 0 < fold["guarded_cases_that_are_folded"] < fold["guarded_cases_per_policy"]
    assert fold["fold_spec"].split(":")[0] in doc, (
        "the docstring does not name the one fold spec the gate swept")
    assert str(fold["guarded_cases_per_policy"]) in doc


def test_the_compile_options_match_all_three_siblings():
    """``--fmad=false`` is CORRECTNESS here and the tuple is spelled per file.

    Spelled rather than imported so a by-path load cannot pick up a different
    tuple than a gate compiled -- which makes four copies, and this is what stops
    them drifting. There are three contraction candidates in this family's single
    line: both additions can absorb the product to their left.
    """
    tuples = {}
    for path in (KERNEL_MODULE, CONSTITUTIVE_MODULE, OFFDIAG_MODULE, CURL_MODULE):
        tuples[path.name] = module_level_literal(
            path.read_text(encoding="utf-8"), "_COMPILE_OPTIONS")
    assert set(map(tuple, tuples.values())) == {("--fmad=false",)}, tuples


# --------------------------------------------------------------------------
# THE TRANSCRIBED TABLES
# --------------------------------------------------------------------------

def test_the_susceptibility_kinds_match_the_engine_and_the_sibling_track():
    """Three copies of one literal, pinned together.

    ``coverage.py`` imports nothing, so the kinds are written out; this is what
    stops the transcription drifting from ``dispersion`` when a kind is added.
    A new kind reaching the engine must fail HERE -- the recurrence this family
    transcribes is the Lorentz/Drude one, and another kind is a different
    difference equation wearing the same coefficient triple.
    """
    assert coverage.COVERED_SUSCEPTIBILITY_KINDS == dispersion.SUSCEPTIBILITY_KINDS
    assert (coverage.COVERED_SUSCEPTIBILITY_KINDS
            == triton_coverage.COVERED_SUSCEPTIBILITY_KINDS)


def test_the_driven_components_match_the_engine():
    assert coverage.ADE_ELECTRIC_COMPONENTS == dispersion.E_COMPONENTS
    assert coverage.ADE_ELECTRIC_COMPONENTS == triton_coverage.ELECTRIC_COMPONENTS


def test_the_admitted_boundary_kinds_include_the_fold_and_exclude_the_axis():
    """The one clause that diverges from both sibling predicates, as data.

    A fold is admitted and the cylindrical axis is not. That asymmetry is
    deliberate and is worth naming in a test, because the obvious edit -- copying
    the sibling's ``kind not in BC_CODES`` clause across -- silently costs five
    corpus rows and would look like a tidy-up.
    """
    assert coverage.MIRROR in coverage.ADE_BOUNDARY_KINDS
    assert coverage.CYL_AXIS not in coverage.ADE_BOUNDARY_KINDS
    assert set(coverage.BC_CODES) < set(coverage.ADE_BOUNDARY_KINDS)


def test_the_mirror_admission_record_is_self_consistent():
    """The projection this family's reach rests on, kept honest.

    Not a device verdict and the record says so; it is what a gate would have to
    cover. The five named rows ARE the difference between the two counts, so the
    numbers cannot drift apart from the row list.
    """
    record = coverage.ADE_MIRROR_ADMISSION
    assert (record["admitted_with_mirror"] - record["admitted_without_mirror"]
            == len(record["mirror_rows"]) == 5)
    assert record["admitted_with_mirror"] <= record["rows_with_update_P"] == 15
    assert (HERE.parents[1] / "parity" / "meep_gpu" / "results"
            / pathlib.Path(record["record"]).name).is_dir(), (
        "the record directory the admission cites does not exist")


# --------------------------------------------------------------------------
# THE PREDICATE, AND EVERY CLAUSE MUTATED
# --------------------------------------------------------------------------

#: The shim's grid, named once so a case that has to construct a volume itself
#: cannot drift from the shape the predicate is told about.
SHAPE = (4, 5, 6)

#: One exception instance, so a refusal that embeds its repr is still an exact
#: string rather than a moving target.
_BOOM = RuntimeError("no")


class _Raising:
    """An object that raises ``_BOOM`` from whatever the predicate reads.

    Iterated for the polarization list and the coefficient triple, read for a
    dtype as a buffer. A RAISE IS NOT A REFUSAL unless something catches it, and
    these are the cases that say which clause catches it.
    """

    def __iter__(self):
        raise _BOOM

    @property
    def dtype(self):
        raise _BOOM


class _Opaque:
    """A volume that passes every array check and exposes NO base address.

    The alias clauses rest on ``_base_address``; this is the input that makes it
    return None, which is a refusal of its own and not an alias.
    """

    def __init__(self, shape):
        self.dtype = np.float32
        self.shape = shape
        self.flags = types.SimpleNamespace(c_contiguous=True)


def _shim(**over):
    """A configuration the predicate can answer about, with one thing wrong.

    ``xp`` is a module literally named ``cupy`` carrying NumPy's ``float32``,
    which is the same shim the 186-row battery uses: it satisfies the backend
    clause and NOTHING else, so every other clause is exercised for real.

    EVERY KNOB DRIVES EXACTLY ONE CLAUSE, and that is the property the table
    below rests on. Two facts a real ``Grid`` always carries together are still
    separate knobs here. ``cylindrical`` and ``axis`` are the case that forced
    it: the shim used to derive ``is_axis(0)`` from ``cylindrical``, so the Dcyl
    case was refused by the r=0 AXIS clause whether or not the cylindrical clause
    existed -- measured on 2026-08-19, deleting ``if facts["cylindrical"]:``
    outright still passed all 40 tests. ``p_dtype`` and ``drive_dtype`` are the
    same lesson on the array checks: one knob for both meant the DRIVE's check
    answered for a mutation of the buffers' own.
    """
    shape = over.get("shape", SHAPE)
    xp = types.ModuleType("cupy")
    xp.float32 = np.float32

    def volume(dtype=np.float32, form=None):
        form = shape if form is None else form
        if over.get("unallocated"):
            # A zero-stride view. The int32 clause asks about a cell count no
            # machine can allocate, so the only way to reach it is not to.
            return np.broadcast_to(np.zeros(1, dtype=dtype), form)
        return np.zeros(form, dtype=dtype)

    class Grid:
        cylindrical = over.get("cylindrical", False)
        has_bloch = over.get("has_bloch", False)
        bfast_active = over.get("bfast", False)
        beta = over.get("beta", 0.0)
        dt = 0.01

        def has_symmetry(self):
            return over.get("symmetry", False)

        def is_mirrored(self, axis):
            return over.get("symmetry", False) and axis == 0

        def is_axis(self, axis):
            # NOT read off ``cylindrical``: see the docstring.
            return over.get("axis", False) and axis == 0

        def is_metallic(self, axis):
            return over.get("metallic", False)

    Grid.xp = xp
    Grid.shape = shape

    class Susceptibility:
        kind = over.get("kind", "lorentzian")

    class State:
        susceptibility = Susceptibility()

        def __init__(self):
            self.grid = Grid()
            self._coefficients = over.get("coefficients", (0.97, -0.41, 0.13))
            buffer_dtype = over.get("p_dtype", np.float32)
            if "p_buffer" in over:
                self.P = {"Ez": over["p_buffer"]}
            else:
                self.P = {"Ez": volume(buffer_dtype, over.get("p_shape"))}
            self.P_prev = {"Ez": volume(buffer_dtype)}
            if over.get("alias"):
                self._scratch = self.P["Ez"]
            elif "scratch" in over:
                self._scratch = over["scratch"]
            else:
                self._scratch = volume(buffer_dtype)
            if over.get("sigma_aliases_p"):
                self.sigma = {"Ez": self.P["Ez"]}
            else:
                self.sigma = {"Ez": over.get("sigma", 0.5)}

        def driven(self):
            if over.get("driven_raises"):
                raise _BOOM
            return over.get("driven", ("Ez",))

        def drives(self, component):
            if over.get("drives_raises"):
                raise _BOOM
            return component in over.get("driven", ("Ez",))

    class Fields:
        force_complex_fields = over.get("complex", False)
        _chi2_components = over.get("chi2", None)
        _chi3_components = over.get("chi3", None)

        def __init__(self):
            self.grid = Grid()
            self._pml_active = over.get("pml_storage", True)
            if over.get("states_unreadable"):
                self.polarizations = _Raising()
            elif over.get("no_states"):
                self.polarizations = ()
            else:
                self.polarizations = (State(),)
            if over.get("drive_aliases_p"):
                self.f_w_Ez = self.polarizations[0].P["Ez"]
            elif not over.get("no_drive"):
                self.f_w_Ez = volume(over.get("drive_dtype", np.float32))
            self.Ez = volume()
            if over.get("no_drive_field"):
                self.drive_field = None

        def drive_field(self, component):
            if over.get("drive_raises"):
                raise _BOOM
            if over.get("wrong_drive"):
                return self.Ez
            return getattr(self, "f_w_" + component)

    class Layer:
        is_active = over.get("pml", True)

    fields = Fields()
    return fields, Layer(), fields.grid


def _verdict(**over):
    fields, layer, grid = _shim(**over)
    state = fields.polarizations[0]
    return coverage.covers_real_pml_ade_component(
        fields, layer, grid, state, over.get("component", "Ez"))


def test_a_well_formed_configuration_is_admitted():
    """A fail-closed predicate that refuses everything is not fail-closed, it is
    broken -- and every mutation below would 'pass' against one."""
    assert _verdict() == (True, "covered")
    assert _verdict(sigma=np.zeros(SHAPE, dtype=np.float32))[0], (
        "a graded sigma volume is the other compiled variant and must be covered")
    assert _verdict(metallic=True)[0], (
        "a metallic wall is on ADE_BOUNDARY_KINDS; the boundary loop must not "
        "refuse what the table admits")


def test_a_mirror_fold_is_admitted_and_that_is_the_point():
    """``stepping.update_P`` runs the whole stored array under a fold.

    Worth five of the corpus's fifteen ``update_P`` rows. The curl and
    constitutive predicates refuse a fold because they index a per-axis
    coefficient VECTOR by cell coordinate; this sub-step indexes nothing but a
    flat element.
    """
    assert _verdict(symmetry=True) == (True, "covered")


#: One case per clause of :func:`coverage.covers_real_pml_ade_component`, each
#: carrying the WHOLE refusal string that clause and only that clause produces.
#:
#: THE STRINGS ARE TRANSCRIBED, not imported or matched by fragment, and that is
#: the point of the table. A fragment is satisfied by a NEIGHBOUR: measured on
#: 2026-08-19 by an adversarial sweep that deleted one clause at a time, two of
#: the nineteen fragment cases still passed with their own clause gone --
#: ``if facts["cylindrical"]:`` deleted was answered by ``axis 0 is the
#: cylindrical r = 0 axis`` and the deleted buffer array-check loop by
#: ``f_w_Ez is float64, not float32``. Both fragments were in the neighbour's
#: text. Equality on the whole string is what makes a clause answer for itself,
#: so a REWORDED refusal must be re-transcribed here rather than re-fragmented.
CLAUSE_CASES = [
    # -- the drive, the clause a no-PML test can never catch -----------------
    ("wrong drive bound", dict(wrong_drive=True),
     "drive_field('Ez') is not f_w_Ez; the kernel binds what drive_field "
     "returns and this run would drive P from the wrong array"),
    ("drive_field raised", dict(drive_raises=True),
     "drive_field('Ez') raised RuntimeError('no')"),
    ("drive_field is not callable", dict(no_drive_field=True),
     "fields does not expose drive_field()"),
    ("the drive field is not allocated", dict(no_drive=True),
     "f_w_Ez is not allocated (the drive field under PML)"),
    ("Fields not in PML storage", dict(pml_storage=False),
     "the layer is active but Fields is not in PML storage mode; drive_field "
     "would hand back the stored Ez, which is the constitutive product only "
     "OUTSIDE the absorber"),
    ("the drive is float64", dict(drive_dtype=np.float64),
     "f_w_Ez is float64, not float32"),
    ("the drive aliases P", dict(drive_aliases_p=True),
     "f_w_Ez aliases P['Ez']; the drive is read while the scratch is written"),
    # -- storage and layer ---------------------------------------------------
    ("no active layer", dict(pml=False),
     "no active PML layer: this family binds f_w as the drive and drive_field "
     "returns the stored E without one"),
    ("complex64 storage", dict(complex=True),
     "complex64 storage: the recurrence is the same but the storage is not"),
    # -- the grid ------------------------------------------------------------
    ("cylindrical", dict(cylindrical=True),
     "cylindrical (Dcyl): element-wise like every other extent, and refused "
     "because no hand-CUDA family has been measured on one at any sub-step"),
    ("the cylindrical r = 0 axis", dict(axis=True),
     "axis 0 is the cylindrical r = 0 axis"),
    ("nonzero Bloch k", dict(has_bloch=True),
     "nonzero Bloch k: the wrapped plane carries a phase real storage cannot "
     "hold"),
    ("BFAST", dict(bfast=True),
     "BFAST: a second additive term on every curl target"),
    ("special_kz", dict(beta=0.3),
     "special_kz (grid.beta != 0): extra out-of-plane coupling terms"),
    ("instantaneous chi2", dict(chi2={"Ez": object()}),
     "instantaneous chi2/chi3: a Pade factor replaces the constitutive product "
     "this sub-step's drive comes from"),
    ("instantaneous chi3", dict(chi3={"Ez": object()}),
     "instantaneous chi2/chi3: a Pade factor replaces the constitutive product "
     "this sub-step's drive comes from"),
    ("grid shape is not 3-D", dict(shape=(4, 5)),
     "grid shape (4, 5) is not three-dimensional"),
    ("the cell count overflows int32",
     dict(shape=(1291, 1291, 1291), unallocated=True),
     "2151685171 cells exceeds the kernel's int32 index range"),
    # -- the susceptibility and its coefficients -----------------------------
    ("uncovered kind", dict(kind="debye"),
     "kind 'debye' is outside ('lorentzian', 'drude')"),
    ("component not driven", dict(driven=("Ex",)),
     "this susceptibility does not drive Ez"),
    ("drives() raised", dict(drives_raises=True),
     "drives('Ez') raised RuntimeError('no')"),
    ("component outside E", dict(component="Bx"),
     "component 'Bx' is outside ('Ex', 'Ey', 'Ez')"),
    ("the coefficient triple is unreadable", dict(coefficients=_Raising()),
     "the coefficient triple is unreadable (RuntimeError: no)"),
    ("short coefficient tuple", dict(coefficients=(1.0, 2.0)),
     "the (c_now, c_prev, c_drive) triple has 2 entries, not 3"),
    ("non-numeric coefficient", dict(coefficients=(1.0, "x", 3.0)),
     "c_prev='x' is not a float"),
    ("NaN coefficient", dict(coefficients=(1.0, 2.0, float("nan"))),
     "c_drive is NaN"),
    ("infinite coefficient", dict(coefficients=(1.0, 2.0, float("inf"))),
     "c_drive=inf is not finite"),
    # -- the three rotating buffers ------------------------------------------
    ("P is not allocated", dict(p_buffer=None),
     "P['Ez'] is not allocated"),
    ("P is float64", dict(p_dtype=np.float64),
     "P['Ez'] is float64, not float32"),
    ("P has the wrong shape", dict(p_shape=(4, 5, 7)),
     "P['Ez'] has shape (4, 5, 7), not the grid's (4, 5, 6)"),
    ("P is not C-contiguous",
     dict(p_buffer=np.zeros((4, 5, 12), dtype=np.float32)[:, :, ::2]),
     "P['Ez'] is not C-contiguous"),
    ("P cannot be inspected", dict(p_buffer=_Raising()),
     "P['Ez'] could not be inspected: RuntimeError: no"),
    ("the scratch has no readable base address", dict(scratch=_Opaque(SHAPE)),
     "_scratch exposes no readable base address; the rotation cannot be shown "
     "to be alias-free"),
    ("the scratch aliases P", dict(alias=True),
     "_scratch aliases P['Ez']; the recurrence reads P and P_prev while "
     "writing the scratch, and the rotation that follows would advance one "
     "buffer twice"),
    # -- sigma, on both compiled variants ------------------------------------
    ("sigma is missing", dict(sigma=None),
     "sigma['Ez'] is missing"),
    ("the sigma volume is float64",
     dict(sigma=np.zeros(SHAPE, dtype=np.float64)),
     "sigma['Ez'] is float64, not float32"),
    ("the sigma volume aliases P", dict(sigma_aliases_p=True),
     "sigma['Ez'] aliases P['Ez']; the coefficient volume is read while the "
     "scratch is written"),
    ("sigma is neither a scalar nor a volume", dict(sigma="x"),
     "sigma['Ez']='x' is neither a scalar nor a volume"),
    ("non-finite sigma", dict(sigma=float("nan")),
     "sigma['Ez']=nan is not finite"),
]


def test_no_two_clauses_are_expected_to_say_the_same_thing():
    """The property that makes equality above mean 'this clause and no other'.

    Two clauses sharing a refusal string would be indistinguishable from the
    outside, and the table could not tell which one answered. The chi2 and chi3
    halves are the one deliberate pair -- they ARE one clause with two inputs --
    so they are named here rather than allowed by a weaker rule.
    """
    reasons = [reason for _, _, reason in CLAUSE_CASES]
    duplicates = {reason for reason in reasons if reasons.count(reason) > 1}
    assert duplicates == {
        "instantaneous chi2/chi3: a Pade factor replaces the constitutive "
        "product this sub-step's drive comes from"}, duplicates


@pytest.mark.parametrize("label,over,reason", CLAUSE_CASES)
def test_every_clause_refuses_with_its_own_whole_string(label, over, reason):
    """One mutation per clause, each required to be caught BY THAT CLAUSE.

    Asserting only ``not covered`` would pass if an unrelated clause fired
    first, which is how a widened predicate hides: it still refuses the case,
    for the wrong reason, and admits the neighbouring one it should not.
    Asserting a FRAGMENT is the same hole one step smaller -- two clauses caught
    each other's mutation through a shared substring until this became equality.
    """
    covered, actual = _verdict(**over)
    assert not covered, f"{label} was ADMITTED: a clause is missing"
    assert actual == reason, f"{label} refused for the wrong reason: {actual!r}"


def test_the_backend_clause_fires_before_anything_else():
    """The battery factors its coverage number on this ordering.

    A NumPy host must refuse for the backend and not for whatever else happens to
    be wrong, or the residual-clause census counts the wrong thing.
    """
    fields, layer, grid = _shim()
    grid.xp = np  # __name__ == "numpy"
    covered, reason = coverage.covers_real_pml_ade_component(
        fields, layer, grid, fields.polarizations[0], "Ez")
    assert (covered, reason) == (False, "backend is not CuPy")


def test_an_unreadable_grid_is_a_refusal_and_not_a_raise():
    """A RAISE IS NOT A REFUSAL unless something catches it.

    The predicate is read by a planner deciding whether to dispatch, so an
    exception escaping it is a crashed run where a fail-closed no was correct.
    """
    fields, layer, grid = _shim()

    def explode(axis):
        raise RuntimeError("this grid cannot say")

    grid.is_metallic = explode
    covered, reason = coverage.covers_real_pml_ade_component(
        fields, layer, grid, fields.polarizations[0], "Ez")
    assert (covered, reason) == (False, (
        "the grid could not answer a question this predicate has to ask: "
        "RuntimeError: this grid cannot say"))


def test_the_boundary_clause_refuses_a_kind_that_was_never_admitted():
    """The one clause no ``Grid`` can reach today, exercised by substitution.

    ``_boundary_kinds_from`` resolves to four spellings; three are on
    ``ADE_BOUNDARY_KINDS`` and the fourth, the cylindrical axis, is refused a
    clause earlier. So the loop is a GHOST-RULE guard -- it is there so a kind
    added to the resolver later is refused BECAUSE it was never admitted -- and
    the only way to reach it is to add one. Nothing else in this file pins it:
    deleting the loop outright failed no test before this.

    Patched by hand rather than through the ``monkeypatch`` fixture so the
    adversarial clause sweep, which calls these functions directly, runs it too.
    """
    fields, layer, grid = _shim()
    original = coverage._boundary_kinds_from
    coverage._boundary_kinds_from = lambda facts: ("periodic", "ghost", "metallic")
    try:
        covered, reason = coverage.covers_real_pml_ade_component(
            fields, layer, grid, fields.polarizations[0], "Ez")
    finally:
        coverage._boundary_kinds_from = original
    assert (covered, reason) == (False, (
        "axis 1 resolves to boundary 'ghost', which this sub-step has never "
        "been measured on"))


@pytest.mark.parametrize("label,over,reason", [
    ("the polarization list is unreadable", dict(states_unreadable=True),
     "the polarization list is unreadable (RuntimeError('no'))"),
    ("driven() raised", dict(driven_raises=True),
     "polarization 0: driven() raised RuntimeError('no')"),
    ("a component refusal carries the component's own text",
     dict(sigma=float("nan")),
     "polarization 0 Ez: sigma['Ez']=nan is not finite"),
])
def test_every_sub_step_clause_refuses_with_its_own_whole_string(label, over, reason):
    """The same discipline one level up, where the reasons gain a row prefix."""
    fields, layer, grid = _shim(**over)
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (
        False, reason), label


def test_the_sub_step_verdict_is_all_or_nothing_over_every_component():
    """Sub-steps compose; halves of one sub-step do not.

    ``PolarizationState.update`` rotates ONE shared scratch from component to
    component inside a single call, so covering two of three and leaving the
    third to the array path would interleave two rotations over one buffer set.
    """
    fields, layer, grid = _shim()
    state = fields.polarizations[0]
    zeros = lambda: np.zeros(SHAPE, dtype=np.float32)
    for component in ("Ex", "Ey"):
        state.P[component] = zeros()
        state.P_prev[component] = zeros()
        state.sigma[component] = 0.5
        setattr(fields, "f_w_" + component, zeros())
    state.driven = lambda: ("Ex", "Ey", "Ez")
    state.drives = lambda c: c in ("Ex", "Ey", "Ez")
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (
        True, "covered")

    # One component broken is the whole sub-step refused, and it says which.
    state.sigma["Ey"] = float("nan")
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (
        False, "polarization 0 Ey: sigma['Ey']=nan is not finite")


def test_the_two_no_op_clauses_are_told_apart():
    """``stepping.update_P`` returns immediately, so there is no sub-step to take
    over; a covered verdict would put a launch where the array path does nothing.

    TWO clauses say ``no-op`` and the old assertion checked that substring, so
    either one satisfied it: with ``if not states:`` deleted the driven-count
    clause answered instead, same substring, different reason, nothing failed.
    Each is now pinned to the configuration only it can see.
    """
    fields, layer, grid = _shim(no_states=True)
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (
        False, "no polarization is registered; update_P is a no-op")

    fields, layer, grid = _shim(driven=())
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (
        False, "no driven component exists; update_P is a no-op")


def test_sigma_is_volume_is_decided_in_exactly_one_place():
    """The compile-time choice and the clause cannot disagree.

    Out of step, it binds a scalar where the signature declares a pointer, or the
    reverse: a wrong answer, not a crash -- the sibling gate's mutation m13. The
    launcher is required to reach the same function rather than re-deriving the
    answer from ``sigma.shape`` itself.
    """
    fields, _, _ = _shim()
    state = fields.polarizations[0]
    assert coverage.ade_sigma_is_volume(state, "Ez") is False
    state.sigma["Ez"] = np.zeros((4, 5, 6), dtype=np.float32)
    assert coverage.ade_sigma_is_volume(state, "Ez") is True

    source = module_source()
    assert "ade_sigma_is_volume(state, component)" in source, (
        "the launcher must ask the predicate's own decider, not re-derive it")
    assert not re.search(r"getattr\(sigma, [\"']shape", source), (
        "the launcher re-derives SIGMA_IS_VOLUME instead of asking")


def test_the_launcher_resolves_pointers_after_the_rotation_not_before():
    """THE DEFECT A SINGLE-LAUNCH GATE CANNOT SEE, pinned structurally.

    ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` per
    component, and the retired history becomes the NEXT component's scratch. A
    launcher that read the three once and reused the views would be stale from
    the second component of the first step -- and still compute, advancing one
    buffer twice and freezing another. Measured: such a launcher differs from the
    array path on 120 of 180 words after ONE step.

    Structural, because the numerical version needs a device. The reads must sit
    inside the per-component loop, and the rotation must follow the launch.
    """
    tree = ast.parse(module_source())
    launcher = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "update_P_fused_pml_real")
    loops = [node for node in ast.walk(launcher) if isinstance(node, ast.For)]
    assert len(loops) == 2, "expected a state loop and a component loop"
    component_loop = loops[-1]
    statements = [ast.unparse(node) for node in component_loop.body]

    # THE READS AS WHOLE STATEMENTS, not as substrings of the loop body. Asking
    # whether "state.P[component]" appears anywhere is satisfied by the ROTATION
    # that writes it, so a launcher that cached the views and only wrote them back
    # would pass -- measured, on 2026-08-19: that mutation was MISSED by the
    # substring form of this assertion and is caught by this one.
    for read in ("w = drive(component)",
                 "p = state.P[component]",
                 "p_prev = state.P_prev[component]",
                 "scratch = state._scratch",
                 "sigma = state.sigma[component]"):
        assert read in statements, (
            f"{read!r} is not a statement of the per-component loop; the "
            f"semantic pointers must be resolved after the previous component's "
            f"rotation, never before the loop")

    launch = next(i for i, text in enumerate(statements) if text.startswith("_launch("))
    rotation = [i for i, text in enumerate(statements)
                if text.startswith(("state.P[component] =",
                                    "state.P_prev[component] =",
                                    "state._scratch ="))]
    assert rotation == sorted(rotation) and min(rotation) > launch, (
        "the rotation must follow the launch, so a raised launch leaves the "
        "state as it found it rather than half-advanced")
    assert statements[rotation[0]] == "state.P[component] = scratch"
    assert statements[rotation[1]] == "state.P_prev[component] = p"
    assert statements[rotation[2]] == "state._scratch = p_prev"


# --------------------------------------------------------------------------
# THE LAUNCH ARGUMENT LIST — the leg that was measured once and thrown away
# --------------------------------------------------------------------------
#
# MEASURED 2026-08-19 by an adversarial sweep: 22 mutations against the 38 tests
# that shipped with this module left NINE survivors, and every survivor was in
# ``_launch``. Four are failure modes this module's own docstring calls fatal:
#
#     p_out := p_now (write P in place)            38 passed
#     swap c_now / c_prev in the launcher unpack   38 passed
#     swap drive / sigma at the kernel call        38 passed
#     blocks = n_elem // 256 (ceiling dropped)     38 passed
#
# The structural tests pin the READS and the ROTATION right-hand sides; nothing
# pinned the argument list itself. None of these is a device-only defect — a fake
# kernel that EXECUTES the recorded arguments catches all four on a laptop in
# under a second, which is the strongest evidence available without a GPU.

_ADE_EXPRESSION_NOTE = (
    "transcribed from the device string in ade_kernels.py, association intact: "
    "p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w))")


def _executing_kernel(launches):
    """A fake kernel that RUNS the recurrence out of the argument list it is given.

    Every defect in ``_launch`` reaches this function as a different argument
    tuple or a different launch geometry, so the numbers move. The association is
    transcribed, not re-derived: ``((p*c_now) + (c_prev*q)) + (c_drive*(s*w))``.
    """
    def kernel(grid, block, args):
        p_out, p_now, p_prev, sigma, drive, c_now, c_prev, c_drive, n_elem = args
        blocks, = grid
        threads, = block
        n = int(n_elem)
        launches.append({"blocks": int(blocks), "threads": int(threads),
                         "reach": int(blocks) * int(threads), "n_elem": n,
                         "wrote_into_input": p_out is p_now or p_out is p_prev})
        flat_out = np.asarray(p_out).reshape(-1)
        p = np.asarray(p_now).reshape(-1)
        q = np.asarray(p_prev).reshape(-1)
        w = np.asarray(drive).reshape(-1)
        s = (np.asarray(sigma).reshape(-1) if np.ndim(sigma) else
             np.full(p.shape, np.float32(sigma), np.float32))
        reach = min(int(blocks) * int(threads), n)   # the guard, honoured here
        idx = np.arange(reach)
        flat_out[idx] = (((p[idx] * np.float32(c_now))
                          + (np.float32(c_prev) * q[idx]))
                         + (np.float32(c_drive) * (s[idx] * w[idx])))
    return kernel


@pytest.fixture
def ade_module(monkeypatch):
    """Import ``ade_kernels`` behind a fake ``cupy``.

    THE REASON THIS FIXTURE EXISTS IS THE REASON THE LEG WAS MISSING. This file's
    docstring says nothing here imports ade_kernels because it imports CuPy at
    module scope — true, and it is also why ``_launch`` shipped untested while
    four docstring-fatal defects in it passed 38 green tests. ``_launch`` itself
    touches only NumPy; the CuPy dependency is the COMPILER's, not the
    launcher's. A stand-in module satisfies the import and nothing else, so the
    argument list can be executed here rather than only on a device.

    Installed through monkeypatch so sys.modules is restored: a real CuPy host
    must not inherit this stand-in.
    """
    fake = types.ModuleType("cupy")
    fake.float32 = np.float32
    fake.ndarray = np.ndarray
    fake.RawKernel = object
    cuda = types.ModuleType("cupy.cuda")
    compiler = types.ModuleType("cupy.cuda.compiler")
    cuda.compiler = compiler
    fake.cuda = cuda
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.compiler", compiler)
    monkeypatch.delitem(sys.modules, "meep_gpu.cuda_kernels.ade_kernels",
                        raising=False)
    import importlib
    module = importlib.import_module("meep_gpu.cuda_kernels.ade_kernels")
    yield module
    sys.modules.pop("meep_gpu.cuda_kernels.ade_kernels", None)


def test_the_launch_argument_list_is_pinned_by_execution(ade_module):
    """Run the launcher against a kernel that executes what it is handed.

    This is the leg whose absence let four docstring-fatal defects pass 38 green
    tests. It asserts the NUMBERS, so an argument swap cannot hide behind a
    structural check that only reads the source.
    """
    ade_kernels = ade_module

    rng = np.random.default_rng(20260819)
    shape = (3, 4, 5)
    p_now = rng.standard_normal(shape, dtype=np.float32)
    p_prev = rng.standard_normal(shape, dtype=np.float32)
    drive = rng.standard_normal(shape, dtype=np.float32)
    out = np.zeros(shape, dtype=np.float32)
    sigma = np.float32(0.375)
    coefficients = (np.float32(1.5), np.float32(-0.25), np.float32(0.125))

    launches = []
    ade_kernels._launch(_executing_kernel(launches), out, p_now, p_prev,
                        sigma, drive, coefficients, sigma_is_volume=False)

    assert len(launches) == 1, launches
    record = launches[0]

    # THE GEOMETRY MUST COVER EVERY ELEMENT. A dropped ceiling leaves a tail
    # unwritten, which is silent in any test that does not count.
    assert record["reach"] >= record["n_elem"], (
        f"launch geometry reaches {record['reach']} of {record['n_elem']} "
        f"elements — the block count lost its ceiling")
    assert record["n_elem"] == out.size

    # P MUST NOT BE WRITTEN IN PLACE: the recurrence reads P and P_prev while
    # writing the result, so aliasing the destination onto either input
    # corrupts the read for every later element in the same launch.
    assert not record["wrote_into_input"], (
        "the destination aliases one of the inputs — P was written in place")

    expected = (((p_now * coefficients[0]) + (coefficients[1] * p_prev))
                + (coefficients[2] * (sigma * drive)))
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32)), (
        f"launcher produced different bytes than the reference recurrence "
        f"({_ADE_EXPRESSION_NOTE})")


def test_both_device_strings_carry_the_bounds_guard():
    """Removing the guard from ONE kernel is caught by the variant diff; from BOTH
    it was not, because no test pinned the idiom itself.

    Read off the SYNTAX TREE, not by importing — the discipline this file already
    follows for every other source assertion.
    """
    # The variants are ASSEMBLED: each is _ADE_UPDATE_P_PROLOGUE + a body, so the
    # guard and the recurrence live in different string constants. Reconstruct the
    # concatenation off the tree rather than importing, and assert on the result —
    # a guard present only in a prologue that some future variant does not use
    # would otherwise read as covered.
    tree = ast.parse(KERNEL_MODULE.read_text(encoding="utf-8"))
    constants = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                constants[node.targets[0].id] = value.value
            elif (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                  and isinstance(value.left, ast.Name)
                  and isinstance(value.right, ast.Constant)
                  and isinstance(value.right.value, str)):
                constants[node.targets[0].id] = (
                    constants.get(value.left.id, "") + value.right.value)
    sources = [text for text in constants.values() if "p_out[idx]" in text]
    assert len(sources) >= 2, (
        f"expected both sigma variants to be discoverable, found {len(sources)}")
    for source in sources:
        assert "n_elem) return;" in source, (
            "a device string lost its bounds guard; with the guard gone from "
            "BOTH variants no other test in this file notices")
