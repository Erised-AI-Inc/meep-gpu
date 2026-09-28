"""The MIRROR FOLD under complex64 storage -- the hand-CUDA curl pair, no CuPy.

WHAT THIS FAMILY IS. Complex64 field storage under a real (split-field) PML on a
grid that carries a MIRROR FOLD. ``coverage.covers_real_pml_complex_curl`` and
``coverage.covers_real_pml_complex_constitutive`` both refuse that configuration
by name -- "mirror symmetry: different ghost rule, a parity mask and two fill
passes" -- and the refusal is honest: the certified complex emitter carries two
boundary codes (``BC_PERIODIC``, ``BC_METALLIC``) and no fold branch at all, so a
folded axis has nowhere to go. THE REAL curl pair does carry the fold, since
2026-08-19/20; this module is that same delta applied to the complex bytes.

WHAT IT IS WORTH, MEASURED. On the 2026-08-20 union census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closeout/``) the
fold clause is the FIRST refusal on 32 unserved slots over 8 complex-storage rows.
Closing the fold ALONE does not buy 32, and the census says so: three of those
rows also carry ``grid.beta`` (refused for beta by an earlier clause -- they are
``complex_beta_kernels``'), and three carry an OFF-DIAGONAL epsilon whose
``update_E`` stays refused for the row product. What is left for this family is
17 slots over 5 rows:

* ``tests_param/TestEigCoeffs.test_binary_grating_special_kz__idx2`` (135x92x1,
  Mirror(Y) periodic, in-plane kx AND a kz on the collapsed axis, beta = 0) --
  4 slots;
* ``tests_param/TestModeDecomposition.test_triangular_lattice_oblique``
  (11x31x189, Mirror(X) periodic, ky and kz phased) -- 4 slots;
* ``examples/solve-cw.py`` (161x161x1, Mirror(X)+Mirror(Y), both METALLIC),
  ``tests/TestArrayMetadata.test_array_metadata`` (201x201x1, the same shape) and
  ``tests/TestHoleyWvgBands.test_fields_at_kx`` (20x122x1, Mirror(Y) periodic with
  an in-plane kx) -- 3 slots each, because all three carry an off-diagonal
  ``chi1inv`` and ``update_E`` is refused for it whatever the fold does.

THE THREE DELTAS, AND NOTHING ELSE
=============================================================================

The device source is ``complex_emitter.complex_source`` TRANSFORMED, not rewritten:
every arithmetic byte -- the word-pair addressing, the arm's three multiply
helpers, the zero cross terms, the ``((sf - f1) + (f2 - ss))`` grouping, the PML
recurrence -- is the certified complex family's, and :data:`DELTAS` is the whole
of what this module changes. Each delta asserts its own site count, so a sibling
edit that moves an anchor fails HERE, on a laptop, instead of silently emitting a
kernel with one fold branch missing.

1. **A THIRD BOUNDARY CODE.** ``#define BC_MIRROR_PERIODIC 2``, the real curl
   pair's own name (``coverage.MIRROR_PERIODIC``, step_curl_kernels.py:1709) for
   the half of ``stepping._boundary_kinds``' ``"mirror"`` answer whose stored array
   carries the slot past MEEP's owned window (``stepping._stored_past_owned``,
   stepping.py:1503-1518). A folded METALLIC axis is handed ``BC_METALLIC`` and
   gets no code of its own.
2. **THE GHOST IS THE METALLIC ZERO.** ``cshift_up`` / ``cshift_dn`` return
   ``cf_zero()`` on the new code, exactly as they do on ``BC_METALLIC``. That is
   NOT the boundary condition -- ``stepping._shift_down``'s MIRROR branch writes
   ``parity * field[2]`` (stepping.py:1870-1873) and ``_shift_up``'s writes
   ``parity * field[reflect_row]`` (:1819-1827) -- it is a value whose only
   consumer plane is dropped by delta 3. See THE GHOST IS DEAD below.
3. **THE MASK SPLITS BY TERMINATION.** ``stepping._mask_non_owned_cells``
   (:1865-1902) zeroes cell 0 on a mirrored axis for every component whose Yee
   shift there is 0 -- which a folded METALLIC axis inherits from ``BC_METALLIC``
   and a folded PERIODIC one needs stated -- and, ONLY where
   ``_stored_past_owned`` holds, also zeroes the LAST stored slot for every
   component whose Yee shift there is 1. The generated block is a character-for-
   character transcription of the real pair's, ``0.0f`` read as ``cf_zero()``, and
   ``test_complex_folded.py`` pins it against ``step_curl_kernels.py``'s text.

THE GHOST IS DEAD, AND WHY THAT ARGUMENT TRANSFERS TO COMPLEX STORAGE
=============================================================================

The real pair's fold verdict rests on an OWNERSHIP argument, not an arithmetic
one: each mirror ghost has exactly ONE consumer plane and both consumer planes are
masked. ``_shift_down`` is the D sub-step's and its ghost lands at stored cell 0,
whose target has Yee shift 0 on that axis -- the cell the mirrored arm of the mask
drops; ``_shift_up`` is the B sub-step's and its ghost lands at the last stored
slot, whose target has Yee shift 1 there -- the cell the top-plane arm drops on a
folded PERIODIC axis, and which on a folded METALLIC axis IS stepped and whose
ghost ``_shift_up`` writes as an exact zero anyway (:1781-1783).

Every step of that is an INDEX fact. It mentions no dtype, no multiply and no
rounding, so it transfers to complex64 word pairs unchanged -- with one thing to
say out loud: under complex storage "an exact zero" is the word pair
``(+0.0f, +0.0f)``, which is what ``cf_zero`` writes and what
``complex_emitter``'s own ``cf_zero`` docstring records the array path assigning
(``stepping.py:1832``, :1943, the INTEGER 0 into a complex64 array).

AN ARGUMENT IS NOT A MEASUREMENT. The real pair's split was measured on a device
before it was written (``gate_cuda_folded_curl.py``, 256 cases per policy: a
folded METALLIC axis handed ``BC_METALLIC`` was ALREADY exact 48/48, a folded
PERIODIC one diverged 0/64 with all 19,649 differing words on the mask-delta
plane) and the Triton sibling ``folded_complex.py`` says in as many words that the
same claim under complex storage is "an ARGUMENT plus a hand check" until its own
sweep runs. :data:`FOLDED_COMPLEX_ADMISSION` is where this family's device verdict
goes, and until it carries one, nothing here is certified.

A FOLDED AXIS MAY NOT CARRY A BLOCH PHASE, AND THIS FAMILY CHECKS IT
=============================================================================

``stepping._shift_up`` (:1765-1783) and ``_shift_down`` (:1815-1826) are mutually
exclusive per axis: the PERIODIC arm applies the Bloch factor, the MIRROR arm
applies the parity and no wrap factor. So a phase on a folded axis is not
"ignored", it is a configuration the array path does not express -- and the
certified complex predicate's own per-axis clause ("only a periodic wrap can carry
a phase") cannot catch it here, because this family satisfies the fold clause at
the INPUT through a proxy grid that reports the folded axis as its DECLARED kind.
:func:`_fold_reasons` therefore refuses a nonzero k component on a folded axis by
name, before delegating. Dropping that clause is a wrong answer, not a crash: a
folded PERIODIC axis would read back as an ordinary periodic wrap and the phase
would be applied to a plane the mask is about to drop -- silent on the interior.

Every corpus row this family serves has k = 0 on its folded axis, so the clause
costs 0 slots and buys the one thing a predicate is for.

THE CONSTITUTIVE SIDES BUILD NO KERNEL, AND THAT IS THE FINDING
=============================================================================

``stepping.update_H`` (:907-925) and ``update_E`` (:926-995) reach
``_apply_constitutive_pml`` (:2065-2098), which is four whole-array operations
with no shift, no ghost and no mask; ``_mask_non_owned_cells`` has three call
sites and no constitutive function is among them. The emitted complex constitutive
template says the same in its own characters -- it takes no ``bc_*``, no ``ph_*``
and no ``dtdx``, so there is no place in it for a fold to be wrong. So
:func:`covers_complex_folded_constitutive` is an ADMISSION over the CERTIFIED
complex constitutive pair with the fold clause inverted, exactly as
``special_kz_curl.covers_special_kz_constitutive`` is over the real one.

THAT READING IS NOT THE EVIDENCE. ``gate_cuda_complex_folded.py`` carries a
constitutive arm that runs the shipped complex pair against ``stepping.update_H``
/ ``update_E`` on FOLDED complex grids whose two curls have already moved the
state. A sub-step admitted because nothing was found to read the fold, with no
device leg, is admitted by argument from absence.

NO DISPATCH. Nothing in ``meep_gpu`` imports ``cuda_kernels``, so a True from the
predicates below licenses a MEASUREMENT and not a production step.

NOTHING HERE IMPORTS CUPY OR NUMPY AT MODULE SCOPE, so the predicates and the
emitter stay callable on the census laptop.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from . import complex_emitter
from . import coverage as _coverage

__all__ = (
    "CERTIFIED_KERNELS",
    "DELTAS",
    "FOLDED_COMPLEX_ADMISSION",
    "FOLDED_KERNELS",
    "UNCERTIFIED_KERNELS",
    "corpus_digest",
    "covers_complex_folded_constitutive",
    "covers_complex_folded_curl",
    "fold_mask_lines",
    "folded_complex_boundary_codes",
    "folded_source",
    "kernel_source",
    "set_kernel_source",
)

#: The two kernels this family emits, keyed by the sub-step they replace. The
#: constitutive sides are NOT here: they are served by the certified complex pair
#: unchanged (see the module docstring).
FOLDED_KERNELS: Dict[str, str] = {
    "step_B": "step_B_pml_complex_folded",
    "step_D": "step_D_pml_complex_folded",
}

#: Byte-identical to the array path with a DEVICE verdict behind it, 2026-09-01:
#: :data:`FOLDED_COMPLEX_ADMISSION` carries ``released`` True with both policy
#: artifacts, and ``certification.json``'s ``cuda_complex_folded_2026-09-01``
#: block names the same two. A name here without a record block is the failure
#: ``test_kernel_partition.py`` exists to catch.
CERTIFIED_KERNELS = (
    "step_B_pml_complex_folded",
    "step_D_pml_complex_folded",
)

#: EMPTY since 2026-09-01, and what emptied it was a RUN rather than an argument:
#: gate_cuda_complex_folded.py on the GPU host (one RTX A6000, cc 8.6, CuPy 13.5.1,
#: NVRTC 11.6), 64/64 curl cases byte-identical per sub-step at one launch and at
#: 60, both fold terminations, both plane parities, both float32 subnormal
#: policies; 15 mutations CAUGHT and 2 NULLS CONFIRMED per policy. Spelled as a
#: dict WITHOUT a type annotation for the reason ``constitutive_kernels.py``
#: records: the partition readers walk the syntax tree so they run where there is
#: no CuPy, and an annotated assignment is an ``ast.AnnAssign`` the
#: plain-assignment readers do not match -- annotating it makes the name invisible
#: and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: The two curl sub-steps, as the predicates and the launcher name them.
CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: The integer code the third boundary kind binds to, restated from
#: ``coverage.REAL_CURL_BC_CODES`` and pinned equal to it by the laptop test. It
#: is deliberately NOT added to ``coverage.BC_CODES``: that dict is shared with
#: ``complex_pml_kernels``, whose kernels have no branch for a fold, and widening
#: it there would hand a folded axis to a kernel that cannot serve one.
BC_MIRROR_PERIODIC_CODE: int = 2

#: ``i``/``j``/``k`` -- the loop variable each axis's index is spelled with in the
#: emitted source. Restated from ``complex_emitter._AXIS_INDEX`` and pinned equal.
_AXIS_INDEX: Dict[str, str] = {"x": "i", "y": "j", "z": "k"}

#: The extent each axis's TOP plane is compared against.
_AXIS_EXTENT: Dict[str, str] = {"x": "nx", "y": "ny", "z": "nz"}


#: ``x``/``y``/``z`` by index, as ``vec.hpp``'s ``cycle_direction`` orders them.
_AXIS_NAMES: Tuple[str, ...] = ("x", "y", "z")


def fold_mask_lines(axes: Sequence[str], target: int) -> str:
    """The full mask block for one curl target, given its Yee-shift-0 axes.

    ``target`` is the component's index in ``(x, y, z)`` -- 0 for Bx/Dx, and so on
    -- and is what puts the third arm's lines in the CURL'S OWN cycle order rather
    than in alphabetical order. That is not cosmetic: it is what makes this block
    character-for-character the real pair's, so ``test_complex_folded.py`` can pin
    the two with string equality instead of a set comparison plus an argument about
    why order does not matter.

    THREE ARMS, and every one of them is a wrong answer rather than a crash if it
    is dropped:

    * ``BC_METALLIC`` at cell 0 on each shift-0 axis -- the certified complex
      family's own mask, ``stepping._mask_non_owned_cells``' metallic arm;
    * ``BC_MIRROR_PERIODIC`` at cell 0 on each shift-0 axis -- the same function's
      MIRROR arm, which a folded METALLIC axis already inherits through
      ``BC_METALLIC`` (arithmetic for arithmetic) and a folded PERIODIC one does
      not;
    * ``BC_MIRROR_PERIODIC`` at the LAST stored slot on each shift-1 axis -- the
      ``_stored_past_owned`` arm (stepping.py:1934-1944), which fires on a folded
      PERIODIC axis at either count parity and must NOT fire on a folded METALLIC
      one, where MEEP steps that plane.

    ``axes`` is the target's shift-0 axis list, exactly
    ``complex_emitter._MASK_AXES[backward][target]``; the shift-1 axes are the
    COMPLEMENT, which is what makes the two arms per-axis complements and is why no
    component ever carries both on the same axis.

    THE TWO ORDERS ARE DIFFERENT AND BOTH ARE TRANSCRIBED. The shift-0 lines keep
    ``_MASK_AXES``' own (alphabetical) order, because the METALLIC half of them IS
    ``complex_emitter``'s existing block and this function only appends to it; the
    shift-1 lines run in ``cycle_direction``'s order from the target's own axis --
    ``(t+1) % 3`` then ``(t+2) % 3``, the same pair and the same order
    ``pml_apply``'s coefficient arguments already take. On the B side that is
    y,z / z,x / x,y and on the D side it is the single own axis.

    The output is a character-for-character transcription of the REAL pair's mask
    block (``step_curl_kernels.py``'s ``step_{B,D}_pml_real``) with
    ``curl = 0.0f`` read as ``curl = cf_zero()``, and ``test_complex_folded.py``
    pins it against that file's text rather than against a second copy of the
    reasoning.
    """
    for axis in axes:
        if axis not in _AXIS_INDEX:
            raise ValueError(f"unknown axis {axis!r}; expected one of x, y, z")
    if target not in (0, 1, 2):
        raise ValueError(f"target must be 0, 1 or 2, got {target!r}")
    zero_shift = tuple(axes)
    one_shift = tuple(
        _AXIS_NAMES[(target + step) % 3] for step in (0, 1, 2)
        if _AXIS_NAMES[(target + step) % 3] not in zero_shift)
    lines = []
    for axis in zero_shift:
        lines.append(f"        if (bc_{axis} == BC_METALLIC && "
                     f"{_AXIS_INDEX[axis]} == 0) curl = cf_zero();")
    for axis in zero_shift:
        lines.append(f"        if (bc_{axis} == BC_MIRROR_PERIODIC && "
                     f"{_AXIS_INDEX[axis]} == 0) curl = cf_zero();")
    for axis in one_shift:
        lines.append(f"        if (bc_{axis} == BC_MIRROR_PERIODIC && "
                     f"{_AXIS_INDEX[axis]} == {_AXIS_EXTENT[axis]} - 1) "
                     f"curl = cf_zero();")
    return "\n".join(lines)


#: WHAT EACH DELTA REPLACES AND HOW MANY SITES IT MUST FIND. The tuple is
#: ``(label, old, new, expected sites)``; the mask deltas are appended per
#: sub-step in :func:`_deltas_for`, because they differ by target.
#:
#: A SITE COUNT IS A WELD. ``complex_emitter`` is a sibling under active
#: development; if one of these anchors moves, the transform must FAIL rather than
#: emit a kernel with one fold branch missing -- which compiles, runs, and is wrong
#: on one plane of one component.
DELTAS: Tuple[Tuple[str, str, str, int], ...] = (
    (
        "third_boundary_code",
        "#define BC_METALLIC 1\n",
        "#define BC_METALLIC 1\n"
        "// A MIRROR-FOLDED axis whose termination is PERIODIC. stepping._boundary_kinds\n"
        "// resolves a fold to \"mirror\" at BOTH terminations (stepping.py:2146); this is\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 2146->2193
        "// the half whose stored array carries the slot past MEEP's owned window\n"
        "// (stepping._stored_past_owned, stepping.py:1454-1469) and which therefore needs\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 1454-1469->1503-1518
        "// the Yee-shift-1 top-plane mask. A folded METALLIC axis is handed BC_METALLIC\n"
        "// and has no code of its own: its ghost rule and its masks ARE metallic's.\n"
        "#define BC_MIRROR_PERIODIC 2\n",
        1,
    ),
    (
        "folded_ghost_is_the_metallic_zero",
        "    if (bc == BC_METALLIC) return cf_zero();\n",
        "    // BC_MIRROR_PERIODIC takes the metallic zero, and the zero is NOT the\n"
        "    // boundary condition: stepping's MIRROR branches write a parity-weighted\n"
        "    // image here (stepping.py:1772-1780 up, :1823-1826 down) whose only consumer\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 1772-1780->1819-1827, 1823-1826->1870-1873
        "    // plane the mask below drops. Under complex storage that zero is the word\n"
        "    // pair (+0.0f, +0.0f), which is what cf_zero writes.\n"
        "    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) return cf_zero();\n",
        2,
    ),
)


def _deltas_for(sub_step: str) -> Tuple[Tuple[str, str, str, int], ...]:
    """:data:`DELTAS` plus this sub-step's three mask replacements and the rename."""
    backward = complex_emitter.KERNELS[sub_step][1]
    out = list(DELTAS)
    for target, axes in enumerate(complex_emitter._MASK_AXES[backward]):
        out.append((f"fold_mask_target_{target}",
                    complex_emitter._mask_lines(axes),
                    fold_mask_lines(axes, target),
                    1))
    out.append((
        "rename",
        f'extern "C" __global__ void {complex_emitter.KERNELS[sub_step][0]}(',
        f'extern "C" __global__ void {FOLDED_KERNELS[sub_step]}(',
        1,
    ))
    return tuple(out)


def folded_source(sub_step: str, expansion) -> str:
    """The device source for one folded-complex curl sub-step under one arm.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``; ``expansion`` is an arm name or
    its code, REQUIRED and never defaulted, for the reason
    ``complex_emitter.normalized_expansion`` refuses rather than defaulting: a
    wrong arm is a wrong answer and not a crash.
    """
    if sub_step not in FOLDED_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(FOLDED_KERNELS)}, got {sub_step!r}")
    source = complex_emitter.complex_source(sub_step, expansion)
    for label, old, new, expected in _deltas_for(sub_step):
        found = source.count(old)
        if found != expected:
            raise RuntimeError(
                f"the folded-complex delta {label!r} for {sub_step} matched "
                f"{found} sites in complex_emitter's source, not {expected}; the "
                f"sibling has moved an anchor and this transform must not emit a "
                f"kernel with one fold branch missing")
        source = source.replace(old, new)
    return source


#: The mutable copy the launcher compiles, keyed by kernel name and arm code. The
#: gate mutates through :func:`set_kernel_source`; ONE seam, so a mutation cannot
#: miss a second copy.
_SOURCES: Dict[Tuple[str, int], str] = {}


def kernel_source(sub_step: str, expansion) -> str:
    """The device text this family would compile for one sub-step and arm.

    Emitted on first ask and then served from :data:`_SOURCES`, so that a gate's
    :func:`set_kernel_source` survives -- a function that re-derived from the
    emitter every call would discard the mutation and report a pass for a
    mutation it never applied.
    """
    arm = complex_emitter.normalized_expansion(expansion)
    key = (sub_step, arm)
    if key not in _SOURCES:
        _SOURCES[key] = folded_source(sub_step, arm)
    return _SOURCES[key]


def set_kernel_source(sub_step: str, expansion, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam, and only that.

    The compile memo keys on the SOURCE STRING, so a rewritten body is a miss and
    reaches NVRTC rather than being served an earlier binary.
    """
    if sub_step not in FOLDED_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(FOLDED_KERNELS)}, got {sub_step!r}")
    _SOURCES[(sub_step, complex_emitter.normalized_expansion(expansion))] = source


def reset_kernel_sources() -> int:
    """Drop every mutated body; returns how many entries went. The gate's undo."""
    count = len(_SOURCES)
    _SOURCES.clear()
    return count


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Two sub-steps times two arms is four sources; a single changed character in
    this module OR in ``complex_emitter`` moves this value, which is the point --
    the arithmetic is the sibling's and a drift there is a drift here.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for sub_step in sorted(FOLDED_KERNELS):
        for name in sorted(complex_emitter.EXPANSIONS):
            digest.update(f"{sub_step}|{name}".encode("ascii"))
            digest.update(folded_source(sub_step, name).encode("utf-8"))
    return digest.hexdigest()


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: EMPTY OF A VERDICT UNTIL THE GATE RUNS. A family whose record says "passed"
#: because its author expected it to is worse than one that says nothing, so this
#: carries the fields and ``released`` is False until
#: ``parity/meep_gpu/gate_cuda_complex_folded.py`` fills them from an artifact.
FOLDED_COMPLEX_ADMISSION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_complex_folded.py",
    # RELEASED ON DEVICE, 2026-09-01: the GPU host, one RTX A6000 (cc 8.6, CuPy
    # 13.5.1, NVRTC 11.6), under BOTH float32 subnormal policies, the arm bound
    # from the per-policy expansion probe (FMA_V1 both). The ``device_leg`` block
    # below carries the numbers, read off the two artifacts; the ``host_leg``
    # block is kept as the 2026-08-20 transcription record it always was.
    "released": True,
    "artifacts": (
        "parity/meep_gpu/results/cuda_complex_folded_2026-09-01/keep/gate.json",
        "parity/meep_gpu/results/cuda_complex_folded_2026-09-01/flush/gate.json",
    ),
    "device_leg": {
        "recorded_utc": "2026-09-01T12:33:50Z",
        "host": "the GPU host",
        "device": "NVIDIA RTX A6000",
        "compute_capability": "8.6",
        "cupy_version": "13.5.1",
        "nvrtc_version": "11.6",
        "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
        "arm": "FMA_V1",
        # 8 fixtures x 2 sub-steps x 2 courants x 2 value classes, PER POLICY.
        "curl_cases_scored": 64,
        "curl_single_launch_identical": 64,
        "curl_multi_step_identical": 64,
        "folded_periodic_cases": 48,
        "folded_metallic_cases": 16,
        "differing_words": 0,
        "fold_is_live_floor": "asserted per case; no case was skipped for it",
        # Per policy: 13 must-catch source + 4 host mutations CAUGHT, 2 nulls.
        "mutation_legs": 17,
        "mutation_legs_caught": 15,
        "mutation_legs_null_confirmed": 2,
        "mutation_legs_uncaught_or_unarmed": 0,
        # THE CONSTITUTIVE SIDES DEFERRED, AND THAT IS THE PARTITION WORKING: all
        # 32 cases per side named coverage.covers_real_pml_complex_constitutive's
        # own fold admission (2026-08-20), whose verdict is that family's gate's.
        "constitutive_cases_deferred": {"H": 32, "E": 32},
    },
    "host_leg": {
        "artifact": ("parity/meep_gpu/results/"
                     "cuda_complex_folded_2026-08-20c_host/gate.json"),
        "backend": "host-compiled (clang++), NumPy state; certifies no NVRTC",
        "recorded_utc": "2026-08-20T09:40:00Z",
        "courants": (0.5, 0.35),
        "value_classes": ("uniform", "subnormal_band"),
        "multi_step_budget": 60,
        # 8 fixtures x 2 sub-steps x 2 courants x 2 value classes.
        "curl_cases_scored": 64,
        "curl_single_launch_identical": 64,
        "curl_multi_step_identical": 64,
        "folded_periodic_cases": 48,
        "folded_metallic_cases": 16,
        "differing_words": 0,
        # THE FOLD'S OWN LIVENESS FLOOR, per case: re-launching fold-blind (the
        # folded axis handed BC_PERIODIC, which is what "no fold branch" means)
        # must change the answer, or the case measures the certified pair.
        "fold_is_live_floor": "asserted per case; no case was skipped for it",
        "mutation_legs": 17,
        "mutation_legs_caught": 15,
        "mutation_legs_null_confirmed": 2,
        "mutation_legs_uncaught_or_unarmed": 0,
        # THE FOLD ARGUMENT, MEASURED RATHER THAN ASSERTED. Letting
        # BC_MIRROR_PERIODIC fall through to the PERIODIC wrap -- so the ghost
        # carries the opposite face instead of the zero the mask makes irrelevant
        # -- changed NO output word on 12 folded-periodic legs. That is
        # "stepping's parity ghost never reaches a surviving word", measured on
        # complex64 storage rather than transferred from the real pair's verdict.
        "fold_ghost_wraps": "NULL CONFIRMED 0/12",
        "drop_the_top_plane_mask": "CAUGHT 12/12",
        "drop_the_fold_cell_zero_mask": "CAUGHT 12/12",
        "top_plane_masks_cell_zero": "CAUGHT 12/12",
        "mask_the_metallic_top_plane": "CAUGHT 4/4",
        "fold_code_read_as_metallic": "CAUGHT 12/12 (the real pair's own 0/64 case)",
        "fold_code_read_as_periodic": "CAUGHT 12/12",
        "constitutive_sides": ("NOT SCORED AND NOT A GAP: all 64 constitutive "
                               "cases deferred to "
                               "coverage.covers_real_pml_complex_constitutive, "
                               "which admits a fold directly since 2026-08-20 "
                               "(COMPLEX_CONSTITUTIVE_FOLD_ADMISSION)"),
    },
    #: SLOTS, recomputed from the census record by
    #: parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_complexfoldbeta/
    #: replay_over_closeout.py and then RE-MEASURED by the union analyzer run
    #: positionally over the patched copy, so before and after come from one
    #: instrument. 10 slots on 5 rows -- step_B and step_D only, because the
    #: certified complex constitutive pair now serves the other two sub-steps.
    "slots": 10,
    "rows": 5,
    "sub_steps": ("step_B", "step_D"),
    "denominator": 759,
    "what_it_does_not_license": (
        "any dispatch: nothing in meep_gpu imports cuda_kernels.",
        "a folded axis carrying a Bloch phase: stepping's PERIODIC and MIRROR "
        "ghost arms are mutually exclusive per axis, so there is no such run; "
        "refused by name in _fold_reasons, not measured.",
        "grid.beta on a folded complex grid: that is complex_beta_kernels', and "
        "this family inherits the certified predicate's beta refusal.",
        "an off-diagonal chi1inv at update_E: refused by the certified complex "
        "constitutive predicate for the row product, fold or no fold.",
        "cylindrical (Dcyl) coordinates: the complex curl refuses them for a real "
        "reason and this family inherits that refusal.",
        "three simultaneous fold planes: no corpus row carries one and the gate "
        "sweeps at most two.",
        "any throughput claim: this is a correctness family and times nothing.",
        "the CONSTITUTIVE sides: the certified complex pair serves them directly "
        "since 2026-08-20 and this family defers to it by name -- and the device "
        "run measured the deferral (all 64 constitutive cases named that "
        "predicate's own admission) rather than scoring a slot it does not own.",
        "a FUSED product on this family's cell: both halves of the B_to_H "
        "folded-complex cell are certified singles as of 2026-09-01, and the "
        "weld that serves the cell's 5 rows is the separate product "
        "complex_folded_fused_magnetic_pair, licensed by its own record "
        "(cuda_complex_folded_fused_magnetic_pair_2026-09-02) and never by "
        "this one.",
    ),
}


# ---------------------------------------------------------------------------
# THE PREDICATES
# ---------------------------------------------------------------------------

class _FoldFreeGrid:
    """The run's grid, answering "no fold" and forwarding everything else.

    Satisfying AT THE INPUT the clauses this family inverts is the technique
    ``special_kz_curl._BetaFreeGrid`` and ``no_pml_curl._ActiveLayerProxy`` use, and
    for the same reason: the shipped predicates SHORT-CIRCUIT, so there is no
    accumulated refusal list to filter afterwards.

    TWO METHODS ARE OVERRIDDEN AND THE CHOICE IS LOAD-BEARING. ``has_symmetry`` and
    ``is_mirrored`` are the two ``coverage._grid_facts`` reads that carry the fold
    (:669, :671), and inverting them makes a folded axis resolve to its DECLARED
    kind -- PERIODIC or METALLIC -- through ``_boundary_kinds_from``. That is what
    lets the base predicate ask every other question. It is ALSO what makes
    :func:`_fold_reasons`' Bloch clause mandatory: a folded PERIODIC axis reads
    back as an ordinary periodic wrap, and the base predicate's "only a periodic
    wrap can carry a phase" clause would then admit a phase the array path does not
    express.

    ``stored_cells`` / ``owned_cells`` are NOT overridden: ``_stored_past_owned_from``
    gates on ``is_mirrored`` and therefore answers all-False through this proxy,
    which is correct for a grid the base predicate is being asked to treat as
    unfolded. The REAL fold termination is read separately, off the REAL grid, by
    :func:`folded_complex_boundary_codes`.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)

    def has_symmetry(self) -> bool:
        return False

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002 - the whole point
        return False

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), item)


def folded_complex_boundary_codes(grid: Any) -> Tuple[Any, Any]:
    """The three integer codes this family's kernels take, or a refusal.

    Returns ``(codes, refusal)`` with exactly one of them None.

    DELEGATED TO ``coverage.real_curl_boundary_codes`` RATHER THAN SPELLED AGAIN,
    and that is not tidiness: that function carries
    ``coverage._fold_termination_problem``, the CROSS-CHECK between
    ``Grid.stored_cells > Grid.owned_cells`` and ``Grid.is_metallic`` which decides
    whether a folded axis gets ``BC_METALLIC`` or ``BC_MIRROR_PERIODIC``. Getting
    that wrong is a top-plane mask applied where MEEP steps the plane, or withheld
    where it does not -- a wrong answer on one plane of one component, not a crash
    -- and a second spelling of it is one too many. It also refuses the cylindrical
    axis, which has no code in either family.
    """
    return _coverage.real_curl_boundary_codes(grid)


def _fold_reasons(fields: Any, grid: Any) -> Any:
    """The clauses THIS family owns, in place of the ones it inverts. None if clear.

    Four, each of which is a wrong answer rather than a crash if it is dropped:

    1. **A FOLD MUST BE ACTIVE.** An unfolded complex run belongs to the certified
       ``covers_real_pml_complex_curl``; admitting one here would put two families
       on one slot, which the union census reports as a FINDING rather than as
       extra coverage.
    2. **THE FOLD'S TERMINATION MUST BE DECIDABLE AND SELF-CONSISTENT**, through
       :func:`folded_complex_boundary_codes`. A grid that cannot say whether it
       stores the slot past MEEP's owned window cannot be given a top-plane mask
       decision, and two routes to that fact that DISAGREE mean neither may be
       trusted.
    3. **NO BLOCH PHASE ON A FOLDED AXIS.** ``stepping._shift_up`` /
       ``_shift_down`` apply the wrap factor on the PERIODIC arm and the parity on
       the MIRROR arm, exclusively; there is no run in which both happen. Read as
       a nonzero k COMPONENT and, separately, as a non-None ``bloch_phase`` --
       both, because they are two routes to the same fact and a grid whose two
       answers differ is a grid neither may be trusted on.
    4. **THE PHASE READER MUST BE READABLE.** An unreadable phase table is not an
       unphased one; admitting one would let the launcher's
       ``bloch_phase_arguments`` raise where a fail-closed "no" was right.
    """
    try:
        has_symmetry = bool(grid.has_symmetry())
        mirrored = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return (f"the grid could not be asked whether it is folded: "
                f"{type(exc).__name__}: {exc}")
    if not (has_symmetry or any(mirrored)):
        return ("no mirror fold is active: an unfolded complex run belongs to "
                "coverage.covers_real_pml_complex_curl, and two families on one "
                "slot is a widening rather than coverage")
    if has_symmetry and not any(mirrored):
        return ("the grid reports a symmetry but no mirrored axis; the two routes "
                "to the fold disagree and neither can be trusted")
    codes, refusal = folded_complex_boundary_codes(grid)
    if refusal is not None:
        return refusal
    if BC_MIRROR_PERIODIC_CODE not in codes and not any(mirrored):
        # Unreachable given the clause above; kept because the code triple is what
        # the KERNEL sees and a disagreement between it and ``mirrored`` would mean
        # the launcher and the predicate are reading different grids.
        return ("the resolved boundary codes carry no fold while the grid reports "
                "one; the launcher and the predicate disagree about this grid")
    reader = getattr(grid, "bloch_phase", None)
    if not callable(reader):
        return ("grid.bloch_phase is missing or not callable; an unreadable phase "
                "table is not an unphased one")
    try:
        k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    except Exception as exc:  # noqa: BLE001
        return f"grid.k_point could not be read: {type(exc).__name__}: {exc}"
    if len(k_point) != 3:
        return f"grid.k_point has {len(k_point)} components, not three"
    for axis in range(3):
        if not mirrored[axis]:
            continue
        try:
            phase = reader(axis)
        except Exception as exc:  # noqa: BLE001
            return (f"grid.bloch_phase({axis}) raised {type(exc).__name__}: {exc}; "
                    f"an unreadable phase is not an unphased one")
        if phase is not None or float(k_point[axis]) != 0.0:
            return (f"axis {axis} is folded AND carries k component "
                    f"{k_point[axis]!r} (phase {phase!r}): stepping._shift_up and "
                    f"_shift_down apply the wrap factor on the PERIODIC arm and "
                    f"the parity on the MIRROR arm exclusively (stepping.py:"
                    f"1812-1830, :1862-1873), so this is a configuration the "
                    f"array path does not express")
    return None


def _already_served(admits: bool, family: str) -> Any:
    """The refusal a family owes when a SHIPPED sibling already serves the slot.

    THE PARTITION IS ENFORCED BY CONSTRUCTION HERE, not by a reading of what the
    siblings happen to refuse today. Measured 2026-08-20, mid-session: the certified
    complex constitutive pair gained ``admit_fold=True`` -- a device verdict of its
    own (:data:`coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION`) -- while this family
    was being written, and every folded ``update_H`` slot became a slot two
    predicates admitted. The union census reports that as a FINDING rather than as
    coverage, and a clause that asks the sibling is the only version of this rule
    that survives the sibling changing.
    """
    if not admits:
        return None
    return (f"{family} already admits this configuration directly; two families on "
            f"one slot is a widening the union census reports as a finding, not "
            f"extra coverage")


def covers_complex_folded_curl(fields: Any, pml: Any, grid: Any, sub_step: str,
                               license: Any = None,
                               subnormal_policy: Any = None) -> tuple:
    """Whether ``step_{B,D}_pml_complex_folded`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. ``license`` is the verdict
    ``triton_kernels.complex_fields.expansion_license`` returns and
    ``subnormal_policy`` the policy name this run installs; both are checked by the
    certified predicate this one delegates to, at the seam where the arm is
    compiled into the binary. Returns ``(covered, reason)``.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except the fold is the
    CERTIFIED complex curl predicate's, asked through it on a proxy grid that
    reports no fold, so this family cannot drift to a weaker standard than the one
    it borrows its arithmetic from -- and a clause added there (the conductivity
    split, the halved int32 bound, the volume and coefficient-vector tail, the
    expansion licence) is inherited here rather than silently skipped.

    THE SUB-STEP ARGUMENT IS REQUIRED for the reason the certified predicate's is:
    ``stepping._apply_curl`` reads ``fields.condfac_for(term.target)`` PER TERM
    (stepping.py:508), so a D-side conductivity routes ``step_D`` to the
    three-history conductive-PML recurrence and leaves ``step_B`` an ordinary curl.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    own = _fold_reasons(fields, grid)
    if own is not None:
        return False, own
    served, _ = _coverage.covers_real_pml_complex_curl(
        fields, pml, grid, sub_step, license=license,
        subnormal_policy=subnormal_policy)
    taken = _already_served(served, "coverage.covers_real_pml_complex_curl")
    if taken is not None:
        return False, taken
    return _coverage.covers_real_pml_complex_curl(
        fields, pml, _FoldFreeGrid(grid), sub_step,
        license=license, subnormal_policy=subnormal_policy)


def covers_complex_folded_constitutive(fields: Any, pml: Any, grid: Any, side: str,
                                       license: Any = None,
                                       subnormal_policy: Any = None) -> tuple:
    """Whether the CERTIFIED complex constitutive pair may serve a FOLDED run.

    ``side`` is ``"H"`` or ``"E"``; returns ``(covered, reason)``.

    THIS FAMILY BUILDS NO CONSTITUTIVE KERNEL -- see the module docstring for why,
    and for why that reading is gated on a device rather than trusted. The two
    sides are asked SEPARATELY because the E side refuses things the H side does
    not (an off-diagonal ``chi1inv``, a registered polarization), which is the same
    per-sub-step split the real track makes.

    AND ON 2026-08-20 IT BECAME WORTH NOTHING, which is the honest state rather than
    a reason to keep the arm. ``coverage.covers_real_pml_complex_constitutive``
    gained ``admit_fold=True`` on its own device verdict
    (:data:`coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION`: 180/180 per policy on 156
    folded cases, three simultaneous planes, the refusal's own premise armed and
    caught), so the certified pair now serves every folded constitutive slot
    directly and this function refuses them by name through :func:`_already_served`.
    It is kept rather than deleted because the deferral is the PARTITION RULE: if
    the certified predicate ever narrows again, this family takes the slots back
    without an edit, and until then the census attributes them to the family whose
    gate measured them.
    """
    if side not in ("H", "E"):
        raise ValueError(f"side must be 'H' or 'E', got {side!r}")
    own = _fold_reasons(fields, grid)
    if own is not None:
        return False, own
    served, _ = _coverage.covers_real_pml_complex_constitutive(
        fields, pml, grid, side, license=license,
        subnormal_policy=subnormal_policy)
    taken = _already_served(
        served, "coverage.covers_real_pml_complex_constitutive")
    if taken is not None:
        return False, taken
    return _coverage.covers_real_pml_complex_constitutive(
        fields, pml, _FoldFreeGrid(grid), side,
        license=license, subnormal_policy=subnormal_policy)


# ---------------------------------------------------------------------------
# COMPILATION AND LAUNCH
# ---------------------------------------------------------------------------
#
# ``cupy`` is imported INSIDE these functions, never at module scope, so the
# predicates and the emitter above stay callable on the census laptop.

#: ``--fmad=false`` is CORRECTNESS on this pair and not tuning, for exactly the
#: reasons ``complex_emitter``'s note 5 gives: the PML recurrence has contraction
#: candidates at ``(fu*kms) - curl`` and ``(f*kms_u) + fu_new``, and the arm's own
#: fusions are spelled with ``__fmaf_rn``, which is a single ``fma.rn.f32``
#: whatever the flag says. The fold adds no new contraction candidate -- it adds
#: predicated zero stores -- but the flag is the sibling's and a family that
#: compiled the same bytes under different options would not be comparable to it.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: One COMPLEX CELL per lane, the certified complex family's block size.
_COMPLEX_THREADS = 256


def _get_kernel(sub_step: str, expansion):
    """Compile one sub-step under one arm, memoized on (name, options, policy, source).

    The source is read through :func:`kernel_source` PER CALL, for the reason every
    sibling rebuilds its map per call: the memo key carries the source string, so a
    body rewritten through :func:`set_kernel_source` is a MISS and reaches NVRTC. A
    memo above that seam would hand back the pre-mutation string forever.
    """
    import cupy as cp  # noqa: PLC0415

    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    arm = complex_emitter.normalized_expansion(expansion)
    name = FOLDED_KERNELS[sub_step]
    code = kernel_source(sub_step, arm)
    key = kernel_cache_key(f"{name}_arm{arm}", True, _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache  # noqa: PLC0415

    return clear_kernel_cache()


def step_folded_complex(sub_step: str, fields: Any, expansion,
                        grid: Any = None, pml: Any = None, *,
                        dtdx: float = None,
                        tables: Dict[str, Any] = None,
                        boundary_codes: Sequence[Any] = None,
                        phase_flags: Sequence[Any] = None,
                        phase_values: Sequence[Any] = None) -> None:
    """One folded-complex curl sub-step, in place, in one launch.

    The argument list is the certified ``complex_pml_kernels.step_fused_pml_complex``
    one, unchanged, so a reader can diff the two launches; the ONLY difference is
    the boundary-code source, which is :func:`folded_complex_boundary_codes` rather
    than ``complex_pml_kernels.complex_boundary_codes`` -- that function refuses a
    fold by name, which is correct for a kernel with no fold branch and wrong for
    this one.

    SUPPLY ``grid`` AND ``pml`` and everything derivable is derived HERE, by the
    same functions the predicate asks; the keyword overrides are the GATE'S DOOR
    (a deliberately swapped sub-lattice, a dropped fold code, an unconjugated
    backward phase) and passing a layer AND its override is refused.
    """
    import numpy as np  # noqa: PLC0415

    from .complex_pml_kernels import (bloch_phase_arguments,  # noqa: PLC0415
                                      complex_curl_tables, word_view)

    if sub_step not in FOLDED_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(FOLDED_KERNELS)}, got {sub_step!r}")
    backward = complex_emitter.KERNELS[sub_step][1]
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the sub-step's "
            "own sub-lattice) or tables (the gate supplies its own)")
    if tables is None:
        tables = complex_curl_tables(pml, complex_emitter.HALF_INTEGER[sub_step])
    if (grid is None) == (boundary_codes is None):
        raise ValueError(
            "pass exactly one of grid (the fold-aware boundary codes and the phase "
            "table are resolved from it) or boundary_codes (the gate's own)")
    if grid is not None:
        codes, refusal = folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(
                f"this grid has no folded-complex boundary codes: {refusal}")
        boundary_codes = codes
        derived_flags, derived_values = bloch_phase_arguments(grid, backward)
        if phase_flags is None:
            phase_flags = derived_flags
        if phase_values is None:
            phase_values = derived_values
        if dtdx is None:
            dtdx = grid.dt / grid.dx  # stepping.py:314, :431.
    if phase_flags is None or phase_values is None or dtdx is None:
        raise ValueError(
            "without a grid the caller must supply phase_flags, phase_values and "
            "dtdx; there is nothing here to derive them from")

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
    shape = getattr(fields, targets[0]).shape
    arguments = [word_view(getattr(fields, name)) for name in targets]
    arguments += [word_view(getattr(fields, "fu_" + name)) for name in targets]
    arguments += [word_view(getattr(fields, name)) for name in sources]
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                  np.float32(dtdx)]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [np.int32(code) for code in boundary_codes]
    arguments += [np.int32(flag) for flag in phase_flags]
    arguments += [np.float32(value) for value in phase_values]
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    blocks = (cells + _COMPLEX_THREADS - 1) // _COMPLEX_THREADS
    _get_kernel(sub_step, expansion)(
        (blocks,), (_COMPLEX_THREADS,), tuple(arguments))
