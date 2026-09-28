"""Raw-CUDA COMPLEX-storage PML sub-steps -- the CuPy half of the complex family.

The device source is :mod:`complex_emitter`, which imports nothing and runs at the
merge bar; this module is the part that needs a GPU: the compile memo, the
coefficient and phase bindings, the word views and the four launchers.

WHAT THIS CLOSES. Before it, every hand-CUDA predicate refused complex64 storage
by name -- ``covers_real_pml_curl`` and ``covers_real_pml_constitutive`` both say
"complex64 storage: the recurrence is the same but the storage is not". Measured
over the 186-row corpus record on 2026-08-16, that clause was the FIRST refusal on
47 rows at ``update_H`` and 47 at ``update_E``; the curl predicate refused the same
storage class. It was a correct refusal about the REAL kernels and an absence of a
complex one, exactly as ``update_P`` was an absence rather than a refusal until
``ade_kernels`` closed it.

WHICH RUNS DEMAND IT. Nine corpus scripts and only nine, per the Triton sibling's
census: five Bloch-phase runs (refl-angular.py, antenna_pec_ground_plane_1D.py,
binary_grating_oblique.py, mode_coeff_phase.py, oblique-planewave.py) and two
complex-storage-at-k=0 runs (wvg-src.py, solve-cw.py's complex-storage class).
``special_kz`` scripts (refl-angular-kz2d.py, parallel-wvgs-force.py) are
``grid.beta``, NOT ``bloch_phase``, and are refused by name.

=============================================================================
THE ARM IS AN ARGUMENT, AND THAT IS THE PACKAGE BOUNDARY
=============================================================================

Every launcher here takes ``expansion``. It has no default, this module derives it
from nothing, and it never opens a probe artifact: the arbiter is
``triton_kernels.complex_fields.expansion_license`` and re-implementing it here
would be a second spelling of the one rule that stands between a guessed constexpr
and a kernel. The GATE reads the artifact, runs the licence, and hands the arm in;
``coverage.covers_real_pml_complex_curl`` takes the VERDICT and checks its shape.

A wrong arm is a WRONG ANSWER rather than a crash -- both arms compile, both run,
and they differ in the last bits of about a quarter of the words -- which is why
:func:`complex_emitter.normalized_expansion` refuses rather than defaulting.

=============================================================================
THE WORD VIEW
=============================================================================

``complex64`` is bit-layout (re, im) interleaved, so a C-contiguous complex volume
viewed as float32 is the SAME allocation with a doubled last axis: no copy, no
move, and the engine's own references stay live. :func:`word_view` is that view and
it REFUSES anything else -- a strided view would be read in the wrong order by the
flat word index, which is a scrambled volume rather than a launch failure.

The int32 bound is HALVED against the real path's for the same reason: the kernels
address ``2*idx`` in ``int``, so ``2*ncells`` is what must stay below ``2**31``.
``coverage`` carries that clause.

=============================================================================
NOT DISPATCHED
=============================================================================

Like every other kernel in this package: no module in ``meep_gpu/`` imports
``cuda_kernels`` at all, ``fastpath.py`` is the *Triton* dispatcher and never names
it, and ``test_package_boundary.py`` pins that absence in both directions. The only
importers are this directory's tests and the parity gates.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import cupy as cp
import numpy as np

# The predicate and the emitter live in CUPY-FREE siblings so they stay
# importable, exercisable and mutable on a machine with no GPU. Imported back here
# because this module is the slice's public face; the by-path fallback is for the
# bit-identity probe, which loads these modules outside the package.
try:
    from . import complex_emitter
    from .coverage import (BC_CODES, PERIODIC, real_pml_boundary_kinds)
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    def _by_path(module_name: str, file_name: str):
        spec = _importlib_util.spec_from_file_location(
            module_name,
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), file_name))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    complex_emitter = _by_path("cuda_kernels_complex_emitter", "complex_emitter.py")
    _coverage = _by_path("cuda_kernels_coverage", "coverage.py")
    BC_CODES = _coverage.BC_CODES
    PERIODIC = _coverage.PERIODIC
    real_pml_boundary_kinds = _coverage.real_pml_boundary_kinds

try:
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _cache_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_compile_cache",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "compile_cache.py"))
    compile_cache = _importlib_util.module_from_spec(_cache_spec)
    _cache_spec.loader.exec_module(compile_cache)

#: Byte-identical to the array path with a gate verdict behind it AND a
#: ``certification.json`` record entry behind the verdict. A name here without a
#: record block is the failure ``test_complex_pml.py`` exists to catch, and a name
#: in the record without one here is the same failure from the other side.
#: RECORDED 2026-08-26. These four sat in ``UNCERTIFIED_KERNELS`` from 2026-08-19 with
#: a passing gate behind them, deliberately, because "certified" had to be a verdict a
#: reader could check. The block is now in ``certification.json`` as
#: ``complex_pml_2026-08-26`` and the two halves landed together, as
#: ``test_every_uncertified_name_has_no_record_block_and_the_reverse`` requires.
#:
#: THE 08-19 VERDICT WAS NOT TRANSCRIBED, IT WAS RE-EARNED. That artifact's recorded
#: digests had drifted on THIS MODULE and on ``complex_emitter.py`` -- the subject and
#: the emitter of its device text -- so it no longer described the bytes that ship. The
#: gate was re-run on the current tree and reproduced it exactly: 1008/1008 single-launch
#: under BOTH float32 subnormal policies, 36/36 at the 60-launch multi-step budget,
#: 20/20 mutation legs, expansion arm ``FMA_V1`` with its basis MEASURED.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "step_B_pml_complex_bloch",
    "step_D_pml_complex_bloch",
    "update_H_pml_complex_bloch",
    "update_E_pml_complex_bloch",
)

#: Shipped but not RECORDED. The partition test requires every shipped kernel to be
#: in exactly one of the two sets, so a kernel added to this family cannot ship
#: unplaced. Spelled as a dict without a type annotation, for the reason the sibling
#: modules spell theirs that way: the partition test reads it off the syntax tree
#: without importing the module, which is the only way to read it on a host with no
#: CuPy, and an annotated assignment is an ``ast.AnnAssign`` that reader does not
#: match.
#:
#: ALL FOUR HAVE PASSED THEIR GATE and this set is where they sit anyway, which is
#: a deliberate state and not an oversight. The verdict is
#: ``parity/meep_gpu/results/cuda_complex_release_2026-08-19/`` -- RELEASED under
#: BOTH float32 subnormal policies on an RTX A6000: 1008/1008 single-launch cases
#: bit-identical to ``stepping``'s four sub-steps per policy, 36/36 at the
#: 60-launch multi-step budget, 20/20 mutation legs, arm ``FMA_V1`` measured. What
#: is missing is the RECORD: ``certification.json`` is the sibling of
#: ``triton_kernels/fingerprints.json`` and the thing a reader is entitled to check
#: a claim against, and "certified" must be a verdict someone can read rather than
#: a word someone typed.
#:
#: THE TWO EDITS LAND TOGETHER, welded by
#: ``test_every_uncertified_name_has_no_record_block_and_the_reverse``: moving a
#: name here into ``CERTIFIED_KERNELS`` without adding its block fails, and adding
#: the block without moving the name fails too. Neither half is valid alone.
UNCERTIFIED_KERNELS = {}

#: NVRTC compile options -- CORRECTNESS, not performance, and identical to the
#: three certified siblings'. Spelled here rather than imported so that loading
#: this file by path (which the probe does) cannot pick up a different tuple than
#: the one the gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane. 256 is what all three certified
#: siblings and ``triton_kernels.kernels.DEFAULT_BLOCK`` landed on independently.
_COMPLEX_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with the sibling modules -- the cache is keyed on the source
    string, so two modules cannot collide, and neither can two arms.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(sub_step: str, expansion):
    """Compile one sub-step under one arm, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason the
    sibling modules rebuild their code maps per call: the source is part of the memo
    key, so it has to be built before a key exists to miss on, and a gate mutates
    this family by monkeypatching :func:`complex_emitter.complex_source`. A source
    memoized at first call would hand back the pre-mutation string forever -- a leg
    reporting a pass for a mutation it never applied.

    THE ARM IS IN THE KEY THROUGH THE SOURCE, which is what stops the two arms
    being served each other's binary: they are different strings, so they are
    different keys, exactly as the off-diagonal family's row-mask specializations
    are.
    """
    arm = complex_emitter.normalized_expansion(expansion)
    name = complex_emitter.KERNELS[sub_step][0]
    code = complex_emitter.complex_source(sub_step, arm)
    key = compile_cache.kernel_cache_key(
        f"{sub_step}_arm{arm}", True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# ---------------------------------------------------------------------------
# The bindings
# ---------------------------------------------------------------------------

def word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume -- the pointer the kernel gets.

    Refuses anything that is not a C-contiguous complex64 volume. A strided view
    would be read in the wrong order by the flat word index: a scrambled volume,
    not a launch failure, which is why this raises rather than reshaping.
    """
    if array is None:
        raise ValueError("expected a complex64 volume, got None")
    if array.dtype != cp.complex64:
        raise ValueError(f"expected a complex64 volume, got dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            "complex volume is not C-contiguous; its float32 word view would not "
            "be either, and the kernel indexes it as a flat word array")
    return array.view(cp.float32)


def complex_curl_tables(pml: 'PML', half_integer: bool) -> Dict[str, Any]:
    """One curl sub-step's six kms/sinv coefficient vectors, as cached device views.

    Called ONCE per frozen configuration, not per launch: the twelve uncertified
    complex wrappers in ``step_curl_kernels`` call ``cp.ascontiguousarray(...ravel())``
    on every launch, which is six device allocations per sub-step for tables that
    never change (port-reference defect 5.14).

    THE COEFFICIENTS STAY FLOAT32 UNDER COMPLEX STORAGE. That is not an assumption:
    ``PML._reshape_for_broadcast`` stores float32 in BOTH storage modes
    (stepping.py:41-50, fields.py:571-573 against fields.py:1203-1204), and the
    zero-imaginary complex product below is what carries them into complex
    arithmetic. A complex coefficient table here would be a different sub-step.

    ``half_integer`` selects the Yee sub-lattice: the B curl reads the HALF-INTEGER
    positions and the D curl the INTEGER ones (``stepping._curl_coefficients``,
    :2418). Backwards, it is a half-cell error in the absorber profile -- converged,
    smooth and wrong -- which is why the gate carries it as a host mutation.
    """
    return _tables(pml, half_integer, ("kms", "sinv"))


def complex_constitutive_tables(pml: 'PML', half_integer: bool) -> Dict[str, Any]:
    """One constitutive side's six kps/kms vectors, as cached device views.

    ``half_integer`` selects the sub-lattice: ``update_H`` reads the INTEGER
    positions (stepping.py:948) and ``update_E`` the half-integer ones (:1015).
    """
    return _tables(pml, half_integer, ("kps", "kms"))


def _tables(pml: 'PML', half_integer: bool,
            stems: Sequence[str]) -> Dict[str, Any]:
    """The flatten-and-check both table builders share."""
    suffix = "_h" if half_integer else ""
    tables: Dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        for stem in stems:
            attribute = f"{stem}_{axis}{suffix}"
            vector = getattr(pml, attribute, None)
            if vector is None:
                raise ValueError(f"pml.{attribute} is missing")
            flat = vector.reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the complex PML kernels index "
                    f"float32 coefficient vectors and carry them into complex "
                    f"arithmetic through the zero-imaginary product")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector")
            tables[f"{stem}_{axis}"] = flat
    return tables


def complex_boundary_codes(grid: 'Grid') -> Tuple[Any, ...]:
    """Map ``stepping._boundary_kinds``' spellings onto the kernels' integer codes."""
    codes = []
    for axis, kind in enumerate(real_pml_boundary_kinds(grid)):
        if kind not in BC_CODES:
            raise ValueError(
                f"axis {axis} resolved to boundary {kind!r}, which the complex PML "
                f"kernels do not serve (only {sorted(BC_CODES)}).")
        codes.append(np.int32(BC_CODES[kind]))
    return tuple(codes)


def bloch_phase_arguments(grid: 'Grid', backward: bool
                          ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Encode the per-axis Bloch table as (three flags, six float32 components).

    Transcribed from ``stepping._bloch_phases`` (:2313-2367) and
    ``stepping._apply_bloch_phase`` (:1846-1862):

    * the value is ``grid.bloch_phase(axis)`` = exp(2*pi*i*k*L), with the Brillouin
      edge EXACTLY -1+0j (grid.py:1011-1017);
    * ``None`` means the multiply is SKIPPED, never done against 1+0j -- that skip
      is the bit-identity of k = 0, so an unphased axis passes flag 0 and the
      kernel emits no multiply for it at all;
    * the phase is rounded to complex64 BEFORE splitting, because the array path
      multiplies by ``shifted.dtype.type(phase)`` (:1862);
    * for the BACKWARD sub-step the imaginary part is NEGATED -- the conjugate
      ``_shift_down`` takes (:1818-1822). Negation is exact, so the order of
      conjugation and rounding is immaterial.

    A PHASED AXIS MUST RESOLVE PERIODIC. ``stepping._bloch_phases`` raises on the
    pairing and the predicate refuses it; this raises too, so a caller who skips
    the predicate is refused loudly rather than handed a mis-bound flag. A PML on a
    phased axis is ADMITTED -- measured 2.82e-07 against CPU MEEP, which is what
    settled it.
    """
    kinds = real_pml_boundary_kinds(grid)
    flags: list = []
    values: list = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            flags.append(np.int32(0))
            values.extend((np.float32(1.0), np.float32(0.0)))
            continue
        if kinds[axis] != PERIODIC:
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        rounded = np.complex64(phase)
        imag = np.float32(rounded.imag)
        flags.append(np.int32(1))
        values.extend((np.float32(rounded.real),
                       np.float32(-imag) if backward else imag))
    return tuple(flags), tuple(values)


# ---------------------------------------------------------------------------
# The launchers
# ---------------------------------------------------------------------------

#: Which volumes each curl sub-step writes and reads, in the kernel's argument
#: order. ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (stepping.py:214-224) give
#: the target order and the sources; the auxiliaries are ``fu_`` + target.
_CURL_ARRAYS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "step_B": (("Bx", "By", "Bz"), ("Ex", "Ey", "Ez")),
    "step_D": (("Dx", "Dy", "Dz"), ("Hx", "Hy", "Hz")),
}

#: Which volumes each constitutive side writes and reads
#: (``H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``, stepping.py:227-228).
_CONSTITUTIVE_ARRAYS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "H": (("Hx", "Hy", "Hz"), ("Bx", "By", "Bz")),
    "E": (("Ex", "Ey", "Ez"), ("Dx", "Dy", "Dz")),
}


def _blocks(shape: Sequence[int]) -> int:
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    return (cells + _COMPLEX_THREADS - 1) // _COMPLEX_THREADS


def step_fused_pml_complex(sub_step: str, fields: 'Fields', expansion,
                           grid: 'Grid' = None, pml: 'PML' = None, *,
                           dtdx: float = None,
                           tables: Dict[str, Any] = None,
                           boundary_codes: Sequence[Any] = None,
                           phase_flags: Sequence[Any] = None,
                           phase_values: Sequence[Any] = None):
    """``stepping.step_B`` (sub_step='step_B') or ``step_D`` for complex storage.

    SUPPLY ``grid`` AND ``pml`` and everything derivable is derived HERE, by the
    same functions the predicate asks -- which is what makes "the launcher and the
    predicate cannot disagree about which sub-lattice, which boundary code or which
    phase this sub-step reads" an enforced property rather than a convention. The
    certified sibling ``update_fused_pml_real`` learned that the hard way: its
    ``tables`` used to be a positional parameter passed straight through, the
    deriving function had no callers at all, and the half-cell error the pairing
    exists to prevent was one argument away on the only live entry point.

    THE OVERRIDES ARE THE GATE'S DOOR and stay, keyword-only and named for what they
    are. A gate feeds DELIBERATELY WRONG tables (the swapped sub-lattice), wrong
    boundary codes (the dropped metallic wall) and wrong phases (the unconjugated
    backward factor, the whole-array rotation); a launcher that could not be handed
    its own could not arm any of those mutations. Passing a layer AND an override
    for the same quantity is refused: two answers to a question with one.
    """
    if sub_step not in _CURL_ARRAYS:
        raise ValueError(
            f"sub_step must be one of {sorted(_CURL_ARRAYS)}, got {sub_step!r}")
    backward = complex_emitter.KERNELS[sub_step][1]
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the sub-step's "
            "own sub-lattice) or tables (the gate supplies its own, including "
            "mis-paired ones)")
    if tables is None:
        tables = complex_curl_tables(
            pml, complex_emitter.HALF_INTEGER[sub_step])
    if (grid is None) == (boundary_codes is None):
        raise ValueError(
            "pass exactly one of grid (the boundary codes and the phase table are "
            "resolved from it) or boundary_codes (the gate supplies its own)")
    if grid is not None:
        boundary_codes = complex_boundary_codes(grid)
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

    targets, sources = _CURL_ARRAYS[sub_step]
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
    kernel = _get_kernel(sub_step, expansion)
    kernel((_blocks(shape),), (_COMPLEX_THREADS,), tuple(arguments))


def update_fused_pml_complex(side: str, fields: 'Fields', expansion,
                             pml: 'PML' = None, *,
                             tables: Dict[str, Any] = None):
    """``stepping.update_H`` (side='H') or ``update_E`` ('E') for complex storage.

    The H source is ``Bx/By/Bz`` directly: ``stepping.update_H`` (:921) passes
    ``getattr(fields, source)`` with ``source`` from ``H_CONSTITUTIVE_TERMS``, and
    mu = 1 is already baked into that choice -- there is no permeability volume to
    multiply by and introducing one here would be a second engine.

    The E source is ``D * inv_eps`` formed in-kernel with D on the LEFT
    (stepping.py:1011-1013), from the PER-COMPONENT inverse permittivity
    ``Fields.inverse_epsilon_for`` (fields.py:1337) -- three pointers, NEVER
    ``fields.inv_eps``, which is the Ez view (fields.py:1259-1260) and is the defect
    the twelve uncertified complex kernels in ``step_curl_kernels`` carry. This is
    valid only where ``displacement_minus_polarization`` returns the D array itself
    (fields.py:1097-1098); the predicate refuses a registered polarization by name
    for exactly that reason.
    """
    if side not in _CONSTITUTIVE_ARRAYS:
        raise ValueError(
            f"side must be one of {sorted(_CONSTITUTIVE_ARRAYS)}, got {side!r}")
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the sub-lattice is decided from the side) or "
            "tables (the gate supplies its own, including mis-paired ones)")
    if tables is None:
        tables = complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER[side])

    targets, sources = _CONSTITUTIVE_ARRAYS[side]
    shape = getattr(fields, targets[0]).shape
    arguments = [word_view(getattr(fields, name)) for name in targets]
    arguments += [word_view(getattr(fields, "f_w_" + name)) for name in targets]
    arguments += [word_view(getattr(fields, name)) for name in sources]
    if side == "E":
        arguments += [fields.inverse_epsilon_for(name) for name in targets]
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kps_{axis}"], tables[f"kms_{axis}"]]
    sub_step = "update_H" if side == "H" else "update_E"
    kernel = _get_kernel(sub_step, expansion)
    kernel((_blocks(shape),), (_COMPLEX_THREADS,), tuple(arguments))
