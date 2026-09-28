"""Sub-step byte-identity gate for the hand-CUDA COMPLEX ``grid.beta`` curl pair.

THE QUESTION, in four parts, each measured separately:

1. **The curl.** Is ``cuda_kernels/complex_beta_kernels.py``'s pair byte-identical,
   as uint32 WORDS, to ``stepping.step_B`` / ``step_D`` on a complex64 grid whose
   ``grid.beta`` is nonzero -- at one launch and at 60, at an exactly representable
   courant and at one that is not, on physical-band and subnormal-band operands,
   FOLDED and UNFOLDED, under both float32 subnormal policies?
2. **The composition.** Twelve of the sixteen census slots this family serves are
   ALSO folded, so the fold and the beta insert are live at once on three quarters
   of them. The sweep therefore carries folded fixtures at both terminations, and
   the ``beta_after_mask`` needle exists because a fold WIDENS the mask that the
   insert has to precede.
3. **The constitutive sides.** ``covers_complex_beta_constitutive`` admits the
   CERTIFIED complex constitutive pair on a beta run, on the reading that
   ``update_H`` / ``update_E`` read nothing beta-dependent. A reading is not
   evidence; the constitutive arm runs them on a beta run whose two curls have
   ALREADY moved the state, per side, scored SEPARATELY.
4. **Is the beta term doing anything at all?** Every curl case runs the array path
   twice from the same frozen state -- once with beta and once through a
   beta-zeroed grid proxy -- and refuses itself if the two agree OR if they differ
   on the wrong components. MEEP's ``cc`` loop runs over ``d_c`` in {X, Y} only
   (step_db.cpp:148-176), so Bz (Dz) must NOT move; a kernel that put a term on the
   z component would pass a gate that only asked "did anything change".

WHY THE COEFFICIENT IS PASSED AND NOT SYNTHESISED
--------------------------------------------------
``coefficient * 1j`` in Python leaves a SIGNED zero in the real word -- ``-0.0``
exactly when the coefficient is negative and the magnetic branch is taken. With the
field's other word exactly zero the cross term carries that sign into an addend, and
``fma(x, y, -0.0f)`` differs from ``fma(x, y, +0.0f)`` wherever ``x*y`` is exactly
zero. The gate's ``synthesise_the_zero_real_word`` mutation is that claim armed.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
----------------------------------------------------------------
Every floor of ``gate_cuda_complex_folded`` (the fixture, the absorber, the mirror
ghost) plus ``beta_term_is_live`` above. The mutation battery includes legs that
MUST BE UNCAUGHT: a battery of only must-be-caught legs scores identically whether
the comparator works or has degenerated into failing everything.

RUNNING IT
----------
Device (the GPU host, ONE verified-empty GPU)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_complex_beta.py --backend cupy \\
        --probe .../expansion_probe_keep.json \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop, no GPU: the HOST backend compiles the SAME emitted characters with the host
C++ compiler and drives them on NumPy. IT CERTIFIES NOTHING ABOUT NVRTC::

    python -u gate_cuda_complex_beta.py --backend host --out /tmp/beta.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the host backend still runs
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE FOLD GATE IS THIS ONE'S HARNESS, imported rather than copied. This family's
# kernels ARE the folded ones plus an insert, so its fixture, its floors, its
# backends and its verdict shape are the same questions about the same emitter --
# and a second copy of a fixture is a second thing to keep in step. What is added
# here is the beta arm: the coefficients, the beta liveness floor, and the
# mutations that only a beta kernel has.
import gate_cuda_complex as grandparent  # noqa: E402
import gate_cuda_complex_folded as fold_gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402
from meep_gpu.cuda_kernels import complex_beta_kernels as family  # noqa: E402
from meep_gpu.cuda_kernels import complex_folded_kernels as folded  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine

SEED = 20260820
MULTI_STEP_BUDGET = fold_gate.MULTI_STEP_BUDGET

BC_PERIODIC = fold_gate.BC_PERIODIC
BC_METALLIC = fold_gate.BC_METALLIC
BC_MIRROR_PERIODIC = fold_gate.BC_MIRROR_PERIODIC

SUB_STEPS = fold_gate.SUB_STEPS
SIDES = fold_gate.SIDES
CURL_ARRAYS = fold_gate.CURL_ARRAYS
CONSTITUTIVE_ARRAYS = fold_gate.CONSTITUTIVE_ARRAYS
COURANTS = fold_gate.COURANTS
INEXACT_COURANT = fold_gate.INEXACT_COURANT
VALUE_CLASSES = fold_gate.VALUE_CLASSES
GUARD_SETS = fold_gate.GUARD_SETS

build = fold_gate.build
seed_state = fold_gate.seed_state
snapshot = fold_gate.snapshot
restore = fold_gate.restore
advance_sources = fold_gate.advance_sources
moved_fraction = fold_gate.moved_fraction
words_differ = fold_gate.words_differ
structure_facts = fold_gate.structure_facts
absorber_is_not_the_identity = fold_gate.absorber_is_not_the_identity
mirror_ghost_is_nonzero = fold_gate.mirror_ghost_is_nonzero
_GridWithCupysName = fold_gate._GridWithCupysName

#: Beta fixtures. ``Grid._resolve_beta`` admits an effective 2-D Cartesian grid and
#: NOTHING else (grid.py:668-697; MEEP fields.cpp:546-547), so every one of these is
#: nz == 1 -- which is not a choice, and is why this family's z ghost rule is
#: exercised at a single cell. The corpus's four complex beta rows are 500x1x1,
#: 420x212x1 and two 135x92x1, all of that same shape class.
BETA_SPECS: Tuple[Dict[str, Any], ...] = (
    # tests_param/TestSpecialKz.test_special_kz: unfolded, an in-plane kx,
    # NEGATIVE beta (the sign that puts the -0.0 in the coefficient's real word on
    # the magnetic side).
    {"label": "beta_plain_kx", "cell": (16.0, 8.0, 0.0), "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "k": (0.31, 0.0, 0.0),
     "beta": -0.39073112848927377},
    # An unfolded METALLIC wall: the cell-0 mask reaches a beta target even with no
    # fold, which is what makes the ``beta_after_mask`` ladder 0 / 32 / 64 rather
    # than a yes/no.
    {"label": "beta_plain_metallic_x", "cell": (16.0, 8.0, 0.0), "axes": "",
     "phase": 1, "boundaries": ("metallic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.3321611318837033},
    # tests_param/TestSpecialKz.test_eigsrc_kz__idx0: Mirror(Y) PERIODIC, k = 0.
    {"label": "beta_fold_Y_periodic", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.2},
    # tests_param/TestEigCoeffs.test_binary_grating_special_kz__idx0/1: Mirror(Y)
    # PERIODIC with an IN-PLANE kx on the PML'd X axis. The one configuration where
    # the fold, the Bloch rotation and the beta insert are all live at once.
    {"label": "beta_fold_Y_periodic_kx", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.27, 0.0, 0.0), "beta": -0.6850526103319672},
    # THE OTHER TERMINATION, which no corpus row carries.
    {"label": "beta_fold_Y_metallic", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "metallic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.2},
    # THE ODD PLANE PARITY, and an odd full count on a fold over the SLOWEST-STRIDE
    # axis: the beta partner for target 1 is the FIRST source's centre, so an index
    # defect confusing a centre with a neighbour moves with the stride.
    {"label": "beta_fold_X_periodic_odd", "cell": (15.0, 8.0, 0.0), "axes": "X",
     "phase": -1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": -0.9120991827764708},
)


def beta_coefficients(sub_step: str, grid):
    return family.beta_curl_coefficients(grid.beta, grid.dt,
                                         family.MAGNETIC[sub_step])


class _BetaZeroGrid:
    """The fixture's grid with beta forced to zero -- the liveness floor's control.

    ``stepping`` reads ``grid.beta`` in exactly two places (:356/:438 and
    ``_special_kz_beta_term`` at :770), so an array-path run through this proxy is
    the SAME sub-step with the term deleted and nothing else changed.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def beta(self) -> float:
        return 0.0

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


def beta_term_is_live(fields, layer, grid, sub_step, frozen, reference):
    """Run the ORACLE again with beta zeroed, and measure what the term moved.

    Two floors in one measurement, and the second is the one that matters:

    * the beta-on and beta-off runs must DIFFER at all -- otherwise this case
      measures the certified complex kernel and says nothing about this family;
    * they must differ on exactly the components the transcription says carry a
      term. Target 2 (Bz / Dz) moving would mean the ORACLE puts a term where this
      kernel does not; targets 0 and 1 not moving would mean the fixture cannot see
      the term at all. Both are refusals, and the pattern is measured against
      ``stepping`` rather than asserted from the reading.
    """
    saved = fields.grid
    try:
        object.__setattr__(fields, "grid", _BetaZeroGrid(saved))
        restore(fields, frozen)
        getattr(stepping, sub_step)(fields, layer)
        without = snapshot(fields)
    finally:
        object.__setattr__(fields, "grid", saved)
    targets = CURL_ARRAYS[sub_step]["targets"]
    aux = CURL_ARRAYS[sub_step]["aux"]
    per_component = {name: words_differ(reference[name], without[name])
                     for name in tuple(targets) + tuple(aux)}
    pattern_ok = True
    for index, target in enumerate(targets):
        moved = (per_component[target] > 0
                 or per_component[aux[index]] > 0)
        if moved != (index in family.BETA_PARTNER):
            pattern_ok = False
    total = sum(per_component.values())
    return {"differing_words_vs_beta_zero": per_component,
            "total": total,
            "component_pattern_matches_stepping": pattern_ok,
            "carries_term": {t: (i in family.BETA_PARTNER)
                             for i, t in enumerate(targets)},
            "meets_floor": total > 0 and pattern_ok}


# ---------------------------------------------------------------------------
# The backends -- the fold gate's, with this family's launcher bound in
# ---------------------------------------------------------------------------

class CupyBackend(fold_gate.CupyBackend):
    """NVRTC + the GPU, launching ``complex_beta_kernels`` instead of the fold's."""

    name = "cupy"

    def __init__(self) -> None:
        super().__init__()
        self._original_curl_options = family._COMPILE_OPTIONS

    def set_guard(self, options: Sequence[str]) -> None:
        family._COMPILE_OPTIONS = tuple(options)
        self.constitutive_module._COMPILE_OPTIONS = tuple(options)
        family._clear_kernel_cache()
        self.constitutive_module._clear_kernel_cache()

    def set_source_transform(self, transform, arm) -> int:
        family.reset_kernel_sources()
        family._clear_kernel_cache()
        self._transform = transform
        if transform is None:
            return 0
        sites = 0
        for sub_step in family.BETA_KERNELS:
            text, count = transform(family.beta_source(sub_step, arm))
            sites = max(sites, count)
            family.set_kernel_source(sub_step, arm, text)
        return sites

    def launch_curl(self, sub_step, fields, layer, grid, arm, *,
                    beta_coefficients_override=None, **overrides) -> None:
        # THE OVERRIDE IS RENAMED AT THIS SEAM ON PURPOSE. The launcher's keyword is
        # ``beta_coefficients`` and the gate's is ``beta_coefficients_override``,
        # because the gate ALSO passes ``None`` to mean "derive them from the grid"
        # -- and the launcher refuses a grid AND an override for the same quantity.
        # One translation, here, rather than two spellings threaded through the case.
        if beta_coefficients_override is not None:
            overrides["beta_coefficients"] = beta_coefficients_override
        # THE LAUNCHER'S EXACTLY-ONE CONTRACT: ``step_complex_beta`` refuses a
        # layer AND a table override in one call ("the tables are derived from
        # the sub-step's own sub-lattice" -- its own docstring), so the
        # swapped-sub-lattice mutation hands it NO layer and the launcher takes
        # the gate's tables whole. ``beta_coefficients`` beside a grid is fine:
        # the launcher derives only what was not supplied. Measured 2026-09-01
        # on the GPU host: the unfixed call died at that host-mutation leg with the
        # launcher's own ValueError after a fully identical sweep.
        if "tables" in overrides:
            layer = None
        family.step_complex_beta(sub_step, fields, arm, grid=grid, pml=layer,
                                 **overrides)
        cp.cuda.runtime.deviceSynchronize()

    def restore(self) -> None:
        family.reset_kernel_sources()
        family._COMPILE_OPTIONS = self._original_curl_options
        self.constitutive_module._COMPILE_OPTIONS = \
            self._original_constitutive_options
        family._clear_kernel_cache()
        self.constitutive_module._clear_kernel_cache()


class HostBackend(fold_gate.HostBackend):
    """The EMITTED beta source, host-compiled and driven on NumPy. Certifies nothing."""

    name = "host"

    def set_source_transform(self, transform, arm) -> int:
        self._transform = transform
        self._cache.clear()
        if transform is None:
            return 0
        sites = 0
        for sub_step in family.BETA_KERNELS:
            _, count = transform(family.beta_source(sub_step, arm))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str, arm):
        import ctypes  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        key = f"{sub_step}|{complex_emitter.normalized_expansion(arm)}"
        if key in self._cache:
            return self._cache[key]
        source = family.beta_source(sub_step, arm)
        if self._transform is not None:
            source, _ = self._transform(source)
        name, parameters = grandparent.parse_signature(source)
        declaration = ", ".join(f"{kind} {identifier}"
                                for kind, identifier in parameters)
        call = ", ".join(identifier for _, identifier in parameters)
        driver = (f'\nextern "C" void launch_{name}(int nblocks, int nthreads, '
                  f'{declaration}) {{\n'
                  f'    blockDim.x = nthreads;\n'
                  f'    for (int b = 0; b < nblocks; ++b) {{\n'
                  f'        blockIdx.x = b;\n'
                  f'        for (int t = 0; t < nthreads; ++t) {{\n'
                  f'            threadIdx.x = t;\n'
                  f'            {name}({call});\n'
                  f'        }}\n    }}\n}}\n')
        stem = os.path.join(
            self._workdir,
            hashlib.sha256((source + driver).encode()).hexdigest()[:16])
        with open(stem + ".cpp", "w", encoding="ascii") as handle:
            handle.write(grandparent._HOST_SHIM + source + driver)
        subprocess.run([self.compiler, *self.options, "-shared", "-fPIC",
                        "-o", stem + ".so", stem + ".cpp"], check=True,
                       capture_output=True)
        library = ctypes.CDLL(stem + ".so")
        function = getattr(library, f"launch_{name}")
        argtypes: List[Any] = [ctypes.c_int, ctypes.c_int]
        for kind, _ in parameters:
            if "*" in kind:
                argtypes.append(ctypes.c_void_p)
            elif kind == "float":
                argtypes.append(ctypes.c_float)
            elif kind == "int":
                argtypes.append(ctypes.c_int)
            else:
                raise ValueError(f"unhandled parameter type {kind!r}")
        function.argtypes = argtypes
        function.restype = None
        self._cache[key] = (function, parameters)
        return self._cache[key]

    def launch_curl(self, sub_step, fields, layer, grid, arm, *,
                    beta_coefficients_override=None, **overrides) -> None:
        import ctypes  # noqa: PLC0415

        function, _ = self._entry(sub_step, arm)
        arguments, cells = self.curl_arguments(sub_step, fields, layer, grid,
                                               **overrides)
        pair = (beta_coefficients_override if beta_coefficients_override is not None
                else beta_coefficients(sub_step, grid))
        (plus_re, plus_im), (minus_re, minus_im) = pair
        arguments += [ctypes.c_float(float(plus_re)), ctypes.c_float(float(plus_im)),
                      ctypes.c_float(float(minus_re)),
                      ctypes.c_float(float(minus_im))]
        function((cells + 255) // 256, 256, *arguments)


def build_backend(name: str):
    if name == "cupy":
        if cp is None:
            raise SystemExit("--backend cupy needs CuPy and a device")
        return CupyBackend()
    if name == "host":
        import shutil  # noqa: PLC0415

        for candidate in ("clang++", "g++", "c++"):
            found = shutil.which(candidate)
            if found:
                return HostBackend(found)
        raise SystemExit("--backend host needs a host C++ compiler on PATH")
    raise SystemExit(f"unknown backend {name!r}")


# ---------------------------------------------------------------------------
# THE FAMILY'S OWN SOURCE MUTATIONS
# ---------------------------------------------------------------------------

def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    out, count = re.subn(pattern, replacement, text)
    return out, count


def m_drop_the_beta_term(text: str) -> Tuple[str, int]:
    """The whole feature deleted: the FOLDED complex kernel, on a beta run."""
    return _sub(r"\n *curl = cf_sub\(curl, mul_imag_coefficient_left\([^\n]*\n",
                "\n", text)


def m_swap_the_beta_signs(text: str) -> Tuple[str, int]:
    """``bp`` and ``bm`` exchanged: the out-of-plane coupling with its sign reversed.

    Both magnitudes stay right and only the sign of the TE/TM coupling moves, which
    is exactly the class of defect that leaves a run looking converged.
    """
    out = text.replace("mul_imag_coefficient_left(bp,", "mul_imag_coefficient_left(bTMP,")
    out = out.replace("mul_imag_coefficient_left(bm,", "mul_imag_coefficient_left(bp,")
    out = out.replace("mul_imag_coefficient_left(bTMP,", "mul_imag_coefficient_left(bm,")
    return out, (out != text) * 2


def m_beta_added_not_subtracted(text: str) -> Tuple[str, int]:
    """``curl + (c*g)`` for ``curl - (c*g)``: stepping.py:811 negates the PRODUCT."""
    return _sub(r"curl = cf_sub\(curl, mul_imag_coefficient_left\(",
                "curl = cf_add(curl, mul_imag_coefficient_left(", text)


def m_beta_after_mask(text: str) -> Tuple[str, int]:
    """The insert moved BELOW the masks -- the composition needle a FOLD makes big.

    The array path adds the increment before ``_mask_non_owned_cells``
    (stepping.py:389/:391 then :397; :472/:474 then :479). A fold widens that mask
    from METALLIC-only at cell 0 to non-PERIODIC at cell 0 PLUS MIRROR_PERIODIC at
    the top plane, and BOTH arms reach a beta target on BOTH sub-steps -- so an
    insert below them leaves a live increment on the mirror plane and on the far
    ghost plane, invisible in the interior. Measured on the host leg at 0 words on
    an all-periodic unfolded grid, 32 with one wall and 64 on a folded PERIODIC
    axis, per sub-step.
    """
    pattern = re.compile(
        r"( *)(curl = cf_sub\(curl, mul_imag_coefficient_left\([^\n]*\n)"
        r"((?: *if \(bc_\w == BC_[A-Z_]+ && [^\n]*\n)+)")
    out, count = pattern.subn(lambda m: m.group(3) + m.group(1) + m.group(2), text)
    return out, count


def m_beta_partner_shifted(text: str) -> Tuple[str, int]:
    """The SHIFTED companion instead of the unshifted centre.

    A MEASURED NULL on the REAL special_kz family (0/12 CAUGHT) and expected to be
    one here for the same STRUCTURAL reason: every beta grid has nz == 1, and both
    partners' only shifted companion is the neighbour along Z, so the two are the
    same word. Kept and scored as a null rather than deleted -- if it ever becomes
    detectable, the structural argument has stopped holding.
    """
    out = text.replace("mul_imag_coefficient_left(bp, f_2)",
                       "mul_imag_coefficient_left(bp, ss)")
    out = out.replace("mul_imag_coefficient_left(bm, f_1)",
                      "mul_imag_coefficient_left(bm, sf)")
    return out, (out != text) * 2


def m_beta_on_the_third_component(text: str) -> Tuple[str, int]:
    """A term added to target 2, which MEEP's ``cc`` loop never reaches.

    ``d_c`` runs over {X, Y} only (step_db.cpp:148-176), so Bz and Dz get nothing.
    A gate that only asked "did the answer change" would not see this at all.
    """
    marker = "    // Target 2:"
    index = text.find(marker)
    if index < 0:
        return text, 0
    tail = text[index:]
    anchor = "        cf curl = mul_coefficient_left(dtdx,"
    position = tail.find(anchor)
    if position < 0:
        return text, 0
    end = tail.find("\n", position) + 1
    insert = "        curl = cf_sub(curl, mul_imag_coefficient_left(bp, f_2));\n"
    return text[:index] + tail[:end] + insert + tail[end:], 1


def m_scale_beta_by_dtdx(text: str) -> Tuple[str, int]:
    """The increment scaled by ``dtdx`` -- an analytic derivative read as a difference.

    ``stepping.py:797`` carries NO ``dtdx``: ``i*2*pi*beta`` is exact, not a finite
    difference (:758-762). This is the single easiest way to get a beta kernel
    plausibly wrong.
    """
    return _sub(r"cf_sub\(curl, mul_imag_coefficient_left\((\w+), (\w+)\)\)",
                r"cf_sub(curl, mul_coefficient_left(dtdx, "
                r"mul_imag_coefficient_left(\1, \2)))", text)


def m_synthesise_the_zero_real_word(text: str) -> Tuple[str, int]:
    """``bp.re``/``bm.re`` written as a literal ``+0.0f`` instead of the passed word.

    THE CLAIM THIS ARMS: Python's complex multiply leaves ``-0.0`` in the real word
    on the negative-coefficient magnetic branch, and with the field's other word
    exactly zero the cross term carries that sign into an ``fma`` addend. If this
    leg comes back UNCAUGHT the pass-through is belt and braces on this platform,
    which is a finding worth recording rather than a failure.
    """
    return _sub(r"cf bp; bp\.re = bp_re; bp\.im = bp_im;\n"
                r"    cf bm; bm\.re = bm_re; bm\.im = bm_im;",
                "cf bp; bp.re = 0.0f; bp.im = bp_im;\n"
                "    cf bm; bm.re = 0.0f; bm.im = bm_im;", text)


def n_beta_subtract_as_add_negative(text: str) -> Tuple[str, int]:
    """``curl + (-(c*g))`` spelled out for ``curl - (c*g)`` -- a NULL by IEEE-754.

    Subtraction IS addition of the negation, and complex negation and complex add
    are both plane-wise, so this MUST be uncaught. Paired with
    ``beta_added_not_subtracted`` on the same expression: one must catch, one must
    not, and a comparator that failed everything would fail this pair.
    """
    out, count = _sub(
        r"cf_sub\(curl, mul_imag_coefficient_left\((\w+), (\w+)\)\)",
        r"cf_add(curl, cf_sub(cf_zero(), mul_imag_coefficient_left(\1, \2)))",
        text)
    return out, count


#: Legs that MUST BE UNCAUGHT. ``beta_partner_shifted`` is here on the REAL
#: family's measured verdict and this gate re-measures it rather than inheriting it.
NULL_MUTATIONS: Tuple[str, ...] = ("beta_subtract_as_add_negative",
                                   "beta_partner_shifted",
                                   "reload_the_target")

#: Legs whose UNCAUGHT is a PLATFORM FINDING rather than a failure, per the leg's
#: own docstring -- ``synthesise_the_zero_real_word`` says in its own words: "If
#: this leg comes back UNCAUGHT the pass-through is belt and braces on this
#: platform, which is a finding worth recording rather than a failure." The 2026-09-01
#: device run recorded exactly that, 0/12 on both policies with every OTHER
#: must-catch leg caught: the emitted arithmetic reaches the beta coefficient
#: through ``mul_imag_coefficient_left``, which reads the imaginary word alone, so
#: the real word this mutation zeroes is defensive pass-through and no fixture in
#: this sweep can make it observable. Scored ``REPORTED`` when uncaught on EVERY
#: leg -- never on a PARTIAL, which stays blocking, because a leg caught on some
#: fixtures and not others is a visibility hole a human has to look at.
FINDING_MUTATIONS: Tuple[str, ...] = ("synthesise_the_zero_real_word",)

SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "drop_the_beta_term": m_drop_the_beta_term,
    "swap_the_beta_signs": m_swap_the_beta_signs,
    "beta_added_not_subtracted": m_beta_added_not_subtracted,
    "beta_after_mask": m_beta_after_mask,
    "beta_on_the_third_component": m_beta_on_the_third_component,
    "scale_beta_by_dtdx": m_scale_beta_by_dtdx,
    "synthesise_the_zero_real_word": m_synthesise_the_zero_real_word,
    "beta_partner_shifted": m_beta_partner_shifted,
    "beta_subtract_as_add_negative": n_beta_subtract_as_add_negative,
    # THE FOLD'S OWN DEFECT CLASSES, re-armed because these kernels carry the fold
    # branches: a beta delta that broke one of them would otherwise be invisible.
    "drop_the_top_plane_mask": fold_gate.m_drop_the_top_plane_mask,
    "drop_the_fold_cell_zero_mask": fold_gate.m_drop_the_fold_cell_zero_mask,
    "reload_the_target": fold_gate.n_reload_the_target,
    # AND THE CERTIFIED COMPLEX FAMILY'S, for the same reason one rung further down.
    "fold_the_zero_cross_terms": grandparent.m_fold_the_zero_cross_terms,
    "left_to_right_curl": grandparent.m_left_to_right_curl,
    "conjugate_the_rotation": grandparent.m_conjugate_the_rotation,
    "column_major_index": grandparent.m_column_major_index,
}

#: Legs whose defect needs a particular boundary code to be able to bite at all.
MUTATION_REQUIRES_CODE: Dict[str, int] = {
    "drop_the_top_plane_mask": BC_MIRROR_PERIODIC,
    "drop_the_fold_cell_zero_mask": BC_MIRROR_PERIODIC,
}

#: ``beta_after_mask`` needs a MASK on a beta target; an all-periodic unfolded grid
#: has none, and a leg run there scores UNCAUGHT for a reason that is about the
#: fixture rather than the kernel.
MUTATION_REQUIRES_A_WALL: Tuple[str, ...] = ("beta_after_mask",)

#: Legs whose defect needs a PHASED axis: the kernel emits no rotation at all when
#: every phase flag is 0, which is the bit-identity of k = 0, so a rotation mutation
#: on an unphased grid is a no-op. Scoping it is what stops a PARTIAL that is about
#: the fixture from reading as one about the kernel.
MUTATION_REQUIRES_A_PHASE: Tuple[str, ...] = ("conjugate_the_rotation",)

#: The CERTIFIED complex family's signed-zero defect classes, re-armed as a
#: REGRESSION check that the beta delta did not disable one. Their full scoring is
#: ``gate_cuda_complex``'s; see ``gate_cuda_complex_folded.INHERITED_MUTATIONS``.
INHERITED_MUTATIONS: Tuple[str, ...] = ("fold_the_zero_cross_terms",)

#: HOST-side mutations: the inputs corrupted rather than the device text.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_the_yee_sub_lattice",
    "beta_coefficients_exchanged",
    "beta_coefficients_zeroed",
    "beta_coefficient_conjugated",
)
HOST_NULL_MUTATIONS: Tuple[str, ...] = ()


def host_mutated_inputs(name: Optional[str], sub_step, layer, grid):
    """The keyword overrides one HOST mutation installs; ``{}`` when there is none."""
    if name is None:
        return {}
    if name == "swap_the_yee_sub_lattice":
        suffix = "" if CURL_ARRAYS[sub_step]["half_integer"] else "_h"
        return {"tables": {
            f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}}
    plus, minus = beta_coefficients(sub_step, grid)
    if name == "beta_coefficients_exchanged":
        return {"beta_coefficients": (minus, plus)}
    if name == "beta_coefficients_zeroed":
        return {"beta_coefficients": ((0.0, 0.0), (0.0, 0.0))}
    if name == "beta_coefficient_conjugated":
        # The +-1j read the other way round: the magnetic side handed the electric
        # side's factor. Same magnitude, the coupling reversed.
        return {"beta_coefficients": ((plus[0], -plus[1]), (minus[0], -minus[1]))}
    raise ValueError(f"unknown host mutation {name!r}")


def spec_codes(spec: Dict[str, Any]) -> Tuple[int, ...]:
    rng = np.random.default_rng(1)
    _f, _l, grid = build(np, spec, COURANTS[0], rng)
    codes, refusal = folded.folded_complex_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses fixture {spec['label']!r}: "
                         f"{refusal}")
    return tuple(int(c) for c in codes)


# ---------------------------------------------------------------------------
# One curl case
# ---------------------------------------------------------------------------

def curl_case(backend, xp, spec, sub_step, courant, value_class, guard, arm,
              host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE SEED COMES FROM A DIGEST, never from ``hash()``: Python salts ``hash()`` of
    a tuple containing strings with ``PYTHONHASHSEED``, so such a gate draws a
    different fixture every process and a failing case cannot be replayed.
    """
    started = time.time()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()
        ).digest()[:4], "big"))
    fields, layer, grid = build(xp, spec, courant, rng)
    host = seed_state(fields, grid, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "arm": complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(arm)],
        "host_mutation": host_mutation, "fold_axes": spec["axes"],
        "beta": float(grid.beta),
        "structure": structure_facts(grid),
        "word_census": grandparent.word_census(host),
    }
    asked = _GridWithCupysName(grid)
    covered, reason = family.covers_complex_beta_curl(
        fields, layer, asked, sub_step, license=LICENCE_SHAPE,
        subnormal_policy=POLICY_SHAPE)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["skipped"] = f"the predicate refuses this configuration: {reason}"
        case["seconds"] = time.time() - started
        return case
    for label, other in (("certified_complex", coverage.covers_real_pml_complex_curl),
                         ("complex_folded", folded.covers_complex_folded_curl)):
        admits, other_reason = other(fields, layer, asked, sub_step,
                                     license=LICENCE_SHAPE,
                                     subnormal_policy=POLICY_SHAPE)
        case[label + "_predicate"] = {"covered": bool(admits),
                                      "reason": other_reason}
        if admits:
            case["skipped"] = (f"the {label} family also admits this slot; two "
                               f"families on one slot is a widening")
            case["seconds"] = time.time() - started
            return case

    absorbs = absorber_is_not_the_identity(layer)
    case["absorber_is_not_the_identity"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("every absorber coefficient is the identity; a "
                           "coefficient-index error would be invisible here")
        case["seconds"] = time.time() - started
        return case
    if spec["axes"]:
        ghost = mirror_ghost_is_nonzero(fields, grid)
        case["mirror_ghost_is_nonzero"] = ghost
        if not ghost["meets_floor"]:
            case["skipped"] = ("the array path's mirror ghost is all-zero words")
            case["seconds"] = time.time() - started
            return case

    frozen = snapshot(fields)
    outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
               + tuple(CURL_ARRAYS[sub_step]["aux"]))

    getattr(stepping, sub_step)(fields, layer)
    reference = snapshot(fields)
    case["oracle_moved"] = moved_fraction(frozen, reference, outputs)
    if case["oracle_moved"] == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    live = beta_term_is_live(fields, layer, grid, sub_step, frozen, reference)
    case["beta_term_is_live"] = live
    if not live["meets_floor"]:
        case["skipped"] = ("the beta term moved no output word, or moved the wrong "
                           "components: this case measures the certified complex "
                           "kernel, not this family")
        case["seconds"] = time.time() - started
        return case

    overrides = host_mutated_inputs(host_mutation, sub_step, layer, grid)
    case["host_overrides"] = sorted(overrides)
    coefficients = overrides.pop("beta_coefficients", None)
    case["beta_coefficients"] = [list(pair) for pair in
                                 (coefficients or beta_coefficients(sub_step, grid))]

    restore(fields, frozen)
    backend.launch_curl(sub_step, fields, layer, grid, arm,
                        beta_coefficients_override=coefficients, **overrides)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs}
    case["single_launch"] = combine(parts)
    case["single_launch_per_array"] = {
        name: parts[name]["differing_floats"] for name in outputs}

    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            getattr(stepping, sub_step)(fields, layer)
            advance_sources(fields, CURL_ARRAYS[sub_step]["sources"])
        oracle_multi = snapshot(fields)
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            backend.launch_curl(sub_step, fields, layer, grid, arm)
            advance_sources(fields, CURL_ARRAYS[sub_step]["sources"])
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


def constitutive_case(backend, xp, spec, side, courant, value_class, guard, arm):
    """The CERTIFIED complex constitutive kernel against the array path, on a BETA run.

    THE CURLS RUN FIRST, both of them, so the state this sub-step reads is a state
    the beta term produced. A constitutive arm seeded with fresh random numbers
    would measure the certified kernel on a grid that merely HAS a beta attribute.
    """
    started = time.time()
    rng = np.random.default_rng(
        SEED + 977 + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()
        ).digest()[:4], "big"))
    fields, layer, grid = build(xp, spec, courant, rng)
    seed_state(fields, grid, value_class, rng)
    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "beta": float(grid.beta), "structure": structure_facts(grid),
    }
    covered, reason = family.covers_complex_beta_constitutive(
        fields, layer, _GridWithCupysName(grid), side, license=LICENCE_SHAPE,
        subnormal_policy=POLICY_SHAPE)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["skipped"] = f"the predicate refuses this configuration: {reason}"
        case["seconds"] = time.time() - started
        return case

    stepping.step_B(fields, layer)
    stepping.step_D(fields, layer)

    frozen = snapshot(fields)
    outputs = (tuple(CONSTITUTIVE_ARRAYS[side]["targets"])
               + tuple(CONSTITUTIVE_ARRAYS[side]["aux"]))
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields)
    case["oracle_moved"] = moved_fraction(frozen, reference, outputs)
    if case["oracle_moved"] == 0.0:
        case["skipped"] = "the array path changed no output word"
        case["seconds"] = time.time() - started
        return case

    restore(fields, frozen)
    backend.launch_constitutive(side, fields, layer, grid, arm)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs}
    case["single_launch"] = combine(parts)

    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if side == "H":
            stepping.update_H(fields, layer)
        else:
            stepping.update_E(fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    oracle_multi = snapshot(fields)
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        backend.launch_constitutive(side, fields, layer, grid, arm)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
             for name in outputs}
    case["multi_step"] = combine(multi)
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET
    case["seconds"] = time.time() - started
    return case


LICENCE_SHAPE: Dict[str, Any] = dict(fold_gate.LICENCE_SHAPE)
POLICY_SHAPE: str = fold_gate.POLICY_SHAPE


def case_product(product: str):
    specs = BETA_SPECS if product == "full" else BETA_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, sub_step, courant, value_class)
            for spec in specs for sub_step in SUB_STEPS
            for courant in courants for value_class in classes]


def run_sweep(results, out_path, backend, xp, product, guard, arm):
    cases = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = curl_case(backend, xp, spec, sub_step, courant, value_class, guard, arm)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"codes={case['structure']['boundary_codes']} beta={case['beta']:+.4f} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={case['single_launch']['differing_floats']} "
            f"beta_moved={case['beta_term_is_live']['total']} "
            f"({case['seconds']:.1f} s)")
    return cases


def run_constitutive(results, out_path, backend, xp, product, guard, arm):
    cases = []
    specs = BETA_SPECS if product == "full" else BETA_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    plan = [(spec, side, courant, value_class)
            for spec in specs for side in SIDES
            for courant in courants for value_class in classes]
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = constitutive_case(backend, xp, spec, side, courant, value_class,
                                 guard, arm)
        cases.append(case)
        results.setdefault("constitutive", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
            f"c={courant} {value_class} "
            f"single={'IDENTICAL' if case['single_launch']['bit_identical'] else 'DIVERGED'} "
            f"multi={'IDENTICAL' if case['multi_step']['bit_identical'] else 'DIVERGED'} "
            f"diff={case['single_launch']['differing_floats']} "
            f"({case['seconds']:.1f} s)")
    return cases


scorable_baselines = fold_gate.scorable_baselines
leg_caught = fold_gate.leg_caught
_score = fold_gate._score


def mutation_plan(product: str, scorable: Optional[set] = None):
    specs = BETA_SPECS if product == "full" else BETA_SPECS[:3]
    plan = [(spec, sub_step, INEXACT_COURANT, "uniform")
            for spec in specs for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [e for e in plan if (e[0]["label"], e[1]) in scorable]


def run_host_mutations(results, out_path, backend, xp, product, scorable, arm):
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    for name in HOST_MUTATIONS:
        legs = [curl_case(backend, xp, spec, sub_step, courant, value_class,
                          "fmad_false", arm, host_mutation=name)
                for spec, sub_step, courant, value_class in plan]
        out[name] = dict(_score(legs, name not in HOST_NULL_MUTATIONS), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results, out_path, backend, xp, product, scorable, arm):
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    try:
        for name, transform in SOURCE_MUTATIONS.items():
            required = MUTATION_REQUIRES_CODE.get(name)
            leg_plan = list(plan)
            if required is not None:
                leg_plan = [e for e in leg_plan if required in spec_codes(e[0])]
            if name in MUTATION_REQUIRES_A_WALL:
                leg_plan = [e for e in leg_plan
                            if any(code != BC_PERIODIC for code in spec_codes(e[0]))]
            if name in MUTATION_REQUIRES_A_PHASE:
                leg_plan = [e for e in leg_plan if any(e[0]["k"])]
            sites = backend.set_source_transform(transform, arm)
            if sites == 0:
                out[name] = {"armed": False,
                             "why": (f"matched {sites} site(s); the mutation and "
                                     f"the emitted source have drifted apart")}
                log(f"[src-mut] {name}: NOT ARMED")
                backend.set_source_transform(None, arm)
                results["source_mutations"] = out
                save(results, out_path)
                continue
            legs = [curl_case(backend, xp, spec, sub_step, courant, value_class,
                              "fmad_false", arm)
                    for spec, sub_step, courant, value_class in leg_plan]
            backend.set_source_transform(None, arm)
            scored = _score(legs, name not in NULL_MUTATIONS)
            if name in INHERITED_MUTATIONS:
                scored["verdict"] = ("CAUGHT" if scored["caught"] else "UNCAUGHT")
                scored["bar"] = ("inherited: caught on at least one fixture "
                                 "(full scoring is gate_cuda_complex's)")
            if (name in FINDING_MUTATIONS and scored["ran"]
                    and scored["caught"] == 0):
                scored["verdict"] = "REPORTED"
                scored["finding"] = (
                    "uncaught on every leg: the leg's own contract calls this a "
                    "platform finding, not a failure -- the emitted arithmetic "
                    "reads the beta coefficient's imaginary word alone "
                    "(mul_imag_coefficient_left), so the real word this mutation "
                    "zeroes is defensive pass-through on this platform. A PARTIAL "
                    "would still block.")
            out[name] = dict(scored, armed=True, sites=sites,
                             requires_code=required,
                             requires_a_phase=name in MUTATION_REQUIRES_A_PHASE,
                             inherited=name in INHERITED_MUTATIONS,
                             legs_in_plan=len(plan), legs_scored=len(leg_plan),
                             caught_on=[f"{c['label']}::{c['sub_step']}"
                                        for c in legs if leg_caught(c)],
                             cases=legs)
            log(f"[src-mut] {name}: caught {scored['caught']}/{scored['ran']} "
                f"sites={sites} -> {out[name]['verdict']}")
            results["source_mutations"] = out
            save(results, out_path)
    finally:
        backend.set_source_transform(None, arm)
    return out


def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    reasons: List[str] = []

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = "folded" if case["fold_axes"] else "unfolded"
        entry = arms.setdefault(key, {"cases": 0, "single_identical": 0,
                                      "multi_cases": 0, "multi_identical": 0,
                                      "differing_words": 0, "labels": set()})
        entry["cases"] += 1
        entry["labels"].add(case["label"])
        entry["single_identical"] += bool(case["single_launch"]["bit_identical"])
        if "multi_step" in case:
            entry["multi_cases"] += 1
            entry["multi_identical"] += bool(case["multi_step"]["bit_identical"])
        entry["differing_words"] += case["single_launch"]["differing_floats"]
    for entry in arms.values():
        entry["labels"] = sorted(entry["labels"])

    if not scored:
        reasons.append("no curl case was scored at all")
    for arm in ("unfolded", "folded"):
        if arm not in arms:
            reasons.append(f"the sweep contained no {arm} beta case; twelve of the "
                           f"sixteen census slots are folded and four are not, so "
                           f"releasing on one arm is releasing on part of the set")
            continue
        entry = arms[arm]
        if entry["single_identical"] != entry["cases"]:
            reasons.append(
                f"{arm}: single-launch divergence on "
                f"{entry['cases'] - entry['single_identical']} of {entry['cases']} "
                f"cases ({entry['differing_words']} words)")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(
                f"{arm}: multi-step divergence on "
                f"{entry['multi_cases'] - entry['multi_identical']} of "
                f"{entry['multi_cases']} cases")

    constitutive = results.get("constitutive", {}).get("fmad_false", [])
    per_side: Dict[str, Dict[str, Any]] = {}
    for case in constitutive:
        entry = per_side.setdefault(case["side"], {"cases": 0, "single": 0,
                                                   "multi_cases": 0, "multi": 0,
                                                   "skipped": 0})
        if case.get("skipped"):
            entry["skipped"] += 1
            continue
        entry["cases"] += 1
        entry["single"] += bool(case["single_launch"]["bit_identical"])
        entry["multi_cases"] += 1
        entry["multi"] += bool(case["multi_step"]["bit_identical"])
    for side in SIDES:
        entry = per_side.get(side)
        if not entry or entry["cases"] == 0:
            reasons.append(f"update_{side} was never scored; the constitutive "
                           f"admission is an ADMISSION and needs its own device leg")
            continue
        if entry["single"] != entry["cases"] or entry["multi"] != entry["multi_cases"]:
            reasons.append(f"update_{side}: divergence on "
                           f"{entry['cases'] - entry['single']} single / "
                           f"{entry['multi_cases'] - entry['multi']} multi cases")

    mutations = {}
    for group in ("host_mutations", "source_mutations"):
        for name, entry in (results.get(group) or {}).items():
            if not entry.get("armed", True):
                mutations[f"{group}:{name}"] = "NOT ARMED"
                reasons.append(f"{group} {name} was not armed; a mutation that "
                               f"matched nothing tests nothing")
                continue
            mutations[f"{group}:{name}"] = entry["verdict"]
            # ``REPORTED`` is the FINDING_MUTATIONS channel: a leg whose own
            # contract names an all-legs-uncaught outcome a platform finding.
            # PARTIAL on such a leg never reaches REPORTED and still blocks here.
            if entry["verdict"] not in ("CAUGHT", "NULL CONFIRMED", "REPORTED"):
                reasons.append(f"{group} {name}: {entry['verdict']} "
                               f"({entry['caught']}/{entry['ran']})")

    certifies = bool(results.get("backend_certifies"))
    if not certifies:
        reasons.append("this backend compiles no NVRTC and certifies nothing; the "
                       "verdict below is about the TRANSCRIPTION only")
    return {
        "released": not reasons and certifies,
        "scored_cases": len(scored),
        "arms": {k: dict(v) for k, v in arms.items()},
        "constitutive_per_side": per_side,
        "mutations": mutations,
        "reasons": reasons,
        "claim": ("the hand-CUDA complex beta curl pair is byte-identical to "
                  "stepping.step_B / step_D on a complex64 special_kz run, per "
                  "sub-step, at one launch and at 60, FOLDED at both terminations "
                  "and UNFOLDED, at an exactly representable courant and at one "
                  "that is not, on physical-band and subnormal-band operands; and "
                  "the CERTIFIED complex constitutive pair is byte-identical to "
                  "update_H / update_E on the same beta runs, scored per side"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate "
            "clause, not a wiring",
            "REAL storage with beta: that is special_kz_curl's certified pair and "
            "is refused here by name",
            "a z-thick grid: Grid._resolve_beta refuses beta outside an effective "
            "2-D Cartesian grid, so nz == 1 on every grid this family can be "
            "handed and the z ghost rule is exercised at one cell only",
            "cylindrical coordinates with beta: MEEP aborts and Grid refuses the "
            "pairing at construction",
            "a folded axis carrying a Bloch phase, a conductivity, BFAST, or three "
            "simultaneous fold planes: refused by inherited clauses",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def save(results, out_path: str) -> None:
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global LICENCE_SHAPE, POLICY_SHAPE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cupy", "host"), default="cupy")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--probe", default=None)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    results: Dict[str, Any] = {
        "gate": "cuda_complex_beta",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("is the hand-CUDA complex beta curl pair byte-identical to "
                     "stepping.step_B / step_D on a complex64 special_kz run, "
                     "folded and unfolded, and is the CERTIFIED complex "
                     "constitutive pair byte-identical to update_H / update_E on "
                     "the same run, per sub-step and per side?"),
        "family_corpus_digest": family.corpus_digest(),
        "folded_corpus_digest": folded.corpus_digest(),
        "emitter_corpus_digest": complex_emitter.corpus_digest(),
        "deltas": list(family.BETA_DELTAS),
    }

    licence = grandparent.load_licence(args.probe, args.subnormal_policy)
    results["expansion_licence"] = licence
    if licence["usable"]:
        LICENCE_SHAPE = dict(licence["verdict"])
        LICENCE_SHAPE["policy_resolved"] = args.subnormal_policy
        POLICY_SHAPE = args.subnormal_policy
        arm = licence["arm"]
    else:
        if args.backend == "cupy":
            log("[fatal] no usable expansion licence; the arm may not be guessed")
            results["status"] = "refused: no usable expansion licence"
            save(results, args.out)
            return 2
        arm = "FMA_V1"
        results["arm_note"] = ("the host backend binds FMA_V1 by default and "
                               "certifies nothing about which arm this platform "
                               "reproduces")
    results["arm"] = arm

    if args.backend == "cupy":
        if cp is None:
            log("[fatal] --backend cupy but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                       _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
        xp = cp
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("host-compiled backend: it executes the "
                                           "EMITTED characters but compiles no "
                                           "NVRTC and certifies nothing about it")}
        xp = np

    backend = build_backend(args.backend)
    results["backend_certifies"] = bool(backend.certifies)
    save(results, args.out)

    try:
        for guard, options in GUARD_SETS:
            if args.backend == "host" and guard != "fmad_false":
                continue
            backend.set_guard(options)
            log(f"[guard] {guard} options={options}")
            run_sweep(results, args.out, backend, xp, args.product, guard, arm)
            run_constitutive(results, args.out, backend, xp, args.product, guard, arm)
        backend.set_guard(("--fmad=false",))

        if not args.skip_mutations:
            scorable, excluded = scorable_baselines(results)
            results["mutation_scope"] = {
                "scorable": sorted("::".join(k) for k in scorable),
                "excluded": excluded,
                "why": ("a mutation leg answers 'can this gate see this defect', "
                        "and only an arm whose unmutated baseline is bit-identical "
                        "can answer it"),
            }
            log(f"[mut-scope] {len(scorable)} scorable pairs; "
                f"{len(excluded)} excluded")
            save(results, args.out)
            run_host_mutations(results, args.out, backend, xp, args.product,
                               scorable, arm)
            run_source_mutations(results, args.out, backend, xp, args.product,
                                 scorable, arm)
    finally:
        backend.restore()

    if args.backend == "cupy":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, entry in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {entry['single_identical']}/{entry['cases']} "
            f"multi {entry['multi_identical']}/{entry['multi_cases']} "
            f"diff={entry['differing_words']}")
    for side, entry in sorted(verdict["constitutive_per_side"].items()):
        log(f"[verdict]   update_{side}: single {entry['single']}/{entry['cases']} "
            f"multi {entry['multi']}/{entry['multi_cases']} "
            f"skipped {entry['skipped']}")
    for name, value in sorted(verdict["mutations"].items()):
        log(f"[verdict]   mut {name}: {value}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
