"""The hand-CUDA DISPERSIVE electric constitutive arm -- ``update_E`` where the source is ``D - sum P``.

THREE FAMILIES, ONE FILE, AND THEY ARE NOT THE SAME SUB-STEP AS EACH OTHER
=============================================================================

The 2026-08-20 union census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closeout/``) reads
653 / 759 slots admitted by some shipped hand-CUDA arm. Fourteen of the 106
unserved slots are refused by clauses about DISPERSION, and this module answers
three of them::

    8 x update_E   cuda_constitutive   "dispersion: update_E's source is
                                        (D - sum P), not D, and update_P closes
                                        the step"                    <- arm 1
    3 x update_E   cuda_no_pml         "fields.stores_E is True: update_E writes
                                        E[...] = (D - sum P) * inv_eps
                                        (stepping.py:1022)"           <- arm 2
    3 x update_P   cuda_ade            "no active PML layer: this family binds
                                        f_w as the drive and drive_field returns
                                        the stored E without one"    <- arm 3

ARM 1 AND ARM 2 ARE DIFFERENT SUB-STEPS, not a flag on one. Read
``stepping.update_E`` (stepping.py:926-993) and the two arms are two branches of
one ``if``::

    if pml_active:
        kps, kms = _constitutive_coefficients(pml, axis_name, half_integer=True)
        _apply_constitutive_pml(field, constitutive, kps, kms, f_w, scratch)   # :985-989
    else:
        # MEEP update_eh.cpp with dsigw == NO_DIRECTION: E = chi1inv * (D - P)
        getattr(fields, component)[...] = constitutive                         # :990-993

Arm 1 accumulates ``f += (kap+sig)*fw - (kap-sig)*fwprev`` into an auxiliary that
is STATE, indexes a per-axis coefficient vector, and reads the previous ``f_w``
before overwriting it. Arm 2 has no auxiliary, no coefficient, no history and no
per-axis index: it is a store. A kernel that served both behind one flag would be
two kernels wearing one name, so they are two device strings, and arm 2's
predicate refuses an active layer by name while arm 1's requires one.

ARM 3 IS NOT A NEW KERNEL AND THIS FILE DOES NOT WRITE ONE. That was the open
question this round was told to settle by measurement, and the answer is a
PREDICATE WIDENING over the shipped ``ade_kernels.update_P_pml_real*``
pair. ``stepping.update_P`` (stepping.py:1360-1384) deletes its ``pml`` argument
at :1424 -- "the absorber reaches the polarization through W and nowhere else" --
and passes ``fields.drive_field`` as the drive. ``Fields.drive_field``
(fields.py:1140-1163) is::

    if self._pml_active:
        return getattr(self, 'f_w_' + component)
    return getattr(self, component)

so the ONLY thing an absorber changes about this sub-step is WHICH ARRAY the
launcher binds -- and ``ade_kernels.update_P_fused_pml_real`` already binds
``drive(component)`` rather than reaching for ``f_w`` itself (ade_kernels.py:
"``drive`` IS THE GATE'S DOOR"). :func:`covers_no_pml_ade_update_p` is therefore
the shipped ADE predicate with one clause inverted and the drive identity check
re-pointed at the stored E. ``gate_cuda_dispersive.py``'s ``arm3`` leg is what
turns that reading into a measurement: it runs the SHIPPED launcher against
``stepping.update_P`` on a layerless run, and carries
``arm3_wrong_drive_CONTROL`` -- the same launcher with ``D`` bound as the drive --
which must DIVERGE, because a leg whose control cannot fail has not shown that
binding the right array is what produced the pass.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

``stepping.update_E``'s diagonal dispersive branch, term for term::

    source       = fields.displacement_minus_polarization(component)   # :981
    constitutive = source * fields.inverse_epsilon_for(component)      # :982-984

and ``Fields.displacement_minus_polarization`` (fields.py:1079-1105)::

    displacement = getattr(self, 'D' + component[1])
    contributors = [s for s in self.polarizations if s.drives(component)]
    if not contributors:
        return displacement            # THE D ARRAY ITSELF, aliased, not copied
    scratch[...] = displacement
    for state in contributors:
        state.subtract_into(component, scratch)   # target -= state.P[component]
    return scratch

FIVE THINGS DECIDE BIT-IDENTITY HERE
------------------------------------

1. THE POLE CHAIN IS SEQUENTIAL AND LEFT-TO-RIGHT, in ``fields.polarizations``
   order filtered by ``drives``. ``((D - P0) - P1) - P2``, never
   ``D - (P0 + P1 + P2)``. float32 addition is not associative, and the
   pre-accumulated form is the one a reader of the phrase "D minus the sum of the
   polarizations" writes by accident. Both forms agree EXACTLY at one pole, which
   is why :data:`POLE_COUNTS_SWEPT` carries multi-pole rows and why a gate leg
   that scored ``sum_then_subtract`` only at one pole would be reporting a
   harness defect as a kernel pass. (The sibling Triton track measured that pair
   at 20/20 caught with two poles and 0/10 with one --
   ``probe_fused_kernel_bit_identity.DISPERSIVE_SOURCE_MUTATIONS``.)

2. A COMPONENT WITH NO CONTRIBUTOR READS ``D`` DIRECTLY, with no subtraction
   formed at all (fields.py:1096-1098). ``D - 0.0`` is bit-exact for every finite
   value, but ``-0.0 - 0.0`` is ``-0.0`` while not subtracting leaves ``-0.0``
   too, and a subtraction of a zero-valued P array is exact anyway -- so this
   costs nothing in the ordinary case and is transcribed because the ALIASING is
   what makes the degenerate configuration reduce to the certified non-dispersive
   kernel by construction rather than by rounding. The emitter emits ZERO
   subtraction lines for a count of 0; :data:`POLE_COUNTS_SWEPT` carries
   ``(0, 0, 0)``, which makes the emitted body character-comparable with
   ``constitutive_kernels._update_E_pml_real_kernel_code``'s and is checked
   as such by ``test_dispersive.py``.

3. ``D`` IS ON THE LEFT OF THE INVERSE-EPSILON MULTIPLY (stepping.py:1011). IEEE
   multiply commutes, so this is transcription discipline rather than a bit --
   and the gate carries ``inv_eps_left`` as a NULL mutation which MUST come back
   UNCAUGHT, paired with ``drop_inverse_epsilon_dispersive`` on the same
   expression which MUST be caught. Without the pair, "the operand order is
   inert" would be a claim about a leg nobody showed could fail.

4. NO FMA. ``--fmad=false`` is CORRECTNESS, not tuning. Arm 1 has three
   contraction candidates per component (``s * inv_eps`` into either
   accumulation, and ``kps * src`` / ``kms * prev`` into their adds); arm 2 has
   one. The option tuple is SPELLED IN THIS FILE rather than imported, for the
   reason every sibling spells its own: a bit-identity probe loads these modules
   BY PATH, outside the package, where an import could pick up a different tuple
   than the one a gate compiled.

5. THE POLE POINTERS ARE RESOLVED IMMEDIATELY BEFORE EACH LAUNCH, never cached.
   ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` every step
   (dispersion.py:689-691), so ``state.P[component]`` is a DIFFERENT buffer after
   ``update_P`` runs. A launcher that resolved the pole list once and reused the
   views would be stale from step two, and stale in a way that still computes: it
   would read the previous step's polarization forever. The shipped ADE launcher
   carries the same hazard in as many words and this one is the other half of it.

=============================================================================
WHY THE POLE COUNT IS COMPILE-TIME
=============================================================================

``sum P`` is variadic: the corpus rows this arm serves carry 1, 2, 5 and 6
registered susceptibilities. Three ways to write that, and the choice is not
taste:

* a device array of pointers plus a count -- one kernel for every arity, but the
  pointer table has to be REBUILT ON EVERY LAUNCH because the P buffers rotate,
  which is a device allocation per sub-step per step. That is port-reference
  defect 5.14 (the complex wrappers' per-launch ``ascontiguousarray``) written
  into a new kernel deliberately;
* a fixed maximum arity with null pointers for the unused slots -- a branch per
  pole per cell, and a null dereference one predicate bug away;
* COMPILE-TIME SPECIALIZATION on the ``(n_Ex, n_Ey, n_Ez)`` triple. No
  indirection, no per-launch allocation, no branch, and the emitted body for a
  count of zero is character-identical to the certified non-dispersive kernel's.

The third is what this file does, and it is the shape ``offdiag_emitter.py``
already uses for the off-diagonal row mask. The kernel NAME is fixed per arm and
the SOURCE varies with the triple; ``compile_cache.kernel_cache_key`` keys the
memo on the source string, so two arities are two entries and never collide.

:data:`POLE_COUNT_CAP` is the ceiling, and it is a MEASURED one rather than an
arbitrary one: the gate sweeps every count from 0 to the cap on both arms, so a
triple within it has been run and a triple past it is refused by name. Raising it
means sweeping it.

=============================================================================
WHAT THIS ARM ADMITS THAT ITS SIBLINGS REFUSE, AND WHAT IT REFUSES ANYWAY
=============================================================================

ADMITTED, on the sibling constitutive predicate's OWN measured grounds:

* **a mirror fold**, up to :data:`ADE_FOLD_PLANES_SWEPT`'s two simultaneous
  planes -- THIS family's own number, not the sibling record's. Five of arm 1's
  eight corpus rows are folded. The admission is not inherited: this kernel adds
  only element-wise subtractions to a sub-step already measured on folded extents,
  and the gate re-scores the fold cases here rather than transferring the
  sibling's verdict. The cap stopped reading
  ``coverage.CONSTITUTIVE_FOLD_ADMISSION`` on 2026-08-20, when that record's own
  number was raised to three by a re-run of the CONSTITUTIVE gate; see the
  constant for why a shared cap is a way to widen a family nobody measured.
* **a conductivity.** ``fields.condfac_for`` is read in ``stepping._apply_curl``
  (stepping.py:508) and nowhere else in the module; ``update_E`` never mentions
  it. Two of arm 2's three corpus rows carry one (``absorber-1d.py``,
  ``TestAbsorber.test_absorber``), and it is charged to the CURL, not here.

REFUSED, each with the line behind it:

* **an off-diagonal chi1inv row.** ``update_E``'s ``elif offdiagonal:`` branch
  (stepping.py:1001-1008) forms a ROW PRODUCT that reads the OTHER components'
  ``D - sum P`` volumes at NEIGHBOURING cells (``_offdiagonal_terms``,
  stepping.py:1196), so the sub-step stops being element-wise and gains a ghost
  rule. That is ``offdiag_emitter.py``'s kernel shape crossed with this one's
  pole chain, and the shipped CUDA off-diagonal family additionally refuses a
  fold. THE COST IS EXACTLY ONE SLOT AND IT IS NAMED: ``absorbed_power_density.py``
  is off-diagonal AND dispersive AND folded, and stays unserved. The Triton
  sibling for it is ``folded_offdiag_dispersive_update_e.py``; nothing on the
  hand-CUDA track carries it and this file does not pretend to.
* **an instantaneous chi2/chi3.** The Pade factor REPLACES the constitutive
  product (stepping.py:999-1000) rather than scaling this one's source.
* **complex storage, Bloch, BFAST, special_kz, Dcyl.** Each is another family's,
  and each would be a second kernel rather than a clause.

=============================================================================
NOTHING DISPATCHES ANY OF THIS
=============================================================================

No module in ``meep_gpu/`` imports ``cuda_kernels`` at all. A predicate returning
True licenses a MEASUREMENT, not a production step. :data:`CERTIFIED_KERNELS` and
:data:`UNCERTIFIED_KERNELS` partition the emitted kernel names, and a name in
neither fails ``test_dispersive.py``.

THIS MODULE IMPORTS NO CuPy AT MODULE SCOPE, deliberately: the predicate census
(``parity/meep_gpu/.../predicate_battery.py``) runs on a laptop with NumPy only,
and a predicate that cannot be imported there cannot be asked -- which is exactly
how a family gets counted as a coverage gap it does not have.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coverage import (ADE_BOUNDARY_KINDS, ADE_ELECTRIC_COMPONENTS,
                       CONSTITUTIVE_BOUNDARY_KINDS,
                       COVERED_SUSCEPTIBILITY_KINDS,
                       _ade_coefficient_problem, _array_problem, _backend,
                       _base_address, _boundary_kinds_from,
                       _coefficient_vector_problem, _grid_facts,
                       ade_sigma_is_volume)
from . import compile_cache

#: SIMULTANEOUS MIRROR PLANES THIS FAMILY'S GATE HAS SCORED, and the reason it is
#: a constant here rather than a read of
#: :data:`~.coverage.CONSTITUTIVE_FOLD_ADMISSION`.
#:
#: Until 2026-08-20 this clause read that record's ``folded_planes_swept``
#: directly. On 2026-08-20 the sibling constitutive gate was re-run with a third
#: fold plane and that field went 2 -> 3 -- which would have silently widened THIS
#: family to three planes too, on a sweep it never ran. The corpus drives no
#: three-plane dispersive row, so the widening would have bought zero slots and
#: been invisible to the census: an over-claim with no measurement anywhere behind
#: it and nothing to catch it.
#:
#: A CAP IS A FACT ABOUT A GATE, and a gate answers for one family. Raising this
#: one means adding a three-plane spec to ``gate_cuda_ade.py``, running it on a
#: device under both float32 subnormal policies, and recording what came back.
ADE_FOLD_PLANES_SWEPT: int = 2

#: The three E components, in ``E_CONSTITUTIVE_TERMS`` order (stepping.py:228).
#: The order is the axis order -- component c reads axis c's coefficient vector --
#: and getting it backwards is the "smooth, converged, entirely wrong absorber"
#: the sibling module's note 2 names as the highest-consequence confusion.
ELECTRIC_TERMS: Tuple[Tuple[str, str, str], ...] = (
    ("Ex", "Dx", "x"), ("Ey", "Dy", "y"), ("Ez", "Dz", "z"))

#: The two arms. ``pml`` is stepping.py:1014-1018, ``no_pml`` is :1019-1022.
DISPERSIVE_ARMS: Tuple[str, ...] = ("pml", "no_pml")

#: The fixed device entry point per arm. The pole arity varies the SOURCE, not
#: the name; ``compile_cache.kernel_cache_key`` keys the memo on the source.
ARM_KERNEL_NAMES: Dict[str, str] = {
    "pml": "update_E_pml_real_dispersive",
    "no_pml": "update_E_no_pml_real_dispersive",
}

#: The largest per-component contributor count this family will serve. MEASURED,
#: not chosen: ``gate_cuda_dispersive.py`` sweeps every count 0..CAP on both arms,
#: so a triple inside it has been run on a device and a triple past it is refused
#: by name rather than served on an extrapolation. The corpus's own maximum is 6
#: (``stochastic_emitter*.py``, six registered susceptibilities each driving all
#: three components).
POLE_COUNT_CAP = 6

#: The arities the gate sweeps, and the reason each is in the list.
POLE_COUNTS_SWEPT: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),   # the reduction: the emitted body is the certified kernel's
    (1, 1, 1),   # absorbed_power_density.py -- and the arity at which
                 # sum_then_subtract and reverse_pole_order are PROVABLY inert
    (2, 2, 2),   # material-dispersion.py -- the smallest arity that can see them
    (2, 0, 3),   # mixed, including the aliasing branch on one component only
    (5, 5, 5),   # TestLoadDump.*_2d, absorber-1d.py, TestAbsorber.test_absorber
    (6, 6, 6),   # stochastic_emitter*.py -- the corpus ceiling
)

#: NVRTC compile options -- CORRECTNESS, not performance. See note 4 of the
#: header for the three contraction candidates this closes on arm 1.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block. One element per lane, flat 1-D grid -- the geometry every
#: certified hand-CUDA family uses. The sub-step is element-wise (the pole chain
#: adds loads, not neighbours), so there is no tile to shape.
_DISPERSIVE_THREADS = 256

#: RECORDED 2026-08-27. The gate was re-run on the GPU host and RELEASED under both
#: float32 subnormal policies; the block is ``cuda_dispersive_2026-08-27`` in
#: ``certification.json``, and the two halves landed together as the partition
#: test requires. The subject module was checked UNCHANGED since that run before
#: the record was landed, so this names the bytes that ship.
CERTIFIED_KERNELS = (
    "update_E_pml_real_dispersive",
    "update_E_no_pml_real_dispersive",
)
#: WHAT THE DEVICE MEASURED, transcribed from the artifact rather than remembered.
#: Two legs, one per float32 subnormal policy, on one RTX A6000 (compute 8.6,
#: CuPy 13.5.1) with ``CUPY_ACCELERATORS=""`` and a private ``CUPY_CACHE_DIR`` per
#: policy -- the cache key is computed ABOVE the ``-ftz=true`` strip seam, so a
#: shared cache would have served one policy's binary to the other leg.
#:
#: ``emitted_corpus_digest`` is the pin that matters, and it is NOT a file hash: a
#: file hash stops matching when a docstring gains a comma, while this is a digest
#: over every emitted device body at every swept arity on both arms. The clause in
#: this module that moved after the gate ran (:data:`ADE_FOLD_PLANES_SWEPT`) is a
#: PREDICATE clause and changed no emitted byte, which is why the verdict below
#: still describes the strings NVRTC compiled.
GATE_VERDICT: dict = {
    "gate": "parity/meep_gpu/gate_cuda_dispersive.py",
    "artifact": "parity/meep_gpu/results/cuda_dispersive_2026-08-21_rename/",
    "legs": {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"},
    "released": True,
    "scored_cases_per_leg": 414,
    "scored_by_arm": {"arm1": 216, "arm2": 108, "arm3": 90},
    "single_launch_identical": 414,
    "multi_step_identical": 414,
    "multi_step_launches": 60,
    "arities_scored": [[0, 0, 0], [1, 1, 1], [2, 0, 3], [2, 2, 2], [5, 5, 5],
                       [6, 6, 6]],
    "value_classes": ["cancellation", "subnormal_band", "uniform"],
    "source_mutations": "9 CAUGHT, 2 NULL CONFIRMED",
    "host_mutations": "6 CAUGHT, 1 NULL CONFIRMED",
    "contraction_guard": ("load-bearing, MEASURED: the unguarded leg diverged on "
                          "138 of 207 cases at the inexact courant"),
    "verdict_flips_against_planted_defect": True,
    "emitted_corpus_digest": 'c86531c75cc9a3637e1eb4f60615b5bb0b46ab5d008fb841a66e9c7e8853c1e2',
}

#: Spelled as an unannotated dict for the reason ``complex_pml_kernels.py``
#: spells its own that way: the partition test reads it off the syntax tree
#: WITHOUT importing the module, so it can run on a host with no CuPy, and an
#: annotated assignment is an ``ast.AnnAssign`` which that reader does not match.
UNCERTIFIED_KERNELS = {}
# =============================================================================
# THE DEVICE CODE -- emitted, because the pole arity is a compile-time axis
# =============================================================================

_POLE_CHAIN_NOTE = r'''
// stepping.update_E's diagonal dispersive branch (stepping.py:968-993), whose
// source is Fields.displacement_minus_polarization (fields.py:1079-1105):
//
//     scratch[...] = D                       // or D ITSELF when nothing drives
//     for state in contributors:             // fields.polarizations order
//         scratch -= state.P[component]      // MEEP subtract_P
//     constitutive = scratch * inv_eps       // :982, D-side on the LEFT
//
// so the chain is ((D - P0) - P1) - P2 ... : SEQUENTIAL, LEFT TO RIGHT, in
// fields.polarizations order filtered by drives(). NOT D - (P0 + P1 + ...):
// float32 addition is not associative and the two agree exactly at one pole,
// which is why the gate's mutation legs are scored at two and above.
//
// A COMPONENT WITH NO CONTRIBUTOR reads D directly and forms no subtraction --
// fields.py:1096-1098 returns the D array itself, aliased and uncopied. The
// emitter emits zero subtraction lines for that component, which makes the
// arity-(0,0,0) body character-identical to the certified non-dispersive
// kernel's.
//
// THE POLE POINTERS ROTATE. PolarizationState.update moves P/P_prev/_scratch
// every step (dispersion.py:689-691), so the launcher resolves state.P[c]
// immediately before each launch and never caches a view.
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 968-993->997-1022

_PML_ACCUMULATION_NOTE = r'''
// stepping._apply_constitutive_pml (stepping.py:2065-2098), MEEP's
// step_update_EDHB with dsigw active:
//
//     realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
//     fw[i] = g[i] * u[i];
//     f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
//
// The two accumulations are kept SEPARATE and LEFT-TO-RIGHT, and `prev` is
// loaded before the store. Character for character the certified
// constitutive_kernels.constitutive_apply body; restated here rather than
// imported because a bit-identity probe rewrites these strings by hand and a
// shared prelude would make one mutation hit two families.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms
) {
    float prev = fw[idx];
    fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2065-2098->2112-2145


def normalized_pole_counts(counts: Sequence[int]) -> Tuple[int, int, int]:
    """Validate and canonicalize a ``(n_Ex, n_Ey, n_Ez)`` arity triple.

    Raises rather than clamping. A count past :data:`POLE_COUNT_CAP` has never
    been run on a device, and emitting for it anyway is exactly the "admitted by
    argument alone" move this package refuses everywhere else.
    """
    values = tuple(int(value) for value in counts)
    if len(values) != 3:
        raise ValueError(
            f"the pole arity is one count per E component, got {len(values)}: "
            f"{counts!r}")
    for component, value in zip(ELECTRIC_TERMS, values):
        if value < 0:
            raise ValueError(f"{component[0]} carries a negative pole count {value}")
        if value > POLE_COUNT_CAP:
            raise ValueError(
                f"{component[0]} carries {value} contributors and POLE_COUNT_CAP "
                f"is {POLE_COUNT_CAP}; gate_cuda_dispersive.py has swept counts "
                f"0..{POLE_COUNT_CAP} and nothing above. Raising the cap means "
                f"sweeping it, not editing this number.")
    return values  # type: ignore[return-value]


def pole_parameter_names(counts: Sequence[int]) -> Tuple[str, ...]:
    """The pole pointer parameter names, in the order the signature declares them."""
    values = normalized_pole_counts(counts)
    names: List[str] = []
    for (target, _source, _axis), count in zip(ELECTRIC_TERMS, values):
        names.extend(f"P_{target}_{index}" for index in range(count))
    return tuple(names)


def _pole_parameter_block(counts: Sequence[int]) -> str:
    names = pole_parameter_names(counts)
    if not names:
        return ""
    return "".join(f"    const float* __restrict__ {name},\n" for name in names)


def _source_lines(counts: Sequence[int]) -> str:
    """The ``D - sum P`` chain and the inverse-epsilon multiply, per component."""
    values = normalized_pole_counts(counts)
    lines: List[str] = []
    for (target, source, axis), count in zip(ELECTRIC_TERMS, values):
        if count == 0:
            # NO INTERMEDIATE AT ALL, because the array path forms none:
            # ``displacement_minus_polarization`` returns the D array ITSELF when
            # nothing drives the component (fields.py:1096-1098). This line is
            # character-identical to the certified non-dispersive kernel's
            # (``constitutive_kernels._update_E_pml_real_kernel_code``),
            # which ``test_dispersive.py`` checks as a string rather than as a
            # claim -- so the degenerate arity reduces to certified bytes by
            # construction rather than by rounding.
            line = f"    float src_{axis} = {source}[idx] * inv_eps_{target}[idx];"
            if axis == "x":
                # The trailing comment sits on the first line only, exactly as in
                # the certified kernel, so the three-line block compares as a
                # string rather than "as a string once you ignore the comments".
                line += "   // stepping.py:982: source * inv_eps"  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
            lines.append(line)
            continue
        lines.append(f"    float s_{axis} = {source}[idx];")
        for index in range(count):
            # ONE SUBTRACTION PER CONTRIBUTOR, in order: the array path's
            # `scratch -= state.P[component]` (fields.py:1103), which rounds to
            # float32 at every step exactly as this register does.
            lines.append(f"    s_{axis} = s_{axis} - P_{target}_{index}[idx];")
        # stepping.py:1011 -- `source * fields.inverse_epsilon_for(component)`,
        # D-side on the LEFT.
        lines.append(f"    float src_{axis} = s_{axis} * inv_eps_{target}[idx];")
    return "\n".join(lines)


def dispersive_source(arm: str, counts: Sequence[int]) -> str:
    """The full device source for one arm at one pole arity.

    ``arm`` is ``"pml"`` (stepping.py:1014-1018) or ``"no_pml"`` (:1019-1022).

    THE INVERSE-EPSILON POINTERS ARE NOT ``__restrict__``, deliberately and for
    the certified sibling's reason: an isotropic run hands the SAME device
    pointer three times (``Fields.set_isotropic_epsilon_volume``,
    fields.py:1321-1326), and ``restrict`` on mutually aliasing arguments is a
    promise the caller cannot keep. They are read-only, so nothing is lost but
    the promise. The POLE pointers ARE ``__restrict__``: two susceptibilities
    never share a P buffer (``PolarizationState.__init__`` allocates its own,
    dispersion.py:645-647), and the launcher and the predicate both check it.
    """
    if arm not in DISPERSIVE_ARMS:
        raise ValueError(f"arm must be one of {DISPERSIVE_ARMS}, got {arm!r}")
    values = normalized_pole_counts(counts)
    name = ARM_KERNEL_NAMES[arm]
    poles = _pole_parameter_block(values)
    body = _source_lines(values)

    if arm == "pml":
        prologue = _POLE_CHAIN_NOTE + _PML_ACCUMULATION_NOTE
        signature = (
            f'extern "C" __global__ void {name}(\n'
            "    float* __restrict__ Ex, float* __restrict__ Ey,\n"
            "    float* __restrict__ Ez,\n"
            "    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,\n"
            "    float* __restrict__ f_w_Ez,\n"
            "    const float* __restrict__ Dx, const float* __restrict__ Dy,\n"
            "    const float* __restrict__ Dz,\n"
            "    const float* inv_eps_Ex, const float* inv_eps_Ey,\n"
            "    const float* inv_eps_Ez,\n"
            f"{poles}"
            "    int nx, int ny, int nz,\n"
            "    const float* __restrict__ kps_x, const float* __restrict__ kms_x,\n"
            "    const float* __restrict__ kps_y, const float* __restrict__ kms_y,\n"
            "    const float* __restrict__ kps_z, const float* __restrict__ kms_z\n"
            ") {\n")
        index_block = (
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (idx >= nx * ny * nz) return;\n"
            "\n"
            "    int k = idx % nz;\n"
            "    int j = (idx / nz) % ny;\n"
            "    int i = idx / (ny * nz);\n"
            "\n")
        tail = (
            "\n\n"
            "    // All three sources are formed BEFORE any store, where the array\n"
            "    // path forms each inside its own loop iteration (stepping.py:\n"
            "    // 968-989). Exact, not merely close: this sub-step READS D, the\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 968-989->997-1018
            "    // pole buffers and inv_eps and WRITES E and f_w_E -- disjoint\n"
            "    // sets -- so no store can reach an operand of a later source. The\n"
            "    // array path's own shared `_fmp_scratch` (fields.py:1099) is\n"
            "    // rewritten per component and never read across them, so it does\n"
            "    // not carry a dependency either.\n"
            "    // Ex <- dsigw x.  Ey <- dsigw y.  Ez <- dsigw z. Component c reads\n"
            "    // AXIS c's table (E_CONSTITUTIVE_TERMS, stepping.py:227).\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 227->228
            "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);\n"
            "    constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);\n"
            "    constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);\n"
            "}\n")
        return prologue + signature + index_block + body + tail

    # arm 2: stepping.py:1019-1022. No auxiliary, no coefficient vector, no history,
    # and therefore no per-axis index -- the linear index is all the kernel needs.
    prologue = _POLE_CHAIN_NOTE + r'''
// stepping.update_E without an absorber (stepping.py:990-993):
//
//     # MEEP update_eh.cpp with dsigw == NO_DIRECTION: E = chi1inv * (D - P),
//     # written into storage rather than returned, because P advances next.
//     getattr(fields, component)[...] = constitutive
//
// A STORE, not an accumulation. There is no f_w to keep, no kps/kms to index and
// no previous value to read, which is why this is a second device string rather
// than a flag on the PML one: a kernel serving both behind a branch would be two
// kernels wearing one name.
//
// `stores_E` IS THE PRECONDITION, not `has_polarizations`. stepping.py:954
// returns before this branch when E is not stored, and Fields.stores_E
// (fields.py:999-1006) is what that line reads.
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 990-993->1019-1022, 954->983
    signature = (
        f'extern "C" __global__ void {name}(\n'
        "    float* __restrict__ Ex, float* __restrict__ Ey,\n"
        "    float* __restrict__ Ez,\n"
        "    const float* __restrict__ Dx, const float* __restrict__ Dy,\n"
        "    const float* __restrict__ Dz,\n"
        "    const float* inv_eps_Ex, const float* inv_eps_Ey,\n"
        "    const float* inv_eps_Ez,\n"
        f"{poles}"
        "    int n_elem\n"
        ") {\n")
    index_block = (
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx >= n_elem) return;\n"
        "\n")
    tail = (
        "\n\n"
        "    Ex[idx] = src_x;\n"
        "    Ey[idx] = src_y;\n"
        "    Ez[idx] = src_z;\n"
        "}\n")
    return prologue + signature + index_block + body + tail


def source_digest(arm: str, counts: Sequence[int]) -> str:
    """sha256 of one emitted body -- what a record pins instead of a file hash."""
    return hashlib.sha256(dispersive_source(arm, counts).encode("utf-8")).hexdigest()


def corpus_digest() -> str:
    """One digest over every arm at every swept arity.

    A file hash stops matching when a docstring gains a comma; this pins the
    strings NVRTC actually compiles, across the whole emitted corpus, so an edit
    to the emitter that changes any body fails on a laptop rather than on a
    device run hours later.
    """
    parts = [f"{arm}|{'-'.join(str(v) for v in counts)}|"
             f"{dispersive_source(arm, counts)}"
             for arm in DISPERSIVE_ARMS for counts in POLE_COUNTS_SWEPT]
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


# =============================================================================
# COMPILATION AND LAUNCH -- CuPy is imported HERE, never at module scope
# =============================================================================

def _cupy():
    """CuPy, imported on demand. See the header's last paragraph for why."""
    import cupy  # noqa: PLC0415
    return cupy


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with every sibling module -- the cache is keyed on the source
    string, so neither two modules nor two arities can collide. The probe drives
    this between guard sets, where the compiler ITSELF is substituted and the
    option tuple is overridden from outside, a change no memo key can see.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm: str, counts: Sequence[int]):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason every
    sibling rebuilds its code map per call: the source is part of the memo key,
    so it has to be built before there is a key to miss on, and a bit-identity
    probe mutates kernels by rewriting the emitted string through
    :data:`SOURCE_TRANSFORM`. A body memoized at first call would hand back the
    pre-mutation string forever -- a leg reporting a pass for a mutation it never
    applied.
    """
    cp = _cupy()
    values = normalized_pole_counts(counts)
    code = dispersive_source(arm, values)
    if SOURCE_TRANSFORM is not None:
        code = SOURCE_TRANSFORM(arm, values, code)
    func_name = ARM_KERNEL_NAMES[arm]
    key = compile_cache.kernel_cache_key(
        f"{func_name}:{'-'.join(str(v) for v in values)}", False,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, func_name, options=_COMPILE_OPTIONS))


#: THE GATE'S DOOR into the emitted text. ``None`` in every shipped path; a
#: bit-identity probe assigns a ``(arm, counts, source) -> source`` callable to
#: plant a defect in a body that does not exist until it is asked for. Sibling
#: families rewrite a module-level string instead; an emitter has none to rewrite,
#: so the seam is a hook rather than an assignment.
SOURCE_TRANSFORM: Optional[Any] = None


def dispersive_tables(pml: "Any") -> Dict[str, Any]:
    """The six HALF-INTEGER kps/kms views arm 1 reads, flattened once.

    HALF-INTEGER, NOT INTEGER, and not a caller's choice: ``update_E`` reads the
    half-integer sub-lattice (stepping.py:1015, ``half_integer=True``) because an
    E component is half-integer in its own direction. Swapped for the integer
    ones it is a half-cell error in the absorber profile -- converged, smooth and
    wrong -- which is why the gate carries ``swap_constitutive_sublattice`` as a
    host mutation rather than trusting the call sites.

    Called ONCE per frozen configuration, not per launch: ``PML._reshape_for_broadcast``
    already stores float32 arrays of shape (n,1,1)/(1,n,1)/(1,1,n), so
    ``reshape(-1)`` is a view and holding it costs nothing.
    """
    cp = _cupy()
    tables: Dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        for stem in ("kps", "kms"):
            attribute = f"{stem}_{axis}_h"
            flat = getattr(pml, attribute).reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; this kernel indexes float32 "
                    f"coefficient vectors.")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the "
                    f"kernel indexes it as a bare vector.")
            tables[f"{stem}_{axis}"] = flat
    return tables


def resolve_pole_plan(fields: "Any") -> Dict[str, List[Any]]:
    """``{'Ex': [P arrays, in order], ...}`` -- the chain, resolved RIGHT NOW.

    Transcribed from ``Fields.displacement_minus_polarization`` (fields.py:1096):
    ``[state for state in self.polarizations if state.drives(component)]``, then
    ``state.P[component]``.

    RESOLVED PER LAUNCH, NEVER CACHED. ``PolarizationState.update`` rotates the
    three buffers (dispersion.py:689-691), so a plan built before ``update_P``
    names last step's arrays after it. That is stale in a way that still
    computes -- the field stays finite and the spectrum moves -- which is the
    worst kind.
    """
    plan: Dict[str, List[Any]] = {}
    for target, _source, _axis in ELECTRIC_TERMS:
        chain: List[Any] = []
        for state in tuple(getattr(fields, "polarizations", ()) or ()):
            drives = getattr(state, "drives", None)
            if callable(drives) and drives(target):
                chain.append(state.P[target])
        plan[target] = chain
    return plan


def pole_counts_of(plan: Dict[str, Sequence[Any]]) -> Tuple[int, int, int]:
    return tuple(len(plan.get(target, ()))  # type: ignore[return-value]
                 for target, _source, _axis in ELECTRIC_TERMS)


def _require_no_aliasing(fields: "Any", plan: Dict[str, Sequence[Any]]) -> None:
    """Every ``__restrict__`` pointer distinct from every other one it may not alias.

    THE PREDICATE ALREADY CHECKED THIS and it is checked again at the launch, for
    the shipped ADE launcher's reason: the two answer about different moments.
    The predicate answers about the state as a planner saw it; the pole buffers
    rotate between steps, so what it certified is one permutation and this is the
    one being run.

    Equality of BASE addresses only -- two overlapping views with different bases
    pass unseen, the accepted limitation every launcher on this track shares. The
    inverse-epsilon volumes are EXCLUDED: they legitimately alias each other on
    an isotropic run and their parameters carry no ``restrict``.
    """
    seen: Dict[int, str] = {}
    names: List[Tuple[str, Any]] = []
    for target, source, _axis in ELECTRIC_TERMS:
        names.append((target, getattr(fields, target, None)))
        names.append((source, getattr(fields, source, None)))
        auxiliary = getattr(fields, "f_w_" + target, None)
        if auxiliary is not None:
            names.append(("f_w_" + target, auxiliary))
        for index, array in enumerate(plan.get(target, ())):
            names.append((f"P_{target}_{index}", array))
    for label, array in names:
        if array is None:
            continue
        address = _base_address(array)
        if address is None:
            raise ValueError(
                f"{label} exposes no readable base address; the restrict "
                f"promises in the device signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]}; every pointer in this "
                f"kernel's signature except the three inverse-epsilon volumes "
                f"carries __restrict__, and an alias there is undefined "
                f"behaviour the compiler is licensed to miscompile silently")
        seen[address] = label


def update_E_fused_pml_real_dispersive(fields: "Any", pml: "Any" = None, *,
                                       tables: Optional[Dict[str, Any]] = None,
                                       plan: Optional[Dict[str, List[Any]]] = None
                                       ) -> Dict[str, Any]:
    """Arm 1: ``stepping.update_E`` under an absorber, dispersive, in ONE launch.

    ``tables`` and ``plan`` are the GATE'S DOORS, keyword-only. Supply ``pml``
    and the half-integer sub-lattice is decided here by :func:`dispersive_tables`;
    supply ``tables`` and a harness can hand in deliberately mis-paired ones.
    ``plan`` defaults to :func:`resolve_pole_plan` -- a harness passes a reversed
    chain to arm the ordering defect, which is the one that actually threatens
    this kernel.

    Returns the arity it launched at, so a caller can record which body ran.
    """
    if tables is None:
        if pml is None:
            raise ValueError(
                "supply either pml (and the HALF-INTEGER sub-lattice is chosen "
                "here, stepping.py:1015) or tables; passing neither leaves the "
                "kernel with no absorber profile to index")
        tables = dispersive_tables(pml)
    if plan is None:
        plan = resolve_pole_plan(fields)
    counts = normalized_pole_counts(pole_counts_of(plan))
    _require_no_aliasing(fields, plan)

    nx, ny, nz = fields.Ex.shape
    blocks = ((nx * ny * nz + _DISPERSIVE_THREADS - 1) // _DISPERSIVE_THREADS)
    arguments: List[Any] = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
    ]
    for target, _source, _axis in ELECTRIC_TERMS:
        arguments.extend(plan[target])
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
    ])
    _get_kernel("pml", counts)((blocks,), (_DISPERSIVE_THREADS,), tuple(arguments))
    return {"arm": "pml", "counts": list(counts), "launches": 1}


def update_E_no_pml_real_dispersive(fields: "Any", pml: "Any" = None, *,
                                    plan: Optional[Dict[str, List[Any]]] = None
                                    ) -> Dict[str, Any]:
    """Arm 2: ``stepping.update_E`` with NO absorber and stored E, in ONE launch.

    ``pml`` IS ACCEPTED AND UNUSED, which is the array path's own signature:
    ``stepping.update_E`` takes it and, on this branch, never reads it past
    ``_pml_is_active``. It is taken here so every launcher on this track has the
    same shape and so the predicate a caller must consult -- which DOES need the
    layer, to know that it is inert -- takes the same arguments.
    """
    if plan is None:
        plan = resolve_pole_plan(fields)
    counts = normalized_pole_counts(pole_counts_of(plan))
    _require_no_aliasing(fields, plan)

    n_elem = int(fields.Ex.size)
    blocks = (n_elem + _DISPERSIVE_THREADS - 1) // _DISPERSIVE_THREADS
    arguments: List[Any] = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
    ]
    for target, _source, _axis in ELECTRIC_TERMS:
        arguments.extend(plan[target])
    arguments.append(np.int32(n_elem))
    _get_kernel("no_pml", counts)((blocks,), (_DISPERSIVE_THREADS,), tuple(arguments))
    return {"arm": "no_pml", "counts": list(counts), "launches": 1}


#: Launch one arm by name, for callers that hold the arm as data.
_LAUNCHERS = {"pml": update_E_fused_pml_real_dispersive,
              "no_pml": update_E_no_pml_real_dispersive}


def update_E_dispersive(arm: str, fields: "Any", pml: "Any" = None, **kwargs):
    """Run arm 1 (``arm='pml'``) or arm 2 (``'no_pml'``)."""
    if arm not in _LAUNCHERS:
        raise ValueError(f"arm must be one of {DISPERSIVE_ARMS}, got {arm!r}")
    return _LAUNCHERS[arm](fields, pml, **kwargs)


# =============================================================================
# THE PREDICATES
# =============================================================================

def _shared_dispersive_refusals(fields: Any, grid: Any) -> Any:
    """Every clause both update_E arms refuse on, in a fixed order, or None.

    ONE SPELLING, TWO ARMS. The arms differ in the absorber clause and in what
    storage they require; everything else -- the storage width, the coordinate
    system, the boundary kinds, the fold budget, the features that reach a
    DIFFERENT product -- is the same question with the same line behind it, and
    two copies would drift.
    """
    if getattr(fields, "force_complex_fields", False):
        # ``Fields._field_dtype`` (fields.py:571-573) reads this attribute alone,
        # so it is the whole storage question -- but every buffer is checked
        # float32 below regardless, because a predicate that inferred the dtype
        # from a flag would follow that flag if it ever stopped deciding.
        return "complex64 storage: the recurrence is the same but the storage is not"

    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return unreadable
    if facts["cylindrical"]:
        # REFUSED FAIL-CLOSED and the asymmetry with the sibling constitutive
        # predicate (which now ADMITS Dcyl, CONSTITUTIVE_CYLINDRICAL_ADMISSION)
        # is deliberate: that admission was measured on a device with the
        # refusal's own premise armed as a defect, and this family has never been
        # run on a Dcyl grid. The corpus drives ZERO dispersive Dcyl rows, so the
        # refusal costs nothing measured and the admission would rest on a
        # transfer between families.
        return ("cylindrical (Dcyl): element-wise like every other extent, and "
                "refused because this family has never been measured on one")
    for axis in range(3):
        if facts["axis"][axis] and not facts["cylindrical"]:
            return (f"axis {axis} reports the cylindrical r = 0 rule on a grid "
                    f"that does not report cylindrical coordinates; this "
                    f"predicate cannot answer for a grid that disagrees with "
                    f"itself")
        if facts["axis"][axis]:
            return f"axis {axis} is the cylindrical r = 0 axis"
    if facts["has_symmetry"] != any(facts["mirrored"]):
        return ("the grid reports has_symmetry() and no mirrored axis, or the "
                "reverse; the fold admission is decided per axis and cannot be "
                "read off a grid that disagrees with itself")
    folded = sum(1 for axis in range(3) if facts["mirrored"][axis])
    if folded > ADE_FOLD_PLANES_SWEPT:
        return (f"{folded} mirror planes at once: this family's gate scored "
                f"{ADE_FOLD_PLANES_SWEPT} simultaneous planes, so a "
                f"{folded}-plane fold would be admitted by argument alone")
    for axis, kind in enumerate(_boundary_kinds_from(facts)):
        # A FOLD IS ON THIS LIST. This sub-step reads no neighbour and has no
        # ghost rule, so the clause is not load-bearing arithmetic -- it stays so
        # that a ghost rule added to ``Grid`` later is refused because it was
        # never admitted.
        if kind not in CONSTITUTIVE_BOUNDARY_KINDS:
            return f"axis {axis} resolves to boundary {kind!r}, which has no kernel"
    if facts["has_bloch"]:
        return ("nonzero Bloch k: the wrapped plane carries a phase real storage "
                "cannot hold")
    if facts["bfast_active"]:
        return "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    # BOTH maps by name, never ``Fields.has_nonlinearity`` (fields.py:966-967),
    # which reads ``_chi2_components`` ALONE -- a chi3-only run answers False
    # through the property. Same spelling as every sibling predicate.
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return ("instantaneous chi2/chi3: a Pade factor REPLACES the "
                "constitutive product (stepping.py:999-1000) rather than scaling "
                "this arm's source")
    if getattr(fields, "has_offdiagonal_epsilon", False):
        # THE ONE SLOT THIS FAMILY DELIBERATELY LEAVES UNSERVED, named in the
        # header: absorbed_power_density.py is off-diagonal AND dispersive AND
        # folded. The row product reads the OTHER components' D - sum P volumes
        # at neighbouring cells (stepping.py:1001-1008, _offdiagonal_terms :1196),
        # so the sub-step gains a ghost rule and stops being element-wise.
        return ("off-diagonal chi1inv: the row product reads the other "
                "components' (D - sum P) volumes at neighbouring cells and this "
                "arm is element-wise")

    shape = facts["shape"]
    if len(shape) != 3:
        return f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        # Both arms index with ``int``. Past 2**31 elements that stops being an
        # arithmetic identity and starts being a wrong answer at a wrapped index.
        return f"{cells} cells exceeds the kernel's int32 index range"
    return None


def _pole_chain_problem(fields: Any, xp: Any, shape: Tuple) -> Any:
    """Why the ``D - sum P`` chain is not one this arm can bind, or None.

    THE CHAIN IS READ THE WAY THE ARRAY PATH READS IT -- ``state.drives(c)``
    filtered over ``fields.polarizations`` in order -- because the ORDER is the
    arithmetic here, not merely the membership.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    counts: List[int] = []
    addresses: Dict[int, str] = {}
    for target, _source, _axis in ELECTRIC_TERMS:
        chain = 0
        for index, state in enumerate(states):
            drives = getattr(state, "drives", None)
            try:
                if not (callable(drives) and drives(target)):
                    continue
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
                return f"polarization {index}.drives({target!r}) raised {exc!r}"
            kind = getattr(getattr(state, "susceptibility", None), "kind", None)
            if kind not in COVERED_SUSCEPTIBILITY_KINDS:
                # NOT this arm's arithmetic but its INPUT: a kind outside the
                # Lorentz/Drude pair advances P by a different difference
                # equation, and a P this kernel subtracted would be a number no
                # transcription in this package produced.
                return (f"polarization {index} drives {target} with kind "
                        f"{kind!r}, outside {COVERED_SUSCEPTIBILITY_KINDS}")
            array = (getattr(state, "P", {}) or {}).get(target)
            problem = _array_problem(f"polarization {index}.P[{target!r}]",
                                     array, xp, shape)
            if problem is not None:
                return problem
            address = _base_address(array)
            if address is None:
                return (f"polarization {index}.P[{target!r}] exposes no readable "
                        f"base address; the restrict promise cannot be shown to hold")
            label = f"P_{target}_{chain}"
            if address in addresses:
                return (f"{label} aliases {addresses[address]}; the pole "
                        f"pointers carry __restrict__ and two susceptibilities "
                        f"never share a P buffer (dispersion.py:645-647)")
            addresses[address] = label
            chain += 1
        if chain > POLE_COUNT_CAP:
            return (f"{target} has {chain} contributors and POLE_COUNT_CAP is "
                    f"{POLE_COUNT_CAP}; the gate swept 0..{POLE_COUNT_CAP} and "
                    f"a longer chain has never been run")
        counts.append(chain)
    # A CHAIN OF ZERO ON EVERY COMPONENT IS ADMITTED, not refused, and that is a
    # DISJOINTNESS decision rather than a generosity: the sibling
    # ``covers_real_pml_constitutive`` refuses on ``fields.polarizations`` being
    # truthy, whether or not any state drives anything, so a run carrying a
    # trivial-sigma susceptibility would otherwise be refused by both families
    # and show as an unserved slot nobody owns. The emitted body at arity
    # (0,0,0) is the certified non-dispersive one.
    return None


def covers_real_pml_dispersive_constitutive(fields: Any, pml: Any,
                                            grid: Any) -> tuple:
    """Whether ``update_E_pml_real_dispersive`` may serve this run's ``update_E``.

    Returns ``(covered, reason)``; ``reason`` names the FIRST refusal, the
    convention every predicate in this package uses.

    THE SUB-STEP: ``stepping.update_E``'s diagonal dispersive branch under an
    active layer -- source ``(D - sum P) * inv_eps`` (stepping.py:1010-1013) into
    ``_apply_constitutive_pml`` (:1016, transcribed at :2112-2145).

    DISJOINT FROM EVERY SIBLING BY CONSTRUCTION, and the census checks it rather
    than trusting this note:

    * ``covers_real_pml_constitutive(side='E')`` refuses ``fields.polarizations``
      truthy; this REQUIRES it;
    * ``covers_no_pml_null_constitutive(side='E')`` and
      :func:`covers_no_pml_dispersive_constitutive` require an inert layer; this
      requires an active one;
    * ``covers_real_pml_offdiag_constitutive`` requires an off-diagonal row; this
      refuses one;
    * the complex, nonlinear, special_kz and BFAST arms each require a feature
      this refuses by name.
    """
    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if not (pml is not None and getattr(pml, "is_active", False)):
        # Without an absorber ``update_E`` takes the plain assignment at :993,
        # which is a different device string. Arm 2 is that one.
        return False, ("no active PML layer: update_E takes the plain assignment "
                       "at stepping.py:1022, which is "
                       "covers_no_pml_dispersive_constitutive's arm")
    if not tuple(getattr(fields, "polarizations", ()) or ()):
        # THE DISJOINTNESS CLAUSE, and it fires first among the feature clauses:
        # a run with no registered susceptibility is the certified
        # ``covers_real_pml_constitutive`` family's, and two arms admitting one
        # slot is a widening past the evidence, not extra coverage.
        return False, ("no susceptibility is registered: update_E's source is D "
                       "itself and the run belongs to "
                       "coverage.covers_real_pml_constitutive(side='E')")
    if not getattr(fields, "stores_E", False):
        # BELT AND BRACES WITH A REASON: an active PML already forces stored E,
        # so an edit that ever makes it optional under a layer has to be caught
        # here rather than reaching a kernel that writes an array nobody reads.
        return False, "E is recomputed from D rather than stored"

    refusal = _shared_dispersive_refusals(fields, grid)
    if refusal is not None:
        return False, refusal
    facts, _ = _grid_facts(grid)
    shape = facts["shape"]

    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "Dx", "Dy", "Dz"):
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    problem = _inverse_epsilon_problem(fields, xp, shape)
    if problem is not None:
        return False, problem
    problem = _pole_chain_problem(fields, xp, shape)
    if problem is not None:
        return False, problem

    # THIS SIDE'S OWN SUB-LATTICE ONLY -- half-integer, stepping.py:1015. Binding
    # the integer one is a half-cell error in the absorber profile.
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}_h"
            problem = _coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


def covers_no_pml_dispersive_constitutive(fields: Any, pml: Any,
                                          grid: Any) -> tuple:
    """Whether ``update_E_no_pml_real_dispersive`` may serve this run's ``update_E``.

    Returns ``(covered, reason)``; ``reason`` names the FIRST refusal.

    THE SUB-STEP, and it is NOT a special case of arm 1's. ``stepping.update_E``
    with ``_pml_is_active`` False and ``fields.stores_E`` True runs
    stepping.py:1019-1022::

        # MEEP update_eh.cpp with dsigw == NO_DIRECTION: E = chi1inv * (D - P),
        # written into storage rather than returned, because P advances next.
        getattr(fields, component)[...] = constitutive

    A STORE. No auxiliary, no coefficient vector, no history, no per-axis index --
    so no kps/kms clause below, and no ``f_w`` array to check.

    THE CLAUSE THIS REPLACES is ``covers_no_pml_null_constitutive``'s::

        "fields.stores_E is True: update_E writes E[...] = (D - sum P) * inv_eps
         (stepping.py:1022) instead of returning at :983 -- that is the STORE arm,
         which this package has not built (STORED_E_ARM_STATUS)"

    This is that arm, built. The null family stays exactly as it is: it claims
    the slot where ``update_E`` RETURNS, and this one claims the slot where it
    STORES. The two are complementary halves of ``stepping.py:983`` and the
    census checks that no slot is admitted by both.

    ``stores_E`` WITHOUT A POLARIZATION IS REFUSED HERE, and that refusal is a
    real gap rather than a disjointness artifact: ``Fields.enable_field_storage``
    can be switched on by an off-diagonal or nonlinear configuration too, and
    those reach a different product. A run with stored E, no layer, no
    susceptibility and no other feature would be ``E = D * inv_eps`` -- the
    certified non-dispersive body with the store tail -- and nothing on this
    track serves it. The corpus drives zero such rows.
    """
    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if pml is not None:
        active = getattr(pml, "is_active", None)
        if active is None:
            return False, ("the layer does not report is_active: "
                           "stepping._pml_is_active (stepping.py:2498-2506) "
                           "branches on exactly that attribute")
        if active:
            return False, ("an active PML layer is installed: update_E runs the "
                           "dsigw accumulation (stepping.py:1014-1018), which is "
                           "covers_real_pml_dispersive_constitutive's arm")
    if bool(getattr(fields, "_pml_active", False)):
        # CONSERVATIVE, and labelled so. ``Fields`` in PML storage mode behind an
        # inert layer means ``drive_field`` would hand ``f_w`` to update_P while
        # this arm wrote the constitutive product straight into E -- the two
        # disagree and nothing would say which is meant.
        return False, ("Fields has PML storage enabled while the layer is inert: "
                       "drive_field would hand f_w to update_P (fields.py:1160-1162) "
                       "while this arm writes the product into E")
    if not getattr(fields, "stores_E", False):
        # stepping.py:983, literally. Without stored E the sub-step RETURNS and
        # the slot belongs to ``covers_no_pml_null_constitutive(side='E')``.
        return False, ("fields.stores_E is False: update_E returns at "
                       "stepping.py:983 and the slot is "
                       "no_pml_constitutive.covers_no_pml_null_constitutive's")
    if not tuple(getattr(fields, "polarizations", ()) or ()):
        return False, ("no susceptibility is registered: stored E without a "
                       "layer and without dispersion is E = D * inv_eps with a "
                       "store tail, which nothing on this track has built")

    refusal = _shared_dispersive_refusals(fields, grid)
    if refusal is not None:
        return False, refusal
    facts, _ = _grid_facts(grid)
    shape = facts["shape"]

    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz"):
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    problem = _inverse_epsilon_problem(fields, xp, shape)
    if problem is not None:
        return False, problem
    problem = _pole_chain_problem(fields, xp, shape)
    if problem is not None:
        return False, problem
    return True, "covered"


def _inverse_epsilon_problem(fields: Any, xp: Any, shape: Tuple) -> Any:
    """THREE volumes, one per component, or why not.

    They may all alias one array (``set_isotropic_epsilon_volume``,
    fields.py:1321-1326) and the kernel binds three pointers either way. What is
    NOT covered is a scalar, and what must never happen is binding
    ``fields.inv_eps`` for all three -- that attribute is the Ez view
    (fields.py:1259-1260), which is what ``update_E_pml_complex`` gets
    wrong and what the gate arms as ``bind_Ez_inv_eps_for_all_three``.
    """
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return "fields does not expose inverse_epsilon_for"
    for target, _source, _axis in ELECTRIC_TERMS:
        try:
            volume = reader(target)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            # ``inverse_epsilon_for`` raises ValueError on a component it has no
            # volume for (fields.py:1341-1344). An exception escaping a predicate
            # a planner is reading is a crashed run where fail-closed was right.
            return f"inverse_epsilon_for({target!r}) raised {exc!r}"
        if volume is None:
            return f"inverse_epsilon_for({target!r}) is None"
        if not getattr(volume, "shape", ()):
            return f"inverse_epsilon_for({target!r}) is a scalar, not a volume"
        problem = _array_problem(f"inverse_epsilon_for({target!r})", volume,
                                 xp, shape)
        if problem is not None:
            return problem
    return None


# ---------------------------------------------------------------------------
# ARM 3 -- the ADE predicate widening. NO NEW KERNEL.
# ---------------------------------------------------------------------------

def _no_pml_ade_drive_problem(fields: Any, component: str) -> Any:
    """Why ``drive_field(component)`` is not the STORED E this arm binds.

    THE MIRROR IMAGE of ``coverage._ade_drive_problem``, and the asymmetry is the
    whole content of arm 3. Under an active layer the drive is ``f_w_<c>``;
    without one it is the stored E (fields.py:1140-1163). The two agree EXACTLY
    outside the absorber, so a predicate that checked only "some array exists"
    would admit a ``Fields`` whose storage mode disagreed with its layer -- and
    that is the state that produces the wrong pointer.

    So the check is an IDENTITY check on both halves: ``_pml_active`` must be
    False, and the array ``drive_field`` actually hands back must BE the stored
    component. A run in PML storage mode behind an inert layer fails the first
    and is refused by name.
    """
    if bool(getattr(fields, "_pml_active", False)):
        return ("Fields is in PML storage mode while the layer is inert; "
                "drive_field would hand back f_w_" + component + ", which is "
                "the constitutive product only INSIDE the absorber")
    expected = getattr(fields, component, None)
    if expected is None:
        return (f"{component} is not allocated; without a layer the drive IS the "
                f"stored E (fields.py:1162) and there is nothing to bind")
    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        return "fields does not expose drive_field()"
    try:
        actual = reader(component)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"drive_field({component!r}) raised {exc!r}"
    if actual is not expected:
        return (f"drive_field({component!r}) is not the stored {component}; the "
                f"launcher binds what drive_field returns and this run would "
                f"drive P from the wrong array")
    return None


def covers_no_pml_ade_component(fields: Any, pml: Any, grid: Any,
                                state: Any, component: str) -> tuple:
    """Whether the SHIPPED ``update_P_pml_real*`` may advance one (state, component)
    on a run with NO absorber.

    NO NEW KERNEL IS INVOLVED AND THAT IS THE MEASURED FINDING, not an
    assumption. ``ade_kernels.update_P_fused_pml_real`` binds ``drive(component)``
    where ``drive`` defaults to ``fields.drive_field``; ``stepping.update_P``
    passes the same bound method (stepping.py:1428) and deletes its ``pml``
    argument (:1424). So the device text, the coefficient triple, the rotation
    and the sigma specialization are all unchanged, and the only thing an
    absorber decides is which array the bound method returns.
    ``gate_cuda_dispersive.py``'s ``arm3`` leg runs the shipped launcher against
    ``stepping.update_P`` on layerless runs and carries a wrong-drive control
    that must DIVERGE.

    Every other clause is ``coverage.covers_real_pml_ade_component``'s, restated
    rather than delegated to: delegating would mean calling a function whose
    SECOND clause is "no active PML layer", which is precisely the one being
    inverted, and a wrapper that had to skip past a refusal would be reading the
    sibling's clause ORDER as an implementation detail. The order is the
    contract.
    """
    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, ("complex64 storage: the recurrence is the same but the "
                       "storage is not")
    if pml is not None and getattr(pml, "is_active", False):
        return False, ("an active PML layer is installed: the drive is f_w and "
                       "the run belongs to coverage.covers_real_pml_ade_update_p")

    if component not in ADE_ELECTRIC_COMPONENTS:
        return False, f"component {component!r} is outside {ADE_ELECTRIC_COMPONENTS}"
    drives = getattr(state, "drives", None)
    try:
        if not (callable(drives) and drives(component)):
            return False, f"this susceptibility does not drive {component}"
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"drives({component!r}) raised {exc!r}"
    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in COVERED_SUSCEPTIBILITY_KINDS:
        return False, f"kind {kind!r} is outside {COVERED_SUSCEPTIBILITY_KINDS}"
    problem = _ade_coefficient_problem(state)
    if problem is not None:
        return False, problem

    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["cylindrical"]:
        return False, ("cylindrical (Dcyl): element-wise like every other extent, "
                       "and refused because no hand-CUDA family has been measured "
                       "on one at this sub-step")
    for axis in range(3):
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    for axis, boundary in enumerate(_boundary_kinds_from(facts)):
        if boundary not in ADE_BOUNDARY_KINDS:
            return False, (f"axis {axis} resolves to boundary {boundary!r}, which "
                           f"this sub-step has never been measured on")
    if facts["has_bloch"]:
        return False, ("nonzero Bloch k: the wrapped plane carries a phase real "
                       "storage cannot hold")
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return False, ("instantaneous chi2/chi3: a Pade factor replaces the "
                       "constitutive product this sub-step's drive comes from")

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"

    buffers = {
        f"P[{component!r}]": (getattr(state, "P", {}) or {}).get(component),
        f"P_prev[{component!r}]": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for label, array in buffers.items():
        problem = _array_problem(label, array, xp, shape)
        if problem is not None:
            return False, problem
    addresses: Dict[int, str] = {}
    for label, array in buffers.items():
        address = _base_address(array)
        if address is None:
            return False, (f"{label} exposes no readable base address; the "
                           f"rotation cannot be shown to be alias-free")
        if address in addresses:
            return False, (f"{label} aliases {addresses[address]}; the recurrence "
                           f"reads P and P_prev while writing the scratch, and the "
                           f"rotation that follows would advance one buffer twice")
        addresses[address] = label

    problem = _no_pml_ade_drive_problem(fields, component)
    if problem is not None:
        return False, problem
    drive = getattr(fields, component, None)
    problem = _array_problem(component, drive, xp, shape)
    if problem is not None:
        return False, problem
    drive_address = _base_address(drive)
    if drive_address is not None and drive_address in addresses:
        return False, (f"the stored {component} aliases {addresses[drive_address]}; "
                       f"the drive is read while the scratch is written")

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        return False, f"sigma[{component!r}] is missing"
    if ade_sigma_is_volume(state, component):
        problem = _array_problem(f"sigma[{component!r}]", sigma, xp, shape)
        if problem is not None:
            return False, problem
        sigma_address = _base_address(sigma)
        if sigma_address is not None and sigma_address in addresses:
            return False, (f"sigma[{component!r}] aliases {addresses[sigma_address]}; "
                           f"the coefficient volume is read while the scratch is "
                           f"written")
    else:
        try:
            number = float(sigma)
        except Exception:  # noqa: BLE001 - neither a scalar nor a volume
            return False, (f"sigma[{component!r}]={sigma!r} is neither a scalar "
                           f"nor a volume")
        if number != number or number in (float("inf"), float("-inf")):
            return False, f"sigma[{component!r}]={number!r} is not finite"
    return True, "covered"


def covers_no_pml_ade_update_p(fields: Any, pml: Any, grid: Any) -> tuple:
    """Whether the hand-CUDA ADE family may serve the WHOLE ``update_P`` sub-step
    on a run with no absorber.

    ALL OR NOTHING OVER EVERY STATE AND EVERY DRIVEN COMPONENT, for the sibling
    predicate's reason and not out of conservatism: ``PolarizationState.update``
    rotates ONE SHARED SCRATCH from component to component inside a single call
    (dispersion.py:689-691), so a sub-step half on the kernel and half on the
    array path would hand the array path a buffer the kernel had already
    retired. ``update_P`` is ONE slot in the 759-slot denominator and a family
    that serves it has to serve all of it.

    A RUN WITH NO DRIVEN COMPONENT AT ALL IS REFUSED, not admitted as a
    vacuous pass: ``stepping.update_P`` still iterates and still does nothing,
    but a family claiming a slot on which it would launch zero kernels is
    claiming coverage of a sub-step it never touched -- and the census would
    count it. The null-arm question ("this sub-step is a no-op here") is
    ``no_pml_constitutive.py``'s shape and has no ADE counterpart today.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        return False, "no susceptibility is registered: there is no update_P to serve"
    launches = 0
    for index, state in enumerate(states):
        driven = getattr(state, "driven", None)
        try:
            components = tuple(driven()) if callable(driven) else ()
        except Exception as exc:  # noqa: BLE001
            return False, f"polarization {index}.driven() raised {exc!r}"
        for component in components:
            covered, reason = covers_no_pml_ade_component(
                fields, pml, grid, state, component)
            if not covered:
                return False, f"polarization {index} {component}: {reason}"
            launches += 1
    if launches == 0:
        return False, ("every registered susceptibility has a trivial sigma and "
                       "drives nothing; this family would launch no kernel and "
                       "cannot claim the slot")
    return True, "covered"
