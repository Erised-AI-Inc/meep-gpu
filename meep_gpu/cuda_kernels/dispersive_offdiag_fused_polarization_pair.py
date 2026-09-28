"""The E->P weld the ROTATION HAZARD refused: monolithic ``update_E``, one component's P.

THE CELL, AND THE REFUSAL THAT NAMED IT. ``E_to_P (cuda_dispersive_offdiag/dispersive
off-diagonal, cuda_ade/ADE)`` is one seam-instance -- ``examples/absorbed_power_
density.py`` -- and the hand-CUDA board prices it POINTWISE-BUILDABLE and serves it
with nothing. It is the only ``E_to_P`` cell on the board with no product, and the
reason is written down in the shipped sibling
(:mod:`.fused_polarization_pair`'s preamble, verbatim):

    "The OFF-DIAGONAL dispersive ``update_E`` forms a row product that reads the
     OTHER components' ``D - sum P`` volumes at neighbouring cells. Split per
     component, launch y would read ``P[x]`` AFTER launch x's rotation moved the
     advanced buffer into that slot -- the driver order reads the OLD one, so the
     weld would diverge, and diverge in a way that still computes. ... Serving it
     would take a different weld shape (all three components in one launch, which
     the buffer count above refuses), not a wider predicate."

THAT IS A REFUSAL OF THE PER-COMPONENT SPLIT, NOT OF FUSION, and the fusion-residue
audit (§1.5) says so. This module is the shape it names: ``update_E`` stays
MONOLITHIC -- one launch, all three components, exactly as the certified
``update_E_pml_real_folded_offdiag_dispersive`` already is -- and the launch
additionally carries the ``update_P`` recurrence of ONE component across every state
that drives it. No rotation happens inside the launch, so no read can land on a
rotated slot, and the hazard has nothing to attach to.

=============================================================================
WHY THE BUFFER COUNT DOES NOT REFUSE THIS ONE
=============================================================================

The sibling's buffer argument is exact and it is about ``update_P``, not
``update_E``: "advancing all three components of one state at once needs three
outputs distinct from six live inputs, and a state owns seven buffers". This weld
advances ONE component per launch, so it needs ONE output (that state's
``_scratch``) distinct from its two live inputs (``P[c]``, ``P_prev[c]``) -- which is
the certified ADE kernel's own arrangement, unchanged. The other two components run
afterwards as certified ADE launches with the host rotation between, exactly as
``ade_kernels.update_P_fused_pml_real`` runs them.

WHAT IT IS WORTH, COUNTED RATHER THAN CLAIMED. On this cell's one corpus row the
census records ONE polarization state driving all three components, so the array path
runs ``1 + 3 = 4`` launches on this seam and this weld runs ``1 + 2 = 3``. That is the
whole saving, and it is the same saving the per-component sibling would have got on
this configuration (three launches either way) for a far smaller edit to certified
text -- the off-diagonal body is lifted WHOLE rather than split three ways.

=============================================================================
THE SEAM, AND WHY THE HAND-OFF IS EXACT
=============================================================================

``update_P``'s drive for component ``c`` is ``fields.drive_field(c)``, which under an
active layer is ``f_w_<c>`` (fields.py:1140-1163) -- and that is the very word this
launch's ``constitutive_apply`` just stored from its ``src`` register
(``fw[idx] = src``). A float32 word stored to global and reloaded is the identity on
the bits, so the register IS the drive; the load is removed and the register used, the
same lift :mod:`.fused_polarization_pair` declares. The predicate checks the drive
identity against the arrays this launch binds rather than assuming it.

THE POLE BUFFER IS THE ALIASING HAZARD HERE, and it is the same one the sibling
records: the E half's ``dmp_<c>`` chain reads ``state.P[c]`` and the P half's
recurrence reads THE SAME ARRAY as its ``p_now``. Binding one allocation to two
``__restrict__`` parameters is undefined behaviour NVRTC miscompiles without a
diagnostic, so each ``P_<c>_<i>`` appears EXACTLY ONCE in the signature -- it is
already there for the E half -- and the recurrence reads through that parameter. The
certified ADE kernel's ``p_now`` is therefore not a parameter of this kernel at all.

=============================================================================
WHAT IS LIFTED AND WHAT IS WRITTEN
=============================================================================

Nothing arithmetic is written here. The E half is
``dispersive_offdiag_update_e.dispersive_offdiag_source`` -- prelude, signature and
body -- and the P half is ``fused_polarization_pair._pole_block``, which is itself
``ade_kernels``' certified recurrence under checked renames. This module supplies the
SIGNATURE SPLICE (the ADE parameters appended to the certified one), the seam line,
and the launcher's choreography. :data:`LIFT_EDITS` is the whole list.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coverage as _coverage
from . import dispersive_offdiag_update_e as _offdiag_e
from . import fused_polarization_pair as _polar
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import ade_sigma_is_volume, covers_real_pml_ade_update_p
from .dispersive_offdiag_update_e import (
    covers_real_pml_dispersive_offdiag_constitutive,
)

try:
    from . import ade_kernels
except Exception:  # noqa: BLE001 - device-only; the predicate still answers
    ade_kernels = None  # type: ignore[assignment]
try:
    from . import dispersive_kernels
except Exception:  # noqa: BLE001
    dispersive_kernels = None  # type: ignore[assignment]

#: BYTE-IDENTICAL TO THE ARRAY PATH WITH A DEVICE VERDICT BEHIND IT.
#: ``certification.json``'s ``cuda_dispersive_offdiag_polarization_pair_2026-09-02`` block names the same
#: two artifacts this line rests on, and ``test_kernel_partition.py`` refuses a
#: name here that no record block claims.
#:
#: THE DECLARATION IS THE FINAL BYTES, AND THAT IS THIS CAMPAIGN'S RULE 4 RATHER
#: THAN a claim made ahead of its evidence: a gate certifies the module it
#: imported, so a module that moved its own name between the two sets AFTER the
#: run would leave the record bound to bytes that no longer ship -- which is
#: exactly the drift ``rebind_cuda_welds.py`` exists to catch. The run this line
#: rests on was cut against these bytes; the earlier pass that measured the
#: design is not what the record binds.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "dispersive_offdiag_fused_polarization_pair_pml_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: FALSE, and by DRIVER CONSTRUCTION rather than by measurement. The driver advances
#: the polarizations immediately after ``update_E`` (driver.py:3313 then :3315) with
#: NOTHING between the two consults -- no injection, no fill, no wall pass -- so
#: ``fused_pairs.FUSED_PAIR_SEAMS`` carries ``None`` for this seam and
#: ``_install_fused_pair`` reads it as unconditionally empty. There is no deposit list
#: for this flag to select, and declaring ``True`` would ask ``_in_seam_indexed`` for
#: one of the two lists the driver injects OUTSIDE this seam.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_dispersive_offdiag_fused_polarization_pair"

KERNEL_NAME = "dispersive_offdiag_fused_polarization_pair_pml_real"

#: The two driver consults ONE launch of this kernel spans, in driver order
#: (driver.py:3313, :3315). The launch performs all of ``update_E`` and ONE
#: component's ``update_P``; the remaining components run as certified ADE launches
#: with the host rotation between, which is what the composition reports.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

SLOT = "update_E"

#: ``fused_polarization_pair``'s own component table, imported by value so the two
#: welds cannot disagree about which component owns which axis.
ELECTRIC_TERMS = _polar.ELECTRIC_TERMS

_FUSED_THREADS = 256
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: The certified off-diagonal dispersive signature's closing line, which this weld
#: splices the ADE parameters in front of.
_SIGNATURE_TAIL = "    float gw_x, float gw_y, float gw_z\n) {\n"

LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float w = drive[idx];",
     "became": "    float w = src_E<c>;",
     "why": "THE SEAM. update_P's drive is fields.drive_field(c), which under an "
            "active layer is f_w_<c> -- the very word the update_E half of this "
            "same thread just stored from its src register (constitutive_apply "
            "stores fw[idx] = src). A float32 word stored to global and reloaded is "
            "the identity on the bits. Emitted ONCE per component rather than once "
            "per pole, because every pole reads the same array and nothing in this "
            "seam writes it between the certified launches."},
    {"line": "    const float* __restrict__ p_now,",
     "became": "(not a parameter: the pole pointer P_<c>_<i> is bound once)",
     "why": "the E half's dmp_<c> chain reads state.P[c] and the recurrence's p_now "
            "IS state.P[c] -- one allocation. Binding it twice with __restrict__ on "
            "both is undefined behaviour, so the signature binds it once (the E "
            "half already declares it) and both halves read through that parameter."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= n_elem) return;   (the ADE kernel's own guard)",
     "became": "(dropped: the E half's guard has already returned)",
     "why": "the update_E body above declares idx and guarded the extent; splicing "
            "the ADE guard would redeclare idx. n_elem is therefore not a parameter "
            "at all, and the predicate requires every P/P_prev/scratch/sigma volume "
            "to carry exactly the stored extent the E half walks, so the dropped "
            "bound is the same number the certified kernel would have used."},
    {"line": 'extern "C" __global__ void '
             "update_E_pml_real_folded_offdiag_dispersive(",
     "became": 'extern "C" __global__ void '
               "dispersive_offdiag_fused_polarization_pair_pml_real(",
     "why": "THE ENTRY POINT'S NAME, and nothing else on that line. A second kernel "
            "wearing the certified one's name would make certification.json's "
            "partition test and any NVRTC binary observation ambiguous about which "
            "body it saw -- the same reason folded_offdiag_kernels states for its "
            "own name."},
    {"line": "    float gw_x, float gw_y, float gw_z\n) {",
     "became": "    float gw_x, float gw_y, float gw_z,\n<the ADE parameters>\n) {",
     "why": "THE SIGNATURE SPLICE, and the only device text this module writes. The "
            "certified off-diagonal signature is lifted whole and the recurrence's "
            "outputs, histories, sigmas and scalar coefficients are appended -- one "
            "group per driving state. No arithmetic lives in a signature."},
    {"line": "(the certified update_P, one launch per (state, component))",
     "became": "(ONE component's recurrence inside the update_E launch; the rest "
               "as certified ADE launches with the host rotation between)",
     "why": "THE SHAPE THE SIBLING'S REFUSAL NAMES. update_E stays MONOLITHIC, so "
            "no rotation happens inside it and its off-diagonal row product cannot "
            "read a rotated P slot -- which is the whole hazard. The buffer count "
            "that refuses advancing three components at once does not arise: one "
            "component needs one output distinct from two live inputs, which is the "
            "certified ADE kernel's own arrangement."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "ELECTRIC_TERMS", "FAMILY",
    "KERNEL_NAME", "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "covers_dispersive_offdiag_fused_polarization_pair",
    "device_sources", "fused_component", "kernel_source",
    "launch_dispersive_offdiag_fused_polarization_pair", "pole_counts",
    "run_dispersive_offdiag_fused_polarization_pair",
]


# =============================================================================
# THE DEVICE CODE
# =============================================================================

def pole_counts(fields: Any) -> Tuple[int, int, int]:
    """How many states drive each component, in ``ELECTRIC_TERMS`` order.

    The certified off-diagonal dispersive emitter's own arity axis, read through
    ``fused_polarization_pair.component_specs`` so the two welds cannot disagree
    about which states drive what.
    """
    specs = _polar.component_specs(fields)
    return tuple(  # type: ignore[return-value]
        len(specs[target]["states"]) for target, _source, _axis in ELECTRIC_TERMS)


def fused_component(fields: Any) -> Optional[str]:
    """Which component's recurrence rides inside the launch: the FIRST driven one.

    THE ORDER IS THE CERTIFIED LAUNCHER'S. ``ade_kernels.update_P_fused_pml_real``
    visits each state's components in ``state.driven()`` order and rotates between
    them, and this weld preserves that sequence by taking the first component any
    state drives and leaving the rest to run after it. Rotations of DIFFERENT states
    commute (each owns its seven buffers); within a state the order is what decides
    which allocation ends up holding which slot, so it is kept rather than chosen.

    ``None`` where nothing is driven -- a configuration the predicate refuses,
    because a weld with no recurrence to carry is the certified update_E alone.
    """
    specs = _polar.component_specs(fields)
    for target, _source, _axis in ELECTRIC_TERMS:
        if specs[target]["states"]:
            return target
    return None


def _ade_parameters(component_index: int, count: int,
                    sigma_kinds: Sequence[bool]) -> str:
    """The recurrence's own parameters, one group per driving state.

    ``p_now`` is ABSENT and that is the aliasing decision, not an omission: the E
    half already binds ``P_<c>_<i>`` and both halves read it through that one
    parameter.
    """
    lines: List[str] = []
    for i in range(count):
        sigma = (f"const float* __restrict__ sigma_{i}," if sigma_kinds[i]
                 else f"float sigma_{i},")
        lines.append(f"    float* __restrict__ p_out_{i}, "
                     f"const float* __restrict__ p_prev_{i}, {sigma}")
        lines.append(f"    float c_now_{i}, float c_prev_{i}, float c_drive_{i},")
    # THE LAST GROUP CLOSES THE LIST. The certified signature this splices into
    # ended on its own final parameter, so a trailing comma here is a parameter
    # declarator NVRTC expects and never finds.
    return "\n".join(lines).rstrip(",")


def kernel_source(row_mask: Sequence[int], counts: Sequence[int],
                  component_index: int, sigma_kinds: Sequence[bool]) -> str:
    """The whole fused kernel: the certified off-diagonal update_E plus one recurrence.

    The E half -- prelude, signature and body -- is
    ``dispersive_offdiag_update_e.dispersive_offdiag_source`` VERBATIM apart from the
    signature splice; the P half is ``fused_polarization_pair._pole_block``, itself
    ``ade_kernels``' certified recurrence under checked renames. A certified string
    that drifts raises rather than emitting a kernel that is quietly not the certified
    arithmetic.
    """
    target = ELECTRIC_TERMS[component_index][0]
    count = len(tuple(sigma_kinds))
    if count < 1:
        raise ValueError(
            f"{target} is driven by no state, so there is no recurrence to weld; "
            f"that configuration is the certified update_E alone")
    if int(counts[component_index]) != count:
        raise ValueError(
            f"the emitter was asked for {count} pole groups on {target} and the "
            f"arity triple says {counts[component_index]}; the E half's chain and "
            f"the P half's recurrence would read different numbers of poles")
    certified = _offdiag_e.dispersive_offdiag_source(row_mask, counts)
    if certified.count(_SIGNATURE_TAIL) != 1:
        raise AssertionError(
            f"the certified off-diagonal dispersive signature no longer closes on "
            f"{_SIGNATURE_TAIL!r} ({certified.count(_SIGNATURE_TAIL)} matches); this "
            f"weld splices the recurrence's parameters in front of that line and "
            f"cannot find where they go")
    entry = ('extern "C" __global__ void '
             f"{_offdiag_e.KERNEL_NAME}(\n")
    if certified.count(entry) != 1:
        raise AssertionError(
            f"the certified off-diagonal dispersive source declares "
            f"{certified.count(entry)} entry points named "
            f"{_offdiag_e.KERNEL_NAME!r}, not one; this weld renames it and cannot "
            f"find which one to rename")
    certified = certified.replace(
        entry, f'extern "C" __global__ void {KERNEL_NAME}(\n', 1)
    spliced = certified.replace(
        _SIGNATURE_TAIL,
        "    float gw_x, float gw_y, float gw_z,\n"
        "    // THE RECURRENCE'S OWN PARAMETERS, one group per driving state.\n"
        "    // p_now is ABSENT: the E half already binds P_<c>_<i> and both halves\n"
        "    // read it through that one parameter, because binding one allocation\n"
        "    // to two __restrict__ pointers is undefined behaviour.\n"
        + _ade_parameters(component_index, count, sigma_kinds) + "\n) {\n", 1)
    if not spliced.endswith("}\n"):
        raise AssertionError(
            "the certified off-diagonal dispersive source does not end with a "
            "closing brace; the recurrence has nowhere to be appended")
    bodies = _polar._verified_ade_bodies()  # noqa: SLF001 - the certified recurrence
    blocks = [
        "",
        "    // ------------------------------------------------------------------",
        f"    // update_P for {target}, every driving state, INSIDE this launch.",
        "    // driver.py:3313 then :3315 with nothing between the two consults, so",
        "    // the array path would read exactly what this thread just wrote.",
        "    // ------------------------------------------------------------------",
        f"    // THE SEAM: the drive is fields.drive_field({target!r}), which under",
        "    // an active layer is f_w_<c> -- the word constitutive_apply above",
        "    // stored from this same register (fw[idx] = src).",
        f"    float w = src_{target};",
    ]
    for i, volume in enumerate(sigma_kinds):
        blocks.append(_polar._pole_block(  # noqa: SLF001
            component_index, i, bool(volume), bodies))
    return (_polar._ade_prologue()  # noqa: SLF001 - the certified provenance comment
            + spliced[: -len("}\n")] + "\n".join(blocks) + "\n}\n")


def device_sources(row_mask: Sequence[int] = (1, 0, 0, 1, 0, 0),
                   counts: Sequence[int] = (1, 1, 1),
                   component_index: int = 0,
                   sigma_kinds: Sequence[bool] = (False,)) -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes."""
    return {KERNEL_NAME: kernel_source(row_mask, counts, component_index,
                                       sigma_kinds)}


def corpus_digest() -> str:
    """One sha256 over the specializations this family can be asked for.

    The product of (row mask x arity x component x sigma kind) is unbounded in
    principle and ONE POINT of it in this corpus, so the digest walks a declared
    envelope rather than an enumeration: the two row masks the corpus drives, arity
    1 and 2, all three components, and both sigma kinds. A changed character
    anywhere in either certified half or in this splice moves it.
    """
    import hashlib  # noqa: PLC0415 - stdlib, at the one call site

    digest = hashlib.sha256()
    for mask in ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1)):
        for count in (1, 2):
            for component in range(3):
                for volume in (False, True):
                    kinds = (volume,) * count
                    counts = tuple(count for _ in range(3))
                    key = (mask, counts, component, kinds)
                    digest.update(repr(key).encode("ascii"))
                    digest.update(kernel_source(mask, counts, component,
                                                kinds).encode("utf-8"))
    return digest.hexdigest()


def _get_kernel(row_mask: Sequence[int], counts: Sequence[int],
                component_index: int, sigma_kinds: Sequence[bool],
                source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source)."""
    import cupy as cp  # noqa: PLC0415 - device-only

    code = (kernel_source(row_mask, counts, component_index, sigma_kinds)
            if source is None else source)
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# =============================================================================
# THE PREDICATE
# =============================================================================

def covers_dispersive_offdiag_fused_polarization_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``update_E`` -> one component's ``update_P`` here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. The off-diagonal dispersive
    constitutive arm's own predicate and the ADE arm's own predicate are both asked
    in full. What this weld ADDS is three clauses: something must be driven, the
    drive identity behind the register hand-off must hold against the arrays this
    launch binds, and the extent the dropped ADE guard assumed must be one number.

    ``sources`` is accepted and NOT consulted, and that is the driver's own answer
    rather than an omission: nothing is injected between driver.py:3313 and :3315,
    so there is no in-seam source list for a clause to ask about. Taking the
    argument keeps every fused predicate on this track the same shape.
    """
    covered, reason = covers_real_pml_dispersive_offdiag_constitutive(
        fields, pml, grid)
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = covers_real_pml_ade_update_p(fields, pml, grid)
    if not covered:
        return False, f"ADE half: {reason}"

    target = fused_component(fields)
    if target is None:
        return False, ("no polarization drives any electric component, so there is "
                       "no recurrence to weld: that configuration is the certified "
                       "off-diagonal dispersive update_E alone")

    # THE DRIVE IDENTITY, checked against the arrays THIS launch binds rather than
    # assumed. The register hand-off is exact only while drive_field(c) really is
    # f_w_<c>; on a run where it is the stored E instead, the weld would hand the
    # recurrence a value the array path does not use.
    problem = _polar._drive_identity_problem(fields, "pml", target)  # noqa: SLF001
    if problem is not None:
        return False, f"the seam register is not the drive: {problem}"

    # THE EXTENT THE DROPPED ADE GUARD ASSUMED. This kernel emits ONE bounds guard
    # -- the E half's -- so every recurrence volume must carry exactly the stored
    # extent that guard walks, or the P half would run over a range its own
    # certified kernel would not have.
    try:
        walked = int(getattr(fields, target).size)
    except Exception as exc:  # noqa: BLE001
        return False, (f"fields.{target} could not state its size: "
                       f"{type(exc).__name__}: {exc}")
    specs = _polar.component_specs(fields)
    for state in specs[target]["states"]:
        for label, volume in (("P", state.P[target]),
                              ("P_prev", state.P_prev[target]),
                              ("_scratch", state._scratch)):
            try:
                size = int(volume.size)
            except Exception as exc:  # noqa: BLE001
                return False, (f"a state's {label}[{target}] could not state its "
                               f"size: {type(exc).__name__}: {exc}")
            if size != walked:
                return False, (f"a state's {label}[{target}] holds {size} cells and "
                               f"the launch walks {walked}; this weld drops the "
                               f"certified update_P guard and may only do so where "
                               f"the two are one number")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

def assert_disjoint_bindings(fields: Any, target: str,
                             states: Sequence[Any]) -> int:
    """The sibling's own check, per state, AFTER the rotation that decided it.

    Delegated rather than restated: ``fused_polarization_pair`` owns this question
    for this exact signature shape, and two copies of an aliasing check are equal
    only until someone edits one.
    """
    return _polar.assert_disjoint_bindings(fields, "pml", target, states) or 0


def launch_dispersive_offdiag_fused_polarization_pair(
        fields: Any, grid: Any, pml: Any, target: str, states: Sequence[Any],
        row_mask: Sequence[int], counts: Sequence[int],
        tables: Dict[str, Any], codes: Sequence[int], wall_mask: Sequence[int],
        weights: Sequence[float],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """The whole ``update_E`` and ONE component's ``update_P``, in ONE launch.

    THE CALLER ROTATES AFTERWARDS, exactly as the certified ADE launcher rotates
    after each of its own launches -- and only once this returns, so a raised launch
    leaves the state as it found it.

    ``kernel`` IS THE GATE'S DOOR: a gate compiles a deliberately broken copy of the
    shipped source and hands it here.
    """
    spec = {t: i for i, (t, _s, _a) in enumerate(ELECTRIC_TERMS)}
    component_index = spec[target]
    kinds = tuple(bool(ade_sigma_is_volume(state, target))
                  for state in states)
    assert_disjoint_bindings(fields, target, states)

    stored = getattr(fields, target)
    nx, ny, nz = (int(n) for n in stored.shape)
    n_elem = nx * ny * nz
    rows = [volume for volume in _coverage.offdiag_row_volumes(fields)
            if volume is not None]
    poles = [state.P[name] for name, _s, _a in ELECTRIC_TERMS
             for state in _polar.component_specs(fields)[name]["states"]]

    arguments: List[Any] = [fields.Ex, fields.Ey, fields.Ez,
                            fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
                            fields.Dx, fields.Dy, fields.Dz]
    arguments += [fields.inverse_epsilon_for(name)
                  for name, _s, _a in ELECTRIC_TERMS]
    arguments += poles
    arguments += rows
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz)]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kps_{axis}"], tables[f"kms_{axis}"]]
    arguments += [np.int32(int(code)) for code in codes]
    arguments += [np.int32(int(bool(flag))) for flag in wall_mask]
    arguments += [np.float32(float(value)) for value in weights]
    for index, state in enumerate(states):
        scratch = state._scratch
        if int(scratch.size) != n_elem:
            raise ValueError(
                f"state {index}'s scratch holds {int(scratch.size)} cells and the "
                f"launch walks {n_elem}; the certified update_P guard this weld "
                f"drops was that size")
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple this kernel "
                f"bakes as scalars")
        sigma = state.sigma[target]
        arguments += [scratch, state.P_prev[target],
                      sigma if kinds[index] else np.float32(sigma)]
        arguments += [np.float32(value) for value in coefficients]

    blocks = (n_elem + _FUSED_THREADS - 1) // _FUSED_THREADS
    launcher = kernel or _get_kernel(row_mask, counts, component_index, kinds)
    launcher((blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"launched": True, "component": target, "poles": len(states),
            "sigma_kinds": list(kinds), "blocks": blocks,
            "threads": _FUSED_THREADS, "elements": n_elem,
            "row_mask": tuple(int(flag) for flag in row_mask),
            "pole_counts": tuple(int(n) for n in counts)}


def _rotate(state: Any, component: str) -> None:
    """``dispersion.py:689-691``, verbatim, once a launch has returned.

    Transcribed from ``ade_kernels.update_P_fused_pml_real`` rather than
    reimplemented, and run only AFTER the launch, so a raised launch leaves the
    state exactly as it found it::

        state.P[component]      = scratch     # this step's result
        state.P_prev[component] = p           # what P held
        state._scratch          = p_prev      # the retired history
    """
    p = state.P[component]
    p_prev = state.P_prev[component]
    scratch = state._scratch
    state.P[component] = scratch
    state.P_prev[component] = p
    state._scratch = p_prev


def run_dispersive_offdiag_fused_polarization_pair(
        fields: Any, grid: Any, pml: Any, *, sources: Any = None,
        tables: Optional[Dict[str, Any]] = None,
        kernel: Optional[Any] = None, rotate: bool = True) -> Dict[str, Any]:
    """Both of :data:`REPLACES`: ONE fused launch, then the remaining recurrences.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path.

    THE TAIL IS THE CERTIFIED LAUNCHER'S OWN LOOP, minus the component the fused
    launch already advanced: ``ade_kernels``' per-(state, component) launch with the
    slots RE-READ after each rotation. Re-reading is not a precaution -- the three
    buffers rotate per component and a launcher that cached the views "would be
    stale from the second component of the first step, and stale in a way that still
    computes".

    ``rotate=False`` is the gate's door for the ``rotation_skipped`` mutation.
    """
    covered, reason = covers_dispersive_offdiag_fused_polarization_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if ade_kernels is None or dispersive_kernels is None:
        raise RuntimeError(
            "ade_kernels/dispersive_kernels are not importable on this host (they "
            "import CuPy at module scope), so the tail launches cannot run. The "
            "predicate needs neither and still answers")
    if tables is None:
        tables = _offdiag_e.dispersive_offdiag_constitutive_tables(pml)

    target = fused_component(fields)
    counts = pole_counts(fields)
    row_mask = _offdiag_e.normalized_row_mask(_coverage.offdiag_row_mask(fields))
    states = list(_polar.component_specs(fields)[target]["states"])
    record = launch_dispersive_offdiag_fused_polarization_pair(
        fields, grid, pml, target, states, row_mask, counts, tables,
        _offdiag_e.dispersive_offdiag_boundary_codes(grid, pml),
        _coverage.offdiag_wall_mask_flags(grid),
        _offdiag_e.mirror_ghost_weights(grid), kernel)
    if rotate:
        for state in states:
            _rotate(state, target)

    # THE TAIL. Every (state, component) pair the fused launch did not advance, in
    # the certified launcher's own order, with the slots re-read after each rotation.
    tail: List[Dict[str, Any]] = []
    for state in tuple(getattr(fields, "polarizations", ()) or ()):
        for component in tuple(state.driven()):
            if component == target and state in states:
                continue
            w = fields.drive_field(component)
            p = state.P[component]
            p_prev = state.P_prev[component]
            scratch = state._scratch
            sigma = state.sigma[component]
            volume = bool(ade_sigma_is_volume(state, component))
            ade_kernels._require_no_aliasing(  # noqa: SLF001 - the certified check
                scratch, p, p_prev, w, sigma if volume else None, component)
            ade_kernels._launch(  # noqa: SLF001 - the certified launch
                ade_kernels._get_kernel(volume), scratch, p, p_prev, sigma, w,
                tuple(state._coefficients), volume)
            tail.append({"component": component, "sigma_is_volume": volume})
            if rotate:
                _rotate(state, component)
    record["tail_launches"] = tail
    record["launches"] = 1 + len(tail)
    record["array_path_launches"] = 1 + sum(
        len(tuple(state.driven()))
        for state in tuple(getattr(fields, "polarizations", ()) or ()))
    record["replaces"] = REPLACES
    record["rotated"] = bool(rotate)
    return record
