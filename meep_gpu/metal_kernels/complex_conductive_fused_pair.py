"""The complex conductive ``step_D`` welded into complex ``update_E``.

ONE DISPATCH FOR ALL THREE COMPONENTS, and the displacement never leaves a
register. The separate composition for this configuration is TWO device dispatches
for the curl-and-constitutive pair — one whole-grid conductive curl plus THREE
per-component stored-E launches, four in all; this is ONE.

It claims the D->E cell the fusion matrix ranks first among Metal's unbuilt cells:
``(complex conductive no-PML curl, complex no-PML stored E)``, four corpus rows —
``TestLoadDump.test_load_dump_{structure,structure_sharded,chunk_layout_file,
chunk_layout_sim}_3d`` — and ``reach 4 (of 4 rows)``.

EVERYTHING HERE IS TRANSCRIBED, and every arithmetic line carries the file:line it
came from:

* the complex conductive D curl — :data:`.complex_no_pml_conductive._TEMPLATE`
  (complex_no_pml_conductive.py:73-145) with its ghost gather
  (:func:`.templates.ghost`), its Bloch wrap plane (:func:`.complex_fields.
  _phase_block`), its ownership mask (:func:`.templates.ownership_mask`) and its
  per-target conductive tail (:func:`.complex_no_pml_conductive._tail`);
* the constitutive half — :func:`.complex_no_pml_stored_e.complex_stored_e_source`
  (complex_no_pml_stored_e.py:78-80), transcribing ``stepping.update_E``:981-984
  through the probe-licensed ``float2`` expansion.

===========================================================================
WHAT IT IS WORTH: +4 SEAM-INSTANCES OF 387, RECOMPUTED
===========================================================================

MEASURED, not inferred from the ranked-gap table's reachable ceiling.
``results/fusion_matrix_metal_2026-08-20_unbuilt_cells/`` re-runs the standing
CLOSED cut (``results/fusion_matrix_metal_2026-08-20_closed``, 118/387) with this
product and one other added and nothing else changed; D->E moves 2 -> 6 and the
total 118 -> 126. The ladder has NO attrition, which is unusual on this seam and
is the whole reason this cell was ranked first::

    complex_conductive_plain_curl@step_D     4
    complex_stored_e@update_E                4
    no_electric_source                       4   <- the driver's clause; the CEILING
    no_folded_axis                           4
    no_metallic_axis                         4
    no_offdiagonal_epsilon                   4
    no_chi2_chi3                             4
    fits_the_binding_ceiling                 4

157 of the 186 rows are blocked at D->E by the source injection alone; these four
are among the 29 that are not, because their single source is MAGNETIC.

===========================================================================
THE SEAM, AND WHY IT IS EMPTY ON THIS CELL — the whole design question
===========================================================================

The driver runs FOUR passes between ``step_D`` and ``update_E``
(driver.py:3293-3304): the electric injection (:3294-3299, which on a conductive
row takes the ``condinv``-scaled path ``FdtdDriver._inject_electric_through_
conductivity`` at :3296), ``fill_symmetry_bc_D`` (:3300), ``zero_metal_D``
(:3301) and ``fill_folded_far_ghosts_D`` (:3302). THIS FAMILY CARRIES NONE OF
THEM. Each is instead refused by a named clause that makes the pass INERT, and
that is a different and stronger thing than not carrying it:

* **the electric sources — REFUSED BY NAME.** The injection lands between the two
  halves, so a fused pair would consume a pre-injection D. Ignorance is never an
  empty set: ``Fields`` does not hold the source list, so an undeclared source set
  is itself a refusal. THE CONDUCTIVE INJECTION IS THE SAME CLAUSE, not a second
  one: ``FdtdDriver.step`` guards it with ``if electric and
  self.fields.has_conductivity`` (driver.py:3295), so with no electric source it
  cannot run whatever the conductivity is. That is MEASURED rather than read off
  the guard — ``results/metal_unbuilt_cells_reachability_2026-08-20`` instruments
  the driver on this cell's own configuration (complex storage, an Absorber on
  both halves, k = (0.4, -1.3, 0.7), one Lorentzian, one MAGNETIC source) and
  records that nothing writes a D word inside the seam, beside a control leg
  carrying an electric source where the conductive pass fires and does;
* **``fill_symmetry_bc_D`` — INERT, because a mirror plane is refused.** Both
  certified halves refuse a live mirror already; the clause is restated here by
  name because a reader should not have to chase a shared grid predicate to learn
  which driver pass that refusal empties. Without a mirror the array path's fill
  returns immediately (stepping.py:1482-1483);
* **``fill_folded_far_ghosts_D`` — INERT for the same reason.** It runs only on a
  folded PERIODIC axis (``_stored_past_owned``, stepping.py:1503-1518);
* **``zero_metal_D`` — REFUSED BY NAME, and the refusal is priced.** It writes zero
  into stored cell 0 of every D component whose Yee shift on a walled axis is 0
  (stepping.py:2206-2247), which ``update_E`` then reads. :mod:`.fused_dispersive_
  pair` and :mod:`.folded_fused_pair` carry it inline and this family could too —
  but a METALLIC axis buys ZERO seam-instances on the measured corpus: all four
  rows of this cell are ``boundary_kinds == ['periodic', 'periodic', 'periodic']``
  (``results/metal_coverage_tranche6_2026-08-19``). A widening that buys nothing
  is declined, and the count is recorded so the decision can be revisited when the
  corpus moves rather than re-argued. Refusing the wall ALSO empties
  :func:`.templates.ownership_mask`, whose emitted block is then the certified
  emitter's own "no metallic axis" comment.

So under this family's clauses the D->E seam holds NOTHING, which is what makes a
single launch legitimate rather than merely convenient.

===========================================================================
WHY THERE IS NO STENCIL PROBLEM HERE, and why that is not general
===========================================================================

The fusion matrix refuses NINETEEN D->E seam-instances as STRUCTURALLY UNFUSABLE
on any backend: ``stepping._offdiagonal_terms`` (stepping.py:1235-1253) reads each
partner component's D volume at FOUR indices while ``step_D`` writes those volumes
in place in the first half of the same launch, and no grid-wide barrier exists
inside one launch. THIS CONSTITUTIVE IS NOT THAT ONE. The certified complex stored
E reads ``d_in[idx]`` — its OWN component, at its OWN index, the word the same
thread computed in the curl half — so the value stays in a register and no thread
reads a word another thread writes. An off-diagonal ``chi1inv`` row is refused by
the E half already, and restated here by name because THIS is the reason the cell
is buildable at all.

===========================================================================
WHAT MOVES, AND IT IS AT MOST ONE LINE PER COMPONENT
===========================================================================

The certified conductive tail (complex_no_pml_conductive.py:148-156) ALREADY names
the stepped displacement::

    float2 value0 = f0[ii];
    value0 = c_mul_field_left(value0, cf0[ii]);
    value0 = value0 - curl0;
    value0 = c_mul_field_left(value0, ci0[ii]);
    f0[ii] = value0;

so on a CONDUCTIVE component NOTHING MOVES AT ALL: the constitutive half simply
reads ``value0`` where the certified body read ``d_in[idx]``. On a component
without a conductivity the certified tail is the single line ``f0[ii] = f0[ii] -
curl0;``, and there the fusion splits it in two — ``float2 value0 = f0[ii] -
curl0; f0[ii] = value0;`` — the same store-then-reload split
:mod:`.fused_ade_chain` makes at the E->P seam. A ``float2`` stored to a ``device
float2*`` and reloaded is bit-identical to the register, so the split is
byte-neutral by construction; the gate's ``byte_neutral_control`` leg MEASURES it
per complete step regardless, because "by construction" is a hypothesis until a
comparator agrees.

===========================================================================
THE POINTER RENAMES, AND WHY THE SCALARS MOVE INTO A STRUCT
===========================================================================

**POINTER RENAMES.** The curl half keeps its own spelling (``f0..f2``, ``g0..g2``,
``cf0..cf2``, ``ci0..ci2``). The constitutive half's three volume names would each
mean three things in one scope, so they take a component index:
``e_out -> e0/e1/e2``, ``inv_e -> iv0/iv1/iv2``, and pole ``k`` of component
``m`` is ``p{m}_{k}``. ``d_in`` IS NOT BOUND AT ALL — it is ``f{m}``, the volume
the curl half just wrote, and that shared binding is the fusion's whole footprint.

**THE SCALARS.** :data:`.device.MAX_BUFFER_BINDINGS` is 31. The certified curl
binds EIGHT scalars separately (``nx``, ``ny``, ``nz``, ``n_elem``, ``dtdx`` and
the three ``float2`` Bloch phases, complex_no_pml_conductive.py:85-92); with the
constitutive half's six volumes on top, a one-pole configuration would need
``12 + 6 + 3 + 8 = 29`` and a TWO-pole one 32, a compile error. Packing every
scalar into ONE ``constant Params&`` — the shape :mod:`.fused_dispersive_pair`
measured to compile — brings the same two-pole configuration to 25 and moves the
ceiling from one pole per component to TWELVE poles in total. :func:`binding_count`
is evaluated BY THE PREDICATE, so a configuration over the ceiling is a REFUSAL BY
NAME rather than a compile error at plan time.

The eight-line prologue that copies ``prm.nx`` / ``prm.ny`` / ``prm.nz`` /
``prm.n_elem`` / ``prm.dtdx`` and rebuilds ``px`` / ``py`` / ``pz`` back into
locals exists so :data:`.templates.GUARD`, :data:`.templates.DECODE_IJK`,
:func:`.templates.ghost`, :func:`.complex_fields._phase_block`,
:func:`.templates.ownership_mask` and the tails below are emitted from the
CERTIFIED emitters unchanged. The phases are packed as SIX floats rather than three
``float2`` members so the blob stays a flat 4-byte word sequence with no alignment
to reason about, and are rebuilt with ``float2(re, im)`` — a construction, not an
arithmetic operation, so no rounding is introduced.

THE POLE SLOTS ARE NOT PADDED TO EIGHT. ``complex_no_pml_stored_e`` declares
``p0..p7`` always and binds the displacement into the unused slots
(complex_no_pml_stored_e.py:196). Here the signature declares exactly the poles
each component carries, because the pad is what puts this over the ceiling: three
components padded to eight is 24 pole pointers against the 12 the ceiling allows.

===========================================================================
WHAT THIS FAMILY REFUSES
===========================================================================

Beyond the seam clauses above: an active PML layer, real (float32) storage, a
mirror plane, cylindrical coordinates, BFAST, a nonzero ``special_kz`` beta, an
off-diagonal ``chi1inv`` row, ``chi2``/``chi3``, and a grid with no conductivity on
any curl target — every one of them already refused by one or both certified
halves, whose reasons are forwarded verbatim with the half named.

NOT WIRED. ``STEP_ORDER`` assigns at most one arm per slot and this product spans
TWO (``step_D`` and ``update_E``), so it registers ``wired=False``: the composer
cannot select it, and the disjointness sweep can still enumerate it. Nothing about
any existing arm's behaviour changes, and ``meep_gpu.fastpath.plan_fast_path``
still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage, _boundary_kinds, _call
from ..triton_kernels.no_pml_conductive import conductive_no_pml_targets
from . import shaders, templates
from .ade_update_p import _physical_name
from .complex_fields import (
    _complex_mirror,
    _phase_block,
    bloch_phase_table,
    expansion_from_probe,
    load_expansion_probe,
    phase_arguments,
)
from .complex_no_pml_conductive import (
    SUB_STEPS,
    _source_binding,
    _tail,
    metal_complex_conductive_no_pml_curl_coverage,
)
from .complex_no_pml_stored_e import (
    _poles,
    metal_complex_stored_e_coverage,
)
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "complex_conductive_fused_pair"

#: The slot the arm is registered on. The product spans two; it holds a row on the
#: FIRST, so a refusal is named on the slot the fusion starts at.
SLOT = "step_D"

#: The driver passes one run of this plan performs, in driver order
#: (driver.py:3293-3304). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "update_E")

#: The field type the driver injects INSIDE this seam (driver.py:3294-3299).
ELECTRIC_FIELD_TYPE = "D"

#: The curl half's own pointers: three D targets, three H sources, three
#: ``condfac`` and three ``condinv`` volumes (complex_no_pml_conductive.py:76-87).
CURL_POINTERS = 12

#: The constitutive half's fixed pointers: three E targets and three inverse
#: epsilons. ``d_in`` is NOT among them — it is the curl's own ``f{m}``.
CONSTITUTIVE_POINTERS = 6

#: ``(component, displacement, curl index)`` — the certified tables joined, so one
#: loop serves both halves (complex_no_pml_stored_e.E_TERMS, SUB_STEPS['step_D']).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

__all__ = [
    "CONSTITUTIVE_POINTERS", "CURL_POINTERS", "E_TERMS", "ELECTRIC_FIELD_TYPE",
    "FAMILY", "REPLACES", "SLOT",
    "MetalComplexConductiveFusedPairPlan", "binding_count",
    "compile_complex_conductive_fused_pair",
    "complex_conductive_fused_pair_coverage",
    "complex_conductive_fused_pair_source", "pack_params", "params_words",
    "plan_metal_complex_conductive_fused_pair",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__
__HELPERS__

struct Params {
    float dtdx;
    float px_re;
    float px_im;
    float py_re;
    float py_im;
    float pz_re;
    float pz_im;
    uint  nx;
    uint  ny;
    uint  nz;
    uint  n_elem;
};

kernel void complex_conductive_fused_pair_step(
__SIGNATURE__
    uint idx [[thread_position_in_grid]])
{
    // The scalars live in ONE binding (see the module docstring's binding
    // arithmetic); copying them back into the names the certified emitters use is
    // what lets GUARD, DECODE_IJK, ghost(), _phase_block(), ownership_mask() and
    // the tails below be those emitters' own text. float2(re, im) is a
    // construction, not an arithmetic operation: no rounding is introduced.
    uint nx = prm.nx;
    uint ny = prm.ny;
    uint nz = prm.nz;
    uint n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = float2(prm.px_re, prm.px_im);
    float2 py = float2(prm.py_re, prm.py_im);
    float2 pz = float2(prm.pz_re, prm.pz_im);
__GUARD__
__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : float2(0.0f, 0.0f);
    float2 a_z = vz ? g0[oz] : float2(0.0f, 0.0f);
    float2 b_x = vx ? g1[ox] : float2(0.0f, 0.0f);
    float2 b_z = vz ? g1[oz] : float2(0.0f, 0.0f);
    float2 c_x = vx ? g2[ox] : float2(0.0f, 0.0f);
    float2 c_y = vy ? g2[oy] : float2(0.0f, 0.0f);

__PHASE_X__
__PHASE_Y__
__PHASE_Z__

    float2 t0 = ((c_y - c) + (b - b_z));
    float2 t1 = ((a_z - a) + (c - c_x));
    float2 t2 = ((b_x - b) + (a - a_y));
    float2 curl0 = c_mul_coefficient_left(dtdx, t0);
    float2 curl1 = c_mul_coefficient_left(dtdx, t1);
    float2 curl2 = c_mul_coefficient_left(dtdx, t2);

    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

__TAIL0__
__TAIL1__
__TAIL2__

    // --- THE SEAM: driver.py:3293-3304 holds NOTHING under this family's
    //     clauses (no electric source, no mirror, no wall). See the module
    //     docstring; the reachability is MEASURED, not read off the guard.
__CONSTITUTIVE__
}
"""


def binding_count(pole_counts: Sequence[int]) -> int:
    """Buffer attribute indices the fused signature would occupy.

    ``pointers = 12 + 6 + sum(poles)`` — the curl half's own twelve volumes, the
    constitutive half's three targets and three inverse epsilons, then one pole
    pointer per (component, pole). ``d_in`` is not counted: it IS the curl's
    ``f{m}``, which is the fusion. Plus ONE for the packed ``Params&``.
    """
    counts = tuple(int(count) for count in pole_counts)
    if len(counts) != len(E_TERMS):
        raise ValueError(
            f"pole_counts must carry one entry per component, got {counts!r}")
    if any(count < 0 for count in counts):
        raise ValueError(f"pole counts must be non-negative, got {counts!r}")
    return (CURL_POINTERS + CONSTITUTIVE_POINTERS + sum(counts)) + 1


def params_words() -> int:
    """Length of the ``Params`` blob in 4-byte words.

    Every member is a 4-byte scalar, so the struct's layout is a flat word sequence
    with no padding to reason about — which is why the three Bloch phases are six
    floats rather than three ``float2`` members.
    """
    return 11


def pack_params(dtdx: float, phase_values: Sequence[Tuple[float, float]],
                nx: int, ny: int, nz: int, n_elem: int) -> np.ndarray:
    """The ``Params`` blob as float32 words, uints carried as their bit patterns.

    ``float()`` on ``dtdx`` for the reason :mod:`.fused_dispersive_pair` gives its
    own: the array path multiplies a float32 volume by a Python float and Metal
    binds a Python float into a ``float`` member the same way, so the two scalars
    are the same bits. The phases arrive ALREADY rounded to complex64 and, for this
    BACKWARD sub-step, already conjugated — :func:`.complex_fields.phase_arguments`
    does both (complex_fields.py:1055-1060), and this packer does neither.
    """
    if len(phase_values) != 3:
        raise ValueError(f"phase_values must be a triple, got {phase_values!r}")
    blob = np.zeros(params_words(), dtype=np.float32)
    blob[0] = np.float32(float(dtdx))
    for axis, (real, imaginary) in enumerate(phase_values):
        blob[1 + 2 * axis] = np.float32(float(real))
        blob[2 + 2 * axis] = np.float32(float(imaginary))
    blob[7:11] = np.frombuffer(
        np.array([int(nx), int(ny), int(nz), int(n_elem)],
                 dtype=np.uint32).tobytes(), dtype=np.float32)
    return blob


def _signature(pole_counts: Sequence[int]) -> str:
    """Every binding, in the order :meth:`MetalComplexConductiveFusedPairPlan.run` passes them."""
    lines: List[str] = []
    slot = 0

    def bind(declaration: str) -> None:
        nonlocal slot
        lines.append(f"    {declaration:<33}[[buffer({slot})]],")
        slot += 1

    for index in range(3):                 # the curl half's own twelve, in its order
        bind(f"device float2*       f{index}")
    for index in range(3):
        bind(f"device const float2* g{index}")
    for index in range(3):
        bind(f"device const float*  cf{index}")
    for index in range(3):
        bind(f"device const float*  ci{index}")
    for index in range(3):                 # the constitutive half's targets
        bind(f"device float2*       e{index}")
    for index in range(3):
        bind(f"device const float*  iv{index}")
    for index in range(3):                 # SHARED: d_in IS f{index}, never bound
        for pole in range(int(pole_counts[index])):
            bind(f"device const float2* p{index}_{pole}")
    bind("constant Params&     prm")
    assert slot == binding_count(pole_counts), (slot, tuple(pole_counts))
    return "\n".join(lines)


def _constitutive(index: int, pole_count: int) -> List[str]:
    """One component's ``update_E``, transcribed from ``complex_stored_e_source``.

    ``complex_no_pml_stored_e.py:78-80`` is the body::

        float2 source = d_in[idx];
        source = source - p0[idx];          (once per pole)
        e_out[idx] = c_mul_field_left(source, inv_e[idx]);

    Reproduced here with the pointer renames the module docstring names and ONE
    substitution — ``d_in[idx]`` becomes the register the curl half just wrote.
    The braces give each component its own ``source`` scope, so the final line is
    character for character that body's modulo the rename.
    """
    return [
        f"    {{  // {E_TERMS[index][0]}: complex_no_pml_stored_e.complex_stored_e_source (:78-80)",
        f"        float2 source = value{index};   // THE SEAM: the register step_D just wrote.",
        *[f"        source = source - p{index}_{pole}[idx];"
          for pole in range(int(pole_count))],
        f"        e{index}[idx] = c_mul_field_left(source, iv{index}[idx]);",
        "    }",
    ]


def complex_conductive_fused_pair_source(
        codes: Sequence[int], phased: Sequence[int], conductive: Sequence[bool],
        pole_counts: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, phases, conductivity, poles, arm, mode).

    ``backward`` is not a parameter: this family exists at ONE seam and ``step_D``
    is always the backward sub-step (SUB_STEPS['step_D']['backward']), so passing
    it would offer a forward specialisation the family has no seam for.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    flags = tuple(bool(flag) for flag in conductive)
    counts = tuple(int(count) for count in pole_counts)
    if len(codes) != 3 or len(phased) != 3 or len(flags) != 3 or len(counts) != 3:
        raise ValueError("codes, phased, conductive and pole_counts must be triples")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag and code != templates.PERIODIC:
            raise ValueError(f"axis {axis} carries a phase but is not periodic")
    bindings = binding_count(counts)
    if bindings > MAX_BUFFER_BINDINGS:
        raise ValueError(
            f"the fused signature at pole counts {counts!r} needs {bindings} "
            f"bindings and this platform allows {MAX_BUFFER_BINDINGS} "
            f"(device.py:72); the predicate refuses this configuration by name "
            f"rather than letting it reach the compiler")

    backward = bool(SUB_STEPS["step_D"]["backward"])
    tails: Dict[str, str] = {}
    for index in range(3):
        if flags[index]:
            # THE CONDUCTIVE TAIL ALREADY NAMES THE VALUE. Nothing moves: the
            # certified emitter's own text, unchanged (complex_no_pml_conductive.py:151-155).
            tails[f"__TAIL{index}__"] = _tail(index, True)
        else:
            # The non-conductive tail is ONE line; the fusion splits it in two so
            # the product has a name the constitutive half can read from a
            # register. A float2 stored and reloaded is exact, so the split is
            # byte-neutral; the gate measures it per complete step regardless.
            certified = _tail(index, False)
            assert certified == f"    f{index}[ii] = f{index}[ii] - curl{index};", certified
            tails[f"__TAIL{index}__"] = (
                f"    float2 value{index} = f{index}[ii] - curl{index};\n"
                f"    f{index}[ii] = value{index};")

    constitutive: List[str] = []
    for index in range(3):
        constitutive.extend(_constitutive(index, counts[index]))

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__SIGNATURE__": _signature(counts),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PHASE_X__": _phase_block("x", bool(phased[0]), backward),
        "__PHASE_Y__": _phase_block("y", bool(phased[1]), backward),
        "__PHASE_Z__": _phase_block("z", bool(phased[2]), backward),
        "__MASK__": templates.ownership_mask(
            codes, backward, zero=templates.COMPLEX_ZERO),
        "__CONSTITUTIVE__": "\n".join(constitutive),
        **tails,
    })


def compile_complex_conductive_fused_pair(
        codes: Sequence[int], phased: Sequence[int], conductive: Sequence[bool],
        pole_counts: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one whole configuration."""
    return compile_source(complex_conductive_fused_pair_source(
        codes, phased, conductive, pole_counts, expansion,
        contract)).complex_conductive_fused_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_conductive_fused_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span complex conductive ``step_D`` -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it — the construction
    :func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage` uses.
    """
    reasons: List[str] = []

    curl = metal_complex_conductive_no_pml_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = metal_complex_stored_e_coverage(fields, pml, residency, probe)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, and for this family it is THE binding clause on the measured
    # corpus. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does NOT disqualify the pair -- which is exactly
    # the polarity that makes this cell's four corpus rows reachable.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299), and on a conductive row through "
            f"_inject_electric_through_conductivity (driver.py:3296)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane; this restates by
    # NAME what that refusal buys on THIS seam — a reader should not have to chase
    # a shared grid clause to learn that two driver passes sit between the halves
    # on a folded grid.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # THE WALL. zero_metal_D writes stored cell 0 of every D component whose Yee
    # shift on a walled axis is 0 (stepping.py:2206-2247) and update_E then reads
    # it. This family REFUSES the wall rather than carrying it, and the module
    # docstring prices that decision: a metallic axis buys zero seam-instances on
    # the measured corpus.
    kinds = _boundary_kinds(grid, None)
    if kinds is None:
        # UNREADABLE IS A REFUSAL, NEVER A CRASH. `_boundary_kinds` returns None
        # for a grid the array path will not resolve (coverage.py:105-121), and a
        # clause that iterated it would raise where the contract says refuse.
        reasons.append(
            "the per-axis boundary rule is unreadable, so this family cannot "
            "establish that stepping.zero_metal_D (driver.py:3301) is inert")
        kinds = ()
    for axis, kind in enumerate(kinds):
        if kind == "metallic":
            reasons.append(
                f"axis {axis} is metallic: stepping.zero_metal_D runs inside this "
                f"seam (driver.py:3301) and this family does not carry it")

    # THE STENCIL, restated. The nineteen structurally unfusable D->E seam-instances
    # on this board are the OFF-DIAGONAL constitutive reading partner D volumes at
    # four indices (stepping.py:1235-1253) while step_D writes them in place. The E
    # half refuses that row already; it is named again because THIS is the fact
    # that makes the present cell buildable at all.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D "
            "volumes step_D writes in place (stepping.py:1235-1253), and no "
            "grid-wide barrier exists inside one launch")

    # THE POLE COUNTS MUST FIT. Evaluated here so an over-ceiling configuration is
    # a refusal BY NAME rather than a compile error at plan time.
    try:
        order = _poles(fields)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))
    counts = [len(tuple(order.get(component, ()))) for component, _, _ in E_TERMS]
    bindings = binding_count(counts)
    if bindings > MAX_BUFFER_BINDINGS:
        reasons.append(
            f"pole counts {tuple(counts)} need {bindings} bindings and this "
            f"platform allows {MAX_BUFFER_BINDINGS} (device.py:72)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """One component's constitutive binding set, resolved once at plan time."""

    component: str
    index: int
    target: Any               # E<c>
    inverse_epsilon: Any
    states: Tuple[Any, ...]


class MetalComplexConductiveFusedPairPlan:
    """ONE dispatch that performs two sub-step passes, the displacement never reloaded.

    ``launches_per_run`` is ONE and is declared rather than inherited: the
    whole-step arbiter asserts the exact per-cycle launch count on every filled
    slot, and that assertion is what stops a slot passing by not executing.

    THE POLE POINTERS ARE RESOLVED AHEAD OF EVERY LAUNCH, not frozen into an
    argument tuple. ``update_P`` rotates which physical allocation holds ``P[c]``
    after each of its own launches (ade_update_p.py:353-355 -> dispersion.py:
    689-691), so a cached pole pointer would subtract the WRONG buffer from step
    two onward — the same discipline ``MetalComplexStoredEPlan.run`` follows
    (complex_no_pml_stored_e.py:193-199). Everything else IS frozen at build time.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = 1

    __slots__ = ("entries", "residency", "volumes", "expansion", "shape",
                 "pole_counts", "conductive", "bc", "phased", "_functions",
                 "_tensors", "_fixed", "launches", "runs")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], expansion: str, shape: Sequence[int],
                 pole_counts: Sequence[int], conductive: Sequence[bool],
                 codes: Sequence[int], phased: Sequence[int],
                 functions: Mapping[str, Any], tensors: Mapping[int, Any],
                 fixed: Sequence[Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.expansion = str(expansion)
        self.shape = tuple(int(n) for n in shape)
        self.pole_counts = tuple(int(count) for count in pole_counts)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.bc = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self._fixed = tuple(fixed)
        self.launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted(self._functions))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete D->E seam, in place, against the mirrors."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._functions.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} "
                f"rather than launching the pinned one")
        poles: List[Any] = []
        for entry in self.entries:
            resolved = [state.P[entry.component] for state in entry.states]
            if len(resolved) != self.pole_counts[entry.index]:
                raise RuntimeError(
                    f"{entry.component} resolved {len(resolved)} poles, not the "
                    f"compiled count {self.pole_counts[entry.index]}")
            poles.extend(self._tensors[id(value)] for value in resolved)
        function(*self._fixed, *poles, self._tensors["params"])
        self.launches += 1
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalComplexConductiveFusedPairPlan(shape={self.shape}, "
                f"bc={self.bc}, phased={self.phased}, "
                f"conductive={self.conductive}, poles={self.pole_counts}, "
                f"expansion={self.expansion!r}, variants={self.variants!r})")


def plan_metal_complex_conductive_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None,
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalComplexConductiveFusedPairPlan]:
    """Build the fused D->E plan, or return ``None`` on any refusal."""
    if not complex_conductive_fused_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    assert residency is not None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:
        return None
    grid = fields.grid
    spec = SUB_STEPS["step_D"]
    kinds = _boundary_kinds(grid, None)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    phased, phase_values = phase_arguments(
        bloch_phase_table(grid, kinds), backward=bool(spec["backward"]))
    conductive = tuple(conductive_no_pml_targets(fields, "step_D"))

    tensors: Dict[int, Any] = {}
    volumes: List[str] = []

    # --- the curl half's twelve, bound EXACTLY as the certified plan binds them,
    #     under the SAME residency names, so a composition holding both binds one
    #     tensor rather than two (complex_no_pml_conductive.py:378-397).
    targets = [_complex_mirror(residency, name, getattr(fields, name))
               for name in spec["targets"]]
    source_bindings = [_source_binding(fields, "step_D", name)
                       for name in spec["sources"]]
    curl_sources = [_complex_mirror(residency, name, value)
                    for name, value in source_bindings]
    condfac = [
        (residency.mirror(f"condfac_{name}:complex_no_pml",
                          fields.condfac_for(name), constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])]
    condinv = [
        (residency.mirror(f"condinv_{name}:complex_no_pml",
                          fields.condinv_for(name), constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])]
    volumes.extend(spec["targets"])
    volumes.extend(name for name, _ in source_bindings)

    # --- the constitutive half's six, under the certified pointwise plan's names
    order = _poles(fields)
    states = tuple(getattr(fields, "polarizations", ()) or ())
    index_of = {id(state): position for position, state in enumerate(states)}
    entries: List[_Entry] = []
    e_targets: List[Any] = []
    inverses: List[Any] = []
    pole_counts: List[int] = []
    for component, _displacement, index in E_TERMS:
        target = getattr(fields, component)
        inverse = fields.inverse_epsilon_for(component)
        e_targets.append(residency.mirror(component, target, dtype=np.complex64))
        inverses.append(residency.mirror(
            f"inv_eps_{component}:no_pml", inverse, constant=True, dtype=np.float32))
        volumes.extend((component, f"inv_eps_{component}:no_pml"))
        chain = tuple(order.get(component, ()))
        pole_counts.append(len(chain))
        entries.append(_Entry(component, index, target, inverse, chain))

    # --- every pole allocation the rotation can hand back, mirrored ONCE and
    #     looked up by identity at launch time (see MetalComplexConductiveFused
    #     PairPlan.run for why the pointer cannot be frozen).
    for state_index, state in enumerate(states):
        for component in tuple(_call(state, "driven", default=()) or ()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                name = _physical_name(state_index, label, host)
                tensors[id(host)] = residency.mirror(name, host, dtype=np.complex64)
                volumes.append(name)
        scratch = state._scratch
        name = _physical_name(state_index, "scratch", scratch)
        tensors[id(scratch)] = residency.mirror(name, scratch, dtype=np.complex64)
        volumes.append(name)

    params = pack_params(grid.dt / grid.dx, phase_values, grid.shape[0],
                         grid.shape[1], grid.shape[2],
                         int(np.prod(grid.shape)))
    params_name = "complex_conductive_fused_pair:params"
    tensors["params"] = residency.mirror(params_name, params, constant=True)
    volumes.append(params_name)

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_complex_conductive_fused_pair(
                codes, phased, conductive, pole_counts, expansion, mode)
    fixed = tuple(targets) + tuple(curl_sources) + tuple(condfac) + tuple(condinv) \
        + tuple(e_targets) + tuple(inverses)
    return MetalComplexConductiveFusedPairPlan(
        entries, residency, volumes, expansion, grid.shape, pole_counts,
        conductive, codes, phased, selected, tensors, fixed)


# ---------------------------------------------------------------------------
# Registration -- one arm, not wired
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"the complex conductive fused pair cannot fill slot {slot!r}",))
    return complex_conductive_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("probe"))


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalComplexConductiveFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_conductive_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, context.extra.get("probe"))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        family=FAMILY,
        slot=SLOT,
        label="fused complex conductive no-PML curl -> complex stored E",
        coverage=_arm_coverage,
        plan=_arm_plan,
        prefix="fused complex conductive no-PML pair: ",
        noun="complex conductive fused D->E pair",
        wired=False,
        replaces=REPLACES,
    ),)


ARMS = register_arms()
