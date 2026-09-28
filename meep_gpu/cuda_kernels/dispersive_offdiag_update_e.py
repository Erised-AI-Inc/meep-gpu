"""Raw-CUDA ``update_E`` where the TENSOR ROW PRODUCT's operands are ``D - sum P``.

THE LAST NON-COMPLEX SLOT IN THE CORPUS, AND THE FOUR REFUSALS THAT SPECIFY IT.
The 2026-08-20 union census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_final/``) reads
753 / 759 slots served by twenty shipped hand-CUDA families. Five of the six
residual slots are complex64. The sixth is ``examples/absorbed_power_density.py``
at ``update_E`` -- 800 x 402 x 1, one mirror plane on Y, one registered
susceptibility, an off-diagonal tensor, an active PML -- and FOUR shipped
families refuse it, each for a different true reason::

    cuda_constitutive     "dispersion: update_E's source is (D - sum P), not D,
                           and update_P closes the step"
    cuda_folded_offdiag   the same clause, same words
    cuda_dispersive       "off-diagonal chi1inv: the row product reads the other
                           components' (D - sum P) volumes at neighbouring cells
                           and this arm is element-wise"
    cuda_offdiag          "mirror symmetry: the fold changes the stored extent,
                           and the extent is what turns a cell index into a
                           coefficient index"

Read together those four ARE the specification: the tensor row product whose
OPERANDS are ``D - sum P`` rather than ``D``, over a mirror-folded stored extent.
This module is that kernel. It changes NOT ONE BYTE of ``offdiag_emitter.py``,
``offdiag_constitutive_kernels.py``, ``folded_offdiag_kernels.py`` or
``dispersive_kernels.py``; the four refusals above STAY, and they are the
disjointness seams that route this exact intersection here and nowhere else.

=============================================================================
WHY A HOST-SIDE FUSION OF TWO SHIPPED KERNELS CANNOT SERVE
=============================================================================

The census note says this slot may want "a new kernel OR a measured fusion of two
shipped ones". IT CANNOT BE A FUSION OF TWO SHIPPED ONES, and the reason is a
fact about their OUTPUT SETS rather than a preference:

* ``update_E_pml_real_dispersive`` (dispersive_kernels) never writes
  ``D - sum P``. It writes ``f_w_Ec = (D - sum P) * inv_eps`` and accumulates
  ``E``. Recovering the row product's operand from that output needs a DIVISION
  by ``inv_eps``, which is not the identity in float32, and its ``E``
  accumulation would then have to be undone.
* ``update_E_pml_real_folded_offdiag`` (folded_offdiag_kernels) reads its
  operands through the three ``D`` pointers and has no pole parameters at all.

So the only composition that reproduces the array path is **materialize then
launch**: form the three ``D - sum P`` volumes first -- which is what
``Fields.displacement_minus_polarization_volumes`` (fields.py:1107-1138) itself
does -- and hand the shipped folded off-diagonal kernel those buffers in place of
``D``. That is a fusion of ONE shipped kernel with the ARRAY PATH, not of two
kernels, and it costs ``sum(pole counts)`` extra full-volume passes plus three
scratch volumes per step.

BOTH WERE MEASURED, and the answer is a number rather than a paragraph.
``parity/meep_gpu/gate_cuda_dispersive_offdiag.py``'s ``composition`` leg ran
exactly that route against the same oracle on the same frozen states, on
the GPU host's RTX A6000 under both float32 subnormal policies::

    composition, bit-identical to the oracle       54 / 54
    composition, extra full-volume passes         234 across those 54 cases
    composition, scratch volumes held live        up to 3
    THIS KERNEL, extra full-volume passes           0

So the composition is CORRECT and it is not free. Counted rather than timed: a
shared box cannot give a trustworthy time, and a pass count is exact and
contention-free. Both numbers are in both artifacts named by
:data:`DISPERSIVE_OFFDIAG_ADMISSION`.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

``stepping.update_E``'s ``elif offdiagonal:`` branch (stepping.py:1001-1008) with
``fields.polarizations`` non-empty. The ONE thing dispersion changes about it is
which volumes ``displacement`` holds (stepping.py:996)::

    displacement = _nonlinear_displacement(fields, pml)          # :967
        -> volumes = fields.displacement_minus_polarization_volumes()   # :1041
    source       = displacement["volumes"][component]            # :974
    constitutive = source * fields.inverse_epsilon_for(component)  # :975
    coupling     = _offdiagonal_terms(fields, component, displacement)  # :976
        -> values = volumes[partner]                             # :1211
        -> pair    = values + _shift_down(values, partner_axis, ...)   # :1214
        -> product = pair * coefficient                          # :1217
        -> term    = 0.25 * (product + _shift_up(product, own_axis, ...))  # :1219
        -> total   = term(offset 1) then + term(offset 2)        # :1221
        -> _mask_metallic_wall_coupling(total)                   # :1223
    constitutive = constitutive + coupling                       # :977-978
    prev = f_w_c ; f_w_c = constitutive                          # :2065-2096
    E_c += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev              # half-integer, :986

THE MIDDLE REFUSAL IS THE WHOLE POINT AND IT IS EASY TO UNDERSTATE. It is NOT
enough to subtract the polarization from the diagonal term. ``values`` at
stepping.py:1240 is ``volumes[partner]`` -- the PARTNER component's
``D - sum P`` -- and it is read at FOUR addresses per term: home, one cell DOWN
the partner's axis, one cell UP the row component's own axis, and the corner. So
every one of those four gathered reads is a subtraction chain over the PARTNER's
poles, at a NEIGHBOURING cell. A kernel that formed ``D - sum P`` for the
diagonal term and then coupled the raw ``D`` volumes would be wrong on every cell
where any pole is nonzero, smoothly and plausibly.

``Fields.displacement_minus_polarization_volumes`` (fields.py:1107-1138) decides
the chain::

    displacement = getattr(self, "D" + component[1])
    contributors = [s for s in self.polarizations if s.drives(component)]
    if not contributors:
        volumes[component] = displacement      # THE D ARRAY ITSELF, aliased
        continue
    scratch[...] = displacement
    for state in contributors:
        state.subtract_into(component, scratch)    # scratch -= state.P[component]

so the chain is ``((D - P0) - P1) - P2``: SEQUENTIAL, LEFT TO RIGHT, in
``fields.polarizations`` order filtered by ``drives``. NOT ``D - (P0 + P1 + ...)``:
float32 addition is not associative, and the two agree EXACTLY at one pole, which
is why the gate's ordering legs are scored at two and above and why
:data:`POLE_COUNTS_SWEPT` carries multi-pole arities. A component with no
contributor forms no subtraction at all, and the emitter emits none for it.

THE ARRAY PATH MATERIALIZES ALL THREE VOLUMES BEFORE THE COMPONENT LOOP AND THIS
KERNEL RECOMPUTES EACH READ. That is the same float32 number every time and it is
worth saying why rather than assuming it: ``dmp`` is a pure function of ``D`` and
the ``P`` buffers, this sub-step WRITES only ``E`` and ``f_w_E``, and those four
sets are disjoint -- so no store in the kernel can reach an operand of a later
load, and evaluating the chain once per read or once per volume is the same
sequence of float32 operations on the same bits.

=============================================================================
THE FOLD, AND WHY IT IS THE SAME THREE FACTS AS THE NON-DISPERSIVE FAMILY'S
=============================================================================

``folded_offdiag_kernels``'s module docstring establishes them against
``stepping``, with lines, and this family inherits all three UNCHANGED because
the fold acts on the SHIFT HELPERS and the poles act on the VALUES those helpers
move:

1. the fold enters through exactly one leg, the PARTNER-axis down shift, whose
   MIRROR branch is ``_symmetry_phase(...) * _mirror_source(field, axis)``
   (stepping.py:1870-1872) -- stored row ``MIRROR_ROW`` weighted by the plane's
   parity, neither the periodic wrap nor the metallic zero;
2. the OWN-axis up shift is an exact zero on a fold, because
   ``_offdiagonal_terms`` calls ``_shift_up`` with four arguments
   (stepping.py:1248-1249) so the reflect branch cannot fire;
3. one ``BC_MIRROR`` code serves BOTH declared terminations, because ``update_E``
   has no ownership mask and no reflect row.

``coord_up`` and ``coord_dn`` are therefore taken OUT OF ``folded_offdiag_kernels
._OWN_PRELUDE`` rather than retyped, exactly as that module takes ``flat``,
``ghosted`` and ``constitutive_apply`` out of ``offdiag_emitter.PRELUDE``, and
for the same reason: two copies of a ghost rule are equal only until someone edits
one, and a silent divergence would be a plane of wrong values that no diff of THIS
file would show. :func:`prelude_provenance` names every borrowed block and its
source module, and ``test_dispersive_offdiag_update_e.py`` asserts each still
appears verbatim in its owner and in every emitted source.

WHAT IS NEW HERE IS EXACTLY ONE THING: the three ghosted loads that the fold and
the metallic wall route through are now ``dmp`` chains rather than plain loads. So
this family emits ``dmp_<component>``, ``dmp_ghosted_<component>`` and
``dmp_ghosted_mirror_<component>`` instead of ``ghosted`` and ``ghosted_mirror``
on the FIELD side, and keeps the certified ``ghosted`` for the COEFFICIENT side
(``ghosted(u, up)``), which carries no poles and never did.

=============================================================================
WHAT IS SPECIALIZED AND WHAT IS NOT
=============================================================================

TWO SOURCE AXES, and each is an ARITY axis in the sense both sibling emitters
use -- it changes how many terms an expression has and therefore its association:

* the six ROW-LIVENESS flags (``offdiag_emitter``'s only source axis);
* the ``(n_Ex, n_Ey, n_Ez)`` POLE COUNT TRIPLE (``dispersive_kernels``' only
  source axis).

The BOUNDARY codes, the WALL flags and the GHOST WEIGHTS stay runtime ``int`` and
``float`` arguments, which is ``folded_offdiag_kernels``' split and its reason:
they select an index or a predicated zero and change no float operation and no
association. The corpus drives ONE point of the product -- row mask
``(1, 0, 0, 1, 0, 0)`` at arity ``(1, 1, 1)`` -- and the gate sweeps a
deliberately wider set.

ZERO-PADDING THE POLE AXIS IS NOT AVAILABLE, on the certified sibling's measured
grounds: binding a dead slot to an all-zero ``P`` so a missing pole contributes
``- 0.0f`` is NOT bit-identical, because ``x - 0.0f`` turns ``-0.0f`` into
``-0.0f`` but ``+0.0f - 0.0f`` is ``+0.0f`` while not subtracting leaves the
operand alone. ``test_offdiag_constitutive_pml_real.py`` carries the sibling
witness for the row axis; the array path's own aliasing branch (fields.py:1113-1115)
is the reason for this one.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON
=============================================================================

* ``--fmad=false`` is CORRECTNESS, not tuning, and the option tuple is SPELLED
  HERE rather than imported so that loading this file by path cannot pick up a
  different one than a gate compiled. Both siblings measured their unguarded
  controls diverging (384/384 and 138/207); this body is their union plus the
  gathered subtraction chains, so the guard is a precondition here too -- carried
  as the gate's own control rather than inherited.
* SIGNED ZERO: CUDA lowers ``-x`` to ``neg.f32``, not to ``0.0f - x``. No unary
  minus appears on any float path here regardless; the parity weight is a bound
  argument.
* DIVISION: none anywhere on this path. (That is also what rules the
  ``f_w / inv_eps`` fusion out.)
* THE DEVICE SOURCE IS PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a
  bare ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE
  encoding. The tests scan AND encode every emitted source.

=============================================================================
IMPORTABLE WITHOUT CUPY, AND NOT CERTIFIED UNTIL THE RECORD SAYS SO
=============================================================================

``cupy`` is imported INSIDE the launcher, never at module scope, because the
predicate is consumed by the coverage census, which runs on a laptop.

NOTHING DISPATCHES THIS. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions), so a
predicate returning True licenses a MEASUREMENT and not a production step. What
has and has not been measured on a device is :data:`DISPERSIVE_OFFDIAG_ADMISSION`,
and nothing here may be read as a device verdict while that record's ``host``
field is None.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import folded_offdiag_kernels as _folded
from . import offdiag_emitter as _flat
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)

# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on
# ---------------------------------------------------------------------------
#
# Imported BY VALUE from whichever family already owns the fact, so there is one
# home per fact and a test can assert the identity rather than diff two spellings.

#: The one kernel this family emits. Its SOURCE varies with the row mask AND the
#: pole arity; its NAME does not, because it is one family and NVRTC compiles one
#: module per source. Deliberately distinct from both siblings' names: a second
#: kernel wearing a certified one's name would make any NVRTC binary observation
#: ambiguous about which body it saw.
KERNEL_NAME = "update_E_pml_real_folded_offdiag_dispersive"

#: Byte-identical to the array path with a device verdict behind it. The evidence is
#: :data:`DISPERSIVE_OFFDIAG_ADMISSION` and ``certification.json``'s
#: ``cuda_dispersive_offdiag_2026-08-20`` block, which names the same two artifacts;
#: a name here without a record block is the failure ``test_kernel_partition.py``
#: exists to catch. Spelled as a LITERAL rather than as ``(KERNEL_NAME,)`` because
#: the partition readers evaluate this with ``ast.literal_eval`` on a host with no
#: CuPy, where a name reference is not evaluable; the walk pins it against the
#: kernel declaration in the emitted device text, so the two cannot drift.
CERTIFIED_KERNELS = ("update_E_pml_real_folded_offdiag_dispersive",)

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: ``stepping.MIRROR_SOURCE_INDEX``, through the family that already restates it.
MIRROR_SOURCE_INDEX: int = _folded.MIRROR_SOURCE_INDEX

#: The three boundary codes, and the mirror one, both the folded family's.
BC_MIRROR_CODE: int = _folded.BC_MIRROR_CODE
FOLDED_BC_CODES: Dict[str, int] = dict(_folded.FOLDED_BC_CODES)

#: The components this sub-step writes, with each component's D volume and OWN
#: axis. ``offdiag_emitter.E_TERMS``, imported through the folded family.
E_TERMS: Tuple[Tuple[str, str, int], ...] = _folded.E_TERMS

#: The D volume of each axis, and the kernel parameter name of each row
#: coefficient in :data:`coverage.OFFDIAG_ROW_SLOTS` order.
PARTNER_VOLUMES: Tuple[str, str, str] = _folded.PARTNER_VOLUMES
ROW_PARAMETERS: Tuple[str, ...] = _folded.ROW_PARAMETERS

#: Every row mask this family can emit -- the 63 non-empty subsets. An all-dead
#: mask belongs to the plain dispersive constitutive kernel
#: (``dispersive_kernels.update_E_pml_real_dispersive``), which is exactly
#: the seam ``covers_real_pml_dispersive_constitutive`` refuses this family on.
LIVE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = _folded.LIVE_ROW_MASKS

#: The largest per-component contributor count this family will serve. MEASURED,
#: not chosen: ``gate_cuda_dispersive_offdiag.py`` sweeps every arity in
#: :data:`POLE_COUNTS_SWEPT` and nothing above this, so a triple inside it has
#: been run on a device and a triple past it is refused BY NAME rather than
#: served on an extrapolation.
#:
#: WHY IT IS THIS FAMILY'S OWN NUMBER AND NOT ``dispersive_kernels.POLE_COUNT_CAP``.
#: That constant is 6 because THAT gate swept 0..6 on an ELEMENT-WISE body. This
#: body gathers each pole at four addresses per term, so its cost per arity is a
#: different curve and its sweep is a different sweep. Reading the sibling's cap
#: would widen this family to arities nobody ran here -- the exact move
#: ``ADE_FOLD_PLANES_SWEPT`` was cut loose from its sibling record to stop. The
#: corpus's own maximum on this intersection is ONE.
POLE_COUNT_CAP: int = 3

#: The arities the gate sweeps, and the reason each is in the list.
POLE_COUNTS_SWEPT: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),   # the reduction: the emitted tree is the folded off-diagonal
                 # family's, which the gate MEASURES rather than claims
    (1, 1, 1),   # absorbed_power_density.py -- the corpus point, and the arity
                 # at which sum_then_subtract and reverse_pole_order are
                 # PROVABLY inert
    (2, 2, 2),   # the smallest arity that can see either of those two
    (2, 0, 3),   # mixed, including the ALIASING branch (fields.py:1113-1115) on
                 # one component only, beside two that carry a chain
)

#: SIMULTANEOUS MIRROR PLANES THIS FAMILY'S GATE HAS SCORED. A cap is a fact
#: about a gate and a gate answers for one family, so this is not read off any
#: sibling record. The corpus row carries ONE plane.
FOLD_PLANES_SWEPT: int = 3

#: Lanes per block. The geometry every certified hand-CUDA family uses. The
#: gathered subtraction chains add loads, not neighbours, so there is no tile to
#: shape beyond the off-diagonal stencil's own.
_DISPERSIVE_OFFDIAG_THREADS = 256

#: NVRTC compile options -- CORRECTNESS, not performance. Spelled here rather
#: than imported so that loading this file by path cannot pick up a different
#: tuple than the one a gate compiled; a test pins it equal to both siblings'.
_COMPILE_OPTIONS = ('--fmad=false',)


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# THE HELPERS TWO CERTIFIED FAMILIES ALREADY OWN ARE READ OUT OF THEIR SOURCES,
# NOT COPIED -- the route ``folded_offdiag_kernels._sibling_block`` takes, widened
# to two donor texts because this family's ancestry is two families rather than
# one. A test asserts every extracted block still appears verbatim in its owner
# and in every emitted source.

def _device_block(text: str, name: str) -> str:
    """One ``__device__`` helper, with its comment, out of a donor prelude.

    Anchored on the DECLARATION line and terminated by the first line that is
    exactly ``}`` -- the shape every helper in both donor preludes has. The
    preceding comment block travels with it, because the comment is where the
    transcription citation lives and a helper carried without its citation is a
    helper whose provenance has to be remembered.

    RAISES if the block cannot be found. A silent fallback to a local copy is the
    one failure this indirection exists to prevent.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("__device__ __forceinline__") and f" {name}(" in line:
            start = index
            break
    if start is None:
        raise RuntimeError(
            f"the donor prelude no longer declares {name!r}; this family reads "
            f"that helper out of a certified sibling rather than copying it, so "
            f"a rename there must fail here instead of forking the arithmetic")
    head = start
    while head > 0 and lines[head - 1].startswith("//"):
        head -= 1
    end = start
    while end < len(lines) and lines[end] != "}":
        end += 1
    if end >= len(lines):
        raise RuntimeError(f"{name!r} in the donor prelude has no closing brace "
                           f"on its own line")
    return "\n".join(lines[head:end + 1])


def _define_block(text: str, name: str) -> str:
    """One ``#define``, with its comment, out of a donor prelude.

    Same rule as :func:`_device_block` and for the same reason: ``MIRROR_ROW``
    carries the whole derivation of which stored row a mirror ghost reflects
    from, and a bare ``#define`` copied without it is a number nobody can check.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"#define {name} "):
            start = index
            break
    if start is None:
        raise RuntimeError(
            f"the donor prelude no longer defines {name!r}; this family reads "
            f"that constant out of a certified sibling rather than copying it")
    head = start
    while head > 0 and lines[head - 1].startswith("//"):
        head -= 1
    return "\n".join(lines[head:start + 1])


#: Helpers taken verbatim from ``offdiag_emitter.PRELUDE``. ``ghosted`` stays for
#: the COEFFICIENT side of every term (``ghosted(u, up)``): a chi1inv row carries
#: no poles and never did, so its ghosted load is the certified one unchanged.
FLAT_HELPERS: Tuple[str, ...] = ("flat", "ghosted", "constitutive_apply")

#: Constants and helpers taken verbatim from ``folded_offdiag_kernels``. The two
#: coordinate rules ARE the fold, and this family inherits them unchanged because
#: the fold acts on the shift helpers while the poles act on the values they move.
FOLDED_DEFINES: Tuple[str, ...] = ("BC_PERIODIC", "BC_METALLIC", "BC_MIRROR",
                                   "MIRROR_ROW")
FOLDED_HELPERS: Tuple[str, ...] = ("coord_up", "coord_dn")


def prelude_provenance() -> Tuple[Tuple[str, str, str], ...]:
    """``(kind, name, owning module)`` for every borrowed block, as data.

    Named so a test can walk it rather than restate the list, and so a reader can
    see at a glance that nothing in the shared half of this device source is
    typed twice anywhere in the package.
    """
    return (tuple(("helper", name, "offdiag_emitter") for name in FLAT_HELPERS)
            + tuple(("define", name, "folded_offdiag_kernels")
                    for name in FOLDED_DEFINES)
            + tuple(("helper", name, "folded_offdiag_kernels")
                    for name in FOLDED_HELPERS))


def _borrowed_prelude() -> str:
    """The shared half of the device source: two donors, nothing retyped."""
    parts = [_device_block(_flat.PRELUDE, name) for name in FLAT_HELPERS]
    parts.extend(_define_block(_folded._OWN_PRELUDE, name)
                 for name in FOLDED_DEFINES)
    parts.extend(_device_block(_folded._OWN_PRELUDE, name)
                 for name in FOLDED_HELPERS)
    return "\n\n".join(parts)


#: The one line ``ghosted_mirror`` and this family's mirror helper share, pinned
#: as data so a test can weld the two texts together. The mirror WEIGHT rule is
#: the folded family's and must not fork; only the loaded VALUE changes here.
MIRROR_SELECT_LINE = "    return mg ? (w * value) : value;"

#: The term's return expression, character for character
#: ``folded_offdiag_kernels``' ``folded_offdiag_term``'s. The association is the
#: certified one and the only edit this family makes to a term is to the two
#: PAIR lines above it.
TERM_RETURN_LINE = ("    return 0.25f * ((near_pair * u[home])"
                    " + (far_pair * ghosted(u, up)));")


def pole_parameter_names(counts: Sequence[int]) -> Tuple[str, ...]:
    """The pole pointer parameter names, in the order the signature declares them.

    ``dispersive_kernels.pole_parameter_names``' spelling and order --
    ``E_TERMS`` order, then chain position -- so a launcher that binds one
    family's plan into the other's signature is a name mismatch at emit time
    rather than a silently permuted chain.
    """
    values = normalized_pole_counts(counts)
    names: List[str] = []
    for (target, _source, _axis), count in zip(E_TERMS, values):
        names.extend(f"P_{target}_{index}" for index in range(count))
    return tuple(names)


def normalized_pole_counts(counts: Sequence[int]) -> Tuple[int, int, int]:
    """Validate and canonicalize a ``(n_Ex, n_Ey, n_Ez)`` arity triple.

    Raises rather than clamping. A count past :data:`POLE_COUNT_CAP` has never
    been run on a device for THIS body, and emitting for it anyway is exactly the
    "admitted by argument alone" move this package refuses everywhere else.
    """
    values = tuple(int(value) for value in counts)
    if len(values) != 3:
        raise ValueError(
            f"the pole arity is one count per E component, got {len(values)}: "
            f"{counts!r}")
    for (target, _source, _axis), value in zip(E_TERMS, values):
        if value < 0:
            raise ValueError(f"{target} carries a negative pole count {value}")
        if value > POLE_COUNT_CAP:
            raise ValueError(
                f"{target} carries {value} contributors and POLE_COUNT_CAP is "
                f"{POLE_COUNT_CAP}; gate_cuda_dispersive_offdiag.py has swept "
                f"{sorted(set(POLE_COUNTS_SWEPT))} and nothing above. Raising "
                f"the cap means sweeping it, not editing this number.")
    return values  # type: ignore[return-value]


def normalized_row_mask(row_mask: Sequence[int]) -> Tuple[int, ...]:
    """Validate and canonicalize a row mask -- the certified emitter's, imported.

    Its refusal message names the PLAIN constitutive kernel. That message is
    slightly off here (a zero-slot dispersive run belongs to
    ``dispersive_kernels``' arm 1, not to the non-dispersive constitutive one),
    and it is still the right function to call: forking it to reword the message
    would fork the VALIDATION, which is the part that matters. The predicate
    below names the right family in its own refusal.
    """
    return _flat.normalized_row_mask(row_mask)


def _pole_signature(component: str, count: int) -> str:
    """``, const float* P_Ex_0, ...`` for one component's chain, or ``""``."""
    return "".join(f", const float* P_{component}_{index}"
                   for index in range(count))


def _pole_arguments(component: str, count: int) -> str:
    """``, P_Ex_0, ...`` -- the same chain at a call site, same order."""
    return "".join(f", P_{component}_{index}" for index in range(count))


def _dmp_helper(component: str, count: int) -> str:
    """``dmp_<component>``: ``Fields.displacement_minus_polarization_volumes``.

    THE CHAIN IS SEQUENTIAL AND LEFT TO RIGHT, one subtraction per contributor,
    in ``fields.polarizations`` order filtered by ``drives`` -- fields.py:1130-1136,
    where each ``state.subtract_into`` rounds to float32 exactly as this register
    does. ``D - (P0 + P1)`` is a DIFFERENT float32 number and is the form a reader
    of the phrase "D minus the sum of the polarizations" writes by accident.

    AT ZERO CONTRIBUTORS THE BODY IS A BARE LOAD, because the array path forms no
    subtraction at all and hands back the D array ITSELF (fields.py:1113-1115).
    That is what makes the arity-(0,0,0) tree the folded off-diagonal family's by
    CONSTRUCTION rather than by rounding, and the gate measures the reduction
    rather than claiming it.
    """
    signature = (f"__device__ __forceinline__ float dmp_{component}(\n"
                 f"    const float* g{_pole_signature(component, count)},"
                 f" int index\n) {{")
    if count == 0:
        return (f"// dmp_{component}: Fields.displacement_minus_polarization_volumes\n"
                f"// (fields.py:1107-1138) for {component}. NOTHING DRIVES IT, so the array\n"
                f"// path returns the D array itself, aliased and uncopied\n"
                f"// (fields.py:1113-1115), and no subtraction is formed here either.\n"
                f"{signature}\n    return g[index];\n}}")
    lines = [f"// dmp_{component}: Fields.displacement_minus_polarization_volumes",
             f"// (fields.py:1107-1138) for {component}. ONE SUBTRACTION PER CONTRIBUTOR,",
             "// in fields.polarizations order filtered by drives(), each rounding to",
             "// float32 where the array path's scratch does (fields.py:1130-1136).",
             "// ((D - P0) - P1) - P2, never D - (P0 + P1 + P2).",
             signature,
             "    float value = g[index];"]
    lines.extend(f"    value = value - P_{component}_{index}[index];"
                 for index in range(count))
    lines.append("    return value;")
    lines.append("}")
    return "\n".join(lines)


def _dmp_ghosted_helper(component: str, count: int) -> str:
    """``ghosted`` with the load replaced by the chain -- the metallic zero ghost.

    The certified ``ghosted`` (offdiag_emitter.PRELUDE, borrowed above and still
    used for the COEFFICIENT) reads one pointer. This reads a chain, and the
    ghost value is unchanged: index -1 is the exact ``+0.0f`` the array path
    writes into that plane (``_shift_up`` :1782-1784, ``_shift_down`` :1826-1828),
    and it is a plain zero rather than ``0 - sum P`` because the array path SHIFTS
    the already-formed ``D - sum P`` volume and its ghost plane is a literal zero.
    """
    return (
        f"// ghosted, with the load replaced by {component}'s pole chain. Index -1 is\n"
        f"// the metallic zero ghost, which the array path writes into that plane as an\n"
        f"// exact +0.0 AFTER the D - sum P volume is formed -- so it is a plain zero,\n"
        f"// not a subtracted one (stepping.py:1782-1784, :1826-1828).\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 1782-1784->1829-1831, 1826-1828->1873-1875
        f"__device__ __forceinline__ float dmp_ghosted_{component}(\n"
        f"    const float* g{_pole_signature(component, count)}, int index\n"
        f") {{\n"
        f"    return (index < 0) ? 0.0f : dmp_{component}(g"
        f"{_pole_arguments(component, count)}, index);\n"
        f"}}")


def _dmp_ghosted_mirror_helper(component: str, count: int) -> str:
    """``ghosted_mirror`` with the load replaced by the chain.

    THE WEIGHT RULE IS THE FOLDED FAMILY'S, LINE FOR LINE, and the last line is
    pinned equal to it by :data:`MIRROR_SELECT_LINE`. ``mg`` is 1 only at face 0
    of an axis whose code is ``BC_MIRROR``, derived from ``bc_*`` in ONE place in
    the kernel body, so the index redirect in ``coord_dn`` and this weight cannot
    disagree about which lane is the ghost.

    THE PARITY MULTIPLIES THE FORMED CHAIN, NOT THE RAW D. The array path shifts
    ``volumes[partner]`` -- which is already ``D - sum P`` -- and writes
    ``parity * value`` into face 0 (stepping.py:1870-1872), so the weight sits
    OUTSIDE the subtraction. Weighting ``D`` and then subtracting unweighted poles
    is a different number wherever the parity is -1, which is every EVEN plane.
    """
    return (
        f"// ghosted_mirror, with the load replaced by {component}'s pole chain. The\n"
        f"// array path shifts the already-formed D - sum P volume and weights face 0 by\n"
        f"// the plane's parity (stepping.py:1823-1825), so the parity multiplies the\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 1823-1825->1870-1872
        f"// FORMED CHAIN and never the raw D. The select below is folded_offdiag_kernels\n"
        f"// .ghosted_mirror's own last line, pinned equal to it by a test.\n"
        f"__device__ __forceinline__ float dmp_ghosted_mirror_{component}(\n"
        f"    const float* g{_pole_signature(component, count)}, int index,\n"
        f"    int mg, float w\n"
        f") {{\n"
        f"    if (index < 0) return 0.0f;\n"
        f"    float value = dmp_{component}(g"
        f"{_pole_arguments(component, count)}, index);\n"
        f"{MIRROR_SELECT_LINE}\n"
        f"}}")


def _term_helper(component: str, count: int) -> str:
    """One partner's OFFDIAG term, with all four field reads as ``dmp`` chains.

    ``folded_offdiag_kernels.folded_offdiag_term`` with each of its four loads on
    the FIELD side routed through this partner's chain, and nothing else touched.
    MEEP step_generic.cpp:582-583 as ``stepping._offdiagonal_terms``
    (stepping.py:1240-1249) associates it::

        0.25*((g[i] + w*g[i-sx])*u[i] + (g[i+s] + w*g[(i+s)-sx])*u[i+s])

    with ``g`` the PARTNER's ``D - sum P`` volume (stepping.py:1240). THE FOUR
    GATHERED READS ARE THE MIDDLE REFUSAL: it is not enough to subtract the
    polarization from the diagonal term, because every one of these addresses is
    a neighbouring cell of ANOTHER component's chain.

    BOTH GHOSTED DOWN LOADS TAKE THE SAME WEIGHT AND THE SAME REDIRECT: ``down``
    and ``corner`` differ only in the OWN-axis coordinate, so the two hit the
    partner-axis ghost plane together and ``mg`` is one predicate for both.

    THE UP LEG IS UNWEIGHTED AND UNREDIRECTED. It is the row component's OWN
    axis, where the array path serves an exact zero on a fold (see the borrowed
    ``coord_up``), so it stays the certified call character for character apart
    from the chain.

    THE COEFFICIENT SIDE IS UNCHANGED, ``ghosted(u, up)``: a chi1inv row carries
    no poles, and the multiply still sits BETWEEN the two shifts because MEEP
    registers the off-diagonal entry at the component's Yee site minus half a cell
    along its own axis (anisotropic_averaging.cpp:248-257, ``here - shift1``).
    """
    poles_sig = _pole_signature(component, count)
    poles_arg = _pole_arguments(component, count)
    return (
        f"// One partner's OFFDIAG term whose FOUR field reads are {component}'s\n"
        f"// D - sum P chain (stepping.py:1211-1220 with volumes[partner],\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 1211-1220->1240-1249
        f"// fields.py:1107-1138). The coefficient side is the certified ghosted load:\n"
        f"// a chi1inv row carries no poles. The return line is\n"
        f"// folded_offdiag_kernels.folded_offdiag_term's, character for character.\n"
        f"__device__ __forceinline__ float folded_dispersive_offdiag_term_{component}(\n"
        f"    const float* g{poles_sig}, const float* u,\n"
        f"    int home, int down, int up, int corner, int mg, float w\n"
        f") {{\n"
        f"    float near_pair = dmp_{component}(g{poles_arg}, home)\n"
        f"        + dmp_ghosted_mirror_{component}(g{poles_arg}, down, mg, w);\n"
        f"    float far_pair = dmp_ghosted_{component}(g{poles_arg}, up)\n"
        f"        + dmp_ghosted_mirror_{component}(g{poles_arg}, corner, mg, w);\n"
        f"{TERM_RETURN_LINE}\n"
        f"}}")


def _live_partner_components(row_mask: Sequence[int]) -> Tuple[str, ...]:
    """Which components appear as a live slot's PARTNER, in E_TERMS order.

    Only those need the ghosted/mirror/term trio. Emitting the trio for a
    component nothing couples would leave three device functions nothing calls --
    dead code a later edit can wire up by accident.
    """
    partners = set()
    for slot, flag in enumerate(row_mask):
        if not flag:
            continue
        component = slot // 2
        offset = slot % 2
        partners.add(_coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset])
    return tuple(E_TERMS[axis][0] for axis in sorted(partners))


def _own_prelude(row_mask: Sequence[int], counts: Sequence[int]) -> str:
    """This family's own device code, at one row mask and one arity."""
    mask = normalized_row_mask(row_mask)
    values = normalized_pole_counts(counts)
    by_name = {term[0]: count for term, count in zip(E_TERMS, values)}
    parts: List[str] = []
    for (target, _source, _axis), count in zip(E_TERMS, values):
        parts.append(_dmp_helper(target, count))
    for target in _live_partner_components(mask):
        count = by_name[target]
        parts.append(_dmp_ghosted_helper(target, count))
        parts.append(_dmp_ghosted_mirror_helper(target, count))
        parts.append(_term_helper(target, count))
    return "\n\n".join(parts)


# THE SIGNATURE CARRIES ONLY THE LIVE ROW COEFFICIENTS AND ONLY THE LIVE POLE
# POINTERS -- both emitters' choice, for the same reason: binding a dead slot to
# some other volume leaves a pointer aimed at an array the kernel must never
# touch, which a later edit can read by accident.
#
# THE THREE GHOST WEIGHTS ARE ALWAYS BOUND, even with no axis folded. They are
# scalars, not pointers, so there is no array to aim wrongly; and the launcher
# validates each to be exactly +1.0f or -1.0f on a folded axis, where a NaN would
# otherwise launch rather than raise.
#
# WHICH POINTERS CARRY __restrict__. The six outputs and the six PML coefficient
# vectors do, as in both siblings. The POLE pointers do, on
# ``dispersive_kernels``' grounds -- two susceptibilities never share a P buffer
# (dispersion.py:645-647) -- and the launcher and the predicate BOTH check that
# every P is distinct from every other bound volume, which is what makes the
# promise one the caller can keep. The D volumes and the inverse-epsilon volumes
# do NOT: an isotropic install hands the same inverse permittivity three times
# (fields.py:1321-1326) and a row coefficient may legally alias a D volume.
TEMPLATE = r'''
extern "C" __global__ void update_E_pml_real_folded_offdiag_dispersive(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* Dx, const float* Dy, const float* Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
__POLE_PARAMETERS____ROW_PARAMETERS__
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z,
    int bc_x, int bc_y, int bc_z,
    int wm_x, int wm_y, int wm_z,
    float gw_x, float gw_y, float gw_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int nyz = ny * nz;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Both neighbour coordinates on every axis: one term reads an axis DOWN (the
    // partner's) and another reads an axis UP (its own), and a component's two
    // terms take different partners, so every axis can be needed either way.
    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);
    int dj = coord_dn(j, ny, bc_y), uj = coord_up(j, ny, bc_y);
    int dk = coord_dn(k, nz, bc_z), uk = coord_up(k, nz, bc_z);

    // Wall-plane predicates for the coupling mask. FACE 0 ONLY: the high wall is
    // already the shift-up zero ghost, and stepping._mask_metallic_wall_coupling
    // writes only _face(axis, 0) (stepping.py:1254).
    int at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

    // The mirror ghost lane: face 0 of a folded axis, the one plane _shift_down
    // weights (stepping.py:1823-1825). DERIVED FROM bc_* AND NOWHERE ELSE, so the
    // index redirect in coord_dn and the weight in dmp_ghosted_mirror_* cannot
    // disagree about which lane is the ghost -- a disagreement would read stored
    // row MIRROR_ROW without the parity, or apply the parity to an ordinary
    // neighbour, and both are a plane of wrong values rather than a crash.
    //
    // A FOLDED AXIS IS NEVER ALSO WALL-MASKED: _mask_metallic_wall_coupling asks
    // is_metallic AND NOT is_mirrored (stepping.py:1253), and
    // coverage.offdiag_wall_mask_flags asks the same question of the same grid.
    // The launcher refuses the combination rather than relying on that.
    int mg_x = (bc_x == BC_MIRROR) && at_x;
    int mg_y = (bc_y == BC_MIRROR) && at_y;
    int mg_z = (bc_z == BC_MIRROR) && at_z;

    // The three components are independent, and that is read off the loop rather
    // than assumed: the coupling reads D, the pole buffers and the inverse
    // permittivity, and the tail writes E and f_w_E. Four disjoint sets, so the
    // order the array path runs them in (stepping.py:969-989) is not a data
    // dependence -- which is what makes a NEIGHBOUR read across components safe
    // here, and what makes recomputing a chain per read the same bits as the
    // array path's materialized volume.
__SRC_Ex__

__SRC_Ey__

__SRC_Ez__
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1254->1283, 1823-1825->1870-1872, 1253->1282, 969-989->998-1018


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------
#
# The three index expressions per term are ASSEMBLED from a per-axis table rather
# than written out eighteen times, and the ghost lane and weight come from the
# SAME ``partner_axis`` number that picks the index -- so a term cannot weight one
# axis's ghost while shifting another's. ``folded_offdiag_kernels``' tables,
# imported by value.

_HOME_COORDS = ("i", "j", "k")
_UP_COORDS = ("ui", "uj", "uk")
_DOWN_COORDS = ("di", "dj", "dk")
_WALL_FLAGS = ("wm_x", "wm_y", "wm_z")
_WALL_PREDICATES = ("at_x", "at_y", "at_z")
_GHOST_LANES = ("mg_x", "mg_y", "mg_z")
_GHOST_WEIGHTS = ("gw_x", "gw_y", "gw_z")
_OWN_AXIS_INDEX = ("i", "j", "k")


def _flat_index(shifted: Dict[int, str]) -> str:
    """``flat(...)`` with the named per-axis substitutions applied."""
    coords = [shifted.get(axis, _HOME_COORDS[axis]) for axis in range(3)]
    return f"flat({coords[0]}, {coords[1]}, {coords[2]}, nyz, nz)"


def _term_lines(component: int, offset: int, counts: Sequence[int]) -> List[str]:
    """One partner's term for one component, as source lines.

    ``offset`` is 0 for MEEP's ``cycle_direction(dim, d_ec, 1)`` and 1 for
    ``(..., 2)`` (stepping.py:1235-1237). The partner axis, the coefficient slot,
    the partner volume, the partner's POLE CHAIN, all three shifted indices AND
    the ghost lane/weight pair come from that one number, so a mispairing would
    have to be introduced deliberately rather than by a typo.
    """
    own_axis = E_TERMS[component][2]
    partner_axis = _coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
    coefficient = ROW_PARAMETERS[2 * component + offset]
    volume = PARTNER_VOLUMES[partner_axis]
    partner_name = E_TERMS[partner_axis][0]
    poles = _pole_arguments(partner_name, int(counts[partner_axis]))
    tag = f"{E_TERMS[component][0]}_{offset}"
    down = _flat_index({partner_axis: _DOWN_COORDS[partner_axis]})
    up = _flat_index({own_axis: _UP_COORDS[own_axis]})
    corner = _flat_index({own_axis: _UP_COORDS[own_axis],
                          partner_axis: _DOWN_COORDS[partner_axis]})
    return [
        f"    // partner ({volume} - sum P[{partner_name}]): the pair half a cell "
        f"DOWN axis {partner_axis},",
        f"    // then the product half a cell UP axis {own_axis} (its own). The "
        f"mirror ghost,",
        f"    // if axis {partner_axis} carries one, is on the DOWN leg and "
        f"weights the FORMED chain.",
        f"    float term_{tag} = folded_dispersive_offdiag_term_{partner_name}(",
        f"        {volume}{poles}, {coefficient}, idx,",
        f"        {down},",
        f"        {up},",
        f"        {corner},",
        f"        {_GHOST_LANES[partner_axis]}, {_GHOST_WEIGHTS[partner_axis]});",
    ]


def _wall_mask_lines(component: int) -> List[str]:
    """``stepping._mask_metallic_wall_coupling`` for one component's total.

    Face 0 of every axis whose Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), ascending -- the mask's own loop
    order (stepping.py:1279-1283). Unchanged from both sibling emitters, because
    the mask itself is unchanged: it asks the grid's DECLARATION and knows nothing
    about poles.
    """
    total = f"total_{E_TERMS[component][0]}"
    return [f"    {total} = ({_WALL_FLAGS[axis]} && {_WALL_PREDICATES[axis]})"
            f" ? 0.0f : {total};"
            for axis in _coverage.OFFDIAG_WALL_MASK_AXES[component]]


def _component_source(component: int, row_mask: Sequence[int],
                      counts: Sequence[int]) -> str:
    """One component's whole block, on the arm its two row slots select.

    Four arms: none, offset-1 only, offset-2 only, both. THE NONE ARM IS THE
    PLAIN DISPERSIVE CONSTITUTIVE BODY -- ``src = (D - sum P) * inv_eps`` and the
    same tail -- because ``_offdiagonal_terms`` returns None for a component with
    no surviving row and ``update_E`` then never forms a ``+ 0`` copy
    (stepping.py:1222-1227, :1006).
    """
    name, source, own_axis = E_TERMS[component]
    axis_letter = "xyz"[own_axis]
    index_letter = _OWN_AXIS_INDEX[own_axis]
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    poles = _pole_arguments(name, int(counts[own_axis]))

    lines = [f"    // --- {name}: own axis {axis_letter}; source {source} - sum "
             f"P[{name}]; dsigw = {axis_letter}.",
             f"    float gs_{name} = dmp_{name}({source}{poles}, idx);",
             f"    float us_{name} = inv_eps_{name}[idx];"]
    if not live:
        lines.append(f"    float src_{name} = gs_{name} * us_{name};")
    else:
        for offset in live:
            lines.extend(_term_lines(component, offset, counts))
        # ``total`` accumulates offset 1 then offset 2 (stepping.py:1250). Both
        # additions are bitwise commutative in float32, so this is transcription
        # fidelity rather than a pinned grouping.
        lines.append(f"    float total_{name} = term_{name}_{live[0]};")
        for offset in live[1:]:
            lines.append(f"    total_{name} = total_{name} + term_{name}_{offset};")
        # The mask runs BEFORE the row sum: stepping.py:1252 precedes :1006-1007.
        lines.extend(_wall_mask_lines(component))
        lines.append(f"    float src_{name} = (gs_{name} * us_{name})"
                     f" + total_{name};")
    lines.append(f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                 f" kps_{axis_letter}[{index_letter}],"
                 f" kms_{axis_letter}[{index_letter}]);")
    return "\n".join(lines)


def _row_parameter_lines(row_mask: Sequence[int]) -> str:
    """The live coefficient parameters, in :data:`coverage.OFFDIAG_ROW_SLOTS` order."""
    return "\n".join(f"    const float* {ROW_PARAMETERS[slot]},"
                     for slot, flag in enumerate(row_mask) if flag)


def _pole_parameter_lines(counts: Sequence[int]) -> str:
    """The live pole parameters, in ``E_TERMS`` order then chain position."""
    names = pole_parameter_names(counts)
    if not names:
        return ""
    return "\n".join(f"    const float* __restrict__ {name}," for name in names)


def dispersive_offdiag_source(row_mask: Sequence[int],
                              counts: Sequence[int]) -> str:
    """The device source specialized on one row mask and one pole arity.

    THE BOUNDARY, WALL and GHOST-WEIGHT axes are NOT baked in -- they are runtime
    arguments -- so the row mask and the arity are this family's only source axes,
    exactly as in both siblings. The corpus drives ONE point of that product.
    """
    mask = normalized_row_mask(row_mask)
    values = normalized_pole_counts(counts)
    poles = _pole_parameter_lines(values)
    rows = _row_parameter_lines(mask)
    body = TEMPLATE.replace("__POLE_PARAMETERS__", poles + "\n" if poles else "")
    body = body.replace("__ROW_PARAMETERS__", rows)
    for component, term in enumerate(E_TERMS):
        body = body.replace(f"__SRC_{term[0]}__",
                            _component_source(component, mask, values))
    return _borrowed_prelude() + "\n\n" + _own_prelude(mask, values) + "\n" + body


def prelude(row_mask: Sequence[int], counts: Sequence[int]) -> str:
    """The whole device prelude -- borrowed blocks first, then this family's."""
    return (_borrowed_prelude() + "\n\n"
            + _own_prelude(normalized_row_mask(row_mask),
                           normalized_pole_counts(counts)))


def source_digest(row_mask: Sequence[int], counts: Sequence[int]) -> str:
    """sha256 of one emitted body -- what a record pins instead of a file hash."""
    return hashlib.sha256(
        dispersive_offdiag_source(row_mask, counts).encode("utf-8")).hexdigest()


def corpus_digest() -> str:
    """One sha256 over every source this family can emit at a SWEPT arity.

    A file hash stops matching when a docstring gains a comma; this pins the
    strings NVRTC actually compiles. The product is 63 masks x
    ``len(POLE_COUNTS_SWEPT)`` arities -- too many to pin one digest each without
    burying the record, and not too many to hash, so a single changed character
    anywhere in the emitter moves one value. The per-source digests of the
    (mask, arity) pairs a gate launches belong in that gate's artifact.
    """
    digest = hashlib.sha256()
    for mask in LIVE_ROW_MASKS:
        for counts in POLE_COUNTS_SWEPT:
            digest.update(repr((mask, counts)).encode("ascii"))
            digest.update(dispersive_offdiag_source(mask, counts).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: str) -> Any:
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


# ---------------------------------------------------------------------------
# The host-side facts a launch needs
# ---------------------------------------------------------------------------

def dispersive_offdiag_boundary_codes(grid: Any, pml: Any = None) -> Tuple[int, ...]:
    """The three ``bc_*`` arguments -- the folded family's reader, imported.

    One home for the mapping from ``stepping._boundary_kinds``' answer to this
    kernel's codes, shared with the family whose ``coord_dn`` this kernel borrows.
    A second spelling would be a second place for the code and the branch to
    disagree.
    """
    return _folded.folded_offdiag_boundary_codes(grid, pml)


def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """``gw_x``/``gw_y``/``gw_z`` -- the folded family's derivation, imported.

    Derived through ``fields.mirror_parity`` itself over there, and ``nan`` where
    a plane cannot be read, which the predicate names by clause. Not restated:
    the parity that weights the ghost is a property of the SHIFT, and the shift is
    the borrowed ``coord_dn``'s.
    """
    return _folded.mirror_ghost_weights(grid)


def dispersive_offdiag_constitutive_tables(pml: Any) -> dict:
    """The six HALF-INTEGER kps/kms views this sub-step reads.

    Through the folded family's reader, which asks
    :func:`coverage.constitutive_sub_lattice`, so no call site can pair the E side
    with the INTEGER tables -- a half-cell error in the absorber profile that is
    converged, smooth and wrong.
    """
    return _folded.folded_offdiag_constitutive_tables(pml)


def fold_axes(grid: Any) -> Tuple[int, ...]:
    """The folded axes, as a tuple. One reader, so a caller cannot ask differently."""
    return _folded.fold_axes(grid)


def resolve_pole_plan(fields: Any) -> Dict[str, List[Any]]:
    """``{'Ex': [P arrays, in order], ...}`` -- the chain, resolved RIGHT NOW.

    Transcribed from ``Fields.displacement_minus_polarization_volumes``
    (fields.py:1125): ``[state for state in self.polarizations if
    state.drives(component)]``, then ``state.P[component]``.

    RESOLVED PER LAUNCH, NEVER CACHED. ``PolarizationState.update`` rotates the
    three buffers (dispersion.py:689-691), so a plan built before ``update_P``
    names last step's arrays after it. That is stale in a way that still
    computes -- the field stays finite and the spectrum moves -- which is the
    worst kind.

    ONE SPELLING WITH ``dispersive_kernels``: the chain ORDER is the arithmetic
    here, not merely the membership, and two readers of ``fields.polarizations``
    would be two places for the order to drift.
    """
    from .dispersive_kernels import resolve_pole_plan as _resolve  # noqa: PLC0415

    return _resolve(fields)


def pole_counts_of(plan: Dict[str, Sequence[Any]]) -> Tuple[int, int, int]:
    """The ``(n_Ex, n_Ey, n_Ez)`` arity a plan launches at."""
    return tuple(len(plan.get(target, ()))  # type: ignore[return-value]
                 for target, _source, _axis in E_TERMS)


# ---------------------------------------------------------------------------
# THE COVERAGE PREDICATE
# ---------------------------------------------------------------------------

#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, and nothing more.
#:
#: ``host`` is the only thing that makes any number here a device verdict; a
#: ``host: None`` record means the predicate returning True licenses a MEASUREMENT
#: and nothing else.
#:
#: STILL NOT DISPATCHED. No module in ``meep_gpu/`` imports ``cuda_kernels``, so
#: nothing reaches this kernel from a production step; what this record licenses
#: is a coverage claim and a future planner branch, not a run.
#:
#: Filled 2026-08-20 by ``parity/meep_gpu/gate_cuda_dispersive_offdiag.py`` on
#: the GPU host's RTX A6000, GPU index 4 (verified physically empty by UUID before
#: launch), under BOTH float32 subnormal policies, each in its own process with
#: its own CuPy cache directory because CuPy's disk-cache key is computed above
#: the strip seam.
DISPERSIVE_OFFDIAG_ADMISSION: dict = {
    "kernel": KERNEL_NAME,
    "gate": "parity/meep_gpu/gate_cuda_dispersive_offdiag.py",
    "oracle": ("stepping.update_E on real Grid/Fields/PML with off-diagonal rows "
               "installed through Fields.set_epsilon_volumes and real "
               "PolarizationState objects registered, compared as uint32 words"),
    "specified_by": ("the four refusals on examples/absorbed_power_density.py in "
                     "parity/meep_gpu/results/"
                     "cuda_predicate_coverage_2026-08-20_final"),
    "recorded_utc": "2026-08-20T20:25:19Z",
    "host": "the GPU host, NVIDIA RTX A6000, GPU index 4, CuPy 13.5.1",
    "artifacts": ("parity/meep_gpu/results/cuda_dispersive_offdiag_2026-08-20b/"
                  "keep/gate.json",
                  "parity/meep_gpu/results/cuda_dispersive_offdiag_2026-08-20b/"
                  "flush/gate.json"),
    #: HOW THIS RECORD IS BOUND TO THOSE RUNS, and why it is NOT an artifact
    #: hash. A digest of ``gate.json`` inside this module is circular: the
    #: artifact's own ``source_sha256.txt`` records THIS FILE's bytes as of the
    #: run, so writing the artifact's digest here changes this file and
    #: invalidates the manifest that was supposed to bind it. The non-circular
    #: binding is the EMITTER CORPUS DIGEST -- one sha256 over every source this
    #: family can emit at a swept arity, carried in both artifacts and asserted
    #: equal to the shipped emitter by ``test_dispersive_offdiag_update_e.py``.
    #: That ties the verdict to the exact DEVICE CODE, which is what the verdict
    #: is about; a docstring edit moves the file hash and not the arithmetic, and
    #: this record is a docstring edit.
    "bound_by": "emitter_corpus_digest, carried in both artifacts",
    "emitter_corpus_digest":
        "0c37196dc22fc03adb3b12ff7927c05d1ea75b48d0d96d33d99b02f27e9232bb",
    "gate_sha256_at_run":
        "ec7cb2afcc63893166e4b45cbdd23f65e1cf4a45c1b62c6b83c69dbe1f97dfeb",
    "post_gate_record_edits": (
        "this DISPERSIVE_OFFDIAG_ADMISSION block was filled in after the run it "
        "describes; it touches no device string and no predicate clause, and the "
        "emitter corpus digest is unchanged across the edit",),
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    #: PER POLICY. The two produced identical case counts and identical verdicts,
    #: which is why one copy carries both; each ran in its own process.
    "cases_scored": 960,
    "cases_refused_as_vacuous": 0,
    "live_chain_single_launch_identical": "360/360",
    "live_chain_at_sixty_launches_identical": "72/72",
    "corpus_shape_identical": "32/32",
    "folded_identical": "384/384",
    "unfolded_identical": "96/96",
    #: THE REDUCTION, MEASURED rather than claimed: at arity (0, 0, 0) the emitted
    #: tree must be the folded off-diagonal family's, and unfolded it must be the
    #: certified off-diagonal emitter's. The device leg scores the outputs; the
    #: merge-bar slice compares the two trees on one frozen state.
    "zero_arity_reduction_identical": "120/120",
    "multi_pole_identical": "240/240",
    "by_arity": {"000": "120/120", "111": "120/120", "222": "120/120",
                 "203": "120/120"},
    "simultaneous_planes_identical": {"1": "288/288", "2": "48/48",
                                      "3": "48/48"},
    #: ``--fmad=false`` is CORRECTNESS here, measured rather than inherited: the
    #: unguarded control is 0/240 identical at the inexact courant.
    "guard_control_identical_at_inexact_courant": "0/240",
    "mutation_legs_as_required": "28/28",
    "mutation_legs": {"must_be_caught": 8, "discriminators": 18,
                      "null_confirmed": 2},
    #: The two that make the battery two-sided rather than a list of catches.
    "null_controls_confirmed": ("distribute_the_quarter",
                                "commute_the_near_product"),
    #: THE MIDDLE REFUSAL, armed as a defect against this kernel: subtract the
    #: polarization from the diagonal term and couple the RAW D volumes. Caught
    #: wherever a live chain is reachable, a NULL at arity (0, 0, 0) where there
    #: is nothing about ``D - sum P`` left to be wrong.
    "middle_refusal_armed_as_a_defect":
        "couple_the_raw_D: 90/120 caught, as_required",
    "release_verdict_shown_to_flip_against_a_planted_defect": True,
    "subnormal_band_is_non_vacuous": True,
    "certified": True,
    #: THE MEASURED ALTERNATIVE, and the answer to "a new kernel OR a measured
    #: fusion of two shipped ones". A fusion of two SHIPPED kernels does not
    #: exist: neither exposes ``D - sum P`` as an output. The only composition is
    #: one shipped kernel plus the ARRAY PATH -- materialize the three volumes,
    #: then launch ``update_E_pml_real_folded_offdiag`` with them bound in
    #: place of D. It reproduces the oracle EXACTLY (54/54), and it costs 234
    #: extra full-volume passes across those 54 cases (up to 3 scratch volumes
    #: live) where this kernel costs ZERO. Both numbers are in both artifacts.
    "composition_identical": "54/54",
    "composition_extra_full_volume_passes": 234,
    "composition_scratch_volumes_max": 3,
    "kernel_extra_full_volume_passes": 0,
    #: WHAT THE RECORD DOES NOT SAY. It is a verdict about the EMITTER at the
    #: corpus digest above, exercised on THREE of its 63 row masks and FOUR of its
    #: arities. A (mask, arity) pair outside those is covered by the emitter's own
    #: tests and by that digest, not by a byte gate. No throughput claim is made
    #: or possible: the box was shared.
    "row_masks_exercised": ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1),
                            (0, 0, 1, 0, 0, 0)),
    "arities_exercised": POLE_COUNTS_SWEPT,
    #: WHAT IT IS WORTH, recomputed 2026-08-20 on ONE tree with the union analyzer
    #: (positional path) -- the family present and absent on the SAME census, so
    #: the delta is attributable to this leg rather than to a tree that moved
    #: between two rounds.
    "corpus": {
        "census": ("parity/meep_gpu/results/"
                   "cuda_predicate_coverage_2026-08-20_dispersive_offdiag"),
        "rows": 186, "slots": 759,
        "union_without_this_family": 753,
        "union_with_this_family": 754,
        "slots_gained": 1,
        "rows_covered_at_every_sub_step_gained": 1,
        "overlaps_introduced": 0,
        #: THE ONE ROW, named: ``examples/absorbed_power_density.py``, 800 x 402
        #: x 1, one mirror plane on Y, one registered susceptibility, row mask
        #: (1, 0, 0, 1, 0, 0). Its other four slots were already served.
        "the_one_row": "examples/absorbed_power_density.py at update_E",
        #: WHAT REMAINS. Five slots, ALL complex64 off-diagonal update_E, which
        #: this family refuses by name in its first storage clause.
        "slots_still_unserved": 5,
        "slots_still_unserved_are_all": "complex64 storage",
    },
}


def _dispersion_reasons(fields: Any, xp: Any, shape: Tuple) -> List[str]:
    """Everything this family needs of the pole chain, as accumulated reasons.

    THE CHAIN IS READ THE WAY THE ARRAY PATH READS IT -- ``state.drives(c)``
    filtered over ``fields.polarizations`` IN ORDER -- because the order is the
    arithmetic here, not merely the membership.

    ``P_prev`` IS NOT CHECKED and that is deliberate: this sub-step reads ``P``
    alone (fields.py:1134). ``P_prev`` belongs to ``update_P``, which is
    ``cuda_ade``'s slot and has its own predicate.
    """
    reasons: List[str] = []
    states = tuple(getattr(fields, "polarizations", ()) or ())
    addresses: Dict[int, str] = {}
    for target, _source, _axis in E_TERMS:
        chain = 0
        for index, state in enumerate(states):
            drives = getattr(state, "drives", None)
            try:
                if not (callable(drives) and drives(target)):
                    continue
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
                reasons.append(
                    f"polarization {index}.drives({target!r}) raised {exc!r}")
                continue
            kind = getattr(getattr(state, "susceptibility", None), "kind", None)
            if kind not in _coverage.COVERED_SUSCEPTIBILITY_KINDS:
                # NOT this family's arithmetic but its INPUT: a kind outside the
                # Lorentz/Drude pair advances P by a different difference
                # equation, and a P this kernel subtracted at four gathered
                # addresses would be a number no transcription here produced.
                reasons.append(
                    f"polarization {index} drives {target} with kind {kind!r}, "
                    f"outside {_coverage.COVERED_SUSCEPTIBILITY_KINDS}")
                continue
            array = (getattr(state, "P", {}) or {}).get(target)
            problem = _coverage._array_problem(
                f"polarization {index}.P[{target!r}]", array, xp, shape)
            if problem is not None:
                reasons.append(problem)
                continue
            address = _coverage._base_address(array)
            if address is None:
                reasons.append(
                    f"polarization {index}.P[{target!r}] exposes no readable base "
                    f"address; the restrict promise cannot be shown to hold")
                continue
            label = f"P_{target}_{chain}"
            if address in addresses:
                reasons.append(
                    f"{label} aliases {addresses[address]}; the pole pointers "
                    f"carry __restrict__ and two susceptibilities never share a "
                    f"P buffer (dispersion.py:645-647)")
            addresses[address] = label
            chain += 1
        if chain > POLE_COUNT_CAP:
            reasons.append(
                f"{target} has {chain} contributors and POLE_COUNT_CAP is "
                f"{POLE_COUNT_CAP}; the gate swept "
                f"{sorted(set(POLE_COUNTS_SWEPT))} and a longer chain has never "
                f"been run on this body")
    # EVERY P MUST ALSO BE DISTINCT FROM EVERY OTHER BOUND VOLUME, not merely
    # from the other P buffers. The pole parameters carry __restrict__ while the D
    # and inverse-epsilon parameters deliberately do not, so the promise this
    # signature makes is "no P aliases anything else in it" -- which is checkable
    # here and is checked again at the launch, because the buffers rotate.
    others: Dict[int, str] = {}
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "Dx", "Dy", "Dz"):
        address = _coverage._base_address(getattr(fields, name, None))
        if address is not None:
            others.setdefault(address, name)
    for volume in _coverage.offdiag_row_volumes(fields):
        address = _coverage._base_address(volume)
        if address is not None:
            others.setdefault(address, "a chi1inv row")
    reader = getattr(fields, "inverse_epsilon_for", None)
    if callable(reader):
        for component in ("Ex", "Ey", "Ez"):
            try:
                address = _coverage._base_address(reader(component))
            except Exception:  # noqa: BLE001 - reported by the array clauses
                continue
            if address is not None:
                others.setdefault(address, f"inverse_epsilon_for({component!r})")
    for address, label in addresses.items():
        if address in others:
            reasons.append(
                f"{label} aliases {others[address]}; the pole pointers carry "
                f"__restrict__ and an alias there is undefined behaviour the "
                f"compiler is licensed to miscompile silently")
    return reasons


def covers_real_pml_dispersive_offdiag_constitutive(fields: Any, pml: Any,
                                                    grid: Any) -> tuple:
    """Whether ``update_E_pml_real_folded_offdiag_dispersive`` may serve this run.

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal -- the
    convention every predicate in this package uses.

    THE SUB-STEP: ``stepping.update_E``'s ``elif offdiagonal:`` branch
    (stepping.py:1001-1008) under an active layer, with ``fields.polarizations``
    non-empty so ``displacement["volumes"]`` holds the three ``D - sum P``
    buffers (stepping.py:996, fields.py:1107-1138), on a grid that may be
    mirror-folded.

    THE FOUR DISJOINTNESS SEAMS, each the exact complement of a shipped refusal:

    * ``covers_real_pml_constitutive(side='E')`` refuses every off-diagonal run
      AND every ``fields.polarizations``-truthy run; this REQUIRES both;
    * ``covers_real_pml_offdiag_constitutive`` refuses a fold and refuses
      ``fields.polarizations`` truthy; this requires the latter, so the two are
      disjoint whether or not an axis is folded;
    * ``folded_offdiag_kernels.covers_folded_offdiag_composition`` refuses
      ``fields.polarizations`` truthy; this requires it;
    * ``dispersive_kernels.covers_real_pml_dispersive_constitutive`` refuses an
      off-diagonal row; this REQUIRES a surviving slot.

    AN UNFOLDED GRID IS ADMITTED, and unlike the folded off-diagonal family that
    is NOT a reduction-only courtesy that a narrower census verdict then takes
    back. There is no shipped family to overlap on the unfolded arm: an
    off-diagonal DISPERSIVE ``update_E`` is refused by ``cuda_offdiag`` for its
    dispersion whether or not a fold is present, and by ``cuda_dispersive`` for
    its off-diagonal row. So one predicate serves both, the census asks THIS one,
    and the gate sweeps unfolded specs beside the folded ones.
    """
    xp, backend = _coverage._backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, ("complex64 storage: the recurrence is the same but the "
                       "storage is not")
    if not (pml is not None and getattr(pml, "is_active", False)):
        # Without an absorber ``update_E`` takes the plain assignment
        # (stepping.py:1022) -- no ``f_w``, no recurrence. A different sub-step,
        # and nothing on this track carries its off-diagonal dispersive form.
        return False, ("no active PML layer: update_E takes the plain assignment "
                       "at stepping.py:1022, which is a different device string "
                       "and no family on this track emits its off-diagonal "
                       "dispersive form")
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["cylindrical"]:
        return False, ("cylindrical (Dcyl): the r axis has its own ghost rule "
                       "(the r_to_minus_r image, stepping.py:1877-1888) and its "
                       "own axial extent")
    for axis in range(3):
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    fold_problems = _folded._fold_reasons(facts, grid)
    if fold_problems:
        return False, fold_problems[0]
    folded = sum(1 for axis in range(3) if facts["mirrored"][axis])
    if folded > FOLD_PLANES_SWEPT:
        return False, (f"{folded} mirror planes at once: this family's gate "
                       f"scored {FOLD_PLANES_SWEPT} simultaneous planes, so a "
                       f"{folded}-plane fold would be admitted by argument alone")
    for axis, kind in enumerate(_coverage._boundary_kinds_from(facts)):
        # Stated positively: the three ghost rules the borrowed helpers write.
        if kind not in FOLDED_BC_CODES:
            return False, (f"axis {axis} resolves to boundary {kind!r}, which "
                           f"has no kernel")
    if facts["has_bloch"]:
        return False, ("nonzero Bloch k: the wrapped plane carries a phase real "
                       "storage cannot hold")
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, ("special_kz (grid.beta != 0): extra out-of-plane coupling "
                       "terms")
    # Both maps by name, never ``has_nonlinearity`` -- that property reads
    # ``_chi2_components`` alone (fields.py:966-967), so a chi3-only run would
    # pass a predicate that asked it. MEEP's most general case scales the WHOLE
    # row product (coupling included) by the Pade factor, step_generic.cpp:590-601
    # as ``stepping._nonlinear_constitutive`` (:1066-1073) transcribes it.
    if (getattr(fields, "_chi2_components", None)
            or getattr(fields, "_chi3_components", None)):
        return False, ("instantaneous chi2/chi3: the Pade factor scales the whole "
                       "row product, coupling included")
    if not tuple(getattr(fields, "polarizations", ()) or ()):
        # THE DISJOINTNESS CLAUSE against the two NON-dispersive off-diagonal
        # families, and it is spelled on ``fields.polarizations`` being truthy
        # rather than on the resolved chain length because THAT is the question
        # both of them ask (coverage.py's off-diagonal predicate and
        # folded_offdiag_kernels'). A run carrying a trivial-sigma susceptibility
        # is refused by both of them and admitted here, so no slot falls between.
        return False, ("no susceptibility is registered: update_E's source is D "
                       "itself and the run belongs to "
                       "coverage.covers_real_pml_offdiag_constitutive or to "
                       "folded_offdiag_kernels.covers_folded_offdiag_composition")
    # INVERTED, and counted over the six slots rather than read off the bare
    # ``has_offdiagonal_epsilon`` flag: a row planted past the installer under a
    # diagonal key sets the flag with every slot dead, and the emitter refuses an
    # all-dead mask by raising.
    rows = _coverage.offdiag_row_volumes(fields)
    if not any(volume is not None for volume in rows):
        return False, ("no off-diagonal chi1inv row survived installation: that "
                       "configuration is "
                       "dispersive_kernels.covers_real_pml_dispersive_"
                       "constitutive's and this predicate must not overlap it")
    if not getattr(fields, "stores_E", False):
        return False, "E is recomputed from D rather than stored"

    spec = _coverage.CONSTITUTIVE_SIDES["E"]
    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    outputs = tuple(spec["targets"]) + tuple(spec["aux"])
    for name in outputs + tuple(spec["sources"]):
        problem = _coverage._array_problem(name, getattr(fields, name, None),
                                           xp, shape)
        if problem is not None:
            return False, problem
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return False, "fields does not expose inverse_epsilon_for"
    for component in ("Ex", "Ey", "Ez"):
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal
            return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
        if volume is None:
            return False, f"inverse_epsilon_for({component!r}) is None"
        if not getattr(volume, "shape", ()):
            return False, (f"inverse_epsilon_for({component!r}) is a scalar, not "
                           f"a volume")
        problem = _coverage._array_problem(
            f"inverse_epsilon_for({component!r})", volume, xp, shape)
        if problem is not None:
            return False, problem
    # The surviving rows, readable, well-formed, and not an output. The alias
    # clause is the folded family's, and the poles sharpen nothing about it: the
    # coupling re-reads the partner volumes at neighbour offsets, on a folded axis
    # including INTERIOR stored row MIRROR_ROW rather than a face.
    output_addresses: Dict[int, str] = {}
    for name in outputs:
        address = _coverage._base_address(getattr(fields, name, None))
        if address is not None:
            output_addresses.setdefault(address, name)
    for index, volume in enumerate(rows):
        if volume is None:
            continue
        row, partner = _coverage.OFFDIAG_ROW_SLOTS[index]
        label = f"chi1inv_offdiagonal[{row!r}][{partner!r}]"
        if not getattr(volume, "shape", ()):
            return False, f"{label} is a scalar, not a volume"
        problem = _coverage._array_problem(label, volume, xp, shape)
        if problem is not None:
            return False, problem
        address = _coverage._base_address(volume)
        if address is not None and address in output_addresses:
            return False, (f"{label} aliases output {output_addresses[address]}: "
                           f"the coupling re-reads the partner volumes at "
                           f"neighbour offsets -- on a folded axis including "
                           f"interior stored row {MIRROR_SOURCE_INDEX} -- while "
                           f"the outputs are written, so the answer would depend "
                           f"on block schedule")
    problems = _dispersion_reasons(fields, xp, shape)
    if problems:
        return False, problems[0]
    # HALF-INTEGER ONLY, the E side's own sub-lattice (stepping.py:1015). ON A
    # FOLDED AXIS THE VECTOR IS BUILT AT THE FOLDED STORED EXTENT, which is what
    # ``_coefficient_vector_problem`` checks by SHAPE rather than by size.
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coverage._coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


# ---------------------------------------------------------------------------
# THE LAUNCHER
# ---------------------------------------------------------------------------

#: The emitted sources a gate has replaced, keyed by ``(row mask, arity)`` -- the
#: mutation seam. Paired with :func:`clear_kernel_cache`: the compile memo keys on
#: the SOURCE, so a mutated body is a miss and reaches NVRTC without the clear,
#: but the clear is what makes "how many compiles came from the mutated bytes"
#: exact.
_SOURCE_OVERRIDES: Dict[Tuple[Tuple[int, ...], Tuple[int, int, int]], str] = {}


def _key(row_mask: Sequence[int], counts: Sequence[int]):
    return normalized_row_mask(row_mask), normalized_pole_counts(counts)


def kernel_source(row_mask: Sequence[int], counts: Sequence[int]) -> str:
    """The device text a launch would compile for this pair -- override or emit."""
    key = _key(row_mask, counts)
    return _SOURCE_OVERRIDES.get(key) or dispersive_offdiag_source(*key)


def set_kernel_source(row_mask: Sequence[int], counts: Sequence[int],
                      source: Optional[str]) -> None:
    """Replace (or, with ``None``, restore) one pair's device text.

    THE GATE'S DOOR, and it is load-bearing: a mutation harness that could not
    route its own bytes through the launcher would launch the shipped kernel and
    report a pass for a defect it never introduced.
    """
    key = _key(row_mask, counts)
    if source is None:
        _SOURCE_OVERRIDES.pop(key, None)
    else:
        _SOURCE_OVERRIDES[key] = source


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return _clear_cache()


def _get_kernel(row_mask: Sequence[int], counts: Sequence[int]):
    """Compile on first use, memoized on (name, options, policy, source).

    THE ROW MASK AND THE ARITY REACH THE KEY THROUGH THE SOURCE, which is already
    in it, so two specializations cannot be served each other's binary -- and
    neither can a mutation harness that rewrote the emitter's output be served the
    unmutated one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = kernel_source(row_mask, counts)
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _require_no_aliasing(fields: Any, rows: Sequence[Any],
                         plan: Dict[str, Sequence[Any]]) -> None:
    """Outputs pairwise distinct, and every ``__restrict__`` pointer distinct.

    THE PREDICATE ALREADY CHECKED THIS and it is checked again here, for the
    shipped ADE launcher's reason: the two answer about different moments. The
    predicate answers about the state as a planner saw it; the pole buffers rotate
    between steps (dispersion.py:689-691), so what it certified is one permutation
    and this is the one being run.

    THE D AND INVERSE-EPSILON VOLUMES ARE ALLOWED TO ALIAS EACH OTHER AND THE ROW
    COEFFICIENTS -- their parameters carry no ``restrict`` for exactly that reason
    -- but nothing may alias an OUTPUT, because the coupling re-reads the partner
    volumes at neighbour offsets while the outputs are being written, and on a
    folded axis one of those offsets is interior stored row 2.

    Equality of BASE addresses only: two overlapping views with different bases
    pass unseen, the accepted limitation every plan on this track shares.
    """
    outputs: Dict[int, str] = {}
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        address = _coverage._base_address(getattr(fields, name, None))
        if address is None:
            raise ValueError(
                f"{name} exposes no readable base address; an unverifiable "
                f"output is not accepted")
        if address in outputs:
            raise ValueError(f"{name} aliases {outputs[address]}; the outputs "
                             f"must be distinct arrays")
        outputs[address] = name

    inputs: List[Tuple[str, Any]] = [
        ("Dx", fields.Dx), ("Dy", fields.Dy), ("Dz", fields.Dz),
        ("inv_eps_Ex", fields.inverse_epsilon_for("Ex")),
        ("inv_eps_Ey", fields.inverse_epsilon_for("Ey")),
        ("inv_eps_Ez", fields.inverse_epsilon_for("Ez")),
    ]
    inputs.extend((f"chi1inv row {index}", volume)
                  for index, volume in enumerate(rows))
    poles: List[Tuple[str, Any]] = []
    for target, _source, _axis in E_TERMS:
        poles.extend((f"P_{target}_{index}", array)
                     for index, array in enumerate(plan.get(target, ())))
    for label, volume in inputs + poles:
        address = _coverage._base_address(volume)
        if address is not None and address in outputs:
            raise ValueError(
                f"{label} aliases {outputs[address]}: the coupling re-reads the "
                f"partner volumes at neighbour offsets -- on a folded axis "
                f"including interior stored row {MIRROR_SOURCE_INDEX} -- while "
                f"the outputs are written, so the answer would depend on block "
                f"schedule")
    seen: Dict[int, str] = {}
    for label, array in poles:
        address = _coverage._base_address(array)
        if address is None:
            raise ValueError(
                f"{label} exposes no readable base address; the restrict "
                f"promises in the device signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]}; the pole pointers carry "
                f"__restrict__ and two susceptibilities never share a P buffer "
                f"(dispersion.py:645-647)")
        seen[address] = label
    for label, volume in inputs:
        address = _coverage._base_address(volume)
        if address is not None and address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]}; the pole pointers carry "
                f"__restrict__ and an alias there is undefined behaviour the "
                f"compiler is licensed to miscompile silently")


def _validate_launch_arguments(shape: Sequence[int], codes: Sequence[int],
                               walls: Sequence[int],
                               weights: Sequence[float]) -> None:
    """Everything a launch would otherwise get silently wrong.

    The folded family's checks, restated by CALLING nothing of its own because
    each is a WRONG ANSWER rather than a crash if it is not checked here: a NaN
    weight multiplies one plane into NaN, a folded axis that is also wall-masked
    zeroes a plane MEEP steps, and a folded axis with too few cells reads a
    neighbouring plane. The predicate names all three, and a gate can reach the
    launcher without the predicate, which is why they are checked twice.
    """
    if len(codes) != 3 or any(int(code) not in FOLDED_BC_CODES.values()
                              for code in codes):
        raise ValueError(
            f"boundary codes must be three of "
            f"{sorted(set(FOLDED_BC_CODES.values()))}, got {tuple(codes)!r}")
    if len(walls) != 3 or any(int(flag) not in (0, 1) for flag in walls):
        raise ValueError(f"wall flags must be three of {{0, 1}}, got "
                         f"{tuple(walls)!r}")
    if len(weights) != 3:
        raise ValueError(f"ghost weights must be three floats, got "
                         f"{tuple(weights)!r}")
    for axis in range(3):
        if int(codes[axis]) != BC_MIRROR_CODE:
            continue
        if int(walls[axis]):
            raise ValueError(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1282) and zeroing the fold plane the way the "
                f"metallic rule does costs 2.0e-02 (stepping.py:1269-1274)")
        if float(weights[axis]) not in (1.0, -1.0):
            raise ValueError(
                f"axis {axis} is folded with ghost weight {weights[axis]!r}; the "
                f"mirror ghost carries mirror_parity('D'+axis, axis, phase) == "
                f"-phase, exactly +1.0 or -1.0 (fields.py:117, "
                f"stepping.py:1824-1825)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 1824-1825->1871-1872
        if int(shape[axis]) <= MIRROR_SOURCE_INDEX:
            raise ValueError(
                f"axis {axis} is folded with {shape[axis]} stored cells; the "
                f"mirror ghost images stored row {MIRROR_SOURCE_INDEX}")


def _launch(kernel, fields: Any, rows: Sequence[Any],
            plan: Dict[str, Sequence[Any]], tables: dict, codes: Sequence[int],
            walls: Sequence[int], weights: Sequence[float]) -> None:
    """One launch. ``rows`` are the LIVE coefficient volumes, in slot order."""
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    nx, ny, nz = fields.Ex.shape
    blocks = ((nx * ny * nz + _DISPERSIVE_OFFDIAG_THREADS - 1)
              // _DISPERSIVE_OFFDIAG_THREADS)
    arguments: List[Any] = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
    ]
    for target, _source, _axis in E_TERMS:
        arguments.extend(plan.get(target, ()))
    arguments.extend(rows)
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
        np.int32(codes[0]), np.int32(codes[1]), np.int32(codes[2]),
        np.int32(walls[0]), np.int32(walls[1]), np.int32(walls[2]),
        np.float32(weights[0]), np.float32(weights[1]), np.float32(weights[2]),
    ])
    kernel((blocks,), (_DISPERSIVE_OFFDIAG_THREADS,), tuple(arguments))


def update_E_dispersive_offdiag_fused_pml_real(
        fields: Any, pml: Any = None, *, tables: dict = None, codes=None,
        walls=None, weights=None, plan: Optional[Dict[str, List[Any]]] = None
) -> Dict[str, Any]:
    """``stepping.update_E``'s dispersive tensor row product, in ONE launch.

    SUPPLY ``pml`` AND EVERY DERIVED ARGUMENT IS DECIDED HERE -- the half-integer
    tables through :func:`coverage.constitutive_sub_lattice`, the boundary codes
    through ``real_pml_boundary_kinds`` and :data:`FOLDED_BC_CODES`, the wall flags
    through the grid's own declaration, the ghost weights through
    ``fields.mirror_parity``. That is what makes "the launcher and the predicate
    cannot disagree" an enforced property of this function rather than a
    convention a caller may keep.

    ``tables``/``codes``/``walls``/``weights`` ARE THE GATE'S DOOR and stay,
    keyword-only: a byte gate feeds synthetic tables no ``PML`` produces, and its
    mutations feed DELIBERATELY WRONG ones. Passing ``pml`` together with an
    override, or neither, is refused: a caller with two answers to a one-answer
    question has a bug either way.

    ``plan`` IS A FIFTH DOOR AND IT IS SEPARATE FROM THE OTHER FOUR, deliberately.
    The pole chain is resolved fresh on every call because the buffers rotate, so
    a harness arming an ORDERING defect -- the one that actually threatens this
    kernel -- passes a reversed chain here while still letting the layer decide
    everything else.

    Returns the arity it launched at, so a caller can record which body ran.
    """
    overrides = (tables, codes, walls, weights)
    supplied = [value is not None for value in overrides]
    if (pml is not None) and any(supplied):
        raise ValueError(
            "pml and an override were both supplied; the derived answer and the "
            "supplied one cannot both be the one this launch used")
    if pml is None and not all(supplied):
        raise ValueError(
            "pass exactly one of pml (tables, codes, walls and weights are all "
            "derived from it and the grid) or ALL FOUR of tables, codes, walls "
            "and weights (the gate supplies its own, including deliberately "
            "wrong ones); got "
            f"tables={'set' if tables is not None else 'None'}, "
            f"codes={'set' if codes is not None else 'None'}, "
            f"walls={'set' if walls is not None else 'None'}, "
            f"weights={'set' if weights is not None else 'None'}")
    if pml is not None:
        tables = dispersive_offdiag_constitutive_tables(pml)
        codes = dispersive_offdiag_boundary_codes(fields.grid, pml)
        walls = _coverage.offdiag_wall_mask_flags(fields.grid)
        weights = mirror_ghost_weights(fields.grid)

    if plan is None:
        plan = resolve_pole_plan(fields)
    counts = normalized_pole_counts(pole_counts_of(plan))
    mask = normalized_row_mask(_coverage.offdiag_row_mask(fields))
    rows = [volume for volume in _coverage.offdiag_row_volumes(fields)
            if volume is not None]
    _validate_launch_arguments(tuple(fields.Ex.shape), codes, walls, weights)
    _require_no_aliasing(fields, rows, plan)
    _launch(_get_kernel(mask, counts), fields, rows, plan, tables, codes, walls,
            weights)
    return {"row_mask": list(mask), "counts": list(counts), "launches": 1}
