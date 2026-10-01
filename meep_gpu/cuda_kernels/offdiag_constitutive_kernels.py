"""Raw-CUDA off-diagonal (tensor epsilon) electric constitutive sub-step.

THE CHEAPEST ROWS ON THE BOARD. After the real-storage constitutive pair landed,
the hand-CUDA track covered 157 of 759 predicate slots and 26 corpus rows at every
sub-step. The rows this file adds are the only remaining family whose
configurations the certified CUDA CURL PAIR ALREADY ADMITS: a tensor epsilon
changes ``update_E`` and nothing else, because ``stepping._offdiagonal_terms`` is
reached from ``update_E`` and from ``_nonlinear_constitutive`` and from nowhere
else. Every row here is one where three of the four sub-steps were already covered
and exactly one was not.

WHY A NEW MODULE AND NOT AN EDIT. ``step_curl_kernels.py``'s fourteen device
strings are pinned by digest in ``certification.json`` and asserted by
``test_certification_record.py``; ``constitutive_kernels.py``'s two are pinned the
same way by ``test_constitutive_pml_real.py``. Adding a kernel to either would
force a record edit that no device run backs. Nothing here changes a byte of
either.

WHERE THE PIECES LIVE, and why they are split three ways. The DEVICE SOURCE and
its emitter are in ``offdiag_emitter.py`` and the PREDICATE is in ``coverage.py``,
both stdlib-only, because this family's device code is specialized per row mask —
so the thing a test has to exercise is a FUNCTION, and a function has to be
imported to be called. This module is the CuPy half: compile, bind, launch. It is
the slice's public face and re-exports the other two rather than copying them.

WHAT IS TRANSCRIBED, AND FROM WHERE. ``stepping.update_E`` (stepping.py:954) with
an active PML, ``fields.has_offdiagonal_epsilon`` True (fields.py:1313-1315) and
no nonlinearity — the ``elif offdiagonal:`` branch (stepping.py:1001-1008). With no
poles admitted ``displacement_minus_polarization_volumes`` aliases each source to
its D primary (fields.py:1107-1138), and all three are alive at once because the
coupling reads the OTHER components' volumes (stepping.py:991-997). Per component
``c`` with own axis ``a``::

    constitutive = D_c * us_c                                # stepping.py:1005
    per surviving partner (offset 1 then 2, cycle X->Y->Z, :1235-1237):
        pair    = g + shift_down(g, partner_axis)            # :1214-1216
        product = pair * coefficient                         # :1217
        term    = 0.25 * (product + shift_up(product, a))    # :1219-1220
        total   accumulates term(offset 1) then term(offset 2)  # :1221
    _mask_metallic_wall_coupling(total)                      # :1223, :1250-1254
    constitutive = (D_c * us_c) + total                      # :978-979
    prev = f_w_c ; f_w_c = constitutive                      # :2065-2096
    E_c += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev          # half-integer, :986

The seven properties that decide bit-identity are enumerated in
``offdiag_emitter.py``, beside the text they are properties of.

=============================================================================
SPECIALIZATION: WHICH AXES ARE COMPILE-TIME, AND WHY NOT THE SIBLING'S CHOICE
=============================================================================

The sibling Triton kernel carries TWELVE ``tl.constexpr`` axes for this family —
six row-liveness flags, three boundary codes, three wall-mask flags — a static
product of 4,096 kernels. Measured over the 186-row corpus, SEVEN are ever
requested, six restricted to the rows the hand-CUDA curl admits.

This track splits the twelve by WHAT THEY DO TO THE ARITHMETIC. The boundary and
wall axes are BRANCH axes: they select an index or a predicated zero and change no
float operation and no association, so they are runtime ``int`` arguments — which
is not a new bet, it is what the certified ``step_B_pml_real`` already does
with ``bc_x``/``bc_y``/``bc_z`` (step_curl_kernels.py:1687). Sixty-four variants
collapse to one source. The six ROW-LIVENESS flags are an ARITY axis: they change
how many terms the inner sum has, and therefore its association, so they stay
compile-time.

THE ARITY AXIS STAYED COMPILE-TIME THIS ROUND, and the two facts behind that point
in opposite directions, so both are stated. The ZERO-PADDING fold — bind a dead
slot to an all-zero coefficient volume so a dropped term contributes ``+0.0``
instead of not existing — is measurably NOT bit-identical: ``x + 0.0f`` is the
identity on the bits for every float32 x EXCEPT ``-0.0f``, which it turns into
``+0.0f``, and a coupling total of ``-0.0f`` is reachable. The same hole sits in a
``total = 0.0f`` accumulator spelling.
``test_offdiag_constitutive_pml_real.py`` carries the explicit witness.

A DYNAMIC LOOP is a different fold and IS licensed:
``the design notes (cuda-kernel-triton-transfer-assessment)`` §3.2 (2026-08-16,
``results/cuda_dynamic_loop_2026-08-16/``) measured a runtime-bounded loop over an
arity axis bit-identical to the unrolled form at every arity 0-8 — 156/156 cases,
2,759,094 output words, zero differing words, max ULP 0 — because NVCC unrolls to
pipeline the loads and leaves the arithmetic a strictly serial dependence chain on
one accumulator. This family's row accumulation is such a chain, so the axis COULD
go runtime and this round simply did not take it. Doing so needs all six
coefficient pointers bound (reintroducing the dead-pointer shape the emitted
signature removes) and needs the accumulation kept in its OWN accumulator with the
final ``+ total`` conditional on a non-zero trip count — because the array path
sums the coupling separately and adds it once (stepping.py:1250 then :1007-1008),
where §3.2's ``NP`` axis accumulates into the diagonal itself. It would buy one
compiled source on this corpus instead of two, and one byte-gate variant instead
of two.

WHAT THE CORPUS DRIVES on this split: the boundary and wall axes cost nothing, and
of the 63 live row masks the 186-row corpus asks for TWO.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON (CUDA path, adjudicated 2026-08-15)
=============================================================================

* ``--fmad=false`` is CORRECTNESS, not tuning, and the option tuple is SPELLED
  HERE rather than imported, so loading this file by path cannot pick up a
  different one than a gate compiled. The contraction candidates in this family
  are the two accumulations of the tail, the ``D * inv_eps`` product, and — new
  here — the two ``pair * coefficient`` products feeding the ``0.25f`` scale. A
  test pins the tuple equal to both siblings'.
* SIGNED ZERO: CUDA lowers ``-x`` to ``neg.f32``, NOT to ``0.0f - x``, so it does
  NOT canonicalize signed zeros. This is the OPPOSITE of the sibling track, whose
  ``(a*b)*-1.0`` idiom exists precisely because its platform does. That idiom is
  not ported here and no unary minus appears on any float path. The asymmetry is
  also why the zero-padding fold above is unsafe in a way a reader arriving from
  the other track would not expect.
* DIVISION: none. This family is divide-free, so the one PTX exception on record —
  ptxas expanding ``div.rn.f32`` into a Newton-Raphson sequence whose range checks
  carry ``.FTZ`` in SASS regardless of the PTX modifier — cannot arise.
* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
  ``open(..., 'w')`` (compiler.py:368), so the bytes go through the interpreter's
  LOCALE encoding — ASCII under C/POSIX, which is what a non-interactive shell on
  the validation host gets. Two em-dashes in a comment killed the sibling's E
  kernel at its first launch on 2026-08-15. The tests scan AND encode every
  emitted source, per row mask.

=============================================================================
CERTIFIED 2026-08-16. STILL NOT DISPATCHED.
=============================================================================

``certification.json``'s ``offdiag_2026-08-16`` block, cut on an RTX A6000 by
``parity/meep_gpu/gate_cuda_offdiag.py`` under BOTH float32 subnormal policies:
384/384 single-launch cases bit-identical to ``stepping.update_E`` per policy
(uniform and subnormal-band operands, two Courants, four row masks, three shapes
including an invariant axis, every boundary triple), 32/32 at the 60-launch
multi-step budget, and 22/22 mutation legs as required — including the four
defects this family exists to be proof against (coupling the wrong components,
dropping a row from the volume sum, inverting the wall mask, transposing the
coefficient reads) and two must-be-UNCAUGHT controls. The unguarded control
DIVERGES on 384/384, so ``--fmad=false`` is measurably load-bearing here rather
than merely prudent — which is not what the plain constitutive pair measured,
where the interior coefficients are the exact identity and a contraction is
invisible outside the layer.

WHAT THE CERTIFICATION DOES NOT SAY. It is a verdict about the EMITTER at the
corpus digest the record carries, exercised on four of its 63 row masks; the
policy probe compiles three and proves each pair of policies produced distinct
binaries. A mask outside those is covered by the emitter's own tests and by that
digest, not by a byte gate. No throughput claim is made or possible: the box was
shared.

Nothing dispatches this. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions); the only
importers are this directory's tests and the parity probes.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML


def _sibling(module_name: str):
    """Load a same-directory sibling by path — the bit-identity probe's route.

    The probe loads these modules OUTSIDE the package, where a relative import has
    no anchor. Every sibling is fetched through this one helper so the fallback
    cannot be written three slightly different ways.
    """
    import importlib.util  # noqa: PLC0415
    import os  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(
        f"cuda_kernels_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The predicate, the family's transcribed tables and the emitter all live in
# CUPY-FREE siblings so they stay importable, exercisable and mutable on a machine
# with no GPU. Imported back here because this module is the slice's public face.
try:
    from . import coverage as _coverage
    from . import compile_cache
    from . import offdiag_emitter
    from .constitutive_kernels import real_constitutive_tables
except ImportError:  # loaded by path, outside the package
    _coverage = _sibling("coverage")
    compile_cache = _sibling("compile_cache")
    offdiag_emitter = _sibling("offdiag_emitter")
    real_constitutive_tables = _sibling("constitutive_kernels").real_constitutive_tables

BC_CODES = _coverage.BC_CODES
OFFDIAG_ROW_SLOTS = _coverage.OFFDIAG_ROW_SLOTS
constitutive_sub_lattice = _coverage.constitutive_sub_lattice
covers_real_pml_offdiag_constitutive = _coverage.covers_real_pml_offdiag_constitutive
offdiag_row_mask = _coverage.offdiag_row_mask
offdiag_row_volumes = _coverage.offdiag_row_volumes
offdiag_wall_mask_flags = _coverage.offdiag_wall_mask_flags
real_pml_boundary_kinds = _coverage.real_pml_boundary_kinds

E_TERMS = offdiag_emitter.E_TERMS
KERNEL_NAME = offdiag_emitter.KERNEL_NAME
LIVE_ROW_MASKS = offdiag_emitter.LIVE_ROW_MASKS
normalized_row_mask = offdiag_emitter.normalized_row_mask
offdiag_source = offdiag_emitter.offdiag_source

#: Byte-identical to the array path with a DEVICE gate verdict behind it.
#: ``certification.json``'s ``offdiag_2026-08-16`` block, cut on an RTX A6000
#: under BOTH float32 subnormal policies: 384/384 single-launch cases identical
#: per policy, 32/32 at the 60-launch multi-step budget, 22/22 mutation legs as
#: required, the unguarded control diverging on 384/384, and the two policies
#: proven to have compiled DISTINCT binaries for each of the three row-mask
#: variants the probe drives. A name added here without a record block fails
#: ``test_offdiag_constitutive_pml_real.py``.
CERTIFIED_KERNELS = ("update_E_pml_real_offdiag",)

#: Shipped but not gated, naming the gate owed. Spelled as a plain dict WITHOUT a
#: type annotation for the reason the sibling records: the partition test reads it
#: off the syntax tree without importing the module — the only way to read it on a
#: host with no CuPy — and an annotated assignment is an ``ast.AnnAssign``, which
#: that reader does not match. EMPTY: this family ships one kernel and it is
#: certified.
UNCERTIFIED_KERNELS = {}

# NVRTC compile options — CORRECTNESS, not performance. Identical to both siblings'
# and spelled here rather than imported so that loading this file by path (which
# the probe does) cannot pick up a different tuple than the one a gate compiled.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block. One element per lane, flat 1-D grid — the geometry the
#: certified curl pair and the certified constitutive pair both use, and the one
#: the sibling track's ``DEFAULT_BLOCK`` arrived at independently.
_OFFDIAG_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with both sibling modules — the cache is keyed on the source
    string, so neither two modules nor two row masks can collide.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(row_mask):
    """Compile on first use, memoized on (name, options, policy, source).

    THE ROW MASK REACHES THE KEY THROUGH THE SOURCE, which is already in it, so
    two specializations cannot be served each other's binary — and neither can a
    mutation harness that rewrites the emitter's output be served the unmutated
    one. The source is rebuilt per call for the reason both siblings rebuild their
    code map per call: it is part of the key, so it has to exist before there is a
    key to miss on.
    """
    code = offdiag_emitter.offdiag_launch_source(row_mask)
    key = compile_cache.kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def offdiag_constitutive_tables(pml: 'PML') -> dict:
    """The six HALF-INTEGER kps/kms views this sub-step reads.

    The sub-lattice is asked of :func:`coverage.constitutive_sub_lattice`, the same
    function the predicate asks and the certified sibling's launcher asks, so no
    call site can pair the E side with the INTEGER tables — a half-cell error in
    the absorber profile that is converged, smooth and wrong. The flatten itself is
    the certified sibling's ``real_constitutive_tables`` rather than a second copy
    of it, dtype and contiguity guards included.
    """
    return real_constitutive_tables(pml, constitutive_sub_lattice("E"))


def offdiag_boundary_codes(grid, pml=None) -> tuple:
    """The three ``bc_*`` arguments: the resolved ghost rule per axis, as codes.

    ``real_pml_boundary_kinds`` reproduces ``stepping._boundary_kinds``; every kind
    it can return that is not in ``BC_CODES`` is refused by the predicate, so a
    KeyError here would mean the launcher ran on a configuration the predicate had
    declined. It is left to raise rather than defaulted, because a default would
    turn that into a silent wrong ghost rule.

    ``pml`` is accepted and unused: the resolution reads the GRID alone
    (``stepping._boundary_kinds`` consults the layer only for a mirror fold, which
    this family refuses), and taking the argument keeps every launcher on this
    track calling the same shape.
    """
    return tuple(BC_CODES[kind] for kind in real_pml_boundary_kinds(grid))


def _require_no_aliasing(fields: 'Fields', rows) -> None:
    """Outputs pairwise distinct and disjoint from every input volume.

    THE COUPLING IS WHY THIS IS STRONGER THAN THE PLAIN CONSTITUTIVE KERNEL'S
    NEEDS. That sub-step reads each cell it writes and nothing else, so an alias
    would be merely wrong; this one re-reads the partner volumes at NEIGHBOUR
    offsets while the outputs are being written, so an alias makes the answer
    depend on block schedule — wrong differently on each run, which no single
    comparison catches reliably.

    Equality of BASE addresses only: two overlapping views with different bases
    pass unseen, the accepted limitation both siblings' plans share.
    """
    outputs = {}
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
    inputs = [fields.Dx, fields.Dy, fields.Dz,
              fields.inverse_epsilon_for("Ex"),
              fields.inverse_epsilon_for("Ey"),
              fields.inverse_epsilon_for("Ez")]
    inputs.extend(rows)
    for volume in inputs:
        address = _coverage._base_address(volume)
        if address is not None and address in outputs:
            raise ValueError(
                f"an input volume aliases {outputs[address]}: the coupling "
                f"re-reads the partner volumes at neighbour offsets while the "
                f"outputs are written, so the answer would depend on block "
                f"schedule")


def _launch(kernel, fields: 'Fields', rows, tables: dict, codes, walls):
    """One launch. ``rows`` are the LIVE coefficient volumes, in slot order.

    Only the live ones: the emitted signature declares no parameter for a dead
    slot, so there is no pointer bound to an array the kernel must never touch.
    """
    nx, ny, nz = fields.Ex.shape
    blocks = (nx * ny * nz + _OFFDIAG_THREADS - 1) // _OFFDIAG_THREADS
    arguments = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
    ]
    arguments.extend(rows)
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
        np.int32(codes[0]), np.int32(codes[1]), np.int32(codes[2]),
        np.int32(walls[0]), np.int32(walls[1]), np.int32(walls[2]),
    ])
    kernel((blocks,), (_OFFDIAG_THREADS,), tuple(arguments))


def update_E_offdiag_fused_pml_real(fields: 'Fields', pml: 'PML' = None, *,
                                    tables: dict = None, codes=None, walls=None):
    """``stepping.update_E`` with off-diagonal chi1inv rows, in one launch.

    SUPPLY ``pml`` AND EVERY DERIVED ARGUMENT IS DECIDED HERE — the half-integer
    tables through :func:`coverage.constitutive_sub_lattice`, the boundary codes
    through ``real_pml_boundary_kinds``, the wall flags through the grid's own
    declaration. That is what makes "the launcher and the predicate cannot
    disagree" an enforced property of this function rather than a convention a
    caller may keep; the certified sibling's docstring claimed exactly that for a
    year while ``tables`` was a positional parameter passed straight through.

    ``tables``/``codes``/``walls`` ARE THE GATE'S DOOR and stay, keyword-only: a
    byte gate feeds synthetic tables no ``PML`` produces, and its mutations feed
    DELIBERATELY WRONG ones — a mis-paired sub-lattice, a dropped wall flag, an
    over-applied one, a swapped ghost rule. A launcher that could not take its own
    could not arm any of them. Passing ``pml`` together with an override, or
    neither, is refused: a caller with two answers to a one-answer question has a
    bug either way.
    """
    overrides = (tables, codes, walls)
    supplied = [value is not None for value in overrides]
    if (pml is not None) and any(supplied):
        raise ValueError(
            "pml and an override were both supplied; the derived answer and the "
            "supplied one cannot both be the one this launch used")
    if pml is None and not all(supplied):
        raise ValueError(
            "pass exactly one of pml (tables, codes and walls are all derived "
            "from it) or ALL THREE of tables, codes and walls (the gate supplies "
            "its own, including deliberately wrong ones); got "
            f"tables={'set' if tables is not None else 'None'}, "
            f"codes={'set' if codes is not None else 'None'}, "
            f"walls={'set' if walls is not None else 'None'}")
    if pml is not None:
        tables = offdiag_constitutive_tables(pml)
        codes = offdiag_boundary_codes(fields.grid, pml)
        walls = offdiag_wall_mask_flags(fields.grid)

    mask = normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    _require_no_aliasing(fields, rows)
    return _launch(_get_kernel(mask), fields, rows, tables, codes, walls)
