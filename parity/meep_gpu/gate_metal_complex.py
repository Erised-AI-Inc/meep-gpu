"""Byte gate for the Metal complex-field / Bloch PML curl and constitutive kernels.

THE CLAIM, and the only one: **byte-identity to ``stepping.py`` on this host,
subject to a declared and CHECKED subnormal-free precondition.** Not a stated
tolerance. Leg ``precondition`` is that check and it is required to FIRE on a scaled
control — a precondition never demonstrated to fire is decorative.

THE PER-CHUNK FORMS AND THE PART-WISE PRODUCTS (bit_parity_0_10, KERNELS.md §7,
step M5). The kernels transcribe the aligned array path, and this gate with them:

* the in-gate transcription keys its forms on the chunk record
  (``gate_chunk_records``): the split-field recurrence where the target's dsigu
  membership holds, the dsig-only form ``((f*kms) - curl)*sinv`` elsewhere; the
  accumulation ``f + ((kps*src) - (kms*prev))`` where the component's own dsigw
  membership holds, the output select ``f = src`` elsewhere. The curl is MEEP's
  left-to-right bracket with partner presence; every product is one per part; a
  phased axis wraps plainly and its unshifted operand is CONNECTED on plane 0 (copy,
  negate, or the unfused product);
* the grouping and product mutations are INVERTED (§7.1): each plants the package's
  PREVIOUS spelling and stays must-catch (``m1_previous_curl_grouping``,
  ``c1_previous_accumulation_grouping``, ``m10``/``m11`` the previous real-scale
  products, ``m12`` the previous fused connection). The §7.2 battery (``b1``..``b13``)
  and leg ``planted_zero`` are new;
* the expansion probe and its leg are gone (the emitted helpers have one body, so
  there is no arm to bind), leg ``call_site_shapes`` is gone with the NumPy
  products it measured, and leg ``zero_cross_terms`` is re-pointed as
  ``part_products`` -- the shipped helpers against ``part_products`` on exhaustive
  signed-zero tables, with the previous spellings as its must-differ arms;
* the contraction directive is load-bearing on both kernels now (bare per-part
  products beside a ``+``/``-``), and legs ``synthetic`` and ``contraction`` assert
  that it moves words.

The audit history below describes the gate as it stood before this change; where it
names an arm, a leg or a mutation this block retires, this block governs.

THE ORACLE IS IN PROCESS. ``stepping.step_B(fields, pml)`` and the Metal launch run
in ONE process against ONE seed on real ``Grid``/``Fields``/``PML`` objects with
complex64 storage. The in-file NumPy transcription STAYS in a reduced role: it is
the mutation substrate and the only way to exercise a deliberately wrong
configuration or an odd shape. LEG ``transcription`` PINS IT against ``stepping.py``
over multiple full cycles BEFORE any Metal kernel is trusted, which is what stops
"bit-identical" from meaning the kernel reproduced whatever this file wrote twice.

WHAT THIS GATE CARRIES BEYOND THE STANDARD DISCIPLINE, each because the
corresponding Triton gate learned it:

* **the plane-wise collapse is classified explicitly.** If a backend took a
  real-scalar fast path the rotation would collapse to plane-wise and the whole
  delta would evaporate, so ``plane_wise`` is a NAMED DIAGNOSTIC ARM measured on the
  exhaustive zero table rather than a possibility ruled out by reading;
* **the scalar operand shape is classified separately.** stepping.py:1909 multiplies
  an array plane by a complex64 SCALAR, not by an array, and that is the shape the
  array path actually uses;
* **the 31-BINDING CEILING is an assertion, not a comment.** Leg ``bindings``
  compiles the re/im-split signature and REQUIRES the compile error, so a later edit
  that splits the complex volumes is refused by a measurement;
* **the wrapped-plane select, and the k = 0 reduction.** An unphased build must be
  byte-identical to the plain complex path, which is what the skip (rather than a
  multiply by 1+0j) buys;
* **the Brillouin edge EXACTLY -1+0j** is a pinned case, not a sampled one.

WHAT AN ADVERSARIAL AUDIT ON 2026-08-15 CHANGED, each because a leg was measuring
less than it read as measuring:

* **three must-catch mutations could not fail.** m6, m7 and m13 overrode the phase
  table while comparing against an oracle ``stepping`` had produced on the case's
  OWN table. The unmutated shipped kernel launched under that override already
  differed by 1,624 words, so ``CAUGHT 1/1`` was reporting the mismatch and not the
  defect. The overrides are gone, :func:`_assert_mutation_configuration` refuses the
  class outright, and on the case's own table the defects are caught at 1,432 /
  7,848 / 11 words;
* **the Y rotation block had never been launched.** Every case carried ``k_y = 0``
  and no synthetic phase set phased y, so ``_phase_block("y", ...)`` and the ``py``
  binding were emitted, fingerprinted and never run against the oracle. Two cases
  (``periodic_ky``, ``periodic_kxyz``), per-axis synthetic singles, an all-three set,
  a per-axis coverage floor and three mutations (m6y, m7y, m14) close it;
* **the precondition covered half the claim.** It ran the two curl sub-steps only,
  so 192 constitutive comparisons were certified with no census while the summary
  said otherwise. It now runs all four, and censuses the kernel's UNSTORED
  intermediates — reconstructed on the host, which the stated weakness now says;
* **a predicted null was refuted by its own leg's numbers.** ``signed_zero``
  recorded the four periodic cases as "no -0.0 output is reachable" beside 760
  negative-zero words. A negative ``kms`` makes a ``-0.0`` from a ``+0.0`` on any
  axis; the floor is asserted on every case now, with a move floor beside it;
* **the probe's operand shapes were an assumption.** Leg ``call_site_shapes``
  measures the arm at the eleven shapes the engine actually evaluates;
* **``c_mul`` had no exhaustive table.** Only the real-coefficient helper did. The
  256-pattern full-product table now runs beside it;
* **the composition probe was overwriting this gate's provenance.** Distinct names.

WHAT A SECOND ADVERSARIAL PASS ON 2026-08-15 CHANGED, each again because a leg read
as measuring more than it measured:

* **the precondition bounded ONE case of eight.** It ran ``periodic_kx`` alone while
  the summary said the family's byte-identity was "under a CHECKED subnormal-free
  precondition" — so seven cases, the whole synthetic sweep and every multi-step
  cycle were certified with no census. The physical-band census now sweeps the whole
  matrix (asserted case by case against ``CONFIGS``), and ``leg_synthetic`` carries
  its own window;
* **the intermediate census was one value deep.** It reconstructed the curl and the
  two constitutive coefficient products; the recurrence forms five more per target
  and the constitutive a partial sum, any of which can land in the band while its
  operands and its result are outside it. The set is now COLLECTED FROM THE PINNED
  REFERENCE rather than re-derived, so it cannot drift from the arithmetic;
* **the constitutive kernel had never run on an odd shape.** Two engine shapes,
  both fully 3-D, were its entire launch history — the fossil being
  ``synthetic_coefficients``' unused ``stems`` parameter. It runs on ``(7, 5, 3)``,
  ``(4, 4, 4)`` and the COLLAPSED ``(9, 1, 6)`` now;
* **two ADMITTED configurations were never launched by any leg** — a 2-D run
  (collapsed z) and an absorber missing on one axis. Found by constructing real
  engine objects and CALLING the predicate over an enumeration of the unsupported
  features; leg ``admitted_domain`` is where the admitted domain gets measured;
* **``h1``/``h2`` reported a launch count they did not count**, passing a literal
  ``2`` and building their verdict by hand, so DISARMED was unreachable for them;
* **the exhaustive tables' ``shipped`` row was a hand copy** of the emitted helper
  and is now checked against ``templates.complex_helpers`` character for character
  (modulo the coefficient's local name);
* **three quoted figures were unreproducible from any artifact.** ``0/512``,
  ``0/512`` and ``36/512`` for the negation spellings had no leg, and "24/128 for
  the folded coefficient-left form" was simply WRONG — it is 12/128, the same as
  field-left; 24/128 is the plane-wise number. All four are arms now;
* **``c4``'s recorded reason contradicted its own verdict**, claiming "1/1 on the
  +-0 lattice" beside ``CAUGHT 0/1`` on that very seeding. Re-measured: 0 on both
  seedings and 0 with all three components swapped;
* **``leg_synthetic``'s docstring asserted the opposite of its code** — that the
  ``fast`` build must DIFFER, where the code requires it not to.

WHAT THE CERTIFICATION ROUND CHANGED, and it is one thing, found by RUNNING the
gate on the tree rather than by reading it:

* **leg ``overlap``'s tail assertion had gone stale under a peer round's wiring
  flip.** It asserted ``not plan.replaces`` — "the complex arms are registered
  wired=False and dispatch must be unchanged by this tranche" — and
  ``complex_fields.register_arms`` now registers ``wired=True``, so on a complex
  configuration ``plan_step`` composes all four sub-steps and the gate FAILED on
  its own staleness rather than on a defect. Nothing arithmetic moved: the kernel
  source hashes are unchanged and every other leg passed on the same run. The
  assertion is re-pointed at what is load-bearing now — all four slots composed,
  every one of them ``complex/Bloch``, asserted per case over the whole matrix
  instead of on ``periodic_kx`` alone.

CASE DISCIPLINE, inherited and non-negotiable: uint32 compares never allclose; a
non-power-of-two Courant FIRST; every leg proves the step MOVED STATE; predicted
nulls recorded WITH reasons; armed mutations launch-counted with DISARMED /
NEEDLE-MISSED classification; the signed-zero census floored.

RANDOM DATA CANNOT CATCH THE ZERO-CROSS-TERM CLASS — measured 0/16,384 words for
every wrong spelling — so leg ``zero_cross_terms`` runs EXHAUSTIVE sign/zero tables
rather than sampled rows. That is the one place this gate deliberately departs from
the real-field gate's shape, and the measurement is why.

STATED WEAKNESS, recorded because it is a real difference from the Triton
certification: THERE IS NO PTX-EQUIVALENT AUDIT on this executor.
``subnormal_policy`` can read every generated instruction and refuse a compile on a
policy violation; ``torch.mps.compile_shader`` exposes no disassembly, so the
mutation legs and this gate are the only arbiters.

PREDICTED NULLS RECORDED RATHER THAN MEASURED (the discipline the BFAST gate states
best): three defects are BYTE-INVISIBLE in float32 here and are pinned by
SOURCE-TEXT assertion in ``test_metal_complex_fields.py`` instead of by a mutation —
the complex add/sub being spelled as ``float2`` operators rather than two scalar
ops, the order of the two word stores within one cell, and the choice of which of
the three phased operands is rotated first. They appear in
``summary.predicted_nulls`` so the record shows they were NOT measured here.

Progress is one flushed line per case and the artifact is rewritten atomically after
every case (the progress-reporting rule).
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep. Requested EXPLICITLY, before anything
# resolves a policy, and stamped into the artifact: the claim is only as good as the
# precondition it was certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import complex_fields as cx  # noqa: E402
from meep_gpu.metal_kernels import device  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import preconditions, shaders, subnormal, templates  # noqa: E402
from meep_gpu.chunk_map import (RECORD_PRESENCE_WORD, membership_from_record,  # noqa: E402
                                 record_index)
from meep_gpu.pml import PML  # noqa: E402

import metal_gate_kit as kit  # noqa: E402
from gate_chunk_records import layer_record, two_run_record  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing
PERIODIC, METALLIC = templates.PERIODIC, templates.METALLIC


# ---------------------------------------------------------------------------
# The case matrix
# ---------------------------------------------------------------------------

#: Non-power-of-two FIRST. The PML coefficients are the same family of
#: non-representable multiplicands, so the grouping guard matters at every Courant
#: number and not only at the odd one.
COURANTS: Tuple[float, ...] = (0.35, 0.5)

#: (name, cell, boundaries, k_point). The k values are chosen to exercise three
#: distinct classes and are PINNED BY NAME:
#:
#: * ``k=0``          the reduction: no phase object is built at all, and the kernel
#:                    must be byte-identical to the plain complex path;
#: * ``k=0.3``        a generic interior point — phase neither real nor a root of
#:                    unity, so both planes of the rotation are live;
#: * ``k=0.5``        the BRILLOUIN EDGE, where ``grid.bloch_phase`` is EXACTLY
#:                    -1+0j. The imaginary plane is an exact zero there, which is
#:                    the one phase value at which a plane-wise collapse would be
#:                    invisible on random data — so it is a required case, not a
#:                    sampled one;
#: * two axes at once, with a PML on a phased axis, because ``_bloch_phases``
#:   ADMITS that pairing (settled by measurement on the array path, S:2322-2337) and
#:   a gate that only ran unabsorbed phased axes would not be testing what ships.
#: * ``periodic_ky`` and ``periodic_kxyz`` EXIST BECAUSE THEY WERE MISSING. Audited
#:   2026-08-15: every case here carried ``k_y = 0`` and ``leg_synthetic``'s phase
#:   sets only ever phased x, or x and z — so ``_phase_block("y", ...)``,
#:   ``_PHASED_OPERANDS["y"] = ("a_y", "c_y")`` and the ``py`` binding were emitted,
#:   fingerprinted, and NEVER LAUNCHED against the oracle by any leg. A wrong
#:   operand pair or a wrong index variable on the y branch was invisible to the
#:   whole gate. The three-axis case closes the ``ph111`` specialisation for the
#:   same reason.
CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any,
                     Tuple[float, float, float]], ...] = (
    ("periodic_k0", (1.2, 1.0, 0.9), "periodic", (0.0, 0.0, 0.0)),
    ("periodic_kx", (1.2, 1.0, 0.9), "periodic", (0.3, 0.0, 0.0)),
    ("periodic_edge_x", (1.2, 1.0, 0.9), "periodic", (0.5, 0.0, 0.0)),
    ("periodic_kxz", (1.2, 1.0, 0.9), "periodic", (0.3, 0.0, -0.25)),
    ("periodic_ky", (1.2, 1.0, 0.9), "periodic", (0.0, 0.4, 0.0)),
    ("periodic_kxyz", (1.2, 1.0, 0.9), "periodic", (0.3, -0.4, 0.2)),
    ("metallic_xy_kz", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.4)),
    ("metallic_all_k0", (1.1, 1.0, 0.9), "metallic", (0.0, 0.0, 0.0)),
)


def config(name: str) -> Tuple[str, Tuple[float, float, float], Any,
                               Tuple[float, float, float]]:
    """One case BY NAME.

    Legs used to reach into :data:`CONFIGS` positionally (``CONFIGS[3]`` for the
    mutation substrate, ``CONFIGS[1]`` for the precondition). Adding a case then
    silently repoints a leg at a different configuration, which is one keystroke
    away from the confound the mutation guard now refuses. By name, adding a case
    cannot move an existing leg.
    """
    for entry in CONFIGS:
        if entry[0] == name:
            return entry
    raise KeyError(f"no case named {name!r}; the matrix is "
                   f"{[entry[0] for entry in CONFIGS]}")

SUB_STEPS = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "sources": ("Ex", "Ey", "Ez"),
               "backward": False, "suffix": "_h"},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "sources": ("Hx", "Hy", "Hz"),
               "backward": True, "suffix": ""},
}
SIDES = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz"), "suffix": "", "step": "update_H"},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz"), "suffix": "_h", "step": "update_E"},
}

STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a
    complex128 scalar, so the product is a full complex multiply whose imaginary
    part comes out ``-0.0`` regardless of what ``imag`` held. Every complex
    construction in this file goes through the word view.
    """
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


def build(cell, boundaries, k_point, courant: float, seed: int, scale: float = 1.0,
          signed_zeros: bool = True, dimensions: int = 3,
          thickness: Any = None) -> Tuple[Any, Any, Any]:
    """A real Grid/Fields/PML triple with COMPLEX storage, seeded in the physical band.

    ``signed_zeros`` seeds a +-0 lattice into both planes of every volume. MEEP keeps
    float32 subnormals on arm64, so the signed-zero class is constructible on the
    HOST side here — unlike on a flushing x86 host.

    ``dimensions`` and ``thickness`` default to what every case in :data:`CONFIGS`
    uses and exist for leg ``admitted_domain``, which builds the two shapes the
    predicate ADMITS and the matrix does not contain: a 2-D run (a collapsed z, one
    stored cell) and an absorber missing on one axis.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=courant, k_point=k_point, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    pml = PML(grid=grid,
              thickness=(tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                               for a in range(3))
                         if thickness is None else thickness))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = (rng.standard_normal(grid.shape) * (0.37 * scale)).astype(np.float32)
        imag = (rng.standard_normal(grid.shape) * (0.29 * scale)).astype(np.float32)
        if signed_zeros:
            real.reshape(-1)[::17] = np.float32(-0.0)
            real.reshape(-1)[7::23] = np.float32(0.0)
            imag.reshape(-1)[3::19] = np.float32(-0.0)
            imag.reshape(-1)[11::29] = np.float32(0.0)
        array[...] = complex_from_planes(real, imag)
    # A SEEDED STATE IS A MID-RUN ONE: H was started from B long ago, so
    # ``stepping.separate_h_from_b`` must not run on it again -- the array path would
    # copy B into H and f_w on the dsigw cells at the first update_H, and no kernel
    # this gate launches does.
    fields.h_separated = True
    return grid, fields, pml


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


def boundary_codes(grid, pml) -> Tuple[int, int, int]:
    kinds = stepping._boundary_kinds(grid, pml)
    return tuple(METALLIC if kind == "metallic" else PERIODIC for kind in kinds)


def phase_table(grid, pml) -> Tuple[Optional[complex], ...]:
    return cx.bloch_phase_table(grid, stepping._boundary_kinds(grid, pml))


# ---------------------------------------------------------------------------
# The in-gate transcription: the MUTATION SUBSTRATE, pinned by leg `transcription`
# ---------------------------------------------------------------------------

#: vec.hpp's cycle_direction: target t takes (dsig, dsigu) = CURL_CYCLE[t].
CURL_CYCLE: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: Each target's two curl partners, as (source index, derivative axis): target 0
#: differences c (g2) along y and b (g1) along z, target 1 a along z and c along x,
#: target 2 b along x and a along y (step_generic.cpp's g1/g2 per component).
CURL_PARTNERS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
    ((2, 1), (1, 2)), ((0, 2), (2, 0)), ((1, 0), (0, 1)))

#: ``part_products.CONNECT_TOLERANCE``, restated: MEEP's phase_isclose window.
CONNECT_TOLERANCE = 1e-13


def _parts(values: Any) -> Tuple[Any, Any]:
    """The float32 real and imaginary parts of complex64 storage, as copies."""
    return (np.array(values.real, dtype=np.float32, copy=True),
            np.array(values.imag, dtype=np.float32, copy=True))


def _scale(coefficient: Any, values: Any) -> Any:
    """A REAL coefficient times complex storage, one float32 product per part.

    MEEP's DOCMP loops (``part_products.scale``). ``coefficient * values`` on a
    complex64 array is NumPy's complex multiply against ``(c + 0j)``, which fuses on
    this host and carries zero cross terms; that is not what either side forms.
    """
    c = np.asarray(coefficient, dtype=np.float32)
    re, im = _parts(values)
    return complex_from_planes(c * re, c * im)


def _shifted(array: Any, axis: int, backward: bool, code: int) -> Any:
    """``stepping._shift_up`` / ``_shift_down`` for one axis, Bloch phase or not.

    A plain roll (``np.roll`` is a pure gather and keeps every bit, signed zeros
    included); a METALLIC axis then serves an exact complex zero past its wall. A
    phased PERIODIC axis wraps PLAINLY: the stored window makes every read one cell
    over an owned word, and the phase lands on the UNSHIFTED operand instead
    (:func:`_connected`).
    """
    out = np.roll(array, 1 if backward else -1, axis=axis)
    if code == METALLIC:
        index: List[Any] = [slice(None)] * 3
        index[axis] = 0 if backward else array.shape[axis] - 1
        out[tuple(index)] = np.complex64(0)
    return out


def _connected(array: Any, axis: int, backward: bool,
               phase: Optional[complex]) -> Any:
    """``stepping._bloch_ghost_operand``: plane 0 of the unshifted operand, connected.

    The B side (forward differences) reads the not-owned word BELOW the window,
    ``thephase = 1.0 * conj(eikna)``; the D side the word ABOVE it, ``1.0 * eikna``,
    both formed in double. MEEP's ``connect_phase`` then copies within 1e-13 of +1,
    negates (a sign-bit operation per part) within 1e-13 of -1, and otherwise forms
    ``complex<realnum>(thephase) * word`` with every product and every sum rounded
    on its own: ``(a*re) - (b*im)``, ``(a*im) + (b*re)`` with ``(a, b)`` the phase
    rounded to float32. Taking the same factor in both directions is the classic
    sign error; mutation m5 plants it.
    """
    if phase is None:
        return array
    eikna = complex(phase)
    fr, fi = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
    pr, pi = 1.0 * fr - 0.0 * fi, 1.0 * fi + 0.0 * fr
    if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
        return array
    out = np.array(array, copy=True)
    index: List[Any] = [slice(None)] * 3
    index[axis] = 0
    face = tuple(index)
    if abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
        out[face] = -array[face]
        return out
    a, b = np.float32(pr), np.float32(pi)
    re, im = _parts(array[face])
    out[face] = complex_from_planes((a * re) - (b * im), (a * im) + (b * re))
    return out


def _presence(record: Any) -> Tuple[bool, bool, bool]:
    """Partner presence per axis, read from the record's presence word."""
    word = int(np.asarray(record).reshape(-1)[RECORD_PRESENCE_WORD])
    return tuple(bool(word & (1 << axis)) for axis in range(3))


def _membership(record: Any, half_integer: bool, axis: int, shape) -> Any:
    """One axis's membership line broadcast to the grid (the kernels' formula)."""
    line = membership_from_record(np.asarray(record).reshape(-1), half_integer, axis,
                                  int(shape[axis]))
    view = [1, 1, 1]
    view[axis] = int(shape[axis])
    return np.broadcast_to(line.reshape(view), tuple(int(n) for n in shape))


def _keep(collect: Optional[Dict[str, Any]], label: str, value: Any,
          where: Any = None) -> Any:
    """Snapshot one UNSTORED intermediate for the subnormal census, or do nothing.

    The census may not form its own version of these expressions: an independently
    written copy is a second transcription, and a second transcription is a second
    thing that can be wrong. So the intermediates are taken FROM THE REFERENCE
    ITSELF, which ``leg_transcription`` pins against ``stepping.py`` over full
    cycles. ``where`` keeps only the cells whose form forms the value (each cell
    takes one of two forms, and the census covers what the kernel forms there); a
    form no cell takes -- the recurrence on a target whose dsigu axis is invariant --
    forms nothing, and nothing is recorded for it. ``collect is None`` on every hot
    path.
    """
    if collect is not None:
        kept = np.array(value, copy=True)
        if where is not None:
            kept = kept[np.asarray(where)]
        if kept.size:
            collect[label] = kept
    return value


def reference_curls(sources: Sequence[Any], codes: Sequence[int], backward: bool,
                    dtdx: float, phases: Sequence[Optional[complex]], record: Any,
                    collect: Optional[Dict[str, Any]] = None) -> List[Any]:
    """The three UNMASKED ``dtdx * curl`` volumes, as ``stepping._curl_from_operands``.

    MEEP's left-to-right ``((g1s - g1) + g2) - g2s`` with both partners;
    ``dtdx * (g1s - g1)`` without g2; ``(-dtdx) * (g2s - g2)`` without g1; an exact
    zero without either (step_generic.cpp:72-76, :104-108). Each operation rounds
    once, per part. The shifted operands are plain rolls; each unshifted partner is
    read CONNECTED along the axis that term differences it on.

    ``collect`` additionally captures the shifted and the connected operands. They are
    unstored, and the connection on a phased plane is a real product whose
    ``b * im`` can land in the band while the field and the phase are outside it.
    """
    present = _presence(record)
    scale = np.float32(dtdx)
    curls: List[Any] = []
    for target, ((g1, axis1), (g2, axis2)) in enumerate(CURL_PARTNERS):
        g1s = _keep(collect, f"shift:{target}:g1",
                    _shifted(sources[g1], axis1, backward, codes[axis1]))
        g2s = _keep(collect, f"shift:{target}:g2",
                    _shifted(sources[g2], axis2, backward, codes[axis2]))
        g1c = _keep(collect, f"connected:{target}:g1",
                    _connected(sources[g1], axis1, backward, phases[axis1]))
        g2c = _keep(collect, f"connected:{target}:g2",
                    _connected(sources[g2], axis2, backward, phases[axis2]))
        if present[axis1] and present[axis2]:
            curls.append(_scale(scale, ((g1s - g1c) + g2c) - g2s))
        elif present[axis1]:
            curls.append(_scale(scale, g1s - g1c))
        elif present[axis2]:
            curls.append(_scale(-scale, g2s - g2c))
        else:
            curls.append(np.zeros_like(sources[g1]))
    return curls


def reference_curl(targets: Sequence[Any], aux: Sequence[Any],
                   sources: Sequence[Any], coefficients: Dict[str, Any],
                   codes: Sequence[int], backward: bool, dtdx: float,
                   phases: Sequence[Optional[complex]], record: Any,
                   collect: Optional[Dict[str, Any]] = None) -> None:
    """``stepping.step_B`` / ``step_D`` on complex storage, transcribed, in place.

    The two forms per structure chunk (``stepping._apply_curl_regions``): a target
    whose chunk carries its dsigu layer -- the record's membership on the curl's
    sub-lattice (half-integer for B, integer for D) along the dsigu axis -- steps
    the split-field recurrence and writes ``fu``; every other cell steps the
    dsig-only form ``((f*kms) - curl)*sinv`` and leaves ``fu`` untouched. Every
    product is one per part.

    ``collect`` gathers every value the kernel FORMS AND NEVER STORES, on the cells
    that form it: the shifted and connected operands, the three unmasked curls, the
    recurrence's partial results on its cells and the dsig-only form's on its own.
    """
    curls = reference_curls(sources, codes, backward, dtdx, phases, record, collect)
    for target in range(3):
        _keep(collect, f"curl:{target}", curls[target])

    pairs = (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)) if backward
             else ((0, 0), (1, 1), (2, 2)))
    for target, axis in pairs:
        if codes[axis] == METALLIC:
            index: List[Any] = [slice(None)] * 3
            index[axis] = 0
            curls[target][tuple(index)] = 0

    shape = targets[0].shape
    half = not backward
    for target, (dsig, dsigu) in enumerate(CURL_CYCLE):
        first, second = "xyz"[dsig], "xyz"[dsigu]
        kms, sinv = coefficients["kms_" + first], coefficients["sinv_" + first]
        kms_u, sinv_u = coefficients["kms_" + second], coefficients["sinv_" + second]
        inside = _membership(record, half, dsigu, shape)
        outside = ~inside
        f, fu, curl = targets[target], aux[target], curls[target]
        # The recurrence, evaluated whole-grid and kept on its own cells.
        p = np.array(fu, copy=True)
        fu_kms = _keep(collect, f"fu*kms:{target}", _scale(kms, p), inside)
        fu_kms_curl = _keep(collect, f"fu*kms-curl:{target}", fu_kms - curl, inside)
        n = _scale(sinv, fu_kms_curl)
        f_kms_u = _keep(collect, f"f*kms_u:{target}", _scale(kms_u, f), inside)
        f_plus = _keep(collect, f"f*kms_u+fu:{target}", f_kms_u + n, inside)
        f_minus = _keep(collect, f"f*kms_u+fu-fprev:{target}", f_plus - p, inside)
        recurrence = _scale(sinv_u, f_minus)
        # The dsig-only form, kept on every other cell.
        f_kms = _keep(collect, f"f*kms:{target}", _scale(kms, f), outside)
        f_kms_curl = _keep(collect, f"f*kms-curl:{target}", f_kms - curl, outside)
        dsig_only = _scale(sinv, f_kms_curl)
        f[...] = np.where(inside, recurrence, dsig_only)
        fu[...] = np.where(inside, n, fu)


def reference_constitutive(targets: Sequence[Any], aux: Sequence[Any],
                           sources: Sequence[Any],
                           inverse_epsilon: Optional[Sequence[Any]],
                           coefficients: Dict[str, Any], record: Any,
                           collect: Optional[Dict[str, Any]] = None) -> None:
    """``stepping._apply_constitutive_regions`` on the component's OWN axis.

    On a cell whose chunk carries the component's own (dsigw) layer -- the record's
    membership on the integer lattice for H, the half-integer one for E -- the
    accumulation ``f + ((kps*src) - (kms*prev))`` with ``f_w = src``; everywhere
    else the output select ``f = src``, ``f_w`` untouched. ``src`` is B, or
    ``D * inv_eps`` per part.

    ``collect`` gathers the source product, the two coefficient products and their
    difference, on the cells that form them.
    """
    shape = targets[0].shape
    half = inverse_epsilon is not None
    for target, axis in enumerate("xyz"):
        kps, kms = coefficients["kps_" + axis], coefficients["kms_" + axis]
        inside = _membership(record, half, target, shape)
        product = (np.array(sources[target], copy=True) if inverse_epsilon is None
                   else _scale(inverse_epsilon[target], sources[target]))
        _keep(collect, f"constitutive_product:{target}", product)
        previous = np.array(aux[target], copy=True)
        gain = _keep(collect, f"kps*fw:{target}", _scale(kps, product), inside)
        loss = _keep(collect, f"kms*fwprev:{target}", _scale(kms, previous), inside)
        difference = _keep(collect, f"kps*fw-kms*fwprev:{target}", gain - loss, inside)
        accumulated = targets[target] + difference
        targets[target][...] = np.where(inside, accumulated, product)
        aux[target][...] = np.where(inside, product, aux[target])


def curl_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def constitutive_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kps", "kms")}


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def run_curl_on_device(state: Dict[str, Any], pml, sub_step: str, codes, dtdx: float,
                       phases: Sequence[Optional[complex]],
                       flat: Optional[Dict[str, Any]] = None,
                       source: Optional[str] = None,
                       contract: str = shaders.CONTRACT_OFF,
                       counter: Optional[List[kit.Counter]] = None,
                       suffix: Optional[str] = None, *,
                       record: Any) -> Dict[str, Any]:
    """One curl launch THROUGH THE SHIPPED PLAN OBJECT, returning the host results.

    ``record`` is the chunk record of the division the reference steps with, and has
    no default (``gate_chunk_records``).
    """
    spec = SUB_STEPS[sub_step]
    arrays = {name: np.array(state[name], copy=True)
              for name in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: np.array(state["fu_" + n], copy=True)
                   for n in spec["targets"]})
    if flat is None:
        flat = {key: np.asarray(value).reshape(-1) for key, value in
                curl_coefficients(
                    pml, spec["suffix"] if suffix is None else suffix).items()}
    residency = metal_launch.Residency()
    flags, _ = cx.phase_arguments(tuple(phases), backward=bool(spec["backward"]))
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF:
        text = source if source is not None else cx.bloch_curl_source(
            codes, spec["backward"], flags, contract)
        function = metal_launch.compile_source(text).bloch_pml_curl_step
        if counter is not None:
            function = kit.Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = cx.plan_complex_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, phases, dtdx, residency,
        functions=functions, chunk_bounds=record)
    plan.run()
    residency.sync_out()
    return arrays


def run_constitutive_on_device(state: Dict[str, Any], pml, side: str,
                               inverse_epsilon: Optional[Dict[str, Any]] = None,
                               source: Optional[str] = None,
                               contract: str = shaders.CONTRACT_OFF,
                               suffix: Optional[str] = None,
                               counter: Optional[List[kit.Counter]] = None,
                               *, record: Any) -> Dict[str, Any]:
    spec = SIDES[side]
    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    arrays = {name: np.array(state[name], copy=True) for name in names}
    if side == "E":
        arrays.update({"inv_eps_" + n: inverse_epsilon[n] for n in spec["targets"]})
    chosen = spec["suffix"] if suffix is None else suffix
    flat = {key: np.asarray(value).reshape(-1) for key, value in
            constitutive_coefficients(pml, chosen).items()}
    residency = metal_launch.Residency()
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF:
        text = source if source is not None else cx.bloch_constitutive_source(
            side, contract)
        function = metal_launch.compile_source(text).bloch_constitutive_step
        if counter is not None:
            function = kit.Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = cx.plan_complex_constitutive_from_arrays(
        side, arrays, flat, residency, functions=functions,
        suffix_key=chosen, chunk_bounds=record)
    plan.run()
    residency.sync_out()
    return arrays


# ---------------------------------------------------------------------------
# LEG transcription — the in-gate reference vs stepping.py, full cycles
# ---------------------------------------------------------------------------

def leg_transcription(payload: Dict[str, Any], out: str, cycles: int = 3) -> None:
    """THIS RUNS BEFORE ANY SHADER IS TRUSTED.

    Without it, "bit-identical" could mean the kernel faithfully reproduced whatever
    this file wrote, twice.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries, k_point in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, k_point, courant, 20260815)
            initial = snapshot(fields)
            codes = boundary_codes(grid, pml)
            phases = phase_table(grid, pml)
            dtdx = float(grid.dt / grid.dx)
            inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
            record = layer_record(grid, pml)

            # The array path, `cycles` complete steps.
            for _ in range(cycles):
                stepping.step_B(fields, pml)
                stepping.update_H(fields, pml)
                stepping.step_D(fields, pml)
                stepping.update_E(fields, pml)
            oracle = snapshot(fields)

            # The transcription, the same cycles from the same seed.
            mirror = {key: np.array(value, copy=True) for key, value in initial.items()}
            for _ in range(cycles):
                for sub_step, spec in SUB_STEPS.items():
                    reference_curl(
                        [mirror[n] for n in spec["targets"]],
                        [mirror["fu_" + n] for n in spec["targets"]],
                        [mirror[n] for n in spec["sources"]],
                        curl_coefficients(pml, spec["suffix"]),
                        codes, spec["backward"], dtdx, phases, record)
                    if sub_step == "step_B":
                        side_spec = SIDES["H"]
                        reference_constitutive(
                            [mirror[n] for n in side_spec["targets"]],
                            [mirror[n] for n in side_spec["aux"]],
                            [mirror[n] for n in side_spec["sources"]],
                            None, constitutive_coefficients(pml, side_spec["suffix"]),
                            record)
                    else:
                        side_spec = SIDES["E"]
                        reference_constitutive(
                            [mirror[n] for n in side_spec["targets"]],
                            [mirror[n] for n in side_spec["aux"]],
                            [mirror[n] for n in side_spec["sources"]],
                            [inverse[n] for n in side_spec["targets"]],
                            constitutive_coefficients(pml, side_spec["suffix"]),
                            record)

            bad = {n: differing(mirror[n], oracle[n]) for n in STATE if n in oracle}
            moved = sum(differing(oracle[n], initial[n]) for n in oracle)
            row = {"case": name, "courant": courant, "cycles": cycles,
                   "moved_words": moved, "k_point": list(k_point),
                   "phases": [None if p is None else [float(np.float32(p.real)),
                                                      float(np.float32(p.imag))]
                              for p in phases],
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[transcription] {name:<17} C={courant} moved={moved} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["transcription"] = rows
            save(payload, out)
            kit.assert_moved(moved, f"{name}/C={courant} {cycles} cycles")
            assert not row["differing"], row
    payload["counts"]["transcription"] = sum(
        len(row["differing"]) + len(STATE) for row in rows)


# ---------------------------------------------------------------------------
# LEG reference — the Metal kernels vs stepping.py, engine objects
# ---------------------------------------------------------------------------

def leg_reference(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    for name, cell, boundaries, k_point in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, k_point, courant, 20260815)
            base = snapshot(fields)
            codes = boundary_codes(grid, pml)
            phases = phase_table(grid, pml)
            dtdx = float(grid.dt / grid.dx)
            record = layer_record(grid, pml)
            for sub_step, spec in SUB_STEPS.items():
                restore(fields, base)
                getattr(stepping, sub_step)(fields, pml)
                after = snapshot(fields)
                got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                         record=record)
                names = tuple(spec["targets"]) + tuple("fu_" + n
                                                       for n in spec["targets"])
                bad = {n: differing(got[n], after[n]) for n in names}
                moved = sum(differing(after[n], base[n]) for n in names)
                compared += len(names)
                row = {"case": name, "courant": courant, "sub_step": sub_step,
                       "moved_words": moved,
                       "differing": {k: v for k, v in bad.items() if v}}
                rows.append(row)
                log(f"[reference] {name:<17} C={courant} {sub_step} moved={moved} "
                    f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                    f"({time.time() - started:.1f}s)")
                payload["legs"]["reference"] = rows
                save(payload, out)
                kit.assert_moved(moved, f"{name}/{sub_step}")
                assert not row["differing"], row
    payload["counts"]["reference"] = compared


# ---------------------------------------------------------------------------
# LEG constitutive
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    for name, cell, boundaries, k_point in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, k_point, courant, 20260815)
            base = snapshot(fields)
            inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
            for side, spec in SIDES.items():
                restore(fields, base)
                getattr(stepping, spec["step"])(fields, pml)
                after = snapshot(fields)
                got = run_constitutive_on_device(base, pml, side, inverse,
                                                 record=layer_record(grid, pml))
                names = tuple(spec["targets"]) + tuple(spec["aux"])
                bad = {n: differing(got[n], after[n]) for n in names}
                moved = sum(differing(after[n], base[n]) for n in names)
                compared += len(names)
                row = {"case": name, "courant": courant, "side": side,
                       "moved_words": moved,
                       "differing": {k: v for k, v in bad.items() if v}}
                rows.append(row)
                log(f"[constitutive] {name:<17} C={courant} update_{side} "
                    f"moved={moved} "
                    f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                    f"({time.time() - started:.1f}s)")
                payload["legs"]["constitutive"] = rows
                save(payload, out)
                kit.assert_moved(moved, f"{name}/update_{side}")
                assert not row["differing"], row
    payload["counts"]["constitutive"] = compared


# ---------------------------------------------------------------------------
# LEG synthetic — odd shapes, every boundary triple, every phase set, both guards
# ---------------------------------------------------------------------------

SHAPES = ((7, 5, 3), (4, 4, 4), (9, 1, 6))


def synthetic_state(shape, seed: int, scale: float = 1.0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    state: Dict[str, Any] = {}
    for name in STATE:
        real = (rng.standard_normal(shape) * (0.41 * scale)).astype(np.float32)
        imag = (rng.standard_normal(shape) * (0.33 * scale)).astype(np.float32)
        real.reshape(-1)[::13] = np.float32(-0.0)
        imag.reshape(-1)[5::17] = np.float32(-0.0)
        state[name] = complex_from_planes(real, imag)
    return state


def synthetic_coefficients(shape, seed: int,
                           stems=("kms", "sinv")) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 7)
    flat: Dict[str, Any] = {}
    for axis, extent in zip("xyz", shape):
        for stem in stems:
            # Deliberately spans NEGATIVE values: a real PML's leading `kms` goes
            # negative, and a positive-only sweep would never exercise the sign that
            # produces -0.0 through the recurrence.
            flat[f"{stem}_{axis}"] = (
                rng.standard_normal(extent) * 0.6 + 0.3).astype(np.float32)
    return flat


#: Three distinct phase values, one per axis, so a block that read the WRONG axis's
#: pair would land a different rotation rather than the same one twice.
_PHI_X = complex(np.exp(2j * np.pi * 0.3))
_PHI_Y = complex(np.exp(-2j * np.pi * 0.4))
_PHI_Z = complex(np.exp(-2j * np.pi * 0.25))
_EDGE = complex(-1.0, 0.0)


def _phase_sets(codes: Sequence[int]) -> List[Tuple[Optional[complex], ...]]:
    """Every phase triple the synthetic sweep runs for one boundary triple.

    PER-AXIS SINGLES ARE THE POINT. The earlier revision phased x, x-at-the-edge and
    x-with-z and nothing else, so the y block — its ``(j == nyi - 1)`` predicate, its
    ``("a_y", "c_y")`` operand pair and the ``py`` binding — was never launched
    anywhere in the gate. Singles on each axis isolate a block; the all-three set
    exercises ``ph111``, which was also emitted and never run.
    """
    sets: List[Tuple[Optional[complex], ...]] = [(None, None, None)]
    singles = ((0, _PHI_X), (1, _PHI_Y), (2, _PHI_Z))
    for axis, value in singles:
        if codes[axis] == PERIODIC:
            triple: List[Optional[complex]] = [None, None, None]
            triple[axis] = value
            sets.append(tuple(triple))
    if codes[0] == PERIODIC:
        # The Brillouin edge, whose imaginary plane is an EXACT zero — the one phase
        # value at which a plane-wise collapse is invisible on random data.
        sets.append((_EDGE, None, None))
    if codes[0] == PERIODIC and codes[2] == PERIODIC:
        sets.append((_PHI_X, None, _PHI_Z))
    if all(code == PERIODIC for code in codes):
        sets.append((_PHI_X, _PHI_Y, _PHI_Z))
    return sets


def leg_synthetic(payload: Dict[str, Any], out: str) -> None:
    """Odd shapes x boundary triples x phase sets x dtdx x GUARD x sub-step.

    THE CHUNK RECORD IS A TWO-RUN ONE (``gate_chunk_records.two_run_record``): on
    every resolved axis the integer and half-integer lattices hold two runs whose
    low boundaries are one row apart, so both curl forms, every boundary plane and a
    lattice swap are exercised on every shape, and a collapsed axis carries no
    presence bit (the absent-partner curl). The reference keys its forms on the same
    record.

    THE GUARD IS LOAD-BEARING HERE, AND THE ASSERTION SAYS SO. Every product this
    family emits is a bare per-part ``a*b`` beside a ``+``/``-``, so ``contract(fast)``
    may fuse it and the shipped ``contract(off)`` is what keeps MEEP's two
    roundings. The sweep records the per-row delta of the ``fast`` build and asserts
    that it moves words somewhere: a zero count would say the directive does not
    reach the emitted code.

    THE CONSTITUTIVE KERNEL RUNS HERE TOO, AND UNTIL THIS ROUND IT DID NOT. Every
    odd-shape row swept the CURL only; the constitutive kernel had been launched on
    exactly two shapes ever, both from the engine matrix, both fully three
    dimensional. Its flat-index decode and its per-axis coefficient reads were
    therefore never exercised on ``(7, 5, 3)``, on the cube ``(4, 4, 4)`` — where a
    swapped extent is invisible — or on the COLLAPSED ``(9, 1, 6)``, which is the
    2-D shape a real run has. The fossil of the missing sweep was
    :func:`synthetic_coefficients`'s ``stems`` parameter: it existed, it defaulted
    to the curl's pair, and nothing ever passed the constitutive's.

    AND THE SWEEP CARRIES ITS OWN SUBNORMAL WINDOW. The claim over these 900-odd
    rows is the same claim leg ``precondition`` bounds on the engine matrix, so the
    rows are censused rather than assumed: one window per shape over the seeded
    state, every result and the reference's own unstored intermediates.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    guard_differences = 0
    compared = 0
    windows: Dict[str, Any] = {}
    # PER-AXIS ROTATION COVERAGE, counted rather than assumed. The audited revision
    # of this leg never phased y at all; a count in the artifact is what makes that
    # kind of hole visible instead of implied by a list of tuples.
    phased_rows = [0, 0, 0]
    all_three_rows = 0
    constitutive_rows = 0
    for shape in SHAPES:
        state = synthetic_state(shape, 424242)
        record = two_run_record(shape)
        # The intermediate floor is per FORM: each is censused on the cells whose
        # form forms it, a two-run line's share of a small shape (the admitted-domain
        # leg's floor).
        window = preconditions.SubnormalWindow(0, 1, per_array_words=64,
                                               per_intermediate_words=8)
        windows[str(shape)] = window
        for volume, array in state.items():
            window.observe(f"in:{shape}:{volume}", array, step=0)
        for codes in itertools.product((PERIODIC, METALLIC), repeat=3):
            phase_sets = _phase_sets(codes)
            for phases in phase_sets:
                for dtdx in (0.35, 0.5):
                    for sub_step, spec in SUB_STEPS.items():
                        flat = synthetic_coefficients(shape, 99)
                        expect = {n: np.array(state[n], copy=True)
                                  for n in tuple(spec["targets"])
                                  + tuple("fu_" + t for t in spec["targets"])}
                        collect: Dict[str, Any] = {}
                        reference_curl(
                            [expect[n] for n in spec["targets"]],
                            [expect["fu_" + n] for n in spec["targets"]],
                            [state[n] for n in spec["sources"]],
                            {k: v.reshape(_broadcast(shape, k))
                             for k, v in flat.items()},
                            codes, spec["backward"], dtdx, phases, record,
                            collect=collect)
                        got = run_curl_on_device(state, None, sub_step, codes, dtdx,
                                                 phases, flat=flat, record=record)
                        names = tuple(expect)
                        bad = {n: differing(got[n], expect[n]) for n in names}
                        moved = sum(differing(expect[n], state[n]) for n in names)
                        compared += len(names)
                        fast = run_curl_on_device(state, None, sub_step, codes, dtdx,
                                                  phases, flat=flat,
                                                  contract=shaders.CONTRACT_FAST,
                                                  record=record)
                        guard_delta = sum(differing(fast[n], expect[n]) for n in names)
                        guard_differences += int(guard_delta > 0)
                        for volume in names:
                            window.observe(f"out:{shape}:{sub_step}:{volume}",
                                           expect[volume], step=1)
                        for tag, volume in collect.items():
                            window.observe_intermediate(f"{shape}:{sub_step}:{tag}",
                                                        volume, step=1)
                        row = {"shape": list(shape), "codes": list(codes),
                               "phased": [int(p is not None) for p in phases],
                               "dtdx": dtdx, "sub_step": sub_step,
                               "moved_words": moved, "guard_delta_words": guard_delta,
                               "differing": {k: v for k, v in bad.items() if v}}
                        rows.append(row)
                        for axis, flag in enumerate(row["phased"]):
                            phased_rows[axis] += flag
                        all_three_rows += int(all(row["phased"]))
                        payload["legs"]["synthetic"] = rows
                        kit.assert_moved(moved, f"{shape}/{codes}/{sub_step}")
                        assert not row["differing"], row

        # --- the CONSTITUTIVE kernel on the same odd shapes ---------------------
        # It reads no neighbour, so it takes no boundary triple and no phase; what
        # it DOES take is the flat-index decode and a coefficient indexed on the
        # component's own axis, and those are exactly what a shape with three
        # different extents (and one with a collapsed axis) discriminates.
        inv_eps = {"inv_eps_" + n: np.abs(
            synthetic_state(shape, 5150 + index)[n].real).astype(np.float32) + 0.25
            for index, n in enumerate(SIDES["E"]["targets"])}
        for side, spec in SIDES.items():
            flat = synthetic_coefficients(shape, 99, stems=("kps", "kms"))
            arrays = {n: np.array(state[n], copy=True)
                      for n in tuple(spec["targets"]) + tuple(spec["aux"])
                      + tuple(spec["sources"])}
            if side == "E":
                arrays.update(inv_eps)
            expect = {n: np.array(arrays[n], copy=True)
                      for n in tuple(spec["targets"]) + tuple(spec["aux"])}
            collect = {}
            reference_constitutive(
                [expect[n] for n in spec["targets"]],
                [expect[n] for n in spec["aux"]],
                [arrays[n] for n in spec["sources"]],
                None if side == "H" else [inv_eps["inv_eps_" + n]
                                          for n in spec["targets"]],
                {k: v.reshape(_broadcast(shape, k)) for k, v in flat.items()},
                record, collect=collect)
            residency = metal_launch.Residency()
            plan = cx.plan_complex_constitutive_from_arrays(
                side, arrays, flat, residency, suffix_key=f":{shape}",
                chunk_bounds=record)
            plan.run()
            residency.sync_out()
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            bad = {n: differing(arrays[n], expect[n]) for n in names}
            moved = sum(differing(expect[n], state[n]) for n in names)
            compared += len(names)
            for volume in names:
                window.observe(f"out:{shape}:update_{side}:{volume}",
                               expect[volume], step=1)
            for tag, volume in collect.items():
                window.observe_intermediate(f"{shape}:update_{side}:{tag}", volume,
                                            step=1)
            row = {"shape": list(shape), "side": side, "sub_step": "update_" + side,
                   "moved_words": moved,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            constitutive_rows += 1
            payload["legs"]["synthetic"] = rows
            kit.assert_moved(moved, f"{shape}/update_{side}")
            assert not row["differing"], row

        preconditions.assert_clean_or_refuse(window, f"synthetic {shape}")
        log(f"[synthetic] shape={shape} rows={len(rows)} "
            f"guard_differences={guard_differences} "
            f"subnormal={window.subnormal_words}/{window.observed_words} "
            f"({time.time()-started:.1f}s)")
        save(payload, out)
    payload["legs"]["synthetic_subnormal"] = {
        str(shape): windows[str(shape)].report() for shape in SHAPES}
    payload["legs"]["synthetic_constitutive_rows"] = constitutive_rows
    assert constitutive_rows == 2 * len(SHAPES), (
        f"the constitutive kernel ran on {constitutive_rows} synthetic rows and "
        f"there are {len(SHAPES)} shapes x 2 sides; a shape it never ran on is a "
        f"shape whose index decode was never exercised")
    payload["legs"]["synthetic_guard"] = {
        "guard_differences": guard_differences, "rows": len(rows),
        "reading": ("rows whose contract(fast) build moved at least one word: every "
                    "product is a bare per-part a*b beside a +/-, so the shipped "
                    "contract(off) is what keeps MEEP's separate roundings")}
    payload["legs"]["synthetic_phase_coverage"] = {
        "rows_phasing_x": phased_rows[0],
        "rows_phasing_y": phased_rows[1],
        "rows_phasing_z": phased_rows[2],
        "rows_phasing_all_three": all_three_rows,
        "rows": len(rows),
    }
    payload["counts"]["synthetic"] = compared
    save(payload, out)
    for axis, count in zip("xyz", phased_rows):
        assert count > 0, (
            f"no synthetic row carried a Bloch phase on {axis}. That axis's rotation "
            f"block, its wrapped-lane predicate and its phase binding are then "
            f"emitted, fingerprinted and never launched — which is exactly the hole "
            f"this counter exists to refuse")
    assert all_three_rows > 0, (
        "no synthetic row phased all three axes, so the ph111 specialisation is "
        "emitted and never launched")
    assert guard_differences > 0, (
        "flipping the contraction directive moved no word in any synthetic row: "
        "the bare per-part products would then never be contracted, and the "
        "directive would not reach the emitted code")


def _broadcast(shape, key: str):
    axis = "xyz".index(key[-1])
    return tuple(shape[axis] if a == axis else 1 for a in range(3))


# ---------------------------------------------------------------------------
# LEG contraction — where the directive IS load-bearing, measured on both arms
# ---------------------------------------------------------------------------

def leg_contraction(payload: Dict[str, Any], out: str) -> None:
    """The contraction directive, measured on BOTH kernels against the reference.

    Every product is a bare per-part ``a*b`` beside a ``+``/``-``: the recurrence's
    ``(p*kms) - curl``, the connection's ``(z.x*p.x) - (z.y*p.y)``, the
    accumulation's ``(kps*src) - (kms*prev)``. Three measurements per sub-step:

    * ``off`` vs the reference -- must be 0: the shipped directive keeps MEEP's
      separate roundings;
    * ``fast`` vs ``off`` -- must DIFFER: the selector reaches the emitted code, so
      the pinned mode is load-bearing rather than decorative;
    * ``fast`` vs the reference -- recorded: the words a contracting build moves.

    The emitted helpers have one body, so there is no arm axis to measure here.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    shape = (7, 5, 3)
    state = synthetic_state(shape, 424242)
    record = two_run_record(shape)
    flat = synthetic_coefficients(shape, 99)
    codes = (PERIODIC, PERIODIC, PERIODIC)
    phases = (complex(np.exp(2j * np.pi * 0.3)), None,
              complex(np.exp(-2j * np.pi * 0.25)))
    dtdx = 0.35
    for sub_step, spec in SUB_STEPS.items():
        names = tuple(spec["targets"]) + tuple("fu_" + t for t in spec["targets"])
        expect = {n: np.array(state[n], copy=True) for n in names}
        reference_curl([expect[n] for n in spec["targets"]],
                       [expect["fu_" + n] for n in spec["targets"]],
                       [state[n] for n in spec["sources"]],
                       {k: v.reshape(_broadcast(shape, k)) for k, v in flat.items()},
                       codes, spec["backward"], dtdx, phases, record)
        flags, _ = cx.phase_arguments(tuple(phases), backward=bool(spec["backward"]))
        built: Dict[str, Dict[str, Any]] = {}
        for mode in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
            source = cx.bloch_curl_source(codes, spec["backward"], flags,
                                          mode)
            built[mode] = run_curl_on_device(state, None, sub_step, codes, dtdx,
                                             phases, flat=flat, source=source,
                                             record=record)

        def delta(left: str, right: Optional[str] = None) -> int:
            target = expect if right is None else built[right]
            return sum(differing(built[left][n], target[n]) for n in names)

        row = {"sub_step": sub_step,
               "off_vs_reference": delta(shaders.CONTRACT_OFF),
               "fast_vs_reference": delta(shaders.CONTRACT_FAST),
               "off_vs_fast": delta(shaders.CONTRACT_OFF, shaders.CONTRACT_FAST),
               "words": len(names) * 2 * int(np.prod(shape))}
        rows.append(row)
        log(f"[contraction] {sub_step} off_vs_ref={row['off_vs_reference']} "
            f"fast_vs_ref={row['fast_vs_reference']} "
            f"off_vs_fast={row['off_vs_fast']} ({time.time() - started:.1f}s)")
        payload["legs"]["contraction"] = rows
        save(payload, out)
        assert row["off_vs_reference"] == 0, row
        assert row["off_vs_fast"] > 0, (
            f"{sub_step}: flipping the contraction directive changed NOTHING on bare "
            f"per-part products. The selector does not reach the emitted code, and "
            f"the pinned mode is then a decoration")

    inv_eps = {"inv_eps_" + n: (np.abs(state[n].real).astype(np.float32) + 0.25)
               for n in SIDES["E"]["targets"]}
    for side, spec in SIDES.items():
        flat_c = synthetic_coefficients(shape, 99, stems=("kps", "kms"))
        names = tuple(spec["targets"]) + tuple(spec["aux"])
        expect = {n: np.array(state[n], copy=True) for n in names}
        reference_constitutive(
            [expect[n] for n in spec["targets"]], [expect[n] for n in spec["aux"]],
            [state[n] for n in spec["sources"]],
            None if side == "H" else [inv_eps["inv_eps_" + n]
                                      for n in spec["targets"]],
            {k: v.reshape(_broadcast(shape, k)) for k, v in flat_c.items()}, record)
        built = {}
        for mode in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
            arrays = {n: np.array(state[n], copy=True)
                      for n in names + tuple(spec["sources"])}
            if side == "E":
                arrays.update(inv_eps)
            residency = metal_launch.Residency()
            function = metal_launch.compile_source(
                cx.bloch_constitutive_source(side, mode)
            ).bloch_constitutive_step
            plan = cx.plan_complex_constitutive_from_arrays(
                side, arrays, flat_c, residency,
                functions={shaders.CONTRACT_OFF: function},
                suffix_key=f":{mode}", chunk_bounds=record)
            plan.run()
            residency.sync_out()
            built[mode] = arrays

        def cdelta(left: str, right: Optional[str] = None,
                   _names=names, _built=built, _expect=expect) -> int:
            target = _expect if right is None else _built[right]
            return sum(differing(_built[left][n], target[n]) for n in _names)

        row = {"sub_step": "update_" + side,
               "off_vs_reference": cdelta(shaders.CONTRACT_OFF),
               "fast_vs_reference": cdelta(shaders.CONTRACT_FAST),
               "off_vs_fast": cdelta(shaders.CONTRACT_OFF, shaders.CONTRACT_FAST),
               "words": len(names) * 2 * int(np.prod(shape))}
        rows.append(row)
        log(f"[contraction] update_{side} off_vs_ref={row['off_vs_reference']} "
            f"fast_vs_ref={row['fast_vs_reference']} "
            f"off_vs_fast={row['off_vs_fast']} ({time.time() - started:.1f}s)")
        payload["legs"]["contraction"] = rows
        save(payload, out)
        assert row["off_vs_reference"] == 0, row
        assert row["off_vs_fast"] > 0, (
            f"update_{side}: flipping the contraction directive changed nothing on "
            f"the accumulation's (kps*src) - (kms*prev): the selector does not reach "
            f"the constitutive kernel")
    payload["counts"]["contraction"] = len(rows) * 3


# ---------------------------------------------------------------------------
# LEG multi_step — the auxiliaries are STATE
# ---------------------------------------------------------------------------

def leg_multi_step(payload: Dict[str, Any], out: str, cycles: int = 4) -> None:
    """B -> H -> D -> E cycles from one state, auxiliaries included.

    A kernel that is right for one launch and wrong forever after is identical in
    the single-launch legs and only diverges here.

    THE DIGEST PAIR IS NOW A CHECK. It used to digest two DIFFERENT key sets — the
    device state picks up the three `inv_eps_E*` volumes the constitutive helper
    returns — so the two hashes disagreed on every row beside `differing: {}` and
    could not have agreed on any. Digested over the oracle's names they are equal,
    and the assertion makes them corroborate the per-name comparison.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    for name, cell, boundaries, k_point in CONFIGS:
        grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
        base = snapshot(fields)
        codes = boundary_codes(grid, pml)
        phases = phase_table(grid, pml)
        dtdx = float(grid.dt / grid.dx)
        inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
        for _ in range(cycles):
            stepping.step_B(fields, pml)
            stepping.update_H(fields, pml)
            stepping.step_D(fields, pml)
            stepping.update_E(fields, pml)
        oracle = snapshot(fields)

        record = layer_record(grid, pml)
        device_state = {k: np.array(v, copy=True) for k, v in base.items()}
        for _ in range(cycles):
            for sub_step, side in (("step_B", "H"), ("step_D", "E")):
                got = run_curl_on_device(device_state, pml, sub_step, codes, dtdx,
                                         phases, record=record)
                device_state.update(got)
                got = run_constitutive_on_device(device_state, pml, side, inverse,
                                                 record=record)
                device_state.update(got)
        bad = {n: differing(device_state[n], oracle[n]) for n in oracle}
        moved = sum(differing(oracle[n], base[n]) for n in oracle)
        compared += len(oracle)
        # THE DIGEST PAIR IS DIGESTED OVER ONE KEY SET, and that is a correction.
        # `run_constitutive_on_device` returns the three `inv_eps_E*` volumes it was
        # handed, so `device_state` carries 27 names against the oracle's 24 and
        # `state_digest(device_state)` could NEVER equal `state_digest(oracle)` — the
        # artifact published two digests that disagreed on all 8 rows beside
        # `differing: {}`, which reads to any artifact reader as a divergence the leg
        # failed to notice. Measured: restricted to the oracle's names the device
        # digest is EXACTLY the oracle's. Restricting it makes the pair comparable,
        # and the assertion below makes it a CHECK rather than a decoration — an
        # independent whole-state cross-check of the per-name comparison, which is
        # the only thing a digest was ever worth here.
        digest_oracle = kit.state_digest(oracle)
        digest_device = kit.state_digest({n: device_state[n] for n in oracle})
        row = {"case": name, "cycles": cycles, "moved_words": moved,
               "digest_oracle": digest_oracle,
               "digest_device": digest_device,
               "digest_scope": sorted(oracle),
               "device_only_names": sorted(set(device_state) - set(oracle)),
               "differing": {k: v for k, v in bad.items() if v}}
        rows.append(row)
        log(f"[multi_step] {name:<17} cycles={cycles} moved={moved} "
            f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["multi_step"] = rows
        save(payload, out)
        kit.assert_moved(moved, f"{name} {cycles} cycles")
        assert not row["differing"], row
        assert digest_device == digest_oracle, (
            f"{name}: the whole-state digests disagree ({digest_device} vs "
            f"{digest_oracle}) while the per-name comparison found nothing. One of "
            f"the two is wrong, and a digest that cannot corroborate the comparison "
            f"is worse than no digest at all")
    payload["counts"]["multi_step"] = compared


# ---------------------------------------------------------------------------
# LEG part_products — EXHAUSTIVE signed-zero tables for the three complex helpers
# ---------------------------------------------------------------------------

_TABLE_VALUES = (np.float32(1.5), np.float32(-2.25), np.float32(0.0),
                 np.float32(-0.0))

_DIAGNOSTIC_SHADERS = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
kernel void shipped(device float2* o [[buffer(0)]],
                    device const float2* a [[buffer(1)]],
                    device const float* c [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return; float2 z = a[i]; float k = c[i];
    o[i] = float2(__RE__, __IM__);
}
"""

#: A REAL coefficient times a field (``c_mul_field_left``), against
#: ``part_products.scale``: one float32 product per part, MEEP's DOCMP loops.
#:
#: ``shipped`` IS THE SHIPPED TEXT, checked by
#: :func:`_assert_diagnostic_arms_are_the_shipped_spelling` against
#: ``templates.complex_helpers`` (the helper names the coefficient ``c``, this shader
#: ``k``). The two ``previous_*`` arms are the package's previous spellings --
#: NumPy's complex multiply against ``(c + 0j)``, fused and unfused -- whose zero
#: cross terms flip the sign of a zero part; they are the inverted mutations of this
#: table and MUST differ. Random data does not reach them (the defect needs an exact
#: zero product beside a zero cross term of the other sign), which is why the table
#: is exhaustive.
_FIELD_LEFT_ARMS: Dict[str, Tuple[str, str, bool]] = {
    # label: (re, im, must_match)
    "shipped": ("z.x * k", "z.y * k", True),
    # Multiplication is commutative on every non-NaN word, signed zero included, so
    # the orientation in a helper's NAME moves no bit. Measured, not asserted.
    "coefficient_left_orientation": ("k * z.x", "k * z.y", True),
    "previous_fma_v1_zero_cross_terms": ("fma(z.x, k, -(z.y * 0.0f))",
                                         "fma(z.x, 0.0f, (z.y * k))", False),
    "previous_naive_zero_cross_terms": ("(z.x * k) - (z.y * 0.0f)",
                                        "(z.x * 0.0f) + (z.y * k)", False),
}

_FULL_PRODUCT_SHADER = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
kernel void shipped(device float2* o [[buffer(0)]],
                    device const float2* a [[buffer(1)]],
                    device const float2* b [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return; float2 z = a[i]; float2 p = b[i];
    o[i] = float2(__RE__, __IM__);
}
"""

#: The Bloch connection (``c_mul``), MEEP's ``complex<realnum>(thephase) * word``
#: with every product and every sum rounded on its own, against
#: ``part_products.connect_parts``. ``None`` means RECORD, not assert: every product
#: on the exhaustive table is exact, so the fused previous arm agrees there and its
#: discrimination lives on the RANDOM table below, where it must differ.
_FULL_PRODUCT_ARMS: Dict[str, Tuple[str, str, Optional[bool]]] = {
    "shipped": ("(z.x * p.x) - (z.y * p.y)", "(z.x * p.y) + (z.y * p.x)", True),
    # IEEE multiplication and two-operand addition are commutative, so the phase
    # may sit on either side: the array path writes (a*re) - (b*im).
    "phase_left_orientation": ("(p.x * z.x) - (p.y * z.y)",
                               "(p.x * z.y) + (p.y * z.x)", True),
    "previous_fma_v1": ("fma(z.x, p.x, -(z.y * p.y))", "fma(z.x, p.y,  (z.y * p.x))",
                        None),
    "plane_wise": ("z.x * p.x", "z.y * p.y", False),
    "conjugated_cross_term": ("(z.x * p.x) + (z.y * p.y)",
                              "(z.x * p.y) + (z.y * p.x)", False),
    "imaginary_cross_negated": ("(z.x * p.x) - (z.y * p.y)",
                                "(z.x * p.y) - (z.y * p.x)", False),
}

#: The arms of the RANDOM connection table (4,096 inexact products): the shipped
#: spelling must reproduce ``connect_parts`` and the fused previous spelling must not.
_RANDOM_PRODUCT_ARMS: Tuple[Tuple[str, bool], ...] = (
    ("shipped", True), ("previous_fma_v1", False))

#: MEEP's CONNECT_NEGATE: a unary minus per part within 1e-13 of -1, never a
#: product by ``(-1, 0)``. The shipped spelling is ``-x`` (``complex_fields``'
#: connection block); against ``numpy.negative`` per part.
_NEGATE_ARMS: Dict[str, Tuple[str, str, bool]] = {
    "shipped": ("-z.x", "-z.y", True),
    "times_minus_one": ("z.x * -1.0f", "z.y * -1.0f", True),
    "zero_minus": ("0.0f - z.x", "0.0f - z.y", False),
    "product_by_minus_one": ("(z.x * -1.0f) - (z.y * 0.0f)",
                             "(z.x * 0.0f) + (z.y * -1.0f)", False),
}


def _assert_diagnostic_arms_are_the_shipped_spelling() -> Dict[str, Any]:
    """The ``shipped`` arms of the tables must BE the shipped helpers' expressions.

    Not "look like". If a table's ``shipped`` row is a hand copy then the table
    certifies the copy. The real-coefficient helper names the coefficient ``c`` and
    the diagnostic shader names it ``k``, which is the ONLY licensed difference; it is
    normalised here rather than allowed to make the check pass vacuously. The
    negation is checked against the connection block the curl emits on a NEGATE axis.
    """
    helper = " ".join(templates.complex_helpers().split())
    checked: Dict[str, Any] = {}
    for table, rename in ((_FIELD_LEFT_ARMS, True), (_FULL_PRODUCT_ARMS, False)):
        re_text, im_text, _ = table["shipped"]
        for part, text in (("re", re_text), ("im", im_text)):
            wanted = " ".join(text.split())
            if rename:
                wanted = wanted.replace("* k", "* c")
            key = f"{'field_left' if rename else 'full_product'}_{part}"
            checked[key] = wanted
            assert wanted in helper, (
                f"the diagnostic table's shipped {key} expression {wanted!r} does "
                f"not occur in templates.complex_helpers(). This leg "
                f"is then measuring a HAND COPY of the shader, not the shader, and "
                f"an edit to the emitted helper would leave it green")
    block = cx._phase_block("x", cx.CONNECT_NEGATE)
    re_text, im_text, _ = _NEGATE_ARMS["shipped"]
    wanted = [line.split(" = ", 1)[1].strip().rstrip(";")
              for line in block.splitlines() if " = " in line]
    checked["negate"] = wanted
    assert wanted and all(value.startswith("-") and "c_mul" not in value
                          for value in wanted) and re_text.startswith("-") \
        and im_text.startswith("-"), (
        f"the connection block on a NEGATE axis spells {wanted!r}; the table's "
        f"shipped negation is a unary minus per part, so this leg would be measuring "
        f"a HAND COPY of the shader")
    return checked


def _run_table(source_template: str, re: str, im: str, operands: Sequence[Any],
               count: int) -> np.ndarray:
    import torch  # noqa: PLC0415

    source = (source_template.replace("__CONTRACT__", shaders.contraction_pragma())
              .replace("__RE__", re).replace("__IM__", im))
    function = metal_launch.compile_source(source).shipped
    tensors = [torch.from_numpy(np.ascontiguousarray(value)).to(torch.device("mps"))
               for value in operands]
    result = torch.empty_like(tensors[0])
    function(result, *tensors, count)
    torch.mps.synchronize()
    return result.cpu().numpy()


def _connect_oracle(z: Any, p: Any) -> Any:
    """``part_products.connect_parts`` per pattern, every phase on its product branch."""
    from meep_gpu import part_products  # noqa: PLC0415

    out = np.empty_like(z)
    for index in range(z.size):
        phase = (float(p[index].real), float(p[index].imag))
        assert not (abs(phase[1]) < part_products.CONNECT_TOLERANCE
                    and abs(abs(phase[0]) - 1.0) < part_products.CONNECT_TOLERANCE), (
            f"pattern {index} phase {phase} takes the copy/negate branch, not the "
            f"product this table measures")
        re, im = part_products.connect_parts(
            [z.real[index:index + 1].astype(np.float32),
             z.imag[index:index + 1].astype(np.float32)], phase)
        out[index:index + 1] = complex_from_planes(re, im)
    return out


def leg_part_products(payload: Dict[str, Any], out: str) -> None:
    """Exhaustive signed-zero tables: the shipped helpers against ``part_products``.

    THREE TABLES AND ONE RANDOM ROW. The real-coefficient scale over every
    ``(z_re, z_im, c)`` triple of {1.5, -2.25, +0.0, -0.0} (64 patterns), the Bloch
    connection product over every ``(z_re, z_im, p_re, p_im)`` quadruple (256), and
    the CONNECT_NEGATE unary minus over every ``(z_re, z_im)`` pair (16), each against
    the array path's own helper. The previous package spellings are this leg's
    inverted mutations: the zero-cross-term scales must DIFFER on the exhaustive
    table, and the fused connection must DIFFER on 4,096 random patterns, where its
    single rounding of ``a*b - c*d`` is not the unfused build's two.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    payload["legs"]["part_products_shipped_spelling"] = (
        _assert_diagnostic_arms_are_the_shipped_spelling())
    from meep_gpu import part_products  # noqa: PLC0415

    def record_row(table: str, label: str, re: str, im: str,
                   must_match: Optional[bool], bad: int, words: int) -> None:
        row = {"table": table, "arm": label, "re": re, "im": im,
               "must_match": must_match, "differing_words": bad, "words": words}
        if must_match is None:
            kit.predicted_null(row, (
                "every product on this table is EXACT, so the fused previous arm "
                "agrees here; its discrimination is the random table's"))
        rows.append(row)
        log(f"[part_products/{table}] {label:<34} differing={bad}/{words} "
            f"must_match={must_match} ({time.time() - started:.1f}s)")
        payload["legs"]["part_products"] = rows
        save(payload, out)
        if must_match is True:
            assert bad == 0, (
                f"{table}/{label} must reproduce the array path's helper exactly and "
                f"missed {bad} words: {row}")
        elif must_match is False:
            assert bad > 0, (
                f"{table}/{label} agreed with the array path on the whole table, so "
                f"the spelling it refutes is not load-bearing and this mutation "
                f"certifies nothing")

    patterns = list(itertools.product(_TABLE_VALUES, repeat=3))
    z = complex_from_planes(np.array([q[0] for q in patterns], dtype=np.float32),
                            np.array([q[1] for q in patterns], dtype=np.float32))
    cc = np.array([q[2] for q in patterns], dtype=np.float32)
    host = part_products.scale(cc, z)
    for label, (re, im, must_match) in _FIELD_LEFT_ARMS.items():
        got = _run_table(_DIAGNOSTIC_SHADERS, re, im, (z, cc), len(patterns))
        record_row("scale", label, re, im, must_match, differing(got, host),
                   2 * len(patterns))

    full = list(itertools.product(_TABLE_VALUES, repeat=4))
    fz = complex_from_planes(np.array([q[0] for q in full], dtype=np.float32),
                             np.array([q[1] for q in full], dtype=np.float32))
    fp = complex_from_planes(np.array([q[2] for q in full], dtype=np.float32),
                             np.array([q[3] for q in full], dtype=np.float32))
    host_full = _connect_oracle(fz, fp)
    for label, (re, im, must_match) in _FULL_PRODUCT_ARMS.items():
        got = _run_table(_FULL_PRODUCT_SHADER, re, im, (fz, fp), len(full))
        record_row("connect", label, re, im, must_match, differing(got, host_full),
                   2 * len(full))

    rng = np.random.default_rng(20261008)
    count = 4096
    rz = complex_from_planes(rng.standard_normal(count).astype(np.float32),
                             rng.standard_normal(count).astype(np.float32))
    angle = rng.uniform(0.05, 0.45, count) * 2 * np.pi
    rp = complex_from_planes(np.cos(angle).astype(np.float32),
                             np.sin(angle).astype(np.float32))
    host_random = _connect_oracle(rz, rp)
    for label, must_match in _RANDOM_PRODUCT_ARMS:
        re, im, _ = _FULL_PRODUCT_ARMS[label]
        got = _run_table(_FULL_PRODUCT_SHADER, re, im, (rz, rp), count)
        record_row("connect_random", label, re, im, must_match,
                   differing(got, host_random), 2 * count)

    pairs = list(itertools.product(_TABLE_VALUES, repeat=2))
    nz = complex_from_planes(np.array([q[0] for q in pairs], dtype=np.float32),
                             np.array([q[1] for q in pairs], dtype=np.float32))
    negated = part_products.connect(nz, (-1.0, 0.0))
    dummy = np.zeros(len(pairs), dtype=np.float32)
    for label, (re, im, must_match) in _NEGATE_ARMS.items():
        got = _run_table(_DIAGNOSTIC_SHADERS, re, im, (nz, dummy), len(pairs))
        record_row("negate", label, re, im, must_match, differing(got, negated),
                   2 * len(pairs))
    payload["counts"]["part_products"] = sum(row["words"] for row in rows)


# ---------------------------------------------------------------------------
# LEG bindings — the 31-binding ceiling, as an assertion
# ---------------------------------------------------------------------------

#: Needle pairs for the ``constant float2&`` scalar ABI. TRANCHE 1 CERTIFIED
#: ``constant float&`` AND NOTHING ELSE — five needles through a Python double into
#: a single float. This family binds a Python ``(re, im)`` TUPLE into a
#: ``constant float2&`` for each of three phases, which is a different ABI surface,
#: and the Brillouin edge makes its zero SIGN load-bearing: ``phase_arguments``
#: negates the imaginary part for the backward sub-step, so ``-1 + 0j`` forward is
#: ``-1 - 0j`` backward, and in ``fma(z.x, p.y, ...)`` with ``z.x`` an exact zero the
#: two round to different words.
_FLOAT2_NEEDLES: Tuple[Tuple[str, Tuple[float, float]], ...] = (
    ("brillouin_edge_forward", (-1.0, 0.0)),
    ("brillouin_edge_backward", (-1.0, -0.0)),
    ("generic_phase", (float(np.float32(np.cos(2 * np.pi * 0.3))),
                       float(np.float32(np.sin(2 * np.pi * 0.3))))),
    ("min_normal", (1.1754943508222875e-38, -1.1754943508222875e-38)),
    ("max_finite", (3.4028234663852886e38, -3.4028234663852886e38)),
)

_FLOAT2_ECHO = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void echo(device float2* o [[buffer(0)]],
                 constant float2& p [[buffer(1)]],
                 constant uint& n [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= n) { return; }
    // Lane 0 echoes the bound value. Lane 1 exposes the SIGN of a zero the way the
    // shipped rotation would: an fma whose product is an exact zero, so the
    // addend's sign is the only thing that decides the result's.
    o[i] = (i == 0) ? p : float2(fma(0.0f, p.x, p.y), fma(0.0f, p.y, p.x));
}
"""


def leg_bindings(payload: Dict[str, Any], out: str) -> None:
    """``float2`` is FORCED, not preferred — and the float2 SCALAR ABI is measured.

    Two questions, and the second had no measurement anywhere on this track. The
    first is the 31-buffer ceiling. The second is whether a Python ``(re, im)`` pair
    bound to a ``constant float2&`` arrives correctly rounded in BOTH lanes and
    carries the sign of a zero: the phase reaches the kernel that way, three times
    per curl launch, and tranche 1's clean-scalar-ABI measurement covered
    ``constant float&`` only. It was until now inferred from the reference legs
    agreeing — true, but an inference, and the one value it turns on (the backward
    Brillouin edge's ``-0.0``) is exactly the value an ABI is most likely to
    normalise silently.
    """
    import torch  # noqa: PLC0415

    started = time.time()
    echo = metal_launch.compile_source(_FLOAT2_ECHO).echo
    scalar_rows: List[Dict[str, Any]] = []
    for label, pair in _FLOAT2_NEEDLES:
        result = torch.zeros(2, dtype=torch.complex64, device=torch.device("mps"))
        echo(result, pair, 2)
        torch.mps.synchronize()
        got = kit.words(result.cpu().numpy())
        wanted = np.array([np.float32(pair[0]), np.float32(pair[1])]).view(np.uint32)
        row = {"needle": label, "bound": list(pair),
               "echoed_words": [int(got[0]), int(got[1])],
               "expected_words": [int(wanted[0]), int(wanted[1])],
               "zero_sign_probe_words": [int(got[2]), int(got[3])]}
        row["exact"] = row["echoed_words"] == row["expected_words"]
        scalar_rows.append(row)
        log(f"[bindings/float2] {label:<26} exact={row['exact']} "
            f"words={row['echoed_words']} ({time.time() - started:.1f}s)")
        assert row["exact"], (
            f"a Python double pair bound to `constant float2&` did not arrive "
            f"correctly rounded: {row}")
    payload["legs"]["float2_scalar_abi"] = scalar_rows
    save(payload, out)
    forward = scalar_rows[0]["zero_sign_probe_words"]
    backward = scalar_rows[1]["zero_sign_probe_words"]
    assert forward != backward, (
        f"the forward and backward Brillouin-edge phases are INDISTINGUISHABLE "
        f"through the float2 binding ({forward} vs {backward}), so the binding "
        f"normalised the sign of the imaginary zero and the conjugation this "
        f"family applies for step_D cannot reach the device")
    shipped = cx.bloch_curl_source((PERIODIC,) * 3, False, (1, 1, 1))
    torch.mps.compile_shader(shipped)
    split_error = None
    try:
        torch.mps.compile_shader(cx.split_plane_curl_signature())
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        split_error = str(exc).splitlines()[0][:200]
    row = {"shipped_bindings": cx.CURL_BINDINGS,
           "split_plane_bindings": cx.split_plane_binding_count(),
           "ceiling": device.MAX_BUFFER_BINDINGS,
           "shipped_compiles": True,
           "split_plane_error": split_error}
    payload["legs"]["bindings"] = row
    save(payload, out)
    log(f"[bindings] shipped={cx.CURL_BINDINGS} "
        f"split={row['split_plane_bindings']} "
        f"ceiling={row['ceiling']} split_refused={split_error is not None} "
        f"({time.time() - started:.1f}s)")
    assert split_error is not None, (
        "the re/im-SPLIT curl signature COMPILED. The float2 binding is then a "
        "preference rather than a forced choice, and the claim that the ceiling "
        "decides the design is false")
    assert "out of bounds" in split_error, split_error
    payload["counts"]["bindings"] = 2 + 2 * len(_FLOAT2_NEEDLES)


# ---------------------------------------------------------------------------
# LEG signed_zero
# ---------------------------------------------------------------------------

def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """Zero init for the curl, a +-0 LATTICE for the constitutive.

    The constitutive sub-step is a FIXED POINT under zero init — ``fw`` becomes the
    source and the accumulation adds and subtracts the same zeros — so it gets its
    own seeding with its own vacuity floor. A census of zero is VACUOUS, not passed.

    THE ZERO-INIT CURL HALF IS ALSO NEARLY A FIXED POINT, and that is exactly why it
    needs BOTH floors. Every value stays zero; what moves is the SIGN BIT, so
    ``moved_words`` and the ``-0.0`` census are the same measurement seen twice and
    each one alone is the vacuity guard the other cannot be.

    THE PREDICTED NULL THAT USED TO LIVE HERE WAS REFUTED BY THIS LEG'S OWN NUMBERS.
    The four periodic cases were recorded ``predicted_null`` with the reason "no
    metallic axis: the ownership mask writes no exact zeros through the recurrence,
    so no -0.0 output is reachable", and the same rows recorded 760 negative-zero
    words on ``step_B`` and 349 on ``step_D`` — identical to the metallic rows. The
    ownership mask is not the only source of a stored ``-0.0``: a real PML's deepest
    ``kms = kappa - sigma`` goes NEGATIVE, and ``(+0.0) * (negative)`` is ``-0.0`` on
    any axis, walled or not. So the floor is asserted on EVERY case now; there is no
    case here that is expected to measure nothing.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    for name, cell, boundaries, k_point in CONFIGS:
        grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
        codes = boundary_codes(grid, pml)
        phases = phase_table(grid, pml)
        dtdx = float(grid.dt / grid.dx)
        for volume in STATE:
            array = getattr(fields, volume, None)
            if array is not None:
                array[...] = np.complex64(0)
        zero_state = snapshot(fields)
        for sub_step, spec in SUB_STEPS.items():
            restore(fields, zero_state)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            got = run_curl_on_device(zero_state, pml, sub_step, codes, dtdx, phases,
                                     record=layer_record(grid, pml))
            names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            bad = {n: differing(got[n], after[n]) for n in names}
            census = sum(subnormal.signed_zero_census(after[n])["negative_zero"]
                         for n in names)
            moved = sum(differing(after[n], zero_state[n]) for n in names)
            compared += len(names)
            row = {"case": name, "sub_step": sub_step, "seeding": "zero_init",
                   "negative_zero_words": census, "moved_words": moved,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[signed_zero] {name:<17} {sub_step} zero_init neg_zero={census} "
                f"moved={moved} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            assert not row["differing"], row
            # Under zero init only the SIGN BIT moves, so this floor and the census
            # below are the same fact twice — and a leg that asserted neither would
            # be a no-op agreeing with a no-op.
            kit.assert_moved(moved, f"{name}/{sub_step} zero_init")
            kit.assert_census_floor(census, f"{name}/{sub_step} zero_init")

        for volume in STATE:
            array = getattr(fields, volume, None)
            if array is None:
                continue
            real = np.zeros(grid.shape, dtype=np.float32)
            imag = np.zeros(grid.shape, dtype=np.float32)
            real.reshape(-1)[::3] = np.float32(-0.0)
            real.reshape(-1)[1::5] = np.float32(0.25)
            imag.reshape(-1)[2::3] = np.float32(-0.0)
            imag.reshape(-1)[4::7] = np.float32(-0.125)
            array[...] = complex_from_planes(real, imag)
        seeded = snapshot(fields)
        inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
        for side, spec in SIDES.items():
            restore(fields, seeded)
            getattr(stepping, spec["step"])(fields, pml)
            after = snapshot(fields)
            got = run_constitutive_on_device(seeded, pml, side, inverse,
                                             record=layer_record(grid, pml))
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            bad = {n: differing(got[n], after[n]) for n in names}
            moved = sum(differing(after[n], seeded[n]) for n in names)
            census = sum(subnormal.signed_zero_census(after[n])["negative_zero"]
                         for n in names)
            compared += len(names)
            row = {"case": name, "side": side, "seeding": "signed_zero_lattice",
                   "moved_words": moved, "negative_zero_words": census,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[signed_zero] {name:<17} update_{side} lattice moved={moved} "
                f"neg_zero={census} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            assert not row["differing"], row
            kit.assert_moved(moved, f"{name}/update_{side} +-0 lattice")
            kit.assert_census_floor(census, f"{name}/update_{side}")
    payload["counts"]["signed_zero"] = compared


# ---------------------------------------------------------------------------
# LEG precondition — the subnormal window, and the control that makes it fire
# ---------------------------------------------------------------------------

SUBNORMAL_SCALES = (("physical", 1.0, True), ("small_normal", 1e-20, True),
                    ("subnormal_band", 1e-38, False))


def _precondition_case(name: str, cell, boundaries, k_point, label: str,
                       scale: float, sub_step: str) -> Tuple[Any, Dict[str, Any]]:
    """One (case, scale, sub-step) census, and the byte comparison beside it.

    Returns the window and the artifact row. Factored out so the physical band can
    be swept over EVERY case while the two controls stay on one.
    """
    grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815,
                              scale=scale, signed_zeros=False)
    before = snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    after = snapshot(fields)
    codes = boundary_codes(grid, pml)
    phases = phase_table(grid, pml)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
    record = layer_record(grid, pml)
    window = preconditions.SubnormalWindow(0, 1, per_array_words=64,
                                           per_intermediate_words=64)
    intermediates: Dict[str, Any] = {}
    if sub_step in SUB_STEPS:
        spec = SUB_STEPS[sub_step]
        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        sources = tuple(spec["sources"])
        got = run_curl_on_device(before, pml, sub_step, codes,
                                 float(grid.dt / grid.dx), phases, record=record)
        # THE INTERMEDIATES COME OUT OF THE PINNED REFERENCE ITSELF, not out of a
        # second transcription written for the census. `reference_curl` is what
        # `leg_transcription` compares against `stepping.py`, so every value
        # collected here is the array path's own.
        reference_curl([np.array(before[n], copy=True) for n in spec["targets"]],
                       [np.array(before["fu_" + n], copy=True)
                        for n in spec["targets"]],
                       [before[n] for n in sources],
                       curl_coefficients(pml, spec["suffix"]),
                       codes, bool(spec["backward"]),
                       float(grid.dt / grid.dx), phases, record,
                       collect=intermediates)
    else:
        side = "H" if sub_step == "update_H" else "E"
        spec = SIDES[side]
        names = tuple(spec["targets"]) + tuple(spec["aux"])
        sources = tuple(spec["sources"])
        got = run_constitutive_on_device(before, pml, side, inverse, record=record)
        reference_constitutive(
            [np.array(before[n], copy=True) for n in spec["targets"]],
            [np.array(before[n], copy=True) for n in spec["aux"]],
            [before[n] for n in sources],
            None if side == "H" else [inverse[n] for n in spec["targets"]],
            constitutive_coefficients(pml, spec["suffix"]), record,
            collect=intermediates)
    for volume in sources + names:
        window.observe(f"in:{volume}", before[volume], step=0)
    for volume in names:
        window.observe(f"out:{volume}", after[volume], step=1)
    for tag, volume in intermediates.items():
        window.observe_intermediate(tag, volume, step=1)
    report = window.report()
    bad = sum(differing(got[n], after[n]) for n in names)
    row = {"case": name, "scale": label, "factor": scale, "sub_step": sub_step,
           "subnormal_words": report["subnormal_words"],
           "observed_words": report["observed_words"],
           "intermediate_words": sum(v["words"]
                                     for v in report["intermediates"].values()),
           "intermediate_subnormal_words": sum(
               v["subnormal_words"] for v in report["intermediates"].values()),
           "intermediates_censused": sorted(report["intermediates"]),
           "clean": report["clean"], "vacuous": report["vacuous"],
           "differing_words": bad,
           "gate_verdict": ("REFUSED (precondition)" if not report["clean"]
                            else ("identical" if not bad else "DIFFERS"))}
    return window, row


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The census that BOUNDS the claim, through the shared window implementation.

    Two halves and the second is the one that matters: on the physical band the
    census over every OPERAND, every RESULT and the reconstructed INTERMEDIATES is
    zero and the bytes are identical; on the 1e-38 control the census FIRES and the
    gate REFUSES the case.

    ALL FOUR SUB-STEPS, AND THAT IS A CORRECTION. This leg used to run the two CURL
    sub-steps only, so the constitutive comparisons were certified with no census at
    all while the summary claimed the whole family's byte-identity was "under a
    CHECKED subnormal-free precondition". A precondition that does not cover half
    the claim is not the claim's precondition.

    EVERY CASE AT THE PHYSICAL BAND, AND THAT IS THE SAME CORRECTION AGAIN, ONE
    LEVEL UP. Until this round the census ran on ``periodic_kx`` ALONE: seven of the
    eight cases in the matrix, and every row of the synthetic sweep, were certified
    with no census while the summary said the claim was bounded by one. The census
    now sweeps the whole case matrix at scale 1.0 — 8 cases x 4 sub-steps — and
    ``leg_synthetic`` carries its own window over the odd shapes. The two SCALED
    CONTROLS stay on one case, because their job is to show the cliff exists rather
    than to bound the claim.

    INTERMEDIATES ARE RECONSTRUCTED ON THE HOST, AND THAT IS WEAKER THAN A REGISTER
    READ — stated, not hidden. ``compile_shader`` exposes no way to read a kernel's
    registers, so the census forms the SAME expressions in the SAME operand order on
    the host and censuses those.

    THE INTERMEDIATE SET USED TO BE ONE VALUE DEEP. It censused the curl and the two
    constitutive coefficient products and nothing else, while the docstring said
    "the reconstructed INTERMEDIATES". The split-field recurrence forms FIVE more
    per target that are never stored — ``fu*kms``, ``fu*kms - curl``, ``f*kms_u``,
    ``f*kms_u + fu``, ``f*kms_u + fu - fprev`` — plus the six rotated shift buffers,
    and the constitutive forms the partial sum ``f + kps*fw``. Every one of them can
    land in the band while its operands and its final result are outside it (a
    coefficient of 1e-4 against a stored 1e-35 is enough). They are collected from
    the pinned reference now, so the set is the arithmetic's, not a list someone
    remembered to keep up to date.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    # The whole matrix at the physical band...
    for name, cell, boundaries, k_point in CONFIGS:
        for sub_step in ("step_B", "step_D", "update_H", "update_E"):
            window, row = _precondition_case(name, cell, boundaries, k_point,
                                             "physical", 1.0, sub_step)
            rows.append(row)
            log(f"[precondition] {name:<17} {'physical':<14} {sub_step:<9} "
                f"subnormal={row['subnormal_words']}/{row['observed_words']} "
                f"(intermediates {row['intermediate_subnormal_words']}/"
                f"{row['intermediate_words']}) "
                f"differing={row['differing_words']} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)
            preconditions.assert_clean_or_refuse(window, f"{name}/{sub_step}")
            assert row["differing_words"] == 0, row

    # ...and the two SCALED classes on one case, whose job is the cliff.
    name, cell, boundaries, k_point = config("periodic_kx")  # the rotation is live
    for label, scale, expect_clean in SUBNORMAL_SCALES[1:]:
        for sub_step in ("step_B", "step_D", "update_H", "update_E"):
            window, row = _precondition_case(name, cell, boundaries, k_point,
                                             label, scale, sub_step)
            rows.append(row)
            log(f"[precondition] {name:<17} {label:<14} {sub_step:<9} "
                f"subnormal={row['subnormal_words']}/{row['observed_words']} "
                f"(intermediates {row['intermediate_subnormal_words']}/"
                f"{row['intermediate_words']}) "
                f"differing={row['differing_words']} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)
            if expect_clean:
                preconditions.assert_clean_or_refuse(window, f"{label}/{sub_step}")
                assert row["differing_words"] == 0, row
            else:
                preconditions.demonstrate_firing(window, f"{label}/{sub_step}")
                assert row["differing_words"] > 0, (
                    f"{label}: the control was expected to DIVERGE and did not, so "
                    f"the cliff this claim rests on was not reproduced")
    payload["legs"]["precondition_coverage"] = {
        "cases_censused_at_physical_band": sorted({r["case"] for r in rows
                                                   if r["scale"] == "physical"}),
        "cases_in_the_matrix": [entry[0] for entry in CONFIGS],
        "sub_steps_per_case": 4,
        "intermediate_labels": sorted({label for r in rows
                                       for label in r["intermediates_censused"]}),
    }
    save(payload, out)
    covered = set(payload["legs"]["precondition_coverage"][
        "cases_censused_at_physical_band"])
    assert covered == {entry[0] for entry in CONFIGS}, (
        f"the physical-band census covered {sorted(covered)} of "
        f"{[e[0] for e in CONFIGS]}; a case certified with no census is a case "
        f"whose claim has no precondition")
    payload["counts"]["precondition"] = len(rows)


# ---------------------------------------------------------------------------
# LEG mutations
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402

_CURL_FLAGS = _re.compile(r"    const bool du0 = (in_[hi])_z, du1 = in_[hi]_x, du2 = in_[hi]_y;\n")
_CONSTITUTIVE_FLAGS = _re.compile(
    r"    const bool dw0 = (in_[hi])_x, dw1 = in_[hi]_y, dw2 = in_[hi]_z;\n")


def _set_flags(source: str, pattern: Any, stem: str, values: Any) -> str:
    """Replace one membership-flags line. ``values`` is a triple, or a function of the
    line's lattice prefix (``in_h`` / ``in_i``) returning one. Raises ``LookupError``
    (NEEDLE-MISSED upstream) unless the line occurs exactly once."""
    found = pattern.findall(source)
    if len(found) != 1:
        raise LookupError(pattern.pattern)
    prefix = found[0]
    triple = values(prefix) if callable(values) else values
    line = (f"    const bool {stem}0 = {triple[0]}, {stem}1 = {triple[1]}, "
            f"{stem}2 = {triple[2]};\n")
    return pattern.sub(line, source, count=1)


_SCALE_BODY = "    return float2(z.x * c, z.y * c);\n"
_CONNECT_BODY = ("    return float2((z.x * p.x) - (z.y * p.y),\n"
                 "                  (z.x * p.y) + (z.y * p.x));\n")

#: ``must_catch``: True (must be caught) | None (record only) | False (must NOT be
#: caught -- a declared null, with its reason).
#:
#: THE GROUPINGS AND THE PRODUCTS ARE INVERTED (KERNELS.md §7.1). The shipped forms
#: are MEEP's now -- the left-to-right curl bracket, the right-hand side formed first,
#: and every product per part with the connection unfused -- so each grouping or
#: product mutation plants the PACKAGE'S PREVIOUS spelling and stays must-catch. The
#: ``b*`` rows are the per-chunk form battery (KERNELS.md §7.2).
CURL_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_previous_curl_grouping": {
        "must_catch": True,
        "why": "the package's previous pairwise grouping (g1s - g1) + (g2 - g2s) "
               "planted in place of MEEP's left-to-right ((g1s - g1) + g2) - g2s",
        "apply": lambda s: kit.needle(kit.needle(kit.needle(
            s, "((c_y - c_dy) + b_dz) - b_z", "(c_y - c_dy) + (b_dz - b_z)"),
            "((a_z - a_dz) + c_dx) - c_x", "(a_z - a_dz) + (c_dx - c_x)"),
            "((b_x - b_dx) + a_dy) - a_y", "(b_x - b_dx) + (a_dy - a_y)"),
    },
    "m2_contract_on": {
        "must_catch": True,
        "why": "the one compile option removed: the per-part products beside a "
               "+/- contract into fmas, which is not MEEP's two roundings",
        "apply": lambda s: kit.needle(s, shaders.contraction_pragma("off"),
                                      shaders.contraction_pragma("fast")),
    },
    "m3_recurrence_axis_swap": {
        "must_catch": True,
        "why": "breaks vec.hpp's dsig/dsigu cycle on target 0: (y,z) -> (z,y)",
        "apply": lambda s: kit.needle(
            kit.needle(
                s,
                "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);",
                "float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_z) - curl0, si_z);"),
            "float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) "
            "- p0, si_z) : n0;",
            "float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_y) + n0) "
            "- p0, si_y) : n0;"),
    },
    "m4_store_before_load": {
        "must_catch": True,
        "why": "the auxiliary's previous value read after its own store",
        "apply": lambda s: kit.needle(
            s, "    float2 p0 = du0 ? u0[ii] : f0[ii];",
            "    if (du0) { u0[ii] = float2(0.0f, 0.0f); }\n"
            "    float2 p0 = du0 ? u0[ii] : f0[ii];"),
    },
    "m5_phase_not_conjugated": {
        "must_catch": True,
        "why": "THE CLASSIC SIGN ERROR: the D side connected with the B side's "
               "conj(eikna) instead of eikna. Every magnitude stays plausible and "
               "only the phase moves. Planted in the HOST's phase encoding, where "
               "the conjugation lives",
        "host": True,
    },
    "m6_connection_on_the_wrong_plane": {
        "must_catch": True,
        "why": "the x connection applied to the top plane instead of plane 0: a "
               "whole plane of wrong values, no crash",
        "apply": lambda s: kit.needle(s, "    if (i == 0) {\n", "    if (i == nxi - 1) {\n"),
        "sub_steps": ("step_B",),
    },
    "m7_connection_on_every_lane": {
        "must_catch": True,
        "why": "the plane-0 select dropped on x, so every lane is connected",
        "apply": lambda s: kit.needle(s, "    if (i == 0) {\n", "    if (true) {\n"),
        "sub_steps": ("step_B",),
    },
    "m6y_connection_on_the_wrong_plane_y_axis": {
        "must_catch": True,
        "why": "the same defect on the y block",
        "apply": lambda s: kit.needle(s, "    if (j == 0) {\n", "    if (j == nyi - 1) {\n"),
        "sub_steps": ("step_B",),
    },
    "m7y_connection_on_every_lane_y_axis": {
        "must_catch": True,
        "why": "the plane-0 select dropped on the y block",
        "apply": lambda s: kit.needle(s, "    if (j == 0) {\n", "    if (true) {\n"),
        "sub_steps": ("step_B",),
    },
    "m14_connection_axes_crossed": {
        "must_catch": True,
        "why": "the y block connects its operands with the x axis's factor; only "
               "measurable with two axes carrying DIFFERENT phases, which "
               "periodic_kxyz does",
        "apply": lambda s: kit.needle(
            kit.needle(s, "c_dy = c_mul(c, py);", "c_dy = c_mul(c, px);"),
            "a_dy = c_mul(a, py);", "a_dy = c_mul(a, px);"),
        "sub_steps": ("step_B",),
    },
    "m8_drop_ownership_mask_one_axis": {
        "must_catch": True,
        "why": "one metallic axis stops dropping its unowned cell -- a plane of "
               "values MEEP's loop never touches",
        "apply": lambda s: kit.needle(s, "curl0 = at_x ? float2(0.0f, 0.0f) : curl0;",
                                      "// ownership mask dropped on x"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m9_ghost_periodic_for_metallic": {
        "must_catch": True,
        "why": "a metallic axis wrapped like a periodic one: the wall reads the far "
               "face instead of an exact complex zero",
        "apply": lambda s: kit.needle(s, "vx = (si >= 0) && (si < nxi);",
                                      "si = (si == nxi) ? 0 : si;"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m10_previous_fma_v1_real_scale": {
        "must_catch": True,
        "why": "the package's previous real-coefficient product -- NumPy's fused "
               "complex multiply against (c + 0j), zero cross terms included -- in "
               "place of one product per part. Byte-wrong on signed zeros and "
               "invisible on random data, so it runs on the +-0 LATTICE",
        "seeding": "lattice",
        "apply": lambda s: kit.needle(
            s, _SCALE_BODY,
            "    return float2(fma(z.x, c,    -(z.y * 0.0f)),\n"
            "                  fma(z.x, 0.0f,  (z.y * c)));\n"),
    },
    "m11_previous_naive_real_scale": {
        "must_catch": True,
        "why": "the same previous product unfused, (z.x*c) - (z.y*0), "
               "(z.x*0) + (z.y*c): its zero cross terms still flip zero signs; on "
               "the +-0 LATTICE",
        "seeding": "lattice",
        "apply": lambda s: kit.needle(
            s, _SCALE_BODY,
            "    return float2((z.x * c)    - (z.y * 0.0f),\n"
            "                  (z.x * 0.0f) + (z.y * c));\n"),
    },
    "m12_previous_fma_v1_connection": {
        "must_catch": True,
        "why": "the package's previous connection product, fused as NumPy's "
               "complex64 multiply: one rounding of z.x*p.x - z.y*p.y where MEEP's "
               "libc++ operator* rounds each product and the sum. Caught on random "
               "data on the phased planes",
        "apply": lambda s: kit.needle(
            s, _CONNECT_BODY,
            "    return float2(fma(z.x, p.x, -(z.y * p.y)),\n"
            "                  fma(z.x, p.y,  (z.y * p.x)));\n"),
    },
    "m13_connection_operands_swapped": {
        "must_catch": False,
        "why": "DECLARED NULL: the connection spelled phase * word instead of word "
               "* phase. With every product and every sum rounded on its own, IEEE "
               "multiplication and two-operand addition are commutative, so the "
               "orientation moves no word (it did under the fused arm, which is "
               "gone)",
        "apply": lambda s: kit.needle(s, "c_mul(b, px)", "c_mul(px, b)"),
        "sub_steps": ("step_B",),
    },
    "m15_neighbour_index_composition_swapped": {
        "must_catch": True,
        "why": "the shifted-neighbour offset composed as si*nyz + k*nzi + j; in "
               "bounds on this (12, 10, 9) grid, so it reads a real cell and returns "
               "a smooth wrong field",
        "apply": lambda s: kit.needle(
            s, "    int ox = si * nyz + j * nzi + k;",
            "    int ox = si * nyz + k * nzi + j;"),
        "sub_steps": ("step_B",),
    },
    "b1_select_removed": {
        "must_catch": True,
        "why": "the whole-grid recurrence (the previous package form): every cell "
               "steps the dsigu form and fu is written everywhere",
        "apply": lambda s: _set_flags(s, _CURL_FLAGS, "du", ("true", "true", "true")),
    },
    "b2_predicate_on_coefficient_identity": {
        "must_catch": True,
        "why": "the form keyed on kms != 1 instead of chunk membership: a sigma = 0 "
               "row inside a layer chunk takes the other form",
        "apply": lambda s: _set_flags(s, _CURL_FLAGS, "du",
                                      ("(kmz[k] != 1.0f)", "(kmx[i] != 1.0f)",
                                       "(kmy[j] != 1.0f)")),
    },
    "b5_dsig_dsigu_swapped": {
        "must_catch": True,
        "why": "each target keyed on its dsig axis's chunk instead of its dsigu "
               "axis's: dsig-only cells and dsigu-only cells trade forms",
        "apply": lambda s: _set_flags(s, _CURL_FLAGS, "du",
                                      lambda p: (f"{p}_y", f"{p}_z", f"{p}_x")),
    },
    "b6_auxiliary_stored_on_a_plain_cell": {
        "must_catch": True,
        "why": "fu written outside the recurrence's cells; the array path leaves "
               "those words alone and the gate compares them",
        "apply": lambda s: kit.needle(s, "    if (du0) { u0[ii] = n0; }\n",
                                      "    u0[ii] = n0;\n"),
    },
    "b7_auxiliary_load_unconditional": {
        "must_catch": False,
        "why": "DECLARED NULL: an fu load outside the guard with the store still "
               "guarded changes no word (the dsig-only form reads f); the guard is "
               "traffic, pinned by test_metal_complex_fields' static needle instead",
        "apply": lambda s: kit.needle(
            s, "    float2 p0 = du0 ? u0[ii] : f0[ii];",
            "    float2 p0u = u0[ii]; float2 p0 = du0 ? p0u : f0[ii];"),
    },
    "b8_plain_curl_through_the_recurrence": {
        "must_catch": True,
        "why": "the dsig-only form spelled as the recurrence with p := 0: differs "
               "on the dsig-only cells whose dsig coefficients are not 1 (and, with "
               "unit coefficients, only on the signed zero leg planted_zero plants)",
        "apply": lambda s: kit.needle(
            kit.needle(s, "    float2 p0 = du0 ? u0[ii] : f0[ii];",
                       "    float2 p0 = du0 ? u0[ii] : float2(0.0f, 0.0f);"),
            "float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) "
            "- p0, si_z) : n0;",
            "float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, "
            "si_z);"),
    },
}

_ACCUMULATION_0 = ("if (dw0) { a0 = f0[ii] + (c_mul_coefficient_left(kp_0, src0) - "
                   "c_mul_coefficient_left(km_0, prev0)); }")

CONSTITUTIVE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "c1_previous_accumulation_grouping": {
        "must_catch": True,
        "why": "the package's previous grouping ((f + kps*src) - kms*prev) planted in "
               "place of MEEP's f + ((kps*src) - (kms*prev))",
        "apply": lambda s: kit.needle(
            s, _ACCUMULATION_0,
            "if (dw0) { a0 = (f0[ii] + c_mul_coefficient_left(kp_0, src0)) - "
            "c_mul_coefficient_left(km_0, prev0); }"),
    },
    "c2_store_before_load": {
        "must_catch": True,
        "why": "`prev` read after `fw` is written -- wrong only where kms != 0, i.e. "
               "INSIDE THE PML ONLY, so it looks like a worse absorber",
        "apply": lambda s: kit.needle(
            s, "    float2 prev0 = dw0 ? w0[ii] : float2(0.0f, 0.0f);\n    float2 src0 =",
            "    if (dw0) { w0[ii] = float2(0.0f, 0.0f); }\n"
            "    float2 prev0 = dw0 ? w0[ii] : float2(0.0f, 0.0f);\n    float2 src0 ="),
    },
    "c3_dsigw_to_dsig_cycle": {
        "must_catch": True,
        "why": "the coefficient index moved off the component's OWN axis onto the "
               "curl's cycle -- MEEP's dsigw is not dsig",
        "apply": lambda s: kit.needle(s, "float kp_0 = kp0[i], km_0 = km0[i];",
                                      "float kp_0 = kp0[j], km_0 = km0[j];"),
    },
    "c4_inv_eps_orientation_is_equivalent": {
        "must_catch": False,
        "seeding": "lattice",
        "why": "DECLARED NULL: `inv_eps * D` instead of `D * inv_eps`, both one "
               "product per part; IEEE multiplication is commutative on every "
               "non-NaN word, signed zero included, so the orientation moves no word "
               "even on the +-0 lattice",
        "apply": lambda s: kit.needle(
            s, "c_mul_field_left(g0[ii], e0[ii])",
            "c_mul_coefficient_left(e0[ii], g0[ii])"),
        "sides": ("E",),
    },
    "b9_constitutive_select_removed": {
        "must_catch": True,
        "why": "the whole-grid accumulation (the previous package form): every cell "
               "accumulates and f_w is written everywhere",
        "apply": lambda s: _set_flags(s, _CONSTITUTIVE_FLAGS, "dw",
                                      ("true", "true", "true")),
    },
    "b10_constitutive_predicate_on_coefficient_identity": {
        "must_catch": True,
        "why": "the form keyed on kps != 1 instead of chunk membership",
        "apply": lambda s: _set_flags(s, _CONSTITUTIVE_FLAGS, "dw",
                                      ("(kp0[i] != 1.0f)", "(kp1[j] != 1.0f)",
                                       "(kp2[k] != 1.0f)")),
    },
    "b11_f_w_stored_on_a_plain_cell": {
        "must_catch": True,
        "why": "f_w written outside the accumulation's cells",
        "apply": lambda s: kit.needle(s, "if (dw0) { w0[ii] = src0; }",
                                      "w0[ii] = src0;"),
    },
    "b12_constitutive_load_unconditional": {
        "must_catch": False,
        "why": "DECLARED NULL: prev is read only on the accumulation's cells, so an "
               "unguarded load changes no word; pinned statically in "
               "test_metal_complex_fields instead",
        "apply": lambda s: kit.needle(
            s, "float2 prev0 = dw0 ? w0[ii] : float2(0.0f, 0.0f);",
            "float2 prev0 = w0[ii];"),
    },
    "b13_plain_constitutive_through_the_accumulation": {
        "must_catch": True,
        "why": "f = src spelled as the accumulation with kps = 1, kms = 0 on zeroed "
               "inputs: wrong on every -0 source part (the seeded state carries a "
               "+-0 lattice in both parts; leg planted_zero plants one)",
        "apply": lambda s: kit.needle(
            s, "    float2 a0 = src0;",
            "    float2 a0 = float2(0.0f, 0.0f) + (c_mul_coefficient_left(1.0f, src0) "
            "- c_mul_coefficient_left(0.0f, prev0));"),
    },
}


def _bound_off_by_one(record: Any) -> Any:
    """Every non-empty low run one row longer: the boundary plane changes form."""
    out = np.array(record, copy=True)
    for half in (False, True):
        for axis in range(3):
            a0, a1, b0 = (int(out[record_index(half, axis, w)]) for w in range(3))
            if a1 > a0 and (b0 == 0 or a1 + 1 < b0):
                out[record_index(half, axis, 1)] = a1 + 1
    return out


def _lattices_swapped(record: Any) -> Any:
    """The integer and half-integer entries exchanged."""
    out = np.array(record, copy=True)
    for axis in range(3):
        for word in range(4):
            i, h = record_index(False, axis, word), record_index(True, axis, word)
            out[i], out[h] = record[h], record[i]
    return out


#: The record mutations (KERNELS.md §7.2): the kernel is right and the record it is
#: handed is not.
RECORD_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "b3_bound_off_by_one": (_bound_off_by_one, "a run end one row past the chunk"),
    "b4_lattices_swapped": (_lattices_swapped,
                            "the integer and half-integer records swapped"),
}


def _assert_mutation_configuration(label: str, entry: Dict[str, Any],
                                   active_codes, active_phases,
                                   reference_codes, reference_phases) -> None:
    """The mutation's configuration must be the REFERENCE's, exactly.

    THIS GUARD EXISTS BECAUSE THREE MUST-CATCH MUTATIONS COULD NOT FAIL. m6, m7 and
    m13 once overrode ``phases`` while being compared against an oracle ``stepping``
    had produced on the case's OWN table: the unmutated shipped kernel launched under
    that override already differed from the oracle, so every ``CAUGHT 1/1`` was
    reporting the phase-table mismatch and not the planted defect. Any override must
    be a no-op against the reference, and that is checked here rather than left to a
    reader.
    """
    assert tuple(active_codes) == tuple(reference_codes), (
        f"{label}: boundary codes {tuple(active_codes)} differ from the reference "
        f"configuration's {tuple(reference_codes)}. The oracle was produced under "
        f"the reference's codes, so this mutation would 'catch' the mismatch rather "
        f"than the defect it names")
    assert tuple(active_phases) == tuple(reference_phases), (
        f"{label}: phase table {tuple(active_phases)} differs from the reference "
        f"configuration's {tuple(reference_phases)}. That is the confound this guard "
        f"was added for -- a mutation compared against an oracle built on a different "
        f"phase table cannot fail, whatever it plants")


def _lattice_state(fields: Any, shape) -> None:
    """The +-0 lattice seeding: zeros of both signs and two small values in each part."""
    for volume in STATE:
        array = getattr(fields, volume, None)
        if array is None:
            continue
        real = np.zeros(shape, dtype=np.float32)
        imag = np.zeros(shape, dtype=np.float32)
        real.reshape(-1)[::3] = np.float32(-0.0)
        real.reshape(-1)[1::5] = np.float32(0.25)
        imag.reshape(-1)[2::3] = np.float32(-0.0)
        imag.reshape(-1)[4::7] = np.float32(-0.125)
        array[...] = complex_from_planes(real, imag)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)
    # ALL THREE AXES PHASED, WITH THREE DIFFERENT FACTORS, every connection a
    # product (CONNECT_PHASE): a crossed-axis defect (m14) and the fused connection
    # (m12) are only measurable there.
    name, cell, boundaries, k_point = config("periodic_kxyz")
    grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
    codes = boundary_codes(grid, pml)
    phases = phase_table(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    record = layer_record(grid, pml)
    base = snapshot(fields)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}

    # THE +-0 LATTICE SEEDING, for the mutations whose defect class a physical-band
    # state does not reach (the previous products' zero cross terms).
    lattice_fields = build(cell, boundaries, k_point, 0.35, 20260815)[1]
    _lattice_state(lattice_fields, grid.shape)
    lattice_base = snapshot(lattice_fields)

    walled = build((1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
                   (0.0, 0.0, 0.0), 0.35, 20260815)
    walled_grid, walled_fields, walled_pml = walled
    walled_base = snapshot(walled_fields)
    walled_codes = boundary_codes(walled_grid, walled_pml)
    walled_phases = phase_table(walled_grid, walled_pml)
    walled_dtdx = float(walled_grid.dt / walled_grid.dx)
    walled_record = layer_record(walled_grid, walled_pml)

    def oracle_for(fields_obj, active_pml, state, step_name):
        restore(fields_obj, state)
        getattr(stepping, step_name)(fields_obj, active_pml)
        return snapshot(fields_obj)

    every = ("step_B", "step_D", "update_H", "update_E")
    oracle = {s: oracle_for(fields, pml, base, s) for s in every}
    lattice_oracle = {s: oracle_for(lattice_fields, pml, lattice_base, s) for s in every}
    walled_oracle = {s: oracle_for(walled_fields, walled_pml, walled_base, s)
                     for s in ("step_B", "step_D")}

    for label, entry in CURL_MUTATIONS.items():
        counters: List[kit.Counter] = []
        caught = ran = 0
        missed = False
        use_walled = "codes" in entry and METALLIC in entry["codes"]
        use_lattice = entry.get("seeding") == "lattice"
        state = (walled_base if use_walled
                 else lattice_base if use_lattice else base)
        active_pml = walled_pml if use_walled else pml
        active_codes = entry.get("codes", walled_codes if use_walled else codes)
        active_phases = entry.get("phases",
                                  walled_phases if use_walled else phases)
        active_dtdx = walled_dtdx if use_walled else dtdx
        active_record = walled_record if use_walled else record
        reference = (walled_oracle if use_walled
                     else lattice_oracle if use_lattice else oracle)
        _assert_mutation_configuration(
            label, entry, active_codes, active_phases,
            walled_codes if use_walled else codes,
            walled_phases if use_walled else phases)

        for sub_step in entry.get("sub_steps", tuple(SUB_STEPS)):
            spec = SUB_STEPS[sub_step]
            flags, _ = cx.phase_arguments(tuple(active_phases),
                                          backward=bool(spec["backward"]))
            if entry.get("host"):
                # m5: the CONJUGATION lives in the host encoder, so the defect is
                # planted there: step_D handed the B side's encoding.
                if sub_step != "step_D":
                    continue
                wrong_flags, wrong_values = cx.phase_arguments(
                    tuple(active_phases), backward=False)
                shipped = cx.bloch_curl_source(active_codes, spec["backward"],
                                               wrong_flags)
                function = kit.Counter(
                    metal_launch.compile_source(shipped).bloch_pml_curl_step)
                counters.append(function)
                arrays = {n: np.array(state[n], copy=True)
                          for n in tuple(spec["targets"]) + tuple(spec["sources"])}
                arrays.update({"fu_" + n: np.array(state["fu_" + n], copy=True)
                               for n in spec["targets"]})
                flat = {key: np.asarray(value).reshape(-1) for key, value in
                        curl_coefficients(active_pml, spec["suffix"]).items()}
                residency = metal_launch.Residency()
                plan = cx.plan_complex_pml_curl_from_arrays(
                    sub_step, arrays, flat, active_codes, tuple(active_phases),
                    active_dtdx, residency,
                    functions={shaders.CONTRACT_OFF: function},
                    chunk_bounds=active_record)
                # The plan encoded the D side's factor, correctly; the three phase
                # pairs sit just before the record tensor. Replace them with the B
                # side's: that, and nothing else, is the defect.
                plan._args = plan._args[:-4] + tuple(wrong_values) + plan._args[-1:]
                plan.run()
                residency.sync_out()
                got = arrays
            else:
                shipped = cx.bloch_curl_source(active_codes, spec["backward"], flags)
                try:
                    shipped = entry["apply"](shipped)
                except LookupError:
                    missed = True
                    continue
                got = run_curl_on_device(state, active_pml, sub_step, active_codes,
                                         active_dtdx, active_phases, source=shipped,
                                         counter=counters, record=active_record)
            names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            ran += 1
            caught += int(any(differing(got[n], reference[sub_step][n])
                              for n in names))
        launches = sum(counter.launches for counter in counters)
        harness.record(label, kit.MutationHarness.verdict(missed, ran, launches,
                                                          caught),
                       launches, caught, ran, entry["must_catch"], entry["why"])

    for label, entry in CONSTITUTIVE_MUTATIONS.items():
        counters = []
        caught = ran = 0
        missed = False
        use_lattice = entry.get("seeding") == "lattice"
        state = lattice_base if use_lattice else base
        reference = lattice_oracle if use_lattice else oracle
        for side in entry.get("sides", tuple(SIDES)):
            shipped = cx.bloch_constitutive_source(side)
            try:
                shipped = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            got = run_constitutive_on_device(state, pml, side, inverse,
                                             source=shipped, counter=counters,
                                             record=record)
            spec = SIDES[side]
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            ran += 1
            caught += int(any(differing(got[n], reference["update_" + side][n])
                              for n in names))
        launches = sum(counter.launches for counter in counters)
        harness.record(label, kit.MutationHarness.verdict(missed, ran, launches,
                                                          caught),
                       launches, caught, ran, entry["must_catch"], entry["why"])

    # --- HOST mutations: the plan's own choices, not the kernel's source ---------
    counters = []
    caught = ran = 0
    for sub_step, spec in SUB_STEPS.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        flags, _ = cx.phase_arguments(tuple(phases),
                                      backward=bool(spec["backward"]))
        shipped = cx.bloch_curl_source(codes, spec["backward"], flags)
        got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                 suffix=wrong_suffix, source=shipped,
                                 counter=counters, record=record)
        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        ran += 1
        caught += int(any(differing(got[n], oracle[sub_step][n]) for n in names))
    launches = sum(counter.launches for counter in counters)
    harness.record("h1_curl_half_integer_suffix_swap",
                   kit.MutationHarness.verdict(False, ran, launches, caught),
                   launches, caught, ran, True,
                   "the curl's PML coefficients taken from the other Yee sub-lattice")

    counters = []
    caught = ran = 0
    for side, spec in SIDES.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        shipped = cx.bloch_constitutive_source(side)
        got = run_constitutive_on_device(base, pml, side, inverse,
                                         suffix=wrong_suffix, source=shipped,
                                         counter=counters, record=record)
        names = tuple(spec["targets"]) + tuple(spec["aux"])
        ran += 1
        caught += int(any(differing(got[n], oracle["update_" + side][n])
                          for n in names))
    launches = sum(counter.launches for counter in counters)
    harness.record("h2_constitutive_suffix_swap",
                   kit.MutationHarness.verdict(False, ran, launches, caught),
                   launches, caught, ran, True,
                   "the constitutive coefficients taken from the other Yee sub-lattice")

    spec = SUB_STEPS["step_D"]
    counters = []
    flags, _ = cx.phase_arguments(tuple(phases), backward=True)
    forward_source = cx.bloch_curl_source(codes, False, flags)
    got = run_curl_on_device(base, pml, "step_D", codes, dtdx, phases,
                             source=forward_source, counter=counters, record=record)
    names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    caught = int(any(differing(got[n], oracle["step_D"][n]) for n in names))
    harness.record("h3_d_stencil_direction_flipped",
                   kit.MutationHarness.verdict(False, 1,
                                               sum(c.launches for c in counters),
                                               caught),
                   sum(c.launches for c in counters), caught, 1, True,
                   "step_D stepped with the forward stencil")

    # b3/b4: THE RECORD IS WRONG, the kernel is the shipped one. Both families,
    # against the oracle stepped on the true division.
    for label, (mutate, why) in RECORD_MUTATIONS.items():
        wrong = mutate(record)
        assert not np.array_equal(wrong, record), f"{label} changed no record word"
        counters = []
        caught = ran = 0
        for sub_step, spec in SUB_STEPS.items():
            flags, _ = cx.phase_arguments(tuple(phases),
                                          backward=bool(spec["backward"]))
            shipped = cx.bloch_curl_source(codes, spec["backward"], flags)
            got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                     source=shipped, counter=counters, record=wrong)
            names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            ran += 1
            caught += int(any(differing(got[n], oracle[sub_step][n]) for n in names))
        for side, spec in SIDES.items():
            shipped = cx.bloch_constitutive_source(side)
            got = run_constitutive_on_device(base, pml, side, inverse, source=shipped,
                                             counter=counters, record=wrong)
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            ran += 1
            caught += int(any(differing(got[n], oracle["update_" + side][n])
                              for n in names))
        launches = sum(counter.launches for counter in counters)
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches, caught),
                       launches, caught, ran, True, why)
    payload["counts"]["mutations"] = len(harness.rows)


# ---------------------------------------------------------------------------
# LEG planted_zero -- the two signed-zero forms only a planted -0 reaches
# ---------------------------------------------------------------------------

NEGATIVE_ZERO_WORD = 0x80000000


def _plain_cell(grid: Any, record: Any) -> Tuple[int, int, int]:
    """A cell outside every layer chunk on both lattices along every axis (so every
    form there is the plain one at unit coefficients), two rows clear of every face
    (so no wrap, wall or connection reaches its stencil). Refused by name when the
    case has none."""
    shape = tuple(int(n) for n in grid.shape)
    lines = [[membership_from_record(np.asarray(record), half, axis, shape[axis])
              for half in (False, True)] for axis in range(3)]
    for i in range(2, shape[0] - 2):
        for j in range(2, shape[1] - 2):
            for k in range(2, shape[2] - 2):
                cell = (i, j, k)
                if not any(lines[axis][h][cell[axis]]
                           for axis in range(3) for h in (0, 1)):
                    return cell
    raise AssertionError(f"no cell of {shape} lies outside every layer chunk; the "
                         f"planted -0 leg needs one")


def leg_planted_zero(payload: Dict[str, Any], out: str) -> None:
    """KERNELS.md §7.2's two rows a random state never draws, planted.

    * CURL: target 0 at a plain cell with ``f = -0`` in both parts and the four
      stencil operands equal, so ``((g1s - g1) + g2) - g2s`` is ``+0``: MEEP's
      ``f - curl`` (the dsig-only form at unit coefficients) stores ``-0``; b8 (the
      recurrence with ``p := 0``) stores ``+0``.
    * CONSTITUTIVE: component 0 at a plain cell with a ``-0`` source in both parts:
      the output select stores ``-0``; b13 (the accumulation with ``kps = 1, kms =
      0`` on zeroed inputs) stores ``+0``.

    The shipped kernels must equal ``stepping`` everywhere and store the -0 word at
    the planted cell; each mutant must turn it into +0.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries, k_point = config("periodic_kxz")
    grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
    codes = boundary_codes(grid, pml)
    phases = phase_table(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    record = layer_record(grid, pml)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
    centre = _plain_cell(grid, record)
    i, j, k = centre
    half = complex_from_planes(np.full(1, 0.5, np.float32),
                               np.full(1, 0.5, np.float32))[0]
    negative_zero = complex_from_planes(np.full(1, -0.0, np.float32),
                                        np.full(1, -0.0, np.float32))[0]
    base = snapshot(fields)

    plants = {
        # Bx at the cell; partners Ez along +y and Ey along +z.
        "step_B": (("Bx", centre), [("Ez", centre), ("Ez", (i, j + 1, k)),
                                    ("Ey", centre), ("Ey", (i, j, k + 1))]),
        # Dx at the cell; partners Hz along -y and Hy along -z.
        "step_D": (("Dx", centre), [("Hz", centre), ("Hz", (i, j - 1, k)),
                                    ("Hy", centre), ("Hy", (i, j, k - 1))]),
        "update_H": (("Bx", centre), []),
        "update_E": (("Dx", centre), []),
    }
    for step_name, ((target, at), equal) in plants.items():
        state = {n: np.array(v, copy=True) for n, v in base.items()}
        state[target][at] = negative_zero
        for volume, where in equal:
            state[volume][where] = half
        restore(fields, state)
        getattr(stepping, step_name)(fields, pml)
        after = snapshot(fields)
        if step_name in SUB_STEPS:
            spec = SUB_STEPS[step_name]
            stored = target
            names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            flags, _ = cx.phase_arguments(tuple(phases),
                                          backward=bool(spec["backward"]))
            shipped = cx.bloch_curl_source(codes, spec["backward"], flags)
            mutant_label = "b8_plain_curl_through_the_recurrence"
            mutant = CURL_MUTATIONS[mutant_label]["apply"](shipped)

            def launch(source: str, _s=step_name) -> Dict[str, Any]:
                return run_curl_on_device(state, pml, _s, codes, dtdx, phases,
                                          source=source, record=record)
        else:
            side = "H" if step_name == "update_H" else "E"
            spec = SIDES[side]
            stored = spec["targets"][0]
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            shipped = cx.bloch_constitutive_source(side)
            mutant_label = "b13_plain_constitutive_through_the_accumulation"
            mutant = CONSTITUTIVE_MUTATIONS[mutant_label]["apply"](shipped)

            def launch(source: str, _side=side) -> Dict[str, Any]:
                return run_constitutive_on_device(state, pml, _side, inverse,
                                                  source=source, record=record)

        got = launch(shipped)
        wrong = launch(mutant)
        oracle_words = [int(w) for w in kit.words(after[stored][centre])]
        shipped_words = [int(w) for w in kit.words(got[stored][centre])]
        mutant_words = [int(w) for w in kit.words(wrong[stored][centre])]
        bad = sum(differing(got[n], after[n]) for n in names)
        row = {"sub_step": step_name, "cell": list(centre), "stored": stored,
               "oracle_words": oracle_words, "shipped_words": shipped_words,
               "mutant": mutant_label, "mutant_words": mutant_words,
               "shipped_differing_words": bad}
        rows.append(row)
        log(f"[planted_zero] {step_name} {stored}{centre} oracle={oracle_words} "
            f"shipped={shipped_words} {mutant_label}={mutant_words} "
            f"differing={bad} ({time.time() - started:.1f}s)")
        payload["legs"]["planted_zero"] = rows
        save(payload, out)
        assert oracle_words == [NEGATIVE_ZERO_WORD, NEGATIVE_ZERO_WORD], (
            f"{step_name}: the array path did not store -0 at the planted cell, so the "
            f"plant did not reach the form it targets: {row}")
        assert bad == 0 and shipped_words == oracle_words, row
        assert mutant_words != oracle_words, (
            f"{step_name}: {mutant_label} stored the planted -0 too, so the plain form "
            f"spelled through the other form is not caught here: {row}")
    payload["counts"]["planted_zero"] = len(rows)


# ---------------------------------------------------------------------------
# LEG reduction — the k = 0 identity, and the disjointness sweep
# ---------------------------------------------------------------------------

def leg_reduction(payload: Dict[str, Any], out: str) -> None:
    """An UNPHASED build must be byte-identical to the plain complex path.

    MEEP's ``connect_phase`` COPIES within 1e-13 of +1 (CONNECT_COPY) instead of
    multiplying by ``1+0j``, and the array path does the same
    (``part_products.connect_parts``). Measured here by building the same
    configuration twice -- once with every axis's connection kind NONE and once with
    the kind forced to PHASE -- and requiring the FIRST to match ``stepping.py``
    and to emit one "no connection" line per axis, and the SECOND to be a genuinely
    different source string carrying the connection products.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries, k_point = config("periodic_k0")
    grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
    base = snapshot(fields)
    codes = boundary_codes(grid, pml)
    phases = phase_table(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    assert all(p is None for p in phases), phases
    record = layer_record(grid, pml)
    forced = (cx.CONNECT_PHASE,) * 3
    for sub_step, spec in SUB_STEPS.items():
        restore(fields, base)
        getattr(stepping, sub_step)(fields, pml)
        after = snapshot(fields)
        got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                 record=record)
        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        bad = sum(differing(got[n], after[n]) for n in names)
        unphased = cx.bloch_curl_source(codes, spec["backward"], (0, 0, 0))
        phased = cx.bloch_curl_source(codes, spec["backward"], forced)
        row = {"case": name, "sub_step": sub_step, "differing_words": bad,
               "unphased_source_bytes": len(unphased),
               "phased_source_bytes": len(phased),
               "sources_differ": unphased != phased,
               "copies_emitted": unphased.count(": no connection ("),
               "unphased_products": unphased.count("= c_mul("),
               "phased_products": phased.count("= c_mul(")}
        rows.append(row)
        log(f"[reduction] {sub_step} differing={bad} copies={row['copies_emitted']} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["reduction"] = rows
        save(payload, out)
        assert bad == 0, row
        assert row["copies_emitted"] == 3 and row["unphased_products"] == 0, (
            "an unphased build must emit one 'no connection' line per axis and no "
            f"connection product at all: {row}")
        assert row["sources_differ"] and row["phased_products"] == 6, (
            "the forced-PHASE build must carry two connection products per axis "
            f"and differ from the unphased one: {row}")
    payload["counts"]["reduction"] = len(rows)


#: (label, kwargs for `build`, why the matrix does not already contain it).
#:
#: THE PREDICATE ADMITS THESE AND NO LEG LAUNCHED THEM. Found by constructing real
#: engine objects and CALLING ``complex_pml_curl_coverage`` /
#: ``complex_constitutive_coverage`` over an enumeration of everything this family
#: does not implement: the two rows below came back ADMITTED on all four slots while
#: every case in :data:`CONFIGS` is fully three-dimensional with an absorber on all
#: three axes. An admitted configuration that no leg runs is domain the family
#: claims and has not measured — the same shape of hole the y-rotation block was,
#: and the reason that one survived a whole certification.
ADMITTED_DOMAIN: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("two_dimensional_collapsed_z",
     {"cell": (1.2, 1.0, 0.0), "boundaries": "periodic", "k_point": (0.3, 0.0, 0.0),
      "dimensions": 2,
      "thickness": {"x": (2, 2), "y": (2, 2), "z": (0, 0)}},
     "grid.shape is (12, 10, 1): the periodic wrap on z lands every lane on the "
     "wrap lane, the phase block's `k == nzi - 1` predicate is `k == 0`, and the "
     "z coefficient vectors are length 1. This is the shape a real 2-D run has and "
     "the matrix had no case with any collapsed axis at all"),
    ("absorber_absent_on_y",
     {"cell": (1.2, 1.0, 0.9), "boundaries": "periodic", "k_point": (0.3, 0.0, 0.0),
      "thickness": {"x": (2, 2), "y": (0, 0), "z": (2, 2)}},
     "an active PML with one axis carrying no absorber at all — kms_y/sinv_y are "
     "the identity profile everywhere, which is the boundary between this product "
     "and the no-PML one and was never launched"),
)


def leg_admitted_domain(payload: Dict[str, Any], out: str) -> None:
    """Every configuration the predicate ADMITS gets launched somewhere.

    This leg exists because the predicate and the case matrix were written against
    each other and drifted: the clause list admits a 2-D run and a partially
    absorbed one, and the matrix contains neither. Each row here builds REAL engine
    objects, asserts the predicate ADMITS all four slots (so the row is in the
    claimed domain rather than an accident), and then runs all four sub-steps
    against ``stepping.py`` with the move floor and the subnormal census the rest of
    the gate uses.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    residency_probe = type("_R", (), {"mirror": lambda *a, **k: None})()
    for label, kwargs, why in ADMITTED_DOMAIN:
        grid, fields, pml = build(kwargs["cell"], kwargs["boundaries"],
                                  kwargs["k_point"], 0.35, 20260815,
                                  dimensions=kwargs.get("dimensions", 3),
                                  thickness=kwargs.get("thickness"))
        admitted = {}
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            verdict = (cx.complex_pml_curl_coverage(fields, pml, slot,
                                                    residency_probe)
                       if slot in SUB_STEPS else
                       cx.complex_constitutive_coverage(
                           fields, pml, "H" if slot == "update_H" else "E",
                           residency_probe))
            admitted[slot] = {"covered": bool(verdict.covered),
                              "reasons": list(verdict.reasons)}
        base = snapshot(fields)
        codes = boundary_codes(grid, pml)
        phases = phase_table(grid, pml)
        dtdx = float(grid.dt / grid.dx)
        inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
        record = layer_record(grid, pml)
        window = preconditions.SubnormalWindow(0, 1, per_array_words=8,
                                               per_intermediate_words=8)
        differences: Dict[str, Dict[str, int]] = {}
        moves: Dict[str, int] = {}
        for sub_step, spec in SUB_STEPS.items():
            restore(fields, base)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                     record=record)
            names = tuple(spec["targets"]) + tuple("fu_" + n
                                                   for n in spec["targets"])
            differences[sub_step] = {n: differing(got[n], after[n]) for n in names
                                     if differing(got[n], after[n])}
            moves[sub_step] = sum(differing(after[n], base[n]) for n in names)
            compared += len(names)
            for volume in names:
                window.observe(f"{label}:{sub_step}:{volume}", after[volume], step=1)
            collect: Dict[str, Any] = {}
            reference_curl([np.array(base[n], copy=True) for n in spec["targets"]],
                           [np.array(base["fu_" + n], copy=True)
                            for n in spec["targets"]],
                           [base[n] for n in spec["sources"]],
                           curl_coefficients(pml, spec["suffix"]),
                           codes, bool(spec["backward"]), dtdx, phases, record,
                           collect=collect)
            for tag, volume in collect.items():
                window.observe_intermediate(f"{label}:{sub_step}:{tag}", volume,
                                            step=1)
        for side, spec in SIDES.items():
            restore(fields, base)
            getattr(stepping, spec["step"])(fields, pml)
            after = snapshot(fields)
            got = run_constitutive_on_device(base, pml, side, inverse, record=record)
            names = tuple(spec["targets"]) + tuple(spec["aux"])
            key = "update_" + side
            differences[key] = {n: differing(got[n], after[n]) for n in names
                                if differing(got[n], after[n])}
            moves[key] = sum(differing(after[n], base[n]) for n in names)
            compared += len(names)
            for volume in names:
                window.observe(f"{label}:{key}:{volume}", after[volume], step=1)
            collect = {}
            reference_constitutive(
                [np.array(base[n], copy=True) for n in spec["targets"]],
                [np.array(base[n], copy=True) for n in spec["aux"]],
                [base[n] for n in spec["sources"]],
                None if side == "H" else [inverse[n] for n in spec["targets"]],
                constitutive_coefficients(pml, spec["suffix"]), record,
                collect=collect)
            for tag, volume in collect.items():
                window.observe_intermediate(f"{label}:{key}:{tag}", volume, step=1)
        row = {"case": label, "why": why, "shape": list(grid.shape),
               "codes": list(codes),
               "phases": [None if p is None else [float(np.float32(p.real)),
                                                  float(np.float32(p.imag))]
                          for p in phases],
               "admitted": admitted, "moved_words": moves,
               "differing": {k: v for k, v in differences.items() if v},
               "subnormal": window.report()}
        rows.append(row)
        log(f"[admitted_domain] {label:<28} shape={tuple(grid.shape)} "
            f"moved={sum(moves.values())} "
            f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
            f"subnormal={window.subnormal_words}/{window.observed_words} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["admitted_domain"] = rows
        save(payload, out)
        for slot, verdict in admitted.items():
            assert verdict["covered"], (
                f"{label}: the predicate REFUSED {slot} ({verdict['reasons']}). "
                f"This leg exists to launch the admitted domain; a refused row "
                f"belongs in the refusal enumeration instead")
        for slot, moved in moves.items():
            kit.assert_moved(moved, f"{label}/{slot}")
        assert not row["differing"], row
        preconditions.assert_clean_or_refuse(window, f"admitted_domain/{label}")
    payload["counts"]["admitted_domain"] = compared


def leg_overlap(payload: Dict[str, Any], out: str) -> None:
    """At most one arm may admit any configuration, and on a complex one it is THIS.

    THE SWEEP IS WHAT THE REGISTRY IS FOR. Every arm in the table is asked for its
    verdict on every configuration in the matrix, real and complex, and at most one
    may admit. A family absent from the table could not be swept and its
    disjointness would be an assumption.

    THE SWEEP OUTLIVED THE DEFERRAL IT WAS WRITTEN UNDER. It ran first while these
    arms were ``wired=False`` — registered so they could be swept, skipped by
    ``arms_for`` so they could not dispatch — and the disjointness it measured is
    exactly what licensed the flip to ``wired=True``. The tail below now asserts the
    composition that flip produced: what the family MUST fill.
    """
    from meep_gpu.metal_kernels import arms as arm_registry  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()
    matrix: List[Tuple[str, Any, Any, Any]] = []
    for name, cell, boundaries, k_point in CONFIGS:
        grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
        matrix.append((f"complex/{name}", fields, pml, grid))
    # And the REAL configurations the shipped family owns, so the sweep covers both
    # directions: the complex arm must refuse every one of them BY NAME.
    for label, boundaries in (("real/periodic", "periodic"),
                              ("real/metallic", "metallic")):
        grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                    dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
        real_fields = Fields(grid=grid, force_complex_fields=False)
        real_fields.enable_pml_storage()
        count = int(np.prod(grid.shape))
        eps = (1.45 + 0.3 * np.sin(np.arange(count, dtype=np.float32) * 0.037)
               ).astype(np.float32).reshape(grid.shape)
        real_fields.set_isotropic_epsilon_volume(
            eps, (np.float32(1.0) / eps).astype(np.float32))
        matrix.append((label, real_fields,
                       PML(grid=grid, thickness=tuple((2, 2) for _ in range(3))),
                       grid))

    for label, fields, pml, _grid in matrix:
        residency = metal_launch.Residency()
        context = arm_registry.StepContext(
            fields, pml, residency, (shaders.CONTRACT_OFF,), sources=())
        # PEERS AND WELDS ARE SWEPT SEPARATELY. A peer replaces exactly its own slot,
        # so two peers on one slot is the over-covering dispatch the composer fails
        # closed on. A weld (``spec.is_weld``) replaces a whole seam and its predicate
        # CONTAINS the wired arm's by construction, so it always co-admits with the arm
        # it welds -- "at most one" is unsatisfiable for a weld by design, not by defect.
        # Partitioning on ``wired`` instead would agree on this tree (all 16 unwired arms
        # are welds) while deleting the deferred single-slot peer this sweep guards.
        admitters: Dict[str, List[str]] = {}
        weld_admitters: Dict[str, List[str]] = {}
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            admitting, welded = [], []
            for spec in arm_registry.registered(slot):
                try:
                    verdict = spec.coverage(context, slot)
                except Exception as exc:  # noqa: BLE001 - a raising predicate refuses
                    (welded if spec.is_weld else admitting).append(
                        f"{spec.family}:RAISED {exc!r}")
                    continue
                if verdict.covered:
                    name = (f"{spec.family}/{spec.label}"
                            f"{'' if spec.wired else ' (unwired)'}")
                    (welded if spec.is_weld else admitting).append(name)
            admitters[slot] = admitting
            weld_admitters[slot] = welded
        row = {"case": label, "admitters": admitters,
               "weld_admitters": weld_admitters}
        rows.append(row)
        log(f"[overlap] {label:<22} " +
            " ".join(f"{s}={len(v)}" for s, v in admitters.items()) +
            " welds=" + ",".join(f"{s}:{len(v)}" for s, v in weld_admitters.items()
                                 if v) +
            f" ({time.time() - started:.1f}s)")
        payload["legs"]["overlap"] = rows
        save(payload, out)
        for slot, admitting in admitters.items():
            assert not any("RAISED" in a for a in admitting), row
            assert len(admitting) <= 1, (
                f"{label}: {slot} is co-admitted by {admitting}; two PEER products on "
                f"one slot is the over-covering dispatch the composer fails closed on")
            # A weld may co-admit, but only ALONGSIDE the single-slot arm it welds. A
            # weld admitting a slot NO peer admits would cover a configuration no
            # certified sub-step arm covers -- the same over-coverage, wearing a weld's
            # clothes -- so it is asserted rather than skipped.
            welded = weld_admitters[slot]
            assert not any("RAISED" in a for a in welded), row
            assert not welded or admitting, (
                f"{label}: {slot} is admitted by WELD(s) {welded} while NO peer arm "
                f"admits it; the weld covers a configuration no certified sub-step arm "
                f"covers")
        if label.startswith("complex/"):
            assert all(admitting and "complex_fields" in admitting[0]
                       for admitting in admitters.values()), row

    # THE COMPOSITION IS THIS FAMILY'S OWN NOW, and the assertion is RE-POINTED
    # rather than dropped. Until 2026-08-15 the arms registered `wired=False` and
    # this tail asserted `not plan.replaces` — "dispatch must be unchanged by this
    # tranche". `complex_fields.register_arms` has since flipped to `wired=True`,
    # after the composition sweep settled the only question the deferral was ever
    # about (whether wiring could change the answer on a configuration ANOTHER
    # family owns; see that function's docstring). The old assertion therefore
    # FAILED on a tree where the wiring is deliberate — measured here at
    # replaces=('step_B', 'update_H', 'step_D', 'update_E') — and a gate that fails
    # because the tree moved under it is reporting its own staleness, not a defect.
    #
    # WHAT REPLACES IT IS STRICTLY STRONGER, and it is the assertion that has
    # teeth now that the arms dispatch: all four sub-steps must be composed on
    # EVERY complex configuration in the matrix, and every one of them must be
    # THIS family's arm. The old form could only have caught a foreign arm reaching
    # a complex configuration by refusing dispatch outright; this one names it.
    expected_slots = {"step_B", "step_D", "update_H", "update_E"}
    composed: Dict[str, Any] = {}
    for name, cell, boundaries, k_point in CONFIGS:
        _grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
        plan = metal_launch.plan_step(
            fields, pml, residency=metal_launch.Residency(), sources=())
        composed[name] = {"replaces": sorted(plan.replaces),
                          "selected": dict(plan.selected)}
        assert set(plan.replaces) == expected_slots, (
            f"{name}: plan_step composed {sorted(plan.replaces)} on a COMPLEX "
            f"configuration; the complex arms register wired=True and must fill all "
            f"four slots — a missing slot is a SILENT fall back to the array path")
        assert set(plan.selected) == expected_slots and all(
            label == "complex/Bloch" for label in plan.selected.values()), (
            f"{name}: plan_step selected {dict(plan.selected)}; a slot on a complex "
            f"configuration filled by any arm but this family's is the over-covering "
            f"the disjointness sweep above exists to refuse")

    payload["legs"]["composition"] = {
        "wired": True,
        "per_case": composed,
    }
    save(payload, out)
    payload["counts"]["overlap"] = len(rows) + len(composed)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("transcription", leg_transcription),
    ("bindings", leg_bindings),
    ("part_products", leg_part_products),
    ("reference", leg_reference),
    ("constitutive", leg_constitutive),
    ("synthetic", leg_synthetic),
    ("contraction", leg_contraction),
    ("multi_step", leg_multi_step),
    ("reduction", leg_reduction),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("admitted_domain", leg_admitted_domain),
    ("mutations", leg_mutations),
    ("planted_zero", leg_planted_zero),
    ("overlap", leg_overlap),
)

#: Defects that are BYTE-INVISIBLE in float32 on this executor and are therefore
#: pinned by SOURCE-TEXT assertion in the test file rather than by a mutation here.
#: Recorded so the artifact shows they were NOT measured, which is the discipline
#: the BFAST gate states best and which matters MORE on this track, because there is
#: no PTX-equivalent audit to fall back on.
PREDICTED_NULLS = (
    {"defect": "complex add/sub spelled as float2 operators rather than two scalar "
               "operations",
     "why_invisible": "MSL vector arithmetic is element-wise IEEE with no horizontal "
                      "step, so the two spellings emit the same two rounds",
     "pinned_by": "test_metal_complex_fields.test_complex_add_is_component_wise"},
    {"defect": "the order of the two word stores within one complex cell",
     "why_invisible": "the planes do not alias, so no read observes a partial cell",
     "pinned_by": "test_metal_complex_fields.test_stores_are_one_float2_per_cell"},
    {"defect": "which of a phased axis's two operands is connected first",
     "why_invisible": "the two connections are independent and neither reads the "
                      "other's result",
     "pinned_by": "test_metal_complex_fields.test_the_connection_connects_both_operands"},
)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    payload: Dict[str, Any] = {
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
        "counts": {},
    }
    save(payload, arguments.out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device is available on this host")
    if not payload["subnormal_policy"]["admitted"]:
        reasons.extend(payload["subnormal_policy"]["reasons"])
    if reasons:
        return kit.cannot_certify(payload, arguments.out, reasons)

    # A DISTINCT NAME, because the composition probe writes into this same
    # directory and both callers used the default. Measured on
    # results/metal_complex_2026-08-15/: the surviving provenance.json held the
    # PROBE's five sources, and the gate's ten were recoverable only from inside
    # gate.json. The kit added `name` for exactly this and neither caller used it.
    payload["provenance"] = kit.provenance(out_dir, {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "metal_kernels/complex_fields.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "complex_fields.py"),
        "metal_kernels/templates.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
        "metal_kernels/shaders.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "shaders.py"),
        "metal_kernels/plans.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "plans.py"),
        "metal_kernels/device.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
        "metal_kernels/arms.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "arms.py"),
        "metal_kernels/preconditions.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "preconditions.py"),
        "gate_metal_complex.py": os.path.abspath(__file__),
        "metal_gate_kit.py": os.path.join(HERE, "metal_gate_kit.py"),
    }, name="provenance_gate.json")
    save(payload, arguments.out)

    log(f"[env] torch={payload['environment'].get('torch')} "
        f"numpy={np.__version__} "
        f"frontend={payload['environment']['metal_frontend']} "
        f"policy={payload['subnormal_policy']['resolved']}")

    wanted = kit.wanted_legs(arguments.legs)
    ran = kit.run_legs(LEGS, payload, arguments.out, wanted)

    # The kernel sources are hashed: a source string is generated, so the module
    # hash alone would not notice a changed specialisation.
    #
    # BOTH CONTRACTION MODES, as `launch.compute_fingerprints` does for the real
    # family. The guard is a property of the SOURCE here, so `off` and `fast` are
    # different kernels — and legs `synthetic` and `contraction` COMPILE AND LAUNCH
    # the `fast` build on every row. Hashing only `off` would have left a launched
    # specialisation with no fingerprint at all.
    payload["kernel_source_sha256"] = {
        f"{label}/contract-{mode}": templates.source_sha256(source)
        for mode in shaders.CONTRACT_MODES
        for label, source in cx.enumerate_complex_sources(mode).items()}
    compared = sum(int(v) for v in payload["counts"].values())
    return kit.summarize(
        payload, arguments.out,
        claim=("byte-identity to stepping.py on this host for complex64 storage "
               "with Bloch phases, under a CHECKED subnormal-free precondition — "
               "not a stated tolerance"),
        scope=("complex PML curl (step_B/step_D) and constitutive (update_H/"
               "update_E) on complex64 storage, each in its per-chunk forms keyed on "
               "the chunk record (the split-field recurrence on dsigu cells, the "
               "dsig-only form elsewhere; the accumulation on dsigw cells, the output "
               "select elsewhere), every product one per part and every Bloch "
               "connection MEEP's (copy, negate, or the unfused product on plane 0 "
               "of the unshifted operand); periodic and metallic ghost rules; partner "
               "presence; no fold, no cylindrical axis, no beta, no "
               "BFAST, no chi2/chi3, no dispersion, no conductivity, no off-diagonal "
               "row. WIRED: the arms register wired=True and plan_step composes all "
               "four sub-steps from this family on every complex configuration in "
               "the matrix — asserted per case in leg `overlap`"),
        stated_weakness=(
            "two, and both are properties of the executor rather than of the "
            "transcription. (1) NO PTX-EQUIVALENT AUDIT exists here: "
            "compile_shader exposes no disassembly, so the mutation legs and this "
            "gate are the only arbiters. (2) THE SUBNORMAL PRECONDITION IS "
            "RECONSTRUCTED, NOT READ: leg `precondition` censuses every stored "
            "operand and result and re-forms the kernel's unstored intermediates "
            "(the curl, and the constitutive kps*fw / kms*fw_previous products) on "
            "the HOST in the same operand order, on the cells whose form forms "
            "them, because a kernel's registers are not readable from here. That "
            "is weaker than a register read and stronger than censusing only what "
            "was stored"),
        started=started, legs_run=ran, compared=compared, certified=True,
        extra={"predicted_nulls": list(PREDICTED_NULLS)})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
