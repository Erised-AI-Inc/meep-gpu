"""Byte gate for the Metal complex-field / Bloch PML curl and constitutive kernels.

THE CLAIM, and the only one: **byte-identity to ``stepping.py`` on this host,
subject to a declared and CHECKED subnormal-free precondition.** Not a stated
tolerance. Leg ``precondition`` is that check and it is required to FIRE on a scaled
control — a precondition never demonstrated to fire is decorative.

THE ORACLE IS IN PROCESS. ``stepping.step_B(fields, pml)`` and the Metal launch run
in ONE process against ONE seed on real ``Grid``/``Fields``/``PML`` objects with
complex64 storage. The in-file NumPy transcription STAYS in a reduced role: it is
the mutation substrate and the only way to exercise a deliberately wrong
configuration or an odd shape. LEG ``transcription`` PINS IT against ``stepping.py``
over multiple full cycles BEFORE any Metal kernel is trusted, which is what stops
"bit-identical" from meaning the kernel reproduced whatever this file wrote twice.

WHAT THIS GATE CARRIES BEYOND THE STANDARD DISCIPLINE, each because the
corresponding Triton gate learned it:

* **the expansion probe comes FIRST and the predicate refuses without an artifact.**
  On Triton the reason was that the kernel could not force its own association. HERE
  THE QUESTION INVERTS: the shader spells the expansion explicitly under the
  file-scope contraction directive, so what is unknown is which arm the HOST ORACLE
  takes. The engine holds NumPy, so leg ``expansion`` measures ``numpy``'s own
  complex64 multiply PER CALL-SITE ORIENTATION on random, signed-zero and underflow
  vectors, writes the artifact, and binds from it. A platform matching NO arm is a
  REFUSAL BY NAME WITH EXAMPLE WORDS;
* **AMBIGUOUS_BOTH is accepted for the real-coefficient orientations and only
  those.** With ``c_im = +0.0`` the fma's addend is an exact zero and both arms are
  exact, so those patterns cannot discriminate and say so;
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
  instead of on ``periodic_kx`` alone — plus the NEGATIVE CONTROL that with the
  expansion probe undiscoverable the composition is EMPTY. That last is the
  silent-fallback shape on this family and it was previously unmeasured: the
  predicate binds its multiply arm from the probe artifact, ``plan_step`` defaults
  ``complex_probe=None``, and the fallback is an environment variable THIS GATE
  sets and a real run does not.

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
import json
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
from meep_gpu.pml import PML  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

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

#: Bound once by the expansion leg and read by every later leg. Deliberately NOT
#: defaulted: a leg that ran before the probe would bind a guess.
EXPANSION: Optional[str] = None
PROBE_ARTIFACT: Optional[str] = None


def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a
    complex128 scalar, so the product is a full complex multiply whose imaginary
    part comes out ``-0.0`` regardless of what ``imag`` held. Measured while writing
    this gate — the first three expansion probes classified NumPy as matching NO arm
    purely because of it. Every complex construction in this file goes through the
    word view.
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

def _shifted(array: Any, axis: int, backward: bool, code: int,
             phase: Optional[complex]) -> Any:
    """``stepping._shift_up`` (:1723) / ``_shift_down`` (:1787) for one axis.

    ``np.roll`` is a pure gather and preserves every bit including signed zeros; the
    metallic wall is then overwritten with an exact complex zero, and a phased
    PERIODIC axis has its single wrapped plane multiplied by the complex64-rounded
    factor — the CONJUGATE on the down shift (S:1818-1822), which is the classic
    sign error that leaves every magnitude plausible and moves only the phase.
    """
    out = np.roll(array, 1 if backward else -1, axis=axis)
    index: List[Any] = [slice(None)] * 3
    if code == METALLIC:
        index[axis] = 0 if backward else array.shape[axis] - 1
        out[tuple(index)] = np.complex64(0)
        return out
    if phase is not None:
        index[axis] = 0 if backward else -1
        factor = np.complex64(phase.conjugate() if backward else phase)
        out[tuple(index)] = out[tuple(index)] * factor
    return out


def _keep(collect: Optional[Dict[str, Any]], label: str, value: Any) -> Any:
    """Snapshot one UNSTORED intermediate for the subnormal census, or do nothing.

    The census may not form its own version of these expressions: an independently
    written copy is a second transcription, and a second transcription is a second
    thing that can be wrong. So the intermediates are taken FROM THE REFERENCE
    ITSELF, which ``leg_transcription`` pins against ``stepping.py`` over full
    cycles. ``collect is None`` on every hot path, and a ``.copy()`` of a value the
    caller is about to overwrite cannot change a bit of the result.
    """
    if collect is not None:
        collect[label] = np.array(value, copy=True)
    return value


def reference_curls(sources: Sequence[Any], codes: Sequence[int], backward: bool,
                    dtdx: float, phases: Sequence[Optional[complex]],
                    collect: Optional[Dict[str, Any]] = None) -> List[Any]:
    """The three UNMASKED ``dtdx * curl`` volumes, exactly as S:1601-1636 forms them.

    Split out of :func:`reference_curl` so leg ``precondition`` can census the
    kernel's largest UNSTORED intermediate on the host. Nothing else moved: the
    caller applies the ownership mask and the recurrence to what this returns, so
    ``leg_transcription`` still pins this expression against ``stepping.py``.

    ``collect`` additionally captures the SIX ROTATED SHIFT BUFFERS. They are
    unstored too, and the Bloch rotation on the wrapped plane is a real complex
    product — ``z.y * p.y`` inside it is a rounded multiply that can land in the
    band while both the field and the phase are outside it.
    """
    a, b, c = sources
    a_y = _keep(collect, "shift:a_y", _shifted(a, 1, backward, codes[1], phases[1]))
    a_z = _keep(collect, "shift:a_z", _shifted(a, 2, backward, codes[2], phases[2]))
    b_x = _keep(collect, "shift:b_x", _shifted(b, 0, backward, codes[0], phases[0]))
    b_z = _keep(collect, "shift:b_z", _shifted(b, 2, backward, codes[2], phases[2]))
    c_x = _keep(collect, "shift:c_x", _shifted(c, 0, backward, codes[0], phases[0]))
    c_y = _keep(collect, "shift:c_y", _shifted(c, 1, backward, codes[1], phases[1]))
    return [dtdx * ((c_y - c) + (b - b_z)),
            dtdx * ((a_z - a) + (c - c_x)),
            dtdx * ((b_x - b) + (a - a_y))]


def reference_curl(targets: Sequence[Any], aux: Sequence[Any],
                   sources: Sequence[Any], coefficients: Dict[str, Any],
                   codes: Sequence[int], backward: bool, dtdx: float,
                   phases: Sequence[Optional[complex]],
                   collect: Optional[Dict[str, Any]] = None) -> None:
    """``stepping.step_B`` / ``step_D`` on complex storage, transcribed, in place.

    ``collect`` gathers every value the kernel FORMS AND NEVER STORES: the six
    rotated shift buffers, the three unmasked curls, and the five partial results of
    the split-field recurrence per target. ``n0`` and ``v0`` are excluded on purpose
    — those ARE stored, so the census sees them as results.
    """
    curls = reference_curls(sources, codes, backward, dtdx, phases, collect)
    for target in range(3):
        _keep(collect, f"curl:{target}", curls[target])

    pairs = (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)) if backward
             else ((0, 0), (1, 1), (2, 2)))
    for target, axis in pairs:
        if codes[axis] == METALLIC:
            index: List[Any] = [slice(None)] * 3
            index[axis] = 0
            curls[target][tuple(index)] = 0

    cycle = (("y", "z"), ("z", "x"), ("x", "y"))
    for target, (first, second) in enumerate(cycle):
        kms, sinv = coefficients["kms_" + first], coefficients["sinv_" + first]
        kms_u, sinv_u = coefficients["kms_" + second], coefficients["sinv_" + second]
        previous = aux[target].copy()
        aux[target] *= kms
        _keep(collect, f"fu*kms:{target}", aux[target])
        aux[target] -= curls[target]
        _keep(collect, f"fu*kms-curl:{target}", aux[target])
        aux[target] *= sinv
        targets[target] *= kms_u
        _keep(collect, f"f*kms_u:{target}", targets[target])
        targets[target] += aux[target]
        _keep(collect, f"f*kms_u+fu:{target}", targets[target])
        targets[target] -= previous
        _keep(collect, f"f*kms_u+fu-fprev:{target}", targets[target])
        targets[target] *= sinv_u


def reference_constitutive(targets: Sequence[Any], aux: Sequence[Any],
                           sources: Sequence[Any],
                           inverse_epsilon: Optional[Sequence[Any]],
                           coefficients: Dict[str, Any],
                           collect: Optional[Dict[str, Any]] = None) -> None:
    """``stepping._apply_constitutive_pml`` (:2065) on the component's OWN axis.

    ``collect`` gathers the two coefficient products and the PARTIAL SUM between
    them. The partial sum was the one the earlier census did not reconstruct, and it
    is the one that can land in the band on its own: ``f + kps*fw`` may cancel to a
    subnormal while ``f``, ``kps*fw`` and the final ``f - kms*fwprev`` are all
    normal.
    """
    for target, axis in enumerate("xyz"):
        kps, kms = coefficients["kps_" + axis], coefficients["kms_" + axis]
        product = (sources[target] if inverse_epsilon is None
                   else sources[target] * inverse_epsilon[target])
        _keep(collect, f"constitutive_product:{target}", product)
        previous = aux[target].copy()
        aux[target][...] = product
        # Named rather than inlined so the census cannot cost a second evaluation:
        # `t += kps * fw` and `p = kps * fw; t += p` are the same two ufunc calls on
        # the same operands, so this is byte-neutral by construction.
        gain = kps * aux[target]
        _keep(collect, f"kps*fw:{target}", gain)
        targets[target] += gain
        _keep(collect, f"f+kps*fw:{target}", targets[target])
        loss = kms * previous
        _keep(collect, f"kms*fwprev:{target}", loss)
        targets[target] -= loss


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
                       suffix: Optional[str] = None) -> Dict[str, Any]:
    """One curl launch THROUGH THE SHIPPED PLAN OBJECT, returning the host results."""
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
            codes, spec["backward"], flags, EXPANSION, contract)
        function = metal_launch.compile_source(text).bloch_pml_curl_step
        if counter is not None:
            function = kit.Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = cx.plan_complex_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, phases, dtdx, EXPANSION, residency,
        functions=functions)
    plan.run()
    residency.sync_out()
    return arrays


def run_constitutive_on_device(state: Dict[str, Any], pml, side: str,
                               inverse_epsilon: Optional[Dict[str, Any]] = None,
                               source: Optional[str] = None,
                               contract: str = shaders.CONTRACT_OFF,
                               suffix: Optional[str] = None,
                               counter: Optional[List[kit.Counter]] = None,
                               ) -> Dict[str, Any]:
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
            side, EXPANSION, contract)
        function = metal_launch.compile_source(text).bloch_constitutive_step
        if counter is not None:
            function = kit.Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = cx.plan_complex_constitutive_from_arrays(
        side, arrays, flat, EXPANSION, residency, functions=functions,
        suffix_key=chosen)
    plan.run()
    residency.sync_out()
    return arrays


# ---------------------------------------------------------------------------
# LEG expansion — the probe that binds the arm, and its refusal path
# ---------------------------------------------------------------------------

def _vectors(rng, kind: str, n: int) -> Any:
    if kind == "random":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "signed_zero":
        v = rng.standard_normal(n).astype(np.float32)
        v[::3] = np.float32(-0.0)
        v[1::5] = np.float32(0.0)
        return v
    if kind == "underflow":
        return (rng.standard_normal(n) * 1e-22).astype(np.float32)
    raise ValueError(kind)


def _fma32(a: Any, b: Any, c: Any) -> Any:
    """round_f32(a*b + c) with ONE rounding, computed EXACTLY.

    ``fractions.Fraction`` rather than a float64 shortcut: ``a*b`` is exact in
    float64 but ``a*b + c`` need not be, and a float64->float32 double rounding would
    show up as a handful of spurious mismatches — which is exactly the signal this
    leg reads. The zero-sign case is handled explicitly because an exact-zero SUM
    carries the sign IEEE addition gives it, and no rational arithmetic knows that.
    """
    from fractions import Fraction  # noqa: PLC0415

    out = np.empty(len(a), dtype=np.float32)
    for index in range(len(a)):
        x, y, z = float(a[index]), float(b[index]), float(c[index])
        product = Fraction(x) * Fraction(y)
        total = product + Fraction(z)
        if total == 0:
            # Both addends are zero (or one is and the other cancels exactly). IEEE
            # round-to-nearest: (+0)+(-0) = +0, (-0)+(-0) = -0.
            if x * y == 0.0 and z == 0.0:
                out[index] = np.float32(x * y) + np.float32(z)
            else:
                out[index] = np.float32(0.0)
            continue
        out[index] = np.float32(float(total))
    return out


def _classify(got: Any, naive: Any, fused: Any) -> Tuple[str, int, int]:
    bad_naive = differing(got, naive)
    bad_fused = differing(got, fused)
    if bad_naive == 0 and bad_fused == 0:
        return cx.AMBIGUOUS_BOTH, bad_naive, bad_fused
    if bad_naive == 0:
        return "NAIVE", bad_naive, bad_fused
    if bad_fused == 0:
        return "FMA_V1", bad_naive, bad_fused
    return "NEITHER", bad_naive, bad_fused


def _pattern_products(rng, kind: str, n: int) -> Dict[str, Tuple[Any, Any, Any, Any, Any]]:
    """(observed, a_re, a_im, b_re, b_im) per call-site orientation."""
    ar, ai = _vectors(rng, kind, n), _vectors(rng, kind, n)
    br, bi = _vectors(rng, kind, n), _vectors(rng, kind, n)
    coefficient = _vectors(rng, kind, n)
    z = complex_from_planes(ar, ai)
    p = complex_from_planes(br, bi)
    zeros = np.zeros(n, dtype=np.float32)
    scalar = np.complex64(complex(float(br[0]), float(bi[0])))
    return {
        "c8_mul_c8": (z * p, ar, ai, br, bi),
        # THE SHAPE S:1862 ACTUALLY USES: an array plane times a complex64 SCALAR.
        "c8_mul_c8_scalar_right": (
            z * scalar, ar, ai,
            np.full(n, np.float32(scalar.real)), np.full(n, np.float32(scalar.imag))),
        "c8_mul_f4_field_left": (z * coefficient, ar, ai, coefficient, zeros),
        "f4_mul_c8_coefficient_left": (coefficient * z, coefficient, zeros, ar, ai),
        "python_float_left": (np.multiply(0.35, z),
                              np.full(n, np.float32(0.35)), zeros, ar, ai),
    }


def leg_expansion(payload: Dict[str, Any], out: str) -> None:
    """Measure which arm the HOST ORACLE takes, per orientation, and BIND it.

    THE CLASSIFYING CLASSES ARE ``random`` AND ``signed_zero``, both subnormal-free.
    ``underflow`` is measured and RECORDED but does NOT classify: it sits inside the
    band the precondition excludes, so a divergence there is the precondition's
    business and not the arm's. Saying so is the honest treatment; dropping the class
    would hide that the band exists, and letting it classify would bind the arm from
    data the claim explicitly does not cover.
    """
    global EXPANSION, PROBE_ARTIFACT
    rng = np.random.default_rng(20260815)
    n = 4096
    rows: List[Dict[str, Any]] = []
    verdicts: Dict[str, List[str]] = {}
    started = time.time()
    for kind in ("random", "signed_zero", "underflow"):
        for name, (got, ar, ai, br, bi) in _pattern_products(rng, kind, n).items():
            naive = complex_from_planes(ar * br - ai * bi, ar * bi + ai * br)
            fused = complex_from_planes(_fma32(ar, br, -(ai * bi)),
                                        _fma32(ar, bi, (ai * br)))
            arm, bad_naive, bad_fused = _classify(got, naive, fused)
            row = {"pattern": name, "class": kind, "arm": arm,
                   "naive_differing_words": bad_naive,
                   "fma_v1_differing_words": bad_fused,
                   "words": 2 * n,
                   "classifies": kind != "underflow"}
            if kind == "underflow":
                kit.predicted_null(row, (
                    "the underflow class sits INSIDE the float32 subnormal band the "
                    "checked precondition excludes; it is measured and recorded, and "
                    "it does not bind the arm"))
            else:
                verdicts.setdefault(name, []).append(arm)
            if arm == "NEITHER":
                index = np.nonzero(kit.words(got) != kit.words(naive))[0][:4]
                row["example_words"] = [
                    {"word": int(w),
                     "observed": float(np.ascontiguousarray(got).view(np.float32)[w]),
                     "naive": float(np.ascontiguousarray(naive).view(np.float32)[w]),
                     "fma_v1": float(np.ascontiguousarray(fused).view(np.float32)[w])}
                    for w in index]
            rows.append(row)
            log(f"[expansion] {name:<28} {kind:<12} {arm:<15} "
                f"naive={bad_naive:<6} fma={bad_fused:<6} ({time.time()-started:.1f}s)")
            payload["legs"]["expansion"] = rows
            save(payload, out)

    # COLLAPSING THE CLASSES, and the rule is not "they must all agree". A class
    # whose products are all EXACT cannot discriminate — the fma's addend is exact
    # too, so both arms give the same bits — and reports AMBIGUOUS_BOTH. That is a
    # measurement, not a disagreement, and it is exactly what the signed-zero table
    # does. What may NOT happen is two DIFFERENT real arms for one orientation.
    patterns: Dict[str, str] = {}
    for name, arms in sorted(verdicts.items()):
        assert "NEITHER" not in arms, (
            f"{name}: the reference matches NEITHER arm ({arms}). REFUSED BY NAME — "
            f"this family may not be certified on a platform whose complex multiply "
            f"is neither of the two forms the shader can spell; see the example "
            f"words in the artifact")
        decided_here = sorted({arm for arm in arms if arm != cx.AMBIGUOUS_BOTH})
        assert len(decided_here) <= 1, (
            f"{name}: the classifying classes disagree ({arms}); one specialisation "
            f"cannot represent an orientation whose arm depends on the data class")
        patterns[name] = decided_here[0] if decided_here else cx.AMBIGUOUS_BOTH
    decided = sorted({arm for arm in patterns.values() if arm != cx.AMBIGUOUS_BOTH})
    assert len(decided) == 1, (
        f"the discriminating orientations do not agree: {patterns}. One arm cannot "
        f"represent a platform whose orientations disagree, and a disagreeing "
        f"platform earns a NEW arm rather than a shrug")
    # A probe on which NOTHING discriminated would bind an arm from a measurement
    # that could not tell them apart.
    assert any(arm != cx.AMBIGUOUS_BOTH for arm in patterns.values()), (
        "every orientation classified AMBIGUOUS_BOTH: nothing discriminated, so "
        "this probe measured no platform fact and may not bind an arm")

    record = {"backend": cx.PROBE_BACKEND, "patterns": patterns,
              "numpy": np.__version__, "measured": time.strftime("%Y-%m-%d")}
    PROBE_ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(out)),
                                  "complex_expansion_probe.json")
    with open(PROBE_ARTIFACT, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(record)  # bytes THIS process imported; see gate_provenance
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.environ[cx.PROBE_PATH_ENVIRONMENT] = PROBE_ARTIFACT
    EXPANSION = cx.expansion_from_probe(record)
    assert EXPANSION is not None

    # THE REFUSAL PATH, exercised rather than described. A record that is missing,
    # for the wrong backend, or internally disagreeing must bind NOTHING.
    refusals = {
        "missing": None,
        "wrong_backend": {"backend": "cupy", "patterns": patterns},
        "unclassified": {"backend": cx.PROBE_BACKEND,
                         "patterns": {**patterns, cx.PROBE_PATTERNS[0]: "NEITHER"}},
        "disagreeing": {"backend": cx.PROBE_BACKEND,
                        "patterns": {**patterns,
                                     cx.PROBE_PATTERNS[0]: "NAIVE",
                                     cx.PROBE_PATTERNS[1]: "FMA_V1"}},
        "all_ambiguous": {"backend": cx.PROBE_BACKEND,
                          "patterns": {p: cx.AMBIGUOUS_BOTH
                                       for p in cx.PROBE_PATTERNS}},
    }
    for label, bad in refusals.items():
        assert cx.expansion_from_probe(bad) is None, (
            f"the {label!r} probe record bound an arm; a probe that cannot decide "
            f"must refuse by name rather than default")
    payload["legs"]["expansion_binding"] = {
        "arm": EXPANSION, "patterns": patterns, "artifact": PROBE_ARTIFACT,
        "refusals_exercised": sorted(refusals)}
    save(payload, out)
    log(f"[expansion] BOUND arm={EXPANSION} from {PROBE_ARTIFACT}")


# ---------------------------------------------------------------------------
# LEG call_site_shapes — the probe measures a shape the engine never uses
# ---------------------------------------------------------------------------

def leg_call_site_shapes(payload: Dict[str, Any], out: str) -> None:
    """Does the reference take the BOUND ARM at the shapes ``stepping.py`` uses?

    THE PROBE CLASSIFIES A SHAPE THE ENGINE NEVER EVALUATES. ``leg_expansion``
    multiplies contiguous, out-of-place, equal-length 1-D arrays. Every call site on
    the complex step path is something else:

    * ``S:1862``  ``shifted[face] *= complex64_scalar`` — IN PLACE, and on axes y and
      z the face view is NOT CONTIGUOUS;
    * ``S:1929``  ``fu *= kms`` — IN PLACE, against a float32 ``(n,1,1)`` BROADCAST;
    * ``S:2086``  ``kps * fw`` — out of place, float32 broadcast on the LEFT;
    * ``S:982``   ``D * inv_eps`` — out of place, a full float32 VOLUME on the right;
    * ``S:1635``  ``multiply(dtdx, total, out=total)`` — IN PLACE, python float LEFT.

    Contiguity, broadcasting and output aliasing all steer numpy's loop selection, so
    "the probe's arm is the engine's arm" was an ASSUMPTION. This leg measures it.
    Each shape must classify as the bound arm or as AMBIGUOUS_BOTH; a shape that
    classifies as the OTHER arm, or as NEITHER, means the artifact bound an arm for a
    shape the engine does not evaluate and the family may not be certified from it.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    rng = np.random.default_rng(20260816)
    shape = (12, 10, 9)
    count = int(np.prod(shape))

    def seeded() -> Any:
        real = (rng.standard_normal(shape) * 0.37).astype(np.float32)
        imag = (rng.standard_normal(shape) * 0.29).astype(np.float32)
        real.reshape(-1)[::17] = np.float32(-0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        return complex_from_planes(real, imag)

    def planes(volume: Any) -> Tuple[Any, Any]:
        flat = np.ascontiguousarray(volume).reshape(-1).view(np.float32)
        return np.array(flat[0::2], copy=True), np.array(flat[1::2], copy=True)

    def classify(observed: Any, ar, ai, br, bi) -> Tuple[str, int, int]:
        naive = complex_from_planes(ar * br - ai * bi, ar * bi + ai * br)
        fused = complex_from_planes(_fma32(ar, br, -(ai * bi)),
                                    _fma32(ar, bi, (ai * br)))
        return _classify(np.ascontiguousarray(observed), naive, fused)

    cases: List[Tuple[str, Any, Any, Any, Any, Any]] = []

    phase = np.complex64(complex(np.exp(2j * np.pi * 0.3)))
    for axis in range(3):
        volume = seeded()
        face: List[Any] = [slice(None)] * 3
        face[axis] = -1
        before = np.array(volume[tuple(face)], copy=True)
        contiguous = bool(volume[tuple(face)].flags.c_contiguous)
        volume[tuple(face)] *= phase
        ar, ai = planes(before)
        cases.append((f"S1862 face *= complex scalar, axis={axis} "
                      f"(contiguous={contiguous})",
                      volume[tuple(face)], ar, ai,
                      np.full(ar.size, np.float32(phase.real)),
                      np.full(ar.size, np.float32(phase.imag))))

    zeros = np.zeros(count, dtype=np.float32)
    for axis in range(3):
        broadcast_shape = tuple(shape[axis] if a == axis else 1 for a in range(3))
        kms = (rng.standard_normal(broadcast_shape) * 0.6 + 0.3).astype(np.float32)
        fu = seeded()
        ar, ai = planes(fu)
        fu *= kms
        cases.append((f"S1929 fu *= kms broadcast, axis={axis} (in place)",
                      fu, ar, ai,
                      np.broadcast_to(kms, shape).reshape(-1).astype(np.float32),
                      zeros))
        kps = (rng.standard_normal(broadcast_shape) * 0.6 + 0.3).astype(np.float32)
        fw = seeded()
        br, bi = planes(fw)
        cases.append((f"S2086 kps * fw broadcast LEFT, axis={axis}",
                      kps * fw,
                      np.broadcast_to(kps, shape).reshape(-1).astype(np.float32),
                      zeros, br, bi))

    inverse = (1.0 / (1.45 + 0.3 * np.sin(np.arange(count, dtype=np.float32) * 0.037))
               ).astype(np.float32).reshape(shape)
    displacement = seeded()
    ar, ai = planes(displacement)
    cases.append(("S982 D * inv_eps full volume, field LEFT",
                  displacement * inverse, ar, ai, inverse.reshape(-1), zeros))

    total = seeded()
    br, bi = planes(total)
    np.multiply(0.35, total, out=total)
    cases.append(("S1635 multiply(dtdx, total, out=total), float LEFT",
                  total, np.full(count, np.float32(0.35)), zeros, br, bi))

    for label, observed, ar, ai, br, bi in cases:
        arm, bad_naive, bad_fused = classify(observed, ar, ai, br, bi)
        row = {"call_site": label, "arm": arm, "bound_arm": EXPANSION,
               "naive_differing_words": bad_naive,
               "fma_v1_differing_words": bad_fused,
               "words": 2 * int(np.size(ar))}
        rows.append(row)
        log(f"[call_site_shapes] {label:<58} {arm:<15} "
            f"naive={bad_naive:<5} fma={bad_fused:<5} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["call_site_shapes"] = rows
        save(payload, out)
        assert arm in (EXPANSION, cx.AMBIGUOUS_BOTH), (
            f"{label} classified {arm!r} while the probe bound {EXPANSION!r}. The "
            f"arm was measured on contiguous out-of-place 1-D arrays and this is a "
            f"shape the ENGINE evaluates, so the binding does not describe the "
            f"reference and this family may not be certified from it")
    discriminating = sum(1 for row in rows if row["arm"] != cx.AMBIGUOUS_BOTH)
    payload["legs"]["call_site_shape_coverage"] = {
        "call_sites": len(rows), "discriminating": discriminating}
    save(payload, out)
    assert discriminating > 0, (
        "every real call-site shape classified AMBIGUOUS_BOTH, so this leg "
        "discriminated nothing and cannot corroborate the probe's binding")
    payload["counts"]["call_site_shapes"] = len(rows)


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
                        codes, spec["backward"], dtdx, phases)
                    if sub_step == "step_B":
                        side_spec = SIDES["H"]
                        reference_constitutive(
                            [mirror[n] for n in side_spec["targets"]],
                            [mirror[n] for n in side_spec["aux"]],
                            [mirror[n] for n in side_spec["sources"]],
                            None, constitutive_coefficients(pml, side_spec["suffix"]))
                    else:
                        side_spec = SIDES["E"]
                        reference_constitutive(
                            [mirror[n] for n in side_spec["targets"]],
                            [mirror[n] for n in side_spec["aux"]],
                            [mirror[n] for n in side_spec["sources"]],
                            [inverse[n] for n in side_spec["targets"]],
                            constitutive_coefficients(pml, side_spec["suffix"]))

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
            for sub_step, spec in SUB_STEPS.items():
                restore(fields, base)
                getattr(stepping, sub_step)(fields, pml)
                after = snapshot(fields)
                got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases)
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
                got = run_constitutive_on_device(base, pml, side, inverse)
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

    THE GUARD IS MEASURED INERT HERE, AND THE ASSERTION SAYS SO. Under the
    probe-bound FMA_V1 arm every product this family emits is an EXPLICIT ``fma()``
    call, so flipping ``contract(off)`` to ``contract(fast)`` leaves no implicit
    ``a*b + c`` for the compiler to fuse and moves zero words across the whole
    sweep. The sweep therefore asserts ``guard_differences == 0``: a nonzero count
    would REFUTE the recorded reason, and the reason — not the assertion — is what
    would then have to be re-derived.

    THAT IS NOT "THE GUARD IS DECORATIVE", AND THIS LEG IS THE WRONG PLACE TO
    SETTLE IT. Leg ``contraction`` compiles the NAIVE arm, where the products are
    bare and contractible, and requires the directive to move bytes there. The two
    legs together are the claim: the selector reaches the emitted code (measured on
    NAIVE) and has nothing left to do under the shipped arm (measured here).

    (An earlier revision of this docstring said the ``fast`` build "is required to
    DIFFER somewhere in the sweep", which is the exact opposite of what the code
    asserts. A reader who trusted it would have read a passing gate as a failing
    one, or relaxed the assertion to match the prose.)

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
        window = preconditions.SubnormalWindow(0, 1, per_array_words=64,
                                               per_intermediate_words=64)
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
                            codes, spec["backward"], dtdx, phases, collect=collect)
                        got = run_curl_on_device(state, None, sub_step, codes, dtdx,
                                                 phases, flat=flat)
                        names = tuple(expect)
                        bad = {n: differing(got[n], expect[n]) for n in names}
                        moved = sum(differing(expect[n], state[n]) for n in names)
                        compared += len(names)
                        fast = run_curl_on_device(state, None, sub_step, codes, dtdx,
                                                  phases, flat=flat,
                                                  contract=shaders.CONTRACT_FAST)
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
                collect=collect)
            residency = metal_launch.Residency()
            plan = cx.plan_complex_constitutive_from_arrays(
                side, arrays, flat, EXPANSION, residency, suffix_key=f":{shape}")
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
    payload["legs"]["synthetic_guard"] = kit.predicted_null(
        {"guard_differences": guard_differences, "rows": len(rows)},
        "MEASURED INERT, and this is a result rather than a gap: under the "
        "probe-bound FMA_V1 arm every product in this family is an EXPLICIT fma() "
        "call, so no `a*b + c` pattern survives for implicit contraction to change. "
        "Flipping the directive moved 0 words across the whole sweep. The directive "
        "is NOT decorative — leg `contraction` measures it against the NAIVE arm, "
        "where the products are contractible and it moves bytes — and it stays in "
        "the source because a later edit that spells a bare product would need it.")
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
    assert guard_differences == 0, (
        f"the contraction directive moved bytes in {guard_differences} synthetic "
        f"rows under the fma arm. That contradicts the recorded reason (every "
        f"product is an explicit fma), so the reason is wrong and must be "
        f"re-derived rather than the assertion relaxed")


def _broadcast(shape, key: str):
    axis = "xyz".index(key[-1])
    return tuple(shape[axis] if a == axis else 1 for a in range(3))


# ---------------------------------------------------------------------------
# LEG contraction — where the directive IS load-bearing, measured on both arms
# ---------------------------------------------------------------------------

def leg_contraction(payload: Dict[str, Any], out: str) -> None:
    """The contraction directive, measured on BOTH expansion arms.

    Leg ``synthetic`` measured the directive INERT under the shipped FMA_V1 arm, and
    the reason is structural: every product there is an explicit ``fma()`` call, so
    there is no ``a*b + c`` left for implicit contraction to touch. That could be
    read as "the guard is decorative", and this leg is what settles which it is.

    Under the NAIVE arm the products ARE bare ``(z.x*p.x) - (z.y*p.y)`` and the
    directive has something to do. Four measurements:

    * ``naive_off`` vs ``naive_fast`` — must DIFFER, which proves the SELECTOR is
      live and the directive reaches the emitted code at all;
    * ``naive_off`` vs the reference — must DIFFER, which is the arm being
      load-bearing (the same fact mutation m12 plants from the other side);
    * ``naive_fast`` vs the reference — RECORDED, not asserted. If the compiler
      contracts the naive difference into exactly ``fma(z.x, p.x, -(z.y*p.y))`` this
      is zero, and that would say the fused arm is reachable two ways;
    * ``fma_off`` vs ``fma_fast`` — the inert measurement, restated here beside its
      counterpart so the artifact carries the comparison rather than the conclusion.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    shape = (7, 5, 3)
    state = synthetic_state(shape, 424242)
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
                       codes, spec["backward"], dtdx, phases)
        flags, _ = cx.phase_arguments(tuple(phases), backward=bool(spec["backward"]))
        built: Dict[str, Dict[str, Any]] = {}
        for arm in ("FMA_V1", "NAIVE"):
            for mode in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
                source = cx.bloch_curl_source(codes, spec["backward"], flags, arm,
                                              mode)
                built[f"{arm}/{mode}"] = run_curl_on_device(
                    state, None, sub_step, codes, dtdx, phases, flat=flat,
                    source=source)
        def delta(left: str, right: Optional[str] = None) -> int:
            target = expect if right is None else built[right]
            return sum(differing(built[left][n], target[n]) for n in names)

        row = {"sub_step": sub_step,
               "fma_off_vs_reference": delta("FMA_V1/off"),
               "fma_off_vs_fma_fast": delta("FMA_V1/off", "FMA_V1/fast"),
               "naive_off_vs_reference": delta("NAIVE/off"),
               "naive_fast_vs_reference": delta("NAIVE/fast"),
               "naive_off_vs_naive_fast": delta("NAIVE/off", "NAIVE/fast"),
               "words": len(names) * 2 * int(np.prod(shape))}
        rows.append(row)
        log(f"[contraction] {sub_step} fma_off_vs_ref={row['fma_off_vs_reference']} "
            f"fma_guard_delta={row['fma_off_vs_fma_fast']} "
            f"naive_off_vs_ref={row['naive_off_vs_reference']} "
            f"naive_guard_delta={row['naive_off_vs_naive_fast']} "
            f"naive_fast_vs_ref={row['naive_fast_vs_reference']} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["contraction"] = rows
        save(payload, out)
        assert row["fma_off_vs_reference"] == 0, row
        assert row["naive_off_vs_reference"] > 0, (
            f"{sub_step}: the NAIVE arm reproduced the reference exactly, so the "
            f"probe-bound arm is not load-bearing and the whole expansion machinery "
            f"certifies nothing")
        assert row["naive_off_vs_naive_fast"] > 0, (
            f"{sub_step}: flipping the contraction directive changed NOTHING even "
            f"on the contractible NAIVE spelling. The selector does not reach the "
            f"emitted code, and the pinned mode is then a decoration everywhere")

    # --- THE SAME FOUR MEASUREMENTS ON THE CONSTITUTIVE KERNEL ------------------
    # It is a SECOND template with its own `__CONTRACT__` slot, and this leg used to
    # measure the curl's only. "The directive reaches the emitted code" was then a
    # statement about one of the two kernels this family ships, generalised.
    #
    # THE ANSWER IS THE OPPOSITE OF THE CURL'S AND THAT IS THE POINT. Every product
    # on this sub-step carries a REAL coefficient, so the two arms are exact and
    # coincide, and the directive has nothing to contract in either of them: all
    # four builds come out bitwise equal. Asserting that (rather than assuming it,
    # or asserting the curl's answer here) is what turns "the arm binding is
    # load-bearing" from a family-wide claim into the true, narrower one — it is
    # load-bearing for the CURL, through the Bloch rotation, and nowhere else.
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
            {k: v.reshape(_broadcast(shape, k)) for k, v in flat_c.items()})
        built = {}
        for arm in ("FMA_V1", "NAIVE"):
            for mode in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
                arrays = {n: np.array(state[n], copy=True)
                          for n in names + tuple(spec["sources"])}
                if side == "E":
                    arrays.update(inv_eps)
                residency = metal_launch.Residency()
                function = metal_launch.compile_source(
                    cx.bloch_constitutive_source(side, arm, mode)
                ).bloch_constitutive_step
                plan = cx.plan_complex_constitutive_from_arrays(
                    side, arrays, flat_c, arm, residency,
                    functions={shaders.CONTRACT_OFF: function},
                    suffix_key=f":{arm}:{mode}")
                plan.run()
                residency.sync_out()
                built[f"{arm}/{mode}"] = arrays

        def cdelta(left: str, right: Optional[str] = None,
                   _names=names, _built=built, _expect=expect) -> int:
            target = _expect if right is None else _built[right]
            return sum(differing(_built[left][n], target[n]) for n in _names)

        row = {"sub_step": "update_" + side,
               "fma_off_vs_reference": cdelta("FMA_V1/off"),
               "fma_off_vs_fma_fast": cdelta("FMA_V1/off", "FMA_V1/fast"),
               "naive_off_vs_reference": cdelta("NAIVE/off"),
               "naive_fast_vs_reference": cdelta("NAIVE/fast"),
               "naive_off_vs_naive_fast": cdelta("NAIVE/off", "NAIVE/fast"),
               "words": len(names) * 2 * int(np.prod(shape))}
        rows.append(row)
        log(f"[contraction] update_{side} "
            f"fma_off_vs_ref={row['fma_off_vs_reference']} "
            f"fma_guard_delta={row['fma_off_vs_fma_fast']} "
            f"naive_off_vs_ref={row['naive_off_vs_reference']} "
            f"naive_guard_delta={row['naive_off_vs_naive_fast']} "
            f"naive_fast_vs_ref={row['naive_fast_vs_reference']} "
            f"({time.time() - started:.1f}s)")
        kit.predicted_null(row, (
            "THE CONSTITUTIVE KERNEL CONTAINS NO FULL COMPLEX PRODUCT — measured "
            "here, and it is the reason both diagnostics come out zero rather than "
            "an excuse for them. Every multiply on this sub-step has a REAL "
            "coefficient (`kps`, `kms`, `inv_eps`), so the imaginary operand is an "
            "exact +0.0, the fma's addend is exact, and the NAIVE and FMA_V1 arms "
            "coincide bit for bit; with the arms coincident there is also nothing "
            "for the contraction directive to change, in either spelling. The "
            "consequence is a real scope statement: the probe-bound arm is "
            "load-bearing for the CURL ONLY, and only through the Bloch rotation "
            "(leg `contraction` measures 6 words on step_B and 1 on step_D with two "
            "axes phased; mutation m12 plants the swap there and it is caught). "
            "There is deliberately no constitutive twin of m12, and this row is why"))
        payload["legs"]["contraction"] = rows
        save(payload, out)
        assert row["fma_off_vs_reference"] == 0, row
        assert row["naive_off_vs_reference"] == 0, (
            f"update_{side}: the two expansion arms DIVERGED on a sub-step whose "
            f"every coefficient is real ({row}). That refutes the recorded reason — "
            f"a full complex product has appeared in this kernel — and the arm must "
            f"be made load-bearing here with its own mutation rather than the "
            f"assertion relaxed")
        assert row["naive_off_vs_naive_fast"] == 0, (
            f"update_{side}: flipping the contraction directive moved bytes on the "
            f"NAIVE spelling ({row}), so a contractible `a*b + c` survives here "
            f"after all and the guard is load-bearing on this kernel too")
    payload["counts"]["contraction"] = len(rows) * 5


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

        device_state = {k: np.array(v, copy=True) for k, v in base.items()}
        for _ in range(cycles):
            for sub_step, side in (("step_B", "H"), ("step_D", "E")):
                got = run_curl_on_device(device_state, pml, sub_step, codes, dtdx,
                                         phases)
                device_state.update(got)
                got = run_constitutive_on_device(device_state, pml, side, inverse)
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
# LEG zero_cross_terms — EXHAUSTIVE tables, because random data catches none of it
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

#: The arms this leg discriminates, with what each one is.
#:
#: ``shipped`` IS THE SHIPPED TEXT, and that is now CHECKED rather than intended:
#: :func:`_assert_diagnostic_arms_are_the_shipped_spelling` requires these
#: expressions to occur in ``templates.complex_helpers(EXPANSION)`` once the
#: coefficient's local name is normalised (the helper calls it ``c``, this shader
#: calls it ``k``). Without that check the whole exhaustive table certified a HAND
#: COPY: an edit to ``templates._COMPLEX_HELPERS`` would leave this leg green while
#: testing the old spelling. (The needles of m10/m11 would still have caught the
#: edit as NEEDLE-MISSED, so the exposure was bounded — but "another leg would have
#: noticed" is the reasoning this file exists to refuse.)
_FIELD_LEFT_ARMS: Dict[str, Tuple[str, str, bool]] = {
    # label: (re, im, must_match)
    "shipped": ("fma(z.x, k, -(z.y * 0.0f))", "fma(z.x, 0.0f, (z.y * k))", True),
    "folded_zero_cross_terms": ("fma(z.x, k, -0.0f)", "fma(z.x, 0.0f, (z.y * k))",
                                False),
    # THE COEFFICIENT-LEFT FOLD, which had never been measured. The module docstrings
    # quoted 24/128 for it; 24/128 is the PLANE-WISE number and the fold is 12/128 in
    # BOTH orientations, measured here 2026-08-15. A figure no leg reproduces is a
    # figure that can be wrong for a year, so it is an arm now.
    "folded_coefficient_left": ("fma(k, z.x, -0.0f)", "fma(k, z.y,  (0.0f * z.x))",
                                False),
    "plane_wise": ("z.x * k", "z.y * k", False),
    "negation_as_zero_minus": ("fma(z.x, k, 0.0f - (z.y * 0.0f))",
                               "fma(z.x, 0.0f, (z.y * k))", False),
    "negation_as_times_minus_one": ("fma(z.x, k, (z.y * 0.0f) * -1.0f)",
                                    "fma(z.x, 0.0f, (z.y * k))", True),
    # THE ORIENTATION QUESTION, MEASURED RATHER THAN ASSERTED. The Triton track
    # holds that the array path's operand order is normative because a fused
    # expansion changes which factor is fused. That is TRUE FOR A FULL COMPLEX
    # PRODUCT (mutation m13 plants it and it is caught) and FALSE for a REAL
    # coefficient, which this arm measures: with c_im = +0.0 the real parts are
    # bitwise commutative and the imaginary parts reduce to the same single
    # rounding plus a signed zero whose addition is itself commutative. Recorded as
    # an equivalence rather than left as an untested belief.
    "coefficient_left_orientation": ("fma(k, z.x, -(0.0f * z.y))",
                                     "fma(k, z.y, (0.0f * z.x))", True),
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

#: ``c_mul`` — the FULL complex product, which is what the Bloch rotation is — on the
#: same exhaustive table. It had no exhaustive arm before this round: only the
#: real-coefficient helper did, and the full product rested on the expansion probe's
#: sampled ``signed_zero`` class plus mutation m13. That leaves the one case where a
#: full product LOOKS like a real one uncovered by any table — the Brillouin edge,
#: where the phase is exactly ``-1 + 0j`` and every cross term is a signed zero, which
#: is precisely the value at which a collapse is hardest to see.
#:
#: ``None`` means RECORD, not assert. Every product on this table is exact, so an arm
#: that agrees here has been shown equivalent ON THIS CLASS and nothing more; saying
#: "must match" would overclaim and "must differ" would be false.
#:
#: THE THREE NEGATION SPELLINGS BELONG ON THIS TABLE AND WERE NOT ON IT. The module
#: docstrings assert "``-x`` and ``x * -1.0f`` are both exact (0/512) while
#: ``0.0f - x`` misses 36/512" — a 512-word count, which is exactly this
#: 256-pattern table, and no leg computed it. The field-left table's negation arms
#: are a DIFFERENT measurement (12/128 on 64 patterns), so the quoted figures were
#: unreproducible from any artifact this family ships. They are arms now, and the
#: numbers come out 0/512, 0/512 and 36/512.
_FULL_PRODUCT_ARMS: Dict[str, Tuple[str, str, Optional[bool]]] = {
    "shipped": ("fma(z.x, p.x, -(z.y * p.y))", "fma(z.x, p.y,  (z.y * p.x))", True),
    "naive_four_products": ("(z.x * p.x) - (z.y * p.y)",
                            "(z.x * p.y) + (z.y * p.x)", None),
    "operands_swapped": ("fma(p.x, z.x, -(p.y * z.y))",
                         "fma(p.x, z.y,  (p.y * z.x))", None),
    "negation_as_times_minus_one": ("fma(z.x, p.x, (z.y * p.y) * -1.0f)",
                                    "fma(z.x, p.y,  (z.y * p.x))", True),
    "negation_as_zero_minus": ("fma(z.x, p.x, 0.0f - (z.y * p.y))",
                               "fma(z.x, p.y,  (z.y * p.x))", False),
    "plane_wise": ("z.x * p.x", "z.y * p.y", False),
    "conjugated_cross_term": ("fma(z.x, p.x,  (z.y * p.y))",
                              "fma(z.x, p.y,  (z.y * p.x))", False),
    "imaginary_cross_negated": ("fma(z.x, p.x, -(z.y * p.y))",
                                "fma(z.x, p.y, -(z.y * p.x))", False),
}


def _assert_diagnostic_arms_are_the_shipped_spelling() -> Dict[str, Any]:
    """The ``shipped`` arms of both tables must BE the shipped helper's expressions.

    Not "look like". The exhaustive tables are the only place three spellings are
    shown to be load-bearing, and if their ``shipped`` row is a hand copy then the
    tables certify the copy. The helper names the coefficient ``c`` and the
    diagnostic shader names it ``k``, which is the ONLY licensed difference; it is
    normalised here rather than allowed to make the check pass vacuously.
    """
    helper = " ".join(templates.complex_helpers(EXPANSION).split())
    checked: Dict[str, Any] = {}
    for table, rename in ((_FIELD_LEFT_ARMS, True), (_FULL_PRODUCT_ARMS, False)):
        re_text, im_text, _ = table["shipped"]
        for part, text in (("re", re_text), ("im", im_text)):
            wanted = " ".join(text.split())
            if rename:
                wanted = wanted.replace(" k,", " c,").replace("* k)", "* c)")
            key = f"{'field_left' if rename else 'full_product'}_{part}"
            checked[key] = wanted
            assert wanted in helper, (
                f"the diagnostic table's shipped {key} expression {wanted!r} does "
                f"not occur in templates.complex_helpers({EXPANSION!r}). This leg "
                f"is then measuring a HAND COPY of the shader, not the shader, and "
                f"an edit to the emitted helper would leave it green")
    return checked


def leg_zero_cross_terms(payload: Dict[str, Any], out: str) -> None:
    """The exhaustive signed-zero table that makes three spellings load-bearing.

    RANDOM DATA CANNOT SEE ANY OF THIS: measured 0/16,384 words for the folded, the
    plane-wise and the shipped spellings alike. Only the exhaustive table
    discriminates, so this leg enumerates every (z_re, z_im, c) triple over
    {1.5, -2.25, +0.0, -0.0} and compares against ``numpy``.

    TWO ARMS ARE ``must_match`` AND THAT IS A REAL RESULT, not a gap:
    ``x * -1.0f`` agrees with ``-x`` exactly here, which REFUTES the Triton track's
    reason for preferring it. ``0.0f - x`` does not, so the spelling is pinned by a
    measurement in both directions.

    TWO TABLES, and the second was missing. The first sweeps ``c_mul_field_left``
    over every ``(z_re, z_im, c)`` triple (64 patterns). The second sweeps ``c_mul``
    — the FULL complex product, which is what the Bloch rotation is — over every
    ``(z_re, z_im, p_re, p_im)`` quadruple (256 patterns). Before this round only the
    real-coefficient helper had an exhaustive table, so the one case where a full
    product looks exactly like a real one — the Brillouin edge, phase ``-1 + 0j``,
    every cross term a signed zero — rested on a sampled class and one mutation.
    """
    import torch  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()
    payload["legs"]["zero_cross_terms_shipped_spelling"] = (
        _assert_diagnostic_arms_are_the_shipped_spelling())
    patterns = list(itertools.product(_TABLE_VALUES, repeat=3))
    zr = np.array([p[0] for p in patterns], dtype=np.float32)
    zi = np.array([p[1] for p in patterns], dtype=np.float32)
    cc = np.array([p[2] for p in patterns], dtype=np.float32)
    z = complex_from_planes(zr, zi)
    host = z * cc
    device = torch.device("mps")
    tz = torch.from_numpy(np.ascontiguousarray(z)).to(device)
    tc = torch.from_numpy(np.ascontiguousarray(cc)).to(device)
    for label, (re, im, must_match) in _FIELD_LEFT_ARMS.items():
        source = (_DIAGNOSTIC_SHADERS
                  .replace("__CONTRACT__", shaders.contraction_pragma())
                  .replace("__RE__", re).replace("__IM__", im))
        function = metal_launch.compile_source(source).shipped
        result = torch.empty_like(tz)
        function(result, tz, tc, len(patterns))
        torch.mps.synchronize()
        bad = differing(result.cpu().numpy(), host)
        row = {"arm": label, "re": re, "im": im, "must_match": must_match,
               "differing_words": bad, "words": 2 * len(patterns),
               "patterns": len(patterns)}
        rows.append(row)
        log(f"[zero_cross_terms] {label:<28} differing={bad}/{2*len(patterns)} "
            f"must_match={must_match} ({time.time() - started:.1f}s)")
        payload["legs"]["zero_cross_terms"] = rows
        save(payload, out)
        if must_match:
            assert bad == 0, (
                f"{label} must reproduce numpy's bytes exactly on the exhaustive "
                f"signed-zero table and missed {bad} words: {row}")
        else:
            assert bad > 0, (
                f"{label} agreed with numpy on the whole exhaustive table, so the "
                f"spelling it refutes is NOT load-bearing and the shipped source's "
                f"choice is unjustified. A diagnostic arm that never fires proves "
                f"nothing")

    # --- the FULL complex product, on the same exhaustive table -----------------
    full = list(itertools.product(_TABLE_VALUES, repeat=4))
    fz = complex_from_planes(np.array([q[0] for q in full], dtype=np.float32),
                             np.array([q[1] for q in full], dtype=np.float32))
    fp = complex_from_planes(np.array([q[2] for q in full], dtype=np.float32),
                             np.array([q[3] for q in full], dtype=np.float32))
    host_full = fz * fp
    tfz = torch.from_numpy(np.ascontiguousarray(fz)).to(device)
    tfp = torch.from_numpy(np.ascontiguousarray(fp)).to(device)
    full_rows: List[Dict[str, Any]] = []
    for label, (re, im, must_match) in _FULL_PRODUCT_ARMS.items():
        source = (_FULL_PRODUCT_SHADER
                  .replace("__CONTRACT__", shaders.contraction_pragma())
                  .replace("__RE__", re).replace("__IM__", im))
        function = metal_launch.compile_source(source).shipped
        result = torch.empty_like(tfz)
        function(result, tfz, tfp, len(full))
        torch.mps.synchronize()
        bad = differing(result.cpu().numpy(), host_full)
        row = {"arm": label, "re": re, "im": im, "must_match": must_match,
               "differing_words": bad, "words": 2 * len(full),
               "patterns": len(full)}
        if must_match is None:
            kit.predicted_null(row, (
                "every product on this table is EXACT, so an arm that agrees here is "
                "equivalent ON THIS CLASS and nothing more; the discrimination "
                "between the arms lives in leg `expansion`'s random class and in "
                "mutation m13, and asserting either direction here would overclaim"))
        full_rows.append(row)
        log(f"[zero_cross_terms/c_mul] {label:<26} differing={bad}/{2*len(full)} "
            f"must_match={must_match} ({time.time() - started:.1f}s)")
        payload["legs"]["zero_cross_terms_full_product"] = full_rows
        save(payload, out)
        if must_match is True:
            assert bad == 0, (
                f"c_mul/{label} must reproduce numpy's bytes exactly on the "
                f"exhaustive 256-pattern table and missed {bad} words: {row}")
        elif must_match is False:
            assert bad > 0, (
                f"c_mul/{label} agreed with numpy on the whole exhaustive table, so "
                f"the spelling it refutes is not load-bearing here")
    payload["counts"]["zero_cross_terms"] = (len(rows) * 2 * len(patterns)
                                             + len(full_rows) * 2 * len(full))


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
    shipped = cx.bloch_curl_source((PERIODIC,) * 3, False, (1, 1, 1), EXPANSION)
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
            got = run_curl_on_device(zero_state, pml, sub_step, codes, dtdx, phases)
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
            got = run_constitutive_on_device(seeded, pml, side, inverse)
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
    window = preconditions.SubnormalWindow(0, 1, per_array_words=64,
                                           per_intermediate_words=64)
    intermediates: Dict[str, Any] = {}
    if sub_step in SUB_STEPS:
        spec = SUB_STEPS[sub_step]
        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        sources = tuple(spec["sources"])
        got = run_curl_on_device(before, pml, sub_step, codes,
                                 float(grid.dt / grid.dx), phases)
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
                       float(grid.dt / grid.dx), phases, collect=intermediates)
    else:
        side = "H" if sub_step == "update_H" else "E"
        spec = SIDES[side]
        names = tuple(spec["targets"]) + tuple(spec["aux"])
        sources = tuple(spec["sources"])
        got = run_constitutive_on_device(before, pml, side, inverse)
        reference_constitutive(
            [np.array(before[n], copy=True) for n in spec["targets"]],
            [np.array(before[n], copy=True) for n in spec["aux"]],
            [before[n] for n in sources],
            None if side == "H" else [inverse[n] for n in spec["targets"]],
            constitutive_coefficients(pml, spec["suffix"]),
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

CURL_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_flatten_curl_parens": {
        "must_catch": True,
        "why": "the ONE grouping the whole claim rests on; C would associate "
               "`sf - f + s - ss` differently and it is a different float32 number",
        "apply": lambda s: kit.needle(
            kit.needle(kit.needle(s, "float2 t0 = ((c_y - c) + (b - b_z));",
                                  "float2 t0 = (c_y - c + b - b_z);"),
                       "float2 t1 = ((a_z - a) + (c - c_x));",
                       "float2 t1 = (a_z - a + c - c_x);"),
            "float2 t2 = ((b_x - b) + (a - a_y));",
            "float2 t2 = (b_x - b + a - a_y);"),
    },
    "m2_contract_on_is_inert_under_the_fma_arm": {
        # MEASURED NOT CAUGHT, and recorded as such rather than relaxed away. Under
        # the probe-bound FMA_V1 arm every product in this family is an explicit
        # fma() call, so flipping the directive leaves no `a*b + c` for implicit
        # contraction to change and the bytes are identical (0 words across the
        # whole synthetic sweep, and 0 here). The three-valued column exists for
        # exactly this: `False` is the arm that proves a spelling is genuinely
        # equivalent rather than merely untested.
        #
        # THE DIRECTIVE IS NOT DECORATIVE, and leg `contraction` is where that is
        # measured: on the NAIVE arm the products are contractible, flipping the
        # directive moves 6 words (step_B) and 1 word (step_D), and — the striking
        # part — `NAIVE` under contraction reproduces the fused reference EXACTLY
        # (0 words). The fused arm is reachable two ways on this toolchain, and the
        # source spells it explicitly so that reachability is not load-bearing.
        "must_catch": False,
        "why": "under the FMA_V1 arm there is no implicit contraction left to "
               "remove; the directive's effect is measured on the NAIVE arm in leg "
               "`contraction`, where it moves bytes",
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
            "float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z);",
            "float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_y) + n0) - p0, si_y);"),
    },
    "m4_store_before_load": {
        "must_catch": True,
        "why": "the auxiliary's previous value read after its own store",
        "apply": lambda s: kit.needle(s, "float2 p0 = u0[ii];",
                                      "u0[ii] = float2(0.0f, 0.0f); float2 p0 = u0[ii];"),
    },
    "m5_phase_not_conjugated": {
        "must_catch": True,
        "why": "THE CLASSIC SIGN ERROR: the down shift takes the same factor as the "
               "up shift instead of its conjugate (S:1818-1822). Every magnitude "
               "stays plausible and only the phase moves, which is what a band "
               "structure is made of. Planted in the HOST's phase encoding rather "
               "than in the source, because that is where the conjugation lives",
        "host": True,
    },
    # m6/m7/m13 CARRY NO `phases` OVERRIDE, AND THAT ABSENCE IS THE FIX FOR A
    # MEASURED CONFOUND. They used to override the phase table to
    # (exp(2*pi*i*0.3), None, None) while comparing against an oracle
    # `stepping.step_B` had produced on THIS case's own table,
    # (exp(2*pi*i*0.3*1.2), None, exp(-2*pi*i*0.25*0.9)). Measured 2026-08-15: the
    # UNMUTATED shipped kernel launched with that override already differs from the
    # oracle by 1,624 words, so `CAUGHT 1/1` was a statement about the phase table
    # and not about the planted defect — three must-catch mutations that could not
    # fail. On this case's own table the defects are genuinely caught: m6 at 1,432
    # words, m7 at 7,848, m13 at 11. `_assert_mutation_configuration` now refuses
    # any entry whose overrides do not match the reference it is compared against.
    "m6_phase_on_the_wrong_plane": {
        "must_catch": True,
        "why": "the rotation applied to the plane the shift did NOT wrap: a whole "
               "plane of wrong values, no crash",
        "apply": lambda s: kit.needle(s, "bool wx = (i == nxi - 1);",
                                      "bool wx = (i == 0);"),
        "sub_steps": ("step_B",),
    },
    "m7_phase_on_every_lane": {
        "must_catch": True,
        "why": "the wrapped-lane SELECT dropped, so every lane is rotated — the "
               "defect a `where` exists to prevent",
        "apply": lambda s: kit.needle(s, "bool wx = (i == nxi - 1);",
                                      "bool wx = true;"),
        "sub_steps": ("step_B",),
    },
    "m6y_phase_on_the_wrong_plane_y_axis": {
        "must_catch": True,
        "why": "the same defect on the Y block. It is a separate entry because the "
               "y block was NEVER LAUNCHED by any leg before this round — every "
               "case carried k_y = 0 and no synthetic phase set phased y — so a "
               "wrong index variable or a wrong operand pair there was invisible",
        "apply": lambda s: kit.needle(s, "bool wy = (j == nyi - 1);",
                                      "bool wy = (j == 0);"),
        "sub_steps": ("step_B",),
    },
    "m7y_phase_on_every_lane_y_axis": {
        "must_catch": True,
        "why": "the wrapped-lane select dropped on the Y block, for the same reason "
               "m6y exists",
        "apply": lambda s: kit.needle(s, "bool wy = (j == nyi - 1);",
                                      "bool wy = true;"),
        "sub_steps": ("step_B",),
    },
    "m14_phase_axes_crossed": {
        "must_catch": True,
        "why": "the Y block rotates its operands by the X axis's factor. Only "
               "reachable as a MEASUREMENT once two axes carry DIFFERENT phases at "
               "once, which is what the periodic_kxyz case is for; with k_y = 0 "
               "everywhere the crossed factor was the same 1+0j either way",
        "apply": lambda s: kit.needle(
            kit.needle(s, "a_y = wy ? c_mul(a_y, py) : a_y;",
                       "a_y = wy ? c_mul(a_y, px) : a_y;"),
            "c_y = wy ? c_mul(c_y, py) : c_y;",
            "c_y = wy ? c_mul(c_y, px) : c_y;"),
        "sub_steps": ("step_B",),
    },
    "m8_drop_ownership_mask_one_axis": {
        "must_catch": True,
        "why": "one metallic axis stops dropping its unowned cell — a plane of "
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
    "m10_zero_cross_terms_folded": {
        "must_catch": True,
        "why": "the zero cross terms folded to literals — byte-wrong on signed "
               "zeros and INVISIBLE on random data (measured 0/16384), which is why "
               "this mutation runs on the +-0 LATTICE seeding rather than on the "
               "random one. MEASURED REACHABILITY, not a guess: on the random state "
               "it is caught 0/2, because the defect needs a cell whose `z.x * c` is "
               "an exact zero AND whose imaginary plane is negative AND whose curl "
               "is also zero, and a physical-band state supplies the conjunction "
               "essentially never",
        "seeding": "lattice",
        "apply": lambda s: kit.needle(
            s, "fma(z.x, c,    -(z.y * 0.0f)),\n                  fma(z.x, 0.0f,  (z.y * c)));",
            "fma(z.x, c, -0.0f),\n                  fma(z.x, 0.0f, (z.y * c)));"),
    },
    "m11_plane_wise_real_multiply": {
        "must_catch": True,
        "why": "THE HEADLINE RISK: a real-scalar fast path collapses the product to "
               "plane-wise and the whole zero-sign delta evaporates. Same "
               "reachability finding as m10: it needs the zero conjunction, so it "
               "runs on the +-0 lattice",
        "seeding": "lattice",
        "apply": lambda s: kit.needle(
            s, "fma(z.x, c,    -(z.y * 0.0f)),\n                  fma(z.x, 0.0f,  (z.y * c)));",
            "z.x * c, z.y * c);"),
    },
    "m13_phase_multiply_operands_swapped": {
        "must_catch": True,
        "why": "the Bloch rotation spelled `phase * plane` instead of `plane * "
               "phase` (S:1862 puts the plane on the LEFT). For a FULL complex "
               "product the orientation is load-bearing: the imaginary part fuses "
               "z.x*p.y in one spelling and p.x*z.y in the other, and those round "
               "differently. This is the mutation that earns the module's claim "
               "that operand order is normative — the real-coefficient orientations "
               "do NOT earn it and are recorded as equivalent instead",
        "apply": lambda s: kit.needle(s, "c_mul(b_x, px)", "c_mul(px, b_x)"),
        "sub_steps": ("step_B",),
    },
    "m15_neighbour_index_composition_swapped": {
        "must_catch": True,
        "why": "the shifted-neighbour offset composed as `si*nyz + k*nzi + j` "
               "instead of `si*nyz + j*nzi + k`. The flat decode itself is pinned "
               "only INDIRECTLY — by every non-cubic reference row — and a swap "
               "here is the one index defect that stays IN BOUNDS on this case's "
               "(12, 10, 9) grid (k*nzi + j <= 81 < nyz = 90), so it reads a real "
               "cell and returns a smooth wrong field rather than depending on "
               "out-of-range behaviour to be visible. A decode mutation that only "
               "'catches' by reading past a buffer would be certifying undefined "
               "behaviour, which is why the swap is composed this way",
        "apply": lambda s: kit.needle(
            s, "    int ox = si * nyz + j * nzi + k;",
            "    int ox = si * nyz + k * nzi + j;"),
        "sub_steps": ("step_B",),
    },
    "m12_expansion_arm_swapped": {
        "must_catch": True,
        "why": "the NAIVE helper bodies compiled in where the probe measured "
               "FMA_V1: the arm binding is a measurement and this proves it is "
               "load-bearing rather than decorative",
        "swap_expansion": True,
    },
}

CONSTITUTIVE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "c1_flatten_accumulation": {
        "must_catch": True,
        "why": "`((f + kps*src) - kms*prev)` flattened to `f + (kps*src - kms*prev)`",
        "apply": lambda s: kit.needle(
            s,
            "    a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"
            "    a0 = a0 - c_mul_coefficient_left(km_0, prev0);",
            "    a0 = a0 + (c_mul_coefficient_left(kp_0, src0) "
            "- c_mul_coefficient_left(km_0, prev0));"),
    },
    "c2_store_before_load": {
        "must_catch": True,
        "why": "`prev` read after `fw` is written — wrong only where kms != 0, i.e. "
               "INSIDE THE PML ONLY, so it looks like a worse absorber",
        "apply": lambda s: kit.needle(
            s, "    float2 prev0 = w0[ii];\n    float2 src0 =",
            "    w0[ii] = float2(0.0f, 0.0f);\n    float2 prev0 = w0[ii];\n"
            "    float2 src0 ="),
    },
    "c3_dsigw_to_dsig_cycle": {
        "must_catch": True,
        "why": "the coefficient index moved off the component's OWN axis onto the "
               "curl's cycle — MEEP's dsigw is not dsig",
        "apply": lambda s: kit.needle(s, "float kp_0 = kp0[i], km_0 = km0[i];",
                                      "float kp_0 = kp0[j], km_0 = km0[j];"),
    },
    "c4_inv_eps_orientation_is_equivalent": {
        # MEASURED NOT CAUGHT, and the measurement CORRECTS a belief inherited from
        # the Triton track rather than papering over a gap. For a REAL coefficient
        # the two orientations are bitwise identical: the real parts are commutative
        # products with commutative addends, and the imaginary parts are both a
        # single rounding of the same exact product plus a signed zero, whose
        # addition is sign-commutative. Confirmed independently on the exhaustive
        # 64-pattern table by the `coefficient_left_orientation` arm of leg
        # `zero_cross_terms` (0/128 words).
        #
        # ORIENTATION DOES MATTER FOR A FULL COMPLEX PRODUCT, where the two spellings
        # fuse DIFFERENT factors in the imaginary part; mutation m13 plants exactly
        # that and it IS caught. So the source keeps the array path's orientation at
        # every call site — but this gate no longer claims a measurement it cannot
        # make.
        "must_catch": False,
        "seeding": "lattice",
        "why": "`inv_eps * D` instead of `D * inv_eps` (S:982-984). MEASURED "
               "EQUIVALENT, on the STRONGEST class this gate can construct: caught "
               "0/1 on the +-0 lattice (which is the seeding this entry runs on), "
               "0/1 on the physical-band state, and 0 words when all THREE "
               "components are swapped rather than component 0 alone — re-measured "
               "2026-08-15. For a real coefficient the real parts are bitwise "
               "commutative products of commutative addends and the imaginary parts "
               "are both a single rounding of the same exact product plus a signed "
               "zero, whose addition is itself sign-commutative; the exhaustive "
               "64-pattern `coefficient_left_orientation` arm agrees at 0/128. The "
               "lattice seeding is KEPT rather than dropped because 'equivalent on "
               "the class most likely to separate them' is a stronger statement "
               "than 'equivalent on random data'. (An earlier revision of this "
               "string claimed '1/1 on the +-0 lattice' beside a recorded verdict "
               "of CAUGHT 0/1 on that very seeding; had the claim been true, "
               "must_catch=False would have been the wrong column.) Orientation "
               "IS load-bearing for a FULL complex product, where the two spellings "
               "fuse different factors in the imaginary part — m13 plants that and "
               "it is caught",
        "apply": lambda s: kit.needle(
            s, "c_mul_field_left(g0[ii], e0[ii])",
            "c_mul_coefficient_left(e0[ii], g0[ii])"),
        "sides": ("E",),
    },
}


def _assert_mutation_configuration(label: str, entry: Dict[str, Any],
                                   active_codes, active_phases,
                                   reference_codes, reference_phases) -> None:
    """The mutation's configuration must be the REFERENCE's, exactly.

    THIS GUARD EXISTS BECAUSE THREE MUST-CATCH MUTATIONS COULD NOT FAIL. m6, m7 and
    m13 overrode ``phases`` to ``(exp(2*pi*i*0.3), None, None)`` while being compared
    against an oracle ``stepping`` had produced on the case's OWN table,
    ``(exp(2*pi*i*0.3*Lx), None, exp(-2*pi*i*0.25*Lz))``. Measured 2026-08-15: the
    UNMUTATED shipped kernel launched under that override already differed from the
    oracle by 1,624 words, so every one of those ``CAUGHT 1/1`` rows was reporting the
    phase-table mismatch and not the planted defect.

    An override that changes the configuration changes the ORACLE, and a mutation
    leg compares against an oracle it did not rebuild. So the rule is not "prefer no
    override" — it is that any override must be a no-op against the reference, and
    that is checked here rather than left to a reader.

    DEMONSTRATED TO FIRE, because a guard never shown to fire is decorative: called
    directly on 2026-08-15 it passes a matching configuration and raises on BOTH
    override shapes — a phase-table override (the one that was actually shipped) and
    a boundary-code override. The gate's mutation table has no override left, so on
    the certified tree it is silent; the demonstration is what says it would not be.
    """
    assert tuple(active_codes) == tuple(reference_codes), (
        f"{label}: boundary codes {tuple(active_codes)} differ from the reference "
        f"configuration's {tuple(reference_codes)}. The oracle was produced under "
        f"the reference's codes, so this mutation would 'catch' the mismatch rather "
        f"than the defect it names")
    assert tuple(active_phases) == tuple(reference_phases), (
        f"{label}: phase table {tuple(active_phases)} differs from the reference "
        f"configuration's {tuple(reference_phases)}. That is the confound this guard "
        f"was added for — a mutation compared against an oracle built on a different "
        f"phase table cannot fail, whatever it plants")


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)
    # ALL THREE AXES PHASED, WITH THREE DIFFERENT FACTORS. The y block had never been
    # launched by any leg, and a crossed-axis defect (m14) is only measurable when the
    # two factors differ — with k_y = 0 the crossed factor was the same 1+0j.
    name, cell, boundaries, k_point = config("periodic_kxyz")
    grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
    codes = boundary_codes(grid, pml)
    phases = phase_table(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    base = snapshot(fields)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}

    # THE +-0 LATTICE SEEDING, for the two mutations whose defect class a
    # physical-band state does not reach. Measured rather than assumed: m10 and m11
    # are caught 0/2 on the random state above and 2/2 here, and the difference is
    # entirely the seeding. A gate that ran them only on random data would have
    # reported two real defects as uncaught.
    lattice_fields = build(cell, boundaries, k_point, 0.35, 20260815)[1]
    for volume in STATE:
        array = getattr(lattice_fields, volume, None)
        if array is None:
            continue
        real = np.zeros(grid.shape, dtype=np.float32)
        imag = np.zeros(grid.shape, dtype=np.float32)
        real.reshape(-1)[::3] = np.float32(-0.0)
        real.reshape(-1)[1::5] = np.float32(0.25)
        imag.reshape(-1)[2::3] = np.float32(-0.0)
        imag.reshape(-1)[4::7] = np.float32(-0.125)
        array[...] = complex_from_planes(real, imag)
    lattice_base = snapshot(lattice_fields)

    walled = build((1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
                   (0.0, 0.0, 0.0), 0.35, 20260815)
    walled_grid, walled_fields, walled_pml = walled
    walled_base = snapshot(walled_fields)
    walled_codes = boundary_codes(walled_grid, walled_pml)
    walled_phases = phase_table(walled_grid, walled_pml)
    walled_dtdx = float(walled_grid.dt / walled_grid.dx)

    def oracle_for(fields_obj, state, step_name):
        restore(fields_obj, state)
        getattr(stepping, step_name)(fields_obj, pml if fields_obj is fields
                                     else walled_pml)
        return snapshot(fields_obj)

    oracle = {s: oracle_for(fields, base, s) for s in
              ("step_B", "step_D", "update_H", "update_E")}
    lattice_oracle = {}
    for step_name in ("step_B", "step_D", "update_H", "update_E"):
        restore(lattice_fields, lattice_base)
        getattr(stepping, step_name)(lattice_fields, pml)
        lattice_oracle[step_name] = snapshot(lattice_fields)
    walled_oracle = {s: oracle_for(walled_fields, walled_base, s)
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
                # planted there. `phase_arguments` negates the imaginary part for the
                # backward sub-step; handing it `backward=False` on step_D is exactly
                # "the same factor in both directions".
                if sub_step != "step_D":
                    continue
                wrong_flags, wrong_values = cx.phase_arguments(
                    tuple(active_phases), backward=False)
                shipped = cx.bloch_curl_source(active_codes, spec["backward"],
                                               wrong_flags, EXPANSION)
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
                    active_dtdx, EXPANSION, residency,
                    functions={shaders.CONTRACT_OFF: function})
                # The plan encoded the CONJUGATE, correctly. Replace the three phase
                # pairs with the unconjugated encoding: that, and nothing else, is
                # the defect this mutation plants.
                plan._args = plan._args[:-3] + tuple(wrong_values)
                plan.run()
                residency.sync_out()
                got = arrays
            else:
                expansion = ("NAIVE" if entry.get("swap_expansion")
                             and EXPANSION == "FMA_V1" else
                             "FMA_V1" if entry.get("swap_expansion") else EXPANSION)
                shipped = cx.bloch_curl_source(active_codes, spec["backward"], flags,
                                               expansion)
                if "apply" in entry:
                    try:
                        shipped = entry["apply"](shipped)
                    except LookupError:
                        missed = True
                        continue
                got = run_curl_on_device(state, active_pml, sub_step, active_codes,
                                         active_dtdx, active_phases, source=shipped,
                                         counter=counters)
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
            shipped = cx.bloch_constitutive_source(side, EXPANSION)
            try:
                shipped = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            got = run_constitutive_on_device(state, pml, side, inverse,
                                             source=shipped, counter=counters)
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
    #
    # THESE TWO USED TO REPORT A LAUNCH COUNT THEY DID NOT COUNT. Both passed the
    # literal `2` to `harness.record` and built their verdict string by hand, so the
    # DISARMED classification — the whole point of counting launches — was
    # unreachable for them: a plan that never launched would still have been
    # recorded as `launches=2` and would have failed only if `caught` happened to
    # come out short. They now pass the SHIPPED source through the same Counter seam
    # every other mutation uses (the source is unchanged; the defect is the HOST's
    # suffix choice) and go through `MutationHarness.verdict` like everything else.
    counters = []
    caught = ran = 0
    for sub_step, spec in SUB_STEPS.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        flags, _ = cx.phase_arguments(tuple(phases),
                                      backward=bool(spec["backward"]))
        shipped = cx.bloch_curl_source(codes, spec["backward"], flags, EXPANSION)
        got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases,
                                 suffix=wrong_suffix, source=shipped,
                                 counter=counters)
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
        shipped = cx.bloch_constitutive_source(side, EXPANSION)
        got = run_constitutive_on_device(base, pml, side, inverse,
                                         suffix=wrong_suffix, source=shipped,
                                         counter=counters)
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
    forward_source = cx.bloch_curl_source(codes, False, flags, EXPANSION)
    got = run_curl_on_device(base, pml, "step_D", codes, dtdx, phases,
                             source=forward_source, counter=counters)
    names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    caught = int(any(differing(got[n], oracle["step_D"][n]) for n in names))
    harness.record("h3_d_stencil_direction_flipped",
                   kit.MutationHarness.verdict(False, 1,
                                               sum(c.launches for c in counters),
                                               caught),
                   sum(c.launches for c in counters), caught, 1, True,
                   "step_D stepped with the forward stencil")
    payload["counts"]["mutations"] = len(harness.rows)


# ---------------------------------------------------------------------------
# LEG reduction — the k = 0 identity, and the disjointness sweep
# ---------------------------------------------------------------------------

def leg_reduction(payload: Dict[str, Any], out: str) -> None:
    """An UNPHASED build must be byte-identical to the plain complex path.

    That reduction is what the SKIP buys, and it is the reason the array path does
    not multiply by ``1+0j`` at k = 0. Measured here by building the same
    configuration twice — once with every phase flag 0 and once with the phase
    values present but the flags forced on at ``1+0j`` — and requiring the FIRST to
    match ``stepping.py`` and the SECOND to be a genuinely different source string.

    The second half is what stops the leg being vacuous: if a ``1+0j`` rotation
    happened to be byte-identical, the skip would be a performance detail rather
    than a correctness one, and the artifact should say which it is.
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
    for sub_step, spec in SUB_STEPS.items():
        restore(fields, base)
        getattr(stepping, sub_step)(fields, pml)
        after = snapshot(fields)
        got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases)
        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        bad = sum(differing(got[n], after[n]) for n in names)
        unphased = cx.bloch_curl_source(codes, spec["backward"], (0, 0, 0), EXPANSION)
        identity = cx.bloch_curl_source(codes, spec["backward"], (1, 1, 1), EXPANSION)
        row = {"case": name, "sub_step": sub_step, "differing_words": bad,
               "unphased_source_bytes": len(unphased),
               "identity_source_bytes": len(identity),
               "sources_differ": unphased != identity,
               "skips_emitted": unphased.count("SKIPPED")}
        rows.append(row)
        log(f"[reduction] {sub_step} differing={bad} skips={row['skips_emitted']} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["reduction"] = rows
        save(payload, out)
        assert bad == 0, row
        assert row["skips_emitted"] == 3, (
            "an unphased build must emit a SKIP per axis and no rotation at all; "
            f"it emitted {row['skips_emitted']}")
        assert row["sources_differ"], (
            "the unphased build and the forced-1+0j build produced the SAME source, "
            "so the skip is not observable and this leg certifies nothing")
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
    probe = cx.load_expansion_probe(PROBE_ARTIFACT)
    for label, kwargs, why in ADMITTED_DOMAIN:
        grid, fields, pml = build(kwargs["cell"], kwargs["boundaries"],
                                  kwargs["k_point"], 0.35, 20260815,
                                  dimensions=kwargs.get("dimensions", 3),
                                  thickness=kwargs.get("thickness"))
        admitted = {}
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            verdict = (cx.complex_pml_curl_coverage(fields, pml, slot,
                                                    residency_probe, probe=probe)
                       if slot in SUB_STEPS else
                       cx.complex_constitutive_coverage(
                           fields, pml, "H" if slot == "update_H" else "E",
                           residency_probe, probe=probe))
            admitted[slot] = {"covered": bool(verdict.covered),
                              "reasons": list(verdict.reasons)}
        base = snapshot(fields)
        codes = boundary_codes(grid, pml)
        phases = phase_table(grid, pml)
        dtdx = float(grid.dt / grid.dx)
        inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
        window = preconditions.SubnormalWindow(0, 1, per_array_words=8,
                                               per_intermediate_words=8)
        differences: Dict[str, Dict[str, int]] = {}
        moves: Dict[str, int] = {}
        for sub_step, spec in SUB_STEPS.items():
            restore(fields, base)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            got = run_curl_on_device(base, pml, sub_step, codes, dtdx, phases)
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
                           codes, bool(spec["backward"]), dtdx, phases,
                           collect=collect)
            for tag, volume in collect.items():
                window.observe_intermediate(f"{label}:{sub_step}:{tag}", volume,
                                            step=1)
        for side, spec in SIDES.items():
            restore(fields, base)
            getattr(stepping, spec["step"])(fields, pml)
            after = snapshot(fields)
            got = run_constitutive_on_device(base, pml, side, inverse)
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
                constitutive_coefficients(pml, spec["suffix"]), collect=collect)
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
    composition that flip produced, in both directions: what the family MUST fill,
    and what it must refuse to fill without a measured expansion arm.
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
            fields, pml, residency, (shaders.CONTRACT_OFF,), sources=(),
            extra={"complex_probe": cx.load_expansion_probe(PROBE_ARTIFACT)})
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

    # THE NEGATIVE CONTROL IS THE SILENT-FALLBACK SHAPE, measured rather than
    # assumed, and it is the fact a reader of this artifact most needs. The
    # predicate binds its multiply arm FROM THE EXPANSION PROBE and `plan_step`
    # defaults `complex_probe=None`, so the family falls back to
    # $MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE — which `leg_expansion` sets at
    # gate_metal_complex.py:729 and which nothing in a real run sets. With that
    # variable UNSET the composition is EMPTY: no complex kernel dispatches, every
    # number the run produces is the array path's, and the log stays green. That is
    # a property of the SHIPPED predicate, not of this gate, and it is why every leg
    # above proves its own launch count instead of trusting a plan.
    saved = os.environ.pop(cx.PROBE_PATH_ENVIRONMENT, None)
    try:
        _, cell, boundaries, k_point = config("periodic_kx")
        _grid, fields, pml = build(cell, boundaries, k_point, 0.35, 20260815)
        unbound = metal_launch.plan_step(
            fields, pml, residency=metal_launch.Residency(), sources=())
    finally:
        if saved is not None:
            os.environ[cx.PROBE_PATH_ENVIRONMENT] = saved
    assert not unbound.replaces, (
        f"with no expansion probe discoverable, plan_step composed "
        f"{sorted(unbound.replaces)}; the arm is a MEASURED platform fact and an "
        f"unbound predicate must refuse rather than pick a default arm")
    assert cx.PROBE_PATH_ENVIRONMENT in os.environ, (
        "the negative control did not restore the probe environment variable; "
        "every later reader of it would be measuring this leg's side effect")

    payload["legs"]["composition"] = {
        "wired": True,
        "per_case": composed,
        "probe_absent_replaces": sorted(unbound.replaces),
        "probe_absent_reasons": {slot: list(reason) for slot, reason
                                 in dict(unbound.reasons).items()}
        if hasattr(unbound, "reasons") else {},
    }
    save(payload, out)
    payload["counts"]["overlap"] = len(rows) + len(composed) + 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("expansion", leg_expansion),
    ("call_site_shapes", leg_call_site_shapes),
    ("transcription", leg_transcription),
    ("bindings", leg_bindings),
    ("zero_cross_terms", leg_zero_cross_terms),
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
    {"defect": "which of a phased axis's two operands is rotated first",
     "why_invisible": "the two rotations are independent and neither reads the "
                      "other's result",
     "pinned_by": "test_metal_complex_fields.test_phase_block_rotates_both_operands"},
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
    if wanted and "expansion" not in wanted:
        # Every later leg reads EXPANSION. Running one alone would bind a guess.
        log("[note] leg 'expansion' forced: it BINDS the arm every other leg reads")
        wanted = ("expansion",) + wanted
    ran = kit.run_legs(LEGS, payload, arguments.out, wanted)

    # The kernel sources are hashed AFTER the arm is bound: a source string is
    # generated, so the module hash alone would not notice a changed specialisation.
    #
    # BOTH CONTRACTION MODES, as `launch.compute_fingerprints` does for the real
    # family. The guard is a property of the SOURCE here, so `off` and `fast` are
    # different kernels — and legs `synthetic` and `contraction` COMPILE AND LAUNCH
    # the `fast` build on every row. Hashing only `off` would have left a launched
    # specialisation with no fingerprint at all.
    payload["kernel_source_sha256"] = {
        f"{label}/contract-{mode}": templates.source_sha256(source)
        for mode in shaders.CONTRACT_MODES
        for label, source in cx.enumerate_complex_sources(EXPANSION, mode).items()}
    compared = sum(int(v) for v in payload["counts"].values())
    return kit.summarize(
        payload, arguments.out,
        claim=("byte-identity to stepping.py on this host for complex64 storage "
               "with Bloch phases, under a CHECKED subnormal-free precondition — "
               "not a stated tolerance"),
        scope=("complex split-field PML curl (step_B/step_D) and the dsigw "
               "constitutive accumulation (update_H/update_E) on complex64 storage; "
               "periodic and metallic ghost rules; per-axis Bloch phase including "
               "the Brillouin edge; no fold, no cylindrical axis, no beta, no "
               "BFAST, no chi2/chi3, no dispersion, no conductivity, no off-diagonal "
               "row. WIRED: the arms register wired=True and plan_step composes all "
               "four sub-steps from this family on every complex configuration in "
               "the matrix — asserted per case in leg `overlap`, together with the "
               "negative control that an undiscoverable expansion probe composes "
               "NOTHING rather than picking a default arm"),
        stated_weakness=(
            "two, and both are properties of the executor rather than of the "
            "transcription. (1) NO PTX-EQUIVALENT AUDIT exists here: "
            "compile_shader exposes no disassembly, so the mutation legs and this "
            "gate are the only arbiters. (2) THE SUBNORMAL PRECONDITION IS "
            "RECONSTRUCTED, NOT READ: leg `precondition` censuses every stored "
            "operand and result and re-forms the kernel's unstored intermediates "
            "(the curl, and the constitutive kps*fw / kms*fw_previous products) on "
            "the HOST in the same operand order, because a kernel's registers are "
            "not readable from here. That is weaker than a register read and "
            "stronger than censusing only what was stored"),
        started=started, legs_run=ran, compared=compared, certified=True,
        extra={"expansion_arm": EXPANSION,
               "expansion_probe": PROBE_ARTIFACT,
               "predicted_nulls": list(PREDICTED_NULLS)})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
