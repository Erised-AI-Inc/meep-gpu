"""Bit-identity gate for the conductivity + PML Triton kernel.

WHY THIS IS A SEPARATE FILE AND NOT A BRANCH OF THE SHARED PROBE. The shared
probe (``probe_fused_kernel_bit_identity.py``) and the shared track adapter
(``track_triton_pml.py``) are being edited RIGHT NOW by concurrent workflows, so
this round does not touch either — the same call ``gate_triton_symmetry.py`` made
for the same reason. Everything reusable is IMPORTED from the shared probe and
nothing is re-implemented: the byte comparator, the ULP key, the ghost rule, the
curl grouping, the ownership mask, the real PML layer, the field seeding, the
shape/boundary/Courant sweep axes and the artifact writer. What is added here is
only what does not exist there — MEEP's FOUR-CASE conductive recurrence, a
coefficient table with an EXACTLY-1.0 inactive run (the shared synthetic tables
never touch 1.0, so under them every cell is case A and three of the four cases
are never exercised), the conductivity volumes, and the decay leg. Folding these
into ``reference_recurrence`` and adding a ``conductive`` axis to the shared case
key is a short edit once those files are free; it is listed with the rest of the
deferred integration in ``meep_gpu/triton_kernels/conductivity.py``'s docstring.

THE COEFFICIENT TABLE IS THE WHOLE DIFFICULTY. ``stepping._apply_conductive_pml_update``
partitions every cell by ``dsig = (kms != 1.0) | (sinv != 1.0)`` and
``dsigu`` likewise, into four cases that are NOT substitutions of one another
(the array path's own docstring: "simply substituting identity coefficients into
the general recurrence is not equivalent after a source has been injected into
D"). On the real ``2d_cond_pml`` layout case A — both directions active — is
6241 of 409600 cells, the four PML corners, 1.5%. A kernel that gets it wrong
produces a corner artefact that looks like a slightly worse absorber. So every
case here records its OWN case mix, and a case that exercised fewer than the
cases it claims to is visible in the artifact rather than assumed away.

Legs, in order:

* ``validate``  — this file's reference against ``stepping.py`` itself, bytewise,
  on NumPy. Runs on a laptop with no GPU; it is what makes "the kernel matches
  the reference" mean "the kernel matches the array path".
* ``synthetic`` — the sweep. Shapes x boundary sets x dtdx {0.5, **0.35**} x
  {step_B, step_D} x conductivity pattern {all three, mixed (1,0,1)} x
  coefficient source {synthetic, real}, guarded and unguarded, comparing the
  target, its ``fu`` AND its ``f_cond`` as bytes.
* ``multi``     — consecutive sub-steps with the sources held fixed, byte-compared
  at every step, to the stated budget.
* ``decay``     — the same recurrence with the sources ZEROED, which is the only
  leg that manufactures subnormals: every case multiplies by
  ``condfac*condinv < 1`` each step on top of the PML decay. It records the
  subnormal cell count per step beside the verdict, because §16 of the plan says
  a subnormal divergence is what blocks dispatch and discovering it as a failure
  is worse than measuring it.
* ``controls``  — the three rows without which ``multi``'s and ``decay``'s
  first-divergence step numbers are uninterpretable. See below.
* ``mutations`` — defects injected into the shipped kernel's SOURCE, each of which
  must be caught.

THE LONG-RUN DIVERGENCE, AND WHAT IS ALLOWED TO ATTRIBUTE IT. Measured twice, on
2026-08-10 and again on 2026-08-19 with identical step numbers: every ``cond =
(1,1,1)`` row is bit-identical for the whole budget, and every ``cond = (1,0,1)``
row — the MIXED pattern, whose middle component is LOSSLESS — first differs
somewhere between step 160 and step 253. A step number cannot tell a defect in
this kernel from the amplifying float32 disagreement plan §16 measured, and a
diagnostic that only asks "does something else also diverge?" cannot either. This
gate therefore carries the same three controls
``gate_triton_cylindrical.py`` carries, adapted to what is mixed HERE:

* ``null``     — the ORACLE against the ORACLE. Two CuPy reference states, seeded
  identically, stepped by :func:`reference_conductive_pml_step` and by nothing
  else, Triton nowhere in the row. It answers "is the oracle even deterministic,
  and is this harness's own comparison quiet?" and it MUST be identical for the
  whole budget. Without it every first-divergence number in this file could be
  reduction-order noise.
* ``one_ulp``  — the oracle against itself with the low mantissa bit of ONE NORMAL
  float in the lossless component's ``fu`` flipped after the first step. It
  answers "does THIS configuration amplify a last-bit disagreement at all?", and
  it is read PER CONFIGURATION rather than as a fleet count. Measured 2026-08-19,
  320 steps: seven of eight rows amplify and ``(32,32,32) step_B driven`` does
  not — and in that same configuration the kernel is bit-identical for the whole
  budget, as are all four ``step_B driven`` rows. A driven sub-step walks to a
  fixed point where a last bit is rounded away rather than grown. So the control
  and the kernel agree row for row, which is what the leg is for; see
  :func:`control_pairs`.
* ``substitution`` — the DISCRIMINATOR, and the one thing the 2026-08-10
  diagnostic could not deliver. The three curl targets are INDEPENDENT in these
  legs (the six source volumes are held fixed and never written), so the lossless
  component's trajectory does not depend on what the other two are doing. That
  makes a clean substitution possible: the same shape, tables, seeds, boundaries,
  dtdx and budget are stepped TWICE side by side — once through
  ``plan_conductive_from_arrays`` at ``cond = (1,0,1)``, and once through the
  SEPARATELY WELDED ordinary ``launch.plan_from_arrays``, all lossless — and both
  are compared against their own oracles AND against each other, per component,
  at every step. Budget and defect predict opposite outcomes here and the row
  reports which happened:

  ===================================================  ==============  ==============
  measurement                                          BUDGET predicts DEFECT predicts
  ===================================================  ==============  ==============
  conductive components (field, ``fu``, ``f_cond``)    never differ    differ
  conductive plan's lossless state vs plain plan's     identical       differ
  plain plan vs its own oracle                         same step       clean
  ===================================================  ==============  ==============

Every first divergence, in every leg, also records FORENSICS — how many floats
differ, and for each side how many of them are subnormal and how many are exactly
zero. That is what separates "the two kernels compute this recurrence
differently" from "one side flushed a subnormal the other side kept", which are
different findings with the same step number. CuPy 13.5.1 appends ``-ftz=true``
to every NVRTC compile and Triton 3.1.0 keeps subnormals natively
(``meep_gpu/subnormal_policy.py``), so unless a policy is installed the oracle
FLUSHES where the kernel KEEPS, and the record must be able to say so rather than
leave it inferred. ``--subnormal-policy`` drives both executors to one policy for
a confirming arm; the default installs nothing and is what every earlier run of
this gate measured.

WHAT THE FORENSICS FOUND, 2026-08-19 on GPU 3, no policy installed. At every
first divergence in the ``multi`` and ``decay`` legs the differing floats number
ONE or TWO out of 1287, and every one of them is EXACTLY ZERO on the array-path
side and SUBNORMAL on the kernel side — ``fu_Dy`` at 1.03e-38 against 0.0,
``fu_By`` at 1.06e-38 against 0.0, against a smallest normal of 1.1755e-38. The
``flush_signature`` count equals the differing count. The divergence is the two
executors' documented disagreement about the subnormal band, not about the
recurrence.

That also retires a measurement hazard this gate carried: ``subnormal_cells``
counts subnormals in the REFERENCE, and the reference is the side that flushes,
so it reported 0 at every checkpoint of every diverging row. Counting subnormals
only on the flushing side is guaranteed to find none. The two-sided forensics are
what made the mechanism visible.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=6 python -u gate_triton_conductivity.py \\
        --out results/triton_conductivity_<date>/gate.json

and on a laptop with neither CuPy nor Triton::

    python -u gate_triton_conductivity.py --legs validate --out /tmp/validate.json

The controls leg alone, when only the attribution is wanted::

    CUDA_VISIBLE_DEVICES=6 python -u gate_triton_conductivity.py \\
        --legs multi,decay,controls --out results/<fresh dir>/gate.json

Every case prints one flushed line as it lands and the JSON artifact is rewritten
incrementally (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:  # A laptop runs the `validate` leg with neither CuPy nor Triton installed.
    import cupy as cp
except Exception:  # noqa: BLE001 - absence is the norm off the measurement host
    cp = None  # type: ignore[assignment]

# Everything reusable comes from the shared probe. Nothing below re-implements
# any of it — a second byte comparator or a second curl would make this gate's
# verdict about this file rather than about the contract.
import probe_fused_kernel_bit_identity as probe  # noqa: E402

SEED = probe.SEED
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
bit_compare = probe.bit_compare
combine = probe.combine

#: The sweep axes, taken from the shared probe so the two gates cover the same
#: product. 0.35 is MANDATORY: at 0.5 the scaling is exact in binary, and the
#: benchmark case this kernel exists for (``2d_cond_pml``, grid dt 0.0125 / dx
#: 0.025) sits at dtdx = 0.5 EXACTLY — so a gate that ran only the benchmark
#: configuration would certify a contracted kernel.
SHAPES = probe.PML_SHAPES
BOUNDARY_SETS = probe.PML_BOUNDARY_SETS
DTDX = probe.PML_DTDX

#: Per-target conductivity patterns. The mixed one is not decoration: `_apply_curl`
#: reads `condfac_for` PER COMPONENT (stepping.py:508-537) and a `D_conductivity_diag`
#: with a zero entry leaves one component lossless beside two lossy ones. A kernel
#: that assumes all-three-or-none is wrong there and does not crash.
COND_PATTERNS: Tuple[Tuple[int, int, int], ...] = ((1, 1, 1), (1, 0, 1))

#: Steps in the `multi` leg. STATED, not implied: §16 measured `2d_pml` identical
#: over 60 steps and FIRST DIFFERENT AT STEP 67, so a budget can certify something
#: that diverges just past it. This is the number this gate may claim and no more.
MULTI_STEP_COUNT = 256

#: Steps in the `decay` leg. Longer than `multi` on purpose: the deepest absorber
#: plane's combined per-step factor on the real 2d_cond_pml tables is 0.72131777,
#: which reaches subnormal from 1.0 in 268 steps.
DECAY_STEP_COUNT = 320

#: Steps in the `controls` leg. The DECAY budget, not the multi one: the controls
#: exist to explain divergences that happen as late as step 253, and a control
#: that stops before the thing it explains explains nothing.
CONTROL_STEP_COUNT = DECAY_STEP_COUNT

#: The pattern every long-run divergence has been measured on. Its middle entry
#: is the LOSSLESS component — the one the conductive kernel compiles its
#: conductive branch out for, and the one the `substitution` control replays
#: through the ordinary welded kernel instead.
MIXED_PATTERN: Tuple[int, int, int] = (1, 0, 1)

#: float32's smallest normal. Below this the hardware is in gradual-underflow
#: territory, where a 1-ulp difference is a large relative one.
SMALLEST_NORMAL = float(np.finfo(np.float32).tiny)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The reference — a SECOND transcription of the four cases, not an import
# ---------------------------------------------------------------------------
#
# Transcribed from `stepping._apply_conductive_pml_update` (stepping.py:2001-2109),
# reached from `_apply_curl` (:508-534). The `validate` leg compares it against
# that function itself, so "the kernel matches this" means "the kernel matches the
# array path" rather than "the kernel reproduces whatever was written twice".

def reference_conductive_recurrence(xp: Any, field: Any, fu: Any, f_cond: Any,
                                    curl: Any, condfac: Any, condinv: Any,
                                    kms: Any, sinv: Any, kms_u: Any,
                                    sinv_u: Any) -> None:
    """MEEP's four subchunk cases, in place on ``field``, ``fu`` and ``f_cond``.

    The case table, with the parenthesisation the array path's in-place sequence
    imposes (``km1``/``si1`` on the ``dsig`` axis, ``km2``/``si2`` on ``dsigu``)::

        A  dsig & dsigu   c <- ((c*cf) - cu) * ci
                          u <- (((u*km1) + c_new) - c) * si1
                          f <- (((f*km2) + u_new) - u) * si2
        B  dsigu only     u <- ((u*cf) - cu) * ci
                          f <- (((f*km2) + u_new) - u) * si2      c UNCHANGED
        C  dsig only      c <- ((c*cf) - cu) * ci
                          f <- (((f*km1) + c_new) - c) * si1      u UNCHANGED
        D  neither        f <- ((f*cf) - cu) * ci                 u AND c UNCHANGED

    Written as five whole-volume branch expressions and two nested selections
    because that is the KERNEL's shape — one branch per cell — rather than the
    array path's compute-all-then-``copyto``. The two agree bitwise because the
    branches are mutually independent: nothing a discarded branch computes feeds a
    retained one. That independence is the claim the ``validate`` leg tests.
    """
    dsig = (kms != np.float32(1.0)) | (sinv != np.float32(1.0))
    dsigu = (kms_u != np.float32(1.0)) | (sinv_u != np.float32(1.0))

    c_new = ((f_cond * condfac) - curl) * condinv
    u_cond = ((fu * condfac) - curl) * condinv
    u_split = (((fu * kms) + c_new) - f_cond) * sinv
    u_new = xp.where(dsig, u_split, u_cond)
    f_split = (((field * kms_u) + u_new) - fu) * sinv_u
    f_first = (((field * kms) + c_new) - f_cond) * sinv
    f_direct = ((field * condfac) - curl) * condinv

    new_field = xp.where(dsigu, f_split, xp.where(dsig, f_first, f_direct))
    new_fu = xp.where(dsigu, u_new, fu)
    new_f_cond = xp.where(dsig, c_new, f_cond)
    field[...] = new_field
    fu[...] = new_fu
    f_cond[...] = new_f_cond


def reference_plain_recurrence(field: Any, fu: Any, curl: Any, kms: Any,
                               sinv: Any, kms_u: Any, sinv_u: Any) -> None:
    """The lossless component's recurrence — the shared probe's, unchanged.

    A mixed launch steps its lossless component through ``kernels.pml_curl_step``'s
    two-line recurrence (the kernel compiles the conductive branch out), so the
    reference has to as well or the mixed leg would compare two different physics.
    """
    probe.reference_recurrence(field, fu, curl, kms, sinv, kms_u, sinv_u)


def reference_conductive_pml_step(xp: Any, sources: Dict[str, Any],
                                  targets: Dict[str, Any],
                                  auxiliaries: Dict[str, Any],
                                  histories: Dict[str, Any],
                                  condfac: Dict[str, Any],
                                  condinv: Dict[str, Any],
                                  coefficients: Dict[str, Any], dtdx: Any,
                                  sub_step: str, boundaries: Sequence[str],
                                  cond: Sequence[int],
                                  grouping: str = "array_order") -> None:
    """One conductive curl sub-step, in place, over all three components.

    The curl, the ghost rule and the ownership mask come from the shared probe
    (``probe.reference_curl`` / ``probe.reference_mask``) unchanged: conductivity
    enters ``_apply_curl`` only AFTER the curl is formed, so this sub-step differs
    from the plain one in the recurrence and in nothing else.
    """
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    for index, (target, g1, a1, g2, a2, dsig, dsigu, iyee) in enumerate(terms):
        curl = probe.reference_curl(xp, sources[g1], sources[g2], a1, a2, dtdx,
                                    backward, boundaries, grouping)
        probe.reference_mask(curl, iyee, boundaries)
        if cond[index]:
            reference_conductive_recurrence(
                xp, targets[target], auxiliaries["fu_" + target],
                histories["f_cond_" + target], curl,
                condfac["condfac_" + target], condinv["condinv_" + target],
                coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
                coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])
        else:
            reference_plain_recurrence(
                targets[target], auxiliaries["fu_" + target], curl,
                coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
                coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


# ---------------------------------------------------------------------------
# Coefficient tables and conductivity volumes
# ---------------------------------------------------------------------------

def conductive_coefficients(xp: Any, shape: Tuple[int, int, int],
                            half_integer: bool,
                            active_fraction: float = 0.5) -> Dict[str, Any]:
    """Seeded kms/sinv with an EXACTLY-1.0 inactive run and a graded active one.

    THE SHARED PROBE'S TABLE CANNOT BE USED HERE. ``probe.synthetic_coefficients``
    draws from [0.5, 1.0) and "never touch[es] 1.0" — deliberately, because for
    the PLAIN kernel a table of ones hides the coefficient indexing. For THIS
    kernel the same choice makes ``dsig`` and ``dsigu`` true everywhere, so every
    cell takes case A and three of the four cases are never exercised. The
    inactive run below is exactly 1.0, which is what ``stepping.py:2055-2058``
    tests for, and the active run carries non-representable float32 values so a
    contracted FMA is visible.
    """
    rng = np.random.default_rng(SEED + (31 if half_integer else 32))
    coefficients: Dict[str, Any] = {}
    for axis, name in enumerate(probe.AXIS_NAMES):
        n = shape[axis]
        n_active = max(1, int(round(n * active_fraction)))
        kms = np.ones(n, dtype=np.float32)
        sinv = np.ones(n, dtype=np.float32)
        kms[:n_active] = (0.7 + 0.2 * rng.random(n_active)).astype(np.float32)
        sinv[:n_active] = (0.83 + 0.11 * rng.random(n_active)).astype(np.float32)
        for label, values in (("kms", kms), ("sinv", sinv)):
            coefficients[f"{label}_{name}"] = xp.asarray(np.ascontiguousarray(
                values.reshape(probe.broadcast_shape(axis, n))))
    return coefficients


def near_one_coefficients(xp: Any, shape: Tuple[int, int, int],
                          half_integer: bool) -> Dict[str, Any]:
    """:func:`conductive_coefficients` with one plane per axis a HAIR off 1.0.

    Drives mutation ``tolerance_partition``. On the real tables that mutation is a
    measured no-op — ``kms_x`` has 561 entries exactly 1.0 and 561 within 1e-7 of
    1.0, the same 561 — so a gate run only against real or ordinary synthetic
    tables would report it UNCAUGHT and mean nothing by it. Here one plane sits at
    ``1.0 - 6e-8``, which ``!= 1.0`` calls active and any tolerance calls inactive.
    """
    coefficients = conductive_coefficients(xp, shape, half_integer)
    near = np.float32(1.0) - np.float32(6e-8)
    for axis, name in enumerate(probe.AXIS_NAMES):
        n = shape[axis]
        if n < 3:
            continue
        host = probe.to_host(coefficients[f"kms_{name}"]).copy()
        flat = host.reshape(-1)
        flat[n - 1] = near
        coefficients[f"kms_{name}"] = xp.asarray(np.ascontiguousarray(host))
        host = probe.to_host(coefficients[f"sinv_{name}"]).copy()
        host.reshape(-1)[n - 1] = np.float32(1.0)
        coefficients[f"sinv_{name}"] = xp.asarray(np.ascontiguousarray(host))
    return coefficients


def conductivity_volumes(xp: Any, shape: Tuple[int, int, int],
                         targets: Sequence[str], cond: Sequence[int],
                         dt: float = 0.0125) -> Dict[str, Any]:
    """``condfac``/``condinv`` volumes, derived the way ``Fields`` derives them.

    ``fields.py:821-822`` (MEEP structure.cpp:693-706)::

        condfac = 1 - sigma*dt/2      condinv = 1/(1 + sigma*dt/2)

    Both are FULL VOLUMES of ``grid.shape``, per component, and a graded sigma is
    the realistic case (``mp.Absorber`` ramps one). A component the pattern marks
    lossless gets neither — the plan binds a placeholder for it and the kernel
    compiles the loads away, which is the property the mixed leg measures.
    """
    rng = np.random.default_rng(SEED + 41)
    half_dt = np.float32(dt / 2.0)
    out: Dict[str, Any] = {}
    for index, target in enumerate(targets):
        if not cond[index]:
            continue
        sigma = (0.4 * (0.3 + rng.random(shape))).astype(np.float32)
        out["condfac_" + target] = xp.asarray(np.ascontiguousarray(
            (np.float32(1.0) - sigma * half_dt).astype(np.float32)))
        out["condinv_" + target] = xp.asarray(np.ascontiguousarray(
            (np.float32(1.0) / (np.float32(1.0) + sigma * half_dt)).astype(np.float32)))
    return out


def case_mix(xp: Any, shape: Tuple[int, int, int], coefficients: Dict[str, Any],
             dsig: str, dsigu: str) -> Dict[str, int]:
    """How many cells of this case took each of the four branches.

    RECORDED PER CASE, not asserted. Case A is 1.5% of cells on the real
    ``2d_cond_pml`` layout and ZERO on two of its three curl targets (the
    invariant z axis makes one direction inactive everywhere), so a sweep that
    never says which branches it exercised cannot claim to have gated four cases.
    """
    active = {}
    for label in (dsig, dsigu):
        km = probe.to_host(coefficients["kms_" + label]).reshape(-1)
        si = probe.to_host(coefficients["sinv_" + label]).reshape(-1)
        active[label] = (km != np.float32(1.0)) | (si != np.float32(1.0))
    axis_of = {"x": 0, "y": 1, "z": 2}
    a = np.broadcast_to(active[dsig].reshape(
        probe.broadcast_shape(axis_of[dsig], shape[axis_of[dsig]])), shape)
    b = np.broadcast_to(active[dsigu].reshape(
        probe.broadcast_shape(axis_of[dsigu], shape[axis_of[dsigu]])), shape)
    return {"A": int(np.count_nonzero(a & b)), "B": int(np.count_nonzero(~a & b)),
            "C": int(np.count_nonzero(a & ~b)), "D": int(np.count_nonzero(~a & ~b))}


def subnormal_count(arrays: Dict[str, Any], names: Sequence[str]) -> int:
    """Nonzero floats below float32's smallest normal, across the named volumes."""
    total = 0
    for name in names:
        host = np.abs(probe.to_host(arrays[name]))
        total += int(np.count_nonzero((host > 0.0) & (host < SMALLEST_NORMAL)))
    return total


def component_of(state_name: str) -> str:
    """``fu_Dy`` -> ``Dy``, ``f_cond_Dx`` -> ``Dx``, ``Dy`` -> ``Dy``.

    Which COMPONENT a state array belongs to is the whole question when the
    pattern is mixed: a divergence in the lossless component's ``fu`` and a
    divergence in a conductive component's ``f_cond`` are different findings, and
    a leg that reports only "the state differs" cannot tell them apart.
    """
    return state_name.rsplit("_", 1)[-1]


def divergence_forensics(left: Dict[str, Any], right: Dict[str, Any],
                         names: Sequence[str], left_label: str = "kernel",
                         right_label: str = "reference") -> Dict[str, Any]:
    """WHAT the differing floats are, per array — not only that they differ.

    A first-divergence STEP NUMBER cannot distinguish two findings that matter
    very differently:

    * the two sides compute this recurrence differently (a DEFECT), or
    * the two sides agree about the arithmetic and disagree about what happens
      below float32's smallest normal, because CuPy 13.5.1 appends ``-ftz=true``
      to every NVRTC compile while Triton 3.1.0 keeps subnormals natively.

    The second has a signature the first does not: a differing float that is
    EXACTLY ZERO on one side and SUBNORMAL on the other. So both sides' zero and
    subnormal counts are recorded, plus the first few raw uint32 words, and the
    reading is left to whoever reads the artifact rather than being asserted here.
    """
    out: Dict[str, Any] = {}
    for name in names:
        a = np.ascontiguousarray(probe.to_host(left[name])).ravel()
        b = np.ascontiguousarray(probe.to_host(right[name])).ravel()
        mask = a.view(np.uint32) != b.view(np.uint32)
        differing = int(np.count_nonzero(mask))
        if differing == 0:
            continue
        av = np.abs(a[mask])
        bv = np.abs(b[mask])
        out[name] = {
            "component": component_of(name),
            "differing": differing,
            "total": int(a.size),
            f"{left_label}_zero": int(np.count_nonzero(av == 0.0)),
            f"{left_label}_subnormal": int(np.count_nonzero(
                (av > 0.0) & (av < SMALLEST_NORMAL))),
            f"{right_label}_zero": int(np.count_nonzero(bv == 0.0)),
            f"{right_label}_subnormal": int(np.count_nonzero(
                (bv > 0.0) & (bv < SMALLEST_NORMAL))),
            "flush_signature": int(np.count_nonzero(
                ((av == 0.0) & (bv > 0.0) & (bv < SMALLEST_NORMAL))
                | ((bv == 0.0) & (av > 0.0) & (av < SMALLEST_NORMAL)))),
            "max_abs": {left_label: float(av.max()), right_label: float(bv.max())},
            "min_abs": {left_label: float(av.min()), right_label: float(bv.min())},
            f"{left_label}_words": [f"0x{int(w):08x}"
                                    for w in a.view(np.uint32)[mask][:8]],
            f"{right_label}_words": [f"0x{int(w):08x}"
                                     for w in b.view(np.uint32)[mask][:8]],
        }
    return out


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

TARGETS = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz")}
SOURCES = {"step_B": ("Ex", "Ey", "Ez"), "step_D": ("Hx", "Hy", "Hz")}


def make_arrays(xp: Any, shape: Tuple[int, int, int], sub_step: str,
                cond: Sequence[int], offset: int = 7,
                quiet_sources: bool = False) -> Dict[str, Any]:
    """Seeded device arrays for one sub-step: fields, ``fu``, ``f_cond``, sigma pair.

    The auxiliaries and the histories start NONZERO for the shared probe's reason:
    a zero ``fu`` makes ``fu*kms`` exactly zero whatever ``kms`` is, so a
    mis-indexed coefficient would only show from step two. ``quiet_sources``
    zeroes the six source volumes, which is the decay leg's whole mechanism —
    with no drive the recurrence is a pure product of factors below one.
    """
    rng = np.random.default_rng(SEED + offset)
    targets = TARGETS[sub_step]
    names = tuple(targets) + tuple("fu_" + t for t in targets) + SOURCES[sub_step]
    arrays: Dict[str, Any] = {}
    for name in names:
        host = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if quiet_sources and name in SOURCES[sub_step]:
            host = np.zeros(shape, dtype=np.float32)
        arrays[name] = xp.asarray(np.ascontiguousarray(host))
    for index, target in enumerate(targets):
        if cond[index]:
            arrays["f_cond_" + target] = xp.asarray(np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)))
    arrays.update(conductivity_volumes(xp, shape, targets, cond))
    return arrays


def compare_state(kernel: Dict[str, Any], reference: Dict[str, Any],
                  names: Sequence[str]) -> Dict[str, Any]:
    """Bytewise, over the target AND its ``fu`` AND its ``f_cond``.

    All three are STATE. A kernel that gets ``field`` right and ``f_cond`` wrong
    is correct for exactly one launch and wrong forever after, and the store mask
    that keeps ``f_cond`` unchanged outside the PML is invisible for one step —
    which is mutation ``store_f_cond_everywhere``'s whole shape.
    """
    return combine({name: bit_compare(kernel[name], reference[name])
                    for name in names})


def state_names(sub_step: str, cond: Sequence[int]) -> Tuple[str, ...]:
    targets = TARGETS[sub_step]
    names = list(targets) + ["fu_" + t for t in targets]
    names += ["f_cond_" + t for index, t in enumerate(targets) if cond[index]]
    return tuple(names)


def case_key(source: str, shape, boundaries, dtdx, sub_step, cond) -> str:
    return (f"{source}|{'x'.join(str(v) for v in shape)}|"
            f"{'/'.join(b[0] for b in boundaries)}|dtdx={dtdx!r}|{sub_step}|"
            f"cond={''.join(str(c) for c in cond)}")


def build_coefficients(xp: Any, shape, boundaries, sub_step: str, source: str,
                       repo_root: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The six broadcast coefficient tables, or the reason there are none."""
    half_integer = sub_step == "step_B"
    if source == "synthetic":
        return conductive_coefficients(xp, shape, half_integer), None
    if source == "near_one":
        return near_one_coefficients(xp, shape, half_integer), None
    walled_invariant = [axis for axis in range(3)
                        if shape[axis] == 1 and boundaries[axis] == METALLIC]
    if walled_invariant:
        return None, (f"axis {walled_invariant[0]} has one cell and a metallic "
                      f"wall; Grid refuses that configuration, so no real layer "
                      f"exists for it")
    try:
        _, layer = probe.real_pml_layer(xp, shape, boundaries, repo_root)
    except Exception as exc:  # noqa: BLE001 - the refusal IS the result
        return None, f"{type(exc).__name__}: {exc}"[:400]
    return probe.layer_coefficients(layer, half_integer), None


# ---------------------------------------------------------------------------
# The synthetic sweep
# ---------------------------------------------------------------------------

def one_case(shape, boundaries, dtdx, sub_step, cond, source, repo_root,
             guarded: bool, kernel=None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One launch of one sub-step against the reference, bytewise.

    ``guarded=True`` means the CONTRACTION GUARD IS APPLIED, i.e. the launch
    passes ``enable_fp_fusion=False``. The plan's ``guard=`` keyword is the raw
    ``enable_fp_fusion`` value and reads the other way round, so the negation
    below is the whole of the translation — and it is written here, once, because
    getting it backwards inverts the gate's verdict while leaving both columns
    populated (measured: a first smoke run reported 0/32 "guarded" and 32/32
    "unguarded", which is a passing kernel described as a failing one).
    """
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    case: Dict[str, Any] = {
        "sub_step": sub_step, "shape": list(shape), "boundaries": list(boundaries),
        "dtdx": repr(dtdx),
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "coefficient_source": source, "cond": list(cond),
        "guarded": bool(guarded), "enable_fp_fusion": not bool(guarded),
        "host_mutation": host_mutation,
    }
    coefficients, skipped = build_coefficients(cp, shape, boundaries, sub_step,
                                               source, repo_root)
    if coefficients is None:
        case["skipped"] = skipped
        return case

    kernel_coefficients = coefficients
    if host_mutation == "integer_coefficients_on_B":
        # The D-side (integer) table handed to the B side — a half-cell error, not
        # a crash. Fed to the KERNEL ONLY: swapping it under both legs is not a
        # mutation at all, because the reference would step the same wrong table.
        kernel_coefficients, _ = build_coefficients(
            cp, shape, boundaries, "step_D", source, repo_root)
        if kernel_coefficients is None:
            case["skipped"] = "no integer table for this configuration"
            return case

    targets = TARGETS[sub_step]
    arrays = make_arrays(cp, shape, sub_step, cond)
    names = state_names(sub_step, cond)
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    case["case_mix"] = {
        target: case_mix(cp, shape, coefficients, term[5], term[6])
        for target, term in zip(targets, terms)}

    reference = {name: arrays[name].copy() for name in names}
    reference_conductive_pml_step(
        cp, {name: arrays[name] for name in SOURCES[sub_step]},
        {t: reference[t] for t in targets},
        {"fu_" + t: reference["fu_" + t] for t in targets},
        {"f_cond_" + t: reference["f_cond_" + t]
         for index, t in enumerate(targets) if cond[index]},
        {k: v for k, v in arrays.items() if k.startswith("condfac_")},
        {k: v for k, v in arrays.items() if k.startswith("condinv_")},
        coefficients, np.float32(dtdx), sub_step, boundaries, cond)

    kernel_arrays = dict(arrays)
    for name in names:
        kernel_arrays[name] = arrays[name].copy()
    codes = [np.int32(0 if kind == PERIODIC else 1) for kind in boundaries]
    if host_mutation == "metallic_as_periodic":
        codes = [np.int32(0)] * 3
    plan_cond = list(cond)
    if host_mutation == "all_three_conductive":
        # The over-covering failure in its exact shape: the plan tells the kernel
        # every component is lossy. On a mixed run the lossless one then takes an
        # extra multiply pair and grows a spurious history — smooth, plausible,
        # wrong, and it does not raise because the placeholder pointer is a real
        # allocation. Only meaningful on a mixed pattern.
        plan_cond = [1, 1, 1]
        for index, target in enumerate(targets):
            if not cond[index]:
                kernel_arrays.setdefault("f_cond_" + target,
                                         cp.zeros(shape, dtype=cp.float32))
                extra = conductivity_volumes(cp, shape, (target,), (1,))
                kernel_arrays.update(extra)

    error: Optional[str] = None
    try:
        plan = conductivity.plan_conductive_from_arrays(
            sub_step, kernel_arrays, probe.flatten_coefficients(kernel_coefficients),
            codes, plan_cond, dtdx, kernel=kernel)
        plan.run(guard=not guarded)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        case["verdict"] = {"bit_identical": False, "differing_floats": -1,
                           "total_floats": 0}
        return case
    case["verdict"] = compare_state(kernel_arrays, reference, names)
    return case


def run_synthetic(results: Dict[str, Any], out_path: str, repo_root: str,
                  label: str = "gate", guards: Sequence[bool] = (True, False),
                  shapes=None, sources=("synthetic", "real"),
                  kernel=None, host_mutation: Optional[str] = None,
                  patterns=COND_PATTERNS) -> Dict[str, Any]:
    """The sweep. ``guards`` is (guarded, unguarded) — the pair, always.

    ``True`` means the contraction guard is APPLIED (``enable_fp_fusion=False``).
    The unguarded half is not decoration: without it a guarded N/N is a comparison
    of two identical things, and the pair is what shows the guard doing the work.
    """
    shapes = SHAPES if shapes is None else shapes
    combinations = [
        (source, shape, boundaries, dtdx, sub_step, cond)
        for source in sources
        for shape in shapes
        for boundaries in BOUNDARY_SETS
        for dtdx in DTDX
        for sub_step in ("step_B", "step_D")
        for cond in patterns
    ]
    cases: List[Dict[str, Any]] = []
    total = len(combinations) * len(guards)
    index = 0
    for guarded in guards:
        for source, shape, boundaries, dtdx, sub_step, cond in combinations:
            index += 1
            started = time.time()
            case = one_case(shape, boundaries, dtdx, sub_step, cond, source,
                            repo_root, guarded, kernel=kernel,
                            host_mutation=host_mutation)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            key = case_key(source, shape, boundaries, dtdx, sub_step, cond)
            if case.get("skipped"):
                log(f"[{label}] {index}/{total} guarded={guarded} {key}: "
                    f"SKIPPED ({case['skipped']})")
            else:
                verdict = case["verdict"]
                log(f"[{label}] {index}/{total} guarded={guarded} {key}: "
                    f"identical={verdict['bit_identical']} "
                    f"(differing={verdict['differing_floats']}/"
                    f"{verdict['total_floats']}, "
                    f"maxulp={verdict.get('max_ulp', 0)}) ({case['seconds']} s)")
            results.setdefault("synthetic", {})[label] = summarize(cases)
            results["synthetic"][label]["cases"] = cases
            save(results, out_path)
    summary = summarize(cases)
    summary["cases"] = cases
    results.setdefault("synthetic", {})[label] = summary
    save(results, out_path)
    log(f"[{label}] guarded {summary['guarded_identical']}/{summary['guarded_ran']} "
        f"| unguarded {summary['unguarded_identical']}/{summary['unguarded_ran']} "
        f"| skipped {summary['skipped']}")
    return summary


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """N/N per guard, never the word 'matches'; plus which branches were exercised."""
    out = {"guarded_ran": 0, "guarded_identical": 0,
           "unguarded_ran": 0, "unguarded_identical": 0, "skipped": 0,
           "branches_exercised": {"A": 0, "B": 0, "C": 0, "D": 0}}
    for case in cases:
        if case.get("skipped"):
            out["skipped"] += 1
            continue
        key = "guarded" if case["guarded"] else "unguarded"
        out[key + "_ran"] += 1
        out[key + "_identical"] += int(case["verdict"]["bit_identical"])
        for mix in (case.get("case_mix") or {}).values():
            for branch, count in mix.items():
                if count:
                    out["branches_exercised"][branch] += 1
    out["guarded_pass"] = bool(out["guarded_ran"]) and \
        out["guarded_identical"] == out["guarded_ran"]
    out["unguarded_control_bites"] = out["unguarded_identical"] == 0
    return out


# ---------------------------------------------------------------------------
# Multi-step and decay
# ---------------------------------------------------------------------------

def run_multi(results: Dict[str, Any], out_path: str, steps: int,
              quiet_sources: bool, label: str) -> Dict[str, Any]:
    """Consecutive sub-steps, byte-compared at EVERY step, with the budget stated.

    ``quiet_sources`` is the decay leg: with the drive removed the recurrence is a
    product of per-step factors below one and the state walks into the subnormal
    range, which is where §16's amplification lives. With the drive present the
    state reaches a fixed point instead and never underflows — so the two legs
    measure different things and both are run.
    """
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    boundaries = BOUNDARY_SETS[3]  # the asymmetric one
    dtdx = 0.35
    runs: List[Dict[str, Any]] = []
    for shape in SHAPES:
        for sub_step in ("step_B", "step_D"):
            for cond in COND_PATTERNS:
                coefficients = conductive_coefficients(cp, shape,
                                                       sub_step == "step_B")
                flat = probe.flatten_coefficients(coefficients)
                arrays = make_arrays(cp, shape, sub_step, cond, offset=9,
                                     quiet_sources=quiet_sources)
                names = state_names(sub_step, cond)
                targets = TARGETS[sub_step]
                reference = {name: arrays[name].copy() for name in names}
                kernel_state = dict(arrays)
                for name in names:
                    kernel_state[name] = arrays[name].copy()
                codes = [np.int32(0 if kind == PERIODIC else 1)
                         for kind in boundaries]
                plan = conductivity.plan_conductive_from_arrays(
                    sub_step, kernel_state, flat, codes, cond, dtdx)
                sources = {name: arrays[name] for name in SOURCES[sub_step]}
                identical_steps = 0
                first_divergence: Optional[int] = None
                forensics: Dict[str, Any] = {}
                # WHICH ARRAY first differs, not only WHEN. Under a mixed pattern
                # a divergence in the lossless component's `fu` and one in a
                # conductive component's `f_cond` carry opposite verdicts, and a
                # run that records one step number for the whole state cannot be
                # asked which happened after the fact.
                component_first: Dict[str, int] = {}
                subnormals: List[Tuple[int, int]] = []
                for step in range(1, steps + 1):
                    reference_conductive_pml_step(
                        cp, sources, {t: reference[t] for t in targets},
                        {"fu_" + t: reference["fu_" + t] for t in targets},
                        {"f_cond_" + t: reference["f_cond_" + t]
                         for index, t in enumerate(targets) if cond[index]},
                        {k: v for k, v in arrays.items() if k.startswith("condfac_")},
                        {k: v for k, v in arrays.items() if k.startswith("condinv_")},
                        coefficients, np.float32(dtdx), sub_step, boundaries, cond)
                    plan.run()
                    cp.cuda.runtime.deviceSynchronize()
                    verdict = compare_state(kernel_state, reference, names)
                    if verdict["bit_identical"]:
                        identical_steps += 1
                    else:
                        for state_name, part in verdict["per_component"].items():
                            if (not part["bit_identical"]
                                    and state_name not in component_first):
                                component_first[state_name] = step
                        if first_divergence is None:
                            first_divergence = step
                            forensics = divergence_forensics(
                                kernel_state, reference, names)
                    if step % 32 == 0 or step == steps:
                        subnormals.append((step, subnormal_count(reference, names)))
                conductive = [t for index, t in enumerate(targets) if cond[index]]
                lossless = [t for index, t in enumerate(targets) if not cond[index]]
                run = {"shape": list(shape), "sub_step": sub_step,
                       "cond": list(cond), "steps": steps,
                       "identical_steps": identical_steps,
                       "first_divergence": first_divergence,
                       "component_first_divergence": dict(component_first),
                       "first_divergence_forensics": forensics,
                       "conductive_components": conductive,
                       "lossless_components": lossless,
                       # THE CLAUSE THE ATTRIBUTION IS BUILT ON. A conductive
                       # component that never differs is the conductive
                       # recurrence — this kernel's whole subject — agreeing with
                       # the array path for the entire budget.
                       "conductive_component_diverged": any(
                           component_of(name) in conductive
                           for name in component_first),
                       "subnormal_cells": [list(entry) for entry in subnormals]}
                runs.append(run)
                where = ",".join(f"{n}@{s}" for n, s in
                                 sorted(component_first.items(), key=lambda kv: kv[1]))
                log(f"[{label}] shape={shape} {sub_step} cond={cond} {steps} steps: "
                    f"identical={identical_steps}/{steps}"
                    + (f" FIRST DIVERGENCE at step {first_divergence} in {where}"
                       f" conductive_component_diverged="
                       f"{run['conductive_component_diverged']}"
                       if first_divergence is not None else "")
                    + f" subnormals={subnormals[-1][1] if subnormals else 0}")
                results[label] = _multi_summary(runs, steps)
                save(results, out_path)
    summary = _multi_summary(runs, steps)
    results[label] = summary
    save(results, out_path)
    log(f"[{label}] {summary['clean_runs']}/{len(runs)} runs clean for the whole "
        f"budget; uniform-conductive clean={summary['uniform_conductive_clean']}; "
        f"any conductive component diverged={summary['conductive_component_diverged']}")
    return summary


def _multi_summary(runs: List[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    """The leg's numbers, with the two facts the attribution needs kept apart.

    ``all_steps_identical`` folds together "the conductive recurrence disagreed"
    and "the lossless component's last bit disagreed", which are the two readings
    the controls leg exists to separate — so both are reported here as well, and
    neither is derived from the other.
    """
    uniform = [r for r in runs if all(r["cond"])]
    return {
        "runs": runs, "steps": steps,
        "all_steps_identical": all(r["identical_steps"] == steps for r in runs),
        "clean_runs": sum(r["identical_steps"] == steps for r in runs),
        # Every row whose pattern is all-conductive: the conductive recurrence
        # driven for the whole budget with nothing lossless in the launch.
        "uniform_conductive_clean": bool(uniform) and all(
            r["identical_steps"] == steps for r in uniform),
        "uniform_conductive_runs": len(uniform),
        "conductive_component_diverged": any(
            r["conductive_component_diverged"] for r in runs),
    }


# ---------------------------------------------------------------------------
# The `controls` leg — without which the two legs above report a number and
# nothing else
# ---------------------------------------------------------------------------
#
# THE PRECEDENT THIS TRANSCRIBES. gate_triton_cylindrical.py faced the same shape:
# a single-launch sweep bit-identical everywhere, and a consecutive-step leg that
# first differed at step 23-72 on a single subnormal auxiliary float. It resolved
# it not by arguing but by adding two rows in which TRITON DOES NOT APPEAR — the
# array path against itself (`null`, must be clean) and the array path against
# itself with one mantissa bit flipped (`one_ulp`, must diverge) — and then
# declaring its long-run number a BUDGET on the strength of those two. Its own
# words: "a long-run divergence is evidence about the kernel only when the
# controls leg says the array path is otherwise deterministic and that this
# configuration amplifies". Both are carried over here unchanged in role.
#
# WHAT THIS GATE NEEDS THAT CYLINDRICAL DID NOT. Cylindrical's divergence was in
# the one kernel under test. Here it is not: every diverging row is the MIXED
# pattern, and the 2026-08-10 diagnostic located the differing floats in the
# LOSSLESS component's `fu` — an array the conductive kernel fills through the
# ordinary two-line recurrence with its conductive branch compiled out. That is a
# claim about which code produced the bytes, and it is checkable rather than
# arguable, because the three curl targets are INDEPENDENT in these legs: the six
# source volumes are read and never written, so component 1's trajectory is a
# function of component 1's own state alone. So the third control SUBSTITUTES the
# separately welded ordinary kernel for the conductive one on the same bytes and
# asks whether the divergence follows the kernel or follows the arithmetic.

def run_controls(results: Dict[str, Any], out_path: str, steps: int,
                 shapes: Optional[Sequence[Tuple[int, int, int]]] = None,
                 label: str = "controls") -> Dict[str, Any]:
    """``null`` and ``one_ulp`` (Triton nowhere) plus the ``substitution`` row."""
    shapes = SHAPES if shapes is None else shapes
    rows: List[Dict[str, Any]] = []

    # The oracle-only pair runs on the two shapes whose divergence steps bracket
    # the measured range (earliest and latest), under both drive modes. Running
    # it on all four would measure the same property four times; running it on
    # none is what made the earlier artifact unattributable.
    for shape in (shapes[1], shapes[3]) if len(shapes) >= 4 else shapes:
        for sub_step in ("step_B", "step_D"):
            for quiet in (False, True):
                for control in ("null", "one_ulp"):
                    row = one_oracle_control(shape, sub_step, quiet, steps,
                                             control)
                    rows.append(row)
                    log(f"[{label}] {control} shape={shape} {sub_step} "
                        f"{'decay' if quiet else 'driven'} {steps} steps: "
                        f"first_divergence={row['first_divergence']}"
                        + (f" seed_flip={row['seed_flip']}"
                           if row.get("seed_flip") else ""))
                    results[label] = {"rows": rows, "steps": steps}
                    save(results, out_path)

    for shape in shapes:
        for sub_step in ("step_B", "step_D"):
            for quiet in (False, True):
                row = one_substitution_control(shape, sub_step, quiet, steps)
                rows.append(row)
                log(f"[{label}] substitution shape={shape} {sub_step} "
                    f"{'decay' if quiet else 'driven'} {steps} steps: "
                    f"conductive_plan_first={row['conductive_vs_oracle']} "
                    f"plain_plan_first={row['plain_vs_oracle']} "
                    f"kernels_agree_on_lossless={row['kernels_agree_on_lossless']} "
                    f"oracles_agree_on_lossless={row['oracles_agree_on_lossless']} "
                    f"conductive_component_diverged="
                    f"{row['conductive_component_diverged']}")
                results[label] = {"rows": rows, "steps": steps}
                save(results, out_path)

    summary = summarize_controls(rows, steps)
    results[label] = summary
    save(results, out_path)
    log(f"[{label}] null {summary['null_identical']}/{summary['null_total']} clean | "
        f"one_ulp {summary['one_ulp_diverged']}/{summary['one_ulp_total']} amplified | "
        f"substitution: kernels agree on the lossless component "
        f"{summary['substitution_kernels_agree']}/{summary['substitution_total']}, "
        f"plain kernel reproduces the divergence "
        f"{summary['substitution_plain_reproduces']}/"
        f"{summary['substitution_diverging']}")
    return summary


def summarize_controls(rows: Sequence[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    """Counts only. The reading is :func:`attribute_long_run`'s job, not this one."""
    nulls = [r for r in rows if r.get("control") == "null"]
    ulps = [r for r in rows if r.get("control") == "one_ulp"]
    subs = [r for r in rows if r.get("control") == "substitution"]
    diverging = [r for r in subs if r["conductive_vs_oracle"] is not None]
    return {
        "rows": list(rows), "steps": steps,
        "null_identical": sum(r["first_divergence"] is None for r in nulls),
        "null_total": len(nulls),
        "one_ulp_diverged": sum(r["first_divergence"] is not None for r in ulps),
        "one_ulp_total": len(ulps),
        "substitution_total": len(subs),
        "substitution_diverging": len(diverging),
        "substitution_kernels_agree": sum(bool(r["kernels_agree_on_lossless"])
                                          for r in subs),
        "substitution_oracles_agree": sum(bool(r["oracles_agree_on_lossless"])
                                          for r in subs),
        # The plain kernel is a DIFFERENT kernel with its own weld. "Reproduces"
        # is deliberately strict: the same step AND the same state array, not
        # merely "also diverged somewhere eventually".
        "substitution_plain_reproduces": sum(bool(r["plain_reproduces"])
                                             for r in diverging),
        "substitution_conductive_component_diverged": any(
            r["conductive_component_diverged"] for r in subs),
    }


def _control_setup(shape, sub_step: str, quiet: bool):
    """The one configuration every control row is run in — built in ONE place.

    ``BOUNDARY_SETS[3]`` (the asymmetric set) and ``dtdx = 0.35`` are not choices
    made here: they are exactly what :func:`run_multi` uses, and a control run in
    a configuration the diverging leg was not run in would answer a different
    question. Same seeds (``offset=9``), same tables, same drive mode.
    """
    boundaries = BOUNDARY_SETS[3]
    dtdx = 0.35
    coefficients = conductive_coefficients(cp, shape, sub_step == "step_B")
    arrays = make_arrays(cp, shape, sub_step, MIXED_PATTERN, offset=9,
                         quiet_sources=quiet)
    return boundaries, dtdx, coefficients, arrays


def _step_oracle(state: Dict[str, Any], arrays: Dict[str, Any],
                 coefficients: Dict[str, Any], dtdx: float, sub_step: str,
                 boundaries, cond) -> None:
    """One conductive oracle sub-step in place on ``state``. CuPy only."""
    targets = TARGETS[sub_step]
    reference_conductive_pml_step(
        cp, {name: arrays[name] for name in SOURCES[sub_step]},
        {t: state[t] for t in targets},
        {"fu_" + t: state["fu_" + t] for t in targets},
        {"f_cond_" + t: state["f_cond_" + t]
         for index, t in enumerate(targets) if cond[index]},
        {k: v for k, v in arrays.items() if k.startswith("condfac_")},
        {k: v for k, v in arrays.items() if k.startswith("condinv_")},
        coefficients, np.float32(dtdx), sub_step, boundaries, cond)


def _flip_one_ulp(volume: Any) -> Optional[Dict[str, Any]]:
    """Flip the low mantissa bit of ONE NORMAL float in ``volume``, in place.

    NORMAL is required, and is not a detail: flipping the low bit of a value that
    is already subnormal asks a question about the flush boundary rather than
    about amplification, which is the other half of what this file is trying to
    tell apart. Returns ``None`` when the volume holds no normal float at all, so
    a row that could not be seeded says so instead of reporting a clean control.
    """
    flat = volume.reshape(-1)
    host = probe.to_host(flat).view(np.uint32)
    exponents = (host >> 23) & 0xFF
    normal = np.flatnonzero((exponents != 0) & (exponents != 0xFF))
    if not normal.size:
        return None
    index = int(normal[normal.size // 2])
    flipped = np.uint32(host[index] ^ np.uint32(1))
    flat[index] = cp.asarray(
        np.array([flipped], dtype=np.uint32).view(np.float32))[0]
    return {"flat_index": index, "before": f"0x{int(host[index]):08x}",
            "after": f"0x{int(flipped):08x}"}


def one_oracle_control(shape, sub_step: str, quiet: bool, steps: int,
                       control: str) -> Dict[str, Any]:
    """Oracle against oracle. The Triton kernel is NOT CONSTRUCTED in this row.

    ``null`` must stay identical for the whole budget — it is the statement that
    this harness's comparison is quiet and the CuPy oracle is deterministic, and
    without it every step number in ``multi`` and ``decay`` could be reduction
    noise. ``one_ulp`` must diverge — it is the statement that this configuration
    amplifies a last-bit disagreement within the budget, and without it a
    divergence at step 160 could not be distinguished from a kernel that is
    simply wrong.
    """
    cond = MIXED_PATTERN
    boundaries, dtdx, coefficients, arrays = _control_setup(shape, sub_step, quiet)
    names = state_names(sub_step, cond)
    targets = TARGETS[sub_step]
    lossless = targets[cond.index(0)]
    left = {name: arrays[name].copy() for name in names}
    right = {name: arrays[name].copy() for name in names}

    row: Dict[str, Any] = {
        "control": control, "shape": list(shape), "sub_step": sub_step,
        "cond": list(cond), "drive": "decay" if quiet else "driven",
        "steps": steps, "dtdx": repr(dtdx),
        "boundaries": list(boundaries), "triton_in_this_row": False,
        "first_divergence": None, "seed_flip": None,
        "component_first_divergence": {}, "first_divergence_forensics": {},
    }
    for step in range(1, steps + 1):
        for state in (left, right):
            _step_oracle(state, arrays, coefficients, dtdx, sub_step, boundaries,
                         cond)
        if control == "one_ulp" and step == 1:
            row["seed_flip"] = _flip_one_ulp(right["fu_" + lossless])
            if row["seed_flip"] is not None:
                row["seed_flip"]["array"] = "fu_" + lossless
            else:
                row["seed_flip_failed"] = (
                    f"fu_{lossless} held no normal float after one step; this row "
                    f"cannot claim amplification")
            continue
        verdict = compare_state(left, right, names)
        if not verdict["bit_identical"]:
            for state_name, part in verdict["per_component"].items():
                if (not part["bit_identical"]
                        and state_name not in row["component_first_divergence"]):
                    row["component_first_divergence"][state_name] = step
            if row["first_divergence"] is None:
                row["first_divergence"] = step
                row["first_divergence_forensics"] = divergence_forensics(
                    left, right, names, "left", "right")
    cp.cuda.runtime.deviceSynchronize()
    return row


def one_substitution_control(shape, sub_step: str, quiet: bool,
                             steps: int) -> Dict[str, Any]:
    """The discriminator: does the divergence follow THIS KERNEL, or the arithmetic?

    Four states are stepped side by side from ONE set of seeded bytes:

    ``conductive_kernel``  ``plan_conductive_from_arrays`` at ``cond = (1,0,1)``
    ``conductive_oracle``  :func:`reference_conductive_pml_step` at the same cond
    ``plain_kernel``       ``launch.plan_from_arrays`` — the ORDINARY PML curl,
                           a different kernel with its own weld, all lossless
    ``plain_oracle``       ``probe.reference_pml_step``

    and four comparisons are taken at EVERY step. Two are the ordinary
    kernel-against-oracle ones. The other two are what make this a control:

    * ``kernels_agree_on_lossless`` — the conductive plan's lossless ``field`` and
      ``fu`` against the plain plan's, bytewise. These are two DIFFERENT kernels
      asked to compute the same recurrence on the same bytes. If they agree for
      the whole budget, the conductive kernel's lossless path IS the ordinary
      path numerically, and a divergence there is not this kernel's to own.
    * ``oracles_agree_on_lossless`` — the same check on the two oracle sides,
      which must hold by construction (``reference_conductive_pml_step`` routes a
      lossless component into ``probe.reference_recurrence``, which is exactly
      what ``probe.reference_pml_step`` calls). It is measured anyway: it is the
      row's own self-check, and if it ever fails the other three comparisons are
      comparing two different physics and mean nothing.
    """
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    cond = MIXED_PATTERN
    boundaries, dtdx, coefficients, arrays = _control_setup(shape, sub_step, quiet)
    flat = probe.flatten_coefficients(coefficients)
    codes = [np.int32(0 if kind == PERIODIC else 1) for kind in boundaries]
    targets = TARGETS[sub_step]
    lossless = targets[cond.index(0)]
    conductive = [t for index, t in enumerate(targets) if cond[index]]
    cond_names = state_names(sub_step, cond)
    plain_names = tuple(targets) + tuple("fu_" + t for t in targets)
    lossless_names = (lossless, "fu_" + lossless)

    conductive_oracle = {name: arrays[name].copy() for name in cond_names}
    plain_oracle = {name: arrays[name].copy() for name in plain_names}
    conductive_kernel = dict(arrays)
    for name in cond_names:
        conductive_kernel[name] = arrays[name].copy()
    plain_kernel = dict(arrays)
    for name in plain_names:
        plain_kernel[name] = arrays[name].copy()

    conductive_plan = conductivity.plan_conductive_from_arrays(
        sub_step, conductive_kernel, flat, codes, cond, dtdx)
    # The ordinary welded curl, ALL LOSSLESS. `cond` is not passed because this
    # plan has no notion of it: that is the point of the substitution.
    plain_plan = launch.plan_from_arrays(sub_step, plain_kernel, flat, codes, dtdx)
    sources = {name: arrays[name] for name in SOURCES[sub_step]}

    row: Dict[str, Any] = {
        "control": "substitution", "shape": list(shape), "sub_step": sub_step,
        "cond": list(cond), "drive": "decay" if quiet else "driven",
        "steps": steps, "dtdx": repr(dtdx), "boundaries": list(boundaries),
        "lossless_component": lossless, "conductive_components": conductive,
        "conductive_vs_oracle": None, "plain_vs_oracle": None,
        "kernels_agree_on_lossless": True, "oracles_agree_on_lossless": True,
        "kernels_first_disagreement": None, "oracles_first_disagreement": None,
        "conductive_component_first_divergence": {},
        "conductive_first_divergence_components": {},
        "plain_first_divergence_components": {},
        "first_divergence_forensics": {},
        "plain_first_divergence_forensics": {},
    }
    for step in range(1, steps + 1):
        _step_oracle(conductive_oracle, arrays, coefficients, dtdx, sub_step,
                     boundaries, cond)
        conductive_plan.run()
        probe.reference_pml_step(
            cp, sources, {t: plain_oracle[t] for t in targets},
            {"fu_" + t: plain_oracle["fu_" + t] for t in targets},
            coefficients, np.float32(dtdx), sub_step, boundaries, "array_order")
        plain_plan.run()
        cp.cuda.runtime.deviceSynchronize()

        conductive_verdict = compare_state(conductive_kernel, conductive_oracle,
                                           cond_names)
        if not conductive_verdict["bit_identical"]:
            for name, part in conductive_verdict["per_component"].items():
                if (not part["bit_identical"]
                        and name not in row["conductive_first_divergence_components"]):
                    row["conductive_first_divergence_components"][name] = step
                    if component_of(name) in conductive:
                        row["conductive_component_first_divergence"][name] = step
            if row["conductive_vs_oracle"] is None:
                row["conductive_vs_oracle"] = step
                row["first_divergence_forensics"] = divergence_forensics(
                    conductive_kernel, conductive_oracle, cond_names)

        plain_verdict = compare_state(plain_kernel, plain_oracle, plain_names)
        if not plain_verdict["bit_identical"]:
            for name, part in plain_verdict["per_component"].items():
                if (not part["bit_identical"]
                        and name not in row["plain_first_divergence_components"]):
                    row["plain_first_divergence_components"][name] = step
            if row["plain_vs_oracle"] is None:
                row["plain_vs_oracle"] = step
                row["plain_first_divergence_forensics"] = divergence_forensics(
                    plain_kernel, plain_oracle, plain_names)

        cross_kernels = compare_state(conductive_kernel, plain_kernel,
                                      lossless_names)
        if not cross_kernels["bit_identical"] and row["kernels_agree_on_lossless"]:
            row["kernels_agree_on_lossless"] = False
            row["kernels_first_disagreement"] = step
            row["kernels_disagreement_forensics"] = divergence_forensics(
                conductive_kernel, plain_kernel, lossless_names,
                "conductive_kernel", "plain_kernel")
        cross_oracles = compare_state(conductive_oracle, plain_oracle,
                                      lossless_names)
        if not cross_oracles["bit_identical"] and row["oracles_agree_on_lossless"]:
            row["oracles_agree_on_lossless"] = False
            row["oracles_first_disagreement"] = step

    # STRICT: the same step AND the same state array. "The plain kernel also
    # diverged eventually" would be satisfied by any two runs that both end up
    # wrong, and would attribute nothing.
    conductive_lossless_steps = {
        name: step for name, step
        in row["conductive_first_divergence_components"].items()
        if component_of(name) == lossless}
    plain_lossless_steps = {
        name: step for name, step
        in row["plain_first_divergence_components"].items()
        if component_of(name) == lossless}
    row["lossless_first_divergence"] = {
        "conductive_plan": conductive_lossless_steps,
        "plain_plan": plain_lossless_steps}
    row["plain_reproduces"] = bool(
        conductive_lossless_steps
        and conductive_lossless_steps == plain_lossless_steps)
    row["conductive_component_diverged"] = bool(
        row["conductive_component_first_divergence"])
    return row


# ---------------------------------------------------------------------------
# The `validate` leg — this file's reference against stepping.py, on NumPy
# ---------------------------------------------------------------------------

def run_validate(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """The reference against ``stepping._apply_conductive_pml_update``, bytewise.

    Runs on any host, GPU or not. Without this leg the gate would only say the
    kernel reproduces this file; with it, the chain reaches the array path.
    """
    from meep_gpu.stepping import _apply_conductive_pml_update  # noqa: PLC0415

    cases: List[Dict[str, Any]] = []
    for shape in ((7, 5, 3), (13, 11, 9), (9, 8, 1), (5, 5, 5)):
        for dtdx in DTDX:
            for target_index, (a1, a2) in enumerate(
                    ((1, 2), (2, 0), (0, 1))):
                for table in ("synthetic", "near_one"):
                    builder = (conductive_coefficients if table == "synthetic"
                               else near_one_coefficients)
                    coefficients = builder(np, shape, target_index == 0)
                    axis_names = probe.AXIS_NAMES
                    kms = coefficients["kms_" + axis_names[a1]]
                    sinv = coefficients["sinv_" + axis_names[a1]]
                    kms_u = coefficients["kms_" + axis_names[a2]]
                    sinv_u = coefficients["sinv_" + axis_names[a2]]
                    rng = np.random.default_rng(SEED + 3 * target_index)
                    volume = lambda: (rng.standard_normal(shape) * 0.5  # noqa: E731
                                      ).astype(np.float32)
                    field, fu, f_cond = volume(), volume(), volume()
                    curl = (volume() * np.float32(dtdx)).astype(np.float32)
                    sigma = (0.4 * (0.3 + rng.random(shape))).astype(np.float32)
                    half_dt = np.float32(0.0125 / 2.0)
                    condfac = (np.float32(1.0) - sigma * half_dt).astype(np.float32)
                    condinv = (np.float32(1.0) /
                               (np.float32(1.0) + sigma * half_dt)).astype(np.float32)

                    got = {"field": field.copy(), "fu": fu.copy(),
                           "f_cond": f_cond.copy()}
                    _apply_conductive_pml_update(
                        np, got["field"], curl, condfac, condinv, kms, sinv,
                        kms_u, sinv_u, got["fu"], got["f_cond"], scratch=None)
                    want = {"field": field.copy(), "fu": fu.copy(),
                            "f_cond": f_cond.copy()}
                    reference_conductive_recurrence(
                        np, want["field"], want["fu"], want["f_cond"], curl,
                        condfac, condinv, kms, sinv, kms_u, sinv_u)
                    detail = {name: {
                        "bit_identical": got[name].tobytes() == want[name].tobytes(),
                        "differing_floats": int(np.count_nonzero(
                            got[name].view(np.uint32) != want[name].view(np.uint32))),
                    } for name in ("field", "fu", "f_cond")}
                    identical = all(v["bit_identical"] for v in detail.values())
                    mix = case_mix(np, shape, coefficients,
                                   probe.AXIS_NAMES[a1], probe.AXIS_NAMES[a2])
                    cases.append({"shape": list(shape), "dtdx": repr(dtdx),
                                  "target": target_index, "table": table,
                                  "bit_identical": identical, "detail": detail,
                                  "case_mix": mix})
                    log(f"[validate] shape={shape} dtdx={dtdx} target={target_index} "
                        f"table={table}: identical={identical} mix={mix}")
                    results["validate"] = {
                        "cases": cases,
                        "ran": len(cases),
                        "identical": sum(int(c["bit_identical"]) for c in cases)}
                    save(results, out_path)
    summary = {"cases": cases, "ran": len(cases),
               "identical": sum(int(c["bit_identical"]) for c in cases)}
    summary["pass"] = summary["ran"] == summary["identical"]
    results["validate"] = summary
    save(results, out_path)
    log(f"[validate] reference vs stepping.py: "
        f"{summary['identical']}/{summary['ran']} bit-identical")
    return summary


# ---------------------------------------------------------------------------
# Mutations — each defect must change the answer
# ---------------------------------------------------------------------------

def flatten_split_grouping(source: str) -> Tuple[str, int]:
    """m1 ``(((f*km2) + u_new) - u)`` -> ``((f*km2) + (u_new - u))``.

    The associativity family, and the one the recon already measured on NumPy
    alone: 0/18 identical.
    """
    needle = "f_split = (((f * km2) + u_new) - u) * si2"
    replacement = "f_split = ((f * km2) + (u_new - u)) * si2"
    return source.replace(needle, replacement), source.count(needle)


def swap_condfac_condinv(source: str) -> Tuple[str, int]:
    """m2 ``condfac`` and ``condinv`` exchanged. Both are near 1, so it is smooth."""
    hits = source.count("* cf) - curl) * ci")
    return source.replace("* cf) - curl) * ci", "* ci) - curl) * cf"), hits


def tolerance_partition(source: str) -> Tuple[str, int]:
    """m3 ``!= 1.0`` becomes a tolerance. A MEASURED NO-OP on the real tables.

    ``kms_x`` on the real ``2d_cond_pml`` layout has 561 entries exactly 1.0 and
    561 within 1e-7 of 1.0 — the same 561 — so this defect is invisible there and
    a gate that only ran real or ordinary synthetic tables would report it
    uncaught and mean nothing. It is driven against ``near_one`` tables, which
    carry one plane at ``1.0 - 6e-8``.
    """
    hits = source.count("!= 1.0)")
    mutated = source.replace(
        "dsig = (km1 != 1.0) | (si1 != 1.0)",
        "dsig = (tl.abs(km1 - 1.0) > 1e-7) | (tl.abs(si1 - 1.0) > 1e-7)")
    mutated = mutated.replace(
        "dsigu = (km2 != 1.0) | (si2 != 1.0)",
        "dsigu = (tl.abs(km2 - 1.0) > 1e-7) | (tl.abs(si2 - 1.0) > 1e-7)")
    return mutated, hits


def case_b_as_case_a(source: str) -> Tuple[str, int]:
    """m4 case B collapsed into case A — the substitution the array path forbids.

    ``u_cond`` becomes the split recurrence with identity coefficients instead of
    conductivity applied to ``f_u`` directly. That is exactly the equivalence
    ``_apply_conductive_pml_update``'s docstring denies (stepping.py:2044-2046).
    """
    needle = "u_cond = ((u * cf) - curl) * ci"
    replacement = "u_cond = (((u * 1.0) + c_new) - c) * 1.0"
    return source.replace(needle, replacement), source.count(needle)


def store_f_cond_everywhere(source: str) -> Tuple[str, int]:
    """m5 the ``dsig`` store mask dropped: the history is written in the interior.

    Invisible for one launch on ``field``; it corrupts ``f_cond`` immediately and
    ``field`` from the step after. The single-launch leg catches it only because
    ``f_cond`` is compared — which is the point of comparing it.
    """
    needle = "tl.store(c_ptr + idx, c_new, mask=live & dsig)"
    replacement = "tl.store(c_ptr + idx, c_new, mask=live)"
    return source.replace(needle, replacement), source.count(needle)


def store_fu_everywhere(source: str) -> Tuple[str, int]:
    """m5b the ``dsigu`` store mask dropped — the same defect on the auxiliary."""
    needle = "tl.store(u_ptr + idx, u_new, mask=live & dsigu)"
    replacement = "tl.store(u_ptr + idx, u_new, mask=live)"
    return source.replace(needle, replacement), source.count(needle)


def swap_dsig_dsigu(source: str) -> Tuple[str, int]:
    """m7 the recurrence reads dsigu's coefficients where dsig's belong.

    The needle carries NO indentation on purpose. ``inspect.getsource`` plus
    ``textwrap.dedent`` re-indents a nested function's continuation lines by an
    amount that depends on how deeply the ``def`` was nested, so a needle spelled
    with the file's own leading whitespace misses — measured, first run: this leg
    reported "NEEDLE MISSED" while every other mutation was caught, which is a leg
    that measures nothing wearing the same shape as one that passed.
    """
    needle = "km_y, si_y, km_z, si_z, COND0)"
    replacement = "km_z, si_z, km_y, si_y, COND0)"
    return source.replace(needle, replacement), source.count(needle)


def regroup_stencil(source: str) -> Tuple[str, int]:
    """m8 ``dtdx*((a-b)+(c-d))`` -> ``((a-b)+c)-d`` — the shared curl's own defect."""
    hits = 0
    mutated = source
    for a, b, c, d in (("c_y", "c", "b", "b_z"), ("a_z", "a", "c", "c_x"),
                       ("b_x", "b", "a", "a_y")):
        needle = f"dtdx * (({a} - {b}) + ({c} - {d}))"
        replacement = f"dtdx * ((({a} - {b}) + {c}) - {d})"
        hits += mutated.count(needle)
        mutated = mutated.replace(needle, replacement)
    return mutated, hits


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "flatten_split_grouping": flatten_split_grouping,
    "swap_condfac_condinv": swap_condfac_condinv,
    "case_b_as_case_a": case_b_as_case_a,
    "store_f_cond_everywhere": store_f_cond_everywhere,
    "store_fu_everywhere": store_fu_everywhere,
    "swap_dsig_dsigu": swap_dsig_dsigu,
    "regroup_stencil": regroup_stencil,
}

#: Driven against the near-one table only; see :func:`tolerance_partition`.
NEAR_ONE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "tolerance_partition": tolerance_partition,
}

#: Host-side defects: the kernel is the shipped one, the PLAN lies to it.
HOST_MUTATIONS = ("integer_coefficients_on_B", "metallic_as_periodic",
                  "all_three_conductive")

_TEMPORARY: List[str] = []


def compile_mutated(source: str):
    """Compile a mutated copy of the shipped kernel PAIR from a real file on disk.

    Triton reads a kernel's text with ``inspect.getsource``, so the mutated
    functions have to live in a file; an ``exec``-ed one raises ``OSError`` at
    first launch. BOTH functions go in, because the outer kernel resolves
    ``_conductive_component`` through its own module globals — a file carrying
    only the outer one would call the SHIPPED helper and every recurrence mutation
    would report itself uncaught.
    """
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_cond.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cond_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    sys.modules[spec.name] = module                       # type: ignore[union-attr]
    spec.loader.exec_module(module)                       # type: ignore[union-attr]
    return module


def shipped_source() -> str:
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    return (textwrap.dedent(inspect.getsource(conductivity._conductive_component.fn))
            + "\n\n"
            + textwrap.dedent(inspect.getsource(
                conductivity.conductive_pml_curl_step.fn)))


def run_mutations(results: Dict[str, Any], out_path: str,
                  repo_root: str) -> Dict[str, Any]:
    """Each defect must change the answer. A leg that reports a pass is a failure."""
    shipped = shipped_source()
    out: Dict[str, Any] = {}
    legs = [(name, transform, ("synthetic", "real"))
            for name, transform in SOURCE_MUTATIONS.items()]
    legs += [(name, transform, ("near_one",))
             for name, transform in NEAR_ONE_MUTATIONS.items()]
    for name, transform, sources in legs:
        mutated, hits = transform(shipped)
        if hits == 0 or mutated == shipped:
            out[name] = {"error": "the mutation matched nothing; the needle has "
                                  "drifted away from the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        module = compile_mutated(mutated)
        summary = run_synthetic(
            results, out_path, repo_root, label="mutation:" + name,
            guards=(True,), shapes=SHAPES[:2], sources=sources,
            kernel=module.conductive_pml_curl_step)
        caught = summary["guarded_identical"] < summary["guarded_ran"]
        out[name] = {"sites": hits, "ran": summary["guarded_ran"],
                     "identical": summary["guarded_identical"], "caught": caught}
        log(f"[mut] {name}: sites={hits} identical={summary['guarded_identical']}"
            f"/{summary['guarded_ran']} CAUGHT={caught}")
        results["mutations"] = out
        save(results, out_path)
    for name in HOST_MUTATIONS:
        patterns = ((1, 0, 1),) if name == "all_three_conductive" else COND_PATTERNS
        sub_sources = ("synthetic",) if name == "integer_coefficients_on_B" else \
            ("synthetic", "real")
        summary = run_synthetic(
            results, out_path, repo_root, label="mutation:" + name,
            guards=(True,), shapes=SHAPES[:2], sources=sub_sources,
            host_mutation=name, patterns=patterns)
        caught = summary["guarded_identical"] < summary["guarded_ran"]
        out[name] = {"host": True, "ran": summary["guarded_ran"],
                     "identical": summary["guarded_identical"], "caught": caught}
        log(f"[mut] {name}: identical={summary['guarded_identical']}/"
            f"{summary['guarded_ran']} CAUGHT={caught}")
        results["mutations"] = out
        save(results, out_path)
    results["mutations"] = out
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The reading of the long-run legs, and the recorded verdict
# ---------------------------------------------------------------------------
#
# WHY BOTH EXIST. Before 2026-08-19 this gate wrote sixteen diverging rows and
# never wrote down what it CONCLUDED from them — no ``status``, no ``verdict``,
# no ``release``. ``gate_provenance.read_verdict`` reports that as
# ``released=None``: UNREADABLE, which is deliberately not the same claim as
# refused, and a weld cannot cite a record that does not state its own outcome.
#
# The verdict below is a ``verdict``-dict carrying ``pass``, the shape
# ``gate_triton_cylindrical`` already writes and ``read_verdict`` already
# understands — not an eleventh spelling of one fact.
#
# THE LONG-RUN LEGS ARE NOT A PASS/FAIL CLAUSE, transcribed from
# ``gate_triton_cylindrical.verdict`` unchanged in role: what ``multi`` and
# ``decay`` report is a MEASURED BUDGET. What IS a clause is whether the
# divergence has been ATTRIBUTED — a gate may not release on a step number it
# cannot explain, and it may not release at all if the explanation is a defect.

#: The legs a release rests on. ``multi`` and ``decay`` are not here: they report
#: a budget, and :func:`attribute_long_run` is what turns that budget into a
#: clause — through ``controls``, which IS here.
CERTIFYING_LEGS: Tuple[str, ...] = ("validate", "synthetic", "controls",
                                    "mutations")

#: Everything ``main`` runs by default.
DEFAULT_LEGS: Tuple[str, ...] = ("validate", "synthetic", "multi", "decay",
                                 "controls", "mutations")

#: The three readings :func:`attribute_long_run` may return. ``UNATTRIBUTED`` is
#: NOT a synonym for either of the others: it is the honest answer when the
#: controls cannot decide, and it refuses a release exactly as ``DEFECT`` does —
#: because releasing on an unexplained divergence is the thing this leg exists to
#: stop — while making a different claim in the record.
BUDGET = "BUDGET"
DEFECT = "DEFECT"
UNATTRIBUTED = "UNATTRIBUTED"
NOTHING_TO_ATTRIBUTE = "NO_DIVERGENCE"


def long_run_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every ``multi``/``decay`` row present, whichever legs ran."""
    rows: List[Dict[str, Any]] = []
    for leg in ("multi_step", "decay"):
        record = results.get(leg)
        if isinstance(record, dict):
            rows.extend(record.get("runs") or [])
    return rows


def _configuration(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (tuple(row.get("shape") or ()), row.get("sub_step"), row.get("drive"))


def control_pairs(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per CONFIGURATION: did the control amplify, and did the kernel diverge?

    WHY PER CONFIGURATION AND NOT A FLEET COUNT. Measured 2026-08-19 on GPU 3,
    320 steps: seven of eight ``one_ulp`` rows diverged, and the eighth —
    ``(32,32,32) step_B driven`` — did not, even though the flip was applied
    (``fu_By`` flat index 16384, ``0x3e7394e3 -> 0x3e7394e2``) and recorded. A
    fleet count reads that as "the control failed", refuses the release and calls
    the run a DEFECT.

    It is the opposite. In that SAME configuration the kernel is bit-identical to
    the array path for the whole budget, and so are the other three ``step_B
    driven`` rows: a driven sub-step walks to a fixed point, where a last-bit
    perturbation is rounded away instead of growing. The control did not fail —
    it AGREED with the kernel, row for row, in the one place where nothing
    diverged.

    So the pairing is the measurement, and it carries a second clause a fleet
    count cannot state at all: where the control is QUIET the kernel must be
    CLEAN. A configuration that does not amplify a single last bit has no excuse
    to offer for a kernel that diverges in it.

    Keyed on (shape, sub_step, drive) — the axes ``_control_setup`` varies. Only
    configurations carrying BOTH row kinds appear; the ``one_ulp`` rows run on two
    shapes and the substitution rows on four, and a key with one half is not a
    pair.
    """
    controls = results.get("controls") or {}
    amplified: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    diverged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in controls.get("rows") or []:
        if row.get("control") == "one_ulp":
            amplified[_configuration(row)] = {
                "amplified": row.get("first_divergence") is not None,
                "one_ulp_first_divergence": row.get("first_divergence"),
                "seed_flip": row.get("seed_flip"),
                "seed_flip_failed": row.get("seed_flip_failed")}
        elif row.get("control") == "substitution":
            lossless = (row.get("lossless_first_divergence")
                        or {}).get("conductive_plan") or {}
            diverged[_configuration(row)] = {
                "kernel_diverged": bool(lossless),
                "lossless_first_divergence": lossless}
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(amplified) & set(diverged), key=repr):
        merged = dict(amplified[key])
        merged.update(diverged[key])
        # A ROW THAT COULD NOT BE SEEDED IS NOT A ROW THAT DID NOT AMPLIFY.
        # ``_flip_one_ulp`` returns None when the volume held no normal float,
        # and reading that as a quiet control would let an unseeded row excuse a
        # divergence — or condemn a clean one.
        merged["usable"] = merged.get("seed_flip") is not None
        out[repr(key)] = merged
    return out


def flush_signature(results: Dict[str, Any]) -> Dict[str, Any]:
    """Are the differing floats a FLUSHED SUBNORMAL, or an arithmetic difference?

    Every first-divergence forensic in the artifact is summed. ``flush_signature``
    counts differing floats that are EXACTLY ZERO on one side and SUBNORMAL on the
    other — the signature of CuPy's ``-ftz=true`` meeting Triton's native keep,
    which is a disagreement about the subnormal band and not about the recurrence.

    Reported, never asserted: the share is a fact about the bytes, and what it
    licenses is stated where the reading is made.
    """
    differing = 0
    flushed = 0
    arrays = 0
    for row in long_run_rows(results):
        for detail in (row.get("first_divergence_forensics") or {}).values():
            arrays += 1
            differing += int(detail.get("differing") or 0)
            flushed += int(detail.get("flush_signature") or 0)
    return {"arrays_at_first_divergence": arrays,
            "differing_floats": differing,
            "carrying_the_flush_signature": flushed,
            "all_differing_floats_are_flushed_subnormals":
                bool(differing) and flushed == differing}


def attribute_long_run(results: Dict[str, Any]) -> Dict[str, Any]:
    """BUDGET, DEFECT or UNATTRIBUTED — and the row that says which.

    THE ORDER OF THESE TESTS IS THE ARGUMENT, so it is written out rather than
    left to be reconstructed from the code:

    1. **No controls, no reading.** A first-divergence step number with no
       control beside it is what the 2026-08-10 and 2026-08-19 artifacts already
       carried, and it attributed nothing then either.
    2. **The null control must be clean.** Two oracle states, identically seeded,
       Triton nowhere in the row. If they drift, the harness's own comparison is
       noisy and every step number in this file is noise — nothing further may be
       read.
    3. **The substitution row's self-check must hold.** The two ORACLES must
       agree on the lossless component by construction; if they do not, the
       conductive and plain arms are stepping different physics and their
       agreement or disagreement means nothing.
    4. **A conductive component that diverges is a DEFECT, full stop.** The
       conductive recurrence is this kernel's entire subject. No budget argument
       reaches it: the controls explain a LOSSLESS component's last bit, not a
       disagreement in the four-case partition the kernel exists to implement.
    5. **No amplification, no budget — PER CONFIGURATION.** If a configuration's
       one-ULP control does NOT diverge within the budget, that configuration does
       not amplify a last-bit disagreement, and a kernel that diverges in it
       cannot be excused by one. That reads as DEFECT, not as UNATTRIBUTED: the
       controls DID decide. A configuration where NEITHER amplifies is the control
       agreeing with the kernel and is not a failure — see :func:`control_pairs`
       for the measurement that forced this to be per-configuration.
    6. **The substitution decides the rest.** If the conductive plan and the
       separately welded ordinary plan produce byte-identical lossless state for
       the whole budget, and the ordinary plan reproduces the same divergence
       against its own oracle at the same step and in the same array, then the
       divergence belongs to the lossless recurrence both kernels share — not to
       the conductive kernel — and it is a BUDGET. If the two kernels DISAGREE,
       the conductive kernel's compiled-out branch introduced the difference and
       that is a DEFECT.
    """
    controls = results.get("controls") or {}
    rows = long_run_rows(results)
    diverging = [r for r in rows if r.get("first_divergence") is not None]
    out: Dict[str, Any] = {
        "long_run_rows": len(rows),
        "diverging_rows": len(diverging),
        "first_divergence_steps": sorted(
            r["first_divergence"] for r in diverging),
        "conductive_component_diverged": any(
            r.get("conductive_component_diverged") for r in rows),
        "controls_present": bool(controls),
    }
    subs = [r for r in (controls.get("rows") or [])
            if r.get("control") == "substitution"]
    out["substitution_rows"] = len(subs)

    def decide(reading: str, why: str) -> Dict[str, Any]:
        out["reading"] = reading
        out["why"] = why
        return out

    if not rows:
        return decide(NOTHING_TO_ATTRIBUTE,
                      "no long-run leg ran; there is no divergence to attribute")
    if not diverging and not out["conductive_component_diverged"]:
        return decide(NOTHING_TO_ATTRIBUTE,
                      f"all {len(rows)} long-run rows are bit-identical for "
                      f"their whole budget")
    if not controls:
        return decide(UNATTRIBUTED,
                      "the controls leg did not run; a first-divergence step "
                      "number with no null control beside it attributes nothing")

    nulls_clean = (controls.get("null_total", 0) > 0
                   and controls["null_identical"] == controls["null_total"])
    out["null"] = f"{controls.get('null_identical')}/{controls.get('null_total')}"
    if not nulls_clean:
        return decide(UNATTRIBUTED,
                      f"the null control is not clean ({out['null']} rows "
                      f"identical): the oracle disagrees with itself, so every "
                      f"step number in this artifact is harness noise")

    out["oracles_agree_on_lossless"] = (
        f"{controls.get('substitution_oracles_agree')}/"
        f"{controls.get('substitution_total')}")
    if subs and controls.get("substitution_oracles_agree") != len(subs):
        return decide(UNATTRIBUTED,
                      f"the substitution row's own self-check failed "
                      f"({out['oracles_agree_on_lossless']}): the conductive and "
                      f"plain oracles disagree about the lossless component, so "
                      f"the two arms are not stepping the same recurrence")

    if out["conductive_component_diverged"] or controls.get(
            "substitution_conductive_component_diverged"):
        return decide(DEFECT,
                      "a CONDUCTIVE component diverged from the array path. The "
                      "four-case conductive recurrence is this kernel's subject; "
                      "no control in this leg speaks to it")

    pairs = control_pairs(results)
    out["one_ulp"] = (f"{controls.get('one_ulp_diverged')}/"
                      f"{controls.get('one_ulp_total')}")
    out["control_pairs"] = pairs
    unusable = sorted(key for key, p in pairs.items() if not p["usable"])
    if unusable:
        return decide(UNATTRIBUTED,
                      f"{len(unusable)} one-ULP control row(s) could not be "
                      f"seeded with a normal float, so their configurations have "
                      f"no amplification measurement at all: {unusable[:3]}")
    unexplained = sorted(key for key, p in pairs.items()
                         if p["kernel_diverged"] and not p["amplified"])
    if unexplained:
        return decide(DEFECT,
                      f"{len(unexplained)} configuration(s) where the kernel "
                      f"diverges and a single flipped last bit does NOT: "
                      f"{unexplained[:3]}. Amplification cannot explain a "
                      f"divergence in a configuration that does not amplify")
    if not pairs:
        return decide(UNATTRIBUTED,
                      "no configuration carries both a one-ULP control and a "
                      "substitution row, so no divergence can be paired with an "
                      "amplification measurement")
    if not any(p["amplified"] for p in pairs.values()):
        return decide(DEFECT,
                      "the one-ULP control is inert in every paired "
                      "configuration: nothing here amplifies, so nothing here "
                      "excuses a divergence")
    out["amplifying_configurations"] = sum(bool(p["amplified"])
                                           for p in pairs.values())
    out["paired_configurations"] = len(pairs)

    if not subs:
        return decide(UNATTRIBUTED,
                      "no substitution row ran; the null and one-ULP controls "
                      "establish that this configuration amplifies, but nothing "
                      "says whether the divergence follows THIS kernel")

    out["kernels_agree_on_lossless"] = (
        f"{controls.get('substitution_kernels_agree')}/{len(subs)}")
    out["plain_reproduces"] = (f"{controls.get('substitution_plain_reproduces')}/"
                               f"{controls.get('substitution_diverging')}")
    if controls.get("substitution_kernels_agree") != len(subs):
        return decide(DEFECT,
                      f"the conductive plan and the separately welded ordinary "
                      f"plan produce DIFFERENT bytes for the same lossless "
                      f"recurrence on the same seeds "
                      f"({out['kernels_agree_on_lossless']} rows agree): the "
                      f"difference is introduced by this kernel")

    diverging_subs = int(controls.get("substitution_diverging") or 0)
    if diverging_subs and (controls.get("substitution_plain_reproduces")
                           != diverging_subs):
        return decide(UNATTRIBUTED,
                      f"the two kernels agree bytewise on the lossless state and "
                      f"the two oracles agree too, yet the ordinary plan does not "
                      f"reproduce the same divergence "
                      f"({out['plain_reproduces']}); those three statements "
                      f"cannot all be true and the row is not readable")

    signature = flush_signature(results)
    out["flush_signature"] = signature
    installed = bool((results.get("subnormal_policy") or {}).get("installed"))
    out["subnormal_policy_installed"] = installed
    mechanism = ""
    if signature["all_differing_floats_are_flushed_subnormals"]:
        mechanism = (
            f" MECHANISM, measured rather than argued: every one of the "
            f"{signature['differing_floats']} differing float(s) at a first "
            f"divergence is EXACTLY ZERO on the array-path side and SUBNORMAL on "
            f"the kernel side"
            + ("" if installed else
               ", and no subnormal policy was installed — CuPy appends "
               "-ftz=true to every NVRTC compile while Triton 3.1.0 keeps "
               "subnormals natively, so the two executors disagree about the "
               "subnormal band BY CONSTRUCTION. This is a half-applied policy, "
               "not an arithmetic disagreement; --subnormal-policy keep is the "
               "confirming arm")
            + ".")
    return decide(BUDGET,
                  f"every divergence is in a LOSSLESS component; the conductive "
                  f"plan and the separately welded ordinary plan are byte-identical "
                  f"there for the whole budget ({out['kernels_agree_on_lossless']}), "
                  f"the ordinary plan reproduces the same divergence against its "
                  f"own oracle ({out['plain_reproduces']}), the oracle is "
                  f"deterministic ({out['null']}) and a single flipped last bit "
                  f"amplifies within the budget in "
                  f"{out['amplifying_configurations']}/{out['paired_configurations']} "
                  f"paired configurations — and in the one that does not amplify, "
                  f"the kernel does not diverge either. The step numbers are a "
                  f"measured budget for the lossless recurrence both kernels share, "
                  f"not a defect in the conductive one." + mechanism)


def certified_budget(record: Optional[Dict[str, Any]]) -> Optional[int]:
    """The step this leg may claim: the EARLIEST divergence, or the full budget."""
    if not isinstance(record, dict):
        return None
    reached = [r["first_divergence"] for r in (record.get("runs") or [])
               if r.get("first_divergence") is not None]
    return min(reached) - 1 if reached else record.get("steps")


def verdict(results: Dict[str, Any], legs: Sequence[str]) -> Dict[str, Any]:
    """The gate's own reading of its own rows. Every clause stated, none inferred.

    A clause whose reader raises is FALSE with the exception recorded: a verdict
    function that dies takes the verdict with it, and an artifact with no verdict
    reads as unreadable rather than refused.
    """
    out: Dict[str, Any] = {
        "legs_requested": list(legs),
        "certifying_legs": list(CERTIFYING_LEGS),
        "clauses": {}, "detail": {}, "reasons": [],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    requested = set(legs)

    # COMPLETENESS FIRST. ``--legs validate`` must not be able to produce a
    # released artifact: with no certifying leg in the run there is nothing to be
    # wrong, and "no clause failed" is not the statement "the gate certified".
    missing = [name for name in CERTIFYING_LEGS if name not in requested]
    out["clauses"]["covers_every_certifying_leg"] = not missing
    if missing:
        out["reasons"].append("this run did not include the certifying leg(s) "
                              + ", ".join(missing))

    def clause(name: str, leg: str, reader: Callable[[], Tuple[bool, Any]]) -> None:
        if leg not in requested:
            return
        try:
            ok, detail = reader()
        except Exception as exc:  # noqa: BLE001 - a verdict must not die reading
            ok, detail = False, {"error": f"{type(exc).__name__}: {exc}"}
        out["clauses"][name] = bool(ok)
        out["detail"][name] = detail
        if not ok:
            out["reasons"].append(f"{name}: {json.dumps(detail, default=str)}")

    def _validate() -> Tuple[bool, Any]:
        record = results.get("validate") or {}
        return bool(record.get("pass")) and int(record.get("ran", 0)) > 0, {
            "identical": f"{record.get('identical')}/{record.get('ran')}"}

    def _guarded() -> Tuple[bool, Any]:
        record = (results.get("synthetic") or {}).get("gate") or {}
        return bool(record.get("guarded_pass")), {
            "guarded": f"{record.get('guarded_identical')}/{record.get('guarded_ran')}",
            "branches_exercised": record.get("branches_exercised")}

    def _control_bites() -> Tuple[bool, Any]:
        record = (results.get("synthetic") or {}).get("gate") or {}
        return (bool(record.get("unguarded_control_bites"))
                and int(record.get("unguarded_ran", 0)) > 0), {
            "unguarded_identical":
                f"{record.get('unguarded_identical')}/{record.get('unguarded_ran')}"}

    def _mutations() -> Tuple[bool, Any]:
        record = results.get("mutations") or {}
        names = (tuple(SOURCE_MUTATIONS) + tuple(NEAR_ONE_MUTATIONS)
                 + tuple(HOST_MUTATIONS))
        absent = [n for n in names if n not in record]
        errored = [n for n in names if (record.get(n) or {}).get("error")]
        uncaught = [n for n in names
                    if n in record and not (record[n] or {}).get("caught")]
        return not (absent or errored or uncaught), {
            "caught": f"{len(names) - len(absent) - len(uncaught)}/{len(names)}",
            "missing": absent, "errored": errored, "not_caught": uncaught}

    def _null() -> Tuple[bool, Any]:
        record = results.get("controls") or {}
        return (int(record.get("null_total", 0)) > 0
                and record["null_identical"] == record["null_total"]), {
            "identical": f"{record.get('null_identical')}/{record.get('null_total')}"}

    def _one_ulp() -> Tuple[bool, Any]:
        """Wherever the kernel diverges, a single flipped last bit must too.

        NOT a fleet count. See :func:`control_pairs`: a configuration that
        amplifies nothing and diverges nowhere is the control agreeing with the
        kernel, and reading that as a failed control turns the strongest
        confirmation in this leg into a refusal.
        """
        record = results.get("controls") or {}
        pairs = control_pairs(results)
        unseeded = sorted(k for k, p in pairs.items() if not p["usable"])
        unexplained = sorted(k for k, p in pairs.items()
                             if p["kernel_diverged"] and not p["amplified"])
        ok = (bool(pairs) and not unseeded and not unexplained
              and any(p["amplified"] for p in pairs.values()))
        return ok, {
            "diverged_rows": f"{record.get('one_ulp_diverged')}/"
                             f"{record.get('one_ulp_total')}",
            "paired_configurations": len(pairs),
            "amplifying": sum(bool(p["amplified"]) for p in pairs.values()),
            "diverging_without_amplification": unexplained,
            "unseeded": unseeded}

    def _quiet_control() -> Tuple[bool, Any]:
        """Where a flipped last bit does NOT grow, the kernel must be CLEAN.

        The clause the 2026-08-19 measurement earned. It has no counterpart in
        ``gate_triton_cylindrical``, whose control never went quiet on any row;
        here one configuration did, and it is only evidence at all because the
        kernel is bit-identical in exactly that configuration.
        """
        pairs = control_pairs(results)
        quiet = {k: p for k, p in pairs.items() if p["usable"] and not p["amplified"]}
        dirty = sorted(k for k, p in quiet.items() if p["kernel_diverged"])
        return not dirty, {"quiet_configurations": sorted(quiet),
                           "diverged_anyway": dirty}

    def _self_check() -> Tuple[bool, Any]:
        record = results.get("controls") or {}
        total = int(record.get("substitution_total") or 0)
        return (total > 0 and record.get("substitution_oracles_agree") == total), {
            "oracles_agree": f"{record.get('substitution_oracles_agree')}/{total}"}

    clause("validate_reference_matches_stepping", "validate", _validate)
    clause("synthetic_guarded_all_identical", "synthetic", _guarded)
    clause("unguarded_control_bites", "synthetic", _control_bites)
    clause("every_mutation_caught", "mutations", _mutations)
    clause("null_control_is_clean", "controls", _null)
    clause("one_ulp_amplifies_where_the_kernel_diverges", "controls", _one_ulp)
    clause("kernel_is_clean_where_the_control_is_quiet", "controls", _quiet_control)
    clause("substitution_oracles_agree", "controls", _self_check)

    # THE LONG-RUN LEGS: a measured budget, reported, never a clause. What IS a
    # clause is the attribution — a gate may not release on a divergence it
    # cannot explain, and may not release at all when the explanation is DEFECT.
    attribution = attribute_long_run(results)
    out["attribution"] = attribution
    for leg in ("multi_step", "decay"):
        if leg in results:
            out[f"{leg}_certified_steps"] = certified_budget(results[leg])
            out[f"{leg}_first_divergence_steps"] = [
                r.get("first_divergence") for r in results[leg].get("runs") or []]
    if any(leg in requested for leg in ("multi", "decay")):
        ok = attribution["reading"] in (BUDGET, NOTHING_TO_ATTRIBUTE)
        out["clauses"]["long_run_divergence_is_attributed"] = ok
        out["detail"]["long_run_divergence_is_attributed"] = attribution
        if not ok:
            out["reasons"].append(
                f"long_run_divergence_is_attributed: {attribution['reading']} — "
                f"{attribution['why']}")

    raised = sorted(name for name in (results.get("leg_errors") or {})
                    if name in CERTIFYING_LEGS)
    out["clauses"]["no_certifying_leg_raised"] = not raised
    if raised:
        out["reasons"].append("certifying leg(s) raised: " + ", ".join(raised))

    out["pass"] = bool(out["clauses"]) and all(out["clauses"].values())
    return out


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
        json.dump(results, handle, indent=2, sort_keys=True)


def _digest(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"host": platform.node(), "python": sys.version.split()[0],
                            "numpy": np.__version__,
                            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if cp is not None:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            info["device"] = props["name"].decode()
            info["cupy"] = cp.__version__
        except Exception as exc:  # noqa: BLE001
            info["device_error"] = repr(exc)
    try:
        import triton  # noqa: PLC0415

        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        info["triton"] = None
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        info["nvidia_smi"] = None
    return info


def install_policy(request: str) -> Dict[str, Any]:
    """Drive every executor to ONE float32 subnormal policy, or record that none is.

    THE DEFAULT IS ``none`` AND IT IS NOT NEUTRAL. With nothing installed CuPy
    13.5.1 appends ``-ftz=true`` to every NVRTC compile while Triton 3.1.0 keeps
    subnormals natively, so the oracle FLUSHES exactly where the kernel KEEPS.
    That is the configuration every earlier run of this gate measured, and it is
    kept as the default so a re-run reproduces those numbers rather than
    replacing them — but the record has to SAY so, because a divergence measured
    under two disagreeing policies is attributable to the policy before it is
    attributable to anything else. ``--subnormal-policy keep`` is the confirming
    arm: same bytes, both executors keeping, and the divergence either survives
    or it does not.

    The install itself is not re-implemented here. ``gate_triton_complex``
    already owns it — the cache-directory guard, the MEEP-resolution rule, the
    strip counters — and seventeen scripts call it; a second installer would be a
    second authority for one process.
    """
    if request == "none":
        return {"policy": "not_installed", "requested": "none", "installed": False,
                "warning": "no policy installed: CuPy appends -ftz=true to every "
                           "NVRTC compile (cupy/cuda/compiler.py) while Triton "
                           "3.1.0 keeps subnormals natively, so the two executors "
                           "disagree below float32's smallest normal BY "
                           "CONSTRUCTION. These bytes are not the ship "
                           "configuration; they reproduce what every earlier run "
                           "of this gate measured."}
    import gate_triton_complex as complex_gate  # noqa: PLC0415

    complex_gate.install_ftz_strip(None if request == "env" else request)
    if cp is not None:
        from meep_gpu import backends  # noqa: PLC0415

        backends.guard_kernel_compilation(cp)
    stamp = complex_gate.policy_stamp("cupy" if cp is not None else "numpy")
    stamp["requested"] = request
    return stamp


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=",".join(DEFAULT_LEGS))
    parser.add_argument("--multi-steps", type=int, default=MULTI_STEP_COUNT)
    parser.add_argument("--decay-steps", type=int, default=DECAY_STEP_COUNT)
    parser.add_argument("--control-steps", type=int, default=CONTROL_STEP_COUNT)
    parser.add_argument("--subnormal-policy", default="none",
                        choices=("none", "keep", "flush", "env"),
                        help="drive BOTH executors to one float32 subnormal "
                             "policy before the first device compile. Default "
                             "'none' installs nothing, which is what every "
                             "earlier run of this gate measured and leaves CuPy "
                             "flushing where Triton keeps. 'env' resolves the "
                             "request from MEEP_GPU_SUBNORMAL_POLICY.")
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    results: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "legs": list(legs),
        "subnormal_policy": install_policy(args.subnormal_policy),
        "environment": device_info(),
        "sources": {
            "conductivity.py": _digest(conductivity.__file__.rstrip("c")),
            "gate": _digest(os.path.abspath(__file__)),
            "probe": _digest(os.path.join(_HERE,
                                          "probe_fused_kernel_bit_identity.py")),
        },
        "step_budget": {"multi": args.multi_steps, "decay": args.decay_steps,
                        "claim": "bit-identity is claimed for these step counts "
                                 "and no further; plan section 16 bounds any "
                                 "whole-run claim independently"},
    }
    save(results, args.out)

    def run_leg(name: str, body: Callable[[], Any]) -> None:
        """One leg, or the record of why it did not finish.

        A LEG IS NOT ALLOWED TO KILL THE RUN. An uncaught exception here writes
        no verdict at all, and ``read_verdict`` reports a verdict-less artifact
        as ``released=None`` — unreadable, which is deliberately NOT the claim
        "refused". Recording the exception and carrying on turns a leg that blew
        up into a verdict that says false, and leaves the legs after it measured
        rather than lost.
        """
        if name not in legs:
            return
        log(f"[leg] {name}")
        try:
            body()
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            import traceback  # noqa: PLC0415

            results.setdefault("leg_errors", {})[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}
            log(f"[leg] {name} RAISED {type(exc).__name__}: {exc}")
            save(results, args.out)

    run_leg("validate", lambda: run_validate(results, args.out))
    if cp is None and tuple(legs) != ("validate",):
        log("cupy is absent: only the `validate` leg can run on this host")
        results["aborted"] = "cupy absent"
        results["verdict"] = verdict(results, legs)
        save(results, args.out)
        return 0
    run_leg("synthetic", lambda: run_synthetic(results, args.out, _REPO_API))
    run_leg("multi", lambda: run_multi(results, args.out, args.multi_steps,
                                       False, "multi_step"))
    run_leg("decay", lambda: run_multi(results, args.out, args.decay_steps,
                                       True, "decay"))
    run_leg("controls", lambda: run_controls(results, args.out,
                                             args.control_steps))
    run_leg("mutations", lambda: run_mutations(results, args.out, _REPO_API))

    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results["verdict"] = verdict(results, legs)
    save(results, args.out)
    log(f"VERDICT: pass={results['verdict']['pass']} "
        f"{json.dumps(results['verdict']['clauses'], sort_keys=True)}")
    log(f"ATTRIBUTION: {results['verdict']['attribution']['reading']} — "
        f"{results['verdict']['attribution']['why']}")
    for reason in results["verdict"]["reasons"]:
        log(f"  REASON {reason}")
    log(f"[done] artifact {args.out}")
    return 0 if results["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
