"""Byte gate for Phase B — the fold under COMPLEX storage and under ``grid.beta``.

WHAT THIS MEASURES, and why each leg is not optional.

The four kernels of ``meep_gpu/triton_kernels/folded_complex.py`` are compared
against a reference transcribed from ``stepping.py`` by hand, WORD FOR WORD as
uint32 — never ``allclose``. The reference itself is pinned against the array
path on NumPy by ``validate_folded_complex_reference_vs_stepping.py`` BEFORE any
device run, which is what stops "bit-identical" from meaning "the kernel
reproduces whatever the harness wrote twice".

NEEDLE PLACEMENT, and the one thing a byte comparison may NOT rest on: a needle
may sit anywhere in the float32 range whose downstream products stay FINITE, and
nowhere past it. Read :data:`FILL_NEEDLE_MAGNITUDE` before adding or moving one —
it carries the bound, its arithmetic, and the measured NaN exposure that forced
the rule. The comparator side of the same rule lives in the validator's
``compare_words``.

LEGS

* ``expansion``  — the preflight probe. Reuses the complex tranche's four base
  patterns and the beta tranche's fifth, and ADDS a sixth this tranche needs:
  the mirror PARITY product, a unit-real coefficient on the LEFT. Inheriting the
  base four patterns' verdict would admit an ambiguous platform silently, so the
  pattern is measured and required. ``AMBIGUOUS_BOTH`` is accepted on ANY
  pattern whose own detail block measures the licensable arms zero words apart
  on a positive vector count — the base family's exclusion clause, reached by
  calling ``complex_fields.expansion_license`` over the longer list rather than
  by a rule of this file's own. The parity pattern is the one EXPECTED to be
  blind (with ``c_re = ±1`` and ``c_im = +0.0`` both arms are exact, so it
  constrains nothing), but that is why it goes blind, not a licence for it
  alone: this leg used to accept the word for that pattern and only for it,
  which let a BASE pattern gone blind under the flush policy veto a licence the
  patterns that could still discriminate had already settled. ``NEITHER``
  refuses by name, so does a disagreement between two patterns that DO
  discriminate, and so does an ambiguity claim the record's own detail block
  contradicts — excluded and disagreeing are opposite situations and only the
  first is ever dropped. THE HEADLINE RISK LIVES HERE:
  if CuPy's complex64 scalar multiply takes a real-scalar fast path rather than
  the ``'FF->F'`` loop, the parity product collapses to plane-wise and K2's whole
  delta evaporates — so the probe classifies the parity pattern EXPLICITLY, with
  the plane-wise form carried as a named diagnostic arm. The classified operand
  shape is the SCALAR one the array path uses (``phase * plane`` with a PYTHON
  INT, S:1450-1451); the numpy-scalar and full-array spellings are measured
  beside it and their byte deltas recorded, because an array-times-array product
  forces the ``'FF->F'`` loop by construction and so cannot see the fast path
  this leg exists to detect.
* ``reference``  — the transcription against ``stepping.py`` itself, on whichever
  backend is present, over folded grids at both terminations and both count
  parities.
* ``synthetic``  — the primary byte gate: the full cross product of termination,
  count parity, mirror phase, folded axis (X, Y, Z and two-at-once with MIXED
  count parity), sub-step, Courant (0.5 AND 0.35 — non-power-of-two mandatory),
  storage (complex / complex+beta / real+beta), and an in-plane Bloch phase on an
  UNFOLDED axis beside its k = 0 control. Compares the TARGET and its ``fu_*``.
  The ASSERTED rows are the fusion-OFF ones (``guard=False``), which is the
  configuration production launches; the fusion-ON rows are measured and
  recorded, never asserted.
* ``fill``       — K2 against ``stepping.fill_symmetry_bc_*`` +
  ``fill_folded_far_ghosts_*``, per family, per axis, on FOUR STATE FAMILIES
  because no single state catches everything: (i) a quiet grid DRIVEN through a
  negative-coefficient absorber, (ii) an engineered signed-zero/subnormal plant
  on stored row 2 and on the reflect row specifically, (iii) that same plant
  STEPPED, which is the only state the two-pass fill split's own justification is
  visible on, (iv) a live post-step state after >= 4 full driver-order steps.
  Per-row needle counts are reported SEPARATELY — an aggregate hides the blind
  row. A (grid, family) CELL whose reference census is 0 in EVERY state family
  FAILS: it can see none of the fill's needles. A single family reading 0 is
  recorded and NOT failed, and the whole 12 x 2 x 4 census matrix was measured
  rather than argued (see :func:`blind_fill_cells`): ``engineered_rows`` reads
  58..1347 everywhere and is what carries the folded METALLIC phase = +1 cells,
  where ``zero_init_absorber`` and ``engineered_rows_post_step`` read 0 for a
  stated reason (near coefficient ``+1``, no far fill), and ``live_post_step``
  reads 0 on EVERY row — that family is the realistic-state arm, not a
  signed-zero arm. The grid list includes a MIXED-PHASE two-folded-axis row,
  without which ``reverse_axis_order`` is a null by construction.
* ``zero_init``  — the ZERO-INIT NEGATIVE-COEFFICIENT-ABSORBER case: an in-run
  signed-zero census AND a byte comparison of both curl sub-steps and both ghost
  fills on the resulting quiet state. A census of 0 FAILS the case as VACUOUS,
  not passes it (random-seeded states are provably blind to the signed-zero
  class), and a refusal to substitute fails it too — a census with no kernel
  behind it establishes reachability, not agreement.
* ``identity``   — the reduction products. Zero folded axes -> K1 byte-identical
  to ``complex_fields.bloch_pml_curl_step``; ``HAS_BETA=0`` -> K3a/K3b
  byte-identical to the plain folded kernels. Equivalence, never routing.
* ``mutations``  — ten armed SOURCE mutations, one RECORDED STRUCTURAL NULL and
  six HOST ones (seventeen; the count is pinned by a laptop test so it cannot
  drift again). A mutation that cannot diverge on any grid this gate builds is a
  PREDICTED NULL: it is recorded with its structural reason and its measured
  numbers, or re-aimed at a state where the defect is byte-visible — never
  deleted, and never silently flipped to "predicted not-caught". It still runs,
  still compiles, is still launch-counted and still PTX-checked, so the DISARMED
  paths stay live and a null whose mutant never ran cannot pass as a null;
  divergence from one is a NULL-INVALIDATED failure, because it falsifies the
  recorded reason. ``synthesize_zero_imag`` is the one such record and the
  policy's worked example — its mutant sets the kernel's imaginary coefficient
  argument to a literal 0.0, and ``mirror_parity_coefficients`` already passes a
  word that is BITWISE 0x00000000 at both mirror phases, so mutant and shipped
  receive identical operands and no state can separate them. Its live claim is
  carried by the host twin ``imaginary_words_are_read_not_synthesised``, and the
  premise is asserted directly by a laptop test, so the retirement re-opens
  itself if the coefficient ever stops being bitwise zero. Every source
  mutant is LAUNCH-COUNTED with explicit DISARMED (the mutant never ran) and
  NEEDLE-MISSED (it ran, its PTX differs, and nothing diverged) failure paths,
  and verified to produce PTX that DIFFERS from every shipped specialization
  first (Triton's cache can serve a stale binary to a renamed-but-similar entry
  point). Every ENTRY mutation carries TWO host-side counts, and the second is
  what makes the first mean anything: ``armed_rows`` (the rows whose entry list
  it changed — the twin of ``hits == 0``) and ``discriminating_rows`` (the rows
  on which the mutated plan CAN store a byte the shipped plan does not). Both
  DISARM on zero, and a leg with discriminating rows and zero diverging words is
  NEEDLE-MISSED even when its predicted verdict is "record the measurement" —
  the vacuity path ``reverse_axis_order`` did not have. The two counts differ for
  exactly one leg and that is the point: reversing the two entries of a
  SAME-PHASE two-folded-axis grid changes the list and is a provable no-op,
  because the doubly-written corner's two coefficients are equal and the
  composition commutes bit-exactly.
* ``engine``     — ``stepping.step_B`` / ``step_D`` and the two fills substituted
  on real folded CuPy grids with real PML tables, >= 4 steps, stated budget.
* ``corpus``     — the three lifted anchors at their own numbers. All three are
  MIRROR_PERIODIC at an EVEN count with phase +1, so they are anchors, NOT
  coverage; the odd-count, metallic and phase = -1 arms are synthetic only.

LEG ISOLATION. Every leg runs through :func:`run_leg`. A leg that RAISES is
recorded — type, message and traceback — under ``leg_errors`` and turned into a
NAMED failure by :func:`build_summary`, and the remaining legs still run. The
legs used to be called bare in a fixed order with ``synthetic`` first, so one
raise there erased every later leg's evidence at once. An errored leg is a
failure, never a skip, and a leg that was requested, was not cleanly skipped, and
left neither a result nor an error is a failure too.

FUSION CONFIGURATION. ``kernels.ENABLE_FP_FUSION`` is False and every plan maps
``enable_fp_fusion = ENABLE_FP_FUSION if guard is None else bool(guard)``, so
``guard=False`` IS the shipped launch. Bit-identity is certified for fusion OFF
and for nothing else; the fusion-ON rows are a divergence control. The artifact
states this, the value is READ from ``kernels.py`` rather than transcribed
(:func:`shipped_fusion_default`), and ``build_summary`` refuses if the package
default ever stops matching the set this gate asserts.

SUBNORMAL POLICY. Every device leg runs under ``ieee_keep_ftz_stripped``, through
``gate_triton_complex.install_ftz_strip`` at the NVRTC seam (never re-implemented
here, and never ``-ftz=false`` as a user option — NVRTC rejects the duplicate).
The strip counters are stamped into the artifact, and a probe record cut under
another policy may not license an expansion.

    python -u gate_triton_folded_complex.py --out results/.../gate.json
"""

from __future__ import annotations

import argparse
import cmath
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import textwrap
import time
import traceback
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
except ImportError:  # Laptop leg: the reference transcription still validates.
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

import probe_fused_kernel_bit_identity as probe  # noqa: E402

import gate_triton_complex as gate  # noqa: E402 - policy machinery + base probe

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, mirror_parity  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import complex_fields, folded_complex, special_kz  # noqa: E402

SEED = probe.SEED


def _to_host(array) -> np.ndarray:
    """A host copy decided by the ARRAY, never by what ``import cupy`` bound.

    The laptop legs used to install ``sys.modules["cupy"] = numpy`` so this module
    could be imported without a device, and then ``isinstance(array, cp.ndarray)``
    was TRUE for a plain NumPy array and dispatched to a ``numpy.asnumpy`` that
    does not exist. The stubs are gone and ``probe.to_host`` now dispatches on the
    array too, so this is no longer a workaround — it is the same rule stated
    locally: module identity is not something a host bridge should need.
    """
    if isinstance(array, np.ndarray):
        return array
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


bit_compare = probe.bit_compare
combine = probe.combine
_face = probe._face
log = gate.log
save = gate.save

PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
MIRROR_METALLIC = "mirror_metallic"
MIRROR_PERIODIC = "mirror_periodic"
IS_FOLD = (MIRROR_METALLIC, MIRROR_PERIODIC)

#: Boundary string -> the four-valued constexpr the plan binds. Asserted against
#: the module's own constants, so a drift is a startup failure, not a wrong plane.
CODE_OF = {
    PERIODIC: folded_complex.CODE_PERIODIC,
    METALLIC: folded_complex.CODE_METALLIC,
    MIRROR_METALLIC: folded_complex.CODE_MIRROR_METALLIC,
    MIRROR_PERIODIC: folded_complex.CODE_MIRROR_PERIODIC,
}
assert CODE_OF[PERIODIC] == gate.CODE_OF[PERIODIC]
assert CODE_OF[METALLIC] == gate.CODE_OF[METALLIC]

FU_NAMES = gate.FU_NAMES
FIELD_12 = gate.FIELD_12
ALL_STATE = gate.ALL_STATE

#: The three lifted corpus anchors' own numbers.
BETA_EIGSRC = 0.2
BETA_GRATING_13_2 = -0.6850526103319672
BETA_GRATING_17_7 = -0.9120991827764708
KX_GRATING_13_2 = 2.9207367086
KX_GRATING_17_7 = 2.8579844438

CURL_SUB_STEPS = ("step_B", "step_D")


def shipped_fusion_default() -> bool:
    """``kernels.ENABLE_FP_FUSION`` — READ, never assumed, and never a comment.

    This is the value that decides which synthetic rows a pass may assert, so it
    is the one number in the fusion stamp that must not be transcribed by hand.
    ``kernels.py`` imports ``triton`` at module scope, so on a laptop it cannot
    be imported; the literal is parsed out of the source instead, which reads the
    same constant from the same file. A source that no longer binds it to a plain
    ``True``/``False`` raises here rather than letting the artifact state a
    configuration nobody measured.
    """
    path = os.path.join(_REPO_API, "meep_gpu", "triton_kernels", "kernels.py")
    try:
        from meep_gpu.triton_kernels import kernels  # noqa: PLC0415

        return bool(kernels.ENABLE_FP_FUSION)
    except ImportError:
        pass
    import ast  # noqa: PLC0415

    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "ENABLE_FP_FUSION":
                value = ast.literal_eval(node.value)
                if not isinstance(value, bool):
                    raise RuntimeError(
                        f"kernels.ENABLE_FP_FUSION is {value!r}, not a bool; the "
                        f"gate cannot state which configuration it certifies")
                return value
    raise RuntimeError(f"no module-level ENABLE_FP_FUSION assignment in {path}")


def write_provenance(results_dir: str) -> str:
    """This tranche binds its own provenance — ``fingerprints.json`` carries no
    entry for it, and its AS-READ hash is recorded so a concurrent edit shows."""
    return gate.write_provenance(results_dir, extra={
        "tranche": "folded_complex (Phase B)",
        "module": gate._digest(os.path.join(
            _REPO_API, "meep_gpu", "triton_kernels", "folded_complex.py")),
        "gate": gate._digest(os.path.abspath(__file__)),
        "depends_on": {
            "complex_fields.py": gate._digest(os.path.join(
                _REPO_API, "meep_gpu", "triton_kernels", "complex_fields.py")),
            "special_kz.py": gate._digest(os.path.join(
                _REPO_API, "meep_gpu", "triton_kernels", "special_kz.py")),
            "symmetry.py": gate._digest(os.path.join(
                _REPO_API, "meep_gpu", "triton_kernels", "symmetry.py")),
        },
        "note": ("K1/K3b INLINE complex_fields' three jit helpers and "
                 "special_kz's _mul_imag_coefficient_left; an edit to either "
                 "invalidates this certification, which is why both digests "
                 "ride here."),
    })


# ---------------------------------------------------------------------------
# The PARITY pattern — the sixth probe pattern, and the tranche's headline risk
# ---------------------------------------------------------------------------

#: The 16 engineered words the parity probe crosses with itself, plus the random
#: tail — one CONTIGUOUS block per parity, which is what lets the probe re-run the
#: same vectors as a SCALAR multiply (the operand shape the array path uses).
PARITY_WORDS: Tuple[float, ...] = (
    0.0, -0.0, 1e-45, -1e-45, 7e-45, -7e-45, 1e-40, -1e-40,
    1.0, -1.0, 1.5, -1.5, 3.4e38, -3.4e38, 0.75, -0.25)
PARITY_BLOCK = len(PARITY_WORDS) ** 2 + gate.RANDOM_VECTORS


def _parity_operands(rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(c_re, c_im, z_re, z_im) rows for the parity product.

    ``c`` is EXACTLY ±1 + 0j — the only coefficient the fill ever passes — and
    ``z`` carries the signed zeros, subnormals and overflow-adjacent magnitudes
    that are the ONLY discriminators between the full complex product and the
    plane-wise sign flip. A random-only vector set is provably blind here.

    Row layout: parity +1's ``PARITY_BLOCK`` rows, then parity -1's.
    """
    words = list(PARITY_WORDS)
    c_re: List[float] = []
    c_im: List[float] = []
    z_re: List[float] = []
    z_im: List[float] = []
    for parity in (1.0, -1.0):
        for first in words:
            for second in words:
                c_re.append(parity)
                c_im.append(0.0)
                z_re.append(first)
                z_im.append(second)
        for value in rng.uniform(-2.0, 2.0, gate.RANDOM_VECTORS):
            c_re.append(parity)
            c_im.append(0.0)
            z_re.append(float(value))
            z_im.append(float(-value * 0.5))
    return (np.asarray(c_re, np.float32), np.asarray(c_im, np.float32),
            np.asarray(z_re, np.float32), np.asarray(z_im, np.float32))


def _parity_candidates(c_re, c_im, z_re, z_im) -> Dict[str, Any]:
    """The two transcription arms plus the PLANE-WISE diagnostic by name.

    The plane-wise arm is what ``symmetry.mirror_ghost_fill``'s certified REAL
    spelling becomes when ported word-for-word to complex storage — i.e. the
    ``plane_wise_parity_fill`` mutation. Carrying it here means the probe can say
    "this platform's parity multiply IS plane-wise" out loud rather than leaving
    the whole tranche's delta resting on an off-device measurement.
    """
    naive_re = (c_re * z_re) - (c_im * z_im)
    naive_im = (c_re * z_im) + (c_im * z_re)
    n = c_re.size
    v1_re = np.empty(n, np.float32)
    v1_im = np.empty(n, np.float32)
    for i in range(n):
        v1_re[i] = gate.fma32(c_re[i], z_re[i], -float(gate.mul32(c_im[i], z_im[i])))
        v1_im[i] = gate.fma32(c_re[i], z_im[i], float(gate.mul32(c_im[i], z_re[i])))
    return {"NAIVE": (naive_re.astype(np.float32), naive_im.astype(np.float32)),
            "FMA_V1": (v1_re, v1_im),
            "PLANEWISE_diagnostic": ((c_re * z_re).astype(np.float32),
                                     (c_re * z_im).astype(np.float32))}


def _parity_spellings(xp, z, c) -> Dict[str, Any]:
    """The THREE operand shapes of ``phase * plane``, brought to the host.

    ``stepping._write_mirror_ghost`` (:1450-1451) and ``_fill_folded_far_ghosts``
    (:1528-1532) spell it ``phase * plane`` with ``phase`` a PYTHON INT — a
    SCALAR times an array. The headline risk this probe exists for is CuPy taking
    a real-scalar fast path on exactly that multiply, and an array-times-array
    product cannot see it: a full complex64 array operand forces the ``'FF->F'``
    loop by construction. All three spellings are measured, the PYTHON INT one is
    what gets classified, and the two others are recorded as byte deltas against
    it so a divergence in any of them is a named platform finding.
    """
    out: Dict[str, Any] = {}
    n = int(z.shape[0])
    if n != 2 * PARITY_BLOCK:
        raise RuntimeError(f"the parity vector set is {n} rows, not "
                           f"{2 * PARITY_BLOCK}; the per-parity blocks moved")
    pieces = {"python_int_scalar_left": [], "numpy_scalar_left": [],
              "array_left": []}
    for index, parity in enumerate((1, -1)):
        segment = z[index * PARITY_BLOCK:(index + 1) * PARITY_BLOCK]
        pieces["python_int_scalar_left"].append(parity * segment)
        pieces["numpy_scalar_left"].append(
            xp.multiply(np.complex64(parity), segment))
        pieces["array_left"].append(xp.multiply(
            c[index * PARITY_BLOCK:(index + 1) * PARITY_BLOCK], segment))
    for name, parts in pieces.items():
        host = _to_host(xp.concatenate(parts)).astype(np.complex64)
        out[name] = (np.ascontiguousarray(host.real),
                     np.ascontiguousarray(host.imag))
    return out


def measure_parity_pattern(xp) -> Dict[str, Any]:
    """The platform's own bytes for ``python_int(±1) * complex64 array``."""
    rng = np.random.default_rng(SEED + 913)
    c_re, c_im, z_re, z_im = _parity_operands(rng)
    z = xp.asarray(gate._interleave_c8(z_re, z_im))
    c = xp.asarray(gate._interleave_c8(c_re, c_im))
    spellings = _parity_spellings(xp, z, c)
    platform = spellings["python_int_scalar_left"]
    detail = gate._classify(platform, _parity_candidates(c_re, c_im, z_re, z_im))
    detail["vectors"] = int(c_re.size)
    detail["operand_shape"] = ("python_int scalar x complex64 array — "
                               "stepping.py:1450-1451's own spelling")
    deltas: Dict[str, int] = {}
    for name, (other_re, other_im) in spellings.items():
        deltas[name] = int(
            np.count_nonzero(other_re.view(np.uint32)
                             != platform[0].view(np.uint32))
            + np.count_nonzero(other_im.view(np.uint32)
                               != platform[1].view(np.uint32)))
    detail["spelling_deltas"] = deltas
    detail["spellings_agree"] = bool(all(count == 0 for count in deltas.values()))
    return detail


def measure_folded_expansion_record(xp, backend_name: str) -> Dict[str, Any]:
    """The base four + the beta pattern + the PARITY pattern, one record.

    EVERY PATTERN CARRIES ITS VECTOR COUNT, including the two this file adds.
    Both licences here are the base family's rule over a superset, and that rule
    requires an ``AMBIGUOUS_BOTH`` claim to be backed by a positive vector count
    and a measured arms-apart word count — a comparison over zero vectors makes
    every arm "match", so an unbacked ambiguity is indistinguishable from a
    comparison of nothing. Until 2026-08-15 the parity and beta patterns were
    written into ``patterns`` and ``detail`` with no ``vectors`` entry at all,
    the same bookkeeping gap ``python_float_left`` carried in the base gate: the
    measurement was honest and the record could not say so.
    """
    record = gate.measure_expansion_record(xp, backend_name)
    # The beta tranche's own extra pattern, measured through its harness so this
    # file carries no second implementation of it.
    import gate_triton_special_kz as beta_gate  # noqa: PLC0415

    beta_detail = beta_gate.measure_beta_expansion_record(xp, backend_name)
    for name, verdict in beta_detail.get("patterns", {}).items():
        record["patterns"].setdefault(name, verdict)
        record["detail"].setdefault(name, beta_detail["detail"].get(name))
        count = beta_detail.get("vectors", {}).get(name)
        if count is not None:
            record["vectors"].setdefault(name, count)

    parity = measure_parity_pattern(xp)
    record["patterns"][folded_complex.PARITY_PROBE_PATTERN] = parity["classified"]
    record["detail"][folded_complex.PARITY_PROBE_PATTERN] = parity
    record["vectors"][folded_complex.PARITY_PROBE_PATTERN] = int(parity["vectors"])
    record["parity_is_planewise"] = bool(
        parity["matches"].get("PLANEWISE_diagnostic"))
    record["parity_spelling_deltas"] = parity["spelling_deltas"]
    record["parity_spellings_agree"] = parity["spellings_agree"]
    record["subnormal_policy"] = gate.policy_stamp(backend_name)
    return record


def run_expansion(results: Dict[str, Any], out_path: str,
                  artifact_path: Optional[str]) -> Dict[str, Any]:
    backends: List[Tuple[Any, str]] = [(np, "numpy")]
    if cp is not None:
        backends.append((cp, "cupy"))
    records: Dict[str, Any] = {}
    for xp, name in backends:
        started = time.time()
        record = measure_folded_expansion_record(xp, name)
        record["seconds"] = round(time.time() - started, 3)
        records[name] = record
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(record)  # bytes THIS process imported; see gate_provenance
        log(f"[expansion:{name}] " + json.dumps(record["patterns"], sort_keys=True)
            + f" parity_is_planewise={record['parity_is_planewise']}"
            + f" parity_spelling_deltas={record['parity_spelling_deltas']}"
            + f" ({record['seconds']} s)")
        results["expansion"] = records
        save(results, out_path)

    chosen = records.get("cupy")
    reasons: List[str] = []
    verdicts: Dict[str, Any] = {}
    if chosen is None:
        reasons.append("no cupy backend on this host; nothing may be licensed")
    else:
        reasons.extend(gate.ftz_strip_license_reasons())
        reasons.extend(gate.probe_record_policy_reasons(chosen))
        # BOTH licences recorded in full, not reduced to licensed/not. 'measured'
        # against 'environment_default' is the difference between this run having
        # discriminated the arm and having inherited it, and the patterns each
        # verdict excluded are what a reader needs to judge either.
        verdicts["parity"] = folded_complex.parity_expansion_license(chosen)
        verdicts["beta"] = folded_complex.folded_beta_expansion_license(chosen)
        for label, verdict in verdicts.items():
            log(f"[expansion:{label}] arm={verdict['arm']} "
                f"basis={verdict['basis']} discriminating="
                f"{sorted(verdict['discriminating'])} non_discriminating="
                f"{verdict['non_discriminating']}")
            if verdict["basis"] == "environment_default":
                log(f"[expansion:{label}] DEFAULTED FROM ENVIRONMENT (not "
                    f"measured this run): {verdict['why_arbitrary']}")
        if verdicts["parity"]["expansion"] is None:
            reasons.append(
                "the measured record does not license a single EXPANSION for "
                "the parity product (K2); refused by name: "
                + "; ".join(verdicts["parity"]["refusals"]))
        if verdicts["beta"]["expansion"] is None:
            reasons.append(
                "the measured record does not license a single EXPANSION for "
                "the beta product (K3b); refused by name: "
                + "; ".join(verdicts["beta"]["refusals"]))
        if chosen.get("parity_is_planewise"):
            reasons.append(
                "MEASURED: this platform's complex64 unit-real scalar multiply "
                "reproduces the PLANE-WISE form exactly — K2's arithmetic delta "
                "does not exist here, and the plane_wise_parity_fill mutation "
                "cannot be a needle. That is a platform FINDING, not a pass.")
        if not chosen.get("parity_spellings_agree", True):
            reasons.append(
                "MEASURED: this platform's three spellings of the parity "
                "multiply disagree byte-for-byte "
                f"({chosen.get('parity_spelling_deltas')}). The array path uses "
                "the PYTHON INT scalar (stepping.py:1450-1451); a platform on "
                "which the scalar and array operand shapes differ needs the "
                "kernel's licensing re-argued against the scalar form, so this "
                "refuses rather than picking one silently.")
    results["expansion_license"] = {"licensed": not reasons, "reasons": reasons,
                                    "licenses": verdicts}
    if chosen is not None and artifact_path:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)) or ".",
                    exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            # The PROBE artifact gets provenance too, not only the gate verdict.
            # Other gates consume this file as licensing evidence and refuse
            # without it, so "which bytes produced this probe?" is a question its
            # readers need answered — measured 2026-08-17, a probe whose header
            # contradicted its body sent a refusal to the platform's account
            # instead of the record's.
            from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
            _stamp_provenance(chosen)
            json.dump(chosen, handle, indent=1, sort_keys=True, default=str)
        results["expansion_artifact"] = artifact_path
        log(f"[expansion] artifact written to {artifact_path}")
    save(results, out_path)
    return results["expansion_license"]


# ---------------------------------------------------------------------------
# The in-file FOLDED COMPLEX reference — stepping.py transcribed
# ---------------------------------------------------------------------------
#
# The complex tranche's transcription is reused where the fold changes nothing
# (the recurrence, the curl grouping, the Bloch wrap); what is added is exactly
# the three things a fold changes: the ghost VALUE on a folded axis, the widened
# cell-0 mask, and the top-plane mask on a folded PERIODIC axis. Note that the
# reference carries the array path's real ghost VALUES — the kernel's claim is
# that those values are DEAD, and the sweep is what measures it.

def folded_complex_shift(xp, field, axis: int, boundary: str, backward: bool,
                         component: str, phase: Optional[complex],
                         mirror_phase: int,
                         reflect_row: Optional[int]):
    """``stepping._shift_up`` (:1723-1786) / ``_shift_down`` (:1787-1845) on a
    complex volume, all four ghost rules.

    * PERIODIC wraps, with ``_apply_bloch_phase`` on the single wrapped plane
      (up: ``*= phase``; down: ``*= conj(phase)``), SKIPPED when the phase is
      None — which is what keeps k = 0 bit-identical;
    * METALLIC serves an exact complex zero;
    * MIRROR near face (down-shift) serves ``parity * field[2]`` on EITHER
      termination (S:1824-1825);
    * MIRROR far face (up-shift) serves ``parity * field[reflect_row]`` on a
      folded PERIODIC axis (S:1778-1779) and an exact zero on a folded METALLIC
      one.
    """
    shifted = xp.roll(field, 1 if backward else -1, axis=axis)
    index = 0 if backward else -1
    if boundary == PERIODIC:
        if phase is not None:
            factor = phase.conjugate() if backward else phase
            shifted[_face(axis, index)] *= shifted.dtype.type(factor)
        return shifted
    if boundary == METALLIC:
        shifted[_face(axis, index)] = 0
        return shifted
    parity = mirror_parity(component, axis, mirror_phase)
    if backward:
        shifted[_face(axis, 0)] = parity * field[
            _face(axis, folded_complex.MIRROR_SOURCE_INDEX)]
        return shifted
    if boundary == MIRROR_PERIODIC:
        if reflect_row is None:
            raise ValueError("a folded PERIODIC axis needs a reflect row")
        shifted[_face(axis, -1)] = parity * field[_face(axis, reflect_row)]
        return shifted
    shifted[_face(axis, -1)] = 0
    return shifted


def folded_complex_curl(xp, sources, term, dtdx, backward: bool,
                        boundaries, phases, mirror_phases, reflect_rows):
    """``stepping._curl_from_operands`` (:1601-1636) — the contract's grouping."""
    _target, g1, a1, g2, a2, _dsig, _dsigu, _iyee = term
    first, second = sources[g1], sources[g2]
    shifted_first = folded_complex_shift(xp, first, a1, boundaries[a1], backward,
                                         g1, phases[a1], mirror_phases[a1],
                                         reflect_rows[a1])
    shifted_second = folded_complex_shift(xp, second, a2, boundaries[a2], backward,
                                          g2, phases[a2], mirror_phases[a2],
                                          reflect_rows[a2])
    return dtdx * ((shifted_first - first) + (second - shifted_second))


def folded_mask(curl, iyee, boundaries) -> None:
    """``stepping._mask_non_owned_cells`` (:1865-1902) in full, in place.

    TWO clauses: shift 0 on a NON-PERIODIC axis -> cell 0 (the fold and the wall
    land on the same plane for different reasons), and shift 1 on a folded
    PERIODIC axis -> the LAST cell (past MEEP's owned window; the fill pass, not
    the curl, is what writes it).
    """
    for axis in range(3):
        if iyee[axis] == 0 and boundaries[axis] != PERIODIC:
            curl[_face(axis, 0)] = 0
        if iyee[axis] == 1 and boundaries[axis] == MIRROR_PERIODIC:
            curl[_face(axis, -1)] = 0


#: target index -> (which source operand is the beta partner, the sign).
#: ``stepping``: Bx/Dx take the SECOND source's centre at +1, By/Dy the FIRST at
#: -1, and target 2 takes nothing (step_db.cpp:148-176 runs over d_c in {X, Y}).
BETA_PARTNERS = {0: ("g2", +1.0), 1: ("g1", -1.0)}


def folded_reference_step(xp, arrays, coefficients, dtdx, sub_step, boundaries,
                          phases, mirror_phases, reflect_rows,
                          beta: float = 0.0, dt: float = 0.0,
                          complex_storage: bool = True) -> None:
    """One folded curl sub-step, in place, beta included when nonzero.

    The beta increment is added to ``curl`` AFTER the dtdx curl and BEFORE the
    masks — ``stepping``'s own order (S:356-363 then :369; :438-445 then :450) —
    which is exactly the slot the fold makes load-bearing.
    """
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    magnetic = sub_step == "step_B"
    for index, term in enumerate(terms):
        target, g1, _a1, g2, _a2, dsig, dsigu, iyee = term
        curl = folded_complex_curl(xp, arrays, term, dtdx, backward, boundaries,
                                   phases, mirror_phases, reflect_rows)
        if beta != 0.0 and index in BETA_PARTNERS:
            slot, sign = BETA_PARTNERS[index]
            partner = arrays[g2 if slot == "g2" else g1]
            coefficient: Any = sign * 2.0 * math.pi * float(beta) * float(dt)
            if complex_storage:
                coefficient = coefficient * (1j if magnetic else -1j)
            curl = curl - (partner.dtype.type(coefficient) * partner)
        folded_mask(curl, iyee, boundaries)
        probe.reference_recurrence(
            arrays[target], arrays["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


def reference_ghost_fill_near(xp, arrays, family: str,
                              axes: Sequence[Dict[str, Any]]) -> None:
    """``stepping.fill_symmetry_bc_*`` (:1426-1451) on plain arrays.

    ``phase * plane`` with a PYTHON INT, exactly as ``_write_mirror_ghost``
    (:1450-1451) spells it — which on complex64 is the FULL complex product.
    This is the reference the plane-wise mutation must diverge from.
    """
    targets = folded_complex.GHOST_FILL_FAMILIES[family]["targets"]
    for name in targets:                  # the array path loops TERMS outermost
        field = arrays[name]
        for entry in axes:                # then axes, X, Y, Z
            axis = int(entry["axis"])
            if folded_complex.TARGET_IYEE[name][axis] != 0:
                continue
            field[_face(axis, 0)] = (
                mirror_parity(name, axis, int(entry["phase"]))
                * field[_face(axis, folded_complex.MIRROR_SOURCE_INDEX)])


def reference_ghost_fill_far(xp, arrays, family: str,
                             axes: Sequence[Dict[str, Any]]) -> None:
    """``stepping.fill_folded_far_ghosts_*`` (:1489-1532) on plain arrays."""
    targets = folded_complex.GHOST_FILL_FAMILIES[family]["targets"]
    for name in targets:
        field = arrays[name]
        for entry in axes:
            axis = int(entry["axis"])
            if not entry["far"] or folded_complex.TARGET_IYEE[name][axis] != 1:
                continue
            field[_face(axis, -1)] = (
                mirror_parity(name, axis, int(entry["phase"]))
                * field[_face(axis, int(entry["reflect_row"]))])


def reference_ghost_fill(xp, arrays, family: str, axes: Sequence[Dict[str, Any]],
                         shape=None) -> None:
    """The two passes, near then far — the array path's own order.

    NOT one pass per axis. The driver puts ``zero_metal_*`` between the two
    (driver.py:3209-3211 / :3222-3224), and under COMPLEX storage the split is
    load-bearing: a component unowned on two axes with different Yee shifts
    (``By`` on a Mirror(X)+Mirror(Y) grid) sees near-then-far here and would see
    far-then-near under a fused per-axis launch. Measured on the laptop: 1 word
    of ``By`` differs. Under real storage the two agree, which is why
    ``symmetry.py`` may fuse them.
    """
    reference_ghost_fill_near(xp, arrays, family, axes)
    reference_ghost_fill_far(xp, arrays, family, axes)


# ---------------------------------------------------------------------------
# Seeding — the states the needles live on
# ---------------------------------------------------------------------------

#: The REAL-storage engineered classes. Signed zeros and subnormals only — no
#: overflow-adjacent magnitude, because a host reference and a device kernel need
#: not agree on a NaN PAYLOAD and ``inf - inf`` in the recurrence would compare a
#: platform choice rather than the transcription.
REAL_NEEDLE_WORDS = np.float32([0.0, -0.0, 1e-45, -1e-45, 7e-45, -7e-45,
                                1e-40, -1e-40])


def _seed_host_real(shape, rng) -> np.ndarray:
    """``gate._seed_host_complex``'s discipline, projected onto REAL storage.

    K3a is the real-storage arm, and a plain ``rng.uniform`` state carries ZERO
    signed zeros and ZERO subnormals — measured 0 of 11583 words on ``SHAPES[0]``
    — so it is PROVABLY blind to the class the whole ieee_keep_ftz_stripped
    policy exists for, and to this module's own single-subtract claim
    (folded_complex.py:918-920, ``curl - (c*g)`` stays ONE subtract "on every
    input including signed zeros"). Three planes are planted on top of the random
    body: a signed-zero checkerboard, a subnormal row, and a negative-zero plane.
    """
    host = np.ascontiguousarray(
        rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    live = [axis for axis in range(3) if int(shape[axis]) > 1]
    if not live:
        return host
    first, last = live[0], live[-1]
    plane = host[_face(last, 0)]
    plane[...] = -0.0
    plane = host[_face(first, -1)]
    checker = (np.indices(plane.shape).sum(axis=0) % 2).astype(bool)
    plane[...] = 0.0
    plane[checker] = -0.0
    if int(shape[last]) > 2:
        plane = host[_face(last, 1)]
        flat = plane.reshape(-1)
        for index in range(flat.size):
            flat[index] = REAL_NEEDLE_WORDS[index % REAL_NEEDLE_WORDS.size]
        plane[...] = flat.reshape(plane.shape)
    return host


def make_state(xp, shape, rng, names, complex_storage: bool) -> Dict[str, Any]:
    if complex_storage:
        return {name: xp.asarray(gate._seed_host_complex(shape, rng))
                for name in names}
    return {name: xp.asarray(_seed_host_real(shape, rng)) for name in names}


#: The (real, imag) word pairs the fill's THREE needle classes actually live in,
#: MEASURED on this laptop over the engineered word set rather than sampled by a
#: diagonal. The diagonal pairing that stood here until 2026-08-12
#: (``words[i % 8]`` / ``words[(i + 3) % 8]``) hit 0 of the 8 plane-wise/identity
#: pairs at mirror phase +1 and 1 of 8 at phase -1, and 0 of the 8 commute pairs
#: — so ``plane_wise_parity_fill`` and ``identity_shortcut_on_even_mirror``
#: reported CAUGHT entirely off the phase = -1 sign flip and ``reverse_axis_order``
#: could not diverge at all. The classes, all measured (see
#: ``test_the_planted_fill_words_cover_every_needle_class``):
#:
#: * A — full product vs plane-wise at mirror phase +1, which is ALSO the
#:   ``phase == +1 -> copy`` identity shortcut: ``z_im`` negatively signed;
#: * B — the same at mirror phase -1: ``z_im`` positively signed with a zero or
#:   subnormal partner;
#: * C — the COMMUTE class: ``c_a (x) (c_b (x) z) != c_b (x) (c_a (x) z)`` when
#:   ``c_a != c_b``, i.e. the mixed-phase two-folded-axis grid.
#:
#: The MAGNITUDE class is :data:`FILL_NEEDLE_MAGNITUDE`; read its derivation
#: before changing any word here.

#: The extreme-magnitude needle, and THE NEEDLE-PLACEMENT POLICY behind it.
#:
#: **A needle may sit anywhere in the float32 range whose DOWNSTREAM products all
#: stay finite. It may never be placed where the path overflows.** A byte
#: comparison across an overflow is not a byte comparison of the transcription:
#: it demands agreement on a NaN's sign bit and payload, and IEEE 754-2019 leaves
#: both unspecified — §6.2.1 (a NaN's sign bit is not interpreted), §6.2.3 (payload
#: propagation is a *should*, not a *shall*), §7.2 (an invalid operation such as
#: ``inf - inf`` delivers a quiet NaN whose payload the standard does not fix).
#: This is not theoretical, and it is MEASURED rather than argued. The shipped
#: ``3.4e38`` pairs overflowed the curl; both platforms then produced the SAME NaN
#: population — 709387 word positions across the validator's 17 cases, on
#: hash-identical sources — but 1414 of those positions disagreed in sign or
#: payload on numpy 2.2.6 and 0 disagreed on numpy 2.4.3, so a raw-bit comparison
#: read 14/17 on the GPU host and 17/17 on the laptop. First divergence: case 13,
#: step_D, ``Dx`` word 268, ``0x7fc00000`` (+qNaN) against ``0xffc00000`` (-qNaN).
#: A real FDTD run that reaches NaN has already diverged, so the overflow needle
#: was testing a state the engine never usefully occupies.
#:
#: THE BOUND. Let ``S`` bound ``|re|`` and ``|im|`` of every state word entering a
#: curl sub-step. With the coefficient extremes MEASURED over the reference
#: validator's 17 grids — ``max|kms| = 1.2666072845458984``, ``max|sinv| = 1.0``,
#: ``max|kps| = 3.2666072845458984``, ``dt/dx = 0.35``, ``|2 pi beta dt| <=
#: 0.15065093735589932`` — each stage multiplies ``S`` by at most::
#:
#:     shift   Bloch mul by unit-modulus c: |c_re z_re - c_im z_im|  <= 2 S
#:     curl    dtdx * ((s1 - f1) + (f2 - s2))    <= 6 dtdx S          = 2.100 S
#:     beta    curl - (c (x) partner)            <= + 2 |c| S         = 0.301 S
#:     fu'     (fu kms - curl) sinv              <= (kms + 2.401) S   = 3.668 S
#:     field'  (field kms_u + fu' - fu) sinv_u   <= (kms + 3.668 + 1) S
#:
#:     A = 2 kms + 6 dtdx + 2 |c| + 1 = 5.9345   per CURL sub-step
#:     C = 1 + kps + kms             = 5.5332    per CONSTITUTIVE sub-step
#:                                               (field += kps fw - kms fw_prev)
#:     G = (A^2 C^2)^3               = 1.2537e9  over the validator's 3 full steps
#:                                               (step_B, update_H, step_D, update_E)
#:
#:     FLT_MAX / G = 2.714e29
#:
#: That is a worst case: it puts every operand at ``S`` with the worst sign at all
#: twelve sub-steps. The MEASURED peak through the same path at this needle is
#: 7.708e29 — an amplification of 7.708, not 1.25e9, and 4.4e8 short of FLT_MAX —
#: with 0 non-finite words in 6925824 inspected and 0 RuntimeWarnings raised.
#:
#: ``1e29`` is the largest ROUND DECIMAL that still leaves a full binary exponent
#: of headroom under the analytic bound (2.714e29 / 1e29 = 2.71), and that margin
#: is deliberate rather than timid: the bound's inputs — kappa + sigma, dt/dx, the
#: step count — are properties of the CURRENT case list, so a fixed literal at the
#: bound would be knife-edge. ``test_the_extreme_fill_needle_stays_FINITE_through
#: _the_whole_path`` recomputes the whole inequality from those inputs, which is
#: what turns a change that would reintroduce the overflow into a red test rather
#: than a silent NaN bloom. It is also mantissa-rich (``0x6fa18f08``), so the
#: products downstream of it round rather than merely shifting an exponent — which
#: a power of two would not.
#:
#: NOT tamed to the ablation's ``1.5``: the magnitude class is real work. The fill
#: itself multiplies by exactly ±1, so every candidate arms the three classes
#: identically (measured: 6/6/6/6 pairs and 36/36/660 diverging fill words at
#: 3.4e38, 1e29 and 1.5 alike) — what the extreme magnitude buys is DOWNSTREAM,
#: where the curl and the split-field recurrence carry a needle 29 decades above
#: the random body through a full sub-step.
#:
#: :data:`PARITY_WORDS` keeps its ``±3.4e38`` and is exempt by MEASUREMENT, not by
#: exception: that probe's coefficient is exactly ``±1 + 0j`` and nothing
#: accumulates, so every product is its own input — 0 non-finite floats across all
#: three candidate arms.
FILL_NEEDLE_MAGNITUDE: float = 1e29

FILL_NEEDLE_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.0, -0.0),        # A, B and C at once
    (-0.0, -0.0),       # A
    (0.0, 0.0),         # B
    (-0.0, 0.0),        # C
    (-0.0, -1.5),       # A
    (0.0, -1.5),        # B
    (-1e-45, 0.0),      # C
    (1e-45, -0.0),      # A
    (1.5, 0.0),         # B
    (0.0, 1.5),         # C
    (7e-45, -0.0),      # A
    (FILL_NEEDLE_MAGNITUDE, 0.0),      # B, and the MAGNITUDE class
    (-1.5, 0.0),        # C
    (FILL_NEEDLE_MAGNITUDE, -0.0),     # A, and the MAGNITUDE class
    (0.0, -1e-45),      # B
    (0.0, 7e-45),       # C
)


def plant_fill_needles(host: np.ndarray, axis: int, reflect_row: Optional[int]
                       ) -> np.ndarray:
    """State family (ii): signed zeros and subnormals on the two rows the fill
    READS — stored row 2 and the reflect row specifically.

    A whole-volume random seed is provably blind to this class; planting on the
    two source rows is what makes the near and the far fill separately visible,
    which the fill-only matrix showed is necessary (the needle MOVES between the
    two fills as the phase flips). The WORD PAIRS are :data:`FILL_NEEDLE_PAIRS`,
    which is the measured union of the three divergent classes — a diagonal walk
    over a word list is not the same thing and was measured blind to two of them.
    """
    def stamp(row: int) -> None:
        plane_re = host.real[_face(axis, row)]
        plane_im = host.imag[_face(axis, row)]
        flat_re = plane_re.reshape(-1)
        flat_im = plane_im.reshape(-1)
        for index in range(flat_re.size):
            real, imag = FILL_NEEDLE_PAIRS[index % len(FILL_NEEDLE_PAIRS)]
            flat_re[index] = np.float32(real)
            flat_im[index] = np.float32(imag)
        plane_re[...] = flat_re.reshape(plane_re.shape)
        plane_im[...] = flat_im.reshape(plane_im.shape)

    stamp(folded_complex.MIRROR_SOURCE_INDEX)
    if reflect_row is not None and reflect_row >= 0:
        stamp(int(reflect_row))
    return host


def signed_zero_census(xp, arrays: Dict[str, Any]) -> int:
    """Count ``0x80000000`` words across the given volumes.

    A census of ZERO means the case is VACUOUS, not that it passed: the class the
    tranche's needles live in was never reached.
    """
    total = 0
    for array in arrays.values():
        host = _to_host(array)
        words = np.ascontiguousarray(host).view(np.uint32).ravel()
        total += int(np.count_nonzero(words == np.uint32(0x80000000)))
    return total


# ---------------------------------------------------------------------------
# The synthetic sweep product
# ---------------------------------------------------------------------------

def _phase(k: float) -> complex:
    return cmath.exp(2j * math.pi * k)


#: >= 4 shapes, one of which has an ``n_plane`` that is NOT a multiple of BLOCK.
SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (13, 11, 9),     # odd non-cube; n_plane 99/117/143, none a multiple of 256
    (12, 14, 10),    # even non-cube
    (9, 7, 1),       # 2-D sheet, invariant z
    (17, 5, 3),      # thin and asymmetric
)

#: Every configuration names which axes are folded, the outer declaration on
#: them, and the reflect-row parity. ``mixed_parity_xy`` is the two-axis case
#: with DIFFERENT count parities in one grid — the arm a single-axis sweep cannot
#: reach (n_full (20, 21) -> reflect (stored-2, stored-3)).
CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "fold_y_periodic", "boundaries": (PERIODIC, MIRROR_PERIODIC, PERIODIC)},
    {"name": "fold_y_metallic", "boundaries": (PERIODIC, MIRROR_METALLIC, PERIODIC)},
    {"name": "fold_x_periodic", "boundaries": (MIRROR_PERIODIC, PERIODIC, PERIODIC)},
    {"name": "fold_z_periodic", "boundaries": (PERIODIC, PERIODIC, MIRROR_PERIODIC)},
    {"name": "fold_z_metallic", "boundaries": (PERIODIC, PERIODIC, MIRROR_METALLIC)},
    {"name": "fold_y_periodic_vs_metallic_wall",
     "boundaries": (METALLIC, MIRROR_PERIODIC, PERIODIC)},
    {"name": "mixed_parity_xy",
     "boundaries": (MIRROR_PERIODIC, MIRROR_PERIODIC, PERIODIC),
     "row_offsets": (2, 3)},
    {"name": "unfolded_control", "boundaries": (PERIODIC, PERIODIC, PERIODIC)},
)

#: Bloch rows: a phase on an UNFOLDED periodic axis beside its k = 0 control.
PHASE_SETS: Tuple[Dict[str, Any], ...] = (
    {"name": "k0", "phases": (None, None, None)},
    {"name": "kx_unfolded", "phases": (_phase(0.23), None, None)},
)

DTDX: Tuple[float, ...] = (0.5, 0.35)   # 0.35 MANDATORY: non-power-of-two
MIRROR_PHASES: Tuple[int, ...] = (1, -1)
ROW_OFFSETS: Tuple[int, ...] = (2, 3)   # even full count / odd full count
BETA_DT = 0.05
MULTI_STEP_COUNT = 4


def reflect_rows_for(shape, boundaries, offsets) -> Tuple[Optional[int], ...]:
    rows: List[Optional[int]] = []
    for axis in range(3):
        if boundaries[axis] != MIRROR_PERIODIC:
            rows.append(None)
            continue
        rows.append(max(0, int(shape[axis]) - int(offsets[axis])))
    return tuple(rows)


def viable(shape, config, phases) -> Optional[str]:
    boundaries = config["boundaries"]
    for axis in range(3):
        if boundaries[axis] != PERIODIC and int(shape[axis]) <= 4:
            return (f"axis {axis} has {shape[axis]} cells; a fold needs room for "
                    f"cell 0, cell 2 and a distinct reflect row")
        if phases[axis] is not None and boundaries[axis] != PERIODIC:
            return f"axis {axis} carries a phase but is not periodic"
    return None


def _coefficients(shape, rng) -> Dict[str, Any]:
    """The six curl tables in BROADCAST shape — (n,1,1) / (1,n,1) / (1,1,n).

    That is ``PML._reshape_for_broadcast``'s layout and the ONLY layout
    ``probe.reference_recurrence`` (probe_fused_kernel_bit_identity.py:347-367,
    ``fu *= kms``) can multiply by. A bare 1-D vector either raises — measured,
    ``operands could not be broadcast together with shapes (13,11,9) (11,)`` on
    ``SHAPES[0]``/``step_B``, which aborted the whole gate at its FIRST case —
    or, on a shape whose ``dsig`` extent happens to equal its z extent, silently
    applies the wrong axis's absorber along z and turns the "reference" into a
    smooth wrong answer. The kernel's own flat views come from
    :func:`_flat_coefficients`, never from this dict directly.
    """
    out: Dict[str, Any] = {}
    for index, axis in enumerate("xyz"):
        n = int(shape[index])
        for stem, low, high in (("kms", 0.2, 1.2), ("sinv", 0.4, 1.4)):
            values = rng.uniform(low, high, n).astype(np.float32)
            out[f"{stem}_{axis}"] = np.ascontiguousarray(
                values.reshape(probe.broadcast_shape(index, n)))
    return out


def _flat_coefficients(xp, coefficients: Dict[str, Any], shape) -> Dict[str, Any]:
    """Contiguous 1-D device views of the broadcast tables — the kernel's layout.

    Asserted rather than assumed: a table that is not the axis's own length, or
    that does not flatten to a contiguous float32 view, is a startup failure
    here instead of a wrong plane on the device.
    """
    flat: Dict[str, Any] = {}
    for key, table in coefficients.items():
        axis = "xyz".index(key.rsplit("_", 1)[1])
        expected = probe.broadcast_shape(axis, int(shape[axis]))
        if tuple(np.shape(table)) != tuple(expected):
            raise RuntimeError(
                f"{key} has shape {tuple(np.shape(table))}, not the broadcast "
                f"shape {tuple(expected)} the reference recurrence needs")
        view = np.ascontiguousarray(table).reshape(-1)
        if view.dtype != np.float32:
            raise RuntimeError(f"{key} is {view.dtype}, not float32")
        flat[key] = xp.asarray(view)
    return flat


def one_curl_case(shape, config, phase_set, dtdx, mirror_phase, offsets,
                  sub_step, guard, expansion, beta, complex_storage,
                  kernel=None) -> Dict[str, Any]:
    """One synthetic sub-step: reference on the host, kernel on the device."""
    started = time.time()
    spec = complex_fields.SUB_STEPS[sub_step] if hasattr(complex_fields, "SUB_STEPS") \
        else folded_complex.SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    names = targets + tuple("fu_" + t for t in targets) + tuple(spec["sources"])
    rng = np.random.default_rng(SEED + 401)
    dtype = np.complex64 if complex_storage else np.float32
    host = {name: (gate._seed_host_complex(shape, rng) if complex_storage
                   else _seed_host_real(shape, rng))
            for name in names}
    coefficients = _coefficients(shape, rng)

    reference = {name: np.array(value, dtype=dtype, copy=True)
                 for name, value in host.items()}
    boundaries = config["boundaries"]
    rows = reflect_rows_for(shape, boundaries, offsets)
    mirror_phases = tuple(mirror_phase for _ in range(3))
    folded_reference_step(np, reference, coefficients, np.float32(dtdx), sub_step,
                          boundaries, phase_set["phases"], mirror_phases, rows,
                          beta=beta, dt=BETA_DT, complex_storage=complex_storage)

    arrays = {name: cp.asarray(np.array(value, dtype=dtype, copy=True))
              for name, value in host.items()}
    flat = _flat_coefficients(cp, coefficients, shape)
    codes = tuple(CODE_OF[b] for b in boundaries)
    row_argument = tuple(-1 if r is None else int(r) for r in rows)
    if beta == 0.0 and complex_storage:
        plan = folded_complex.plan_folded_complex_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phase_set["phases"], dtdx, expansion,
            kernel=kernel)
    elif complex_storage:
        words = special_kz.beta_curl_coefficients(
            beta, BETA_DT, magnetic=(sub_step == "step_B"), complex_storage=True)
        plan = folded_complex.plan_folded_beta_bloch_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phase_set["phases"], dtdx, expansion,
            words, kernel=kernel)
    else:
        plus, minus = special_kz.beta_curl_coefficients(
            beta, BETA_DT, magnetic=(sub_step == "step_B"), complex_storage=False)
        plan = folded_complex.plan_folded_beta_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, dtdx, plus, minus, kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    compared = targets + tuple("fu_" + t for t in targets)
    verdict = combine({name: bit_compare(arrays[name], reference[name])
                       for name in compared})
    return {
        "shape": list(shape), "config": config["name"], "phases": phase_set["name"],
        "dtdx": dtdx, "mirror_phase": mirror_phase, "row_offsets": list(offsets),
        "reflect_rows": list(row_argument), "sub_step": sub_step, "guard": guard,
        "beta": beta, "storage": "complex" if complex_storage else "real",
        "verdict": verdict, "seconds": round(time.time() - started, 3),
    }


def _summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``guard`` is the FUSION setting, not a safety flag.

    ``FoldedComplexPmlCurlPlan.run`` (folded_complex.py:2211) maps
    ``enable_fp_fusion = ENABLE_FP_FUSION if guard is None else bool(guard)`` and
    ``kernels.ENABLE_FP_FUSION`` is False — so ``guard=False`` is the SHIPPED
    configuration and ``guard=True`` is fusion ON. The asserted set is therefore
    the ``guard is False`` rows, which is what ``gate_triton_complex.py:1114`` and
    ``gate_triton_special_kz.py:683`` also spell; the fusion-ON rows are MEASURED
    and never asserted, because with fp fusion enabled the compiler may contract
    either grouping (gate_triton_complex.py:1401). Until 2026-08-12 this file had
    the test inverted and asserted the fusion-ON rows only, so every row in the
    configuration production launches could have diverged with ``passed`` true.
    """
    asserted = [c for c in cases if c["guard"] is False]
    measured = [c for c in cases if c["guard"] is True]
    return {
        "ran": len(cases),
        "fusion_off_asserted_identical": sum(int(c["verdict"]["bit_identical"])
                                             for c in asserted),
        "fusion_off_asserted_total": len(asserted),
        "fusion_on_measured_identical": sum(int(c["verdict"]["bit_identical"])
                                            for c in measured),
        "fusion_on_measured_total": len(measured),
    }


def run_synthetic(results: Dict[str, Any], out_path: str, expansion: int,
                  quick: bool = False) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    shapes = SHAPES[:2] if quick else SHAPES
    configs = CONFIGS[:3] if quick else CONFIGS
    total = 0
    plan_rows: List[Tuple[Any, ...]] = []
    for shape in shapes:
        for config in configs:
            for phase_set in PHASE_SETS:
                if viable(shape, config, phase_set["phases"]):
                    continue
                for dtdx in DTDX:
                    for mirror_phase in MIRROR_PHASES:
                        for offsets in ([config["row_offsets"]]
                                        if "row_offsets" in config
                                        else [(o, o, o) for o in ROW_OFFSETS]):
                            offsets = tuple(offsets) if len(offsets) == 3 else offsets
                            for sub_step in CURL_SUB_STEPS:
                                for beta, storage in ((0.0, True),
                                                      (BETA_GRATING_13_2, True),
                                                      (BETA_GRATING_13_2, False)):
                                    plan_rows.append((shape, config, phase_set,
                                                      dtdx, mirror_phase, offsets,
                                                      sub_step, beta, storage))
    total = len(plan_rows)
    started_all = time.time()
    for index, row in enumerate(plan_rows, start=1):
        shape, config, phase_set, dtdx, mirror_phase, offsets, sub_step, beta, storage = row
        if not storage and phase_set["phases"] != (None, None, None):
            continue  # real storage carries no Bloch phase; the array path raises
        for guard in (False, True):     # fusion OFF first — it is the asserted row
            case = one_curl_case(shape, config, phase_set, dtdx, mirror_phase,
                                 offsets, sub_step, guard, expansion, beta, storage)
            cases.append(case)
        latest = cases[-2]              # the fusion-OFF row
        log(f"case {index}/{total} {config['name']}/{phase_set['name']} "
            f"shape={tuple(shape)} dtdx={dtdx} phase={mirror_phase:+d} "
            f"rows={offsets} {sub_step} "
            f"{'complex' if storage else 'real'} beta={beta:g}: "
            f"fusion_off_identical={latest['verdict']['bit_identical']} "
            f"differing={latest['verdict']['differing_floats']}/"
            f"{latest['verdict']['total_floats']} "
            f"elapsed={round(time.time() - started_all, 1)} s")
        results["synthetic"] = {"summary": _summarize(cases), "cases": cases}
        save(results, out_path)
    return results["synthetic"]["summary"]


# ---------------------------------------------------------------------------
# The fill legs — three state families, per-row needle counts reported apart
# ---------------------------------------------------------------------------

FILL_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"axis": "Y", "boundaries": "periodic", "phase": 1, "cell": (1.6, 2.0, 0.0)},
    {"axis": "Y", "boundaries": "periodic", "phase": 1, "cell": (1.6, 2.1, 0.0)},
    {"axis": "Y", "boundaries": "periodic", "phase": -1, "cell": (1.6, 2.0, 0.0)},
    {"axis": "Y", "boundaries": "periodic", "phase": -1, "cell": (1.6, 2.1, 0.0)},
    {"axis": "Y", "boundaries": "metallic", "phase": 1, "cell": (1.6, 2.0, 0.0)},
    {"axis": "Y", "boundaries": "metallic", "phase": 1, "cell": (1.6, 2.1, 0.0)},
    {"axis": "Y", "boundaries": "metallic", "phase": -1, "cell": (1.6, 2.0, 0.0)},
    {"axis": "Y", "boundaries": "metallic", "phase": -1, "cell": (1.6, 2.1, 0.0)},
    {"axis": "X", "boundaries": "periodic", "phase": 1, "cell": (2.1, 1.6, 0.0)},
    {"axis": "Z", "boundaries": "periodic", "phase": -1, "cell": (1.2, 1.2, 2.0)},
    {"axis": "XY", "boundaries": "periodic", "phase": 1, "cell": (2.0, 2.1, 1.2)},
    # The MIXED-PHASE two-folded-axis grid. Without it `reverse_axis_order` is a
    # null BY CONSTRUCTION: with one declared phase per grid the two planes carry
    # the SAME coefficient on every component, and `c (x) (c (x) z)` commutes
    # bit-exactly. MEASURED: 0/128 diverging words at (+1,+1) and at (-1,-1),
    # 8/128 at (+1,-1) — so the commute question only exists on this row.
    {"axis": "XY", "boundaries": "periodic", "phase": (1, -1),
     "cell": (2.0, 2.1, 1.2)},
)

#: The --quick subset. It MUST include the mixed-phase two-axis grid or
#: `reverse_axis_order` is a literal no-op on every row it runs.
QUICK_FILL_GRID_INDICES = (0, 2, len(FILL_GRIDS) - 1)

STATE_FAMILIES = ("zero_init_absorber", "engineered_rows",
                  "engineered_rows_post_step", "live_post_step")


def _phase_label(spec: Dict[str, Any]) -> str:
    phase = spec["phase"]
    if isinstance(phase, (tuple, list)):
        return "/".join(f"{int(p):+d}" for p in phase)
    return f"{int(phase):+d}"


def build_folded_fields(spec: Dict[str, Any], xp, complex_storage=True,
                        beta: float = 0.0, thin_pml: bool = False):
    declared_phase = spec["phase"]
    phases = (tuple(declared_phase) if isinstance(declared_phase, (tuple, list))
              else tuple(declared_phase for _ in spec["axis"]))
    if len(phases) != len(spec["axis"]):
        raise ValueError(f"{spec['axis']!r} folded axes but {len(phases)} phases")
    planes = tuple(Mirror(name, int(phase))
                   for name, phase in zip(spec["axis"], phases))
    dimensions = 2 if float(spec["cell"][2]) == 0.0 else 3
    declared = spec["boundaries"]
    if dimensions == 2 and declared != "periodic":
        declared = {name.lower(): declared for name in spec["axis"]}
    grid = Grid(resolution=10.0, cell_size=spec["cell"], dimensions=dimensions,
                courant=0.35, boundaries=declared, symmetry=planes, beta=beta,
                xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    thickness = []
    for index in range(3):
        if int(grid.shape[index]) < 6:
            thickness.append((0, 0))
        elif "XYZ"[index] in spec["axis"]:
            thickness.append((0, 3 if thin_pml else 2))
        else:
            thickness.append((3, 3) if thin_pml else (2, 2))
    return grid, fields, PML(grid=grid, thickness=tuple(thickness))


def _drive_array_path(fields, pml, steps: int) -> None:
    """``steps`` complete driver-order steps on the array path, fills included."""
    for _ in range(steps):
        stepping.step_B(fields, pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.fill_symmetry_bc_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        stepping.update_E(fields, pml)


def seed_fill_state(fields, grid, names, family: str, pml) -> Dict[str, Any]:
    """One of the FOUR state families, written into ``fields`` in place.

    ``zero_init_absorber`` ZEROES AND THEN STEPS. Zeroing alone is not the case
    its name promises: on an all-``+0.0`` state at mirror phase +1 the full
    product and the plane-wise form both return ``(+0.0, +0.0)``
    (``re = (+1)(+0.0) - (+0.0)(+0.0) = +0.0``), so every phase = +1 grid was
    provably blind to plane_wise_parity_fill, identity_shortcut,
    synthesize_zero_imag and unary_minus_addend, and the reference census on the
    metallic even-mirror rows was 0 — VACUOUS, recorded and never failed. Driving
    the quiet grid through the negative-coefficient absorber is what makes the
    ``-0.0`` stores the family exists for actually appear.

    ``engineered_rows_post_step`` plants AND steps. Neither of the other two does
    both, and the two-pass fill split's own measured justification (1 word of
    ``By`` under a fused per-axis launch, folded_complex.py:752-780) lives in the
    combination — a plant with no step and a step with no plant are each blind
    to it.
    """
    rng = np.random.default_rng(SEED + 77)
    folded_axes = [a for a in range(3) if grid.is_mirrored(a)]
    rows = stepping._far_reflect_rows(grid)
    zeroed = family == "zero_init_absorber"
    planted = family in ("engineered_rows", "engineered_rows_post_step")
    for name in (ALL_STATE if zeroed else names):
        target = getattr(fields, name, None)
        if target is None:
            continue
        if zeroed:
            target[...] = 0
            continue
        host = gate._seed_host_complex(tuple(grid.shape), rng)
        if planted:
            for axis in folded_axes:
                host = plant_fill_needles(host, axis, rows[axis])
        target[...] = fields.grid.xp.asarray(host)
    if family == "live_post_step":
        _drive_array_path(fields, pml, MULTI_STEP_COUNT)
    elif zeroed:
        # The absorber is what turns a quiet grid into signed-zero stores; the
        # case's census precondition is asserted per case in run_fill.
        _drive_array_path(fields, pml, ZERO_INIT_STEPS)
    elif family == "engineered_rows_post_step":
        # The overflow suppression that used to wrap these two calls is GONE with
        # the 3.4e38 plants (see FILL_NEEDLE_MAGNITUDE): FILL_NEEDLE_MAGNITUDE is
        # bounded so this whole path stays finite, and an `over`/`invalid`
        # RuntimeWarning here is now the SIGNAL that a needle has been placed past
        # the bound — suppressing it is what let the NaN exposure ride.
        stepping.step_B(fields, pml)
        stepping.step_D(fields, pml)
    return {}


def _per_row_needles(mine, reference, axis: int, shape) -> Dict[str, int]:
    """Diverging word counts on the two written planes, reported SEPARATELY.

    An aggregate hides the blind row: the fill-only matrix showed the needle
    MOVES between the near fill and the far fill as the mirror phase flips.
    """
    out: Dict[str, int] = {}
    for label, index in (("near_row_0", 0), ("far_row_last", -1)):
        a = np.ascontiguousarray(_to_host(mine)[_face(axis, index)])
        b = np.ascontiguousarray(_to_host(reference)[_face(axis, index)])
        out[label] = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    return out


def run_fill(results: Dict[str, Any], out_path: str, expansion: int,
             kernel=None, entry_mutation: Optional[str] = None,
             label: str = "gate", quick: bool = False,
             fuse_passes: bool = False) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    grids = ([FILL_GRIDS[i] for i in QUICK_FILL_GRID_INDICES] if quick
             else list(FILL_GRIDS))
    families = STATE_FAMILIES[:2] if quick else STATE_FAMILIES
    total = len(grids) * len(families) * 2
    index = 0
    armed_rows = 0
    discriminating_rows = 0
    started_all = time.time()
    for spec in grids:
        for family in ("B", "D"):
            for state in families:
                index += 1
                started = time.time()
                grid, fields, pml = build_folded_fields(spec, cp, thin_pml=True)
                names = folded_complex.GHOST_FILL_FAMILIES[family]["targets"]
                seed_fill_state(fields, grid, names, state, pml)

                reference = Fields(grid=grid, force_complex_fields=True)
                reference.enable_pml_storage()
                for name in names:
                    getattr(reference, name)[...] = getattr(fields, name)
                if family == "B":
                    stepping.fill_symmetry_bc_B(reference)
                    stepping.fill_folded_far_ghosts_B(reference)
                else:
                    stepping.fill_symmetry_bc_D(reference)
                    stepping.fill_folded_far_ghosts_D(reference)

                entries = [dict(e) for e in
                           folded_complex.ghost_fill_axis_entries(grid, family)]
                entries, armed, discriminating = apply_entry_mutation(
                    entries, entry_mutation, grid)
                plan = folded_complex.plan_folded_mirror_ghost_fill_complex_from_arrays(
                    family, {name: getattr(fields, name) for name in names},
                    entries, expansion, kernel=kernel)
                if fuse_passes:
                    # `_launch_pass(1, 1)` IS the fused per-axis form: ONE launch
                    # per axis doing near AND far, which is what
                    # `symmetry.MirrorGhostFillPlan` may do under real storage.
                    # The module claims the split is load-bearing under complex
                    # storage (folded_complex.py:752-770); this is the needle.
                    # Armed where the entry list has more than one axis; ARMED AND
                    # DISCRIMINATING only where some target is near on one folded
                    # axis and far on another, which is the whole of the question.
                    armed = int(len(entries) > 1)
                    discriminating = fusion_is_discriminating(entries)
                    plan._launch_pass(1, 1, None)
                else:
                    plan.run()
                armed_rows += int(armed)
                discriminating_rows += int(discriminating)
                cp.cuda.runtime.deviceSynchronize()

                per_row = {}
                for entry in folded_complex.ghost_fill_axis_entries(grid, family):
                    axis = int(entry["axis"])
                    per_row[f"axis{axis}"] = {
                        name: _per_row_needles(getattr(fields, name),
                                               getattr(reference, name), axis,
                                               grid.shape)
                        for name in names}
                census = signed_zero_census(
                    cp, {name: getattr(reference, name) for name in names})
                case = {
                    "spec": dict(spec), "family": family, "state": state,
                    "entry_mutation": entry_mutation,
                    "entry_mutation_armed": int(armed),
                    "entry_mutation_discriminating": int(discriminating),
                    "stored": [int(grid.stored_cells(a)) for a in range(3)],
                    "reflect_rows": [None if r is None else int(r)
                                     for r in stepping._far_reflect_rows(grid)],
                    "census": census,
                    "vacuous": bool(census == 0),
                    "per_row_needles": per_row,
                    "verdict": combine({
                        name: bit_compare(getattr(fields, name),
                                          getattr(reference, name))
                        for name in names}),
                    "seconds": round(time.time() - started, 3),
                }
                cases.append(case)
                log(f"case {index}/{total} [fill:{label}] {spec['axis']}/"
                    f"{spec['boundaries']}/phase{_phase_label(spec)}/{family}/"
                    f"{state}: identical={case['verdict']['bit_identical']} "
                    f"differing={case['verdict']['differing_floats']}/"
                    f"{case['verdict']['total_floats']} "
                    f"census={case['census']} armed={armed} "
                    f"discriminating={discriminating} "
                    f"elapsed={round(time.time() - started_all, 1)} s")
                results.setdefault("fill", {})[label] = {
                    "ran": len(cases),
                    "identical": sum(int(c["verdict"]["bit_identical"])
                                     for c in cases),
                    "armed_rows": armed_rows,
                    "discriminating_rows": discriminating_rows,
                    "vacuous_cases": [
                        {"spec": c["spec"], "family": c["family"],
                         "state": c["state"]} for c in cases if c["vacuous"]],
                    "blind_cells": blind_fill_cells(cases),
                    "cases": cases}
                save(results, out_path)
    return results["fill"][label]


def blind_fill_cells(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """(grid, family) cells whose REFERENCE census is 0 in EVERY state family.

    A single family reading 0 is information, not a defect, and the WHOLE matrix
    was re-measured on this laptop (2026-08-12, all 12 grids x 2 families x 4
    states) rather than argued:

    * ``engineered_rows`` reads 58..1347 on EVERY row and is the only family that
      never reads 0 — it is what carries the metallic +1 cells;
    * ``zero_init_absorber`` reads 16..588 except on the two folded METALLIC
      phase = +1 grids, where it reads 0. That one IS a predicted null with a
      reason: the near coefficient is ``+1`` and there is no far fill, so a quiet
      grid driven through the absorber stores no signed zero at all;
    * ``engineered_rows_post_step`` reads 8..400, and 0 on those same metallic +1
      grids, for the same reason;
    * ``live_post_step`` reads 0 on EVERY row of the matrix. That is not a
      metallic-only null and it is not stated as one: four driven steps from a
      random seed leave generic nonzero floats, so this family reaches none of
      the signed-zero class anywhere. It is the REALISTIC-state family — it is
      what catches a wrong reflect row or a wrong plane — and its census is 0 by
      construction, not by configuration.

    A CELL blind in all four families is a different thing: the row cannot see
    any of the fill's needles at all and its coverage claim has nothing behind
    it. That is the granularity the check runs at — an aggregate over the matrix
    would hide it, and a per-case rule would fail every ``live_post_step`` row.
    """
    cells: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        spec = case["spec"]
        key = (f"{spec.get('axis')}/{spec.get('boundaries')}/"
               f"phase{spec.get('phase')}/{tuple(spec.get('cell', ()))}/"
               f"{case['family']}")
        entry = cells.setdefault(key, {"cell": key, "max_census": 0,
                                       "states": {}})
        entry["max_census"] = max(entry["max_census"], int(case["census"]))
        entry["states"][case["state"]] = int(case["census"])
    return [entry for entry in cells.values() if entry["max_census"] == 0]


def _plane_writers(entries: Sequence[Dict[str, Any]]
                   ) -> Tuple[Dict[int, List[Tuple[int, bytes]]],
                              Dict[int, List[Tuple[int, bytes]]]]:
    """``target index -> [(entry index, coefficient BYTES)]``, near and far.

    Every entry carries ``shifts`` — the Yee shift of each of the family's three
    targets on that axis (folded_complex.py:2159) — which is the whole of the
    ownership information the two passes need, so this reads for either family
    without naming a component. The coefficient is compared as BYTES, never as a
    float: this file's entire subject is the words a ``±0.0`` distinguishes.
    """
    near: Dict[int, List[Tuple[int, bytes]]] = {}
    far: Dict[int, List[Tuple[int, bytes]]] = {}
    for index, entry in enumerate(entries):
        for target, shift in enumerate(tuple(entry["shifts"])):
            if shift == 0:
                near.setdefault(target, []).append(
                    (index, np.float32(entry["near_words"]).tobytes()))
            elif entry["far"]:
                far.setdefault(target, []).append(
                    (index, np.float32(entry["far_words"]).tobytes()))
    return near, far


def order_is_discriminating(entries: Sequence[Dict[str, Any]]) -> int:
    """1 when RE-ORDERING ``entries`` can change a stored byte on this row.

    A plane is written twice only when two entries write the SAME target in the
    SAME pass; the corner where those two planes intersect then sees
    ``c_a (x) (c_b (x) z)`` one way round and ``c_b (x) (c_a (x) z)`` the other.
    That composition is bit-exactly equal when ``c_a == c_b``, so a two-folded-axis
    grid carrying ONE declared phase is a provable no-op however many entries it
    has. This is the arming test ``reverse_axis_order`` needs, and "the list
    changed" is NOT it: MEASURED on this laptop over the gate's own fill matrix,
    ``FILL_GRIDS[10]`` (XY, phase +1 on both planes) reverses a TWO-element list —
    ``armed = 1`` — and diverges in 0 words in all 8 of its (family, state) cells,
    while the mixed-phase ``FILL_GRIDS[11]`` diverges in 5 words of ``Bz`` /
    5 of ``Dz`` on ``engineered_rows`` and 4 / 3 on ``engineered_rows_post_step``.
    """
    near, far = _plane_writers(entries)
    for group in list(near.values()) + list(far.values()):
        if len(group) > 1 and len({coefficient for _index, coefficient in group}) > 1:
            return 1
    return 0


def value_change_is_discriminating(before: Sequence[Dict[str, Any]],
                                   after: Sequence[Dict[str, Any]]) -> int:
    """1 when a VALUE mutation changed a word the plan will actually READ.

    ``far_words`` and ``reflect_row`` are DEAD on a folded METALLIC axis: the
    entry's ``far`` is 0, the kernel compiles the whole far arm away
    (folded_complex.py:862), and rewriting them cannot move a byte. MEASURED over
    the gate's own fill matrix: ``far_fill_uses_plus_phase`` changes the entry
    list on all 96 rows and can only differ on the 64 folded-PERIODIC ones — the
    32 metallic rows were armed and empty, the same shape of hole
    ``reverse_axis_order`` had.

    Positional, so it is for VALUE mutations only; an ORDER mutation permutes the
    list and needs :func:`order_is_discriminating`.
    """
    if len(before) != len(after):
        return 1
    for old, new in zip(before, after):
        for key in ("axis", "far", "near_words", "shifts"):
            if np.float32(old[key]).tobytes() != np.float32(new[key]).tobytes():
                return 1
        if not (old["far"] or new["far"]):
            continue                      # the far arm is compiled away here
        for key in ("far_words", "reflect_row"):
            if np.float32(old[key]).tobytes() != np.float32(new[key]).tobytes():
                return 1
    return 0


def fusion_is_discriminating(entries: Sequence[Dict[str, Any]]) -> int:
    """1 when fusing the near and far passes PER AXIS can change a stored byte.

    Only a target that is NEAR on one folded axis and FAR on another sees a
    different order under ``_launch_pass(1, 1)``: the array path gives it
    near-then-far and a fused per-axis launch gives it far-then-near. ``len(entries)
    > 1`` is not the same test — a two-axis grid whose second fold is METALLIC
    runs no far pass there at all, so the question cannot arise on that pair.
    """
    near, far = _plane_writers(entries)
    for target, group in near.items():
        others = far.get(target, ())
        if any(mine != theirs for mine, _c in group for theirs, _d in others):
            return 1
    return 0


def apply_entry_mutation(entries, mutation: Optional[str], grid
                         ) -> Tuple[List[Dict[str, Any]], int, int]:
    """Host-side fill mutations — the ones that live in the ENTRY, not the body.

    Returns ``(entries, armed, discriminating)``, and the two counts answer two
    DIFFERENT questions:

    * ``armed`` is 0 when the mutation left the entry list BYTE-IDENTICAL — the
      host-side twin of a source mutation's ``hits == 0``. ``reverse_axis_order``
      on a single-folded-axis grid reverses a one-element list and measures the
      shipped plan.
    * ``discriminating`` is 0 when the mutated plan CANNOT produce a byte the
      shipped plan does not, whatever the state. For a VALUE mutation the two
      coincide: different words in, different words out. For the ORDER mutation
      they do not, and the gap is exactly where this leg was measuring nothing —
      reversing the two entries of a SAME-PHASE two-folded-axis grid changes the
      list (``armed = 1``) and is a provable no-op (measured 0 diverging words on
      every one of ``FILL_GRIDS[10]``'s eight (family, state) cells). Without the
      second count a run in which every armed row was a no-op records
      ``verdict: MEASURED, diverged: 0`` and passes.
    """
    if mutation is None:
        return entries, 0, 0
    before = [dict(entry) for entry in entries]
    if mutation == "reflect_row_n_minus_two":
        for entry in entries:
            if entry["far"]:
                entry["reflect_row"] = int(grid.stored_cells(entry["axis"])) - 2
    elif mutation == "reverse_axis_order":
        entries = list(reversed(entries))
    elif mutation == "far_fill_uses_plus_phase":
        for entry in entries:
            entry["far_words"] = entry["near_words"]
    elif mutation == "nonzero_imaginary_words":
        # The host-side twin of "the coefficient words are PASSED, never
        # synthesised". Setting the imaginary word to +0.0 (what stood here until
        # 2026-08-12) is a PROVABLE no-op: `mirror_parity_coefficients` already
        # returns `(±1.0, 0.0)`, so the mutated entry list was byte-identical to
        # the shipped one on every row and the leg — predicted CAUGHT — could
        # only ever record diverged = 0. Passing a NONZERO imaginary word is the
        # armed form of the same claim: a kernel that reads the passed word must
        # follow it away from the array path, and one that synthesises `+0.0`
        # cannot.
        for entry in entries:
            entry["near_words"] = (entry["near_words"][0], 0.5)
            entry["far_words"] = (entry["far_words"][0], -0.25)
    else:
        raise ValueError(f"unknown entry mutation {mutation!r}")
    changed = [dict(entry) for entry in entries] != before
    if mutation == "reverse_axis_order":
        discriminating = int(changed) * order_is_discriminating(entries)
    else:
        discriminating = value_change_is_discriminating(before, entries)
    return entries, int(changed), discriminating


# ---------------------------------------------------------------------------
# The ZERO-INIT NEGATIVE-COEFFICIENT-ABSORBER case, with the census
# ---------------------------------------------------------------------------

ZERO_INIT_SPEC = {"axis": "Y", "boundaries": "periodic", "phase": 1,
                  "cell": (2.0, 2.1, 0.0)}
ZERO_INIT_STEPS = 4
#: Laptop expectation to beat on the odd-count periodic +1 configuration.
#: MEASURED on this laptop 2026-08-12 by executing this leg's own build on NumPy
#: (shape (20,13,1), stored [20,13,1], owned [20,12,1], reflect row 10;
#: kms_min x/y = -0.5110714, kms_h_min x/y = -0.0493551 — the absorber's deepest
#: coefficient is genuinely negative). The (162, 60, 162, 60) that stood here
#: until then matched no configuration this leg can build.
ZERO_INIT_LAPTOP_CENSUS = (225, 120, 225, 120)


def run_zero_init(results: Dict[str, Any], out_path: str,
                  expansion: Optional[int] = None) -> Dict[str, Any]:
    """All arrays +0.0; a thin absorber whose deepest ``kms`` goes NEGATIVE.

    TWO halves, and both are required.

    1. REACHABILITY. The census counts ``0x80000000`` words across the stored
       primaries and the six ``fu_*`` volumes after EVERY step and stamps the
       count into the artifact. A census of 0 FAILS the case as vacuous:
       random-seeded states are provably blind to this class, so a quiet zero
       here means the leg measured nothing.
    2. BIT IDENTITY on that state. Until 2026-08-12 this leg ran no kernel at
       all — it stepped the array path and counted signed zeros — so the case the
       discipline names for K1/K2 ("zero-init under a negative-coefficient
       absorber") was never byte-compared for any curl kernel. It is now: after
       the census steps, both curl sub-steps and both ghost fills are run through
       the folded plans on the quiet state and compared word for word against the
       array path continuing on a mirror.
    """
    started = time.time()
    grid, fields, pml = build_folded_fields(ZERO_INIT_SPEC, cp, thin_pml=True)
    for name in ALL_STATE:
        volume = getattr(fields, name, None)
        if volume is not None:
            volume[...] = 0
    kms = {axis: float(_to_host(getattr(pml, f"kms_{axis}")).min())
           for axis in "xyz"}
    kms_h = {axis: float(_to_host(getattr(pml, f"kms_{axis}_h")).min())
             for axis in "xyz"}
    census: List[int] = []
    tracked = {name: getattr(fields, name) for name in ALL_STATE
               if getattr(fields, name, None) is not None}
    for _ in range(ZERO_INIT_STEPS):
        _drive_array_path(fields, pml, 1)
        census.append(signed_zero_census(cp, tracked))
        log(f"[zero_init] step {len(census)}/{ZERO_INIT_STEPS} "
            f"signed_zero_words={census[-1]}")
        results["zero_init"] = {"census": census, "kms_min": kms,
                                "kms_h_min": kms_h}
        save(results, out_path)
    vacuous = any(count == 0 for count in census)
    negative = min(list(kms.values()) + list(kms_h.values())) < 0.0

    byte = _zero_init_bit_identity(grid, fields, pml, expansion)
    verdict = {
        "census": census,
        "laptop_expectation": list(ZERO_INIT_LAPTOP_CENSUS),
        "kms_min": kms, "kms_h_min": kms_h,
        "absorber_has_negative_coefficient": negative,
        "vacuous": vacuous,
        "bit_identity": byte,
        "passed": bool(negative and not vacuous and byte["passed"]),
        "seconds": round(time.time() - started, 3),
    }
    if vacuous:
        verdict["failure"] = ("a census of 0 means the signed-zero class was "
                              "never REACHED; the case is VACUOUS, not passed")
    if not negative:
        verdict["failure"] = ("the absorber's deepest kms = kappa - sigma never "
                              "goes negative, so the quiet state cannot produce "
                              "-0.0 stores; the case measures nothing")
    if not byte["passed"]:
        verdict["failure"] = byte.get(
            "failure", "the kernels are not byte-identical on the quiet state")
    results["zero_init"] = verdict
    save(results, out_path)
    log(f"[zero_init] census={census} negative_coefficient={negative} "
        f"kernels_identical={byte['identical']}/{byte['ran']} "
        f"passed={verdict['passed']}")
    return verdict


def _zero_init_bit_identity(grid, fields, pml,
                            expansion: Optional[int]) -> Dict[str, Any]:
    """K1 (both sub-steps) and K2 (both families) on the QUIET absorber state.

    The leg the case-discipline's ZERO-INIT clause actually asks for: a byte
    comparison at a layer where a signed-zero defect is visible in f32, on a
    state a random seed cannot produce. Substitution is REQUIRED — a refusal is a
    failure here, not a skip, because a leg that silently ran the array path on
    both sides passes vacuously.
    """
    if expansion is None:
        return {"ran": 0, "identical": 0, "passed": False, "cases": [],
                "failure": "no licensed EXPANSION; the leg could launch nothing"}
    mirror = Fields(grid=grid, force_complex_fields=True)
    mirror.enable_pml_storage()
    for name in ALL_STATE:
        source = getattr(fields, name, None)
        if source is not None:
            getattr(mirror, name)[...] = source

    coefficients = {f"{stem}_{axis}{suffix}":
                    getattr(pml, f"{stem}_{axis}{suffix}")
                    for axis in "xyz" for stem in ("kms", "sinv")
                    for suffix in ("", "_h")}
    codes, fold_reasons = folded_complex.folded_axis_kinds(grid, pml)
    cases: List[Dict[str, Any]] = []
    if codes is None:
        return {"ran": 0, "identical": 0, "passed": False, "cases": [],
                "failure": f"the fold could not be classified: {list(fold_reasons)}"}
    phases = (None, None, None)
    for sub_step in CURL_SUB_STEPS:
        getattr(stepping, sub_step)(mirror, pml)
        spec = folded_complex.SUB_STEPS[sub_step]
        targets = tuple(spec["targets"])
        names = targets + tuple("fu_" + t for t in targets) + tuple(spec["sources"])
        suffix = "_h" if sub_step == "step_B" else ""
        flat = {f"{stem}_{axis}":
                cp.ascontiguousarray(
                    coefficients[f"{stem}_{axis}{suffix}"].reshape(-1))
                for axis in "xyz" for stem in ("kms", "sinv")}
        plan = folded_complex.plan_folded_complex_pml_curl_from_arrays(
            sub_step, {n: getattr(fields, n) for n in names}, flat, tuple(codes),
            phases, float(grid.dt / grid.dx), int(expansion))
        plan.run()
        cp.cuda.runtime.deviceSynchronize()
        compared = targets + tuple("fu_" + t for t in targets)
        cases.append({"leg": sub_step, "substituted": True,
                      "verdict": combine({n: bit_compare(getattr(fields, n),
                                                         getattr(mirror, n))
                                          for n in compared})})

        family = "B" if sub_step == "step_B" else "D"
        getattr(stepping, f"fill_symmetry_bc_{family}")(mirror)
        getattr(stepping, f"fill_folded_far_ghosts_{family}")(mirror)
        fill_names = folded_complex.GHOST_FILL_FAMILIES[family]["targets"]
        fill_plan = folded_complex.plan_folded_mirror_ghost_fill_complex_from_arrays(
            family, {n: getattr(fields, n) for n in fill_names},
            folded_complex.ghost_fill_axis_entries(grid, family), int(expansion))
        fill_plan.run()
        cp.cuda.runtime.deviceSynchronize()
        cases.append({"leg": f"fill_{family}", "substituted": True,
                      "verdict": combine({n: bit_compare(getattr(fields, n),
                                                         getattr(mirror, n))
                                          for n in fill_names})})
        update = stepping.update_H if sub_step == "step_B" else stepping.update_E
        update(mirror, pml)
        update(fields, pml)
    identical = sum(int(c["verdict"]["bit_identical"]) for c in cases)
    out = {"ran": len(cases), "identical": identical,
           "passed": bool(cases) and identical == len(cases), "cases": cases}
    if not out["passed"]:
        out["failure"] = ("a folded kernel is not byte-identical to the array "
                          "path on the zero-init negative-coefficient-absorber "
                          "state")
    return out


# ---------------------------------------------------------------------------
# The identity / reduction products
# ---------------------------------------------------------------------------

def run_identity(results: Dict[str, Any], out_path: str,
                 expansion: int) -> Dict[str, Any]:
    """Zero folded axes -> K1 == the certified unfolded kernel, byte for byte;
    ``HAS_BETA=0`` -> K3a/K3b == the plain folded kernels. Equivalence products,
    never routing rules: which kernel a composed plan would launch is a separate
    decision this leg deliberately does not make."""
    cases: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 555)
    shape = (9, 7, 5)
    for sub_step in CURL_SUB_STEPS:
        spec = folded_complex.SUB_STEPS[sub_step]
        targets = tuple(spec["targets"])
        names = targets + tuple("fu_" + t for t in targets) + tuple(spec["sources"])
        host = {name: gate._seed_host_complex(shape, rng) for name in names}
        coefficients = _coefficients(shape, rng)
        flat = _flat_coefficients(cp, coefficients, shape)

        # (a) zero folded axes: K1 against complex_fields.bloch_pml_curl_step.
        mine = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
        theirs = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
        codes = (CODE_OF[PERIODIC], CODE_OF[METALLIC], CODE_OF[PERIODIC])
        phases = (_phase(0.19), None, None)
        folded_complex.plan_folded_complex_pml_curl_from_arrays(
            sub_step, mine, flat, codes, phases, 0.35, expansion).run()
        complex_fields.plan_complex_pml_curl_from_arrays(
            sub_step, theirs, flat, codes, phases, 0.35, expansion).run()
        cp.cuda.runtime.deviceSynchronize()
        cases.append({
            "leg": "zero_folded_axes_reduces_to_certified_complex",
            "sub_step": sub_step,
            "verdict": combine({n: bit_compare(mine[n], theirs[n])
                                for n in targets + tuple("fu_" + t for t in targets)}),
            "predicted": "NULL (identical) — the constexpr branches reduce exactly",
        })

        # (b) HAS_BETA=0 on K3b against the plain folded complex kernel.
        mine = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
        theirs = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
        fold_codes = (CODE_OF[PERIODIC], CODE_OF[MIRROR_PERIODIC], CODE_OF[PERIODIC])
        words = special_kz.beta_curl_coefficients(
            BETA_GRATING_13_2, BETA_DT, magnetic=(sub_step == "step_B"),
            complex_storage=True)
        folded_complex.plan_folded_beta_bloch_pml_curl_from_arrays(
            sub_step, mine, flat, fold_codes, (None, None, None), 0.35, expansion,
            words, has_beta=0).run()
        folded_complex.plan_folded_complex_pml_curl_from_arrays(
            sub_step, theirs, flat, fold_codes, (None, None, None), 0.35,
            expansion).run()
        cp.cuda.runtime.deviceSynchronize()
        cases.append({
            "leg": "has_beta_zero_reduces_to_plain_folded_complex",
            "sub_step": sub_step,
            "verdict": combine({n: bit_compare(mine[n], theirs[n])
                                for n in targets + tuple("fu_" + t for t in targets)}),
            "predicted": "NULL (identical) — the term is compiled away",
        })

        # (c) HAS_BETA=0 on K3a against symmetry's certified folded real kernel.
        real_host = {name: _seed_host_real(shape, rng) for name in names}
        mine = {n: cp.asarray(np.array(v, copy=True)) for n, v in real_host.items()}
        theirs = {n: cp.asarray(np.array(v, copy=True)) for n, v in real_host.items()}
        plus, minus = special_kz.beta_curl_coefficients(
            BETA_GRATING_13_2, BETA_DT, magnetic=(sub_step == "step_B"),
            complex_storage=False)
        folded_complex.plan_folded_beta_pml_curl_from_arrays(
            sub_step, mine, flat, fold_codes, 0.35, plus, minus, has_beta=0).run()
        from meep_gpu.triton_kernels import symmetry as symmetry_module  # noqa: PLC0415

        symmetry_module.plan_folded_from_arrays(
            sub_step, theirs, flat, fold_codes, 0.35).run()
        cp.cuda.runtime.deviceSynchronize()
        cases.append({
            "leg": "has_beta_zero_reduces_to_symmetry_folded_real",
            "sub_step": sub_step,
            "verdict": combine({n: bit_compare(mine[n], theirs[n])
                                for n in targets + tuple("fu_" + t for t in targets)}),
            "predicted": "NULL (identical) — K3a is symmetry's kernel plus a "
                         "compiled-away term",
        })
        for case in cases[-3:]:
            log(f"[identity] {case['leg']} {sub_step}: "
                f"identical={case['verdict']['bit_identical']} "
                f"differing={case['verdict']['differing_floats']}")
        results["identity"] = {"ran": len(cases),
                               "identical": sum(int(c["verdict"]["bit_identical"])
                                                for c in cases),
                               "cases": cases}
        save(results, out_path)
    return results["identity"]


# ---------------------------------------------------------------------------
# Mutations — source-level (compiled, launch-counted, PTX-checked) and host-level
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def shipped_source(which: str) -> str:
    """Helpers first (they may themselves be the mutation target), then kernels."""
    import inspect  # noqa: PLC0415

    helpers = (complex_fields._rotate_field_left,
               complex_fields._mul_field_left,
               complex_fields._mul_coefficient_left,
               special_kz._mul_imag_coefficient_left)
    bodies = {
        "curl": (folded_complex.folded_bloch_pml_curl_step,),
        "fill": (folded_complex.folded_mirror_ghost_fill_complex,),
        "beta_real": (folded_complex.folded_beta_pml_curl_step,),
        "beta_complex": (folded_complex.folded_beta_bloch_pml_curl_step,),
    }[which]
    functions = helpers + bodies
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn)) for f in functions)


def compile_mutated(source: str, kernel_name: str):
    """Compile a mutated copy from a real file on disk (Triton reads source via
    ``inspect``, so exec'd text raises at first launch). The constexpr codes are
    re-declared in the header because a ``@triton.jit`` body may not read a plain
    module global."""
    header = (
        "import triton\nimport triton.language as tl\n"
        f"PERIODIC = tl.constexpr({CODE_OF[PERIODIC]})\n"
        f"METALLIC = tl.constexpr({CODE_OF[METALLIC]})\n"
        f"MIRROR_METALLIC = tl.constexpr({CODE_OF[MIRROR_METALLIC]})\n"
        f"MIRROR_PERIODIC = tl.constexpr({CODE_OF[MIRROR_PERIODIC]})\n"
        f"NAIVE = tl.constexpr(0)\nFMA_V1 = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_folded.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


class CountingKernel(gate.CountingKernel):
    """Proof the mutant actually launched, and that its PTX is not the shipped
    one served from Triton's cache (platform fact: the cache can hand a stale
    binary to a renamed-but-similar entry point)."""

    def ptx_signatures(self) -> List[str]:
        out: List[str] = []
        cache = getattr(self.kernel, "cache", {})
        for device_cache in (cache.values() if isinstance(cache, dict) else ()):
            for compiled in getattr(device_cache, "values", lambda: ())():
                asm = getattr(compiled, "asm", None)
                if isinstance(asm, dict) and "ptx" in asm:
                    out.append(str(hash(asm["ptx"])))
        return out


def _sub(pattern: str, replacement: str):
    needle = re.compile(pattern)

    def apply(source: str) -> Tuple[str, int]:
        return needle.sub(replacement, source), len(needle.findall(source))

    return apply


#: The fill's product call, with its own indentation captured — the near arm sits
#: at one level and the far arm one deeper inside ``if DO_FAR:``, so a mutant that
#: hard-codes the indentation produces an IndentationError and DISARMS the leg.
_FILL_PRODUCT = re.compile(
    r"( *)o_re, o_im = _mul_imag_coefficient_left\((near|far)_re, (?:near|far)_im, "
    r"z_re, z_im,\s*\n\s*EXPANSION\)")


def mutate_plane_wise_parity_fill(source: str) -> Tuple[str, int]:
    """m1, THE headline needle: the fill's full complex product becomes
    ``PHASE * word`` — ``symmetry.mirror_ghost_fill``'s certified REAL spelling
    ported word-for-word. The parity's real word IS ±1.0, so the mutant spells it
    against ``near_re`` / ``far_re``."""
    def replace(match: "re.Match[str]") -> str:
        indent, prefix = match.group(1), match.group(2)
        return (f"{indent}o_re = {prefix}_re * z_re\n"
                f"{indent}o_im = {prefix}_re * z_im")

    return _FILL_PRODUCT.sub(replace, source), len(_FILL_PRODUCT.findall(source))


def mutate_identity_shortcut(source: str) -> Tuple[str, int]:
    """m2: the ``phase == +1`` case short-circuited to a plain copy. Run on the
    +1 rows ONLY, which is where it would bite — and where every corpus row is."""
    def replace(match: "re.Match[str]") -> str:
        indent = match.group(1)
        return f"{indent}o_re = z_re\n{indent}o_im = z_im"

    return _FILL_PRODUCT.sub(replace, source), len(_FILL_PRODUCT.findall(source))


def mutate_synthesize_zero_imag(source: str) -> Tuple[str, int]:
    """m3: the coefficient's imaginary word built in-kernel as a literal +0.0
    instead of the host word that was passed."""
    hits = 0
    for prefix in ("near", "far"):
        pattern = re.compile(prefix + r"_re, " + prefix + r"_im, z_re, z_im")
        source, n = pattern.subn(prefix + "_re, 0.0, z_re, z_im", source)
        hits += n
    return source, hits


mutate_fold_ghost_wraps = _sub(
    r"if BC([XYZ]) == PERIODIC:", r"if BC\1 != METALLIC:")
mutate_drop_top_plane_mask = _sub(
    r"if BC([XYZ]) == MIRROR_PERIODIC:\n(\s+)curl(\d)(_re)? = tl\.where\(last_[xyz], 0\.0, curl\3\4\)",
    r"if False:\n\2curl\3\4 = curl\3\4")
mutate_mask_only_metallic = _sub(r"!= PERIODIC", "== METALLIC")
def mutate_fold_zero_cross_terms(source: str) -> Tuple[str, int]:
    """m7: the zero cross terms folded to a literal ``+0.0`` — exactly the
    constant-fold the compiler must not perform, byte-wrong on signed zeros only.

    BOTH arms are rewritten, not just FMA_V1: which arm the platform binds is a
    probe result, and a mutation that only touches the arm this laptop measured
    would be DISARMED on a NAIVE platform without saying so.

    BOTH helpers are rewritten too. ``_mul_field_left`` carries the cross terms
    of ``fu *= kms`` / ``field *= sinv_u``; ``_mul_coefficient_left`` carries the
    zero-sign-asymmetric ones of the scalar-left ``dtdx * total`` (S:1635), whose
    left zero passes the FIELD's imaginary sign. Leaving the second helper
    unmutated left the curl's own scale factor's cross terms unarmed on every
    row.
    """
    hits = 0
    for pattern, replacement in (
            # _mul_field_left — the field-on-the-left cross terms
            (r"\(z_im \* 0\.0\) \* -1\.0", "0.0"),
            (r"tl\.math\.fma\(z_re, 0\.0, z_im \* c\)", "z_im * c"),
            (r"\(z_re \* c\) - \(z_im \* 0\.0\)", "z_re * c"),
            (r"\(z_re \* 0\.0\) \+ \(z_im \* c\)", "z_im * c"),
            # _mul_coefficient_left — the coefficient-on-the-left cross terms
            (r"\(0\.0 \* z_im\) \* -1\.0", "0.0"),
            (r"tl\.math\.fma\(c, z_im, 0\.0 \* z_re\)", "c * z_im"),
            (r"\(c \* z_re\) - \(0\.0 \* z_im\)", "c * z_re"),
            (r"\(c \* z_im\) \+ \(0\.0 \* z_re\)", "c * z_im")):
        source, n = re.subn(pattern, replacement, source)
        hits += n
    return source, hits
def mutate_unary_minus_addend(source: str) -> Tuple[str, int]:
    """m11: a negated addend respelled through unary minus, in BOTH arms.

    Platform fact: Triton 3.1.0 lowers ``-x`` as ``0.0 - x``, canonicalizing ±0
    addends to +0 before the add — MEASURED REACHABLE at the driver level,
    which is why the shipped helpers spell it ``(a*b) * -1.0``.

    The FMA_V1 arm carries the ``* -1.0`` spelling and is respelled ``-(a*b)``.
    The NAIVE arm has no ``* -1.0`` at all — it spells the same term as a plain
    SUBTRACT, which does NOT lose the addend's zero sign — so arming it means
    turning ``a - b`` into ``a + (-b)``, the same lowering by another route.
    Until 2026-08-12 this transform touched the FMA arm only: on a platform whose
    probe binds NAIVE the mutant's SOURCE differed (so the ``hits == 0`` DISARMED
    path did not fire) while its compiled body was identical to the shipped one,
    ``_ptx_differs`` reported overlap, and the leg failed for a non-defect.
    """
    hits = 0
    for pattern, replacement in (
            # FMA_V1 arms: (a*b) * -1.0  ->  -(a*b)
            (r"\(c_im \* z_im\) \* -1\.0", "-(c_im * z_im)"),
            (r"\(z_im \* 0\.0\) \* -1\.0", "-(z_im * 0.0)"),
            (r"\(g_im \* p_im\) \* -1\.0", "-(g_im * p_im)"),
            (r"\(0\.0 \* z_im\) \* -1\.0", "-(0.0 * z_im)"),
            # NAIVE arms: a - b  ->  a + (-b)
            (r"\(c_re \* z_re\) - \(c_im \* z_im\)",
             "(c_re * z_re) + (-(c_im * z_im))"),
            (r"\(z_re \* c\) - \(z_im \* 0\.0\)",
             "(z_re * c) + (-(z_im * 0.0))"),
            (r"\(g_re \* p_re\) - \(g_im \* p_im\)",
             "(g_re * p_re) + (-(g_im * p_im))"),
            (r"\(c \* z_re\) - \(0\.0 \* z_im\)",
             "(c * z_re) + (-(0.0 * z_im))")):
        source, n = re.subn(pattern, replacement, source)
        hits += n
    return source, hits
mutate_rotate_beta_partner = _sub(
    r"_mul_imag_coefficient_left\(bp_re, bp_im, b_re, b_im, EXPANSION\)",
    "_mul_imag_coefficient_left(bp_re, bp_im, b_x_re, b_x_im, EXPANSION)")


def mutate_beta_after_mask(source: str) -> Tuple[str, int]:
    """m8, THE fold-order needle: the beta insert moved BELOW both ownership
    masks, so a live increment survives on the mirror plane and on the far ghost
    plane while the interior stays byte-identical."""
    complex_block = (
        "    if HAS_BETA:\n"
        "        t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)\n"
        "        curl0_re = curl0_re - t_re\n"
        "        curl0_im = curl0_im - t_im\n"
        "        t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)\n"
        "        curl1_re = curl1_re - t_re\n"
        "        curl1_im = curl1_im - t_im\n")
    real_block = (
        "    if HAS_BETA:\n"
        "        curl0 = curl0 - (beta_plus * b)\n"
        "        curl1 = curl1 - (beta_minus * a)\n")
    for block in (complex_block, real_block):
        if block in source:
            source = source.replace(block, "", 1)
            anchor = "    # --- split-field recurrence"
            index = source.index(anchor)
            return source[:index] + block + source[index:], 1
    return source, 0


#: name -> (transform, which shipped body it applies to, entry point, note).
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str, str, str]] = {
    "plane_wise_parity_fill": (
        mutate_plane_wise_parity_fill, "fill",
        "folded_mirror_ghost_fill_complex",
        "THE headline needle; predicted counts to beat: 8/128 engineered words "
        "at BOTH mirror parities (folded_complex.py:64-66), 40 words on the live "
        "state. The 8-at-both-parities prediction is only reachable because the "
        "engineered plant is FILL_NEEDLE_PAIRS — the measured union of the three "
        "divergent classes — rather than a diagonal walk over a word list"),
    "identity_shortcut_on_even_mirror": (
        mutate_identity_shortcut, "fill", "folded_mirror_ghost_fill_complex",
        "the even mirror is NOT the identity under complex storage; measured "
        "across every row, and the +1 rows are the load-bearing ones because "
        "every corpus anchor uses an even mirror. The +1 rows only carry the "
        "needle now that plant_fill_needles stamps FILL_NEEDLE_PAIRS: the "
        "diagonal pairing it replaced hit 0 of the 8 divergent pairs at phase "
        "+1, so this leg's CAUGHT verdict came entirely off the phase = -1 sign "
        "flip and its stated coverage claim had nothing behind it"),
    "synthesize_zero_imag": (
        mutate_synthesize_zero_imag, "fill", "folded_mirror_ghost_fill_complex",
        "RETIRED as a RECORDED STRUCTURAL NULL on 2026-08-13 — still armed, "
        "still compiled, still launched, and judged against "
        "SOURCE_MUTATION_NULLS rather than against a divergence count; the live "
        "source-layer claim is carried by the host twin "
        "imaginary_words_are_read_not_synthesised"),
    "fold_ghost_wraps": (
        mutate_fold_ghost_wraps, "curl", "folded_bloch_pml_curl_step",
        "predicted CAUGHT on MIRROR_METALLIC only"),
    "drop_top_plane_mask": (
        mutate_drop_top_plane_mask, "curl", "folded_bloch_pml_curl_step",
        "predicted CAUGHT on MIRROR_PERIODIC only"),
    "mask_only_metallic": (
        mutate_mask_only_metallic, "curl", "folded_bloch_pml_curl_step",
        "the cell-0 mask narrowed back to == METALLIC"),
    "fold_zero_cross_terms": (
        mutate_fold_zero_cross_terms, "curl", "folded_bloch_pml_curl_step",
        "a 0.0*t cross term folded to a literal +0.0"),
    "unary_minus_addend": (
        mutate_unary_minus_addend, "fill", "folded_mirror_ghost_fill_complex",
        "platform fact: Triton lowers -x as 0.0-x and canonicalizes signed "
        "zeros; MEASURED REACHABLE at the driver level"),
    "beta_after_mask": (
        mutate_beta_after_mask, "beta_complex",
        "folded_beta_bloch_pml_curl_step",
        "THE fold-order needle; predicted caught on the mirror plane and the far "
        "ghost plane of every folded axis, on both sub-steps and both targets"),
    "beta_after_mask_real": (
        mutate_beta_after_mask, "beta_real", "folded_beta_pml_curl_step",
        "the real-storage twin of the fold-order needle"),
    "rotate_beta_partner": (
        mutate_rotate_beta_partner, "beta_complex",
        "folded_beta_bloch_pml_curl_step",
        "the Bloch rotation applied to the beta CENTRE partner"),
}

#: SOURCE mutations retired as RECORDED STRUCTURAL NULLS: name -> the record.
#:
#: THE RULE. A mutation that cannot diverge on any grid this gate builds is a
#: predicted null, and it must be recorded WITH its reason and its evidence, or
#: re-aimed at a state where the defect is byte-visible. Two things it may NOT
#: become: deleted (the table would show sixteen mutations and a disappearance,
#: and nothing would record that the seventeenth was ever asked), or silently
#: flipped to "predicted not-caught" (an assertion in place of a measurement).
#:
#: A null is judged, not skipped. It is still transformed, still compiled, still
#: LAUNCH-COUNTED and still PTX-checked, so all three DISARMED paths stay live —
#: a null whose mutant never ran has not been shown to be a null, it has been
#: shown to be unmeasured. Only the divergence clause changes.
#:
#: Each record carries the three things an auditor needs:
#:
#: * ``reason`` — why NO state can catch it, stated as a property of the SHIPPED
#:   code rather than of this gate's grids or needles. "our states are too weak"
#:   is a blind configuration and belongs in NEEDLE-MISSED; "mutant and shipped
#:   receive bit-identical operands" is a null.
#: * ``twin`` — the mutation that still carries the same source-layer claim, so
#:   the retirement can be checked to subtract no coverage. A null with no twin
#:   and no re-aim is a coverage LOSS and must be argued as one.
#: * ``measured`` — the numbers the retirement rests on, off a named run. The
#:   retirement is evidence, not assertion: the mutant DID run and DID compile
#:   to different PTX, and still moved nothing.
#:
#: SELF-INVALIDATION, in two places, because a null's vacuity is a property of
#: the shipped code and shipped code changes:
#:
#: 1. HERE — a predicted null that DIVERGES is a ``NULL-INVALIDATED`` failure,
#:    never a quiet CAUGHT. Divergence means the recorded reason is false, so the
#:    artifact is asserting something untrue about its own coverage table.
#: 2. ON THE LAPTOP — ``premise_test`` names a test that asserts the structural
#:    property DIRECTLY. If the premise ever stops holding, that test fails on a
#:    machine with no GPU, instead of a dead mutation sitting in the table
#:    claiming coverage it cannot deliver until someone next schedules a device
#:    run. ``build_summary`` refuses a declared null that the run never executed,
#:    which is the third way this could go quiet.
SOURCE_MUTATION_NULLS: Dict[str, Dict[str, Any]] = {
    "synthesize_zero_imag": {
        "reason": (
            "VACUOUS BY CONSTRUCTION, and by a property of the shipped HOST code "
            "rather than of this gate's states: the mutant rewrites the kernel's "
            "near_im / far_im argument to a literal 0.0, while "
            "folded_complex.mirror_parity_coefficients (folded_complex.py:2100) "
            "returns numpy.complex64(+-1), whose IMAGINARY WORD IS BITWISE "
            "0x00000000 for both mirror phases and for both the near and the far "
            "coefficient — not merely equal to zero in value, but the identical "
            "32-bit pattern, so not even a -0.0 sign survives to be read. Mutant "
            "and shipped kernel therefore receive BIT-IDENTICAL operands and no "
            "grid, no state family and no needle can drive them apart. This is "
            "NOT a blind configuration to be repaired with a better needle: "
            "there is no state to find."),
        "twin": "imaginary_words_are_read_not_synthesised",
        "twin_claim": (
            "the same source-layer claim — the kernel READS the passed imaginary "
            "word and does not synthesise it — aimed at a state where the defect "
            "IS byte-visible. The host passes a NONZERO imaginary word "
            "(apply_entry_mutation's 'nonzero_imaginary_words', +0.5 near / "
            "-0.25 far); a kernel that reads it must follow the passed word away "
            "from the array path and one that synthesises +0.0 cannot. CAUGHT on "
            "run direct_20260813T084108Z: armed 96 / discriminating 96 / "
            "diverged 82. Re-aiming this source mutation would therefore "
            "duplicate live coverage, which is why retiring it with proof is the "
            "resolution and not the second-best one."),
        "measured": {
            "run": "direct_20260813T084108Z",
            "device": "one RTX A6000, pinned GPU 6",
            "hits": 6,
            "launches": 192,
            "ptx_differs": True,
            "mutant_specializations": 20,
            "shipped_specializations": 24,
            "overlapping_specializations": 0,
            "fill_ran": 96,
            "fill_identical": 96,
            "diverged_cases": 0},
        "premise_test": (
            "meep_gpu/test_triton_folded_complex.py::"
            "test_the_parity_coefficients_imaginary_word_is_bitwise_zero"),
    },
}

#: Host-side mutations (the entry list, the plan arguments, the constexprs).
#: value = predicted verdict; None means "record the measurement, do not judge".
HOST_MUTATIONS: Dict[str, Optional[bool]] = {
    "reflect_row_n_minus_two": True,     # caught at ODD n_full only
    # The shift-1 sign inversion dropped. Armed on all 96 fill rows and
    # DISCRIMINATING on 64: a folded METALLIC axis runs no far pass, so its
    # `far_words` is dead and rewriting it is a provable no-op — the same shape
    # of hole `reverse_axis_order` had, measured the same way.
    "far_fill_uses_plus_phase": True,
    # The kernel must READ the passed imaginary word. Its predecessor
    # ("synthesize_zero_imag_host") set that word to +0.0, which is what the host
    # already passes — a provable no-op recorded as a predicted CAUGHT.
    "imaginary_words_are_read_not_synthesised": True,
    # The two-pass split's OWN needle. The module states the split is
    # load-bearing under complex storage and cites a measured 1-word By
    # divergence; nothing in the gate exercised it. `_launch_pass(1, 1)` is the
    # fused per-axis form. MEASURED on the laptop from the gate's own state
    # families, over the whole fill matrix: 5 words of By / 5 of Dx on
    # `engineered_rows` and 3 / 4 on `engineered_rows_post_step`, on the
    # two-folded-axis grids; 0 on every one-axis grid and 0 on
    # `zero_init_absorber` and `live_post_step`, where the plants that carry the
    # class are absent. The row count is `fusion_is_discriminating`, not
    # `len(entries) > 1`: a second fold that is METALLIC runs no far pass, so no
    # target is near on one axis and far on another and the question cannot arise.
    "fuse_fill_passes": True,
    # A MEASUREMENT, not a null — but ONLY on the MIXED-PHASE two-folded-axis
    # grid, and the leg is judged on `discriminating_rows` for that reason.
    # MEASURED on the laptop over the gate's own fill matrix: FILL_GRIDS[10] (XY,
    # phase +1 on both planes) is ARMED on all 8 of its (family, state) cells and
    # diverges in 0 words on every one of them — reversing a list is not the same
    # as being able to change a byte. FILL_GRIDS[11] (phase +1 on X, -1 on Y) is
    # the row that measures: 5 words of Bz / 5 of Dz on `engineered_rows`, 4 / 3
    # on `engineered_rows_post_step`, 0 on `zero_init_absorber` and
    # `live_post_step`. Bz is FAR on both folded axes and Dz NEAR on both, so
    # those are the two doubly-written corners on this family pair.
    "reverse_axis_order": None,
    "phase_on_folded_axis": None,        # CAUGHT on metallic, NULL on periodic
}


def _ptx_differs(mutant: Any, shipped: Any) -> Dict[str, Any]:
    """Platform fact (c): Triton's cache can serve a STALE binary to a
    renamed-but-similar entry point, so a mutant whose PTX matches a shipped
    specialization has not been measured at all."""
    def signatures(kernel) -> List[str]:
        out: List[str] = []
        cache = getattr(kernel, "cache", {})
        if isinstance(cache, dict):
            for device_cache in cache.values():
                items = getattr(device_cache, "values", None)
                for compiled in (items() if callable(items) else ()):
                    asm = getattr(compiled, "asm", None)
                    if isinstance(asm, dict) and "ptx" in asm:
                        out.append(str(hash(asm["ptx"])))
        return out

    mine = signatures(mutant)
    theirs = signatures(shipped)
    overlap = sorted(set(mine) & set(theirs))
    return {"mutant_specializations": len(mine),
            "shipped_specializations": len(theirs),
            "overlapping": overlap,
            "differs": bool(mine) and not overlap}


#: The measured note ``reverse_axis_order`` carries. Kept beside the verdict
#: function so the numbers and the rule that uses them cannot drift apart.
REVERSE_AXIS_ORDER_NOTE = (
    "reverse_axis_order is a MEASUREMENT here, not a null: the real-storage "
    "commute argument (every fill is a multiply by exactly +-1) does not "
    "transfer to complex, where the product is full and float multiplication is "
    "not associative. It is a measurement only on the MIXED-PHASE "
    "two-folded-axis grid, which is why the leg is judged on "
    "discriminating_rows and NOT on armed_rows: with one declared phase per grid "
    "the two planes carry the SAME coefficient, the composition commutes "
    "bit-exactly, and reversing the list is armed and empty at once. MEASURED on "
    "the laptop over the gate's own fill matrix — FILL_GRIDS[10] (XY, phase +1 "
    "on both planes) armed on 8 of 8 (family, state) cells and 0 diverging words "
    "on every one of them; FILL_GRIDS[11] (+1 on X, -1 on Y) 5 words of Bz / 5 "
    "of Dz on engineered_rows, 4 / 3 on engineered_rows_post_step, 0 on "
    "zero_init_absorber and live_post_step. Bz is far on both folded axes and Dz "
    "near on both, so those are the two doubly-written corners.")

PHASE_ON_FOLDED_AXIS_NOTE = (
    "phase_on_folded_axis: predicted CAUGHT on MIRROR_METALLIC (the top plane is "
    "unmasked there) and NULL on MIRROR_PERIODIC (both consumers of that plane "
    "are top-plane masked)")


def host_mutation_verdict(name: str, predicted: Optional[bool],
                          summary: Dict[str, Any], diverged: int
                          ) -> Dict[str, Any]:
    """The host-mutation verdict, as a pure function of the counts it reads.

    Extracted from :func:`run_mutations` so every branch can be EXECUTED on a
    laptop with no device: the ordering of these clauses is the whole repair, and
    a text-order assertion pins the source rather than the behaviour.

    The order is load-bearing and each clause exists because a leg was measured
    to be measuring nothing:

    1. ``armed_rows == 0`` — the entry list never changed. The host twin of a
       source mutation's ``hits == 0``.
    2. ``discriminating_rows == 0`` — it changed, and no changed row CAN store a
       byte the shipped plan does not. ARMED IS NOT ARMING: reversing the two
       entries of a SAME-PHASE two-folded-axis grid is armed and is a provable
       no-op (measured 0 diverging words on all eight of ``FILL_GRIDS[10]``'s
       (family, state) cells).
    3. discriminating rows and NOTHING diverged — NEEDLE-MISSED, and this applies
       to a ``predicted is None`` leg too. That is the vacuity path
       ``reverse_axis_order`` did not have: it reached ``verdict: MEASURED``
       before any divergence check, so a run in which nothing was varied recorded
       a measurement with a coverage note behind it.
    4. only then is a ``predicted is None`` leg a MEASURED record.
    """
    armed = summary.get("armed_rows")
    discriminating = summary.get("discriminating_rows")
    out: Dict[str, Any] = {"armed_rows": armed,
                           "discriminating_rows": discriminating}
    if armed is not None and armed == 0:
        out["verdict"] = "DISARMED"
        out["failure"] = ("the entry mutation left the plan's entry list "
                          "BYTE-IDENTICAL on every row; the leg measured the "
                          "shipped plan")
    elif discriminating is not None and discriminating == 0:
        out["verdict"] = "DISARMED"
        out["failure"] = (
            f"the entry mutation changed the list on {armed} rows but NONE of "
            f"them can store a byte the shipped plan does not; every armed row "
            f"is a provable no-op and the leg measured nothing")
    elif discriminating and diverged == 0:
        out["verdict"] = "NEEDLE-MISSED"
        out["failure"] = (
            f"launched on {discriminating} rows where the mutated plan CAN store "
            f"a different byte by construction, and nothing diverged: this "
            f"configuration is BLIND to the defect and its coverage claim is "
            f"withdrawn")
    elif predicted is None:
        out["verdict"] = "MEASURED"
        out["note"] = (REVERSE_AXIS_ORDER_NOTE if name == "reverse_axis_order"
                       else PHASE_ON_FOLDED_AXIS_NOTE)
    elif diverged == 0:
        out["verdict"] = "NEEDLE-MISSED"
        out["failure"] = "launched but nothing diverged"
    else:
        out["verdict"] = "CAUGHT"
    return out


def source_mutation_verdict(name: str, hits: int, launches: Optional[int],
                            ptx_differs: Optional[bool],
                            diverged: Optional[int]) -> Dict[str, Any]:
    """The SOURCE-mutation verdict, as a pure function of the counts it reads.

    Extracted from :func:`run_mutations` for the same reason
    :func:`host_mutation_verdict` was: every branch — including the two the
    predicted-null policy adds — must be EXECUTABLE on a laptop with no device,
    or the policy is prose that only a scheduled GPU run can test.

    The clause order is load-bearing:

    1. ``hits == 0`` — the transform matched nothing. The shipped kernel was
       measured against itself.
    2. ``launches == 0`` — it compiled and never ran.
    3. the mutant's PTX matches a shipped specialization — Triton's cache served
       a stale binary and nothing was measured.

    All three DISARM, and they run BEFORE the null clause on purpose: a declared
    null whose mutant never ran has not been shown to be a null, only to be
    unmeasured, and letting it report PREDICTED-NULL would launder one into the
    other.

    4. a name in :data:`SOURCE_MUTATION_NULLS` that diverged in NOTHING is
       ``PREDICTED-NULL`` — armed, compiled, launched, PTX-distinct and provably
       unable to move a byte, recorded with its structural reason, its live twin
       and the run its numbers came off.
    5. that same name diverging in ANYTHING is ``NULL-INVALIDATED`` and FAILS.
       Divergence falsifies the recorded reason, so the retirement rests on a
       claim the run just contradicted; the honest response is to stop the gate,
       not to relabel the leg CAUGHT and leave the false reason in the table.
    6. anything else with zero divergence is ``NEEDLE-MISSED`` — the blind
       configuration, whose coverage claim is withdrawn.
    """
    out: Dict[str, Any] = {}
    null = SOURCE_MUTATION_NULLS.get(name)
    if hits == 0:
        out["verdict"] = "DISARMED"
        out["failure"] = ("the mutation matched NOTHING in the shipped source; "
                          "the leg measured the shipped kernel")
        return out
    if not launches:
        out["verdict"] = "DISARMED"
        out["failure"] = "the mutant kernel never launched"
        return out
    if not ptx_differs:
        out["verdict"] = "DISARMED"
        out["failure"] = ("the mutant's PTX matches a shipped specialization; "
                          "Triton served a stale binary and nothing was "
                          "measured")
        return out
    if null is not None:
        out["null_reason"] = null["reason"]
        out["live_claim_carried_by"] = null["twin"]
        out["twin_claim"] = null["twin_claim"]
        out["retirement_evidence"] = null["measured"]
        out["premise_test"] = null["premise_test"]
        if diverged:
            out["verdict"] = "NULL-INVALIDATED"
            out["failure"] = (
                f"{name} is RECORDED as a predicted structural null, and it "
                f"diverged in {diverged} cases: the recorded reason "
                f"({null['reason'][:60]}...) is now FALSE, so the retirement and "
                f"the coverage table rest on a claim this run contradicts. "
                f"Re-arm it as a live mutation or restate the null — do not "
                f"relabel this CAUGHT. The premise is asserted directly by "
                f"{null['premise_test']}, which should be failing too")
        else:
            out["verdict"] = "PREDICTED-NULL"
        return out
    if not diverged:
        out["verdict"] = "NEEDLE-MISSED"
        out["failure"] = ("launched, PTX differs, zero diverging words: this "
                          "configuration is BLIND to the defect and its "
                          "coverage claim is withdrawn")
        return out
    out["verdict"] = "CAUGHT"
    return out


def run_mutations(results: Dict[str, Any], out_path: str, expansion: int,
                  quick: bool = False) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    index = 0
    total = len(SOURCE_MUTATIONS) + len(HOST_MUTATIONS)
    started_all = time.time()

    for name, (transform, which, entry, note) in SOURCE_MUTATIONS.items():
        index += 1
        started = time.time()
        source, hits = transform(shipped_source(which))
        record: Dict[str, Any] = {"kind": "source", "body": which, "hits": hits,
                                  "note": note}
        if name in SOURCE_MUTATION_NULLS:
            record["predicted_null"] = True
        if hits == 0:
            record.update(source_mutation_verdict(name, hits, None, None, None))
        else:
            mutant = compile_mutated(source, entry)
            counting = CountingKernel(mutant)
            shipped = {"fill": folded_complex.folded_mirror_ghost_fill_complex,
                       "curl": folded_complex.folded_bloch_pml_curl_step,
                       "beta_real": folded_complex.folded_beta_pml_curl_step,
                       "beta_complex":
                           folded_complex.folded_beta_bloch_pml_curl_step}[which]
            if which == "fill":
                summary = run_fill(results, out_path, expansion,
                                   kernel=counting, label=name, quick=quick)
                diverged = summary["ran"] - summary["identical"]
            else:
                sub = run_mutation_curl(expansion, counting, which, quick)
                summary = sub["summary"]
                diverged = sub["diverged"]
                record["cases"] = sub["cases"]
            record["launches"] = counting.launches
            record["ptx"] = _ptx_differs(mutant, shipped)
            record["summary"] = summary
            record["diverged_cases"] = diverged
            record.update(source_mutation_verdict(
                name, hits, counting.launches, record["ptx"]["differs"],
                diverged))
        records[name] = record
        log(f"case {index}/{total} [mutation:{name}] {record['verdict']} "
            f"hits={hits} launches={record.get('launches')} "
            f"diverged={record.get('diverged_cases')} "
            f"elapsed={round(time.time() - started_all, 1)} s "
            f"({round(time.time() - started, 1)} s)")
        results["mutations"] = records
        save(results, out_path)

    for name, predicted in HOST_MUTATIONS.items():
        index += 1
        started = time.time()
        record = {"kind": "host", "predicted_caught": predicted}
        if name in ("reflect_row_n_minus_two", "far_fill_uses_plus_phase",
                    "reverse_axis_order"):
            mutation = name
            summary = run_fill(results, out_path, expansion,
                               entry_mutation=mutation, label=name, quick=quick)
            diverged = summary["ran"] - summary["identical"]
        elif name == "imaginary_words_are_read_not_synthesised":
            summary = run_fill(results, out_path, expansion,
                               entry_mutation="nonzero_imaginary_words",
                               label=name, quick=quick)
            diverged = summary["ran"] - summary["identical"]
        elif name == "fuse_fill_passes":
            summary = run_fill(results, out_path, expansion, label=name,
                               quick=quick, fuse_passes=True)
            diverged = summary["ran"] - summary["identical"]
        else:  # phase_on_folded_axis
            sub = run_phase_on_folded_axis(expansion)
            summary = sub["summary"]
            diverged = sub["diverged"]
            record["cases"] = sub["cases"]
        record["summary"] = summary
        record["diverged_cases"] = diverged
        record.update(host_mutation_verdict(name, predicted, summary, diverged))
        records[name] = record
        log(f"case {index}/{total} [mutation:{name}] {record['verdict']} "
            f"diverged={diverged} elapsed={round(time.time() - started_all, 1)} s "
            f"({round(time.time() - started, 1)} s)")
        results["mutations"] = records
        save(results, out_path)
    return records


def run_mutation_curl(expansion: int, counting, which: str,
                      quick: bool) -> Dict[str, Any]:
    """A curl mutation over the rows that can see it, at the SHIPPED fusion
    configuration (``guard=False``) — the same one the asserted synthetic rows
    and ``run_fill``'s ``plan.run()`` use, so the two mutation families' verdicts
    are comparable.

    THE (shape, config) PRODUCT IS NOT RECTANGULAR, and the rows it omits are
    omitted because the ARRAY PATH cannot build them, not because they are
    uninteresting: :func:`viable` refuses a fold on an axis with four cells or
    fewer, since the near ghost reads cell 2 and the far ghost needs a reflect
    row distinct from it. ``SHAPES[:3]`` includes the 2-D sheet ``(9, 7, 1)`` and
    ``configs`` includes ``fold_z_*``, so the product contains a fold of a
    ONE-CELL axis. :func:`run_synthetic` has always skipped those rows through
    the same predicate; this leg did not, and built them — the reference shift
    read ``field[2]`` on an axis of size 1 and raised ``IndexError``, which
    ``run_leg`` recorded as a leg failure that ABORTED the whole mutation leg
    after the three fill mutants, leaving the fourteen curl and host mutants
    unrun and their coverage claims unestablished.

    The skipped rows are RECORDED, not silently dropped. A mutant that ends up
    with no viable row at all must reach the DISARMED path with its emptiness
    visible, rather than reporting a clean ``diverged=0`` off an empty product.
    """
    cases: List[Dict[str, Any]] = []
    shapes = SHAPES[:1] if quick else SHAPES[:3]
    configs = [c for c in CONFIGS if c["name"] != "unfolded_control"]
    configs = configs[:2] if quick else configs
    storage_rows = {"curl": ((0.0, True),),
                    "beta_complex": ((BETA_GRATING_13_2, True),),
                    "beta_real": ((BETA_GRATING_13_2, False),)}[which]
    skipped: List[Dict[str, Any]] = []
    for shape in shapes:
        for config in configs:
            # The SAME predicate run_synthetic applies, on the same phase set
            # this leg passes to one_curl_case — not a re-derivation of it.
            reason = viable(shape, config, PHASE_SETS[0]["phases"])
            if reason:
                skipped.append({"shape": list(shape), "config": config["name"],
                                "reason": reason})
                continue
            for mirror_phase in MIRROR_PHASES:
                for offsets in ([config["row_offsets"]] if "row_offsets" in config
                                else [(o, o, o) for o in ROW_OFFSETS]):
                    for sub_step in CURL_SUB_STEPS:
                        for beta, storage in storage_rows:
                            # guard=False — the SHIPPED fusion configuration, the
                            # same one run_fill's plan.run() (guard=None) uses.
                            # Arming two mutation families under two different
                            # fusion settings would make their verdicts
                            # incomparable.
                            case = one_curl_case(
                                shape, config, PHASE_SETS[0], 0.35, mirror_phase,
                                tuple(offsets), sub_step, False, expansion, beta,
                                storage, kernel=counting)
                            cases.append(case)
    diverged = sum(0 if c["verdict"]["bit_identical"] else 1 for c in cases)
    per_config: Dict[str, int] = {}
    for case in cases:
        if not case["verdict"]["bit_identical"]:
            per_config[case["config"]] = per_config.get(case["config"], 0) + 1
    return {"summary": {"ran": len(cases), "diverged": diverged,
                        "per_config": per_config,
                        "skipped_unbuildable": skipped},
            "diverged": diverged, "cases": cases}


def run_phase_on_folded_axis(expansion: int) -> Dict[str, Any]:
    """Plant ``PH=1`` on a FOLDED axis — the configuration the predicate refuses.

    Predicted CAUGHT on MIRROR_METALLIC (the top plane is stepped there, and
    rotating the ``(+0.0, +0.0)`` ghost pair can flip a zero sign) and NULL on
    MIRROR_PERIODIC (both consumers of that plane are top-plane masked). Both
    verdicts are recorded WITH the reason.
    """
    cases: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 909)
    shape = (9, 9, 5)
    for termination in (MIRROR_METALLIC, MIRROR_PERIODIC):
        for sub_step in CURL_SUB_STEPS:
            spec = folded_complex.SUB_STEPS[sub_step]
            targets = tuple(spec["targets"])
            names = (targets + tuple("fu_" + t for t in targets)
                     + tuple(spec["sources"]))
            host = {name: gate._seed_host_complex(shape, rng) for name in names}
            coefficients = _coefficients(shape, rng)
            flat = _flat_coefficients(cp, coefficients, shape)
            codes = (CODE_OF[PERIODIC], CODE_OF[termination], CODE_OF[PERIODIC])
            mine = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
            theirs = {n: cp.asarray(np.array(v, copy=True)) for n, v in host.items()}
            folded_complex.plan_folded_complex_pml_curl_from_arrays(
                sub_step, mine, flat, codes, (None, _phase(0.27), None), 0.35,
                expansion).run()
            folded_complex.plan_folded_complex_pml_curl_from_arrays(
                sub_step, theirs, flat, codes, (None, None, None), 0.35,
                expansion).run()
            cp.cuda.runtime.deviceSynchronize()
            compared = targets + tuple("fu_" + t for t in targets)
            verdict = combine({n: bit_compare(mine[n], theirs[n]) for n in compared})
            cases.append({
                "termination": termination, "sub_step": sub_step,
                "verdict": verdict,
                "predicted": ("CAUGHT (top plane stepped, zero-sign flip reachable)"
                              if termination == MIRROR_METALLIC else
                              "NULL (both consumers of that plane are top-plane "
                              "masked)")})
            log(f"[phase_on_folded_axis] {termination}/{sub_step}: "
                f"identical={verdict['bit_identical']} "
                f"differing={verdict['differing_floats']}")
    diverged = sum(0 if c["verdict"]["bit_identical"] else 1 for c in cases)
    return {"summary": {"ran": len(cases), "diverged": diverged}, "cases": cases,
            "diverged": diverged}


# ---------------------------------------------------------------------------
# The engine-route legs, and the three corpus anchors
# ---------------------------------------------------------------------------

ENGINE_STEPS = 4

ENGINE_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"name": "fold_y_periodic_even", "axis": "Y", "boundaries": "periodic",
     "phase": 1, "cell": (1.6, 2.0, 0.0), "beta": 0.0},
    {"name": "fold_y_periodic_odd", "axis": "Y", "boundaries": "periodic",
     "phase": 1, "cell": (1.6, 2.1, 0.0), "beta": 0.0},
    {"name": "fold_y_metallic_odd_phase_minus", "axis": "Y",
     "boundaries": "metallic", "phase": -1, "cell": (1.6, 2.1, 0.0), "beta": 0.0},
    {"name": "fold_y_periodic_beta", "axis": "Y", "boundaries": "periodic",
     "phase": 1, "cell": (1.6, 2.0, 0.0), "beta": BETA_GRATING_13_2},
)

CORPUS_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"name": "eigsrc_0_complex", "resolution": 30.0, "cell": (14.0, 14.0, 0.0),
     "thickness": ((2, 2), (0, 2), (0, 0)), "phase": 1, "beta": BETA_EIGSRC,
     "kx": 0.0},
    {"name": "special_kz_0_13_2", "resolution": 30.0, "cell": (4.5, 6.0, 0.0),
     "thickness": ((1, 1), (0, 0), (0, 0)), "phase": 1,
     "beta": BETA_GRATING_13_2, "kx": KX_GRATING_13_2},
    {"name": "special_kz_1_17_7", "resolution": 30.0, "cell": (4.5, 6.0, 0.0),
     "thickness": ((1, 1), (0, 0), (0, 0)), "phase": 1,
     "beta": BETA_GRATING_17_7, "kx": KX_GRATING_17_7},
)


def _substituted_step(fields, pml, sub_step, expansion, probe_record):
    """Build the plan the engine route WOULD build, or None with its reasons."""
    grid = fields.grid
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        return folded_complex.plan_folded_beta_bloch_pml_curl(
            fields, pml, sub_step, probe=probe_record)
    return folded_complex.plan_folded_complex_pml_curl(
        fields, pml, sub_step, probe=probe_record)


def run_engine(results: Dict[str, Any], out_path: str, expansion: int,
               probe_record: Dict[str, Any], specs=ENGINE_GRIDS,
               label: str = "engine") -> Dict[str, Any]:
    """``stepping.step_*`` and the two fills substituted on REAL folded grids.

    Stated step budget: ``ENGINE_STEPS`` complete driver-order steps per case,
    with the sub-step compared after every one — not only at the end, so a
    divergence reports WHERE it started.
    """
    cases: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        started = time.time()
        grid, fields, pml = build_folded_fields(
            {"axis": spec.get("axis", "Y"),
             "boundaries": spec.get("boundaries", "periodic"),
             "phase": spec["phase"], "cell": spec["cell"]},
            cp, complex_storage=True, beta=spec.get("beta", 0.0))
        if "resolution" in spec:
            planes = (Mirror("Y", spec["phase"]),)
            grid = Grid(resolution=spec["resolution"], cell_size=spec["cell"],
                        dimensions=2, courant=0.35, boundaries="periodic",
                        symmetry=planes, beta=spec["beta"],
                        k_point=(spec.get("kx", 0.0), 0.0, 0.0), xp=cp)
            fields = Fields(grid=grid, force_complex_fields=True)
            fields.enable_pml_storage()
            pml = PML(grid=grid, thickness=spec["thickness"])
        rng = np.random.default_rng(SEED + 131 + index)
        for name in ALL_STATE:
            volume = getattr(fields, name, None)
            if volume is not None:
                volume[...] = cp.asarray(
                    gate._seed_host_complex(tuple(grid.shape), rng))

        mirror = Fields(grid=grid, force_complex_fields=True)
        mirror.enable_pml_storage()
        for name in ALL_STATE:
            source = getattr(fields, name, None)
            if source is not None:
                getattr(mirror, name)[...] = source

        per_step: List[Dict[str, Any]] = []
        refusals: Dict[str, Any] = {}
        for step in range(ENGINE_STEPS):
            for sub_step in CURL_SUB_STEPS:
                getattr(stepping, sub_step)(mirror, pml)
                plan = _substituted_step(fields, pml, sub_step, expansion,
                                         probe_record)
                if plan is None:
                    verdict = folded_complex.folded_complex_pml_curl_coverage(
                        fields, pml, sub_step, probe=probe_record)
                    refusals[sub_step] = list(verdict.reasons)
                    getattr(stepping, sub_step)(fields, pml)
                else:
                    plan.run()
                    cp.cuda.runtime.deviceSynchronize()
                names = folded_complex.SUB_STEPS[sub_step]["targets"]
                compared = tuple(names) + tuple("fu_" + n for n in names)
                per_step.append({
                    "step": step, "sub_step": sub_step,
                    "substituted": plan is not None,
                    "verdict": combine({
                        n: bit_compare(getattr(fields, n), getattr(mirror, n))
                        for n in compared})})
                fill = ("fill_symmetry_bc_B" if sub_step == "step_B"
                        else "fill_symmetry_bc_D")
                far = ("fill_folded_far_ghosts_B" if sub_step == "step_B"
                       else "fill_folded_far_ghosts_D")
                family = "B" if sub_step == "step_B" else "D"
                getattr(stepping, fill)(mirror)
                getattr(stepping, far)(mirror)
                fill_plan = folded_complex.plan_folded_mirror_ghost_fill_complex(
                    fields, family, probe=probe_record)
                if fill_plan is None:
                    getattr(stepping, fill)(fields)
                    getattr(stepping, far)(fields)
                    refusals[fill] = list(
                        folded_complex.folded_mirror_ghost_fill_complex_coverage(
                            fields, family, probe=probe_record).reasons)
                else:
                    fill_plan.run()
                    cp.cuda.runtime.deviceSynchronize()
                per_step.append({
                    "step": step, "sub_step": fill,
                    "substituted": fill_plan is not None,
                    "verdict": combine({
                        n: bit_compare(getattr(fields, n), getattr(mirror, n))
                        for n in folded_complex.GHOST_FILL_FAMILIES[family]["targets"]})})
                update = stepping.update_H if sub_step == "step_B" else stepping.update_E
                update(mirror, pml)
                update(fields, pml)
        case = {
            "spec": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in spec.items()},
            "steps": ENGINE_STEPS, "step_budget_stated": ENGINE_STEPS,
            "stored": [int(grid.stored_cells(a)) for a in range(3)],
            "owned": [int(grid.owned_cells(a)) for a in range(3)],
            "reflect_rows": [None if r is None else int(r)
                             for r in stepping._far_reflect_rows(grid)],
            "refusals": refusals,
            "per_step": per_step,
            "identical": all(entry["verdict"]["bit_identical"] for entry in per_step),
            "substituted_any": any(entry["substituted"] for entry in per_step),
            "seconds": round(time.time() - started, 3),
        }
        cases.append(case)
        log(f"case {index}/{len(specs)} [{label}] {spec.get('name')}: "
            f"identical={case['identical']} substituted={case['substituted_any']} "
            f"stored={case['stored']} rows={case['reflect_rows']} "
            f"({case['seconds']} s)")
        results[label] = {"ran": len(cases),
                          "identical": sum(int(c["identical"]) for c in cases),
                          "substituted": sum(int(c["substituted_any"]) for c in cases),
                          "cases": cases}
        save(results, out_path)
    return results[label]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_LEGS = ("expansion,reference,synthetic,fill,zero_init,identity,"
                "mutations,engine,corpus")
DEVICE_LEGS = ("synthetic", "fill", "zero_init", "identity", "mutations",
               "engine", "corpus")


def run_leg(results: Dict[str, Any], out_path: str, name: str,
            thunk: Callable[[], Any]) -> Any:
    """Run one leg; on an exception RECORD it and carry on to the next leg.

    ``main`` used to call every leg bare, in a fixed order with ``synthetic``
    first, so a single raise in the first leg took the entire gate down: no
    ``fill``, ``zero_init``, ``identity``, ``mutations``, ``engine`` or
    ``corpus`` result was produced, the artifact was left at whatever the last
    ``save`` wrote, and the traceback landed on stdout with no verdict attached
    to it. One leg's defect must not erase the other legs' evidence.

    An errored leg is a FAILURE, never a skip. Two things make that so and both
    are load-bearing:

    * the error — type, message and the full traceback — is stamped into
      ``results["leg_errors"][name]`` and saved immediately, so the artifact
      names what broke rather than merely lacking a key;
    * :func:`build_summary` keys OFF that dict, so a leg that raised cannot be
      mistaken for a leg that was never requested. Without that clause the
      failure would be silent: ``build_summary`` reads ``results.get("fill")``
      and friends, and a leg that raised before its first ``save`` leaves no key
      at all — exactly the shape of a leg the caller never asked for.

    ``KeyboardInterrupt`` and ``SystemExit`` are deliberately NOT caught: an
    operator stopping a device run wants it stopped, not converted into a
    recorded failure and eight more legs of GPU time.
    """
    try:
        return thunk()
    except Exception as exc:  # noqa: BLE001 - the whole point is to not abort
        record = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        results.setdefault("leg_errors", {})[name] = record
        save(results, out_path)
        log(f"[{name}] RAISED {record['type']}: {record['message']} — recorded "
            f"as a FAILURE; continuing to the next leg")
        log(record["traceback"].rstrip())
        return None


def run_reference_validation(results: Dict[str, Any], out_path: str,
                             xp) -> Dict[str, Any]:
    """The reference against ``stepping.py`` itself, on whichever backend is here.

    The laptop twin (``validate_folded_complex_reference_vs_stepping.py``) is what
    must pass BEFORE anything is staged; this leg re-runs the same comparison on
    the device host so the artifact carries it too.
    """
    import validate_folded_complex_reference_vs_stepping as validator  # noqa: PLC0415

    record = validator.run(xp)
    results["reference"] = record
    save(results, out_path)
    log(f"[reference] {record['identical']}/{record['ran']} configurations "
        f"bit-identical to stepping.py")
    # The NaN census rides in the LOG as well as the record: the comparator
    # forgives NaN against NaN (unspecified sign and payload, IEEE 754-2019
    # §6.2.1/§6.2.3/§7.2), and a bloom hiding inside that exemption is exactly
    # what the count exists to make visible. With FILL_NEEDLE_MAGNITUDE finite it
    # reads 0, which is what makes this verdict and a raw-bit verdict the same
    # measurement.
    log(f"[reference] NaN census {record.get('nan_words', 0)} words in "
        f"{record.get('nan_checks', 0)} array-checks, "
        f"{record.get('nan_exempt', 0)} forgiven")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(_HERE, "results",
                                                      "triton_folded_complex",
                                                      "gate.json"))
    parser.add_argument("--probe-artifact", default=None,
                        help="where to write the measured expansion record")
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    parser.add_argument("--quick", action="store_true",
                        help="a smoke subset; never a certification")
    args = parser.parse_args(argv)

    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    results_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(results_dir, exist_ok=True)
    artifact = args.probe_artifact or os.path.join(results_dir, "probe.json")

    if cp is not None:
        gate.install_ftz_strip()

    results: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "legs": legs,
        "quick": bool(args.quick),
        "triton_available": _TRITON_AVAILABLE,
        "cupy_available": cp is not None,
        "provenance": write_provenance(results_dir),
        "subnormal_policy": gate.policy_stamp("cupy" if cp is not None else "numpy"),
        # The configuration this artifact CERTIFIES, stated rather than implied.
        # `kernels.ENABLE_FP_FUSION` is False and every plan spells
        # `enable_fp_fusion = ENABLE_FP_FUSION if guard is None else bool(guard)`
        # (folded_complex.py:2227, :2372, :2492, :2612), so `guard=False` IS the
        # shipped launch and `guard=True` is the divergence control. Bit-identity
        # is claimed for fusion OFF and for nothing else: with fp fusion enabled
        # the compiler may contract either grouping, which is why the fusion-ON
        # rows are recorded and never asserted — the same split both certified
        # sibling tranches use (gate_triton_complex.py:1114,
        # gate_triton_special_kz.py:683).
        "fusion_configuration": {
            "shipped": "enable_fp_fusion=False (kernels.ENABLE_FP_FUSION)",
            "certified": "enable_fp_fusion=False — the guard=False rows are the "
                         "ASSERTED set",
            "measured_not_asserted": "enable_fp_fusion=True (guard=True), a "
                                     "divergence control only",
            "enable_fp_fusion_default": shipped_fusion_default(),
        },
        "step_budget": {"engine": ENGINE_STEPS, "zero_init": ZERO_INIT_STEPS,
                        "fill_live_state": MULTI_STEP_COUNT},
    }
    save(results, args.out)

    licence: Dict[str, Any] = {"licensed": False, "reasons": ["expansion leg not run"]}
    if "expansion" in legs:
        # ``is None`` and not a truthiness test: a licence dict is data, and a
        # falsy-but-real record must not be read as "the leg raised".
        returned = run_leg(results, args.out, "expansion",
                           lambda: run_expansion(results, args.out, artifact))
        licence = ({"licensed": False, "reasons": ["the expansion leg RAISED"]}
                   if returned is None else returned)
    if "reference" in legs:
        run_leg(results, args.out, "reference",
                lambda: run_reference_validation(
                    results, args.out, cp if cp is not None else np))

    device_legs = [leg for leg in legs if leg in DEVICE_LEGS]
    if device_legs and (cp is None or not _TRITON_AVAILABLE):
        results["device_legs_skipped"] = (
            "no cupy and/or no triton on this host; the reference leg is the "
            "only meaningful one here")
        save(results, args.out)
        log("[skip] device legs need cupy + triton")
        device_legs = []

    record = results.get("expansion", {}).get("cupy")
    expansion = (folded_complex.parity_expansion_from_probe(record)
                 if record is not None else None)
    if device_legs and expansion is None:
        results["device_legs_skipped"] = (
            "the measured probe record licenses no EXPANSION; a guessed "
            "constexpr certifies nothing")
        save(results, args.out)
        device_legs = []

    # Every device leg through ``run_leg``: one leg's raise records a named
    # FAILURE and the remaining legs still produce their evidence.
    device_thunks: Tuple[Tuple[str, Callable[[], Any]], ...] = (
        ("synthetic", lambda: run_synthetic(results, args.out, expansion,
                                            quick=args.quick)),
        ("fill", lambda: run_fill(results, args.out, expansion,
                                  quick=args.quick)),
        ("zero_init", lambda: run_zero_init(results, args.out, expansion)),
        ("identity", lambda: run_identity(results, args.out, expansion)),
        ("mutations", lambda: run_mutations(results, args.out, expansion,
                                            quick=args.quick)),
        ("engine", lambda: run_engine(results, args.out, expansion, record)),
        ("corpus", lambda: run_engine(results, args.out, expansion, record,
                                      specs=CORPUS_GRIDS, label="corpus")),
    )
    for name, thunk in device_thunks:
        if name in device_legs:
            run_leg(results, args.out, name, thunk)

    # RE-STAMP THE POLICY, and this is not bookkeeping. `policy_stamp` COPIES
    # the strip's counters (`int(state["calls"])`, gate_triton_complex.py:404),
    # so the stamp taken when `results` was built froze the counts as they stood
    # BEFORE any leg ran — structurally 0 and 0, whatever the run then did. The
    # artifact therefore advertised "ieee_keep_ftz_stripped" beside a counter
    # saying the strip had never fired, which is the exact shape of unevidenced
    # claim this gate refuses everywhere else: the run that found this had 6
    # NVRTC compiles with -ftz=true removed from all 6, recorded only in the
    # expansion sub-record because that one is stamped after its own work.
    # The second stamp is the sibling probe's own pattern
    # (probe_triton_nonlinear_composition.py:818 then :845).
    results["subnormal_policy"] = gate.policy_stamp(
        "cupy" if cp is not None else "numpy")
    summary = build_summary(results, licence)
    results["summary"] = summary
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(results, args.out)
    log("[summary] " + json.dumps(summary, indent=1, sort_keys=True))
    for path in _TEMPORARY:
        try:
            os.unlink(path)
        except OSError:
            pass
    return 0 if summary["passed"] else 1


def build_summary(results: Dict[str, Any], licence: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict, with every failure path named rather than aggregated away."""
    failures: List[str] = []
    summary: Dict[str, Any] = {"expansion_licensed": licence.get("licensed"),
                               "expansion_reasons": licence.get("reasons", [])}
    if not licence.get("licensed"):
        failures.append("the expansion probe licensed nothing")

    # A leg that RAISED is a failure by name. It has to be checked here and not
    # only where its own key is read: a leg that raised before its first ``save``
    # leaves no key at all, which is indistinguishable from a leg the caller
    # never asked for — so without this clause the loudest possible defect would
    # be the quietest verdict.
    leg_errors = results.get("leg_errors") or {}
    if leg_errors:
        summary["leg_errors"] = {name: f"{record['type']}: {record['message']}"
                                 for name, record in sorted(leg_errors.items())}
        for name, record in sorted(leg_errors.items()):
            failures.append(f"leg {name} RAISED {record['type']}: "
                            f"{record['message']}")

    reference = results.get("reference")
    if reference is not None:
        summary["reference"] = f"{reference['identical']}/{reference['ran']}"
        if reference["identical"] != reference["ran"]:
            failures.append("the reference diverges from stepping.py")

    synthetic = results.get("synthetic", {}).get("summary")
    if synthetic is not None:
        summary["synthetic"] = synthetic
        # The SHIPPED configuration is fusion OFF (kernels.ENABLE_FP_FUSION is
        # False), so the fusion-OFF rows are the asserted ones; the fusion-ON
        # rows are recorded because with fp fusion enabled the compiler may
        # contract either grouping.
        #
        # ...and that sentence is CHECKED, not asserted in prose: the asserted
        # set is spelled `guard is False`, so if the package's default ever flips
        # the gate would go on certifying a configuration production no longer
        # launches — which is the exact defect this leg was repaired for, in the
        # other direction.
        if shipped_fusion_default():
            failures.append(
                "kernels.ENABLE_FP_FUSION is True, so the SHIPPED launch is now "
                "fusion ON, but this gate asserts the guard=False (fusion OFF) "
                "rows; the asserted set no longer tracks what production runs")
        if synthetic["fusion_off_asserted_total"] == 0:
            failures.append("the synthetic leg ran no fusion-OFF row; the "
                            "configuration production launches was never "
                            "asserted")
        elif (synthetic["fusion_off_asserted_identical"]
              != synthetic["fusion_off_asserted_total"]):
            failures.append("fusion-OFF synthetic rows are not bit-identical")

    fills = results.get("fill", {}).get("gate")
    if fills is not None:
        summary["fill"] = f"{fills['identical']}/{fills['ran']}"
        if fills["identical"] != fills["ran"]:
            failures.append("the ghost fill diverges from the array path")
        vacuous = fills.get("vacuous_cases") or []
        if vacuous:
            # RECORDED, not failed. Two different reasons live in this list and
            # both are measured (blind_fill_cells): a folded METALLIC axis at
            # mirror phase +1 stores no signed zero from a quiet grid (near = +1,
            # no far fill), and `live_post_step` stores none on ANY row, because
            # four driven steps from a random seed leave generic nonzero floats.
            # Neither is a blind row; a cell blind in all four families is.
            summary["vacuous_fill_states"] = vacuous
        blind = fills.get("blind_cells")
        if blind is None:
            blind = blind_fill_cells(fills.get("cases") or [])
        if blind:
            # A (grid, family) cell blind in EVERY state family cannot see any of
            # the fill's needles at all. Recording that without failing is the
            # silence the discipline forbids.
            summary["blind_fill_cells"] = blind
            failures.append(
                f"{len(blind)} fill CELLS are blind in every state family "
                f"(reference signed-zero census 0 throughout): "
                f"{[entry['cell'] for entry in blind]}")

    zero_init = results.get("zero_init")
    if zero_init is not None:
        summary["zero_init"] = {
            "census": zero_init.get("census"),
            "bit_identity": (zero_init.get("bit_identity") or {}).get("ran"),
            "passed": zero_init.get("passed")}
        if not zero_init.get("passed"):
            failures.append(
                "the zero-init negative-coefficient-absorber case did not pass: "
                + str(zero_init.get("failure", "unknown")))

    identity = results.get("identity")
    if identity is not None:
        summary["identity"] = f"{identity['identical']}/{identity['ran']}"
        if identity["identical"] != identity["ran"]:
            failures.append("a reduction/identity product is not byte-identical")

    mutations = results.get("mutations", {})
    verdicts = {name: record.get("verdict") for name, record in mutations.items()}
    summary["mutations"] = verdicts
    # Keyed off the LEG HAVING RUN, not off the records being non-empty: a
    # mutation leg that ran and produced nothing must still be caught by the
    # declared-null check below, and a laptop run that skipped the leg entirely
    # (no key at all) must not manufacture a failure about a device measurement
    # it never attempted.
    if "mutations" in results:
        # THE SPLIT, counted rather than narrated. A retired mutation stays in
        # the denominator: the artifact shows seventeen asked and how each one
        # answered, so a null reads as a null and never as a disappearance.
        counts: Dict[str, int] = {}
        for verdict in verdicts.values():
            counts[str(verdict)] = counts.get(str(verdict), 0) + 1
        summary["mutation_counts"] = {
            "asked": len(SOURCE_MUTATIONS) + len(HOST_MUTATIONS),
            "ran": len(verdicts),
            **{key: counts[key] for key in sorted(counts)}}
        # ...and the reason travels WITH the count. A verdict line reading
        # PREDICTED-NULL with the reason left in the source file is the same
        # silence one step further on.
        nulls = {name: {"reason": record.get("null_reason"),
                        "live_claim_carried_by":
                            record.get("live_claim_carried_by"),
                        "twin_claim": record.get("twin_claim"),
                        "twin_verdict":
                            verdicts.get(record.get("live_claim_carried_by")),
                        "retirement_evidence":
                            record.get("retirement_evidence"),
                        "premise_test": record.get("premise_test")}
                 for name, record in mutations.items()
                 if record.get("verdict") == "PREDICTED-NULL"}
        if nulls:
            summary["mutation_nulls"] = nulls
        for name, record in nulls.items():
            twin = record.get("live_claim_carried_by")
            # A null whose twin is not itself established carries the claim
            # NOWHERE: the retirement's whole argument is that the coverage moved
            # rather than vanished, and that argument is checkable.
            if twin and record["twin_verdict"] not in ("CAUGHT", "MEASURED"):
                failures.append(
                    f"mutation {name} is recorded as a predicted null whose "
                    f"live claim is carried by {twin}, and {twin} came back "
                    f"{record['twin_verdict']!r}: the retired mutation's source-"
                    f"layer claim is now established by nothing")
        # The third way a null could go quiet: declared here, never executed
        # there. Only checkable when the mutation leg produced records at all.
        for name in SOURCE_MUTATION_NULLS:
            if name not in verdicts:
                failures.append(
                    f"{name} is declared a predicted structural null but the "
                    f"mutation leg produced no record for it; a null is judged, "
                    f"not skipped")
    for name, verdict in verdicts.items():
        if verdict in ("DISARMED", "NEEDLE-MISSED", "NULL-INVALIDATED"):
            failures.append(f"mutation {name}: {verdict} — "
                            f"{mutations[name].get('failure')}")

    for label in ("engine", "corpus"):
        record = results.get(label)
        if record is not None:
            summary[label] = f"{record['identical']}/{record['ran']}"
            if record["identical"] != record["ran"]:
                failures.append(f"the {label} route diverges from the array path")
            if record["substituted"] != record["ran"]:
                failures.append(f"the {label} route refused to substitute on "
                                f"{record['ran'] - record['substituted']} cases")

    # A device leg that was REQUESTED, was not cleanly skipped, and left neither
    # a result nor a recorded error, VANISHED. That can only mean a leg returned
    # without writing its key, and a gate that reports nothing about a leg it was
    # asked to run has not run it.
    requested = tuple(results.get("legs") or ())
    if requested and not results.get("device_legs_skipped"):
        for name in DEVICE_LEGS:
            if name in requested and name not in leg_errors \
                    and results.get(name) is None:
                failures.append(f"leg {name} was requested, was not skipped, and "
                                f"produced neither a result nor an error")

    summary["failures"] = failures
    summary["passed"] = not failures
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
