"""Bit-identity gate for the FOLDED off-diagonal epsilon ``update_E`` Triton kernel.

The module under test is ``meep_gpu/triton_kernels/folded_offdiag_update_e.py``
— NOT wired into production dispatch (the engine's fast-path hook keeps
returning None; nothing here changes that). This gate is the arbiter of the one
thing that family adds to the certified off-diagonal kernel: THE MIRROR GHOST ON
THE PARTNER-AXIS DOWN SHIFT, and the equally load-bearing claim that the own-axis
UP shift stays an exact zero on a fold.

NO BYTE-IDENTITY CLAIM IS MADE UNTIL A DEVICE LEG COMPARES BYTES. A leg that was
requested and did not compare — because there is no device, or because its
device half is not written yet — never stamps ``passed``, and is NAMED in the
status line (exit 75, the cannot-certify-here convention). That holds even when
a sibling leg did compare.

AND A LEG THAT DID COMPARE MUST ALSO HAVE CERTIFIED. ``compared`` is a COUNT and
``certified`` is the OUTCOME, and the top-level verdict reads BOTH: every leg
that ran a uint32 comparison must report every comparison identical, under the
configuration named below, or the gate FAILS by name and exits 1. This is
stated because the opposite was true here — the verdict read only the count, so
a device round in which every compare failed still stamped ``passed: true``,
printed ``[gate] passed`` and exited 0 (demonstrated on the laptop by stubbing
the five device host halves to ``{compared: 156, identical: 0}``).

EVERY DEVICE LEG IS SPLIT INTO A HOST HALF AND A DEVICE HALF, and the host half
runs on a laptop. It is the case enumeration, the reference (through the
transcription the ``reference`` leg pins against ``stepping.update_E``) and the
NON-VACUITY assertions; it can FAIL, and it fails in the round that wrote the
case rather than after spending device time on a green row that measured
nothing. The device half is the launch and the uint32 compare. EVERY DEVICE HALF
NOW EXISTS: ``DEVICE_HALF_TODO`` is empty and ``DEVICE_HALF_IMPLEMENTED`` is
True on every leg. They were written in the device round rather than before it,
for the reason that list used to give — mutant loading, launch counting, build
identity and real CuPy engine objects cannot be exercised at all without
hardware, and a harness written blind is how a leg arrives green and vacuous.
The round proved the point twice: the mutation and identity legs died on a
``UnicodeDecodeError`` under the measurement host's ASCII locale, and the
two-builds check read a warm-cache HIT as a stale binary. Both are closed below,
and both were found by an artifact rather than by reading.

Legs, in order (``--legs`` selects a subset):

* ``license``     — DEVICE, and the leg to read first: THE MEASUREMENT THAT
  LICENSES A NEW KERNEL. The cheap alternative to this whole family was to
  restate ``coverage._grid_reasons`` clause 5 and let the CERTIFIED off-diagonal
  body serve folds unchanged, so the file that rejects that owes a measurement.
  This leg launches the SHIPPED CERTIFIED KERNEL on all 16 real folded reference
  configurations under BOTH readings of the fold (its alphabet has two values,
  so "admit the fold" forces a choice of arm and a single reading would invite
  "you handed it the wrong arm"), and compares every result against the
  array-path transcription. The claim is TWO-SIDED: the certified body must
  DIVERGE exactly where a live row slot takes the folded axis as its partner and
  AGREE where none does — a body that diverged everywhere would evidence a
  broken harness, not an insufficient kernel — and this family's kernel must be
  byte-identical on all 16. Its verdict function is pure, so the laptop tests
  drive every outcome.

* ``reference``   — the in-file transcription pinned against ``stepping.update_E``
  itself, on real ``Grid``/``Fields``/``PML`` objects with installed tensors and
  real mirror planes, byte for byte, over repeated sub-step calls (state chained
  through ``f_w``), on 16 grids INCLUDING TWO WITH THREE SIMULTANEOUS MIRROR
  PLANES. Runs on NumPy (the laptop merge bar) and again on CuPy when a
  device is present; emits a per-call state sha256 chain. EVERY reference grid
  additionally runs the SIX DISCRIMINATING CONTROLS below, so a reference that
  passed by accident (both sides wrong the same way) is impossible: each control
  is a different transcription of one ghost arm, and each must DIFFER.

      down = 0 (the certified METALLIC arm)   must differ  <- the certified
      down = the periodic wrap                must differ      kernel cannot
      down = +phase * g[row 2] (sign flipped) must differ      serve a fold
      up   = parity * g[reflect_row]          must differ  <- the array path
      up   = the periodic wrap                must differ      serves an exact 0
      fold plane masked like a metallic wall   must differ  <- the mask abstains

  A control that fails to differ is recorded as VACUOUS and fails the leg: it
  means the case could not see the arm it was built to see.

* ``refusals``    — every predicted refusal returns None from the engine builder
  with its named reason; 0 admissions. Includes the disjointness seams against
  BOTH parents (``offdiag_update_e`` refuses the fold through
  ``coverage._grid_reasons`` clause 5; ``symmetry.folded_constitutive_coverage``
  refuses the rows), the STALE-DOCSTRING GUARD (the installer does NOT refuse
  folded rows, whatever stepping.py:1219-1220 / fields.py:1237-1241 /
  driver.py:1202-1204 claim — this whole family exists because it does not), and
  the fold-classification refusals inherited from ``symmetry.folded_axis_kinds``.
  EVERY CASE MUST BE BUILDABLE AND MUST MATCH ITS REASON BY NAME. Two clauses
  had no witness and said nothing about it: the thin folded axis (a folded
  PERIODIC axis bottoms out at 3 stored cells at every cell size tried, so the
  condition is unreachable that way and the case recorded the backend clause
  alone — it is now built on a folded METALLIC axis, 2 stored cells, and matches
  the thin-axis reason by name) and BFAST (``bfast_active`` is a read-only
  property over ``bfast_scaled_k``, so poking it RAISED and a bare
  ``except: continue`` dropped the case while the leg still reported 12/12
  passing). Both it and ``beta`` are now built through the Grid CONSTRUCTOR —
  beta on the reduced 2-D grid, the only shape the engine accepts it on — and a
  case that cannot be BUILT is a leg FAILURE, never a silent skip.

* ``identity_host`` — the REACHABILITY claim, on NumPy, needing no device: on the
  same grid, the same coefficients and the same seeds, flipping ONLY the folded
  axis's ghost code between MIRROR and METALLIC must be
    - byte-IDENTICAL when NO SURVIVING ROW SLOT takes ``E_a`` as its PARTNER for
      the folded axis ``a``, and
    - byte-DIFFERENT as soon as one surviving slot does.
  THE TEST IS AT SLOT LEVEL, and the difference from the coarser row-level
  reading is a case, not a nicety: a single live slot ``Ey <- Ez`` on a folded X
  leaves a live row that is not ``Ex`` while the fold enters NEITHER role, and
  the bytes agree. The leg carries three such counterexamples (one per axis) and
  FAILS if it carries none — before they existed it drew only from ``only_row_*``
  (both partners live) and ``varying_full``, so a row-level predicate passed it
  unchanged. This is what makes "the delta is exactly the partner-axis down
  ghost" a measurement rather than a docstring. The NEGATIVE half is what stops
  the positive half from being vacuous.

* ``synthetic``   — DEVICE. The sweep: shapes (16,16,16) / (13,17,11)
  non-power-of-two / (1,160,160) 2-D reduced (the invariant-axis pair is g+g) /
  (11,13,15), x fold configurations covering BOTH terminations (folded PERIODIC
  and folded METALLIC — a sweep carrying one proves nothing about the other),
  BOTH plane phases, a fold on each of X, Y and Z, TWO planes at once, THREE
  planes at once (the predicate ADMITS three simultaneous mirror planes and
  nothing in this family carried one: with every axis folded all six row slots
  take a ghosted down shift, under three ghost weights that are not all equal),
  and ZERO folded axes (the reduction row), x row forms (uniform full oracle tensor with
  its negative off-diagonals / MANDATORY spatially varying volumes / single-slot
  arms placing the fold in BOTH roles — a live row's partner axis and only its
  own axis), x inverse-epsilon forms (three distinct volumes AND three aliases),
  x amplitude classes (normal; a CANCELLATION class with gs*us ~ -coupling so the
  row-sum association is byte-visible in f32; a ZERO-INIT class, below),
  x guard off/on, with real-layer coefficient tables cut at Courant 0.5 AND the
  NON-POWER-OF-TWO 0.35 and 0.3125. The reference side of every comparison is
  computed on NUMPY (host IEEE keep). Every case asserts the coupling is nonzero
  somewhere AND that its reference bytes differ from a coefficient-dropped
  (diagonal-engine) control, so no case can pass with the feature dark.

  THE ZERO-INIT CLASS IS NOT A RANDOM SEED SET TO ZERO. Measured on the laptop:
  with every array zero the mirror ghost's parity sign is ANNIHILATED by the
  ``values + ghost`` add (``+0.0 + -0.0 == +0.0``), the negative-zero census over
  E and f_w is 0, and a sign-flipped ghost is NOT byte-visible — a case built
  that way is VACUOUS and is recorded as such, never as coverage. The needle that
  works is ZERO EVERYWHERE EXCEPT STORED ROW ``MIRROR_SOURCE_INDEX`` of the
  folded axis, over a NEGATIVE-ZERO background, which makes the mirror ghost the
  ONLY path to a nonzero fold-plane E.

  MEASURED BY THIS FILE'S OWN SWEEP, at ``SEED = 20260813`` on the shipped
  enumeration, recorded per case as ``fold_plane_max_abs_E`` and
  ``negative_zero_census`` (fold plane max|E| / census):

      foldX_periodic_even   4.759e-02 / 328     foldY_periodic_odd  5.479e-02 / 301
      foldX_metallic_odd    5.725e-02 / 267     foldZ_periodic_even 4.878e-02 / 345

  with BOTH the sign-flipped ghost and the metallic-zero ghost byte-visible on
  every one. The two signed-zero cases measure 3.631e-02 / 302 and 4.287e-02 /
  327, and both see the unary-minus lowering. Each case ASSERTS census > 0 and
  fold-plane E > 0 and both ghost controls visible; census 0 fails the case. The
  census is the NEGATIVE-ZERO count throughout — ``(x == 0) & signbit(x)``,
  never a sign-bit tally, which on a random seed counts about half the stored
  words and would relabel an ordinary-negative count as this class's witness.

* ``subnormal``   — DEVICE. The cancellation class scaled into underflow,
  REPORTED SEPARATELY under the stamped policy, never merged.

* ``identity``    — DEVICE, the compiled seam, three claims, each ASSERTED
  BESIDE ITS OWN NEGATIVE CONTROL in the same configuration (all three are
  equalities, and an equality with no control is satisfied by a kernel that
  wrote nothing):
  (i) with ZERO folded axes the kernel must be byte-identical to the certified
      ``offdiag_update_e.offdiag_constitutive_step`` on the same seeds — the
      reduction row;
  (ii) with a folded axis that is no live row's partner axis it must be
      byte-identical to the certified kernel with that axis coded METALLIC (the
      device half of ``identity_host``);
  (iii) MIRROR_METALLIC and MIRROR_PERIODIC must produce IDENTICAL bytes — the
      predicted null of the module docstring's point D, derived from this
      sub-step having no ownership mask and no reflect row. Recorded WITH its
      reason and with PTX evidence that the two specializations really are two
      builds, so a null cannot be a stale-cache artifact.
      THE TWO-BUILDS EVIDENCE IS MEASURED ON AN EMPTY CACHE, and its first
      spelling was wrong in a way the artifact caught. It counted NEW
      specializations on the SHIPPED kernel across the two launches and required
      one from each — but the sweep leg runs first and had already compiled 63
      specializations, one of them this exact MIRROR_PERIODIC signature, so the
      second launch added none and the check read "one binary served both codes"
      when what happened was a cache HIT ON THE RIGHT ENTRY. What is measured
      now is a FRESH renamed copy of the shipped source with an empty cache,
      launched once per code: two specialization keys must appear, the copy's
      results must equal the shipped kernel's (a faithful stand-in, not merely
      an available one), and whether the two PTX texts are EQUAL is recorded as
      the null's positive evidence. Measured: 2 specializations, 2 distinct
      keys, PTX texts equal, and f10's mutant as the live control at 748 words.

      THE HOST HALF OF (iii) IS A SOURCE AND PLAN MEASUREMENT, NOT A REFERENCE
      DRIVE: ``reference_update_e`` takes KINDS and ``KIND_OF_CODE`` maps both
      mirror codes to MIRROR, so driving it twice with the two codes hands it
      byte-identical arguments and the equality is an identity of inputs
      (verified by instrumenting both calls). What is measured instead is that
      the COMPILED SURFACE never names either mirror code — both land in the
      same ``else`` after ``== PERIODIC`` / ``== METALLIC`` — with f10's mutant
      as the live control that the scan can see such a name, and that two plans
      built from the two codes agree in every launched field but the code.

* ``mutations``   — DEVICE. Defects planted in the shipped kernel's SOURCE
  (compiled from a real file, launch-counted, DISARMED / NEEDLE-MISSED fail
  paths, and the mutant's PTX verified to DIFFER from every shipped
  specialization — the stale-cache platform fact):

    f1  mirror down ghost served as the METALLIC zero      (the fold's whole delta)
    f2  mirror down ghost served as the PERIODIC wrap
    f3  ghost index n-1 instead of MIRROR_SOURCE_INDEX = 2 (reflects about the
        window top rather than about the plane)
    f4  ghost index MIRROR_SOURCE_INDEX + 1                (off-by-one row)
    f5  ghost weight forced to +1.0                        (parity dropped)
    f6  ghost weight respelled as a UNARY MINUS on the loaded value — the
        platform fact (semantic.minus lowers -x as 0.0 - x and canonicalises
        signed zeros) re-measured on the class where it can bite, rather than
        inherited. Its outcome is RECORDED either way: caught means the
        canonicalisation is reachable here, not caught means it is not, and the
        zero-init needle is the case that decides it.
    f7  ghost weight applied to the WHOLE down lane rather than the ghost lane
    f8  mirror UP ghost served as the parity-weighted reflect row instead of 0
        (the array-path finding's opposite: this kernel must reproduce the ZERO)
    f9  the wall mask STOPS ABSTAINING ON THE FOLDED AXIS'S LANE, and nothing
        else: the constexpr forced on at the four call sites that carry that
        lane, which zeroes exactly what the ``mask_fold`` host analogue zeroes.
        (Its previous spelling dropped BOTH guards unconditionally and also
        zeroed planes on two PERIODIC axes, so a 'caught' verdict evidenced
        "some wall masking changed bytes" rather than the abstention.)
    f11 BOTH mask guards dropped — f9's coarser sibling, kept under its own name
        and its own claim: that the ``WM_*`` constexprs really are off on this
        needle's PERIODIC axes.
    f10 the two mirror codes given different behaviour (the null made real)

  plus the certified family's own m1/m2/m3/m9 re-run on a FOLDED case, because a
  fold changes which lanes are live and a mutation caught unfolded is not thereby
  caught folded. EVERY PATCH DECLARES ITS SITE COUNT AND THE COUNT IS ENFORCED:
  several patchers increment once per INDEPENDENT needle (m2 seven, m3 nine, f9
  four, f6 two), so ``hits > 0`` alone would read as armed while the mutant
  carried only part of the declared defect. A mismatch is an error, not a note.

  NULL CONTROLS, each with its recorded reason and each still required to launch
  with PTX differing from every shipped specialization: the commuted row sum
  (f32 addition is bitwise commutative), m4 (0.25 distributed — an exact power
  of two commutes with round-to-nearest away from underflow; the offdiag gate
  measured it not caught), and (iii) above. m4 is carried TWICE: asserted not-caught on
  the normal needle, and RECORDED on a subnormal needle, because "away from
  underflow" is its own caveat and the subnormal class is where it can fail —
  measured on the host today, the distributed spelling leaves every byte equal
  on the normal needle and MOVES bytes on the subnormal one.

* ``engine``      — DEVICE. ``plan_folded_offdiagonal_constitutive`` from the
  engine's own objects (``Fields.set_epsilon_volumes``-installed tensors on real
  folded CuPy grids) against ``stepping.update_E``, repeated calls, both
  terminations and both phases.

SUBNORMAL POLICY — certification runs UNDER ``ieee_keep_ftz_stripped``. CuPy
unconditionally appends ``-ftz=true`` to every NVRTC compile; this gate installs
the demonstrated strip (``gate_triton_complex.install_ftz_strip``) BEFORE any
CuPy compile, refuses to certify when the strip cannot be confirmed exercised,
and stamps every artifact with the policy plus the in-run strip counters. THE
STAMP IS TAKEN AFTER THE LEGS RUN (and after the licensing check, so
``zero_compiles_explained`` is in it); the pre-leg stamp is kept beside it as
``subnormal_policy_at_start``. Stamping only at the start — which this file did
— pinned ``nvrtc_calls = 0`` and ``ftz_removed = 0`` into every artifact by
construction, because those counters move only during the compiles inside the
legs.

CONFIGURATION CERTIFIED: ``ENABLE_FP_FUSION = False``. Fusion is not byte-uniform
across families (measured: complex 68/68 fusion-on rows identical, special_kz
24/96, nonlinear 0/108, offdiag 0/28, bfast 0/52); this family's tail is the
offdiag tail plus a ghost-lane multiply, so it certifies fusion-off and the
sweep's ``guard`` axis records the fusion-on outcome without merging it.
``synthetic.certified`` is therefore the GUARDED (fusion-off) rows alone —
merging the two, as this file once did, made a byte-perfect kernel report
``certified: false`` on this tranche's own measured expectation. The fusion axis
additionally carries the certified family's NON-VACUITY CONTROL: if every
fusion-on row was byte-identical, the guard axis measured nothing and the leg
FAILS with ``fusion-control-never-diverged`` rather than quietly certifying a
distinction it did not observe.

STEP BUDGET, counted from the code rather than asserted:

* ``reference`` — 16 grids x 4 ``stepping.update_E`` calls, each followed by the
  transcription evaluated SEVEN times (shipped + the six controls), plus one
  coefficient-dropped call: 4 engine calls and 29 host calls per grid.
* ``identity_host`` — 18 configurations x 2 drives (MIRROR, METALLIC) x 3
  chained host calls = 108.
* ``synthetic`` — per case, 2 to 6 HOST sub-step calls (the reference; the
  coefficient-dropped control; the metallic-arm control on a folded case; and on
  the needle classes the sign-flip, metallic-zero and unary-minus controls) plus
  three coupling evaluations, and EXACTLY ONE device launch. No sweep case
  chains state across calls.
* ``subnormal`` — 1 host call and 1 launch per row, 2 rows.
* ``mutations`` — 2 host calls per entry (the needle's reference and its
  analogue) plus 2 for the variant-machinery check; the launches are the device
  half's.
* ``engine`` — the HOST half makes NO sub-step call at all: it measures that the
  predicate's only refusal is the array-module clause. Its device half, when
  written, steps 4 calls per grid.

Nothing here runs longer than a few seconds per case on either backend.

Every case prints one flushed line as it lands; the JSON artifact is rewritten
atomically after every case (the progress-reporting rule); the gate writes its own
provenance record (``fingerprints.json`` is a shared ledger this gate does not
write; it is only hashed). Correctness only: no throughput or timing claims.

Usage (the GPU host, one clear device; the cache dir MUST carry the policy token)::

    CUDA_VISIBLE_DEVICES=N \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_folded_offdiag.py \\
        --out results/triton_folded_offdiag_<date>/gate.json

Laptop (NumPy only; device legs and the strip skip cleanly and say so)::

    python -u gate_triton_folded_offdiag.py \\
        --legs reference,refusals,identity_host \\
        --out /tmp/gate_folded_offdiag_local.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
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
except ImportError:  # Laptop leg: the reference transcription still validates.
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# The subnormal-POLICY machinery and the byte comparator are the certified
# families' own, imported rather than restated: a second copy of the -ftz strip
# could drift from the one whose exercise counters license every other family's
# numbers. (Coverage REASONS are the opposite case and are restated — see the
# module docstring of folded_offdiag_update_e.) Neither import needs a device.
import gate_triton_complex as gate                               # noqa: E402
import probe_fused_kernel_bit_identity as probe                  # noqa: E402

from meep_gpu import stepping                                    # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS, mirror_parity   # noqa: E402
from meep_gpu.grid import Grid, Mirror                           # noqa: E402
from meep_gpu.pml import PML                                     # noqa: E402
from meep_gpu.triton_kernels import folded_offdiag_update_e as fomod  # noqa: E402
from meep_gpu.triton_kernels import offdiag_update_e as odmod    # noqa: E402

#: uint32 byte compare with the ULP gap when unequal — never allclose.
bit_compare = probe.bit_compare
combine = probe.combine
from meep_gpu.triton_kernels import symmetry as symmod           # noqa: E402

SEED = 20260813
E_NAMES = ("Ex", "Ey", "Ez")
D_NAMES = ("Dx", "Dy", "Dz")
FW_NAMES = tuple("f_w_" + n for n in E_NAMES)
STATE_NAMES = E_NAMES + FW_NAMES        # what the sub-step writes
ALL_NAMES = STATE_NAMES + D_NAMES
AXES = "xyz"

PERIODIC = stepping.PERIODIC
METALLIC = stepping.METALLIC
MIRROR = stepping.MIRROR

#: test_tensor_epsilon._EPS_TENSOR — symmetric, positive definite; its INVERSE's
#: off-diagonals are NEGATIVE, so the uniform row form already carries negative
#: coefficients without contriving them.
EPS_TENSOR = np.array([
    [2.0, 0.35, 0.20],
    [0.35, 2.5, 0.15],
    [0.20, 0.15, 3.0],
], dtype=np.float64)
INV_TENSOR = np.linalg.inv(EPS_TENSOR)


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Any, path: str) -> None:
    """Atomic rewrite after every case (the progress-reporting rule)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def _face(axis: int, index: Any) -> Tuple[Any, ...]:
    return (slice(None),) * axis + (index,)


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_provenance(results_dir: str) -> str:
    import meep_gpu  # noqa: PLC0415

    package_dir = os.path.dirname(os.path.abspath(meep_gpu.__file__))
    kernel_dir = os.path.join(package_dir, "triton_kernels")
    tracked = {
        "meep_gpu/driver.py": os.path.join(package_dir, "driver.py"),
        "meep_gpu/fields.py": os.path.join(package_dir, "fields.py"),
        "meep_gpu/grid.py": os.path.join(package_dir, "grid.py"),
        "meep_gpu/pml.py": os.path.join(package_dir, "pml.py"),
        "meep_gpu/stepping.py": os.path.join(package_dir, "stepping.py"),
        "triton_kernels/coverage.py": os.path.join(kernel_dir, "coverage.py"),
        "triton_kernels/kernels.py": os.path.join(kernel_dir, "kernels.py"),
        "triton_kernels/launch.py": os.path.join(kernel_dir, "launch.py"),
        "triton_kernels/symmetry.py": os.path.join(kernel_dir, "symmetry.py"),
        "triton_kernels/offdiag_update_e.py": os.path.join(
            kernel_dir, "offdiag_update_e.py"),
        "triton_kernels/folded_offdiag_update_e.py": os.path.join(
            kernel_dir, "folded_offdiag_update_e.py"),
        "triton_kernels/fingerprints.json": os.path.join(
            kernel_dir, "fingerprints.json"),
        "parity/gate_triton_folded_offdiag.py": os.path.abspath(__file__),
    }
    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": None if cp is None else cp.__version__,
        "triton": triton.__version__ if _TRITON_AVAILABLE else None,
        "seed": SEED,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "enable_fp_fusion_certified": False,
        "sha256": {name: _digest(path) for name, path in tracked.items()
                   if os.path.exists(path)},
    }
    path = os.path.join(results_dir, "provenance.json")
    save(record, path)
    return path


# ---------------------------------------------------------------------------
# The in-file reference — the array path transcribed, computed on NumPy
# ---------------------------------------------------------------------------
#
# EVERY sweep comparison is kernel-bytes against THIS transcription evaluated on
# the host (IEEE subnormal-keep, the ship configuration's reference side); the
# ``reference`` leg is what pins the transcription against stepping.py itself on
# real objects, BEFORE any kernel is measured against it. The ``down_ghost`` /
# ``up_ghost`` / ``mask_fold`` knobs exist ONLY so the reference leg's
# discriminating controls can vary one arm at a time; the shipped path is the
# default in every argument.

#: The shipped arm is FIRST in each tuple and is the default in every argument.
#: The rest are the reference-leg controls and the MUTATION ANALOGUES — the host
#: half of the mutation leg evaluates the analogue of each planted defect and
#: measures whether the chosen needle makes it BYTE-VISIBLE, so a needle that
#: could not have caught its mutation is found here rather than on the device.
DOWN_GHOSTS = ("parity_row2",          # shipped: -phase * g[stored row 2]
               "metallic_zero",        # f1  the certified METALLIC arm
               "periodic_wrap",        # f2  the certified PERIODIC arm
               "plus_phase_row2",      # f5  parity dropped
               "parity_last_row",      # f3  ghost index n-1
               "parity_row_three",     # f4  ghost index 2+1; also f10's spelling
               "whole_lane_parity",    # f7  weight on the whole down lane
               "minus_lowering_row2")  # f6  the addend respelled as unary minus
UP_GHOSTS = ("zero",                   # shipped: the exact 0 of :1781-1783
             "parity_reflect",         # the folded-PERIODIC reflect row
             "periodic_wrap",          # the wrap
             "far_row")                # f8  served from stored row n-2


def reference_shift_down(field: np.ndarray, axis: int, kind: str,
                         phase: Optional[int], ghost: str = "parity_row2"):
    """``f[i-1]``: ``stepping._shift_down`` (:1787).

    PERIODIC wraps (:1819-1822); METALLIC serves an exact 0 at face 0
    (:1827-1829); MIRROR serves ``mirror_parity('D'+axis, axis, phase) *
    field[MIRROR_SOURCE_INDEX]`` (:1823-1826), which collapses to ``-phase``
    because every D component has Yee shift 1 on its own axis (fields.py:117,
    :214-219).
    """
    out = np.roll(field, 1, axis=axis)
    if kind == PERIODIC:
        return out
    if kind == METALLIC:
        out[_face(axis, 0)] = 0
        return out
    if kind != MIRROR:
        raise ValueError(f"boundary {kind!r} has no reference here")
    if ghost not in DOWN_GHOSTS:
        # A typo must NOT fall through to the shipped arm: a mutation analogue
        # that silently evaluated the shipped path would report NEEDLE-MISSED
        # for a needle that was never planted (the DISARMED failure mode).
        raise ValueError(f"unknown down ghost {ghost!r}; one of {DOWN_GHOSTS}")
    if ghost == "metallic_zero":
        out[_face(axis, 0)] = 0
        return out
    if ghost == "periodic_wrap":
        return out
    weight = mirror_parity("D" + AXES[axis], axis, int(phase))
    if ghost == "plus_phase_row2":
        weight = -weight
    row = fomod.MIRROR_SOURCE_INDEX
    if ghost == "parity_last_row":         # f3: reflects about the window top
        row = field.shape[axis] - 1
    elif ghost == "parity_row_three":      # f4 / f10: off by one row
        row = min(fomod.MIRROR_SOURCE_INDEX + 1, field.shape[axis] - 1)
    if ghost == "whole_lane_parity":       # f7: the weight escapes the ghost lane
        out = (np.float32(weight) * out).astype(np.float32)
        out[_face(axis, 0)] = np.float32(weight) * field[_face(axis, row)]
        return out
    if ghost == "minus_lowering_row2":
        # f6: the ghost addend respelled with a UNARY MINUS. Triton's
        # semantic.minus lowers -x as 0.0 - x (measured reachable at driver
        # level), so the respelling is `0.0 - ((0.0 - w) * v)`:
        # algebraically identical, and byte-identical too EXCEPT on signed
        # zeros, where 0.0 - (+0.0) == +0.0 while (-1.0) * (+0.0) == -0.0.
        value = field[_face(axis, row)]
        out[_face(axis, 0)] = np.float32(0.0) - (
            (np.float32(0.0) - np.float32(weight)) * value)
        return out
    out[_face(axis, 0)] = weight * field[_face(axis, row)]
    return out


def reference_shift_up(field: np.ndarray, axis: int, kind: str,
                       phase: Optional[int], reflect_row: Optional[int],
                       ghost: str = "zero"):
    """``f[i+1]``: ``stepping._shift_up`` (:1723) AS ``_offdiagonal_terms``
    CALLS IT (:1219-1220) — four arguments, no ``component``, no
    ``reflect_row``. So the folded-PERIODIC reflect branch (:1772-1780) cannot
    fire and BOTH mirror terminations take :1781-1783, an exact 0.0.

    ``ghost='parity_reflect'`` and ``'periodic_wrap'`` exist only as reference-leg
    controls: they are what the array path would serve if it passed those
    arguments, and they must DIFFER from the shipped transcription (see the
    module docstring's ARRAY-PATH FINDING).
    """
    out = np.roll(field, -1, axis=axis)
    if kind == PERIODIC:
        return out
    if kind == METALLIC:
        out[_face(axis, -1)] = 0
        return out
    if kind != MIRROR:
        raise ValueError(f"boundary {kind!r} has no reference here")
    if ghost not in UP_GHOSTS:
        raise ValueError(f"unknown up ghost {ghost!r}; one of {UP_GHOSTS}")
    if ghost == "periodic_wrap":
        return out
    if ghost == "far_row":
        # f8's analogue: the up ghost served from a stored row instead of the
        # exact zero. n-2 is the reflect row at an EVEN full count (the gate's
        # reference grids measure 10 for stored 12); the array path's row is
        # count-dependent, which is why the reflect claim itself rides the
        # `up_parity_reflect` control and this spelling is only the needle.
        out[_face(axis, -1)] = field[_face(axis, max(field.shape[axis] - 2, 0))]
        return out
    if ghost == "parity_reflect" and reflect_row is not None:
        out[_face(axis, -1)] = (-int(phase)) * field[_face(axis, reflect_row)]
        return out
    out[_face(axis, -1)] = 0
    return out


def reference_coupling(volumes, rows, own_axis, kinds, phases, reflect_rows,
                       wall_axes, component, down_ghost="parity_row2",
                       up_ghost="zero", mask_fold=False, mask_all=False):
    """``stepping._offdiagonal_terms`` (:1206-1224) + ``_mask_metallic_wall_
    coupling`` (:1227-1254): pair DOWN the partner axis on the raw partner
    volume, the coefficient multiply BETWEEN the shifts, the PRODUCT up the own
    axis, 0.25 scaling the sum applied last, offset 1 then offset 2, then face-0
    zeroing of the total on every WALL axis where the component's Yee shift is 0
    — and on NO mirrored axis (:1253).
    """
    total = None
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        coefficient = rows.get("E" + AXES[partner_axis])
        if coefficient is None:
            continue
        values = volumes["E" + AXES[partner_axis]]
        pair = values + reference_shift_down(
            values, partner_axis, kinds[partner_axis], phases[partner_axis],
            down_ghost)
        product = pair * coefficient
        term = 0.25 * (product + reference_shift_up(
            product, own_axis, kinds[own_axis], phases[own_axis],
            reflect_rows[own_axis], up_ghost))
        total = term if total is None else total + term
    if total is not None:
        iyee = IYEE_SHIFTS[component]
        for axis in range(3):
            if iyee[axis] != 0:
                continue
            if (mask_all or wall_axes[axis]
                    or (mask_fold and kinds[axis] == MIRROR)):
                total[_face(axis, 0)] = 0
    return total


def reference_update_e(state, coefficients, rows_by_component, inv_eps, kinds,
                       phases, reflect_rows, wall_axes, **ghosts) -> None:
    """The whole folded offdiag sub-step, in place on ``state``.

    ``stepping.update_E`` (:967-989), ``elif offdiagonal:`` branch (:972-979)
    with no poles: the D volumes ARE the sources, the row sum is diagonal-first,
    and the tail is ``_apply_constitutive_pml``'s no-scratch branch (:2083-2088).
    """
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        constitutive = volumes[target] * inv_eps[target]
        coupling = reference_coupling(
            volumes, rows_by_component.get(target, {}), own_axis, kinds, phases,
            reflect_rows, wall_axes, target, **ghosts)
        if coupling is not None:
            constitutive = constitutive + coupling
        fw = state["f_w_" + target]
        field = state[target]
        previous = fw.copy()
        fw[...] = constitutive
        field += coefficients["kps_" + AXES[own_axis]] * fw
        field -= coefficients["kms_" + AXES[own_axis]] * previous


# ---------------------------------------------------------------------------
# Real-object fabric
# ---------------------------------------------------------------------------

def row_form(name: str, shape, rng) -> Dict[str, Dict[str, np.ndarray]]:
    """The four row shapes the sweep and the reference grids draw from."""
    if name == "uniform_full":
        return {row: {partner: np.full(shape, INV_TENSOR[r, p], np.float32)
                      for p, partner in enumerate(E_NAMES) if p != r}
                for r, row in enumerate(E_NAMES)}
    if name == "varying_full":
        return {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                      for p, partner in enumerate(E_NAMES) if p != r}
                for r, row in enumerate(E_NAMES)}
    if name.startswith("only_row_"):
        row = "E" + name[-1]
        return {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                      for partner in E_NAMES if partner != row}}
    if name.startswith("single_"):
        _, row, partner = name.split("_")
        return {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)}}
    raise ValueError(f"unknown row form {name!r}")


def build_reference_fields(xp, spec: Dict[str, Any]):
    """Real ``Grid``/``Fields``/``PML`` with a real fold and installed rows.

    ``spec['grid']`` passes extra keyword arguments to :class:`Grid` itself —
    the ONLY way a refusal case may install ``beta`` or BFAST, because
    ``bfast_active`` is a read-only property over ``bfast_scaled_k``
    (grid.py:745) and poking it onto a built object raises.
    """
    symmetry = tuple(Mirror(letter, spec["phase"]) for letter in spec["fold"])
    grid = Grid(resolution=spec.get("resolution", 10.0),
                cell_size=spec["cell"], courant=spec["courant"],
                symmetry=symmetry, boundaries=spec.get("boundaries"),
                dimensions=spec.get("dimensions", 3), xp=xp,
                **dict(spec.get("grid") or {}))
    fields = Fields(grid=grid)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(SEED + spec.get("seed_offset", 0))
    if spec.get("inv_form") == "aliased":
        shared_inverse = xp.asarray(np.full(shape, 0.45, np.float32))
        shared_eps = xp.asarray(np.full(shape, 1.0 / 0.45, np.float32))
        inverse_map = {name: shared_inverse for name in E_NAMES}
        epsilon_map = {name: shared_eps for name in E_NAMES}
    else:
        inverse_map = {name: xp.asarray(np.full(shape, INV_TENSOR[i, i], np.float32))
                       for i, name in enumerate(E_NAMES)}
        epsilon_map = {name: xp.asarray(np.full(shape, 1.0 / INV_TENSOR[i, i],
                                                np.float32))
                       for i, name in enumerate(E_NAMES)}
    rows = row_form(spec["rows"], shape, rng)
    fields.set_epsilon_volumes(
        epsilon_map, inverse_map,
        chi1inv_offdiagonal={row: {p: xp.asarray(v) for p, v in partners.items()}
                             for row, partners in rows.items()})
    fields.enable_pml_storage()
    # A folded axis's PLANE face carries no layer (an absorber there would eat
    # the mirrored half); the driver's own arrangement, and PML refuses the
    # combination otherwise (stepping._require_consistent_pml).
    thickness = tuple(
        (0, 2) if grid.is_mirrored(axis)
        else ((2, 2) if shape[axis] >= 6 else (0, 0))
        for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def seed_state(fields, xp, kind: str, rng) -> None:
    shape = tuple(fields.grid.shape)
    if kind == "zero_init_row2":
        # The NON-VACUOUS zero-init needle: everything zero except stored row
        # MIRROR_SOURCE_INDEX of the folded axis, so the mirror ghost is the ONLY
        # path to a nonzero fold-plane E. A plain all-zero seed is VACUOUS —
        # measured: census 0 and a sign-flipped ghost invisible.
        folded = [a for a in range(3) if fields.grid.is_mirrored(a)]
        axis = folded[0] if folded else 0
        plane = rng.uniform(0.2, 0.6,
                            tuple(n for i, n in enumerate(shape) if i != axis)
                            ).astype(np.float32)
        for name in ALL_NAMES:
            getattr(fields, name)[...] = xp.asarray(np.zeros(shape, np.float32))
        for name in D_NAMES:
            getattr(fields, name)[_face(axis, fomod.MIRROR_SOURCE_INDEX)] = \
                xp.asarray(plane)
        return
    for name in ALL_NAMES:
        getattr(fields, name)[...] = xp.asarray(
            rng.uniform(-0.4, 0.4, shape).astype(np.float32))


def grid_wall_axes(grid) -> Tuple[int, int, int]:
    """The mask's own question (stepping.py:1279-1283), on a real grid — read
    through the certified family's own helper so the two cannot disagree."""
    return odmod.wall_mask_axes(grid)


def to_host(array) -> np.ndarray:
    return np.asarray(array if cp is None else cp.asnumpy(array))


def state_bytes(source, names=STATE_NAMES) -> bytes:
    return b"".join(to_host(source[n] if isinstance(source, dict)
                            else getattr(source, n)).tobytes() for n in names)


def negative_zero_census(state, names=STATE_NAMES) -> int:
    """How many stored words are NEGATIVE ZERO — not merely negative.

    The distinction is the whole clause, and one number carrying two meanings is
    exactly what the case discipline forbids. ``np.signbit`` over every word
    counts ordinary negative numbers too, which a random-seeded grid has in
    quantity (measured on the reference grids: about half of the 10368 stored
    words, while the true negative-zero count is 0) — so a sign-bit tally is >0
    for reasons that have nothing to do with the class the discipline names: a
    defect visible only in the SIGN OF A ZERO. Counting ``(x == 0) &
    signbit(x)`` is what makes census 0 mean "this case cannot see that class".

    The zero-init class's own gate: with every array zero the mirror ghost's
    parity sign is annihilated by the ``values + ghost`` add (+0.0 + -0.0 ==
    +0.0), the census is 0, and a sign-flipped ghost is invisible. Census 0 on a
    zero-init case means VACUOUS, not passed.
    """
    total = 0
    for name in names:
        array = np.asarray(state[name] if isinstance(state, dict)
                           else getattr(state, name))
        total += int(np.count_nonzero((array == 0) & np.signbit(array)))
    return total


def negative_word_tally(state, names=STATE_NAMES) -> int:
    """Stored words whose SIGN BIT is set, zeros and ordinary negatives alike.

    Recorded beside :func:`negative_zero_census` under its own name so the two
    can never be read for each other. It is a seed statistic, nothing more; no
    verdict reads it.
    """
    total = 0
    for name in names:
        array = np.asarray(state[name] if isinstance(state, dict)
                           else getattr(state, name))
        total += int(np.count_nonzero(np.signbit(array)))
    return total


# ---------------------------------------------------------------------------
# The reference grids
# ---------------------------------------------------------------------------
#
# Each grid is here because it is the ONLY witness to something: a termination,
# a count parity, a plane phase, a fold axis, a fold count, a reduced run, a
# Courant that is not a power of two, or a row form that puts the fold in a
# particular role.

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    dict(name="foldX_periodic_even_varying", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="varying_full"),
    dict(name="foldX_periodic_even_oddcount", fold="X", phase=1,
         cell=(2.1, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="varying_full", seed_offset=1),
    dict(name="foldX_periodic_odd_uniform", fold="X", phase=-1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="uniform_full", seed_offset=2),
    dict(name="foldY_metallic_even_varying", fold="Y", phase=1,
         cell=(1.2, 2.0, 1.2), boundaries="metallic", courant=0.35,
         rows="varying_full", seed_offset=3),
    dict(name="foldY_metallic_odd_varying", fold="Y", phase=-1,
         cell=(1.2, 2.0, 1.2), boundaries="metallic", courant=0.5,
         rows="varying_full", seed_offset=4),
    dict(name="foldZ_periodic_even_varying", fold="Z", phase=1,
         cell=(1.2, 1.2, 2.0), boundaries=None, courant=0.3125,
         rows="varying_full", seed_offset=5),
    dict(name="foldXY_periodic_even_varying", fold="XY", phase=1,
         cell=(2.0, 2.0, 1.2), boundaries=None, courant=0.35,
         rows="varying_full", seed_offset=6),
    dict(name="foldXY_periodic_odd_varying", fold="XY", phase=-1,
         cell=(2.1, 2.0, 1.2), boundaries=None, courant=0.35,
         rows="varying_full", seed_offset=7),
    dict(name="foldY_reduced_2d_varying", fold="Y", phase=1,
         cell=(1.6, 2.0, 0.0), boundaries=None, courant=0.35,
         dimensions=2, rows="varying_full", seed_offset=8),
    dict(name="foldX_periodic_even_aliased_inveps", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="varying_full", inv_form="aliased", seed_offset=9),
    dict(name="foldX_only_row_x_unreachable", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="only_row_x", seed_offset=10),
    dict(name="foldX_single_Ey_Ez_no_x_partner", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="single_Ey_Ez", seed_offset=11),
    dict(name="foldX_single_Ey_Ex_x_is_partner", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), boundaries=None, courant=0.35,
         rows="single_Ey_Ex", seed_offset=12),
    dict(name="foldZ_metallic_even_uniform", fold="Z", phase=1,
         cell=(1.2, 1.2, 2.0), boundaries="metallic", courant=0.35,
         rows="uniform_full", seed_offset=13),
    # THREE SIMULTANEOUS MIRROR PLANES. The predicate ADMITS this configuration
    # (measured: covered with the array-module clause spoofed), so a later
    # plan_step round would route it — and until this grid existed nothing in
    # the gate, the validator or the laptop tests carried a case with more than
    # two folded axes. Every partner axis is then folded, so all six row slots
    # take a ghosted down shift at once.
    dict(name="foldXYZ_periodic_even_varying", fold="XYZ", phase=1,
         cell=(2.0, 2.0, 2.0), boundaries=None, courant=0.35,
         rows="varying_full", seed_offset=14),
    dict(name="foldXYZ_metallic_odd_varying", fold="XYZ", phase=-1,
         cell=(2.0, 2.0, 2.0), boundaries="metallic", courant=0.3125,
         rows="varying_full", seed_offset=15),
)

REFERENCE_STEPS = 4

#: One control per ghost arm. Each must DIFFER from the shipped transcription on
#: at least one grid that can see it; a control that never differs anywhere is
#: VACUOUS and fails the leg.
CONTROLS: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("down_metallic_zero", dict(down_ghost="metallic_zero"),
     "the certified kernel's METALLIC arm: proves offdiag_update_e cannot serve "
     "a fold, i.e. that this family is not a redundant kernel"),
    ("down_periodic_wrap", dict(down_ghost="periodic_wrap"),
     "the periodic wrap: proves the fold is not simply a wrap"),
    ("down_sign_flipped", dict(down_ghost="plus_phase_row2"),
     "+phase instead of -phase: proves the parity weight is byte-visible"),
    ("up_parity_reflect", dict(up_ghost="parity_reflect"),
     "the folded-PERIODIC reflect row on the own axis: proves the array path "
     "really serves an exact zero there (ARRAY-PATH FINDING). Predicted "
     "IDENTICAL on a folded METALLIC axis, where there is no reflect row — "
     "recorded per grid, not required globally"),
    ("up_periodic_wrap", dict(up_ghost="periodic_wrap"),
     "the periodic wrap on the own axis: proves the zero is written, not "
     "inherited"),
    ("fold_plane_masked", dict(mask_fold=True),
     "the wall mask applied on the fold plane: proves the abstention at "
     "stepping.py:1282 is byte-visible (2.0e-02 by the array path's own "
     "measurement)"),
)


def run_reference(results: Dict[str, Any], out_path: str, xp,
                  backend_name: str) -> Dict[str, Any]:
    """Pin the in-file transcription against ``stepping.update_E`` on real
    objects, and run every discriminating control beside it."""
    leg: Dict[str, Any] = {"backend": backend_name, "grids": [],
                           "steps_per_grid": REFERENCE_STEPS}
    control_seen: Dict[str, int] = {name: 0 for name, _, _ in CONTROLS}
    for spec in REFERENCE_GRIDS:
        started = time.time()
        row: Dict[str, Any] = {"name": spec["name"]}
        fields, pml = build_reference_fields(xp, spec)
        grid = fields.grid
        rng = np.random.default_rng(SEED + 900 + spec.get("seed_offset", 0))
        seed_state(fields, xp, spec.get("seed", "random"), rng)
        kinds = tuple(stepping._boundary_kinds(grid, pml))
        phases = stepping._mirror_phases(grid)
        reflect = stepping._far_reflect_rows(grid)
        walls = grid_wall_axes(grid)
        coefficients = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}_h")
                        for axis in AXES for stem in ("kps", "kms")}
        rows_by = {name: fields.chi1inv_offdiagonal_for(name) for name in E_NAMES}
        inv_eps = {name: fields.inverse_epsilon_for(name) for name in E_NAMES}

        variants = {"shipped": {}}
        variants.update({name: kwargs for name, kwargs, _ in CONTROLS})
        states = {key: {n: to_host(getattr(fields, n)).copy() for n in ALL_NAMES}
                  for key in variants}
        identical = {key: True for key in variants}
        chain = hashlib.sha256()
        host_coefficients = {k: to_host(v) for k, v in coefficients.items()}
        host_rows = {c: {p: to_host(v) for p, v in r.items()}
                     for c, r in rows_by.items()}
        host_inv = {c: to_host(v) for c, v in inv_eps.items()}
        for step in range(REFERENCE_STEPS):
            stepping.update_E(fields, pml)
            for key, kwargs in variants.items():
                reference_update_e(states[key], host_coefficients, host_rows,
                                   host_inv, kinds, phases, reflect, walls,
                                   **kwargs)
                for name in STATE_NAMES:
                    ours = states[key][name]
                    theirs = to_host(getattr(fields, name))
                    if key == "shipped":
                        chain.update(ours.tobytes())
                    if ours.tobytes() != theirs.tobytes():
                        identical[key] = False
                        if key == "shipped":
                            row.setdefault("first_divergence", {
                                "step": step + 1, "array": name,
                                "max_abs": float(np.max(np.abs(ours - theirs)))})
                # Every variant stays on the engine's own D trajectory, so the
                # only difference measured is the ghost arm under test.
                for name in D_NAMES:
                    states[key][name][...] = to_host(getattr(fields, name))

        row["kinds"] = list(kinds)
        row["shape"] = list(map(int, grid.shape))
        row["stored_cells"] = [int(grid.stored_cells(a)) for a in range(3)]
        row["owned_cells"] = [int(grid.owned_cells(a)) for a in range(3)]
        row["reflect_rows"] = [None if r is None else int(r) for r in reflect]
        row["wall_axes"] = list(map(int, walls))
        row["ghost_weights"] = list(fomod.mirror_ghost_weights(grid))
        row["courant"] = float(spec["courant"])
        row["courant_is_power_of_two"] = bool(
            abs(spec["courant"] * 2 ** 8 - round(spec["courant"] * 2 ** 8)) < 1e-12
            and (round(spec["courant"] * 2 ** 8) & (round(spec["courant"] * 2 ** 8) - 1)) == 0)
        row["identical"] = identical["shipped"]
        row["state_sha256_chain"] = chain.hexdigest()
        row["controls"] = {}
        for name, _kwargs, why in CONTROLS:
            caught = not identical[name]
            row["controls"][name] = {"differs": caught, "why": why}
            control_seen[name] += int(caught)
        # Non-vacuity, per grid: the coupling must actually be nonzero, and the
        # feature must change the bytes against a coefficient-dropped control.
        dropped = {n: to_host(getattr(fields, n)).copy() for n in ALL_NAMES}
        reference_update_e(dropped, host_coefficients, {}, host_inv, kinds,
                           phases, reflect, walls)
        row["coupling_changes_bytes"] = (
            state_bytes(dropped) != state_bytes(states["shipped"]))
        row["max_abs_E"] = max(float(np.max(np.abs(to_host(getattr(fields, n)))))
                               for n in E_NAMES)
        folded = [a for a in range(3) if grid.is_mirrored(a)]
        row["fold_plane_max_abs_E"] = max(
            float(np.max(np.abs(to_host(getattr(fields, n))[_face(folded[0], 0)])))
            for n in E_NAMES) if folded else None
        row["far_face_max_abs_D"] = max(
            float(np.max(np.abs(to_host(getattr(fields, n))[_face(folded[0], -1)])))
            for n in D_NAMES) if folded else None
        # The census is the NEGATIVE-ZERO count, through the gate's own helper —
        # never a bare sign-bit tally, which on these random-seeded grids counts
        # ordinary negative numbers (about half the stored words) and would read
        # as if the reference leg saw the signed-zero class in quantity. It does
        # not: on a continuous seed the true census is 0 by construction, and the
        # class's witnesses are the sweep's zero-init and signed-zero needles.
        # Both numbers are recorded, under names that cannot be read for each
        # other; no verdict in this leg reads either.
        host_state = {n: to_host(getattr(fields, n)) for n in STATE_NAMES}
        row["negative_zero_census"] = negative_zero_census(host_state)
        row["negative_word_tally"] = negative_word_tally(host_state)
        row["census_note"] = ("a continuous random seed carries no negative "
                              "zero; the signed-zero class is the synthetic "
                              "leg's zero_init_row2 / signed_zero_row2 needles")
        row["seconds"] = round(time.time() - started, 3)
        leg["grids"].append(row)
        log(f"[reference:{backend_name}] {spec['name']}: identical="
            f"{row['identical']} coupling_live={row['coupling_changes_bytes']} "
            f"controls_differing="
            f"{sum(1 for c in row['controls'].values() if c['differs'])}"
            f"/{len(CONTROLS)} ({row['seconds']} s)")
        results.setdefault("reference", {})[backend_name] = leg
        save(results, out_path)

    leg["control_grids_differing"] = control_seen
    leg["vacuous_controls"] = [name for name, count in control_seen.items()
                               if count == 0]
    leg["pass"] = (all(g["identical"] for g in leg["grids"])
                   and all(g["coupling_changes_bytes"] for g in leg["grids"])
                   and not leg["vacuous_controls"])
    results.setdefault("reference", {})[backend_name] = leg
    save(results, out_path)
    log(f"[reference:{backend_name}] leg pass={leg['pass']} "
        f"vacuous_controls={leg['vacuous_controls']}")
    return leg


# ---------------------------------------------------------------------------
# The reachability / identity leg — no device needed
# ---------------------------------------------------------------------------

#: (row form, plane phase, boundaries, DECLARED expectation, why).
#:
#: THE EXPECTATION IS DECLARED BY HAND, never computed from the predicate the
#: leg arbitrates — a case whose expectation came from
#: ``mirror_arm_is_reachable`` could only ever confirm that function against
#: itself.
#:
#: THE FOURTH ROW IS THE COUNTEREXAMPLE THAT MAKES THE CLAIM SLOT-LEVEL. A
#: single slot ``E_{a+1} <- E_{a+2}`` leaves a live row that is not ``E_a``
#: while NO live slot takes ``E_a`` as its partner, so the fold enters neither
#: role and the bytes agree. A row-level reading of the reachability statement
#: answers "reachable" there and is wrong; before this row existed the leg drew
#: only from ``only_row_*`` (both partners live) and ``varying_full``, and the
#: counterexample never reached the leg that arbitrates the claim.
_IDENTITY_FORMS: Tuple[Tuple[str, int, Optional[str], bool, str], ...] = (
    ("own_row", 1, None, True,
     "every live slot sits in row E_a, whose own axis the fold is: unreachable"),
    ("own_row", -1, "metallic", True,
     "the same, on the OTHER termination and the odd plane phase"),
    ("next_row", 1, None, False,
     "row E_{a+1} carries BOTH partners, so one of them is E_a: reachable"),
    ("single_next_far", 1, None, True,
     "single slot E_{a+1} <- E_{a+2}: the live row is not E_a, yet no live "
     "slot takes E_a as its partner — the fold is in NEITHER role"),
    ("single_next_own", 1, None, False,
     "single slot E_{a+1} <- E_a: the fold IS the partner axis"),
    ("varying_full", 1, None, False,
     "every slot live: the fold is a partner axis of two rows"),
)


def _identity_row_form(form: str, axis: int) -> str:
    own = "E" + AXES[axis]
    nxt = "E" + AXES[(axis + 1) % 3]
    far = "E" + AXES[(axis + 2) % 3]
    if form == "own_row":
        return f"only_row_{AXES[axis]}"
    if form == "next_row":
        return f"only_row_{AXES[(axis + 1) % 3]}"
    if form == "single_next_far":
        return f"single_{nxt}_{far}"
    if form == "single_next_own":
        return f"single_{nxt}_{own}"
    return form


IDENTITY_CASES: Tuple[Dict[str, Any], ...] = tuple(
    dict(name=(f"fold{letter}_{_identity_row_form(form, axis)}_"
               f"{'metallic' if boundaries else 'periodic'}"
               f"_phase{'p' if phase > 0 else 'm'}"),
         fold=letter, phase=phase,
         cell=tuple(2.0 if i == axis else 1.2 for i in range(3)),
         boundaries=boundaries, courant=0.35,
         rows=_identity_row_form(form, axis),
         seed_offset=40 + axis * len(_IDENTITY_FORMS) + index,
         expect_identical=expect, why=why)
    for axis, letter in enumerate("XYZ")
    for index, (form, phase, boundaries, expect, why)
    in enumerate(_IDENTITY_FORMS)
)


def run_identity_host(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """MIRROR vs METALLIC ghost code on the SAME grid, coefficients and seeds.

    The reachability claim, measured: identical exactly when NO LIVE ROW SLOT
    takes the folded axis's component as its PARTNER. The negative half is what
    keeps the positive half from being vacuous, and the SLOT-LEVEL
    COUNTEREXAMPLE (a live row that is not ``E_a`` whose only surviving slot
    still never takes ``E_a`` as a partner) is what keeps the claim from being
    satisfiable by the coarser row-level reading. A leg carrying no such case is
    recorded VACUOUS and fails.
    """
    leg: Dict[str, Any] = {"cases": [], "calls_per_case": 6,
                           "calls_note": "two drives (MIRROR, METALLIC) x 3 "
                                         "chained sub-step calls each"}
    for spec in IDENTITY_CASES:
        started = time.time()
        fields, pml = build_reference_fields(np, spec)
        grid = fields.grid
        rng = np.random.default_rng(SEED + 700 + spec["seed_offset"])
        seed_state(fields, np, "random", rng)
        kinds = list(stepping._boundary_kinds(grid, pml))
        axis = "XYZ".index(spec["fold"])
        phases = stepping._mirror_phases(grid)
        reflect = stepping._far_reflect_rows(grid)
        walls = grid_wall_axes(grid)
        coefficients = {f"{stem}_{a}": getattr(pml, f"{stem}_{a}_h")
                        for a in AXES for stem in ("kps", "kms")}
        rows_by = {n: fields.chi1inv_offdiagonal_for(n) for n in E_NAMES}
        inv_eps = {n: fields.inverse_epsilon_for(n) for n in E_NAMES}

        def drive(active_kinds):
            state = {n: getattr(fields, n).copy() for n in ALL_NAMES}
            for _ in range(3):
                reference_update_e(state, coefficients, rows_by, inv_eps,
                                   tuple(active_kinds), phases, reflect, walls)
            return state_bytes(state)

        as_mirror = drive(kinds)
        swapped = list(kinds)
        swapped[axis] = METALLIC
        as_metallic = drive(swapped)
        reachable, notes = fomod.mirror_arm_is_reachable(grid, fields)
        identical = as_mirror == as_metallic
        row = {
            "name": spec["name"], "rows": spec["rows"], "phase": spec["phase"],
            "fold_axis": axis, "kinds": kinds,
            "mirror_equals_metallic": identical,
            "predicate_says_reachable": reachable,
            "predicate_notes": list(notes),
            "expected_identical": bool(spec["expect_identical"]),
            "why": spec["why"],
            "agrees": identical == (not reachable)
                      and identical == bool(spec["expect_identical"]),
            "seconds": round(time.time() - started, 3),
        }
        leg["cases"].append(row)
        log(f"[identity_host] {spec['name']}: MIRROR==METALLIC={identical} "
            f"predicate_reachable={reachable} agrees={row['agrees']}")
        results["identity_host"] = leg
        save(results, out_path)
    leg["pass"] = all(c["agrees"] for c in leg["cases"])
    leg["positive_cases"] = sum(1 for c in leg["cases"]
                                if c["mirror_equals_metallic"])
    leg["negative_cases"] = sum(1 for c in leg["cases"]
                                if not c["mirror_equals_metallic"])
    # THE SLOT-LEVEL CONTROL. A case whose live row is NOT the folded axis's own
    # row and whose bytes still agree is the only configuration that separates
    # "no live SLOT takes E_a as its partner" from "every live ROW is E_a". A
    # leg without one cannot tell the two readings apart.
    leg["slot_level_counterexamples"] = sum(
        1 for c in leg["cases"]
        if c["rows"].startswith("single_")
        and not c["rows"].endswith("_E" + AXES[c["fold_axis"]])
        and c["mirror_equals_metallic"])
    if leg["positive_cases"] == 0 or leg["negative_cases"] == 0:
        leg["pass"] = False
        leg["vacuous"] = ("one side of the reachability claim carried no case; "
                          "a leg with only identical rows, or only differing "
                          "rows, measures nothing")
    if leg["slot_level_counterexamples"] == 0:
        leg["pass"] = False
        leg["vacuous_slot_level"] = (
            "no case separates the SLOT-level claim from the ROW-level one: "
            "every identical row here also has every live row equal to E_a, so "
            "a row-level predicate would pass this leg unchanged")
    results["identity_host"] = leg
    save(results, out_path)
    log(f"[identity_host] leg pass={leg['pass']} "
        f"({leg['positive_cases']} identical / {leg['negative_cases']} differing"
        f"; {leg['slot_level_counterexamples']} slot-level counterexamples)")
    return leg


# ---------------------------------------------------------------------------
# The refusal enumeration leg
# ---------------------------------------------------------------------------

def _covered(fields, pml) -> Tuple[bool, Tuple[str, ...]]:
    verdict = fomod.folded_offdiag_constitutive_coverage(fields, pml)
    return verdict.covered, verdict.reasons


def run_refusals(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Every predicted refusal, by name. 0 admissions, and the two disjointness
    seams asserted from BOTH sides."""
    leg: Dict[str, Any] = {"cases": []}

    def record(name: str, covered: bool, reasons, expect_covered: bool,
               must_match: str, note: str = "") -> None:
        """``must_match`` IS REQUIRED — there is no ``None`` default.

        With one, a case's verdict collapses to ``covered == expect_covered``,
        which the array-module clause alone guarantees on this host: the row is
        then byte-indistinguishable from a pristine grid and measures nothing.
        The thin-fold case was exactly that.
        """
        matched = any(must_match in reason for reason in reasons)
        row = {"name": name, "covered": covered,
               "expect_covered": expect_covered,
               "reasons": list(reasons)[:8],
               "must_match": must_match,
               "reason_matched": matched, "note": note,
               "pass": covered == expect_covered and matched}
        leg["cases"].append(row)
        log(f"[refusals] {name}: covered={covered} expected={expect_covered} "
            f"matched={matched}")
        results["refusals"] = leg
        save(results, out_path)

    base = dict(name="base", fold="X", phase=1, cell=(2.0, 1.2, 1.2),
                boundaries=None, courant=0.35, rows="varying_full")

    # 1. The laptop/NumPy backend is itself a refusal.
    fields, pml = build_reference_fields(np, base)
    covered, reasons = _covered(fields, pml)
    record("numpy_backend", covered, reasons, False, "not cupy")

    # 2. THE STALE-DOCSTRING GUARD. stepping.py:1219-1220, fields.py:1237-1241
    #    and driver.py:1202-1204 all claim the installer refuses folded rows.
    #    It does not (fields.py:1262-1310) — and this family exists BECAUSE it
    #    does not. Asserted as a positive fact about the engine, not a coverage
    #    verdict.
    installed = bool(fields.has_offdiagonal_epsilon)
    stepped = True
    try:
        stepping.update_E(fields, pml)
    except Exception as exc:  # noqa: BLE001
        stepped = False
        stepped_error = repr(exc)
    else:
        stepped_error = None
    leg["cases"].append({
        "name": "stale_docstring_guard",
        "folded_rows_installed": installed,
        "stepping_update_E_ran": stepped,
        "error": stepped_error,
        "pass": installed and stepped,
        "note": "the installer does NOT refuse folded off-diagonal rows "
                "(fields.py:1262-1310) and stepping.update_E steps them; the "
                "three docstrings that claim otherwise are stale; the fix "
                "belongs in those files",
    })
    log(f"[refusals] stale_docstring_guard: installed={installed} "
        f"stepped={stepped}")
    save(results, out_path)

    # 3. DISJOINTNESS, seam one: the certified off-diagonal predicate must
    #    refuse this same folded run (coverage._grid_reasons clause 5).
    od_covered = odmod.offdiag_constitutive_coverage(fields, pml)
    od_fold_named = any("fold" in r or "mirror" in r for r in od_covered.reasons)
    leg["cases"].append({
        "name": "seam_offdiag_refuses_the_fold",
        "covered": od_covered.covered, "expect_covered": False,
        "fold_named_in_reasons": od_fold_named,
        "reasons": list(od_covered.reasons)[:8],
        "pass": (not od_covered.covered) and od_fold_named,
    })
    # 4. DISJOINTNESS, seam two: the folded ELEMENT-WISE predicate must refuse
    #    the rows (symmetry.py:814-817).
    sym_covered = symmod.folded_constitutive_coverage(fields, pml, "E")
    sym_rows_named = any("off-diagonal" in r for r in sym_covered.reasons)
    leg["cases"].append({
        "name": "seam_symmetry_refuses_the_rows",
        "covered": sym_covered.covered, "expect_covered": False,
        "rows_named_in_reasons": sym_rows_named,
        "reasons": list(sym_covered.reasons)[:8],
        "pass": (not sym_covered.covered) and sym_rows_named,
    })
    log("[refusals] disjointness seams recorded")
    save(results, out_path)

    # 5. No surviving row slot: that folded run is symmetry's, not this file's.
    plain = dict(base, rows="varying_full", name="no_rows")
    fields2, pml2 = build_reference_fields(np, plain)
    fields2._chi1inv_offdiagonal = {}
    covered, reasons = _covered(fields2, pml2)
    record("no_surviving_row_slot", covered, reasons, False,
           "no off-diagonal chi1inv row survived installation")

    # 6. A registered polarization.
    fields3, pml3 = build_reference_fields(np, base)

    class _FakeState:
        def drives(self, name):
            return name == "Ex"

        def driven(self):
            return ("Ex",)
        susceptibility = type("S", (), {"kind": "lorentzian"})()

    fields3.polarizations = (_FakeState(),)
    covered, reasons = _covered(fields3, pml3)
    record("registered_polarization", covered, reasons, False,
           "a susceptibility is registered")

    # 7. A nonlinearity. CHI3, not chi2, and that is an engine fact rather than
    #    a harness convenience: `Fields._require_chi2_respects_the_mirror_planes`
    #    (fields.py:887-916) refuses a chi2 on any component a mirror plane
    #    forces odd — a chi2 medium is not centrosymmetric, so the plane is not
    #    a symmetry of the material (measured there: a folded chi2 run breaks by
    #    4.1e-02 in one step while a folded chi3 run is bit-identical). So the
    #    only nonlinearity this family can even be handed on a fold is chi3, and
    #    that is the one the clause has to refuse.
    fields4, pml4 = build_reference_fields(np, base)
    fields4.set_nonlinear_volumes(
        {}, {"Ex": np.full(tuple(fields4.grid.shape), 0.01, np.float32)})
    covered, reasons = _covered(fields4, pml4)
    record("instantaneous_nonlinearity_chi3", covered, reasons, False, "chi2/chi3",
           "chi2 cannot reach a folded run at all (fields.py:887-916); chi3 can, "
           "and is refused here")

    # 8. No active PML: update_E takes the plain assignment branch (:993).
    fields5, pml5 = build_reference_fields(np, base)
    covered, reasons = _covered(fields5, None)
    record("no_active_pml", covered, reasons, False, "no active PML layer")

    # 9. A folded axis with too few stored cells for the ghost's stored row 2 —
    #    stepping._mirror_source raises below that.
    #
    #    THE ROUTE MATTERS AND IS MEASURED. A folded PERIODIC axis BOTTOMS OUT AT
    #    THREE stored cells: cell sizes 0.05 / 0.1 / 0.15 / 0.2 at resolution 10
    #    all give shape (3, 12, 12), so `stored_cells <= MIRROR_SOURCE_INDEX = 2`
    #    is UNREACHABLE that way and a case built on it records the backend
    #    clause alone — byte-indistinguishable from a pristine grid, i.e. a case
    #    that measures nothing. The condition IS reachable on a folded METALLIC
    #    axis: cell 0.15 with `boundaries='metallic'` gives shape (2, 12, 12).
    #    The reason string is asserted by MATCH, never left to `must_match=None`,
    #    which would reduce the case's verdict to the backend clause.
    thin = dict(base, cell=(0.15, 1.2, 1.2), boundaries="metallic",
                name="thin_fold")
    fields6, pml6 = build_reference_fields(np, thin)
    stored = int(fields6.grid.stored_cells(0))
    covered, reasons = _covered(fields6, pml6)
    record("folded_axis_too_thin", covered, reasons, False,
           "is folded with 2 stored cells",
           f"stored cells {stored} <= MIRROR_SOURCE_INDEX "
           f"{fomod.MIRROR_SOURCE_INDEX}; the ghost images stored row 2 "
           f"(stepping.py:1583-1588). Reached through a folded METALLIC axis "
           f"because a folded PERIODIC axis bottoms out at 3 stored cells")
    leg["cases"][-1]["stored_cells_on_the_folded_axis"] = stored
    if stored > fomod.MIRROR_SOURCE_INDEX:
        leg["cases"][-1]["pass"] = False
        leg["cases"][-1]["vacuous"] = (
            f"the grid this case built is not thin: {stored} stored cells on "
            f"the folded axis, so the clause under test cannot fire")
    save(results, out_path)

    # 10. Complex storage — folded complex + offdiag is Phase B's composed leg.
    fields7, pml7 = build_reference_fields(np, base)
    fields7.force_complex_fields = True
    covered, reasons = _covered(fields7, pml7)
    record("complex_storage", covered, reasons, False, "force_complex_fields")

    # 11. A row volume aliasing an output — reachable through the public
    #     installer, which keeps the caller's array without copying.
    fields8, pml8 = build_reference_fields(np, base)
    fields8._chi1inv_offdiagonal = {"Ex": {"Ey": fields8.Ez}}
    covered, reasons = _covered(fields8, pml8)
    record("row_aliases_output", covered, reasons, False, "aliases output")

    # 12. Beta and BFAST — BUILT THROUGH THE GRID CONSTRUCTOR, never poked onto
    #     a built object. `bfast_active` is a read-only property over
    #     `bfast_scaled_k` (grid.py:745), so `object.__setattr__(grid,
    #     'bfast_active', True)` RAISES; the previous spelling caught that
    #     exception and `continue`d, which dropped the BFAST case silently while
    #     the leg still reported every case passing. A case that cannot be built
    #     is now a FAILURE with the exception recorded, never a silent skip.
    #     beta is MEEP's out-of-plane 2-D wavevector and the Grid REFUSES it on
    #     a 3-D cell (fields.cpp:546-547), so its case is built on the reduced
    #     2-D grid the reference leg already carries — the configuration the
    #     engine can actually reach, rather than an attribute poked past the
    #     constructor's own validation.
    beta_spec = dict(base, fold="Y", cell=(1.6, 2.0, 0.0), dimensions=2,
                     grid=dict(beta=0.3))
    for name, spec, needle in (
            ("grid_beta", beta_spec, "beta"),
            ("grid_bfast_active",
             dict(base, grid=dict(bfast_scaled_k=(0.3, 0.0, 0.0))), "BFAST")):
        try:
            fields9, pml9 = build_reference_fields(np, spec)
            covered, reasons = _covered(fields9, pml9)
        except Exception as exc:  # noqa: BLE001
            leg["cases"].append({
                "name": name, "covered": None, "expect_covered": False,
                "pass": False, "reasons": [repr(exc)],
                "note": "the case could not be BUILT, so the clause it exists "
                        "to witness has no witness; a build failure is a leg "
                        "failure, never a silent skip"})
            log(f"[refusals] {name}: BUILD FAILED {exc!r}")
            save(results, out_path)
            continue
        record(name, covered, reasons, False, needle)

    leg["admissions"] = sum(1 for c in leg["cases"]
                            if c.get("covered") and not c.get("expect_covered"))
    leg["pass"] = all(c.get("pass", False) for c in leg["cases"])
    results["refusals"] = leg
    save(results, out_path)
    log(f"[refusals] leg pass={leg['pass']} admissions={leg['admissions']}")
    return leg


# ---------------------------------------------------------------------------
# The device legs' fabric — enumeration, the HOST HALF, the device compare
# ---------------------------------------------------------------------------
#
# WHY EVERY DEVICE LEG IS SPLIT IN TWO HALVES. This is a grouping choice; the
# array path forces nothing here. Each leg below is written so that everything
# EXCEPT the Triton launch and the uint32 compare runs on a laptop: the case
# enumeration, the reference computed through the transcription the ``reference``
# leg pinned against ``stepping.update_E``, and every case's NON-VACUITY
# assertion. A case that cannot see what it was built to see therefore fails
# HERE — on a host with no GPU, in the round that wrote it — instead of spending
# device time to report a green row that measured nothing. The device half is
# one ``if`` away and consumes the same case objects.
#
# THE HOST HALF MAKES NO BYTE-IDENTITY CLAIM. It compares the reference against
# CONTROLS, never against kernel bytes; ``certified`` stays False and the leg's
# ``pass`` stays False until a device compares. That is the whole point of the
# exit-75 convention this file already uses.

C_PERIODIC = fomod.CODE_PERIODIC
C_METALLIC = fomod.CODE_METALLIC
C_MIRROR_METALLIC = fomod.CODE_MIRROR_METALLIC
C_MIRROR_PERIODIC = fomod.CODE_MIRROR_PERIODIC

#: The kind each boundary CODE resolves to in the reference. Both mirror codes
#: resolve to MIRROR: identical behaviour in this sub-step is the module's
#: predicted null (no ownership mask, no reflect row), and the `identity` leg
#: holds the equality with PTX evidence rather than collapsing the two.
KIND_OF_CODE: Dict[int, str] = {
    C_PERIODIC: PERIODIC,
    C_METALLIC: METALLIC,
    C_MIRROR_METALLIC: MIRROR,
    C_MIRROR_PERIODIC: MIRROR,
}

SWEEP_SHAPES = ((16, 16, 16), (13, 17, 11), (1, 160, 160), (11, 13, 15))

#: Real-layer coefficient tables are cut at these Courants. 0.35 and 0.3125 are
#: NOT powers of two (the sweep mandate). dtdx does not enter THIS sub-step's
#: arithmetic — the mandate lands here, in the tables the layer produces.
LAYER_COURANTS = (0.5, 0.35, 0.3125)

#: (label, per-axis boundary codes, per-axis declared mirror phase). Covers both
#: TERMINATIONS (a sweep carrying only one proves nothing about the other), both
#: PLANE PHASES, a fold on each of X/Y/Z, TWO planes at once, and ZERO folded
#: axes — the reduction row the `identity` leg turns into a claim.
SWEEP_FOLDS: Tuple[Tuple[str, Tuple[int, int, int],
                         Tuple[Optional[int], ...]], ...] = (
    ("unfolded", (C_PERIODIC, C_PERIODIC, C_PERIODIC), (None, None, None)),
    ("foldX_periodic_even", (C_MIRROR_PERIODIC, C_PERIODIC, C_PERIODIC),
     (1, None, None)),
    ("foldX_metallic_odd", (C_MIRROR_METALLIC, C_METALLIC, C_PERIODIC),
     (-1, None, None)),
    ("foldY_periodic_odd", (C_PERIODIC, C_MIRROR_PERIODIC, C_PERIODIC),
     (None, -1, None)),
    ("foldY_metallic_even", (C_METALLIC, C_MIRROR_METALLIC, C_PERIODIC),
     (None, 1, None)),
    ("foldZ_periodic_even", (C_PERIODIC, C_PERIODIC, C_MIRROR_PERIODIC),
     (None, None, 1)),
    ("foldZ_metallic_odd", (C_METALLIC, C_PERIODIC, C_MIRROR_METALLIC),
     (None, None, -1)),
    ("foldXY_two_planes", (C_MIRROR_PERIODIC, C_MIRROR_METALLIC, C_PERIODIC),
     (1, -1, None)),
    # THREE PLANES AT ONCE — admitted by the predicate and, before this row, in
    # no case anywhere in the family. Every one of the six row slots then takes a
    # ghosted down shift, and the three ghost weights are not all equal (mixed
    # phases), so a kernel that shared one weight across axes is byte-visible.
    ("foldXYZ_three_planes",
     (C_MIRROR_PERIODIC, C_MIRROR_METALLIC, C_MIRROR_PERIODIC), (1, -1, 1)),
)


def sweep_config_viable(shape: Sequence[int],
                        codes: Sequence[int]) -> Optional[str]:
    """Why this shape cannot carry these codes — by name, never silently."""
    for axis in range(3):
        n = int(shape[axis])
        if n == 1 and codes[axis] != C_PERIODIC:
            return (f"axis {axis} is REDUCED (n=1) and resolves to PERIODIC: "
                    f"the wrap returns the same plane and the pair is 2*g, "
                    f"MEEP's stride(d)=0 double read")
        if codes[axis] in fomod.MIRROR_CODES and n <= fomod.MIRROR_SOURCE_INDEX:
            return (f"axis {axis} is folded with {n} stored cells; "
                    f"stepping._mirror_source raises at "
                    f"<= {fomod.MIRROR_SOURCE_INDEX} (:1536-1541)")
    return None


def sweep_wall_axes(codes: Sequence[int]) -> Tuple[int, int, int]:
    """The mask's own question on a synthetic configuration: ``is_metallic and
    not is_mirrored`` (stepping.py:1282) — so a MIRROR code is 0 by
    construction, which is what leaves the fold plane's coupling alive."""
    return tuple(1 if code == C_METALLIC else 0 for code in codes)


def sweep_ghost_weights(codes: Sequence[int],
                        phases: Sequence[Optional[int]]) -> Tuple[float, ...]:
    """``-phase`` on a folded axis, read through ``fields.mirror_parity`` on the
    partner component the array path actually passes (stepping.py:1243-1245),
    never a hard-coded sign; 1.0 on an unfolded axis (never consulted)."""
    weights = []
    for axis, code in enumerate(codes):
        if code in fomod.MIRROR_CODES:
            weights.append(float(mirror_parity("D" + AXES[axis], axis,
                                               int(phases[axis]))))
        else:
            weights.append(1.0)
    return tuple(weights)


def make_host_state(shape, rng, inv_form: str = "distinct"):
    state = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
             for name in ALL_NAMES}
    if inv_form == "distinct":
        inv_eps = {name: rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
                   for name in E_NAMES}
    elif inv_form == "aliased":   # the isotropic install: three references
        shared = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
        inv_eps = {name: shared for name in E_NAMES}
    else:
        raise ValueError(f"unknown inverse-epsilon form {inv_form!r}")
    return state, inv_eps


def synthetic_flat_coefficients(shape, rng) -> Dict[str, np.ndarray]:
    """Seeded kps/kms per axis, never 1.0, so a swapped axis or a dropped
    coefficient cannot reproduce bit-for-bit; kms nonzero everywhere, so the
    store-order defect is visible at every cell."""
    flat = {}
    for axis, name in enumerate(AXES):
        for stem in ("kps", "kms"):
            flat[f"{stem}_{name}"] = rng.uniform(
                0.5, 1.0, size=shape[axis]).astype(np.float32)
    return flat


def layer_flat_coefficients(shape, courant: float) -> Optional[Dict[str, Any]]:
    """The half-integer kps/kms of a REAL layer at ``courant``.

    The vectors are cut at the STORED extent on each axis, which is what a
    folded axis needs (symmetry.py:318-320 builds a folded axis's tables at the
    folded length); a synthetic sweep gets the same lengths by construction
    because the table is cut on a grid of exactly this shape."""
    dims = 3 - sum(1 for n in shape if n == 1)
    cell = tuple(n / 10.0 if n > 1 else 0.0 for n in shape)
    try:
        grid = Grid(resolution=10.0, cell_size=cell,
                    dimensions=max(1, dims), courant=courant)
    except ValueError:
        return None
    if tuple(grid.shape) != tuple(shape):
        return None
    thickness = tuple((2, 2) if shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    return {f"{stem}_{axis}": np.ascontiguousarray(
        np.asarray(getattr(pml, f"{stem}_{axis}_h")).reshape(-1)
        ).astype(np.float32)
        for axis in AXES for stem in ("kps", "kms")}


def broadcast(flat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Flat per-axis vectors -> the broadcast shapes the reference multiplies
    by, the same shapes ``PML._reshape_for_broadcast`` hands the array path."""
    out = {}
    for axis, name in enumerate(AXES):
        for stem in ("kps", "kms"):
            vector = np.asarray(flat[f"{stem}_{name}"]).reshape(-1)
            shape = [1, 1, 1]
            shape[axis] = vector.size
            out[f"{stem}_{name}"] = vector.reshape(shape)
    return out


def fold_plane_max_abs(state, axis: int) -> float:
    return max(float(np.max(np.abs(np.asarray(state[name])[_face(axis, 0)])))
               for name in E_NAMES)


def seed_zero_init_row2(state, axis: int, rng) -> None:
    """THE ZERO-INIT NEEDLE: zero everywhere EXCEPT stored row
    ``MIRROR_SOURCE_INDEX`` of the folded axis, so the mirror ghost is the ONLY
    path to a nonzero fold-plane E.

    TWO THINGS ARE MEASURED, NOT ASSUMED, AND THE BACKGROUND'S SIGN FOLLOWS
    FROM THE SECOND:

    1. A plain all-zero seed is VACUOUS — with every array zero the ghost is
       the only nonzero term nowhere, the fold plane stays at zero and a
       sign-flipped ghost is invisible. Row ``MIRROR_SOURCE_INDEX`` alone fixes
       that: it makes the mirror ghost the sole path to a nonzero fold-plane E.

    2. The BACKGROUND IS NEGATIVE ZERO, and that is load-bearing rather than
       decorative. Measured on this laptop with a ``+0.0`` background: the
       stored negative-zero census is 0 on all four folded configurations,
       because the row sum's ``diag + total`` add annihilates the ghost's parity
       sign (``+0.0 + -0.0 == +0.0``) — the same annihilation that makes an
       all-zero seed vacuous, one level in. A ``-0.0`` background keeps both
       addends negative (``D*us`` is ``-0.0`` under a positive inverse epsilon,
       and the coupling's ``pair*coefficient`` is ``-0.0`` wherever the row
       coefficient is negative), so the sign SURVIVES into stored ``f_w`` and
       the in-run census can see it. Census 0 fails the case (the discipline's
       clause); this is what makes the census meet it honestly instead of by
       relabelling.

    Everything stays finite: a zero or a number in [0.2, 0.6] (platform fact
    (f) — no needle may produce a NaN and then compare raw words)."""
    shape = np.asarray(state["Ex"]).shape
    plane = rng.uniform(0.2, 0.6, tuple(n for i, n in enumerate(shape)
                                        if i != axis)).astype(np.float32)
    for name in ALL_NAMES:
        state[name] = np.full(shape, -0.0, np.float32)
    for name in D_NAMES:
        state[name][_face(axis, fomod.MIRROR_SOURCE_INDEX)] = plane


def seed_signed_zero_row2(state, axis: int, rng) -> None:
    """THE SIGNED-ZERO NEEDLE — built for platform fact (b), f6's decider.

    Every stored word starts at NEGATIVE zero and the ghost row carries EXACT
    +0.0 on alternating lines. With a ghost weight of -1 the shipped spelling
    loads +0.0 and multiplies: ``(-1.0) * (+0.0) == -0.0``, so ``-0.0 + -0.0 ==
    -0.0`` survives into the stored row sum. A unary-minus respelling lowers as
    ``0.0 - (+0.0) == +0.0``, and ``-0.0 + +0.0 == +0.0`` — the same magnitude,
    a different word. Nothing here can produce a NaN or an infinity (platform
    fact (f)): every value is a zero or a number in [0.2, 0.6]."""
    shape = np.asarray(state["Ex"]).shape
    for name in ALL_NAMES:
        state[name] = np.full(shape, -0.0, np.float32)
    for name in D_NAMES:
        plane = rng.uniform(0.2, 0.6, tuple(n for i, n in enumerate(shape)
                                            if i != axis)).astype(np.float32)
        plane[::2] = np.float32(0.0)     # exact POSITIVE zeros in the ghost row
        state[name][_face(axis, fomod.MIRROR_SOURCE_INDEX)] = plane


def seed_cancellation(state, rows, inv_eps, kinds, phases, reflect_rows,
                      wall_axes) -> Dict[str, float]:
    """Make ``gs*us ~ -coupling`` on Ex, so the row-sum association is
    BYTE-VISIBLE in f32 — the assert-where-the-defect-is-visible rule. Values
    stay in the normal range; the underflow variant is the subnormal leg's."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    total = reference_coupling(volumes, rows.get("Ex", {}), 0, kinds, phases,
                               reflect_rows, wall_axes, "Ex")
    if total is None or not np.any(total):
        raise AssertionError("cancellation class: Ex carries no coupling — the "
                             "class would be vacuous")
    noise = (1.0 + 1e-3 * np.random.default_rng(SEED + 77).uniform(
        -1.0, 1.0, size=total.shape)).astype(np.float32)
    state["Dx"] = np.ascontiguousarray(
        (-(total / inv_eps["Ex"]) * noise).astype(np.float32))
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    total = reference_coupling(volumes, rows.get("Ex", {}), 0, kinds, phases,
                               reflect_rows, wall_axes, "Ex")
    row_sum = state["Dx"] * inv_eps["Ex"] + total
    scale = float(np.max(np.abs(total)))
    residual = float(np.max(np.abs(row_sum)))
    if scale == 0.0 or residual > 0.1 * scale:
        raise AssertionError(f"cancellation class failed to cancel: residual "
                             f"{residual:.3e} vs coupling scale {scale:.3e}")
    return {"coupling_scale": scale, "row_sum_residual": residual}


def fold_is_a_live_partner_axis(codes: Sequence[int],
                                rows: Dict[str, Dict[str, Any]]) -> bool:
    """Is a folded axis the PARTNER axis of a surviving row slot?

    The mirror rule enters ONLY through the partner-axis down shift
    (stepping.py:1243-1245), and axis ``a`` is never row ``E_a``'s partner axis,
    so this predicate is exactly 'the fold can change a byte'. Measured against
    the reference case by case, not assumed."""
    for row, partner in fomod.ROW_SLOTS:
        if (rows.get(row) or {}).get(partner) is None:
            continue
        if codes[E_NAMES.index(partner)] in fomod.MIRROR_CODES:
            return True
    return False


def _sweep_case(name: str, shape, fold: str, rows: str, *, inv_form="distinct",
                amplitude="normal", guard=False, coefficients="synthetic",
                courant=None, seed_offset=0) -> Dict[str, Any]:
    codes, phases = next((c, p) for label, c, p in SWEEP_FOLDS if label == fold)
    return {"name": name, "shape": tuple(shape), "fold": fold,
            "codes": tuple(codes), "phases": tuple(phases), "rows": rows,
            "inv_form": inv_form, "amplitude": amplitude, "guard": bool(guard),
            "coefficients": coefficients, "courant": courant,
            "seed_offset": seed_offset}


def sweep_cases() -> List[Dict[str, Any]]:
    """The whole synthetic tranche, enumerated as data so a test can assert
    what it covers without running it."""
    cases: List[Dict[str, Any]] = []
    offset = 0
    for guard in (False, True):
        for shape in SWEEP_SHAPES:
            for label, _codes, _phases in SWEEP_FOLDS:
                for form in ("uniform_full", "varying_full"):
                    offset += 1
                    cases.append(_sweep_case(
                        f"{label}_{'x'.join(str(n) for n in shape)}_{form}"
                        f"_guard{int(guard)}", shape, label, form,
                        guard=guard, seed_offset=offset))
    # Single-slot arms placing the fold in BOTH roles, and in neither.
    for label, form, why in (
            ("foldX_periodic_even", "single_Ey_Ex", "fold IS a partner axis"),
            ("foldX_periodic_even", "single_Ey_Ez", "fold in neither role"),
            ("foldX_metallic_odd", "only_row_x", "fold is the row's OWN axis"),
            ("foldY_periodic_odd", "single_Ez_Ey", "fold IS a partner axis"),
            ("foldZ_metallic_odd", "single_Ex_Ez", "fold IS a partner axis"),
            ("foldZ_periodic_even", "only_row_z", "fold is the row's OWN axis")):
        offset += 1
        cases.append(_sweep_case(f"{label}_{form}", (13, 17, 11), label, form,
                                 seed_offset=offset))
        cases[-1]["why"] = why
    # The aliased inverse-epsilon install, on a fold.
    for label in ("foldX_periodic_even", "foldZ_metallic_odd"):
        offset += 1
        cases.append(_sweep_case(f"{label}_aliased_inveps", (16, 16, 16), label,
                                 "varying_full", inv_form="aliased",
                                 seed_offset=offset))
    # The CANCELLATION class — the row-sum association's byte-visible home.
    for label, shape in (("foldX_periodic_even", (13, 17, 11)),
                         ("foldXY_two_planes", (16, 16, 16))):
        offset += 1
        cases.append(_sweep_case(f"{label}_cancellation", shape, label,
                                 "uniform_full", amplitude="cancellation",
                                 seed_offset=offset))
    # The ZERO-INIT needle (row 2 alone) on both terminations and both phases,
    # and the SIGNED-ZERO needle that decides f6.
    for label in ("foldX_periodic_even", "foldX_metallic_odd",
                  "foldY_periodic_odd", "foldZ_periodic_even"):
        offset += 1
        cases.append(_sweep_case(f"{label}_zero_init_row2", (11, 13, 15), label,
                                 "varying_full", amplitude="zero_init_row2",
                                 seed_offset=offset))
    for label in ("foldX_periodic_even", "foldZ_periodic_even"):
        offset += 1
        cases.append(_sweep_case(f"{label}_signed_zero_row2", (11, 13, 15),
                                 label, "varying_full",
                                 amplitude="signed_zero_row2",
                                 seed_offset=offset))
    # REAL-LAYER coefficient tables, including two non-power-of-two Courants.
    for shape in SWEEP_SHAPES:
        for courant in LAYER_COURANTS:
            offset += 1
            cases.append(_sweep_case(
                f"foldY_periodic_odd_layer_{courant}_"
                f"{'x'.join(str(n) for n in shape)}", shape,
                "foldY_periodic_odd", "varying_full", coefficients="layer",
                courant=courant, seed_offset=offset))
    return cases


def one_sweep_case(spec: Dict[str, Any], device_ok: bool,
                   kernel: Any = None) -> Dict[str, Any]:
    """One sweep case: the host half always, the device compare when there is
    a device. Returns the case record; ``error`` marks a VACUOUS or broken
    case and fails the leg."""
    case: Dict[str, Any] = {
        "name": spec["name"], "shape": list(spec["shape"]),
        "fold": spec["fold"], "codes": list(spec["codes"]),
        "phases": [None if p is None else int(p) for p in spec["phases"]],
        "row_form": spec["rows"], "inv_form": spec["inv_form"],
        "amplitude": spec["amplitude"], "guard": spec["guard"],
        "coefficient_source": spec["coefficients"],
        "courant": spec["courant"],
        "courant_is_power_of_two": (
            None if spec["courant"] is None
            else bool(abs(spec["courant"] * 2 ** 8
                          - round(spec["courant"] * 2 ** 8)) < 1e-12
                      and (round(spec["courant"] * 2 ** 8)
                           & (round(spec["courant"] * 2 ** 8) - 1)) == 0)),
        "compared": False,
    }
    shape = tuple(spec["shape"])
    codes = tuple(spec["codes"])
    phases = tuple(spec["phases"])
    skip = sweep_config_viable(shape, codes)
    if skip:
        case["skipped"] = skip
        return case
    kinds = tuple(KIND_OF_CODE[code] for code in codes)
    walls = sweep_wall_axes(codes)
    weights = sweep_ghost_weights(codes, phases)
    reflect: Tuple[Optional[int], ...] = (None, None, None)
    folded = [axis for axis in range(3) if codes[axis] in fomod.MIRROR_CODES]
    case["wall_axes"] = list(walls)
    case["ghost_weights"] = list(weights)
    case["folded_axes"] = folded

    rng = np.random.default_rng(SEED + 4100 + spec["seed_offset"])
    state, inv_eps = make_host_state(shape, rng, spec["inv_form"])
    rows = row_form(spec["rows"], shape, rng)
    if spec["amplitude"] == "zero_init_row2":
        if not folded:
            case["skipped"] = "the zero-init needle needs a folded axis"
            return case
        seed_zero_init_row2(state, folded[0], rng)
    elif spec["amplitude"] == "signed_zero_row2":
        if not folded:
            case["skipped"] = "the signed-zero needle needs a folded axis"
            return case
        seed_signed_zero_row2(state, folded[0], rng)
    elif spec["amplitude"] == "cancellation":
        if "Ex" not in rows:
            case["skipped"] = "the cancellation class needs an Ex row"
            return case
        case["cancellation"] = seed_cancellation(state, rows, inv_eps, kinds,
                                                 phases, reflect, walls)

    if spec["coefficients"] == "synthetic":
        flat = synthetic_flat_coefficients(shape, rng)
    else:
        flat = layer_flat_coefficients(shape, float(spec["courant"]))
        if flat is None:
            case["skipped"] = (f"no real layer reproduces shape {shape} at "
                               f"courant {spec['courant']}")
            return case
    coefficients = broadcast(flat)

    def drive(**ghosts) -> Dict[str, np.ndarray]:
        scratch = {name: state[name].copy() for name in ALL_NAMES}
        reference_update_e(scratch, coefficients, rows, inv_eps, kinds, phases,
                           reflect, walls, **ghosts)
        return scratch

    reference = drive()

    # NON-VACUITY 1 — the coupling is nonzero somewhere.
    couplings = [reference_coupling(
        {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]},
        rows.get(name, {}), axis, kinds, phases, reflect, walls, name)
        for axis, name in enumerate(E_NAMES)]
    case["coupling_live"] = bool(any(c is not None and bool(np.any(c))
                                     for c in couplings))
    if not case["coupling_live"]:
        case["error"] = "VACUOUS: no component's coupling is nonzero anywhere"
        return case

    # NON-VACUITY 2 — the reference differs from the coefficient-dropped
    # (diagonal-engine) control, so no case can pass with the feature dark.
    dropped = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(dropped, coefficients, {}, inv_eps, kinds, phases,
                       reflect, walls)
    case["coupling_changes_bytes"] = any(
        reference[name].tobytes() != dropped[name].tobytes()
        for name in STATE_NAMES)
    if not case["coupling_changes_bytes"]:
        case["error"] = ("VACUOUS: the coupling left every byte equal to the "
                         "diagonal engine on this seed")
        return case

    # NON-VACUITY 3 — the FOLD's own arm. Predicted from the row slots, measured
    # against the certified METALLIC arm. Disagreement is a finding, not a pass.
    predicted = fold_is_a_live_partner_axis(codes, rows)
    case["fold_arm_predicted_visible"] = bool(predicted)
    if folded:
        metallic_arm = drive(down_ghost="metallic_zero")
        measured = any(reference[name].tobytes() != metallic_arm[name].tobytes()
                       for name in STATE_NAMES)
        case["fold_arm_measured_visible"] = bool(measured)
        if bool(measured) != bool(predicted):
            case["error"] = (
                f"the fold arm's visibility disagrees with the row slots: "
                f"predicted {predicted}, measured {measured}")
            return case
    case["max_abs_E"] = max(float(np.max(np.abs(reference[n])))
                            for n in E_NAMES)
    case["negative_zero_census"] = negative_zero_census(reference)
    if folded:
        case["fold_plane_max_abs_E"] = fold_plane_max_abs(reference, folded[0])

    # The two needle classes carry their OWN gates: a census of 0 is vacuous.
    if spec["amplitude"] in ("zero_init_row2", "signed_zero_row2"):
        # Measure all three facts, THEN judge: an artifact that recorded only
        # the first failure could not say whether the needle was blind or the
        # census merely quiet.
        flipped = drive(down_ghost="plus_phase_row2")
        case["sign_flip_visible"] = any(
            reference[n].tobytes() != flipped[n].tobytes()
            for n in STATE_NAMES)
        zeroed = drive(down_ghost="metallic_zero")
        case["metallic_zero_visible"] = any(
            reference[n].tobytes() != zeroed[n].tobytes()
            for n in STATE_NAMES)
        if case["negative_zero_census"] == 0:
            case["error"] = ("VACUOUS: the needle produced no stored negative "
                             "zero, so this class cannot see a sign-bit defect "
                             "(census 0 is vacuous, not passed)")
            return case
        if spec["amplitude"] == "zero_init_row2":
            if not case["fold_plane_max_abs_E"]:
                case["error"] = ("VACUOUS: the zero-init needle left the fold "
                                 "plane at zero")
                return case
            if not (case["sign_flip_visible"]
                    and case["metallic_zero_visible"]):
                case["error"] = ("VACUOUS: the zero-init needle cannot see the "
                                 "ghost's sign or its presence")
                return case
        else:
            lowered = drive(down_ghost="minus_lowering_row2")
            case["unary_minus_lowering_visible"] = any(
                reference[n].tobytes() != lowered[n].tobytes()
                for n in STATE_NAMES)
            # RECORDED EITHER WAY (platform fact (b)); the mutation leg reads
            # this field to decide whether f6 can be asserted or only recorded.
    if not np.all(np.isfinite(np.concatenate(
            [reference[n].ravel() for n in STATE_NAMES]))):
        case["error"] = ("a needle produced a non-finite word; platform fact "
                         "(f) forbids comparing raw NaN words")
        return case

    if not device_ok:
        case["why_not_compared"] = (
            "no CUDA device in this round: the host half ran (enumeration, "
            "reference, non-vacuity); NO byte-identity claim is made")
        return case

    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    if spec["inv_form"] == "aliased":
        shared = cp.asarray(inv_eps["Ex"])
        for name in E_NAMES:
            arrays["inv_eps_" + name] = shared
    else:
        for name in E_NAMES:
            arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
    plan = fomod.plan_folded_offdiagonal_constitutive_from_arrays(
        arrays, {k: cp.asarray(v) for k, v in flat.items()},
        {row: {partner: cp.asarray(volume)
               for partner, volume in partners.items()}
         for row, partners in rows.items()},
        codes, walls, weights, kernel=kernel)
    case["row_mask"] = list(plan.row_mask)
    case["ghost_axes"] = list(plan.ghost_axes)
    plan.run(guard=spec["guard"])
    cp.cuda.runtime.deviceSynchronize()
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in STATE_NAMES})
    case["compared"] = True
    return case


def summarize_sweep(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ran = [c for c in cases if not c.get("skipped") and not c.get("error")]
    compared = [c for c in ran if c.get("compared")]
    guarded = [c for c in compared if c.get("guard") is False]
    fusion_on = [c for c in compared if c.get("guard") is True]
    return {
        "cases": len(cases),
        "host_half_ran": len(ran),
        "skipped": sum(1 for c in cases if c.get("skipped")),
        "errors": sum(1 for c in cases if c.get("error")),
        "compared": len(compared),
        "identical": sum(int(c["verdict"]["bit_identical"]) for c in compared),
        "guarded_ran": len(guarded),
        "guarded_identical": sum(int(c["verdict"]["bit_identical"])
                                 for c in guarded),
        "fusion_on_ran": len(fusion_on),
        "fusion_on_identical": sum(int(c["verdict"]["bit_identical"])
                                   for c in fusion_on),
        "folded_cases_with_a_live_fold_arm": sum(
            1 for c in ran if c.get("fold_arm_measured_visible")),
    }


def certify_sweep(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The sweep's VERDICT fields, from its counts. Separate from the loop so a
    laptop test can drive every outcome without a device.

    CERTIFICATION IS THE FUSION-OFF ROWS ALONE. The certified configuration is
    ``ENABLE_FP_FUSION = False`` (module docstring point F, and this file's own
    CONFIGURATION line); the sweep's ``guard`` axis RECORDS the fusion-on
    outcome and must never be merged into the verdict. Merging them made a
    byte-perfect kernel report ``certified: false`` on this tranche's own
    measured expectation (offdiag 0/28 fusion-on rows identical, and this
    family's tail is that shape plus a ghost-lane multiply).

    THE FUSION AXIS CARRIES THE CERTIFIED FAMILY'S NON-VACUITY CONTROL
    (gate_triton_offdiag.py:1708-1710): if the fusion-on rows never diverged,
    the guard axis measured nothing and "we certify fusion-off" is a
    distinction this sweep did not observe.
    """
    leg["host_half_pass"] = leg["errors"] == 0 and leg["host_half_ran"] > 0
    leg["certified"] = bool(leg["guarded_ran"]
                            and leg["guarded_ran"] == leg["guarded_identical"])
    leg["certified_configuration"] = "ENABLE_FP_FUSION=False (guard off)"
    leg["fusion_on_identical_recorded_not_merged"] = {
        "ran": leg["fusion_on_ran"], "identical": leg["fusion_on_identical"]}
    leg["fusion_control_diverged"] = bool(
        leg["fusion_on_ran"]
        and leg["fusion_on_identical"] != leg["fusion_on_ran"])
    leg.pop("fusion_control_note", None)
    if leg["fusion_on_ran"] and not leg["fusion_control_diverged"]:
        leg["fusion_control_note"] = (
            "fusion-control-never-diverged: every fusion-on row was "
            "byte-identical, so this sweep does not distinguish the "
            "configuration it claims to certify")
    leg["pass"] = bool(leg["host_half_pass"] and leg["certified"]
                       and (leg["fusion_on_ran"] == 0
                            or leg["fusion_control_diverged"]))
    return leg


def run_synthetic(results: Dict[str, Any], out_path: str, device_ok: bool,
                  kernel: Any = None, label: str = "gate") -> Dict[str, Any]:
    specs = sweep_cases()
    cases: List[Dict[str, Any]] = []
    leg: Dict[str, Any] = {}
    for index, spec in enumerate(specs, start=1):
        started = time.time()
        case = one_sweep_case(spec, device_ok, kernel=kernel)
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        if case.get("skipped"):
            log(f"[synthetic] {index}/{len(specs)} {case['name']}: "
                f"SKIPPED {case['skipped'][:60]}")
        elif case.get("error"):
            log(f"[synthetic] {index}/{len(specs)} {case['name']}: "
                f"ERROR {case['error'][:70]}")
        elif case.get("compared"):
            log(f"[synthetic] {index}/{len(specs)} {case['name']}: "
                f"identical={case['verdict']['bit_identical']} "
                f"({case['seconds']} s)")
        else:
            log(f"[synthetic] {index}/{len(specs)} {case['name']}: HOST HALF "
                f"coupling_live={case['coupling_live']} "
                f"fold_arm={case.get('fold_arm_measured_visible')} "
                f"({case['seconds']} s)")
        leg = certify_sweep(summarize_sweep(cases))
        leg["cases"] = cases
        leg["label"] = label
        results["synthetic"] = leg
        save(results, out_path)
    log(f"[synthetic] host half {leg['host_half_ran']} ran / "
        f"{leg['errors']} errors / {leg['skipped']} skipped; "
        f"compared {leg['compared']}")
    return leg


def run_subnormal(results: Dict[str, Any], out_path: str,
                  device_ok: bool, kernel: Any = None) -> Dict[str, Any]:
    """The cancellation class scaled into UNDERFLOW — reported separately under
    the stamped policy, never merged into the sweep's account."""
    # The policy NAME is read from the seam, never spelled here: this run may be
    # cut under either policy, and a hardcoded label would put the keep-era name
    # on flush-era bytes.
    leg: Dict[str, Any] = {"rows": [], "policy": gate.active_policy_name()}
    for label, shape in (("foldX_periodic_even", (13, 17, 11)),
                         ("foldZ_metallic_odd", (11, 13, 15))):
        codes, phases = next((c, p) for name, c, p in SWEEP_FOLDS
                             if name == label)
        rng = np.random.default_rng(SEED + 5100)
        state, inv_eps = make_host_state(shape, rng)
        rows = row_form("varying_full", shape, rng)
        kinds = tuple(KIND_OF_CODE[code] for code in codes)
        walls = sweep_wall_axes(codes)
        weights = sweep_ghost_weights(codes, phases)
        reflect = (None, None, None)
        tiny = np.float32(1e-38)
        for name in ALL_NAMES:
            state[name] = (state[name] * tiny).astype(np.float32)
        flat = synthetic_flat_coefficients(shape, rng)
        reference = {name: state[name].copy() for name in ALL_NAMES}
        reference_update_e(reference, broadcast(flat), rows, inv_eps, kinds,
                           phases, reflect, walls)
        subnormals = int(sum(
            int(np.count_nonzero((np.abs(reference[name]) < 2.0 ** -126)
                                 & (reference[name] != 0)))
            for name in STATE_NAMES))
        row: Dict[str, Any] = {"fold": label, "shape": list(shape),
                               "subnormal_words": subnormals,
                               "compared": False}
        if subnormals == 0:
            row["error"] = ("VACUOUS: the seed produced no subnormal word, so "
                            "the class the policy question lives in is absent")
        elif device_ok:
            arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
            for name in E_NAMES:
                arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
            plan = fomod.plan_folded_offdiagonal_constitutive_from_arrays(
                arrays, {k: cp.asarray(v) for k, v in flat.items()},
                {r: {p: cp.asarray(v) for p, v in partners.items()}
                 for r, partners in rows.items()},
                codes, walls, weights, kernel=kernel)
            plan.run(guard=False)
            cp.cuda.runtime.deviceSynchronize()
            row["verdict"] = combine({name: bit_compare(arrays[name],
                                                        reference[name])
                                      for name in STATE_NAMES})
            row["compared"] = True
        else:
            row["why_not_compared"] = ("no CUDA device in this round; the "
                                       "class was seeded and counted only")
        leg["rows"].append(row)
        log(f"[subnormal] {label}: subnormal_words={subnormals} "
            f"compared={row['compared']}")
        results["subnormal"] = leg
        save(results, out_path)
    leg["host_half_pass"] = all("error" not in row for row in leg["rows"])
    leg["compared"] = sum(1 for row in leg["rows"] if row.get("compared"))
    leg["certified"] = bool(leg["rows"]) and all(
        row.get("compared") and row["verdict"]["bit_identical"]
        for row in leg["rows"])
    leg["pass"] = bool(leg["host_half_pass"] and leg["certified"])
    results["subnormal"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# identity — the reduction row, the unreachable arm, and the predicted null
# ---------------------------------------------------------------------------

def _module_source(module) -> str:
    # encoding IS LOAD-BEARING. This file's docstrings carry non-ASCII, the
    # measurement hosts run under an ASCII locale, and `open(..., "r")` takes its
    # codec from the locale: the mutation and identity legs died on
    # `UnicodeDecodeError` on the device box while passing on the laptop. Every
    # text read and write in this gate names utf-8 for that reason.
    with open(module.__file__, "r", encoding="utf-8") as handle:
        return handle.read()


def _normalise(text: str) -> str:
    return " ".join(text.split())


_KERNEL_BLOCK_HEAD = "if triton is not None:\n\n    @triton.jit"
_KERNEL_BLOCK_TAIL = "else:  # pragma: no cover - the laptop path"


def kernel_block_source(source: Optional[str] = None) -> str:
    """The COMPILED surface of the module: the three ``@triton.jit`` bodies.

    Cut from the file itself so a claim about what the kernel branches on is
    measured against the text that is compiled, not against a docstring. The
    constant block above it (which does mention both mirror codes, because it
    binds them) is deliberately excluded.
    """
    text = _module_source(fomod) if source is None else source
    parts = text.split(_KERNEL_BLOCK_HEAD, 1)
    if len(parts) != 2:
        raise AssertionError(
            "the kernel block's opening moved; this cut is measured against "
            "the shipped file and must not silently return the whole module")
    body = parts[1].split(_KERNEL_BLOCK_TAIL, 1)
    if len(body) != 2:
        raise AssertionError("the kernel block's closing moved")
    return body[0]


def strip_comments_and_docstrings(text: str) -> str:
    """The EXECUTABLE text of a block — comments and docstrings removed.

    The shipped kernel's mirror arm is reached through a bare ``else`` whose
    COMMENT reads ``# MIRROR_METALLIC or MIRROR_PERIODIC``; scanning raw text
    for those tokens would report the kernel as distinguishing the two codes on
    the strength of a comment. Both failure modes of this helper are loud rather
    than silent: strip too little and the shipped scan finds a token (the check
    fails), strip too much and f10's control finds none (the check fails).
    """
    out: List[str] = []
    in_doc = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_doc:
            if '"""' in stripped:
                in_doc = False
            continue
        if stripped.startswith('"""'):
            if not (stripped.endswith('"""') and len(stripped) > 5):
                in_doc = True
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


#: The tokens a kernel would have to spell to give the two MIRROR codes
#: different behaviour. The shipped kernel reaches the mirror arm through a bare
#: ``else`` after ``BC* == PERIODIC`` / ``elif BC* == METALLIC``, so neither
#: appears in the compiled surface; f10's mutant introduces one, which is the
#: control that keeps this check from being vacuous.
MIRROR_CODE_TOKENS: Tuple[str, ...] = ("MIRROR_METALLIC", "MIRROR_PERIODIC")


def mirror_codes_share_one_arm(source: Optional[str] = None) -> Dict[str, Any]:
    """Docstring point D's PREDICTED NULL, as a host measurement.

    NOT a reference-level equality. ``reference_update_e`` takes ``kinds``, and
    ``KIND_OF_CODE`` maps both mirror codes to ``MIRROR`` — so driving it twice
    with the two codes passes IDENTICAL ARGUMENTS and the equality is an
    identity of inputs, not a measurement (verified by instrumenting both
    drives: same kinds, same phases, same walls, same row and inverse-epsilon
    objects). What CAN be measured without a device is the two facts the null
    rests on:

    1. the compiled surface never distinguishes the two codes — neither
       ``MIRROR_METALLIC`` nor ``MIRROR_PERIODIC`` appears in it, so both land
       in the same ``else`` arm after ``== PERIODIC`` / ``== METALLIC``; and
    2. two plans built from the two codes agree in EVERY launched field except
       the boundary code itself — same row mask, same ghost axes, same weights,
       same wall axes, same block and launch grid.

    Fact 1 carries its own control: f10's mutant DOES spell a mirror code, so a
    check that could not see the difference is caught here rather than trusted.
    THE BYTE CLAIM REMAINS THE DEVICE HALF'S, together with the PTX evidence
    that the two specializations are two builds (platform fact (c)).
    """
    shipped = _module_source(fomod) if source is None else source
    block = strip_comments_and_docstrings(kernel_block_source(shipped))
    present = tuple(token for token in MIRROR_CODE_TOKENS if token in block)
    mutant_block = strip_comments_and_docstrings(kernel_block_source(
        mutate_f10_two_mirror_codes_differ(shipped)[0]))
    control = tuple(token for token in MIRROR_CODE_TOKENS
                    if token in mutant_block)
    shape = (12, 14, 10)
    rng = np.random.default_rng(SEED + 6200)
    state, inv_eps = make_host_state(shape, rng)
    rows = row_form("varying_full", shape, rng)
    flat = synthetic_flat_coefficients(shape, rng)
    plans = {}
    for label, code in (("mirror_metallic", C_MIRROR_METALLIC),
                        ("mirror_periodic", C_MIRROR_PERIODIC)):
        codes = (code, C_PERIODIC, C_PERIODIC)
        arrays = {name: state[name].copy() for name in ALL_NAMES}
        for name in E_NAMES:
            arrays["inv_eps_" + name] = inv_eps[name]
        plan = fomod.plan_folded_offdiagonal_constitutive_from_arrays(
            arrays, dict(flat), rows, codes, sweep_wall_axes(codes),
            sweep_ghost_weights(codes, (1, None, None)))
        plans[label] = {
            "row_mask": list(plan.row_mask), "ghost_axes": list(plan.ghost_axes),
            "ghost_weights": list(plan.ghost_weights),
            "wall_axes": list(plan.wall_axes), "block": int(plan.block),
            "n_elem": int(plan.n_elem), "shape": list(plan.shape),
        }
    differing = [key for key in plans["mirror_metallic"]
                 if plans["mirror_metallic"][key] != plans["mirror_periodic"][key]]
    return {
        "compiled_surface_mentions_a_mirror_code": list(present),
        "f10_mutant_mentions_one": list(control),
        "check_is_armed": bool(control) and not present,
        "plan_fields": plans,
        "plan_fields_differing_apart_from_the_code": differing,
        "identical": (not present) and bool(control) and not differing,
        "why": ("MIRROR_METALLIC and MIRROR_PERIODIC differ in this sub-step "
                "only through arms it does not have: update_E never calls "
                "_mask_non_owned_cells, and _offdiagonal_terms calls _shift_up "
                "with four arguments, so the folded-PERIODIC reflect branch "
                "(stepping.py:1772-1780) cannot fire"),  # stepping.py live lines for the frozen device-text citation(s) in this string: 1772-1780->1819-1827
        "not_a_byte_claim": ("the host half measures the SOURCE and the PLAN; "
                             "the kernel bytes and the PTX evidence are the "
                             "device half's"),
    }


def verbatim_else_arm() -> Dict[str, Any]:
    """CLAIM (i)'s STRUCTURAL half, checkable without a device.

    ``_folded_offdiag_term``'s ``MG=False`` arm must be
    ``offdiag_update_e._offdiag_term``'s body VERBATIM, so an unfolded axis
    compiles to the certified instruction sequence by CONSTRUCTION rather than
    by trusting a compiler to fold a multiply by 1.0 away. Text, not bytes: the
    bytes are the device half's job.
    """
    ours = _module_source(fomod).split("def _folded_offdiag_term(", 1)[1]
    ours = ours.split("@triton.jit", 1)[0]
    theirs = _module_source(odmod).split("def _offdiag_term(", 1)[1]
    theirs = theirs.split("@triton.jit", 1)[0]
    ours_body = ours.split('"""')[2]
    theirs_body = theirs.split('"""')[2]
    ours_else = ours_body.split("else:", 1)[1]
    ours_statements, ours_return = ours_else.split("return", 1)
    theirs_statements, theirs_return = theirs_body.split("return", 1)
    return {
        "else_arm_is_verbatim": _normalise(ours_statements)
                                == _normalise(theirs_statements),
        "return_is_verbatim": _normalise(ours_return)
                              == _normalise(theirs_return),
        "certified_statements": _normalise(theirs_statements),
        "folded_else_statements": _normalise(ours_statements),
    }


# ---------------------------------------------------------------------------
# DEVICE MACHINERY — the mutant's file, its launch count, and its build identity
# ---------------------------------------------------------------------------
#
# Written in the device round rather than before it, for the reason
# DEVICE_HALF_TODO gave: none of it can be exercised at all without hardware,
# and a harness written blind is how a leg arrives green and vacuous. Every
# piece below is either exercised by a device leg here or is a PURE function the
# laptop tests drive.

_MUTANT_DIR: Optional[str] = None

#: Every mutant module written this run: path, entry point, sha256. The artifact
#: carries it so "the mutant was compiled from a real file" is checkable after
#: the fact rather than asserted.
MUTANT_FILES: List[Dict[str, str]] = []


class CountingKernel:
    """Proof the kernel under test actually LAUNCHED.

    A mutation leg whose mutated kernel never ran reports a hollow pass, which
    is worse than no leg at all — the certified families record three separate
    harness-disarm defects of exactly that shape. ``launches`` is a COUNT and
    every device row states it; a row with ``launches == 0`` is classified
    NO-LAUNCH and fails, never NEEDLE-MISSED (which would blame the needle for
    a harness fault).

    The wrapper delegates ``cache`` and ``cache_key`` so the build-identity
    check reads the real ``JITFunction`` through it.
    """

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def launch(*args, **kwargs):
            self.launches += 1
            return launcher(*args, **kwargs)

        return launch

    @property
    def cache(self):
        return getattr(self.kernel, "cache", None)

    @property
    def cache_key(self):
        return getattr(self.kernel, "cache_key", "unavailable")


def specialization_keys(kernel: Any) -> List[str]:
    """Every compiled specialization's CACHE KEY, or an empty list.

    Deliberately defensive: the shape of ``JITFunction.cache`` is a Triton
    internal, and a gate that CRASHES on a missing attribute reports nothing at
    all. An empty list is recorded as unavailable and the caller says so rather
    than reading it as evidence.
    """
    keys: List[str] = []
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        for per_device in cache.values():
            names = getattr(per_device, "keys", None)
            if callable(names):
                keys.extend(str(key) for key in names())
    return sorted(keys)


def ptx_texts(kernel: Any) -> List[str]:
    """Every compiled specialization's PTX, or an empty list when unreadable."""
    texts: List[str] = []
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        for per_device in cache.values():
            values = getattr(per_device, "values", None)
            for compiled in (values() if callable(values) else []):
                asm = getattr(compiled, "asm", None)
                if isinstance(asm, dict) and isinstance(asm.get("ptx"), str):
                    texts.append(asm["ptx"])
    return texts


def compile_mutant_module(source: str, label: str) -> Tuple[Any, Dict[str, str]]:
    """Compile ONE mutated copy of the whole module from a REAL FILE ON DISK.

    Two mechanics, both forced by the platform rather than chosen:

    * Triton reads a jitted function's body through ``inspect``, so an ``exec``'d
      module raises at first launch. The mutant is written to a real ``.py``.
    * :mod:`folded_offdiag_update_e` imports its siblings RELATIVELY
      (``from . import coverage as _coverage``), so the file is loaded under a
      DOTTED name inside ``meep_gpu.triton_kernels`` and its relative imports
      resolve against the real package. Rewriting those imports to absolute ones
      would have edited source outside the kernel block, which is the one thing
      a mutation harness may not do.

    THE ENTRY POINT IS RENAMED (platform fact (c)): Triton's cache can serve a
    stale binary to a mutant that kept the shipped name, and a leg that measured
    the shipped bytes would report a NEEDLE-MISSED that is really a harness
    failure. The rename is a whole-identifier substitution, so the module's own
    accessor is renamed with it and the file stays self-consistent.
    """
    global _MUTANT_DIR
    if _MUTANT_DIR is None:
        _MUTANT_DIR = tempfile.mkdtemp(prefix="folded_offdiag_mutants_")
    entry = "folded_offdiag_constitutive_step__" + label
    renamed = source.replace("folded_offdiag_constitutive_step", entry)
    if entry not in renamed:  # pragma: no cover - the rename is the mechanism
        raise AssertionError("the kernel rename matched nothing; the mutant "
                             "would have launched the shipped entry point")
    module_name = "meep_gpu.triton_kernels._folded_offdiag_mutant_" + label
    path = os.path.join(_MUTANT_DIR, f"_folded_offdiag_mutant_{label}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(renamed)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = module               # type: ignore[union-attr]
    spec.loader.exec_module(module)                 # type: ignore[union-attr]
    record = {"label": label, "path": path, "entry_point": entry,
              "sha256": _digest(path)}
    MUTANT_FILES.append(record)
    return getattr(module, entry), record


def build_distinctness(mutant_kernel: Any) -> Dict[str, Any]:
    """PLATFORM FACT (c): the cache can serve a STALE binary to a mutant.

    Two independent checks, both recorded: the JIT cache keys must differ, and —
    when PTX is readable — the mutant's PTX must equal NO shipped
    specialization's. A mutant that compiled to the shipped bytes tested
    nothing.
    """
    shipped = fomod.folded_offdiag_constitutive_step_kernel()
    shipped_ptx = ptx_texts(shipped)
    mutant_ptx = ptx_texts(mutant_kernel)
    shipped_key = str(getattr(shipped, "cache_key", "unavailable"))
    mutant_key = str(getattr(mutant_kernel, "cache_key", "unavailable"))
    overlap = [text for text in mutant_ptx if text in shipped_ptx]
    return {
        "shipped_cache_key": shipped_key[:16],
        "mutant_cache_key": mutant_key[:16],
        "cache_keys_differ": shipped_key != mutant_key,
        "ptx_available": bool(mutant_ptx) and bool(shipped_ptx),
        "shipped_specializations": len(shipped_ptx),
        "mutant_specializations": len(mutant_ptx),
        "ptx_differs_from_every_shipped": not overlap,
    }


#: A status in this set FAILS the mutation leg. NEEDLE-MISSED and the harness
#: faults are separate names on purpose: only one of them is the needle's fault.
MUTATION_FAILURE_STATUSES: Tuple[str, ...] = (
    "DISARMED", "NO-LAUNCH", "STALE-BINARY", "NEEDLE-MISSED",
    "UNEXPECTEDLY-CAUGHT", "ERROR")


def classify_mutation(expectation: Optional[bool], differing_words: int,
                      launches: int, distinctness: Dict[str, Any],
                      hits: int) -> str:
    """The classifier. PURE, so the laptop tests drive every branch.

    Order matters: a needle that never matched, never launched, or compiled to
    the shipped bytes has to be reported as a HARNESS failure, not as a
    measurement about the kernel.
    """
    if not hits:
        return "DISARMED"
    if launches == 0:
        return "NO-LAUNCH"
    if not distinctness.get("cache_keys_differ", False):
        return "STALE-BINARY"
    if distinctness.get("ptx_available") and not distinctness.get(
            "ptx_differs_from_every_shipped"):
        return "STALE-BINARY"
    caught = differing_words > 0
    if expectation is True:
        return "CAUGHT" if caught else "NEEDLE-MISSED"
    if expectation is False:
        return "NULL-AS-PREDICTED" if not caught else "UNEXPECTEDLY-CAUGHT"
    return "RECORDED-CAUGHT" if caught else "RECORDED-NOT-CAUGHT"


def _device_arrays(state: Dict[str, np.ndarray],
                   inv_eps: Dict[str, np.ndarray],
                   inv_form: str = "distinct") -> Dict[str, Any]:
    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    if inv_form == "aliased":
        shared = cp.asarray(inv_eps["Ex"])
        for name in E_NAMES:
            arrays["inv_eps_" + name] = shared
    else:
        for name in E_NAMES:
            arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
    return arrays


def launch_folded(state, inv_eps, rows, flat, codes, walls, weights, *,
                  kernel: Any = None, guard: Optional[bool] = False,
                  inv_form: str = "distinct") -> Dict[str, Any]:
    """One launch of THIS family's kernel on device copies of a host state."""
    arrays = _device_arrays(state, inv_eps, inv_form)
    plan = fomod.plan_folded_offdiagonal_constitutive_from_arrays(
        arrays, {k: cp.asarray(v) for k, v in flat.items()},
        {row: {partner: cp.asarray(volume)
               for partner, volume in partners.items()}
         for row, partners in rows.items()},
        codes, walls, weights, kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()
    return arrays


def launch_certified(state, inv_eps, rows, flat, codes, walls, *,
                     kernel: Any = None, guard: Optional[bool] = False,
                     inv_form: str = "distinct") -> Dict[str, Any]:
    """One launch of the CERTIFIED off-diagonal kernel on the same state.

    ``codes`` must be the certified kernel's own two-valued alphabet: it has no
    mirror arm, which is the whole point of the licensing leg.
    """
    arrays = _device_arrays(state, inv_eps, inv_form)
    plan = odmod.plan_offdiagonal_constitutive_from_arrays(
        arrays, {k: cp.asarray(v) for k, v in flat.items()},
        {row: {partner: cp.asarray(volume)
               for partner, volume in partners.items()}
         for row, partners in rows.items()},
        codes, walls, kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()
    return arrays


def compare_state(arrays: Dict[str, Any],
                  reference: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """uint32 words over every array the sub-step writes. Never allclose."""
    return combine({name: bit_compare(arrays[name], reference[name])
                    for name in STATE_NAMES})


def compare_launches(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """uint32 words between two DEVICE results of the same seeded state."""
    return combine({name: bit_compare(left[name], right[name])
                    for name in STATE_NAMES})


def _identity_state(shape, rows_form: str, seed_offset: int):
    rng = np.random.default_rng(SEED + 8000 + seed_offset)
    state, inv_eps = make_host_state(shape, rng)
    return (state, inv_eps, row_form(rows_form, shape, rng),
            synthetic_flat_coefficients(shape, rng))


def identity_device_half(leg: Dict[str, Any], source: str) -> None:
    """The compiled seam, on device: claims (i), (ii) and (iii) in BYTES.

    Every claim here carries its own NEGATIVE CONTROL in the same
    configuration, because all three are equalities and an equality with no
    control is satisfied by a kernel that wrote nothing:

    (i)   the reduction is asserted BESIDE a folded configuration where the two
          kernels must DIFFER, and beside the assertion that the launch moved
          words at all;
    (ii)  the unreachable arm is asserted BESIDE the same grid with a row form
          that makes the fold a live partner, where it must DIFFER;
    (iii) the two mirror codes are asserted BESIDE f10's mutant, which gives
          them different behaviour and must DIFFER — and beside the evidence
          that the two codes really compiled TWO SPECIALIZATIONS, so an equality
          served from one cached binary cannot be read as a null.
    """
    shape = (13, 17, 11)
    claims = leg["claims"]
    compared = 0
    identical = 0
    launches = 0

    def record(name: str, verdict: Dict[str, Any], expect_identical: bool,
               why: str, **extra: Any) -> bool:
        nonlocal compared, identical
        compared += 1
        ok = bool(verdict["bit_identical"]) == expect_identical
        identical += int(bool(verdict["bit_identical"]))
        claims[name] = {"expect_identical": expect_identical,
                        "bit_identical": bool(verdict["bit_identical"]),
                        "differing_words": int(verdict["differing_floats"]),
                        "total_words": int(verdict["total_floats"]),
                        "agrees": ok, "why": why, **extra}
        log(f"[identity] {name}: identical={verdict['bit_identical']} "
            f"(expected {expect_identical}) "
            f"ndiff={verdict['differing_floats']}")
        return ok

    # ---- (i) THE REDUCTION ROW: zero folded axes -> the certified kernel -----
    codes = (C_PERIODIC, C_METALLIC, C_PERIODIC)
    walls = sweep_wall_axes(codes)
    weights = sweep_ghost_weights(codes, (None, None, None))
    state, inv_eps, rows, flat = _identity_state(shape, "varying_full", 1)
    ours = CountingKernel(fomod.folded_offdiag_constitutive_step_kernel())
    theirs = CountingKernel(odmod.offdiag_constitutive_step_kernel())
    folded_result = launch_folded(state, inv_eps, rows, flat, codes, walls,
                                  weights, kernel=ours)
    certified_result = launch_certified(state, inv_eps, rows, flat, codes,
                                        walls, kernel=theirs)
    launches += ours.launches + theirs.launches
    # NON-VACUITY: the launch has to have MOVED something, or "identical" is a
    # statement about two untouched copies of the seed.
    moved = combine({name: bit_compare(folded_result[name], state[name])
                     for name in STATE_NAMES})
    claims["reduction_launch_moved_words"] = {
        "differing_words": int(moved["differing_floats"]),
        "total_words": int(moved["total_floats"]),
        "why": ("the reduction claim is an EQUALITY; without this the two "
                "kernels could agree by both writing nothing")}
    record("reduction_bytes", compare_launches(folded_result, certified_result),
           True,
           "with zero folded axes this kernel must BE the certified kernel, "
           "byte for byte, on the same seeds",
           folded_launches=ours.launches, certified_launches=theirs.launches)

    # ---- (ii) THE UNREACHABLE ARM, and the row form that makes it live ------
    #
    # The walls stay the FOLDED grid's (WM_X = 0) in both drives, exactly as the
    # host leg does it, so the only thing flipped is the ghost arm: giving the
    # metallic drive WM_X = 1 would move bytes for a second reason and the
    # comparison would stop being about the fold.
    fold_codes = (C_MIRROR_PERIODIC, C_PERIODIC, C_PERIODIC)
    flip_codes = (C_METALLIC, C_PERIODIC, C_PERIODIC)
    fold_walls = sweep_wall_axes(fold_codes)
    fold_weights = sweep_ghost_weights(fold_codes, (1, None, None))
    flip_weights = (1.0, 1.0, 1.0)
    for label, rows_form, expect in (("unreachable", "only_row_x", True),
                                     ("live_partner", "varying_full", False)):
        state, inv_eps, rows, flat = _identity_state(shape, rows_form, 2)
        counters = [CountingKernel(fomod.folded_offdiag_constitutive_step_kernel())
                    for _ in range(2)]
        certified = CountingKernel(odmod.offdiag_constitutive_step_kernel())
        as_mirror = launch_folded(state, inv_eps, rows, flat, fold_codes,
                                  fold_walls, fold_weights, kernel=counters[0])
        as_metallic = launch_folded(state, inv_eps, rows, flat, flip_codes,
                                    fold_walls, flip_weights, kernel=counters[1])
        as_certified = launch_certified(state, inv_eps, rows, flat, flip_codes,
                                        fold_walls, kernel=certified)
        launches += sum(c.launches for c in counters) + certified.launches
        record(f"unreachable_arm_{label}_vs_metallic",
               compare_launches(as_mirror, as_metallic), expect,
               "MIRROR vs METALLIC ghost code, same grid, same walls, same "
               "seeds: identical exactly when no live slot takes the folded "
               "axis's component as its partner",
               row_form=rows_form,
               mirror_launches=counters[0].launches,
               metallic_launches=counters[1].launches)
        record(f"unreachable_arm_{label}_vs_certified",
               compare_launches(as_mirror, as_certified), expect,
               "the same statement against the CERTIFIED kernel itself: on an "
               "unreachable fold this family reduces to it",
               row_form=rows_form, certified_launches=certified.launches)

    # ---- (iii) THE PREDICTED NULL: the two mirror codes ---------------------
    state, inv_eps, rows, flat = _identity_state(shape, "varying_full", 3)
    metallic_codes = (C_MIRROR_METALLIC, C_PERIODIC, C_PERIODIC)
    periodic_codes = (C_MIRROR_PERIODIC, C_PERIODIC, C_PERIODIC)
    weights = sweep_ghost_weights(metallic_codes, (1, None, None))
    shipped_kernel = fomod.folded_offdiag_constitutive_step_kernel()
    left_counter = CountingKernel(shipped_kernel)
    as_mirror_metallic = launch_folded(state, inv_eps, rows, flat,
                                       metallic_codes, fold_walls, weights,
                                       kernel=left_counter)
    right_counter = CountingKernel(shipped_kernel)
    as_mirror_periodic = launch_folded(state, inv_eps, rows, flat,
                                       periodic_codes, fold_walls, weights,
                                       kernel=right_counter)
    launches += left_counter.launches + right_counter.launches

    # TWO BUILDS, MEASURED ON A KERNEL WITH AN EMPTY CACHE.
    #
    # THIS CHECK'S FIRST SPELLING WAS WRONG AND THE ARTIFACT CAUGHT IT. It
    # counted NEW specializations on the SHIPPED kernel across the two launches
    # and required one from each. But the sweep leg runs first and had already
    # compiled 63 specializations, one of them this exact MIRROR_PERIODIC
    # signature — so the second launch added 0 new entries and the check read
    # "one binary served both codes" when what actually happened was a cache HIT
    # ON THE RIGHT ENTRY. The claim and the measurement had drifted apart.
    #
    # What is measured instead: a FRESH renamed copy of the SHIPPED source, with
    # an empty cache, launched once per code. Two entries must appear — that is
    # the two-builds claim with the confound removed — and the copy's results
    # must equal the shipped kernel's, so the stand-in is faithful rather than
    # merely available. Whether the two PTX texts are EQUAL is then the null's
    # positive evidence and is recorded either way.
    probe_kernel, probe_file = compile_mutant_module(source, "identity_null_probe")
    probe_counters = [CountingKernel(probe_kernel) for _ in range(2)]
    probe_metallic = launch_folded(state, inv_eps, rows, flat, metallic_codes,
                                   fold_walls, weights, kernel=probe_counters[0])
    probe_periodic = launch_folded(state, inv_eps, rows, flat, periodic_codes,
                                   fold_walls, weights, kernel=probe_counters[1])
    launches += sum(counter.launches for counter in probe_counters)
    probe_keys = specialization_keys(probe_kernel)
    probe_ptx = ptx_texts(probe_kernel)
    faithful = (compare_launches(probe_metallic, as_mirror_metallic)["bit_identical"]
                and compare_launches(probe_periodic,
                                     as_mirror_periodic)["bit_identical"])
    claims["mirror_codes_two_builds"] = {
        "probe_module": probe_file,
        "specializations_on_the_fresh_copy": len(probe_keys),
        "distinct_specialization_keys": len(set(probe_keys)),
        "two_distinct_specializations": len(set(probe_keys)) == 2,
        "fresh_copy_reproduces_the_shipped_kernel": bool(faithful),
        "ptx_texts_captured": len(probe_ptx),
        "the_two_ptx_texts_are_equal": (len(set(probe_ptx)) == 1
                                        if len(probe_ptx) == 2 else None),
        "launches": sum(counter.launches for counter in probe_counters),
        "why": ("the null is an EQUALITY between two compiled surfaces; if the "
                "second code had been served the FIRST code's binary the "
                "equality would be an artifact of the cache, not a property of "
                "the kernel (platform fact (c)). Measured on an EMPTY cache "
                "because the shipped kernel's is warm by this point in the run"),
    }
    record("mirror_codes_bytes",
           compare_launches(as_mirror_metallic, as_mirror_periodic), True,
           "MIRROR_METALLIC and MIRROR_PERIODIC: the predicted null, in bytes. "
           "This sub-step has no ownership mask and calls _shift_up with four "
           "arguments, so the folded-PERIODIC reflect branch cannot fire",
           two_distinct_specializations=claims["mirror_codes_two_builds"][
               "two_distinct_specializations"],
           mirror_metallic_launches=left_counter.launches,
           mirror_periodic_launches=right_counter.launches)

    # THE ARMED CONTROL for (iii): f10's mutant gives the two codes different
    # behaviour, so a comparison that could not see such a difference is caught
    # here rather than trusted.
    mutated, hits = mutate_f10_two_mirror_codes_differ(source)
    control: Dict[str, Any] = {"hits": hits}
    if hits:
        mutant, record_file = compile_mutant_module(mutated, "identity_f10")
        control["mutant_file"] = record_file
        left = CountingKernel(mutant)
        right = CountingKernel(mutant)
        mutant_metallic = launch_folded(state, inv_eps, rows, flat,
                                        metallic_codes, fold_walls, weights,
                                        kernel=left)
        mutant_periodic = launch_folded(state, inv_eps, rows, flat,
                                        periodic_codes, fold_walls, weights,
                                        kernel=right)
        launches += left.launches + right.launches
        verdict = compare_launches(mutant_metallic, mutant_periodic)
        control["launches"] = left.launches + right.launches
        control["distinctness"] = build_distinctness(mutant)
        control["differing_words"] = int(verdict["differing_floats"])
        control["control_is_live"] = bool(
            not verdict["bit_identical"] and control["launches"] == 2
            and control["distinctness"]["cache_keys_differ"])
    else:
        control["error"] = "DISARMED: f10's patch matched nothing"
        control["control_is_live"] = False
    claims["mirror_codes_armed_control"] = control
    log(f"[identity] mirror_codes_armed_control: live="
        f"{control.get('control_is_live')} "
        f"ndiff={control.get('differing_words')}")

    leg["compared"] = compared
    leg["identical"] = identical
    leg["launches"] = launches
    # NAME THE FAILING CONJUNCT. Two of this leg's six comparisons are REQUIRED
    # to differ, so a bare "identical 4 of 6" reads as a failure when it is the
    # expected outcome; the r2 artifact said exactly that while the real fault
    # was the two-builds check.
    leg["certification_detail"] = "; ".join(
        [f"{name} disagreed with its expectation"
         for name in claims
         if isinstance(claims[name], dict) and claims[name].get("agrees") is False]
        + ([] if claims["reduction_launch_moved_words"]["differing_words"]
           else ["the reduction launch moved no word"])
        + ([] if claims["mirror_codes_two_builds"][
            "two_distinct_specializations"]
           else ["the two mirror codes did not produce two specializations"])
        + ([] if claims["mirror_codes_two_builds"][
            "fresh_copy_reproduces_the_shipped_kernel"]
           else ["the fresh copy did not reproduce the shipped kernel"])
        + ([] if control.get("control_is_live")
           else ["f10's armed control for the null was not live"])) or "all claims agree"
    leg["certified"] = bool(
        compared == 6
        and all(claims[name]["agrees"] for name in (
            "reduction_bytes",
            "unreachable_arm_unreachable_vs_metallic",
            "unreachable_arm_unreachable_vs_certified",
            "unreachable_arm_live_partner_vs_metallic",
            "unreachable_arm_live_partner_vs_certified",
            "mirror_codes_bytes"))
        and claims["reduction_launch_moved_words"]["differing_words"] > 0
        and claims["mirror_codes_two_builds"]["two_distinct_specializations"]
        and claims["mirror_codes_two_builds"][
            "fresh_copy_reproduces_the_shipped_kernel"]
        and bool(control.get("control_is_live")))
    log(f"[identity] device half: compared={compared} certified="
        f"{leg['certified']} launches={launches}")


def run_identity(results: Dict[str, Any], out_path: str,
                 device_ok: bool) -> Dict[str, Any]:
    """(i) reduction, (ii) unreachable arm, (iii) the two mirror codes.

    The host half proves (i) at the REFERENCE level against the CERTIFIED
    GATE'S OWN transcription — a different file, written for a different kernel
    — and structurally through the verbatim else-arm check; it proves (iii) by
    construction and records why. The device half is what compares kernel BYTES
    and carries the PTX evidence that two specializations are two builds.
    """
    import gate_triton_offdiag as odgate  # noqa: PLC0415

    leg: Dict[str, Any] = {"claims": {}, "compared": False}
    source = _module_source(fomod)
    leg["shipped_sha256"] = hashlib.sha256(source.encode()).hexdigest()

    # (i) REDUCTION, structural.
    leg["claims"]["reduction_source"] = verbatim_else_arm()

    # (i) REDUCTION, at the reference level: with ZERO folded axes this file's
    # transcription and the CERTIFIED GATE's must agree byte for byte.
    shape = (13, 17, 11)
    codes = (C_PERIODIC, C_METALLIC, C_PERIODIC)
    kinds = tuple(KIND_OF_CODE[code] for code in codes)
    walls = sweep_wall_axes(codes)
    rng = np.random.default_rng(SEED + 6100)
    state, inv_eps = make_host_state(shape, rng)
    rows = row_form("varying_full", shape, rng)
    flat = synthetic_flat_coefficients(shape, rng)
    ours = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(ours, broadcast(flat), rows, inv_eps, kinds,
                       (None, None, None), (None, None, None), walls)
    theirs = {name: state[name].copy() for name in ALL_NAMES}
    odgate.reference_update_e(np, theirs, odgate.broadcast(flat), rows, inv_eps,
                              kinds, walls)
    leg["claims"]["reduction_reference"] = {
        "identical": state_bytes(ours) == state_bytes(theirs),
        "why": ("with zero folded axes this family's transcription must equal "
                "the CERTIFIED family's, in its own file, on the same seeds"),
    }

    # (iii) THE PREDICTED NULL: the two mirror codes.
    #
    # NOT a second reference drive. `reference_update_e` takes KINDS, and
    # `KIND_OF_CODE` maps both mirror codes to MIRROR, so driving it twice with
    # the two codes hands it byte-identical arguments — measured by instrumenting
    # both calls: same kinds, same phases, same reflect rows, same wall axes, and
    # the same row and inverse-epsilon objects. `identical` could not be False
    # for any input, and a tautology has no business as a conjunct of a leg's
    # verdict. What the host CAN measure is the compiled surface and the plan;
    # the bytes and the PTX evidence stay with the device half.
    leg["claims"]["mirror_codes_null"] = mirror_codes_share_one_arm(source)
    leg["claims"]["mirror_codes_null"]["device_half"] = (
        "must compare kernel BYTES and show the two mirror codes are two "
        "SEPARATE BUILDS, so the null cannot be a stale-cache artifact "
        "(platform fact (c))")
    leg["host_half_pass"] = bool(
        leg["claims"]["reduction_source"]["else_arm_is_verbatim"]
        and leg["claims"]["reduction_source"]["return_is_verbatim"]
        and leg["claims"]["reduction_reference"]["identical"]
        and leg["claims"]["mirror_codes_null"]["identical"])
    leg["certified"] = False    # bytes certify nothing until the device half runs
    if not device_ok:
        leg["why_not_compared"] = (
            "claims (i) bytes, (ii) and (iii) need a device; the host half "
            "proved (i) structurally and at the reference level and (iii) by "
            "construction. NO byte-identity claim is made")
    else:
        identity_device_half(leg, source)
    leg["pass"] = bool(leg["host_half_pass"] and leg["compared"]
                       and leg["certified"])
    results["identity"] = leg
    save(results, out_path)
    log(f"[identity] host half pass={leg['host_half_pass']} "
        f"(reduction_source="
        f"{leg['claims']['reduction_source']['else_arm_is_verbatim']}, "
        f"reduction_reference="
        f"{leg['claims']['reduction_reference']['identical']}, "
        f"mirror_null={leg['claims']['mirror_codes_null']['identical']} "
        f"armed={leg['claims']['mirror_codes_null']['check_is_armed']})")
    return leg


# ---------------------------------------------------------------------------
# mutations — the planted defects, their needles, and the analogue that says
# TODAY whether each needle could have caught its defect
# ---------------------------------------------------------------------------

def _mirror_arm(axis: int) -> str:
    """The shipped MIRROR down arm for one axis, verbatim from the kernel."""
    letter = AXES[axis]
    coord = "ijk"[axis]
    return (f"            d{coord} = tl.where(at_{letter}, MIRROR_ROW, "
            f"d{coord})\n"
            f"            w{letter} = tl.where(at_{letter}, gw{letter}, 1.0)\n")


def _replace_every_mirror_arm(source: str,
                              build: Callable[[int], str]) -> Tuple[str, int]:
    hits = 0
    for axis in range(3):
        needle = _mirror_arm(axis)
        count = source.count(needle)
        if count:
            source = source.replace(needle, build(axis))
            hits += count
    return source, hits


def mutate_f1_ghost_is_metallic_zero(source: str) -> Tuple[str, int]:
    """f1: the mirror down ghost served as the certified METALLIC zero — the
    fold's whole delta, deleted."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            dv{AXES[a]} = live & "
                   f"(d{'ijk'[a]} >= 0)\n"))


def mutate_f2_ghost_is_periodic_wrap(source: str) -> Tuple[str, int]:
    """f2: the mirror down ghost served as the certified PERIODIC wrap."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(d{'ijk'[a]} < 0, "
                   f"n{AXES[a]} - 1, d{'ijk'[a]})\n"))


def mutate_f3_ghost_row_is_last(source: str) -> Tuple[str, int]:
    """f3: the ghost reflects about the window TOP (n-1) instead of the
    plane (stored row MIRROR_SOURCE_INDEX)."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(at_{AXES[a]}, "
                   f"n{AXES[a]} - 1, d{'ijk'[a]})\n"
                   f"            w{AXES[a]} = tl.where(at_{AXES[a]}, "
                   f"gw{AXES[a]}, 1.0)\n"))


def mutate_f4_ghost_row_off_by_one(source: str) -> Tuple[str, int]:
    """f4: stored row MIRROR_SOURCE_INDEX + 1 — the off-by-one image row."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(at_{AXES[a]}, "
                   f"MIRROR_ROW + 1, d{'ijk'[a]})\n"
                   f"            w{AXES[a]} = tl.where(at_{AXES[a]}, "
                   f"gw{AXES[a]}, 1.0)\n"))


def mutate_f5_parity_dropped(source: str) -> Tuple[str, int]:
    """f5: the ghost weight forced to +1.0 — the plane's parity dropped."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(at_{AXES[a]}, "
                   f"MIRROR_ROW, d{'ijk'[a]})\n"
                   f"            w{AXES[a]} = tl.where(at_{AXES[a]}, 1.0, "
                   f"1.0)\n"))


_F6_NEEDLES = (
    "w_d * tl.load(g + o_d, mask=v_d, other=0.0)",
    "w_d * tl.load(g + o_ud, mask=v_ud, other=0.0)",
)


def mutate_f6_unary_minus_respell(source: str) -> Tuple[str, int]:
    """f6: the ghost addend respelled with a UNARY MINUS.

    ``w * v`` -> ``-((-w) * v)``: algebraically identical, and identical in
    IEEE too EXCEPT on signed zeros, because Triton's ``semantic.minus`` lowers
    ``-x`` as ``0.0 - x`` and ``0.0 - (+0.0) == +0.0`` while
    ``(-1.0) * (+0.0) == -0.0``. Measured reachable at driver level on other
    families; the signed-zero needle is what decides it here,
    and the outcome is RECORDED either way."""
    hits = 0
    for needle in _F6_NEEDLES:
        count = source.count(needle)
        if count:
            inner = needle.replace("w_d * ", "(0.0 - w_d) * ", 1)
            source = source.replace(needle, f"-({inner})")
            hits += count
    return source, hits


def mutate_f7_weight_on_the_whole_lane(source: str) -> Tuple[str, int]:
    """f7: the ghost weight applied to the WHOLE down lane, not the ghost
    lane — every interior neighbour picks up the parity."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(at_{AXES[a]}, "
                   f"MIRROR_ROW, d{'ijk'[a]})\n"
                   f"            w{AXES[a]} = gw{AXES[a]}\n"))


def mutate_f8_up_ghost_serves_a_row(source: str) -> Tuple[str, int]:
    """f8: the mirror UP ghost served from a stored row instead of the exact
    zero of stepping.py:1828-1830 — the array-path finding's opposite."""
    hits = 0
    for axis in range(3):
        letter, coord = AXES[axis], "ijk"[axis]
        needle = (_mirror_arm(axis)
                  + f"            uv{letter} = live & (u{coord} < n{letter})\n")
        count = source.count(needle)
        if count:
            source = source.replace(needle, (
                _mirror_arm(axis)
                + f"            u{coord} = tl.where(u{coord} == n{letter}, "
                  f"n{letter} - 2, u{coord})\n"
                  f"            uv{letter} = live\n"))
            hits += count
    return source, hits


#: The two ``_masked_row_sum`` call sites that carry the ``at_x`` lane, at both
#: spellings (one line and wrapped). Component 0 (Ex) has NO x lane: the mask
#: loops the axes whose Yee shift is 0 and Ex's is 1 on x — which is why f9 is
#: exactly four sites, not six.
_F9_X_LANE_NEEDLES: Tuple[str, ...] = ("WM_X, WM_Z)", "WM_X, WM_Y)")


def mutate_f9_wall_mask_stops_abstaining_on_the_fold(
        source: str) -> Tuple[str, int]:
    """f9: THE ABSTENTION AT stepping.py:1282 DROPPED — and NOTHING ELSE.

    ``_mask_metallic_wall_coupling`` asks ``is_metallic and not is_mirrored``,
    so a folded axis compiles to ``WM = 0``; this mutant forces the constexpr on
    for the FOLDED AXIS'S LANE ONLY, at the call sites that carry it. On the
    declared needle (fold X) that zeroes ``at_x`` for Ey and Ez — exactly and
    only what the host analogue ``mask_fold=True`` does, since the reference's
    mask loop skips an axis whose Yee shift is nonzero and Ex's is 1 on x.

    THE PREVIOUS SPELLING WAS NOT MINIMAL and its verdict did not mean what it
    said: it replaced ``if WMA: ... if WMB: ...`` with two unconditional
    ``tl.where``, which on the same needle also zeroes ``at_y``/``at_z`` for Ex
    and ``at_z``/``at_y`` for Ey/Ez — planes on two PERIODIC axes the array path
    never masks. A device 'caught' verdict then evidenced "some wall masking
    changed bytes", not "the fold's abstention is byte-visible". That coarser
    defect is still carried, under its own name (f11), with its own claim.
    """
    hits = 0
    for needle in _F9_X_LANE_NEEDLES:
        count = source.count(needle)
        if count:
            source = source.replace(needle, needle.replace("WM_X", "1"))
            hits += count
    return source, hits


_F11_NEEDLE = """        if WMA:
            total = tl.where(at_a, 0.0, total)
        if WMB:
            total = tl.where(at_b, 0.0, total)"""


def mutate_f11_wall_mask_unconditional(source: str) -> Tuple[str, int]:
    """f11: BOTH mask guards dropped — every masked lane zeroed on every axis.

    The coarser sibling of f9, kept because it pins a different fact: that the
    ``WM_*`` constexprs really are OFF on this needle's PERIODIC axes, so
    turning them on moves bytes. Its host analogue is the whole-mask control,
    not ``mask_fold``; catching it says "some wall masking changed bytes" and
    the file says so rather than claiming the fold's abstention."""
    replacement = """        total = tl.where(at_a, 0.0, total)
        total = tl.where(at_b, 0.0, total)"""
    return source.replace(_F11_NEEDLE, replacement), source.count(_F11_NEEDLE)


def mutate_f10_two_mirror_codes_differ(source: str) -> Tuple[str, int]:
    """f10: the predicted null MADE REAL — MIRROR_PERIODIC given a different
    ghost row from MIRROR_METALLIC. ``BC*`` is a constexpr, so the conditional
    resolves at trace time and the two codes become two behaviours."""
    return _replace_every_mirror_arm(
        source,
        lambda a: (f"            d{'ijk'[a]} = tl.where(at_{AXES[a]}, "
                   f"MIRROR_ROW if BC{AXES[a].upper()} == MIRROR_METALLIC "
                   f"else MIRROR_ROW + 1, d{'ijk'[a]})\n"
                   f"            w{AXES[a]} = tl.where(at_{AXES[a]}, "
                   f"gw{AXES[a]}, 1.0)\n"))


_M1_NEEDLE = """        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))"""


def mutate_m1_hoisted_coefficient(source: str) -> Tuple[str, int]:
    """m1 re-spelled at THIS file's indentation: the coefficient hoisted out of
    the pair, erasing the integer-node registration. The certified gate's own
    ``_TERM_NEEDLE`` is cut at four-space indentation and does not match a
    kernel nested one level deeper — a drift the host half reports rather than
    silently counting as zero."""
    replacement = ("        return 0.25 * ((near + far)"
                   " * tl.load(u + o_c, mask=v_c, other=0.0))")
    return source.replace(_M1_NEEDLE, replacement), source.count(_M1_NEEDLE)


def mutate_m4_distribute_quarter(source: str) -> Tuple[str, int]:
    """m4 re-spelled at THIS file's indentation: ``0.25`` distributed before the
    sum — a NULL control.

    An exact power-of-two factor commutes with round-to-nearest AWAY FROM
    UNDERFLOW, so the distributed form is bitwise identical wherever nothing is
    subnormal (measured: launched, PTX-verified-different and byte identical
    on the certified family's cancellation class). It is carried on
    two needles here: the normal one, where it is asserted NOT caught, and the
    SUBNORMAL one, where the caveat lives and the outcome is only RECORDED.

    Re-spelled rather than inherited for the same reason as m1: the certified
    gate's ``_TERM_NEEDLE`` is cut at four-space indentation and matches ZERO
    sites in this kernel (measured: 0), while this file's helper is nested one
    level deeper."""
    replacement = ("        return (0.25 * (near * tl.load(u + o_c, mask=v_c, "
                   "other=0.0))\n"
                   "                + 0.25 * (far * tl.load(u + o_u, mask=v_u, "
                   "other=0.0)))")
    return source.replace(_M1_NEEDLE, replacement), source.count(_M1_NEEDLE)


def reference_variant_update_e(state, coefficients, rows_by_component, inv_eps,
                               kinds, phases, reflect_rows, wall_axes,
                               variant: str) -> None:
    """The array path with ONE structural defect — the host ANALOGUE of a
    planted mutation.

    ``variant='shipped'`` must reproduce :func:`reference_update_e` byte for
    byte; :func:`run_mutations` asserts exactly that before trusting any
    analogue, because two transcriptions that drifted apart would measure the
    drift instead of the defect."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        rows = dict(rows_by_component.get(target, {}) or {})
        if variant == "swapped_slots":
            first = "E" + AXES[(own_axis + 1) % 3]
            second = "E" + AXES[(own_axis + 2) % 3]
            if first in rows and second in rows:
                rows[first], rows[second] = rows[second], rows[first]
        diagonal = volumes[target] * inv_eps[target]
        total = None
        for offset in (1, 2):
            partner_axis = (own_axis + offset) % 3
            coefficient = rows.get("E" + AXES[partner_axis])
            if coefficient is None:
                continue
            values = volumes["E" + AXES[partner_axis]]
            pair = values + reference_shift_down(
                values, partner_axis, kinds[partner_axis],
                phases[partner_axis])
            if variant == "hoisted_coefficient":
                pair_up = reference_shift_up(pair, own_axis, kinds[own_axis],
                                             phases[own_axis],
                                             reflect_rows[own_axis])
                term = 0.25 * ((pair + pair_up) * coefficient)
            else:
                product = pair * coefficient
                if variant == "own_shift_down":
                    shifted = reference_shift_down(product, own_axis,
                                                   kinds[own_axis],
                                                   phases[own_axis])
                else:
                    shifted = reference_shift_up(product, own_axis,
                                                 kinds[own_axis],
                                                 phases[own_axis],
                                                 reflect_rows[own_axis])
                if variant == "distributed_quarter":
                    term = 0.25 * product + 0.25 * shifted
                else:
                    term = 0.25 * (product + shifted)
            total = term if total is None else total + term
        if total is not None:
            iyee = IYEE_SHIFTS[target]
            for axis in range(3):
                if iyee[axis] != 0:
                    continue
                if wall_axes[axis]:
                    total[_face(axis, 0)] = 0
            constitutive = ((total + diagonal) if variant == "commuted_row_sum"
                            else (diagonal + total))
        else:
            constitutive = diagonal
        fw = state["f_w_" + target]
        field = state[target]
        previous = fw.copy()
        fw[...] = constitutive
        if variant == "store_before_prev":
            previous = fw
        field += coefficients["kps_" + AXES[own_axis]] * fw
        field -= coefficients["kms_" + AXES[own_axis]] * previous


#: name -> (patch, host analogue, needle case, expectation, why).
#: expectation True = the device leg MUST catch it; None = record the outcome.
#: The analogue is a kwargs dict for the reference (a ghost/mask control) or a
#: ``variant`` name for :func:`reference_variant_update_e`; None means the
#: defect has no host analogue and the device leg is its only measurement.
MUTATIONS: Dict[str, Dict[str, Any]] = {
    "f1_ghost_is_metallic_zero": dict(
        patch=mutate_f1_ghost_is_metallic_zero, expected_hits=3,
        analogue={"down_ghost": "metallic_zero"}, needle="fold_partner",
        expectation=True, why="the fold's whole delta, deleted"),
    "f2_ghost_is_periodic_wrap": dict(
        patch=mutate_f2_ghost_is_periodic_wrap, expected_hits=3,
        analogue={"down_ghost": "periodic_wrap"}, needle="fold_partner",
        expectation=True, why="the fold served as a wrap"),
    "f3_ghost_row_is_last": dict(
        patch=mutate_f3_ghost_row_is_last, expected_hits=3,
        analogue={"down_ghost": "parity_last_row"}, needle="fold_partner",
        expectation=True, why="reflects about the window top, not the plane"),
    "f4_ghost_row_off_by_one": dict(
        patch=mutate_f4_ghost_row_off_by_one, expected_hits=3,
        analogue={"down_ghost": "parity_row_three"}, needle="fold_partner",
        expectation=True, why="stored row 3 instead of 2"),
    "f5_parity_dropped": dict(
        patch=mutate_f5_parity_dropped, expected_hits=3,
        analogue={"down_ghost": "plus_phase_row2"}, needle="zero_init",
        expectation=True, why="+phase instead of -phase"),
    "f6_unary_minus_respell": dict(
        patch=mutate_f6_unary_minus_respell, expected_hits=2,
        analogue={"down_ghost": "minus_lowering_row2"}, needle="signed_zero",
        expectation=None,
        why=("platform fact (b) re-measured on the class where it can bite; "
             "the signed-zero needle decides it and the outcome is RECORDED "
             "either way")),
    "f7_weight_on_the_whole_lane": dict(
        patch=mutate_f7_weight_on_the_whole_lane, expected_hits=3,
        analogue={"down_ghost": "whole_lane_parity"}, needle="fold_partner",
        expectation=True, why="the parity escapes the ghost lane"),
    "f8_up_ghost_serves_a_row": dict(
        patch=mutate_f8_up_ghost_serves_a_row, expected_hits=3,
        analogue={"up_ghost": "far_row"}, needle="fold_own_axis",
        expectation=True,
        why="the own-axis UP ghost must stay the exact zero of :1781-1783"),
    "f9_wall_mask_stops_abstaining_on_the_fold": dict(
        patch=mutate_f9_wall_mask_stops_abstaining_on_the_fold,
        expected_hits=4,
        analogue={"mask_fold": True}, needle="fold_partner",
        expectation=True,
        why=("the abstention at stepping.py:1253 dropped ON THE FOLDED AXIS'S "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1253->1282
             "LANE and nothing else — the mutant zeroes exactly what the "
             "mask_fold analogue zeroes")),
    "f11_wall_mask_unconditional": dict(
        patch=mutate_f11_wall_mask_unconditional, expected_hits=1,
        analogue={"mask_all": True}, needle="fold_partner",
        expectation=True,
        why=("BOTH mask guards dropped: the coarser sibling of f9, which pins "
             "that the WM_* constexprs really are off on this needle's "
             "PERIODIC axes. Catching it evidences 'some wall masking changed "
             "bytes', not the fold's abstention")),
    "f10_two_mirror_codes_differ": dict(
        patch=mutate_f10_two_mirror_codes_differ, expected_hits=3,
        analogue={"down_ghost": "parity_row_three"}, needle="fold_periodic",
        expectation=True,
        why=("the predicted null made real; the analogue is f4's because the "
             "spelling gives MIRROR_PERIODIC the off-by-one row")),
    "m1_hoisted_coefficient": dict(
        patch=mutate_m1_hoisted_coefficient, expected_hits=1,
        analogue={"variant": "hoisted_coefficient"}, needle="fold_partner",
        expectation=True,
        why="the certified family's m1, re-run on a FOLDED case"),
    "m2_same_direction_double_shift": dict(
        patch=None, expected_hits=7,
        analogue={"variant": "own_shift_down"}, needle="fold_partner",
        expectation=True,
        why="the certified family's m2, re-run on a FOLDED case"),
    # NINE, not six: this kernel spells each partner's term call THREE times —
    # under `if R01:`, under the nested `if R02:`, and again in the `else: if
    # R02:` arm — and the certified patcher counts `source.count(first_call) +
    # source.count(second_call)` per row. Measured against the shipped file; the
    # count is now ENFORCED, so a drift that silently patched fewer sites fails
    # the leg instead of leaving a note.
    "m3_mispaired_coefficients": dict(
        patch=None, expected_hits=9,
        analogue={"variant": "swapped_slots"}, needle="fold_partner",
        expectation=True,
        why="the certified family's m3, re-run on a FOLDED case"),
    "m9_store_order": dict(
        patch=None, expected_hits=3,
        analogue={"variant": "store_before_prev"}, needle="fold_partner",
        expectation=True,
        why="the certified family's m9, re-run on a FOLDED case"),
    "m4_distributed_quarter": dict(
        patch=mutate_m4_distribute_quarter, expected_hits=1,
        analogue={"variant": "distributed_quarter"}, needle="fold_partner",
        expectation=False,
        why=("NULL: an exact power of two commutes with round-to-nearest away "
             "from underflow (measured: launched, PTX-different and "
             "not caught). Declared in this leg's description from the start "
             "and previously carried by no entry at all")),
    "m4_distributed_quarter_subnormal": dict(
        patch=mutate_m4_distribute_quarter, expected_hits=1,
        analogue={"variant": "distributed_quarter"}, needle="subnormal",
        expectation=None,
        why=("the SAME null on the class its own caveat names: 'away from "
             "underflow'. Where the sum is subnormal the distributed form need "
             "not commute, so this one is RECORDED either way rather than "
             "asserted — and the host analogue says today whether the needle "
             "can see it at all")),
    "null_commuted_row_sum": dict(
        patch=None, expected_hits=1,
        analogue={"variant": "commuted_row_sum"}, needle="fold_partner",
        expectation=False,
        why=("NULL: float32 addition is bitwise commutative, so a leg that "
             "reports this caught is comparing something other than bytes")),
}

#: The certified family's own patchers, COMPOSED rather than re-spelled: m2, m3,
#: m9 and the commuted-row-sum null are indentation-independent and apply to
#: this kernel unchanged. m1 is NOT — the certified ``_TERM_NEEDLE`` is cut at
#: four-space indentation and this kernel's helper is nested one level deeper —
#: which is why this file re-spells that one and says so. name -> attribute on
#: ``gate_triton_offdiag``.
INHERITED_PATCHES: Dict[str, str] = {
    "m2_same_direction_double_shift": "mutate_m2_same_direction_double_shift",
    "m3_mispaired_coefficients": "mutate_m3_mispaired_coefficients",
    "m9_store_order": "mutate_m9_store_order",
    "null_commuted_row_sum": "mutate_null_commuted_row_sum",
}

#: The needle each mutation is measured on. Every one is a REAL folded
#: configuration; the fold changes which lanes are live, so a mutation caught
#: unfolded is not thereby caught folded.
MUTATION_NEEDLES: Dict[str, Dict[str, Any]] = {
    "fold_partner": dict(shape=(13, 17, 11), fold="foldX_periodic_even",
                         rows="varying_full", amplitude="normal"),
    "fold_own_axis": dict(shape=(13, 17, 11), fold="foldX_periodic_even",
                          rows="only_row_x", amplitude="normal"),
    "fold_periodic": dict(shape=(13, 17, 11), fold="foldX_periodic_even",
                          rows="varying_full", amplitude="normal"),
    "zero_init": dict(shape=(11, 13, 15), fold="foldX_periodic_even",
                      rows="varying_full", amplitude="zero_init_row2"),
    "signed_zero": dict(shape=(11, 13, 15), fold="foldX_periodic_even",
                        rows="varying_full", amplitude="signed_zero_row2"),
    # The class m4's own caveat names ("away from underflow"). Scaled the way
    # the subnormal leg scales it, on a folded case; the row values land in the
    # subnormal range, where an exact power of two need NOT commute with
    # round-to-nearest. Everything stays finite (platform fact (f)): scaling by
    # 1e-38 can only shrink.
    "subnormal": dict(shape=(13, 17, 11), fold="foldX_periodic_even",
                      rows="varying_full", amplitude="subnormal"),
}


def _needle_state(name: str):
    """Build one needle's arrays, rows and reference inputs."""
    spec = MUTATION_NEEDLES[name]
    shape = spec["shape"]
    codes, phases = next((c, p) for label, c, p in SWEEP_FOLDS
                         if label == spec["fold"])
    kinds = tuple(KIND_OF_CODE[code] for code in codes)
    walls = sweep_wall_axes(codes)
    weights = sweep_ghost_weights(codes, phases)
    rng = np.random.default_rng(SEED + 7100 + len(name))
    state, inv_eps = make_host_state(shape, rng)
    rows = row_form(spec["rows"], shape, rng)
    folded = [axis for axis in range(3) if codes[axis] in fomod.MIRROR_CODES]
    if spec["amplitude"] == "zero_init_row2":
        seed_zero_init_row2(state, folded[0], rng)
    elif spec["amplitude"] == "signed_zero_row2":
        seed_signed_zero_row2(state, folded[0], rng)
    elif spec["amplitude"] == "subnormal":
        for name in ALL_NAMES:
            state[name] = (state[name] * np.float32(1e-38)).astype(np.float32)
    flat = synthetic_flat_coefficients(shape, rng)
    return dict(shape=shape, codes=codes, phases=phases, kinds=kinds,
                walls=walls, weights=weights, state=state, inv_eps=inv_eps,
                rows=rows, flat=flat, coefficients=broadcast(flat),
                reflect=(None, None, None), amplitude=spec["amplitude"])


def shipped_on_needle(needle: Dict[str, Any],
                      base: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """The SHIPPED kernel's own verdict on one needle — the row every mutant is
    read against.

    Without it, "the mutant moved bytes" could be a property of the
    configuration rather than of the defect: on a needle the shipped kernel
    already diverged on, every mutant would read CAUGHT and the leg would
    certify nothing at all.
    """
    counter = CountingKernel(fomod.folded_offdiag_constitutive_step_kernel())
    arrays = launch_folded(needle["state"], needle["inv_eps"], needle["rows"],
                           needle["flat"], needle["codes"], needle["walls"],
                           needle["weights"], kernel=counter)
    verdict = compare_state(arrays, base)
    return {"bit_identical": bool(verdict["bit_identical"]),
            "differing_words": int(verdict["differing_floats"]),
            "total_words": int(verdict["total_floats"]),
            "launches": counter.launches}


def mutation_device_half(record: Dict[str, Any], name: str,
                         entry: Dict[str, Any], mutated: str, hits: int,
                         needle: Dict[str, Any], base: Dict[str, np.ndarray],
                         shipped_rows: Dict[str, Dict[str, Any]]) -> None:
    """Compile the mutant from a real file, LAUNCH it, and classify the result.

    The four fail paths the case discipline names are all statuses here, and
    each is a DIFFERENT diagnosis:

    * ``DISARMED``      — the patch matched nothing (already caught on the host,
      restated so a device row can never be read as a measurement);
    * ``NO-LAUNCH``     — the mutant never ran, so a byte-equality says nothing;
    * ``STALE-BINARY``  — the cache served the shipped build (platform fact (c));
    * ``NEEDLE-MISSED`` — the mutant ran, is a distinct build, and still moved
      no byte: the only one of the four that is about the KERNEL.

    Nulls are the mirror image: they must launch, be distinct, and move NOTHING;
    a null that moved bytes is ``UNEXPECTEDLY-CAUGHT`` and fails.
    """
    needle_name = entry["needle"]
    if needle_name not in shipped_rows:
        shipped_rows[needle_name] = shipped_on_needle(needle, base)
        log(f"[mutations] shipped kernel on needle {needle_name}: identical="
            f"{shipped_rows[needle_name]['bit_identical']}")
    record["shipped_on_this_needle"] = shipped_rows[needle_name]

    try:
        mutant, file_record = compile_mutant_module(mutated, name)
    except Exception as exc:            # noqa: BLE001
        record["status"] = "ERROR"
        record["error"] = f"the mutant would not import: {exc!r}"
        record["compared"] = False
        return
    record["mutant_file"] = file_record
    counter = CountingKernel(mutant)
    try:
        arrays = launch_folded(needle["state"], needle["inv_eps"],
                               needle["rows"], needle["flat"], needle["codes"],
                               needle["walls"], needle["weights"],
                               kernel=counter)
        verdict = compare_state(arrays, base)
    except Exception as exc:            # noqa: BLE001
        record["status"] = "ERROR"
        record["error"] = f"the mutant would not launch: {exc!r}"
        record["compared"] = False
        record["launches"] = counter.launches
        return
    distinctness = build_distinctness(mutant)
    record["distinctness"] = distinctness
    record["launches"] = counter.launches
    record["differing_words"] = int(verdict["differing_floats"])
    record["total_words"] = int(verdict["total_floats"])
    record["verdict"] = verdict
    record["compared"] = True
    record["status"] = classify_mutation(entry["expectation"],
                                         record["differing_words"],
                                         counter.launches, distinctness, hits)


def run_mutations(results: Dict[str, Any], out_path: str,
                  device_ok: bool) -> Dict[str, Any]:
    """Plant each defect in the shipped kernel's SOURCE and measure its needle.

    HOST HALF, this round: every patch must apply the expected number of times
    (a patch that applies zero times is DISARMED BY DRIFT and fails here), the
    patched source must still compile, and the defect's host ANALOGUE must move
    the reference's bytes on its needle — so a needle that could not have caught
    its mutation is found now rather than after a device run.

    DEVICE HALF, later: compile the mutant from a real file, LAUNCH-COUNT it,
    verify the mutant PTX differs from every shipped specialization (platform
    fact (c)), and require the byte compare to FAIL for every armed mutation and
    to PASS for every null.
    """
    import gate_triton_offdiag as odgate  # noqa: PLC0415

    inherited = {name: getattr(odgate, attribute)
                 for name, attribute in INHERITED_PATCHES.items()}
    source = _module_source(fomod)
    leg: Dict[str, Any] = {"mutations": {}, "compared": False,
                           "shipped_sha256": hashlib.sha256(
                               source.encode()).hexdigest()}

    # The analogue machinery is only trustworthy if its 'shipped' spelling
    # reproduces the pinned transcription byte for byte. Assert that FIRST.
    probe_state = _needle_state("fold_partner")
    left = {name: probe_state["state"][name].copy() for name in ALL_NAMES}
    reference_update_e(left, probe_state["coefficients"], probe_state["rows"],
                       probe_state["inv_eps"], probe_state["kinds"],
                       probe_state["phases"], probe_state["reflect"],
                       probe_state["walls"])
    right = {name: probe_state["state"][name].copy() for name in ALL_NAMES}
    reference_variant_update_e(right, probe_state["coefficients"],
                               probe_state["rows"], probe_state["inv_eps"],
                               probe_state["kinds"], probe_state["phases"],
                               probe_state["reflect"], probe_state["walls"],
                               "shipped")
    leg["variant_machinery_matches_reference"] = (state_bytes(left)
                                                  == state_bytes(right))

    #: Per needle, the SHIPPED kernel's own verdict — the preamble every mutant
    #: row is read against. Filled by the device half; empty on the laptop.
    shipped_rows: Dict[str, Dict[str, Any]] = {}

    for name, entry in MUTATIONS.items():
        started = time.time()
        record: Dict[str, Any] = {"why": entry["why"],
                                  "expectation": entry["expectation"],
                                  "needle": entry["needle"], "compared": False}
        patch = entry["patch"] or inherited.get(name)
        if patch is None:
            record["error"] = f"no patcher bound for {name}"
            leg["mutations"][name] = record
            continue
        mutated, hits = patch(source)
        record["hits"] = hits
        record["source_differs"] = mutated != source
        if hits == 0 or not record["source_differs"]:
            record["error"] = ("DISARMED BY DRIFT: the patch matched nothing "
                               "in the shipped kernel source")
            leg["mutations"][name] = record
            log(f"[mutations] {name}: DISARMED (0 hits)")
            save(results, out_path)
            continue
        # THE COUNT IS A VERDICT, NOT A NOTE. Several patchers here increment
        # once per INDEPENDENT needle (m2 has seven, f9 four, f6 two): if some
        # drifted and one still matched, `hits > 0` reads as armed while the
        # mutant carries a PARTIAL defect, and the device leg would certify a
        # partial redirect as if it were the whole one. A mismatch is an error.
        record["expected_hits"] = entry["expected_hits"]
        if entry["expected_hits"] is not None and hits != entry["expected_hits"]:
            record["error"] = (
                f"HIT-COUNT DRIFT: expected {entry['expected_hits']} sites, "
                f"patched {hits} — the mutant is not the declared defect")
            leg["mutations"][name] = record
            log(f"[mutations] {name}: HIT-COUNT DRIFT "
                f"({hits} != {entry['expected_hits']})")
            save(results, out_path)
            continue
        try:
            compile(mutated, f"<mutant {name}>", "exec")
            record["compiles"] = True
        except SyntaxError as exc:      # noqa: BLE001
            record["compiles"] = False
            record["error"] = f"the mutant does not parse: {exc}"
            leg["mutations"][name] = record
            log(f"[mutations] {name}: MUTANT DOES NOT PARSE")
            save(results, out_path)
            continue

        # NEEDLE ADEQUACY, measured on the host.
        needle = _needle_state(entry["needle"])
        base = {n: needle["state"][n].copy() for n in ALL_NAMES}
        reference_update_e(base, needle["coefficients"], needle["rows"],
                           needle["inv_eps"], needle["kinds"],
                           needle["phases"], needle["reflect"],
                           needle["walls"])
        analogue = dict(entry["analogue"] or {})
        variant = analogue.pop("variant", None)
        moved = {n: needle["state"][n].copy() for n in ALL_NAMES}
        if variant is not None:
            reference_variant_update_e(
                moved, needle["coefficients"], needle["rows"],
                needle["inv_eps"], needle["kinds"], needle["phases"],
                needle["reflect"], needle["walls"], variant)
        else:
            reference_update_e(moved, needle["coefficients"], needle["rows"],
                               needle["inv_eps"], needle["kinds"],
                               needle["phases"], needle["reflect"],
                               needle["walls"], **analogue)
        record["analogue"] = variant or analogue
        record["needle_amplitude"] = needle["amplitude"]
        record["needle_moves_the_reference"] = (state_bytes(base)
                                                != state_bytes(moved))
        if needle["amplitude"] == "subnormal":
            record["subnormal_words_in_the_reference"] = int(sum(
                int(np.count_nonzero((np.abs(base[n]) < 2.0 ** -126)
                                     & (base[n] != 0)))
                for n in STATE_NAMES))
            if record["subnormal_words_in_the_reference"] == 0:
                record["error"] = (
                    "VACUOUS: the subnormal needle produced no subnormal word, "
                    "so the class this entry exists to measure is absent")
        if entry["expectation"] is True and not record[
                "needle_moves_the_reference"]:
            record["error"] = (
                "NEEDLE-MISSED on the host: the defect's analogue left every "
                "byte equal, so this needle could not catch it on a device "
                "either")
        record["seconds"] = round(time.time() - started, 3)
        if not device_ok:
            record["why_not_compared"] = (
                "the device half compiles the mutant from a real file, counts "
                "launches and checks its build identity; no device in this "
                "round")
        elif "error" not in record:
            mutation_device_half(record, name, entry, mutated, hits, needle,
                                 base, shipped_rows)
        record["seconds"] = round(time.time() - started, 3)
        leg["mutations"][name] = record
        log(f"[mutations] {name}: hits={hits} needle_moves="
            f"{record['needle_moves_the_reference']} "
            f"expectation={entry['expectation']} "
            f"status={record.get('status', 'host-half-only')}")
        save(results, out_path)

    leg["shipped_kernel_on_every_needle"] = shipped_rows
    leg["compared"] = sum(1 for record in leg["mutations"].values()
                          if record.get("compared"))
    leg["launches"] = sum(int(record.get("launches") or 0)
                          for record in leg["mutations"].values())
    if device_ok:
        # THE PER-NEEDLE PREAMBLE IS PART OF THE VERDICT. Every needle the leg
        # measures on must first be one the SHIPPED kernel reproduces byte for
        # byte; on a needle where the shipped kernel already diverged, "the
        # mutant moved bytes" would be a statement about the configuration
        # rather than about the defect.
        leg["shipped_identical_on_every_needle"] = bool(
            shipped_rows and all(row.get("bit_identical")
                                 for row in shipped_rows.values()))
        statuses = [record.get("status") for record in leg["mutations"].values()]
        leg["status_counts"] = {status: statuses.count(status)
                                for status in sorted(set(map(str, statuses)))}
        leg["failed"] = sorted(
            name for name, record in leg["mutations"].items()
            if record.get("status") in MUTATION_FAILURE_STATUSES)
        leg["certified"] = bool(
            leg["shipped_identical_on_every_needle"]
            and leg["compared"] == len(MUTATIONS)
            and not leg["failed"])
    leg["host_half_pass"] = bool(
        leg["variant_machinery_matches_reference"]
        and all("error" not in record
                for record in leg["mutations"].values()))
    leg["armed"] = sum(1 for e in MUTATIONS.values()
                       if e["expectation"] is True)
    leg["recorded_only"] = sum(1 for e in MUTATIONS.values()
                               if e["expectation"] is None)
    leg["nulls"] = sum(1 for e in MUTATIONS.values()
                       if e["expectation"] is False)
    leg.setdefault("certified", False)   # the device half is not written yet
    leg["pass"] = bool(leg["host_half_pass"] and leg["compared"]
                       and leg["certified"])
    results["mutations"] = leg
    save(results, out_path)
    log(f"[mutations] host half pass={leg['host_half_pass']} "
        f"({leg['armed']} armed, {leg['nulls']} null, "
        f"{leg['recorded_only']} recorded-only)")
    return leg


#: The engine leg's configurations. Each is here as the ONLY witness to
#: something the engine route has to carry: both TERMINATIONS, both PLANE
#: PHASES, an ODD stored count, and TWO folded axes at once.
ENGINE_CASES: Tuple[Dict[str, Any], ...] = (
    dict(name="foldX_periodic_even", fold="X", phase=1,
         cell=(2.0, 1.2, 1.2), courant=0.5, rows="varying_full"),
    dict(name="foldX_periodic_even_oddcount", fold="X", phase=1,
         cell=(2.1, 1.2, 1.2), courant=0.35, rows="varying_full",
         seed_offset=3),
    dict(name="foldY_metallic_odd", fold="Y", phase=-1,
         cell=(1.2, 2.0, 1.2), boundaries="metallic",
         courant=0.35, rows="uniform_full", seed_offset=1),
    dict(name="foldXY_periodic_odd_two_planes", fold="XY", phase=-1,
         cell=(2.0, 2.0, 1.2), courant=0.3125, rows="varying_full",
         seed_offset=2),
)

#: Stated budget: sub-step calls per engine case, chained through ``f_w``.
ENGINE_STEPS = 4


def engine_device_half(leg: Dict[str, Any]) -> None:
    """Step real folded CuPy grids through the ENGINE builder, against
    ``stepping.update_E`` itself, comparing uint32 words after EVERY call.

    Two ``Fields``/``PML`` pairs on the same seed: the array path drives one,
    the plan drives the other. The comparison is per CALL, not at the end — a
    divergence that cancels is still a divergence, and the artifact has to name
    the call it first appeared at.
    """
    rows: List[Dict[str, Any]] = []
    compared = 0
    identical = 0
    for spec in ENGINE_CASES:
        started = time.time()
        reference, reference_pml = build_reference_fields(cp, spec)
        candidate, candidate_pml = build_reference_fields(cp, spec)
        seed = SEED + 9200 + spec.get("seed_offset", 0)
        seed_state(reference, cp, "random", np.random.default_rng(seed))
        seed_state(candidate, cp, "random", np.random.default_rng(seed))
        verdict = fomod.folded_offdiag_constitutive_coverage(candidate,
                                                             candidate_pml)
        row: Dict[str, Any] = {
            "name": spec["name"], "steps": ENGINE_STEPS,
            "covered_on_device": bool(verdict.covered),
            "reasons": list(verdict.reasons),
            "shape": [int(n) for n in candidate.grid.shape],
            "stored_cells": [int(candidate.grid.stored_cells(a))
                             for a in range(3)],
            "stored_count_parity": ["even" if int(candidate.grid.stored_cells(a)) % 2 == 0
                                    else "odd" for a in range(3)],
            "courant": float(spec["courant"]),
            "phase": int(spec["phase"]),
            "termination": spec.get("boundaries") or "periodic",
            "folded_axes": [a for a in range(3)
                            if candidate.grid.is_mirrored(a)],
            "per_call": [],
        }
        plan = fomod.plan_folded_offdiagonal_constitutive(candidate,
                                                          candidate_pml)
        row["plan_built"] = plan is not None
        if plan is None:
            row["error"] = ("the ENGINE builder refused a configuration the "
                            "predicate admits on device: "
                            + "; ".join(verdict.reasons))
            rows.append(row)
            log(f"[engine] {spec['name']}: BUILDER REFUSED")
            continue
        row["ghost_weights"] = list(plan.ghost_weights)
        row["boundary_codes"] = list(plan.boundary_codes)
        row["ghost_axes"] = list(plan.ghost_axes)
        row["wall_axes"] = list(plan.wall_axes)
        row["row_mask"] = list(plan.row_mask)
        counter = CountingKernel(fomod.folded_offdiag_constitutive_step_kernel())
        plan._kernel = counter
        before = {name: to_host(getattr(reference, name)).copy()
                  for name in STATE_NAMES}
        for call in range(1, ENGINE_STEPS + 1):
            stepping.update_E(reference, reference_pml)
            plan.run(guard=False)
            cp.cuda.runtime.deviceSynchronize()
            parts = {name: bit_compare(getattr(candidate, name),
                                       getattr(reference, name))
                     for name in STATE_NAMES}
            check = combine(parts)
            compared += 1
            identical += int(bool(check["bit_identical"]))
            point = {"call": call,
                     "bit_identical": bool(check["bit_identical"]),
                     "differing_words": int(check["differing_floats"]),
                     "total_words": int(check["total_floats"]),
                     "differing_arrays": sorted(
                         name for name, part in parts.items()
                         if not part["bit_identical"])}
            row["per_call"].append(point)
            log(f"[engine] {spec['name']} call {call}/{ENGINE_STEPS}: "
                f"identical={point['bit_identical']} "
                f"ndiff={point['differing_words']}")
            if not point["bit_identical"]:
                row["first_divergence"] = point
                break
        row["launches"] = counter.launches
        # NON-VACUITY: the ARRAY PATH's own state must have MOVED across the
        # budget, or every call's "identical" is a statement about two untouched
        # seeds. Measured on the reference driver, against its own snapshot.
        row["state_moved_words"] = int(combine({
            name: bit_compare(getattr(reference, name), before[name])
            for name in STATE_NAMES})["differing_floats"])
        row["bit_identical"] = all(point["bit_identical"]
                                   for point in row["per_call"])
        row["seconds"] = round(time.time() - started, 3)
        rows.append(row)
        log(f"[engine] {spec['name']}: identical={row['bit_identical']} "
            f"launches={row['launches']} moved={row['state_moved_words']} "
            f"({row['seconds']} s)")
    leg["device_cases"] = rows
    leg["compared"] = compared
    leg["identical"] = identical
    leg["certified"] = bool(
        rows and compared == len(ENGINE_CASES) * ENGINE_STEPS
        and compared == identical
        and all(row.get("plan_built") and row.get("covered_on_device")
                and row.get("launches") == ENGINE_STEPS
                and row.get("state_moved_words", 0) > 0 for row in rows))
    leg["terminations"] = sorted({row["termination"] for row in rows})
    leg["phases"] = sorted({row["phase"] for row in rows})
    leg["fold_counts"] = sorted({len(row["folded_axes"]) for row in rows})


def run_engine(results: Dict[str, Any], out_path: str,
               device_ok: bool) -> Dict[str, Any]:
    """``plan_folded_offdiagonal_constitutive`` from the ENGINE's own objects.

    HOST HALF: on a real folded ``Grid``/``Fields``/``PML`` built exactly as the
    device leg builds it, the predicate must refuse for the ARRAY-MODULE clause
    and NOTHING ELSE — which is the statement that every other engine-side
    condition the leg needs is already admitted, measured rather than assumed.

    DEVICE HALF: the same configurations on CuPy, where that one clause no
    longer refuses, stepped against ``stepping.update_E`` with a uint32 compare
    after every call.
    """
    leg: Dict[str, Any] = {"cases": [], "compared": False,
                           "steps_per_case": ENGINE_STEPS}
    for spec in ENGINE_CASES:
        fields, pml = build_reference_fields(np, spec)
        verdict = fomod.folded_offdiag_constitutive_coverage(fields, pml)
        reasons = list(verdict.reasons)
        only_backend = bool(reasons) and all("not cupy" in reason
                                             for reason in reasons)
        row = {"name": spec["name"], "covered": verdict.covered,
               "reasons": reasons, "only_the_array_module_clause": only_backend,
               "plan_is_none": fomod.plan_folded_offdiagonal_constitutive(
                   fields, pml) is None}
        leg["cases"].append(row)
        log(f"[engine] {spec['name']}: host refusal is the array-module clause "
            f"alone = {only_backend}")
        results["engine"] = leg
        save(results, out_path)
    leg["host_half_pass"] = all(row["only_the_array_module_clause"]
                                and row["plan_is_none"]
                                for row in leg["cases"])
    leg.setdefault("certified", False)
    if not device_ok:
        leg["why_not_compared"] = (
            "the engine leg steps real CuPy folded grids against "
            "stepping.update_E; no device in this round")
    else:
        engine_device_half(leg)
    leg["pass"] = bool(leg["host_half_pass"] and leg["compared"]
                       and leg["certified"])
    results["engine"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# license — the measurement that LICENSES a new kernel
# ---------------------------------------------------------------------------
#
# A new kernel has to earn its existence. The cheap alternative to this family
# was never "write nothing": it was RESTATE THE PREDICATE over the CERTIFIED
# off-diagonal body — drop `coverage._grid_reasons` clause 5's fold refusal and
# let `offdiag_constitutive_step` serve folded grids unchanged. That is a real
# proposal and it costs one line, so the file that rejects it owes a
# measurement, not an argument. This leg is that measurement, and it plays the
# role the stripped-complex control played for the cylindrical family.
#
# The certified kernel's boundary alphabet has TWO values, so "admit the fold"
# forces a choice of arm for the folded axis. BOTH readings are measured, on
# every configuration, because a single reading invites the answer "you handed
# it the wrong arm":
#
#   fold -> METALLIC   the near ghost is an exact 0.0
#   fold -> PERIODIC   the near ghost is the wrap from the far face
#
# and the mirror ghost is neither: a parity-weighted copy of stored row 2.
#
# THE LEG IS TWO-SIDED. Divergence alone would not license anything — a kernel
# that diverged everywhere, including where the fold cannot be seen, would be
# evidence of a broken harness. So the expectation is PREDICTED PER CASE from
# the row slots (`fold_is_a_live_partner_axis`) and the leg fails if measurement
# and prediction disagree anywhere: the certified body must diverge exactly on
# the configurations where a live slot takes the folded axis as its partner, and
# must be BYTE-IDENTICAL on the ones where none does.

LICENSE_STEPS = 1


def license_case_inputs(spec: Dict[str, Any]) -> Dict[str, Any]:
    """One REAL folded configuration, reduced to arrays both kernels can take."""
    fields, pml = build_reference_fields(np, spec)
    grid = fields.grid
    seed_state(fields, np, "random",
               np.random.default_rng(SEED + 9500 + spec.get("seed_offset", 0)))
    codes, reasons = symmod.folded_axis_kinds(grid, pml)
    flat = {f"{stem}_{axis}": np.ascontiguousarray(
        np.asarray(to_host(getattr(pml, f"{stem}_{axis}_h"))).reshape(-1)
        ).astype(np.float32)
        for axis in AXES for stem in ("kps", "kms")}
    return {
        "fields": fields, "pml": pml, "grid": grid,
        "codes": None if codes is None else tuple(int(c) for c in codes),
        "code_reasons": list(reasons or ()),
        "kinds": tuple(stepping._boundary_kinds(grid, pml)),
        "phases": stepping._mirror_phases(grid),
        "reflect": stepping._far_reflect_rows(grid),
        "walls": grid_wall_axes(grid),
        "flat": flat,
        "coefficients": broadcast(flat),
        "rows": {name: {p: to_host(v) for p, v in
                        fields.chi1inv_offdiagonal_for(name).items()}
                 for name in E_NAMES},
        "inv_eps": {name: to_host(fields.inverse_epsilon_for(name))
                    for name in E_NAMES},
        "state": {name: to_host(getattr(fields, name)).copy()
                  for name in ALL_NAMES},
        "folded_axes": [a for a in range(3) if grid.is_mirrored(a)],
    }


def run_license(results: Dict[str, Any], out_path: str,
                device_ok: bool) -> Dict[str, Any]:
    """Does the CERTIFIED body, predicate restated, already serve a fold?

    HOST HALF (no device): the same question at the REFERENCE level, on the
    transcription the reference leg pins against ``stepping.update_E``. It is
    the leg's enumeration and its prediction; the KERNEL answer is the device
    half's.
    """
    leg: Dict[str, Any] = {
        "cases": [], "compared": False,
        "steps_per_case": LICENSE_STEPS,
        "question": ("would the CERTIFIED off-diagonal kernel, with the fold "
                     "refusal removed from its predicate and nothing else "
                     "changed, serve these folded configurations?"),
        "readings": ["fold read as METALLIC (near ghost 0.0)",
                     "fold read as PERIODIC (near ghost is the wrap)"],
    }
    for spec in REFERENCE_GRIDS:
        started = time.time()
        inputs = license_case_inputs(spec)
        row: Dict[str, Any] = {
            "name": spec["name"], "rows": spec["rows"],
            "folded_axes": inputs["folded_axes"],
            "kinds": list(inputs["kinds"]),
            "codes": list(inputs["codes"] or ()),
            "shape": [int(n) for n in inputs["grid"].shape],
            "walls": list(inputs["walls"]),
        }

        def drive(kinds) -> Dict[str, np.ndarray]:
            scratch = {n: inputs["state"][n].copy() for n in ALL_NAMES}
            reference_update_e(scratch, inputs["coefficients"], inputs["rows"],
                               inputs["inv_eps"], tuple(kinds),
                               inputs["phases"], inputs["reflect"],
                               inputs["walls"])
            return scratch

        folded_reference = drive(inputs["kinds"])
        row["predicted_certified_diverges"] = bool(
            fold_is_a_live_partner_axis(
                inputs["codes"] or (), {k: v for k, v in inputs["rows"].items()
                                        if v}))
        for label, kind in (("metallic", METALLIC), ("periodic", PERIODIC)):
            swapped = list(inputs["kinds"])
            for axis in inputs["folded_axes"]:
                swapped[axis] = kind
            other = drive(swapped)
            row[f"host_certified_{label}_differs"] = bool(
                state_bytes(folded_reference) != state_bytes(other))
        row["host_agrees_with_prediction"] = bool(
            row["host_certified_metallic_differs"]
            == row["predicted_certified_diverges"])
        row["seconds"] = round(time.time() - started, 3)
        leg["cases"].append(row)
        log(f"[license] {spec['name']}: predicted_diverges="
            f"{row['predicted_certified_diverges']} host_metallic="
            f"{row['host_certified_metallic_differs']} host_periodic="
            f"{row['host_certified_periodic_differs']}")
        results["license"] = leg
        save(results, out_path)

    leg["host_configurations"] = len(leg["cases"])
    leg["host_certified_metallic_diverging"] = sum(
        1 for row in leg["cases"] if row["host_certified_metallic_differs"])
    leg["host_certified_periodic_diverging"] = sum(
        1 for row in leg["cases"] if row["host_certified_periodic_differs"])
    leg["host_half_pass"] = bool(
        leg["cases"]
        and all(row["host_agrees_with_prediction"] for row in leg["cases"])
        and 0 < leg["host_certified_metallic_diverging"] < len(leg["cases"]))
    leg.setdefault("certified", False)
    if not device_ok:
        leg["why_not_compared"] = (
            "the licensing claim is about the CERTIFIED KERNEL's bytes, not "
            "about a transcription of it; no device in this round")
    else:
        license_device_half(leg)
    leg["pass"] = bool(leg["host_half_pass"] and leg["compared"]
                       and leg["certified"])
    results["license"] = leg
    save(results, out_path)
    log(f"[license] host half pass={leg['host_half_pass']}: the certified "
        f"transcription diverges on {leg['host_certified_metallic_diverging']}"
        f"/{leg['host_configurations']} configurations (fold as METALLIC), "
        f"{leg['host_certified_periodic_diverging']}/"
        f"{leg['host_configurations']} (fold as PERIODIC)")
    return leg


def license_device_half(leg: Dict[str, Any]) -> None:
    """The same question, of the SHIPPED CERTIFIED KERNEL, in bytes."""
    compared = 0
    new_identical = 0
    for row in leg["cases"]:
        spec = next(s for s in REFERENCE_GRIDS if s["name"] == row["name"])
        inputs = license_case_inputs(spec)
        scratch = {n: inputs["state"][n].copy() for n in ALL_NAMES}
        reference_update_e(scratch, inputs["coefficients"], inputs["rows"],
                           inputs["inv_eps"], inputs["kinds"],
                           inputs["phases"], inputs["reflect"],
                           inputs["walls"])
        rows = {k: v for k, v in inputs["rows"].items() if v}
        weights = fomod.mirror_ghost_weights(inputs["grid"])

        ours = CountingKernel(fomod.folded_offdiag_constitutive_step_kernel())
        new_result = launch_folded(inputs["state"], inputs["inv_eps"], rows,
                                   inputs["flat"], inputs["codes"],
                                   inputs["walls"], weights, kernel=ours)
        new_verdict = compare_state(new_result, scratch)
        compared += 1
        new_identical += int(bool(new_verdict["bit_identical"]))
        row["new_kernel"] = {
            "bit_identical": bool(new_verdict["bit_identical"]),
            "differing_words": int(new_verdict["differing_floats"]),
            "total_words": int(new_verdict["total_floats"]),
            "launches": ours.launches}

        for label, code in (("metallic", C_METALLIC), ("periodic", C_PERIODIC)):
            codes = [code if axis in inputs["folded_axes"] else
                     int(inputs["codes"][axis]) for axis in range(3)]
            counter = CountingKernel(odmod.offdiag_constitutive_step_kernel())
            certified = launch_certified(inputs["state"], inputs["inv_eps"],
                                         rows, inputs["flat"], codes,
                                         inputs["walls"], kernel=counter)
            verdict = compare_state(certified, scratch)
            compared += 1
            row[f"certified_{label}"] = {
                "codes": codes,
                "bit_identical": bool(verdict["bit_identical"]),
                "differing_words": int(verdict["differing_floats"]),
                "total_words": int(verdict["total_floats"]),
                "launches": counter.launches}
        row["device_agrees_with_prediction"] = bool(
            (not row["certified_metallic"]["bit_identical"])
            == row["predicted_certified_diverges"])
        log(f"[license] {row['name']}: new={row['new_kernel']['bit_identical']} "
            f"certified_metallic_ndiff="
            f"{row['certified_metallic']['differing_words']} "
            f"certified_periodic_ndiff="
            f"{row['certified_periodic']['differing_words']}")

    leg["compared"] = compared
    leg["new_kernel_identical"] = new_identical
    certify_license(leg)
    log(f"[license] device half: {leg['verdict']}")


def certify_license(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The licensing leg's VERDICT, from its rows. PURE, so a laptop test can
    drive every outcome without a device.

    THE CLAIM IS TWO-SIDED, and each conjunct closes a different way of being
    wrong:

    * the NEW kernel is byte-identical on every configuration — otherwise this
      leg is measuring a broken kernel, not a licensing question;
    * the certified body diverges under BOTH readings of the fold, so "you
      handed it the wrong arm" is not available as an answer;
    * it AGREES on at least one configuration — a body that diverged everywhere,
      including where the fold cannot be seen, would be evidence of a broken
      harness rather than of an insufficient kernel;
    * measurement matches the per-case PREDICTION from the row slots, so the
      divergence pattern is explained rather than merely observed;
    * every launch is counted.
    """
    cases = leg["cases"]
    leg["configurations"] = len(cases)
    leg["certified_kernel_diverging_metallic"] = sum(
        1 for row in cases if not row["certified_metallic"]["bit_identical"])
    leg["certified_kernel_diverging_periodic"] = sum(
        1 for row in cases if not row["certified_periodic"]["bit_identical"])
    leg["certified_kernel_agrees_where_fold_is_unreachable"] = sum(
        1 for row in cases if row["certified_metallic"]["bit_identical"]
        and not row["predicted_certified_diverges"])
    leg["prediction_disagreements"] = sorted(
        row["name"] for row in cases if not row["device_agrees_with_prediction"])
    leg["total_launches"] = sum(
        row["new_kernel"]["launches"] + row["certified_metallic"]["launches"]
        + row["certified_periodic"]["launches"] for row in cases)
    leg["certified"] = bool(
        cases
        and leg.get("new_kernel_identical") == len(cases)
        and not leg["prediction_disagreements"]
        and leg["certified_kernel_diverging_metallic"] > 0
        and leg["certified_kernel_diverging_periodic"] > 0
        and leg["certified_kernel_agrees_where_fold_is_unreachable"] > 0
        and all(row["new_kernel"]["launches"] == 1
                and row["certified_metallic"]["launches"] == 1
                and row["certified_periodic"]["launches"] == 1
                for row in cases))
    leg["verdict"] = (
        f"the CERTIFIED kernel, unmodified, diverges from the array path on "
        f"{leg['certified_kernel_diverging_metallic']}/{len(cases)} real "
        f"folded configurations with the fold read as METALLIC and "
        f"{leg['certified_kernel_diverging_periodic']}/{len(cases)} with it "
        f"read as PERIODIC; it agrees on "
        f"{leg['certified_kernel_agrees_where_fold_is_unreachable']}, which "
        f"are exactly the configurations where no live row slot takes the "
        f"folded axis as its partner. A restated predicate over the certified "
        f"body is therefore INSUFFICIENT, and this family's kernel is "
        f"byte-identical on all {len(cases)}")
    return leg


# ---------------------------------------------------------------------------
# Device legs — declared, gated, and never silently skipped
# ---------------------------------------------------------------------------

DEVICE_LEGS = ("license", "synthetic", "subnormal", "identity", "mutations",
               "engine")

#: Each device leg's HOST HALF: what runs, and returns, without a device.
DEVICE_LEG_HOST_HALVES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "license": run_license,
    "synthetic": run_synthetic,
    "subnormal": run_subnormal,
    "identity": run_identity,
    "mutations": run_mutations,
    "engine": run_engine,
}

#: WHICH DEVICE HALVES EXIST IN CODE TODAY. `synthetic` and `subnormal` build
#: the plan, launch it and compare uint32 words, so a device host runs them
#: unchanged. The other three need machinery that cannot be exercised at all
#: without a GPU — PTX extraction for the identity null and the stale-cache
#: check, mutant module loading and launch counting for the mutations, real
#: CuPy engine objects for the engine leg — and writing it blind is how a leg
#: arrives green and vacuous. They run their HOST HALF here and declare the
#: gap; a device round writes them against hardware it can measure.
DEVICE_HALF_IMPLEMENTED: Dict[str, bool] = {
    "license": True,
    "synthetic": True,
    "subnormal": True,
    "identity": True,
    "mutations": True,
    "engine": True,
}

#: What an unimplemented device half would still have to do, by name. Empty:
#: every leg's device half is written and exercised against hardware.
DEVICE_HALF_TODO: Dict[str, str] = {}


def device_available() -> Tuple[bool, str]:
    if cp is None:
        return False, "cupy is not importable"
    if not _TRITON_AVAILABLE:
        return False, "triton is not importable"
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        return False, f"no CUDA device: {exc!r}"
    return True, "ok"


def skip_device_leg(results: Dict[str, Any], out_path: str, name: str,
                    why: str) -> Dict[str, Any]:
    leg = {"ran": False, "why": why,
           "note": "no byte-identity claim is made for this leg"}
    results[name] = leg
    save(results, out_path)
    log(f"[{name}] SKIPPED: {why}")
    return leg


def run_device_leg(results: Dict[str, Any], out_path: str, name: str,
                   device_ok: bool, why: str) -> Dict[str, Any]:
    """Run a device leg's HOST HALF, and its device half when there is a device.

    Without a device this is NOT a skip and NOT a certification: the leg's case
    enumeration, its reference and its non-vacuity assertions all execute and
    can FAIL, while ``compared`` stays False, ``pass`` stays False, and the
    caller still returns the cannot-certify-here exit code.
    """
    implemented = DEVICE_HALF_IMPLEMENTED[name]
    leg = DEVICE_LEG_HOST_HALVES[name](results, out_path,
                                       device_ok and implemented)
    leg.setdefault("ran_host_half", True)
    leg["device_half_implemented"] = implemented
    if not implemented:
        todo = DEVICE_HALF_TODO.get(name, "(unstated)")
        leg["device_half_todo"] = todo
        leg["why_not_compared"] = (
            "this leg's DEVICE HALF is not written yet: " + todo)
    elif not leg.get("compared"):
        leg.setdefault("why_not_compared", why)
    else:
        # A leg that DID compare must not carry a "why it did not" line: the
        # r2 artifact stamped every green device leg with "the device half was
        # not requested to certify in this round" beside its comparison counts.
        leg.pop("why_not_compared", None)
    leg["note"] = ("no byte-identity claim is made for this leg until "
                   "`compared` is true")
    results[name] = leg
    save(results, out_path)
    log(f"[{name}] host half pass={leg.get('host_half_pass')} "
        f"compared={leg.get('compared')}")
    return leg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_LEGS = ("reference", "refusals", "identity_host") + DEVICE_LEGS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/triton_folded_offdiag/gate.json")
    parser.add_argument("--legs", default=",".join(ALL_LEGS))
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    for name in legs:
        if name not in ALL_LEGS:
            parser.error(f"unknown leg {name!r}; choose from {ALL_LEGS}")

    out_path = os.path.abspath(args.out)
    results_dir = os.path.dirname(out_path)
    os.makedirs(results_dir, exist_ok=True)

    ok, why = device_available()
    requested_device = [name for name in DEVICE_LEGS if name in legs]
    if cp is not None and requested_device:
        # Strip FIRST, guard second: the guard wraps whatever sits at the seam,
        # so both apply. install_ftz_strip raises loudly on a cache directory
        # that could let one policy's binaries serve the other — before any
        # artifact exists. On a host without CuPy it records a no-op.
        gate.install_ftz_strip()
        from meep_gpu import backends  # noqa: PLC0415

        backends.guard_kernel_compilation(cp)

    results: Dict[str, Any] = {
        "gate": "triton_folded_offdiag",
        "module": "meep_gpu/triton_kernels/folded_offdiag_update_e.py",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "legs_requested": list(legs),
        "enable_fp_fusion_certified": False,
        "subnormal_policy": gate.policy_stamp(
            "cupy" if cp is not None else "numpy"),
        "dispatch": "DISABLED (the engine's fast-path hook returns None)",
        "courant_note": ("dtdx does not enter this sub-step's arithmetic; the "
                         "non-power-of-two-Courant mandate lands in the "
                         "real-layer coefficient tables (0.5 / 0.35 / 0.3125)"),
        "source_control_note": ("this sub-step reads no source array; the "
                                "source-driven-vs-source-free control belongs "
                                "to the composition probe, and non-vacuity "
                                "here rides the coefficient-dropped control"),
    }
    save(results, out_path)
    results["provenance"] = write_provenance(results_dir)

    results["device_available"] = ok
    results["device_note"] = why
    save(results, out_path)

    if "reference" in legs:
        run_reference(results, out_path, np, "numpy")
        if ok:
            run_reference(results, out_path, cp, "cupy")
    if "refusals" in legs:
        run_refusals(results, out_path)
    if "identity_host" in legs:
        run_identity_host(results, out_path)

    # DEVICE LEGS. Without a device each one still runs its HOST HALF —
    # enumeration, reference and non-vacuity — which can fail; only the launch
    # and the uint32 compare wait for hardware.
    device_ran: List[str] = []
    device_host_halves: List[str] = []
    for name in DEVICE_LEGS:
        if name not in legs:
            continue
        leg = run_device_leg(
            results, out_path, name, ok,
            why if not ok else
            "the device half was not requested to certify in this round")
        device_host_halves.append(name)
        if leg.get("compared"):
            device_ran.append(name)

    host_legs = [name for name in ("reference", "refusals", "identity_host")
                 if name in legs]
    host_pass = all(
        (results.get(name, {}).get("pass")
         if name != "reference"
         else all(entry.get("pass") for entry in results.get(name, {}).values()))
        for name in host_legs) if host_legs else False
    device_host_pass = all(results.get(name, {}).get("host_half_pass")
                           for name in device_host_halves) \
        if device_host_halves else True

    # THE BYTE VERDICT. Every leg that COMPARED must also have CERTIFIED, and
    # the two are different questions: `compared` is a COUNT (how many uint32
    # comparisons ran) while `certified` is the OUTCOME (every one of them
    # identical, under the configuration this file certifies). Reading only the
    # count is how a device round in which every compare FAILED still stamped
    # `passed: true`, printed `[gate] passed` and exited 0 — demonstrated on the
    # laptop by stubbing the five host halves to {compared: 156, identical: 0}.
    # A leg that compared and did not certify is a FAILURE, named.
    byte_failures: List[str] = []
    for name in device_host_halves:
        leg = results.get(name, {})
        if not leg.get("compared"):
            continue
        if not leg.get("certified"):
            # For the sweep the DECIDING count is the guarded (fusion-off) one,
            # never the merged total — naming the merged number in the failure
            # would report the axis this file does not certify under.
            detail = (leg.get("certification_detail")
                      or (f"guarded rows {leg.get('guarded_identical')}/"
                          f"{leg.get('guarded_ran')} identical"
                          if "guarded_ran" in leg
                          else f"identical {leg.get('identical')}"))
            byte_failures.append(
                f"{name}: compared {leg.get('compared')} and did NOT certify "
                f"({detail})")
        note = leg.get("fusion_control_note")
        if note:
            byte_failures.append(f"{name}: {note}")
    # A SWEEP ERROR ROW is a case that measured nothing — a failure, never an
    # exclusion from 'ran' (the certified family's rule, gate_triton_offdiag.py:
    # 1702-1707). `host_half_pass` already carries it for the sweep; named here
    # so the status line says which leg.
    for name in ("synthetic",):
        if results.get(name, {}).get("errors"):
            byte_failures.append(
                f"{name}: {results[name]['errors']} error rows "
                f"(VACUOUS/exception cases fail the leg)")

    if cp is not None and device_ran:
        residual = gate.ftz_strip_license_reasons()
        if residual:
            results["subnormal_policy_residual"] = residual
            device_host_pass = False
    # RE-STAMP AFTER THE LEGS. `policy_stamp` reads the strip's live counters,
    # and those only move during the NVRTC compiles inside the device legs; the
    # stamp taken with the initial `results` literal is therefore always
    # nvrtc_calls=0 / ftz_removed=0, and `zero_compiles_explained` (written into
    # the strip record by `ftz_strip_license_reasons`) could never reach the
    # artifact. The LICENSING CHECK above is separate and was always correctly
    # placed; this is artifact fidelity.
    results["subnormal_policy_at_start"] = results["subnormal_policy"]
    results["subnormal_policy"] = gate.policy_stamp(
        "cupy" if cp is not None else "numpy")

    everything_host_pass = (bool(host_pass or not host_legs)
                            and device_host_pass and not byte_failures)
    # A leg that was REQUESTED and did not COMPARE must never leave a
    # certification-shaped headline — including when a SIBLING leg did compare.
    uncompared = [name for name in requested_device if name not in device_ran]
    results["device_legs_ran"] = device_ran
    results["device_legs_requested_but_not_compared"] = uncompared
    results["device_leg_host_halves_ran"] = device_host_halves
    results["host_legs_pass"] = bool(host_pass)
    results["device_leg_host_halves_pass"] = bool(device_host_pass)
    results["byte_failures"] = byte_failures
    results["passed"] = bool(everything_host_pass) and not uncompared
    if byte_failures:
        results["status"] = "FAILED: " + "; ".join(byte_failures)
    elif not everything_host_pass:
        results["status"] = "FAILED"
    elif uncompared:
        results["status"] = (
            f"host legs and every device leg's HOST HALF pass; these legs "
            f"compared NO bytes and so certify nothing: "
            f"{', '.join(uncompared)}")
    else:
        results["status"] = "passed"
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, out_path)
    log(f"[gate] {results['status']}; artifact {out_path}")
    if not everything_host_pass:
        return 1
    if uncompared:
        return 75  # cannot certify here
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
