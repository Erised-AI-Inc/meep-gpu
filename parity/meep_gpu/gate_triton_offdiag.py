"""Bit-identity gate for the tensor / off-diagonal epsilon update_E Triton kernel.

The module under test is ``meep_gpu/triton_kernels/offdiag_update_e.py`` — NOT
wired into production dispatch (the engine's fast-path hook keeps returning
None; nothing here changes that). This gate is the arbiter of every grouping
choice stepping.py does not force by construction: the coefficient multiply
sitting BETWEEN the two shifts (the family's one genuinely new arithmetic
element), the opposite shift directions, the 0.25-scales-the-sum association,
the metallic wall-coupling mask (both dropped and over-applied), the metallic
near ghost, the compiled zero-row seam, and the ``fw``/``f`` store order.

Legs, in order (``--legs`` selects a subset):

* ``reference``  — the in-file transcription pinned against ``stepping.update_E``
  itself, on real ``Grid``/``Fields``/``PML`` objects with installed tensors,
  byte for byte, over repeated sub-step calls (state chained through ``f_w``).
  Runs on NumPy (the laptop merge bar) and again on CuPy when a device is
  present; emits a per-call state sha256 chain.
* ``refusals``   — every predicted refusal returns None from the engine builder
  with its named reason; 0 admissions. Includes the TWO distinct beta facts
  (predicate refusal AND stepping's own ValueError for real+offdiag+beta,
  stepping.py:800-810) and the STALE-DOCSTRING GUARD: the predicate refuses a
  folded run that stepping demonstrably steps (the install does NOT refuse
  folded rows, whatever stepping.py:1219-1220 / fields.py:1237-1241 claim).
* ``synthetic``  — the sweep. Shapes (16,16,16) / (13,17,11) non-power-of-two /
  (1,160,160) 2-D reduced (invariant-axis pair = g+g) / (11,13,15) the
  test_tensor_epsilon._CASE oracle shape, x boundary configs (all-periodic,
  all-metallic, mixed with the metallic axis in BOTH roles — a coupled
  component's transverse axis and an own axis), x row forms (the uniform full
  oracle tensor — negative entries included, its inverse's off-diagonals are
  negative — / MANDATORY spatially varying volumes / single-row single-partner
  arms / signed varying), x inverse-epsilon forms (three distinct volumes AND
  three aliases), x amplitude classes (normal; a CANCELLATION class with
  gs*us ~ -coupling so the row-sum association is byte-visible in f32),
  x guard off/on, plus real-layer coefficient tables cut at Courant 0.35 AND
  0.5. The reference side of every comparison is computed on NUMPY (host IEEE
  keep). NOTE dtdx does not enter this sub-step's arithmetic at all — the
  non-power-of-two-Courant mandate lands in the real-layer coefficient tables
  here and in the composition probe's whole steps (courant_note in the
  artifact). Every case asserts the coupling is nonzero somewhere AND that its
  reference bytes differ from a coefficient-dropped (diagonal-engine) control,
  so no case can pass with the feature dark; the source-driven-vs-source-free
  control lives in the composition probe, whose cases carry real sources.
* ``subnormal``  — the deliberately-seeded cancellation class scaled into
  underflow, REPORTED SEPARATELY under the stamped policy, never merged.
* ``identity``   — the compiled seam: the rows-all-dead build launched directly
  must be byte-identical to the certified ``kernels.constitutive_step`` on the
  same seeds, and a partial-row case's UNCOUPLED components must match the
  certified kernel while the coupled one differs (non-vacuous seam).
* ``mutations``  — defects planted in the shipped kernel's SOURCE (compiled
  from a real file, launch-counted, DISARMED / NEEDLE-MISSED fail paths, and
  the mutant's PTX verified to DIFFER from every shipped specialization — the
  stale-cache platform fact): m1 hoisted coefficient (with the uniform-row
  outcome RECORDED — measured on the laptop: distributivity is not a bitwise
  identity, so the plan's predicted null is refined to a recorded outcome),
  m2 both shifts the same direction, m3 mispaired coefficient slots (the
  cycle-order defect the slot binding makes real), m6 wall mask dropped,
  m7 wall mask over-applied to the component's OWN (iyee==1) axis,
  m9 fw/f store order swapped. THREE NULL controls, each with its recorded
  reason and each still required to launch with PTX differing from every
  shipped specialization (a null served a stale shipped binary would measure
  nothing): the commuted row sum (f32 addition is bitwise commutative);
  m4 0.25 distributed before the sum — the first cut measured it NOT caught on the
  cancellation class, and the derivation agrees: 0.25 is an exact power of
  two, scaling by it commutes with round-to-nearest (the rounding grid scales
  exactly), so ``0.25*(A+B) == 0.25*A + 0.25*B`` bitwise wherever nothing is
  subnormal (laptop: 0/2M mismatches on normal AND cancellation classes;
  565k/2M once 0.25*x underflows — the subnormal leg's territory, not this
  one); and m8 metallic near ghost left as the periodic wrap — the first cut
  measured it NOT caught on the all-metallic sweep with PTX-verified-different
  binary, and the derivation agrees: the wrap differs from the zero ghost only
  on the partner-axis face-0 plane, and a metallic partner axis is always a
  Yee-shift-0 transverse axis of the component, so ``_mask_metallic_wall_
  coupling`` (stepping.py:1256-1283) zeroes exactly that plane before the row
  sum (mirrors, where the mask abstains, are refused by this family's
  predicate) — within the admitted space the near-ghost zeroing is retained by
  transcription fidelity, unobservable to any byte gate. The allclose-
  blindness control rides m1 (the hoist) on UNIFORM rows at LOW amplitude
  (field state scaled by an exact power of two, 2**-16): there the hoist is
  algebraically equal and differs only through distributivity rounding
  (~1 ulp relative, far below rtol=1e-5), while the scaling — which
  commutes with rounding, so the byte mismatches survive — holds any
  cancellation-tail absolute delta below atol=1e-8; it must pass allclose
  while failing bytes. The measured wrong turns are recorded: m4 cannot
  serve (a true null), and m1 on VARYING rows cannot either — a
  first-order registration defect (O(1) relative) that allclose correctly
  catches at any amplitude, measured at unit and 2**-16 amplitude
  alike, because rtol is scale-free. Blindness is a statement about
  rounding-scale defects, and the control now embodies it.
* ``engine``     — ``plan_offdiagonal_constitutive`` from the engine's own
  objects (``Fields.set_epsilon_volumes``-installed tensors) against
  ``stepping.update_E`` on real CuPy grids, repeated calls, several
  boundary/tensor configs including the uniform-tensor oracle geometry.

SUBNORMAL POLICY — certification runs UNDER ``ieee_keep_ftz_stripped``. CuPy
unconditionally appends ``-ftz=true`` to every NVRTC compile
(cupy/cuda/compiler.py:552); this gate installs the demonstrated strip
(``gate_triton_complex.install_ftz_strip``) BEFORE any CuPy compile, refuses
to certify when the strip cannot be confirmed exercised, and stamps every
artifact with the policy plus the in-run strip counters. Never pass
``-ftz=false`` as a user option — NVRTC rejects the duplicate.

Every case prints one flushed line as it lands; the JSON artifact is rewritten
atomically after every case (the progress-reporting rule); the gate writes its own
provenance record (``fingerprints.json`` is a shared ledger this gate does not
write; it is only hashed). Correctness only: no throughput or timing claims.

Usage (the GPU host, one clear device; the cache dir MUST carry the policy token)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_offdiag.py \\
        --out results/triton_offdiag_<date>/gate.json

Laptop (NumPy only; device legs and the strip skip cleanly and say so)::

    python -u gate_triton_offdiag.py --legs reference,refusals \\
        --out /tmp/gate_offdiag_local.json

A run whose ``--legs`` include any device leg that could NOT run (no
CUDA/triton) never stamps ``passed``: the summary records
``device_legs_ran``, the status says the device legs were skipped, and the
exit code is 75 (the composition probe's cannot-certify-here convention).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import os
import re
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

try:
    import cupy as cp
except ImportError:  # Laptop leg: the reference transcription still validates.
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# The byte comparator, the coefficient fabric and the policy machinery (ftz
# strip, stamping, counters, launch counting) come from the shared modules.
# Nothing is re-implemented.
import gate_triton_complex as gate  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import offdiag_update_e as odmod  # noqa: E402

SEED = probe.SEED
save = gate.save
log = gate.log
bit_compare = probe.bit_compare
combine = probe.combine
_face = probe._face
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
CODE_OF = {PERIODIC: 0, METALLIC: 1}

E_NAMES = ("Ex", "Ey", "Ez")
D_NAMES = ("Dx", "Dy", "Dz")
FW_NAMES = ("f_w_Ex", "f_w_Ey", "f_w_Ez")
STATE_NAMES = E_NAMES + FW_NAMES  # what the sub-step writes
ALL_NAMES = E_NAMES + FW_NAMES + D_NAMES

#: test_tensor_epsilon._EPS_TENSOR — the uniform tensor every oracle run
#: shares (symmetric, positive definite; its INVERSE's off-diagonals are
#: NEGATIVE, so the uniform form already carries negative coefficients).
EPS_TENSOR = np.array([
    [2.0, 0.35, 0.20],
    [0.35, 2.5, 0.15],
    [0.20, 0.15, 3.0],
], dtype=np.float64)
INV_TENSOR = np.linalg.inv(EPS_TENSOR)


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
        "meep_gpu/sources.py": os.path.join(package_dir, "sources.py"),
        "meep_gpu/stepping.py": os.path.join(package_dir, "stepping.py"),
        "triton_kernels/__init__.py": os.path.join(kernel_dir, "__init__.py"),
        "triton_kernels/coverage.py": os.path.join(kernel_dir, "coverage.py"),
        "triton_kernels/launch.py": os.path.join(kernel_dir, "launch.py"),
        "triton_kernels/kernels.py": os.path.join(kernel_dir, "kernels.py"),
        "triton_kernels/offdiag_update_e.py": os.path.join(
            kernel_dir, "offdiag_update_e.py"),
        "triton_kernels/fingerprints.json": os.path.join(
            kernel_dir, "fingerprints.json"),
        "meep_gpu/test_triton_offdiag_update_e.py": os.path.join(
            package_dir, "test_triton_offdiag_update_e.py"),
        "meep_gpu/test_tensor_epsilon.py": os.path.join(
            package_dir, "test_tensor_epsilon.py"),
        "parity/gate_triton_offdiag.py": os.path.abspath(__file__),
        "parity/gate_triton_complex.py": os.path.join(
            _HERE, "gate_triton_complex.py"),
        "parity/probe_fused_kernel_bit_identity.py": os.path.join(
            _HERE, "probe_fused_kernel_bit_identity.py"),
    }
    composition = os.path.join(_HERE, "probe_triton_offdiag_composition.py")
    if os.path.exists(composition):
        tracked["parity/probe_triton_offdiag_composition.py"] = composition
    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": None if cp is None else cp.__version__,
        "triton": triton.__version__ if _TRITON_AVAILABLE else None,
        "seed": SEED,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device": (probe.device_info() if cp is not None else None),
        "sha256": {name: _digest(path) for name, path in tracked.items()},
    }
    path = os.path.join(results_dir, "provenance.json")
    save(record, path)
    return path


# ---------------------------------------------------------------------------
# The in-file reference — the array path transcribed, computed on NumPy
# ---------------------------------------------------------------------------
#
# EVERY sweep comparison is kernel-bytes against THIS transcription evaluated
# on the host (IEEE subnormal-keep, the ship configuration's reference side);
# the ``reference`` leg is what pins the transcription against stepping.py
# itself on real objects, BEFORE any kernel is measured against it.

def reference_shift_down(xp: Any, field: Any, axis: int, boundary: str) -> Any:
    """f[i-1]: stepping._shift_down (:1787), plain PERIODIC/METALLIC branches."""
    rolled = xp.roll(field, 1, axis=axis)
    if boundary == METALLIC:
        rolled[_face(axis, 0)] = 0
    elif boundary != PERIODIC:
        raise ValueError(f"boundary {boundary!r} has no reference here")
    return rolled


def reference_shift_up(xp: Any, field: Any, axis: int, boundary: str) -> Any:
    """f[i+1]: stepping._shift_up (:1723), plain PERIODIC/METALLIC branches."""
    rolled = xp.roll(field, -1, axis=axis)
    if boundary == METALLIC:
        rolled[_face(axis, -1)] = 0
    elif boundary != PERIODIC:
        raise ValueError(f"boundary {boundary!r} has no reference here")
    return rolled


def reference_coupling(xp: Any, volumes: Dict[str, Any], rows: Dict[str, Any],
                       own_axis: int, boundaries: Sequence[str],
                       wall_axes: Sequence[int],
                       component: str) -> Optional[Any]:
    """stepping._offdiagonal_terms (:1206-1224) + _mask_metallic_wall_coupling
    (:1227-1254): pair DOWN the partner axis on the raw partner volume, the
    coefficient multiply BETWEEN the shifts, the PRODUCT up the own axis, 0.25
    scaling the sum applied last, offset 1 then offset 2, then face-0 zeroing
    of the total on every wall axis where the component's Yee shift is 0."""
    axes = "xyz"
    total = None
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        partner = "E" + axes[partner_axis]
        coefficient = rows.get(partner)
        if coefficient is None:
            continue
        values = volumes[partner]
        pair = values + reference_shift_down(xp, values, partner_axis,
                                             boundaries[partner_axis])
        product = pair * coefficient
        term = 0.25 * (product + reference_shift_up(xp, product, own_axis,
                                                    boundaries[own_axis]))
        total = term if total is None else total + term
    if total is not None:
        iyee = IYEE_SHIFTS[component]
        for axis in range(3):
            if iyee[axis] != 0:
                continue
            if wall_axes[axis]:
                total[_face(axis, 0)] = 0
    return total


def reference_update_e(xp: Any, state: Dict[str, Any],
                       coefficients: Dict[str, Any],
                       rows_by_component: Dict[str, Dict[str, Any]],
                       inv_eps: Dict[str, Any],
                       boundaries: Sequence[str],
                       wall_axes: Sequence[int]) -> None:
    """The whole offdiag sub-step, in place on ``state``.

    stepping.update_E (:967-989), ``elif offdiagonal:`` branch (:972-979) with
    no poles: the D volumes ARE the sources (fields.py:1107-1138 aliases
    undriven components), the row sum is diagonal-first, and the tail is
    _apply_constitutive_pml's no-scratch branch (:2083-2088).
    """
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        gs = volumes[target]
        us = inv_eps[target]
        constitutive = gs * us  # MEEP step_update_EDHB: fw[i] = gs*us.
        coupling = reference_coupling(xp, volumes,
                                      rows_by_component.get(target, {}),
                                      own_axis, boundaries, wall_axes, target)
        if coupling is not None:
            constitutive = constitutive + coupling
        kps = coefficients["kps_" + "xyz"[own_axis]]
        kms = coefficients["kms_" + "xyz"[own_axis]]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = constitutive
        field += kps * fw
        field -= kms * fw_previous


# ---------------------------------------------------------------------------
# Case fabric
# ---------------------------------------------------------------------------

SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 16),    # 3-D
    (13, 17, 11),    # non-power-of-two extents
    (1, 160, 160),   # 2-D reduced: the invariant-axis pair is g+g, not zero
    (11, 13, 15),    # test_tensor_epsilon._CASE: cell 1.1x1.3x1.5 at res 10
)
BOUNDARY_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    (PERIODIC, PERIODIC, PERIODIC),
    (METALLIC, METALLIC, METALLIC),
    # Mixed, metallic axis y: transverse (wall-masked) for Ex and Ez, OWN
    # axis (far-face product ghost) for Ey — both roles in one config.
    (PERIODIC, METALLIC, PERIODIC),
    # Mixed, metallic axis x: the same two roles rotated onto Ex.
    (METALLIC, PERIODIC, PERIODIC),
)
#: Real-layer coefficient tables are additionally cut at these Courant numbers
#: (the mandate's home in a sub-step dtdx never enters).
LAYER_COURANTS = (0.35, 0.5)

ROW_FORMS = ("uniform_full", "varying_full", "single_ex_ey", "single_ex_ez",
             "single_ey_ez", "single_ey_ex", "single_ez_ey", "varying_signed")


def config_viable(shape: Sequence[int], boundaries: Sequence[str]) -> Optional[str]:
    for axis in range(3):
        if boundaries[axis] == METALLIC and int(shape[axis]) == 1:
            return (f"axis {axis} is collapsed (n=1) and metallic: Grid "
                    f"resolves an invariant axis periodic, so the pairing is "
                    f"unreachable")
    return None


def row_form(name: str, shape: Tuple[int, int, int],
             rng) -> Dict[str, Dict[str, np.ndarray]]:
    """Row component -> partner -> f32 coefficient volume."""
    def uniform(value):
        return np.full(shape, value, dtype=np.float32)

    if name == "uniform_full":  # the oracle form; entries include negatives
        return {row: {partner: uniform(INV_TENSOR[i, j])
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    if name == "varying_full":  # MANDATORY: the between-shifts association
        return {row: {partner: (uniform(INV_TENSOR[i, j])
                                * rng.uniform(0.5, 1.5, size=shape)
                                .astype(np.float32)).astype(np.float32)
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    if name == "varying_signed":  # entries crossing zero
        return {row: {partner: rng.uniform(-0.08, 0.08, size=shape)
                      .astype(np.float32)
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    if name == "single_ex_ey":  # component 0, first-partner-only arm (R01)
        return {"Ex": {"Ey": uniform(INV_TENSOR[0, 1])}}
    if name == "single_ex_ez":  # component 0, second-partner-only arm (R02)
        return {"Ex": {"Ez": uniform(INV_TENSOR[0, 2])}}
    if name == "single_ey_ez":  # component 1, first-partner-only arm (R11)
        return {"Ey": {"Ez": uniform(INV_TENSOR[1, 2])}}
    if name == "single_ey_ex":  # component 1, second-partner-only arm (R12)
        return {"Ey": {"Ex": uniform(INV_TENSOR[1, 0])}}
    if name == "single_ez_ey":  # component 2, second-partner-only arm (R22)
        return {"Ez": {"Ey": uniform(INV_TENSOR[2, 1])}}
    raise ValueError(name)


def make_host_state(shape: Tuple[int, int, int], rng,
                    inv_form: str = "distinct") -> Tuple[Dict[str, Any],
                                                         Dict[str, Any]]:
    state = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
             for name in ALL_NAMES}
    if inv_form == "distinct":
        inv_eps = {name: rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
                   for name in E_NAMES}
    elif inv_form == "aliased":  # the isotropic install: three references
        shared = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
        inv_eps = {name: shared for name in E_NAMES}
    else:
        raise ValueError(inv_form)
    return state, inv_eps


def synthetic_flat_coefficients(shape: Tuple[int, int, int],
                                rng) -> Dict[str, np.ndarray]:
    """Seeded random kps/kms per axis, never 1.0, so a swapped axis or a
    dropped coefficient cannot reproduce bit-for-bit — and kms is nonzero
    everywhere, so the store-order mutation is visible at every cell."""
    flat = {}
    for axis, name in enumerate("xyz"):
        for label in ("kps", "kms"):
            flat[f"{label}_{name}"] = rng.uniform(
                0.5, 1.0, size=shape[axis]).astype(np.float32)
    return flat


def layer_flat_coefficients(shape: Tuple[int, int, int],
                            courant: float) -> Optional[Dict[str, np.ndarray]]:
    """The half-integer kps/kms of a REAL layer on a real grid at ``courant``."""
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
    return {f"{label}_{axis}": np.ascontiguousarray(
        getattr(pml, f"{label}_{axis}_h").reshape(-1)).astype(np.float32)
        for axis in "xyz" for label in ("kps", "kms")}


def broadcast(flat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Flat vectors -> the broadcast shapes the reference's arithmetic uses."""
    out = {}
    for axis, name in enumerate("xyz"):
        for label in ("kps", "kms"):
            vector = flat[f"{label}_{name}"]
            shape = [1, 1, 1]
            shape[axis] = vector.size
            out[f"{label}_{name}"] = vector.reshape(shape)
    return out


def seed_cancellation(state: Dict[str, Any], rows, inv_eps,
                      boundaries, wall_axes) -> Dict[str, float]:
    """Make gs*us ~ -coupling on component Ex, so the row-sum association is
    byte-visible in f32 (the assert-where-byte-visible lesson). Dx is rebuilt
    from the coupling the OTHER two D volumes produce; the residual row sum is
    ~1e-3 of the coupling. Values stay in the normal range — the deliberately
    subnormal-seeded variant is the subnormal leg's, never this class's."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    total0 = reference_coupling(np, volumes, rows.get("Ex", {}), 0,
                                boundaries, wall_axes, "Ex")
    if total0 is None or not np.any(total0):
        raise AssertionError("cancellation class: component Ex carries no "
                             "coupling — the class would be vacuous")
    us0 = inv_eps["Ex"]
    noise = 1.0 + 1e-3 * np.random.default_rng(SEED + 7).uniform(
        -1.0, 1.0, size=total0.shape).astype(np.float32)
    state["Dx"] = np.ascontiguousarray(
        (-(total0 / us0) * noise).astype(np.float32))
    # Measure the cancellation actually achieved.
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    total0 = reference_coupling(np, volumes, rows.get("Ex", {}), 0,
                                boundaries, wall_axes, "Ex")
    row_sum = state["Dx"] * us0 + total0
    scale = float(np.max(np.abs(total0)))
    residual = float(np.max(np.abs(row_sum)))
    if scale == 0.0 or residual > 0.1 * scale:
        raise AssertionError(
            f"cancellation class failed to cancel: residual {residual:.3e} "
            f"vs coupling scale {scale:.3e}")
    return {"coupling_scale": scale, "row_sum_residual": residual}


def _wall_axes_for(boundaries: Sequence[str]) -> Tuple[int, ...]:
    """The sweep's declaration-to-mask pairing: with no folds and only the two
    covered kinds, ``is_metallic and not is_mirrored`` coincides with the
    declared METALLIC axes (the engine route re-derives it from the grid)."""
    return tuple(1 if b == METALLIC else 0 for b in boundaries)


def one_case(shape, boundaries, form_name, amplitude, guard,
             inv_form: str = "distinct", coefficient_source: str = "synthetic",
             courant: Optional[float] = None, kernel: Any = None,
             seed_offset: int = 0) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "row_form": form_name, "amplitude": amplitude, "guard": guard,
        "inv_form": inv_form, "coefficient_source": coefficient_source,
    }
    skip = config_viable(shape, boundaries)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 13 + seed_offset)
    state, inv_eps = make_host_state(tuple(shape), rng, inv_form)
    rows = row_form(form_name, tuple(shape), rng)
    wall_axes = _wall_axes_for(boundaries)
    if amplitude == "cancellation":
        if "Ex" not in rows:
            case["skipped"] = ("cancellation class needs an Ex row to cancel "
                               "against")
            return case
        case["cancellation"] = seed_cancellation(state, rows, inv_eps,
                                                 boundaries, wall_axes)

    if coefficient_source == "synthetic":
        flat = synthetic_flat_coefficients(tuple(shape), rng)
    else:
        flat = layer_flat_coefficients(tuple(shape), float(courant))
        if flat is None:
            case["skipped"] = (f"no real layer reproduces shape {shape} at "
                               f"courant {courant}")
            return case
        case["courant"] = courant

    # The reference, on the HOST (IEEE keep — the ship configuration's side).
    reference = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(np, reference, broadcast(flat), rows, inv_eps,
                       boundaries, wall_axes)
    # Non-vacuity 1: the coupling is nonzero somewhere.
    couplings = [reference_coupling(
        np, {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]},
        rows.get(name, {}), axis, boundaries, wall_axes, name)
        for axis, name in enumerate(E_NAMES)]
    if not any(c is not None and bool(np.any(c)) for c in couplings):
        case["error"] = "VACUOUS: no component's coupling is nonzero anywhere"
        return case
    # Non-vacuity 2: the reference differs from the coefficient-dropped
    # (diagonal-engine) control, so no case can pass with the feature dark.
    control = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(np, control, broadcast(flat), {}, inv_eps,
                       boundaries, wall_axes)
    if all(reference[name].tobytes() == control[name].tobytes()
           for name in STATE_NAMES):
        case["error"] = ("VACUOUS: the coupling left every byte equal to the "
                         "diagonal engine on this seed")
        return case

    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    if inv_form == "aliased":
        shared = cp.asarray(inv_eps["Ex"])
        for name in E_NAMES:
            arrays["inv_eps_" + name] = shared
    else:
        for name in E_NAMES:
            arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
    device_rows = {row: {partner: cp.asarray(volume)
                         for partner, volume in partners.items()}
                   for row, partners in rows.items()}
    device_flat = {key: cp.asarray(value) for key, value in flat.items()}
    codes = tuple(CODE_OF[b] for b in boundaries)
    plan = odmod.plan_offdiagonal_constitutive_from_arrays(
        arrays, device_flat, device_rows, codes, wall_axes, kernel=kernel)
    case["row_mask"] = list(plan.row_mask)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in STATE_NAMES})
    return case


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ran = [c for c in cases if not c.get("skipped") and not c.get("error")]
    guarded = [c for c in ran if c.get("guard") is False]
    fusion_on = [c for c in ran if c.get("guard") is True]
    return {
        "ran": len(ran),
        "identical": sum(int(c["verdict"]["bit_identical"]) for c in ran),
        "skipped": sum(1 for c in cases if c.get("skipped")),
        "errors": sum(1 for c in cases if c.get("error")),
        "guarded_ran": len(guarded),
        "guarded_identical": sum(int(c["verdict"]["bit_identical"])
                                 for c in guarded),
        "guarded_pass": bool(guarded) and all(c["verdict"]["bit_identical"]
                                              for c in guarded),
        "fusion_on_ran": len(fusion_on),
        "fusion_on_identical": sum(int(c["verdict"]["bit_identical"])
                                   for c in fusion_on),
    }


def run_synthetic(results: Dict[str, Any], out_path: str,
                  kernel: Any = None, label: str = "gate",
                  guards=(False, True), shapes=SHAPES,
                  boundary_configs=BOUNDARY_CONFIGS,
                  forms=("uniform_full", "varying_full")) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    combos = [(shape, boundaries, form_name)
              for shape in shapes for boundaries in boundary_configs
              for form_name in forms]
    extras = []
    # Single-row single-partner forms — every one of the kernel's six one-slot
    # row-mask specializations launches somewhere in this tranche: the five
    # rows here (each on a boundary config that makes its wall role live) plus
    # R21-only (single_ez_ex), which the identity leg, the engine leg and the
    # composition probe's partial case all compile. R12-only (single_ey_ex)
    # and R22-only (single_ez_ey) matter doubly: their only executing code is
    # the kernel's textually duplicated else-arm blocks for components 1 and 2
    # (offdiag_update_e.py), which NO other leg's mask compiles — without
    # these two rows a defect there would certify green.
    extras.append(((13, 17, 11), (METALLIC, PERIODIC, PERIODIC),
                   "single_ex_ey", "normal", "distinct", "synthetic", None))
    extras.append(((13, 17, 11), (PERIODIC, METALLIC, PERIODIC),
                   "single_ex_ez", "normal", "distinct", "synthetic", None))
    extras.append(((16, 16, 16), (METALLIC, PERIODIC, PERIODIC),
                   "single_ey_ez", "normal", "distinct", "synthetic", None))
    extras.append(((13, 17, 11), (METALLIC, PERIODIC, PERIODIC),
                   "single_ey_ex", "normal", "distinct", "synthetic", None))
    extras.append(((16, 16, 16), (PERIODIC, METALLIC, PERIODIC),
                   "single_ez_ey", "normal", "distinct", "synthetic", None))
    # Signed varying coefficients.
    extras.append(((13, 17, 11), (PERIODIC, METALLIC, PERIODIC),
                   "varying_signed", "normal", "distinct", "synthetic", None))
    # The aliased (isotropic) inverse-epsilon install.
    extras.append(((16, 16, 16), (PERIODIC, METALLIC, PERIODIC),
                   "varying_full", "normal", "aliased", "synthetic", None))
    # The cancellation class — the row-sum association's byte-visible home.
    extras.append(((13, 17, 11), (PERIODIC, METALLIC, PERIODIC),
                   "uniform_full", "cancellation", "distinct", "synthetic",
                   None))
    extras.append(((16, 16, 16), (PERIODIC, PERIODIC, PERIODIC),
                   "varying_full", "cancellation", "distinct", "synthetic",
                   None))
    # Real-layer tables at both Courant numbers.
    for shape in shapes:
        for courant in LAYER_COURANTS:
            extras.append((shape, (PERIODIC, PERIODIC, PERIODIC),
                           "varying_full", "normal", "distinct",
                           "layer", courant))
    total = len(guards) * len(combos) + len(extras)
    index = 0
    for guard in guards:
        for shape, boundaries, form_name in combos:
            index += 1
            started = time.time()
            case = one_case(shape, boundaries, form_name, "normal", guard,
                            kernel=kernel)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            _log_case(label, index, total, case)
            summary = summarize(cases)
            summary["cases"] = cases
            results.setdefault("synthetic", {})[label] = summary
            save(results, out_path)
    for shape, boundaries, form_name, amplitude, inv_form, source, courant in extras:
        index += 1
        started = time.time()
        case = one_case(shape, boundaries, form_name, amplitude, False,
                        inv_form=inv_form, coefficient_source=source,
                        courant=courant, kernel=kernel)
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        _log_case(label, index, total, case)
        summary = summarize(cases)
        summary["cases"] = cases
        results.setdefault("synthetic", {})[label] = summary
        save(results, out_path)
    return results["synthetic"][label]


def _log_case(label, index, total, case) -> None:
    if case.get("skipped"):
        log(f"[{label}] {index}/{total} SKIPPED {case['skipped'][:70]}")
    elif case.get("error"):
        log(f"[{label}] {index}/{total} ERROR {case['error'][:80]}")
    else:
        verdict = case["verdict"]
        log(f"[{label}] {index}/{total} guard={case['guard']} "
            f"{'x'.join(str(n) for n in case['shape'])} "
            f"{'/'.join(b[0] for b in case['boundaries'])} {case['row_form']} "
            f"{case['amplitude']} {case['inv_form']}: "
            f"identical={verdict['bit_identical']} "
            f"(differing={verdict['differing_floats']}/"
            f"{verdict['total_floats']}, maxulp={verdict.get('max_ulp', 0)}) "
            f"({case['seconds']} s)")


# ---------------------------------------------------------------------------
# The subnormal (cancellation) class — reported separately, never merged
# ---------------------------------------------------------------------------

def run_subnormal(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Deliberately-seeded underflow: the coupling products go subnormal.

    Under the STRIPPED policy Triton keeps subnormals as NumPy does, so these
    rows are expected identical — but they are reported as their own account
    under the stamped policy, because the unqualified words 'bit-identical'
    must not absorb the class the policy question lives in."""
    rows_out = []
    for shape, boundaries in (((13, 17, 11), (PERIODIC, METALLIC, PERIODIC)),
                              ((16, 16, 16), (PERIODIC, PERIODIC, PERIODIC))):
        rng = np.random.default_rng(SEED + 91)
        state, inv_eps = make_host_state(shape, rng)
        rows = row_form("varying_full", shape, rng)
        wall_axes = _wall_axes_for(boundaries)
        tiny = np.float32(1e-38)
        for name in ALL_NAMES:
            state[name] = (state[name] * tiny).astype(np.float32)
        flat = synthetic_flat_coefficients(shape, rng)
        reference = {name: state[name].copy() for name in ALL_NAMES}
        reference_update_e(np, reference, broadcast(flat), rows, inv_eps,
                           boundaries, wall_axes)
        subnormals = int(sum(
            int(np.count_nonzero((np.abs(reference[name]) < 2.0 ** -126)
                                 & (reference[name] != 0)))
            for name in STATE_NAMES))
        arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
        for name in E_NAMES:
            arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
        plan = odmod.plan_offdiagonal_constitutive_from_arrays(
            arrays, {k: cp.asarray(v) for k, v in flat.items()},
            {row: {partner: cp.asarray(volume)
                   for partner, volume in partners.items()}
             for row, partners in rows.items()},
            tuple(CODE_OF[b] for b in boundaries), wall_axes)
        plan.run(guard=False)
        cp.cuda.runtime.deviceSynchronize()
        verdict = combine({name: bit_compare(arrays[name], reference[name])
                           for name in STATE_NAMES})
        row = {"shape": list(shape), "boundaries": list(boundaries),
               "reference_subnormals": subnormals,
               "identical": verdict["bit_identical"],
               "differing_floats": verdict["differing_floats"]}
        if subnormals == 0:
            row["error"] = ("VACUOUS: the seeding produced no subnormal in "
                            "the reference — the class measured nothing")
        rows_out.append(row)
        log(f"[subnormal] {row}")
    leg = {"rows": rows_out,
           "subnormal_policy": gate.policy_stamp("cupy"),
           "note": "reported separately from the headline by design"}
    results["subnormal"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The identity legs — the compiled zero-row seam, measured
# ---------------------------------------------------------------------------

def run_identity(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

    leg: Dict[str, Any] = {}
    shape = (13, 17, 11)
    rng = np.random.default_rng(SEED + 29)
    state, inv_eps = make_host_state(shape, rng)
    flat = synthetic_flat_coefficients(shape, rng)

    # (1) The rows-all-dead build, launched directly (the Plan refuses to
    # build an all-dead plan — that refusal is itself under test in the laptop
    # file — so the seam is measured at the kernel boundary), against the
    # certified kernels.constitutive_step through its own plan.
    ours = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    ours.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    certified = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    certified.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    device_flat = {k: cp.asarray(v) for k, v in flat.items()}

    kernel = odmod.offdiag_constitutive_step_kernel()
    n_elem = shape[0] * shape[1] * shape[2]
    block = odmod.DEFAULT_BLOCK
    pointers = ([CupyPointer(ours[n]) for n in E_NAMES]
                + [CupyPointer(ours["f_w_" + n]) for n in E_NAMES]
                + [CupyPointer(ours[n]) for n in D_NAMES]
                + [CupyPointer(ours["inv_eps_" + n]) for n in E_NAMES]
                + [CupyPointer(ours[n]) for n in D_NAMES] * 2  # dead row slots
                + [CupyPointer(device_flat[f"{stem}_{axis}"])
                   for axis in "xyz" for stem in ("kps", "kms")])
    from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION  # noqa: PLC0415
    kernel[((n_elem + block - 1) // block,)](
        *pointers,
        shape[0], shape[1], shape[2], n_elem,
        R01=0, R02=0, R11=0, R12=0, R21=0, R22=0,
        BCX=0, BCY=1, BCZ=0, WM_X=0, WM_Y=1, WM_Z=0, BLOCK=block,
        enable_fp_fusion=ENABLE_FP_FUSION)
    plan = launch_module.plan_constitutive_from_arrays("E", certified,
                                                       device_flat)
    plan.run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    verdict = combine({name: bit_compare(ours[name], certified[name])
                       for name in STATE_NAMES})
    leg["all_dead_rows_vs_certified"] = {
        "identical": verdict["bit_identical"],
        "differing_floats": verdict["differing_floats"],
        "expected": "identical: the R=(0,)*6 arm must compile to the "
                    "certified plain body (module docstring point 6)"}
    log(f"[identity] R=all-dead vs certified constitutive_step: "
        f"identical={verdict['bit_identical']}")

    # (2) The partial split: a row on Ez only. The certified kernel runs the
    # same seeds; Ex/Ey must MATCH it (the compiled seam) while Ez must NOT
    # (the coupling is live) — the non-vacuity half of the same measurement.
    rng = np.random.default_rng(SEED + 31)
    state, inv_eps = make_host_state(shape, rng)
    rows = {"Ez": {"Ex": rng.uniform(-0.1, 0.1, size=shape)
                   .astype(np.float32)}}
    flat = synthetic_flat_coefficients(shape, rng)
    device_flat = {k: cp.asarray(v) for k, v in flat.items()}
    partial = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    partial.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    certified = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    certified.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    plan = odmod.plan_offdiagonal_constitutive_from_arrays(
        partial, device_flat,
        {row: {partner: cp.asarray(volume)
               for partner, volume in partners.items()}
         for row, partners in rows.items()},
        (0, 1, 0), (0, 1, 0))
    plan.run(guard=False)
    launch_module.plan_constitutive_from_arrays(
        "E", certified, device_flat).run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    plain_names = ("Ex", "Ey", "f_w_Ex", "f_w_Ey")
    plain_verdict = combine({name: bit_compare(partial[name], certified[name])
                             for name in plain_names})
    ez_verdict = combine({name: bit_compare(partial[name], certified[name])
                          for name in ("Ez", "f_w_Ez")})
    leg["partial_split"] = {
        "uncoupled_components_identical": plain_verdict["bit_identical"],
        "coupled_component_differs": not ez_verdict["bit_identical"],
        "expected": "Ex/Ey identical to the certified kernel, Ez different — "
                    "the byte-visible per-component split"}
    log(f"[identity] partial split: uncoupled identical="
        f"{plain_verdict['bit_identical']} "
        f"Ez differs={not ez_verdict['bit_identical']}")
    leg["pass"] = bool(
        leg["all_dead_rows_vs_certified"]["identical"]
        and leg["partial_split"]["uncoupled_components_identical"]
        and leg["partial_split"]["coupled_component_differs"])
    results["identity"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The reference leg — the transcription against stepping itself
# ---------------------------------------------------------------------------

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"name": "oracle_metallic_uniform", "cell": (1.1, 1.3, 1.5),
     "dimensions": 3, "boundaries": "metallic", "courant": 0.5,
     "rows": "uniform_full", "inv_form": "distinct"},
    {"name": "oracle_pml_uniform", "cell": (1.1, 1.3, 1.5), "dimensions": 3,
     "boundaries": None, "courant": 0.5, "rows": "uniform_full",
     "inv_form": "distinct", "pml_thickness": 4},
    {"name": "mixed_varying", "cell": (2.0, 1.6, 0.8), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "periodic"), "courant": 0.35,
     "rows": "varying_full", "inv_form": "distinct"},
    {"name": "partial_row_anisotropic", "cell": (0.8, 0.8, 0.8),
     "dimensions": 3, "boundaries": "metallic", "courant": 0.35,
     "rows": "single_ez_ex", "inv_form": "distinct"},
    {"name": "reduced_2d", "cell": (1.6, 1.6, 0.0), "dimensions": 2,
     "boundaries": None, "courant": 0.35, "rows": "varying_full",
     "inv_form": "aliased"},
)
REFERENCE_STEPS = 4


def _grid_rows(spec_rows: str, shape, rng) -> Dict[str, Dict[str, np.ndarray]]:
    if spec_rows == "single_ez_ex":
        return {"Ez": {"Ex": np.full(shape, INV_TENSOR[2, 0],
                                     dtype=np.float32)}}
    return row_form(spec_rows, tuple(shape), rng)


def _build_reference_fields(xp, spec: Dict[str, Any]):
    grid = Grid(resolution=10.0, cell_size=spec["cell"],
                dimensions=spec["dimensions"], boundaries=spec["boundaries"],
                courant=spec["courant"], xp=xp)
    fields = Fields(grid=grid)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(SEED + 47)
    if spec["inv_form"] == "aliased":
        inverse = np.full(shape, 0.45, dtype=np.float32)
        epsilon = np.full(shape, 1.0 / 0.45, dtype=np.float32)
        epsilon_map = {name: xp.asarray(epsilon) for name in E_NAMES}
        shared = xp.asarray(inverse)
        inverse_map = {name: shared for name in E_NAMES}
    else:
        epsilon_map, inverse_map = {}, {}
        for i, name in enumerate(E_NAMES):
            inverse_map[name] = xp.asarray(
                np.full(shape, INV_TENSOR[i, i], dtype=np.float32))
            epsilon_map[name] = xp.asarray(
                np.full(shape, 1.0 / INV_TENSOR[i, i], dtype=np.float32))
    rows = _grid_rows(spec["rows"], shape, rng)
    fields.set_epsilon_volumes(
        epsilon_map, inverse_map,
        chi1inv_offdiagonal={row: {partner: xp.asarray(volume)
                                   for partner, volume in partners.items()}
                             for row, partners in rows.items()})
    fields.enable_pml_storage()
    for name in D_NAMES + STATE_NAMES:
        getattr(fields, name)[...] = xp.asarray(rng.uniform(
            -0.4, 0.4, size=shape).astype(np.float32))
    pml_cells = int(spec.get("pml_thickness", 2))
    thickness = tuple((pml_cells, pml_cells) if shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _grid_wall_axes(grid) -> Tuple[int, ...]:
    """The mask's own question (stepping.py:1279-1283), on a real grid."""
    return tuple(int(grid.is_metallic(axis) and not grid.is_mirrored(axis))
                 for axis in range(3))


def run_reference_validation(results: Dict[str, Any], out_path: str, xp,
                             backend_name: str) -> Dict[str, Any]:
    """Pin the in-file transcription against stepping.update_E on real objects."""
    leg: Dict[str, Any] = {"backend": backend_name, "grids": []}
    for spec in REFERENCE_GRIDS:
        started = time.time()
        row: Dict[str, Any] = {"name": spec["name"], "steps": REFERENCE_STEPS}
        fields, pml = _build_reference_fields(xp, spec)
        boundaries = tuple(stepping._boundary_kinds(fields.grid, pml))
        wall_axes = _grid_wall_axes(fields.grid)
        state = {name: getattr(fields, name).copy()
                 for name in D_NAMES + STATE_NAMES}
        coefficients = {
            f"{label}_{axis}": getattr(pml, f"{label}_{axis}_h")
            for axis in "xyz" for label in ("kps", "kms")}
        rows_by_component = {name: fields.chi1inv_offdiagonal_for(name)
                             for name in E_NAMES}
        inv_eps = {name: fields.inverse_epsilon_for(name) for name in E_NAMES}
        chain = hashlib.sha256()
        identical = True
        for step in range(REFERENCE_STEPS):
            stepping.update_E(fields, pml)
            reference_update_e(xp, state, coefficients, rows_by_component,
                               inv_eps, boundaries, wall_axes)
            for name in STATE_NAMES:
                ours = probe.to_host(state[name])
                theirs = probe.to_host(getattr(fields, name))
                chain.update(ours.tobytes())
                if ours.tobytes() != theirs.tobytes():
                    identical = False
                    row.setdefault("first_divergence", {
                        "step": step + 1, "array": name,
                        "max_abs": float(np.max(np.abs(ours - theirs)))})
        row["identical"] = identical
        row["state_sha256_chain"] = chain.hexdigest()
        row["seconds"] = round(time.time() - started, 3)
        leg["grids"].append(row)
        log(f"[reference:{backend_name}] {spec['name']}: identical={identical} "
            f"({row['seconds']} s)")
        results.setdefault("reference", {})[backend_name] = leg
        save(results, out_path)
    leg["pass"] = all(row["identical"] for row in leg["grids"])
    results.setdefault("reference", {})[backend_name] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The refusal enumeration leg
# ---------------------------------------------------------------------------

def run_refusals(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    from types import SimpleNamespace  # noqa: PLC0415

    xp = cp if cp is not None else np
    backend = "cupy" if cp is not None else "numpy"

    def build(**overrides):
        spec = {"cell": (0.8, 0.8, 0.8), "dimensions": 3,
                "boundaries": "metallic", "courant": 0.35}
        symmetry = overrides.pop("symmetry", ())
        k_point = overrides.pop("k_point", (0.0, 0.0, 0.0))
        beta = overrides.pop("beta", 0.0)
        bfast = overrides.pop("bfast", (0.0, 0.0, 0.0))
        complex_storage = overrides.pop("complex_storage", False)
        rows_kind = overrides.pop("rows", "uniform_full")
        pml_thickness = overrides.pop("pml_thickness", 2)
        spec.update(overrides)
        grid = Grid(resolution=10.0, cell_size=spec["cell"],
                    dimensions=spec["dimensions"],
                    boundaries=spec["boundaries"], courant=spec["courant"],
                    symmetry=symmetry, k_point=k_point, beta=beta,
                    bfast_scaled_k=bfast, xp=xp)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        shape = tuple(grid.shape)
        rng = np.random.default_rng(SEED + 59)
        inverse_map = {name: xp.asarray(np.full(
            shape, INV_TENSOR[i, i], dtype=np.float32))
            for i, name in enumerate(E_NAMES)}
        epsilon_map = {name: xp.asarray(np.full(
            shape, 1.0 / INV_TENSOR[i, i], dtype=np.float32))
            for i, name in enumerate(E_NAMES)}
        if rows_kind == "zeros":
            rows = {row: {partner: np.zeros(shape, dtype=np.float32)
                          for partner in E_NAMES if partner != row}
                    for row in E_NAMES}
        elif rows_kind is None:
            rows = None
        else:
            rows = row_form(rows_kind, shape, rng)
        fields.set_epsilon_volumes(
            epsilon_map, inverse_map,
            chi1inv_offdiagonal=None if rows is None else {
                row: {partner: xp.asarray(volume)
                      for partner, volume in partners.items()}
                for row, partners in rows.items()})
        fields.enable_pml_storage()
        thickness = tuple(
            (pml_thickness, pml_thickness) if shape[axis] >= 6 else (0, 0)
            for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness)

    rows_out = []
    admissions = 0

    def check(name: str, needle: str, builder: Callable[[], Tuple[Any, Any]],
              mutate: Optional[Callable[[Any], None]] = None,
              also: Optional[Callable[[Any, Any, Dict[str, Any]], None]] = None
              ) -> None:
        nonlocal admissions
        row: Dict[str, Any] = {"case": name, "expected_reason": needle}
        try:
            fields, pml = builder()
            if mutate is not None:
                mutate(fields)
            plan = odmod.plan_offdiagonal_constitutive(fields, pml)
            reasons = list(odmod.offdiag_constitutive_coverage(
                fields, pml).reasons)
            if backend == "numpy":
                reasons = [r for r in reasons if "array module" not in r]
            row["plan_is_none"] = plan is None
            row["named"] = any(needle in reason for reason in reasons)
            row["reasons"] = reasons[:6]
            if plan is not None:
                admissions += 1
            row["ok"] = row["plan_is_none"] and row["named"]
            if also is not None:
                also(fields, pml, row)
        except ValueError as exc:
            # A constructor that refuses the configuration outright is itself
            # a refusal — recorded verbatim.
            row["constructor_refusal"] = str(exc)[:200]
            row["ok"] = True
        rows_out.append(row)
        log(f"[refusals] {name}: ok={row['ok']}")

    def fold_still_steps(fields, pml, row):
        # THE STALE-DOCSTRING GUARD: the install kept the rows and the array
        # path steps the folded run; only THIS family refuses it.
        row["install_kept_rows"] = bool(fields.has_offdiagonal_epsilon)
        stepping.update_E(fields, pml)
        row["stepping_steps_it"] = True
        row["ok"] = row["ok"] and row["install_kept_rows"]

    def beta_engine_raises(fields, pml, row):
        # The ENGINE's own refusal, distinct from the predicate's: the curl
        # pass raises for real storage + offdiag + beta (stepping.py:800-810).
        try:
            stepping.step_B(fields, pml)
            row["engine_valueerror"] = False
        except ValueError as exc:
            row["engine_valueerror"] = "off-diagonal" in str(exc)
        row["ok"] = row["ok"] and bool(row["engine_valueerror"])

    check("zero_rows", "no off-diagonal chi1inv row",
          lambda: build(rows="zeros"))
    check("chi2_chi3", "chi2/chi3", lambda: build(),
          mutate=lambda fields: fields.set_nonlinear_volumes(
              {}, {c: 0.08 for c in E_NAMES}))
    check("no_pml", "no active PML", lambda: build(pml_thickness=0))
    check("complex_storage", "force_complex_fields",
          lambda: build(complex_storage=True))
    check("k_nonzero", "k_point", lambda: build(boundaries=None,
                                                k_point=(0.3, 0.0, 0.0)))
    check("beta_predicate_and_engine", "beta",
          lambda: build(cell=(0.8, 0.8, 0.0), dimensions=2,
                        boundaries=None, beta=0.25),
          also=beta_engine_raises)
    check("bfast", "BFAST", lambda: build(bfast=(0.5, 0.0, 0.0),
                                          boundaries=None))
    check("fold_predicate_refuses_stepping_steps", "folded",
          lambda: build(symmetry=("X",), boundaries=None, pml_thickness=0),
          also=fold_still_steps)
    check("polarization", "susceptibility is registered", lambda: build(),
          mutate=lambda fields: fields.polarizations.append(SimpleNamespace(
              driven=lambda: ("Ez",), drives=lambda name: name == "Ez")))
    check("cylindrical", "cylindrical", lambda: build(),
          mutate=lambda fields: setattr(fields.grid, "cylindrical", True))
    check("scalar_inverse_epsilon", "scalar, not a volume", lambda: build(),
          mutate=lambda fields: fields._inv_eps_components.update(Ez=0.5))
    check("malformed_row_volume", "float32", lambda: build(),
          mutate=lambda fields: fields._chi1inv_offdiagonal["Ex"].update(
              Ey=xp.full(fields.grid.shape, 0.1, dtype=xp.float64)))

    leg = {"backend": backend, "rows": rows_out, "admissions": admissions,
           "pass": admissions == 0 and all(r["ok"] for r in rows_out)}
    results["refusals"] = leg
    save(results, out_path)
    log(f"[refusals] admissions={admissions} pass={leg['pass']}")
    return leg


# ---------------------------------------------------------------------------
# Mutations — compiled from source, launch-counted, PTX-verified
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def compile_mutated(source: str, kernel_name: str = "offdiag_constitutive_step"):
    """Compile a mutated copy of the shipped kernel from a real file on disk
    (Triton reads source via inspect). The constexpr codes are re-declared in
    the header because a @triton.jit body may not read a plain module global."""
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_offdiag.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_offdiag_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    """Helpers first (they may be the mutation target), then the kernel."""
    functions = (odmod._offdiag_term, odmod._masked_row_sum,
                 odmod.offdiag_constitutive_step)
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn))
                       for f in functions)


def kernel_ptx_set(jit_fn: Any) -> List[str]:
    """Every compiled specialization's PTX — the stale-cache tripwire's data."""
    out: List[str] = []
    try:
        caches = getattr(jit_fn, "cache", None) or {}
        for per_device in caches.values():
            for compiled in per_device.values():
                asm = getattr(compiled, "asm", None)
                if asm and "ptx" in asm:
                    out.append(asm["ptx"])
    except Exception:  # noqa: BLE001 - unavailable PTX is reported, not fatal here
        return out
    return out


_TERM_NEEDLE = """    return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                   + (far * tl.load(u + o_u, mask=v_u, other=0.0)))"""


def mutate_m1_hoisted_coefficient(source: str) -> Tuple[str, int]:
    """m1: u[i] times the plain four-point average — the hoist that erases the
    integer-node registration. Different ALGEBRA on a varying coefficient;
    on a uniform one the two differ only through distributivity rounding
    (measured on the laptop), so the uniform row is recorded, not asserted."""
    replacement = ("    return 0.25 * ((near + far)"
                   " * tl.load(u + o_c, mask=v_c, other=0.0))")
    return source.replace(_TERM_NEEDLE, replacement), source.count(_TERM_NEEDLE)


def mutate_m2_same_direction_double_shift(source: str) -> Tuple[str, int]:
    """m2: the own-axis shift taken DOWN instead of UP — the half-cell
    registration error stepping.py:1160-1164 warns about. The up-neighbour
    machinery is globally redirected one cell down."""
    hits = 0
    for needle, replacement in (
            ("ui, uj, uk = i + 1, j + 1, k + 1", "ui, uj, uk = i - 1, j - 1, k - 1"),
            ("uvx = live & (ui < nx)", "uvx = live & (ui >= 0)"),
            ("uvy = live & (uj < ny)", "uvy = live & (uj >= 0)"),
            ("uvz = live & (uk < nz)", "uvz = live & (uk >= 0)"),
            ("ui = tl.where(ui == nx, 0, ui)", "ui = tl.where(ui < 0, nx - 1, ui)"),
            ("uj = tl.where(uj == ny, 0, uj)", "uj = tl.where(uj < 0, ny - 1, uj)"),
            ("uk = tl.where(uk == nz, 0, uk)", "uk = tl.where(uk < 0, nz - 1, uk)")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


def mutate_m3_mispaired_coefficients(source: str) -> Tuple[str, int]:
    """m3: the coefficient slots exchanged between the two partner terms of
    every component (the offset-1 entry multiplies the offset-2 partner's pair
    and vice versa). In the ARRAY path the cycle order only permutes a
    bitwise-commutative sum; in the KERNEL the slot binding makes it a real
    defect — each row entry multiplies the WRONG partner's pair. Byte-visible
    wherever the two entries differ (the oracle tensor's do). The rewrite
    touches the TERM CALLS only, never the signature — swapping both would be
    an alpha-rename, not a defect."""
    hits = 0
    for g_first, u_first, g_second, u_second in (
            ("g1", "u01", "g2", "u02"),
            ("g2", "u11", "g0", "u12"),
            ("g0", "u21", "g1", "u22")):
        first_call = f"{g_first}, {u_first},"
        second_call = f"{g_second}, {u_second},"
        count = source.count(first_call) + source.count(second_call)
        if count == 0:
            continue
        token = f"{g_first}, __SWAP__,"
        swapped = source.replace(first_call, token)
        swapped = swapped.replace(second_call, f"{g_second}, {u_first},")
        swapped = swapped.replace(token, f"{g_first}, {u_second},")
        if swapped != source:
            hits += count
            source = swapped
    return source, hits


def mutate_m4_distribute_quarter(source: str) -> Tuple[str, int]:
    """m4: 0.25 distributed before the sum — a NULL control, measured
    (launched, PTX-verified-different, bytes identical on the
    cancellation class) and derivable: an exact power-of-two factor commutes
    with round-to-nearest away from underflow, so the distributed form is
    bitwise identical wherever nothing is subnormal. Contrast m1: an
    ARBITRARY coefficient's distribution rounds differently and is caught."""
    replacement = ("    return (0.25 * (near * tl.load(u + o_c, mask=v_c, "
                   "other=0.0)))\\\n"
                   "        + (0.25 * (far * tl.load(u + o_u, mask=v_u, "
                   "other=0.0)))")
    return source.replace(_TERM_NEEDLE, replacement), source.count(_TERM_NEEDLE)


_MASK_NEEDLE = """    if WMA:
        total = tl.where(at_a, 0.0, total)
    if WMB:
        total = tl.where(at_b, 0.0, total)
    return diag + total"""


def mutate_m6_wall_mask_dropped(source: str) -> Tuple[str, int]:
    """m6: the metallic wall-coupling mask dropped — the 2.6e-02-class defect,
    byte-visible at the first call on any metallic-transverse case."""
    return (source.replace(_MASK_NEEDLE, "    return diag + total"),
            source.count(_MASK_NEEDLE))


_M7_NEEDLE = "src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)"
_M7_REPLACEMENT = "src0 = _masked_row_sum(gs0 * us0, total0, at_x, at_z, WM_X, WM_Z)"


def mutate_m7_wall_mask_overapplied(source: str) -> Tuple[str, int]:
    """m7: component 0's mask moved onto its OWN axis (iyee==1): Ex's coupling
    zeroed at the x wall, which the array path deliberately does NOT zero —
    the wall plane the mask protects is the one where the component itself is
    never stepped, and Ex IS stepped on an x wall."""
    return (source.replace(_M7_NEEDLE, _M7_REPLACEMENT),
            source.count(_M7_NEEDLE))


def mutate_m8_metallic_near_ghost_wrapped(source: str) -> Tuple[str, int]:
    """m8: the metallic near (partner-axis down) ghost left as the periodic
    wrap — the down-guard replaced by the wrap, so the coupling reads the far
    plane's live values through the wall instead of zero. A NULL control,
    measured (launched, PTX-verified-different, bytes identical on
    the all-metallic sweep) and derivable: the wrap's entire support is the
    partner-axis face-0 plane, which ``_mask_metallic_wall_coupling``
    (stepping.py:1256-1283) zeroes before the row sum — a metallic partner
    axis is always a Yee-shift-0 transverse axis of the component, and
    mirrors (where the mask abstains) are refused by this family's
    predicate. The near-ghost zeroing is transcription fidelity, not a
    byte-observable choice, within the admitted space."""
    hits = 0
    for needle, replacement in (
            ("dvx = live & (di >= 0)", "di = tl.where(di < 0, nx - 1, di)"),
            ("dvy = live & (dj >= 0)", "dj = tl.where(dj < 0, ny - 1, dj)"),
            ("dvz = live & (dk >= 0)", "dk = tl.where(dk < 0, nz - 1, dk)")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


def mutate_m9_store_order(source: str) -> Tuple[str, int]:
    """m9: prev read AFTER the fw store — the certified constitutive family's
    retained store-order defect, whose observable is prev == src."""
    pattern = re.compile(r"a(\d) = a\1 - km_\1 \* prev\1")
    return (pattern.sub(r"a\1 = a\1 - km_\1 * src\1", source),
            len(pattern.findall(source)))


def mutate_null_commuted_row_sum(source: str) -> Tuple[str, int]:
    """NULL control: the row sum commuted — ``diag + total`` ->
    ``total + diag``. float32 addition is bitwise commutative, so this must be
    caught 0 times — a leg that reports it caught is comparing something other
    than bytes. (This is also why the plan's 'row-sum order flipped' candidate
    is a control here rather than a mutation.)"""
    needle = "    return diag + total"
    return (source.replace(needle, "    return total + diag"),
            source.count(needle))


#: name -> (transform, row form, boundaries, amplitude, expectation).
#: expectation True = must catch; None = record only; False = must NOT catch.
MUTATION_SHAPE = (13, 17, 11)
MUTATION_MIXED = (PERIODIC, METALLIC, PERIODIC)
MUTATION_METALLIC = (METALLIC, METALLIC, METALLIC)
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str,
                                  Tuple[str, str, str], str,
                                  Optional[bool]]] = {
    "m1_hoisted_coefficient": (mutate_m1_hoisted_coefficient,
                               "varying_full", MUTATION_MIXED, "normal", True),
    "m2_same_direction_double_shift": (mutate_m2_same_direction_double_shift,
                                       "varying_full", MUTATION_MIXED,
                                       "normal", True),
    "m3_mispaired_coefficients": (mutate_m3_mispaired_coefficients,
                                  "uniform_full", MUTATION_MIXED, "normal",
                                  True),
    "m4_distribute_quarter": (mutate_m4_distribute_quarter,
                              "uniform_full", MUTATION_MIXED, "cancellation",
                              False),
    "m6_wall_mask_dropped": (mutate_m6_wall_mask_dropped,
                             "uniform_full", MUTATION_METALLIC, "normal",
                             True),
    "m7_wall_mask_overapplied": (mutate_m7_wall_mask_overapplied,
                                 "uniform_full", MUTATION_METALLIC, "normal",
                                 True),
    "m8_metallic_near_ghost_wrapped": (mutate_m8_metallic_near_ghost_wrapped,
                                       "uniform_full", MUTATION_METALLIC,
                                       "normal", False),
    "m9_store_order": (mutate_m9_store_order,
                       "uniform_full", MUTATION_MIXED, "normal", True),
    "null_commuted_row_sum": (mutate_null_commuted_row_sum,
                              "varying_full", MUTATION_MIXED, "normal",
                              False),
}

# Every NULL control carries its reason into the artifact — a null without a
# recorded reason is indistinguishable from a mutation nobody could catch.
NULL_REASONS: Dict[str, str] = {
    "m4_distribute_quarter":
        "measured null (launched, PTX-verified-different, bytes "
        "identical on the cancellation class) with the derivation: 0.25 is an "
        "exact power of two, scaling by it commutes with round-to-nearest "
        "(the rounding grid scales exactly), so 0.25*(A+B) == 0.25*A + 0.25*B "
        "bitwise wherever nothing is subnormal — laptop measurement 0/2M "
        "mismatches on normal and cancellation classes, 565k/2M once 0.25*x "
        "underflows (the subnormal leg's territory). Contrast the recorded m1 "
        "uniform outcome: ARBITRARY-coefficient distribution is not bitwise "
        "neutral and is caught.",
    "m8_metallic_near_ghost_wrapped":
        "measured null (launched, PTX-verified-different on 3 "
        "sites, bytes identical on the all-metallic sweep) with the "
        "derivation: the wrap differs from the zero ghost only on the "
        "partner-axis face-0 plane; a metallic partner axis is always a "
        "Yee-shift-0 transverse axis of the component, so "
        "_mask_metallic_wall_coupling (stepping.py:1256-1283) zeroes exactly "
        "that plane before the row sum, and mirrors — where the mask "
        "abstains — are refused by this family's predicate. Within the "
        "admitted space the near-ghost zeroing is transcription fidelity, "
        "unobservable to any byte gate.",
    "null_commuted_row_sum":
        "f32 addition is bitwise commutative, so the commuted row sum must "
        "be caught 0 times — a leg that reports it caught is comparing "
        "something other than bytes.",
}


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()
    shipped_ptx = kernel_ptx_set(odmod.offdiag_constitutive_step)
    if not shipped_ptx:
        log("[mut] WARNING: no shipped PTX cached yet (run the synthetic leg "
            "first); the stale-cache check will treat an empty set as pass-through")

    for name, (transform, form_name, boundaries, amplitude,
               must_catch) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "NEEDLE MISSED: the mutation matched nothing; "
                                  "the needle drifted off the kernel source"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        counter = gate.CountingKernel(compile_mutated(mutated))
        case = one_case(MUTATION_SHAPE, boundaries, form_name, amplitude,
                        False, kernel=counter, seed_offset=3)
        caught = (not case.get("skipped") and not case.get("error")
                  and not case["verdict"]["bit_identical"])
        entry: Dict[str, Any] = {"sites": hits, "launches": counter.launches,
                                 "row_form": form_name,
                                 "boundaries": list(boundaries),
                                 "amplitude": amplitude,
                                 "caught": caught,
                                 "expected_caught": must_catch}
        if counter.launches == 0:
            entry["error"] = ("DISARMED: the mutated kernel never launched — "
                              "the leg measured nothing and FAILS")
        mutant_ptx = kernel_ptx_set(counter.kernel)
        entry["mutant_ptx_specializations"] = len(mutant_ptx)
        if must_catch:
            if not mutant_ptx:
                entry["error"] = ("PTX UNAVAILABLE for an armed mutation: the "
                                  "stale-cache check cannot run and the leg "
                                  "FAILS (measured platform fact: the cache "
                                  "can serve a stale binary)")
            elif shipped_ptx and all(ptx in shipped_ptx for ptx in mutant_ptx):
                entry["error"] = ("STALE CACHE: every mutant specialization's "
                                  "PTX matches a shipped one — the mutant "
                                  "binary never actually differed")
            entry["ok"] = caught and "error" not in entry
        elif must_catch is False:
            entry["null_reason"] = NULL_REASONS[name]
            if not mutant_ptx:
                entry["error"] = ("PTX UNAVAILABLE for a null control: "
                                  "identical bytes from a stale shipped "
                                  "binary would be indistinguishable from "
                                  "the arithmetic null — the leg measured "
                                  "nothing and FAILS")
            elif shipped_ptx and all(ptx in shipped_ptx for ptx in mutant_ptx):
                entry["error"] = ("STALE CACHE on a null control: every "
                                  "mutant specialization's PTX matches a "
                                  "shipped one, so 'not caught' measured the "
                                  "cache, not the arithmetic")
            if caught:
                entry["error"] = ("a recorded-null rewrite was 'caught' — "
                                  "either the comparator is not comparing "
                                  "bytes or the null's stated reason no "
                                  "longer holds on this case; see "
                                  "null_reason")
            entry["ok"] = ((not caught) and counter.launches > 0
                           and "error" not in entry)
        else:
            entry["ok"] = counter.launches > 0
        log(f"[mut] {name}: sites={hits} launches={counter.launches} "
            f"caught={caught} ok={entry.get('ok')}")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # m1's UNIFORM-coefficient row: the plan predicted a null (algebraic
    # equality), the laptop measured bitwise separation through distributivity
    # rounding — so the outcome is RECORDED, either way, with its reason.
    mutated, hits = mutate_m1_hoisted_coefficient(shipped)
    counter = gate.CountingKernel(compile_mutated(mutated))
    case = one_case(MUTATION_SHAPE, MUTATION_MIXED, "uniform_full", "normal",
                    False, kernel=counter, seed_offset=5)
    uniform_caught = (not case.get("error")
                      and not case["verdict"]["bit_identical"])
    out["m1_uniform_coefficient_outcome"] = {
        "launches": counter.launches,
        "caught": uniform_caught,
        "recorded": "the plan predicted a NULL on a uniform coefficient "
                    "(algebraic equality); the laptop measured the hoist "
                    "bitwise-different there through distributivity rounding "
                    "(a*u + b*u vs (a+b)*u), so either outcome is recorded — "
                    "the varying-coefficient row above is the catch that "
                    "certifies the REGISTRATION",
        "ok": counter.launches > 0}
    log(f"[mut] m1_uniform_coefficient_outcome: caught={uniform_caught}")
    results["mutations"] = out
    save(results, out_path)

    # The allclose-blindness control: m1 (the coefficient hoist) on UNIFORM
    # rows at LOW amplitude. Three measured wrong turns shaped this leg, each
    # recorded: m4 served in the first cut and is a TRUE bitwise null — a
    # null cannot demonstrate blindness. m1 on VARYING rows was
    # measured failing allclose at unit amplitude AND at 2**-16
    # amplitude — on varying coefficients the hoist is a
    # FIRST-ORDER registration defect (u[i] vs u[i+s] is O(1) relative), and
    # rtol is scale-free: a first-order defect is exactly what allclose CAN
    # see. Blindness needs a ROUNDING-SCALE defect, so the control rides the
    # hoist where it is ALGEBRAICALLY equal and differs only through
    # distributivity rounding: UNIFORM rows (a*u + b*u vs (a+b)*u, ~1 ulp
    # relative, far below rtol=1e-5 — the m1_uniform outcome leg measures
    # those bytes separating on device), with the field state scaled by an
    # exact power of two (2**-16; scaling commutes with rounding, so byte
    # mismatches survive) to hold any cancellation-tail ABSOLUTE delta below
    # atol=1e-8 as well.
    mutated, hits = mutate_m1_hoisted_coefficient(shipped)
    counter = gate.CountingKernel(compile_mutated(mutated))
    rng = np.random.default_rng(SEED + 61)
    state, inv_eps = make_host_state(MUTATION_SHAPE, rng)
    blindness_scale = np.float32(2.0 ** -16)
    for name in ALL_NAMES:  # field state only; materials keep their scale
        state[name] *= blindness_scale
    rows = row_form("uniform_full", MUTATION_SHAPE, rng)
    wall_axes = _wall_axes_for(MUTATION_MIXED)
    flat = synthetic_flat_coefficients(MUTATION_SHAPE, rng)
    reference = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(np, reference, broadcast(flat), rows, inv_eps,
                       MUTATION_MIXED, wall_axes)
    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    arrays.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    plan = odmod.plan_offdiagonal_constitutive_from_arrays(
        arrays, {k: cp.asarray(v) for k, v in flat.items()},
        {row: {partner: cp.asarray(volume)
               for partner, volume in partners.items()}
         for row, partners in rows.items()},
        tuple(CODE_OF[b] for b in MUTATION_MIXED), wall_axes, kernel=counter)
    plan.run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    bytes_fail = any(not bit_compare(arrays[name],
                                     reference[name])["bit_identical"]
                     for name in STATE_NAMES)
    allclose_pass = all(bool(np.allclose(probe.to_host(arrays[name]),
                                         reference[name],
                                         rtol=1e-5, atol=1e-8))
                        for name in STATE_NAMES)
    out["allclose_blindness_control"] = {
        "launches": counter.launches,
        "bytes_fail": bytes_fail, "allclose_passes": allclose_pass,
        "ok": bytes_fail and allclose_pass and counter.launches > 0,
        "note": "a defect allclose forgives at rtol=1e-5 while the uint32 "
                "compare convicts it — the reason this gate never uses "
                "allclose as authority"}
    log(f"[mut] allclose_blindness: bytes_fail={bytes_fail} "
        f"allclose_passes={allclose_pass}")

    out["predicate_mutations"] = {
        "delegated": "meep_gpu/test_triton_offdiag_update_e.py (the "
                     "clause-dropping tests) — runs at every merge on the "
                     "laptop, no device required"}
    results["mutations"] = out
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The engine leg — engine-route plans against stepping, on CuPy
# ---------------------------------------------------------------------------

ENGINE_STEPS = 4


def run_engine(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    leg: Dict[str, Any] = {"grids": []}
    for spec in REFERENCE_GRIDS:
        started = time.time()
        row: Dict[str, Any] = {"name": spec["name"], "steps": ENGINE_STEPS}
        reference_fields, reference_pml = _build_reference_fields(cp, spec)
        candidate_fields, candidate_pml = _build_reference_fields(cp, spec)
        plan = odmod.plan_offdiagonal_constitutive(candidate_fields,
                                                   candidate_pml)
        if plan is None:
            row["error"] = ("REFUSED: " + "; ".join(
                odmod.offdiag_constitutive_coverage(
                    candidate_fields, candidate_pml).reasons))
            leg["grids"].append(row)
            log(f"[engine] {spec['name']}: {row['error'][:100]}")
            results["engine"] = leg
            save(results, out_path)
            continue
        row["row_mask"] = list(plan.row_mask)
        row["boundary_codes"] = list(plan.boundary_codes)
        row["wall_axes"] = list(plan.wall_axes)
        identical = True
        for step in range(ENGINE_STEPS):
            stepping.update_E(reference_fields, reference_pml)
            plan.run()
            cp.cuda.runtime.deviceSynchronize()
            for name in STATE_NAMES:
                part = bit_compare(getattr(candidate_fields, name),
                                   getattr(reference_fields, name))
                if not part["bit_identical"]:
                    identical = False
                    row.setdefault("first_divergence",
                                   {"step": step + 1, "array": name,
                                    "differing": part["differing_floats"]})
        row["identical"] = identical
        row["seconds"] = round(time.time() - started, 3)
        leg["grids"].append(row)
        log(f"[engine] {spec['name']}: identical={identical} "
            f"({row['seconds']} s)")
        results["engine"] = leg
        save(results, out_path)
    # REFERENCE_GRIDS is the predicate's own admitted set: a refusal here is a
    # predicate regression, not a smaller sample — so a refused grid FAILS the
    # leg rather than shrinking it.
    ran = [row for row in leg["grids"] if "error" not in row]
    leg["refused"] = [row["name"] for row in leg["grids"] if "error" in row]
    leg["pass"] = (not leg["refused"]
                   and len(ran) == len(REFERENCE_GRIDS)
                   and all(row["identical"] for row in ran))
    results["engine"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_LEGS = "reference,refusals,synthetic,subnormal,identity,mutations,engine"
DEVICE_LEGS = ("synthetic", "subnormal", "identity", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    device_requested = any(leg in DEVICE_LEGS for leg in legs)
    if cp is not None and device_requested:
        # Strip FIRST, guard second: the guard wraps whatever is installed at
        # the seam, so both apply. Raises loudly on a cache dir that could mix
        # policies — before any artifact exists.
        gate.install_ftz_strip()
        from meep_gpu import backends  # noqa: PLC0415

        backends.guard_kernel_compilation(cp)
    write_provenance(out_dir)

    results: Dict[str, Any] = {
        "legs": list(legs),
        "subnormal_policy": gate.policy_stamp(
            "cupy" if cp is not None else "numpy"),
        "courant_note": "dtdx does not enter this sub-step's arithmetic; the "
                        "non-power-of-two-Courant mandate lands in the "
                        "real-layer coefficient tables (0.35 and 0.5) and in "
                        "the composition probe's whole steps",
        "source_control_note": "this sub-step has no sources; the "
                               "source-driven-vs-source-free non-vacuity "
                               "control lives in the composition probe",
    }
    save(results, args.out)

    failures: List[str] = []

    if "reference" in legs:
        if not run_reference_validation(results, args.out, np, "numpy")["pass"]:
            failures.append("reference:numpy")
        if cp is not None:
            if not run_reference_validation(results, args.out, cp,
                                            "cupy")["pass"]:
                failures.append("reference:cupy")
        else:
            results.setdefault("reference", {})["cupy"] = (
                "skipped: cupy is not importable on this host")
            save(results, args.out)

    if "refusals" in legs:
        if not run_refusals(results, args.out)["pass"]:
            failures.append("refusals")

    device_available = cp is not None and _TRITON_AVAILABLE
    device_legs_executed: List[str] = []
    for leg in legs:
        if leg not in DEVICE_LEGS:
            continue
        if not device_available:
            results[leg] = ("SKIPPED cleanly: this leg needs CUDA "
                            "(cupy + triton); this host has neither usable")
            log(f"[{leg}] SKIPPED cleanly: no CUDA/triton on this host")
            save(results, args.out)
            continue
        device_legs_executed.append(leg)
        if leg == "synthetic":
            summary = run_synthetic(results, args.out)
            if not summary["guarded_pass"]:
                failures.append("synthetic")
            # An error row (VACUOUS, or any exception) is a case that measured
            # nothing — a FAILURE, never an exclusion from 'ran'.
            if summary["errors"]:
                failures.append(
                    f"synthetic:{summary['errors']}-error-rows "
                    f"(VACUOUS/exception cases fail the leg)")
            if summary["fusion_on_ran"] and (summary["fusion_on_identical"]
                                             == summary["fusion_on_ran"]):
                failures.append("synthetic:fusion-control-never-diverged")
        elif leg == "subnormal":
            rows = run_subnormal(results, args.out)["rows"]
            if any(row.get("error") for row in rows):
                failures.append("subnormal:vacuous")
        elif leg == "identity":
            if not run_identity(results, args.out)["pass"]:
                failures.append("identity")
        elif leg == "mutations":
            out = run_mutations(results, args.out)
            for name, entry in out.items():
                if isinstance(entry, dict) and entry.get("ok") is False:
                    failures.append(f"mutations:{name}")
                if isinstance(entry, dict) and "error" in entry:
                    failures.append(f"mutations:{name}:{entry['error'][:40]}")
        elif leg == "engine":
            if not run_engine(results, args.out)["pass"]:
                failures.append("engine")

    if cp is not None and device_requested and device_available:
        results["subnormal_policy"] = gate.policy_stamp("cupy")
        residual = gate.ftz_strip_license_reasons()
        if residual:
            failures.append("subnormal-policy: " + "; ".join(residual))

    # A device leg that was REQUESTED but could not run must not leave a
    # certification-shaped headline (75 is the cannot-certify-here convention).
    device_legs_skipped = device_requested and not device_legs_executed
    if failures:
        status = "FAILED"
    elif device_legs_skipped:
        status = ("SKIPPED-DEVICE-LEGS: laptop legs passed but no device leg "
                  "ran (no CUDA/triton) — this artifact certifies nothing "
                  "about the kernel's bytes")
    else:
        status = "passed"
    results["summary"] = {
        "status": status,
        "failures": failures,
        "device_legs_ran": bool(device_legs_executed),
        "device_legs_executed": device_legs_executed,
        "certified_under_subnormal_policy": results["subnormal_policy"].get(
            "policy"),
    }
    save(results, args.out)
    log(f"OFFDIAG GATE "
        f"{'FAILED' if failures else ('DEVICE LEGS SKIPPED (exit 75)' if device_legs_skipped else 'PASSED')}: "
        f"{results['summary']}")
    if failures:
        return 1
    return 75 if device_legs_skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
