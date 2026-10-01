"""Bit-identity gate for the chi2/chi3 nonlinear update_E Triton kernel.

The module under test is ``meep_gpu/triton_kernels/nonlinear_update_e.py`` — NOT
wired into production dispatch (``plan_fast_path`` keeps returning None; nothing
here changes that). This gate is the arbiter of every grouping choice
stepping.py does not force by construction: the four-corner pair association,
the opposite shift directions, the host-rounded double-power scalar arms, the
Pade quotient's division lowering, the compiled linear-arm seam, and the
``fw``/``f`` store order.

Legs, in order (``--legs`` selects a subset):

* ``reference``  — the in-file transcription pinned against ``stepping.update_E``
  itself, on real ``Grid``/``Fields``/``PML`` objects, byte for byte, over
  repeated sub-step calls. Runs on NumPy (the laptop merge bar) and again on
  CuPy when a device is present; emits a per-call state sha256 chain so the two
  backends' bytes can be compared offline.
* ``refusals``   — every predicted refusal (zero chi, no PML, complex storage,
  k != 0, beta != 0, BFAST, fold, cylindrical, registered polarization,
  off-diagonal epsilon) returns None from the engine builder with its named
  reason; 0 admissions counted.
* ``synthetic``  — the sweep. Shapes (16,16,16) / (13,17,11) non-power-of-two /
  (1,1,2000) matching 3rd-harm-1d / (1,160,160), x boundary configs
  (all-periodic, all-metallic, mixed), x chi forms (chi3-only scalar — the
  corpus form — chi2-only, chi2+chi3, chi3 volume, chi2 volume, partial Ez
  only), x amplitude classes (normal ~1e-3 expansion; NEAR-POLE ~0.22 < 1/3, u
  far from 1 so a Pade defect is byte-visible in f32 — the complex gate's
  assertion-layer lesson), x guard off/on, plus scalar-epsilon rows and
  real-layer coefficient tables cut at Courant 0.35 AND 0.5. The reference side
  of every comparison is computed on NUMPY (host IEEE keep), so the sweep is
  the ship-configuration comparison end to end. NOTE dtdx does not enter this
  sub-step's arithmetic at all — the non-power-of-two-Courant mandate lands in
  the real-layer coefficient tables here and in the composition probe's whole
  steps, and the artifact says so rather than sweeping a dead parameter.
* ``subnormal``  — the deliberately-seeded cancellation class, REPORTED
  SEPARATELY under the stamped policy, never merged into the headline.
* ``identity``   — the compiled seam: the NL=(0,0,0) build launched directly
  must be byte-identical to the certified ``kernels.constitutive_step`` on the
  same seeds, and the partial-nonlinearity case's LINEAR components must match
  the certified kernel while its nonlinear component differs (non-vacuous).
* ``mutations``  — defects planted in the shipped kernel's SOURCE (compiled
  from a real file, launch-counted, DISARMED / NEEDLE-MISSED fail paths, and
  the mutant's PTX verified to DIFFER from every shipped specialization — the
  stale-cache platform fact), plus predicted nulls recorded WITH reasons and
  an allclose-blindness control that must pass allclose while failing bytes.
* ``engine``     — ``plan_nonlinear_constitutive`` from the engine's own
  objects against ``stepping.update_E`` on real CuPy grids, repeated calls.

SUBNORMAL POLICY — certification runs UNDER ``ieee_keep_ftz_stripped``. CuPy
unconditionally appends ``-ftz=true`` to every NVRTC compile
(cupy/cuda/compiler.py:552); this gate installs the demonstrated strip
(``gate_triton_complex.install_ftz_strip``: wrap ``compile_using_nvrtc``,
filter the option, private policy-suffixed ``CUPY_CACHE_DIR``) BEFORE any CuPy
compile, refuses to certify when the strip cannot be confirmed exercised, and
stamps every artifact with the policy plus the in-run strip counters. Never
pass ``-ftz=false`` as a user option — NVRTC rejects the duplicate.

Every case prints one flushed line as it lands; the JSON artifact is rewritten
atomically after every case (the progress-reporting rule); the gate writes its own
provenance record (``fingerprints.json`` is a shared ledger this gate does not
write; it is only hashed). Correctness only: no throughput or timing claims.

Usage (the GPU host, one clear device; the cache dir MUST carry the policy token)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_nonlinear.py \\
        --out results/triton_nonlinear_<date>/gate.json

Laptop (NumPy only; device legs and the strip skip cleanly and say so)::

    python -u gate_triton_nonlinear.py --legs reference,refusals \\
        --out /tmp/gate_nonlinear_local.json

A run whose ``--legs`` include any device leg that could NOT run (no
CUDA/triton) never stamps ``passed``: the summary records
``device_legs_ran``, the status says the device legs were skipped, and the
exit code is 75 (the composition probe's cannot-certify-here convention) —
a certification-shaped headline must not be reachable with zero kernel
launches. Ask for the laptop legs explicitly, as above, to get a clean
``passed`` on this host.
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

# The byte comparator, the ULP key and the coefficient fabric come from the
# shared probe; the policy machinery (ftz strip, stamping, counters) and the
# launch-counting wrapper come from the complex gate. Nothing is re-implemented.
import gate_triton_complex as gate  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import nonlinear_update_e as nlmod  # noqa: E402

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

#: 3rd-harm-1d.py's own decade of chi3 values (the script's ``k``); the gate's
#: scalar chis sit in the corpus family, then the amplitude classes scale D.
CHI3_SCALAR = 0.08
CHI2_SCALAR = 0.045
#: A scalar inverse epsilon whose double-rounded square AND cube both differ
#: from the once-rounded ones (test-verified discriminator).
US_SCALAR = 0.37

#: max(|c2| + |c3|) targets per amplitude class. The pole is at expansion 1/3
#: (derived, stepping.py:1354-1357); near_pole sits NEAR it, never past.
AMPLITUDE_TARGETS = {"normal": 1.0e-3, "near_pole": 0.22, "tiny": 3.0e-6}


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
        "triton_kernels/nonlinear_update_e.py": os.path.join(
            kernel_dir, "nonlinear_update_e.py"),
        "triton_kernels/fingerprints.json": os.path.join(
            kernel_dir, "fingerprints.json"),
        "meep_gpu/test_triton_nonlinear_update_e.py": os.path.join(
            package_dir, "test_triton_nonlinear_update_e.py"),
        "parity/gate_triton_nonlinear.py": os.path.abspath(__file__),
        "parity/gate_triton_complex.py": os.path.join(
            _HERE, "gate_triton_complex.py"),
        "parity/probe_fused_kernel_bit_identity.py": os.path.join(
            _HERE, "probe_fused_kernel_bit_identity.py"),
    }
    composition = os.path.join(_HERE, "probe_triton_nonlinear_composition.py")
    if os.path.exists(composition):
        tracked["parity/probe_triton_nonlinear_composition.py"] = composition
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


def reference_transverse_sums(xp: Any, volumes: Dict[str, Any], own_axis: int,
                              boundaries: Sequence[str]) -> Tuple[Any, Any]:
    """stepping._nonlinear_transverse_sums (:1121-1164): the PAIR association,
    half a cell DOWN the partner's axis then the formed pair half a cell UP the
    own axis — (g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1]), NOT MEEP C's
    left-to-right sum."""
    sums = []
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        values = volumes[E_NAMES[partner_axis]]
        pair = values + reference_shift_down(xp, values, partner_axis,
                                             boundaries[partner_axis])
        sums.append(pair + reference_shift_up(xp, pair, own_axis,
                                              boundaries[own_axis]))
    return (sums[0], sums[1])


def reference_nonlinear_u(gs: Any, dsqr: Any, us: Any, chi2: Any,
                          chi3: Any) -> Any:
    """stepping.calc_nonlinear_u (:1025-1027), transcribed in-file::

        c2 = di   * chi2 * (chi1inv * chi1inv)
        c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
        u  = (1 + c2 + 2*c3) / (1 + 2*c2 + 3*c3)

    Written with stepping's own spellings so a Python-float ``us`` takes the
    double powers rounded once (NumPy's weak-scalar semantics) — the behaviour
    the plan's scalar arm reproduces on the device.
    """
    c2 = gs * chi2 * (us * us)
    c3 = dsqr * chi3 * (us * us * us)
    return (1 + c2 + 2 * c3) / (1 + 2 * c2 + 3 * c3)


def reference_update_e(xp: Any, state: Dict[str, Any],
                       coefficients: Dict[str, Any],
                       chi2: Dict[str, Any], chi3: Dict[str, Any],
                       inv_eps: Dict[str, Any],
                       boundaries: Sequence[str]) -> None:
    """The whole nonlinear sub-step, in place on ``state``.

    stepping.update_E (:967-989) with no poles: the D volumes ARE the sources
    (fields.py:1107-1138 aliases undriven components), the nonlinear branch is
    ``(gs * us) * calc_nonlinear_u(...)`` (:1078-1080), the linear branch is
    plain ``gs * us`` (:1071-1073), and the tail is _apply_constitutive_pml's
    no-scratch branch (:2083-2088).
    """
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        gs = volumes[target]
        us = inv_eps[target]
        if chi2.get(target) is None:
            constitutive = gs * us  # MEEP's `else if (u)` branch.
        else:
            first, second = reference_transverse_sums(xp, volumes, own_axis,
                                                      boundaries)
            dsqr = gs * gs + 0.0625 * (first * first + second * second)
            row = gs * us
            constitutive = row * reference_nonlinear_u(
                gs, dsqr, us, chi2[target], chi3[target])
        kps = coefficients["kps_" + "xyz"[own_axis]]
        kms = coefficients["kms_" + "xyz"[own_axis]]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = constitutive
        field += kps * fw
        field -= kms * fw_previous


def reference_expansion(state: Dict[str, Any], chi2: Dict[str, Any],
                        chi3: Dict[str, Any], inv_eps: Dict[str, Any],
                        boundaries: Sequence[str]) -> Tuple[float, float]:
    """(max |c2| + |c3|, min denominator) — stepping.nonlinear_margin's numbers,
    on the host, so the amplitude classes are placed against the derived bound."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    worst_expansion, worst_denominator = 0.0, float("inf")
    for own_axis, target in enumerate(E_NAMES):
        if chi2.get(target) is None:
            continue
        gs = volumes[target]
        us = inv_eps[target]
        first, second = reference_transverse_sums(np, volumes, own_axis,
                                                  boundaries)
        dsqr = gs * gs + 0.0625 * (first * first + second * second)
        c2 = gs * chi2[target] * (us * us)
        c3 = dsqr * chi3[target] * (us * us * us)
        worst_expansion = max(worst_expansion,
                              float(np.max(np.abs(c2) + np.abs(c3))))
        worst_denominator = min(worst_denominator,
                                float(np.min(1 + 2 * c2 + 3 * c3)))
    return worst_expansion, worst_denominator


# ---------------------------------------------------------------------------
# Case fabric
# ---------------------------------------------------------------------------

SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 16),    # 3-D
    (13, 17, 11),    # non-power-of-two extents
    (1, 1, 2000),    # 3rd-harm-1d's collapsed family
    (1, 160, 160),   # 2-D sheet
)
BOUNDARY_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    (PERIODIC, PERIODIC, PERIODIC),
    (METALLIC, METALLIC, METALLIC),
    (PERIODIC, METALLIC, PERIODIC),
)
#: Real-layer coefficient tables are additionally cut at these Courant numbers
#: (the mandate's home in a sub-step dtdx never enters).
LAYER_COURANTS = (0.35, 0.5)


def config_viable(shape: Sequence[int], boundaries: Sequence[str]) -> Optional[str]:
    for axis in range(3):
        if boundaries[axis] == METALLIC and int(shape[axis]) == 1:
            return (f"axis {axis} is collapsed (n=1) and metallic: Grid "
                    f"resolves an invariant axis periodic, so the pairing is "
                    f"unreachable")
    return None


def chi_form(name: str, shape: Tuple[int, int, int],
             rng) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(chi2 map, chi3 map), None entries marking linear components."""
    if name == "chi3_scalar":  # the corpus form (3rd-harm-1d)
        return ({c: 0.0 for c in E_NAMES}, {c: CHI3_SCALAR for c in E_NAMES})
    if name == "chi2_scalar":
        return ({c: CHI2_SCALAR for c in E_NAMES}, {c: 0.0 for c in E_NAMES})
    if name == "chi2_chi3_scalar":
        return ({c: CHI2_SCALAR for c in E_NAMES},
                {c: CHI3_SCALAR for c in E_NAMES})
    if name == "chi3_volume":
        volume = {c: rng.uniform(0.2, 1.4, size=shape).astype(np.float32)
                  * np.float32(CHI3_SCALAR) for c in E_NAMES}
        return ({c: 0.0 for c in E_NAMES}, volume)
    if name == "chi2_volume":
        volume = {c: rng.uniform(-1.4, 1.4, size=shape).astype(np.float32)
                  * np.float32(CHI2_SCALAR) for c in E_NAMES}
        return (volume, {c: 0.0 for c in E_NAMES})
    if name == "partial_ez":
        return ({"Ex": None, "Ey": None, "Ez": CHI2_SCALAR},
                {"Ex": None, "Ey": None, "Ez": CHI3_SCALAR})
    raise ValueError(name)


CHI_FORMS = ("chi3_scalar", "chi2_scalar", "chi2_chi3_scalar",
             "chi3_volume", "chi2_volume", "partial_ez")


def make_host_state(shape: Tuple[int, int, int], rng,
                    us_form: str = "volume") -> Tuple[Dict[str, Any],
                                                      Dict[str, Any]]:
    state = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
             for name in ALL_NAMES}
    if us_form == "volume":
        inv_eps = {name: rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
                   for name in E_NAMES}
    else:
        inv_eps = {name: US_SCALAR for name in E_NAMES}
    return state, inv_eps


def synthetic_flat_coefficients(shape: Tuple[int, int, int],
                                rng) -> Dict[str, np.ndarray]:
    """Seeded random kps/kms per axis, never 1.0, so a swapped axis or a
    dropped coefficient cannot reproduce bit-for-bit."""
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


def scale_to_expansion(state: Dict[str, Any], chi2: Dict[str, Any],
                       chi3: Dict[str, Any], inv_eps: Dict[str, Any],
                       boundaries: Sequence[str], target: float) -> float:
    """Scale the D volumes so max(|c2| + |c3|) lands near ``target``.

    Bisection on the measured expansion (c2 is linear in the scale, c3
    quadratic, so the map is monotone); the scale is rounded to float32 before
    it multiplies the float32 volumes. Returns the measured expansion."""
    low, high = 0.0, 1.0
    measured, _ = reference_expansion(state, chi2, chi3, inv_eps, boundaries)
    if measured <= 0.0:
        raise AssertionError("the seeded state produced zero expansion — the "
                             "nonlinear case would be vacuous")
    while True:
        trial = {**state, **{n: state[n] * np.float32(high)
                             for n in D_NAMES}}
        measured, _ = reference_expansion(trial, chi2, chi3, inv_eps, boundaries)
        if measured >= target or high > 1e12:
            break
        high *= 4.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        trial = {**state, **{n: state[n] * np.float32(mid) for n in D_NAMES}}
        measured, _ = reference_expansion(trial, chi2, chi3, inv_eps, boundaries)
        if abs(measured - target) <= 0.02 * target:
            break
        if measured < target:
            low = mid
        else:
            high = mid
    else:
        mid = 0.5 * (low + high)
    factor = np.float32(mid)
    for name in D_NAMES:
        state[name] = (state[name] * factor).astype(np.float32)
    expansion, denominator = reference_expansion(state, chi2, chi3, inv_eps,
                                                 boundaries)
    if expansion >= 1.0 / 3.0:
        raise AssertionError(f"amplitude class overshot the pole bound: "
                             f"expansion {expansion:.3f} >= 1/3 "
                             f"(denominator {denominator:.3f})")
    return expansion


def _to_device(value: Any) -> Any:
    return cp.asarray(value) if getattr(value, "shape", None) is not None else value


def one_case(shape, boundaries, chi_name, amplitude, guard,
             us_form: str = "volume", coefficient_source: str = "synthetic",
             courant: Optional[float] = None, kernel: Any = None,
             seed_offset: int = 0) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "chi_form": chi_name, "amplitude": amplitude, "guard": guard,
        "us_form": us_form, "coefficient_source": coefficient_source,
    }
    skip = config_viable(shape, boundaries)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 13 + seed_offset)
    state, inv_eps = make_host_state(tuple(shape), rng, us_form)
    chi2, chi3 = chi_form(chi_name, tuple(shape), rng)
    case["expansion"] = scale_to_expansion(
        state, chi2, chi3, inv_eps, boundaries, AMPLITUDE_TARGETS[amplitude])

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
    reference_update_e(np, reference, broadcast(flat), chi2, chi3, inv_eps,
                       boundaries)
    # Non-vacuity: the nonlinear branch must have changed bytes vs the linear
    # arm, or the case measures nothing.
    linear = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(np, linear, broadcast(flat),
                       {c: None for c in E_NAMES}, {c: None for c in E_NAMES},
                       inv_eps, boundaries)
    if all(reference[name].tobytes() == linear[name].tobytes()
           for name in STATE_NAMES):
        case["error"] = ("VACUOUS: the nonlinear branch left every byte equal "
                         "to the linear arm on this seed")
        return case

    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    for name in E_NAMES:
        arrays["inv_eps_" + name] = _to_device(inv_eps[name])
    device_flat = {key: cp.asarray(value) for key, value in flat.items()}
    codes = tuple(CODE_OF[b] for b in boundaries)
    plan = nlmod.plan_nonlinear_constitutive_from_arrays(
        arrays, device_flat,
        {name: (None if chi2[name] is None else _to_device(chi2[name]))
         for name in E_NAMES},
        {name: (None if chi3[name] is None else _to_device(chi3[name]))
         for name in E_NAMES},
        codes, kernel=kernel)
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
                  boundary_configs=BOUNDARY_CONFIGS, chi_forms=CHI_FORMS,
                  amplitudes=("normal", "near_pole")) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    combos = [(shape, boundaries, chi_name, amplitude)
              for shape in shapes for boundaries in boundary_configs
              for chi_name in chi_forms for amplitude in amplitudes]
    extras = []
    # Scalar-epsilon rows: the arm the engine route never reaches.
    for chi_name in ("chi2_chi3_scalar", "chi3_volume"):
        extras.append(((13, 17, 11), (PERIODIC, METALLIC, PERIODIC), chi_name,
                       "near_pole", "scalar", "synthetic", None))
    # Real-layer tables at both Courant numbers.
    for shape in shapes:
        for courant in LAYER_COURANTS:
            extras.append((shape, (PERIODIC, PERIODIC, PERIODIC),
                           "chi2_chi3_scalar", "near_pole", "volume",
                           "layer", courant))
    total = len(guards) * len(combos) + len(extras)
    index = 0
    for guard in guards:
        for shape, boundaries, chi_name, amplitude in combos:
            index += 1
            started = time.time()
            case = one_case(shape, boundaries, chi_name, amplitude, guard,
                            kernel=kernel)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            _log_case(label, index, total, case)
            summary = summarize(cases)
            summary["cases"] = cases
            results.setdefault("synthetic", {})[label] = summary
            save(results, out_path)
    for shape, boundaries, chi_name, amplitude, us_form, source, courant in extras:
        index += 1
        started = time.time()
        case = one_case(shape, boundaries, chi_name, amplitude, False,
                        us_form=us_form, coefficient_source=source,
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
            f"{'/'.join(b[0] for b in case['boundaries'])} {case['chi_form']} "
            f"{case['amplitude']} exp={case.get('expansion', 0):.2e}: "
            f"identical={verdict['bit_identical']} "
            f"(differing={verdict['differing_floats']}/"
            f"{verdict['total_floats']}, maxulp={verdict.get('max_ulp', 0)}) "
            f"({case['seconds']} s)")


# ---------------------------------------------------------------------------
# The subnormal (cancellation) class — reported separately, never merged
# ---------------------------------------------------------------------------

def run_subnormal(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Deliberately-seeded underflow: the sub-step's products go subnormal.

    Under the STRIPPED policy Triton keeps subnormals as NumPy does, so these
    rows are expected identical — but they are reported as their own account
    under the stamped policy, because the unqualified words 'bit-identical'
    must not absorb the class the policy question lives in."""
    rows = []
    for shape, boundaries in (((13, 17, 11), (PERIODIC, METALLIC, PERIODIC)),
                              ((16, 16, 16), (PERIODIC, PERIODIC, PERIODIC))):
        rng = np.random.default_rng(SEED + 91)
        state, inv_eps = make_host_state(shape, rng)
        chi2, chi3 = chi_form("chi2_chi3_scalar", shape, rng)
        tiny = np.float32(1e-38)
        for name in ALL_NAMES:
            state[name] = (state[name] * tiny).astype(np.float32)
        flat = synthetic_flat_coefficients(shape, rng)
        reference = {name: state[name].copy() for name in ALL_NAMES}
        reference_update_e(np, reference, broadcast(flat), chi2, chi3,
                           inv_eps, boundaries)
        subnormals = int(sum(
            int(np.count_nonzero((np.abs(reference[name]) < 2.0 ** -126)
                                 & (reference[name] != 0)))
            for name in STATE_NAMES))
        arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
        for name in E_NAMES:
            arrays["inv_eps_" + name] = cp.asarray(inv_eps[name])
        plan = nlmod.plan_nonlinear_constitutive_from_arrays(
            arrays, {k: cp.asarray(v) for k, v in flat.items()},
            chi2, chi3, tuple(CODE_OF[b] for b in boundaries))
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
        rows.append(row)
        log(f"[subnormal] {row}")
    leg = {"rows": rows,
           "subnormal_policy": gate.policy_stamp("cupy"),
           "note": "reported separately from the headline by design"}
    results["subnormal"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The identity legs — the compiled linear-arm seam, measured
# ---------------------------------------------------------------------------

def run_identity(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

    leg: Dict[str, Any] = {}
    shape = (13, 17, 11)
    rng = np.random.default_rng(SEED + 29)
    state, inv_eps = make_host_state(shape, rng)
    flat = synthetic_flat_coefficients(shape, rng)

    # (1) The NL=(0,0,0) build, launched directly (the Plan refuses to build an
    # all-linear plan — that refusal is itself under test in the laptop file —
    # so the seam is measured at the kernel boundary), against the certified
    # kernels.constitutive_step through its own plan.
    ours = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    ours.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    certified = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    certified.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    device_flat = {k: cp.asarray(v) for k, v in flat.items()}

    kernel = nlmod.nonlinear_constitutive_step_kernel()
    n_elem = shape[0] * shape[1] * shape[2]
    block = nlmod.DEFAULT_BLOCK
    pointers = ([CupyPointer(ours[n]) for n in E_NAMES]
                + [CupyPointer(ours["f_w_" + n]) for n in E_NAMES]
                + [CupyPointer(ours[n]) for n in D_NAMES]
                + [CupyPointer(ours["inv_eps_" + n]) for n in E_NAMES]
                + [CupyPointer(ours[n]) for n in D_NAMES] * 2  # unused chi slots
                + [CupyPointer(device_flat[f"{stem}_{axis}"])
                   for axis in "xyz" for stem in ("kps", "kms")])
    from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION  # noqa: PLC0415
    kernel[((n_elem + block - 1) // block,)](
        *pointers, *([0.0] * 9), *([0.0] * 6),
        shape[0], shape[1], shape[2], n_elem,
        NL0=0, NL1=0, NL2=0, EV0=1, EV1=1, EV2=1,
        QV20=0, QV21=0, QV22=0, QV30=0, QV31=0, QV32=0,
        BCX=0, BCY=1, BCZ=0, BLOCK=block,
        enable_fp_fusion=ENABLE_FP_FUSION)
    plan = launch_module.plan_constitutive_from_arrays("E", certified,
                                                       device_flat)
    plan.run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    verdict = combine({name: bit_compare(ours[name], certified[name])
                       for name in STATE_NAMES})
    leg["all_linear_build_vs_certified"] = {
        "identical": verdict["bit_identical"],
        "differing_floats": verdict["differing_floats"],
        "expected": "identical: the NL=0 arm must compile to the certified "
                    "plain body (module docstring point 5)"}
    log(f"[identity] NL=(0,0,0) vs certified constitutive_step: "
        f"identical={verdict['bit_identical']}")

    # (2) The partial split: chi on Ez only. The certified kernel runs the same
    # seeds; Ex/Ey must MATCH it (the compiled seam) while Ez must NOT (the
    # nonlinearity is live) — the non-vacuity half of the same measurement.
    rng = np.random.default_rng(SEED + 31)
    state, inv_eps = make_host_state(shape, rng)
    chi2, chi3 = chi_form("partial_ez", shape, rng)
    scale_to_expansion(state, chi2, chi3, inv_eps,
                       (PERIODIC, METALLIC, PERIODIC),
                       AMPLITUDE_TARGETS["near_pole"])
    flat = synthetic_flat_coefficients(shape, rng)
    device_flat = {k: cp.asarray(v) for k, v in flat.items()}
    partial = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    partial.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    certified = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    certified.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    plan = nlmod.plan_nonlinear_constitutive_from_arrays(
        partial, device_flat, chi2, chi3, (0, 1, 0))
    plan.run(guard=False)
    launch_module.plan_constitutive_from_arrays(
        "E", certified, device_flat).run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    linear_names = ("Ex", "Ey", "f_w_Ex", "f_w_Ey")
    linear_verdict = combine({name: bit_compare(partial[name], certified[name])
                              for name in linear_names})
    ez_verdict = combine({name: bit_compare(partial[name], certified[name])
                          for name in ("Ez", "f_w_Ez")})
    leg["partial_split"] = {
        "linear_components_identical": linear_verdict["bit_identical"],
        "nonlinear_component_differs": not ez_verdict["bit_identical"],
        "expected": "Ex/Ey identical to the certified kernel, Ez different — "
                    "the byte-visible per-component split"}
    log(f"[identity] partial split: linear identical="
        f"{linear_verdict['bit_identical']} "
        f"Ez differs={not ez_verdict['bit_identical']}")
    leg["pass"] = bool(
        leg["all_linear_build_vs_certified"]["identical"]
        and leg["partial_split"]["linear_components_identical"]
        and leg["partial_split"]["nonlinear_component_differs"])
    results["identity"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The reference leg — the transcription against stepping itself
# ---------------------------------------------------------------------------

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"name": "corpus_metallic_chi3", "cell": (0.8, 0.8, 0.8), "dimensions": 3,
     "boundaries": "metallic", "courant": 0.35, "chi": "chi3_scalar"},
    {"name": "mixed_chi2_chi3", "cell": (2.0, 1.6, 0.0), "dimensions": 2,
     "boundaries": ("periodic", "metallic", "periodic"), "courant": 0.5,
     "chi": "chi2_chi3_scalar"},
    {"name": "partial_volume", "cell": (0.8, 0.8, 0.8), "dimensions": 3,
     "boundaries": None, "courant": 0.35, "chi": "partial_volume"},
    {"name": "marquee_1d", "cell": (0.0, 0.0, 8.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "courant": 0.35,
     "chi": "chi3_scalar"},
    {"name": "near_pole", "cell": (0.8, 0.8, 0.8), "dimensions": 3,
     "boundaries": "metallic", "courant": 0.35, "chi": "chi3_scalar",
     "amplitude": 3.9},
)
REFERENCE_STEPS = 4


def _build_reference_fields(xp, spec: Dict[str, Any]):
    grid = Grid(resolution=10.0, cell_size=spec["cell"],
                dimensions=spec["dimensions"], boundaries=spec["boundaries"],
                courant=spec["courant"], xp=xp)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(SEED + 47)
    if spec["chi"] == "chi3_scalar":
        fields.set_nonlinear_volumes({}, {c: CHI3_SCALAR for c in E_NAMES})
    elif spec["chi"] == "chi2_chi3_scalar":
        fields.set_nonlinear_volumes({c: CHI2_SCALAR for c in E_NAMES},
                                     {c: CHI3_SCALAR for c in E_NAMES})
    elif spec["chi"] == "partial_volume":
        chi2 = rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)
        chi3 = rng.uniform(0.0, 0.1, size=shape).astype(np.float32)
        fields.set_nonlinear_volumes({"Ez": xp.asarray(chi2)},
                                     {"Ez": xp.asarray(chi3)})
    else:
        raise ValueError(spec["chi"])
    fields.enable_pml_storage()
    amplitude = float(spec.get("amplitude", 0.4))
    for name in ("Dx", "Dy", "Dz") + STATE_NAMES:
        getattr(fields, name)[...] = xp.asarray(rng.uniform(
            -amplitude, amplitude, size=shape).astype(np.float32))
    thickness = tuple((2, 2) if shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def run_reference_validation(results: Dict[str, Any], out_path: str, xp,
                             backend_name: str) -> Dict[str, Any]:
    """Pin the in-file transcription against stepping.update_E on real objects."""
    leg: Dict[str, Any] = {"backend": backend_name, "grids": []}
    for spec in REFERENCE_GRIDS:
        started = time.time()
        row: Dict[str, Any] = {"name": spec["name"], "steps": REFERENCE_STEPS}
        fields, pml = _build_reference_fields(xp, spec)
        boundaries = tuple(stepping._boundary_kinds(fields.grid, pml))
        state = {name: getattr(fields, name).copy()
                 for name in ("Dx", "Dy", "Dz") + STATE_NAMES}
        coefficients = {
            f"{label}_{axis}": getattr(pml, f"{label}_{axis}_h")
            for axis in "xyz" for label in ("kps", "kms")}
        chi2 = {name: (fields.chi2_for(name) if fields.is_nonlinear(name)
                       else None) for name in E_NAMES}
        chi3 = {name: (fields.chi3_for(name) if fields.is_nonlinear(name)
                       else None) for name in E_NAMES}
        inv_eps = {name: fields.inverse_epsilon_for(name) for name in E_NAMES}
        margin = stepping.nonlinear_margin(fields, pml)
        row["expansion"] = None if margin is None else margin.expansion
        chain = hashlib.sha256()
        identical = True
        for step in range(REFERENCE_STEPS):
            stepping.update_E(fields, pml)
            reference_update_e(xp, state, coefficients, chi2, chi3, inv_eps,
                               boundaries)
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
            f"expansion={row['expansion']} ({row['seconds']} s)")
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
                "boundaries": "metallic", "courant": 0.35,
                "chi": "chi3_scalar"}
        symmetry = overrides.pop("symmetry", ())
        k_point = overrides.pop("k_point", (0.0, 0.0, 0.0))
        beta = overrides.pop("beta", 0.0)
        bfast = overrides.pop("bfast", (0.0, 0.0, 0.0))
        complex_storage = overrides.pop("complex_storage", False)
        chis = overrides.pop("chis", "chi3_scalar")
        pml_thickness = overrides.pop("pml_thickness", 2)
        spec.update(overrides)
        grid = Grid(resolution=10.0, cell_size=spec["cell"],
                    dimensions=spec["dimensions"],
                    boundaries=spec["boundaries"], courant=spec["courant"],
                    symmetry=symmetry, k_point=k_point, beta=beta,
                    bfast_scaled_k=bfast, xp=xp)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.set_background_eps(2.25)
        if chis == "chi3_scalar":
            fields.set_nonlinear_volumes({}, {c: CHI3_SCALAR for c in E_NAMES})
        elif chis == "zero":
            fields.set_nonlinear_volumes({c: 0.0 for c in E_NAMES},
                                         {c: 0.0 for c in E_NAMES})
        fields.enable_pml_storage()
        shape = tuple(grid.shape)
        thickness = tuple(
            (pml_thickness, pml_thickness) if shape[axis] >= 6 else (0, 0)
            for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness)

    rows = []
    admissions = 0

    def check(name: str, needle: str, builder: Callable[[], Tuple[Any, Any]],
              mutate: Optional[Callable[[Any], None]] = None) -> None:
        nonlocal admissions
        row: Dict[str, Any] = {"case": name, "expected_reason": needle}
        try:
            fields, pml = builder()
            if mutate is not None:
                mutate(fields)
            plan = nlmod.plan_nonlinear_constitutive(fields, pml)
            reasons = list(nlmod.nonlinear_constitutive_coverage(
                fields, pml).reasons)
            if backend == "numpy":
                reasons = [r for r in reasons if "array module" not in r]
            row["plan_is_none"] = plan is None
            row["named"] = any(needle in reason for reason in reasons)
            row["reasons"] = reasons[:6]
            if plan is not None:
                admissions += 1
            row["ok"] = row["plan_is_none"] and row["named"]
        except ValueError as exc:
            # A constructor that refuses the configuration outright is itself
            # a refusal — recorded verbatim, exactly as the composition probes
            # record grid-level refusals.
            row["constructor_refusal"] = str(exc)[:200]
            row["ok"] = True
        rows.append(row)
        log(f"[refusals] {name}: ok={row['ok']}")

    check("zero_chi", "no chi2/chi3", lambda: build(chis="zero"))
    check("no_pml", "1019-1022", lambda: build(pml_thickness=0))
    check("complex_storage", "force_complex_fields",
          lambda: build(complex_storage=True))
    check("k_nonzero", "k_point", lambda: build(boundaries=None,
                                                k_point=(0.3, 0.0, 0.0)))
    check("beta", "beta", lambda: build(cell=(0.8, 0.8, 0.0), dimensions=2,
                                        boundaries=None, beta=0.25))
    check("bfast", "BFAST", lambda: build(bfast=(0.5, 0.0, 0.0),
                                          boundaries=None))
    check("mirror", "mirror", lambda: build(symmetry=("X",), boundaries=None,
                                            pml_thickness=0))
    check("polarization", "susceptibility is registered", lambda: build(),
          mutate=lambda fields: fields.polarizations.append(SimpleNamespace(
              driven=lambda: ("Ez",), drives=lambda name: name == "Ez")))
    check("offdiag", "off-diagonal", lambda: build(),
          mutate=lambda fields: fields.set_epsilon_volumes(
              {c: fields.epsilon_for(c) for c in E_NAMES},
              {c: fields.inverse_epsilon_for(c) for c in E_NAMES},
              chi1inv_offdiagonal={"Ex": {"Ey": xp.full(
                  fields.grid.shape, 0.05, dtype=xp.float32)}}))
    check("cylindrical", "cylindrical", lambda: build(),
          mutate=lambda fields: setattr(fields.grid, "cylindrical", True))

    leg = {"backend": backend, "rows": rows, "admissions": admissions,
           "pass": admissions == 0 and all(r["ok"] for r in rows)}
    results["refusals"] = leg
    save(results, out_path)
    log(f"[refusals] admissions={admissions} pass={leg['pass']}")
    return leg


# ---------------------------------------------------------------------------
# Mutations — compiled from source, launch-counted, PTX-verified
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def compile_mutated(source: str, kernel_name: str = "nonlinear_constitutive_step"):
    """Compile a mutated copy of the shipped kernel from a real file on disk
    (Triton reads source via inspect). The constexpr codes are re-declared in
    the header because a @triton.jit body may not read a plain module global."""
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_nonlinear.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_nonlinear_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    """Helpers first (they may be the mutation target), then the kernel."""
    functions = (nlmod._four_point_sum, nlmod._pade_u,
                 nlmod.nonlinear_constitutive_step)
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


#: The kernel spells the quotient ``tl.math.div_rn(num, den)`` — the plain
#: ``/`` is ``div.full.f32`` (~2 ulp) on this platform, measured by a
#: 0/116 sweep and pinned by the divprobe (``div_rn`` 0 differing
#: words on 3 x 2^20 vectors, plain ``/`` ~30%% differing at max_ulp 2).
_QUOTIENT = "return tl.math.div_rn(num, den)"


def mutate_m1_quotient_swapped(source: str) -> Tuple[str, int]:
    """m1: u's numerator and denominator swapped."""
    replacement = "return tl.math.div_rn(den, num)"
    return source.replace(_QUOTIENT, replacement), source.count(_QUOTIENT)


def mutate_m2_drop_2x_on_c3(source: str) -> Tuple[str, int]:
    """m2: the numerator's 2*c3 becomes c3 — u = (1 + c2 + c3)/(1 + 2c2 + 3c3).
    Byte-visible only where c3 is large enough to survive the +1 (the NEAR-POLE
    class); the tiny-amplitude row is the predicted, recorded null."""
    needle = "num = (1.0 + c2) + 2.0 * c3"
    return (source.replace(needle, "num = (1.0 + c2) + c3"),
            source.count(needle))


def mutate_m3_drop_0625(source: str) -> Tuple[str, int]:
    """m3: the (1/4)^2 renormalization of the two unnormalized four-point sums
    dropped from Dsqr."""
    pattern = re.compile(r"0\.0625 \* \(g1s \* g1s \+ g2s \* g2s\)")
    return (pattern.sub("(g1s * g1s + g2s * g2s)", source),
            len(pattern.findall(source)))


def mutate_m4_same_direction_double_shift(source: str) -> Tuple[str, int]:
    """m4: the own-axis shift taken DOWN instead of UP — the half-cell
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


_M5_NEEDLE = """    near = (tl.load(g + o_c, mask=v_c, other=0.0)
            + tl.load(g + o_d, mask=v_d, other=0.0))
    far = (tl.load(g + o_u, mask=v_u, other=0.0)
           + tl.load(g + o_ud, mask=v_ud, other=0.0))
    return near + far"""

_M5_REPLACEMENT = """    return ((tl.load(g + o_c, mask=v_c, other=0.0)
             + tl.load(g + o_u, mask=v_u, other=0.0))
            + tl.load(g + o_d, mask=v_d, other=0.0)) \\
        + tl.load(g + o_ud, mask=v_ud, other=0.0)"""


def mutate_m5_left_to_right_association(source: str) -> Tuple[str, int]:
    """m5: MEEP C's left-to-right corner sum (step_generic.cpp:646-648)
    substituted for stepping's pair association (:1158-1162)."""
    return (source.replace(_M5_NEEDLE, _M5_REPLACEMENT),
            source.count(_M5_NEEDLE))


_M6_NEEDLE = """        g1s = _four_point_sum(
            g1, idx,
            i * nyz + dj * nz + k,
            ui * nyz + j * nz + k,
            ui * nyz + dj * nz + k,
            live, dvy, uvx, uvx & dvy)"""

_M6_REPLACEMENT = """        g1s = _four_point_sum(
            g1, idx,
            i * nyz + dj * nz + k,
            i * nyz + uj * nz + k,
            idx,
            live, dvy, uvy, uvy)"""


def mutate_m6_pair_up_partner_axis(source: str) -> Tuple[str, int]:
    """m6: component 0's first pair shifted up the PARTNER axis instead of the
    own axis — shift_up(pair, partner) makes the far pair (g[j+1] + g[j]), so
    the corner collapses onto the center."""
    return (source.replace(_M6_NEEDLE, _M6_REPLACEMENT),
            source.count(_M6_NEEDLE))


# _pade_u's DOCSTRING repeats the c3 spelling as an example (its c2 example is
# spacing-padded and never matched). The needles therefore carry the newline
# plus the CODE indentation — after shipped_source()'s dedent, _pade_u's body
# sits at four spaces while its docstring example sits at eight — so the
# docstring line is neither counted nor rewritten and 'sites' reports exactly
# the arithmetic sites (the audit's m7/m8 sites=2-for-1 inflation).
_M7_C2_NEEDLE = "\n    c2 = (gs * chi2) * us_sq"
_M7_C3_NEEDLE = "\n    c3 = (dsqr * chi3) * us_cu"


def mutate_m7_chi2_chi3_swapped(source: str) -> Tuple[str, int]:
    """m7: chi2 and chi3 exchanged between c2 and c3."""
    hits = 0
    source, n = ((source.replace(_M7_C2_NEEDLE,
                                 "\n    c2 = (gs * chi3) * us_sq"), 1)
                 if _M7_C2_NEEDLE in source else (source, 0))
    hits += n
    source, n = ((source.replace(_M7_C3_NEEDLE,
                                 "\n    c3 = (dsqr * chi2) * us_cu"), 1)
                 if _M7_C3_NEEDLE in source else (source, 0))
    hits += n
    return source, hits


def mutate_m8_power_miscount(source: str) -> Tuple[str, int]:
    """m8: c3 built on chi1inv SQUARED instead of cubed."""
    return (source.replace(_M7_C3_NEEDLE, "\n    c3 = (dsqr * chi3) * us_sq"),
            source.count(_M7_C3_NEEDLE))


def mutate_m9_dsqr_from_scaled_gs(source: str) -> Tuple[str, int]:
    """m9: gs scaled by inv_eps BEFORE Dsqr is formed — |E|^2 standing in for
    the |D|^2 MEEP's Pade factor is defined on."""
    pattern = re.compile(r"dsqr(\d) = gs\1 \* gs\1")
    return (pattern.sub(r"dsqr\1 = (gs\1 * us\1) * (gs\1 * us\1)", source),
            len(pattern.findall(source)))


def mutate_m10_store_order(source: str) -> Tuple[str, int]:
    """m10: prev read AFTER the fw store — the certified constitutive family's
    store-order defect, whose observable is prev == src."""
    pattern = re.compile(r"a(\d) = a\1 - km_\1 \* prev\1")
    return (pattern.sub(r"a\1 = a\1 - km_\1 * src\1", source),
            len(pattern.findall(source)))


def mutate_null_commuted_multiply(source: str) -> Tuple[str, int]:
    """NULL control: the NONLINEAR arm's constitutive product commuted —
    ``(gs * us) * u`` -> ``(us * gs) * u``. float32 multiplication is bitwise
    commutative, so this must be caught 0 times — a leg that reports it caught
    is comparing something other than bytes.

    The rewrite targets the NL arm ON PURPOSE: armed on ``chi3_scalar`` the
    plan is ``nonlinear=(1,1,1)``, so the ``tl.constexpr`` specialization
    traces exactly these three lines. (The audit demonstrated the earlier
    form of this control rewrote only the ``else:`` linear arms — compile-time
    DEAD under NL=(1,1,1) — so its 'not caught' verdict was an unchanged
    binary, not commutativity. The linear arms get their own live control
    below, armed on ``partial_ez``.)"""
    pattern = re.compile(r"src(\d) = \(gs\1 \* us\1\) \* u\1")
    return (pattern.sub(r"src\1 = (us\1 * gs\1) * u\1", source),
            len(pattern.findall(source)))


def mutate_null_commuted_multiply_linear_arm(source: str) -> Tuple[str, int]:
    """NULL control for the compiled LINEAR arm: ``gs * us`` -> ``us * gs`` in
    the ``else:`` branches of components 0 and 1 only. Armed on ``partial_ez``
    (``nonlinear=(0,0,1)``) those two arms are the LIVE code of the launched
    specialization; component 2's else-arm is constexpr-dead there, so the
    regex excludes it rather than let 'sites' count dead code."""
    pattern = re.compile(r"src([01]) = gs\1 \* us\1")
    return (pattern.sub(r"src\1 = us\1 * gs\1", source),
            len(pattern.findall(source)))


#: name -> (transform, chi form to arm it on, amplitude, expectation).
#: expectation True = must catch; None = record only; False = must NOT catch.
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str, str,
                                  Optional[bool]]] = {
    "m1_quotient_swapped": (mutate_m1_quotient_swapped,
                            "chi2_chi3_scalar", "near_pole", True),
    "m2_drop_2x_on_c3": (mutate_m2_drop_2x_on_c3,
                         "chi3_scalar", "near_pole", True),
    "m3_drop_0625": (mutate_m3_drop_0625, "chi3_scalar", "near_pole", True),
    "m4_same_direction_double_shift": (mutate_m4_same_direction_double_shift,
                                       "chi3_scalar", "near_pole", True),
    "m5_left_to_right_association": (mutate_m5_left_to_right_association,
                                     "chi3_scalar", "near_pole", True),
    "m6_pair_up_partner_axis": (mutate_m6_pair_up_partner_axis,
                                "chi3_scalar", "near_pole", True),
    "m7_chi2_chi3_swapped": (mutate_m7_chi2_chi3_swapped,
                             "chi2_chi3_scalar", "near_pole", True),
    "m8_power_miscount": (mutate_m8_power_miscount,
                          "chi3_scalar", "near_pole", True),
    "m9_dsqr_from_scaled_gs": (mutate_m9_dsqr_from_scaled_gs,
                               "chi3_scalar", "near_pole", True),
    "m10_store_order": (mutate_m10_store_order,
                        "chi3_scalar", "near_pole", True),
    "null_commuted_multiply": (mutate_null_commuted_multiply,
                               "chi3_scalar", "near_pole", False),
    "null_commuted_multiply_linear_arm": (
        mutate_null_commuted_multiply_linear_arm,
        "partial_ez", "near_pole", False),
}

#: The dedicated mutation shape: non-power-of-two, mixed boundaries, so every
#: needle's arithmetic is live.
MUTATION_SHAPE = (13, 17, 11)
MUTATION_BOUNDARIES = (PERIODIC, METALLIC, PERIODIC)


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()
    shipped_ptx = kernel_ptx_set(nlmod.nonlinear_constitutive_step)
    if not shipped_ptx:
        log("[mut] WARNING: no shipped PTX cached yet (run the synthetic leg "
            "first); the stale-cache check will treat an empty set as pass-through")

    for name, (transform, chi_name, amplitude,
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
        case = one_case(MUTATION_SHAPE, MUTATION_BOUNDARIES, chi_name,
                        amplitude, False, kernel=counter, seed_offset=3)
        caught = (not case.get("skipped") and not case.get("error")
                  and not case["verdict"]["bit_identical"])
        entry: Dict[str, Any] = {"sites": hits, "launches": counter.launches,
                                 "chi_form": chi_name, "amplitude": amplitude,
                                 "expansion": case.get("expansion"),
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
            entry["ok"] = (not caught) and counter.launches > 0
            if caught:
                entry["error"] = ("a bitwise-commutative rewrite was 'caught' "
                                  "— the comparator is not comparing bytes")
        else:
            entry["ok"] = counter.launches > 0
        log(f"[mut] {name}: sites={hits} launches={counter.launches} "
            f"caught={caught} ok={entry.get('ok')}")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # m2's predicted null: at TINY amplitude the dropped 2* is below the +1's
    # float32 resolution and the bytes coincide — recorded WITH its reason,
    # never as a silent skip.
    mutated, hits = mutate_m2_drop_2x_on_c3(shipped)
    counter = gate.CountingKernel(compile_mutated(mutated))
    case = one_case(MUTATION_SHAPE, MUTATION_BOUNDARIES, "chi3_scalar",
                    "tiny", False, kernel=counter, seed_offset=5)
    null_identical = (not case.get("error")
                      and case["verdict"]["bit_identical"])
    out["m2_tiny_amplitude_null"] = {
        "launches": counter.launches,
        "expansion": case.get("expansion"),
        "identical": null_identical,
        "predicted": "NULL at expansion ~3e-6 ONLY IF c3's contribution to u "
                     "rounds away against 1.0 in f32; either outcome is "
                     "recorded — the near-pole row above is the catch that "
                     "certifies, this row is the amplitude-resolution account",
        "ok": counter.launches > 0}
    log(f"[mut] m2_tiny_amplitude_null: identical={null_identical} "
        f"(expansion={case.get('expansion') or 0.0:.2e})")
    results["mutations"] = out
    save(results, out_path)

    # The allclose-blindness control: the SAME m2 mutant at an amplitude where
    # its deltas sit below allclose's rtol but above a float32 ulp — allclose
    # must PASS while the byte compare FAILS, or the harness's instrument is
    # not the byte compare it claims.
    counter = gate.CountingKernel(compile_mutated(mutated))
    rng = np.random.default_rng(SEED + 61)
    state, inv_eps = make_host_state(MUTATION_SHAPE, rng)
    chi2, chi3 = chi_form("chi3_scalar", MUTATION_SHAPE, rng)
    expansion = scale_to_expansion(state, chi2, chi3, inv_eps,
                                   MUTATION_BOUNDARIES, 3.0e-6)
    flat = synthetic_flat_coefficients(MUTATION_SHAPE, rng)
    reference = {name: state[name].copy() for name in ALL_NAMES}
    reference_update_e(np, reference, broadcast(flat), chi2, chi3, inv_eps,
                       MUTATION_BOUNDARIES)
    arrays = {name: cp.asarray(state[name]) for name in ALL_NAMES}
    arrays.update({"inv_eps_" + n: cp.asarray(inv_eps[n]) for n in E_NAMES})
    plan = nlmod.plan_nonlinear_constitutive_from_arrays(
        arrays, {k: cp.asarray(v) for k, v in flat.items()}, chi2, chi3,
        tuple(CODE_OF[b] for b in MUTATION_BOUNDARIES), kernel=counter)
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
        "launches": counter.launches, "expansion": expansion,
        "bytes_fail": bytes_fail, "allclose_passes": allclose_pass,
        "ok": bytes_fail and allclose_pass and counter.launches > 0,
        "note": "a defect allclose forgives at rtol=1e-5 while the uint32 "
                "compare convicts it — the reason this gate never uses "
                "allclose as authority"}
    log(f"[mut] allclose_blindness: bytes_fail={bytes_fail} "
        f"allclose_passes={allclose_pass}")

    out["predicate_mutations"] = {
        "delegated": "meep_gpu/test_triton_nonlinear_update_e.py (the "
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
        plan = nlmod.plan_nonlinear_constitutive(candidate_fields,
                                                 candidate_pml)
        if plan is None:
            row["error"] = ("REFUSED: " + "; ".join(
                nlmod.nonlinear_constitutive_coverage(
                    candidate_fields, candidate_pml).reasons))
            leg["grids"].append(row)
            log(f"[engine] {spec['name']}: {row['error'][:100]}")
            results["engine"] = leg
            save(results, out_path)
            continue
        row["nonlinear"] = list(plan.nonlinear)
        row["boundary_codes"] = list(plan.boundary_codes)
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
    # predicate regression (or a coverage narrowing underfoot), not a smaller
    # sample — so a refused grid FAILS the leg rather than shrinking it. A
    # pass means every reference grid planned AND matched.
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
            # An error row (VACUOUS, or any exception one_case recorded) is a
            # case that measured nothing — a FAILURE, never an exclusion from
            # 'ran'. The audit demonstrated a sweep of vacuous cases could
            # otherwise still headline 'passed'.
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
    # certification-shaped headline: 'passed' with every device leg skipped is
    # reachable with zero kernel launches (the audit demonstrated it on this
    # laptop). Status and exit code say so instead — 75 is the composition
    # probe's cannot-certify-here convention.
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
    log(f"NONLINEAR GATE "
        f"{'FAILED' if failures else ('DEVICE LEGS SKIPPED (exit 75)' if device_legs_skipped else 'PASSED')}: "
        f"{results['summary']}")
    if failures:
        return 1
    return 75 if device_legs_skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
