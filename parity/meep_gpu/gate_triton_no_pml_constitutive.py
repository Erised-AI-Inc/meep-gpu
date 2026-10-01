"""Byte gate for the NO-PML CONSTITUTIVE family (``triton_kernels.no_pml_constitutive``).

THIS FAMILY SHIPS NO KERNEL, so this gate is shaped differently from the five
certified ones and the difference is the point rather than a shortcut. There is no
kernel-vs-reference sweep because there is no kernel; the certifying leg is an
IDENTITY leg — ``stepping.update_H`` / ``stepping.update_E`` called on a fully seeded
``Fields`` with an inert layer must leave EVERY live array byte-unchanged — and its
control is the same assertion on configurations the predicate refuses, where it must
FAIL. An identity leg with no failing control measures nothing.

Legs (``--legs``, default all):

* ``identity``  the covered arm. Per case: seed every allocated array
                non-degenerately (uniform + a signed-zero plane + dense subnormals),
                snapshot uint32 words, call the array path, byte-compare. PASS
                requires zero differing words AND ``plan.runs > 0`` AND a non-vacuous
                census (nonzero words > 0, signed-zero words > 0). A zero-init
                ``Fields`` satisfies "no bytes moved" trivially — that is THE failure
                mode for a null family, so the census is a pass condition, not a
                statistic. Every case records how many field arrays and how many
                polarization arrays it compared; on the covered arm the second count
                is STRUCTURALLY ZERO and the record says so rather than claiming a
                pin it cannot carry (a driven polarization switches ``stores_E`` on
                and the predicate refuses; the zero-sigma case registers a state
                whose ``P`` table is empty). The pin lives on ``controls``.
* ``controls``  the non-vacuity control: the SAME assertion on an active PML and on a
                no-PML run with ``stores_E``. Bytes MUST move, and the predicate MUST
                refuse. Either half passing is a gate failure. This is also the only
                leg on which ``P``/``P_prev`` exist, so it is where "update_E consumes
                a polarization and does not advance it" (stepping.py:969-973) is
                measured: on the dispersive controls E moves and not one P word does.
* ``breadth``   the claim the module makes and no other predicate here makes: the
                null is exact on a fold, a cylindrical grid, complex storage, a Bloch
                phase, a metallic wall, beta, BFAST and a conductivity — every one of
                which every kernel predicate in this package refuses. Admission plus
                identity on each.
* ``overlap``   every other ``update_H``/``update_E`` predicate must refuse where this
                one admits, and this one must refuse where they admit. The partition
                is by construction (both sides ask ``pml.is_active``) and is measured
                rather than asserted.
* ``mutations`` armed, run-counted, DISARMED / NEEDLE-MISSED are failures. Mutants are
                COMPILED FROM A TEXT EDIT of the shipped module into a fresh module
                object with its own sha256, so a mutation that silently ran the
                shipped code is impossible (there is no kernel cache to serve a stale
                binary here, but the equivalent hazard — importing the shipped
                predicate under the mutant's name — is closed the same way).
                Mutations are classified ``byte_visible`` or ``verdict_only`` and the
                classification is part of the record: a null family's predicate
                guards two different things, and only one of them moves floats.
* ``engine``    the substitution on a REAL driver: two ``FdtdDriver``s from one spec,
                one stepped by the array path, one with ``driver.update_H`` /
                ``driver.update_E`` replaced by the null plan, N whole steps, every
                live array byte-compared after every step. Run WITH a source and
                WITHOUT one, and the source-free control must differ from the driven
                run — that is what proves the driven case's arrays were not a
                plausible-looking wall of zeros.

DISCIPLINE, AND WHERE EACH CLAUSE LANDS ON A FAMILY THAT COMPUTES NOTHING:

* uint32 byte compare only, never ``allclose`` — as everywhere.
* a NON-POWER-OF-TWO Courant in every sweep — carried, and recorded as a PREDICTED
  NULL on the covered arm with its reason (the covered arm performs no floating-point
  operation, so no association or FMA hazard exists for a Courant to expose). It is
  load-bearing on the ``controls`` leg, where real arithmetic runs, and that is where
  the record claims it bites.
* PTX evidence — NOT APPLICABLE, stated rather than omitted: this family compiles
  nothing. The substitutes are (a) a per-plan ``runs`` counter, (b) mutant module
  sha256 inequality, and (c) an assertion that the whole covered path imports and
  runs with ``triton`` and ``cupy`` absent, which is measured on the laptop this gate
  was first run on.
* stated step budget — the covered arm has NO divergence mechanism (nothing rounds),
  so its own budget is unbounded; the composed whole-step budget stays whatever the
  curl's is, which for the no-PML curl is the 66 steps ``no_pml.GATE_VERDICT``
  records. This gate does not raise that number and does not claim to.
* extreme-magnitude needles stay finite: the seed carries dense subnormals and signed
  zeros and NO infinity or NaN. Bound: on the control arm the largest product is
  ``kps * src`` with ``kps = kap + sig`` bounded by the layer's own profile and
  ``|src| <= 1``, so nothing overflows and no NaN can be produced to be compared as
  a raw word.
* flux/DFT sample count — NOT APPLICABLE, no monitor is in this family's path. The
  engine leg asserts ``driver.step_count`` instead, which is the same guarantee for
  the thing this leg actually counts.
* MULTIPLE FULL CYCLES, not one call: every identity case runs ``IDENTITY_CYCLES``
  complete ``update_H``+``update_E`` cycles and compares against the ORIGINAL
  snapshot each time, because a sub-step that wrote back exactly what it read would
  pass a single-call identity test and drift on the second. The whole-step form of
  the same question is the engine leg's 24 driver steps.
* SUBNORMAL POLICY stamped, never re-implemented: ``gate_triton_complex``'s
  ``install_ftz_strip`` / ``ftz_strip_license_reasons`` / ``policy_stamp`` are
  imported and called, and the stamp records that the strip is INERT here (nothing
  compiles, so no NVRTC option tuple exists to strip) rather than omitting the
  question. Subnormals are seeded and censused regardless.
* SELF-ENFORCING: :func:`validate_payload` runs over the artifact this gate just
  wrote and fails the run on a contract violation a per-case check cannot see — a
  vacuous partition leg, a missing power-of-two Courant control, a byte-visible
  mutation that flipped a verdict but never ran a plan. The laptop suite
  mutation-tests that function itself.
* THE LEG SET IS PART OF THE CONTRACT. Every check is guarded on "is this leg
  present", so an artifact cut with ``--legs identity`` used to record
  ``contract: passed`` and ``all_ok: true`` with no controls leg, no mutation
  battery, no engine leg and no negative control. ``validate_payload`` now requires
  all six on the named backend, so a partial run exits non-zero and says which legs
  are missing. ``--legs`` remains useful for debugging; it just cannot mint a green
  artifact.
* IDENTITY AND BREADTH MEET THE SAME PER-CASE CONTRACT, through one function. The
  breadth leg — where this family's one distinguishing claim lives — used to be
  checked on two counters and a ``covered`` flag, and passed with a case that moved
  7 words, an all-zero census, one cycle, zero plan runs, or the leg stripped to its
  non-distinguishing wall control.

Usage (laptop, NumPy only — this is the whole gate, no device leg is skipped)::

    python -u gate_triton_no_pml_constitutive.py \\
        --out results/triton_no_pml_constitutive_<date>/gate.json

Usage (the GPU host, to add the CuPy backend rows; no kernel is compiled either way)::

    CUDA_VISIBLE_DEVICES=<one clear device> python -u \\
        gate_triton_no_pml_constitutive.py --backend cupy \\
        --out results/triton_no_pml_constitutive_<date>/gate_cupy.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _path in (_HERE, _REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # The whole gate runs without it; the backend row says so.
    cp = None

# The subnormal-policy machinery is SHARED, never re-implemented: the strip, its
# licensing reasons and the per-backend stamp all live in the complex gate, which is
# where the mechanism was measured (results/device_subnormal_policy_2026-08-11/).
# Carried here even though this family compiles nothing, because a gate
# that silently omits the policy stamp cannot be told apart from one whose policy was
# never considered — and because the ``--backend cupy`` row must be attributable to
# the ship configuration like every other artifact in this package.
import gate_triton_complex as gate  # noqa: E402

from meep_gpu import driver as driver_module  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402
from meep_gpu.dispersion import Susceptibility  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import no_pml_constitutive as family  # noqa: E402

install_ftz_strip = gate.install_ftz_strip
ftz_strip_license_reasons = gate.ftz_strip_license_reasons
policy_stamp = gate.policy_stamp
STRIPPED_POLICY = gate.STRIPPED_POLICY

SHIPPED_SOURCE = os.path.join(_REPO_API, "meep_gpu", "triton_kernels",
                              "no_pml_constitutive.py")

SEED = 20260813

#: Every array a ``Fields`` may own that a constitutive sub-step could conceivably
#: write. Read by name and skipped when unallocated, because which of them EXIST is
#: itself configuration-dependent (no-PML runs own no H, no f_w and no fu).
CANDIDATE_ARRAYS: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
    "f_cond_Bx", "f_cond_By", "f_cond_Bz",
    "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz",
)

#: NOT a power of two, in every sweep. On the covered arm this is a predicted null
#: (see the module docstring); on ``controls`` it is what makes the PML coefficient
#: products non-representable and the divergence unmissable.
COURANT_NP2 = 0.4056
COURANT_P2 = 0.5

#: How many full ``update_H``+``update_E`` cycles each identity case runs before the
#: byte comparison is believed. One is not enough: a sub-step that wrote back exactly
#: what it read would pass a single-call identity test and drift on the second.
IDENTITY_CYCLES = 3


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic, fsync'd rewrite after every case — an interrupted run keeps its rows."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # UTF-8 explicitly: ``json.dump`` defaults to ``ensure_ascii=True`` so the bytes
    # happen to be ASCII today, but the artifact must not depend on that default
    # surviving while the process locale is ASCII (see ``shipped_source``).
    with open(tmp, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, sort_keys=True, default=repr)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def to_host(array: Any) -> np.ndarray:
    """A host copy, decided by the OBJECT rather than by what ``import cupy`` bound.

    ``isinstance(array, cp.ndarray)`` + ``cp.asnumpy`` was the spelling the package
    carried, and it is wrong for a gate whose functions the merge-bar suite calls
    directly: a sibling test module that stubs ``cupy`` into ``sys.modules`` makes
    ``cp`` something that is not CuPy, ``isinstance`` then answers True for a plain
    NumPy array, and ``cp.asnumpy`` does not exist. Measured: every gate-driving
    test in this family's suite failed that way when the whole ``meep_gpu`` suite ran
    in one process, and passed alone. Asking the array for its own ``.get`` needs no
    module identity at all and is correct on both backends — which is why it is now
    what ``probe.to_host`` does as well, rather than only what this gate does.
    """
    if isinstance(array, np.ndarray):
        return array
    getter = getattr(array, "get", None)
    if callable(getter):
        return np.asarray(getter())
    return np.asarray(array)


def words(array: Any) -> np.ndarray:
    """The uint32 view of an array's bytes. The ONLY comparison this gate makes."""
    host = np.ascontiguousarray(to_host(array))
    if host.dtype == np.complex64:
        host = host.view(np.float32)
    elif host.dtype == np.complex128:
        host = host.view(np.float64).astype(np.float32)
    return host.ravel().view(np.uint32).copy()


# ---------------------------------------------------------------------------
# Snapshot, census, compare
# ---------------------------------------------------------------------------

def polarization_arrays(fields: Any) -> Dict[str, Any]:
    """``P`` and ``P_prev`` per (state, component), empty when a run carries none.

    Included in every snapshot so that ``update_E`` CONSUMING a polarization without
    advancing it (``stepping.update_E``'s docstring, stepping.py:969-973; measured
    30/30 by ``dispersive_update_e`` on the PML path) is pinned wherever those arrays
    exist. WHERE THEY EXIST IS THE CONTROLS LEG, not the identity leg, and the
    difference is structural rather than incidental: a driven polarization switches
    ``stores_E`` on, which the predicate refuses, so the covered arm cannot hold one.
    The identity leg's one susceptibility case registers a ZERO-SIGMA state, whose
    ``P`` table is empty by construction (driver.py:1538-1542). Each case therefore
    records ``polarization_arrays_compared``, and the contract requires the controls
    leg to have compared at least one — the gate used to claim the pin for the
    identity leg, where no case has ever put a polarization array in the set.
    """
    out: Dict[str, Any] = {}
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for label in ("P", "P_prev"):
            table = getattr(state, label, None) or {}
            for component, array in table.items():
                if array is not None:
                    out[f"pol{index}.{label}[{component}]"] = array
    return out


def snapshot(fields: Any) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = words(array)
    for name, array in polarization_arrays(fields).items():
        out[name] = words(array)
    return out


def diff(before: Dict[str, np.ndarray],
         after: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """uint32 inequality per array, plus the totals. Never ``allclose``."""
    per: Dict[str, int] = {}
    total = 0
    differing = 0
    names = sorted(set(before) | set(after))
    for name in names:
        if name not in before or name not in after:
            per[name] = -1  # An array that appeared or vanished is a divergence.
            differing += 1
            continue
        a, b = before[name], after[name]
        moved = int(np.count_nonzero(a != b))
        total += int(a.size)
        differing += moved
        if moved:
            per[name] = moved
    return {"bit_identical": differing == 0, "differing_words": differing,
            "total_words": total, "per_array": per,
            "arrays_compared": len(names)}


def census(fields: Any) -> Dict[str, int]:
    """The in-run census that decides whether an identity assertion meant anything.

    ``signed_zero`` counts words equal to ``0x80000000``. A random seed is PROVABLY
    BLIND to that class — ``rng.uniform`` never returns a negative zero — so the seed
    plants a plane of them deliberately and this census is what proves the plane
    survived into the arrays the sub-step could have touched. A census of 0 is
    VACUOUS, not passed.
    """
    nonzero = 0
    signed_zero = 0
    subnormal = 0
    total = 0
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is None:
            continue
        w = words(array)
        total += int(w.size)
        nonzero += int(np.count_nonzero(w != np.uint32(0)))
        signed_zero += int(np.count_nonzero(w == np.uint32(0x80000000)))
        exponent = (w >> np.uint32(23)) & np.uint32(0xFF)
        mantissa = w & np.uint32(0x7FFFFF)
        subnormal += int(np.count_nonzero((exponent == 0) & (mantissa != 0)))
    return {"total_words": total, "nonzero_words": nonzero,
            "signed_zero_words": signed_zero, "subnormal_words": subnormal}


# ---------------------------------------------------------------------------
# Case construction — the engine's own objects, through FdtdDriver
# ---------------------------------------------------------------------------

def seed_arrays(fields: Any, rng: np.random.Generator) -> None:
    """Fill every allocated array with a non-degenerate pattern.

    Three classes on purpose, and each one is here because a defect hides in the
    other two: ordinary normals (the bulk), a plane of NEGATIVE ZEROS (the class a
    random seed cannot reach, and the exact class that separates a plain store from
    ``constitutive_step``'s accumulation), and dense SUBNORMALS (the class CuPy
    flushes and Triton does not — irrelevant to a sub-step that performs no
    operation, which is precisely the claim, so it is seeded rather than assumed).

    No infinity and no NaN. A NaN's sign and payload are IEEE-unspecified, so a raw
    word comparison over one is version-sensitive; every value here stays finite
    through both the covered arm (which touches nothing) and the control arm (whose
    largest product is a bounded PML coefficient times a value of magnitude <= 1).

    THE STRIDED PLANT IS NOT ENOUGH ON A SMALL GRID, and that is measured rather than
    anticipated: the first run of this gate FAILED ``plain_1voxel`` as VACUOUS —
    ``host[3::11]`` and ``host[5::23]`` both miss a one-element array, so a 1x1x1
    case carried 0 signed zeros and 0 subnormals and its "no bytes moved" verdict
    proved nothing. A 1-voxel cell is a REAL corpus row
    (``material-dispersion.py``), so the fix is not to drop the case: each array
    additionally gets one class planted at an ordinal-rotated index, which spreads
    the three classes across the six-or-more arrays of even a single-cell grid and
    keeps the AGGREGATE census — which is what the pass condition reads — non-empty.
    """
    xp = fields.grid.xp
    shape = tuple(fields.grid.shape)
    size = int(np.prod(shape))
    small_classes = (np.float32(-0.0), np.float32(1e-45), np.float32(-3e-44))

    def plant(host: np.ndarray, ordinal: int) -> np.ndarray:
        host[3::11] = np.float32(-0.0)          # the signed-zero plane
        host[5::23] = np.float32(1e-45)         # dense subnormals
        host[7::29] = np.float32(-3e-44)
        # The rotation: guarantees every class appears somewhere on ANY grid size.
        host[(ordinal * 5) % size] = small_classes[ordinal % len(small_classes)]
        return host

    ordinal = 0
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = plant(rng.uniform(-1.0, 1.0, size=size).astype(np.float32), ordinal)
        ordinal += 1
        if str(getattr(array, "dtype", "")) in ("complex64", "complex128"):
            imag = plant(rng.uniform(-1.0, 1.0, size=size).astype(np.float32),
                         ordinal)
            ordinal += 1
            array[...] = xp.asarray(host.reshape(shape)
                                    + 1j * imag.reshape(shape))
        else:
            array[...] = xp.asarray(np.ascontiguousarray(host.reshape(shape)))
    for _, array in polarization_arrays(fields).items():
        host = plant(rng.uniform(-1.0, 1.0, size=size).astype(np.float32), ordinal)
        ordinal += 1
        if str(getattr(array, "dtype", "")) in ("complex64", "complex128"):
            array[...] = xp.asarray(host.reshape(shape)
                                    + 1j * host.reshape(shape))
        else:
            array[...] = xp.asarray(np.ascontiguousarray(host.reshape(shape)))


def build(spec: Dict[str, Any], xp: Any) -> FdtdDriver:
    """One case, built from the engine's own constructor rather than from bare arrays.

    Bare-array construction is what a kernel gate does because a kernel takes
    pointers. This family takes a ``Fields`` and a ``PML`` and asks them two
    questions, so the objects have to be the real ones or the questions are not the
    real questions.

    THE DEVICE PATH GOES THROUGH ``prefer_gpu``, and that is a repair rather than a
    style choice. This function used to build a host driver and then reassign
    ``fdtd.fields.grid.xp = xp`` afterwards — which allocates every array on the
    HOST first (``FdtdDriver.__init__`` resolves its module at driver.py:943 and
    there is no ``xp`` parameter), leaves ``fdtd.xp`` NumPy, and then makes
    :func:`seed_arrays` assign a device array into a NumPy one. Measured on the
    laptop with a stand-in module whose ``asarray`` refuses host conversion exactly
    as ``cupy.ndarray`` does: ``build RAISED: TypeError Implicit conversion to a
    NumPy array is not allowed`` on the FIRST case. Under the slurm's ``set -euo
    pipefail`` that aborted the whole job before the only leg that needs a GPU.
    ``prefer_gpu=True`` is the engine's own route (driver.py:798, :943) and raises
    rather than degrading silently, and the post-condition below is what turns "the
    driver says it is on the device" into a measurement.
    """
    kwargs: Dict[str, Any] = {
        "cell_size": spec["cell"],
        "resolution": spec.get("resolution", 10.0),
        "courant": spec.get("courant", COURANT_NP2),
        "force_complex_fields": spec.get("complex", False),
        "dimensions": spec.get("dimensions", 3),
        "boundaries": spec.get("boundaries", "periodic"),
        "symmetry": spec.get("symmetry", ()),
        "k_point": spec.get("k_point", (0.0, 0.0, 0.0)),
        "beta": spec.get("beta", 0.0),
        "bfast_scaled_k": spec.get("bfast", (0.0, 0.0, 0.0)),
    }
    if spec.get("cylindrical"):
        kwargs["cylindrical"] = True
        kwargs["m"] = spec.get("m", 0)
    if xp is not np:
        kwargs["prefer_gpu"] = True
        kwargs["gpu_id"] = int(os.environ.get("MEEP_GPU_GATE_DEVICE", "0"))
    fdtd = FdtdDriver(**kwargs)
    if fdtd.xp is not xp or fdtd.fields.grid.xp is not xp:
        raise RuntimeError(
            f"case {spec['name']!r} asked for array module {getattr(xp, '__name__', xp)!r} "
            f"and the driver resolved {getattr(fdtd.xp, '__name__', fdtd.xp)!r} "
            f"(grid: {getattr(fdtd.fields.grid.xp, '__name__', fdtd.fields.grid.xp)!r}); "
            "the backend row would not be attributable to the backend it names")
    shape = tuple(fdtd.grid.shape)
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fdtd.set_epsilon(np.ascontiguousarray(epsilon))
    if spec.get("conductivity"):
        fdtd.set_conductivity(float(spec["conductivity"]))
        fdtd.set_b_conductivity(float(spec["conductivity"]))
    if spec.get("dispersion"):
        sigma = np.full(shape, np.float32(0.35), dtype=np.float32)
        fdtd.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=0.05), sigma)
    if spec.get("zero_sigma_polarization"):
        # driver.py:1538-1542 — a term whose sigma is identically zero drives
        # nothing and must NOT switch storage on. The predicate's polarization
        # clause asks driven(), not "is one registered", for exactly this case.
        sigma = np.zeros(shape, dtype=np.float32)
        fdtd.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=0.05), sigma)
    if spec.get("pml"):
        fdtd.setup_pml(spec["pml"])
    if spec.get("inert_pml") is not None:
        # AN INERT LAYER OBJECT, and it has to be built directly. This is the family's
        # load-bearing partition test — ``pml is not None and pml.is_active`` False,
        # the case pml.py:427-434 exists to name — and it was previously spelled
        # ``{"pml": 0.0}``, which is FALSY, so ``setup_pml`` was never called and the
        # row was a salt-only duplicate of plain_3d_np2 with ``fdtd.pml`` None
        # (measured: identical shape, identical arrays_compared, identical census).
        # ``setup_pml(0.0)`` could not have produced it either: a request that absorbs
        # nowhere raises by design (driver.py:2162-2166, measured "PML thickness 0.0
        # absorbs on no face at all"). A ``PML`` object built directly is the
        # documented different matter — "simply inert, and steps bit-identically to no
        # PML" — and is what the merge-bar suite builds too.
        fdtd.pml = PML(grid=fdtd.grid, thickness=spec["inert_pml"])
        if fdtd.pml.is_active:
            raise RuntimeError(
                f"case {spec['name']!r} asked for an INERT layer and got an active one")
    if spec.get("source"):
        fdtd.add_source(dict(spec["source"]))
    seed_arrays(fdtd.fields, np.random.default_rng(SEED + spec.get("salt", 0)))
    return fdtd


ELECTRIC_SOURCE = {"component": "Ez", "center": (0.0, 0.0, 0.0),
                   "size": (0.0, 0.0, 0.0), "source_type": "gaussian",
                   "frequency": 1.0, "fwidth": 0.4, "amplitude": 1.0}

#: The COVERED arm. Every one of these must be admitted on both sides and must move
#: zero bytes. The four names in the ``row`` field are the measured corpus rows the
#: case stands in for (results/predicate_coverage_2026-08-12).
IDENTITY_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "plain_3d_np2", "cell": (1.2, 1.0, 1.4), "courant": COURANT_NP2,
     "salt": 1, "row": "TestSimulation.test_harminv_warnings class"},
    {"name": "plain_3d_p2", "cell": (1.2, 1.0, 1.4), "courant": COURANT_P2,
     "salt": 2, "row": "power-of-two control for the Courant clause"},
    {"name": "plain_2d_metallic", "cell": (2.0, 2.0, 0.0), "dimensions": 2,
     "boundaries": {"x": "metallic", "y": "metallic"}, "courant": COURANT_NP2,
     "salt": 3,
     "row": "TestMedium.test_check_material_frequencies"},
    {"name": "plain_1voxel", "cell": (0.0, 0.0, 0.1), "dimensions": 1,
     "courant": COURANT_NP2, "salt": 4,
     "row": "TestMaterialDispersion.test_material_dispersion_with_user_material"},
    {"name": "plain_inert_pml_object", "cell": (1.2, 1.0, 1.4),
     "courant": COURANT_NP2, "inert_pml": 0, "salt": 5,
     "requires_pml_object": True,
     "row": "an all-zero-face layer: PML.is_active False (pml.py:427-434)"},
    {"name": "plain_zero_sigma_pol", "cell": (1.2, 1.0, 1.4),
     "courant": COURANT_NP2, "zero_sigma_polarization": True, "salt": 6,
     "row": "driver.py:1538-1542, the registered-but-not-driving case"},
    {"name": "plain_with_source", "cell": (1.2, 1.0, 1.4), "courant": COURANT_NP2,
     "source": ELECTRIC_SOURCE, "salt": 7,
     "row": "TestEigenModeSource.test_amp_func_change_sources"},
)

#: The BREADTH arm — admitted here and refused by every kernel predicate in the
#: package. Each name is the clause the other families carry and this one does not.
BREADTH_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "fold_y", "cell": (1.2, 1.6, 1.4), "symmetry": ("y",),
     "courant": COURANT_NP2, "salt": 11,
     "refused_elsewhere": "coverage._grid_reasons clause 5 (fold changes n_a)"},
    {"name": "complex_storage", "cell": (1.2, 1.0, 1.4), "complex": True,
     "courant": COURANT_NP2, "salt": 12,
     "refused_elsewhere": "coverage._grid_reasons clause 2 (complex64 storage)"},
    {"name": "bloch_k", "cell": (1.2, 1.0, 1.4), "complex": True,
     "k_point": (0.4, -1.3, 0.7), "courant": COURANT_NP2, "salt": 13,
     "refused_elsewhere": "coverage._grid_reasons clause 7 (k != 0)"},
    {"name": "cylindrical_m0", "cell": (1.2, 0.0, 1.4), "cylindrical": True,
     "m": 0, "complex": True, "dimensions": 2, "courant": COURANT_NP2, "salt": 14,
     "refused_elsewhere": "coverage._grid_reasons clause 6 (Dcyl)"},
    {"name": "beta_kz", "cell": (1.2, 1.0, 0.0), "dimensions": 2, "beta": 0.2,
     "complex": True, "courant": COURANT_NP2, "salt": 15,
     "refused_elsewhere": "coverage._grid_reasons clause 12 (beta != 0)"},
    {"name": "bfast", "cell": (0.1, 0.1, 4.8), "bfast": (0.31, 0.0, 0.0),
     "courant": COURANT_NP2, "salt": 16,
     "refused_elsewhere": "coverage._grid_reasons clause 11 (BFAST)"},
    {"name": "conductivity", "cell": (1.2, 1.0, 1.4), "conductivity": 0.7,
     "courant": COURANT_NP2, "salt": 17,
     "refused_elsewhere": "no_pml._no_pml_grid_reasons clause 8 (conductive curl)"},
    {"name": "metallic_walls", "cell": (1.2, 1.0, 1.4), "boundaries": "metallic",
     "courant": COURANT_NP2, "salt": 18,
     "refused_elsewhere": "admitted there too; carried as the wall control"},
)

#: The CONTROL arm. The predicate must REFUSE each of these AND the array path must
#: move bytes on the named side. Both halves are pass conditions.
CONTROL_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "active_pml", "cell": (1.2, 1.0, 1.4), "pml": 3, "salt": 21,
     "courant": COURANT_NP2, "must_move": ("H", "E")},
    {"name": "active_pml_p2_courant", "cell": (1.2, 1.0, 1.4), "pml": 3,
     "salt": 22, "courant": COURANT_P2, "must_move": ("H", "E")},
    {"name": "no_pml_dispersive", "cell": (1.2, 1.0, 1.4), "dispersion": True,
     "courant": COURANT_NP2, "salt": 23, "must_move": ("E",)},
    {"name": "no_pml_dispersive_1voxel", "cell": (0.0, 0.0, 0.1), "dimensions": 1,
     "dispersion": True, "courant": COURANT_NP2, "salt": 24, "must_move": ("E",),
     "row": "material-dispersion.py — the one reachable Arm S row"},
)

SIDE_CALL = {"H": stepping.update_H, "E": stepping.update_E}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def one_identity_case(spec: Dict[str, Any], xp: Any,
                      module: Any = family) -> Dict[str, Any]:
    """Admit, snapshot, call the array path, compare, and census. One case."""
    started = time.time()
    fdtd = build(spec, xp)
    fields, pml = fdtd.fields, fdtd.pml
    entry: Dict[str, Any] = {
        "name": spec["name"], "shape": list(fdtd.grid.shape),
        "courant": repr(spec.get("courant", COURANT_NP2)),
        "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
        "stores_E": bool(fields.stores_E),
        # BOTH halves of the partition test are recorded, not just the second: an
        # INERT LAYER PRESENT and NO LAYER AT ALL are different configurations that
        # reach the same ``stepping`` return, and the row that claims the first has
        # to be able to prove a layer was there.
        "pml_object_present": bool(pml is not None),
        "pml_active": bool(pml is not None and pml.is_active),
        "requires_pml_object": bool(spec.get("requires_pml_object")),
        "row": spec.get("row"),
    }
    verdicts = {side: module.null_constitutive_coverage(fields, pml, side)
                for side in ("H", "E")}
    entry["covered"] = {side: bool(v.covered) for side, v in verdicts.items()}
    entry["reasons"] = {side: list(v.reasons) for side, v in verdicts.items()}
    plan = module.plan_null_constitutive_step(fields, pml)
    entry["planned"] = list(plan.covered)

    entry["census_before"] = census(fields)
    before = snapshot(fields)
    # HONEST ACCOUNTING OF WHAT WAS COMPARED. The gate used to claim in its own
    # docstring that P/P_prev are byte-compared on this leg, "which pins update_E
    # consumes P and does not advance it on this path as well". No identity case ever
    # puts one in the set and on the covered arm none can: a DRIVEN polarization
    # switches stores_E on and the predicate refuses (the zero-sigma case registers a
    # state whose P table is empty — measured: driven() -> (), P {} , P_prev {}).
    # So the count is recorded per case, the docstring now says where the pin really
    # lives, and the CONTROLS leg is where it is measured.
    entry["polarization_arrays_compared"] = len(polarization_arrays(fields))
    entry["field_arrays_compared"] = len(before) - entry["polarization_arrays_compared"]
    # MULTIPLE FULL CYCLES, not one call. A single call cannot tell "writes nothing"
    # apart from "writes back what it read at cycle 1 and would drift at cycle 2", and
    # a null that is only idempotent is still not a null. Compared against the
    # ORIGINAL snapshot every cycle, so a drift that cancels between two cycles is
    # still caught. (The whole-step version of the same question is the engine leg's
    # 24 driver steps; this is the sub-step version on the seeded state.)
    per_cycle: List[Dict[str, Any]] = []
    combined = {"bit_identical": True, "differing_words": 0, "total_words": 0,
                "per_array": {}, "arrays_compared": 0}
    for cycle in range(1, IDENTITY_CYCLES + 1):
        for side in ("H", "E"):
            SIDE_CALL[side](fields, pml)
        plan.run()
        comparison = diff(before, snapshot(fields))
        per_cycle.append({"cycle": cycle,
                          "differing_words": comparison["differing_words"]})
        combined["bit_identical"] &= bool(comparison["bit_identical"])
        combined["differing_words"] += comparison["differing_words"]
        combined["total_words"] += comparison["total_words"]
        combined["per_array"].update(comparison["per_array"])
        combined["arrays_compared"] = comparison["arrays_compared"]
    entry["cycles"] = IDENTITY_CYCLES
    entry["per_cycle"] = per_cycle
    entry["comparison"] = combined
    entry["plan_runs"] = {name: p.runs for name, p in plan.plans.items()}
    entry["census_after"] = census(fields)

    non_vacuous = (entry["census_before"]["nonzero_words"] > 0
                   and entry["census_before"]["signed_zero_words"] > 0
                   and entry["census_before"]["subnormal_words"] > 0)
    entry["non_vacuous"] = bool(non_vacuous)
    entry["ok"] = bool(
        all(entry["covered"].values())
        and set(entry["planned"]) == {"update_H", "update_E"}
        and entry["comparison"]["bit_identical"]
        and entry["comparison"]["arrays_compared"] > 0
        # A row that claims an inert LAYER must have had one installed.
        and (entry["pml_object_present"] or not entry["requires_pml_object"])
        # DISARMED guard: not "the plan ran" but "the plan ran once per cycle". A
        # counter that stopped advancing is the kernel-less analogue of a launch
        # count that never moved.
        and all(runs == IDENTITY_CYCLES for runs in entry["plan_runs"].values())
        and non_vacuous)
    if not non_vacuous:
        entry["error"] = ("VACUOUS: the seeded state carried no nonzero / no signed "
                          "zero / no subnormal word, so 'no bytes moved' proves "
                          "nothing")
    entry["seconds"] = round(time.time() - started, 3)
    return entry


def run_identity(results: Dict[str, Any], out_path: str, xp: Any,
                 backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in IDENTITY_CASES:
        entry = one_identity_case(spec, xp)
        cases.append(entry)
        log(f"[identity:{backend}] {entry['name']:26s} shape={tuple(entry['shape'])} "
            f"covered={entry['covered']} identical="
            f"{entry['comparison']['bit_identical']} "
            f"({entry['comparison']['differing_words']}/"
            f"{entry['comparison']['total_words']} words) "
            f"census(nz/-0/sub)={entry['census_before']['nonzero_words']}/"
            f"{entry['census_before']['signed_zero_words']}/"
            f"{entry['census_before']['subnormal_words']} "
            f"ok={entry['ok']} ({entry['seconds']} s)")
        results.setdefault("identity", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["identity"][backend]


def run_breadth(results: Dict[str, Any], out_path: str, xp: Any,
                backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in BREADTH_CASES:
        entry = one_identity_case(spec, xp)
        entry["refused_elsewhere"] = spec["refused_elsewhere"]
        cases.append(entry)
        log(f"[breadth:{backend}] {entry['name']:22s} covered={entry['covered']} "
            f"identical={entry['comparison']['bit_identical']} ok={entry['ok']} "
            f"| refused elsewhere by {spec['refused_elsewhere']}")
        results.setdefault("breadth", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["breadth"][backend]


def run_controls(results: Dict[str, Any], out_path: str, xp: Any,
                 backend: str) -> Dict[str, Any]:
    """The leg that makes ``identity`` a measurement instead of a tautology."""
    cases: List[Dict[str, Any]] = []
    for spec in CONTROL_CASES:
        started = time.time()
        fdtd = build(spec, xp)
        fields, pml = fdtd.fields, fdtd.pml
        entry: Dict[str, Any] = {
            "name": spec["name"], "shape": list(fdtd.grid.shape),
            "courant": repr(spec.get("courant")),
            "stores_E": bool(fields.stores_E),
            "pml_active": bool(pml is not None and pml.is_active),
            "must_move": list(spec["must_move"]), "row": spec.get("row"),
        }
        verdicts = {side: family.null_constitutive_coverage(fields, pml, side)
                    for side in ("H", "E")}
        entry["covered"] = {side: bool(v.covered) for side, v in verdicts.items()}
        entry["reasons"] = {side: list(v.reasons) for side, v in verdicts.items()}
        # THE POLARIZATION PIN LIVES HERE, and only here. P/P_prev exist on a driven
        # run, and a driven run is exactly the arm this family refuses — so the
        # identity leg cannot carry the pin its docstring once claimed for it. On the
        # dispersive controls the arrays ARE in the comparison set, and what the leg
        # measures is that ``update_E`` CONSUMES them without advancing them
        # (stepping.py:969-973): D and E move, P and P_prev do not.
        polarization_names = sorted(polarization_arrays(fields))
        entry["polarization_arrays_compared"] = len(polarization_names)
        entry["moved"] = {}
        for side in ("H", "E"):
            before = snapshot(fields)
            SIDE_CALL[side](fields, pml)
            after = snapshot(fields)
            comparison = diff(before, after)
            comparison["polarization_words_moved"] = sum(
                int(np.count_nonzero(before[name] != after[name]))
                for name in polarization_names)
            entry["moved"][side] = comparison
        refused_where_required = all(
            not entry["covered"][side] for side in spec["must_move"])
        moved_where_required = all(
            entry["moved"][side]["differing_words"] > 0 for side in spec["must_move"])
        # On a control that carries polarizations, update_E must have moved bytes
        # WITHOUT moving a single P/P_prev word. A leg that measured "bytes moved"
        # alone would pass on an update_E that advanced the polarization instead.
        polarization_held = (
            entry["polarization_arrays_compared"] == 0
            or entry["moved"]["E"]["polarization_words_moved"] == 0)
        entry["polarization_held"] = bool(polarization_held)
        entry["refused_where_required"] = refused_where_required
        entry["moved_where_required"] = moved_where_required
        entry["ok"] = bool(refused_where_required and moved_where_required
                           and polarization_held)
        if not moved_where_required:
            entry["error"] = ("CONTROL DID NOT BITE: the array path moved no bytes on "
                              "a side this leg exists to prove is not a no-op; the "
                              "identity leg's assertion is unfalsified")
        elif not polarization_held:
            entry["error"] = ("update_E ADVANCED a polarization: "
                              f"{entry['moved']['E']['polarization_words_moved']} "
                              "P/P_prev words moved, and stepping.py:969-973 says it "
                              "consumes them without writing them")
        entry["seconds"] = round(time.time() - started, 3)
        cases.append(entry)
        moved_counts = {side: entry["moved"][side]["differing_words"]
                        for side in ("H", "E")}
        log(f"[controls:{backend}] {entry['name']:26s} covered={entry['covered']} "
            f"moved={moved_counts} ok={entry['ok']} ({entry['seconds']} s)")
        results.setdefault("controls", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["controls"][backend]


#: EVERY ``update_H``/``update_E`` predicate in the package, by module and name, with
#: the signature each one takes. The overlap leg used to ask FOUR of them — and the
#: four it asked were the general ones, not the families whose configurations the
#: breadth leg is built out of, so the partition claim was unmeasured exactly where it
#: is most interesting (a folded, complex, Bloch, beta, BFAST or cylindrical grid is
#: where a sibling predicate is designed to admit). ``sides`` records whether the
#: predicate takes a ``side`` argument or is E-only, because four of them are.
SIBLING_CONSTITUTIVE_PREDICATES: Tuple[Dict[str, str], ...] = (
    {"label": "ordinary", "module": "meep_gpu.triton_kernels.coverage",
     "name": "constitutive_coverage", "sides": "both"},
    {"label": "dispersive", "module": "meep_gpu.triton_kernels.coverage",
     "name": "dispersive_constitutive_coverage", "sides": "E"},
    {"label": "dispersive_kernel", "module": "meep_gpu.triton_kernels.dispersive_update_e",
     "name": "dispersive_constitutive_coverage", "sides": "E"},
    {"label": "folded", "module": "meep_gpu.triton_kernels.symmetry",
     "name": "folded_constitutive_coverage", "sides": "both"},
    {"label": "cylindrical", "module": "meep_gpu.triton_kernels.cylindrical_triton",
     "name": "cylindrical_constitutive_coverage", "sides": "both"},
    {"label": "complex", "module": "meep_gpu.triton_kernels.complex_fields",
     "name": "complex_constitutive_coverage", "sides": "both"},
    {"label": "cylindrical_complex", "module": "meep_gpu.triton_kernels.cylindrical_complex",
     "name": "cylindrical_complex_constitutive_coverage", "sides": "both"},
    {"label": "beta", "module": "meep_gpu.triton_kernels.special_kz",
     "name": "beta_run_constitutive_coverage", "sides": "both"},
    {"label": "beta_complex", "module": "meep_gpu.triton_kernels.special_kz",
     "name": "beta_run_complex_constitutive_coverage", "sides": "both"},
    {"label": "folded_complex", "module": "meep_gpu.triton_kernels.folded_complex",
     "name": "folded_complex_constitutive_coverage", "sides": "both"},
    {"label": "folded_beta", "module": "meep_gpu.triton_kernels.folded_complex",
     "name": "folded_beta_run_constitutive_coverage", "sides": "both"},
    {"label": "bfast", "module": "meep_gpu.triton_kernels.bfast_curl",
     "name": "bfast_run_constitutive_coverage", "sides": "both"},
    {"label": "nonlinear", "module": "meep_gpu.triton_kernels.nonlinear_update_e",
     "name": "nonlinear_constitutive_coverage", "sides": "E"},
    {"label": "offdiag", "module": "meep_gpu.triton_kernels.offdiag_update_e",
     "name": "offdiag_constitutive_coverage", "sides": "E"},
    {"label": "folded_offdiag", "module": "meep_gpu.triton_kernels.folded_offdiag_update_e",
     "name": "folded_offdiag_constitutive_coverage", "sides": "E"},
    # The three ADMISSIONS the residual-group closure round added (2026-08-14).
    # None of them plans a new kernel — each restates a shipped clause set with
    # one over-broad clause inverted so a CERTIFIED body may step a configuration
    # it was refused for. They are listed here for the same reason the others are:
    # the partition claim is only measured over the predicates it actually asks.
    {"label": "nonlinear_run", "module": "meep_gpu.triton_kernels.nonlinear_update_e",
     "name": "nonlinear_run_constitutive_coverage", "sides": "both"},
    {"label": "folded_complex_offdiag", "module": "meep_gpu.triton_kernels.folded_complex",
     "name": "folded_complex_offdiag_constitutive_coverage", "sides": "both"},
    {"label": "folded_dispersive",
     "module": "meep_gpu.triton_kernels.folded_dispersive_update_e",
     "name": "folded_dispersive_constitutive_coverage", "sides": "E"},
    # Added 2026-08-17. The package-completeness test caught this one absent: the
    # family shipped a constitutive predicate and the overlap leg never asked it,
    # so nothing was checking whether it double-admits with a sibling. An
    # unconsulted predicate is the quiet half of the disjointness failure — two
    # admitters leave the slot UNSELECTED and it falls to the array path, which
    # is a coverage LOSS that no test reports and no wrong answer reveals.
    {"label": "folded_offdiag_dispersive",
     "module": "meep_gpu.triton_kernels.folded_offdiag_dispersive_update_e",
     "name": "folded_offdiag_dispersive_constitutive_coverage", "sides": "E"},
    # Added 2026-08-17. These three are CONSTITUTIVE predicates that do not carry
    # the ``constitutive_coverage`` suffix, which is exactly why the completeness
    # test below never demanded them: it discovered by name. Three predicates
    # across two wired families were therefore never consulted by the overlap
    # leg, so nothing checked whether they double-admit with a sibling — and a
    # double admission leaves the slot UNSELECTED, falling to the array path as
    # a silent coverage LOSS that no test reports and no wrong answer reveals.
    # They take an optional ``probe`` and so remain callable with the leg's
    # two-argument signature; without an artifact they refuse by name, which is
    # an honest refusal rather than an overlap.
    {"label": "complex_stored_e",
     "module": "meep_gpu.triton_kernels.complex_no_pml_stored_e",
     "name": "complex_stored_e_coverage", "sides": "E"},
    {"label": "complex_no_pml_offdiag",
     "module": "meep_gpu.triton_kernels.complex_offdiag_update_e",
     "name": "complex_no_pml_offdiag_update_e_coverage", "sides": "E"},
    {"label": "complex_folded_offdiag",
     "module": "meep_gpu.triton_kernels.complex_offdiag_update_e",
     "name": "complex_folded_offdiag_update_e_coverage", "sides": "E"},
)


def _bind_predicate(function: Any, sides: str) -> Callable[..., Any]:
    """One uniform ``(fields, pml, side)`` call over two different signatures."""
    if sides == "E":
        return lambda f, p, s: (function(f, p) if s == "E" else None)
    return lambda f, p, s: function(f, p, s)


def sibling_constitutive_predicates() -> Tuple[Dict[str, Callable[..., Any]],
                                               Dict[str, str]]:
    """Import every sibling predicate, reporting the ones that could not be imported.

    An unimportable sibling is RECORDED, never swallowed: a predicate that silently
    dropped out of the leg would leave the partition claim unmeasured for that family
    while the leg still reported green off the ones that remained.
    """
    out: Dict[str, Callable[..., Any]] = {}
    failures: Dict[str, str] = {}
    for entry in SIBLING_CONSTITUTIVE_PREDICATES:
        try:
            module = importlib.import_module(entry["module"])
            function = getattr(module, entry["name"])
        except Exception as exc:  # noqa: BLE001
            failures[entry["label"]] = repr(exc)
            continue
        out[entry["label"]] = _bind_predicate(function, entry["sides"])
    return out, failures


def run_overlap(results: Dict[str, Any], out_path: str, xp: Any,
                backend: str) -> Dict[str, Any]:
    """No sub-step may be claimed by two products. Measured, not argued.

    ``plan_step`` fails closed on a double admission, so an overlap would not produce
    a wrong answer — it would produce a REFUSAL, silently costing this family every
    row it exists for. That is why the leg is here and why it runs over the control
    cases too: the interesting direction is the one where the OTHER predicate admits.
    """
    from meep_gpu.triton_kernels.coverage import (  # noqa: PLC0415
        CONSTITUTIVE_SIDES)

    # The two side tables must name the same slots. ``no_pml_constitutive`` spells its
    # own out rather than importing ``coverage``'s (its VALUES are not array names),
    # so this is where the two are pinned against each other.
    sides_agree = set(family.NULL_SIDES) == set(CONSTITUTIVE_SIDES)

    others, import_failures = sibling_constitutive_predicates()
    missing = sorted({entry["label"] for entry in SIBLING_CONSTITUTIVE_PREDICATES}
                     - set(others))

    cases: List[Dict[str, Any]] = []
    for spec in IDENTITY_CASES + BREADTH_CASES + CONTROL_CASES:
        fdtd = build(spec, xp)
        fields, pml = fdtd.fields, fdtd.pml
        entry: Dict[str, Any] = {"name": spec["name"], "admitted": {},
                                 "admitted_modulo_backend": {}}
        for side in ("H", "E"):
            admitted = []
            modulo = []
            if family.null_constitutive_coverage(fields, pml, side).covered:
                admitted.append("null")
                modulo.append("null")
            for label, predicate in others.items():
                try:
                    verdict = predicate(fields, pml, side)
                except Exception as exc:  # noqa: BLE001 - a raising predicate refuses
                    entry.setdefault("raised", {})[f"{label}@{side}"] = repr(exc)
                    continue
                if verdict is None:
                    continue
                if verdict.covered:
                    admitted.append(label)
                # MODULO THE BACKEND CLAUSE, and this is what makes the leg mean
                # something on a laptop. Every kernel predicate opens with "array
                # module is not cupy"; on a NumPy host that one clause refuses all of
                # them, so a raw overlap check would pass vacuously. Dropping exactly
                # that reason — the way the predicate-coverage harness's
                # ``residual_reasons`` does — asks the question the device would ask.
                residual = [reason for reason in verdict.reasons
                            if "not cupy" not in reason]
                if not residual:
                    modulo.append(label)
            entry["admitted"][side] = admitted
            entry["admitted_modulo_backend"][side] = modulo
        entry["side_tables_agree"] = bool(sides_agree)
        entry["predicates_consulted"] = sorted(others)
        entry["predicates_unimportable"] = dict(import_failures)
        entry["predicates_missing"] = list(missing)
        # A predicate that RAISED is not a refusal: it is a predicate whose answer this
        # leg does not have, and the partition claim is unmeasured for it on that case.
        # It used to be recorded under ``raised`` and never read by anything.
        raised = sorted(entry.get("raised", {}))
        entry["ok"] = bool(sides_agree and not missing and not import_failures
                           and not raised and all(
                               len(entry["admitted"][side]) <= 1
                               and len(entry["admitted_modulo_backend"][side]) <= 1
                               for side in ("H", "E")))
        if not entry["ok"]:
            problems = []
            if any(len(entry["admitted_modulo_backend"][s]) > 1 for s in ("H", "E")):
                problems.append("OVERLAP: two products claim the same sub-step")
            if missing or import_failures:
                problems.append(f"predicates absent from the leg: {missing} "
                                f"{sorted(import_failures)}")
            if raised:
                problems.append(f"predicates RAISED rather than refusing: {raised}")
            if not sides_agree:
                problems.append("NULL_SIDES and coverage.CONSTITUTIVE_SIDES disagree")
            entry["error"] = "; ".join(problems)
        cases.append(entry)
        log(f"[overlap:{backend}] {entry['name']:26s} admitted={entry['admitted']} "
            f"modulo_backend={entry['admitted_modulo_backend']} ok={entry['ok']}")
        results.setdefault("overlap", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "predicates_consulted": sorted(others),
            "predicates_declared": [e["label"]
                                    for e in SIBLING_CONSTITUTIVE_PREDICATES],
            "predicates_unimportable": dict(import_failures),
            "note": ("every kernel predicate here is CuPy-gated at clause 1 while the "
                     "null is not, so the raw ``admitted`` column is vacuous on a "
                     "NumPy host. ``admitted_modulo_backend`` drops that one reason "
                     "and is the column that carries the verdict; both are required "
                     "to hold at most one product per sub-step. EVERY sibling "
                     "constitutive predicate in the package is consulted, including "
                     "the ones whose own families the breadth leg's grids belong to."),
            "cases": cases}
        save(results, out_path)
    return results["overlap"][backend]


# ---------------------------------------------------------------------------
# Mutations — text edits of the shipped module, compiled into fresh modules
# ---------------------------------------------------------------------------

def shipped_source() -> str:
    """The shipped module's text, read as UTF-8 REGARDLESS OF THE PROCESS LOCALE.

    ``encoding="utf-8"`` is not tidiness here; without it this function cannot run on
    a CUDA host at all, and the whole mutation battery goes with it. MEASURED on
    the GPU host, 2026-08-13: ``locale.getpreferredencoding(False)`` is ``UTF-8`` at
    interpreter start, still ``UTF-8`` after ``import cupy`` — and
    ``ANSI_X3.4-1968`` after the FIRST device allocation, because CUDA context
    initialisation calls ``setlocale`` and resets ``LC_CTYPE``. Every later ``open``
    that defers to the locale therefore decodes as ASCII, and this module's own text
    is full of em-dashes.

    The consequence was not hypothetical: ``--backend cupy`` ran identity, controls,
    breadth and overlap and then died here with ``UnicodeDecodeError: 'ascii' codec
    can't decode byte 0xe2 in position 240`` at the mutations leg, and the pytest
    suite failed the same way one test after the first one to build a device driver.
    A gate whose mutation battery cannot run on the only host that has a GPU is a
    gate with no armed mutations on the device.

    Reproducible on a laptop with no CUDA at all —
    ``PYTHONCOERCECLOCALE=0 LC_ALL=C`` produces the identical traceback — which is
    what ``test_triton_no_pml_constitutive`` uses to pin it.
    """
    with open(SHIPPED_SOURCE, encoding="utf-8") as handle:
        return handle.read()


#: The one edit ``compile_mutant`` makes that is NOT the mutation, spelled out so the
#: record can be audited: a package-relative import cannot resolve from a loose file.
RELATIVE_IMPORT_REWRITE = ("from .coverage import",
                           "from meep_gpu.triton_kernels.coverage import")


def compile_mutant(source: str, label: str) -> Any:
    """Import a mutated copy of the module under its own name.

    The measured stale-binary hazard has no kernel-cache analogue here, but
    it has a MODULE-cache one: importing the shipped module under the mutant's name
    would report every defect uncaught. ``spec_from_file_location`` onto a distinct
    temporary file with a distinct module name closes it, and the caller records the
    source sha256 so the record shows the two really differed.

    ONE non-mutating rewrite is applied and recorded: the module's ``from .coverage
    import ...`` becomes an absolute import, because a file outside the package has
    no parent to resolve a relative one against. It touches the import line and
    nothing else, and ``RELATIVE_IMPORT_REWRITE`` names the exact substitution so the
    record does not have to be taken on trust.
    """
    import tempfile  # noqa: PLC0415

    directory = tempfile.mkdtemp(prefix="no_pml_const_mut_")
    path = os.path.join(directory, f"{label}.py")
    source = source.replace(*RELATIVE_IMPORT_REWRITE)
    # UTF-8 for the same measured reason as shipped_source: after a CUDA context
    # exists the process locale is ASCII, and writing this text through it raises
    # UnicodeEncodeError on the first em-dash. The read side of the pair is
    # importlib's, which is UTF-8 by language rule (PEP 3120) rather than by locale.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"_mutant_{label}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> Tuple[str, int]:
    count = source.count(old)
    return (source.replace(old, new, 1) if count else source), count


def mutate_drop_inactive_layer_clause(source: str) -> Tuple[str, int]:
    """m1 — the clause the WHOLE family rests on: the layer must be inert."""
    return _replace_once(
        source,
        '        return ["an active PML layer is installed: update_H/update_E run the dsigw "\n'
        '                "accumulation (stepping.py:946-951, :1014-1018), which is not a no-op"]',
        '        return []  # MUTATED m1')


def mutate_drop_stores_e_clause(source: str) -> Tuple[str, int]:
    """m2 — the E side's own clause: stepping.py:983 branches on stores_E."""
    return _replace_once(
        source,
        '    elif stored:\n'
        '        reasons.append(\n'
        '            "fields.stores_E is True: update_E writes E[...] = (D - sum P) * inv_eps "',
        '    elif False:  # MUTATED m2\n'
        '        reasons.append(\n'
        '            "fields.stores_E is True: update_E writes E[...] = (D - sum P) * inv_eps "')


def mutate_plan_run_writes(source: str) -> Tuple[str, int]:
    """m3 — the plan must genuinely do nothing. Here it CLEARS a B component.

    THIS MUTANT REALLY WRITES, and that is a repair. The previous m3 injected a method
    whose body was ``pass`` and recorded its "flip" as ``hasattr(plan,
    '_mutant_side_effect') and not hasattr(shipped, ...)`` — a statement about the edit
    the gate had just made, true by construction, with no float compared anywhere
    (measured: the compiled mutant's plan, run five times against a fully seeded
    ``Fields``, moved 0 words). The defect class m3 exists to arm — "the null plan
    performs work" — was therefore armed by nothing but a source-text AST pin in the
    merge-bar suite, which is the lowest rank of evidence in this package.

    A writing plan is not constructible with one edit: ``NullConstitutivePlan.run``
    holds no reference to any ``Fields`` (``run.__code__.co_varnames[:2] ==
    ('self', 'guard')``) and ``__slots__`` forbids attaching one. So m3 is THREE
    coordinated edits, each counted in the returned site total:

    1. ``__slots__`` gains the slot the mutant needs;
    2. ``plan_null_constitutive`` binds the ``fields`` it was asked about;
    3. ``run`` zeroes ``Bx`` after advancing the counter.

    The evaluation then drives the MUTANT MODULE through :func:`one_identity_case` —
    the ``module`` parameter that had no caller — and requires the identity comparison
    to fail. ``Bx`` carries seeded normals, a signed-zero plane and dense subnormals,
    so zeroing it is byte-visible in f32 by construction.
    """
    total = 0
    source, count = _replace_once(
        source,
        '    __slots__ = ("side", "sub_step", "runs", "block")',
        '    __slots__ = ("side", "sub_step", "runs", "block", "_mutant_fields")'
        '  # MUTATED m3')
    total += count
    source, count = _replace_once(
        source,
        "    if not null_constitutive_coverage(fields, pml, side).covered:\n"
        "        return None\n"
        "    return NullConstitutivePlan(side, block)",
        "    if not null_constitutive_coverage(fields, pml, side).covered:\n"
        "        return None\n"
        "    plan = NullConstitutivePlan(side, block)  # MUTATED m3\n"
        "    plan._mutant_fields = fields\n"
        "    return plan")
    total += count
    source, count = _replace_once(
        source,
        "        del guard\n        self.runs += 1",
        "        del guard  # MUTATED m3\n        self.runs += 1\n"
        "        target = getattr(self, '_mutant_fields', None)\n"
        "        array = getattr(target, 'Bx', None) if target is not None else None\n"
        "        if array is not None:\n"
        "            array[...] = 0")
    total += count
    return source, total


def mutate_polarization_clause_on_registration(source: str) -> Tuple[str, int]:
    """m6 — ask 'is one registered' instead of 'does it drive'.

    OVER-REFUSAL, not a wrong answer: it refuses the zero-sigma case driver.py:1538
    keeps byte-identical. Byte-invisible by construction, so its evidence is the
    verdict flip and the record says so.
    """
    return _replace_once(
        source,
        "        for name in names:\n"
        "            if name in ELECTRIC_COMPONENTS:",
        "        for name in tuple(ELECTRIC_COMPONENTS):  # MUTATED m6\n"
        "            if True:")


def mutate_drop_magnetic_susceptibility_clause(source: str) -> Tuple[str, int]:
    """m4 — the H side's only feature clause. Verdict-only, deliberately."""
    return _replace_once(
        source,
        "            if _call(state, \"drives\", name, default=False):\n"
        "                reasons.append(\n"
        "                    f\"polarization {index} drives magnetic component {name}: the \"",
        "            if False:  # MUTATED m4\n"
        "                reasons.append(\n"
        "                    f\"polarization {index} drives magnetic component {name}: the \"")


def mutate_drop_storage_switch_clause(source: str) -> Tuple[str, int]:
    """m5 — the conservative clause. Verdict-only BY DESIGN; recorded as such."""
    return _replace_once(
        source,
        '        return ["Fields has PML storage enabled while the layer is inert (get_H would "',
        '        return []  # MUTATED m5\n'
        '    if False:\n'
        '        return ["Fields has PML storage enabled while the layer is inert (get_H would "')


class _MagneticState:
    """A duck-typed polarization that claims a magnetic component.

    The engine refuses magnetic susceptibilities at the driver, so this needle cannot
    be built from the engine's own objects — and the predicate takes duck-typed
    states (``coverage._call``) precisely so that "another module already guards it"
    is not the reason a clause is absent. Synthetic, and labelled synthetic.
    """

    def driven(self):
        return ()

    def drives(self, component):
        return component in ("Bx", "Hx")


MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"name": "m1_drop_inactive_layer_clause", "fn": mutate_drop_inactive_layer_clause,
     "kind": "byte_visible", "needle": "active_pml",
     "why": "admits an ACTIVE-PML run, where the null leaves H and E one step stale"},
    {"name": "m2_drop_stores_e_clause", "fn": mutate_drop_stores_e_clause,
     "kind": "byte_visible", "needle": "no_pml_dispersive",
     "why": "admits no-PML + stores_E, where update_E really writes E"},
    {"name": "m3_plan_run_writes", "fn": mutate_plan_run_writes,
     "kind": "byte_visible", "needle": "plain_3d_np2",
     "why": "the plan must perform no operation. The mutant's plan clears Bx on every "
            "run and the identity comparison — taken AFTER plan.run() — catches it in "
            "raw uint32 words. Previously this row injected a method whose body was "
            "'pass' and reported a hasattr check as its flip, which measured the "
            "gate's own edit rather than the shipped run() body"},
    {"name": "m4_drop_magnetic_susceptibility_clause",
     "fn": mutate_drop_magnetic_susceptibility_clause,
     "kind": "verdict_only", "needle": "synthetic_magnetic_state",
     "why": "update_H returns regardless, so no float moves; the clause guards a "
            "sub-step that does not exist yet and its evidence is the verdict"},
    {"name": "m5_drop_storage_switch_clause", "fn": mutate_drop_storage_switch_clause,
     "kind": "verdict_only", "needle": "inert_layer_with_pml_storage",
     "why": "conservative clause; the null is byte-exact there, so the verdict IS "
            "the evidence and the record must not claim otherwise"},
    {"name": "m6_polarization_clause_on_registration",
     "fn": mutate_polarization_clause_on_registration,
     "kind": "verdict_only", "needle": "plain_zero_sigma_pol",
     "why": "over-refuses the zero-sigma case driver.py:1538-1542 keeps "
            "byte-identical; costs coverage rather than correctness"},
)

#: Recorded, NOT armed, with the reason each one cannot bite. A gate that arms these
#: reports 'uncaught' for arithmetic that does not exist.
PREDICTED_NULLS: Tuple[Dict[str, str], ...] = (
    {"name": "non_power_of_two_courant_on_the_covered_arm",
     "reason": "the covered arm executes no floating-point operation, so there is no "
               "association or FMA hazard for a Courant number to expose. Carried in "
               "every sweep anyway (plain_3d_np2 vs plain_3d_p2) and load-bearing on "
               "the controls leg, where real coefficient products run."},
    {"name": "guard_true_vs_guard_false_on_plan_run",
     "reason": "ENABLE_FP_FUSION gates contraction of floating-point operations; "
               "NullConstitutivePlan.run performs none, so both guards are the same "
               "instruction stream (there is no instruction stream)."},
    {"name": "block_size",
     "reason": "no launch, no grid, no block. The parameter is accepted only so the "
               "builder's signature matches the package's other plan builders."},
    {"name": "ptx_distinctness_of_mutants",
     "reason": "NOT APPLICABLE: this family compiles nothing. The stale-artifact "
               "hazard is closed at the MODULE level instead — each mutant is a "
               "distinct file, a distinct module name and a distinct sha256, all "
               "three recorded per mutation."},
    {"name": "cupy_ftz_vs_triton_ieee_subnormal_policy",
     "reason": "the divergence the package tracks is CuPy flushing float32 "
               "subnormals where Triton keeps them. Neither side computes here, so "
               "a subnormal is copied by nobody. Subnormals are seeded anyway and "
               "the census proves they were present."},
    {"name": "boundary_fold_cylindrical_complex_bloch_clauses",
     "reason": "carrying them would only OVER-refuse; a null cannot be wrong about "
               "an index it never forms. Measured by the breadth leg rather than "
               "left as an argument."},
)


def run_mutations(results: Dict[str, Any], out_path: str, xp: Any,
                  backend: str) -> Dict[str, Any]:
    base = shipped_source()
    base_sha = sha256_text(base)
    out: Dict[str, Any] = {"shipped_sha256": base_sha,
                           "shipped_file_sha256": sha256_file(SHIPPED_SOURCE),
                           "predicted_nulls": list(PREDICTED_NULLS),
                           "mutations": {}}
    by_name = {spec["name"]: spec
               for spec in IDENTITY_CASES + BREADTH_CASES + CONTROL_CASES}

    def battery(with_synthetic: bool) -> List[Tuple[str, Any, Any]]:
        """Every (name, fields, pml) a verdict mutation is asked about.

        ``with_synthetic`` adds the one configuration NO corpus row and no
        ``FdtdDriver`` call sequence produces: stored E switched on with nothing
        driving it. It exists because the first run of this gate reported m2 as
        NEEDLE MISSED — dropping the ``stores_E`` clause changed no verdict, because
        every reachable stored-E run ALSO has a driving polarization and clause 4
        refused it anyway. That is a real result about the predicate (clause 3 and
        clause 4's polarization sub-clause are redundant on every reachable
        configuration) and the needle that separates them has to be built by hand:
        ``Fields.enable_field_storage()`` is public, so the state is constructible
        even though no driver path reaches it.
        """
        cases = []
        for name, spec in by_name.items():
            fdtd = build(spec, xp)
            cases.append((name, fdtd.fields, fdtd.pml))
        if with_synthetic:
            fdtd = build(by_name["plain_3d_np2"], xp)
            fdtd.fields.enable_field_storage()
            seed_arrays(fdtd.fields, np.random.default_rng(SEED + 99))
            cases.append(("synthetic_stored_e_no_driver", fdtd.fields, fdtd.pml))
        return cases

    for mutation in MUTATIONS:
        started = time.time()
        entry: Dict[str, Any] = {"kind": mutation["kind"], "why": mutation["why"],
                                 "needle": mutation["needle"]}
        mutated, sites = mutation["fn"](base)
        entry["sites"] = sites
        if sites == 0:
            entry["error"] = ("NEEDLE MISSED: the mutation matched nothing in the "
                              "shipped source — the gate is measuring the shipped "
                              "module under a mutant's name")
            entry["ok"] = False
            out["mutations"][mutation["name"]] = entry
            log(f"[mut:{backend}] {mutation['name']}: NEEDLE MISSED")
            save(results, out_path)
            continue
        entry["mutant_sha256"] = sha256_text(mutated)
        entry["distinct_from_shipped"] = bool(entry["mutant_sha256"] != base_sha)
        module = compile_mutant(mutated, mutation["name"])
        entry["mutant_module"] = module.__name__

        evaluated = 0
        flipped: List[str] = []
        byte_evidence: Dict[str, Any] = {}
        runs = 0

        if mutation["needle"] == "synthetic_magnetic_state":
            fdtd = build(by_name["plain_3d_np2"], xp)
            fdtd.fields.polarizations.append(_MagneticState())
            shipped_v = family.null_constitutive_coverage(
                fdtd.fields, fdtd.pml, "H").covered
            mutant_v = module.null_constitutive_coverage(
                fdtd.fields, fdtd.pml, "H").covered
            evaluated += 1
            if shipped_v != mutant_v:
                flipped.append("synthetic_magnetic_state@H")
        elif mutation["needle"] == "inert_layer_with_pml_storage":
            fdtd = build(by_name["plain_3d_np2"], xp)
            fdtd.fields.enable_pml_storage()  # storage on, layer still inert
            for side in ("H", "E"):
                shipped_v = family.null_constitutive_coverage(
                    fdtd.fields, fdtd.pml, side).covered
                mutant_v = module.null_constitutive_coverage(
                    fdtd.fields, fdtd.pml, side).covered
                evaluated += 1
                if shipped_v != mutant_v:
                    flipped.append(f"inert_layer_with_pml_storage@{side}")
        elif mutation["name"] == "m3_plan_run_writes":
            # THE MUTANT MODULE IS DRIVEN THROUGH THE IDENTITY LEG ITSELF, which is
            # what ``one_identity_case``'s ``module`` parameter was written for and
            # never used for. The identity snapshot is taken AFTER ``plan.run()``, so a
            # plan that writes is caught by the same uint32 comparison that certifies
            # the shipped one — no separate assertion, no source-text pin.
            needle_spec = dict(by_name[mutation["needle"]])
            mutant_entry = one_identity_case(needle_spec, xp, module=module)
            shipped_entry = one_identity_case(needle_spec, xp, module=family)
            evaluated += 1
            moved = int(mutant_entry["comparison"]["differing_words"])
            entry["mutant_identity"] = {
                "bit_identical": bool(mutant_entry["comparison"]["bit_identical"]),
                "differing_words": moved,
                "per_array": mutant_entry["comparison"]["per_array"],
                "plan_runs": mutant_entry["plan_runs"]}
            entry["shipped_identity"] = {
                "bit_identical": bool(shipped_entry["comparison"]["bit_identical"]),
                "differing_words": int(
                    shipped_entry["comparison"]["differing_words"])}
            entry["shipped_run_body_is_empty"] = bool(
                shipped_entry["comparison"]["bit_identical"])
            runs = sum(int(v) for v in mutant_entry["plan_runs"].values())
            if moved > 0 and shipped_entry["comparison"]["bit_identical"]:
                flipped.append(f"{mutation['needle']}@identity")
                byte_evidence[f"{mutation['needle']}@identity"] = {
                    "differing_words": moved,
                    "per_array": mutant_entry["comparison"]["per_array"],
                    "caught": True}
        else:
            for name, fields, pml in battery(with_synthetic=True):
                for side in ("H", "E"):
                    shipped_v = family.null_constitutive_coverage(
                        fields, pml, side).covered
                    mutant_v = module.null_constitutive_coverage(
                        fields, pml, side).covered
                    evaluated += 1
                    if shipped_v == mutant_v:
                        continue
                    flipped.append(f"{name}@{side}")
                    if not mutant_v:
                        continue  # An over-refusal moves no bytes; recorded, not armed.
                    # The mutant ADMITS: the null would be substituted here. Measure
                    # what the array path does that the null would not.
                    plan = module.plan_null_constitutive(fields, pml, side)
                    before = snapshot(fields)
                    SIDE_CALL[side](fields, pml)
                    moved = diff(before, snapshot(fields))
                    if plan is not None:
                        plan.run()
                        runs += plan.runs
                    byte_evidence[f"{name}@{side}"] = {
                        "differing_words": moved["differing_words"],
                        "per_array": moved["per_array"],
                        "caught": bool(moved["differing_words"] > 0)}

        entry["configurations_evaluated"] = evaluated
        entry["verdict_flips"] = flipped
        entry["byte_evidence"] = byte_evidence
        entry["runs"] = runs

        if evaluated == 0:
            entry["error"] = "DISARMED: the mutated predicate was never evaluated"
            entry["ok"] = False
        elif not flipped:
            entry["error"] = ("NEEDLE MISSED: the mutant agreed with the shipped "
                              "predicate on every configuration in the battery")
            entry["ok"] = False
        elif mutation["kind"] == "byte_visible":
            caught = [key for key, value in byte_evidence.items() if value["caught"]]
            entry["caught_on"] = caught
            if runs == 0:
                entry["error"] = ("DISARMED: the mutant admitted a configuration but "
                                  "no null plan was ever run on it")
                entry["ok"] = False
            elif not caught:
                entry["error"] = ("NEEDLE MISSED: the mutant admitted a configuration "
                                  "where the array path moved no bytes, so the "
                                  "substitution would have been harmless")
                entry["ok"] = False
            else:
                entry["ok"] = True
        else:
            entry["ok"] = True
        entry["seconds"] = round(time.time() - started, 3)
        out["mutations"][mutation["name"]] = entry
        log(f"[mut:{backend}] {mutation['name']}: kind={mutation['kind']} "
            f"sites={sites} evaluated={evaluated} flips={len(flipped)} "
            f"runs={runs} ok={entry['ok']}")
        results.setdefault("mutations", {})[backend] = out
        save(results, out_path)
    results.setdefault("mutations", {})[backend] = out
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# Engine — the substitution on a real driver
# ---------------------------------------------------------------------------

ENGINE_STEPS = 24


class _Substitution:
    """Swap ``driver.update_H`` / ``driver.update_E`` for the null plans, for ONE driver.

    ``driver.py:182-194`` imports the two by name, so the swap has to be on the DRIVER
    module's globals — patching ``stepping``'s would leave the driver holding the
    originals and the leg would silently measure nothing.

    THE PATCH IS PER STEP CALL, not per case, and that is a fix rather than a
    refinement. The first run of this leg patched the globals around BOTH drivers'
    step calls, so the "reference" driver ran the null too: the comparison was a null
    against a null and reported 0 differing words on every case. The call counter
    caught it — 48 calls for 24 steps, exactly twice — which is the whole reason a
    substitution leg counts its own invocations. Now the swap is entered immediately
    before the substituted driver's ``step()`` and left immediately after, and
    ``expected`` makes the arming positive: the substituted function RAISES if it is
    ever handed a ``Fields`` other than the one it was built for, so a leak onto the
    reference driver is an error rather than a silent pass.
    """

    def __init__(self, expected_fields: Any, plans: Dict[str, Any]) -> None:
        self.expected = expected_fields
        self.plans = plans
        self.counters: Dict[str, int] = {"update_H": 0, "update_E": 0}
        self.leaks = 0
        self._originals: Optional[Tuple[Any, Any]] = None

    def _make(self, name: str):
        def substituted(fields, pml):
            del pml
            if fields is not self.expected:
                self.leaks += 1
                raise RuntimeError(
                    "the null substitution was invoked for a Fields it was not built "
                    "for: the patch has leaked onto another driver and the leg is "
                    "comparing a null against a null")
            plan = self.plans.get(name)
            if plan is None:
                raise RuntimeError(
                    f"{name} has no null plan; the engine leg must not run the array "
                    "path behind the substitution's back")
            plan.run()
            self.counters[name] += 1
        return substituted

    def __enter__(self) -> "_Substitution":
        self._originals = (driver_module.update_H, driver_module.update_E)
        driver_module.update_H = self._make("update_H")
        driver_module.update_E = self._make("update_E")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._originals is not None:
            driver_module.update_H, driver_module.update_E = self._originals
            self._originals = None
        return None


def run_engine(results: Dict[str, Any], out_path: str, xp: Any,
               backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    specs = (
        {"name": "engine_driven", "cell": (1.2, 1.0, 1.4), "courant": COURANT_NP2,
         "source": ELECTRIC_SOURCE, "salt": 31},
        {"name": "engine_source_free", "cell": (1.2, 1.0, 1.4),
         "courant": COURANT_NP2, "salt": 31},
        {"name": "engine_metallic_driven", "cell": (1.2, 1.0, 1.4),
         "boundaries": "metallic", "courant": COURANT_NP2,
         "source": ELECTRIC_SOURCE, "salt": 32},
    )
    driven_final: Optional[Dict[str, np.ndarray]] = None
    free_final: Optional[Dict[str, np.ndarray]] = None
    for spec in specs:
        started = time.time()
        reference = build(spec, xp)
        substituted = build(spec, xp)
        plan = family.plan_null_constitutive_step(substituted.fields,
                                                  substituted.pml)
        entry: Dict[str, Any] = {
            "name": spec["name"], "shape": list(reference.grid.shape),
            "steps": ENGINE_STEPS, "planned": list(plan.covered),
            "has_source": bool(spec.get("source")),
        }
        if set(plan.covered) != {"update_H", "update_E"}:
            entry["error"] = "the null plan did not cover both sides on this spec"
            entry["ok"] = False
            cases.append(entry)
            continue
        swap = _Substitution(substituted.fields, plan.plans)
        # THE SURROUNDING STEP MUST STILL HAVE DONE WORK. For a family whose whole
        # product is a no-op, "the two drivers agree bit for bit" is satisfied just as
        # well by a step that moved nothing at all — a build that failed to seed, a
        # spec whose source never fired, a metallic case zeroed everywhere. Then the
        # leg would be certifying that two frozen states are equal. The array-path
        # driver's own state before and after the 24 steps is what rules that out, and
        # it is a PASS CONDITION below rather than a statistic: the curls are the only
        # thing that can have moved it, because this family's two sub-steps return
        # before their first statement.
        reference_initial = snapshot(reference.fields)
        first_divergence = None
        totals = {"differing_words": 0, "total_words": 0, "comparisons": 0}
        for step in range(1, ENGINE_STEPS + 1):
            reference.step()          # array path, globals untouched
            with swap:                # null plans, this driver only
                substituted.step()
            comparison = diff(snapshot(reference.fields),
                              snapshot(substituted.fields))
            totals["differing_words"] += comparison["differing_words"]
            totals["total_words"] += comparison["total_words"]
            totals["comparisons"] += 1
            if not comparison["bit_identical"] and first_divergence is None:
                first_divergence = {"step": step,
                                    "per_array": comparison["per_array"]}
            if step % 8 == 0:
                log(f"[engine:{backend}] {spec['name']} step {step}/"
                    f"{ENGINE_STEPS} differing={totals['differing_words']}")
        entry["substitution_calls"] = dict(swap.counters)
        entry["substitution_leaks"] = swap.leaks
        entry["plan_runs"] = {name: p.runs for name, p in plan.plans.items()}
        entry["step_count"] = {"reference": reference.step_count,
                               "substituted": substituted.step_count}
        entry["totals"] = totals
        entry["first_divergence"] = first_divergence
        entry["census_final"] = census(substituted.fields)
        # SAMPLE-COUNT analogue: a leg that claims N steps must have taken N, and the
        # substitution must have been invoked EXACTLY once per step per side — not
        # twice, which is what a patch leaking onto the reference driver looks like.
        counted = (reference.step_count == ENGINE_STEPS
                   and substituted.step_count == ENGINE_STEPS
                   and swap.counters["update_H"] == ENGINE_STEPS
                   and swap.counters["update_E"] == ENGINE_STEPS
                   and swap.leaks == 0)
        entry["step_and_call_counts_agree"] = bool(counted)
        moved = diff(reference_initial, snapshot(reference.fields))
        entry["reference_state_moved"] = {
            "differing_words": moved["differing_words"],
            "total_words": moved["total_words"],
            "arrays_moved": sorted(moved["per_array"]),
            "note": "the ARRAY-PATH driver's own state, before the first step against "
                    "after the last. This family's update_H/update_E return before "
                    "their first statement, so every one of these words was moved by "
                    "the curls and the source — i.e. the step the substitution sat "
                    "inside really ran. Zero here makes the identity verdict vacuous."}
        entry["ok"] = bool(first_divergence is None and counted
                           and entry["census_final"]["nonzero_words"] > 0
                           and moved["differing_words"] > 0)
        entry["seconds"] = round(time.time() - started, 3)
        if spec["name"] == "engine_driven":
            driven_final = snapshot(substituted.fields)
        elif spec["name"] == "engine_source_free":
            free_final = snapshot(substituted.fields)
        cases.append(entry)
        log(f"[engine:{backend}] {entry['name']:22s} identical="
            f"{first_divergence is None} calls={entry['substitution_calls']} "
            f"steps={entry['step_count']} ok={entry['ok']} ({entry['seconds']} s)")
        results.setdefault("engine", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c.get("ok", False)) for c in cases),
            "cases": cases}
        save(results, out_path)

    # The source non-vacuity control: the driven run's state must DIFFER from the
    # source-free one at the same seed and step count. Without this the whole engine
    # leg could be comparing two identically-decaying seeded fields and calling it
    # coverage.
    if driven_final is not None and free_final is not None:
        source_effect = diff(free_final, driven_final)
        results["engine"][backend]["source_non_vacuity"] = {
            "differing_words": source_effect["differing_words"],
            "total_words": source_effect["total_words"],
            "ok": bool(source_effect["differing_words"] > 0),
            "note": ("same seed, same steps, one with an Ez gaussian source and one "
                     "without: the two states MUST differ, or the driven leg proved "
                     "nothing about a run that carries a source")}
    # THE LEG'S OWN NEGATIVE CONTROL. Everything above reports "identical", which is
    # what a harness that never ran also reports. So: force the null into a driver
    # whose layer IS active — the substitution the predicate exists to forbid, built
    # by calling NullConstitutivePlan directly and bypassing coverage — and require
    # the SAME loop to diverge. If this does not diverge, none of the passes above
    # mean anything.
    control_spec = {"name": "engine_negative_control_active_pml",
                    "cell": (1.2, 1.0, 1.4), "courant": COURANT_NP2,
                    "pml": 3, "source": ELECTRIC_SOURCE, "salt": 33}
    reference = build(control_spec, xp)
    forced = build(control_spec, xp)
    refusal = {side: family.null_constitutive_coverage(
        forced.fields, forced.pml, side) for side in ("H", "E")}
    plans = {"update_H": family.NullConstitutivePlan("H"),
             "update_E": family.NullConstitutivePlan("E")}
    swap = _Substitution(forced.fields, plans)
    control_divergence = None
    for step in range(1, ENGINE_STEPS + 1):
        reference.step()
        with swap:
            forced.step()
        comparison = diff(snapshot(reference.fields), snapshot(forced.fields))
        if not comparison["bit_identical"] and control_divergence is None:
            control_divergence = {"step": step,
                                  "differing_words": comparison["differing_words"]}
    results["engine"][backend]["negative_control"] = {
        "predicate_refused": {side: not verdict.covered
                              for side, verdict in refusal.items()},
        "refusal_reasons": {side: list(verdict.reasons)
                            for side, verdict in refusal.items()},
        "first_divergence": control_divergence,
        "substitution_calls": dict(swap.counters),
        "ok": bool(control_divergence is not None
                   and all(not v.covered for v in refusal.values())
                   and swap.counters["update_H"] == ENGINE_STEPS),
        "note": ("the null forced onto an ACTIVE-PML driver, past its own predicate. "
                 "It must diverge, and the predicate must have refused it. Without "
                 "this row the leg's three 'identical' verdicts are unfalsified.")}
    log(f"[engine:{backend}] negative control (active PML, null forced): "
        f"first divergence {control_divergence}, predicate refused "
        f"{ {s: not v.covered for s, v in refusal.items()} }")
    save(results, out_path)

    if driven_final is not None and free_final is not None:
        log(f"[engine:{backend}] source non-vacuity: "
            f"{source_effect['differing_words']} words differ between the driven and "
            f"source-free runs")
        save(results, out_path)
    return results["engine"][backend]


# ---------------------------------------------------------------------------
# Self-enforcement — the gate must fail, not merely record
# ---------------------------------------------------------------------------

def _check_identity_shaped_leg(label: str, block: Dict[str, Any],
                               expected_names: Sequence[str],
                               failures: List[str]) -> None:
    """The per-case contract BOTH byte-identity legs must meet.

    ``identity`` used to get all of this and ``breadth`` got two counters and a
    ``covered`` check — which is backwards, because breadth is where the module's one
    distinguishing claim lives (the null is exact on a fold / Dcyl / complex64 / Bloch
    / beta / BFAST grid that every kernel predicate in the package refuses). Measured
    on a breadth case detached from the shared fixture dicts, the old contract passed
    all four of: ``bit_identical`` False with 7 differing words; an all-zero census;
    ``cycles`` 1; ``plan_runs`` 0 — and it passed a breadth leg stripped to its one
    NON-distinguishing control row. One function now serves both legs, so a clause can
    no longer be present on one and absent on the other.
    """
    ran, ok = int(block.get("ran", 0)), int(block.get("ok", -1))
    if ran <= 0 or ok != ran:
        failures.append(f"{label} leg: {ok}/{ran} cases passed")
    seen = set()
    for case in block.get("cases", ()):
        name = case.get("name")
        seen.add(name)
        comparison = case.get("comparison") or {}
        if not comparison.get("bit_identical"):
            failures.append(f"{label} case {name} moved bytes: "
                            f"{comparison.get('differing_words')}")
        # THE COMPARISON MUST HAVE COMPARED SOMETHING. "0 differing words of 0" is the
        # identity leg's analogue of a flux case that compared two empty buffers, and
        # it passed this contract until now.
        if int(comparison.get("arrays_compared", 0)) <= 0:
            failures.append(f"{label} case {name} compared NO array: "
                            "arrays_compared == 0")
        if int(comparison.get("total_words", 0)) <= 0:
            failures.append(f"{label} case {name} compared NO word: total_words == 0")
        census = case.get("census_before", {})
        for key in ("nonzero_words", "signed_zero_words", "subnormal_words"):
            if int(census.get(key, 0)) <= 0:
                failures.append(f"{label} case {name} is VACUOUS: {key} == 0")
        cycles = int(case.get("cycles", 0))
        if cycles < 2:
            failures.append(f"{label} case {name} ran {cycles} cycle(s); one call "
                            "cannot separate a null from an idempotent write-back")
        # DISARMED: not "every counter that is present advanced" but "BOTH counters are
        # present and advanced". An empty mapping ran zero comparisons and passed —
        # "no plan at all" slipped through where "a plan that ran 0 times" failed.
        plan_runs = case.get("plan_runs") or {}
        if set(plan_runs) != {"update_H", "update_E"}:
            failures.append(f"{label} case {name} recorded plan_runs "
                            f"{sorted(plan_runs)}; DISARMED unless both null sub-steps "
                            "are named")
        for sub_step, runs in plan_runs.items():
            if int(runs) != cycles:
                failures.append(f"{label} case {name}: {sub_step} ran {runs} times "
                                f"for {cycles} cycles (DISARMED)")
        if case.get("requires_pml_object") and not case.get("pml_object_present"):
            failures.append(f"{label} case {name} claims an INERT LAYER and carries no "
                            "PML object: the family's partition test is unmeasured")
    missing = sorted(set(expected_names) - seen)
    if missing:
        failures.append(f"the {label} leg is missing its distinguishing cases: "
                        f"{missing}")


def validate_payload(payload: Dict[str, Any],
                     backend: str = "numpy",
                     require_all_legs: bool = True) -> Dict[str, Any]:
    """Raise unless every leg meets its release contract — and unless every leg is here.

    A gate that writes a divergent number and exits zero is a recorder. This is the
    contract, spelled positively, and the laptop suite mutation-tests IT — flipping
    one field of a passing payload must raise, or the contract is decorative.

    ``require_all_legs`` closes the hole that made the rest of it optional. Every check
    below is guarded on "is this leg present", so an artifact cut with ``--legs
    identity`` used to record ``contract: passed`` and ``all_ok: true`` with no
    controls leg, no mutation battery, no engine leg and no negative control. Nothing
    pinned the leg set the release claim rests on. It is False only for the suite's
    targeted per-leg tests, and :func:`main` always passes True, so a partial run
    cannot produce a green artifact.

    Pure Python over the artifact: no CuPy, no Triton, no engine objects, so it can
    be called on a saved ``gate.json`` from any host.
    """
    failures: List[str] = []

    present = [leg for leg in LEGS if payload.get(leg, {}).get(backend) is not None]
    absent = [leg for leg in LEGS if leg not in present]
    if require_all_legs and absent:
        failures.append(
            f"the artifact carries only {present} on backend {backend!r}: a release "
            f"claim needs every leg and {absent} are missing (a leg that is not in the "
            "artifact is not a leg that passed)")

    identity = payload.get("identity", {}).get(backend)
    if identity is not None:
        _check_identity_shaped_leg(
            "identity", identity, [case["name"] for case in IDENTITY_CASES], failures)
        courants = {bool(case.get("np2_courant"))
                    for case in identity.get("cases", ())}
        if courants != {True, False}:
            failures.append("the identity sweep must carry BOTH a non-power-of-two "
                            f"Courant and its power-of-two control; saw {courants}")

    breadth = payload.get("breadth", {}).get(backend)
    if breadth is not None:
        _check_identity_shaped_leg(
            "breadth", breadth, [case["name"] for case in BREADTH_CASES], failures)
        distinguishing = 0
        for case in breadth.get("cases", ()):
            if not all(case.get("covered", {}).values()):
                failures.append(f"breadth case {case.get('name')} was refused: the "
                                "module's whole claim is that these clauses are absent")
            if "admitted there too" not in str(case.get("refused_elsewhere", "")):
                distinguishing += 1
        if distinguishing <= 0:
            failures.append("the breadth leg carries no case that another predicate "
                            "refuses: stripped to its wall control it measures nothing")

    controls = payload.get("controls", {}).get(backend)
    if controls is not None:
        ran, ok = int(controls.get("ran", 0)), int(controls.get("ok", -1))
        if ran <= 0 or ok != ran:
            failures.append(f"controls leg: {ok}/{ran} cases passed")
        bit: Dict[str, int] = {"H": 0, "E": 0}
        for case in controls.get("cases", ()):
            for side in case.get("must_move", ()):
                moved = int(case.get("moved", {}).get(side, {})
                            .get("differing_words", 0))
                if moved <= 0:
                    failures.append(f"control {case.get('name')} moved no bytes on "
                                    f"{side}: the identity leg is unfalsified")
                else:
                    bit[side] = bit.get(side, 0) + 1
                if case.get("covered", {}).get(side):
                    failures.append(f"control {case.get('name')} was ADMITTED on {side}")
            if not case.get("polarization_held", True):
                failures.append(
                    f"control {case.get('name')}: update_E advanced a polarization "
                    f"({case.get('moved', {}).get('E', {}).get('polarization_words_moved')}"
                    " P/P_prev words moved) — stepping.py:969-973 says it consumes them")
        for side, count in bit.items():
            if count <= 0:
                failures.append(f"no control bit on the {side} side at all")
        # THE POLARIZATION PIN. The identity leg cannot carry it (a driven polarization
        # switches stores_E on and the predicate refuses), so it has to be measured
        # here or nowhere — and "nowhere" is what the gate's docstring used to claim
        # was "this path as well".
        if not any(int(case.get("polarization_arrays_compared", 0)) > 0
                   for case in controls.get("cases", ())):
            failures.append("no control compared a P/P_prev array: 'update_E consumes "
                            "a polarization and does not advance it' is unmeasured "
                            "everywhere in this artifact")

    overlap = payload.get("overlap", {}).get(backend)
    if overlap is not None:
        ran, ok = int(overlap.get("ran", 0)), int(overlap.get("ok", -1))
        if ran <= 0 or ok != ran:
            failures.append(f"overlap leg: {ok}/{ran} cases passed")
        other_admissions = 0
        declared = {entry["label"] for entry in SIBLING_CONSTITUTIVE_PREDICATES}
        for case in overlap.get("cases", ()):
            if not case.get("side_tables_agree"):
                failures.append("NULL_SIDES and coverage.CONSTITUTIVE_SIDES disagree")
            # EVERY sibling predicate must have answered. A predicate that raised, or
            # that never got imported, is a predicate whose partition claim this leg
            # did not measure — and both were previously recorded and never read.
            consulted = set(case.get("predicates_consulted", ()))
            if declared - consulted:
                failures.append(f"overlap case {case.get('name')} consulted "
                                f"{sorted(consulted)}; missing "
                                f"{sorted(declared - consulted)}")
            if case.get("raised"):
                failures.append(f"overlap case {case.get('name')}: predicates RAISED "
                                f"rather than refusing: {sorted(case['raised'])}")
            for side in ("H", "E"):
                labels = list(case.get("admitted_modulo_backend", {}).get(side, ()))
                if len(labels) > 1:
                    failures.append(f"OVERLAP on {case.get('name')}@{side}: {labels}")
                other_admissions += sum(1 for label in labels if label != "null")
        if other_admissions <= 0:
            failures.append("no OTHER constitutive predicate admitted anything modulo "
                            "the backend clause: the partition leg is vacuous — it "
                            "would pass on a package with one predicate in it")

    mutations = payload.get("mutations", {}).get(backend)
    if mutations is not None:
        rows = mutations.get("mutations", {})
        missing = sorted({spec["name"] for spec in MUTATIONS} - set(rows))
        if missing:
            failures.append(f"mutation rows are missing: {missing}")
        if not mutations.get("predicted_nulls"):
            failures.append("predicted nulls must be RECORDED with reasons, not omitted")
        for name, row in sorted(rows.items()):
            if not row.get("ok"):
                failures.append(f"mutation {name} not caught: {row.get('error')}")
            if int(row.get("sites", 0)) <= 0:
                failures.append(f"mutation {name} NEEDLE MISSED (0 sites edited)")
            if row.get("mutant_sha256") and not row.get("distinct_from_shipped"):
                failures.append(f"mutation {name} compiled a module identical to the "
                                "shipped one — the stale-artifact hazard is open")
            if int(row.get("configurations_evaluated", 0)) <= 0:
                failures.append(f"mutation {name} DISARMED (evaluated nothing)")
            if row.get("kind") == "byte_visible":
                if int(row.get("runs", 0)) <= 0:
                    failures.append(f"byte-visible mutation {name} ran no null plan")
                if not row.get("caught_on"):
                    failures.append(f"byte-visible mutation {name} moved no bytes")
        kinds = {row.get("kind") for row in rows.values()}
        if "byte_visible" not in kinds:
            failures.append("no byte-visible mutation in the battery: a verdict-only "
                            "battery proves nothing about the arrays")

    engine = payload.get("engine", {}).get(backend)
    if engine is not None:
        ran, ok = int(engine.get("ran", 0)), int(engine.get("ok", -1))
        if ran <= 0 or ok != ran:
            failures.append(f"engine leg: {ok}/{ran} cases passed")
        for case in engine.get("cases", ()):
            if case.get("first_divergence") is not None:
                failures.append(f"engine case {case.get('name')} diverged: "
                                f"{case.get('first_divergence')}")
            if not case.get("step_and_call_counts_agree"):
                failures.append(f"engine case {case.get('name')}: step/call counts "
                                f"disagree {case.get('substitution_calls')} "
                                f"{case.get('step_count')}")
            # THE RAW NUMBERS, not the gate's own boolean. The leg's whole discipline
            # is "one substitution call per step per side, no patch leak onto the
            # reference driver" — the defect its docstring records is 48 calls for 24
            # steps — and the contract used to read a summary flag while a payload with
            # 0 calls and 0 steps passed.
            budget = int(case.get("steps", ENGINE_STEPS))
            if budget != ENGINE_STEPS:
                failures.append(f"engine case {case.get('name')} claims {budget} steps; "
                                f"the stated budget is {ENGINE_STEPS}")
            for sub_step in ("update_H", "update_E"):
                calls = int((case.get("substitution_calls") or {}).get(sub_step, -1))
                if calls != budget:
                    failures.append(
                        f"engine case {case.get('name')}: {sub_step} was substituted "
                        f"{calls} times for {budget} steps — twice is a patch leaking "
                        "onto the reference driver, zero is a patch that never took")
            for which in ("reference", "substituted"):
                steps = int((case.get("step_count") or {}).get(which, -1))
                if steps != budget:
                    failures.append(f"engine case {case.get('name')}: the {which} "
                                    f"driver took {steps} steps, not {budget}")
            if int(case.get("substitution_leaks", 0)) != 0:
                failures.append(f"engine case {case.get('name')}: the substitution "
                                f"leaked onto {case.get('substitution_leaks')} other "
                                "driver call(s)")
            # THE STEP MUST HAVE DONE WORK. For a null family this is the difference
            # between a byte claim and the emptiest possible pass: two drivers that
            # both moved nothing agree perfectly. Absent counts as a failure rather
            # than as "not checked" — a case that stopped recording this number is a
            # case whose identity verdict is no longer falsifiable.
            moved = case.get("reference_state_moved")
            if not moved or int(moved.get("differing_words", 0)) <= 0:
                failures.append(
                    f"engine case {case.get('name')}: the ARRAY-PATH driver's own "
                    "state did not move across the whole run, so 'the substituted "
                    "driver matched it' compares two frozen states and certifies "
                    f"nothing ({moved})")
        source = engine.get("source_non_vacuity")
        if not source or not source.get("ok"):
            failures.append("the source-free control did not differ from the driven "
                            "run: the driven case proves nothing")
        control = engine.get("negative_control")
        if not control or not control.get("ok"):
            failures.append("the negative control did not diverge: every 'identical' "
                            "verdict in this artifact is unfalsified")
        elif control.get("first_divergence") is None:
            failures.append("negative control recorded no divergence step")
        elif not all((control.get("predicate_refused") or {}).values()):
            failures.append("the predicate ADMITTED the negative control")

    policy = payload.get("policy")
    if policy is None:
        failures.append("no subnormal-policy stamp: an artifact that does not say "
                        "which policy it was cut under cannot license anything")

    if failures:
        raise AssertionError("; ".join(failures))
    return {"status": "passed", "legs_validated": sorted(present),
            "legs_absent": sorted(absent),
            "all_legs_required": bool(require_all_legs)}


# ---------------------------------------------------------------------------
# Provenance and main
# ---------------------------------------------------------------------------

def provenance(xp: Any, backend: str) -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    record: Dict[str, Any] = {
        "family": "no-PML constitutive (update_H / update_E with an inert layer)",
        "ships_a_kernel": False,
        "backend": backend,
        "python": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": getattr(cp, "__version__", None),
        "triton_importable": importlib.util.find_spec("triton") is not None,
        "module_sha256": sha256_file(SHIPPED_SOURCE),
        "gate_sha256": sha256_file(os.path.abspath(__file__)),
        "step_budget": {
            "this_family": "unbounded — the covered arm performs no floating-point "
                           "operation, so it has no divergence mechanism of its own",
            "composed_whole_step": "bounded by the curl's; no_pml.GATE_VERDICT "
                                   "records step_budget 66 for the no-PML curl and "
                                   "this gate does not raise it",
        },
        # DERIVED FROM THE RUN, not asserted about it. This was a frozen string —
        # "NOT CLAIMED ON A DEVICE. No device leg has been run." — emitted by every
        # artifact including the ones a device leg wrote, so the first green CuPy run
        # produced a record denying its own existence. What a null family can claim is
        # narrow and is spelled out rather than summarised.
        "byte_identity_claim": (
            "HOST ONLY. This artifact was cut on NumPy: it certifies that the "
            "PLANNER-LEVEL substitution is byte-null against the array path on this "
            "host, and says nothing about device arrays. NO KERNEL IS CERTIFIED HERE "
            "because the family ships none."
            if backend == "numpy" else
            "ON DEVICE ARRAYS, and stated narrowly. What is certified is that "
            "substituting the null plan for stepping.update_H / update_E leaves every "
            "live array bit-identical to the array path across the configurations the "
            "predicate admits — measured by uint32 comparison after every complete "
            "driver step, with the plan's run counter proving the substitution was "
            "installed and the curls proving the step still moved state. NO KERNEL IS "
            "CERTIFIED HERE because the family ships none; the arithmetic in the "
            "surrounding step remains the array path's."),
    }
    return record


LEGS = ("identity", "controls", "breadth", "overlap", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/no_pml_constitutive/gate.json")
    parser.add_argument("--legs", default=",".join(LEGS))
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    args = parser.parse_args(argv)

    if args.backend == "cupy":
        if cp is None:
            log("cupy is not importable; refusing to fabricate a device row")
            return 2
        xp: Any = cp
    else:
        xp = np

    requested = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    unknown = [leg for leg in requested if leg not in LEGS]
    if unknown:
        log(f"unknown legs {unknown}; known: {LEGS}")
        return 2

    # The ship policy, installed BEFORE any CuPy work (a clean no-op without CuPy;
    # raises at startup on a cache dir that could mix policies). This family compiles
    # nothing, so the strip has nothing to strip — that is the POINT of stamping it:
    # "zero NVRTC compiles" is the expected reading here and is recorded as such,
    # where in every kernel family it would be a refusal to license.
    install_ftz_strip()

    out_path = os.path.abspath(args.out)
    results: Dict[str, Any] = {"provenance": provenance(xp, args.backend),
                               "policy": policy_stamp(args.backend),
                               "legs_requested": requested}
    results["policy"]["license_reasons"] = list(ftz_strip_license_reasons())
    results["policy"]["inert_for_this_family"] = (
        "PREDICTED NULL FOR THE FAMILY'S OWN PRODUCT, and the prediction is about the "
        "product rather than about the process: no_pml_constitutive ships no kernel "
        "and launches nothing, so NOT ONE NVRTC option tuple originates in the thing "
        "being certified, and the subnormal question the strip exists to settle "
        "(CuPy flushes float32 subnormals, Triton keeps them) has no surface in it — "
        "nobody computes on a subnormal. The seeds carry dense subnormals anyway and "
        "the census proves they were present and unmoved, which is the measurement "
        "that replaces the policy evidence.\n\n"
        "THE HARNESS AROUND IT IS A DIFFERENT MATTER, and this file used to conflate "
        "the two. Seeding, the uint32 comparison and above all the CONTROLS leg's "
        "real active-PML arithmetic are ordinary CuPy elementwise work, and CuPy "
        "compiles that through the same NVRTC seam. Measured on the GPU host 2026-08-13: "
        "48 .cubin files landed in this run's private cache. So the counters below "
        "are expected to be NON-ZERO on the cupy backend, and a zero there is a "
        "broken counter rather than a quiet family — which is exactly what the "
        "previous artifacts recorded, because the stamp was taken before the first "
        "leg ran and never refreshed. See 'measured_after_legs'.")
    save(results, out_path)
    started = time.time()

    runners = {
        "identity": run_identity,
        "controls": run_controls,
        "breadth": run_breadth,
        "overlap": run_overlap,
        "mutations": run_mutations,
        "engine": run_engine,
    }
    for leg in requested:
        log(f"=== leg {leg} ({args.backend}) ===")
        runners[leg](results, out_path, xp, args.backend)

    # RE-STAMP THE POLICY, because the startup stamp cannot have measured anything.
    # ``policy_stamp`` reads live counters off the installed strip, and it was called
    # BEFORE the first leg — so ``nvrtc_calls`` and ``ftz_removed`` were snapshotted at
    # zero and frozen there. The artifact then read as "no compile happened on any
    # backend", which the family's own docstring cited as confirmation that it compiles
    # nothing. It was not a measurement: it could not have been anything but zero.
    # Measured against the private cache dir of the very run that printed it, 48 CuPy
    # .cubin files existed beside a stamp claiming zero NVRTC calls. Both numbers are
    # kept, labelled, so the artifact shows the difference rather than replacing one
    # unmeasured number with another.
    results["policy"]["at_startup"] = {
        "nvrtc_calls": results["policy"].get("nvrtc_calls"),
        "ftz_removed": results["policy"].get("ftz_removed"),
        "note": "taken before the first leg; structurally zero, measures nothing"}
    results["policy"]["measured_after_legs"] = policy_stamp(args.backend)
    results["policy"]["license_reasons"] = list(ftz_strip_license_reasons())

    summary: Dict[str, Any] = {}
    for leg in requested:
        block = results.get(leg, {}).get(args.backend, {})
        if leg == "mutations":
            entries = block.get("mutations", {})
            summary[leg] = {"ran": len(entries),
                            "ok": sum(int(e.get("ok", False))
                                      for e in entries.values())}
        else:
            summary[leg] = {"ran": block.get("ran", 0), "ok": block.get("ok", 0)}
    all_ok = all(value["ran"] > 0 and value["ok"] == value["ran"]
                 for value in summary.values())
    engine_block = results.get("engine", {}).get(args.backend, {})
    for key in ("source_non_vacuity", "negative_control"):
        extra = engine_block.get(key)
        if extra is not None:
            summary.setdefault("engine_extras", {})[key] = bool(extra["ok"])
            all_ok = all_ok and bool(extra["ok"])
    results["summary"] = summary
    # SELF-ENFORCEMENT, not self-report: the contract runs over the artifact this run
    # just wrote, and a leg that passed its own per-case check but violates the
    # release contract (a vacuous partition, a missing Courant control, a byte-visible
    # mutation that never ran a plan) fails HERE.
    try:
        results["contract"] = validate_payload(results, args.backend)
    except AssertionError as exc:
        results["contract"] = {"status": "failed", "failures": str(exc)}
        all_ok = False
        log(f"[contract] FAILED: {exc}")
    results["all_ok"] = bool(all_ok)
    results["seconds"] = round(time.time() - started, 3)
    save(results, out_path)
    log(f"=== summary: {json.dumps(summary, sort_keys=True)} all_ok={all_ok} "
        f"({results['seconds']} s) -> {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
