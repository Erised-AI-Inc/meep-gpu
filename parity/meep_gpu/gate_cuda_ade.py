"""Byte-identity gate for the hand-CUDA ADE polarization step (``update_P``).

The family under test is ``meep_gpu/cuda_kernels/ade_kernels.py`` -- two device
strings, ``update_P_pml_real`` (a graded sigma volume) and
``update_P_pml_real_uniform`` (a uniform scalar sigma), behind one
launcher. Nothing dispatches them; ``CERTIFIED_KERNELS`` is EMPTY and both names
sit in ``UNCERTIFIED_KERNELS``. This gate is what would move them.

=============================================================================
WHY THIS FAMILY GETS ITS OWN GATE FILE
=============================================================================

``gate_cuda_offdiag.py`` is the structural reference and this file follows it
leg for leg -- the same byte comparator, the same operand census, the same
subnormal-band draw, the same policy install, the same NVRTC binary observer,
the same device stamp, all by import from ``probe_fused_kernel_bit_identity``.
What it cannot share is the SUBJECT. Every other certified hand-CUDA family
writes a field array from coefficient tables indexed by cell coordinate; this
one writes a THIRD BUFFER from two others and rotates three names afterwards.
The thing under test is therefore a STATE MACHINE, not a stencil, and three of
this file's legs exist only because of that:

* the multi-step leg is LOAD-BEARING here rather than confirmatory. The rotation
  (dispersion.py:689-691) is invisible to a single launch by construction;
* the stale-pointer pair is the discriminator that proves it. The SAME planted
  host defect must be a NULL at one launch with one driven component and a CATCH
  at sixty -- otherwise "the multi-step leg catches things" is a statement about
  the budget rather than about the rotation;
* the WRONG-DRIVE control, which has no analogue in a stencil family at all.

WHICH CALLER EACH LEG DRIVES, because it is not one caller. The two primary legs
(single-launch and multi-step) drive the SHIPPED ``update_P_fused_pml_real``,
aliasing check and all. The mutation legs drive ``run_shipped``, a transcription
of it in this file -- because a HOST defect lives in the caller and there is no
other way to plant one. Legs ``p0`` and ``p1`` run the same product through the
real launcher and through the transcription, so the two are welded by measurement
rather than by inspection.

=============================================================================
THE ORACLE IS THE ARRAY PATH ITSELF
=============================================================================

``stepping.update_P`` on real ``Grid``/``Fields``/``PML``/``PolarizationState``
objects -- i.e. ``PolarizationState.update`` (dispersion.py:658-691), reached
exactly as the engine reaches it, with ``fields.drive_field`` bound as the drive
exactly as ``stepping.update_P`` binds it (stepping.py:1428).

TWO FIXTURES, NOT ONE FIXTURE AND A RESTORE. The sibling gates freeze one state,
run the oracle, restore, and run the kernel. That does not work here: the sub-step
PERMUTES WHICH ARRAY OBJECT SITS IN WHICH SLOT, so "restore" would have to undo a
rotation as well as a write, and a restore that got the permutation wrong would
silently compare two different arrangements. Two fixtures built from the same seed
avoid the question -- and the gate does not assume they came out identical, it
MEASURES it: every buffer of both fixtures is compared word for word before either
side steps, and a case whose fixtures did not start identical is refused rather
than scored.

Never ``allclose``: ``-0.0 == 0.0`` and ``NaN != NaN`` both lie. The relaxed
comparator appears exactly once, in a harness leg that COUNTS how many cases of a
real defect walk through it.

=============================================================================
A GATE THAT CANNOT FAIL CERTIFIES NOTHING -- THE FOUR FLOORS
=============================================================================

Recorded per case, so non-vacuity is a property of the RECORD rather than of a
reader's trust in the generator. A case failing any of them is REFUSED, not
passed:

* ``oracle_moved`` -- the array path changed at least one output word. Zero-init
  is a fixed point of this recurrence, so a fixture that moved nothing would pass
  against a kernel that did nothing.
* ``drive_term_is_live`` -- ``max |c_drive * (sigma * w)|`` over the frozen state
  is nonzero. Where the drive term vanishes, ``drop_sigma`` and the WHOLE
  wrong-drive control are indistinguishable from the shipped kernel.
* ``history_term_is_live`` -- ``max |c_prev * P_prev|`` is nonzero. Where the
  history term vanishes the recurrence is first order and ``drop_previous_history``
  and ``swap_p_and_q`` cannot be seen.
* ``oracle_all_finite`` -- an unstable susceptibility that overflows to inf makes
  every word agree for a reason that has nothing to do with the kernel.

Plus ``launches == expected_launches`` on every certifying case: BYTES ALONE
CANNOT PROVE THE KERNEL RAN. Two fixtures seeded identically and a kernel that
never launched differ only in that the oracle moved -- which is why the launch
counter is asserted per case AND why leg ``x1`` runs the whole comparison with the
launch suppressed and REQUIRES divergence.

=============================================================================
THE MUTATIONS, AND THE DISARM CHECK
=============================================================================

Source defects are planted by assigning over the module-level device strings --
``_get_kernel`` rebuilds its code map per call and the compile memo is keyed
THROUGH the source, so a mutated body is a miss and reaches NVRTC. Every leg
records how many compiles came from the mutated bytes, because a leg reporting a
pass for a mutation it never applied is worse than no leg.

HOST defects are planted by driving the launcher's own parts (``_get_kernel`` and
``_launch``) from this file, which is the only way to plant a defect in a CALLER.
The rotation, the pointer freshness, the drive binding and the sigma
specialization are all the caller's to get wrong, and none of them is reachable
by editing the device text.

Every leg is followed by a DISARM CHECK: the module's two device strings are
compared against the pristine bytes and the leg records whether they were
restored. The gate then ends with a full unmutated re-run of the mutation
product, so "the arming was undone" is a measurement rather than a convention.

Run (the GPU host, one verified-empty device; the cache dir MUST carry the policy
token because CuPy's cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=N CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_ade.py --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA; the kernel side becomes a NumPy evaluator over the same device
text, which compiles nothing and therefore certifies nothing)::

    python -u gate_cuda_ade.py --backend numpy --product reduced \\
        --out /tmp/ade_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only: no throughput or timing claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
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
except ImportError:  # laptop: the harness legs still run against the evaluator
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402

log = probe.log
save = probe.save
bit_compare = probe.bit_compare
combine = probe.combine
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts
to_host = probe.to_host

SEED = 20260819

COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: Consecutive ``update_P`` calls in the multi-step leg. 60 is the budget the two
#: certified hand-CUDA records are cut at, kept so the three families' claims stay
#: commensurable. "Identical for N steps" is a claim about N, and for THIS family
#: it is the claim that carries the rotation.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence driven. Held fixed, the drive term
#: stops changing and a 60-step leg becomes a slow single-step leg. Applied to the
#: SAME bytes on both fixtures, so it cancels out of the comparison; it is not
#: exactly representable in float32, which is deliberate -- an exact scale would
#: leave the mantissas untouched.
_ADVANCE = np.float32(0.97)

#: Resolution 8 rather than 1. dt = courant / resolution enters the recurrence
#: coefficients through ``omega0dtsqr = (2*pi*f*dt)**2``, and at resolution 1 a
#: susceptibility with f ~ 1 is violently unstable: P overflows to inf within the
#: multi-step budget and every word then agrees for a reason that has nothing to do
#: with the kernel. ``oracle_all_finite`` refuses such a case; this keeps it from
#: arising.
RESOLUTION = 8.0

#: (target shape, label). The MIDDLE one is the point: 13*17*11 = 2431 cells is not
#: a multiple of the 256-lane block, so the tail guard ``if (idx >= n_elem) return;``
#: decides real threads there. 8*8*8 = 512 and 1*16*16 = 256 are exact multiples and
#: cannot exercise it.
SHAPES: Tuple[Tuple[Tuple[int, int, int], str], ...] = (
    ((8, 8, 8), "8x8x8"),
    ((13, 17, 11), "13x17x11"),
    ((1, 16, 16), "1x16x16"),
)

#: Boundary conditions are a CONTROL on this sub-step, not a coverage axis, and
#: saying so is half of what the axis is for. ``stepping.update_P`` reads no
#: neighbour (stepping.py:1398-1403), so a wall cannot reach the arithmetic; the
#: axis is swept anyway because the predicate ADMITS these kinds and a claim that
#: they are inert is worth a measurement.
BOUNDARY_TRIPLES: Tuple[Tuple[str, str, str], ...] = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "periodic"),
)

#: The two recurrence kinds ``COVERED_SUSCEPTIBILITY_KINDS`` names. The Drude flag
#: zeroes the restoring force in ``c_now`` ALONE (dispersion.py:643-655), so the two
#: kinds are two different coefficient triples through one kernel.
KINDS: Tuple[str, ...] = (LORENTZIAN, DRUDE)

#: The compile-time specialization. Both variants must be swept or half the family
#: is uncertified: they are two device strings, two kernel names and two NVRTC
#: compiles.
SIGMA_FORMS: Tuple[str, ...] = ("volume", "uniform")

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: (label, state count, driven components). The component count is what makes the
#: PER-COMPONENT rotation observable: with one driven component the retired history
#: buffer is handed straight back to the same component next step, and with three it
#: becomes the NEXT component's scratch inside a single call -- which is the
#: arrangement ``h_stale_pointers`` is wrong about.
LAYOUTS: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("one_state_one_component", 1, ("Ex",)),
    ("one_state_two_components", 1, ("Ex", "Ez")),
    ("two_states_three_components", 2, ("Ex", "Ey", "Ez")),
)

#: dt = courant / resolution. Two values so the coefficient triple is not one
#: number the whole gate long; unlike the stencil families there is NO exact/inexact
#: story here, because this kernel carries no dt and no dx -- the coefficients
#: arrive as baked float32 scalars.
COURANTS: Tuple[float, ...] = (0.5, 0.35)

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: A mirror fold is THE clause that separates this predicate from its two siblings
#: (``coverage.ADE_MIRROR_ADMISSION``: 5 of the corpus's 15 ``update_P`` rows). It
#: is swept as its own case rather than as an axis, because a fold halves the stored
#: extent and so is not orthogonal to the shape axis.
FOLD_CASES: Tuple[Dict[str, Any], ...] = (
    {"shape": (16, 16, 8), "label": "foldX_16x16x8", "symmetry": ("X",)},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    ``coverage._backend`` asks the grid's array module for its NAME first, and that
    is the one thing about the device library a laptop cannot supply. Everything
    else the fixture exercises -- dtype, shape, contiguity, base addresses, the
    polarization buffers, the drive identity -- is a real object either way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def config_viable(shape: Sequence[int],
                  boundaries: Sequence[str]) -> Optional[str]:
    """Why this (shape, boundary) pair cannot be built, or None.

    ``Grid`` REFUSES a metallic axis holding exactly one cell and says why at
    length. Filtering here keeps that refusal out of the case list instead of
    turning it into a leg failure that says nothing about the kernel.
    """
    for axis, extent in enumerate(shape):
        if extent <= 1 and boundaries[axis] == "metallic":
            return (f"axis {axis} holds one cell and cannot be metallic; Grid "
                    f"refuses it and MEEP has no such run")
    return None


def _susceptibility(kind: str, index: int) -> Susceptibility:
    """One term. Frequencies and dampings differ per state so two states in one
    fixture do not share a coefficient triple -- otherwise a defect that confused
    the two states' coefficients would be invisible."""
    return Susceptibility(frequency=0.85 + 0.31 * index,
                          gamma=0.09 + 0.04 * index,
                          kind=kind)


def build(xp, spec: Dict[str, Any], seed: int):
    """One complete ``(fields, layer, grid)`` triple, seeded deterministically.

    THE DRIVE AND THE STORED E ARE SEEDED APART unless the case asks otherwise.
    Inside a PML they genuinely differ -- the stored E has had the absorbing
    accumulation applied and ``f_w`` has not (fields.py:1140-1163) -- and a fixture
    that seeded them alike would make the wrong-drive control a null for a reason
    belonging to the fixture. ``drive_equals_stored_E`` asks for exactly that null,
    on purpose, as the control's discriminator.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(spec["shape"])
    cell = tuple(float(n) / RESOLUTION for n in shape)
    grid = Grid(resolution=RESOLUTION, cell_size=cell,
                boundaries=tuple(spec["boundaries"]),
                symmetry=tuple(spec.get("symmetry", ())),
                xp=xp, courant=spec["courant"])
    stored = tuple(int(n) for n in grid.shape)

    # A mirror plane sits on the LOW face, which PML refuses to absorb into: cell 0
    # is a boundary condition, not an absorber (pml.py:405-412). Ask for the high
    # face alone on a folded axis.
    thickness = []
    for axis in range(3):
        if stored[axis] < 6:
            thickness.append((0, 0))
        elif grid.is_mirrored(axis):
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    layer = PML(grid=grid, thickness=tuple(thickness))
    fields = Fields(grid=grid)
    fields.enable_pml_storage()

    driven = tuple(spec["driven"])
    for index in range(int(spec["states"])):
        sigma: Dict[str, Any] = {}
        for position, component in enumerate(COMPONENTS):
            if component not in driven:
                # MEEP's trivial_sigma: no P is allocated and the component is not
                # driven (dispersion.py:610-620). This is how a layout names the
                # components it carries.
                sigma[component] = 0.0
            elif spec["sigma_form"] == "volume":
                # THE MATERIAL STAYS NORMAL under the band class, deliberately.
                # Driving sigma into the band too would underflow every product and
                # the leg would measure the fixture rather than the policy's reach.
                sigma[component] = xp.asarray(
                    rng.uniform(0.15, 0.85, size=stored).astype(np.float32))
            else:
                sigma[component] = float(
                    np.float32(0.19 + 0.07 * index + 0.03 * position))
        state = PolarizationState(_susceptibility(spec["kind"], index),
                                  sigma, grid, np.float32)
        fields.polarizations.append(state)

    value_class = spec["value_class"]
    names = ["f_w_" + c for c in COMPONENTS]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.append(f"P{index}_{component}")
            names.append(f"Q{index}_{component}")
        names.append(f"S{index}")
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=stored).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(names, stored, rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")

    for component in COMPONENTS:
        getattr(fields, "f_w_" + component)[...] = xp.asarray(
            np.ascontiguousarray(host["f_w_" + component]))
        if spec.get("drive_equals_stored_E"):
            getattr(fields, component)[...] = getattr(fields, "f_w_" + component)
        else:
            getattr(fields, component)[...] = xp.asarray(
                np.ascontiguousarray(
                    rng.uniform(0.65, 0.95, size=stored).astype(np.float32)))
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            state.P[component][...] = xp.asarray(
                np.ascontiguousarray(host[f"P{index}_{component}"]))
            state.P_prev[component][...] = xp.asarray(
                np.ascontiguousarray(host[f"Q{index}_{component}"]))
        # THE SCRATCH IS SEEDED NONZERO. It is this launch's OUTPUT and it starts as
        # whatever the previous rotation retired; leaving it at zero would let a
        # kernel that failed to write some cells agree with an oracle that also
        # wrote nothing there.
        state._scratch[...] = xp.asarray(np.ascontiguousarray(host[f"S{index}"]))
    return fields, layer, grid, host


def state_slots(fields) -> List[Tuple[str, Any]]:
    """Every array this sub-step may write, named by SLOT rather than by identity.

    Slot names are the whole point: the rotation moves array OBJECTS between
    ``P``, ``P_prev`` and ``_scratch``, so a launcher that rotated wrongly would
    hold the right numbers in the wrong slots. Comparing by slot catches that;
    comparing by object identity could not.
    """
    out: List[Tuple[str, Any]] = []
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            out.append((f"P[{index}][{component}]", state.P[component]))
            out.append((f"P_prev[{index}][{component}]", state.P_prev[component]))
        out.append((f"_scratch[{index}]", state._scratch))
    return out


def slot_hosts(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy() for name, array in state_slots(fields)}


def fixture_hosts(fields) -> Dict[str, np.ndarray]:
    """Every array either side READS or WRITES -- the equality the two fixtures
    must start at. The drive and the stored E are in it because a case whose two
    fixtures disagreed about the drive would compare two different problems."""
    out = slot_hosts(fields)
    for component in COMPONENTS:
        out["f_w_" + component] = to_host(getattr(fields, "f_w_" + component)).copy()
        out["stored_" + component] = to_host(getattr(fields, component)).copy()
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            sigma = state.sigma[component]
            if coverage.ade_sigma_is_volume(state, component):
                out[f"sigma[{index}][{component}]"] = to_host(sigma).copy()
    return out


def words_differ(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(
        np.ascontiguousarray(left).ravel().view(np.uint32)
        != np.ascontiguousarray(right).ravel().view(np.uint32)))


def advance_drive(fields) -> None:
    """Move the drive between steps, identically on both fixtures."""
    for component in COMPONENTS:
        getattr(fields, "f_w_" + component)[...] = (
            getattr(fields, "f_w_" + component) * _ADVANCE)
        getattr(fields, component)[...] = getattr(fields, component) * _ADVANCE


def term_magnitudes(fields) -> Dict[str, float]:
    """The two floors, taken on the FROZEN state in float64 on the host.

    ``drive_term`` is ``c_drive * (sigma * w)`` and ``history_term`` is
    ``c_prev * P_prev``. A case where either is identically zero cannot distinguish
    the defects that live in it, and is refused by name rather than counted.
    """
    drive_max = 0.0
    history_max = 0.0
    for state in fields.polarizations:
        c_now, c_prev, c_drive = (float(v) for v in state._coefficients)
        for component in state.driven():
            w = to_host(fields.drive_field(component)).astype(np.float64)
            sigma = state.sigma[component]
            sigma_host = (to_host(sigma).astype(np.float64)
                          if coverage.ade_sigma_is_volume(state, component)
                          else float(sigma))
            drive_max = max(drive_max,
                            float(np.max(np.abs(c_drive * (sigma_host * w)))))
            history = to_host(state.P_prev[component]).astype(np.float64)
            history_max = max(history_max,
                              float(np.max(np.abs(c_prev * history))))
    return {"drive_term_max_abs": drive_max,
            "history_term_max_abs": history_max}


def expected_launch_count(fields, steps: int) -> int:
    return steps * sum(len(state.driven()) for state in fields.polarizations)


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------
#
# Both drive the SAME device text. The CuPy backend hands it to NVRTC; the NumPy
# backend executes it. The second compiles nothing and certifies nothing -- it is
# there so the whole harness (case product, floors, arming, catch accounting) can
# be shown to FAIL where it must on a laptop, before a device slot is spent.

_LOAD = re.compile(r"^float (\w+) = (\w+)\[idx\];$")
_SCALAR_LOAD = re.compile(r"^float (\w+) = (\w+);$")
_STORE = re.compile(r"^(\w+)\[idx\] = (.+);$")

#: Template text the evaluator neither executes nor needs to: the signature, the
#: thread index and the tail guard. The guard is scaffolding HERE and not in
#: general -- the evaluator is whole-array, so it has no out-of-range lane to
#: return from; on device the compiler executes that line and a defect in it is an
#: ordinary catch.
_SCAFFOLDING: Tuple[str, ...] = (
    'extern "C" __global__', "float* __restrict__", "const float*",
    "float c_now, float c_prev, float c_drive,", "int n_elem",
    "int idx = blockIdx.x", "if (idx >= n_elem) return;",
    "float sigma,", ") {", "}", "{",
)


def _float32_literals(expression: str) -> str:
    """``0.0f`` -> ``0.0``. C float literals are not Python."""
    return re.sub(r"(\d)f\b", r"\1", expression)


def body_lines(text: str) -> List[str]:
    body = text[text.index('extern "C"'):]
    return [line.strip() for line in body.split("\n")
            if line.strip() and not line.strip().startswith("//")]


def evaluator_blind_lines(text: str) -> List[str]:
    """Lines of a device body the NumPy evaluator neither executes nor recognises.

    THE HAZARD THIS EXISTS FOR is the one the sibling gate measured: a mutation
    that rewrites a statement into a form the evaluator's grammar does not match is
    SILENTLY NOT APPLIED, and a verdict scored off such a run is a statement about
    the parser rather than about the kernel. A leg whose mutated text leaves a line
    here is recorded as NOT MEASURABLE on this backend instead of being scored. On
    device every line is executed by the compiler and the question does not arise,
    which is exactly why the two backends' catch tables differ and both are
    reported.
    """
    blind = []
    for line in body_lines(text):
        if any(token in line for token in _SCAFFOLDING):
            continue
        if _LOAD.match(line) or _SCALAR_LOAD.match(line) or _STORE.match(line):
            continue
        blind.append(line)
    return blind


def evaluate_device_source(text: str, arrays: Dict[str, Any],
                           scalars: Dict[str, Any]) -> None:
    """Execute one device body in float32 NumPy, from the text itself.

    Loads and the single store are read off the source, so a mutation that swaps
    two load statements or reassociates the stored expression is really applied
    here rather than assumed. Everything is float32 on both sides of every
    operator; NumPy 2's weak promotion keeps a bare Python literal from widening
    the result, and the dtype is asserted rather than trusted.
    """
    namespace: Dict[str, Any] = dict(scalars)
    store: Optional[Tuple[str, str]] = None
    for line in body_lines(text):
        if any(token in line for token in _SCAFFOLDING):
            continue
        match = _LOAD.match(line)
        if match:
            namespace[match.group(1)] = arrays[match.group(2)]
            continue
        match = _SCALAR_LOAD.match(line)
        if match:
            namespace[match.group(1)] = scalars[match.group(2)]
            continue
        match = _STORE.match(line)
        if match:
            store = (match.group(1), match.group(2))
            continue
        raise ValueError(f"the evaluator has no grammar for {line!r}")
    if store is None:
        raise ValueError("no store statement in the device body")
    target, expression = store
    value = eval(_float32_literals(expression), {"__builtins__": {}}, namespace)  # noqa: S307
    value = np.asarray(value)
    if value.dtype != np.float32:
        raise ValueError(f"the evaluator produced {value.dtype}, not float32; a "
                         f"widened intermediate is a different number")
    arrays[target][...] = value


class EvaluatorBackend:
    """The kernel side, executed in NumPy from the device text. Compiles nothing.

    Answers ``certifies = False``, and every summary carries that through, so no
    run on this backend can be mistaken for a certification.
    """

    name = "numpy"
    certifies = False

    def __init__(self, pristine: Dict[bool, str]):
        self.pristine = dict(pristine)
        self.sources: Dict[bool, str] = dict(pristine)

    def set_guard(self, guard: Sequence[str]) -> None:
        """No compiler, so no guard. Recorded rather than silently ignored."""

    def source_for(self, volume: bool) -> str:
        return self.sources[bool(volume)]

    def set_sources(self, sources: Dict[bool, str]) -> None:
        self.sources = dict(sources)

    def disarm(self) -> bool:
        self.sources = dict(self.pristine)
        return True

    def is_pristine(self) -> bool:
        return self.sources == self.pristine

    def launch_one(self, scratch, p, q, sigma, w, coefficients,
                   volume: bool) -> None:
        c_now, c_prev, c_drive = coefficients
        arrays = {"p_out": scratch, "p_now": p, "p_prev": q, "drive": w}
        scalars = {"c_now": np.float32(c_now), "c_prev": np.float32(c_prev),
                   "c_drive": np.float32(c_drive)}
        if volume:
            arrays["sigma"] = sigma
        else:
            scalars["sigma"] = np.float32(sigma)
        evaluate_device_source(self.source_for(volume), arrays, scalars)

    def synchronize(self) -> None:
        pass

    def compile_log(self) -> List[Dict[str, Any]]:
        return []

    def clear(self) -> int:
        return 0


class KernelBackend:
    """The shipped CuPy launcher, driven through the module's own doors.

    THE GUARD AND THE MUTATION BOTH REACH NVRTC THROUGH THE SOURCE-KEYED MEMO.
    ``compile_cache.kernel_cache_key`` keys on ``(name, is_complex, options,
    policy, source)``, so overriding ``_COMPILE_OPTIONS`` or assigning over a
    device string is a MISS and the compiler sees the bytes this leg intends. The
    memo is dropped at both ends anyway, because a leg that measured a guard it
    never applied is a failure family this project has three measured instances of.
    """

    name = "cupy"
    certifies = True

    def __init__(self, kernels, pristine: Dict[bool, str]):
        self.kernels = kernels
        self.pristine = dict(pristine)

    #: Module attribute holding each variant's device string.
    ATTRIBUTE = {True: "_update_P_pml_real_kernel_code",
                 False: "_update_P_pml_real_uniform_kernel_code"}

    def set_guard(self, guard: Sequence[str]) -> None:
        """Install the option tuple, and drop the memo only when it CHANGED.

        Dropping it on every case would recompile both variants several hundred
        times and turn a correctness gate into a compiler benchmark. It is safe to
        skip because ``kernel_cache_key`` already carries ``options``: an entry
        compiled under the other tuple can never be served for this one. The clear
        stays on a real change as belt-and-braces, in the same spirit as the clear
        at both ends of every mutation leg.
        """
        requested = tuple(guard)
        if requested == tuple(self.kernels._COMPILE_OPTIONS):
            return
        self.kernels._COMPILE_OPTIONS = requested
        self.kernels._clear_kernel_cache()

    def source_for(self, volume: bool) -> str:
        return getattr(self.kernels, self.ATTRIBUTE[bool(volume)])

    def set_sources(self, sources: Dict[bool, str]) -> None:
        for volume, text in sources.items():
            setattr(self.kernels, self.ATTRIBUTE[bool(volume)], text)
        self.kernels._clear_kernel_cache()

    def disarm(self) -> bool:
        self.set_sources(self.pristine)
        return self.is_pristine()

    def is_pristine(self) -> bool:
        return all(self.source_for(volume) == self.pristine[volume]
                   for volume in (True, False))

    def launch_one(self, scratch, p, q, sigma, w, coefficients,
                   volume: bool) -> None:
        self.kernels._launch(self.kernels._get_kernel(volume), scratch, p, q,
                             sigma, w, coefficients, volume)

    def synchronize(self) -> None:
        cp.cuda.runtime.deviceSynchronize()

    def compile_log(self) -> List[Dict[str, Any]]:
        return list(self.kernels.compile_cache.compile_log())

    def clear(self) -> int:
        return self.kernels._clear_kernel_cache()


def pristine_sources(kernels_module) -> Dict[bool, str]:
    return {True: kernels_module._update_P_pml_real_kernel_code,
            False: kernels_module._update_P_pml_real_uniform_kernel_code}


def device_source_texts() -> Dict[bool, str]:
    """The two device strings, read off the SYNTAX TREE so the laptop backend needs
    no CuPy. The same folding ``test_ade_update_p.device_sources`` does, imported
    rather than copied so the two cannot drift."""
    from meep_gpu.cuda_kernels import test_ade_update_p as slice_tests  # noqa: PLC0415
    by_kernel = slice_tests.device_sources(slice_tests.module_source())
    return {True: by_kernel["update_P_pml_real"],
            False: by_kernel["update_P_pml_real_uniform"]}


def build_backend(name: str):
    if name == "numpy":
        return EvaluatorBackend(device_source_texts())
    if cp is None:
        raise SystemExit("--backend cupy needs CuPy; this host has none")
    from meep_gpu.cuda_kernels import ade_kernels as kernels  # noqa: PLC0415
    return KernelBackend(kernels, pristine_sources(kernels))


# ---------------------------------------------------------------------------
# The launch plans -- the SHIPPED one and the planted host defects
# ---------------------------------------------------------------------------
#
# The shipped launcher is ``ade_kernels.update_P_fused_pml_real`` and the two
# PRIMARY legs call it. It is ALSO transcribed here, for one reason: a HOST defect
# lives in the caller, and there is no other way to plant one. The transcription is
# checked against the real launcher on the certifying backend (legs ``p0`` and
# ``p1`` over one product), so it is measured rather than asserted.


def _resolve(fields, state, component, drive):
    volume = bool(coverage.ade_sigma_is_volume(state, component))
    return {"w": drive(component), "p": state.P[component],
            "q": state.P_prev[component], "scratch": state._scratch,
            "sigma": state.sigma[component], "volume": volume,
            "coefficients": tuple(state._coefficients)}


def _rotate(state, component, slot) -> None:
    """dispersion.py:689-691, verbatim, and only once the launch returned."""
    state.P[component] = slot["scratch"]
    state.P_prev[component] = slot["p"]
    state._scratch = slot["q"]


def run_shipped(backend, fields, drive, steps: int) -> int:
    """Pointers resolved IMMEDIATELY BEFORE each launch, rotation after it."""
    launches = 0
    for _ in range(steps):
        for state in tuple(fields.polarizations):
            for component in tuple(state.driven()):
                slot = _resolve(fields, state, component, drive)
                backend.launch_one(slot["scratch"], slot["p"], slot["q"],
                                   slot["sigma"], slot["w"],
                                   slot["coefficients"], slot["volume"])
                launches += 1
                _rotate(state, component, slot)
        advance_drive(fields)
    backend.synchronize()
    return launches


def run_real_launcher(backend, fields, layer, drive, steps: int) -> int:
    """The SHIPPED entry point itself, ``update_P_fused_pml_real``.

    Present so that ``run_shipped`` above -- which every other leg drives, because
    a host defect has to be plantable -- is measured against the real thing rather
    than believed. CuPy only: the launcher imports CuPy at module scope.
    """
    launches = 0
    for _ in range(steps):
        launches += backend.kernels.update_P_fused_pml_real(
            fields, layer, drive=drive)
        advance_drive(fields)
    backend.synchronize()
    return launches


def run_stale_pointers(backend, fields, drive, steps: int) -> int:
    """HOST DEFECT: the three buffers resolved ONCE and the views reused.

    The hazard the launcher's docstring names in as many words, and the sibling
    plan's too (``triton_kernels/launch.py:1768-1782``). The slot bookkeeping is
    still done correctly -- only the ARITHMETIC uses stale views -- so the defect
    is a smooth, converged, entirely wrong polarization rather than a crash.

    ITS DISCRIMINATOR IS THE SHAPE OF ITS OWN VERDICT: with ONE driven component
    and ONE launch the stale views ARE the fresh ones and this is a null. It is
    caught only where a rotation has happened in between, which is what makes the
    multi-step leg a statement about the rotation.
    """
    plan = [(state, component, _resolve(fields, state, component, drive))
            for state in tuple(fields.polarizations)
            for component in tuple(state.driven())]
    launches = 0
    for _ in range(steps):
        for state, component, slot in plan:
            backend.launch_one(slot["scratch"], slot["p"], slot["q"],
                               slot["sigma"], slot["w"],
                               slot["coefficients"], slot["volume"])
            launches += 1
            _rotate(state, component, slot)
        advance_drive(fields)
    backend.synchronize()
    return launches


def run_no_rotation(backend, fields, drive, steps: int) -> int:
    """HOST DEFECT: the launch is right and the three names never move."""
    launches = 0
    for _ in range(steps):
        for state in tuple(fields.polarizations):
            for component in tuple(state.driven()):
                slot = _resolve(fields, state, component, drive)
                backend.launch_one(slot["scratch"], slot["p"], slot["q"],
                                   slot["sigma"], slot["w"],
                                   slot["coefficients"], slot["volume"])
                launches += 1
        advance_drive(fields)
    backend.synchronize()
    return launches


def run_scalar_sigma_for_a_volume(backend, fields, drive, steps: int) -> int:
    """HOST DEFECT: the UNIFORM variant compiled for a graded sigma.

    The compile-time specialization and the predicate disagreeing is what
    ``coverage.ade_sigma_is_volume`` exists as ONE function to prevent, and the
    sibling track's mutation m13 is exactly this. The scalar handed over is the
    volume's mean, so the answer is smooth, plausible and wrong everywhere the
    grading is.
    """
    launches = 0
    for _ in range(steps):
        for state in tuple(fields.polarizations):
            for component in tuple(state.driven()):
                slot = _resolve(fields, state, component, drive)
                sigma = slot["sigma"]
                if slot["volume"]:
                    sigma = float(to_host(sigma).mean())
                backend.launch_one(slot["scratch"], slot["p"], slot["q"],
                                   sigma, slot["w"], slot["coefficients"], False)
                launches += 1
                _rotate(state, component, slot)
        advance_drive(fields)
    backend.synchronize()
    return launches


def run_nothing(backend, fields, drive, steps: int) -> int:
    """HARNESS: no launch at all. Must DIVERGE, or bytes prove nothing."""
    for _ in range(steps):
        advance_drive(fields)
    return 0


HOST_PLANS: Dict[str, Callable[..., int]] = {
    "shipped": run_shipped,
    "stale_pointers": run_stale_pointers,
    "no_rotation": run_no_rotation,
    "scalar_sigma_for_a_volume": run_scalar_sigma_for_a_volume,
    "no_launch": run_nothing,
}


# ---------------------------------------------------------------------------
# The source defects
# ---------------------------------------------------------------------------
#
# Every one of these is a SILENT wrong answer: a smooth, converged, plausible
# polarization with the wrong recurrence in it. None is a crash, and none would be
# caught by a magnitude comparison at any tolerance a physicist would accept.

_EXPRESSION = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"


def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    new, count = re.subn(pattern, replacement, text)
    return new, count


def m_reassociate_three_terms(text: str) -> Tuple[str, int]:
    """The left association across the two ``+=`` regrouped to the right.

    THE FIRST OF THE FOUR THINGS THAT DECIDE BIT-IDENTITY HERE. The array path's
    two ``+=`` statements fix the association (dispersion.py:684-686) and float32
    addition is not associative. Algebraically equal, numerically not.
    """
    return _sub(re.escape(_EXPRESSION),
                "(p * c_now) + ((c_prev * q) + (c_drive * (s * w)))", text)


def m_associate_c_drive_with_sigma(text: str) -> Tuple[str, int]:
    """``c_drive * (s * w)`` regrouped as ``(c_drive * s) * w``.

    THE SECOND. ``scratch += c_drive * (sigma * w)`` (dispersion.py:686) builds the
    inner product first; moving the parenthesis rounds a different intermediate.
    """
    return _sub(re.escape("(c_drive * (s * w))"), "((c_drive * s) * w)", text)


def m_drop_previous_history(text: str) -> Tuple[str, int]:
    """``c_prev * P^(n-1)`` zeroed: a second-order recurrence reduced to first
    order. Stable, smooth, and the wrong material."""
    return _sub(re.escape("(c_prev * q)"), "(0.0f * q)", text)


def m_swap_p_and_q(text: str) -> Tuple[str, int]:
    """``P^n`` and ``P^(n-1)`` read from each other's buffers.

    The defect an author makes by transcribing the rotation's three names in the
    order they appear in the docstring rather than the order they are read in.
    """
    out = text.replace("float p = p_now[idx];", "float p = p_prev[idx];")
    out = out.replace("float q = p_prev[idx];", "float q = p_now[idx];")
    return out, int(out != text) * 2


def m_swap_c_now_and_c_prev(text: str) -> Tuple[str, int]:
    """The two recurrence coefficients bound to each other's terms.

    Both are finite, both are order one, and the resulting oscillator is entirely
    plausible -- it simply is not the susceptibility the caller asked for.
    """
    return _sub(re.escape("(p * c_now) + (c_prev * q)"),
                "(p * c_prev) + (c_now * q)", text)


def m_drop_sigma(text: str) -> Tuple[str, int]:
    """The per-cell (or per-term) strength dropped from the drive product.

    Every cell then carries sigma = 1: a uniform material where a graded one was
    asked for, which converges to a perfectly reasonable wrong answer.
    """
    return _sub(re.escape("(s * w)"), "(1.0f * w)", text)


def n_commute_the_history_product(text: str) -> Tuple[str, int]:
    """NULL: ``c_prev * q`` written ``q * c_prev``. IEEE multiply commutes bitwise.

    Its discriminator is ``swap_c_now_and_c_prev``, which REBINDS the same operands
    and must be caught -- so the pair says the comparator is sensitive to which
    coefficient multiplies which term and insensitive to operand order.
    """
    return _sub(re.escape("(c_prev * q)"), "(q * c_prev)", text)


def n_commute_the_drive_product(text: str) -> Tuple[str, int]:
    """NULL: ``s * w`` written ``w * s``.

    Paired with ``associate_c_drive_with_sigma``, which regroups the SAME
    sub-expression and must be caught: grouping matters, order does not.
    """
    return _sub(re.escape("(s * w)"), "(w * s)", text)


def n_reorder_the_two_loads(text: str) -> Tuple[str, int]:
    """NULL: the ``p`` and ``q`` load statements swapped, bindings untouched.

    Distinct from ``swap_p_and_q``, which swaps the BUFFERS. Two independent loads
    reordered change nothing, and a comparator that called this a defect would be
    reporting on the source text rather than on the arithmetic.
    """
    before = "    float p = p_now[idx];\n    float q = p_prev[idx];"
    after = "    float q = p_prev[idx];\n    float p = p_now[idx];"
    return _sub(re.escape(before), after, text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "reassociate_three_terms": m_reassociate_three_terms,
    "associate_c_drive_with_sigma": m_associate_c_drive_with_sigma,
    "drop_previous_history": m_drop_previous_history,
    "swap_p_and_q": m_swap_p_and_q,
    "swap_c_now_and_c_prev": m_swap_c_now_and_c_prev,
    "drop_sigma": m_drop_sigma,
    "commute_the_history_product": n_commute_the_history_product,
    "commute_the_drive_product": n_commute_the_drive_product,
    "reorder_the_two_loads": n_reorder_the_two_loads,
}


def mutate_sources(pristine: Dict[bool, str],
                   transform: Callable[[str], Tuple[str, int]]
                   ) -> Tuple[Dict[bool, str], int]:
    """Both variants mutated by the same needle, with the total site count.

    BOTH, always. The two device strings differ in one parameter declaration and
    one load; a leg that armed only the volume variant would report a verdict for
    the uniform one it never touched.
    """
    out, sites = {}, 0
    for volume, text in pristine.items():
        mutated, count = transform(text)
        out[volume] = mutated
        sites += count
    return out, sites


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec: Dict[str, Any], guard_label: str, steps: int) -> str:
    return "|".join([
        spec["label"],
        "".join(b[0] for b in spec["boundaries"]),
        spec["kind"][:4], spec["sigma_form"], spec["layout"],
        f"C{spec['courant']}", spec["value_class"], guard_label, f"n{steps}"])


def one_case(backend, xp, spec: Dict[str, Any], guard_label: str,
             guard: Sequence[str], steps: int,
             source_mutation: Optional[str] = None,
             host_plan: str = "shipped",
             wrong_drive: bool = False,
             comparator=None,
             use_real_launcher: bool = False) -> Dict[str, Any]:
    """Oracle and kernel on two identically seeded fixtures; slots compared as words."""
    key = case_key(spec, guard_label, steps)
    seed = (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16)) % (2 ** 32)

    reference, ref_layer, grid, _ = build(xp, spec, seed)
    actual, act_layer, _grid2, host = build(xp, spec, seed)

    # THE FIXTURES ARE MEASURED IDENTICAL, NOT ASSUMED IDENTICAL. Two builds from
    # one seed is a claim about the draw order, and a claim is a hypothesis until
    # measured.
    before_ref = fixture_hosts(reference)
    before_act = fixture_hosts(actual)
    started_differing = sum(words_differ(before_ref[name], before_act[name])
                            for name in before_ref)

    floors = term_magnitudes(actual)
    predicate_covered, predicate_reason = coverage.covers_real_pml_ade_update_p(
        actual, act_layer, grid)

    # --- the oracle: the array path itself -------------------------------
    for _ in range(steps):
        stepping.update_P(reference, ref_layer)
        advance_drive(reference)
    oracle = slot_hosts(reference)

    moved = sum(words_differ(oracle[name], before_ref[name]) for name in oracle)
    total_words = sum(int(values.size) for values in oracle.values())
    all_finite = all(bool(np.all(np.isfinite(values))) for values in oracle.values())

    # --- the kernel, on the twin fixture ---------------------------------
    backend.set_guard(guard)
    drive = ((lambda component: getattr(actual, component)) if wrong_drive
             else actual.drive_field)
    launch_error = None
    launches = -1
    # The real launcher lives in a CuPy-importing module; on the evaluator backend
    # there is none, and the leg falls back to the transcription and SAYS so rather
    # than claiming it drove the shipped entry point.
    use_real_launcher = bool(use_real_launcher) and hasattr(backend, "kernels")
    try:
        if use_real_launcher:
            launches = run_real_launcher(backend, actual, act_layer, drive, steps)
        else:
            launches = HOST_PLANS[host_plan](backend, actual, drive, steps)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]

    compare = comparator or bit_compare
    parts = ({name: compare(oracle[name], array)
              for name, array in state_slots(actual)}
             if launch_error is None else {})
    verdict = combine(parts) if parts else {"bit_identical": False,
                                            "differing_floats": 0,
                                            "total_floats": 0,
                                            "per_component": {}}
    expected = expected_launch_count(actual, steps)
    if host_plan == "no_launch":
        expected = 0

    record: Dict[str, Any] = {
        "key": key,
        "shape": list(grid.shape),
        "boundaries": list(spec["boundaries"]),
        "symmetry": [str(s) for s in spec.get("symmetry", ())],
        "kind": spec["kind"],
        "sigma_form": spec["sigma_form"],
        "layout": spec["layout"],
        "states": spec["states"],
        "driven": list(spec["driven"]),
        "courant": spec["courant"],
        "value_class": spec["value_class"],
        "guard": guard_label,
        "steps": steps,
        "source_mutation": source_mutation,
        "host_plan": host_plan,
        "wrong_drive": wrong_drive,
        "used_real_launcher": use_real_launcher,
        "bit_identical": bool(verdict["bit_identical"]) and launch_error is None,
        "differing_floats": verdict["differing_floats"],
        "total_floats": verdict["total_floats"],
        "max_ulp": verdict.get("max_ulp"),
        "launch_error": launch_error,
        "launches": launches,
        "expected_launches": expected,
        "launch_count_as_expected": launches == expected,
        "fixtures_started_identical": started_differing == 0,
        "fixture_words_differing_at_start": started_differing,
        "oracle_moved_words": moved,
        "oracle_total_words": total_words,
        "oracle_moved": moved > 0,
        "oracle_all_finite": all_finite,
        "predicate_covers": bool(predicate_covered),
        "predicate_reason": predicate_reason,
        "operand_census": operand_census(host),
    }
    record.update(floors)
    record["drive_term_is_live"] = floors["drive_term_max_abs"] > 0.0
    record["history_term_is_live"] = floors["history_term_max_abs"] > 0.0
    return record


def case_is_valid(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if not record["fixtures_started_identical"]:
        return False, ("the two fixtures did not start byte-identical; this case "
                       "would compare two different problems")
    if not record["oracle_moved"]:
        return False, ("the array path changed no output word: this fixture is a "
                       "fixed point of the recurrence")
    if not record["oracle_all_finite"]:
        return False, ("the oracle overflowed to inf or NaN; every word would then "
                       "agree for a reason unrelated to the kernel")
    if not record["drive_term_is_live"]:
        return False, ("c_drive * (sigma * w) is identically zero: this case cannot "
                       "distinguish the drive binding or the sigma product")
    if not record["history_term_is_live"]:
        return False, ("c_prev * P_prev is identically zero: the recurrence is "
                       "first order here and its history term cannot be tested")
    if not record["launch_count_as_expected"] and record["launch_error"] is None:
        return False, (f"{record['launches']} launches where "
                       f"{record['expected_launches']} were owed; bytes alone "
                       f"cannot prove the kernel ran")
    return True, None


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def case_product(product: str) -> List[Dict[str, Any]]:
    """The swept cases, before the guard axis multiplies them."""
    shapes = SHAPES if product == "full" else SHAPES[:2]
    triples = BOUNDARY_TRIPLES if product == "full" else BOUNDARY_TRIPLES[:1]
    kinds = KINDS
    layouts = LAYOUTS if product == "full" else LAYOUTS[1:]
    courants = COURANTS if product == "full" else COURANTS[:1]
    cases: List[Dict[str, Any]] = []
    for shape, label in shapes:
        for triple in triples:
            if config_viable(shape, triple) is not None:
                continue
            for kind in kinds:
                for sigma_form in SIGMA_FORMS:
                    for layout, states, driven in layouts:
                        for courant in courants:
                            for value_class in VALUE_CLASSES:
                                cases.append({
                                    "shape": shape, "label": label,
                                    "boundaries": triple, "kind": kind,
                                    "sigma_form": sigma_form,
                                    "layout": layout, "states": states,
                                    "driven": driven, "courant": courant,
                                    "value_class": value_class})
    if product == "full":
        # The mirror fold, this predicate's own admission clause.
        for fold in FOLD_CASES:
            for sigma_form in SIGMA_FORMS:
                cases.append({
                    "shape": fold["shape"], "label": fold["label"],
                    "symmetry": fold["symmetry"],
                    "boundaries": BOUNDARY_TRIPLES[0], "kind": LORENTZIAN,
                    "sigma_form": sigma_form,
                    "layout": LAYOUTS[2][0], "states": LAYOUTS[2][1],
                    "driven": LAYOUTS[2][2], "courant": COURANTS[0],
                    "value_class": "uniform"})
    return cases


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in cases:
        bucket = per_guard.setdefault(case["guard"], {"identical": 0, "ran": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
    by_class: Dict[str, Dict[str, int]] = {}
    by_sigma: Dict[str, Dict[str, int]] = {}
    for case in cases:
        if case["guard"] != "fmad_false":
            continue
        for table, key in ((by_class, "value_class"), (by_sigma, "sigma_form")):
            bucket = table.setdefault(case[key], {"identical": 0, "ran": 0})
            bucket["ran"] += 1
            bucket["identical"] += int(case["bit_identical"])
    census = {"values": 0, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    for case in cases:
        if case["value_class"] != "subnormal_band":
            continue
        for key in census:
            census[key] += case["operand_census"][key]
    return {
        "per_guard": per_guard,
        "by_value_class": by_class,
        "by_sigma_form": by_sigma,
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in cases),
        "every_case_had_a_live_drive_term": all(c["drive_term_is_live"] for c in cases),
        "every_case_had_a_live_history_term": all(c["history_term_is_live"] for c in cases),
        "every_case_launched_as_expected": all(c["launch_count_as_expected"] for c in cases),
        "every_case_admitted_by_the_predicate": all(c["predicate_covers"] for c in cases),
        "min_drive_term_max_abs": (min(c["drive_term_max_abs"] for c in cases)
                                   if cases else None),
        "min_history_term_max_abs": (min(c["history_term_max_abs"] for c in cases)
                                     if cases else None),
        "total_launches": sum(max(c["launches"], 0) for c in cases),
    }


def run_bytes(backend, xp, results: Dict[str, Any], out_path: str,
              product: str, steps: int, leg_name: str,
              guards: Sequence[Tuple[str, Tuple[str, ...], bool]] = GUARD_SETS
              ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    plan = case_product(product)
    swept = [g for g in guards if backend.certifies or g[0] == "fmad_false"]
    total = len(plan) * len(swept)
    index = 0
    started = time.time()
    for guard_label, guard, _primary in swept:
        for spec in plan:
            index += 1
            # THE SHIPPED ENTRY POINT, not the transcription. The two primary legs
            # plant nothing, so there is no reason for them to drive anything but
            # ``update_P_fused_pml_real`` itself -- with its aliasing check, its
            # coefficient check and its own rotation. ``run_shipped`` exists only
            # where a HOST defect has to be plantable, and legs p0/p1 are the pair
            # that welds it to the real caller.
            record = one_case(backend, xp, spec, guard_label, guard, steps,
                              use_real_launcher=True)
            valid, why = case_is_valid(record)
            record["case_is_valid"] = valid
            record["refused_because"] = why
            (cases if valid else refused).append(record)
            log(f"[{leg_name}] {index}/{total} {record['key']} "
                f"identical={record['bit_identical']} "
                f"diff={record['differing_floats']}/{record['total_floats']} "
                f"launches={record['launches']} "
                f"drive={record['drive_term_max_abs']:.3e}"
                + ("" if valid else f" REFUSED: {why}"))
            results[leg_name] = {"cases": cases, "refused": refused,
                                 "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


def allclose_compare(a: Any, b: Any) -> Dict[str, Any]:
    """The magnitude comparison a byte gate exists to replace, for the harness leg."""
    host_a, host_b = to_host(a), to_host(b)
    identical = bool(np.allclose(host_a, host_b, rtol=1e-5, atol=1e-8,
                                 equal_nan=True))
    return {"bit_identical": identical, "differing_floats": 0 if identical else 1,
            "total_floats": int(host_a.size)}


#: (leg, what is armed, expectation, why).
#: "pass"                -> every valid case must stay identical
#: "caught"              -> every valid case must DIVERGE
#: "uncaught"            -> every valid case must stay identical, and that IS the
#:                          measurement
#: "caught_only_multi_step" -> uncaught at one launch with one driven component,
#:                          caught at the multi-step budget; the PAIR is the claim
#: "measure"             -> a number is reported, not a bar
MUTATION_LEGS: Tuple[Dict[str, Any], ...] = (
    {"leg": "p0_shipped_launcher", "expect": "pass", "real_launcher": True,
     "why": "the SHIPPED update_P_fused_pml_real itself, unmutated, on the "
            "mutation product -- so the transcribed run_shipped every other leg "
            "drives is measured against the real caller rather than believed."},
    {"leg": "p1_pristine", "expect": "pass",
     "why": "the shipped source through the transcribed launcher, so a divergence "
            "on any later leg is attributable to the arming and not to the product."},
    {"leg": "m1_reassociate_three_terms", "mutation": "reassociate_three_terms",
     "kind": "source", "expect": "caught",
     "why": "the left association across the two += regrouped right; float32 "
            "addition is not associative and this is the family's first claim"},
    {"leg": "m2_associate_c_drive_with_sigma",
     "mutation": "associate_c_drive_with_sigma", "kind": "source",
     "expect": "caught",
     "why": "c_drive * (s * w) regrouped as (c_drive * s) * w: the second claim, "
            "a differently rounded intermediate"},
    {"leg": "m3_drop_previous_history", "mutation": "drop_previous_history",
     "kind": "source", "expect": "caught",
     "why": "a second-order recurrence reduced to first order: stable, smooth and "
            "the wrong material"},
    {"leg": "m4_swap_p_and_q", "mutation": "swap_p_and_q", "kind": "source",
     "expect": "caught",
     "why": "P^n and P^(n-1) read from each other's buffers"},
    {"leg": "m5_swap_c_now_and_c_prev", "mutation": "swap_c_now_and_c_prev",
     "kind": "source", "expect": "caught",
     "why": "the two recurrence coefficients bound to each other's terms; an "
            "entirely plausible oscillator that is not the one asked for"},
    {"leg": "m6_drop_sigma", "mutation": "drop_sigma", "kind": "source",
     "expect": "caught",
     "why": "every cell carries sigma = 1: a uniform material where a graded one "
            "was asked for"},
    {"leg": "n1_commute_the_history_product",
     "mutation": "commute_the_history_product", "kind": "source",
     "expect": "uncaught",
     "why": "NULL: IEEE multiply commutes bitwise. Its discriminator is m5, which "
            "REBINDS the same operands and must be caught."},
    {"leg": "n2_commute_the_drive_product", "mutation": "commute_the_drive_product",
     "kind": "source", "expect": "uncaught",
     "why": "NULL, paired with m2 on the same sub-expression: grouping matters, "
            "operand order does not."},
    {"leg": "n3_reorder_the_two_loads", "mutation": "reorder_the_two_loads",
     "kind": "source", "expect": "uncaught",
     "why": "NULL: two independent loads swapped, bindings untouched. Paired with "
            "m4, which swaps the BUFFERS and must be caught."},
    {"leg": "h1_wrong_drive_stored_E", "kind": "host", "wrong_drive": True,
     "expect": "caught",
     "why": "THE CONTROL THIS FAMILY MOST OWES. Fields.drive_field returns f_w "
            "under an active layer and the stored E without one (fields.py:1140-"
            "1163); the two agree EXACTLY outside the absorber, so binding the "
            "wrong one is invisible in every no-PML test. The sibling track "
            "measured 3072/15360 words differing."},
    {"leg": "h2_wrong_drive_is_null_when_E_equals_f_w", "kind": "host",
     "wrong_drive": True, "expect": "uncaught", "drive_equals_stored_E": True,
     "why": "THE DISCRIMINATOR for h1. Seed the stored E equal to f_w -- the state "
            "the whole no-PML corpus is in -- and the same planted binding must be "
            "a NULL. Without this pair, 'the wrong drive is caught' could be a "
            "statement about any difference between two arrays."},
    {"leg": "h3_stale_pointers", "kind": "host", "host_plan": "stale_pointers",
     "expect": "caught_only_multi_step",
     "why": "THE ROTATION LEG. The three buffers resolved once and the views "
            "reused: identical at one launch with one driven component, wrong from "
            "the rotation onward. The pair is what makes the multi-step budget a "
            "statement about the rotation rather than about the budget."},
    {"leg": "h4_no_rotation", "kind": "host", "host_plan": "no_rotation",
     "expect": "caught",
     "why": "the launch is right and the three names never move: this step's "
            "result is written and then discarded"},
    {"leg": "h5_scalar_sigma_for_a_volume", "kind": "host",
     "host_plan": "scalar_sigma_for_a_volume", "expect": "caught",
     "sigma_forms": ("volume",),
     "why": "the compile-time specialization and the predicate out of step: the "
            "uniform variant run for a graded sigma. Armed only where sigma IS a "
            "volume, because on a uniform sigma it is the shipped plan."},
    {"leg": "x1_no_launch_at_all", "kind": "host", "host_plan": "no_launch",
     "expect": "caught",
     "why": "HARNESS: the comparison run with the kernel never launched. It must "
            "DIVERGE -- two identically seeded fixtures and an oracle that moved "
            "is the only thing standing between this gate and a vacuous pass."},
    {"leg": "x2_allclose_hides_the_reassociation",
     "mutation": "reassociate_three_terms", "kind": "source", "expect": "measure",
     "comparator": "allclose",
     "why": "HARNESS: relax the byte comparison to np.allclose and COUNT how many "
            "cases of a real defect walk through. The reassociation is "
            "algebraically EQUAL and differs by ~1 ulp relative, far below "
            "rtol=1e-5, so this is where the blindness lives."},
    {"leg": "z1_disarmed", "expect": "pass",
     "why": "THE DISARM CHECK, after everything. The same product as p1 with every "
            "needle withdrawn: a gate whose arming leaked would report catches "
            "forever, and its last leg would be the one to say so."},
)


def mutation_case_plan(leg: Dict[str, Any], product: str,
                       steps: int) -> List[Dict[str, Any]]:
    """The reduced product one leg is armed over."""
    shape, label = SHAPES[1] if product == "full" else SHAPES[0]
    layouts = LAYOUTS[1:] if leg["expect"] != "caught_only_multi_step" else LAYOUTS[:1]
    sigma_forms = leg.get("sigma_forms", SIGMA_FORMS)
    plan = []
    for sigma_form in sigma_forms:
        for layout, states, driven in layouts:
            plan.append({
                "shape": shape, "label": label,
                "boundaries": BOUNDARY_TRIPLES[0], "kind": LORENTZIAN,
                "sigma_form": sigma_form, "layout": layout, "states": states,
                "driven": driven, "courant": COURANTS[0],
                "value_class": "uniform",
                "drive_equals_stored_E": bool(leg.get("drive_equals_stored_E"))})
    return plan


def adjudicate(leg: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did this leg's verdict come out the way the leg requires?"""
    valid = [c for c in cases if c["case_is_valid"]]
    identical = sum(1 for c in valid if c["bit_identical"])
    ran = len(valid)
    expect = leg["expect"]
    out: Dict[str, Any] = {"identical": identical, "ran": ran,
                           "single": f"{identical}/{ran}"}
    if expect == "pass":
        out["as_required"] = ran > 0 and identical == ran
    elif expect == "caught":
        out["as_required"] = ran > 0 and identical == 0
    elif expect == "uncaught":
        out["as_required"] = ran > 0 and identical == ran
    elif expect == "caught_only_multi_step":
        single = [c for c in valid if c["steps"] == 1]
        multi = [c for c in valid if c["steps"] > 1]
        out["single_launch_identical"] = (
            f"{sum(c['bit_identical'] for c in single)}/{len(single)}")
        out["multi_step_identical"] = (
            f"{sum(c['bit_identical'] for c in multi)}/{len(multi)}")
        out["as_required"] = bool(
            single and multi
            and all(c["bit_identical"] for c in single)
            and not any(c["bit_identical"] for c in multi))
    elif expect == "measure":
        out["as_required"] = ran > 0
        out["walked_through_allclose"] = identical
    else:
        raise ValueError(f"unknown expectation {expect!r}")
    return out


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, steps: int, multi_budget: int) -> Dict[str, Any]:
    legs: List[Dict[str, Any]] = []
    started = time.time()
    pristine = dict(backend.pristine)
    for leg in MUTATION_LEGS:
        leg_started = time.time()
        transform = SOURCE_MUTATIONS.get(leg.get("mutation")) if leg.get("mutation") else None
        comparator = allclose_compare if leg.get("comparator") == "allclose" else None
        host_plan = leg.get("host_plan", "shipped")

        # ARM IT, and refuse the leg if the needle matched nothing: a mutation that
        # has drifted away from the text exercises nothing and reports a pass while
        # doing it.
        sites = None
        blind: List[str] = []
        if transform is not None:
            mutated, sites = mutate_sources(pristine, transform)
            backend.set_sources(mutated)
            for text in mutated.values():
                blind.extend(evaluator_blind_lines(text))
        before_compiles = len(backend.compile_log())

        # THE EVALUATOR BACKEND CANNOT SEE EVERY DEFECT, and a leg it cannot see
        # must SAY so rather than score a number.
        skip_reason = None
        if not backend.certifies and blind:
            skip_reason = "outside_the_grammar"

        cases: List[Dict[str, Any]] = []
        if skip_reason is None:
            plan = mutation_case_plan(leg, product, steps)
            step_counts = ((1, multi_budget)
                           if leg["expect"] == "caught_only_multi_step"
                           else (leg.get("steps", steps),))
            for spec in plan:
                for count in step_counts:
                    record = one_case(
                        backend, xp, spec, "fmad_false", ("--fmad=false",), count,
                        source_mutation=leg.get("mutation"),
                        host_plan=host_plan,
                        wrong_drive=bool(leg.get("wrong_drive")),
                        comparator=comparator,
                        use_real_launcher=bool(leg.get("real_launcher")))
                    valid, why = case_is_valid(record)
                    # A leg that plants a HOST defect changes the launch count on
                    # purpose (x1 launches nothing); the launch clause is checked
                    # against what the LEG asked for, not against the shipped plan.
                    record["case_is_valid"] = valid
                    record["refused_because"] = why
                    cases.append(record)

        # DID THE MUTATED BYTES REACH THE COMPILER? Apply-time text matching is
        # necessary and not sufficient: between the rewrite and NVRTC sit a memo
        # that has to miss and a disk cache that has to not answer.
        compiles = backend.compile_log()[before_compiles:]
        mutated_digests = set()
        if transform is not None:
            mutated, _n = mutate_sources(pristine, transform)
            mutated_digests = {hashlib.sha256(text.encode("utf-8")).hexdigest()
                               for text in mutated.values()}
        from_mutated = sum(1 for entry in compiles
                           if entry.get("source_sha256") in mutated_digests)

        disarmed = backend.disarm()
        backend.clear()

        record = dict(leg)
        if skip_reason is not None:
            record.update({
                "identical": 0, "ran": 0, "single": "0/0", "as_required": None,
                "measurable_on_this_backend": False,
                "why_not_measured_here":
                    "the mutation rewrites a statement into a form the evaluator's "
                    "grammar does not match, so off device it is not applied at "
                    "all; a verdict here would be about the parser, not the kernel",
                "unmatched_lines": blind[:4]})
        else:
            record.update(adjudicate(leg, cases))
            record["measurable_on_this_backend"] = True
        record["mutation_sites"] = sites
        record["compiles_on_this_leg"] = len(compiles)
        record["compiles_from_mutated_source"] = from_mutated
        record["mutated_source_reached_the_compiler"] = (
            None if transform is None else (from_mutated > 0))
        record["source_restored_after_the_leg"] = bool(disarmed)
        record["cases"] = cases
        record["seconds"] = round(time.time() - leg_started, 1)
        if transform is not None and sites == 0:
            record["as_required"] = False
            record["refused"] = ("the mutation matched nothing in either device "
                                 "string; it has drifted apart from the kernel and "
                                 "is exercising nothing")
        if backend.certifies and transform is not None and from_mutated == 0:
            record["as_required"] = False
            record["refused"] = ("the mutated bytes never reached the compiler; a "
                                 "verdict for a mutation that never compiled is "
                                 "indistinguishable from one that found nothing")
        if not disarmed:
            record["as_required"] = False
            record["refused"] = ("the device strings were not restored after this "
                                 "leg; every later leg would be measuring this "
                                 "one's needle")
        legs.append(record)
        log(f"[mutations] {leg['leg']}: expect={leg['expect']} "
            f"{record.get('single')} as_required={record['as_required']} "
            f"sites={sites} compiles_from_mutated={from_mutated} "
            f"disarmed={disarmed} ({record['seconds']} s)")
        results["mutations"] = {
            "legs": legs,
            "legs_as_required":
                f"{sum(1 for l in legs if l['as_required'])}/"
                f"{sum(1 for l in legs if l['as_required'] is not None)}",
            "legs_not_measurable_on_this_backend":
                sum(1 for l in legs
                    if l.get("measurable_on_this_backend") is False),
            "all_legs_as_required": all(
                l["as_required"] for l in legs if l["as_required"] is not None),
            "every_leg_disarmed": all(l["source_restored_after_the_leg"]
                                      for l in legs),
            "seconds": round(time.time() - started, 1)}
        save(results, out_path)
    return results["mutations"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

LEG_NAMES = ("bytes", "multistep", "mutations")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cupy", "numpy"))
    parser.add_argument("--legs", default=",".join(LEG_NAMES))
    parser.add_argument("--product", default="auto",
                        choices=("auto", "full", "reduced"))
    parser.add_argument("--steps", type=int, default=1,
                        help="launches per case in the single-launch leg")
    parser.add_argument("--multi-step-budget", type=int, default=MULTI_STEP_BUDGET)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("keep", "flush", "match_meep", "ieee"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    args = parser.parse_args(argv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    legs = tuple(name for name in args.legs.split(",") if name)
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = "cupy" if cp is not None else "numpy"
    product = args.product
    if product == "auto":
        product = "full" if backend_name == "cupy" else "reduced"

    results: Dict[str, Any] = {
        "gate": "cuda_ade_update_p",
        "kernels": ["update_P_pml_real", "update_P_pml_real_uniform"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "resolution": RESOLUTION,
        "multi_step_budget": args.multi_step_budget,
        "subjects": {
            name: probe.source_digest(os.path.join(
                _REPO_API, "meep_gpu", "cuda_kernels", name))
            for name in ("ade_kernels.py", "coverage.py", "compile_cache.py")},
        "oracle_subjects": {
            name: probe.source_digest(os.path.join(_REPO_API, "meep_gpu", name))
            for name in ("dispersion.py", "stepping.py", "fields.py")},
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
        # SET FALSE BEFORE ANYTHING RUNS. ``save`` stamps a canonical verdict on
        # every write, and ``gate_provenance.read_verdict`` reads this key before it
        # falls back to ``certifies`` -- which says only that the backend COULD
        # certify. Without this, an artifact from a run that died mid-leg would read
        # as RELEASED. It is overwritten by the real verdict at the end.
        "passed": False,
    }

    # THE ORDER IS THE WHOLE POINT: the observer wraps NVRTC BELOW the strip so it
    # records the options NVRTC really received, and the policy installs before the
    # first compile. Install-then-compile keeps a planted subnormal;
    # compile-then-install flushes it and stays flushed through a cache clear.
    if backend_name == "cupy":
        results["nvrtc_binary_observer"] = probe.install_nvrtc_binary_observer()
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    backend = build_backend(backend_name)
    results["device_source_sha256"] = {
        "update_P_pml_real": hashlib.sha256(
            backend.pristine[True].encode("utf-8")).hexdigest(),
        "update_P_pml_real_uniform": hashlib.sha256(
            backend.pristine[False].encode("utf-8")).hexdigest()}
    xp = cp if backend_name == "cupy" else _NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
    save(results, out_path)

    if "bytes" in legs:
        run_bytes(backend, xp, results, out_path, product, args.steps, "bytes")
    if "multistep" in legs:
        # THE PRIMARY GUARD ONLY. The control's job is to show the guard does work,
        # and one launch per case already shows it; paying 60x for the same
        # demonstration buys nothing, and this leg's cost is what a device slot is
        # spent on.
        run_bytes(backend, xp, results, out_path,
                  "reduced" if product == "full" else product,
                  args.multi_step_budget, "multistep", guards=GUARD_SETS[:1])
    if "mutations" in legs:
        run_mutations(backend, xp, results, out_path, product, args.steps,
                      args.multi_step_budget)

    if backend_name == "cupy":
        results["nvrtc_binaries"] = probe.nvrtc_binary_report()

    single = results.get("bytes", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    primary = single.get("per_guard", {}).get("fmad_false", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    multi_primary = multi.get("per_guard", {}).get("fmad_false", {})
    verdict = {
        "single_launch": f"{primary.get('identical', 0)}/{primary.get('ran', 0)}",
        "multi_step": f"{multi_primary.get('identical', 0)}/{multi_primary.get('ran', 0)}",
        "multi_step_budget": args.multi_step_budget,
        "guard_control_identical": f"{control.get('identical', 0)}/{control.get('ran', 0)}",
        "guard_control_diverged": (control.get("ran", 0) > 0
                                   and control.get("identical", 0) < control.get("ran", 0)),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_had_a_live_drive_term": single.get("every_case_had_a_live_drive_term"),
        "every_case_had_a_live_history_term": single.get("every_case_had_a_live_history_term"),
        "every_case_launched_as_expected": single.get("every_case_launched_as_expected"),
        "every_case_admitted_by_the_predicate": single.get("every_case_admitted_by_the_predicate"),
        "by_sigma_form": single.get("by_sigma_form"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        "every_leg_disarmed": mutations.get("every_leg_disarmed"),
        # ON A CERTIFYING BACKEND THIS MUST BE ZERO: the compiler executes every
        # line, so no defect is invisible for a reason belonging to the harness.
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
    }
    by_sigma = single.get("by_sigma_form", {}) or {}
    both_variants = all(
        by_sigma.get(form, {}).get("ran", 0) > 0
        and by_sigma[form]["identical"] == by_sigma[form]["ran"]
        for form in SIGMA_FORMS)
    verdict["both_sigma_variants_certified"] = both_variants
    verdict["passed"] = bool(
        results["certifies"]
        and primary.get("ran", 0) > 0 and primary["identical"] == primary["ran"]
        and multi_primary.get("ran", 0) > 0
        and multi_primary["identical"] == multi_primary["ran"]
        and both_variants
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_had_a_live_drive_term")
        and single.get("every_case_had_a_live_history_term")
        and single.get("every_case_moved_the_oracle")
        and single.get("every_case_launched_as_expected")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("every_leg_disarmed", False)
        and mutations.get("legs_not_measurable_on_this_backend", 1) == 0)
    if not results["certifies"]:
        verdict["why_not_certified"] = (
            "the numpy backend executes the device text and compiles nothing; it "
            "exercises the harness, it does not measure NVRTC's output")
    results["verdict"] = verdict
    # The one key gate_provenance.read_verdict reads first among bools. Written
    # explicitly so the canonical verdict cannot land on ``certifies``, which says
    # only that the backend COULD certify.
    results["passed"] = verdict["passed"]
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, out_path)

    log(f"[done] backend={backend_name} single={verdict['single_launch']} "
        f"multi={verdict['multi_step']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"guard_control_diverged={verdict['guard_control_diverged']} "
        f"passed={verdict['passed']}")
    if not results["certifies"]:
        return 0 if (mutations.get("all_legs_as_required", True)
                     and primary.get("identical", 0) == primary.get("ran", 0)) else 1
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
