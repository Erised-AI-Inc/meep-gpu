"""Sub-step byte-identity gate for the COMPLEX / NO-ABSORBER CUDA family.

THE QUESTION
============
``meep_gpu/cuda_kernels/complex_no_pml_kernels.py`` carries six kernels for a run
with complex64 storage and NO ACTIVE ABSORBER. None has ever run on a device.
This gate asks, per sub-step and per arm, whether each is byte-identical to
``stepping.step_B`` / ``step_D`` / ``update_E`` / ``update_P`` from ONE frozen
state -- at one launch and at 60 -- across phased and unphased periodic walls,
metallic walls, 3-D / 2-D / 1-D shapes, an absent absorber and an inert layer,
zero / one / two poles, uniform and volume sigma, BOTH float32 subnormal policies,
three value classes and both courants.

WHAT THE CORPUS ASKED FOR, AND WHY THERE ARE FOUR KERNEL SHAPES AND NOT THREE
=============================================================================
The six unserved corpus rows were lifted and their real ``Grid``/``Fields``
objects interrogated before any kernel was written
(``results/cuda_complex_no_pml_recon_2026-08-20/facts/``). All six store E, so
the DERIVED-E arm the real no-absorber family needed does not arise. What does
arise, and was not in this family's brief, is a CONDUCTIVITY: four of the six
rows return a float32 volume from ``fields.condfac_for`` for every one of
Bx..Dz, so ``stepping._apply_curl`` routes them to ``_apply_conductive_update``
(stepping.py:536-537) -- a different tail from ``target -= curl``. Eight of the
twelve curl slots are conductive and four are plain, so there are two curl pairs.

THE ARMS
========
* ``step_B`` / ``plain``       -- ``step_B_no_pml_complex``
* ``step_D`` / ``plain``       -- ``step_D_no_pml_complex``
* ``step_B`` / ``conductive``  -- ``step_B_no_pml_complex_conductive``
* ``step_D`` / ``conductive``  -- ``step_D_no_pml_complex_conductive``
* ``update_E``                 -- ``update_E_no_pml_complex_stored``
* ``update_P``                 -- ``update_P_no_pml_complex[_uniform]``

TWO FIXTURES, NOT ONE FIXTURE RESTORED
======================================
Every case builds the SAME configuration twice from the same seed and asserts the
two start byte-identical (``fixtures_agree``, a floor, not a comment). The oracle
runs on one and the kernel on the other. That is the ADE gate's shape and it is
required here for the same reason: ``update_P`` ROTATES array objects between
``P``, ``P_prev`` and ``_scratch``, so "restore the frozen state" would have to
reproduce a permutation as well as a set of values, and a restore that got the
permutation wrong would hide exactly the defect the multi-step leg exists to
catch. Comparison is BY SLOT NAME rather than by object identity, for the same
reason.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
================================================================
* ``fixtures_agree`` -- the two fixtures start byte-identical, or the case is
  comparing two different problems and a pass means nothing.
* ``oracle_moved`` -- the array path must change at least one output word from
  the frozen input. A case that moved nothing is SKIPPED, not passed.
* ``coefficient_profile`` -- every per-cell coefficient volume the case binds
  (``inv_eps``, ``condfac``, ``condinv``, volume ``sigma``) must be NON-IDENTITY
  and NON-CONSTANT, and the three components' volumes pairwise distinct.
  Against a table of ones a dropped multiply is invisible; against one shared
  volume a component-aliasing defect is invisible; against a constant volume a
  coefficient-INDEX error is invisible. This is the "a coefficient-index error is
  invisible and the case measures nothing" floor, and it is asserted per case.
* ``sources_are_distinct`` -- the three source volumes differ pairwise, so a
  component-binding defect has somewhere to show.
* ``phase_is_live`` -- on a phased spec at least one axis must carry a Bloch
  factor that is not 1+0j, or every rotation mutation is a null for a reason
  about the fixture.
* The mutation battery, INCLUDING legs that must come back UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything. The four DEAD-PRELUDE nulls
  (``pml_apply``'s and ``constitutive_apply``'s bodies, compiled into every
  kernel here and called by none) are the measurement that this family's tails
  really are ``target -= curl`` and the conductive triple.
* ``verdict_flips_against_planted_defect`` -- the release verdict is recomputed
  against a record with a defect planted in it, three ways, and the run fails if
  it does not flip. A gate whose verdict cannot go red is not a gate.

THERE IS NO NumPy BACKEND, DELIBERATELY
=======================================
The sibling curl gates carry one. It costs a SECOND TRANSCRIPTION of the kernel
-- the ghost rule, the Bloch rotation, the masks, the tails -- on the leg that is
supposed to be checking the first, and this family's bodies are the complex ones,
where the zero cross terms and the FMA arm are the whole question and NumPy
cannot express either. What a laptop CAN settle for this family is in
``meep_gpu/cuda_kernels/test_complex_no_pml.py``: the emitted text against the
certified template it is built from, the three float32 hazards measured in NumPy,
and the predicate's whole verdict table on real engine objects.

RUNNING IT
==========
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_complex_no_pml.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

The flush leg takes a cache directory WITHOUT the keep token, imports MEEP for
the host FTZ bits, and reads the flush-cut expansion record (the default for
that policy -- see :data:`PROBE_RECORD_BY_POLICY`)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/flush \\
        MEEP_GPU_SUBNORMAL_POLICY=flush python -u gate_cuda_complex_no_pml.py \\
        --subnormal-policy flush --import-meep-for-host-policy \\
        --out $OUT/flush/gate.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # the laptop can still import this file and read its tables
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402
from meep_gpu.cuda_kernels import complex_no_pml_kernels as family  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260820

#: Consecutive launches in the multi-step leg, the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the operands moving. Held fixed, a sub-step reads
#: the same numbers every launch and a 60-launch leg becomes a slow single-launch
#: leg. The same exact float32 scale is applied on both fixtures.
_ADVANCE = np.float32(0.97)

#: The DEFAULT seven-pattern expansion record, cut under ``keep``. ``update_P``
#: needs ``c8_mul_python_float_field_left`` and the four-pattern record does not
#: carry it -- see the module docstring of ``complex_no_pml_kernels``.
PROBE_RECORD = ("parity/meep_gpu/results/triton_welds_2026-08-19/"
                "unified_expansion/gate.json")

#: The same record cut under ``flush``. AN EXPANSION LICENCE IS POLICY-CONDITIONAL
#: (``complex_fields.POLICY_CONDITIONAL_LICENCE``), so the keep-cut record above
#: refuses a flush run by name and the flush leg was owed a record of its own
#: rather than a wider clause. Cut 2026-08-27 by ``gate_triton_unified_expansion``
#: under ``MEEP_GPU_SUBNORMAL_POLICY=flush`` on the GPU host; it classifies all seven
#: patterns exactly as the keep record does, which is a measurement and not an
#: assumption -- the keep leg of that same cut reproduced the 2026-08-19 table.
PROBE_RECORD_FLUSH = ("parity/meep_gpu/results/unified_expansion_2026-08-27/"
                      "flush/gate.json")

#: Which record a policy reads when ``--expansion-probe`` is not given. Spelled as
#: a table rather than as an if, so adding a policy without cutting its record is
#: a KeyError at startup and not a silent fall-through to another policy's bytes.
PROBE_RECORD_BY_POLICY: Dict[str, str] = {"keep": PROBE_RECORD,
                                          "flush": PROBE_RECORD_FLUSH}

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D", "update_E", "update_P")

#: 0.5 is exactly representable in float32 and 0.35 is not. ONLY THE SECOND can
#: distinguish a contracted expression from an uncontracted one, so the unguarded
#: control is scored at the inexact one.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``signed_zero`` is here because this family's whole complex multiply rests on
#: the zero cross terms, and ``uniform(-1, 1)`` provably draws no signed zero.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "signed_zero")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: The grids. Every wall kind appears, the Bloch phase appears and is ABSENT on
#: one spec (k = 0 must reduce to the plain complex path bit for bit, which is
#: what the ``ph`` flags being 0 is supposed to guarantee), and 1-D / 2-D / 3-D
#: shapes all appear. The corpus's own six rows are (35,32,41) and (25,25,25) and
#: (25,25,1), so a 3-D-only sweep would still miss the 2-D one.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "3d_phased_periodic", "cell": (8.0, 10.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic",) * 3, "k_point": (0.2, -0.35, 0.1),
     "absorber": "none"},
    {"label": "3d_unphased_periodic", "cell": (8.0, 10.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic",) * 3, "k_point": (0.0, 0.0, 0.0),
     "absorber": "none"},
    {"label": "3d_metallic", "cell": (8.0, 10.0, 12.0), "dimensions": 3,
     "boundaries": ("metallic",) * 3, "k_point": (0.0, 0.0, 0.0),
     "absorber": "none"},
    {"label": "3d_mixed_phased", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "periodic"),
     "k_point": (0.25, 0.0, -0.15), "absorber": "none"},
    # THE INERT LAYER. ``_pml_is_active`` is False for a zero-thickness PML, so
    # stepping takes the same tail and the predicate admits it. Swept because
    # "inert is the same as absent" is otherwise an inference about pml.py.
    {"label": "3d_phased_inert_layer", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "metallic"),
     "k_point": (0.3, 0.1, 0.0), "absorber": "inert_layer"},
    {"label": "2d_phased", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.2, 0.4, 0.0), "absorber": "none"},
    {"label": "2d_metallic_walls", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("metallic", "metallic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "absorber": "none"},
    {"label": "1d_phased", "cell": (0.0, 0.0, 24.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.35), "absorber": "none"},
)

#: The conductivity axis. ``none`` takes the plain curl tail and ``volume`` the
#: conductive one; the curl arm is READ OFF the fields by the shipped
#: classifier, never assumed from this label.
CONDUCTIVITIES: Tuple[str, ...] = ("none", "volume")

#: The dispersion axis. 0 exercises ``update_E``'s pole-free store (a real write,
#: not a no-op); 1 is the corpus's shape; 2 is what makes the LEFT-TO-RIGHT
#: subtraction order observable at all -- with one pole every order agrees.
POLE_COUNTS: Tuple[int, ...] = (0, 1, 2)

#: The sigma specialization, which is a COMPILE-TIME axis in the kernel. Both
#: must be swept or one of the two ``update_P`` kernels ships uncertified.
SIGMA_FORMS: Tuple[str, ...] = ("volume", "uniform")

COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")
DISPLACEMENT = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def _complex_host(shape, value_class: str, rng) -> np.ndarray:
    """One complex64 volume in a value class.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured by
    the sibling tranche before it was reasoned about: ``1j * im`` carries a real
    part of ``0.0 * im``, so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0``
    and DESTROYS every negative zero in the plane the zero cross terms act on.
    """
    if value_class == "uniform":
        real = rng.uniform(-1.0, 1.0, size=shape)
        imag = rng.uniform(-1.0, 1.0, size=shape)
    elif value_class == "subnormal_band":
        real = probe.subnormal_band_hosts(("re",), tuple(shape), rng)["re"]
        imag = probe.subnormal_band_hosts(("im",), tuple(shape), rng)["im"]
    elif value_class == "signed_zero":
        values = rng.uniform(-1.0, 1.0, size=shape)
        zeroed = rng.integers(0, 2, size=shape) == 0
        signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
        real = np.where(zeroed, signs, values)
        imag = rng.uniform(-1.0, 1.0, size=shape)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    host = np.empty(tuple(shape), dtype=np.complex64)
    host.real = np.asarray(real, dtype=np.float32)
    host.imag = np.asarray(imag, dtype=np.float32)
    return np.ascontiguousarray(host)


#: Every complex volume a fixture owns. Seeded whatever the sub-step reads, so
#: the two fixtures of one case are equal on every array either path could touch.
STATE: Tuple[str, ...] = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")


def build(xp, spec: Dict[str, Any], courant: float, conductivity: str,
          poles: int, sigma_form: str, value_class: str, seed: int):
    """One complete ``(fields, layer, grid)`` fixture, built from ``seed`` alone.

    Called TWICE per case with the same seed, which is what makes the oracle
    fixture and the kernel fixture the same problem rather than two draws.
    """
    rng = np.random.default_rng(seed)
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                dimensions=spec.get("dimensions", 3), xp=xp, courant=courant,
                k_point=tuple(spec["k_point"]))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = tuple(grid.shape)

    # THREE DISTINCT, INHOMOGENEOUS inverse-epsilon volumes drawn away from 1.0.
    # Against ones the multiply is invisible; against one shared volume the
    # component-aliasing defect is invisible; against a constant volume a
    # coefficient-index error is invisible. ``coefficient_profile`` is the floor
    # that refuses a case where any of those could not bite.
    inverse, epsilon = {}, {}
    for component in COMPONENTS:
        values = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
        inverse[component] = xp.asarray(np.ascontiguousarray(values))
        epsilon[component] = xp.asarray(
            np.ascontiguousarray((1.0 / values).astype(np.float32)))
    fields.set_epsilon_volumes(epsilon, inverse)

    if conductivity == "volume":
        # A GRADED conductivity, PER COMPONENT, which is the mapping form
        # ``mp.Absorber`` produces (fields.py:755-759: MEEP point-samples each
        # component at its own Yee position). One shared volume is also legal and
        # is what a UNIFORM sigma gives -- but under it the three condfac
        # pointers are the same array, so a component-aliasing defect is
        # unobservable and ``coefficient_profile`` refuses the case. Drawing three
        # distinct graded volumes is what makes the floor satisfiable at all.
        for setter, names in ((fields.set_d_conductivity, ("Dx", "Dy", "Dz")),
                              (fields.set_b_conductivity, ("Bx", "By", "Bz"))):
            setter({name: xp.asarray(np.ascontiguousarray(
                rng.uniform(0.05, 0.4, size=shape).astype(np.float32)))
                for name in names})

    for index in range(poles):
        if sigma_form == "volume":
            # PER COMPONENT, for the reason the conductivity is: ``normalize_sigma``
            # accepts a mapping, and one shared volume makes a component-aliasing
            # defect in update_P's sigma binding unobservable.
            sigma = {name: xp.asarray(np.ascontiguousarray(
                rng.uniform(0.1, 0.9, size=shape).astype(np.float32)))
                for name in COMPONENTS}
        else:
            sigma = float(0.2 + 0.13 * index)
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.4 * index, gamma=0.05 + 0.02 * index),
            sigma, grid, fields._field_dtype()))

    for name in STATE:
        getattr(fields, name)[...] = xp.asarray(
            _complex_host(shape, value_class, rng))
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = xp.asarray(
                _complex_host(shape, value_class, rng))
            state.P_prev[component][...] = xp.asarray(
                _complex_host(shape, value_class, rng))
        # THE SCRATCH IS SEEDED NONZERO. It is update_P's OUTPUT and starts as
        # whatever the previous rotation retired; left at zero, a kernel that
        # failed to write some cells would agree with an oracle that also wrote
        # nothing there.
        state._scratch[...] = xp.asarray(_complex_host(shape, value_class, rng))

    layer = (PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
             if spec["absorber"] == "inert_layer" else None)
    return fields, layer, grid


def state_slots(fields, sub_step: str) -> List[Tuple[str, Any]]:
    """Every array this sub-step may write, named by SLOT rather than by identity.

    Slot names are the point for ``update_P``: the rotation moves array OBJECTS
    between ``P``, ``P_prev`` and ``_scratch``, so a launcher that rotated wrongly
    would hold the right numbers in the wrong slots. Comparing by slot catches
    that; comparing by object identity could not.
    """
    if sub_step == "step_B":
        return [(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
    if sub_step == "step_D":
        return [(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    if sub_step == "update_E":
        return [(name, getattr(fields, name)) for name in COMPONENTS]
    out: List[Tuple[str, Any]] = []
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            out.append((f"P[{index}][{component}]", state.P[component]))
            out.append((f"P_prev[{index}][{component}]", state.P_prev[component]))
        out.append((f"_scratch[{index}]", state._scratch))
    return out


def slot_hosts(fields, sub_step: str) -> Dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(to_host(array)).copy()
            for name, array in state_slots(fields, sub_step)}


def fixture_hosts(fields, sub_step: str) -> Dict[str, np.ndarray]:
    """Every array either path READS or WRITES: the equality the two must start at."""
    out = {name: np.ascontiguousarray(to_host(getattr(fields, name))).copy()
           for name in STATE}
    out.update(slot_hosts(fields, sub_step))
    for component in COMPONENTS:
        out["inv_eps_" + component] = np.ascontiguousarray(
            to_host(fields.inverse_epsilon_for(component))).copy()
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        for reader, stem in ((fields.condfac_for, "condfac"),
                             (fields.condinv_for, "condinv")):
            volume = reader(name)
            if volume is not None:
                out[f"{stem}_{name}"] = np.ascontiguousarray(
                    to_host(volume)).copy()
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            sigma = state.sigma[component]
            if coverage.ade_sigma_is_volume(state, component):
                out[f"sigma[{index}][{component}]"] = np.ascontiguousarray(
                    to_host(sigma)).copy()
    return out


def words_differ(left: np.ndarray, right: np.ndarray) -> int:
    a = np.ascontiguousarray(left)
    b = np.ascontiguousarray(right)
    if a.dtype == np.complex64:
        a = a.view(np.float32)
        b = b.view(np.float32)
    return int(np.count_nonzero(a.ravel().view(np.uint32)
                                != b.ravel().view(np.uint32)))


def advance_sources(fields, sub_step: str) -> None:
    """Move the operands between launches, identically on both fixtures."""
    if sub_step == "step_B":
        names = COMPONENTS
    elif sub_step == "step_D":
        names = ("Bx", "By", "Bz")
    elif sub_step == "update_E":
        names = ("Dx", "Dy", "Dz")
    else:
        names = COMPONENTS  # update_P's drive is the stored E
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def _profile(volume) -> Dict[str, Any]:
    host = np.asarray(to_host(volume), dtype=np.float64)
    return {"min": float(host.min()), "max": float(host.max()),
            "spread": float(host.max() - host.min()),
            "max_abs_from_one": float(np.max(np.abs(host - 1.0)))}


def coefficient_profile(fields, sub_step: str, conductivity: str) -> Dict[str, Any]:
    """Every per-cell coefficient this sub-step's kernel INDEXES, profiled.

    THE FLOOR THIS ENFORCES is the one that makes a coefficient-INDEX error
    visible. A constant volume is indexed correctly by every index, so against
    one the whole class of "read cell j where cell i was meant" defects is
    unobservable and the case measures nothing about the binding. Three
    conditions, all required where the sub-step reads a coefficient at all:
    non-identity, non-constant, and pairwise distinct across components.
    """
    volumes: Dict[str, Any] = {}
    if sub_step == "update_E":
        for component in COMPONENTS:
            volumes["inv_eps_" + component] = fields.inverse_epsilon_for(component)
    elif sub_step in ("step_B", "step_D") and conductivity == "volume":
        targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
        for name in targets:
            volumes["condfac_" + name] = fields.condfac_for(name)
            volumes["condinv_" + name] = fields.condinv_for(name)
    elif sub_step == "update_P":
        for index, state in enumerate(fields.polarizations):
            for component in state.driven():
                if coverage.ade_sigma_is_volume(state, component):
                    volumes[f"sigma[{index}][{component}]"] = state.sigma[component]

    if not volumes:
        return {"volumes": {}, "meets_floor": True,
                "why": ("this sub-step and arm index no per-cell coefficient "
                        "volume, so there is no index to get wrong")}
    profiles = {name: _profile(volume) for name, volume in volumes.items()}
    identity = [n for n, p in profiles.items() if p["max_abs_from_one"] == 0.0]
    constant = [n for n, p in profiles.items() if p["spread"] == 0.0]
    hosts = {n: np.asarray(to_host(v), dtype=np.float32).ravel()
             for n, v in volumes.items()}
    names = sorted(hosts)
    shared = [f"{a}=={b}" for i, a in enumerate(names) for b in names[i + 1:]
              if np.array_equal(hosts[a].view(np.uint32), hosts[b].view(np.uint32))]
    return {"volumes": profiles, "identity": identity, "constant": constant,
            "shared": shared,
            "meets_floor": not identity and not constant and not shared}


def sources_are_distinct(hosts: Dict[str, np.ndarray],
                         sub_step: str) -> Dict[str, Any]:
    """The three source volumes differ pairwise, so a mis-binding has somewhere to show."""
    if sub_step == "step_B":
        names = COMPONENTS
    elif sub_step == "step_D":
        names = ("Bx", "By", "Bz")
    elif sub_step == "update_E":
        names = ("Dx", "Dy", "Dz")
    else:
        names = COMPONENTS
    pairs = [f"{a}=={b}" for i, a in enumerate(names) for b in names[i + 1:]
             if words_differ(hosts[a], hosts[b]) == 0]
    return {"checked": list(names), "identical_pairs": pairs,
            "meets_floor": not pairs}


def phase_is_live(grid) -> Dict[str, Any]:
    """At least one axis carries a Bloch factor that is not 1+0j.

    Without one, every rotation mutation in the battery is a null for a reason
    about the fixture rather than about the kernel -- and this family's curls
    inherit the rotation from the certified template, so a silent loss of it is
    exactly what a phased case is here to catch.
    """
    live = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is not None and complex(phase) != complex(1.0, 0.0):
            live.append(axis)
    return {"phased_axes": live, "has_bloch": bool(getattr(grid, "has_bloch", False)),
            "live": bool(live)}


def oracle_moved(before: Dict[str, np.ndarray],
                 after: Dict[str, np.ndarray]) -> float:
    total = changed = 0
    for name, values in before.items():
        a = np.ascontiguousarray(values).view(np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name]).view(np.float32).ravel().view(np.uint32)
        total += a.size
        changed += int(np.count_nonzero(a != b))
    return changed / total if total else 0.0


# ---------------------------------------------------------------------------
# The two paths
# ---------------------------------------------------------------------------

def run_oracle(fields, layer, sub_step: str) -> None:
    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    elif sub_step == "step_D":
        stepping.step_D(fields, layer)
    elif sub_step == "update_E":
        stepping.update_E(fields, layer)
    else:
        stepping.update_P(fields, layer)


def run_kernel(fields, grid, sub_step: str, arm: int, *,
               drive=None, poles=None, conductivity=None) -> None:
    if sub_step in ("step_B", "step_D"):
        family.step_complex_no_pml_curl(fields, sub_step, arm, grid,
                                        conductivity=conductivity)
    elif sub_step == "update_E":
        family.update_E_complex_no_pml_stored(fields, arm, poles=poles)
    else:
        family.update_P_complex_no_pml(fields, arm, drive=drive)
    cp.cuda.runtime.deviceSynchronize()


# ---------------------------------------------------------------------------
# The mutation battery
# ---------------------------------------------------------------------------

def _replace_once(source: str, old: str, new: str) -> Tuple[str, int]:
    count = source.count(old)
    return (source.replace(old, new), count) if count else (source, 0)


def _flip_tail_sign(source: str) -> Tuple[str, int]:
    """``target -= curl`` becomes ``target += curl``. Must be CAUGHT."""
    return _replace_once(source, "cf_sub(cf_load(f, idx), curl)",
                         "cf_add(cf_load(f, idx), curl)")


def _drop_tail_store(source: str) -> Tuple[str, int]:
    """The plain tail writes nothing. Must be CAUGHT."""
    return _replace_once(
        source, "    cf_store(f, idx, cf_sub(cf_load(f, idx), curl));",
        "    (void) curl;")


def _conductive_order(source: str) -> Tuple[str, int]:
    """``(f - curl) * condfac * condinv`` instead of the array path's order.

    MEASURED to change 400000 of 400000 float32 words on the host, so this must
    be CAUGHT on any live case.
    """
    return _replace_once(
        source,
        "    cf t = mul_field_left(cf_load(f, idx), condfac);\n"
        "    t = cf_sub(t, curl);\n"
        "    cf_store(f, idx, mul_field_left(t, condinv));",
        "    cf t = cf_sub(cf_load(f, idx), curl);\n"
        "    t = mul_field_left(t, condfac);\n"
        "    cf_store(f, idx, mul_field_left(t, condinv));")


def _conductive_swap_tables(source: str) -> Tuple[str, int]:
    """condfac and condinv exchanged. Must be CAUGHT."""
    return _replace_once(
        source,
        "    cf t = mul_field_left(cf_load(f, idx), condfac);\n"
        "    t = cf_sub(t, curl);\n"
        "    cf_store(f, idx, mul_field_left(t, condinv));",
        "    cf t = mul_field_left(cf_load(f, idx), condinv);\n"
        "    t = cf_sub(t, curl);\n"
        "    cf_store(f, idx, mul_field_left(t, condfac));")


def _conductive_planewise(source: str) -> Tuple[str, int]:
    """Scale the two planes instead of the zero-imaginary complex product.

    The natural implementation and the wrong one. Must be CAUGHT on the
    ``signed_zero`` class; on ``uniform`` it may legitimately be a null, which is
    why the leg plan carries a signed-zero spec.
    """
    return _replace_once(
        source,
        "    cf t = mul_field_left(cf_load(f, idx), condfac);",
        "    cf t = cf_load(f, idx); t.re = t.re * condfac; t.im = t.im * condfac;")


def _presum_poles(source: str) -> Tuple[str, int]:
    """Accumulate the poles and subtract once. Must be CAUGHT at two poles."""
    return _replace_once(
        source,
        "    cf s = cf_load(g, idx);\n"
        "    if (np > 0) s = cf_sub(s, cf_load(p0, idx));\n"
        "    if (np > 1) s = cf_sub(s, cf_load(p1, idx));",
        "    cf s = cf_load(g, idx);\n"
        "    cf acc = cf_zero();\n"
        "    if (np > 0) acc = cf_add(acc, cf_load(p0, idx));\n"
        "    if (np > 1) acc = cf_add(acc, cf_load(p1, idx));\n"
        "    if (np > 1) { s = cf_sub(s, acc); return s; }\n"
        "    if (np > 0) s = cf_sub(s, cf_load(p0, idx));")


def _drop_inverse_epsilon(source: str) -> Tuple[str, int]:
    """``E = (D - sum P)`` with no inv_eps. Must be CAUGHT."""
    out, n = _replace_once(
        source, "        inv_eps_0[idx]));", "        1.0f));")
    out, m = _replace_once(out, "        inv_eps_1[idx]));", "        1.0f));")
    out, k = _replace_once(out, "        inv_eps_2[idx]));", "        1.0f));")
    return out, min(n, m, k)


def _alias_inverse_epsilon(source: str) -> Tuple[str, int]:
    """Every component reads Ez's inverse epsilon -- the ``fields.inv_eps`` defect."""
    out, n = _replace_once(source, "        inv_eps_0[idx]));",
                           "        inv_eps_2[idx]));")
    out, m = _replace_once(out, "        inv_eps_1[idx]));",
                           "        inv_eps_2[idx]));")
    return out, min(n, m)


def _ade_regroup(source: str) -> Tuple[str, int]:
    """Right-associate the two additions. Must be CAUGHT."""
    return _replace_once(
        source,
        "    cf out = cf_add(mul_field_left(p, c_now), mul_coefficient_left(c_prev, q));\n"
        "    return cf_add(out, mul_coefficient_left(c_drive, mul_coefficient_left(s, w)));",
        "    return cf_add(mul_field_left(p, c_now), cf_add(\n"
        "        mul_coefficient_left(c_prev, q),\n"
        "        mul_coefficient_left(c_drive, mul_coefficient_left(s, w))));")


def _ade_distribute_drive(source: str) -> Tuple[str, int]:
    """``(c_drive * sigma) * w`` instead of ``c_drive * (sigma * w)``."""
    return _replace_once(
        source, "mul_coefficient_left(c_drive, mul_coefficient_left(s, w))",
        "mul_coefficient_left(c_drive * s, w)")


def _ade_swap_history(source: str) -> Tuple[str, int]:
    """P and P_prev exchanged. Must be CAUGHT."""
    return _replace_once(
        source, "        cf_load(p_now, idx), cf_load(p_prev, idx), cf_load(drive, idx),",
        "        cf_load(p_prev, idx), cf_load(p_now, idx), cf_load(drive, idx),")


#: The four DEAD-PRELUDE nulls. ``pml_apply`` and ``constitutive_apply`` are
#: compiled into every kernel here and CALLED BY NONE, so every one of these must
#: come back UNCAUGHT -- which is the measurement that this family's tails really
#: are ``target -= curl`` and the conductive triple rather than the split-field
#: recurrence, a claim that otherwise rests on reading the source.
def _dead_drop_fu_store(source: str) -> Tuple[str, int]:
    return _replace_once(source, "    cf_store(fu, idx, fu_new);", "")


def _dead_swap_pml_coefficients(source: str) -> Tuple[str, int]:
    return _replace_once(
        source, "float kms, float sinv, float kms_u, float sinv_u",
        "float sinv, float kms, float sinv_u, float kms_u")


def _dead_constitutive_order(source: str) -> Tuple[str, int]:
    return _replace_once(
        source,
        "    cf prev = cf_load(fw, idx);\n    cf_store(fw, idx, src);",
        "    cf_store(fw, idx, src);\n    cf prev = cf_load(fw, idx);")


def _dead_constitutive_flatten(source: str) -> Tuple[str, int]:
    return _replace_once(
        source,
        "    a = cf_add(a, mul_coefficient_left(kps, src));\n"
        "    a = cf_sub(a, mul_coefficient_left(kms, prev));",
        "    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src),\n"
        "                         mul_coefficient_left(kms, prev)));")


#: ``name -> (transform, must_be_caught, kernel keys, leg filter or None)``.
#:
#: THE LEG FILTER IS NOT A CONVENIENCE. A mutation scored on a leg where it is
#: PROVABLY inert comes back ESCAPED and reads exactly like a kernel defect.
#: ``conductive_planewise`` is the case that forced this: plane-wise
#: ``{re*c, im*c}`` and the zero-imaginary complex product agree EXACTLY on every
#: normal nonzero operand -- ``fma(re, c, (im*0.0)*-1.0)`` is ``re*c`` whenever
#: ``re*c`` is nonzero -- and separate only on signed zeros and on operands that
#: underflow. Scoring it on a ``uniform`` leg would be scoring a null.
#: ``presum_poles`` is the same shape: at one pole every order agrees.
MUTATIONS: Dict[str, Tuple[Any, bool, Tuple[str, ...], Any]] = {
    "flip_tail_sign": (_flip_tail_sign, True, ("step_B", "step_D"), None),
    "drop_tail_store": (_drop_tail_store, True, ("step_B", "step_D"), None),
    "conductive_order": (_conductive_order, True,
                         ("step_B_conductive", "step_D_conductive"), None),
    "conductive_swap_tables": (_conductive_swap_tables, True,
                               ("step_B_conductive", "step_D_conductive"), None),
    "conductive_planewise": (_conductive_planewise, True,
                             ("step_B_conductive", "step_D_conductive"),
                             lambda leg: leg[6] == "signed_zero"),
    "presum_poles": (_presum_poles, True, ("update_E",),
                     lambda leg: leg[3] >= 2),
    "drop_inverse_epsilon": (_drop_inverse_epsilon, True, ("update_E",), None),
    "alias_inverse_epsilon_to_Ez": (_alias_inverse_epsilon, True, ("update_E",),
                                    None),
    "ade_regroup": (_ade_regroup, True, ("update_P", "update_P_uniform"), None),
    "ade_distribute_drive": (_ade_distribute_drive, True,
                             ("update_P", "update_P_uniform"), None),
    "ade_swap_history": (_ade_swap_history, True,
                         ("update_P", "update_P_uniform"), None),
    # The four dead-prelude nulls, armed on EVERY kernel.
    "dead_drop_fu_store": (_dead_drop_fu_store, False,
                           tuple(family.KERNEL_KEYS), None),
    "dead_swap_pml_coefficients": (_dead_swap_pml_coefficients, False,
                                   tuple(family.KERNEL_KEYS), None),
    "dead_constitutive_order": (_dead_constitutive_order, False,
                                tuple(family.KERNEL_KEYS), None),
    "dead_constitutive_flatten": (_dead_constitutive_flatten, False,
                                  tuple(family.KERNEL_KEYS), None),
}

#: The HOST mutations: the tables a launch BINDS, corrupted at the pointer rather
#: than in the device text, so the defect is on the argument every other leg
#: shares and not on a rewritten string.
HOST_MUTATIONS: Tuple[str, ...] = (
    "reverse_conductivity_volumes",     # must be CAUGHT
    "alias_condinv_to_condfac",         # must be CAUGHT
    "reverse_pole_order",               # must be CAUGHT at two poles
    "bind_the_stored_E_twice",          # must be CAUGHT (update_P drive control)
    "identity_rebind",                  # must be UNCAUGHT
)
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("identity_rebind",)


def host_mutated_conductivity(name: str, fields, sub_step: str):
    """The six volumes ``step_complex_no_pml_curl`` binds, corrupted; or None."""
    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    condfac = [fields.condfac_for(target) for target in targets]
    condinv = [fields.condinv_for(target) for target in targets]
    if any(volume is None for volume in condfac + condinv):
        return None
    if name == "reverse_conductivity_volumes":
        return tuple(condfac[::-1] + condinv[::-1])
    if name == "alias_condinv_to_condfac":
        return tuple(condfac + condfac)
    if name == "identity_rebind":
        return tuple(condfac + condinv)
    return None


def host_mutated_poles(name: str, fields):
    """The pole bank ``update_E_complex_no_pml_stored`` binds, corrupted; or None."""
    order = family.poles_per_component(fields)
    banks = {component: [state.P[component] for state in states]
             for component, states in order.items()}
    if name == "reverse_pole_order":
        if all(len(bank) < 2 for bank in banks.values()):
            return None  # nothing to reorder at fewer than two poles
        return {component: list(reversed(bank))
                for component, bank in banks.items()}
    if name == "identity_rebind":
        return {component: list(bank) for component, bank in banks.items()}
    return None


def host_mutated_drive(name: str, fields):
    """``update_P``'s drive callable, corrupted; or None.

    ``bind_the_stored_E_twice`` binds Ex for every component. Without a layer the
    correct drive IS the stored E, so this is not the ``f_w`` control the PML
    family runs -- it is the COMPONENT control, and it is the one a no-PML gate
    can actually arm.
    """
    if name == "bind_the_stored_E_twice":
        return lambda _component: fields.Ex
    if name == "identity_rebind":
        return fields.drive_field
    return None


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(spec: Dict[str, Any], sub_step: str, conductivity: str, poles: int,
             sigma_form: str, courant: float, value_class: str, guard: str,
             arm: int, host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One configuration, built twice: the array path on one, the kernel on the other."""
    started = time.time()
    # SEEDED FROM A DIGEST, NOT FROM hash(). Python salts hash() of a tuple
    # containing strings with PYTHONHASHSEED, so a hash-seeded gate draws a
    # different fixture every process and a failing case cannot be replayed.
    seed = SEED + int.from_bytes(hashlib.sha256(
        f"{spec['label']}|{sub_step}|{conductivity}|{poles}|{sigma_form}|"
        f"{courant}|{value_class}".encode()).digest()[:4], "big")

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step,
        "conductivity": conductivity, "poles": poles, "sigma_form": sigma_form,
        "absorber": spec["absorber"], "courant": courant,
        "value_class": value_class, "guard": guard, "seed": seed,
        "host_mutation": host_mutation, "k_point": list(spec["k_point"]),
        "boundaries": list(spec["boundaries"]),
    }

    oracle_fixture = build(cp, spec, courant, conductivity, poles, sigma_form,
                           value_class, seed)
    kernel_fixture = build(cp, spec, courant, conductivity, poles, sigma_form,
                           value_class, seed)
    fields_o, layer_o, grid_o = oracle_fixture
    fields_k, layer_k, grid_k = kernel_fixture
    case["shape"] = [int(n) for n in grid_o.shape]

    if sub_step == "update_P" and poles == 0:
        case["skipped"] = ("no polarization is registered, so update_P is a no-op "
                           "and there is no sub-step to take over")
        case["seconds"] = time.time() - started
        return case

    # THE FLOOR THAT MAKES THE COMPARISON A COMPARISON.
    before_o = fixture_hosts(fields_o, sub_step)
    before_k = fixture_hosts(fields_k, sub_step)
    disagree = sorted(name for name in before_o
                      if words_differ(before_o[name], before_k[name]))
    case["fixtures_agree"] = {"arrays": len(before_o), "disagreeing": disagree,
                              "meets_floor": not disagree}
    if disagree:
        case["skipped"] = (f"the two fixtures start different on {disagree[:3]}; "
                           f"this case would compare two problems")
        case["seconds"] = time.time() - started
        return case

    case["operand_census"] = operand_census(
        {name: np.ascontiguousarray(values).view(np.float32)
         for name, values in before_o.items()
         if np.asarray(values).dtype == np.complex64})

    # THE PREDICATE'S ANSWER, recorded rather than acted on. The gate exists to
    # decide whether it should be believed, so it must not gate the measurement.
    licence = _licence()
    if sub_step in ("step_B", "step_D"):
        covered, reason = family.covers_complex_no_pml_curl(
            fields_o, layer_o, grid_o, sub_step, license=licence,
            subnormal_policy=_POLICY_NAME)
        try:
            case["curl_arm"] = family.complex_no_pml_curl_arm(fields_o, sub_step)
        except Exception as exc:  # noqa: BLE001
            case["curl_arm"] = f"UNCLASSIFIED {type(exc).__name__}"
    elif sub_step == "update_E":
        covered, reason = family.covers_complex_no_pml_stored_e(
            fields_o, layer_o, grid_o, license=licence,
            subnormal_policy=_POLICY_NAME)
    else:
        covered, reason = family.covers_complex_no_pml_ade_update_p(
            fields_o, layer_o, grid_o, license=licence,
            subnormal_policy=_POLICY_NAME)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    profile = coefficient_profile(fields_o, sub_step, conductivity)
    case["coefficient_profile"] = profile
    if not profile["meets_floor"]:
        case["skipped"] = ("a coefficient volume is the identity, is constant, or "
                           "is shared between components; a coefficient-index "
                           "error would be invisible on this case")
        case["seconds"] = time.time() - started
        return case

    distinct = sources_are_distinct(before_o, sub_step)
    case["sources_are_distinct"] = distinct
    if not distinct["meets_floor"]:
        case["skipped"] = ("two source volumes are identical; a component-binding "
                           "defect would be invisible on this case")
        case["seconds"] = time.time() - started
        return case

    case["phase_is_live"] = phase_is_live(grid_o)

    # Leg 1: the oracle.
    frozen = slot_hosts(fields_o, sub_step)
    run_oracle(fields_o, layer_o, sub_step)
    reference = slot_hosts(fields_o, sub_step)
    moved = oracle_moved(frozen, reference)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, on the twin fixture.
    kwargs: Dict[str, Any] = {}
    if host_mutation is not None:
        if sub_step in ("step_B", "step_D"):
            kwargs["conductivity"] = host_mutated_conductivity(
                host_mutation, fields_k, sub_step)
            if kwargs["conductivity"] is None:
                case["skipped"] = (f"host mutation {host_mutation} has nothing to "
                                   f"corrupt on this case")
                case["seconds"] = time.time() - started
                return case
        elif sub_step == "update_E":
            kwargs["poles"] = host_mutated_poles(host_mutation, fields_k)
            if kwargs["poles"] is None:
                case["skipped"] = (f"host mutation {host_mutation} has nothing to "
                                   f"corrupt on this case")
                case["seconds"] = time.time() - started
                return case
        else:
            kwargs["drive"] = host_mutated_drive(host_mutation, fields_k)
            if kwargs["drive"] is None:
                case["skipped"] = (f"host mutation {host_mutation} has nothing to "
                                   f"corrupt on this case")
                case["seconds"] = time.time() - started
                return case
    run_kernel(fields_k, grid_k, sub_step, arm, **kwargs)

    produced = slot_hosts(fields_k, sub_step)
    parts = {name: bit_compare(reference[name], produced[name])
             for name in reference}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step, on unmutated legs only. 60 launches of a deliberately
    # wrong kernel measure how a wrong answer compounds, which this gate does not ask.
    if host_mutation is None:
        multi_o = build(cp, spec, courant, conductivity, poles, sigma_form,
                        value_class, seed)
        multi_k = build(cp, spec, courant, conductivity, poles, sigma_form,
                        value_class, seed)
        for _ in range(MULTI_STEP_BUDGET):
            run_oracle(multi_o[0], multi_o[1], sub_step)
            advance_sources(multi_o[0], sub_step)
        for _ in range(MULTI_STEP_BUDGET):
            run_kernel(multi_k[0], multi_k[2], sub_step, arm)
            advance_sources(multi_k[0], sub_step)
        oracle_multi = slot_hosts(multi_o[0], sub_step)
        kernel_multi = slot_hosts(multi_k[0], sub_step)
        case["multi_step"] = combine({
            name: bit_compare(oracle_multi[name], kernel_multi[name])
            for name in oracle_multi})
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str, guard: str = "fmad_false"):
    """The plan, assembled PER SUB-STEP rather than as one blind cross product.

    A full cross product of spec x conductivity x poles x sigma x courant x class
    is 1824 cases, and most of them are redundant: the pole count cannot reach a
    curl, and the sigma specialization cannot reach anything but ``update_P``. The
    axes each sub-step actually has are swept exhaustively; the ones it does not
    have are pinned at a value chosen to be a CONTROL rather than a default --
    the curls run with a pole registered on their conductive leg, so "dispersion
    does not reach the curl" is measured rather than assumed, and ``update_E``
    runs with and without a conductivity, so "the conductivity is read in
    _apply_curl and nowhere else" is measured too.

    THE UNGUARDED CONTROL RUNS AT THE INEXACT COURANT ONLY. 0.5 is exactly
    representable and cannot separate a contracted expression from an
    uncontracted one, so an unguarded leg at 0.5 costs a case and answers nothing.
    """
    specs = SPECS if product == "full" else SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    if guard != "fmad_false":
        courants = (INEXACT_COURANT,)
    out = []
    for spec in specs:
        for courant in courants:
            for value_class in classes:
                for conductivity in CONDUCTIVITIES:
                    poles = 1 if conductivity == "volume" else 0
                    for sub_step in ("step_B", "step_D"):
                        out.append((spec, sub_step, conductivity, poles,
                                    "volume", courant, value_class))
                for poles in (POLE_COUNTS if product == "full" else (1,)):
                    conductivity = "volume" if poles == 1 else "none"
                    out.append((spec, "update_E", conductivity, poles, "volume",
                                courant, value_class))
                plan_P = (((1, "volume"), (1, "uniform"), (2, "volume"))
                          if product == "full" else ((1, "volume"),))
                for poles, sigma_form in plan_P:
                    conductivity = "volume" if poles == 1 else "none"
                    out.append((spec, "update_P", conductivity, poles, sigma_form,
                                courant, value_class))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, product: str, guard: str,
              arm: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product, guard)
    for index, entry in enumerate(plan, start=1):
        spec, sub_step, conductivity, poles, sigma_form, courant, value_class = entry
        case = one_case(spec, sub_step, conductivity, poles, sigma_form, courant,
                        value_class, guard, arm)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"cond={conductivity} np={poles} SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"cond={conductivity} np={poles} sigma={sigma_form} c={courant} "
            f"{value_class} shape={case['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={case['single_launch']['differing_floats']}"
            f"/{case['single_launch']['total_floats']} "
            f"({case['seconds']:.1f} s)")
    return cases


#: The legs a mutation is scored on. CHOSEN, not sliced off the front of the case
#: product: a leg taken from the head landed entirely on unconductive controls in
#: a sibling gate, so the conductive-only mutation was skipped on every leg and
#: scored 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel one.
MUTATION_LEGS: Tuple[Tuple[str, str, str, int, str, float, str], ...] = (
    # PLAIN curls, two specs each so one degenerate grid cannot silence a leg.
    ("3d_phased_periodic", "step_B", "none", 0, "volume", INEXACT_COURANT, "uniform"),
    ("3d_phased_periodic", "step_D", "none", 0, "volume", INEXACT_COURANT, "uniform"),
    ("3d_metallic", "step_B", "none", 0, "volume", INEXACT_COURANT, "uniform"),
    ("3d_metallic", "step_D", "none", 0, "volume", INEXACT_COURANT, "uniform"),
    # CONDUCTIVE curls.
    ("3d_mixed_phased", "step_B", "volume", 1, "volume", INEXACT_COURANT, "uniform"),
    ("3d_mixed_phased", "step_D", "volume", 1, "volume", INEXACT_COURANT, "uniform"),
    ("2d_phased", "step_B", "volume", 1, "volume", INEXACT_COURANT, "uniform"),
    ("2d_phased", "step_D", "volume", 1, "volume", INEXACT_COURANT, "uniform"),
    # THE SIGNED-ZERO CONDUCTIVE LEGS, without which conductive_planewise has no
    # case that can separate the zero-imaginary product from a plane-wise scale:
    # the two agree exactly on every normal nonzero operand.
    ("3d_metallic", "step_B", "volume", 1, "volume", INEXACT_COURANT, "signed_zero"),
    ("3d_metallic", "step_D", "volume", 1, "volume", INEXACT_COURANT, "signed_zero"),
    ("3d_phased_periodic", "step_B", "volume", 1, "volume", INEXACT_COURANT,
     "signed_zero"),
    ("3d_phased_periodic", "step_D", "volume", 1, "volume", INEXACT_COURANT,
     "signed_zero"),
    # update_E. TWO POLES appear twice, without which presum_poles and
    # reverse_pole_order are nulls for a reason about the fixture.
    ("3d_phased_periodic", "update_E", "none", 2, "volume", INEXACT_COURANT,
     "uniform"),
    ("3d_metallic", "update_E", "volume", 2, "volume", INEXACT_COURANT, "uniform"),
    ("3d_mixed_phased", "update_E", "volume", 1, "volume", INEXACT_COURANT,
     "uniform"),
    ("2d_phased", "update_E", "none", 0, "volume", INEXACT_COURANT, "uniform"),
    # update_P, both sigma specializations and both pole counts.
    ("3d_phased_periodic", "update_P", "none", 2, "volume", INEXACT_COURANT,
     "uniform"),
    ("3d_mixed_phased", "update_P", "volume", 1, "volume", INEXACT_COURANT,
     "uniform"),
    ("3d_mixed_phased", "update_P", "volume", 1, "uniform", INEXACT_COURANT,
     "uniform"),
    ("3d_metallic", "update_P", "none", 2, "uniform", INEXACT_COURANT, "uniform"),
)


def _spec_by_label(label: str) -> Dict[str, Any]:
    for spec in SPECS:
        if spec["label"] == label:
            return spec
    raise KeyError(label)


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    single = not case["single_launch"]["bit_identical"]
    multi = case.get("multi_step", {}).get("bit_identical")
    return bool(single or multi is False)


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    ran = [c for c in legs if not c.get("skipped")]
    caught = [c for c in ran if leg_caught(c)]
    if not ran:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = "CAUGHT" if len(caught) == len(ran) else "ESCAPED"
    else:
        verdict = "NULL CONFIRMED" if not caught else "NULL BROKEN"
    return {"ran": len(ran), "caught": len(caught), "verdict": verdict}


def _legs_for(kernel_key: str, leg_filter=None):
    """Which planned legs exercise the kernel a mutation was applied to."""
    out = []
    for entry in MUTATION_LEGS:
        if leg_filter is not None and not leg_filter(entry):
            continue
        label, sub_step, conductivity, poles, sigma_form, courant, value_class = entry
        if sub_step in ("step_B", "step_D"):
            wanted = sub_step + ("_conductive" if conductivity == "volume" else "")
        elif sub_step == "update_E":
            wanted = "update_E"
        else:
            wanted = "update_P" if sigma_form == "volume" else "update_P_uniform"
        if wanted == kernel_key:
            out.append(entry)
    return out


def run_source_mutations(results: Dict[str, Any], out_path: str,
                         arm: int) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes, because a leg reporting a pass for a mutation it never applied is worse
    than no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    out: Dict[str, Any] = {}
    try:
        for name, (transform, must_catch, keys, leg_filter) in MUTATIONS.items():
            for key in keys:
                original = family.kernel_source(key, arm)
                mutated, sites = transform(original)
                tag = f"{key}:{name}"
                legs_plan = _legs_for(key, leg_filter)
                if sites == 0 or mutated == original:
                    out[tag] = {"armed": False, "sites": sites,
                                "must_be_caught": must_catch,
                                "why": (f"matched {sites} site(s) and changed "
                                        f"nothing in {family.kernel_name(key)}")}
                    log(f"[src-mut] {tag}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                family.set_kernel_source(key, arm, mutated)
                family.clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = [one_case(_spec_by_label(entry[0]), entry[1], entry[2],
                                 entry[3], entry[4], entry[5], entry[6],
                                 "fmad_false", arm)
                        for entry in legs_plan]
                family.set_kernel_source(key, arm, None)
                family.clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, must_catch)
                out[tag] = dict(scored, armed=True, sites=sites,
                                kernel=family.kernel_name(key),
                                must_be_caught=must_catch,
                                legs_planned=len(legs_plan),
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    out[tag]["verdict"] = "UNACCOUNTED"
                    out[tag]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg measured the SHIPPED kernel")
                log(f"[src-mut] {tag}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[tag]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for key in family.KERNEL_KEYS:
            family.set_kernel_source(key, arm, None)
        family.clear_kernel_cache()
    return out


def run_host_mutations(results: Dict[str, Any], out_path: str,
                       arm: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in HOST_MUTATIONS:
        legs = [one_case(_spec_by_label(entry[0]), entry[1], entry[2], entry[3],
                         entry[4], entry[5], entry[6], "fmad_false", arm,
                         host_mutation=name)
                for entry in MUTATION_LEGS]
        scored = _score(legs, name not in NULL_HOST_MUTATIONS)
        out[name] = dict(scored, cases=legs,
                         must_be_caught=name not in NULL_HOST_MUTATIONS)
        log(f"[host-mut] {name}: caught {scored['caught']}/{scored['ran']} "
            f"-> {scored['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = case["sub_step"]
        if case["sub_step"] in ("step_B", "step_D"):
            key += "::" + ("conductive" if case["conductivity"] == "volume"
                           else "plain")
        elif case["sub_step"] == "update_P":
            key += "::" + case["sigma_form"] + "_sigma"
        entry = arms.setdefault(key, {
            "cases": 0, "single_identical": 0, "multi_cases": 0,
            "multi_identical": 0, "differing_floats": 0, "labels": set(),
            "value_classes": set(), "absorbers": set(), "courants": set()})
        entry["cases"] += 1
        entry["labels"].add(case["label"])
        entry["value_classes"].add(case["value_class"])
        entry["absorbers"].add(case["absorber"])
        entry["courants"].add(case["courant"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        else:
            entry["differing_floats"] += case["single_launch"]["differing_floats"]
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
    for entry in arms.values():
        for field in ("labels", "value_classes", "absorbers", "courants"):
            entry[field] = sorted(entry[field])
        entry["all_identical"] = (
            entry["single_identical"] == entry["cases"]
            and entry["multi_identical"] == entry["multi_cases"])

    reasons: List[str] = []

    # EVERY ARM IS REQUIRED. A run that swept only the plain curl would release on
    # four of the twenty reachable corpus slots and say nothing about the sixteen
    # the conductive, stored-E and ADE arms carry.
    for required in ("step_B::plain", "step_D::plain", "step_B::conductive",
                     "step_D::conductive", "update_E", "update_P::volume_sigma",
                     "update_P::uniform_sigma"):
        if required not in arms:
            reasons.append(f"the sweep contained no {required} case")
    for key, entry in sorted(arms.items()):
        if entry["single_identical"] != entry["cases"]:
            reasons.append(
                f"{key}: single-launch divergence on "
                f"{entry['cases'] - entry['single_identical']} of {entry['cases']}")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(
                f"{key}: multi-step divergence on "
                f"{entry['multi_cases'] - entry['multi_identical']} of "
                f"{entry['multi_cases']}")

    # BOTH ABSORBER SHAPES, or "an inert layer steps like no layer" stays an
    # inference about pml.py rather than a measurement.
    absorbers = {c["absorber"] for c in scored}
    for shape in ("none", "inert_layer"):
        if shape not in absorbers:
            reasons.append(f"no case ran with absorber={shape}")

    # BOTH COURANTS, and the signed-zero class, which is the only one that can
    # separate the zero cross terms from a plane-wise scale.
    if {c["courant"] for c in scored} != set(COURANTS):
        reasons.append("both courants are required; only one was scored")
    if "signed_zero" not in {c["value_class"] for c in scored}:
        reasons.append("no signed_zero case was scored; the zero cross terms are "
                       "unmeasured and they are what the complex product rests on")
    if "subnormal_band" not in {c["value_class"] for c in scored}:
        reasons.append("no subnormal_band case was scored; the float32 subnormal "
                       "policy is unmeasured on this family")

    # NON-VACUITY, aggregated: at least one scored case must carry subnormal
    # operands and at least one a negative zero, or the classes are names only.
    census = [c.get("operand_census", {}) for c in scored]
    if not any(entry.get("subnormals", 0) for entry in census):
        reasons.append("no scored case carried a subnormal operand")
    if not any(entry.get("negative_zeros", 0) for entry in census):
        reasons.append("no scored case carried a negative zero")

    # A PHASED CASE MUST HAVE BEEN LIVE, or the Bloch rotation this family
    # inherits from the certified template is untested here.
    if not any(c.get("phase_is_live", {}).get("live") for c in scored):
        reasons.append("no scored case carried a live Bloch phase")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is "
                           f"not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for tag, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {tag} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {tag} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")

    if results.get("product") == "full" and results.get("skip_mutations"):
        reasons.append("--skip-mutations: no mutation evidence in this record")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded_identical = {
        (c["label"], c["sub_step"], c["conductivity"], c["poles"],
         c["sigma_form"], c["courant"], c["value_class"])
        for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["conductivity"], c["poles"],
                      c["sigma_form"], c["courant"], c["value_class"])
                  in guarded_identical]
    control_diverged = [c for c in comparable
                        if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where "
                    "the guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this family"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "arms": arms,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("the complex/no-absorber CUDA family -- the plain and "
                  "conductive curl pairs, the stored-E update_E and the two "
                  "update_P sigma specializations -- is byte-identical to "
                  "stepping.step_B / step_D / update_E / update_P per sub-step "
                  f"from one frozen state, at one launch and at "
                  f"{MULTI_STEP_BUDGET}, over phased and unphased periodic walls, "
                  "metallic walls, 3-D / 2-D / 1-D shapes, an absent absorber and "
                  "an inert layer, zero / one / two poles, uniform and volume "
                  "sigma, both float32 subnormal policies, three value classes "
                  "and both courants"),
        "does_not_claim": [
            "the OFF-DIAGONAL complex update_E. Two of the six corpus rows carry "
            "has_offdiagonal_epsilon and their update_E is MEEP's tensor row "
            "product; complex_no_pml_kernels refuses it by name and this gate "
            "never runs it. Those are 2 of the family's 22 slots.",
            "nothing dispatches these kernels; this gate licenses a predicate, "
            "not a wiring",
            "a mirror fold, a cylindrical grid, BFAST, special_kz and chi2/chi3 "
            "are refused by inherited clauses and untested here",
            "a sub-step whose three targets disagree about conductivity is "
            "refused, not measured",
            "stores_E False: no corpus row in this family is in that state and no "
            "derived-E arm was built",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the verdict against a record with a defect planted in it.

    A GATE WHOSE VERDICT CANNOT GO RED IS NOT A GATE. Three independent defects
    are planted, one at a time, into a COPY of the finished record: a scored case
    flipped to diverging, a must-be-caught source mutation flipped to ESCAPED,
    and a must-be-caught host mutation flipped to ESCAPED. Each must make
    ``released`` False on its own. A plant the record cannot carry is reported as
    ``applicable: False`` rather than passed.
    """
    out: Dict[str, Any] = {"plants": [], "all_flipped": True, "inapplicable": []}
    base = summarize(results)

    def plant(name: str, mutate, why_absent: str) -> None:
        copy_of = copy.deepcopy(results)
        applied = mutate(copy_of)
        verdict = summarize(copy_of) if applied else None
        flipped = bool(applied) and not verdict["released"]
        out["plants"].append({
            "plant": name, "applicable": bool(applied),
            "why_not_applicable": None if applied else why_absent,
            "released_after_plant": None if verdict is None else verdict["released"],
            "flipped": flipped,
            "first_reason": None if verdict is None or verdict["released"]
            else verdict["reasons"][0][:160]})
        if not applied:
            out["inapplicable"].append(name)
        elif not flipped:
            out["all_flipped"] = False

    def flip_a_case(record) -> bool:
        for case in record.get("sweep", {}).get("fmad_false", []):
            if (not case.get("skipped")
                    and case["single_launch"]["bit_identical"]):
                case["single_launch"]["bit_identical"] = False
                return True
        return False

    def flip_a_source_mutation(record) -> bool:
        for leg in record.get("source_mutations", {}).values():
            if leg.get("armed") and leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "ESCAPED"
                leg["caught"] = 0
                return True
        return False

    def flip_a_host_mutation(record) -> bool:
        for leg in record.get("host_mutations", {}).values():
            if leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "ESCAPED"
                leg["caught"] = 0
                return True
        return False

    plant("a_scored_case_diverges", flip_a_case,
          "no scored case in the record was bit-identical to begin with")
    plant("a_must_be_caught_source_mutation_escapes", flip_a_source_mutation,
          "the record carries no armed CAUGHT source mutation")
    plant("a_must_be_caught_host_mutation_escapes", flip_a_host_mutation,
          "the record carries no CAUGHT host mutation")
    out["released_unplanted"] = base["released"]
    return out


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# The expansion licence
# ---------------------------------------------------------------------------

_LICENCE: Optional[Dict[str, Any]] = None
_POLICY_NAME: Optional[str] = None


def _licence() -> Optional[Dict[str, Any]]:
    return _LICENCE


def load_licence(policy: str, record_path: Optional[str] = None) -> Dict[str, Any]:
    """The seven-pattern verdict, and the arm it binds.

    THIS GATE DOES NOT DERIVE AN ARM. ``expansion_license`` is the arbiter and a
    second spelling of it is one too many; what happens here is that the record is
    read, the licence is run, and the verdict's shape is checked by the SHIPPED
    predicate clause rather than by a copy of it.

    ``record_path`` overrides :data:`PROBE_RECORD_BY_POLICY`; it is a path, either
    absolute or relative to ``the repository root``. THE POLICY STILL DECIDES THE DEFAULT and
    the override does not weaken the comparison: a record handed in by hand is put
    through the same ``expansion_policy_reasons`` clause as a defaulted one, so
    pointing a flush run at the keep record refuses exactly as it did before this
    argument existed. The sha256 is recorded because the path names a file and the
    verdict rests on its bytes.
    """
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    relative = record_path or PROBE_RECORD_BY_POLICY[policy]
    path = (relative if os.path.isabs(relative)
            else os.path.join(_REPO_API, relative))
    with open(path, "rb") as handle:
        raw = handle.read()
    record = json.loads(raw.decode("utf-8"))
    verdict = complex_fields.expansion_license(record)
    verdict.setdefault("patterns", record.get("patterns"))
    reasons = list(complex_fields.expansion_policy_reasons(record, policy))
    verdict["policy_reasons"] = reasons
    verdict["record"] = relative
    verdict["record_sha256"] = hashlib.sha256(raw).hexdigest()
    verdict["record_subnormal_policy"] = record.get("subnormal_policy")
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _LICENCE, _POLICY_NAME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--expansion-probe", default=None, help=(
        "the seven-pattern expansion record to read, overriding the per-policy "
        "default in PROBE_RECORD_BY_POLICY. Path, absolute or relative to "
        "apps/api. The policy comparison is unchanged: a record cut under the "
        "other policy refuses here whether it arrived by default or by hand."))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    _POLICY_NAME = args.subnormal_policy

    results: Dict[str, Any] = {
        "gate": "cuda_complex_no_pml",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": "cuda",
        "product": args.product,
        "subnormal_policy": args.subnormal_policy,
        "skip_mutations": bool(args.skip_mutations),
        "question": ("are the six complex/no-absorber CUDA kernels byte-identical "
                     "to stepping.step_B / step_D / update_E / update_P per "
                     "sub-step -- including the CONDUCTIVE curl arm, which four of "
                     "the six corpus rows route to because they carry a material "
                     "conductivity on every one of Bx..Dz?"),
        "kernels": {key: family.kernel_name(key) for key in family.KERNEL_KEYS},
        "recon": ("parity/meep_gpu/results/"
                  "cuda_complex_no_pml_recon_2026-08-20"),
        "corpus_digest": family.corpus_digest(),
    }

    if cp is None:
        log("[fatal] CuPy did not import; this gate has no NumPy backend by design")
        results["status"] = "refused: no CuPy"
        save(results, args.out)
        return 2

    licence = load_licence(args.subnormal_policy, args.expansion_probe)
    results["expansion_licence"] = {
        k: v for k, v in licence.items() if k != "record"}
    results["expansion_probe_record"] = licence["record"]
    results["expansion_probe_given"] = args.expansion_probe
    refusal = coverage.complex_expansion_refusal(licence, args.subnormal_policy)
    if refusal is not None or licence["policy_reasons"]:
        log(f"[fatal] the expansion licence refuses this run: {refusal!r} "
            f"{licence['policy_reasons'][:1]}")
        results["status"] = f"refused: {refusal or licence['policy_reasons'][0]}"
        save(results, args.out)
        return 2
    _LICENCE = licence
    arm = int(licence["expansion"])
    results["expansion_arm"] = {"code": arm,
                                "name": complex_emitter.EXPANSION_NAMES[arm]}
    results["kernel_source_sha256"] = {
        family.kernel_name(key): hashlib.sha256(
            family.kernel_source(key, arm).encode("utf-8")).hexdigest()
        for key in family.KERNEL_KEYS}

    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
        args.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, args.out)

    for guard, options, _primary in GUARD_SETS:
        family._COMPILE_OPTIONS = tuple(options)
        family.clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.product, guard, arm)

    family._COMPILE_OPTIONS = ("--fmad=false",)
    family.clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, arm)
        run_source_mutations(results, args.out, arm)

    results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["verdict_flips_against_planted_defect"] = \
        verdict_flips_against_planted_defect(results)
    if not results["verdict_flips_against_planted_defect"]["all_flipped"]:
        results["summary"]["released"] = False
        results["summary"]["reasons"].append(
            "the release verdict did not flip against every planted defect; a "
            "verdict that cannot go red is not a verdict")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, entry in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {entry['single_identical']}/"
            f"{entry['cases']} multi {entry['multi_identical']}/"
            f"{entry['multi_cases']} diff={entry['differing_floats']}")
    for plant in results["verdict_flips_against_planted_defect"]["plants"]:
        log(f"[verdict]   plant {plant['plant']}: "
            f"applicable={plant['applicable']} flipped={plant['flipped']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
