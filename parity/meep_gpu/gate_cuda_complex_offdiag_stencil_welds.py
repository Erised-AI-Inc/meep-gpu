"""Byte-identity gate for the two hand-CUDA COMPLEX SCRATCH-OUTPUT stencil welds.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/folded_complex_offdiag_fused_electric_pair
.py`` and ``complex_no_pml_offdiag_fused_electric_pair.py`` occupy the LAST two
``D_to_E`` cells the hand-CUDA board records STENCIL-BLOCKED::

    "the constitutive half reads the curl half's IN-PLACE output at a cell this
     thread does not own. In one launch that cell is written by another block, and
     CUDA offers no grid-wide barrier inside an ordinary launch."

That verdict was MEASURED and it is true of the weld it describes. The products under
test are a different weld: the curl half writes ``D_new``/``fu_new`` to a LAUNCH-LOCAL
SCRATCH, the constitutive half RECOMPUTES each foreign cell from PRE-LAUNCH state
through the same ``__device__`` function, and the launcher rotates the ``D``/``fu``
bindings only after the launch returns. So this gate does not ask whether the recorded
refusal was wrong; it asks whether a launch with neither of that refusal's two
premises is byte-identical to the array path.

``gate_cuda_offdiag_stencil_welds.py`` asked that question of the FLOAT32 pair on
2026-09-02 and this file is its COMPLEX64 twin. It is not a re-parameterisation:
under complex storage every multiply in the resolution changes, and the change is
exactly the class a float32 gate cannot see.

=============================================================================
THE THREE COMPLEX ARMS, AND THE VALUE CLASS THAT MAKES THEM DECIDABLE
=============================================================================

1. **THE PARITY IS A FULL COMPLEX PRODUCT, NOT A WORD SCALE.** The array path spells
   a mirror parity ``phase * plane`` with ``phase`` a Python int and ``plane``
   complex64 (stepping.py:1451, :1529-1532); the backend promotes the scalar and runs
   its complex multiply loop, whose zero cross terms carry the OTHER word's sign into
   an addend. Armed as ``parity_as_word_scale``.
2. **AN EVEN MIRROR MAY NOT BE SKIPPED.** ``mul_coefficient_left(+1.0f, z)`` is
   ``fma(1, z.re, (0*z.im) * -1.0f)``, which turns ``re = -0.0`` with a negative
   imaginary part into ``+0.0``. Armed as ``parity_skipped_at_plus_one``.
3. **THE WALL CLEAR IS BOTH WORDS.** ``stepping._zero_metal`` assigns the INTEGER 0
   to a complex64 array (:2247), which is ``(+0.0f, +0.0f)``. Armed as
   ``clear_real_word_only`` -- **the arm the host probe could not express**, because
   NumPy has no spelling of a real-word-only clear that is not simply the truth. Here
   the store is two explicit float32 writes and the arm is real.

NONE OF THE THREE CAN FIRE ON AN ALL-NORMAL FIXTURE, and a battery that shipped them
against ``uniform(-1, 1)`` would report three UNCAUGHT holes that are really three
vacuous arms. So this gate carries a THIRD VALUE CLASS, ``adversarial_words``: every
stored complex volume tiled from ``probe_cuda_complex_offdiag_scratch_weld
.ADVERSARIAL_WORDS`` -- exact +0.0 and -0.0 real words beside negative imaginary
words, both signed subnormals, and normals -- so a signed zero sits under every
redirect source, every cleared plane and every reflect row. The catalogue is IMPORTED
from the host probe rather than respelled: two copies of an adversarial input are two
things to keep in step.

=============================================================================
THE BAR
=============================================================================

* **S1: identical, 60 complete driver steps, every stored volume, every step**, on
  all three value classes.
* **S1 repeated** ``SCHEDULE_REPEATS`` times from the same seed, because the finding
  the in-place weld's record made was that its answer DEPENDED ON THE BLOCK SCHEDULE
  and a single run cannot see that.
* **S2: identical at EVERY thread-block size.**
* **Both float32 subnormal policies**, each in its own process with its own CuPy
  cache directory.
* **Launch counts from two independent counters** -- a proxy over the shipped compile
  memo and the launcher's own returned report -- which must agree, equal the step
  budget, and name ONLY the fused kernel.
* **The mutation battery**, every arm armed through the SHIPPED launcher's own doors
  (``kernel=``, ``rotate=``, the resolved tables) rather than by editing anything
  under ``meep_gpu/``.

=============================================================================
WHAT THIS GATE DOES NOT CLAIM
=============================================================================

No throughput claim is made or possible: the box is shared and this file times
nothing. Nothing here is a dispatch claim either -- ``meep_gpu.fastpath
.plan_fast_path`` still returns ``None`` on every branch, so no production step
reaches these kernels. And the resolution's ARITHMETIC against the driver's pass
order is measured off-device by
``parity/meep_gpu/probe_cuda_complex_offdiag_scratch_weld.py``; what is measured here
is a DEVICE claim about a launch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402

log = probe.log
operand_census = probe.operand_census

#: Per-case RNG seed base, from the case LABEL rather than from ``hash()``.
SEED = 20260902

#: Complete DRIVER STEPS the S1 legs run. Sixty, the budget every hand-CUDA record on
#: this track is cut at.
STEPS = 60

#: How many times each S1 fixture is repeated from the same seed.
SCHEDULE_REPEATS = 4

#: The block sizes S2 sweeps. 32 is one warp, 1024 the maximum, and the shipped
#: launcher's own 256 sits between them.
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

#: Every stored volume a complete step can touch. All of them are compared: the B/H
#: half is here even though neither product touches it, because a pair that corrupted
#: E reaches B through ``step_B`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The volumes that must be BIT-UNCHANGED by a run: the PML coefficient vectors on
#: both Yee sub-lattices.
IMMUTABLE_PML: Tuple[str, ...] = tuple(
    f"{name}_{axis}{suffix}"
    for axis in "xyz" for name in ("kms", "sinv", "kps") for suffix in ("", "_h"))

#: The families the SUBNORMAL BAND and ADVERSARIAL classes are held to. Under
#: ``flush`` a state seeded entirely in the band drives ``constitutive_apply`` to
#: ``f[idx] = (f[idx] + 0) - 0``, so E is the identity and need not move; D, fu_D and
#: f_w_E are written unconditionally and must move under either policy. The
#: adversarial class is held to the same list for the same reason: half its
#: catalogue is exact zeros and subnormals.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
    + [f"f_w_E{axis}" for axis in "xyz"])

#: THE THIRD CLASS IS THIS GATE'S OWN. See the module docstring: it is the only one
#: on which the three complex arms are decidable.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "adversarial_words")

#: The classes whose divergence is measured on WORDS a plain scale would rewrite.
COMPLEX_ARM_CLASS = "adversarial_words"

#: NVRTC options the shipped modules compile with, restated so the guard control can
#: drop them. NOT imported from either module.
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


# ---------------------------------------------------------------------------
# The two products, their licences and the adversarial catalogue
# ---------------------------------------------------------------------------

def products() -> Dict[str, Any]:
    """The two modules under test, imported at call time (they need CuPy)."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_no_pml_offdiag_fused_electric_pair as no_pml,
        folded_complex_offdiag_fused_electric_pair as folded,
    )

    return {"folded": folded, "no_pml": no_pml}


def host_probe() -> Any:
    """The host probe module, imported for its ADVERSARIAL CATALOGUE and nothing else.

    IMPORTED THROUGH A SHIELD, and the shield is the point. That module carries
    ``os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "keep")`` at import scope,
    so importing it into a ``flush`` process with that variable unset would pin
    ``keep`` -- and the released gate records on this track all show the variable
    UNSET, so simply setting it would also be a change to the environment the bytes
    execute under. Both are avoided the same way: the variable is set to THIS run's
    policy across the import, which makes the ``setdefault`` a no-op, and then
    restored to exactly what it was, which makes the whole import invisible.

    The catalogue is imported rather than respelled because two copies of an
    adversarial input are two things to keep in step, and this one is the input the
    off-device probe measured the closed form against.
    """
    saved = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    os.environ["MEEP_GPU_SUBNORMAL_POLICY"] = _POLICY_NAME or saved or "keep"
    try:
        import probe_cuda_complex_offdiag_scratch_weld as hostprobe  # noqa: PLC0415
    finally:
        if saved is None:
            os.environ.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
        else:
            os.environ["MEEP_GPU_SUBNORMAL_POLICY"] = saved
    return hostprobe


#: The FLUSH cut of the four-pattern expansion probe, beside the keep cut the
#: battery names. AN EXPANSION LICENCE IS POLICY-CONDITIONAL and does not transfer:
#: the keep-cut record refuses a flush run BY NAME, which is how this gate's first
#: flush leg ended after six seconds ("the expansion licence was cut under the
#: 'keep' float32 subnormal policy and this run requires 'flush'"). This is the same
#: probe directory, the same run, the other leg -- ``cut_flush.log`` beside
#: ``cut_keep.log`` -- and the policy stamp is VERIFIED below rather than inferred
#: from the filename.
COMPLEX_PROBE_RECORD_FLUSH = ("parity/meep_gpu/results/expansion_probe_2026-08-17/"
                              "expansion_probe_flush.json")


def licences(policy: str) -> Dict[str, Any]:
    """The THREE expansion licences these two products bind, by half AND by policy.

    TWO PER PRODUCT, AND THAT IS NOT A CONVENIENCE. Each product's halves sit in
    different licence families: the folded complex curl binds ``LICENSE_COMPLEX``,
    the no-absorber complex curl ``LICENSE_COMPLEX_NO_PML``, and BOTH off-diagonal
    ``update_E`` halves bind ``LICENSE_COMPLEX_OFFDIAG``, whose arbiter classifies a
    fifth multiply orientation the base-four records do not carry. Handing one
    verdict to both halves would record that one artifact's pattern set answered for
    the other's.

    THREE PER POLICY, AND THAT IS NOT ONE EITHER. ``complex_fields`` marks the
    licence ``POLICY_CONDITIONAL_LICENCE``: bytes cut under ``keep`` differ from
    bytes cut under ``flush`` in the subnormal range, so a keep-cut record refuses a
    flush run rather than being widened to cover it. The record for each half is
    chosen by POLICY from a table, so adding a policy without cutting its records is
    a ``KeyError`` at startup and never a silent fall-through to the other policy's
    bytes.

    The ARBITERS are the battery's own -- ``folded_complex.parity_expansion_license``
    for the off-diagonal half and ``complex_fields.expansion_license`` for the two
    curls -- read from the modules rather than respelled, so a census row and a gate
    leg cannot disagree about what a record licenses.
    """
    import cuda_predicate_battery as battery  # noqa: PLC0415
    import gate_cuda_complex_no_pml as no_pml_gate  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    records = {
        "constitutive": {"keep": battery.COMPLEX_PROBE_RECORD,
                         "flush": COMPLEX_PROBE_RECORD_FLUSH}[policy],
        "folded_curl": {"keep": battery.COMPLEX_PROBE_RECORD,
                        "flush": COMPLEX_PROBE_RECORD_FLUSH}[policy],
        "no_pml_curl": {"keep": battery.COMPLEX_NO_PML_PROBE_RECORD,
                        "flush": no_pml_gate.PROBE_RECORD_FLUSH}[policy],
    }
    out: Dict[str, Any] = {}
    for half, relative in records.items():
        path = os.path.join(_REPO_API, relative)
        with open(path, "rb") as handle:
            raw = handle.read()
        record = json.loads(raw.decode("utf-8"))
        # THE STAMP IS CHECKED, NOT THE FILENAME. A record named _flush that was cut
        # under keep would license the wrong bytes with no other symptom.
        stamped = (record.get("subnormal_policy") or {}).get("resolved")
        if stamped != policy:
            raise SystemExit(
                f"{relative} carries subnormal_policy.resolved={stamped!r} but this "
                f"run installed {policy!r}; an expansion licence is "
                f"POLICY-CONDITIONAL and may not be read across that boundary")
        arbiter = (folded_complex.parity_expansion_license if half == "constitutive"
                   else complex_fields.expansion_license)
        verdict = arbiter(record)
        verdict.setdefault("patterns", record.get("patterns"))
        verdict["policy_reasons"] = list(
            complex_fields.expansion_policy_reasons(record, policy))
        verdict["record"] = relative
        verdict["record_sha256"] = hashlib.sha256(raw).hexdigest()
        verdict["record_subnormal_policy"] = stamped
        verdict["arbiter"] = ("folded_complex.parity_expansion_license"
                              if half == "constitutive"
                              else "complex_fields.expansion_license")
        out[half] = verdict
    return out


def _arm(licence: Dict[str, Any]) -> str:
    """The expansion arm both halves are emitted under, from the CONSTITUTIVE licence.

    ONE ARM PER KERNEL because there is one kernel: the two halves are spliced into
    a single translation unit and the multiply helpers are emitted once. The
    constitutive licence is the arbiter because it is the stricter of the two -- it
    classifies the parity orientation the curl's record does not carry -- and the
    launcher is handed this same value, so the gate cannot compile one arm and
    launch another.
    """
    return str(licence["constitutive"]["arm"])


#: The fixture sweep. Each spec names the shape, the boundaries, the fold, the
#: off-diagonal ROW MASK and which product must admit it -- built so that every axis
#: is walled somewhere, every axis is folded somewhere on the folded family, BOTH
#: fold terminations appear, both full-count parities appear, and both an isotropic
#: and a diagonally anisotropic permittivity appear.
#:
#: THE ANISOTROPIC HALF IS NOT DECORATION. ``update_E`` binds THREE
#: inverse-permittivity pointers and on an isotropic run all three are the SAME
#: allocation, so a kernel that read ``inv_eps_Ez`` for all three would be
#: byte-identical on every isotropic row.
#:
#: THE NO-PML FAMILY CARRIES NO FOLD AND THAT IS ITS PREDICATE'S DOING, not a gap in
#: this sweep: ``covers_complex_no_pml_offdiag_fused_electric_pair`` refuses a grid
#: with any mirror fill or folded far image BY NAME, and ``leg_refusal`` measures
#: that refusal rather than this table asserting it.
SPECS: Tuple[Dict[str, Any], ...] = (
    # --- the FOLDED complex PML family --------------------------------------
    {"label": "fold_y_even_one_row", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": None, "symmetry": (("Y", 1),),
     "k": (0.0, 0.0, 0.0), "rows": {"Ex": ("Ey",)}, "anisotropic": False},
    {"label": "fold_y_odd_all_rows", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": None, "symmetry": (("Y", -1),),
     "k": (0.0, 0.0, 0.0), "rows": "all", "anisotropic": True},
    {"label": "fold_y_odd_count", "product": "folded",
     "cell": (1.6, 2.1, 1.6), "boundaries": None, "symmetry": (("Y", 1),),
     "k": (0.0, 0.0, 0.0), "rows": "all", "anisotropic": True},
    {"label": "fold_y_metallic_termination", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k": (0.0, 0.0, 0.0), "rows": "all",
     "anisotropic": False},
    {"label": "fold_y_odd_wall_x", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "k": (0.0, 0.0, 0.0), "rows": "all",
     "anisotropic": True},
    {"label": "fold_xy_mixed_phase", "product": "folded",
     "cell": (2.0, 2.0, 1.6), "boundaries": None,
     "symmetry": (("X", 1), ("Y", -1)), "k": (0.0, 0.0, 0.0), "rows": "all",
     "anisotropic": True},
    # --- the UNFOLDED complex no-absorber family ----------------------------
    # A LIVE BLOCH VECTOR ON THE NO-PML HALF, because the corpus row that puts this
    # cell on the board has one (TestMaterialGrid.test_subpixel_smoothing,
    # k = (0.3892, 0.1597, 0)) and the phase multiply is skipped entirely at k = 0.
    {"label": "no_pml_bloch_one_row", "product": "no_pml",
     "cell": (1.6, 1.6, 1.6), "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (), "k": (0.3892, 0.1597, 0.0), "rows": {"Ex": ("Ey",)},
     "anisotropic": False},
    {"label": "no_pml_bloch_all_rows", "product": "no_pml",
     "cell": (1.6, 1.7, 1.5), "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (), "k": (0.3892, 0.1597, 0.24), "rows": "all",
     "anisotropic": True},
    {"label": "no_pml_k_zero_all_rows", "product": "no_pml",
     "cell": (1.6, 1.7, 1.5), "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (), "k": (0.0, 0.0, 0.0), "rows": "all", "anisotropic": True},
    {"label": "no_pml_walls_all_rows", "product": "no_pml",
     "cell": (1.6, 1.7, 1.5), "boundaries": ("metallic", "metallic", "metallic"),
     "symmetry": (), "k": (0.0, 0.0, 0.0), "rows": "all", "anisotropic": True},
    {"label": "no_pml_wall_x_only", "product": "no_pml",
     "cell": (1.6, 1.6, 1.6), "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (), "k": (0.0, 0.0, 0.0), "rows": "all", "anisotropic": False},
)

#: The subset the mutations are scored on. Chosen so every arm has at least one
#: fixture where its machinery is LIVE: a wall on each family, a fold at each
#: termination, an odd parity, an odd full count and a doubly-unowned corner.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "fold_y_odd_all_rows", "fold_y_odd_count", "fold_y_metallic_termination",
    "fold_y_odd_wall_x", "fold_xy_mixed_phase", "fold_y_even_one_row",
    "no_pml_walls_all_rows", "no_pml_bloch_all_rows")

REDUCED_LABELS: Tuple[str, ...] = (
    "fold_y_odd_all_rows", "fold_xy_mixed_phase", "no_pml_bloch_all_rows",
    "no_pml_walls_all_rows")

ALL_ROWS: Dict[str, Tuple[str, ...]] = {
    "Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")}


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def _rows_for(spec: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    return ALL_ROWS if spec["rows"] == "all" else spec["rows"]


def _complex_hosts(names: Sequence[str], shape: Tuple[int, int, int],
                   value_class: str, rng) -> Dict[str, np.ndarray]:
    """The HOST complex64 arrays of one case, by value class.

    THE TWO WORD PLANES ARE WRITTEN DIRECTLY into ``.real`` and ``.imag`` and never
    assembled as ``re + 1j*im``: that expression is itself a complex multiply and
    rewrites exactly the zero signs the adversarial class exists to carry.
    """
    hostprobe = host_probe()
    out: Dict[str, np.ndarray] = {}
    for offset, name in enumerate(names):
        if value_class == "adversarial_words":
            out[name] = hostprobe.planted_volume(shape, offset)
            continue
        if value_class == "uniform":
            real = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            imag = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        else:
            band = probe.subnormal_band_hosts((f"{name}.re", f"{name}.im"),
                                              shape, rng)
            real, imag = band[f"{name}.re"], band[f"{name}.im"]
        volume = np.empty(shape, dtype=np.complex64)
        volume.real[...] = real
        volume.imag[...] = imag
        out[name] = volume
    return out


def build(spec: Dict[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml, dtdx)`` on the DEVICE, seeded for this class.

    ``force_complex_fields=True`` ALWAYS. Both cells this gate serves are complex by
    the census's own reading (``force_complex_fields`` is true on every corpus row of
    either), and forcing it explicitly is what makes the k = 0 fixtures real rather
    than an unreachable branch.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=spec["cell"], courant=0.35, xp=cp,
                boundaries=spec["boundaries"], k_point=tuple(spec["k"]),
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]))
    # THE NO-ABSORBER FAMILY GETS NO LAYER AT ALL -- its curl half refuses one by
    # name -- and the folded family gets the far face only of a folded axis, whose
    # near face is the mirror plane.
    thickness = (((0, 0),) * 3 if spec["product"] == "no_pml" else
                 tuple((0, 0) if grid.shape[axis] < 6
                       else (0, 2) if grid.is_mirrored(axis)
                       else (2, 2) for axis in range(3)))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    # PML STORAGE ONLY WHERE THE LAYER IS LIVE, and this is a REFUSAL rather than a
    # preference: the complex no-absorber curl declines a Fields in PML storage mode
    # beside an inert layer BY NAME ("get_H would serve a stored H that update_H
    # never writes, and this family writes no f_w"). Enabling it unconditionally
    # made all five no-absorber fixtures refuse, which is how this line got written.
    if spec["product"] != "no_pml":
        fields.enable_pml_storage()
    shape = tuple(int(n) for n in grid.shape)
    # THE PERMITTIVITY. Isotropic binds ONE allocation three times, which is the
    # configuration the certified update_E is built for and the one a per-component
    # binding error hides in; anisotropic binds three.
    values = (2.0, 2.0, 2.0) if not spec["anisotropic"] else (2.0, 2.5, 3.0)
    names = ("Ex", "Ey", "Ez")
    epsilon = {n: cp.full(shape, v, cp.float32) for n, v in zip(names, values)}
    inverse = {n: cp.full(shape, np.float32(1.0 / v), cp.float32)
               for n, v in zip(names, values)}
    rows = {row: {partner: cp.asarray(
        rng.uniform(-0.25, 0.25, shape).astype(np.float32))
        for partner in partners}
        for row, partners in _rows_for(spec).items()}
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    present = [name for name in STATE_NAMES if getattr(fields, name, None) is not None]
    hosts = _complex_hosts(present, shape, value_class, rng)
    for name in present:
        getattr(fields, name)[...] = cp.asarray(np.ascontiguousarray(hosts[name]))
    return fields, grid, pml, float(grid.dt / grid.dx)


# ---------------------------------------------------------------------------
# Word-level comparison
# ---------------------------------------------------------------------------

def to_host(array: Any) -> np.ndarray:
    return cp.asnumpy(array) if cp is not None and isinstance(
        array, cp.ndarray) else np.asarray(array)


def words(array: Any) -> np.ndarray:
    """Every stored word as uint32 -- complex64 volumes give TWO per cell."""
    return np.frombuffer(np.ascontiguousarray(to_host(array)).tobytes(),
                         dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(value).copy() for name, value in state_of(fields).items()}


def compare(left: Dict[str, np.ndarray],
            right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: n for name in sorted(left)
            if (n := differing(left[name], right[name]))}


def word_census(state: Dict[str, np.ndarray]) -> Dict[str, int]:
    """The operand census over the WORD PLANES of a complex state.

    ``probe.operand_census`` classifies float32 bits and a complex64 array is a
    float32 array of twice the length, so the volumes are viewed rather than
    converted -- ``np.asarray(complex, dtype=float32)`` would raise, and anything
    that silently took the real part would report half a census.
    """
    return operand_census({
        name: np.ascontiguousarray(value).view(np.float32)
        if np.iscomplexobj(value) else np.asarray(value, dtype=np.float32)
        for name, value in state.items()})


def read_only_snapshot(fields, pml) -> Dict[str, np.ndarray]:
    """Every volume a launch must leave BIT-UNCHANGED, by base address.

    The three inverse-permittivity volumes are keyed by address rather than by
    component because an isotropic run hands the same allocation three times, and
    listing it three times would report one drift as three.
    """
    out: Dict[str, np.ndarray] = {}
    for name in IMMUTABLE_PML:
        array = getattr(pml, name, None)
        if array is not None:
            out[f"pml.{name}"] = to_host(array).copy()
    seen: Dict[int, str] = {}
    for component in ("Ex", "Ey", "Ez"):
        volume = fields.inverse_epsilon_for(component)
        address = int(volume.data.ptr)
        if address in seen:
            continue
        seen[address] = component
        out[f"inv_eps@{component}"] = to_host(volume).copy()
    return out


def read_only_drift(before: Dict[str, np.ndarray], fields, pml) -> List[str]:
    after = read_only_snapshot(fields, pml)
    return sorted(name for name in before
                  if name in after and differing(before[name], after[name]))


# ---------------------------------------------------------------------------
# The oracle: the driver's own pass order, read from the driver
# ---------------------------------------------------------------------------

STEP_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H", "step_D", "fill_symmetry_bc_D", "zero_metal_D",
    "fill_folded_far_ghosts_D", "update_E")


def leg_driver_order() -> Dict[str, Any]:
    """The order this gate walks IS ``FdtdDriver.step``'s, read from its source.

    An oracle a gate invented proves nothing about the engine, so the ordered list of
    ``stepping`` calls is extracted by ``ast`` and both products' ``REPLACES`` are
    asserted to be CONTIGUOUS subsequences of it.
    """
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(driver_module.FdtdDriver.step))
    tree = ast.parse(source)
    called: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name in STEP_PASSES and name not in called:
                called.append(name)
    order = [name for name in STEP_PASSES if name in called]
    record: Dict[str, Any] = {"driver_calls": called, "walked_order": order,
                              "products": {}}
    ok = sorted(called) == sorted(order) and len(order) == len(STEP_PASSES)
    for name, module in products().items():
        replaces = list(module.REPLACES)
        start, end = order.index(replaces[0]), order.index(replaces[-1])
        span = order[start:end + 1]
        # THE GAP IS NOT A HOLE. A product may declare a SHORTER run than the
        # driver's span so long as every pass it skips is one its own predicate
        # makes INERT. The unfolded weld skips the two mirror fills because it
        # refuses every folded grid, and leg_refusal MEASURES that refusal rather
        # than this leg asserting it.
        gap = [step for step in span if step not in replaces]
        declared_inert = {
            "no_pml": ["fill_symmetry_bc_D", "fill_folded_far_ghosts_D"],
            "folded": []}[name]
        entry = {
            "replaces": replaces,
            "driver_span": span,
            "ordered_subsequence": [s for s in span if s in replaces] == replaces,
            "skipped_passes": gap,
            "skipped_passes_are_the_declared_inert_ones": gap == declared_inert,
            "why_the_skipped_passes_cannot_run": (
                "this product's predicate refuses any grid carrying a mirror fill "
                "or a folded far image BY NAME, and both mirror fills return early "
                "unless grid.has_symmetry() (stepping.py:1482-1483, :1568-1570), so "
                "neither runs inside this seam on any grid it admits"
                if declared_inert else
                "the product declares the whole span; nothing is skipped"),
        }
        entry["passed"] = (entry["ordered_subsequence"]
                           and entry["skipped_passes_are_the_declared_inert_ones"])
        record["products"][name] = entry
        ok = ok and entry["passed"]
    record["passed"] = bool(ok)
    return record


def run_pass(name: str, fields, pml) -> None:
    getattr(stepping, name)(fields, pml) if name in (
        "step_B", "step_D", "update_H", "update_E") else getattr(
        stepping, name)(fields)


def array_step(fields, pml) -> None:
    for name in STEP_PASSES:
        run_pass(name, fields, pml)


# ---------------------------------------------------------------------------
# Launch counting: two independent counters
# ---------------------------------------------------------------------------

class _CountingKernel:
    """A ``cp.RawKernel`` that counts its launches and delegates everything else."""

    __slots__ = ("_kernel", "_counter", "_name")

    def __init__(self, kernel: Any, counter: Dict[str, int], name: str) -> None:
        self._kernel, self._counter, self._name = kernel, counter, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


class MemoLaunchCounter:
    """Wrap every memoized device kernel; restore on exit.

    THE MEMO IS THE RIGHT SEAM and it is the shipped one: every launcher in this
    package reaches its kernel through ``compile_cache.get_or_compile``. NOTHING IS
    INSTALLED UNLESS THE MEMO IS ALREADY WARM -- a kernel first compiled inside the
    counted region is launched and NOT counted, which is why :func:`_warm_memo` runs
    first.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self._saved: Dict[Any, Any] = {}

    def __enter__(self) -> "MemoLaunchCounter":
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001 - the shipped memo
        self._saved = dict(store)
        for key, kernel in list(store.items()):
            store[key] = _CountingKernel(kernel, self.counts, str(key[0]))
        return self

    def __exit__(self, *exc: Any) -> None:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001
        for key, kernel in list(store.items()):
            if isinstance(kernel, _CountingKernel) and key in self._saved:
                store[key] = self._saved[key]

    @property
    def total(self) -> int:
        return int(self.counts.get("_total", 0))

    def named(self) -> Dict[str, int]:
        return {k: v for k, v in sorted(self.counts.items()) if k != "_total"}


# ---------------------------------------------------------------------------
# One product case
# ---------------------------------------------------------------------------

def _runner(module) -> Callable[..., Dict[str, Any]]:
    """The shipped ``run_*`` of whichever family this is."""
    for name in ("run_folded_complex_offdiag_fused_electric_pair",
                 "run_complex_no_pml_offdiag_fused_electric_pair"):
        if hasattr(module, name):
            return getattr(module, name)
    raise SystemExit(f"{module.FAMILY} exposes no run_* entry point")


def _make_scratch(module, fields) -> Dict[str, Any]:
    for name in ("folded_complex_offdiag_fused_electric_pair_scratch",
                 "complex_no_pml_offdiag_fused_electric_pair_scratch"):
        if hasattr(module, name):
            return getattr(module, name)(fields)
    raise SystemExit(f"{module.FAMILY} exposes no scratch builder")


def memo_name(module, licence: Dict[str, Any]) -> str:
    """The name this family's kernel is MEMOIZED under, which is not ``KERNEL_NAME``.

    A complex family emits one kernel per EXPANSION ARM and memoizes on
    ``f"{KERNEL_NAME}_arm{normalized_expansion(arm)}"`` -- two arms compiled under
    one key would serve whichever was compiled first, which is the whole reason the
    suffix exists. The launch-count leg reads the memo, so it must ask for the name
    the memo actually holds; it is composed HERE from the family's own helper rather
    than matched by prefix, because a prefix match would also accept the other arm's
    binary and that is precisely the confusion the suffix prevents.
    """
    from meep_gpu.cuda_kernels import complex_emitter as emitter  # noqa: PLC0415

    return (f"{module.KERNEL_NAME}_arm"
            f"{emitter.normalized_expansion(_arm(licence))}")


def _licence_for(module, licence: Dict[str, Any]) -> Dict[str, Any]:
    """The ``license`` / ``curl_license`` pair this product's predicate wants."""
    curl = ("folded_curl" if module.ARM == "pml" else "no_pml_curl")
    return {"license": licence["constitutive"], "curl_license": licence[curl]}


#: The policy NAME the predicates are asked with, set by :func:`main` from
#: ``--subnormal-policy``. NOT None and not inferred: both products' curl halves
#: refuse an unnamed policy BY NAME ("an expansion licence is POLICY-CONDITIONAL"),
#: because a licence cut under one policy says nothing about bytes executed under the
#: other. A gate that left it None would see every fixture refused.
_POLICY_NAME: Optional[str] = None


def _launch_once(module, fields, grid, pml, dtdx, scratch, licence, *,
                 kernel: Optional[Any] = None, rotate: bool = True) -> Dict[str, Any]:
    """One fused launch through the SHIPPED launcher, with the gate's doors open."""
    bound = _licence_for(module, licence)
    report = _runner(module)(
        fields, grid, pml, dtdx, _arm(licence), scratch=scratch, sources=(),
        license=bound["license"], subnormal_policy=_POLICY_NAME,
        kernel=kernel, rotate=rotate)
    if not report.get("launched"):
        raise SystemExit(f"{module.FAMILY} refused a fixture the sweep admits: "
                         f"{report.get('reason')}")
    return report


def run_case(spec: Dict[str, Any], value_class: str, steps: int, licence, *,
             kernel: Optional[Any] = None, rotate: bool = True,
             threads: int = 256, repeat: int = 0,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP.

    The reference runs ``stepping``'s own ten-pass sequence; the subject runs the
    fused launch plus the passes it does not replace. Compared as raw uint32 WORDS
    over all stored volumes after EVERY step -- two words per complex cell.
    """
    module = products()[spec["product"]]
    expected_memo = memo_name(module, licence)
    label = f"{spec['label']}/{value_class}{label_suffix}"
    reference, grid_r, pml_r, dtdx = build(spec, value_class, case_rng(label))
    subject, grid_s, pml_s, _ = build(spec, value_class, case_rng(label))
    seeded = frozen(reference)
    assert not compare(seeded, frozen(subject)), (
        f"{label}: the two engines were not seeded identically")

    read_only = read_only_snapshot(subject, pml_s)
    census = word_census({name: to_host(value)
                          for name, value in state_of(subject).items()})
    scratch = _make_scratch(module, subject)
    saved_threads = module._FUSED_THREADS  # noqa: SLF001 - the launcher's own constant
    module._FUSED_THREADS = int(threads)  # noqa: SLF001
    first_divergence: Optional[Dict[str, Any]] = None
    launch_reports: List[Dict[str, Any]] = []
    try:
        with MemoLaunchCounter() as counter:
            for step in range(steps):
                array_step(reference, pml_r)
                for name in STEP_PASSES:
                    if name in module.REPLACES:
                        if name != module.REPLACES[0]:
                            continue
                        report = _launch_once(module, subject, grid_s, pml_s, dtdx,
                                              scratch, licence, kernel=kernel,
                                              rotate=rotate)
                        launch_reports.append(
                            {k: v for k, v in report.items() if k != "scratch"})
                        scratch = report["scratch"]
                        continue
                    run_pass(name, subject, pml_s)
                cp.cuda.runtime.deviceSynchronize()
                moved = compare(frozen(reference), frozen(subject))
                if moved and first_divergence is None:
                    first_divergence = {"step": step + 1, "volumes": moved}
                    break
            named = counter.named()
            total = counter.total
    finally:
        module._FUSED_THREADS = saved_threads  # noqa: SLF001

    final = frozen(subject)
    # THE NO-ABSORBER FAMILY ALLOCATES NO fu/f_w AT ALL, so the must-move list is
    # intersected with what the fixture actually holds and the intersection is
    # RECORDED. A list that silently shrank to nothing would report non_vacuous on a
    # run that moved nothing.
    must_move = [name for name in
                 (STATE_NAMES if value_class == "uniform" else BAND_MUST_MOVE)
                 if name in seeded]
    unmoved = sorted(name for name in must_move
                     if not differing(seeded[name], final[name]))
    return {
        "label": label, "spec": spec["label"], "product": spec["product"],
        "value_class": value_class, "steps": steps, "threads": int(threads),
        "repeat": repeat,
        "shape": [int(n) for n in grid_s.shape],
        "row_mask": launch_reports[0]["row_mask"] if launch_reports else None,
        "arm": launch_reports[0]["arm"] if launch_reports else None,
        "bit_identical": first_divergence is None,
        "first_divergence": first_divergence,
        "launch_counts": {"memo_named": named, "memo_total": total,
                          "launcher_reports": len(launch_reports),
                          "memo_key_expected": expected_memo},
        "launch_counts_agree": (total == len(launch_reports) == steps
                                and set(named) == {expected_memo}
                                and named.get(expected_memo) == steps),
        "read_only_drift": read_only_drift(read_only, subject, pml_s),
        "operand_census": census,
        "must_move_volumes": must_move,
        "unmoved_volumes": unmoved,
        "non_vacuous": bool(must_move) and not unmoved,
        "launch_geometry": launch_reports[0] if launch_reports else None,
        "rotated": bool(rotate),
    }


# ---------------------------------------------------------------------------
# S3: the PLANTED D SEAM -- one launch against the driver's own D-seam passes
# ---------------------------------------------------------------------------

#: The five driver passes the D seam is made of, in driver order (driver.py:3317,
#: :3326, :3327, :3330, :3332). The folded product declares all five; the
#: no-absorber one declares three and its predicate makes the other two inert.
D_SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E")


def run_seam_case(spec: Dict[str, Any], licence, *, kernel: Optional[Any] = None,
                  value_class: str = COMPLEX_ARM_CLASS,
                  label_suffix: str = "") -> Dict[str, Any]:
    """ONE fused launch against the driver's D-SEAM PASSES ALONE, on a planted state.

    WHY THIS LEG EXISTS, and it is a MEASUREMENT rather than a preference. The
    complete-step walk (S1) cannot reach the three complex arms, and the reason was
    measured on device on 2026-09-02 rather than guessed: a complete step runs the
    MAGNETIC half first -- ``step_B``, its two fills, ``zero_metal_B`` and
    ``update_H`` -- and ``update_H`` OVERWRITES H before the D seam is entered. So
    whatever was planted in H is gone, ``step_D``'s output at every imaged plane is
    an ordinary normal, and the parity multiplies a value with no signed zero in it.
    Scored that way, ``parity_as_word_scale`` and ``parity_skipped_at_plus_one``
    came back UNCAUGHT on all six folded fixtures with ``first_divergence: null``,
    while the SAME mutants launched on the planted pre-launch state moved 11-16
    stored words. The arms were not weak; the leg could not deliver them their input.

    So this leg enters at the seam. The reference runs the five ``stepping`` passes
    the seam is made of and nothing else; the subject runs ONE fused launch and the
    rotation. The claim is the same claim -- byte identity over every stored volume
    -- on the one input class that makes the complex spellings decidable.

    This is the device twin of the host probe's leg 2, which measured the same
    closed form against the same three passes on the same catalogue.
    """
    module = products()[spec["product"]]
    label = f"{spec['label']}/{value_class}/seam{label_suffix}"
    reference, _grid_r, pml_r, dtdx = build(spec, value_class, case_rng(label))
    subject, grid_s, pml_s, _ = build(spec, value_class, case_rng(label))
    seeded = frozen(reference)
    assert not compare(seeded, frozen(subject)), (
        f"{label}: the two engines were not seeded identically")
    read_only = read_only_snapshot(subject, pml_s)
    census = word_census({name: to_host(value)
                          for name, value in state_of(subject).items()})
    for name in D_SEAM_PASSES:
        run_pass(name, reference, pml_r)
    scratch = _make_scratch(module, subject)
    report = _launch_once(module, subject, grid_s, pml_s, dtdx, scratch, licence,
                          kernel=kernel)
    cp.cuda.runtime.deviceSynchronize()
    moved = compare(frozen(reference), frozen(subject))
    final = frozen(subject)
    must_move = [name for name in ("Dx", "Dy", "Dz") if name in seeded]
    unmoved = sorted(name for name in must_move
                     if not differing(seeded[name], final[name]))
    return {
        "label": label, "spec": spec["label"], "product": spec["product"],
        "value_class": value_class,
        "shape": [int(n) for n in grid_s.shape],
        "seam_passes": list(D_SEAM_PASSES),
        "bit_identical": not moved,
        "differing_volumes": moved,
        "differing_words": sum(moved.values()),
        "read_only_drift": read_only_drift(read_only, subject, pml_s),
        "operand_census": census,
        "must_move_volumes": must_move,
        "unmoved_volumes": unmoved,
        "non_vacuous": bool(must_move) and not unmoved,
        "row_mask": report["row_mask"], "arm": report["arm"],
    }


def leg_planted_seam(specs: Sequence[Dict[str, Any]],
                     licence: Dict[str, Any]) -> Dict[str, Any]:
    """S3 on the shipped kernel: identity at the seam, on the adversarial words."""
    cases = [run_seam_case(spec, licence) for spec in specs]
    for record in cases:
        log(f"[S3] {record['label']:56s} "
            f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'} "
            f"negzero={record['operand_census'].get('negative_zeros')}")
    return {"cases": cases,
            "passed": bool(cases) and all(c["bit_identical"] for c in cases)
            and all(c["non_vacuous"] for c in cases)}


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def fixture_facts(spec: Dict[str, Any]) -> Dict[str, Any]:
    """What is actually LIVE on this fixture, read off a real grid.

    THE MUTATION BATTERY'S HONESTY DEPENDS ON THIS. An arm scored on a fixture where
    the machinery it disables is inert comes back UNCAUGHT and looks like a hole; an
    arm predicted null on a fixture where it IS live is a catch nobody made. Both are
    settled by asking the grid rather than by reading the spec's boundary strings --
    ``zero_metal_axes`` EXCLUDES a folded metallic axis, so "declared metallic" and
    "walled" are different questions.
    """
    from meep_gpu.cuda_kernels import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
    from meep_gpu.cuda_kernels import complex_offdiag_update_e as offdiag  # noqa: PLC0415

    module = products()[spec["product"]]
    fields, grid, pml, _ = build(spec, "uniform", np.random.default_rng(3))
    plan = weld.fill_plan(grid)
    if spec["product"] == "folded":
        codes = module.folded_complex_offdiag_fused_electric_pair_codes(grid)
    else:
        triple = module.complex_no_pml_offdiag_fused_electric_pair_codes(grid)
        codes = {"curl": tuple(triple), "constitutive": tuple(triple)}
    weights = [float(w) for w in offdiag.mirror_ghost_weights(grid)]
    phases = (module.folded_complex_offdiag_fused_electric_pair_phases(grid)
              if spec["product"] == "folded"
              else module.complex_no_pml_offdiag_fused_electric_pair_phases(grid))
    shape = [int(n) for n in grid.shape]
    return {
        "shape": shape,
        "near": list(plan["near"]), "wall": list(plan["wall"]),
        "reflect": list(plan["reflect"]),
        "near_phase": list(plan["near_phase"]), "far_phase": list(plan["far_phase"]),
        "ghost_weights": weights,
        "codes": {k: list(v) for k, v in codes.items()},
        "bloch_flags": [int(f) for f in phases["flags"]],
        "any_wall": any(plan["wall"]),
        "any_near": any(plan["near"]),
        "any_far": any(int(row) >= 0 for row in plan["reflect"]),
        "any_bloch": any(int(f) for f in phases["flags"]),
        "ghost_weight_is_negative": any(
            plan["near"][axis] and weights[axis] != 1.0 for axis in range(3)),
        "reflect_row_is_not_n_minus_two": any(
            int(plan["reflect"][axis]) >= 0
            and int(plan["reflect"][axis]) != shape[axis] - 2
            for axis in range(3)),
        "boundary_readings_differ": list(codes["curl"]) != list(
            codes["constitutive"]),
        # MEASURED, not read off the grid -- see parity_divergence_census.
        "parity_divergence": parity_divergence_census(spec),
        # THE ONE SHAPE `clear_order_flipped` CAN MOVE A WORD ON: a component with a
        # near parity on one of its axes AND a wall on another of them, so a near
        # ghost lands in a plane the clear owns. Under COMPLEX storage the parity is
        # live at BOTH signs -- mul_coefficient_left(-1, cf_zero()) is (-0.0, +0.0)
        # and mul_coefficient_left(+1, .) rewrites a signed zero too -- so unlike
        # the float32 twin this does not additionally require an ODD parity.
        "near_parity_meets_a_cleared_plane": any(
            plan["near"][a] and plan["wall"][b]
            for component in range(3)
            for a in weld.near_axes(component)
            for b in weld.clear_axes(component) if a != b),
    }


def _fused_parity(coefficient: float, plane: np.ndarray) -> np.ndarray:
    """``mul_coefficient_left`` on a host plane, words written direct.

    NOT ``coefficient * plane``: NumPy's own scalar-times-complex loop is the very
    spelling under test, and building the answer as ``re + 1j*im`` is a second
    complex multiply that rewrites the zero signs again.
    """
    one = np.float32(coefficient)
    zero = np.float32(0.0)
    out = np.empty(plane.shape, np.complex64)
    out.real[...] = one * plane.real + (zero * plane.imag) * np.float32(-1.0)
    out.imag[...] = one * plane.imag + zero * plane.real
    return out


def _scaled_parity(coefficient: float, plane: np.ndarray) -> np.ndarray:
    """The plane-wise two-word scale ``parity_as_word_scale`` substitutes."""
    one = np.float32(coefficient)
    out = np.empty(plane.shape, np.complex64)
    out.real[...] = one * plane.real
    out.imag[...] = one * plane.imag
    return out


def _uint32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.float32).view(np.uint32)


def parity_divergence_census(spec: Dict[str, Any]) -> Dict[str, int]:
    """How many WORDS the two wrong parity spellings could move on this fixture.

    THE TWO COMPLEX PARITY ARMS ARE VALUE-SENSITIVE, not structure-sensitive, and
    their liveness cannot be read off a grid. ``mul_coefficient_left`` and the plane
    scale differ ONLY where a signed zero meets the other word's sign, so a fixture
    can carry a live fold, a live parity and every structural precondition and still
    have nothing for either arm to move. Scoring such a fixture would report a hole
    that is really a vacuous arm -- the failure this campaign has already paid for.

    So liveness is MEASURED here, on the array path's own ``step_D`` output from the
    planted state, over exactly the (component, axis) pairs the kernel's resolution
    multiplies: ``near_axes(component)`` where the fill is live, and
    ``far_axis(component)`` where the reflect row is. The counts go into the record
    beside the outcome, so a predicted null carries a number rather than a claim.
    """
    from meep_gpu.cuda_kernels import complex_offdiag_stencil_weld as weld  # noqa: PLC0415

    fields, grid, pml, _ = build(spec, COMPLEX_ARM_CLASS, case_rng(spec["label"]))
    plan = weld.fill_plan(grid)
    stepping.step_D(fields, pml)
    scale = skip = 0
    for component, name in enumerate(("Dx", "Dy", "Dz")):
        volume = to_host(getattr(fields, name))
        reads: List[Tuple[int, int, float]] = []
        for axis in weld.near_axes(component):
            if plan["near"][axis]:
                reads.append((axis, weld.NEAR_SOURCE_ROW,
                              float(plan["near_phase"][axis])))
        far = weld.far_axis(component)
        if int(plan["reflect"][far]) >= 0:
            reads.append((far, int(plan["reflect"][far]),
                          float(plan["far_phase"][far])))
        for axis, row, phase in reads:
            if row >= volume.shape[axis]:
                continue
            source = np.ascontiguousarray(np.take(volume, row, axis=axis))
            fused = _uint32(_fused_parity(phase, source))
            scale += int((fused != _uint32(_scaled_parity(phase, source))).sum())
            if phase == 1.0:
                skip += int((fused != _uint32(source)).sum())
    return {"word_scale": int(scale), "skip_at_plus_one": int(skip),
            "measured_on": COMPLEX_ARM_CLASS,
            "measured_over": "the array path's own step_D output at the planes "
                             "near_axes/far_axis declare for each D component"}


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``old`` with ``new``, requiring exactly ``count`` occurrences.

    A mutation that did not land is a mutation that was not tested, and it would be
    reported UNCAUGHT -- which is the one way a mutation battery lies.
    """
    found = source.count(old)
    if found != count:
        raise SystemExit(
            f"the mutation anchor {old!r} appears {found} times, not {count}; the "
            f"mutation would not have landed and would have scored UNCAUGHT")
    return source.replace(old, new, count)


#: The near/far parity call sites, one per axis per component. Spelled from the
#: emitter's own shape so a mutation counts what the source actually holds.
_NEAR_PARITY = tuple(
    f"        if (image_{axis}) value = mul_coefficient_left("
    f"weld.near_phase_{axis}, value);" for axis in "xyz")
_FAR_PARITY = tuple(
    f"    if (far) value = mul_coefficient_left(weld.far_phase_{axis}, value);"
    for axis in "xyz")


def entry_body(source: str) -> str:
    """The body of the ONE ``extern "C" __global__`` entry point in ``source``."""
    marker = 'extern "C" __global__ void '
    if source.count(marker) != 1:
        raise SystemExit(
            f"the certified source carries {source.count(marker)} entry points, not "
            f"1; its body cannot be identified")
    return source.split(marker, 1)[1].split("\n) {\n", 1)[1]


def kernel_body(source: str, kernel_name: str) -> str:
    """Everything below the ENTRY POINT's parameter list.

    NOT ``source.split("\\n) {\\n", 1)[-1]``. Several ``__device__`` helpers in this
    emitted text also close their parameter lists on their own line, so that split
    lands on the first helper and hands back a "body" that still contains the
    kernel's own SIGNATURE -- which is how a ``kms_half_`` declaration read as a
    surviving body reference and failed a mutation that had in fact landed.
    """
    marker = f'extern "C" __global__ void {kernel_name}('
    if source.count(marker) != 1:
        raise SystemExit(
            f"the emitted source carries {source.count(marker)} entry points named "
            f"{kernel_name}, not 1; its body cannot be identified")
    tail = source.split(marker, 1)[1]
    if "\n) {\n" not in tail:
        raise SystemExit(f"{kernel_name}'s parameter list does not close on its own "
                         f"line; the body cannot be identified")
    return tail.split("\n) {\n", 1)[1]


def _word_scale_helper(source: str) -> str:
    """Insert the plane-wise two-word scale beside the certified multiply."""
    anchor = "__device__ __forceinline__ cf mul_coefficient_left(float c, cf z) {"
    helper = ("__device__ __forceinline__ cf mul_word_scale(float c, cf z) {\n"
              "    cf o;\n"
              "    o.re = c * z.re;\n"
              "    o.im = c * z.im;\n"
              "    return o;\n"
              "}\n\n")
    return needle(source, anchor, helper + anchor)


def _parity_as_word_scale(source: str) -> str:
    """THE COMPLEX ARM: every parity becomes ``{re*w, im*w}``.

    Byte-wrong on signed zeros and nowhere else, which is why it is scored on the
    adversarial class alone.
    """
    out = _word_scale_helper(source)
    for line in _NEAR_PARITY + _FAR_PARITY:
        if line not in out:
            continue
        out = out.replace(line, line.replace("mul_coefficient_left",
                                             "mul_word_scale"))
    if "mul_word_scale(weld." not in out:
        raise SystemExit("no parity call site was rewritten; the arm would not land")
    return out


def _parity_skipped_at_plus_one(source: str) -> str:
    """THE EVEN-MIRROR SHORTCUT, which float32 allows and complex64 does not."""
    out = source
    landed = 0
    for line in _NEAR_PARITY + _FAR_PARITY:
        if line not in out:
            continue
        head, call = line.split("value = ", 1)
        phase = call.split("mul_coefficient_left(", 1)[1].split(",", 1)[0]
        out = out.replace(
            line, f"{head}value = ({phase} == 1.0f) ? value : "
                  f"mul_coefficient_left({phase}, value);")
        landed += 1
    if not landed:
        raise SystemExit("no parity call site was rewritten; the arm would not land")
    return out


def _clear_real_word_only(arm: str) -> Callable[[str], str]:
    """THE ARM THE HOST PROBE COULD NOT EXPRESS: a clear that leaves ``im`` alone.

    ``stepping._zero_metal`` assigns the INTEGER 0 into a complex64 array, which is
    the pair ``(+0.0f, +0.0f)``. Here the store is two explicit float32 writes, so a
    clear that zeroed only the real word is expressible -- and is a defect class this
    tree has paid for. The replacement takes the cleared cell's RAW stepped value for
    its imaginary word, which is what a half-clear would leave behind.

    ARM-PARAMETERISED, because ``raw_step_D_cell``'s ARITY DIFFERS BETWEEN THE TWO
    ARMS: the ``pml`` lift returns the split-field auxiliary through a sixth
    parameter and the ``no_pml`` one has no auxiliary at all. A single spelling
    would land its anchor on both and then fail to COMPILE on one -- a mutation that
    never ran, scored UNCAUGHT, which is the one way a battery lies.
    """
    call = ("raw_step_D_cell(i, j, k, weld, half_raw, half_fu);" if arm == "pml"
            else "raw_step_D_cell(i, j, k, weld, half_raw);")
    declare = ("            cf half_raw[3];\n            cf half_fu[3];\n"
               if arm == "pml" else "            cf half_raw[3];\n")

    def rewrite(source: str) -> str:
        anchor = ("        // The array path assigns the INTEGER 0 to a complex64 "
                  "array\n"
                  "        // here, which is the word pair (+0.0f, +0.0f).\n"
                  "        value = cf_zero();")
        if source.count(anchor) != 3:
            raise SystemExit(
                f"the wall-clear anchor appears {source.count(anchor)} times, not "
                f"3 (one per D component); the arm would not land whole")
        out = source
        for component in range(3):
            out = out.replace(anchor, (
                "        // MUTANT: only the real word is cleared.\n"
                "        {\n"
                f"{declare}"
                f"            {call}\n"
                f"            value = half_raw[{component}];\n"
                "            value.re = 0.0f;\n"
                "        }"), 1)
        return out

    return rewrite


def _far_reflect_row_mutation(source: str) -> str:
    """Every ``resolve_D*``'s far redirect pointed at the fixed row ``n - 2``."""
    out = source
    for axis, extent in zip("xyz", ("nx", "ny", "nz")):
        coordinate = {"x": "i", "y": "j", "z": "k"}[axis]
        out = needle(out, f"    if (far) {coordinate} = weld.reflect_{axis};",
                     f"    if (far) {coordinate} = weld.{extent} - 2;")
    return out


def _constitutive_codes_mutation(kernel_name: str) -> Callable[[str], str]:
    """Every BODY read of ``cbc_*`` replaced by ``bc_*``; the DECLARATION stays.

    Rewriting the declaration as well is a duplicate parameter name and a compile
    error -- which scores as a mutation that never ran, the one way a battery lies.
    So the rewrite is applied to the body ALONE, split at the entry point.
    """

    def rewrite(source: str) -> str:
        marker = f'extern "C" __global__ void {kernel_name}('
        head, tail = source.split(marker, 1)
        signature, body = tail.split("\n) {\n", 1)
        if "cbc_" not in body:
            raise SystemExit(
                "the constitutive body reads no cbc_ code; this family does not "
                "make the edit this arm undoes and the arm would not land")
        moved = body.replace("cbc_", "bc_")
        if "cbc_" not in signature:
            raise SystemExit(
                "the signature declares no cbc_ parameter; rewriting the body "
                "alone would not compile")
        return head + marker + signature + "\n) {\n" + moved

    return rewrite


def _rename_local_value(source: str) -> str:
    """``value`` -> ``resolved_value`` everywhere, a rename and nothing else."""
    import re  # noqa: PLC0415 - stdlib, at the one call site

    renamed, count = re.subn(r"\bvalue\b", "resolved_value", source)
    if count < 10:
        raise SystemExit(
            f"the null control renamed only {count} occurrences of `value`; it is "
            f"meant to move the whole resolution's local naming and would otherwise "
            f"be a null that changed almost nothing")
    return renamed


def _in_place_curl_read(source: str) -> str:
    """THE IN-PLACE WELD, REBUILT: the weld struct binds the SCRATCH being written."""
    out = needle(source,
                 "    weld.f0 = f0;\n    weld.f1 = f1;\n    weld.f2 = f2;",
                 "    weld.f0 = f0_out;\n    weld.f1 = f1_out;\n"
                 "    weld.f2 = f2_out;")
    if "    weld.u0 = u0;\n    weld.u1 = u1;\n    weld.u2 = u2;" in out:
        out = out.replace(
            "    weld.u0 = u0;\n    weld.u1 = u1;\n    weld.u2 = u2;",
            "    weld.u0 = u0_out;\n    weld.u1 = u1_out;\n    weld.u2 = u2_out;", 1)
    return out


def _halo_recompute_dropped(source: str) -> str:
    """A foreign sample takes a FIXED in-bounds cell instead of the neighbour's."""
    out = needle(
        source, "    return (index < 0) ? cf_zero() : resolve_D_at(comp, index, weld);",
        "    return (index < 0) ? cf_zero() : resolve_D(comp, 0, 0, 0, weld);")
    out = needle(out, "    cf z = resolve_D_at(comp, index, weld);",
                 "    cf z = resolve_D(comp, 0, 0, 0, weld);")
    return out


def _sub_lattice_shadow(kernel_name: str) -> Callable[[str], str]:
    """The constitutive tail reads the CURL's INTEGER split-field vector."""

    def rewrite(source: str) -> str:
        out = source
        for axis, coordinate in zip("xyz", "ijk"):
            out = needle(out, f"kms_half_{axis}[{coordinate}]",
                         f"kms_{axis}[{coordinate}]")
        if "kms_half_" in kernel_body(out, kernel_name):
            raise SystemExit(
                "a kms_half_ read survived in the body; the shadow did not land "
                "whole and would score a partial catch")
        return out

    return rewrite


def _wall_table_from_the_b_family(source: str) -> str:
    """``zero_metal_D``'s OFF-DIAGONAL table replaced by the B family's diagonal."""
    return needle(
        source, "    if ((weld.wall_y && j == 0) || (weld.wall_z && k == 0)) {",
        "    if (weld.wall_x && i == 0) {")


def _mirror_ghost_weight_dropped(source: str) -> str:
    """``_shift_down``'s mirror arm loses its parity multiply, in BOTH spellings.

    TWO OCCURRENCES AND BOTH ARE MUTATED: the emitted text carries the certified
    ``down_sample`` (which loads a stored word) and the weld's
    ``resolved_down_sample`` (which recomputes it), and the mirror arm is the same
    line in each. Mutating one would leave the other correct and score a PARTIAL
    catch that says nothing about either.
    """
    return needle(source,
                  "    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);",
                  "    if (bc == BC_MIRROR) return z;", count=2)


def _near_source_row_wrong(source: str) -> str:
    """The near fill images stored row 1 instead of MEEP's halved-origin row 2."""
    return needle(source, "#define NEAR_SOURCE_ROW 2", "#define NEAR_SOURCE_ROW 1")


def _fixed(rewrite: Callable[[str], str]) -> Callable[[Any], Callable[[str], str]]:
    """An arm whose rewrite is the same text edit on either family."""
    return lambda module: rewrite


#: Arms that are DECIDABLE UNDER SOME POLICIES ONLY, with the mechanism. An arm not
#: listed here is decidable under every policy and must be caught on every leg.
#:
#: THIS IS NOT A WAIVER AND IT IS CHECKED BOTH WAYS. Where a policy is excluded the
#: gate still MEASURES the arm's divergence class on every fixture and REQUIRES the
#: measurement to be zero: a declaration that said "unreachable here" while the class
#: was in fact live would fail the leg, exactly as an armed-and-silent arm does. What
#: the declaration buys is the difference between NOT ARMED (no input exists) and
#: UNCAUGHT (an input exists and the gate missed it), which a single outcome cannot
#: express and which matters here because one of them is a hole and the other is a
#: property of the float32 policy.
POLICY_CONDITIONAL_ARMS: Dict[str, Dict[str, Any]] = {
    "parity_skipped_at_plus_one": {
        "decidable_under": ("keep",),
        "why": (
            "the divergence needs a real word of -0.0 beside a NEGATIVE imaginary "
            "word at a +1 parity plane, and under 'flush' that operand class does "
            "not survive to the parity: flush maps the subnormal operands that "
            "produce a sign-carrying zero in the recomputed step value to +0.0 "
            "before the multiply sees them. This is the same mechanism the "
            "expansion records state about themselves -- 'every row that separated "
            "the arms there fed the device a subnormal OPERAND, which flush "
            "destroys'. MEASURED 2026-09-02 on the GPU host over ten fold shapes under "
            "flush, including a triple fold with three +1 far parities that carries "
            "157 word_scale divergences: skip_at_plus_one was 0 on every one, while "
            "the same census under keep found it live on fold_y_odd_all_rows. The "
            "arm is CAUGHT on the keep leg, so the campaign does test the spelling; "
            "what the flush leg records is that its input does not exist there"),
    },
}


#: ``(name, applies_to, rewrite FACTORY, why, live?, value class)``.
#:
#: THE REWRITE IS A FACTORY OF THE MODULE, not a bare edit, because three arms
#: genuinely differ between the two families: ``raw_step_D_cell``'s arity differs by
#: arm, and the body split needs the entry point's own name. An arm that landed its
#: anchor on both families and then failed to COMPILE on one would be scored
#: UNCAUGHT, which is the one way a battery lies.
#:
#: ``live`` decides whether the arm can move a word ON THIS FIXTURE. Where it cannot,
#: the arm is recorded ``predicted_null`` WITH the reason. ``value class`` is the
#: seeding an arm is DECIDABLE under: the three complex arms move signed zeros and
#: nothing else, so scoring them on ``uniform`` would report three vacuous holes.
SOURCE_MUTATIONS: Tuple[Tuple[str, str, Callable[[Any], Callable[[str], str]], str,
                              Callable[[Dict[str, Any]], bool], str], ...] = (
    ("halo_recompute_dropped", "both", _fixed(_halo_recompute_dropped),
     "the foreign sample stops being recomputed AT THE NEIGHBOUR and takes a fixed "
     "cell instead -- the shortcut a reader reaches for when the recompute looks "
     "expensive, and the one the whole design says is unavailable",
     lambda facts: True, "uniform"),
    ("foreign_read_from_live_buffer", "both", _fixed(_in_place_curl_read),
     "THE IN-PLACE WELD, REBUILT: the resolution reads the scratch this launch is "
     "writing instead of the pre-launch volume. If this is not caught, this gate "
     "cannot tell the two designs apart and its S1 means nothing",
     lambda facts: True, "uniform"),
    ("parity_as_word_scale", "folded", _fixed(_parity_as_word_scale),
     "THE COMPLEX ARM. Every mirror parity becomes the plane-wise two-word scale "
     "{re*w, im*w} instead of the certified mul_coefficient_left. The array path "
     "spells phase * plane as a FULL complex multiply whose zero cross terms carry "
     "the other word's sign into an addend, so the two differ on signed zeros AND "
     "NOWHERE ELSE -- which is why the arm is scored on the adversarial class at "
     "the PLANTED SEAM, and why its liveness is a MEASURED word count rather than a "
     "grid fact",
     lambda facts: facts["parity_divergence"]["word_scale"] > 0,
     COMPLEX_ARM_CLASS),
    ("parity_skipped_at_plus_one", "folded", _fixed(_parity_skipped_at_plus_one),
     "THE SECOND COMPLEX ARM. An EVEN mirror is skipped, which is correct in "
     "float32 and wrong here: mul_coefficient_left(+1, z) is fma(1, z.re, "
     "(0*z.im) * -1.0f), which turns re = -0.0 with a negative imaginary word into "
     "+0.0. Scored on the adversarial class for the same reason as the arm above, "
     "and its divergence class is the NARROWEST on this board -- it needs a real "
     "word of -0.0 beside a negative imaginary word at an EVEN parity plane, so its "
     "liveness is measured rather than assumed",
     lambda facts: facts["parity_divergence"]["skip_at_plus_one"] > 0,
     COMPLEX_ARM_CLASS),
    ("clear_real_word_only", "both",
     lambda module: _clear_real_word_only(module.ARM),
     "THE THIRD COMPLEX ARM, AND THE ONE THE HOST PROBE COULD NOT EXPRESS: the "
     "wall clear writes +0.0f into the real word and leaves the imaginary one. "
     "stepping._zero_metal assigns the INTEGER 0 into a complex64 array (:2247), "
     "which is the pair (+0.0f, +0.0f); NumPy has no spelling of the half-clear "
     "that is not simply the truth, and on the device the store is two explicit "
     "float32 writes",
     lambda facts: facts["any_wall"], "uniform"),
    ("wall_table_from_the_b_family", "both", _fixed(_wall_table_from_the_b_family),
     "zero_metal_D's OFF-DIAGONAL table replaced by the B family's diagonal -- two "
     "components per wall become one, and the wrong one. Scored off the WALL PLAN "
     "rather than off the fixture's boundary strings, because zero_metal_axes "
     "EXCLUDES a folded metallic axis and a fold-terminated wall is not a wall",
     lambda facts: facts["any_wall"], "uniform"),
    ("sub_lattice_shadow", "folded",
     lambda module: _sub_lattice_shadow(module.KERNEL_NAME),
     "the constitutive tail reads the CURL's INTEGER split-field vector instead of "
     "its own HALF-INTEGER one -- half a cell in the absorber profile, converged, "
     "smooth and wrong. Inert without a layer, so it is predicted null on the "
     "no-absorber family whose curl half refuses one by name",
     lambda facts: True, "uniform"),
    ("near_source_row_wrong", "folded", _fixed(_near_source_row_wrong),
     "the near fill images stored row 1 instead of stored row 2 -- MEEP's io = -2 "
     "halved origin is what puts the ghost on row 2, and row 1 is a whole cell wrong",
     lambda facts: facts["any_near"], "uniform"),
    ("far_reflect_row_n_minus_two", "folded", _fixed(_far_reflect_row_mutation),
     "the far fill images the fixed row n - 2 instead of _far_reflect_rows' "
     "n_full - stored + 2. THE TWO AGREE AT AN EVEN FULL COUNT and are a whole cell "
     "apart at an odd one, so the arm is scored only on the fixture whose count is "
     "odd -- which is the entire reason that fixture is in the sweep",
     lambda facts: facts["reflect_row_is_not_n_minus_two"], "uniform"),
    ("mirror_ghost_weight_dropped", "folded", _fixed(_mirror_ghost_weight_dropped),
     "_shift_down's mirror arm loses its parity: the ghost lane reads stored row "
     "MIRROR_ROW unweighted, which is the certified family's own measured defect. "
     "The weight is mirror_parity('D'+axis, axis, phase) == -phase, so on the "
     "UNIFORM class it moves a word only where that weight is not +1 and the arm is "
     "scored there",
     lambda facts: facts["ghost_weight_is_negative"], "uniform"),
    ("constitutive_codes_from_the_curl", "folded",
     lambda module: _constitutive_codes_mutation(module.KERNEL_NAME),
     "THE ONE EDIT THIS FAMILY MAKES AND ITS TWIN DOES NOT, undone: the "
     "constitutive half reads the CURL's boundary codes, which split the two folded "
     "terminations where update_E serves both with BC_MIRROR. Only the six BODY "
     "reads are rewritten -- rewriting the declaration too would be a duplicate "
     "parameter name and a compile error, which is a mutation that never ran",
     lambda facts: facts["boundary_readings_differ"], "uniform"),
    ("local_name_changed", "both", _fixed(_rename_local_value),
     "THE CONFIRMED NULL. Every local named `value` in the resolution renamed, and "
     "nothing else: it moves the source text and the compiled binary's symbol table "
     "and not one float, so it MUST come back UNCAUGHT. A battery of catches with "
     "no confirmed null is a battery nobody showed to be two-sided -- it could be "
     "reporting CAUGHT for the act of recompiling",
     lambda facts: True, "uniform"),
)

#: Mutations armed on the HOST, through the launcher's own doors rather than the
#: source. Each returns a ``(record, caught)`` pair.
HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage")

#: The one arm above that must come back UNCAUGHT.
NULL_CONTROLS: Tuple[str, ...] = ("local_name_changed",)


def compiled_variant(module, row_mask: Sequence[int], arm: str,
                     rewrite: Callable[[str], str]) -> Any:
    """A ``cp.RawKernel`` for a REWRITTEN copy of the shipped source.

    Compiled with the shipped option tuple and the SHIPPED ARM, so the only
    difference between this and the product is the edit under test.
    """
    source = rewrite(module.kernel_source(row_mask, arm))
    return cp.RawKernel(source, module.KERNEL_NAME, options=GUARD_OPTIONS)


def leg_mutations(specs: Sequence[Dict[str, Any]], steps: int,
                  licence: Dict[str, Any]) -> Dict[str, Any]:
    """Every arm, on every fixture whose machinery it can move.

    A CAUGHT arm is one whose run diverges from the array path; an UNCAUGHT one does
    not. The scoreboard records the expectation BESIDE the outcome, so a mutation
    that was expected to be null and was caught fails the leg just as loudly as one
    that was expected caught and was not.
    """
    from meep_gpu.cuda_kernels.coverage import offdiag_row_mask  # noqa: PLC0415

    arm = _arm(licence)
    all_facts = {spec["label"]: fixture_facts(spec) for spec in specs}
    for label, facts in all_facts.items():
        log(f"[facts] {label:32s} wall={facts['wall']} near={facts['near']} "
            f"reflect={facts['reflect']} gw={facts['ghost_weights']} "
            f"bloch={facts['bloch_flags']} "
            f"codes_differ={facts['boundary_readings_differ']}")
    scoreboard: Dict[str, Any] = {"_fixture_facts": all_facts}
    for name, applies, factory, why, live, value_class in SOURCE_MUTATIONS:
        arms: List[Dict[str, Any]] = []
        for spec in specs:
            if applies != "both" and applies != spec["product"]:
                continue
            module = products()[spec["product"]]
            rewrite = factory(module)
            facts = all_facts[spec["label"]]
            if not live(facts):
                arms.append({"spec": spec["label"], "predicted_null": True,
                             "reason": ("the divergence class this arm needs does "
                                        "not occur on this fixture: "
                                        f"{facts['parity_divergence']}"
                                        if value_class == COMPLEX_ARM_CLASS else
                                        "the machinery this arm disables is not "
                                        "live on this fixture"),
                             "facts": {k: v for k, v in facts.items()
                                       if isinstance(v, (bool, list))},
                             "parity_divergence": facts["parity_divergence"]})
                continue
            fields, grid, pml, _ = build(spec, "uniform", case_rng(spec["label"]))
            mask = offdiag_row_mask(fields)
            try:
                kernel = compiled_variant(module, mask, arm, rewrite)
            except SystemExit as exc:
                arms.append({"spec": spec["label"], "anchor_error": str(exc)})
                continue
            # WHICH LEG SCORES THIS ARM. The three complex arms are scored on the
            # PLANTED D SEAM and the rest on the complete-step walk, and that split
            # is the 2026-09-02 measurement recorded in run_seam_case's docstring:
            # a complete step's magnetic half overwrites H before the D seam, so the
            # adversarial catalogue never reaches the parity and both parity arms
            # come back UNCAUGHT with first_divergence null -- while the same
            # mutants at the seam move 11-16 stored words.
            if value_class == COMPLEX_ARM_CLASS:
                record = run_seam_case(spec, licence, kernel=kernel,
                                       label_suffix=f"/M:{name}")
                arms.append({"spec": spec["label"], "value_class": value_class,
                             "scored_by": "S3 planted D seam, one launch",
                             "caught": not record["bit_identical"],
                             "differing_words": record["differing_words"],
                             "differing_volumes": record["differing_volumes"]})
                continue
            record = run_case(spec, value_class, steps=min(steps, 8),
                              licence=licence, kernel=kernel,
                              label_suffix=f"/M:{name}")
            arms.append({"spec": spec["label"], "value_class": value_class,
                         "scored_by": "S1 complete driver steps",
                         "caught": not record["bit_identical"],
                         "first_divergence": record["first_divergence"],
                         "launch_counts_agree": record["launch_counts_agree"]})
        expected_null = name in NULL_CONTROLS
        armed = [a for a in arms if "caught" in a]
        broken = [a for a in arms if "anchor_error" in a]
        # THE POLICY-CONDITIONAL CHECK, BOTH WAYS. Where an arm declares this policy
        # out of scope, EVERY fixture must have measured its class at zero; a live
        # fixture here means the declaration is wrong and the leg fails rather than
        # inheriting the exemption.
        conditional = POLICY_CONDITIONAL_ARMS.get(name)
        out_of_scope = bool(conditional) and _POLICY_NAME not in tuple(
            conditional["decidable_under"])
        if out_of_scope and armed:
            raise SystemExit(
                f"{name} declares itself undecidable under {_POLICY_NAME!r} but its "
                f"divergence class measured LIVE on "
                f"{[a['spec'] for a in armed]}; the declaration is wrong and would "
                f"have excused a real hole")
        if broken:
            raise SystemExit(
                f"the mutation {name} could not be applied on "
                f"{[a['spec'] for a in broken]}: {broken[0]['anchor_error']}. A "
                f"mutation that did not land scores UNCAUGHT and is the one way a "
                f"battery lies, so it stops the run instead")
        outcome = ("NULL CONFIRMED" if expected_null and armed
                   and not any(a["caught"] for a in armed)
                   else "CAUGHT" if armed and all(a["caught"] for a in armed)
                   else "PARTIAL" if armed and any(a["caught"] for a in armed)
                   else "UNCAUGHT" if armed else "NOT ARMED")
        expected = ("NOT ARMED" if out_of_scope
                    else "NULL CONFIRMED" if expected_null else "CAUGHT")
        scoreboard[name] = {
            "why": why, "expected": expected,
            "outcome": outcome, "applies_to": applies,
            "scored_on_value_class": value_class, "arms": arms,
            "policy": _POLICY_NAME,
            "decidable_under_this_policy": not out_of_scope,
            "not_decidable_reason": (conditional["why"] if out_of_scope else None),
            "caught_on_policies": (tuple(conditional["decidable_under"])
                                   if conditional else None),
            "as_required": outcome == expected}
        log(f"[mutation] {name:34s} {outcome:14s} "
            f"({len(armed)} armed on {value_class}, "
            f"{len(arms) - len(armed)} predicted null"
            f"{'' if not out_of_scope else '; NOT DECIDABLE under ' + str(_POLICY_NAME)})")

    # THE HOST ARMS.
    for spec in specs:
        module = products()[spec["product"]]
        record = run_case(spec, "uniform", steps=min(steps, 8), licence=licence,
                          rotate=False, label_suffix="/M:rotation_skipped")
        scoreboard.setdefault("rotation_skipped", {
            "why": "the launch computes the step and throws it away: without the "
                   "rotation the scratch never becomes the live D, so every "
                   "subsequent step reads a stale flux density",
            "expected": "CAUGHT", "applies_to": "both",
            "scored_on_value_class": "uniform", "arms": []})
        scoreboard["rotation_skipped"]["arms"].append(
            {"spec": spec["label"], "caught": not record["bit_identical"],
             "first_divergence": record["first_divergence"]})
        # THE ALIAS ARM IS SCORED AS A REFUSAL, not as a byte divergence: the check
        # runs BEFORE any launch and is meant to stop the in-place weld from being
        # assembled at all.
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng(spec["label"]))
        aliased = {volume: getattr(fields, volume)
                   for volume in module.SCRATCH_VOLUMES}
        try:
            if module.ARM == "pml":
                tables = module.folded_complex_offdiag_fused_electric_pair_tables(pml)
                module.assert_scratch_is_disjoint(fields, aliased, tables)
            else:
                module.assert_scratch_is_disjoint(fields, aliased)
            refused, why_refused = False, None
        except ValueError as exc:
            refused, why_refused = True, str(exc)[:300]
        scoreboard.setdefault("scratch_aliased_to_storage", {
            "why": "the same defect as foreign_read_from_live_buffer, from the HOST "
                   "side: binding the scratch to the storage IS the in-place weld. "
                   "Scored as a refusal because the check runs before any launch",
            "expected": "REFUSED", "applies_to": "both",
            "scored_on_value_class": "n/a", "arms": []})
        scoreboard["scratch_aliased_to_storage"]["arms"].append(
            {"spec": spec["label"], "refused": refused, "reason": why_refused})
    for name in HOST_MUTATIONS:
        entry = scoreboard[name]
        arms = entry["arms"]
        if name == "rotation_skipped":
            entry["outcome"] = ("CAUGHT" if arms and all(a["caught"] for a in arms)
                                else "UNCAUGHT")
            entry["as_required"] = entry["outcome"] == "CAUGHT"
        else:
            entry["outcome"] = ("REFUSED" if arms and all(a["refused"] for a in arms)
                                else "ADMITTED")
            entry["as_required"] = entry["outcome"] == "REFUSED"
        log(f"[mutation] {name:34s} {entry['outcome']:14s} ({len(arms)} armed)")
    return {"scoreboard": scoreboard,
            "passed": all(entry["as_required"] for name, entry in scoreboard.items()
                          if not name.startswith("_"))}


# ---------------------------------------------------------------------------
# The structural legs
# ---------------------------------------------------------------------------

def leg_lift(licence: Dict[str, Any]) -> Dict[str, Any]:
    """Both emitted sources against the certified text, and the declared edits.

    A device gate that did not check the SPLICE would certify whatever the emitter
    happened to produce.
    """
    from meep_gpu.cuda_kernels import complex_offdiag_stencil_weld as weld  # noqa: PLC0415

    arm = _arm(licence)
    record: Dict[str, Any] = {"expansion_arm": arm, "products": {}}
    ok = True
    for name, module in products().items():
        source = module.kernel_source((1, 1, 1, 1, 1, 1), arm)
        # THE CERTIFIED KERNEL'S OWN BODY, not its prelude. The prelude is the shared
        # helper block (cf_load, the multiply arm, pml_apply / no_pml_apply) which the
        # lift rewrites into pure form by declared edit; what must arrive VERBATIM is
        # the curl's three component blocks -- their stencils, ghost calls, phase
        # arguments, fold masks and coefficient pairings.
        certified = weld.certified_curl_source(module.ARM, arm)
        curl_body = entry_body(certified)
        edited = {edit["line"].splitlines()[0].strip()
                  for edit in module.LIFT_EDITS}
        missing = []
        for line in curl_body.splitlines():
            stripped = line.strip()
            if not stripped or stripped in edited:
                continue
            # The two blocks that tie the certified body to a THREAD, which the cell
            # lift necessarily drops; both are declared edits and are skipped by
            # prefix because the emitter respells them rather than deleting them.
            if stripped.startswith(("pml_apply(", "no_pml_apply(",
                                    "int idx = blockIdx", "if (idx >=",
                                    "int k = idx", "int j = (idx", "int i = idx",
                                    "const int idx = blockIdx", "const int k = idx",
                                    "const int j = (idx", "const int i = idx")):
                continue
            if line not in source:
                missing.append(line)
        body = kernel_body(source, module.KERNEL_NAME)
        # THE COMPLEX CLAIM, READ OFF THE TEXT. Every line that reads a plane phase
        # must read it as the LEFT operand of the certified full complex multiply.
        # A bare word scale is byte-wrong on signed zeros and NOWHERE ELSE, so it is
        # invisible to any leg that only compares ordinary values -- which is why it
        # is asserted on the source here as well as armed on the device below.
        phase_reads = [line.strip() for line in source.splitlines()
                       if ("weld.near_phase_" in line or "weld.far_phase_" in line)
                       # The struct BINDINGS are not reads: `weld.near_phase_x =
                       # near_phase_x;` copies the launcher's argument in and
                       # multiplies nothing.
                       and not line.strip().startswith("weld.")]
        unguarded_phase = [line for line in phase_reads
                           if "mul_coefficient_left(" not in line]
        entry = {
            "kernel": module.KERNEL_NAME,
            "arm": module.ARM,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "corpus_digest": module.corpus_digest(),
            "lift_edits": len(module.LIFT_EDITS),
            "certified_curl_lines_missing": missing,
            # THE SCRATCH DISCIPLINE, READ OFF THE TEXT. Scoped to the WHOLE source
            # rather than the kernel body: the resolution lives in the __device__
            # helpers ABOVE the entry point, so a body-only scan would check the one
            # place the defect cannot be.
            "no_direct_flux_density_load": not any(
                f"{name}[" in source for name in ("f0", "f1", "f2")),
            "no_store_into_pre_launch_state": not any(
                f"cf_store({name}," in source for name in ("f0", "f1", "f2",
                                                           "u0", "u1", "u2")),
            "stores_only_into_scratch": all(
                f"cf_store({name}_out, idx," in body
                for name in (("f0", "f1", "f2", "u0", "u1", "u2")
                             if module.ARM == "pml" else ("f0", "f1", "f2"))),
            "phase_reads": len(phase_reads),
            "phase_reads_not_through_mul_coefficient_left": unguarded_phase,
            # ONE PER D COMPONENT, in the resolution -- which lives above the entry
            # point, so this too is counted on the whole source.
            "wall_clear_is_both_words": source.count("value = cf_zero();") == 3,
        }
        entry["passed"] = (not missing and entry["no_direct_flux_density_load"]
                           and entry["no_store_into_pre_launch_state"]
                           and entry["stores_only_into_scratch"]
                           and entry["phase_reads"] > 0
                           and not unguarded_phase
                           and entry["wall_clear_is_both_words"])
        ok = ok and entry["passed"]
        record["products"][name] = entry
    record["passed"] = ok
    return record


def leg_refusal(licence: Dict[str, Any]) -> Dict[str, Any]:
    """The configurations both products refuse BY NAME.

    The one failure mode no bit comparison catches: a predicate that admits what the
    launch cannot serve is invisible to a byte gate, because the byte gate only ever
    runs on rows the predicate admitted.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    def make(symmetry=(), rows=True, layer=True, k=(0.0, 0.0, 0.0),
             complex_storage=True, pml_storage=None):
        """``pml_storage`` FOLLOWS THE LAYER unless a case overrides it.

        The complex no-absorber curl refuses a Fields in PML storage mode beside an
        inert layer BY NAME, so a fold-free fixture built with storage on would
        refuse for THAT reason and a case meaning to measure a different clause
        would silently be measuring this one.
        """
        if pml_storage is None:
            pml_storage = layer
        grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 1.6), courant=0.35, xp=cp,
                    k_point=tuple(k),
                    symmetry=tuple(Mirror(a, p) for a, p in symmetry))
        thickness = (tuple((0, 0) if grid.shape[axis] < 6
                           else (0, 2) if grid.is_mirrored(axis)
                           else (2, 2) for axis in range(3))
                     if layer else ((0, 0),) * 3)
        pml = PML(grid=grid, thickness=thickness)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        if pml_storage:
            fields.enable_pml_storage()
        shape = tuple(int(n) for n in grid.shape)
        names = ("Ex", "Ey", "Ez")
        eps = {n: cp.full(shape, v, cp.float32) for n, v in zip(names, (2., 2.5, 3.))}
        inv = {n: cp.full(shape, np.float32(1.0 / v), cp.float32)
               for n, v in zip(names, (2., 2.5, 3.))}
        off = ({"Ex": {"Ey": cp.asarray(np.full(shape, 0.1, np.float32))}}
               if rows else None)
        fields.set_epsilon_volumes(eps, inv, chi1inv_offdiagonal=off)
        return fields, pml, grid

    folded, no_pml = products()["folded"], products()["no_pml"]

    def ask(module, fields, pml, grid, sources):
        bound = _licence_for(module, licence)
        call = getattr(module, f"covers_{module.FAMILY[len('cuda_'):]}")
        return call(fields, pml, grid, sources, bound["license"], _POLICY_NAME,
                    curl_license=bound["curl_license"])

    cases: List[Dict[str, Any]] = []
    fields, pml, grid = make(symmetry=(("Y", 1),))
    cases.append({"case": "folded admits its own cell",
                  "covered": ask(folded, fields, pml, grid, ())[0], "want": True})
    covered, why = ask(no_pml, fields, pml, grid, ())
    cases.append({"case": "no_pml refuses a folded grid", "covered": covered,
                  "want": False, "reason": why})
    covered, why = ask(folded, fields, pml, grid, None)
    cases.append({"case": "folded refuses an undeclared source set",
                  "covered": covered, "want": False, "reason": why})
    source = VolumeSource(grid=grid, component="Ez", center=(0., 0., 0.),
                          size=(0., 0., 0.),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    covered, why = ask(folded, fields, pml, grid, (source,))
    cases.append({"case": "folded refuses an electric deposit", "covered": covered,
                  "want": False, "reason": why,
                  "names_the_repair_clause": "off-diagonal chi1inv row" in str(why)})
    fields, pml, grid = make(symmetry=(("Y", 1),), rows=False)
    covered, why = ask(folded, fields, pml, grid, ())
    cases.append({"case": "folded refuses a diagonal run", "covered": covered,
                  "want": False, "reason": why})
    fields, pml, grid = make(symmetry=(("Y", 1),), complex_storage=False)
    covered, why = ask(folded, fields, pml, grid, ())
    cases.append({"case": "folded refuses REAL storage", "covered": covered,
                  "want": False, "reason": why})
    fields, pml, grid = make(layer=False, k=(0.3892, 0.1597, 0.0))
    cases.append({"case": "no_pml admits its own cell",
                  "covered": ask(no_pml, fields, pml, grid, ())[0], "want": True})
    covered, why = ask(folded, fields, pml, grid, ())
    cases.append({"case": "folded refuses a fold-free grid", "covered": covered,
                  "want": False, "reason": why})
    fields, pml, grid = make(layer=False, k=(0.3892, 0.1597, 0.0),
                             complex_storage=False)
    covered, why = ask(no_pml, fields, pml, grid, ())
    cases.append({"case": "no_pml refuses REAL storage", "covered": covered,
                  "want": False, "reason": why})
    fields, pml, grid = make(layer=True, k=(0.3892, 0.1597, 0.0))
    covered, why = ask(no_pml, fields, pml, grid, ())
    cases.append({"case": "no_pml refuses a live absorber", "covered": covered,
                  "want": False, "reason": why})
    fields, pml, grid = make(layer=False, k=(0.3892, 0.1597, 0.0), rows=False)
    covered, why = ask(no_pml, fields, pml, grid, ())
    cases.append({"case": "no_pml refuses a diagonal run", "covered": covered,
                  "want": False, "reason": why})
    passed = all(case["covered"] == case["want"] for case in cases)
    return {"cases": cases, "passed": bool(passed)}


def leg_guard_control(spec: Dict[str, Any], steps: int,
                      licence: Dict[str, Any]) -> Dict[str, Any]:
    """``--fmad=false`` is CORRECTNESS here, and this measures it rather than citing it.

    The same source compiled WITHOUT the guard must diverge: if it does not, the
    option is not load-bearing on this kernel and the record should say so rather
    than inheriting a sibling's claim.
    """
    from meep_gpu.cuda_kernels.coverage import offdiag_row_mask  # noqa: PLC0415

    module = products()[spec["product"]]
    fields, _grid, _pml, _ = build(spec, "uniform", case_rng(spec["label"]))
    mask = offdiag_row_mask(fields)
    unguarded = cp.RawKernel(module.kernel_source(mask, _arm(licence)),
                             module.KERNEL_NAME, options=())
    record = run_case(spec, "uniform", steps=min(steps, 8), licence=licence,
                      kernel=unguarded, label_suffix="/unguarded")
    return {"spec": spec["label"], "diverged": not record["bit_identical"],
            "first_divergence": record["first_divergence"],
            "note": "a NON-divergence here is a finding, not a failure: it would "
                    "mean the compiler contracted nothing in this body at this "
                    "courant, and the record must say so rather than claim the "
                    "guard is load-bearing"}


def leg_adversarial_reach(specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Does the adversarial class actually put signed zeros where the parity reads?

    THE THREE COMPLEX ARMS ARE ONLY HONEST IF IT DOES. A class that claimed to plant
    signed zeros and did not would score those arms UNCAUGHT and report three holes
    that were really three vacuous fixtures -- the failure this campaign has already
    paid for once. So the planted state is counted at the planes the resolution
    reads: stored row 0 (both fills' destination), stored row NEAR_SOURCE_ROW (the
    near fill's source) and each axis's reflect row.
    """
    from meep_gpu.cuda_kernels import complex_offdiag_stencil_weld as weld  # noqa: PLC0415

    out: List[Dict[str, Any]] = []
    for spec in specs:
        fields, grid, _pml, _ = build(spec, COMPLEX_ARM_CLASS,
                                      case_rng(spec["label"]))
        plan = weld.fill_plan(grid)
        counts = {"row_0": 0, "near_source_row": 0, "reflect_rows": 0}
        for component in ("Dx", "Dy", "Dz"):
            volume = to_host(getattr(fields, component))
            for axis in range(3):
                if volume.shape[axis] <= weld.NEAR_SOURCE_ROW:
                    continue
                for key, index in (("row_0", 0),
                                   ("near_source_row", weld.NEAR_SOURCE_ROW),
                                   ("reflect_rows", int(plan["reflect"][axis]))):
                    if index < 0:
                        continue
                    plane = np.take(volume, index, axis=axis)
                    raw = np.ascontiguousarray(plane).view(np.float32).view(np.uint32)
                    counts[key] += int(np.count_nonzero(raw == 0x80000000))
        out.append({"spec": spec["label"], "negative_zero_words": counts,
                    "reaches_every_plane": all(v > 0 for k, v in counts.items()
                                               if k != "reflect_rows"
                                               or any(int(r) >= 0
                                                      for r in plan["reflect"]))})
    return {"fixtures": out, "passed": all(r["reaches_every_plane"] for r in out)}


def save(results: Dict[str, Any], path: str) -> None:
    """Serialize, with the WELD stamped on every write.

    ``gate_provenance.stamp`` records the sha256 of every repo module this process
    imported, which is what ``rebind_cuda_welds.py`` binds the certification entry
    to. Without it the record would carry a verdict and no way for the merge bar to
    notice when the files it ran on moved.
    """
    gate_provenance.stamp(results)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _warm_memo(specs: Sequence[Dict[str, Any]], licence: Dict[str, Any]) -> None:
    """Compile and launch every kernel this file drives, once, off the counted path."""
    for spec in specs:
        module = products()[spec["product"]]
        fields, grid, pml, dtdx = build(spec, "uniform", np.random.default_rng(1))
        scratch = _make_scratch(module, fields)
        _launch_once(module, fields, grid, pml, dtdx, scratch, licence)
    cp.cuda.runtime.deviceSynchronize()


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _POLICY_NAME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--repeats", type=int, default=SCHEDULE_REPEATS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True,
                        help="the policy NAME the predicates are asked with; both "
                             "curl halves refuse an unnamed one because an "
                             "expansion licence is policy-conditional")
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    _POLICY_NAME = args.subnormal_policy

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    started = time.perf_counter()
    results: Dict[str, Any] = {
        "gate": "cuda_complex_offdiag_stencil_welds",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("do ONE launch of fused_electric_pair_pml_complex_folded_offdiag "
                     "and one of fused_electric_pair_no_pml_complex_offdiag leave "
                     "every stored volume byte-identical to stepping's own D-seam "
                     "pass sequence inside a complete driver step -- at every block "
                     "size, on every repeat, under both float32 subnormal policies, "
                     "and on an adversarial complex word catalogue?"),
        "what_it_does_not_claim": [
            "throughput: the box is shared and nothing here is timed",
            "dispatch: fastpath.plan_fast_path still returns None on every branch",
            "the resolution's arithmetic against the driver's pass order, which is "
            "measured off-device by probe_cuda_complex_offdiag_scratch_weld.py",
        ],
        "budget_steps": args.steps,
        "schedule_repeats": args.repeats,
        "block_sizes": list(BLOCK_SIZES),
        "value_classes": list(VALUE_CLASSES),
        "legs": {
            "S1": "complete driver steps, every stored volume, per step",
            "S2": "S1 at every thread-block size",
            "S3": "ONE launch against the driver's five D-seam passes alone, on "
                  "the planted adversarial word catalogue -- the only leg on which "
                  "the two complex parity spellings are decidable, because a "
                  "COMPLETE step's magnetic half overwrites H before the D seam is "
                  "entered and the catalogue never reaches the parity (measured "
                  "2026-09-02: both parity mutants UNCAUGHT with first_divergence "
                  "null over complete steps, 11-16 stored words moved at the seam)",
        },
    }
    if cp is None or args.no_device:
        log("[structural] no device: only the source legs run")
        results["device_mode"] = False
        results["driver_order"] = leg_driver_order()
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        return 0

    results["device_mode"] = True
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
        args.subnormal_policy, _REPO_API)
    results["policy_asked_of_the_predicates"] = _POLICY_NAME
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    # THE HOST PROBE IS IMPORTED ONLY NOW, and the assertion is the reason: its
    # module scope carries os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "keep"),
    # which on an un-set environment would pin the wrong policy for a flush run.
    # host_probe() shields the import; this MEASURES that the shield held rather
    # than trusting it, because a variable this run does not control is a variable
    # the bytes might execute under.
    before = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    hostprobe = host_probe()
    after = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    if before != after:
        raise SystemExit(
            f"importing the host probe moved MEEP_GPU_SUBNORMAL_POLICY from "
            f"{before!r} to {after!r}; the policy this run installed is not the one "
            f"its bytes would execute under")
    results["host_probe_policy_unmoved"] = {
        "before": before, "after": after,
        "adversarial_word_pairs": [list(pair)
                                   for pair in hostprobe.ADVERSARIAL_WORDS],
        "catalogue_source": "probe_cuda_complex_offdiag_scratch_weld"
                            ".ADVERSARIAL_WORDS -- imported, not respelled"}

    licence = licences(args.subnormal_policy)
    results["licences"] = {
        half: {key: verdict.get(key) for key in
               ("arm", "refusals", "policy_reasons", "record", "record_sha256",
                "record_subnormal_policy", "arbiter")}
        for half, verdict in licence.items()}
    results["expansion_arm"] = _arm(licence)
    unusable = {half: verdict for half, verdict in licence.items()
                if not verdict.get("arm") or verdict.get("refusals")
                or verdict.get("policy_reasons")}
    if unusable:
        results["status"] = "refused: an expansion licence is unusable"
        results["verdict"] = {"passed": False, "why": {
            half: {"arm": v.get("arm"), "refusals": v.get("refusals"),
                   "policy_reasons": v.get("policy_reasons")}
            for half, v in unusable.items()}}
        save(results, args.out)
        return 1
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    by_label = {spec["label"]: spec for spec in SPECS}

    results["driver_order"] = leg_driver_order()
    log(f"[driver] order matches driver.py: {results['driver_order']['passed']}")
    results["lift"] = leg_lift(licence)
    log(f"[lift] both emitters are the certified bytes: {results['lift']['passed']}")
    results["refusal"] = leg_refusal(licence)
    log(f"[refusal] the seam clauses hold: {results['refusal']['passed']}")
    results["adversarial_reach"] = leg_adversarial_reach(specs)
    log(f"[adversarial] the catalogue reaches every read plane: "
        f"{results['adversarial_reach']['passed']}")
    save(results, args.out)

    _warm_memo(specs, licence)

    # ------------------------------------------------------------------ S1
    s1: List[Dict[str, Any]] = []
    for spec in specs:
        for value_class in VALUE_CLASSES:
            for repeat in range(args.repeats):
                record = run_case(spec, value_class, args.steps, licence,
                                  repeat=repeat, label_suffix=f"/r{repeat}")
                s1.append(record)
                log(f"[S1] {record['label']:56s} "
                    f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'} "
                    f"launches={record['launch_counts']['memo_total']} "
                    f"subnormals={record['operand_census']['subnormals']} "
                    f"negzero={record['operand_census'].get('negative_zeros')}")
                save({**results, "S1_subject": s1}, args.out)
    results["S1_subject"] = s1

    # ------------------------------------------------------------------ S2
    s2: List[Dict[str, Any]] = []
    for spec in specs:
        for threads in BLOCK_SIZES:
            record = run_case(spec, COMPLEX_ARM_CLASS, steps=min(args.steps, 12),
                              licence=licence, threads=threads,
                              label_suffix=f"/b{threads}")
            s2.append(record)
            log(f"[S2] {record['label']:56s} "
                f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'}")
        save({**results, "S2_block_sizes": s2}, args.out)
    results["S2_block_sizes"] = s2

    # ------------------------------------------------------------------ S3
    results["S3_planted_seam"] = leg_planted_seam(specs, licence)
    save(results, args.out)

    results["guard_control"] = [
        leg_guard_control(by_label[label], args.steps, licence)
        for label in ("fold_y_odd_all_rows", "no_pml_walls_all_rows")
        if label in by_label]
    save(results, args.out)

    if not args.skip_mutations:
        results["mutations"] = leg_mutations(
            [by_label[label] for label in MUTATION_SPEC_LABELS], args.steps, licence)
        save(results, args.out)

    # ------------------------------------------------------------------ verdict
    band = [r for r in s1 if r["value_class"] == "subnormal_band"]
    uniform = [r for r in s1 if r["value_class"] == "uniform"]
    adversarial = [r for r in s1 if r["value_class"] == COMPLEX_ARM_CLASS]
    clauses = {
        "S1 every case is bit-identical for the whole budget":
            bool(s1) and all(r["bit_identical"] for r in s1),
        "S1 every case is identical on every repeat":
            bool(s1) and len({(r["spec"], r["value_class"],
                               r["bit_identical"]) for r in s1}) ==
            len({(r["spec"], r["value_class"]) for r in s1}),
        "S1 the launch counts agree between two independent counters":
            bool(s1) and all(r["launch_counts_agree"] for r in s1),
        "S1 nothing read-only drifted": all(not r["read_only_drift"] for r in s1),
        "S1 every case moved the state it must move":
            all(r["non_vacuous"] for r in s1),
        "S1 covers both products":
            {r["product"] for r in s1} == {"folded", "no_pml"},
        "S2 identical at every block size":
            bool(s2) and all(r["bit_identical"] for r in s2),
        "S3 identical at the planted D seam, where the complex spellings differ":
            results["S3_planted_seam"]["passed"],
        "the subnormal band really contains subnormals":
            bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
        "the uniform class contains none":
            bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                  for r in uniform),
        "the adversarial class really contains negative zeros":
            bool(adversarial) and all(
                r["operand_census"].get("negative_zeros", 0) > 0
                for r in adversarial),
        "the adversarial class reaches every plane the resolution reads":
            results["adversarial_reach"]["passed"],
        "the driver order is the driver's": results["driver_order"]["passed"],
        "the emitted sources are the certified bytes": results["lift"]["passed"],
        "the seam clauses refuse what the launch cannot serve":
            results["refusal"]["passed"],
        "every mutation scored as required":
            results.get("mutations", {}).get("passed", True),
        # THE CAMPAIGN-LEVEL CLAUSE. An arm undecidable under THIS policy must be
        # decidable under another, and the artifact must name which; an arm that
        # were undecidable everywhere would be an untested spelling wearing a
        # measurement's clothes.
        "every policy-conditional arm is decidable under some policy":
            all(bool(entry.get("decidable_under"))
                for entry in POLICY_CONDITIONAL_ARMS.values()),
    }
    results["policy_conditional_arms"] = {
        name: {"decidable_under": list(entry["decidable_under"]),
               "decidable_under_this_policy":
                   _POLICY_NAME in tuple(entry["decidable_under"]),
               "why": entry["why"]}
        for name, entry in POLICY_CONDITIONAL_ARMS.items()}
    results["verdict"] = {
        "clauses": clauses, "passed": all(clauses.values()),
        "kernels": sorted({products()[spec["product"]].KERNEL_NAME
                           for spec in specs}),
        "denominators": {
            "S1_cases": len(s1), "S2_cases": len(s2),
            "S3_cases": len(results["S3_planted_seam"]["cases"]),
            "fixtures": len(specs), "block_sizes": len(BLOCK_SIZES),
            "value_classes": len(VALUE_CLASSES),
            "repeats": args.repeats, "steps_per_case": args.steps,
        },
    }
    # THE CANONICAL VERDICT, in the spelling the fleet's readers agree on.
    # ``record_stencil_welds.py`` transcribes a campaign only where EVERY policy leg
    # carries ``canonical_verdict.released`` true, and ``gate_provenance
    # .read_verdict`` already reads seven spellings across three tracks; writing the
    # canonical one here means this gate needs no eighth.
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"]),
        "reasons": [clause for clause, value in clauses.items() if not value],
        "what_it_licenses": (
            "two CUDA kernels measured byte-identical to the array path's D-seam "
            "pass sequence over complete driver steps under COMPLEX64 storage, at "
            "every block size, on an adversarial signed-zero word catalogue as well "
            "as ordinary and subnormal values, under both float32 subnormal "
            "policies, with the mutation battery two-sided and its three "
            "complex-specific arms armed. It licenses NO throughput claim and NO "
            "dispatch claim"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 2)
    save(results, args.out)
    for clause, value in clauses.items():
        log(f"[verdict] {'PASS' if value else 'FAIL'}  {clause}")
    log(f"[verdict] {'PASSED' if results['verdict']['passed'] else 'FAILED'} "
        f"({results['elapsed_s']} s) -> {args.out}")
    return 0 if results["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
