"""Byte gate for the Metal COMPLEX/BLOCH H->D pair: complex ``update_H`` welded into
complex ``step_D``, one dispatch, scratch output and the post-launch rotation.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.complex_fused_hd_pair.metal_complex_fused_hd_pair_coverage`
admits, ONE dispatch of ``complex_fused_hd_pair_step`` followed by the H/f_w_H
rotation leaves the engine in a state that is BIT-IDENTICAL, PER COMPLETE DRIVER STEP,
to

  * the NumPy array path (``stepping.update_H`` then ``stepping.step_D``, inside the
    driver's full pass list), and
  * the two SEPARATELY CERTIFIED Metal products it replaces --
    ``complex_fields.plan_complex_constitutive`` on ``H`` and
    ``complex_fields.plan_complex_pml_curl`` on ``step_D``,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere, and
every "0 differing words" in this artifact stands beside a control that BITES.

WHAT IS NEW AND MEASURED HERE, beyond what the two certified singles already carry:
the SEAM. The curl half's own-cell magnetic operand is the register the constitutive
half just produced, and each of its six BACKWARD taps is a RECOMPUTE of the certified
complex ``update_H`` at the neighbour cell from PRE-LAUNCH state. Legs ``product`` and
``null_control`` measure that those two things equal what the array path loads, and
mutations ``own_cell_reads_pre_launch_h`` and ``halo_tap_reads_pre_launch_h`` are the
two defects the whole shape exists to avoid.

THE LEGS

  1  binding_ceiling  the shipped signature (31 bindings) COMPILES on BOTH expansion
                      arms with the count read off the emitted text; the same shape at
                      32, 33 and 34 is REFUSED with the platform's own message; a
                      generic float2 sweep bisects the ceiling; the three refuted
                      signatures (one more pointer, unshared kms, separate scalars) and
                      the re/im split form all fail to compile
  2  lift             the h_cell body is the certified complex H constitutive with
                      EXACTLY the declared edits and its arithmetic lines untouched;
                      the curl half is the certified complex curl with EXACTLY the nine
                      declared load redirections -- measured by REVERSING them and
                      requiring the certified text back, byte for byte
  3  driver_order     the only statement between the update_H and step_D consults is
                      the electric withdraw loop (h_to_d_seam.driver_seam_fact), and
                      REPLACES / SEAM are the driver's span and seam name
  4  refusal          a standing integrated electric withdraw is refused BY NAME; a
                      non-integrated electric source and a magnetic integrated one are
                      admitted; an undeclared source set, a missing residency, real
                      storage, a cylindrical grid, a folded axis, a beta run and a
                      registered susceptibility are all refused
  5  product          eight configurations x twelve complete driver steps against both
                      references, all 24 stored volumes, launch counts asserted per
                      cycle, Residency.verify() empty after every seam
  6  value_class      the +-0 lattice, with the sign floor in place of the movement one
  7  null_control     the in-place control: the rotation undone, so the next launch and
                      the host read the pre-launch H. It MUST diverge, or every 0 above
                      is vacuous
  8  purity           a static read/write ledger over the fused source: nothing the
                      launch writes is read by it, and h_cell stores nothing
  9  divide_spelling  the complex-by-real divide fact, re-measured on THIS kernel: the
                      emitted text contains no float division at all (the PML
                      reciprocal is precomputed on the host into sinv_a), and numpy's
                      own complex64/float32 behaviour is measured beside it on a
                      signed-zero-rich table so the fact is carried forward rather than
                      cited
 10  corpus_rows      every row of the cell, lifted from the standing census by its
                      recorded arms, its grid rebuilt from its recorded configuration,
                      admitted through the predicate and DRIVEN
 11  arbitration      what the SHIPPED composer installs on the neighbouring seams of
                      each corpus row today -- the measurement behind INSTALLABLE=False
 12  mutations        armed kernel and host defects through the launch-counted harness,
                      each on both value classes, each must-catch or a measured
                      equivalence
 13  disarm           the shipped bytes through the mutation path must NOT diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.

THIS GATE IS ALSO A CENSUS BATTERY (``SUBJECT_PACKAGE``, :func:`evaluate`,
:func:`runtime_reasons`), so ``measure_predicate_coverage.py --battery
gate_metal_complex_fused_hd_pair`` can re-lift a corpus row in its own interpreter and
run this product's identity on it. Leg ``corpus_rows`` does not use that route: it
rebuilds each row's GRID from the census's recorded configuration rather than re-running
the row's own script, and says so in its own record.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import h_to_d_seam  # noqa: E402
import metal_composition_matrix as matrix  # noqa: E402
import metal_gate_kit as kit  # noqa: E402
import metal_value_classes as values  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields,
    complex_fused_hd_pair as family,
    launch as metal_launch,
    shaders,
    subnormal,
    templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source,
)
from meep_gpu.pml import PML  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words

#: The census this gate's cell is read from, and the seam record that supplies each
#: row's withdraw fact. NAMED, never derived from a directory listing: a gate that
#: picked "the newest" census would silently change its own denominator.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The cell: the two arms the census's own ``plan_step.selected`` records for a row
#: this product would span.
CELL_ARMS: Tuple[str, str] = ("complex/Bloch", "complex/Bloch")

#: The battery's declaration for ``measure_predicate_coverage``'s subject digest.
SUBJECT_PACKAGE = "metal_kernels"

#: Twelve complete steps, the budget every fused-pair gate on this backend runs: the
#: classes this gate exists for COMPOUND (f_w_H and fu_D are state), and a wrong
#: coefficient index needs several steps to reach the low bits of the interior.
STEPS = 12

#: Every volume one complete step can touch on a complex Cartesian run.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's complete step (driver.py:3279-3330), in order. The two passes this
#: product replaces are marked; the rest run on the HOST in every engine, so the
#: comparison is of one seam inside one step and not of two different steppers.
PASSES: Tuple[Tuple[str, Callable[..., Any], bool], ...] = (
    ("step_B", lambda f, p: stepping.step_B(f, p), False),
    ("fill_B", lambda f, p: stepping.fill_symmetry_bc_B(f), False),
    ("zero_metal_B", lambda f, p: stepping.zero_metal_B(f), False),
    ("fill_folded_far_ghosts_B", lambda f, p: stepping.fill_folded_far_ghosts_B(f),
     False),
    ("update_H", lambda f, p: stepping.update_H(f, p), True),
    ("step_D", lambda f, p: stepping.step_D(f, p), True),
    ("fill_D", lambda f, p: stepping.fill_symmetry_bc_D(f), False),
    ("zero_metal_D", lambda f, p: stepping.zero_metal_D(f), False),
    ("fill_folded_far_ghosts_D", lambda f, p: stepping.fill_folded_far_ghosts_D(f),
     False),
    ("update_E", lambda f, p: stepping.update_E(f, p), False),
)

#: (name, cell, boundaries, k_point). The k values exercise the three classes the
#: certified complex gate pins BY NAME -- k = 0 (no phase object is built at all and
#: the kernel must be byte-identical to the plain complex path), a generic interior
#: point (both planes of the rotation live), and the BRILLOUIN EDGE where the phase is
#: exactly -1+0j (the one value at which a plane-wise collapse is invisible on random
#: data) -- times the axes. ``periodic_ky`` and ``periodic_kxyz`` are here for the
#: reason the complex gate records: without them the ``y`` phase branch and the
#: ``ph111`` specialisation are emitted and never launched.
CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any,
                     Tuple[float, float, float]], ...] = (
    ("periodic_k0", (1.2, 1.0, 0.9), "periodic", (0.0, 0.0, 0.0)),
    ("periodic_kx", (1.2, 1.0, 0.9), "periodic", (0.3, 0.0, 0.0)),
    ("periodic_edge_x", (1.2, 1.0, 0.9), "periodic", (0.5, 0.0, 0.0)),
    ("periodic_ky", (1.2, 1.0, 0.9), "periodic", (0.0, 0.4, 0.0)),
    ("periodic_kxz", (1.2, 1.0, 0.9), "periodic", (0.3, 0.0, -0.25)),
    ("periodic_kxyz", (1.2, 1.0, 0.9), "periodic", (0.3, -0.4, 0.2)),
    ("metallic_xy_kz", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.4)),
    ("metallic_all_k0", (1.1, 1.0, 0.9), "metallic", (0.0, 0.0, 0.0)),
)

#: The substrate every mutation is planted on, BY NAME rather than by index: a case
#: added to :data:`CONFIGS` must not silently repoint the mutation leg.
MUTATION_CASE = "periodic_kxz"


def config(name: str) -> Tuple[str, Tuple[float, float, float], Any,
                               Tuple[float, float, float]]:
    for entry in CONFIGS:
        if entry[0] == name:
            return entry
    raise KeyError(f"no case named {name!r}; the matrix is "
                   f"{[entry[0] for entry in CONFIGS]}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a complex128
    scalar, so the product is a full complex multiply whose imaginary part comes out
    ``-0.0`` regardless of what ``imag`` held -- measured while writing
    ``gate_metal_complex``, where it classified NumPy as matching no expansion arm.
    Every complex construction in this file goes through the word view.
    """
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


def build_grid(cell: Sequence[float], boundaries: Any,
               k_point: Sequence[float], courant: float = 0.35,
               resolution: float = 10.0,
               dimensions: Optional[int] = None) -> Any:
    """A Grid with COMPLEX-capable extents, its ``dimensions`` DERIVED from the cell.

    A zero-extent axis at ``dimensions=3`` is a ``ValueError`` from the engine, so the
    dimensionality is the number of axes the cell actually resolves. That is how the
    corpus's 1-D and 2-D rows rebuild at the shapes the census recorded.
    """
    resolved = (sum(1 for extent in cell if float(extent) > 0.0)
                if dimensions is None else int(dimensions))
    return Grid(resolution=resolution, cell_size=tuple(float(c) for c in cell),
                boundaries=boundaries, dimensions=resolved, courant=courant,
                k_point=tuple(float(k) for k in k_point), xp=np)


def build_from_grid(grid: Any, seed: int,
                    value_class: str = values.UNIFORM) -> Tuple[Any, Any]:
    """Fields (complex storage, PML storage) and a PML, seeded in the physical band.

    THE SPLIT-FIELD AND AUXILIARY VOLUMES ARE SEEDED TOO. A zero auxiliary is a fixed
    point of the split-field recurrence for one step and would hide a wrong
    coefficient index; ``f_w_H`` in particular is what the constitutive half reads
    before it writes, and the whole scratch design turns on that read coming from the
    PRE-LAUNCH buffer.
    """
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    pml = PML(grid=grid,
              thickness=tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                              for axis in range(3)))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if value_class == values.PM_ZERO_LATTICE:
            array[...] = values.pm_zero_lattice_complex(array.shape)
            continue
        real = (rng.standard_normal(grid.shape) * np.float32(0.37)).astype(np.float32)
        imag = (rng.standard_normal(grid.shape) * np.float32(0.29)).astype(np.float32)
        # A SPRINKLE OF SIGNED ZEROS IN THE UNIFORM CLASS TOO. The complex helpers'
        # zero cross terms only misbehave where an operand plane is an exact zero, and
        # a purely random band would leave that whole family of defects unarmed on the
        # class the product leg runs.
        real.reshape(-1)[::17] = np.float32(-0.0)
        real.reshape(-1)[7::23] = np.float32(0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        imag.reshape(-1)[11::29] = np.float32(0.0)
        array[...] = complex_from_planes(real, imag)
    return fields, pml


def build(label: str, seed: int, value_class: str = values.UNIFORM) -> Tuple[Any, Any]:
    _name, cell, boundaries, k_point = config(label)
    return build_from_grid(build_grid(cell, boundaries, k_point), seed, value_class)


def state_of(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in left:
        count = differing(left[name], right[name])
        if count:
            out[name] = count
    return out


def state_census(fields: Any) -> int:
    return sum(int(subnormal.census(array)) for array in state_of(fields).values())


def volume_source(fields: Any, component: str, integrated: bool) -> Any:
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


def bound_expansion() -> Tuple[Optional[str], str]:
    probe = complex_fields.load_expansion_probe()
    arm = complex_fields.expansion_from_probe(probe) if probe else None
    return arm, (arm or "FMA_V1")


# ---------------------------------------------------------------------------
# The three engines
# ---------------------------------------------------------------------------

class ArrayEngine:
    """The NumPy array path, the driver's pass list, no device."""

    def __init__(self, fields: Any, pml: Any) -> None:
        self.fields, self.pml = fields, pml

    def step(self) -> None:
        for _name, function, _replaced in PASSES:
            function(self.fields, self.pml)


class HalfLatticePml:
    """A PML that answers the HALF-INTEGER volumes under the integer names.

    The host mutation for the one silent failure this signature has: both halves share
    one ``kms`` group because they sit on the same Yee sub-lattice, and binding the
    half-integer set instead COMPILES and produces a smooth, converged,
    half-cell-wrong absorber profile. Only the CONSTITUTIVE group is swapped, which is
    the mistake a reader would actually make.
    """

    def __init__(self, pml: Any) -> None:
        self._pml = pml

    def __getattr__(self, name: str) -> Any:
        if name.startswith("kps_") and not name.endswith("_h"):
            return getattr(self._pml, name + "_h")
        return getattr(self._pml, name)


class ProductEngine:
    """The device engine: host passes everywhere except the seam, which the plan owns."""

    def __init__(self, fields: Any, pml: Any,
                 functions: Optional[Dict[str, Any]] = None,
                 rotate: bool = True, half_lattice: bool = False) -> None:
        self.fields, self.pml = fields, pml
        self.residency = Residency()
        self.plan = family.plan_metal_complex_fused_hd_pair(
            fields, HalfLatticePml(pml) if half_lattice else pml, sources=(),
            residency=self.residency, functions=functions)
        if self.plan is None:
            raise AssertionError("the product refused an admitted configuration")
        self.rotate = rotate
        self.residency.sync_in()

    def run_seam(self) -> None:
        self.plan.run()
        if not self.rotate:
            # THE IN-PLACE NULL CONTROL: put the engine's references back on the
            # pre-launch buffers, so the next launch and the host read the stale H.
            for name in family.ROTATED_NAMES:
                current = getattr(self.fields, name)
                setattr(self.fields, name, self.plan.rotated[name])
                self.plan.rotated[name] = current

    def step(self) -> Tuple[Tuple[int, int], Dict[str, int]]:
        """One complete step; returns the sync counters the SEAM consumed and the
        residency invariant measured immediately after the seam's ``sync_out``."""
        seam_done = False
        inside = (0, 0)
        stale: Dict[str, int] = {}
        for _name, function, replaced in PASSES:
            if replaced:
                if not seam_done:
                    self.residency.sync_in()
                    before = (self.residency.syncs_out, self.residency.syncs_in)
                    self.run_seam()
                    inside = (self.residency.syncs_out - before[0],
                              self.residency.syncs_in - before[1])
                    self.residency.sync_out()
                    stale = self.residency.verify()
                    seam_done = True
                continue
            function(self.fields, self.pml)
        return inside, stale


class SinglesEngine:
    """The two separately certified complex Metal products, on ONE residency."""

    def __init__(self, fields: Any, pml: Any) -> None:
        self.fields, self.pml = fields, pml
        self.residency = Residency()
        self.h_plan = complex_fields.plan_complex_constitutive(fields, pml, "H",
                                                               self.residency)
        self.d_plan = complex_fields.plan_complex_pml_curl(fields, pml, "step_D",
                                                           self.residency)
        if self.h_plan is None or self.d_plan is None:
            raise AssertionError("a certified single refused an admitted configuration")
        self.residency.sync_in()

    def step(self) -> Dict[str, int]:
        seam_done = False
        stale: Dict[str, int] = {}
        for _name, function, replaced in PASSES:
            if replaced:
                if not seam_done:
                    self.residency.sync_in()
                    self.h_plan.run()
                    self.d_plan.run()
                    self.residency.sync_out()
                    stale = self.residency.verify()
                    seam_done = True
                continue
            function(self.fields, self.pml)
        return stale


# ---------------------------------------------------------------------------
# LEG binding_ceiling
# ---------------------------------------------------------------------------

def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, (message.splitlines()[0] if message else "")


def _sweep_source(complex_volumes: int, real_vectors: int) -> Tuple[int, str]:
    """A kernel of the SHIPPED SHAPE at an arbitrary binding count.

    ``real_vectors`` is at least one on every call the leg makes: a zero-length touch
    expression emits ``float touch = ;``, which fails to compile for a reason that has
    nothing to do with the ceiling and reads exactly like the measurement. Encountered
    while writing this leg, and the guard below is why it cannot recur.
    """
    if real_vectors < 1:
        raise ValueError("a ceiling sweep needs at least one real vector to touch")
    lines = [f"    device float2* q{n} [[buffer({n})]]," for n in range(complex_volumes)]
    base = complex_volumes
    lines += [f"    device const float* r{base + n} [[buffer({base + n})]],"
              for n in range(real_vectors)]
    slot = complex_volumes + real_vectors
    touch = " + ".join(f"r{base + n}[0]" for n in range(real_vectors))
    return slot + 1, "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { float2 px; uint n_elem; float dtdx; };", "",
        "kernel void ceiling_sweep(", *lines,
        f"    constant Params& prm [[buffer({slot})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float touch = {touch};",
        "    float2 acc = float2(touch, prm.dtdx);",
        *(f"    acc = acc + q{n}[idx];" for n in range(complex_volumes)),
        "    q0[idx] = acc;",
        "}", ""))


def leg_binding_ceiling(payload: Dict[str, Any], out: str) -> None:
    probe_arm, expansion = bound_expansion()
    row: Dict[str, Any] = {
        "declared_max_buffer_bindings": MAX_BUFFER_BINDINGS,
        "packed_bindings_declared": family.PACKED_BINDINGS,
        "probe_bound_expansion": probe_arm,
        "shipped": {}, "shipped_shape_sweep": {}, "generic_sweep": {}, "refuted": {},
    }
    for arm in templates.EXPANSION_ARMS:
        text_bindings = family.shipped_signature_bindings(arm)
        ok, error = _compiles(
            family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), arm))
        row["shipped"][arm] = {"text_bindings": text_bindings, "compiled": ok,
                               "error": error}
        log(f"[binding_ceiling] shipped {arm} bindings={text_bindings} compiled={ok}")
    # THE BISECT, ON THE SHIPPED SHAPE: 21 float2 volumes and a growing real group,
    # packed Params. 28..31 must compile and 32..34 must not.
    for reals in range(len(family._REAL_VECTORS) - 3,  # noqa: SLF001
                       len(family._REAL_VECTORS) + 4):  # noqa: SLF001
        if reals < 1:
            continue
        bindings, source = _sweep_source(len(family._COMPLEX_VOLUMES), reals)  # noqa: SLF001
        ok, error = _compiles(source)
        row["shipped_shape_sweep"][str(bindings)] = {
            "compiled": ok, "error": error, "complex_volumes":
                len(family._COMPLEX_VOLUMES), "real_vectors": reals}  # noqa: SLF001
        log(f"[binding_ceiling] shipped-shape {bindings} bindings compiled={ok} "
            f"{error[:70]}")
    for bindings in range(MAX_BUFFER_BINDINGS - 3, MAX_BUFFER_BINDINGS + 2):
        measured, source = _sweep_source(bindings - 2, 1)
        assert measured == bindings, (measured, bindings)
        ok, _error = _compiles(source)
        row["generic_sweep"][str(bindings)] = ok
    for label, builder, declared in (
            ("one_more_pointer", family.refuted_one_more_pointer_source,
             family.ONE_MORE_POINTER_BINDINGS),
            ("unshared_kms", family.refuted_unshared_kms_source,
             family.UNSHARED_KMS_BINDINGS),
            ("separate_scalar", family.refuted_separate_scalar_source,
             family.SEPARATE_SCALAR_BINDINGS),
            ("split_plane", family.split_plane_pair_signature,
             family.SPLIT_PLANE_BINDINGS)):
        ok, error = _compiles(builder())
        row["refuted"][label] = {"declared_bindings": declared, "compiled": ok,
                                 "error": error}
        log(f"[binding_ceiling] refuted {label} ({declared} bindings) compiled={ok}")
    largest = max((int(k) for k, ok in row["generic_sweep"].items() if ok), default=-1)
    smallest = min((int(k) for k, ok in row["generic_sweep"].items() if not ok),
                   default=-1)
    row["ceiling_measured"] = largest
    row["ceiling_measured_equals_declared"] = (largest == MAX_BUFFER_BINDINGS
                                               and smallest == largest + 1)
    row["headroom_measured"] = MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS
    at_ceiling = row["shipped_shape_sweep"][str(MAX_BUFFER_BINDINGS)]
    over = row["shipped_shape_sweep"][str(MAX_BUFFER_BINDINGS + 1)]
    row["passed"] = bool(
        all(entry["compiled"] for entry in row["shipped"].values())
        and all(entry["text_bindings"] == family.PACKED_BINDINGS
                for entry in row["shipped"].values())
        and all(entry["compiled"] for key, entry in row["shipped_shape_sweep"].items()
                if int(key) <= MAX_BUFFER_BINDINGS)
        and at_ceiling["compiled"] and not over["compiled"]
        and "out of bounds" in over["error"] and "buffer" in over["error"]
        and not any(entry["compiled"] for entry in row["refuted"].values())
        and all("out of bounds" in entry["error"]
                for entry in row["refuted"].values())
        and row["ceiling_measured_equals_declared"]
        and row["headroom_measured"] == 0)
    payload["legs"]["binding_ceiling"] = row
    save(payload, out)
    log(f"[binding_ceiling] ceiling={row['ceiling_measured']} "
        f"headroom={row['headroom_measured']} passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG lift
# ---------------------------------------------------------------------------

def _arithmetic_lines(text: str) -> List[str]:
    """The lines that decide a bit in the constitutive half, comments stripped.

    THE COEFFICIENT LOADS ARE DELIBERATELY NOT IN HERE. ``km_0 = km0[i]`` becomes
    ``km_0 = kmx[i]`` under the declared lift edits -- a pointer rename onto the SAME
    volume, since both halves share one sub-lattice -- so including it would make this
    comparison fail for the one reason the edit table already accounts for. The loads
    are checked separately, by applying the three declared renames and requiring
    equality, so neither the arithmetic nor the rename goes unmeasured.
    """
    return [line.split("//", 1)[0].strip() for line in text.splitlines()
            if "c_mul_coefficient_left" in line.split("//", 1)[0]]


def _coefficient_lines(text: str) -> List[str]:
    return [line.split("//", 1)[0].strip() for line in text.splitlines()
            if line.split("//", 1)[0].strip().startswith("float kp_")]


def leg_lift(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    certified = complex_fields.bloch_constitutive_source("H", expansion)
    certified_tail = certified.split(family.DECODE_END, 1)[1][: -len("}\n")]
    tail = family.certified_complex_constitutive_tail(expansion)
    # RE-DERIVE THE LIFT FROM THE DECLARED EDIT TABLE and require equality: the table
    # in the module is then DATA the gate checks, not prose beside the code.
    rederived = certified_tail
    edits: List[Tuple[str, str]] = [
        ("    float kp_0 = kp0[i], km_0 = km0[i];\n",
         "    float kp_0 = kp0[i], km_0 = kmx[i];\n"),
        ("    float kp_1 = kp1[j], km_1 = km1[j];\n",
         "    float kp_1 = kp1[j], km_1 = kmy[j];\n"),
        ("    float kp_2 = kp2[k], km_2 = km2[k];\n",
         "    float kp_2 = kp2[k], km_2 = kmz[k];\n"),
    ]
    for target in range(3):
        edits.extend((
            (f"    float2 prev{target} = w{target}[ii];\n",
             f"    float2 prev{target} = wi{target}[ii];\n"),
            (f"    float2 src{target} = g{target}[ii];\n",
             f"    float2 src{target} = b{target}[ii];\n"),
            (f"    w{target}[ii] = src{target};\n", ""),
            (f"    float2 a{target} = f{target}[ii];\n",
             f"    float2 a{target} = hi{target}[ii];\n"),
            (f"    f{target}[ii] = a{target};\n", ""),
        ))
    for old, new in edits:
        assert rederived.count(old) == 1, old
        rederived = rederived.replace(old, new)

    # THE CURL HALF, MEASURED BY REVERSAL. The nine redirections are undone on the
    # emitted body and the result must be the certified emitter's own text, byte for
    # byte -- which is a stronger statement than "the redirections were applied": it
    # says NOTHING ELSE moved.
    reversed_rows: Dict[str, bool] = {}
    for codes, phased in ((( 0, 0, 0), (1, 1, 1)), ((1, 1, 0), (0, 0, 1)),
                          ((1, 1, 1), (0, 0, 0)), ((0, 1, 0), (1, 0, 1))):
        prologue, body = family.welded_curl_tail(codes, phased, expansion)
        restored = body
        for var, target in family.OWN_LOAD_EDITS:
            restored = restored.replace(f"    float2 {var}   = own.a{target};\n",
                                        f"    float2 {var}   = g{target}[ii];\n")
        for var, target in family.HALO_TAPS:
            offsets = family.offset_coordinates(body)
            for offset, coordinates in offsets.items():
                call = (f"h_cell({', '.join(coordinates)}, "
                        f"{family.H_CELL_TAIL_ARGS}).a{target}")
                restored = restored.replace(call, f"g{target}[{offset}]")
        theirs = complex_fields.bloch_curl_source(codes, True, phased, expansion)
        theirs_body = theirs.split(
            "uint idx [[thread_position_in_grid]])\n{\n", 1)[1][: -len("}\n")]
        reversed_rows[f"{codes}/{phased}"] = (prologue + restored) == theirs_body

    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), expansion)
    row = {
        "expansion": expansion,
        "lift_equals_declared_edits": rederived == tail,
        "declared_edit_rows": len(family.CONSTITUTIVE_LIFT_EDITS),
        "arithmetic_lines_identical": (_arithmetic_lines(certified_tail)
                                       == _arithmetic_lines(tail)),
        "arithmetic_line_count": len(_arithmetic_lines(tail)),
        "coefficient_loads_differ_only_by_the_declared_rename": (
            [line.replace("km0[", "kmx[").replace("km1[", "kmy[")
                 .replace("km2[", "kmz[")
             for line in _coefficient_lines(certified_tail)]
            == _coefficient_lines(tail)),
        "coefficient_load_count": len(_coefficient_lines(tail)),
        "curl_reverses_to_the_certified_text": reversed_rows,
        "carries_certified_helpers": templates.complex_helpers(expansion) in source,
        "contraction_guard_once": source.count(
            shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1,
        "h_cell_returns": all(f"out.{register} = {register};" in source
                              for register in ("a0", "a1", "a2", "src0", "src1",
                                               "src2")),
        "own_cell_stored_to_scratch": (
            "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n" in source),
        "no_magnetic_pointer_survives": all(f"g{target}[" not in source
                                            for target in range(3)),
        "phase_block_untouched": all(
            f"c_mul({operand}, p{axis})" in source
            for axis, operands in (("x", ("b_x", "c_x")), ("y", ("a_y", "c_y")),
                                   ("z", ("a_z", "b_z")))
            for operand in operands),
    }
    row["passed"] = bool(
        row["lift_equals_declared_edits"] and row["arithmetic_lines_identical"]
        and row["arithmetic_line_count"] == 6
        and row["coefficient_loads_differ_only_by_the_declared_rename"]
        and row["coefficient_load_count"] == 3
        and all(row["curl_reverses_to_the_certified_text"].values())
        and row["carries_certified_helpers"] and row["contraction_guard_once"]
        and row["h_cell_returns"] and row["own_cell_stored_to_scratch"]
        and row["no_magnetic_pointer_survives"] and row["phase_block_untouched"])
    payload["legs"]["lift"] = row
    save(payload, out)
    log(f"[lift] passed={row['passed']} arithmetic_lines={row['arithmetic_line_count']} "
        f"reversal={sum(row['curl_reverses_to_the_certified_text'].values())}/"
        f"{len(row['curl_reverses_to_the_certified_text'])}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG driver_order
# ---------------------------------------------------------------------------

def leg_driver_order(payload: Dict[str, Any], out: str) -> None:
    fact = h_to_d_seam.driver_seam_fact()
    row = {
        "driver_seam_fact": fact,
        "replaces_is_the_span": tuple(family.REPLACES) == tuple(withdraw_hoist.SEAM_SPAN),
        "seam_is_the_hoist_seam": family.SEAM == withdraw_hoist.SEAM,
        "fused_pair_seams_routes_to_hoist": (
            metal_launch.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)),
        "between_is_the_withdraw_loop_only": all(
            "withdraw" in entry["statement"]
            or "for source in electric" in entry["statement"]
            or "update_H(self.fields, self.pml)" in entry["statement"]
            for entry in fact["between"]),
        "launches_per_run": ProductEngine(*build(MUTATION_CASE, 1)).plan.launches_per_run,
        "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
        "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
        "installable": family.INSTALLABLE,
        "weld_owed_nonempty": bool(family.WELD_OWED),
        # WIRED 2026-09-07, so this pin INVERTED with the wiring: the row is now
        # registered on ``update_H`` and it is registered UNWIRED, which is the
        # property that matters. A registered row is what lets the seam loop ASK the
        # product (through its ``FUSED_PAIR_ARMS`` absorb row) and then refuse it on
        # INSTALLABLE; ``wired=False`` is what keeps ``plan_step`` from selecting it
        # regardless of the flag. Pinning ABSENCE, as this leg did pre-wiring, would
        # go red on the wiring and say nothing about the seam.
        "registered_unwired_on_update_H": [
            (arm.family, bool(arm.wired)) for arm in _registered_arms()
            if arm.family == family.FAMILY] == [(family.FAMILY, False)],
        "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
    }
    row["passed"] = bool(row["replaces_is_the_span"] and row["seam_is_the_hoist_seam"]
                         and row["fused_pair_seams_routes_to_hoist"]
                         and row["between_is_the_withdraw_loop_only"]
                         and row["launches_per_run"] == 1
                         and not row["carries_deposit_repair"]
                         and not row["hoists_the_withdraw"]
                         and not row["installable"]
                         and row["registered_unwired_on_update_H"]
                         and row["absorb_row"] == ("complex/Bloch", "complex/Bloch"))
    payload["legs"]["driver_order"] = row
    save(payload, out)
    log(f"[driver_order] consults {fact['update_H_consult']} -> {fact['step_D_consult']}"
        f" passed={row['passed']}")
    assert row["passed"], row


def _registered_arms() -> Tuple[Any, ...]:
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415

    return tuple(metal_arms.registered("update_H"))


# ---------------------------------------------------------------------------
# LEG refusal
# ---------------------------------------------------------------------------

def leg_refusal(payload: Dict[str, Any], out: str) -> None:
    fields, pml = build(MUTATION_CASE, 1)
    residency = Residency()
    cover = family.metal_complex_fused_hd_pair_coverage

    def named(verdict: Any, *needles: str) -> bool:
        return all(any(needle in reason for reason in verdict.reasons)
                   for needle in needles)

    admitted = cover(fields, pml, (), residency)
    integrated = cover(fields, pml, (volume_source(fields, "Ez", True),), residency)
    plain = cover(fields, pml, (volume_source(fields, "Ez", False),), residency)
    magnetic = cover(fields, pml, (volume_source(fields, "Hz", True),), residency)
    undeclared = cover(fields, pml, None, residency)
    no_residency = cover(fields, pml, (), None)
    real_fields, real_pml = matrix.cart(complex_storage=False)
    real = cover(real_fields, real_pml, (), Residency())
    cyl_fields, cyl_pml = matrix.cylindrical(m=1, complex_storage=True)
    cylindrical = cover(cyl_fields, cyl_pml, (), Residency())
    folded_fields, folded_pml = matrix.folded(complex_storage=True)
    folded = cover(folded_fields, folded_pml, (), Residency())
    beta_fields, beta_pml = matrix.flat(complex_storage=True, beta=0.4)
    beta = cover(beta_fields, beta_pml, (), Residency())
    no_pml_fields, no_pml = matrix.cart(complex_storage=True, pml=0)
    inactive = cover(no_pml_fields, no_pml, (), Residency())
    row = {
        "admitted_with_empty_sources": admitted.covered,
        "admitted_reasons": list(admitted.reasons),
        "integrated_electric_refused_by_name": (
            not integrated.covered
            and named(integrated, "standing", "HOISTS_THE_WITHDRAW = False")),
        "integrated_electric_reasons": list(integrated.reasons),
        "non_integrated_electric_admitted": plain.covered,
        "magnetic_integrated_admitted": magnetic.covered,
        "undeclared_sources_refused": (not undeclared.covered
                                       and named(undeclared, "not declared")),
        "no_residency_refused": not no_residency.covered,
        "real_storage_refused": not real.covered,
        "real_storage_reasons_head": list(real.reasons)[:2],
        "cylindrical_refused": not cylindrical.covered,
        "cylindrical_reasons_head": list(cylindrical.reasons)[:2],
        "folded_refused": not folded.covered,
        "folded_reasons_head": list(folded.reasons)[:2],
        "folded_named_by_this_module": named(folded, "is folded"),
        "beta_refused": not beta.covered,
        "beta_reasons_head": list(beta.reasons)[:2],
        "inactive_absorber_refused": not inactive.covered,
        "inactive_absorber_reasons_head": list(inactive.reasons)[:2],
    }
    row["passed"] = bool(
        row["admitted_with_empty_sources"]
        and row["integrated_electric_refused_by_name"]
        and row["non_integrated_electric_admitted"]
        and row["magnetic_integrated_admitted"]
        and row["undeclared_sources_refused"] and row["no_residency_refused"]
        and row["real_storage_refused"] and row["cylindrical_refused"]
        and row["folded_refused"] and row["folded_named_by_this_module"]
        and row["beta_refused"] and row["inactive_absorber_refused"])
    payload["legs"]["refusal"] = row
    save(payload, out)
    log(f"[refusal] passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG product / value_class
# ---------------------------------------------------------------------------

def run_case(label: str, seed: int, steps: int,
             value_class: str = values.UNIFORM) -> Dict[str, Any]:
    reference = ArrayEngine(*build(label, seed, value_class))
    product = ProductEngine(*build(label, seed, value_class))
    singles = SinglesEngine(*build(label, seed, value_class))
    plan = product.plan
    per_step: List[Dict[str, Any]] = []
    total_compared = 0
    first_divergence = None
    sign_floor = None
    for step in range(1, steps + 1):
        reference.step()
        product_syncs, stale = product.step()
        singles_stale = singles.step()
        ref_state = state_of(reference.fields)
        prod_state = state_of(product.fields)
        single_state = state_of(singles.fields)
        versus_array = compare(prod_state, ref_state)
        versus_singles = compare(prod_state, single_state)
        singles_versus_array = compare(single_state, ref_state)
        compared = sum(int(words(array).size) for array in ref_state.values())
        total_compared += compared
        band = state_census(reference.fields)
        assert not singles_stale, (label, step, singles_stale)
        moved = sum(differing(prod_state[name], np.zeros_like(prod_state[name]))
                    for name in prod_state)
        if value_class == values.PM_ZERO_LATTICE and step == 1:
            sign_floor = values.zero_sign_census(list(ref_state.values()))
        per_step.append({
            "step": step, "compared": compared, "versus_array": versus_array,
            "versus_singles": versus_singles,
            "singles_versus_array": singles_versus_array,
            "launches": plan.launches, "product_seam_syncs": list(product_syncs),
            "stale_mirrors": stale, "reference_subnormal_words": band,
            "moved_words": moved})
        if (versus_array or versus_singles) and first_divergence is None:
            first_divergence = step
        assert band == 0, (label, step, "the reference entered the subnormal band")
        assert not stale, (label, step, stale)
        assert plan.launches == step, (label, step, plan.launches)
        log(f"[case] {label:<18} {value_class:<16} step {step:>2}/{steps} "
            f"vs_array={versus_array or 0} vs_singles={versus_singles or 0} "
            f"launches={plan.launches} seam_syncs={product_syncs}")
    identical = all(not entry["versus_array"] and not entry["versus_singles"]
                    for entry in per_step)
    if value_class == values.PM_ZERO_LATTICE:
        assert sign_floor and sign_floor["positive_zero_words"] > 0 \
            and sign_floor["negative_zero_words"] > 0, sign_floor
    else:
        kit.assert_moved(per_step[-1]["moved_words"], f"{label} moved nothing",
                         floor=64)
    return {"case": label, "value_class": value_class, "steps": steps,
            "compared_words": total_compared, "bit_identical": identical,
            "first_divergence": first_divergence, "per_step": per_step,
            "shape": list(reference.fields.grid.shape),
            "codes": list(plan.codes), "phased": list(plan.phased),
            "phase_values": [list(pair) for pair in plan.phase_values],
            "expansion": plan.expansion, "zero_sign_census": sign_floor,
            "digest": kit.state_digest(state_of(reference.fields))}


def leg_product(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for offset, (label, _cell, _boundaries, _k) in enumerate(CONFIGS):
        rows.append(run_case(label, 91000 + offset, STEPS))
        payload["legs"]["product"] = rows
        save(payload, out)
        log(f"[product] {label:<18} bit_identical={rows[-1]['bit_identical']} "
            f"compared={rows[-1]['compared_words']}")
    assert all(row["bit_identical"] for row in rows), [
        (row["case"], row["first_divergence"]) for row in rows]


def leg_value_class(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for offset, entry in enumerate(CONFIGS[:4]):
        rows.append(run_case(entry[0], 93000 + offset, 4, values.PM_ZERO_LATTICE))
        payload["legs"]["value_class"] = rows
        save(payload, out)
        log(f"[value_class] {entry[0]:<18} pm_zero_lattice bit_identical="
            f"{rows[-1]['bit_identical']} census={rows[-1]['zero_sign_census']}")
    assert all(row["bit_identical"] for row in rows), rows


# ---------------------------------------------------------------------------
# LEG null_control
# ---------------------------------------------------------------------------

def leg_null_control(payload: Dict[str, Any], out: str) -> None:
    """The rotation undone: an IN-PLACE product, which must diverge.

    Without this, every "0 differing words" above is compatible with a comparison that
    could not tell the two states apart at all.
    """
    rows: List[Dict[str, Any]] = []
    for offset, entry in enumerate(CONFIGS[:3]):
        label = entry[0]
        reference = ArrayEngine(*build(label, 95000 + offset))
        engine = ProductEngine(*build(label, 95000 + offset), rotate=False)
        diverged = 0
        first: Dict[str, int] = {}
        for step in range(3):
            reference.step()
            engine.step()
            delta = compare(state_of(engine.fields), state_of(reference.fields))
            if delta:
                diverged += 1
                first = first or delta
        rows.append({"case": label, "steps": 3, "steps_diverged": diverged,
                     "first_divergence_volumes": first,
                     "launches": engine.plan.launches})
        log(f"[null_control] {label:<18} diverged {diverged}/3 volumes="
            f"{sorted(first)[:6]}")
        payload["legs"]["null_control"] = rows
        save(payload, out)
    row = {"cases": rows, "passed": all(entry["steps_diverged"] == entry["steps"]
                                        and entry["launches"] == entry["steps"]
                                        for entry in rows)}
    payload["legs"]["null_control"] = row
    save(payload, out)
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG purity
# ---------------------------------------------------------------------------

def leg_purity(payload: Dict[str, Any], out: str) -> None:
    """A static read/write ledger over the fused source, comments stripped."""
    _probe, expansion = bound_expansion()
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), expansion)
    kernel = source.split(f"kernel void {family.KERNEL}(", 1)[1].split("{", 1)[1]
    code = "\n".join(line.split("//", 1)[0] for line in kernel.splitlines())
    written = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2")
    in_place = ("f0", "f1", "f2", "u0", "u1", "u2")
    read = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2",
            "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz", "kp0", "kp1", "kp2")
    ledger: Dict[str, Dict[str, int]] = {}
    for name in written + in_place + read:
        stores = len(re.findall(rf"\b{name}\[[^\]]*\]\s*=(?!=)", code))
        loads = len(re.findall(rf"\b{name}\[", code)) - stores
        ledger[name] = {"stores": stores, "loads": loads}
    helper = source.split("static inline h_cell_result h_cell(", 1)[1].split(
        "\nkernel void", 1)[0]
    helper_code = "\n".join(line.split("//", 1)[0] for line in helper.splitlines())
    helper_stores = len(re.findall(r"\b\w+\[[^\]]*\]\s*=(?!=)", helper_code))
    row = {
        "ledger": ledger,
        "scratch_never_loaded": all(ledger[name]["loads"] == 0 for name in written),
        "scratch_always_stored": all(ledger[name]["stores"] == 1 for name in written),
        "read_group_never_stored": all(ledger[name]["stores"] == 0 for name in read),
        # D and fu_D update IN PLACE and that is the design: the curl reads and writes
        # them at the thread's OWN cell only, which is what the index check below says.
        "in_place_only_at_own_cell": all(
            not re.findall(rf"\b{name}\[(?!ii\])", code) for name in in_place),
        "h_cell_stores_nothing": helper_stores == 0,
        "h_cell_call_sites": code.count("h_cell("),
    }
    row["passed"] = bool(row["scratch_never_loaded"] and row["scratch_always_stored"]
                         and row["read_group_never_stored"]
                         and row["in_place_only_at_own_cell"]
                         and row["h_cell_stores_nothing"]
                         and row["h_cell_call_sites"] == 7)
    payload["legs"]["purity"] = row
    save(payload, out)
    log(f"[purity] passed={row['passed']} h_cell call sites={row['h_cell_call_sites']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG divide_spelling
# ---------------------------------------------------------------------------

def _signed_zero_rich(rng: Any, count: int) -> np.ndarray:
    real = rng.standard_normal(count).astype(np.float32)
    imag = rng.standard_normal(count).astype(np.float32)
    real[::5] = np.float32(0.0)
    real[1::7] = np.float32(-0.0)
    imag[2::5] = np.float32(-0.0)
    imag[3::11] = np.float32(0.0)
    return complex_from_planes(real, imag)


def leg_divide_spelling(payload: Dict[str, Any], out: str) -> None:
    """The complex-by-real divide fact, re-measured HERE rather than cited.

    TWO STATEMENTS, and the first is the one that matters for this kernel: the emitted
    text contains NO float division at all, because every PML reciprocal is precomputed
    on the host into ``sinv_a`` and the kernel multiplies by it. The second measures
    numpy's own ``complex64 / float32`` beside the reciprocal multiply on a
    signed-zero-rich table, so the neighbouring family's finding is carried forward as
    a measurement on this host and this numpy rather than as a reference.
    """
    _probe, expansion = bound_expansion()
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), expansion)
    code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
    # Integer index arithmetic uses `/` (`ii / nzi`); a FLOAT divide is what this leg
    # is about, and it is counted by the operand shapes the emitters can produce.
    float_divides = re.findall(r"[\w\.\)]\s*/\s*[\w\(]", code)
    integer_divides = re.findall(r"\b(?:ii|plane)\s*/\s*nzi|\bplane\s*/\s*nyi", code)
    rng = np.random.default_rng(20260907)
    measured: Dict[str, Dict[str, int]] = {}
    for class_name in ("random", "signed_zero_rich"):
        if class_name == "random":
            z = complex_from_planes(rng.standard_normal(8192).astype(np.float32),
                                    rng.standard_normal(8192).astype(np.float32))
        else:
            z = _signed_zero_rich(rng, 8192)
        d = (rng.uniform(0.25, 4.0, 8192)).astype(np.float32)
        reciprocal = (z * (np.float32(1.0) / d)).astype(np.complex64)
        componentwise = complex_from_planes(z.real / d, z.imag / d)
        oracle = (z / d).astype(np.complex64)
        measured[class_name] = {
            "words": int(words(oracle).size),
            "reciprocal_multiply_differing": differing(reciprocal, oracle),
            "componentwise_differing": differing(componentwise, oracle),
        }
    row = {
        "float_divides_in_the_emitted_kernel": len(float_divides) - len(integer_divides),
        "integer_index_divides": len(integer_divides),
        "finding": ("this seam performs no complex-by-real division: every PML "
                    "reciprocal is precomputed on the host into sinv_a and the kernel "
                    "multiplies by it through c_mul_field_left, so the spelling "
                    "question the Dcyl scan had to settle does not arise inside these "
                    "bytes. The mutation `componentwise_divide_by_reciprocal` plants "
                    "it anyway, so the fact is armed and not merely stated"),
        "numpy_complex64_over_float32": measured,
        "host_finding": (
            "on RANDOM data numpy's complex64/float32 IS the reciprocal multiply "
            "z * (1/d) bit for bit and is NOT the componentwise divide; on a "
            "signed-zero-rich table both spellings differ from it, and the "
            "differences are the SIGN of exactly-zero words (the neighbouring Dcyl "
            "scan's refinement, re-measured here on this host and this numpy)"),
    }
    row["passed"] = bool(
        row["float_divides_in_the_emitted_kernel"] == 0
        and row["integer_index_divides"] >= 2
        # The host measurement must DISCRIMINATE, or it is not evidence for anything.
        and measured["random"]["reciprocal_multiply_differing"] == 0
        and measured["random"]["componentwise_differing"] > 0
        and measured["signed_zero_rich"]["componentwise_differing"] > 0)
    payload["legs"]["divide_spelling"] = row
    save(payload, out)
    log(f"[divide_spelling] float divides in kernel="
        f"{row['float_divides_in_the_emitted_kernel']} numpy random "
        f"reciprocal={measured['random']['reciprocal_multiply_differing']} "
        f"componentwise={measured['random']['componentwise_differing']} "
        f"signed-zero componentwise="
        f"{measured['signed_zero_rich']['componentwise_differing']} "
        f"passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG helper_spellings
# ---------------------------------------------------------------------------

_SPELLING_PROBE = r"""
#include <metal_stdlib>
using namespace metal;

__HELPERS__

kernel void spelling_probe(
    device float2*       out    [[buffer(0)]],
    device const float2* z      [[buffer(1)]],
    device const float2* p      [[buffer(2)]],
    device const float*  c      [[buffer(3)]],
    constant uint&       n_elem [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    __OP__
}
"""

#: The four operand values an exhaustive sign/zero table is built from. ``+-0`` is what
#: the three refuted spellings differ on; ``+-1`` gives a nonzero product to place
#: beside it, so a pattern that is all zeros is not the only one in the table.
_SIGN_VALUES: Tuple[float, ...] = (0.0, -0.0, 1.0, -1.0)


def _exhaustive_table(components: int) -> np.ndarray:
    """Every assignment of :data:`_SIGN_VALUES` to ``components`` float32 slots."""
    total = len(_SIGN_VALUES) ** components
    out = np.empty((total, components), dtype=np.float32)
    for index in range(total):
        remainder = index
        for slot in range(components):
            out[index, slot] = np.float32(_SIGN_VALUES[remainder % len(_SIGN_VALUES)])
            remainder //= len(_SIGN_VALUES)
    return out


def _run_spelling(helpers: str, operation: str, z: Any, p: Any, c: Any) -> np.ndarray:
    import torch  # noqa: PLC0415

    source = templates.substitute(_SPELLING_PROBE, {"__HELPERS__": helpers,
                                                    "__OP__": operation})
    function = compile_source(source).spelling_probe
    count = int(z.size)
    out = torch.from_numpy(np.full(count, np.complex64(-3.5 - 3.5j))).to("mps")
    function(out, torch.from_numpy(z).to("mps"), torch.from_numpy(p).to("mps"),
             torch.from_numpy(c).to("mps"), count)
    torch.mps.synchronize()
    return out.cpu().numpy()


def leg_helper_spellings(payload: Dict[str, Any], out: str) -> None:
    """The three load-bearing complex spellings, RE-MEASURED at the helper.

    WHY THIS LEG EXISTS BESIDE THE MUTATION LEG, and it is a finding rather than
    bookkeeping. A whole-step walk and an operand table are DIFFERENT INSTRUMENTS. The
    zero cross terms and the negation spelling differ from the shipped ones only in the
    SIGN OF AN EXACTLY-ZERO word, and a complete driver step then subtracts, scales and
    accumulates that word: the curl differences two operands and the split-field
    recurrence adds into a register, and IEEE addition does not preserve a lone
    negative zero through either. So a step-level mutation can measure INERT on a
    spelling that is genuinely wrong, and reporting that as "equivalent" would be a
    false equivalence rather than a false pass. This leg feeds the helper directly, at
    the exhaustive table the certified complex family uses, so both numbers exist and
    the mutation leg's entries can name which instrument saw them.
    """
    _probe, expansion = bound_expansion()
    shipped = templates.complex_helpers(expansion)
    full = _exhaustive_table(4)
    z_full = complex_from_planes(full[:, 0], full[:, 1])
    p_full = complex_from_planes(full[:, 2], full[:, 3])
    c_full = np.zeros(z_full.size, dtype=np.float32)
    real = _exhaustive_table(3)
    z_real = complex_from_planes(real[:, 0], real[:, 1])
    p_real = np.zeros_like(z_real)
    c_real = np.ascontiguousarray(real[:, 2])
    cases = (
        ("c_mul/negation_zero_minus_x", "out[idx] = c_mul(z[idx], p[idx]);",
         lambda text: kit.needle(text, "fma(z.x, p.x, -(z.y * p.y))",
                                 "fma(z.x, p.x, (0.0f - (z.y * p.y)))"),
         z_full, p_full, c_full),
        ("c_mul_field_left/plane_wise",
         "out[idx] = c_mul_field_left(z[idx], c[idx]);",
         lambda text: kit.needle(
             text, "static inline float2 c_mul_field_left(float2 z, float c) {",
             "static inline float2 c_mul_field_left(float2 z, float c) {\n"
             "    return float2(z.x * c, z.y * c);"),
         z_real, p_real, c_real),
        ("c_mul_coefficient_left/folded_zero_cross_term",
         "out[idx] = c_mul_coefficient_left(c[idx], z[idx]);",
         lambda text: kit.needle(text, "return float2(fma(c, z.x, -(0.0f * z.y)),",
                                 "return float2(fma(c, z.x, -0.0f),"),
         z_real, p_real, c_real),
        ("c_mul_coefficient_left/field_left_orientation",
         "out[idx] = c_mul_coefficient_left(c[idx], z[idx]);",
         lambda text: kit.needle(
             text, "static inline float2 c_mul_coefficient_left(float c, float2 z) {",
             "static inline float2 c_mul_coefficient_left(float c, float2 z) {\n"
             "    return c_mul_field_left(z, c);"),
         z_real, p_real, c_real),
    )
    rows: List[Dict[str, Any]] = []
    for label, operation, transform, z, p, c in cases:
        reference = _run_spelling(shipped, operation, z, p, c)
        mutant = _run_spelling(transform(shipped), operation, z, p, c)
        entry = {"spelling": label, "patterns": int(z.size),
                 "words": int(words(reference).size),
                 "differing": differing(mutant, reference)}
        rows.append(entry)
        log(f"[helper_spellings] {label:<48} {entry['differing']}/{entry['words']} "
            f"words differ")
        payload["legs"]["helper_spellings"] = rows
        save(payload, out)
    by_label = {entry["spelling"]: entry for entry in rows}
    block = {
        "rows": rows, "expansion": expansion,
        "finding": (
            "the three refuted spellings ARE discriminating at the helper, on the "
            "exhaustive sign/zero table, on this host and this toolchain. Whether a "
            "given one also moves a stored word through a COMPLETE DRIVER STEP is the "
            "mutation leg's separate measurement, and the two can disagree: a "
            "sign-of-zero difference in an intermediate does not survive the curl's "
            "difference or the split-field accumulation"),
    }
    block["passed"] = bool(
        rows
        and by_label["c_mul/negation_zero_minus_x"]["differing"] > 0
        and by_label["c_mul_field_left/plane_wise"]["differing"] > 0
        and by_label["c_mul_coefficient_left/folded_zero_cross_term"]["differing"] > 0
        # The orientation swap is the family's measured EQUIVALENCE for a real
        # coefficient, and it must stay one here or the null it licenses is wrong.
        and by_label["c_mul_coefficient_left/field_left_orientation"]["differing"] == 0)
    payload["legs"]["helper_spellings"] = block
    save(payload, out)
    log(f"[helper_spellings] passed={block['passed']}")
    assert block["passed"], block


# ---------------------------------------------------------------------------
# LEG corpus_rows
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cell_rows() -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in this product's cell, and the seam facts.

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer chose
    for each row -- never from a list in this file. The seam record supplies each row's
    ``withdraw_in_seam`` flag, which names the rows the predicate must refuse.
    """
    census = os.path.join(HERE, "results", CENSUS)
    seam = os.path.join(HERE, "results", SEAM_RECORD)
    if not os.path.isdir(census) or not os.path.isdir(seam):
        raise SystemExit(f"the corpus_rows leg needs {census} and {seam}")
    seam_rows = {row["label"]: row
                 for row in _load_jsonl(os.path.join(seam, "h_to_d_seam.jsonl"))}
    rows: Dict[str, dict] = {}
    for leg in ("examples", "tests", "tests_param_matched"):
        path = os.path.join(census, f"{leg}.jsonl")
        if not os.path.isfile(path):
            continue
        for record in _load_jsonl(path):
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            if (selected.get("update_H"), selected.get("step_D")) != CELL_ARMS:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows[label] = {
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                "configuration": record.get("configuration") or {},
                "facts": record.get("facts") or {},
                "grid_shape": record.get("grid_shape"),
                "grid_cells": record.get("grid_cells"),
                "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                         .get("withdraw_in_seam")),
                "census_selected": selected,
            }
    facts = {
        "census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS),
        "rows_in_cell": len(rows),
        "rows_with_a_standing_withdraw": sorted(
            label for label, row in rows.items() if row["withdraw_in_seam"]),
        "what_is_lifted": (
            "the census's recorded grid CONFIGURATION -- shape, boundary kinds, "
            "k_point, resolution and Courant number -- rebuilt as a Grid and checked "
            "against the recorded shape. NOT the row's own script: its materials, "
            "sources, absorber thickness and geometry are the synthetic fixture's. "
            "What this leg therefore measures is the product's identity on every "
            "SPECIALISATION the cell reaches (boundary triple x phase triple x "
            "extent), which is what the kernel is a function of; it does not re-run "
            "the corpus"),
    }
    return list(rows.values()), facts


def census_grid(row: dict) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild one census row's Grid BY THE ENGINE'S OWN RULE, and report the inputs.

    NOT by dividing the recorded shape by the resolution. That route silently maps a
    zero-extent axis of a ``dimensions = 3`` run onto an INVARIANT axis, and the engine
    then refuses the row's own Bloch vector -- ``k_point component x ... on an axis that
    dimensions=1 makes translationally invariant`` -- which reads exactly like a
    refusal of the row and is a refusal of the reconstruction. Measured on
    ``tests:TestBoundaries1D`` and ``tests:TestReflectanceAngular...1_20_6``, the
    corpus's two 1-D oblique-incidence rows.

    The rule reproduced here is ``from_meep._effective_dimensions``'s declared-wins
    branch plus ``from_meep._reduced_cell_size``: a declared ``dimensions`` of 1 or 2
    wins outright; otherwise the run is 3-D unless ``cell_size.z`` is zero; an
    INVARIANT axis is forced to exactly 0; and a RESOLVED axis given zero extent
    becomes ONE CELL, ``1/resolution`` long, which is what ``vol3d`` does.
    ``_invariant_axes`` is imported rather than retyped.
    """
    from meep_gpu.from_meep import _invariant_axes  # noqa: PLC0415

    facts = row["facts"]
    configuration = row["configuration"]
    resolution = float(facts.get("resolution") or 10.0)
    courant = float(facts.get("courant") or 0.5)
    shape = tuple(int(n) for n in (row.get("grid_shape")
                                   or configuration.get("shape") or ()))
    declared = facts.get("dimensions_attr")
    cell_size = facts.get("cell_size")
    route = "meep_cell_size"
    if not cell_size:
        # NO RECORDED CELL: fall back to the shape, and SAY SO on the row, because the
        # two routes are not the same reconstruction.
        route = "shape_over_resolution"
        cell_size = [0.0 if shape[axis] == 1 else shape[axis] / resolution
                     for axis in range(3)]
    cell_size = tuple(float(value) for value in cell_size)
    dimensions = (int(declared) if declared in (1, 2)
                  else (2 if cell_size[2] == 0.0 else 3))
    invariant = _invariant_axes(dimensions)
    cell = tuple(0.0 if axis in invariant
                 else (1.0 / resolution if cell_size[axis] == 0.0
                       else cell_size[axis])
                 for axis in range(3))
    kinds = tuple(configuration.get("boundary_kinds")
                  or ("periodic", "periodic", "periodic"))
    k_point = tuple(float(k) for k in (configuration.get("k_point")
                                       or (0.0, 0.0, 0.0)))
    inputs = {"census_shape": list(shape), "boundary_kinds": list(kinds),
              "k_point": list(k_point), "resolution": resolution,
              "courant": courant, "declared_dimensions": declared,
              "resolved_dimensions": dimensions, "cell_size": list(cell),
              "reconstruction_route": route}
    return build_grid(cell, kinds, k_point, courant=courant, resolution=resolution,
                      dimensions=dimensions), inputs


def drive_corpus_row(row: dict, steps: int, seed: int) -> Dict[str, Any]:
    shape = tuple(int(n) for n in (row.get("grid_shape")
                                   or row["configuration"].get("shape") or ()))
    record: Dict[str, Any] = {"label": row["label"],
                              "withdraw_in_seam": row["withdraw_in_seam"],
                              "steps_requested": steps}
    try:
        grid, inputs = census_grid(row)
    except Exception as exc:  # noqa: BLE001 - a refusal, recorded with its reason
        record.update({"rebuilt": False, "driven": False,
                       "census_shape": list(shape),
                       "why_not": f"{type(exc).__name__}: {exc}"[:300]})
        return record
    record.update(inputs)
    record["rebuilt_shape"] = list(grid.shape)
    record["rebuilt"] = tuple(grid.shape) == shape
    if not record["rebuilt"]:
        record.update({"driven": False,
                       "why_not": (f"the census recorded shape {shape} and this "
                                   f"configuration rebuilds at {tuple(grid.shape)}; "
                                   f"refused rather than driven at the wrong extent")})
        return record
    fields, pml = build_from_grid(grid, seed)
    verdict = family.metal_complex_fused_hd_pair_coverage(fields, pml, (), Residency())
    record["predicate_admits"] = bool(verdict.covered)
    record["predicate_reasons"] = list(verdict.reasons)
    if not verdict.covered:
        record.update({"driven": False,
                       "why_not": "the predicate refused this configuration"})
        return record
    started = time.time()
    reference = ArrayEngine(*build_from_grid(grid, seed))
    product = ProductEngine(*build_from_grid(grid, seed))
    singles = SinglesEngine(*build_from_grid(grid, seed))
    per_step: List[Dict[str, Any]] = []
    compared_total = 0
    for step in range(1, steps + 1):
        reference.step()
        _syncs, stale = product.step()
        singles.step()
        reference_state = state_of(reference.fields)
        product_state = state_of(product.fields)
        versus_array = compare(product_state, reference_state)
        versus_singles = compare(product_state, state_of(singles.fields))
        compared = sum(int(words(array).size) for array in reference_state.values())
        compared_total += compared
        band = state_census(reference.fields)
        moved = sum(differing(product_state[name],
                              np.zeros_like(product_state[name]))
                    for name in product_state)
        per_step.append({"step": step, "compared": compared,
                         "versus_array": versus_array,
                         "versus_singles": versus_singles,
                         "stale_mirrors": stale, "launches": product.plan.launches,
                         "reference_subnormal_words": band, "moved_words": moved})
        assert not stale, (row["label"], step, stale)
        assert product.plan.launches == step, (row["label"], step)
    # THE SUBNORMAL BAND IS A REFUSAL ON THIS EXECUTOR, PER STEP, NOT A DIVERGENCE.
    # MPS flushes subnormal intermediates; a row whose REFERENCE enters that band is
    # outside what a byte comparison can decide here, and is recorded as such inside
    # the denominator rather than scored either way.
    clean = [entry for entry in per_step if entry["reference_subnormal_words"] == 0]
    record.update({
        "driven": True,
        "codes": list(product.plan.codes), "phased": list(product.plan.phased),
        "expansion": product.plan.expansion,
        "steps_run": len(per_step), "steps_in_band": len(per_step) - len(clean),
        "compared_words": sum(entry["compared"] for entry in clean),
        "bit_identical": all(not entry["versus_array"] and not entry["versus_singles"]
                             for entry in clean),
        "moved_words": per_step[-1]["moved_words"],
        "per_step": per_step, "seconds": round(time.time() - started, 1),
    })
    return record


def leg_corpus_rows(payload: Dict[str, Any], out: str) -> None:
    rows, facts = cell_rows()
    steps = int(os.environ.get("MEEP_GPU_CX_HD_GATE_STEPS", 4))
    measured: List[Dict[str, Any]] = []
    for index, row in enumerate(sorted(rows, key=lambda entry: entry["label"]), 1):
        record = drive_corpus_row(row, steps, 97000 + index)
        measured.append(record)
        payload["legs"]["corpus_rows"] = {"facts": facts, "rows": measured}
        save(payload, out)
        log(f"[corpus_rows] {index}/{len(rows)} {record['label']:<58} "
            f"shape={record['census_shape']} admits="
            f"{record.get('predicate_admits')} driven={record.get('driven')} "
            f"identical={record.get('bit_identical')} "
            f"words={record.get('compared_words', 0)} "
            f"({record.get('seconds', 0)} s)")
    driven = [record for record in measured if record.get("driven")]
    admitted = [record for record in measured if record.get("predicate_admits")]
    block = {
        "facts": facts, "rows": measured,
        "rows_in_cell": len(measured),
        "rows_admitted": len(admitted),
        "rows_driven": len(driven),
        "rows_bit_identical": sum(1 for record in driven if record["bit_identical"]),
        "compared_words": sum(record.get("compared_words", 0) for record in driven),
        "distinct_specialisations": sorted({
            f"codes{tuple(record['codes'])}/phased{tuple(record['phased'])}"
            for record in driven}),
    }
    block["passed"] = bool(
        block["rows_in_cell"] > 0
        and block["rows_admitted"] == block["rows_in_cell"]
        and block["rows_driven"] == block["rows_in_cell"]
        and block["rows_bit_identical"] == block["rows_driven"]
        and block["compared_words"] > 0)
    payload["legs"]["corpus_rows"] = block
    save(payload, out)
    log(f"[corpus_rows] {block['rows_bit_identical']}/{block['rows_driven']} driven "
        f"identical of {block['rows_in_cell']} in the cell "
        f"({block['compared_words']} words) passed={block['passed']}")
    assert block["passed"], {k: v for k, v in block.items() if k != "rows"}


# ---------------------------------------------------------------------------
# LEG arbitration
# ---------------------------------------------------------------------------

def leg_arbitration(payload: Dict[str, Any], out: str) -> None:
    """What the SHIPPED composer installs on this cell's neighbouring seams TODAY.

    The measurement behind :data:`INSTALLABLE_REASON`. This product is NOT registered
    in this round, so it cannot be one of the answers -- what is measured is how many
    fused pairs the composer installs over ``step_B..update_E`` without it, which is
    exactly the number a product spanning ``update_H``/``step_D`` would have to beat.
    """
    rows, _facts = cell_rows()
    measured: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda entry: entry["label"]):
        grid, _inputs = census_grid(row)
        fields, pml = build_from_grid(grid, 99000)
        residency = Residency()
        composed = metal_launch.plan_step(fields, pml, residency=residency,
                                          sources=(), fuse=True)
        selected = dict(composed.selected)
        installed = sorted({name for name in selected.values()
                            if isinstance(name, str) and "fused" in name})
        entry = {"label": row["label"], "selected": selected,
                 "fused_pairs_installed": installed,
                 "n_fused_pairs_installed": len(installed)}
        # 4 - (installed pairs) launches over the four-slot path; this product would
        # take one slot from EACH neighbour, so it can only ever install ONE pair.
        entry["launches_today"] = 4 - len(installed)
        entry["launches_with_this_product"] = 3
        entry["verdict"] = ("loss" if entry["launches_with_this_product"]
                            > entry["launches_today"] else
                            ("tie" if entry["launches_with_this_product"]
                             == entry["launches_today"] else "gain"))
        measured.append(entry)
        log(f"[arbitration] {row['label']:<58} installed={installed} "
            f"today={entry['launches_today']} with_product=3 -> {entry['verdict']}")
        payload["legs"]["arbitration"] = measured
        save(payload, out)
    verdicts = {name: sum(1 for entry in measured if entry["verdict"] == name)
                for name in ("loss", "tie", "gain")}
    block = {
        "rows": measured, "verdicts": verdicts,
        "installable_declared": family.INSTALLABLE,
        "finding": (f"over the cell's {len(measured)} rows the shipped composer "
                    f"installs the neighbouring fused pairs as recorded above; "
                    f"installing this product instead is a LOSS on "
                    f"{verdicts['loss']}, a TIE on {verdicts['tie']} and a GAIN on "
                    f"{verdicts['gain']}. INSTALLABLE is False, and a gain count of "
                    f"0 is what that flag stands on"),
    }
    block["passed"] = bool(measured and verdicts["gain"] == 0
                           and not family.INSTALLABLE)
    payload["legs"]["arbitration"] = block
    save(payload, out)
    log(f"[arbitration] {verdicts} passed={block['passed']}")
    assert block["passed"], {k: v for k, v in block.items() if k != "rows"}


# ---------------------------------------------------------------------------
# LEG mutations
# ---------------------------------------------------------------------------

def kernel_mutations(expansion: str) -> Tuple[Tuple[str, Callable[[str], str],
                                                    Optional[bool], str, str], ...]:
    """``(label, transform, must_catch, value_class, why)``.

    The value class is part of the ENTRY, not of the leg: three of these defects are
    invisible on random data by construction (the complex helpers' zero cross terms and
    the negation spelling only differ where an operand plane is an exact zero), and an
    entry that named the wrong class would report a real defect as an equivalence.
    """
    own_tap = ("    float2 a   = own.a0;\n")
    halo_tap = (f"h_cell(si, j, k, {family.H_CELL_TAIL_ARGS}).a1")
    return (
        ("own_cell_reads_pre_launch_h",
         lambda s: kit.needle(s, own_tap, "    float2 a   = hi0[ii];\n"), True,
         values.UNIFORM,
         "the curl half takes its OWN cell's Hx from the PRE-update buffer instead of "
         "the register the constitutive half just produced: the first of the two "
         "defects this weld's whole shape exists to avoid"),
        ("halo_tap_reads_pre_launch_h",
         lambda s: kit.needle(s, halo_tap, "hi1[si * nyz + j * nzi + k]"), True,
         values.UNIFORM,
         "a BACKWARD tap reads the pre-update Hy at the neighbour cell instead of "
         "recomputing it: the second defect, and the one an in-place weld would "
         "commit non-deterministically"),
        ("constitutive_kms_wrong_axis",
         lambda s: kit.needle(s, "float kp_0 = kp0[i], km_0 = kmx[i];",
                              "float kp_0 = kp0[i], km_0 = kmy[i];"), True,
         values.UNIFORM,
         "the shared kms group indexed on the wrong axis -- the silent failure the "
         "sharing makes possible, and the reason the sub-lattice equality is asserted "
         "in the plan"),
        ("constitutive_accumulations_swapped",
         lambda s: kit.needle(
             s,
             # EIGHT SPACES: the lifted constitutive body sits inside `h_cell` and
             # `step_cell_function` indents it by four more than the certified text.
             "        a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"
             "        a0 = a0 - c_mul_coefficient_left(km_0, prev0);\n",
             "        a0 = a0 - c_mul_coefficient_left(km_0, prev0);\n"
             "        a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"), True,
         values.UNIFORM,
         "the two split-field accumulations reordered: float addition is not "
         "associative, and the array path's order is (f + kps*src) - kms*prev"),
        ("phase_multiply_operands_swapped",
         lambda s: kit.needle(s, "b_x = wx ? c_mul(b_x, px) : b_x;",
                              "b_x = wx ? c_mul(px, b_x) : b_x;"), True,
         values.UNIFORM,
         "the Bloch rotation with the phase on the LEFT: a full complex product's "
         "imaginary part fuses a different pair in each orientation, and the array "
         "path's is field-left (S:1862)"),
        ("componentwise_divide_by_reciprocal",
         lambda s: kit.needle(
             s,
             "static inline float2 c_mul_field_left(float2 z, float c) {",
             "static inline float2 c_mul_field_left(float2 z, float c) {\n"
             "    return float2(z.x / (1.0f / c), z.y / (1.0f / c));"), True,
         values.UNIFORM,
         "the split-field multiply respelled as a COMPONENTWISE complex-by-real "
         "divide by the reciprocal's reciprocal -- the spelling the Dcyl scan measured "
         "wrong, armed here so this kernel's freedom from division is a measurement"),
        ("plane_wise_field_multiply",
         lambda s: kit.needle(
             s,
             "static inline float2 c_mul_field_left(float2 z, float c) {",
             "static inline float2 c_mul_field_left(float2 z, float c) {\n"
             "    return float2(z.x * c, z.y * c);"), True,
         values.PM_ZERO_LATTICE,
         "the plane-wise fast path {re*c, im*c}: 24/128 words on the exhaustive "
         "signed-zero table and 0 on random data, which is why this entry names the "
         "+-0 lattice"),
        ("negation_zero_minus_x",
         lambda s: kit.needle(s, "fma(z.x, p.x, -(z.y * p.y))",
                              "fma(z.x, p.x, (0.0f - (z.y * p.y)))"), None,
         values.PM_ZERO_LATTICE,
         "negation spelled `0.0f - x` inside the FULL complex product, which only the "
         "Bloch rotation uses. RECORD-ONLY, and the classification is a measurement "
         "rather than a concession: leg helper_spellings measures this exact edit at "
         "36/512 words on the exhaustive sign/zero table, so the spelling IS wrong; "
         "what it is wrong by is the SIGN of an exactly-zero word, and a complete "
         "driver step does not carry one -- the curl differences the rotated operand "
         "against another and the split-field recurrence accumulates into a register, "
         "and IEEE addition does not preserve a lone negative zero through either. "
         "Measured 0/3 on BOTH value classes here. Recording it as an equivalence "
         "would be a false equivalence; the helper leg is where the defect is caught"),
        ("folded_zero_cross_term",
         lambda s: kit.needle(
             s, "return float2(fma(c, z.x, -(0.0f * z.y)),",
             "return float2(fma(c, z.x, -0.0f),"), None,
         values.PM_ZERO_LATTICE,
         "the coefficient-left zero cross term folded to a literal. RECORD-ONLY on the "
         "same measured grounds as `negation_zero_minus_x`: leg helper_spellings "
         "measures this exact edit at 12/128 words, so the folding IS wrong, and its "
         "difference is the sign of an exactly-zero word. This helper feeds the "
         "constitutive ACCUMULATION (a = a +/- c_mul_coefficient_left(...)), and an "
         "addition does not carry a lone negative zero into the stored accumulator, so "
         "it measures 0/3 on both value classes here. Contrast "
         "`plane_wise_field_multiply`, whose helper output IS stored directly by the "
         "split-field recurrence (u0[ii] = n0) and which this same walk catches 3/3 -- "
         "the pair is what shows the step-level instrument is discriminating and the "
         "two nulls are a property of where each helper's result goes"),
        ("byte_neutral_store_order",
         lambda s: kit.needle(
             s,
             "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
             "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n",
             "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n"
             "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"),
         False, values.UNIFORM,
         "the two scratch store lines swapped: no arithmetic, the control that shows "
         "this leg does not flag every edit"),
        ("params_unpack_order",
         lambda s: kit.needle(
             s,
             "    float dtdx = prm.dtdx;\n"
             "    float2 px = prm.px, py = prm.py, pz = prm.pz;\n",
             "    float2 px = prm.px, py = prm.py, pz = prm.pz;\n"
             "    float dtdx = prm.dtdx;\n"),
         False, values.UNIFORM,
         "the packed record unpacked in the other order: a second byte-neutral "
         "control, on the seam where a wrong struct layout would show"),
        ("coefficient_left_orientation",
         lambda s: kit.needle(s, "a0 = a0 + c_mul_coefficient_left(kp_0, src0);",
                              "a0 = a0 + c_mul_field_left(src0, kp_0);"),
         False, values.PM_ZERO_LATTICE,
         "the constitutive product with the FIELD on the left: measured 0/128 on the "
         "exhaustive signed-zero table by gate_metal_complex, and re-measured here as "
         "an equivalence rather than left as an inherited belief"),
    )


def mutated_walk(label: str, seed: int, steps: int, value_class: str,
                 function: Any = None, rotate: bool = True,
                 half_lattice: bool = False) -> Tuple[int, int, int]:
    """(steps diverged, launches, steps run) for one mutant against the array path."""
    reference = ArrayEngine(*build(label, seed, value_class))
    counter = kit.Counter(function) if function is not None else None
    engine = ProductEngine(*build(label, seed, value_class),
                           functions=({shaders.CONTRACT_OFF: counter}
                                      if counter is not None else None),
                           rotate=rotate, half_lattice=half_lattice)
    diverged = 0
    for _step in range(steps):
        reference.step()
        engine.step()
        if compare(state_of(engine.fields), state_of(reference.fields)):
            diverged += 1
    launches = counter.launches if counter is not None else engine.plan.launches
    return diverged, launches, steps


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    harness = kit.MutationHarness(payload, out, key="mutations")
    _name, _cell, boundaries, k_point = config(MUTATION_CASE)
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    fields, pml = build(MUTATION_CASE, 1)
    kinds = _boundary_kinds(fields.grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    flags, _values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(fields.grid, kinds), backward=True)
    shipped = family.complex_fused_hd_pair_source(codes, flags, expansion)
    steps = 3
    for label, transform, must_catch, value_class, why in kernel_mutations(expansion):
        try:
            mutant = transform(shipped)
        except LookupError:
            harness.record(label, "NEEDLE-MISSED", 0, 0, 0, must_catch, why)
            continue
        assert mutant != shipped, label
        function = getattr(compile_source(mutant), family.KERNEL)
        diverged, launches, ran = mutated_walk(MUTATION_CASE, 71000, steps,
                                               value_class, function)
        # The OTHER class is recorded as data: an entry naming the +-0 lattice that
        # also fires on random data is stronger than the entry claims, and one naming
        # the uniform class that fires on neither would be a bug in the entry.
        other = (values.PM_ZERO_LATTICE if value_class == values.UNIFORM
                 else values.UNIFORM)
        other_diverged, _other_launches, _other_ran = mutated_walk(
            MUTATION_CASE, 71500, steps, other, getattr(
                compile_source(mutant), family.KERNEL))
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches,
                                                          diverged),
                       launches, diverged, ran, must_catch, why,
                       extra={"kind": "kernel", "steps": steps,
                              "value_class": value_class,
                              "also_on": {other: other_diverged}})
    # HOST DEFECTS: the plan protocol and the host's choice of Yee sub-lattice, not
    # the shader.
    for label, rotate, half_lattice, why in (
            ("rotation_undone", False, False,
             "the H/f_w_H rotation undone after the launch: the next launch and the "
             "host read the pre-launch H, which is the in-place weld this shape "
             "exists to avoid"),
            ("half_integer_constitutive_lattice", True, True,
             "the constitutive kps group bound from the HALF-INTEGER sub-lattice: it "
             "compiles, converges and is a half-cell-wrong absorber profile, which is "
             "why it is a leg and not a comment")):
        diverged, launches, ran = mutated_walk(MUTATION_CASE, 73000, steps,
                                               values.UNIFORM, None, rotate,
                                               half_lattice)
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches,
                                                          diverged),
                       launches, diverged, ran, True, why,
                       extra={"kind": "host", "steps": steps,
                              "value_class": values.UNIFORM})


def leg_disarm(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    fields, pml = build(MUTATION_CASE, 1)
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    kinds = _boundary_kinds(fields.grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    flags, _values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(fields.grid, kinds), backward=True)
    function = family.compile_complex_fused_hd_pair(codes, flags, expansion)
    diverged, launches, ran = mutated_walk(MUTATION_CASE, 75000, 3, values.UNIFORM,
                                           function)
    row = {"diverged": diverged, "launches": launches, "steps": ran,
           "passed": diverged == 0 and launches == ran}
    payload["legs"]["disarm"] = row
    save(payload, out)
    log(f"[disarm] diverged={diverged} launches={launches}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# The census battery hook
# ---------------------------------------------------------------------------

class _HostGrid:
    xp = np


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    from meep_gpu.metal_kernels import coverage  # noqa: PLC0415

    return list(coverage._metal_backend_reasons(_HostGrid()))  # noqa: SLF001


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the predicate verdict on ONE lifted row.

    DELIBERATELY PREDICATE-ONLY. Driving a re-lifted corpus row through three engines
    inside the census child is a different measurement from the one leg ``corpus_rows``
    makes, and a battery that half-did it would put a number in the census that no leg
    of this gate stands behind.
    """
    sources = tuple(getattr(driver, "_sources", ()) or ())
    verdict = family.metal_complex_fused_hd_pair_coverage(
        driver.fields, driver.pml, sources, Residency())
    return {"metal_complex_fused_hd_pair_gate": {
        "family": family.FAMILY,
        "predicate_admits": bool(verdict.covered),
        "predicate_reasons": list(verdict.reasons),
        "n_sources": len(sources),
        "grid_cells": int(np.prod(driver.grid.shape)),
        "installable": family.INSTALLABLE,
    }}


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    started = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    payload: Dict[str, Any] = {
        "gate": "metal_complex_fused_hd_pair",
        "family": family.FAMILY,
        "slot": family.SLOT,
        "replaces": list(family.REPLACES),
        "cell": {"census": CENSUS, "seam_record": SEAM_RECORD,
                 "arms": list(CELL_ARMS)},
        "environment": kit.environment_stamp(),
        "probe_environment": ENVIRONMENT,
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    probe_arm, expansion = bound_expansion()
    if probe_arm is None:
        reasons.append("the complex expansion probe is not bound; the product cannot "
                       "be built and an arm chosen without it would be a guess")
    if reasons:
        return kit.cannot_certify(payload, out, reasons)
    payload["expansion"] = {"probe_bound": probe_arm, "measured": expansion}
    save(payload, out)

    legs: List[Tuple[str, Callable[[Dict[str, Any], str], None]]] = [
        ("binding_ceiling", leg_binding_ceiling),
        ("lift", leg_lift),
        ("driver_order", leg_driver_order),
        ("refusal", leg_refusal),
        ("product", leg_product),
        ("value_class", leg_value_class),
        ("null_control", leg_null_control),
        ("purity", leg_purity),
        ("divide_spelling", leg_divide_spelling),
        ("helper_spellings", leg_helper_spellings),
        ("corpus_rows", leg_corpus_rows),
        ("arbitration", leg_arbitration),
        ("mutations", leg_mutations),
        ("disarm", leg_disarm),
    ]
    ran = kit.run_legs(legs, payload, out, kit.wanted_legs(args.legs))

    kit.provenance(
        os.path.dirname(out),
        {"meep_gpu/metal_kernels/complex_fused_hd_pair.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                          "complex_fused_hd_pair.py"),
         "meep_gpu/metal_kernels/complex_fields.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "complex_fields.py"),
         "meep_gpu/metal_kernels/offdiag_weld_common.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                          "offdiag_weld_common.py"),
         "meep_gpu/metal_kernels/templates.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
         "meep_gpu/withdraw_hoist.py":
             os.path.join(API_ROOT, "meep_gpu", "withdraw_hoist.py"),
         "meep_gpu/stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
         "meep_gpu/driver.py": os.path.join(API_ROOT, "meep_gpu", "driver.py"),
         "parity/meep_gpu/gate_metal_complex_fused_hd_pair.py":
             os.path.abspath(__file__)},
        kernel_sources={label: source for arm in templates.EXPANSION_ARMS
                        for label, source in family.enumerate_sources(arm).items()},
        name="provenance.json")

    compared = 0
    certified = True
    for key in ("product", "value_class"):
        for row in payload["legs"].get(key, ()):
            compared += int(row["compared_words"])
            if not row["bit_identical"]:
                certified = False
    corpus = payload["legs"].get("corpus_rows")
    if isinstance(corpus, dict):
        compared += int(corpus.get("compared_words", 0))
        if not corpus.get("passed"):
            certified = False
    for key in ("binding_ceiling", "lift", "driver_order", "refusal", "null_control",
                "purity", "divide_spelling", "helper_spellings", "arbitration",
                "disarm"):
        if key in payload["legs"] and not payload["legs"][key].get("passed"):
            certified = False
    mutations = payload["legs"].get("mutations", [])
    armed = [row for row in mutations if row["must_catch"] is True]
    nulls = [row for row in mutations if row["must_catch"] is False]
    for row in armed:
        if row["caught"] != row["ran"] or row["ran"] == 0:
            certified = False
    for row in nulls:
        if row["caught"] != 0:
            certified = False

    corpus_block = corpus if isinstance(corpus, dict) else {}
    return kit.summarize(
        payload, out,
        claim=("ONE dispatch spanning complex update_H and complex step_D -- the "
               "certified complex constitutive body evaluated into write-only scratch "
               "for the thread's own cell and RECOMPUTED for each of the curl's three "
               "backward taps, then the H/f_w_H rotation -- reproduces the array path "
               "AND the two certified complex singles bit for bit per complete driver "
               "step on every configuration this predicate admits"),
        scope=(f"{len(CONFIGS)} configurations x {STEPS} steps (uniform band) + "
               f"4 x 4 steps (+-0 lattice) + "
               f"{corpus_block.get('rows_driven', 0)} of "
               f"{corpus_block.get('rows_in_cell', 0)} corpus rows of the cell driven "
               f"at their recorded grid configuration, all 24 stored volumes; launch "
               f"counts asserted per cycle; the binding ceiling bisected on this host"),
        stated_weakness=(
            "leg corpus_rows rebuilds each row's GRID from the census's recorded "
            "configuration (shape, boundary kinds, k_point, resolution, Courant) and "
            "does NOT re-run the row's own script: its materials, sources, absorber "
            "thickness and geometry are the synthetic fixture's. INSTALLABLE is False "
            "and the product is NOT REGISTERED in this round, so it executes nowhere "
            "outside this gate and its tests, and no dispatch or timing claim is "
            "licensed. The subnormal band is a per-step refusal on this executor, not "
            "a divergence. The folded complex, cylindrical complex and complex beta "
            "cells are refused and untouched"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"mutations_armed": len(armed),
               "mutations_caught_of_armed": sum(1 for row in armed
                                                if row["caught"] == row["ran"]),
               "mutations_null_controls": len(nulls),
               "mutation_verdicts": {row["mutation"]: row["verdict"]
                                     for row in mutations},
               "cell_rows_in_census": corpus_block.get("rows_in_cell"),
               "cell_rows_driven": corpus_block.get("rows_driven"),
               "expansion": payload.get("expansion")})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
