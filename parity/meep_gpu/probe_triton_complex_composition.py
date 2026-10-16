"""Complete-driver CUDA gate for the complex-field (Bloch) Triton composition.

The sub-step gate (``gate_triton_complex.py``) certifies each kernel against an
in-file transcription; this complementary probe asks the composition question:
with ``step_B``/``update_H``/``step_D``/``update_E`` served by
``complex_fields``' engine-route plans — installed by monkeypatch, production
dispatch untouched, ``plan_fast_path`` still returning None — does the complete
CuPy driver state reproduce the array-path driver EXACTLY after every step, for
a STATED number of steps, sources, PML, boundary fills and metal zeroing
included?

The ten cases span the corpus demand plus the zero-coincidence class the
random-seeded certification provably could not see:

1. ``complex_k0``               — force_complex_fields at k = 0, metallic x/y
   walls under a two-axis PML, no source (wvg-src / solve-cw storage class;
   also pins that every phase is None -> the multiply is SKIPPED, measured on
   seeds that carry imag -0.0 planes).
2. ``complex_k0_electric_src``  — case 1 + an Ez point source with a COMPLEX
   amplitude (the wvg-src CW / pw_amp complex-amplitude class), proven
   non-vacuous against a source-free control.
3. ``complex_k0_magnetic_src``  — case 1 + an Hy source (closes the B seam).
4. ``bloch_inplane``            — mode_coeff_phase's own k = (0.8753, 1.2181, 0)
   on a fully wrapped 2-D cell with the absorber graded under the phased-axis
   wrap, Gaussian Ez source (binary_grating_oblique / oblique-planewave class).
5. ``bloch_one_axis_phase_none``— k = (kx, 0, 0): x phased while y wraps
   UNPHASED — the None-skip must stay bit-identical inside a phased run.
6. ``bloch_transverse_collapsed`` — refl-angular's oblique legs put k on a
   COLLAPSED axis. ``Grid`` refuses that construction TODAY (grid.py:980-991),
   so this case ATTEMPTS the build and records the refusal verbatim — the row
   stays in the artifact so the open question (task #9's lift vs the grid
   constructor) is visible, not silently dropped. The kernel-side n==1
   wrap+phase mechanics are measured gridlessly by the sub-step gate's
   ``transverse_collapsed`` rows.
7. ``bloch_k_on_absorbing_axis``— refl-angular's recorded theta=0 point:
   k = (0, 0, 1.25) on the same z axis that carries the PML, Ex source.
8. ``zero_init_quiet_k0``       — the ZERO-INIT class, k = 0: all fields start
   +0.0 (no seeding), Ez CW source with a complex amplitude, THIN 3-cell x/y
   absorber whose deepest ``kms = kappa - sigma`` is NEGATIVE (-0.5111
   integer / -0.0494 half-integer at res 12, courant 0.35). A negative
   coefficient times a quiet +0.0 word is the reachable producer of stored
   ``-0.0`` words on the array path (NumPy census 2026-08-12: 224 words
   on every ODD step across fu_Bx/fu_Bz/fu_Dx/fu_Dz/Bz/f_w_Hz, 0 on even —
   the -0 feeds back as ``-0 * negative = +0`` — hence the ODD budget). Since
   the per-part alignment (bit_parity 0.10) the array path forms every such
   product part by part, so BOTH words of the cell are -0 (566 words, 283
   cells, on every odd step): the discriminating class is now the cell whose
   imaginary word is -0 beside a -0 real word, which a whole-complex product
   (the coefficient promoted to ``c + 0i``) stores as ``-0 + 0i`` instead
   (``c*im + 0*re = -0 + +0 = +0``).
9. ``zero_init_quiet_bloch``    — the same class under a SECOND-QUADRANT
   Bloch phase (k = 0.11 on x, exp(2*pi*i*0.33): p_re < 0 < p_im), periodic
   both axes, thin 3-cell x/y absorber graded under the phased wrap. Every
   quiet read of a connected plane is a connection event: the phase product's
   real word ``(a*re) - (b*im)`` is ``-0 - +0 = -0`` where a fused, negated
   addend would store +0 (counted per step, required > 0; since the alignment
   the connection sits on the UNSHIFTED operand's plane 0 in both directions).
10. ``edge_sparse_y``           — the Brillouin edge on y (k = 0.25, L = 2:
   ``grid.bloch_phase`` is exactly -1, MEEP's NEGATION), 3-D between metallic
   x and z walls under their layers, seeded SPARSE: most words are zeros of
   random sign, so a stencil whose every operand is zero is common and the
   sign of a negated connection reaches stored words. The armed negation leg
   below runs here (on the host interpreter, with the lowered spelling planted,
   it diverged at step 1 in 44 words; the 2-D edge row it replaced, z
   invariant, never did in 16 steps).

Two-level NON-VACUITY, both asserted in-run for the zero-init rows: (1) the
source must separate the run from a source-free control (every sourced case's
existing discipline); (2) the REFERENCE (array-path) state must carry at least
one negative-zero word (uint32 0x80000000) in the compared volumes at the
final compared step, and a persisting QUIET REGION (>= 25% of cells all-zero
across the 24 arrays) — otherwise the case cannot discriminate the
canonicalization and reports itself VACUOUS by AssertionError.

TWO ARMED mutation legs, both launch-counted, DISARMED/NOT-CAUGHT are
failures: (a) case 4 re-run with the step_D plan carrying step_B's connection
factor (``conj(eikna)`` for ``eikna``: the band-structure sign error); (b) case
10 re-run with the curl kernel compiled from a source mutation that respells
the connection's negation ``v * -1.0`` as the unary ``-v`` (Triton lowers it as
``0.0 - v``, which turns -(+0) into +0) — the negation-lowering gap itself,
which must diverge from the array path within the budget on the sparse edge
state.

Requirements carried from prior defects: per-complete-step uint32 compare of
EVERY allocated primary and auxiliary; a non-power-of-two Courant (0.35)
everywhere; the step budget STATED in the artifact (bit-identity is claimed for
exactly that many steps — 2d_pml certified 60/60 first differed at step 67);
unbuffered per-case progress; atomic JSON rewrite per case; own provenance
record (``fingerprints.json`` is another track's file and is only HASHED).

SUBNORMAL POLICY. This probe runs UNDER THE STRIPPED IEEE-KEEP POLICY — the
ship configuration — installed via the gate's ``install_ftz_strip`` (wrap
``cupy.cuda.compiler.compile_using_nvrtc``, filter ``-ftz=true``, private
policy-suffixed ``CUPY_CACHE_DIR``) BEFORE ``guard_kernel_compilation`` so the
guard wraps the strip and both apply; every claim in this artifact states the
policy it was cut under.

Usage (the GPU host, one clear device; the cache dir must carry the policy token)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_complex_composition.py \\
        --out results/triton_complex_<date>/composition.json

Laptop (no CUDA/triton): ``--self-check`` builds every case's drivers on NumPy,
steps the array path two steps per seeded case, runs the zero-init rows' FULL
budget with the signed-zero census and its non-vacuity asserts (the local
reachability evidence), records case 6's constructor refusal, and says in so
many words that the Triton legs were skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The atomic writer, provenance and the seeding all live in the sub-step
# gate; re-implementing any of them here would make this
# probe's verdict about this file rather than about the contract.
import gate_triton_complex as gate  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log


def cw_source(component: str, amplitude: complex) -> Dict[str, Any]:
    return {"component": component, "frequency": 0.7,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude}


def gaussian_source(component: str) -> Dict[str, Any]:
    return {"component": component, "source_type": "gaussian",
            "frequency": 0.8, "fwidth": 0.4,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": 1.0}


# name, dimensions, cell, boundaries, k_point, pml, sources, steps,
# expected per-axis phased flags (None -> the case is EXPECTED to refuse
# at Grid construction and is recorded, not run), checks (None -> the
# random+targeted seeded class; a dict -> the ZERO-INIT class with its
# in-run non-vacuity requirements, and the step budget MUST be odd: the
# -0.0 census is 220/221 words on odd steps and exactly 0 on even ones).
CASES: Tuple[Tuple[Any, ...], ...] = (
    ("complex_k0", 2, (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.0), {"x": 5, "y": 5}, (), 8, (0, 0, 0), None),
    ("complex_k0_electric_src", 2, (3.0, 3.0, 0.0),
     ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0), {"x": 5, "y": 5},
     (cw_source("Ez", complex(0.75, 0.4)),), 10, (0, 0, 0), None),
    ("complex_k0_magnetic_src", 2, (3.0, 3.0, 0.0),
     ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0), {"x": 5, "y": 5},
     (cw_source("Hy", 0.75),), 10, (0, 0, 0), None),
    ("bloch_inplane", 2, (3.0, 3.0, 0.0), "periodic",
     (0.8753, 1.2181, 0.0), {"x": 5}, (gaussian_source("Ez"),), 10, (1, 1, 0),
     None),
    ("bloch_one_axis_phase_none", 2, (3.0, 3.0, 0.0), "periodic",
     (0.37, 0.0, 0.0), {"x": 5}, (), 8, (1, 0, 0), None),
    ("bloch_transverse_collapsed", 1, (0.0, 0.0, 11.0), "periodic",
     (0.6, 0.45, 0.0), {"z": 5}, (gaussian_source("Ex"),), 10, None, None),
    ("bloch_k_on_absorbing_axis", 1, (0.0, 0.0, 12.0), "periodic",
     (0.0, 0.0, 1.25), {"z": 5}, (gaussian_source("Ex"),), 10, (0, 0, 1),
     None),
    # The zero-init class (docstring cases 8 and 9). Thin 3-cell absorbers ON
    # PURPOSE: their deepest kms = kappa - sigma is negative (-0.5111 integer
    # lattice at res 12 / courant 0.35), and negative-coefficient * quiet-+0.0
    # words are the array path's reachable -0.0 producer. The 5-cell absorber
    # keeps kms >= +0.0934 everywhere and was measured census-silent across
    # amplitudes 1.0 down to 1e-42 (reachability sweep, 2026-08-12).
    ("zero_init_quiet_k0", 2, (3.0, 3.0, 0.0),
     ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0), {"x": 3, "y": 3},
     (cw_source("Ez", complex(0.75, 0.4)),), 11, (0, 0, 0),
     {"zero_init": True, "min_final_neg_zero_words": 1,
      "min_final_quiet_fraction": 0.25, "min_rotate_events_per_step": 0}),
    ("zero_init_quiet_bloch", 2, (3.0, 3.0, 0.0), "periodic",
     (0.11, 0.0, 0.0), {"x": 3, "y": 3}, (cw_source("Ez", 1.0),), 11,
     (1, 0, 0),
     {"zero_init": True, "min_final_neg_zero_words": 1,
      "min_final_quiet_fraction": 0.25, "min_rotate_events_per_step": 1}),
    # The Brillouin edge (docstring case 10): k*L = 0.5 exactly, so the y phase is
    # exactly -1 and MEEP connects by negation; the seeded state is sparse.
    ("edge_sparse_y", 3, (1.5, 2.0, 1.5), ("metallic", "periodic", "metallic"),
     (0.0, 0.25, 0.0), {"x": 4, "z": 4}, (), 8, (0, 1, 0), {"sparse": 0.04}),
)

#: The connection's negation (``complex_fields._connect``, MEEP's CONNECT_NEGATE),
#: spelled ``* -1.0`` per part -> the unary spelling the armed negation-mutation
#: leg compiles back in. Every other product of the kernels has a real operand and
#: no negated addend, so these are the only negation sites.
NEGATION_MUTATION_SITES: Tuple[Tuple[str, str], ...] = (
    ("out_re = v_re * -1.0", "out_re = -v_re"),
    ("out_im = v_im * -1.0", "out_im = -v_im"),
)

EXPECTED_PLANS = {"step_B": "ComplexPmlCurlPlan",
                  "update_H": "ComplexConstitutivePlan",
                  "step_D": "ComplexPmlCurlPlan",
                  "update_E": "ComplexConstitutivePlan"}

STATE_NAMES = gate.ALL_STATE
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
#: Every complex run under PML must carry the full 24-array state; a missing
#: array would silently shrink the comparison, so the inventory is asserted.
REQUIRED_STATE = set(STATE_NAMES)


def build_driver(xp, name, dimensions, cell, boundaries, k_point, pml_spec,
                 source_specs, seed: int, prefer_gpu: bool,
                 zero_init: bool = False, sparse: Optional[float] = None):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, mandatory
        boundaries=boundaries, k_point=k_point,
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(xp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    for declaration in source_specs:
        driver.add_source(dict(declaration))
    if zero_init:
        # The zero-init class: every field stays the allocator's +0.0 words
        # (fields.py zero-initializes storage); the SOURCE is the only writer,
        # and the -0.0 census below is what the case exists to reach.
        return driver
    rng = np.random.default_rng(seed)
    for primary in PRIMARY_NAMES:
        # Scaled INSIDE the builder: a complex multiply by 0.25 afterwards
        # would rewrite the targeted signed-zero planes.
        host = gate._seed_host_complex(tuple(shape), rng, scale=0.25)
        if sparse:
            # Most words become zeros of random sign (case 10's class).
            for part in (host.real, host.imag):
                keep = rng.random(size=tuple(shape)) < sparse
                zeros = np.where(rng.random(size=tuple(shape)) < 0.5,
                                 np.float32(-0.0), np.float32(0.0))
                part[...] = np.where(keep, part, zeros)
        driver.set_field(primary, xp.asarray(host))
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def compare(cp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(cp.asnumpy(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(cp.asnumpy(right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_floats": int(np.count_nonzero(a != b)),
            "total_floats": int(a.size)}


# ---------------------------------------------------------------------------
# The signed-zero census — the zero-init rows' non-vacuity level 2
# ---------------------------------------------------------------------------

NEG_ZERO_WORD = np.uint32(0x80000000)


def host_state(cp, arrays: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """The reference state brought to the host, complex64, one dict."""
    return {name: np.ascontiguousarray(cp.asnumpy(arrays[name]))
            for name in sorted(arrays)}


def negative_zero_census(state_host: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """-0.0 word count per array, plus the DISCRIMINATING cell count: cells
    whose real AND imaginary words are both -0.0 — what a negative real
    coefficient times a quiet ``+0 + 0i`` word stores part by part (MEEP's
    ``DOCMP`` products), and what a whole-complex product with the coefficient
    promoted to ``c + 0i`` would store as ``-0 + 0i`` instead
    (``c*im + 0*re = -0 + +0 = +0``)."""
    total = 0
    per_array: Dict[str, int] = {}
    discriminating = 0
    for name, arr in state_host.items():
        words = arr.view(np.uint32)
        neg = int(np.count_nonzero(words == NEG_ZERO_WORD))
        if neg:
            per_array[name] = neg
        total += neg
        re, im = arr.real, arr.imag
        discriminating += int(np.count_nonzero(
            (re == 0) & np.signbit(re) & (im == 0) & np.signbit(im)))
    return {"neg_zero_words": total, "per_array": per_array,
            "discriminating_cells": discriminating}


def quiet_fraction(state_host: Dict[str, np.ndarray]) -> float:
    """Fraction of cells all-zero MAGNITUDE across every compared array — the
    persisting quiet region the case sizes its domain against. A -0.0 word is
    magnitude-zero and deliberately counts as quiet: signed zeros riding the
    quiet region are exactly the class under test."""
    shape = next(iter(state_host.values())).shape
    quiet = np.ones(shape, dtype=bool)
    for arr in state_host.values():
        quiet &= (arr.real == 0) & (arr.imag == 0)
    return float(np.count_nonzero(quiet)) / quiet.size


def rotate_event_count(state_host: Dict[str, np.ndarray],
                       phases: Sequence[Optional[complex]]) -> int:
    """Connection-event occupancy: connected operands (plane 0 of each phased
    axis, the factor of each direction: ``conj(eikna)`` under B, ``eikna``
    under D, as ``complex_fields.connection_arguments`` rounds it) whose
    connected real word depends on the subtraction's zero sign —
    f32(g_re*p_re) an exact -0 while f32(g_im*p_im) is an exact +0, so the
    unfused ``-0 - +0`` stores -0.0 where a fused, negated addend stores +0.0.
    Counted on the reference state; whether an event survives to a stored
    word is the byte compare's question."""
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    total = 0
    f32 = np.float32
    for axis, phase in enumerate(phases):
        if phase is None:
            continue
        for backward in (False, True):
            table: List[Optional[complex]] = [None, None, None]
            table[axis] = phase
            kinds, values = complex_fields.connection_arguments(table, backward)
            if kinds[axis] != complex_fields.CONNECTION_KINDS["phase"]:
                continue
            p_re = f32(values[2 * axis])
            p_im = f32(values[2 * axis + 1])
            sources = ("Hx", "Hy", "Hz") if backward else ("Ex", "Ey", "Ez")
            plane = 0  # the connected plane, both directions
            for component, name in enumerate(sources):
                if component == axis or name not in state_host:
                    continue
                selector = [slice(None)] * 3
                selector[axis] = plane
                lane = state_host[name][tuple(selector)]
                product_re = (lane.real.astype(f32) * p_re).astype(f32)
                product_im = (lane.imag.astype(f32) * p_im).astype(f32)
                event = ((product_re == 0) & np.signbit(product_re)
                         & (product_im == 0) & ~np.signbit(product_im))
                total += int(np.count_nonzero(event))
    return total


def census_point(cp, reference, checks: Optional[Dict[str, Any]]
                 ) -> Optional[Dict[str, Any]]:
    """One step's census on the REFERENCE driver, zero-init rows only."""
    if not (checks and checks.get("zero_init")):
        return None
    state_arrays = host_state(cp, state(reference))
    point = negative_zero_census(state_arrays)
    point["quiet_fraction"] = round(quiet_fraction(state_arrays), 4)
    phases = tuple(reference.grid.bloch_phase(axis) for axis in range(3))
    point["rotate_events"] = rotate_event_count(state_arrays, phases)
    return point


def assert_zero_init_nonvacuous(name: str, checks: Dict[str, Any],
                                census_rows: Sequence[Dict[str, Any]]) -> None:
    """Non-vacuity level 2, asserted in-run: a zero-init case whose reference
    carries no -0.0 word at the final compared step cannot discriminate the
    canonicalization and must FAIL as vacuous, never pass."""
    final = census_rows[-1]
    if final["neg_zero_words"] < checks["min_final_neg_zero_words"]:
        raise AssertionError(
            f"{name} is VACUOUS: the reference carried "
            f"{final['neg_zero_words']} negative-zero words at the final "
            f"compared step (needs >= {checks['min_final_neg_zero_words']}); "
            f"the case cannot discriminate the per-part products' "
            f"signed zeros and must not be reported as passing")
    if final["quiet_fraction"] < checks["min_final_quiet_fraction"]:
        raise AssertionError(
            f"{name}: the quiet region did not persist — final quiet "
            f"fraction {final['quiet_fraction']} < "
            f"{checks['min_final_quiet_fraction']}; grow the cell or cut "
            f"the budget")
    floor = checks.get("min_rotate_events_per_step", 0)
    if floor:
        least = min(row["rotate_events"] for row in census_rows)
        if least < floor:
            raise AssertionError(
                f"{name}: the connection was not exercised on its signed-zero "
                f"class every step (min {least} events < {floor}); a case "
                f"that never exercises the connection product cannot see "
                f"_connect")


def build_plans(driver) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    plans = {
        "step_B": complex_fields.plan_complex_pml_curl(
            driver.fields, driver.pml, "step_B"),
        "update_H": complex_fields.plan_complex_constitutive(
            driver.fields, driver.pml, "H"),
        "step_D": complex_fields.plan_complex_pml_curl(
            driver.fields, driver.pml, "step_D"),
        "update_E": complex_fields.plan_complex_constitutive(
            driver.fields, driver.pml, "E"),
    }
    missing = {slot for slot, plan in plans.items() if plan is None}
    if missing:
        reasons = {
            "step_B": complex_fields.complex_pml_curl_coverage(
                driver.fields, driver.pml, "step_B").reasons,
            "update_H": complex_fields.complex_constitutive_coverage(
                driver.fields, driver.pml, "H").reasons,
            "step_D": complex_fields.complex_pml_curl_coverage(
                driver.fields, driver.pml, "step_D").reasons,
            "update_E": complex_fields.complex_constitutive_coverage(
                driver.fields, driver.pml, "E").reasons,
        }
        raise AssertionError(f"plans refused for {sorted(missing)}: "
                             f"{ {k: list(v) for k, v in reasons.items()} }")
    selected = {slot: type(plan).__name__ for slot, plan in plans.items()}
    if selected != EXPECTED_PLANS:
        raise AssertionError(f"selected={selected}, expected={EXPECTED_PLANS}")
    return plans


class PlanCounter:
    """Launch counting for the armed mutation leg — a leg whose plan never ran
    measures nothing and must FAIL, not pass."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.launches = 0

    def run(self, guard: Optional[bool] = None) -> None:
        self.launches += 1
        self.plan.run(guard=guard)


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the four sub-steps from the plans, for the owner fields only.
    Sources, symmetry fills (no-ops here: no folds), ``zero_metal_*`` and
    ``update_P`` (no polarizations) stay the driver's own. Production dispatch
    is untouched — this is a monkeypatch on the driver module's names."""
    names = ("step_B", "update_H", "step_D", "update_E")
    originals = {name: getattr(driver_module, name) for name in names}

    def wrapper(name):
        def wrapped(fields, pml=None):
            if fields is owner and name in plans:
                plans[name].run()
                return None
            return originals[name](fields, pml)
        return wrapped

    for name in names:
        setattr(driver_module, name, wrapper(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def run_case(cp, name, dimensions, cell, boundaries, k_point, pml_spec,
             source_specs, steps, expected_phased,
             checks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    zero_init = bool(checks and checks.get("zero_init"))
    sparse = (checks or {}).get("sparse")
    if expected_phased is None:
        # Case 6: the construction is EXPECTED to refuse (grid.py:980-991).
        try:
            build_driver(cp, name, dimensions, cell, boundaries, k_point,
                         pml_spec, source_specs, 20260811, prefer_gpu=True)
        except ValueError as exc:
            log(f"case {name}: EXPECTED grid refusal recorded")
            return {"case": name, "skipped": str(exc),
                    "expected_refusal": "grid.py:980-991 refuses a nonzero k "
                                        "component on an invariant axis; the "
                                        "task-#9 lift's engine route is an "
                                        "open question, tracked in the "
                                        "sub-step gate's transverse_collapsed "
                                        "rows"}
        raise AssertionError(
            f"{name}: Grid ACCEPTED a nonzero k component on a collapsed axis "
            f"— grid.py:980-991 has changed; the composition case must now be "
            f"written and measured, not skipped")

    reference = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True,
                             zero_init=zero_init, sparse=sparse)
    candidate = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True,
                             zero_init=zero_init, sparse=sparse)
    control = (build_driver(cp, name, dimensions, cell, boundaries, k_point,
                            pml_spec, (), 20260811, prefer_gpu=True,
                            zero_init=zero_init, sparse=sparse)
               if source_specs else None)
    undo = lambda: None  # noqa: E731
    try:
        plans = build_plans(candidate)
        # The grid's phased axes; the plan's connection kinds are MEEP's classes of
        # them (a phase within 1e-13 of +1, e.g. case 7's exp(2*pi*i*15), connects
        # as a copy and carries kind 0).
        phased = tuple(0 if candidate.grid.bloch_phase(axis) is None else 1
                       for axis in range(3))
        if phased != tuple(expected_phased):
            raise AssertionError(f"{name}: grid phased flags {phased} != "
                                 f"expected {expected_phased}")
        missing = REQUIRED_STATE - set(state(candidate))
        if missing:
            raise AssertionError(f"{name}: state inventory is short of the "
                                 f"complex PML set: missing {sorted(missing)}")
        undo = install(driver_module, plans, candidate.fields)
        row: Dict[str, Any] = {
            "case": name,
            "shape": [int(v) for v in candidate.shape],
            "boundaries": boundaries if isinstance(boundaries, str)
            else list(boundaries),
            "k_point": list(k_point),
            "pml": dict(pml_spec),
            "phased": list(phased),
            "connections": list(plans["step_B"].connections),
            "connection_values": list(plans["step_B"].connection_values),
            "sources": [f"{d['component']} amplitude={d.get('amplitude')!r}"
                        for d in source_specs],
            "steps": steps,
            "step_budget_note": "bit-identity is claimed for exactly this many "
                                "steps and no further",
            "init": "zero (all fields +0.0, source only)" if zero_init
                    else ("seeded sparse (zeros of random sign)" if sparse
                          else "seeded (random + targeted signed-zero planes)"),
            "checks": checks,
            "per_step": [],
        }
        census_rows: List[Dict[str, Any]] = []
        source_effect_seen = not source_specs
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            got_state = state(candidate)
            if set(ref_state) != set(got_state):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs: "
                    f"{sorted(ref_state)} vs {sorted(got_state)}")
            parts = {key: compare(cp, got_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            # No per-step source-STATE compare, deliberately: the declared
            # sources build ContinuousSource / GaussianPulsedSource, which
            # implement only the non-integrated path and carry no mutable
            # per-step state (``_applied_dipole`` is VolumeSource's,
            # sources.py:2027). A compare on that attribute measured
            # 0j == 0j unconditionally (2026-08-11 vacuity audit) — a
            # decorative field recorded as if measured — so it was dropped;
            # a source's entire per-step effect is in the field arrays the
            # 24-array uint32 compare covers.
            if control is not None:
                source_effect_seen = source_effect_seen or any(
                    not compare(cp, ref_state[key],
                                getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            point = {
                "step": step,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"]
                                        for p in parts.values()),
                "total_floats": sum(p["total_floats"] for p in parts.values()),
                "differing_arrays": sorted(
                    key for key, p in parts.items() if not p["bit_identical"]),
                "source_effect_seen": source_effect_seen,
            }
            census = census_point(cp, reference, checks)
            if census is not None:
                census_rows.append(census)
                point["reference_census"] = census
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} "
                f"identical={point['bit_identical']} "
                f"source_effect={source_effect_seen} "
                f"ndiff={point['differing_floats']}"
                + (f" neg0={census['neg_zero_words']}"
                   f" quiet={census['quiet_fraction']}"
                   f" rot={census['rotate_events']}" if census else "")
                + f" ({time.time() - started:.1f} s)")
            if not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        if not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from its "
                f"control — the case is vacuous")
        if zero_init:
            assert_zero_init_nonvacuous(name, checks, census_rows)
            row["census_final"] = census_rows[-1]
        row["bit_identical"] = True
        row["source_effect_seen"] = source_effect_seen
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
        cp.get_default_memory_pool().free_all_blocks()


def run_armed_mutation(cp) -> Dict[str, Any]:
    """Case 4 with the step_D plan carrying step_B's connection factor
    (``conj(eikna)`` for ``eikna``) — MUST diverge within the budget, and the
    mutated plan MUST demonstrably have launched."""
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    (name, dimensions, cell, boundaries, k_point, pml_spec, source_specs,
     steps, _phased, _checks) = CASES[3]
    reference = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True)
    candidate = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True)
    undo = lambda: None  # noqa: E731
    try:
        plans = build_plans(candidate)
        from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

        mutated = plans["step_D"]
        phases = tuple(candidate.grid.bloch_phase(axis) for axis in range(3))
        _kinds, values = complex_fields.connection_arguments(phases, backward=False)
        flipped = sum(1 for axis, kind in enumerate(mutated.connections)
                      if kind == complex_fields.CONNECTION_KINDS["phase"]
                      and values[2 * axis:2 * axis + 2]
                      != mutated.connection_values[2 * axis:2 * axis + 2])
        if flipped == 0:
            raise AssertionError("armed mutation: no connected axis to flip — "
                                 "the leg is disarmed")
        mutated.connection_values = tuple(values)
        counter = PlanCounter(mutated)
        plans["step_D"] = counter
        undo = install(driver_module, plans, candidate.fields)
        first_divergence = None
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            cp.cuda.runtime.deviceSynchronize()
            parts = {key: compare(cp, getattr(candidate.fields, key),
                                  getattr(reference.fields, key))
                     for key in PRIMARY_NAMES}
            identical = all(p["bit_identical"] for p in parts.values())
            log(f"[armed] step {step}/{steps} identical={identical} "
                f"launches={counter.launches}")
            if not identical:
                first_divergence = step
                break
        if counter.launches == 0:
            raise AssertionError("armed mutation DISARMED: the mutated step_D "
                                 "plan never launched")
        if first_divergence is None:
            raise AssertionError(
                f"armed mutation NOT CAUGHT in {steps} steps with "
                f"{counter.launches} mutated launches — the per-step compare "
                f"cannot see the phase-conjugate sign error and the harness "
                f"is not trustworthy")
        return {"case": name, "mutation": "step_D carries step_B's connection factor",
                "axes_flipped": flipped, "launches": counter.launches,
                "caught_at_step": first_divergence, "caught": True}
    finally:
        undo()
        reference.close()
        candidate.close()
        cp.get_default_memory_pool().free_all_blocks()


def run_armed_negation_mutation(cp) -> Dict[str, Any]:
    """Case 10 (the sparse Brillouin-edge state) with the curl kernel compiled
    from the unary spelling of the connection's negation — the
    negation-lowering gap itself. Triton lowers ``-x`` as ``0.0 - x``
    (semantic.py:386-391), which turns the negation of a +0 word into +0
    where MEEP's sign-bit negation stores -0, so the mutant MUST diverge from
    the array-path reference within the budget; a leg that never launched or
    never diverged proves case 10 discriminates nothing and FAILS."""
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    case = next(row for row in CASES if row[0] == "edge_sparse_y")
    (name, dimensions, cell, boundaries, k_point, pml_spec, source_specs,
     steps, _phased, checks) = case
    sparse = checks["sparse"]

    shipped = gate.shipped_source()
    mutated = shipped
    hits = 0
    for spelled, unary in NEGATION_MUTATION_SITES:
        if spelled in mutated:
            mutated = mutated.replace(spelled, unary)
            hits += 1
    if hits != len(NEGATION_MUTATION_SITES):
        raise AssertionError(
            f"armed negation mutation NEEDLE MISSED ({hits}/"
            f"{len(NEGATION_MUTATION_SITES)} sites): the helper spelling "
            f"drifted and the leg no longer compiles the old form")
    mutated_curl = gate.compile_mutated(mutated, "bloch_pml_curl_step")

    reference = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True,
                             sparse=sparse)
    candidate = build_driver(cp, name, dimensions, cell, boundaries, k_point,
                             pml_spec, source_specs, 20260811, prefer_gpu=True,
                             sparse=sparse)
    undo = lambda: None  # noqa: E731
    try:
        plans = build_plans(candidate)
        plans["step_B"]._kernel = mutated_curl
        plans["step_D"]._kernel = mutated_curl
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        undo = install(driver_module, counters, candidate.fields)
        first_divergence = None
        detail = None
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            parts = {key: compare(cp, getattr(candidate.fields, key),
                                  ref_state[key])
                     for key in sorted(ref_state)}
            identical = all(p["bit_identical"] for p in parts.values())
            launches = sum(c.launches for c in counters.values())
            log(f"[armed-negation] step {step}/{steps} identical={identical} "
                f"launches={launches}")
            if not identical:
                first_divergence = step
                detail = {
                    "differing_floats": sum(p["differing_floats"]
                                            for p in parts.values()),
                    "differing_arrays": sorted(
                        key for key, p in parts.items()
                        if not p["bit_identical"]),
                }
                break
        launches = sum(c.launches for c in counters.values())
        if launches == 0:
            raise AssertionError("armed negation mutation DISARMED: the "
                                 "mutated plans never launched")
        if first_divergence is None:
            raise AssertionError(
                f"armed negation mutation NOT CAUGHT in {steps} steps with "
                f"{launches} mutated launches — the sparse edge case cannot "
                f"see the unary-minus lowering and its non-vacuity claim is "
                f"false")
        return {"case": name,
                "mutation": "the connection's negation respelled unary "
                            "(2 sites, complex_fields._connect)",
                "sites": hits, "launches": launches,
                "caught_at_step": first_divergence, "detail": detail,
                "caught": True}
    finally:
        undo()
        reference.close()
        candidate.close()
        cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Laptop self-check — case builders + array path, no CUDA, no Triton
# ---------------------------------------------------------------------------

class _NumpyAsCupy:
    """The two cp spellings the census helpers use, served by NumPy."""

    @staticmethod
    def asnumpy(array):
        return np.asarray(array)

    @staticmethod
    def asarray(array):
        return np.asarray(array)


def zero_init_reference_leg(case: Sequence[Any]) -> Dict[str, Any]:
    """One zero-init case's ARRAY-PATH reference leg on NumPy, full budget:
    builds the sourced driver and its source-free control, steps the declared
    budget, computes the per-step signed-zero census, and ASSERTS the case's
    own non-vacuity checks — the laptop half of the reachability evidence
    (the device run re-computes the same census on the CuPy reference).
    Used by ``--self-check`` and by the merge-bar tests."""
    (name, dimensions, cell, boundaries, k_point, pml_spec, source_specs,
     steps, expected_phased, checks) = case
    if not (checks and checks.get("zero_init")):
        raise ValueError(f"{name} is not a zero-init case")
    if steps % 2 == 0:
        raise AssertionError(
            f"{name}: the budget must be ODD — the -0.0 census is nonzero on "
            f"odd steps only (the -0 feeds back as -0*negative = +0), so an "
            f"even final step compares a census-empty state")
    driver = build_driver(np, name, dimensions, cell, boundaries, k_point,
                          pml_spec, source_specs, 20260811, prefer_gpu=False,
                          zero_init=True)
    control = build_driver(np, name, dimensions, cell, boundaries, k_point,
                           pml_spec, (), 20260811, prefer_gpu=False,
                           zero_init=True)
    try:
        phases = tuple(driver.grid.bloch_phase(axis) for axis in range(3))
        got_phased = tuple(0 if p is None else 1 for p in phases)
        if got_phased != tuple(expected_phased):
            raise AssertionError(f"{name}: grid phased flags {got_phased} != "
                                 f"expected {expected_phased}")
        census_rows: List[Dict[str, Any]] = []
        separation_step = None
        started = time.time()
        for step in range(1, steps + 1):
            driver.step()
            control.step()
            census = census_point(_NumpyAsCupy, driver, checks)
            census_rows.append(census)
            if separation_step is None:
                for primary in PRIMARY_NAMES:
                    ours = np.ascontiguousarray(
                        getattr(driver.fields, primary)).view(np.uint32)
                    theirs = np.ascontiguousarray(
                        getattr(control.fields, primary)).view(np.uint32)
                    if not np.array_equal(ours, theirs):
                        separation_step = step
                        break
            log(f"[zero-init] {name} step {step}/{steps} "
                f"neg0={census['neg_zero_words']} "
                f"disc={census['discriminating_cells']} "
                f"quiet={census['quiet_fraction']} "
                f"rot={census['rotate_events']} sep={separation_step} "
                f"({time.time() - started:.1f} s)")
        if separation_step is None:
            raise AssertionError(
                f"{name}: the source never separated the reference from its "
                f"control — the case is vacuous at level 1")
        assert_zero_init_nonvacuous(name, checks, census_rows)
        return {"case": name, "steps_run": steps,
                "separation_step": separation_step,
                "census_per_step": census_rows,
                "census_final": census_rows[-1], "ok": True}
    finally:
        driver.close()
        control.close()


def self_check(out_path: str) -> int:
    """Build every case's driver on NumPy and step the ARRAY PATH two steps;
    record case 6's constructor refusal; run the zero-init rows' FULL-budget
    reference leg with the signed-zero census and its non-vacuity asserts.
    The Triton/CuPy legs are skipped and the artifact says so explicitly —
    nothing here measures the kernels."""
    payload: Dict[str, Any] = {
        "mode": "self-check (laptop, NumPy array path only)",
        "triton_legs": "SKIPPED: no CUDA/triton on this host — the plan "
                       "construction, kernel launches, and per-step "
                       "bit-identity claims are NOT exercised here",
        "cases": [],
    }
    save(payload, out_path)
    failures = 0
    for case in CASES:
        (name, dimensions, cell, boundaries, k_point, pml_spec, source_specs,
         steps, expected_phased, checks) = case
        started = time.time()
        row: Dict[str, Any] = {"case": name, "steps_run": 0,
                               "declared_budget": steps}
        if checks and checks.get("zero_init"):
            # The zero-init rows run their FULL budget here: the census IS the
            # reachability evidence, and its asserts are the case's own.
            try:
                row.update(zero_init_reference_leg(case))
            except Exception as exc:  # noqa: BLE001 - each failure recorded
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
                log(f"[self-check] {name}: FAILED {row['error']}")
            payload["cases"].append(row)
            save(payload, out_path)
            continue
        try:
            driver = build_driver(np, name, dimensions, cell, boundaries,
                                  k_point, pml_spec, source_specs, 20260811,
                                  prefer_gpu=False)
        except ValueError as exc:
            if expected_phased is None:
                row["skipped"] = str(exc)
                row["expected_refusal"] = True
                log(f"[self-check] {name}: EXPECTED grid refusal recorded "
                    f"({time.time() - started:.1f} s)")
            else:
                row["error"] = str(exc)
                failures += 1
                log(f"[self-check] {name}: UNEXPECTED refusal: {exc}")
            payload["cases"].append(row)
            save(payload, out_path)
            continue
        if expected_phased is None:
            row["error"] = ("Grid accepted the collapsed-axis k — "
                            "grid.py:980-991 changed; write the real case")
            failures += 1
            payload["cases"].append(row)
            save(payload, out_path)
            driver.close()
            continue
        try:
            inventory = sorted(state(driver))
            missing = REQUIRED_STATE - set(inventory)
            if missing:
                raise AssertionError(f"state inventory short: {sorted(missing)}")
            phases = tuple(driver.grid.bloch_phase(axis) for axis in range(3))
            got_phased = tuple(0 if p is None else 1 for p in phases)
            if got_phased != tuple(expected_phased):
                raise AssertionError(f"grid phased flags {got_phased} != "
                                     f"expected {expected_phased}")
            for _ in range(2):
                driver.step()
                row["steps_run"] += 1
            total = sum(float(np.abs(np.asarray(getattr(driver.fields, n))).sum())
                        for n in PRIMARY_NAMES)
            if not np.isfinite(total):
                raise AssertionError("array path produced non-finite state")
            row.update({"shape": [int(v) for v in driver.shape],
                        "phases": [repr(p) for p in phases],
                        "state_arrays": len(inventory),
                        "primary_l1_after_2_steps": total,
                        "ok": True})
            log(f"[self-check] {name}: shape={tuple(driver.shape)} "
                f"phased={got_phased} 2 array-path steps ok "
                f"({time.time() - started:.1f} s)")
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {name}: FAILED {row['error']}")
        finally:
            driver.close()
        payload["cases"].append(row)
        save(payload, out_path)
    payload["summary"] = {
        "built": sum(1 for c in payload["cases"] if c.get("ok")),
        "expected_refusals": sum(1 for c in payload["cases"]
                                 if c.get("expected_refusal")),
        "zero_init_reference_legs": sum(
            1 for c in payload["cases"] if c.get("census_final")),
        "failures": failures,
        "status": "passed" if failures == 0 else "FAILED",
    }
    save(payload, out_path)
    log(f"[self-check] {payload['summary']}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-check", action="store_true",
                        help="laptop mode: build the cases on NumPy, no CUDA")
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.out)

    try:
        import cupy as cp  # noqa: PLC0415
        import triton  # noqa: PLC0415, F401
    except ImportError as exc:
        log(f"SKIPPED cleanly: this probe's measurement legs need CUDA "
            f"(cupy + triton); this host has neither usable: {exc}. "
            f"Run --self-check for the laptop-safe case-builder validation.")
        return 75

    from meep_gpu import backends  # noqa: PLC0415

    # Strip FIRST, guard second: the guard wraps whatever is installed at the
    # seam, so both apply (the demonstrated composition order).
    # Raises at startup on a cache dir that could mix policies.
    gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    gate.write_provenance(out_dir)

    policy_problems = gate.ftz_strip_license_reasons()
    if policy_problems:
        payload = {"environment": environment(cp),
                   "subnormal_policy": gate.policy_stamp("cupy"),
                   "summary": {"status": "REFUSED",
                               "certified_under_subnormal_policy": None,
                               "reason": "subnormal policy: "
                                         + "; ".join(policy_problems)}}
        save(payload, args.out)
        log(f"SUBNORMAL POLICY REFUSAL: {policy_problems}")
        return 1
    payload: Dict[str, Any] = {
        "environment": environment(cp),
        "subnormal_policy": gate.policy_stamp("cupy"),  # refreshed at the end
        "step_budget": {case[0]: case[7] for case in CASES},
        "step_budget_note": "bit-identity is claimed per case for exactly the "
                            "stated budget and no further",
        "cases": [],
    }
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case[0]}: k={case[4]} pml={case[5]} "
            f"sources={[d['component'] for d in case[6]]}")
        payload["cases"].append(run_case(cp, *case))
        save(payload, args.out)
    payload["armed_mutation"] = run_armed_mutation(cp)
    save(payload, args.out)
    payload["armed_negation_mutation"] = run_armed_negation_mutation(cp)
    save(payload, args.out)

    # Final policy check with every case's compiles counted: a run whose strip
    # went unexercised (fresh cache, zero compiles) certified nothing.
    payload["subnormal_policy"] = gate.policy_stamp("cupy")
    residual = gate.ftz_strip_license_reasons()
    if residual:
        payload["summary"] = {"status": "FAILED",
                              "reason": "subnormal policy: "
                                        + "; ".join(residual)}
        save(payload, args.out)
        log(f"SUBNORMAL POLICY FAILURE AT CLOSE: {residual}")
        return 1

    ran = [row for row in payload["cases"] if not row.get("skipped")]
    total_steps = sum(row["steps"] for row in ran)
    # The n/n counters below are FORMAT, not authority: every row in `ran` has
    # already passed (run_case raises on the first non-identical step), so the
    # arithmetic is self-referential by construction. The artifact's pass
    # authority is the process exit code plus the per-step rows.
    payload["summary"] = {
        "status": "passed",
        "certified_under_subnormal_policy":
            payload["subnormal_policy"].get("policy"),
        "cases_exact": f"{len(ran)}/{len(ran)}",
        "expected_refusals_recorded": sum(
            1 for row in payload["cases"] if row.get("skipped")),
        "complete_steps_exact": f"{total_steps}/{total_steps}",
        "source_cases_nonvacuous": f"{sum(1 for r in ran if r['sources'])}/"
                                   f"{sum(1 for r in ran if r['sources'])}",
        "phase_none_skip_pinned": "complex_k0 family ran with phased=(0,0,0) "
                                  "on imag -0.0 seeds",
        "zero_init_cases_nonvacuous": "{0}/{0}".format(
            sum(1 for r in ran if r.get("census_final"))),
        "zero_init_final_neg_zero_words": {
            r["case"]: r["census_final"]["neg_zero_words"]
            for r in ran if r.get("census_final")},
        "armed_mutation_caught_at_step":
            payload["armed_mutation"]["caught_at_step"],
        "armed_negation_mutation":
            payload["armed_negation_mutation"].get(
                "caught_at_step",
                payload["armed_negation_mutation"].get("skipped")),
    }
    save(payload, args.out)
    log(f"COMPLEX COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
