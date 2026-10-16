"""Byte gate for the Metal special_kz (``grid.beta``) PML curls, real and complex.

THE CLAIM, and the only one: **byte-identity to ``stepping.py`` on this host,
subject to a declared and CHECKED subnormal-free precondition** (leg
``precondition``). Not a stated tolerance — the justification is a measured cliff,
identity through 1e-30 and total divergence at 1e-38, so a tolerance would be
either meaninglessly loose or would falsely admit the divergent case.

THE ORACLE IS IN PROCESS. ``stepping.step_B(fields, pml)`` and the Metal launch run
in ONE process against ONE seed on real ``Grid``/``Fields``/``PML`` objects, and the
array path IS the reference: it steps MEEP's per-chunk forms and runs the beta
coupling as MEEP's own pass after every component's curl (``_apply_folded_increment``),
which the kernels transcribe in register (:mod:`meep_gpu.metal_kernels.folded_pass`).
There is no in-gate transcription: every mutation row compares the mutant with
``stepping`` itself, so a defect shared between a harness reference and the kernel
cannot make a mutation row a comparison of one wrong answer with another.

THE LEGS, and what each is FOR beyond the standard discipline:

  reference      both kernels vs stepping.py: both corpus beta SIGNS, Courants
                 0.34/0.4375/0.35, the Brillouin-edge phase EXACTLY -1+0j, a
                 metallic axis, 4 cycles per case
  identity       THE HAS_BETA ARM: a beta-compiled-out build must be byte-identical
                 to the CERTIFIED plain curl on the same seeds. Without it "the
                 beta kernel works" could mean "the beta kernel is a different
                 kernel that happens to agree"
  constitutive   BOTH arms' companions LAUNCHED, not merely admitted: the certified
                 real and complex constitutive kernels under restated predicates,
                 against stepping.py, four cycles, isolated from the curls. The
                 complex half was a predicted null until 2026-08-19
  guard          the contraction directive, MEASURED, PER ARM: the contract=fast
                 build must diverge on both storages (the pass's multiply-subtract
                 shapes and the forms' are contractable in both)
  signed_zero    +-0 lattice seeding with a census floor (the beta term is a pure
                 scalar-by-field product, so +-0 partners flow straight through it)
  precondition   the subnormal census over BOTH storage families, with the beta
                 term's own intermediate reconstructed per storage, AND the scaled
                 control that makes the window fire
  mutations      armed, launch-counted, three-valued must_catch, every row against
                 stepping.py: the beta pass's own defects (dropped, signs, partner,
                 position folded back into the curl, the mask, the ladder's dsig
                 and dsigu stages, fu leaking onto a plain cell), the per-chunk
                 form battery of the certified curl the source is derived from
                 (KERNELS.md 7.2), the two record mutations, and the host's own
                 choices (sub-lattice, conjugation, the coefficient's dt)
  refusals       every named refusal, including the two the ENGINE ITSELF raises
                 (offdiag+beta, beta off 2-D) rather than only the predicate's.
                 A configuration this leg cannot CONSTRUCT is a hard failure, never
                 a note: the offdiag block once caught its own AttributeError and
                 skipped both assertions while the summary claimed them measured
  engine         engine-route plans + the disjointness sweep against every arm in
                 the tree, in both directions, and WHICH arm the shipped composer
                 selects per slot
  shipped        THE COMPOSER'S OWN PLAN against stepping.py, 4 cycles. Added when
                 the family was WIRED into `launch.plan_step`: that made a third
                 route to a beta kernel user-reachable, and `engine` checks only
                 WHICH arm it picks, not what the picked plan computes

WHAT IS COMPARED: uint32 word equality on float32 and complex64 storage, never
``allclose``. The compared arrays are the sub-step's TARGETS **and** its
AUXILIARIES (``fu_*``) — the auxiliaries are STATE, so a kernel that is right for
one launch and wrong forever after only diverges in a multi-cycle leg.

NaN AND OVERFLOW NEEDLES ARE EXCLUDED, on this track as on Triton's: NaN sign and
payload are IEEE-unspecified and measure the toolchain's mood.

STATED WEAKNESS, unchanged from tranche 1 and repeated because a claim inherits
it: there is NO PTX-EQUIVALENT AUDIT on this executor. ``compile_shader`` exposes
no disassembly, so the mutation legs and this gate are the only arbiters.

WHAT THIS GATE DOES NOT RUN, and where it is run instead. Every leg here launches a
sub-step in ISOLATION from a seeded state. It never calls ``FdtdDriver.step()``, so
the driver's interleave — sources injecting at both seams, the metallic wall passes,
the symmetry and folded-ghost fills, all of it array-path work between the device
sub-steps — is outside this artifact's scope, and so is any defect that only
compounds over a budget. ``probe_metal_special_kz_driver_step`` is where that is
measured: complete driver steps for a stated budget, uint32 over the full enumerated
inventory, with the dispatch accounted so a silent fall-back to the array path
cannot pass as agreement. The family's artifact set is gate.json + composition.json +
driver_step.json, and a reader who has only this file has the
sub-steps without the run.

    python -u gate_metal_special_kz.py \\
        --out results/metal_special_kz_2026-08-15/gate.json
"""

from __future__ import annotations

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

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep. Request flush EXPLICITLY, before anything
# resolves a policy, and stamp the resolution into the artifact: the claim is only
# as good as the precondition it was certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, special_kz  # noqa: E402
from meep_gpu.metal_kernels import subnormal, templates  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from gate_chunk_records import layer_record  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing
words = kit.words

#: THE VERDICT READS BOTH A COUNT AND AN OUTCOME. `compared` is how many
#: comparisons ran and `certified` is whether they all agreed; a gate that stamped
#: `passed` from either alone would pass a round in which every comparison failed,
#: or one in which nothing was compared. Both are accumulated here and handed to
#: `kit.summarize`.
TALLY = {"compared": 0, "disagreed": 0}


def compare(got: Dict[str, Any], oracle: Dict[str, Any],
            names: Sequence[str]) -> Dict[str, int]:
    """Per-array differing WORDS, counted into the run-level tally."""
    out = {name: differing(got[name], oracle[name]) for name in names}
    TALLY["compared"] += len(names)
    bad = {name: count for name, count in out.items() if count}
    TALLY["disagreed"] += len(bad)
    return bad

SEED = 20260815
CYCLES = 4

#: The corpus anchors' own numbers.
BETA_KZ2D = 0.3321611318837033        # refl-angular-kz2d.py
BETA_GRATING = -0.6850526103319672    # binary_grating special_kz
BETA_SPECIAL_KZ = -0.3907             # test_special_kz.test_special_kz
KX_SPECIAL_KZ = 0.9205                # ...its in-plane Bloch component

#: NON-POWER-OF-TWO FIRST, and three of them: the beta coefficient is
#: `2*pi*beta*dt` and dt rides the Courant, so an odd Courant is what makes the
#: coefficient a non-representable multiplicand.
CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "r1_kz2d_periodic", "beta": BETA_KZ2D, "courant": 0.34,
     "complex": False, "k": (0.0, 0.0, 0.0), "boundaries": "periodic"},
    {"name": "r2_metallic_x_negative_beta", "beta": BETA_GRATING,
     "courant": 0.4375, "complex": False, "k": (0.0, 0.0, 0.0),
     "boundaries": ("metallic", "periodic", "periodic")},
    {"name": "r3_kz2d_courant_half", "beta": BETA_KZ2D, "courant": 0.5,
     "complex": False, "k": (0.0, 0.0, 0.0), "boundaries": "periodic"},
    {"name": "c1_complex_k0", "beta": 0.2, "courant": 0.35, "complex": True,
     "k": (0.0, 0.0, 0.0), "boundaries": "periodic"},
    {"name": "c2_marquee_inplane_bloch", "beta": BETA_SPECIAL_KZ, "courant": 0.35,
     "complex": True, "k": (KX_SPECIAL_KZ, 0.0, 0.0), "boundaries": "periodic"},
    # The Brillouin edge: resolution 10 x Lx 2.0 -> 20 cells, k = 0.25 makes the
    # x phase EXACTLY -1+0j, which is the one phase whose imaginary word is a
    # signed zero and whose real word is exact.
    {"name": "c3_brillouin_edge_metallic_y", "beta": 0.2, "courant": 0.35,
     "complex": True, "k": (0.25, 0.0, 0.0),
     "boundaries": ("periodic", "metallic", "periodic")},
    # THE TWO ROWS BELOW WERE ADDED BECAUSE THE MATRIX ABOVE LEFT THE SECOND
    # IN-PLANE AXIS UNMEASURED ON BOTH ARMS, and the enumeration had drifted with
    # it (`special_kz._emittable_bloch_specialisations`).
    #
    # r4: a metallic Y. Every real case above walls X or nothing, so `curl1` — the
    # target the `beta_minus` term rides — never met the ownership mask in step_B,
    # and the `ghost`/`ownership_mask` emitters were exercised on one axis only.
    # r4 is also the real arm's (0,1,0) boundary specialisation, which no gate case
    # compiled.
    {"name": "r4_metallic_y_kz2d", "beta": BETA_KZ2D, "courant": 0.34,
     "complex": False, "k": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "metallic", "periodic")},
    # c4: TWO LIVE BLOCH PHASES, both general (neither 1+0j nor -1+0j). Every
    # complex case above rotates at most ONE wrapped plane, so `_phase_lines`'
    # per-axis operand table (x -> b_x/c_x, y -> a_y/c_y) was only ever half
    # exercised, and the guard leg's NAIVE control had no row where two general
    # complex products are live at once. Courant 0.4375 also gives the complex arm
    # its first non-0.35 dt.
    {"name": "c4_two_live_phases", "beta": BETA_SPECIAL_KZ, "courant": 0.4375,
     "complex": True, "k": (KX_SPECIAL_KZ, 0.31, 0.0), "boundaries": "periodic"},
)


def case_named(name: str) -> Dict[str, Any]:
    """One case BY NAME.

    THE LEGS USED TO INDEX ``CASES`` POSITIONALLY — ``CASES[1]`` was the mutation
    substrate, ``CASES[4]`` the complex one, ``CASES[0]``/``CASES[3]`` the refusals'
    fixtures. Inserting a row anywhere but the end silently re-points every one of
    them, and the artifact still records a green mutation leg against whatever case
    moved into the slot. A name cannot slide.
    """
    for case in CASES:
        if case["name"] == name:
            return case
    raise KeyError(f"{name!r} is not one of {[c['name'] for c in CASES]}")


#: Configurations where the ownership mask is LIVE, which is what makes the beta
#: term's POSITION (before the mask) byte-visible. PINNED BY NAME: on an all
#: periodic grid nothing is masked, the position mutation is byte-invisible, and a
#: position leg run only there would pass while measuring nothing.
MASKED_CASES: Tuple[str, ...] = ("r2_metallic_x_negative_beta",
                                 "c3_brillouin_edge_metallic_y",
                                 "r4_metallic_y_kz2d")

#: EVERY volume a complete step reads or writes, the constitutive auxiliaries
#: INCLUDED. They are not read by either curl, and leaving them out cost the
#: composition probe an afternoon: `restore` then rewound only part of the state,
#: so the array-path oracle for a WHOLE step started from the device run's `f_w_*`
#: and diverged for a reason that had nothing to do with the kernels.
STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

SUB_STEPS = special_kz.SUB_STEPS
PERIODIC, METALLIC = templates.PERIODIC, templates.METALLIC


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any], seed: int = SEED, scale: float = 1.0,
          signed_zeros: bool = True, beta: Optional[float] = None,
          offdiagonal: bool = False):
    """A real Grid/Fields/PML triple for one beta case, seeded in the physical band.

    ``signed_zeros`` seeds a ±0 LATTICE into every volume. MEEP keeps float32
    subnormals on arm64 (``set_zero_subnormals`` is a no-op under
    ``#if HAVE_IMMINTRIN_H``), so the signed-zero class is constructible on the
    HOST side here — and the beta term is a pure scalar-by-field product, so a ±0
    partner flows straight through it into the recurrence.

    ``offdiagonal`` installs a chi1inv row through the ENGINE'S OWN API
    (``Fields.set_epsilon_volumes(..., chi1inv_offdiagonal=...)``) — the refusals
    leg needs the configuration that stepping.py:800-810 raises on, and it has to
    be the real one.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0),
                boundaries=case["boundaries"], dimensions=2,
                courant=case["courant"], k_point=case["k"],
                beta=case["beta"] if beta is None else beta, xp=np)
    fields = Fields(grid=grid, force_complex_fields=bool(case["complex"]))
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    if offdiagonal:
        fields.set_epsilon_volumes(
            {component: epsilon for component in ("Ex", "Ey", "Ez")},
            {component: inverse for component in ("Ex", "Ey", "Ez")},
            chi1inv_offdiagonal={"Ex": {"Ey": np.full(grid.shape, 0.05,
                                                      dtype=np.float32)}})
    else:
        fields.set_isotropic_epsilon_volume(epsilon, inverse)
    pml = PML(grid=grid, thickness=tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                                         for a in range(3)))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape) * (0.37 * scale)).astype(np.float32)
        if signed_zeros:
            host.reshape(-1)[::17] = np.float32(-0.0)
            host.reshape(-1)[7::23] = np.float32(0.0)
        if case["complex"]:
            imag = (rng.standard_normal(grid.shape) * (0.29 * scale)).astype(np.float32)
            if signed_zeros:
                imag.reshape(-1)[3::19] = np.float32(-0.0)
            array[...] = _complex_of(host, imag)
        else:
            array[...] = host
    # A SEEDED STATE IS A MID-RUN ONE: H was started from B long ago, so
    # ``stepping.separate_h_from_b`` must not run on it again -- the array path would
    # copy B into H and f_w on the dsigw cells at the first update_H, and no kernel
    # this gate launches does (``launch.h_start_reasons``).
    fields.h_separated = True
    return grid, fields, pml


def _complex_of(real: Any, imag: Any) -> Any:
    """Build complex64 from two planes WITHOUT losing a signed zero.

    ``real + 1j*imag`` multiplies through complex128 and canonicalizes ``-0.0`` on
    the imaginary half — measured while writing the expansion probe, where it
    silently deleted three of four signed-zero needles. Writing the words is the
    only spelling that keeps them, and the signed-zero leg's census would be
    vacuous without it.
    """
    out = np.zeros(real.shape, dtype=np.complex64)
    view = out.reshape(-1).view(np.float32).reshape(-1, 2)
    view[:, 0] = real.reshape(-1)
    view[:, 1] = imag.reshape(-1)
    return out


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


def boundary_codes(grid, pml) -> Tuple[int, int, int]:
    return tuple(METALLIC if kind == "metallic" else PERIODIC
                 for kind in stepping._boundary_kinds(grid, pml))


def curl_coefficients(pml, suffix: str) -> Dict[str, Any]:
    """The PML vectors AS THE ARRAY PATH HOLDS THEM — broadcast-shaped per axis.

    The reference transcription applies them with ``*=``, which needs the (nx,1,1)
    / (1,ny,1) / (1,1,nz) shapes; flattening them here would broadcast every axis
    along the last one and step a different absorber. The device path takes
    :func:`flat_coefficients` instead, because the kernel indexes them by i/j/k.
    """
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def flat_coefficients(pml, suffix: str) -> Dict[str, Any]:
    """The same vectors flattened for the kernel's own indexing."""
    return {key: np.asarray(value).reshape(-1).astype(np.float32)
            for key, value in curl_coefficients(pml, suffix).items()}


def oracle_after(fields, pml, base: Dict[str, Any], sub_step: str) -> Dict[str, Any]:
    """``stepping``'s own answer for one sub-step from ``base``; ``fields`` ends at ``base``.

    The array path is the aligned oracle: its first PML sub-step installs the one-rank
    region plan the device legs bind as their chunk record (:func:`layer_record`).
    """
    restore(fields, base)
    getattr(stepping, sub_step)(fields, pml)
    oracle = snapshot(fields)
    restore(fields, base)
    return oracle


def _reference_arrays(state: Dict[str, Any], sub_step: str):
    spec = SUB_STEPS[sub_step]
    arrays = {n: np.array(state[n], copy=True)
              for n in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: np.array(state["fu_" + n], copy=True)
                   for n in spec["targets"]})
    return arrays


def phase_table(grid, pml) -> Tuple[Optional[complex], ...]:
    """The per-axis ``eikna`` table the complex plans choose their connection from."""
    from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

    return complex_fields.bloch_phase_table(grid, stepping._boundary_kinds(grid, pml))


# ---------------------------------------------------------------------------
# Launching, through the SHIPPED plan objects
# ---------------------------------------------------------------------------

def run_on_device(state: Dict[str, Any], grid, pml, sub_step: str, codes,
                  complex_storage: bool,
                  source: Optional[str] = None,
                  contract: str = templates.CONTRACT_OFF,
                  has_beta: bool = True,
                  beta_words: Any = None,
                  suffix: Optional[str] = None,
                  conjugate: bool = True,
                  record: Any = None,
                  counters: Optional[List[kit.Counter]] = None) -> Dict[str, Any]:
    """One beta curl launch through the shipped plan, returning the host results.

    Everything goes through ``plan_*_from_arrays -> KernelPlan.run`` so the bytes
    this gate certifies are the bytes the engine route would launch. ``source`` is
    the MUTATION SEAM; dropping it would silently disarm every mutation leg.
    ``record`` replaces the one-rank chunk record (a record mutation);
    ``conjugate=False`` hands the plan the conjugated phase table, so the D side
    connects with the B side's factor (a host mutation); ``beta_words`` replaces the
    host-rounded ``(plus, minus)`` words.
    """
    spec = SUB_STEPS[sub_step]
    arrays = _reference_arrays(state, sub_step)
    flat = flat_coefficients(pml, spec["suffix"] if suffix is None else suffix)
    residency = device.Residency()
    functions = None
    entry = ("beta_bloch_pml_curl_step" if complex_storage
             else "beta_pml_curl_step")
    chunk = layer_record(grid, pml) if record is None else record
    if beta_words is None:
        beta_words = special_kz.beta_curl_coefficients(
            grid.beta, grid.dt, magnetic=(sub_step == "step_B"),
            complex_storage=complex_storage)

    if complex_storage:
        from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

        phases = phase_table(grid, pml)
        if not conjugate:
            phases = tuple(None if p is None else complex(np.conj(p)) for p in phases)
        if source is not None or contract != templates.CONTRACT_OFF:
            phased, _values = complex_fields.phase_arguments(
                phases, backward=bool(spec["backward"]))
            text = source if source is not None else special_kz.beta_bloch_curl_source(
                codes, spec["backward"], phased, has_beta, contract)
            function = getattr(device.compile_source(text), entry)
            if counters is not None:
                function = kit.Counter(function)
                counters.append(function)
            functions = {templates.CONTRACT_OFF: function}
        plan = special_kz.plan_beta_bloch_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phases, float(grid.dt / grid.dx),
            beta_words[0], beta_words[1], residency,
            functions=functions, has_beta=has_beta, chunk_bounds=chunk)
    else:
        if source is not None or contract != templates.CONTRACT_OFF:
            text = source if source is not None else special_kz.beta_curl_source(
                codes, spec["backward"], has_beta, contract)
            function = getattr(device.compile_source(text), entry)
            if counters is not None:
                function = kit.Counter(function)
                counters.append(function)
            functions = {templates.CONTRACT_OFF: function}
        plan = special_kz.plan_beta_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, float(grid.dt / grid.dx),
            beta_words[0], beta_words[1], residency, functions=functions,
            has_beta=has_beta, chunk_bounds=chunk)
    plan.run()
    residency.sync_out()
    if counters is None and plan.launches != 1:
        raise AssertionError(f"the plan launched {plan.launches} times, not once")
    return arrays


def compared_names(sub_step: str) -> Tuple[str, ...]:
    spec = SUB_STEPS[sub_step]
    return tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])


def apply_to_fields(fields, arrays: Dict[str, Any], sub_step: str) -> None:
    for name in compared_names(sub_step):
        getattr(fields, name)[...] = arrays[name]


# ---------------------------------------------------------------------------
# LEG reference — both kernels vs stepping.py
# ---------------------------------------------------------------------------

def leg_reference(payload: Dict[str, Any], out: str) -> None:
    """The shipped kernels against ``stepping.py`` on real objects, 4 cycles.

    The DEVICE result is fed back into ``fields`` each cycle, so cycle N+1 starts
    from the device's own state: a per-launch-correct kernel that drifts is only
    caught this way.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        codes = boundary_codes(grid, pml)
        for cycle in range(CYCLES):
            for sub_step in SUB_STEPS:
                base = snapshot(fields)
                getattr(stepping, sub_step)(fields, pml)
                oracle = snapshot(fields)
                restore(fields, base)
                got = run_on_device(base, grid, pml, sub_step, codes,
                                    bool(case["complex"]))
                names = compared_names(sub_step)
                moved = sum(differing(oracle[n], base[n]) for n in names)
                bad = compare(got, oracle, names)
                rows.append({"case": case["name"], "cycle": cycle,
                             "sub_step": sub_step, "complex": case["complex"],
                             "courant": case["courant"], "beta": case["beta"],
                             "codes": list(codes), "moved": moved,
                             "differing": bad})
                payload["legs"]["reference"] = rows
                save(payload, out)
                kit.assert_moved(moved, f"{case['name']}/{sub_step}")
                assert not bad, rows[-1]
                # The device's own answer becomes the next cycle's input.
                apply_to_fields(fields, got, sub_step)
        log(f"[reference] {case['name']:<28} {CYCLES}x2 IDENTICAL "
            f"({time.time() - started:.1f}s)")


# ---------------------------------------------------------------------------
# LEG identity — the HAS_BETA arm against the CERTIFIED plain curl
# ---------------------------------------------------------------------------

def leg_identity(payload: Dict[str, Any], out: str) -> None:
    """A beta-compiled-out build must be the CERTIFIED curl, byte for byte.

    Two halves and both matter:

    1. the REAL arm's ``HAS_BETA=0`` build against ``shaders.curl_source`` — the
       kernel tranche 1 certified — launched on the same seeds through the
       certified plan. This is what makes "the beta kernel adds a term" a
       measurement rather than a description: everything ELSE in the body is
       pinned to already-certified bytes;
    2. the same build against ``stepping.py`` on a beta = 0 run, the oracle either
       way. The COMPLEX arm's certified-kernel comparison is against
       ``complex_fields``' Bloch curl, which its source is derived from.

    NON-VACUITY: the beta term must MATTER. Each case also runs the HAS_BETA=1
    build on the same state and asserts it DIFFERS from the HAS_BETA=0 one — a
    "beta kernel" that agreed with the plain curl would pass every other leg while
    the feature was dark.
    """
    from meep_gpu.metal_kernels import launch as metal_launch

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        codes = boundary_codes(grid, pml)
        for sub_step in SUB_STEPS:
            spec = SUB_STEPS[sub_step]
            base = snapshot(fields)
            names = compared_names(sub_step)

            # --- half 2: beta compiled out vs stepping.py on a beta = 0 run ----
            zero_case = dict(case, beta=0.0)
            zgrid, zfields, zpml = build(zero_case)
            zbase = snapshot(zfields)
            getattr(stepping, sub_step)(zfields, zpml)
            zoracle = snapshot(zfields)
            zcodes = boundary_codes(zgrid, zpml)
            plain = run_on_device(zbase, zgrid, zpml, sub_step, zcodes,
                                  bool(case["complex"]), has_beta=False,
                                  beta_words=(0.0, 0.0))
            against_stepping = compare(plain, zoracle, names)

            # --- half 1: against the CERTIFIED curl the source is derived from ---
            arrays = _reference_arrays(zbase, sub_step)
            flat = flat_coefficients(zpml, spec["suffix"])
            residency = device.Residency()
            if case["complex"]:
                from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

                certified = complex_fields.plan_complex_pml_curl_from_arrays(
                    sub_step, arrays, flat, zcodes, phase_table(zgrid, zpml),
                    float(zgrid.dt / zgrid.dx), residency,
                    chunk_bounds=layer_record(zgrid, zpml))
            else:
                certified = metal_launch.plan_from_arrays(
                    sub_step, arrays, flat, zcodes,
                    float(zgrid.dt / zgrid.dx), residency,
                    chunk_bounds=layer_record(zgrid, zpml))
            certified.run()
            residency.sync_out()
            against_certified = compare(plain, arrays, names)

            # --- non-vacuity: the term must MOVE something ----------------------
            with_beta = run_on_device(base, grid, pml, sub_step, codes,
                                      bool(case["complex"]), has_beta=True)
            without_beta = run_on_device(base, grid, pml, sub_step, codes,
                                         bool(case["complex"]), has_beta=False)
            term_words = sum(differing(with_beta[n], without_beta[n])
                             for n in names)

            row = {"case": case["name"], "sub_step": sub_step,
                   "complex": case["complex"],
                   "beta0_vs_stepping": against_stepping,
                   "beta0_vs_certified_kernel": against_certified,
                   "beta_term_words": term_words}
            rows.append(row)
            payload["legs"]["identity"] = rows
            save(payload, out)
            log(f"[identity] {case['name']:<28} {sub_step} beta0-vs-stepping="
                f"{'IDENTICAL' if not against_stepping else against_stepping} "
                f"beta0-vs-certified="
                f"{'IDENTICAL' if not against_certified else against_certified} "
                f"beta_words={term_words} ({time.time() - started:.1f}s)")
            assert not against_stepping, row
            assert not against_certified, row
            kit.assert_census_floor(
                term_words, f"{case['name']}/{sub_step} beta term effect (the "
                            f"HAS_BETA=1 and HAS_BETA=0 builds agreed: the term "
                            f"is DARK and every other leg would pass while "
                            f"measuring nothing)")


# ---------------------------------------------------------------------------
# LEG constitutive — the REAL arm's companion, LAUNCHED rather than admitted
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """Both constitutive companions against ``stepping.py`` on a beta run.

    The refusals leg records that :func:`beta_run_constitutive_coverage` and
    :func:`beta_run_complex_constitutive_coverage` ADMIT a beta run of their own
    storage. That is a claim about a PREDICATE. The module's stronger claim — that
    nothing in the constitutive sub-step is beta-dependent, so the CERTIFIED kernel
    steps a beta run unchanged — is a claim about BYTES, and a predicate verdict
    cannot carry it. The composition probe exercises the plans inside a four-sub-step
    comparison, where a constitutive defect and a curl defect would be
    indistinguishable; this leg isolates it.

    THE COMPLEX HALF USED TO BE A PREDICTED NULL HERE, because the complex
    constitutive companion did not exist and the leg recorded the refusal instead of
    measuring anything. It exists as of 2026-08-19 —
    ``complex_fields.ComplexConstitutivePlan`` under a one-clause restatement — so
    every case in :data:`CASES` is now LAUNCHED, and the four complex ones are what
    close the census's last two slots.

    Multi-cycle, because ``f_w_*`` are STATE: a constitutive plan that is right on
    its first launch and wrong forever after only diverges on the second.
    """
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        residency = device.Residency()
        plans = {}
        for slot, side in (("update_H", "H"), ("update_E", "E")):
            plans[slot] = (
                special_kz.plan_beta_run_complex_constitutive(
                    fields, pml, side, residency,
                    chunk_bounds=layer_record(fields.grid, pml))
                if case["complex"] else
                special_kz.plan_beta_run_constitutive(fields, pml, side, residency,
                                                      chunk_bounds=layer_record(fields.grid, pml)))
            assert plans[slot] is not None, (
                f"{case['name']}/{slot}: the restated predicate ADMITS this beta "
                f"run but the plan builder refused it")
        for cycle in range(CYCLES):
            # THE CURL RUNS BETWEEN THE CONSTITUTIVE COMPARISONS, on the array path.
            # Without it the second launch of update_H reads the state the first
            # one produced, the recurrence walks toward its own fixed point, and by
            # cycle three the comparison is between two nearly-still arrays. The
            # beta curl is what makes each cycle's constitutive input fresh — and
            # it is exactly the sub-step whose output this companion has to consume.
            for slot, side, curl in (("update_H", "H", "step_B"),
                                     ("update_E", "E", "step_D")):
                getattr(stepping, curl)(fields, pml)
                residency.sync_in()
                spec = CONSTITUTIVE_SIDES[side]
                names = tuple(spec["targets"]) + tuple(spec["aux"])
                base = snapshot(fields)
                plans[slot].run()
                residency.sync_out()
                got = snapshot(fields)
                restore(fields, base)
                getattr(stepping, slot)(fields, pml)
                oracle = snapshot(fields)
                moved = sum(differing(oracle[n], base[n]) for n in names)
                bad = compare(got, oracle, names)
                rows.append({"case": case["name"], "cycle": cycle, "slot": slot,
                             "storage": "complex" if case["complex"] else "real",
                             "curl_run_first": curl, "beta": case["beta"],
                             "moved": moved, "differing": bad})
                payload["legs"]["constitutive"] = rows
                save(payload, out)
                kit.assert_moved(moved, f"{case['name']}/{slot}/cycle{cycle}")
                assert not bad, rows[-1]
                # The DEVICE's own answer carries into the next cycle, so the
                # mirrors and the host stay the one state the next launch reads.
                for name in names:
                    getattr(fields, name)[...] = got[name]
                residency.sync_in()
        log(f"[constitutive] {case['name']:<28} {CYCLES}x2 IDENTICAL "
            f"({time.time() - started:.1f}s)")


# ---------------------------------------------------------------------------
# LEG guard — the contraction directive, measured
# ---------------------------------------------------------------------------

def leg_guard(payload: Dict[str, Any], out: str) -> None:
    """``contract(fast)`` must DIVERGE, on each storage. A guard nobody can measure
    guards nothing.

    On Metal this is a SOURCE variant rather than a launch keyword, so the two modes
    are two compiled kernels and the plan holds one per mode. Both arms carry
    contractable shapes: the forms' ``(p * km) - curl`` and the pass's
    ``v - (si * df)``, spelled per part on the complex arm. THE FLOOR IS PER ARM: a
    single sum would be satisfied by one storage while the other sat at zero.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        codes = boundary_codes(grid, pml)
        for sub_step in SUB_STEPS:
            base = snapshot(fields)
            oracle = oracle_after(fields, pml, base, sub_step)
            guarded = run_on_device(base, grid, pml, sub_step, codes,
                                    bool(case["complex"]))
            fast = run_on_device(base, grid, pml, sub_step, codes,
                                 bool(case["complex"]), contract=templates.CONTRACT_FAST)
            names = compared_names(sub_step)
            guarded_bad = sum(differing(guarded[n], oracle[n]) for n in names)
            fast_bad = sum(differing(fast[n], oracle[n]) for n in names)
            row = {"case": case["name"], "sub_step": sub_step,
                   "arm": "complex" if case["complex"] else "real",
                   "guarded_differing": guarded_bad, "fast_differing": fast_bad}
            rows.append(row)
            payload["legs"]["guard"] = rows
            save(payload, out)
            log(f"[guard] {case['name']:<28} {sub_step} guarded={guarded_bad} "
                f"fast={fast_bad} ({time.time() - started:.1f}s)")
            assert guarded_bad == 0, row
    for storage in ("real", "complex"):
        kit.assert_census_floor(
            sum(row["fast_differing"] for row in rows if row["arm"] == storage),
            f"{storage.upper()}-arm contraction guard effect (the contract=fast "
            f"build agreed with stepping.py on every {storage} case, so the one "
            f"compile option this certification rests on is not measurable there)")


# ---------------------------------------------------------------------------
# LEG signed_zero
# ---------------------------------------------------------------------------

def _seed_signed_zero_lattice(fields, grid, complex_storage: bool) -> Dict[str, int]:
    """A DISTINCT ±0 lattice per volume, with the distinctness asserted.

    ``-0.0`` on one stride, a finite value on another, and both the strides and the
    magnitude keyed to the volume's position in :data:`STATE`. The strides stay
    coprime-ish so the finite overwrite never erases every ``-0.0`` — the census
    floor in the caller is what would catch it if it did — and the returned per
    volume ``-0`` counts go into the artifact.
    """
    counts: Dict[str, int] = {}
    seen: Dict[bytes, str] = {}
    for index, volume in enumerate(STATE):
        array = getattr(fields, volume, None)
        if array is None:
            continue
        lattice = np.zeros(grid.shape, dtype=np.float32)
        lattice.reshape(-1)[index % 3::3] = np.float32(-0.0)
        lattice.reshape(-1)[(index + 1) % 5::5] = np.float32(0.25 + 0.0625 * index)
        if complex_storage:
            imag = np.zeros(grid.shape, dtype=np.float32)
            imag.reshape(-1)[(index + 2) % 7::7] = np.float32(-0.0)
            imag.reshape(-1)[(index + 4) % 11::11] = np.float32(-0.125 - 0.03125 * index)
            array[...] = _complex_of(lattice, imag)
        else:
            array[...] = lattice
        counts[volume] = int(subnormal.signed_zero_census(
            np.asarray(getattr(fields, volume)))["negative_zero"])
        key = np.ascontiguousarray(getattr(fields, volume)).tobytes()
        assert key not in seen, (
            f"VACUOUS SEEDING: {volume} is byte-identical to {seen[key]}. With two "
            f"source volumes equal the beta term's two partners are the same words "
            f"and this leg cannot tell them apart")
        seen[key] = volume
        assert counts[volume] > 0, (
            f"VACUOUS census on the {volume} seeding: no -0.0 survived the finite "
            f"overwrite, so the class this leg exists to carry was never built")
    return counts


def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """A ±0 lattice through the beta term, census-floored.

    The beta term is a PURE SCALAR-BY-FIELD PRODUCT, so a ±0 partner flows through
    it unchanged into the recurrence — which makes this family's signed-zero class
    reachable from the seeding alone, unlike the constitutive sub-step (a fixed
    point of zero init). A census of zero is VACUOUS, not passed.

    THE LATTICE IS PER VOLUME, and it was not. Every volume used to be seeded from
    an identical expression, so ``Ex == Ey == Ez == Bx == ...`` held exactly and the
    leg could not tell the beta term's two partners apart at all — ``beta_plus * b``
    and ``beta_minus * a`` read the same words — nor could it see a source-pointer
    permutation, since g0, g1 and g2 held one array's values three times. The stride
    phases and the finite magnitude now vary with the volume, and
    :func:`_seed_signed_zero_lattice` asserts the volumes are actually distinct, so
    the leg's own non-degeneracy is measured rather than assumed.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        codes = boundary_codes(grid, pml)
        per_volume = _seed_signed_zero_lattice(fields, grid, bool(case["complex"]))
        seeded = snapshot(fields)
        for sub_step in SUB_STEPS:
            restore(fields, seeded)
            getattr(stepping, sub_step)(fields, pml)
            oracle = snapshot(fields)
            got = run_on_device(seeded, grid, pml, sub_step, codes,
                                bool(case["complex"]))
            names = compared_names(sub_step)
            moved = sum(differing(oracle[n], seeded[n]) for n in names)
            census_in = sum(
                subnormal.signed_zero_census(seeded[n])["negative_zero"]
                for n in names)
            census_out = sum(
                subnormal.signed_zero_census(oracle[n])["negative_zero"]
                for n in names)
            bad = compare(got, oracle, names)
            row = {"case": case["name"], "sub_step": sub_step,
                   "negative_zero_words_in": census_in,
                   "negative_zero_words_out": census_out,
                   "negative_zero_words_per_seeded_volume": per_volume,
                   "distinct_seeded_volumes": len(per_volume),
                   "moved": moved, "differing": bad}
            rows.append(row)
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            log(f"[signed_zero] {case['name']:<28} {sub_step} -0 in={census_in} "
                f"out={census_out} moved={moved} "
                f"{'IDENTICAL' if not bad else bad} "
                f"({time.time() - started:.1f}s)")
            assert not bad, row
            kit.assert_census_floor(census_in,
                                    f"{case['name']}/{sub_step} -0 seeding")
            kit.assert_moved(moved, f"{case['name']}/{sub_step}")


# ---------------------------------------------------------------------------
# LEG precondition — the census, and the control that makes it fire
# ---------------------------------------------------------------------------

SUBNORMAL_SCALES = (("physical", 1.0, True), ("small_normal", 1e-20, True),
                    ("subnormal_band", 1e-38, False))


#: Which cases the census runs over — ONE PER STORAGE FAMILY, and that is the
#: correction. The leg used to run the real metallic case alone, so the complex
#: arm's operands, results and beta intermediate were never censused at all while
#: the claim covered complex64 storage. The complex intermediate is the one that
#: differs in kind: a complex product forms FOUR sub-products, any of which can land
#: in the band while both stored words look ordinary.
PRECONDITION_CASES = (case_named("r2_metallic_x_negative_beta"),
                      case_named("c2_marquee_inplane_bloch"))


def _beta_intermediates(grid, sub_step: str, base: Dict[str, Any],
                        complex_storage: bool) -> Dict[str, Any]:
    """The beta pass's un-storable products, reconstructed on the host.

    The word times the UNSHIFTED centre partner, target 0 taking the second source
    and target 1 the first. Under complex storage the pass forms one product per part
    (the quarter turn), so each part's product is observed on its own.
    """
    spec = SUB_STEPS[sub_step]
    plus, minus = (np.float32(word) for word in special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(sub_step == "step_B"),
        complex_storage=complex_storage))
    second, first = base[spec["sources"][1]], base[spec["sources"][0]]
    if complex_storage:
        return {"beta_product_plus_re": plus * second.real,
                "beta_product_plus_im": plus * second.imag,
                "beta_product_minus_re": minus * first.real,
                "beta_product_minus_im": minus * first.imag}
    return {"beta_product_plus": plus * second,
            "beta_product_minus": minus * first}


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The census that BOUNDS the claim, through the shared window, plus its control.

    The window covers OPERANDS, RESULTS and the beta term's own INTERMEDIATE — the
    product ``c * partner``, reconstructed on the host. That intermediate is this
    family's specific reason to look: the coefficient is ``2*pi*beta*dt`` (order
    1e-2 on the corpus rows), so the product is SMALLER than its operand and a
    field near the bottom of the normal range can be pushed into the band by the
    term alone, with every stored array clean.

    BOTH STORAGE FAMILIES ARE CENSUSED (:data:`PRECONDITION_CASES`). A claim that
    names complex64 storage and censused only float32 would be stating a
    precondition it had checked on half its own scope.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in PRECONDITION_CASES:
        for label, scale, expect_clean in SUBNORMAL_SCALES:
            for sub_step in SUB_STEPS:
                grid, fields, pml = build(case, scale=scale, signed_zeros=False)
                codes = boundary_codes(grid, pml)
                base = snapshot(fields)
                getattr(stepping, sub_step)(fields, pml)
                oracle = snapshot(fields)
                restore(fields, base)
                got = run_on_device(base, grid, pml, sub_step, codes,
                                    bool(case["complex"]))
                names = compared_names(sub_step)
                spec = SUB_STEPS[sub_step]
                window = preconditions.SubnormalWindow(
                    0, 0, per_array_words=1, per_intermediate_words=1)
                for name in tuple(spec["sources"]) + names:
                    window.observe(f"operand:{name}", base[name])
                for name in names:
                    window.observe(f"result:{name}", oracle[name])
                for name, value in _beta_intermediates(
                        grid, sub_step, base, bool(case["complex"])).items():
                    window.observe_intermediate(name, value)
                report = window.report()
                bad = sum(differing(got[n], oracle[n]) for n in names)
                row = {"case": case["name"], "complex": case["complex"],
                       "scale": label, "factor": scale, "sub_step": sub_step,
                       "window": report, "differing_words": bad,
                       "gate_verdict": ("REFUSED (precondition)"
                                        if not report["clean"]
                                        else ("identical" if not bad
                                              else "DIFFERS"))}
                rows.append(row)
                payload["legs"]["precondition"] = rows
                save(payload, out)
                log(f"[precondition] {case['name']:<28} {label:<14} {sub_step} "
                    f"subnormal_words={report['subnormal_words']} "
                    f"observed={report['observed_words']} differing={bad} "
                    f"({time.time() - started:.1f}s)")
                context = f"{case['name']}/{label}/{sub_step}"
                # The two halves are DIFFERENT assertions and the module spells
                # them separately: a clean window is enforced by
                # assert_clean_or_refuse, and the scaled control must DEMONSTRATE
                # the window firing — a precondition never shown to fire is
                # decorative.
                if expect_clean:
                    preconditions.assert_clean_or_refuse(window, context)
                    assert bad == 0, row
                else:
                    preconditions.demonstrate_firing(window, context)
                    assert bad > 0, (
                        f"{context}: the scaled control was expected to DIVERGE "
                        f"and did not, so the cliff this claim rests on was not "
                        f"reproduced")


# ---------------------------------------------------------------------------
# LEG mutations
# ---------------------------------------------------------------------------

def _drop_pass(source: str) -> str:
    """The whole beta pass removed (its two marker lines and everything between)."""
    from meep_gpu.metal_kernels import folded_pass  # noqa: PLC0415

    start = source.find(folded_pass.FOLDED_PASS_BEGIN)
    end = source.find(folded_pass.FOLDED_PASS_END)
    if start < 0 or end < 0:
        raise LookupError("the folded pass block")
    end += len(folded_pass.FOLDED_PASS_END) + 1
    return source[:start] + "    // beta pass dropped\n" + source[end:]


def _fold_into_curl(source: str, complex_storage: bool) -> str:
    """The PREVIOUS package form: the term added to the curl before the mask and the
    forms, with no pass of its own -- what the array path did before MEEP's order."""
    if complex_storage:
        terms = ("    curl0 = curl0 - float2(-(beta_plus * b.y), beta_plus * b.x);\n"
                 "    curl1 = curl1 - float2(-(beta_minus * a.y), beta_minus * a.x);\n")
    else:
        terms = ("    curl0 = curl0 - (beta_plus * b);\n"
                 "    curl1 = curl1 - (beta_minus * a);\n")
    anchor = "    // --- ownership mask"
    return kit.needle(_drop_pass(source), anchor, terms + "\n" + anchor)


def _pass_unmasked(source: str) -> str:
    """The increment's ownership rows removed: MEEP's pass visits owned cells only."""
    import re  # noqa: PLC0415

    out, count = re.subn(r"^    bi[01] = \w+ \? .+ : bi[01];\n", "", source,
                         flags=re.MULTILINE)
    if not count:
        raise LookupError("the increment's ownership rows")
    return out


#: The beta pass's own defects, on the REAL source. ``must_catch``: True (must be
#: caught) | None (record only, with its reason) | False (must NOT be caught).
REAL_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_beta_pass_dropped": {
        "must_catch": True,
        "why": "the whole feature deleted: proves every other leg is measuring the "
               "pass rather than the certified curl underneath it",
        "apply": _drop_pass,
    },
    "m2_beta_sign_swap": {
        "must_catch": True,
        "why": "the two host-rounded signs exchanged (Bx is +1, By is -1)",
        "apply": lambda s: kit.needle(
            kit.needle(s, "-(beta_plus * b)", "-(beta_minus * b)"),
            "-(beta_minus * a)", "-(beta_plus * a)"),
    },
    "m3_beta_partner_swap": {
        "must_catch": True,
        "why": "Bx takes Ex instead of Ey -- the cross product's pairing",
        "apply": lambda s: kit.needle(
            kit.needle(s, "-(beta_plus * b)", "-(beta_plus * a)"),
            "-(beta_minus * a)", "-(beta_minus * b)"),
    },
    "m4_beta_partner_shifted_on_a_live_axis": {
        "must_catch": True,
        "why": "the pass reads a SHIFTED operand rather than the stored centre",
        "apply": lambda s: kit.needle(s, "-(beta_plus * b)", "-(beta_plus * b_x)"),
    },
    "m4b_beta_partner_shifted_on_the_invariant_axis": {
        "must_catch": None,
        "predicted_null_reason": (
            "beta requires an effective-2-D grid, so the invariant axis has extent "
            "1 and a shift along it is the identity: b_z IS b"),
        "why": "the z-shifted partner, byte-identical to the centre on every grid "
               "beta admits",
        "apply": lambda s: kit.needle(s, "-(beta_plus * b)", "-(beta_plus * b_z)"),
    },
    "m5_beta_scaled_by_dtdx": {
        "must_catch": True,
        "why": "the analytic derivative treated as a finite difference",
        "apply": lambda s: kit.needle(s, "-(beta_plus * b)",
                                      "-(dtdx * (beta_plus * b))"),
    },
    "m6_beta_on_the_third_target": {
        "must_catch": True,
        "why": "an increment added to the z component; step_db.cpp:161-175 runs "
               "d_c over {X, Y} only",
        "apply": lambda s: kit.needle(
            s, "    float bi1 = -(beta_minus * a);",
            "    float bi1 = -(beta_minus * a);\n    v2 = v2 + (beta_plus * c);"),
    },
    "m7_beta_folded_into_the_curl": {
        "must_catch": True,
        "why": "the PREVIOUS package form: the term inside the curl, before the "
               "mask and the per-chunk forms, instead of MEEP's separate pass after "
               "them (with the passes folded into the curl, the array path differed "
               "from MEEP in 147,772 of 251,596 words on case bz; CHANGELOG)",
        "apply": lambda s: _fold_into_curl(s, False),
    },
    "m7b_increment_not_masked": {
        "must_catch": True,
        "codes": (METALLIC, METALLIC, PERIODIC),
        "why": "the increment kept on the cells MEEP's owned-cell loop never "
               "visits (the pass masks to +0 where the curl is masked)",
        "apply": _pass_unmasked,
    },
    "m7c_ladder_ignores_dsigu": {
        "must_catch": True,
        "why": "the dsigu chunks take the outside stage: fu not advanced and f not "
               "scaled by sinv_u (By's dsigu is x, which carries the layer)",
        "apply": lambda s: kit.needle(
            kit.needle(s, "    n1 = du1 ? (n1 - bd1) : n1;", "    n1 = n1;"),
            "    v1 = du1 ? (v1 - (si_x * bd1)) : (v1 - bd1);",
            "    v1 = (v1 - bd1);"),
    },
    "m7d_ladder_skips_the_dsig_coefficient": {
        "must_catch": True,
        "why": "df = raw instead of raw * siginv: wrong in a dsig chunk (Bx's dsig "
               "is y, which carries the layer)",
        "apply": lambda s: kit.needle(s, "    float bd0 = (si_y * bi0);",
                                      "    float bd0 = bi0;"),
    },
    "m7e_fu_advanced_on_a_plain_cell": {
        "must_catch": True,
        "why": "the pass's fu stage run outside the dsigu chunks and stored there; "
               "the array path leaves fu untouched on those cells",
        "apply": lambda s: kit.needle(
            kit.needle(s, "    n1 = du1 ? (n1 - bd1) : n1;", "    n1 = (n1 - bd1);"),
            "    if (du1) { u1[ii] = n1; }", "    u1[ii] = n1;"),
    },
    "m8_contract_fast": {
        "must_catch": True,
        "why": "removes the one compile option; the pass's multiply-subtracts "
               "contract",
        "apply": lambda s: kit.needle(s, templates.contraction_pragma("off"),
                                      templates.contraction_pragma("fast")),
    },
    "m9_subtraction_as_addition_of_the_negation": {
        "must_catch": False,
        "why": "IEEE-754 identity: `x - y` IS `x + (-y)`, signed zeros included",
        "apply": lambda s: kit.needle(
            s, "    v0 = du0 ? (v0 - (si_z * bd0)) : (v0 - bd0);",
            "    v0 = du0 ? (v0 - (si_z * bd0)) : (v0 + (-bd0));"),
    },
    "m10_negation_as_zero_minus": {
        "must_catch": None,
        "predicted_null_reason": (
            "`0.0f - x` and `-x` differ only on x = +0, and the ladder's following "
            "subtraction then differs only where the field word is -0 in the same "
            "cell; the physical-band state reaches neither"),
        "why": "the Triton `0.0 - x` lowering, planted on the increment",
        "apply": lambda s: kit.needle(s, "    float bi0 = -(beta_plus * b);",
                                      "    float bi0 = 0.0f - (beta_plus * b);"),
    },
}

COMPLEX_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m11_quarter_turn_parts_swapped": {
        "must_catch": True,
        "why": "the quarter turn's two parts exchanged: multiplication by -i "
               "instead of +i",
        "apply": lambda s: kit.needle(
            s, "float2(-(beta_plus * b.y), beta_plus * b.x)",
            "float2(beta_plus * b.x, -(beta_plus * b.y))"),
    },
    "m12_beta_partner_bloch_connected": {
        "must_catch": True,
        "why": "the beta partner taken AFTER MEEP's Bloch connection; the array "
               "path's partner is the stored same-cell word",
        "apply": lambda s: kit.needle(
            s, "float2(-(beta_plus * b.y), beta_plus * b.x)",
            "float2(-(beta_plus * b_dx.y), beta_plus * b_dx.x)"),
    },
    "m13_complex_beta_folded_into_the_curl": {
        "must_catch": True,
        "why": "the PREVIOUS package form on complex storage: the term inside the "
               "curl before the mask and the forms",
        "apply": lambda s: _fold_into_curl(s, True),
    },
    "m13b_complex_ladder_ignores_dsigu": {
        "must_catch": True,
        "why": "the dsigu chunks take the outside stage on complex storage",
        "apply": lambda s: kit.needle(
            kit.needle(s, "    n1 = du1 ? (n1 - bd1) : n1;", "    n1 = n1;"),
            "    v1 = du1 ? (v1 - float2(si_x * bd1.x, si_x * bd1.y)) : (v1 - bd1);",
            "    v1 = (v1 - bd1);"),
    },
}


def _score(harness: Any, label: str, entry: Dict[str, Any], missed: bool, ran: int,
           launches: int, caught: int) -> None:
    harness.record(label, harness.verdict(missed, ran, launches, caught), launches,
                   caught, ran, entry["must_catch"], entry["why"],
                   extra=kit.predicted_null({}, entry["predicted_null_reason"])
                   if "predicted_null_reason" in entry else None)


def _source_rows(case: Dict[str, Any], table: Dict[str, Dict[str, Any]],
                 harness: Any, storage_complex: bool) -> None:
    """Every row of ``table`` on one case, each against ``stepping`` itself."""
    from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

    for label, entry in table.items():
        grid, fields, pml = build(case)
        codes = tuple(entry.get("codes", boundary_codes(grid, pml)))
        if entry.get("codes") is not None:
            # the row needs a wall the case does not carry: build that grid
            boundaries = tuple("metallic" if code == METALLIC else "periodic"
                               for code in entry["codes"])
            grid, fields, pml = build(dict(case, boundaries=boundaries))
            codes = boundary_codes(grid, pml)
        base = snapshot(fields)
        counters: List[kit.Counter] = []
        caught = ran = 0
        missed = False
        for sub_step in SUB_STEPS:
            spec = SUB_STEPS[sub_step]
            if storage_complex:
                phased, _ = complex_fields.phase_arguments(
                    phase_table(grid, pml), backward=bool(spec["backward"]))
                shipped = special_kz.beta_bloch_curl_source(
                    codes, spec["backward"], phased, True)
            else:
                shipped = special_kz.beta_curl_source(codes, spec["backward"], True)
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            oracle = oracle_after(fields, pml, base, sub_step)
            got = run_on_device(base, grid, pml, sub_step, codes, storage_complex,
                                source=mutated, counters=counters)
            ran += 1
            caught += int(any(differing(got[n], oracle[n])
                              for n in compared_names(sub_step)))
        _score(harness, label, entry, missed, ran,
               sum(counter.launches for counter in counters), caught)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Every mutation ARMED, LAUNCH-COUNTED and classified three ways, against stepping.

    Four groups: the beta pass's own defects on each storage; the per-chunk form
    battery of the certified curl the source is derived from
    (:mod:`metal_fused_battery`, KERNELS.md 7.2), whose needles resolve in the derived
    source because its forms ARE the certified text; the record mutations (the kernel
    right, the record it is handed wrong); and the host's own choices -- the Yee
    sub-lattice, the step_D connection, the coefficient's dt -- each a silent wrong
    answer no kernel source would show.
    """
    import metal_fused_battery as battery  # noqa: PLC0415

    harness = kit.MutationHarness(payload, out)
    record = harness.record
    real_case = case_named("r2_metallic_x_negative_beta")  # mask and wall live
    complex_case = case_named("c2_marquee_inplane_bloch")  # connection live

    _source_rows(real_case, REAL_MUTATIONS, harness, False)
    _source_rows(complex_case, COMPLEX_MUTATIONS, harness, True)

    # --- the certified curl's per-chunk form battery, on the derived source ----
    grid, fields, pml = build(real_case)
    codes = boundary_codes(grid, pml)
    base = snapshot(fields)
    rows_by_side = {sub_step: battery.source_rows(
        special_kz.beta_curl_source(codes, spec["backward"], True),
        curl=True, constitutive=False) for sub_step, spec in SUB_STEPS.items()}
    for name in battery.CURL_ROWS:
        counters = []
        caught = ran = 0
        entry = rows_by_side["step_B"][name]
        for sub_step in SUB_STEPS:
            oracle = oracle_after(fields, pml, base, sub_step)
            got = run_on_device(base, grid, pml, sub_step, codes, False, source=rows_by_side[sub_step][name]["source"],
                                counters=counters)
            ran += 1
            caught += int(any(differing(got[n], oracle[n])
                              for n in compared_names(sub_step)))
        _score(harness, name, {"must_catch": entry["must_catch"], "why": entry["why"]},
               False, ran, sum(counter.launches for counter in counters), caught)

    # --- the record mutations: the kernel right, the record wrong --------------
    for name, (mutate, why) in battery.RECORD_ROWS.items():
        caught = ran = 0
        for sub_step in SUB_STEPS:
            oracle = oracle_after(fields, pml, base, sub_step)
            got = run_on_device(base, grid, pml, sub_step, codes, False, record=mutate(layer_record(grid, pml)))
            ran += 1
            caught += int(any(differing(got[n], oracle[n])
                              for n in compared_names(sub_step)))
        record(name, f"CAUGHT {caught}/{ran}", ran, caught, ran, True, why)

    # --- HOST mutations ---------------------------------------------------------
    # m15: the step_D connection taken with the B side's factor.
    cgrid, cfields, cpml = build(complex_case)
    ccodes = boundary_codes(cgrid, cpml)
    cbase = snapshot(cfields)
    caught = ran = 0
    for sub_step in ("step_D",):
        oracle = oracle_after(cfields, cpml, cbase, sub_step)
        got = run_on_device(cbase, cgrid, cpml, sub_step, ccodes, True, conjugate=False)
        ran += 1
        caught += int(any(differing(got[n], oracle[n])
                          for n in compared_names(sub_step)))
    record("m15_step_d_connection_not_conjugated", f"CAUGHT {caught}/{ran}", ran,
           caught, ran, True,
           "step_D reads the word above the stored window, so its connection is "
           "eikna, not the B side's conj(eikna)")

    # m16: the Yee sub-lattice swapped -- the PML coefficients of the other half-cell.
    caught = ran = 0
    for sub_step, spec in SUB_STEPS.items():
        wrong = "" if spec["suffix"] == "_h" else "_h"
        oracle = oracle_after(fields, pml, base, sub_step)
        got = run_on_device(base, grid, pml, sub_step, codes, False, suffix=wrong)
        ran += 1
        caught += int(any(differing(got[n], oracle[n])
                          for n in compared_names(sub_step)))
    record("m16_half_integer_suffix_swap", f"CAUGHT {caught}/{ran}", ran, caught,
           ran, True,
           "the curl's PML coefficients taken from the other Yee sub-lattice")

    # m17: the coefficient rounded from a perturbed dt.
    caught = ran = 0
    for sub_step in SUB_STEPS:
        wrong_words = special_kz.beta_curl_coefficients(
            grid.beta, float(grid.dt) * 1.0001,
            magnetic=(sub_step == "step_B"), complex_storage=False)
        oracle = oracle_after(fields, pml, base, sub_step)
        got = run_on_device(base, grid, pml, sub_step, codes, False, beta_words=wrong_words)
        ran += 1
        caught += int(any(differing(got[n], oracle[n])
                          for n in compared_names(sub_step)))
    record("m17_coefficient_from_a_perturbed_dt", f"CAUGHT {caught}/{ran}", ran,
           caught, ran, True,
           "the host-rounded coefficient built from a dt 1e-4 off: the pass is the "
           "only place dt enters this sub-step without dtdx")


# ---------------------------------------------------------------------------
# LEG refusals
# ---------------------------------------------------------------------------

def leg_refusals(payload: Dict[str, Any], out: str) -> None:
    """Every named refusal, INCLUDING the two the engine itself raises.

    A predicate refusal and an engine refusal are different claims: the first says
    "this kernel will not step it", the second says "this configuration does not
    exist". Both are recorded, because a reader who saw only the predicate's
    refusal could reasonably ask why the family does not simply cover the case.
    """
    rows: List[Dict[str, Any]] = []
    residency = device.Residency()

    def add(label: str, covered: bool, reasons: Sequence[str], expect: bool,
            kind: str = "predicate") -> None:
        row = {"case": label, "kind": kind, "covered": bool(covered),
               "expected_covered": expect,
               "reasons": [str(reason) for reason in reasons][:6]}
        rows.append(row)
        payload["legs"]["refusals"] = rows
        save(payload, out)
        log(f"[refusal] {label:<44} covered={covered} expected={expect}")
        assert bool(covered) == expect, row
        if not covered:
            assert reasons, f"{label} refused with NO reason, which is not a refusal"

    # 1. beta = 0 — the certified plain curl's configuration.
    grid, fields, pml = build(case_named("r1_kz2d_periodic"), beta=0.0)
    verdict = special_kz.beta_pml_curl_coverage(fields, pml, "step_B", residency)
    add("beta_zero_belongs_to_the_plain_curl", verdict.covered, verdict.reasons,
        False)

    # 2. the positive case, so the refusals above are not all this predicate does.
    grid, fields, pml = build(case_named("r1_kz2d_periodic"))
    verdict = special_kz.beta_pml_curl_coverage(fields, pml, "step_B", residency)
    add("real_beta_run_is_ADMITTED", verdict.covered, verdict.reasons, True)

    # 3. no residency declared.
    verdict = special_kz.beta_pml_curl_coverage(fields, pml, "step_B", None)
    add("no_residency_declared", verdict.covered, verdict.reasons, False)

    # 4. no active absorber.
    inert = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
    verdict = special_kz.beta_pml_curl_coverage(fields, inert, "step_B", residency)
    add("no_active_pml", verdict.covered, verdict.reasons, False)

    # 5. complex storage on the REAL predicate, and real storage on the COMPLEX
    #    one — the inversion, measured in both directions.
    cgrid, cfields, cpml = build(case_named("c1_complex_k0"))
    verdict = special_kz.beta_pml_curl_coverage(cfields, cpml, "step_B", residency)
    add("complex_storage_on_the_real_arm", verdict.covered, verdict.reasons, False)
    verdict = special_kz.beta_bloch_pml_curl_coverage(fields, pml, "step_B",
                                                      residency)
    add("real_storage_on_the_complex_arm", verdict.covered, verdict.reasons, False)
    verdict = special_kz.beta_bloch_pml_curl_coverage(cfields, cpml, "step_B",
                                                      residency)
    add("complex_beta_run_is_ADMITTED", verdict.covered, verdict.reasons, True)

    # 7. a conductivity on a curl target.
    grid7, fields7, pml7 = build(case_named("r1_kz2d_periodic"))
    fields7.set_d_conductivity(np.full(grid7.shape, 0.15, dtype=np.float32))
    verdict = special_kz.beta_pml_curl_coverage(fields7, pml7, "step_D", residency)
    add("conductivity_on_a_curl_target", verdict.covered, verdict.reasons, False)

    # 8. off-diagonal epsilon + beta in REAL storage: the predicate refuses AND
    #    the ENGINE RAISES. Both are recorded; the second is the stronger fact.
    #
    # THE INSTALLATION IS A HARD PRECONDITION, not a best effort. This block used to
    # call a method that does not exist on Fields, catch the AttributeError, record
    # it as a `note` and CONTINUE — so clause 12b was never put to the predicate,
    # the engine was never asked to raise, and the gate's own scope line still said
    # the engine refusal had been "measured in the refusals leg". A leg that cannot
    # construct the configuration it claims to test must FAIL, not annotate.
    grid8, fields8, pml8 = build(case_named("r1_kz2d_periodic"), offdiagonal=True)
    assert fields8.has_offdiagonal_epsilon, (
        "the off-diagonal row did not install, so clause 12b and the engine "
        "refusal below would both be skipped; a refusal leg that cannot build its "
        "own configuration measures nothing")
    verdict = special_kz.beta_pml_curl_coverage(fields8, pml8, "step_B", residency)
    add("offdiagonal_epsilon_with_beta_real_storage", verdict.covered,
        verdict.reasons, False)
    # ...and the CONSTITUTIVE companion, which inherits the same clause: the
    # restated predicate must not admit an update_E the curl's own run cannot reach.
    verdict = special_kz.beta_run_constitutive_coverage(fields8, pml8, "E",
                                                        residency)
    add("offdiagonal_epsilon_on_the_constitutive_companion", verdict.covered,
        verdict.reasons, False)
    raised = None
    try:
        stepping.step_B(fields8, pml8)
    except ValueError as exc:
        raised = str(exc)
    rows.append({"case": "offdiag_beta_engine_raises", "kind": "engine",
                 "raised": raised})
    payload["legs"]["refusals"] = rows
    save(payload, out)
    assert raised and "complex" in raised, (
        "stepping.py:800-810 did not raise on off-diagonal epsilon with beta; "
        "the predicate's refusal was believed to mirror an engine refusal")
    log("[refusal] offdiag+beta: the ENGINE raises "
        f"({raised.splitlines()[0][:60]}...)")

    # 9. the engine's OWN geometry refusals: beta off a 2-D Cartesian grid.
    for label, kwargs in (
            ("beta_on_a_3d_grid", {"dimensions": 3, "cell_size": (2.0, 1.6, 1.2)}),
            ("beta_on_a_cylindrical_grid", {"dimensions": 2, "cylindrical": True,
                                            "cell_size": (2.0, 0.0, 1.6)})):
        raised = None
        try:
            Grid(resolution=10.0, boundaries="periodic", courant=0.35,
                 k_point=(0.0, 0.0, 0.0), beta=BETA_KZ2D, xp=np, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
            raised = str(exc)
        rows.append({"case": label, "kind": "engine", "raised": raised})
        payload["legs"]["refusals"] = rows
        save(payload, out)
        log(f"[refusal] {label:<44} engine raises: {bool(raised)}")
        assert raised, (
            f"{label}: the Grid accepted a configuration MEEP aborts on "
            f"(fields.cpp:546-547), so this family's restated clause is guarding "
            f"something the engine no longer refuses")

    # 10. the two constitutive companions, both ADMITTED and both delegating to a
    #     certified kernel. The complex one was an unconditional refusal until
    #     2026-08-19; these rows are the inversion, measured in both directions so
    #     "admitted" cannot mean "admits anything".
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        cfields, cpml, "E", residency)
    add("complex_constitutive_on_a_COMPLEX_beta_run_is_ADMITTED", verdict.covered,
        verdict.reasons, True)
    verdict = special_kz.beta_run_constitutive_coverage(fields, pml, "H", residency)
    add("real_constitutive_on_a_beta_run_is_ADMITTED", verdict.covered,
        verdict.reasons, True)
    # ...and each refuses the OTHER storage, which is what keeps the pair disjoint
    # rather than merely both-green.
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        fields, pml, "H", residency)
    add("complex_constitutive_on_a_REAL_beta_run", verdict.covered,
        verdict.reasons, False)
    verdict = special_kz.beta_run_constitutive_coverage(cfields, cpml, "H",
                                                        residency)
    add("real_constitutive_on_a_COMPLEX_beta_run", verdict.covered,
        verdict.reasons, False)
    # ...and no residency declared.
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        cfields, cpml, "E", None)
    add("complex_constitutive_without_a_residency", verdict.covered,
        verdict.reasons, False)
    # 10b. an off-diagonal chi1inv row: the complex CURL admits it (the row product
    #      is constitutive-only), the E-side constitutive must NOT, and the H side
    #      is untouched by it. The three rows together are the scope of the clause.
    grid10, fields10, pml10 = build(case_named("c1_complex_k0"), offdiagonal=True)
    assert fields10.has_offdiagonal_epsilon, (
        "the off-diagonal row did not install on complex storage, so the E-side "
        "clause below would be measured against a configuration that does not "
        "carry it")
    verdict = special_kz.beta_bloch_pml_curl_coverage(fields10, pml10, "step_D",
                                                      residency)
    add("offdiagonal_on_the_complex_beta_CURL_is_ADMITTED", verdict.covered,
        verdict.reasons, True)
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        fields10, pml10, "E", residency)
    add("offdiagonal_on_the_complex_constitutive_E_side", verdict.covered,
        verdict.reasons, False)
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        fields10, pml10, "H", residency)
    add("offdiagonal_on_the_complex_constitutive_H_side_is_ADMITTED",
        verdict.covered, verdict.reasons, True)


# ---------------------------------------------------------------------------
# The arm table, made DETERMINISTIC before anything sweeps it
# ---------------------------------------------------------------------------

#: The FLOOR: families that must appear in the registry once every module is
#: imported. A sweep that silently shrinks — because a family stopped registering,
#: or because nobody imported it — would report "no co-admission" over a smaller
#: table and look identical to a sweep that measured everything.
#:
#: THIS IS A FLOOR AND NO LONGER THE WHOLE TABLE, and the difference is the defect
#: this constant caused. It was written as the complete expected membership while
#: ``special_kz`` and ``bfast_curl`` were unregistered; when a later round WIRED
#: them, the literal still described the old tree. The floor cannot notice a family
#: that JOINS, so :func:`assert_table_is_whole` now derives the rest from the
#: modules' own ``ARMS`` tuples — every arm a family declares must be findable in
#: the registry — and keeps this literal only so an empty declaration cannot make
#: the derived half vacuous.
EXPECTED_REGISTERED_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "step_B": ("pml_curl", "complex_fields"),
    "step_D": ("pml_curl", "complex_fields"),
    "update_H": ("constitutive", "no_pml_constitutive", "complex_fields"),
    "update_E": ("constitutive", "no_pml_constitutive", "complex_fields",
                 "offdiag_constitutive"),
}

#: Modules that declare a module-level ``ARMS`` tuple, DISCOVERED rather than named.
#:
#: THE LITERAL THIS REPLACES READ ``("special_kz", "bfast_curl")`` AND WAS A
#: DOUBLE-COUNT WAITING FOR A WIRING. It meant "families that register NOTHING",
#: and :func:`unregistered_arms` fed its entries into the sweep as a second,
#: supposedly disjoint source. Measured 2026-08-15, after another round wired this
#: family: ``arms.registered('step_B')`` and ``special_kz.ARMS`` return THE SAME
#: ``ArmSpec`` OBJECTS, so every beta case reported ``step_B`` co-admitted by
#: ``['wired:special_kz_real/...', 'unregistered:special_kz_real/...']`` — one arm,
#: counted twice, failing the disjointness assertion on itself. The list is derived
#: now, and membership is decided by OBJECT IDENTITY against the registry rather
#: than by which literal a module's name appears in.
def arm_declaring_modules() -> Tuple[str, ...]:
    from meep_gpu import metal_kernels  # noqa: PLC0415

    return tuple(name for name in metal_kernels.__all__
                 if getattr(getattr(metal_kernels, name), "ARMS", None))


def load_every_family() -> Tuple[str, ...]:
    """Import every family module, so the arm registry is a fact and not a side effect.

    THE REGISTRY'S MEMBERSHIP IS AN IMPORT SIDE EFFECT and this leg used to inherit
    whatever the gate happened to have imported. Measured 2026-08-15 from this
    gate's own import set: ``arms.registered('step_B')`` returned ``[pml_curl]``
    alone — ``complex_fields`` registers on all four slots and ``offdiag_update_e``
    on ``update_E``, and NEITHER module was ever imported, so the sweep proved
    disjointness against a table missing half the families that have one. No verdict
    changed (both refuse a beta run), but "no arm co-admits" was a statement about a
    table, and the table was not the tree's.
    """
    from meep_gpu import metal_kernels  # noqa: PLC0415

    for name in metal_kernels.__all__:
        getattr(metal_kernels, name)
    return tuple(metal_kernels.__all__)


def declared_arms() -> Tuple[Any, ...]:
    """Every ``ArmSpec`` any family declares in a module-level ``ARMS`` tuple."""
    from meep_gpu import metal_kernels  # noqa: PLC0415

    load_every_family()
    out: List[Any] = []
    for name in arm_declaring_modules():
        out.extend(getattr(getattr(metal_kernels, name), "ARMS", ()) or ())
    return tuple(out)


def unregistered_arms() -> Tuple[Any, ...]:
    """Declared arms that are NOT in the registry — compared by OBJECT IDENTITY.

    IDENTITY, NOT ``(family, slot, label)``, and the distinction is the whole point.
    A family that registers the very objects it declares must contribute them ONCE
    to the sweep; comparing by name would work here only because no family declares
    a second arm with a matching triple, which is a property of today's tree rather
    than of the check. What remains after the subtraction is the genuinely
    unreachable set — arms a module declares and never registers, which
    ``arms.registered()`` cannot see and a whole-table sweep must still consult.
    """
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415

    registry = metal_arms.registered()
    return tuple(spec for spec in declared_arms()
                 if not any(spec is known for known in registry))


def peer_admitters(names: List[str]) -> List[str]:
    """The single-slot admitters among :func:`admitters_for`'s tagged names."""
    return [n for n in names if not n.startswith("weld:")]


def weld_admitters(names: List[str]) -> List[str]:
    """The seam-replacing admitters among :func:`admitters_for`'s tagged names."""
    return [n for n in names if n.startswith("weld:")]


def admitters_for(slot: str, context: Any) -> List[str]:
    """Every product that admits this slot — WIRED, UNWIRED-REGISTERED and UNREGISTERED.

    Unwired arms are consulted rather than skipped, and that is the correction that
    matters most here: the pair most likely to collide on a curl slot is THIS
    family's complex arm against ``complex_fields``' complex arm, so a sweep that
    skipped unwired rows could never see it. A raise is a refusal on every track,
    but it is RECORDED as an admitter string so a predicate that started raising
    cannot pass as a refusal without appearing in the artifact.

    EACH ARM APPEARS AT MOST ONCE. :func:`unregistered_arms` subtracts the registry
    by object identity, so a family that both declares and registers its arms — as
    this one now does — contributes one row and not two. Before that subtraction
    every beta case failed the disjointness assertion against ITSELF.
    """
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415

    names: List[str] = []
    for spec in metal_arms.registered(slot):
        try:
            verdict = spec.coverage(context, slot)
        except Exception as exc:  # noqa: BLE001 - a raise is a refusal, and is recorded
            names.append(f"{spec.family}/{spec.label}:RAISED {exc!r}")
            continue
        if getattr(verdict, "covered", False):
            # A WELD IS TAGGED APART FROM A PEER, and not by ``wired``. A weld
            # replaces a whole seam and its predicate CONTAINS the wired arm's by
            # construction, so it always co-admits with the arm it welds; a peer
            # replaces one slot and two peers on a slot is the over-covering
            # dispatch. ``wired`` cannot separate them -- every unwired arm here is
            # a weld today, so filtering on it would delete the deferred single-slot
            # peer this sweep exists to catch (see the beta_zero check below, which
            # is exactly that case).
            if spec.is_weld:
                tag = "weld"
            else:
                tag = "wired" if spec.wired else "unwired-registered"
            names.append(f"{tag}:{spec.family}/{spec.label}")
    for spec in unregistered_arms():
        if spec.slot != slot:
            continue
        try:
            verdict = spec.coverage(context, slot)
        except Exception as exc:  # noqa: BLE001
            names.append(f"{spec.family}/{spec.label}:RAISED {exc!r}")
            continue
        if getattr(verdict, "covered", False):
            names.append(f"unregistered:{spec.family}/{spec.label}")
    return names


def assert_table_is_whole(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record what the registry holds, and FAIL if the sweep cannot see the tree.

    TWO CHECKS, because neither alone survives a wiring change. The LITERAL FLOOR
    catches a family that stops registering — the failure that shrinks the table
    silently. The DERIVED check catches the opposite one, which is what actually
    happened: a family that JOINS the registry is invisible to a literal, and this
    leg swept a table it believed was complete while ``special_kz`` and
    ``bfast_curl`` had moved into it. Every arm a module DECLARES must be findable
    in the registry, or be recorded as deliberately unregistered — there is no
    third state, and a declared arm that is neither is an arm the composer will
    never consult and no sweep would notice.

    THE SWEEP MUST ALSO SEE THIS FAMILY. The old form asserted a non-empty
    unregistered list to guarantee that; now that this family registers, the same
    guarantee is spelled directly against the registry.
    """
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415

    loaded = load_every_family()
    slots = ("step_B", "step_D", "update_H", "update_E")
    table = {slot: sorted({spec.family for spec in metal_arms.registered(slot)})
             for slot in slots}
    declared = declared_arms()
    unregistered = sorted({f"{spec.family}/{spec.slot}"
                           for spec in unregistered_arms()})
    record = {"modules_imported": list(loaded),
              "arm_declaring_modules": list(arm_declaring_modules()),
              "registered_families": table,
              "declared_arm_count": len(declared),
              "unregistered_arms": unregistered,
              "wired_families": {
                  slot: sorted({spec.family for spec in metal_arms.registered(slot)
                                if spec.wired}) for slot in slots},
              "expected_floor": {k: list(v) for k, v in
                                 EXPECTED_REGISTERED_FAMILIES.items()}}
    payload["arm_table"] = record

    for slot, expected in EXPECTED_REGISTERED_FAMILIES.items():
        missing = [family for family in expected if family not in table[slot]]
        assert not missing, (
            f"the arm table for {slot} is missing {missing}; the disjointness sweep "
            f"below would prove 'no co-admission' over a SMALLER table than the tree "
            f"carries, which is what happens when a family stops registering or "
            f"nobody imports it")

    registry = metal_arms.registered()
    for spec in declared:
        known = any(spec is entry for entry in registry)
        assert known or spec in unregistered_arms(), (
            f"{spec.family}/{spec.slot} is declared in a module ARMS tuple but is "
            f"neither in the registry nor in the unregistered set; the sweep would "
            f"consult it under neither name")

    assert any(spec.family.startswith("special_kz") for spec in registry), (
        "no special_kz arm is in the registry, so the disjointness sweep cannot "
        "see the family it is certifying: 'exactly one admitter' would then be a "
        "statement about everyone ELSE")
    return record


# ---------------------------------------------------------------------------
# LEG engine — the engine route and the disjointness sweep
# ---------------------------------------------------------------------------

def leg_engine(payload: Dict[str, Any], out: str) -> None:
    """Plans from the engine's own objects, and NO arm co-admitting with any other.

    The engine route and the ``from_arrays`` route produce the same object and go
    through the same ``run``, which is what makes the gate's bytes the engine's
    bytes. The sweep then asks EVERY arm in the tree — wired, unwired-registered and
    unregistered — for a verdict on every case and asserts at most one admits, in
    both directions, which is the only form in which "clause 12 is inverted" is a
    measurement. :func:`assert_table_is_whole` is what makes "every arm" checkable
    rather than a description of whatever happened to be imported.
    """
    from meep_gpu.metal_kernels import arms as metal_arms
    from meep_gpu.metal_kernels import launch as metal_launch

    table = assert_table_is_whole(payload)
    log(f"[engine] arm table: " + ", ".join(
        f"{slot}={families}" for slot, families in
        sorted(table["registered_families"].items())))
    log(f"[engine] unregistered arms swept: {table['unregistered_arms']}")

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        residency = device.Residency()
        grid, fields, pml = build(case)
        context = metal_arms.StepContext(fields, pml, residency,
                                         (templates.CONTRACT_OFF,),
                                         sources=())
        wired: Dict[str, List[str]] = {
            slot: admitters_for(slot, context)
            for slot in ("step_B", "step_D", "update_H", "update_E")}

        plan = None
        built = {}
        if case["complex"]:
            for sub_step in SUB_STEPS:
                built[sub_step] = special_kz.plan_beta_bloch_pml_curl(
                    fields, pml, sub_step, residency,
                    chunk_bounds=layer_record(grid, pml))
        else:
            for sub_step in SUB_STEPS:
                built[sub_step] = special_kz.plan_beta_pml_curl(
                    fields, pml, sub_step, residency,
                    chunk_bounds=layer_record(grid, pml))
        shipped = metal_launch.plan_step(fields, pml, residency=residency,
                                         sources=())
        row = {"case": case["name"], "complex": case["complex"],
               "admitters": wired,
               "engine_route_built": {k: v is not None for k, v in built.items()},
               "shipped_plan_replaces": list(shipped.replaces),
               "shipped_curl_reasons": {
                   slot: list(shipped.reasons.get(slot, ()))[:2]
                   for slot in ("step_B", "step_D")}}
        rows.append(row)
        payload["legs"]["engine"] = rows
        save(payload, out)
        log(f"[engine] {case['name']:<28} built={row['engine_route_built']} "
            f"shipped_replaces={shipped.replaces} ({time.time() - started:.1f}s)")
        for slot, admitting in wired.items():
            peers, welds = peer_admitters(admitting), weld_admitters(admitting)
            assert len(peers) <= 1, (
                f"{case['name']}: {slot} is co-admitted by {peers}; two PEER "
                f"products on one slot is the over-covering dispatch the composer "
                f"fails closed on")
            assert not welds or peers, (
                f"{case['name']}: {slot} is admitted by WELD(s) {welds} while no "
                f"peer arm admits it; the weld covers a configuration no certified "
                f"sub-step arm covers")
        # THE VACUITY FLOOR, without which `<= 1` passes hardest when the sweep is
        # broken. Zero admitters satisfies the disjointness assertion perfectly, so
        # a product switched off by its harness STRENGTHENS the green. Every case
        # in CASES is a beta run this
        # family covers on both curl slots, so anything less than one admitter
        # there means the sweep measured a product it had switched off.
        # COUNTED OVER PEERS, for the same reason the disjointness above is: a weld
        # co-admits by construction, so counting it here would make "exactly one"
        # unsatisfiable on any slot a weld covers. The floor's PURPOSE is unchanged
        # — zero peers still means the sweep switched off the product it measures.
        for slot in ("step_B", "step_D"):
            slot_peers = peer_admitters(wired[slot])
            assert len(slot_peers) == 1, (
                f"{case['name']}: {slot} has {len(slot_peers)} PEER admitters "
                f"({slot_peers}; welds {weld_admitters(wired[slot])}), not exactly "
                f"one. This family covers both curl slots on every case here — zero "
                f"means the sweep disabled the very product whose disjointness it "
                f"claims to measure")
        # The constitutive pair used to be ASYMMETRIC and this assertion said so —
        # the real arm shipped a companion, the complex arm had none, so the
        # expected admitter count was 0 on a complex case. Both arms ship one as of
        # 2026-08-19, so exactly one is now the rule on all four slots. The FLOOR is
        # what matters: zero would mean the sweep switched off the very product
        # whose disjointness it claims to measure, and two would mean the real and
        # complex arms co-admit.
        for slot in ("update_H", "update_E"):
            slot_peers = peer_admitters(wired[slot])
            assert len(slot_peers) == 1, (
                f"{case['name']}: {slot} has {len(slot_peers)} PEER admitters "
                f"({slot_peers}; welds {weld_admitters(wired[slot])}), not exactly "
                f"one. Both storages now ship a constitutive companion, and clause 2 "
                f"is what keeps them apart")
        for sub_step, plan_object in built.items():
            assert plan_object is not None, (
                f"{case['name']}/{sub_step}: the ENGINE route refused a case the "
                f"from_arrays route steps, so the gate's bytes are not the "
                f"engine's bytes")
            plan_object.run()
            assert plan_object.launches == 1
        residency.sync_out()
        # THE ASSERTION HERE USED TO BE ITS OWN INVERSE, and the change is a fact
        # about the tree rather than a relaxation. It read `slot not in
        # shipped.replaces` — "clause 12 refuses every shipped arm on a beta run" —
        # which was true only while this family was UNWIRED. A later round wired it,
        # so the shipped composer now selects THIS family on both curl slots, and
        # the old form would have failed on the wiring working. The replacement is
        # strictly stronger: the slot must be filled, and it must be filled BY THIS
        # FAMILY. "Something replaced step_B" would still pass if the plain curl had
        # started admitting beta runs, which is the actual hazard clause 12 guards.
        for slot in ("step_B", "step_D"):
            assert slot in shipped.replaces, (
                f"{case['name']}: the SHIPPED composer left {slot} on the array "
                f"path for a beta run this family is wired for. Both curl slots are "
                f"checked because the predicate is sub-step SCOPED — a clause that "
                f"leaked on one of them only would be invisible. Reasons: "
                f"{list(shipped.reasons.get(slot, ()))[:3]}")
            chosen = shipped.selected.get(slot, "")
            assert "special_kz" in chosen, (
                f"{case['name']}: the shipped composer filled {slot} with "
                f"{chosen!r}, not a special_kz arm. A beta run served by another "
                f"family's kernel is the over-covering dispatch clause 12 exists to "
                f"prevent, and it would read as a green 'slot replaced' row")
        row["shipped_selected"] = {slot: shipped.selected.get(slot)
                                   for slot in ("step_B", "step_D",
                                                "update_H", "update_E")}
        # THE CONSTITUTIVE HALF IS NO LONGER ASYMMETRIC, and this block is where
        # that shows up as a composition fact rather than as prose. Until 2026-08-19
        # a complex beta run kept its constitutive pair on the array path — there
        # was no complex companion — and this asserted the EMPTY list, with a
        # predicted null carrying the reason. Both storages ship a companion now, so
        # the shipped composer must fill BOTH slots on EVERY case. Half a pair still
        # fails: it would mean the composition is asymmetric in a way no other leg
        # here would see. The complex rows are the last two slots of the Metal
        # census (`TestSpecialKz.test_special_kz`, update_H and update_E).
        constitutive = [slot for slot in ("update_H", "update_E")
                        if slot in shipped.replaces]
        row["shipped_constitutive_slots"] = constitutive
        assert constitutive == ["update_H", "update_E"], (
            f"{case['name']}: the shipped composer filled {constitutive} of the "
            f"constitutive pair on a "
            f"{'COMPLEX' if case['complex'] else 'REAL'} beta run; the certified "
            f"companion covers both. Reasons: "
            f"{ {s: list(shipped.reasons.get(s, ()))[:2] for s in ('update_H', 'update_E')} }")
        for slot in ("update_H", "update_E"):
            chosen = shipped.selected.get(slot, "")
            assert "special_kz" in chosen, (
                f"{case['name']}: the shipped composer filled {slot} with "
                f"{chosen!r}, not a special_kz arm — a beta run's constitutive "
                f"pair served by another family's admission is the over-covering "
                f"dispatch the beta clauses exist to prevent")
        payload["legs"]["engine"] = rows
        save(payload, out)

    # The other direction: this family's arms must refuse a beta = 0 run, which is
    # exactly what the shipped composition covers.
    residency = device.Residency()
    grid, fields, pml = build(case_named("r1_kz2d_periodic"), beta=0.0)
    context = metal_arms.StepContext(fields, pml, residency,
                                     (templates.CONTRACT_OFF,), sources=())
    unwired_on_plain = [spec.label for spec in special_kz.ARMS
                        if spec.slot == "step_B"
                        and spec.coverage(context, "step_B").covered]
    shipped = metal_launch.plan_step(fields, pml, residency=residency, sources=())
    rows.append({"case": "beta_zero_reverse_direction",
                 "unwired_admitters_on_a_plain_run": unwired_on_plain,
                 "admitters_step_B": admitters_for("step_B", context),
                 "shipped_plan_replaces": list(shipped.replaces)})
    payload["legs"]["engine"] = rows
    save(payload, out)
    log(f"[engine] beta_zero reverse: unwired admitters={unwired_on_plain} "
        f"shipped replaces={shipped.replaces}")
    assert not unwired_on_plain, (
        "a beta arm admitted a beta = 0 run, which the certified curl covers: the "
        "two would co-admit the moment this family is wired")
    assert "step_B" in shipped.replaces, (
        "the shipped curl did NOT admit a plain beta = 0 run, so the disjointness "
        "measured above is vacuous — nothing covers anything")

    # ...and the SAME control for the COMPLEX half, which had none. Every complex
    # case above reports exactly one admitter — this family's complex arm — and that
    # would read identically if the only other complex product on the slot refused
    # EVERYTHING it was ever shown. `complex_fields` must ADMIT a complex beta = 0
    # run, or the complex rows measure an absence rather than a disjointness.
    residency = device.Residency()
    grid, fields, pml = build(case_named("c2_marquee_inplane_bloch"), beta=0.0)
    context = metal_arms.StepContext(fields, pml, residency,
                                     (templates.CONTRACT_OFF,), sources=())
    complex_control = {slot: admitters_for(slot, context)
                       for slot in ("step_B", "step_D")}
    control_row = {"case": "complex_beta_zero_control",
                   "admitters": complex_control}
    rows.append(control_row)
    payload["legs"]["engine"] = rows
    save(payload, out)
    log(f"[engine] complex beta_zero control: {complex_control}")
    for slot, admitting in complex_control.items():
        peers, welds = peer_admitters(admitting), weld_admitters(admitting)
        assert len(peers) <= 1, (
            f"complex_beta_zero_control: {slot} is co-admitted by {peers}")
        assert not welds or peers, (
            f"complex_beta_zero_control: {slot} is admitted by WELD(s) {welds} "
            f"while no peer arm admits it")
        admitting = peers
        assert any("complex_fields" in name for name in admitting), (
            f"complex_beta_zero_control: {slot} was admitted by {admitting}, and "
            f"no complex_fields arm is among them. The complex disjointness rows "
            f"above are then vacuous — "
            f"the only other complex curl on the slot refuses every configuration "
            f"it is shown, so 'exactly one admitter' says nothing about overlap")


# ---------------------------------------------------------------------------
# LEG shipped — the COMPOSER'S OWN CHOICE, byte-checked against stepping.py
# ---------------------------------------------------------------------------

def leg_shipped(payload: Dict[str, Any], out: str) -> None:
    """The plan ``launch.plan_step`` SELECTS, against ``stepping.py``, 4 cycles.

    THIS LEG EXISTS BECAUSE THE FAMILY WAS WIRED, and until it was, nothing needed
    it. Three routes reach a beta kernel and this gate previously certified two:
    ``plan_*_from_arrays`` (every other leg's route, and the mutation seam) and
    ``plan_beta_*`` from the engine's own objects (``leg_engine``). The third is
    ``launch.plan_step``, the composer a caller actually reaches — it picks the arm,
    binds the probes and hands back the plan. ``leg_engine`` asserts WHICH arm it
    picked; it never ran the picked plan against the oracle, so a composer that
    selected the right family and bound it wrongly — a mirror to the wrong array, a
    coefficient block from the wrong sub-step's suffix — would have passed every
    leg in this file while a user got wrong bytes.

    IT IS NOT A DUPLICATE OF ``leg_reference``: that leg builds the plan itself, so
    the two agree only if the composer's binding matches the harness's. Here the
    harness supplies the SEEDS and nothing else, and the composer supplies the plan.

    THE COMPARED SET AND THE DISCIPLINE ARE THE FAMILY'S USUAL ONES: uint32 words
    over targets AND auxiliaries, the device's own answer fed back so cycle N+1
    starts from device state, and `assert_moved` on every row so a no-op cannot
    agree with a no-op.

    IT SWEEPS ALL FOUR SUB-STEPS AS OF 2026-08-19, not just the two curls. Before
    that a complex beta run had no constitutive companion, so half the whole step
    was on the array path and there was nothing for the composer to be checked on;
    the curl-only sweep was the honest shape. Now both constitutive slots compose on
    both storages, and a composer that picked the right constitutive arm and bound
    it to the wrong sub-lattice's coefficient block (`kps_x` against `kps_x_h`) is
    exactly the defect this route — and only this route — can see.
    """
    from meep_gpu.metal_kernels import launch as metal_launch
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

    #: Which words a slot's comparison covers. The curls' set is the family's
    #: `compared_names`; a constitutive slot writes its targets and its `f_w_*`
    #: auxiliaries, which is where a wrong coefficient block would show up first.
    def compared(slot: str) -> Tuple[str, ...]:
        if slot in SUB_STEPS:
            return compared_names(slot)
        spec = CONSTITUTIVE_SIDES["H" if slot == "update_H" else "E"]
        return tuple(spec["targets"]) + tuple(spec["aux"])

    whole_step = tuple(SUB_STEPS) + ("update_H", "update_E")
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        grid, fields, pml = build(case)
        for cycle in range(CYCLES):
            for sub_step in whole_step:
                base = snapshot(fields)
                getattr(stepping, sub_step)(fields, pml)
                oracle = snapshot(fields)
                restore(fields, base)

                residency = device.Residency()
                shipped = metal_launch.plan_step(
                    fields, pml, residency=residency, sources=())
                plan = shipped.plans.get(sub_step)
                assert plan is not None, (
                    f"{case['name']}/{sub_step}: the shipped composer left this "
                    f"slot on the array path; leg_engine asserts it does not, so "
                    f"one of the two is measuring a different configuration. "
                    f"Reasons: {list(shipped.reasons.get(sub_step, ()))[:3]}")
                plan.run()
                assert plan.launches == 1, (
                    f"{case['name']}/{sub_step}: the composer's plan launched "
                    f"{plan.launches} times, not once — a leg that compared bytes "
                    f"without counting launches cannot tell one kernel from two")
                residency.sync_out()

                got = snapshot(fields)
                names = compared(sub_step)
                moved = sum(differing(oracle[n], base[n]) for n in names)
                bad = compare(got, oracle, names)
                rows.append({"case": case["name"], "cycle": cycle,
                             "sub_step": sub_step, "complex": case["complex"],
                             "selected": shipped.selected.get(sub_step),
                             "moved": moved, "differing": bad})
                payload["legs"]["shipped"] = rows
                save(payload, out)
                kit.assert_moved(moved, f"{case['name']}/{sub_step} (shipped)")
                assert not bad, rows[-1]
                # `fields` already holds the device's answer — sync_out wrote it
                # back through the mirrors, so the next cycle starts from it.
        log(f"[shipped] {case['name']:<28} {CYCLES}x{len(whole_step)} IDENTICAL "
            f"via the composer ({time.time() - started:.1f}s)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("reference", leg_reference),
    ("identity", leg_identity),
    ("constitutive", leg_constitutive),
    ("guard", leg_guard),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("refusals", leg_refusals),
    ("engine", leg_engine),
    ("shipped", leg_shipped),
)

REQUIRED = tuple(name for name, _ in LEGS)


def weld_state() -> Dict[str, Any]:
    """The LIVE host/kernel weld, and whether the checked-in one still describes it.

    TWO DIFFERENT FACTS, and conflating them is how a certified artifact stops
    naming the bytes it certified. ``fingerprints.json`` records what was last
    welded DELIBERATELY; this record is what THIS run actually launched. They
    normally coincide and the interesting case is when they do not.

    A DRIFT IS RECORDED, NOT FAILED, and the asymmetry is deliberate. This gate
    launches the live tree and hashes it, so a host module edited by another round
    leaves the claim well defined — it is a claim about the tree as it stands, and
    re-running the gate is the correct response to the edit, not refusing to run.
    What would be dishonest is minting a green artifact whose only byte record is a
    weld that no longer describes the run, which is precisely what happened here:
    measured 2026-08-15, ``launch.py`` moved 19:52 against a weld cut 19:47, while
    ``kernel_source_sha256`` still matched to the word because no shipped
    specialisation had changed. ``test_the_checked_in_fingerprints_match_the_tree``
    is the thing that must go red for that, and it does; this record is so the
    artifact says which bytes won.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    live = metal_launch.compute_fingerprints()["metal_kernels"]
    try:
        stored = metal_launch.load_fingerprints()["metal_kernels"]
    except (OSError, ValueError) as error:
        return {"live_host_sha256": live["host_sha256"],
                "checked_in_weld_readable": False,
                "agrees_with_checked_in_weld": False,
                "reason": f"fingerprints.json could not be read: {error}"}

    def drifted(key: str) -> List[str]:
        was, now = stored.get(key, {}), live.get(key, {})
        return sorted(set(was) ^ set(now)) + sorted(
            name for name in set(was) & set(now) if was[name] != now[name])

    hosts, sources = drifted("host_sha256"), drifted("kernel_source_sha256")
    return {
        "live_host_sha256": live["host_sha256"],
        "live_kernel_source_count": len(live["kernel_source_sha256"]),
        "checked_in_weld_readable": True,
        "agrees_with_checked_in_weld": not hosts and not sources,
        "drifted_host_modules": hosts,
        "drifted_kernel_sources": sources,
        "reads_as": (
            "the live hashes are what this run certified; a drifted HOST module "
            "means another round edited the tree after the weld was cut, and a "
            "drifted KERNEL SOURCE would mean a shipped specialisation changed "
            "without the weld being re-cut — a correctness event, not a "
            "bookkeeping one"),
    }


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()

    started = time.time()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    environment = kit.environment_stamp()
    kernel_dir = os.path.join(API_ROOT, "meep_gpu", "metal_kernels")
    payload: Dict[str, Any] = {
        "environment": environment,
        # THE CLAIM IS ONLY AS GOOD AS THE PRECONDITION IT WAS CERTIFIED UNDER.
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": kit.provenance(out_dir, {
            "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
            "metal_kernels/special_kz.py": os.path.join(kernel_dir, "special_kz.py"),
            "metal_kernels/templates.py": os.path.join(kernel_dir, "templates.py"),
            "metal_kernels/folded_pass.py": os.path.join(kernel_dir, "folded_pass.py"),
            "metal_kernels/complex_fields.py": os.path.join(kernel_dir,
                                                            "complex_fields.py"),
            "metal_kernels/chunk_bounds.py": os.path.join(kernel_dir, "chunk_bounds.py"),
            "metal_kernels/shaders.py": os.path.join(kernel_dir, "shaders.py"),
            "metal_kernels/plans.py": os.path.join(kernel_dir, "plans.py"),
            "metal_kernels/device.py": os.path.join(kernel_dir, "device.py"),
            "metal_kernels/coverage.py": os.path.join(kernel_dir, "coverage.py"),
            "metal_kernels/preconditions.py": os.path.join(kernel_dir,
                                                           "preconditions.py"),
            # THE PLANNER IS IN THIS FAMILY'S PATH AND WAS NOT HASHED. `leg_engine`
            # calls `launch.plan_step` and asserts on `shipped.replaces` in BOTH
            # directions, and `arms`/`registry` decide the table that sweep is run
            # against — so their bytes decide a verdict this artifact stamps.
            # Measured 2026-08-15: `launch.py` was edited by a concurrent round
            # AFTER the checked-in weld was cut, and neither the hand-listed
            # provenance above nor `kernel_sources` (unchanged — no shipped
            # specialisation moved) could see it. `weld` below is the general fix;
            # these three are named because they are called by name here.
            "metal_kernels/launch.py": os.path.join(kernel_dir, "launch.py"),
            "metal_kernels/arms.py": os.path.join(kernel_dir, "arms.py"),
            "metal_kernels/registry.py": os.path.join(kernel_dir, "registry.py"),
            "gate_metal_special_kz.py": os.path.abspath(__file__),
        }, kernel_sources=special_kz.enumerate_sources()),
        "weld": weld_state(),
        "legs": {},
        "tranche": "special_kz (grid.beta) — real + Bloch arms, Metal",
    }
    save(payload, arguments.out)

    log(f"[env] torch={environment.get('torch')} numpy={environment['numpy']} "
        f"frontend={environment.get('metal_frontend')} "
        f"policy={payload['subnormal_policy']['resolved']}")
    weld = payload["weld"]
    if weld.get("agrees_with_checked_in_weld"):
        log("[weld] the checked-in weld describes this tree exactly")
    else:
        log(f"[weld] DRIFT — this run certifies the LIVE tree, not the checked-in "
            f"weld: hosts={weld.get('drifted_host_modules')} "
            f"kernel_sources={weld.get('drifted_kernel_sources')}")

    if not environment.get("mps_available"):
        return kit.cannot_certify(payload, arguments.out, [
            "no MPS device on this host: every leg of this gate launches a Metal "
            "kernel, and a run that skipped them all must not mint a green "
            "artifact"])
    if not payload["subnormal_policy"]["admitted"]:
        return kit.cannot_certify(payload, arguments.out, [
            f"the MPS executor refused the resolved subnormal policy: "
            f"{payload['subnormal_policy']['reasons']}"])

    wanted = kit.wanted_legs(arguments.legs)
    ran = kit.run_legs(LEGS, payload, arguments.out, wanted)
    missing = [name for name in REQUIRED if name not in ran]
    payload["counters"] = dict(TALLY)
    if missing:
        return kit.cannot_certify(payload, arguments.out, [
            f"required legs did not run: {missing}. A --legs subset is for "
            f"debugging and must not be able to mint a certified artifact"])
    return kit.summarize(
        payload, arguments.out,
        claim=("byte-identity to stepping.py on this host, under a CHECKED "
               "subnormal-free precondition — not a stated tolerance"),
        scope=("the special_kz (grid.beta) split-field PML curl, real float32 and "
               "complex64 storage, periodic and metallic ghost rules, in-plane "
               "Bloch phase, both corpus beta signs, Courants 0.34/0.4375/0.35/0.5; "
               "NO fold, no cylindrical axis, no conductivity, no dispersion, no "
               "chi2/chi3, no BFAST, and no off-diagonal row under real storage "
               "(the ENGINE itself refuses that one, measured in the refusals leg)"),
        stated_weakness=("no PTX-equivalent audit exists on this executor: "
                         "compile_shader exposes no disassembly, so the mutation "
                         "legs and this gate are the only arbiters"),
        started=started, legs_run=ran,
        compared=TALLY["compared"], certified=(TALLY["disagreed"] == 0),
        extra={"not_covered": (
            "the FOLDED beta run, in either storage: folded_beta is that family "
            "and carries its own gate. What this gate stopped excluding on "
            "2026-08-19 is the constitutive pair of a COMPLEX beta run — it is now "
            "LAUNCHED in the constitutive leg on all four complex cases and "
            "composed in the engine leg, so all four sub-steps of an unfolded beta "
            "run of either storage are on the device"),
            "counters": dict(TALLY)})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
