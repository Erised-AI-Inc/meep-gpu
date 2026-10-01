"""Byte gate for the Metal NO-PML CONSTITUTIVE family.

THIS FAMILY SHIPS NO KERNEL, so this gate is shaped differently from the certified
PML one and the difference is the point rather than a shortcut. There is no
kernel-vs-reference sweep because there is no kernel; there is no contraction-guard
leg because there is no source to build in two modes; there is no launch counter
because there is no launch. The certifying leg is an IDENTITY leg — ``stepping.update_H``
/ ``stepping.update_E`` on a fully seeded ``Fields`` with an inert layer must leave
EVERY live array byte-unchanged — and its control is the same assertion on
configurations the predicate refuses, where it must FAIL.

**A NULL AGREEING WITH A NO-OP IS TRIVIALLY IDENTICAL.** That is this family's
characteristic failure and every leg here is built around it. On the Triton track
the pass condition "the step MOVED STATE" caught a live row that had been
certifying two frozen states. So: every identity case carries a nonzero-word floor
and a subnormal census floor; every control must be shown to actually WRITE; the
engine leg's driven run must differ from its source-free control; and the plan's
``runs`` counter — which is what a null family has instead of a launch count — is a
pass condition, not a statistic.

THE LEGS:

  identity      the covered arm: seed, N full cycles, byte-compare AFTER EVERY
                cycle, census floors
  controls      active PML and stored E: bytes MUST move and the predicate MUST refuse
  breadth       fold / complex / Bloch / beta / BFAST / cylindrical / conductivity /
                non-contiguous / chi3 / unrecognised pole — admitted here, refused
                by a Metal KERNEL predicate NAMING the dropped clause, identical on
                each; plus the wall CONTROL and the totality assertion over
                METAL_NON_CLAUSES
  over_coverage the OTHER direction: every volume poisoned and the sub-step proved
                to touch none (with three refused configurations as the controls
                that MUST trip it), the cited return sites READ OUT OF stepping.py,
                every sibling clause on every admitted configuration mapped to a
                declared non-clause key (unmapped FAILS), and the arm gate proved
                both to change no verdict and to actually be there
  overlap       no two registered arms admit one slot on any configuration
  residency     THE METAL LEG: the measured false refusal and its fix, with the
                unallocated demanded set COUNTED per side
  precondition  the subnormal census, and why its shape differs on this family
  mutations     armed, run-counted, three-valued, mutant-module sha256 asserted,
                and each row carries what its probe MEASURED
  engine        the substitution across N whole steps, with a source-free control

WHAT THE 2026-08-15 AUDIT CHANGED, because a gate that quietly grew teeth is worse
than one that never had them:

* the breadth leg asserted only ``not sibling.covered``. Measured: every grid in
  that sweep has an inactive absorber, so ``_grid_reasons`` clause 3 refuses it
  whatever the fold / complex / Bloch / beta / BFAST clause does — the assertion
  would have held with all five clauses deleted from the sibling. The universal
  clause is now subtracted and the row's NAMED clause must appear in the residual.
  That is the Metal analogue of the Triton suite's ``_residual()`` helper;
* ``METAL_NON_CLAUSES`` said the breadth leg iterated it. It did not: six of the
  eleven entries had no case anywhere. ``cylindrical_axis``, ``conductivity`` and
  ``layout`` are now grid cases; ``metal_backend``, ``residency_declaration`` and
  ``subnormal_policy`` are named checks; and the leg FAILS on an orphan entry;
* the ``metallic_walls`` row claimed "admitted by the curl too" while the same row
  recorded ``curl_predicate_refused: true``. ``metallic`` is inside
  ``COVERED_BOUNDARIES`` and exercises no omitted clause; it is now the labelled
  CONTROL that shows the named-clause assertion discriminates;
* the residency leg sampled three names and three files stated the count as FOUR of
  nine per side. Counted: SIX, both sides;
* the identity leg compared only after the last cycle;
* ``m1`` was labelled ``byte_visible`` while its probe read a boolean, and ``m2``'s
  reason claimed a residency refusal its probe never computed. Both probes now
  measure what their rows claim.

WHAT THE SECOND 2026-08-15 AUDIT PASS ADDED, and both are holes rather than polish:

* **the slot->side table was a byte-visible ARMING HOLE.** ``NULL_SLOTS`` was a
  second literal beside the imported ``NULL_SIDES``. Measured with the two entries
  swapped: ``launch.plan_step`` on a no-PML run with ``stores_E`` filled the
  ``update_E`` SLOT with a null plan whose own ``sub_step`` said ``update_H``, on a
  run where ``stepping.update_E`` moves 3240 words — the composition silently
  deleting a real sub-step. The 93-test suite and all eight legs stayed GREEN. The
  table is DERIVED now, ``slot_side_reasons`` refuses the crossing fail-closed, and
  ``m5`` is the armed mutation;
* **the non-clause table was INCOMPLETE and its totality assertion could not see
  it.** ``breadth`` asserts table -> case; an OMITTED entry has no key to orphan.
  Running the missing direction found three clauses the siblings fire on
  configurations this family ADMITS with no entry at all — ``chi2/chi3 is
  installed`` (H side), ``<volume> is not allocated`` and
  ``kind 'sellmeier' is outside ('lorentzian','drude')``. ``over_coverage`` runs
  clause -> table on every cut and FAILS on anything unmapped;
* the engine leg cited ``driver.py:3212-3225`` for the injection slot; those lines
  are inside ``FdtdDriver.step``'s own DOCSTRING. The calls are at :3282 / :3289 /
  :3293 / :3304 and the injection slot is :3283-3284;
* every leg declared ``live=LIVE_STEP`` for every case including the WALLED ones,
  where ``launch.live_sub_steps`` derives six entries. No verdict moved — which is
  why it was invisible — and the artifact now states the derived set.

WHAT IS METAL-SPECIFIC HERE, and it is the reason this gate exists beside the
Triton family's rather than being satisfied by it:

* **the residency leg.** Tranche 1 composed this family through
  ``launch.plan_step`` and its residency verdict REFUSED the one configuration the
  family exists for, demanding mirrors of ``Hx`` and ``f_w_Hx`` — arrays a no-PML
  run does not allocate. Recorded as ``residency_covered: false`` on the
  ``no_pml_no_storage`` row of ``results/metal_pml_2026-08-14/gate.json``, which
  passed because leg 8 recorded the field without asserting it. This leg asserts
  it, in both directions;
* **the subnormal claim inverts.** The MPS executor flushes float32 subnormals
  natively and exposes no lever. Every other family's precondition is "no subnormal
  is reachable"; this one's is "subnormals ARE present and nothing touched them",
  which is a direct measurement that no Metal code ran. See :func:`leg_precondition`
  for why ``SubnormalWindow``'s ``clean`` verdict is deliberately NOT the pass
  condition here;
* **the contraction guard is answered rather than skipped.** ``run(contract=...)``
  is accepted under every mode and launches nothing under all of them.

PREDICTED NULLS, recorded WITH reasons rather than dropped:

* the NON-POWER-OF-TWO COURANT is carried in every sweep and is a predicted null on
  the covered arm — that arm performs no floating-point operation, so no
  association or FMA hazard exists for a Courant to expose. It is load-bearing on
  the ``controls`` leg and the record says so there;
* the CONTRACTION GUARD is not measured: there is no source, so there is no
  variant of one, and ``variants`` is asserted EMPTY rather than compared;
* FLUX/DFT SAMPLE COUNTS are not applicable — no monitor is in this family's path.
  The engine leg asserts its own step count instead, which is the same guarantee
  for the thing it actually counts;
* the STEP BUDGET on the covered arm is unbounded because nothing rounds. This gate
  does not raise the composed whole-step budget and does not claim to.

STATED WEAKNESS, inherited and restated because it applies here too: there is NO
PTX-EQUIVALENT AUDIT on this executor. ``compile_shader`` exposes no disassembly.
It bites less on this family than on any other — nothing is compiled — but the
claim's shape is the same and the artifact says so.

Usage (this host is both target and oracle; no device is needed at all)::

    python -u gate_metal_no_pml_constitutive.py \\
        --out parity/meep_gpu/results/metal_no_pml_constitutive_<date>/gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep. Requested EXPLICITLY before anything
# resolves a policy, and stamped into the artifact — even though this family
# reaches no executor at all, because "the policy was never consulted" and "the
# policy was consulted and admitted" are different facts and the artifact should
# not be able to confuse them.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import arms as metal_arms  # noqa: E402
from meep_gpu.metal_kernels import coverage as metal_coverage  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import no_pml_constitutive as family  # noqa: E402
from meep_gpu.metal_kernels import preconditions, subnormal  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


# ---------------------------------------------------------------------------
# Scaffolding — SELF-CONTAINED, and that is a deliberate choice with a reason
# ---------------------------------------------------------------------------
#
# ``metal_gate_kit.py`` exists beside this file and carries the same shapes. This
# gate does NOT import it, and the reason is not taste: the kit is shared,
# actively being extended by the families landing alongside this one, and a gate
# whose helpers move underneath it cannot be re-run to reproduce its own artifact.
# A byte gate's whole value is that re-running it later gives the same verdict on
# the same tree, so its scaffolding is pinned in the file whose sha256 the
# artifact records. The duplication is ~60 lines and is stated rather than hidden.
#
# THE VACUITY DISCIPLINE IS NOT IN THESE HELPERS. They make the checks one call;
# they cannot make them happen. Every leg below still floors its own census.

def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic rewrite: tmp + fsync + os.replace, after EVERY case (progress reporting)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def words(array: Any) -> Any:
    """One array as uint32 WORDS. complex64 yields two words per cell."""
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    return contiguous.astype(np.float32, copy=False).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    """THE comparison; there is no other. Never ``allclose``."""
    return int(np.count_nonzero(words(left) != words(right)))


def total_differing(left: Dict[str, Any], right: Dict[str, Any]) -> int:
    return sum(differing(left[name], right[name]) for name in sorted(left))


def per_array_differing(left: Dict[str, Any],
                        right: Dict[str, Any]) -> Dict[str, int]:
    counts = {name: differing(left[name], right[name]) for name in sorted(left)}
    return {name: count for name, count in counts.items() if count}


def nonzero_words(state: Dict[str, Any]) -> int:
    """Words that are not +0.0 — the vacuity denominator for a null family."""
    return sum(int(np.count_nonzero(words(array))) for array in state.values())


def state_digest(state: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(words(state[name]).tobytes())
    return digest.hexdigest()


def vacuity_floor(label: str, measured: int, floor: int = 1, why: str = "") -> None:
    """A census of zero is VACUOUS, not passed."""
    if measured < floor:
        raise AssertionError(
            f"VACUOUS: {label} measured {measured}, floor {floor}. "
            + (why or "the class this case claims to cover was never constructed"))


class Counter:
    """A call-counted wrapper — the DISARMED classification's evidence.

    On a kernel family this wraps the compiled entry point. Here there is no
    launch, so it wraps the probe: the count is what makes "the mutant ran and the
    verdict did not flip" a different statement from "nothing happened".
    """

    __slots__ = ("function", "runs")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.runs = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.runs += 1
        return self.function(*args, **kwargs)


def needle(source: str, old: str, new: str) -> str:
    """Plant one defect, or raise so the row reports NEEDLE-MISSED."""
    if old not in source:
        raise LookupError(old)
    return source.replace(old, new)


def classify(missed: bool, ran: int, runs: int, caught: int) -> str:
    if missed and ran == 0:
        return "NEEDLE-MISSED"
    if runs == 0:
        return "DISARMED"
    return f"CAUGHT {caught}/{ran}"


def assert_mutation(label: str, verdict: str, caught: int, ran: int,
                    must_catch: Optional[bool]) -> None:
    """DISARMED and NEEDLE-MISSED are failures under EVERY ``must_catch`` value."""
    if verdict == "DISARMED":
        raise AssertionError(
            f"{label} never ran: a mutation leg that certifies a defect it did "
            f"not execute is a hollow pass")
    if verdict == "NEEDLE-MISSED":
        raise AssertionError(
            f"{label}: the transform matched nothing in the shipped source, so "
            f"the defect it names was never planted")
    if must_catch is True and not (caught == ran and ran > 0):
        raise AssertionError(f"{label} was not caught: caught={caught} ran={ran}")
    if must_catch is False and caught:
        raise AssertionError(f"{label} was caught but must not be: {caught}")


class ModuleMutator:
    """Compile a text-edited copy of a module into a FRESH module object.

    THE MUTATION SEAM FOR A FAMILY THAT SHIPS NO KERNEL. A kernel family plants
    its defect in a shader source; a family whose whole product is a PREDICATE has
    to plant it in Python, and the hazard is the mirror image of a stale kernel
    cache — importing the shipped module under the mutant's name, so the "mutant"
    silently runs the certified code. Closed the same way a stale binary is: the
    mutant's source sha256 must DIFFER, and every row asserts it before counting.
    """

    __slots__ = ("path", "name", "source", "sha256")

    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name
        # Explicit UTF-8. A locale-dependent open killed the whole Triton mutation
        # battery on every host that had initialised a GPU (CUDA context init
        # resets LC_CTYPE to ASCII). Cheap insurance on any platform.
        with open(path, "r", encoding="utf-8") as handle:
            self.source = handle.read()
        self.sha256 = hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    def build(self, transform: Any, label: str) -> Tuple[Any, str]:
        mutated = transform(self.source)
        digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
        if digest == self.sha256:
            raise LookupError(f"{label}: the transform produced identical source")
        module = types.ModuleType(f"{self.name}__mutant__{label}")
        module.__file__ = self.path
        module.__package__ = self.name.rpartition(".")[0]
        module.__source_sha256__ = digest
        exec(compile(mutated, f"{self.path}#{label}", "exec"),  # noqa: S102
             module.__dict__)
        return module, digest


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def provenance(files: Dict[str, str], out_dir: str) -> Dict[str, str]:
    record = {label: sha256_of(path) for label, path in files.items()}
    save({"sources": record}, os.path.join(out_dir, "provenance.json"))
    return record


def environment() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch  # noqa: PLC0415

        record["torch"] = str(torch.__version__)
        record["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 - this family certifies without torch
        record["torch"] = None
        record["mps_available"] = False
        record["torch_import_error"] = repr(exc)
    try:
        from meep_gpu.metal_kernels.device import metal_frontend_version  # noqa: PLC0415

        record["metal_frontend"] = metal_frontend_version()
    except Exception:  # noqa: BLE001
        record["metal_frontend"] = None
    return record


#: "This host cannot arbitrate the claim", distinct from 1 ("it failed"). Not
#: reachable on this family — every leg runs anywhere — and defined anyway, so the
#: convention is one convention across the track.
CANNOT_CERTIFY = 75


def run_legs(legs, payload: Dict[str, Any], out: str, wanted, required) -> int:
    """Run the requested legs, write the summary, return the exit code.

    THE LEG SET IS PART OF THE CONTRACT. ``--legs`` is for debugging and must not
    be able to mint a green artifact: a run that skipped a required leg records
    ``status: partial`` and exits :data:`CANNOT_CERTIFY`, naming what is missing.
    """
    started = time.time()
    ran: List[str] = []
    for name, leg in legs:
        if wanted and name not in wanted:
            log(f"[skip] leg {name} (not in --legs)")
            continue
        log(f"=== LEG {name} ===")
        leg(payload, out)
        ran.append(name)
    missing = [name for name in required if name not in ran]
    payload["summary"] = dict(payload.get("summary", {}))
    payload["summary"].update({
        "legs_run": ran, "legs_required": list(required),
        "legs_missing": missing,
        "status": "passed" if not missing else "partial",
        "elapsed_s": round(time.time() - started, 1)})
    save(payload, out)
    if missing:
        log(f"CANNOT CERTIFY: required legs did not run: {missing} -> {out}")
        return CANNOT_CERTIFY
    log(f"METAL NO-PML CONSTITUTIVE GATE PASSED in "
        f"{payload['summary']['elapsed_s']}s -> {out}")
    return 0


#: NON-POWER-OF-TWO FIRST. Predicted null on the covered arm (nothing rounds);
#: load-bearing on `controls`, where the dsigw accumulation really runs.
COURANT_NP2 = 0.35
COURANT_P2 = 0.5

STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

LIVE_STEP = ("step_B", "update_H", "step_D", "update_E")

#: MULTIPLE FULL CYCLES, not one call. A sub-step that wrote back exactly what it
#: read would pass a single-call identity test and drift on the second.
IDENTITY_CYCLES = 4

#: The engine leg's stated step budget. Unbounded in principle on this arm —
#: nothing rounds — so this number bounds the MEASUREMENT, not the claim.
ENGINE_STEPS = 24


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build(cell=(1.2, 1.0, 0.9), boundaries="periodic", courant=COURANT_NP2,
          pml_cells=0, seed=20260815, complex_fields=False, symmetry=(),
          cylindrical=False, m=0, k_point=(0.0, 0.0, 0.0), beta=0.0,
          bfast=(0.0, 0.0, 0.0), dimensions=3, storage=False, layer=True,
          pml_storage=False, conductivity=None, noncontiguous=(), nonlinear=False,
          poles=()):
    """A real Grid/Fields/PML triple, seeded in three deliberate classes.

    ``layer=False`` builds a run with NO ``PML`` OBJECT AT ALL. An inert layer and
    an absent one are DIFFERENT configurations reaching the same ``stepping``
    return, and every row records which it was rather than conflating them.

    ``conductivity`` and ``noncontiguous`` exist for the ``breadth`` leg's last two
    non-clauses, and both are applied AFTER the seed so the state is the same one
    every other case carries. ``noncontiguous`` rebinds a volume to a reversed VIEW
    of its own buffer: the bytes are unchanged, ``_volume_reasons`` reports "is not
    C-contiguous", and ``stepping`` returns before forming an index into it — which
    is exactly the claim the ``layout`` non-clause makes.

    ``nonlinear`` and ``poles`` are the two ADDED 2026-08-15, for the
    ``nonlinearity`` and ``susceptibility_kind`` non-clauses. ``nonlinear`` goes
    through ``Fields.set_nonlinear_volumes`` DELIBERATELY and not through
    ``driver.set_chi2/set_chi3``: the driver route calls ``enable_field_storage``
    (driver.py:2016-2022) and would refuse both sides on ``stores_E``, which is a
    different measurement from the one the entry claims.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=courant, k_point=k_point,
                symmetry=symmetry, cylindrical=cylindrical, m=m, beta=beta,
                bfast_scaled_k=bfast, xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    pml = None
    if layer:
        thickness = tuple(
            (pml_cells, pml_cells) if grid.shape[a] >= 2 * pml_cells + 2 else (0, 0)
            for a in range(3))
        pml = PML(grid=grid, thickness=thickness)
        if pml_cells or pml_storage:
            fields.enable_pml_storage()
    if storage:
        fields.enable_field_storage()
    seed_state(fields, grid, seed, complex_fields)
    if nonlinear:
        fields.set_nonlinear_volumes({n: 0.0 for n in ("Ex", "Ey", "Ez")},
                                     {n: 0.4 for n in ("Ex", "Ey", "Ez")})
    for state in tuple(poles):
        fields.polarizations.append(state)
    if conductivity is not None:
        fields.set_d_conductivity(
            np.full(grid.shape, np.float32(conductivity), dtype=np.float32))
    for name in tuple(noncontiguous):
        setattr(fields, name, getattr(fields, name)[::-1])
    return grid, fields, pml


class UnrecognisedPole:
    """A registered susceptibility of a kind no kernel in this package transcribes.

    Drives NOTHING, deliberately: a pole that drove an E component would be refused
    by the null's own clause 3 (``polarization 0 drives Ez``), and the entry being
    measured is ``susceptibility_kind``, not the driving clause. With ``driven()``
    empty the null admits BOTH sides while every sibling refuses on 9b, which is
    exactly the omission the table now names.
    """

    class _Kind:
        kind = "sellmeier"

    susceptibility = _Kind()

    def driven(self):
        return ()

    def drives(self, component):
        return False


class QuietLorentzianPole(UnrecognisedPole):
    """A pole of a COVERED kind that drives nothing — the sharper of the two.

    ``UnrecognisedPole`` trips clause 9b (the KIND) and the E-side registration
    clause at once, so it cannot show which one is being measured. This one is
    ``lorentzian``: 9b is silent, and the only thing left for the sibling to refuse
    is that a polarization is REGISTERED AT ALL. That is the clause this family
    deliberately does not carry — it asks ``driven()`` instead, because a pole whose
    sigma is identically zero drives nothing and does not switch storage on
    (driver.py:1542-1547), so ``stepping.update_E`` really does return at :954 with a
    pole in the register.
    """

    class _Kind:
        kind = "lorentzian"

    susceptibility = _Kind()


def seed_state(fields, grid, seed: int, complex_fields: bool) -> None:
    """Ordinary normals, a ``+-0`` lattice, and DENSE SUBNORMALS.

    All three classes are pass conditions somewhere below. The subnormals are the
    pointed one on this backend: the MPS executor flushes them natively, so a
    byte-identical result WITH a nonzero census is a direct measurement that no
    Metal code touched these arrays.
    """
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
        flat = host.reshape(-1)
        flat[::17] = np.float32(-0.0)
        flat[7::23] = np.float32(0.0)
        flat[3::11] = np.float32(1e-40)
        flat[5::13] = np.float32(-3e-41)
        if complex_fields:
            array[...] = host + 1j * host[::-1, ::-1, ::-1]
        else:
            array[...] = host


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def census_of(state: Dict[str, Any]) -> int:
    return sum(subnormal.census(array) for array in state.values())


def negative_zeros_of(state: Dict[str, Any]) -> int:
    return sum(subnormal.signed_zero_census(array)["negative_zero"]
               for array in state.values())


def _residency() -> Any:
    """An EMPTY mirror registry — a real answer, unlike an undeclared one."""
    from meep_gpu.metal_kernels.device import Residency

    return Residency()


def live_for(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The live sub-step set for THIS configuration, DERIVED rather than declared.

    Every leg used to hand ``LIVE_STEP`` — the four heavy slots — to every case,
    including the WALLED ones. Measured 2026-08-15: on a metallic grid
    ``launch.live_sub_steps`` derives SIX entries, adding ``zero_metal_B`` and
    ``zero_metal_D``, so those rows recorded a live set the run did not have. No
    verdict moves (this family mirrors nothing, so nothing can go stale across a
    wall clear) — which is exactly why it was invisible, and exactly why an
    artifact that states it must state the measured one.

    ``sources=()`` is DECLARED, not inferred: ``live_sub_steps`` returns ``None``
    for an undeclared source list rather than guessing, and these fixtures really
    carry no source. A ``None`` here would be a harness defect, so it raises.
    """
    live = metal_launch.live_sub_steps(fields, pml, ())
    if live is None:
        raise AssertionError(
            "live_sub_steps could not answer for this fixture; a leg that declares "
            "a live set it did not derive is stating something it did not measure")
    return live


# ---------------------------------------------------------------------------
# The case matrix
# ---------------------------------------------------------------------------

IDENTITY_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "inert_layer_np2", "kwargs": dict(courant=COURANT_NP2)},
    {"name": "inert_layer_p2", "kwargs": dict(courant=COURANT_P2)},
    {"name": "no_layer_at_all", "kwargs": dict(layer=False)},
    {"name": "metallic_walls", "kwargs": dict(boundaries="metallic")},
    {"name": "one_cell_z", "kwargs": dict(cell=(1.2, 1.0, 0.0), dimensions=2)},
)

#: Admitted HERE and refused by a Metal KERNEL predicate NAMING THE CLAUSE this
#: family drops. Both halves are pass conditions, and the second half is the one
#: that was missing: asserting only ``not sibling.covered`` was VACUOUS, because
#: every grid in this sweep has an inactive absorber and is therefore already
#: refused by ``_grid_reasons`` clause 3 ("no active PML layer") — plus the
#: residency clause and six "is not allocated" clauses — whatever the fold,
#: complex, Bloch, beta or BFAST clause does. Measured 2026-08-15: the residual
#: refusal set minus clause 3 is non-empty on every case, but nothing checked that
#: it contained the clause the row CLAIMS. It does now.
#:
#: ``non_clause`` keys each row to its :data:`family.METAL_NON_CLAUSES` entry, and
#: the leg asserts the mapping is TOTAL — a table entry with no case here and no
#: named check below fails the leg.
#:
#: ``sibling`` names WHICH predicate carries the clause, because they are not all in
#: one: the conductivity clause is ``pml_curl_coverage``'s clause 8 and fires on
#: ``step_D`` only (a D conductivity is not on a B target), while every
#: ``_grid_reasons`` clause reaches both. Pointing the conductivity row at
#: ``constitutive_coverage`` would have recorded a refusal that predicate never
#: makes.
BREADTH_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "fold_y", "non_clause": "fold",
     "kwargs": dict(cell=(1.2, 1.6, 1.4), symmetry=("y",)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "is folded by a mirror plane",
     "refused_elsewhere": "coverage._grid_reasons clause 5 (a fold changes n_a)"},
    {"name": "complex_storage", "non_clause": "storage_width",
     "kwargs": dict(complex_fields=True),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "force_complex_fields=True",
     "refused_elsewhere": "coverage._grid_reasons clause 2 (complex64 storage)"},
    {"name": "bloch_k", "non_clause": "bloch_phase",
     "kwargs": dict(complex_fields=True, k_point=(0.4, -1.3, 0.7)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "is not exactly zero",
     "refused_elsewhere": "coverage._grid_reasons clause 7 (k != 0)"},
    {"name": "beta_kz", "non_clause": "beta",
     "kwargs": dict(cell=(1.2, 1.0, 0.0), dimensions=2, beta=0.2,
                    complex_fields=True),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "special_kz beta=",
     "refused_elsewhere": "coverage._grid_reasons clause 12 (beta != 0)"},
    {"name": "bfast", "non_clause": "bfast",
     "kwargs": dict(cell=(0.4, 0.4, 1.2), bfast=(0.31, 0.0, 0.0)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "BFAST is active",
     "refused_elsewhere": "coverage._grid_reasons clause 11 (BFAST)"},
    # ADDED 2026-08-15. `cylindrical_axis` and `conductivity` were entries in
    # METAL_NON_CLAUSES with no case anywhere in this gate or the suite, so the
    # table's own "an entry with no case is a claim nothing measured" rule was
    # being broken by the table.
    {"name": "cylindrical_m0", "non_clause": "cylindrical_axis",
     "kwargs": dict(complex_fields=True, cylindrical=True, m=0, dimensions=2,
                    cell=(1.2, 0.0, 1.4)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "cylindrical (Dcyl) coordinates are not carried",
     "refused_elsewhere": "coverage._grid_reasons clause 6 (Dcyl + the r=0 axis)"},
    {"name": "d_conductivity", "non_clause": "conductivity",
     "kwargs": dict(conductivity=0.5),
     "sibling": ("curl", "step_D"),
     "sibling_clause": "a conductivity is installed on Dx",
     "refused_elsewhere": ("coverage.pml_curl_coverage clause 8, on step_D only — "
                           "a D conductivity is not installed on a B target")},
    {"name": "noncontiguous_Bx", "non_clause": "layout",
     "kwargs": dict(noncontiguous=("Bx",)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "Bx is not C-contiguous",
     "refused_elsewhere": ("coverage._layout_reasons — contiguity is what a flat "
                           "index assumes, and this arm computes none")},
    # ADDED 2026-08-15, and these two are the ones the table was MISSING rather
    # than merely not exercising. Both fire on a configuration this family ADMITS.
    {"name": "chi3_h_side_only", "non_clause": "nonlinearity",
     "kwargs": dict(nonlinear=True), "sides": ("H",),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "chi2/chi3 is installed",
     "refused_elsewhere": ("coverage._grid_reasons clause 10, on EVERY sub-step. "
                           "THE ONE SIDED ROW: the null admits H (update_H returns "
                           "at :916-917 whatever is installed) and REFUSES E, "
                           "conservatively — CONSERVATIVE_CLAUSES["
                           "'chi2_chi3_installed_without_stored_e']"),
     "why_the_other_side_is_refused": (
         "the E side's product for those components is nonlinear_update_e's Pade "
         "factor; the null is byte-exact there only because stores_E happens to be "
         "False, which is a fact about the Fields object rather than about the "
         "nonlinearity, so the clause is kept and labelled conservative")},
    {"name": "unrecognised_pole", "non_clause": "susceptibility_kind",
     "kwargs": dict(poles=(UnrecognisedPole(),)),
     "sibling": ("constitutive", "H"),
     "sibling_clause": "kind 'sellmeier' is outside",
     "refused_elsewhere": ("coverage._susceptibility_reasons clause 9b — the ADE "
                           "recurrence is transcribed for lorentzian and drude "
                           "only, and no recurrence runs on this arm")},
    {"name": "quiet_lorentzian_pole", "non_clause": "susceptibility_registration",
     "kwargs": dict(poles=(QuietLorentzianPole(),)),
     "sibling": ("constitutive", "E"),
     "sibling_clause": "a susceptibility is registered",
     "refused_elsewhere": ("coverage.constitutive_coverage's E-side registration "
                           "clause. THIS FAMILY ASKS driven() INSTEAD: a pole whose "
                           "sigma is identically zero drives nothing, does not "
                           "switch storage on (driver.py:1542-1547) and reaches the "
                           "return at stepping.py:983, so refusing on registration "
                           "alone would be a refusal that is FALSE where it fires")},
    # THE CONTROL, and it is now labelled as one. It used to claim
    # "admitted by the curl too", which the row's own `curl_predicate_refused: true`
    # contradicted in the same artifact: `metallic` is INSIDE COVERED_BOUNDARIES, so
    # a walled grid exercises NO omitted clause at all. It is kept because a leg
    # that only ever asserts "the named clause is present" cannot show the assertion
    # discriminates; this row is the one where there is no named clause to find, and
    # the leg asserts the residual refusal set carries no grid-feature clause.
    {"name": "metallic_walls", "non_clause": None,
     "kwargs": dict(boundaries="metallic"),
     "sibling": ("constitutive", "H"),
     "sibling_clause": None,
     "refused_elsewhere": ("NOTHING — metallic walls are inside "
                           "COVERED_BOUNDARIES. Carried as the discriminating "
                           "control for the named-clause assertion")},
)

#: The ``METAL_NON_CLAUSES`` entries no GRID can exercise, each with the check in
#: this leg that does. A non-clause about the backend, the residency declaration or
#: the subnormal policy is not a property of a grid, so it gets a named check rather
#: than a case — and it is listed HERE rather than silently omitted, because the
#: whole point of the totality assertion is that an entry with no home fails.
NON_CLAUSE_CHECKS: Dict[str, str] = {
    "metal_backend": "no_torch_subprocess",
    "residency_declaration": "residency_undeclared",
    "subnormal_policy": "keep_policy",
    # `volume_allocation` gets a check rather than a grid case because NO grid can
    # single it out: it fires on EVERY no-PML configuration in the sweep at once
    # (18 distinct volume names), which is the same fact the residency leg counts
    # as six-of-nine per side. A "case" for it would be every case.
    "volume_allocation": "unallocated_volumes",
}

#: The covered path, run in a FRESH interpreter that has imported nothing else. The
#: claim "the only family in this package whose gate needs no device" is otherwise
#: unmeasurable from inside a process that has already imported torch to stamp its
#: own environment record — which this gate does, at ``environment()``.
#:
#: THE VERDICT IS A JSON LINE, PARSED EXACTLY, and that is not fussiness: the first
#: version of this check read ``"torch_imported=False" not in stdout``, and a probe
#: that armed it by appending a second print — leaving BOTH answers in the stream —
#: passed. A substring test over a whole stdout is satisfied by any line that says
#: the right thing, including one that another line contradicts.
NO_TORCH_PROGRAM = """
import json
import sys
sys.path.insert(0, {api_root!r})
import numpy as np
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import no_pml_constitutive as family
from meep_gpu.pml import PML

grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
            dimensions=3, courant=0.35, xp=np)
fields = Fields(grid=grid)
pml = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
step = family.plan_metal_null_constitutive_step(
    fields, pml, live=("step_B", "update_H", "step_D", "update_E"))
assert step.covered == ("update_H", "update_E"), step.covered
step.run()
step.run("fast")
runs = sum(plan.runs for plan in step.plans.values())
print("METAL_NO_TORCH " + json.dumps(
    {{"torch_imported": "torch" in sys.modules, "runs": runs,
      "covered": list(step.covered)}}))
"""

#: The one line :data:`NO_TORCH_PROGRAM` is allowed to answer with.
NO_TORCH_MARKER = "METAL_NO_TORCH "

#: The predicate must REFUSE each of these AND the array path must MOVE BYTES on
#: the named side. Both halves are pass conditions; either alone measures nothing.
CONTROL_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "active_pml_np2", "kwargs": dict(pml_cells=2, courant=COURANT_NP2),
     "must_move": ("H", "E")},
    {"name": "active_pml_p2", "kwargs": dict(pml_cells=2, courant=COURANT_P2),
     "must_move": ("H", "E")},
    {"name": "no_pml_stored_e", "kwargs": dict(storage=True), "must_move": ("E",)},
)

SIDE_SLOT = {"H": "update_H", "E": "update_E"}


def side_call(side: str, fields: Any, pml: Any) -> None:
    """Run the oracle for one side, resolved AT CALL TIME.

    THIS WAS ``SIDE_CALL = {"H": stepping.update_H, ...}`` — a dict of function
    OBJECTS captured at import. Every other oracle call site in this gate (ten of
    them: the identity leg, breadth, precondition, mutations, engine) spells
    ``stepping.update_H(...)`` and therefore resolves the attribute when it runs.
    The controls leg was the ONLY leg reading a snapshot, which is the stale-cached-
    pointer class, and it was the worst leg to have it: the control exists solely to
    prove the identity leg is not vacuous, so binding the two legs to potentially
    DIFFERENT function objects breaks the join those two halves are supposed to make.

    MEASURED 2026-08-15 rather than argued, by ``probe_metal_no_pml_gate_arming.py``
    injection ``f5_control_goes_quiet``: with ``stepping.update_H`` replaced by a
    no-op, this leg still reported ``moved={'H': 5176, 'E': 5215}`` and PASSED — a
    word count produced by a function that was no longer the oracle. The gate went
    red only later, in the mutations leg, on a different assertion. After this change
    the same injection fails HERE, on the floor that names the guarantee.

    The verdict on the certified tree does not move: nothing rebinds ``stepping`` in
    a real run, and ``leg_controls`` asserts the two spellings resolve to the same
    object before it uses either.
    """
    return getattr(stepping, SIDE_SLOT[side])(fields, pml)


# ---------------------------------------------------------------------------
# LEG identity
# ---------------------------------------------------------------------------

def leg_identity(payload: Dict[str, Any], out: str) -> None:
    """The covered arm: admit, run the NULL PLAN N cycles, byte-compare, census.

    THE FLOORS ARE PASS CONDITIONS, NOT STATISTICS. A zero-init ``Fields``
    satisfies "no bytes moved" trivially, so each case must prove it constructed
    the classes it claims: nonzero words, negative zeros, and subnormal words.

    Both the PLAN and the ARRAY PATH are run, from the same snapshot. The plan
    reproducing a no-op is only interesting if ``stepping`` is a no-op here too —
    that equivalence is the substitution's whole content, and asserting only one
    half would leave the other unmeasured.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(IDENTITY_CASES, start=1):
        _, fields, pml = build(**case["kwargs"])
        verdicts = {side: family.metal_null_constitutive_coverage(fields, pml, side)
                    for side in ("H", "E")}
        before = snapshot(fields)
        nonzero = nonzero_words(before)
        census = census_of(before)
        zeros = negative_zeros_of(before)

        # COMPARED AFTER EVERY CYCLE, not only after the last. Keeping the final
        # comparison alone would let a defect that moved bytes on cycle 1 and moved
        # them back on cycle 4 pass; the budget is the claim's extent, so every
        # point inside it is a pass condition.
        live = live_for(fields, pml)
        step = family.plan_metal_null_constitutive_step(
            fields, pml, residency=_residency(), live=live)
        plan_diff: Dict[str, int] = {}
        for cycle in range(IDENTITY_CYCLES):
            step.run()
            plan_diff = per_array_differing(before, snapshot(fields))
            assert not plan_diff, (case["name"], "plan", cycle, plan_diff)

        # The array path itself, from the same state.
        array_diff: Dict[str, int] = {}
        for cycle in range(IDENTITY_CYCLES):
            stepping.update_H(fields, pml)
            stepping.update_E(fields, pml)
            array_diff = per_array_differing(before, snapshot(fields))
            assert not array_diff, (case["name"], "array", cycle, array_diff)

        row = {
            "case": case["name"],
            "courant": case["kwargs"].get("courant", COURANT_NP2),
            "np2_courant": case["kwargs"].get("courant", COURANT_NP2) != COURANT_P2,
            "pml_object_present": pml is not None,
            "stores_E": bool(fields.stores_E),
            "arrays_compared": len(before),
            "nonzero_words": nonzero,
            "negative_zero_words": zeros,
            "subnormal_words": census,
            "cycles": IDENTITY_CYCLES,
            # DERIVED through launch.live_sub_steps, not the four-slot literal every
            # case used to get: a walled grid really runs zero_metal_B/zero_metal_D
            # and the row now says so.
            "live_sub_steps": list(live),
            "covered": {side: v.covered for side, v in verdicts.items()},
            "plan_runs": {slot: plan.runs for slot, plan in step.plans.items()},
            "plan_differing": plan_diff,
            "array_path_differing": array_diff,
            "residency_covered": bool(step.residency.covered),
            "predicted_null": (
                "the non-power-of-two Courant is carried and is a PREDICTED NULL "
                "on this arm: no floating-point operation runs, so no association "
                "or FMA hazard exists for it to expose. It bites on `controls`"),
        }
        rows.append(row)
        log(f"[identity] case {index}/{len(IDENTITY_CASES)} {case['name']:<18} "
            f"arrays={len(before):<2} nonzero={nonzero:<6} subnormal={census:<5} "
            f"-0={zeros:<5} plan_runs={sum(row['plan_runs'].values())} "
            f"{'IDENTICAL' if not plan_diff and not array_diff else 'DIFFERS'} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["identity"] = rows
        save(payload, out)

        assert all(v.covered for v in verdicts.values()), (case["name"], verdicts)
        vacuity_floor(f"{case['name']} nonzero words", nonzero, 1,
                          "a zero-init Fields satisfies 'no bytes moved' trivially, "
                          "which is THE failure mode for a null family")
        vacuity_floor(f"{case['name']} subnormal words", census, 1,
                          "the subnormal class is what proves no Metal code ran")
        vacuity_floor(f"{case['name']} negative zeros", zeros, 1)
        vacuity_floor(f"{case['name']} plan runs",
                          sum(row["plan_runs"].values()), 2 * IDENTITY_CYCLES,
                          "a byte comparison is trivially satisfied by a plan that "
                          "was never invoked")
        assert not plan_diff, (case["name"], plan_diff)
        assert not array_diff, (case["name"], array_diff)
        assert step.residency.covered, step.residency.reasons


# ---------------------------------------------------------------------------
# LEG controls
# ---------------------------------------------------------------------------

def leg_controls(payload: Dict[str, Any], out: str) -> None:
    """The non-vacuity control. BOTH halves are pass conditions.

    An identity leg with no failing control is decorative: "no bytes moved" is
    satisfied by a harness that never called anything. So each control asserts the
    predicate REFUSES *and* that ``stepping`` really does write on the named side.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    # NO IMPORT-TIME SNAPSHOT OF THE ORACLE MAY SURVIVE IN THIS MODULE. Comparing
    # ``getattr(stepping, name)`` to ``stepping.update_H`` would be a TAUTOLOGY —
    # both resolve at the same instant — so the check that means something is
    # structural: scan this gate's own globals for any callable that IS one of
    # ``stepping``'s functions. Such a global is a cached pointer by definition, and
    # it is what ``SIDE_CALL`` used to be. The live evidence that the remaining
    # binding is late is the arming probe's ``f5_control_goes_quiet``, which must
    # break THIS leg.
    oracle_functions = {getattr(stepping, name) for name in SIDE_SLOT.values()}
    cached = sorted(name for name, value in globals().items()
                    if callable(value) and value in oracle_functions)
    assert not cached, ("this gate holds an import-time snapshot of the oracle; "
                        "the controls leg would certify a different function from "
                        "the identity leg", cached)

    for index, case in enumerate(CONTROL_CASES, start=1):
        _, fields, pml = build(**case["kwargs"])
        refusals = {}
        moved = {}
        for side in case["must_move"]:
            verdict = family.metal_null_constitutive_coverage(fields, pml, side)
            refusals[side] = list(verdict.reasons)
            before = snapshot(fields)
            side_call(side, fields, pml)
            moved[side] = total_differing(before, snapshot(fields))
            assert not verdict.covered, (case["name"], side)

        row = {"case": case["name"], "must_move": list(case["must_move"]),
               "moved_words": moved, "refusal_reasons": refusals,
               "oracle_cached_globals": cached,
               "oracle_resolved": {side: getattr(stepping, SIDE_SLOT[side]).__module__
                                   + "." + getattr(stepping, SIDE_SLOT[side]).__qualname__
                                   for side in case["must_move"]},
               "courant": case["kwargs"].get("courant", COURANT_NP2)}
        rows.append(row)
        log(f"[controls] case {index}/{len(CONTROL_CASES)} {case['name']:<18} "
            f"moved={moved} REFUSED ({time.time() - started:.1f}s)")
        payload["legs"]["controls"] = rows
        save(payload, out)
        for side, count in moved.items():
            vacuity_floor(f"{case['name']}/{side} control", count, 1,
                              "the control moved NO bytes, so the identity leg it "
                              "exists to make non-vacuous is still unproven")


# ---------------------------------------------------------------------------
# LEG breadth
# ---------------------------------------------------------------------------

#: The clause every sibling predicate carries on EVERY case in this sweep, because
#: every case has an inactive absorber. Subtracted before the residual is read, the
#: same way the Triton suite's ``_residual()`` subtracts "array module is not cupy"
#: — without it the partition question answers itself.
UNIVERSAL_SIBLING_CLAUSE = "no active PML layer"

#: The grid-feature clauses ``_grid_reasons`` can report. Used only by the
#: ``metallic_walls`` control, which must carry NONE of them.
GRID_FEATURE_MARKERS = ("force_complex_fields", "mirror plane", "folded by",
                        "cylindrical", "r = 0 axis", "k_point", "BFAST",
                        "special_kz beta", "chi2/chi3", "boundary")


def _sibling_verdict(kind: str, argument: str, fields: Any, pml: Any) -> Any:
    """The Metal KERNEL predicate a breadth row names, called."""
    if kind == "constitutive":
        return metal_coverage.constitutive_coverage(fields, pml, argument, object())
    if kind == "curl":
        return metal_coverage.pml_curl_coverage(fields, pml, argument, object())
    raise ValueError(kind)


def _no_torch_subprocess() -> Dict[str, Any]:
    """Run the whole covered path in a fresh interpreter; did it import torch?

    EXACTLY ONE MARKED LINE IS ACCEPTED, and zero or two is a failure rather than a
    parse to be recovered from. A stream carrying two answers is a stream in which
    the assertion below can be satisfied by whichever one it happens to find first.
    """
    import subprocess  # noqa: PLC0415

    program = NO_TORCH_PROGRAM.format(api_root=API_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        timeout=300, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    stdout = completed.stdout.strip()
    marked = [line for line in stdout.splitlines()
              if line.startswith(NO_TORCH_MARKER)]
    verdict: Optional[Dict[str, Any]] = None
    if len(marked) == 1:
        try:
            verdict = json.loads(marked[0][len(NO_TORCH_MARKER):])
        except ValueError:
            verdict = None
    return {"returncode": completed.returncode, "stdout": stdout,
            "stderr": completed.stderr.strip()[-2000:],
            "marked_lines": len(marked),
            "verdict": verdict,
            # UNPARSEABLE IS A FAILURE, not an unknown: `None` here reads as
            # "torch was imported" so a broken harness cannot mint a pass.
            "torch_imported": True if verdict is None else bool(
                verdict.get("torch_imported", True)),
            "plan_runs": None if verdict is None else verdict.get("runs")}


def leg_breadth(payload: Dict[str, Any], out: str) -> None:
    """The one claim this family makes that no other predicate here makes.

    Admitted on a fold, complex storage, a Bloch phase, beta, BFAST, a cylindrical
    axis, a conductivity and a non-contiguous volume — and byte-identical on each —
    while a Metal KERNEL predicate REFUSES each one NAMING THE CLAUSE this family
    drops.

    THE NAMED-CLAUSE HALF IS WHAT MAKES THE LEG NON-VACUOUS, and it was not here.
    Asserting ``not sibling.covered`` is satisfied on every one of these grids by
    ``_grid_reasons`` clause 3 alone — they all have an inactive absorber — so the
    old assertion would have held even if the fold, Bloch and BFAST clauses had all
    been deleted from the sibling. :data:`UNIVERSAL_SIBLING_CLAUSE` is subtracted
    and the named clause is looked for in what remains.

    THE ``metallic_walls`` ROW IS THE CONTROL for that assertion: metallic is inside
    ``COVERED_BOUNDARIES``, so it exercises no omitted clause, and the leg asserts
    its residual carries NO grid-feature clause. Without it "the named clause was
    found" could be a marker that matches everything.

    THE TOTALITY ASSERTION closes the table's own rule: every
    :data:`family.METAL_NON_CLAUSES` key must be exercised by a case here or named
    in :data:`NON_CLAUSE_CHECKS`, and the three checks in that mapping RUN.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(BREADTH_CASES, start=1):
        _, fields, pml = build(**case["kwargs"])
        # THE SIDES THIS ROW CLAIMS, declared, because one row legitimately claims
        # only ONE. `chi3_h_side_only` admits H and is REFUSED on E, and a leg that
        # demanded both would either have to drop the row or weaken the assertion
        # for every other row. Declared per row, asserted per row, and the refused
        # side is asserted REFUSED rather than left unmentioned.
        claimed = tuple(case.get("sides", ("H", "E")))
        covered = {side: family.metal_null_constitutive_coverage(
            fields, pml, side).covered for side in ("H", "E")}
        sibling_kind, sibling_argument = case["sibling"]
        sibling = _sibling_verdict(sibling_kind, sibling_argument, fields, pml)
        residual = [reason for reason in sibling.reasons
                    if UNIVERSAL_SIBLING_CLAUSE not in reason]
        wanted = case["sibling_clause"]
        names_clause = (None if wanted is None
                        else any(wanted in reason for reason in residual))

        before = snapshot(fields)
        live = live_for(fields, pml)
        step = family.plan_metal_null_constitutive_step(
            fields, pml, residency=_residency(), live=live)
        diff: Dict[str, int] = {}
        for cycle in range(IDENTITY_CYCLES):
            step.run()
            stepping.update_H(fields, pml)
            stepping.update_E(fields, pml)
            diff = per_array_differing(before, snapshot(fields))
            assert not diff, (case["name"], cycle, diff)
        nonzero = nonzero_words(before)

        row = {"case": case["name"], "covered": covered,
               "sides_claimed": list(claimed),
               "sides_refused": [s for s in ("H", "E") if s not in claimed],
               "why_the_other_side_is_refused": case.get(
                   "why_the_other_side_is_refused"),
               "non_clause": case["non_clause"],
               "refused_elsewhere": case["refused_elsewhere"],
               "sibling_predicate": f"{sibling_kind}:{sibling_argument}",
               "sibling_refused": not sibling.covered,
               "sibling_names_the_clause": names_clause,
               "sibling_clause_sought": wanted,
               "sibling_residual_reasons": residual,
               "arrays_compared": len(before), "nonzero_words": nonzero,
               "cycles": IDENTITY_CYCLES, "differing": diff,
               "live_sub_steps": list(live),
               "plan_runs": {s: p.runs for s, p in step.plans.items()}}
        rows.append(row)
        log(f"[breadth] case {index}/{len(BREADTH_CASES)} {case['name']:<18} "
            f"covered={all(covered.values())} sibling_refused={not sibling.covered} "
            f"names_clause={names_clause} nonzero={nonzero:<6} "
            f"{'IDENTICAL' if not diff else 'DIFFERS'} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["breadth"] = rows
        save(payload, out)

        for side in ("H", "E"):
            if side in claimed:
                assert covered[side], (case["name"], side, "claimed but refused")
            else:
                assert not covered[side], (
                    f"{case['name']}: side {side} is NOT claimed by this row and "
                    f"the predicate admitted it. Either the row is stale or the "
                    f"predicate stopped carrying a clause: "
                    f"{case.get('why_the_other_side_is_refused')}")
        assert not sibling.covered, (
            f"{case['name']}: the Metal KERNEL {sibling_kind} predicate ADMITTED "
            f"this grid, so the breadth claim is vacuous here")
        vacuity_floor(f"{case['name']} nonzero words", nonzero, 1)
        vacuity_floor(f"{case['name']} plan runs",
                      sum(row["plan_runs"].values()),
                      len(claimed) * IDENTITY_CYCLES)
        if wanted is not None:
            assert names_clause, (
                f"{case['name']}: the sibling refused, but not for the clause this "
                f"row claims ({wanted!r}). Every grid here is already refused by "
                f"{UNIVERSAL_SIBLING_CLAUSE!r}, so 'it refused' certifies nothing "
                f"about the clause this family drops. Residual: {residual}")
        else:
            leaked = [reason for reason in residual
                      if any(marker in reason for marker in GRID_FEATURE_MARKERS)]
            assert not leaked, (
                f"{case['name']} is the CONTROL and must exercise no omitted "
                f"clause, but its residual carries {leaked}")

    # ---- the three non-clauses no grid can exercise -----------------------
    checks: Dict[str, Any] = {}

    subprocess_row = _no_torch_subprocess()
    checks["no_torch_subprocess"] = subprocess_row
    log(f"[breadth] check no_torch_subprocess -> verdict={subprocess_row['verdict']} "
        f"marked_lines={subprocess_row['marked_lines']} "
        f"rc={subprocess_row['returncode']}")

    _, plain_fields, plain_pml = build()
    undeclared_null = {
        side: family.metal_null_constitutive_coverage(
            plain_fields, plain_pml, side, residency=None).covered
        for side in ("H", "E")}
    undeclared_sibling = metal_coverage.constitutive_coverage(
        plain_fields, plain_pml, "H", None)
    checks["residency_undeclared"] = {
        "null_covered": undeclared_null,
        "sibling_refused": not undeclared_sibling.covered,
        "sibling_names_the_clause": any(
            "residency was not declared" in reason
            for reason in undeclared_sibling.reasons),
        "sibling_reasons": list(undeclared_sibling.reasons)}
    log(f"[breadth] check residency_undeclared -> null={undeclared_null} "
        f"sibling_refused={not undeclared_sibling.covered}")

    # `volume_allocation`: the sibling refuses a volume that is not allocated, this
    # arm binds none, and the run this family exists for does not allocate two
    # thirds of the names the refusal would list. Counted here, not sampled.
    unallocated = {
        slot: sorted(name for name in metal_coverage.sub_step_volumes(slot)
                     if getattr(plain_fields, name, None) is None)
        for slot in ("update_H", "update_E")}
    allocation_clauses = [reason for reason in undeclared_sibling.reasons
                          if "is not allocated" in reason]
    checks["unallocated_volumes"] = {
        "null_covered": {
            side: family.metal_null_constitutive_coverage(
                plain_fields, plain_pml, side).covered for side in ("H", "E")},
        "sibling_allocation_clauses": allocation_clauses,
        "sibling_allocation_clause_count": len(allocation_clauses),
        "unallocated_per_slot": unallocated,
        "why": ("every kernel predicate binds one pointer per volume and refuses "
                "an unallocated one; this arm binds none, and on its own "
                "configuration SIX of the nine volumes per side do not exist")}
    log(f"[breadth] check unallocated_volumes -> sibling clauses="
        f"{len(allocation_clauses)} unallocated per slot="
        f"{ {k: len(v) for k, v in unallocated.items()} }")

    keep_reasons = subnormal.mps_policy_reasons("keep")
    flush_reasons = subnormal.mps_policy_reasons("flush")
    checks["keep_policy"] = {
        "keep_refusal_reasons": list(keep_reasons),
        "flush_refusal_reasons": list(flush_reasons),
        "null_covered_regardless": {
            side: family.metal_null_constitutive_coverage(
                plain_fields, plain_pml, side).covered for side in ("H", "E")},
        "why": ("the clause that refuses every sibling under `keep` is shown to "
                "FIRE, and the null admits the same configuration because it has "
                "no operand, result or intermediate for a policy to apply to")}
    log(f"[breadth] check keep_policy -> keep_reasons={len(keep_reasons)} "
        f"flush_reasons={len(flush_reasons)}")

    covered_entries = {case["non_clause"] for case in BREADTH_CASES
                       if case["non_clause"]} | set(NON_CLAUSE_CHECKS)
    orphans = sorted(set(family.METAL_NON_CLAUSES) - covered_entries)
    payload["legs"]["breadth_non_clauses"] = {
        "table_entries": sorted(family.METAL_NON_CLAUSES),
        "exercised_by_a_grid_case": sorted(
            case["non_clause"] for case in BREADTH_CASES if case["non_clause"]),
        "exercised_by_a_named_check": NON_CLAUSE_CHECKS,
        "orphans": orphans,
        "checks": checks,
        "universal_sibling_clause_subtracted": UNIVERSAL_SIBLING_CLAUSE,
    }
    save(payload, out)

    assert not orphans, (
        f"METAL_NON_CLAUSES entries {orphans} have no case in this leg and no "
        f"named check: the table's own rule is that an entry with no case is a "
        f"claim nothing measured")
    # AND THE OTHER DIRECTION OF THE SAME JOIN: a case or check naming a key the
    # table does not carry is a case measuring a claim nobody made. Without it,
    # deleting an entry leaves both this assertion and `orphans` satisfied.
    unknown = sorted(covered_entries - set(family.METAL_NON_CLAUSES))
    assert not unknown, (
        f"these cases and checks name non-clause keys METAL_NON_CLAUSES does not "
        f"carry: {unknown}")
    assert subprocess_row["returncode"] == 0, subprocess_row
    assert subprocess_row["marked_lines"] == 1, (
        f"the no-torch subprocess emitted {subprocess_row['marked_lines']} marked "
        f"verdict lines; exactly one is the contract, because a stream carrying two "
        f"answers lets the assertion below pick the one it likes: "
        f"{subprocess_row['stdout']!r}")
    assert not subprocess_row["torch_imported"], (
        "the covered path IMPORTED TORCH in a fresh interpreter, so the "
        "`metal_backend` non-clause — 'this product is correct on a host with no "
        "GPU' — is false as written")
    assert subprocess_row["plan_runs"] == 4, (
        f"the fresh interpreter ran the covered path {subprocess_row['plan_runs']} "
        f"times; a subprocess that imported nothing because it DID nothing proves "
        f"nothing about the covered path")
    assert all(undeclared_null.values()), undeclared_null
    assert not undeclared_sibling.covered
    assert checks["residency_undeclared"]["sibling_names_the_clause"], (
        "the sibling refused an undeclared residency for some other reason, so the "
        "`residency_declaration` non-clause was not the thing measured")
    vacuity_floor("keep-policy refusal reasons", len(keep_reasons), 1,
                  "the clause the `subnormal_policy` non-clause is about did not "
                  "fire, so dropping it certifies nothing")
    assert not flush_reasons, flush_reasons
    assert all(checks["keep_policy"]["null_covered_regardless"].values())
    vacuity_floor("sibling allocation clauses", len(allocation_clauses), 1,
                  "the clause the `volume_allocation` non-clause is about did not "
                  "fire on the sibling, so dropping it certifies nothing")
    assert all(checks["unallocated_volumes"]["null_covered"].values())
    for slot, names in unallocated.items():
        assert len(names) == 6, (
            f"{slot}: {len(names)} of the nine volumes a kernel would bind are "
            f"unallocated on the family's own run, and every file that states this "
            f"number says SIX: {names}")


# ---------------------------------------------------------------------------
# LEG over_coverage — the direction the table's totality assertion did not run
# ---------------------------------------------------------------------------
#
# `leg_breadth` asserts table -> case: every METAL_NON_CLAUSES key has a home. That
# cannot find an omission, because an omitted entry has no key to be orphaned.
# MEASURED 2026-08-15 by running the missing direction by hand: the siblings fire
# FOURTEEN distinct clause families on configurations this family ADMITS and the
# table named ELEVEN. `chi2/chi3 is installed` (H side), `<volume> is not allocated`
# and `kind 'sellmeier' is outside (...)` had no entry at all.
#
# So this leg runs clause -> table: sweep the configurations, collect every reason
# both sibling predicates give on all four slots, map each through a DECLARED
# marker table, and FAIL on anything unmapped. An unmapped reason is a feature this
# family does not implement and has not written down.

#: reason substring -> the METAL_NON_CLAUSES key that explains dropping it, or one
#: of the two non-omission buckets below. Ordered longest-first at match time so a
#: specific marker wins over a generic one.
CLAUSE_MARKERS: Dict[str, str] = {
    "BFAST is active": "bfast",
    "force_complex_fields=True": "storage_width",
    "dtype complex64 is not float32": "storage_width",
    "is not C-contiguous": "layout",
    "!= grid shape": "layout",
    "is not three-dimensional": "layout",
    "exceeds the kernel's int32 index range": "layout",
    "is not allocated": "volume_allocation",
    "a conductivity is installed on": "conductivity",
    "conductivity is installed but condfac_for": "conductivity",
    "a mirror plane is active": "fold",
    "is folded by a mirror plane": "fold",
    # SUBSUMED, NOT OMITTED, and mapped by name rather than given an entry.
    # COVERED_BOUNDARIES is ('periodic','metallic') and the only other kinds
    # stepping._boundary_kinds can return are 'mirror' and 'axis'
    # (stepping.py:2191-2195), so the ghost clause never fires without the fold or
    # cylindrical clause beside it. The leg asserts that co-fire rather than
    # assuming it.
    "boundary 'mirror' is outside": "fold",
    "boundary 'axis' is outside": "cylindrical_axis",
    "cylindrical (Dcyl) coordinates": "cylindrical_axis",
    "is the cylindrical r = 0 axis": "cylindrical_axis",
    "chi2/chi3 is installed": "nonlinearity",
    "k_point": "bloch_phase",
    "special_kz beta=": "beta",
    "residency": "residency_declaration",
    "is outside ('lorentzian', 'drude')": "susceptibility_kind",
    # FOUND BY THIS LEG on the cut that introduced it. The sibling refuses on
    # REGISTRATION; this family asks driven(), because a pole whose sigma is
    # identically zero drives nothing, does not switch storage on
    # (driver.py:1542-1547) and really does reach the return at stepping.py:983.
    "a susceptibility is registered": "susceptibility_registration",
    "torch is not importable": "metal_backend",
    "torch.backends.mps": "metal_backend",
    "built with MPS": "metal_backend",
    "MPS device is available": "metal_backend",
    "compile_shader is missing": "metal_backend",
    "array module is": "metal_backend",
    "subnormal policy": "subnormal_policy",
}

#: Reasons that are NOT omissions and must not be counted as ones. The first is the
#: exact complement of this family's clause 2 and the second of its clause 3 — the
#: partition itself, positively named in the null's own predicate — so mapping them
#: onto a non-clause entry would record the family's own coverage set as a hole.
INVERSE_CLAUSES: Dict[str, str] = {
    "no active PML layer": "the null's clause 2, inverted: this family covers the "
                           "exact complement, and both are the same call to "
                           "pml.is_active",
    "E is recomputed from D rather than stored": (
        "the null's clause 3, inverted: the sibling requires stored E and this "
        "family requires it absent (stepping.py:983)"),
}

#: The volumes the poison rebinds. ``scratch`` is in the list deliberately: the PML
#: constitutive path passes it to ``_apply_constitutive_pml``, so leaving it real
#: would give the ACTIVE-PML control one untouched array to work through.
POISON_VOLUMES = STATE + ("scratch",)


class PoisonedVolume:
    """Not ``None`` — so an allocation check passes — and unusable as an array.

    THE MEASUREMENT THE WHOLE NON-CLAUSE TABLE RESTS ON. "A sub-step that returns
    before its first statement reads no pointer, forms no index and rounds no
    float" was a READING of stepping.py:944-945 / :983-984 and nothing executed it.
    Rebinding every allocated volume to this object and calling the real sub-step
    turns it into a measurement: if the call returns, the sub-step demonstrably
    touched no field array, and every clause about strides, indices, ghost rules
    and rounding modes is a clause about something that did not happen.

    ``shape`` and ``dtype`` are REAL class attributes so ``__getattr__`` does not
    intercept them — a predicate reading them is not the array access this is
    about. Everything else raises, including unknown attributes, because
    ``_apply_constitutive_pml`` reaches for ``xp`` before it multiplies and an
    AttributeError from that read is a weaker signal than a named one.
    """

    shape = (1, 1, 1)

    def __init__(self) -> None:
        self.dtype = np.dtype(np.float32)

    @staticmethod
    def _touched(how: str):
        raise AssertionError(f"the sub-step TOUCHED a field array ({how})")

    def __getattr__(self, name):
        self._touched(f"attribute {name!r}")

    def __getitem__(self, key):
        self._touched("__getitem__")

    def __setitem__(self, key, value):
        self._touched("__setitem__")

    def __array__(self, *args, **kwargs):
        self._touched("__array__")

    def __mul__(self, other):
        self._touched("__mul__")

    __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __mul__

    def __iter__(self):
        self._touched("__iter__")


#: (label, build kwargs, side, must the poisoned call RAISE?). The controls are what
#: make the first two rows a measurement rather than a tautology: on a configuration
#: this family REFUSES, the same poison must be detected immediately.
POISON_CASES = (
    ("covered_no_pml_H", dict(), "H", False),
    ("covered_no_pml_E", dict(), "E", False),
    ("covered_no_layer_H", dict(layer=False), "H", False),
    ("covered_no_layer_E", dict(layer=False), "E", False),
    ("CONTROL_active_pml_H", dict(pml_cells=2), "H", True),
    ("CONTROL_active_pml_E", dict(pml_cells=2), "E", True),
    ("CONTROL_stored_e_E", dict(storage=True), "E", True),
)

#: The configurations the clause sweep runs over. Every BREADTH case plus the ones
#: that carry a clause no breadth grid does — a B-side conductivity fires clause 8
#: on ``step_B`` where the D-side row fires it on ``step_D``.
SWEEP_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
    (case["name"], case["kwargs"]) for case in BREADTH_CASES) + (
    ("plain_inert", dict()),
    ("no_layer", dict(layer=False)),
    ("one_cell_z", dict(cell=(1.2, 1.0, 0.0), dimensions=2)),
)


def _map_reason(reason: str) -> Optional[str]:
    """Which METAL_NON_CLAUSES key explains dropping this sibling clause, if any.

    LONGEST MARKER FIRST, so a specific marker wins over a generic one. STATED
    BOUND: this is substring matching, so a NEW sibling clause that happens to
    contain an existing marker would be mapped rather than flagged. What the leg
    catches is therefore an unrecorded clause that shares no wording with a
    recorded one — which is the case that actually occurs, and is how all four of
    the missing entries were found. It does not catch a clause disguised as
    another; nothing short of a structural clause id would, and the siblings do not
    carry one.
    """
    for marker in sorted(CLAUSE_MARKERS, key=len, reverse=True):
        if marker in reason:
            return CLAUSE_MARKERS[marker]
    return None


def leg_over_coverage(payload: Dict[str, Any], out: str) -> None:
    """Everything this family does NOT implement, enumerated by MEASUREMENT.

    Three parts, and the first is what licenses the other two:

    1. ``poison`` — every allocated volume rebound to an object that raises on any
       access, and the real ``stepping`` sub-steps called. The covered arm must
       RETURN; three refused configurations must RAISE;
    2. ``clause_sweep`` — every reason both sibling predicates give on every
       configuration the null admits, mapped to a declared non-clause key. An
       unmapped reason FAILS the leg: it is a feature this family does not carry
       and has not written down, which is the omission the table's own
       (one-directional) totality assertion structurally cannot see;
    3. ``gate_is_an_optimisation`` — the ``absorber_inactive`` arm gate is claimed
       to decide only whether the arm is CONSULTED, never what it answers. Measured
       by computing every arm's verdict with the gate honoured and with it forced
       open, and asserting the ADMITTED set is identical on every configuration
       while the CONSULTED count really does differ.
    """
    started = time.time()

    # ---- 1. the poison ----------------------------------------------------
    poison_rows: List[Dict[str, Any]] = []
    for index, (label, kwargs, side, must_raise) in enumerate(POISON_CASES, start=1):
        _, fields, pml = build(**kwargs)
        verdict = family.metal_null_constitutive_coverage(fields, pml, side)
        replaced = []
        for name in POISON_VOLUMES:
            if getattr(fields, name, None) is not None:
                setattr(fields, name, PoisonedVolume())
                replaced.append(name)
        raised: Optional[str] = None
        try:
            side_call(side, fields, pml)
        except Exception as exc:  # noqa: BLE001 - the control's whole point
            raised = f"{type(exc).__name__}: {exc}"
        row = {"case": label, "side": side, "null_covered": bool(verdict.covered),
               "volumes_poisoned": len(replaced), "poisoned": replaced,
               "must_raise": must_raise, "raised": raised,
               "touched_a_field_array": raised is not None}
        poison_rows.append(row)
        log(f"[over_coverage] poison {index}/{len(POISON_CASES)} {label:<22} "
            f"poisoned={len(replaced):<2} "
            f"{'TOUCHED (' + str(raised)[:40] + ')' if raised else 'RETURNED clean'}")
        payload["legs"]["over_coverage_poison"] = poison_rows
        save(payload, out)
        vacuity_floor(f"{label} poisoned volumes", len(replaced), 1,
                      "no volume was rebound, so the call had nothing to be caught "
                      "touching and this row measures nothing")
        if must_raise:
            assert raised is not None, (
                f"{label}: the poison was NOT detected on a configuration this "
                f"family REFUSES. The covered rows above are then a tautology — "
                f"they would pass against a poison nothing can trip")
            assert not verdict.covered, (label, verdict.reasons)
        else:
            assert raised is None, (
                f"{label}: the covered sub-step TOUCHED a field array ({raised}). "
                f"Every entry in METAL_NON_CLAUSES rests on it not doing that")
            assert verdict.covered, (label, verdict.reasons)

    # ---- 1b. the CITED return sites, READ OUT OF stepping.py -------------
    #
    # The artifact stamps `family.return_sites` — "stepping.py:944-945" and
    # ":954-955" — and nothing checked them. This round has already found TWO stale
    # citations in this tranche (driver.py:3212-3225 pointing into a docstring, and
    # the same one corrected once already in the family module), so a cited line
    # number that no leg reads is exactly the class of claim that rots. Read the
    # cited span out of the oracle and require it to END in a bare `return` inside
    # the function the side names.
    with open(os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
              "r", encoding="utf-8") as handle:
        oracle_lines = handle.read().splitlines()
    citation_rows: List[Dict[str, Any]] = []
    for side, spec in family.NULL_SIDES.items():
        site = spec["return_site"]
        span = site.rsplit(":", 1)[1]
        first, last = (int(part) for part in span.split("-"))
        cited = oracle_lines[first - 1:last]
        enclosing = next(
            (line for line in reversed(oracle_lines[:first])
             if line.startswith("def ")), "")
        row = {"side": side, "sub_step": spec["sub_step"], "return_site": site,
               "cited_lines": cited, "enclosing_def": enclosing.split("(")[0]}
        citation_rows.append(row)
        log(f"[over_coverage] citation {site} -> {enclosing.split('(')[0]} :: "
            f"{cited[-1].strip()[:60]}")
        assert cited, (site, "the cited span is empty")
        assert cited[-1].strip().startswith("return"), (
            f"{site} is cited as {spec['sub_step']}'s no-PML return and its last "
            f"line is {cited[-1].strip()!r}. A citation no leg reads is how "
            f"driver.py:3212-3225 survived a round pointing into a docstring")
        assert enclosing == f"def {spec['sub_step']}" or enclosing.startswith(
            f"def {spec['sub_step']}("), (
            f"{site} is inside {enclosing.split('(')[0]!r}, not "
            f"{spec['sub_step']!r}")
    payload["legs"]["over_coverage_citations"] = citation_rows
    save(payload, out)

    # THE CONTROLS ARE A PASS CONDITION OF THEIR OWN, not a row count. Without them
    # the four covered rows are a tautology: a poison nothing can trip returns clean
    # everywhere. A leg that ran with its controls removed measured nothing.
    controls_raised = sum(1 for row in poison_rows
                          if row["must_raise"] and row["raised"])
    vacuity_floor("poison controls that were detected", controls_raised, 3,
                  "the covered rows are only evidence beside controls that PROVE "
                  "the poison is detectable, and fewer than three ran")

    # ---- 2. clause -> table ----------------------------------------------
    sweep_rows: List[Dict[str, Any]] = []
    unmapped: Dict[str, List[str]] = {}
    mapped_keys: Dict[str, List[str]] = {}
    ghost_without_a_partner: List[str] = []
    for index, (name, kwargs) in enumerate(SWEEP_CASES, start=1):
        _, fields, pml = build(**kwargs)
        covered = {side: family.metal_null_constitutive_coverage(
            fields, pml, side).covered for side in ("H", "E")}
        if not any(covered.values()):
            continue  # Nothing is dropped where nothing is admitted.
        reasons = set()
        for kind, argument in (("constitutive", "H"), ("constitutive", "E"),
                               ("curl", "step_B"), ("curl", "step_D")):
            reasons.update(_sibling_verdict(kind, argument, fields, pml).reasons)
        inverse = sorted(r for r in reasons
                         if any(m in r for m in INVERSE_CLAUSES))
        remaining = sorted(r for r in reasons if r not in set(inverse))
        keys: Dict[str, List[str]] = {}
        for reason in remaining:
            key = _map_reason(reason)
            if key is None:
                unmapped.setdefault(name, []).append(reason)
            else:
                keys.setdefault(key, []).append(reason)
                mapped_keys.setdefault(key, []).append(name)
        # The subsumption claim, asserted rather than assumed.
        ghosts = [r for r in remaining if "boundary" in r and "is outside" in r]
        if ghosts and not ({"fold", "cylindrical_axis"} & set(keys)):
            ghost_without_a_partner.append(name)
        row = {"case": name, "null_covered": covered,
               "sibling_reasons": len(remaining),
               "inverse_clauses": inverse,
               "non_clause_keys": sorted(keys),
               "unmapped": unmapped.get(name, []),
               "ghost_rule_clauses": ghosts}
        sweep_rows.append(row)
        log(f"[over_coverage] sweep {index}/{len(SWEEP_CASES)} {name:<20} "
            f"reasons={len(remaining):<3} keys={sorted(keys)} "
            f"unmapped={len(unmapped.get(name, []))}")
        payload["legs"]["over_coverage_sweep"] = sweep_rows
        save(payload, out)

    # ---- 3. the arm gate is an optimisation, not a verdict ----------------
    gate_rows: List[Dict[str, Any]] = []
    for name, kwargs in OVERLAP_CASES:
        _, fields, pml = build(**kwargs)
        context = metal_arms.StepContext(fields, pml, _residency(), ("off",))
        gated_admitters: Dict[str, List[str]] = {}
        open_admitters: Dict[str, List[str]] = {}
        consulted = {"gated": 0, "open": 0}
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            for spec in metal_arms.registered(slot):
                admits = spec.coverage(context, slot).covered
                open_admitters.setdefault(slot, [])
                gated_admitters.setdefault(slot, [])
                consulted["open"] += 1
                if admits:
                    open_admitters[slot].append(spec.label)
                if spec.gate is not None and not spec.gate(context):
                    continue
                consulted["gated"] += 1
                if admits:
                    gated_admitters[slot].append(spec.label)
        gate_rows.append({"case": name, "consulted": consulted,
                          "admitters_with_the_gate": gated_admitters,
                          "admitters_with_the_gate_forced_open": open_admitters,
                          "identical": gated_admitters == open_admitters})
        log(f"[over_coverage] gate  {name:<24} consulted={consulted} "
            f"identical={gate_rows[-1]['identical']}")
    payload["legs"]["over_coverage_arm_gate"] = gate_rows
    save(payload, out)

    payload["legs"]["over_coverage_summary"] = {
        "poison_cases": len(poison_rows),
        "poison_controls_that_raised": sum(
            1 for row in poison_rows if row["must_raise"] and row["raised"]),
        "sweep_cases": len(sweep_rows),
        "non_clause_keys_exercised_by_the_sweep": sorted(mapped_keys),
        "table_entries": sorted(family.METAL_NON_CLAUSES),
        "unmapped_reasons": unmapped,
        "inverse_clauses_subtracted": INVERSE_CLAUSES,
        "ghost_rule_subsumption": (
            "COVERED_BOUNDARIES is ('periodic','metallic') and the only other kinds "
            "stepping._boundary_kinds can return are 'mirror' and 'axis' "
            "(stepping.py:2191-2195), so the ghost clause is SUBSUMED by fold / "
            "cylindrical_axis rather than being a separate omission. Asserted, not "
            "assumed: a ghost clause with neither partner fails this leg"),
        "arm_gate_cases": len(gate_rows),
        "arm_gate_is_an_optimisation": all(row["identical"] for row in gate_rows),
        "predicted_null": (
            "forcing the arm gate open changes NO verdict — that is the recorded "
            "PREDICTED NULL, and it is what makes `absorber_inactive` an "
            "optimisation rather than a second predicate. The leg still asserts "
            "the consulted COUNT differs, or the gate would be measurably absent"),
        "elapsed_s": round(time.time() - started, 2),
    }
    save(payload, out)

    assert not unmapped, (
        f"these sibling clauses fire on a configuration this family ADMITS and no "
        f"METAL_NON_CLAUSES entry explains dropping them: {unmapped}. That is an "
        f"UNRECORDED non-clause — the omission the table's table->case totality "
        f"assertion structurally cannot see")
    # THE JOIN, and without it this leg maps clauses onto a table of its OWN and
    # never touches the shipped one. Measured by the arming probe's
    # `f7_non_clause_table_shrunk`: deleting `volume_allocation` from
    # METAL_NON_CLAUSES left the gate GREEN, because CLAUSE_MARKERS still named it.
    # A marker whose key the table does not carry is a clause explained by nothing.
    homeless = sorted(set(CLAUSE_MARKERS.values()) - set(family.METAL_NON_CLAUSES))
    assert not homeless, (
        f"CLAUSE_MARKERS maps sibling clauses onto {homeless}, which "
        f"METAL_NON_CLAUSES does not carry: the mapping would explain a dropped "
        f"clause with an entry that does not exist")
    fired = sorted(set(mapped_keys) - set(family.METAL_NON_CLAUSES))
    assert not fired, (
        f"these keys were produced by the sweep and are absent from "
        f"METAL_NON_CLAUSES: {fired}")
    assert not ghost_without_a_partner, (
        f"{ghost_without_a_partner}: a ghost-rule clause fired with neither the "
        f"fold nor the cylindrical clause beside it, so it is NOT subsumed and "
        f"needs an entry of its own")
    vacuity_floor("sweep cases that admit", len(sweep_rows), 1,
                  "no configuration in the sweep was admitted, so no clause was "
                  "dropped and this leg measured nothing")
    for key in ("fold", "storage_width", "bloch_phase", "beta", "bfast",
                "cylindrical_axis", "conductivity", "layout", "nonlinearity",
                "volume_allocation", "susceptibility_kind",
                "susceptibility_registration"):
        vacuity_floor(f"sweep coverage of {key}", len(mapped_keys.get(key, ())), 1,
                      f"no configuration in the sweep fired the clause "
                      f"{key!r} names, so its entry is unmeasured here")
    for row in gate_rows:
        assert row["identical"], (
            f"{row['case']}: forcing the arm gate open CHANGED the admitted set. "
            f"`absorber_inactive` is documented as deciding only whether the arm is "
            f"CONSULTED — 'the predicate always wins' — and it just decided an "
            f"answer: {row['admitters_with_the_gate']} vs "
            f"{row['admitters_with_the_gate_forced_open']}")
    assert any(row["consulted"]["gated"] < row["consulted"]["open"]
               for row in gate_rows), (
        "the gate never suppressed a consultation anywhere in the sweep, so 'the "
        "arm gate is an optimisation' is a claim about a gate that is not there")


# ---------------------------------------------------------------------------
# LEG overlap
# ---------------------------------------------------------------------------

OVERLAP_CASES = (
    ("inert", dict()),
    ("inert_no_layer", dict(layer=False)),
    ("active_pml", dict(pml_cells=2)),
    ("stored_e", dict(storage=True)),
    ("walls", dict(boundaries="metallic")),
    ("walls_active_pml", dict(boundaries="metallic", pml_cells=2)),
    ("pml_storage_inert_layer", dict(pml_storage=True)),
)


def leg_overlap(payload: Dict[str, Any], out: str) -> None:
    """Every registered arm's verdict on every configuration; at most one admits.

    THE PARTITION IS BY CONSTRUCTION — the null's clause 2 and every kernel
    predicate's clause 3 are the same call to ``pml.is_active`` — and it is
    MEASURED rather than asserted, because "by construction" is exactly the kind of
    claim that stops being true when a clause moves.

    The row also records where NOTHING admits, which is a real and correct answer
    (the array path) and must not be read as a defect.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, (name, kwargs) in enumerate(OVERLAP_CASES, start=1):
        _, fields, pml = build(**kwargs)
        context = metal_arms.StepContext(fields, pml, _residency(), ("off",))
        admitters: Dict[str, List[str]] = {}
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            admitting = []
            for spec in metal_arms.registered(slot):
                if spec.gate is not None and not spec.gate(context):
                    continue
                if spec.coverage(context, slot).covered:
                    admitting.append(f"{spec.family}/{spec.label}")
            admitters[slot] = admitting

        row = {"case": name, "admitters": admitters,
               "unfilled_slots": [slot for slot, a in admitters.items() if not a]}
        rows.append(row)
        log(f"[overlap] case {index}/{len(OVERLAP_CASES)} {name:<24} "
            f"{ {k: len(v) for k, v in admitters.items()} } "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["overlap"] = rows
        save(payload, out)
        for slot, admitting in admitters.items():
            assert len(admitting) <= 1, (
                f"{name}: {slot} is co-admitted by {admitting}; two products on "
                f"one slot is the over-covering dispatch the composer fails "
                f"closed on")

    # NON-VACUITY: the null arm must have admitted SOMEWHERE in this sweep, or the
    # leg consulted it on nothing and certified nothing about it.
    admitted_anywhere = sum(
        1 for row in rows for admitting in row["admitters"].values()
        if any(label.startswith(family.FAMILY) for label in admitting))
    vacuity_floor("null arm admissions across the overlap sweep",
                      admitted_anywhere, 1,
                      "the arm was consulted on every configuration and admitted on "
                      "none, which certifies nothing about it")
    payload["legs"]["overlap_summary"] = {"null_arm_admissions": admitted_anywhere}
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG residency — the Metal leg
# ---------------------------------------------------------------------------

def leg_residency(payload: Dict[str, Any], out: str) -> None:
    """THE MEASURED DEFECT AND ITS FIX, asserted in both directions.

    Recorded in ``results/metal_pml_2026-08-14/gate.json`` as
    ``residency_covered: false`` on the ``no_pml_no_storage`` row — the one
    configuration this family exists for — and passed there because leg 8 recorded
    the field without asserting it.

    Three rows, and all three are pass conditions:

    1. ``planned`` — the tranche-1 declaration. MUST REFUSE, naming volumes a
       no-PML run does not allocate. If this stops refusing, the fix stopped
       being necessary and the leg should be re-derived rather than deleted;
    2. ``null`` — the fix. MUST HOLD;
    3. ``composed`` — the same question through ``launch.plan_step``, which is the
       path the engine would take. MUST HOLD, and the slots must actually be
       filled by the null arm rather than left empty.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    planned = metal_coverage.residency_coverage(
        mirrored=(), planned=("update_H", "update_E"), live=LIVE_STEP)
    # COUNTED, NOT SAMPLED. This used to probe three hand-picked names ("Hx",
    # "f_w_Hx", "f_w_Ex") and record whichever of them happened to be None, while
    # three files stated the count as FOUR of nine per side. Measured over the whole
    # demanded set: SIX of nine, both sides. A number nobody counts is a number that
    # drifts, so the leg counts it and the assertion below pins it.
    _, plain_fields, inert = build(layer=True)
    per_side: Dict[str, Dict[str, Any]] = {}
    for slot in ("update_H", "update_E"):
        demanded = metal_coverage.sub_step_volumes(slot)
        missing = [name for name in demanded
                   if getattr(plain_fields, name, None) is None]
        per_side[slot] = {"demanded": list(demanded), "demanded_count": len(demanded),
                          "unallocated": missing, "unallocated_count": len(missing)}
    unallocated = sorted({name for spec in per_side.values()
                          for name in spec["unallocated"]})
    rows.append({
        "declaration": "planned (tranche 1)", "covered": bool(planned.covered),
        "reasons": list(planned.reasons),
        "volumes_demanded_but_not_allocated": unallocated,
        "per_side": per_side,
        "note": ("H is never stored without PML (fields.py:664-666), E is served on "
                 "demand as D*inv_eps and is not stored either, and f_w_* is "
                 "allocated only by enable_pml_storage — so SIX of the nine names "
                 "demanded per side name arrays that do not exist. Only Bx/By/Bz "
                 "and Dx/Dy/Dz are allocated on such a run")})
    log(f"[residency] planned -> covered={planned.covered} unallocated per side "
        f"{ {slot: spec['unallocated_count'] for slot, spec in per_side.items()} }"
        f" of {per_side['update_H']['demanded_count']} "
        f"({time.time() - started:.1f}s)")

    null = metal_coverage.residency_coverage(
        mirrored=(), planned=(), live=LIVE_STEP, null=("update_H", "update_E"))
    rows.append({"declaration": "null (the fix)", "covered": bool(null.covered),
                 "reasons": list(null.reasons)})
    log(f"[residency] null    -> covered={null.covered}")

    # The stale direction: a null must not WAIVE a mirror an array-path sub-step
    # writes. Mirror `step_B`'s own volumes plus E, plan only `step_B`.
    mirrored = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez")
    stale = metal_coverage.residency_coverage(
        mirrored=mirrored, planned=("step_B",), live=LIVE_STEP)
    held = metal_coverage.residency_coverage(
        mirrored=mirrored, planned=("step_B",), live=LIVE_STEP,
        null=("update_H", "update_E"))
    rows.append({"declaration": "stale direction",
                 "undeclared_covered": bool(stale.covered),
                 "undeclared_reasons": list(stale.reasons),
                 "declared_null_covered": bool(held.covered)})
    log(f"[residency] stale   -> undeclared={stale.covered} declared={held.covered}")

    contradiction = metal_coverage.residency_coverage(
        mirrored=(), planned=("update_H",), live=LIVE_STEP, null=("update_H",))
    rows.append({"declaration": "planned AND null",
                 "covered": bool(contradiction.covered),
                 "reasons": list(contradiction.reasons)})

    _, fields, layer = build(layer=True)
    residency = _residency()
    composed = metal_launch.plan_step(fields, layer, residency=residency,
                                      sources=())
    rows.append({"declaration": "composed through launch.plan_step",
                 "replaces": list(composed.replaces),
                 "selected": dict(composed.selected),
                 "mirrors": list(residency.names),
                 "covered": bool(composed.residency.covered),
                 "reasons": list(composed.residency.reasons)})
    log(f"[residency] composed-> replaces={composed.replaces} "
        f"selected={composed.selected} covered={composed.residency.covered}")

    payload["legs"]["residency"] = rows
    save(payload, out)

    assert not planned.covered, (
        "the tranche-1 declaration NO LONGER REFUSES. That is not a pass: this "
        "leg pins the defect the null declaration exists to fix, and if the "
        "refusal is gone the model changed underneath it")
    assert unallocated, (
        "no demanded volume was unallocated, so the sharpest half of the finding "
        "— that the refusal named arrays that do not exist — was not reproduced")
    for slot, spec in per_side.items():
        assert spec["demanded_count"] == 9, (slot, spec)
        assert spec["unallocated_count"] == 6, (
            f"{slot}: {spec['unallocated_count']} of {spec['demanded_count']} "
            f"demanded volumes are unallocated, and every file that states this "
            f"number says SIX. Re-measure and correct the prose rather than "
            f"loosening this assertion: {spec['unallocated']}")
    assert null.covered, null.reasons
    assert not stale.covered, "a null must not waive a mirror the array path writes"
    assert held.covered, held.reasons
    assert not contradiction.covered
    assert composed.replaces == LIVE_STEP, composed.reasons
    assert composed.selected == {
        "step_B": "no-PML curl",
        "update_H": "no-PML null",
        "step_D": "no-PML curl",
        "update_E": "no-PML null",
    }, composed.selected
    assert composed.residency.covered, composed.residency.reasons


# ---------------------------------------------------------------------------
# LEG precondition
# ---------------------------------------------------------------------------

def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The subnormal census, and why its SHAPE is inverted on this family.

    Every other Metal family's claim is byte-identity subject to a CHECKED
    subnormal-free precondition: no operand, result or intermediate may land in
    the band, because the MPS executor flushes natively and cannot be told not to.

    THIS FAMILY HAS NO OPERAND, NO RESULT AND NO INTERMEDIATE. The sub-step returns
    at stepping.py:944-945 / :983-984 before reading an array, so the window over
    them is EMPTY BY CONSTRUCTION — and an empty window is VACUOUS, not clean, which
    is exactly what ``SubnormalWindow`` refuses to call a pass. So the verdict is
    NOT ``assert_clean_or_refuse``; the honest shape is recorded instead:

    * the OPERAND/INTERMEDIATE window is empty and is labelled
      ``vacuous_by_construction`` with the reason;
    * the STATE window is censused and must FIRE — subnormals must be PRESENT —
      and the state must be byte-unchanged across the whole budget anyway.

    That second pair is a stronger statement than it looks. Any Metal code that
    touched these arrays would flush every one of those words to zero. Byte
    equality with a nonzero census is a direct measurement that nothing on the
    device ran, on the one backend where that measurement is available for free.
    """
    started = time.time()
    _, fields, pml = build()
    before = snapshot(fields)

    window = preconditions.SubnormalWindow(first_step=0, last_step=IDENTITY_CYCLES)
    for name, array in sorted(before.items()):
        window.observe(name, array, step=0)

    step = family.plan_metal_null_constitutive_step(
        fields, pml, residency=_residency(), live=live_for(fields, pml))
    for cycle in range(1, IDENTITY_CYCLES + 1):
        step.run()
        stepping.update_H(fields, pml)
        stepping.update_E(fields, pml)
        for name, array in sorted(snapshot(fields).items()):
            window.observe(name, array, step=cycle)

    after = snapshot(fields)
    diff = per_array_differing(before, after)
    report = window.report()
    row = {
        "precondition_shape": "vacuous_by_construction",
        "why": ("the covered sub-step returns before reading an array, so the set "
                "of operands, results and intermediates is EMPTY. There is no "
                "subnormal-free precondition to check because there is nothing to "
                "check it over"),
        "operand_window_size": 0,
        "state_window": report,
        "state_subnormal_words_before": census_of(before),
        "state_subnormal_words_after": census_of(after),
        "negative_zero_words_before": negative_zeros_of(before),
        "negative_zero_words_after": negative_zeros_of(after),
        "differing": diff,
        "budget_steps": IDENTITY_CYCLES,
        "flush_claim": ("MPS flushes float32 subnormals natively and exposes no "
                        "lever; any Metal code touching these arrays would zero "
                        "every subnormal word. Byte equality with a nonzero census "
                        "measures that nothing on the device ran"),
        "ptx_equivalent_audit": None,
        "ptx_equivalent_audit_note": (
            "compile_shader exposes no disassembly. It bites less here than on any "
            "other family — nothing is compiled — but the claim's shape is the same"),
    }
    payload["legs"]["precondition"] = [row]
    save(payload, out)
    log(f"[precondition] state subnormals {row['state_subnormal_words_before']} -> "
        f"{row['state_subnormal_words_after']} subnormal_words={report['subnormal_words']} "
        f"{'IDENTICAL' if not diff else 'DIFFERS'} ({time.time() - started:.1f}s)")

    vacuity_floor("state subnormal census", census_of(before), 1,
                      "the class this leg exists to measure was never constructed")
    assert report["subnormal_words"] > 0, (
        "the state window did NOT fire. On this family firing is the DESIRED "
        "outcome — it means subnormals are present to be preserved — and a window "
        "that did not fire censused nothing")
    assert census_of(after) == census_of(before)
    assert not diff, diff


# ---------------------------------------------------------------------------
# LEG mutations
# ---------------------------------------------------------------------------

#: ``must_catch``: True must flip a verdict | None record only | False must not.
#:
#: EVERY MUTANT IS A MODULE, not a shader — this family compiles nothing. The
#: hazard is the mirror image of a stale kernel cache (importing the shipped
#: predicate under the mutant's name) and is closed the same way: the mutant's
#: source sha256 must DIFFER from the shipped one, asserted before the row counts.
#:
#: Classified ``byte_visible`` or ``verdict_only``. A null family's predicate
#: guards two different things and only one of them moves floats; conflating them
#: is how a "caught" row would claim a numerical defect it never had.
MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_admit_an_active_layer": {
        "must_catch": True,
        "kind": "byte_visible",
        "why": "the load-bearing clause: an active layer runs the dsigw "
               "accumulation, so a null there DELETES a real sub-step",
        "apply": lambda s: needle(
            s, "    return null_constitutive_coverage(fields, pml, side)",
            "    return Coverage(True, ())  # MUTANT: the shared predicate dropped"),
        "probe": ("active", dict(pml_cells=2)),
    },
    "m2_ignore_the_residency_null_split": {
        "must_catch": True,
        "kind": "verdict_only",
        "why": "declaring the null slots `planned` restores the tranche-1 false "
               "refusal on the family's own configuration",
        "apply": lambda s: needle(s, "performs_device_work = False",
                                      "performs_device_work = True"),
        "probe": ("residency", dict()),
    },
    "m3_drop_the_run_counter": {
        "must_catch": True,
        "kind": "verdict_only",
        "why": "the run counter is what a null family has instead of a launch "
               "count; without it 'the plan reproduced the array path' is "
               "satisfied by a plan that was never invoked",
        "apply": lambda s: needle(s, "        self.runs += 1",
                                      "        pass  # counter dropped"),
        "probe": ("counter", dict()),
    },
    "m4_report_a_contraction_variant_it_does_not_hold": {
        "must_catch": True,
        "kind": "verdict_only",
        "why": "an artifact showing ('off',) here claims a compiled guard that "
               "does not exist",
        "apply": lambda s: needle(s, "        return ()\n\n    @property\n"
                                         "    def volumes",
                                      "        return ('off',)\n\n    @property\n"
                                      "    def volumes"),
        "probe": ("variants", dict()),
    },
    # ADDED 2026-08-15, and it is the one this battery was MISSING. Measured with
    # the shipped literal swapped by hand: `launch.plan_step` on a no-PML stores_E
    # run filled the update_E SLOT with a null plan whose own sub_step said
    # update_H, on a run where stepping.update_E moves 3240 words — and the
    # 93-test suite and all eight legs stayed GREEN. The table is DERIVED from
    # NULL_SIDES now, so this needle edits the derivation, which is the only place
    # the crossing is still expressible.
    "m5_cross_the_slot_side_table": {
        "must_catch": True,
        "kind": "byte_visible",
        "why": "a null built for one sub-step and placed in the other's SLOT "
               "deletes a sub-step the array path performs; slot_side_reasons "
               "is what refuses it fail-closed",
        "apply": lambda s: needle(
            s, '    spec["sub_step"]: side for side, spec in NULL_SIDES.items()}',
            '    spec["sub_step"]: ("E" if side == "H" else "H")\n'
            '    for side, spec in NULL_SIDES.items()}  # MUTANT: crossed'),
        "probe": ("slot_side", dict(storage=True)),
    },
}


#: What each mutation probe MEASURED, filled in by :func:`_probe_mutant` so a row
#: classified ``byte_visible`` carries the words it would have cost rather than a
#: label. Keyed by mutation name; read by :func:`leg_mutations` into the row.
_PROBE_EVIDENCE: Dict[str, Any] = {}


def _probe_mutant(module: Any, kind: str, kwargs: Dict[str, Any]) -> bool:
    """Run one mutant against the shipped answer. True when the defect was CAUGHT."""
    _, fields, pml = build(**kwargs)
    if kind == "active":
        # THE SHIPPED PREDICATE REFUSES; A MUTANT THAT ADMITS IS CAUGHT — and the
        # row calls itself `byte_visible`, so the probe MEASURES the bytes rather
        # than leaving the classification to the label. On the configuration the
        # mutant admits, the array path's update_H really does run the dsigw
        # accumulation, and a null there would have deleted every one of those
        # words. That number is the row's evidence.
        try:
            admitted = bool(module.metal_null_constitutive_coverage(
                fields, pml, "H").covered)
        except Exception as exc:  # noqa: BLE001 - a raising mutant is a caught one
            _PROBE_EVIDENCE["active"] = {"raised": repr(exc)}
            return True
        before = snapshot(fields)
        stepping.update_H(fields, pml)
        moved = total_differing(before, snapshot(fields))
        _PROBE_EVIDENCE["active"] = {
            "mutant_admitted": admitted,
            "array_path_moved_words": moved,
            "byte_visibility": ("the words a null would have deleted on the "
                                "configuration the mutant admits")}
        if admitted and moved <= 0:
            raise AssertionError(
                "the mutant admitted a configuration on which the array path moved "
                "NO bytes, so this row cannot be called byte_visible")
        return admitted
    if kind == "residency":
        plan = module.plan_metal_null_constitutive(fields, pml, "H")
        if plan is None:
            _PROBE_EVIDENCE["residency"] = {"plan": None}
            return True
        # CARRIED THROUGH THE VERDICT, not read off the attribute. The row's `why`
        # claims the mutation "restores the tranche-1 false refusal"; reading
        # `performs_device_work` proves only that the flag moved. So the flag is
        # routed through exactly the split `launch.plan_step` performs, and the
        # residency verdict is what the row records.
        does_device_work = bool(getattr(plan, "performs_device_work", True))
        slots = ("update_H", "update_E")
        verdict = metal_coverage.residency_coverage(
            mirrored=(),
            planned=slots if does_device_work else (),
            live=LIVE_STEP,
            null=() if does_device_work else slots)
        _PROBE_EVIDENCE["residency"] = {
            "performs_device_work": does_device_work,
            "residency_covered": bool(verdict.covered),
            "residency_reasons": list(verdict.reasons),
            "shipped_answer": "performs_device_work False -> residency held"}
        return not verdict.covered
    if kind == "counter":
        plan = module.plan_metal_null_constitutive(fields, pml, "H")
        plan.run()
        plan.run()
        return plan.runs != 2
    if kind == "variants":
        plan = module.plan_metal_null_constitutive(fields, pml, "H")
        return plan.variants != ()
    if kind == "slot_side":
        # BYTE_VISIBLE, and the probe measures the bytes. On a no-PML run with
        # stores_E the E side is NOT null — stepping.update_E writes at :993 — so a
        # composition that puts the null in the update_E slot deletes every one of
        # those words. Caught iff the mutant does NOT end up filling that slot.
        before = snapshot(fields)
        stepping.update_E(fields, pml)
        moved = total_differing(before, snapshot(fields))
        try:
            step = module.plan_metal_null_constitutive_step(
                *build(**kwargs)[1:], residency=_residency(), live=LIVE_STEP)
            covered = tuple(step.covered)
            refusals = {slot: list(r) for slot, r in step.refusals.items()}
        except Exception as exc:  # noqa: BLE001 - a raising composer is a caught one
            _PROBE_EVIDENCE["slot_side"] = {"raised": repr(exc),
                                            "array_path_moved_words": moved}
            return True
        _PROBE_EVIDENCE["slot_side"] = {
            "mutant_covered_slots": list(covered),
            "mutant_refusals": refusals,
            "array_path_moved_words": moved,
            "shipped_answer": "('update_H',) — the E side is the unbuilt STORE arm",
            "byte_visibility": ("the words stepping.update_E writes on the run the "
                                "crossed table would have handed to a null")}
        if moved <= 0:
            raise AssertionError(
                "stepping.update_E moved NO bytes on the stores_E fixture, so this "
                "row cannot be called byte_visible")
        return "update_E" not in covered
    raise ValueError(kind)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Armed, RUN-COUNTED, three-valued, and the mutant's sha256 asserted different."""
    rows: List[Dict[str, Any]] = []
    started = time.time()
    path = os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                        "no_pml_constitutive.py")
    mutator = ModuleMutator(path, "meep_gpu.metal_kernels.no_pml_constitutive")

    for label, entry in MUTATIONS.items():
        missed = False
        caught = ran = launches = 0
        digest = None
        try:
            module, digest = mutator.build(entry["apply"], label)
        except LookupError:
            missed = True
            module = None
        evidence: Any = None
        if module is not None:
            probe_kind, probe_kwargs = entry["probe"]
            _PROBE_EVIDENCE.pop(probe_kind, None)
            counter = Counter(_probe_mutant)
            ran = 1
            caught = int(bool(counter(module, probe_kind, probe_kwargs)))
            launches = counter.runs
            evidence = _PROBE_EVIDENCE.get(probe_kind)

        verdict = classify(missed, ran, launches, caught)
        row = {"mutation": label, "verdict": verdict, "kind": entry["kind"],
               "must_catch": entry["must_catch"], "why": entry["why"],
               "ran": ran, "caught": caught, "runs_counted": launches,
               # WHAT THE PROBE MEASURED, not only what it concluded. A row that
               # calls itself `byte_visible` while its probe read a boolean is a
               # classification nothing supports; the `byte_visible` row now carries
               # the words the defect would have deleted, and the residency row
               # carries the verdict it flipped rather than the flag it read.
               "probe_evidence": evidence,
               "shipped_sha256": mutator.sha256, "mutant_sha256": digest,
               "sha256_differs": bool(digest and digest != mutator.sha256)}
        rows.append(row)
        log(f"[mutations] {label:<44} {verdict} kind={entry['kind']} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["mutations"] = rows
        save(payload, out)

        assert row["sha256_differs"], (
            f"{label}: the mutant's source hash equals the shipped one, so the "
            f"'mutant' was the certified module under another name")
        assert_mutation(label, verdict, caught, ran, entry["must_catch"])

    byte_visible = [row for row in rows if row["kind"] == "byte_visible"]
    payload["legs"]["mutations_summary"] = {
        "total": len(rows), "byte_visible": len(byte_visible),
        "verdict_only": len(rows) - len(byte_visible),
        "byte_visible_words": {
            row["mutation"]: (row["probe_evidence"] or {}).get(
                "array_path_moved_words") for row in byte_visible},
        "note": ("a null family's predicate guards two different things and only "
                 "one of them moves floats; the split is recorded so a caught row "
                 "cannot claim a numerical defect it never had — and the "
                 "byte_visible rows now carry the MEASURED word count rather than "
                 "resting on the label")}
    save(payload, out)
    for row in byte_visible:
        moved = (row["probe_evidence"] or {}).get("array_path_moved_words")
        vacuity_floor(f"{row['mutation']} byte visibility", int(moved or 0), 1,
                      "a row classified byte_visible whose probe measured no moved "
                      "words is a classification nothing supports")


# ---------------------------------------------------------------------------
# LEG engine
# ---------------------------------------------------------------------------

def leg_engine(payload: Dict[str, Any], out: str) -> None:
    """The substitution across N WHOLE STEPS, with a source-free control.

    Two runs from one seed: one stepped entirely by the array path, one with
    ``update_H``/``update_E`` replaced by the null plans. Every live array is
    byte-compared after EVERY step, not only at the end.

    THE SOURCE IS PROVEN NON-VACUOUS AGAINST A SOURCE-FREE CONTROL. Without a
    source a periodic no-PML run from a seeded state still evolves, but the run
    that is DRIVEN must differ from the one that is not — otherwise the injection
    did nothing and the "24 steps, bit-identical" claim is about a state that
    barely moved. Both runs must also MOVE STATE from their own initial snapshot;
    that is the pass condition that caught a live Triton row comparing two frozen
    states.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    def drive(substitute: bool, driven: bool) -> Tuple[Dict[str, Any], Dict[str, Any],
                                                       int, List[str]]:
        _, fields, pml = build()
        initial = snapshot(fields)
        step = family.plan_metal_null_constitutive_step(
            fields, pml, residency=_residency(), live=live_for(fields, pml))
        digests: List[str] = []
        for number in range(ENGINE_STEPS):
            stepping.step_B(fields, pml)
            if driven:
                # A deterministic magnetic injection between step_B and update_H,
                # which is exactly the slot the driver injects in
                # (driver.py:3283-3284, between the step_B call at :3282 and the
                # update_H call at :3289).
                #
                # THE CITATION WAS driver.py:3212-3225 FOR A ROUND, and those lines
                # are inside FdtdDriver.step's own DOCSTRING — the paragraph about
                # MEEP's two source slots — not its body. The family module had
                # already caught and corrected the same stale citation on
                # MetalNullConstitutiveStepPlan.run; the gate kept it. Re-measured
                # 2026-08-15 against driver.py: the sub-step calls are at :3282
                # (step_B), :3289 (update_H), :3293 (step_D) and :3304 (update_E).
                fields.Bz.reshape(-1)[13] += np.float32(0.5 * np.sin(0.31 * number))
            if substitute:
                step.plans["update_H"].run()
            else:
                stepping.update_H(fields, pml)
            stepping.step_D(fields, pml)
            if substitute:
                step.plans["update_E"].run()
            else:
                stepping.update_E(fields, pml)
            digests.append(state_digest(snapshot(fields)))
        return initial, snapshot(fields), sum(
            plan.runs for plan in step.plans.values()), digests

    finals: Dict[bool, Dict[str, Any]] = {}
    for driven in (True, False):
        initial_a, final_a, runs_a, digests_a = drive(substitute=False, driven=driven)
        finals[driven] = final_a
        initial_b, final_b, runs_b, digests_b = drive(substitute=True, driven=driven)
        moved = total_differing(initial_a, final_a)
        diff = per_array_differing(final_a, final_b)
        per_step = [i for i, (x, y) in enumerate(zip(digests_a, digests_b)) if x != y]

        row = {"driven": driven, "steps": ENGINE_STEPS,
               "array_path_moved_words": moved,
               "substituted_plan_runs": runs_b,
               "steps_compared": len(digests_a),
               "steps_differing": per_step,
               "final_differing": diff}
        rows.append(row)
        log(f"[engine] driven={driven} steps={ENGINE_STEPS} moved={moved} "
            f"plan_runs={runs_b} steps_differing={len(per_step)} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["engine"] = rows
        save(payload, out)

        assert runs_a == 0, "the array-path run must not have invoked a null plan"
        vacuity_floor(f"driven={driven} whole-step movement", moved, 1,
                          "the array-path driver moved NO state across the budget, "
                          "so 'bit-identical' would be comparing two frozen states")
        vacuity_floor(f"driven={driven} plan runs", runs_b,
                          2 * ENGINE_STEPS)
        assert not per_step, (driven, per_step[:4])
        assert not diff, (driven, diff)
        assert len(digests_a) == ENGINE_STEPS

    # THE SOURCE IS PROVEN AGAINST ITS OWN CONTROL BY COMPARING THE STATES, not by
    # comparing how many words each run moved. The word COUNT saturates: after 24
    # steps of a periodic no-PML run essentially every word already differs from
    # the seed in both runs (measured: 6480 of 6480 either way), so a count
    # comparison would have called a dead injection non-vacuous. The states
    # themselves are the discriminator, and they are what "the source did
    # something" actually means.
    source_delta = total_differing(finals[True], finals[False])
    payload["legs"]["engine_summary"] = {
        "source_non_vacuous": source_delta > 0,
        "driven_vs_source_free_differing_words": source_delta,
        "driven_moved_from_seed": rows[0]["array_path_moved_words"],
        "source_free_moved_from_seed": rows[1]["array_path_moved_words"],
        "moved_word_count_saturates": (
            rows[0]["array_path_moved_words"]
            == rows[1]["array_path_moved_words"]),
        "step_budget": ENGINE_STEPS,
        "budget_note": ("the covered arm has no divergence mechanism — nothing "
                        "rounds — so this number bounds the MEASUREMENT and not "
                        "the claim. This gate does not raise the composed "
                        "whole-step budget and does not claim to"),
    }
    save(payload, out)
    log(f"[engine] source non-vacuity: driven vs source-free differ on "
        f"{source_delta} words")
    assert source_delta > 0, (
        "the driven run and the source-free control ended in the SAME state, so "
        "the injection did nothing and the 24-step claim is about a run with no "
        "source in it")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("identity", leg_identity),
    ("controls", leg_controls),
    ("breadth", leg_breadth),
    ("over_coverage", leg_over_coverage),
    ("overlap", leg_overlap),
    ("residency", leg_residency),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("engine", leg_engine),
)

REQUIRED = tuple(name for name, _ in LEGS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact JSON path")
    parser.add_argument("--legs", default="", help="comma-separated leg subset")
    arguments = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    sources = {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "metal_kernels/no_pml_constitutive.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "no_pml_constitutive.py"),
        "metal_kernels/coverage.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "coverage.py"),
        "metal_kernels/launch.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "launch.py"),
        "metal_kernels/arms.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "arms.py"),
        "triton_kernels/no_pml_constitutive.py": os.path.join(
            API_ROOT, "meep_gpu", "triton_kernels", "no_pml_constitutive.py"),
        # RECORDED, NOT IMPORTED, and the label says so. Hashing a file this gate
        # does not read would otherwise read as "the kit is an input to what was
        # certified" — which contradicts the whole reason for the self-containment
        # above. It is kept because the ~60 duplicated lines are a stated debt, and
        # a reader comparing two cuts should be able to see the kit move.
        "metal_gate_kit.py (NOT IMPORTED — the duplicated shapes, recorded so the "
        "divergence is visible)": os.path.join(HERE, "metal_gate_kit.py"),
        "gate_metal_no_pml_constitutive.py": os.path.abspath(__file__),
    }

    payload: Dict[str, Any] = {
        "environment": environment(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": provenance(sources, out_dir),
        "family": {
            "name": family.FAMILY,
            "ships_kernel": False,
            "return_sites": {side: spec["return_site"]
                             for side, spec in family.NULL_SIDES.items()},
            "non_clauses": family.METAL_NON_CLAUSES,
            "stored_e_arm_built": family.STORED_E_ARM["built"],
            "predicate_is_shared_object": True,
        },
        "claim": (
            "a PLANNER-LEVEL SUBSTITUTION: on a run with an inactive absorber (and "
            "stores_E False on the E side) stepping.update_H returns at "
            "stepping.py:944-945 and update_E at :983-984, so the null plan and the "
            "array path leave every live array byte-identical — proved over "
            f"{IDENTITY_CYCLES} full cycles and {ENGINE_STEPS} whole steps, with "
            "the step PROVEN to have moved state, because a null agreeing with a "
            "no-op is trivially identical"),
        "legs": {},
    }
    save(payload, arguments.out)

    log(f"[env] numpy={payload['environment']['numpy']} "
        f"torch={payload['environment']['torch']} "
        f"policy={payload['subnormal_policy']['resolved']} "
        f"mps={payload['environment']['mps_available']}")
    log("[env] this family compiles nothing and launches nothing; no device is "
        "required for any leg")

    wanted = tuple(name.strip() for name in arguments.legs.split(",") if name.strip())
    return run_legs(LEGS, payload, arguments.out, wanted, REQUIRED)


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
