"""Device confirmation for the hand-CUDA NO-PML constitutive NULL arm.

WHAT THIS GATE IS FOR, AND WHY IT IS NOT A BIT-IDENTITY SWEEP. The family under test
(``meep_gpu/cuda_kernels/no_pml_constitutive.py``) ships NO KERNEL. Without an
absorber ``stepping.update_H`` returns at stepping.py:944-945 and, with E unstored,
``stepping.update_E`` returns at :983-984, so the whole product of the sub-step is a
``return``. There is nothing to launch and nothing to compare against a launch.

WHAT CAN STILL BE WRONG IS THE ADMISSION, and it is the only thing that can. A null
that admits ONE configuration on which the array path actually moves is silent
corruption of the worst kind available in this codebase: the composer skips a
sub-step that had work to do, and no byte ever disagrees with the wrong answer,
because the wrong answer is "whatever was there before". So the certifying leg here
is an IDENTITY leg -- every live device word unchanged across the array path's own
call, over several complete cycles -- and its control is the same assertion on
configurations the predicate REFUSES, where it must fail. An identity leg with no
failing control measures nothing: it is satisfied just as well by a fixture in which
nothing could have moved.

Legs (``--legs``, default all):

* ``identity``   the covered arm. Seed every allocated array non-degenerately
                 (uniform + a signed-zero plane + dense subnormals), snapshot uint32
                 words, run ``update_H``+``update_E`` for ``IDENTITY_CYCLES``
                 complete cycles, byte-compare against the ORIGINAL each cycle. PASS
                 requires the predicate to ADMIT both sides, zero differing words,
                 and a NON-VACUOUS census. The census is a pass condition rather than
                 a statistic: for a null family "no bytes moved" is exactly what a
                 zero-initialised ``Fields`` also reports.
* ``controls``   the leg that makes ``identity`` a measurement. An active layer and a
                 no-PML run with stored E: the predicate MUST refuse, and the array
                 path MUST move words. Either half passing alone is a gate failure.
* ``breadth``    the family's one distinguishing claim: every clause the other CUDA
                 predicates carry -- the fold, the cylindrical axis, complex storage,
                 a Bloch phase, beta, BFAST, a conductivity, a metallic wall -- is
                 ABSENT here, because a sub-step that returns before its first
                 statement forms no index and takes no pointer. Admission plus
                 identity on each, on device.
* ``overlap``    every shipped CUDA constitutive predicate must REFUSE where this one
                 admits, and this one must refuse where they admit. The partition is
                 by construction (both sides ask ``pml.is_active``) and is measured
                 rather than asserted.
* ``mutations``  each clause deleted from the REAL module text, compiled into a fresh
                 module object with its own sha256, and required to change a verdict.
                 DISARMED (the edit did not apply) and NEEDLE-MISSED (the verdict did
                 not move) are failures.
* ``engine``     the substitution on a real ``FdtdDriver``: two drivers from one spec,
                 one stepped by the array path and one with ``driver.update_H`` /
                 ``driver.update_E`` replaced by a no-op, ``ENGINE_STEPS`` whole
                 steps, every live array byte-compared after every step. Run WITH a
                 source and WITHOUT one, and the source-free control must DIFFER from
                 the driven run -- which is what proves the driven case's arrays were
                 not a plausible-looking wall of zeros.

DISCIPLINE, and where each item lands on a family that computes nothing:

* uint32 byte compare only, never ``allclose``.
* a NON-POWER-OF-TWO Courant in every sweep, carried and recorded as a PREDICTED NULL
  on the covered arm (nothing rounds, so no association or FMA hazard exists for it
  to expose) and load-bearing on ``controls``, where real arithmetic runs.
* PTX evidence -- NOT APPLICABLE, stated rather than omitted: this family compiles
  nothing. The substitutes are the mutant module digests, the substitution counters,
  and an NVRTC count taken AFTER the legs rather than before (the sibling gate froze
  that stamp at zero by taking it first, and then cited the zero as confirmation that
  the family compiles nothing -- a number that could not have read anything else).
* the subnormal policy is DRIVEN and STAMPED, never re-implemented, and through
  THIS track's spelling: ``probe_fused_kernel_bit_identity``'s
  ``install_subnormal_policy_for_run`` before the first leg and
  ``subnormal_policy_stamp`` after the last, the same pair every other CUDA gate
  uses. A refused install exits non-zero rather than running under a policy nobody
  asked for. The arm compiles nothing of its own; the harness does, and the counters
  read the harness.
* THE FIXTURES ARE IMPORTED, NOT RE-TRANSCRIBED. ``build``, the seeding, the
  snapshot/diff/census and the three case tables come from
  ``gate_triton_no_pml_constitutive``, which measured them into their present shape
  (including the plant rotation that a 1-voxel corpus row needs). A second copy would
  drift, and this package's rule is that a second spelling of an arbiter is one too
  many. What is this gate's own is the predicate under test, the overlap leg against
  the CUDA siblings, the mutation battery over the CUDA source, and the contract.
* SELF-ENFORCING: :func:`validate_payload` runs over the artifact this gate just
  wrote and fails the run on a contract violation no per-case check can see -- a
  missing leg, a vacuous partition, a mutation that flipped nothing, an engine leg
  whose two arms did not differ.
* THE ARTIFACT MANIFEST IS CHECKED ON EVERY RUN, and only written when
  ``--write-manifest`` is passed into a directory that has none.
  ``source_sha256.txt`` names the predicate, its suite, ``stepping``, ``coverage``
  (the siblings the partition is measured against) and both gates; a disagreement
  REFUSES to release. Editing a gate or a predicate after a run means a FRESH
  directory, not a re-used one. A ``keep/`` or ``flush/`` subdirectory inherits its
  root's manifest and nothing else does.

Usage (laptop, NumPy -- every leg runs, nothing is skipped)::

    python -u gate_cuda_no_pml_null_constitutive.py \\
        --out results/cuda_no_pml_null_constitutive_<date>/gate.json

Usage (the GPU host, the device confirmation, on ONE verified-empty GPU)::

    CUDA_VISIBLE_DEVICES=<index> CUPY_CACHE_DIR=<fresh, per-leg> python -u \\
        gate_cuda_no_pml_null_constitutive.py --backend cupy \\
        --out results/cuda_no_pml_null_constitutive_<date>/gate_cupy.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
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

# The Triton no-PML gate is imported for its FIXTURES and its byte machinery only --
# not for its verdicts. Every assertion below is against the hand-CUDA predicate.
import gate_triton_no_pml_constitutive as ref  # noqa: E402

# THE SUBNORMAL POLICY COMES FROM THIS TRACK'S OWN SPELLING, not the sibling's. Both
# exist and both work; ``gate_cuda_folded_constitutive``, ``gate_cuda_ade`` and
# ``gate_cuda_offdiag`` all drive it through ``probe_fused_kernel_bit_identity``, and
# a CUDA artifact whose policy block had a different shape from every other CUDA
# artifact would be one a comparison script has to special-case.
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import driver as driver_module  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import coverage as cuda_coverage  # noqa: E402
from meep_gpu.cuda_kernels import no_pml_constitutive as family  # noqa: E402

build = ref.build
seed_arrays = ref.seed_arrays
snapshot = ref.snapshot
diff = ref.diff
census = ref.census
polarization_arrays = ref.polarization_arrays
log = ref.log
save = ref.save
sha256_text = ref.sha256_text
sha256_file = ref.sha256_file
COURANT_NP2 = ref.COURANT_NP2
COURANT_P2 = ref.COURANT_P2
IDENTITY_CASES = ref.IDENTITY_CASES
BREADTH_CASES = ref.BREADTH_CASES
CONTROL_CASES = ref.CONTROL_CASES
ELECTRIC_SOURCE = ref.ELECTRIC_SOURCE

SHIPPED_SOURCE = os.path.join(_REPO_API, "meep_gpu", "cuda_kernels",
                              "no_pml_constitutive.py")

#: Sources whose digests the artifact manifest pins. The predicate and its suite are
#: obvious; the two gates are here because a gate edited after a run is exactly the
#: case the manifest exists to refuse.
MANIFEST_SOURCES: Tuple[str, ...] = (
    "meep_gpu/cuda_kernels/no_pml_constitutive.py",
    "meep_gpu/cuda_kernels/test_no_pml_constitutive.py",
    # THE SIBLINGS THE PARTITION IS MEASURED AGAINST. Under active edit by other
    # work, which is exactly why the artifact has to record WHICH version the
    # overlap leg ran against: "no slot is claimed twice" is a statement about a
    # pair of predicates, and half of that pair lives here.
    "meep_gpu/cuda_kernels/coverage.py",
    "meep_gpu/stepping.py",
    "parity/meep_gpu/gate_cuda_no_pml_null_constitutive.py",
    "parity/meep_gpu/gate_triton_no_pml_constitutive.py",
)

SIDES = ("H", "E")
SIDE_CALL = {"H": stepping.update_H, "E": stepping.update_E}

#: The subdirectory names a run may create INSIDE one artifact root, and therefore
#: the only ones allowed to inherit that root's ``source_sha256.txt``.
POLICY_SUBDIRECTORIES: Tuple[str, ...] = ("keep", "flush")

IDENTITY_CYCLES = ref.IDENTITY_CYCLES
ENGINE_STEPS = 24
LEGS = ("identity", "controls", "breadth", "overlap", "mutations", "engine")


def verdict(fields: Any, pml: Any, grid: Any, side: str) -> Tuple[bool, str]:
    covered, reason = family.covers_no_pml_null_constitutive(fields, pml, grid, side)
    return bool(covered), str(reason)


# ---------------------------------------------------------------------------
# identity / breadth — one shared per-case contract
# ---------------------------------------------------------------------------

#: THE CENSUS CONTRACT, PER SUBNORMAL POLICY -- and the split is a MEASUREMENT, not
#: an allowance made for a leg that would not pass.
#:
#: The seed plants three classes a null could conceivably disturb: ordinary normals,
#: a plane of NEGATIVE ZEROS (which no random draw can produce), and dense
#: SUBNORMALS. Under ``keep`` all three survive into the arrays and all three are
#: required, because "no bytes moved" is otherwise satisfied by a wall of zeros.
#:
#: UNDER ``flush`` THE SUBNORMALS CANNOT SURVIVE THE SEED, and that was measured
#: here rather than anticipated: ``install_subnormal_policy('flush')`` drives the
#: HOST FPU through ``mp.set_zero_subnormals``, so the ``1e-45`` and ``-3e-44``
#: literals are flushed in NumPy, before anything is copied to the device. On
#: ``plain_3d_np2`` the keep census reads 10080 nonzero / 847 signed zero / 777
#: subnormal, and the flush census reads 9653 / 1197 / 0 -- and the three numbers
#: ACCOUNT FOR EACH OTHER EXACTLY: 1197 - 847 = 350 negative subnormals became
#: ``-0.0``, 10080 - 9653 = 427 positive ones became ``+0.0``, and 350 + 427 = 777,
#: the planted count.
#:
#: SO THE FLUSH RULE IS STRICTER, NOT LOOSER. It requires ``subnormal_words == 0``,
#: which is a POSITIVE measurement that the installed policy reached the data: a
#: surviving subnormal under a policy that claims to flush would mean the policy was
#: installed somewhere the seed does not go. Demanding ``> 0`` there, as the first
#: cut of this gate did, is a condition no correct run can satisfy -- it failed 15 of
#: 15 cases whose identity comparison had reported zero differing words.
CENSUS_CONTRACT: Dict[str, Dict[str, Any]] = {
    "keep": {"nonzero": "> 0", "signed_zero": "> 0", "subnormal": "> 0",
             "why": "all three classes survive and all three are required"},
    "flush": {"nonzero": "> 0", "signed_zero": "> 0", "subnormal": "== 0",
              "why": "the policy flushes the planted subnormals at the HOST before "
                     "the copy; a survivor would mean the policy did not reach the "
                     "seed"},
}

#: The rule a run with no ``--subnormal-policy`` is scored under. NOT "flush":
#: nothing was installed, so the host keeps whatever it was doing, and on both hosts
#: this gate has run on that is IEEE keep.
DEFAULT_CENSUS_RULE = "keep"


def census_is_non_vacuous(block: Dict[str, int], rule: str) -> Tuple[bool, str]:
    """Does this seeded state carry the classes its policy says it should?"""
    contract = CENSUS_CONTRACT[rule]
    if block["nonzero_words"] <= 0:
        return False, "VACUOUS: no nonzero word -- 'nothing moved' is trivial here"
    if block["signed_zero_words"] <= 0:
        return False, ("VACUOUS: no signed-zero word, and no random draw can produce "
                       "one, so the plant did not reach the arrays")
    if rule == "flush":
        if block["subnormal_words"] != 0:
            return False, (f"{block['subnormal_words']} subnormal words SURVIVED "
                           "under the flush policy: the policy was not attained on "
                           "the seeded data")
        return True, contract["why"]
    if block["subnormal_words"] <= 0:
        return False, ("VACUOUS: no subnormal word under a keep policy -- the class "
                       "a flush would move is absent from the comparison")
    return True, contract["why"]


def one_identity_case(spec: Dict[str, Any], xp: Any,
                      predicate: Callable[..., Tuple[bool, str]] = verdict,
                      rule: str = DEFAULT_CENSUS_RULE) -> Dict[str, Any]:
    """Admit, snapshot, call the array path N times, compare, census. One case.

    IDENTITY AND BREADTH MEET THE SAME CONTRACT THROUGH THIS ONE FUNCTION. The
    sibling gate recorded that its breadth leg -- where the family's distinguishing
    claim lives -- had once been checked on two counters and a flag, and passed with
    a case that moved 7 words. Both legs go through here so that cannot recur.
    """
    started = time.time()
    fdtd = build(spec, xp)
    fields, pml, grid = fdtd.fields, fdtd.pml, fdtd.grid
    entry: Dict[str, Any] = {
        "name": spec["name"],
        "shape": list(fdtd.grid.shape),
        "courant": repr(spec.get("courant", COURANT_NP2)),
        "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
        "stores_E": bool(fields.stores_E),
        # An INERT LAYER PRESENT and NO LAYER AT ALL reach the same ``return`` and
        # are different configurations; a row claiming the first must prove a layer
        # was there.
        "pml_object_present": bool(pml is not None),
        "pml_active": bool(pml is not None and pml.is_active),
        "requires_pml_object": bool(spec.get("requires_pml_object")),
        "row": spec.get("row"),
    }
    verdicts = {side: predicate(fields, pml, grid, side) for side in SIDES}
    entry["covered"] = {side: v[0] for side, v in verdicts.items()}
    entry["reasons"] = {side: v[1] for side, v in verdicts.items()}

    entry["census_before"] = census(fields)
    before = snapshot(fields)
    entry["polarization_arrays_compared"] = len(polarization_arrays(fields))
    entry["field_arrays_compared"] = len(before) - entry["polarization_arrays_compared"]

    per_cycle: List[Dict[str, Any]] = []
    worst = {"bit_identical": True, "differing_words": 0, "total_words": 0,
             "per_array": {}, "arrays_compared": 0}
    for cycle in range(1, IDENTITY_CYCLES + 1):
        for side in SIDES:
            SIDE_CALL[side](fields, pml)
        comparison = diff(before, snapshot(fields))
        per_cycle.append({"cycle": cycle,
                          "differing_words": comparison["differing_words"]})
        worst["bit_identical"] &= bool(comparison["bit_identical"])
        worst["differing_words"] += comparison["differing_words"]
        worst["total_words"] = comparison["total_words"]
        worst["per_array"].update(comparison["per_array"])
        worst["arrays_compared"] = comparison["arrays_compared"]
    entry["cycles"] = IDENTITY_CYCLES
    entry["per_cycle"] = per_cycle
    entry["comparison"] = worst
    entry["census_after"] = census(fields)

    non_vacuous, census_reason = census_is_non_vacuous(entry["census_before"], rule)
    entry["census_rule"] = rule
    entry["census_verdict"] = census_reason
    entry["non_vacuous"] = bool(non_vacuous)
    entry["ok"] = bool(
        all(entry["covered"].values())
        and entry["comparison"]["bit_identical"]
        and entry["comparison"]["arrays_compared"] > 0
        and (entry["pml_object_present"] or not entry["requires_pml_object"])
        and non_vacuous)
    if not non_vacuous:
        entry["error"] = census_reason
    entry["seconds"] = round(time.time() - started, 3)
    return entry


def run_identity(results: Dict[str, Any], out_path: str, xp: Any,
                 backend: str, rule: str = DEFAULT_CENSUS_RULE) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in IDENTITY_CASES:
        entry = one_identity_case(spec, xp, rule=rule)
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
                backend: str, rule: str = DEFAULT_CENSUS_RULE) -> Dict[str, Any]:
    """Every clause the OTHER CUDA predicates carry, admitted and byte-checked here.

    The refusal each case would meet elsewhere is recorded per case, and measured
    rather than quoted: the shipped sibling predicate is CALLED on the same objects
    and its first refusal is stored beside this arm's admission.
    """
    cases: List[Dict[str, Any]] = []
    for spec in BREADTH_CASES:
        entry = one_identity_case(spec, xp, rule=rule)
        fdtd = build(spec, xp)
        sibling: Dict[str, Any] = {}
        for side in SIDES:
            try:
                sibling[side] = list(cuda_coverage.covers_real_pml_constitutive(
                    fdtd.fields, fdtd.pml, fdtd.grid, side))
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                sibling[side] = [False, f"raised {type(exc).__name__}: {exc}"]
        entry["cuda_sibling_verdict"] = sibling
        entry["refused_elsewhere"] = spec["refused_elsewhere"]
        entry["sibling_refuses_both_sides"] = bool(
            not sibling["H"][0] and not sibling["E"][0])
        cases.append(entry)
        log(f"[breadth:{backend}] {entry['name']:22s} covered={entry['covered']} "
            f"identical={entry['comparison']['bit_identical']} ok={entry['ok']} "
            f"| cuda sibling H: {sibling['H'][1][:60]}")
        results.setdefault("breadth", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["breadth"][backend]


# ---------------------------------------------------------------------------
# controls — where the array path MUST move
# ---------------------------------------------------------------------------

def run_controls(results: Dict[str, Any], out_path: str, xp: Any,
                 backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in CONTROL_CASES:
        started = time.time()
        fdtd = build(spec, xp)
        fields, pml, grid = fdtd.fields, fdtd.pml, fdtd.grid
        entry: Dict[str, Any] = {
            "name": spec["name"], "shape": list(fdtd.grid.shape),
            "courant": repr(spec.get("courant")),
            "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
            "stores_E": bool(fields.stores_E),
            "pml_active": bool(pml is not None and pml.is_active),
            "must_move": list(spec["must_move"]), "row": spec.get("row"),
        }
        verdicts = {side: verdict(fields, pml, grid, side) for side in SIDES}
        entry["covered"] = {side: v[0] for side, v in verdicts.items()}
        entry["reasons"] = {side: v[1] for side, v in verdicts.items()}
        entry["census_before"] = census(fields)
        entry["polarization_arrays_compared"] = len(polarization_arrays(fields))

        moved: Dict[str, int] = {}
        polarization_moved: Dict[str, int] = {}
        for side in SIDES:
            before = snapshot(fields)
            SIDE_CALL[side](fields, pml)
            after = snapshot(fields)
            moved[side] = diff(before, after)["differing_words"]
            names = [n for n in before if n.startswith("pol")]
            polarization_moved[side] = sum(
                int(np.count_nonzero(before[n] != after[n])) for n in names)
        entry["moved_words"] = moved
        # ``update_E`` CONSUMES a polarization and does not advance it
        # (stepping.update_E's docstring, "P is consumed here and advanced
        # afterwards"). This is the only leg on which P/P_prev exist at all, so it
        # is the only place that pin can be measured.
        entry["polarization_words_moved"] = polarization_moved
        # THE SOUNDNESS IMPLICATION, PER SIDE, and it is the whole property this
        # family can violate: ADMITTED => the array path moved nothing. Its
        # contrapositive is what the control cases are for.
        #
        # NOT "every side is refused". A no-PML dispersive run is REFUSED on E and
        # ADMITTED on H, correctly and measurably -- H is never stored without PML
        # (fields.py:664-665) and the array path moves 0 words there. Requiring both
        # sides to be refused would fail the gate on the predicate being RIGHT, which
        # is how a control leg turns into a leg that measures its own expectations.
        entry["soundness"] = {side: not (entry["covered"][side] and moved[side] > 0)
                              for side in SIDES}
        entry["ok"] = bool(
            all(entry["soundness"].values())
            and all(moved[side] > 0 and not entry["covered"][side]
                    for side in spec["must_move"])
            and all(polarization_moved[side] == 0 for side in SIDES))
        entry["seconds"] = round(time.time() - started, 3)
        cases.append(entry)
        log(f"[controls:{backend}] {entry['name']:26s} refused={entry['covered']} "
            f"moved={moved} pol_moved={polarization_moved} ok={entry['ok']}")
        results.setdefault("controls", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases,
            "polarization_arrays_compared": sum(
                c["polarization_arrays_compared"] for c in cases)}
        save(results, out_path)
    return results["controls"][backend]


# ---------------------------------------------------------------------------
# overlap — the partition against every shipped CUDA constitutive predicate
# ---------------------------------------------------------------------------

def _sibling_predicates() -> Dict[str, Callable[..., Any]]:
    """Every shipped predicate in ``cuda_kernels.coverage`` that answers for a
    constitutive sub-step, bound to this gate's ``(fields, pml, grid, side)`` shape.

    Enumerated by NAME from the module rather than from a list kept here, so a
    constitutive predicate added to ``coverage.py`` without a row in this leg fails
    the contract check below instead of quietly going unmeasured.
    """
    bound: Dict[str, Callable[..., Any]] = {
        "covers_real_pml_constitutive":
            lambda f, p, g, s: cuda_coverage.covers_real_pml_constitutive(f, p, g, s),
        "covers_real_pml_offdiag_constitutive":
            lambda f, p, g, s: (
                cuda_coverage.covers_real_pml_offdiag_constitutive(f, p, g)
                if s == "E" else (False, "the H side has no off-diagonal kernel")),
        "covers_real_pml_complex_constitutive":
            lambda f, p, g, s: cuda_coverage.covers_real_pml_complex_constitutive(
                f, p, g, s, "NAIVE"),
    }
    discovered = {name for name in dir(cuda_coverage)
                  if name.startswith("covers_") and "constitutive" in name}
    missing = discovered - set(bound)
    if missing:
        raise RuntimeError(
            f"coverage.py has constitutive predicates this leg does not measure: "
            f"{sorted(missing)}. A partition claim that omits an admitter is not a "
            f"partition claim.")
    return bound


class _CupyNamedGrid:
    """A grid presenting CuPy's ``__name__`` for its array module, delegating the rest.

    WHY THE PARTITION LEG NEEDS ONE AND THE REST OF THE GATE DOES NOT. Every sibling
    predicate opens with ``backend is not CuPy``, so on a NumPy host all three refuse
    everything for that clause alone and the partition is satisfied VACUOUSLY -- two
    predicates that never admit anything are trivially disjoint. This is the census's
    own shim (``covered_modulo_backend``), and it is used ONLY to ask the siblings
    their non-backend verdict. This arm's own verdict is always taken on the real
    grid, because it has no backend clause to shim away.
    """

    class _Module:
        __name__ = "cupy"

        def __getattr__(self, item):
            return getattr(np, item)

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", _CupyNamedGrid._Module())

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


def run_overlap(results: Dict[str, Any], out_path: str, xp: Any,
                backend: str) -> Dict[str, Any]:
    siblings = _sibling_predicates()
    rows: List[Dict[str, Any]] = []
    both_admitted = 0
    mine_admitted = 0
    theirs_admitted = 0
    for spec in tuple(IDENTITY_CASES) + tuple(BREADTH_CASES) + tuple(CONTROL_CASES):
        fdtd = build(spec, xp)
        fields, pml, grid = fdtd.fields, fdtd.pml, fdtd.grid
        for side in SIDES:
            mine = verdict(fields, pml, grid, side)
            mine_admitted += int(mine[0])
            shimmed = grid if backend == "cupy" else _CupyNamedGrid(grid)
            for name, call in siblings.items():
                try:
                    raw = tuple(call(fields, pml, grid, side))
                    raised = False
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    raw = (False, f"raised {type(exc).__name__}: {exc}")
                    raised = True
                try:
                    theirs = tuple(call(fields, pml, shimmed, side))
                except Exception as exc:  # noqa: BLE001
                    theirs = (False, f"raised {type(exc).__name__}: {exc}")
                    raised = True
                theirs_admitted += int(bool(theirs[0]))
                overlap = bool(mine[0] and theirs[0])
                both_admitted += int(overlap)
                rows.append({"case": spec["name"], "side": side, "sibling": name,
                             "mine": [mine[0], mine[1]],
                             "theirs_modulo_backend": [bool(theirs[0]), str(theirs[1])],
                             "theirs_raw": [bool(raw[0]), str(raw[1])],
                             "sibling_raised": raised, "overlap": overlap})
    block = {
        "rows": len(rows), "overlaps": both_admitted,
        "slots_this_arm_admits": mine_admitted,
        "slots_a_sibling_admits": theirs_admitted,
        "siblings": sorted(siblings),
        "sibling_raises": sum(int(r["sibling_raised"]) for r in rows),
        "detail": rows,
        "ok": bool(both_admitted == 0 and mine_admitted > 0 and theirs_admitted > 0),
    }
    log(f"[overlap:{backend}] rows={block['rows']} overlaps={block['overlaps']} "
        f"mine={mine_admitted} theirs={theirs_admitted} "
        f"sibling_raises={block['sibling_raises']} ok={block['ok']}")
    results.setdefault("overlap", {})[backend] = block
    save(results, out_path)
    return block


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------

def compile_mutant(source: str, label: str) -> Any:
    """Compile edited TEXT into a fresh module object with its own digest.

    There is no kernel cache here to serve a stale binary, but the equivalent hazard
    -- importing the shipped predicate under the mutant's name -- is closed the same
    way: the mutant never touches ``sys.modules`` and the digest is recorded so a
    mutant identical to the shipped source is visible as such.
    """
    spec = importlib.util.spec_from_loader(f"_mutant_{label}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = SHIPPED_SOURCE
    exec(compile(source, f"<mutant {label}>", "exec"), module.__dict__)
    return module


def _replace_once(source: str, old: str, new: str) -> Tuple[str, int]:
    return source.replace(old, new), source.count(old)


#: (label, edit, case spec, sides, what an uncaught mutation would cost).
MUTATIONS: Tuple[Tuple[str, Callable[[str], Tuple[str, int]], Dict[str, Any],
                       Tuple[str, ...], str], ...] = (
    ("admit_an_active_layer",
     lambda s: _replace_once(s, "        if active:\n", "        if False:\n"),
     {"name": "mutant_active_layer", "cell": (1.2, 1.0, 1.4),
      "courant": COURANT_NP2, "salt": 41}, ("H", "E"),
     "the composer would skip both constitutive sub-steps of an absorbing run"),
    ("admit_a_stored_e_run",
     lambda s: _replace_once(s, "        if stored:\n", "        if False:\n"),
     {"name": "mutant_stored_e", "cell": (1.2, 1.0, 1.4),
      "courant": COURANT_NP2, "salt": 42}, ("E",),
     "the composer would skip the store at stepping.py:993"),  # stepping.py live lines for the frozen device-text citation(s) in this string: 993->1022
    ("default_an_unreadable_flag_to_admit",
     lambda s: _replace_once(s, "        return _UNREADABLE\n    if value is None:",
                             "        return False\n    if value is None:"),
     {"name": "mutant_unreadable", "cell": (1.2, 1.0, 1.4),
      "courant": COURANT_NP2, "salt": 43}, ("H", "E"),
     "an object that cannot answer would be covered rather than refused"),
    ("admit_pml_storage_behind_an_inert_layer",
     lambda s: _replace_once(s, '    if bool(getattr(fields, "_pml_active", False)):',
                             "    if False:"),
     {"name": "mutant_pml_storage", "cell": (1.2, 1.0, 1.4), "inert_pml": 0,
      "courant": COURANT_NP2, "salt": 44, "requires_pml_object": True}, ("H",),
     "half a step would be supplied to a configuration the curl families refuse"),
    ("admit_a_nonlinearity",
     lambda s: _replace_once(
         s,
         '        if (getattr(fields, "_chi2_components", None)\n'
         '                or getattr(fields, "_chi3_components", None)):',
         "        if False:"),
     {"name": "mutant_nonlinear", "cell": (1.2, 1.0, 1.4),
      "courant": COURANT_NP2, "salt": 45}, ("E",),
     "a Pade constitutive factor would be reported as no work"),
)

# ---------------------------------------------------------------------------
# THE MUTATION NEEDLES, AND WHY EACH ONE IS BUILT THE WAY IT IS
#
# A mutation battery measures a CLAUSE, and in a FIRST-REFUSAL predicate a clause can
# only be measured on a configuration that trips it ALONE. This was not a design
# preference here: the battery's first run reported 3 of 5 mutations "caught" when
# they were not, because the obvious fixtures trip two clauses each and the surviving
# clause refused the mutant just as the deleted one had.
#
#   * an ACTIVE PML built through the driver also switches PML storage on, so
#     deleting the layer clause leaves clause 5 to refuse it;
#   * a DISPERSIVE no-PML run has storage on AND a driven polarization, so deleting
#     the stores_E clause leaves the polarization clause to refuse it;
#   * an inert-layer case built through the driver does NOT switch PML storage on, so
#     the shipped predicate ADMITS it and there is no refusal to flip at all.
#
# Each needle below therefore isolates one clause, and says which real object it
# stands in for. Where the driver can build the isolating object it is used; where it
# cannot -- an active layer with no PML storage, a nonlinearity with no stored E --
# the stand-in is duck-typed and labelled as such.
# ---------------------------------------------------------------------------

class _ActiveLayer:
    """A layer that absorbs, handed to a ``Fields`` that has no PML storage.

    ``driver.setup_pml`` allocates the absorber's storage as part of installing it
    (fields.py:678-702), so this pair cannot be built through the driver -- and the
    pair is exactly what isolates the layer clause from clause 5.
    """

    is_active = True


class _UnreadableLayer:
    """A layer whose ``is_active`` raises: the needle for the ``_flag`` mutation."""

    @property
    def is_active(self):
        raise RuntimeError("is_active is not answerable on this object")


class _NonlinearNotStored:
    """chi2 installed with storage NOT switched on -- ``set_nonlinear_volumes``'s object.

    ``driver.set_chi2`` forces stored E (driver.py:2023-2029), so a driver-built run
    is refused by the ``stores_E`` clause before the nonlinearity clause is reached
    and this mutation could not be caught on one. This is the object on which the
    clause fires alone, and it is the same object the sibling track measured moving 0
    of 10080 words.
    """

    stores_E = False
    polarizations = ()
    has_offdiagonal_epsilon = False
    _chi2_components = ("Ex",)
    _chi3_components = ()


def _mutation_subject(label: str, spec: Dict[str, Any], xp: Any):
    """The (fields, pml, grid) triple that trips exactly the clause under test."""
    fdtd = build(spec, xp)
    if label == "admit_an_active_layer":
        # A real Fields with no storage of any kind + an absorbing layer.
        return fdtd.fields, _ActiveLayer(), fdtd.grid
    if label == "default_an_unreadable_flag_to_admit":
        return fdtd.fields, _UnreadableLayer(), fdtd.grid
    if label == "admit_a_stored_e_run":
        # A real Fields with E STORED and no polarization: the store arm's own
        # configuration, and the only clause it trips is stores_E.
        fdtd.fields.enable_field_storage()
        return fdtd.fields, None, fdtd.grid
    if label == "admit_pml_storage_behind_an_inert_layer":
        # Real PML storage behind a real INERT layer object -- the configuration
        # fields.py:695-702 creates and pml.py:427-434 names, on the H side where
        # stores_E is not a clause at all.
        fdtd.fields.enable_pml_storage()
        return fdtd.fields, fdtd.pml, fdtd.grid
    if label == "admit_a_nonlinearity":
        return _NonlinearNotStored(), None, fdtd.grid
    raise RuntimeError(f"no needle is defined for mutation {label!r}")


def run_mutations(results: Dict[str, Any], out_path: str, xp: Any,
                  backend: str) -> Dict[str, Any]:
    shipped = open(SHIPPED_SOURCE, encoding="utf-8").read()
    shipped_digest = sha256_text(shipped)
    legs: List[Dict[str, Any]] = []
    for label, edit, spec, sides, cost in MUTATIONS:
        mutated, matches = edit(shipped)
        digest = sha256_text(mutated)
        entry: Dict[str, Any] = {
            "label": label, "sites_matched": matches,
            "mutated_source_sha256": digest,
            "distinct_from_shipped": digest != shipped_digest,
            "sides": list(sides), "uncaught_would_cost": cost,
        }
        if matches != 1 or digest == shipped_digest:
            entry["ok"] = False
            entry["error"] = ("DISARMED: the edit matched "
                              f"{matches} sites and the source digest "
                              f"{'moved' if digest != shipped_digest else 'did not move'}")
            legs.append(entry)
            log(f"[mutations:{backend}] {label:42s} DISARMED")
            continue
        module = compile_mutant(mutated, label)
        caught = True
        rows = []
        for side in sides:
            fields, pml, grid = _mutation_subject(label, spec, xp)
            ship = verdict(fields, pml, grid, side)
            mutant = module.covers_no_pml_null_constitutive(fields, pml, grid, side)
            rows.append({"side": side, "shipped": [bool(ship[0]), ship[1]],
                         "mutant": [bool(mutant[0]), str(mutant[1])]})
            caught &= bool(ship[0] is False and bool(mutant[0]) is True)
        entry["rows"] = rows
        entry["ok"] = bool(caught)
        if not caught:
            entry["error"] = ("NEEDLE MISSED: the shipped predicate did not refuse, "
                              "or the mutant did not admit -- either way this "
                              "mutation measured nothing")
        legs.append(entry)
        log(f"[mutations:{backend}] {label:42s} caught={entry['ok']}")
        results.setdefault("mutations", {})[backend] = {
            "ran": len(legs), "ok": sum(int(m["ok"]) for m in legs),
            "shipped_source_sha256": shipped_digest, "legs": legs}
        save(results, out_path)
    return results["mutations"][backend]


# ---------------------------------------------------------------------------
# engine — the substitution on a real driver
# ---------------------------------------------------------------------------

class _NullSubstitution:
    """Replace ``driver.update_H`` / ``update_E`` with a counted no-op, for ONE driver.

    ``driver.py`` imports the two by name, so the swap has to be on the DRIVER
    module's globals; patching ``stepping``'s would leave the driver holding the
    originals and the leg would silently measure nothing.

    THE PATCH IS PER STEP CALL, not per case, and the substituted function RAISES if
    it is ever handed a ``Fields`` other than the one it was built for. The sibling
    gate measured why: its first version patched around BOTH drivers' step calls, so
    the "reference" ran the null too and every case reported 0 differing words --
    a null compared against a null.
    """

    def __init__(self, expected_fields: Any) -> None:
        self.expected = expected_fields
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
                    "for: the patch has leaked onto the reference driver and the leg "
                    "is comparing a null against a null")
            self.counters[name] += 1
        return substituted

    def __enter__(self) -> "_NullSubstitution":
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
    specs = (
        {"name": "engine_driven", "cell": (1.2, 1.0, 1.4), "courant": COURANT_NP2,
         "source": ELECTRIC_SOURCE, "salt": 51},
        {"name": "engine_source_free", "cell": (1.2, 1.0, 1.4),
         "courant": COURANT_NP2, "salt": 51},
        {"name": "engine_metallic_driven", "cell": (1.2, 1.0, 1.4),
         "boundaries": "metallic", "courant": COURANT_NP2,
         "source": ELECTRIC_SOURCE, "salt": 52},
    )
    cases: List[Dict[str, Any]] = []
    finals: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in specs:
        started = time.time()
        reference = build(spec, xp)
        substituted = build(spec, xp)
        entry: Dict[str, Any] = {
            "name": spec["name"], "shape": list(reference.grid.shape),
            "steps": ENGINE_STEPS,
            "covered": {side: verdict(substituted.fields, substituted.pml,
                                      substituted.grid, side)[0]
                        for side in SIDES},
            "has_source": bool(spec.get("source")),
        }
        opening = snapshot(reference.fields)
        patch = _NullSubstitution(substituted.fields)
        differing = 0
        for step in range(ENGINE_STEPS):
            reference.step()
            with patch:
                substituted.step()
            comparison = diff(snapshot(reference.fields),
                              snapshot(substituted.fields))
            differing += comparison["differing_words"]
        closing = snapshot(reference.fields)
        entry["substitution_calls"] = dict(patch.counters)
        entry["substitution_leaks"] = patch.leaks
        entry["differing_words"] = differing
        entry["reference_step_count"] = int(reference.step_count)
        # THE ARRAY-PATH DRIVER'S OWN STATE, before the first step against after the
        # last. A null certifies by agreeing with a no-op, so "the two drivers are
        # bit-identical" is satisfied just as well by a run in which NOTHING MOVED.
        # The sibling gate caught a row that had been passing for exactly that
        # reason. This is a PASS CONDITION, not a statistic.
        entry["reference_moved_words"] = diff(opening, closing)["differing_words"]
        entry["ok"] = bool(
            differing == 0
            and patch.leaks == 0
            and all(entry["covered"].values())
            and patch.counters["update_H"] == ENGINE_STEPS
            and patch.counters["update_E"] == ENGINE_STEPS
            and int(reference.step_count) == ENGINE_STEPS
            and entry["reference_moved_words"] > 0)
        entry["seconds"] = round(time.time() - started, 3)
        finals[spec["name"]] = closing
        cases.append(entry)
        log(f"[engine:{backend}] {entry['name']:24s} differing={differing} "
            f"calls={patch.counters} reference_moved="
            f"{entry['reference_moved_words']} ok={entry['ok']}")
        results.setdefault("engine", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "cases": cases}
        save(results, out_path)
    block = results["engine"][backend]
    # The driven run and the source-free run must DIFFER. Without this, "24 steps,
    # bit-identical" is equally true of two drivers that both sat at zero.
    if "engine_driven" in finals and "engine_source_free" in finals:
        separation = diff(finals["engine_driven"], finals["engine_source_free"])
        block["driven_vs_source_free_differing_words"] = separation["differing_words"]
        block["source_separation_ok"] = bool(separation["differing_words"] > 0)
    else:
        block["source_separation_ok"] = False
    save(results, out_path)
    return block


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def validate_payload(payload: Dict[str, Any], backend: str,
                     rule: str = DEFAULT_CENSUS_RULE) -> List[str]:
    """Contract violations no per-case check can see. Empty list means release.

    THE LEG SET IS PART OF THE CONTRACT. An artifact cut with ``--legs identity``
    must not be able to record a release: a partial run is a debugging aid, not a
    verdict.
    """
    problems: List[str] = []

    for leg in LEGS:
        if backend not in payload.get(leg, {}):
            problems.append(f"leg {leg!r} is missing for backend {backend!r}")
    if problems:
        return problems

    for leg in ("identity", "breadth"):
        block = payload[leg][backend]
        if block["ran"] == 0 or block["ok"] != block["ran"]:
            problems.append(f"{leg}: {block['ok']}/{block['ran']} cases ok")
        if not any(c["np2_courant"] for c in block["cases"]):
            problems.append(f"{leg}: no non-power-of-two Courant in the sweep")
        for case in block["cases"]:
            if case["comparison"]["total_words"] <= 0:
                problems.append(f"{leg}: case {case['name']} compared no words")
            if case["cycles"] < 2:
                problems.append(f"{leg}: case {case['name']} ran a single cycle")
            # THE CENSUS RULE IS RE-APPLIED HERE, from the recorded numbers rather
            # than from the per-case flag: a payload whose cases were scored under
            # one rule and labelled with another cannot pass both checks.
            if case.get("census_rule") != rule:
                problems.append(f"{leg}: case {case['name']} was scored under census "
                                f"rule {case.get('census_rule')!r}, not {rule!r}")
            replayed, _ = census_is_non_vacuous(case["census_before"], rule)
            if replayed is not bool(case["non_vacuous"]):
                problems.append(f"{leg}: case {case['name']} census verdict does not "
                                "reproduce from its own recorded numbers")
    identity = payload["identity"][backend]
    if not any(c["pml_object_present"] for c in identity["cases"]):
        problems.append("identity: no case carried an INERT LAYER OBJECT, so the "
                        "partition's own boundary was never exercised")

    controls = payload["controls"][backend]
    if controls["ran"] == 0 or controls["ok"] != controls["ran"]:
        problems.append(f"controls: {controls['ok']}/{controls['ran']} cases ok")
    if not any(c["moved_words"]["E"] > 0 for c in controls["cases"]):
        problems.append("controls: nothing moved on any control -- the identity leg "
                        "has no failing control and measures nothing")
    if controls.get("polarization_arrays_compared", 0) <= 0:
        problems.append("controls: no polarization array was compared, so 'update_E "
                        "consumes P and does not advance it' was not measured")
    if not any(c["np2_courant"] for c in controls["cases"]):
        problems.append("controls: no non-power-of-two Courant, which is the leg "
                        "where that Courant is load-bearing")
    if not any(not c["np2_courant"] for c in controls["cases"]):
        problems.append("controls: no power-of-two Courant control")

    overlap = payload["overlap"][backend]
    if overlap["overlaps"] != 0:
        problems.append(f"overlap: {overlap['overlaps']} slots claimed twice")
    if overlap["slots_this_arm_admits"] == 0 or overlap["slots_a_sibling_admits"] == 0:
        problems.append("overlap: VACUOUS -- a partition in which one side never "
                        "admits anything is not a partition")

    mutations = payload["mutations"][backend]
    if mutations["ran"] < len(MUTATIONS) or mutations["ok"] != mutations["ran"]:
        problems.append(f"mutations: {mutations['ok']}/{mutations['ran']} caught "
                        f"(battery has {len(MUTATIONS)})")
    for leg in mutations["legs"]:
        if not leg["distinct_from_shipped"]:
            problems.append(f"mutations: {leg['label']} compiled the shipped source")

    engine = payload["engine"][backend]
    if engine["ran"] == 0 or engine["ok"] != engine["ran"]:
        problems.append(f"engine: {engine['ok']}/{engine['ran']} cases ok")
    if not engine.get("source_separation_ok"):
        problems.append("engine: the driven and source-free runs did not differ, so "
                        "the bit-identity claim is over two indistinguishable states")
    for case in engine["cases"]:
        if case["reference_moved_words"] <= 0:
            problems.append(f"engine: case {case['name']} -- the ARRAY-PATH driver "
                            "moved nothing across the whole run, so the comparison "
                            "was between two frozen states")
    return problems


def check_manifest(out_dir: str, write: bool) -> Dict[str, Any]:
    """Compare the artifact's ``source_sha256.txt`` against the sources loaded now.

    A gate edited after a run must land in a FRESH directory. Re-using one silently
    re-labels an old measurement with new code, which is the one failure a reader of
    the artifact cannot detect from its contents.
    """
    digests = {name: sha256_file(os.path.join(_REPO_API, name))
               for name in MANIFEST_SOURCES}
    # THE PARENT IS SEARCHED TOO, and that is not a convenience. A run cuts one
    # directory per policy under one artifact root, so the manifest belongs to the
    # ROOT: writing one per policy would let two legs of the same run disagree about
    # what they ran and both call themselves consistent.
    def read(path: str) -> Dict[str, str]:
        stored: Dict[str, str] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    digest, name = line.split(None, 1)
                    stored[name.strip()] = digest
        return stored

    # THE WRITE TARGET IS BOUND SEPARATELY FROM THE SEARCH, and that is a repair
    # rather than a style choice. Binding both to the loop variable made
    # ``--write-manifest`` write to whichever candidate the loop happened to end on
    # -- the PARENT -- so a manifest for one artifact landed in the shared results
    # ROOT, where it would be read as every sibling directory's own and could
    # overwrite another run's file. Measured here: the first two artifact
    # directories cut with this gate got no manifest of their own and one appeared
    # at results/source_sha256.txt instead.
    write_path = os.path.join(out_dir, "source_sha256.txt")
    candidates = [write_path]
    # THE PARENT IS CONSULTED ONLY FROM A POLICY SUBDIRECTORY. A run lays out one
    # artifact root with ``keep/`` and ``flush/`` inside it, and those two legs share
    # the root's manifest; every OTHER directory is an artifact root itself and must
    # not inherit anything. Searching the parent unconditionally reached the shared
    # ``results/`` directory, where a leftover file -- this gate's own, left by the
    # write-path defect above -- was then read as every sibling artifact's manifest
    # and refused releases that had nothing to do with it. Measured, twice, in both
    # directions.
    if os.path.basename(os.path.normpath(out_dir)) in POLICY_SUBDIRECTORIES:
        candidates.append(
            os.path.join(os.path.dirname(os.path.normpath(out_dir)),
                         "source_sha256.txt"))
    for path in candidates:
        if not os.path.exists(path):
            continue
        stored = read(path)
        # A PARENT MANIFEST IS ONLY OURS IF IT NAMES OUR SOURCES. Measured the hard
        # way: an unrelated manifest left in a shared scratch directory by another
        # run was picked up as this gate's own and refused the release, which is the
        # right verdict for the wrong reason -- and the wrong reason would have been
        # indistinguishable from a real disagreement in the log.
        if path != candidates[0] and not set(MANIFEST_SOURCES) <= set(stored):
            continue
        disagreements = sorted(name for name in digests
                               if stored.get(name) != digests[name])
        return {"path": path, "written": False, "agrees": not disagreements,
                "disagreements": disagreements, "digests": digests}
    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(write_path, "w", encoding="utf-8") as handle:
            for name in MANIFEST_SOURCES:
                handle.write(f"{digests[name]}  {name}\n")
    return {"path": write_path, "written": bool(write), "agrees": True,
            "searched": candidates, "disagreements": [], "digests": digests}


def provenance(xp: Any, backend: str) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "backend": backend,
        "array_module": getattr(xp, "__name__", str(xp)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "shipped_source_sha256": sha256_file(SHIPPED_SOURCE),
        "kernels_in_family": 0,
        "device": None,
        "cupy": None,
    }
    if cp is not None and backend == "cupy":
        block["cupy"] = cp.__version__
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        block["device"] = {
            "id": int(device.id),
            "name": properties["name"].decode(),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "total_memory_bytes": int(properties["totalGlobalMem"]),
        }
    return block


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--legs", default=",".join(LEGS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default=None,
                        help="DRIVEN before the legs and stamped after them. The arm "
                             "compiles nothing of its own; the harness does")
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help="import MEEP so the HOST half of 'flush' is attainable "
                             "(install_subnormal_policy('flush') refuses otherwise)")
    parser.add_argument("--write-manifest", action="store_true",
                        help="write source_sha256.txt when the directory has none")
    args = parser.parse_args(argv)

    if args.backend == "cupy" and cp is None:
        log("cupy is not importable on this host")
        return 3
    xp = cp if args.backend == "cupy" else np
    backend = args.backend
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    manifest = check_manifest(out_dir, args.write_manifest)
    if not manifest["agrees"]:
        log("MANIFEST DISAGREES -- refusing to release into this directory: "
            f"{manifest['disagreements']}")
        log("Edit a gate or a predicate after a run -> cut into a FRESH directory.")
        return 4

    results: Dict[str, Any] = {
        "family": "cuda_kernels.no_pml_constitutive",
        "predicate": "covers_no_pml_null_constitutive",
        "kernels": [],
        "provenance": provenance(xp, backend),
        "manifest": manifest,
        "identity_cycles": IDENTITY_CYCLES,
        "engine_steps": ENGINE_STEPS,
    }
    # THE POLICY IS DRIVEN, NOT NARRATED, AND BEFORE ANYTHING COMPILES. This family
    # compiles nothing of its OWN -- and the harness is ordinary CuPy work that does.
    # The sibling gate measured exactly that trap: it reported "nvrtc_calls 0" as
    # confirmation that the family compiles nothing, while 48 .cubin files sat beside
    # the stamp, because the stamp had been taken BEFORE the legs. So: install first,
    # run, then stamp, and let the stamp read whatever it reads.
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy is not None:
        try:
            results["subnormal_policy"] = {
                "requested": args.subnormal_policy,
                "install": probe.install_subnormal_policy_for_run(
                    args.subnormal_policy, _REPO_API),
                "stamp_taken": "after the legs",
                "family_compiles_nothing": True,
                "note": "the ARM has no device source, so no NVRTC option tuple of "
                        "its own exists to strip; the harness's own CuPy kernels do "
                        "compile and are what the counters below read",
            }
        except Exception as exc:  # noqa: BLE001 - a policy that fails is recorded
            results["subnormal_policy"] = {"requested": args.subnormal_policy,
                                           "install_error": repr(exc)}
            log(f"SUBNORMAL POLICY {args.subnormal_policy!r} REFUSED: {exc!r}")
            return 5

    # WHICH CENSUS RULE THIS RUN IS SCORED UNDER, decided once and recorded, so a
    # reader never has to infer it from the numbers it produced.
    rule = args.subnormal_policy or DEFAULT_CENSUS_RULE
    results["census_rule"] = rule
    results["census_contract"] = CENSUS_CONTRACT[rule]

    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    runners = {
        "identity": run_identity, "controls": run_controls, "breadth": run_breadth,
        "overlap": run_overlap, "mutations": run_mutations, "engine": run_engine,
    }
    started = time.time()
    for leg in legs:
        if leg not in runners:
            log(f"unknown leg {leg!r}")
            return 2
        log(f"===== leg {leg} ({backend}, census rule {rule}) =====")
        if leg in ("identity", "breadth"):
            runners[leg](results, args.out, xp, backend, rule)
        else:
            runners[leg](results, args.out, xp, backend)

    try:
        # STAMPED AFTER THE LEGS, unconditionally -- including on a run that asked
        # for no policy, where "nothing was installed" is itself the answer.
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    except Exception as exc:  # noqa: BLE001 - a stamp that fails is recorded
        results["subnormal_policy_stamp"] = {"error": repr(exc)}

    problems = validate_payload(results, backend, rule)
    results["contract"] = {
        "legs_requested": legs, "legs_required": list(LEGS),
        "problems": problems, "passed": not problems,
    }
    results["all_ok"] = bool(not problems)
    results["seconds"] = round(time.time() - started, 2)
    save(results, args.out)
    for problem in problems:
        log(f"CONTRACT: {problem}")
    log(f"all_ok={results['all_ok']} ({results['seconds']} s) -> {args.out}")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
