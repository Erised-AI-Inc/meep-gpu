"""What the hand-CUDA NO-PML NULL arm is worth, replayed off the predicate census.

WHAT THIS ANSWERS. ``gate_cuda_no_pml_null_constitutive.py`` measures WHETHER the
arm is sound -- whether the array path really moves nothing on every configuration
``covers_no_pml_null_constitutive`` admits. This answers WHAT THAT IS WORTH, in the
only currency the plan counts: slots of the predicate census, where a slot is one
(corpus row, sub-step) pair.

WHY A REPLAY AND NOT A RE-CENSUS. The census lifted every corpus row into a real
``Grid``/``Fields``/``PML`` triple and recorded, per row and per sub-step, each
shipped predicate's verdict AND its first refusal string. Re-lifting costs hours and
a MEEP install. Every clause of this arm reads facts the record already carries in
its per-row ``configuration`` block, so the whole predicate can be re-run against
stand-ins built from it -- unlike the kernel predicates, whose array and
coefficient-vector tail cannot be replayed at all.

THE VALIDATION COMES FIRST, on the record's own numbers, and the replay refuses to
report a count until it passes:

1. every recorded ``covers_real_pml_constitutive`` verdict must reproduce from the
   stand-ins (``covered_modulo_backend``);
2. every recorded first-refusal string must reproduce.

Only then is the null predicate run over the same stand-ins.

THE COUNT IS AN UPPER BOUND, and the reasons are written into the replay rather than
only into the prose. THREE facts the record does not carry are supplied by
DERIVATION from the engine's source, and each derivation is where the bound is loose:

* ``Fields._pml_active`` (PML storage switched on) is not recorded. It is taken to be
  ``pml_active``: ``driver.setup_pml`` (driver.py:2250) is the ONLY engine caller of
  ``Fields.enable_pml_storage``, and it refuses a request that absorbs on no face --
  so a row the record marks ``pml_active: false`` has no PML storage. Tight in
  practice; loose if a run ever enabled the storage by hand;
* whether a polarization DRIVES a component is not recorded, only ``n_polarizations``.
  A driven state switches storage on (driver.py:1543-1549), so a row with
  ``stores_E: false`` carries no driven state and the stand-in reports ``drives()``
  False. A row with ``stores_E: true`` is refused by the stores_E clause before the
  polarization clause is reached, so the modelling cannot change its verdict either;
* the record's ``has_nonlinearity`` is ``Fields.has_nonlinearity``, which reads
  ``_chi2_components`` ALONE (fields.py:966-967). A CHI3-ONLY row would therefore be
  admitted here and refused by the shipped predicate, which reads both maps. That is
  a real loosening, and the count reports how many rows could be affected.

WHAT IT IS NOT AN UPPER BOUND ON, and this is the difference from the kernel
families' replays: there is no array or coefficient tail to skip. This arm reads no
dtype, no stride, no contiguity and no coefficient vector, so the usual "a row whose
real arrays would fail the tail is counted here and would not be served" caveat does
not apply. The only stand-in fictions are the three above.

Run::

    python -u replay_null_constitutive_slots.py \\
        --census results/cuda_predicate_coverage_2026-08-16_offdiag \\
        --out /tmp/null_constitutive_slots.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)

from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import no_pml_constitutive as null  # noqa: E402

SIDES = ("H", "E")
SUB_STEP = {"H": "update_H", "E": "update_E"}

#: The refusal the whole question is about, verbatim from
#: ``coverage.covers_real_pml_constitutive``'s inactive-layer clause. Cited by NAME:
#: coverage.py is under active edit by other work and its line numbers move.
LAYER_REFUSAL = "no active PML layer"

#: THE ONE RECORDED REFUSAL THE SHIPPED PREDICATE NO LONGER EMITS. The 2026-08-16
#: census predates the 2026-08-19 fold widening
#: (``coverage.CONSTITUTIVE_FOLD_ADMISSION``), so every folded row's recorded
#: first refusal is a string the function has since retired. Those slots are
#: EXCLUDED from the validation and counted separately, exactly as
#: ``replay_folded_curl_slots.py`` excluded them: a replay that demanded they
#: reproduce would be asserting that a measured widening never happened.
#:
#: IT CANNOT SHADOW THIS ARM'S COUNT, and that is a property of the clause ORDER
#: rather than a hope: ``covers_real_pml_constitutive`` tests the layer at
#: layer BEFORE it reaches the fold clause, so a no-PML row is
#: refused with LAYER_REFUSAL whether it is folded or not -- then and now.
RETIRED_FOLD_REFUSAL_PREFIX = "mirror symmetry:"


class _CupyNamed:
    """NumPy wearing CuPy's ``__name__`` -- the census's own shim, same spelling."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


class _FakeArray:
    """A float32 C-contiguous volume of the recorded shape, WITHOUT allocating it."""

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.float32
        self.flags = type("F", (), {"c_contiguous": True})()

    @property
    def size(self) -> int:
        out = 1
        for n in self.shape:
            out *= int(n)
        return out


class _Grid:
    def __init__(self, configuration: Dict[str, Any]):
        self._c = configuration
        self.xp = _CupyNamed()
        self.shape = tuple(int(n) for n in configuration["shape"])
        self.cylindrical = bool(configuration.get("cylindrical", False))
        self.has_bloch = bool(configuration.get("has_bloch", False))
        self.bfast_active = bool(configuration.get("bfast_active", False))
        self.beta = float(configuration.get("beta", 0.0) or 0.0)

    def has_symmetry(self) -> bool:
        return bool(self._c.get("has_symmetry", False))

    def is_mirrored(self, axis: int) -> bool:
        return bool(self._c.get("mirrored", [False] * 3)[axis])

    def is_axis(self, axis: int) -> bool:
        return bool(self._c.get("is_axis", [False] * 3)[axis])

    def is_metallic(self, axis: int) -> bool:
        return bool(self._c.get("metallic", [False] * 3)[axis])


class _State:
    """A polarization stand-in. See the module docstring for what ``drives`` models."""

    def __init__(self, driven: bool):
        self._driven = bool(driven)

    def drives(self, component: str) -> bool:
        return self._driven and component in ("Ex", "Ey", "Ez")

    def driven(self) -> Tuple[str, ...]:
        return ("Ex", "Ey", "Ez") if self._driven else ()


class _Fields:
    def __init__(self, configuration: Dict[str, Any]):
        self._c = configuration
        shape = tuple(int(n) for n in configuration["shape"])
        self.force_complex_fields = bool(configuration.get("force_complex_fields", False))
        self.stores_E = bool(configuration.get("stores_E", True))
        nonlinear = bool(configuration.get("has_nonlinearity", False))
        self._chi2_components = ("Ex",) if nonlinear else ()
        self._chi3_components = ()
        self.has_offdiagonal_epsilon = bool(
            configuration.get("has_offdiagonal_epsilon", False))
        # DERIVED, not recorded: driver.setup_pml (driver.py:2250) is the only engine
        # caller of enable_pml_storage and it refuses a layer that absorbs nowhere.
        self._pml_active = bool(configuration.get("pml_active", False))
        # DERIVED, not recorded: a driven state forces stored E (driver.py:1543-1549).
        self.polarizations = tuple(
            _State(self.stores_E)
            for _ in range(int(configuration.get("n_polarizations", 0) or 0)))
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "Hx", "Hy", "Hz",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            setattr(self, name, _FakeArray(shape))

    def condfac_for(self, component: str):
        electric = bool(self._c.get("has_conductivity", False))
        magnetic = bool(self._c.get("has_magnetic_conductivity", False))
        if component.startswith("D") and electric:
            return object()
        if component.startswith("B") and magnetic:
            return object()
        return None

    def inverse_epsilon_for(self, component: str):
        return _FakeArray(tuple(int(n) for n in self._c["shape"]))


class _PML:
    def __init__(self, configuration: Dict[str, Any]):
        self.is_active = bool(configuration.get("pml_active", False))
        shape = tuple(int(n) for n in configuration["shape"])
        for axis, name in enumerate("xyz"):
            broadcast = tuple(shape[axis] if a == axis else 1 for a in range(3))
            for half in ("", "_h"):
                for label in ("kps", "kms", "sinv"):
                    setattr(self, f"{label}_{name}{half}", _FakeArray(broadcast))


def stand_ins(configuration: Dict[str, Any]):
    return _Fields(configuration), _PML(configuration), _Grid(configuration)


def load_rows(census: str) -> List[Dict[str, Any]]:
    """Every census row carrying a configuration and a constitutive verdict.

    Deduplicated on ``(leg, row)``: the parameterised legs are re-emitted into a
    ``_matched`` file and counting a row twice would inflate every total here.
    """
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(census, "*.jsonl"))):
        leg = os.path.basename(path).replace(".jsonl", "")
        base = leg.replace("_matched", "")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "configuration" not in record or "cuda_constitutive" not in record:
                    continue
                key = (base, record.get("row") or record.get("case"))
                if key in rows and "_matched" not in leg:
                    continue
                rows[key] = record
    return list(rows.values())


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproduce the record before trusting the replay with a number."""
    out: Dict[str, Any] = {
        "slots": 0, "verdict_reproduced": 0, "verdict_mismatch": [],
        "refusal_reproduced": 0, "refusal_mismatch": [],
        "slots_excluded_retired_fold_refusal": 0,
    }
    for record in rows:
        fields, pml, grid = stand_ins(record["configuration"])
        for side in SIDES:
            recorded = record["cuda_constitutive"].get(SUB_STEP[side])
            if recorded is None:
                continue
            if str(recorded.get("first_refusal") or "").startswith(
                    RETIRED_FOLD_REFUSAL_PREFIX):
                out["slots_excluded_retired_fold_refusal"] += 1
                continue
            out["slots"] += 1
            covered, reason = coverage.covers_real_pml_constitutive(
                fields, pml, grid, side)
            if bool(covered) == bool(recorded.get("covered_modulo_backend")):
                out["verdict_reproduced"] += 1
            else:
                out["verdict_mismatch"].append(
                    {"row": record.get("row"), "side": side,
                     "recorded": recorded.get("covered_modulo_backend"),
                     "replayed": bool(covered), "replayed_reason": reason})
            expected = recorded.get("first_refusal")
            matched = (expected in (None, "", "covered")) if covered else (expected == reason)
            if matched:
                out["refusal_reproduced"] += 1
            else:
                out["refusal_mismatch"].append(
                    {"row": record.get("row"), "side": side,
                     "recorded": expected, "replayed": reason})
    out["validated"] = not out["verdict_mismatch"] and not out["refusal_mismatch"]
    return out


def _sibling_admits(fields, pml, grid, side: str) -> Tuple[bool, List[str]]:
    """Does ANY shipped CUDA constitutive predicate admit this slot today?

    THE DELTA HAS TO BE MEASURED THIS WAY AND NOT OFF THE FIRST REFUSAL, and the
    first version of this replay got it wrong in exactly that way -- it counted only
    slots whose recorded first refusal was ``no active PML layer`` and reported 13
    where the answer is 19. ``covers_real_pml_constitutive`` tests
    ``force_complex_fields`` BEFORE the layer, so the
    six complex-storage no-PML rows are refused for their STORAGE and never name the
    layer at all -- while this arm, which has no storage clause because it reads no
    array, admits their ``update_H`` regardless. A count keyed on someone else's
    first refusal measures the order of someone else's clauses.
    """
    admitters: List[str] = []
    for name, call in (
            ("covers_real_pml_constitutive",
             lambda: coverage.covers_real_pml_constitutive(fields, pml, grid, side)),
            ("covers_real_pml_offdiag_constitutive",
             lambda: (coverage.covers_real_pml_offdiag_constitutive(fields, pml, grid)
                      if side == "E" else (False, "H has no off-diagonal kernel"))),
            ("covers_real_pml_complex_constitutive",
             lambda: coverage.covers_real_pml_complex_constitutive(
                 fields, pml, grid, side, "NAIVE"))):
        try:
            covered, _ = call()
        except Exception:  # noqa: BLE001 - a predicate that raised did not admit
            covered = False
        if covered:
            admitters.append(name)
    return bool(admitters), admitters


def count(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The slot arithmetic. Every number here is a DELTA, never a project total."""
    out: Dict[str, Any] = {
        "rows": len(rows),
        "slots_total": 0,
        "slots_a_shipped_cuda_predicate_admits_today": 0,
        "slots_this_arm_admits": 0,
        "slots_this_arm_adds": 0,
        "slots_claimed_twice": [],
        "by_side": {"H": 0, "E": 0},
        "no_pml_rows": [],
        "rows_taken_at_both_constitutive_sub_steps": [],
        "rows_taken_at_every_sub_step": [],
        "curl_verdicts_on_the_arm_s_rows": {},
    }
    for record in rows:
        configuration = record["configuration"]
        fields, pml, grid = stand_ins(configuration)
        added: List[str] = []
        for side in SIDES:
            if record["cuda_constitutive"].get(SUB_STEP[side]) is None:
                continue
            out["slots_total"] += 1
            theirs, admitters = _sibling_admits(fields, pml, grid, side)
            out["slots_a_shipped_cuda_predicate_admits_today"] += int(theirs)
            mine, _ = null.covers_no_pml_null_constitutive(fields, pml, grid, side)
            if not mine:
                continue
            out["slots_this_arm_admits"] += 1
            if theirs:
                out["slots_claimed_twice"].append(
                    {"row": record.get("row"), "side": side, "admitters": admitters})
                continue
            out["slots_this_arm_adds"] += 1
            out["by_side"][side] += 1
            added.append(SUB_STEP[side])
        if not added:
            continue
        # THE WHOLE-STEP QUESTION, asked rather than assumed. A sub-step gained is
        # not a run taken over: the CUDA curl families all require an ACTIVE layer,
        # so a no-PML row's step_B/step_D stay on the array path and the row is not
        # covered end to end. The curl verdicts are re-run here rather than read off
        # the record, because the record predates the fold widening.
        curls = {}
        for sub_step in ("step_B", "step_D"):
            try:
                covered, reason = coverage.covers_real_pml_curl(
                    fields, pml, grid, sub_step)
            except Exception as exc:  # noqa: BLE001
                covered, reason = False, f"raised {type(exc).__name__}"
            curls[sub_step] = [bool(covered), str(reason)]
        row_entry = {
            "row": record.get("row"),
            "stores_E": bool(configuration.get("stores_E", True)),
            "force_complex_fields": bool(
                configuration.get("force_complex_fields", False)),
            "added": added,
            "curls": curls,
        }
        out["no_pml_rows"].append(row_entry)
        if set(added) == {"update_H", "update_E"}:
            out["rows_taken_at_both_constitutive_sub_steps"].append(record.get("row"))
        if set(added) == {"update_H", "update_E"} and all(
                curls[name][0] for name in curls):
            out["rows_taken_at_every_sub_step"].append(record.get("row"))
    out["curl_verdicts_on_the_arm_s_rows"] = {
        "rows": len(out["no_pml_rows"]),
        "rows_with_both_curls_covered": len(out["rows_taken_at_every_sub_step"]),
        "distinct_curl_refusals": sorted({
            row["curls"][name][1] for row in out["no_pml_rows"]
            for name in ("step_B", "step_D") if not row["curls"][name][0]}),
    }
    out["no_pml_row_count"] = len(out["no_pml_rows"])
    out["rows_with_stored_e"] = sum(int(r["stores_E"]) for r in out["no_pml_rows"])
    out["rows_with_complex_storage"] = sum(
        int(r["force_complex_fields"]) for r in out["no_pml_rows"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = load_rows(args.census)
    print(f"[replay] {len(rows)} census rows carrying a constitutive verdict",
          flush=True)
    validation = validate(rows)
    print(f"[replay] validation: {validation['verdict_reproduced']}/"
          f"{validation['slots']} verdicts, "
          f"{validation['refusal_reproduced']}/{validation['slots']} refusals, "
          f"validated={validation['validated']}", flush=True)
    payload: Dict[str, Any] = {"census": args.census, "validation": validation}
    if validation["validated"]:
        payload["count"] = count(rows)
        block = payload["count"]
        print(f"[replay] slots total {block['slots_total']}; a shipped CUDA "
              f"predicate admits {block['slots_a_shipped_cuda_predicate_admits_today']}"
              f"; THIS ARM ADMITS {block['slots_this_arm_admits']} and ADDS "
              f"{block['slots_this_arm_adds']} (H {block['by_side']['H']}, "
              f"E {block['by_side']['E']}) across {block['no_pml_row_count']} rows; "
              f"claimed twice {len(block['slots_claimed_twice'])}; both constitutive "
              f"sub-steps on "
              f"{len(block['rows_taken_at_both_constitutive_sub_steps'])} rows; "
              f"EVERY sub-step on {len(block['rows_taken_at_every_sub_step'])}",
              flush=True)
    else:
        print("[replay] REFUSING to report a count: the replay does not reproduce "
              "the record", flush=True)
    payload["upper_bound"] = (
        "UPPER BOUND. Three facts are derived rather than recorded: "
        "Fields._pml_active (taken as pml_active, driver.py:2250), whether a "
        "polarization drives (taken as stores_E, driver.py:1543-1549), and "
        "chi3-only nonlinearity (the record's has_nonlinearity reads _chi2 alone, "
        "fields.py:966-967). There is NO array or coefficient tail to skip: this "
        "arm reads no dtype, stride, contiguity or coefficient vector.")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"[replay] -> {args.out}", flush=True)
    return 0 if validation["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
