"""How many corpus slots the folded off-diagonal arm is worth, replayed off the census.

WHAT THIS ANSWERS. ``gate_cuda_folded_offdiag.py`` measures WHETHER the shipped
CUDA off-diagonal ``update_E`` kernel reproduces ``stepping`` on a fold, and on
which arm. This answers WHAT THAT ARM IS WORTH, in the currency the plan counts:
slots of the predicate census, where a slot here is one corpus row's
``update_E`` off-diagonal sub-step.

THE ARM IS NARROWER THAN THE CURL'S AND THE CONSTITUTIVE PAIR'S, and the reason
is a property of the ROW MASK rather than of the grid: the mirror ghost enters
this sub-step through exactly one leg, the PARTNER-axis ``_shift_down``
(stepping.py:1243-1245). So a folded axis that no surviving row slot takes as its
PARTNER cannot change a byte, and one that some slot does take changes bytes on
one plane. The census records the row mask per row
(``cuda_offdiag.row_mask.mask``), which is exactly the fact the split needs, so
the partition is measured off the record rather than assumed uniform.

WHY A REPLAY AND NOT A RE-CENSUS. The census
(``results/cuda_predicate_coverage_2026-08-16_offdiag``) lifted every corpus row
into a real ``Grid``/``Fields``/``PML`` triple and recorded, per row, the
predicate's verdict AND its first refusal string. Re-lifting costs hours and a
MEEP install. Every clause of ``covers_real_pml_offdiag_constitutive`` above the
array tail reads facts the record already carries.

THE COUNTS ARE AN UPPER BOUND, and the reason is built into the replay rather
than only stated: the array, row-volume and coefficient-vector tail cannot be
replayed from the record -- it carries no dtypes, strides, contiguity or base
addresses -- so the stand-ins are built to PASS that tail by construction. A row
whose real arrays would fail it is counted here and would not be served.

THE REPLAY IS VALIDATED BEFORE IT IS USED, on the record's own numbers:

1. every recorded ``covered_modulo_backend`` verdict must reproduce;
2. every recorded ``first_refusal`` string must reproduce;
3. the WIDENED predicate must agree with the shipped one on every UNFOLDED row.

(3) is what makes this a widening rather than a leak: a widened predicate that
also moved an unfolded verdict would have changed something other than the fold,
and the totals alone cannot tell the two apart.

Run::

    python -u replay_folded_offdiag_slots.py --census \\
        results/cuda_predicate_coverage_2026-08-16_offdiag \\
        --out /tmp/folded_offdiag_slots.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)

from meep_gpu.cuda_kernels import coverage  # noqa: E402

#: The refusal the fold question is about, verbatim from ``coverage.py``.
MIRROR_REFUSAL = ("mirror symmetry: the fold changes the stored extent, and the "
                  "extent is what turns a cell index into a coefficient index")

#: HOW MANY SIMULTANEOUS MIRROR PLANES THE GATE ACTUALLY SWEPT. Unlike the
#: constitutive widening, which capped at 2 because its gate stopped there, this
#: gate sweeps ``fold_XYZ_periodic`` and ``fold_XYZ_metallic``, so nothing on this
#: corpus is held back for depth.
FOLD_PLANES_SWEPT = 3


class _CupyNamed:
    """NumPy wearing CuPy's ``__name__`` -- the census's own shim, same spelling."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


class _FakeArray:
    """A float32 C-contiguous volume of the recorded shape, WITHOUT allocating it.

    THAT IS THE UPPER BOUND, stated as an object: a corpus row whose real arrays
    are the wrong dtype, non-contiguous, or aliasing an output is admitted here
    and would be refused in reality.
    """

    def __init__(self, shape: Tuple[int, ...], address: int):
        self.shape = tuple(shape)
        self.dtype = np.float32
        self.flags = type("F", (), {"c_contiguous": True})()
        self.__array_interface__ = {"data": (address, False)}

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


class _Fields:
    """Everything ``covers_real_pml_offdiag_constitutive`` asks a ``Fields`` for.

    THE ROW SLOTS COME FROM THE RECORD, not from a default: the census stores the
    six-flag mask under ``cuda_offdiag.row_mask.mask``, and it is the fact the
    fold split turns on. A replay that invented a full tensor would license the
    corpus's folded rows on an arm the measurement never reached.

    EVERY VOLUME GETS A DISTINCT ADDRESS, so the alias clause is exercised as a
    pass rather than skipped: the record cannot say whether a real row volume
    aliases an output, and pretending it does not is part of the upper bound.
    """

    def __init__(self, configuration: Dict[str, Any], mask: Sequence[int]):
        self._c = configuration
        shape = tuple(int(n) for n in configuration["shape"])
        self.force_complex_fields = bool(
            configuration.get("force_complex_fields", False))
        self.stores_E = bool(configuration.get("stores_E", True))
        nonlinear = bool(configuration.get("has_nonlinearity", False))
        self._chi2_components = ("Ex",) if nonlinear else ()
        self._chi3_components = ()
        self.polarizations = (("p",) if int(
            configuration.get("n_polarizations", 0) or 0) else ())
        self.has_offdiagonal_epsilon = bool(any(mask))
        address = 0x1000
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "Hx", "Hy", "Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                     "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
            address += 0x1000
            setattr(self, name, _FakeArray(shape, address))
        self._inverse = {}
        for name in ("Ex", "Ey", "Ez"):
            address += 0x1000
            self._inverse[name] = _FakeArray(shape, address)
        self._rows: Dict[str, Dict[str, Any]] = {}
        for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
            if not mask[slot]:
                continue
            address += 0x1000
            self._rows.setdefault(row, {})[partner] = _FakeArray(shape, address)

    def inverse_epsilon_for(self, component: str):
        return self._inverse[component]

    def chi1inv_offdiagonal_for(self, component: str):
        return self._rows.get(component, {})

    def condfac_for(self, component: str):
        return None


class _PML:
    def __init__(self, configuration: Dict[str, Any]):
        self.is_active = bool(configuration.get("pml_active", False))
        shape = tuple(int(n) for n in configuration["shape"])
        address = 0x900000
        for axis, name in enumerate("xyz"):
            broadcast = tuple(shape[axis] if a == axis else 1 for a in range(3))
            for half in ("", "_h"):
                for label in ("kps", "kms", "sinv"):
                    address += 0x1000
                    setattr(self, f"{label}_{name}{half}",
                            _FakeArray(broadcast, address))


def stand_ins(configuration: Dict[str, Any], mask: Sequence[int]):
    return (_Fields(configuration, mask), _PML(configuration),
            _Grid(configuration))


# ---------------------------------------------------------------------------
# The widened predicate
# ---------------------------------------------------------------------------

def fold_is_a_live_partner_axis(configuration: Dict[str, Any],
                                mask: Sequence[int]) -> Tuple[bool, List[int]]:
    """Is a folded axis the PARTNER axis of a surviving row slot?

    THE MIRROR RULE ENTERS ONLY THERE (stepping.py:1243-1245), and a component's
    own axis is never its own partner, so this predicate is exactly "the fold can
    change a byte on the arm the gate licensed". It is the same question the
    gate's :func:`~gate_cuda_folded_offdiag.predicted_diverges` asks of a live
    grid, and the gate measured it case by case rather than assuming it.
    """
    mirrored = configuration.get("mirrored", [False] * 3)
    folded = [axis for axis in range(3) if mirrored[axis]]
    for slot, (_row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        if "xyz".index(partner[1]) in folded:
            return True, folded
    return False, folded


def widened_verdict(configuration: Dict[str, Any],
                    mask: Sequence[int]) -> Tuple[bool, str]:
    """``covers_real_pml_offdiag_constitutive`` with the fold clause replaced.

    TWO CLAUSES MOVE, and only two:

    * ``has_symmetry`` / ``mirrored[axis]`` -- were unconditional refusals; become
      a refusal only when a folded axis is some surviving slot's PARTNER axis;
    * the boundary-kind loop -- ``mirror`` is not in ``BC_CODES`` and never will
      be; a folded axis is handed ``metallic``, which is the substitution the gate
      measured the licensed arm identical under, and which the array path's own
      own-axis leg already spells (``_shift_up``'s zero far face,
      stepping.py:1834-1836).

    EVERY OTHER CLAUSE IS THE SHIPPED FUNCTION'S, reached by delegating to it once
    the fold has been rewritten away. That delegation is what stops this from
    being a second, drifting transcription -- and is why the validation asserts
    agreement on every unfolded row.

    THE REWRITE IS A DEVICE OF THE REPLAY, NOT A PROPOSED CHANGE TO THE GRID. In
    a real launch the wall flags stay ``is_metallic and not is_mirrored``, which
    is 0 on a folded axis either way; declaring the axis metallic here only
    reaches the boundary-kind clause, which is the one that has no ``mirror``
    code.
    """
    live_partner, folded = fold_is_a_live_partner_axis(configuration, mask)
    if live_partner:
        return False, ("a folded axis is the PARTNER axis of a surviving row "
                       "slot: the partner-axis shift reads the mirror ghost "
                       "(parity * g[2]) and the kernel has no mirror branch")
    if folded and len(folded) > FOLD_PLANES_SWEPT:
        return False, (f"{len(folded)} mirror planes at once: the gate swept "
                       f"{FOLD_PLANES_SWEPT}")
    if folded:
        configuration = dict(configuration)
        configuration["has_symmetry"] = False
        configuration["mirrored"] = [False] * 3
        metallic = configuration.get("metallic", [False] * 3)
        configuration["metallic"] = [bool(metallic[a] or a in folded)
                                     for a in range(3)]
    fields, pml, grid = stand_ins(configuration, mask)
    return coverage.covers_real_pml_offdiag_constitutive(fields, pml, grid)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def load_rows(census: str) -> List[Dict[str, Any]]:
    """Every census row that carries both a configuration and an off-diagonal verdict.

    Deduplicated on ``(leg, row)``: the parameterised legs are re-emitted into a
    ``_matched`` file and counting a row twice would inflate every total.
    """
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(census, "*.jsonl"))):
        leg = os.path.basename(path).replace(".jsonl", "")
        base = leg.replace("_matched", "")
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "configuration" not in record or "cuda_offdiag" not in record:
                    continue
                key = (base, record.get("row") or record.get("case"))
                if key in rows and "_matched" not in leg:
                    continue
                rows[key] = record
    return list(rows.values())


def row_mask_of(record: Dict[str, Any]) -> Optional[List[int]]:
    block = (record.get("cuda_offdiag") or {}).get("row_mask") or {}
    mask = block.get("mask")
    if not isinstance(mask, list) or len(mask) != len(coverage.OFFDIAG_ROW_SLOTS):
        return None
    return [int(flag) for flag in mask]


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproduce the record before trusting the replay with a number."""
    out: Dict[str, Any] = {
        "slots": 0, "verdict_reproduced": 0, "verdict_mismatch": [],
        "refusal_reproduced": 0, "refusal_mismatch": [],
        "rows_without_a_row_mask": [],
        "unfolded_slots": 0, "unfolded_widened_agrees": 0,
        "unfolded_widened_disagrees": []}
    for record in rows:
        configuration = record["configuration"]
        mask = row_mask_of(record)
        if mask is None:
            out["rows_without_a_row_mask"].append(record.get("row"))
            continue
        fields, pml, grid = stand_ins(configuration, mask)
        recorded = record["cuda_offdiag"]
        out["slots"] += 1
        covered, reason = coverage.covers_real_pml_offdiag_constitutive(
            fields, pml, grid)
        if bool(covered) == bool(recorded.get("covered_modulo_backend")):
            out["verdict_reproduced"] += 1
        else:
            out["verdict_mismatch"].append(
                {"row": record.get("row"),
                 "recorded": recorded.get("covered_modulo_backend"),
                 "replayed": bool(covered), "replayed_reason": reason})
        expected = recorded.get("first_refusal")
        matched = (expected in (None, "", "covered")) if covered else (
            expected == reason)
        if matched:
            out["refusal_reproduced"] += 1
        else:
            out["refusal_mismatch"].append(
                {"row": record.get("row"), "recorded": expected,
                 "replayed": reason})
        if not any(configuration.get("mirrored", [False] * 3)):
            out["unfolded_slots"] += 1
            widened = widened_verdict(configuration, mask)
            if bool(widened[0]) == bool(covered) and widened[1] == reason:
                out["unfolded_widened_agrees"] += 1
            else:
                out["unfolded_widened_disagrees"].append(
                    {"row": record.get("row"),
                     "shipped": [bool(covered), reason],
                     "widened": [bool(widened[0]), widened[1]]})
    out["validated"] = (not out["verdict_mismatch"] and not out["refusal_mismatch"]
                        and not out["unfolded_widened_disagrees"])
    return out


def count(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The slot arithmetic, partitioned by the role the fold plays."""
    out: Dict[str, Any] = {
        "slots_total": 0,
        "slots_covered_today": 0,
        "slots_refused_first_by_mirror_clause": 0,
        "arm_fold_not_a_live_partner": {"slots": 0, "gained": 0, "rows": []},
        "arm_fold_is_a_live_partner": {"slots": 0, "gained": 0, "rows": []},
        "still_refused_after_widening": [],
        "fold_planes_swept": FOLD_PLANES_SWEPT,
    }
    for record in rows:
        configuration = record["configuration"]
        mask = row_mask_of(record)
        if mask is None:
            continue
        fields, pml, grid = stand_ins(configuration, mask)
        out["slots_total"] += 1
        covered, reason = coverage.covers_real_pml_offdiag_constitutive(
            fields, pml, grid)
        if covered:
            out["slots_covered_today"] += 1
            continue
        if reason != MIRROR_REFUSAL:
            continue
        out["slots_refused_first_by_mirror_clause"] += 1
        live_partner, folded = fold_is_a_live_partner_axis(configuration, mask)
        arm = ("arm_fold_is_a_live_partner" if live_partner
               else "arm_fold_not_a_live_partner")
        out[arm]["slots"] += 1
        widened, widened_reason = widened_verdict(configuration, mask)
        entry = {"row": record.get("row"), "row_mask": mask,
                 "mirrored": list(configuration.get("mirrored", [False] * 3)),
                 "metallic": list(configuration.get("metallic", [False] * 3)),
                 "fold_planes": len(folded),
                 "shape": list(configuration["shape"])}
        if widened:
            out[arm]["gained"] += 1
            out[arm]["rows"].append(entry)
        else:
            out["still_refused_after_widening"].append(
                dict(entry, arm=arm, next_refusal=widened_reason))
    out["slots_licensed"] = out["arm_fold_not_a_live_partner"]["gained"]
    out["slots_needing_a_mirror_branch"] = out["arm_fold_is_a_live_partner"]["slots"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = load_rows(args.census)
    print(f"[replay] {len(rows)} census rows carry an off-diagonal verdict",
          flush=True)

    results: Dict[str, Any] = {
        "census": os.path.abspath(args.census),
        "rows": len(rows),
        "validation": validate(rows),
    }
    validation = results["validation"]
    print(f"[replay] validation: verdicts {validation['verdict_reproduced']}/"
          f"{validation['slots']}, refusal strings "
          f"{validation['refusal_reproduced']}/{validation['slots']}, "
          f"unfolded widened agreement {validation['unfolded_widened_agrees']}/"
          f"{validation['unfolded_slots']} -> "
          f"{'VALIDATED' if validation['validated'] else 'NOT VALIDATED'}",
          flush=True)

    if not validation["validated"]:
        results["counts"] = None
        results["why_no_counts"] = ("the replay did not reproduce the record; no "
                                    "slot arithmetic is emitted off it")
    else:
        results["counts"] = count(rows)
        counts = results["counts"]
        print(f"[replay] slots total {counts['slots_total']}, covered today "
              f"{counts['slots_covered_today']}, refused first by the mirror "
              f"clause {counts['slots_refused_first_by_mirror_clause']}",
              flush=True)
        print(f"[replay]   arm fold-not-a-live-partner: "
              f"{counts['arm_fold_not_a_live_partner']['gained']} gained of "
              f"{counts['arm_fold_not_a_live_partner']['slots']} slots "
              f"-> LICENSED", flush=True)
        print(f"[replay]   arm fold-IS-a-live-partner: "
              f"{counts['arm_fold_is_a_live_partner']['slots']} slots, NOT "
              f"licensed: they need a mirror branch in the kernel", flush=True)
        print(f"[replay]   still refused after widening: "
              f"{len(counts['still_refused_after_widening'])}", flush=True)

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"[replay] wrote {args.out}", flush=True)
    return 0 if validation["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
