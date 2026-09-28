"""What the complex Dcyl constitutive admission is worth, replayed off the census.

WHAT THIS ANSWERS. ``gate_cuda_complex_cylindrical_constitutive.py`` measures
WHETHER the shipped CUDA complex constitutive pair reproduces ``stepping`` on a
Dcyl grid. This answers WHAT THAT IS WORTH, in the only currency the plan counts:
slots of the predicate census, where a slot is one (corpus row, sub-step) pair.

WHY A REPLAY AND NOT A RE-CENSUS. The census
(``results/cuda_predicate_coverage_2026-08-20_all_families``) lifted every corpus
row into a real ``Grid``/``Fields``/``PML`` triple and recorded, per row and per
sub-step, each family's verdict AND its first refusal string. Re-lifting costs
hours and a MEEP install. Every clause of
``covers_real_pml_complex_constitutive`` above the array tail reads facts the
record already carries in its per-row ``configuration`` block, so those clauses
can be re-run against stand-ins.

THE COUNTS ARE AN UPPER BOUND, and the reason is written into the replay rather
than only into the prose: the complex-volume and coefficient-vector tail
(``_complex_volume_problem`` / ``_array_problem`` / ``_coefficient_vector_problem``)
cannot be replayed from the record -- it carries no dtypes, strides or contiguity
-- so the stand-ins are built to PASS that tail by construction. A row whose real
arrays would fail it is counted here and would not be served in reality.

THE REPLAY IS VALIDATED BEFORE IT IS USED, on the record's own numbers:

1. every recorded ``covered_modulo_backend`` verdict for ``cuda_complex`` at
   ``update_H`` and ``update_E`` must reproduce under the BEFORE predicate;
2. every recorded ``first_refusal`` string must reproduce under it;
3. the SHIPPED (after) predicate must agree with the BEFORE one on every
   NON-CYLINDRICAL row. That is the clause that makes this a widening rather than
   a leak: a widened predicate that also changed a Cartesian verdict would have
   moved something other than the Dcyl clause, and the totals alone cannot tell
   the two apart;
4. the CURL must be unchanged everywhere, cylindrical rows included -- the whole
   point of the parameter is that only one caller passes it.

WHAT IT WRITES. A FRESH census directory whose ``examples.jsonl`` / ``tests.jsonl``
carry the replayed ``cuda_complex`` constitutive verdicts and are otherwise the
recorded bytes, so the SAME union analyzer can be run over it positionally and the
before/after union numbers are produced by one instrument rather than two. The
original census directory is never written to.

Run::

    python -u replay_complex_cylindrical_slots.py \\
        --census results/cuda_predicate_coverage_2026-08-20_all_families \\
        --out results/cuda_complex_cylindrical_constitutive_2026-08-20/slots.json \\
        --patched-census results/cuda_complex_cylindrical_constitutive_2026-08-20/census_after
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)

from meep_gpu.cuda_kernels import coverage  # noqa: E402

#: The two constitutive sub-steps, and the census key each is filed under.
SIDES: Tuple[Tuple[str, str], ...] = (("H", "update_H"), ("E", "update_E"))
CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: The refusal this replay is about, verbatim from the shipped clause.
DCYL_REFUSAL = "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules"

#: The policy and licence the CENSUS BATTERY used, restated here so the replay
#: asks the predicate the same question the record answered. ``predicate_battery``
#: loads the 'keep' probe artifact and runs ``expansion_license`` over it; the
#: verdict that comes back classifies FMA_V1 on a measured basis with no refusals,
#: which is the shape reproduced here. A PLACEHOLDER STRING WOULD BE REFUSED with
#: "the expansion licence is str, not the verdict dict" -- a refusal about the
#: harness reported as one about the corpus, and the reason this is a dict.
COMPLEX_POLICY = "keep"
COMPLEX_LICENCE: Dict[str, Any] = {
    "arm": "FMA_V1", "expansion": 1, "basis": "measured", "refusals": [],
    "policy_resolved": COMPLEX_POLICY,
}


class _CupyNamed:
    """NumPy wearing CuPy's ``__name__`` -- the census's own shim, same spelling."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


class _FakeArray:
    """A C-contiguous volume of the recorded shape and dtype, WITHOUT allocating it.

    THAT IS THE UPPER BOUND, stated as an object: a corpus row whose real volumes
    are the wrong dtype or non-contiguous is admitted here and would be refused in
    reality.
    """

    def __init__(self, shape: Sequence[int], dtype=np.complex64):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = dtype
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
        self.k_point = tuple(float(k) for k in
                             (configuration.get("k_point") or (0.0, 0.0, 0.0)))

    def has_symmetry(self) -> bool:
        return bool(self._c.get("has_symmetry", False))

    def is_mirrored(self, axis: int) -> bool:
        return bool(self._c.get("mirrored", [False] * 3)[axis])

    def is_axis(self, axis: int) -> bool:
        return bool(self._c.get("is_axis", [False] * 3)[axis])

    def is_metallic(self, axis: int) -> bool:
        return bool(self._c.get("metallic", [False] * 3)[axis])

    def bloch_phase(self, axis: int):
        """The per-axis wrap factor, reconstructed from the recorded k_point.

        THE VALUE IS NOT RECORDED, only ``k_point`` and ``has_bloch``, so this
        answers the ONE question the predicate asks of it: is this axis phased or
        not (``phase is not None``). It never asks for the number. Reconstructing a
        number instead would be inventing a fact the record does not hold.
        """
        if not self.has_bloch:
            return None
        return complex(-1.0, 0.0) if float(self.k_point[axis]) != 0.0 else None


class _Fields:
    def __init__(self, configuration: Dict[str, Any]):
        self._c = configuration
        shape = tuple(int(n) for n in configuration["shape"])
        self.force_complex_fields = bool(
            configuration.get("force_complex_fields", False))
        self.stores_E = bool(configuration.get("stores_E", True))
        polarizations = int(configuration.get("n_polarizations", 0) or 0)
        self.polarizations = tuple(range(polarizations))
        self.has_polarizations = bool(polarizations)
        self.has_offdiagonal_epsilon = bool(
            configuration.get("has_offdiagonal_epsilon", False))
        nonlinear = bool(configuration.get("has_nonlinearity", False))
        self._chi2_components = ("Ex",) if nonlinear else ()
        self._chi3_components = ()
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "Hx", "Hy", "Hz",
                     "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez",
                     "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            setattr(self, name, _FakeArray(shape, np.complex64))
        self._shape = shape

    def condfac_for(self, component: str):
        electric = bool(self._c.get("has_conductivity", False))
        magnetic = bool(self._c.get("has_magnetic_conductivity", False))
        if component.startswith("D") and electric:
            return object()
        if component.startswith("B") and magnetic:
            return object()
        return None

    def inverse_epsilon_for(self, component: str):
        # FLOAT32 UNDER COMPLEX STORAGE, which is the predicate's own clause and
        # not an accident of this stand-in (stepping.py:41-50 against
        # fields.py:1203-1204).
        return _FakeArray(self._shape, np.float32)


class _PML:
    def __init__(self, configuration: Dict[str, Any]):
        self.is_active = bool(configuration.get("pml_active", False))
        shape = tuple(int(n) for n in configuration["shape"])
        for axis, name in enumerate("xyz"):
            broadcast = tuple(shape[axis] if a == axis else 1 for a in range(3))
            for half in ("", "_h"):
                for label in ("kps", "kms", "sinv"):
                    setattr(self, f"{label}_{name}{half}",
                            _FakeArray(broadcast, np.float32))


def stand_ins(configuration: Dict[str, Any]):
    return _Fields(configuration), _PML(configuration), _Grid(configuration)


# ---------------------------------------------------------------------------
# THE TWO STATES OF THE PREDICATE
# ---------------------------------------------------------------------------

def verdict_before(configuration: Dict[str, Any], side: str) -> Tuple[bool, str]:
    """The pre-2026-08-20 predicate: every Dcyl run refused.

    RECONSTRUCTED BY DELEGATION, never by re-transcribing the other clauses. The
    change was exactly one argument, so the old behaviour is the old argument
    followed by the shipped function: the licence clause first (it is first in the
    shipped predicate too), then ``_complex_grid_refusal`` with
    ``admit_cylindrical`` FALSE, then the shipped predicate for everything below.
    A second copy of the fifteen clauses under it is a second thing to drift, and
    the validation's non-cylindrical-agreement leg is what proves the delegation
    is faithful.
    """
    fields, pml, grid = stand_ins(configuration)
    refusal = coverage.complex_expansion_refusal(COMPLEX_LICENCE, COMPLEX_POLICY)
    if refusal is not None:
        return False, refusal
    refusal = coverage._complex_grid_refusal(fields, pml, grid,
                                             admit_cylindrical=False)
    if refusal is not None:
        return False, refusal
    return coverage.covers_real_pml_complex_constitutive(
        fields, pml, grid, side, COMPLEX_LICENCE, COMPLEX_POLICY)


def verdict_after(configuration: Dict[str, Any], side: str) -> Tuple[bool, str]:
    """The SHIPPED predicate, asked directly."""
    fields, pml, grid = stand_ins(configuration)
    return coverage.covers_real_pml_complex_constitutive(
        fields, pml, grid, side, COMPLEX_LICENCE, COMPLEX_POLICY)


def curl_verdict(configuration: Dict[str, Any], sub_step: str) -> Tuple[bool, str]:
    fields, pml, grid = stand_ins(configuration)
    return coverage.covers_real_pml_complex_curl(
        fields, pml, grid, sub_step, COMPLEX_LICENCE, COMPLEX_POLICY)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def load_rows(root: str) -> List[Dict[str, Any]]:
    """The census's own row assembly, restated: examples + tests + the recovery file."""
    rows: List[Dict[str, Any]] = []
    for leg in ("examples", "tests"):
        path = os.path.join(root, f"{leg}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    recovered = os.path.join(root, "tests_param_matched.jsonl")
    if os.path.exists(recovered):
        with open(recovered, encoding="utf-8") as handle:
            replacements = {}
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    replacements[(row.get("leg"), row.get("row"))] = row
        rows = [replacements.get((r.get("leg"), r.get("row")), r) for r in rows]
    return rows


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproduce the record before pricing anything against it."""
    out: Dict[str, Any] = {
        "rows": len(rows), "slots_checked": 0,
        "verdict_mismatches": [], "refusal_mismatches": [],
        "non_cylindrical_moved": [], "curl_moved": [],
    }
    for row in rows:
        configuration = row.get("configuration") or {}
        if not configuration.get("shape"):
            continue
        cylindrical = bool(configuration.get("cylindrical", False))
        block = row.get("cuda_complex") or {}
        for side, key in SIDES:
            entry = block.get(key)
            if entry is None:
                continue
            out["slots_checked"] += 1
            covered, reason = verdict_before(configuration, side)
            if bool(covered) != bool(entry.get("covered_modulo_backend")):
                out["verdict_mismatches"].append(
                    {"row": row.get("row"), "side": side,
                     "recorded": entry.get("covered_modulo_backend"),
                     "replayed": bool(covered), "replayed_reason": reason})
            recorded_reason = entry.get("first_refusal") or ""
            if not covered and recorded_reason and reason != recorded_reason:
                out["refusal_mismatches"].append(
                    {"row": row.get("row"), "side": side,
                     "recorded": recorded_reason, "replayed": reason})
            after_covered, after_reason = verdict_after(configuration, side)
            if not cylindrical and (bool(after_covered) != bool(covered)):
                out["non_cylindrical_moved"].append(
                    {"row": row.get("row"), "side": side,
                     "before": bool(covered), "after": bool(after_covered),
                     "after_reason": after_reason})
        for sub_step in CURL_SUB_STEPS:
            entry = block.get(sub_step)
            if entry is None:
                continue
            covered, reason = curl_verdict(configuration, sub_step)
            if bool(covered) != bool(entry.get("covered_modulo_backend")):
                out["curl_moved"].append(
                    {"row": row.get("row"), "sub_step": sub_step,
                     "recorded": entry.get("covered_modulo_backend"),
                     "replayed": bool(covered), "replayed_reason": reason})
    out["sound"] = not (out["verdict_mismatches"] or out["refusal_mismatches"]
                        or out["non_cylindrical_moved"] or out["curl_moved"])
    return out


def price(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Which slots the widening moves, and what every OTHER family said about them.

    A SLOT THE COMPLEX PAIR NEWLY ADMITS IS ONLY A GAIN IF NOTHING ELSE ALREADY
    SERVED IT. The union census takes the union per slot, so a flip on a slot some
    other family already covers buys nothing -- and would additionally be an
    OVERLAP, which the analyzer prints as a finding because the arms are meant to
    partition.
    """
    #: Which families may serve each constitutive sub-step, in the analyzer's own
    #: order. Restated rather than imported: the analyzer lives inside a results
    #: directory and importing it would bind this tool to one census.
    union = {"update_H": ("cuda_constitutive", "cuda_complex", "cuda_no_pml"),
             "update_E": ("cuda_constitutive", "cuda_offdiag", "cuda_complex",
                          "cuda_no_pml")}
    out: Dict[str, Any] = {"flipped": [], "flipped_but_already_served": [],
                           "gained": 0, "rows_gained": set()}
    for row in rows:
        configuration = row.get("configuration") or {}
        if not configuration.get("shape"):
            continue
        for side, key in SIDES:
            entry = (row.get("cuda_complex") or {}).get(key)
            if entry is None:
                continue
            before, before_reason = verdict_before(configuration, side)
            after, after_reason = verdict_after(configuration, side)
            if before or not after:
                continue
            others = []
            for family in union[key]:
                if family == "cuda_complex":
                    continue
                block = row.get(family)
                if block is None:
                    continue
                # cuda_offdiag files its single verdict at the block's top level.
                candidate = block.get(key) if key in block else block
                if candidate and candidate.get("covered_modulo_backend"):
                    others.append(family)
            record = {"row": row.get("row"), "leg": row.get("leg"), "side": side,
                      "sub_step": key, "before_refusal": before_reason,
                      "after": after_reason,
                      "cylindrical": bool(configuration.get("cylindrical")),
                      "shape": configuration.get("shape"),
                      "already_served_by": others}
            if others:
                out["flipped_but_already_served"].append(record)
            else:
                out["flipped"].append(record)
                out["gained"] += 1
                out["rows_gained"].add(row.get("row"))
    out["rows_gained"] = sorted(out["rows_gained"])
    out["every_gained_slot_was_refused_for_dcyl_and_nothing_else"] = all(
        entry["before_refusal"] == DCYL_REFUSAL for entry in out["flipped"])
    return out


def write_patched_census(source: str, destination: str,
                         rows_by_key: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """A FRESH census directory carrying the replayed constitutive verdicts.

    THE ORIGINAL IS NEVER WRITTEN TO. What is produced is a second directory the
    SAME union analyzer can be run over positionally, so the before and after
    numbers come from one instrument rather than from this file's own arithmetic.
    Only the two ``cuda_complex`` constitutive entries are rewritten; every other
    byte of every row is the recorded one.
    """
    os.makedirs(destination, exist_ok=True)
    written: Dict[str, Any] = {"files": {}, "entries_rewritten": 0}
    for name in ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl"):
        path = os.path.join(source, name)
        if not os.path.exists(path):
            continue
        out_lines: List[str] = []
        rewritten = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                configuration = row.get("configuration") or {}
                block = row.get("cuda_complex")
                if block and configuration.get("shape"):
                    for side, key in SIDES:
                        if key not in block:
                            continue
                        covered, reason = verdict_after(configuration, side)
                        block[key]["covered_modulo_backend"] = bool(covered)
                        block[key]["first_refusal"] = (
                            None if covered else reason)
                        block[key]["replayed_by"] = (
                            "replay_complex_cylindrical_slots.py")
                        rewritten += 1
                out_lines.append(json.dumps(row))
        with open(os.path.join(destination, name), "w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines) + "\n")
        written["files"][name] = {"rows": len(out_lines), "rewritten": rewritten}
        written["entries_rewritten"] += rewritten
    # The analyzer is copied in beside the data so the patched directory can be
    # analyzed by the SAME instrument, positionally, with no path juggling.
    analyzer = os.path.join(source, "analyze_cuda_coverage.py")
    if os.path.exists(analyzer):
        shutil.copy2(analyzer, os.path.join(destination, "analyze_cuda_coverage.py"))
        written["analyzer_copied"] = True
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patched-census", default=None,
                        help="write a fresh census directory carrying the "
                             "replayed verdicts, for the union analyzer")
    args = parser.parse_args(argv)

    rows = load_rows(args.census)
    print(f"[replay] {len(rows)} census rows", flush=True)

    validation = validate(rows)
    print(f"[replay] validation: slots={validation['slots_checked']} "
          f"verdict_mismatches={len(validation['verdict_mismatches'])} "
          f"refusal_mismatches={len(validation['refusal_mismatches'])} "
          f"non_cylindrical_moved={len(validation['non_cylindrical_moved'])} "
          f"curl_moved={len(validation['curl_moved'])} "
          f"sound={validation['sound']}", flush=True)

    results: Dict[str, Any] = {
        "census": os.path.abspath(args.census),
        "predicate": "cuda_kernels.coverage.covers_real_pml_complex_constitutive",
        "clause": DCYL_REFUSAL,
        "validation": validation,
        "upper_bound_note": (
            "the complex-volume and coefficient-vector tail cannot be replayed "
            "from the record; the stand-ins pass it by construction, so a row "
            "whose real volumes would be refused is counted here"),
    }
    if not validation["sound"]:
        results["status"] = "refused: the replay does not reproduce the record"
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)
        print("[replay] REFUSED: the replay does not reproduce the record",
              flush=True)
        return 2

    priced = price(rows)
    results["slots"] = priced
    print(f"[replay] slots gained by the complex constitutive pair: "
          f"{priced['gained']} over {len(priced['rows_gained'])} rows; "
          f"already served elsewhere: "
          f"{len(priced['flipped_but_already_served'])}", flush=True)
    print(f"[replay] every gained slot was refused for Dcyl and nothing else: "
          f"{priced['every_gained_slot_was_refused_for_dcyl_and_nothing_else']}",
          flush=True)

    if args.patched_census:
        results["patched_census"] = write_patched_census(
            args.census, args.patched_census, {})
        results["patched_census"]["path"] = os.path.abspath(args.patched_census)
        print(f"[replay] patched census written to {args.patched_census}; run the "
              f"union analyzer over it POSITIONALLY", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
