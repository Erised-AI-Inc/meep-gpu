"""Is fu_Dz's 1.922e-28 the flush cliff, or an arithmetic difference? Replayed on the HOST.

WHY THIS AND NOT A DEVICE RUN. The question is what MPS's native denormal flushing does to
a stored split-field PML INTERMEDIATE. That is answerable without MPS: run the array path
twice on the same lift -- once as it ships, once with every subnormal INTERMEDIATE of the
PML recurrence flushed to zero exactly where a flushing executor would flush it -- and see
whether the second reproduces the magnitude and the element count the device measured. If it
does, the 1.922e-28 is the cliff and nothing else. This is the instrument
reference_subnormal_intermediate_instrument_gap prescribes, in the direction that settles a
bound rather than the direction that bands a row.

THE RECURRENCE, meep_gpu/stepping.py:1975-1983 (_apply_pml_update):

    fu_previous = fu.copy()
    fu *= kms ;  fu -= curl ;  fu *= sinv                       # stage 1 -> fu_D*
    field *= kms_u ;  field += fu ;  field -= fu_previous ;  field *= sinv_u   # stage 2 -> D*

Stage 1 writes the stored intermediate. Stage 2 consumes it as a DIFFERENCE of successive
values, which is why every other component's intermediate and consumer carry the same
differing magnitude while fu_Dz's sits ~3000x above Dz's.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALLEST_NORMAL_F32 = np.float32(1.1754943508222875e-38)
STEPS = int(os.environ.get("FU_DZ_STEPS", "24"))

import gate_dispatch_metal_route as route          # noqa: E402
from meep_gpu import stepping                      # noqa: E402

_REAL = stepping._apply_pml_update
_TALLY = {"calls": 0, "flushed_words": 0, "arrays_touched": 0}


def _flushing_pml_update(field, curl, kms, sinv, kms_u, sinv_u, fu, scratch=None):
    """The shipped recurrence with stage 1's OUTPUT flushed where it is subnormal.

    A flushing executor turns a subnormal RESULT into zero at the instruction that
    produces it. Stage 1's last operation is ``fu *= sinv``, so its output is the
    intermediate a flushing device would zero; stage 2 then reads the zeroed value. The
    flush is a VALUE test (0 < |x| < smallest normal), not a bit mask, because the band is
    a property of the value and an exact difference of two normals can land in it.
    """
    fu_previous = fu.copy()
    fu *= kms
    fu -= curl
    fu *= sinv
    band = (np.abs(fu) > 0) & (np.abs(fu) < SMALLEST_NORMAL_F32)
    hit = int(np.count_nonzero(band))
    _TALLY["calls"] += 1
    if hit:
        _TALLY["flushed_words"] += hit
        _TALLY["arrays_touched"] += 1
        fu[band] = 0
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u


def _state(driver):
    import gate_dispatch_end_to_end as e2e
    return e2e.collect_state(driver)


def main() -> int:
    # THE CASE IS SELECTABLE because the question recurred on a second case. It was
    # hardcoded to pml_3d_diagonal when the instrument was built for fu_Dz; the beta
    # electric pair's release on 2026-09-12 put special_kz_2d's fields.Dy over the
    # field ceiling at 1.894e-28 while its own fu_Dy sat at 1.896e-28 UNDER the
    # intermediate ceiling, which is the same question asked of a different array.
    # The default is the original case, so the invocation that settled fu_Dz is
    # unchanged.
    case = os.environ.get("FU_DZ_CASE", "pml_3d_diagonal")
    print(f"lifting {case} twice on the ARRAY path (NumPy, no device)", flush=True)
    plain, _m, _u = route._lift(case, None)
    flushed, _m2, _u2 = route._lift(case, None)

    import gate_dispatch_end_to_end as e2e
    pre = e2e.compare_state(_state(plain), _state(flushed))
    print(f"precondition identical: {pre['identical']}", flush=True)
    if not pre["identical"]:
        print("REFUSING: the two lifts differ before a step ran")
        return 2

    plain.run(num_steps=STEPS)
    stepping._apply_pml_update = _flushing_pml_update
    try:
        flushed.run(num_steps=STEPS)
    finally:
        stepping._apply_pml_update = _REAL

    print(f"flush hook: {_TALLY['calls']} stage-1 calls, "
          f"{_TALLY['flushed_words']} subnormal intermediates flushed "
          f"over {_TALLY['arrays_touched']} array-passes", flush=True)

    verdict = e2e.compare_state(_state(plain), _state(flushed))
    print(f"\narrays differing after {STEPS} steps: {verdict['arrays_differing']}", flush=True)
    rows = [e for e in verdict["differences"] if "largest_differing_magnitude" in e]
    rows.sort(key=lambda e: -(e["largest_differing_magnitude"] or 0))
    print(f"{'array':16s} {'words':>6} {'maxmag':>11} {'maxabs':>11} {'zero1':>6} {'subn':>5}")
    for e in rows:
        print(f"{e['array']:16s} {e['words_mismatched']:6d} "
              f"{e['largest_differing_magnitude']:11.3e} {e['max_abs_diff']:11.3e} "
              f"{e.get('elements_one_side_exactly_zero'):6d} "
              f"{e.get('elements_subnormal_magnitude'):5d}")

    out = {"case": case, "steps": STEPS, "tally": _TALLY,
           "arrays_differing": verdict["arrays_differing"],
           "rows": [{k: e.get(k) for k in
                     ("array", "words_mismatched", "largest_differing_magnitude",
                      "max_abs_diff", "max_rel_diff",
                      "elements_one_side_exactly_zero",
                      "elements_subnormal_magnitude")} for e in rows]}
    path = os.environ.get("FU_DZ_OUT", "/tmp/fu_dz_instrument.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {path}", flush=True)
    plain.close()
    flushed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
