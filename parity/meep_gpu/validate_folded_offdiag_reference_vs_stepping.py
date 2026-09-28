"""Pin the folded off-diagonal gate's REFERENCE against ``stepping.py``, on NumPy, no GPU.

The twin of ``validate_fold_reference_vs_stepping.py`` and
``validate_pml_reference_vs_stepping.py``, for the fold ∩ off-diagonal family.
The device gate (``gate_triton_folded_offdiag.py``) will compare the KERNEL
against a reference transcribed by hand from ``stepping.py``; this runs that
reference against the array path itself, which is what stops "bit-identical"
from ever meaning "the kernel reproduces whatever the harness wrote twice".

It runs anywhere — the gate imports ``cupy`` defensively, so its host legs need
no device and no Triton::

    python -u validate_folded_offdiag_reference_vs_stepping.py

Three host legs, and the exit status is the verdict:

* ``reference``     — 16 real ``Grid``/``Fields``/``PML`` configurations with
  real mirror planes and installed off-diagonal rows (including TWO with three
  simultaneous planes), 4 ``update_E`` sub-step calls each, compared BYTE FOR
  BYTE against the transcription, with the state chained through ``f_w``. Each
  configuration additionally runs the six DISCRIMINATING CONTROLS — one per
  ghost arm — and each control must DIFFER somewhere, so a transcription that
  matched by accident cannot pass.
* ``refusals``      — every predicted refusal, by name, 0 admissions, plus the
  two disjointness seams (``offdiag_update_e`` refuses the fold;
  ``symmetry.folded_constitutive_coverage`` refuses the rows) and the
  STALE-DOCSTRING GUARD (the installer does NOT refuse folded rows). Every case
  must be BUILDABLE and must match its reason by name; a case that cannot be
  built fails the leg rather than being skipped.
* ``identity_host`` — the REACHABILITY measurement: MIRROR vs METALLIC ghost
  code on the same grid, same coefficients, same seeds. Identical exactly when
  NO LIVE ROW SLOT takes the folded axis's component as its PARTNER, different
  as soon as one does. The test is at SLOT level: three of the eighteen cases
  are the counterexample that separates it from the coarser row-level reading
  (a live row that is not ``E_a`` whose only surviving slot still never takes
  ``E_a`` as a partner), and a leg carrying none of them fails as vacuous.

NO BYTE-IDENTITY CLAIM ABOUT ANY KERNEL IS MADE HERE. This validator pins a
transcription against the array path. What the compiled kernel does is the
device gate's question and has not been asked yet.

The artifact is written wherever ``--out`` says (default: a temporary path), and
is the same JSON the device gate extends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gate_triton_folded_offdiag as gate  # noqa: E402

HOST_LEGS = ("reference", "refusals", "identity_host")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join(tempfile.gettempdir(),
                             "validate_folded_offdiag_reference.json"))
    args = parser.parse_args(argv)

    status = gate.main(["--legs", ",".join(HOST_LEGS), "--out", args.out])
    with open(args.out) as handle:
        results = json.load(handle)

    print(flush=True)
    print("=" * 78, flush=True)
    reference = results.get("reference", {}).get("numpy", {})
    grids = reference.get("grids", [])
    identical = sum(1 for row in grids if row.get("identical"))
    print(f"reference      {identical}/{len(grids)} configurations bit-identical to "
          f"stepping.update_E over {reference.get('steps_per_grid')} sub-step calls",
          flush=True)
    for name, count in sorted(reference.get("control_grids_differing", {}).items()):
        print(f"    control {name:22s} differs on {count:2d} configuration(s)",
              flush=True)
    vacuous = reference.get("vacuous_controls", [])
    print(f"    vacuous controls: {vacuous or 'none'}", flush=True)

    refusals = results.get("refusals", {})
    print(f"refusals       {sum(1 for c in refusals.get('cases', []) if c.get('pass'))}"
          f"/{len(refusals.get('cases', []))} cases pass, "
          f"{refusals.get('admissions')} admissions", flush=True)

    identity = results.get("identity_host", {})
    print(f"identity_host  {identity.get('positive_cases')} identical / "
          f"{identity.get('negative_cases')} differing "
          f"(reachability agrees with the predicate on "
          f"{sum(1 for c in identity.get('cases', []) if c.get('agrees'))}"
          f"/{len(identity.get('cases', []))}; "
          f"{identity.get('slot_level_counterexamples')} slot-level "
          f"counterexamples)", flush=True)
    print("=" * 78, flush=True)
    print(f"VERDICT: {'PASS' if results.get('host_legs_pass') else 'FAIL'} — "
          f"no byte-identity claim is made for any kernel; no device leg ran.",
          flush=True)
    print(f"artifact: {args.out}", flush=True)
    return 0 if results.get("host_legs_pass") else max(status, 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
