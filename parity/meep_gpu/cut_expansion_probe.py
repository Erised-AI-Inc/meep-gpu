"""Cut the SIX-pattern complex-expansion probe artifact on a CUDA host.

WHY THIS EXISTS. Every gate that binds an ``EXPANSION`` constexpr reads a probe
artifact and refuses without one that covers ITS pattern set. Three writers
exist and each emits a different five-pattern record:

* ``gate_triton_complex.measure_expansion_record`` — the four in
  ``complex_fields.PROBE_PATTERNS`` plus a scalar-broadcast diagnostic;
* ``gate_triton_special_kz.measure_beta_expansion_record`` — those four plus
  ``special_kz.BETA_PROBE_PATTERN``;
* ``gate_triton_folded_complex.measure_folded_expansion_record`` — those four,
  the beta pattern merged in, AND ``folded_complex.PARITY_PROBE_PATTERN``.

Only the third covers the UNION of six. A run that reads one of the first two
finds the folded parity set or a beta set unclassified and refuses by name, and
that refusal has been read as a coverage gap when it is an artifact gap: the
2026-08-16 census measured five candidate artifacts and found ONE class that
licenses anything, with nothing licensing the folded mirror-fill parity set or
either beta set. This driver cuts the record that covers all six, so a gate
refusing afterwards is refusing about the PLATFORM rather than about the file.

It adds no measurement of its own — deliberately. The arithmetic, the candidate
construction, the ambiguity rule and the policy stamp all stay in the gates that
own them, so this file cannot drift away from what the gates check.

Usage (from ``the repository root``, on a host with CuPy and a device)::

    PYTHONPATH=. python -u parity/meep_gpu/cut_expansion_probe.py --out <path>.json

The environment matters and is NOT set here, because getting it wrong must fail
loudly rather than be papered over: ``CUPY_ACCELERATORS=''`` before CuPy is
imported (CUB's reduction dispatch reads a list fixed at import), and a private
``CUPY_CACHE_DIR`` whose name carries the policy it was cut under.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def cut(backend_name: str = "cupy", policy: str = "keep") -> Dict[str, Any]:
    """The six-pattern record, straight from the gate that already merges them.

    THE POLICY IS INSTALLED HERE, BEFORE ANY MEASUREMENT, and that is the whole
    reason this function exists rather than a bare call. Measured 2026-08-17: a
    record cut with no policy installed stamps itself
    ``cupy_default_ftz_flush`` and says in its own warning field that it
    "license[s] nothing" — CuPy appends ``-ftz=true`` to every NVRTC compile
    while the host FPU and Triton decide separately, so the candidate arms and
    the platform bytes are built under different conventions and five of the six
    patterns classify NEITHER. That is not a platform fact; it is a half-applied
    policy, and it looks exactly like a platform that licenses no arm.
    """
    import gate_triton_complex as G  # noqa: PLC0415

    # INSTALL THROUGH THE GATE'S OWN ENTRY POINT, not through
    # subnormal_policy.install_subnormal_policy directly.
    #
    # Measured 2026-08-17: installing directly DOES install the policy — every
    # executor attains it — but it leaves ``gate_triton_complex._FTZ_STRIP``
    # unset, and ``policy_stamp`` reads exactly that memo. The artifact was
    # therefore stamped 'cupy_default_ftz_flush' with the warning "measured
    # WITHOUT any policy installed ... license[s] nothing", while its patterns
    # had in fact been measured under a correctly installed 'keep'. A record
    # whose header contradicts its body is worse than a missing one: a consumer
    # reads the warning, refuses, and the refusal is blamed on the platform —
    # the exact artifact-gap-read-as-platform-gap this driver exists to end.
    #
    # One authority, and it is the one the stamp reads.
    stamp = G.install_ftz_strip(policy)
    print(f"[policy] installed {policy!r} via install_ftz_strip: "
          f"resolved={stamp.get('resolved')} installed={stamp.get('installed')}",
          flush=True)

    import gate_triton_folded_complex as folded  # noqa: PLC0415

    xp = __import__(backend_name)
    return folded.measure_folded_expansion_record(xp, backend_name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="cupy")
    parser.add_argument("--policy", default="keep")
    args = parser.parse_args(argv)

    record = cut(args.backend, args.policy)

    # The union the licence consumers ask for, asserted rather than assumed: a
    # record silently missing a pattern would be written, read, and refused
    # later with the refusal blamed on the platform.
    from meep_gpu.triton_kernels import complex_fields, folded_complex, special_kz  # noqa: PLC0415
    wanted = (set(complex_fields.PROBE_PATTERNS)
              | {special_kz.BETA_PROBE_PATTERN, folded_complex.PARITY_PROBE_PATTERN})
    present = set(record.get("patterns", {}))
    missing = sorted(wanted - present)
    if missing:
        raise SystemExit(
            f"REFUSED to write a partial artifact: {len(present)} patterns present, "
            f"missing {missing}. The consumers of this file refuse without them, "
            f"and a partial record turns an artifact gap into an apparent platform gap.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)

    print(f"backend            : {record.get('backend')}")
    print(f"patterns ({len(present)})      :")
    for name in sorted(present):
        print(f"    {record['patterns'][name]:16s} {name}")
    print(f"subnormal policy   : {record.get('subnormal_policy')}")
    print(f"written            : {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - a driver
    raise SystemExit(main())
