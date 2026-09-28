"""Cut one device-measured expansion record for every complex Triton family.

The folded-complex gate measures the base four orientations plus the special-kz
imaginary coefficient and folded parity orientations. Complex ADE adds the
seventh orientation, ``complex field * Python float``. A user run supplies one
artifact to ``plan_step``; keeping those measurements in separate files would
therefore make the artifact path depend on which arm the planner has not yet
selected. This gate measures all seven in one CuPy process under one installed
subnormal policy and refuses unless every current family resolves to one arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for path in (str(HERE), str(API)):
    if path not in sys.path:
        sys.path.insert(0, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    started = time.time()

    import cupy as cp
    import gate_triton_complex as base
    import gate_triton_complex_ade as ade_gate
    import gate_triton_folded_complex as folded_gate
    from meep_gpu.triton_kernels import complex_ade, complex_fields
    from meep_gpu.triton_kernels import folded_complex

    base.install_ftz_strip()
    record = folded_gate.measure_folded_expansion_record(cp, "cupy")
    record = ade_gate.augment_expansion_probe(record, cp)

    licenses = {
        "base": complex_fields.expansion_license(record),
        "folded_parity": folded_complex.parity_expansion_license(record),
        "special_kz_beta": folded_complex.folded_beta_expansion_license(record),
        "complex_ade": complex_fields.expansion_license(
            record, complex_ade.COMPLEX_ADE_PROBE_PATTERNS),
    }
    policy_reasons = base.ftz_strip_license_reasons()
    failures = list(policy_reasons)
    for name, license_record in licenses.items():
        if license_record.get("expansion") is None:
            failures.append(
                f"{name} licensed no expansion: "
                + "; ".join(license_record.get("refusals", ())))

    record["unified_licenses"] = licenses
    record["unified_pattern_count"] = len(record.get("patterns", {}))
    record["policy_refusals"] = policy_reasons
    record["provenance"] = {
        "gate": digest(Path(__file__)),
        "gate_triton_complex": digest(HERE / "gate_triton_complex.py"),
        "gate_triton_folded_complex": digest(
            HERE / "gate_triton_folded_complex.py"),
        "gate_triton_complex_ade": digest(HERE / "gate_triton_complex_ade.py"),
        "complex_fields": digest(
            API / "meep_gpu/triton_kernels/complex_fields.py"),
        "folded_complex": digest(
            API / "meep_gpu/triton_kernels/folded_complex.py"),
        "complex_ade": digest(API / "meep_gpu/triton_kernels/complex_ade.py"),
    }
    record["seconds"] = round(time.time() - started, 3)
    record["summary"] = {
        "passed": not failures,
        "failures": failures,
        "patterns": sorted(record.get("patterns", {})),
        "licenses": {name: value.get("expansion")
                     for name, value in licenses.items()},
    }
    save(out, record)
    print(json.dumps(record["summary"], indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
