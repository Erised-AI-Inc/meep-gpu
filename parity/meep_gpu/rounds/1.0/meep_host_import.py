"""Run a gate that takes no --import-meep-for-host-policy with MEEP imported first.

WHY. install_subnormal_policy('flush') refuses in a process that has not imported MEEP
(the host FTZ/DAZ bits are only reachable through mp.set_zero_subnormals). Two CUDA gates,
gate_cuda_complex_beta.py and gate_cuda_complex_folded.py, carry no flag for it; their
2026-09-01 runner and the 2026-09-02 re-gate cured the flush leg in-process, and its
driver.log shows exactly this pair of lines before the gate's own output:
    [policy] imported MEEP 1.33.0 so the host half of a flush policy is reachable (...)
    [meep-host-import] {'imported': True, 'meep_version': '1.33.0'}
This reproduces that -- the record probe_fused_kernel_bit_identity.import_meep_for_host_policy()
returns, built here -- then runs the gate's own __main__ with sys.argv set to
[gate, *args] so the payload's recorded `argv` keeps the gate's shape.

NOTHING BUT MEEP IS IMPORTED BEFORE THE GATE. The record is built here rather than by
importing probe_fused_kernel_bit_identity, because that module imports CuPy at load, and
three Triton ADE-chain gates (gate_triton_fused_ade_chain.py,
gate_triton_folded_offdiag_fused_ade_chain.py, gate_triton_complex_fused_ade_chain.py) set
CUPY_ACCELERATORS themselves and refuse a process in which CuPy is already imported.
MEASURED on the 1.0 tree (pre-certification pass FPC): with the probe imported here, all three
refused in the flush fleet ("cupy was imported before this gate could set
CUPY_ACCELERATORS"). The wrapper refuses to hand over if CuPy is in sys.modules.

It runs from the LANE directory (outside the staged tree), so gate_provenance -- which
records only modules under apps/api -- never records this wrapper.

usage: python -u meep_host_import.py <apps/api> <gate script path> <gate args...>
"""
import runpy
import sys

api, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path[:0] = [api, f"{api}/parity/meep_gpu"]

record = {"requested": True}
try:
    import meep  # noqa: E402 - the host half of a flush policy is reachable only through it
except Exception as exc:  # noqa: BLE001 - a refusal is the result
    record.update(imported=False, why=f"{type(exc).__name__}: {exc}"[:400])
else:
    record.update(imported=True, meep_version=getattr(meep, "__version__", "<unknown>"),
                  has_set_zero_subnormals=hasattr(meep, "set_zero_subnormals"))
    print(f"[policy] imported MEEP {record['meep_version']} so the host half of a flush "
          f"policy is reachable (set_zero_subnormals={record['has_set_zero_subnormals']})",
          flush=True)
print("[meep-host-import]", {k: record.get(k) for k in ("imported", "meep_version")}, flush=True)
if not record.get("imported"):
    raise SystemExit(f"MEEP did not import: {record.get('why')}")
if "cupy" in sys.modules:
    raise SystemExit("CuPy is imported before the gate runs; a gate that sets "
                     "CUPY_ACCELERATORS itself would refuse this process")
sys.argv = [script, *args]
runpy.run_path(script, run_name="__main__")
