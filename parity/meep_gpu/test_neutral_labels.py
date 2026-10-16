"""Every tool that copies a gate's machine stamp or environment into a ledger writes
the neutral label (``neutral_labels.py``), never the stamp.

One must-catch test per tool: each is handed a run that stamps a machine name and
home paths, and must write neither. The names and paths are made up here at run time
from neutral pieces; what is checked is that none of them reaches the value the tool
writes, and that the hygiene guard's ledger rule passes that value.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neutral_labels as N  # noqa: E402
import rebind_cuda_welds  # noqa: E402
import rebind_triton_welds  # noqa: E402
import record_cuda_ade_recut  # noqa: E402
import record_cuda_regate  # noqa: E402
import recut_composition_records  # noqa: E402
import recut_driver_dispatch_record  # noqa: E402
import seed_triton_welds  # noqa: E402
from meep_gpu import fastpath  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "check_hygiene", ROOT / "tools" / "ci" / "check_hygiene.py")
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard
_spec.loader.exec_module(guard)

#: A machine and an account the run "stamps"; made of letters so the guard's word rule
#: applies to them once they are listed (see _listed).
MACHINE = "zq" + "gpunode"
ACCOUNT = "zq" + "acct"
HOME = "/" + "ho" + "me"


def _listed():
    rules = guard.load_rules()
    rules.add_name("host_name", [MACHINE])
    rules.add_name("account_name", [ACCOUNT])
    return rules


def _assert_neutral(value):
    """``value`` (a string or a JSON-able structure) carries no stamp the run made."""
    text = json.dumps(value)
    assert MACHINE not in text and ACCOUNT not in text and HOME + "/" not in text, text
    hits = guard.scan_text(_listed(), text, {})
    assert not hits, hits


def _host_field_passes_the_ledger_rule(value: str) -> bool:
    return not guard._HOST_STAMP.match(value)


def _identity(**extra):
    identity = {"hostname": f"{MACHINE}.cluster.invalid", "device": "NVIDIA RTX A6000",
                "compute_capability": "8.6", "cupy": "13.5.1",
                "triton": fastpath.validated_triton_versions()[0],
                "cuda_visible_devices": "1"}
    identity.update(extra)
    return identity


# ------------------------------------------------------------------ the label module
def test_the_labels_are_the_spellings_the_shipped_records_carry():
    assert N.GPU_HOST == "the GPU host" and N.STAGING_ROOT == "<staging-root>"
    assert N.host_label(MACHINE) == N.GPU_HOST and N.host_label("") is None
    assert N.machine_stamped("flush/gate.json:host") == \
        "the GPU host (machine-stamped: flush/gate.json:host)"
    assert N.one_host([MACHINE, MACHINE]) == N.GPU_HOST
    with pytest.raises(N.LabelRefusal):
        N.one_host([MACHINE, MACHINE + "b"])
    with pytest.raises(N.LabelRefusal):
        N.one_host([])


def test_a_recorded_environment_is_written_under_its_staging_root():
    stage = f"{HOME}/{ACCOUNT}/stagedir"
    environment = {"CUPY_CACHE_DIR": f"{stage}/round/lanes/l2/cupy_cache",
                   "MEEP_GPU_DISPATCH_LOG": f"{stage}/round/repo/results/x/log.jsonl",
                   "TRITON_LIBCUDA_PATH": f"{stage}/libcuda_stub",
                   "CUDA_VISIBLE_DEVICES": "0"}
    derived, root, residue = N.neutral_environment(environment)
    assert root == stage and residue == []
    assert derived["CUPY_CACHE_DIR"] == "<staging-root>/round/lanes/l2/cupy_cache"
    assert derived["TRITON_LIBCUDA_PATH"] == "<staging-root>/libcuda_stub"
    assert derived["CUDA_VISIBLE_DEVICES"] == "0"
    _assert_neutral(derived)
    given, root, residue = N.neutral_environment(environment, staging_root=stage + "/round")
    assert given["CUPY_CACHE_DIR"] == "<staging-root>/lanes/l2/cupy_cache"
    assert residue == ["TRITON_LIBCUDA_PATH"]


def test_a_single_local_path_has_no_derivable_root_and_is_residue():
    _env, root, residue = N.neutral_environment({"X": f"{HOME}/{ACCOUNT}/cache"})
    assert root is None and residue == ["X"]


def test_a_stamped_name_in_free_text_is_replaced_by_the_label():
    text = f"ran on {MACHINE} GPU 1; {MACHINE}.cluster.invalid stamped it"
    out = N.neutral_text(text, [f"{MACHINE}.cluster.invalid"])
    assert out == "ran on the GPU host GPU 1; the GPU host stamped it"


# ------------------------------------------------------------------ one test per tool
def test_rebind_triton_welds_writes_the_label_in_the_host_line():
    line = rebind_triton_welds._host_line(_identity())
    assert line.startswith("the GPU host: NVIDIA RTX A6000, cc 8.6; CuPy 13.5.1; Triton ")
    assert line.endswith("(GPU 1)") and _host_field_passes_the_ledger_rule(line)
    _assert_neutral(line)
    # The machine is still REQUIRED: a run that stamps none composes no line.
    assert rebind_triton_welds._host_line(_identity(hostname=None)) is None


def test_rebind_triton_welds_writes_the_label_in_the_records_line():
    note = rebind_triton_welds.campaign_host_note({"host": MACHINE, "gpu": 1})
    assert note == "the GPU host GPU 1"
    assert rebind_triton_welds.campaign_host_note({}) == "?"


def test_seed_triton_welds_composes_its_host_line_with_the_rebind_tool():
    assert seed_triton_welds.rebind_triton_welds._host_line is rebind_triton_welds._host_line
    _assert_neutral(seed_triton_welds.rebind_triton_welds._host_line(_identity()))


def test_recut_composition_records_derives_the_labelled_host_line():
    host, why_not = recut_composition_records.derived_host(_identity())
    assert host and host.startswith("the GPU host: ") and why_not == ""
    _assert_neutral(host)


def test_rebind_cuda_welds_writes_the_label_for_a_stamped_or_declared_machine():
    stamped = rebind_cuda_welds.weld_host([{"host": MACHINE}], ["flush"], {}, "block")
    assert stamped == "the GPU host (machine-stamped: flush:host)"
    declared = rebind_cuda_welds.weld_host([], [], {"environment": {"hostname": MACHINE}},
                                           "block")
    assert declared == ("the GPU host (NOT machine-stamped by this gate; from "
                        "certification.json:block)")
    for value in (stamped, declared):
        assert _host_field_passes_the_ledger_rule(value)
        _assert_neutral(value)


def test_record_cuda_regate_writes_the_label_and_never_a_list_of_machines(tmp_path):
    stamps = {"started_utc": {"2026-01-01T00:00:00Z"}, "hostname": {MACHINE},
              "subnormal_policy": {"keep", "flush"}, "environment": set()}
    block = record_cuda_regate.build_block(tmp_path, [], stamps, "test")
    assert block["host"] == "the GPU host"
    _assert_neutral(block)
    with pytest.raises(N.LabelRefusal):
        record_cuda_regate.build_block(tmp_path, [], dict(stamps, hostname={MACHINE, "b"}),
                                       "test")


def test_record_cuda_ade_recut_writes_the_label(tmp_path):
    payloads = {}
    for leg in record_cuda_ade_recut.POLICY_LEGS:
        payload = {"started_utc": "2026-01-01T00:00:00Z", "host": MACHINE,
                   "gate": "gate_cuda_ade", "kernels": ["update_p"],
                   "device_source_sha256": {"update_p": "0" * 64}}
        (tmp_path / leg).mkdir()
        (tmp_path / leg / "gate.json").write_text(json.dumps(payload))
        payloads[leg] = payload
    block = record_cuda_ade_recut.build_block(tmp_path, payloads, {}, "block", {}, {}, [])
    assert block["host"] == "the GPU host"
    _assert_neutral(block)


def test_recut_driver_dispatch_record_writes_a_neutral_licence_environment():
    stage = f"{HOME}/{ACCOUNT}/stagedir"
    environment = {"CUPY_CACHE_DIR": f"{stage}/lanes/route_1/cupy_cache",
                   "TRITON_CACHE_DIR": f"{stage}/lanes/route_1/triton_cache",
                   "TRITON_LIBCUDA_PATH": f"{stage}/libcuda_stub"}
    written, residue = recut_driver_dispatch_record.licence_environment(environment)
    assert residue == [] and written["TRITON_LIBCUDA_PATH"] == "<staging-root>/libcuda_stub"
    _assert_neutral(written)
    # A recorded path outside the root is refused, not written.
    _written, residue = recut_driver_dispatch_record.licence_environment(
        environment, staging_root=stage + "/lanes")
    assert residue == ["TRITON_LIBCUDA_PATH"]
    assert recut_driver_dispatch_record.neutral_labels.host_label(MACHINE) == "the GPU host"


# ------------------------------------------- what an earlier bind left: the rewrite tool
import neutralize_ledger_labels  # noqa: E402


def _ledger_root(tmp_path, entry, indent=2):
    (tmp_path / "tools" / "ci").mkdir(parents=True)
    for name in ("check_hygiene.py", "hygiene_rules.json"):
        (tmp_path / "tools" / "ci" / name).write_bytes((ROOT / "tools" / "ci" / name).read_bytes())
    for name in neutralize_ledger_labels.LEDGERS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}\n")
    first = tmp_path / neutralize_ledger_labels.LEDGERS[0]
    first.write_text(json.dumps({"weld": entry}, indent=indent, sort_keys=True) + "\n")
    return first


def _weld():
    entry = {"source_sha256": {"meep_gpu/a.py": "0" * 64}, "runs": {}}
    entry["runs"]["8.6"] = {
        "bound_sha256": fastpath.bound_digest(entry),
        "host": f"{MACHINE}: NVIDIA RTX A6000, cc 8.6; CuPy 13.5.1; Triton 3.1.0 (GPU 1)",
        "records": f"apps/api/parity/meep_gpu/results/run/x/ - released; {MACHINE} GPU 1.",
        "licence": {"environment_at_start": {
            "CUPY_CACHE_DIR": f"{HOME}/{ACCOUNT}/stage/round/cupy_cache",
            "TRITON_LIBCUDA_PATH": f"{HOME}/{ACCOUNT}/stage/libcuda_stub"}}}
    return entry


def test_neutralize_ledger_labels_rewrites_the_label_leaves_and_nothing_else(tmp_path):
    ledger = _ledger_root(tmp_path, _weld())
    assert neutralize_ledger_labels.main(["--root", str(tmp_path), "--write"]) == 0
    run = json.loads(ledger.read_text())["weld"]["runs"]["8.6"]
    assert run["host"] == ("the GPU host: NVIDIA RTX A6000, cc 8.6; CuPy 13.5.1; "
                           "Triton 3.1.0 (GPU 1)")
    assert run["records"].endswith("released; the GPU host GPU 1.")
    assert run["licence"]["environment_at_start"] == {
        "CUPY_CACHE_DIR": "<staging-root>/round/cupy_cache",
        "TRITON_LIBCUDA_PATH": "<staging-root>/libcuda_stub"}
    assert fastpath.live_capabilities(json.loads(ledger.read_text())["weld"]) == ("8.6",)
    _assert_neutral(json.loads(ledger.read_text()))
    # Idempotent: a second run finds nothing to rewrite.
    before = ledger.read_bytes()
    assert neutralize_ledger_labels.main(["--root", str(tmp_path), "--write"]) == 0
    assert ledger.read_bytes() == before


def test_neutralize_ledger_labels_refuses_a_file_no_writer_wrote(tmp_path):
    ledger = _ledger_root(tmp_path, _weld(), indent=4)
    before = ledger.read_bytes()
    assert neutralize_ledger_labels.main(["--root", str(tmp_path), "--write"]) == 1
    assert ledger.read_bytes() == before
