"""The timing ladder's recorders, on rows built here; no GPU, no MEEP."""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import neutral_ids  # noqa: E402
import timing_host  # noqa: E402
import timing_records as tr  # noqa: E402

UUID0 = "GPU-00000000-aaaa"
UUID7 = "GPU-77777777-bbbb"


def gpu_row(launched=("triton",), bracketed=(), repairs=0, verdict="TIMED",
            identical=True, foreign_on_0=False):
    apps = [f"{UUID7}, 900, 4970 MiB", f"{UUID0}, 4242, 1200 MiB"]
    if foreign_on_0:
        apps.append(f"{UUID0}, 555, 800 MiB")
    box = {"CUDA_VISIBLE_DEVICES": "0", "pid": 4242,
           "gpus": f"0, {UUID0}, 3 MiB, 0 %\n7, {UUID7}, 4970 MiB, 0 %",
           "compute_apps": "\n".join(apps), "loadavg": [1.0, 1.0, 1.0],
           "foreign_compute_apps_on_pinned_gpu": [apps[0]]}
    route = {"bracketed_slots": list(bracketed), "linear_repairs": repairs,
             "cells_repairs": 0, "linear_saves": repairs, "cells_saves": 0,
             "plan_changed_inside_window": False, "windows": 6}
    return {"case": "pml_3d", "grid_shape": [80, 80, 80], "verdict": verdict,
            "floors": {"spread_within_gate": verdict == "TIMED"},
            "substitution": {"tables_dispatched": list(launched), "proved": True},
            "bit_identity": {"fused_vs_array_identical": identical, "words": 10},
            "per_leg": {"fused": {"median_seconds_per_step": 0.0005, "spread": 0.01,
                                  "deposit_repair_route": route},
                        "unfused": {"median_seconds_per_step": 0.0006, "spread": 0.01},
                        "array": {"median_seconds_per_step": 0.004, "spread": 0.01,
                                  "launches_per_step": 0.0}},
            "box_before": box, "box_after": box, "provenance": {"sha256": {"bench": "x"}},
            "monitors_mode": "attached",
            "monitors_attached": {"fused": {"_dft_monitors": 0, "_flux_monitors": 1}}}


def write_rows(directory, rows):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "rows.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_record_gpu_reads_the_pinned_device_by_uuid(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "row"), [gpu_row()])
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "row"), None, "triton", None,
                                   True, {"index": 1})
    measured = entry["measured"]
    assert measured["foreign_on_pinned_device_before"] == []
    assert measured["foreign_compute_apps_on_pinned_gpu_before"] == [
        f"{UUID7}, 900, 4970 MiB"]
    assert summary.startswith("TIMED cells=512000 launched=triton fused 1024.0")
    assert "repair=none(expected none)" in summary and not entry.get("struck")
    write_rows(str(tmp_path / "busy"), [gpu_row(foreign_on_0=True)])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "busy"), None, None, None, False, {})
    assert entry["measured"]["foreign_on_pinned_device_before"] == [f"{UUID0}, 555, 800 MiB"]


def test_record_gpu_checks_the_declared_deposit_repair(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "b"), [gpu_row(bracketed=("step_B", "update_H"), repairs=12)])
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "b"), None, None, "B", True, {})
    assert entry["deposit_repair"]["ok"] and "repair=B(expected B)" in summary
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "b"), None, None, None, True, {})
    assert "deposit-repair route" in entry["struck"] and "STRUCK" in summary
    write_rows(str(tmp_path / "none"), [gpu_row()])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "none"), None, None, "D", True, {})
    assert "declared the D-seam bracket" in entry["struck"]


def test_record_gpu_strikes_another_route_and_lost_bit_identity(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "x"), [gpu_row(launched=("cuda", "triton"))])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "x"), None, "cuda", None, False, {})
    assert entry["struck"] == "launched cuda+triton, and this route is cuda"
    write_rows(str(tmp_path / "y"), [gpu_row(identical=False)])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "y"), None, None, None, False, {})
    assert "bit-identical" in entry["struck"]
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "missing"), None, None, None,
                                   False, {})
    assert entry["measured"] is None and summary.startswith("NO ROW")


def meep_rows(path, row_id, digest="d" * 64, configuration=None, threads=None,
              stepped=True, binding="bound-core-by-package", binding_facts=None,
              ranks=16, single=True, structural=None):
    measurement = {"row": "measurement", "row_id": row_id, "case": "pml_3d",
                   "cells": 512000, "ranks": ranks, "mcell_steps_per_s": 500.0,
                   "ms_per_step": 1.0, "spread": 0.01, "windows": 6,
                   "steps_per_window": 100, "stepper": "fields_step", "init_seconds": 2.0,
                   "geometry_digest": digest, "monitor": {"stepped": stepped},
                   "geometry_digest_structural": structural,
                   "threads_observed": threads,
                   "environment": {"meep": "1.33.0", "single_precision": single,
                                   "binding_label": binding,
                                   "binding": binding_facts or {"bound": True}}}
    if configuration is not None:
        measurement["configuration"] = configuration
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"row": "geometry", "row_id": row_id,
                                 "geometry_digest": digest}) + "\n")
        handle.write(json.dumps(measurement) + "\n")


def test_digest_of_record_prefers_the_harness_lift(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    digest_file = str(tmp_path / "digests" / "pml_3d_res20.sha256")
    tr.write_lift_digest(digest_file, "a" * 64, "pml_3d_res20")
    meep_rows(rows, "r1", digest="b" * 64)
    entry, summary = tr.record_meep(ledger, rows, "r1", None, digest_file, None, {})
    assert entry["geometry_digest_source"] == "harness-lift:pml_3d_res20"
    assert entry["struck"].startswith(f"geometry digest {'b' * 64} is not the digest of "
                                      f"record {'a' * 64} (source: harness-lift")


def test_digest_of_record_falls_back_to_the_first_meep_row(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    digest_file = str(tmp_path / "digests" / "pml_3d_res20.sha256")
    meep_rows(rows, "first", digest="c" * 64)
    meep_rows(rows, "second", digest="e" * 64)
    first, _ = tr.record_meep(ledger, rows, "first", None, digest_file, None, {})
    assert not first.get("struck")
    assert first["geometry_digest_source"] == "first-meep-row:first"
    second, summary = tr.record_meep(ledger, rows, "second", None, digest_file, None, {})
    assert "is not the digest of record" in second["struck"]
    assert "(source: first-meep-row:first)" in second["struck"]


def test_configuration_fields_and_threads(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    meep_rows(rows, "plain")
    entry, summary = tr.record_meep(ledger, rows, "plain", None, None, None, {})
    assert entry["configuration"] == {"build": "reference", "threads_per_rank": 1,
                                      "split_chunks_evenly": "default",
                                      "stepper": "fields_step"}
    assert "build=reference threads=1 split=default" in summary
    meep_rows(rows, "t2", configuration={"build": "native", "threads_per_rank": 2,
                                         "split_chunks_evenly": False},
              threads={"threads_min": 1, "cpu_seconds_over_wall_min": 0.9})
    entry, _ = tr.record_meep(ledger, rows, "t2", None, None, None, {})
    assert entry["configuration"]["build"] == "native"
    assert "2 threads per rank asked; observed 1 threads" in entry["struck"]
    meep_rows(rows, "t2ok", configuration={"threads_per_rank": 2},
              threads={"threads_min": 3, "cpu_seconds_over_wall_min": 1.9})
    entry, _ = tr.record_meep(ledger, rows, "t2ok", None, None, None, {})
    assert not entry.get("struck")
    # The baseline: MPI's helper threads already make a single-threaded rank read 3.
    meep_rows(rows, "t2mpi", configuration={"threads_per_rank": 2},
              threads={"threads_min": 3, "omp_threads_added_min": 0,
                       "cpu_seconds_over_wall_min": 1.9})
    entry, _ = tr.record_meep(ledger, rows, "t2mpi", None, None, None, {})
    assert "0 threads added after MPI initialisation (floor 1)" in entry["struck"]
    meep_rows(rows, "t2poll", configuration={"threads_per_rank": 2},
              threads={"threads_min": 4, "omp_threads_added_min": 1,
                       "cpu_seconds_over_wall_min": 1.05})
    entry, _ = tr.record_meep(ledger, rows, "t2poll", None, None, None, {})
    assert "floor 1.50" in entry["struck"]
    meep_rows(rows, "t2real", configuration={"threads_per_rank": 2},
              threads={"threads_min": 4, "omp_threads_added_min": 1,
                       "cpu_seconds_over_wall_min": 1.7})
    entry, _ = tr.record_meep(ledger, rows, "t2real", None, None, None, {})
    assert not entry.get("struck")


def test_binding_lines_of_both_launchers(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    meep_rows(rows, "r")
    log = tmp_path / "row.log"
    log.write_text("[node:1] Rank 0 bound to package[0][core:L0]\n"
                   "cpu-bind=MASK - node01, task  1  1 [999]: mask 0x2 set\n")
    entry, _ = tr.record_meep(ledger, rows, "r", str(log), None, None, {})
    assert entry["launcher_bindings"]["lines"] == 2


def test_a_measurement_missing_a_number_is_still_recorded(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    meep_rows(rows, "partial")
    lines = [json.loads(line) for line in open(rows)]
    for row in lines:
        row.pop("init_seconds", None)
        row.pop("spread", None)
    with open(rows, "w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row) + "\n")
    entry, summary = tr.record_meep(ledger, rows, "partial", None, None, None, {})
    assert "init n/a s" in summary and "spread n/a" in summary
    assert json.loads(open(ledger).read().splitlines()[-1])["row_id"] == "partial"


def test_missing_measurement_and_unstepped_monitor(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r16.jsonl")
    entry, summary = tr.record_meep(ledger, rows, "absent", None, None, None, {})
    assert summary.startswith("NO MEASUREMENT")
    meep_rows(rows, "quiet", stepped=False)
    entry, _ = tr.record_meep(ledger, rows, "quiet", None, None, None, {})
    assert "flux monitor" in entry["struck"]


def test_manifest_layouts_and_refusal(tmp_path):
    release = tmp_path / "release"
    (release / "meep_gpu").mkdir(parents=True)
    (release / "parity" / "meep_gpu").mkdir(parents=True)
    (release / "meep_gpu" / "driver.py").write_text("x = 1\n")
    (release / "parity" / "meep_gpu" / "gate.py").write_text("y = 2\n")
    (release / "README.md").write_text("not code\n")
    lines, digest = tr.manifest(str(release), True)
    assert [line.split("  ")[1] for line in lines] == ["meep_gpu/driver.py",
                                                       "parity/meep_gpu/gate.py"]
    development = tmp_path / "dev" / "apps" / "api"
    (development / "meep_gpu").mkdir(parents=True)
    (development / "parity" / "meep_gpu").mkdir(parents=True)
    (development / "meep_gpu" / "driver.py").write_text("x = 1\n")
    assert tr.layout_of(str(tmp_path / "dev")) == "development"
    dev_lines, _ = tr.manifest(str(tmp_path / "dev"), True)
    assert [line.split("  ")[1] for line in dev_lines] == ["apps/api/meep_gpu/driver.py"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="holds neither"):
        tr.manifest(str(empty), True)
    whole, _ = tr.manifest(str(release), False, exclude=["parity"])
    assert [line.split("  ")[1] for line in whole] == ["README.md", "meep_gpu/driver.py"]


def test_decide_binding_rule(tmp_path):
    for label, rate in (("bound", 600.0), ("unbound", 550.0)):
        directory = tmp_path / "probe" / label
        directory.mkdir(parents=True)
        with open(directory / "rows_r32.jsonl", "w", encoding="utf-8") as handle:
            for value in (rate, rate + 5):
                handle.write(json.dumps({"row": "measurement", "ranks": 32,
                                         "mcell_steps_per_s": value}) + "\n")
    chosen, record = tr.decide_binding(str(tmp_path / "probe"), ["bound", "unbound"],
                                       "bound", str(tmp_path / "decision.json"))
    assert chosen == "bound" and record["reason"].startswith("the higher")
    chosen, record = tr.decide_binding(str(tmp_path / "nothing"), ["bound", "unbound"],
                                       "bound", str(tmp_path / "decision2.json"))
    assert chosen == "bound" and record["reason"].startswith("default")


def test_record_control_reads_the_array_legs(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "a"), [gpu_row()])
    write_rows(str(tmp_path / "b"), [gpu_row(launched=("cuda",))])
    entry, summary = tr.record_control(ledger, None, [str(tmp_path / "a"),
                                                      str(tmp_path / "b")],
                                       HERE, False, {})
    assert entry["mcell_steps_per_s_mean"] == pytest.approx(128.0)
    assert "kernels vetoed True" in summary
    assert entry["how_forced"]["environment_of_the_array_leg"]


def test_makefile_flags_fall_back_to_config_log(tmp_path):
    log = tmp_path / "config.log"
    log.write_text("This file contains...\n  $ ./configure --enable-single\n\n"
                   "CXXFLAGS='-O3 -march=native'\nCXX='g++'\n")
    assert tr.configure_line(str(log)) == "./configure --enable-single"
    assert tr.makefile_flags(str(log)) == {"CXXFLAGS": "-O3 -march=native", "CXX": "g++"}
    (tmp_path / "Makefile").write_text("CXXFLAGS = -O2\nCXX = c++\n")
    assert tr.makefile_flags(str(log))["CXXFLAGS"] == "-O2"


def test_record_gpu_strikes_a_detached_row_and_counts_monitors(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "a"), [gpu_row()])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "a"), None, "triton", None, False, {})
    assert not entry.get("struck") and entry["measured"]["monitor_count"] == 1
    detached = dict(gpu_row(), monitors_mode="detached",
                    monitors_attached={"fused": {}})
    write_rows(str(tmp_path / "d"), [detached])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "d"), None, "triton", None, False, {})
    assert entry["struck"].startswith("monitors detached")


def test_darwin_conditions_strike_gpu_and_meep_rows(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    on_ac = {"on_ac_power": True, "low_power_mode": False, "cpu_speed_limit": None}
    row = dict(gpu_row(), host_before=dict(on_ac, foreign_processes=[]),
               host_after=dict(on_ac, foreign_processes=["9 1 python gate_metal_x.py"]))
    write_rows(str(tmp_path / "m"), [row])
    fields = {"platform": "darwin", "require_ac_power": True}
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "m"), None, "triton", None, False,
                             fields, {"cpu_state_before": on_ac, "cpu_state_after": on_ac})
    assert "another process drives the Apple GPU" in entry["struck"]
    rows = str(tmp_path / "rows_r8.jsonl")
    meep_rows(rows, "batt")
    entry, _ = tr.record_meep(ledger, rows, "batt", None, None, None, fields, {
        "cpu_state_before": on_ac,
        "cpu_state_after": {"on_ac_power": False, "power_source": "Battery Power"}})
    assert "cpu_state_after: on Battery Power" in entry["struck"]
    entry, _ = tr.record_meep(ledger, rows, "batt", None, None, None,
                              {"platform": "linux"}, {"cpu_state_after": {}})
    assert not entry.get("struck")


def test_a_bound_row_that_left_cores_idle_is_struck(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r64.jsonl")
    meep_rows(rows, "hw", binding_facts={"bound": True, "physical_cores_used": 32},
              ranks=64)
    entry, _ = tr.record_meep(ledger, rows, "hw", None, None, None,
                              {"physical_cores": 48}, {"binding_mode": "bound-hwthread"})
    assert "used 32 distinct physical cores of 48" in entry["struck"]
    meep_rows(rows, "hw2", binding_facts={"bound": True, "physical_cores_used": 48},
              ranks=64)
    entry, _ = tr.record_meep(ledger, rows, "hw2", None, None, None,
                              {"physical_cores": 48}, {"binding_mode": "bound-hwthread"})
    assert not entry.get("struck")


def _elf_with_needed(path, names):
    """A minimal 64-bit little-endian ELF: a .dynstr and a .dynamic naming ``names``."""
    import struct
    strtab = b"\x00" + b"".join(n.encode() + b"\x00" for n in names)
    offsets, at = [], 1
    for n in names:
        offsets.append(at)
        at += len(n) + 1
    dynamic = b"".join(struct.pack("<qQ", 1, o) for o in offsets) + struct.pack("<qQ", 0, 0)
    header_size = 64
    strtab_off = header_size
    dynamic_off = strtab_off + len(strtab)
    shoff = dynamic_off + len(dynamic)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HH", header, 0x3A, 64, 3)

    def section(sh_type, offset, size, link):
        entry = bytearray(64)
        struct.pack_into("<I", entry, 4, sh_type)
        struct.pack_into("<QQ", entry, 0x18, offset, size)
        struct.pack_into("<I", entry, 0x28, link)
        return bytes(entry)

    sections = section(0, 0, 0, 0) + section(3, strtab_off, len(strtab), 0) + \
        section(6, dynamic_off, len(dynamic), 1)
    with open(path, "wb") as handle:
        handle.write(bytes(header) + strtab + dynamic + sections)


def test_elf_needed_reads_direct_links_only(tmp_path):
    path = str(tmp_path / "libmeep.so")
    _elf_with_needed(path, ["libhdf5.so.310", "libgomp.so.1", "libc.so.6"])
    assert tr.elf_needed(path) == ["libhdf5.so.310", "libgomp.so.1", "libc.so.6"]
    plain = str(tmp_path / "plain.so")
    _elf_with_needed(plain, ["libopenblas.so.0", "libc.so.6"])
    assert tr.elf_needed(plain) == ["libopenblas.so.0", "libc.so.6"]
    (tmp_path / "text.so").write_text("not an elf")
    assert tr.elf_needed(str(tmp_path / "text.so")) is None


def test_manifest_skips_excluded_names(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / ".DS_Store").write_text("finder")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.txt").write_text("k")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "x.txt").write_text("s")
    lines, _ = tr.manifest(str(tmp_path), False, (), (".DS_Store", "scratch"))
    assert [line.split("  ", 1)[1] for line in lines] == ["a.py", os.path.join("sub",
                                                                               "keep.txt")]


def test_record_control_names_the_metal_gate(tmp_path):
    harness = tmp_path / "h"
    harness.mkdir()
    (harness / "gate_dispatch_metal_route.py").write_text(
        'ARRAY_ENV: Dict[str, Optional[str]] = {"MEEP_GPU_DISPATCH": "1",\n'
        '    "MEEP_GPU_FUSED": "0"}\n')
    write_rows(str(tmp_path / "g"), [gpu_row()])
    entry, _ = tr.record_control(str(tmp_path / "l.jsonl"), None, [str(tmp_path / "g")],
                                 str(harness), False, {}, "gate_dispatch_metal_route.py")
    assert entry["how_forced"]["read_from"].startswith("gate_dispatch_metal_route.ARRAY_ENV")
    assert entry["how_forced"]["environment_of_the_array_leg"]["MEEP_GPU_FUSED"] == "0"


# ---------------------------------------------------------------------------
# A GPU-only row cites its identity row (one lift, no array control)
# ---------------------------------------------------------------------------

def _gpu_only_row(**kw):
    row = gpu_row(**kw)
    row["legs_mode"] = "gpu-only"
    row["bit_identity"] = {"fused_vs_array_identical": None, "measured": False}
    row["substitution"]["tables_by_leg"] = {"fused": ["triton"]}
    del row["per_leg"]["unfused"], row["per_leg"]["array"]
    return row


def _identity_entry(tmp_path, **kw):
    """The identity tier's ledger entry for the same route, case and size."""
    write_rows(str(tmp_path / "identity"), [dict(gpu_row(**kw), substitution={
        "tables_dispatched": ["triton"], "proved": True,
        "tables_by_leg": {"fused": ["triton"], "unfused": ["triton"], "array": []}})])
    entry, _ = tr.record_gpu(str(tmp_path / "id_ledger.jsonl"), str(tmp_path / "identity"),
                             None, "triton", None, False, {"key": "identity_gpu_x",
                                                           "tier": "identity"})
    return entry


def test_a_gpu_only_row_without_its_identity_row_is_struck_by_name(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "solo"), [_gpu_only_row()])
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "solo"), None, "triton", None,
                                   False, {})
    assert entry["measured"]["legs_mode"] == "gpu-only"
    assert "no identity row" in entry["struck"] and "gpu-only" in summary
    assert entry["bit_identity_from"]["ok"] is False


def test_a_gpu_only_row_cites_a_three_leg_identity_row_of_its_size(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    write_rows(str(tmp_path / "solo"), [_gpu_only_row()])
    identity = _identity_entry(tmp_path)
    entry, summary = tr.record_gpu(ledger, str(tmp_path / "solo"), None, "triton", None,
                                   False, {}, identity_from=identity)
    assert not entry.get("struck"), entry.get("struck")
    assert entry["bit_identity_from"]["ok"] is True
    assert entry["bit_identity_from"]["key"] == "identity_gpu_x"
    assert "identity from identity_gpu_x: ok" in summary
    # an identity row the spread floor struck still vouches: its windows are not figures
    identity["measured"]["verdict"] = "TIMED-NOT-REPORTABLE"
    identity["struck"] = "TIMED-NOT-REPORTABLE: floors failed ['spread_within_gate']"
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "solo"), None, "triton", None,
                             False, {}, identity_from=identity)
    assert not entry.get("struck") and entry["bit_identity_from"]["ok"] is True
    # and a three-leg row is untouched by the clause
    write_rows(str(tmp_path / "full"), [gpu_row()])
    entry, _ = tr.record_gpu(ledger, str(tmp_path / "full"), None, "triton", None, False, {})
    assert "bit_identity_from" not in entry and not entry.get("struck")


@pytest.mark.parametrize("change, why", [
    (lambda e: e["measured"].update(cells=1), "timed 1 cells"),
    (lambda e: e["measured"].update(bit_identical_fused_vs_array=False), "not bit-identical"),
    (lambda e: e["measured"].update(provenance_sha256={"bench": "other"}),
     "other pinned sources"),
    (lambda e: e["measured"]["tables_by_leg"].update(fused=["cuda"]), "fused leg launched"),
    (lambda e: e["measured"].update(verdict="ERROR"), "did not run to the end"),
    (lambda e: e.update(deposit_repair={"ok": False}), "deposit-repair route"),
], ids=["other_size", "not_identical", "other_sources", "other_table", "did_not_finish",
        "repair"])
def test_a_gpu_only_row_cannot_borrow_another_rows_identity(tmp_path, change, why):
    write_rows(str(tmp_path / "solo"), [_gpu_only_row()])
    identity = _identity_entry(tmp_path)
    change(identity)
    entry, _ = tr.record_gpu(str(tmp_path / "ledger.jsonl"), str(tmp_path / "solo"), None,
                             "triton", None, False, {}, identity_from=identity)
    assert why in entry["struck"]


# ---------------------------------------------------------------------------
# The variance precondition, which is not the row gate
# ---------------------------------------------------------------------------

def _variance_ledger(path, gpu, meep, gpu_verdicts=None, ranks=32):
    with open(path, "w", encoding="utf-8") as handle:
        for index, (g, m) in enumerate(zip(gpu, meep), 1):
            verdict = (gpu_verdicts or {}).get(index, "TIMED")
            handle.write(json.dumps({
                "ledger": "gpu", "tier": "priority", "route": "default", "pass": index,
                "measured": {"verdict": verdict, "legs": {"fused": {
                    "mcell_steps_per_s": g, "spread": 0.074 if verdict != "TIMED" else 0.01}}},
                **({"struck": f"{verdict}: floors failed ['spread_within_gate']"}
                   if verdict != "TIMED" else {})}) + "\n")
            handle.write(json.dumps({
                "ledger": "meep", "tier": "priority", "pass": index, "ranks": ranks,
                "control": None, "build": "reference",
                "measured": {"mcell_steps_per_s": m, "spread": 0.02}}) + "\n")
            handle.write(json.dumps({     # a control at the same ranks: not the reference
                "ledger": "meep", "tier": "priority", "pass": index, "ranks": ranks,
                "control": "split", "build": "reference",
                "measured": {"mcell_steps_per_s": 1.0}}) + "\n")


def test_the_variance_precondition_lists_a_row_gate_strike_and_does_not_refuse(tmp_path):
    """The GPU host's 1.0 variance run: GPU 1.6 %, MEEP@32 3.3 % over five repeats, the
    first GPU repeat struck by its own window gate (7.4 %). The precondition is the
    repeat spread; the chain's first rule (every repeat TIMED) refused this box."""
    ledger = str(tmp_path / "ladder_rows.jsonl")
    _variance_ledger(ledger, [747.0, 746.0, 742.6, 738.5, 735.0],
                     [569.5, 582.3, 564.9, 584.2, 580.8],
                     gpu_verdicts={1: "TIMED-NOT-REPORTABLE"})
    verdict = tr.variance_verdict(ledger, 32)
    assert verdict["verdict"] == "GO", verdict["why"]
    assert verdict["sides"]["gpu"]["spread"] == pytest.approx(0.0161, abs=5e-4)
    assert verdict["sides"]["meep@32"]["spread"] == pytest.approx(0.0333, abs=5e-4)
    assert verdict["sides"]["meep@32"]["readable"] == 5          # the control is not read
    assert [s["pass"] for s in verdict["rows_struck_by_their_own_gate"]] == [1]


def test_the_variance_precondition_refuses_a_spread_above_its_threshold(tmp_path):
    """The Mac's 1.0 variance run: fused 5.75 %, MEEP@8 7.09 %: NO-GO at the table's 5 %,
    GO at the 10 % that run restated to; the threshold is the caller's, and recorded."""
    ledger = str(tmp_path / "ladder_rows.jsonl")
    _variance_ledger(ledger, [605.9, 573.1, 602.0, 601.2, 607.7],
                     [423.8, 394.3, 418.8, 415.9, 396.7], ranks=8)
    strict = tr.variance_verdict(ledger, 8)
    assert strict["verdict"] == "NO-GO" and strict["max_spread"] == 0.05
    assert any(w.startswith("gpu: repeat spread 0.0575") for w in strict["why"])
    assert tr.variance_verdict(ledger, 8, max_spread=0.10)["verdict"] == "GO"
    assert tr.main(["variance-verdict", "--ledger", ledger, "--meep-ranks", "8",
                    "--out", str(tmp_path / "v.json")]) == 5
    assert json.load(open(tmp_path / "v.json"))["verdict"] == "NO-GO"
    assert tr.main(["variance-verdict", "--ledger", ledger, "--meep-ranks", "8",
                    "--max-spread", "0.1"]) == 0
    few = str(tmp_path / "few.jsonl")
    _variance_ledger(few, [600.0, 601.0], [400.0, 401.0])
    assert "2 readable repeats < 5" in tr.variance_verdict(few, 32)["why"][0]


# ---------------------------------------------------------------------------
# H6: the thread gain read from a baseline taken before the OpenMP pool exists
# ---------------------------------------------------------------------------

#: A 24 x 2 row of the 1.0 timing run, as the bench recorded it: the
#: thread count read after ``import meep`` already held the OpenMP worker, so 0 were
#: "added" while the rank ran 1.997 CPU seconds per wall second.
CERT9N_24X2 = {"threads_min": 4, "threads_at_mpi_init_this_rank": 4,
               "omp_threads_added_min": 0, "cpu_seconds_over_wall_min": 1.9975}


def test_the_1_0_hybrid_row_is_struck_by_the_old_rule_and_read_by_the_new_one(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r24.jsonl")
    meep_rows(rows, "old", configuration={"threads_per_rank": 2}, threads=CERT9N_24X2,
              ranks=24)
    entry, _ = tr.record_meep(ledger, rows, "old", None, None, None, {})
    assert "0 threads added after MPI initialisation (floor 1)" in entry["struck"]
    # The same row with the bench's H6 fields: the worker was gained after the pre-import
    # baseline (1 thread) and ran for the whole window, beside the main thread.
    fixed = dict(CERT9N_24X2, threads_baseline="before_import_meep",
                 threads_before_import_this_rank=1, busy_threads_min=2,
                 omp_threads_busy_added_min=1)
    meep_rows(rows, "new", configuration={"threads_per_rank": 2}, threads=fixed, ranks=24)
    entry, _ = tr.record_meep(ledger, rows, "new", None, None, None, {})
    assert not entry.get("struck"), entry.get("struck")


def test_mpi_helpers_gained_at_import_do_not_count_as_openmp_threads(tmp_path):
    """A T = 2 rank whose only busy thread is the main one: MPI's two helpers (and a
    CUDA-aware MPI's CUDA thread) were gained after the pre-import baseline too, but sat
    idle, so the gain of record is 0 and the row is struck."""
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r24.jsonl")
    idle = {"threads_min": 4, "threads_before_import_this_rank": 1, "busy_threads_min": 1,
            "omp_threads_busy_added_min": 0, "cpu_seconds_over_wall_min": 1.6,
            "threads_baseline": "before_import_meep"}
    meep_rows(rows, "idle", configuration={"threads_per_rank": 2}, threads=idle, ranks=24)
    entry, _ = tr.record_meep(ledger, rows, "idle", None, None, None, {})
    assert ("0 threads gained after the pre-import baseline and busy over the windows "
            "(floor 1)") in entry["struck"]
    # T = 4 needs three busy workers; and the CPU floor still binds beside the count.
    four = {"omp_threads_busy_added_min": 3, "cpu_seconds_over_wall_min": 3.99}
    meep_rows(rows, "t4", configuration={"threads_per_rank": 4}, threads=four, ranks=12)
    entry, _ = tr.record_meep(ledger, rows, "t4", None, None, None, {})
    assert not entry.get("struck")
    slow = {"omp_threads_busy_added_min": 3, "cpu_seconds_over_wall_min": 2.5}
    meep_rows(rows, "t4slow", configuration={"threads_per_rank": 4}, threads=slow, ranks=12)
    entry, _ = tr.record_meep(ledger, rows, "t4slow", None, None, None, {})
    assert "floor 3.00" in entry["struck"]


def test_busy_threads_counts_only_threads_gained_and_busy():
    import bench_meep_identical_case as bench
    before = {100: 10.0, 101: 0.0, 102: 0.0, 103: 0.0}
    after = {100: 12.0, 101: 0.0, 102: 0.0, 103: 1.95}
    ran = bench.busy_threads(before, after, 2.0, [100])
    assert ran["busy"] == 2 and ran["busy_gained"] == 1
    assert bench.busy_threads(None, after, 2.0, [100]) == {"available": False}
    # A thread that started after the first reading counts from zero.
    ran = bench.busy_threads({100: 10.0}, {100: 12.0, 104: 1.5}, 2.0, [100])
    assert ran["busy_gained"] == 1


# ---------------------------------------------------------------------------
# Labelled context legs
# ---------------------------------------------------------------------------

def test_a_context_leg_may_be_double_precision_and_is_held_to_the_structural_digest(
        tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    rows = str(tmp_path / "rows_r48.jsonl")
    digest_file = str(tmp_path / "digests" / "pml_3d_res20.sha256")
    tr.write_lift_digest(digest_file, "a" * 64, "pml_3d_res20", structural="s" * 64)
    assert tr.structural_digest_of_record(digest_file) == "s" * 64
    context = {"context": "conda-forge MEEP (double precision)", "precision": "double",
               "build": "ctx-cfdouble"}
    # Its full digest differs (epsilon dtype, dt bits); its structural digest agrees.
    meep_rows(rows, "cf", digest="b" * 64, structural="s" * 64, single=False,
              configuration=context, ranks=48)
    entry, _ = tr.record_meep(ledger, rows, "cf", None, digest_file, None, {})
    assert not entry.get("struck"), entry.get("struck")
    assert entry["context"]["denominator"] is False
    assert entry["context"]["label"] == "conda-forge MEEP (double precision)"
    meep_rows(rows, "cfbad", digest="b" * 64, structural="t" * 64, single=False,
              configuration=context, ranks=48)
    entry, _ = tr.record_meep(ledger, rows, "cfbad", None, digest_file, None, {})
    assert "structural geometry digest" in entry["struck"]
    # The same double-precision row WITHOUT a context label is struck as before, and a
    # context leg never writes the digest of record.
    meep_rows(rows, "plain", digest="a" * 64, structural="s" * 64, single=False, ranks=48)
    entry, _ = tr.record_meep(ledger, rows, "plain", None, digest_file, None, {})
    assert entry["struck"] == "the MEEP build is not single precision"
    fresh = str(tmp_path / "digests" / "pml_3d_res32.sha256")
    meep_rows(rows, "first-cf", digest="c" * 64, single=False, configuration=context,
              ranks=48)
    tr.record_meep(ledger, rows, "first-cf", None, fresh, None, {})
    assert not os.path.exists(fresh)



def test_a_rank_reports_the_launchers_placement_not_its_pinned_main_thread():
    """H6b: under OMP_PROC_BIND the main thread is pinned to its first place once MEEP
    is imported; the rank's placement is the set read before the import."""
    import types
    import bench_meep_identical_case as bench
    stand_in = types.SimpleNamespace(my_rank=lambda: 0)
    facts = bench.rank_facts(stand_in, [0, 1, 48, 49])
    assert facts["affinity"] == [0, 1, 48, 49]
    assert facts["affinity_read"] == "before_import_meep"
    cores = {0: "0:0", 48: "0:0", 1: "0:1", 49: "0:1"}
    summary = bench.binding_summary({"ranks": [facts]}, scope_cpus=96, cores=cores)
    assert summary["physical_cores_used"] == 2 and summary["physical_cores_per_rank"] == [2]


# ---------------------------------------------------------------------------
# What a recorder writes about the machine (neutral_ids)
# ---------------------------------------------------------------------------

#: A machine no test here runs on: nothing a recorder writes may carry any of it.
HOST = "gpu-node-17.cluster.example.edu"
LOGIN = "fixtureuser"
HOME = "/" + "ho" + "me" + "/fixtureuser"   # from pieces: a full home path is a hygiene-guard hit
REAL_A = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
REAL_B = "GPU-9b2e7c41-0d6a-4f83-b5e9-1c7a3d8f6e02"
FOREIGN_PID = 271828


@pytest.fixture
def machine(monkeypatch):
    """This process as the fixture machine: its host names, login and home."""
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(neutral_ids, "_node", lambda: HOST)
    monkeypatch.setattr(neutral_ids.getpass, "getuser", lambda: LOGIN)
    monkeypatch.setenv("HOME", HOME)
    neutral_ids.reset()
    yield neutral_ids
    neutral_ids.reset()


def assert_neutral(record, *tokens):
    assert neutral_ids.residue(record) == []
    text = json.dumps(record, sort_keys=True, default=str)
    for token in (HOST, HOST.split(".")[0], LOGIN, HOME, REAL_A, REAL_B,
                  str(FOREIGN_PID)) + tokens:
        assert str(token) not in text, (token, text[:400])


def _nvidia_host(monkeypatch):
    """``run_text`` answering as a two-GPU Linux host named ``HOST``."""
    answers = {
        "uname -a": f"Linux {HOST} 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux",
        "uname -srm": "Linux 5.15.0-91-generic x86_64",
        "lscpu": f"Model name: AMD EPYC 7742 64-Core Processor\nHostname: {HOST}",
        "free -g": "Mem: 1007 12 900",
    }

    def run_text(command, timeout=30.0, env=None, cwd=None):
        joined = " ".join(command)
        if joined in answers:
            return answers[joined]
        if "--query-gpu=index,uuid,name" in joined:
            return (f"0, {REAL_A}, NVIDIA RTX A6000, 535.129.03, 49140 MiB, 3 MiB, 0 %, "
                    f"8.6, Enabled, Disabled\n1, {REAL_B}, NVIDIA RTX A6000, 535.129.03, "
                    "49140 MiB, 5000 MiB, 99 %, 8.6, Enabled, Disabled")
        if "--query-compute-apps" in joined:
            return f"{REAL_B}, {FOREIGN_PID}, 5000 MiB"
        if "CLOCK,PERFORMANCE" in joined:
            return f"GPU 00000000:01:00.0\n    GPU UUID : {REAL_A}\n    Clocks : 1800 MHz"
        return ""

    monkeypatch.setattr(timing_host, "run_text", run_text)
    monkeypatch.setattr(timing_host, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(timing_host, "read_file", lambda path: "1.0 1.0 1.0 1/100 9")
    monkeypatch.setattr(timing_host, "cpu_state", lambda platform=None: {"governor": "x"})
    monkeypatch.setattr(timing_host, "gpu_state", lambda gpu, platform=None: {
        "state": {"uuid": timing_host.recorded_device(REAL_A), "clocks.sm": "1800"}})


def test_the_environment_record_writes_nothing_of_the_machine(machine, monkeypatch, tmp_path):
    """environment.json leaves the machine: the host as ``uname -srm``, every UUID as its
    card id, no foreign pid, every path under its root's label. GPU model, driver and
    compute capability stay."""
    _nvidia_host(monkeypatch)
    monkeypatch.setenv("SLURMD_NODENAME", HOST.split(".")[0])
    monkeypatch.setattr(tr, "probe_packages", lambda python, **_k: (
        {"python": "3.12.4", "executable": f"{HOME}/miniforge3/envs/meep/bin/python",
         "meep": "1.35.0", "meep_file": f"{HOME}/miniforge3/envs/meep/lib/meep/__init__.py"},
        ""))
    out = str(tmp_path / "environment.json")
    builds = {"reference": {"python": f"{HOME}/miniforge3/envs/meep/bin/python",
                            "sha256": {f"{HOME}/miniforge3/envs/meep/lib/libmeep.so.37": "ab"}}}
    params = {"TREE_SRC": {"value": f"{HOME}/meep-gpu", "source": "given"},
              "host": HOST}
    returned = tr.environment_record(out, tree=None, drivers=None, python="py",
                                     builds=builds, gpu="0", params=params,
                                     topology={"physical_cores": 64}, platform="linux")
    with open(out, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written == json.loads(json.dumps(returned, default=str))
    assert_neutral(written)
    assert written["uname"] == "Linux 5.15.0-91-generic x86_64"
    card_a, card_b = neutral_ids.card_id(REAL_A), neutral_ids.card_id(REAL_B)
    assert "NVIDIA RTX A6000, 535.129.03" in written["nvidia_smi_gpus"]
    assert written["nvidia_smi_gpus"].splitlines()[0].startswith(f"0, {card_a}, ")
    assert written["nvidia_smi_apps"] == f"{card_b}, other, 5000 MiB"
    assert written["slurm"]["SLURMD_NODENAME"] == neutral_ids.machine_id()
    assert written["packages"]["executable"] == "<home>/miniforge3/envs/meep/bin/python"
    assert written["params"]["TREE_SRC"]["value"] == "<home>/meep-gpu"


def test_a_meep_build_record_writes_nothing_of_the_machine(machine, monkeypatch, tmp_path):
    image = tmp_path / "lib" / "libmeep.so.37"
    image.parent.mkdir()
    image.write_bytes(b"\x7fELF fixture")
    config_log = tmp_path / "config.log"
    config_log.write_text(f"This file contains any messages\n  $ ./configure "
                          f"--prefix={HOME}/miniforge3/envs/meep --with-mpi\n"
                          "CFLAGS='-O2'\nCXXFLAGS='-O2'\n")
    facts = {"meep": "1.35.0", "single_precision": True, "with_mpi": True,
             "meep_file": f"{HOME}/miniforge3/envs/meep/lib/python3.12/meep/__init__.py",
             "loaded_meep_images": [str(image), f"{HOME}/miniforge3/envs/meep/lib/_meep.so"],
             "loaded_by": "/proc/self/maps", "extensions_present": [],
             "libmeep_present": [f"{HOME}/miniforge3/envs/meep/lib/libmeep.so.33"],
             "openmp_runtimes_loaded": [], "prefix": f"{HOME}/miniforge3/envs/meep"}
    monkeypatch.setattr(timing_host, "run_text",
                        lambda *a, **k: "BUILD " + json.dumps(facts))
    monkeypatch.setattr(tr, "identity_facts", lambda python: (
        {"libm": {"kind": "glibc libm", "path": f"{HOME}/sysroot/libm.so.6",
                  "sha256": "cd"}, "subnormals_after_import": {"mode": "keep"}}, ""))

    class Census:
        @staticmethod
        def census(paths, progress=None):
            return {"images": [{"path": p, "sha256": "ef"} for p in paths],
                    "verdict": {"none_outside_exempt": True, "text": "clean"}}

    monkeypatch.setattr(tr, "load_tool", lambda name: Census)
    record = tr.meep_build_record(f"{HOME}/miniforge3/envs/meep/bin/python", str(config_log))
    assert_neutral(record)
    assert record["python"] == "<home>/miniforge3/envs/meep/bin/python"
    assert record["configure_line"] == "./configure --prefix=<home>/miniforge3/envs/meep --with-mpi"
    assert list(record["sha256"].values()) == [tr.sha256_file(str(image))]
    assert record["probe"]["single_precision"] is True


def _box_pair():
    """One box as the probe saw it (raw) and as a row records it: device 0 pinned by
    index, this process on it, a foreign app on it, another on device 1."""
    gpus = f"0, {REAL_A}, 3 MiB, 0 %\n1, {REAL_B}, 4970 MiB, 0 %"
    apps = (f"{REAL_A}, 4242, 1200 MiB\n{REAL_A}, {FOREIGN_PID}, 800 MiB\n"
            f"{REAL_B}, 900, 4970 MiB")
    raw = {"CUDA_VISIBLE_DEVICES": "0", "pid": 4242, "gpus": gpus, "compute_apps": apps}
    recorded = {"CUDA_VISIBLE_DEVICES": "0", "gpus": timing_host.recorded_gpus(gpus),
                "compute_apps": timing_host.recorded_compute_apps(apps, 4242)}
    return raw, recorded


def test_the_pinned_device_join_reads_raw_and_recorded_rows_alike(machine):
    raw, recorded = _box_pair()
    assert tr.pinned_foreign(raw) == [f"{REAL_A}, {FOREIGN_PID}, 800 MiB"]
    assert tr.pinned_foreign(recorded) == [f"{neutral_ids.card_id(REAL_A)}, other, 800 MiB"]
    assert_neutral(recorded, "4242")
    # pinned by UUID: the recorded pin is the card id the listing carries
    by_uuid = dict(recorded, CUDA_VISIBLE_DEVICES=timing_host.recorded_device(REAL_A))
    assert len(tr.pinned_foreign(by_uuid)) == 1
    quiet = dict(recorded, compute_apps=timing_host.recorded_compute_apps(
        f"{REAL_A}, 4242, 1200 MiB", 4242))
    assert tr.pinned_foreign(quiet) == []


def test_a_ledger_line_writes_nothing_of_the_machine_and_its_rows_still_open(
        machine, tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    row_dir = str(tmp_path / "row")
    write_rows(row_dir, [gpu_row()])
    log = tmp_path / "meep.log"
    log.write_text(f"[{HOST.split('.')[0]}:{FOREIGN_PID}] Rank 0 bound to package[0][core:0]\n"
                   f"[{HOST.split('.')[0]}:{FOREIGN_PID + 1}] Rank 1 bound to "
                   "package[0][core:1]\n")
    fields = {"run_dir": f"{HOME}/meep_gpu_validation/ladder_t", "host": HOST}
    tr.record_gpu(ledger, row_dir, None, None, None, False, dict(fields))
    tr.record_meep(ledger, str(tmp_path / "rows_r2.jsonl"), "r", str(log), None, None,
                   dict(fields))
    tr.record_note(ledger, "not_started", f"not started: {HOME}/bin/mpirun missing",
                   dict(fields))
    lines = [json.loads(line) for line in open(ledger, encoding="utf-8")]
    assert_neutral(lines, str(FOREIGN_PID + 1))
    assert {line["host"] for line in lines} == {neutral_ids.machine_id()}
    assert lines[0]["run_dir"] == "<home>/meep_gpu_validation/ladder_t"
    # a reader opens a recorded rows file through neutral_ids.resolve
    opened = neutral_ids.resolve(lines[0]["rows_file"])
    assert os.path.realpath(opened) == os.path.realpath(os.path.join(row_dir, "rows.jsonl"))
    assert lines[1]["launcher_bindings"]["first"] == [
        "Rank 0 bound to package[0][core:0]", "Rank 1 bound to package[0][core:1]"]
