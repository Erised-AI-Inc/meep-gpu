"""The timing ladder's recorders, on rows built here; no GPU, no MEEP."""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

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
              ranks=16):
    measurement = {"row": "measurement", "row_id": row_id, "case": "pml_3d",
                   "cells": 512000, "ranks": ranks, "mcell_steps_per_s": 500.0,
                   "ms_per_step": 1.0, "spread": 0.01, "windows": 6,
                   "steps_per_window": 100, "stepper": "fields_step", "init_seconds": 2.0,
                   "geometry_digest": digest, "monitor": {"stepped": stepped},
                   "threads_observed": threads,
                   "environment": {"meep": "1.33.0", "single_precision": True,
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
    # THE BASELINE: MPI's helper threads already make a single-threaded rank read 3.
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
