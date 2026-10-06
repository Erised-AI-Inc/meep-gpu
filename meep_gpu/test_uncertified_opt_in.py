"""Kernels on a device or toolchain that is not certified.

``MEEP_GPU_ALLOW_UNCERTIFIED`` is a strict switch; every value other than ``1`` and
``0`` is refused by name. Every kernel table runs by default on an identity it
SUPPORTS but that no cited gate ran on, records ``certified: False`` with the identity
read, and prints one line naming what is certified and why the kernels ran; ``0``
restricts the kernels to certified identities and refuses one it cannot judge.

On the NVIDIA tables supported is a range: compute capability 7.0 to 9.0 on both
tables, and Triton ``>=3.1,<3.2`` on the Triton table. An identity read outside it is
refused by name unless the switch is ``1``, which runs it too, recorded
``supported: False``. A table that is certified for this device and toolchain
outranks one that is only supported: under the default the supported one is dropped
by name (rung 4e), and ``1`` composes both. On the Metal table every Apple GPU is
supported.

The NVIDIA half substitutes the identity: a stand-in for CuPy that reports a
compute capability (a supported one neither NVIDIA table certifies, such as 8.9 or
7.5 while 8.6 is the only certified one, or an unsupported one, such as 6.1 or 10.0),
a stand-in Triton of another version, and, for rung 4e, ledgers that certify a
capability on one table only, so it runs on a host with no NVIDIA device. The Apple
half substitutes both the environment the Metal ladder reads and the one the cited
welds record, and steps a real driver; it is skipped by a declared resource on a host
with no Apple GPU.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath, fastpath_cuda
from meep_gpu.driver import FdtdDriver
from meep_gpu.test_capability_records import ledger_also_certified_on
from meep_gpu.test_dispatch_contract import (
    CountingPlan,
    block_cuda,
    block_triton,
    certified_envelope_fields,
    certified_envelope_grid,
    certified_envelope_pml,
    cuda_step_plan,
    cupy_like_grid,
    cupy_like_grid_with_device,
    install_composer,
    install_cuda_composer,
    step_plan,
    stub_triton,
    supported_uncertified_capabilities,
    unsupported_capabilities,
)

SWITCH = "MEEP_GPU_ALLOW_UNCERTIFIED"
SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
            "MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_FUSE_ARMS",
            "MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_DISPATCH_LOG",
            "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL", SWITCH)

#: Supported compute capabilities no cited gate ran on: the first two candidates both
#: NVIDIA tables support and neither certifies (8.9 and 7.5 while 8.6 is the only
#: certified one), so certifying another architecture (9.0 is the next) leaves them
#: supported and uncertified.
SUPPORTED_UNCERTIFIED = [(capability, f"{capability[0]}.{capability[1]}")
                         for capability in supported_uncertified_capabilities(2)]
#: Compute capabilities neither NVIDIA table supports: below the floor and above the
#: ceiling.
UNSUPPORTED = [(capability, f"{capability[0]}.{capability[1]}")
               for capability in unsupported_capabilities(2)]
#: Values that are neither ``1`` nor ``0``, the two a user is most likely to type
#: among them.
UNACCEPTED_VALUES = ["true", "yes", "", " 1", "2", "on"]
STEPS = 5
#: How the NOTE line ends, whatever made the run uncertified.
NOTE_TAIL = ("compare the results with a prefer_gpu=False run of the same simulation "
             "before relying on them")


#: The supported range as the NOTE and the per-entry reasons spell it.
CAPABILITY_SCOPE = "compute capability 7.0 to 9.0"
TRITON_SCOPE = "Triton >=3.1,<3.2"


def _supported_entry_because(what, read, table, scope) -> str:
    """An admitted entry's own reason: THAT identity, THAT table's range."""
    return (f"{what} {read} is supported by the {table} table ({scope}) but not "
            "certified bit-identical")


def _unsupported_entry_because(what, read, table, scope) -> str:
    return f"{SWITCH}=1 runs {what} {read}, outside what the {table} table supports ({scope})"


def _supported_clause(subject, where="") -> str:
    """The NOTE's reason for the supported identities, ``subject`` naming them."""
    return fastpath.NVIDIA_SUPPORTED_BECAUSE.format(
        subject=subject, where=where, supported=fastpath.nvidia_supported_text(),
        switch=SWITCH)


def _unsupported_clause(subject, with_range) -> str:
    text = fastpath.nvidia_supported_text()
    return fastpath.NVIDIA_UNSUPPORTED_BECAUSE.format(
        switch=SWITCH, subject=subject, range=f" ({text})" if with_range else "")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """Every test starts with the switches unset, dispatch enabled and nothing said."""
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    flushing_at_entry = backends.subnormals_flushed()
    yield
    try:
        from meep_gpu import host_writes, metal_dispatch  # noqa: PLC0415

        for residency in host_writes._live():  # noqa: SLF001
            metal_dispatch.release_residency_hold(residency, None)
    except Exception:  # noqa: BLE001
        pass
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    if backends.subnormals_flushed() != flushing_at_entry:
        subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


def _meep_lines(err: str) -> list:
    return [line for line in err.splitlines() if line.startswith("meep_gpu:")]


def _notes(err: str) -> list:
    return [line for line in _meep_lines(err) if "NOT CERTIFIED" in line]


def _status(err: str) -> list:
    return [line for line in _meep_lines(err)
            if line.startswith("meep_gpu: step path fused;")]


def _plan_on(monkeypatch, grid, version="3.1.0"):
    """One freeze on a stand-in NVIDIA host whose composer fills one slot."""
    stub_triton(monkeypatch, version)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    return fastpath.plan_fast_path(object(), object(), grid)


def _set(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)


# --- the switch itself ------------------------------------------------------------------


def test_the_switch_is_named_and_strict():
    assert fastpath.UNCERTIFIED_SWITCH == SWITCH
    assert dict(fastpath.UNCERTIFIED_VALUES) == {"1": True, "0": False}


@pytest.mark.parametrize("value,allowed,unsupported,restricted", [
    (None, True, False, False), ("0", False, False, True), ("1", True, True, False)]
    + [(value, False, False, False) for value in UNACCEPTED_VALUES])
def test_the_record_states_what_each_value_allows(monkeypatch, value, allowed,
                                                  unsupported, restricted):
    """``allowed``: a supported identity that is not certified may compose (unset and
    ``1``); ``unsupported_allowed``: one outside the supported range may (``1``)."""
    _set(monkeypatch, value)
    assert fastpath.certified_only() is restricted
    assert fastpath.unsupported_allowed() is unsupported
    block = fastpath._base_record()["uncertified"]  # noqa: SLF001
    assert block == {"variable": SWITCH, "value": value, "allowed": allowed,
                     "unsupported_allowed": unsupported, "admitted": [], "dropped": []}


@pytest.mark.parametrize("value", [None, "0", "1"] + UNACCEPTED_VALUES)
def test_both_ladders_read_the_restriction_the_same_way(monkeypatch, value):
    """The NVIDIA ladder's ``certified_only`` and the Metal ladder's are one rule."""
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    _set(monkeypatch, value)
    assert fastpath.certified_only() is metal_dispatch.certified_only()


# --- NVIDIA: what is supported ----------------------------------------------------------


@pytest.mark.parametrize("capability,supported", [
    ((7, 0), True), ((7, 5), True), ((8, 6), True), ((8, 9), True), ((9, 0), True),
    ((6, 1), False), ((10, 0), False), ((12, 0), False), ("86", True), ("90", True),
    ("120", False), ("sm_86", False), (None, None)])
@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_both_tables_support_compute_capability_7_0_to_9_0(capability, supported, table):
    assert fastpath.capability_supported(capability, table) is supported
    assert fastpath.supported_compute_capabilities(table) == ((7, 0), (9, 0))


@pytest.mark.parametrize("version,supported", [
    ("3.1.0", True), ("3.1.1", True), ("3.1.0+git1234", True), ("3.2.0", False),
    ("3.0.0", False), ("9.9.9-unreleased", False), ("unknown", False)])
def test_the_triton_table_supports_triton_3_1(version, supported):
    """A version with no leading ``major.minor`` was read and cannot be placed in the
    range, so it is unsupported, not unknown."""
    assert fastpath.triton_version_supported(version) is supported


def test_the_supported_range_names_both_cupy_builds_nvrtc():
    text = fastpath.nvidia_supported_text()
    assert "compute capability 7.0 to 9.0" in text, text
    assert "Triton >=3.1,<3.2" in text, text
    for build, nvrtc in fastpath_cuda.SUPPORTED_NVRTC_BUILDS.items():
        assert f"{nvrtc} for {build}" in text, text
    assert set(fastpath_cuda.SUPPORTED_NVRTC_BUILDS) == {"cupy-cuda11x", "cupy-cuda12x"}


# --- NVIDIA: a supported compute capability no gate ran on ------------------------------


@pytest.mark.parametrize("capability,spelled", SUPPORTED_UNCERTIFIED)
@pytest.mark.parametrize("value", [None, "1"])
def test_a_supported_uncertified_device_dispatches_by_default_and_says_so_once(
        monkeypatch, capsys, capability, spelled, value):
    _set(monkeypatch, value)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    plan = _plan_on(monkeypatch, grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["decision"] == "dispatched" and record["step_path"] == "fused"
    assert record["certified"] is False
    block = record["uncertified"]
    assert block["variable"] == SWITCH and block["value"] == value and block["allowed"]
    assert block["dropped"] == []
    served = block["served"]
    assert served, block
    assert {entry["table"] for entry in served} <= {"triton", "cuda"}
    # What each table certifies, derived from its ledger rather than typed, so a round
    # that certifies another architecture does not have to come and edit this line.
    certified = {"triton": fastpath.validated_compute_capabilities(),
                 "cuda": fastpath_cuda.validated_compute_capabilities()}
    for entry in served:
        assert entry["what"] == "GPU compute capability"
        assert entry["read"] == spelled
        assert entry["certified"] == list(certified[entry["table"]]), entry
        assert entry["supported"] is True, entry
        assert entry["because"] == _supported_entry_because(
            "GPU compute capability", spelled, entry["table"], CAPABILITY_SCOPE), entry
        # The gates that would certify it are named, as a refusal names them.
        assert entry["welds_without_a_live_run_here"], entry
        assert entry["welds_without_a_live_run_here_count"] >= len(
            entry["welds_without_a_live_run_here"]), entry
    # The identity is recorded as READ: running it changes what runs, not what is true.
    environment = record["environment"]
    assert environment["device"]["compute_capability"] == spelled
    assert environment["device"]["name"] == "another device"
    assert environment["device_certified"] is False
    assert environment["device_supported"] is True
    assert environment["device_supported_by_table"] == {"triton": True, "cuda": True}
    assert environment["triton_certified"] is True
    assert environment["certified_only"] is False

    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 2, lines
    assert lines[0].startswith("meep_gpu: step path fused;"), lines
    assert lines[0].endswith("another device supported, UNCERTIFIED"), lines
    note = lines[1]
    assert note.startswith("meep_gpu: NOTE the kernels are NOT CERTIFIED on this "
                           "host: "), note
    for entry in served:
        listed = ", ".join(entry["certified"])
        assert f"GPU compute capability {spelled} (certified: {listed})" in note, note
    # THE OWNER'S WORDING, where it is exact: the GPU is supported and not certified,
    # and the Triton version beside it (certified) is supported too.
    assert (". They were dispatched because this NVIDIA GPU and toolchain are "
            "supported but not certified bit-identical") in note, note
    assert f". They were dispatched because {_supported_clause('NVIDIA GPU and toolchain are')}; " \
        in note, note
    assert f"{SWITCH}=0 would restrict the kernels to certified ones" in note, note
    assert note.count("the supported range is") == 1, note
    assert ("counted in the dispatch record under uncertified.served[], which names "
            "the first five in welds_without_a_live_run_here") in note, note
    assert note.endswith(NOTE_TAIL), note
    assert f"because {SWITCH}=1" not in note, note

    # A second freeze in the same process repeats neither line.
    assert _plan_on(monkeypatch, grid) is not None
    assert _meep_lines(capsys.readouterr().err) == []


@pytest.mark.parametrize("capability,spelled", SUPPORTED_UNCERTIFIED)
def test_certified_only_refuses_a_supported_uncertified_device_by_name(
        monkeypatch, capsys, capability, spelled):
    monkeypatch.setenv(SWITCH, "0")
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    assert _plan_on(monkeypatch, grid) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    assert record["environment"]["certified_only"] is True
    reason = record["refused_because"]
    assert f"GPU compute capability {spelled} is not in the recorded" in reason, reason
    for table in ("triton", "cuda"):
        row = record["tables"][table]
        assert row["candidate"] is False
        assert row["refused_because"].endswith(fastpath.CERTIFIED_ONLY_TAIL), row
        assert row["welds_without_a_live_run_here"], row
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"{SWITCH}=0 restricts" in lines[0], lines
    assert _notes("\n".join(lines)) == []


# --- NVIDIA: a compute capability outside the supported range ---------------------------


@pytest.mark.parametrize("capability,spelled", UNSUPPORTED)
@pytest.mark.parametrize("value", [None, "0"])
def test_an_unsupported_device_is_refused_by_name(monkeypatch, capsys, capability,
                                                 spelled, value):
    """Unset names the way past it (``1``); ``0`` names the restriction instead."""
    _set(monkeypatch, value)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    assert _plan_on(monkeypatch, grid) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    reason = record["refused_because"]
    assert f"GPU compute capability {spelled} is not in the recorded" in reason, reason
    tail = fastpath.UNSUPPORTED_HINT if value is None else fastpath.CERTIFIED_ONLY_TAIL
    for table in ("triton", "cuda"):
        row = record["tables"][table]
        assert row["candidate"] is False
        assert row["refused_because"].endswith(tail), row
        if value is None:
            assert ("outside the compute capabilities that table supports "
                    "(7.0 to 9.0)") in row["refused_because"], row
    assert record["environment"]["device_supported"] is False
    assert record["environment"]["device_certified"] is False
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert (f"set {SWITCH}=1" in lines[0]) is (value is None), lines
    assert _notes("\n".join(lines)) == []


@pytest.mark.parametrize("capability,spelled", UNSUPPORTED)
def test_an_unsupported_device_runs_under_one_recorded_as_unsupported(
        monkeypatch, capsys, capability, spelled):
    monkeypatch.setenv(SWITCH, "1")
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    plan = _plan_on(monkeypatch, grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["uncertified"]["unsupported_allowed"] is True
    served = record["uncertified"]["served"]
    assert served, record["uncertified"]
    for entry in served:
        assert entry["read"] == spelled
        assert entry["supported"] is False, entry
        assert entry["because"] == _unsupported_entry_because(
            "GPU compute capability", spelled, entry["table"], CAPABILITY_SCOPE), entry
    lines = _meep_lines(capsys.readouterr().err)
    status, notes = _status("\n".join(lines)), _notes("\n".join(lines))
    assert len(status) == 1 and status[0].endswith("another device UNCERTIFIED"), lines
    assert "supported," not in status[0], status
    assert len(notes) == 1, lines
    # Only the GPU is outside the range: the certified Triton is not named as such,
    # and nothing is called supported.
    assert f"because {_unsupported_clause('NVIDIA GPU', with_range=True)}; " in notes[0], \
        notes
    assert "supported but not certified" not in notes[0], notes
    assert notes[0].endswith(NOTE_TAIL), notes


# --- NVIDIA: the certified device, and one that cannot be read --------------------------


def test_the_certified_device_is_certified_and_said_the_same_way_whatever_the_switch(
        monkeypatch, capsys):
    """Nothing on a certified host depends on the switch: decision, slots, line."""
    seen = {}
    for value in (None, "0", "1"):
        _set(monkeypatch, value)
        fastpath.reset_dispatch_announcements()
        plan = _plan_on(monkeypatch, cupy_like_grid_with_device())
        assert plan is not None, value
        record = plan.report()
        assert record["certified"] is True, value
        assert record["uncertified"]["admitted"] == [], value
        assert record["uncertified"]["served"] == [], value
        assert record["uncertified"]["dropped"] == [], value
        lines = _meep_lines(capsys.readouterr().err)
        assert _notes("\n".join(lines)) == [], (value, lines)
        seen[value] = (sorted(plan.slots), lines)
    assert seen[None] == seen["0"] == seen["1"], seen
    assert seen[None][1][0].endswith("NVIDIA RTX A6000 certified"), seen


@pytest.mark.parametrize("value", [None, "1"])
def test_an_unreadable_device_is_neither_certified_nor_admitted(monkeypatch, capsys,
                                                                value):
    """Three-valued: an identity that was not read is not judged, unset or ``1``."""
    _set(monkeypatch, value)
    plan = _plan_on(monkeypatch, cupy_like_grid())
    assert plan is not None
    record = plan.report()
    assert record["environment"]["device_certified"] is None
    assert record["environment"]["device_supported"] is None
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    assert _notes(capsys.readouterr().err) == []


def test_certified_only_refuses_a_device_it_cannot_judge(monkeypatch, capsys):
    """``0`` runs the kernels only where they are certified, and an identity that was
    not read is not certified (the Metal table's rule, rung 4M)."""
    monkeypatch.setenv(SWITCH, "0")
    assert _plan_on(monkeypatch, cupy_like_grid()) is None
    record = fastpath.last_dispatch_report()
    for table in ("triton", "cuda"):
        reason = record["tables"][table]["refused_because"]
        assert reason.startswith("the GPU compute capability could not be read on this "
                                 "host"), reason
        assert reason.endswith(fastpath.CERTIFIED_ONLY_TAIL + ", and an identity that "
                               "cannot be judged is not one"), reason
    assert _notes(capsys.readouterr().err) == []


@pytest.mark.parametrize("value", [None, "0"])
def test_a_ledger_that_cannot_be_read_leaves_the_device_unjudged(monkeypatch, value):
    """Unset keeps the table a candidate, unjudged, as before; ``0`` refuses it by name."""
    _set(monkeypatch, value)
    block_triton(monkeypatch)
    install_cuda_composer(monkeypatch, cuda_step_plan(pairs=()))
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": dict(fastpath._fingerprints("triton")),  # noqa: SLF001
        "cuda_kernels": {"_unreadable": "a stand-in read error"}})
    fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device())
    record = fastpath.last_dispatch_report()
    assert record["environment"]["device_certified_by_table"]["cuda"] is None
    row = record["tables"]["cuda"]
    if value is None:
        assert row["candidate"] is True, row
    else:
        assert row["candidate"] is False
        assert row["refused_because"].startswith(
            "the cuda table's certification record could not be read, so GPU compute "
            "capability 8.6 cannot be judged"), row


# --- NVIDIA: the Triton version ---------------------------------------------------------


@pytest.mark.parametrize("value", [None, "0"])
def test_an_unsupported_triton_is_refused_by_name(monkeypatch, value):
    _set(monkeypatch, value)
    block_cuda(monkeypatch)
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(),
                    version="9.9.9-unreleased") is None
    record = fastpath.last_dispatch_report()
    row = record["tables"]["triton"]["refused_because"]
    assert row.startswith("Triton 9.9.9-unreleased is not in the recorded "
                          "validated_triton_versions"), row
    if value is None:
        assert ("outside the Triton versions the triton table supports "
                "(>=3.1,<3.2)") in row, row
        assert row.endswith(fastpath.UNSUPPORTED_HINT), row
    else:
        assert row.endswith(fastpath.CERTIFIED_ONLY_TAIL), row
    assert record["environment"]["triton_supported"] is False
    assert record["certified"] is None


def test_an_unsupported_triton_runs_under_one(monkeypatch, capsys):
    monkeypatch.setenv(SWITCH, "1")
    block_cuda(monkeypatch)
    plan = _plan_on(monkeypatch, cupy_like_grid_with_device(),
                    version="9.9.9-unreleased")
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["tables"]["triton"] == {"candidate": True, "refused_because": None}
    assert record["uncertified"]["served"] == [{
        "table": "triton", "what": "Triton", "read": "9.9.9-unreleased",
        "certified": list(fastpath.validated_triton_versions()),
        "because": _unsupported_entry_because("Triton", "9.9.9-unreleased", "triton",
                                              TRITON_SCOPE),
        "supported": False}]
    assert record["environment"]["triton"] == "9.9.9-unreleased"
    assert record["environment"]["triton_certified"] is False
    assert record["environment"]["device_certified"] is True
    notes = _notes(capsys.readouterr().err)
    assert len(notes) == 1, notes
    certified = ", ".join(fastpath.validated_triton_versions())
    assert f"Triton 9.9.9-unreleased (certified: {certified})" in notes[0], notes
    assert (f"because {_unsupported_clause('Triton version', with_range=True)}; "
            in notes[0]), notes
    assert "NVIDIA GPU" not in notes[0], notes


@pytest.mark.parametrize("value", [None, "1"])
def test_a_supported_uncertified_triton_dispatches_by_default(monkeypatch, capsys, value):
    _set(monkeypatch, value)
    block_cuda(monkeypatch)
    plan = _plan_on(monkeypatch, cupy_like_grid_with_device(), version="3.1.1")
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["uncertified"]["served"] == [{
        "table": "triton", "what": "Triton", "read": "3.1.1",
        "certified": list(fastpath.validated_triton_versions()),
        "because": _supported_entry_because("Triton", "3.1.1", "triton", TRITON_SCOPE),
        "supported": True}]
    assert record["environment"]["triton_supported"] is True
    lines = _meep_lines(capsys.readouterr().err)
    status = _status("\n".join(lines))
    assert len(status) == 1 and "triton 3.1.1 supported, UNCERTIFIED" in status[0], lines
    notes = _notes("\n".join(lines))
    assert len(notes) == 1 and "Triton 3.1.1 (certified:" in notes[0], notes
    assert "welds_without_a_live_run_here" not in notes[0], notes
    # THE DEVICE IS CERTIFIED HERE, so the NOTE names the Triton version alone and
    # never calls the GPU uncertified (the status line says it is certified).
    assert f"because {_supported_clause('Triton version is')}; " in notes[0], notes
    assert "NVIDIA GPU" not in notes[0], notes
    assert status[0].endswith("NVIDIA RTX A6000 certified"), status


def test_certified_only_refuses_a_supported_uncertified_triton(monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    block_cuda(monkeypatch)
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(), version="3.1.1") is None
    row = fastpath.last_dispatch_report()["tables"]["triton"]["refused_because"]
    assert row.startswith("Triton 3.1.1 is not in the recorded "
                          "validated_triton_versions"), row
    assert row.endswith(fastpath.CERTIFIED_ONLY_TAIL), row


def test_a_table_admitted_on_one_identity_and_refused_on_another_is_dropped_whole(
        monkeypatch, capsys):
    """A supported Triton on an unsupported device: the version admission moves to
    ``dropped`` with the device refusal, so the NOTE never names what did not run."""
    block_cuda(monkeypatch)
    (capability, spelled), = UNSUPPORTED[:1]
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(capability=capability),
                    version="3.1.1") is None
    record = fastpath.last_dispatch_report()
    assert record["uncertified"]["admitted"] == []
    (dropped,) = record["uncertified"]["dropped"]
    assert (dropped["table"], dropped["what"], dropped["read"]) == (
        "triton", "Triton", "3.1.1")
    assert f"GPU compute capability {spelled}" in dropped["dropped_because"], dropped
    assert _notes(capsys.readouterr().err) == []


# --- NVIDIA: a certified table outranks a supported, uncertified one (rung 4e) ----------


def _envelope_grid_with_device(capability):
    """The configuration the CUDA release admits, on a device that reports ``capability``."""
    grid = certified_envelope_grid()
    grid.xp.cuda = cupy_like_grid_with_device(capability=capability).xp.cuda
    return grid


def _one_table_certifies_9_0(monkeypatch, table):
    """The shipped ledgers, with compute capability 9.0 certified on ``table`` only."""
    other = "triton" if table == fastpath.CUDA_TABLE else fastpath.CUDA_TABLE
    ledgers = {f"{table}_kernels": ledger_also_certified_on(table, "9.0"),
               f"{other}_kernels": dict(fastpath._fingerprints(other))}  # noqa: SLF001
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", ledgers)


def _two_table_plan(monkeypatch, capability=(9, 0)):
    stub_triton(monkeypatch)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    return fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _envelope_grid_with_device(capability))


#: Which table certifies 9.0, the slots it serves alone, and the other table.
OUTRANKING = [("cuda", ["step_D", "update_E"], "triton"),
              ("triton", ["step_B"], "cuda")]


@pytest.mark.parametrize("certified,slots,dropped", OUTRANKING)
@pytest.mark.parametrize("value", [None, "0"])
def test_a_certified_table_outranks_a_supported_uncertified_one(
        monkeypatch, capsys, certified, slots, dropped, value):
    """The default never makes a host less certified than the certified table alone."""
    _set(monkeypatch, value)
    _one_table_certifies_9_0(monkeypatch, certified)
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is True
    assert record["composition"]["tables_dispatched"] == [certified]
    assert sorted(plan.slots) == slots
    assert record["uncertified"]["admitted"] == []
    row = record["tables"][dropped]
    assert row["candidate"] is False
    if value is None:
        assert row["refused_because"] == (
            f"the {dropped} table is supported but not certified here (GPU compute "
            f"capability 9.0), and the {certified} table is certified for this device "
            f"and toolchain; set {SWITCH}=1 to compose it too"), row
        assert row["dropped_for_a_certified_table"] == [certified]
        assert row["welds_without_a_live_run_here"], row
        (entry,) = record["uncertified"]["dropped"]
        assert (entry["table"], entry["read"]) == (dropped, "9.0")
        assert entry["dropped_because"] == row["refused_because"]
    else:
        assert row["refused_because"].endswith(fastpath.CERTIFIED_ONLY_TAIL), row
    lines = _meep_lines(capsys.readouterr().err)
    assert _notes("\n".join(lines)) == [], lines
    (status,) = _status("\n".join(lines))
    assert status.endswith("NVIDIA RTX A6000 certified"), status


@pytest.mark.parametrize("certified,slots,dropped", OUTRANKING)
def test_one_composes_the_supported_table_beside_the_certified_one(
        monkeypatch, capsys, certified, slots, dropped):
    monkeypatch.setenv(SWITCH, "1")
    _one_table_certifies_9_0(monkeypatch, certified)
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert sorted(record["composition"]["tables_dispatched"]) == ["cuda", "triton"]
    assert [entry["table"] for entry in record["uncertified"]["served"]] == [dropped]
    assert record["uncertified"]["dropped"] == []
    (note,) = _notes(capsys.readouterr().err)
    # The GPU IS certified for the other table, so the reason names the table it is
    # about; only a Triton-table entry speaks for the toolchain too.
    subject = ("NVIDIA GPU and toolchain are" if dropped == "triton"
               else "NVIDIA GPU is")
    assert (f"because {_supported_clause(subject, where=f' on the {dropped} table')}; "
            in note), note


@pytest.mark.parametrize("certified,slots,dropped", OUTRANKING)
def test_a_preference_for_the_dropped_table_is_refused_with_the_drop(
        monkeypatch, certified, slots, dropped):
    """Rung 4d's named refusal, carrying why the table it names is not a candidate."""
    _one_table_certifies_9_0(monkeypatch, certified)
    monkeypatch.setenv(fastpath.BACKEND_PREFERENCE_SWITCH, dropped)
    assert _two_table_plan(monkeypatch) is None
    record = fastpath.last_dispatch_report()
    reason = record["refused_because"]
    assert reason.startswith(f"{fastpath.BACKEND_PREFERENCE_SWITCH}={dropped} names no "
                             "candidate table on this host"), reason
    assert reason.endswith(f"; {dropped}: {record['tables'][dropped]['refused_because']}")
    assert record["arbitration"]["refused_because"] == reason
    # ...and with ``1`` the preference is honoured, uncertified.
    monkeypatch.setenv(SWITCH, "1")
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["arbitration"]["effective_order"][0] == dropped
    assert plan.report()["certified"] is False


def test_a_preference_for_the_certified_table_is_unchanged(monkeypatch):
    _one_table_certifies_9_0(monkeypatch, "cuda")
    monkeypatch.setenv(fastpath.BACKEND_PREFERENCE_SWITCH, "cuda")
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["certified"] is True
    assert plan.report()["arbitration"]["effective_order"] == ["cuda"]


def test_the_kernel_table_switch_naming_the_uncertified_table_runs_it_uncertified(
        monkeypatch, capsys):
    """Rung 3 leaves the other table unconsulted, so no certified table is a candidate
    and rung 4e has nothing to prefer: the table asked for runs, with the NOTE."""
    _one_table_certifies_9_0(monkeypatch, "cuda")
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "triton")
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["composition"]["tables_dispatched"] == ["triton"]
    assert len(_notes(capsys.readouterr().err)) == 1


def test_an_unreadable_table_is_not_dropped_for_a_certified_one(monkeypatch):
    """Rung 4e drops what was ADMITTED uncertified; an unjudged table was admitted by
    nothing and composes as it did before."""
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": dict(fastpath._fingerprints("triton")),  # noqa: SLF001
        "cuda_kernels": {"_unreadable": "a stand-in read error"}})
    plan = _two_table_plan(monkeypatch, capability=(8, 6))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert sorted(record["composition"]["tables_dispatched"]) == ["cuda", "triton"]
    assert record["certified"] is None
    assert record["uncertified"]["dropped"] == []


# --- NVIDIA: the NOTE names what was judged, and nothing else ---------------------------


@pytest.mark.parametrize("capability,version,supported,outside", [
    ((6, 1), "3.1.1", "Triton version is", "NVIDIA GPU"),
    ((7, 5), "3.2.0", "NVIDIA GPU is", "Triton version")])
def test_under_one_a_mixed_host_says_which_identity_is_supported(
        monkeypatch, capsys, capability, version, supported, outside):
    """One identity inside the supported range, the other outside it: the NOTE calls
    supported only the one that is, names the other as what ``1`` ran, and states the
    range once."""
    monkeypatch.setenv(SWITCH, "1")
    plan = _plan_on(monkeypatch, cupy_like_grid_with_device(capability=capability),
                    version=version)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    verdicts = {(entry["what"], entry["supported"])
                for entry in record["uncertified"]["served"]}
    gpu_supported = supported == "NVIDIA GPU is"
    assert verdicts == {("Triton", not gpu_supported),
                        ("GPU compute capability", gpu_supported)}, verdicts
    (note,) = _notes(capsys.readouterr().err)
    assert (f"because {_supported_clause(supported)}; and "
            f"{_unsupported_clause(outside, with_range=False)}") in note, note
    assert note.count("the supported range is") == 1, note
    assert "GPU and toolchain" not in note, note
    assert note.endswith(NOTE_TAIL), note


def test_a_hand_cuda_run_claims_no_toolchain(monkeypatch, capsys):
    """The hand-CUDA table reads no toolchain version, so on a supported, uncertified
    GPU its NOTE calls the GPU supported and says nothing about a toolchain."""
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "cuda")
    plan = _two_table_plan(monkeypatch, capability=(8, 9))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["composition"]["tables_dispatched"] == ["cuda"]
    (entry,) = record["uncertified"]["served"]
    assert entry["because"] == _supported_entry_because(
        "GPU compute capability", "8.9", "cuda", CAPABILITY_SCOPE), entry
    lines = _meep_lines(capsys.readouterr().err)
    (status,) = _status("\n".join(lines))
    assert status.endswith("supported, UNCERTIFIED"), status
    (note,) = _notes("\n".join(lines))
    assert f"because {_supported_clause('NVIDIA GPU is')}; " in note, note
    assert "toolchain are" not in note, note


# --- the value ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", UNACCEPTED_VALUES)
@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (10, 0)])
def test_any_other_value_is_refused_by_name_on_every_device(
        monkeypatch, capsys, value, capability):
    """Refused on the certified device as well: a typed value is never ignored."""
    monkeypatch.setenv(SWITCH, value)
    grid = cupy_like_grid_with_device(capability=capability)
    assert _plan_on(monkeypatch, grid) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    reason = record["refused_because"]
    assert reason.startswith(f"{SWITCH}={value!r} is not an accepted value"), reason
    assert record["uncertified"]["value"] == value
    assert record["uncertified"]["allowed"] is False
    assert record["certified"] is None
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"{SWITCH}={value!r}" in lines[0], lines


@pytest.mark.parametrize("value", [None, "1"])
def test_the_kill_switch_and_the_disable_answer_first(monkeypatch, value):
    """Admitting an identity enables nothing the rungs above refuse."""
    _set(monkeypatch, value)
    grid = cupy_like_grid_with_device(capability=SUPPORTED_UNCERTIFIED[0][0])
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    assert _plan_on(monkeypatch, grid) is None
    assert fastpath.DISPATCH_ENABLE in fastpath.last_dispatch_report()["refused_because"]
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    assert _plan_on(monkeypatch, grid) is None
    assert fastpath.FUSED_KILL_SWITCH in fastpath.last_dispatch_report()["refused_because"]


# --- Apple: the torch version and the Metal frontend ------------------------------------


def _require_metal() -> None:
    if backends.available_gpu() != "metal":
        requires_resource_skip("mps_device", "the Metal ladder is stepped on an "
                                             "Apple GPU")


def _driver() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(6.0, 6.0, 0.0), resolution=10.0, dimensions=2,
                        force_complex_fields=False, prefer_gpu=True)
    driver.setup_pml({"x": 10.0, "y": 10.0})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                       "amplitude": 1.0})
    return driver


#: The environment the substituted cited welds record, and that the substituted host
#: reads unless a test changes one fact. Synthetic on both sides, so no test depends
#: on what the shipped ledger records or on which Mac it runs.
CERTIFIED_ENVIRONMENT = {"architecture": "applegpu_test0", "torch": "9.9.9-certified",
                         "metal_frontend": "99999.1"}
#: The toolchain keys ``metal_dispatch.metal_toolchain`` reads each fact into; fast
#: math is certified unset, whatever the welds recorded.
TOOLCHAIN_KEYS = {"architecture": "architecture", "torch": "version",
                  "metal_frontend": "metal_frontend", "fast_math": "fast_math"}
#: Why an uncertified environment ran: on an Apple GPU, that every Apple GPU is
#: supported; on any other GPU, that the Metal table runs uncertified by default.
SUPPORTED_REMINDER = (f"this Apple GPU is supported, and {SWITCH}=0 would restrict the "
                      f"Metal kernels to certified environments")
UNCERTIFIED_REMINDER = (f"the Metal table runs on environments outside the certified "
                        f"set unless {SWITCH}=0")


def _environment(monkeypatch, fact=None, value=None, recorded=None) -> None:
    """Substitute what the Metal ladder reads and what the cited welds recorded.

    The cited welds record ``recorded`` (default :data:`CERTIFIED_ENVIRONMENT`), and
    the host reads that environment with ``fact`` set to ``value``. Substituting one
    side only would leave the other to this host and the shipped ledger, which need
    not agree, and a test would then judge a fact it did not choose.
    """
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    recorded = dict(CERTIFIED_ENVIRONMENT if recorded is None else recorded)
    real = metal_dispatch.metal_toolchain

    def substituted():
        toolchain = dict(real())
        for name, key in TOOLCHAIN_KEYS.items():
            toolchain[key] = CERTIFIED_ENVIRONMENT.get(name)
        toolchain["metal_frontend"] = "metalfe-" + CERTIFIED_ENVIRONMENT["metal_frontend"]
        if fact is not None:
            toolchain[TOOLCHAIN_KEYS[fact]] = value
        return toolchain

    monkeypatch.setattr(metal_dispatch, "metal_toolchain", substituted)
    # ONE CITED WELD, holding one live run for the recorded architecture; a recorded
    # architecture of None is a weld whose entry cannot be read as per-architecture
    # runs (``metal_dispatch.cited_environments``).
    runs = (recorded,) if recorded.get("architecture") else None
    monkeypatch.setattr(metal_dispatch, "cited_environments",
                        lambda ledger=None: (("metal_test_device_gate", runs),))


METAL_FACTS = [
    ("architecture", "applegpu_other", "GPU architecture", "device_certified"),
    ("torch", "0.0.0-not-certified", "torch", "torch_certified"),
    ("metal_frontend", "metalfe-0.0.0", "Metal frontend", "frontend_certified"),
    ("fast_math", "1", "PYTORCH_MPS_FAST_MATH", "fast_math_certified"),
]


def _run(steps=STEPS):
    driver = _driver()
    for _ in range(steps):
        driver.step()
    return driver, driver.fast_path_report()


@pytest.mark.parametrize("fact,value,what,mark", METAL_FACTS)
@pytest.mark.parametrize("switch", [None, "1"])
def test_an_uncertified_metal_environment_runs_by_default_and_matches_a_certified_one(
        monkeypatch, capsys, fact, value, what, mark, switch):
    """The kernels that run are the ones a certified environment runs: same bytes."""
    _require_metal()
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    _environment(monkeypatch)
    driver, report = _run()
    assert driver.active_step_path == "fused", report["refused_because"]
    assert report["certified"] is True
    certified_field = np.array(driver.get_field("Ez"), copy=True)
    driver.close()
    capsys.readouterr()
    fastpath.reset_dispatch_announcements()

    _environment(monkeypatch, fact, value)
    for _ in range(2):
        driver, report = _run()
        assert driver.active_step_path == "fused", report["refused_because"]
        assert report["decision"] == "dispatched" and report["table"] == "metal"
        assert report["certified"] is False
        assert report["environment"][mark] is False
        served = report["uncertified"]["served"]
        assert [(entry["table"], entry["what"], entry["read"]) for entry in served] \
            == [("metal", what, value)], served
        wanted = CERTIFIED_ENVIRONMENT.get(fact, "unset")
        assert served[0]["certified"] == [wanted], served
        assert served[0]["because"] == SUPPORTED_REMINDER, served
        field = np.array(driver.get_field("Ez"), copy=True)
        driver.close()
        assert field.tobytes() == certified_field.tobytes()
    lines = _meep_lines(capsys.readouterr().err)
    notes = _notes("\n".join(lines))
    assert len(notes) == 1, lines
    assert f"{what} {value} (certified: {wanted})" in notes[0], notes
    assert f"dispatched because {SUPPORTED_REMINDER};" in notes[0], notes
    assert f"{SWITCH}=1" not in notes[0], notes
    status = [line for line in lines if line.startswith("meep_gpu: step path fused;")]
    assert len(status) == 1, lines
    # SUPPORTED IS NOT CERTIFIED: an Apple GPU no weld ran on is both, by name.
    expected = ("(applegpu_other) supported, UNCERTIFIED" if fact == "architecture"
                else "(applegpu_test0) certified")
    assert status[0].endswith(expected), status


def test_a_gpu_that_is_not_apples_runs_uncertified_and_is_not_called_supported(
        monkeypatch, capsys):
    _require_metal()
    _environment(monkeypatch, "architecture", "other_gpu_arch")
    driver, report = _run()
    assert driver.active_step_path == "fused", report["refused_because"]
    assert report["certified"] is False
    assert report["environment"]["device_supported"] is False
    assert report["uncertified"]["served"][0]["because"] == UNCERTIFIED_REMINDER
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    status = [line for line in lines if line.startswith("meep_gpu: step path fused;")]
    assert status and status[0].endswith("(other_gpu_arch) UNCERTIFIED"), lines
    assert f"dispatched because {UNCERTIFIED_REMINDER};" in _notes("\n".join(lines))[0]


@pytest.mark.parametrize("fact,value,what,mark", METAL_FACTS)
def test_certified_only_refuses_an_uncertified_metal_environment_by_name(
        monkeypatch, capsys, fact, value, what, mark):
    _require_metal()
    monkeypatch.setenv(SWITCH, "0")
    _environment(monkeypatch, fact, value)
    driver, report = _run()
    assert driver.active_step_path == "array"
    assert report["decision"] == "refused" and report["certified"] is None
    reason = report["refused_because"]
    expected = (f"{what}={value} compiles the Metal kernels in fast-math mode"
                if fact == "fast_math" else
                f"{what} {value} is not the one every Metal weld this table cites ran on")
    assert reason.startswith(expected), reason
    assert reason.endswith(f"{SWITCH}=0 restricts the Metal kernels to certified "
                           "environments"), reason
    assert report["environment"][mark] is False
    assert report["environment"]["certified_only"] is True
    driver.close()
    assert _notes(capsys.readouterr().err) == []


def test_certified_only_runs_a_certified_metal_environment(monkeypatch, capsys):
    _require_metal()
    monkeypatch.setenv(SWITCH, "0")
    _environment(monkeypatch)
    driver, report = _run()
    assert driver.active_step_path == "fused", report["refused_because"]
    assert report["certified"] is True
    assert [report["environment"][key] for key in (
        "device_certified", "torch_certified", "frontend_certified")] == [True] * 3
    driver.close()
    assert _notes(capsys.readouterr().err) == []


#: An environment the records cannot judge: a fact this host could not read, and a
#: cited weld whose entry holds no per-architecture run record.
UNJUDGED = [
    ("unread", "architecture", None, CERTIFIED_ENVIRONMENT,
     "the GPU architecture could not be read on this host"),
    ("unrecorded", None, None, dict(CERTIFIED_ENVIRONMENT, architecture=None),
     "GPU architecture applegpu_test0 cannot be certified: 1 of the 1 Metal welds "
     "this table cites hold no per-architecture run record to judge it by"),
]


@pytest.mark.parametrize("case,fact,value,recorded,cause", UNJUDGED,
                         ids=[row[0] for row in UNJUDGED])
@pytest.mark.parametrize("switch", [None, "0"])
def test_an_environment_that_cannot_be_judged_runs_unjudged_or_is_refused(
        monkeypatch, capsys, case, fact, value, recorded, cause, switch):
    """Unjudged is not uncertified: it runs by default, and certified-only refuses it."""
    _require_metal()
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    _environment(monkeypatch, fact, value, recorded)
    driver, report = _run()
    assert report["environment"]["device_certified"] is None
    if switch is None:
        assert driver.active_step_path == "fused", report["refused_because"]
        assert report["certified"] is None
        assert report["uncertified"]["served"] == []
    else:
        assert driver.active_step_path == "array"
        assert report["refused_because"].startswith(cause), report["refused_because"]
        assert report["refused_because"].endswith(
            "and an environment that cannot be judged is not one")
    driver.close()
    assert _notes(capsys.readouterr().err) == []


@pytest.mark.parametrize("value", UNACCEPTED_VALUES)
def test_any_other_value_is_refused_by_name_on_an_apple_gpu(monkeypatch, capsys, value):
    _require_metal()
    monkeypatch.setenv(SWITCH, value)
    driver = _driver()
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "array"
    assert report["refused_because"].startswith(
        f"{SWITCH}={value!r} is not an accepted value"), report["refused_because"]
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert lines[0].startswith("meep_gpu: step path array on the host CPU; dispatch "
                               f"refused: {SWITCH}={value!r}"), lines
