"""The opt-in for a device or toolchain that is not certified.

``MEEP_GPU_ALLOW_UNCERTIFIED`` is a strict switch: ``1`` lets the kernels dispatch
on an identity that was read and is not in the certified set, ``0`` and unset keep
the refusal to the array path, and every other value is refused by name. A run it
admits carries ``certified: False`` in the dispatch record, with the identity
read, and the process prints one line naming what is certified.

The NVIDIA half substitutes the identity: a stand-in for CuPy that reports
compute capability 8.9 or 9.0, and a stand-in Triton of another version, so it
runs on a host with no NVIDIA device. The Apple half substitutes the toolchain
the Metal ladder reads and steps a real driver; it is skipped by a declared
resource on a host with no Apple GPU.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath
from meep_gpu.driver import FdtdDriver
from meep_gpu.test_dispatch_contract import (
    CountingPlan,
    block_cuda,
    cupy_like_grid,
    cupy_like_grid_with_device,
    install_composer,
    step_plan,
    stub_triton,
)

SWITCH = "MEEP_GPU_ALLOW_UNCERTIFIED"
SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
            "MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_FUSE_ARMS",
            "MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_DISPATCH_LOG",
            "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL", SWITCH)

#: Compute capabilities no recorded gate ran on; the certified one is 8.6.
UNCERTIFIED_CAPABILITIES = [((8, 9), "8.9"), ((9, 0), "9.0")]
#: Values that are neither ``1`` nor ``0``, the two a user is most likely to type
#: among them.
UNACCEPTED_VALUES = ["true", "yes", "", " 1", "2", "on"]
STEPS = 5


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


def _plan_on(monkeypatch, grid, version="3.1.0"):
    """One freeze on a stand-in NVIDIA host whose composer fills one slot."""
    stub_triton(monkeypatch, version)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    return fastpath.plan_fast_path(object(), object(), grid)


# --- the switch itself ------------------------------------------------------------------


def test_the_switch_is_named_and_strict():
    assert fastpath.UNCERTIFIED_SWITCH == SWITCH
    assert dict(fastpath.UNCERTIFIED_VALUES) == {"1": True, "0": False}


@pytest.mark.parametrize("value,allowed", [(None, False), ("0", False), ("1", True)]
                         + [(value, False) for value in UNACCEPTED_VALUES])
def test_only_one_admits(monkeypatch, value, allowed):
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    assert fastpath.uncertified_allowed() is allowed
    block = fastpath._base_record()["uncertified"]  # noqa: SLF001
    assert block == {"variable": SWITCH, "value": value, "allowed": allowed,
                     "admitted": []}


# --- NVIDIA: the compute capability -----------------------------------------------------


@pytest.mark.parametrize("capability,spelled", UNCERTIFIED_CAPABILITIES)
@pytest.mark.parametrize("value", [None, "0"])
def test_an_uncertified_device_is_refused_and_the_refusal_names_the_switch(
        monkeypatch, capsys, capability, spelled, value):
    """Unset is the refusal the package has always made, and ``0`` is the same one."""
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    assert _plan_on(monkeypatch, grid) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    reason = record["refused_because"]
    assert f"GPU compute capability {spelled} is not in the recorded" in reason, reason
    assert f"set {SWITCH}=1" in reason, reason
    for table in ("triton", "cuda"):
        row = record["tables"][table]
        assert row["candidate"] is False
        assert row["refused_because"].endswith(fastpath.UNCERTIFIED_HINT), row
    assert record["environment"]["device"]["compute_capability"] == spelled
    assert record["environment"]["device_certified"] is False
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"set {SWITCH}=1" in lines[0], lines
    assert _notes("\n".join(lines)) == []


@pytest.mark.parametrize("capability,spelled", UNCERTIFIED_CAPABILITIES)
def test_the_refusal_is_the_same_unset_and_at_zero(monkeypatch, capability, spelled):
    grid = cupy_like_grid_with_device(capability=capability)
    assert _plan_on(monkeypatch, grid) is None
    unset = fastpath.last_dispatch_report()["refused_because"]
    monkeypatch.setenv(SWITCH, "0")
    assert _plan_on(monkeypatch, grid) is None
    assert fastpath.last_dispatch_report()["refused_because"] == unset


@pytest.mark.parametrize("capability,spelled", UNCERTIFIED_CAPABILITIES)
def test_an_opted_in_device_dispatches_uncertified_and_says_so_once(
        monkeypatch, capsys, capability, spelled):
    monkeypatch.setenv(SWITCH, "1")
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    plan = _plan_on(monkeypatch, grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["decision"] == "dispatched" and record["step_path"] == "fused"
    assert record["certified"] is False
    block = record["uncertified"]
    assert block["variable"] == SWITCH and block["value"] == "1" and block["allowed"]
    served = block["served"]
    assert served, block
    assert {entry["table"] for entry in served} <= {"triton", "cuda"}
    for entry in served:
        assert entry["what"] == "GPU compute capability"
        assert entry["read"] == spelled
        assert entry["certified"] == ["8.6"]
    # The identity is recorded as READ: the opt-in changes what runs, not what is true.
    environment = record["environment"]
    assert environment["device"]["compute_capability"] == spelled
    assert environment["device"]["name"] == "another device"
    assert environment["device_certified"] is False
    assert environment["triton_certified"] is True

    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 2, lines
    assert lines[0].startswith("meep_gpu: step path fused;"), lines
    assert lines[0].endswith("another device UNCERTIFIED"), lines
    note = lines[1]
    assert note.startswith("meep_gpu: NOTE the kernels are NOT CERTIFIED"), note
    assert f"GPU compute capability {spelled} (certified: 8.6)" in note, note
    assert f"{SWITCH}=1" in note and "prefer_gpu=False" in note, note

    # A second freeze in the same process repeats neither line.
    assert _plan_on(monkeypatch, grid) is not None
    assert _meep_lines(capsys.readouterr().err) == []


def test_the_certified_device_is_certified_with_or_without_the_opt_in(
        monkeypatch, capsys):
    for value in (None, "0", "1"):
        if value is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, value)
        plan = _plan_on(monkeypatch, cupy_like_grid_with_device())
        assert plan is not None, value
        record = plan.report()
        assert record["certified"] is True, value
        assert record["uncertified"]["admitted"] == [], value
        assert record["uncertified"]["served"] == [], value
    assert _notes(capsys.readouterr().err) == []


def test_an_unreadable_device_is_neither_certified_nor_admitted(monkeypatch, capsys):
    """Three-valued: the opt-in acts on an identity that was read, and on no other."""
    monkeypatch.setenv(SWITCH, "1")
    plan = _plan_on(monkeypatch, cupy_like_grid())
    assert plan is not None
    record = plan.report()
    assert record["environment"]["device_certified"] is None
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    assert _notes(capsys.readouterr().err) == []


# --- NVIDIA: the Triton version ---------------------------------------------------------


def test_an_uncertified_triton_is_refused_and_the_refusal_names_the_switch(monkeypatch):
    block_cuda(monkeypatch)
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(),
                    version="9.9.9-unreleased") is None
    record = fastpath.last_dispatch_report()
    row = record["tables"]["triton"]["refused_because"]
    assert row.startswith("Triton 9.9.9-unreleased is not in the recorded "
                          "validated_triton_versions"), row
    assert row.endswith(fastpath.UNCERTIFIED_HINT), row
    assert record["certified"] is None


def test_an_opted_in_triton_dispatches_uncertified(monkeypatch, capsys):
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
        "certified": list(fastpath.validated_triton_versions())}]
    assert record["environment"]["triton"] == "9.9.9-unreleased"
    assert record["environment"]["triton_certified"] is False
    assert record["environment"]["device_certified"] is True
    notes = _notes(capsys.readouterr().err)
    assert len(notes) == 1, notes
    assert "Triton 9.9.9-unreleased (certified: 3.1.0)" in notes[0], notes


# --- the value ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", UNACCEPTED_VALUES)
@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0)])
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


def test_the_kill_switch_and_the_disable_answer_first(monkeypatch):
    """The opt-in admits an identity; it enables nothing the rungs above refuse."""
    monkeypatch.setenv(SWITCH, "1")
    grid = cupy_like_grid_with_device(capability=(9, 0))
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


def _certified_pair() -> tuple:
    """A (PyTorch, Metal frontend) pair the Metal ledger certifies, as it records it."""
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    for torch_version, frontend in metal_dispatch.validated_toolchains():
        if frontend is not None:
            return torch_version, frontend
    raise AssertionError("the Metal ledger names no certified (PyTorch, Metal frontend) "
                         "pair")


def _another_toolchain(monkeypatch, key=None, value=None) -> dict:
    """Substitute the identities the Metal ladder reads; returns what it really is.

    BOTH halves are substituted: the ledger's certified pair, then ``key`` set to
    ``value``. Substituting one half would leave the other to this host's real
    toolchain, which need not be certified (a newer PyTorch is not), and the ladder
    would then refuse on the half the test did not choose. With no ``key`` the host
    reads as the certified pair.
    """
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    real = metal_dispatch.metal_toolchain
    actual = dict(real())
    torch_version, frontend = _certified_pair()

    def substituted():
        toolchain = dict(real())
        toolchain["version"] = torch_version
        toolchain["metal_frontend"] = frontend
        if key is not None:
            toolchain[key] = value
        return toolchain

    monkeypatch.setattr(metal_dispatch, "metal_toolchain", substituted)
    return actual


METAL_IDENTITIES = [
    ("version", "0.0.0-not-certified", "torch", "torch_certified"),
    ("metal_frontend", "metalfe-0.0.0", "Metal frontend", "frontend_certified"),
]


@pytest.mark.parametrize("key,value,what,mark", METAL_IDENTITIES)
@pytest.mark.parametrize("switch", [None, "0"])
def test_an_uncertified_metal_toolchain_is_refused_and_names_the_switch(
        monkeypatch, capsys, key, value, what, mark, switch):
    _require_metal()
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    _another_toolchain(monkeypatch, key, value)
    driver = _driver()
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "array"
    assert report["decision"] == "refused" and report["certified"] is None
    reason = report["refused_because"]
    assert reason.startswith(f"{what} {value} is not in the toolchains any Metal "
                             "weld recorded running on"), reason
    assert reason.endswith(fastpath.UNCERTIFIED_HINT), reason
    assert report["environment"][mark] is False
    driver.close()
    assert _notes(capsys.readouterr().err) == []


@pytest.mark.parametrize("key,value,what,mark", METAL_IDENTITIES)
def test_an_opted_in_metal_toolchain_dispatches_uncertified_and_matches_the_reference(
        monkeypatch, capsys, key, value, what, mark):
    """The kernels that run are the ones a certified identity runs: same bytes."""
    _require_metal()
    monkeypatch.setenv(SWITCH, "1")
    _another_toolchain(monkeypatch)
    driver = _driver()
    for _ in range(STEPS):
        driver.step()
    assert driver.active_step_path == "fused"
    assert driver.fast_path_report()["certified"] is True
    certified_field = np.array(driver.get_field("Ez"), copy=True)
    driver.close()
    capsys.readouterr()
    fastpath.reset_dispatch_announcements()

    _another_toolchain(monkeypatch, key, value)
    for _ in range(2):
        driver = _driver()
        for _ in range(STEPS):
            driver.step()
        report = driver.fast_path_report()
        assert driver.active_step_path == "fused", report["refused_because"]
        assert report["decision"] == "dispatched" and report["table"] == "metal"
        assert report["certified"] is False
        assert report["environment"][mark] is False
        served = report["uncertified"]["served"]
        assert [(entry["table"], entry["what"], entry["read"]) for entry in served] \
            == [("metal", what, value)], served
        assert served[0]["certified"], served
        field = np.array(driver.get_field("Ez"), copy=True)
        driver.close()
        assert field.tobytes() == certified_field.tobytes()
    lines = _meep_lines(capsys.readouterr().err)
    notes = _notes("\n".join(lines))
    assert len(notes) == 1, lines
    assert f"{what} {value} (certified: " in notes[0], notes
    assert f"{SWITCH}=1" in notes[0], notes
    assert sum(line.startswith("meep_gpu: step path fused;") for line in lines) == 1


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
