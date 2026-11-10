"""Kernels on a device or toolchain that is not certified.

``MEEP_GPU_HARDWARE`` names the hardware scope by value, ``certified``, ``supported``
(the default, when unset) or ``any``; every other value is refused by name, exact
match, and so is the retired ``MEEP_GPU_ALLOW_UNCERTIFIED`` set to anything, with the
value of ``MEEP_GPU_HARDWARE`` that matches what it meant. Every kernel table runs by
default on an identity it SUPPORTS but that no cited gate ran on, records
``certified: False`` with the identity read, and prints one line naming what is
certified and why the kernels ran; ``certified`` restricts the kernels to certified
identities and refuses one it cannot judge.

On the NVIDIA tables supported is a SELF-CHECK, not a range: a Triton ``>=3.1`` whose
health probe compiled and launched a kernel on the device, for the Triton table, and a
CuPy whose NVRTC can target the device, for the hand-CUDA table. An identity read
outside it is refused by name unless the scope is ``any``, which runs it too, recorded
``supported: False``; a Triton that fails its health probe is dropped by name on every
scope. A table that is only supported composes beside one that is certified (there is
no rung 4e), and the NOTE names the table it is uncertified on. On the Metal table
every Apple GPU is supported.

The NVIDIA half substitutes the identity: a stand-in for CuPy that reports a
compute capability and its NVRTC's targets (a supported device neither NVIDIA table
certifies, such as 7.5 or 8.0 while 8.6 and 8.9 are the certified ones, or one NVRTC 11.8
cannot target, such as 10.0 or 12.0), a stand-in Triton of another version with a
stand-in health probe, a stand-in stack, and ledgers that certify a capability on one
table only, so it runs on a host with no NVIDIA device. The Apple
half substitutes both the environment the Metal ladder reads and the one the cited
welds record, and steps a real driver; it is skipped by a declared resource on a host
with no Apple GPU.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath, fastpath_cuda
from meep_gpu.driver import FdtdDriver
from meep_gpu.test_capability_records import (ledger_also_certified_on,
                                              record_certified_stack)
from meep_gpu.test_dispatch_contract import (
    NVRTC_118_ARCHS,
    CountingPlan,
    block_cuda,
    block_triton,
    certified_envelope_fields,
    certified_envelope_grid,
    certified_envelope_pml,
    cuda_step_plan,
    cupy_like_grid,
    cupy_like_grid_with_device,
    failing_triton_probe,
    install_composer,
    install_cuda_composer,
    step_plan,
    stub_triton,
    supported_uncertified_capabilities,
    unsupported_capabilities,
)

SWITCH = "MEEP_GPU_HARDWARE"
#: The name ``SWITCH`` replaced: refused by name, set to anything.
RETIRED = "MEEP_GPU_ALLOW_UNCERTIFIED"
SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
            "MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_FUSE_ARMS",
            "MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_DISPATCH_LOG",
            "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL", SWITCH, RETIRED)

#: Supported compute capabilities no cited gate ran on: the first two candidates both
#: NVIDIA tables support and neither certifies (7.5 and 8.0 while 8.6 and 8.9 are the
#: certified ones), so certifying another architecture (9.0 is the next) leaves them
#: supported and uncertified.
SUPPORTED_UNCERTIFIED = [(capability, f"{capability[0]}.{capability[1]}")
                         for capability in supported_uncertified_capabilities(2)]
#: Compute capabilities the stand-in NVRTC (11.8) cannot target: Blackwell devices.
UNSUPPORTED = [(capability, f"{capability[0]}.{capability[1]}")
               for capability in unsupported_capabilities(2)]
#: Values that are not one of the three, exact match: the retired ``0`` and ``1``,
#: empty and whitespace, another case, padding, and words a user might type.
UNACCEPTED_VALUES = ["0", "1", "", " ", "Any", "ANY", "Certified", " any", "any ",
                     "supported\n", "true", "on"]
#: Values the retired name might still be set to: its two accepted values and others.
RETIRED_VALUES = ["0", "1", "", "x", "true"]
STEPS = 5
#: How the NOTE line ends, whatever made the run uncertified.
NOTE_TAIL = ("compare the results with a prefer_gpu=False run of the same simulation "
             "before relying on them")


#: What the Triton version is judged against, as the per-entry reasons spell it.
TRITON_SCOPE = "Triton >=3.1"
#: How the Triton table's self-check reads on a device its health probe launched on.
TRITON_DEVICE_SCOPE = ("the Triton health probe compiled and launched a kernel on this "
                       "device")


def _capability_scope(table, spelled):
    """An admitted compute capability's self-check, as its entry's reason spells it."""
    if table == "triton":
        return TRITON_DEVICE_SCOPE
    return f"the NVRTC 11.8 of this CuPy targets sm_{spelled.replace('.', '')}"


def _nvrtc_outside(spelled):
    """The hand-CUDA table's self-check on a device NVRTC 11.8 cannot target."""
    return (f"the NVRTC 11.8 of this CuPy targets sm_{NVRTC_118_ARCHS[0]} to "
            f"sm_{NVRTC_118_ARCHS[-1]}, and none of those loads on "
            f"sm_{spelled.replace('.', '')}")


def _supported_entry_because(what, read, table, scope) -> str:
    """An admitted entry's own reason: THAT identity, THAT table's range."""
    return (f"{what} {read} is supported by the {table} table ({scope}) but not "
            "certified bit-identical")


def _unsupported_entry_because(what, read, table, scope) -> str:
    return (f"{SWITCH}=any runs {what} {read}, outside what the {table} table supports "
            f"({scope})")


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
    """Every test starts with the switches unset, dispatch enabled, nothing said, and
    a host FPU that keeps subnormals.

    The last clause is session order, not tidiness. MEEP's own initialization flushes
    the host on x86-64 and never undoes it, so after any earlier test in the session
    imported MEEP the first freeze here reads "installed by dispatch over the run's
    own flush" and the second does not: two different status lines, and the "said
    once" assertions fail in a full run while passing alone. Measured on the NVIDIA
    reference environment: these tests fail after ``test_meep_precision_notice.py``
    and pass without it. The entry state is driven to keep here and put back after.
    """
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    flushing_at_entry = backends.subnormals_flushed()
    if flushing_at_entry:
        subnormal_policy._drive_host_fpu(False)  # noqa: SLF001
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


def _plan_on(monkeypatch, grid, version="3.1.0", probe=None):
    """One freeze on a stand-in NVIDIA host whose composer fills one slot."""
    stub_triton(monkeypatch, version, probe=probe)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    return fastpath.plan_fast_path(object(), object(), grid)


def _set(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)


# --- the switch itself ------------------------------------------------------------------


def test_the_switch_is_named_by_value_and_strict():
    assert fastpath.HARDWARE_SWITCH == SWITCH
    assert fastpath.HARDWARE_VALUES == ("certified", "supported", "any")
    assert fastpath.HARDWARE_DEFAULT == "supported"
    assert (fastpath.HARDWARE_CERTIFIED, fastpath.HARDWARE_SUPPORTED,
            fastpath.HARDWARE_ANY) == fastpath.HARDWARE_VALUES
    assert fastpath.RETIRED_HARDWARE_SWITCH == RETIRED
    assert dict(fastpath.RETIRED_HARDWARE_VALUES) == {"0": "certified", "1": "any"}


def test_the_old_names_are_gone_with_no_alias():
    """Renamed outright: neither ladder keeps a reader, constant or wrapper under the
    old names, so nothing can read the retired variable by accident."""
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    for name in ("UNCERTIFIED_SWITCH", "UNCERTIFIED_VALUES", "certified_only",
                 "unsupported_allowed", "CERTIFIED_ONLY_TAIL",
                 "_uncertified_value_refusal"):
        assert not hasattr(fastpath, name), name
    assert not hasattr(metal_dispatch, "certified_only")


@pytest.mark.parametrize("value,scope,allowed,unsupported", [
    (None, "supported", True, False), ("certified", "certified", False, False),
    ("supported", "supported", True, False), ("any", "any", True, True)]
    + [(value, None, False, False) for value in UNACCEPTED_VALUES])
def test_the_record_states_what_each_value_allows(monkeypatch, value, scope, allowed,
                                                  unsupported):
    """``allowed``: a supported identity that is not certified may compose
    (``supported``, the default, and ``any``); ``unsupported_allowed``: one outside the
    supported range may (``any``). A value rung 3b refuses is no scope at all."""
    _set(monkeypatch, value)
    assert fastpath.hardware_scope() == scope
    block = fastpath._base_record()["uncertified"]  # noqa: SLF001
    assert block == {"variable": SWITCH, "value": value, "allowed": allowed,
                     "unsupported_allowed": unsupported, "admitted": [], "dropped": []}


@pytest.mark.parametrize("value", [None, "certified", "supported", "any"]
                         + UNACCEPTED_VALUES)
def test_both_ladders_record_the_restriction_from_one_reader(monkeypatch, value):
    """The NVIDIA and the Metal environment blocks state ``certified_only`` from the one
    scope reader, so the two ladders cannot read the switch two ways."""
    import types  # noqa: PLC0415

    from meep_gpu import metal_dispatch  # noqa: PLC0415

    _set(monkeypatch, value)
    restricted = value == "certified"
    assert (fastpath.hardware_scope() == fastpath.HARDWARE_CERTIFIED) is restricted
    metal = metal_dispatch.environment_block(types.SimpleNamespace(xp=np), {})
    assert metal["certified_only"] is restricted


# --- the retired name -------------------------------------------------------------------


@pytest.mark.parametrize("value", RETIRED_VALUES)
@pytest.mark.parametrize("scope", [None, "supported", "any"])
@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (10, 0)])
def test_the_retired_name_is_refused_by_name_with_its_replacement(
        monkeypatch, capsys, value, scope, capability):
    """Set to anything, beside any scope, on the certified device as well: never read,
    and the refusal names the replacement and the value that does what it did."""
    monkeypatch.setenv(RETIRED, value)
    _set(monkeypatch, scope)
    grid = cupy_like_grid_with_device(capability=capability)
    assert _plan_on(monkeypatch, grid) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    reason = record["refused_because"]
    assert reason.startswith(f"{RETIRED}={value!r} is set, and that variable is "
                             f"retired: its replacement is {SWITCH} (certified | "
                             "supported | any; unset means supported)"), reason
    meant = {"0": "certified", "1": "any"}.get(value)
    if meant is not None:
        assert f"{RETIRED}={value} is {SWITCH}={meant}." in reason, reason
    else:
        assert (f"{RETIRED} accepted only 0, which is {SWITCH}=certified, and 1, which "
                f"is {SWITCH}=any.") in reason, reason
    assert reason.endswith("so this run takes the array path"), reason
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"{RETIRED}={value!r} is set" in lines[0], lines
    assert _notes("\n".join(lines)) == []


# --- NVIDIA: what is supported ----------------------------------------------------------


NVRTC_86 = (35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86)
NVRTC_128 = (50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120)


@pytest.mark.parametrize("capability,archs,supported,compiles_for", [
    ((8, 6), NVRTC_118_ARCHS, True, "sm_86"), ((8, 9), NVRTC_118_ARCHS, True, "sm_89"),
    ((7, 0), NVRTC_118_ARCHS, True, "sm_70"), ((6, 1), NVRTC_118_ARCHS, True, "sm_61"),
    ((10, 0), NVRTC_118_ARCHS, False, None), ((12, 0), NVRTC_118_ARCHS, False, None),
    ((3, 0), NVRTC_118_ARCHS, False, None),
    ((10, 0), NVRTC_128, True, "sm_100"), ((12, 0), NVRTC_128, True, "sm_120"),
    # A binary for X.y loads on X.z, z >= y: the newest listed target of the same major.
    ((8, 9), NVRTC_86, True, "sm_86"), ((9, 0), NVRTC_86, False, None),
    ("86", NVRTC_118_ARCHS, True, "sm_86"), ("120", NVRTC_118_ARCHS, False, None)])
def test_the_hand_cuda_table_supports_what_its_nvrtc_can_target(capability, archs,
                                                                 supported, compiles_for):
    """Support is a self-check since 1.0.1, not the range 7.0 to 9.0: a device the
    installed NVRTC can produce a loadable binary for."""
    grid = cupy_like_grid_with_device(capability=capability if isinstance(
        capability, tuple) else (int(capability[:-1]), int(capability[-1])),
        nvrtc_archs=archs)
    target = fastpath_cuda.nvrtc_target(grid.xp, capability)
    assert target["supported"] is supported, target
    assert target.get("compiles_for") == compiles_for, target
    assert target["nvrtc"] == "11.8" and target["nvrtc_targets"] == list(archs), target


def test_the_hand_cuda_self_check_is_unknown_when_it_has_nothing_to_read():
    assert fastpath_cuda.nvrtc_target(cupy_like_grid_with_device().xp, None)[
        "supported"] is None
    target = fastpath_cuda.nvrtc_target(cupy_like_grid_with_device(nvrtc=None).xp, "8.6")
    assert target["supported"] is None
    assert target["because"].startswith("this CuPy's NVRTC would not say what it "
                                        "targets"), target


@pytest.mark.parametrize("version,supported", [
    ("3.1.0", True), ("3.1.1", True), ("3.1.0+git1234", True), ("3.2.0", True),
    ("3.3.1", True), ("9.9.9-unreleased", True), ("3.0.0", False), ("2.3.1", False),
    ("unknown", None), ("", None)])
def test_the_triton_table_supports_triton_3_1_and_later(version, supported):
    """``>=3.1``, open above: a later release is checked rather than refused. A version
    with no leading ``major.minor`` cannot be placed, so it is unknown."""
    assert fastpath.triton_version_supported(version) is supported


def test_the_supported_text_names_both_self_checks():
    text = fastpath.nvidia_supported_text()
    assert "Triton >=3.1 that compiled and launched the health probe" in text, text
    assert "a CuPy whose NVRTC can target this device's architecture" in text, text
    assert "7.0 to 9.0" not in text and "<3.2" not in text, text


# --- NVIDIA: a supported compute capability no gate ran on ------------------------------


@pytest.mark.parametrize("capability,spelled", SUPPORTED_UNCERTIFIED)
@pytest.mark.parametrize("value", [None, "supported", "any"])
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
    # What each table certifies, derived from its ledger rather than typed, so a later
    # certification of another architecture does not have to come and edit this line.
    certified = {"triton": fastpath.validated_compute_capabilities(),
                 "cuda": fastpath_cuda.validated_compute_capabilities()}
    for entry in served:
        assert entry["what"] == "GPU compute capability"
        assert entry["read"] == spelled
        assert entry["certified"] == list(certified[entry["table"]]), entry
        assert entry["supported"] is True, entry
        assert entry["because"] == _supported_entry_because(
            "GPU compute capability", spelled, entry["table"],
            _capability_scope(entry["table"], spelled)), entry
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
    # The owner's wording, where it is exact: the GPU is supported and not certified,
    # and the Triton version beside it (certified) is supported too.
    assert (". They were dispatched because this NVIDIA GPU and toolchain are "
            "supported but not certified bit-identical") in note, note
    assert f". They were dispatched because {_supported_clause('NVIDIA GPU and toolchain are')}; " \
        in note, note
    assert f"{SWITCH}=certified would restrict the kernels to certified ones" in note, note
    assert note.count("supported means") == 1, note
    assert ("counted in the dispatch record under uncertified.served[], which names "
            "the first five in welds_without_a_live_run_here") in note, note
    assert note.endswith(NOTE_TAIL), note
    assert f"because {SWITCH}=any" not in note, note
    assert RETIRED not in note, note

    # A second freeze in the same process repeats neither line.
    assert _plan_on(monkeypatch, grid) is not None
    assert _meep_lines(capsys.readouterr().err) == []


@pytest.mark.parametrize("capability,spelled", SUPPORTED_UNCERTIFIED)
def test_certified_only_refuses_a_supported_uncertified_device_by_name(
        monkeypatch, capsys, capability, spelled):
    monkeypatch.setenv(SWITCH, "certified")
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
        assert row["refused_because"].endswith(fastpath.CERTIFIED_SCOPE_TAIL), row
        assert row["welds_without_a_live_run_here"], row
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"{SWITCH}=certified restricts" in lines[0], lines
    assert _notes("\n".join(lines)) == []


# --- NVIDIA: a compute capability neither table can run on ------------------------------


@pytest.mark.parametrize("capability,spelled", UNSUPPORTED)
@pytest.mark.parametrize("value", [None, "supported", "certified"])
def test_an_unsupported_device_is_refused_by_name(monkeypatch, capsys, capability,
                                                 spelled, value):
    """A device the installed NVRTC cannot target and the installed Triton cannot launch
    on: the hand-CUDA table is refused by name at rung 4b -- ``supported`` (the default)
    names the way past it (``any``), ``certified`` names the restriction instead -- and
    the Triton table is dropped by its health probe at rung 4, on every scope."""
    _set(monkeypatch, value)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    assert _plan_on(monkeypatch, grid, probe=failing_triton_probe([capability])) is None
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    assert record["certified"] is None
    assert record["uncertified"]["admitted"] == []
    default = value != "certified"
    tail = fastpath.UNSUPPORTED_HINT if default else fastpath.CERTIFIED_SCOPE_TAIL
    cuda = record["tables"]["cuda"]
    assert cuda["candidate"] is False
    assert cuda["refused_because"].startswith(
        f"GPU compute capability {spelled} is not in the recorded"), cuda
    assert cuda["refused_because"].endswith(tail), cuda
    if default:
        assert (f"outside what that table can run on: {_nvrtc_outside(spelled)}"
                in cuda["refused_because"]), cuda
    triton = record["tables"]["triton"]
    assert triton["candidate"] is False
    assert triton["refused_because"].startswith(
        "Triton 3.1.0 imports but cannot compile and launch a kernel on this device "
        "(RuntimeError: ptxas fatal"), triton
    assert record["environment"]["device_supported"] is False
    assert record["environment"]["device_supported_by_table"] == {"triton": False,
                                                                  "cuda": False}
    assert record["environment"]["device_certified"] is False
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert (f"set {SWITCH}=any" in lines[0]) is default, lines
    assert _notes("\n".join(lines)) == []


@pytest.mark.parametrize("capability,spelled", UNSUPPORTED)
def test_an_unsupported_device_runs_under_any_recorded_as_unsupported(
        monkeypatch, capsys, capability, spelled):
    """``any`` runs the hand-CUDA table on a device its NVRTC cannot target (the Triton
    table is not consulted here: a Triton that cannot launch is dropped on any scope)."""
    monkeypatch.setenv(SWITCH, "any")
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "cuda")
    stub_triton(monkeypatch)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    install_cuda_composer(monkeypatch, cuda_step_plan())
    grid = certified_envelope_grid()
    grid.xp.cuda = cupy_like_grid_with_device(capability=capability,
                                              name=b"another device").xp.cuda
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["uncertified"]["unsupported_allowed"] is True
    (entry,) = record["uncertified"]["served"]
    assert (entry["table"], entry["read"], entry["supported"]) == ("cuda", spelled, False)
    assert entry["because"] == _unsupported_entry_because(
        "GPU compute capability", spelled, "cuda", _nvrtc_outside(spelled)), entry
    lines = _meep_lines(capsys.readouterr().err)
    status, notes = _status("\n".join(lines)), _notes("\n".join(lines))
    assert len(status) == 1 and status[0].endswith("another device UNCERTIFIED"), lines
    assert "supported," not in status[0], status
    assert len(notes) == 1, lines
    assert f"because {_unsupported_clause('NVIDIA GPU', with_range=True)}; " in notes[0], \
        notes
    assert "supported but not certified" not in notes[0], notes
    assert notes[0].endswith(NOTE_TAIL), notes


# --- NVIDIA: the certified device, and one that cannot be read --------------------------


def test_the_certified_device_is_certified_and_said_the_same_way_whatever_the_switch(
        monkeypatch, capsys):
    """Nothing on a certified host depends on the scope: decision, slots, line."""
    seen = {}
    for value in (None, "certified", "supported", "any"):
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
    assert seen[None] == seen["certified"] == seen["supported"] == seen["any"], seen
    assert seen[None][1][0].endswith("NVIDIA RTX A6000 certified"), seen


@pytest.mark.parametrize("value", [None, "supported", "any"])
def test_an_unreadable_device_is_neither_certified_nor_admitted(monkeypatch, capsys,
                                                                value):
    """Three-valued: an identity that was not read is not judged, under every scope
    but ``certified``."""
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
    """``certified`` runs the kernels only where they are certified, and an identity
    that was not read is not certified (the Metal table's rule, rung 4M)."""
    monkeypatch.setenv(SWITCH, "certified")
    assert _plan_on(monkeypatch, cupy_like_grid()) is None
    record = fastpath.last_dispatch_report()
    for table in ("triton", "cuda"):
        reason = record["tables"][table]["refused_because"]
        assert reason.startswith("the GPU compute capability could not be read on this "
                                 "host"), reason
        assert reason.endswith(fastpath.CERTIFIED_SCOPE_TAIL + ", and an identity that "
                               "cannot be judged is not one"), reason
    assert _notes(capsys.readouterr().err) == []


@pytest.mark.parametrize("value", [None, "certified"])
def test_a_ledger_that_cannot_be_read_leaves_the_device_unjudged(monkeypatch, value):
    """The default keeps the table a candidate, unjudged, as before; ``certified``
    refuses it by name."""
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


@pytest.mark.parametrize("value", [None, "supported", "certified"])
def test_an_unsupported_triton_is_refused_by_name(monkeypatch, value):
    """Older than 3.1: the release the kernels were written against is the floor."""
    _set(monkeypatch, value)
    block_cuda(monkeypatch)
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(),
                    version="3.0.0") is None
    record = fastpath.last_dispatch_report()
    row = record["tables"]["triton"]["refused_because"]
    assert row.startswith("Triton 3.0.0 is not in the recorded "
                          "validated_triton_versions"), row
    if value != "certified":
        assert ("outside the Triton versions the triton table supports "
                "(>=3.1)") in row, row
        assert row.endswith(fastpath.UNSUPPORTED_HINT), row
    else:
        assert row.endswith(fastpath.CERTIFIED_SCOPE_TAIL), row
    assert record["environment"]["triton_supported"] is False
    assert record["certified"] is None


def test_an_unsupported_triton_runs_under_any(monkeypatch, capsys):
    monkeypatch.setenv(SWITCH, "any")
    block_cuda(monkeypatch)
    plan = _plan_on(monkeypatch, cupy_like_grid_with_device(), version="3.0.0")
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["tables"]["triton"] == {"candidate": True, "refused_because": None}
    assert record["uncertified"]["served"] == [{
        "table": "triton", "what": "Triton", "read": "3.0.0",
        "certified": list(fastpath.validated_triton_versions()),
        "because": _unsupported_entry_because("Triton", "3.0.0", "triton",
                                              TRITON_SCOPE),
        "supported": False}]
    assert record["environment"]["triton"] == "3.0.0"
    assert record["environment"]["triton_certified"] is False
    assert record["environment"]["device_certified"] is True
    notes = _notes(capsys.readouterr().err)
    assert len(notes) == 1, notes
    certified = ", ".join(fastpath.validated_triton_versions())
    assert f"Triton 3.0.0 (certified: {certified})" in notes[0], notes
    assert (f"because {_unsupported_clause('Triton version', with_range=True)}; "
            in notes[0]), notes
    assert "NVIDIA GPU" not in notes[0], notes


@pytest.mark.parametrize("version", ["3.1.1", "3.2.0"])
@pytest.mark.parametrize("value", [None, "supported", "any"])
def test_a_supported_uncertified_triton_dispatches_by_default(monkeypatch, capsys, value,
                                                              version):
    _set(monkeypatch, value)
    block_cuda(monkeypatch)
    plan = _plan_on(monkeypatch, cupy_like_grid_with_device(), version=version)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["uncertified"]["served"] == [{
        "table": "triton", "what": "Triton", "read": version,
        "certified": list(fastpath.validated_triton_versions()),
        "because": _supported_entry_because("Triton", version, "triton", TRITON_SCOPE),
        "supported": True}]
    assert record["environment"]["triton_supported"] is True
    assert record["environment"]["triton_health"]["healthy"] is True
    lines = _meep_lines(capsys.readouterr().err)
    status = _status("\n".join(lines))
    assert len(status) == 1 and f"triton {version} supported, UNCERTIFIED" in status[0], \
        lines
    notes = _notes("\n".join(lines))
    assert len(notes) == 1 and f"Triton {version} (certified:" in notes[0], notes
    assert "welds_without_a_live_run_here" not in notes[0], notes
    # The device is certified here, so the NOTE names the Triton version alone and
    # never calls the GPU uncertified (the status line says it is certified).
    assert f"because {_supported_clause('Triton version is')}; " in notes[0], notes
    assert "NVIDIA GPU" not in notes[0], notes
    assert status[0].endswith("NVIDIA RTX A6000 certified"), status


def test_certified_only_refuses_a_supported_uncertified_triton(monkeypatch):
    monkeypatch.setenv(SWITCH, "certified")
    block_cuda(monkeypatch)
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(), version="3.1.1") is None
    row = fastpath.last_dispatch_report()["tables"]["triton"]["refused_because"]
    assert row.startswith("Triton 3.1.1 is not in the recorded "
                          "validated_triton_versions"), row
    assert row.endswith(fastpath.CERTIFIED_SCOPE_TAIL), row


def test_a_table_admitted_on_one_identity_and_refused_on_another_is_dropped_whole(
        monkeypatch, capsys):
    """A supported Triton that fails its health probe on this device: the version
    admission moves to ``dropped`` with the probe's refusal, so the NOTE never names
    what did not run."""
    block_cuda(monkeypatch)
    (capability, spelled), = UNSUPPORTED[:1]
    assert _plan_on(monkeypatch, cupy_like_grid_with_device(capability=capability),
                    version="3.1.1", probe=failing_triton_probe([capability])) is None
    record = fastpath.last_dispatch_report()
    assert record["uncertified"]["admitted"] == []
    (dropped,) = record["uncertified"]["dropped"]
    assert (dropped["table"], dropped["what"], dropped["read"]) == (
        "triton", "Triton", "3.1.1")
    assert dropped["dropped_because"].startswith(
        "Triton 3.1.1 imports but cannot compile and launch a kernel"), dropped
    assert _notes(capsys.readouterr().err) == []


# --- NVIDIA: a supported table composes beside a certified one (no rung 4e) -------------


def _envelope_grid_with_device(capability):
    """The configuration the CUDA release admits, on a device that reports ``capability``."""
    grid = certified_envelope_grid()
    grid.xp.cuda = cupy_like_grid_with_device(capability=capability).xp.cuda
    return grid


def _one_table_certifies_9_0(monkeypatch, table):
    """The shipped ledgers, with compute capability 9.0 certified on ``table`` only."""
    other = "triton" if table == fastpath.CUDA_TABLE else fastpath.CUDA_TABLE
    ledgers = {f"{table}_kernels": ledger_also_certified_on(table, "9.0"),
               f"{other}_kernels": copy.deepcopy(
                   dict(fastpath._fingerprints(other)))}  # noqa: SLF001
    # The bind of 9.0 records its certified stack in the Triton ledger whichever table
    # it binds.
    record_certified_stack(ledgers["triton_kernels"], "9.0")
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
def test_certified_only_runs_the_certified_table_alone(monkeypatch, capsys, certified,
                                                       slots, dropped):
    """``certified`` refuses the uncertified table at rung 4b, so that scope runs the
    certified table alone, certified, as it did when rung 4e existed."""
    monkeypatch.setenv(SWITCH, "certified")
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
    assert row["refused_because"].endswith(fastpath.CERTIFIED_SCOPE_TAIL), row
    assert "dropped_for_a_certified_table" not in row, row
    lines = _meep_lines(capsys.readouterr().err)
    assert _notes("\n".join(lines)) == [], lines
    (status,) = _status("\n".join(lines))
    assert status.endswith("NVIDIA RTX A6000 certified"), status


@pytest.mark.parametrize("certified,slots,dropped", OUTRANKING)
@pytest.mark.parametrize("value", [None, "supported", "any"])
def test_the_supported_table_composes_beside_the_certified_one(
        monkeypatch, capsys, certified, slots, dropped, value):
    """A certification record no longer decides a slower composition. By
    default (and under ``any``) the supported table composes beside the certified one,
    the run is uncertified on it, and the NOTE names the table it is about."""
    _set(monkeypatch, value)
    _one_table_certifies_9_0(monkeypatch, certified)
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert sorted(record["composition"]["tables_dispatched"]) == ["cuda", "triton"]
    assert sorted(plan.slots) == ["step_B", "step_D", "update_E"]
    assert record["tables"][dropped] == {"candidate": True, "refused_because": None}
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
@pytest.mark.parametrize("value", [None, "any"])
def test_a_preference_for_the_uncertified_table_is_honoured_uncertified(
        monkeypatch, certified, slots, dropped, value):
    """Both tables are candidates, so a preference for either orders them; the run is
    uncertified on the table that is not certified here."""
    _set(monkeypatch, value)
    _one_table_certifies_9_0(monkeypatch, certified)
    monkeypatch.setenv(fastpath.BACKEND_PREFERENCE_SWITCH, dropped)
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["arbitration"]["effective_order"] == [dropped, certified]
    assert plan.report()["certified"] is False


def test_a_preference_for_the_certified_table_puts_it_first(monkeypatch):
    _one_table_certifies_9_0(monkeypatch, "cuda")
    monkeypatch.setenv(fastpath.BACKEND_PREFERENCE_SWITCH, "cuda")
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["arbitration"]["effective_order"] == ["cuda", "triton"]
    # The supported Triton table fills the slot the hand-CUDA table left free, so the
    # run is uncertified on it.
    assert plan.report()["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert plan.report()["certified"] is False


def test_the_kernel_table_switch_naming_the_uncertified_table_runs_it_uncertified(
        monkeypatch, capsys):
    """Rung 3 leaves the other table unconsulted: the table asked for runs, with the
    NOTE."""
    _one_table_certifies_9_0(monkeypatch, "cuda")
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "triton")
    plan = _two_table_plan(monkeypatch)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["composition"]["tables_dispatched"] == ["triton"]
    assert len(_notes(capsys.readouterr().err)) == 1


def test_an_unreadable_table_composes_beside_a_certified_one(monkeypatch):
    """An unjudged table was admitted by nothing and composes beside the certified
    one; the verdict is unknown."""
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
    ((7, 5), "3.0.0", "NVIDIA GPU is", "Triton version")])
def test_under_any_a_mixed_host_says_which_identity_is_supported(
        monkeypatch, capsys, capability, version, supported, outside):
    """One identity supported, the other not: the NOTE calls supported only the one that
    is, names the other as what ``any`` ran, and states what supported means once. (A
    supported Triton on an unsupported GPU is no longer a Triton-table state: the GPU is
    supported for that table exactly when the health probe launches on it, and a Triton
    whose probe fails is dropped on every scope.)"""
    monkeypatch.setenv(SWITCH, "any")
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
    assert note.count("supported means") == 1, note
    assert "GPU and toolchain" not in note, note
    assert note.endswith(NOTE_TAIL), note


def test_a_hand_cuda_run_claims_no_toolchain(monkeypatch, capsys):
    """The hand-CUDA table reads no toolchain version, so on a supported, uncertified
    GPU its NOTE calls the GPU supported and says nothing about a toolchain."""
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "cuda")
    # 8.7 retires 8.9 as the uncertified example: 8.9 is certified now; no round ran on 8.7.
    plan = _two_table_plan(monkeypatch, capability=(8, 7))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["composition"]["tables_dispatched"] == ["cuda"]
    (entry,) = record["uncertified"]["served"]
    assert entry["because"] == _supported_entry_because(
        "GPU compute capability", "8.7", "cuda", _capability_scope("cuda", "8.7")), entry
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
    assert reason.endswith("The value is refused rather than interpreted, so this run "
                           "takes the array path"), reason
    for accepted in ("=certified", "=supported (the default when unset)", "=any"):
        assert f"{SWITCH}{accepted}" in reason, reason
    assert record["uncertified"]["value"] == value
    assert record["uncertified"]["allowed"] is False
    assert record["uncertified"]["unsupported_allowed"] is False
    assert record["certified"] is None
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and f"{SWITCH}={value!r}" in lines[0], lines


@pytest.mark.parametrize("value", [None, "any"])
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
SUPPORTED_REMINDER = (f"this Apple GPU is supported, and {SWITCH}=certified would "
                      f"restrict the Metal kernels to certified environments")
UNCERTIFIED_REMINDER = (f"the Metal table runs on environments outside the certified "
                        f"set unless {SWITCH}=certified")


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
    # One cited weld, holding one live run for the recorded architecture; a recorded
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
@pytest.mark.parametrize("switch", [None, "supported", "any"])
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
        assert report["uncertified"]["value"] == switch
        assert report["uncertified"]["allowed"] is True
        assert report["environment"]["certified_only"] is False
        field = np.array(driver.get_field("Ez"), copy=True)
        driver.close()
        assert field.tobytes() == certified_field.tobytes()
    lines = _meep_lines(capsys.readouterr().err)
    notes = _notes("\n".join(lines))
    assert len(notes) == 1, lines
    assert f"{what} {value} (certified: {wanted})" in notes[0], notes
    assert f"dispatched because {SUPPORTED_REMINDER};" in notes[0], notes
    assert f"{SWITCH}=any" not in notes[0] and RETIRED not in notes[0], notes
    status = [line for line in lines if line.startswith("meep_gpu: step path fused;")]
    assert len(status) == 1, lines
    # Supported is not certified: an Apple GPU no weld ran on is both, by name.
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
    monkeypatch.setenv(SWITCH, "certified")
    _environment(monkeypatch, fact, value)
    driver, report = _run()
    assert driver.active_step_path == "array"
    assert report["decision"] == "refused" and report["certified"] is None
    reason = report["refused_because"]
    expected = (f"{what}={value} compiles the Metal kernels in fast-math mode"
                if fact == "fast_math" else
                f"{what} {value} is not the one every Metal weld this table cites ran on")
    assert reason.startswith(expected), reason
    assert reason.endswith(f"{SWITCH}=certified restricts the Metal kernels to "
                           "certified environments"), reason
    assert report["environment"][mark] is False
    assert report["environment"]["certified_only"] is True
    driver.close()
    assert _notes(capsys.readouterr().err) == []


def test_certified_only_runs_a_certified_metal_environment(monkeypatch, capsys):
    _require_metal()
    monkeypatch.setenv(SWITCH, "certified")
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
@pytest.mark.parametrize("switch", [None, "supported", "any", "certified"])
def test_an_environment_that_cannot_be_judged_runs_unjudged_or_is_refused(
        monkeypatch, capsys, case, fact, value, recorded, cause, switch):
    """Unjudged is not uncertified: it runs by default (and under ``any``), and the
    ``certified`` scope refuses it."""
    _require_metal()
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    _environment(monkeypatch, fact, value, recorded)
    driver, report = _run()
    assert report["environment"]["device_certified"] is None
    if switch != "certified":
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


@pytest.mark.parametrize("value", RETIRED_VALUES)
def test_the_retired_name_is_refused_by_name_on_an_apple_gpu(monkeypatch, capsys, value):
    _require_metal()
    monkeypatch.setenv(RETIRED, value)
    driver = _driver()
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "array"
    assert report["refused_because"].startswith(
        f"{RETIRED}={value!r} is set, and that variable is retired: its replacement is "
        f"{SWITCH}"), report["refused_because"]
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert lines[0].startswith("meep_gpu: step path array on the host CPU; dispatch "
                               f"refused: {RETIRED}={value!r} is set"), lines
