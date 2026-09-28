"""The Triton probes' device-identity block, held to what the weld tools read.

WHY THIS IS ON THE LAPTOP MERGE BAR. ``parity/meep_gpu/triton_device_identity.py``
runs inside a device gate, on a shared box, at the end of a run that has already
cost GPU minutes — the worst possible place to discover that it raises. Every claim
its docstring makes is checked here instead: that it never imports a device stack,
that it never raises when there is no device, that it does not restate a fact the
probe already measured, and that the one value it DOES overwrite lands on the
spelling ``test_triton_weld_contract.py`` validates.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
MODULE_PATH = PARITY / "triton_device_identity.py"

#: The names that would make this module a device import. ``meep_gpu`` is on the
#: list for a different reason than the other three: the module must answer inside a
#: gate that refused, and the engine package is exactly what may have refused.
FORBIDDEN_IMPORT_ROOTS = {"numpy", "cupy", "triton", "meep_gpu"}


def _load():
    """Load by PATH, not by package name.

    ``parity/meep_gpu`` is not a package and is on ``sys.path`` only inside a probe
    run; importing by name here would test whichever ``triton_device_identity``
    happened to be importable rather than the file this tree ships.
    """
    spec = importlib.util.spec_from_file_location(
        "triton_device_identity_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_device_identity_answers_with_no_device_stack_loaded(monkeypatch):
    """No CuPy in ``sys.modules`` is the LAPTOP, and it must still record a machine.

    Both spellings of absence are exercised, because they reach different branches:
    a key that is gone, and a key present but ``None`` (which is what a partially
    torn-down import leaves behind, and what a naive truthiness check would trip on
    in the opposite direction).
    """
    identity = _load()
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    out = identity.device_identity()
    assert out["hostname"], "a record with no machine in it cannot be gone back to"
    assert out["cuda_visible_devices"] == "3"
    assert "device" not in out, (
        "a device was reported with no CuPy loaded — the module read something it "
        "cannot have measured")

    monkeypatch.setitem(sys.modules, "cupy", None)
    again = identity.device_identity()
    assert again["hostname"] == out["hostname"]
    assert "device" not in again


def test_record_does_not_restate_what_the_probe_already_measured():
    """The probe's ``environment()`` is authored evidence; this helper only fills gaps."""
    identity = _load()
    measured = {
        "hostname": "the GPU host",
        "device": "NVIDIA RTX A6000",
        "cupy": "13.5.1",
        "triton": "3.1.0",
    }
    out = identity.record(dict(measured))
    for key, value in measured.items():
        assert out[key] == value, (
            f"record() overwrote {key!r}, which the probe measured: "
            f"{value!r} -> {out[key]!r}")
    assert out is not measured
    assert "cuda_visible_devices" in out, "the key the GPU-placement claim rests on"


def test_record_normalises_the_capability_to_the_spelling_the_contract_validates():
    """``"86"`` -> ``"8.6"``: the ONE value ``record`` is allowed to overwrite.

    ``test_every_triton_weld_names_a_validated_triton_and_capability`` tests a
    SUBSTRING of the host line against ``validated_compute_capabilities``, so an
    artifact recorded in CuPy's ``"86"`` spelling would seed a weld that reports
    "names no validated cc" for a run that was on a validated architecture. The
    laptop copy of the rule is compared against the engine's own here, so the
    deliberate duplication in the module cannot drift without this going red.
    """
    identity = _load()
    assert identity.record({"compute_capability": "86"})["compute_capability"] == "8.6"
    assert identity.record({"compute_capability": "8.6"})["compute_capability"] == "8.6"
    assert identity.record({"compute_capability": (8, 6)})["compute_capability"] == "8.6"

    from meep_gpu import fastpath

    for value in ("86", "8.6", (8, 6), [9, 0], "90", "sm_86", ""):
        assert identity.normalized_capability(value) == \
            fastpath._normalized_capability(value), (
            f"the laptop copy of the capability rule has drifted from "
            f"meep_gpu.fastpath._normalized_capability on {value!r}")


def test_the_module_imports_nothing_that_needs_a_device():
    """Read off the AST, not off a successful import.

    An import that happens to succeed on this machine proves nothing about the gate
    host, and a device import buried in a function body is exactly the shape that
    would pass a smoke test and then allocate on a GPU mid-refusal. The check is
    over EVERY ``Import``/``ImportFrom`` in the file, at any depth.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import inside a non-package directory
                found.add(f"<relative level {node.level}>")
            elif node.module:
                found.add(node.module.split(".")[0])
    forbidden = sorted(found & FORBIDDEN_IMPORT_ROOTS)
    assert not forbidden, (
        f"{MODULE_PATH.name} imports {forbidden}; it must read sys.modules instead, "
        f"so it can answer inside a gate that has already refused")
    assert found <= {"__future__", "os", "socket", "sys", "typing"}, (
        f"unexpected imports {sorted(found)}: this module is stdlib only")
