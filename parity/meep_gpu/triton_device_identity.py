"""WHICH GPU produced this Triton artifact, recorded by the probe that ran on it.

WHY THIS EXISTS. A Triton weld is a claim about generated PTX, so the machine, the
device and its compute capability are part of the claim — ``test_triton_weld_contract
.py::test_every_triton_weld_names_a_validated_triton_and_capability`` binds every
weld's ``host`` string to the record's ``validated_triton_versions`` and
``validated_compute_capabilities``. But a weld may only be CUT from what a run
measured, never from what someone types, and measured 2026-09-09 across the fleet
campaign the probes' own ``environment()`` blocks answer only part of it: the four
probes re-gated this round record ``device``, ``cupy``, ``triton`` and
``cuda_visible_devices`` and NO ``hostname``, while the conductive probe records the
capability in CuPy's ``"86"`` spelling and the others in ``"8.6"``. So
``seed_triton_welds.py`` had to either refuse them or supply the missing half itself
— and supplying it is the forgery that tool exists to avoid. This module closes the
gap at the only honest end: the gate records the identity, the tools read it.

WHY IT IS TRITON-SIDE AND NOT IN ``gate_provenance.stamp()``. The campaign log's
2026-09-10 entry rules it: ``parity/meep_gpu/gate_provenance.py`` is pinned by 42 of
42 Metal gate-bound manifests and by two Triton ledger welds
(``triton_dispersive_fused_pair_device_gate`` and
``triton_fused_ade_state_device_gate``), so an edit there moves every Metal weld for
a Triton-only need. A separate module pinned by nothing costs one call site per
probe instead.

WHAT IT NEVER DOES. It never IMPORTS CuPy or Triton. It reads ``sys.modules`` —
which answers "did the process that is writing this artifact already have a device
open?" and cannot itself open one, cannot allocate, cannot change a device's state
and cannot fail on a laptop that has neither installed. Stdlib only (``os``,
``socket``, ``sys``), for the same reason ``gate_provenance`` is: it must answer
inside a gate that has already refused, and on the laptop that is the merge bar.

THE SIX KEYS THE TOOLS READ. ``seed_triton_welds.py`` and
``rebind_triton_welds._host_line`` compose one host spelling out of ``hostname``,
``device``, ``compute_capability``, ``cupy``, ``triton`` and
``cuda_visible_devices``; ``device_index`` is recorded beside them because
``CUDA_VISIBLE_DEVICES`` names the MASK and not the ordinal inside it. Adding a key
here without teaching those two to read it records a fact nothing consumes; dropping
one makes a seeded weld refuse.

Usage, at the probe's own environment site::

    payload["environment"] = triton_device_identity.record(environment(cp))
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
import socket
import sys
from typing import Any, Dict


def normalized_capability(value: Any) -> str:
    """``(8, 6)``, ``"86"``, ``"8.6"`` -> ``"8.6"``. One spelling, so a comparison means something.

    A DELIBERATE RE-STATEMENT of :func:`meep_gpu.fastpath._normalized_capability`
    rather than a call to it. This module is imported by a gate that may have
    already refused — possibly because ``meep_gpu`` itself would not import — and an
    identity block that cannot be written is worse than one written without the
    engine's help. The behaviour is pinned to the original by
    ``test_triton_device_identity.py``, which imports both and compares them, so the
    duplication cannot drift silently.

    CuPy hands the number out three ways: ``Device().compute_capability`` is the
    string ``"86"``, ``getDeviceProperties()`` gives ``major``/``minor`` ints, and
    every recorded gate wrote ``"8.6"``. The weld contract tests a SUBSTRING of the
    host line against ``validated_compute_capabilities`` (``["8.6"]``), so a weld
    seeded from an ``"86"`` artifact would report "names no validated cc" and refuse
    a run that was in fact on a validated architecture.
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{int(value[0])}.{int(value[1])}"
    text = str(value).strip()
    if text.isdigit() and len(text) >= 2:
        return f"{text[:-1]}.{text[-1]}"
    return text


def device_identity() -> Dict[str, Any]:
    """The machine and the device, read from modules the process ALREADY imported.

    ``sys.modules.get("cupy")`` rather than ``import cupy``: on a host with a device
    this returns the very module object the probe has been launching through, so the
    identity describes the device the run actually used; on the laptop it returns
    ``None`` and the device keys are simply absent. Never raises — an unreadable
    device is RECORDED under ``device_unreadable`` and the artifact still gets
    written, because losing a completed gate run to an identity read would be the
    expensive failure.
    """
    out: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            index = int(cupy.cuda.runtime.getDevice())
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            name = properties["name"]
            out["device"] = name.decode() if isinstance(name, bytes) else str(name)
            out["compute_capability"] = f"{properties['major']}.{properties['minor']}"
            out["device_index"] = index
            out["cupy"] = cupy.__version__
        except Exception as exc:  # noqa: BLE001 - an unreadable device is recorded, not raised
            out["device_unreadable"] = repr(exc)

    triton = sys.modules.get("triton")
    out["triton"] = getattr(triton, "__version__", None)
    return out


def record(environment: Dict[str, Any]) -> Dict[str, Any]:
    """Fill the identity keys a probe's own ``environment()`` left out, in place.

    ``setdefault`` on every key, so a probe that already answers one keeps ITS
    answer: those blocks were authored against what the probe measured and this
    helper is not entitled to restate them. The ONE overwrite is
    ``compute_capability``, normalised through :func:`normalized_capability` — the
    conductive probe records CuPy's ``"86"`` and the other three record ``"8.6"``,
    and two spellings of one architecture is what makes the weld contract's
    substring test decide a gate by which reader answered first.

    Returns the same dict it was handed, so the call reads as one expression at the
    probe's existing assignment site.
    """
    # HALF A MEASUREMENT IS NOT A MEASUREMENT. A probe whose own `environment()`
    # fell to its except branch records `cupy = "unavailable: ..."` and no device;
    # filling the device and the capability around that string would compose an
    # identity that reads as a complete one and is not. Where the probe reports its
    # toolchain unavailable, this helper adds only the host facts it can stand
    # behind and leaves the device keys absent, so the seed refuses the artifact by
    # name instead of welding it.
    identity = device_identity()
    toolchain = str(environment.get("cupy") or environment.get("triton") or "")
    if "unavailable" in toolchain or "Error" in toolchain:
        identity = {key: value for key, value in identity.items()
                    if key in ("hostname", "cuda_visible_devices")}
    for key, value in identity.items():
        environment.setdefault(key, value)
    if "compute_capability" in environment:
        environment["compute_capability"] = normalized_capability(
            environment["compute_capability"])
    return environment
