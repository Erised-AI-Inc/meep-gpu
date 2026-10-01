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

WHY IT ALSO ANSWERS "WHICH CAPABILITY DID THIS RUN MEASURE?". A certification record
is now keyed by compute capability (one ``runs`` slot per capability), so every
writer has to READ the capability off the run rather than type it, and refuse when
the run does not say. :func:`run_capability` is that single reader: four sources in
one order, agreement required, no guess. It is here rather than in a writer because
five writers need the same answer and five copies of a precedence rule drift.

WHY IT HAS A ``--write`` MODE AND WHAT THAT COSTS. Twelve of the thirty-seven cited
fleet welds, every family gate and the composition probes stamp NO capability of
their own — the multi-capability design's census of the 2026-09-25 fleet, not a
count taken here — so a round that only ran those gates has no evidence of the
architecture it ran on. ``--write DIR`` closes that by recording
``DIR/device.json`` from the live host in the run's own environment, which
:func:`run_capability` then reads as a source of last resort. It is the ONE place in
this module that loads a device stack, and it loads it through
``importlib.import_module`` INSIDE the ``--write`` path: the library functions above
must still answer inside a gate that has already refused, so no import statement in
this file may name ``cupy``, ``triton``, ``numpy`` or ``meep_gpu``. A separate
process with the same ``CUDA_VISIBLE_DEVICES`` is weaker evidence than an in-process
stamp, and the standing plan is to stamp each producer through :func:`record` at its
next edit; this is what makes a round certifiable before that work lands.

Usage, at the probe's own environment site::

    payload["environment"] = triton_device_identity.record(environment(cp))

Usage, from a campaign driver, in the environment its gates run in::

    python parity/meep_gpu/triton_device_identity.py --write <run dir>
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import json
import os
import socket
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

#: What ``--write`` returns when it cannot record a capability. The same number
#: ``drive_triton_weld_gates.EXIT_CANNOT_CERTIFY`` uses, deliberately: the driver
#: propagates this child's refusal as its own, and two numbers for one outcome is
#: how a campaign log stops being readable.
EXIT_CANNOT_RECORD = 75


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


# ---------------------------------------------------------------------------
# Which capability did this run measure?
# ---------------------------------------------------------------------------

#: The identity keys the weld tools read, and every spelling a recorded run has
#: been measured to use for them. ``hostname``/``host`` and ``device``/
#: ``device_name`` are the 2026-09-09 fleet's two authored spellings (see
#: ``rebind_triton_welds._host_line``); ``cc`` is the short form a probe's own
#: ``environment()`` may write. ONE spelling comes back out, because a caller that
#: has to try two is a caller that will one day try only one.
_SPELLINGS: Dict[str, Tuple[str, ...]] = {
    "hostname": ("hostname", "host"),
    "device": ("device", "device_name"),
    "compute_capability": ("compute_capability", "cc"),
    "device_index": ("device_index",),
    # ``cupy_version``/``triton_version`` are what ``probe_fused_kernel_bit_identity``
    # writes, and ``CUDA_VISIBLE_DEVICES`` is the spelling the gates carry; a reader
    # that knew only the short lowercase forms saw no identity in those blocks.
    "cupy": ("cupy", "cupy_version"),
    "triton": ("triton", "triton_version"),
    "cuda_visible_devices": ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES"),
}


def _device_block(payload: Dict[str, Any], *path: str) -> Optional[Dict[str, Any]]:
    """One nested dict of a payload, or ``None`` when the path does not reach one."""
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _stated(block: Dict[str, Any], key: str) -> Any:
    for spelling in _SPELLINGS[key]:
        value = block.get(spelling)
        if value is not None and value != "":
            return value
    return None


def _is_normalised(capability: str) -> bool:
    """``8.6`` yes, ``86`` / ``sm_86`` / ``8.6.0`` / ``0.6`` no.

    The same shape ``meep_gpu.fastpath`` accepts as a ``runs`` key, re-stated
    without ``re`` for the reason the whole module is stdlib-thin. Checked HERE so
    an unparseable spelling is refused where the run can still be named, not three
    tools later as a record error with no run in hand.
    """
    major, dot, minor = capability.partition(".")
    return bool(dot) and major.isdigit() and major[:1] != "0" \
        and len(minor) == 1 and minor.isdigit()


def run_capability(payload: Dict[str, Any], dirs: Iterable[Any] = ()
                   ) -> Tuple[Optional[str], Any]:
    """The capability THIS RUN measured, with the identity it was read from.

    ``(capability, identity)`` when the run says, and ``(None, reason)`` when it
    does not — never a guess and never a default. A certification slot is keyed by
    capability, so a writer that typed the key would be welding a claim about an
    architecture no measurement names; the writers call this and SKIP the entry by
    the reason it returns.

    FOUR SOURCES, ONE ORDER, AGREEMENT REQUIRED:

    1. ``environment.compute_capability|cc`` — the probe's own block, which
       :func:`record` normalises. The strongest: measured in the process that
       launched the kernels.
    2. ``provenance.device.compute_capability`` — ``gate_provenance``'s stamp, the
       key ``recut_driver_dispatch_record.stamped_capabilities`` already reads.
    3. ``device_info.compute_capability`` — a payload that keeps its self-stamp
       under its own key rather than under ``environment``. Measured 2026-09-30: no
       artifact in the tree writes this key, and it is read because the spec names
       it as the self-stamp shape, so a probe that adopts it is not read as
       unmeasured.
    4. ``<dir>/device.json`` for each of ``dirs``, in the order given — the
       ``--write`` record. Last because it comes from a SEPARATE process that
       shared only ``CUDA_VISIBLE_DEVICES``; it is what makes a round whose gates
       stamp nothing certifiable at all.

    EVERY SOURCE THAT ANSWERS MUST AGREE, and so must every hostname that is
    stated. One run happened on one machine on one architecture: two answers mean
    the directory pools artifacts from two runs, and binding either value would
    certify a card that never ran these bytes. An unreadable ``device.json`` is
    refused rather than skipped — it is the only evidence a gate that stamps
    nothing has, and skipping it would silently fall back to a weaker source.
    """
    blocks: list = []
    for label, path in (("environment", ("environment",)),
                        ("provenance.device", ("provenance", "device")),
                        ("device_info", ("device_info",))):
        block = _device_block(payload, *path)
        if block is not None:
            blocks.append((label, block))

    looked_in = []
    for directory in dirs:
        path = os.path.join(str(directory), "device.json")
        looked_in.append(path)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                written = json.loads(handle.read())
        except (OSError, ValueError) as exc:
            return None, (f"{path} is unreadable ({type(exc).__name__}: {exc}); it "
                          f"is the capability evidence for a run whose gates stamp "
                          f"none, and reading past it would bind the run from a "
                          f"weaker source")
        if not isinstance(written, dict):
            return None, (f"{path} holds {type(written).__name__}, not an object: "
                          f"nothing in it can be read as a device identity")
        blocks.append((path, written))

    answers: Dict[str, str] = {}
    hostnames: Dict[str, str] = {}
    identity: Dict[str, Any] = {}
    for label, block in blocks:
        stated = _stated(block, "compute_capability")
        if stated is not None:
            answers[label] = normalized_capability(stated)
        host = _stated(block, "hostname")
        if host is not None:
            hostnames[label] = str(host)
        for key in _SPELLINGS:
            value = _stated(block, key)
            if value is not None and key not in identity:
                identity[key] = value
        # THE PLACEMENT CLAIM, ONE LEVEL DOWN. Some probes write the mask inside
        # their own ``env`` sub-block rather than beside the device keys (measured
        # across the 2026-09-09 fleet; ``rebind_triton_welds._host_line`` reads both),
        # and it is the part of a host line that makes "it ran alone on GPU 6"
        # checkable. A nested path is not a spelling, so it is read here.
        if "cuda_visible_devices" not in identity:
            nested = (block.get("env") if isinstance(block.get("env"), dict)
                      else {}).get("CUDA_VISIBLE_DEVICES")
            if nested is not None:
                identity["cuda_visible_devices"] = nested

    if not answers:
        return None, (
            "no source states a compute capability: environment.compute_capability"
            "|cc, provenance.device.compute_capability, device_info."
            "compute_capability"
            + (f" and device.json in {looked_in}" if looked_in
               else " and no directory was offered for a device.json")
            + ". A run that cannot name the architecture it ran on cannot be bound "
              "to a capability record")

    spelled = {label: value for label, value in sorted(answers.items())}
    distinct = sorted(set(answers.values()))
    if len(distinct) > 1:
        return None, (f"the sources disagree about the compute capability "
                      f"({spelled}): one run ran on one architecture, so this is a "
                      f"pooled or mixed-device artifact and not a capability")
    capability = distinct[0]
    if not _is_normalised(capability):
        return None, (f"the compute capability reads {capability!r} ({spelled}), "
                      f"which is not the MAJOR.MINOR spelling a capability record "
                      f"is keyed by")

    machines = sorted(set(hostnames.values()))
    if len(machines) > 1:
        return None, (f"the sources name different machines ({hostnames}): these "
                      f"device facts cannot all describe one run")

    identity["compute_capability"] = capability
    #: WHICH source answered, carried beside the answer. A weld bound from a
    #: separate-process ``device.json`` rests on weaker evidence than one bound
    #: from the launching process's own block, and a reader must be able to tell
    #: them apart without re-opening the run.
    identity["capability_sources"] = spelled
    return capability, identity


# ---------------------------------------------------------------------------
# --write: the run's own device, recorded by a process that opens it
# ---------------------------------------------------------------------------

def write_device_json(directory: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Record ``<directory>/device.json`` from the LIVE host, or refuse by name.

    ``(payload, None)`` on success and ``(None, reason)`` otherwise. Nothing is
    written on a refusal: a device.json that names no capability is worse than none
    at all, because :func:`run_capability` would read it as a source that answered.

    THE ONE PLACE THIS MODULE OPENS A DEVICE, and it does so through
    ``importlib.import_module`` rather than an ``import`` statement, because every
    other function here must answer inside a gate that has already refused — the
    module's own merge-bar test reads the import list off the AST and no entry in it
    may name ``cupy``, ``triton``, ``numpy`` or ``meep_gpu``. Loading the two
    modules is all this does: :func:`device_identity` then reads them out of
    ``sys.modules`` exactly as it does inside a probe, so the CLI and the probes
    record one shape by construction rather than by two code paths agreeing.

    A MISSING TRITON IS RECORDED, NOT REFUSED. A CUDA-table round has no Triton to
    report, and the rule that a run's toolchain must be a validated one belongs at
    the rebind, where the validated list lives. What this refuses is the one fact
    the record is KEYED by: a capability it could not read.
    """
    import importlib  # noqa: PLC0415 - see above; the device stack loads here only

    unavailable: Dict[str, str] = {}
    for name in ("cupy", "triton"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported in the refusal, not raised
            unavailable[name] = f"{type(exc).__name__}: {exc}"

    identity = device_identity()
    if "device_unreadable" in identity:
        return None, (f"the device is unreadable on {identity['hostname']}: "
                      f"{identity['device_unreadable']} (CUDA_VISIBLE_DEVICES="
                      f"{identity.get('cuda_visible_devices')!r})")
    if not identity.get("compute_capability"):
        return None, (f"no compute capability on {identity['hostname']}: CuPy is "
                      f"what reports it and it is "
                      f"{unavailable.get('cupy', 'loaded but silent')} "
                      f"(CUDA_VISIBLE_DEVICES="
                      f"{identity.get('cuda_visible_devices')!r})")

    payload = dict(identity)
    payload["compute_capability"] = normalized_capability(
        payload["compute_capability"])
    payload["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    path = os.path.join(str(directory), "device.json")
    # A RUN ROOT RE-USED ON ANOTHER CARD IS THE HAZARD THIS WHOLE RECORD EXISTS TO
    # CATCH. Re-recording the same device is a resumed campaign and is allowed; a
    # file naming a different card or machine is evidence from another run, and
    # replacing it would hand the round's gates a capability they did not run on.
    if os.path.isfile(path):
        standing, standing_identity = run_capability({}, [str(directory)])
        if standing is None:
            return None, (f"{path} exists and cannot be read as a device identity "
                          f"({standing_identity}); it is another run's evidence "
                          f"until someone decides otherwise, and this tool does not "
                          f"replace it")
        disagree = []
        if standing != payload["compute_capability"]:
            disagree.append(f"cc {standing} against this host's cc "
                            f"{payload['compute_capability']}")
        recorded_host = standing_identity.get("hostname")
        if recorded_host is not None and identity.get("hostname") is not None \
                and str(recorded_host) != str(identity["hostname"]):
            disagree.append(f"machine {recorded_host!r} against {identity['hostname']!r}")
        if disagree:
            return None, (f"{path} already records {'; '.join(disagree)}: two cards "
                          f"in one run root, and the gates' artifacts here cannot be "
                          f"keyed by both")

    os.makedirs(str(directory), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, None


def main(argv: Sequence[str]) -> int:
    """``--write DIR``, and nothing else.

    HAND-PARSED rather than through ``argparse``: this module's import list is
    pinned by its own merge-bar test, so every stdlib name it adds is a name that
    test has to carry, and a single two-token flag does not need a parser. Anything
    else is refused by name rather than ignored.
    """
    if len(argv) == 2 and argv[0] == "--write":
        payload, reason = write_device_json(argv[1])
        if reason is not None:
            print(f"REFUSED: {reason}", flush=True)
            return EXIT_CANNOT_RECORD
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0
    print(f"REFUSED: {list(argv)} is not an invocation of this tool.\n"
          f"usage: triton_device_identity.py --write DIR\n"
          f"  writes DIR/device.json: the machine, the device, its compute "
          f"capability and the toolchain, as the LIVE host reports them. Run it in "
          f"the environment the campaign's gates run in, so CUDA_VISIBLE_DEVICES "
          f"names the same card.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
