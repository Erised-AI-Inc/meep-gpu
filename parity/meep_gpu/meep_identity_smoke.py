#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""MEEP-only digest smoke of an identity reference: four cases, every stored word hashed.

    python parity/meep_gpu/meep_identity_smoke.py [--mode keep] [--record PATH] [--rows OUT]
    python parity/meep_gpu/meep_identity_smoke.py --mint --evidence DIR --build TAG --out PATH

WHAT IT ANSWERS. Does the MEEP this interpreter imports, stepped on its own, store the
same float32 words as the identity reference that minted the record? A reference build
is identified by more than its flags and its census (``tools/fma_census.py``): its own
words on known cases must reproduce. This is that check, and it is MEEP alone, so it
runs in seconds, before anything else of a session that compares against MEEP.

THE CASES are the four of the 2026-10-05 bit-identity probe, all real-valued,
diagonal-epsilon and lift-accepted: (a) 2-D vacuum, PML(1.0), a Gaussian Ez point
source; (b) (a) with an epsilon-12 block two cells from the source; (c) 3-D, PML(0.5),
two blocks, two mirror symmetries, a Gaussian Hx source; (c') (c) with fwidth 1.0.
The steps compared are a named schedule (``SCHEDULE``): 0, 1, 2, 5, 10, 20 and, for
(a), (b) and (c), the last step of Table 2's step rule (N = 200, 200, 375), past the
step (57-64 in (a)) where the fields reach the subnormal range and a flushing MEEP
leaves the keep reference.

THE DIGEST of one (case, step, component) is the SHA-256 of the component's words as
little-endian float32, in the order of the recorded lattice (x, then y, then z, each
axis ``first + 2 i`` for ``i < count`` in MEEP's ivec units: the package's stored
cells). A word is read with ``fields.get_field(component, ivec)``, which returns the
stored float32 times the symmetry phase, so a ghost cell of a mirrored axis reads as
MEEP's own phase times its owned word; the real part is cast back to float32 and the
cast is checked exact (no rounding, no imaginary part), refused otherwise. MEEP is
stepped with ``sim.run(until=(m - 1/2) dt)`` for the m steps to the next compared
step, and ``fields.t`` is asserted.

MODE. ``keep`` (subnormals kept) is the reference of both hosts and every thread
count; ``flush`` exists only on x86 at ``OMP_NUM_THREADS=1``, and the record holds no
flush set, so ``--mode flush`` refuses by name. MEEP's own switch
(``set_zero_subnormals``) is set for the mode, then the processor's handling is
measured and the case refused unless it is the mode.

MINTING (``--mint``) is the only way a record is written. It needs the reference
words of every compared step from an evidence directory laid out as the probe's
(``<DIR>/<case>/<TAG>/<leg>/index.json`` with ``dumps/meep_raw_NNNN.bin``; each
dump's sha256 is checked against its index): each case is stepped twice, in two
processes; the first run's words are compared with the evidence word for word, and
the record is written only if every word is equal, the second process reproduced
every digest, and a planted 1-ulp flip of one word was caught by both the word
compare and the digest compare.

Each case runs in a process of its own, one after the other. Progress: one flushed
line per case and compared step; ``--rows`` appends one JSON line per case.

Exit status: 0 when every digest equals the record's and the planted flip was
caught (``--mint``: when the record was written); 1 when a digest or a word differs;
3 when the smoke could not run (no MEEP, the processor not in the mode, a record or
evidence that cannot be read).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RECORD = os.path.join(HERE, "meep_identity_smoke.json")
SCHEMA = "meep-gpu/meep-identity-smoke/1"
MARK = "IDENTITY-SMOKE-JSON "
FIELD_ORDER = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz")
CASES = ("a", "b", "c", "c_prime")
#: The compared steps of each case (step 0 is the state after ``init_sim``).
SCHEDULE = {"a": (0, 1, 2, 5, 10, 20, 200), "b": (0, 1, 2, 5, 10, 20, 200),
            "c": (0, 1, 2, 5, 10, 20, 375), "c_prime": (0, 1, 2, 5, 10, 20)}
CASE_TIMEOUT_S = 900


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# --------------------------------------------------------------------------
# The cases (the probe's, verbatim in what they build)
# --------------------------------------------------------------------------

def _base_2d(mp: Any, **extra: Any) -> Any:
    return mp.Simulation(
        resolution=20, cell_size=mp.Vector3(10, 10), boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(src=mp.GaussianSource(1.0, fwidth=1.0), center=mp.Vector3(),
                           component=mp.Ez)], **extra)


def _case_3d(mp: Any, fwidth: float) -> Any:
    one = mp.Vector3(1, 1, 1)
    return mp.Simulation(
        resolution=15, cell_size=mp.Vector3(2.6, 2.2, 2.3), boundary_layers=[mp.PML(0.5)],
        geometry=[mp.Block(material=mp.Medium(index=2.4), center=mp.Vector3(), size=one),
                  mp.Block(material=mp.Medium(epsilon=7.9), center=mp.Vector3(1), size=one)],
        symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=-1)],
        sources=[mp.Source(src=mp.GaussianSource(1, fwidth=fwidth), center=mp.Vector3(),
                           component=mp.Hx)])


def build_case(mp: Any, case: str) -> Any:
    if case == "a":
        return _base_2d(mp)
    if case == "b":
        return _base_2d(mp, geometry=[mp.Block(size=mp.Vector3(2, 2, mp.inf),
                                               center=mp.Vector3(1.1, 0),
                                               material=mp.Medium(epsilon=12))])
    if case == "c":
        return _case_3d(mp, 0.4)
    if case == "c_prime":
        return _case_3d(mp, 1.0)
    raise ValueError(f"unknown case {case!r}; the cases are {', '.join(CASES)}")


# --------------------------------------------------------------------------
# Words and digests (no MEEP)
# --------------------------------------------------------------------------

def expand_lattice(compact: List[List[int]]) -> Tuple[List[int], ...]:
    """``[[first, count], x3]`` -> the three ivec coordinate lists (stride 2)."""
    if len(compact) != 3 or any(len(axis) != 2 or axis[1] < 1 for axis in compact):
        raise ValueError(f"a lattice is three [first, count] pairs, not {compact!r}")
    return tuple([first + 2 * i for i in range(count)] for first, count in compact)


def compact_lattice(axes: List[List[int]]) -> List[List[int]]:
    """The inverse of :func:`expand_lattice`; refuses an axis that is not stride 2."""
    out = []
    for axis in axes:
        axis = [int(v) for v in axis]
        if not axis or any(b - a != 2 for a, b in zip(axis, axis[1:])):
            raise ValueError("a lattice axis is not an arithmetic run of stride 2")
        out.append([axis[0], len(axis)])
    return out


def digest(words: Any) -> str:
    """SHA-256 of float32 words, little-endian, in C order."""
    import numpy as np  # noqa: PLC0415

    array = np.ascontiguousarray(words, dtype=np.float32)
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def flip_one_word(words: Any) -> Any:
    """A copy with the least significant bit of one word flipped (the first nonzero
    word, or the first word when all are zero): a 1-ulp change."""
    import numpy as np  # noqa: PLC0415

    flipped = np.array(words, dtype=np.float32, copy=True)
    bits = flipped.reshape(-1).view(np.uint32)
    nonzero = np.flatnonzero(bits & np.uint32(0x7FFFFFFF))
    index = int(nonzero[0]) if nonzero.size else 0
    bits[index] ^= np.uint32(1)
    return flipped


def word_compare(ours: Any, theirs: Any) -> Dict[str, int]:
    """``{words, equal, value_differ, sign_of_zero_only}``: a compare of bit patterns."""
    import numpy as np  # noqa: PLC0415

    a = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(theirs, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return {"words": int(max(a.size, b.size)), "equal": 0,
                "value_differ": int(max(a.size, b.size)), "sign_of_zero_only": 0}
    same_bits = a.view(np.uint32) == b.view(np.uint32)
    with np.errstate(invalid="ignore"):
        same_value = (a == b) | (np.isnan(a) & np.isnan(b))
    return {"words": int(a.size), "equal": int(same_bits.sum()),
            "value_differ": int((~same_value).sum()),
            "sign_of_zero_only": int((same_value & ~same_bits).sum())}


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def load_record(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} record "
                         f"(schema {record.get('schema') if isinstance(record, dict) else None!r})")
    for case, entry in (record.get("cases") or {}).items():
        steps = [int(s) for s in entry.get("steps") or []]
        if sorted(entry.get("digests") or {}, key=int) != [str(s) for s in steps]:
            raise ValueError(f"{path}: case {case} digests do not cover its steps")
        for name in FIELD_ORDER:
            expand_lattice(entry["lattice"][name])
    return record


def load_tool(name: str) -> Any:
    """``tools/<name>.py`` as a module of its own (the import path is not touched)."""
    path = os.path.join(ROOT, "tools", name + ".py")
    spec = importlib.util.spec_from_file_location("meep_gpu_tool_" + name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Evidence (the minting reference's words)
# --------------------------------------------------------------------------

def evidence_dumps(root: str, case: str, build: str) -> Dict[int, Tuple[str, Dict[str, Any]]]:
    """``{step: (leg directory, dump entry)}`` of every MEEP raw dump of one case."""
    found: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for index_path in sorted(glob.glob(os.path.join(root, case, build, "*", "index.json"))):
        leg = os.path.dirname(index_path)
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        if index.get("case") != case or index.get("build") != build:
            continue
        for entry in index.get("dumps") or []:
            name = os.path.basename(entry.get("file", ""))
            if name.startswith("meep_raw_") and name.endswith(".bin"):
                step = int(name[len("meep_raw_"):-len(".bin")])
                found.setdefault(step, (leg, entry))
    return found


def evidence_lattice(root: str, case: str, build: str) -> Dict[str, List[List[int]]]:
    lattices = None
    for index_path in sorted(glob.glob(os.path.join(root, case, build, "*", "index.json"))):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        lattice = index.get("lattice")
        if not lattice:
            continue
        compact = {name: compact_lattice(lattice[name]) for name in FIELD_ORDER}
        if lattices is None:
            lattices = compact
        elif lattices != compact:
            raise ValueError(f"{case}/{build}: two legs record different lattices")
    if lattices is None:
        raise ValueError(f"no index.json with a lattice under {os.path.join(root, case, build)}")
    return lattices


def read_evidence(leg: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """The arrays of one dump, after its sha256 is checked against its index."""
    import numpy as np  # noqa: PLC0415

    path = os.path.join(leg, entry["file"])
    with open(path, "rb") as handle:
        data = handle.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != entry.get("sha256"):
        raise ValueError(f"{path}: sha256 {actual} is not the index's {entry.get('sha256')}")
    out = {}
    for name, spec in entry["arrays"].items():
        count = int(np.prod(spec["shape"]))
        out[name] = np.frombuffer(data, dtype=spec["dtype"], count=count,
                                  offset=int(spec["offset"])).reshape(spec["shape"])
    return out


# --------------------------------------------------------------------------
# The child: one case, MEEP only
# --------------------------------------------------------------------------

def read_component(mp: Any, sim: Any, name: str, axes: Tuple[List[int], ...]):
    """``(float32 words, inexact, imaginary)`` of one component on the lattice."""
    import numpy as np  # noqa: PLC0415

    get_field = sim.fields.get_field
    component = getattr(mp, name)
    ivec = mp.ivec
    xs, ys, zs = axes
    real = np.empty((len(xs), len(ys), len(zs)), dtype=np.float64)
    imaginary = 0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                value = get_field(component, ivec(x, y, z))
                real[i, j, k] = value.real
                if value.imag != 0.0:
                    imaginary += 1
    words = real.astype(np.float32)
    with np.errstate(invalid="ignore"):
        inexact = int(np.count_nonzero((words.astype(np.float64) != real) & ~np.isnan(real)))
    return words, inexact, imaginary


def child(case: str, mode: str, lattice: Dict[str, List[List[int]]], steps: List[int],
          evidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    check = load_tool("check_install")
    import numpy  # noqa: PLC0415, F401 - NumPy before MEEP, as every reference run
    import meep as mp  # noqa: PLC0415

    mp.verbosity(0)
    mp.set_zero_subnormals(mode == "flush")
    state = check.subnormal_state()
    out: Dict[str, Any] = {"case": case, "mode": mode, "subnormals": state,
                           "meep": {"version": str(mp.__version__),
                                    "single_precision": bool(mp.is_single_precision()),
                                    "file": mp.__file__},
                           "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    if state["mode"] != mode:
        out["error"] = (f"the processor's subnormal handling is {state['mode']} after "
                        f"MEEP's set_zero_subnormals({mode == 'flush'}), not {mode}")
        return out
    if not mp.is_single_precision():
        out["error"] = "this MEEP is not a single-precision build"
        return out
    axes = {name: expand_lattice(lattice[name]) for name in FIELD_ORDER}
    sim = build_case(mp, case)
    started = time.time()
    sim.init_sim()
    digests: Dict[str, Dict[str, str]] = {}
    compared = {"words": 0, "equal": 0, "value_differ": 0, "sign_of_zero_only": 0}
    reads = {"inexact": 0, "imaginary": 0}
    flip = None
    for number, step in enumerate(steps, 1):
        todo = step - int(sim.fields.t)
        if todo < 0:
            raise RuntimeError(f"MEEP is at t={int(sim.fields.t)}, cannot step to {step}")
        if todo:
            sim.run(until=(todo - 0.5) * sim.fields.dt)
        if int(sim.fields.t) != step:
            raise RuntimeError(f"MEEP took {int(sim.fields.t)} steps, wanted {step}")
        reference = None
        if evidence is not None:
            leg, entry = evidence[str(step)]
            reference = read_evidence(leg, entry)
        digests[str(step)] = {}
        for name in FIELD_ORDER:
            words, inexact, imaginary = read_component(mp, sim, name, axes[name])
            reads["inexact"] += inexact
            reads["imaginary"] += imaginary
            digests[str(step)][name] = digest(words)
            if reference is not None:
                for key, value in word_compare(words, reference[name]).items():
                    compared[key] += value
            if number == len(steps) and name == FIELD_ORDER[-1]:
                flipped = flip_one_word(words)
                flip = {"component": name, "step": step, "digest": digest(flipped)}
                if reference is not None:
                    counts = word_compare(flipped, reference[name])
                    flip["words_differing"] = counts["words"] - counts["equal"]
        say(f"  {case} {number}/{len(steps)} step {step}: {len(FIELD_ORDER)} components "
            f"read ({time.time() - started:.1f} s)")
    out.update(digests=digests, reads=reads, planted_flip=flip,
               seconds=round(time.time() - started, 3))
    if evidence is not None:
        out["evidence_compare"] = compared
    if reads["inexact"] or reads["imaginary"]:
        out["error"] = (f"{reads['inexact']} read(s) were not exactly a float32 and "
                        f"{reads['imaginary']} had an imaginary part: not stored words")
    return out


def child_main(arguments: argparse.Namespace) -> int:
    with open(arguments.child_input, encoding="utf-8") as handle:
        spec = json.load(handle)
    try:
        result = child(spec["case"], spec["mode"], spec["lattice"], spec["steps"],
                       spec.get("evidence"))
    except Exception as exc:  # noqa: BLE001 - reported by the parent
        result = {"case": spec.get("case"), "error": f"{type(exc).__name__}: {exc}"}
    print(MARK + json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_case(case: str, mode: str, lattice, steps, evidence=None) -> Dict[str, Any]:
    """One case in a process of its own; the child's result, or an error."""
    with tempfile.TemporaryDirectory(prefix="meep-identity-smoke-") as scratch:
        spec_path = os.path.join(scratch, "case.json")
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump({"case": case, "mode": mode, "lattice": lattice, "steps": list(steps),
                       "evidence": evidence}, handle)
        command = [sys.executable, os.path.abspath(__file__), "--child-input", spec_path]
        errors_path = os.path.join(scratch, "stderr.txt")
        result, other = None, []
        with open(errors_path, "w", encoding="utf-8") as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                       text=True, cwd=scratch)
            timer = threading.Timer(CASE_TIMEOUT_S, process.kill)
            timer.start()
            try:
                # The child's progress lines are passed on as they arrive.
                for line in process.stdout:
                    if line.startswith(MARK):
                        result = json.loads(line[len(MARK):])
                    elif line.startswith("["):
                        print(line.rstrip("\n"), flush=True)
                    else:
                        other.append(line.rstrip("\n"))
                process.wait()
            finally:
                timer.cancel()
        if result is not None:
            return result
        with open(errors_path, encoding="utf-8", errors="replace") as handle:
            tail = (handle.read().strip().splitlines() or other)[-3:]
    return {"case": case, "error": f"exited {process.returncode} before reporting "
                                   f"(a kill after {CASE_TIMEOUT_S} s reads -9): "
                                   f"{' | '.join(tail)}"}


def portable(path: str) -> str:
    """A path a committed record may carry: below its last ``results/`` component
    (the evidence archive), or its last component only."""
    parts = os.path.abspath(path).split(os.sep)
    if "results" in parts:
        last = len(parts) - 1 - parts[::-1].index("results")
        return "/".join(parts[last:])
    return parts[-1]


def append_row(path: Optional[str], row: Dict[str, Any]) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


# --------------------------------------------------------------------------
# Run (judge against the record) and mint (write the record)
# --------------------------------------------------------------------------

def compare_digests(expected: Dict[str, Dict[str, str]], found: Dict[str, Dict[str, str]]):
    """``(equal, total, first difference or None)`` over (step, component)."""
    equal = total = 0
    first = None
    for step in sorted(expected, key=int):
        for name in FIELD_ORDER:
            total += 1
            if (found.get(step) or {}).get(name) == expected[step][name]:
                equal += 1
            elif first is None:
                first = f"step {step} {name}"
    return equal, total, first


def command_run(arguments: argparse.Namespace) -> int:
    try:
        record = load_record(arguments.record)
    except (OSError, ValueError, KeyError) as exc:
        print(f"SMOKE NOT RUN: the record could not be read: {exc}", flush=True)
        return 3
    if arguments.mode != record.get("mode"):
        print(f"SMOKE NOT RUN: the record holds the {record.get('mode')} digest set and no "
              f"{arguments.mode} set (a flush set is minted only from an x86 flush "
              "reference at OMP_NUM_THREADS=1)", flush=True)
        return 3
    cases = [c for c in arguments.cases.split(",") if c] if arguments.cases else list(record["cases"])
    status = 0
    totals = [0, 0]
    for number, case in enumerate(cases, 1):
        entry = record["cases"].get(case)
        if entry is None:
            print(f"SMOKE NOT RUN: the record has no case {case!r}", flush=True)
            return 3
        say(f"case {number}/{len(cases)} {case}: steps {entry['steps']}")
        result = run_case(case, arguments.mode, entry["lattice"], entry["steps"])
        if result.get("error"):
            say(f"case {case}: NOT RUN: {result['error']}")
            append_row(arguments.rows, {"case": case, "error": result["error"]})
            return 3
        equal, total, first = compare_digests(entry["digests"], result["digests"])
        flip = result.get("planted_flip") or {}
        caught = flip.get("digest") != entry["digests"][str(flip.get("step"))][flip.get("component")]
        totals[0] += equal
        totals[1] += total
        say(f"case {case}: {equal} of {total} digests equal"
            + (f" (first difference: {first})" if first else "")
            + f"; planted 1-ulp flip {'caught' if caught else 'NOT caught'} "
              f"({result.get('seconds')} s)")
        append_row(arguments.rows, {"case": case, "digests_equal": equal, "digests": total,
                                    "first_difference": first, "planted_flip_caught": caught,
                                    "seconds": result.get("seconds"),
                                    "subnormals": result.get("subnormals")})
        if equal != total or not caught:
            status = 1
    minted_libm = (record.get("minted") or {}).get("libm")
    try:
        here = load_tool("check_install").libm_identity()
        check = load_tool("check_install")
        if check.libm_key(here) != check.libm_key(minted_libm):
            print(f"NOTE: this host's math library ({check.libm_text(here)}) is not the "
                  f"minting host's ({check.libm_text(minted_libm)}); the words above are "
                  "what is judged", flush=True)
    except Exception:  # noqa: BLE001 - a note only
        pass
    print(f"MEEP identity smoke: {totals[0]} of {totals[1]} digests equal over "
          f"{len(cases)} case(s), mode {arguments.mode}: "
          + ("REPRODUCED." if status == 0 else "NOT REPRODUCED."), flush=True)
    return status


def command_mint(arguments: argparse.Namespace) -> int:
    if not arguments.evidence or not arguments.build or not arguments.out:
        print("MINT NOT RUN: --mint needs --evidence, --build and --out", flush=True)
        return 3
    cases = [c for c in arguments.cases.split(",") if c] if arguments.cases else list(CASES)
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "about": ("MEEP-only digest smoke of an identity reference "
                  "(parity/meep_gpu/meep_identity_smoke.py): per case, the compared steps, "
                  "the lattice ([first, count] per axis, stride 2, MEEP ivec units) and the "
                  "SHA-256 of each component's little-endian float32 words in lattice order. "
                  "Written only by --mint."),
        "mode": arguments.mode, "cases": {},
    }
    with open(__file__, "rb") as handle:
        tool_sha256 = hashlib.sha256(handle.read()).hexdigest()
    minted: Dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "tool_sha256": tool_sha256,
                              "evidence": {"root": portable(arguments.evidence),
                                           "build": arguments.build, "cases": {}},
                              "python": sys.version.split()[0]}
    failures = []
    for number, case in enumerate(cases, 1):
        try:
            lattice = evidence_lattice(arguments.evidence, case, arguments.build)
            dumps = evidence_dumps(arguments.evidence, case, arguments.build)
        except (OSError, ValueError, KeyError) as exc:
            print(f"MINT NOT RUN: case {case}: {exc}", flush=True)
            return 3
        steps = list(SCHEDULE[case])
        missing = [s for s in steps if s not in dumps]
        if missing:
            print(f"MINT NOT RUN: case {case}: the evidence has no MEEP words at steps "
                  f"{missing}", flush=True)
            return 3
        evidence = {str(s): [dumps[s][0], dumps[s][1]] for s in steps}
        say(f"case {number}/{len(cases)} {case}: first run, words compared with "
            f"{arguments.build} evidence at steps {steps}")
        first = run_case(case, arguments.mode, lattice, steps, evidence)
        if first.get("error"):
            print(f"MINT NOT RUN: case {case}: {first['error']}", flush=True)
            return 3
        say(f"case {number}/{len(cases)} {case}: second run, a new process")
        second = run_case(case, arguments.mode, lattice, steps)
        if second.get("error"):
            print(f"MINT NOT RUN: case {case}: {second['error']}", flush=True)
            return 3
        compared = first["evidence_compare"]
        equal, total, difference = compare_digests(first["digests"], second["digests"])
        flip = first.get("planted_flip") or {}
        flip_caught = (flip.get("words_differing") == 1 and flip.get("digest")
                       != first["digests"][str(flip["step"])][flip["component"]])
        say(f"case {case}: {compared['equal']} of {compared['words']} words equal to the "
            f"evidence ({compared['value_differ']} value-differing, "
            f"{compared['sign_of_zero_only']} sign-of-zero only); second process "
            f"{equal} of {total} digests equal; planted flip "
            f"{'caught' if flip_caught else 'NOT caught'}")
        minted["evidence"]["cases"][case] = {
            "words_compared": compared["words"], "words_equal": compared["equal"],
            "value_differ": compared["value_differ"],
            "sign_of_zero_only": compared["sign_of_zero_only"],
            "second_process_digests_equal": equal, "digests": total,
            "planted_flip_caught": flip_caught,
            "dumps": {str(s): {"leg": os.path.relpath(dumps[s][0], arguments.evidence),
                               "sha256": dumps[s][1].get("sha256")} for s in steps}}
        append_row(arguments.rows, {"case": case, **minted["evidence"]["cases"][case]})
        if compared["equal"] != compared["words"] or equal != total or not flip_caught:
            failures.append(case)
            continue
        record["cases"][case] = {"steps": steps, "lattice": lattice, "digests": first["digests"]}
        minted.setdefault("meep", {key: (first.get("meep") or {}).get(key)
                                   for key in ("version", "single_precision")})
        minted.setdefault("subnormals", first.get("subnormals"))
    if failures:
        print(f"MINT REFUSED: case(s) {', '.join(failures)} did not reproduce the evidence "
              "word for word, or were not reproduced by a second process, or missed the "
              "planted flip; nothing was written", flush=True)
        return 1
    try:
        check = load_tool("check_install")
        import meep  # noqa: PLC0415, F401 - for the images the minting process loads
        minted["meep_images"] = {os.path.basename(path): check.sha256_of(path)
                                 for path in check.loaded_image_paths()
                                 if os.path.basename(path).startswith(("libmeep", "_meep"))}
        minted["libm"] = check.libm_identity()
    except Exception as exc:  # noqa: BLE001 - provenance only
        minted["provenance_unread"] = f"{type(exc).__name__}: {exc}"
    record["minted"] = minted
    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
    words = sum(c["words_compared"] for c in minted["evidence"]["cases"].values())
    print(f"MINTED {arguments.out}: {len(record['cases'])} case(s), {words} of {words} words "
          "equal to the evidence, every digest reproduced by a second process, the planted "
          "flip caught", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("keep", "flush"), default="keep")
    parser.add_argument("--record", default=RECORD, help="the digest record to judge by")
    parser.add_argument("--cases", help="a comma-separated subset of the cases")
    parser.add_argument("--rows", help="append one JSON line per case here")
    parser.add_argument("--mint", action="store_true",
                        help="write a record from evidence, after the checks above")
    parser.add_argument("--evidence", help="--mint: the evidence directory")
    parser.add_argument("--build", help="--mint: the evidence's build tag (e.g. coff)")
    parser.add_argument("--out", help="--mint: where to write the record")
    parser.add_argument("--child-input", dest="child_input", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.child_input:
        return child_main(arguments)
    if arguments.mint:
        return command_mint(arguments)
    return command_run(arguments)


if __name__ == "__main__":
    sys.exit(main())
