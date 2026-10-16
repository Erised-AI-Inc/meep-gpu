#!/usr/bin/env python3
"""The 1.0 Metal certification round on one Mac: every Metal gate, then the held route.

Stages (``all`` runs preflight, fleet and status in order, skips one that finished, and
exits with the status stage's code; ``route`` runs on its own, after the bind):

``preflight``
    the commit and a clean checkout (``--expect``), no ``MEEP_GPU_*`` shaping switch in
    the calling shell, the package imported from the checkout, the reference
    environment (``tools/check_install.py --require-reference``), the Metal
    environment readable, the disk floor, no output of this stamp yet, the MEEP corpus
    (below), and the LIFT CELLS: in one child under ``--python`` with torch
    unimportable, every fleet gate that launches lift children (:func:`lifting_kind`:
    nine gates, each defining ``lift_basis`` or built on ``hd_tail_gate_common``)
    computes its own lift basis. An absent input (``lift_host.lift_inputs``: the
    census's three row files and the seam record, none of them shipped) stops the
    preflight, naming the gate and the file. Each row this host cannot lift (a need its
    lift record declares, ``lift_host.LIFT_RECORD_NEEDS``, probed absent under
    ``--python``) is reported by gate and name, as the gate will record it: unlifted on
    this host and out of its denominator. That stops nothing, and neither does a basis
    the child cannot compute while its inputs are present (the gate's leg reports it).
    A fleet gate that reads the same four files in process and launches no lift child
    (:func:`census_reader`: ``gate_metal_complex_fused_hd_pair``) has them checked too.
``fleet`` (CERT-3)
    ``recut_metal_gates.sh``'s fleet, gate by gate: the environment record, then every
    ``gate_metal_*`` in directory order with the three longest last, then the
    environment record again. Per gate: the environment ``recut_metal_gates.sh`` pins
    (flush policy, dispatch off, residency ``shipped``, the campaign source manifest),
    the disk floor, a watchdog, a marker and a JSON row. Outputs land where the rebind reads
    them: ``results/metal_<family>_<stamp>/``.
``route`` (CERT-4)
    ``run_metal_dispatch_campaign.sh`` under held residency, into
    ``results/dispatch_metal_route_<route stamp>/`` (the route constant's stamp and
    this Mac's architecture, which ``recut_driver_dispatch_record.py --backend metal``
    requires). Its watchdog comes from the MEASURED route walls (:data:`ROUTE_EXPECT`,
    :func:`route_watchdog`; ``--route-timeout-factor``, ``--route-min-timeout``):
    route campaign names are one-shot, so a campaign its watchdog kills needs a new
    route stamp, which is a new freeze.
``status``
    released N of D gate artifacts, and the welds the Metal table cites, N of D.

THE MAC IS SHARED (slot v2). With ``--slot-wrapper`` every gate and the whole
route campaign is run through that wrapper with ``--classes`` (default
``mac_gpu+mac_cpu``) in ONE call, so the slots are taken all-or-nothing and the one
GPU slot is never held while a CPU slot is awaited (a nested ``mac_gpu`` then
``mac_cpu`` acquisition held the GPU slot idle for hours in a 1.0 pre-certification
pass). Without it the commands run directly, as on a quiet Mac.

Subnormal policy: flush only (R13: the Apple GPU flushes and exposes no lever).

THE MEEP CORPUS the lift legs walk is the MEEP release's ``python/examples`` and
``python/tests``, which the reference environment's build keeps inside the prefix and
records file by file: the ``corpus`` entry of
``<prefix>/share/meep-gpu-build/reference_build.json`` (:data:`BUILD_RECORD`; the prefix
is ``--python``'s) and the sha256 list it pins. ``--corpus-root`` names another copy of
the same tree. Either way every file the record lists must be present with its sha256,
every corpus sibling it names (a file outside ``python/`` that a corpus row reads, kept
beside it) must be there with its sha256 (a record without a ``siblings`` entry is
refused), and no unlisted file Python could import (:data:`CORPUS_REFUSED_EXTRAS`: source, byte
code outside ``__pycache__``, an extension module) may sit under the root, since it could
shadow an import, or the preflight and the fleet stop by name. The root reaches every
fleet child as ``MEEP_GPU_CORPUS_ROOT``, the variable the gates read first; one exported
in the calling shell is refused (:data:`SCRUBBED`).

THE USER SITE DIRECTORY IS OFF for every child (``PYTHONNOUSERSITE=1`` in
:meth:`Round.base_env`), as the certification reference runs (``tools/check_install.py``
notes it otherwise): a distribution there sits on the import path AHEAD of the
environment's own site-packages. The preflight refuses a child interpreter that still
enables it.

RESUME. Every finished item writes ``<logs>/done/<label>.rc`` and a row in
``<logs>/gates.jsonl``; a rerun skips an item with a marker. A stage that stopped
writes ``STOPPED`` with the reason. Delete a gate's marker AND move its output
directory out of ``results/`` (keep it) to rerun that gate.

It writes no ledger, fingerprint or record.

CHUNKED (``--chunk-index I --chunk m<k>``; preflight, fleet, status and all). The chunk's
stamp (``chunks.py``: ``<round>_<architecture>_m<k>``) replaces ``--stamp``, so its outputs,
its environment records, its campaign source manifest and its logs (``logs_metal_<stamp>``)
are its own; only its gates run, in this driver's order; ``<logs>/LAUNCH.lock`` keeps a
second launcher of the same chunk out while a stage runs; the preflight also refuses an index
whose chunks could collide, a checkout whose route-stamp registry names another round, and an
index of another architecture or mode (:func:`architecture_problems`). Chunks share the one
Mac GPU through the slot wrapper, one gate at a time. ``status --chunk-index I`` without
``--chunk`` reads each gate under the newest launch record that names it; the round's bind is
``chunks.py verify --selection F`` then ``chunks.py bind --index I --selection F``
(``rebind_metal_welds.py --stamp`` once per selected stamp).

A NEW ARCHITECTURE (``--new-architecture``; an Apple M3 beside the M1 Max): a chunk of an
index partitioned with ``--new-device`` for this Mac's architecture. The preflight and the
fleet stage each read this Mac's architecture and hold it to the Metal ledger
(:meth:`Round.device_problems`): in a chunk, the flag both ways (an architecture the ledger
admits is certified, one it does not is new); unchunked, the ledger must admit it. The flag
is refused without a chunk and on the route stage, and the bind adds the architecture's runs
beside the certified ones.

usage: metal_round.py <preflight|fleet|route|status|all> --repo R --stamp S
           [--expect HASH] [--meep-source DIR] [--corpus-root DIR]
           [--slot-wrapper W --slot-log F]
       metal_round.py <preflight|fleet|status|all> --repo R --chunk-index I --chunk m1 [...]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: 2026-10-04 round walls on one Apple M1 Max (seconds), the 0.9.2 fleet this
#: driver re-runs. A watchdog is max(900, 4 x this + 600).
EXPECT = {
    "gate_metal_complex": 23,
    "gate_metal_ade_update_p": 2, "gate_metal_below_the_cut_fused_pairs": 20,
    "gate_metal_beta_complex_fused_hd_pair": 98, "gate_metal_beta_real_fused_hd_pair": 103,
    "gate_metal_bfast": 29, "gate_metal_bfast_fused_hd_pair": 615,
    "gate_metal_complex_conductive_fused_pair": 9, "gate_metal_complex_conductive_pml": 1,
    "gate_metal_complex_dispersive_update_e": 2, "gate_metal_complex_folded_offdiag": 2,
    "gate_metal_complex_fused_ade_chain": 11, "gate_metal_complex_fused_hd_pair": 18,
    "gate_metal_complex_fused_magnetic_pair": 10, "gate_metal_complex_no_pml_conductive": 2,
    "gate_metal_complex_no_pml_curl": 1, "gate_metal_complex_no_pml_offdiag_update_e": 2,
    "gate_metal_complex_no_pml_stored_e": 1, "gate_metal_conductive_fused_hd_pair": 196,
    "gate_metal_conductive_pml": 2, "gate_metal_cylindrical_complex": 31,
    "gate_metal_cylindrical_complex_fused_hd_pair": 7, "gate_metal_cylindrical_complex_scan": 14,
    "gate_metal_cylindrical_fused_electric_pair": 15, "gate_metal_cylindrical_fused_magnetic_pair": 16,
    "gate_metal_cylindrical_real": 39, "gate_metal_cylindrical_real_fused_electric_pair": 11,
    "gate_metal_cylindrical_real_fused_hd_pair": 74, "gate_metal_cylindrical_real_fused_magnetic_pair": 9,
    "gate_metal_dispersive_update_e": 2, "gate_metal_folded_beta": 22,
    "gate_metal_folded_beta_complex_fused_magnetic_pair": 21, "gate_metal_folded_complex": 5,
    "gate_metal_folded_complex_fused_hd_pair": 106, "gate_metal_folded_complex_fused_magnetic_pair": 21,
    "gate_metal_folded_dispersive_update_e": 2, "gate_metal_folded_fused_hd_pair": 832,
    "gate_metal_folded_fused_magnetic_pair": 33, "gate_metal_folded_fused_pair": 13,
    "gate_metal_folded_offdiag": 8, "gate_metal_folded_offdiag_dispersive": 2,
    "gate_metal_fused_ade_chain": 13, "gate_metal_fused_dispersive_pair": 15,
    "gate_metal_fused_electric_pair": 17, "gate_metal_fused_hd_pair": 1685,
    "gate_metal_fused_hd_pair_nonlinear": 142, "gate_metal_fused_magnetic_pair": 15,
    "gate_metal_no_pml_conductive": 1, "gate_metal_no_pml_constitutive": 5,
    "gate_metal_no_pml_curl": 2, "gate_metal_no_pml_fused_electric_pairs": 14,
    "gate_metal_no_pml_stored_e": 2, "gate_metal_nonlinear": 8, "gate_metal_offdiag": 38,
    "gate_metal_offdiag_stencil_welds": 28, "gate_metal_pml": 8,
    "gate_metal_residue_fused_pairs": 10, "gate_metal_special_kz": 29,
    "gate_metal_spine_byte_neutrality": 1, "gate_metal_symmetry": 15,
    "gate_metal_tranche7_fused_pairs": 18, "gate_metal_whole_step": 34,
}
#: The three longest gates run last, so a stop late in the fleet costs the most
#: measurements already landed.
LONG = ("gate_metal_fused_hd_pair", "gate_metal_folded_fused_hd_pair",
        "gate_metal_bfast_fused_hd_pair")
#: THE ROUTE LEGS' MEASURED WALLS (seconds), the four legs of
#: ``run_metal_dispatch_campaign.sh`` (its ``SERIAL_LEGS`` then ``LANE_LEGS``): the 1.0
#: pre-certification pass's held route campaign on one Apple M1 Max (2026-10-10/11,
#: series entries 01 to 114, tree 15da8cd4; 3 lanes; every leg released), under host
#: contention -- harness_keep took 6,479 s where an earlier quiet run took 286 s, and the
#: campaign 12,648 s. Contended walls are over-estimates of a quiet run, which is what a
#: watchdog base should be. The shell driver's own ``MEASURED_SECONDS`` only orders its
#: lanes; nothing there limits a leg, and neither ``metal_gate_runner.py`` nor the gate
#: carries a timeout, so this item's watchdog is the campaign's only one.
ROUTE_EXPECT: Dict[str, int] = {"band_witness": 883, "harness_keep": 6479,
                                "shipped": 5283, "harness_flush": 5275}
#: The campaign's watchdog is ``ROUTE_TIMEOUT_FACTOR`` times the SERIAL sum of
#: :data:`ROUTE_EXPECT` (the wall at ``--lanes 1``, so the bound holds at any lane
#: count), and at least ``ROUTE_MIN_TIMEOUT`` (the 6 h this driver used before): 53,760 s
#: by default, 3.0x the serial wall and 4.25x the measured 3-lane campaign. A longer
#: watchdog only costs wall time on a hung campaign, which burns the stamp either way.
ROUTE_TIMEOUT_FACTOR = 3
ROUTE_MIN_TIMEOUT = 6 * 3600


def route_watchdog(factor: int = ROUTE_TIMEOUT_FACTOR,
                   floor: int = ROUTE_MIN_TIMEOUT) -> int:
    """The route campaign's watchdog, in seconds (slot waits are ``--slot-patience``'s)."""
    return max(int(floor), int(factor) * sum(ROUTE_EXPECT.values()))


#: Shaping switches the calling shell must not export; the fleet sets what it needs.
#: The retired MEEP_GPU_ALLOW_UNCERTIFIED is listed beside its replacement: the package
#: refuses it by name, set to anything, so the fleet must not inherit it either.
SCRUBBED = ("MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_HARDWARE", "MEEP_GPU_ALLOW_UNCERTIFIED",
            "MEEP_GPU_SUBNORMAL_POLICY",
            "MEEP_GPU_KERNEL_TABLE", "MEEP_GPU_CORPUS_ROOT", "MEEP_GPU_H_TO_D_PROBE",
            "MEEP_GPU_DISPATCH",
            "MEEP_GPU_FUSED", "MEEP_GPU_FUSE_ARMS", "MEEP_GPU_GATE_SOURCE_MANIFEST")
ORDER = ("preflight", "fleet", "route", "status")
#: The one file the route stage may find changed in the frozen checkout: the Metal
#: ledger the bind rewrote after the fleet, copied in uncommitted (staged or not).
ROUTE_STAGE_LEDGERS = ("meep_gpu/metal_kernels/fingerprints.json",)

#: What the route stage runs in the frozen checkout to name its campaign: the GPU
#: architecture the fleet recorded, the registry's Metal stamp, the campaign
#: ``metal_runs.route_campaign`` spells for the two, and the registry's refusal of that
#: name (None when it may run). One JSON line on stdout.
ROUTE_NAMES_PROGRAM = (
    "import json, metal_environment as m\n"
    "from meep_gpu import metal_runs as r, route_stamps as s\n"
    "a = m.read()['architecture']\n"
    "g = s.route_gate('metal')\n"
    "c = r.route_campaign(g, a)\n"
    "print(json.dumps({'arch': a, 'stamp': g, 'campaign': c,\n"
    "                  'refused': s.campaign_refusal('metal', c)}))\n")

#: THE CORPUS RULE IS legs.py's, one home for both round drivers: the reference build
#: record under the prefix of ``--python`` (``legs.BUILD_RECORD_DIR``, ``legs.BUILD_RECORD``),
#: whose ``corpus`` entry names the kept MEEP corpus, the sha256 list that pins it and the
#: siblings beside it, read by ``legs.read_corpus_record`` and checked by
#: ``legs.verify_corpus``; ``round.py`` and ``legs.py check`` apply the same functions on
#: the NVIDIA host.
from legs import (BUILD_RECORD, BUILD_RECORD_DIR, CORPUS_REFUSED_EXTRAS,  # noqa: E402,F401
                  read_corpus_record, resolve_corpus, sha256_of, verify_corpus)

#: What the preflight of a chunk runs in the checkout to name this Mac's architecture and
#: the architectures the Metal ledger admits (``metal_runs.admission_report`` over the welds
#: ``metal_dispatch.ARM_CERTIFICATION`` cites). One JSON line on stdout.
ADMISSION_PROGRAM = (
    "import json, metal_environment as m\n"
    "from meep_gpu import metal_runs as r\n"
    "ledger = json.load(open('meep_gpu/metal_kernels/fingerprints.json'))\n"
    "admitted = r.admission_report(ledger, r.cited_keys())['admitted']\n"
    "print(json.dumps({'arch': m.read()['architecture'],\n"
    "                  'admitted': None if admitted is None else list(admitted)}))\n")
#: Why a new architecture runs chunked: only a chunk stamp carries the architecture, so an
#: unchunked stamp's roots could equal, or be extended by, another Mac's of the same round.
NEW_ARCHITECTURE_UNCHUNKED = (
    "--new-architecture runs chunked: partition an index with chunks.py partition --machine "
    "metal --architecture <this Mac's architecture> --new-device and run its chunks "
    "(--chunk-index, --chunk), whose stamps carry the architecture")


def architecture_problems(index: Mapping[str, Any], architecture: Optional[str],
                          admitted: Optional[List[str]], new_architecture: bool) -> List[str]:
    """Why a chunk index does not certify this Mac in this mode: its device is another
    architecture, its ``device_mode`` disagrees with ``--new-architecture``, or the flag
    disagrees with the Metal ledger (an architecture it admits is certified, one it does not
    is new). Nothing about a new architecture is inferred."""
    problems: List[str] = []
    if not architecture:
        return ["this Mac's GPU architecture could not be read (metal_environment.read())"]
    if index.get("device") != architecture:
        problems.append(f"the chunk index certifies {index.get('device')}, and this Mac is "
                        f"{architecture}")
    mode = "new" if new_architecture else "certified"
    if index.get("device_mode") != mode:
        problems.append(f"the chunk index is of a {index.get('device_mode')} device and this "
                        f"launch says {mode} ("
                        + ("drop --new-architecture" if new_architecture else
                           "pass --new-architecture") + ")")
    if admitted is None:
        problems.append("the Metal ledger's admission could not be read, so whether "
                        f"{architecture} is new cannot be checked")
    elif new_architecture and architecture in admitted:
        problems.append(f"--new-architecture: the Metal ledger already admits {architecture} "
                        f"({admitted}); it is a certified architecture, so run without the flag")
    elif not new_architecture and architecture not in admitted:
        problems.append(f"the Metal ledger admits {admitted}, not {architecture}: an "
                        f"architecture it holds no live run for is gated with "
                        f"--new-architecture")
    return problems


#: THE PER-CELL LIFT PREFLIGHT, run as ``--python -c LIFT_CELLS_PROGRAM <this file's
#: directory> <harness> <results> <gate stems as JSON>``: :func:`lift_cells_child` in a
#: child, so the gates' own ``lift_basis`` and ``lift_host`` run under the round's
#: interpreter, never in this process. Its answer is one ``LIFT_CELLS=<json>`` line.
LIFT_CELLS_PREFIX = "LIFT_CELLS="
LIFT_CELLS_PROGRAM = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import metal_round\n"
    "sys.exit(metal_round.lift_cells_child(sys.argv[2:]))\n")
#: Seconds the lift preflight child may take (it reads four files per gate and imports
#: MEEP once for the capability probe).
LIFT_CELLS_TIMEOUT = 900


class Round:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.g = self.repo / "parity" / "meep_gpu"
        self.results = Path(args.results).resolve() if args.results else self.g / "results"
        self.logs = Path(args.logs).resolve() if args.logs else self.results / f"logs_metal_{args.stamp}"
        self.root = self.results / f"metal_campaign_{args.stamp}" / "fleet"
        self.py = args.python
        self.child: Optional[subprocess.Popen] = None
        self._corpus: Optional[Dict[str, Any]] = None
        for sub in ("done", "logs", "timing"):
            (self.logs / sub).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- reporting ----
    def say(self, stage: str, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{stage}] {message}"
        print(line, flush=True)
        with (self.logs / "round.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stop(self, stage: str, why: str) -> int:
        self.say(stage, f"STOP: {why}")
        (self.logs / "STOPPED").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} [{stage}] {why}\n")
        return 3

    def row(self, row: Dict[str, Any]) -> None:
        with (self.logs / "gates.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------- environment ----
    def free_kb(self) -> int:
        stat = shutil.disk_usage(self.results if self.results.exists() else self.repo)
        return stat.free // 1024

    def base_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
        env.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                    "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(self.repo),
                    "PYTHONNOUSERSITE": "1",
                    "MGPU_SITE_PYTHON": self.py, "MGPU_SITE_SOURCE_ROOT": str(self.repo)})
        if self.args.meep_source:
            env["MGPU_SITE_MEEP_SOURCE"] = self.args.meep_source
        return env

    def fleet_env(self) -> Dict[str, str]:
        env = self.base_env()
        env.update({"MEEP_GPU_SUBNORMAL_POLICY": "flush", "MEEP_GPU_DISPATCH": "0",
                    "MEEP_GPU_METAL_RESIDENCY": "shipped",
                    "MEEP_GPU_GATE_SOURCE_MANIFEST": str(self.root / "source_sha256.txt")})
        # THE ROUND'S CORPUS, never the calling shell's (SCRUBBED): resolved here, so a
        # resumed ``all`` that skips the preflight still exports it.
        root = self.corpus()["root"]
        if root:
            env["MEEP_GPU_CORPUS_ROOT"] = root
        return env

    def corpus(self) -> Dict[str, Any]:
        """The MEEP corpus the lift legs walk (:func:`resolve_corpus`), resolved once."""
        if self._corpus is None:
            self._corpus = resolve_corpus(self.py, getattr(self.args, "corpus_root", "") or "",
                                          self.base_env())
        return self._corpus

    def fleet_gates(self) -> List[str]:
        """The fleet's gate stems in directory order, narrowed by ``--only``."""
        gates = sorted(p.stem for p in self.g.glob("gate_metal_*.py"))
        only = {s for s in (getattr(self.args, "only", "") or "").split(",") if s}
        return [g for g in gates if g in only] if only else gates

    def lifting_gates(self) -> List[str]:
        """The fleet's gates that launch lift children (:func:`lifting_kind`)."""
        return [stem for stem in self.fleet_gates()
                if lifting_kind((self.g / f"{stem}.py").read_text(encoding="utf-8"))]

    def census_readers(self) -> List[str]:
        """The fleet's gates that read the lift inputs in process and launch no lift child
        (:func:`census_reader`)."""
        return [stem for stem in self.fleet_gates()
                if census_reader((self.g / f"{stem}.py").read_text(encoding="utf-8"))]

    def lift_preflight(self) -> Tuple[List[str], List[str]]:
        """``(problems, notes)`` of the per-cell lift preflight (:func:`lift_cells_child`):
        the lifting gates' bases, and the inputs of the gates that read them in process."""
        stems = self.lifting_gates() + self.census_readers()
        if not stems:
            return [], ["no gate of this fleet launches lift children"]
        env = {**self.fleet_env(), "PYTHONPATH": f"{self.repo}{os.pathsep}{self.g}"}
        argv = [self.py, "-c", LIFT_CELLS_PROGRAM, str(Path(__file__).resolve().parent),
                str(self.g), str(self.results), json.dumps(stems)]
        try:
            done = subprocess.run(argv, cwd=str(self.repo), env=env, capture_output=True,
                                  text=True, timeout=LIFT_CELLS_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"the lift preflight under {self.py} did not finish: "
                    f"{type(exc).__name__}: {exc}"], []
        found = [line[len(LIFT_CELLS_PREFIX):] for line in (done.stdout or "").splitlines()
                 if line.startswith(LIFT_CELLS_PREFIX)]
        try:
            answer = json.loads(found[-1])
        except (IndexError, ValueError):
            return [f"the lift preflight under {self.py} printed no answer (exit "
                    f"{done.returncode}): {(done.stderr or '').strip()[-300:]}"], []
        return summarize_lift_cells(answer, stems, self.results)

    def wrapped(self, argv: List[str]) -> List[str]:
        if not self.args.slot_wrapper:
            return argv
        return ["zsh", self.args.slot_wrapper, self.args.classes,
                self.args.slot_log or str(self.logs / "slots.log"), "--", *argv]

    # ---------------------------------------------------------------- one item ----
    def run_item(self, stage: str, label: str, argv: List[str], env: Dict[str, str],
                 artifact: Optional[Path], timeout: int) -> int:
        marker = self.logs / "done" / f"{label}.rc"
        if marker.is_file():
            self.say(stage, f"{label}: already done (rc {marker.read_text().strip()}); skipped")
            return int(marker.read_text().strip() or 1)
        floor = self.args.disk_floor_kb
        if self.free_kb() < floor:
            self.say(stage, f"DISK FLOOR: {self.free_kb()} KB free < {floor} KB; {label} not started")
            raise SystemExit(self.stop(stage, f"disk floor before {label}"))
        log = self.logs / "logs" / f"{label}.log"
        queued = time.time()
        with log.open("w", encoding="utf-8") as handle:
            handle.write(f"# {time.strftime('%FT%TZ', time.gmtime())} {label}: {' '.join(argv)}\n")
            handle.flush()
            self.child = subprocess.Popen(self.wrapped(["nice", "-n", "10", "caffeinate", "-i", *argv]
                                                       if shutil.which("caffeinate") else argv),
                                          cwd=str(self.repo), env=env, stdout=handle,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            killed = False
            while True:
                try:
                    self.child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if time.time() - queued > timeout + self.args.slot_patience:
                    killed = True
                    self.kill_child()
                    break
        rc = 124 if killed else self.child.returncode
        self.child = None
        elapsed = round(time.time() - queued, 1)
        verdict = read_verdict(artifact) if artifact is not None else None
        self.row({"label": label, "stage": stage, "rc": rc, "elapsed_s": elapsed,
                  "timed_out": killed, "artifact": str(artifact) if artifact else None,
                  "released": verdict, "log": str(log),
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        marker.write_text(f"{rc}\n")
        workdir = (artifact.parent / "lift" / "workdir") if artifact is not None else None
        if workdir is not None and workdir.is_dir():
            shutil.rmtree(workdir, ignore_errors=True)
        self.say(stage, f"{label} rc={rc} {elapsed} s released={verdict}"
                        + (" (watchdog)" if killed else ""))
        return rc

    def kill_child(self) -> None:
        if self.child is None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except OSError:
                pass
            self.child.wait()

    def limit(self, label: str) -> int:
        return max(900, 4 * EXPECT.get(label, 300) + 600)

    # ------------------------------------------------------------------ stages ----
    def tree_is_the_commit(self, stage: str) -> Optional[str]:
        if not self.args.expect:
            return None
        head = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head != self.args.expect:
            return f"HEAD is {head}, not {self.args.expect}"
        dirty = subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain"],
                               capture_output=True, text=True).stdout
        # Each porcelain line is two status columns, a space and the path, and the
        # first column is a space for a change that is not staged (" M <path>"). The
        # output is split as it stands, never stripped first: a strip of the whole
        # output removes the first line's leading space, so its path is read one
        # character short and the one change the route allows (the Metal ledger the
        # bind writes, unstaged) is refused (legs.py's check reads it the same way).
        lines = [l for l in dirty.splitlines() if l.strip()]
        allowed = set(ROUTE_STAGE_LEDGERS) if stage == "route" else set()
        stray = [l for l in lines if l[3:] not in allowed]
        return f"the checkout has local changes: {stray[:3]}" if stray else None

    def device_problems(self, stage: str) -> List[str]:
        """THIS MAC'S ARCHITECTURE AGAINST THE METAL LEDGER, read at every stage that runs
        gates (the preflight and the fleet, so a fleet launched alone is held to it too).
        A chunk: :func:`architecture_problems` against its index and ``--new-architecture``.
        Unchunked: the ledger must admit the architecture; one it does not admit is new,
        and a new architecture runs chunked with the flag (never silently)."""
        named = subprocess.run([self.py, "-c", ADMISSION_PROGRAM], cwd=str(self.repo),
                               env={**self.base_env(),
                                    "PYTHONPATH": f"{self.repo}{os.pathsep}{self.g}"},
                               capture_output=True, text=True)
        try:
            facts = json.loads(named.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError, AttributeError):
            facts = None
        if not isinstance(facts, dict):
            facts = {"arch": None, "admitted": None}
        architecture, admitted = facts.get("arch"), facts.get("admitted")
        self.say(stage, f"architecture {architecture}; the Metal ledger admits {admitted}")
        if getattr(self.args, "chunk_manifest", None) is not None:
            return [f"chunk index: {line}" for line in architecture_problems(
                self.args.chunk_index_data, architecture, admitted,
                bool(getattr(self.args, "new_architecture", False)))]
        if not architecture:
            return ["this Mac's GPU architecture could not be read (metal_environment.read())"]
        if admitted is None:
            return [f"the Metal ledger's admission could not be read, so whether "
                    f"{architecture} is new cannot be checked"]
        if architecture not in admitted:
            return [f"the Metal ledger admits {admitted}, not this Mac's {architecture}: an "
                    f"architecture it holds no live run for is new, and a new architecture "
                    f"runs chunked with --new-architecture (chunks.py partition --machine "
                    f"metal --architecture {architecture} --new-device)"]
        return []

    def stage_preflight(self) -> int:
        st = "preflight"
        problems = []
        why = self.tree_is_the_commit(st)
        if why:
            problems.append(why)
        if getattr(self.args, "chunk_manifest", None) is not None:
            import chunks as C  # noqa: PLC0415 - beside this file
            index = self.args.chunk_index_data
            problems += [f"chunk index: {line}" for line in
                         C.collisions(index, results=self.results, repo=self.repo)]
            problems += C.round_problems(self.repo, index)
        problems += self.device_problems(st)
        exported = [v for v in SCRUBBED if v in os.environ]
        if exported:
            problems.append(f"{exported} exported in the calling shell")
        if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
            problems.append("PYTORCH_MPS_FAST_MATH is set; no weld ran fast-math sources")
        probe = subprocess.run([self.py, "-c", "import os, site, meep_gpu; "
                                "print(site.ENABLE_USER_SITE); print(os.path.realpath(meep_gpu.__file__))"],
                               cwd=str(self.repo), env=self.base_env(), capture_output=True, text=True)
        lines = probe.stdout.strip().splitlines()
        where = lines[-1] if lines else ""
        if not where.startswith(str(self.repo) + os.sep):
            problems.append(f"meep_gpu imports from {where or probe.stderr[-200:]}, not the checkout")
        if lines[:1] != ["False"]:
            problems.append(f"the child interpreter enables the user site directory "
                            f"(site.ENABLE_USER_SITE {lines[:1]}) with PYTHONNOUSERSITE=1")
        check = subprocess.run([self.py, "tools/check_install.py", "--require-reference"],
                               cwd=str(self.repo), env=self.base_env(), capture_output=True, text=True)
        (self.logs / "check_install.out").write_text(check.stdout + check.stderr)
        if check.returncode != 0:
            problems.append(f"tools/check_install.py --require-reference: "
                            f"{(check.stdout + check.stderr).strip().splitlines()[-1:]}")
        envread = subprocess.run([self.py, "-c", "import metal_environment as m; e = m.read(); "
                                  "print(e['architecture'], e['torch'], e['device_name'])"],
                                 cwd=str(self.repo), env={**self.base_env(),
                                                          "PYTHONPATH": f"{self.repo}:{self.g}"},
                                 capture_output=True, text=True)
        self.say(st, f"Metal environment: {envread.stdout.strip() or envread.stderr.strip()[-200:]}")
        if envread.returncode != 0 or "None" in envread.stdout.split():
            problems.append("a Metal environment fact could not be read")
        # THE CORPUS is the lift legs' alone: a fleet with no lifting gate notes it.
        corpus = self.corpus()
        self.say(st, f"MEEP corpus: {corpus['root'] or 'not resolved'} "
                     f"(record {corpus['record'] or 'not found'})")
        for note in corpus["notes"]:
            self.say(st, f"corpus: {note}")
        if self.lifting_gates():
            problems += [f"corpus: {line}" for line in corpus["problems"]]
        else:
            for line in corpus["problems"]:
                self.say(st, f"corpus (no gate of this fleet lifts): {line}")
        lift_problems, lift_notes = self.lift_preflight()
        for note in lift_notes:
            self.say(st, f"lift: {note}")
        problems += [f"lift: {line}" for line in lift_problems]
        taken = sorted(p.name for p in self.results.glob(f"metal_*_{self.args.stamp}"))
        if taken:
            problems.append(f"outputs of stamp {self.args.stamp} exist: {taken[:3]}")
        if self.free_kb() < self.args.disk_floor_kb:
            problems.append(f"{self.free_kb()} KB free < floor {self.args.disk_floor_kb} KB")
        for problem in problems:
            self.say(st, f"FAIL {problem}")
        if problems and not self.args.dry_run_allow:
            return self.stop(st, f"{len(problems)} preflight problem(s)")
        if problems:
            self.say(st, f"DRY RUN: going on past {len(problems)} problem(s) (--dry-run-allow)")
        (self.logs / ".done_preflight").touch()
        return 0

    def stage_fleet(self) -> int:
        st = "fleet"
        why = self.tree_is_the_commit(st)
        if why and not self.args.dry_run_allow:
            return self.stop(st, why)
        corpus = self.corpus()
        if corpus["problems"] and self.lifting_gates() and not self.args.dry_run_allow:
            return self.stop(st, f"the MEEP corpus: {'; '.join(corpus['problems'][:3])}")
        device = self.device_problems(st)
        if device and not self.args.dry_run_allow:
            return self.stop(st, "; ".join(device))
        if getattr(self.args, "chunk_manifest", None) is not None:
            import chunks as C  # noqa: PLC0415 - beside this file
            why = C.write_launch_record(self.results, self.args.chunk_index_data,
                                        self.args.chunk_manifest, self.repo)
            if why:
                return self.stop(st, why)
        self.root.mkdir(parents=True, exist_ok=True)
        env = self.fleet_env()
        environment = self.results / f"metal_environment_{self.args.stamp}"
        if not self.args.skip_environment:
            self.run_item(st, "metal_environment_start",
                          [self.py, str(self.g / "metal_environment.py"), "--write",
                           str(environment / "start.json")], self.base_env(), None, 600)
        gates = sorted(p.stem for p in self.g.glob("gate_metal_*.py"))
        ordered = [g for g in gates if g not in LONG]
        ordered += [g for g in LONG if g in gates]
        only = {s for s in self.args.only.split(",") if s}
        if only:
            ordered = [g for g in ordered if g in only]
        for i, name in enumerate(ordered, 1):
            family = name[len("gate_"):]
            directory = self.results / f"{family}_{self.args.stamp}"
            directory.mkdir(parents=True, exist_ok=True)
            out = directory if name == "gate_metal_whole_step" else directory / "gate.json"
            artifact = (directory / "whole_step.json") if name == "gate_metal_whole_step" else out
            self.say(st, f"[{i}/{len(ordered)}] {name} (expect ~{EXPECT.get(name, '?')} s)")
            self.run_item(st, name, [self.py, "-u", str(self.g / f"{name}.py"), "--out", str(out)],
                          env, artifact, self.limit(name))
        if not self.args.skip_environment:
            self.run_item(st, "metal_environment_end",
                          [self.py, str(self.g / "metal_environment.py"), "--write",
                           str(environment / "end.json")], self.base_env(), None, 600)
        (self.logs / ".done_fleet").touch()
        return 0

    def stage_route(self) -> int:
        st = "route"
        why = self.tree_is_the_commit(st)
        if why and not self.args.dry_run_allow:
            return self.stop(st, why)
        # THE CAMPAIGN NAME IS THE REGISTRY'S (meep_gpu/route_stamps.py), read in the
        # frozen checkout, and a name it refuses stops the stage: the stamp still the
        # placeholder the freeze replaces, or a campaign that has already run.
        names = subprocess.run(
            [self.py, "-c", ROUTE_NAMES_PROGRAM],
            cwd=str(self.repo), env={**self.base_env(), "PYTHONPATH": f"{self.repo}:{self.g}"},
            capture_output=True, text=True)
        try:
            facts = json.loads(names.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            facts = None
        if not isinstance(facts, dict) or not {"campaign", "refused"} <= set(facts):
            return self.stop(st, f"could not name the route campaign: {names.stderr[-300:]}")
        if facts["refused"] and not self.args.dry_run_allow:
            return self.stop(st, f"the route campaign {facts['campaign']} is refused: "
                                 f"{facts['refused']}")
        stamp = facts["campaign"][len("dispatch_metal_route_"):]
        run = self.results / facts["campaign"]
        if run.exists() and not (self.logs / "done" / "route_campaign.rc").is_file():
            return self.stop(st, f"{run} exists; route campaign names are one-shot")
        argv = ["zsh", str(self.g / "run_metal_dispatch_campaign.sh"), "--lanes",
                str(self.args.route_lanes), stamp]
        env = self.base_env()
        watchdog = route_watchdog(self.args.route_timeout_factor, self.args.route_min_timeout)
        self.say(st, f"route campaign {facts['campaign']} ({self.args.route_lanes} lane(s), held "
                     f"residency); watchdog {watchdog} s = max({self.args.route_min_timeout}, "
                     f"{self.args.route_timeout_factor} x {sum(ROUTE_EXPECT.values())} s measured), "
                     f"plus up to {self.args.slot_patience} s for slots")
        rc = self.run_item(st, "route_campaign", argv, env, None, watchdog)
        (self.logs / ".done_route").touch()
        self.say(st, f"route campaign exit {rc}; recut with recut_driver_dispatch_record.py "
                     f"--backend metal --run {facts['campaign']} on the bind host")
        return 0 if rc == 0 else 1

    def stage_status(self) -> int:
        st = "status"
        released, refused = [], []
        stamp_of = getattr(self.args, "stamp_of", None) or {}
        for script in sorted(self.g.glob("gate_metal_*.py")):
            family = script.stem[len("gate_"):]
            if stamp_of and script.stem not in stamp_of:
                if getattr(self.args, "chunk_manifest", None) is not None:
                    continue              # another chunk's gate
                refused.append((script.stem, "in no chunk of the index"))
                continue
            directory = self.results / f"{family}_{stamp_of.get(script.stem, self.args.stamp)}"
            artifact = next((directory / n for n in ("gate.json", "whole_step.json")
                             if (directory / n).is_file()), None)
            if artifact is None:
                refused.append((script.stem, "no artifact"))
            elif read_verdict(artifact):
                released.append(script.stem)
            else:
                refused.append((script.stem, "not released"))
        self.say(st, f"{len(released)} of {len(released) + len(refused)} Metal gate artifacts released")
        for name, why in refused:
            self.say(st, f"  NOT released: {name}: {why}")
        cited = subprocess.run(
            [self.py, "-c", "from meep_gpu import metal_dispatch as d; "
             "print(' '.join(sorted({g for _f, g in d.ARM_CERTIFICATION.values()})))"],
            cwd=str(self.repo), env=self.base_env(), capture_output=True, text=True).stdout.split()
        fams = [g[len("metal_"):-len("_device_gate")] for g in cited if g.endswith("_device_gate")]
        ok = [f for f in fams if f"gate_metal_{f}" in released]
        self.say(st, f"cited welds released: {len(ok)} of {len(cited)}")
        return 0 if not refused else 1


# ------------------------------------------------------------------ the lift cells -----
# THE ROUND'S SIDE OF THE LIFT LEGS. The rule (which interpreter a lift child runs under,
# which inputs a basis reads, which rows need what) is ``parity/meep_gpu/lift_host.py``,
# in the lifting gates' closures; what the round does with it is here, in none, so a
# change to the preflight re-runs no gate.
def lifting_kind(source: str) -> Optional[str]:
    """``"direct"`` for a gate that defines its own ``lift_basis``, ``"tail"`` for one
    built on ``hd_tail_gate_common`` (whose ``lift_basis`` takes the gate's ``PRODUCT``),
    None for a gate that launches no lift child. Read off the source, importing nothing:
    a gate that only NAMES a census (``gate_metal_complex_fused_hd_pair``, whose lift leg
    runs in process) is None."""
    body = ast.parse(source).body
    if any(isinstance(node, ast.FunctionDef) and node.name == "lift_basis" for node in body):
        return "direct"
    if any(isinstance(node, ast.Import)
           and any(alias.name == "hd_tail_gate_common" for alias in node.names)
           for node in body):
        return "tail"
    return None


def census_names(source: str) -> Optional[Tuple[str, str]]:
    """A gate's module-level ``CENSUS`` and ``SEAM_RECORD`` string constants, read off
    its source, or None. Every lifting gate names both (the tail gates hand them to
    their ``PRODUCT``)."""
    names: Dict[str, str] = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            names.update({target.id: node.value.value for target in node.targets
                          if isinstance(target, ast.Name)})
    if "CENSUS" in names and "SEAM_RECORD" in names:
        return names["CENSUS"], names["SEAM_RECORD"]
    return None


def census_reader(source: str) -> bool:
    """A gate that names the census and the seam record (:func:`census_names`) and launches
    no lift child (:func:`lifting_kind`): ``gate_metal_complex_fused_hd_pair``, whose
    ``cell_rows`` reads the same four files in process and requires them by name. The
    preflight checks its inputs and computes no basis for it."""
    return lifting_kind(source) is None and census_names(source) is not None


def lift_cells_child(argv: List[str]) -> int:
    """In a child under ``--python``: each named gate's lift basis, the inputs it lacks,
    and the rows this host cannot lift. ``argv``: harness directory, results directory,
    the gate stems as JSON. Torch is unimportable here (no gate's basis needs it, and a
    preflight must not touch the GPU). Prints one ``LIFT_CELLS=<json>`` line."""
    import importlib  # noqa: PLC0415
    import importlib.abc  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):  # noqa: D102 - finder API
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not imported by the lift preflight")
            return None

    sys.meta_path.insert(0, _NoTorch())
    harness, results, stems = Path(argv[0]), Path(argv[1]), json.loads(argv[2])
    for entry in (str(harness.parent.parent), str(harness)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import lift_host  # noqa: PLC0415

    gates: Dict[str, Dict[str, Any]] = {}
    needing: List[Dict[str, Any]] = []
    for stem in stems:
        source = (harness / f"{stem}.py").read_text(encoding="utf-8")
        entry: Dict[str, Any] = {"kind": lifting_kind(source)}
        # THE INPUTS ARE NAMED FIRST, off the source: a gate this child cannot import
        # still has its four files checked.
        names = census_names(source)
        if entry["kind"] is None:
            # A gate that reads the inputs in process: they are checked, nothing imported.
            if names:
                entry["inputs"] = lift_host.lift_inputs(*names)
                entry["missing"] = lift_host.missing_inputs(results, *names)
            gates[stem] = entry
            continue
        rows_of = None
        try:
            module = importlib.import_module(stem)
            if entry["kind"] == "tail":
                import hd_tail_gate_common as common  # noqa: PLC0415
                product = module.PRODUCT
                names = (product.census, product.seam_record)
                rows_of = (lambda product=product: common.lift_basis(product, results))
            else:
                names = (module.CENSUS, module.SEAM_RECORD)
                rows_of = (lambda module=module: module.lift_basis(results))
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported by name
            entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
        if names:
            entry["inputs"] = lift_host.lift_inputs(*names)
            entry["missing"] = lift_host.missing_inputs(results, *names)
        if rows_of is not None and names and not entry["missing"]:
            try:
                rows, _facts = rows_of()
                entry["rows"] = len(rows)
                entry["needs"] = {row["label"]: list(row.get("lift_needs") or ())
                                  for row in rows if row.get("lift_needs")}
                needing += [row for row in rows if row.get("lift_needs")]
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported by name
                entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
        gates[stem] = entry
    with tempfile.TemporaryDirectory() as scratch:
        host = lift_host.host_check(needing, environment=dict(os.environ), cwd=Path(scratch))
    for entry in gates.values():
        entry["unlifted"] = {label: host["unlifted_on_this_host"][label]
                             for label in entry.get("needs") or {}
                             if label in host["unlifted_on_this_host"]}
    print(LIFT_CELLS_PREFIX + json.dumps(
        {"gates": gates, "host": {key: host[key] for key in ("interpreter", "needs",
                                                              "unreadable")}},
        default=str), flush=True)
    return 0


def summarize_lift_cells(answer: Mapping[str, Any], stems: List[str],
                         results: Path) -> Tuple[List[str], List[str]]:
    """``(problems, notes)`` from :func:`lift_cells_child`'s answer. The one problem is
    an input that is absent or cannot be named; a basis that could not be computed here,
    an unlifted row and an unreadable capability probe are notes, because the gate's own
    lift leg reports each by name and none concerns the fleet's other gates."""
    problems: List[str] = []
    notes: List[str] = []
    gates = answer.get("gates") or {}
    total = unlifted = 0
    for stem in stems:
        entry = gates.get(stem)
        if entry is None or "missing" not in entry:
            problems.append(f"{stem}: its lift inputs could not be named"
                            + (f" ({entry['error']})" if entry and entry.get("error") else ""))
        elif entry["missing"]:
            problems.append(f"{stem}: lift input(s) absent under {results}: "
                            f"{', '.join(entry['missing'])}")
        elif not entry.get("kind"):
            notes.append(f"{stem}: its four lift inputs are present (it reads them in "
                         f"process and launches no lift child)")
        elif entry.get("error"):
            notes.append(f"WARNING {stem}: its four inputs are present, but its lift basis "
                         f"could not be computed here ({entry['error']}); its own lift leg "
                         f"will report it")
        else:
            rows, gone = int(entry.get("rows") or 0), entry.get("unlifted") or {}
            total, unlifted = total + rows, unlifted + len(gone)
            notes.append(f"{stem}: {rows} lift rows, {len(gone)} of them unlifted on this "
                         f"host")
            notes += [f"{stem}: {label}: UNLIFTED ON THIS HOST, {why}"
                      for label, why in sorted(gone.items())]
    host = answer.get("host") or {}
    if host.get("unreadable"):
        notes.append(f"WARNING: {host['unreadable']}; a gate whose rows declare a need "
                     f"will fail its lift leg by name")
    lifting = [stem for stem in stems if (gates.get(stem) or {}).get("kind")]
    notes.append(f"{total} lift rows over {len(lifting)} gates; {unlifted} of {total} "
                 f"unlifted on this host; needs probed under {host.get('interpreter')}: "
                 f"{host.get('needs') or {}}")
    return problems, notes


# ------------------------------------------------------------------ the MEEP corpus -----
def read_verdict(artifact: Optional[Path]) -> Optional[bool]:
    if artifact is None or not artifact.is_file():
        return None
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stated = [(payload.get(k) or {}).get("released") for k in ("canonical_verdict", "release")
              if isinstance(payload.get(k), dict)]
    stated = [v for v in stated if v is not None]
    if not stated and isinstance(payload.get("summary"), dict) and "passed" in payload["summary"]:
        stated = [bool(payload["summary"]["passed"])]
    return bool(stated) and all(stated)


def apply_chunk(args: argparse.Namespace) -> Optional[str]:
    """Resolve ``--chunk-index``/``--chunk`` onto ``args``: the chunk's stamp in place of
    ``--stamp`` and its gates as ``--only`` (narrowed further by a given ``--only``); for an
    index-wide status, each gate's chunk stamp (``stamp_of``). Returns why the arguments are
    refused, or None."""
    args.chunk_manifest = None
    args.stamp_of = {}
    if not args.chunk_index:
        if args.chunk:
            return "--chunk needs --chunk-index"
        if getattr(args, "new_architecture", False):
            return NEW_ARCHITECTURE_UNCHUNKED
        return None if args.stamp else "--stamp is required (or --chunk-index with --chunk)"
    import chunks as C  # noqa: PLC0415 - beside this file; chunks.py imports this module
    args.chunk_index_data = C.read_index(Path(args.chunk_index))
    if args.chunk_index_data["machine"] != "metal":
        return f"{args.chunk_index} is a {args.chunk_index_data['machine']} index"
    if getattr(args, "new_architecture", False) and args.stage == "route":
        return ("--new-architecture is a fleet mode: the route stage runs once the bind has "
                "added the architecture's runs and the ledger admits it")
    # STATUS READS RUNS, NOT THE PLAN: each gate under the newest launch record that names
    # it (a unit may be in several chunks), else under the plan's last chunk naming it.
    args.stamp_of = {gate: chunk["stamp"] for chunk in args.chunk_index_data["chunks"]
                     for gate in C.metal_gates(chunk)}
    results = (Path(args.results) if getattr(args, "results", "") else
               Path(getattr(args, "repo", None) or ".") / "parity" / "meep_gpu" / "results")
    for launch in C.launches(args.chunk_index_data, results):
        if not launch["problems"]:
            for unit in launch["record"].get("units") or ():
                if unit.startswith("metal/"):
                    args.stamp_of[unit[len("metal/"):]] = launch["stamp"]
    if not args.chunk:
        if args.stage != "status":
            return "--chunk-index without --chunk serves the status stage only"
        args.stamp = args.stamp or args.chunk_index_data["round"]
        # The round stamp names no root of a chunked round: the status log lives beside
        # the index, never under results/.
        args.logs = args.logs or str(Path(args.chunk_index).resolve().parent
                                     / "metal_status_logs")
        return None
    if args.stage == "route":
        return "the route stage is not chunked (one route stamp per round)"
    args.chunk_manifest = C.find_chunk(args.chunk_index_data, args.chunk)
    if args.stamp and args.stamp != args.chunk_manifest["stamp"]:
        return (f"--stamp {args.stamp} is not chunk {args.chunk}'s stamp "
                f"{args.chunk_manifest['stamp']}")
    args.stamp = args.chunk_manifest["stamp"]
    gates = C.metal_gates(args.chunk_manifest)
    args.stamp_of = {gate: args.stamp for gate in gates}
    wanted = {s for s in args.only.split(",") if s}
    args.only = ",".join(g for g in gates if not wanted or g in wanted) or "-"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=ORDER + ("all",))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stamp", default=None, help="e.g. <date>_10_g13s; new per round "
                    "(with --chunk-index, the chunk's stamp is used)")
    ap.add_argument("--chunk-index", default="",
                    help="a chunks.py index (metal): with --chunk, run only that chunk under "
                         "its own stamp; status without --chunk reads every chunk")
    ap.add_argument("--chunk", default="", help="the chunk id of --chunk-index (m<k>)")
    ap.add_argument("--results", default="")
    ap.add_argument("--logs", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--expect", default=None)
    ap.add_argument("--meep-source", default=os.environ.get("MGPU_SITE_MEEP_SOURCE", ""))
    ap.add_argument("--corpus-root", default="",
                    help="the MEEP corpus (python/ tree) the lift legs walk; default: the "
                         "one the reference environment's build keeps and records "
                         "(<prefix>/share/meep-gpu-build/reference_build.json). Verified "
                         "against that record's sha256 list either way")
    ap.add_argument("--new-architecture", action="store_true",
                    help="this Mac's GPU architecture is one the Metal ledger holds no live "
                         "run for (an Apple M3): a chunked fleet whose index was partitioned "
                         "with --new-device")
    ap.add_argument("--slot-wrapper", default="", help="e.g. run_in_slot.sh (slot v2)")
    ap.add_argument("--slot-log", default="")
    ap.add_argument("--classes", default="mac_gpu+mac_cpu")
    ap.add_argument("--slot-patience", type=int, default=6 * 3600,
                    help="seconds a wrapped item may wait for its slots on top of its watchdog")
    ap.add_argument("--disk-floor-kb", type=int, default=9 * 1024 * 1024 + 500_000)
    ap.add_argument("--route-lanes", type=int, default=3)
    ap.add_argument("--route-timeout-factor", type=int, default=ROUTE_TIMEOUT_FACTOR,
                    help="route stage: the campaign's watchdog is this many times the serial "
                         "sum of the legs' measured walls (ROUTE_EXPECT), and at least "
                         "--route-min-timeout")
    ap.add_argument("--route-min-timeout", type=int, default=ROUTE_MIN_TIMEOUT,
                    help="route stage: the shortest watchdog of the route campaign (s)")
    ap.add_argument("--only", default="", help="dry run: gate script stems to keep")
    ap.add_argument("--skip-environment", action="store_true",
                    help="dry run: no environment record")
    ap.add_argument("--dry-run-allow", action="store_true")
    args = ap.parse_args(argv)
    why = apply_chunk(args)
    if why:
        ap.error(why)
    lock = None
    rnd = Round(args)
    if args.chunk_manifest is not None and args.stage in ("preflight", "fleet", "all"):
        lock = rnd.logs / "LAUNCH.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            pid = (lock / "pid").read_text().strip() if (lock / "pid").is_file() else ""
            if pid.isdigit() and _alive(int(pid)):
                rnd.say("chunk", f"STOP: {lock} is held by pid {pid}: this chunk is running")
                return 3
            rnd.say("chunk", f"{lock} was left by pid {pid or '?'}, which is gone; taken over")
        (lock / "pid").write_text(f"{os.getpid()}\n")

    def on_signal(signum, _frame):
        rnd.say("signal", f"terminated by signal {signum}; stopping the running item")
        rnd.kill_child()
        (rnd.logs / "STOPPED").write_text(f"signal {signum}\n")
        sys.exit(143)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    stages = ("preflight", "fleet", "status") if args.stage == "all" else (args.stage,)
    try:
        rc = 0
        for stage in stages:
            if args.stage == "all" and (rnd.logs / f".done_{stage}").is_file():
                rnd.say(stage, "already finished; skipped")
                continue
            rnd.say(stage, "start")
            rc = getattr(rnd, f"stage_{stage}")()
            if rc != 0 and stage != "status":
                return rc
        # THE LAST STAGE'S CODE: for ``all`` the status stage's, non-zero when a gate of
        # the fleet did not release, so a round's exit code is its verdict.
        return rc
    finally:
        if lock is not None:
            for name in ("pid",):
                try:
                    (lock / name).unlink()
                except OSError:
                    pass
            try:
                lock.rmdir()
            except OSError:
                pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    sys.exit(main())
