#!/usr/bin/env python3
"""Run every Triton device gate SEQUENTIALLY on one physically empty GPU.

WHY A DRIVER AND NOT FOURTEEN SHELL SCRIPTS. The certified families each got a
bespoke ``run_triton_*_direct.sh``; the fourteen UNWELDED ones would need
fourteen more, and every one of them would re-implement the residency guard, the
subnormal-policy cache naming, the libcuda stub and the artifact readout — four
places to get one thing wrong, times fourteen. Worse, a shell fleet has no
uniform record: the campaign's answer to "which gates released?" would again be
recovered by reading fourteen differently shaped logs by eye.

WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT.

* It owns PLACEMENT (one physically empty GPU, re-verified before every gate, so
  a process that arrives mid-campaign stops the next gate rather than sharing a
  device with it), the ENVIRONMENT every gate needs on this box, and the RECORD.
* It owns NOTHING about a verdict. It runs the gate's own ``main`` in its own
  process, reads the artifact the gate wrote, and reports
  ``canonical_verdict`` — the gate's word, normalised by
  :mod:`gate_provenance`, never re-derived here. A driver that computed its own
  pass/fail would be a second opinion competing with the measurement.
* It WELDS NOTHING. ``fingerprints.json`` is not opened, let alone written. The
  campaign's output is a proposal packet; a human applies it.

PROCESS PER GATE, NOT ONE PROCESS. Every gate installs a subnormal policy at the
NVRTC seam, several import each other, and three write the expansion probe a
fourth consumes. In one process the first gate's install would silently decide
the tenth gate's verdict. Separate processes also mean a segfault costs one row.

PROGRESS (the progress-reporting rule). One flushed line per gate as it lands — name, exit
code, released, imported-digest count, elapsed — plus a heartbeat every
``--heartbeat`` seconds naming the running gate, its elapsed time and the last
line its own log emitted, so a long gate is distinguishable from a hang by
reading the driver log alone. Each row is appended to ``gates.jsonl`` and fsynced
as it lands, so an interrupted campaign keeps everything up to the failure.

Usage (from the staged tree's root)::

    nohup python -u parity/meep_gpu/drive_triton_weld_gates.py \\
        --gpu 7 --out-root "$RUN_ROOT/results/gates_$TAG" \\
        > "$RUN_ROOT/logs/driver_$TAG.log" 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]

#: Devices this campaign may never take, whatever ``nvidia-smi`` reports about
#: them. They are set aside on the host this driver ran on; "idle right now" is not
#: permission, and a residency check that passes on one of them is answering the
#: wrong question.
#:
#: A DEFAULT, NOT A FACT ABOUT EVERY HOST, since 2026-09-30. Certifying a second
#: compute capability means running this fleet on a card this driver has never seen,
#: on a box where 4 and 5 may be the only free devices — a set hardcoded here ties
#: the round to one machine. It stays the DEFAULT rather than becoming empty because
#: every standing invocation on the current GPU host omits the flag, and a flag whose
#: absence newly permits a set-aside device would be a silent change of who this
#: driver may disturb. A different host passes its own set, or ``--forbidden-gpus ''``
#: to declare that nothing is set aside.
DEFAULT_FORBIDDEN_GPUS = frozenset({4, 5})

#: A device carrying more than this with no listed compute process is still
#: treated as occupied. Matches the direct runners' guard: an unattributable
#: resident is still a resident.
IDLE_MIB_CEILING = 64

EXIT_CANNOT_CERTIFY = 75


class Gate:
    """One gate: how to invoke it, and where its artifact lands.

    The argv is BUILT rather than stored because three of these families need a
    path that only exists at run time (an out directory, or the expansion probe
    a previous gate in the same campaign measured). ``artifact`` is stated
    separately because the fleet is not uniform: most gates take ``--out`` and
    write exactly there, ``no_pml`` takes ``--out-dir`` and writes ``gate.json``
    inside it, and guessing which from the flag name is how a runner reports
    "artifact missing" about a gate that wrote one.
    """

    def __init__(self, name: str, script: str, argv, artifact: str,
                 timeout: int = 5400, writes_probe: Optional[str] = None,
                 note: str = "", policy_flag: Optional[str] = None) -> None:
        self.name = name
        self.script = script
        self._argv = argv
        self._artifact = artifact
        self.timeout = timeout
        self.writes_probe = writes_probe
        self.note = note
        #: The flag this gate takes the campaign's subnormal policy on, when
        #: its install is ARGUMENT-GATED rather than unconditional. Threaded
        #: rather than hardcoded: ``--policy`` also accepts ``flush``, and a
        #: literal "keep" here would be refused at gate startup on a flush
        #: campaign while silently disagreeing with the environment the
        #: driver already set for every other gate.
        self.policy_flag = policy_flag

    def argv(self, out_dir: Path, probe: Optional[str],
             policy: str) -> List[str]:
        built = [str(part) for part in self._argv(out_dir, probe)]
        if self.policy_flag is not None:
            built += [self.policy_flag, policy]
        return built

    def artifact(self, out_dir: Path) -> Path:
        return out_dir / self._artifact

    def probe_path(self, out_dir: Path) -> Optional[Path]:
        return None if self.writes_probe is None else out_dir / self.writes_probe


def _gates() -> Tuple[Gate, ...]:
    """Every Triton gate this runner knows how to invoke.

    Tranche 1 (the first fourteen) were the gates with no ``status: PASS`` weld
    on 2026-08-19. Tranche 2 (2026-08-28) adds the twelve the routing re-gate
    needed; see the comment at its head for why.

    ORDER IS LOAD-BEARING at exactly one place: ``complex`` runs before
    ``cylindrical_complex`` because the latter binds its ``EXPANSION`` constexpr
    from a MEASURED probe it does not cut itself, and a probe measured in this
    campaign, on this device, under the policy this campaign installs is a
    stronger provenance than one carried in from another run. The fallback is
    still there (``--probe``) and is recorded per row when used.
    """
    return (
        # Cheap and self-contained first: a placement or environment mistake
        # should surface in a minute, not ninety.
        Gate("unified_expansion", "gate_triton_unified_expansion.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=1800,
             note="cuts one expansion record for every complex family"),
        Gate("complex", "gate_triton_complex.py",
             lambda out, probe: ["--out", out / "gate.json",
                                 "--probe-artifact", out / "probe.json"],
             "gate.json", timeout=7200, writes_probe="probe.json",
             note="also measures the expansion probe the cylindrical-complex "
                  "gate consumes"),
        Gate("fused_electric", "gate_triton_fused_electric.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=3600),
        Gate("no_pml", "gate_triton_no_pml.py",
             lambda out, probe: ["--out-dir", out], "gate.json", timeout=3600,
             note="--out-dir, not --out; the artifact is gate.json inside it"),
        Gate("no_pml_constitutive", "gate_triton_no_pml_constitutive.py",
             lambda out, probe: ["--out", out / "gate.json",
                                 "--backend", "cupy"],
             "gate.json", timeout=3600,
             note="--backend cupy: the numpy default certifies nothing on a "
                  "device host"),
        Gate("conductivity", "gate_triton_conductivity.py",
             # --subnormal-policy IS REQUIRED HERE and nowhere else in this list. This
             # gate is the only one that declares the flag, and its default is "none"
             # (gate_triton_conductivity.py:2132) -- deliberately, so a re-run reproduces
             # the original numbers. `install_policy("none")` returns the not_installed
             # record and never reaches install_ftz_strip, so the 2026-08-23 campaign
             # earned this gate's PASS under a HALF-APPLIED policy: CuPy flushing at the
             # NVRTC seam while Triton keeps subnormals natively. The artifact stamped
             # {"installed": false, "policy": "not_installed"}, which cannot be welded
             # beside the seven that stamped ieee_keep_ftz_stripped. The driver owns the
             # environment every gate needs, so it passes the policy it was asked for.
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=3600, policy_flag="--subnormal-policy"),
        Gate("symmetry", "gate_triton_symmetry.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("cylindrical", "gate_triton_cylindrical.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("offdiag", "gate_triton_offdiag.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("nonlinear", "gate_triton_nonlinear.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("bfast", "gate_triton_bfast.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("special_kz", "gate_triton_special_kz.py",
             lambda out, probe: ["--out", out / "gate.json",
                                 "--probe-artifact", out / "probe.json"],
             "gate.json", timeout=7200, writes_probe="probe.json"),
        Gate("folded_complex", "gate_triton_folded_complex.py",
             lambda out, probe: ["--out", out / "gate.json",
                                 "--probe-artifact", out / "probe.json"],
             "gate.json", timeout=10800, writes_probe="probe.json"),
        Gate("cylindrical_complex", "gate_triton_cylindrical_complex.py",
             lambda out, probe: ["--out", out / "gate.json", "--probe-artifact",
                                 probe or ""],
             "gate.json", timeout=10800,
             note="consumes a measured expansion probe; refuses without one"),

        # ------------------------------------------------- TRANCHE 2, 2026-08-28
        # The twelve families the 2026-08-19 tranche did not need and this one does.
        # WHY THEY ARE HERE NOW: the deposit-repair routing edited triton_kernels/
        # launch.py, which 28 of the 30 PASS welds pin, and a weld is owed a RE-RUN
        # rather than an edited record. The 2026-08-27 campaign covered fourteen of
        # them; rebind_triton_welds.py then reported the other eighteen as UNCOVERED
        # -- "no row in CAMPAIGN_DIRS -- needs a device re-run, not a guessed
        # directory" -- which is this list plus the six fused-pair families that
        # ship no gate script of their own.
        #
        # Every one of these takes --out as a FILE PATH, not a directory: each does
        # out.parent.mkdir / os.path.dirname(out) and writes its jsonl beside it
        # (gate_triton_complex_ade.py:365-368, gate_triton_fused_ade_chain.py:1540-1542).
        # That is checked per script rather than assumed from the flag name, which is
        # the mistake the Gate docstring above already warns about.
        Gate("complex_ade", "gate_triton_complex_ade.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("complex_no_pml_curl", "gate_triton_complex_no_pml_curl.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("complex_no_pml_conductive", "gate_triton_complex_no_pml_conductive.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("complex_no_pml_stored_e", "gate_triton_complex_no_pml_stored_e.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("complex_offdiag", "gate_triton_complex_offdiag.py",
             lambda out, probe: ["--out", out / "gate.json", "--probe", probe or ""],
             "gate.json", timeout=7200,
             note="REQUIRES THE UNIFIED_EXPANSION artifact, not the complex probe. "
                  "Measured 2026-08-28: its folded products bind the EXTENDED pattern "
                  "set (the base four plus c8_mul_c8_parity_coefficient_left) and the "
                  "complex gate's probe licenses only the base four, so the shipped "
                  "builder refuses every folded product with 'the expansion probe "
                  "artifact does not license an EXPANSION over the EXTENDED pattern "
                  "set'. Pass --probe <campaign>/unified_expansion/gate.json.\n\n"
                  "THIS NOTE SAID folded_complex/probe.json UNTIL 2026-09-17, and the "
                  "artifacts say otherwise -- which is why a probe question is settled "
                  "by reading the last run that RELEASED and what it recorded as its "
                  "probe, not by reading prose. On record: "
                  "triton_fleet_2026-09-15_gapclose passed "
                  "unified_expansion/gate.json and released; "
                  "triton_fleet_2026-09-15_ledger passed complex/probe.json and did "
                  "not; triton_fleet_2026-09-17_allpaths passed complex/probe.json and "
                  "did not, and the tranche that followed passed "
                  "unified_expansion/gate.json and released. The note's diagnosis was "
                  "right and its prescription named the wrong artifact.\n\n"
                  "Also note --probe takes an ABSOLUTE path: the gate child runs from "
                  "a different cwd than the driver, and a relative one dies with "
                  "FileNotFoundError"),
        Gate("no_pml_conductive", "gate_triton_no_pml_conductive.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("no_pml_stored_e", "gate_triton_no_pml_stored_e.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=5400),
        Gate("folded_offdiag", "gate_triton_folded_offdiag.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("folded_offdiag_dispersive", "gate_triton_folded_offdiag_dispersive.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200),
        Gate("fused_ade_chain", "gate_triton_fused_ade_chain.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200, policy_flag="--subnormal-policy"),
        Gate("complex_fused_ade_chain", "gate_triton_complex_fused_ade_chain.py",
             lambda out, probe: ["--out", out / "gate.json", "--probe", probe or ""],
             "gate.json", timeout=7200, policy_flag="--subnormal-policy",
             note="REQUIRES the UNIFIED EXPANSION RECORD, not a per-family probe. "
                  "Measured 2026-08-28: this family needs a licence for BOTH halves, and "
                  "only unified_expansion/gate.json carries one -- its unified_licenses "
                  "block holds {base, folded_parity, special_kz_beta, complex_ade}, and "
                  "the ADE half reads complex_ade. With the complex gate's probe (or with "
                  "none) the gate reports 'expansion: E half=1 ADE half=None shared=None' "
                  "and refuses at one_arm_for_both_halves. Pass "
                  "--probe <campaign>/unified_expansion/gate.json, absolute"),
        # 2026-09-10: the THIRD ADE chain — the folded off-diagonal dispersive
        # E->P weld, the last buildable E->P cell on the Triton board. The gate
        # NAME is the PRODUCT name and must stay so: this driver names the
        # campaign subdirectory after Gate.name, and seed_triton_welds derives
        # the ledger key `triton_<subdirectory>_device_gate` from it.
        Gate("folded_offdiag_fused_ade_chain",
             "gate_triton_folded_offdiag_fused_ade_chain.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200, policy_flag="--subnormal-policy",
             note="--out is a FILE. NO expansion probe: this family's E half is "
                  "real float32 storage, so nothing in its verdict reads an "
                  "expansion record. It records environment.compute_capability "
                  "and environment.device so seed_triton_welds can mint its "
                  "entry, and its curated source_sha256 names BOTH folded "
                  "off-diagonal modules — the kernel executes JIT helpers out of "
                  "them, which is why the board binds this family by its gate "
                  "rather than by a fingerprint"),
        # ------------------------------------------------- TRANCHE 3, 2026-08-30
        # The two D->E FUSED PRODUCTS the fusion round built. Neither is a re-gate:
        # each is a NEW family's FIRST device run, and each takes ``--out`` as a
        # DIRECTORY (checked per script, not assumed from the flag name — both do
        # ``os.makedirs(out)`` and write ``gate.json`` inside it).
        #
        # BOTH CONSUME THE KEEP-CUT EXPANSION PROBE and install the policy
        # themselves before the first device compile, so they take the campaign's
        # ``--subnormal-policy`` rather than relying on the environment.
        Gate("cylindrical_fused_electric_pair",
             "probe_triton_cylindrical_fused_electric_pair.py",
             lambda out, probe: (["--out", out]
                                 + (["--probe-artifact", probe] if probe else [])),
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. Consumes a keep-cut expansion probe; "
                  "falls back to the in-tree candidates when the campaign has "
                  "measured none. Its carry legs need the deposit repair, so a "
                  "tree without deposit_repair.LeadingRepairPlan refuses before "
                  "the first device leg"),
        # 2026-09-04: the MAGNETIC twin of the cylindrical complex pair and the
        # family's COMPOSITION probe, added to the driver when the family grew its
        # m = 0 arm (``cylindrical_complex.M_ZERO``) and every artifact that pins
        # ``cylindrical_complex.py`` was owed a re-run. Neither takes
        # ``--subnormal-policy``: the magnetic probe installs keep AND flush
        # itself and reports the two cuts separately; the composition probe
        # installs the strip through gate_triton_complex and reads the policy
        # from the environment this driver sets.
        Gate("cylindrical_fused_magnetic_pair",
             "probe_triton_cylindrical_fused_magnetic_pair.py",
             lambda out, probe: (["--out", out]
                                 + (["--probe-artifact", probe] if probe else [])),
             "gate.json", timeout=10800,
             note="--out is a DIRECTORY. Consumes a keep-cut expansion probe and "
                  "installs both policies itself (no --subnormal-policy flag)"),
        Gate("cylindrical_complex_composition",
             "probe_triton_cylindrical_complex_composition.py",
             lambda out, probe: ["--out", out / "gate.json", "--probe-artifact",
                                 probe or ""],
             "gate.json", timeout=7200,
             note="consumes a measured expansion probe; refuses without one. NO "
                  "--subnormal-policy flag: the strip is installed through "
                  "gate_triton_complex and resolved from the environment"),
        Gate("complex_conductive_fused_pair",
             "probe_triton_complex_conductive_fused_pair.py",
             lambda out, probe: (["--out", out]
                                 + (["--probe-artifact", probe] if probe else [])),
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. No-PML family: it installs NO absorber "
                  "and its own predicate refuses one, so a fixture that set a PML "
                  "up would report a family failure that was really a fixture "
                  "failure"),
        # ------------------------------------------------- TRANCHE 4, 2026-08-31
        # The FOLDED DISPERSIVE D->E weld — the largest remaining unbuilt cell on
        # the Triton board, and a NEW family's FIRST device run rather than a
        # re-gate. Like the tranche-3 pair above it takes ``--out`` as a DIRECTORY.
        Gate("folded_dispersive_fused_pair",
             "probe_triton_folded_dispersive_fused_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes no "
                  "expansion probe at all. Its carry legs need the deposit repair, "
                  "so a tree without deposit_repair.LeadingRepairPlan refuses before "
                  "the first device leg; and its corpus leg refuses a staged tree "
                  "that does not carry the 186-row Triton census AND the fusion "
                  "board it cross-checks its cell against"),
        # The Dcyl m = 0 D->E weld — the electric twin of the released
        # ``cylindrical_real_fused_magnetic_pair``, and the first run to execute any
        # of that kernel's ``BACKWARD == 1`` arms. ``--out`` is a DIRECTORY.
        Gate("cylindrical_real_fused_electric_pair",
             "probe_triton_cylindrical_real_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage at m = 0, so it consumes no "
                  "expansion probe. Its carry legs need the deposit repair, and its "
                  "corpus leg refuses a staged tree that does not carry the 186-row "
                  "Triton census AND the fusion board it cross-checks against. A "
                  "CuPy bump is a correctness event for this family: everything "
                  "downstream of the radial prefix scan is bit-identical only for a "
                  "given cupy.cumsum summation order, and the artifact records the "
                  "version"),
        # The REAL-BETA D->E weld — the ELECTRIC twin of the released
        # ``beta_fused_magnetic_pair``, and a cell that magnetic twin's own docstring
        # wrote off as worth zero before ``deposit_repair`` existed. ``--out`` is a
        # DIRECTORY.
        Gate("beta_fused_electric_pair",
             "probe_triton_beta_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes no "
                  "expansion probe. Its carry legs need the deposit repair, so a "
                  "tree without deposit_repair.LeadingRepairPlan refuses before the "
                  "first device leg; its corpus leg refuses a staged tree that does "
                  "not carry the 186-row Triton census AND the fusion board it "
                  "cross-checks its cell against; and its REDUCTION leg needs "
                  "launch.plan_fused_pair_from_arrays, because "
                  "coverage.fused_pair_coverage refuses a beta run by name and the "
                  "engine route would return None"),
        # The TWO FOLDED-BETA welds — the last "not built" cells on the
        # 2026-08-31 plainrepair7 board, one corpus row driving both
        # (tests:TestSpecialKz.test_eigsrc_kz_1_real_imag, which injects into
        # BOTH seams). ``--out`` is a DIRECTORY for each.
        Gate("folded_beta_fused_electric_pair",
             "probe_triton_folded_beta_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes "
                  "no expansion probe. Its carry legs need the deposit repair's "
                  "FOLD closure (deposit_repair.repair_cells), so a tree whose "
                  "repair cannot image a deposit through the mirror fills "
                  "refuses before the first device leg; its corpus leg refuses "
                  "a staged tree that does not carry the 186-row plainrepair "
                  "census AND the plainrepair7 board it cross-checks against; "
                  "and its REDUCTION leg needs "
                  "folded_fused_pair.plan_folded_fused_pair_from_arrays, "
                  "because the plain folded predicate refuses a beta run by "
                  "name and the engine route would return None"),
        Gate("folded_beta_fused_magnetic_pair",
             "probe_triton_folded_beta_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. The MAGNETIC twin: the same corpus "
                  "row's TWO magnetic sources land in this seam, so its carry "
                  "legs inject magnetically and its REDUCTION leg needs "
                  "folded_fused_magnetic_pair."
                  "plan_folded_fused_magnetic_pair_from_arrays. REAL storage "
                  "throughout; same census/board preconditions as the electric "
                  "twin"),
        # The NO-ABSORBER stored-E D->E weld — the FIRST product on
        # ``deposit_repair.PLAIN_PATH``, the second repair, which inverts
        # ``update_E``'s plain overwrite rather than the split-field recurrence every
        # other family here inverts. ``--out`` is a DIRECTORY.
        Gate("no_pml_fused_electric_pair",
             "probe_triton_no_pml_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes no "
                  "expansion probe. It installs NO absorber and its own predicate "
                  "refuses one, so a fixture that set a PML up would report a family "
                  "failure that was really a fixture failure. Its carry legs need the "
                  "deposit repair AND its second path: a tree whose "
                  "deposit_repair has no PLAIN_PATH refuses in the no-device legs, "
                  "before any GPU time is spent. Its PRICED REFUSAL leg deliberately "
                  "steps around the shipped predicate to measure the one "
                  "configuration the product refuses, and REQUIRES it to diverge"),
        # The BFAST D->E weld — the ELECTRIC twin of the released
        # ``bfast_fused_magnetic_pair``. ``--out`` is a DIRECTORY.
        Gate("bfast_fused_electric_pair",
             "probe_triton_bfast_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes no "
                  "expansion probe. Its cases are 3-D by necessity: BFAST's k1/k2 "
                  "are cross-gated on grid.is_invariant, so a 2-D cell zeroes two "
                  "of the six words. Its carry legs need the deposit repair; its "
                  "corpus leg refuses a staged tree without the 186-row census and "
                  "the fusion board; and its REDUCTION leg needs "
                  "launch.plan_fused_pair_from_arrays, because "
                  "coverage.fused_pair_coverage refuses a BFAST run by name"),
        # The CONDUCTIVE PML D->E weld -- the last buildable non-folded cell on
        # this seam. ``--out`` is a DIRECTORY.
        Gate("conductive_fused_electric_pair",
             "probe_triton_conductive_fused_electric_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=10800, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. REAL storage throughout, so it consumes no "
                  "expansion probe. Its cases are 2-D and one of them is MIXED (one "
                  "lossy component beside two lossless), which is the only shape "
                  "that measures the COND == 0 arm beside a live COND == 1 arm in "
                  "the SAME launch. Its carry legs need the deposit repair; its "
                  "corpus leg refuses a staged tree without the 186-row census and "
                  "the fusion board; and its REDUCTION leg needs "
                  "launch.plan_fused_pair_from_arrays, because "
                  "coverage.fused_pair_coverage refuses a conductive run by name "
                  "and the engine route would return None"),
        # The FOLDED COMPLEX D->E weld — the largest remaining unbuilt cell on the
        # 2026-08-31 board, and a NEW family's FIRST device run rather than a re-gate.
        # ``--out`` is a DIRECTORY.
        Gate("folded_complex_fused_pair",
             "probe_triton_folded_complex_fused_pair.py",
             lambda out, probe: ["--out", out],
             "gate.json", timeout=14400, policy_flag="--subnormal-policy",
             note="--out is a DIRECTORY. COMPLEX storage, so it consumes an "
                  "EXPANSION probe and resolves it ITSELF through "
                  "complex_fields.load_expansion_probe — the record must license the "
                  "EXTENDED pattern set (the base four plus "
                  "c8_mul_c8_parity_coefficient_left), which is what "
                  "results/unified_expansion_2026-08-27/keep/gate.json carries and "
                  "what no per-family probe does. Its carry legs need the deposit "
                  "repair, so a tree without deposit_repair.LeadingRepairPlan "
                  "refuses before the first device leg; and its corpus leg refuses a "
                  "staged tree that does not carry the 186-row Triton census AND the "
                  "fusion board it cross-checks its cell against"),
        Gate("fused_offdiag_electric", "gate_triton_fused_offdiag_electric.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200, policy_flag="--subnormal-policy",
             note="PASSES but reports released=None, and that is a READER gap rather "
                  "than a refusal. It writes verdict.clauses -- a clause->bool dict -- "
                  "and the shared normaliser (gate_provenance.py read_verdict) recognises "
                  "verdict.{pass,passed,released,certified}, so it stamps "
                  "canonical_verdict.unreadable = 'no recognised verdict shape in this "
                  "payload; UNREADABLE is not the same claim as refused'. Every clause in "
                  "its 2026-08-28 run was true. Do not read this as a failing gate, and "
                  "do not weld it until the reader is widened"),
        # ------------------------------------------------- TRANCHE 5, 2026-09-02
        # THE SETTLEMENT TRANCHE. Tranches 1-4 leave 19 of the 40 drifted ledger
        # entries undischargeable, because `rebind_triton_welds.CAMPAIGN_DIRS`
        # points them at directories no gate above produces: 9 at `probe_triton_*`
        # runs, and 10 at records with no row at all. MEASURED 2026-09-02 against
        # launch.py@48845c8c / driver.py@8cce1c5b / __init__.py@87f8dee2:
        # 40 of 42 ledger entries carrying source_sha256 are in drift, and the
        # tranches above reach 21 of them.
        #
        # WHY EVERY ONE OF THESE IS A RE-RUN AND NOT A WAIVER. The comment-only
        # exemption that took the CUDA track to ZERO re-gates does not exist here:
        # `device_identity.weld_survives_edit` needs a `code_sha256` or
        # `device_sha256` for the file it is asked about, and the probe artifacts
        # these rows bind record ONLY `source_sha256`. The rule then returns False
        # by its "cannot be established -> refuse" clause. Measured over the board's
        # 35 credited products: 16 moved-file instances CODE-MOVED, 54 NO-BASIS,
        # ZERO comment-only. There is no re-run list to shrink.
        #
        # `--out` IS A FILE FOR EVERY ROW BELOW, which is why they all pass
        # `out / "gate.json"` rather than the bare `out` the tranche-3/4 rows use.
        # Checked per script, not assumed: 15 of the 17 fused-pair probes call
        # `save(payload, args.out)` (save() makes the PARENT directory), and the two
        # `folded_beta_complex_*` probes are dual-mode -- `if out.endswith(".json"):
        # artifact = out` -- so the same argument is correct for them too. The five
        # composition probes take a bare `--out <file>` and nothing else.
        #
        # THE COMPOSITION ROWS TAKE NO `--subnormal-policy`: none of the five
        # declares the flag, so they read the policy from the environment the runner
        # exports (the CUPY_CACHE_DIR "ftz_stripped" token). Passing a flag they do
        # not define would abort them at argparse.
        #
        # TWO OF THEM IMPORT `cases`, which lives only under results/*/scripts/:
        # `conductivity_composition` and `engine_route`. That is the same gap that
        # failed `no_pml` in the 2026-09-02_foldedrelease campaign -- its staged
        # copy and the runner's PYTHONPATH edit are both mtime 03:59:13 while the
        # gate ran at 03:48, so the fix post-dated the failure it was written for.
        # The driver APPENDS the inherited PYTHONPATH (see _gate_env below), so a
        # runner that exports the staged scripts directory covers all three.
        Gate("complex_fused_electric_pair",
             "probe_triton_complex_fused_electric_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="binds triton_complex_fused_electric_pair_device_gate; the "
                  "CAMPAIGN_DIRS row already names this directory"),
        Gate("probe_triton_complex_fused_magnetic_pair",
             "probe_triton_complex_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="DIRECTORY NAME IS THE CAMPAIGN_DIRS ROW, probe_triton_ prefix and "
                  "all -- the rebind maps the ledger key to this exact string, and a "
                  "shorter name would fail closed. Discharges the two red "
                  "test_triton_complex_fused_magnetic_pair tests"),
        Gate("probe_triton_cylindrical_real_fused_magnetic_pair",
             "probe_triton_cylindrical_real_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("probe_triton_dispersive_fused_pair",
             "probe_triton_dispersive_fused_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("probe_triton_folded_fused_magnetic_pair",
             "probe_triton_folded_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="the EXTERNAL_DRIFT_AT_CUT_TIME row on the board -- a concurrent "
                  "change edited its module and probe on 2026-08-20. This run "
                  "re-binds it to the bytes the tree ships now"),
        Gate("probe_triton_folded_fused_pair",
             "probe_triton_folded_fused_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("probe_triton_fused_dispersive_chain",
             "probe_triton_fused_dispersive_chain.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="--out is REQUIRED on this script (no default)"),
        Gate("probe_triton_folded_deposit_closure",
             "probe_triton_folded_deposit_closure.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("probe_triton_fused_ade_state",
             "probe_triton_fused_ade_state.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        # The ten GATE_BOUND board artifacts the tranches above do not produce.
        # Each is the SAME PROGRAM that cut the artifact the board cites today, so
        # re-pointing those rows is a fresh run of the same measurement rather than
        # the stem-match hazard rebind_triton_welds.CAMPAIGN_DIRS warns about.
        Gate("beta_fused_magnetic_pair",
             "probe_triton_beta_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("bfast_fused_magnetic_pair",
             "probe_triton_bfast_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("complex_beta_fused_electric_pair",
             "probe_triton_complex_beta_fused_electric_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("complex_beta_fused_magnetic_pair",
             "probe_triton_complex_beta_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        Gate("folded_beta_complex_fused_magnetic_pair",
             "probe_triton_folded_beta_complex_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="dual-mode --out: a path ending .json is used as the artifact"),
        Gate("folded_beta_complex_fused_pair",
             "probe_triton_folded_beta_complex_fused_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="dual-mode --out: a path ending .json is used as the artifact"),
        Gate("probe_triton_folded_complex_fused_magnetic_pair",
             "probe_triton_folded_complex_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy",
             note="ONE artifact backs TWO board products -- "
                  "folded_complex_fused_magnetic_pair and its _offdiag row. "
                  "Discharges the red test_triton_folded_complex_fused_magnetic_pair"),
        Gate("nonlinear_fused_magnetic_pair",
             "probe_triton_nonlinear_fused_magnetic_pair.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=10800, policy_flag="--subnormal-policy"),
        # THE COMPOSITION RECORDS. No CAMPAIGN_DIRS row exists for any of these, so
        # the rebind fails closed on them until one is typed. They are here because
        # two of the ten red tests name them directly: the no-absorber dispatch
        # contract fails on conductivity_composition_gate (__init__.py 017f710e ->
        # 87f8dee2) and test_triton_cylindrical_composition fails on
        # cylindrical_composition_gate (launch.py f3031c82 -> 48845c8c).
        #
        # NOT INCLUDED, DELIBERATELY: no_pml_composition_gate and
        # fused_electric_gate. Both are cut by scripts this driver ALREADY runs
        # (gate_triton_no_pml.py, gate_triton_fused_electric.py) but under
        # COMPOSITION-SPECIFIC arguments, so pointing their ledger rows at the
        # `no_pml` / `fused_electric` directories would bind a record to a run that
        # measured something else -- silently, because the digests would all verify.
        # They need their own invocation, and inventing one here without reading
        # what the composition record claims would be the same mistake in a
        # different place.
        Gate("conductivity_composition",
             "probe_triton_conductivity_composition.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200,
             note="NO --subnormal-policy flag on this script; it reads the policy "
                  "from the environment. IMPORTS `cases` -- needs results/*/scripts "
                  "on PYTHONPATH"),
        Gate("cylindrical_composition",
             "probe_triton_cylindrical_composition.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200,
             note="NO --subnormal-policy flag. The still_owed note names this run as "
                  "what moves source_sha256 and host_sha256 TOGETHER, the coupling "
                  "test_triton_kernels.py:388-390 enforces"),
        Gate("symmetry_composition",
             "probe_triton_symmetry_composition.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200,
             note="NO --subnormal-policy flag"),
        Gate("source_seams",
             "probe_triton_source_seams.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200,
             note="NO --subnormal-policy flag"),
        Gate("engine_route",
             "probe_triton_engine_route.py",
             lambda out, probe: ["--out", out / "gate.json"], "gate.json",
             timeout=7200,
             note="NO --subnormal-policy flag. Cuts the dispersive_composition_gate "
                  "record. IMPORTS `cases` -- needs results/*/scripts on PYTHONPATH"),
    )


GATES = {gate.name: gate for gate in _gates()}


def log(message: str) -> None:
    print(message, flush=True)


def _nvidia_smi(args: Sequence[str]) -> str:
    """Query NVML with ``CUDA_VISIBLE_DEVICES`` SCRUBBED from the environment.

    The pinning this driver applies to the gate is inherited by anything it
    spawns, and a residency question asked through a mask is not the question:
    physical index 7 would either be invisible or answer for whatever the mask
    renamed to 0.
    """
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(["nvidia-smi", *args], capture_output=True,
                                   text=True, env=env, check=False)
    except OSError as exc:
        # NO NVIDIA-SMI IS NOT AN EMPTY DEVICE. The caller reads an unparseable
        # answer as "occupied" and refuses, which is the right way for a
        # placement guard to fail on a host that cannot answer the question.
        return f"nvidia-smi unavailable: {exc!r}"
    return completed.stdout.strip()


def device_is_empty(index: int, forbidden: frozenset) -> Tuple[bool, str]:
    """Is this PHYSICAL device free right now, and what says so.

    ``forbidden`` is threaded rather than read from a module constant so that one
    campaign's set-aside devices are the ones it was told about; see
    :data:`DEFAULT_FORBIDDEN_GPUS`.
    """
    if index in forbidden:
        return False, (f"GPU {index} is set aside on this host "
                       f"(--forbidden-gpus {sorted(forbidden)}) and is never "
                       f"available to this campaign")
    resident = _nvidia_smi(["--id", str(index), "--query-compute-apps=pid,"
                            "process_name,used_memory", "--format=csv,noheader"])
    if resident:
        return False, f"GPU {index} carries a compute process: {resident}"
    used = _nvidia_smi(["--id", str(index), "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"])
    try:
        used_mib = int(used.splitlines()[0])
    except (ValueError, IndexError):
        return False, f"GPU {index} memory unreadable: {used!r}"
    if used_mib > IDLE_MIB_CEILING:
        return False, (f"GPU {index} reports {used_mib} MiB used with no listed "
                       f"process; an unattributable resident is still a resident")
    return True, f"GPU {index} empty: {used_mib} MiB, no compute process"


def wait_for_device(index: int, patience_s: int, *, forbidden: frozenset,
                    poll_s: int = 60) -> Tuple[bool, str]:
    """Block until the device is empty, or give up and REFUSE — never share."""
    deadline = time.time() + patience_s
    empty, why = device_is_empty(index, forbidden)
    while not empty and time.time() < deadline and index not in forbidden:
        log(f"[wait] {why}; re-checking in {poll_s}s "
            f"({int(deadline - time.time())}s of patience left)")
        time.sleep(poll_s)
        empty, why = device_is_empty(index, forbidden)
    return empty, why


def gate_environment(gpu: int, cache_root: Path, gate_name: str,
                     policy: str) -> Dict[str, str]:
    """The environment ONE gate runs in on this box.

    Every entry is here because its absence has already cost a run:

    * ``MEEP_GPU_SUBNORMAL_POLICY`` — THE CAMPAIGN'S POLICY, NAMED. Without it
      ``install_ftz_strip`` resolves ``match_meep`` by measuring the FPU that
      MEEP's own initialization left behind, which on this x86 host is FLUSH;
      every existing weld in ``fingerprints.json`` records ``keep``, so an
      unnamed run would quietly cut records under the other policy and pair them
      with a keep expansion probe — the broken comparison the campaign brief
      names. Measured 2026-08-19, first launch: the unified-expansion and
      complex gates both died at startup ("refusing to install the 'flush'
      subnormal policy: CUPY_CACHE_DIR carries the keep-policy token") in 2.8 s,
      which is the guard doing its job. The name is not lost to the shell: the
      resolution is stamped into every artifact as ``requested`` /
      ``resolved_from: environment``.
    * ``TRITON_LIBCUDA_PATH`` + ``LD_LIBRARY_PATH`` — this box ships only
      ``libcuda.so.1`` and Triton links ``-lcuda``; without the stub directory
      every device leg fails at link time and the gate reports a clean skip.
    * ``CUPY_CACHE_DIR`` carrying the policy token — CuPy computes its cache key
      ABOVE the seam where the ``-ftz=true`` strip installs, so a directory
      shared with a flush run serves flushed binaries under a keep record's
      name. PRIVATE PER GATE here, one step beyond the per-run privacy the
      direct runners use, because this campaign runs fourteen policy installs
      back to back.
    * ``TRITON_CACHE_DIR`` fresh per gate — the mutation batteries rename the
      kernel precisely because the JIT cache can serve a stale binary to a
      renamed mutant, and a cache shared across gates would make that hazard
      depend on what an earlier FAMILY compiled.
    * single-threaded BLAS — the host halves of these gates are NumPy oracles on
      a many-core shared host; nothing here is timed, and grabbing every
      thread to compute an oracle is rude, not fast.

    TWO ENTRIES ARE HERE AHEAD OF THE FAILURE RATHER THAN BEHIND IT (2026-09-30),
    and that is the whole reason a family gate needs no certification switch:

    * ``MEEP_GPU_DISPATCH=0`` — PINNED OFF, never inherited. The gates and probes
      in this fleet reach ``plan_step`` directly and monkeypatch sub-steps, and the
      reference drivers several of them build are lifted with ``prefer_gpu=True``;
      with dispatch on by default, an ADMITTED capability turns the reference half
      of a comparison into a kernel run and the gate compares a kernel against
      itself. Every test oracle in the package is pinned the same way for the same
      reason.
    * ``MEEP_GPU_ALLOW_UNCERTIFIED`` — SET TO ``0`` in the child's environment. The
      package runs a supported card the record does not certify by default, and
      ``1`` runs an unsupported one too; a round whose point is to EARN that
      certification must neither inherit ``1`` from the shell that launched it nor
      take the default, or the evidence it cuts would rest on the admission it is
      supposed to produce. ``0`` restricts every kernel to certified identities.
    """
    home = Path.home()
    stub = os.environ.get("TRITON_LIBCUDA_PATH", str(home / "triton_libcuda_stub"))
    # THE DIRECTORY NAME IS PART OF THE POLICY MECHANISM. ``ftz_stripped`` is the
    # token ``cupy_cache_reasons`` REQUIRES under keep and FORBIDS under flush,
    # so it tracks the policy rather than being pasted in.
    token = "ftz_stripped" if policy == "keep" else policy
    cupy_cache = cache_root / f"cupy_cache_{token}_{gate_name}"
    triton_cache = cache_root / f"triton_cache_{gate_name}"
    cupy_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MEEP_GPU_ALLOW_UNCERTIFIED"] = "0"
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MEEP_GPU_SUBNORMAL_POLICY": policy,
        "MEEP_GPU_DISPATCH": "0",
        "TRITON_LIBCUDA_PATH": stub,
        "LD_LIBRARY_PATH": stub + ":" + env.get("LD_LIBRARY_PATH", ""),
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUPY_CACHE_DIR": str(cupy_cache),
        "TRITON_CACHE_DIR": str(triton_cache),
        # THE INHERITED PYTHONPATH IS APPENDED, NOT DISCARDED. Measured
        # 2026-08-19: the no-PML gate's ENGINE leg does ``import cases`` — the
        # shared MEEP builder module that lives in a results script bundle and
        # not on any package path — and an overwritten PYTHONPATH turned that
        # into ModuleNotFoundError after the gate's other three legs had already
        # passed, losing 43 s of measurement to an environment decision this
        # driver had no business making.
        "PYTHONPATH": os.pathsep.join(
            [str(API_ROOT), str(HERE)]
            + ([inherited] if (inherited := env.get("PYTHONPATH")) else [])),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def sha256_of(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _last_line(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            tail = handle.read()[-4096:]
    except OSError:
        return "(no output yet)"
    lines = [line for line in tail.decode("utf-8", "replace").splitlines() if line]
    return lines[-1][:160] if lines else "(no output yet)"


def probe_policy(path: Optional[str]) -> Optional[str]:
    """The subnormal policy an expansion probe was CUT UNDER, or None.

    A stamp spells the policy TWICE and the two are different words:
    ``resolved`` is the request's answer (``keep`` / ``flush``) and ``policy``
    is the record's full name (``ieee_keep_ftz_stripped``). Comparing the wrong
    one against ``--policy`` would refuse every honest keep probe, so
    ``resolved`` is read first and the long name is normalised rather than
    matched.
    """
    if not path:
        return None
    try:
        stamp = json.loads(Path(path).read_text(encoding="utf-8")).get(
            "subnormal_policy")
    except (OSError, ValueError):
        return None
    if not isinstance(stamp, dict):
        stamp = {"policy": stamp} if stamp else {}
    for key in ("resolved", "requested", "policy"):
        value = stamp.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in ("keep", "flush"):
            return value
        if "keep" in value:
            return "keep"
        if "flush" in value:
            return "flush"
    return None


def probe_note(path: Optional[str]) -> str:
    return "none" if not path else f"{path} (cut under {probe_policy(path)!r})"


def read_artifact(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"no artifact at {path}"
    except (OSError, ValueError) as exc:
        return None, f"artifact unreadable at {path}: {exc!r}"


def record_device(python: str, out_root: Path, env: Dict[str, str]
                  ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``<out_root>/device.json`` — WHICH card this campaign's gates ran on.

    Returns ``(payload, None)`` or ``(None, reason)``; the caller refuses the whole
    campaign on a reason, because a fleet whose device is unknown cannot be bound
    to anything.

    WHY A CAMPAIGN CANNOT SKIP IT (2026-09-30). A certification record is keyed by
    compute capability, and every writer has to READ that key off the run. The
    family gates do not supply it: by the multi-capability design's census of the
    2026-09-25 fleet, the family artifacts' ``environment`` blocks are empty and
    twelve of the thirty-seven cited welds name no device at all (that census is the
    design's, not a count taken here) — so a round that ran only this fleet would leave
    the rebind nothing to key a slot by, and the round would end with gates that
    released and no capability to certify. One recorder process, before the first
    gate, in the environment the gates themselves get, closes that.

    IN THE GATE ENVIRONMENT, NOT THIS PROCESS'S. ``CUDA_VISIBLE_DEVICES`` is what
    decides which card a child sees; recording the identity anywhere else would
    describe whatever device index 0 is on the box rather than the one every gate in
    this campaign is pinned to. It is a separate process from the gates for the
    reason every gate is: this driver never imports a device stack, so it cannot
    read a device itself.
    """
    tool = HERE / "triton_device_identity.py"
    argv = [python, "-u", str(tool), "--write", str(out_root)]
    log_path = out_root / "device.log"
    spelled = " ".join(shlex.quote(part) for part in argv)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(argv, cwd=str(API_ROOT), env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        return None, (f"{spelled} exited {completed.returncode}: "
                      f"{_last_line(log_path)} (full output in {log_path})")
    # EXIT 0 IS NOT A RECORD. The recorder writes nothing on its own refusals, and
    # a campaign that trusted the status code alone would run the fleet and only
    # discover at rebind time that there is no capability to key the slots by.
    payload, problem = read_artifact(out_root / "device.json")
    if payload is None:
        return None, (f"{spelled} exited 0 and left no readable record: {problem}; "
                      f"output in {log_path}")
    if not payload.get("compute_capability"):
        return None, (f"{out_root / 'device.json'} names no compute capability "
                      f"(keys: {sorted(payload)}); a slot cannot be keyed by it")
    return payload, None


def native_verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The gate's OWN spelling of its outcome, beside the canonical reading.

    ``canonical_verdict`` answers "did it release?" for every shape
    :mod:`gate_provenance` knows. It answers ``None`` for the shapes it does
    not — and UNREADABLE is not REFUSED. Carrying the raw keys here is what
    lets a reader tell those two apart without opening a 40 MB artifact: a gate
    whose canonical verdict is None but whose own ``verdict.pass`` is true has
    a spelling problem, not a correctness problem.
    """
    out: Dict[str, Any] = {}
    for key in ("status", "passed", "released", "certifies", "aborted",
                "device_legs_skipped"):
        if key in payload:
            out[key] = payload[key]
    for key in ("verdict", "summary", "validation", "release", "license"):
        value = payload.get(key)
        if isinstance(value, (str, bool, int)):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = {k: v for k, v in value.items()
                        if isinstance(v, (str, bool, int, float))
                        or (isinstance(v, dict)
                            and all(isinstance(x, bool) for x in v.values()))}
    for key in ("failures", "skipped_legs", "leg_errors"):
        value = payload.get(key)
        if value:
            out[key] = value if not isinstance(value, list) else value[:20]
    return out


def run_one(gate: Gate, *, python: str, out_root: Path, cache_root: Path,
            gpu: int, probe: Optional[str], heartbeat: int, patience: int,
            policy: str, forbidden: frozenset) -> Dict[str, Any]:
    """Run one gate to completion and return the row the campaign records."""
    out_dir = out_root / gate.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gate.log"
    started = time.time()
    row: Dict[str, Any] = {
        "gate": gate.name,
        "script": gate.script,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "gpu": gpu,
        "probe_supplied": probe,
        "policy_requested": policy,
        "note": gate.note,
    }

    empty, why = wait_for_device(gpu, patience, forbidden=forbidden)
    row["placement"] = why
    if not empty:
        row.update({"exit_code": None, "released": None, "elapsed_s": 0.0,
                    "imported_digests": 0, "refused_by_driver": why})
        log(f"[gate] {gate.name:<20} REFUSED BY DRIVER — {why}")
        return row

    argv = [python, "-u", str(HERE / gate.script), *gate.argv(out_dir, probe, policy)]
    row["argv"] = " ".join(shlex.quote(part) for part in argv)
    log(f"[gate] {gate.name:<20} starting — {why}")
    log(f"[gate] {gate.name:<20} argv: {row['argv']}")

    env = gate_environment(gpu, cache_root, gate.name, policy)
    row["cupy_cache_dir"] = env["CUPY_CACHE_DIR"]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, cwd=str(API_ROOT), env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        next_beat = time.time() + heartbeat
        timed_out = False
        while True:
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if now - started > gate.timeout:
                timed_out = True
                process.kill()
                process.wait()
                break
            if now >= next_beat:
                next_beat = now + heartbeat
                log(f"[beat] {gate.name:<20} {int(now - started):>6}s elapsed | "
                    f"{_last_line(log_path)}")
    elapsed = time.time() - started
    row["exit_code"] = None if timed_out else process.returncode
    row["timed_out"] = timed_out
    row["elapsed_s"] = round(elapsed, 1)
    row["log"] = str(log_path)

    artifact_path = gate.artifact(out_dir)
    payload, problem = read_artifact(artifact_path)
    row["artifact"] = str(artifact_path)
    row["artifact_sha256"] = sha256_of(artifact_path)
    row["artifact_bytes"] = (artifact_path.stat().st_size
                             if artifact_path.exists() else 0)
    if payload is None:
        row.update({"released": None, "imported_digests": 0,
                    "reasons": [problem or "artifact unreadable"],
                    "canonical_read_from": None})
    else:
        canonical = payload.get("canonical_verdict") or {}
        imported = payload.get("imported_source_sha256") or {}
        row.update({
            "released": canonical.get("released"),
            "canonical_read_from": canonical.get("read_from"),
            "reasons": canonical.get("reasons") or [],
            "unreadable": canonical.get("unreadable"),
            "imported_digests": len(imported),
            "imported_source_sha256": imported,
            "native_verdict": native_verdict(payload),
            "subnormal_policy": (payload.get("subnormal_policy") or {}).get(
                "policy") if isinstance(payload.get("subnormal_policy"), dict)
                else payload.get("subnormal_policy"),
            "host_environment": payload.get("environment") or payload.get("host")
                or payload.get("device"),
        })
        probe_path = gate.probe_path(out_dir)
        if probe_path is not None and probe_path.exists():
            row["probe_written"] = str(probe_path)

    log(f"[gate] {gate.name:<20} exit={row['exit_code']} "
        f"released={row.get('released')} "
        f"imported_digests={row.get('imported_digests')} "
        f"elapsed={row['elapsed_s']}s"
        + ("  TIMED OUT" if timed_out else ""))
    for reason in (row.get("reasons") or [])[:6]:
        log(f"       refusal: {str(reason)[:220]}")
    return row


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one row and FSYNC. A campaign killed mid-gate keeps every row."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", type=int, default=7,
                        help="PHYSICAL device index; anything in --forbidden-gpus "
                             "is refused")
    parser.add_argument("--forbidden-gpus",
                        default=",".join(str(index) for index
                                         in sorted(DEFAULT_FORBIDDEN_GPUS)),
                        help="comma-separated PHYSICAL device indices this "
                             "campaign may never take, whatever nvidia-smi says "
                             "about them; the default is the set aside on the "
                             "current GPU host, and '' declares that this host "
                             "sets none aside")
    parser.add_argument("--out-root", required=True,
                        help="campaign directory; one subdirectory per gate")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gates", default=",".join(GATES),
                        help="comma-separated gate names, run in this order")
    parser.add_argument("--probe", default=None,
                        help="expansion probe for gates that consume one; the "
                             "probe measured by this campaign's complex gate "
                             "wins over it when that gate wrote one")
    parser.add_argument("--policy", default="keep", choices=("keep", "flush"),
                        help="the subnormal policy every gate installs; 'keep' "
                             "is what every weld in fingerprints.json records, "
                             "and it must match the expansion probe's own stamp")
    parser.add_argument("--heartbeat", type=int, default=120)
    parser.add_argument("--patience", type=int, default=1800,
                        help="seconds to wait for a busy device before refusing")
    args = parser.parse_args(argv)

    try:
        forbidden = frozenset(int(part) for part
                              in args.forbidden_gpus.split(",") if part.strip())
    except ValueError:
        log(f"ABORT: --forbidden-gpus {args.forbidden_gpus!r} is not a "
            f"comma-separated list of device indices")
        return 2

    if args.gpu in forbidden:
        log(f"ABORT: GPU {args.gpu} is set aside on this host "
            f"(--forbidden-gpus {sorted(forbidden)})")
        return EXIT_CANNOT_CERTIFY

    out_root = Path(args.out_root).resolve()
    cache_root = out_root / "caches"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    jsonl = out_root / "gates.jsonl"

    names = [name.strip() for name in args.gates.split(",") if name.strip()]
    unknown = [name for name in names if name not in GATES]
    if unknown:
        log(f"ABORT: unknown gate(s) {unknown}; known: {sorted(GATES)}")
        return 2

    # A PROBE FROM THE OTHER POLICY IS A BROKEN COMPARISON, NOT A FALLBACK. The
    # EXPANSION constexpr the complex families bind is a measured platform fact
    # UNDER ONE POLICY; a keep probe read by a flush run (or the reverse)
    # licenses nothing, and the failure is silent because both files parse.
    # Checked BEFORE anything is logged or created, so a mis-paired campaign
    # cannot leave half an artifact tree behind.
    supplied = probe_policy(args.probe)
    if args.probe and supplied != args.policy:
        log(f"ABORT: the probe at {args.probe} was cut under {supplied!r} and "
            f"this campaign installs {args.policy!r}; pairing them would be a "
            f"broken comparison, not a platform verdict")
        return EXIT_CANNOT_CERTIFY

    log(f"===== triton weld campaign {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====")
    log(f"host: {socket.gethostname()}  {platform.platform()}")
    log(f"python: {args.python}")
    log(f"api root: {API_ROOT}")
    log(f"out root: {out_root}")
    log(f"gpu: {args.gpu}  gates: {len(names)}  jsonl: {jsonl}")
    log(f"devices set aside on this host: {sorted(forbidden) or 'none'}")
    log(f"subnormal policy requested for every gate: {args.policy!r}")
    log(f"expansion probe handed to gates that consume one: {probe_note(args.probe)}")
    log(_nvidia_smi(["--query-gpu=index,name,memory.used,utilization.gpu",
                     "--format=csv,noheader"]))

    # THE DEVICE, BEFORE THE FIRST GATE AND IN THE GATES' OWN ENVIRONMENT. The
    # record a rebind keys a slot by is the compute capability this campaign ran
    # on, and the family gates stamp none; a campaign that discovered that after
    # the fleet had run would have spent the GPU hours and still have nothing to
    # certify. The cache directories this environment names are the recorder's own,
    # for the same reason every gate's are private: nothing it compiles may serve a
    # later gate a binary from another policy.
    device, why = record_device(
        args.python, out_root,
        gate_environment(args.gpu, cache_root, "device_identity", args.policy))
    if device is None:
        log(f"ABORT: the campaign's device cannot be recorded, so nothing it cuts "
            f"could be bound to a capability — {why}")
        return EXIT_CANNOT_CERTIFY
    log(f"device: {device.get('device')} cc {device.get('compute_capability')} "
        f"on {device.get('hostname')} (CUDA_VISIBLE_DEVICES="
        f"{device.get('cuda_visible_devices')!r}, CuPy {device.get('cupy')}, "
        f"Triton {device.get('triton')}) -> {out_root / 'device.json'}")

    probe = args.probe
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        gate = GATES[name]
        log(f"----- [{index}/{len(names)}] {name} -----")
        row = run_one(gate, python=args.python, out_root=out_root,
                      cache_root=cache_root, gpu=args.gpu, probe=probe,
                      heartbeat=args.heartbeat, patience=args.patience,
                      policy=args.policy, forbidden=forbidden)
        row["order"] = index
        rows.append(row)
        append_jsonl(jsonl, row)
        written = row.get("probe_written")
        if name == "complex" and written:
            # SAME CHECK AS THE SUPPLIED PROBE GETS, on the probe this campaign
            # just measured. It is keep by construction here, and a construction
            # is exactly the kind of argument this campaign does not accept
            # where a measurement is available.
            cut_under = probe_policy(written)
            if cut_under == args.policy:
                probe = written
                log(f"[probe] later gates will bind the probe this campaign "
                    f"measured: {probe} (cut under {cut_under!r})")
            else:
                log(f"[probe] KEEPING the supplied probe: the one just measured "
                    f"reads as {cut_under!r}, not {args.policy!r}")

    summary = {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "gpu": args.gpu,
        "forbidden_gpus": sorted(forbidden),
        # THE DEVICE, IN THE CAMPAIGN RECORD AND NOT ONLY IN A SIBLING FILE. A
        # reader asking "which card certified this?" reads one artifact, and a
        # rebind reading the campaign row does not have to find device.json beside
        # it. Same bytes, both places, written once by the recorder.
        "device": device,
        "released": sorted(r["gate"] for r in rows if r.get("released") is True),
        "refused": sorted(r["gate"] for r in rows if r.get("released") is False),
        "unreadable": sorted(r["gate"] for r in rows
                             if r.get("released") is None),
        "nonzero_exit": sorted(r["gate"] for r in rows if r.get("exit_code")),
        "rows": rows,
    }
    (out_root / "campaign.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    log("===== campaign summary =====")
    for row in rows:
        log(f"  {row['gate']:<22} exit={str(row.get('exit_code')):<6} "
            f"released={str(row.get('released')):<6} "
            f"digests={row.get('imported_digests', 0):<4} "
            f"{row.get('elapsed_s')}s")
    log(f"released:   {summary['released']}")
    log(f"refused:    {summary['refused']}")
    log(f"unreadable: {summary['unreadable']}")
    log(f"wrote {out_root / 'campaign.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
