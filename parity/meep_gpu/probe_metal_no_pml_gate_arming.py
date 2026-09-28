"""Does ``gate_metal_no_pml_constitutive.py`` ACTUALLY FAIL? Fault injection, from outside.

A gate that has never been seen red is a script, not a gate. Its own ``mutations``
leg arms four defects and catches all four — but those are the gate author's four,
and a leg cannot audit the assumption that picked it. This probe injects defects the
gate's own m1-m4 DO NOT COVER, from outside the gate, and requires it to go red.

THE INJECTIONS, each aimed at one REQUIRED property of the certification:

  f1_census_blinded      ``subnormal.census`` returns 0. The subnormal precondition
                         is then unmeasured. REQUIRED: the census is a PASS
                         CONDITION, not a statistic, so the gate must fail.
  f2_oracle_writes       ``stepping.update_E`` writes one word on the admitted
                         configuration. REQUIRED: "the reference leg is
                         byte-identical to stepping.py" must be an assertion about
                         the ORACLE, not only about the plan.
  f3_plan_never_ran      the step plan's ``run`` stops counting. REQUIRED: a byte
                         comparison satisfied by a plan that was never invoked is
                         the null family's characteristic false pass.
  f4_residency_refuses   the residency verdict comes back REFUSED on the covered
                         arm. REQUIRED: the Metal-specific leg must assert its
                         verdict rather than record it — which is exactly what
                         tranche 1's leg 8 failed to do.
  f5_control_goes_quiet  ``stepping.update_H`` becomes a no-op under an ACTIVE
                         layer. The controls leg's "bytes MUST move" half must
                         fail; if it does not, the identity leg is decorative.
  f6_slot_side_crossed   the slot->side table crossed and its fail-closed clause
                         removed. REQUIRED: a null placed in the OTHER sub-step's
                         slot deletes a sub-step the array path performs. Measured
                         2026-08-15 BEFORE the fix — ``plan_step`` filled
                         ``update_E`` with a null on a run where
                         ``stepping.update_E`` moves 3240 words, and the 93-test
                         suite and all eight legs stayed GREEN.
  f7_non_clause_table_shrunk
                         one ``METAL_NON_CLAUSES`` entry deleted. REQUIRED: the
                         table's own totality assertion runs table -> case and
                         cannot see an OMITTED entry, so the ``over_coverage`` leg
                         has to run clause -> table AND join its marker table to
                         the shipped one. It did not at first — this injection is
                         what found the missing join.
  f8_poison_controls_dropped
                         the poison leg's CONTROL rows removed. REQUIRED: "the
                         covered sub-step touched no field array" is a tautology
                         beside a poison nothing can trip, so a leg that still
                         passes without its controls is measuring nothing.
  f9_arm_gate_forced_open
                         the ``absorber_inactive`` arm gate removed AT THE REGISTRY
                         (patching the module global is a no-op — ``arms.register``
                         captured the function object, and that was measured before
                         this entry was written). REQUIRED: the leg claims the gate
                         changes no verdict AND is really there; the second half is
                         what this breaks.

NOTHING IN THE TREE IS EDITED. Each injection runs the real gate in a SUBPROCESS
with a small patch applied after import and before ``main()``, so the certified
module and the gate keep the sha256 the artifact records.

A PASS HERE IS A RED GATE. Any injection that leaves the gate green is reported as
an ARMING HOLE and fails this probe.

Usage::

    python -u probe_metal_no_pml_gate_arming.py \\
        --out parity/meep_gpu/results/metal_no_pml_<date>/arming.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

GATE = os.path.join(HERE, "gate_metal_no_pml_constitutive.py")

# Each entry: the patch source, and the leg it must break. The leg is named so a
# green run reports WHICH guarantee was missing rather than only that one was.
INJECTIONS: Dict[str, Dict[str, str]] = {
    "f1_census_blinded": {
        "leg": "identity",
        "why": "the subnormal census is a pass condition (fact h: the claim is "
               "conditional and the condition must be CHECKED)",
        "patch": """
from meep_gpu.metal_kernels import subnormal
subnormal.census = lambda array: 0
""",
    },
    "f2_oracle_writes": {
        "leg": "identity",
        "why": "byte-identity is a claim about stepping.py, not only about the plan",
        "patch": """
from meep_gpu import stepping
_real = stepping.update_E
def _writing_update_E(fields, pml=None):
    _real(fields, pml)
    target = getattr(fields, 'Dx', None)
    if target is not None and target.size:
        flat = target.reshape(-1)
        flat[0] = flat[0] + type(flat[0])(1.0)
stepping.update_E = _writing_update_E
""",
    },
    "f3_plan_never_ran": {
        "leg": "identity",
        "why": "a byte comparison is trivially satisfied by a plan never invoked",
        "patch": """
from meep_gpu.metal_kernels import no_pml_constitutive as family
family.MetalNullConstitutiveStepPlan.run = lambda self, contract=None: None
""",
    },
    "f4_residency_refuses": {
        "leg": "residency",
        "why": "tranche 1 RECORDED residency_covered without asserting it",
        "patch": """
from meep_gpu.metal_kernels import no_pml_constitutive as family
from meep_gpu.metal_kernels.coverage import Coverage
_real = family.plan_metal_null_constitutive_step
def _refusing(*args, **kwargs):
    plan = _real(*args, **kwargs)
    if plan is not None:
        plan.residency = Coverage(False, ('injected residency refusal',))
    return plan
family.plan_metal_null_constitutive_step = _refusing
""",
    },
    "f5_control_goes_quiet": {
        "leg": "controls",
        "why": "the control must prove the oracle really writes on a refused config",
        "patch": """
from meep_gpu import stepping
stepping.update_H = lambda fields, pml=None: None
""",
    },
    # ADDED 2026-08-15. Both of these left the gate GREEN before this round's fixes
    # — measured, not supposed — and both are the arming holes rather than polish.
    "f6_slot_side_crossed": {
        "leg": "mutations",
        "why": "a null placed in the OTHER sub-step's slot deletes a sub-step the "
               "array path performs. Measured with the two entries swapped: "
               "plan_step filled update_E with a null on a run where "
               "stepping.update_E moves 3240 words, and the 93-test suite and all "
               "eight legs stayed green. `slot_side_reasons` refuses it fail-closed "
               "and m5 is the armed mutation; this injection removes the refusal",
        "patch": """
from meep_gpu.metal_kernels import no_pml_constitutive as family
family.NULL_SLOTS = {'update_H': 'E', 'update_E': 'H'}
family.slot_side_reasons = lambda slot: ()
""",
    },
    "f7_non_clause_table_shrunk": {
        "leg": "over_coverage",
        "why": "the table's OWN totality assertion runs table -> case and cannot "
               "see an OMITTED entry. The over_coverage leg runs clause -> table, "
               "so deleting an entry must go red there; if it does not, an "
               "unrecorded non-clause is invisible again",
        "patch": """
from meep_gpu.metal_kernels import no_pml_constitutive as family
family.METAL_NON_CLAUSES = {k: v for k, v in family.METAL_NON_CLAUSES.items()
                            if k != 'volume_allocation'}
""",
    },
    "f9_arm_gate_forced_open": {
        "leg": "over_coverage",
        "why": "the `absorber_inactive` arm gate is claimed to decide only whether "
               "the arm is CONSULTED. The leg asserts BOTH halves: forcing it open "
               "must change no verdict (a PREDICTED NULL, and it holds), and the "
               "gate must really suppress a consultation somewhere or the claim is "
               "about a gate that is not there. THE INJECTION POINT MATTERS: "
               "`arms.register` captured the function object, so patching the "
               "module global is a no-op — measured, the gate stayed green. The "
               "registry entry is what the leg reads and what this patches",
        "patch": """
from meep_gpu.metal_kernels import arms, launch  # noqa: F401 - import registers

for _slot in ('update_H', 'update_E'):
    for _spec in arms.registered(_slot):
        if _spec.label == 'no-PML null':
            _spec.gate = None
""",
    },
    "f8_poison_controls_dropped": {
        "leg": "over_coverage",
        "why": "the 'the covered sub-step touches no field array' rows are a "
               "TAUTOLOGY on their own — they would pass against a poison nothing "
               "can trip. The CONTROLS are what make them evidence, so a leg that "
               "still passes with its controls removed is measuring nothing",
        "patch": """
module.POISON_CASES = tuple(row for row in module.POISON_CASES if not row[3])
""",
    },
}

RUNNER = """
import runpy, sys, types
sys.argv = ['gate', '--out', {out!r}]
module = types.ModuleType('gate_under_injection')
module.__file__ = {gate!r}
source = open({gate!r}, 'r', encoding='utf-8').read()
code = compile(source, {gate!r}, 'exec')
module.__dict__['__name__'] = 'gate_under_injection'
exec(code, module.__dict__)
{patch}
raise SystemExit(module.__dict__['main']())
"""


def run_injection(name: str, entry: Dict[str, str], workdir: str) -> Dict[str, Any]:
    out = os.path.join(workdir, f"{name}.json")
    script = RUNNER.format(out=out, gate=GATE, patch=entry["patch"])
    path = os.path.join(workdir, f"{name}_runner.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script)
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = API_ROOT + os.pathsep + HERE + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.time()
    completed = subprocess.run([sys.executable, "-u", path], capture_output=True,
                               text=True, cwd=API_ROOT, env=environment, timeout=900)
    tail = (completed.stdout + completed.stderr).strip().splitlines()
    return {
        "injection": name,
        "targets_leg": entry["leg"],
        "why_required": entry["why"],
        "returncode": completed.returncode,
        "gate_went_red": completed.returncode != 0,
        "elapsed_s": round(time.time() - started, 2),
        "tail": tail[-6:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    holes: List[str] = []
    started = time.time()

    # THE BASELINE IS A PASS CONDITION. If the unpatched gate does not go GREEN in
    # this same subprocess harness, every red below is attributable to the harness
    # rather than to the injection.
    with tempfile.TemporaryDirectory() as workdir:
        baseline = run_injection("f0_baseline_unpatched",
                                 {"leg": "-", "why": "the harness itself must be clean",
                                  "patch": ""}, workdir)
        rows.append(baseline)
        print(f"[arming] f0_baseline_unpatched               rc={baseline['returncode']} "
              f"{'GREEN' if not baseline['gate_went_red'] else 'RED'} "
              f"({baseline['elapsed_s']}s)", flush=True)
        if baseline["gate_went_red"]:
            print("[arming] BASELINE IS RED: the harness is broken, not the gate", flush=True)
            payload = {"rows": rows, "holes": ["baseline"], "baseline_green": False}
            with open(arguments.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            return 1

        for index, (name, entry) in enumerate(INJECTIONS.items(), start=1):
            row = run_injection(name, entry, workdir)
            rows.append(row)
            if not row["gate_went_red"]:
                holes.append(name)
            print(f"[arming] {index}/{len(INJECTIONS)} {name:<24} leg={row['targets_leg']:<12} "
                  f"rc={row['returncode']} "
                  f"{'CAUGHT (gate red)' if row['gate_went_red'] else 'ARMING HOLE (gate green)'} "
                  f"({row['elapsed_s']}s)", flush=True)
            payload = {"rows": rows, "holes": holes, "baseline_green": True,
                       "elapsed_s": round(time.time() - started, 2)}
            os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
            with open(arguments.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

    if holes:
        print(f"[arming] ARMING HOLES: {holes}", flush=True)
        print("GATE ARMING PROBE FAILED", flush=True)
        return 1
    print(f"GATE ARMING PROBE PASSED in {time.time() - started:.1f}s -> {arguments.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
