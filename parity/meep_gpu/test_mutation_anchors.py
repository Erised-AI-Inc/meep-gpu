"""The mutation batteries' anchors must still resolve — checked on every test run.

Two harnesses, the same failure. THE CONVERTER'S BATTERY (the first three tests):
``mutate_from_meep.py`` proves the converter's tests would NOTICE a converter bug, by
applying each plausible one-line defect and requiring a failure. That proof is only
worth what its anchors are worth: a mutation whose ``find`` text no longer appears in
the module is never applied, is reported as "anchor missing" rather than "survived",
and silently stops testing the defect it was written for.

The battery takes tens of minutes, so it runs rarely, and anchors rot in between. This
test is the cheap half — it reads two files and counts substrings, so it costs about a
second and runs whenever anyone runs the suite.

It caught its own motivating case: the reduced-dimension and PEC work moved the code
out from under FOUR anchors at once (``declared_dimensions_ignored``,
``flat_axis_only_checked_on_z`` and both ``k_point`` mutations), and nothing noticed
until they were counted. Two were re-anchored; ``flat_axis_only_checked_on_z`` was
deleted outright, because that work inverted it — checking only ``cell_size.z`` is what
MEEP's ``use_2d`` does, so the old mutation had come to describe the correct code.

A failure here means: re-anchor the mutation onto the rule that replaced the old one,
or delete it if the change made it meaningless. Do not silence it — an unanchored
mutation is a test that has stopped testing.

THE GATE HARNESS'S NEEDLE TABLES (the rest). Every CUDA gate that arms the shared
probe's defects builds its own table with ``gate_needles.gate_table``; a device leg's
verdict is a function of the mutated bytes it compiles, so a host derivation of those
bytes is what a CPU run can check about it. ``record_gate_needle_anchors.py``
derives, per table, every needle's site count and mutated-text digest on the device
strings that gate rewrites, and ``gate_needle_anchors.json`` is the record. The check
runs in a child process because the derivation imports the CUDA kernel modules under
a stub ``cupy`` that must not leak into this suite. A failure names the table, the
needle and the target: a kernel a gate mutates was edited (its mutation legs now
compile different bytes), a needle was re-spelled, or a needle stopped matching.
Re-mint with ``record_gate_needle_anchors.py --write`` in the change that moved it,
so the record's diff says which.

THE INVERSION (the last two tests). The shared default's grouping defects plant the
grouping the kernels carried until 0.10 on MEEP's, and no table keeps its own copy of
a shared spelling: a gate either borrows the default's entry or spells a different
defect, so a later re-spelling of the default reaches every gate that shares it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).with_name("mutate_from_meep.py")
_HERE = Path(__file__).parent
_ANCHOR_TOOL = _HERE / "record_gate_needle_anchors.py"
_ANCHOR_RECORD = _HERE / "gate_needle_anchors.json"


def _harness():
    spec = importlib.util.spec_from_file_location("mutate_from_meep", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _HARNESS.exists(), reason="mutation harness not present")
def test_every_mutation_anchor_still_resolves_exactly_once():
    harness = _harness()
    stale = harness.stale_anchors()
    detail = "\n".join(
        f"  {name}: anchor matches {count}x in {module} (expected exactly 1)"
        for name, module, count in stale
    )
    assert not stale, (
        f"{len(stale)} of {len(harness.MUTATIONS)} mutation anchors no longer resolve, so "
        f"those mutations are never applied and the defects they pin are untested:\n"
        f"{detail}"
    )


@pytest.mark.skipif(not _HARNESS.exists(), reason="mutation harness not present")
def test_every_mutation_actually_changes_its_module():
    """A mutation whose replacement equals its anchor edits nothing and always 'survives'."""
    harness = _harness()
    inert = [
        name for name, entry in harness.MUTATIONS.items()
        if harness.unpack(entry)[0] == harness.unpack(entry)[1]
    ]
    assert not inert, f"these mutations replace their anchor with itself: {inert}"


@pytest.mark.skipif(not _HARNESS.exists(), reason="mutation harness not present")
def test_every_mutation_names_a_module_and_tests_that_exist():
    """Targets are paths, and a typo'd one would run the wrong suite or none at all."""
    harness = _harness()
    missing = []
    for name, entry in sorted(harness.MUTATIONS.items()):
        _find, _replace, _selector, module, tests = harness.unpack(entry)
        if not module.exists():
            missing.append(f"{name}: module {module} does not exist")
        missing += [f"{name}: test target {t} does not exist"
                    for t in tests if not Path(t).exists()]
    assert not missing, "\n".join(missing)


# --------------------------------------------------------------------------
# The gate harness's per-gate needle tables.
# --------------------------------------------------------------------------

def _load(path: Path, name: str):
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record():
    return json.loads(_ANCHOR_RECORD.read_text(encoding="utf-8"))


def test_every_needle_table_the_harness_builds_is_in_the_record():
    """A table built and not recorded is a battery nobody checks between device runs."""
    built = set()
    for path in sorted(_HERE.glob("*.py")):
        # The machinery and the tests build tables too; neither is a gate.
        if path.name == "gate_needles.py" or path.name.startswith("test_"):
            continue
        built |= set(re.findall(r'gate_needles\.gate_table\(\s*"([^"]+)"',
                                path.read_text(encoding="utf-8")))
    tool = _load(_ANCHOR_TOOL, "_record_gate_needle_anchors")
    listed = {gate for gate, _module, _attribute, _targets in tool.TABLES}
    assert built == listed, (
        f"built but not listed in record_gate_needle_anchors.TABLES: "
        f"{sorted(built - listed)}; listed but built nowhere: {sorted(listed - built)}")
    assert listed == set(_record()["tables"]), (
        "gate_needle_anchors.json is not the record of the listed tables; re-mint it "
        "with record_gate_needle_anchors.py --write")


def test_every_gate_needle_hands_the_compiler_the_recorded_bytes():
    """Host-side derivation of every mutation leg's device text, against the record."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MEEP_GPU_DISPATCH="0")
    done = subprocess.run([sys.executable, str(_ANCHOR_TOOL)], cwd=str(_HERE),
                          env=env, capture_output=True, text=True, timeout=600)
    differing = [line for line in done.stdout.splitlines()
                 if line.startswith("DIFFERS")]
    assert done.returncode == 0, (
        f"the gate harness's needles no longer derive the recorded bytes "
        f"({len(differing)} difference(s)); each line names table/needle/target:\n"
        + "\n".join(differing[:60])
        + (f"\n... and {len(differing) - 60} more" if len(differing) > 60 else "")
        + f"\nstderr tail:\n{done.stderr[-2000:]}")


def test_every_recorded_needle_matches_and_edits_a_target_of_its_gate():
    """A needle that matches nothing, or rewrites into itself, arms no defect."""
    record = _record()
    dead, inert = [], []
    for gate, entry in sorted(record["tables"].items()):
        for name, needle in sorted(entry["needles"].items()):
            if not needle["sites"]:
                dead.append(f"{gate}/{name}")
            for label, digest in needle["mutated_sha256"].items():
                if digest == entry["targets"][label]:
                    inert.append(f"{gate}/{name} on {label}")
    beta = record["beta_hd_device_mutations"]
    assert not beta["unresolved"], f"beta H->D needles not matching exactly once: " \
                                   f"{beta['unresolved']}"
    inert += [f"beta_hd/{name}" for name, digest in beta["mutated_sha256"].items()
              if digest == beta["target"]]
    assert not dead, f"needles matching no target of their gate: {dead}"
    assert not inert, f"needles rewriting a target into itself: {inert}"


def test_every_hand_leg_arms_a_needle_of_the_hand_table():
    """The hand track's legs name entries of ITS table, not of the shared default."""
    probe = _load(_HERE / "probe_fused_kernel_bit_identity.py", "_probe_anchor_check")
    runner = _load(_HERE / "run_pml_gate_mutations.py", "_runner_anchor_check")
    armed = [leg["args"][leg["args"].index("--source-mutation") + 1]
             for leg in runner.LEGS_BY_TRACK["hand"]
             if "--source-mutation" in leg["args"]]
    assert armed, "the hand leg table arms no source mutation"
    missing = sorted(set(armed) - set(probe.HAND_NEEDLES))
    assert not missing, f"hand legs arming names outside probe.HAND_NEEDLES: {missing}"
    assert probe.HandTrackAdapter(None).source_mutations is probe.HAND_NEEDLES


def test_the_shared_grouping_defects_plant_the_pre_alignment_groupings():
    """MEEP's grouping is the shipped text, so each grouping defect installs the old one.

    The curl: MEEP's left-to-right ``((a-b)+c)-d`` (step_generic.cpp:104) found, the
    pairwise ``(a-b)+(c-d)`` planted. The constitutive: MEEP's
    ``f + ((kps*src) - (kms*prev))`` (step_generic.cpp:689) found, the two left-to-right
    accumulations planted. Both arm on the hand pair's kernels in the record.
    """
    probe = _load(_HERE / "probe_fused_kernel_bit_identity.py", "_probe_inversion_check")
    stencil = probe.SOURCE_MUTATIONS["regroup_stencil"]
    assert "dtdx * (((sf - f1) + f2) - ss)" in stencil.find, stencil.find
    assert "dtdx * ((sf - f1) + (f2 - ss))" in stencil.replace, stencil.replace
    constitutive = probe.SOURCE_MUTATIONS["regroup_constitutive"]
    assert "(f[idx] + ((kps * src) - (kms * prev)))" in constitutive.find
    assert "((f[idx] + (kps * src)) - (kms * prev))" in constitutive.replace
    hand = _record()["tables"]["probe_fused_kernel_bit_identity:hand"]["needles"]
    for name in ("regroup_stencil", "regroup_constitutive"):
        assert hand[name]["borrowed"] and sum(hand[name]["sites"].values()) >= 2, (
            name, hand[name])


def test_no_table_keeps_its_own_copy_of_a_shared_spelling():
    """An owned entry spelled as the shared default's entry of the same name is a copy.

    A copy does not move when the default is re-spelled, so the gate holding it would
    keep measuring the old spelling while its record calls it the gate's own.
    """
    probe = _load(_HERE / "probe_fused_kernel_bit_identity.py", "_probe_copy_check")
    shared = {name: probe.gate_needles.transform_spec(needle)
              for name, needle in probe.SOURCE_MUTATIONS.items()}
    copies = [f"{gate}/{name}"
              for gate, entry in sorted(_record()["tables"].items())
              for name, needle in sorted(entry["needles"].items())
              if not needle["borrowed"] and shared.get(name) == needle["identity"]]
    assert not copies, f"owned entries identical to the shared default's: {copies}"
