"""The mutation battery's anchors must still resolve — checked on every test run.

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
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HARNESS = Path(__file__).with_name("mutate_from_meep.py")


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
