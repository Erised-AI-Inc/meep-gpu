# SPDX-License-Identifier: GPL-2.0-or-later
"""Tier B of MEEP identity, read from the evidence archive.

``record_meep_identity_archive.py`` writes Tier B into the evidence archive
(``parity/meep_gpu/results/meep_identity_tier_b/``): per reference (host and subnormal
mode) every corpus row's Tier C verdict, MEEP's digests, the plane digests and, where the
capsule schema carries the row, its capsule and currents. These tests read that record:
every file must read back, every row that is not a named exclusion must hold or be refused
by name, and the smallest replayable rows of each keep set must replay with no MEEP.

They are listed under ``[evidence_archive]`` in ``tools/ci/declared_resources.txt``: a
checkout without the archive skips them by that resource.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parity.meep_gpu import record_meep_identity_archive as archive

HERE = Path(__file__).resolve().parent
TIER_B = HERE / "results" / "meep_identity_tier_b"
REPLAYED_PER_SET = 3


def _record():
    path = TIER_B / archive.ARCHIVE_FILE
    if not path.exists():
        pytest.fail(f"{path} does not exist: the evidence archive carries no Tier B record "
                    "(record_meep_identity_archive.py writes it)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_tier_b_file_reads_back():
    record = _record()
    assert record["schema"] == archive.SCHEMA and record["version"] == archive.VERSION
    for name in record["sets"]:
        assert archive.check_set(TIER_B, name) > 0, name


def test_every_row_that_is_not_a_named_exclusion_holds_or_is_refused_by_name():
    for name, leaf in _record()["sets"].items():
        bad = sorted((label, row["status"], (row.get("problems") or [""])[0][:200])
                     for label, row in leaf["rows"].items()
                     if not row.get("excluded") and row["status"] not in ("holds", "refused"))
        assert not bad, f"{name}: {len(bad)} rows neither hold nor were refused: {bad[:5]}"
        unnamed = sorted(label for label, row in leaf["rows"].items()
                         if row["status"] == "refused" and not (row.get("problems") or [""])[0].strip())
        assert not unnamed, f"{name}: refused with no reason: {unnamed}"


def test_the_smallest_replayable_rows_of_each_keep_set_replay_with_no_meep():
    replayed = 0
    for name, leaf in _record()["sets"].items():
        if leaf["mode"] != "keep":
            continue
        candidates = sorted(
            (label for label, row in leaf["rows"].items()
             if row["status"] == "holds" and "capsule" in (row.get("files") or {}) and not row.get("excluded")),
            key=lambda label: sum(spec["bytes"] for spec in leaf["rows"][label]["files"].values()))
        for label in candidates[:REPLAYED_PER_SET]:
            outcome = archive.replay_row(TIER_B / name / label, "keep", label)
            assert outcome.get("holds") and outcome.get("meep_imported") is False, (name, label, outcome)
            replayed += 1
    assert replayed, "no keep set carries a replayable row"
