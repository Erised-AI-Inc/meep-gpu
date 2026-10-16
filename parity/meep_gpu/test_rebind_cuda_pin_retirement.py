"""A pin a CUDA weld's gate stopped importing leaves that weld by name, inside a rebind only.

``rebind_cuda_welds.py`` keeps each entry's curated pin set, so an entry that pins a file
its gate no longer imports can never rebind: no fresh run records the file, and the tool
skips the entry with "the artifact does not record" although its gate released.
``RETIRED_PINS`` names each such pin by ``(weld, path)``; the files still ship and other
welds still pin them. These tests pin the rule: the table names exactly the two reviewed
rows, each weld's gate no longer imports its row's file, each file is still pinned by other
welds, and a retired pin sits in both digest maps or in neither; a rebind drops a retired
pin from both maps and reports ``RETIRED-PIN``; a leg that records a retired path (the gate
imports it again) refuses the entry, before or after the bind; a non-retired missing pin
still refuses; the same path on another weld is compared as strictly as ever; an entry
that does not rebind keeps its pin; a drop that would strand another capability's record
still refuses; and a second ``--write`` leaves the ledger byte-identical.

Host-only: no GPU, no MEEP. The real ledger and record are read, never written; the
``main()`` legs write a temporary ledger under a temporary repository root.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rebind_cuda_welds as R  # noqa: E402

from meep_gpu import fastpath  # noqa: E402
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

STENCIL = "cuda_complex_offdiag_stencil_welds_2026-09-17"
CYLINDRICAL = "cuda_cylindrical_fused_hd_pair_2026-09-07"

#: The two reviewed rows, and the gate script each weld's campaign runs.
ROWS = {
    (STENCIL, "parity/meep_gpu/gate_cuda_complex_no_pml.py"):
        "parity/meep_gpu/gate_cuda_complex_offdiag_stencil_welds.py",
    (CYLINDRICAL, "parity/meep_gpu/gate_cuda_cylindrical_real_fused_hd_pair.py"):
        "parity/meep_gpu/gate_cuda_cylindrical_fused_hd_pair.py",
}
ENTRY_OF = {STENCIL: "89-X1", CYLINDRICAL: "76-KC8"}


# ---------------------------------------------------------------------------
# THE TABLE AGAINST THE CHECKOUT AND THE SHIPPED RECORDS
# ---------------------------------------------------------------------------


def test_the_table_names_exactly_the_two_reviewed_rows():
    assert set(R.RETIRED_PINS) == set(ROWS)
    for (weld, path), why in R.RETIRED_PINS.items():
        assert ENTRY_OF[weld] in why, (weld, why)
        assert R.retired_pins_of(weld) == {path: why}
    assert R.retired_pins_of("cuda_complex_no_pml_2026-08-27") == {}


def _imported_modules(path: pathlib.Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_each_weld_gate_no_longer_imports_its_retired_file():
    """The claim each row makes, read off the gate's own source: no import statement
    names the module. A gate that imports it again fails here before any device run."""
    for (weld, path), gate in ROWS.items():
        stem = pathlib.PurePosixPath(path).stem
        assert stem not in _imported_modules(R._API / gate), (weld, gate, stem)


def test_a_retired_path_is_never_a_kernel_module_or_a_shared_seam():
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    record = json.loads(R.RECORD.read_text(encoding="utf-8"))
    for weld, path in R.RETIRED_PINS:
        assert weld in ledger and weld in record, f"{weld}: a dead row"
        assert path not in R.SHARED_PACKAGE_FILES + R.SHARED_HARNESS_FILES
        assert path not in (ledger[weld].get("kernel_module") or []), (weld, path)


def test_the_retirement_is_keyed_by_weld_because_other_welds_still_pin_each_file():
    """Both files still ship, and other welds' gates still import them. A path-keyed
    rule would drop a pin those welds legitimately carry."""
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    for weld, path in R.RETIRED_PINS:
        others = sorted(key for key, entry in ledger.items()
                        if key != weld and isinstance(entry, dict)
                        and path in (entry.get("source_sha256") or {}))
        assert others, f"no other weld pins {path}; the row could be keyed by path"
        assert all(path not in R.retired_pins_of(key) for key in others)


def test_a_retired_pin_leaves_both_digest_maps_together_or_not_at_all():
    """Before the bind each weld carries its row's path; after it, neither map does. What
    never holds is half a drop: a code digest without its source digest, or a source digest
    without its code digest that the entry does not declare as drift. Holds on both sides
    of the bind; a hand edit that dropped one map's key would fail here."""
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    for weld, path in R.RETIRED_PINS:
        entry = ledger[weld]
        source, code = entry.get("source_sha256") or {}, entry.get("code_sha256") or {}
        assert path in source or path not in code, (weld, path, "code without source")
        assert path not in source or path in code or path in (entry.get("source_drift") or []), (
            weld, path, "source without code, and not declared as drift")


# ---------------------------------------------------------------------------
# THE RULE, THROUGH main(), ON A PLANTED REPOSITORY ROOT
# ---------------------------------------------------------------------------

WELD = "cuda_planted_stencil_weld"
OTHER = "cuda_planted_other_weld"
KERNEL = "meep_gpu/cuda_kernels/planted_kernel.py"
GATE = "parity/meep_gpu/gate_cuda_planted.py"
STALE = "parity/meep_gpu/gate_cuda_planted_stale.py"
GONE = "parity/meep_gpu/gate_cuda_planted_gone.py"
CAMPAIGN = "cuda_regate_planted"
CC = "8.6"
LEGS = ["flush/gate.json", "keep/gate.json"]
POLICY_OF = {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}
WHY = "the planted gate stopped importing it in a planted entry"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Plant:
    """A temporary repository root: live files, a certification record, a ledger."""

    def __init__(self, tmp_path, monkeypatch):
        self.api = tmp_path / "api"
        for rel, text in ((KERNEL, "K = 1\n"), (GATE, "G = 1\n"), (STALE, "S = 1\n")):
            (self.api / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.api / rel).write_text(text, encoding="utf-8")
        self.results = self.api / "parity" / "meep_gpu" / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        self.ledger = self.api / "meep_gpu" / "cuda_kernels" / "fingerprints.json"
        self.record = self.api / "meep_gpu" / "cuda_kernels" / "certification.json"
        monkeypatch.setattr(R, "_API", self.api)
        monkeypatch.setattr(R, "LEDGER", self.ledger)
        monkeypatch.setattr(R, "RECORD", self.record)
        monkeypatch.setitem(R.RETIRED_PINS, (WELD, STALE), WHY)
        # THE TOOL'S CLOCK, advanced by hand: two runs inside one second would stamp
        # the same recorded_utc and make a byte-identity check vacuous.
        self.now = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        plant = self

        class Clock:
            @staticmethod
            def now(tz=None):
                return plant.now

        monkeypatch.setattr(R, "datetime", Clock)

    def tick(self) -> None:
        self.now += timedelta(minutes=5)

    def live(self, rel: str) -> str:
        return _sha(self.api / rel)

    def entry(self, *pins: str, capabilities=(CC,), live=False) -> dict:
        """An entry in the per-capability shape, bound to an earlier run, pinning the
        kernel and gate (at their live digests when ``live``) plus ``pins`` at digests
        no later run will match."""
        source = ({KERNEL: self.live(KERNEL), GATE: self.live(GATE)} if live
                  else {KERNEL: "a" * 64, GATE: "b" * 64})
        source.update({rel: "e" * 64 for rel in pins})
        entry = {"source_sha256": source, "code_sha256": {rel: "f" * 64 for rel in source},
                 "kernel_module": [KERNEL], "kernels": ["planted_kernel"],
                 "legs": list(LEGS), "purpose": "a weld", "source_drift": [],
                 "status": "PASS", "_notes": {"_narrative_lives_in": "planted"}}
        for capability in capabilities:
            fastpath.bind_capability(entry, bound_before=None, capability=capability, run={
                "artifact_sha256": "c" * 64, "gate_started_utc": "2026-10-01T00:00:00Z",
                "host": "an earlier host", "recorded_utc": "2026-10-01T00:00:00Z",
                "records": "apps/api/parity/meep_gpu/results/earlier/ - 2 legs",
                "subnormal_policy": "meep_x86_flush; ieee_keep_ftz_stripped",
                "verdict_read_from": "earlier/{flush/gate.json,keep/gate.json}"},
                run_fields=R.SLOT_FIELDS)
        return entry

    def campaign(self, legs_of: dict) -> pathlib.Path:
        """``legs_of``: weld -> (imports for keep, imports for flush, flush released).
        Writes the campaign tree and a record whose re-gate block digests it."""
        root = self.results / CAMPAIGN
        record = {}
        for weld, (keep_imports, flush_imports, flush_released) in legs_of.items():
            for policy, imports in (("keep", keep_imports), ("flush", flush_imports)):
                released = flush_released if policy == "flush" else True
                path = root / weld / policy / "gate.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "canonical_verdict": {"released": released,
                                          "reasons": [] if released else ["planted"]},
                    "started_utc": "2026-10-11T00:00:00Z", "host": "the planted host",
                    "environment": {"compute_capability": CC},
                    "subnormal_policy_install": {"policy": POLICY_OF[policy]},
                    "imported_source_sha256": dict(imports)}), encoding="utf-8")
            record[weld] = {"device": "planted device", "kernel_module": KERNEL,
                            "certified_kernels": ["planted_kernel"]}
        record[CAMPAIGN] = {
            "artifacts": f"parity/meep_gpu/results/{CAMPAIGN}",
            "compute_capability": CC, "environment": {"compute_capability": CC},
            "artifact_sha256": R.manifest_digest(root),
            R.RE_GATE_KEY: [{"block": weld, "directory": weld, "legs": list(LEGS),
                             "released": True} for weld in legs_of]}
        self.record.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return root

    def imports(self, *extra: str) -> dict:
        found = {KERNEL: self.live(KERNEL), GATE: self.live(GATE)}
        found.update({rel: self.live(rel) for rel in extra})
        return found

    def write_ledger(self, entries: dict) -> str:
        text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
        self.ledger.write_text(text, encoding="utf-8")
        return text

    def text(self) -> str:
        return self.ledger.read_text(encoding="utf-8")

    def run(self, root: pathlib.Path, *extra: str) -> int:
        return R.main(["--campaign", str(root), *extra])


@pytest.fixture
def plant(tmp_path, monkeypatch):
    return Plant(tmp_path, monkeypatch)


def test_report_mode_names_the_retired_pin_and_writes_nothing(plant, capsys):
    text = plant.write_ledger({WELD: plant.entry(STALE)})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True)})
    plant.run(root)
    out = capsys.readouterr().out
    assert f"rebound  {WELD}: runs[{CC}]" in out
    assert f"RETIRED-PIN  {WELD}: {STALE} ({WHY}; its gate no longer imports it)" in out
    assert "1 pin(s) retired, 0 unchanged" in out and f"SKIPPED  {WELD}" not in out
    assert plant.text() == text


def test_write_drops_the_pin_from_both_maps_and_nothing_else(plant, capsys):
    old = plant.entry(STALE)
    plant.write_ledger({WELD: old})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    capsys.readouterr()
    entry = json.loads(plant.text())[WELD]
    assert entry["source_sha256"] == plant.imports()
    assert entry["code_sha256"] == {rel: code_digest_of_path(plant.api / rel)
                                    for rel in (KERNEL, GATE)}
    assert entry["status"] == "PASS" and entry["source_drift"] == []
    for field in ("kernel_module", "kernels", "legs", "purpose", "_notes"):
        assert entry[field] == old[field], field
    run = entry["runs"][CC]
    assert run["bound_sha256"] == fastpath.bound_digest(entry)
    assert fastpath.live_capabilities(entry) == (CC,)
    assert sorted(run[fastpath.CERTIFIED_POLICIES]) == ["flush", "keep"]
    assert f"results/{CAMPAIGN}/{WELD}/" in run["records"]


def test_the_gate_importing_the_file_again_refuses_the_entry(plant, capsys):
    text = plant.write_ledger({WELD: plant.entry(STALE)})
    root = plant.campaign({WELD: (plant.imports(STALE), plant.imports(STALE), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}: RETIRED_PINS retires ['{STALE}']" in out
    assert "imports them again" in out and "RETIRED-PIN" not in out
    assert plant.text() == text


def test_one_leg_importing_the_file_again_is_enough_to_refuse(plant, capsys):
    """Stricter than the agreed-import rule: a file one leg ran is a file the gate ran."""
    text = plant.write_ledger({WELD: plant.entry(STALE)})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(STALE), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}: RETIRED_PINS retires ['{STALE}']" in out
    assert plant.text() == text


def test_a_row_still_refuses_after_the_bind_dropped_its_pin(plant, capsys):
    """The table cannot rot silently: an entry no longer carrying the path still
    refuses when its gate imports the path again."""
    text = plant.write_ledger({WELD: plant.entry()})
    root = plant.campaign({WELD: (plant.imports(STALE), plant.imports(STALE), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}: RETIRED_PINS retires ['{STALE}']" in out
    assert plant.text() == text


def test_a_missing_pin_that_is_not_retired_still_refuses(plant, capsys):
    """Every pin outside the table is compared exactly as before the table existed."""
    text = plant.write_ledger({WELD: plant.entry(STALE, GONE)})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}: the artifact does not record ['{GONE}']" in out
    assert "RETIRED-PIN" not in out and "0 rebound" in out
    assert plant.text() == text


def test_the_same_path_on_another_weld_is_compared_as_strictly_as_ever(plant, capsys):
    """Keyed by weld: OTHER pins the same file, has no row, and its run did not record
    it, so OTHER is refused while WELD retires it."""
    other = plant.entry(STALE)
    plant.write_ledger({WELD: plant.entry(STALE), OTHER: other})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True),
                           OTHER: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {OTHER}: the artifact does not record ['{STALE}']" in out
    assert f"RETIRED-PIN  {WELD}: {STALE}" in out and f"RETIRED-PIN  {OTHER}" not in out
    written = json.loads(plant.text())
    assert written[OTHER] == other
    assert STALE not in written[WELD]["source_sha256"]


def test_an_entry_that_does_not_rebind_keeps_its_retired_pin(plant, capsys):
    """No drop without a rebind: a dropped pin with no fresh run would leave the
    entry's recorded run bound to digests nobody re-measured."""
    idle = plant.entry(STALE)
    text = plant.write_ledger({WELD: idle})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), False)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}: flush/gate.json is not released" in out
    assert "RETIRED-PIN" not in out
    assert plant.text() == text


def test_a_drop_that_would_strand_another_capability_still_refuses(plant, capsys):
    """Dropping a pin moves the bound digest, so another architecture's record on the
    old bytes is stranded: refused by name unless superseded out loud. The kernel and
    gate are pinned at their live digests, so the drop is the only byte that moves."""
    text = plant.write_ledger({WELD: plant.entry(STALE, capabilities=(CC, "9.0"),
                                                 live=True)})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert f"SKIPPED  {WELD}:" in out and "9.0" in out and "RETIRED-PIN" not in out
    assert plant.text() == text
    plant.run(root, "--write", "--supersede", "9.0")
    out = capsys.readouterr().out
    assert f"RETIRED-PIN  {WELD}: {STALE}" in out and f"SUPERSEDED  {WELD}" in out
    assert STALE not in json.loads(plant.text())[WELD]["source_sha256"]


def test_a_second_write_is_byte_identical_and_retires_nothing(plant, capsys):
    plant.write_ledger({WELD: plant.entry(STALE)})
    root = plant.campaign({WELD: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    first = plant.text()
    capsys.readouterr()
    plant.tick()
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert plant.text() == first
    assert "RETIRED-PIN" not in out
    assert f"rebound  {WELD}: runs[{CC}] (live: {CC})  (unchanged)" in out
    assert "0 pin(s) retired, 1 unchanged" in out


def test_a_rerun_restates_an_entry_without_a_retired_pin_byte_for_byte(plant, capsys):
    """The rerun rule is not the retirement's: any weld rebound twice on one campaign
    keeps its recorded time, so the second write moves no byte."""
    plant.write_ledger({OTHER: plant.entry()})
    root = plant.campaign({OTHER: (plant.imports(), plant.imports(), True)})
    plant.run(root, "--write")
    first = copy.deepcopy(json.loads(plant.text()))
    text = plant.text()
    assert first[OTHER]["runs"][CC]["recorded_utc"] == "2026-10-11T12:00:00Z"
    capsys.readouterr()
    plant.tick()
    plant.run(root, "--write")
    out = capsys.readouterr().out
    assert plant.text() == text, "the second write moved bytes"
    assert json.loads(plant.text())[OTHER]["runs"][CC]["recorded_utc"] == \
        first[OTHER]["runs"][CC]["recorded_utc"]
    assert "0 pin(s) retired, 1 unchanged" in out
