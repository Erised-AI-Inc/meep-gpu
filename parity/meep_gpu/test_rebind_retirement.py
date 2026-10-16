"""A weld whose gate was deleted is retired by name, by the rebind tool, and by nothing else.

``rebind_triton_welds.py`` binds a ledger entry to a fresh campaign's artifact for its
gate. An entry whose gate left the tree can never be bound again, and before
``RETIRED_WELDS`` existed the tool reported it UNCOVERED ("needs a device re-run") and
left it in the ledger, where every weld contract test kept reading a stale weld. These
tests pin the retirement: the tables are disjoint, the one retired key is the deleted
expansion gate's, the drop is exact and idempotent, ``--only`` is honoured, the two
guards refuse, and a ``--write`` run that rebinds nothing still persists the drop.

Host-only: no GPU, no MEEP. The real ledger is read, never written; the ``main()`` legs
write a temporary ledger.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rebind_triton_welds as R  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

KEY = "triton_unified_expansion_device_gate"


def _entry() -> dict:
    return {"status": "PASS", "source_sha256": {"meep_gpu/a.py": "1" * 64},
            "purpose": "a weld"}


def test_the_retired_and_rebound_tables_are_disjoint():
    assert not set(R.RETIRED_WELDS) & set(R.CAMPAIGN_DIRS)


def test_the_deleted_expansion_gate_is_the_retired_key():
    assert set(R.RETIRED_WELDS) == {KEY}
    gate, why = R.RETIRED_WELDS[KEY]
    assert gate == "parity/meep_gpu/gate_triton_unified_expansion.py"
    assert not (R._API / gate).exists(), "the retired gate is back in the checkout"
    assert why
    cited = {key for _family, key in fastpath.ARM_CERTIFICATION.values()}
    assert KEY not in cited, "a Triton arm cites the retired weld"


def test_the_shipped_ledger_entry_is_dropped_and_nothing_else_moves():
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    before = copy.deepcopy(ledger)
    dropped = R.retire_welds(ledger)
    assert [key for key, _why in dropped] == ([KEY] if KEY in before else [])
    assert set(before) - set(ledger) == {key for key, _why in dropped}
    for key, value in ledger.items():
        assert value == before[key], key


def test_the_drop_is_exact_and_idempotent():
    ledger = {KEY: _entry(), "triton_complex_device_gate": _entry(), "_note": "kept"}
    other = copy.deepcopy(ledger["triton_complex_device_gate"])
    dropped = R.retire_welds(ledger)
    assert [key for key, _why in dropped] == [KEY]
    assert "not in the checkout" in dropped[0][1]
    assert ledger == {"triton_complex_device_gate": other, "_note": "kept"}
    assert R.retire_welds(ledger) == []


def test_only_names_what_may_be_dropped():
    ledger = {KEY: _entry(), "triton_complex_device_gate": _entry()}
    assert R.retire_welds(ledger, {"triton_complex_device_gate"}) == []
    assert KEY in ledger
    assert [key for key, _why in R.retire_welds(ledger, {KEY})] == [KEY]


def test_a_gate_back_in_the_checkout_refuses(monkeypatch):
    present = pathlib.Path(__file__).resolve().relative_to(R._API).as_posix()
    monkeypatch.setitem(R.RETIRED_WELDS, KEY, (present, "why"))
    ledger = {KEY: _entry()}
    with pytest.raises(SystemExit, match="is in the checkout"):
        R.retire_welds(ledger)
    assert KEY in ledger


def test_a_cited_key_refuses(monkeypatch):
    cited = dict(fastpath.ARM_CERTIFICATION, **{"some arm": ("family", KEY)})
    monkeypatch.setattr(fastpath, "ARM_CERTIFICATION", cited)
    ledger = {KEY: _entry()}
    with pytest.raises(SystemExit, match="still cites it"):
        R.retire_welds(ledger)
    assert KEY in ledger


#: A mapped weld the campaign does not hold: the pass walks it to the shape check and the
#: campaign lookup, and skips it, so the retirement must survive a key the loop visits.
MAPPED = "triton_complex_device_gate"


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A temporary API root holding a ledger and one empty in-tree campaign."""
    api = tmp_path / "api"
    campaign = api / "parity" / "meep_gpu" / "results" / "triton_fleet_x"
    campaign.mkdir(parents=True)
    ledger = api / "fingerprints.json"
    ledger.write_text(json.dumps({KEY: _entry(), MAPPED: _entry(), "_note": "kept"},
                                 indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(R, "_API", api)
    monkeypatch.setattr(R, "LEDGER", ledger)
    return ledger, campaign


def test_report_mode_names_the_retirement_and_writes_nothing(scratch, capsys):
    ledger, campaign = scratch
    text = ledger.read_text(encoding="utf-8")
    assert R.main(["--campaign", str(campaign)]) == 1      # MAPPED is skipped
    out = capsys.readouterr().out
    assert f"RETIRED   {KEY}:" in out and "1 skipped" in out and "1 retired" in out
    assert f"SKIPPED   {MAPPED}:" in out and "UNCOVERED" not in out
    assert ledger.read_text(encoding="utf-8") == text


def test_write_persists_the_retirement_when_nothing_rebinds(scratch, capsys):
    ledger, campaign = scratch
    assert R.main(["--campaign", str(campaign), "--write"]) == 1
    out = capsys.readouterr().out
    assert "0 rebound" in out and "1 retired" in out
    assert json.loads(ledger.read_text(encoding="utf-8")) == {MAPPED: _entry(),
                                                               "_note": "kept"}
