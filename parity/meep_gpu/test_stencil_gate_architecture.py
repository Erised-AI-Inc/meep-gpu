"""The complex off-diagonal stencil gate licenses a card only from that card's own record.

``gate_cuda_complex_offdiag_stencil_welds.py`` binds three expansion licences, one per
half, and used to read them from fixed archive paths checking only their subnormal
policy. On an architecture other than the one those records were measured on, it would
have licensed the card from another card's measurement. It now compares each record's
stated ``environment.compute_capability`` with the card the run is on, refusing by name
on a mismatch, and takes ``--expansion-probe`` for the card's own record.

The arbiters are stubbed here: they are tested where they live, and what changed is
which record each half reads and whether it may license this card. The policy check is
the real one.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cuda_predicate_battery as battery  # noqa: E402
import gate_cuda_complex_no_pml as no_pml_gate  # noqa: E402
import gate_cuda_complex_offdiag_stencil_welds as gate  # noqa: E402

from meep_gpu.triton_kernels import complex_fields, folded_complex  # noqa: E402

HALVES = ("constitutive", "folded_curl", "no_pml_curl")


def record(policy: str, capability) -> dict:
    environment = {} if capability is None else {"compute_capability": capability}
    return {"subnormal_policy": {"resolved": policy}, "environment": environment,
            "patterns": []}


def write(path: pathlib.Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """The archive records at the paths the gate defaults to, measured on cc 8.6."""
    monkeypatch.setattr(gate, "_REPO_API", str(tmp_path))
    for policy, paths in (("keep", (battery.COMPLEX_PROBE_RECORD,
                                    battery.COMPLEX_NO_PML_PROBE_RECORD)),
                          ("flush", (gate.COMPLEX_PROBE_RECORD_FLUSH,
                                     no_pml_gate.PROBE_RECORD_FLUSH))):
        for relative in paths:
            write(tmp_path / relative, record(policy, "8.6"))
    licensed = lambda _record: {"arm": "FMA_V1", "basis": "measured"}  # noqa: E731
    monkeypatch.setattr(complex_fields, "expansion_license", licensed)
    monkeypatch.setattr(folded_complex, "parity_expansion_license", licensed)
    return tmp_path


@pytest.mark.parametrize("policy", ["keep", "flush"])
def test_the_certified_card_reads_the_archive_records_unchanged(archive, policy):
    """On the architecture the archive records were measured on, nothing moves."""
    licence = gate.licences(policy, None, "8.6")
    assert gate.unusable_halves(licence) == {}
    expected = {"keep": (battery.COMPLEX_PROBE_RECORD, battery.COMPLEX_PROBE_RECORD,
                         battery.COMPLEX_NO_PML_PROBE_RECORD),
                "flush": (gate.COMPLEX_PROBE_RECORD_FLUSH,
                          gate.COMPLEX_PROBE_RECORD_FLUSH,
                          no_pml_gate.PROBE_RECORD_FLUSH)}[policy]
    assert tuple(licence[half]["record"] for half in HALVES) == expected
    assert all(licence[half]["record_compute_capability"] == "8.6" for half in HALVES)


@pytest.mark.parametrize("policy", ["keep", "flush"])
def test_an_archive_record_from_another_architecture_refuses_every_half(archive, policy):
    licence = gate.licences(policy, None, "9.0")
    refused = gate.unusable_halves(licence)
    assert sorted(refused) == sorted(HALVES)
    for half in HALVES:
        (why,) = licence[half]["architecture_reasons"]
        assert "measured on compute capability 8.6" in why and "on 9.0" in why
        assert "--expansion-probe" in why


def test_a_record_from_another_architecture_handed_in_is_refused_too(archive, tmp_path):
    other = write(tmp_path / "a6000" / "gate.json", record("keep", "8.6"))
    licence = gate.licences("keep", other, "9.0")
    assert sorted(gate.unusable_halves(licence)) == sorted(HALVES)


@pytest.mark.parametrize("policy", ["keep", "flush"])
def test_this_cards_own_record_licenses_every_half(archive, tmp_path, policy):
    own = write(tmp_path / "h100" / "gate.json", record(policy, "9.0"))
    licence = gate.licences(policy, own, "9.0")
    assert gate.unusable_halves(licence) == {}
    assert all(licence[half]["record"] == own for half in HALVES)
    assert all(licence[half]["architecture_reasons"] == [] for half in HALVES)


def test_a_record_that_names_no_architecture_is_refused(archive, tmp_path):
    silent = write(tmp_path / "silent" / "gate.json", record("keep", None))
    licence = gate.licences("keep", silent, "9.0")
    assert sorted(gate.unusable_halves(licence)) == sorted(HALVES)
    assert "states no compute capability" in licence["constitutive"]["architecture_reasons"][0]


def test_with_no_device_there_is_no_card_to_compare_against(archive):
    licence = gate.licences("keep", None, None)
    assert all(licence[half]["architecture_reasons"] == [] for half in HALVES)


def test_a_record_handed_in_still_passes_the_policy_check(archive, tmp_path):
    """The override changes WHICH record is read, never whether its policy is checked."""
    flushed = write(tmp_path / "h100_flush" / "gate.json", record("flush", "9.0"))
    with pytest.raises(SystemExit, match="POLICY-CONDITIONAL"):
        gate.licences("keep", flushed, "9.0")


def test_the_capability_comparison_normalises_its_spellings(archive, tmp_path):
    """A record and a device that name one architecture differently are one architecture."""
    own = write(tmp_path / "h100" / "gate.json", record("keep", [9, 0]))
    assert gate.licences("keep", own, "9.0")["constitutive"]["architecture_reasons"] == []


def test_an_architecture_refusal_alone_makes_a_half_unusable():
    licence = {half: {"arm": "FMA_V1", "refusals": [], "policy_reasons": [],
                      "architecture_reasons": []} for half in HALVES}
    assert gate.unusable_halves(licence) == {}
    licence["no_pml_curl"]["architecture_reasons"] = ["measured elsewhere"]
    assert list(gate.unusable_halves(licence)) == ["no_pml_curl"]


def test_a_licence_taken_from_the_environment_default_table_is_refused_on_a_device(
        archive, monkeypatch):
    """That table records backend, machine and CuPy version but not the architecture,
    so a fallback to it would license this card from another card's row."""
    fallback = lambda _record: {"arm": "FMA_V1", "basis": "environment_default"}  # noqa: E731
    monkeypatch.setattr(complex_fields, "expansion_license", fallback)
    monkeypatch.setattr(folded_complex, "parity_expansion_license", fallback)
    licence = gate.licences("keep", None, "8.6")
    assert sorted(gate.unusable_halves(licence)) == sorted(HALVES)
    assert "not 'measured'" in licence["no_pml_curl"]["architecture_reasons"][0]
    # with no device there is no card to describe, and the fallback is not judged here
    assert gate.unusable_halves(gate.licences("keep", None, None)) == {}


def test_a_record_that_cannot_be_read_is_a_named_refusal_not_a_crash(archive, tmp_path):
    licence = gate.licences("keep", str(tmp_path / "absent.json"), "9.0")
    assert sorted(gate.unusable_halves(licence)) == sorted(HALVES)
    assert all("could not be read" in licence[half]["refusals"][0] for half in HALVES)


FIXTURES = HERE / "fixtures" / "expansion_records"


@pytest.mark.parametrize("policy, name", [("keep", "unified_keep_cc86.json"),
                                          ("flush", "unified_flush_cc86.json")])
def test_one_unified_record_licenses_every_half_under_the_real_arbiters(
        policy, name, tmp_path):
    """The claim the override rests on, checked with the arbiters themselves.

    The fixtures are trimmed copies of unified expansion records measured on
    compute capability 8.6 (cache paths scrubbed; every field the arbiters read is
    kept). One record has to license all three halves with the same measured arm,
    on the card that measured it -- and the same record, relabelled as another
    card's, has to license that card and refuse this one.
    """
    record = str(FIXTURES / name)
    licence = gate.licences(policy, record, "8.6")
    assert gate.unusable_halves(licence) == {}, gate.unusable_halves(licence)
    assert {licence[half]["arm"] for half in HALVES} == {"FMA_V1"}
    assert {licence[half]["basis"] for half in HALVES} == {"measured"}

    relabelled = json.loads(pathlib.Path(record).read_text())
    relabelled["environment"]["compute_capability"] = "9.0"
    other = write(tmp_path / "h100" / name, relabelled)
    assert gate.unusable_halves(gate.licences(policy, other, "9.0")) == {}
    assert sorted(gate.unusable_halves(gate.licences(policy, other, "8.6"))) == sorted(HALVES)
