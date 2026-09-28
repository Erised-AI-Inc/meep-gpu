"""Host-safe checks for the fixed, source-welded Metal corpus harness."""

from __future__ import annotations

import json

import pytest

from parity.meep_gpu import recut_metal_corpus as corpus


def _ledger(case: str, facts: dict) -> dict:
    return {"module": "test_example.py", "case": case, "facts": facts}


def _raw(row: str, facts: dict, *, measured: bool = True) -> dict:
    return {"module": "test_example.py", "row": row, "facts": facts,
            "measured": measured, "slots": ["step_B"], "selected": {}}


def test_accepted_rows_uses_only_true_and_refuses_duplicate_identities(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join([
            json.dumps({"script": "a.py", "accepted": True}),
            json.dumps({"script": "b.py", "accepted": False}),
            json.dumps({"script": "a.py", "accepted": True}),
        ]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        corpus.accepted_rows(ledger, leg="examples")


def test_source_inventory_reads_each_selected_example_and_test_module_once(tmp_path):
    examples, tests = tmp_path / "examples", tmp_path / "tests"
    examples.mkdir()
    tests.mkdir()
    (examples / "one.py").write_text("example\n", encoding="utf-8")
    (tests / "test_one.py").write_text("test\n", encoding="utf-8")
    sources, errors = corpus.source_inventory(
        examples_dir=examples, tests_dir=tests,
        examples=[{"script": "one.py"}],
        tests=[{"module": "test_one.py", "case": "Case.a"},
               {"module": "test_one.py", "case": "Case.b"}],
    )
    assert not errors
    assert set(sources) == {str((examples / "one.py").resolve()),
                            str((tests / "test_one.py").resolve())}


def test_parameterized_match_uses_exact_facts_not_generated_method_name():
    expected = [_ledger("Case.original_a", {"k": 1}), _ledger("Case.original_b", {"k": 2})]
    raw = [_raw("Case.__mps_corpus_0", {"k": 2}),
           _raw("Case.__mps_corpus_1", {"k": 1}),
           _raw("Case.nonledger", {"k": 99})]

    matched, errors = corpus.match_parameterized_rows(raw, expected)

    assert not errors
    assert [row["row"] for row in matched] == ["Case.original_a", "Case.original_b"]
    assert [row["shim_row"] for row in matched] == ["Case.__mps_corpus_1", "Case.__mps_corpus_0"]


def test_parameterized_equal_facts_group_is_marked_as_a_permutation():
    facts = {"cell": [1, 2, 3]}
    expected = [_ledger("Case.one", facts), _ledger("Case.two", facts)]
    raw = [_raw("Case.generated_1", facts), _raw("Case.generated_0", facts)]

    matched, errors = corpus.match_parameterized_rows(raw, expected)

    assert not errors
    assert [row["row"] for row in matched] == ["Case.one", "Case.two"]
    assert all(row["group_permutation_unresolved"] for row in matched)


def test_parameterized_facts_group_size_mismatch_is_a_hard_error():
    facts = {"cell": [1, 2, 3]}
    matched, errors = corpus.match_parameterized_rows(
        [_raw("Case.generated_0", facts)], [_ledger("Case.one", facts), _ledger("Case.two", facts)])

    assert not matched
    assert len(errors) == 1
    assert "1 generated rows but 2 ledger rows" in errors[0]


def test_merge_sources_rejects_a_mixed_source_tree():
    sources, errors = corpus.merge_sources([
        {"leg": "examples", "row": "a", "measured": True,
         "source_sha256": {"/api/a.py": "a" * 64}},
        {"leg": "tests", "row": "b", "measured": True,
         "source_sha256": {"/api/a.py": "b" * 64}},
    ])
    assert sources == {"/api/a.py": "a" * 64}
    assert errors == ["mixed runtime source digest for /api/a.py"]


def test_verify_runtime_sources_rehashes_the_source_at_campaign_close(tmp_path):
    source = tmp_path / "planner.py"
    source.write_text("before\n", encoding="utf-8")
    sources = {str(source): corpus.sha256(source)}
    assert corpus.verify_runtime_sources(sources, root=tmp_path) == []
    source.write_text("after\n", encoding="utf-8")
    assert corpus.verify_runtime_sources(sources, root=tmp_path) == [
        f"runtime source changed during campaign: {source}"]


def test_probe_validation_uses_special_kzs_beta_specific_parser(monkeypatch, tmp_path):
    probes = {}
    for name in corpus.PROBE_ENVIRONMENTS:
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        probes[name] = path

    class Parser:
        @staticmethod
        def expansion_from_probe(_record):
            return "FMA_V1"

    class Special:
        @staticmethod
        def beta_expansion_from_probe(_record):
            return "NAIVE"

    import meep_gpu.metal_kernels as kernels

    monkeypatch.setattr(kernels, "complex_fields", Parser, raising=False)
    monkeypatch.setattr(kernels, "folded_complex", Parser, raising=False)
    monkeypatch.setattr(kernels, "cylindrical_complex", Parser, raising=False)
    monkeypatch.setattr(kernels, "special_kz", Special, raising=False)
    assert corpus.validate_probe_records(probes) == {
        "complex": "FMA_V1", "special_kz": "NAIVE",
        "folded_complex": "FMA_V1", "cylindrical_complex": "FMA_V1",
    }


def test_canonicalize_preserves_a_missing_ledger_row_as_unmeasured():
    expected = [_ledger("Case.one", {"a": 1})]
    rows, errors = corpus.canonicalize_rows(
        leg="tests", raw=[], expected=expected, parameterized=False)
    assert rows == [{"leg": "tests", "row": "Case.one", "measured": False,
                     "note": "child returned no canonical row", "ledger_facts": {"a": 1},
                     "module": "test_example.py"}]
    assert errors == ["missing canonical tests row Case.one"]


def test_module_child_dispatch_uses_output_jsonl_keyword(monkeypatch, tmp_path):
    """The parent must reach module replay instead of failing on its output name."""
    probes = {name: tmp_path / f"{name}.json" for name in corpus.PROBE_ENVIRONMENTS}
    observed = {}

    def fake_child_module(*, module_path, wanted, output_jsonl, progress_log,
                          probes, case_timeout, take_all):
        observed.update({
            "module_path": module_path,
            "wanted": wanted,
            "output_jsonl": output_jsonl,
            "progress_log": progress_log,
            "probes": probes,
            "case_timeout": case_timeout,
            "take_all": take_all,
        })
        return 23

    monkeypatch.setattr(corpus, "_probe_paths", lambda _values: probes)
    monkeypatch.setattr(corpus, "validate_probe_records", lambda _probes: {})
    monkeypatch.setattr(corpus, "child_module", fake_child_module)

    result = corpus.main([
        "--child", "module", "--module-path", str(tmp_path / "module.py"),
        "--wanted-json", '["Case.test"]', "--out-jsonl", str(tmp_path / "rows.jsonl"),
        "--progress-log", str(tmp_path / "progress.log"), "--case-timeout", "17",
        "--take-all",
    ])

    assert result == 23
    assert observed == {
        "module_path": (tmp_path / "module.py").resolve(),
        "wanted": {"Case.test"},
        "output_jsonl": tmp_path / "rows.jsonl",
        "progress_log": tmp_path / "progress.log",
        "probes": probes,
        "case_timeout": 17.0,
        "take_all": True,
    }
