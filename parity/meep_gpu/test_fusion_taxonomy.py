"""The reconciled taxonomy's seventh floor and the boards' standing censuses.

What each test pins, and what would break it:

- ``Taxonomy.finish`` lists the SERVED instances beside the other seven buckets and
  groups them by product, so an instance-by-instance diff of two boards no longer has
  to rebuild each board's universe from its own per-row tables.
- ``headline_agreement`` walks a finished artifact for every occurrence of a
  :data:`fusion_taxonomy.SERVED_HEADLINE_KEYS` name at any depth and refuses the cut
  where one disagrees with the served bucket. The null controls plant the exact split
  the 2026-09-06 ``_hd`` boards carried (a three-seam total under the headline key
  beside a four-seam bucket), a disagreement two levels down, an artifact with no
  served key at all, and a block whose served list does not match its count.
- The three builders' argument-free default censuses are the STANDING ones: each
  exists, each carries the m = 0 complex cylindrical fact that separates it from the
  superseded record it replaced (measured on ``examples:dipole_in_vacuum_cyl_off_axis.py``,
  the one row the 2026-09-03 and 2026-09-04 censuses disagree on), and the CUDA one
  is the census whose recorded subject manifest this tree hashes to -- read with the
  boards' own row rule and the CUDA board's own :func:`subject_pin`.

The builders are read as TEXT for their defaults rather than imported: importing the
Metal board pulls ``meep_gpu.metal_kernels`` (and torch) at module load, which the
default-census fact does not need.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import dispatch_agreement  # noqa: E402
import fusion_taxonomy  # noqa: E402

RESULTS = _HERE / "results"
OFF_AXIS = "examples:dipole_in_vacuum_cyl_off_axis.py"


# --- the served ledger ---------------------------------------------------------------


def _taxonomy(rows: int = 2, e_to_p: int = 0) -> fusion_taxonomy.Taxonomy:
    return fusion_taxonomy.Taxonomy(
        "metal", 3 * rows + e_to_p, rows, "measured, none",
        {"B_to_H": rows, "D_to_E": rows, "H_to_D": rows, "E_to_P": e_to_p})


def _filled(h_to_d_served: bool = True) -> dict:
    taxonomy = _taxonomy()
    for row in ("a", "b"):
        taxonomy.add("served", "B_to_H", row, "served", {"product": "pair_B"})
        taxonomy.add("served", "D_to_E", row, "served", {"product": "pair_D"})
    if h_to_d_served:
        taxonomy.add("served", "H_to_D", "a", "served", {"product": "hd_pair"})
    else:
        taxonomy.add("buildable_not_built", "H_to_D", "a", "not built", {})
    taxonomy.add("buildable_not_built", "H_to_D", "b", "not built", {})
    return taxonomy.finish(served_expected=5 if h_to_d_served else 4,
                           served_expected_source="test")


def test_finish_lists_the_served_instances_and_groups_them_by_product():
    block = _filled()
    assert set(block["instances"]) == set(fusion_taxonomy.BUCKETS)
    assert block["instances"]["served"] == [
        ("B_to_H", "a"), ("B_to_H", "b"), ("D_to_E", "a"), ("D_to_E", "b"),
        ("H_to_D", "a")]
    assert len(block["instances"]["served"]) == block["buckets"]["served"] == 5
    assert block["served_by_product"] == {
        "pair_B": [("B_to_H", "a"), ("B_to_H", "b")],
        "pair_D": [("D_to_E", "a"), ("D_to_E", "b")],
        "hd_pair": [("H_to_D", "a")]}
    # The eight lists partition PRICED exactly, so a diff of two boards can rebuild
    # {(seam, row): bucket} from this block alone.
    listed = sum(len(v) for v in block["instances"].values())
    assert listed == block["tiers"]["PRICED"]["value"] == 6


# --- floor 7: the headline derives from the ledger -----------------------------------


def test_the_headline_keys_are_the_ones_the_three_boards_publish():
    keys = set(fusion_taxonomy.SERVED_HEADLINE_KEYS)
    assert {"served_by_predicate", "served_by_a_fused_product",
            "served_by_a_fused_product_today", "seam_credited_total",
            "served_counted"} <= keys
    # And every builder writes at least one of them from the taxonomy, by text: a
    # builder that stopped deriving would still carry the key but not this expression.
    for name in ("build_fusion_matrix.py", "build_triton_fusion_matrix.py",
                 "build_cuda_fusion_matrix.py"):
        text = (_HERE / name).read_text(encoding="utf-8")
        assert 'taxonomy_block["buckets"]["served"]' in text, name
        assert "fusion_taxonomy.headline_agreement(" in text, name


def test_the_withheld_sentence_has_one_home():
    # Byte-pinned to the string results/fusion_matrix_triton_2026-09-08_regate filed
    # 70 times, so the fourth seam's withheld rows join the three-seam walk's under
    # ONE sub-reason key rather than a spelling per seam.
    assert fusion_taxonomy.credit_withheld_sub_reason("x") == (
        "a product exists on this cell (x) and its predicate ADMITS, but its RELEASE "
        "BINDING is stale — credit withheld. The KERNEL is not what is missing; a "
        "re-gate against the shipped bytes is")
    # And by text: both callers call the helper and neither carries a private copy,
    # so a copy cannot regrow in either. The second fragment is the one the Triton
    # walk's own f-string carried contiguously before 2026-09-10.
    for name in ("build_triton_fusion_matrix.py", "h_to_d_seam.py"):
        text = (_HERE / name).read_text(encoding="utf-8")
        assert "fusion_taxonomy.credit_withheld_sub_reason(" in text, name
        assert "credit withheld. The KERNEL" not in text, name
        assert "RELEASE BINDING is stale — credit" not in text, name


def test_a_headline_derived_from_the_ledger_agrees_and_names_its_paths():
    block = _filled()
    result = {"headline": {"served_by_predicate": 5,
                           "served_in_dispatch_measurement": {"served_counted": 5}},
              "served_by_a_fused_product": 5,
              "served_three_seam_ledger": 4,
              "disjointness_cross_check": {"seam_credited_total": 5}}
    agreement = fusion_taxonomy.headline_agreement(result, block, "metal")
    assert agreement["agree"] is True and agreement["served"] == 5
    assert sorted(agreement["paths_checked"]) == [
        "disjointness_cross_check.seam_credited_total",
        "headline.served_by_predicate",
        "headline.served_in_dispatch_measurement.served_counted",
        "served_by_a_fused_product"]
    assert agreement["per_seam"] == {"B_to_H": 2, "D_to_E": 2, "H_to_D": 1}


def test_the_2026_09_06_split_is_refused_by_name():
    """The exact shape the `_hd` boards carried: the three-seam walk's total under the
    headline key beside a four-seam served bucket."""
    block = _filled()
    result = {"headline": {"served_by_predicate": 4}, "served_by_a_fused_product": 4,
              "taxonomy": block}
    with pytest.raises(SystemExit, match="headline.served_by_predicate=4") as raised:
        fusion_taxonomy.headline_agreement(result, block, "metal")
    assert "served_by_a_fused_product=4" in str(raised.value)
    assert "files 5 seam-instances as SERVED" in str(raised.value)


def test_a_disagreement_two_levels_down_is_still_found():
    block = _filled()
    result = {"headline": {"served_by_predicate": 5},
              "aggregate": {"served_in_dispatch_measurement": {
                  "buckets_total_the_served_count": {"served_counted": 4}}}}
    with pytest.raises(SystemExit, match=re.escape(
            "aggregate.served_in_dispatch_measurement."
            "buckets_total_the_served_count.served_counted=4")):
        fusion_taxonomy.headline_agreement(result, block, "triton")


def test_an_artifact_with_no_served_key_is_refused():
    block = _filled()
    with pytest.raises(SystemExit, match="publishes none of"):
        fusion_taxonomy.headline_agreement({"headline": {"denominator": 6}}, block, "cuda")


def test_the_dispatch_keys_are_the_ones_the_three_boards_publish():
    """FLOOR 8's own key set, both directions, and that all three boards call it.

    The served floor had this test and the dispatch axis had no floor at all, which is
    how the Metal board shipped a hardcoded ``served_by_a_fused_product_in_dispatch: 0``
    beside its own measured 176 in one artifact (2026-09-11).
    """
    keys = set(dispatch_agreement.DISPATCH_HEADLINE_KEYS)
    assert {"served_in_dispatch", "served_by_a_fused_product_in_dispatch"} <= keys
    for name in ("build_fusion_matrix.py", "build_triton_fusion_matrix.py",
                 "build_cuda_fusion_matrix.py"):
        text = (_HERE / name).read_text(encoding="utf-8")
        assert "dispatch_agreement.dispatch_agreement(" in text, name
    # AND THE LITERAL CANNOT COME BACK. The Metal board is the one that carried it.
    metal = (_HERE / "build_fusion_matrix.py").read_text(encoding="utf-8")
    assert '"served_by_a_fused_product_in_dispatch": 0,' not in metal


def test_no_board_reads_its_dispatch_measurement_before_it_is_measured():
    """THE DEFECT THIS FILE'S OWN FIRST VERSION SHIPPED, and why it stayed green.

    The repair that replaced the Metal board's hardcoded dispatch 0 put
    ``dispatch["served_in_dispatch"]`` into the ``result`` dict literal — which is built
    about two hundred lines BEFORE ``dispatch`` is assigned. Every Metal cut then raised
    ``UnboundLocalError`` and the board could not produce an artifact at all. The test
    written beside it asserted the presence of that expression as a SUBSTRING of the
    source, so it passed on a board that was dead, and the full suite passed with it:
    a board build is not part of the suite, and a grep is not an execution.

    So this checks the ORDER, statically, in each board's own parse tree: every
    subscript of the local holding the dispatch measurement must come after that local
    is assigned. A static order check is what a substring check should have been — it
    cannot be satisfied by text that happens to appear in the file.
    """
    import ast  # noqa: PLC0415

    boards = {"build_fusion_matrix.py": "dispatch",
              "build_triton_fusion_matrix.py": "dispatch_reach",
              "build_cuda_fusion_matrix.py": "dispatch_reach"}
    for name, local in sorted(boards.items()):
        tree = ast.parse((_HERE / name).read_text(encoding="utf-8"))
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)]
        seen = False
        for function in functions:
            assigns = [node.lineno for node in ast.walk(function)
                       if isinstance(node, ast.Assign)
                       for target in node.targets
                       if isinstance(target, ast.Name) and target.id == local]
            reads = [node.lineno for node in ast.walk(function)
                     if isinstance(node, ast.Subscript)
                     and isinstance(node.value, ast.Name)
                     and node.value.id == local]
            if not reads:
                continue
            assert assigns, (
                f"{name}:{function.name} subscripts {local!r} at line {min(reads)} and "
                f"never assigns it in that function — an UnboundLocalError on every cut")
            assert min(assigns) < min(reads), (
                f"{name}:{function.name} reads {local}[...] at line {min(reads)} but "
                f"only assigns {local} at line {min(assigns)}: the board raises "
                f"UnboundLocalError before it can write an artifact. This is exactly "
                f"how the Metal dispatch-total repair shipped a dead board behind a "
                f"green suite")
            seen = True
        assert seen, (
            f"{name} no longer reads {local}[...] anywhere — either the local was "
            f"renamed or the dispatch measurement stopped being consulted, and this "
            f"check has gone vacuous")


def test_each_board_can_be_imported_and_its_main_compiles():
    """A board that cannot even be imported publishes nothing, and nothing noticed.

    Cheap, and it closes the other half of the same gap: the suite never executes a
    board, so an import-time or compile-time break in one is invisible until someone
    cuts a matrix by hand. Importing is safe — all three do their work in ``main``.
    """
    import importlib  # noqa: PLC0415

    for name in ("build_fusion_matrix", "build_triton_fusion_matrix",
                 "build_cuda_fusion_matrix"):
        module = importlib.import_module(name)
        assert callable(getattr(module, "main", None)), name


def test_a_dispatch_headline_read_from_the_measurement_agrees_and_names_its_paths():
    measured = {"served_in_dispatch": 176, "is_an_upper_bound": True}
    result = {"headline": {"served_in_dispatch": 176,
                           "served_in_dispatch_measurement": measured},
              "served_by_a_fused_product_in_dispatch": 176,
              "served_by_a_fused_product": 531}
    agreement = dispatch_agreement.dispatch_agreement(result, measured, "metal")
    assert agreement["agree"] is True and agreement["served_in_dispatch"] == 176
    assert agreement["is_an_upper_bound"] is True
    assert sorted(agreement["paths_checked"]) == [
        "headline.served_in_dispatch",
        "headline.served_in_dispatch_measurement.served_in_dispatch",
        "served_by_a_fused_product_in_dispatch"]
    assert agreement["conditional_totals_not_compared"] == []


def test_the_hardcoded_metal_zero_is_refused_by_name():
    """THE EXACT SHAPE THE 2026-09-11 METAL BOARD CARRIED, armed as a regression.

    A literal 0 at the top level beside a measured 176 two levels down. Before FLOOR 8
    this cut cleanly and `headline_agreement` reported `agree: true`, because every
    floor the boards had was about the served axis.
    """
    measured = {"served_in_dispatch": 176}
    result = {"headline": {"served_in_dispatch": 176,
                           "served_in_dispatch_measurement": measured},
              "served_by_a_fused_product_in_dispatch": 0}
    with pytest.raises(SystemExit, match=re.escape(
            "served_by_a_fused_product_in_dispatch=0")) as raised:
        dispatch_agreement.dispatch_agreement(result, measured, "metal")
    assert "measured 176" in str(raised.value)
    assert "READ from the measurement, never typed beside it" in str(raised.value)


def test_a_declared_condition_container_is_listed_and_not_compared():
    """The CUDA board's by-DEFAULT 0 beside its by-preference 209 — two questions.

    Both are correct, and an equality floor that did not know the difference would
    refuse a correct board; the wrong repair would be deleting the by-default number,
    which is the one a user who sets no preference actually needs.
    """
    measured = {"served_in_dispatch": 209,
                "by_default_precedence": {"served_in_dispatch": 0}}
    result = {"headline": {"served_in_dispatch": 209},
              "served_in_dispatch_measurement": measured}
    agreement = dispatch_agreement.dispatch_agreement(result, measured, "cuda")
    assert agreement["agree"] is True and agreement["served_in_dispatch"] == 209
    assert agreement["conditional_totals_not_compared"] == [{
        "path": "served_in_dispatch_measurement.by_default_precedence."
                "served_in_dispatch",
        "value": 0,
        "condition": "by_default_precedence"}]
    assert "served_in_dispatch_measurement.by_default_precedence.served_in_dispatch" \
        not in agreement["paths_checked"]


def test_an_undeclared_container_does_not_launder_a_disagreement():
    """The exemption is BY NAME. Any other nesting is still compared."""
    measured = {"served_in_dispatch": 209,
                "some_other_condition": {"served_in_dispatch": 0}}
    result = {"headline": {"served_in_dispatch": 209},
              "served_in_dispatch_measurement": measured}
    with pytest.raises(SystemExit, match=re.escape(
            "some_other_condition.served_in_dispatch=0")):
        dispatch_agreement.dispatch_agreement(result, measured, "cuda")


def test_an_artifact_with_no_dispatch_key_is_refused():
    with pytest.raises(SystemExit, match="publishes none of"):
        dispatch_agreement.dispatch_agreement(
            {"headline": {"denominator": 597}}, {"served_in_dispatch": 0}, "metal")


def test_an_artifact_whose_only_dispatch_total_is_conditional_is_refused():
    """A board that published ONLY a by-default number would have this floor check
    nothing while reporting agreement, which is the failure mode it exists to stop."""
    measured = {"served_in_dispatch": 209,
                "by_default_precedence": {"served_in_dispatch": 0}}
    result = {"served_in_dispatch_measurement": {
        "by_default_precedence": {"served_in_dispatch": 0}}}
    with pytest.raises(SystemExit, match="checked nothing"):
        dispatch_agreement.dispatch_agreement(result, measured, "cuda")


def test_a_measurement_block_with_no_total_is_refused():
    with pytest.raises(SystemExit, match="nothing for the artifact's dispatch"):
        dispatch_agreement.dispatch_agreement(
            {"served_in_dispatch": 0}, {"rows": 0}, "metal")


def test_a_zero_dispatch_total_still_has_to_agree():
    """A backend that genuinely dispatches nothing is not exempt: 0 must be the
    MEASURED 0, not a literal that happens to match today."""
    measured = {"served_in_dispatch": 0, "why_zero": "no product reaches the seam"}
    result = {"headline": {"served_in_dispatch": 0},
              "served_by_a_fused_product_in_dispatch": 0}
    assert dispatch_agreement.dispatch_agreement(result, measured, "metal")["agree"]
    result["served_by_a_fused_product_in_dispatch"] = 3
    with pytest.raises(SystemExit, match="measured 0"):
        dispatch_agreement.dispatch_agreement(result, measured, "metal")


def test_a_block_whose_served_list_does_not_match_its_count_is_refused():
    block = _filled()
    broken = dict(block)
    broken["instances"] = {**block["instances"],
                           "served": block["instances"]["served"][:-1]}
    with pytest.raises(SystemExit, match="counts 5 served instances and lists 4"):
        fusion_taxonomy.headline_agreement({"served_by_predicate": 5}, broken, "metal")
    missing = dict(block)
    missing["instances"] = {k: v for k, v in block["instances"].items() if k != "served"}
    with pytest.raises(SystemExit, match="lists None"):
        fusion_taxonomy.headline_agreement({"served_by_predicate": 5}, missing, "metal")


# --- the standing censuses -----------------------------------------------------------


def _default_census(builder: str, constant: str) -> Path:
    """The string an argument-free run reads, off the builder's source by ast."""
    tree = ast.parse((_HERE / builder).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == constant):
            # RESULTS / (os.environ.get(...) or "<default>")
            names = [n.value for n in ast.walk(node.value)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and not n.value.startswith("MEEP_GPU_")]
            assert len(names) == 1, (builder, constant, names)
            return RESULTS / names[0]
    raise AssertionError(f"{builder} defines no {constant}")


def _rows(census: Path) -> dict:
    def load(path: Path):
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()] if path.is_file() else []
    record = load(census / "examples.jsonl") + load(census / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load(census / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return {f"{r['leg']}:{r['row']}": r for r in record if r.get("measured")}


@pytest.mark.parametrize("builder, constant", [
    ("build_fusion_matrix.py", "CENSUS"),
    ("build_triton_fusion_matrix.py", "CENSUS"),
    ("build_triton_fusion_matrix.py", "METAL_CENSUS"),
    ("build_cuda_fusion_matrix.py", "CENSUS"),
])
def test_the_argument_free_default_census_is_a_cut_record_on_the_194_row_basis(
        builder, constant):
    census = _default_census(builder, constant)
    if not (census / "examples.jsonl").is_file():
        pytest.skip(f"{census.name} is not in this checkout")
    rows = _rows(census)
    assert len(rows) == 194, (census.name, len(rows))
    assert OFF_AXIS in rows


@pytest.mark.parametrize("builder, constant", [
    ("build_fusion_matrix.py", "CENSUS"),
    ("build_triton_fusion_matrix.py", "METAL_CENSUS"),
])
def test_the_standing_metal_census_carries_the_m0_complex_arm_on_the_off_axis_row(
        builder, constant):
    """What separates the standing Metal record from the one the default named until
    2026-09-06: the m = 0 complex cylindrical arm, selected at all four slots on the
    one row the two records disagree on. A default repointed at a record that predates
    the arm fails here rather than pricing two served instances as a missing half."""
    census = _default_census(builder, constant)
    if not (census / "examples.jsonl").is_file():
        pytest.skip(f"{census.name} is not in this checkout")
    selected = (_rows(census)[OFF_AXIS].get("plan_step") or {}).get("selected") or {}
    assert {selected.get(slot) for slot in ("step_B", "update_H", "step_D", "update_E")} \
        == {"cylindrical complex"}, selected


def test_the_standing_triton_census_admits_the_off_axis_row_to_the_cylindrical_complex_arms():
    census = _default_census("build_triton_fusion_matrix.py", "CENSUS")
    if not (census / "examples.jsonl").is_file():
        pytest.skip(f"{census.name} is not in this checkout")
    predicates = _rows(census)[OFF_AXIS]["predicates"]
    for key in ("cylindrical_complex_constitutive@update_E",
                "cylindrical_complex_constitutive@update_H",
                "cylindrical_complex_curl@step_B", "cylindrical_complex_curl@step_D"):
        assert predicates[key]["covered_modulo_backend"] is True, key


def test_the_standing_cuda_census_is_the_one_this_tree_hashes_to():
    """The CUDA board's own subject pin: a default census whose recorded manifest is not
    this tree's is refused by name by `subject_pin`, so the default must be the record
    cut on these bytes -- checked here with the board's own function."""
    census = _default_census("build_cuda_fusion_matrix.py", "CENSUS")
    if not (census / "examples.jsonl").is_file():
        pytest.skip(f"{census.name} is not in this checkout")
    import build_cuda_fusion_matrix as board  # noqa: PLC0415 - light import
    assert board.CENSUS == census
    record = list(_rows(census).values())
    pin = board.subject_pin(record)
    assert pin["agrees_with_census"] is True
    assert {r["subject_manifest_sha256"] for r in record} == {pin["manifest_sha256"]}
