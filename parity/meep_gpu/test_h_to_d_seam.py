"""The fourth priced seam, update_H -> step_D: the probe's measurement, the shared
pricing rule, and the taxonomy floors that make the new denominator refusable.

What each test pins, and what would break it:

- ``driver_seam_fact`` reads the driver's own text: the only statements between the
  update_H consult and the step_D consult inside ``FdtdDriver.step`` are the electric
  withdraw loop. A pass added there would be priced into no bucket; the locator
  raises instead. The null control plants a pass and requires the refusal.
- ``probe_h_to_d_seam.evaluate`` MEASURES the withdraw on a lifted row rather than
  reading a name: an integrated electric source with a deposit point puts the
  withdraw in the seam; the same source non-integrated, or integrated but magnetic,
  does not. The update_H null verdict is measured by poisoning B and watching H.
- ``classify`` applies the taxonomy's precedence (not_fusion_surface > served >
  withdraw_seam > missing_half > buildable_not_built — read off
  ``fusion_taxonomy.BUCKETS`` here rather than spelled) and names the missing kernel
  and the integrated sources the floors require.
- ``served`` is the BOARD'S verdict carried, never re-derived: with no product
  spanning the seam a served row is refused, with one every row is owed a verdict and
  the refusal NAMES the rows that lack one, a served row whose seam holds the withdraw
  must name what licenses it, and a served row whose update_H launches nothing is a
  contradiction rather than a credit. The synthetic fixture drives the branch that no
  board can drive today, because no product spans the seam.
- ``credit_withheld`` is the THIRD per-row answer (2026-09-10): the board's reason
  sentence beside ``served_by``, for a row the predicate admits whose product's release
  binding is stale. Decided at the served rung, filed ``buildable_not_built`` under the
  one withheld sentence the Triton three-seam walk files, out of ``served`` /
  ``served_rows`` / ``served_by_product``, and pinned to move an instance from served
  to buildable_not_built and NOTHING else. It still owes every refusal a served row
  owes, and it is refused by name when it names no product or is a flag rather than a
  sentence; None or absent is credited, which is what keeps the rule a no-op while the
  Triton table is empty.
- ``Taxonomy`` refuses the pre-2026-09-04 three-seam denominator, requires one H_to_D
  instance per row, subtracts withdraw_seam from ATTAINABLE, and refuses a
  withdraw_seam instance filed off the H_to_D seam or without its source count.
- Where the probe artifact is in the tree, every row of the three standing censuses'
  basis is measured in it (the join the boards refuse without).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fusion_taxonomy  # noqa: E402
import h_to_d_seam  # noqa: E402
import probe_h_to_d_seam as probe  # noqa: E402

RESULTS = _HERE / "results"


# --- the driver fact -----------------------------------------------------------------


def test_the_driver_runs_only_the_electric_withdraw_between_update_H_and_step_D():
    fact = h_to_d_seam.driver_seam_fact()
    statements = tuple(item["statement"] for item in fact["between"])
    assert statements == h_to_d_seam.EXPECTED_BETWEEN, statements
    # Inside `step`, and the consults in the driver's order.
    h_line = int(fact["update_H_consult"].split(":")[1])
    d_line = int(fact["step_D_consult"].split(":")[1])
    assert fact["step_span"][0] < h_line < d_line < fact["step_span"][1]


def test_a_pass_planted_in_the_seam_is_refused_by_name(tmp_path):
    """The null control: the locator must be able to fail."""
    source = h_to_d_seam.DRIVER.read_text(encoding="utf-8")
    needle = '            getattr(source, "withdraw", _no_withdraw)(self.fields)\n        if fast is None or not fast.dispatch("step_D", self.fields):'
    assert source.count(needle) == 1, "the seam's text moved; re-anchor this control"
    planted = source.replace(
        needle,
        '            getattr(source, "withdraw", _no_withdraw)(self.fields)\n'
        '        zero_metal_D(self.fields)\n'
        '        if fast is None or not fast.dispatch("step_D", self.fields):')
    copy = tmp_path / "driver.py"
    copy.write_text(planted, encoding="utf-8")
    with pytest.raises(SystemExit, match="zero_metal_D"):
        h_to_d_seam.driver_seam_fact(copy)


# --- the probe's measurement ---------------------------------------------------------


def _lift(mp, **overrides):
    import meep_gpu  # noqa: PLC0415

    spec = dict(cell_size=mp.Vector3(4, 4, 0), resolution=8,
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.ContinuousSource(frequency=1.0, is_integrated=True),
                                   component=mp.Ez, center=mp.Vector3(0, -0.5, 0))],
                k_point=mp.Vector3(0, 0, 0))
    spec.update(overrides)
    sim = mp.Simulation(**spec)
    return meep_gpu.lift_simulation(sim, prefer_gpu=False)


@pytest.fixture(scope="module")
def mp():
    return pytest.importorskip("meep", reason="CPU MEEP is not installed")


def test_an_integrated_electric_source_puts_the_withdraw_in_the_seam(mp):
    driver = _lift(mp)
    try:
        block = probe.evaluate(driver, None)["h_to_d_seam"]
    finally:
        driver.close()
    assert block["n_electric_sources"] == 1
    assert block["n_integrated_electric_sources"] == 1
    assert block["n_electric_withdraws_that_do_work"] == 1
    assert block["withdraw_in_seam"] is True
    (source,) = block["sources"]
    assert source["is_integrated"] and not source["withdraw_is_the_no_op"]
    assert source["n_source_points"] >= 1
    # PML is active, so update_H launches: read off the driver AND measured.
    assert block["update_H_array_path"]["pml_is_active"] is True
    assert block["update_H_array_path"]["returns_before_its_first_statement"] is False
    poison = block["update_H_array_path"]["poison_measurement"]
    assert poison["measured"] and poison["changed"], poison


def test_the_same_source_non_integrated_leaves_the_seam_empty(mp):
    """The null control for the withdraw: the name is the same, the flag differs."""
    driver = _lift(mp, sources=[mp.Source(mp.ContinuousSource(frequency=1.0),
                                          component=mp.Ez, center=mp.Vector3(0, -0.5, 0))])
    try:
        block = probe.evaluate(driver, None)["h_to_d_seam"]
    finally:
        driver.close()
    assert block["n_electric_sources"] == 1
    assert block["n_integrated_electric_sources"] == 0
    assert block["withdraw_in_seam"] is False


def test_an_integrated_magnetic_source_is_not_this_seams_withdraw(mp):
    """The magnetic withdraw runs before step_B; it is B->H's business, not H->D's."""
    driver = _lift(mp, sources=[mp.Source(mp.ContinuousSource(frequency=1.0, is_integrated=True),
                                          component=mp.Hx, center=mp.Vector3(0, -0.5, 0))])
    try:
        block = probe.evaluate(driver, None)["h_to_d_seam"]
    finally:
        driver.close()
    assert block["n_magnetic_sources"] == 1
    assert block["n_integrated_magnetic_sources"] == 1
    assert block["n_electric_sources"] == 0
    assert block["withdraw_in_seam"] is False


def test_without_a_pml_update_H_launches_nothing_and_the_poison_says_so(mp):
    driver = _lift(mp, boundary_layers=[])
    try:
        block = probe.evaluate(driver, None)["h_to_d_seam"]
    finally:
        driver.close()
    path = block["update_H_array_path"]
    assert path["pml_is_active"] is False
    assert path["returns_before_its_first_statement"] is True
    assert path["poison_measurement"]["measured"]
    assert path["poison_measurement"]["changed"] is False


# --- the rule ------------------------------------------------------------------------


def _block(withdraws: int = 0, electric: int = 1) -> dict:
    sources = [{"type": "VolumeSource", "component": "Ez", "field_type": "D",
                "electric": True, "envelope": "ContinuousEnvelope",
                "is_integrated": i < withdraws, "withdraw_is_the_no_op": False,
                "n_source_points": 1, "withdraw_does_work": i < withdraws}
               for i in range(electric)]
    return {"withdraw_in_seam": withdraws > 0, "n_electric_sources": electric,
            "n_integrated_electric_sources": withdraws, "sources": sources,
            "update_H_array_path": {"returns_before_its_first_statement": False,
                                    "poison_measurement": {"measured": True,
                                                           "changed": True}}}


def test_classify_walks_the_precedence_and_names_what_the_floors_require():
    null = h_to_d_seam.classify("r", "no-PML null", "PML", True, _block(withdraws=1),
                                "metal", "metal_kernels")
    assert null[0] == "not_fusion_surface"          # ahead of the withdraw
    withdraw = h_to_d_seam.classify("r", None, "PML", False,
                                    _block(withdraws=2, electric=2),
                                    "metal", "metal_kernels")
    assert withdraw[0] == "withdraw_seam"           # ahead of the missing half
    assert withdraw[2]["integrated_electric_sources"] == 2
    missing = h_to_d_seam.classify("r", None, "PML", False, _block(), "triton",
                                   "triton_kernels")
    assert missing[0] == "missing_half"
    assert missing[2]["missing_slot"] == "update_H"
    assert "triton_kernels arm on update_H" in missing[2]["missing_kernel"]
    missing_d = h_to_d_seam.classify("r", "ordinary", None, False, _block(), "cuda",
                                     "cuda_kernels")
    assert missing_d[2]["missing_slot"] == "step_D"
    built = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(), "cuda",
                                 "cuda_kernels")
    assert built[0] == "buildable_not_built"
    assert built[2]["cell"] == ["ordinary", "PML"]


def test_a_product_spanning_the_seam_stops_the_walk_until_the_board_files_its_rows():
    taxonomy = fusion_taxonomy.Taxonomy("metal", 3, 1, "n/a",
                                        {"B_to_H": 1, "D_to_E": 1, "H_to_D": 1})
    with pytest.raises(SystemExit, match="span update_H\\+step_D") as raised:
        h_to_d_seam.price(taxonomy, "metal", "metal_kernels", [],
                          products_spanning_the_seam=["some_fused_h_to_d_pair"])
    assert "filed no H->D instances at all" in str(raised.value)


# --- the served branch, driven on a synthetic product ---------------------------------
#
# NO PRODUCT SPANS update_H+step_D IN THE TREE, so no board can drive this branch and
# a rule nothing drives is a rule nothing has shown can fire. The fixture below is the
# smallest thing that stands in for one: three rows, a named spanning product, and the
# board's own per-row verdict. It lives here and not under ``results/`` — a synthetic
# verdict is not a measurement and must never be able to be mistaken for a cut.

PRODUCT = "synthetic_fused_hd_pair"
LICENCE = ("gate_synthetic_hd_pair.py leg `withdraw`: complete driver steps, uint32 "
           "compare, the un-hoisted null control divergent at step 2")


def _synthetic_probe() -> dict:
    """Three rows: one clean, one carrying the withdraw, one the product refuses."""
    return {"leg:clean": {**_block(), "seam": "H_to_D"},
            "leg:withdraw": {**_block(withdraws=1), "seam": "H_to_D"},
            "leg:refused": {**_block(), "seam": "H_to_D"}}


def _entries(**overrides) -> list:
    rows = [{"row": "leg:clean", "update_H": "ordinary", "step_D": "PML",
             "update_H_is_null": False, "served_by": PRODUCT},
            {"row": "leg:withdraw", "update_H": "ordinary", "step_D": "PML",
             "update_H_is_null": False, "served_by": PRODUCT,
             "served_over_the_withdraw": LICENCE},
            {"row": "leg:refused", "update_H": "ordinary", "step_D": "PML",
             "update_H_is_null": False, "served_by": None}]
    for row, changes in overrides.items():
        entry = next(e for e in rows if e["row"].endswith(row))
        entry.update(changes)
        for key, value in list(changes.items()):
            if value is ...:
                entry.pop(key)
    return rows


def _priced(monkeypatch, entries, spanning=(PRODUCT,), probe=None):
    blocks = _synthetic_probe() if probe is None else probe
    monkeypatch.setattr(h_to_d_seam, "probe_rows", lambda *a, **k: blocks)
    monkeypatch.setattr(h_to_d_seam, "probe_provenance",
                        lambda *a, **k: {"probe": "synthetic fixture, not a cut"})
    taxonomy = fusion_taxonomy.Taxonomy("metal", 9, 3, "measured, none",
                                        {"B_to_H": 3, "D_to_E": 3, "H_to_D": 3})
    for row in ("leg:clean", "leg:withdraw", "leg:refused"):
        taxonomy.add("served", "B_to_H", row, "served")
        taxonomy.add("served", "D_to_E", row, "served")
    block = h_to_d_seam.price(taxonomy, "metal", "metal_kernels", entries,
                              products_spanning_the_seam=list(spanning))
    return taxonomy, block


def test_served_sits_second_in_the_precedence_the_taxonomy_declares():
    # DERIVED, not spelled: the branch order in classify must be the bucket order the
    # taxonomy publishes, or one instance can land in two boards' worth of buckets.
    assert fusion_taxonomy.BUCKETS.index("served") == 1
    assert fusion_taxonomy.BUCKETS[0] == "not_fusion_surface"
    assert (fusion_taxonomy.BUCKETS.index("served")
            < fusion_taxonomy.BUCKETS.index("withdraw_seam")
            < fusion_taxonomy.BUCKETS.index("missing_half"))

    served = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(), "metal",
                                  "metal_kernels", served_by=PRODUCT)
    assert served[0] == "served"
    assert f"SERVED by {PRODUCT}" in served[1]
    assert served[2]["product"] == PRODUCT and served[2]["cell"] == ["ordinary", "PML"]
    assert served[2]["withdraw_in_seam"] is False

    # Ahead of the withdraw — and the withdraw's own measurement survives the move.
    over = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(withdraws=1),
                                "metal", "metal_kernels", served_by=PRODUCT,
                                served_over_the_withdraw=LICENCE)
    assert over[0] == "served"
    assert over[2]["withdraw_in_seam"] is True
    assert over[2]["integrated_electric_sources"] == 1
    assert LICENCE in over[1]

    # Behind the null constitutive: nothing to serve where update_H launches nothing.
    null = h_to_d_seam.classify("r", "no-PML null", "PML", True, _block(), "metal",
                                "metal_kernels", served_by=PRODUCT)
    assert null[0] == "not_fusion_surface"


def test_price_accepts_a_board_that_files_the_rows_its_product_admits(monkeypatch):
    taxonomy, block = _priced(monkeypatch, _entries())
    assert block["served"] == 2
    assert block["served_rows"] == ["leg:clean", "leg:withdraw"]
    assert block["served_by_product"] == {PRODUCT: ["leg:clean", "leg:withdraw"]}
    assert block["served_rows_carrying_the_withdraw"] == ["leg:withdraw"]
    assert block["products_spanning_the_seam"] == [PRODUCT]
    assert block["buckets"] == {"served": 2, "buildable_not_built": 1}
    assert PRODUCT in block["served_note"]
    # The floors accept it, and the served count the board hands in is the one the
    # taxonomy files — which is the whole point of routing it through the same add().
    finished = taxonomy.finish(served_expected=6 + block["served"],
                               served_expected_source="test")
    assert finished["per_seam"]["H_to_D"] == {"served": 2, "buildable_not_built": 1}
    assert finished["buckets"]["served"] == 8
    # ATTAINABLE no longer subtracts the withdraw row: it is served, and that is
    # exactly why the licence above is required.
    assert finished["buckets"]["withdraw_seam"] == 0
    assert finished["tiers"]["ATTAINABLE"]["value"] == 9


def test_answering_none_on_every_row_is_a_measurement_not_a_refusal(monkeypatch):
    entries = _entries(
        clean={"served_by": None},
        withdraw={"served_by": None, "served_over_the_withdraw": ...})
    _taxo, block = _priced(monkeypatch, entries)
    assert block["served"] == 0
    assert block["buckets"] == {"buildable_not_built": 2, "withdraw_seam": 1}


def test_an_unanswered_row_is_named_as_owed(monkeypatch):
    entries = _entries(refused={"served_by": ...})
    with pytest.raises(SystemExit, match="1 row\\(s\\) are OWED one") as raised:
        _priced(monkeypatch, entries)
    assert "leg:refused" in str(raised.value)
    assert "served_by" in str(raised.value)


def test_a_served_row_with_no_product_behind_it_is_refused(monkeypatch):
    with pytest.raises(SystemExit, match="names NO product spanning"):
        _priced(monkeypatch, _entries(), spanning=())


def test_a_served_row_naming_a_product_the_board_does_not_list_is_refused(monkeypatch):
    entries = _entries(clean={"served_by": "some_other_pair"})
    with pytest.raises(SystemExit, match="does not list as spanning"):
        _priced(monkeypatch, entries)


def test_a_served_withdraw_row_without_its_licence_is_refused(monkeypatch):
    entries = _entries(withdraw={"served_over_the_withdraw": None})
    with pytest.raises(SystemExit, match="served_over_the_withdraw") as raised:
        _priced(monkeypatch, entries)
    assert "leg:withdraw" in str(raised.value)


def test_a_served_row_whose_update_H_launches_nothing_is_a_contradiction(monkeypatch):
    entries = _entries(clean={"update_H": "no-PML null", "update_H_is_null": True})
    with pytest.raises(SystemExit, match="filed served .* and the precedence files it"):
        _priced(monkeypatch, entries)


# --- the third answer: a credit withheld, on the same synthetic product ----------------
#
# THE TRITON TABLE (CREDIT_WITHHELD_STALE_BINDING) IS EMPTY, so no board can drive this
# branch either. What it stands in for: a product whose predicate ADMITS the row and
# whose release binding is stale, so the board will not CREDIT it. That verdict has to
# be a third per-row answer beside served_by = name / None — `served_by: None` would say
# the predicate refused, which is a different and false claim — and it has to file under
# the ONE withheld sentence the three-seam walk files, or the bucket grows a second
# spelling. REASON is what the Triton table's own sentence looks like: the gate, the
# file that moved, and its two digests.

REASON = ("gate_synthetic_hd_pair.json: kernels/synthetic.py 726a52f4 -> f7a8c561 "
          "(moved under the cut)")


def test_classify_decides_a_withheld_credit_at_the_served_rung_and_files_it_eighth():
    # DECIDED SECOND — where `served` sits — and FILED LAST: the bucket is the
    # taxonomy's eighth, read off BUCKETS rather than spelled.
    assert fusion_taxonomy.BUCKETS[-1] == "buildable_not_built"
    withheld = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(), "triton",
                                    "triton_kernels", served_by=PRODUCT,
                                    credit_withheld=REASON)
    assert withheld[0] == "buildable_not_built"
    assert withheld[1] == fusion_taxonomy.credit_withheld_sub_reason(PRODUCT)
    assert withheld[2]["product"] == PRODUCT              # served_by is NOT rewritten
    assert withheld[2]["cell"] == ["ordinary", "PML"]
    assert withheld[2]["credit_withheld"] is True
    assert withheld[2]["credit_withheld_reason"] == REASON
    assert withheld[2]["predicate_admits"] is True
    # The same call without the reason is the served verdict: one rung, two answers.
    credited = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(), "triton",
                                    "triton_kernels", served_by=PRODUCT)
    assert credited[0] == "served"
    # Behind the null constitutive, credited or withheld alike: where update_H
    # launches nothing there is nothing to serve and nothing to withhold.
    null = h_to_d_seam.classify("r", "no-PML null", "PML", True, _block(), "triton",
                                "triton_kernels", served_by=PRODUCT,
                                credit_withheld=REASON)
    assert null[0] == "not_fusion_surface"


def test_a_withheld_credit_is_filed_buildable_not_built_under_the_withheld_sentence_and_out_of_served(monkeypatch):
    # One served (leg:withdraw, licensed), one refused (leg:refused, None), one
    # withheld (leg:clean): the predicate admits two, the board credits one.
    entries = _entries(clean={"credit_withheld": REASON})
    taxonomy, block = _priced(monkeypatch, entries)
    assert block["served"] == 1
    assert block["served_rows"] == ["leg:withdraw"]
    assert block["served_by_product"] == {PRODUCT: ["leg:withdraw"]}      # who is served
    assert block["credit_withheld_rows"] == ["leg:clean"]                  # who is withheld
    assert block["credit_withheld_by_product"] == {PRODUCT: ["leg:clean"]}
    assert block["predicate_admits_h_to_d"] == 2
    assert block["buckets"] == {"served": 1, "buildable_not_built": 2}
    withheld = next(i for i in block["instances"] if i["row"] == "leg:clean")
    assert withheld["bucket"] == "buildable_not_built"
    assert withheld["served_by"] == PRODUCT and withheld["credit_withheld"] == REASON
    assert "CREDIT WITHHELD" in block["served_note"] and PRODUCT in block["served_note"]
    finished = taxonomy.finish(served_expected=6 + block["served"],
                               served_expected_source="test")
    assert finished["per_seam"]["H_to_D"] == {"served": 1, "buildable_not_built": 2}
    assert finished["buckets"]["served"] == 7
    # The SAME sentence the three-seam walk files, so the bucket has one withheld key.
    sentence = fusion_taxonomy.credit_withheld_sub_reason(PRODUCT)
    assert finished["sub_reasons"]["buildable_not_built"][sentence] == 1
    assert finished["served_by_product"][PRODUCT] == [("H_to_D", "leg:withdraw")]
    assert all(("H_to_D", "leg:clean") not in rows
               for rows in finished["served_by_product"].values())
    assert ("H_to_D", "leg:clean") in finished["instances"]["buildable_not_built"]
    # Withholding did not move the denominator.
    assert finished["tiers"]["ATTAINABLE"]["value"] == 9


def test_withholding_moves_served_to_buildable_not_built_and_nothing_else(monkeypatch):
    t0, b0 = _priced(monkeypatch, _entries())
    f0 = t0.finish(served_expected=6 + b0["served"], served_expected_source="test")
    t1, b1 = _priced(monkeypatch, _entries(clean={"credit_withheld": REASON}))
    f1 = t1.finish(served_expected=6 + b1["served"], served_expected_source="test")
    delta = {k: f1["buckets"][k] - f0["buckets"][k]
             for k in f0["buckets"] if f1["buckets"][k] != f0["buckets"][k]}
    assert delta == {"served": -1, "buildable_not_built": 1}
    for tier in ("PRICED", "FUSABLE", "ATTAINABLE"):
        assert f1["tiers"][tier]["value"] == f0["tiers"][tier]["value"], tier
    assert f1["tiers"]["TO_DO"]["value"] == f0["tiers"]["TO_DO"]["value"] + 1
    assert b1["instances_priced"] == b0["instances_priced"] == 3
    assert b1["one_per_row"] is True
    assert b1["withdraw_rows"] == b0["withdraw_rows"]


def test_a_withheld_credit_on_a_withdraw_row_keeps_its_licence_and_its_measurement(monkeypatch):
    # Filed buildable_not_built, NOT withdraw_seam: its attainability was measured by
    # the product's own licensed gate; the binding is what is stale. So ATTAINABLE
    # does not re-subtract it, and the withdraw facts survive in the detail.
    taxonomy, block = _priced(monkeypatch,
                              _entries(withdraw={"credit_withheld": REASON}))
    assert block["buckets"] == {"served": 1, "buildable_not_built": 2}
    assert block["served_rows_carrying_the_withdraw"] == []
    assert block["credit_withheld_rows"] == ["leg:withdraw"]
    assert block["withdraw_rows"] == ["leg:withdraw"]
    finished = taxonomy.finish(served_expected=6 + block["served"],
                               served_expected_source="test")
    assert finished["buckets"]["withdraw_seam"] == 0
    assert finished["tiers"]["ATTAINABLE"]["value"] == 9
    detail = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(withdraws=1),
                                  "metal", "metal_kernels", served_by=PRODUCT,
                                  served_over_the_withdraw=LICENCE,
                                  credit_withheld=REASON)
    assert detail[0] == "buildable_not_built"
    assert detail[1] == fusion_taxonomy.credit_withheld_sub_reason(PRODUCT)
    assert detail[2]["withdraw_in_seam"] is True
    assert detail[2]["integrated_electric_sources"] == 1
    assert detail[2]["served_over_the_withdraw"] == LICENCE
    assert detail[2]["credit_withheld"] is True
    assert detail[2]["credit_withheld_reason"] == REASON
    # The sentence is identical on a withdraw row: the withdraw facts live in the
    # detail, never in the sentence, so the bucket carries one withheld key.
    clean = h_to_d_seam.classify("r", "ordinary", "PML", False, _block(), "metal",
                                 "metal_kernels", served_by=PRODUCT,
                                 credit_withheld=REASON)
    assert clean[1] == detail[1]
    # AND THE LICENCE IS STILL OWED. The refusal stays keyed on `served_by`, and
    # withholding does not bypass it: the row becomes served the moment the table
    # entry is deleted, so better it fails now than then.
    with pytest.raises(SystemExit, match="served_over_the_withdraw") as raised:
        _priced(monkeypatch, _entries(withdraw={"credit_withheld": REASON,
                                                "served_over_the_withdraw": None}))
    assert "leg:withdraw" in str(raised.value)


def test_a_withheld_credit_with_no_product_behind_it_is_refused(monkeypatch):
    # Credit is withheld FROM a product whose predicate admits the row. A withheld
    # verdict naming no product is a served_by=None row wearing a second label.
    entries = _entries(refused={"credit_withheld": REASON})
    with pytest.raises(SystemExit, match="no `served_by`") as raised:
        _priced(monkeypatch, entries)
    assert "leg:refused" in str(raised.value)


def test_a_withheld_credit_must_be_a_reason_not_a_flag(monkeypatch):
    # Keyed on PRESENCE: only None / absent means credited. False and the empty
    # sentence are refused too — the empty sentence is exactly what a table entry
    # with no text behind it would hand over, and it must not read as credited.
    for value in (True, "", False, "   ", 0):
        entries = _entries(clean={"credit_withheld": value})
        with pytest.raises(SystemExit, match="REASON") as raised:
            _priced(monkeypatch, entries)
        assert "leg:clean" in str(raised.value), value
        assert repr(value) in str(raised.value), value


def test_a_withheld_row_whose_update_H_launches_nothing_is_a_contradiction(monkeypatch):
    # Held to the same contradiction as a served row: served_by on a null-update_H
    # row is a claim about a seam with one half, credited or not.
    entries = _entries(clean={"credit_withheld": REASON, "update_H": "no-PML null",
                              "update_H_is_null": True})
    with pytest.raises(SystemExit,
                       match="withheld from .* and the precedence files it"):
        _priced(monkeypatch, entries)


def test_a_none_or_absent_credit_withheld_is_credited(monkeypatch):
    # THE NO-MOVE PROOF IN MINIATURE: an explicit None and an absent key price
    # identically, and identically to the credited run on every pre-existing key —
    # which is what keeps the rule a no-op on every board while the table is empty.
    _t0, absent = _priced(monkeypatch, _entries())
    _t1, explicit = _priced(monkeypatch, _entries(clean={"credit_withheld": None}))
    assert explicit == absent
    assert absent["served"] == 2
    assert absent["served_rows"] == ["leg:clean", "leg:withdraw"]
    assert absent["served_by_product"] == {PRODUCT: ["leg:clean", "leg:withdraw"]}
    assert absent["buckets"] == {"served": 2, "buildable_not_built": 1}
    assert absent["credit_withheld_rows"] == []
    assert absent["credit_withheld_by_product"] == {PRODUCT: []}
    assert absent["predicate_admits_h_to_d"] == 2
    assert all(i["credit_withheld"] is None for i in absent["instances"])
    assert "CREDIT WITHHELD" not in absent["served_note"]
    assert explicit["served_note"] == absent["served_note"]


# --- the taxonomy floors -------------------------------------------------------------


def _taxonomy(rows: int = 2, e_to_p: int = 0) -> fusion_taxonomy.Taxonomy:
    return fusion_taxonomy.Taxonomy(
        "metal", 3 * rows + e_to_p, rows, "measured, none",
        {"B_to_H": rows, "D_to_E": rows, "H_to_D": rows, "E_to_P": e_to_p})


def test_the_three_seam_denominator_is_refused_by_name():
    with pytest.raises(SystemExit, match="H_to_D"):
        fusion_taxonomy.Taxonomy("metal", 4, 2, "n/a", {"B_to_H": 2, "D_to_E": 2})
    with pytest.raises(SystemExit, match="exactly one\\s+instance per row"):
        fusion_taxonomy.Taxonomy("metal", 5, 2, "n/a",
                                 {"B_to_H": 2, "D_to_E": 2, "H_to_D": 1})
    with pytest.raises(SystemExit, match="imply 6"):
        fusion_taxonomy.Taxonomy("metal", 7, 2, "n/a",
                                 {"B_to_H": 2, "D_to_E": 2, "H_to_D": 2})


def test_the_justification_derives_the_fourth_seam_from_the_row_count():
    text = fusion_taxonomy.denominator_justification(194, 15)
    assert "194 rows x 1 constitutive->curl seam (H->D" in text
    assert "= 597 seam-instances" in text


def _fill(taxonomy: fusion_taxonomy.Taxonomy, h_to_d: list) -> None:
    for row in ("a", "b"):
        taxonomy.add("served", "B_to_H", row, "served")
        taxonomy.add("served", "D_to_E", row, "served")
    for row, bucket, detail in h_to_d:
        taxonomy.add(bucket, "H_to_D", row, f"{bucket} on {row}", detail)


def test_withdraw_seam_is_subtracted_from_attainable_and_counted_once():
    taxonomy = _taxonomy()
    _fill(taxonomy, [("a", "withdraw_seam", {"integrated_electric_sources": 1}),
                     ("b", "buildable_not_built", {})])
    block = taxonomy.finish(served_expected=4, served_expected_source="test")
    tiers = {k: v["value"] for k, v in block["tiers"].items()}
    assert tiers == {"PRICED": 6, "FUSABLE": 6, "ATTAINABLE": 5, "COVERAGE": 4,
                     "TO_DO": 1}
    assert block["per_seam"]["H_to_D"] == {"withdraw_seam": 1, "buildable_not_built": 1}
    assert block["instances"]["withdraw_seam"] == [("H_to_D", "a")]


def test_a_null_update_H_outranks_the_withdraw_and_leaves_fusable():
    taxonomy = _taxonomy()
    _fill(taxonomy, [("a", "not_fusion_surface", {}),
                     ("b", "withdraw_seam", {"integrated_electric_sources": 1})])
    block = taxonomy.finish(served_expected=4, served_expected_source="test")
    tiers = {k: v["value"] for k, v in block["tiers"].items()}
    assert tiers["FUSABLE"] == 5 and tiers["ATTAINABLE"] == 4


def test_a_withdraw_instance_off_the_seam_or_without_its_sources_is_refused():
    taxonomy = _taxonomy()
    taxonomy.add("withdraw_seam", "B_to_H", "a", "misfiled", {"integrated_electric_sources": 1})
    taxonomy.add("served", "D_to_E", "a", "served")
    taxonomy.add("served", "B_to_H", "b", "served")
    taxonomy.add("served", "D_to_E", "b", "served")
    taxonomy.add("buildable_not_built", "H_to_D", "a", "x", {})
    taxonomy.add("buildable_not_built", "H_to_D", "b", "x", {})
    with pytest.raises(SystemExit, match="filed at \\['B_to_H'\\]"):
        taxonomy.finish(served_expected=3, served_expected_source="test")

    taxonomy = _taxonomy()
    _fill(taxonomy, [("a", "withdraw_seam", {}),
                     ("b", "buildable_not_built", {})])
    with pytest.raises(SystemExit, match="name no integrated electric source"):
        taxonomy.finish(served_expected=4, served_expected_source="test")


# --- the artifact --------------------------------------------------------------------


@pytest.mark.skipif(not (h_to_d_seam.PROBE_DIR / "examples.jsonl").is_file(),
                    reason=f"{h_to_d_seam.PROBE_DIR.name} is not in this tree")
def test_the_probe_measured_every_row_of_the_standing_basis():
    measured = h_to_d_seam.probe_rows()
    for backend, census in probe.STANDING_CENSUSES.items():
        directory = RESULTS / census
        if not (directory / "examples.jsonl").is_file():
            pytest.skip(f"{census} is not in this tree")
        labels = [probe.label(r) for r in probe.rows(directory)]
        assert labels, backend
        joined = h_to_d_seam.join(labels, measured)
        assert len(joined) == len(labels)
        # Every block carries the seam fact the boards file from.
        for block in joined.values():
            assert block["seam"] == "H_to_D"
            assert isinstance(block["withdraw_in_seam"], bool)
    summary = json.loads((h_to_d_seam.PROBE_DIR / "summary.json").read_text())
    assert summary["basis_missing_from_the_probe"] == []
