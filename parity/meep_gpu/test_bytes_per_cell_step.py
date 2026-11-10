"""The compulsory-traffic counter (bytes_per_cell_step.py): a hand count, a degenerate case, the
fusion saving, the cross-table agreement and the refusals. No device and no MEEP: the kernel texts
come from the package's emitters, the facts are the pml_3d lift's chunk record written inline."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # the harness modules sit beside this file
    sys.path.insert(0, HERE)

import bytes_per_cell_step as bpc  # noqa: E402

#: The 7,077,888-cell pml_3d lift (192^3, 27 chunks): chunk_map.chunk_bounds_record as recorded in
#: a ``--lift-facts`` file. Integer lines keep rows [1,40) U [154,192) (77 of 192), half-integer
#: lines [0,39) U [153,192) (78 of 192), on every axis.
FACTS_7M = {"shape": [192, 192, 192], "cells": 192 ** 3,
            "record": [1, 40, 154, 192] * 3 + [0, 39, 153, 192] * 3 + [7],
            "offdiag": {"Ex": {"Ey": {}, "Ez": {}}, "Ey": {"Ex": {}, "Ez": {}}, "Ez": {"Ex": {}, "Ey": {}}}}


@pytest.fixture(scope="module")
def tables():
    return bpc.plan_table(bpc.kernel_sources(row_mask=bpc.row_mask_of(FACTS_7M)), FACTS_7M)


def test_the_fused_plan_matches_the_hand_count_on_every_table(tables):
    i, h = 77 / 192, 78 / 192
    pair = 3 * 2 + 3 + 3 * 2 * h + 3 * (i + 1) + 3 * 2 * i      # B rw, E, fu_B, H, f_w_H
    curl_d = 3 * 2 + 3 + 3 * 2 * i                               # D rw, H, fu_D
    offdiag = 3 + 3 + 6 + 3 * (h + 1) + 3 * 2 * h                # D, inv_eps, six rows, E, f_w_E
    words = pair + curl_d + offdiag
    assert words == pytest.approx(48.109375)
    for table in ("triton", "cuda", "metal"):
        got = tables[table]["plans"]["fused"]["volume_bytes_per_cell"]
        assert got == pytest.approx(4 * words, abs=1e-9), (table, got)


def test_the_fused_pair_saves_exactly_the_b_round_trip(tables):
    for table in ("triton", "cuda", "metal"):
        plans = tables[table]["plans"]
        saving = plans["unfused"]["volume_bytes_per_cell"] - plans["fused"]["volume_bytes_per_cell"]
        assert saving == pytest.approx(12.0), (table, saving)   # three B words reloaded by update_H


def test_an_auxiliary_reaches_only_its_record_line(tables):
    arrays = tables["cuda"]["kernels"]["pml_curl_D"]["arrays"]
    assert arrays["fu_Dx"]["load_where"] == "int_z" and arrays["fu_Dx"]["store_where"] == "int_z"
    assert arrays["Dx"]["load_where"] == "all" and arrays["Hx"]["store_where"] == "none"


def test_a_pair_is_one_launch_and_an_unread_family_is_refused_by_name():
    slot = lambda family, arm: {"backend": "cuda", "family": family, "arm": arm, "dispatches": 6}  # noqa: E731
    slots = {"step_B": slot("cuda_fused_magnetic_pair", "pair"), "update_H": slot("cuda_fused_magnetic_pair", "pair"),
             "step_D": slot("cuda_curl", "PML"), "update_E": slot("cuda_offdiag", "off-diagonal")}
    assert bpc.plan_from_counters(slots) == ("cuda", ("fused_magnetic_pair", "pml_curl_D", "offdiag_E"))
    slots["update_P"] = slot("cuda_ade", "ADE update_P")
    with pytest.raises(bpc.Refused, match="ADE update_P"):
        bpc.plan_from_counters(slots)
    del slots["update_P"]
    slots["update_E"] = slot("cuda_dispersive_offdiag", "dispersive")
    with pytest.raises(bpc.Refused, match="cuda_dispersive_offdiag"):
        bpc.plan_from_counters(slots)


def test_meep_on_one_lossless_isotropic_chunk_steps_only_the_two_curls():
    division = {"cells": 1000, "shape": [10, 10, 10],
                "chunks": [{"cells": 1000, "box": [[0, 10]] * 3, "pml_axes": [False] * 3,
                            "keep_diag": {"Ex": False, "Ey": False, "Ez": False}, "keep_offdiag": {}}]}
    m = bpc.meep_traffic(division)
    assert m["words_per_cell"] == {"step_db_B": 12.0, "update_eh_H": 0.0, "step_db_D": 12.0, "update_eh_E": 0.0}
    assert m["per_substep_total_words_per_cell"] == pytest.approx(18.0)
