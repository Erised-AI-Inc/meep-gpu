"""MEEP identity, Tier A: the minted cases replay from their capsules, with no MEEP.

The ledger-bound tests (the replay, the planted flip, the multiply watch, the oracle and
file checks) read ``parity/meep_gpu/fixtures/meep_identity/`` and declare the resource
``meep_identity_fixtures`` (``tools/ci/declared_resources.txt``): a checkout carries the
fixtures, an installed wheel does not. They are parametrised by the ledger's own
(mode, case) list. The schema tests below them need nothing but NumPy.

A Tier A failure names the first step, component and x-plane whose digest moved. When
the array path changes on purpose, Tier C re-proves identity against MEEP and
``parity/meep_gpu/record_meep_identity.py`` re-mints the set; nothing else may write it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy
import pytest

from . import meep_identity as identity

_ROOT = Path(__file__).resolve().parent.parent


def _ledger_cases():
    """``(mode, case)`` of every minted case; one placeholder when there is no ledger.

    The placeholder is skipped by the declared resource, so a host without the fixtures
    reports a sanctioned skip rather than an empty parametrisation.
    """
    try:
        ledger = identity.read_ledger(check_files=False)
    except identity.IdentityRefusal:
        return [("keep", "no-fixtures")]
    return [(mode, case) for mode in sorted(ledger["sets"])
            for case in sorted(ledger["sets"][mode]["cases"])]


TIER_A = _ledger_cases()
#: The four probe cases of ``parity/meep_gpu/meep_identity_smoke.py``, minted in every set.
PROBE_CASES = ("a", "b", "c", "c_prime")
TIER_A_IDS = [f"{mode}-{case}" for mode, case in TIER_A]
_REPLAYS = {}


@pytest.fixture(scope="module")
def ledger():
    return identity.read_ledger()


def _require_mode(mode):
    if mode == "flush":
        try:
            with identity.host_subnormal_mode("flush"):
                pass
        except identity.IdentityRefusal as problem:
            from conftest import requires_resource_skip  # noqa: PLC0415 - the root conftest

            requires_resource_skip("host_fpu_mode", f"a flush digest set is judged under a "
                                                    f"flushing FPU, which no lever reaches in "
                                                    f"this process: {problem}")


def _replay(mode, case, ledger, watch=False):
    key = (mode, case, watch)
    if key not in _REPLAYS:
        _require_mode(mode)
        _REPLAYS[key] = identity.replay(case, mode=mode, ledger=ledger, watch=watch)
    return _REPLAYS[key]


# ---------------------------------------------------------------------------------
# Tier A (declared resource: meep_identity_fixtures)
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,case", TIER_A, ids=TIER_A_IDS)
def test_tier_a_replays_every_meep_digest_from_the_capsule(mode, case, ledger):
    result = _replay(mode, case, ledger)
    assert result.stored_equal == result.digests, (
        f"{case} ({mode}): {result.stored_equal} of {result.digests} stored digests equal MEEP's; "
        f"first difference {result.first_difference}")
    assert result.reader_equal == result.digests, (
        f"{case} ({mode}): {result.reader_equal} of {result.digests} get_array digests equal "
        f"MEEP's; first difference {result.first_difference}")
    assert result.fpu["measured"]["mode"] == mode and result.fpu["checks"] == len(result.steps)


@pytest.mark.parametrize("mode,case", TIER_A, ids=TIER_A_IDS)
def test_tier_a_catches_a_planted_flip_and_names_its_plane(mode, case, ledger):
    flip = _replay(mode, case, ledger).planted_flip
    assert flip["caught"], f"{case} ({mode}): a one-ulp flip left the digest unchanged: {flip}"
    assert flip["placed"], (f"{case} ({mode}): the flip at x-plane {flip['word_plane']} moved "
                            f"planes {flip.get('planes_differing')}")


@pytest.mark.parametrize("mode,case", TIER_A, ids=TIER_A_IDS)
def test_tier_a_steps_multiply_no_two_complex_operands(mode, case, ledger):
    result = _replay(mode, case, ledger, watch=True)
    watch = result.multiply_watch
    assert watch["multiplies_observed"] > 0, f"{case}: the watch observed no multiply at all"
    assert watch["complex_times_complex"] == 0, (
        f"{case}: {watch['complex_times_complex']} multiplies of two complex operands, at "
        f"{watch['sites']}; the array path forms complex products part by part")
    assert result.holds, f"{case}: the watched run did not reproduce the digests: {result.first_difference}"


def test_an_arithmetic_change_to_the_curl_turns_tier_a_red_and_is_located(ledger, monkeypatch):
    """The pairwise grouping ``(g1s - g1) + (g2 - g2s)`` in place of MEEP's left-to-right
    ``((g1s - g1) + g2) - g2s``: a different float32 number, so the replay must fail and
    name the step, component and x-planes where the words first moved."""
    from . import stepping  # noqa: PLC0415

    original = stepping._curl_from_operands  # noqa: SLF001

    def regrouped(operands, dtdx, presence, scratch=None):
        if all(presence):
            return dtdx * ((operands.shifted_first - operands.first)
                           + (operands.second - operands.shifted_second))
        return original(operands, dtdx, presence, scratch=scratch)

    monkeypatch.setattr(stepping, "_curl_from_operands", regrouped)
    # The smallest probe case: a 2-D or 3-D Cartesian grid, where every curl has both
    # partners. A family row can be smaller and still never reach the regrouped arm (the
    # dispersive row is a 0-D cell, whose curls have no partner at all).
    # In keep, which every host reaches: a flush set is judged only where a lever
    # reaches flush, and this test is about the arithmetic, not the mode.
    probes = [pair for pair in TIER_A if pair[1] in PROBE_CASES and pair[0] == "keep"] or TIER_A
    mode, case = min(probes, key=lambda pair: ledger["sets"][pair[0]]["cases"][pair[1]]
                     ["files"]["capsule"]["bytes"])
    result = identity.replay(case, mode=mode, ledger=ledger)
    first = result.first_difference
    assert result.stored_equal < result.digests and first is not None, result.as_dict()
    assert first["step"] <= identity.PLANE_STEPS and first["planes_differing"], first
    assert result.planted_flip["caught"]


def test_the_ledger_records_the_live_array_path_oracle(ledger):
    from . import array_path_oracle  # noqa: PLC0415

    live = array_path_oracle.live()
    expected = {"fixture_set": live.fixture_set, "fingerprint_sha256": live.fingerprint_sha256}
    stale = sorted(mode for mode, leaf in ledger["sets"].items()
                   if leaf.get("array_path_oracle") != expected)
    assert not stale, (
        f"the {stale} digest set(s) were minted on another array path (oracle "
        f"{[ledger['sets'][m].get('array_path_oracle') for m in stale]}, live {expected}): re-prove "
        f"identity with Tier C and re-mint with parity/meep_gpu/record_meep_identity.py")


def test_the_fixtures_are_the_files_the_ledger_records_within_budget(ledger):
    assert ledger["fixture_bytes"] <= identity.FIXTURE_BUDGET_BYTES
    for mode, leaf in ledger["sets"].items():
        for case, entry in leaf["cases"].items():
            tier_c = entry["tier_c"]
            assert tier_c["stored"]["word_differing"] == 0 and tier_c["stored"]["value_differing"] == 0
            assert tier_c["reader"]["word_differing"] == 0 and tier_c["planted_flip"]["caught"]
            assert tier_c["steps"] == identity.compared_steps(tier_c["n_steps"]), (mode, case)


# ---------------------------------------------------------------------------------
# The schema (NumPy only)
# ---------------------------------------------------------------------------------

def test_the_schedule_is_every_step_to_20_and_table_2s_checkpoints():
    assert identity.compared_steps(200) == list(range(21)) + [40, 60, 80, 100, 120, 140, 160, 180, 200]
    assert identity.compared_steps(375) == list(range(21)) + [38, 75, 113, 150, 188, 225, 263, 300, 338, 375]
    assert identity.compared_steps(5) == [0, 1, 2, 3, 4, 5]
    assert identity.table2_checkpoints(375)[0] == 20


def _driver(**overrides):
    from .driver import FdtdDriver  # noqa: PLC0415

    settings = dict(cell_size=(1.2, 1.0, 0.0), resolution=10, force_complex_fields=False,
                    dimensions=2)
    settings.update(overrides)
    return FdtdDriver(**settings)


def test_mirror_ghosts_are_excluded_and_walls_are_compared():
    from .grid import Mirror  # noqa: PLC0415

    folded = _driver(symmetry=(Mirror("Y", -1),))
    nx = folded.grid.nx
    ex, ey = identity.cell_classes(folded.grid, "Ex"), identity.cell_classes(folded.grid, "Ey")
    # Shift 0 on the folded axis: stored cell 0 is the ghost. Shift 1 on a folded periodic
    # axis: the far slot past MEEP's big corner is.
    assert ex["ghost"][:, 0, :].all() and int(ex["ghost"].sum()) == nx
    assert ey["ghost"][:, -1, :].all() and int(ey["ghost"].sum()) == nx
    assert not (ex["ghost"] & ex["compared"]).any() and int(ex["wall"].sum()) == 0
    walled = _driver(cell_size=(0.6, 0.5, 0.4), dimensions=3, boundaries="metallic")
    ez = identity.cell_classes(walled.grid, "Ex")
    assert int(ez["ghost"].sum()) == 0 and int(ez["wall"].sum()) > 0
    assert ez["compared"].all(), "a metallic wall is stored by MEEP and is compared"


def test_words_are_the_compared_cells_in_c_order_with_parts_interleaved():
    values = numpy.arange(12, dtype=numpy.float32).reshape(2, 3, 2) + 1
    mask = numpy.ones(values.shape, dtype=bool)
    mask[0, 1, 0] = False
    real = identity.compared_words(values, mask)
    assert real.view(numpy.float32).tolist() == [v for v in values.reshape(-1).tolist() if v != 3.0]
    complex_values = (values + 1j * (values + 100)).astype(numpy.complex64)
    words = identity.compared_words(complex_values, mask).view(numpy.float32)
    assert words[:4].tolist() == [1.0, 101.0, 2.0, 102.0]
    assert words.size == 2 * int(mask.sum())
    with pytest.raises(identity.IdentityRefusal, match="float64"):
        identity.compared_words(values.astype(numpy.float64), mask)
    assert identity.digest_words(real) == hashlib.sha256(real.astype("<u4").tobytes()).hexdigest()


def test_a_flipped_word_moves_its_digest_and_exactly_its_own_plane():
    rng = numpy.random.default_rng(7)
    values = rng.standard_normal((6, 5, 3)).astype(numpy.float32)
    mask = rng.random(values.shape) > 0.2
    recorded = identity.plane_digests(values, mask)
    words = identity.compared_words(values, mask)
    for index in (0, 17, words.size - 1):
        cell, _part = identity.word_cell(mask, 1, index)
        assert identity.compare_planes(recorded, identity.flipped_planes(values, mask, index)) == [cell[0]]
        assert identity.digest_words(identity.flip_word(words, index)) != identity.digest_words(words)
    assert identity.compare_planes(recorded, identity.plane_digests(values, mask)) == []


def _write_case(tmp_path, n_steps=3):
    steps = identity.compared_steps(n_steps)
    entry = {name: "0" * 64 for name in identity.COMPONENTS}
    document = {"schema": identity.DIGESTS_SCHEMA, "version": identity.DIGESTS_VERSION,
                "case": "t", "mode": "keep", "n_steps": n_steps, "steps": steps,
                "components": list(identity.COMPONENTS),
                "stored": {str(s): dict(entry) for s in steps},
                "reader": {str(s): dict(entry) for s in steps},
                "planes": {"file": "planes.npy", "steps": steps, "axis": "x", "digest_bytes": 8}}
    planes = numpy.zeros((len(steps), 12, 4, 8), dtype=numpy.uint8)
    return identity.write_digests(tmp_path, document, planes), document


def test_a_digest_record_reads_back_and_refuses_what_it_does_not_hold(tmp_path):
    _write_case(tmp_path)
    document, planes = identity.read_digests(tmp_path / "digests.json", tmp_path / "planes.npy")
    assert planes.shape == (4, 12, 4, 8) and document["steps"] == [0, 1, 2, 3]
    numpy.save(tmp_path / "planes.npy", numpy.ones((4, 12, 4, 8), dtype=numpy.uint8))
    with pytest.raises(identity.IdentityRefusal, match="sha256"):
        identity.read_digests(tmp_path / "digests.json", tmp_path / "planes.npy")
    _write_case(tmp_path)
    text = json.loads((tmp_path / "digests.json").read_text())
    text["steps"] = [0, 1, 3]
    (tmp_path / "digests.json").write_text(json.dumps(text))
    with pytest.raises(identity.IdentityRefusal, match="schedule"):
        identity.read_digests(tmp_path / "digests.json", tmp_path / "planes.npy")
    text["schema"] = "something.else"
    (tmp_path / "digests.json").write_text(json.dumps(text))
    with pytest.raises(identity.IdentityRefusal, match="reads"):
        identity.read_digests(tmp_path / "digests.json", tmp_path / "planes.npy")


def _write_ledger(tmp_path, *, fixture_bytes=None):
    (tmp_path / "keep" / "t").mkdir(parents=True)
    files, total = {}, 0
    for role, name in (("capsule", "capsule.capsule"), ("currents", "currents.json"),
                       ("digests", "digests.json"), ("planes", "planes.npy")):
        data = role.encode()
        (tmp_path / "keep" / "t" / name).write_bytes(data)
        files[role] = {"file": f"keep/t/{name}", "sha256": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data)}
        total += len(data)
    ledger = {"schema": identity.LEDGER_SCHEMA, "version": identity.LEDGER_VERSION,
              "fixture_bytes": total if fixture_bytes is None else fixture_bytes,
              "sets": {"keep": {"cases": {"t": {"files": files}}}}}
    (tmp_path / identity.LEDGER_FILE).write_text(json.dumps(ledger))
    return ledger


def test_the_ledger_refuses_by_name_every_way_it_can_be_wrong(tmp_path):
    _write_ledger(tmp_path)
    assert identity.read_ledger(tmp_path)["fixture_bytes"] > 0
    (tmp_path / "keep" / "t" / "planes.npy").write_bytes(b"other")
    with pytest.raises(identity.IdentityRefusal, match="not the file the ledger records"):
        identity.read_ledger(tmp_path)
    (tmp_path / "keep" / "t" / "planes.npy").write_bytes(b"planes")
    _ = identity.read_ledger(tmp_path)
    ledger = json.loads((tmp_path / identity.LEDGER_FILE).read_text())
    ledger["fixture_bytes"] = 1
    (tmp_path / identity.LEDGER_FILE).write_text(json.dumps(ledger))
    with pytest.raises(identity.IdentityRefusal, match="total"):
        identity.read_ledger(tmp_path)
    ledger["sets"] = {"fz": ledger["sets"]["keep"]}
    (tmp_path / identity.LEDGER_FILE).write_text(json.dumps(ledger))
    with pytest.raises(identity.IdentityRefusal, match="unknown modes"):
        identity.read_ledger(tmp_path)
    ledger["version"] = 2
    (tmp_path / identity.LEDGER_FILE).write_text(json.dumps(ledger))
    with pytest.raises(identity.IdentityRefusal, match="version 1 only"):
        identity.read_ledger(tmp_path)


def test_the_fixtures_resource_names_an_absent_directory_and_a_ledger_that_does_not_read(tmp_path, monkeypatch):
    assert "does not exist" in identity.fixtures_absence(tmp_path / "absent")
    assert "do not read" in identity.fixtures_absence(tmp_path)
    _write_ledger(tmp_path)
    assert identity.fixtures_absence(tmp_path) is None
    monkeypatch.setenv(identity.FIXTURES_ENVIRONMENT, str(tmp_path / "elsewhere"))
    assert identity.fixtures_root() == tmp_path / "elsewhere"
    monkeypatch.delenv(identity.FIXTURES_ENVIRONMENT)
    assert identity.fixtures_root() == _ROOT / identity.FIXTURES_RELATIVE


def test_the_interceptor_records_complex_times_complex_and_nothing_else():
    a = (numpy.arange(4) + 1j).astype(numpy.complex64).view(identity.WatchedArray)
    r = numpy.arange(4, dtype=numpy.float32).view(identity.WatchedArray)
    watch = identity._Watch()  # noqa: SLF001 - the record a watch block keeps
    identity._ACTIVE.append(watch)  # noqa: SLF001
    try:
        product = a * a
        _ = r * a
        _ = a * numpy.float32(2.0)
        _ = a * (1.0 + 0.0j)
        product *= a
        numpy.multiply(r, r, out=r)
        _ = numpy.dot(a, a)
    finally:
        identity._ACTIVE.remove(watch)  # noqa: SLF001
    report = watch.report()
    assert report["multiplies_observed"] == 7
    assert report["complex_times_complex"] == 4  # a*a, a*complex scalar, product*=a, dot(a, a)
    assert report["functions_observed"] == {"dot": 1}
    assert isinstance(product, identity.WatchedArray)
    plain = a.view(numpy.ndarray)
    assert product.view(numpy.ndarray).tobytes() == ((plain * plain) * plain).tobytes()


def test_the_interceptor_views_a_drivers_arrays_and_puts_them_back():
    driver = _driver()
    before = {name: driver.fields.get_component(name) for name in ("Dz", "Bx")}
    with identity.watching_complex_multiplies(driver) as watch:
        assert isinstance(driver.fields.get_component("Dz"), identity.WatchedArray)
        driver.step()
    assert watch.multiplies > 0 and watch.complex_multiplies == 0
    for name, array in before.items():
        assert driver.fields.get_component(name) is array and type(array) is numpy.ndarray


def test_flush_is_reached_without_meep_where_a_meep_less_lever_exists(monkeypatch):
    """MEEP blocked, as in a Tier A replay: on x86-64 glibc (MXCSR FTZ|DAZ) and on Darwin
    (FE_DFL_DISABLE_DENORMS_ENV) the host reaches flush, measured in all four probes,
    and the entry mode is back afterwards; on any other ABI the refusal names it."""
    from . import subnormal_policy  # noqa: PLC0415

    monkeypatch.delitem(sys.modules, "meep", raising=False)
    entry = identity.subnormal_state()["mode"]
    if sys.platform == "darwin" or subnormal_policy.glibc_x86_64_abi()[0]:
        with identity.host_subnormal_mode("flush") as report:
            assert report["measured"]["mode"] == "flush", report
            assert entry == "flush" or report["driven_by"].startswith("fenv:"), report
        assert identity.subnormal_state()["mode"] == entry
        return
    with pytest.raises(identity.IdentityRefusal, match="x86-64 glibc MXCSR lever"):
        with identity.host_subnormal_mode("flush"):
            pass


def test_the_host_mode_is_measured_and_an_unknown_mode_refused():
    with identity.host_subnormal_mode("keep") as report:
        assert report["measured"]["mode"] == "keep"
        report["check"]("a test")
    assert identity.subnormal_state()["mode"] == report["entry"]["mode"]
    with pytest.raises(identity.IdentityRefusal, match="unknown subnormal mode"):
        with identity.host_subnormal_mode("fz"):
            pass
