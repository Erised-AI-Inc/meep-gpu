"""Tier C of MEEP identity and its minting tool.

The judging rules of ``record_meep_identity.py`` and the word counts of
``meep_identity_tier_c.py`` are pinned with NumPy only. One test runs Tier C itself on a
probe case to step 20 and replays its output with MEEP blocked; it declares the resource
``meep_identity_reference`` (``tools/ci/declared_resources.txt``): an identity-reference
MEEP, measured by ``tools/check_install.py``'s identity check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from parity.meep_gpu import meep_identity_tier_c as tier_c
from parity.meep_gpu import record_meep_identity as mint

HERE = Path(__file__).resolve().parent


def _good_case():
    steps = list(range(21))
    tier = {"stored": {"words": 100, "word_differing": 0, "value_differing": 0, "sign_of_zero_only": 0},
            "reader": {"words": 50, "word_differing": 0, "value_differing": 0, "sign_of_zero_only": 0},
            "ghost": {"words": 10, "word_differing": 3, "value_differing": 0, "sign_of_zero_only": 3},
            "reads": {"inexact": 0, "imaginary": 0, "point_read_differing": 0,
                      "not_stored_compared": 0, "held_copies_differing": 0,
                      "held_aliased_unexplained": 0, "held_aliased_complex": 0,
                      "held_unconnected_unexplained": 0},
            "subnormals": {"mode": "keep"},
            "steps": steps, "fpu_checks": len(steps),
            "planted_flip": {"caught": True, "words_differing": 1, "digest_caught": True,
                             "planes_differing": [4], "plane": 4},
            "smoke_record": {"equal": 72, "of": 72, "lattice_equal": True}}
    replay = {"meep_imported": False, "digests": 12 * len(steps), "stored_equal": 12 * len(steps),
              "reader_equal": 12 * len(steps), "planted_flip": {"caught": True, "placed": True},
              "fpu": {"measured": {"mode": "keep"}}, "first_difference": None}
    return tier, replay


def test_a_case_that_reproduces_every_claim_may_be_minted():
    tier, replay = _good_case()
    assert mint.judge_case("keep", tier, replay) == []
    assert tier_c.compare_holds(dict(tier, smoke_record=tier["smoke_record"]))


@pytest.mark.parametrize("change,named", [
    (lambda t, r: t["stored"].update(word_differing=1), "word-differing"),
    (lambda t, r: t["stored"].update(value_differing=1, word_differing=1), "value-differing"),
    (lambda t, r: t["stored"].update(words=0), "compared no stored word"),
    (lambda t, r: t["reader"].update(word_differing=2), "reader words differ"),
    (lambda t, r: t["reads"].update(inexact=1), "exact float32"),
    (lambda t, r: t["reads"].update(point_read_differing=1), "point read returns"),
    (lambda t, r: t["reads"].update(not_stored_compared=1), "no MEEP stored word"),
    (lambda t, r: t["reads"].update(held_copies_differing=1), "different words by two MEEP chunks"),
    (lambda t, r: t["reads"].update(held_aliased_unexplained=1), "neither B's nor H's"),
    (lambda t, r: t["reads"].update(held_aliased_complex=2), "2 were read through the point read"),
    (lambda t, r: t["reads"].update(held_unconnected_unexplained=3),
     "3 compared B points of a chunk that took an H array"),
    (lambda t, r: t.update(subnormals={"mode": "flush"}), "FPU flush"),
    (lambda t, r: t.update(fpu_checks=3), "measured the FPU at 3"),
    (lambda t, r: t["planted_flip"].update(caught=False), "planted flip"),
    (lambda t, r: t["planted_flip"].update(planes_differing=[4, 5]), "planted flip"),
    (lambda t, r: t["smoke_record"].update(equal=71), "smoke record"),
    (lambda t, r: r.update(meep_imported=True), "MEEP was imported"),
    (lambda t, r: r.update(stored_equal=251), "reproduced 251 stored"),
    (lambda t, r: r.update(reader_equal=0), "0 reader digests"),
    (lambda t, r: r["planted_flip"].update(placed=False), "caught and placed"),
    (lambda t, r: r.update(fpu={"measured": {"mode": "mixed"}}), "FPU mixed"),
    (lambda t, r: t.update(error="no MEEP"), "Tier C did not run"),
    (lambda t, r: r.update(error="blocked"), "replay did not run"),
])
def test_every_claim_the_ledger_makes_refuses_the_mint_by_name(change, named):
    tier, replay = _good_case()
    change(tier, replay)
    problems = mint.judge_case("keep", tier, replay)
    assert any(named in problem for problem in problems), problems


def test_ghost_words_are_reported_and_never_judged():
    tier, replay = _good_case()
    tier["ghost"].update(word_differing=10**6, value_differing=0, sign_of_zero_only=10**6)
    assert mint.judge_case("keep", tier, replay) == []


def test_the_word_counts_separate_values_from_signs_of_zero():
    ours = numpy.array([[[1.0, 0.0, 2.0, numpy.nan]]], dtype=numpy.float32)
    theirs = numpy.array([[[1.0, -0.0, 3.0, numpy.nan]]], dtype=numpy.float32)
    every = numpy.ones(ours.shape, dtype=bool)
    assert tier_c.word_counts(ours, theirs, every) == {
        "words": 4, "word_differing": 2, "value_differing": 1, "sign_of_zero_only": 1}
    some = every.copy()
    some[0, 0, 2] = False
    assert tier_c.word_counts(ours, theirs, some)["value_differing"] == 0


def _run(arguments, out_log):
    command = [sys.executable, "-u", str(HERE / "meep_identity_tier_c.py"), *arguments]
    done = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
    out_log.write_text(done.stdout + done.stderr)
    return done


def test_tier_c_holds_on_a_probe_case_and_its_replay_reproduces_it(tmp_path):
    out = tmp_path / "tier_c"
    done = _run(["compare", "--out", str(out), "--cases", "c_prime", "--steps", "20"],
                tmp_path / "compare.log")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    result = json.loads((out / "keep" / "c_prime" / "result.json").read_text())
    assert result["stored"]["word_differing"] == 0 and result["stored"]["words"] > 0
    assert result["reader"]["word_differing"] == 0 and result["planted_flip"]["caught"]
    assert result["smoke_record"]["equal"] == result["smoke_record"]["of"] > 0
    replayed = _run(["replay", "--source", str(out)], tmp_path / "replay.log")
    assert replayed.returncode == 0, replayed.stdout[-3000:]
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    if identity.fixtures_absence() is None:
        ledger = identity.read_ledger()
        entry = ledger["sets"].get("keep", {}).get("cases", {}).get("c_prime")
        if entry is not None:
            root = identity.fixtures_root()
            minted, _planes = identity.read_digests(root / entry["files"]["digests"]["file"],
                                                    root / entry["files"]["planes"]["file"])
            here, _ = identity.read_digests(out / "keep" / "c_prime" / "digests.json",
                                            out / "keep" / "c_prime" / "planes.npy")
            for step in here["steps"]:
                assert here["stored"][str(step)] == minted["stored"][str(step)], step
                assert here["reader"][str(step)] == minted["reader"][str(step)], step


def _exact_reader_cases(mp):
    """A complex 2-D grid (Bloch k, PML, several chunks) and a Dcyl m = 1 grid."""
    planar = mp.Simulation(
        cell_size=mp.Vector3(4, 3, 0), resolution=10, boundary_layers=[mp.PML(0.5)],
        k_point=mp.Vector3(0.3, 0, 0),
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), mp.Ez, center=mp.Vector3(0.2, 0.1))],
        geometry=[mp.Block(mp.Vector3(1, 1, mp.inf), material=mp.Medium(epsilon=4))])
    axial = mp.Simulation(
        cell_size=mp.Vector3(3, 0, 4), resolution=10, dimensions=mp.CYLINDRICAL, m=1,
        boundary_layers=[mp.PML(0.5)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), mp.Er, center=mp.Vector3(0.7, 0, 0.2))],
        geometry=[mp.Block(mp.Vector3(0.8, mp.inf, 1), center=mp.Vector3(0.6, 0, 0),
                           material=mp.Medium(epsilon=4))])
    return [(planar, False, mp.Ez), (axial, True, mp.Er)]


def test_the_exact_reader_reads_the_signed_zero_meeps_point_read_drops():
    """MEEP's point read returns (1, +0) * word: an imaginary -0 beside a positive real part
    reads +0. A word planted in MEEP's own array is read back exactly, every other owned
    word equals MEEP's point read once that read is applied, the metallic walls of the
    Dcyl grid (no chunk owns them) are read from the chunks that hold them, and a reader
    pointed at the wrong word is caught by its own check."""
    import meep as mp  # noqa: PLC0415

    mp.verbosity(0)
    for sim, cylindrical, component in _exact_reader_cases(mp):
        sim.init_sim()
        sim.run(until=2)
        gv = sim.fields.gv
        low, high, shift = gv.little_corner(), gv.big_corner(), gv.iyee_shift(component)

        def owned(first, last, odd):
            # grid_volume::owns: 0 < p - little_corner <= 2 n on every axis
            return [*range(first + (odd or 2), last + 1, 2)]

        if cylindrical:  # r = 0 is owned by the axis chunk; z = little_corner is a wall
            axes = ([*range(low.r() + shift.r(), high.r() + 1, 2)], [0],
                    [*range(low.z() + shift.z(), high.z() + 1, 2)])
        else:
            axes = (owned(low.x(), high.x(), shift.x()), owned(low.y(), high.y(), shift.y()), [0])
        reader = tier_c.MeepReader(mp, sim, component, axes, True, cylindrical)
        words, inexact, imaginary, differing = reader.read()
        assert words.dtype == numpy.complex64 and words.shape == reader.shape
        assert (inexact, imaginary, differing) == (0, 0, 0)
        assert reader.stored.all() and len(reader.groups) > 1
        walls = reader.stored & ~reader.owned
        assert walls.any() == cylindrical and not reader.held_copies_differing.any()
        if cylindrical:  # MEEP allocates the low wall as +0 and never writes it
            assert not words[walls].view(numpy.uint32).any()
        chunk, positions, indices = reader.groups[-1]
        mine = reader.owned.reshape(-1)[positions]
        positions, indices = positions[mine], indices[mine]
        address = int((__import__("ctypes").c_void_p * reader.num_chunks).from_address(
            int(sim.fields.chunks))[chunk])
        first, second = tier_c._chunk_parts(address, int(component))  # noqa: SLF001
        real = tier_c._float_words(first, reader.ntot[chunk])  # noqa: SLF001
        imag = tier_c._float_words(second, reader.ntot[chunk])  # noqa: SLF001
        position, index = int(positions[0]), int(indices[0])
        kept = (real[index].copy(), imag[index].copy())
        try:
            real[index], imag[index] = numpy.float32(0.75), numpy.float32(-0.0)
            words, _inexact, _imaginary, differing = reader.read()
            word = words.reshape(-1)[position]
            assert numpy.float32(word.real) == numpy.float32(0.75)
            assert numpy.float32(word.imag).view(numpy.uint32) == numpy.uint32(0x80000000)
            point = sim.fields.get_field(component, reader.points[position])
            assert point.real == 0.75 and point.imag == 0.0 and not numpy.signbit(point.imag)
            assert differing == 0
            # The check: a reader pointed one word off is refused by its own count.
            whole = reader.groups[-1]
            reader.groups[-1] = (chunk, whole[1], numpy.where(whole[2] == index, index + 1, whole[2]))
            assert reader.read()[3] > 0
            reader.groups[-1] = whole
        finally:
            real[index], imag[index] = kept


def test_a_held_b_word_of_a_chunk_whose_b_array_is_its_h_array_reads_meeps_b():
    """MEEP allocates B and H as one array and gives H its own only where it needs one (a
    PML along the component, mu, a polarization). On a Bloch axis with a PML at one end, the
    bottom chunk keeps Hy == By, and its not-owned low row receives the image's By and then
    the image's Hy into the same word: the word left is Hy's. The reader reads MEEP's By
    there (its point read, the image's B), every substituted word is H's point read, and
    the counts name them."""
    import ctypes  # noqa: PLC0415

    import meep as mp  # noqa: PLC0415

    mp.verbosity(0)
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 3, 0), resolution=10, k_point=mp.Vector3(),
        boundary_layers=[mp.PML(0.5, direction=mp.Y, side=mp.High)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), mp.Ez, center=mp.Vector3(0.1, 0.4))])
    sim.init_sim()
    sim.run(until=12)
    gv = sim.fields.gv
    low, high, shift = gv.little_corner(), gv.big_corner(), gv.iyee_shift(mp.By)
    xs = [*range(low.x() + (shift.x() or 2), high.x() + 1, 2)]
    ys = [*range(low.y() + shift.y(), high.y() + 1, 2)]  # y = little_corner: no chunk owns it
    assert shift.y() == 0 and ys[0] == low.y()
    reader = tier_c.MeepReader(mp, sim, mp.By, (xs, ys, [0]), False)
    words, inexact, imaginary, differing = reader.read()
    assert (inexact, imaginary, differing) == (0, 0, 0)
    held = reader.stored & ~reader.owned
    assert held[:, 0, 0].all() and not held[:, 1:, :].any()
    assert (reader.held_aliased == held).all()
    substituted = reader.held_aliased_substituted
    assert substituted.any() and not reader.held_aliased_unexplained.any()
    # The substituted words are MEEP's B (its point read); the shared word is H's.
    assert (words.view(numpy.uint32) == reader.point_words.view(numpy.uint32)).all()
    chunks = (ctypes.c_void_p * reader.num_chunks).from_address(int(sim.fields.chunks))
    for chunk, positions, indices in reader.groups:
        moved = substituted.reshape(-1)[positions]
        if not moved.any():
            continue
        first, _ = tier_c._chunk_parts(int(chunks[chunk]), int(mp.By))  # noqa: SLF001
        h_first, _ = tier_c._chunk_parts(int(chunks[chunk]), int(mp.Hy))  # noqa: SLF001
        assert first == h_first
        shared = tier_c._float_words(first, reader.ntot[chunk])[indices[moved]]  # noqa: SLF001
        h_read = numpy.asarray([sim.fields.get_field(mp.Hy, reader.points[k]) for k in positions[moved]])
        assert (shared.view(numpy.uint32) == h_read.real.astype(numpy.float32).view(numpy.uint32)).all()
        assert (shared.view(numpy.uint32) != words.reshape(-1)[positions[moved]].view(numpy.uint32)).all()


@pytest.mark.parametrize("complex_fields", [False, True], ids=["real", "complex"])
def test_a_held_b_word_of_a_chunk_that_took_its_own_h_array_this_step_reads_its_image(complex_fields):
    """MEEP builds a step's connections before update_eh gives H an array of its own, so in
    the first step a chunk with a PML along the component receives no B at its held points:
    the word stays the one it held before the step. On a Bloch axis with k = 0, a PML at both
    ends and a field present before the first step, the bottom chunk's held By row keeps its
    +0 while its image one lattice vector up carries a nonzero By. The reader reads the
    image's exact word there, names the points, and refuses a held word that is neither the
    image's nor the earlier one; from the second step on, B's connection writes the image's
    word and nothing is substituted."""
    import ctypes  # noqa: PLC0415
    import math  # noqa: PLC0415

    import meep as mp  # noqa: PLC0415

    mp.verbosity(0)
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 3, 0), resolution=10, k_point=mp.Vector3(),
        boundary_layers=[mp.PML(0.5, direction=mp.Y)], force_complex_fields=complex_fields,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), mp.Ez, center=mp.Vector3(0.1, 0.4))])
    sim.init_sim()
    sim.initialize_field(mp.Ez, lambda p: math.cos(math.pi * p.x) + 0.5)
    gv = sim.fields.gv
    low, high, shift = gv.little_corner(), gv.big_corner(), gv.iyee_shift(mp.By)
    xs = [*range(low.x() + (shift.x() or 2), high.x() + 1, 2)]
    ys = [*range(low.y() + shift.y(), high.y() + 1, 2)]  # y = little_corner: no chunk owns it
    assert shift.y() == 0 and ys[0] == low.y() and int(sim.fields.t) == 0
    reader = tier_c.MeepReader(mp, sim, mp.By, (xs, ys, [0]), complex_fields)
    held = reader.stored & ~reader.owned
    assert held[:, 0, 0].all() and not held[:, 1:, :].any()
    sim.run(until=0.5 * sim.fields.dt)
    assert int(sim.fields.t) == 1
    words, inexact, imaginary, differing = reader.read()
    assert (inexact, imaginary, differing) == (0, 0, 0)
    assert (reader.held_unconnected == held).all() and not reader.held_aliased.any()
    substituted = reader.held_unconnected_substituted
    assert substituted.any() and not reader.held_unconnected_unexplained.any()
    chunks = (ctypes.c_void_p * reader.num_chunks).from_address(int(sim.fields.chunks))
    volumes = sim.structure.get_chunk_volumes()
    span = 2 * gv.ny()
    values = []
    for chunk, positions, indices in reader.groups:
        moved = substituted.reshape(-1)[positions]
        if not moved.any():
            continue
        first, second = tier_c._chunk_parts(int(chunks[chunk]), int(mp.By))  # noqa: SLF001
        h_first, _ = tier_c._chunk_parts(int(chunks[chunk]), int(mp.Hy))  # noqa: SLF001
        assert first != h_first  # H took an array of its own in this step
        # MEEP's word at the held point is still the word it was allocated with ...
        stale = tier_c._float_words(first, reader.ntot[chunk])[indices[moved]]  # noqa: SLF001
        assert not stale.view(numpy.uint32).any()
        # ... and the word read is the image's, one lattice vector up in y.
        for k in positions[moved]:
            point = reader.points[k]
            image = mp.ivec(point.x(), point.y() + span, point.z())
            owner = next(q for q, volume in enumerate(volumes) if volume.owns(image))
            o_first, o_second = tier_c._chunk_parts(int(chunks[owner]), int(mp.By))  # noqa: SLF001
            at = int(volumes[owner].index(mp.By, image))
            real = tier_c._float_words(o_first, reader.ntot[owner])[at]  # noqa: SLF001
            word = words.reshape(-1)[k]
            assert numpy.float32(numpy.real(word)).view(numpy.uint32) == numpy.float32(real).view(numpy.uint32)
            if complex_fields:
                imag = tier_c._float_words(o_second, reader.ntot[owner])[at]  # noqa: SLF001
                assert (numpy.float32(numpy.imag(word)).view(numpy.uint32)
                        == numpy.float32(imag).view(numpy.uint32))
            values.append(float(real))
    assert any(value != 0.0 for value in values)  # a value the held word does not carry
    if not complex_fields:  # on real storage MEEP's point read of the held point is the image's word
        assert (words[substituted].view(numpy.uint32)
                == reader.point_words[substituted].view(numpy.uint32)).all()
    # A held word that is neither the image's nor the earlier one is refused by name.
    chunk, positions, indices = next(group for group in reader.groups
                                     if substituted.reshape(-1)[group[1]].any())
    first, _ = tier_c._chunk_parts(int(chunks[chunk]), int(mp.By))  # noqa: SLF001
    array = tier_c._float_words(first, reader.ntot[chunk])  # noqa: SLF001
    index = int(indices[substituted.reshape(-1)[positions]][0])
    kept = array[index].copy()
    try:
        array[index] = numpy.float32(0.75)
        reader.read()
        assert int(reader.held_unconnected_unexplained.sum()) == 1
    finally:
        array[index] = kept
    reader.read()
    assert not reader.held_unconnected_unexplained.any() and substituted.sum() == reader.held_unconnected_substituted.sum()
    # From the second step on, B's connection writes the image's word: nothing is substituted.
    sim.run(until=0.5 * sim.fields.dt)
    assert int(sim.fields.t) == 2
    words, _inexact, _imaginary, differing = reader.read()
    assert differing == 0
    assert not reader.held_unconnected.any() and not reader.held_unconnected_substituted.any()
    for chunk, positions, indices in reader.groups:
        rows = held.reshape(-1)[positions]
        if not rows.any():
            continue
        first, _ = tier_c._chunk_parts(int(chunks[chunk]), int(mp.By))  # noqa: SLF001
        raw = tier_c._float_words(first, reader.ntot[chunk])[indices[rows]]  # noqa: SLF001
        assert (raw.view(numpy.uint32) == numpy.real(words.reshape(-1)[positions[rows]])
                .astype(numpy.float32).view(numpy.uint32)).all()
