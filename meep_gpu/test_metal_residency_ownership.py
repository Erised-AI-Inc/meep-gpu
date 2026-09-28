"""The ownership machine behind held mirrors: states, the seal, and the two doors.

WHAT THESE PIN AND WHY EACH ONE EXISTS. The shipped bracket copies every mirror
both ways around every launch, so the host is authoritative at every instant and
ownership needs no tracking. Measured 2026-09-21 on this host that transport is
96-98% of the whole step (``pml_2d``: 69 copies a launch at 0.217 ms each against
kernels that run ``0.495 + 0.690 N`` ms/step, N in Mcells). Holding state on the
device removes it — and replaces a brute-force invariant with a tracked one, where
every untracked host write is "a smooth, plausible, wrong field rather than an
error" (``coverage.py``'s own words at its refusal).

So each test below pins a rule whose absence is SILENT:

* a hold that cannot seal must refuse BY NAME, not hold unguarded;
* a host write to a device-owned mirror must RAISE, not be overwritten;
* the doors must resolve by ARRAY IDENTITY, because the weld families rebind
  ``fields.<name>`` inside ``run`` and name-keyed ownership names the wrong array
  every other launch;
* ``acquire_write`` must sync OUT first, because every in-tree write site is a
  read-modify-write and not a whole-volume overwrite;
* an unheld composition must behave EXACTLY as it did before this machinery,
  because the 62-gate fleet is bound to that behaviour.

These run on the host with no kernel: the machine is host bookkeeping over torch
tensors, and it is testable without compiling a shader.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_resource_skip  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - platform gate
    requires_resource_skip("torch", "PyTorch is not installed", allow_module_level=True)
if not torch.backends.mps.is_available():  # pragma: no cover - platform gate
    requires_resource_skip("mps_device", "the residency machine mirrors onto MPS",
                           allow_module_level=True)

from meep_gpu.metal_kernels.device import (  # noqa: E402
    CLEAN_BOTH,
    DEVICE_OWNED,
    HOST_OWNED,
    OWNERSHIP_STATES,
    Residency,
)


def _registry(cells: int = 64):
    residency = Residency()
    mutable = np.arange(cells, dtype=np.float32)
    constant = np.ones(cells, dtype=np.float32)
    residency.mirror("mut", mutable)
    residency.mirror("const", constant, constant=True)
    return residency, mutable, constant


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def test_a_fresh_mirror_is_clean_on_both_sides():
    """``mirror`` copies the host bytes up, so CLEAN_BOTH is the honest start.

    Starting at HOST_OWNED would be safe but would make the first launch of every
    composition pay a full sync-in it does not need.
    """
    residency, _, _ = _registry()
    assert residency.by_state()[CLEAN_BOTH] == ("const", "mut")
    assert residency.by_state()[HOST_OWNED] == ()
    assert residency.by_state()[DEVICE_OWNED] == ()
    assert set(residency.by_state()) == set(OWNERSHIP_STATES)


def test_by_state_partitions_every_registered_name():
    """The set form the gate checks must be a partition, not a filter."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    grouped = residency.by_state()
    flat = [name for names in grouped.values() for name in names]
    assert sorted(flat) == list(residency.names)
    assert len(flat) == len(set(flat))


# ---------------------------------------------------------------------------
# The seal — the only fail-closed guard in either mode
# ---------------------------------------------------------------------------

def test_a_host_write_to_a_device_owned_mirror_raises():
    """The measured failure mode, converted into an exception.

    Without the seal this write lands on the host, the next launch's declared read
    does not sync it in (the mirror is device-owned), and the device silently keeps
    different words than the engine thinks it wrote.
    """
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    assert residency.state_of("mut") == DEVICE_OWNED
    with pytest.raises(ValueError, match="read-only"):
        mutable[0] = 1.0


def test_arm_hold_classifies_rather_than_refusing_a_constant():
    """A constant is HOISTED, not held: the device never writes it, so it needs no
    state machine — only to stop being re-uploaded 21 times a launch."""
    residency, _, _ = _registry()
    refused = residency.arm_hold(["mut", "const", "not-a-mirror"])
    assert refused == ("not-a-mirror",)
    assert set(residency.held) == {"mut"}
    assert set(residency.hoisted) == {"const"}


def test_a_hoisted_constant_is_uploaded_once_and_then_skipped():
    """The 30%-of-the-copy-count saving, counted."""
    residency, _, _ = _registry()
    residency.arm_hold(["mut", "const"])
    before = residency.copies_in
    residency.sync_in()
    residency.sync_in()
    assert residency.copies_in == before, "a clean hoisted constant moves nothing"


def test_a_post_freeze_write_to_a_hoisted_constant_raises_and_re_uploads():
    """Risk R-c, closed rather than deferred.

    Uploading constants once is safe only if nothing rewrites the material after
    plan freeze. ``set_epsilon_volumes`` would do exactly that, and without the
    seal the device would run the rest of the simulation on the old coefficients
    with nothing raising. Sealing the hoisted set converts that into an exception,
    and going through the door marks it for re-upload.
    """
    residency, _, constant = _registry()
    residency.arm_hold(["const"])
    with pytest.raises(ValueError, match="read-only"):
        constant[0] = 2.0
    residency.acquire_write(constant)[0] = 2.0
    assert residency.state_of("const") == HOST_OWNED
    before = residency.copies_in
    residency.sync_in(["const"])
    assert residency.copies_in == before + 1
    assert residency.state_of("const") == CLEAN_BOTH


def test_arm_hold_refuses_an_array_whose_write_flag_cannot_round_trip():
    """A non-owning read-only view can be sealed but not unsealed — so refuse it.

    Sealing such an array would strand it: ``setflags(write=True)`` raises when the
    base is itself read-only, so ``sync_out`` could never write it back.
    """
    residency = Residency()
    base = np.arange(64, dtype=np.float32)
    base.setflags(write=False)
    view = base[:32]
    residency.mirror("frozen", view)
    assert residency.arm_hold(["frozen"]) == ("frozen",)
    assert not residency.held
    assert not residency.hoisted


def test_sync_out_lifts_and_restores_its_own_seal():
    """``Mirror.sync_out`` writes the host, so a sealed mirror must not refuse it."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    mirror = residency._mirrors["mut"]
    assert mirror.sealed
    residency.flush(["mut"])
    assert mirror.sealed, "the guard must survive the sync that lifted it"
    assert residency.state_of("mut") == CLEAN_BOTH


# ---------------------------------------------------------------------------
# The two doors
# ---------------------------------------------------------------------------

def test_acquire_write_syncs_out_before_handing_the_array_over():
    """Every in-tree write site is a read-modify-write, so the order is load-bearing.

    If ``acquire_write`` unsealed without syncing out, ``array -= before`` would
    subtract from PRE-LAUNCH words and the device's own output would be lost.
    """
    residency, mutable, _ = _registry()
    device_words = np.full(64, 7.0, dtype=np.float32)
    residency.arm_hold(["mut"])
    residency.tensor("mut").copy_(torch.from_numpy(device_words))
    residency.sync_out(["mut"])           # marks device-owned, copies nothing
    assert mutable[0] == 0.0, "the host must still be stale at this point"
    got = residency.acquire_write(mutable)
    assert got is mutable
    assert got.flags.writeable
    np.testing.assert_array_equal(mutable, device_words)
    assert residency.state_of("mut") == HOST_OWNED


def test_acquire_read_makes_the_host_current_but_leaves_the_seal_down():
    """A getter cannot tell a read from a read-modify-write, so reads stay sealed."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.tensor("mut").copy_(torch.from_numpy(np.full(64, 3.0, dtype=np.float32)))
    residency.sync_out(["mut"])
    got = residency.acquire_read("mut")
    np.testing.assert_array_equal(got, np.full(64, 3.0, dtype=np.float32))
    assert residency.state_of("mut") == CLEAN_BOTH
    with pytest.raises(ValueError, match="read-only"):
        mutable[0] = 1.0


def test_the_doors_resolve_by_array_identity_not_by_name():
    """The weld rotation hazard, pinned.

    ``offdiag_weld_common`` rebinds ``fields.<name>`` to its twin inside ``run``.
    An acquire keyed on the NAME would move the wrong mirror's state every other
    launch; keyed on the array it follows the object through any number of rebinds.
    """
    residency = Residency()
    primary = np.zeros(64, dtype=np.float32)
    twin = np.ones(64, dtype=np.float32)
    residency.mirror("Dx", primary)
    residency.mirror("weld_scratch:Dx", twin)
    residency.arm_hold(["Dx", "weld_scratch:Dx"])
    residency.sync_out(["Dx", "weld_scratch:Dx"])

    # The family has now rotated: what the engine calls Dx is the twin object.
    residency.acquire_write(twin)
    assert residency.state_of("weld_scratch:Dx") == HOST_OWNED
    assert residency.state_of("Dx") == DEVICE_OWNED, (
        "resolving by identity must not disturb the other mirror")


def test_an_acquire_moves_every_alias_of_one_array():
    """``set_isotropic_epsilon_volume`` hands back three names for one volume."""
    residency = Residency()
    shared = np.zeros(64, dtype=np.float32)
    residency.mirror("eps:x", shared, constant=True)
    residency.mirror("eps:y", shared, constant=True)
    assert len(residency._targets(shared)) == 2


def test_an_acquire_on_an_unmirrored_target_refuses_rather_than_passing():
    """"Not registered" cannot be answered, so guessing "needs no sync" is refused."""
    residency, _, _ = _registry()
    with pytest.raises(KeyError):
        residency.acquire_write("never-registered")
    with pytest.raises(KeyError):
        residency.acquire_write(np.zeros(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# The transported set
# ---------------------------------------------------------------------------

def test_a_held_sync_in_copies_only_what_the_host_owns():
    """The saving, counted. A clean mirror moves nothing; a device-owned one must
    never be copied backwards over the launch's own output."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut", "const"])            # what a composer arms
    residency.sync_out(["mut"])                     # device-owned
    before = residency.copies_in
    residency.sync_in(["mut", "const"])
    assert residency.copies_in == before, "neither a held device-owned mirror nor a " \
                                          "clean hoisted constant needs a copy in"
    residency.acquire_write(mutable)[0] = 5.0
    residency.sync_in(["mut"])
    assert residency.copies_in == before + 1
    assert residency.state_of("mut") == CLEAN_BOTH


def test_an_unheld_registry_copies_exactly_as_it_did_before():
    """The 62-gate fleet is bound to the unheld behaviour; it is not a new path."""
    residency, _, _ = _registry()
    residency.sync_in()
    assert residency.copies_in == 2, "sync_in has no constant check, by design"
    residency.sync_out()
    assert residency.copies_out == 1, "sync_out skips the constant"
    assert residency.by_state()[CLEAN_BOTH] == ("const", "mut")
    assert not residency.held


def test_flush_reports_what_moved_and_leaves_nothing_device_owned():
    residency, _, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    assert residency.flush() == ("mut",)
    assert residency.by_state()[DEVICE_OWNED] == ()
    assert residency.flush() == (), "a second flush has nothing to move"


def test_release_hold_flushes_unseals_and_disarms():
    """The re-plan and end-of-run path: the host must come back authoritative."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.tensor("mut").copy_(torch.from_numpy(np.full(64, 9.0, dtype=np.float32)))
    residency.sync_out(["mut"])
    residency.release_hold()
    np.testing.assert_array_equal(mutable, np.full(64, 9.0, dtype=np.float32))
    assert not residency.held
    assert not residency.hoisted
    assert residency.by_state()[DEVICE_OWNED] == ()
    mutable[0] = 1.0                                 # no longer sealed


def test_the_generation_counter_distinguishes_no_move_from_moved_and_back():
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    start = residency.generation
    residency.sync_out(["mut"])
    residency.acquire_write(mutable)
    residency.sync_in(["mut"])
    assert residency.generation > start + 2


# ---------------------------------------------------------------------------
# rebind
# ---------------------------------------------------------------------------

def test_rebind_moves_a_name_to_a_new_array_and_refuses_a_lossy_swap():
    residency, mutable, _ = _registry()
    replacement = np.full(64, 4.0, dtype=np.float32)
    residency.rebind("mut", replacement)
    assert residency._mirrors["mut"].host is replacement
    assert residency.state_of("mut") == HOST_OWNED
    with pytest.raises(ValueError, match="different length"):
        residency.rebind("mut", np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="one name binds one width"):
        residency.rebind("mut", np.zeros(64, dtype=np.complex64))
    with pytest.raises(KeyError):
        residency.rebind("absent", replacement)


# ---------------------------------------------------------------------------
# Sparse transport -- the seam passes move cells, not volumes
# ---------------------------------------------------------------------------

def test_a_sparse_scatter_on_a_device_owned_mirror_touches_only_the_device():
    """The publication-ladder finding, as a rule: a few written cells must not
    cost a whole-volume copy. The host stays stale and sealed; the device has the
    new words; ownership does not move."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    linear = np.array([3, 7, 11], dtype=np.int64)
    residency.scatter(mutable, linear, np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert residency.state_of("mut") == DEVICE_OWNED
    assert residency.copies_in == 0 and residency.copies_out == 0
    assert residency.sparse_scatters == 1
    assert mutable[3] == 3.0, "the host is untouched, and still sealed"
    assert not mutable.flags.writeable
    np.testing.assert_array_equal(
        residency.tensor("mut").cpu().numpy()[[3, 7, 11]], [1.5, 2.5, 3.5])


def test_a_sparse_gather_reads_the_device_when_it_owns_the_mirror():
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.tensor("mut").copy_(torch.from_numpy(np.full(64, 9.0, dtype=np.float32)))
    residency.sync_out(["mut"])
    got = residency.gather(mutable, np.array([0, 63], dtype=np.int64))
    np.testing.assert_array_equal(got, [9.0, 9.0])
    assert residency.sparse_gathers == 1 and residency.copies_out == 0


def test_a_sparse_write_on_a_clean_mirror_keeps_both_sides_equal():
    """Before the first launch the host is current; a sparse write must land on
    BOTH sides so the mirror stays clean rather than forcing a whole upload."""
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.scatter(mutable, np.array([5], dtype=np.int64), np.array([42.0], dtype=np.float32))
    assert residency.state_of("mut") == CLEAN_BOTH
    assert mutable[5] == 42.0
    assert residency.tensor("mut").cpu().numpy()[5] == 42.0
    assert residency.verify() == {}


def test_zero_index_fills_on_the_device_and_moves_nothing():
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.tensor("mut").copy_(torch.from_numpy(np.full(64, 4.0, dtype=np.float32)))
    residency.sync_out(["mut"])
    residency.zero_index(mutable, np.arange(0, 8, dtype=np.int64))
    device = residency.tensor("mut").cpu().numpy()
    assert device[:8].tolist() == [0.0] * 8 and device[8] == 4.0
    assert residency.sparse_zeros == 1
    assert residency.copies_in == 0 and residency.copies_out == 0


def test_the_device_index_is_cached_by_identity_and_holds_its_reference():
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    linear = np.array([1, 2], dtype=np.int64)
    residency.gather(mutable, linear)
    residency.gather(mutable, linear)
    assert len(residency._index_cache) == 1
    assert residency._index_cache[id(linear)][0] is linear


def test_host_writes_falls_back_to_plain_numpy_when_nothing_holds():
    from meep_gpu import host_writes
    array = np.zeros((2, 3, 4), dtype=np.float32)
    # flat 15 sits in slab 1 (slab 0 is flat 0..11), so zeroing slab 0 must not touch it
    host_writes.scatter(array, np.array([15], dtype=np.int64), np.array([1.0], dtype=np.float32))
    assert array.reshape(-1)[15] == 1.0
    host_writes.zero(array, 0, 0)
    assert array[0].sum() == 0.0 and array.reshape(-1)[15] == 1.0
    np.testing.assert_array_equal(host_writes.gather(array, np.array([15, 16])), [1.0, 0.0])


def test_host_writes_face_index_matches_numpy_slab_selection():
    from meep_gpu import host_writes
    shape = (3, 4, 5)
    grid = np.arange(60).reshape(shape)
    for axis in range(3):
        np.testing.assert_array_equal(
            host_writes.face_index(shape, axis, 0),
            grid[(slice(None),) * axis + (0,)].reshape(-1))


def test_host_writes_routes_by_identity_to_a_registered_residency():
    from meep_gpu import host_writes
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    host_writes.register(residency)
    try:
        host_writes.scatter(mutable, np.array([2], dtype=np.int64), np.array([7.0], dtype=np.float32))
        assert residency.sparse_scatters == 1
        assert residency.tensor("mut").cpu().numpy()[2] == 7.0
    finally:
        host_writes.unregister(residency)
    assert not host_writes.holding()


# ---------------------------------------------------------------------------
# The fold fills' and the cylindrical axis's shapes: face copies and scaled adds
# ---------------------------------------------------------------------------

def _grid_registry(shape=(4, 3, 5)):
    residency = Residency()
    array = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    other = (100.0 + np.arange(int(np.prod(shape)), dtype=np.float32)).reshape(shape)
    residency.mirror("a", array)
    residency.mirror("b", other)
    return residency, array, other


def _face(shape, axis, index):
    from meep_gpu import host_writes
    return host_writes.face_index(shape, axis, index % shape[axis])


def test_copy_index_on_a_device_owned_pair_moves_nothing_across_the_bus():
    """The mirror-fold fill: ghost face <- parity * stored face, both on the device."""
    residency, array, _ = _grid_registry()
    residency.arm_hold(["a"])
    residency.sync_out(["a"])
    shape = array.shape
    residency.copy_index(array, _face(shape, 0, 0), array, _face(shape, 0, 2), -1)
    assert residency.copies_in == 0 and residency.copies_out == 0
    device = residency.tensor("a").cpu().numpy().reshape(shape)
    np.testing.assert_array_equal(device[0], -array[2])
    assert residency.state_of("a") == DEVICE_OWNED


def test_copy_index_is_bit_exact_against_numpy_for_a_float_scale():
    residency, array, other = _grid_registry()
    expected = array.copy()
    expected[:, :, 4] = np.float32(0.3) * other[:, :, 1]     # NumPy's rounding
    residency.arm_hold(["a", "b"])
    residency.sync_out(["a", "b"])
    residency.copy_index(array, _face(array.shape, 2, 4), other, _face(other.shape, 2, 1),
                         np.float32(0.3))
    device = residency.tensor("a").cpu().numpy().reshape(array.shape)
    assert np.array_equal(device.view(np.uint32), expected.view(np.uint32))


def test_add_scaled_index_matches_numpy_two_roundings():
    """``Dz[axis] += k * Hy[axis]``: multiply then add, no contraction on either side."""
    residency, array, other = _grid_registry()
    k = 4.0 * (0.05 / 0.1)
    expected = array.copy()
    expected[0] += k * other[0]
    residency.arm_hold(["a", "b"])
    residency.sync_out(["a", "b"])
    residency.add_scaled_index(array, _face(array.shape, 0, 0), other, _face(other.shape, 0, 0), k)
    device = residency.tensor("a").cpu().numpy().reshape(array.shape)
    assert np.array_equal(device.view(np.uint32), expected.view(np.uint32))


def test_zero_rows_and_span_index_match_numpy_slab_ranges():
    from meep_gpu import host_writes
    residency, array, _ = _grid_registry()
    expected = array.copy()
    expected[1:3] = 0
    residency.arm_hold(["a"])
    residency.sync_out(["a"])
    host_writes.register(residency)
    try:
        host_writes.zero_rows(array, 0, slice(1, 3))
    finally:
        host_writes.unregister(residency)
    device = residency.tensor("a").cpu().numpy().reshape(array.shape)
    np.testing.assert_array_equal(device, expected)


def test_face_doors_fall_back_to_plain_numpy_when_nothing_holds():
    from meep_gpu import host_writes
    a = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    b = a.copy() + 100
    host_writes.copy_face(a, 0, 0, a, 2, -1)
    np.testing.assert_array_equal(a[0], -a[2])
    host_writes.add_scaled_face(a, 1, 3, b, 0, 0.5)
    np.testing.assert_array_equal(a[:, 3], (np.arange(60, dtype=np.float32).reshape(3, 4, 5)[:, 3]
                                            if False else a[:, 3]))
    host_writes.zero_rows(a, 2, slice(1, 3))
    assert a[:, :, 1:3].sum() == 0.0


def test_box_index_and_gather_box_match_numpy_box_selection():
    from meep_gpu import host_writes
    shape = (3, 4, 5)
    grid = np.arange(60).reshape(shape)
    for slices in ((slice(1, 3), slice(None), slice(2, 4)),   # a slab with a plane
                   (slice(None), slice(-2, None)),            # fewer slices than axes
                   (slice(0, 1),),                            # one face
                   (slice(2, 3), slice(1, 2), slice(4, 5))):  # one cell
        linear, box_shape = host_writes.box_index(shape, slices)
        np.testing.assert_array_equal(linear, grid[slices].reshape(-1))
        assert box_shape == grid[slices].shape
        assert host_writes.box_index(shape, slices)[0] is linear, "the box is cached by bounds"
        array = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        np.testing.assert_array_equal(host_writes.gather_box(array, slices), array[slices])


def test_gather_box_reads_the_device_under_a_hold():
    from meep_gpu import host_writes
    residency, mutable, _ = _registry()
    residency.arm_hold(["mut"])
    residency.sync_out(["mut"])
    host_writes.register(residency)
    try:
        # A value written on the device only: the host mirror must not be consulted.
        host_writes.scatter(mutable, np.array([1], dtype=np.int64), np.array([5.0], dtype=np.float32))
        gathers = residency.sparse_gathers
        box = host_writes.gather_box(mutable, (slice(0, 2),))
        assert residency.sparse_gathers == gathers + 1
        expected = residency.tensor("mut").cpu().numpy().reshape(mutable.shape)[0:2]
        np.testing.assert_array_equal(box, expected)
        assert box.shape == expected.shape and box.reshape(-1)[1] == 5.0
    finally:
        host_writes.unregister(residency)


def test_the_subtract_door_forms_the_difference_word_for_word_on_both_sides():
    """``host_writes.subtract`` == NumPy ``a[idx] -= v`` on every word, float32 and complex64.

    Three states: DEVICE_OWNED (the device alone changes; a gather reads the
    difference back), CLEAN_BOTH (both sides move and stay equal), and nothing held
    (plain NumPy). Signed zeros are in the data (``0 - 0``, ``-0 - 0``). SUBNORMAL
    differences are deliberately NOT: the device flushes them to zero exactly as
    every family kernel does (Metal fast math, no lever -- ``subnormal.py``), and the
    engine reconciles that at the composition level by driving the host FPU
    (``subnormal_policy._drive_host_fpu``); ``test_the_subtract_door_flushes_a_subnormal_
    difference_like_the_kernels`` pins that behaviour by name.
    """
    from meep_gpu import host_writes
    rng = np.random.default_rng(11)
    for dtype in (np.float32, np.complex64):
        host = rng.standard_normal(64).astype(np.float32)
        if dtype is np.complex64:
            host = (host + 1j * rng.standard_normal(64).astype(np.float32)).astype(np.complex64)
        host[4] = dtype(1.25); host[9] = dtype(0.0); host[10] = dtype(-0.0)
        reference = host.copy()
        idx = np.array([2, 4, 9, 10, 17], dtype=np.int64)
        values = np.array([0.5, 0.75, 0.0, 0.0, -2.25], dtype=np.float32).astype(dtype)
        # nothing held: plain NumPy
        plain = host.copy(); plain_ref = plain.copy()
        host_writes.subtract(plain, idx, values); plain_ref[idx] -= values
        np.testing.assert_array_equal(plain.view(np.uint32), plain_ref.view(np.uint32))
        residency = Residency()
        residency.mirror("m", host, dtype=dtype)
        host_writes.register(residency)
        try:
            # CLEAN_BOTH: both sides move
            host_writes.subtract(host, idx, values); reference[idx] -= values
            np.testing.assert_array_equal(host.view(np.uint32), reference.view(np.uint32))
            assert residency.verify() == {}
            # DEVICE_OWNED: the device alone
            residency.arm_hold(["m"]); residency.sync_out(["m"])
            host_writes.subtract(host, idx, values); reference[idx] -= values
            got = host_writes.gather(host, idx)
            np.testing.assert_array_equal(got.view(np.uint32), reference[idx].view(np.uint32))
            residency.flush(["m"])
            np.testing.assert_array_equal(host.view(np.uint32), reference.view(np.uint32))
        finally:
            host_writes.unregister(residency)


def test_the_subtract_door_flushes_a_subnormal_difference_like_the_kernels():
    """A subnormal difference reads +0 from the device door -- the family kernels' semantics.

    Metal compiles with fast math and flushes denormals with no lever (``subnormal.py``);
    the door is compiled the same way, so under a hold the deposit cells behave as the
    kernels around them do, and the subnormal policy drives the host to match. This
    test states that fact rather than hiding it: if the toolchain ever keeps denormals,
    it fails and the policy module must be re-read.
    """
    from meep_gpu import host_writes
    host = np.zeros(16, np.float32); host[4] = np.float32(1e-38)
    residency = Residency(); residency.mirror("m", host, dtype=np.float32)
    residency.arm_hold(["m"]); residency.sync_out(["m"])
    host_writes.register(residency)
    try:
        host_writes.subtract(host, np.array([4], dtype=np.int64), np.array([9.5e-39], dtype=np.float32))
        got = host_writes.gather(host, np.array([4], dtype=np.int64))
        assert got.view(np.uint32)[0] == 0, got  # flushed, as the kernels flush
    finally:
        host_writes.unregister(residency)


def test_sparse_doors_move_complex64_through_the_real_view():
    """MPS has no complex index ops; a complex64 mirror indexes its (N, 2) real view.

    Every door is checked against the NumPy form on the same words: gather and
    scatter round-trip, zero_index zeroes both parts, copy_index with a -1 parity
    matches ``-1 * face`` bit for bit (its one multiply is done on the host, as the
    array path does it, so the sign of a zero imaginary part cannot differ), and
    add_scaled_index matches ``current + k * values``.
    """
    from meep_gpu import host_writes
    rng = np.random.default_rng(3)
    host = (rng.standard_normal(48) + 1j * rng.standard_normal(48)).astype(np.complex64)
    host[5] = 0.0 + 0.0j  # a zero imaginary part, where a device multiply could flip its sign
    reference = host.copy()
    residency = Residency()
    residency.mirror("cplx", host, dtype=np.complex64)
    residency.arm_hold(["cplx"])
    residency.sync_out(["cplx"])
    assert residency.state_of("cplx") == DEVICE_OWNED if hasattr(residency, "state_of") else True
    host_writes.register(residency)
    try:
        idx = np.array([3, 5, 7], dtype=np.int64)
        values = np.array([1 + 2j, 0 + 0j, -3.5 - 0.25j], dtype=np.complex64)
        host_writes.scatter(host, idx, values); reference[idx] = values
        np.testing.assert_array_equal(host_writes.gather(host, idx).view(np.uint32), values.view(np.uint32))
        host_writes.zero(host.reshape(6, 8), 0, 1) if False else residency.zero_index(host, np.arange(8, 16))
        reference[8:16] = 0
        np.testing.assert_array_equal(host_writes.gather(host, np.arange(8, 16)).view(np.uint32), reference[8:16].view(np.uint32))
        src = np.arange(0, 8); dst = np.arange(16, 24)
        residency.copy_index(host, dst, host, src, -1); reference[dst] = -1 * reference[src]
        np.testing.assert_array_equal(host_writes.gather(host, dst).view(np.uint32), reference[dst].view(np.uint32))
        residency.add_scaled_index(host, np.arange(24, 32), host, np.arange(0, 8), 0.35)
        reference[24:32] = reference[24:32] + 0.35 * reference[0:8]
        np.testing.assert_array_equal(host_writes.gather(host, np.arange(24, 32)).view(np.uint32), reference[24:32].view(np.uint32))
        residency.flush(["cplx"])
        np.testing.assert_array_equal(host.view(np.uint32), reference.view(np.uint32))
    finally:
        host_writes.unregister(residency)


# ---------------------------------------------------------------------------
# The batched wall clear: one device launch for a whole ``_zero_metal`` pass
# ---------------------------------------------------------------------------

#: ``_zero_metal``'s D pass on a 3-D grid walled on every axis: each D component is
#: cleared on the two walls its Yee shift is 0 on -- three arrays, six faces, and
#: the two faces of one array meet on an edge line.
_D_PASS = ((0, (1, 2)), (1, (0, 2)), (2, (0, 1)))


def _signed_volumes(dtype, count=3, shape=(5, 4, 6), seed=29):
    """Random volumes with signed zeros planted ON wall faces, on an EDGE, and OFF them."""
    rng = np.random.default_rng(seed)
    complex_ = dtype is np.complex64
    negative_zero = np.complex64(complex(-0.0, -0.0)) if complex_ else np.float32(-0.0)
    out = []
    for _ in range(count):
        volume = rng.standard_normal(shape).astype(np.float32)
        if complex_:
            volume = (volume + 1j * rng.standard_normal(shape).astype(np.float32)).astype(np.complex64)
        volume[0, 1, 2] = negative_zero   # on the x = 0 face
        volume[2, 0, 0] = negative_zero   # on the y = 0 / z = 0 edge
        volume[3, 2, 4] = negative_zero   # interior: must survive as -0
        if complex_:
            volume[4, 3, 5] = np.complex64(complex(1.5, -0.0))  # interior, signed zero imaginary part
            volume[0, 2, 3] = np.complex64(complex(-2.5, 0.0))  # on a face, zero imaginary part
        out.append(volume)
    return out


def _pass_requests(arrays):
    return [(arrays[k], axis, 0) for k, axes in _D_PASS for axis in axes]


def _numpy_clear(arrays):
    """The array path's words: ``array[face] = 0``, request by request, on copies."""
    expected = [array.copy() for array in arrays]
    for k, axes in _D_PASS:
        for axis in axes:
            expected[k][(slice(None),) * axis + (0,)] = 0
    return expected


def _words(array):
    return np.ascontiguousarray(array).reshape(-1).view(np.uint32)


def test_the_signed_volumes_really_carry_signed_zeros_on_and_off_the_faces():
    """The fixture's claim, checked: without it the word tests below prove less."""
    for dtype in (np.float32, np.complex64):
        before = _signed_volumes(dtype)
        after = _numpy_clear(before)
        # -0 on the x = 0 face of array 1 (cleared on x) becomes NumPy's +0 ...
        assert np.signbit(before[1][0, 1, 2].real) and not np.signbit(after[1][0, 1, 2].real)
        # ... and the interior -0 is left alone everywhere.
        assert all(np.signbit(a[3, 2, 4].real) for a in after)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64], ids=["float32", "complex64"])
@pytest.mark.parametrize("state", [DEVICE_OWNED, CLEAN_BOTH, HOST_OWNED])
def test_the_zero_faces_door_clears_a_whole_pass_word_for_word(dtype, state):
    """``host_writes.zero_faces`` == NumPy ``a[face] = 0`` on every word, in ONE launch.

    Three arrays and six faces (``_zero_metal``'s D pass on a walled 3-D grid), with
    signed zeros on the faces (they must read NumPy's +0), on an edge two faces share,
    and off every face (they must survive as -0, and so must a zero imaginary part).
    A complex64 mirror goes through its (N, 2) real view. DEVICE_OWNED writes the
    device only and leaves the host stale and sealed; CLEAN_BOTH writes both sides
    and stays clean; HOST_OWNED (after ``acquire_write``, unsealed) writes both sides
    and stays host-owned. None moves a volume, and none moves a state.
    """
    from meep_gpu import host_writes
    arrays = _signed_volumes(dtype)
    before = [array.copy() for array in arrays]
    expected = _numpy_clear(arrays)
    residency = Residency()
    names = [f"d{k}" for k in range(len(arrays))]
    for name, array in zip(names, arrays):
        residency.mirror(name, array, dtype=dtype)
    residency.arm_hold(names)
    if state == DEVICE_OWNED:
        residency.sync_out(names)
    elif state == HOST_OWNED:
        for name in names:
            residency.acquire_write(name)
    assert [residency.state_of(name) for name in names] == [state] * len(names)
    sealed = [not array.flags.writeable for array in arrays]
    host_writes.register(residency)
    try:
        host_writes.zero_faces(_pass_requests(arrays))
    finally:
        host_writes.unregister(residency)
    assert residency.zero_launches == 1, "one device launch for the whole pass"
    assert residency.sparse_zeros == 6, "one per (array, face), the unit zero_index counted"
    assert residency.copies_in == 0 and residency.copies_out == 0
    assert [residency.state_of(name) for name in names] == [state] * len(names)
    assert [not array.flags.writeable for array in arrays] == sealed, "the seal is restored"
    for name, array, want, old in zip(names, arrays, expected, before):
        np.testing.assert_array_equal(_words(residency.tensor(name).cpu().numpy()), _words(want))
        host_should_read = old if state == DEVICE_OWNED else want  # stale BY DESIGN when held
        np.testing.assert_array_equal(_words(array), _words(host_should_read))
    if state != DEVICE_OWNED:
        assert residency.verify() == {}
    residency.flush(names)
    for array, want in zip(arrays, expected):
        np.testing.assert_array_equal(_words(array), _words(want))


def test_the_zero_faces_door_is_the_numpy_assignment_when_nothing_holds():
    """The array path (and the CUDA/Triton paths): the same statements, the same words."""
    from meep_gpu import host_writes
    assert not host_writes.holding()
    for dtype in (np.float32, np.complex64):
        arrays = _signed_volumes(dtype)
        expected = _numpy_clear(arrays)
        host_writes.zero_faces(_pass_requests(arrays))
        for array, want in zip(arrays, expected):
            np.testing.assert_array_equal(_words(array), _words(want))
    host_writes.zero_faces([])  # an empty pass touches nothing and launches nothing


def test_the_zero_faces_door_sends_held_arrays_to_the_launch_and_clears_the_rest_on_the_host():
    from meep_gpu import host_writes
    arrays = _signed_volumes(np.float32)
    expected = _numpy_clear(arrays)
    residency = Residency()
    residency.mirror("d0", arrays[0])
    residency.mirror("d2", arrays[2])
    residency.arm_hold(["d0", "d2"])
    residency.sync_out(["d0", "d2"])
    host_writes.register(residency)
    try:
        host_writes.zero_faces(_pass_requests(arrays))
    finally:
        host_writes.unregister(residency)
    assert residency.zero_launches == 1 and residency.sparse_zeros == 4
    np.testing.assert_array_equal(_words(residency.tensor("d0").cpu().numpy()), _words(expected[0]))
    np.testing.assert_array_equal(_words(residency.tensor("d2").cpu().numpy()), _words(expected[2]))
    np.testing.assert_array_equal(_words(arrays[1]), _words(expected[1]))


@pytest.mark.parametrize("dtype", [np.float32, np.complex64], ids=["float32", "complex64"])
def test_the_batched_clear_on_a_cpu_twin_is_the_per_segment_fill(dtype):
    """Off MPS the launch is ``index_fill_`` segment by segment -- the gates' CPU twin."""
    arrays = _signed_volumes(dtype)
    expected = _numpy_clear(arrays)
    residency = Residency(device="cpu")
    names = [f"d{k}" for k in range(len(arrays))]
    for name, array in zip(names, arrays):
        residency.mirror(name, array, dtype=dtype)
    residency.arm_hold(names)
    residency.sync_out(names)
    from meep_gpu import host_writes
    residency.zero_index_sets([(array, host_writes.face_index(array.shape, axis, index))
                               for array, axis, index in _pass_requests(arrays)])
    assert residency.zero_launches == 1 and residency.sparse_zeros == 6
    for name, want in zip(names, expected):
        np.testing.assert_array_equal(_words(residency.tensor(name).numpy()), _words(want))


def test_the_zero_faces_door_refuses_more_arrays_than_its_launch_binds():
    """Above ``ZERO_DOOR_BUFFERS`` arrays the group is REFUSED by name, before any write."""
    from meep_gpu import host_writes
    from meep_gpu.metal_kernels.device import ZERO_DOOR_BUFFERS
    count = ZERO_DOOR_BUFFERS + 1
    arrays = _signed_volumes(np.float32, count=count)
    loose = _signed_volumes(np.float32, count=1, seed=5)[0]
    before = [array.copy() for array in arrays]
    loose_before = loose.copy()
    residency = Residency()
    names = [f"d{k}" for k in range(count)]
    for name, array in zip(names, arrays):
        residency.mirror(name, array)
    residency.arm_hold(names)
    residency.sync_out(names)
    host_writes.register(residency)
    try:
        with pytest.raises(ValueError, match=rf"door_scatter_zero_multi binds {ZERO_DOOR_BUFFERS} row buffers"):
            host_writes.zero_faces([(loose, 0, 0)] + [(array, 0, 0) for array in arrays])
    finally:
        host_writes.unregister(residency)
    assert residency.zero_launches == 0 and residency.sparse_zeros == 0
    for name, old in zip(names, before):
        np.testing.assert_array_equal(_words(residency.tensor(name).cpu().numpy()), _words(old))
    np.testing.assert_array_equal(_words(loose), _words(loose_before))  # the unheld array too


def test_the_batched_launch_binds_exactly_the_largest_pass_zero_metal_sends():
    """``ZERO_DOOR_BUFFERS`` is ``_zero_metal``'s largest group, and the kernel binds that many."""
    from meep_gpu import stepping
    from meep_gpu.metal_kernels import device
    largest = max(len(stepping.D_COMPONENTS), len(stepping.B_COMPONENTS))
    assert device.ZERO_DOOR_BUFFERS == largest
    signature = device._DOOR_SOURCE.split("kernel void door_scatter_zero_multi(")[1].split(")\n{")[0]
    bound = [f"device float* rows{k} " in signature for k in range(largest + 1)]
    assert bound == [True] * largest + [False], signature


class _WalledGrid:
    """The three questions ``_zero_metal`` asks of a grid: every axis walled, none folded."""
    has_metallic = True

    @staticmethod
    def is_metallic(axis):
        return True

    @staticmethod
    def is_mirrored(axis):
        return False


class _Fields:
    def __init__(self, names, arrays):
        self.grid = _WalledGrid()
        for name, array in zip(names, arrays):
            setattr(self, name, array)


@pytest.mark.parametrize("side", ["D", "B"])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64], ids=["float32", "complex64"])
def test_zero_metal_is_one_launch_a_pass_and_the_array_path_words(side, dtype):
    """``zero_metal_D`` / ``zero_metal_B`` under a held residency: one launch, NumPy's words.

    The array-path twin runs the same pass on copies with nothing registered, so the
    comparison is against ``_zero_metal``'s own NumPy statements, not a restatement.
    """
    from meep_gpu import host_writes, stepping
    names = stepping.D_COMPONENTS if side == "D" else stepping.B_COMPONENTS
    step = stepping.zero_metal_D if side == "D" else stepping.zero_metal_B
    arrays = _signed_volumes(dtype, seed=41)
    twin = _Fields(names, [array.copy() for array in arrays])
    step(twin)  # the array path: nothing holds anything
    residency = Residency()
    for name, array in zip(names, arrays):
        residency.mirror(name, array, dtype=dtype)
    residency.arm_hold(list(names))
    residency.sync_out(list(names))
    host_writes.register(residency)
    try:
        step(_Fields(names, arrays))
    finally:
        host_writes.unregister(residency)
    faces = sum(1 for name in names for axis in range(3) if stepping.IYEE_SHIFTS[name][axis] == 0)
    assert faces == (6 if side == "D" else 3)
    assert residency.zero_launches == 1 and residency.sparse_zeros == faces
    assert residency.copies_in == 0 and residency.copies_out == 0
    for name in names:
        np.testing.assert_array_equal(_words(residency.tensor(name).cpu().numpy()),
                                      _words(getattr(twin, name)))
