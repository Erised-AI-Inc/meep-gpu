"""The host READ barrier: the half the seal cannot cover.

The seal makes a host WRITE to a device-owned mirror raise. A host READ of one
returns stale numbers and raises nothing, and NumPy offers no read hook — so the
interception has to happen at whatever hands the array out. Two shapes, because the
engine has two:

* attribute access on ``Fields`` -> a ``property`` on a generated subclass, which
  outranks the instance ``__dict__`` because a property is a DATA descriptor;
* a dict subscript on ``PolarizationState.P`` / ``P_prev`` -> a dict subclass,
  because no descriptor can intercept ``self.P[component]``.

Each test below pins either the mechanism or one of its named gaps. The gaps are
tested too: a barrier whose limits are undocumented reads as total coverage.
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

from meep_gpu.metal_kernels import barrier as barrier_module  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    CLEAN_BOTH,
    DEVICE_OWNED,
    HOST_OWNED,
    Residency,
)


class _Fields:
    """A stand-in with the shape that matters: plain attributes, no __slots__."""


def _case(cells: int = 32):
    residency = Residency()
    fields = _Fields()
    fields.Dx = np.zeros(cells, dtype=np.float32)
    fields.Dy = np.ones(cells, dtype=np.float32)
    residency.mirror("Dx", fields.Dx)
    residency.mirror("Dy", fields.Dy)
    return residency, fields


# ---------------------------------------------------------------------------
# The attribute barrier
# ---------------------------------------------------------------------------

def test_a_property_on_the_generated_subclass_outranks_the_instance_dict():
    """The mechanism the whole design rests on, pinned rather than assumed.

    If a property did NOT outrank the instance ``__dict__``, the class swap would
    install descriptors that never fire and the barrier would be decorative.
    """
    residency, fields = _case()
    installed = barrier_module.install_read_barrier(fields, residency, ("Dx", "Dy"))
    assert installed == ("Dx", "Dy")
    assert type(fields).__name__ == "Barriered_Fields"
    assert isinstance(type(fields).__dict__["Dx"], property)
    assert "Dx" in fields.__dict__, "the array must still live in the instance dict"


def test_the_barrier_hands_back_the_identical_array_object():
    """A barrier that copied would break every plan that resolves by ``id(array)``."""
    residency, fields = _case()
    original = fields.Dx
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    assert fields.Dx is original
    assert getattr(fields, "Dx") is original


def test_fetching_a_device_owned_mirror_syncs_it_out_and_leaves_it_sealed():
    """The defect the barrier exists to stop: a plausible, stale read.

    READ intent. A fetch makes the host current and returns the identical array,
    still SEALED. The write-intent form this replaced -- unseal on every fetch --
    was correct and was measured to cost a whole-volume round trip per seam
    pass, ~14 volumes a step at 7M cells (the 2026-09-22 ladder). The seam
    passes now fetch raw and move only their cells; the getter serves genuine
    readers, and an unwired in-place write still raises at its own line.
    """
    residency, fields = _case()
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    residency.arm_hold(["Dx"])
    residency.tensor("Dx").copy_(torch.from_numpy(np.full(32, 5.0, dtype=np.float32)))
    residency.sync_out(["Dx"])
    assert residency.state_of("Dx") == DEVICE_OWNED
    got = fields.Dx                                   # the barrier fires here
    assert residency.state_of("Dx") == CLEAN_BOTH
    assert not got.flags.writeable, "a read leaves the seal down"
    np.testing.assert_array_equal(got, np.full(32, 5.0, dtype=np.float32))


def test_an_unwired_in_place_write_through_a_fetched_attribute_raises():
    """``_zero_metal``'s old shape, on a held mirror, without the sparse door.

    This is the fail-closed contract: a site that writes a held array without
    going through ``host_writes`` is enumerated by its own traceback, never by a
    wrong field. The three seam sites that used to look like this now go
    through ``host_writes.zero`` / ``scatter``.
    """
    residency, fields = _case()
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    residency.arm_hold(["Dx"])
    residency.sync_out(["Dx"])                        # device owns it
    with pytest.raises(ValueError, match="read-only"):
        fields.Dx[0] = 9.0


def test_the_barrier_covers_getattr_as_well_as_dotted_access():
    residency, fields = _case()
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    residency.arm_hold(["Dx"])
    residency.tensor("Dx").copy_(torch.from_numpy(np.full(32, 7.0, dtype=np.float32)))
    residency.sync_out(["Dx"])
    np.testing.assert_array_equal(getattr(fields, "Dx"),
                                  np.full(32, 7.0, dtype=np.float32))
    assert residency.state_of("Dx") == CLEAN_BOTH


def test_assigning_the_attribute_rebinds_the_mirror_rather_than_stranding_it():
    """``set_field`` and the weld families assign a whole new array to the name."""
    residency, fields = _case()
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    replacement = np.full(32, 3.0, dtype=np.float32)
    fields.Dx = replacement
    assert fields.Dx is replacement
    assert residency._mirrors["Dx"].host is replacement
    assert residency.state_of("Dx") == HOST_OWNED, (
        "the new array's bytes have never been on the device")


def test_a_name_whose_attribute_is_not_the_mirrored_array_is_not_barriered():
    """A descriptor that acquired the WRONG volume is worse than no descriptor."""
    residency, fields = _case()
    fields.Dx = np.zeros(32, dtype=np.float32)        # rebound before arming
    installed = barrier_module.install_read_barrier(fields, residency, ("Dx", "Dy"))
    assert installed == ("Dy",)


def test_the_barrier_is_reversible_and_idempotent():
    residency, fields = _case()
    plain = type(fields)
    barrier_module.install_read_barrier(fields, residency, ("Dx",))
    barrier_module.install_read_barrier(fields, residency, ("Dx", "Dy"))
    assert barrier_module.barriered_names(fields) == ("Dx", "Dy")
    assert barrier_module.remove_read_barrier(fields)
    assert type(fields) is plain
    assert not barrier_module.remove_read_barrier(fields)


def test_an_unbarriered_instance_keeps_its_plain_class_and_pays_nothing():
    """Every non-held run must be byte-for-byte the path that was certified."""
    residency, fields = _case()
    plain = type(fields)
    assert barrier_module.install_read_barrier(fields, residency, ()) == ()
    assert type(fields) is plain
    assert barrier_module.barriered_names(fields) == ()


# ---------------------------------------------------------------------------
# The dict barrier — R9, the dispersive rows
# ---------------------------------------------------------------------------

def _polarization_case():
    class _State:
        pass

    residency = Residency()
    state = _State()
    p = np.zeros(16, dtype=np.float32)
    p_prev = np.ones(16, dtype=np.float32)
    state.P = {"Ez": p}
    state.P_prev = {"Ez": p_prev}
    residency.mirror("ade:0:P:Ez", p)
    residency.mirror("ade:0:P_prev:Ez", p_prev)
    return residency, state, p, p_prev


def test_a_property_cannot_intercept_a_dict_subscript_which_is_why_this_exists():
    """The premise of R9, demonstrated rather than asserted."""
    residency, state, _, _ = _polarization_case()
    barrier_module.install_read_barrier(state, residency, ("P",))
    # ``P`` is an attribute holding a dict, not a mirrored array, so nothing is
    # barriered — and even if it were, ``state.P["Ez"]`` goes through the dict.
    assert barrier_module.barriered_names(state) == ()


def test_the_dict_barrier_acquires_on_subscript():
    residency, state, p, _ = _polarization_case()
    replaced = barrier_module.install_dict_barrier(state, residency)
    assert replaced == ("P", "P_prev")
    residency.arm_hold(["ade:0:P:Ez"])
    residency.tensor("ade:0:P:Ez").copy_(
        torch.from_numpy(np.full(16, 2.0, dtype=np.float32)))
    residency.sync_out(["ade:0:P:Ez"])
    assert residency.state_of("ade:0:P:Ez") == DEVICE_OWNED
    got = state.P["Ez"]                                # the barrier fires here
    assert residency.state_of("ade:0:P:Ez") == CLEAN_BOTH
    assert got is p
    np.testing.assert_array_equal(got, np.full(16, 2.0, dtype=np.float32))


def test_the_rotation_through_setitem_is_a_rebind_not_a_host_write():
    """The ADE pool rotates a closed 3-cycle every launch.

    Treating that as a host write would sync out an array the host never wrote, and
    would do it on every launch of every dispersive row. Ownership is keyed on array
    identity, so a slot changing which array it names moves nothing.
    """
    residency, state, p, p_prev = _polarization_case()
    barrier_module.install_dict_barrier(state, residency)
    residency.arm_hold(residency.names)
    residency.sync_out(residency.names)
    before = (residency.copies_in, residency.copies_out)
    state.P["Ez"] = p_prev                             # the rotation
    assert (residency.copies_in, residency.copies_out) == before
    assert residency.state_of("ade:0:P_prev:Ez") == DEVICE_OWNED


def test_the_dict_barrier_preserves_value_identity_and_is_reversible():
    """The ADE families resolve their operands by ``id(array)``."""
    residency, state, p, p_prev = _polarization_case()
    barrier_module.install_dict_barrier(state, residency)
    assert state.P["Ez"] is p
    assert isinstance(state.P, barrier_module.BarrierDict)
    assert barrier_module.remove_dict_barrier(state) == ("P", "P_prev")
    assert type(state.P) is dict
    assert state.P["Ez"] is p


def test_a_mapping_holding_no_mirrored_value_is_left_alone():
    residency, state, _, _ = _polarization_case()
    state.other = {"Ez": np.zeros(16, dtype=np.float32)}
    assert barrier_module.install_dict_barrier(state, residency, ("other",)) == ()
    assert type(state.other) is dict
