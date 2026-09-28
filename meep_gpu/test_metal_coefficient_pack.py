"""Array packing on Metal, as claims a laptop can check.

WHAT THIS SUITE OWNS. :mod:`meep_gpu.metal_kernels.coefficient_pack` puts several
read-only coefficient vectors in ONE Metal buffer and addresses them by element
offsets carried in the kernel's ``Params`` struct. That buys back buffer bindings on a
platform whose ceiling is 31 (``device.MAX_BUFFER_BINDINGS``), and it is the reason
the Dcyl m = 0 D->E pair — 31 pointers unpacked, one over — exists at all.

WHAT IT DELIBERATELY DOES NOT OWN. Whether the packed kernel computes the same
NUMBERS is the device gate's
(``parity/meep_gpu/gate_metal_cylindrical_real_fused_electric_pair.py``, legs
``pack_identity``, ``binding_ceiling`` and the eleven OFFSET mutations). No assertion
here duplicates that. What lives here is everything true about the packer WITHOUT a
device: the layout arithmetic, the refusals, the emitted MSL text, and the two
invariants a wrong pack would break silently.

THE ONE DESIGN RULE THIS SUITE PINS: a pack is READ-ONLY. There is no writer-side
API, ``packed_mirror`` registers ``constant=True``, and a family that wanted to fold a
device-written scratch volume into a pack would have to add the capability rather than
call an existing one. The cylindrical radial prefix is the live example and it keeps
its own binding for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from meep_gpu.metal_kernels import coefficient_pack as pack
from meep_gpu.metal_kernels.device import Residency

NAMES = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz")


def vectors(lengths=(4, 4, 1, 1, 3, 3)):
    return [np.arange(n, dtype=np.float32) + 10.0 * (index + 1)
            for index, n in enumerate(lengths)]


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------

def test_the_pack_is_the_vectors_end_to_end_and_the_offsets_are_the_cumulative_sums():
    columns = vectors()
    host, layout = pack.build_pack(np, NAMES, columns)
    assert layout.names == NAMES
    assert layout.lengths == tuple(v.size for v in columns)
    assert layout.offsets == (0, 4, 8, 9, 10, 13)
    assert layout.total == host.size == sum(v.size for v in columns)
    for name, column in zip(NAMES, columns):
        start = layout.offset(name)
        got = host[start:start + column.size]
        assert got.tobytes() == column.tobytes(), name


def test_the_packed_bytes_are_the_bytes_each_vector_would_have_been_mirrored_as():
    """The float32 cast happens ONCE, here, exactly as ``Residency.mirror`` casts.

    A pack that re-rounded a value would be a different absorber, silently — so the
    comparison is uint32 words against the same cast applied separately.
    """
    columns = [np.linspace(0.1, 0.9, n, dtype=np.float64) for n in (5, 5, 1, 1, 2, 2)]
    host, layout = pack.build_pack(np, NAMES, columns)
    for name, column in zip(NAMES, columns):
        start = layout.offset(name)
        separate = np.ascontiguousarray(column, dtype=np.float32).reshape(-1)
        got = host[start:start + separate.size]
        assert got.view(np.uint32).tolist() == separate.view(np.uint32).tolist(), name


def test_heterogeneous_lengths_are_the_normal_case_and_not_an_edge():
    """A Dcyl grid stores ONE phi cell, so two of the six vectors are length 1.

    That is the shipped shape rather than a corner: ``kms_y`` and ``sinv_y`` are
    single elements on every Dcyl plan, and a packer that assumed equal lengths would
    have worked on Cartesian grids and mis-addressed every cylindrical one.
    """
    _host, layout = pack.build_pack(np, NAMES, vectors((20, 20, 1, 1, 20, 20)))
    assert layout.lengths == (20, 20, 1, 1, 20, 20)
    assert layout.offsets == (0, 20, 40, 41, 42, 62)
    assert layout.total == 82


def test_a_zero_length_member_is_refused_by_name():
    """Two members with the same offset cannot be told apart by the kernel."""
    columns = vectors()
    columns[2] = np.zeros(0, dtype=np.float32)
    with pytest.raises(ValueError, match="is empty"):
        pack.build_pack(np, NAMES, columns)


def test_a_missing_vector_is_refused_before_any_allocation():
    columns = vectors()
    columns[1] = None
    with pytest.raises(ValueError, match="is None"):
        pack.build_pack(np, NAMES, columns)


def test_a_name_count_mismatch_is_refused():
    with pytest.raises(ValueError, match="one name per vector"):
        pack.build_pack(np, NAMES[:3], vectors())


def test_a_repeated_name_is_refused():
    with pytest.raises(ValueError, match="cannot repeat a name"):
        pack.PackLayout(("a", "a"), (0, 1), (1, 1))


def test_a_non_contiguous_layout_is_refused():
    """A gap would leave a hole nothing writes and a length nothing accounts for."""
    with pytest.raises(ValueError, match="not contiguous"):
        pack.PackLayout(("a", "b"), (0, 5), (1, 1))


# ---------------------------------------------------------------------------
# The residency seam
# ---------------------------------------------------------------------------

def test_rebuilding_the_same_pack_under_one_name_reuses_the_resident_host_array():
    """A plan built twice against one residency hands the packer two arrays.

    ``Residency.mirror`` refuses two different host arrays under one name — the
    registry's aliasing defence, and right — so the packer must go through
    ``Residency.host`` the way the cylindrical scratch does.
    """
    residency = Residency(device="cpu")
    first, _layout = pack.packed_mirror(residency, "pml_pack", np, NAMES, vectors())
    second, _layout = pack.packed_mirror(residency, "pml_pack", np, NAMES, vectors())
    assert first is second
    assert residency.host("pml_pack") is not None


def test_a_pack_rebuilt_with_DIFFERENT_coefficients_under_one_name_is_REFUSED():
    """The failure the reuse above would otherwise hide, and it is not a crash.

    Reusing a resident pack whose bytes describe another configuration's absorber
    steps the second plan with the first plan's PML — a smooth, plausible, wrong
    field. So the cached bytes are compared as uint32 words and a mismatch is refused
    by name.
    """
    residency = Residency(device="cpu")
    pack.packed_mirror(residency, "pml_pack", np, NAMES, vectors())
    other = vectors()
    other[0] = other[0] + 1.0
    with pytest.raises(ValueError, match="already resident with DIFFERENT bytes"):
        pack.packed_mirror(residency, "pml_pack", np, NAMES, other)


def test_a_pack_of_a_different_LENGTH_under_one_name_is_refused_too():
    residency = Residency(device="cpu")
    pack.packed_mirror(residency, "pml_pack", np, NAMES, vectors())
    with pytest.raises(ValueError, match="already resident with DIFFERENT bytes"):
        pack.packed_mirror(residency, "pml_pack", np, NAMES,
                           vectors((5, 4, 1, 1, 3, 3)))


def test_the_pack_is_registered_as_a_CONSTANT_mirror():
    """READ-ONLY BY CONSTRUCTION. Nothing on the device writes a pack, so nothing
    needs syncing out — and a family that folded a device-written scratch volume in
    would have to change this call rather than reuse it."""
    residency = Residency(device="cpu")
    pack.packed_mirror(residency, "pml_pack", np, NAMES, vectors())
    mirror = residency._mirrors["pml_pack"]
    assert mirror.constant is True
    assert np.dtype(mirror.dtype) == np.dtype(np.float32)


def test_the_module_exposes_no_writer_side_api():
    """The read-only rule is enforced by there being nothing to call, not by a
    comment: a pack is built, mirrored and read, and that is the whole surface."""
    import pathlib

    source = pathlib.Path(pack.__file__).read_text(encoding="utf-8")
    for forbidden in ("def write", "def store", "def update_pack", "constant=False"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# The emitted MSL
# ---------------------------------------------------------------------------

def test_the_params_fields_are_one_uint_per_vector_in_layout_order():
    text = pack.params_fields(NAMES)
    assert text.splitlines() == [f"    uint off_{name};" for name in NAMES]


def test_the_record_dtype_matches_the_struct_field_order():
    """The kernel decodes by OFFSET, so a reordered record reinterprets every field.

    Both sides are generated from the same tuple, and this is the assertion that keeps
    them so: a hand-written record whose order drifted would compile, launch and read
    the wrong six numbers.
    """
    fields = pack.record_dtype_fields(NAMES)
    assert [name for name, _kind in fields] == [f"off_{name}" for name in NAMES]
    assert {kind for _name, kind in fields} == {"<u4"}
    record = np.zeros(1, dtype=np.dtype([("n", "<u4")] + fields))
    # Every member is 4 bytes and MSL packs a struct of uints at 4-byte alignment, so
    # the NumPy record and the Metal struct have the same layout member for member.
    assert record.itemsize == 4 * (1 + len(NAMES))


def test_the_prologue_recreates_each_vector_under_its_own_name():
    """THE WHOLE POINT: the lifted bodies are not forked.

    After these lines ``kmx`` is a ``device const float*`` exactly as it was when it
    had its own ``[[buffer(n)]]``, so every certified line that indexes it is
    character for character what its emitter produced.
    """
    text = pack.prologue("cpml", NAMES)
    assert text.splitlines() == [
        f"    device const float* {name} = cpml + prm.off_{name};" for name in NAMES]


def test_the_prologue_takes_the_base_and_the_struct_by_name():
    text = pack.prologue("basebuf", NAMES[:1], struct="params", element="float2")
    assert text == "    device const float2* kmx = basebuf + params.off_kmx;"


def test_offsets_for_reads_back_the_layouts_own_numbers():
    _host, layout = pack.build_pack(np, NAMES, vectors())
    assert pack.offsets_for(layout) == dict(zip(layout.names, layout.offsets))
