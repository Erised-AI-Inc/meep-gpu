"""SEVERAL read-only coefficient VECTORS in ONE Metal buffer, addressed by offset.

WHY THIS EXISTS, and it is a platform fact rather than a taste. A Metal kernel may
bind at most 31 buffers (:data:`.device.MAX_BUFFER_BINDINGS`, measured on this
toolchain: attribute index 31 is the compile error ``'buffer' attribute parameter is
out of bounds``). Two shipped families already buy slots back by packing SCALARS into
one ``constant Params&`` — :mod:`.complex_conductive_pml` turned three ``float2&``
bindings into one to get 32 down to 30, and :mod:`.nonlinear_update_e` six chi scalars
into one, recording that six separate ones "would have been 34 bindings and would not
have compiled at all".

THIS MODULE IS THE ARRAY VERSION OF THAT MOVE AND IT IS A NEW STEP, not a repeat.
A scalar pack costs nothing because the struct exists anyway; an ARRAY pack has to
answer two questions a scalar pack never raises — where each vector starts, and who
may write the buffer — and this module answers both by construction:

* **the offsets are DATA, carried in the same ``Params`` struct the scalars ride in**,
  so a packed signature costs ONE pointer and N extra uints, and uints are free;
* **a pack is READ-ONLY, on the device and on the host.** Only vectors nothing on the
  device writes may go in one. That is not a convention this module hopes for: it
  binds ``constant=True`` and there is no writer-side API at all. A scratch volume the
  device writes — the cylindrical radial prefix is the live example — stays its OWN
  binding, because a wrong offset into a mixed pack would let a scan corrupt a PML
  coefficient vector silently and permanently (nothing re-uploads a constant mirror).

WHY IT COSTS NOTHING ON THIS BACKEND, and this is the fact that makes packing cheap
here and expensive elsewhere. Every Metal device buffer in this package is a
:class:`.device.Mirror` the package allocates itself — ``Residency.mirror`` flattens
the host array and copies it to MPS once, at plan build. There is no driver-owned
device allocation to sub-allocate out of and no per-step copy to price: the pack is
built once from arrays the plan already has in hand, and the launch path binds one
tensor instead of six. ``device.py``'s own docstring records that this residency
layer has no Triton counterpart, which is why the CuPy/Triton track never had to
answer the question.

THE BODY TEXT IS NOT FORKED. :func:`prologue` re-creates each vector's own NAME as a
``device const float*`` into the pack, once, before any lifted text runs — so a
certified body that reads ``kmx[i]`` goes on reading ``kmx[i]``, character for
character, and the lift stays a lift. That is the same discipline the fused pairs
already apply to the packed scalars, which unpack ``prm.nx`` into ``nx`` for exactly
this reason.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "PackLayout", "build_pack", "packed_mirror", "params_fields", "prologue",
    "record_dtype_fields",
]


class PackLayout:
    """Where each named vector starts inside one packed buffer, in ELEMENTS.

    Offsets are element counts and not bytes, because the kernel indexes
    ``pack + off`` on a typed ``device const float*`` — where the compiler applies the
    element stride — and because a byte offset would have to be re-derived against the
    element width at every use site, which is one more place to be wrong.
    """

    __slots__ = ("names", "offsets", "lengths", "total")

    def __init__(self, names: Sequence[str], offsets: Sequence[int],
                 lengths: Sequence[int]) -> None:
        self.names = tuple(str(name) for name in names)
        self.offsets = tuple(int(offset) for offset in offsets)
        self.lengths = tuple(int(length) for length in lengths)
        if not (len(self.names) == len(self.offsets) == len(self.lengths)):
            raise ValueError(
                f"a pack layout needs one offset and one length per name; got "
                f"{len(self.names)} names, {len(self.offsets)} offsets, "
                f"{len(self.lengths)} lengths")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"a pack layout cannot repeat a name: {self.names}")
        self.total = int(sum(self.lengths))
        running = 0
        for name, offset, length in zip(self.names, self.offsets, self.lengths):
            if offset != running:
                raise ValueError(
                    f"the pack layout is not contiguous at {name!r}: it starts at "
                    f"element {offset} where the vectors before it end at {running}. "
                    f"A gap would make the packed buffer longer than the sum of its "
                    f"parts and a hole in it would never be written")
            running += length

    def offset(self, name: str) -> int:
        return self.offsets[self.names.index(name)]

    def describe(self) -> str:
        parts = ", ".join(f"{name}@{offset}:{length}" for name, offset, length
                          in zip(self.names, self.offsets, self.lengths))
        return f"PackLayout({parts}, total={self.total})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.describe()


def build_pack(np: Any, names: Sequence[str], vectors: Sequence[Any],
               dtype: Any = None) -> Tuple[Any, PackLayout]:
    """Concatenate ``vectors`` into one contiguous host array plus its layout.

    Every vector is flattened and cast ONCE, here, to the element type the kernel
    binds — the same cast :meth:`.device.Residency.mirror` would apply to each
    separately, so the packed bytes are the separate bytes end to end and nothing
    about the arithmetic moves. The gate measures that claim rather than asserting it.
    """
    resolved = np.float32 if dtype is None else np.dtype(dtype).type
    names = tuple(str(name) for name in names)
    if len(names) != len(vectors):
        raise ValueError(
            f"build_pack needs one name per vector; got {len(names)} names and "
            f"{len(vectors)} vectors")
    flat: List[Any] = []
    lengths: List[int] = []
    offsets: List[int] = []
    running = 0
    for name, vector in zip(names, vectors):
        if vector is None:
            raise ValueError(f"the packed vector {name!r} is None")
        column = np.ascontiguousarray(vector, dtype=resolved).reshape(-1)
        if column.size == 0:
            raise ValueError(
                f"the packed vector {name!r} is empty; a zero-length member would "
                f"make two offsets equal and the kernel could not tell them apart")
        flat.append(column)
        lengths.append(int(column.size))
        offsets.append(running)
        running += int(column.size)
    return np.concatenate(flat), PackLayout(names, offsets, lengths)


def packed_mirror(residency: Any, mirror_name: str, np: Any,
                  names: Sequence[str], vectors: Sequence[Any],
                  dtype: Any = None) -> Tuple[Any, PackLayout]:
    """Register ONE constant device mirror holding ``vectors`` end to end.

    :meth:`.device.Residency.mirror` refuses a second, different host array under a
    name it already holds — that refusal is the registry's whole aliasing defence and
    is right — but a pack is built INSIDE the plan builder, so building the same plan
    twice against one residency hands it two arrays. :meth:`.device.Residency.host` is
    the seam that case already has (the cylindrical scratch uses it); this function
    takes the same route and adds the check that route needs: the cached pack must be
    BIT-IDENTICAL to the one this build produced, or the name is bound to different
    coefficients and reusing it would step the second plan with the first plan's
    absorber profile.
    """
    host, layout = build_pack(np, names, vectors, dtype)
    existing = residency.host(mirror_name)
    if existing is not None:
        cached = np.ascontiguousarray(existing).reshape(-1)
        fresh = np.ascontiguousarray(host).reshape(-1)
        if cached.shape != fresh.shape or int(np.count_nonzero(
                cached.view(np.uint32) != fresh.view(np.uint32))):
            raise ValueError(
                f"the coefficient pack {mirror_name!r} is already resident with "
                f"DIFFERENT bytes ({cached.size} elements against {fresh.size}); "
                f"reusing it would bind this plan to another configuration's "
                f"coefficient vectors, which is a smooth wrong absorber rather than "
                f"an error")
        host = existing
    return residency.mirror(mirror_name, host, constant=True, dtype=dtype), layout


def record_dtype_fields(names: Sequence[str]) -> List[Tuple[str, str]]:
    """The ``numpy`` structured-dtype fields the offsets ride in, in layout order."""
    return [(f"off_{name}", "<u4") for name in names]


def params_fields(names: Sequence[str], indent: str = "    ") -> str:
    """The offset members of the kernel's ``Params`` struct, one line per vector.

    ``uint`` and not ``ulong``: the offsets index a per-axis coefficient vector whose
    length is a grid extent, and this engine's dispatch is already sized by a ``uint``
    ``n_elem``. A pack that overflowed a uint would have overflowed the dispatch first.
    """
    return "\n".join(f"{indent}uint off_{name};" for name in names)


def prologue(base: str, names: Sequence[str], struct: str = "prm",
             element: str = "float", indent: str = "    ") -> str:
    """Re-create each packed vector under its OWN name, as a device pointer.

    This is the whole reason the lifted bodies need no edit: after these lines,
    ``kmx``, ``sinvx`` and the rest are ``device const float*`` exactly as they were
    when each had its own ``[[buffer(n)]]``, and every certified line that indexes
    them is character for character what its emitter produced.
    """
    return "\n".join(
        f"{indent}device const {element}* {name} = {base} + {struct}.off_{name};"
        for name in names)


def offsets_for(layout: PackLayout) -> Dict[str, int]:
    """``{name: element offset}`` — the values the plan writes into ``Params``."""
    return {name: offset for name, offset in zip(layout.names, layout.offsets)}
