"""How the engine touches arrays an execution backend may be holding on a device.

WHAT THIS EXISTS FOR, and why it is not in a backend package. Most of the time the
engine owns its arrays outright. One backend breaks that -- the Metal path mirrors
each volume into a device tensor and, when it HOLDS a volume, the device keeps the
newer words between launches. A host write to the engine's array is then lost or
silently overwritten, and a host read returns stale bytes; neither raises on its
own. The backend seals held host arrays so a stray write raises, and routes every
legitimate host access through the functions here.

THE ENGINE MUST NOT IMPORT A BACKEND, so this is a registry of live residencies
rather than a reference to one. Empty, every function below is the plain NumPy
operation it names and costs one list check; the Metal dispatch layer registers a
residency when it arms a hold and unregisters it when it releases one.

WHY THE OPERATIONS ARE SPARSE, and the measurement that made them so. The first
held implementation acquired WHOLE volumes: a host write to eight source cells
copied the whole array down and, at the next launch, the whole array back up. On
a 3-D grid at 7M cells that is 28 MB a copy, ~14 copies a step, ~400 MB a step,
and the timing ladder of 2026-09-22 read 60 ms/step at that size. The seam
passes touch a few cells (the source points, the deposit-repair cells) or one
wall face (``_zero_metal``); moving those cells instead of the volume removes
that cost from the step.

THE THREE SHAPES OF ACCESS, and the rule for each:

* :func:`raw` fetches an attribute WITHOUT the read barrier, for a caller that will
  touch it only through :func:`gather`/:func:`scatter`/:func:`zero`. A barriered
  fetch would sync the whole volume before the sparse operation ever ran.
* :func:`gather` / :func:`scatter` move exactly the flat indices named, from or to
  whichever side is current.
* :func:`zero` fills one slab of a 3-D array with zero, on the device when the
  device owns it -- no transfer at all. :func:`zero_faces` does a whole wall-clear
  pass of them, one device launch per owner.
* :func:`acquire` is the whole-volume declaration, kept for a site that really does
  rewrite a volume (the bench's ``thaw``); it is not the fast path.

A site that writes a held array WITHOUT going through one of these raises
``ValueError: assignment destination is read-only`` at that statement, which is
the enumeration of what still needs wiring.
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Live residencies, held WEAKLY. A process holds one per dispatched driver, and
#: the timing bench lifts three legs per case over forty-odd cases without ever
#: releasing a hold; a strong registry would pin every one of those residencies
#: -- and every device tensor they own -- for the life of the process. A weak
#: reference lets a closed driver's residency go, and a dead entry is skipped.
_RESIDENCIES: List["weakref.ref"] = []

#: Flat C-order indices of one slab of a 3-D array, keyed by (shape, axis, index).
#: ``_zero_metal`` asks for the same faces every step.
_FACE_CACHE: Dict[Tuple[Tuple[int, ...], int, int], Any] = {}


def _live() -> List[Any]:
    """The residencies still alive, dropping dead references as it goes."""
    alive = []
    keep = []
    for ref in _RESIDENCIES:
        residency = ref()
        if residency is not None:
            alive.append(residency)
            keep.append(ref)
    _RESIDENCIES[:] = keep
    return alive


def register(residency: Any) -> None:
    """A backend has started holding arrays; route their host accesses here."""
    if residency not in _live():
        _RESIDENCIES.append(weakref.ref(residency))


def unregister(residency: Any) -> None:
    _RESIDENCIES[:] = [ref for ref in _RESIDENCIES
                       if ref() is not None and ref() is not residency]


def holding() -> bool:
    """Whether any backend is currently holding arrays."""
    return bool(_live())


def _owner(array: Any) -> Optional[Any]:
    """The residency holding ``array``, resolved by array IDENTITY, or None."""
    for residency in _live():
        if residency._by_host_identity(array) is not None:
            return residency
    return None


def raw(obj: Any, name: str, default: Any = None) -> Any:
    """``obj.<name>`` without the read barrier -- the instance's own object.

    The barrier is a property on a generated subclass; the array itself lives in
    the instance ``__dict__`` and this reads it from there. Use it ONLY when every
    subsequent touch goes through :func:`gather`/:func:`scatter`/:func:`zero`: a raw
    array read directly under a hold is stale, and nothing will say so.
    """
    mapping = getattr(obj, "__dict__", None)
    if mapping is not None and name in mapping:
        return mapping[name]
    return getattr(obj, name, default)


def raw_item(mapping: Any, key: Any) -> Any:
    """``mapping[key]`` without the dict barrier, same contract as :func:`raw`."""
    if isinstance(mapping, dict):
        return dict.__getitem__(mapping, key)
    return mapping[key]


def gather(array: Any, linear: Any) -> Any:
    """The values of ``array`` at flat indices ``linear``; a fresh array."""
    owner = _owner(array)
    if owner is None:
        return array.reshape(-1)[linear]
    return owner.gather(array, linear)


def scatter(array: Any, linear: Any, values: Any) -> None:
    """Write ``values`` into ``array`` at flat indices ``linear``."""
    owner = _owner(array)
    if owner is None:
        array.reshape(-1)[linear] = values
        return
    owner.scatter(array, linear, values)


def subtract(array: Any, linear: Any, values: Any) -> None:
    """``array.reshape(-1)[linear] -= values`` -- the difference formed where the words live."""
    owner = _owner(array)
    if owner is None:
        array.reshape(-1)[linear] -= values
        return
    owner.subtract_index(array, linear, values)


def index_module(index: Any) -> Any:
    """The array module that owns an index array: cupy for a device array, numpy otherwise.

    A source's point list lives where its fields live, so on the CuPy path the three
    index arrays are device arrays and NumPy refuses to read them implicitly (measured:
    every point-source case of the CUDA route preflight, 2026-09-23). The flat index
    is formed by the module that owns the points; a NumPy flat index into a CuPy
    array, and the reverse, are both accepted by the gather and scatter doors.
    """
    if type(index).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    import numpy as np  # noqa: PLC0415

    return np


def ravel_points(ix: Any, iy: Any, iz: Any, shape: Tuple[int, ...]) -> Any:
    """C-order flat indices of ``array[ix, iy, iz]`` for an array of ``shape``."""
    xp = index_module(ix)
    return xp.ravel_multi_index(
        (xp.asarray(ix), xp.asarray(iy), xp.asarray(iz)), tuple(shape)).reshape(-1)


def face_index(shape: Tuple[int, ...], axis: int, index: int) -> Any:
    """Flat C-order indices of the slab ``array[..., index, ...]`` along ``axis``."""
    import numpy as np  # noqa: PLC0415

    key = (tuple(int(n) for n in shape), int(axis), int(index))
    hit = _FACE_CACHE.get(key)
    if hit is None:
        selector = (slice(None),) * axis + (index,)
        hit = np.ascontiguousarray(
            np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)[selector]
        ).reshape(-1)
        _FACE_CACHE[key] = hit
    return hit


def zero(array: Any, axis: int, index: int) -> None:
    """``array[<slab index along axis>] = 0`` -- on the device if it owns the array."""
    owner = _owner(array)
    if owner is None:
        array[(slice(None),) * axis + (index,)] = 0
        return
    owner.zero_index(array, face_index(array.shape, axis, index))


def zero_faces(requests: Sequence[Tuple[Any, int, int]]) -> None:
    """:func:`zero` for every ``(array, axis, index)`` in ``requests`` -- one launch per owner.

    A whole wall-clear pass in one call (``stepping._zero_metal``). With nothing
    holding an array this is exactly :func:`zero`'s assignment, request by request
    in the order given, so the array path -- NumPy or CuPy -- writes the same words
    by the same statements it always did. The requests an owner holds are handed to
    it together (``Residency.zero_index_sets``), which clears them in ONE device
    launch: the per-face form paid a door call and a launch for each of the nine
    faces a ``pml_3d`` step clears. The held groups go first, so an owner that
    refuses its group (more arrays than its launch binds) raises before any array
    in the call is written; the unheld requests follow, still in the order given.
    """
    held: Dict[int, Tuple[Any, List[Tuple[Any, Any]]]] = {}
    plain: List[Tuple[Any, int, int]] = []
    owners: Dict[int, Any] = {}
    for array, axis, index in requests:
        if id(array) not in owners:
            owners[id(array)] = _owner(array)
        owner = owners[id(array)]
        if owner is None:
            plain.append((array, axis, index))
            continue
        group = held.get(id(owner))
        if group is None:
            group = held[id(owner)] = (owner, [])
        group[1].append((array, face_index(array.shape, axis, index)))
    for owner, pairs in held.values():
        owner.zero_index_sets(pairs)
    for array, axis, index in plain:
        array[(slice(None),) * axis + (index,)] = 0


#: Flat indices of a RANGE of slabs along one axis, keyed like the face cache.
_SPAN_CACHE: Dict[Tuple[Tuple[int, ...], int, int, int], Any] = {}


def span_index(shape: Tuple[int, ...], axis: int, start: int, stop: int) -> Any:
    """Flat C-order indices of ``array[<start:stop along axis>]``."""
    import numpy as np  # noqa: PLC0415

    key = (tuple(int(n) for n in shape), int(axis), int(start), int(stop))
    hit = _SPAN_CACHE.get(key)
    if hit is None:
        selector = (slice(None),) * axis + (slice(start, stop),)
        hit = np.ascontiguousarray(
            np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)[selector]
        ).reshape(-1)
        _SPAN_CACHE[key] = hit
    return hit


#: Flat indices of a 3-D slice box, keyed by (shape, (start, stop) per axis).
_BOX_CACHE: Dict[Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]], Any] = {}


def box_index(shape: Tuple[int, ...], slices: Tuple[slice, ...]) -> Tuple[Any, Tuple[int, ...]]:
    """Flat C-order indices of ``array[slices]`` and the box's shape.

    A monitor's region (plus the plane the Yee average consumes) is a box, and on a
    3-D grid a flux plane is a slab of tens of thousands of cells against millions
    in the volume. Reading it through the barrier synced the whole volume, per
    component, per undecimated step -- six volumes a step for one flux monitor.
    """
    bounds = tuple(s_.indices(n)[:2] for s_, n in zip(slices, shape))
    key = (tuple(int(n) for n in shape), tuple((int(a), int(b)) for a, b in bounds))
    hit = _BOX_CACHE.get(key)
    if hit is None:
        import numpy as np  # noqa: PLC0415

        selector = tuple(slice(a, b) for a, b in bounds)
        sub = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)[selector]
        hit = (np.ascontiguousarray(sub).reshape(-1), tuple(int(n) for n in sub.shape))
        _BOX_CACHE[key] = hit
    return hit


def gather_box(array: Any, slices: Tuple[slice, ...]) -> Any:
    """``array[slices]`` as a fresh box, from the device when it owns the array."""
    owner = _owner(array)
    if owner is None:
        return array[tuple(slices)]
    linear, shape = box_index(array.shape, slices)
    return owner.gather(array, linear).reshape(shape)


def zero_rows(array: Any, axis: int, rows: slice) -> None:
    """``array[<rows along axis>] = 0`` -- the cylindrical near-axis constraint's shape."""
    owner = _owner(array)
    if owner is None:
        array[(slice(None),) * axis + (rows,)] = 0
        return
    start, stop, _ = rows.indices(array.shape[axis])
    owner.zero_index(array, span_index(array.shape, axis, start, stop))


def copy_face(destination: Any, axis: int, dst_index: int, source: Any, src_index: int,
              scale: Any = 1) -> None:
    """``destination[face dst] = scale * source[face src]`` -- device-side under a hold.

    The mirror-fold fills: the ghost plane at one end of an axis is the image of a
    stored plane, times a parity of +1 or -1. Both planes are on the device under a
    hold, so nothing crosses the bus.
    """
    owner = _owner(destination)
    dst_index = dst_index % destination.shape[axis]
    src_index = src_index % source.shape[axis]
    if owner is None and _owner(source) is None:
        destination[(slice(None),) * axis + (dst_index,)] = (
            scale * source[(slice(None),) * axis + (src_index,)])
        return
    (owner or _owner(source)).copy_index(
        destination, face_index(destination.shape, axis, dst_index),
        source, face_index(source.shape, axis, src_index), scale)


def add_scaled_face(destination: Any, axis: int, dst_index: int, source: Any,
                    src_index: int, scale: Any) -> None:
    """``destination[face dst] += scale * source[face src]``."""
    owner = _owner(destination) or _owner(source)
    if owner is None:
        destination[(slice(None),) * axis + (dst_index,)] += (
            scale * source[(slice(None),) * axis + (src_index,)])
        return
    owner.add_scaled_index(destination, face_index(destination.shape, axis, dst_index),
                           source, face_index(source.shape, axis, src_index), scale)


def acquire(array: Any) -> Any:
    """Declare a WHOLE-volume host write of ``array``; returns it for chaining.

    Not the fast path. It exists for the one site that genuinely rewrites a volume
    on the host -- the timing harness restoring a frozen state -- and for any site
    not yet given a sparse form, where correctness is worth a whole copy.
    """
    owner = _owner(array)
    if owner is not None:
        owner.write_hook(array)
    return array
