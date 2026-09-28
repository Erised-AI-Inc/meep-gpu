"""Host READ barriers for held mirrors — the half the seal cannot cover.

THE ASYMMETRY THIS MODULE EXISTS FOR. Under a hold the device owns a volume between
launches, and there are two ways the host can be wrong about it:

* the host WRITES it — closed, universally and fail-closed, by
  :meth:`..device.Mirror.seal`: the array's write flag is down and the write raises
  ``ValueError: assignment destination is read-only`` at the statement that made it;
* the host READS it — **not closed by anything**. A read of a stale host array
  returns plausible numbers and raises nothing, which is the failure class
  ``coverage`` names at its own refusal: "a smooth, plausible, wrong field rather
  than an error".

A seal cannot help with the read, because NumPy has no read hook. The interception
has to happen at the attribute that hands the array out, and that is what this
module installs.

WHY A CLASS SWAP AND NOT AN EDIT TO ``fields.py``. ``Fields`` stores its primaries
as plain instance attributes and declares no ``__slots__``, and a ``property`` is a
DATA descriptor, so a property defined on a generated subclass outranks the instance
``__dict__`` and fires for both ``f.Dx`` and ``getattr(f, "Dx")`` while handing back
the identical array object. Swapping ``fields.__class__`` therefore installs the
barrier from inside ``metal_kernels`` without touching the engine — which matters
because ``fields.py`` is pinned by all three ``driver_dispatch`` records and
``metal_kernels`` is pinned by one.

WHAT IT DOES NOT REACH, stated because the gap is the risk. A property is an
attribute hook, so it covers attribute access and nothing else. It does NOT see:

* ``PolarizationState.P[component]`` — a dict subscript, not an attribute. That is
  :class:`BarrierDict`'s job, below;
* a walk over ``instance.__dict__`` — the end-to-end comparison's plain-object
  branch reads straight out of the mapping with no descriptor. Such a walk must
  ``flush`` first rather than rely on the barrier;
* a view a caller took BEFORE the hold was armed and reads later.

Each is a named, bounded hole rather than an assumed-covered one.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from .device import DEVICE_OWNED as _DEVICE_OWNED

#: Generated barrier classes, keyed by (base class, the held name tuple). Generating
#: one class per composition rather than per instance keeps ``type()`` calls off the
#: step loop and lets two identical compositions share a class.
_BARRIER_CLASSES: Dict[Tuple[Any, Tuple[str, ...]], Any] = {}

#: The attribute a barriered instance carries its pre-swap class on, so
#: :func:`remove_read_barrier` can put it back exactly.
_ORIGINAL_CLASS = "_meep_gpu_barrier_original_class"

#: And the residency it acquires through.
_BARRIER_RESIDENCY = "_meep_gpu_barrier_residency"


def _barrier_property(name: str) -> property:
    """One held name's data descriptor.

    The getter must never call ``getattr(self, name)`` — that is this descriptor and
    the recursion is infinite — so it reads the instance mapping directly. The
    acquire comes FIRST: if the device owns the volume the host copy is stale, and
    handing it out before syncing is the exact defect the barrier exists to stop.
    """

    def get(self: Any) -> Any:
        mapping = self.__dict__
        residency = mapping.get(_BARRIER_RESIDENCY)
        if residency is not None:
            # READ INTENT, and the write-intent form this replaced is worth recording
            # because it was measured wrong. A getter cannot tell a read from a
            # read-modify-write, so the previous version acquired for WRITE on
            # every fetch: correct, and it made every seam pass's fetch cost a
            # whole-volume round trip -- the 2026-09-22 publication ladder moved
            # ~14 volumes a step at 7M cells for a few hundred written cells and
            # lost to stock MEEP at every size. The seam passes now fetch RAW
            # (``host_writes.raw``) and move only the cells they touch, so the
            # getter is left to do the one thing only a getter can: make a genuine
            # host READ -- a monitor, ``get_E``, the end-to-end walk -- current.
            # It syncs out when the device owns the volume and leaves the array
            # SEALED, so an unwired in-place write still raises at its own line.
            current = mapping[name]
            mirror = residency._by_host_identity(current)
            if mirror is not None and mirror.state == _DEVICE_OWNED:
                residency.acquire_read(current)
            return current
            # THE FAST PATH IS THE COMMON PATH and it must not go through
            # ``acquire_read``. A held mirror is ``DEVICE_OWNED`` only between the
            # launch that wrote it and the first host read; every other read finds
            # it clean, and on a held row there are few host reads by construction
            # (``coverage.residency_reasons`` refuses the hold when a live host pass
            # touches the volume). Measured on this host: routing every read through
            # ``acquire_read`` -- which resolves the name, builds a tuple and loops
            # -- costs +341.6 ns per access against a plain attribute's 39.3 ns.
            # One dict lookup and one string compare instead brings it back to the
            # +120 ns the design was priced at, and the slow path is unchanged.
            mirror = residency._mirrors.get(name)
            if mirror is not None and mirror.state == _DEVICE_OWNED:
                residency.acquire_read(name)
        return mapping[name]

    def set_(self: Any, value: Any) -> None:
        # A REBIND, NOT A WRITE. ``driver.set_field`` and the weld families assign a
        # whole new array to the attribute; the mirror registered under this name
        # still shadows the OLD object, so the registry is told rather than left to
        # discover it. ``rebind`` marks the name HOST_OWNED, which is right: the new
        # array's bytes have never been on the device.
        residency = self.__dict__.get(_BARRIER_RESIDENCY)
        if residency is not None and value is not self.__dict__.get(name):
            # A ROTATION MUST NOT REBIND, and getting this wrong took out every weld
            # row in the first held campaign. The weld families do
            # ``setattr(self.fields, name, self.rotated[name])`` inside ``run``, and
            # the array they swap in is ALREADY a registered mirror under its own
            # name (``weld_scratch:Dx``). Rebinding ``Dx`` onto it leaves two mirror
            # names holding one host allocation with two different device tensors,
            # and the weld's very next ``tensor_for_host`` refuses it by name --
            # "one host allocation has multiple device mirrors; a live pointer
            # refresh cannot choose a producer safely".
            #
            # Nothing needs rebinding in that case anyway: ownership is keyed on
            # array IDENTITY (see ``Residency._targets``), so the state of both
            # arrays already follows them through any number of rotations. The
            # rebind exists only for a genuinely NEW array -- ``driver.set_field``
            # assigning an allocation the registry has never seen.
            if residency.tensor_for_host(value) is None:
                try:
                    residency.rebind(name, value)
                except (KeyError, ValueError):
                    # Not a mirrored name, or a deliberately different shape/width.
                    # The registry refuses rather than guesses; the launch-time
                    # bound-tensor check is what catches a plan later binding the
                    # stale tensor.
                    pass
        self.__dict__[name] = value

    def delete(self: Any) -> None:
        self.__dict__.pop(name, None)

    return property(get, set_, delete,
                    f"held mirror {name!r}: acquired from the device on read")


def barrier_class(base: Any, names: Tuple[str, ...]) -> Any:
    """The generated subclass of ``base`` carrying one property per held name."""
    key = (base, names)
    existing = _BARRIER_CLASSES.get(key)
    if existing is not None:
        return existing
    namespace = {name: _barrier_property(name) for name in names}
    namespace["__module__"] = __name__
    namespace["__doc__"] = (
        f"{base.__name__} with a residency read barrier over {len(names)} held "
        f"mirror(s). Generated by meep_gpu.metal_kernels.barrier; the instance's "
        f"original class is restored by remove_read_barrier.")
    generated = type(f"Barriered{base.__name__}", (base,), namespace)
    _BARRIER_CLASSES[key] = generated
    return generated


def install_read_barrier(instance: Any, residency: Any,
                         names: Iterable[str]) -> Tuple[str, ...]:
    """Swap ``instance.__class__`` so reads of ``names`` acquire first.

    Only names that are BOTH an attribute of the instance and a registered mirror
    whose host array IS that attribute are barriered — an attribute holding a
    different object than the mirror shadows would give a descriptor that acquires
    the wrong volume, which is worse than no descriptor. Returns what was installed.

    Idempotent and reversible: a second call on an already-barriered instance
    replaces the class rather than stacking, and :func:`remove_read_barrier` puts the
    original back. An instance that is never barriered keeps its plain class and
    pays nothing — measured +120 ns per access on a barriered attribute, and exactly
    zero on every unheld run.
    """
    if instance is None:
        return ()
    original = instance.__dict__.get(_ORIGINAL_CLASS) or type(instance)
    installed = []
    for name in names:
        if name not in instance.__dict__:
            continue
        mirror = residency._mirrors.get(name)
        if mirror is None or mirror.host is not instance.__dict__[name]:
            continue
        installed.append(name)
    if not installed:
        return ()
    ordered = tuple(sorted(installed))
    instance.__dict__[_ORIGINAL_CLASS] = original
    instance.__dict__[_BARRIER_RESIDENCY] = residency
    instance.__class__ = barrier_class(original, ordered)
    return ordered


def remove_read_barrier(instance: Any) -> bool:
    """Restore the pre-swap class. Returns whether anything was installed."""
    original = instance.__dict__.pop(_ORIGINAL_CLASS, None)
    instance.__dict__.pop(_BARRIER_RESIDENCY, None)
    if original is None:
        return False
    instance.__class__ = original
    return True


def barriered_names(instance: Any) -> Tuple[str, ...]:
    """The held names an instance's barrier covers, or ``()`` when it has none."""
    if instance.__dict__.get(_ORIGINAL_CLASS) is None:
        return ()
    return tuple(sorted(
        name for name, value in type(instance).__dict__.items()
        if isinstance(value, property)))


class BarrierDict(dict):
    """A dict whose subscript reads acquire the mirror the value shadows.

    ``PolarizationState`` keeps its polarization arrays in ``P`` and ``P_prev``,
    which are DICTS, and ``dispersion.subtract_into`` reads ``self.P[component]``
    every step on a dispersive row through the deposit-repair path. A property
    cannot intercept a subscript, and ``PolarizationState`` is a plain class that is
    not a ``Fields`` attribute anyway, so the class swap above reaches none of it.

    Covering those rows therefore needs this: a mapping that acquires on
    ``__getitem__``. It is substituted at plan time for the state's own dict and
    carries the same items, so identity of the VALUES is preserved — which matters
    because the ADE families resolve their operands by ``id(array)``.

    The rotation that family performs (``state.P[c] = scratch`` and friends, a
    closed 3-cycle over the pool) goes through ``__setitem__`` and is deliberately
    NOT treated as a host write: it rebinds which array a slot names, exactly as the
    weld families rebind an attribute, and the registry keys ownership on array
    identity so nothing needs to move. A ``sync_out`` on the new value would be
    wrong — it has not been written by the host.
    """

    __slots__ = ("_residency", "_names")

    def __init__(self, items: Any = (), residency: Any = None,
                 names: Optional[Dict[int, str]] = None) -> None:
        super().__init__(items)
        self._residency = residency
        #: id(host array) -> mirror name, so a subscript resolves without a scan.
        self._names = dict(names or {})

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        residency = self._residency
        if residency is not None:
            name = self._names.get(id(value))
            if name is not None:
                mirror = residency._mirrors.get(name)
                if mirror is not None and mirror.state == _DEVICE_OWNED:
                    residency.acquire_read(name)
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    # ONE SUBSCRIPT IS NOT THE ONLY WAY OUT OF A MAPPING, and overriding only
    # ``__getitem__`` is why seven dispersive rows still diverged after the dict
    # barrier was installed. The end-to-end state walk reads these mappings with
    # ``.items()``; ``dict.items`` is a C-level view that never calls
    # ``__getitem__``, so every array came out of it un-acquired and the comparison
    # read host words the device had moved past. The signature was unmistakable once
    # the differing arrays were read rather than guessed at: BOTH dispatched legs
    # differed from the array path on the same three ``P`` arrays, with one side
    # exactly zero on every differing word.
    #
    # So every accessor that hands an array out acquires. ``keys`` and ``__len__``
    # are deliberately NOT overridden: they expose no array.

    def _acquire(self, value: Any) -> Any:
        residency = self._residency
        if residency is not None:
            name = self._names.get(id(value))
            if name is not None:
                mirror = residency._mirrors.get(name)
                if mirror is not None and mirror.state == _DEVICE_OWNED:
                    residency.acquire_read(name)
        return value

    def items(self):  # noqa: ANN201 - mirrors dict.items
        return [(key, self._acquire(value)) for key, value in super().items()]

    def values(self):  # noqa: ANN201 - mirrors dict.values
        return [self._acquire(value) for value in super().values()]

    def pop(self, key: Any, *default: Any) -> Any:
        if key in self:
            self._acquire(super().__getitem__(key))
        return super().pop(key, *default)

    def copy(self) -> Dict[Any, Any]:
        """A PLAIN dict of acquired arrays — a copy must not carry a stale value."""
        return {key: value for key, value in self.items()}

    def plain(self) -> Dict[Any, Any]:
        """A plain dict with the same items, for putting the state back.

        Goes through :meth:`items`, so every array is acquired on the way out: a
        state handed back to the engine must not carry words the device has moved
        past.
        """
        return {key: value for key, value in self.items()}


#: The polarization attributes that are plain arrays rather than mappings.
#: ``_scratch`` is the ADE pool's third member -- ``ade_update_p`` stores into it on
#: every launch and rotates it through ``P``/``P_prev`` afterwards -- and it is
#: reached as ``state._scratch``, an attribute, so the dict barrier cannot see it
#: and neither can the ``Fields`` class swap, which is installed on a different
#: object. It needs the attribute barrier, on the state.
POLARIZATION_ARRAY_ATTRIBUTES: Tuple[str, ...] = ("_scratch",)


def install_dict_barrier(state: Any, residency: Any,
                         attributes: Iterable[str] = ("P", "P_prev")) -> Tuple[str, ...]:
    """Replace a polarization state's mapping attributes with barriered ones.

    Returns the attribute names replaced. Only mappings whose VALUES are registered
    mirrors are replaced, and the mirror names are resolved once here rather than
    per subscript, so the step-loop cost is one dict lookup on an integer key.
    """
    replaced = []
    by_id = {id(m.host): m.name for m in residency._mirrors.values()}
    for attribute in attributes:
        mapping = getattr(state, attribute, None)
        if not isinstance(mapping, dict) or isinstance(mapping, BarrierDict):
            continue
        # THE MAP MUST COVER THE WHOLE POOL, NOT THIS DICT'S CURRENT CONTENTS.
        # ``ade_update_p`` rotates a closed 3-cycle over
        # {P[c], P_prev[c], _scratch} after every launch, so the array sitting in
        # ``P['Ex']`` three steps from now is one that is in ``P_prev`` or
        # ``_scratch`` today. Keying the lookup on only the values present at
        # install time means a rotated-in array resolves to no mirror and is handed
        # out WITHOUT acquiring -- which is exactly how seven dispersive rows
        # diverged while every other class passed, and why the divergence was in
        # ``P`` and grew with step count rather than appearing at step one.
        #
        # Every registered mirror is in the map instead. An id that never appears in
        # this mapping simply never gets looked up, so the cost of the wider map is
        # a few dict entries and nothing on the hot path.
        if not any(id(v) in by_id for v in mapping.values()):
            continue
        setattr(state, attribute, BarrierDict(mapping, residency, by_id))
        replaced.append(attribute)
    # And the plain-array members of the same pool, through the attribute barrier.
    installed = install_read_barrier(state, residency,
                                     POLARIZATION_ARRAY_ATTRIBUTES)
    replaced.extend(installed)
    return tuple(replaced)


def remove_dict_barrier(state: Any,
                        attributes: Iterable[str] = ("P", "P_prev")) -> Tuple[str, ...]:
    """Put plain dicts back, preserving the current (possibly rotated) contents."""
    restored = []
    for attribute in attributes:
        mapping = getattr(state, attribute, None)
        if isinstance(mapping, BarrierDict):
            setattr(state, attribute, mapping.plain())
            restored.append(attribute)
    return tuple(restored)
