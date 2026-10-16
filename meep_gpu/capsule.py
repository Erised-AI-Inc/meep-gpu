"""Lifted-state capsules: a lifted driver saved so that it loads back with no MEEP.

A capsule is a versioned, compressed bundle that rebuilds an :class:`~.driver.FdtdDriver`
word for word: every float32 and complex64 word of the fields, the PML auxiliaries and
coefficient tables, the material arrays, the source deposition tables and every scalar
reachable from the driver. It is the test-support object the MEEP-identity tiers are
built on: a lift is made once where MEEP is installed, and the capsule is then stepped
on any host, against stored MEEP digests, without MEEP.

WHAT IS CARRIED. Two things, both needed:

* The CONSTRUCTION LOG: the :class:`FdtdDriver` constructor call and, in order, every
  configuration call the lift made on the driver (:data:`CONSTRUCTION_CALLS`), with
  their arguments. :func:`recording_construction` records them while a lift (or a
  hand-built driver) runs. Arguments are snapshotted when the call is made.
* The STATE: the whole object graph below the driver, with every array stored as its
  bytes and every scalar as its IEEE-754 bit pattern. Arrays reached under two paths
  are stored once, and the identity is restored.

THE LOADER replays the log through the driver's own constructor and mutators, so the
classes, the validation and every object the code builds are this tree's, then checks
that the rebuilt object graph has the stored one's shape -- the same classes, the same
attribute names, the same array shapes, dtypes and aliasing -- and refuses by name where
it does not. It then writes every stored array's bytes into the rebuilt array and every
stored scalar over the rebuilt one. The loaded state is therefore the stored state, not
a recomputation of it: a coefficient a host's libm would round differently, or a source
amplitude a caller's function produced, comes back as the minting host stored it.
:attr:`Capsule.load_report` counts the arrays and scalars whose rebuilt value differed
from the stored one before the paste. That count is reported, not judged.

FLOATS ARE BIT PATTERNS. Every float and complex scalar is written as the hexadecimal
of its IEEE-754 bytes and read back with :func:`struct.unpack`, so no decimal is parsed:
a decimal parsed under a flushing FPU loses its subnormals.

CALLABLES. A caller's function that crosses into the driver (an ``amp_func``, a custom
source's ``src_func``) is wrapped during recording by a proxy that keeps every call it
answers. The capsule stores that table, and the loaded driver gets a stand-in that
answers exactly those calls and refuses any other by name. A source time function is
called while stepping, so its calls are not in the capsule; a run that needs them takes
an injected currents table (below).

INJECTED CURRENTS. :func:`recording_currents` records, while a driver steps, every value
each source's time function returns (``current(t, dt)`` and ``dipole(t)`` on the object
that ``inject`` calls them on), and :func:`install_currents` makes another driver's
sources return exactly those values. A lifted run forms its currents through MEEP's own
``src_time`` (``sources._MeepSrcTimeBinding``), which no capsule can carry: such a
capsule steps only with a currents table. It is a test double at the source-time boundary,
not a driver option: the source classes, their ``inject`` and every other method are
untouched, so the dispatch planner's source checks read the same objects.

WHAT IS REFUSED, BY NAME. When writing: a driver not built under
:func:`recording_construction`; any monitor (DFT, flux, energy, force, near-to-far),
which this schema does not carry; a closed driver; synchronized magnetic fields; an
object of a class from outside this package; and a callable that was not recorded or
whose calls cannot be keyed exactly. When reading and loading: a capsule of any other
schema or version, a member the document does not list or whose digest differs from the
recorded one, a rebuilt object graph of another shape, and a custom waveform, or a
source whose currents came from MEEP's own ``src_time``, loaded without a currents
table.

THE CHUNK DIVISION. The lift installs MEEP's division with ``set_chunk_layout``, a
construction call: the capsule carries the :class:`~.chunk_map.ChunkLayout` it was given
(boxes, per-chunk PML flags, axis frames, faces and membership lines), the loader replays
it, and the loaded driver steps the lifted simulation's own division. When the minting
call also passes the MEEP simulation, ``sim.structure.get_chunk_volumes()`` (boxes in
MEEP's doubled lattice coordinates), the chunk owners and the user volume are recorded
beside it as provenance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import hashlib
import inspect
import io
import json
import platform
import struct
import sys
import types
import weakref
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy

from .backends import to_numpy
from .code_identity import code_digest_of_path
from .driver import FdtdDriver
from .sources import SourceEnvelope

__all__ = [
    "CONSTRUCTION_CALLS",
    "Capsule",
    "CapsuleInfo",
    "CapsuleRefusal",
    "ConstructionRecorder",
    "CurrentsRecorder",
    "CurrentsTable",
    "SCHEMA",
    "SCHEMA_VERSION",
    "install_currents",
    "load_capsule",
    "read_capsule",
    "recording_construction",
    "recording_currents",
    "state_arrays",
    "state_digest",
    "write_capsule",
]

#: The capsule schema this tree writes and the only one it reads.
SCHEMA = "meep_gpu.capsule"
SCHEMA_VERSION = 1
#: The injected-currents table schema.
CURRENTS_SCHEMA = "meep_gpu.capsule_currents"
CURRENTS_SCHEMA_VERSION = 1

#: The driver calls a lift configures a driver with, recorded in order and replayed by
#: the loader. A call made from inside one of these (a mutator delegating to another)
#: is the outer call's business and is not recorded.
CONSTRUCTION_CALLS: Tuple[str, ...] = (
    "set_epsilon", "set_epsilon_components", "set_epsilon_smoothed", "add_susceptibility",
    "set_conductivity", "set_b_conductivity", "set_absorber", "set_chi2", "set_chi3",
    "setup_pml", "add_source", "set_chunk_layout", "set_absent_components",
)
#: Monitor calls: recorded only so that the writer can refuse a driver that made one.
MONITOR_CALLS: Tuple[str, ...] = (
    "add_dft_monitor", "add_yee_region_dft", "add_energy_monitor", "add_force_monitor",
    "add_flux_monitor",
)
#: Attributes ``lift_simulation`` sets on the driver directly rather than through a call.
#: ``chi1inv_read`` is the lift's report of how it read MEEP's permittivity (the read path,
#: the build stamp, per component the points copied and read one at a time, and the seconds
#: each took); it travels with the capsule as ``sigma_lift`` does.
LIFT_ATTRIBUTES: Tuple[str, ...] = ("lift_record", "sigma_lift", "migrated_monitors", "chi1inv_read")
#: Attribute names that are not state, on every object: the array module and backend
#: (the loader's choice), uninitialised scratch buffers, the dispatch plan, the
#: readback's memos (the point stencils and the B/H alias rule), the memos of the
#: chunk division (the driver's one-rank default and kernel record, ``ChunkLayout``'s
#: derived lines, the region plan, the fields' copy of the lifted division and their
#: off-diagonal presence masks) and of the sources' merged ``src_vol`` tables (the
#: driver's binding key and each source's bound entries, ``sources.bind_src_vols``, and
#: the request geometry they are cut from, ``sources._source_geometry``, which the
#: replayed construction builds again), all rebuilt on demand; and a lifted source's MEEP
#: ``src_time`` (``_meep_src_time``), which is MEEP's object and cannot leave the
#: process. The writer lists the sources that held
#: one (``provenance["source_times_from_meep"]``), and the loader refuses to build such
#: a capsule without a currents table.
NOT_CARRIED = frozenset({
    "xp", "gpu", "scratch", "_scratch", "_fmp_scratch", "_fmp_scratch_by_component",
    "_fast_path", "_fast_path_stale", "_fast_path_record", "_point_stencils", "_h_alias_cache",
    "_default_chunk_layout", "_chunk_record", "_cache", "_region_plan",
    "_installed_chunk_layout", "_offdiag_presence",
    "_src_vol_key", "_src_vol", "_src_geometry", "_meep_src_time",
})
#: Driver lists that must be empty: this schema carries no monitor.
MONITOR_ATTRIBUTES: Tuple[str, ...] = (
    "_dft_monitors", "_flux_monitors", "_pending_monitor_regions",
    "_automatic_decimation_monitors",
)
#: The modules whose code produced the lifted state, recorded by digest.
LIFT_MODULES: Tuple[str, ...] = (
    "absorber", "amp_interpolation", "capsule", "deposit_repair", "dispersion", "driver",
    "fields", "from_meep", "gaussian_beam", "grid", "host_writes", "part_products", "pml", "readback",
    "sigma_lookup", "sigma_recovery", "smoothing", "sources", "stepping", "withdraw_hoist",
)
_TIME_FUNCTIONS: Tuple[str, ...] = ("current", "dipole")
_JSON_MEMBER = "capsule.json"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class CapsuleRefusal(ValueError):
    """A capsule cannot be written, read or loaded; the message names why."""


# ---------------------------------------------------------------------------------
# Scalars as bit patterns
# ---------------------------------------------------------------------------------

def _f64_hex(value: float) -> str:  # IEEE-754 bytes of a double, little-endian, as hex.
    return struct.pack("<d", value).hex()


def _f64_from_hex(text: str) -> float:  # The inverse: no decimal is ever parsed.
    return struct.unpack("<d", bytes.fromhex(text))[0]


def _qualified(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_class(name: str) -> type:
    module_name, _, qualname = name.partition(":")
    if not module_name.split(".")[0] == "meep_gpu":
        raise CapsuleRefusal(f"the capsule names class {name!r}, which is not this package's")
    module = sys.modules.get(module_name)
    if module is None:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise CapsuleRefusal(f"the capsule names class {name!r}, which this tree does not define")
    if not isinstance(target, type):
        raise CapsuleRefusal(f"the capsule names {name!r}, which is not a class in this tree")
    return target


def _is_package_object(value: Any) -> bool:
    cls = type(value)
    return (cls.__module__.split(".")[0] == "meep_gpu" and not isinstance(value, type)
            and not isinstance(value, (types.FunctionType, types.MethodType,
                                       types.BuiltinFunctionType, functools.partial)))


def _is_array(value: Any) -> bool:
    if isinstance(value, numpy.ndarray):
        return True
    return type(value).__module__.split(".")[0] == "cupy" and hasattr(value, "dtype") \
        and hasattr(value, "shape")


def _attributes(obj: Any) -> List[Tuple[str, Any]]:
    """The carried attributes of one package object, by name, from the instance itself.

    Read from ``__dict__`` (and ``__slots__``) rather than through ``getattr``, so a
    property or a residency barrier is never consulted, and taken whole: a dataclass's
    declared fields and anything ``__post_init__`` set beside them alike. A dataclass
    field declared ``init=False`` with a plain default lives on the class until the
    first assignment (``VolumeSource._applied_dipole`` before the first injection), so
    such a field is read from the class when the instance has not set it: the attribute
    set of an object does not depend on how far it has been stepped.
    """
    found: Dict[str, Any] = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for declared in dataclasses.fields(obj):
            default = next((cls.__dict__[declared.name] for cls in type(obj).__mro__
                            if declared.name in cls.__dict__), dataclasses.MISSING)
            if default is not dataclasses.MISSING and not hasattr(default, "__get__"):
                found[declared.name] = default
    namespace = getattr(obj, "__dict__", None)
    if isinstance(namespace, dict):
        found.update(namespace)
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()) or ():
            if name in ("__dict__", "__weakref__") or name in found:
                continue
            if hasattr(obj, name):
                found[name] = getattr(obj, name)
    return [(name, found[name]) for name in _attribute_order(found)
            if name not in NOT_CARRIED and not name.startswith("__")]


def _attribute_order(names: Any) -> List[str]:
    """The one visiting order of an object's attributes, for the writer and both readers.

    Public names first, then private ones, each sorted: an object reached under several
    paths is first reached under its public one (``driver.pml``, not a private cache
    that also holds it), which is the path the records name. The writer numbers
    objects in this order and a later visit is a reference, so every reader walks in it.
    """
    return sorted(names, key=lambda name: (name.startswith("_"), name))


# ---------------------------------------------------------------------------------
# Recorded and replayed callables
# ---------------------------------------------------------------------------------

def _call_key(args: Sequence[Any]) -> str:
    parts = []
    for value in args:
        if isinstance(value, bool) or not isinstance(value, (int, float, numpy.floating, numpy.integer)):
            raise CapsuleRefusal(
                f"a recorded callable was called with a {type(value).__name__} argument; only "
                f"real scalar arguments can be keyed exactly")
        parts.append(_f64_hex(float(value)))
    return "|".join(parts)


class _RecordingCallable:
    """A caller's function, answering as itself and keeping every call it answered.

    A call the table cannot hold exactly (an array argument, a non-scalar result, two
    results for one argument) never fails the caller's run: the proxy still answers as
    the function did, notes why, and :func:`write_capsule` refuses the capsule by name.
    """

    def __init__(self, function: Callable[..., Any], label: str):
        self.function = function
        self.label = label
        self.calls: Dict[str, Any] = {}
        self.unrecordable: Optional[str] = None

    def __call__(self, *args: Any) -> Any:
        value = self.function(*args)
        if self.unrecordable is not None:
            return value
        try:
            key = _call_key(args)
            node = _scalar_node(value)
        except (CapsuleRefusal, TypeError) as exc:
            self.unrecordable = f"a call it answered cannot be keyed or stored exactly ({exc})"
            return value
        if key in self.calls and _scalar_node(self.calls[key]) != node:
            self.unrecordable = "it returned two different values for the same arguments"
            return value
        self.calls[key] = value
        return value


class _ReplayedCallable:
    """The stand-in a loaded driver holds: the recorded calls, and a refusal for any other."""

    def __init__(self, label: str, calls: Mapping[str, Any]):
        self.label = label
        self.calls = dict(calls)

    def __call__(self, *args: Any) -> Any:
        key = _call_key(args)
        if key not in self.calls:
            raise CapsuleRefusal(
                f"{self.label} was called with arguments the capsule did not record "
                f"({len(self.calls)} calls recorded). A source time function is called while "
                f"stepping: load the capsule with currents=, a table recorded by "
                f"recording_currents")
        return self.calls[key]


# ---------------------------------------------------------------------------------
# The value encoding
# ---------------------------------------------------------------------------------

def _scalar_node(value: Any) -> Any:
    """The JSON node of one scalar, or raise ``TypeError`` when ``value`` is not one."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numpy.generic):
        if value.dtype.kind == "O":
            raise TypeError("object scalar")
        return {"g": value.dtype.str, "b": value.tobytes().hex()}
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return {"f": _f64_hex(value)}
    if isinstance(value, complex):
        return {"c": [_f64_hex(value.real), _f64_hex(value.imag)]}
    raise TypeError(type(value).__name__)


def _scalar_from_node(node: Any) -> Any:
    if node is None or isinstance(node, (bool, int, str)):
        return node
    if "f" in node:
        return _f64_from_hex(node["f"])
    if "c" in node:
        return complex(_f64_from_hex(node["c"][0]), _f64_from_hex(node["c"][1]))
    if "g" in node:
        return numpy.frombuffer(bytes.fromhex(node["b"]), dtype=numpy.dtype(node["g"]))[0]
    raise CapsuleRefusal(f"not a scalar node: {sorted(node)}")


def _is_scalar_node(node: Any) -> bool:
    return (node is None or isinstance(node, (bool, int, str))
            or (isinstance(node, dict) and len(node) in (1, 2) and bool(set(node) & {"f", "c", "g"})
                and set(node) <= {"f", "c", "g", "b"}))


class _Encoder:
    """One pass over a value tree: JSON nodes out, arrays and callables into tables."""

    def __init__(self, arrays: "_ArrayTable", callables: "_CallableTable"):
        self.arrays = arrays
        self.callables = callables
        self._objects: Dict[int, int] = {}
        self._keep: List[Any] = []

    def encode(self, value: Any, path: str) -> Any:
        if isinstance(value, enum.Enum):
            if not _is_package_object(value):
                raise CapsuleRefusal(f"{path}: the enum {type(value).__qualname__} is not this package's")
            return {"e": _qualified(type(value)), "n": value.name}
        try:
            return _scalar_node(value)
        except TypeError:
            pass
        if _is_array(value):
            return {"a": self.arrays.add(value, path)}
        if isinstance(value, numpy.dtype):
            return {"dt": value.str}
        if isinstance(value, (_RecordingCallable, _ReplayedCallable)):
            return {"fn": self.callables.add(value)}
        if isinstance(value, tuple):
            items = [self.encode(item, f"{path}[{index}]") for index, item in enumerate(value)]
            if hasattr(value, "_fields"):
                if not _is_package_object(value):
                    raise CapsuleRefusal(f"{path}: a {type(value).__name__} is not this package's")
                return {"nt": _qualified(type(value)), "v": items}
            if type(value) is not tuple:
                raise CapsuleRefusal(f"{path}: a {type(value).__name__} is not carried")
            return {"t": items}
        if type(value) is list:
            return {"l": [self.encode(item, f"{path}[{index}]") for index, item in enumerate(value)]}
        if type(value) is dict:
            return {"d": [[self.encode(key, f"{path}<key>"), self.encode(item, f"{path}[{key!r}]")]
                          for key, item in value.items()]}
        if _is_package_object(value):
            known = self._objects.get(id(value))
            if known is not None:
                return {"r": known}
            index = len(self._objects)
            self._objects[id(value)] = index
            self._keep.append(value)
            attributes = {name: self.encode(item, f"{path}.{name}") for name, item in _attributes(value)}
            return {"o": _qualified(type(value)), "i": index, "s": attributes}
        if callable(value):
            raise CapsuleRefusal(
                f"{path}: the callable {getattr(value, '__qualname__', type(value).__name__)!r} was "
                f"not recorded; a capsule carries a caller's function only as the table of calls "
                f"recording_construction saw it answer")
        raise CapsuleRefusal(
            f"{path}: a {type(value).__module__}.{type(value).__qualname__} is not carried by "
            f"capsule schema {SCHEMA_VERSION}")


class _ArrayTable:
    """Distinct arrays by identity, each stored once as host bytes."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.payloads: List[numpy.ndarray] = []
        self._index: Dict[int, int] = {}
        self._keep: List[Any] = []
        self.first_paths: List[str] = []

    def add(self, array: Any, path: str) -> int:
        known = self._index.get(id(array))
        if known is not None:
            return known
        host = numpy.ascontiguousarray(to_numpy(array))
        if host.dtype.kind == "O":
            raise CapsuleRefusal(f"{path}: an object array is not state")
        index = len(self.entries)
        self._index[id(array)] = index
        self._keep.append(array)
        self.payloads.append(host)
        self.first_paths.append(path)
        self.entries.append({
            "member": f"arrays/{index:05d}.npy", "dtype": host.dtype.str,
            "shape": list(host.shape), "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
            "path": path,
        })
        return index


class _CallableTable:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._index: Dict[int, int] = {}
        self._keep: List[Any] = []

    def add(self, function: Any) -> int:
        known = self._index.get(id(function))
        if known is not None:
            return known
        if getattr(function, "unrecordable", None):
            raise CapsuleRefusal(f"{function.label} cannot be replayed: {function.unrecordable}")
        index = len(self.entries)
        self._index[id(function)] = index
        self._keep.append(function)
        self.entries.append({"label": function.label, "calls": function.calls})
        return index

    def document(self) -> List[Dict[str, Any]]:
        return [{"label": entry["label"],
                 "calls": [[key, _scalar_node(value)] for key, value in entry["calls"].items()]}
                for entry in self.entries]


# ---------------------------------------------------------------------------------
# Recording the construction
# ---------------------------------------------------------------------------------

@dataclasses.dataclass
class ConstructionLog:
    """The calls that built one driver, with their arguments as they were passed."""

    constructor: Dict[str, Any]
    calls: List[Tuple[str, Dict[str, Any]]] = dataclasses.field(default_factory=list)
    monitor_calls: List[str] = dataclasses.field(default_factory=list)
    depth: int = 0


class ConstructionRecorder:
    """The logs of every driver constructed while :func:`recording_construction` is open."""

    def __init__(self):
        self._logs: "weakref.WeakKeyDictionary[FdtdDriver, ConstructionLog]" = weakref.WeakKeyDictionary()
        self._memo: Dict[int, Tuple[Any, Any]] = {}

    def log_for(self, driver: FdtdDriver) -> ConstructionLog:
        log = self._logs.get(driver)
        if log is None:
            raise CapsuleRefusal(
                "this driver was not constructed under recording_construction(), so the calls "
                "that configured it are unknown and a capsule of it cannot be replayed")
        return log

    def _snapshot(self, value: Any, label: str) -> Any:
        """A copy of one argument as it is at call time; callables become recording proxies."""
        if _is_array(value):
            known = self._memo.get(id(value))
            host = numpy.array(to_numpy(value), copy=True)
            if known is not None and known[0] is value and numpy.array_equal(
                    known[1].view(numpy.uint8), host.view(numpy.uint8)):
                return known[1]
            self._memo[id(value)] = (value, host)
            return host
        if isinstance(value, dict):
            return type(value)((key, self._snapshot(item, f"{label}[{key!r}]")) for key, item in value.items())
        if isinstance(value, list):
            return [self._snapshot(item, f"{label}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, tuple) and type(value) is tuple:
            return tuple(self._snapshot(item, f"{label}[{index}]") for index, item in enumerate(value))
        if callable(value) and not isinstance(value, type) and not _is_package_object(value):
            return _RecordingCallable(value, label)
        return value


_ACTIVE_RECORDER: List[ConstructionRecorder] = []


@contextlib.contextmanager
def recording_construction() -> Iterator[ConstructionRecorder]:
    """Record the constructor call and the configuration calls of every driver built inside.

    Use it around ``lift_simulation`` (or around a hand-built driver's construction), then
    pass the recorder to :func:`write_capsule`. The recording wraps the methods on the
    class for the duration of the block and restores them on exit; it changes no
    argument except that a callable is handed on as a proxy that answers as the
    original and keeps its answers.
    """
    if _ACTIVE_RECORDER:
        raise CapsuleRefusal("recording_construction() is already open; one recording at a time")
    recorder = ConstructionRecorder()
    originals: Dict[str, Any] = {}
    init_signature = inspect.signature(FdtdDriver.__init__)
    for name in ("__init__",) + CONSTRUCTION_CALLS:
        kinds = {parameter.kind for parameter in inspect.signature(FdtdDriver.__dict__[name]).parameters.values()}
        if kinds - {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            raise CapsuleRefusal(
                f"FdtdDriver.{name} takes positional-only or variadic parameters, which a "
                f"recorded call cannot be replayed through by name")

    def wrap_init(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def __init__(self: FdtdDriver, *args: Any, **kwargs: Any) -> None:
            bound = init_signature.bind(self, *args, **kwargs)
            arguments = {name: recorder._snapshot(value, f"FdtdDriver({name}=)")
                         for name, value in bound.arguments.items() if name != "self"}
            log = ConstructionLog(constructor=arguments)
            recorder._logs[self] = log
            log.depth += 1
            try:
                original(self, *args, **kwargs)
            finally:
                log.depth -= 1
        return __init__

    def wrap_call(name: str, original: Callable[..., Any], monitor: bool) -> Callable[..., Any]:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def method(self: FdtdDriver, *args: Any, **kwargs: Any) -> Any:
            log = recorder._logs.get(self)
            if log is None or log.depth:
                return original(self, *args, **kwargs)
            if monitor:
                log.monitor_calls.append(name)
                log.depth += 1
                try:
                    return original(self, *args, **kwargs)
                finally:
                    log.depth -= 1
            bound = signature.bind(self, *args, **kwargs)
            arguments = {key: recorder._snapshot(value, f"{name}({key}=)")
                         for key, value in bound.arguments.items() if key != "self"}
            # The proxies, not the caller's functions, are what the driver receives, so
            # that every call the driver makes on them is recorded.
            live = {key: (arguments[key] if isinstance(arguments[key], _RecordingCallable)
                          else _rebind_proxies(value, arguments[key]))
                    for key, value in bound.arguments.items() if key != "self"}
            entry = (name, arguments)
            log.calls.append(entry)
            log.depth += 1
            try:
                return original(self, **live)
            except BaseException:
                log.calls.remove(entry)
                raise
            finally:
                log.depth -= 1
        return method

    _ACTIVE_RECORDER.append(recorder)
    try:
        originals["__init__"] = FdtdDriver.__dict__["__init__"]
        FdtdDriver.__init__ = wrap_init(originals["__init__"])  # type: ignore[method-assign]
        for name in CONSTRUCTION_CALLS + MONITOR_CALLS:
            originals[name] = FdtdDriver.__dict__[name]
            setattr(FdtdDriver, name, wrap_call(name, originals[name], name in MONITOR_CALLS))
        yield recorder
    finally:
        for name, original in originals.items():
            setattr(FdtdDriver, name, original)
        _ACTIVE_RECORDER.clear()


def _rebind_proxies(live: Any, snapshot: Any) -> Any:
    """``live`` with every callable replaced by the proxy its snapshot holds."""
    if isinstance(snapshot, _RecordingCallable):
        return snapshot
    if isinstance(live, dict) and isinstance(snapshot, dict):
        if not any(isinstance(item, _RecordingCallable) for item in snapshot.values()):
            return live
        return type(live)((key, _rebind_proxies(live[key], snapshot[key])) for key in live)
    return live


# ---------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CapsuleInfo:
    """What :func:`write_capsule` wrote."""

    path: Path
    file_sha256: str
    document_sha256: str
    arrays: int
    array_bytes: int
    construction_calls: int
    step_count: int


def _subnormal_mode() -> str:  # The calling thread's float32 subnormal behaviour, measured.
    tiny = numpy.frombuffer(struct.pack("<I", 0x00800000), dtype=numpy.float32)[0]
    value = tiny * numpy.float32(0.25)
    return "keep" if value != 0 else "flush"


def _code_identity() -> Dict[str, Dict[str, str]]:
    here = Path(__file__).resolve().parent
    out = {}
    for name in LIFT_MODULES:
        path = here / f"{name}.py"
        out[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "code_sha256": code_digest_of_path(path)}
    return out


def _package_version() -> str:
    try:
        import meep_gpu  # noqa: PLC0415

        return str(meep_gpu.__version__)
    except Exception as exc:  # noqa: BLE001 - provenance, reported as unavailable
        return f"unavailable: {type(exc).__name__}"


def _chunk_division(sim: Any, driver: FdtdDriver) -> Dict[str, Any]:
    """MEEP's chunk division of ``sim``, as data (boxes in doubled lattice coordinates)."""
    cylindrical = bool(getattr(driver.grid, "cylindrical", False))

    def corner(vec: Any) -> List[int]:
        if cylindrical:
            return [int(vec.r()), 0, int(vec.z())]
        return [int(vec.x()), int(vec.y()), int(vec.z())]

    structure = sim.structure
    if structure is None:
        raise CapsuleRefusal("the simulation passed to write_capsule has not been initialised")
    volumes = list(structure.get_chunk_volumes())
    owners = [int(owner) for owner in structure.get_chunk_owners()]
    return {
        "format": "meep.structure.get_chunk_volumes/1",
        "boxes": [[corner(volume.little_corner()), corner(volume.big_corner())] for volume in volumes],
        "owners": owners,
        "user_volume": [corner(structure.user_volume.little_corner()),
                        corner(structure.user_volume.big_corner())],
        "num_chunks": int(getattr(sim, "num_chunks", 0) or 0),
        "chunk_layout_class": type(getattr(sim, "chunk_layout", None)).__name__,
    }


def _meep_build(sim: Any) -> Optional[Dict[str, Any]]:
    mp = sys.modules.get("meep")
    if sim is None or mp is None:
        return None
    return {"version": str(getattr(mp, "__version__", "unknown")),
            "single_precision": bool(mp.is_single_precision()),
            "module_file": _portable_module_path(Path(mp.__file__).resolve())}


def _portable_module_path(path: Path) -> str:
    """A module's path from its ``site-packages`` (or ``dist-packages``) directory on.

    A capsule is a fixture a repository may carry, so it names no host directory; the
    MEEP build itself is identified by its images' digests where an identity record needs
    it (``meep_identity.json``).
    """
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in ("site-packages", "dist-packages"):
            return "/".join(parts[index:])
    return "/".join(parts[-2:])


def _require_writable(driver: FdtdDriver, log: ConstructionLog) -> None:
    if getattr(driver, "_closed", False):
        raise CapsuleRefusal("the driver is closed; its arrays are gone")
    if log.monitor_calls:
        raise CapsuleRefusal(
            f"the driver was given monitors ({', '.join(log.monitor_calls)}); capsule schema "
            f"{SCHEMA_VERSION} carries no monitor state")
    for name in MONITOR_ATTRIBUTES:
        if getattr(driver, name, None):
            raise CapsuleRefusal(f"driver.{name} is not empty; capsule schema {SCHEMA_VERSION} carries no monitor")
    migrated = getattr(driver, "migrated_monitors", None)
    if migrated is not None and len(migrated):
        raise CapsuleRefusal(
            f"the lift migrated {len(migrated)} MEEP monitor(s); capsule schema {SCHEMA_VERSION} "
            f"carries no monitor state")
    if getattr(driver, "_synchronized_magnetic", 0) or getattr(driver, "_magnetic_backup", {}):
        raise CapsuleRefusal("the magnetic fields are synchronized; restore them before writing a capsule")


def write_capsule(path: Any, driver: FdtdDriver, recorder: ConstructionRecorder, *,
                  sim: Any = None) -> CapsuleInfo:
    """Write ``driver`` as a capsule at ``path``.

    ``recorder`` is the :func:`recording_construction` block the driver was built in.
    ``sim``, when given, is the MEEP simulation the driver was lifted from: its chunk
    division and the MEEP build are recorded beside the state. The driver may have been
    stepped; the capsule then holds the state at its ``step_count``.
    """
    log = recorder.log_for(driver)
    _require_writable(driver, log)
    arrays = _ArrayTable()
    callables = _CallableTable()
    encoder = _Encoder(arrays, callables)
    construction = {
        "constructor": encoder.encode(log.constructor, "constructor"),
        "calls": [[name, encoder.encode(arguments, f"call {index} {name}")]
                  for index, (name, arguments) in enumerate(log.calls)],
    }
    # The state is encoded with its own object numbering but the same array and
    # callable tables, so a function the lift handed to the driver is one table.
    state_encoder = _Encoder(arrays, callables)
    state = state_encoder.encode(driver, "driver")
    document = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "construction": construction,
        "callables": callables.document(),
        "state": state,
        "arrays": arrays.entries,
        "chunk_division": _chunk_division(sim, driver) if sim is not None else None,
        "provenance": {
            "package_version": _package_version(),
            "numpy": numpy.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "subnormal_mode": _subnormal_mode(),
            "code": _code_identity(),
            "meep": _meep_build(sim),
            "step_count": int(driver.step_count),
            "source_times_from_meep": _meep_bound_sources(driver),
        },
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    total = 0
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for entry, payload in zip(arrays.entries, arrays.payloads):
            buffer = io.BytesIO()
            numpy.save(buffer, payload, allow_pickle=False)
            total += payload.nbytes
            bundle.writestr(_zip_info(entry["member"]), buffer.getvalue())
        bundle.writestr(_zip_info(_JSON_MEMBER), text)
    temporary.replace(target)
    return CapsuleInfo(
        path=target, file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        document_sha256=hashlib.sha256(text).hexdigest(), arrays=len(arrays.entries),
        array_bytes=total, construction_calls=len(log.calls), step_count=int(driver.step_count))


def _zip_info(name: str) -> zipfile.ZipInfo:  # Fixed timestamp and mode: the bytes are reproducible.
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


# ---------------------------------------------------------------------------------
# Reading and loading
# ---------------------------------------------------------------------------------

class Capsule:
    """A capsule read from disk and verified, ready to build drivers from."""

    def __init__(self, path: Path, document: Dict[str, Any], members: Dict[str, bytes],
                 file_sha256: str, document_sha256: str):
        self.path = path
        self.document = document
        self._members = members
        self.file_sha256 = file_sha256
        self.document_sha256 = document_sha256
        self.load_report: Optional[Dict[str, Any]] = None
        self._arrays: Dict[int, numpy.ndarray] = {}

    @property
    def provenance(self) -> Dict[str, Any]:
        return self.document["provenance"]

    @property
    def chunk_division(self) -> Optional[Dict[str, Any]]:
        return self.document["chunk_division"]

    @property
    def step_count(self) -> int:
        return int(self.document["provenance"]["step_count"])

    def array(self, index: int) -> numpy.ndarray:
        """Stored array ``index``, read-only, after its digest was checked."""
        cached = self._arrays.get(index)
        if cached is not None:
            return cached
        entry = self.document["arrays"][index]
        array = numpy.load(io.BytesIO(self._members[entry["member"]]), allow_pickle=False)
        if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
            raise CapsuleRefusal(f"capsule member {entry['member']} is not the array the document records")
        if hashlib.sha256(numpy.ascontiguousarray(array).tobytes()).hexdigest() != entry["sha256"]:
            raise CapsuleRefusal(f"capsule member {entry['member']} ({entry['path']}) fails its sha256")
        array.setflags(write=False)
        self._arrays[index] = array
        return array

    def build_driver(self, *, prefer_gpu: bool = False, gpu_id: int = 0,
                     currents: Optional["CurrentsTable"] = None) -> FdtdDriver:
        """A new driver holding exactly the stored state.

        ``prefer_gpu``/``gpu_id`` choose the backend as :class:`FdtdDriver` does; False is
        the NumPy reference. ``currents`` injects a recorded table into the sources (see
        :func:`install_currents`); without it the sources evaluate their own time
        functions, except a custom waveform, which the capsule cannot evaluate and which
        is then refused here.
        """
        construction = self.document["construction"]
        decoder = _Decoder(self)
        constructor = decoder.decode(construction["constructor"])
        constructor["prefer_gpu"] = bool(prefer_gpu)
        constructor["gpu_id"] = int(gpu_id)
        try:
            driver = FdtdDriver(**constructor)
        except TypeError as exc:
            raise CapsuleRefusal(f"replaying the FdtdDriver constructor: {exc}") from None
        try:
            for name, node in construction["calls"]:
                if name not in CONSTRUCTION_CALLS:
                    raise CapsuleRefusal(f"the capsule replays {name!r}, which is not a construction call")
                arguments = decoder.decode(node)
                try:
                    getattr(driver, name)(**arguments)
                except TypeError as exc:
                    raise CapsuleRefusal(f"replaying {name}({', '.join(arguments)}): {exc}") from None
            stored_driver = self.document["state"]
            for name in LIFT_ATTRIBUTES:
                if name in stored_driver["s"]:
                    # Numbered in the state's object space, not the log's: a decoder of its own.
                    setattr(driver, name, _Decoder(self).decode(stored_driver["s"][name]))
            reconciler = _Reconciler(self)
            reconciler.reconcile(driver, stored_driver, "driver")
            self.load_report = reconciler.report()
            if currents is not None:
                install_currents(driver, currents)
            else:
                _require_evaluable_sources(driver, self.provenance.get("source_times_from_meep") or ())
        except BaseException:
            driver.close()
            raise
        return driver


def _meep_bound_sources(driver: FdtdDriver) -> List[int]:
    """The sources whose time function the lift bound to MEEP's own ``src_time``."""
    return [index for index, source in enumerate(driver._sources)  # noqa: SLF001
            if vars(_time_function_owner(source)).get("_meep_src_time") is not None]


def _require_evaluable_sources(driver: FdtdDriver, from_meep: Sequence[int] = ()) -> None:
    from_meep = {int(index) for index in from_meep}
    for index, source in enumerate(driver._sources):  # noqa: SLF001 - the driver's source list
        envelope = vars(source).get("envelope")
        function = vars(envelope).get("func") if envelope is not None else None
        if isinstance(function, _ReplayedCallable):
            raise CapsuleRefusal(
                f"source {index} ({type(source).__name__}) carries a custom waveform, which a "
                f"capsule cannot evaluate; load it with currents=, a table recorded by "
                f"recording_currents")
        if index in from_meep:
            raise CapsuleRefusal(
                f"source {index} ({type(source).__name__}) formed its currents through the "
                f"lifted simulation's own MEEP src_time, which a capsule does not carry and "
                f"the package's own waveform does not reproduce word for word; load it with "
                f"currents=, a table recorded by recording_currents")


def read_capsule(path: Any) -> Capsule:
    """Read and verify the capsule at ``path``; any other schema or version is refused."""
    target = Path(path)
    data = target.read_bytes()
    try:
        bundle = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CapsuleRefusal(f"{target} is not a capsule: {exc}") from None
    with bundle:
        names = bundle.namelist()
        if _JSON_MEMBER not in names:
            raise CapsuleRefusal(f"{target} is not a capsule: it has no {_JSON_MEMBER}")
        text = bundle.read(_JSON_MEMBER)
        document = json.loads(text.decode("utf-8"))
        if document.get("schema") != SCHEMA:
            raise CapsuleRefusal(f"{target} is not a capsule: schema {document.get('schema')!r}")
        if document.get("version") != SCHEMA_VERSION:
            raise CapsuleRefusal(
                f"{target} is capsule schema version {document.get('version')!r}; this tree reads "
                f"version {SCHEMA_VERSION} only. Re-mint the capsule with this tree")
        expected = {entry["member"] for entry in document["arrays"]} | {_JSON_MEMBER}
        if set(names) != expected:
            raise CapsuleRefusal(
                f"{target}: members {sorted(set(names) ^ expected)} differ from the document's list")
        members = {name: bundle.read(name) for name in names if name != _JSON_MEMBER}
    return Capsule(target, document, members, hashlib.sha256(data).hexdigest(),
                   hashlib.sha256(text).hexdigest())


def load_capsule(path: Any, *, prefer_gpu: bool = False, gpu_id: int = 0,
                 currents: Optional["CurrentsTable"] = None) -> FdtdDriver:
    """:func:`read_capsule` then :meth:`Capsule.build_driver`."""
    return read_capsule(path).build_driver(prefer_gpu=prefer_gpu, gpu_id=gpu_id, currents=currents)


class _Decoder:
    """Nodes of the construction log back to arguments: arrays as fresh writable copies."""

    def __init__(self, capsule: Capsule):
        self.capsule = capsule
        self._arrays: Dict[int, numpy.ndarray] = {}
        self._objects: Dict[int, Any] = {}
        self._callables: Dict[int, _ReplayedCallable] = {}

    def decode(self, node: Any) -> Any:
        if _is_scalar_node(node):
            return _scalar_from_node(node)
        if "a" in node:
            index = node["a"]
            if index not in self._arrays:
                self._arrays[index] = numpy.array(self.capsule.array(index), copy=True)
            return self._arrays[index]
        if "dt" in node:
            return numpy.dtype(node["dt"])
        if "fn" in node:
            index = node["fn"]
            if index not in self._callables:
                entry = self.capsule.document["callables"][index]
                self._callables[index] = _ReplayedCallable(
                    entry["label"], {key: _scalar_from_node(value) for key, value in entry["calls"]})
            return self._callables[index]
        if "e" in node:
            return _resolve_class(node["e"])[node["n"]]
        if "t" in node:
            return tuple(self.decode(item) for item in node["t"])
        if "nt" in node:
            return _resolve_class(node["nt"])(*(self.decode(item) for item in node["v"]))
        if "l" in node:
            return [self.decode(item) for item in node["l"]]
        if "d" in node:
            return {self.decode(key): self.decode(item) for key, item in node["d"]}
        if "r" in node:
            if node["r"] not in self._objects:
                raise CapsuleRefusal(
                    "a construction argument refers to an object before it is complete (a cycle), "
                    "which a capsule does not carry")
            return self._objects[node["r"]]
        if "o" in node:
            return self._object(node)
        raise CapsuleRefusal(f"unknown capsule node {sorted(node)}")

    def _object(self, node: Dict[str, Any]) -> Any:
        cls = _resolve_class(node["o"])
        attributes = {name: self.decode(node["s"][name]) for name in _attribute_order(node["s"])}
        obj = cls(**_constructor_arguments(cls, attributes, node["o"]))
        for name, value in attributes.items():
            object.__setattr__(obj, name, value)
        for name, _ in _attributes(obj):
            if name not in attributes:
                raise CapsuleRefusal(
                    f"{node['o']} in this tree sets {name!r}, which the capsule does not carry")
        self._objects[node["i"]] = obj
        return obj


def _constructor_arguments(cls: type, attributes: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """The arguments that rebuild a ``cls`` through its own constructor from its attributes.

    A dataclass takes its ``init`` fields. Any other class takes, for each constructor
    parameter, the attribute of that name or of that name with a leading underscore
    (``MigratedMonitors(pairs=...)`` stores ``_pairs``); a parameter with a default may
    go unfilled. The stored attributes are written over the result afterwards, so what
    the constructor derived is replaced, and what it initialised that is not carried
    (a memo, a scratch buffer) is the constructor's.
    """
    if dataclasses.is_dataclass(cls):
        return {field.name: attributes[field.name] for field in dataclasses.fields(cls)
                if field.init and field.name in attributes}
    arguments: Dict[str, Any] = {}
    for parameter in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        for candidate in (parameter.name, "_" + parameter.name):
            if candidate in attributes:
                arguments[parameter.name] = attributes[candidate]
                break
        else:
            if parameter.default is inspect.Parameter.empty:
                raise CapsuleRefusal(
                    f"{name} cannot be rebuilt through its constructor: no carried attribute "
                    f"answers its argument {parameter.name!r}")
    return arguments


class _Reconciler:
    """Walk the rebuilt driver beside the stored graph: check the shape, paste the state."""

    def __init__(self, capsule: Capsule):
        self.capsule = capsule
        self._array_of_index: Dict[int, Any] = {}
        self._index_of_array: Dict[int, int] = {}
        self._object_of_index: Dict[int, Any] = {}
        self._index_of_object: Dict[int, int] = {}
        self.arrays_pasted = 0
        self.arrays_differing: List[str] = []
        self.scalars_compared = 0
        self.scalars_differing: List[str] = []

    def report(self) -> Dict[str, Any]:
        return {
            "arrays_pasted": self.arrays_pasted,
            "arrays_rebuilt_differently": len(self.arrays_differing),
            "arrays_rebuilt_differently_paths": self.arrays_differing[:50],
            "scalars_compared": self.scalars_compared,
            "scalars_rebuilt_differently": len(self.scalars_differing),
            "scalars_rebuilt_differently_paths": self.scalars_differing[:50],
        }

    def reconcile(self, live: Any, node: Any, path: str) -> Any:
        if _is_scalar_node(node):
            return self._scalar(live, node, path)
        if "a" in node:
            return self._array(live, node["a"], path)
        if "dt" in node:
            if not isinstance(live, numpy.dtype) or live.str != node["dt"]:
                raise CapsuleRefusal(f"{path}: the rebuilt dtype {live!r} is not the stored {node['dt']}")
            return live
        if "fn" in node:
            if not isinstance(live, _ReplayedCallable):
                raise CapsuleRefusal(f"{path}: the stored callable was rebuilt as {type(live).__name__}")
            return live
        if "e" in node:
            if not isinstance(live, enum.Enum) or _qualified(type(live)) != node["e"] or live.name != node["n"]:
                raise CapsuleRefusal(f"{path}: the rebuilt {live!r} is not the stored {node['e']}.{node['n']}")
            return live
        if "t" in node or "nt" in node:
            items = node["t"] if "t" in node else node["v"]
            if not isinstance(live, tuple) or len(live) != len(items):
                raise CapsuleRefusal(f"{path}: the rebuilt {type(live).__name__} is not a stored tuple of {len(items)}")
            if "nt" in node and _qualified(type(live)) != node["nt"]:
                raise CapsuleRefusal(f"{path}: the rebuilt {_qualified(type(live))} is not the stored {node['nt']}")
            values = [self.reconcile(item, child, f"{path}[{index}]")
                      for index, (item, child) in enumerate(zip(live, items))]
            if all(new is old for new, old in zip(values, live)):
                return live
            return type(live)(*values) if "nt" in node else tuple(values)
        if "l" in node:
            if type(live) is not list or len(live) != len(node["l"]):
                raise CapsuleRefusal(
                    f"{path}: the rebuilt {type(live).__name__} of {len(live) if hasattr(live, '__len__') else '?'} "
                    f"is not the stored list of {len(node['l'])}")
            for index, child in enumerate(node["l"]):
                live[index] = self.reconcile(live[index], child, f"{path}[{index}]")
            return live
        if "d" in node:
            if type(live) is not dict:
                raise CapsuleRefusal(f"{path}: the rebuilt {type(live).__name__} is not the stored dict")
            keys = [_Decoder(self.capsule).decode(key) for key, _ in node["d"]]
            if list(live.keys()) != keys:
                raise CapsuleRefusal(
                    f"{path}: the rebuilt keys {list(live.keys())} are not the stored {keys}")
            for key, (_, child) in zip(keys, node["d"]):
                dict.__setitem__(live, key, self.reconcile(dict.__getitem__(live, key), child, f"{path}[{key!r}]"))
            return live
        if "r" in node:
            expected = self._object_of_index.get(node["r"])
            if expected is not live:
                raise CapsuleRefusal(
                    f"{path}: the stored graph reaches one object here and at an earlier path; "
                    f"the rebuilt driver holds a different one")
            return live
        if "o" in node:
            return self._object(live, node, path)
        raise CapsuleRefusal(f"{path}: unknown capsule node {sorted(node)}")

    def _scalar(self, live: Any, node: Any, path: str) -> Any:
        stored = _scalar_from_node(node)
        try:
            rebuilt = _scalar_node(live)
        except TypeError:
            raise CapsuleRefusal(
                f"{path}: the stored scalar was rebuilt as a {type(live).__name__}") from None
        if type(live) is not type(stored) and not (live is None and stored is None):
            raise CapsuleRefusal(
                f"{path}: the stored {type(stored).__name__} was rebuilt as a {type(live).__name__}")
        self.scalars_compared += 1
        if rebuilt != node:
            self.scalars_differing.append(path)
        return stored

    def _array(self, live: Any, index: int, path: str) -> Any:
        if not _is_array(live):
            raise CapsuleRefusal(f"{path}: the stored array was rebuilt as a {type(live).__name__}")
        known = self._array_of_index.get(index)
        if known is not None:
            if known is not live:
                raise CapsuleRefusal(
                    f"{path}: the stored driver holds one array here and at "
                    f"{self.capsule.document['arrays'][index]['path']}; the rebuilt driver holds two")
            return live
        if id(live) in self._index_of_array:
            raise CapsuleRefusal(
                f"{path}: the rebuilt driver holds one array here and at "
                f"{self.capsule.document['arrays'][self._index_of_array[id(live)]]['path']}; the "
                f"stored driver holds two")
        stored = self.capsule.array(index)
        if tuple(live.shape) != stored.shape or numpy.dtype(live.dtype) != stored.dtype:
            raise CapsuleRefusal(
                f"{path}: rebuilt {tuple(live.shape)} {numpy.dtype(live.dtype).str}, stored "
                f"{stored.shape} {stored.dtype.str}")
        if not live.flags.c_contiguous:
            raise CapsuleRefusal(f"{path}: the rebuilt array is not C-contiguous; its bytes cannot be pasted")
        self._array_of_index[index] = live
        self._index_of_array[id(live)] = index
        before = numpy.ascontiguousarray(to_numpy(live))
        if before.tobytes() == stored.tobytes():
            return live
        self.arrays_differing.append(path)
        raw = stored.reshape(-1).view(numpy.uint8)
        if isinstance(live, numpy.ndarray):
            if not live.flags.writeable:
                raise CapsuleRefusal(f"{path}: the rebuilt array is read-only and differs from the stored one")
            live.reshape(-1).view(numpy.uint8)[...] = raw
        else:
            module = sys.modules[type(live).__module__.split(".")[0]]
            live.reshape(-1).view(module.uint8)[...] = module.asarray(raw)
        self.arrays_pasted += 1
        return live

    def _object(self, live: Any, node: Dict[str, Any], path: str) -> Any:
        if _qualified(type(live)) != node["o"]:
            raise CapsuleRefusal(f"{path}: the stored {node['o']} was rebuilt as {_qualified(type(live))}")
        if id(live) in self._index_of_object:
            raise CapsuleRefusal(
                f"{path}: the rebuilt driver reaches one {node['o']} here and earlier; the stored "
                f"driver holds two")
        self._object_of_index[node["i"]] = live
        self._index_of_object[id(live)] = node["i"]
        rebuilt = dict(_attributes(live))
        stored = node["s"]
        if set(rebuilt) != set(stored):
            only_stored = sorted(set(stored) - set(rebuilt))
            only_rebuilt = sorted(set(rebuilt) - set(stored))
            raise CapsuleRefusal(
                f"{path} ({node['o']}): the capsule carries {only_stored} that the rebuilt object "
                f"lacks, and the rebuilt object holds {only_rebuilt} that the capsule does not "
                f"carry. The tree that minted the capsule built this object differently")
        for name in _attribute_order(stored):
            value = self.reconcile(rebuilt[name], stored[name], f"{path}.{name}")
            if value is not rebuilt[name]:
                object.__setattr__(live, name, value)
        return live


# ---------------------------------------------------------------------------------
# The state, flattened: what the capsule pins, for comparisons and digests
# ---------------------------------------------------------------------------------

def state_arrays(driver: FdtdDriver) -> Dict[str, numpy.ndarray]:
    """Every array a capsule of ``driver`` stores, as host copies keyed by first path.

    The same walk :func:`write_capsule` makes: the fields, the PML coefficient tables
    and auxiliaries, the material arrays, the source deposition tables and the grid
    coordinates, each distinct array once. Callables recorded or replayed are skipped
    rather than refused, so a loaded driver with injected currents can be compared.
    """
    arrays = _ArrayTable()
    _walk_arrays(driver, "driver", arrays, set())
    return {path: numpy.array(payload, copy=True) for path, payload in zip(arrays.first_paths, arrays.payloads)}


def _walk_arrays(value: Any, path: str, arrays: _ArrayTable, seen: set) -> None:
    if _is_array(value):
        arrays.add(value, path)
        return
    if isinstance(value, (str, bytes, int, float, complex, numpy.generic, numpy.dtype)) or value is None:
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_arrays(item, f"{path}[{key!r}]", arrays, seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_arrays(item, f"{path}[{index}]", arrays, seen)
    elif _is_package_object(value) and not isinstance(value, (_RecordingCallable, _ReplayedCallable)):
        for name, item in _attributes(value):
            if name in _TIME_FUNCTIONS and callable(item):
                continue
            _walk_arrays(item, f"{path}.{name}", arrays, seen)


def state_digest(driver: FdtdDriver) -> str:
    """SHA-256 over :func:`state_arrays`: path, dtype, shape and bytes of each, in path order."""
    digest = hashlib.sha256()
    for path, array in sorted(state_arrays(driver).items()):
        digest.update(path.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------------
# Injected currents
# ---------------------------------------------------------------------------------

def _time_function_owner(source: Any) -> Any:
    """The object whose ``current``/``dipole`` the source's ``inject`` calls.

    ``VolumeSource`` asks its envelope; the dedicated electric classes ask themselves.
    """
    envelope = vars(source).get("envelope")
    return envelope if isinstance(envelope, SourceEnvelope) else source


@dataclasses.dataclass
class CurrentsTable:
    """Per source: every value its time function returned, keyed by the exact arguments.

    ``sources[i]`` is ``{"source": class, "owner": class, "component": name,
    "calls": {"current": {key: value}, "dipole": {key: value}}}``; keys are the
    arguments' IEEE-754 bit patterns, values the returned scalars.
    """

    sources: List[Dict[str, Any]]
    steps_recorded: int
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def write(self, path: Any) -> str:
        """Write the table as JSON (bit patterns throughout); returns its sha256."""
        document = {
            "schema": CURRENTS_SCHEMA, "version": CURRENTS_SCHEMA_VERSION,
            "steps_recorded": self.steps_recorded, "provenance": self.provenance,
            "sources": [{
                "source": entry["source"], "owner": entry["owner"], "component": entry["component"],
                "calls": {name: [[key, _scalar_node(value)] for key, value in calls.items()]
                          for name, calls in entry["calls"].items()},
            } for entry in self.sources],
        }
        text = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        Path(path).write_bytes(text)
        return hashlib.sha256(text).hexdigest()

    @classmethod
    def read(cls, path: Any) -> "CurrentsTable":
        document = json.loads(Path(path).read_bytes().decode("utf-8"))
        if document.get("schema") != CURRENTS_SCHEMA or document.get("version") != CURRENTS_SCHEMA_VERSION:
            raise CapsuleRefusal(
                f"{path} is {document.get('schema')!r} version {document.get('version')!r}; this tree "
                f"reads {CURRENTS_SCHEMA} version {CURRENTS_SCHEMA_VERSION} only")
        sources = [{
            "source": entry["source"], "owner": entry["owner"], "component": entry["component"],
            "calls": {name: {key: _scalar_from_node(value) for key, value in calls}
                      for name, calls in entry["calls"].items()},
        } for entry in document["sources"]]
        return cls(sources=sources, steps_recorded=int(document["steps_recorded"]),
                   provenance=dict(document.get("provenance", {})))


class CurrentsRecorder:
    """Live while :func:`recording_currents` is open: the calls seen so far."""

    def __init__(self, driver: FdtdDriver):
        self.driver = driver
        self.start_step = int(driver.step_count)
        self._sources = list(driver._sources)  # noqa: SLF001 - the driver's source list
        self._calls: List[Dict[str, Dict[str, Any]]] = [{name: {} for name in _TIME_FUNCTIONS}
                                                        for _ in self._sources]

    def table(self) -> CurrentsTable:
        return CurrentsTable(
            sources=[{"source": _qualified(type(source)),
                      "owner": _qualified(type(_time_function_owner(source))),
                      "component": str(getattr(source, "component", "")),
                      "calls": {name: dict(calls) for name, calls in self._calls[index].items()}}
                     for index, source in enumerate(self._sources)],
            steps_recorded=int(self.driver.step_count) - self.start_step,
            provenance={"recorded_from": "the stepped driver's own source time functions",
                        "start_step": self.start_step,
                        "numpy": numpy.__version__, "machine": platform.machine(),
                        "system": platform.system()})


def _install_time_function(owner: Any, name: str, function: Callable[..., Any]) -> None:
    if name in vars(owner):
        raise CapsuleRefusal(
            f"{type(owner).__name__}.{name} is already overridden on this object; one injected "
            f"or recorded time function at a time")
    object.__setattr__(owner, name, function)


@contextlib.contextmanager
def recording_currents(driver: FdtdDriver) -> Iterator[CurrentsRecorder]:
    """Record every value each source's time function returns while the block steps ``driver``.

    The recording calls the class's own method and returns its value unchanged, so the
    run is the run it would have been. Only the outermost call is kept: a ``current``
    that differences two ``dipole`` calls records the ``current``.
    """
    recorder = CurrentsRecorder(driver)
    installed: List[Tuple[Any, str]] = []
    try:
        for index, source in enumerate(recorder._sources):
            owner = _time_function_owner(source)
            depth = [0]
            for name in _TIME_FUNCTIONS:
                method = getattr(type(owner), name)

                def record(*args: Any, _owner: Any = owner, _method: Any = method, _name: str = name,
                           _index: int = index, _depth: List[int] = depth) -> Any:
                    if _depth[0]:
                        return _method(_owner, *args)
                    _depth[0] += 1
                    try:
                        value = _method(_owner, *args)
                    finally:
                        _depth[0] -= 1
                    recorder._calls[_index][_name][_call_key(args)] = value
                    return value

                _install_time_function(owner, name, record)
                installed.append((owner, name))
        yield recorder
    finally:
        for owner, name in installed:
            vars(owner).pop(name, None)


def install_currents(driver: FdtdDriver, table: CurrentsTable) -> None:
    """Make ``driver``'s sources return exactly ``table``'s values, and refuse any other query.

    The table must describe the same sources, in the same order, with the same classes
    and components; a query the table does not hold raises :class:`CapsuleRefusal`
    naming the source and the arguments.
    """
    sources = list(driver._sources)  # noqa: SLF001 - the driver's source list
    if len(sources) != len(table.sources):
        raise CapsuleRefusal(
            f"the currents table describes {len(table.sources)} source(s); the driver holds {len(sources)}")
    for index, (source, entry) in enumerate(zip(sources, table.sources)):
        owner = _time_function_owner(source)
        described = (_qualified(type(source)), _qualified(type(owner)), str(getattr(source, "component", "")))
        recorded = (entry["source"], entry["owner"], entry["component"])
        if described != recorded:
            raise CapsuleRefusal(f"source {index} is {described}; the currents table recorded {recorded}")
        for name in _TIME_FUNCTIONS:
            calls = entry["calls"].get(name, {})

            def replay(*args: Any, _calls: Mapping[str, Any] = calls, _name: str = name,
                       _index: int = index, _kind: str = described[0]) -> Any:
                key = _call_key(args)
                if key not in _calls:
                    raise CapsuleRefusal(
                        f"source {_index} ({_kind}) asked {_name}"
                        f"({', '.join(repr(float(a)) for a in args)}), which the currents table "
                        f"does not hold ({len(_calls)} {_name} values recorded)")
                return _calls[key]

            _install_time_function(owner, name, replay)
