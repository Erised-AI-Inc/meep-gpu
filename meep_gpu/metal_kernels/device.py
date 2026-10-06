"""Compilation, device residency and toolchain provenance for the Metal families.

Extracted from ``launch.py`` when the package went from one family to a tree; the
contents are that module's, unchanged in behaviour, with exactly one deliberate
addition (:class:`Mirror` now carries a DTYPE rather than coercing to float32).
``launch.py`` re-imports the names, so the certified launch path is the same path.

WHY THE EXTRACTION IS A FINGERPRINT EVENT and was done deliberately rather than
avoided: ``compute_fingerprints`` hashes the HOST modules beside the kernel sources,
because the host chooses the Yee sub-lattice, binds the coefficient pointers and
decides the specialisation constants — each of those a silent wrong answer if it
changes and none of them visible in a kernel source. Moving code between host
modules therefore moves host hashes. The re-cut records that the delta was
HOST-MODULE HASHES ONLY, with every kernel-source hash unchanged, and that
comparison is better evidence of byte-neutrality than not refactoring would have
produced.

NO TORCH AT MODULE LEVEL. Every torch import sits inside the function that needs it,
so a predicate-only host with no GPU can still import this package.

THE RESIDENCY LAYER HAS NO TRITON COUNTERPART. The CuPy path is zero copy —
``CupyPointer`` renames an existing device allocation and the kernel writes the
engine's own array in place. On this host the engine holds NumPy and a torch MPS
tensor is a separate allocation, so a plan owns persistent device MIRRORS of the
volumes it steps. That buys back the round trip (measured 12x-64x the kernel time)
and creates one invariant this package must CHECK rather than comment on: a mirror
is valid only while nothing else writes the host array. See
:func:`..coverage.residency_reasons`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Compilation, memoized by source string
# ---------------------------------------------------------------------------

#: Compiled shader libraries, keyed by the exact source string. Each specialisation
#: is a distinct string and therefore a distinct key, which is the same discipline
#: Triton's own cache applies to a constexpr signature. The cache is a
#: plan-build-time convenience, never a launch-path one.
_LIBRARY_CACHE: Dict[str, Any] = {}


def compile_source(source: str) -> Any:
    """Compile one Metal source string, memoized.

    Cache poisoning was PROBED on this platform rather than assumed: an edited
    source with the same kernel name compiles and runs as the edit, and re-compiling
    the original serves the original. No defence is needed. The LAUNCH COUNTER on
    the gate's mutation legs stays anyway, because it certifies that the mutant RAN,
    which is a different claim from "the binary was fresh" and is what makes a
    hollow pass impossible.
    """
    library = _LIBRARY_CACHE.get(source)
    if library is None:
        import torch  # noqa: PLC0415

        library = torch.mps.compile_shader(source)
        _LIBRARY_CACHE[source] = library
    observer = _LAUNCH_OBSERVER
    return library if observer is None else _WatchedLibrary(library, observer)


# ---------------------------------------------------------------------------
# The launch-time bound-tensor check
# ---------------------------------------------------------------------------

#: The installed launch observer, or None. Module-level and default OFF: when it is
#: None :func:`compile_source` hands back the library object itself and the launch
#: path is byte-for-byte the certified one, so an unarmed run pays nothing and
#: cannot behave differently.
_LAUNCH_OBSERVER: Optional[Any] = None


class _WatchedLibrary:
    """A compiled library whose kernels report their operands before launching.

    WHY THE LIBRARY AND NOT THE PLAN. The obvious place for a "the tensors this
    launch binds are the ones it declared" check is the plan's argument tuple, and
    that place does not work: ``_args`` is assigned in exactly ONE location
    (``plans.KernelPlan.__init__``) and **25 of the real launching plan classes
    never populate it** — ``ScratchWeldPairPlan`` and its four subclasses, all four
    cylindrical fused plans, both fused ADE chains, both fused dispersive pairs, the
    stored-E and dispersive-E plans and the fill plans all build their operand tuple
    inside ``run()`` and launch it directly. A check on ``_args`` therefore passes
    VACUOUSLY on exactly the families that are most under-declared, which is the
    worst possible failure for a guard.

    Every family, without exception, reaches its kernel the same way: it calls
    :func:`compile_source` and takes the entry point off the returned library
    (``compile_source(...).fused_electric_pair_step``). That attribute access is the
    one choke point the whole arm table passes through, so the wrapper goes here.

    WHAT IT CATCHES that the per-plan read/write declaration cannot: a launch that
    binds a device tensor which is NOT a registered mirror at all. That is the
    mechanism by which a volume gets written without ever appearing in a family's
    ``volumes`` list — a plan-owned pack or scratch allocated in the builder and
    refreshed by the plan itself — and it is a measured defect class, not a
    hypothetical one (``folded_offdiag_dispersive_update_e`` copies its pack
    host->device itself, and the pack is absent from ``volumes``). Under a hold a
    mirror nobody declared is a mirror nobody syncs.
    """

    __slots__ = ("_library", "_observer")

    def __init__(self, library: Any, observer: Any) -> None:
        self._library = library
        self._observer = observer

    def __getattr__(self, name: str) -> Any:
        function = getattr(self._library, name)
        observer = self._observer

        def launch(*args: Any, **kwargs: Any) -> Any:
            observer(name, args)
            return function(*args, **kwargs)

        return launch


class LaunchAudit:
    """Records, per kernel entry point, the operands it bound — and what was foreign.

    Installed by a gate, a probe or a composition arming a hold; never by a release
    run. ``strict`` makes a foreign operand RAISE at the launch that binds it rather
    than at the end of the run, which is what turns "the field came out wrong
    somewhere in 138,447 steps" into a stack trace on one line.
    """

    __slots__ = ("residency", "strict", "launches", "foreign", "bound",
                 "scope", "out_of_scope")

    def __init__(self, residency: "Residency", strict: bool = False,
                 scope: Optional[Sequence[str]] = None) -> None:
        self.residency = residency
        self.strict = bool(strict)
        #: The mirror names the launches under audit are DECLARED to touch, or None
        #: to record without judging. When given, a launch that binds a registered
        #: mirror OUTSIDE this set is an out-of-scope binding — the failure a wrong
        #: ``reads``/``writes`` declaration produces, and the one that must not be
        #: allowed to surface as a wrong field 138,447 steps later.
        #:
        #: WHY A DECLARATION CANNOT BE TRUSTED WITHOUT THIS. A 49-family audit
        #: (2026-09-21) found 28 families writing mirrors that are absent from their
        #: own ``volumes`` list, and an adversarial pass overturned the read/write
        #: split on 4 more — all four by the same mechanism, a mirror whose role
        #: ALTERNATES with launch parity because the family rebinds the semantic
        #: name mid-run. A static table over that surface is a hypothesis; this is
        #: what turns it into a checked one.
        self.scope = None if scope is None else frozenset(scope)
        self.launches: Dict[str, int] = {}
        #: entry point -> the foreign operand positions seen
        self.foreign: Dict[str, Tuple[int, ...]] = {}
        #: entry point -> the mirror names its operands resolved to
        self.bound: Dict[str, Tuple[str, ...]] = {}
        #: entry point -> registered mirrors bound but not declared
        self.out_of_scope: Dict[str, Tuple[str, ...]] = {}

    def __call__(self, name: str, args: Sequence[Any]) -> None:
        known = {id(m.tensor): m.name for m in self.residency._mirrors.values()}
        resolved = []
        foreign = []
        for position, arg in enumerate(args):
            if not _is_device_tensor(arg):
                continue
            mirror = known.get(id(arg))
            if mirror is None:
                foreign.append(position)
            else:
                resolved.append(mirror)
        self.launches[name] = self.launches.get(name, 0) + 1
        if resolved:
            self.bound[name] = tuple(sorted(set(self.bound.get(name, ())) | set(resolved)))
        if foreign:
            self.foreign[name] = tuple(sorted(set(self.foreign.get(name, ())) | set(foreign)))
            if self.strict:
                raise AssertionError(
                    f"{name} bound {len(foreign)} device tensor(s) at position(s) "
                    f"{tuple(foreign)} that are not registered mirrors of this "
                    f"residency. Under a hold a mirror nobody declared is a mirror "
                    f"nobody syncs: either register it through Residency.mirror so "
                    f"the composer can scope it, or declare the family as writing "
                    f"outside the kernel")
        if self.scope is not None:
            undeclared = sorted(set(resolved) - self.scope)
            if undeclared:
                self.out_of_scope[name] = tuple(
                    sorted(set(self.out_of_scope.get(name, ())) | set(undeclared)))
                if self.strict:
                    raise AssertionError(
                        f"{name} bound registered mirror(s) {undeclared} that its "
                        f"slot did not declare. A scoped bracket syncs only what is "
                        f"declared, so this mirror is read stale on the way in or "
                        f"left stale on the host on the way out, and neither raises "
                        f"on its own. Widen the slot's reads/writes to the UNION of "
                        f"what it binds — which is the only sound answer for a "
                        f"family that rebinds a semantic name mid-run, since the "
                        f"role of a name there depends on launch parity")

    def report(self) -> Dict[str, Any]:
        return {
            "launches": dict(self.launches),
            "bound": {k: list(v) for k, v in self.bound.items()},
            "foreign": {k: list(v) for k, v in self.foreign.items()},
            "out_of_scope": {k: list(v) for k, v in self.out_of_scope.items()},
            "scope_declared": None if self.scope is None else sorted(self.scope),
            "entry_points": len(self.launches),
            "total_launches": sum(self.launches.values()),
            "clean": not self.foreign and not self.out_of_scope,
        }


def _is_device_tensor(value: Any) -> bool:
    """Whether ``value`` is a torch tensor on a non-CPU device.

    Deliberately duck-typed, with no torch import: this runs on the launch path
    once armed, and a predicate-only host with no GPU must still import the module.
    """
    device = getattr(value, "device", None)
    if device is None or not hasattr(value, "data_ptr"):
        return False
    return getattr(device, "type", "cpu") != "cpu"


def install_launch_observer(observer: Optional[Any]) -> Optional[Any]:
    """Install (or clear) the launch observer; returns the previous one.

    CLEARS THE LIBRARY CACHE, because a library handed out before the observer was
    installed is the bare object and would not report. The cache is a build-time
    convenience only (its own docstring says so), so dropping it costs a
    recompilation per source and never changes what a kernel does.
    """
    global _LAUNCH_OBSERVER
    previous = _LAUNCH_OBSERVER
    _LAUNCH_OBSERVER = observer
    _LIBRARY_CACHE.clear()
    return previous


#: Metal's buffer attribute indices run 0..30, so a kernel may bind at most 31
#: buffers; a 32nd is a COMPILE ERROR ("'buffer' attribute parameter is out of
#: bounds: must be between 0 and 30"), measured on this toolchain 2026-08-15. This
#: is a hard platform limit and it FORCES design decisions rather than informing
#: them — a complex curl binding re/im planes separately needs 35 buffers and cannot
#: be built at all. Families assert against it; the complex gate proves the split
#: signature fails to compile rather than leaving a future edit to rediscover it.
MAX_BUFFER_BINDINGS = 31

#: The only two element widths a Metal kernel in this package can bind, and the
#: ONE place that pairing is written down: ``float32`` binds ``device float*`` and
#: ``complex64`` binds ``device float2*`` (measured 0/16,384 words against NumPy on
#: this host). Anything else is REFUSED BY NAME in :meth:`Residency.mirror`.
#:
#: THE REFUSAL IS NOT DEFENSIVE TIDINESS, IT IS A MEASURED SILENT WRONG ANSWER.
#: Before it existed, ``mirror(name, host, dtype=numpy.complex128)`` produced a
#: ``torch.complex128`` tensor on ``mps:0``, bound it to a ``device float2*``
#: signature, and the launch SUCCEEDED — the kernel read one 128-bit cell as two
#: 64-bit ``float2`` cells, wrote garbage over the first half of the buffer and
#: raised nothing (probed on this host: 16 complex128 cells in, doubles reinterpreted
#: as float pairs out). ``float64`` happens to fail inside ``Tensor.to`` because the
#: MPS framework has no float64, but that is the platform refusing an unrelated
#: thing, not this layer refusing the binding; both are named here so the two do not
#: depend on which one the platform happens to catch.
BINDABLE_DTYPES: Tuple[str, ...] = ("float32", "complex64")


#: THE THREE OWNERSHIP STATES a mirror can be in, and the invariant each one names.
#: They exist because the shipped bracket answers the ownership question by brute
#: force — it copies every mirror both ways around every launch, so the host is
#: authoritative at every instant and nothing has to be tracked. Measured
#: 2026-09-21 on this host, that costs 0.217 ms per copy, 69 copies a launch on
#: ``pml_2d``, and **96-98% of the whole step**; the kernels themselves run at
#: ``0.495 + 0.690 N`` ms/step against an array path of ``0.245 + 19.163 N``
#: (N in Mcells). Holding the state on the device between launches is what turns
#: the dispatch from 40-50x slower than not dispatching into an accelerator above
#: ~28,000-48,000 cells.
#:
#: What each state PROMISES, since a wrong promise here is a smooth, plausible,
#: wrong field rather than an error (``..coverage`` says so at its own refusal):
#:
#: * ``CLEAN_BOTH`` — host and device words are equal. Either may be read; a write
#:   to either side must first be declared, because after it they are not.
#: * ``HOST_OWNED`` — the host array carries words the device does not have. A
#:   launch that READS this mirror must sync it in first.
#: * ``DEVICE_OWNED`` — the device tensor carries words the host does not have. Any
#:   host read must sync it out first, and the seal (:meth:`Mirror.seal`) is what
#:   makes a host WRITE raise instead of being silently overwritten.
CLEAN_BOTH = "clean_both"
HOST_OWNED = "host_owned"
DEVICE_OWNED = "device_owned"

#: Every state, for the gate's set-form checks.
OWNERSHIP_STATES: Tuple[str, ...] = (CLEAN_BOTH, HOST_OWNED, DEVICE_OWNED)


# ---------------------------------------------------------------------------
# The residency layer
# ---------------------------------------------------------------------------

class Mirror:
    """One host array and the persistent MPS tensor that shadows it.

    ``constant`` marks a volume the kernels only READ and nothing on the device
    writes — the inverse-epsilon volumes and the PML coefficient vectors. Those
    never need a sync out, and marking them says so rather than leaving a reader to
    infer it from the kernel signature.

    ``dtype`` IS CARRIED, NOT COERCED. The float32 families mirror float32 and the
    complex families mirror complex64: a ``torch`` complex64 tensor binds directly to
    a Metal ``device float2*`` (measured 0/16,384 words against NumPy on this host),
    and the dispatch is then sized in COMPLEX CELLS, which is exactly the element
    count those kernels index. Coercing to float32 here — which is what this class
    did while one family existed — would silently halve a complex plan's grid.
    """

    __slots__ = ("name", "host", "tensor", "constant", "dtype", "state", "sealed")

    def __init__(self, name: str, host: Any, tensor: Any, constant: bool,
                 dtype: Any = None) -> None:
        self.name = name
        self.host = host
        self.tensor = tensor
        self.constant = bool(constant)
        self.dtype = dtype
        #: Ownership, one of :data:`OWNERSHIP_STATES`. A freshly mirrored volume is
        #: ``CLEAN_BOTH`` because :meth:`Residency.mirror` copies the host bytes up
        #: at registration. Only :class:`Residency` moves this.
        self.state = CLEAN_BOTH
        #: Whether the host array's write flag is currently down. False for every
        #: mirror until a hold is armed, so an unheld composition behaves exactly
        #: as it did before this machinery existed.
        self.sealed = False

    # -- the seal: how a missed host write fails CLOSED rather than silently ---

    def sealable(self) -> bool:
        """Whether this host array's write flag can be taken down and put back.

        A NumPy view that does not own its data can be made read-only but cannot
        always be made writeable again — ``setflags(write=True)`` raises
        ``ValueError`` when the base is itself read-only. Sealing such an array
        would strand it, so a mirror that cannot round-trip its own flag is
        REFUSED FROM THE HOLD by name rather than sealed and hoped for.
        """
        host = self.host
        flags = getattr(host, "flags", None)
        if flags is None or not getattr(flags, "writeable", False):
            return False
        try:
            host.setflags(write=False)
            host.setflags(write=True)
        except (ValueError, AttributeError):
            return False
        return True

    def seal(self) -> None:
        """Take the host array's write flag down.

        THIS IS THE ONLY FAIL-CLOSED GUARD IN EITHER MODE. Every in-tree mutation
        of a mirrored volume is ``arr = getattr(fields, name)`` followed by an
        in-place write, and no property getter can tell a read from a
        read-modify-write — so unsealing on read would make every one of those
        silent. Sealed always, and every write site takes
        :meth:`Residency.acquire_write`; a site that does not raises
        ``ValueError: assignment destination is read-only`` the first time it runs.
        It does NOT close a write through a view taken before the seal — see
        ``Residency.acquire_write``'s note and the plan's section 4.7.
        """
        if self.constant or self.sealed:
            return
        self.host.setflags(write=False)
        self.sealed = True

    def unseal(self) -> None:
        """Put the host array's write flag back up."""
        if not self.sealed:
            return
        self.host.setflags(write=True)
        self.sealed = False

    def index_view(self) -> Any:
        """The tensor as an index target: itself for float32, its (N, 2) real view for complex64.

        MPS HAS NO COMPLEX INDEX OPS. ``index_fill_(): Complex types are yet not
        supported`` is what the first complex family under a hold raised
        (``complex_nobloch_2d``, 2026-09-23), and ``index_copy_``/``index_select`` sit
        behind the same gap. Indexing the interleaved float32 view on dim 0 moves the
        same cells, both parts, bit for bit -- it is data movement, not arithmetic.
        """
        import torch  # noqa: PLC0415

        return torch.view_as_real(self.tensor) if self.tensor.is_complex() else self.tensor

    def _numpy_dtype(self) -> Any:
        import numpy as np  # noqa: PLC0415

        return np.float32 if self.dtype is None else self.dtype

    def sync_in(self) -> None:
        """Host -> device, into the existing allocation.

        LIFTS ITS OWN SEAL for the duration, like :meth:`sync_out`, and for a
        narrower reason: nothing here writes the host, but ``ascontiguousarray``
        returns the SAME array when it is already contiguous and correctly typed,
        and ``torch.from_numpy`` on a non-writeable array emits "PyTorch does not
        support non-writable tensors ... writing to this tensor will result in
        undefined behavior". The copy is device-bound and never writes back, so the
        warning is spurious — but a spurious warning on the launch path trains
        readers to ignore the real one, and silencing it by filter would silence the
        real one too.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        was_sealed = self.sealed
        if was_sealed:
            self.host.setflags(write=True)
        try:
            flat = np.ascontiguousarray(self.host, dtype=self._numpy_dtype()).reshape(-1)
            self.tensor.copy_(torch.from_numpy(flat))
        finally:
            if was_sealed:
                self.host.setflags(write=False)

    def sync_out(self) -> None:
        """Device -> host, IN PLACE, so the engine's references stay valid.

        LIFTS ITS OWN SEAL. ``self.host[...] = ...`` is a host write like any
        other, so a sealed mirror would refuse the very copy that makes it clean
        again; the flag goes down again afterwards so the guard survives the sync.
        """
        if self.constant:
            return
        was_sealed = self.sealed
        if was_sealed:
            self.unseal()
        try:
            self.host[...] = self.tensor.cpu().numpy().reshape(self.host.shape)
        finally:
            if was_sealed:
                self.seal()

    def differing_words(self) -> int:
        """How many float32 WORDS the device copy and the host array disagree on.

        uint32 word equality, never ``allclose``: this is the residency INVARIANT
        check, and a tolerance here would hide exactly the stale-mirror drift it
        exists to catch. A complex64 mirror is compared as its interleaved float32
        words — two per cell — for the same reason every gate compares words.
        """
        import numpy as np  # noqa: PLC0415

        device = self.tensor.cpu().numpy().reshape(-1)
        host = np.ascontiguousarray(self.host, dtype=self._numpy_dtype()).reshape(-1)
        return int(np.count_nonzero(device.view(np.uint32) != host.view(np.uint32)))


# ---------------------------------------------------------------------------
# The sparse-door kernels
# ---------------------------------------------------------------------------

#: The three sparse doors as Metal kernels: one thread per index, ``width`` floats
#: per row (1 for a float32 mirror, 2 for the ``(N, 2)`` real view a complex64
#: mirror indexes through -- see :meth:`Mirror.index_view`). DATA MOVEMENT ONLY:
#: no arithmetic, so the words written are the words given, and duplicates in an
#: index set are unordered exactly as ``index_copy_`` left them (the callers'
#: tables are unique by construction).
#:
#: WHY NOT THE TORCH INDEX OPS THEY REPLACE, measured on this host (2026-09-24,
#: 160^3 float32 volume, ``parity/meep_gpu/results/round_2026-09-24_night_drivers/
#: bench_scatter_sizes.json``): ``index_copy_`` is SERIAL in the index count --
#: 1.2 ms at 1 cell, 3.7 ms at 160, 21 ms at 1,600 and 338 ms at a 25,600-cell
#: face, ~13 us per index -- so the fold fills' face copy cost a third of a second
#: a call; ``index_fill_`` pays a fixed 0.2-0.3 ms a call whatever the size, which
#: over the nine wall faces ``zero_metal_B``/``zero_metal_D`` clear each step was
#: 3.7 of a ~9 ms ``pml_3d`` step at 4.1M cells (``profile_doors_res40_shipped``).
#: These kernels read 0.07-0.13 ms at every size measured.
#:
#: COMPILED THROUGH ``torch.mps.compile_shader`` DIRECTLY, not :func:`compile_source`.
#: A door's operands are an index tensor and a payload, never a registered mirror,
#: so a :class:`LaunchAudit` watching the library cache would report a foreign
#: operand on every held step; and a door is not a sub-step launch, so it must not
#: be counted as one.
_DOOR_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void door_scatter_copy(device float* rows          [[buffer(0)]],
                              device const long* index    [[buffer(1)]],
                              device const float* values  [[buffer(2)]],
                              constant uint& n            [[buffer(3)]],
                              constant uint& width        [[buffer(4)]],
                              uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    ulong row = ulong(index[g]);
    for (uint w = 0; w < width; ++w) {
        rows[row * width + w] = values[ulong(g) * width + w];
    }
}

kernel void door_scatter_sub(device float* rows           [[buffer(0)]],
                             device const long* index     [[buffer(1)]],
                             device const float* values   [[buffer(2)]],
                             constant uint& n             [[buffer(3)]],
                             constant uint& width         [[buffer(4)]],
                             uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    ulong row = ulong(index[g]);
    for (uint w = 0; w < width; ++w) {
        // ONE float32 subtraction per word, the array path's ``current - values``
        // (a complex64 row is two words, and complex subtraction is componentwise).
        rows[row * width + w] = rows[row * width + w] - values[ulong(g) * width + w];
    }
}

kernel void door_scatter_zero(device float* rows          [[buffer(0)]],
                              device const long* index    [[buffer(1)]],
                              constant uint& n            [[buffer(2)]],
                              constant uint& width        [[buffer(3)]],
                              uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    ulong row = ulong(index[g]);
    for (uint w = 0; w < width; ++w) {
        rows[row * width + w] = 0.0f;
    }
}

// ``door_scatter_zero`` for up to three arrays in ONE launch (ZERO_DOOR_BUFFERS).
// ``index`` is the concatenation of each array's row set; ``segments`` is
// [end0, end1, end2, width0, width1, width2], so thread g clears row index[g] of
// the array whose segment holds g, ``width`` floats per row. An unused slot is
// bound to rows0 with an empty segment, so no thread ever selects it.
kernel void door_scatter_zero_multi(device float* rows0         [[buffer(0)]],
                                    device float* rows1         [[buffer(1)]],
                                    device float* rows2         [[buffer(2)]],
                                    device const long* index    [[buffer(3)]],
                                    device const uint* segments [[buffer(4)]],
                                    constant uint& n            [[buffer(5)]],
                                    uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    uint s = (g < segments[0]) ? 0u : ((g < segments[1]) ? 1u : 2u);
    device float* rows = (s == 0u) ? rows0 : ((s == 1u) ? rows1 : rows2);
    uint width = segments[3u + s];
    ulong row = ulong(index[g]);
    for (uint w = 0; w < width; ++w) {
        rows[row * width + w] = 0.0f;
    }
}
"""

#: How many arrays ``door_scatter_zero_multi`` clears in one launch -- its row-buffer
#: count, and the largest group :meth:`Residency.zero_index_sets` accepts. Chosen from
#: the largest pass that sends one: ``stepping._zero_metal`` clears the faces of ONE
#: field's three components (``D_COMPONENTS`` or ``B_COMPONENTS``), so a pass names at
#: most three arrays however many walls it clears. A larger group is REFUSED by name
#: rather than split: a caller that needs more has changed shape and should say so.
ZERO_DOOR_BUFFERS = 3

#: The compiled door library, one per process. Not :data:`_LIBRARY_CACHE`: that
#: cache is what a witness measures for growth between steps, and a door library
#: compiled lazily on the first held door would read as a kernel compile.
_DOOR_LIBRARY: Dict[str, Any] = {}


def _door_kernels() -> Any:
    library = _DOOR_LIBRARY.get("mps")
    if library is None:
        import torch  # noqa: PLC0415

        library = torch.mps.compile_shader(_DOOR_SOURCE)
        _DOOR_LIBRARY["mps"] = library
    return library


def _rows_width(view: Any) -> int:
    """Floats per indexed row of an index view: 1 for float32, 2 for the complex real view."""
    return int(view.shape[1]) if view.dim() == 2 else 1


def _scatter_rows(view: Any, index: Any, values_dev: Any) -> None:
    """``view[index] = values_dev`` on whichever device holds the view.

    The Metal door on MPS; ``index_copy_`` elsewhere (the CPU twin a gate builds),
    which is the op this replaces and whose words it reproduces.
    """
    n = int(index.numel())
    if n == 0:
        return
    if view.device.type != "mps":
        view.index_copy_(0, index, values_dev)
        return
    _door_kernels().door_scatter_copy(view, index, values_dev.contiguous(), n,
                                      _rows_width(view), threads=n)


def _subtract_rows(view: Any, index: Any, values_dev: Any) -> None:
    """``view[index] = view[index] - values_dev`` on whichever device holds the view.

    The Metal door on MPS; ``index_add_`` of the negated payload elsewhere (the CPU
    twin a gate builds): IEEE subtraction IS addition of the negation, word for word,
    signed zeros included. Duplicate indices are the caller's to refuse, as for
    :func:`_scatter_rows`.
    """
    n = int(index.numel())
    if n == 0:
        return
    if view.device.type != "mps":
        view.index_add_(0, index, -values_dev)
        return
    _door_kernels().door_scatter_sub(view, index, values_dev.contiguous(), n,
                                     _rows_width(view), threads=n)


def _zero_rows(view: Any, index: Any) -> None:
    """``view[index] = 0`` on whichever device holds the view; see :func:`_scatter_rows`."""
    n = int(index.numel())
    if n == 0:
        return
    if view.device.type != "mps":
        view.index_fill_(0, index, 0)
        return
    _door_kernels().door_scatter_zero(view, index, n, _rows_width(view), threads=n)


def _zero_rows_multi(views: Sequence[Any], index: Any, segments: Any,
                     ends: Sequence[int]) -> None:
    """``views[k][index[segment k]] = 0`` for every k, in ONE launch on MPS.

    ``index`` concatenates the views' row sets and ``ends`` is where each one stops
    (``segments`` is the same table on the device, followed by the widths; see
    ``door_scatter_zero_multi``). ``index_fill_`` per view elsewhere -- the CPU twin a
    gate builds -- which is :func:`_zero_rows` segment by segment.
    """
    n = int(index.numel())
    if n == 0:
        return
    if views[0].device.type != "mps":
        start = 0
        for view, end in zip(views, ends):
            if end > start:
                view.index_fill_(0, index[start:end], 0)
            start = end
        return
    bound = tuple(views) + (views[0],) * (ZERO_DOOR_BUFFERS - len(views))
    _door_kernels().door_scatter_zero_multi(*bound, index, segments, n, threads=n)


class Residency:
    """The persistent device mirrors one composition's plans share.

    THE POINT IS SHARING. ``step_B`` writes ``Bx`` and ``update_H`` reads it at the
    same index; if each plan mirrored ``Bx`` privately the second launch would read
    the first one's stale bytes and the field would be smooth, plausible and wrong.
    So a mirror is registered once, by name, and every plan that touches that volume
    binds the SAME tensor — which is why a plan built with no residency is a
    coverage refusal rather than a plan with a private copy.

    THE COST THIS BUYS BACK, measured on this host: a host-to-device-and-back round
    trip of the six in-place volumes costs 12x-64x the kernel it surrounds. That is
    the number the design exists for; it is not an argument.

    Registering the same NAME twice with a different host array is refused. Two names
    may legitimately share one host array — ``set_isotropic_epsilon_volume`` hands
    back three aliases of one volume — and that is allowed, because those are constant
    mirrors nothing writes.
    """

    __slots__ = ("device", "_mirrors", "syncs_in", "syncs_out", "held", "hoisted",
                 "generation", "copies_in", "copies_out", "flushes",
                 "sparse_gathers", "sparse_scatters", "sparse_zeros", "zero_launches",
                 "_index_cache", "_zero_set_cache", "__weakref__")

    def __init__(self, device: str = "mps") -> None:
        self.device = device
        self._mirrors: Dict[str, Mirror] = {}
        self.syncs_in = 0
        self.syncs_out = 0
        #: The names this composition has ARMED for holding, as a frozenset. Empty
        #: by default, and while it is empty every method below behaves exactly as
        #: it did before the ownership machinery existed — the unheld path is not a
        #: new path. A composer arms a hold only for mirrors ``coverage``'s
        #: ``residency_reasons`` returns no refusal for AND whose every owner the
        #: barrier actually reaches.
        self.held: frozenset = frozenset()
        #: The CONSTANTS this composition uploaded once at arm time and stops
        #: re-uploading. ``Mirror.sync_in`` has no ``constant`` check where
        #: ``sync_out`` has one, so on ``pml_2d`` 21 of 45 host->device copies a
        #: launch — 42 of 138 a step, 30% of the copy COUNT — re-upload the three
        #: inverse-epsilon volumes and eighteen PML coefficient vectors that no
        #: kernel ever writes. Hoisting them is the cheapest copy reduction in the
        #: package and it needs no ownership tracking, because the device never
        #: writes them.
        #:
        #: IT DOES NEED AN INVALIDATION HOOK, and this is it: a hoisted constant is
        #: SEALED like any held mirror, so ``set_epsilon_volumes`` or
        #: ``set_isotropic_epsilon_volume`` rewriting a coefficient volume after
        #: plan freeze RAISES instead of leaving the device on stale material for
        #: the rest of the run. Going through :meth:`acquire_write` marks it
        #: ``HOST_OWNED``, which puts it back in the next ``sync_in``.
        self.hoisted: frozenset = frozenset()
        #: Bumped on every state transition, so an instrument can tell "nothing
        #: moved" from "moved and moved back".
        self.generation = 0
        #: COPIES, not calls. ``syncs_in``/``syncs_out`` count bracket invocations;
        #: these count the per-mirror transfers those invocations actually
        #: performed, which is the number the 0.217 ms/copy cost attaches to and
        #: the calibration set every future campaign now gets for free.
        self.copies_in = 0
        self.copies_out = 0
        self.flushes = 0
        #: SPARSE transfers, counted separately from whole-volume copies because
        #: they are a different cost class entirely: measured 2026-09-21, a gather
        #: of 8 cells costs 0.269 ms and one of 64 cells 0.258 -- latency, not
        #: bandwidth -- while a whole-volume copy at 7M cells is 28 MB. The
        #: publication ladder of 2026-09-22 showed why the distinction is the whole
        #: result: with WHOLE-volume acquires a held step moved ~14 volumes a step
        #: and read 60 ms at 7M cells.
        self.sparse_gathers = 0
        self.sparse_scatters = 0
        #: One per (array, index set) cleared -- what one :meth:`zero_index` call
        #: counts, and what :meth:`zero_index_sets` counts per pair, so a pass's
        #: figure does not change with how many launches carried it (nine a
        #: ``pml_3d`` step either way).
        self.sparse_zeros = 0
        #: Device zero OPERATIONS issued by the two zero doors: one per
        #: :meth:`zero_index` and one per :meth:`zero_index_sets` group (a Metal
        #: launch each on MPS). The batched wall clear is read off this counter.
        self.zero_launches = 0
        #: Device index tensors, keyed by the id() of the host index array and
        #: holding a reference to it so the id cannot be reused while cached. The
        #: deposit tables and the source point lists are built once and reused
        #: every step, so this hits every step after the first.
        self._index_cache: Dict[int, Tuple[Any, Any]] = {}
        #: :meth:`zero_index_sets`' launch tables, keyed by the ids of the (array,
        #: index) pairs it was sent and holding both, like :attr:`_index_cache`.
        self._zero_set_cache: Dict[Tuple[Tuple[int, int], ...], Any] = {}

    # -- resolution: ownership is keyed on HOST-ARRAY IDENTITY ----------------

    def _targets(self, target: Any) -> Tuple[Mirror, ...]:
        """The mirrors one acquire call refers to, resolved by name OR by array.

        NAME-KEYED OWNERSHIP IS WRONG FOR THE WELD FAMILIES and that is measured,
        not feared: ``offdiag_weld_common.py`` does
        ``setattr(self.fields, name, self.rotated[name])`` INSIDE ``run``, every
        launch, so after an odd number of launches the mirror registered as ``Dx``
        shadows the plan's twin and ``fields.Dx`` is the array registered as
        ``weld_scratch:Dx``. Bookkeeping keyed on the NAME names the wrong array
        every other launch.

        Resolving an ARRAY by identity is rotation-proof — it follows the object
        through any number of rebinds and needs no rebind hook on the hot path —
        and it is the lookup the weld code already performs through
        :meth:`tensor_for_host`. So a caller that has the array in hand (every
        engine write site does) passes the array; a caller that only has a name
        passes the name and accepts that a rotating family must use the array form.
        Several names may legitimately share one array (``set_isotropic_epsilon_volume``
        hands back three aliases), so this returns a TUPLE and every state move
        applies to all of them.
        """
        if isinstance(target, str):
            mirror = self._mirrors.get(target)
            if mirror is None:
                raise KeyError(
                    f"{target!r} is not a registered mirror; an acquire on an "
                    f"unregistered name cannot be answered, and guessing that it "
                    f"needs no sync is the silent-wrong-answer case this layer "
                    f"exists to refuse")
            return (mirror,)
        found = tuple(m for m in self._mirrors.values() if m.host is target)
        if not found:
            raise KeyError(
                "that array is not mirrored by this residency; see _targets on why "
                "an unmirrored acquire is refused rather than ignored")
        return found

    def _by_host_identity(self, array: Any) -> Any:
        """The mirror shadowing ``array``, or None — the hot-path identity lookup.

        Unlike :meth:`_targets` this never raises and never builds a tuple: it is
        called from the read barrier on every barriered attribute fetch, so it is a
        linear scan kept deliberately small by the fact that a composition's
        registry is tens of mirrors, and it answers None for an array this
        residency does not hold rather than treating that as an error.
        """
        for mirror in self._mirrors.values():
            if mirror.host is array:
                return mirror
        return None

    def state_of(self, target: Any) -> str:
        """The ownership state of one mirror, by name or by array."""
        return self._targets(target)[0].state

    def by_state(self) -> Dict[str, Tuple[str, ...]]:
        """Every mirror name grouped by ownership state — the gate's set form.

        The falsifiable check under a hold is NOT ``verify() == {}``: the host is
        stale for every ``DEVICE_OWNED`` mirror BY DESIGN, so blanket equality goes
        red on correct code. It is instead "the set that differs equals the set the
        bookkeeping says should differ", which is strictly stronger because it also
        catches OVER-syncing, which equality cannot.
        """
        out: Dict[str, list] = {state: [] for state in OWNERSHIP_STATES}
        for name, mirror in self._mirrors.items():
            out[mirror.state].append(name)
        return {state: tuple(sorted(names)) for state, names in out.items()}

    # -- the host's two doors ------------------------------------------------

    def acquire_read(self, target: Any) -> Any:
        """Make the host copy current for a host READ, and return the array.

        Leaves the mirror SEALED. A getter cannot tell a read from a
        read-modify-write, so unsealing here would make every
        ``arr = getattr(f, name); arr += ...`` in the tree silent. Callers that
        intend to write say so with :meth:`acquire_write`.
        """
        mirrors = self._targets(target)
        for mirror in mirrors:
            if mirror.state == DEVICE_OWNED:
                mirror.sync_out()
                self.copies_out += 1
                mirror.state = CLEAN_BOTH
                self.generation += 1
        return mirrors[0].host

    def acquire_write(self, target: Any) -> Any:
        """Declare a host WRITE, unseal, and return the writable array.

        Order matters and the wrong order loses data: a ``DEVICE_OWNED`` mirror is
        synced OUT first, because a read-modify-write needs the device's words
        before it adds to them, and every in-tree site is a read-modify-write
        (``array -= ...``, ``views[0][linear] = restored``, ``array.fill(0)``)
        rather than a whole-volume overwrite. Then the state moves to
        ``HOST_OWNED`` so the next launch that declares this mirror a read syncs it
        back in.

        THE ONE HOLE, MEASURED: sealing a parent array does not seal a view a
        caller took EARLIER. Both in-tree stashers are safe — the deposit repair
        saves fancy-index copies and the magnetic backup is ``copy=True`` — so the
        residual is caller code that stashes a slice and writes through it later.
        Only :meth:`verify` catches that, which is why the flush-then-verify form
        stays in the gate and in debug mode.
        """
        mirrors = self._targets(target)
        for mirror in mirrors:
            if mirror.state == DEVICE_OWNED:
                mirror.sync_out()
                self.copies_out += 1
            if mirror.sealed:
                mirror.host.setflags(write=True)
                mirror.sealed = False
            mirror.state = HOST_OWNED
            self.generation += 1
        return mirrors[0].host

    def write_hook(self, array: Any) -> Any:
        """The tolerant form of :meth:`acquire_write`, for ``host_writes.acquire``.

        The engine's declaration point calls this on EVERY array a site writes,
        including arrays this residency has never mirrored — a scratch buffer, a
        volume belonging to a different composition, a plain temporary. That is not
        an error there, so this returns quietly instead of refusing.

        :meth:`acquire_write` keeps refusing an unmirrored target, and the two
        behaviours are deliberately different: a caller that NAMES a mirror and is
        wrong about it has made a mistake worth raising on, while a blanket
        declaration hook that raised on every unmirrored array would make the engine
        un-runnable the moment anything held.
        """
        try:
            return self.acquire_write(array)
        except KeyError:
            return array

    # -- sparse transport: the seam passes touch a few cells, so move a few cells --

    def _device_index(self, linear: Any) -> Any:
        """The device int64 tensor for a host linear-index array, cached by identity."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        hit = self._index_cache.get(id(linear))
        if hit is not None and hit[0] is linear:
            return hit[1]
        flat = np.ascontiguousarray(linear, dtype=np.int64).reshape(-1)
        tensor = torch.from_numpy(flat).to(torch.device(self.device))
        self._index_cache[id(linear)] = (linear, tensor)
        return tensor

    def gather(self, array: Any, linear: Any) -> Any:
        """The values at C-order flat indices ``linear`` -- from the device if it owns them.

        Returns a fresh host array of the mirror's dtype, exactly what
        ``array.reshape(-1)[linear]`` would return from a current host array. Under a
        hold the host array is stale by design, so the read goes to the tensor; when
        the host is current (CLEAN_BOTH or HOST_OWNED) the host copy is read
        directly, which is free.
        """
        mirror = self._by_host_identity(array)
        if mirror is None or mirror.state != DEVICE_OWNED:
            source = array if mirror is None else mirror.host
            return source.reshape(-1)[linear]
        self.sparse_gathers += 1
        return self._to_host(mirror, mirror.index_view().index_select(0, self._device_index(linear)))

    def scatter(self, array: Any, linear: Any, values: Any) -> None:
        """Write ``values`` at flat indices ``linear`` -- to whichever side is current.

        THE RULE, and it keeps both invariants without a whole-volume copy:

        * DEVICE_OWNED: write the tensor only. The host stays stale at those cells
          exactly as it is stale everywhere else, and stays sealed; a later host read
          acquires the volume as usual. No state change -- the device still owns it.
        * otherwise (CLEAN_BOTH / HOST_OWNED): write BOTH the host array and the
          tensor, so a clean mirror stays clean and a host-owned one stays exactly as
          far ahead as it was. Writing only the host would force a whole-volume
          upload at the next launch, which is the cost this method exists to remove.

        ``linear`` must not repeat an index: NumPy's fancy assignment is last-wins and
        a device ``index_copy_`` with duplicates is unordered, so a duplicate would be
        the one place the two could disagree. The seam tables are unique by
        construction (one entry per cell) and the caller checks once per table.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        mirror = self._by_host_identity(array)
        if mirror is None:
            array.reshape(-1)[linear] = values
            return
        payload, payload_dev = self._payload(mirror, values)
        index = self._device_index(linear)
        if mirror.state != DEVICE_OWNED:
            was_sealed = mirror.sealed
            if was_sealed:
                mirror.host.setflags(write=True)
            try:
                mirror.host.reshape(-1)[linear] = payload
            finally:
                if was_sealed:
                    mirror.host.setflags(write=False)
        _scatter_rows(mirror.index_view(), index, payload_dev)
        self.sparse_scatters += 1

    def subtract_index(self, array: Any, linear: Any, values: Any) -> None:
        """``array.reshape(-1)[linear] -= values`` -- the difference formed where the words live.

        The source deposit's shape (``sources._sparse_subtract``): a few cells, one
        subtraction each. Gathering the cells to the host, subtracting there and
        scattering back was word-identical but cost a blocking device->host read a
        step -- under a held residency the one host wait left after the per-launch
        drain went (2.6-2.7 ms of a 4.6 ms ``pml_3d`` step at 4.1M cells, measured
        2026-09-24, ``results/round_2026-09-24_night_drivers/profile_held_step_res40_patched.json``).
        Forming the difference on the device removes the wait; the ownership rule is
        :meth:`scatter`'s (DEVICE_OWNED: tensor only; otherwise both sides, so the
        state need not move). The array path is untouched: with no residency this is
        NumPy's own in-place fancy subtraction.
        """
        mirror = self._by_host_identity(array)
        if mirror is None:
            array.reshape(-1)[linear] -= values
            return
        payload, payload_dev = self._payload(mirror, values)
        index = self._device_index(linear)
        if mirror.state != DEVICE_OWNED:
            self._host_write(mirror, linear, mirror.host.reshape(-1)[linear] - payload)
        _subtract_rows(mirror.index_view(), index, payload_dev)
        self.sparse_scatters += 1

    def zero_index(self, array: Any, linear: Any) -> None:
        """Write zero at flat indices ``linear``, same ownership rule as :meth:`scatter`.

        One slab or row range of one array -- the cylindrical axis's clears
        (``host_writes.zero`` / ``zero_rows``); a whole wall-clear pass goes through
        :meth:`zero_index_sets` instead -- and a seam write with no arithmetic at all,
        so it moves NOTHING across the bus: the index lives on the device and the
        fill is a single kernel (``door_scatter_zero``; ``index_fill_`` paid a fixed
        0.2-0.3 ms a call on this host). Bit-exact by construction; zero is zero.
        """
        mirror = self._by_host_identity(array)
        if mirror is None:
            array.reshape(-1)[linear] = 0
            return
        if mirror.state != DEVICE_OWNED:
            was_sealed = mirror.sealed
            if was_sealed:
                mirror.host.setflags(write=True)
            try:
                mirror.host.reshape(-1)[linear] = 0
            finally:
                if was_sealed:
                    mirror.host.setflags(write=False)
        _zero_rows(mirror.index_view(), self._device_index(linear))
        self.sparse_zeros += 1
        self.zero_launches += 1

    def zero_index_sets(self, pairs: Sequence[Tuple[Any, Any]]) -> None:
        """:meth:`zero_index` for every ``(array, linear)`` in ``pairs`` -- ONE device launch.

        A whole ``_zero_metal`` pass at once: on ``pml_3d`` that is six faces of three
        D arrays, or three faces of three B arrays, which :meth:`zero_index` cleared in
        one launch per face -- nine ``door_scatter_zero`` launches a step, each paying
        the fixed host cost of a door call. Here the faces are merged per array (their
        union, so a cell two walls share is written once) and cleared by
        ``door_scatter_zero_multi`` in one launch. Zero is zero: the words are those
        of the per-face clears in any order.

        The ownership rule is :meth:`zero_index`'s, array by array: DEVICE_OWNED is
        written on the device only; any other state on BOTH sides, the host seal
        lifted and restored around the write, so no state moves. An array this
        residency does not mirror is cleared on the host, as :meth:`zero_index` does,
        and is not part of the launch.

        More than :data:`ZERO_DOOR_BUFFERS` mirrored arrays in one call is REFUSED
        before anything is written.
        """
        mirrored = []    # (array, linear, slot) for every pair this residency mirrors
        unmirrored = []  # (array, linear) for every pair it does not
        mirrors = []     # one per distinct mirrored array, in first-appearance order
        slot_of: Dict[int, int] = {}
        for array, linear in pairs:
            slot = slot_of.get(id(array))
            if slot is None:
                mirror = self._by_host_identity(array)
                if mirror is None:
                    unmirrored.append((array, linear))
                    continue
                slot = slot_of[id(array)] = len(mirrors)
                mirrors.append(mirror)
            mirrored.append((array, linear, slot))
        if len(mirrors) > ZERO_DOOR_BUFFERS:
            raise ValueError(
                f"zero_index_sets was sent {len(mirrors)} mirrored arrays in one call; "
                f"door_scatter_zero_multi binds {ZERO_DOOR_BUFFERS} row buffers "
                f"(ZERO_DOOR_BUFFERS, sized from stepping._zero_metal's largest pass), "
                f"so this group is refused rather than split")
        for array, linear in unmirrored:
            array.reshape(-1)[linear] = 0
        if not mirrors:
            return
        views = tuple(mirror.index_view() for mirror in mirrors)
        _, unions, ends, index, segments = self._zero_set_table(mirrored, views)
        for mirror, union in zip(mirrors, unions):
            if mirror.state != DEVICE_OWNED:
                self._host_write(mirror, union, 0)
        _zero_rows_multi(views, index, segments, ends)
        self.sparse_zeros += len(mirrored)
        self.zero_launches += 1

    def _zero_set_table(self, mirrored: Sequence[Tuple[Any, Any, int]],
                        views: Sequence[Any]) -> Tuple[Any, ...]:
        """The launch table of one :meth:`zero_index_sets` call, built once per pair set.

        Per array slot, the UNION of its index sets (sorted, each cell once); their
        concatenation as the device index; the running ends; and the device
        ``segments`` table ``door_scatter_zero_multi`` reads -- the ends padded to
        :data:`ZERO_DOOR_BUFFERS` with the total, then the widths padded with 1.
        Cached by the ids of the pairs and holding them, so an id cannot be reused
        while its entry lives; a rotating weld family alternates between a bounded
        few pair sets, and the cache is dropped whole if it ever grows past 64.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        key = tuple((id(array), id(linear)) for array, linear, _ in mirrored)
        widths = tuple(_rows_width(view) for view in views)
        table = self._zero_set_cache.get(key)
        if table is not None and table[0][0] == widths and all(
                held_array is array and held_linear is linear
                for (held_array, held_linear), (array, linear, _)
                in zip(table[0][1], mirrored)):
            return table
        parts = [[] for _ in views]
        for _, linear, slot in mirrored:
            parts[slot].append(np.asarray(linear, dtype=np.int64).reshape(-1))
        unions = [np.unique(np.concatenate(chunks)) for chunks in parts]
        ends = tuple(int(end) for end in np.cumsum([union.size for union in unions]))
        spare = ZERO_DOOR_BUFFERS - len(ends)
        segments = np.array(ends + (ends[-1],) * spare + widths + (1,) * spare,
                            dtype=np.int32)
        device = torch.device(self.device)
        table = ((widths, tuple((array, linear) for array, linear, _ in mirrored)),
                 unions, ends, torch.from_numpy(np.concatenate(unions)).to(device),
                 torch.from_numpy(segments).to(device))
        if len(self._zero_set_cache) >= 64:
            self._zero_set_cache.clear()
        self._zero_set_cache[key] = table
        return table

    def copy_index(self, destination: Any, dst_linear: Any, source: Any, src_linear: Any,
                   scale: Any = 1) -> None:
        """``destination.flat[dst] = scale * source.flat[src]`` -- on the device when it owns both.

        THE FOLD FILLS' SHAPE: one wall face copied from another face of the same (or a
        sibling) array, times a parity of +1 or -1. Under a hold this is two device
        index ops and no transfer at all; the whole-volume form it replaces synced
        both faces' volumes down and back up. Bit-exact by construction for a +-1
        scale (negation is exact), and for a float scale it is one rounding of one
        multiply, which is what NumPy's ``phase * face`` performs too.

        Ownership: if the destination is DEVICE_OWNED the device is written only;
        otherwise both sides, so the state need not move. The source is read from
        whichever side is current for it.
        """
        import numpy as np  # noqa: PLC0415

        dst = self._by_host_identity(destination)
        src = self._by_host_identity(source)
        if dst is None:
            values = self.gather(source, src_linear) if src is not None else source.reshape(-1)[src_linear]
            destination.reshape(-1)[dst_linear] = scale * values
            return
        if src is not None and src.state == DEVICE_OWNED:
            values_dev = src.index_view().index_select(0, self._device_index(src_linear))
            if scale != 1 and src.tensor.is_complex():
                # NumPy's ``phase * face`` on a complex face is a complex multiply by
                # (phase + 0j), and its imaginary part ``a*0 + b*phase`` can differ from
                # ``b*phase`` in the sign of a zero. The word must match the array
                # path's, so the one multiply is done where the array path does it.
                self.scatter(destination, dst_linear, scale * self._to_host(src, values_dev))
                return
            if scale != 1:
                values_dev = values_dev * scale
            if dst.state != DEVICE_OWNED:
                self._host_write(dst, dst_linear, self._to_host(src, values_dev))
            _scatter_rows(dst.index_view(), self._device_index(dst_linear), values_dev)
            self.sparse_scatters += 1
            return
        host_source = src.host if src is not None else source
        values = host_source.reshape(-1)[src_linear]
        if scale != 1:
            values = scale * values
        self.scatter(destination, dst_linear, values)

    def add_scaled_index(self, destination: Any, dst_linear: Any, source: Any,
                         src_linear: Any, scale: Any) -> None:
        """``destination.flat[dst] += scale * source.flat[src]``, same ownership rules.

        The cylindrical axis's m=0 correction: ``Dz[axis] += k * Hy[axis]``. NumPy
        forms ``k * Hy`` (one rounding) then adds (one rounding); the device forms the
        same product with ``mul`` then ``index_add_`` -- two separate ops, so no fused
        multiply-add can contract them, and the words match.
        """
        dst = self._by_host_identity(destination)
        src = self._by_host_identity(source)
        current = self.gather(destination, dst_linear) if dst is not None else destination.reshape(-1)[dst_linear]
        values = self.gather(source, src_linear) if src is not None else source.reshape(-1)[src_linear]
        self.scatter(destination, dst_linear, current + scale * values) if dst is not None \
            else destination.reshape(-1).__setitem__(dst_linear, current + scale * values)

    @staticmethod
    def _to_host(mirror: Any, values_dev: Any) -> Any:
        """Device values read through :meth:`Mirror.index_view`, as the mirror's own dtype."""
        import numpy as np  # noqa: PLC0415

        out = values_dev.cpu().numpy()
        if mirror.tensor.is_complex():
            out = np.ascontiguousarray(out).view(mirror._numpy_dtype()).reshape(-1)
        return out

    @staticmethod
    def _payload(mirror: Any, values: Any) -> Tuple[Any, Any]:
        """``values`` as the mirror's dtype (host) and as the index view's dtype (device)."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        payload = np.ascontiguousarray(values, dtype=mirror._numpy_dtype()).reshape(-1)
        view = payload.view(np.float32).reshape(-1, 2) if np.iscomplexobj(payload) else payload
        return payload, torch.from_numpy(np.ascontiguousarray(view)).to(mirror.tensor.device)

    def _host_write(self, mirror: Any, linear: Any, values: Any) -> None:
        """Write the host side of a mirror at flat indices, lifting the seal around it."""
        was_sealed = mirror.sealed
        if was_sealed:
            mirror.host.setflags(write=True)
        try:
            mirror.host.reshape(-1)[linear] = values
        finally:
            if was_sealed:
                mirror.host.setflags(write=False)

    def rebind(self, name: str, host: Any) -> Any:
        """Point an existing mirror NAME at a different host array, deliberately.

        The rotating families do not need this — :meth:`_targets` follows the array
        by identity — but a reporting surface keyed on names (``gate.json``'s
        ``mirror_names``, ``verify()``) is wrong after a rotation unless the
        registry is told. This is the explicit, plan-owned move that
        :meth:`mirror`'s "one name, one array" refusal otherwise forbids, so it
        carries the same dtype and shape checks and refuses silently-lossy swaps.
        """
        import numpy as np  # noqa: PLC0415

        mirror = self._mirrors.get(name)
        if mirror is None:
            raise KeyError(f"cannot rebind {name!r}: it is not registered")
        if np.dtype(mirror._numpy_dtype()) != np.dtype(host.dtype):
            raise ValueError(
                f"rebinding {name!r} from {np.dtype(mirror._numpy_dtype())} to "
                f"{np.dtype(host.dtype)} would resize the dispatch; one name binds "
                f"one width")
        if host.size != mirror.host.size:
            raise ValueError(
                f"rebinding {name!r} from {mirror.host.size} to {host.size} words "
                f"would leave the device tensor a different length than the host")
        if mirror.sealed:
            mirror.unseal()
        mirror.host = host
        mirror.state = HOST_OWNED
        self.generation += 1
        return host

    def flush(self, names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
        """Device -> host for every DEVICE_OWNED mirror; returns what moved.

        THE HOLD'S ESCAPE HATCH, and three separate rules need it: the re-plan
        (a new plan build mirrors host arrays that are stale by however many steps
        the device has been holding), the magnetic-synchronization consult (the
        driver dispatches a half-step and then averages ON THE HOST), and the
        acceptance instruments (flush, then assert ``verify()`` empty over ALL
        names — a device write the machine believed clean is never flushed and so
        shows up).
        """
        moved = []
        for name in (self.names if names is None else tuple(names)):
            mirror = self._mirrors[name]
            if mirror.state == DEVICE_OWNED:
                mirror.sync_out()
                self.copies_out += 1
                mirror.state = CLEAN_BOTH
                moved.append(name)
        self.flushes += 1
        if moved:
            self.generation += 1
        return tuple(moved)

    def arm_hold(self, names: Sequence[str]) -> Tuple[str, ...]:
        """Arm a hold over ``names``, sealing each; returns the names REFUSED.

        Classifies rather than accepts wholesale, because the two kinds of mirror
        need different treatment and conflating them costs either correctness or
        copies:

        * a **non-constant** mirror joins :attr:`held` — its ownership is tracked
          through the three states and the device may own it between launches;
        * a **constant** joins :attr:`hoisted` — the device never writes it, so it
          needs no state machine, only to stop being re-uploaded every launch;
        * a name that is unregistered, or whose host array cannot round-trip its
          own write flag (:meth:`Mirror.sealable`), is REFUSED and returned, so the
          caller records a refusal by name instead of holding a volume it cannot
          guard. A hold with no seal is not a weaker hold, it is an unguarded one.

        Both classes are sealed. For the held set the seal is what makes a missed
        host write raise; for the hoisted set it is what makes a post-freeze
        material rewrite raise instead of silently stranding the device on old
        coefficients.
        """
        refused = []
        held = set(self.held)
        hoisted = set(self.hoisted)
        sealed_hosts = {id(m.host) for m in self._mirrors.values() if m.sealed}
        for name in names:
            mirror = self._mirrors.get(name)
            if mirror is None:
                refused.append(name)
                continue
            # AN ALIAS OF AN ALREADY-SEALED ARRAY IS HOLDABLE, and missing this
            # refused two real mirrors when it was first run:
            # ``set_isotropic_epsilon_volume`` hands back ONE array under three
            # names, so after ``inv_eps_Ex`` is sealed the array's write flag is
            # already down and ``sealable()`` — which probes by taking the flag
            # down and putting it back — reads the array as unsealable and refuses
            # ``inv_eps_Ey`` and ``inv_eps_Ez``. They are the same volume that was
            # just accepted. Identity is the question, not the flag's current value.
            if id(mirror.host) in sealed_hosts:
                mirror.sealed = True
                (hoisted if mirror.constant else held).add(name)
                continue
            if not mirror.sealable():
                refused.append(name)
                continue
            if mirror.constant:
                # Seal a constant through the same door, which needs the mirror to
                # not short-circuit on ``constant`` the way sync_out does.
                mirror.host.setflags(write=False)
                mirror.sealed = True
                sealed_hosts.add(id(mirror.host))
                hoisted.add(name)
                continue
            mirror.seal()
            sealed_hosts.add(id(mirror.host))
            held.add(name)
        self.held = frozenset(held)
        self.hoisted = frozenset(hoisted)
        self.generation += 1
        return tuple(refused)

    def release_hold(self) -> None:
        """Flush everything the device owns, unseal every mirror, drop the hold."""
        self.flush()
        for mirror in self._mirrors.values():
            if mirror.sealed:
                mirror.host.setflags(write=True)
                mirror.sealed = False
        # An alias may have been cleared through its partner above; the flag is
        # already up and the second setflags would be a no-op, so nothing more is
        # owed here. Stated rather than left to the reader because the aliased case
        # is what broke arm_hold.
        self.held = frozenset()
        self.hoisted = frozenset()
        self.generation += 1

    def mirror(self, name: str, host: Any, constant: bool = False,
               dtype: Any = None) -> Any:
        """Register (or fetch) the device mirror of one host volume.

        ``dtype`` defaults to float32 — the shipped real families' storage, and the
        behaviour this class carried while one family existed. A complex family
        passes ``numpy.complex64`` and gets a tensor that binds to ``float2*``.
        Passing a DIFFERENT dtype for a name already registered is refused for the
        same reason a different host array is: one name, one binding.

        A dtype outside :data:`BINDABLE_DTYPES` is refused BY NAME, before any
        allocation. See that constant for the measured reason: a complex128 mirror
        bound and LAUNCHED against a ``float2*`` signature without raising anything.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        resolved = np.float32 if dtype is None else np.dtype(dtype).type
        if np.dtype(resolved).name not in BINDABLE_DTYPES:
            raise ValueError(
                f"mirror {name!r} was asked for {np.dtype(resolved)}, which no "
                f"kernel in this package can bind: a Metal buffer is "
                f"{' or '.join(BINDABLE_DTYPES)} here (device float* / device "
                f"float2*). A wider element silently reinterprets — a complex128 "
                f"host array bound to a float2* signature LAUNCHES and returns "
                f"garbage rather than raising — so the width is refused at "
                f"registration rather than at the first wrong number")
        existing = self._mirrors.get(name)
        if existing is not None:
            if existing.host is not host:
                raise ValueError(
                    f"mirror {name!r} is already bound to a different host array; "
                    f"two arrays under one name is the aliasing defect this "
                    f"registry exists to refuse")
            if np.dtype(existing._numpy_dtype()) != np.dtype(resolved):
                raise ValueError(
                    f"mirror {name!r} is already bound as "
                    f"{np.dtype(existing._numpy_dtype())} and was asked for "
                    f"{np.dtype(resolved)}; one name binds one dtype, because the "
                    f"dispatch is sized from the tensor's element count and the two "
                    f"counts differ by a factor of two")
            return existing.tensor
        flat = np.ascontiguousarray(host, dtype=resolved).reshape(-1)
        tensor = torch.from_numpy(flat).to(torch.device(self.device))
        self._mirrors[name] = Mirror(name, host, tensor, constant, resolved)
        return tensor

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._mirrors))

    def tensor(self, name: str) -> Any:
        return self._mirrors[name].tensor

    def tensor_for_host(self, host: Any) -> Any:
        """The unique device mirror for ``host``, or ``None`` when absent.

        A plan-owned device pack may need to consume a live auxiliary pointer after
        another plan rotates it.  Reading the NumPy array at that point would copy
        stale pre-rotation bytes back to MPS.  This identity lookup lets that pack
        copy directly from the already resident physical buffer instead.  Multiple
        mirror names for one host allocation are permitted for immutable material
        data, but a mutable physical state with two different device tensors has no
        unambiguous producer and is therefore refused rather than guessed.
        """
        matches = [mirror.tensor for mirror in self._mirrors.values()
                   if mirror.host is host]
        if not matches:
            return None
        first = matches[0]
        if any(tensor is not first for tensor in matches[1:]):
            raise ValueError(
                "one host allocation has multiple device mirrors; a live pointer "
                "refresh cannot choose a producer safely")
        return first

    def host(self, name: str) -> Any:
        """The HOST array a registered mirror shadows, or ``None`` if unregistered.

        EXISTS FOR PLAN-OWNED SCRATCH, and the case that needed it is real. Every
        volume the first families mirrored belonged to the ENGINE, so a builder
        always had the array in hand and one name always meant one array. A family
        whose plan owns a scratch volume — the cylindrical radial prefix and its two
        row vectors — allocates that array in the BUILDER, so building the same
        sub-step's plan twice against one residency produced two arrays under one
        name and :meth:`mirror` refused it as an aliasing defect. That refusal is
        right for engine volumes and wrong here: the scratch is fully overwritten
        before it is read, so two plans SHOULD share it.

        Read-only by construction — it hands back the existing array so a builder can
        pass it straight back to :meth:`mirror` and get the same binding — and it
        returns ``None`` rather than raising, because "not registered yet" is the
        ordinary first-build answer and not an error.
        """
        mirror = self._mirrors.get(name)
        return None if mirror is None else mirror.host

    def sync_in(self, names: Optional[Sequence[str]] = None) -> None:
        """Host -> device for the named mirrors (all of them by default).

        TWO BEHAVIOURS, chosen per mirror by whether it is held.

        Unheld — the shipped behaviour, unchanged: copy, unconditionally. Note it
        has NO ``constant`` check where :meth:`Mirror.sync_out` has one, so 21 of
        ``pml_2d``'s 45 host->device copies per launch re-upload data no kernel
        writes. Under a hold a constant is ``CLEAN_BOTH`` forever and is skipped by
        the rule below, which is where that waste goes.

        Held — copy only what the host actually owns. A ``CLEAN_BOTH`` mirror is
        equal on both sides, so copying it moves nothing; a ``DEVICE_OWNED`` mirror
        would be copied BACKWARDS and destroy the launch's own output. That
        reduction is the whole saving, and it is sound only because every host
        write goes through :meth:`acquire_write` and every mirror that does not is
        sealed and raises.
        """
        for name in (self.names if names is None else tuple(names)):
            mirror = self._mirrors[name]
            if name in self.held or name in self.hoisted:
                if mirror.state != HOST_OWNED:
                    continue
                mirror.sync_in()
                self.copies_in += 1
                mirror.state = CLEAN_BOTH
                self.generation += 1
                continue
            mirror.sync_in()
            self.copies_in += 1
        self.syncs_in += 1

    def sync_out(self, names: Optional[Sequence[str]] = None) -> None:
        """Device -> host for the named non-constant mirrors.

        THE DRAIN IS PAID ONLY WHEN A COPY-OUT FOLLOWS IT. With no hold armed
        (the shipped bracket) it is unconditional and comes FIRST, once, before
        any copy, exactly as it always was. Under a hold it runs only when some
        named mirror is neither held nor constant -- the one case in which this
        method is about to read the device -- and a fully-held ``sync_out`` makes
        NO host wait at all: it marks and seals, and returns with the launch still
        queued.

        WHY THE WAIT WAS THE COST UNDER A HOLD, measured (the counted profile of
        2026-09-24, ``results/round_2026-09-24_night_drivers/
        run_profile_held_step.driver.log``, SUMMARY lines): on ``pml_3d`` held +
        sparse, ``copies_out`` was 0 on every launch and the three
        ``torch.mps.synchronize`` calls a step read 1.641 ms of a 3.194 ms step at
        512k cells and 7.925 ms of 8.426 ms at 4.1M -- the host stood waiting for
        each kernel to finish before it encoded the next seam op. The earlier
        "0.000 ms" figure (2026-09-21) was the drain of an EMPTY queue and said
        nothing about a drain behind a queued launch.

        WHY IT IS SAFE TO SKIP. torch's MPS backend has one stream per process
        (``MPSStreamImpl::getInstance``), so a compiled kernel launch and every
        later device index op or copy on the same tensors are encoded on one
        command queue in program order; and every device->host read this layer
        performs -- ``Mirror.sync_out``, ``differing_words``, ``_to_host`` --
        goes through a blocking ``Tensor.cpu()``, which commits and waits
        (``copy_and_sync`` with ``non_blocking=False`` -> ``COMMIT_AND_WAIT``).
        A held mirror is only ever read on the host through those doors or
        through :meth:`acquire_read` / :meth:`flush` / :meth:`verify`, which take
        them; the seal turns any other host access into a raise rather than a
        stale read. :meth:`verify` keeps its own explicit drain.

        A HELD mirror is not copied -- it is marked ``DEVICE_OWNED`` and SEALED, so
        the device keeps the words and the next host read syncs them out on demand
        through :meth:`acquire_read` while a host write raises. An unheld mirror is
        copied exactly as before.
        """
        import torch  # noqa: PLC0415

        selected = self.names if names is None else tuple(names)
        if not self.held or any(
                name not in self.held and not self._mirrors[name].constant
                for name in selected):
            torch.mps.synchronize()
        for name in selected:
            mirror = self._mirrors[name]
            if name in self.held:
                if mirror.constant:
                    continue
                mirror.state = DEVICE_OWNED
                mirror.seal()
                self.generation += 1
                continue
            mirror.sync_out()
            if not mirror.constant:
                self.copies_out += 1
        self.syncs_out += 1

    def verify(self) -> Dict[str, int]:
        """Per-mirror differing word counts — the residency invariant, measured.

        The composition probe asserts this is empty after every complete step. An
        empty dict from an empty registry proves nothing, so the probe also asserts
        the registry is non-empty; that is the same vacuity discipline the gate's
        signed-zero census follows.
        """
        import torch  # noqa: PLC0415

        torch.mps.synchronize()
        out: Dict[str, int] = {}
        for name, mirror in self._mirrors.items():
            differing = mirror.differing_words()
            if differing:
                out[name] = differing
        return out


# ---------------------------------------------------------------------------
# Toolchain provenance
# ---------------------------------------------------------------------------

#: Where the Metal frontend records its own version. Read from the shipped compiler
#: library rather than guessed from the OS build, because a frontend bump is a
#: CORRECTNESS EVENT for these files exactly as a Triton version bump is for the
#: Triton ones — the arithmetic is held by measurement, not by construction.
_GPU_COMPILER_ROOT = ("/System/Library/PrivateFrameworks/GPUCompiler.framework"
                      "/Versions/Current/Libraries/libGPUCompilerImplLazy.dylib")


def metal_frontend_version() -> Optional[str]:
    """The Metal frontend's own version string, or None when it cannot be read.

    ``None`` is recorded as ``None`` and never replaced by a guess: an artifact that
    names a toolchain it did not measure is worse than one that says it could not
    tell.
    """
    import re  # noqa: PLC0415

    pattern = re.compile(rb"metalfe-[0-9][0-9.]*")
    try:
        with open(_GPU_COMPILER_ROOT, "rb") as handle:
            tail = b""
            while True:
                chunk = handle.read(1 << 22)
                if not chunk:
                    return None
                match = pattern.search(tail + chunk)
                if match:
                    return match.group(0).decode("ascii")
                tail = chunk[-64:]
    except OSError:
        return None


#: Where ``MTLCreateSystemDefaultDevice`` lives.
_METAL_FRAMEWORK = "/System/Library/Frameworks/Metal.framework/Metal"

_APPLE_GPU: Optional[Dict[str, Optional[str]]] = None


def apple_gpu_identity() -> Dict[str, Optional[str]]:
    """This host's Apple GPU as Metal names it: ``name`` and ``architecture``.

    ``architecture`` is ``MTLDevice.architecture.name`` (macOS 14 and later), the
    unit Apple compiles GPU code for: ``applegpu_g13s`` on an M1 Max. It separates
    GPU generations that share a Metal GPU family (M3 and M4 are both Apple9), and
    it is the fact of a Metal environment that torch and the Metal frontend cannot
    show: every Mac on one macOS build reports the same frontend whatever its GPU.

    Read once per process through the Objective-C runtime with ``ctypes``, so it
    needs neither torch nor PyObjC. Every message send goes through a prototype
    typed for that selector: on arm64 ``objc_msgSend`` must be called with the
    callee's exact signature. Never raises; a fact that cannot be read is ``None``
    and ``error`` says why.
    """
    global _APPLE_GPU
    if _APPLE_GPU is not None:
        return dict(_APPLE_GPU)
    out: Dict[str, Optional[str]] = {"name": None, "architecture": None, "error": None}
    try:
        import ctypes  # noqa: PLC0415
        import ctypes.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        if sys.platform != "darwin":
            out["error"] = f"not macOS ({sys.platform})"
        else:
            objc = ctypes.CDLL(ctypes.util.find_library("objc"))
            metal = ctypes.CDLL(_METAL_FRAMEWORK)
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
            objc.objc_autoreleasePoolPush.argtypes = []
            objc.objc_autoreleasePoolPop.restype = None
            objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
            metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
            metal.MTLCreateSystemDefaultDevice.argtypes = []
            send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

            def typed(restype: Any, *argtypes: Any) -> Any:
                return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p,
                                        *argtypes)(send)

            sel = objc.sel_registerName
            send_id = typed(ctypes.c_void_p)
            send_utf8 = typed(ctypes.c_char_p)
            send_responds = typed(ctypes.c_byte, ctypes.c_void_p)
            send_void = typed(None)

            def text(obj: Any) -> Optional[str]:
                raw = send_utf8(obj, sel(b"UTF8String")) if obj else None
                return None if raw is None else raw.decode("utf-8", "replace")

            pool = objc.objc_autoreleasePoolPush()
            device = None
            try:
                device = metal.MTLCreateSystemDefaultDevice()
                if not device:
                    out["error"] = "MTLCreateSystemDefaultDevice returned no device"
                else:
                    out["name"] = text(send_id(device, sel(b"name")))
                    if send_responds(device, sel(b"respondsToSelector:"),
                                     sel(b"architecture")):
                        architecture = send_id(device, sel(b"architecture"))
                        out["architecture"] = (text(send_id(architecture, sel(b"name")))
                                               if architecture else None)
                    else:
                        out["error"] = ("this MTLDevice has no architecture property "
                                        "(it needs macOS 14 or later)")
            finally:
                if device:
                    send_void(device, sel(b"release"))
                objc.objc_autoreleasePoolPop(pool)
    except Exception as exc:  # noqa: BLE001 - an unreadable identity is recorded, not raised
        out["error"] = repr(exc)
    if out["architecture"] is None and out["error"] is None:
        out["error"] = "Metal named no architecture for this device"
    _APPLE_GPU = dict(out)
    return out


def module_sha256(name: str) -> str:
    """sha256 of one module file in this package, by bare file name."""
    import hashlib  # noqa: PLC0415
    import os  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
