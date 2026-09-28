"""Shared machinery of the SCRATCH-OUTPUT off-diagonal D/E welds.

WHAT THIS SUPERSEDES, AND ON WHAT TERMS.
``parity/meep_gpu/results/triton_fused_offdiag_electric_2026-08-20`` is THIS
backend's own recorded refusal of the D->E off-diagonal pair: on an RTX A6000,
both float32 subnormal policies, a kernel spliced verbatim out of the two
shipped bodies disagreed with its two-launch reference on 42 of 60 subject cases
at up to 6.58e-02, and the disagreement was SCHEDULE DEPENDENT (differing-word
counts moving from 4384 to 0 as BLOCK went 64 -> 1024 on a 4096-cell grid).
That measurement stands and is not disputed here. What it measured is named in
its own PROVENANCE: a ONE-ORDINARY-LAUNCH weld of the SHIPPED IN-PLACE halves,
in which ``step_D`` writes D while the off-diagonal constitutive arm reads its
neighbours' D. Its closure over the alternative — recomputing the neighbour curl
instead of loading it — is, in its own words, "a DERIVATION from a parsed fact
rather than a device measurement", and the form it derives against is the one
that still steps D in place, so the halo recompute reads the same clobbered
``D[m]``/``fu_D[m]``.

THE DESIGN THIS MODULE CARRIES REMOVES BOTH PREMISES:

* the curl half writes ``D_new``/``fu_new`` to LAUNCH-LOCAL SCRATCH and never
  in place, so ``D``/``fu_D``/``H`` are pre-launch state for the whole dispatch;
* the constitutive half takes its own cell's displacement from registers and
  RE-DERIVES every foreign stencil tap from that pre-launch state through the
  same inlined :func:`step_cell` body, so NOTHING WRITTEN IS EVER READ;
* the in-seam passes the driver runs between ``step_D`` and ``update_E`` are
  carried per cell by a CLOSED FORM over the raw stepped value rather than by a
  second pass;
* the plan ROTATES the D/fu references after the launch returns.

The closed form is not this lane's invention and was measured before it was
built: ``results/metal_scratch_weld_closed_form_2026-09-01/probe.json`` walked
ten fixtures two complete steps each and found the per-cell fill/clear
resolution over the raw stepped D byte-identical to the driver's
``fill_D`` / ``zero_metal_D`` / ``fill_folded_far_ghosts_D`` pass order, with
every reachable null diverging. That is a Metal artifact and certifies nothing
here; this lane owes its own NumPy probe on its own arms and its own device
gate, and both exist (``parity/meep_gpu/probe_triton_offdiag_scratch_weld.py``,
``parity/meep_gpu/gate_triton_offdiag_stencil_welds.py``).

WHY THE ROTATION IS SOUND UNDER DISPATCH. The plan is the only writer of the two
buffers it swaps, and it swaps the ENGINE's own attributes so every later pass,
read-back and monitor sees the freshly written buffer under the engine's own
name. ``Fields`` holds ``Dx``/``fu_Dx`` as plain attributes and rebuilds
``displacement_minus_polarization_volumes`` on every call (fields.py:1107-1138),
so no cached alias survives the swap; the predicate additionally refuses every
configuration with a registered polarization, which is the only branch that
would hold a separate scratch view.

This paragraph was headed "where ``fastpath`` is not involved" until 2026-09-15,
and the qualifier was a mechanism rather than caution. With no ``warm`` on this
class, ``fastpath.warm_plan`` fell through to ``_warm_with_empty_grid``, which
empties the settable ``_grid`` slot and CALLS ``run`` — so the plan-time warm
pass would have launched nothing and rotated anyway, leaving the engine's D
volumes on the zero twins before step 1. :meth:`ScratchWeldPairPlan.warm`
removes that path (``warm_plan`` asks for a callable ``warm`` before anything
else), and the two products are routed through ``triton_kernels/launch.py``'s
certified-product table from the same edit. Three facts carry the rotation
under dispatch:

* THE WARM NEVER ROTATES AND NEVER COUNTS. It resolves roles, launches once at
  grid ``(0,)`` and puts the grid back. ``launches`` counts ``run`` alone, so the
  plan's counter and ``fastpath``'s dispatch counter stay the two independent
  witnesses ``fastpath._plan_launches_of`` reads them as.
* ONE PLAN ON THESE SHAPES HOLDS A D POINTER FROM BUILD TIME: the mirror-ghost
  fill at ``fill_D`` on a folded grid (``symmetry.MirrorGhostFillPlan`` wraps its
  targets in ``__init__``). It therefore fills the TWIN on alternate steps, which
  moves no byte the step keeps: the next launch rewrites every stored cell of the
  twin, and on the other steps the fill reapplies ghosts the closed form already
  wrote. The same composition has run on device on the hand-CUDA table:
  ``results/dispatch_fused_route_cuda_2026-09-15_weldgrid/cuda_shipped`` drove the
  rotating ``cuda:folded off-diagonal stencil weld`` beside Triton's mirror fill
  on ``folded_offdiag_magnetic_2d``, PASS-FUSED with substitution EXACT over 1999
  steps (6.0 -> 4.0 launches per step), and ``cuda:off-diagonal stencil weld`` on
  ``offdiag_magnetic_2d`` over 2399 steps (4.0 -> 2.0).
* NO DEPOSIT REPAIR IS BRACKETED AROUND THE LAUNCH: both predicates refuse an
  electric source in the seam (``CARRIES_DEPOSIT_REPAIR`` is False), so no repair
  plan captures a D pointer beside it.

Whether Triton 3.1 compiles without executing at grid ``(0,)`` stays the open
measurement ``fastpath.warm_plan``'s docstring names. For these two kernels the
stencil gate's ``warm`` device leg takes it before the ledger entry is cut, and the
Triton route campaign takes it again through ``plan_fast_path``.

WHAT IS LIFTED AND WHAT IS NEW. The curl arithmetic inside each family's
``_step_cell`` is the certified curl kernel's own body between its decode
prologue and its stores, re-headed as a function of ``(i, j, k, live)`` and
returning the six registers instead of storing them; :func:`certified_tail` and
:func:`lifted_tail` cut both bodies at their declared anchors and the probe
asserts the two byte-identical. The constitutive text is the certified
off-diagonal emitter's own statements with the displacement loads replaced by
registers and re-derivations, asserted as contiguous statement runs by the same
probe. The NEW text — the only text these families can be blamed for — is the
closed form and the tap plumbing.

Import contract: importable WITHOUT Triton. Everything here is host code.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MIRROR_ROW", "ScratchWeldPairPlan", "SCRATCH_PREFIX",
    "certified_tail", "lifted_tail", "scratch_twin", "twin_table",
]

#: The stored row ``_fill_symmetry_ghost_cells`` images cell 0 from
#: (``stepping.py:1426-1470``: ``cell 0 = parity * cell 2``). Spelled once, here,
#: because both folded families need it and a second spelling is a second thing
#: to keep in step with the array path.
MIRROR_ROW = 2

#: How a plan-owned twin is named when it is registered anywhere. Only the
#: repr and the diagnostics use it; the buffers themselves are anonymous arrays.
SCRATCH_PREFIX = "weld_scratch:"


# ---------------------------------------------------------------------------
# The machine-checked lift
# ---------------------------------------------------------------------------

def certified_tail(source: Any, *, decode_end: str, store_start: str) -> str:
    """A certified curl kernel's body BETWEEN its decode prologue and its stores.

    ``source`` is either the kernel's source TEXT or the function itself; the
    text form is what the laptop uses, where Triton is absent and the kernel
    object is ``None``.

    ``decode_end`` is the last line of the index decode (the certified real curls
    end it with ``i = plane // ny``) and ``store_start`` the first store line.
    Both are PER-FAMILY ANCHORS rather than constants: the complex and no-PML
    curls decode differently and store differently, and cutting at the wrong
    anchor silently returns a shorter body that then compares equal to a shorter
    lift.

    Refuses rather than returns a partial cut. A lift check that quietly compared
    two empty strings would pass on any kernel at all, which is the exact shape
    of failure this whole discipline exists to prevent.
    """
    if not isinstance(source, str):
        source = inspect.getsource(source)
    if decode_end not in source:
        raise AssertionError(
            f"the certified curl body no longer carries the decode anchor "
            f"{decode_end.strip()!r}; the lift has nowhere to cut")
    if store_start not in source:
        raise AssertionError(
            f"the certified curl body no longer carries the store anchor "
            f"{store_start.strip()!r}; the lift has nowhere to stop")
    tail = source.split(decode_end, 1)[1].split(store_start, 1)[0]
    if not tail.strip():
        raise AssertionError(
            "the certified curl body between its anchors is empty; the two "
            "anchors have crossed and the lift would compare nothing")
    return tail


def lifted_tail(source: Any, *, decode_end: str, return_start: str) -> str:
    """The weld helper's body between its own re-derived decode and its return.

    The mirror of :func:`certified_tail`, cut at the helper's own anchors, and
    accepting the same two input forms. The probe compares the two strings for
    byte equality, which is what makes the curl arithmetic in the weld the
    certified arithmetic rather than a transcription of it.
    """
    if not isinstance(source, str):
        source = inspect.getsource(source)
    if decode_end not in source:
        raise AssertionError(
            f"the weld helper no longer carries its decode anchor "
            f"{decode_end.strip()!r}")
    if return_start not in source:
        raise AssertionError(
            f"the weld helper no longer carries its return anchor "
            f"{return_start.strip()!r}")
    tail = source.split(decode_end, 1)[1].split(return_start, 1)[0]
    if not tail.strip():
        raise AssertionError("the weld helper's lifted body is empty")
    return tail


# ---------------------------------------------------------------------------
# The twins
# ---------------------------------------------------------------------------

def scratch_twin(host: Any) -> Any:
    """A plan-owned twin of one rotating volume: a fresh ZERO array, same dtype.

    Zeroing rather than copying is deliberate and is measurable. The kernel
    writes EVERY stored cell of the twin on every launch — the dispatch is sized
    from ``n_elem`` and the closed form is total — so a leftover value can only
    surface if that totality breaks, and then it surfaces as a zero plane, which
    a uint32 word comparison catches, rather than as a plausible stale field.
    """
    module = type(host).__module__.split(".")[0]
    if module == "cupy":  # pragma: no cover - device path
        import cupy  # noqa: PLC0415

        return cupy.zeros_like(host)
    import numpy  # noqa: PLC0415

    return numpy.zeros_like(numpy.ascontiguousarray(host))


def twin_table(fields: Any, names: Sequence[str]) -> Dict[str, Any]:
    """``{attribute name: fresh twin}`` for every rotating volume, or refuse.

    A missing volume is refused here rather than at the launcher: binding a
    ``None`` twin gives a ``TypeError`` at Triton's argument marshalling, which
    reports the launcher and not the cause.
    """
    table: Dict[str, Any] = {}
    for name in names:
        host = getattr(fields, name, None)
        if host is None:
            raise ValueError(
                f"{name} is not allocated, so the scratch weld has no volume to "
                f"twin; the plan refuses rather than binding a null pointer")
        table[name] = scratch_twin(host)
    return table


# ---------------------------------------------------------------------------
# The plan base — the rotation, and nothing else
# ---------------------------------------------------------------------------

class ScratchWeldPairPlan:
    """ONE launch spanning the D seam, with the post-launch buffer rotation.

    Subclasses own their kernel and their argument order and implement
    :meth:`_launch`; this base owns the part that is easy to get subtly wrong.

    THE TWIN BUFFERS BELONG TO THE PLAN. The array path never rotates D, so
    there is no engine rotation to follow (contrast ``PolarizationState.update``,
    which rotates P and which the ADE plans therefore READ rather than drive).
    Roles are resolved immediately before every launch by reading the engine's
    current attribute, and the engine's references move only after the launch
    call has returned.

    AN ALIASED PAIR IS REFUSED RATHER THAN LAUNCHED. The whole design is that
    nothing written is read; a plan whose scratch and live buffer were the same
    allocation would be the 2026-08-20 in-place weld again, wearing this class's
    name, and would race exactly as that one did.
    """

    __slots__ = ("family", "fields", "rotated", "rotated_names", "launches",
                 "shape", "n_elem", "block", "num_warps", "replaces_sub_steps",
                 "_grid")

    #: The driver call sites one launch performs. Subclasses override.
    replaces: Tuple[str, ...] = ()

    def __init__(self, family: str, fields: Any, rotated: Mapping[str, Any],
                 rotated_names: Sequence[str], shape: Sequence[int],
                 block: int, replaces: Sequence[str],
                 num_warps: Optional[int] = 1) -> None:
        self.family = str(family)
        self.fields = fields
        self.rotated = dict(rotated)
        self.rotated_names = tuple(rotated_names)
        if set(self.rotated_names) != set(self.rotated):
            raise ValueError(
                f"the rotation order {self.rotated_names} does not name the same "
                f"volumes as the twin table {tuple(sorted(self.rotated))}; the "
                f"launch binds by that order and a mismatch is a wrong buffer")
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3:
            raise ValueError("a scratch weld plan needs a three-axis shape")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.replaces_sub_steps = tuple(replaces)
        self.launches = 0
        self._grid = ((self.n_elem + self.block - 1) // self.block,)

    # -- roles -------------------------------------------------------------
    def _resolve(self) -> Tuple[List[Any], List[Any]]:
        """``(scratch write arrays, pre-launch read arrays)`` by CURRENT role."""
        writes: List[Any] = []
        reads: List[Any] = []
        for name in self.rotated_names:
            live = getattr(self.fields, name, None)
            twin = self.rotated[name]
            if live is None or twin is None:
                raise RuntimeError(
                    f"the {self.family} weld lost the buffer for {name}; a launch "
                    f"against an unresolved volume would step garbage")
            if live is twin:
                raise RuntimeError(
                    f"the {self.family} weld resolved {name} and its scratch to ONE "
                    f"array; the whole design is that nothing written is read, so "
                    f"an aliased pair is refused rather than launched")
            reads.append(live)
            writes.append(twin)
        return writes, reads

    def _rotate(self) -> None:
        """Move the engine's references onto the freshly written buffers."""
        for name in self.rotated_names:
            current = getattr(self.fields, name)
            setattr(self.fields, name, self.rotated[name])
            self.rotated[name] = current

    # -- the launch --------------------------------------------------------
    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch once, then rotate — in that order, and only on success."""
        writes, reads = self._resolve()
        self._launch(writes, reads, guard)
        self.launches += 1
        self._rotate()

    def warm(self) -> None:
        """Compile the kernel at PLAN TIME: one launch on an empty grid, no rotation.

        ``fastpath.warm_plan`` asks for a callable ``warm`` before anything else.
        Without one it falls to ``_warm_with_empty_grid``, which empties ``_grid``
        and calls :meth:`run` — whose rotation is unconditional, so a zero-program
        launch would still move the engine's D references onto the zero twins
        before step 1. This is that same empty-grid launch with the rotation left
        out, and it is the whole reason the two scratch-output products can sit in
        a composer's table.

        IN ORDER. Roles are resolved exactly as :meth:`run` resolves them, so an
        aliased pair is refused here, at plan time, rather than on the first step.
        The grid is emptied and the subclass's own :meth:`_launch` is called once:
        Triton compiles at the subscript and enqueues zero programs, so nothing is
        read and nothing is written. The grid is put back in a ``finally``, so a
        compile that raises still leaves the plan's grid as it was, and the
        exception propagates — ``warm_plan``'s contract for a warm that was
        attempted and failed, on which the caller unfills the slot before anything
        has been written.

        WHAT IT DOES NOT DO. It never calls :meth:`_rotate`, and it never increments
        ``launches``: that counter is the plan's own witness of STEP launches, read
        by ``fastpath._plan_launches_of`` beside the dispatch counter, and a warm that
        counted would put the two witnesses one apart on every dispatched run.

        Returns ``None``, which ``warm_plan`` reads as "warmed".
        """
        writes, reads = self._resolve()
        saved = self._grid
        self._grid = (0,)
        try:
            self._launch(writes, reads, None)
        finally:
            self._grid = saved
        return None

    @property
    def runs(self) -> int:
        return self.launches

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"{type(self).__name__}(family={self.family!r}, "
                f"shape={self.shape}, block={self.block}, "
                f"rotates={self.rotated_names})")
