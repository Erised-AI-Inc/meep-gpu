"""One Triton launch for every driven component of one susceptibility.

``PolarizationState.update`` advances Ex, Ey and Ez sequentially while rotating one
shared scratch buffer through their P-prev slots.  The arithmetic is independent
per component, but the buffer ownership is not.  This product resolves the complete
destination chain before launch, evaluates all live components in one grid, and
then performs the identical host-side reference rotation only after the launch
succeeds.

This is an enabling product for a future D/ADE/E kernel.  It does not cross the
``update_E``/``update_P`` sub-step boundary and is not production-dispatched.

INTRA-LAUNCH ALIASING — read this before changing the kernel body or reusing it.

The rotation this plan resolves ahead of the launch does not merely reorder buffer
ownership: it puts one arm's OUTPUT on top of an earlier arm's ``p_prev`` INPUT.
Naming the t0 buffers ``A_c = P[c]``, ``B_c = P_prev[c]`` and ``S = _scratch``, the
chain :meth:`FusedAdeStatePlan.run` builds is

    arm 0 (Ex):  out = S      p_now = A_Ex  p_prev = B_Ex
    arm 1 (Ey):  out = B_Ex   p_now = A_Ey  p_prev = B_Ey     <- out1 IS arm 0's p_prev
    arm 2 (Ez):  out = B_Ey   p_now = A_Ez  p_prev = B_Ez     <- out2 IS arm 1's p_prev

That is the same chain ``dispersion.py:687-691`` walks, and it is the reason the
sentence above it at ``dispersion.py:668-669`` — "Every buffer is owned by exactly
one slot at a time, so no two names ever alias" — is true of the ARRAY path and of
:class:`AdeUpdatePPlan`, which quotes it verbatim at ``launch.py:1830-1831``, where
the rotation happens BETWEEN launches, and false INSIDE this one.  Nothing in either
sentence carries over to a fused launch, so the claim is restated here rather than
inherited.  A trivial ``sigma`` skips an arm without breaking the chain: with
``sigma[Ey] == 0`` the driven set is ``(Ex, Ez)``, arm 1 is compiled out by its
``DRIVEN1`` constexpr, and ``out2`` becomes arm 0's ``p_prev``.  The alias follows
the next LIVE arm, not the next index.

The ordering is load-bearing, not vacuous.  Executing the three arms with each
store hoisted above the earlier arms' loads — a host model of the reordering, on a
192-cell grid — leaves 384 of 576 written floats differing from source order at a
worst absolute difference of 4.0e-01 on values of order 1e-1.  A hoist would not
perturb the answer, it would destroy it.

WHY IT IS CORRECT HERE.  Every arm loads and stores at the SAME ``idx`` and under
the same ``live`` mask (both computed once, above the arms).  The aliasing pair is
therefore index-identical, which confines it to a single lane: it is a write-after-
read on one address by one thread, never a cross-program-instance race.  Distinct
program instances hold disjoint ``idx``, so no two of them touch the same element,
and a thread always observes its own accesses to one address in program order.
This is the benign alias class, and it is a different animal from the one
``nonlinear_update_e._require_no_aliasing`` and ``offdiag_update_e`` refuse outright
— those kernels re-read their inputs at NEIGHBOUR offsets, so an alias there makes
the answer depend on block schedule.

WHAT THAT SAFETY RESTS ON — AN OBSERVATION, NOT A GUARANTEE.  Say it plainly: no
document promises this.  Two links in the argument are measured properties of one
toolchain rather than contracts.

  * That the compiler does not hoist arm N's store above arm N-1's load.  Triton
    attaches no aliasing attribute to kernel pointer arguments: in this repo's own
    captured IR (``parity/meep_gpu/results/triton_feasibility_2026-08-09/
    triton_ptx/``) the TTIR parameters carry ``tt.divisibility = 16`` and nothing
    else, and the LLVM IR signature is bare — ``define void @triton_pml_update(ptr
    addrspace(1) %0, ptr addrspace(1) %1, ...)``, with no ``noalias`` on any
    parameter.  Without ``noalias`` LLVM's alias analysis must return MayAlias for
    two pointer arguments and may not reorder the pair.  That covers the LLVM stage
    only.  It is not a Triton language contract — the absence of ``noalias`` is a
    fact about what Triton 3.1.0 emitted, not a promise about what it will emit —
    and it says nothing about Triton's own MLIR passes above LLVM or about ptxas
    below it.  No Triton source is available on the host this note was written on,
    so no file:line for a Triton-side ordering rule is cited: none was checked, and
    none is asserted.
  * That element i is handled by the same lane in every arm.  It is, when one
    layout serves the whole kernel.  The captured TTGIR for a structurally
    identical 1-D elementwise kernel at BLOCK=256 / num_warps=4 uses a single
    ``#blocked<{sizePerThread=[2], threadsPerWarp=[32], warpsPerCTA=[4],
    order=[0]}>`` for every load and store with no ``convert_layout`` anywhere.
    That is an observation on a neighbouring kernel, not a rule.

WHAT WAS ACTUALLY MEASURED.  ``parity/meep_gpu/results/triton_fused_ade_state_
2026-08-11/`` — 3/3 cases, 36/36 complete steps bit-identical against BOTH the array
path and a separate per-component Triton ADE (three launches, so the oracle itself
carries no intra-launch alias).  Triton 3.1.0, CuPy 13.5.1, NVIDIA RTX A6000,
``BLOCK = DEFAULT_BLOCK = 256``, Triton's default warp count, ``enable_fp_fusion =
False``, ONE grid shape.  That is the extent of the evidence: one compiler version,
one architecture, one block size, one layout.
``test_plan_resolves_the_shared_scratch_chain_then_rotates_only_after_launch``
pins the chain by asserting ``(args[0], args[5], args[10]) == (S, B_Ex, B_Ey)``,
which IS this aliasing — but it pins it as a rotation, and never names it a hazard.

WHAT WOULD BREAK IT.

  * A Triton release that marks kernel pointer arguments ``noalias``.  It would be
    a legitimate optimization under a language contract that forbids aliasing
    arguments, and it would make this launch undefined behaviour rather than
    slower.  The failure is silent: bit-identity is the only thing that detects it.
  * Any arm reading at an offset other than ``idx`` — which is exactly what the
    "future D/ADE/E kernel" this module exists to enable would introduce, since the
    curl reads neighbours.  The moment a read leaves its own index the alias stops
    being one lane's write-after-read and becomes a cross-instance race, and no
    compiler ordering can repair it.  Fusing further REQUIRES breaking the chain
    first, not inheriting it.
  * Reordering the arms in the kernel body, or reversing the rotation in
    :meth:`run`, so a store precedes the load it overwrites.
  * A layout split that gave any arm a different lane assignment.

Restoring a guarantee, if one is ever wanted, is a plan-level change and not a
kernel-level one: give the launch one more scratch buffer per susceptibility so no
output aliases any input, and the question stops being asked.  Reordering the
kernel body does not do it — under a wrongly-asserted ``noalias`` every arrangement
is undefined.  That cost (one buffer per driven term) has not been paid because
nothing dispatches this: ``plan_step``'s ``fuse_ade`` defaults to False, the
production seam composes with ``fuse_ade=False`` explicitly (``fastpath.py:1963,
2004``), and ``test_dispatch_contract.py:813`` pins it there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .coverage import (
    ELECTRIC_COMPONENTS,
    Coverage,
    ade_update_p_coverage,
    sigma_is_volume,
)

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:
    @triton.jit
    def fused_ade_state(
        out0, p0_now, p0_prev, sigma0, drive0,
        out1, p1_now, p1_prev, sigma1, drive1,
        out2, p2_now, p2_prev, sigma2, drive2,
        c_now, c_prev, c_drive, n_elem,
        DRIVEN0: tl.constexpr, DRIVEN1: tl.constexpr, DRIVEN2: tl.constexpr,
        SIGMA_IS_VOLUME0: tl.constexpr,
        SIGMA_IS_VOLUME1: tl.constexpr,
        SIGMA_IS_VOLUME2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Advance one susceptibility's live Ex/Ey/Ez ADE recurrences together."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem

        if DRIVEN0:
            p0 = tl.load(p0_now + idx, mask=live, other=0.0)
            q0 = tl.load(p0_prev + idx, mask=live, other=0.0)
            d0 = tl.load(drive0 + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME0:
                s0 = tl.load(sigma0 + idx, mask=live, other=0.0)
            else:
                s0 = sigma0
            tl.store(out0 + idx,
                     ((p0 * c_now) + (c_prev * q0)) +
                     (c_drive * (s0 * d0)), mask=live)

        if DRIVEN1:
            p1 = tl.load(p1_now + idx, mask=live, other=0.0)
            q1 = tl.load(p1_prev + idx, mask=live, other=0.0)
            d1 = tl.load(drive1 + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME1:
                s1 = tl.load(sigma1 + idx, mask=live, other=0.0)
            else:
                s1 = sigma1
            tl.store(out1 + idx,
                     ((p1 * c_now) + (c_prev * q1)) +
                     (c_drive * (s1 * d1)), mask=live)

        if DRIVEN2:
            p2 = tl.load(p2_now + idx, mask=live, other=0.0)
            q2 = tl.load(p2_prev + idx, mask=live, other=0.0)
            d2 = tl.load(drive2 + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME2:
                s2 = tl.load(sigma2 + idx, mask=live, other=0.0)
            else:
                s2 = sigma2
            tl.store(out2 + idx,
                     ((p2 * c_now) + (c_prev * q2)) +
                     (c_drive * (s2 * d2)), mask=live)
else:  # pragma: no cover - laptop path
    fused_ade_state = None  # type: ignore[assignment]


def fused_ade_state_kernel() -> Any:
    if fused_ade_state is None:
        raise ImportError(
            "the fused ADE-state kernel needs the optional `triton` package "
            f"(pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return fused_ade_state


def fused_ade_state_coverage(fields: Any, state: Any) -> Coverage:
    """May one launch advance all driven components of this susceptibility?"""
    reasons: List[str] = []
    reporter = getattr(state, "driven", None)
    if not callable(reporter):
        return Coverage(False, ("the susceptibility does not report driven()",))
    components = tuple(reporter())
    if not components:
        reasons.append("the susceptibility has no driven components")
    canonical = tuple(name for name in ELECTRIC_COMPONENTS if name in components)
    if components != canonical or len(set(components)) != len(components):
        reasons.append(
            f"driven components {components!r} are not the canonical ordered subset "
            f"of {ELECTRIC_COMPONENTS}")
    for component in components:
        verdict = ade_update_p_coverage(fields, state, component)
        reasons.extend(f"{component}: {reason}" for reason in verdict.reasons)
    return Coverage(not reasons, tuple(reasons))


class FusedAdeStatePlan:
    """One allocation-free launch plus the exact shared-scratch reference rotation."""

    __slots__ = (
        "state", "components", "shape", "n_elem", "block", "_grid", "_kernel",
        "_pointer", "_sigma_is_volume",
    )

    def __init__(self, state: Any, components: Sequence[str], shape: Sequence[int],
                 block: int, kernel: Any = None, pointer: Any = None) -> None:
        if pointer is None:
            from .launch import CupyPointer  # noqa: PLC0415
            pointer = CupyPointer
        self.state = state
        self.components = tuple(components)
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel
        self._pointer = pointer
        self._sigma_is_volume = {
            component: bool(sigma_is_volume(state, component))
            for component in self.components
        }

    def run(self, drive, guard: Optional[bool] = None) -> None:
        """Launch first; rotate Python ownership only after a successful launch."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        state = self.state
        kernel = self._kernel if self._kernel is not None else fused_ade_state_kernel()
        pointer = self._pointer
        scratch = state._scratch
        entries: Dict[str, Tuple[Any, Any, Any, Any, Any]] = {}
        for component in self.components:
            p = state.P[component]
            p_prev = state.P_prev[component]
            entries[component] = (
                scratch, p, p_prev, state.sigma[component], drive(component))
            scratch = p_prev

        dummy = state._scratch
        arguments: List[Any] = []
        driven_flags: List[int] = []
        volume_flags: List[int] = []
        for component in ELECTRIC_COMPONENTS:
            entry = entries.get(component)
            if entry is None:
                arguments.extend((pointer(dummy), pointer(dummy), pointer(dummy),
                                  0.0, pointer(dummy)))
                driven_flags.append(0)
                volume_flags.append(0)
                continue
            output, p, p_prev, sigma, component_drive = entry
            volume = self._sigma_is_volume[component]
            arguments.extend((
                pointer(output), pointer(p), pointer(p_prev),
                pointer(sigma) if volume else float(sigma),
                pointer(component_drive),
            ))
            driven_flags.append(1)
            volume_flags.append(1 if volume else 0)

        c_now, c_prev, c_drive = state._coefficients
        kernel[self._grid](
            *arguments, float(c_now), float(c_prev), float(c_drive), self.n_elem,
            DRIVEN0=driven_flags[0], DRIVEN1=driven_flags[1],
            DRIVEN2=driven_flags[2],
            SIGMA_IS_VOLUME0=volume_flags[0],
            SIGMA_IS_VOLUME1=volume_flags[1],
            SIGMA_IS_VOLUME2=volume_flags[2], BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

        for component in self.components:
            output, p, _p_prev, _sigma, _drive = entries[component]
            state.P[component] = output
            state.P_prev[component] = p
        state._scratch = scratch

    def __repr__(self) -> str:
        return (f"FusedAdeStatePlan({self.components}, shape={self.shape}, "
                f"block={self.block})")


def plan_fused_ade_state(fields: Any, state: Any,
                         block: Optional[int] = None) -> Optional[FusedAdeStatePlan]:
    """Build one fused susceptibility plan, or ``None`` when refused."""
    if not fused_ade_state_coverage(fields, state).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    components = tuple(state.driven())
    return FusedAdeStatePlan(
        state, components, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block)
