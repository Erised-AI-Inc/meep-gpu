"""The radial prefix the CUDA cylindrical curl pair consumes — WITHOUT CuPy.

NOTHING HERE IMPORTS CUPY, NUMPY OR THE KERNEL MODULE, for the reason ``coverage.py``
gives for the predicate: the pieces of this track that are pure decision logic and
pure array work have to be importable, exercisable and mutable on a machine with no
GPU. ``cylindrical_kernels`` holds ``cp.RawKernel`` objects and imports ``cupy`` at
module scope, so a prefix helper living there could not be exercised at the merge bar
— and the wall-row extension is exactly the sort of off-by-one a laptop test should
be able to pin.

WHY THE PREFIX IS NOT A KERNEL ON THIS BACKEND, in one paragraph; the argument in
full is in ``cylindrical_kernels``' docstring. ``stepping.cylindrical_rderiv_prefix``
ends in ``xp.cumsum`` (stepping.py:1315, :1333), and on every configuration the CUDA
predicate admits ``xp`` is CuPy. ``cupy.cumsum`` in float32 is deterministic but is
NOT a sequential accumulation, so a hand-written column-serial CUDA scan — which
computes the sequential sum — is a DIFFERENT float32 number, and this family's whole
claim is bytewise identity. The scan therefore stays on the array path and is CALLED
here rather than re-derived.
"""

from __future__ import annotations

from typing import Any, Dict

#: ``ir0 = origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)``
#: (``stepping.cylindrical_rderiv_prefix``:1266-1270). The r origin IS the axis here,
#: so the whole value is the half-shift: Ep sits at the node (r-shift 0) and Hp half
#: a cell out (r-shift 1). Getting these two backwards is a HALF-CELL error in the
#: radial weights and is silent — a smooth, plausible, slightly wrong field — which
#: is why they are a table keyed by sub-step rather than a literal at a call site.
PREFIX_IR0: Dict[str, float] = {"step_B": 0.0, "step_D": 0.5}

#: Which sub-step's scan carries the ZERO WALL ROW. Only the B side: its Bz curl is a
#: forward difference of the prefix and would otherwise read minus the whole
#: accumulated sum at the last row (``stepping.step_B``:303-314, :326-333). The D
#: side's backward difference reads rows i and i-1 and never looks past the top.
PREFIX_WALL_ROW: Dict[str, bool] = {"step_B": True, "step_D": False}

#: Which stored component each sub-step's prefix is built from: Ep on the B side and
#: Hp on the D side, which are ``Ey`` and ``Hy`` under this engine's
#: ``x -> r, y -> phi, z -> z`` mapping (grid.py:365-369).
PREFIX_COMPONENT: Dict[str, str] = {"step_B": "Ey", "step_D": "Hy"}


def prefix_rows(sub_step: str, radial_rows: int) -> int:
    """How many rows this sub-step's prefix has — ``nr`` or ``nr + 1``."""
    if sub_step not in PREFIX_IR0:
        raise ValueError(f"sub_step must be one of {sorted(PREFIX_IR0)}, got {sub_step!r}")
    return int(radial_rows) + (1 if PREFIX_WALL_ROW[sub_step] else 0)


def cylindrical_prefix(fields: Any, sub_step: str, scratch: Any = None) -> Any:
    """This launch's radial prefix, from the SHIPPED array-path scan.

    ``stepping.cylindrical_rderiv_prefix`` is CALLED, not re-derived: it is the one
    function whose float32 summation order defines the answer on this backend, and
    the same ``StepScratch`` is threaded through so its two cached invariant row
    vectors are shared with the array path rather than rebuilt twice a timestep.

    On the B side the source is Ep (= Ey) EXTENDED BY ONE ZERO WALL ROW to
    (nr + 1, ...) — ``stepping.step_B``:326-333, transcribed line for line INCLUDING
    the pooled-versus-fresh allocation branch, because the two writes below are what
    ``xp.concatenate`` used to do and the pool is what stopped it allocating the
    whole volume again every step. On the D side the source is Hp (= Hy), unextended
    (:418-419).
    """
    from ..stepping import _face, _span, cylindrical_rderiv_prefix  # noqa: PLC0415

    if sub_step not in PREFIX_IR0:
        raise ValueError(f"sub_step must be one of {sorted(PREFIX_IR0)}, got {sub_step!r}")
    source = getattr(fields, PREFIX_COMPONENT[sub_step])
    ir0 = PREFIX_IR0[sub_step]
    xp = fields.grid.xp
    if not PREFIX_WALL_ROW[sub_step]:
        return cylindrical_rderiv_prefix(xp, source, ir0, scratch=scratch)

    rows = source.shape[0]
    extended_shape = (rows + 1,) + source.shape[1:]
    extended = (xp.empty(extended_shape, dtype=source.dtype) if scratch is None
                else scratch.take("cyl_extended", extended_shape, source.dtype))
    extended[_span(0, 0, rows)] = source
    extended[_face(0, rows)] = 0
    return cylindrical_rderiv_prefix(xp, extended, ir0, scratch=scratch)
