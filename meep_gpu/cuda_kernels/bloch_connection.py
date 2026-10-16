"""The Bloch connection a complex kernel binds per axis: a kind and two float32 parts. No CuPy.

:func:`bloch_phase_arguments` encodes ``grid.bloch_phase`` the way the array path
applies it on plane 0 of a phased axis (``stepping._bloch_phases``,
``stepping._bloch_ghost_operand``), and :func:`connection_of` resolves one phase to a
copy, a negation or the unfused product by ``part_products.connect_parts``' rule.
``complex_pml_kernels`` binds the result to its launches and imports both from here;
the fused products' predicates resolve the same tables before they admit a run
(``complex_offdiag_update_e.bloch_connection_table``).

WHY THIS IS A MODULE OF ITS OWN. The two functions are host arithmetic over NumPy
scalars, but they lived in ``complex_pml_kernels``, which imports CuPy at module scope
for its compiled kernels. A predicate clause that resolved the table there raised
``ModuleNotFoundError: No module named 'cupy'`` on a host without CuPy, and the
coverage census, which measures these predicates on such a host with the backend
clause satisfied at the input (``cuda_predicate_battery._factored``), scored that raise
as a refusal: measured on the 1.0 census, 5 cells of two off-diagonal fused products,
2 of them rows the products otherwise admit. Here the clause evaluates on every host,
so a refusal it returns is about the run, never about the host.

NUMPY ONLY, like ``coverage.py``'s siblings ``offdiag_presence.py`` and
``chunk_coverage.py``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:
    from . import complex_emitter
    from .coverage import PERIODIC, real_pml_boundary_kinds
    from ..part_products import CONNECT_TOLERANCE
    from ..readback import cmul
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    from meep_gpu.cuda_kernels import complex_emitter
    from meep_gpu.cuda_kernels.coverage import PERIODIC, real_pml_boundary_kinds
    from meep_gpu.part_products import CONNECT_TOLERANCE
    from meep_gpu.readback import cmul

__all__ = ["bloch_phase_arguments", "connection_of"]


def connection_of(thephase: Tuple[float, float]) -> Tuple[int, Tuple[Any, Any]]:
    """``(kind, (a, b))`` for one Bloch connection, by ``part_products.connect_parts``' rule.

    ``thephase`` is MEEP's ``thephase`` as ``complex<double>`` parts. Within
    ``part_products.CONNECT_TOLERANCE`` of +1 the connection is a copy (kind
    ``CONNECT_NONE``: the kernel reads the stored word), of -1 a per-part negation
    (``CONNECT_NEGATE``); otherwise the unfused product (``CONNECT_PHASE``) with the phase
    rounded to the part dtype, ``a = float32(pr)``, ``b = float32(pi)``
    (boundaries.cpp:495). ``test_complex_chunk_forms.py`` pins the kernel's application
    of the result against ``part_products.connect`` word for word.
    """
    pr, pi = float(thephase[0]), float(thephase[1])
    if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
        return complex_emitter.CONNECT_NONE, (np.float32(1.0), np.float32(0.0))
    if abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
        return complex_emitter.CONNECT_NEGATE, (np.float32(-1.0), np.float32(0.0))
    return complex_emitter.CONNECT_PHASE, (np.float32(pr), np.float32(pi))


def bloch_phase_arguments(grid: Any, backward: bool
                          ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Encode the per-axis Bloch connection as (three kinds, six float32 parts).

    Transcribed from ``stepping._bloch_phases`` and ``stepping._bloch_ghost_operand``:

    * the value is ``grid.bloch_phase(axis)`` = ``eikna``, exp(2*pi*i*k*L), with the
      Brillouin edge EXACTLY -1+0j;
    * ``None`` means no connection at all -- kind ``CONNECT_NONE``, and the kernel reads
      the stored word;
    * the connection is applied on plane 0 of the operand read at the cell ITSELF. The B
      curl (forward differences, ``backward`` False) reads the not-owned word BELOW the
      window and takes ``conj(eikna)``; the D curl reads the one ABOVE it and takes
      ``eikna``. ``thephase = complex<double>(1.0) * factor`` as the array path forms it
      (``readback.cmul``);
    * the kind and the parts are :func:`connection_of`'s: a copy, a negation, or the
      unfused product with the phase rounded to float32 parts.

    A PHASED AXIS MUST RESOLVE PERIODIC. ``stepping._bloch_phases`` raises on the pairing
    and the predicate refuses it; this raises too, so a caller who skips the predicate is
    refused loudly rather than handed a mis-bound kind. A PML on a phased axis is ADMITTED:
    MEEP makes the axis periodic and the layer is a material under the wrap.
    """
    kinds = real_pml_boundary_kinds(grid)
    flags: list = []
    values: list = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            flags.append(np.int32(complex_emitter.CONNECT_NONE))
            values.extend((np.float32(1.0), np.float32(0.0)))
            continue
        if kinds[axis] != PERIODIC:
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        eikna = complex(phase)
        factor = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
        kind, parts = connection_of(cmul((1.0, 0.0), factor))
        flags.append(np.int32(kind))
        values.extend(parts)
    return tuple(flags), tuple(values)
