"""Host references of the fold passes, shared by the Triton fold family's gates.

The array path's special-kz beta pass and BFAST pass, transcribed per cell on bare
arrays, for the gates that step the kernels on synthetic and recorded states
(``gate_triton_special_kz``, ``gate_triton_bfast``, ``gate_triton_folded_complex``):

* :func:`beta_increment` -- ``stepping._special_kz_beta_term``: ``-(c * g)`` on real
  storage, the quarter turn ``(c * g_im, -(c * g_re))`` of the partner's parts on
  complex storage, with ``c`` the host-rounded word
  (``special_kz.beta_curl_coefficients``);
* :func:`bfast_increment` -- ``stepping._bfast_term``: the state's two roundings, the
  one-partner forms (the absent partner is not read; the plain branch drops ``- F``),
  the increment masked and the state written on owned cells only;
* :func:`ladder` -- ``stepping._apply_folded_increment`` without a conductivity:
  ``fu - df`` and ``f - (sinv_u * df)`` on the component's dsigu cells, ``f - df``
  elsewhere, ``df = sinv * increment``, keyed on the chunk record as the curl keys;
* :func:`constitutive_step` -- ``stepping._apply_constitutive_regions`` on real or
  complex storage, keyed on the record, with MEEP's start of H.

MEEP runs the BFAST pass on each component right after its curl and the beta pass
after every component's curl (step_db.cpp:124-142, :161-175); a curl reads its sources
only, so applying each pass after the whole curl sub-step is the same word per cell.
Every product is one float32 rounding per part; nothing here contracts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from meep_gpu.chunk_map import membership_from_record

#: (target index, partner source index, sign) of the beta pass: target 0 takes the
#: SECOND source at +1 (Bx <- Ey, Dx <- Hy), target 1 the FIRST at -1 (By <- Ex,
#: Dy <- Hx); target 2 gets nothing (step_db.cpp:161-175).
BETA_TARGETS: Tuple[Tuple[int, int, float], ...] = ((0, 1, +1.0), (1, 0, -1.0))

#: The record's presence word index (``chunk_map``: 6 lines of 4 words, then the word).
RECORD_PRESENCE_WORD = 24


def membership(xp: Any, record: Sequence[int], half_integer: bool, axis: int,
               shape: Sequence[int]) -> Any:
    """One axis's chunk membership line, broadcast to the grid (the kernels' formula)."""
    line = membership_from_record(np.asarray(record).reshape(-1), half_integer, axis,
                                  int(shape[axis]))
    view = [1, 1, 1]
    view[axis] = int(shape[axis])
    return xp.broadcast_to(xp.asarray(line.reshape(view)), tuple(int(n) for n in shape))


def presence(record: Sequence[int]) -> Tuple[bool, bool, bool]:
    """Partner presence per axis, the record's presence word."""
    word = int(np.asarray(record).reshape(-1)[RECORD_PRESENCE_WORD])
    return tuple(bool((word >> axis) & 1) for axis in range(3))  # type: ignore[return-value]


def _complex(xp: Any, real: Any, imag: Any) -> Any:
    pair = xp.stack((xp.asarray(real, dtype=xp.float32),
                     xp.asarray(imag, dtype=xp.float32)), axis=-1)
    return xp.ascontiguousarray(xp.ascontiguousarray(pair).view(xp.complex64)[..., 0])


def scale(xp: Any, coefficient: Any, values: Any) -> Any:
    """A REAL coefficient times real or complex storage, one product per part.

    A scalar coefficient is a host float32 word, as ``part_products.scale`` takes
    ``realnum(k)``: CuPy's ``asarray`` of a Python ``-0.0`` is ``+0.0`` (measured on
    CuPy 13.5.1), and a BFAST k gated to zero on the D side is exactly that word, whose
    sign reaches the stored state.
    """
    if isinstance(coefficient, (float, int, np.floating, np.integer)):
        c = np.float32(coefficient)
    else:
        c = xp.asarray(coefficient, dtype=xp.float32)
    if values.dtype.kind == "c":
        return _complex(xp, values.real * c, values.imag * c)
    return values * c


def ladder(xp: Any, field: Any, fu: Any, increment: Any, sinv: Any, sinv_u: Any,
           inside: Any) -> None:
    """One folded pass's increment through MEEP's additive ladder, in place.

    ``increment`` is the NEGATED raw word with ``+0`` on unowned cells; every stage
    subtracts it, so an unowned cell keeps its word, ``-0`` included.
    """
    df = scale(xp, sinv, increment)
    recurrence = field - scale(xp, sinv_u, df)
    plain = field - df
    fu[...] = xp.where(inside, fu - df, fu)
    field[...] = xp.where(inside, recurrence, plain)


def beta_increment(xp: Any, partner: Any, word: float) -> Any:
    """``stepping._special_kz_beta_term`` with the host-rounded word, NEGATED."""
    c = np.float32(word)
    if partner.dtype.kind == "c":
        return _complex(xp, partner.imag * c, -(partner.real * c))
    return -(partner * c)


def beta_pass(xp: Any, arrays: Dict[str, Any], coefficients: Dict[str, Any],
              terms: Sequence[Tuple[Any, ...]], half_integer: bool,
              words: Tuple[float, float], record: Sequence[int],
              mask: Callable[[Any, Sequence[int]], None]) -> None:
    """The special-kz beta pass over a sub-step's terms, after every curl, in place.

    ``terms`` are the probe's ``(target, g1, a1, g2, a2, dsig, dsigu, iyee)`` rows;
    ``words`` the plus- and minus-sign words; ``mask(increment, iyee)`` zeroes the
    cells MEEP's owned loop does not visit (the curl's own ownership mask).
    """
    shape = tuple(int(n) for n in arrays[terms[0][0]].shape)
    for index, partner_index, _sign in BETA_TARGETS:
        target, g1, _a1, g2, _a2, dsig, dsigu, iyee = terms[index]
        partner = arrays[(g1, g2)[partner_index]]
        increment = beta_increment(xp, partner, words[index])
        mask(increment, iyee)
        inside = membership(xp, record, half_integer, "xyz".index(dsigu), shape)
        ladder(xp, arrays[target], arrays["fu_" + target], increment,
               coefficients["sinv_" + dsig], coefficients["sinv_" + dsigu], inside)


def bfast_increment(xp: Any, state: Any, shifted_first: Any, first: Any,
                    shifted_second: Any, second: Any, k1: float, k2: float,
                    have_first: bool, have_second: bool, plain: Any,
                    iyee: Sequence[int], mask: Callable[[Any, Sequence[int]], None]
                    ) -> Any:
    """``stepping._bfast_term`` for one component: the NEGATED increment, the state
    advanced in place on owned cells (MEEP writes ``F`` in its owned loop only)."""
    if have_first and have_second:
        updated = (scale(xp, k1, shifted_first + first)
                   - scale(xp, k2, shifted_second + second)) - state
    elif have_first or have_second:
        single = (scale(xp, k1, shifted_first + first) if have_first
                  else scale(xp, k2, shifted_second + second))
        updated = xp.where(plain, single, single - state)
    else:
        return xp.zeros_like(state)
    increment = -(updated - state)
    mask(increment, iyee)
    owned = xp.ones(state.shape, dtype=bool)
    mask(owned, iyee)
    state[...] = xp.where(owned, updated, state)
    return increment


def bfast_words(bfast_k: Sequence[float], first_axis: int, second_axis: int,
                first: str, second: str, have: Tuple[bool, bool],
                magnetic: bool) -> Tuple[float, float]:
    """``(k1, k2)`` of one component (step_db.cpp:129-136): each k is indexed by the
    OTHER partner's own direction, zero where its partner is absent, negated on D."""
    k1 = float(bfast_k["xyz".index(second[-1].lower())]) if have[1] else 0.0
    k2 = float(bfast_k["xyz".index(first[-1].lower())]) if have[0] else 0.0
    if not magnetic:
        k1, k2 = -k1, -k2
    return k1, k2


def constitutive_step(xp: Any, arrays: Dict[str, Any], coefficients: Dict[str, Any],
                      side: str, terms: Sequence[Tuple[str, str, int]],
                      record: Sequence[int], start_h: bool = False) -> None:
    """``stepping._apply_constitutive_regions`` on each component's own axis, in place.

    On a cell whose chunk carries the component's own (dsigw) layer -- the record's
    membership on the integer lattice for H, the half-integer one for E -- the
    accumulation ``f + ((kps*src) - (kms*prev))`` with ``f_w = src``; everywhere else
    ``f = src`` with ``f_w`` untouched. ``src`` is B, or ``D * inv_eps`` per part.
    ``start_h``: MEEP's start of H on the run's first ``update_H`` -- H and ``f_w`` are
    copies of B on the accumulation's cells before it.
    """
    for target, source_name, axis in terms:
        name = "xyz"[axis]
        kps, kms = coefficients["kps_" + name], coefficients["kms_" + name]
        field, fw = arrays[target], arrays["f_w_" + target]
        inside = membership(xp, record, side == "E", axis, field.shape)
        product = (scale(xp, arrays["inv_eps_" + target], arrays[source_name])
                   if side == "E" else arrays[source_name].copy())
        if side == "H" and start_h:
            field[...] = xp.where(inside, product, field)
            fw[...] = xp.where(inside, product, fw)
        previous = fw.copy()
        accumulated = field + (scale(xp, kps, product) - scale(xp, kms, previous))
        field[...] = xp.where(inside, accumulated, product)
        fw[...] = xp.where(inside, product, fw)


def plain_cells(xp: Any, record: Sequence[int], half_integer: bool, dsig: str,
                dsigu: str, shape: Sequence[int]) -> Any:
    """``stepping._bfast_plain_cells`` without a conductivity: the cells in neither the
    component's dsig chunk nor its dsigu chunk."""
    in_dsig = membership(xp, record, half_integer, "xyz".index(dsig), shape)
    in_dsigu = membership(xp, record, half_integer, "xyz".index(dsigu), shape)
    return xp.logical_not(in_dsig) & xp.logical_not(in_dsigu)


__all__ = ["BETA_TARGETS", "RECORD_PRESENCE_WORD", "beta_increment", "beta_pass",
           "bfast_increment", "bfast_words", "constitutive_step", "ladder",
           "membership", "plain_cells", "presence", "scale"]
