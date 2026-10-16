"""MEEP's real-field PML forms, selected per structure chunk: the shared Triton helpers.

Every helper here is ``@triton.jit`` and is inlined into the kernel that calls it, so a
helper boundary is not a rounding boundary: the kernel compiles one program and every
operation in it rounds to float32 once, under the launch's ``enable_fp_fusion=False``
(:data:`.kernels.ENABLE_FP_FUSION`). They are the one spelling of each form for the
kernels of :mod:`.kernels`; a kernel calls them rather than restating an expression.

WHAT EACH HELPER TRANSCRIBES (the array path, which transcribes MEEP 1.33.0):

* :func:`chunk_in` -- membership of a stored row in one line of the chunk record
  (``chunk_map.chunk_bounds_record``): ``(a0 <= i < a1) or (b0 <= i < b1)``, which is
  ``chunk_map.membership_from_record`` on the device. The run's
  ``chunk_map.RegionPlan`` keys its forms on the same lines, so the kernels and the
  array path read one predicate.
* :func:`record_in` -- :func:`chunk_in` on the line that starts at word ``base`` of the
  record (``chunk_map.record_index(half_integer, axis, 0)``), read with four scalar
  loads.
* :func:`axis_present` -- bit ``axis`` of the record's partner-presence word
  (``chunk_map.RECORD_PRESENCE_WORD``, which the caller loads): is a curl partner
  whose derivative runs along that axis present (``stepping.curl_presence``, MEEP's
  ``have_plus_deriv`` / ``have_minus_deriv``, fields.cpp:428-453)?
* :func:`curl_stencil` -- ``stepping._curl_from_operands`` (step_generic.cpp:72-76,
  :104, :108): ``dtdx * (((g1s - g1) + g2) - g2s)`` with both partners, which C
  associates left to right; ``dtdx * (g1s - g1)`` without ``g2``; ``(-dtdx) * (g2s -
  g2)`` without ``g1``; an exact ``+0`` without either, which is the array path's
  zero-filled buffer. The full bracket is NOT the absent-partner word even where the
  absent operand's difference is an exact zero: ``(a + g2) - g2`` rounds twice more
  than ``a`` does, which is what a reduced-dimension grid steps on.
* :func:`curl_form` -- ``stepping._apply_curl_regions`` without a conductivity: on a
  cell whose chunk carries the component's dsigu layer, the split-field recurrence
  (``stepping._apply_pml_update``, step_generic.cpp:96-107)::

      n = ((fu * kms) - curl) * sinv
      f = (((f * kms_u) + n) - fu) * sinv_u
      fu = n

  everywhere else the dsig-only form on the field alone
  (``stepping._apply_dsig_only_update``, step_generic.cpp:183), ``((f * kms) - curl) *
  sinv``, which IS MEEP's plain ``f - curl`` (:104) wherever ``kms = sinv = 1.0f``: the
  region plan's unit-coefficient precondition, checked when the plan is built. The
  returned auxiliary word is the caller's to store on the recurrence's cells ONLY --
  the auxiliary contract. ``fu`` keeps its word everywhere else, which is why the
  caller masks the auxiliary store and not the field store.
* :func:`constitutive_form` -- ``stepping._apply_constitutive_regions``: on a cell whose
  chunk carries the component's own (dsigw) layer, MEEP's accumulation
  (``stepping._apply_constitutive_pml``, step_generic.cpp:689, :697) ``f + ((kps * src)
  - (kms * prev))``; everywhere else ``src`` itself (update_eh.cpp:162-170: H is B's
  word, E is ``(D - P) * chi1inv``). ``f_w`` takes ``src`` on the accumulation's cells
  only, again the caller's masked store.
* :func:`fold_ladder` -- ``stepping._apply_folded_increment``: a folded pass (the
  special-kz beta pass, the BFAST pass) adds its increment to the component through
  MEEP's additive ladder after the curl form, keyed on the same dsigu chunk:
  ``fu - df`` and ``f - (sinv_u * df)`` on the recurrence's cells, ``f - df``
  elsewhere, with ``df = sinv * increment``.
* :func:`bfast_advance` -- ``stepping._bfast_term``: the BFAST state's two roundings,
  MEEP's one-partner forms (the absent partner is not read; the plain branch drops
  ``- F``) and the increment the ladder takes.

THE SELECTS ARE OUTPUT SELECTS. ``tl.where`` chooses a word and does no arithmetic, so
the discarded form -- computed on whatever the masked loads delivered -- never reaches a
store. The plain forms are therefore selected, never folded into the others with unit
or zero coefficients: ``kps = 1, kms = 0`` turns a stored ``-0`` into ``+0``, and
``fu := 0`` in the recurrence does the same to ``f = -0, curl = +0``.

MULTIPLICATION OPERAND ORDER IS FREE AND ADDITION ORDER IS NOT. IEEE-754
multiplication is commutative on every word, signed zero included, so ``fu * kms`` and
the array path's ``fu *= kms`` are one word; every sum and difference here is grouped
exactly as the array path groups it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def chunk_in(i, a0, a1, b0, b1):
    """Is stored row ``i`` in ``[a0, a1) U [b0, b1)``? An empty run is ``[0, 0)``."""
    return ((i >= a0) & (i < a1)) | ((i >= b0) & (i < b1))


@triton.jit
def record_in(cb, base, i):
    """:func:`chunk_in` on the record line whose ``a0`` is word ``base`` of ``cb``.

    ``base`` is a compile-time constant in every caller (``chunk_map.record_index`` of
    the component's (dsigu or dsigw) axis and its sub-lattice), so the four loads are
    four scalar reads of the uploaded ``int32[25]`` and the comparisons broadcast over
    the block.
    """
    return chunk_in(i, tl.load(cb + base), tl.load(cb + base + 1),
                    tl.load(cb + base + 2), tl.load(cb + base + 3))


@triton.jit
def axis_present(presence, axis):
    """Bit ``axis`` of the presence word: a curl partner differenced along it exists."""
    return ((presence >> axis) & 1) != 0


@triton.jit
def curl_stencil(g1s, g1, g2, g2s, dtdx, first, second):
    """MEEP's ``dtdx``-scaled curl for one component, by partner presence (A1)."""
    both = dtdx * (((g1s - g1) + g2) - g2s)
    only_first = dtdx * (g1s - g1)
    only_second = (-dtdx) * (g2s - g2)
    return tl.where(first & second, both,
                    tl.where(first, only_first,
                             tl.where(second, only_second, 0.0)))


@triton.jit
def curl_form(f, fu, curl, kms, sinv, kms_u, sinv_u, dsigu):
    """``(field word, auxiliary word)``: the recurrence on dsigu cells, dsig-only elsewhere.

    The auxiliary word is MEEP's ``fu[i]`` after the recurrence. It is meaningless off
    the recurrence's cells and must not be stored there: see the module docstring.
    """
    n = ((fu * kms) - curl) * sinv
    recurrence = (((f * kms_u) + n) - fu) * sinv_u
    dsig_only = ((f * kms) - curl) * sinv
    return tl.where(dsigu, recurrence, dsig_only), n


@triton.jit
def constitutive_form(f, prev, src, kps, kms, dsigw):
    """The field word: the accumulation on dsigw cells, ``src`` elsewhere."""
    return tl.where(dsigw, f + ((kps * src) - (kms * prev)), src)


@triton.jit
def fold_ladder(f, fu, increment, sinv, sinv_u, dsigu):
    """``(field word, auxiliary word)`` after one folded pass: MEEP's additive ladder.

    ``stepping._apply_folded_increment`` without a conductivity, which is the tail the
    special-kz beta pass (step_generic.cpp:255-333) and the BFAST pass (:335-533) share,
    keyed on the component's dsigu chunk exactly as :func:`curl_form` keys::

        df = sinv * increment                   # sinv of the dsig axis
        dsigu cell:  fu = fu - df;  f = f - (sinv_u * df)
        elsewhere:   f = f - df

    ``increment`` is the pass's NEGATED raw word (the curl sign convention), with an
    exact ``+0`` on every cell MEEP's owned loop does not visit: the caller masks it as
    the array path masks it (``stepping._mask_non_owned_cells``). Every stage SUBTRACTS
    it, which is the array path's spelling and MEEP's ``f += raw`` word for word
    (``x - (-y)`` is ``x + y`` and ``(-a) * b`` is ``-(a * b)`` exactly), while
    ``x - (+0)`` is ``x`` for every ``x``, ``-0`` included -- so an unowned cell keeps
    its word. Adding a masked ``+0`` instead would turn a stored ``-0`` into ``+0``.

    The pass follows its component's curl form IN REGISTER, before any store: a float32
    stored and reloaded is the identity on its bits, so applying the ladder to the
    registers :func:`curl_form` returned is the same two roundings as MEEP's separate
    loop over the stored words. ``sinv`` is exactly 1 outside the chunks that carry the
    dsig layer (the region plan's unit-coefficient precondition), so ``sinv * increment``
    IS the increment there. The returned auxiliary word, like :func:`curl_form`'s, is
    the caller's to store on the recurrence's cells ONLY.
    """
    df = increment * sinv
    return tl.where(dsigu, f - (sinv_u * df), f - df), fu - df


@triton.jit
def bfast_advance(state, g1s, g1, g2s, g2, k1, k2, first, second, layer):
    """``(state word, increment)`` of MEEP's BFAST pass for one component.

    ``stepping._bfast_term`` (step_db.cpp:129-142, step_generic.cpp:335-533). With both
    partners the state takes MEEP's two roundings, the new state stored and the
    increment a second rounding of its difference from the previous one::

        updated   = (k1 * (g1s + g1) - k2 * (g2s + g2)) - state
        increment = -(updated - state)

    With ONE partner (``first`` xor ``second``, the record's presence bits) MEEP hands
    ``step_bfast`` a NULL for the other and swaps it into the ``g2`` slot
    (step_generic.cpp:342-346), so the surviving partner's term alone is read:
    ``single = k1 * (g1s + g1)`` or ``k2 * (g2s + g2)``. In MEEP's PLAIN branch -- a
    chunk carrying neither the component's dsig nor its dsigu layer (``layer`` false)
    -- the state is ``single`` itself, with no ``- state`` (:373-379); everywhere else
    ``single - state``. With NEITHER partner the component is not stepped: the state
    keeps its word and the increment is an exact ``+0``.

    The negation is spelled ``* -1.0``, never unary minus, which Triton lowers as
    ``0.0 - x`` and which turns ``-(+0)`` into ``+0``. The caller masks the increment
    to ``+0`` on unowned cells and stores the state on owned cells only, as MEEP writes
    ``F`` inside its owned loop.
    """
    s1 = k1 * (g1s + g1)
    s2 = k2 * (g2s + g2)
    both = (s1 - s2) - state
    single = tl.where(first, s1, s2)
    one = tl.where(layer, single - state, single)
    updated = tl.where(first & second, both, tl.where(first | second, one, state))
    increment = tl.where(first | second, (updated - state) * -1.0, 0.0)
    return updated, increment
