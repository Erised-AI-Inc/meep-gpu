"""Verify a per-point susceptibility sigma against MEEP's own ``get_chi1inv``.

THE DECLARED-GEOMETRY PATH, second half. The first half is a *lookup*: MEEP never
subpixel-averages sigma (``Subpixel_Smoothing.md:153``; ``geom_epsilon::sigma_row``,
meepgeom.cpp:1655, point-samples the geometry tree and returns
``susc.sigma_diag`` verbatim), so the sigma at a Yee point is simply the declared
sigma of the medium containing that point. Reproducing that lookup in Python is
easy. Being *sure* it agrees with the structure MEEP actually built is not, and
this module is the part that makes it a READ rather than a reimplementation.

WHAT IS CHECKED, and why it subsumes the hazards. MEEP evaluates, at every point
and every nonzero frequency (``structure_chunk::get_chi1inv_at_pt``,
monitor.cpp:263-352)::

    eps(w) = eps_inf + SUM_n sigma_n * chi_n(w)

with ``eps_inf`` the STORED instantaneous permittivity, which the ``frequency ==
0`` branch returns directly, subpixel averaging included. ``eps_inf`` is read from
MEEP; only ``sigma_n`` comes from the lookup. So evaluating this identity at
several frequencies and comparing against MEEP's own ``get_chi1inv`` there tests
exactly the quantity the lookup supplied, at exactly the point it supplied it for.

There is no ``D_conductivity`` factor in that formula even though monitor.cpp:
338-342 contains one, and the omission is measured rather than assumed. The
conductivity branch reads ``conductivity[cc][dd]`` with ``cc`` an *electric*
component, and ``structure::set_materials`` (structure.cpp:378-380) only ever
fills that array over ``FOR_D_AND_B(c)`` — so the E-row pointer is always NULL and
the branch is unreachable from an E component. Measured on a uniform dispersive
medium with and without ``D_conductivity=0.2``: ``get_chi1inv(Ez, Z, iloc, w)``
returns ``4.53097+0.0353982j`` at w = 0.5 in both cases, bit for bit, while
``get_chi1inv(Dz, Z, iloc, 0.5)`` returns ``1+0.4j`` in the second and ``1+0j`` in
the first. Adding the factor here would therefore make a correct lookup FAIL its
own check — measured, on a two-block cell with a shared ``D_conductivity=0.2``,
at a residual of 3.7e-01.

That single check covers, without needing to be reasoned about separately: the
1-ulp disagreement between MEEP's exported ``is_point_in_object`` and the
predicate ``tree_search`` uses internally; object priority and ``default_material``;
which coordinate the containment test is evaluated at (the trap this module exists
to catch — see :data:`VERIFICATION_TOLERANCE`); a point MEEP does not own, where
it answers vacuum for every frequency; and a periodic image, of an object or of
the lattice point itself, that the lookup did not follow. Each of those turns a
correct-looking sigma into a wrong one at a handful of interface cells, and each
of them moves the residual by six orders or more. The last two are ALSO handled up
front by ``from_meep._meep_reads_here`` — not because the check would miss them,
but because a cell that has to fall back to the inversion for a plane of edge
points is a cell the lookup should simply have got right.

WHAT THE CHECK CANNOT SEE, stated rather than hoped. Two poles that are
indistinguishable over the sampled band make an error in one cancellable by an
error in the other; that is the same degeneracy :mod:`~.sigma_recovery` refuses,
measured the same way (the condition number of the same matrix over the same
band), and it is refused here too rather than left implicit. Away from that
degeneracy the check is quantitative: :attr:`SigmaVerification.sensitivity`
reports, per pole, how much permittivity one unit of sigma is worth at the
strongest sampled frequency, so the tolerance converts directly into the largest
sigma error that can pass.

The sampling band is :func:`~.sigma_recovery.choose_frequencies`, shared with the
inversion path so that both routes ask MEEP about the same frequencies. One
sample per unknown plus one (``oversample=1``) is enough: the check EVALUATES a
hypothesis rather than solving for it, and each complex sample is two real
equations, so ``N + 1`` frequencies over-determine ``N`` sigmas twofold.

This module holds no MEEP dependency at all — it is given ``eps_inf``, the
looked-up sigma and MEEP's measured permittivity as arrays. The MEEP-side half
(the geometry search and the ``chi1inv`` sampling) is
``from_meep._lookup_sigma_volumes``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .sigma_recovery import (
    CONDITION_LIMIT,
    SusceptibilityPole,
    choose_frequencies,
)

# Largest relative disagreement between MEEP's permittivity and the one the
# looked-up sigma predicts, at any sampled frequency and any Yee point, that still
# accepts the lookup.
#
# Measured on this environment (MEEP 1.29.0 and 1.33.0 both store ``realnum`` as
# float32, so a declared sigma of 0.4 comes back as 0.400000005960464 and the
# floor is that rounding rather than zero) — every number below is printed by
# ``test_sigma_lookup.py``:
#
#     two Lorentz/Drude blocks, res 10, 4800 points     1.8e-07
#     two OVERLAPPING blocks, res 10, 4800 points       2.4e-07
#     meep.materials.Au, 6 poles, res 20, 1200 points   1.6e-07
#     a block face at +-1/3, res 30, 10800 points       1.9e-08
#
# and against the one defect this bound exists to catch — moving the containment
# test off the coordinate MEEP snapped the Yee point to by ONE ULP, the size of the
# gap between MEEP's coordinate and this engine's own, which flips 81 of those
# 10800 points — **1.8e-01**. This limit sits 400x above the worst honest residual
# and 1800x below the defect, which is what "the two populations are separated"
# means here.
VERIFICATION_TOLERANCE = 1e-4


class SigmaLookupRefused(RuntimeError):
    """The looked-up sigma does not reproduce MEEP's permittivity, so it is not used.

    Carries the measured disagreement, the frequency and the grid point where it
    was worst. Raised rather than returned because the alternative is a structure
    that differs from MEEP's at a few interface cells and steps a smooth,
    complete, wrong field.
    """


@dataclass
class SigmaVerification:
    """A factored per-point check for one pole set — reusable across every point.

    Everything here depends only on the poles and the sampled frequencies, so it
    is built once and applied to whole components as array arithmetic.
    """

    poles: tuple
    frequencies: np.ndarray
    condition_number: float
    _columns: np.ndarray = field(repr=False)  # (nfreq, npoles): chi_n(w) per unit sigma.

    @property
    def sample_count(self) -> int:
        return int(self.frequencies.size)

    @property
    def sensitivity(self) -> np.ndarray:
        """Per pole, the most permittivity one unit of its sigma is worth in the band.

        ``max_w |chi_n(w)| / sigma_n``. Divide :data:`VERIFICATION_TOLERANCE` by
        this to get the largest error in that pole's sigma that can pass the check
        unnoticed, at a permittivity of order 1.

        Read it RELATIVE to the sigma in question, never as an absolute floor. A
        Drude pole in MEEP's idiom carries ``w0 = 1e-10``, so its sensitivity comes
        out at 3.5e-19 per unit sigma on ``meep.materials.Au`` — which looks blind
        and is not: the sigma beside it is 4.03e+21, so the same bound is a
        relative error of **7e-08** on that pole. What ``chi1`` depends on is the
        product ``sigma*w0**2``, and the sensitivity carries the ``w0**2``.
        """
        return np.max(np.abs(self._columns), axis=0)

    def predict(self, eps_inf, sigma) -> np.ndarray:
        """MEEP's own ``eps(w)`` for the given per-point sigma, at every sampled frequency.

        Transcribed from ``structure_chunk::get_chi1inv_at_pt`` — the susceptibility
        loop at monitor.cpp:322-336, which is the function the measured side of the
        comparison comes out of. The conductivity factor two lines below it is
        deliberately absent; the module docstring carries the measurement that says
        why it never applies to an E row.

        Args:
            eps_inf: ``(npoints,)`` the STORED instantaneous permittivity, i.e.
                ``1 / get_chi1inv(c, d, iloc, 0)``. Read from MEEP, not predicted:
                it already carries whatever subpixel average MEEP applied.
            sigma: ``(npoles, npoints)`` the looked-up sigma per pole.

        Returns:
            ``(nfreq, npoints)`` complex permittivity.
        """
        eps_inf = np.asarray(eps_inf, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(len(self.poles), -1)
        return eps_inf[None, :] + self._columns @ sigma

    def residuals(self, eps_inf, sigma, measured) -> np.ndarray:
        """``(nfreq, npoints)`` relative disagreement against MEEP's measured permittivity.

        Scaled by ``max(|measured|, |predicted|, 1)`` — the 1 is vacuum, the natural
        unit of a permittivity, and it keeps a point where MEEP reports eps near
        zero from reporting an enormous relative error for an absolute one that is
        at the float32 floor.

        A point where either side is not finite scores ``inf`` rather than ``nan``.
        The distinction matters: ``nan`` compares False against every bound, so a
        check written the obvious way would ACCEPT it, and an overflowing prediction
        (MEEP's Drude sigma reaches 4e+21) is exactly where that would happen.
        """
        predicted = self.predict(eps_inf, sigma)
        measured = np.asarray(measured, dtype=np.complex128).reshape(predicted.shape)
        difference = np.abs(predicted - measured)
        scale = np.maximum(np.maximum(np.abs(measured), np.abs(predicted)), 1.0)
        usable = np.isfinite(difference) & np.isfinite(scale) & (scale > 0.0)
        return np.divide(difference, scale, out=np.full(difference.shape, np.inf), where=usable)

    def require(
        self,
        eps_inf,
        sigma,
        measured,
        where: str = "",
        locate=None,
        tolerance: float = VERIFICATION_TOLERANCE,
    ) -> float:
        """Check every point, and refuse — with the number — if any of them fails.

        Args:
            where: A label for the message (the component being checked).
            locate: ``index -> str`` naming the worst point, so the refusal says
                which Yee point disagreed rather than only by how much.

        Returns:
            The worst relative residual, for the caller to record.

        Raises:
            SigmaLookupRefused: if that worst residual exceeds ``tolerance``.
        """
        residuals = self.residuals(eps_inf, sigma, measured)
        flat = int(np.argmax(residuals)) if residuals.size else 0
        worst = float(residuals.flat[flat]) if residuals.size else 0.0
        if not np.isfinite(worst) or worst > tolerance:
            frequency_index, point_index = np.unravel_index(flat, residuals.shape)
            point = locate(int(point_index)) if locate is not None else f"index {int(point_index)}"
            raise SigmaLookupRefused(
                f"the sigma looked up from the declared geometry does not reproduce MEEP's own "
                f"permittivity on {where or 'this component'}: worst relative disagreement "
                f"{worst:.3e} (limit {tolerance:g}) at {point}, frequency "
                f"{float(self.frequencies[frequency_index]):g}. MEEP point-samples sigma from the "
                f"geometry tree (meepgeom.cpp:1655) and this lookup reproduces that sample, so a "
                f"disagreement means the two are not sampling the same thing — a containment test "
                f"evaluated at a coordinate MEEP did not snap the Yee point to, an object "
                f"precedence or default_material this search resolved differently, or a periodic "
                f"image of an object that the declared list does not carry. The installed "
                f"structure would differ from MEEP's at those cells and step a smooth complete "
                f"field for it, so no sigma is returned from this route."
            )
        return worst


def plan_verification(
    poles: Sequence[SusceptibilityPole],
    frequencies: Optional[Sequence[float]] = None,
    oversample: int = 1,
) -> SigmaVerification:
    """Build the check for one pole set, refusing a set the check could not see through.

    ``oversample=1`` gives ``N + 1`` frequencies, one per unknown plus one. That is
    the minimum the INVERSION needs to solve; the check only evaluates, and each
    complex sample is two real equations, so the same grid over-determines the
    ``N`` looked-up sigmas twofold.

    Raises:
        SigmaLookupRefused: when two poles are indistinguishable over the band
            (condition number past :data:`~.sigma_recovery.CONDITION_LIMIT`), where
            an error in one sigma is cancellable by an error in the other and the
            residual would stay small for a wrong lookup.
    """
    poles = tuple(poles)
    if not poles:
        raise SigmaLookupRefused("no susceptibilities to verify.")
    grid = (np.asarray(frequencies, dtype=float) if frequencies is not None
            else choose_frequencies(poles, oversample))
    # Per unit of SIGMA, not of the inversion's c = sigma*w0**2: the lookup hands
    # over sigma itself, so the columns are MEEP's chi1(w, sigma=1) directly.
    columns = np.array([[pole.basis(np.array([w]))[0] * pole.scale for pole in poles]
                        for w in grid])
    # The degeneracy question is about the SHAPES of the poles over the band, not
    # their magnitudes — MEEP's Drude idiom carries sigma = 4e+21 beside a Lorentz
    # sigma of 0.4, and an unnormalized condition number would report that spread
    # rather than any confusability. Normalising each column is the same
    # reparameterization sigma_recovery.plan_recovery makes for the same reason.
    # Measured over this band: **1.4e+00** for a single Lorentz pole, **4.5e+00**
    # for the two-block Lorentz+Drude pair, **5.2e+00** for meep.materials.Au's six
    # — all of them seven orders inside the limit, so the refusal below fires on
    # duplicated poles rather than on real materials.
    scaled = columns / np.maximum(np.max(np.abs(columns), axis=0), 1e-300)[None, :]
    stacked = np.concatenate([
        np.concatenate([np.ones((grid.size, 1)), scaled.real], axis=1),
        np.concatenate([np.zeros((grid.size, 1)), scaled.imag], axis=1),
    ], axis=0)
    condition = float(np.linalg.cond(stacked))
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        raise SigmaLookupRefused(
            f"two of these {len(poles)} susceptibilities are indistinguishable over the sampled "
            f"band [{grid[0]:g}, {grid[-1]:g}] (condition number {condition:.3e}, limit "
            f"{CONDITION_LIMIT:.0e}). A wrong sigma on one of them could be cancelled by a wrong "
            f"sigma on the other and still reproduce MEEP's permittivity at every sampled "
            f"frequency, so the per-point check would accept a lookup it cannot actually see "
            f"through. Nothing is verified and nothing is installed from this route."
        )
    return SigmaVerification(
        poles=poles,
        frequencies=grid,
        condition_number=condition,
        _columns=columns,
    )


def sigma_of_medium(medium, keys, component_axis: int, key_of) -> np.ndarray:
    """One medium's declared sigma for each chain entry, on one E component's row.

    ``geom_epsilon::sigma_row`` (meepgeom.cpp:1680-1705) walks the medium's
    ``E_susceptibilities`` looking for the one entry ``susceptibility_equiv``
    considers the same as the chain entry being filled, and takes
    ``sigma_diag[component_index(c)]`` off it; a medium that declares no such
    entry contributes exactly zero (``sigrow[] = 0`` before the loop). This is
    that function, for all chain entries at once.

    Args:
        keys: the equivalence key of each chain entry, in the order the volumes
            are stacked.
        component_axis: 0, 1 or 2 — MEEP's ``component_index(c)``, which selects
            which diagonal entry of the sigma tensor this row carries.
        key_of: the equivalence-key function, injected so that all three sigma
            routes provably ask the same question (``from_meep
            ._susceptibility_equivalence_key``).
    """
    values = np.zeros(len(keys), dtype=np.float64)
    declared = getattr(medium, "E_susceptibilities", ()) or ()
    for index, key in enumerate(keys):
        for term in declared:
            if key_of(term) == key:
                sigma_diag = term.sigma_diag
                values[index] = float((sigma_diag.x, sigma_diag.y, sigma_diag.z)[component_axis])
                break  # meepgeom breaks at the first equivalent entry; so does this.
    return values
