"""Mutation-test the mp.Simulation converter: break it on purpose, one edit at a time.

A green suite proves the code passes its tests; it does not prove the tests would
notice if the code were wrong. Each mutation below is a plausible one-line
converter bug — a dropped attribute, an enum read the intuitive way, a
registration off by a cell — applied to a copy of ``from_meep.py``. The battery
passes only when EVERY mutation makes at least one test fail. A surviving mutation
names a defect the suite cannot see.

The mutations are chosen to be silent: none of them crashes, and every one leaves
a smooth, finite, complete field behind. That is the point.

Rule 7: one flushed line per mutation, results appended as they land.

Usage (from ``the repository root``)::

    python -u -m parity.meep_gpu.mutate_from_meep
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODULE = Path("meep_gpu/from_meep.py")
DEFAULT_TESTS = ("meep_gpu/test_from_meep.py",)

# name -> (exact text to find, replacement, pytest -k selector[, module[, test files]]).
# The selector keeps each mutation to the tests that could plausibly catch it, so the
# battery runs in minutes instead of hours; "" means the whole target suite.
#
# The last two elements are optional and default to the converter, which is what the
# original battery mutated. They exist because the boundary and reduced-dimension
# physics do NOT live in from_meep.py — a metallic wall is implemented in stepping.py
# and declared in grid.py — and a battery that can only edit the converter cannot
# reach the code where those defects would actually sit.
MUTATIONS: dict[str, tuple] = {
    # --- registration: the defect class this package has hit most often ----------
    "chi1inv_sampled_one_cell_high": (
        "exact = numpy.asarray(to_numpy(coordinates), dtype=numpy.float64) * lattice_per_unit",
        "exact = (numpy.asarray(to_numpy(coordinates), dtype=numpy.float64) + grid.dx) "
        "* lattice_per_unit",
        "structured or mis_sampled",
    ),
    "chi1inv_sampled_half_cell_high": (
        "exact = numpy.asarray(to_numpy(coordinates), dtype=numpy.float64) * lattice_per_unit",
        "exact = (numpy.asarray(to_numpy(coordinates), dtype=numpy.float64) + 0.5 * grid.dx) "
        "* lattice_per_unit",
        "structured or mis_sampled",
    ),
    "chi1inv_registration_pin_removed": (
        "if int(rounded[0]) != first_expected[axis]:",
        "if False:",
        "mis_sampled",
    ),
    "chi1inv_lattice_rounded_instead_of_pinned": (
        "if drift > _LATTICE_TOLERANCE:",
        "if False:",
        "mis_sampled or structured",
    ),
    "chi1inv_yee_shift_ignored": (
        "first_expected = (corner.x() + wanted[0], corner.y() + wanted[1], corner.z() + wanted[2])",
        "first_expected = (corner.x(), corner.y(), corner.z())",
        "structured",
    ),
    "bulk_cross_check_removed": (
        "    _require_bulk_matches_meep(sim, driver, volumes, declined=declined_any)",
        "    pass  # mutation",
        "structured or mis_sampled",
    ),
    # --- the tensor: what MEEP's anisotropic averaging does that this cannot -----
    # The off-diagonal chi1inv rows are INGESTED now, so the mutations guard the
    # ingestion instead of the refusal it replaced. Dropping the rows reproduces
    # the old 5.7e-02-class diagonal-only run; flipping their sign is the
    # conjugate/orientation slip that keeps every magnitude plausible.
    "offdiagonal_chi1inv_dropped_on_ingest": (
        "            if numpy.any(entries):",
        "            if False:",
        "cylinder_smoothed or rotated_block_smoothed or uniform_epsilon_offdiag",
    ),
    "offdiagonal_chi1inv_sign_flipped_on_ingest": (
        "                        coupling[axis][i, j, k] = entry.real",
        "                        coupling[axis][i, j, k] = -entry.real",
        "cylinder_smoothed or rotated_block_smoothed",
    ),
    "rotated_block_treated_as_axis_aligned": (
        "    axes = (_vector3(obj.e1), _vector3(obj.e2), _vector3(obj.e3))",
        "    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))",
        "refused_by_name",
    ),
    # --- the tensor stencil itself, in stepping.py ---------------------------------
    # The "stable averaging" pairs each two-point partner average with the
    # coefficient AT ITS OWN NODE (u at i and i+s); collapsing that to a plain
    # four-point field average times u[i] is identical arithmetic wherever the
    # coefficient is uniform — the uniform-tensor oracles cannot see it — and
    # wrong exactly on the material interfaces the coefficient exists for.
    "offdiagonal_average_unstabilized": (
        "        product = pair * coefficient\n"
        "        # (pair*u)[i] + (pair*u)[i+s]: the same product half a cell UP this axis.\n"
        "        term = 0.25 * (product + _shift_up(xp, product, own_axis,\n"
        "                                           boundaries[own_axis], phases[own_axis]))",
        "        product = pair * coefficient\n"
        "        # (pair*u)[i] + (pair*u)[i+s]: the same product half a cell UP this axis.\n"
        "        term = 0.25 * (pair + _shift_up(xp, pair, own_axis,\n"
        "                                        boundaries[own_axis], phases[own_axis])) * coefficient",
        "cylinder_smoothed or rotated_block_smoothed or hand_loop",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_from_meep.py", "meep_gpu/test_tensor_epsilon.py"),
    ),
    "offdiagonal_partner_neighbour_dropped": (
        "        # g[i] + g[i-sx]: half a cell DOWN the partner's own axis.\n"
        "        pair = values + _shift_down(xp, values, partner_axis, boundaries[partner_axis],",
        "        # g[i] + g[i-sx]: half a cell DOWN the partner's own axis.\n"
        "        pair = values + values + 0 * _shift_down(xp, values, partner_axis, boundaries[partner_axis],",
        "hand_loop or uniform_tensor",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_tensor_epsilon.py",),
    ),
    # --- cylindrical: each entry is a defect class actually hit while landing it --
    # The PML builder's restated wraps rule graded the axis row's integer position 0
    # with the outer wall's peak sigma (kps[0]=2.295); invisible at m=0, 2.3x on the
    # stored Hr at m=1. The one-definition deferral is the fix; this reinstates it.
    "cylindrical_pml_axis_wraps_restated": (
        "        return self.grid.axis_wraps(axis)",
        "        return not self.grid.is_mirrored(axis)",
        "cylindrical_pml",
        Path("meep_gpu/pml.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The B-side prefix is a cumsum whose far ghost must be the ZERO wall row;
    # wrapping the axis row there instead reproduces the last-two-rows divergence
    # class (born at step 3, grew inward to 4.9e-01, invisible at m=0).
    # Re-anchored 2026-08-09: the step-loop allocation work replaced the
    # `zeros_like` + `concatenate` pair this used to match with a pooled `extended`
    # buffer written in two statements. The MUTATION is unchanged — the wall row
    # reads the live axis row instead of zero — only the line it edits moved.
    "cylindrical_prefix_wall_row_wrapped": (
        "        extended[_face(0, rows)] = 0",
        '        extended[_face(0, rows)] = electric["Ey"][_face(0, 0)]',
        "cylindrical_m1",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The readback's periodic far-wall wrap is exact-by-accident on Cartesian
    # metallic axes and read the LIVE axis row across the PEC wall on cylindrical
    # (Ez 1.27e-01 at m=0).
    "cylindrical_readback_wall_wrapped": (
        "            if self.grid.is_mirrored(axis) or self.grid.is_axis(axis):",
        "            if self.grid.is_mirrored(axis):",
        "cylindrical_m0",
        Path("meep_gpu/fields.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The i*m/r coupling's -i is MEEP's (1-2*cmp) real/imag swap; +i is the
    # conjugate run — smooth, plausible, rotating the wrong way.
    # Re-anchored 2026-08-09: the step-loop allocation work moved this factor into a
    # memoized builder, so the expression is now the closure's return and wraps a
    # line. The MUTATION is unchanged — the coupling runs conjugated.
    "cylindrical_imr_conjugated": (
        "        return ((-1j) * (sign * 2.0 * grid.m * (grid.dt / grid.dx))",
        "        return ((+1j) * (sign * 2.0 * grid.m * (grid.dt / grid.dx))",
        "cylindrical_m1",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The m=0 axis rule's factor 4 is the analytic limit of (1/r)d(r*Hp)/dr plus
    # the doubled-ivec compensation; halving it is a plausible transcription slip.
    "cylindrical_axis_limit_halved": (
        '                                    4.0 * (grid.dt / grid.dx))',
        '                                    2.0 * (grid.dt / grid.dx))',
        "cylindrical_m0",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The flux ring weight's +0.5 is the centered-grid half-cell: MEEP evaluates
    # 2*pi*r at each sample's own ring (loop_in_chunks.cpp:505-515), and samples
    # sit at (j+0.5)*dx. Dropping it weights every ring at the row BELOW it — the
    # off-by-half a porter writes reading "position 0" as r=0.
    # Re-anchored when the near-field surface grew its own ring weight: the two sites
    # now spell the same formula, so each anchor carries the line that follows it.
    "cylindrical_flux_ring_at_integer_r": (
        "        ring = 2.0 * np.pi * (starts[0] + rows + 0.5) * grid.dx\n"
        "        axis_weights[0] = axis_weights[0]",
        "        ring = 2.0 * np.pi * (starts[0] + rows) * grid.dx\n"
        "        axis_weights[0] = axis_weights[0]",
        "cylindrical_flux",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # The SAME half-cell, on the raw-Yee near2far accumulator rather than the
    # cell-centred flux monitor — and here the centred form is the wrong one. A
    # `YeeRegionDFT` stores the component's own Yee sites, so its ring radius is
    # `(origin_doubled + 2j + parity)/2 * dx`; borrowing `(j + 0.5)*dx` from
    # `_face_weights` (which is correct THERE, because that monitor samples centres)
    # moves every shift-0 component — Ep, Hr, Ez — half a cell outward and puts the
    # axis row at 2*pi*dx/2 where MEEP has exactly 0. Nothing raises: the far field
    # stays smooth and plausible, and MEEP's own dV0 for those chunks is 0.
    "greencyl_ring_at_the_cell_centre_not_the_yee_radius": (
        "    return weights * (2.0 * np.pi * (doubled.astype(np.float64) * 0.5 * dx))",
        "    rows = start + np.arange(count, dtype=np.float64)\n"
        "    return weights * (2.0 * np.pi * (rows + 0.5) * dx)",
        "yee_radius or cylindrical_near2far",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # A one-pixel periodic axis makes MEEP's loop_in_chunks store one near2far chunk
    # per LATTICE IMAGE of the single cell — identical bounds, single-site weights
    # w0 and w1 = 1 - w0 (measured 0.0 and 1.0 on binary_grating_n2f's point
    # region). This engine holds ONE plane at the summed weight, so each chunk is
    # emitted scaled by its own s0; dropping that scale hands BOTH images the full
    # plane, which doubles the surface — a smooth, plausible far field at exactly
    # 2x (measured 1.00e+00 relative, parity 5.05e-07). Nothing raises, every size
    # check passes, and 2x in amplitude is invisible without a reference.
    "near2far_lattice_image_weight_dropped": (
        "                                          image_weight=image_weight,",
        "                                          image_weight=1.0,",
        "point_near2far",
    ),
    # A flux-box CORNER site lies in two regions carrying the same component at
    # OPPOSITE stored weights (the tangential-pair sign times w->weight), and MEEP
    # keeps one single-site chunk per region there. The stored-weight filter is what
    # routes each chunk to ITS region's accumulator; matching by containment alone
    # emits the corner with the other face's sign — measured exactly 2.000e+00 on
    # that one chunk and 4.26e-02 on the stored total while every other chunk of the
    # run sat at 1e-06. Two coincident regions at different weights (the
    # `duplicate_plane` cell) carry the same `vc`, so nothing else can separate them:
    # measured 6.32e-01 against a 3.45e-07 floor.
    "near2far_corner_stored_weight_filter_dropped": (
        "            if stored_weight is not None and not _same_stored_weight(complex(weight),",
        "            if False and not _same_stored_weight(complex(weight),",
        "point_near2far or corner_claimants",
    ),
    # The SAME filter, defeated one level down rather than removed: comparing the two
    # weights through abs(a - b) is the natural spelling, and it cannot see that
    # (1+0j) and (1-0j) are MEEP's labels for two different near2far regions — their
    # difference has modulus exactly 0.0. Measured 2.69e-02 on antenna-radiation.py's
    # packed data and, under the double fold, 1.40e-03 in the far field.
    "near2far_corner_weight_compared_through_abs": (
        "        if abs(ours - theirs) > 1e-12 * (1.0 + abs(theirs)):\n"
        "            return False\n"
        "        if ours == 0.0 and theirs == 0.0 and math.copysign(1.0, ours) != math.copysign(1.0, theirs):\n"
        "            return False",
        "        if abs(ours - theirs) > 1e-12 * (1.0 + abs(theirs)):\n"
        "            return False",
        "same_stored_weight_separates or corner_claimants",
    ),
    # The entry weight built with Python's PROMOTED product instead of MEEP's
    # componentwise one. Every value is identical; only the sign of a zero imaginary
    # part changes, and that sign is the region label the corner filter reads.
    "near2far_stored_weight_promoted_not_componentwise": (
        "    return complex(sign * weight.real, sign * weight.imag)",
        "    return sign * weight",
        "stored_weight_is_meeps or corner_claimants",
    ),
    # MEEP's equivalent-source label `vc` dropped from the claimant filter. With real
    # region weights the signed zero still separates the corner, so this is silent
    # there; with complex weights the two claimants' stored weights are identical bit
    # for bit and only `vc` separates them — measured 1.84e-02 against 7.11e-07.
    "near2far_chunk_source_component_ignored": (
        "            if source_component is not None and source != source_component:",
        "            if False:",
        "corner_claimants",
    ),
    # MEEP's volume SORTS its corners (vec.cpp:163-167), so a negative size component
    # names the same span. Passing the sign through reaches the monitors as an
    # inverted span and raises on a simulation the gate accepted — both of MEEP's own
    # test_cavity_farfield cases, which write size = 2*dpml - sx = -10.4.
    "negative_region_size_passed_through": (
        "        0.0 if is_invariant(axis) else abs(float(size[axis])) for axis in range(3)",
        "        0.0 if is_invariant(axis) else float(size[axis]) for axis in range(3)",
        "negative_size_is_meeps",
    ),
    # MEEP allows a FluxRegion with extent along its DECLARED normal — a flux volume,
    # which its own test_visualization.py builds. Restoring the flat-axis refusal is
    # the accepted-by-the-gate, raises-on-lift hole this closed.
    "flux_volume_refused_on_declared_normal": (
        "            # A DECLARED normal may have extent: MEEP allows a flux VOLUME and its own",
        "            if plane_size[normal] >= self.grid.dx: raise ValueError(\"mutation\")\n"
        "            # A DECLARED normal may have extent: MEEP allows a flux VOLUME and its own",
        "extend_along_its_own_normal",
        Path("meep_gpu/driver.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # --- the two-run normalization idiom's MINUS convention -------------------------
    # MEEP's load_minus_flux_data (python/simulation.py:3642-3649) loads the saved
    # transform and calls scale_dfts(-1), so the second run ends at E2 - E1 and
    # H2 - H1 and reports Re[(E2-E1) x conj(H2-H1)] — the reflected wave's own power,
    # not a difference of two powers. Both defects below leave the run smooth, finite
    # and complete, and both return a reflectance in the right decade.
    # --- result readers: get_dft_array and the flux packer -----------------------
    # `sim.get_dft_array` collapses a zero-extent axis by INTERPOLATING onto the
    # requested coordinate, and the bracket it interpolates between is the requested
    # COMPONENT's own Yee lattice — `loop_in_chunks` shifts by
    # `iyee_shift(Centered) - iyee_shift(cgrid)` before flooring, and `add_dft` hands
    # it `use_centered_grid ? Centered : c` (dft.cpp:231). Measuring the fraction from
    # the cell centres instead is right for every CENTRED monitor, which is what makes
    # it the plausible edit: the box case, the mirrored case and every add_dft_monitor
    # row stay exact, and only a `yee_grid=True` monitor with a collapsed axis moves —
    # measured 1.67e-01 on a zero-thickness line and 2.04e-01 on a point, both smooth,
    # both finite, both the right shape.
    "dft_array_collapse_bracketed_on_the_centred_lattice": (
        "            first_doubled = tuple(accumulator.site_doubled(axis, accumulator.first_index(axis))\n"
        "                                  for axis in range(3))",
        "            from .dft import yee_shifts as _centred_lattice\n"
        "            first_doubled = tuple(\n"
        "                accumulator.site_doubled(axis, accumulator.first_index(axis))\n"
        "                - _centred_lattice(component)[axis] + 1 for axis in range(3))",
        "get_dft_array",
    ),
    # MEEP folds `w = IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0)` into every stored E
    # sample and divides it back out on read (dft.cpp:276-281, :1003-1006), and that
    # ladder has FOUR distinct edge terms per axis, not two: `s0` at i=0 and `s1` at
    # i=1, `e0` at i=n-1 and `e1` at i=n-2 (vec.hpp:372-378). Reading the
    # second-from-edge terms as the edge ones is the plausible simplification, and the
    # H side (no measure at all) and the site mapping both stay exact under it, so
    # only the E numbers say anything. Measured on the three packer cases: the packed
    # E moves 7.29e-02 / 7.32e-02 / 2.82e-01, but what MEEP's reader ANSWERS barely
    # moves on the two full-plane lists (alpha 2.53e-07 and 1.97e-05) and moves a lot
    # on `symmetry::reduce`'s half plane — alpha 7.77e-02, flux 1.03e-01 — because
    # the halved region's new inner edge measures 0.875 and 0.125 in the two slots
    # this mutation conflates. So the E comparison is what catches it early, and the
    # reduced case is what makes it a wrong answer rather than a wrong intermediate.
    "flux_chunk_second_edge_weight_read_as_the_edge": (
        '    return MeepChunkWeight(s0=vectors["s0"], s1=vectors["s1"], e0=vectors["e0"],\n'
        '                           e1=vectors["e1"], dV0=dV0, include=include,',
        '    return MeepChunkWeight(s0=vectors["s0"], s1=vectors["s0"], e0=vectors["e0"],\n'
        '                           e1=vectors["e0"], dV0=dV0, include=include,',
        "pack_flux_data",
    ),
    # A flux monitor on a reduced run registers the INVARIANT axis through the same
    # ladder as any other, so it holds two half-weight planes of the identical field;
    # MEEP has no such direction at all. Production takes the measure off MEEP's own
    # chunk, so this weight model is now reached only through
    # `test_pack_flux_data_serves_meeps_own_eigenmode_reader`'s CONTROL, which packs
    # the same chunk list with it and requires 0 sites to differ from MEEP's stored E
    # on a full-plane list. `packed` sums the weight over the invariant axis
    # (`sum_k w_k * f == (sum_k w_k) * f`); taking one plane instead is exactly half
    # of MEEP's stored E, and the H side — which carries no measure — stays exact, so
    # the shape, the chunk order and half the data all still check out. Measured
    # 5.000e-01 on the packed E, standard deviation 2.7e-06.
    "flux_pack_invariant_axis_weight_not_summed": (
        "            if measure is not None:\n"
        "                measure = measure.sum(axis=axis, keepdims=True)",
        "            if measure is not None:\n"
        "                measure = measure[_slab(axis, slice(0, 1))]",
        "pack_flux_data",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # --- the migrated-monitor map's LIFETIME ----------------------------------------
    # `MigratedMonitors` is keyed by `id(meep monitor)`, and an id is an address: CPython
    # hands a freed object's address to the next allocation of the same size. The whole
    # of the fix is that the container OWNS the MEEP monitors, so no run's key can ever
    # name a block that a later run's monitor now occupies. This mutation removes exactly
    # that and nothing else — a `weakref.ref` type-checks, keeps every LIVE lookup
    # working (so the numerical suites stay green under it), and reinstates the defect
    # verbatim the moment a first run's monitors are dropped. It is the shape that
    # returned the FIRST run's transmission spectrum for the SECOND run's monitor on
    # test_binary_grating_oblique_0_0_0, twice in nine reruns, with the normalized
    # transmittance coming out exactly 1.0 and nothing raising.
    "migrated_monitor_held_weakly": (
        '    def add(self, meep_monitor, migrated):\n'
        '        """Record one ``(MEEP monitor, ours)`` pair, the OBJECT and not its id; '
        'returns ours."""\n'
        '        self._pairs.append((meep_monitor, migrated))\n'
        '        return migrated\n'
        '\n'
        '    def resolve(self, meep_monitor):\n'
        '        """Ours for this MEEP monitor, or ``None``; matched by IDENTITY, not by '
        'address."""\n'
        '        for original, migrated in self._pairs:\n'
        '            if original is meep_monitor:\n'
        '                return migrated\n'
        '        return None',
        '    def add(self, meep_monitor, migrated):\n'
        '        """Record one ``(MEEP monitor, ours)`` pair, the OBJECT and not its id; '
        'returns ours."""\n'
        '        import weakref\n'
        '        self._pairs.append((weakref.ref(meep_monitor), migrated))\n'
        '        return migrated\n'
        '\n'
        '    def resolve(self, meep_monitor):\n'
        '        """Ours for this MEEP monitor, or ``None``; matched by IDENTITY, not by '
        'address."""\n'
        '        for original, migrated in self._pairs:\n'
        '            if original() is meep_monitor:\n'
        '                return migrated\n'
        '        return None',
        "pins_the_address",
    ),
    "minus_flux_data_negation_dropped": (
        "        self.load_dft_data(data)\n"
        "        self.scale_dfts(-1.0)",
        "        self.load_dft_data(data)\n"
        "        self.scale_dfts(1.0)",
        "minus_flux_data or two_run_normalization",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # dft_flux::scale_dfts (dft.cpp:588-591) scales the E-side AND the H-side chunk
    # lists. Scaling only E is the copy-paste this reproduces, and the flux alone
    # cannot see it: the reduction is quadratic, so a whole-monitor scale of both
    # sides is invisible and a scale of one side flips the sign instead of
    # subtracting the field. Only the SUBTRACTION notices.
    "flux_scale_dfts_skips_the_H_side": (
        "        for accumulator in self._dft_H.values():\n"
        "            accumulator *= factor",
        "        for accumulator in self._dft_E.values():\n"
        "            accumulator *= factor",
        "scale_dfts or minus_flux_data",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py",),
    ),
    # A zero-extent request on a one-pixel periodic axis is MEEP's whole-direction
    # emulation (nosize_direction, fields.cpp:730-737) and takes the single stored
    # plane at weight 1. Reverting it to the ownership ladder clips the axis's only
    # site (first_owned = 1 - parity removes index 0) and the constructor raises for
    # a region MEEP accepts — the measured pre-fix failure of binary_grating_n2f.
    "near2far_unit_axis_runs_the_ownership_ladder": (
        "            if (high == low and callable(nosize) and nosize(axis)",
        "            if False and (high == low and callable(nosize) and nosize(axis)",
        "unit_axis_region or point_near2far",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # MEEP's Dcyl little_owned_corner claws an integer-r component's owned corner
    # back from doubled 2 to 0 (vec.cpp:432-436): the axis row of Ez/Ep is owned.
    # Reverting to the generic corner clips an r=0 source on those components to
    # nothing — the run completes with every component identically zero.
    "cylindrical_axis_row_disowned": (
        "            if axis_reader(d) and little_owned[d] == 2:",
        "            if False and axis_reader(d) and little_owned[d] == 2:",
        "cylindrical_axis_sources",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # --- source against the absorber: the two defect fixes and the chunk rule ------
    # Defect A's fix: the integrated dipole is WITHDRAWN from D/B before every curl
    # ladder, so no rescale (PML dsigu damping, conductivity) ever touches it.
    # Stubbing the withdraw restores the old telescoping, which is exact only while
    # nothing rescales D between injections — the integrated-in-PML parity case
    # reads 9.5837e-01 with it stubbed against 4.9e-07 fixed (measured).
    "integrated_dipole_left_standing_in_d": (
        "        if self._applied_dipole == 0:",
        "        if True:  # mutation: the pre-withdraw telescoping",
        "integrated_source_in_the_pml",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # Defect B's fix: the non-integrated deposit is mirrored into f_u wherever
    # MEEP's chunking runs the unsplit recurrence. Dropping the mirror restores the
    # broken invariant — the full-width sheet parity case reads ~3e-01 without it
    # (evidence packet §2.1 rows 16-17) against 6.3e-07 with it.
    "fu_deposit_mirror_dropped": (
        "    if source._fu_amps is None:",
        "    if True:  # mutation: deposit into f alone, as before",
        "source_spans_the_pml",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # The Dcyl arm of the same fix: the mirror runs on CYLINDRICAL grids since the
    # sweep measured the truth table transferring (x = R, y = P, z = Z; Ez's
    # dsigu = P never absorbs, so every r-layer deposit is unsplit, corners
    # included — 4.08e+00 broken vs 3.12e-06 mirrored on the sweep's mid-layer
    # case). Regrowing the pre-sweep cylindrical exclusion is exactly the
    # plausible revert this pins against: the run completes, Cartesian cases are
    # untouched, and only the cylindrical in-layer parity case sees it.
    "fu_deposit_mirror_excluded_on_cylindrical": (
        '    if pml is None or not getattr(pml, "is_active", False):\n        return None',
        '    if pml is None or not getattr(pml, "is_active", False):\n        return None\n'
        '    if getattr(grid, "cylindrical", False):\n'
        '        return None  # mutation: the pre-sweep exclusion',
        "cylindrical_source_in_pml",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # The B-side symmetry fill put back a sub-step early, which is where it lived
    # until mode-decomposition.py's standing Hy defect was traced to it. MEEP's
    # magnetic half is step_db(B_stuff) / step_source(B_stuff) /
    # step_boundaries(B_stuff) (step.cpp:67-72) — the D half verbatim at :95-103 — so
    # filling inside step_B images every owned cell BEFORE its current lands and each
    # mirror ghost stays one injection stale for the whole run. Reachable only by a
    # run that HAS a magnetic current, and on a Y fold only by By: ownership is
    # little_owned_corner0(c) = little_corner + 2 - iyee_shift(c), so of the three B
    # components only the one with Yee shift 0 on the folded axis has a ghost row.
    # That is the sheet an EigenModeSource drives, and it is the whole defect —
    # mode-decomposition.py at resolution 17 read Hy 2.72e-03 with Ez and Hx at
    # 1.3e-06 / 2.5e-06, and 1.62e-06 on Hy once the fill moved.
    # ANCHOR REPAIRED 2026-09-07. It had matched 0x since the dispatch round put
    # `fill_symmetry_bc_B` behind a `fill_B` consult and a comment block between
    # the injection and the fill, so this mutation was never applied and the
    # defect it pins was untested while its entry still read as live coverage.
    # The MEANING is unchanged -- the guarded fill moves ABOVE the injection, so
    # every mirror ghost is imaged before the current lands -- and the anchor now
    # carries the comment block because that is what makes it unique to `step`
    # rather than also matching `synchronize_magnetic_fields`.
    "b_ghosts_filled_before_the_source": (
        "        for source in magnetic:\n"
        "            source.inject(self.fields, self.time)\n"
        "        # THE FILL SUB-STEP IS TWO CONSULTS AND ONE UNCONSULTED PASS BETWEEN THEM.\n"
        "        # ``fastpath.DRIVER_SLOTS``' ``fill_B`` names the near-symmetry pass and the\n"
        "        # folded-far pass together, but the wall clear sits between them in MEEP's\n"
        "        # own order, so a single consult could only answer for a span the composer\n"
        "        # does not own. The near site is consulted with the SLOT name and the far\n"
        "        # site with the driver's own PASS name; each answer is total for its own\n"
        "        # pass, and an adapter that carries neither name answers False to both,\n"
        "        # which is the array path. ``zero_metal_B`` stays behind NO consult, which\n"
        "        # is the fact ``deposit_repair`` reads when it says the field is injected,\n"
        "        # filled and cleared by the second consult of a fused pair.\n"
        "        if fast is None or not fast.dispatch(\"fill_B\", self.fields):\n"
        "            fill_symmetry_bc_B(self.fields)\n"
        ,
        "        if fast is None or not fast.dispatch(\"fill_B\", self.fields):\n"
        "            fill_symmetry_bc_B(self.fields)\n"
        "        for source in magnetic:\n"
        "            source.inject(self.fields, self.time)\n"
        "        # THE FILL SUB-STEP IS TWO CONSULTS AND ONE UNCONSULTED PASS BETWEEN THEM.\n"
        "        # ``fastpath.DRIVER_SLOTS``' ``fill_B`` names the near-symmetry pass and the\n"
        "        # folded-far pass together, but the wall clear sits between them in MEEP's\n"
        "        # own order, so a single consult could only answer for a span the composer\n"
        "        # does not own. The near site is consulted with the SLOT name and the far\n"
        "        # site with the driver's own PASS name; each answer is total for its own\n"
        "        # pass, and an adapter that carries neither name answers False to both,\n"
        "        # which is the array path. ``zero_metal_B`` stays behind NO consult, which\n"
        "        # is the fact ``deposit_repair`` reads when it says the field is injected,\n"
        "        # filled and cleared by the second consult of a fused pair.\n"
        ,
        "eigenmode_source_waveguide_folded",
        Path("meep_gpu/driver.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # The folded-live-face lattice images: MEEP's loop_in_chunks keys its
    # lattice-shift loop on the boundary being Periodic (loop_in_chunks.cpp:393),
    # not on the symmetry, so a full-width source on a mirror-folded periodic
    # axis deposits the far end's edge weight onto the top of the stored window
    # through the ishift = -1 image. Emptying the shift list restores the
    # clip-only behaviour — the corpus family's deposit profile [1, ..., 1, 0.5]
    # against MEEP's uniform 1.0, which read 9.5e-03 … 6.0e-02 whole-volume
    # (binary_grating.py, chirped_pulse.py, diffracted_planewave.py,
    # binary_grating_phasemap.py) against floors of 8e-07 … 1.1e-06 with the
    # images deposited.
    "folded_live_lattice_images_dropped": (
        "        shift_lists[d] = list(range(low, high + 1))",
        "        shift_lists[d] = [0]  # mutation: the clip-only fold",
        "full_width_source_on_folded_live_face",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # THE ODD-COUNT TOP CELL. A folded PERIODIC axis stores MEEP's num + 1 slots
    # (vec.cpp:293-296) at BOTH parities: the top shift-0 sample sits AT
    # `big_corner` and `owns` includes it (vec.cpp:445-462). This mutation restores
    # the rule that only an EVEN count stored it, which leaves the odd count's
    # `big_corner` cell unstored and served by reflection about the second mirror
    # at doubled n_full — exact only where the MEDIUM is symmetric about that
    # plane, which at an odd count nothing makes it: MEEP anchors the absorber at
    # the window top half a cell higher (structure.cpp:226 with vec.cpp:692-700).
    # Measured whole-volume against CPU MEEP's own folded run: 1.32e-02 with a
    # 1.0-thick absorber on the folded axis and 9.74e-01 with a bare dielectric
    # edge under the window top, against ~3e-07 with the cell stored. It also
    # subsumes the retired `folded_live_overhang_clamped_back`, whose premise
    # (a shift-0 ladder rung with nowhere to land) exists only when the cell is
    # missing.
    "odd_fold_big_corner_unstored": (
        "            + (1 if (self.mirrors_by_axis[axis]\n"
        "                     and not self.metallic_axes[axis]) else 0)\n",
        "            + (1 if (self.mirrors_by_axis[axis]\n"
        "                     and not self.metallic_axes[axis]\n"
        "                     and counts_full[axis] % 2 == 0) else 0)  # mutation\n",
        "odd_fold or far_ghost_images_the_second_mirror or halved_window",
        Path("meep_gpu/grid.py"),
        ("meep_gpu/test_from_meep.py", "meep_gpu/test_stepping.py",
         "meep_gpu/test_grid_pml.py"),
    ),
    # The far ghost's IMAGE ROW. The shift-1 slot past `big_corner` images about
    # the SECOND MIRROR at doubled n_full, which is `n_full - stored + 2` — two
    # rows below the top at an even count and three at an odd one. The fixed
    # `-2` this restores reflects about the window top instead: identical at even
    # counts, a whole cell wrong at every odd one, and smooth either way.
    "folded_far_ghost_images_the_window_top": (
        '            host_writes.copy_face(array, axis, -1, array, reflect_rows[axis],',
        '            host_writes.copy_face(array, axis, -1, array, -2,  # mutation: reflect about the window top',
        "odd_fold or far_ghost_images_the_second_mirror",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_from_meep.py", "meep_gpu/test_stepping.py"),
    ),
    # The metallic-wall deposit rule: a ladder ring landing ON the wall plane is
    # DROPPED, the exact image of MEEP's store-then-zero (find_metals +
    # zero_metal run first in the same step_boundaries half-step as step_source,
    # step.cpp:100-103, :245). Regrowing the old clamp folds the wall ring's edge
    # weight onto the last interior ring — the defect that read 7.1e-02 on a
    # full-radius Dcyl Ep plane, 1.00e+00 on a point Ep in the last half-open
    # cell (exactly doubled current), and 3.2e-02 on a full-width Ey line in a
    # Cartesian PEC box, floors of 6.9e-07 … 9.6e-07 with the drop. Verified to
    # flip both detectors when applied: the lift parity case reads 1.24e-02
    # against its 5e-6 bound and the deposition test loses the wall-ring drop.
    "metal_wall_deposit_clamped_back_inward": (
        "                    elif metal_high[d] and ladder_index >= count:",
        "                    elif False and metal_high[d] and ladder_index >= count:  # mutation: the clamp regrown",
        "cylindrical_source_on_r_wall or metallic_wall_plane",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py", "meep_gpu/test_sources.py"),
    ),
    # THE SUBTLEST CHOICE in that fix: mirror-or-not follows MEEP's per-CHUNK
    # ownership (structure.cpp:509-524 effort volume, int(cells + 1.5) cells, owned
    # corner + 2 - iyee), NOT the sigma support — the chunk owns integer planes
    # whose own sigma is exactly zero, and mirroring one measured 4.59e-01 against
    # 7.3e-07 unmirrored (12x12 evidence-packet cell, deposits at y = -4.0 /
    # -3.975). Shrinking the owned span to the graded cells is precisely the
    # plausible rewrite this pins against.
    "fu_mirror_chunk_ownership_shrunk_to_sigma_support": (
        "                inside[1: min(n_cells, span + 1)] = True",
        "                inside[1: min(n_cells, span - 1)] = True  # mutation",
        "source_spans_the_pml or in_pml_chunk or unsplit_deposit_flags",
        Path("meep_gpu/pml.py"),
        ("meep_gpu/test_grid_pml.py", "meep_gpu/test_sources.py", "meep_gpu/test_from_meep.py"),
    ),
    # The |m|>1 near-axis zeroing follows upstream #3164 (all six components);
    # exempting Bp is the 1.29-era bug this engine adopted the fix for, and the
    # LIVE half of it: Bp near the axis genuinely accumulates field (the D-side
    # Dr line, by contrast, is starved to zero transitively in vacuum — a
    # Dr-exempt mutation survived both the oracle band and the declaration
    # test, measured). Killed by the six-component declaration test.
    "cylindrical_m2_zeroing_missing_bp": (
        '        for name in ("Bx", "By", "Bz",',
        '        for name in ("Bx", "Bz",  # mutation: the 1.29-era zeroing set, Bp exempt',
        "cylindrical_m2",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # BFAST was SILENTLY ACCEPTED until 2026-08-04 (supported=True, empty reason
    # list) and would have stepped MEEP's extra STEP_BFAST pass with the ordinary
    # update equations. Removing the guard restores exactly that hole.
    # --- accurate_fields_near_cylorigin: MEEP's alternative |m| >= 2 near-axis branch ---
    # The flag ignored, so the accurate branch silently steps the DEFAULT one. Measured
    # against pristine CPU MEEP 1.33.0 on the m=2 cell at Courant 1/(|m|+0.6):
    # 2.0345e-02 with the default rows against 7.5444e-07 with the r = 0 row alone, both
    # complete runs with the same signal and the same step count.
    "cylindrical_accurate_branch_ignored": (
        "    return slice(0, 1) if grid.accurate_fields_near_cylorigin "
        "else slice(0, abs(grid.m))",
        "    return slice(0, abs(grid.m))  # mutation",
        "accurate",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_cylindrical.py",),
    ),
    # A "cylindrical_accurate_branch_pre3164_component_set" mutation was written here
    # and DELETED after it was measured to SURVIVE: dropping `fields.Dx[rows] = 0`
    # alone changes nothing, because Dr near the axis is starved to zero transitively
    # in vacuum — the same fact `cylindrical_m2_zeroing_missing_dr` recorded for the
    # default branch. The COMPONENT SET of the accurate branch is established instead
    # by the oracle measurement in
    # `test_accurate_fields_near_cylorigin_matches_cpu_meep_and_is_not_the_default`
    # (the whole pre-#3164 set, D and B together, measures 6.0673e-03 against
    # 7.5444e-07), and the live half of it is pinned by
    # `cylindrical_m2_zeroing_missing_bp` above.
    # --- mp.metal: MEEP's -1e20 permittivity sentinel, carried through unchanged -----
    # THE SUBTLEST CHOICE in the PEC work, and it is a choice about WHAT NUMBER TO
    # INSTALL rather than about a code path. MEEP has no PEC update for a material
    # metal: mp.metal is an ordinary MEDIUM whose epsilon is -1e20, inverted by the
    # ordinary sym_matrix_invert branch (meepgeom.cpp:785-796) to chi1inv = -1e-20, and
    # the ordinary E = (D - sum P) * inv_eps drives E to ~0 with it. Substituting the
    # background for those cells is the obvious repair and it is measurably wrong:
    # 7.39e-01 (pec_cavity_2d), 3.60e-01 (pec_box_3d), 1.72e-01 (pec_touching_pml)
    # against parities of 3.67e-07 / 2.55e-07 / 4.39e-07, every one a COMPLETE run with
    # the full signal and the same step count.
    "pec_sentinel_installed_as_vacuum": (
        "        volumes[name] = 1.0 / values",
        "        volumes[name] = 1.0 / numpy.where(is_pec_chi1inv(values), 1.0, values)"
        "  # mutation",
        "pec_",
    ),
    # THE OTHER WAY TO GET IT WRONG, and the one a reviewer would wave through: relax
    # the permittivity validators for "a negative epsilon" instead of for MEEP's
    # sentinel by name. An epsilon of -3 is not a conductor, it has no stable leapfrog
    # at any Courant factor, and admitting it turns a named refusal into a silently
    # divergent run. Anchored in the driver, which is where the predicate lives so that
    # the gate and the setter cannot drift apart.
    "pec_relaxed_for_any_negative_epsilon": (
        "    return values < _PEC_EPSILON_CEILING",
        "    return values < 0.0  # mutation: 'negative is fine' instead of the sentinel",
        "negative_epsilon or pec_",
        Path("meep_gpu/driver.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # --- mp.GaussianBeam2DSource: Love's equivalent current PAIR ---------------------
    # get_equiv_sources returns BOTH the electric current K = nHat x H and the magnetic
    # current N = -nHat x E (source.py:786-794), and it is the pair that makes the sheet
    # radiate into one half-space. Keeping only the electric half is the plausible
    # converter bug — one source drives one component — and it launches the beam in BOTH
    # directions at half amplitude: measured 6.89e-01 against a parity of 2.81e-07, on a
    # run with the full signal and the same step count.
    "gaussian_beam_2d_magnetic_current_dropped": (
        "    return [_lift_source(mp, plain, dimensions) for plain in equivalent]",
        "    return [_lift_source(mp, plain, dimensions) for plain in equivalent][:1]"
        "  # mutation",
        "gaussian_beam_2d",
    ),
    # The sheet NORMAL, read off the wrong axis. MEEP's branch is `if size.x(): nHat =
    # (0,1)*sign(kdir.y) elif size.y(): nHat = (1,0)*sign(kdir.x)` (source.py:1059-1063)
    # — the normal is PERPENDICULAR to the extent, and reading it as parallel is the
    # intuitive misreading. Both equivalent currents are still deposited, both still
    # carry MEEP's own amp_data, and the run is smooth and complete.
    "gaussian_beam_2d_normal_taken_along_the_extent": (
        "    if size.x():  # :1059-1063 — the branch order is MEEP's, and it is not symmetric.",
        "    if size.y():  # mutation",
        "gaussian_beam_2d",
    ),
    # --- BFAST: MEEP's broadband fixed-angle source technique -----------------------
    # The whole feature is one extra additive pass inside step_db (step_db.cpp:129-142
    # -> step_generic.cpp:335-471), folded into this engine's curl on the operands the
    # stencil already gathered. Every mutation below leaves a stable, smooth, complete
    # run that is simply at the WRONG INCIDENCE ANGLE — which is the whole hazard: a
    # BFAST run degraded in any of these ways looks exactly like a working one.
    #
    # The Cartesian gate is gone (BFAST steps now); what this anchor still guards is
    # the CYLINDRICAL refusal, whose reason is upstream — MEEP's Dcyl branch makes
    # step_generic.cpp:376 reachable, and that branch omits the `- F[i]` its seven
    # siblings carry.
    "bfast_cylindrical_guard_removed": (
        "    bfast = tuple(getattr(sim, \"bfast_scaled_k\", None) or (0.0, 0.0, 0.0))",
        "    bfast = (0.0, 0.0, 0.0)  # mutation: the cylindrical branch un-refused",
        "refused_by_name",
        Path("meep_gpu/from_meep.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # THE PRE-2026-08-04 BLIND SPOT, restored where it now lives: the gate accepts the
    # run and the lift silently forgets the k, so the ordinary update equations step a
    # question they were not asked. Measured: 1.53e+00 from CPU MEEP on the
    # bfast_fixed_angle case, against its 2.33e-06 parity — and MEEP's OWN
    # normal-incidence run of the same cell sits 1.32e+00 away, i.e. the degraded
    # engine has converged on a different, entirely physical answer.
    "bfast_dropped_on_lift": (
        "        bfast_scaled_k=tuple(\n"
        "            float(value)\n"
        "            for value in (getattr(sim, \"bfast_scaled_k\", None) or (0.0, 0.0, 0.0))\n"
        "        ),",
        "        bfast_scaled_k=(0.0, 0.0, 0.0),  # mutation",
        "bfast",
        Path("meep_gpu/from_meep.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # THE SUBTLEST CHOICE IN THE PASS. step_db.cpp:129-136 indexes each k by the OTHER
    # partner's own direction — `k1 = bfast_scaled_k[component_index(c_m)]` multiplies
    # g1 = f_p — and MEEP annotated both lines in place ("puts k1 in direction of g2")
    # precisely because the pairing reads backwards. Interchanging them transposes the
    # cross product: (k x E) becomes -(E x k) evaluated on the wrong operands, which is
    # a different, smooth, finite shear at a plausible-looking angle. It cannot be
    # caught by any magnitude check and it cannot be caught on a k with one nonzero
    # component if the two indices happened to coincide — they never do, which is what
    # test_bfast_k_is_indexed_by_the_components_own_direction_not_the_derivative pins.
    "bfast_k_pair_interchanged": (
        "    k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0\n"
        "    k2 = bfast[_bfast_axis(term.first)] if have_p else 0.0",
        "    k1 = bfast[_bfast_axis(term.first)] if have_m else 0.0  # mutation\n"
        "    k2 = bfast[_bfast_axis(term.second)] if have_p else 0.0",
        "bfast",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_stepping.py", "meep_gpu/test_from_meep.py"),
    ),
    # step_db.cpp:137-140. The sign that makes dB/dt = -curl(E) + d/dt(k x E) and
    # dD/dt = +curl(H) - d/dt(k x H) rather than the same sign on both: the shear then
    # runs forwards for the magnetic half and backwards for the electric one.
    "bfast_d_side_sign_not_flipped": (
        "    if not magnetic:  # MEEP's `if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`.\n"
        "        k1, k2 = -k1, -k2",
        "    if False:  # mutation: the D-side flip dropped\n"
        "        k1, k2 = -k1, -k2",
        "bfast",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_stepping.py", "meep_gpu/test_from_meep.py"),
    ),
    # The SUM, not the curl's difference. S is twice the two-point average of the cross
    # product interpolated onto the target's Yee position; taking the difference of the
    # same two samples instead is the copy-paste from the line above it in the same
    # function, and it turns a field value into a first derivative — small, smooth, and
    # scaling with the resolution rather than being obviously wrong.
    "bfast_sum_taken_as_the_curls_difference": (
        "    total = (dtype.type(k1) * (operands.shifted_first + operands.first)\n"
        "             - dtype.type(k2) * (operands.shifted_second + operands.second))",
        "    total = (dtype.type(k1) * (operands.shifted_first - operands.first)  # mutation\n"
        "             - dtype.type(k2) * (operands.shifted_second - operands.second))",
        "bfast",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_stepping.py", "meep_gpu/test_from_meep.py"),
    ),
    # F_n = S_n - F_{n-1} with output F_n - F_{n-1} is the TUSTIN (bilinear) derivative.
    # Dropping the previous term from the state update leaves a backward difference at
    # twice the amplitude — and this is not a hypothetical slip: it is the exact shape
    # of MEEP's own inconsistency at step_generic.cpp:376, where the single-operand
    # branch omits the `- F[i]` its seven siblings carry. Catching it proves this
    # transcription is the consistent one.
    "bfast_state_update_drops_the_previous_value": (
        "    advance = total - dtype.type(2.0) * state",
        "    advance = total - state  # mutation: step_generic.cpp:376's shape",
        "bfast",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_stepping.py", "meep_gpu/test_from_meep.py"),
    ),
    # energy_and_flux.cpp:113/:130 BACKUP and RESTORE f_bfast around the magnetic
    # synchronization half-step. Omitting it is the one BFAST defect that produces a
    # SLOW drift instead of an immediate change: the IIR is marginally stable — its
    # homogeneous mode is (-1)^n, undamped forever — so a single flux call leaves a
    # permanent alternating error that no absorption ever removes. Measured 7.86e-01
    # after one flux call and 40 further steps, against 0.0e+00 with it restored.
    "bfast_state_not_restored_around_the_flux_half_step": (
        "                       \"f_bfast_Bx\", \"f_bfast_By\", \"f_bfast_Bz\",\n",
        "                       # mutation: the IIR state left one step ahead\n",
        "bfast",
        Path("meep_gpu/driver.py"),
        ("meep_gpu/test_field_reducers.py",),
    ),
    # MEEP's have_p / have_m (fields.cpp:428-455) are false when the derivative direction
    # is not one the grid resolves, and step_db.cpp:131-134 then forces THAT k to zero.
    # Dropping the guard is the reading that assumes BFAST inherits the curl's own
    # treatment of an invariant axis — and it does not: an invariant axis makes the
    # curl's DIFFERENCE an exact zero all by itself, so nothing there needs a guard,
    # while the BFAST SUM of the same two samples is 2*g. The unguarded version therefore
    # adds a term MEEP does not have to EVERY reduced-dimension BFAST run, silently and
    # at full strength, while leaving every 3-D run bit-identical — which is why no
    # 3-D case can catch it and why the pin has to be a 2-D one.
    "bfast_invariant_axis_guard_removed": (
        "    k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0\n"
        "    k2 = bfast[_bfast_axis(term.first)] if have_p else 0.0",
        "    k1 = bfast[_bfast_axis(term.second)]  # mutation: the have_m guard dropped\n"
        "    k2 = bfast[_bfast_axis(term.first)]   # mutation: the have_p guard dropped",
        "bfast",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_stepping.py",),
    ),
    # MEEP allocates f_bfast on `use_bfast` ALONE (step_db.cpp:76-79) — the test sits
    # beside the f_cond and f_u allocations, not inside them. Gating ours on PML is the
    # natural "it is an auxiliary, auxiliaries are a PML thing" slip, and it drops the
    # entire term from every boundary-free BFAST run without a word.
    "bfast_storage_gated_on_pml": (
        "        if not self.grid.bfast_active:\n            return",
        "        if not (self.grid.bfast_active and self._pml_active):  # mutation\n"
        "            return",
        "bfast",
        Path("meep_gpu/fields.py"),
        ("meep_gpu/test_stepping.py",),
    ),
    # --- negative-frequency sources: the plus/minus-omega superposition ------------
    # MEEP stores a carrier frequency RAW (gaussian_src_time, sources.cpp:72-96) and
    # rotates exp(-i*2*pi*f*t) with its sign; abs()-ing it on lift is the one-line
    # "normalize the input" fix that keeps every envelope, every deposit and every
    # step count identical and turns the pm-pair's real sine current into one
    # complex rotation — smooth, complete, 5.3e-01 from MEEP (measured, against the
    # case's 1.9e-07 floor).
    "negative_source_frequency_absed_on_lift": (
        '        data["source_type"] = "gaussian"\n'
        '        data["frequency"] = float(src.frequency)',
        '        data["source_type"] = "gaussian"\n'
        '        data["frequency"] = abs(float(src.frequency))',
        "gaussian_pm_pair",
    ),
    # The same normalization one layer down, in the waveform itself: the carrier
    # sign dropped where omega is resolved. dipole(-f) must be conj(dipole(+f))
    # bit-exactly (envelope even in f, carrier and amp factor conjugate); with
    # |f| in omega the minus pulse rotates the same way as the plus one and only
    # the conjugate-identity test can tell.
    "negative_carrier_sign_dropped_in_the_pulse": (
        "        self._omega = 2 * np.pi * self.frequency\n"
        "        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104\n"
        "\n"
        "        # MEEP sources.cpp:94-95 — width = 1/fwidth, peak = midpoint of "
        "[start, start + 2*cutoff*width]",
        "        self._omega = 2 * np.pi * abs(self.frequency)\n"
        "        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104\n"
        "\n"
        "        # MEEP sources.cpp:94-95 — width = 1/fwidth, peak = midpoint of "
        "[start, start + 2*cutoff*width]",
        "conjugate_carrier",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_sources.py",),
    ),
    # --- the wall planes MEEP declines to answer (ring_gds's class) ----------------
    # The structured lift excludes from the declared-media spread check exactly the
    # boundary planes MEEP measurably declines (grid_volume::owns needs o > 0,
    # vec.cpp:445-463; the unowned diagonal defaults to vacuum,
    # monitor.cpp:180-183). Dropping the exclusion refuses every structured cell
    # whose background is not vacuum — the pre-fix ring_gds failure verbatim.
    "wall_plane_exclusion_dropped": (
        "        keep = ~declined\n"
        "        sampled_volume = volumes[name][keep] if keep.any() else volumes[name]",
        "        keep = ~declined\n"
        "        sampled_volume = volumes[name]",
        "structured_nonvacuum_background",
    ),
    # THE SECOND CONSUMER of the same exclusion, and the subtlest choice in it: the
    # bulk cross-check classifies and compares `stacked[0]` — Ex — so the mask it is
    # given must be the UNION over the three components, not Ex's own. On the
    # oblique-waveguide class (a rotated block crossing the low-x wall) the one cell
    # that fails is OWNED for Ex and declined only for Ey and Ez: Ex's window opens at
    # corner+1, Ey's and Ez's open at the corner. Both mutations below are the
    # plausible readings of "mask the component being compared", and both put that cell
    # back into the comparison, so the lift refuses a structure it read correctly.
    #   `&` instead of `|`  - one character, and the intersection is just the corner cell.
    #   `declined_any` kept - the first component's mask (Ex), the "mask what you compare"
    #                         reading stated outright.
    "bulk_wall_exclusion_intersected_not_unioned": (
        "        declined_any = declined if declined_any is None else (declined_any | declined)",
        "        declined_any = declined if declined_any is None else (declined_any & declined)",
        "declined_wall_planes or wall_exclusion",
    ),
    "bulk_wall_exclusion_takes_only_the_compared_component": (
        "        declined_any = declined if declined_any is None else (declined_any | declined)",
        "        declined_any = declined if declined_any is None else declined_any",
        "declined_wall_planes or wall_exclusion",
    ),
    # The exclusion never reaching the bulk check at all — the pre-fix state.
    "bulk_wall_exclusion_not_passed": (
        "    _require_bulk_matches_meep(sim, driver, volumes, declined=declined_any)",
        "    _require_bulk_matches_meep(sim, driver, volumes, declined=None)",
        "declined_wall_planes or wall_exclusion",
    ),
    # The check the exclusion feeds, deleted outright. The spread test is the only
    # place the lift ever compares a sampled permittivity against the declaration,
    # and with the wall planes now excluded a mutation here survives every
    # honest-geometry case — the forced gate (wall exclusion disabled, ring_gds's
    # class) is what has to notice.
    # Re-anchored when the material-grid work put `not smoothed_grids and` in front of
    # the spread test: the old find-text no longer existed, so the mutation was silently
    # a no-op and the battery would have reported it green without ever applying it.
    "declared_span_check_removed": (
        "        if bracketed and (spread[0] < low - 1e-4 * high "
        "or spread[1] > high + 1e-4 * high):",
        "        if False:",
        "keeps_its_teeth",
    ),
    # The same check disarmed the OTHER way — by widening the material-grid exemption to
    # every cell instead of only the ones where the declared bracket genuinely is not a
    # bound (a smoothed MaterialGrid, and a per-point epsilon route). The exemption is
    # correct exactly where `fallback_chi1inv_row` extrapolates u past [0, 1]
    # (meepgeom.cpp:1208-1218) and nowhere else, and a cell with no grid at all must
    # still face the bracket. Anchored on the same teeth test, whose cell (ring_gds's
    # class: SiO2 background, Si block, metallic walls) contains no MaterialGrid.
    "declared_span_check_exempted_for_every_cell": (
        "    bracketed = not smoothed_grids and not routes and bool(declared)",
        "    bracketed = bool(smoothed_grids)  # mutation",
        "keeps_its_teeth",
    ),
    # --- materials: a MaterialGrid, and the three things its endpoints cannot say -----
    # THE SUBTLEST OF THE THREE, and the one that is wrong in the direction nobody looks:
    # MEEP DISCARDS mu / chi2 / chi3 / B_conductivity declared on a grid endpoint
    # (md->medium is default-constructed, material_data.cpp:47-50, and only
    # E_susceptibilities are merged into it, typemap_utils.cpp:528-547), while
    # `_lift_material` would install chi3 from the representative medium and step a
    # nonlinear cell MEEP is stepping linear. Measured: MEEP's grid run with chi3=20 on
    # both endpoints is BIT-IDENTICAL (0.0000e+00) to the same grid with no chi3, while
    # the same claim on a plain medium moves 1.4093e+00. Without this refusal the run
    # completes, the field is smooth and complete, and nothing says which material ran.
    "material_grid_endpoint_discard_ignored": (
        "                if value is not None and _vector3(value) != unset:",
        "                if False:",
        "material_grid",
    ),
    # --- the per-point D_conductivity read ---------------------------------------
    #
    # THE SUBTLEST CHOICE IN THE PACKAGE'S MATERIAL PATH, and the one this battery is
    # here for: sigma_D is recovered from the **D row** of chi1inv at a **NONZERO**
    # frequency. Both halves of that sentence have an obvious, plausible, wrong
    # alternative, and each alternative produces a complete, smooth, absorbing run.
    #
    # The E row is where every other material read in this file goes, so reaching for
    # it here is the natural mistake. It is not merely a different number: MEEP's E-row
    # chi1inv at a nonzero frequency inverts to (1 + i*sigma/w)*eps, so `w*Im(1/chi1inv)`
    # comes back as eps*sigma — 8.4 on the structured_conductivity cell's 0.7, right in
    # vacuum and wrong by a factor of 12 in the block, which is exactly the shape of a
    # bug that survives a sanity check. It is caught by the read's own identity
    # premise-check, which is the reason that check exists.
    "conductivity_read_off_the_E_row": (
        'd_components = {"Ex": mp.Dx, "Ey": mp.Dy, "Ez": mp.Dz}',
        'd_components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}',
        "structured_conductivity or material_grid_damping or conductivity_read",
    ),
    # Frequency 0 is where every OTHER chi1inv read in this file is taken, and it is
    # where monitor.cpp returns before it ever reaches the conductivity branch
    # (monitor.cpp:268). So the volume comes back identically zero and the loss simply
    # vanishes — the pre-read behaviour, measured at 1.2185e-02 on the block cell and
    # 1.6377e-02 on the damped grid against parities of 3.3888e-07 and 3.1668e-07, every
    # one of them a complete run with the full signal and the same step count. Caught by
    # MEASUREMENT alone: neither self-check fires, because zero is a perfectly
    # self-consistent, frequency-independent answer.
    "conductivity_read_at_frequency_zero": (
        "_CONDUCTIVITY_PROBE_FREQUENCIES = (0.9, 0.3)",
        "_CONDUCTIVITY_PROBE_FREQUENCIES = (0.0, 0.0)",
        "structured_conductivity or material_grid_damping",
    ),
    # MEEP's conductivity[c][d] is the material value with each absorber face's ramp
    # ALREADY ADDED INTO IT (geom_epsilon::conductivity, meepgeom.cpp:1596-1625). The
    # driver's own composition would add it a second time — the naive "install what you
    # read" — for a stable, smooth, MORE absorbing run. Measured 6.0919e-02 against
    # 2.2353e-07.
    "conductivity_read_absorber_ramp_double_counted": (
        "                absorber_included=True,",
        "                absorber_included=False,",
        "structured_conductivity_in_an_absorber",
    ),
    # The three D components' volumes come from three different Yee points, so a graded
    # conductivity differs between them by half a cell. Installing one row under all
    # three keys is the pre-anisotropic engine restored, and on a 2-D TM cell it is the
    # Ez row that is stepped — so this substitutes the Ex row's registration for it and
    # nothing else changes. Anchored on the damped MaterialGrid, whose loss is graded by
    # construction; a uniform sigma would not notice.
    "conductivity_read_shared_across_the_D_components": (
        '                {"D" + name[1:]: read[name] for name in E_COMPONENTS},',
        '                {"D" + name[1:]: read["Ex"] for name in E_COMPONENTS},',
        "material_grid_damping",
    ),
    # A MaterialGrid's damping is a conductivity that NO medium in the cell declares, so
    # the signature comparison above it finds nothing and the read is never taken. This
    # is the old `material_grid_damping_accepted` defect in its post-fix form: the run
    # completes and is wrong by 1.0531e-02 at damping=0.3, 1.7316e-02 at 0.5 and
    # 9.1963e-02 at pi, against 2.9431e-07 at damping=0.
    "material_grid_damping_not_read_per_point": (
        '    return any(float(getattr(grid, "damping", 0.0) or 0.0) != 0.0',
        '    return False and any(float(getattr(grid, "damping", 0.0) or 0.0) != 0.0',
        "material_grid_damping",
    ),
    # The three signature entries a structured cell may differ in, cut back to the
    # permittivity alone — the comparison before epsilon_offdiag and D_conductivity_diag
    # became per-point recoverable. Every structured-loss and tensor-media cell is then
    # refused outright.
    "structured_media_may_differ_in_permittivity_only": (
        "PER_POINT_SIGNATURE_INDICES = (0, 1, 8)",
        "PER_POINT_SIGNATURE_INDICES = (0,)",
        "structured_conductivity or material_grid_damping",
    ),
    # The DANGEROUS direction of the callable-dispersion gate. With no registered set,
    # a callable whose pole a declared medium ALSO carries reads as one MEEP drops —
    # and MEEP does not drop it, it steps it per point, which this lift cannot do.
    "callable_dispersion_registration_ignored": (
        "    registered = _registered_susceptibility_keys(mp, sim)",
        "    registered = frozenset()",
        "refused_by_name",
    ),
    # Only ONE endpoint taken. The pair is what BRACKETS the interpolation
    # (meepgeom.cpp:569-583); one of them is a homogeneous material the cell does not
    # have, and the declared-media span the structured lift checks against collapses to a
    # point. Anchored on the material_grid parity cases, whose endpoint_medium_only
    # control — the grid collapsed to medium2 — sits at 0.82 against a parity of 2.9e-07.
    "material_grid_expanded_to_one_endpoint": (
        "            materials.extend((material.medium1, material.medium2))",
        "            materials.append(material.medium1)",
        "material_grid",
    ),
    "offdiagonal_wall_mask_dropped": (
        "        _mask_metallic_wall_coupling(total, fields.grid, IYEE_SHIFTS[component])",
        "        pass  # mutation: the wall plane keeps the coupling it never has in MEEP",
        "uniform_tensor_epsilon_matches_cpu_meep and metallic",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_tensor_epsilon.py",),
    ),
    # Masking the FOLD plane the way the metallic rule does destroys the even-parity
    # coupling that is genuinely nonzero there — measured 2.0e-02 against the exact
    # fold. The two boundary kinds went through one mask once, and this is the
    # regression that would quietly bring that back.
    "offdiagonal_fold_plane_masked": (
        "        if grid.is_metallic(axis) and not grid.is_mirrored(axis):",
        "        if grid.is_metallic(axis) or grid.is_mirrored(axis):",
        "fold_equivalence",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_tensor_epsilon.py",),
    ),
    "chi1inv_not_inverted": (
        "        volumes[name] = 1.0 / values",
        "        volumes[name] = values.copy()",
        "structured",
    ),
    # --- materials: what chi1inv does not carry ----------------------------------
    "media_may_differ_in_dispersion": (
        "        if len(without_per_point) > 1:",
        "        if False:",
        "refused_by_name or structured",
    ),
    "structured_epsilon_taken_from_the_diagnostic_array": (
        "            _lift_epsilon_structured(mp, sim, driver, materials, progress_cb, routes=routes)",
        "            _lift_epsilon(sim, driver, medium)",
        "structured",
    ),
    # --- dimensionality: the corpus false positive -------------------------------
    # `flat_axis_only_checked_on_z` used to live here. It is GONE rather than repaired,
    # because the reduced-dimension work inverted it: checking only cell_size.z is what
    # MEEP's use_2d actually does, so the old mutation now describes the correct code.
    # Its live replacement is `zero_extent_on_any_axis_reduces`, which mutates in the
    # other direction. Likewise the old `min(declared, from_cell)` rule this next
    # mutation targeted is gone — it got dimensions=1 wrong — so it is re-anchored onto
    # the rule that replaced it.
    "declared_dimensions_ignored": (
        "    if declared != 3:\n        return declared",
        "    if False:\n        return declared",
        "reduced_dimensions or refused_by_name",
    ),
    # --- post-init mutation: the invisible PEC wall ------------------------------
    "initialized_simulation_accepted": (
        '        reasons.append(\n            "this simulation has already been initialized',
        '        _unused = (\n            "this simulation has already been initialized',
        "refused_by_name",
    ),
    # --- boundaries --------------------------------------------------------------
    # `bloch_on_a_pml_axis_accepted` used to live here, mutating away a refusal that
    # no longer exists: a Bloch phase on an absorbing axis is STEPPED now, because it
    # reproduces CPU MEEP at 2.82e-07 (MEEP's use_bloch keeps the axis periodic and
    # grades the PML underneath). Deleted rather than re-anchored — there is no
    # refusal left to delete — and the behaviour it guarded is covered by the
    # `bloch_on_pml_axis` parity case, whose no-x-absorber control sits at 1.4e-01.
    # --- mp.Absorber: a graded D AND B conductivity, not a matched layer ----------
    # The four ways to build one that still absorbs, still runs, and still looks like a
    # field. Their measured separations from CPU MEEP, against a 1.9e-07 .. 4.0e-07
    # floor: substituting a PML 2.97e-01, dropping the magnetic half 1.62e-01, sharing
    # one sigma volume across the six components 6.33e-02, and the seam rule below.
    "absorber_seam_takes_its_own_sigma": (
        # THE SUBTLEST CHOICE IN THE FAMILY. MEEP's little_owned_corner0(c) =
        # little_corner + 2 - iyee_shift(c) (vec.hpp:1102-1104) means an INTEGER
        # component's index 0 is never stepped on a wrapping axis — step_boundaries
        # fills it from the owned copy at index N, one lattice vector up — so the seam's
        # sigma is the FAR face's. Taking index 0's own sigma is what any reading of
        # "sample each cell at its own coordinate" produces, it is invisible for every
        # symmetric layer, and on a one-sided absorber it drifts 2.9e-06 at t = 2,
        # 1.7e-04 at t = 4 and 1.5e-01 at t = 8 against CPU MEEP.
        "    if shift != 0 or not grid.axis_wraps(axis):\n"
        "        return numpy.arange(stored, dtype=numpy.int64)",
        "    if True:\n"
        "        return numpy.arange(stored, dtype=numpy.int64)",
        "absorber_z_low_only",
        "meep_gpu/absorber.py",
        ("meep_gpu/test_absorber.py", "meep_gpu/test_from_meep.py"),
    ),
    "absorber_magnetic_half_dropped": (
        # Re-anchored when the two sides' composition was unified: the B install is no
        # longer a literal absorber-only dict, so the mutation now withholds the RAMP
        # from the B composition and leaves a material's own B_conductivity intact.
        # That is the sharper edit anyway — it drops exactly the absorber's magnetic
        # half and nothing else.
        "        self.fields.set_b_conductivity(self._compose_conductivity_side(\n"
        "            B_COMPONENTS, self._material_b_conductivity, absorber))",
        "        self.fields.set_b_conductivity(self._compose_conductivity_side(\n"
        "            B_COMPONENTS, self._material_b_conductivity, None))",
        "absorber",
        "meep_gpu/driver.py",
        ("meep_gpu/test_stepping.py", "meep_gpu/test_from_meep.py"),
    ),
    # --- B_conductivity: a MATERIAL's magnetic loss, not a boundary's ---------------
    # Two mutations, because the two things that can go wrong here are independent and
    # each one produces a complete, smooth, plausible field.
    "b_conductivity_composed_as_a_precedence": (
        # THE SUBTLEST CHOICE IN PHASE 0, and the one no earlier test could see. MEEP
        # ADDS the absorber face's profile onto the material's own get_cnd value —
        # "isotropically, for both magnetic and electric conductivity", MEEP's own
        # comment at meepgeom.cpp:1596-1600. "The medium's own declaration wins where it
        # has one" is the other natural reading of a composition, it is EXACTLY
        # EQUIVALENT on every absorber cell that existed before this one (their media
        # are lossless, so `material` is empty and the branch never fires), and it is
        # equally invisible for a material loss outside the layers. It separates only
        # where a lossy medium overlaps an absorbing face: measured 6.60e-02 against a
        # parity of 6.55e-07 on b_conductivity_in_an_absorber.
        "        if not absorber:\n"
        "            return material",
        "        if not absorber or material:\n"
        "            return material",
        "b_conductivity_in_an_absorber",
        "meep_gpu/driver.py",
        ("meep_gpu/test_from_meep.py",),
    ),
    "b_conductivity_lifted_onto_the_d_side": (
        # "A conductivity is a conductivity." MEEP's get_cnd really is one switch over
        # both sides (meepgeom.cpp:1545-1559) and step_db really does hand both into the
        # same STEP_CURL (step_db.cpp:125-127), which is what makes reading
        # B_conductivity_diag onto the D components look like a spelling difference. It
        # is a different material: measured 9.44e-02 away on b_conductivity.
        '        driver.set_b_conductivity(\n'
        '            b_conductivity[0] if len(set(b_conductivity)) == 1\n'
        '            else {"Bx": b_conductivity[0], "By": b_conductivity[1], '
        '"Bz": b_conductivity[2]})',
        '        driver.set_conductivity(\n'
        '            b_conductivity[0] if len(set(b_conductivity)) == 1\n'
        '            else {"Dx": b_conductivity[0], "Dy": b_conductivity[1], '
        '"Dz": b_conductivity[2]})',
        "b_conductivity",
        "meep_gpu/from_meep.py",
        ("meep_gpu/test_from_meep.py",),
    ),
    "b_conductivity_shared_from_the_x_component": (
        # The anisotropic collapse, which the D side already carries a mutation for.
        # B_conductivity_diag=(0, 1, 0.4) has a ZERO first entry, so this one puts every
        # B component on the lossless path while MEEP steps two of them lossy — a run
        # that completes with the full signal, 1.51e-01 away.
        '            b_conductivity[0] if len(set(b_conductivity)) == 1\n'
        '            else {"Bx": b_conductivity[0], "By": b_conductivity[1], '
        '"Bz": b_conductivity[2]})',
        '            b_conductivity[0])',
        "anisotropic_b_conductivity",
        "meep_gpu/from_meep.py",
        ("meep_gpu/test_from_meep.py",),
    ),
    "absorber_profile_sampled_per_pixel_not_per_half_pixel": (
        "    half_pixel = 0.5 / resolution  # gv.inva * 0.5",
        "    half_pixel = 1.0 / resolution  # gv.inva * 0.5",
        "absorber",
        "meep_gpu/absorber.py",
        ("meep_gpu/test_absorber.py", "meep_gpu/test_from_meep.py"),
    ),
    "absorber_prefac_uses_the_snapped_extent": (
        # prefac and the lookup both divide by the RAW thickness; substituting the
        # quantized N*dx is the same "surely they are the same number" slip the PML
        # path already carries a mutation for, and it is exact at every whole and half
        # cell — only a fractional layer separates them.
        "    prefac = (-math.log(R_asymptotic)) / (4.0 * thickness * profile_integral)",
        "    prefac = (-math.log(R_asymptotic)) / (4.0 * count * half_pixel * profile_integral)",
        "absorber",
        "meep_gpu/absorber.py",
        ("meep_gpu/test_absorber.py", "meep_gpu/test_from_meep.py"),
    ),
    "pml_side_enum_swapped": (
        "                faces[direction][0 if one_side == MEEP_SIDE_LOW else 1] = cells",
        "                faces[direction][1 if one_side == MEEP_SIDE_LOW else 0] = cells",
        "pml_z_low_only",
    ),
    "pml_thickness_used_as_cells": (
        "        cells = float(layer.thickness) * float(sim.resolution)",
        "        cells = float(layer.thickness)",
        "pml_uniform or structured_pml",
    ),
    # The converter used to round this product to a whole cell, behind a refusal that
    # kept the rounding from ever being reached. Both are gone; what replaces them is
    # the mutation that puts the rounding back, which is the one-line "fix" anyone
    # touching this line would reach for. It is caught by the fractional-PML parity
    # cases, whose whole-cell controls sit 3.26e-02 / 2.89e-02 and 8.75e-04 / 9.27e-04
    # from the references they must not match.
    #
    # `pml_fractional_thickness_rounded` used to live below here, mutating that refusal
    # away. Deleted rather than re-anchored, as `bloch_on_a_pml_axis_accepted` was —
    # there is no refusal left to delete.
    "pml_thickness_rounded_to_whole_cells": (
        "        cells = float(layer.thickness) * float(sim.resolution)",
        "        cells = float(round(float(layer.thickness) * float(sim.resolution)))",
        "pml_fractional",
    ),
    # MEEP snaps the layer's extent to a whole HALF cell with a C cast of v + 0.5;
    # numpy/Python round() is half-to-even and floor() is a half cell short over half
    # the range. Mutating the primitive is silent — every whole and half cell agrees —
    # so it is aimed at the grading pins, which draw thicknesses on both sides of a
    # half cell, and at the fractional parity cases.
    "pml_extent_floored_instead_of_rounded": (
        "    return int(2.0 * float(cells) + 0.5)",
        "    return int(2.0 * float(cells))",
        "",
        "meep_gpu/pml.py",
        ("meep_gpu/test_grid_pml.py",),
    ),
    # MEEP snaps the EXTENT and nothing else: prefac divides by the raw requested dx
    # (structure.cpp:635). Substituting the snapped extent is the tidying-up a reader
    # reaches for once they have seen half_cell_extent, and it is silent at every whole
    # cell AND at every exact half cell, because 0.5*N == c there — which was every
    # thickness the suite pinned and both fractional parity cases. It is worth +5.0000%
    # of peak sigma at 2.1 cells and +4.0000% at 5.2. Aimed at the grading oracle, which
    # now transcribes prefac instead of calling it, and at the non-half-cell parity
    # case, which moves from 7.34e-07 to 6.80e-03 under it.
    "pml_prefac_divided_by_the_snapped_extent": (
        "        dx_pml = n_pml * self.grid.dx",
        "        dx_pml = 0.5 * half_cell_extent(n_pml) * self.grid.dx",
        "grading or pml_fractional",
        "meep_gpu/pml.py",
        ("meep_gpu/test_grid_pml.py", "meep_gpu/test_from_meep.py"),
    ),
    # --- the clock, the lattice and the storage mode -----------------------------
    "k_point_dropped": (
        "        k_point=_lift_k_point(sim, dimensions),",
        "        k_point=(0.0, 0.0, 0.0),",
        "bloch",
    ),
    "k_point_scaled_by_two_pi": (
        "        k_point=_lift_k_point(sim, dimensions),",
        "        k_point=tuple(2.0 * math.pi * value "
        "for value in _lift_k_point(sim, dimensions)),",
        "bloch",
    ),
    "courant_not_forwarded": (
        "        courant=float(sim.Courant),",
        "        courant=0.5,",
        "courant_non_default",
    ),
    "storage_mode_from_the_constructor_argument": (
        "        force_complex_fields=not sim.fields.is_real,",
        "        force_complex_fields=bool(sim.force_complex_fields),",
        "bloch or real_fields",
    ),
    "grid_shape_pin_removed": (
        "    if epsilon.shape != expected:",
        "    if False:",
        "grid_meep_built_differently or folded_axis",
    ),
    # --- sources ------------------------------------------------------------------
    "gaussian_fwidth_not_inverted": (
        '        data["fwidth"] = 1.0 / float(src.width)',
        '        data["fwidth"] = float(src.width)',
        "gaussian_source",
    ),
    "amp_func_given_absolute_coordinates": (
        "            return meep_amp_func(mp.Vector3(x, y, z))",
        "            return meep_amp_func(mp.Vector3(x, y, z) + mp.Vector3(*data[\"center\"]))",
        "amp_func_sheet",
    ),
    "complex_amplitude_truncated": (
        '        "amplitude": complex(source.amplitude),',
        '        "amplitude": complex(source.amplitude.real),',
        "complex_amplitude",
    ),
    "is_integrated_dropped": (
        '        "is_integrated": bool(src.is_integrated),',
        '        "is_integrated": False,',
        "integrated_source",
    ),
    "continuous_ramp_dropped": (
        '        data["width"] = float(src.width)',
        '        data["width"] = 0.0',
        "continuous_ramp",
    ),
    # --- the eigenmode source: the one-way launch is a sign-and-phase relation ----
    # A flipped magnetic sheet reverses/breaks the forward-backward cancellation while
    # keeping the total current — the very ghost class the readback dead end produced
    # (directionality matched at 6.17:1 vs 6.15:1 with the FIELD uncorrelated at
    # 0.9996), so the kill must come from the full-volume parity, not a ratio.
    #
    # ONE MUTATION PER POLARIZATION, because each drives a different pair of the four
    # sheets: the Hz-sheet flip SURVIVED the TM-only battery (the TM mode's Ey is ~0,
    # so that sheet carried nothing to flip) — the hole that forced the TE case.
    "eigenmode_tm_magnetic_sheet_sign_flipped": (
        "        (cH[np1], cE[np2], +1.0),",
        "        (cH[np1], cE[np2], -1.0),",
        "eigenmode_source_waveguide",
    ),
    "eigenmode_te_magnetic_sheet_sign_flipped": (
        "        (cH[np2], cE[np1], -1.0),",
        "        (cH[np2], cE[np1], +1.0),",
        "eigenmode_source_waveguide_te",
    ),
    # A lift that forgot the magnetic currents entirely: half the Love equivalence,
    # a weaker bidirectional launch that still steps cleanly.
    "eigenmode_magnetic_sheets_dropped": (
        "        (cH[np2], cE[np1], -1.0),\n        (cH[np1], cE[np2], +1.0),\n    )",
        "    )",
        "eigenmode_source_waveguide",
    ),
    # A lift that dropped the mode profile: four uniform sheets with the right signs.
    "eigenmode_mode_profile_dropped": (
        '        spec["amp_func"] = profile(mode_component)',
        '        spec["amp_func"] = None',
        "eigenmode_source_waveguide",
    ),
    # --- the OBLIQUE eigenmode launch: the normal and the solve direction ---------
    # The natural wrong implementation, and the one this feature exists to exclude:
    # collapse the SHEET NORMAL and the SOLVE DIRECTION back into one quantity.
    # mpb.cpp:875 takes the normal from an axis direction, :876-884 from the source
    # VOLUME for NO_DIRECTION, and get_eigenmode branches on the solve direction into
    # a wholly different Newton step (:605-612). Feeding the normal axis to the solve
    # aborts inside MEEP for oblique-planewave.py's one-pixel eig_vol but CONVERGES
    # for the rotated waveguide, at the same frequency, a group velocity within 0.7%
    # and a peak within 1% — and 4.1e-01 wrong. Only a field comparison catches it,
    # which is exactly why it is here.
    "oblique_solve_direction_collapsed_onto_the_normal": (
        "        solve_direction = mp.NO_DIRECTION if direction == mp.NO_DIRECTION else axes[normal]",
        "        solve_direction = axes[normal]",
        "eigenmode_oblique",
    ),
    # The mirror image: the SHEET NORMAL taken from the direction argument, which is
    # 5 (NO_DIRECTION) and not an axis at all. Loud here, but a lift that "fixed" the
    # IndexError by clamping would be silent, so the anchor pins the split itself.
    "oblique_sheet_normal_taken_from_the_direction_argument": (
        "        normal = _source_normal_axis(sim, source, dimensions)",
        "        normal = min(direction, 2)",
        "eigenmode_oblique",
    ),
    # --- the Gaussian beam --------------------------------------------------------
    # MEEP's uninitialized read (sources.cpp:691/:716) written as the physics it was
    # meant to be. The single most plausible "improvement" to this transcription: it
    # makes every plane-wave sanity check look better, costs 0.74% at the sheet, and
    # only reaches a sheet when Re(E0) or Im(E0) has an x component — invisible to any
    # suite built around gaussian-beam.py's own E0 = zhat.
    "beam_gb_hx_written_as_the_intended_gb_ey": (
        "    return 0.0 * gb_Ey",
        "    return gb_Ey",
        "beam_three_d",
        Path("meep_gpu/gaussian_beam.py"),
    ),
    # The beam vectors reduced to the cell's dimensionality — correct for a source's
    # centre and size (_lift_source does exactly that), wrong for beam_x0/beam_kdir,
    # which py_v3_to_vec keeps at full length on a D2 vec (simulation.py:130).
    "beam_vectors_reduced_to_the_cell_dimensionality": (
        "    beam_kdir = tuple(float(value) for value in _vector3(source.beam_kdir))",
        "    beam_kdir = tuple(float(value) for value in _vector3(source.beam_kdir))[:2] + (0.0,)",
        "beam_two_d_tm",
    ),
    # MEEP drops `amplitude` on a beam (measured ratio exactly 1.000000); an
    # EigenModeSource honours it, so reusing that branch here is a one-word slip that
    # rescales the whole run.
    "beam_honours_the_source_amplitude": (
        '        spec["amplitude"] = sign\n        spec["amp_func"] = profile(slots[sampled])',
        '        spec["amplitude"] = sign * complex(source.amplitude)\n'
        '        spec["amp_func"] = profile(slots[sampled])',
        "beam_two_d_te",
    ),
    # --- materials, the uniform path ---------------------------------------------
    "drude_lifted_as_lorentz": (
        '        kind=DRUDE if type(term).__name__ == "DrudeSusceptibility" else LORENTZIAN,',
        "        kind=LORENTZIAN,",
        "drude",
    ),
    # Re-anchored 2026-08-08: the call is `_scalar_or_components`-shaped now that a
    # DIAGONAL anisotropic D_conductivity_diag is supported, so the old find-text
    # (`set_conductivity(conductivity[0])`) no longer exists and the mutation had
    # stopped being applied.
    "conductivity_dropped": (
        "        driver.set_conductivity(\n"
        "            conductivity[0] if len(set(conductivity)) == 1",
        "        driver.set_conductivity(\n"
        "            None if True else conductivity[0] if len(set(conductivity)) == 1",
        "conductivity",
    ),
    # THE ANISOTROPY ITSELF, collapsed back to the pre-2026-08-08 engine: one shared
    # volume taken from component x. MEEP is per-component here (get_cnd,
    # meepgeom.cpp:1545-1559) and so is Fields._set_conductivity_side; only the
    # driver's public setter used to funnel the three together. Measured on the
    # anisotropic_conductivity parity case: 3.38e-01 against 4.81e-07, a complete run
    # with the full signal.
    "anisotropic_conductivity_collapsed_to_one_volume": (
        "            conductivity[0] if len(set(conductivity)) == 1",
        "            conductivity[0] if True  # mutation",
        "anisotropic_conductivity",
    ),
    "chi3_dropped": (
        "        driver.set_chi3(_scalar_or_components(chi3))",
        "        driver.set_chi3(0.0)",
        "chi3",
    ),
    "anisotropic_epsilon_collapsed": (
        "    driver.set_epsilon_components({\n        component: numpy.full(shape, diagonal[axis], "
        "dtype=numpy.float64)\n        for axis, component in enumerate(E_COMPONENTS)\n    })",
        "    driver.set_epsilon(numpy.full(shape, diagonal[2], dtype=numpy.float64))",
        "anisotropic",
    ),
    "susceptibility_kind_checked_with_isinstance": (
        "    if kind is not mp.LorentzianSusceptibility and kind is not mp.DrudeSusceptibility:",
        "    if not isinstance(term, (mp.LorentzianSusceptibility, mp.DrudeSusceptibility)):",
        "refused_by_name",
    ),
    # --- materials, the structured-dispersion path (sigma recovered from chi1inv) ---
    # The recovery's whole viability rests on solving for c = sigma*w0**2 rather than
    # for sigma: the naive unknown is singular on any real metal (cond 1e+17 on
    # meep.materials.Au, sigma returned NEGATIVE for a passive material) while raising
    # nothing. Both halves of the reparameterization are mutated, in sigma_recovery.py,
    # because either alone silently undoes it.
    "sigma_recovery_basis_parameterized_by_sigma": (
        "        return 1.0 / denominator",
        "        return self.scale / denominator",
        "basis or synthetic",
        Path("meep_gpu/sigma_recovery.py"),
        ("meep_gpu/test_sigma_recovery.py",),
    ),
    "sigma_recovery_scale_division_dropped": (
        "            sigma[index] = coefficients[index] / pole.scale",
        "            sigma[index] = coefficients[index]",
        "synthetic or drude_frequency or reader",
        Path("meep_gpu/sigma_recovery.py"),
        ("meep_gpu/test_sigma_recovery.py",),
    ),
    # Conditioning cannot see a wrong pole list; only the held-out frequency can, and
    # without it the failure mode is a plausible number and no error at all.
    "sigma_recovery_holdout_check_removed": (
        "        if not np.isfinite(residual) or residual > HOLDOUT_TOLERANCE:",
        "        if False:",
        "held_out",
        Path("meep_gpu/sigma_recovery.py"),
        ("meep_gpu/test_sigma_recovery.py",),
    ),
    # The recovery's unknowns are MEEP's CHAIN entries. susceptibility_equiv ignores
    # sigma (meepgeom.cpp:1633-1647), so deduplicating on the medium signature — which
    # includes it — gives two identical basis columns for one chain entry.
    "sigma_dedup_includes_sigma_like_the_medium_signature": (
        '        type(term).__name__ == "DrudeSusceptibility",\n'
        '        float(getattr(term, "frequency", 0.0)),\n'
        '        float(getattr(term, "gamma", 0.0)),',
        '        type(term).__name__ == "DrudeSusceptibility",\n'
        '        float(getattr(term, "frequency", 0.0)),\n'
        '        float(getattr(term, "gamma", 0.0)), _vector3(term.sigma_diag),',
        "shared_pole",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_recovery.py",),
    ),
    # --- materials, the structured-dispersion path (sigma looked up in the geometry) ---
    # THE TRAP THIS ROUTE EXISTS AROUND. MEEP fills each sigma array at the location
    # its own grid_volume gives the Yee point (anisotropic_averaging.cpp:336-338), and
    # containment is decided by fabs(proj) <= 0.5*size at the LAST BIT (libctl
    # geom.c:305). This engine's own coordinate for the same point agrees to 1e-7 of a
    # half cell and NOT bit for bit, which on a block face at +-1/3 puts 81 of 10800
    # points in a different medium. The mutation is that one ulp, applied to the one
    # line that chooses the coordinate — the smallest possible edit, and the exact
    # defect that produced a false result during this route's development.
    "sigma_lookup_yee_location_off_by_one_ulp": (
        "    return coordinates",
        "    return tuple(numpy.nextafter(value, -numpy.inf) for value in coordinates)",
        "nominal_coordinate or reader",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # A point outside MEEP's owned region reads vacuum-with-no-susceptibility from
    # chi1inv and from the sigma reader alike (monitor.cpp:182-184). _lattice_indices
    # enumerates the full closed lattice, so ~5% of its points are such points, and
    # taking the declared sigma there instead is wrong at exactly the cells where a
    # structure touches the cell edge — 21 of 1281 on the gold cell.
    "sigma_lookup_ownership_gate_removed": (
        "                    reading = location if gv.owns(location) else _meep_reads_here(",
        "                    reading = location if True else _meep_reads_here(",
        "reader or flush_with_the_cell_face",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # libctl stores the geometry tree in REVERSE declaration order (geom.c:1579), which
    # is what "later objects take precedence" means. Searching forwards puts the wrong
    # medium in every overlap.
    "sigma_lookup_object_precedence_reversed": (
        "    for obj in reversed(objects):\n        if mp.is_point_in_object(vector, obj):",
        "    for obj in objects:\n        if mp.is_point_in_object(vector, obj):",
        "overlapping",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # A point below the low corner of a PERIODIC axis is wrapped to the high face
    # before MEEP looks for an owning chunk (locate_point_in_user_volume,
    # boundaries.cpp), so on a k_point run that plane holds the high face's material
    # rather than vacuum. Skipping the wrap zeroes the sigma of any dispersive object
    # that reaches the cell edge.
    "sigma_lookup_periodic_lattice_wrap_dropped": (
        "    if cylindrical or not sim.k_point:\n        return (None, None, None)",
        "    if True:\n        return (None, None, None)",
        "periodic_cell_wraps",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # MEEP searches every SYMMETRY image before answering vacuum, so on a folded cell
    # the low plane of the folded grid_volume is read at its reflection. Dropping the
    # image calls that plane vacuum and zeroes a whole plane of sigma.
    "sigma_lookup_symmetry_image_dropped": (
        "    for axis in mirrors:",
        "    for axis in ():",
        "mirror_folded",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # libctl replicates each object by the lattice vectors when ensure_periodicity and
    # k_point (geom.c LOOP_PERIODIC), so an object crossing a face still covers the
    # owned points inside the opposite one.
    "sigma_lookup_periodic_object_images_dropped": (
        "    return tuple(shifts)",
        "    return ()  # mutation",
        "periodic_cell_wraps",
        DEFAULT_MODULE,
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # Without the per-point check the lookup is a REIMPLEMENTATION of MEEP's geometry
    # search rather than a read of it, and every hazard above becomes silent.
    "sigma_lookup_verification_removed": (
        "        if not np.isfinite(worst) or worst > tolerance:",
        "        if False:",
        "nominal_coordinate",
        Path("meep_gpu/sigma_lookup.py"),
        ("meep_gpu/test_sigma_lookup.py",),
    ),
    # --- symmetry ------------------------------------------------------------------
    "mirror_phase_forced_even": (
        '        Mirror("XYZ"[int(symmetry.direction)], int(complex(symmetry.phase).real))',
        '        Mirror("XYZ"[int(symmetry.direction)], 1)',
        "mirror",
    ),
    "mirror_axis_reversed": (
        '        Mirror("XYZ"[int(symmetry.direction)], int(complex(symmetry.phase).real))',
        '        Mirror("ZYX"[int(symmetry.direction)], int(complex(symmetry.phase).real))',
        "mirror",
    ),
    # --- the stopping condition ------------------------------------------------------
    "until_after_sources_read_as_a_plain_stop": (
        "            end_time = float(sim.fields.last_source_time()) + float(until_after_sources)",
        "            end_time = float(until_after_sources)",
        "until_after_sources or gaussian_until",
    ),
    # --- hosting MEEP's own step functions -------------------------------------------
    # The facade over the driver that `run_on_gpu` hands to MEEP's step-function
    # wrappers. Both mutations leave a run that completes, collects the same number of
    # samples, and returns a mode within a part in 10^6 of MEEP's.
    "round_time_carries_the_double_instead_of_meeps_float32": (
        "        return float(numpy.float32(self._driver.meep_time()))",
        "        return float(self._driver.meep_time())",
        "step_functions",
    ),
    "facade_falls_back_to_the_un_stepped_simulation": (
        "        try:\n"
        "            return getattr(self._driver, name)\n"
        "        except AttributeError:\n"
        "            raise StepFunctionNotHosted(",
        "        try:\n"
        "            return getattr(self._driver, name)\n"
        "        except AttributeError:\n"
        "            return getattr(self._sim, name)\n"
        "        if False:\n"
        "            raise StepFunctionNotHosted(",
        "step_function",
    ),
    # --- the report ------------------------------------------------------------------
    "lift_runs_despite_an_unsupported_report": (
        "    if not verdict.supported:\n        raise MeepSimulationNotLiftable(verdict.reasons)",
        "    if False:\n        raise MeepSimulationNotLiftable(verdict.reasons)",
        "refused_by_name",
    ),
    "bloch_correction_on_the_duplicate_plane_removed": (
        "            values[first] = values[first] * numpy.conjugate(phase)",
        "            values[first] = values[first] * 1.0",
        "bloch",
    ),
    # --- reduced dimensions: the traps the converter's own docstring names ---------
    # Each of these is a rule transcribed from MEEP's _infer_dimensions / use_2d /
    # _create_grid_volume. They are grouped here because a wrong answer is a DIFFERENT
    # SIMULATION that still steps, converges and returns a smooth field.
    "invariant_axis_thickness_passed_through": (
        "0.0 if axis in invariant\n        else (1.0 / float(sim.resolution) "
        "if float(length) == 0.0 else float(length))",
        "float(length) if axis in invariant\n        else (1.0 / float(sim.resolution) "
        "if float(length) == 0.0 else float(length))",
        "reduced_dimensions",
    ),
    "kz_never_blocks_the_collapse": (
        '    phased_z = bool(wavevector) and not getattr(sim, "special_kz", False) and (',
        "    phased_z = False and (",
        "reduced_dimensions",
    ),
    "zero_extent_on_any_axis_reduces": (
        "    return 2 if float(_vector3(sim.cell_size)[2]) == 0.0 and not phased_z else 3",
        "    return 2 if any(float(v) == 0.0 for v in _vector3(sim.cell_size)) "
        "and not phased_z else 3",
        "reduced_dimensions",
    ),
    # --- the PEC wall: implemented in the stepper, declared on the grid -----------
    # These reach modules the original battery could not edit at all.
    "invariant_axis_may_be_walled": (
        "        if not self.has_invariant:\n            return\n"
        "        for axis in range(3):\n"
        "            if not self.is_invariant(axis) or not self.metallic_axes[axis]:\n"
        "                continue",
        "        if not self.has_invariant:\n            return\n"
        "        for axis in range(3):\n"
        "            if True:\n"
        "                continue",
        "invariant or metallic",
        "meep_gpu/grid.py",
        ("meep_gpu/test_grid_pml.py", "meep_gpu/test_from_meep.py"),
    ),
    "metal_wall_D_not_held_at_zero": (
        "    _zero_metal(fields, D_COMPONENTS)",
        "    pass  # mutation",
        "metal",
        "meep_gpu/stepping.py",
        ("meep_gpu/test_stepping.py",),
    ),
    "metal_wall_B_not_held_at_zero": (
        "    _zero_metal(fields, B_COMPONENTS)",
        "    pass  # mutation",
        "metal",
        "meep_gpu/stepping.py",
        ("meep_gpu/test_stepping.py",),
    ),
    # --- the cylindrical ring integral: greencyl, which lives in dft.py -----------
    # Every one of these leaves a smooth far field that falls off as 1/r, integrates to
    # a plausible power and looks like a radiation pattern. Measured against CPU MEEP on
    # the m=1 resolution-20 cell, where the correct answer scores 2.33e-02: the ring
    # weight at the wrong radius 1.31e+00, a conjugated exp(i*m*phi) 8.27e-01, the ring
    # measure dropped 8.60e-01.
    "ring_weight_at_the_cell_edge_not_its_centre": (
        "            ring = 2.0 * np.pi * (starts[0] + rows + 0.5) * self.grid.dx\n"
        "            weights[0] = weights[0]",
        "            ring = 2.0 * np.pi * (starts[0] + rows) * self.grid.dx\n"
        "            weights[0] = weights[0]",
        "cylindrical or ring or near2far",
        "meep_gpu/dft.py",
        ("meep_gpu/test_cylindrical_near2far.py",),
    ),
    "greencyl_azimuthal_phase_conjugated": (
        "                    amplitude = complex(np.exp(1j * self.m * angle)) * spacing",
        "                    amplitude = complex(np.exp(-1j * self.m * angle)) * spacing",
        "cylindrical or ring or near2far or greencyl",
        "meep_gpu/dft.py",
        ("meep_gpu/test_cylindrical_near2far.py",),
    ),
    "greencyl_source_current_never_rotates": (
        "        vector[..., 0] * cosine - vector[..., 1] * sine,\n"
        "        vector[..., 0] * sine + vector[..., 1] * cosine,",
        "        vector[..., 0],\n"
        "        vector[..., 1],",
        "cylindrical or ring or near2far or greencyl",
        "meep_gpu/dft.py",
        ("meep_gpu/test_cylindrical_near2far.py",),
    ),
    "greencyl_quadrature_never_doubles": (
        "                live = live[change > norm * self.greencyl_tol]",
        "                live = live[:0]",
        "cylindrical or ring or near2far or greencyl",
        "meep_gpu/dft.py",
        ("meep_gpu/test_cylindrical_near2far.py",),
    ),
    # The regression that cost 1576x once already: terminating a wrapping axis with a
    # perfect mirror instead. Invisible whenever the absorber swallows the field first.
    "wrapping_axis_terminated_metallically": (
        "        else (METALLIC if grid.is_metallic(axis) else PERIODIC)",
        "        else METALLIC",
        "metal or periodic or wrap",
        "meep_gpu/stepping.py",
        ("meep_gpu/test_stepping.py", "meep_gpu/test_grid_pml.py"),
    ),
    # --- odd-count mirror folding: the parity the quadrant shape cannot carry ----
    # An odd folded count stores one MORE cell than the even rule (halve() gives
    # N - N//2 + 1), whose centre sits at exactly +L/2 with no lower image inside
    # MEEP's shifted window. Starting the mirrored slice from the QUADRANT COUNT
    # instead of n_full//2 is identical at every even count and writes every
    # odd-count mirror cell one cell off — the exact defect the fix removed, and
    # silent everywhere the pre-fix corpus could reach.
    "odd_fold_mirror_slice_from_quadrant_count": (
        "                phase * result[_slab(axis, slice(centre, 1, -1))]",
        "                phase * result[_slab(axis, slice(n_q - 1, 1, -1))]  # mutation",
        "round_trips",
        Path("meep_gpu/fields.py"),
        ("meep_gpu/test_fields.py",),
    ),
    # The DFT unfold used to re-derive n_full = 2*(n_quadrant - 1) from the stored
    # shape alone — the one place parity information is genuinely lost (m stored
    # cells could mean 2m-2 or 2m-3). Re-deriving it is bit-identical at even
    # counts and hands an odd-count monitor one phantom cell per folded axis.
    "dft_full_count_rederived_from_quadrant_shape": (
        "            centre = self._mirrored_cells(axis, n_quadrant)  # First positive-side cell.",
        "            centre = n_quadrant - 1  # mutation: parity re-derived from the quadrant",
        "odd_folded_axis_clips or full_coordinates",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py",),
    ),
    # --- the folded periodic far plane: the second mirror at +L/2 ----------------
    # The far-ghost fill images the stored row `n_full - stored + 2` into the slot
    # past big_corner WITH the component's parity about the second mirror.
    # Dropping the parity is the classic silent half: every component the plane
    # makes even is untouched, and the odd ones flip sign only ON the reflected
    # plane — smooth, plausible, and wrong by exactly 2x the face amplitude in the
    # stencils that read it.
    "folded_far_ghost_parity_dropped": (
        '            host_writes.copy_face(array, axis, -1, array, reflect_rows[axis],\n                                  _symmetry_phase(term.target, axis, parities[axis]))',
        '            host_writes.copy_face(array, axis, -1, array, reflect_rows[axis])  # mutation',
        "live_far_face or far_ghost_images_the_second_mirror",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_driver_integration.py", "meep_gpu/test_stepping.py"),
    ),
    # The reflect row is n_full - n_q + 2: the TOP stored row at an odd count and
    # the row BELOW the stored plane at an even one. The off-by-one form images
    # the beyond-top read one row too low — exactly where the reflect fill is
    # load-bearing: at an odd count the top cell's half-integer slot IS the
    # second-mirror plane, a stepped DOF whose forward difference reads past the
    # stored top, so the wrong image row corrupts an owned sample every step.
    # (The top-row misreading `stored_cells - 1` was measured EQUIVALENT and is
    # deliberately not a mutation here: it is arithmetically identical at odd
    # counts, and at even ones the differing value feeds only the top ghost-slot
    # updates, which _mask_non_owned_cells discards and fill_folded_far_ghosts_*
    # overwrites — fff probe vs CPU MEEP identical to the last digit.)
    "folded_far_reflect_row_one_low": (
        "        rows.append(n_full - grid.stored_cells(axis) + 2)",
        "        rows.append(n_full - grid.stored_cells(axis) + 1)  # mutation",
        "live_far_face",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_driver_vs_meep.py",),
    ),
    # --- the reflected monitor gather: regions in the half a fold discards ---------
    # The gather's sign is per component — MEEP's S.phase_shift(c, sn), which
    # add_dft_chunkloop bakes into the reflected chunk's scale. Collapsing it to
    # the even table leaves every even component exact and mirrors the odd ones
    # (Hx/Hz under a Y fold, every component under an odd-phase plane) with the
    # wrong sign: the flux through a discarded-half plane then reads the MIRROR
    # plane's sign, the below/above pair stops being sign-symmetric, and nothing
    # raises anywhere.
    "fold_gather_parity_collapsed_to_even": (
        "        if component_parity > 0:\n"
        "            return self.extended_even if interpolates else self.base_even",
        "        if True:  # mutation: every component takes the even factors\n"
        "            return self.extended_even if interpolates else self.base_even",
        "fold_gather_dft_and_flux",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # The reflection about doubled 0 is d -> -d exactly (symmetry::transform with
    # i_symmetry_point = icenter() = 0). An off-by-one-cell image — the slip every
    # registration defect in this package's history has taken — samples the row
    # beside the true image: fields stay smooth, shapes agree, and only the
    # against-MEEP comparison sees it.
    "fold_gather_image_one_row_off": (
        "        if doubled < -parity:  # Below the plane: the sn image, with its parity sign.\n"
        "            doubled = -doubled",
        "        if doubled < -parity:  # Below the plane: the sn image, with its parity sign.\n"
        "            doubled = -doubled + 2",
        "fold_gather or region_mapping_known_values_with_symmetry or spanning_the_fold",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # The raw-Yee accumulator folds the image's parity into its weight vector —
    # time-independent, so exact. Dropping it hands MEEP's load_near2far_data
    # reflected chunk data with the wrong sign; the far field stays smooth and
    # plausible, and only the packed / farfield comparison sees the flip.
    "near2far_fold_weight_parity_dropped": (
        "                weights = weights * np.where(\n"
        "                    image_codes == MIRROR_SITE_ZERO, 0.0,\n"
        "                    np.where(image_codes == MIRROR_SITE_REFLECTED, float(parity_sign), 1.0),\n"
        "                )",
        "                weights = weights * np.where(\n"
        "                    image_codes == MIRROR_SITE_ZERO, 0.0, 1.0,\n"
        "                )",
        "fold_gather_near2far",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # sn decomposes over the DECLARED planes, lowest bit first (measured: sn=3 is
    # both flips, sn=2 the second-listed plane alone). Shifting the bit reads the
    # wrong plane set: a single-mirror chunk with sn=1 decodes to NO flip, its
    # user-space bounds land outside every accumulator, and the migration dies
    # loudly — while a two-mirror run mixes the planes and emits the wrong rows.
    "near2far_sn_bits_shifted": (
        "        axis for bit, axis in enumerate(mirror_axes) if (sn >> bit) & 1",
        "        axis for bit, axis in enumerate(mirror_axes) if (sn >> (bit + 1)) & 1",
        "fold_gather_near2far",
    ),
    # A reflected chunk loops its STORED sites ascending — the requested span
    # DESCENDING — so the emission reverses those axes. Dropping the flip emits a
    # multi-site reflected face backwards: same values, same totals, wrong sites,
    # exactly the scrambled-layout class the chunk-walk exists to prevent.
    "near2far_sn_flip_order_dropped": (
        "            return accumulator.packed(extra_scale=weight * image_weight, box=box,\n"
        "                                      flip=tuple(flips))",
        "            return accumulator.packed(extra_scale=weight * image_weight, box=box,\n"
        "                                      flip=())",
        "fold_gather_near2far",
    ),
    # The is_integrated dipole must be withdrawn from D/B before every curl ladder
    # under a fold exactly as anywhere else; skipping the withdraw on folded grids
    # — the old refusal's fear, implemented — leaves the offset standing through
    # the PML rescale and reads ~1e0 wrong (the Defect-A number) while completing
    # smoothly.
    # --- the source-parity rule: WHICH extent, and what a uniform profile means ----
    # The rule refuses a source on a mirror plane that cannot carry the parity the
    # fold demands, and the whole question is which extent it reads. Only extent
    # ALONG THE FOLDED AXIS lets a source span the plane; extent on the other two
    # leaves it lying IN the plane, where the parity condition still constrains it
    # against itself. "Extended somewhere, therefore foldable" is the plausible
    # convention — it is what a reader who saw only sheets spanning the folded axis
    # would write, it passes every corpus row this rule unlocked, and it silently
    # accepts a sheet MEEP answers 57%-283% away from the unfolded run (measured,
    # stock MEEP 1.33.0, parity/meep_gpu/results/source_parity_predicate_2026-08-08/profile_predicate.jsonl).
    "source_parity_extent_read_on_any_axis": (
        "        if size[axis] <= tol:",
        "        if all(extent <= tol for extent in size):  # mutation: any extent at all",
        "parity_rule_is_the_profile",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_sources.py",),
    ),
    # The other half of the same rule. A source with no amp_func has a CONSTANT
    # profile, and a constant is even about every plane, so an odd parity is
    # impossible by inspection however far the sheet spans. Dropping that arm is the
    # revert to a pure extent predicate: it costs nothing on the corpus (every row
    # that lifts here carries an amp_func) and hands back a run that completes,
    # never warns, and is 0.569 (even phase) to 2.829 (odd phase) wrong.
    "source_parity_uniform_profile_accepted": (
        "        elif not has_amp_func:",
        "        elif False:  # mutation: extent alone is enough",
        "parity_rule_is_the_profile",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_sources.py",),
    ),
    # --- the same rule, at the GATE: what it reads, and what "absent" looks like -----
    # `gpu_compatibility` must reach the same verdict `FdtdDriver.add_source` will, and
    # both failure directions are silent. These two pin the pair of choices that decide
    # it, and each is a convention a careful reader would plausibly write.
    #
    # 1. WHAT THE GATE READS. `mp.EigenModeSource.component` is ALL_COMPONENTS, which
    #    names no field component, so a gate that reads that attribute asks the parity
    #    rule about nothing and answers "supported" for every eigenmode source. The
    #    driver then expands the declaration into the four Love-equivalence sheets
    #    (mpb.cpp:874-900) and raises on the odd pair. That is not hypothetical: it is
    #    what the gate did, and it is why five of MEEP's own python/tests cases
    #    (test_special_kz x2, test_mode_coeffs x3) sat in the accepted-but-does-not-lift
    #    column. The mutation is the revert, and it is invisible to any test that only
    #    checks the VERDICT — reading ALL_COMPONENTS also yields "supported", which is
    #    the right answer for these rows. Only the expansion itself separates them.
    "source_parity_gate_reads_component_only": (
        '    if kind in ("EigenModeSource", "GaussianBeamSource"):',
        '    if False:  # mutation: the gate reads source.component and stops',
        "mirror_parity_gate or parity_gate_expands",
        Path("meep_gpu/from_meep.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # 2. WHAT "NO PROFILE" LOOKS LIKE. MEEP's three amp spellings do not share an absent
    #    value: `amp_func` and `amp_data` default to None, but `amp_func_file` defaults
    #    to the EMPTY STRING (mp.Source.__init__). `is not None` on all three is the
    #    obvious line to write — it was written here first — and it reports a profile on
    #    every plain mp.Source ever declared. That silently disables the uniform-profile
    #    arm of the parity rule AT THE GATE only, so the gate says supported and the
    #    driver raises; and were both to read it that way, the refused case is a sheet
    #    stock MEEP 1.33.0 answers 0.569 (even phase) to 2.829 (odd phase) away from its
    #    own unfolded run — a smooth, finished, plausible field, not an error. Costs
    #    nothing on the corpus, where every row that reaches this rule carries a real
    #    amp_func.
    "source_parity_amp_func_file_absence_read_as_none": (
        '            or bool(getattr(source, "amp_func_file", "") or "")',
        '            or getattr(source, "amp_func_file", None) is not None  # mutation',
        "mirror_parity_gate or parity_gate_expands",
        Path("meep_gpu/from_meep.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    "integrated_withdraw_skipped_under_symmetry": (
        "        if not self.envelope.is_integrated or self._n_source_points == 0:\n"
        "            return",
        "        if (not self.envelope.is_integrated or self._n_source_points == 0\n"
        "                or any(_axis_is_mirrored(self.grid, d) for d in range(3))):\n"
        "            return",
        "integrated_source_through_the_fold",
        Path("meep_gpu/sources.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # A region extent on an invariant axis is MEEP's flat region (a reduced volume
    # has no slot for it — measured identical fluxes at 0.0e+00). Forwarding the
    # extent instead reaches the driver's direct-API refusal and the lift dies on a
    # run the gate reported supported — coupler.py's measured contract violation.
    "invariant_extent_forwarded_to_the_driver": (
        "    return tuple(\n"
        "        0.0 if is_invariant(axis) else abs(float(size[axis])) for axis in range(3)\n"
        "    )",
        "    return tuple(abs(float(size[axis])) for axis in range(3))",
        "invariant_axis",
    ),
    # The automatic decimation the migration carries is MEEP's RESOLVED factor,
    # read off the live chunk per monitor kind (F / E / chunks). Reading only the
    # near2far head regresses flux and dft_fields to the nominal-fwidth
    # re-derivation, which resolves 9 where MEEP resolved 8 — the monitors then
    # accumulate on different steps and read 5-20 % off while the stepped fields
    # agree at 1e-06.
    "resolved_decimation_read_for_near2far_only": (
        '    for attribute in ("F", "E", "diag", "chunks"):',
        '    for attribute in ("F",):',
        "fold_gather_dft_and_flux",
    ),
    # yee_grid=True is use_centered_grid=false — per-component chunks on the
    # component's OWN lattice. Migrating it to the centred monitor instead hands
    # back bilinear averages under the raw-Yee name; the shapes differ per
    # component parity and the against-MEEP comparison refuses to line up.
    "yee_grid_migrated_as_centred": (
        "        yee_grid = len(args) > 5 and args[5] is False",
        "        yee_grid = False",
        "yee_grid_dft_fields",
    ),
    # --- the subnormal guard on CuPy's kernel compiler ---------------------------
    # MEEP sets FTZ/DAZ when it initializes (src/mympi.cpp, meep#1708), which makes
    # NVRTC's front end read the subnormal literals in CCCL's <cuda/std/limits> as
    # underflow and refuse to build any kernel that includes it. Both mutations are
    # silent on a NumPy host and on a warm kernel cache — which is exactly how the
    # defect reached a 57-script corpus sweep as ONE failing script.
    "subnormal_guard_not_installed_on_the_gpu_backend": (
        "        if not guard_kernel_compilation(cupy) and subnormals_flushed():",
        "        if False and guard_kernel_compilation(cupy) and subnormals_flushed():",
        "guard",
        Path("meep_gpu/backends.py"),
        ("meep_gpu/test_backends.py",),
    ),
    # Leaving the default environment installed after a compile hands MEEP's own
    # stepping back a CPU that no longer flushes subnormals — the speedup meep#1708
    # exists for, silently withdrawn by a GPU-side helper.
    "subnormal_guard_leaks_the_default_environment": (
        "    try:\n"
        "        yield True\n"
        "    finally:\n"
        "        library.fesetenv(ctypes.addressof(saved))",
        "    try:\n"
        "        yield True\n"
        "    finally:\n"
        "        pass  # mutation",
        "keep_subnormals",
        Path("meep_gpu/backends.py"),
        ("meep_gpu/test_backends.py",),
    ),
    # The trap one level down, and the reason the guard first refused itself: CPython
    # parses float literals with dtoa's strtod, which finishes in floating-point
    # arithmetic, so a subnormal literal in a module compiled under MEEP's FTZ mode is
    # 0.0 in co_consts forever. Spelling the smallest subnormal instead of building it
    # from its bits makes the flush probe answer "flushing" unconditionally.
    "smallest_subnormal_spelled_as_a_literal": (
        '    return numpy.frombuffer((1).to_bytes(8, sys.byteorder), dtype=numpy.float64)[0]',
        '    return numpy.float64(5e-324)  # mutation',
        "subnormal",
        Path("meep_gpu/backends.py"),
        ("meep_gpu/test_backends.py",),
    ),
    # THE COLLAPSE OF A ZERO-EXTENT AXIS, restored to "keep the plane". MEEP's default
    # slice is `fields::total_volume()` = `gv.interior()`, which has zero extent on a
    # ONE-CELL axis (fields.cpp:717-724, vec.cpp:289-291); `array_slice` loops it on
    # the centred grid, so the bounds straddle the requested plane and both samples
    # carry weight 0.5 (loop_in_chunks.cpp:352-356, :275-287, array_slice.cpp:353-368),
    # and `collapse_array` sums them (:582-587). The lower sample is one lattice vector
    # down, so the collapse costs `(1 + conj(eikna))/2`. Selecting the stored plane
    # instead is EXACT at k = 0 — which is why it stood — and short by tan(pi*k*dx)
    # under a Bloch phase: FIRST ORDER in dx, so it halves with resolution instead of
    # converging. Measured on MEEP's own test_reflectance_angular_1_20_6 (kx on the
    # zero-extent x): 1.9346e-02 / 9.6721e-03 / 4.8357e-03 at resolution 100 / 200 /
    # 400, and on test_boundaries_1D 2.5781e-02 / 1.2888e-02 / 6.4439e-03 at 50 / 100 /
    # 200 — each equal to tan(pi*kx*dx) to five digits — against ~1e-06 with the mean.
    # The stepped fields were never involved: raw Yee storage matched MEEP at 3.0e-07.
    # THE REFUSAL'S OWN CLAIM, restored as a defect. Until 2026-08-08 this engine
    # refused a mirror plane combined with a Bloch phase on ANOTHER axis, on the
    # stated ground that "the folded-axis cell-centred readback averages across the
    # mirror plane without a Bloch factor". This mutation makes that true: a folded
    # run drops the lattice phase from every wrapped readback. It is the exact wrong
    # convention a reader of the old message would implement, and it is INVISIBLE to
    # everything the pre-fix suite could reach — inert without a fold, inert at k = 0
    # (the phase is None and the multiply is skipped anyway), and it leaves a smooth,
    # finite, complete complex field behind. It bites only where a fold and a Bloch
    # axis meet in one readback, which is why it needs the Ez case whose Yee shift
    # interpolates across BOTH the folded x and the phased y. Measured against CPU
    # MEEP: the folded leg moves 2.8e-06 -> 9.321311259e-02 while the unfolded leg
    # stays at its 2.5e-07 floor, and that folded number is bit-for-bit the one the
    # unconditional phase deletion produces — the equality that says the fold routes
    # its Bloch axis through the unfolded code and adds no hazard of its own.
    "bloch_factor_dropped_on_a_folded_run": (
        "        shifted = self.grid.xp.roll(array, -1, axis=axis)",
        "        shifted = self.grid.xp.roll(array, -1, axis=axis)\n"
        "        if any(self.grid.is_mirrored(d) for d in range(3)):\n"
        "            return shifted  # mutation: a folded run drops the lattice phase",
        "fold_and_a_bloch",
        Path("meep_gpu/fields.py"),
        ("meep_gpu/test_driver_vs_meep.py",),
    ),
    "collapsed_bloch_axis_takes_one_plane": (
        "        return arr * arr.dtype.type(0.5 * (1.0 + phase.conjugate()))",
        "        return arr  # mutation: keep the stored plane, drop the lattice image",
        "collapsed_axis or unit_axes_three_d_bloch",
        Path("meep_gpu/fields.py"),
        ("meep_gpu/test_fields.py", "meep_gpu/test_from_meep.py"),
    ),
    # THE SAME COLLAPSE, one level up, on the READ side of the assertion driver. When
    # MEEP's `get_array` is handed a zero-extent axis it weights the two bracketing
    # cells `1 - f` and `f` towards the requested coordinate and sums them away
    # (loop_in_chunks.cpp compute_boundary_weights, array_slice.cpp:414 and
    # collapse_array). Pairing the weights the other way round is the whole defect
    # class this mutation exists for, and it is INVISIBLE to MEEP's own slice tests:
    # every request in test_cavity_arrayslice sits exactly halfway between two cells,
    # where both weights are 0.5. Only the asymmetric offsets in
    # test_slice_like_meep.py can tell the two apart — which is why the mutation is
    # anchored here rather than left to the corpus.
    "flat_axis_collapse_weights_paired_the_other_way": (
        "    return (numpy.asarray([lower, lower + 1], dtype=int),\n"
        "            numpy.asarray([1.0 - fraction, fraction], dtype=float))",
        "    return (numpy.asarray([lower, lower + 1], dtype=int),\n"
        "            numpy.asarray([fraction, 1.0 - fraction], dtype=float))  # mutation",
        "flat_axis_slice",
        Path("parity/meep_gpu/drive_meep_test_assertions.py"),
        ("parity/meep_gpu/test_slice_like_meep.py",),
    ),
    # --- per-point material routes and the amp_data profile -----------------------
    # THE HALF CELL INSIDE MEEP'S ARRAY INTERPOLATOR, which is the subtlest choice in
    # the amp_data route by a distance. `map_coordinates` (fields.cpp:775-777) measures
    # the offset from the array cell's CENTRE — `rx*nx - x1 - 0.5` — and picks its
    # second bracketing index from the SIGN of that offset, so dropping the 0.5 leaves
    # a complete, smooth, correctly-shaped profile displaced by half an array cell.
    # Nothing about the source table, the deposition or the stepper changes.
    "amp_data_map_coordinates_no_half_cell": (
        "    dx = rx * nx - x1 - 0.5\n"
        "    dy = ry * ny - y1 - 0.5\n"
        "    dz = rz * nz - z1 - 0.5",
        "    dx = rx * nx - x1  # mutation\n"
        "    dy = ry * ny - y1  # mutation\n"
        "    dz = rz * nz - z1  # mutation",
        "amp_data_sheet",
        Path("meep_gpu/amp_interpolation.py"),
    ),
    # `epsilon_route_accepts_a_dispersive_medium` used to live here. Its anchor was the
    # BLANKET susceptibility refusal in `_epsilon_only_problem`, and that refusal is
    # gone rather than moved: MEEP itself drops a pole a MATERIAL_USER callable returns
    # (`geom_epsilon::add_susceptibilities` walks media only, meepgeom.cpp:1816-1830;
    # measured, get_epsilon reads 2.25 at frequency 0 and at 0.9 on such a cell), so the
    # old mutation now describes correct code. The defect it pinned — a callable's
    # dispersion silently dropped when MEEP is stepping it — is pinned by
    # `callable_dispersion_registration_ignored` above, on the predicate that replaced
    # it.
    # --- the stress tensor and the energy density ---------------------------------
    # THE SQRT IN sqrt_dV_and_interp_weights, which is the subtlest choice in the
    # force reduction by a distance. MEEP builds the diagonal chunks with
    # `sqrt_dV_and_interp_weights = true` (stress.cpp:181-184) and folds `sqrt(w)` in
    # PER STEP (dft.cpp:296-300), so the interpolation weight lands on |F|^2 exactly
    # ONCE. This accumulator stores the raw transform and applies the full weight at
    # read time, which is the same thing — but the OBVIOUS spelling, reusing the
    # linear `packed()` path every other consumer uses, applies the weight to the
    # transform and squares it here. Squaring the weight is exactly that bug, and it
    # is silent: the force stays finite, smooth and negative, just scaled by the cell
    # weight on every site.
    "force_diagonal_weight_applied_twice": (
        "        return float(xp.sum(weights * (real * real + imaginary * imaginary)))",
        "        return float(xp.sum(weights * weights "
        "* (real * real + imaginary * imaginary)))  # mutation",
        "diagonal_force_region or diagonal_force_reduction",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # THE TRACE SUBTRACTION IN THE DIAGONAL STRESS TERMS. `weight1 = where->weight *
    # (d == fd ? +0.5 : -0.5)` (stress.cpp:180): the force axis ADDS half its squared
    # field and the other two SUBTRACT theirs. Giving every axis +0.5 is a plausible
    # transcription slip — the constant is right, the sign rule is gone — and it
    # returns a force of the wrong sign and magnitude while everything about the run,
    # the registration and the accumulation stays correct.
    "force_diagonal_trace_not_subtracted": (
        "            weight.real * (0.5 if axis == self.force_direction else -0.5)",
        "            weight.real * 0.5  # mutation",
        "diagonal_force_region or diagonal_stress_weights",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # THE MEASURE APPLIED ONCE PER PAIR IN THE ENERGY REDUCTION. `add_dft_energy`
    # gives `include_dV_and_interp_weights = true` to E and H and FALSE to D and B
    # (dft.cpp:722-729), so the volume measure multiplies each pair exactly once.
    # Weighting both members — the symmetric-looking spelling — squares it, which on
    # a region whose ends are partially covered is neither MEEP's answer nor a
    # constant factor from it, and still returns a positive, smooth energy.
    "energy_measure_applied_to_both_sides_of_the_pair": (
        "                xp.conj(self._dft[weighted][index] * self._weights) "
        "* self._dft[plain][index]",
        "                xp.conj(self._dft[weighted][index] * self._weights) "
        "* self._dft[plain][index] * self._weights  # mutation",
        "energy_region_straddling or energy_reduction_weights",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_dft.py", "meep_gpu/test_from_meep.py"),
    ),
    # --- the field-function reducers: which lattice `loop_in_chunks` runs on -------
    # THE SUBTLEST CHOICE IN THE FAMILY. `fields::integrate` picks `cgrid = Centered`
    # only when the requested components do NOT all share a Yee shift
    # (integrate.cpp:135-143); a pair like (Ex, Dx) DOES share one, so
    # `electric_energy_in_box` loops that component's own raw lattice and reads RAW Yee
    # values. Forcing Centered everywhere is the natural-looking simplification — one
    # sample set, one interpolation rule — and it is wrong by a factor of 2.76 on
    # |Ez|^2 over a whole cell (measured, with and without symmetry alike) and by
    # 64%-99% on an energy. Every number it returns is finite, positive and smooth.
    "reducer_cgrid_always_centered": (
        "    if not components:\n        return None\n    first = reducer_component_shift(components[0])",
        "    if True:  # mutation: always the centred lattice\n        return None\n"
        "    first = reducer_component_shift(components[0])",
        "cgrid or constant_integrand or own_lattice",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_field_reducers.py",),
    ),
    # The material integrands are averaged onto cell centres with the COMPONENT's own
    # `yee2cent_offsets`, not read raw at the loop site: integrate.cpp:73-74 sits
    # outside the `if (cgrid == Centered)` branch at :70. Taking the centred shift here
    # skips the average entirely — exact on any uniform cell, and measured 1.7e-02 to
    # 3.6e-01 wrong against MEEP on the eps=12 waveguide cell of test_wvg_src.
    "reducer_material_read_raw_instead_of_averaged": (
        "    shifts = yee_shifts(component)\n    sampled = arrays.get_component(component)",
        "    shifts = _CENTERED_SHIFT if material else yee_shifts(component)  # mutation\n"
        "    sampled = arrays.get_component(component)",
        "material_average",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_field_reducers.py",),
    ),
    # `loop_in_chunks` clamps its loop at `user_volume.little_owned_corner(cgrid)`, so
    # the low PEC wall of a parity-0 component is never visited. `folded_axis_sites`
    # serves it as a ZERO because a DFT monitor's chunk stores a zero there — the same
    # answer for a FIELD, a different one for a general integrand. Measured on the
    # folded 10x10 cell at resolution 20, integrating the constant 1 with cgrid = Hx:
    # 99.875 with the site visited against MEEP's 99.6253125 without it.
    "reducer_folded_low_wall_visited": (
        "        while int(indices.size) and int(codes[0]) == MIRROR_SITE_ZERO:",
        "        while False and int(indices.size) and int(codes[0]) == MIRROR_SITE_ZERO:",
        "constant_integrand",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_field_reducers.py",),
    ),
    # `maxabs` is updated from the RAW integrand and the weight is applied after
    # (integrate.cpp:122-124). Taking the maximum of the weighted term instead scales
    # the peak by whatever fractional weight its cell happens to carry — 0.125 at a
    # partially covered end — and still returns a positive number of the right order.
    "reducer_max_abs_taken_after_the_weight": (
        "                integrand = complex(func(location, *(complex(block[i, j, k]) for block in host)))\n"
        "                magnitude = abs(integrand)",
        "                integrand = complex(func(location, *(complex(block[i, j, k]) for block in host)))\n"
        "                magnitude = abs(integrand * float(weight[i, j, k]))  # mutation",
        "max_abs",
        Path("meep_gpu/dft.py"),
        ("meep_gpu/test_field_reducers.py",),
    ),
    # `special_kz_phasefix` turns Ez, Hx and Hy — the TM triple a real-storage beta run
    # implicitly stores an i on (step_db.cpp:148-160) — and leaves Ex, Ey and Hz alone
    # (mpb.cpp:299-303). Turning the OTHER three instead is the same one-line edit with
    # the complementary component list; it injects a complete, stable mode with the
    # quarter turn on the wrong half and nothing raises.
    "special_kz_phasefix_turns_the_te_triple": (
        "    return -1j if mode_component in (mp.Ez, mp.Hx, mp.Hy) else 1.0",
        "    return -1j if mode_component in (mp.Hz, mp.Ex, mp.Ey) else 1.0  # mutation",
        "eigenmode_source_special_kz_real_imag",
    ),
    # --- special_kz: MEEP's out-of-plane beta term (step_db.cpp:148-176) -----------
    # Reading the two real/imaginary rows of MEEP's beta pass as ONE complex update
    # gives delta_f = sign * (-i) * 2*pi*beta*dt * g for D and (+i) for B — the
    # (2*cmp - 1) part-swap IS multiplication by -i, and (ft == D_stuff ? -1 : +1)
    # is what separates the two sides. Swapping the pair keeps a perfectly stable,
    # smooth, correctly scaled TE/TM coupling and simply runs it the wrong way round:
    # the driven Ez barely moves (its own case still reads ~1e-07 under this
    # mutation), and only the beta-GENERATED Ex sees it. The `beta_sign_flipped`
    # control in test_from_meep is precisely the reference this lands on.
    "special_kz_beta_ft_sign_swapped": (
        '        coefficient = coefficient * (1j if magnetic else -1j)',
        '        coefficient = coefficient * (-1j if magnetic else 1j)  # mutation',
        "special_kz",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # THE SUBTLEST CHOICE IN THE TERM, and the one only the real-storage case can see.
    # In real storage MEEP's `f[c_g][1 - cmp]` is NULL, so the whole
    # (ft == D_stuff ? -1 : +1) * (2*cmp - 1) factor collapses to 1 and BOTH sides
    # take the SAME real coefficient — that is the implicit-i-on-TM convention
    # (step_db.cpp:148-160), not an oversight. Carrying the ft sign into the real
    # branch as well is the natural "surely D and B differ" slip; it is INVISIBLE in
    # complex storage, where the branch is not taken at all, so it survives every
    # complex case in the suite and is caught only by special_kz_real_imag — which is
    # why that case exists and why refl-angular-kz2d.py is the corpus asset that
    # pinned it (3.43e-06 against CPU MEEP at is_real=1).
    "special_kz_beta_real_storage_takes_the_ft_sign": (
        "    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt",
        '    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt * (  # mutation\n'
        '        -1.0 if (magnetic and partner_values.dtype.kind != "c") else 1.0)',
        "special_kz_real_imag",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
    # d/dz on the invariant axis is the EXACT analytic factor i*2*pi*beta, not a
    # finite difference, so the beta term carries dt and no 1/dx — MEEP's betadt is
    # `2*pi*beta*dt`, against the main curl's Courant = dt/dx. Scaling it like the
    # curl leaves a stable run with a coupling too strong by the resolution.
    "special_kz_beta_scaled_like_a_finite_difference": (
        "    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt",
        "    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt / grid.dx  # mutation",
        "special_kz",
        Path("meep_gpu/stepping.py"),
        ("meep_gpu/test_from_meep.py",),
    ),
}


def unpack(entry: tuple) -> tuple[str, str, str, Path, tuple[str, ...]]:
    """(find, replace, selector) with the optional module and test targets defaulted."""
    find, replace, selector = entry[:3]
    module = Path(entry[3]) if len(entry) > 3 else DEFAULT_MODULE
    tests = tuple(entry[4]) if len(entry) > 4 else DEFAULT_TESTS
    return find, replace, selector, module, tests


def stale_anchors() -> list[tuple[str, str, int]]:
    """(mutation, module, count) for every anchor that does not resolve exactly once.

    An anchor that matches zero times means the battery reports the mutation as
    "not applied" instead of running it — the mutation silently stops testing
    anything, and the defect it was written to catch becomes invisible again. That
    is not hypothetical: the reduced-dimension work moved the code out from under
    four of them at once, and nothing noticed until they were counted.
    """
    stale = []
    for name, entry in sorted(MUTATIONS.items()):
        find, _replace, _selector, module, _tests = unpack(entry)
        count = module.read_text(encoding="utf-8").count(find)
        if count != 1:
            stale.append((name, module.name, count))
    return stale


def check_anchors() -> int:  # --check-anchors: the cheap guard the slow battery needs.
    stale = stale_anchors()
    for name, module, count in stale:
        print(f"  STALE {name:<52} anchor x{count} in {module}", flush=True)
    print(f"{len(MUTATIONS)} mutations, {len(stale)} stale anchors", flush=True)
    return 1 if stale else 0


def run_pytest(selector: str, timeout: float, tests: tuple[str, ...]) -> tuple[bool, str]:
    command = [sys.executable, "-m", "pytest", *tests, "-x", "-q"]
    if selector:
        command += ["-k", selector]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    tail = (completed.stdout or "")[-400:]
    return completed.returncode == 0, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="parity/meep_gpu/results/mutations.jsonl")
    parser.add_argument("--only", default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--check-anchors", action="store_true",
                        help="verify every anchor still resolves exactly once, then exit. "
                             "Runs in a second; the full battery takes tens of minutes, "
                             "which is why anchors rot unnoticed between runs.")
    args = parser.parse_args()

    if args.check_anchors:
        return check_anchors()

    # Every module any selected mutation touches is backed up BEFORE the first edit and
    # restored in the finally, so an interrupt cannot leave a mutated engine on disk.
    # This is the one script in the package that writes to its own source tree.
    names = sorted(MUTATIONS) if not args.only else args.only.split(",")
    targets = sorted({unpack(MUTATIONS[name])[3].resolve() for name in names})
    originals: dict[Path, str] = {}
    backups: dict[Path, Path] = {}
    for target in targets:
        assert target.exists(), target
        backups[target] = target.with_suffix(".py.mutation-backup")
        shutil.copy2(target, backups[target])
        originals[target] = target.read_text(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")

    print(f"mutating {len(targets)} module(s) : {len(names)} mutations", flush=True)
    for target in targets:
        print(f"  target {target.name}", flush=True)
    caught, survived, invalid = [], [], []
    try:
        for index, name in enumerate(names, start=1):
            find, replace, selector, module, tests = unpack(MUTATIONS[name])
            module = module.resolve()
            original = originals[module]
            started = time.time()
            if original.count(find) != 1:
                invalid.append(name)
                print(f"  {index:>2}/{len(names)} {name:<52} ANCHOR x{original.count(find)} "
                      f"(mutation not applied)", flush=True)
                with out.open("a") as handle:
                    handle.write(json.dumps({"mutation": name, "status": "anchor_missing"}) + "\n")
                continue
            module.write_text(original.replace(find, replace), encoding="utf-8")
            try:
                passed, tail = run_pytest(selector, args.timeout, tests)
            except subprocess.TimeoutExpired:
                passed, tail = False, "TIMEOUT (counted as caught: the suite did not finish)"
            module.write_text(original, encoding="utf-8")
            status = "SURVIVED" if passed else "caught"
            (survived if passed else caught).append(name)
            elapsed = time.time() - started
            with out.open("a") as handle:
                handle.write(json.dumps({"mutation": name, "status": status,
                                         "module": module.name, "selector": selector,
                                         "seconds": round(elapsed, 1),
                                         "pytest_tail": tail}) + "\n")
            print(f"  {index:>2}/{len(names)} {name:<52} {status:<9} ({elapsed:5.1f} s)",
                  flush=True)
    finally:
        for target, text in originals.items():
            target.write_text(text, encoding="utf-8")
            assert target.read_text(encoding="utf-8") == text, f"{target.name} was NOT restored"
            backups[target].unlink(missing_ok=True)

    print("", flush=True)
    print(f"caught   : {len(caught)}/{len(names)}", flush=True)
    print(f"SURVIVED : {len(survived)}  {survived}", flush=True)
    if invalid:
        print(f"anchor missing (mutation never applied): {invalid}", flush=True)
    return 1 if survived or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
