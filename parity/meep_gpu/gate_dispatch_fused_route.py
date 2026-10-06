#!/usr/bin/env python3
"""THE DRIVER-ROUTE GATE FOR THE FUSED ROUTE: a cross-sub-step pair, through the real seam.

WHAT CLAUSE (8) SAYS AND WHY THIS FILE EXISTS. ``fastpath._decide`` rung (8)
refuses every fused label with "a fused cross-sub-step product won {slots}; no
fused arm has been driven through the driver seam". That sentence is a statement
about whether a GATE HAS RUN, and until the fusion opt-in landed
(``fastpath.FUSE_ARMS_SWITCH``, 2026-08-29) no gate could run it: ``_decide``
passed ``fuse=False`` as a literal, so a fused product could not reach
``FastPathPlan.dispatch`` at all and the refusal was unfalsifiable. This gate is
the measurement that sentence names.

"DRIVEN THROUGH THE DRIVER SEAM" MEANS THE SEVEN REAL CONSULTS, nothing weaker.
``driver.MeepGPUDriver.step`` makes five (``step_B``, ``update_H``, ``step_D``,
``update_E``, ``update_P``) and ``synchronize_magnetic_fields`` makes two more
(``step_B``, ``update_H``) off a plan it deliberately does not re-freeze. Between
them the driver runs work that is behind NO consult — source withdraw and inject,
``fill_symmetry_bc_*``, ``zero_metal_*``, ``fill_folded_far_ghosts_*``. A fused
pair takes TWO of those five slots: the launch happens at the leading consult and
the absorbed consult answers off a sentinel (``NoopPlan``) or off
``deposit_repair.TrailingRepairPlan``. So the composition this gate measures is
not "a kernel matched an array call" — every weld already says that — it is:

    a launch that performed the curl, the metallic wipe AND the constitutive
    update, with the driver's own inject/fill/clear passes running BETWEEN the
    two consults and, where the kernel already did them, running a SECOND time
    on top.

THE PROTOCOL, per case, in ONE process on ONE device. Four legs, all lifted from
identical ``mp.Simulation`` declarations and byte-equal before a step is taken::

    fused      MEEP_GPU_DISPATCH=1  (the release admits, or FUSE_ARMS names)  the claim
    unfused    MEEP_GPU_DISPATCH=1  MEEP_GPU_FUSE_ARMS=0            the substitution baseline
    array      MEEP_GPU_DISPATCH=1  MEEP_GPU_FUSED=0                the oracle
    withheld   fused, with the ABSORBED consult forced False        the armed null control

THE BASELINE NEEDS THE VETO, and it did not until 2026-08-29. An unset
``MEEP_GPU_FUSE_ARMS`` used to mean "dispatch, do not fuse"; the release made it
mean the fused composition, so the ``unfused`` leg silently became a second copy
of ``fused``. MEASURED, not foreseen: run against the released tree on the GPU host
GPU 6, pml_2d reported 2.0 launches per step on BOTH legs,
``launch_drop_per_step 0.0``, verdict ``NO-DROP``, and this gate refused to
release. ``fastpath.FUSE_ARMS_VETO`` is the switch value that gives the baseline
back, and :func:`substitution_proof` now asserts the baseline actually took it
rather than trusting the environment.

``fused`` and ``unfused`` and ``array`` are compared word for word as uint32 over
every array on ``Fields`` and the PML, at every rung of a checkpoint ladder that
always contains 12 complete steps, and their flux spectra are compared at the
end. ``withheld`` is the control that makes the rest evidence: it runs the same
fused launch and then lets the driver's array ``update_H``/``update_E`` run ON TOP
of the constitutive half the kernel already did, which
``_apply_constitutive_pml`` ACCUMULATES rather than assigns, so it MUST diverge.
A run in which it does not diverge is a run whose byte comparison cannot see
whether the fused kernel's second half ran at all, and this gate says so and
fails.

FOUR INDEPENDENT ANSWERS TO "DID A KERNEL ACTUALLY RUN", because a leg that
quietly fell back matches trivially and this project has caught that shape of
pass repeatedly:

1. ``driver.active_step_path == "fused"`` and the record's ``fusion.driven``
   names a fused arm at BOTH slots of every pair;
2. ``fast_path_report()["launch_counters"]`` shows a nonzero dispatch count on
   the leading slot AND on the absorbed slot, with nonzero
   ``programs_per_dispatch`` on the leading one (a ``grid=(0,)`` warm launch
   counts in every counter and computes nothing);
3. a Triton-level counter on ``JITFunction.run`` and on
   ``CompiledKernel.launch_enter_hook``, installed outside anything this package
   owns, nonzero on ``fused`` and ZERO on ``array``;
4. THE SUBSTITUTION PROOF, which is the point of fusing and is new here: device
   launches PER STEP must DROP against the ``unfused`` leg by exactly one per
   fused pair per step, counted twice over (the JIT entry point and the launcher
   hook), while the two legs stay byte-identical.

THE SEAM-COMPOSITION CONTROL. On every case here the grid is metallically walled,
so ``zero_metal_B``/``zero_metal_D`` are LIVE array passes that the driver runs
unconditionally after the leading consult — and the fused pair's kernel already
performed that wipe in-launch (``FusedPairPlan.zero_metal``, resolved by the same
``coverage.zero_metal_axes`` the predicate checked with). Byte identity therefore
rests on the array pass being a value no-op on top of the launch. That is a NULL,
and a null is evidence only when it is shown discriminating, so the gate MUTATES
the pass — writes a value into the wall plane that a wipe cannot produce — and
requires the comparison to catch it.

Progress reporting: one flushed line per leg per chunk, and every row is appended to
``cases.jsonl`` as it lands.

A non-smoke leg starts only on a card the table under test certifies
(:func:`uncertified_card_refusal`): the package dispatches a supported, uncertified
card by default, and the record's recut would refuse every row such a leg wrote. It
also starts only with ``MEEP_GPU_ALLOW_UNCERTIFIED`` unset
(:func:`uncertified_switch_refusal`), so the legs measure the default.

Run (the GPU host, ONE pinned GPU, nothing installed by the harness so rung 8b does
the policy install itself)::

    CUDA_VISIBLE_DEVICES=<idx> python -u parity/meep_gpu/gate_dispatch_fused_route.py \\
        --out results/dispatch_fused_route_<stamp>/shipped
"""

from __future__ import annotations


import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy

# RESOLVED BY NAME, NEVER BY DEPTH. A ``parents[N]`` walk is correct only while
# this file stays at the depth it was written at; the harness traps this campaign
# already paid for include exactly that failure in a moved file.
HERE = os.path.dirname(os.path.abspath(__file__))
# The root is the nearest ancestor that holds both the package and the harness
# (``meep_gpu/`` and ``parity/meep_gpu/``): the repository root of this layout,
# and the same directory in any tree the harness is copied into whole.
_API = HERE
while _API != os.path.dirname(_API) and not (
        os.path.isdir(os.path.join(_API, "meep_gpu"))
        and os.path.isdir(os.path.join(_API, "parity", "meep_gpu"))):
    _API = os.path.dirname(_API)
if not os.path.isdir(os.path.join(_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for path in (HERE, _API):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate_dispatch_end_to_end as e2e  # noqa: E402  the byte comparator and the cases
import gate_provenance  # noqa: E402


def say(message: str) -> None:
    e2e.say(message)


# ---------------------------------------------------------------------------
# The cases: the end-to-end gate's nine, plus one this gate needs and it has not
# ---------------------------------------------------------------------------

def case_folded_complex_2d(mp, res=20):
    """A Y mirror AND a nonzero k_point: a folded grid whose FILL ARMS BOTH REFUSE.

    WHY A NEW CASE. ``folded_2d`` is a folded REAL run, and on it the mirror-fill
    arms WIN, so the composer fills ``fill_B``/``fill_D`` — slots outside
    ``DRIVER_SLOTS`` — and rung (6) refuses the plan for completeness before the
    fold rung is ever consulted. That makes the fold rung unreachable on it, and a
    gate that only ran ``folded_2d`` would report the fold rung as "standing"
    without ever having reached it.

    Complex storage is what separates the two. Measured on this configuration:
    both fill arms refuse (``mirror fill`` on the dtype, ``folded complex fill``
    on the expansion probe), so nothing outside ``DRIVER_SLOTS`` is filled, rung
    (6) passes, and whichever rung answers next is the one this case measures.
    """
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                         component=mp.Ez, center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y, side=mp.High)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_pml_3d_diagonal(mp, res=10):
    """3-D PML with an AXIS-ALIGNED block: the ordinary pairs, outside the envelope.

    WHY NOT ``pml_3d``. The end-to-end gate's 3-D case is a SPHERE, and MEEP's
    subpixel averaging on a curved surface writes off-diagonal ``chi1inv`` rows.
    Measured on this host: that grid answers ``off_diagonal_epsilon: True``, so
    ``launch.py:3054``'s specialized-family guard refuses BOTH ordinary pairs by
    name and no fused label reaches clause (8) at all — the leg that needs an
    un-admitted fused arm would have had nothing to be refused. An axis-aligned
    block keeps the tensor diagonal, so the composer selects ``fused pair B`` and
    ``fused pair D`` exactly as it does in 2-D.

    IT WAS THE TRITON ENVELOPE CONTROL UNTIL 2026-09-13, and it is a DRIVE case on
    every table now. What made it the right witness — ``dimensions=3`` being the
    ONLY reason ``fused_release_reasons`` gave — is exactly what this round's
    widening removes: the ordinary pairs' row moves to frozenset({1, 2, 3}) and the
    release admits the case, so it can no longer be the shape the release is shown
    declining. :data:`ENVELOPE_ROWS` names ``folded_dispersive5_2d`` instead.

    Small on purpose — resolution 10 on a 4x4x4 cell — because extent is not an axis
    any release row reads, and a modest cell ladders quickly. It is byte-compared
    now rather than read as a decision, so the ladder and the substitution proof
    apply to it like any other dispatch case.
    """
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                         component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        geometry=geometry, sources=sources, resolution=res)
    monitor = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [monitor], 20.0


def case_magnetic_seam_2d(mp, res=20):
    """A MAGNETIC source under 2-D PML: the deposit repair on the B SEAM.

    WHY THIS CASE. Every other dispatching case here carries an electric source,
    so the driver's injection lands between ``step_D`` and ``update_E`` and the
    B seam is CLEAN — ``update_H`` holds a ``launch.NoopPlan`` sentinel. That
    left one of the two installed pair shapes undriven by any gate: measured on
    the 2026-08-29 veto campaign, ``leading_repair_slots_left_running`` reads
    ``{"step_D": 12}`` in all three legs and ``step_B`` never appears. The
    dispatched composition nevertheless runs the B-seam repair 11 times over the
    corpus (``parity/meep_gpu/dispatch_reachability.served_in_dispatch``: 38
    deposit-repair pairs, 27 on D and 11 on B), so those rows rested on a
    byte-parity probe and on no gate at all.

    An ``mp.Hz`` source is what moves the deposit to the other seam. Measured on
    a NumPy lift of this configuration rather than predicted: the lifted source
    reports ``field_type == 'B'``, ``deposit_repair.in_seam_sources`` counts one
    on the B seam and zero on the D seam, ``repairable`` answers True, and
    ``seam_source_reasons`` returns ``()`` — so the only clauses left refusing on
    that host are the cupy-only ones every case here shares. It is the exact
    mirror of ``pml_2d``, and the two together drive both shapes on both seams:
    here ``step_B``/``update_H`` are the Leading/Trailing repair pair and
    ``step_D``/``update_E`` the sentinel pair, and there it is the other way
    round.

    Metallically walled and PML-bounded like the rest, so the in-seam wall-pass
    mutation control arms on both sides.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                         component=mp.Hz, center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(3.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 60.0


def case_pml_1d(mp, res=40):
    """A 1-D PML run: ONE live axis, real float32, the ordinary pairs on both seams.

    WHY IT IS HERE. ``FUSED_RELEASE_ENVELOPE`` pinned ``dimensions`` at 2 and
    refused four corpus seam-instances on that row alone, and nothing had ever
    asked whether the ordinary pairs are even SELECTED on a 1-D grid. Measured on
    the GPU host before this case was written: they are — the composer reports the same
    ``fused pair B``/``fused pair D`` cells it reports in 2-D, on a lift whose
    ``run_shape`` reads ``dimensions 1`` with ``grid_shape (1, 1, 640)``.

    ``dimensions=1`` IS PASSED EXPLICITLY. MEEP infers 1-D from a cell with two
    zero extents, and the inference is what the corpus's own 1-D rows rely on;
    stating it here makes the case's subject the axis the release is being widened
    on rather than a property of the cell size.

    An ELECTRIC source, so the deposit lands on the D seam and the B seam carries
    the sentinel pair — the same split ``pml_2d`` has, which is what makes the
    launch-drop arithmetic comparable between the two.
    """
    cell = mp.Vector3(0, 0, 16)
    geometry = [mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                         component=mp.Ex, center=mp.Vector3(0, 0, -5))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        dimensions=1)
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_folded_dispersive_2d(mp, res=20):
    """A Y mirror over a LORENTZ medium: the folded magnetic pair beside a live update_P.

    WHY A SECOND FOLDED DISPATCH CASE. ``folded_2d`` drives both folded pairs on a
    lossless, susceptibility-free grid, which is the shape the folded ELECTRIC pair
    is released on. This one adds the axis that case cannot reach: a susceptibility.
    Measured on the GPU host, the composition here is different in kind rather than in
    degree — ``update_P`` is live, the folded DISPERSIVE product wins the D seam,
    and the folded MAGNETIC pair takes ``step_B``/``update_H`` beside all of it. So
    it is the case that releases ``fused pair B (folded)`` on a dispersive grid and
    deliberately does NOT release ``fused pair D (folded)`` there.

    BOTH ITS SEAMS FUSE SINCE THE RE-ATTRIBUTION ROUND, and until then only the B
    one did. The D-seam product had no ledger entry until 2026-09-13 and was then
    held on a second-consult divergence that has since been attributed to the lazy
    subnormal-policy install rather than to this seam; its weld is PASS and its
    release row names this case, so the composer's offer now carries both labels.

    IT IS ALSO THE ONE CASE WHERE THE LABEL OFFER IS LOAD-BEARING, and what it
    measures there has changed direction rather than gone away. While the D product
    was refused, ``plan_step``'s ``fuse_labels`` is what kept clause (8) from
    rejecting the WHOLE plan over a label the run could not admit; now that both
    labels are admitted, the offer is what hands the composer a set that matches the
    release, and the case measures a two-family fold instead of a withheld one.
    """
    cell = mp.Vector3(4, 6, 0)
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.05, sigma=0.5)])
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                         component=mp.Ez, center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_cylindrical_m0(mp, res=20):
    """Dcyl at m = 0 WITH A FLUX MONITOR: the two cylindrical_real pairs.

    WHY A NEW CASE. ``FUSED_RELEASE_ENVELOPE`` pinned ``cylindrical`` at False and
    said why in one line: no cylindrical case had been driven, and a Dcyl grid also
    reports ``dimensions 2``, so a release keyed on dimensionality alone would have
    admitted a coordinate system no gate ran. This is that case. It is the
    end-to-end gate's ``case_cylindrical`` — the cylindrical family's own shape —
    rebuilt here rather than imported, for the reason the case list below states: a
    route-gate case is a NAME that ``fastpath.RELEASED_FUSED_ARMS`` binds, and a
    builder shared with another gate would let that gate's edits move what this
    release is cited from.

    WHAT THE COMPOSER DOES HERE, measured on a NumPy lift 2026-09-10 rather than
    predicted: ``launch.py``'s ``has_cylindrical`` branch refuses the ordinary and
    folded pairs by name, and the certified-product loop installs exactly the two
    ``cylindrical_real`` pairs and nothing else fused. The Ez source puts the
    deposit on the D seam, so ``step_D``/``update_E`` are the Leading/Trailing
    repair pair and ``step_B``/``update_H`` the ``NoopPlan`` sentinel pair — the
    same split ``pml_2d`` has, which is what makes the launch-drop arithmetic
    comparable between them.

    THE MONITOR IS WHAT MAKES TWO MEASUREMENTS NON-VACUOUS. ``case_cylindrical``
    returns no monitors, so on it the flux-spectrum comparison has nothing to
    compare and the second consult site (``synchronize_magnetic_fields``, which any
    flux or energy monitor reaches) is exercised only incidentally. One
    ``add_flux`` at r = 1.5 over the full z extent fixes both. If a preflight shows
    the Dcyl lift rejecting the ``FluxRegion``, the fallback is the monitor-less
    builder under THIS SAME case name — the name is what the release binds, the
    builder is harness-only — and the fallback is recorded rather than silent.
    """
    cell = mp.Vector3(4, 0, 4)
    sources = [mp.Source(mp.GaussianSource(frequency=0.3, fwidth=0.3),
                         component=mp.Ez, center=mp.Vector3(0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        dimensions=mp.CYLINDRICAL, m=0)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0, 0), size=mp.Vector3(0, 0, 2)))
    return sim, [monitor], 40.0


# ---------------------------------------------------------------------------
# The Tier 2 CUDA cases (2026-09-13). Every builder below was lifted and driven
# through this gate's own ``run_case`` on the GPU host before its release row was typed;
# the run shape each one reads is quoted in ``fastpath_cuda.CUDA_FUSED_RELEASE_ARM_AXES``.
# ---------------------------------------------------------------------------


def _cylindrical_complex_case(mp, res, m, complex_fields=False):
    """``case_cylindrical``'s cell, PML, resolution and a flux monitor at order ``m``.

    An Er source off-axis for ``m != 0`` (an Ez source at m = 1 is not a mode of the
    axis) and the m = 0 Ez source otherwise; ``complex_fields`` forces complex storage
    at m = 0, which is the ``dipole_in_vacuum_cyl_off_axis`` shape. The monitor makes
    the observable comparison and the second consult site non-vacuous.
    """
    cell = mp.Vector3(4, 0, 4)
    component = mp.Ez if m == 0 else mp.Er
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=component,  # noqa: SLF001
                         center=mp.Vector3(0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        dimensions=mp.CYLINDRICAL, m=m,
                        force_complex_fields=complex_fields)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0, 0), size=mp.Vector3(0, 0, 2)))
    return sim, [monitor], 40.0


def case_cylindrical_m0_complex(mp, res=20):
    """Dcyl m = 0 with ``force_complex_fields``: the cylindrical COMPLEX pairs at m = 0."""
    return _cylindrical_complex_case(mp, res, 0, complex_fields=True)


def case_cylindrical_mneg1(mp, res=20):
    """Dcyl m = -1: the cylindrical complex pairs on a negative order."""
    return _cylindrical_complex_case(mp, res, -1)


def case_cylindrical_m1(mp, res=20):
    """Dcyl m = 1 (complex storage): the cylindrical complex pairs' first case.

    Same builder as ``gate_dispatch_metal_route.case_cylindrical_m1`` (the Metal
    round's), kept as a second copy on purpose: that module is bound by the Metal
    driver record and this one by the CUDA record, and a shared home would put a
    Metal-only case edit on the CUDA campaign's bill and vice versa.
    """
    return _cylindrical_complex_case(mp, res, 1)


def case_cylindrical_m2(mp, res=20):
    """Dcyl m = 2."""
    return _cylindrical_complex_case(mp, res, 2)


def case_cylindrical_m3(mp, res=20):
    """Dcyl m = 3, the largest order the corpus carries."""
    return _cylindrical_complex_case(mp, res, 3)


def case_offdiag_2d(mp, res=20):
    """``pml_2d`` with its slab replaced by a CYLINDER: an off-diagonal chi1inv, Ez source.

    A curved surface is what makes MEEP's subpixel averaging write off-diagonal
    ``chi1inv`` rows (the mechanism the sphere in ``pml_3d`` relies on); every other
    axis of ``pml_2d`` is kept, so the two run shapes differ in
    ``off_diagonal_epsilon`` alone. With an Ez source the live E is out of plane and
    the D seam keeps the incumbent's ``offdiag`` singles: this is a ONE-seam case for
    the ordinary magnetic pair's off-diagonal corner.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(3.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 60.0


def case_offdiag_magnetic_2d(mp, res=20):
    """The cylinder with an Hz source: in-plane E on the D seam, so the STENCIL WELD runs.

    ``off-diagonal stencil weld`` spans ``step_D``/``update_E`` on the components the
    off-diagonal rows couple (Ex, Ey), which an Ez run never steps; the Hz source is
    what makes them live, and it keeps the driver's deposit on the B seam, where the
    ordinary magnetic pair carries it. Both products install here: measured 2026-09-13
    on the GPU host, 4 slots, byte-identical to 96 steps.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Hz,  # noqa: SLF001
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(3.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 60.0


def case_folded_offdiag_magnetic_2d(mp, res=20):
    """``folded_2d`` with the cylinder centred on the fold plane and an Hz source.

    Hz is odd under a mirror in Y (a pseudovector's z component flips), so the fold
    is ``Mirror(Y, phase=-1)``; the curved surface centred on the plane keeps the
    symmetry exact and writes the off-diagonal rows. The folded stencil weld takes
    the D seam, the ordinary magnetic pair the B seam, and the two fill slots are
    served by the Triton table's fill arms beside them, as on ``folded_2d``.
    """
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Hz,  # noqa: SLF001
                         center=mp.Vector3(-2.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y, phase=-1)])
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 50.0


def case_folded_offdiag_dispersive2_2d(mp, res=20):
    """``folded_offdiag_magnetic_2d`` at TWO Lorentz poles, carried by EVERY medium.

    THE ROUTE CASE OF ``cuda:dispersive off-diagonal``, the real-storage update_E
    single that gained a launch on 2026-09-27: a curved interface under subpixel
    averaging writes the off-diagonal chi1inv rows, a registered susceptibility moves
    update_E to ``D - sum P``, and that intersection is this family's alone. Each
    choice below was measured on a NumPy lift of this builder through the census's
    CuPy-named grid shim on 2026-09-27
    (``results/dispatch_flip_impl_2026-09-27/followup_cuda_case_selection.log``):

    * THE BACKGROUND IS DISPERSIVE TOO, with the SAME two poles as the cylinder, so
      the media differ in epsilon alone. A Lorentz cylinder in vacuum does not lift
      on a stock MEEP 1.33.0: with no ``fields.get_susceptibility_sigma`` the lift
      checks the susceptibility sigma against ``1/chi1inv[c][c]``, which is refused
      wherever chi1inv is not diagonal -- the refusal ``folded_offdiag_dispersive_2d``
      below the builders records. With one susceptibility everywhere there is no
      structured sigma to recover and the off-diagonal rows lift as permittivity.
    * TWO POLES, because at one pole a released TRITON product covers the B seam of
      every Lorentz-PML real grid -- ``fused pair B`` unfolded (no susceptibility
      pin) and ``fused pair B (folded)`` at susceptibilities {0, 1} -- and on
      ``cuda_default_precedence`` that fused product would read
      FUSED-WITHOUT-A-RELEASE under this row's ``no-fused-arm``. At two poles
      ``fastpath.released_fused_arms`` and ``fastpath_cuda.released_fused_arms``
      both read ``[]`` on this shape, so no leg drives a fused product. (2, 2, 2) is
      in ``dispersive_offdiag_update_e.POLE_COUNTS_SWEPT``.
    * FOLDED, for the same reason: the unfolded shape collides with ``fused pair B``
      at any pole count. The fold also puts the CUDA mirror-fill twins beside the
      single on the legs where this table composes first or alone.
    * AN Hz SOURCE, as on ``folded_offdiag_magnetic_2d``: it keeps Ex and Ey live, the
      two components the off-diagonal rows couple, and lands the driver's injection
      on the B seam, where two singles need no deposit repair.

    Lifted: dimensions 2, grid (160, 61, 1), folded 'mirror plane on Y',
    complex_storage False, off_diagonal_epsilon True, susceptibilities 2,
    pml_active True, bloch False. The CUDA table alone composes ``fill_B``/``fill_D``
    'mirror fill', step_B/step_D 'PML', update_H 'ordinary' and update_E
    'dispersive off-diagonal'; update_P ('ADE') has no launch and stays with the array
    path, or with the Triton table where it composes second.

    THE ROW FLIPS RED THE DAY EITHER TABLE RELEASES A FUSED PRODUCT ON THIS SHAPE
    (a fold release widened to two poles, or the pending
    ``cuda:dispersive off-diagonal polarization pair``), which is the fail-loud
    direction: the row is then rewritten for the product rather than passing on it.
    """
    chain = [mp.LorentzianSusceptibility(frequency=0.4 + 0.15 * index, gamma=0.02,
                                         sigma=1.1 / (index + 1))
             for index in range(2)]
    background = mp.Medium(epsilon=1.0, E_susceptibilities=chain)
    core = mp.Medium(epsilon=6.0, E_susceptibilities=chain)
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(), material=core)]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.3), component=mp.Hz,  # noqa: SLF001
                         center=mp.Vector3(-2.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        default_material=background,
                        symmetries=[mp.Mirror(mp.Y, phase=-1)])
    incident = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 50.0


def case_complex_beta_2d(mp, res=20):
    """``special_kz_2d`` stored COMPLEX (``kz_2d="complex"``): beta 0.4 under complex storage.

    The default ``kz_2d`` spelling forces complex64 storage, which routes the seam to
    the COMPLEX beta pairs and their expansion-probe licence; the real beta arm has
    ``special_kz_2d``. The lift reads ``bloch`` False: a z-only k_point is the beta,
    not a Bloch phase.
    """
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0, 0, 0.4), kz_2d="complex")
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_folded_special_kz_2d(mp, res=20):
    """``special_kz_2d`` with a mirror plane on Y: the FOLDED real beta pairs (Triton).

    Same cell, PML, slab, source and ``kz_2d="real/imag"`` spelling as
    ``special_kz_2d``, plus ``Mirror(Y)``; the lift reads ``folded`` "mirror plane
    on Y", ``beta`` 0.4, real storage, ``grid_shape`` (120, 62, 1).
    """
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0, 0, 0.4), kz_2d="real/imag",
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_folded_complex_beta_2d(mp, res=20):
    """``complex_beta_2d`` with a mirror plane on Y: the FOLDED complex beta pairs (Triton)."""
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0, 0, 0.4), kz_2d="complex",
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_bfast_1d(mp, res=25):
    """refl-angular's BFAST shape: a z-only cell DECLARED 3-D, ``bfast_scaled_k`` set, real storage.

    ``TestReflectanceAngular.test_reflectance_angular`` (the one BFAST row of the
    corpus) builds exactly this: ``dimensions=3`` on a ``(0, 0, L)`` cell, so MEEP
    builds a 3-D grid with one cell in x and y (the lift reads ``dimensions`` 3 and
    ``grid_shape`` (1, 1, 250)), a zero ``k_point`` with the incident angle carried
    by ``bfast_scaled_k`` = (n sin theta, 0, 0) and the Courant factor reduced by
    ``(1 - k_bar)/sqrt(3)``; the storage stays real, which is what the BFAST pairs
    read (a BFAST grid stored complex is refused by the family's own predicate).
    """
    import math  # noqa: PLC0415

    theta = math.radians(35.7)
    scaled = (math.sin(theta), 0, 0)
    courant = (1 - scaled[0]) / 3 ** 0.5
    cell = mp.Vector3(0, 0, 10)
    sources = [mp.Source(e2e._gaussian(mp, 0.5, 0.4), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=3,
                        default_material=mp.Medium(index=1), k_point=mp.Vector3(),
                        bfast_scaled_k=scaled, Courant=courant)
    monitor = sim.add_flux(0.5, 0.4, 5, mp.FluxRegion(center=mp.Vector3(0, 0, -2.0)))
    return sim, [monitor], 40.0


def case_nonlinear_1d(mp, res=40):
    """3rd-harm-1d's shape: a chi3 medium under PML, 1-D, Ex source.

    The Triton table's ``fused pair B (nonlinear)`` case (phase B). It was the CUDA
    envelope control for one preflight and cannot be: with that Triton arm released,
    the shipped precedence dispatches it here on the CUDA legs too, so the "nothing
    fused" direction fails on the OTHER table's release. ``nonlinear_3d`` is the CUDA
    witness (both tables refuse it on their own rows).
    """
    cell = mp.Vector3(0, 0, 10)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -3.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=1,
                        default_material=mp.Medium(index=1, chi3=0.1))
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3.0)))
    return sim, [monitor], 40.0


def case_nonlinear_3d(mp, res=10):
    """A UNIFORM chi3 medium in the 3-D PML cell: THE CUDA ENVELOPE CONTROL.

    Not a DRIVE case. Measured 2026-09-13 on the GPU host through ``envelope_leg``: with
    the switch unset nothing is released here on EITHER table (the CUDA magnetic pair
    refuses on its ``nonlinearity`` row, the Triton nonlinear pair on its
    ``dimensions`` row, every other arm on one of its own), so the request is empty;
    the opt-in of ``cuda:fused magnetic pair`` alone is admitted by its predicate
    (the nonlinear cell is welded) and driven while the electric pair is kept out by
    the offer -- the split both directions need. A 2-D nonlinear cell could not
    witness it: the special_kz pairs and the three-slot weld were still admitted
    there by rows that (until this round) carried no ``nonlinearity`` pin. The cell
    is UNIFORM because a structured chi3 cell is not liftable (two media differing in
    more than epsilon / conductivity / susceptibilities).
    """
    cell = mp.Vector3(4, 4, 4)
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        default_material=mp.Medium(index=1.5, chi3=0.1))
    monitor = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [monitor], 35.0


def case_complex_nobloch_2d(mp, res=20):
    """``pml_2d`` under ``force_complex_fields`` with NO k_point: complex, bloch False.

    The shape of ``wvg-src.py`` and ``TestWvgSrc``: complex storage asked for
    directly rather than implied by a k_point. Identical to ``bloch_2d``'s run shape
    except ``bloch`` False and ``k_point`` zero, so it is the case that lets the
    complex pairs' rows drop their ``bloch`` row. Same expansion licence as
    ``bloch_2d``; probe leg only.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        force_complex_fields=True)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 60.0


# ---------------------------------------------------------------------------
# THE TARGET-ROUND CASES (2026-09-13), REAL STORAGE FIRST.
#
# Every builder below was CONSTRUCTED AND LIFTED on this repository's stock MEEP
# 1.33.0 through ``meep_gpu.lift_simulation(sim, prefer_gpu=False)`` before it was
# written here, monitors included, and the shape quoted in each docstring is that
# lift's own ``fastpath._run_shape`` rather than a prediction. What the lift does
# NOT settle is which fused labels the COMPOSER selects on a device: the fused
# builders refuse on a NumPy host ("array module is 'numpy', not cupy"), so the arm
# sets in :data:`DRIVE` and :data:`DRIVE_CUDA` below are read off ``plan_step``'s
# refusal sets the way both tables' own notes prescribe — a label refused ONLY on
# the array module is one the composer WILL select once the builders can run — and
# each row says so rather than claiming a device measurement that has not happened.
#
# A CASE HERE IS A NAME THE RELEASE BINDS. ``fastpath.RELEASED_FUSED_ARMS`` and
# ``fastpath_cuda.CUDA_RELEASED_FUSED_ARMS`` cite these names, so a builder that is
# edited later changes what a shipped release is evidence for; edit the row and the
# builder in one change or not at all.
# ---------------------------------------------------------------------------


def case_folded_3d(mp, res=10):
    """A Y mirror on a 3-D PML cell: both folded pairs on a THIRD dimension.

    WHY IT IS HERE. ``fused pair B (folded)`` and ``fused pair D (folded)`` pin
    ``dimensions`` at 2 and their only case (``folded_2d``) is 2-D, so ten corpus
    seam-instances on folded 3-D grids are refused on that row alone
    (``mie_scattering.py``, ``TestLDOS.test_ldos_3D``, two ``TestLoadDump`` 3-D rows
    and ``TestModeDecomposition.test_grating_3d``). This is the fold at 3-D.

    AN AXIS-ALIGNED BLOCK, NOT A SPHERE, AND THAT IS THE WHOLE CONSTRUCTION. A
    curved surface makes MEEP's subpixel averaging write off-diagonal ``chi1inv``
    rows — the mechanism ``pml_3d`` relies on — and the grid would then answer
    ``off_diagonal_epsilon True``, which would make the case about that axis rather
    than about ``dimensions``. Measured on a lift of this builder, 2026-09-13,
    stock MEEP 1.33.0: ``dimensions 3, grid_shape (40, 21, 40), folded 'mirror
    plane on Y', complex_storage False, bloch False, beta 0.0, conductivity False,
    nonlinearity False, off_diagonal_epsilon False, susceptibilities 0,
    pml_active True``.

    Extent is not a pinned axis — the corpus rows this serves run to 207x106x106 —
    so a modest cell at resolution 10 is the shape the file's own convention asks
    for: small enough to ladder, exact on every axis the release reads.

    Same construction as ``gate_dispatch_metal_route.case_folded_3d``, which
    overrides this one when that module is imported; the two must agree cell for
    cell, because a divergence between them would be invisible.
    """
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=9))]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    incident = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [incident], 35.0


def case_dispersive5_2d(mp, res=15):
    """``dispersive_2d``'s shape at FIVE Lorentz poles: the pole COUNT as an axis.

    WHY A SECOND DISPERSIVE CASE. ``cuda:fused magnetic pair`` pins
    ``susceptibilities`` at ``{0, 1}`` and its cases carry exactly those two counts,
    so the four ``TestLoadDump`` 2-D rows — which carry five polarization states —
    are refused on that row. The axis is an integer COUNT
    (``len(fields.polarizations)``), and an integer axis left OPEN is precisely
    what the 2026-09-13 verifier caught crediting these rows on evidence of 0 and
    1; so the axis is widened to the values a case drove, and this case is the
    five.

    FIVE EXPLICIT LORENTZIANS RATHER THAN A CATALOG MATERIAL, for determinism: the
    census records a state COUNT, and an explicit list makes the count something
    the builder states rather than something a material table happens to carry.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (90, 90, 1), complex_storage False, bloch False, beta 0.0, conductivity False,
    off_diagonal_epsilon False, susceptibilities 5, pml_active True``.
    """
    medium = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(5)])
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_dispersive6_2d(mp, res=15):
    """``case_dispersive5_2d`` with ``range(6)``, and nothing else: the SIX point.

    The two builders differ in the pole count ALONE, which is what makes each one
    evidence for one value of one axis rather than for a new cell. Six is the count
    the three ``stochastic_emitter`` corpus rows carry (their material's census
    polarization block reads six states), and it is the upper value the CUDA
    magnetic pair's row and the Metal ``fused magnetic B/H pair``'s row are widened
    to.

    Measured on a lift of this builder, 2026-09-13: identical to
    ``dispersive5_2d`` except ``susceptibilities 6``.
    """
    medium = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(6)])
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_folded_dispersive5_2d(mp, res=20):
    """``folded_dispersive_2d`` at FIVE poles. **THE TRITON ENVELOPE WITNESS.**

    NOT A TRITON DRIVE CASE, DELIBERATELY, and the asymmetry is the point. The
    Triton envelope control has to be a shape on which the release DECLINES
    something its predicates admit — without one, no device run has ever shown the
    release able to say no, and a predicate that has only said yes is not known to
    be able to refuse. ``pml_3d_diagonal`` was that shape until this round widened
    ``dimensions`` on the ordinary pairs and made it dispatch; this case replaces
    it, on the axis the same round pins.

    WHY IT WITNESSES. ``fused pair B (folded)`` gains a ``susceptibilities`` row
    this round — it had none, while its only case carried zero poles, which is an
    unevidenced admit of exactly the class the 2026-09-13 verifier caught — and
    ``fused pair D (folded)`` already pins the axis at 0. THREE rows have to refuse
    five poles for this case to witness anything, not two, and the third arrived
    with the re-attribution round: ``fused pair D (folded dispersive)`` is released
    on ``folded_dispersive_2d`` and its row pins ``susceptibilities`` at 1 for this
    reason, having declined the unpinned exemption its sibling ``dispersive fused
    pair`` holds. MEASURED off-device against the post-batch Triton table — first
    2026-09-13 on two rows, re-measured on a NumPy lift of this builder after the
    third landed: ``fastpath.released_fused_arms`` returns an EMPTY set here, and
    every arm refuses BY NAME rather than by accident — the three folded rows on
    ``susceptibilities``, the unfolded pairs and the dispersive pair on ``folded``,
    the complex pairs on the absent mirror, the cylindrical pair on
    ``cylindrical``. That is direction 1 of :func:`envelope_leg`.

    WHY THE OPT-IN NAMES THE MAGNETIC ARM (see :data:`ENVELOPE_ROWS`): on a
    dispersive fold the composer gives the D seam to the folded DISPERSIVE product,
    not to ``fused pair D (folded)``, so an opt-in naming the D arm would leave the
    named arm undriven for a reason that has nothing to do with the envelope. The
    folded dispersive product is released now, which does not change that: its row
    refuses FIVE poles, so on this shape it is neither offered nor driven and
    direction 2 still reads it under ``not_installed`` as offered to run.
    ``ENVELOPE_ROWS['cuda']`` withholds the magnetic pair for the same reason.

    IT IS ``folded_dispersive_2d`` WITH FIVE POLES INSTEAD OF ONE — same cell, PML,
    mirror, source and monitor — so the two differ in ``susceptibilities`` alone.
    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (80, 61, 1), folded 'mirror plane on Y', complex_storage False, bloch False,
    beta 0.0, conductivity False, off_diagonal_epsilon False, susceptibilities 5,
    pml_active True``.

    ON THE METAL TABLE IT IS A DISPATCH CASE, not a witness, and that is
    precedented rather than confusing: ``pml_3d_diagonal`` is a Triton control and
    a CUDA/Metal dispatch case, ``no_pml_2d`` a Metal control and a ``DRIVE_CUDA``
    negative. ``released_here`` is per table, and a Triton leg with no backend
    preference does not dispatch another table's arms.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(5)])
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_absorber_1d(mp, res=40):
    """A 1-D ABSORBER cell over a five-pole medium: no PML, and a conductivity.

    WHY IT IS HERE. An ``mp.Absorber`` is a scalar conductivity rather than a
    split-field PML, so the lift hands the fast path no PML object at all and the
    run shape reads ``pml_active False`` WITH ``conductivity True`` — the one
    combination no existing case carries. ``no_pml_2d`` evidences neither row: it
    has no poles and no conductivity. This is the case the no-absorber dispersive
    products on both tables are waiting for, and the CUDA table's shipped pending
    text names it by this spelling.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 1, grid_shape
    (1, 1, 400), complex_storage False, bloch False, beta 0.0, conductivity True,
    off_diagonal_epsilon False, susceptibilities 5, pml_active False``.

    ``dimensions=1`` IS PASSED EXPLICITLY, for ``pml_1d``'s reason: a declared 1
    wins outright in the dimensions reader, and stating it makes the case's subject
    the axis the release is widened on rather than a property of the cell extents.
    Five explicit Lorentzians for ``dispersive5_2d``'s reason — the axis is a count.
    Serves ``examples:absorber-1d.py`` and ``TestAbsorber.test_absorber``.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(5)])
    cell = mp.Vector3(0, 0, 10)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -3.0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.Absorber(1.0, direction=mp.Z)],
                        default_material=medium, sources=sources,
                        resolution=res, dimensions=1)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_material_dispersion_0d(mp, res=20):
    """A ZERO-EXTENT cell over a two-pole medium: no boundary layer of any kind.

    THE SECOND HALF OF ``absorber_1d``'s ROW, and the reason both no-absorber
    dispersive rows can leave ``conductivity`` unpinned honestly: this case carries
    ``conductivity False`` where that one carries True, and ``dimensions 2`` where
    that one carries 1, so between them the two values of both axes are DRIVEN
    rather than assumed. Without it the CUDA no-absorber weld's row measures +4
    instead of +6 and the Triton row has to pin ``dimensions`` 1.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (1, 1, 1), complex_storage False, bloch False, conductivity False,
    off_diagonal_epsilon False, susceptibilities 2, pml_active False``. MEEP's own
    inference reports 2 on a zero-extent cell, which is the ``dimensions = 2``
    point the rows need, and ``pml_active False`` comes free from declaring no
    boundary layers.

    IT RETURNS NO MONITORS, and that is a property of the cell rather than an
    omission: a single-cell grid has nothing to put a flux plane on. The flux
    comparison and the SECOND CONSULT SITE are therefore NOT-APPLICABLE here — the
    same record the Metal table keeps for ``cylindrical`` — and must not be scored
    as passes. Serves ``examples:material-dispersion.py``.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(2)])
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3())]
    sim = mp.Simulation(cell_size=mp.Vector3(), default_material=medium,
                        sources=sources, resolution=res)
    return sim, [], 40.0


def case_no_pml_dispersive_2d(mp, res=20):
    """A 2-D NO-ABSORBER dispersive cell: no boundary layer, no conductivity.

    THE REPLACEMENT FOR ``material_dispersion_0d`` ON THIS TABLE, and it drives the
    same two axis values that case was carrying -- ``dimensions`` 2 against
    ``absorber_1d``'s 1, ``conductivity`` False against its True -- on a cell with
    REAL EXTENT. That is the whole point: the four non-vacuity controls this gate
    owns are all vacuous on a (1, 1, 1) grid, and the in-seam mutation decisively so,
    because the wall plane it writes IS the single cell, which is also the source's
    deposit cell ``deposit_repair.apply`` recomputes. On a (120, 120, 1) grid the
    mutated plane has an in-launch twin and the control can diverge.

    ``no_pml_2d`` cannot serve: it carries no poles and no conductivity, and it lifts
    ``off_diagonal_epsilon True`` (a cylinder writes the off-diagonal rows), which
    this arm refuses on that axis instead.

    Declaring NO boundary layers at all is what lifts ``pml_active`` False; the two
    Lorentzians are what make ``fields.stores_E`` True, which is Arm S's second
    positive clause (``no_pml_stored_e.stored_e_constitutive_coverage``).

    Measured on a lift of this builder, 2026-09-14: ``dimensions 2, grid_shape
    (120, 120, 1), complex_storage False, bloch False, beta 0.0, conductivity False,
    off_diagonal_epsilon False, susceptibilities 2, pml_active False,
    nonlinearity False, cylindrical False``.

    IT RETURNS A MONITOR, unlike the case it replaces, so the flux comparison and the
    SECOND CONSULT SITE are measurements here rather than NOT-APPLICABLE rulings.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3, gamma=0.05, sigma=0.4),
        mp.LorentzianSusceptibility(frequency=0.45, gamma=0.05, sigma=0.2)])
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=mp.Vector3(6, 6, 0), boundary_layers=[],
                        default_material=medium, sources=sources, resolution=res)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


# ---------------------------------------------------------------------------
# THE TARGET-ROUND CASES, COMPLEX CARTESIAN. Every one dispatches on an expansion
# probe leg and on no other: the complex products' licence comes from the unified
# probe record and nothing else offers it. Lifted the same way and on the same day
# as the real-storage block above; the Dcyl rows stay where they are, LAST.
# ---------------------------------------------------------------------------


def case_complex_1d(mp, res=40):
    """ONE declared dimension under complex storage and a Bloch phase in z.

    WHY IT IS HERE. The complex pairs on all three tables pin ``dimensions`` at 2
    and their cases are all 2-D, so every 1-D complex corpus row is refused on that
    row alone. This is the 1-D point of that widening, and — the Metal table's own
    finding — the FIRST 1-D evidence either of its complex arms carries, since
    neither family gate has driven ``dimensions=1``.

    ``dimensions=1`` IS PASSED EXPLICITLY, as ``pml_1d`` passes it: a declared 1
    wins outright in the dimensions reader, and stating it makes the case's subject
    the axis rather than a property of the cell extents. The k_point is what forces
    complex storage physically; ``force_complex_fields`` is declared beside it so
    the storage does not depend on the phase being nonzero.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 1, grid_shape
    (1, 1, 480), complex_storage True, bloch True, k_point (0.0, 0.0, 1.25),
    beta 0.0, conductivity False, off_diagonal_epsilon False, susceptibilities 0,
    pml_active True``. Serves ``refl-angular.py``, ``TestPlanewave1D`` and
    ``TestReflectanceAngular_0_0``.
    """
    cell = mp.Vector3(0, 0, 12)
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=1,
                        k_point=mp.Vector3(0, 0, 1.25),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_complex_3d_thinline(mp, res=25):
    """A ``(0, 0, L)`` cell DECLARED 3-D, complex, with an oblique k: the 3-D point.

    WHY THE DEGENERATE GRID IS THE RIGHT SHAPE HERE RATHER THAN A COMPROMISE. Four
    corpus rows (``antenna_pec_ground_plane_1D.py``, ``dipole_in_vacuum_1D.py``,
    ``TestBoundaries1D.test_boundaries_1D``,
    ``TestReflectanceAngular.test_reflectance_angular_1_20_6``) are physically 1-D
    looking and answer ``dimensions 3``, because MEEP's rule reads a DECLARED 3
    with a nonzero z extent as 3 and builds a grid one cell wide in x and y. The
    case declares exactly that, so what it evidences is the value the census
    actually reports for those rows.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 3, grid_shape
    (1, 1, 275), complex_storage True, bloch True, k_point (0.41, 0.0, 1.8),
    beta 0.0, susceptibilities 0, pml_active True``.

    IT IS NOT THE WHOLE 3-D EVIDENCE, and :func:`case_complex_3d` is why: a genuine
    3-D complex grid is driven beside it so the widened row is not licensed by
    degenerate cells alone.
    """
    cell = mp.Vector3(0, 0, 11)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=3,
                        k_point=mp.Vector3(0.41, 0, 1.8),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_complex_3d(mp, res=10):
    """A GENUINE 3-D complex grid, and it is here for honesty rather than for count.

    WHAT IT ADDS. The seven corpus rows the complex ``dimensions`` widening serves
    are all ``(1, 1, N)`` cells, so ``complex_1d`` and ``complex_3d_thinline``
    together would license ``dimensions = 3`` on evidence from degenerate grids —
    while the corpus carries genuine 3-D complex rows (the four ``TestLoadDump``
    3-D rows at (35, 32, 41) and ``TestMaterialGrid.test_matgrid_3d`` at
    (25, 25, 25)) that the widened row would admit AT RUNTIME even though no board
    credit moves (they are served by other products today). Measured by the Metal
    round: the instance count is identical with and without this case. It is here
    because a row that admits a shape no case drove is the defect class this round
    exists to close, and a zero-instance case that closes one is worth its campaign
    minutes.

    AN AXIS-ALIGNED BLOCK, for ``folded_3d``'s reason: a curved surface would move
    the case onto the off-diagonal axis. The PML is on Z alone so the in-plane
    Bloch phase is carried by a periodic direction rather than absorbed.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 3, grid_shape
    (40, 40, 40), complex_storage True, bloch True, k_point (0.3, 0.0, 0.0),
    beta 0.0, off_diagonal_epsilon False, susceptibilities 0, pml_active True``.
    """
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(0.8, direction=mp.Z)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [monitor], 35.0


def case_folded_complex_3d(mp, res=20):
    """The triangular-lattice oblique cell: a mirror on X, complex, 3-D.

    WHY IT IS HERE. The folded complex pairs pin ``dimensions`` at 2 on all three
    tables and their only case (``folded_complex_2d``) is 2-D, so
    ``TestModeDecomposition.test_triangular_lattice_oblique`` is refused on that row
    alone. This is that row's 3-D point, built on the corpus row's own cell — an
    oblique lattice with an in-plane Bloch phase beside the z one, which is what
    keeps ``bloch`` True, the value the arm already pins.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 3, grid_shape
    (8, 21, 126), folded 'mirror plane on X', complex_storage True, bloch True,
    k_point (0.0, -1.7035, 2.4694), beta 0.0, off_diagonal_epsilon False,
    susceptibilities 0, pml_active True``.

    THE MONITOR IS WHAT MAKES TWO MEASUREMENTS NON-VACUOUS — the flux comparison and
    the second consult site, ``synchronize_magnetic_fields``, which any flux monitor
    reaches. If a preflight shows the lift rejecting the ``FluxRegion``, the
    fallback is the monitor-less builder under THIS SAME case name, recorded rather
    than silent: the name is what the release binds, the builder is harness-only.
    That is ``cylindrical_m0``'s ritual, applied here.
    """
    cell = mp.Vector3(0.6, 1.0392304845413263, 6.3)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -2.0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Z)],
                        sources=sources, resolution=res,
                        k_point=mp.Vector3(0, -1.7035, 2.4694),
                        symmetries=[mp.Mirror(mp.X)],
                        force_complex_fields=True)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 0, 2.0), size=mp.Vector3(0.6, 1.0392304845413263, 0)))
    return sim, [monitor], 40.0


def case_folded_complex_kz2d_3d(mp, res=20):
    """A ``kz_2d="3d"`` cell with a mirror on Y: the folded complex pairs' SECOND 3-D corner.

    WHY A SECOND ONE. ``folded_complex_3d`` is a genuine volumetric grid;
    ``TestEigCoeffs.test_binary_grating_special_kz_2_21_2`` is not — it is a zero-z
    cell that MEEP nonetheless carries as 3-D because ``kz_2d="3d"`` keeps the z
    component of the k_point from lifting as a beta. The two corners are built
    differently and both answer ``dimensions 3``, which is what makes the widened
    row evidence for the axis rather than for one cell shape.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 3, grid_shape
    (90, 62, 1), folded 'mirror plane on Y', complex_storage True, bloch True,
    k_point (2.797, 0.0, -1.0849), beta 0.0, susceptibilities 0, pml_active True``.

    THE SOURCE POSITION IS LOAD-BEARING AND IT WAS MEASURED THE HARD WAY. With the
    source at ``y = -2.0`` the lift REFUSES by name: every source sits in the half
    its mirror plane DISCARDS, and this engine stores the upper half of a folded
    axis as MEEP does. MEEP itself does not refuse it — sources are added with
    ``use_symmetry=false``, so none lands in a stored chunk and the run returns a
    field whose peak is exactly 0 while reading as complete. Declared at
    ``y = +2.0`` it lifts. EVERY folded case in this file puts its sources in the
    KEPT half for that reason.
    """
    cell = mp.Vector3(4.5, 6.0, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(0, 2.0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(2.7970, 0, -1.0849), kz_2d="3d",
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4.5, 0)))
    return sim, [monitor], 45.0


def case_folded_complex_offdiag_2d(mp, res=20):
    """``folded_complex_2d`` with the slab replaced by a CYLINDER on the fold plane.

    The same cell, PML, k_point, mirror and source as ``folded_complex_2d``, so the
    two run shapes differ in ``off_diagonal_epsilon`` ALONE — which is why the cell
    is 4 x 6 here rather than the corpus row's own extent: the case is about one
    axis of a row whose other values that case already drove. A curved surface
    centred on the fold plane is what makes MEEP's subpixel averaging write the
    off-diagonal ``chi1inv`` rows while keeping the symmetry exact.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (80, 62, 1), folded 'mirror plane on Y', complex_storage True, bloch True,
    k_point (0.3, 0.0, 0.0), beta 0.0, conductivity False,
    off_diagonal_epsilon True, susceptibilities 0, pml_active True``.

    Serves ``TestHoleyWvgBands.test_fields_at_kx`` as one of the three instances
    behind the folded-complex magnetic pair's off-diagonal corner on both device
    tables. It is NOT the CUDA folded-complex off-diagonal stencil weld's case:
    its Ez source lands on the D seam, which that weld refuses by design -- see
    ``case_folded_complex_offdiag_hz_2d`` below.
    """
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y, side=mp.High)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_folded_complex_offdiag_hz_2d(mp, res=20):
    """``folded_complex_offdiag_2d`` with the corpus row's OWN source: Hz under an odd mirror.

    Same cell, PML, cylinder, k_point and resolution as ``folded_complex_offdiag_2d``.
    The source is ``mp.Hz`` and the mirror is ``mp.Mirror(mp.Y, phase=-1)``, which is
    the declaration ``TestHoleyWvgBands.test_fields_at_kx`` makes
    (``test_holey_wvg_bands.py:17-23``); the two builders differ in the SEAM the
    driver's injection lands on and in nothing else.

    WHY A SECOND BUILDER RATHER THAN A SOURCE SWAP. An electric source is injected
    BETWEEN step_D and update_E, and the folded complex off-diagonal stencil weld
    declares ``CARRIES_DEPOSIT_REPAIR = False``, so on the Ez builder its own
    predicate refuses at the source-seam clause and the D seam falls to the single
    arms -- measured off-device 2026-09-17 on the real lift:
    ``seam_source_reasons(..., "D", carries_repair=False)`` names ``source 0
    (GaussianPulsedSource)``. A magnetic source lifts as a ``VolumeSource`` of field
    type ``B``, leaves the D seam empty, and the weld admits. The Ez builder stays as
    it is because the Triton DRIVE row and the magnetic pair's off-diagonal corner
    drive it; this one is the weld's, and the 2026-09-17_allpaths campaign read
    VACUOUS-PASS ("names 2, expected 4") on the Ez row for exactly this reason.

    Composed off-device on a lift of this builder, 2026-09-17, through the census's
    CuPy-named grid shim: ``step_B``/``update_H`` folded complex fused magnetic pair,
    ``step_D``/``update_E`` folded complex off-diagonal stencil weld -- two pairs.
    Lift: ``dimensions 2, grid_shape (80, 62, 1), complex_storage True, bloch True,
    k_point (0.3, 0.0, 0.0), off_diagonal_epsilon True, pml_active True``; the
    source lifts as ``('VolumeSource', 'B', 'Hz')``. Hz under the EVEN mirror does
    not lift at all (odd parity on the plane), which is why the phase changes with
    the component.

    Serves ``TestHoleyWvgBands.test_fields_at_kx``: the single instance behind the
    CUDA folded-complex off-diagonal stencil weld.
    """
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Hz,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y, side=mp.High)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        symmetries=[mp.Mirror(mp.Y, phase=-1)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_folded_complex_nobloch_offdiag_2d(mp, res=10):
    """Complex storage with NO k_point, two mirrors, and a cylinder: bloch False.

    THE OTHER HALF OF THE SAME ROW. ``folded_complex_offdiag_2d`` drives the
    off-diagonal value at ``bloch`` True; the folded complex magnetic pair pins
    ``bloch`` True today, and ``examples:solve-cw.py`` and
    ``TestArrayMetadata.test_array_metadata`` are refused TWICE over — on bloch and
    on off-diagonal. Dropping either pin alone therefore measures nothing; this
    case is what makes dropping both an evidenced move rather than a widening.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (81, 81, 1), folded 'mirror plane on X, Y', complex_storage True, bloch False,
    k_point (0.0, 0.0, 0.0), beta 0.0, off_diagonal_epsilon True,
    susceptibilities 0, pml_active True``.

    THE SOURCE IS IN THE KEPT QUADRANT, at (+4, +4), for the reason
    ``folded_complex_kz2d_3d``'s docstring records: with two mirror planes the
    stored quadrant is the one BOTH planes keep, and a source at (-4, 0) makes the
    lift refuse by name.
    """
    cell = mp.Vector3(16, 16, 0)
    geometry = [mp.Cylinder(radius=1.2, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(4.0, 4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        force_complex_fields=True,
                        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(6.0, 0), size=mp.Vector3(0, 8)))
    return sim, [monitor], 50.0


def case_complex_beta_bloch_2d(mp, res=20):
    """A z beta AND an in-plane Bloch phase: the two together, which no case carried.

    WHAT THE LIFT PROVES, and it is the whole reason the case exists. The complex
    beta arms pin ``bloch`` False, and the refusal text on that row ("a z-only
    k_point lifts as beta, not as a Bloch phase") describes ``complex_beta_2d`` —
    which drove the same beta at ``k_x = 0`` and reads ``bloch False``. The corpus
    rows carry a z beta AND an in-plane phase, and are refused on the row that
    describes the case rather than them. With ``k_x`` nonzero beside the z
    component the grid reports ``beta 0.4`` AND ``bloch True``, so this case moves
    exactly one axis against the released one.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (120, 120, 1), complex_storage True, bloch True, k_point (0.25, 0.0, 0.0),
    beta 0.4, susceptibilities 0, pml_active True``. Serves
    ``TestSpecialKz.test_special_kz``.
    """
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.25, 0, 0.4), kz_2d="complex")
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_folded_complex_beta_bloch_2d(mp, res=20):
    """``complex_beta_bloch_2d`` with a mirror on Y: the folded corner of the same axis.

    ON THE TRITON TABLE it is the folded half of the ``bloch`` widening — the folded
    complex beta pairs carry their own rows and need their own case. ON THE CUDA
    TABLE it does more: that table has ONE complex-beta pair for both foldings, so
    this case (folded, bloch True) beside ``complex_beta_2d`` (unfolded, bloch
    False) drives BOTH values of BOTH axes, which is what lets those two rows be
    DELETED rather than widened. ``folded`` has only three spellings, so deleting
    the row is the only way to admit both foldings, and it is sound here precisely
    because both values were driven.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (120, 62, 1), folded 'mirror plane on Y', complex_storage True, bloch True,
    beta 0.4, susceptibilities 0, pml_active True``. The source at (-1.5, 0) sits
    ON the mirror plane, which the lift accepts — it is in no discarded half.

    Serves ``TestEigCoeffs.test_binary_grating_special_kz_{0_13_2, 1_17_7}`` and,
    on the CUDA table, ``TestSpecialKz.test_eigsrc_kz_0_complex`` and
    ``TestSpecialKz.test_special_kz`` as well.
    """
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.25, 0, 0.4), kz_2d="complex",
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_complex_no_pml_3d(mp, res=10):
    """A 3-D ABSORBER cell, complex storage, one Lorentz pole: conductive and PML-free.

    ``absorber_1d``'s combination at 3-D and under complex storage, which is the
    shape the four ``TestLoadDump`` 3-D rows carry (``chunk_layout_file``,
    ``chunk_layout_sim``, ``structure``, ``structure_sharded``) and the one the
    complex no-absorber products on both NVIDIA tables name. An ``mp.Absorber``
    rather than a PML is again what makes ``pml_active`` False and ``conductivity``
    True at once; the k_point is oblique so the Bloch phase is genuinely
    three-component.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 3, grid_shape
    (23, 21, 27), complex_storage True, bloch True, k_point (0.4, -1.3, 0.7),
    beta 0.0, conductivity True, off_diagonal_epsilon False, susceptibilities 1,
    pml_active False``.

    AN Hy SOURCE, so the driver's deposit lands on the B seam while the arm this
    case is evidence for takes the D seam — the two are then measured in one step
    rather than in two runs.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3, gamma=0.05, sigma=0.4)])
    cell = mp.Vector3(2.3, 2.1, 2.7)
    geometry = [mp.Block(mp.Vector3(mp.inf, 0.6, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Hy,  # noqa: SLF001
                         center=mp.Vector3(-0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.Absorber(0.2)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.4, -1.3, 0.7),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.6, 0, 0), size=mp.Vector3(0, 1.0, 1.0)))
    return sim, [monitor], 30.0


def case_complex_no_pml_offdiag(mp, res=25):
    """A bare periodic cell with a cylinder: complex, off-diagonal, NO boundary layer.

    WHY IT IS BUILT HERE AND DRIVEN NOWHERE YET. It is the case
    ``cuda:complex no-absorber off-diagonal stencil weld`` is waiting for, and that
    product's entry is seed-only — its gate block exists with its pins checked and
    only the ``fingerprints.json`` entry is missing. The builder is written and
    lifted now so the seed and the route row can land in one step whenever the
    owner takes that lever; it carries no DRIVE row until the release row it
    evidences exists, because a route row for an arm the release does not admit
    fails the gate on a case that is behaving correctly.

    No boundary layers AT ALL is where ``pml_active False`` comes from; the
    cylinder's curved surface writes the off-diagonal rows; and there is no
    susceptibility anywhere in the cell, so the sigma-reader refusal that blocks
    ``folded_offdiag_dispersive_2d`` does not apply.

    Measured on a lift of this builder, 2026-09-13: ``dimensions 2, grid_shape
    (25, 25, 1), complex_storage True, bloch True, k_point (0.3892, 0.1597, 0.0),
    beta 0.0, conductivity False, off_diagonal_epsilon True, susceptibilities 0,
    pml_active False``. Serves ``TestMaterialGrid.test_subpixel_smoothing``.

    NO MONITORS on a 1 x 1 cell, so the flux comparison and the second consult site
    are NOT-APPLICABLE here exactly as they are for ``material_dispersion_0d``.
    """
    cell = mp.Vector3(1, 1, 0)
    geometry = [mp.Cylinder(radius=0.301943, center=mp.Vector3(),
                            material=mp.Medium(index=3.5))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Hz,  # noqa: SLF001
                         center=mp.Vector3(0.2, 0.1))]
    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=sources,
                        resolution=res, k_point=mp.Vector3(0.3892, 0.1597, 0),
                        eps_averaging=True)
    return sim, [], 30.0


# THE CASE THAT IS NOT HERE, named so it is not re-derived next round.
# ``folded_offdiag_dispersive_2d`` — a folded 1-pole grid whose chi1inv carries
# off-diagonal rows — is the single case three levers of this round rest on (the
# Triton folded magnetic pair's susceptibility corner, the CUDA magnetic pair's
# off-diagonal corner at one pole and the whole release of
# ``cuda:dispersive off-diagonal polarization pair``, and the Metal folded B/H
# pair's corner). IT CANNOT BE BUILT ON A STOCK MEEP, measured here 2026-09-13:
# ``lift_simulation`` refuses because MEEP's chi1inv tensor is not diagonal at a
# lattice point and this MEEP carries no ``fields.get_susceptibility_sigma``, so the
# susceptibility sigma cannot be checked. Both escapes the refusal offers fail the
# case's purpose — ``eps_averaging=False`` lifts but reads ``off_diagonal_epsilon
# False``, which deletes the axis the case exists to drive, and moving the
# dispersion onto an axis-aligned block beside a NON-dispersive cylinder still
# refuses, because the check is not per-region. A MEEP built with
# ``MEEP_SIGMA_PATCH=1`` (parity/meep_gpu/build_meep_133_macos.sh, macOS only) is
# what the case needs, and no Linux patched build exists in this tree. The builder
# is specified in full in the round's case contract; it is deliberately NOT
# registered in ``CASES``, because an unliftable builder in a file every other route
# gate inherits its case list from is a trap rather than a placeholder.
#
# NARROWED 2026-09-27: "cannot be built on a stock MEEP" is true of the shape above
# -- a dispersive cylinder in a NON-dispersive background -- and not of every
# off-diagonal dispersive grid. When EVERY medium carries the same susceptibility the
# media differ in epsilon alone, no structured sigma has to be recovered, and the
# off-diagonal rows lift as permittivity: ``case_folded_offdiag_dispersive2_2d``
# lifts on MEEP 1.33.0 that way, at two poles (its docstring says why two). It is
# registered, and it is the CUDA dispersive off-diagonal single's route case. Not
# acted on here, and reported instead: the same construction at ONE pole is a
# candidate route case for the three levers above (it is not what they were
# specified on -- a uniform susceptibility rather than a dispersive cylinder in
# vacuum -- so whether it evidences their rows is a decision, not an inference).


CASES: Dict[str, Any] = dict(e2e.CASES)
CASES["folded_complex_2d"] = case_folded_complex_2d
CASES["pml_3d_diagonal"] = case_pml_3d_diagonal
CASES["magnetic_seam_2d"] = case_magnetic_seam_2d
CASES["pml_1d"] = case_pml_1d
CASES["folded_dispersive_2d"] = case_folded_dispersive_2d
CASES["cylindrical_m0"] = case_cylindrical_m0
for _name, _builder in (("cylindrical_m0_complex", case_cylindrical_m0_complex),
                        ("cylindrical_mneg1", case_cylindrical_mneg1),
                        ("cylindrical_m1", case_cylindrical_m1),
                        ("cylindrical_m2", case_cylindrical_m2),
                        ("cylindrical_m3", case_cylindrical_m3),
                        ("offdiag_2d", case_offdiag_2d),
                        ("offdiag_magnetic_2d", case_offdiag_magnetic_2d),
                        ("folded_offdiag_magnetic_2d", case_folded_offdiag_magnetic_2d),
                        ("complex_beta_2d", case_complex_beta_2d),
                        ("folded_special_kz_2d", case_folded_special_kz_2d),
                        ("folded_complex_beta_2d", case_folded_complex_beta_2d),
                        ("bfast_1d", case_bfast_1d),
                        ("nonlinear_1d", case_nonlinear_1d),
                        ("nonlinear_3d", case_nonlinear_3d),
                        ("complex_nobloch_2d", case_complex_nobloch_2d),
                        # THE TARGET ROUND (2026-09-13). Real storage first, complex
                        # Cartesian next; no Dcyl case is added, and nothing is
                        # inserted after the Dcyl complex rows of either DRIVE table.
                        ("folded_3d", case_folded_3d),
                        ("dispersive5_2d", case_dispersive5_2d),
                        ("dispersive6_2d", case_dispersive6_2d),
                        ("folded_dispersive5_2d", case_folded_dispersive5_2d),
                        ("absorber_1d", case_absorber_1d),
                        ("material_dispersion_0d", case_material_dispersion_0d),
                        # INSERTED BESIDE THE CASE IT REPLACES, inside the
                        # real-storage block: the comment above this tuple says
                        # nothing is inserted after the Dcyl complex rows, and
                        # ``pml_2d`` must stay FIRST (the subnormal policy installs
                        # lazily, so the first case lifted immunises the process).
                        ("no_pml_dispersive_2d", case_no_pml_dispersive_2d),
                        # 2026-09-27, the route case of the CUDA dispersive
                        # off-diagonal update_E single: real storage, so it goes
                        # here, before the complex Cartesian block.
                        ("folded_offdiag_dispersive2_2d",
                         case_folded_offdiag_dispersive2_2d),
                        ("complex_1d", case_complex_1d),
                        ("complex_3d_thinline", case_complex_3d_thinline),
                        ("complex_3d", case_complex_3d),
                        ("folded_complex_3d", case_folded_complex_3d),
                        ("folded_complex_kz2d_3d", case_folded_complex_kz2d_3d),
                        ("folded_complex_offdiag_2d", case_folded_complex_offdiag_2d),
                        ("folded_complex_offdiag_hz_2d",
                         case_folded_complex_offdiag_hz_2d),
                        ("folded_complex_nobloch_offdiag_2d",
                         case_folded_complex_nobloch_offdiag_2d),
                        ("complex_beta_bloch_2d", case_complex_beta_bloch_2d),
                        ("folded_complex_beta_bloch_2d",
                         case_folded_complex_beta_bloch_2d),
                        ("complex_no_pml_3d", case_complex_no_pml_3d),
                        ("complex_no_pml_offdiag", case_complex_no_pml_offdiag)):
    CASES[_name] = _builder
CASE_INTENT: Dict[str, str] = dict(e2e.CASE_INTENT)
CASE_INTENT["magnetic_seam_2d"] = (
    "the deposit repair on the B seam: a magnetic source puts the driver's "
    "injection between step_B and update_H, the one installed pair shape no "
    "release gate had driven")
CASE_INTENT["pml_3d_diagonal"] = (
    "the ordinary pairs at THREE DIMENSIONS on a diagonal medium: the 3 of their "
    "dimensions row, byte-compared on every table. It was this gate's Triton "
    "envelope control until 2026-09-13, when that row was widened to admit it and "
    "the witness moved to folded_dispersive5_2d")
CASE_INTENT["folded_complex_2d"] = (
    "a folded grid whose two mirror-fill arms both refuse, so the completeness "
    "rung passes and the next rung is the one measured")
CASE_INTENT["pml_1d"] = (
    "the ordinary pairs on ONE live axis: the shape FUSED_RELEASE_ENVELOPE's "
    "dimensions row refused on four corpus seam-instances, and the case that "
    "widens it")
CASE_INTENT["folded_dispersive_2d"] = (
    "a fold with a live update_P, whose two seams go to different families: the "
    "folded magnetic pair on B and the folded DISPERSIVE product on D, with fused "
    "pair D (folded) refused beside them on susceptibilities. On the CUDA table it "
    "is the three-slot weld beside the magnetic pair. It was dropped from the "
    "Triton DRIVE table on 2026-09-13 for a second-consult divergence read with "
    "fusion OFF, and is back because that divergence was re-attributed to the lazy "
    "subnormal-policy install and the seam itself measured clean")
CASE_INTENT["cylindrical_m0"] = (
    "the two cylindrical_real pairs on a Dcyl m = 0 grid, the shape the shared "
    "envelope refused; the monitor makes the observable comparison and the second "
    "consult site non-vacuous")
CASE_INTENT.update({
    "cylindrical_m0_complex": "the cylindrical COMPLEX pairs at m = 0 (complex storage forced)",
    "cylindrical_mneg1": "the cylindrical complex pairs at m = -1",
    "cylindrical_m1": "the cylindrical complex pairs at m = 1",
    "cylindrical_m2": "the cylindrical complex pairs at m = 2",
    "cylindrical_m3": "the cylindrical complex pairs at m = 3",
    "offdiag_2d": ("the ordinary magnetic pair on a 2-D off-diagonal grid (Ez source): "
                   "one point of its off-diagonal corner"),
    "offdiag_magnetic_2d": ("the off-diagonal scratch-output weld on the D seam (Hz "
                            "source) beside the ordinary magnetic pair on the B seam"),
    "folded_offdiag_magnetic_2d": ("the folded off-diagonal scratch-output weld beside "
                                   "the folded magnetic pair, with the Triton fills"),
    "complex_beta_2d": "the complex beta pairs on the special_kz cell stored complex",
    "folded_special_kz_2d": "the folded real beta pairs (Triton) on the special_kz cell with a mirror",
    "folded_complex_beta_2d": "the folded complex beta pairs (Triton) on the complex beta cell with a mirror",
    "bfast_1d": "the BFAST pairs on refl-angular's real-storage broadband-angle shape",
    "nonlinear_1d": ("the Triton nonlinear B pair on 3rd-harm-1d's shape (phase B); "
                     "on the CUDA table the magnetic pair's row refuses it"),
    "nonlinear_3d": ("the CUDA envelope control: both tables refuse a 3-D nonlinear "
                     "grid on their own rows, and the magnetic pair's predicate admits "
                     "the opt-in; driven only by envelope_leg"),
    "complex_nobloch_2d": ("the complex pairs under force_complex_fields with no "
                           "k_point: bloch False, the case that frees their bloch row"),
    # THE TARGET ROUND (2026-09-13). One sentence per key, saying what the case is
    # chosen to exercise, so a case that stops exercising it is visible in the
    # artifact rather than silently redundant.
    "folded_3d": ("both folded pairs on a THIRD dimension: the axis-aligned block "
                  "keeps the tensor diagonal so dimensions is the only axis moving"),
    "dispersive5_2d": ("the pole COUNT at five on an unfolded 2-D grid: the value "
                       "four TestLoadDump rows carry and no case had driven"),
    "dispersive6_2d": ("the same shape at six poles, which is the stochastic_emitter "
                       "rows' count; the two differ in the count alone"),
    "folded_dispersive5_2d": ("THE TRITON ENVELOPE WITNESS: a fold at five poles, "
                              "which both folded pairs' susceptibilities rows refuse "
                              "while their predicates admit it. Driven only by "
                              "envelope_leg on the Triton legs; a Metal DRIVE case"),
    "absorber_1d": ("pml_active False WITH conductivity True, at one dimension and "
                       "five poles: an Absorber is a conductivity, not a split-field "
                       "PML, and no released case carried that combination"),
    "material_dispersion_0d": ("the conductivity-False, dimensions-2 half of the same "
                               "no-absorber rows, on a zero-extent cell; no monitor, "
                               "so the flux comparison and the second consult site "
                               "are NOT-APPLICABLE rather than passing"),
    "no_pml_dispersive_2d": ("the same conductivity-False, dimensions-2 half of the "
                             "no-absorber rows on a cell with REAL EXTENT, so the "
                             "four controls that are vacuous on a one-cell grid are "
                             "measurements here — it carries a monitor and the "
                             "in-seam mutation's wall plane is not the deposit cell"),
    "complex_1d": "the complex pairs at ONE declared dimension under a z Bloch phase",
    "complex_3d_thinline": ("the complex pairs on a (0, 0, L) cell DECLARED 3-D — the "
                            "shape four physically-1-D-looking corpus rows answer"),
    "complex_3d": ("the complex pairs on a GENUINE 3-D grid, so the widened dimensions "
                   "row is not licensed by degenerate (1, 1, N) cells alone"),
    "folded_complex_3d": ("the folded complex pairs at 3-D on the triangular-lattice "
                          "oblique cell, which is the corpus row's own shape"),
    "folded_complex_kz2d_3d": ("the folded complex pairs' second 3-D corner: a zero-z "
                               "cell MEEP carries as 3-D because kz_2d='3d'"),
    "folded_complex_offdiag_2d": ("an off-diagonal fold under complex storage with an "
                                  "ELECTRIC point source: the folded complex magnetic "
                                  "pair's off-diagonal value on both device tables, "
                                  "beside the certified `complex folded off-diagonal` "
                                  "update_E arm on the Triton one; the CUDA stencil "
                                  "weld refuses this seam by design"),
    "folded_complex_offdiag_hz_2d": ("the same fold with the corpus row's MAGNETIC "
                                     "source under an odd mirror, so the D seam is "
                                     "empty and the CUDA folded complex off-diagonal "
                                     "stencil weld admits beside the magnetic pair"),
    "folded_complex_nobloch_offdiag_2d": ("the bloch-False value of the same fold, on a "
                                          "grid that is off-diagonal too; on the "
                                          "Triton table the magnetic pair's bloch-False "
                                          "value, with the same update_E arm beside it"),
    # THE THIRD OF THE 2026-09-17 OFF-DIAGONAL ROWS, and the one whose ABSENCE from
    # this table cost the CUDA route campaign a case. ``run_case`` reads
    # ``CASE_INTENT[case]`` while building its row -- two lines in, BEFORE the
    # ``needs_probe`` clause that would have skipped this case on a leg carrying no
    # licence -- so a DRIVE row with no intent here raises ``KeyError`` at 0.0 s and
    # is recorded as ERROR, which is indistinguishable in a summary from a product
    # that refused to dispatch. Its two siblings above were given intents when they
    # were added; this one was not, and nothing checked.
    "complex_no_pml_offdiag": ("the off-diagonal value of the complex no-absorber "
                               "rows: complex storage and an off-diagonal chi1inv on "
                               "a grid with no PML at all, which is the only shape "
                               "the `cuda:complex no-absorber off-diagonal stencil "
                               "weld` is released for"),
    "complex_beta_bloch_2d": ("a z beta AND an in-plane Bloch phase at once, which the "
                              "released complex-beta case (k_x = 0) cannot carry"),
    "folded_complex_beta_bloch_2d": ("the folded corner of the same pair of axes; on "
                                     "the CUDA table, where one arm serves both "
                                     "foldings, it is what lets the folded row go"),
    "complex_no_pml_3d": ("a complex, conductive, PML-free 3-D grid at one pole: the "
                          "shape the four TestLoadDump 3-D rows carry"),
    "folded_offdiag_dispersive2_2d": (
        "a folded two-pole grid whose chi1inv carries off-diagonal rows: the "
        "hand-CUDA dispersive off-diagonal update_E single beside the PML step_D "
        "single and the mirror-fill twins, with no fused product released on either "
        "NVIDIA table"),
})


#: The rungs of ``fastpath._decide`` that can refuse a FOLDED configuration, and
#: the phrases that identify each. Read off the shipped refusal text rather than
#: predicted: the campaign's first run expected the fold rung on ``folded_2d`` and
#: got the completeness rung, because the mirror-fill arms WON there and filled two
#: slots outside ``DRIVER_SLOTS``. Naming which rung answered is the fact worth
#: recording — a refusal that silently moved from one rung to another is a change
#: in what is refused, and the artifact must show it rather than absorb it.
FOLD_REFUSAL_RUNGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("6-completeness", ("the composer filled",
                        "which the driver seam has no consult for")),
    ("6b-fold", ("mirror-folded", "fill sub-steps")),
    # RE-READ 2026-08-29 OFF THE SHIPPED TEXT. This needle used to be "no fused arm
    # has been driven through the driver seam" — the sentence clause (8) carried
    # while the refusal was unconditional. The release narrowed it to "it has not
    # been driven through the driver seam ON THIS CONFIGURATION", so the old needle
    # matches nothing and this classifier would have answered ``None`` — i.e.
    # "REFUSAL-CHANGED" — on the one rung the release actually moved. A classifier
    # that stops recognising the rung it is watching is the same failure as a leg
    # that pins a pre-flip answer, so it is re-read rather than widened: both
    # needles below are present-tense substrings of ``fastpath``'s current text.
    ("8-fused-arm-not-admitted", ("a fused cross-sub-step product won",
                                  "has not been driven through the driver seam")),
    ("8b-subnormal-policy", ("float32 subnormal policy",)),
    ("9-no-slot-carries-a-kernel", ("no slot is left carrying a kernel",)),
    # THE THREE RUNGS THE SECOND NVIDIA TABLE ADDED, read off fastpath's current
    # text the same way every needle above was. A classifier that stops recognising
    # the rung it watches reports REFUSAL-CHANGED rather than the narrowing, which
    # is the failure this table has already paid for once.
    ("4d-preference-names-no-candidate", ("names no candidate table",)),
    ("4-no-candidate-table", ("no kernel table is a candidate",)),
    ("3-preference-outside-the-hardware", ("names no kernel table this host "
                                           "can run",)),
    # THE PENDING-GATE RUNG, read off the text it returned on 2026-09-15 for
    # folded_complex_offdiag_2d: a selected arm (or a fused label's constituent) in
    # ``fastpath.PENDING_DEVICE_GATE_ARMS`` refuses the plan with that map's own
    # reason. The needle is the job-2365/2366 wording; a pending reason worded
    # otherwise still reads REFUSAL-CHANGED. NO TRITON DRIVE CASE REACHES THIS RUNG
    # after the 2026-09-15 ruling: the two off-diagonal folds that did
    # (dispatch_fused_route_2026-09-15_offdiag, refusing_rung pending-device-gate on
    # both) dispatch now that `complex folded off-diagonal` is certified. The needle
    # is kept because both sub-step arms still pending carry the same wording, and
    # test_substitution_expectation pins that it classifies every one of them.
    ("pending-device-gate", ("recertification and tracked fingerprint entry "
                             "are pending",)),
)


def classify_refusal(reason: str) -> Optional[str]:
    for name, needles in FOLD_REFUSAL_RUNGS:
        if all(needle in reason for needle in needles):
            return name
    return None


# ---------------------------------------------------------------------------
# What to drive, per case
# ---------------------------------------------------------------------------

#: Which fused arms each case is asked for, and what the ladder must find.
#:
#: MEASURED, NOT GUESSED. Every entry was read on 2026-08-29 by calling
#: ``triton_kernels.plan_step(fields, pml, fuse=True, fuse_ade=True, sources=...)``
#: on a real lifted driver for that case and reading ``selected``/``reasons``. On a
#: NumPy host the predicates decide and only the BUILDERS refuse ("array module is
#: 'numpy', not cupy"), so admission is exactly measurable off the device and the
#: arm set below is what the composer WILL select once the builders can run.
#:
#: ``arms`` IS EVERY FUSED LABEL THE COMPOSER SELECTS, not a preference. Clause (8)
#: refuses the whole plan when ANY selected fused label is neither opted into NOR
#: released for the configuration, so a partial admission does not dispatch the
#: part — it refuses everything.
#:
#: ``reached_by`` NAMES THE ROUTE, and it is the 2026-08-29 change. The EIGHT
#: dispatching cases run with ``MEEP_GPU_FUSE_ARMS`` UNSET, because that is what
#: SHIPS: ``fastpath.RELEASED_FUSED_ARMS`` admits them, narrowed per arm by
#: ``FUSED_RELEASE_ARM_AXES``, and the switch is not involved. That now includes the
#: two FOLDED cases, which is the 2026-09-02 change — the folded pairs are released
#: on a fold and refused off one, so the release is the route there too.
#: ``folded_complex_2d`` keeps the opt-in: complex storage is a SHARED envelope row
#: and no folded arm is released on it, so the switch really is the only way to
#: reach the refusal that case exists to record.
#:
#: THE PARTIAL-ADMISSION REFUSAL MOVED WITH IT, rather than being dropped. It used
#: to be measured here by a ``withhold`` entry naming a strict subset of the
#: selected arms — which the release makes unreachable on these cases, since the
#: un-named arm is now admitted anyway. It is measured instead by
#: :func:`envelope_leg`, on whichever case :data:`ENVELOPE_ROWS` names for the
#: backend in hand — ``folded_dispersive5_2d`` on the Triton table since
#: 2026-09-13, where the release genuinely does not apply — so the refusal is
#: reached without holding any product constant at a value it does not ship with.
#: That leg is also the only device evidence that the ENVELOPE refuses at all, and
#: the case it names MOVES whenever a round widens the row it rested on: it named
#: ``pml_3d`` until the off-diagonal axis became a per-arm corner, then
#: ``pml_3d_diagonal`` until this round widened ``dimensions``.
#:
#: ``pairs`` is how many launches the fusion is expected to SUBSTITUTE per step,
#: which is what the launch-count drop is compared against.
DRIVE: Dict[str, Dict[str, Any]] = {
    "pml_2d": {
        "arms": ["fused pair B", "fused pair D"],
        "pairs": 2,
        "reached_by": "release",   # the SHIPPED route: MEEP_GPU_FUSE_ARMS unset
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("the ordinary unfolded real-float32 PML pair on BOTH sides. The "
                "B seam is CLEAN (no magnetic source) so update_H holds a NoopPlan; "
                "the D seam carries the electric Gaussian, so step_D holds a "
                "LeadingRepairPlan and update_E the TrailingRepairPlan — both "
                "sentinel shapes in one case."),
    },
    "magnetic_seam_2d": {
        "arms": ["fused pair B", "fused pair D"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("THE B-SEAM DEPOSIT REPAIR, which no release gate had driven. The "
                "magnetic Gaussian is injected between step_B and update_H, so "
                "step_B holds a LeadingRepairPlan and update_H the "
                "TrailingRepairPlan; the D seam is clean here, so step_D/update_E "
                "are the sentinel pair. Exactly the mirror of pml_2d, and the "
                "reason the pair is in the set: the veto campaign's withheld "
                "control reported leading_repair_slots_left_running "
                "{'step_D': 12} on every leg and never once step_B, while the "
                "shipped composition runs the B-seam repair on 11 corpus "
                "seam-instances."),
    },
    "conductive_2d": {
        "arms": ["fused pair B", "fused pair D (conductive)"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("BOTH SEAMS, and what changed on 2026-09-11 is the admission and "
                "not the composition. The D side has never been the ordinary pair "
                "here — a D_conductivity is installed, so that curl belongs to the "
                "conductive-PML product and ``conductive_fused_electric_pair`` is "
                "what wins the seam. Until this round its label had no ledger entry, "
                "so ``plan_step``'s ``fuse_labels`` offer withheld it and the record "
                "carried it under ``fusion.not_installed`` as 'offered to run'; the "
                "step was MIXED for that reason alone. With the product released the "
                "case drives the conductive electric pair beside ``fused pair B`` "
                "and the substitution moves from 4 -> 3 launches per step to 4 -> 2."),
    },
    "dispersive_2d": {
        "arms": ["fused pair B", "dispersive fused pair"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("a second fused FAMILY beside the ordinary pair, and the one case "
                "whose update_P slot also dispatches, so the fused pair sits in a "
                "step that still has a live fifth consult."),
    },
    "pml_1d": {
        "arms": ["fused pair B", "fused pair D"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("ONE LIVE AXIS, and it is the axis the envelope was widened on. "
                "``dimensions`` was pinned at 2 and refused four corpus "
                "seam-instances on that row alone; measured before the widening "
                "landed, the composer selects the SAME two ordinary pairs here "
                "that it selects in 2-D. Electric source, so the seams split the "
                "way pml_2d's do and the launch arithmetic is comparable."),
    },
    "folded_2d": {
        "arms": ["fused pair B (folded)", "fused pair D (folded)"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "fills": ["fill_B", "fill_D"],
        "why": ("THE REFUSAL THAT NO LONGER STANDS, and the composition three "
                "campaigns recorded as a refusal without ever running it. Until "
                "the 2026-09-02 fill consults, the mirror-fill arms won fill_B and "
                "fill_D — slots outside DRIVER_SLOTS — and the COMPLETENESS rung "
                "(6) refused the whole plan before the fold rung was reached, so "
                "``fused pair B (folded)`` had been SELECTED and ADMITTED and never "
                "once driven. With the fills consulted the seam runs six of seven "
                "slots: both folded pairs absorbing four, and the mirror-fill "
                "kernel in both fill slots, which is what rung (6b) still requires "
                "of a fold. A folded case that does not carry a fill IN-LAUNCH "
                "measures nothing about this composition, which is why the fill "
                "slots are named in the evidence rather than assumed."),
    },
    # THE ROW THAT LEFT THIS TABLE ON 2026-09-13 AND CAME BACK ON THE
    # RE-ATTRIBUTION ROUND, and it came back because the reason it left was a
    # MISATTRIBUTION rather than because a defect was fixed. It left because this
    # case's SECOND consult site was read, on 2026-09-11, as diverging from the
    # array path WITH FUSION OFF as well as on — a case that diverges in its own
    # control cannot be release evidence for any arm — which also pinned ``fused
    # pair B (folded)``'s ``susceptibilities`` row at 0 for want of the evidence
    # this case carries.
    #
    # THE SEAM WAS THEN MEASURED AND IT IS CLEAN. Driving this gate's own
    # ``run_case`` on the GPU host with ``fused pair D (folded dispersive)`` admitted
    # in-process at 7 of 7 slots, over the full 1600-step ladder, three times in
    # fresh processes: PASS-FUSED every time, ``first_divergent_checkpoint`` null, 0
    # of 41 arrays differing over 200,080 words on BOTH ``fused_vs_array`` and
    # ``unfused_vs_array``, the second consult site PASS with
    # ``state_entering_the_site`` {true, true}, substitution 7.00 against 9.00
    # launches per step. The armed nulls fired on the same runs, so it is not a
    # degenerate pass, and across every artifact on both machines 1,606 recorded
    # second-consult verdicts read PASS with none failing — 86 of them this case's,
    # over 23 runs and all three backends.
    #
    # WHAT THE SIGNATURE ACTUALLY WAS: the subnormal policy installs LAZILY, at the
    # first dispatch's plan freeze, so CuPy kernels compiled earlier keep CuPy's own
    # ``-ftz=true`` while the Triton kernels do not. The ARRAY comparison leg
    # flushes subnormals that both dispatch legs keep, which is precisely why both
    # legs appeared to diverge from it by identical amounts and why no mechanism
    # could be found (this arm spans ``step_D``/``update_E`` while
    # ``synchronize_magnetic_fields`` consults only the magnetic slots). Controlled
    # A/B, same case list and same arm, one flag: the shipped ordering diverges at
    # checkpoint 48 and installing the policy first passes. The engine-side fix is a
    # separate round; every campaign in the meantime runs with the policy installed
    # first, which is what the harness wrappers' minted ``keep`` cache does.
    #
    # WHAT THIS ROW STILL OWES, and it is why the ``why`` below claims no release
    # result: all three of those drives reached the arm by OPT-IN, because no
    # release row admitted it, so the release-route control read NOT-APPLICABLE on
    # every one. This row plus ``fastpath.RELEASED_FUSED_ARMS`` is what makes the
    # shipped route reachable, and THIS campaign is the run that drives it with
    # ``MEEP_GPU_FUSE_ARMS`` unset. ``folded_dispersive5_2d``, the same fold at five
    # poles, stays the ENVELOPE WITNESS and is not a DRIVE row: both folded
    # susceptibility rows refuse it (measured on a NumPy lift after the release rows
    # landed — ``released_fused_arms`` is empty there and each refusal names the
    # axis), and driving it would widen them and destroy the witness.
    "folded_dispersive_2d": {
        "arms": ["fused pair B (folded)", "fused pair D (folded dispersive)"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "fills": ["fill_B", "fill_D"],
        "why": ("A FOLD WITH A LIVE ``update_P``, and the only case on this table "
                "where the two folded seams go to DIFFERENT families: the B seam to "
                "``fused pair B (folded)`` and the D seam to the folded DISPERSIVE "
                "product, because a pole is what makes the constitutive arm "
                "dispersive. ``fused pair D (folded)`` is refused here by its own "
                "``susceptibilities`` row and that refusal is half of what the case "
                "measures. Both mirror fills run in-launch, as rung (6b) requires "
                "of a fold. The arm's seam evidence is banked — 7 of 7 slots, 1600 "
                "steps, 0 of 41 arrays differing, second consult site PASS, three "
                "fresh processes — but every one of those drives reached it through "
                "``MEEP_GPU_FUSE_ARMS``, so what this campaign adds, and what no "
                "measurement carries yet, is the same composition reached by the "
                "RELEASE with the switch unset."),
    },
    "cylindrical_m0": {
        "arms": ["fused pair B (cylindrical)", "fused pair D (cylindrical)"],
        "pairs": 2,
        "reached_by": "release",
        "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("THE Dcyl m = 0 SHAPE, and the first cylindrical configuration any "
                "release gate has driven — which is why ``cylindrical`` sat in the "
                "SHARED envelope at False until this round and is a per-arm axis "
                "after it. ``launch.py``'s ``has_cylindrical`` branch refuses the "
                "ordinary and folded pairs here BY NAME, and the certified-product "
                "loop installs both ``cylindrical_real`` pairs: measured on a NumPy "
                "lift 2026-09-10, exactly those two are admitted and nothing else "
                "fused. The Ez source puts the deposit on the D seam, so that seam "
                "carries the Leading/Trailing repair pair and the B seam the "
                "``NoopPlan`` sentinel — the same split as pml_2d, and the reason "
                "the launch arithmetic is comparable with it."),
    },
    "folded_complex_2d": {
        "arms": ["fused pair B (folded complex)", "fused pair D (folded complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLD WHERE, UNTIL PHASE B, NO FUSED PRODUCT COULD RUN: "
                "``complex_storage`` was a shared envelope row, so the case's job "
                "was to record that both mirror-fill arms refuse without the "
                "expansion licence (rung 6b) and that with it six of seven slots "
                "dispatch as SEPARATE arms. Phase B (2026-09-13) released the "
                "folded complex pairs on their seeded welds and moved the axis "
                "per-arm; measured on the GPU host, both folded complex pairs install "
                "with the complex fills beside them, byte-identical. Without the "
                "licence the case is PASS-NOT-THIS-LEG, exactly as bloch_2d is on "
                "the CUDA table."),
    },
    # ------------------------------------------------------------------
    # PHASE B (2026-09-13): the fifteen labels the 2026-09-12 identity fleet seeded.
    # Every row was driven through this gate's own run_case on the GPU host before it
    # was typed. Real cases first, complex Cartesian next, the Dcyl complex cases
    # LAST (a Dcyl complex case stepped earlier in the process poisons later
    # real-storage cases; see fastpath_cuda's release table for the measurement).
    # ------------------------------------------------------------------
    "special_kz_2d": {
        "arms": ["fused pair B (real beta)", "fused pair D (real beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped",
        "why": "the real beta pairs on the special_kz cell (kz_2d='real/imag')",
    },
    "folded_special_kz_2d": {
        "arms": ["fused pair B (folded real beta)", "fused pair D (folded real beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped", "fills": ["fill_B", "fill_D"],
        "why": "the folded real beta pairs, with the mirror fills beside them",
    },
    "bfast_1d": {
        "arms": ["fused pair B (BFAST)", "fused pair D (BFAST)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped",
        "why": "the BFAST pairs on refl-angular's real-storage broadband-angle shape",
    },
    "nonlinear_1d": {
        "arms": ["fused pair B (nonlinear)"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "shipped",
        "why": ("the nonlinear B pair on 3rd-harm-1d's shape; the D seam keeps the "
                "nonlinear constitutive single"),
    },
    # ------------------------------------------------------------------
    # THE TARGET ROUND (2026-09-13), REAL STORAGE. Every ``arms`` list below was
    # read off ``launch.plan_step`` on a NumPy lift of the case named, with the
    # composer asked ``fuse=True, fuse_ade=True`` and NO label restriction: the
    # products listed are the ones whose every refusal is a HOST clause (the array
    # module, and on the complex block the unreadable subnormal policy), which is
    # this table's standing rule for "what the composer WILL select once the
    # builders can run". THE METHOD WAS CALIBRATED BEFORE IT WAS USED: run against
    # the fifteen rows already in this table it reproduces each one's ``arms``
    # exactly, including the two that name one arm rather than two.
    #
    # WHAT IS STILL OWED PER ROW, and it is why no reason below claims a byte
    # result: the ladder, the substitution proof and the second consult site are
    # the campaign's, and until that run lands each of these rows is a prediction
    # about selection plus a release row widened on a lift.
    # ------------------------------------------------------------------
    "pml_3d_diagonal": {
        "arms": ["fused pair B", "fused pair D"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped",
        "why": ("THE 3-D POINT OF THE ORDINARY PAIRS' ``dimensions`` ROW, and a "
                "case this gate has owned since 2026-09-02 while driving it only as "
                "the envelope control. The row moves frozenset({1, 2}) -> "
                "frozenset({1, 2, 3}) and this is the case that evidences the 3; "
                "``examples:differential_cross_section.py`` is the corpus row behind "
                "it, and the CUDA table has driven this exact shape since "
                "2026-09-13. Arm set read off plan_step on a NumPy lift 2026-09-13: "
                "both ordinary pairs, refused there on the array module alone. "
                "THE PROMOTION COSTS THE ENVELOPE ITS WITNESS — a case the release "
                "ADMITS cannot also be the case it is shown declining — which is why "
                "ENVELOPE_ROWS['triton'] moves to folded_dispersive5_2d in the same "
                "change. Do not split the two edits."),
    },
    "pml_3d": {
        "arms": ["fused pair B"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "shipped",
        "why": ("THE 3-D x OFF-DIAGONAL CONJUNCTION, and the case without which the "
                "off-diagonal widening would be credited on evidence no run carries. "
                "This table has NO off-diagonal corner table (the CUDA and Metal "
                "tables do), so a row widened on 2-D off-diagonal evidence plus a "
                "separate 3-D diagonal case spells no conjunction and would admit "
                "``examples:grating2d_triangular_lattice.py``'s 3-D off-diagonal grid "
                "on evidence from two other shapes. The sphere IS that conjunction "
                "and it lifts today. Measured on a NumPy lift 2026-09-13, after the "
                "composer's off-diagonal guard was narrowed to the ``update_E`` seam: "
                "``fused pair B`` is refused on the array module alone, and "
                "``fused pair D`` is refused BY NAME — 'a specialized family owns "
                "this grid's sub-steps' — which is the per-pair split the narrowing "
                "claims and the half the risk register requires in the artifact. The "
                "D seam keeps the incumbent's offdiag singles."),
    },
    "offdiag_2d": {
        "arms": ["fused pair B"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "shipped",
        "why": ("THE SINGLE-AXIS ISOLATION FOR THE GUARD NARROWING: ``pml_2d`` with "
                "its slab replaced by a cylinder, so the two run shapes differ in "
                "``off_diagonal_epsilon`` alone and what the case measures is one "
                "axis rather than a new cell. With an Ez source the live E is out of "
                "plane, so the D seam keeps the incumbent's ``offdiag`` singles and "
                "this is a ONE-seam row. Measured on a NumPy lift 2026-09-13 under "
                "the narrowed guard: ``fused pair B`` refused on the array module "
                "alone, ``fused pair D`` refused by name. Neither could be measured "
                "at all before the guard edit — it is what refused them — so this row "
                "could not have been typed in an earlier round."),
    },
    "offdiag_magnetic_2d": {
        "arms": ["fused pair B", "fused pair D (off-diagonal)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped",
        "why": ("THE CONTENDED SEAM, SERVED ON BOTH SIDES SINCE 2026-09-15. An Hz "
                "source keeps the in-plane E (Ex, Ey) the off-diagonal rows couple "
                "LIVE and deposits on the B seam, so the D seam is the scratch-output "
                "weld's: ``fused pair D (off-diagonal)`` (the "
                "``offdiag_fused_electric_pair`` product) absorbs step_D/update_E "
                "beside ``fused pair B``. MEASURED before the weld was routed, on "
                "``dispatch_fused_route_2026-09-15_offdiag``'s shipped leg: "
                "PASS-FUSED, substitution EXACT, 4.0 -> 3.0 launches per step over "
                "2399 steps with the B pair alone fused. With the weld the expected "
                "drop is 2 (4.0 -> 2.0), the drop ``cuda:off-diagonal stencil weld`` "
                "measured beside the CUDA magnetic pair on this case "
                "(``dispatch_fused_route_cuda_2026-09-15_weldgrid``, EXACT over 2399 "
                "steps). The ORDINARY ``fused pair D`` is still refused by name here "
                "('a specialized family owns this grid's sub-steps'), and so is "
                "``fused_pair_dispersive_D`` ('specialized-family dispersive D/E "
                "fusion is not implemented'); both refusals must still appear by name "
                "in the record, because a refusal that goes silent is "
                "indistinguishable from one that stopped happening. Drive this beside "
                "offdiag_2d, whose Ez source is the in-seam electric deposit this "
                "weld's predicate refuses by name, never instead of it."),
    },
    "folded_offdiag_magnetic_2d": {
        "arms": ["fused pair B (folded)", "fused pair D (folded off-diagonal)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped", "fills": ["fill_B", "fill_D"],
        "why": ("THE FOLDED OFF-DIAGONAL CASE, SERVED ON BOTH SEAMS SINCE 2026-09-15. "
                "``launch.py``'s fold branch runs BEFORE the specialized-family "
                "guard, so the folded magnetic pair takes the B seam, and ``fused "
                "pair D (folded off-diagonal)`` (the "
                "``folded_offdiag_fused_electric_pair`` scratch-output weld) absorbs "
                "step_D/update_E with fill_symmetry_bc_D and zero_metal_D carried "
                "inline by its closed form. Both fill slots still carry the "
                "mirror-fill kernel, which rung 6b requires of a fold; the fill_D "
                "plan wraps its D pointers at build time, so on alternate steps it "
                "fills the weld's twin, which the next launch rewrites whole. "
                "MEASURED before the weld was routed, on "
                "``dispatch_fused_route_2026-09-15_offdiag``'s shipped leg: "
                "PASS-FUSED, EXACT, 6.0 -> 5.0 launches per step over 1999 steps "
                "with the B pair alone fused. With the weld the expected drop is 2 "
                "(6.0 -> 4.0), the drop ``cuda:folded off-diagonal stencil weld`` "
                "measured beside Triton's mirror fill on this case "
                "(``dispatch_fused_route_cuda_2026-09-15_weldgrid``, EXACT over 1999 "
                "steps). This case folds on Y alone; the XY folds the release row "
                "admits rest on the product gate's fold_xy_mixed_walled fixture, and "
                "that row's reason names them."),
    },
    "folded_3d": {
        "arms": ["fused pair B (folded)", "fused pair D (folded)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped", "fills": ["fill_B", "fill_D"],
        "why": ("A FOLD AT THREE DIMENSIONS, the axis both folded pairs pin at 2 "
                "while their only case is 2-D. Ten corpus seam-instances are refused "
                "on that row alone. Lifted off-device 2026-09-13 to dimensions 3, "
                "folded 'mirror plane on Y', grid (40, 21, 40), diagonal — the "
                "axis-aligned block is what keeps the tensor diagonal, so the case "
                "moves ``dimensions`` and nothing else. Arm set read off plan_step on "
                "that lift: both folded pairs, refused on the array module alone, "
                "with both fill slots host-refused too, so the mirror fill runs "
                "in-launch as rung 6b requires. What the campaign still owes is that "
                "the fill covers both slots IN 3-D, which no gate has driven."),
    },
    "absorber_1d": {
        "arms": ["fused pair D (no-PML stored E)"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "leg": "shipped",
        "why": ("THE FIRST CASE FOR AN ARM THAT HAS NEVER BEEN DRIVEN, and the shape "
                "no released case carries: ``pml_active`` False WITH a conductivity "
                "and five poles. Lifted off-device 2026-09-13 to dimensions 1, grid "
                "(1, 1, 400), conductivity True, susceptibilities 5, pml_active "
                "False — an mp.Absorber is a scalar conductivity, so the lift hands "
                "the fast path no PML at all. Arm set read off plan_step on that "
                "lift: ``fused pair D (no-PML stored E)`` alone, refused on the array "
                "module alone, while BOTH ordinary pairs are refused by name on 'no "
                "active PML layer (these kernels implement the split-field path "
                "only)' — so the arm is not merely admitted here, it is the only one "
                "contending. The B seam keeps its separate arms. TWO THINGS RIDE "
                "WITH THIS ROW rather than following it: the arm's "
                "PENDING_DEVICE_GATE_ARMS entry has to go and its ARM_CERTIFICATION "
                "row has to land, because runtime rung 7a reads both and "
                "``served_in_dispatch`` models neither."),
    },
    # ``material_dispersion_0d`` WAS DRIVEN HERE AND IS NOT ANY MORE, replaced by
    # ``no_pml_dispersive_2d`` below. It drove the right two axis values and its lift
    # was correct; what a (1, 1, 1) cell cannot carry is EVIDENCE ABOUT A SEAM
    # COMPOSITION, which is the only thing this gate measures. All four of this
    # gate's non-vacuity controls are vacuous there, each for a property of the CELL,
    # and the in-seam mutation decisively: the wall plane it writes IS the single
    # cell, which is also the source's deposit cell ``deposit_repair.apply``
    # recomputes, so the mutated value has no in-launch twin. Cells MUTATED BUT NOT
    # REPAIRED, measured off-device 2026-09-14: absorber_1d 399 of 400,
    # dispersive5_2d 90 of 8100, material_dispersion_0d ZERO of 1 -- and the
    # 2026-09-14 Triton shipped leg measured the consequence, refusing with
    # ``controls not passing: ['material_dispersion_0d:in-seam-pass-mutated-D']``
    # (verdict NULL-DID-NOT-DIVERGE, arrays_differing 0) while ZERO of its 36 cases
    # failed. DRIVE_CUDA had already dropped it for the same reason; this table kept
    # it pending a release decision, and that decision was to build a real replacement
    # rather than narrow the arm's row -- which is what the row below is.
    # THE BUILDER STAYS (``case_material_dispersion_0d``): it is a correct lift and
    # ``case_complex_no_pml_offdiag`` cites it.
    "no_pml_dispersive_2d": {
        "arms": ["fused pair D (no-PML stored E)"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "leg": "shipped",
        "why": ("THE SECOND VALUE OF TWO AXES ON THE SAME ROW, which is what lets "
                "``fused pair D (no-PML stored E)`` leave ``conductivity`` and "
                "``susceptibilities`` unpinned honestly: absorber_1d drives "
                "conductivity True at dimensions 1 over five poles, this case drives "
                "False at dimensions 2 over two. Those are the same two values "
                "``material_dispersion_0d`` supplied, so the swap changes no axis on "
                "the arm's row — only the cell that evidences them. "
                "Lifted off-device 2026-09-14 to dimensions 2, grid (120, 120, 1), "
                "susceptibilities 2, conductivity False, pml_active False, "
                "complex_storage False, off_diagonal_epsilon False, bloch False, "
                "beta 0.0, bfast False, nonlinearity False, cylindrical False. "
                "ADMISSION READ OFF THE ARM'S OWN PREDICATE on that lift, modulo the "
                "backend clause — the ``covered_modulo_backend`` factoring the "
                "predicate census uses on a NumPy host: "
                "``no_pml_fused_electric_pair_coverage`` and its stored-E half both "
                "returned ZERO non-backend residual, every reason being \"array "
                "module is 'numpy', not cupy\". A step on this host selected the "
                "arm's three CONSTITUENTS by name (no_pml_curl, no_pml_stored_e, "
                "ade_update_p); that is NOT evidence about Triton's composer, "
                "because the table that served was Metal, and this leg is what "
                "establishes the selection. "
                "IT CARRIES A MONITOR AND A REAL EXTENT, which is the whole reason "
                "it exists: on the (1, 1, 1) cell it replaces, the flux comparison "
                "and the SECOND CONSULT SITE were NOT-APPLICABLE and the in-seam "
                "mutation could not diverge, so the case could only pass. Here all "
                "four controls are measurements."),
    },
    "bloch_2d": {
        "arms": ["fused pair B (complex)", "fused pair D (complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the complex pairs under a Bloch phase; probe leg only",
    },
    "complex_nobloch_2d": {
        "arms": ["fused pair B (complex)", "fused pair D (complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the complex pairs under force_complex_fields with no k_point: "
                "bloch False, the case that frees their bloch row"),
    },
    "complex_beta_2d": {
        "arms": ["fused pair B (complex beta)", "fused pair D (complex beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the complex beta pairs on the special_kz cell stored complex",
    },
    "folded_complex_beta_2d": {
        "arms": ["fused pair B (folded complex beta)", "fused pair D (folded complex beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the folded complex beta pairs, with the complex fills beside them",
    },
    # ------------------------------------------------------------------
    # THE TARGET ROUND, COMPLEX CARTESIAN. Same measurement as the real block, with
    # one addition the complex products force: on this NumPy host the subnormal
    # policy in force is UNREADABLE, so every complex arm carries that refusal
    # BESIDE the array-module one. Both are facts about the host rather than about
    # the case, and the calibration says so rather than the reader assuming it —
    # bloch_2d, complex_nobloch_2d, complex_beta_2d and folded_complex_beta_2d,
    # every one of them typed from a device campaign, carry EXACTLY those two
    # clauses and nothing else on the same lift.
    #
    # These rows go here and not after the Dcyl pair below: a Dcyl COMPLEX case
    # stepped earlier in a process makes later real-storage Cartesian cases diverge
    # at subnormal magnitudes on both dispatch legs, so the cylindrical rows stay
    # LAST and nothing is appended after them.
    # ------------------------------------------------------------------
    "complex_1d": {
        "arms": ["fused pair B (complex)", "fused pair D (complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("ONE DECLARED DIMENSION under complex storage, which the complex "
                "pairs' ``dimensions`` row (pinned at 2) refuses today. Lifted "
                "off-device 2026-09-13 to dimensions 1, grid (1, 1, 480), "
                "complex_storage True, bloch True, susceptibilities 0. Arm set read "
                "off plan_step on that lift with the expansion probe offered: both "
                "complex pairs, refused on the host clauses alone. Serves "
                "refl-angular.py, TestPlanewave1D and TestReflectanceAngular_0_0."),
    },
    "complex_3d_thinline": {
        "arms": ["fused pair B (complex)", "fused pair D (complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE OTHER END OF THE SAME ROW: a (0, 0, L) cell DECLARED 3-D, which "
                "MEEP builds as a 3-D grid one cell wide in x and y and the "
                "dimensions reader keeps at 3. Four corpus rows are physically 1-D "
                "looking and answer dimensions 3, so this is the value they carry "
                "rather than the one they look like. Lifted off-device 2026-09-13 to "
                "dimensions 3, grid (1, 1, 275), complex_storage True, bloch True; "
                "arm set read off plan_step on that lift, both complex pairs on the "
                "host clauses alone. It is NOT the whole 3-D evidence — complex_3d "
                "below is why."),
    },
    "complex_3d": {
        "arms": ["fused pair B (complex)", "fused pair D (complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("A GENUINE 3-D COMPLEX GRID, and this row is worth ZERO board "
                "instances by construction. It is here because the seven corpus rows "
                "the widening serves are all (1, 1, N) cells, while the corpus also "
                "carries genuine 3-D complex rows the widened row would admit AT "
                "RUNTIME — so without this case ``dimensions = 3`` would be licensed "
                "on degenerate grids alone, which is the unevidenced-admit class this "
                "round exists to close. Lifted off-device 2026-09-13 to dimensions 3, "
                "grid (40, 40, 40), complex_storage True, bloch True, diagonal; arm "
                "set read off plan_step on that lift, both complex pairs on the host "
                "clauses alone. If the campaign budget bites, this is the one row "
                "that can be dropped without losing an instance, at that cost."),
    },
    "folded_complex_3d": {
        "arms": ["fused pair B (folded complex)", "fused pair D (folded complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX PAIRS AT 3-D, on the triangular-lattice oblique "
                "cell that is the corpus row's own shape. Lifted off-device "
                "2026-09-13 to dimensions 3, grid (8, 21, 126), folded 'mirror plane "
                "on X', complex_storage True, bloch True — the in-plane Bloch "
                "component keeps ``bloch`` at the True these arms already pin, so "
                "``dimensions`` is the one axis moving. Arm set read off plan_step on "
                "that lift: both folded complex pairs on the host clauses alone, with "
                "both fill slots host-refused beside them. Serves "
                "TestModeDecomposition.test_triangular_lattice_oblique."),
    },
    "folded_complex_kz2d_3d": {
        "arms": ["fused pair B (folded complex)", "fused pair D (folded complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE SECOND 3-D CORNER OF THE SAME ROW, built differently on purpose: "
                "a zero-z cell that MEEP carries as 3-D because kz_2d='3d' keeps the "
                "z component of the k_point from lifting as a beta. Two corners "
                "answering dimensions 3 for two different reasons is what makes the "
                "widened row evidence for the AXIS rather than for one cell shape. "
                "Lifted off-device 2026-09-13 to dimensions 3, grid (90, 62, 1), "
                "folded 'mirror plane on Y', complex_storage True, bloch True, beta "
                "0.0; arm set read off plan_step on that lift, both folded complex "
                "pairs on the host clauses alone. Serves "
                "TestEigCoeffs.test_binary_grating_special_kz_2_21_2."),
    },
    "folded_complex_offdiag_2d": {
        "arms": ["fused pair B (folded complex)"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX MAGNETIC PAIR'S OFF-DIAGONAL VALUE: "
                "folded_complex_2d with its slab replaced by a cylinder on the fold "
                "plane, so the two run shapes differ in ``off_diagonal_epsilon`` "
                "alone (dimensions 2, grid (80, 62, 1), folded 'mirror plane on Y', "
                "complex_storage True, bloch True). The composer selects the folded "
                "complex magnetic pair for step_B/update_H and the single arm "
                "`complex folded off-diagonal` for update_E; the electric twin is "
                "refused by name on the row product, so the D seam carries that "
                "single arm and one pair fuses. The update_E arm left "
                "PENDING_DEVICE_GATE_ARMS on the 2026-09-15 ruling, conditioned on "
                "the re-gate and rebind of triton_complex_offdiag_device_gate. Before "
                "it did, this row was a refusal control: "
                "dispatch_fused_route_2026-09-15_weldgrid drove it as `dispatch` and "
                "read DID-NOT-FUSE / NO-DROP with every slot on the array path, and "
                "dispatch_fused_route_2026-09-15_offdiag drove it as `no-fused-arm` "
                "and read PASS-NO-FUSED-ARM with refusing_rung pending-device-gate. "
                "Both runs recorded the composer building the update_E arm under "
                "this leg's licence. Serves TestHoleyWvgBands.test_fields_at_kx."),
    },
    "folded_complex_nobloch_offdiag_2d": {
        "arms": ["fused pair B (folded complex)"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE bloch = False VALUE OF THE SAME ARM, on a grid that is "
                "off-diagonal too. A dispatch row again from the 2026-09-15 ruling "
                "that lifted `complex folded off-diagonal` (update_E) out of "
                "PENDING_DEVICE_GATE_ARMS; while that arm was pending, "
                "dispatch_fused_route_2026-09-15_weldgrid read DID-NOT-FUSE / "
                "NO-DROP here and dispatch_fused_route_2026-09-15_offdiag, driving "
                "the row as a `no-fused-arm` control, read PASS-NO-FUSED-ARM with "
                "refusing_rung pending-device-gate. examples:solve-cw.py and "
                "TestArrayMetadata.test_array_metadata are refused TWICE over — on "
                "bloch AND on off-diagonal — so dropping either pin alone moves "
                "nothing, and this case beside folded_complex_offdiag_2d is what "
                "makes dropping both an evidenced move. Lifted off-device 2026-09-13 "
                "to dimensions 2, grid (81, 81, 1), folded 'mirror plane on X, Y', "
                "complex_storage True, bloch False, off_diagonal_epsilon True; arm "
                "set read off plan_step on that lift, the folded complex magnetic "
                "pair alone on the host clauses, its electric twin refused by name on "
                "the row product. The source sits in the quadrant BOTH mirrors keep — "
                "at (-4, 0) the lift refuses, because a source in a discarded half "
                "makes MEEP return an exactly-zero field that reads as complete."),
    },
    "complex_beta_bloch_2d": {
        "arms": ["fused pair B (complex beta)", "fused pair D (complex beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("A z BETA AND AN IN-PLANE BLOCH PHASE AT ONCE, which is what the "
                "corpus rows carry and what the released case cannot: "
                "``complex_beta_2d`` drives the same beta at k_x = 0 and reads bloch "
                "False, so the arms' ``bloch`` pin describes that case rather than "
                "the rows it refuses. Lifted off-device 2026-09-13 to dimensions 2, "
                "grid (120, 120, 1), complex_storage True, bloch True, beta 0.4 — one "
                "axis moved against the released case. Arm set read off plan_step on "
                "that lift: both complex beta pairs on the host clauses alone. Serves "
                "TestSpecialKz.test_special_kz."),
    },
    "folded_complex_beta_bloch_2d": {
        "arms": ["fused pair B (folded complex beta)",
                 "fused pair D (folded complex beta)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED CORNER OF THE SAME TWO AXES. On this table the folded "
                "complex beta pairs carry their own rows, so the fold needs its own "
                "case; on the CUDA table ONE arm pair serves both foldings and this "
                "case is what lets its ``folded`` row be deleted rather than widened. "
                "Lifted off-device 2026-09-13 to dimensions 2, grid (120, 62, 1), "
                "folded 'mirror plane on Y', complex_storage True, bloch True, beta "
                "0.4; arm set read off plan_step on that lift, both folded complex "
                "beta pairs on the host clauses alone, both fill slots host-refused. "
                "The source sits ON the mirror plane, which the lift accepts. Serves "
                "TestEigCoeffs.test_binary_grating_special_kz_{0_13_2, 1_17_7}."),
    },
    # RESTORED 2026-09-15 after it was WITHDRAWN 2026-09-14 on the instruction the
    # row carried itself. The withdrawal named what restoring it wanted: the Triton
    # pair weld cut and seeded, the arm's FUSED_RELEASE_ARM_AXES row, its
    # PENDING_DEVICE_GATE_ARMS entry gone, and both constituent arms' pending
    # entries retired. The constituents left on the 2026-09-14 release decision; this
    # batch lands the axes row, the RELEASED_FUSED_ARMS row naming this case, the
    # ARM_CERTIFICATION row and the pending deletion in fastpath.py. The weld is
    # ORDERED rather than answered by an edit: the pair gate re-runs after the batch
    # and seed_triton_welds.py cuts ``triton_complex_conductive_fused_pair_device_gate``
    # BEFORE this campaign starts, so a dispatching record quotes a readable ledger
    # entry. If that seed does not land, this row is withdrawn again with those
    # fastpath.py rows, for the reason the 2026-09-14 text gave: a dispatch
    # expectation on an arm the release refuses fails the gate on a correct case.
    #
    # PLACED BEFORE THE Dcyl COMPLEX ROWS, which stay last (see the note above the
    # complex Cartesian block), and after ``pml_2d``, which stays first.
    "complex_no_pml_3d": {
        "arms": ["fused pair D (complex conductive no-PML)"],
        "pairs": 1, "poles": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FIRST TRITON DRIVE OF THE COMPLEX NO-ABSORBER D PAIR, on the "
                "shape the four TestLoadDump 3-D rows it serves carry "
                "(chunk_layout_file_3d, chunk_layout_sim_3d, structure_3d, "
                "structure_sharded_3d). Lifted off-device 2026-09-13 to dimensions "
                "3, grid (23, 21, 27), complex_storage True, bloch True, beta 0.0, "
                "conductivity True, off_diagonal_epsilon False, susceptibilities 1, "
                "pml_active False. THE ARM SET WAS READ OFF A DEVICE LEG, not a "
                "NumPy lift: on dispatch_fused_route_cuda_2026-09-15_weldgrid's "
                "cuda_shipped_expansion_probe leg the unfused comparison ran the "
                "Triton table alone (tables_dispatched ['triton'], opted_in []) with "
                "that same run_shape and selected step_D 'complex conductive no-PML "
                "curl', update_E 'complex no-PML stored E' and update_P 'complex ADE "
                "update_P' at 6.0 launches a step -- the two arms this pair absorbs. "
                "Expected fused: 5.0, a drop of one for one pair at one pole, with "
                "update_P still three launches. The Hy source puts the deposit on "
                "the B seam, so the D seam holds the NoopPlan sentinel; on the same "
                "case the CUDA leg's absorbed-consult-withheld and "
                "in-seam-pass-mutated-D controls read DIVERGED-AS-REQUIRED, which "
                "makes a vacuous D-seam control here less likely without proving "
                "it."),
    },
    "cylindrical_m1": {
        "arms": ["fused pair B (cylindrical complex)", "fused pair D (cylindrical complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the cylindrical complex pairs at m = 1 (LAST: see the order note)",
    },
    "cylindrical_m0_complex": {
        "arms": ["fused pair B (cylindrical complex)", "fused pair D (cylindrical complex)"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the cylindrical complex pairs at m = 0 under forced complex storage",
    },
}

#: The ladder every case is compared at. 12 COMPLETE STEPS IS EXPLICIT because it
#: is this repository's house budget for "long enough for an error to accumulate",
#: and the rungs on either side of it are what separate a per-sub-step arithmetic
#: difference from a single late disagreement.
LADDER_HEAD: Tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48)

#: How many complete steps every CONTROL runs. Same house budget: a control that
#: ran one step could miss an error that needs the next step's neighbour reads.
CONTROL_STEPS = 12

#: The policy the harness installed, when it is NOT the one every dispatchable
#: family was certified under. Set by :func:`main` from ``--install-policy``, and
#: it changes what a dispatch case is required to DO: nothing may fuse, rung 8b
#: must refuse BY NAME, and the run must still be byte-identical to the array path.
#:
#: WHY IT IS AN ASSERTION AND NOT A NOTE. "The seam supports exactly one policy"
#: is a claim, and the leg that measures it is one where nothing dispatches — the
#: same shape a leg that quietly fell back has. Measured 2026-08-29 on the first
#: campaign: the flush leg reported ``DID-NOT-FUSE`` on every case and its armed
#: controls reported ``NOT-ARMED``, so its verdict line was indistinguishable from
#: a broken run and a reader had to go to the log to find the refusal. Naming the
#: expected outcome turns the leg's success into something the gate checks.
UNCERTIFIED_POLICY_INSTALLED: Optional[str] = None

#: The phrases rung 8b's refusal is identified by. Read off ``fastpath``'s own
#: text rather than restated, and matched as a conjunction so a refusal that moved
#: to another rung is not absorbed by a substring that happens to still appear.
POLICY_REFUSAL_NEEDLES: Tuple[str, ...] = (
    "float32", "subnormal policy", "certified under")


# ---------------------------------------------------------------------------
# The second NVIDIA table, as a RUN AXIS rather than a second gate
# ---------------------------------------------------------------------------
#
# ``--backend triton`` IS BYTE-FOR-BYTE WHAT THIS GATE ALWAYS RAN. Everything below
# is additive: a second DRIVE table, a preference in the leg environment, a second
# launch counter, and three arbitration controls. The reason it is one gate and not
# two is that the thing being measured is THE SEAM — the driver's nine consults, the
# byte ladder, the withheld control, the substitution proof — and a second copy of
# that would be a second chance to get the seam's evidence wrong.

#: The hand-CUDA table's per-case expectations. MEASURED OFF-DEVICE 2026-09-10 by
#: calling ``cuda_kernels.arms.plan_step(fields, pml, grid, fuse=True, sources=...)``
#: on a NumPy lift of each case and reading ``selected`` — the same rule
#: :data:`DRIVE` states for the Triton table, and the same reason it is sound: on a
#: NumPy host the PREDICATES decide and only the builders refuse, so admission is
#: exactly measurable off the device.
#:
#: A CASE WHOSE SELECTION HAS NOT BEEN MEASURED IS NOT IN THIS TABLE — the rule the
#: Triton table has carried since 2026-08-29, and the one that keeps a case that
#: does not lift from making the whole record un-cuttable. It is why the Tier 2
#: cases arrived in two rounds rather than one, and why TWO cases this file now
#: builds are still absent from the rows below: ``complex_no_pml_offdiag``, whose
#: arm is not released on this table yet, and ``folded_complex_kz2d_3d``, whose
#: release rows do not name it (see the note beside the complex block).
#: ``complex_no_pml_3d`` LEFT AND RETURNED TO THIS LIST ON 2026-09-14, both within
#: the same day: the N2 re-gate seeded its weld and the row was driven, the leg
#: measured the weld dispatching BY PREFERENCE, the row was withdrawn when
#: ``arbitration-default-precedence`` refused it while three Triton arms were
#: pending, and it returned when the release decision released those arms — see the
#: note at the end of DRIVE_CUDA, which records what the run measured both times.
#: ``folded_offdiag_dispersive_2d``, which earlier
#: rounds listed here as pending measurement, is not built at all: it does not lift
#: on a stock MEEP, and the comment above ``CASES`` says why.
#:
#: ``leg`` NAMES WHERE THE CASE DISPATCHES. Complex-storage products get their
#: expansion licence from the unified probe record and nothing else offers it, so
#: they dispatch on ``cuda_shipped_expansion_probe`` and on no other leg.
DRIVE_CUDA: Dict[str, Dict[str, Any]] = {
    # WITHDRAWN 2026-09-14: folded_complex_offdiag_2d and
    # folded_complex_nobloch_offdiag_2d were driven here for the folded complex
    # magnetic pair's off-diagonal corner. Both dispatch this table by PREFERENCE
    # (PASS-FUSED, substitution EXACT), but under DEFAULT precedence the step is
    # refused -- a D-side slot on a folded complex off-diagonal grid falls to the
    # Triton single arm `complex folded off-diagonal`, which is PENDING (its family
    # byte gate passed on the device; no planner/driver recertification stands behind
    # it, and the nine families the 2026-08-14 recert covered do not include its
    # own). So arbitration-default-precedence reads REFUSED and the leg cannot
    # release. The builders stay in CASES -- the TRITON table still drives both --
    # and these rows return in the round that certifies that arm.
    #
    # THAT ARM WAS LIFTED BY THE 2026-09-15 RULING, and the rows were still held out,
    # deliberately: the same batch changed the Triton table's answer on these grids
    # (the Triton folded complex magnetic pair now holds step_B/update_H under
    # default precedence), so the CUDA arbitration verdict there is a new
    # measurement rather than a restored one, and that batch kept the CUDA
    # campaign's case set unchanged. They return on a CUDA round of their own,
    # with the CUDA_OFF_DIAGONAL_CORNERS entry fastpath_cuda.py names.
    #
    # 2026-09-17 IS THAT ROUND, and both rows are below. Read the expectation
    # honestly: `expect` is `dispatch` because that is what the release table now
    # claims, and the arbitration-default-precedence control is what decides whether
    # the claim stands. A REFUSED reading here is a legitimate outcome, not a gate
    # defect -- it would mean the Triton table wins those two D-side slots on merit
    # under the shipped precedence, and the rows and their fastpath_cuda.py release
    # entries come back out together, as they did on 2026-09-14.

    "folded_complex_offdiag_2d": {
        "arms": ["cuda:folded complex fused magnetic pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE OFF-DIAGONAL CORNER OF THE FOLDED COMPLEX MAGNETIC PAIR, on this "
                "table. Same builder the Triton DRIVE row uses: folded_complex_2d "
                "with its slab replaced by a cylinder on the fold plane, so the two "
                "run shapes differ in ``off_diagonal_epsilon`` ALONE (dimensions 2, "
                "grid (80, 62, 1), folded 'mirror plane on Y', complex_storage True, "
                "bloch True). ONE pair, not two, and the row says so again after one "
                "campaign in which its `pairs` read 2 (2026-09-17_allpaths, "
                "VACUOUS-PASS 'names 2, expected 4'): this builder's source is an Ez "
                "point source, the driver injects it BETWEEN step_D and update_E, and "
                "`cuda:folded complex off-diagonal stencil weld` declares "
                "CARRIES_DEPOSIT_REPAIR False -- its own predicate refuses at the "
                "source-seam clause (reproduced off-device 2026-09-17), and the D "
                "seam falls to single arms. That is the weld's measured design, not a "
                "defect, so the weld's route case is the sibling row below with the "
                "corpus row's magnetic source; the count here is the claim the "
                "campaign checks, and a wrong one fails loudly at the slot count. "
                "Driven on 2026-09-14: PASS-FUSED with the substitution EXACT BY "
                "PREFERENCE, and REFUSED under default precedence because the D-side "
                "slot fell to the Triton single arm `complex folded off-diagonal` "
                "while that arm was PENDING; it is no longer pending. "
                "Serves TestHoleyWvgBands.test_fields_at_kx."),
    },
    "folded_complex_offdiag_hz_2d": {
        "arms": ["cuda:folded complex fused magnetic pair",
                 "cuda:folded complex off-diagonal stencil weld"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX OFF-DIAGONAL STENCIL WELD'S ROUTE CASE: the row "
                "above with the corpus row's OWN source, Hz under Mirror(Y, phase=-1), "
                "which lifts as a VolumeSource of field type B, lands the driver's "
                "injection on the B seam (where the magnetic pair carries a deposit "
                "repair) and leaves the D seam empty for a weld that carries none. "
                "TWO pairs, typed from a measurement rather than a hope: composed "
                "off-device 2026-09-17 through the census's CuPy-named grid shim, "
                "step_B/update_H hold the folded complex fused magnetic pair and "
                "step_D/update_E the weld; same lift otherwise (dimensions 2, grid "
                "(80, 62, 1), folded 'mirror plane on Y', complex_storage True, "
                "bloch True, off_diagonal_epsilon True, pml_active True). "
                "Serves TestHoleyWvgBands.test_fields_at_kx, the single instance "
                "behind the weld."),
    },
    # WITHDRAWN 2026-09-17, AND BOTH HALVES ARE THE RECORD. The row below was driven
    # once, on the 2026-09-17_singles probe leg, and the weld WORKED: PASS-FUSED,
    # substitution EXACT-ABSORBED-ARRAY-SLOT (counted drop 0 = 1 - 1 for the declared
    # update_E), 19/19 arrays identical, arbitration-cuda-preferred INCUMBENT-YIELDED.
    # What refused was arbitration-default-precedence -- REFUSED where the leg's other
    # 38 cases read HELD -- because under the shipped order the TRITON table selects
    # its PENDING `complex no-PML off-diagonal` at update_E and rung 7a sends the
    # WHOLE step to the array path (control record: step_path "array", the CUDA unit
    # refused by name "slots step_D held by triton:complex no-PML curl; update_E held
    # by triton:complex no-PML off-diagonal"). REFUSED is not in CONTROL_PASS, so the
    # row is withdrawn rather than left to refuse the leg -- the same property of the
    # incumbent that withdrew complex_no_pml_3d on 2026-09-14 -- and its release entry
    # sits in fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS with this reason. The
    # builder, CASE_INTENT and `baseline_array_slots` machinery stay: the row returns
    # verbatim the day fastpath.PENDING_DEVICE_GATE_ARMS releases the Triton arm.
    #
    # "complex_no_pml_offdiag": {
    #     "arms": ["cuda:complex no-absorber off-diagonal stencil weld"],
    #     "pairs": 1, "reached_by": "release", "expect": "dispatch",
    #     "seam": ["D"], "leg": "cuda_shipped_expansion_probe",
    #     "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
    #     # THE BASELINE CANNOT SERVE update_E ON THIS SHAPE, and the row says so
    #     # rather than letting the proof discover it: the only Triton candidate,
    #     # `complex no-PML off-diagonal`, is PENDING (fastpath.PENDING_DEVICE_GATE_ARMS)
    #     # and the only CUDA single, `complex off-diagonal no-PML`, carries no
    #     # resolver, so the veto leg leaves update_E on the array path and the weld
    #     # ABSORBS it (measured 2026-09-17_allpaths: fused [step_B, step_D, update_E],
    #     # veto [step_B, step_D], 2.0 launches a step on both legs). Declared so the
    #     # counted expectation reads pairs - 1 = 0 and the row goes red the day either
    #     # table serves that slot.
    #     "baseline_array_slots": ["update_E"],
    #     "why": ("THE UNFOLDED COMPLEX OFF-DIAGONAL WELD, WITHOUT A BOUNDARY LAYER, "
    #             "and the case this file listed for three weeks as BUILT AND LIFTED "
    #             "BUT DRIVEN NOWHERE -- because its arm's ledger entry was not seeded "
    #             "and a route row for an arm the release does not admit fails on a "
    #             "correct case. The entry is seeded now, from the 2026-09-17 re-run of "
    #             "gate_cuda_complex_offdiag_stencil_welds.py, so the row goes in. "
    #             "Lifted 2026-09-17: dimensions 2, grid (25, 25, 1), complex_storage "
    #             "True, bloch True at k_point (0.3892, 0.1597, 0), off_diagonal True, "
    #             "lossless, no susceptibility, NO boundary layer -- pml_active False "
    #             "is the axis that separates this arm from the folded complex weld on "
    #             "folded_complex_offdiag_2d. ONE ROUTE CASE, so every axis of its "
    #             "release row is pinned from a single lift; that narrowness is what "
    #             "cost material_dispersion_0d its row on 2026-09-14 and is called out "
    #             "in the release row itself. THE SUBSTITUTION READS "
    #             "EXACT-ABSORBED-ARRAY-SLOT, NOT EXACT: the weld replaces one Triton "
    #             "step_D launch AND an array-path update_E with one launch, so the "
    #             "counted drop is 0 against the declared `baseline_array_slots` above, "
    #             "and the uncounted saving is update_E's array-path elementwise "
    #             "kernels, which no witness in this gate measures. Serves "
    #             "tests:TestMaterialGrid.test_subpixel_smoothing."),
    # },
    "folded_complex_nobloch_offdiag_2d": {
        "arms": ["cuda:folded complex fused magnetic pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE bloch = False VALUE OF THE SAME CORNER, off-diagonal too: the "
                "same shape unphased, two mirror planes, grid (81, 81, 1), "
                "complex_storage True, bloch False, off_diagonal True. Both values of "
                "``bloch`` are driven because the CUDA_OFF_DIAGONAL_CORNERS entry "
                "lists both, and a corner that named a value it never drove would be "
                "the widening-without-evidence this table exists to refuse. "
                "examples:solve-cw.py and TestArrayMetadata.test_array_metadata are "
                "refused twice over -- on bloch AND on off-diagonal -- so this case "
                "beside folded_complex_offdiag_2d is what makes dropping both pins an "
                "evidenced move rather than two half-moves."),
    },

    "pml_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        # THE VETO LEG IS THIS TABLE'S OWN, since 2026-09-17: under
        # MEEP_GPU_BACKEND_PREFERENCE=cuda the merge adopts the two certified
        # real-PML single SEAMS (fastpath_cuda.SINGLE_ARM_SEAMS), so with fused arms
        # vetoed every one of this row's four slots is served by cuda:PML /
        # cuda:ordinary and the fusion number is measured against the kernels the
        # pairs replace. Declared, and checked against the veto leg's own record.
        "unfused_baseline": "cuda",
        "why": "the ordinary unfolded real-float32 PML pair on both seams",
    },
    "magnetic_seam_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        # THE VETO LEG IS THIS TABLE'S OWN, since 2026-09-17: under
        # MEEP_GPU_BACKEND_PREFERENCE=cuda the merge adopts the two certified
        # real-PML single SEAMS (fastpath_cuda.SINGLE_ARM_SEAMS), so with fused arms
        # vetoed every one of this row's four slots is served by cuda:PML /
        # cuda:ordinary and the fusion number is measured against the kernels the
        # pairs replace. Declared, and checked against the veto leg's own record.
        "unfused_baseline": "cuda",
        "why": "the B-seam deposit repair, on this table's own installer shapes",
    },
    "pml_1d": {
        "arms": ["cuda:fused magnetic pair", "cuda:fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        # THE VETO LEG IS THIS TABLE'S OWN, since 2026-09-17: under
        # MEEP_GPU_BACKEND_PREFERENCE=cuda the merge adopts the two certified
        # real-PML single SEAMS (fastpath_cuda.SINGLE_ARM_SEAMS), so with fused arms
        # vetoed every one of this row's four slots is served by cuda:PML /
        # cuda:ordinary and the fusion number is measured against the kernels the
        # pairs replace. Declared, and checked against the veto leg's own record.
        "unfused_baseline": "cuda",
        "why": "one dimension, the same two products",
    },
    "folded_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "fills": ["fill_B", "fill_D"],
        # THE VETO LEG IS THIS TABLE'S OWN on the four seam slots (see pml_2d): the
        # real-PML curl and constitutive singles cover the fold, and only the two
        # mirror fills stay with the incumbent -- measured off-device 2026-09-17,
        # all four slots adoptable with fused arms vetoed.
        "unfused_baseline": "cuda",
        "why": ("A CROSS-BACKEND COMPOSITION, and the only kind this campaign "
                "drives deliberately: the hand-CUDA table registers no mirror-fill "
                "arm at all, so ``fill_B``/``fill_D`` are served by TRITON's fill "
                "arms in the same step while the curls and constitutives are this "
                "table's. Rung 6b requires both fill slots to carry a kernel, so "
                "the fold is what MAKES the mix measurable rather than a corner of "
                "it."),
    },
    "pml_3d_diagonal": {
        "arms": ["cuda:fused magnetic pair", "cuda:fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        # THE VETO LEG IS THIS TABLE'S OWN, since 2026-09-17: under
        # MEEP_GPU_BACKEND_PREFERENCE=cuda the merge adopts the two certified
        # real-PML single SEAMS (fastpath_cuda.SINGLE_ARM_SEAMS), so with fused arms
        # vetoed every one of this row's four slots is served by cuda:PML /
        # cuda:ordinary and the fusion number is measured against the kernels the
        # pairs replace. Declared, and checked against the veto leg's own record.
        "unfused_baseline": "cuda",
        "why": "three dimensions with a diagonal epsilon",
    },
    "pml_3d": {
        "arms": ["cuda:fused magnetic pair"], "pairs": 1, "reached_by": "release",
        "expect": "dispatch", "seam": ["B"], "leg": "cuda_shipped",
        "why": ("THE CURVED 3-D CASE: the ordinary magnetic pair on an OFF-DIAGONAL "
                "3-D grid. The builder is a SPHERE, so MEEP's subpixel averaging "
                "writes off-diagonal chi1inv rows and the grid answers "
                "``off_diagonal_epsilon: true`` (measured on the GPU host 2026-09-11). "
                "Until 2026-09-13 the arm's release axes pinned that False and this "
                "case expected 'no-fused-arm' and served as the envelope control; "
                "the axis is a per-arm CORNER now (fastpath_cuda."
                "CUDA_OFF_DIAGONAL_CORNERS) and the sphere is its 3-D point: the B "
                "seam dispatches the pair, the D seam keeps the incumbent's "
                "``offdiag`` singles (no D-seam product admits an Ez run on an "
                "off-diagonal grid). Measured 2026-09-13, byte-identical to 96 "
                "steps when the case runs before any Dcyl complex case. "
                "``pml_3d_diagonal`` remains the 3-D case both pairs drive."),
    },
    "conductive_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:conductive fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "why": "the conductive D product; the B seam is lossless and ordinary",
    },
    "dispersive_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:three-slot dispersive weld"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped", "triple": True,
        "why": ("THE THREE-SLOT WELD, which is the shape the seam had never "
                "dispatched: ``step_D``/``update_E``/``update_P`` in ONE unit, the "
                "deposit repair between its two device groups, and ``update_P`` "
                "holding a single plan rather than the Triton table's list."),
    },
    "folded_dispersive_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:three-slot dispersive weld"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped", "triple": True,
        "fills": ["fill_B", "fill_D"],
        "why": "the three-slot weld on a fold, with Triton's fills beside it",
    },
    "special_kz_2d": {
        "arms": ["cuda:special_kz fused magnetic pair",
                 "cuda:special_kz fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "why": "real storage under a special-kz beta",
    },
    "cylindrical": {
        "arms": ["cuda:cylindrical real fused magnetic pair",
                 "cuda:cylindrical real fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "why": "the Dcyl m = 0 real shape, with the second consult site exercised",
    },
    "bloch_2d": {
        "arms": ["cuda:complex fused magnetic pair",
                 "cuda:complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("complex storage under a Bloch phase; the expansion licence comes "
                "from the unified probe record, so this case dispatches on the "
                "probe leg and on NO other"),
    },
    "folded_complex_2d": {
        "arms": ["cuda:folded complex fused magnetic pair",
                 "cuda:folded complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the folded complex products under the probe. On the NON-probe legs "
                "this case keeps the Triton table's own 'no-fused-arm' expectation, "
                "because without the licence the complex products refuse by name."),
    },
    "no_pml_2d": {
        "arms": [], "pairs": 0, "reached_by": "release", "expect": "no-fused-arm",
        "seam": [], "leg": "cuda_shipped",
        "why": ("THE NEGATIVE CASE, and it is a real one rather than a gap: this "
                "table selects UNLAUNCHABLE single arms plus its null constitutive "
                "arm here (the two no-absorber families carry no resolver), and the "
                "merge adopts fused products or certified launchable single SEAMS "
                "only — so the seam stays on whatever the primary table gave it and "
                "no CUDA unit dispatches. A case that dispatched here would mean the "
                "merge had started adopting an unlaunchable single arm."),
    },
    # ------------------------------------------------------------------
    # TIER 2 (2026-09-13). ORDER IS LOAD-BEARING: a Dcyl COMPLEX case stepped earlier
    # in the process makes every later REAL-storage Cartesian case diverge from the
    # array path at subnormal magnitudes on both dispatch legs (measured 2026-09-13;
    # fastpath_cuda's release table records the experiments), so the five
    # cylindrical cases come LAST and lift only on the probe leg. The real cases go
    # first, then the complex Cartesian ones, which were measured not to poison.
    # ------------------------------------------------------------------
    "offdiag_2d": {
        "arms": ["cuda:fused magnetic pair"], "pairs": 1, "reached_by": "release",
        "expect": "dispatch", "seam": ["B"], "leg": "cuda_shipped",
        "why": ("the ordinary magnetic pair on a 2-D off-diagonal grid with an Ez "
                "source: the 2-D point of its off-diagonal corner, one seam (the D "
                "seam keeps the incumbent's offdiag singles)"),
    },
    "offdiag_magnetic_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:off-diagonal stencil weld"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "why": ("THE STENCIL WELD, which is the scratch-output shape the board "
                "marks STENCIL-BLOCKED for an in-place weld: step_D/update_E in one "
                "launch on the in-plane E an Hz source keeps live, beside the "
                "ordinary magnetic pair on the B seam"),
    },
    "folded_offdiag_magnetic_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:folded off-diagonal stencil weld"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "fills": ["fill_B", "fill_D"],
        "why": "the folded stencil weld on a fold, with Triton's fills beside it",
    },
    "bfast_1d": {
        "arms": ["cuda:BFAST fused magnetic pair", "cuda:BFAST fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped",
        "why": ("the BFAST pairs on refl-angular's shape: a z-only cell declared "
                "3-D with the angle carried by bfast_scaled_k, real storage"),
    },
    # ------------------------------------------------------------------
    # THE TARGET ROUND (2026-09-13), REAL STORAGE. Measured the way this table's
    # note above prescribes and through the instrument the census already uses:
    # ``arms.plan_step(fields, pml, grid, fuse=True)`` on a NumPy lift of each case
    # with the grid wrapped in ``cuda_predicate_battery._GridWithCupyBackend`` — the
    # lifted grid with ``xp``'s NAME reported as cupy and every other read
    # forwarded. The shim is needed because every shipped CUDA predicate
    # SHORT-CIRCUITS on "backend is not CuPy", so the refusal list on this host
    # cannot be read the way the Triton table's can; with it, ``selected`` is the
    # composer's real per-slot answer. CALIBRATED FIRST: run against the rows
    # already in this table it reproduces every one of their ``arms``, including the
    # one-arm off-diagonal rows and the three-slot weld's slot map.
    # ------------------------------------------------------------------
    "folded_special_kz_2d": {
        "arms": ["cuda:special_kz fused magnetic pair",
                 "cuda:special_kz fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped", "fills": ["fill_B", "fill_D"],
        "why": ("THE special_kz PAIRS ON A FOLD, which is the whole of the lever "
                "that deletes their ``folded`` row: that axis has only three "
                "spellings, so admitting both foldings means deleting the row, and "
                "the row may be deleted only because ``special_kz_2d`` drove the "
                "unfolded value and this case drives the folded one. The builder has "
                "been in CASES since the Triton phase-B round and is a Triton DRIVE "
                "row already; this is the CUDA row it never had, and it is the "
                "cheapest lever of the batch — real storage, no probe, no new "
                "builder. Selection measured on a NumPy lift through the backend "
                "shim 2026-09-13: both special_kz pairs, on all four seam slots, "
                "with Triton's mirror-fill arms beside them as on folded_2d. Serves "
                "TestSpecialKz.test_eigsrc_kz_1_real_imag."),
    },
    "dispersive5_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:three-slot dispersive weld"],
        "pairs": 2, "poles": 5, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped", "triple": True,
        "why": ("FIVE POLES, which is a value of an INTEGER axis and not a widening "
                "of it: ``cuda:fused magnetic pair`` pins ``susceptibilities`` at "
                "{0, 1} and its cases carry exactly 0 and 1, so the four TestLoadDump "
                "2-D rows at five states are refused on that row. Lifted off-device "
                "2026-09-13 to dimensions 2, grid (90, 90, 1), real, diagonal, "
                "susceptibilities 5, pml_active True. THE ROW NAMES TWO ARMS BECAUSE "
                "THE COMPOSER SELECTS TWO, measured through the backend shim on that "
                "lift: the magnetic pair takes step_B/update_H and the THREE-SLOT "
                "dispersive weld takes step_D/update_E/update_P — the weld's release "
                "row carries no susceptibilities clause at all, so it is admitted "
                "here as well, and a row naming only the magnetic pair would fail "
                "this gate's own consistency leg. That means the weld's own released "
                "cases must list this case too."),
    },
    "dispersive6_2d": {
        "arms": ["cuda:fused magnetic pair", "cuda:three-slot dispersive weld"],
        "pairs": 2, "poles": 6, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped", "triple": True,
        "why": ("SIX POLES, the count the three stochastic_emitter corpus rows "
                "carry, on a builder that differs from dispersive5_2d's in the count "
                "ALONE — which is what makes each case evidence for one value rather "
                "than for a new cell. Lifted off-device 2026-09-13 to the same shape "
                "at susceptibilities 6; selection measured through the backend shim "
                "on that lift, the same two arms on the same five slots. It is also "
                "the whole of the Metal ``fused magnetic B/H pair``'s six-pole "
                "widening, which is why the builder lives here and both route files "
                "share it."),
    },
    "absorber_1d": {
        "arms": ["cuda:no-absorber three-slot dispersive weld"],
        "pairs": 1, "poles": 5, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "leg": "cuda_shipped", "triple": True,
        "why": ("THE WHOLE RELEASE OF A WELDED PRODUCT THAT HAS NEVER HAD A CASE, on "
                "the shape its own pending text names: an absorber rather than a "
                "PML, so ``pml_active`` False and ``conductivity`` True at once. "
                "Lifted off-device 2026-09-13 to dimensions 1, grid (1, 1, 400), "
                "conductivity True, susceptibilities 5, pml_active False. Selection "
                "measured through the backend shim on that lift: the no-absorber "
                "three-slot dispersive weld takes step_D/update_E/update_P — one "
                "UNIT occupying three slots and saving one launch — while step_B "
                "and update_H keep this table's single ``conductive`` and "
                "``no-PML null`` arms, so the row is one pair, one seam, and "
                "``triple``. The release row this evidences must move ``pml_active`` "
                "out of the shared CUDA envelope onto every arm first, or the shared "
                "row refuses the case before its own row is read."),
    },
    # THE ROUTE CASE OF THE CUDA DISPERSIVE OFF-DIAGONAL SINGLE (2026-09-27). Until
    # this row, 0 of this table's 38 route cases selected ``cuda:dispersive
    # off-diagonal`` on update_E (NumPy lift through the census shim), so an arm with
    # a resolver, a launcher and a certification row had no driver-route evidence.
    # It is a NO-FUSED-ARM row because nothing fuses on its shape on either NVIDIA
    # table, and the ``singles`` map is what makes it evidence rather than a pass
    # that holds whether or not the single ran: see the no-fused-arm branch of
    # ``run_case``.
    "folded_offdiag_dispersive2_2d": {
        "arms": [], "pairs": 0, "reached_by": "release", "expect": "no-fused-arm",
        "seam": [], "leg": "cuda_shipped",
        "singles": {"fill_B": "cuda:mirror fill", "fill_D": "cuda:mirror fill",
                    "step_B": "cuda:PML", "update_H": "cuda:ordinary",
                    "step_D": "cuda:PML",
                    "update_E": "cuda:dispersive off-diagonal"},
        "why": ("THE DISPERSIVE OFF-DIAGONAL update_E SINGLE ON THE DRIVER ROUTE, with "
                "the PML step_D single as its seam partner and the mirror-fill twins "
                "beside them. Lifted off-device 2026-09-27 to dimensions 2, grid "
                "(160, 61, 1), folded 'mirror plane on Y', real storage, "
                "off_diagonal_epsilon True, susceptibilities 2, pml_active True; "
                "fastpath.released_fused_arms and fastpath_cuda.released_fused_arms "
                "both read [] on that shape, and the CUDA table alone composes the six "
                "slots ``singles`` names with update_P ('ADE', no launch) left to the "
                "array path or to Triton (results/dispatch_flip_impl_2026-09-27/"
                "followup_cuda_case_selection.log). No fused product may be driven on "
                "any leg; on the by-preference legs and cuda_alone the six singles "
                "must be the ones that dispatched; on cuda_default_precedence the "
                "Triton table composes first and the row asks only for byte identity "
                "(the CROSS_TABLE_SEAM_REFUSED_SINGLES rule keeps this single out of a "
                "D/E seam whose step_D Triton holds); on cuda_flush rung 8b refuses by "
                "name. The builder's docstring says why the background carries the "
                "poles, why two poles and why the fold."),
    },
    # ``material_dispersion_0d`` IS NOT DRIVEN HERE, and the reason is worth the
    # paragraph because the case is correct, the builder is kept, and the axes it
    # drove were real.
    #
    # It was added to drive the OTHER value of two axes on this row — dimensions 2
    # against absorber_1d's 1, conductivity False against its True — on a ZERO-EXTENT
    # cell, grid (1, 1, 1). Everything about that lift is true. What a one-cell grid
    # cannot do is carry EVIDENCE ABOUT A SEAM COMPOSITION, which is the only thing
    # this gate measures, and the 2026-09-14 CUDA shipped leg is where that became
    # measurable rather than arguable. Every non-vacuity control this gate owns is
    # vacuous there, each for a property of the CELL:
    #
    #   * the FLUX comparison — the builder returns no monitors, because a one-cell
    #     grid has nothing to put a flux plane on;
    #   * the SECOND CONSULT SITE — same fact, and beneath it every curl is zero, so
    #     the magnetic half-step genuinely changes nothing and the site's own
    #     non-vacuity clause fires on a true statement;
    #   * the WITHHELD ABSORBED CONSULT — the layer is inactive, so the array
    #     constitutive is a pure overwrite and a second application is the identity
    #     (measured off-device 2026-09-14: 0 of 29 arrays);
    #   * the IN-SEAM PASS MUTATION — and this is the one that decides it. The
    #     mutation writes the wall plane ``array[0, :, :]``, which on a (1, 1, 1)
    #     grid is the single cell; that cell is also the source's deposit cell, which
    #     ``deposit_repair.apply`` recomputes. Measured off-device 2026-09-14, cells
    #     MUTATED BUT NOT REPAIRED: absorber_1d 399 of 400, dispersive5_2d 90 of
    #     8100, material_dispersion_0d ZERO of 1. There is no in-launch twin left for
    #     the mutation to be compared against, so the control cannot diverge and the
    #     2026-09-14 leg recorded exactly that.
    #
    # Four vacuous controls leave byte identity and a launch count, which every weld
    # gate already carries. A case that can only pass is not evidence, and scoring it
    # would have released this leg on the absence of a measurement. So the case goes,
    # and the two axes it drove are PINNED on the arm's row to the values
    # ``absorber_1d`` actually drove (``fastpath_cuda``: ``dimensions`` 1,
    # ``conductivity`` True) rather than left open on its authority.
    #
    # THE BUILDER STAYS (``case_material_dispersion_0d``): it is a correct lift, it is
    # cited by ``case_complex_no_pml_offdiag``, and a non-degenerate 2-D cell with no
    # boundary layer is what would drive those two axes honestly.
    "complex_nobloch_2d": {
        "arms": ["cuda:complex fused magnetic pair", "cuda:complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the complex pairs under force_complex_fields with NO k_point: "
                "bloch False beside bloch_2d's True, the case that frees their "
                "bloch row; probe leg, as bloch_2d"),
    },
    "complex_beta_2d": {
        "arms": ["cuda:complex beta fused magnetic pair",
                 "cuda:complex beta fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": "the complex beta pairs on the special_kz cell stored complex",
    },
    # ------------------------------------------------------------------
    # THE TARGET ROUND, COMPLEX CARTESIAN. Same instrument as the real block above,
    # with the three expansion licences the census binds handed to the composer, so
    # the complex families answer on their licence rather than on its absence. These
    # rows go BEFORE the five cylindrical rows and nothing is appended after them:
    # a Dcyl complex case stepped earlier in a process makes every later
    # real-storage Cartesian case diverge at subnormal magnitudes on both dispatch
    # legs, and the cause is open.
    # ------------------------------------------------------------------
    "complex_1d": {
        "arms": ["cuda:complex fused magnetic pair",
                 "cuda:complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("ONE DECLARED DIMENSION under complex storage: the complex pairs pin "
                "``dimensions`` at 2 and this is the 1 of the frozenset({1, 2, 3}) "
                "they move to. Lifted off-device 2026-09-13 to dimensions 1, grid "
                "(1, 1, 480), complex_storage True, bloch True; selection measured "
                "through the backend shim on that lift, both complex pairs on all "
                "four seam slots. Probe leg and no other, as bloch_2d."),
    },
    "complex_3d_thinline": {
        "arms": ["cuda:complex fused magnetic pair",
                 "cuda:complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE 3 OF THE SAME MOVE, on the shape four corpus rows actually "
                "carry: a (0, 0, L) cell DECLARED 3-D, which MEEP builds one cell "
                "wide in x and y and the dimensions reader keeps at 3. Lifted "
                "off-device 2026-09-13 to dimensions 3, grid (1, 1, 275), "
                "complex_storage True, bloch True; selection measured through the "
                "backend shim, both complex pairs."),
    },
    "complex_3d": {
        "arms": ["cuda:complex fused magnetic pair",
                 "cuda:complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("A GENUINE 3-D COMPLEX GRID, worth ZERO instances on this board and "
                "driven anyway: without it the widened ``dimensions`` row would be "
                "licensed by (1, 1, N) cells alone while admitting the corpus's real "
                "3-D complex rows at runtime, which is the unevidenced-admit shape "
                "this round is closing. Lifted off-device 2026-09-13 to dimensions 3, "
                "grid (40, 40, 40), complex_storage True, bloch True, diagonal; "
                "selection measured through the backend shim, both complex pairs."),
    },
    "folded_complex_3d": {
        "arms": ["cuda:folded complex fused magnetic pair",
                 "cuda:folded complex fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX PAIRS AT 3-D, on the triangular-lattice oblique "
                "cell. Lifted off-device 2026-09-13 to dimensions 3, grid "
                "(8, 21, 126), folded 'mirror plane on X', complex_storage True, "
                "bloch True — the in-plane Bloch component holds ``bloch`` at the "
                "value these arms already pin, so ``dimensions`` is the only axis "
                "moving. Selection measured through the backend shim on that lift: "
                "both folded complex pairs, with Triton's mirror-fill arms in the "
                "two fill slots as on folded_complex_2d."),
    },
    # ``folded_complex_kz2d_3d`` IS DELIBERATELY NOT A ROW HERE, and the omission is
    # a reconciliation rather than an oversight. It is the folded complex pairs'
    # SECOND 3-D corner — a zero-z cell MEEP carries as 3-D because kz_2d='3d' — and
    # it lifts and composes on this table exactly as folded_complex_3d does
    # (measured through the backend shim 2026-09-13: both folded complex pairs on all
    # four seam slots). What it is not is CITED: this table's release rows name
    # ``folded_complex_3d`` alone as their 3-D case, and the merge bar checks the two
    # lists for EQUALITY in both directions, so a row here for a case the release
    # does not name would refuse the cut. It costs nothing to leave out — the corner
    # is the same axis folded_complex_3d already drives — and the Metal table, whose
    # own lever requires two corners, drives it from the same builder. Add the row
    # the day the release names the case, not before.
    "folded_complex_beta_bloch_2d": {
        "arms": ["cuda:complex beta fused magnetic pair",
                 "cuda:complex beta fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("A FOLDED COMPLEX-BETA GRID WITH AN IN-PLANE BLOCH PHASE, and on "
                "this table ONE case settles two axes: there is no folded "
                "complex-beta product here — the complex_beta family serves both "
                "foldings itself — so ``complex_beta_2d`` (unfolded, bloch False) "
                "beside this case (folded, bloch True) drives both values of both "
                "axes, which is what lets the ``folded`` and ``bloch`` rows be "
                "DELETED rather than widened. ``folded`` has only three spellings, "
                "so deletion is the only way to admit both. Lifted off-device "
                "2026-09-13 to dimensions 2, grid (120, 62, 1), folded 'mirror plane "
                "on Y', complex_storage True, bloch True, beta 0.4; selection "
                "measured through the backend shim on that lift: the two "
                "complex-beta pairs — NOT a folded variant, because the table has "
                "none — with Triton's fills beside them."),
    },
    "cylindrical_m0_complex": {
        "arms": ["cuda:cylindrical fused magnetic pair",
                 "cuda:cylindrical fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the cylindrical COMPLEX pairs at m = 0 (complex storage forced); five orders drive "
                "the m row, and these run LAST (see the order note above)"),
    },
    "cylindrical_mneg1": {
        "arms": ["cuda:cylindrical fused magnetic pair",
                 "cuda:cylindrical fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the cylindrical COMPLEX pairs at m = -1; five orders drive "
                "the m row, and these run LAST (see the order note above)"),
    },
    "cylindrical_m1": {
        "arms": ["cuda:cylindrical fused magnetic pair",
                 "cuda:cylindrical fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the cylindrical COMPLEX pairs at m = 1; five orders drive "
                "the m row, and these run LAST (see the order note above)"),
    },
    "cylindrical_m2": {
        "arms": ["cuda:cylindrical fused magnetic pair",
                 "cuda:cylindrical fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the cylindrical COMPLEX pairs at m = 2; five orders drive "
                "the m row, and these run LAST (see the order note above)"),
    },
    "cylindrical_m3": {
        "arms": ["cuda:cylindrical fused magnetic pair",
                 "cuda:cylindrical fused electric pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "leg": "cuda_shipped_expansion_probe",
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("the cylindrical COMPLEX pairs at m = 3; five orders drive "
                "the m row, and these run LAST (see the order note above)"),
    },
    # WITHDRAWN AND RESTORED ON 2026-09-14, AND BOTH HALVES ARE THE RECORD. The row
    # was driven once the N2 re-gate seeded the complex no-absorber three-slot weld,
    # and the expansion-probe leg measured the weld WORKING: ``arms_driven`` gave it
    # step_D/update_E/update_P, the release-route control read RELEASE-ADMITTED and
    # ``arbitration-cuda-preferred`` read INCUMBENT-YIELDED. What refused was
    # ``arbitration-default-precedence`` — REFUSED where that leg's other 33 cases
    # read ARBITRATION-DEFAULT-HELD — because under the shipped default order the
    # TRITON table is asked first and refused the D-side slot by name while
    # ``complex ADE update_P``, ``complex conductive no-PML curl`` and
    # ``complex no-PML stored E`` sat in fastpath.PENDING_DEVICE_GATE_ARMS. REFUSED
    # is not in CONTROL_PASS, so the row was withdrawn rather than left to refuse
    # the leg — the same property of the incumbent table the two folded-complex rows
    # at the head of this table were withdrawn for.
    #
    # THE ROW RETURNS because that refusal is gone: those three arms were RELEASED
    # on 2026-09-14 by the release decision recorded beside
    # fastpath.PENDING_DEVICE_GATE_ARMS, so the incumbent no longer refuses the
    # D-side slot and the default-precedence control has an incumbent to hold.
    # Worth 8 seam-instances, measured by a same-instrument A/B before the
    # withdrawal with no displacement of any released arm.
    "complex_no_pml_3d": {
        "arms": ["cuda:complex no-absorber three-slot weld"],
        "pairs": 1, "poles": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "leg": "cuda_shipped_expansion_probe", "triple": True,
        "tail_in_leading": True,
        "needs_probe": "MEEP_GPU_COMPLEX_EXPANSION_PROBE",
        "why": ("THE WHOLE RELEASE OF THE COMPLEX NO-ABSORBER SPAN. The shape is the "
                "one thing that separates it from ``absorber_1d`` above: complex "
                "storage and no boundary layer at all, so ``complex_storage`` True "
                "and ``pml_active`` False together. Lifted off-device 2026-09-13 to "
                "dimensions 3, grid (23, 21, 27), bloch True, conductivity True, "
                "susceptibilities 1. ONE UNIT occupying three slots "
                "(step_D/update_E/update_P) and saving TWO launches at P = 1, not "
                "the one a three-slot weld usually saves, because THIS WELD IS "
                "COMPONENT-MAJOR: its leading group welds all three sub-steps per "
                "component (3 launches a step) and its trailing group launches "
                "nothing at all, where the real weld's trailing group still "
                "advances update_P once per component. Measured on the GPU host "
                "2026-09-14 over 600 dispatches: unfused 6.0 launches a step "
                "against fused 4.0, the hook counter agreeing at 2.0 and the slot "
                "sets identical. ``poles`` and ``tail_in_leading`` are both "
                "declared here rather than left to default: "
                "this table states its expectation rather than reading it off the "
                "run. ITS TWO HISTORICAL BLOCKERS ARE BOTH ANSWERED: "
                "cuda_kernels/fingerprints.json bound no entry for this weld until "
                "the 2026-09-14 N2 re-gate (both canonical policy legs re-run on "
                "device, results/cuda_regate_2026-09-14_n2/, rebind_cuda_welds "
                "--seed cutting the entry FROM that run), and the incumbent table's "
                "three complex no-PML arms were pending until the same day's owner "
                "ruling released them. This arm carries a per-arm "
                "CUDA_FUSED_RELEASE_ARM_AXES row, so the release admits the case. "
                "SINCE 2026-09-15 THE TRITON TABLE DRIVES THIS CASE TOO, on its own "
                "two-slot ``fused pair D (complex conductive no-PML)`` (see the "
                "Triton row of the same name), so under default precedence the "
                "incumbent this weld's arbitration controls meet on step_D/update_E "
                "is that fused label rather than the two Triton single arms. The "
                "controls' verdicts against a FUSED incumbent on this case are "
                "unmeasured until the CUDA route campaign runs on those bytes."),
    },
}

#: Which legs each backend's campaign runs. The CUDA legs carry a ``cuda_`` prefix
#: so the two campaigns' artifacts cannot be confused in one results tree, and
#: ``cuda_default_precedence`` is the leg that MEASURES the shipped arbitration
#: rather than asserting it.
#:
#: ``cuda_alone`` (2026-09-27) is the composition a host WITHOUT a validated Triton
#: gets, which no earlier leg drove: every other CUDA leg ran with Triton 3.1.0
#: importable, and on the by-preference legs Triton filled whatever the hand-CUDA
#: table refused (the fills, and the D/E seam of the off-diagonal cases). It runs
#: with the preference UNSET -- a user on such a host sets nothing -- and with Triton
#: made unimportable INSIDE this process (``--without-triton``, :func:`withhold_triton`),
#: so rung 4 drops the Triton table by name exactly as it does on that host.
#: ``MEEP_GPU_KERNEL_TABLE=cuda`` is NOT the same measurement: it narrows the
#: candidate list before rung 4 asks about Triton at all, and with Triton importable
#: the subnormal gate still drives Triton's policy (``TABLE_GOVERNED_EXECUTORS``
#: governs it for the CUDA table), which a Triton-less host never does.
CAMPAIGN_LEGS: Dict[str, Tuple[str, ...]] = {
    "triton": ("flush", "harness_keep", "shipped", "shipped_expansion_probe"),
    "cuda": ("cuda_alone", "cuda_default_precedence", "cuda_flush",
             "cuda_harness_keep", "cuda_shipped", "cuda_shipped_expansion_probe"),
}

#: The legs that run with ``MEEP_GPU_BACKEND_PREFERENCE`` UNSET. Every other CUDA
#: leg is a by-preference leg and must run with ``--table-preference cuda``.
UNSET_PREFERENCE_LEGS: Tuple[str, ...] = ("cuda_default_precedence", "cuda_alone")

#: The one leg that runs with Triton withheld, and the only leg ``--without-triton``
#: may be passed to.
TRITON_ABSENT_LEG = "cuda_alone"

#: Set once by :func:`main` from :func:`withhold_triton` on the ``cuda_alone`` leg:
#: the record of HOW Triton was made unimportable and that it took. ``None`` on every
#: other leg, and every clause below that reads it treats ``None`` as "Triton is
#: whatever this host has".
TRITON_WITHHELD: Optional[Dict[str, Any]] = None


def withhold_triton() -> Dict[str, Any]:
    """Make this process a host WITHOUT Triton, before anything imports it.

    THE MECHANISM IS ``sys.modules["triton"] = None``, and the choice is measured
    against the one alternative that looks equivalent. With a ``None`` entry,
    ``import triton`` raises ``ModuleNotFoundError`` and ``importlib.util.find_spec``
    returns ``None`` -- both exactly what an interpreter with no Triton installed
    answers -- so rung 4's ``import triton`` drops the Triton table by name ("Triton
    is not importable on this host"), and ``subnormal_policy._executor_present``
    reads the executor as ABSENT and installs nothing on it ("Triton not installed").
    A PYTHONPATH shim that shadowed the package with a module raising ``ImportError``
    would still be FOUND by ``find_spec``: the policy installer would then read
    Triton as present, fail to reach its backend class, report the policy
    unattained, and rung 8b would refuse every plan -- for a reason no Triton-less
    host has.

    REFUSES rather than proceeding when Triton is already imported: a sentinel set
    after the import changes nothing about the module already bound, and a leg that
    believed it had no Triton while one was loaded would be the vacuity this gate
    exists to catch. The gate module and its end-to-end sibling import neither
    ``meep_gpu`` nor Triton at module scope, so :func:`main` reaches this first.

    Returns the record the provenance and the summary carry.
    """
    import importlib.util  # noqa: PLC0415

    present = sys.modules.get("triton")
    if present is not None:
        raise RuntimeError(
            f"Triton is already imported in this process ({present!r}); a sentinel "
            "set now would leave the bound module in place, so this leg cannot be "
            "a Triton-less host")
    try:
        installed = importlib.util.find_spec("triton") is not None
    except Exception:  # noqa: BLE001 - an unreadable spec is recorded, not fatal
        installed = None
    try:
        from importlib import metadata  # noqa: PLC0415

        version: Optional[str] = metadata.version("triton")
    except Exception:  # noqa: BLE001 - not installed, or no metadata
        version = None
    sys.modules["triton"] = None  # type: ignore[assignment]
    spec_after = importlib.util.find_spec("triton")
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError as exc:
        import_error: Optional[str] = repr(exc)
    else:
        import_error = None
    if spec_after is not None or import_error is None:
        raise RuntimeError(
            "the sentinel did not take: find_spec('triton') returned "
            f"{spec_after!r} and the import raised {import_error!r}")
    return {
        "mechanism": ("sys.modules['triton'] = None, set by --without-triton before "
                      "any module imported Triton"),
        "installed_triton_hidden": installed,
        "installed_triton_version": version,
        "find_spec_after": None,
        "import_error": import_error,
        "why_not_a_path_shim": (
            "a shadowing package is still found by importlib.util.find_spec, so the "
            "subnormal-policy installer would read the Triton executor as present, "
            "fail to reach its backend class and refuse the plan at rung 8b -- a "
            "refusal no host without Triton makes"),
    }


class AbsentTritonCounter:
    """The Triton launch witness on a process where Triton cannot be imported.

    Reads zero on every snapshot, BY CONSTRUCTION rather than by observation: no
    Triton kernel can launch in a process whose ``import triton`` raises, and
    :func:`withhold_triton` verified that it does before this was built. The snapshot
    carries ``installed: False`` and the reason, so an artifact never presents this
    as a hook that watched and saw nothing. The CUDA witnesses are the ones this leg
    is scored on (``fused_leg_is_real`` asks them whenever the CUDA table served).
    """

    installed = False
    why = ("Triton is withheld in this process (--without-triton), so no Triton "
           "kernel can launch and there is no Triton entry point to count at")

    def install(self) -> None:
        raise RuntimeError(self.why)

    def snapshot(self) -> Dict[str, Any]:
        return {"total": 0, "hook_calls": 0, "zero_grid_launches": 0,
                "by_kernel": {}}

    delta = staticmethod(e2e.TritonLaunchCounter.delta)


def sole_table_verdict(record: Dict[str, Any]) -> str:
    """Did this one-step record compose the hand-CUDA table ALONE, for the right reason?

    ``SOLE-TABLE-COMPOSED`` needs all three: the effective order is ``["cuda"]``,
    the record's own Triton block says the table was not a candidate, and the reason
    it gives is the import -- not a version, not a device, not a preference. Anything
    else is ``SOLE-TABLE-NOT-SHOWN``, which is not in the control pass set.
    """
    arbitration = record.get("arbitration") or {}
    triton = (record.get("tables") or {}).get("triton") or {}
    if (list(arbitration.get("effective_order") or []) == ["cuda"]
            and triton.get("candidate") is False
            and "not importable" in str(triton.get("refused_because") or "")):
        return "SOLE-TABLE-COMPOSED"
    return "SOLE-TABLE-NOT-SHOWN"


#: The substitution verdict of a ``cuda_alone`` proof whose ONLY departure from
#: ``EXACT-ABSORBED-ARRAY-SLOT`` is that the absorbed slots are not the ones the drive
#: row declares (:func:`tritonless_baseline_only`). Spelled apart from the two
#: :data:`EXACT_SUBSTITUTION_VERDICTS` so a reader of ``summary['substitution']`` can
#: tell a proof against this table's own Triton-less baseline from one against the
#: baseline the drive row was written for.
TRITONLESS_SUBSTITUTION_VERDICT = "EXACT-ABSORBED-TRITONLESS-BASELINE"


def baseline_launches_per_step(slot: str, spec: Dict[str, Any]) -> int:
    """What the drive row's expected drop assumes the veto leg spent on ``slot``.

    The baseline model :func:`expected_substitution` writes out and the 2026-09-14
    the GPU host campaign measured on six dispersive cases: ONE launch a step for a curl
    or constitutive slot, and ``3P`` for ``update_P`` -- the Triton ADE advances once
    per pole per component. A slot the Triton-less veto leg left on the array path
    launched none of those, so each absorbed slot lowers the counted drop by exactly
    what this returns. For every slot but ``update_P`` it is 1, which is the
    ``EXACT-ABSORBED-ARRAY-SLOT`` branch's ``expected_drop - len(absorbed)``.
    """
    return 3 * int(spec.get("poles", 1)) if slot == "update_P" else 1


def tritonless_baseline_only(substitution: Dict[str, Any],
                             spec: Dict[str, Any]) -> bool:
    """Is a NOT-COMPARABLE substitution on the ``cuda_alone`` leg ONLY the baseline, and EXACT?

    On a host without Triton the veto leg's baseline is this table's launchable
    singles and the ARRAY PATH for every other slot -- there is no second table to
    serve them. So wherever a fused unit spans a slot no CUDA single serves (the
    dispersive three-slot welds, the conductive, special-kz, cylindrical and BFAST
    products), the veto leg left that slot on the array path, the drive row's
    ``baseline_array_slots`` (written for the legs whose baseline is another table's
    kernel) does not name it, and the strict clause reads NOT-COMPARABLE on a correct
    result.

    ONE RULE IS WAIVED AND ONLY ONE: ``absorbed == declared``. Everything else the
    ``EXACT-ABSORBED-ARRAY-SLOT`` branch of :func:`substitution_proof` requires is
    required here, off the proof's own recorded facts: no slot the veto leg
    dispatched may be missing from the fused leg, every absorbed slot must lie inside
    a unit the fused leg drove, and the LAUNCH ARITHMETIC must hold -- the counted
    drop is the drive row's expected drop minus what the baseline would have spent on
    each absorbed slot (:func:`baseline_launches_per_step`), and the measured drop
    must equal it on both witnesses (``counts_agree``). So a fused leg that saved the
    wrong number of launches still fails the case.

    On success the proof's verdict is REWRITTEN to
    :data:`TRITONLESS_SUBSTITUTION_VERDICT`, the strict reading is kept under
    ``strict_verdict`` / ``strict_why``, and ``expected_drop_per_step`` becomes the
    counted drop -- so ``summary['substitution']`` never reads NOT-COMPARABLE beside a
    PASS-FUSED row. On failure the verdict stays NOT-COMPARABLE and the counted drop
    it was checked against is recorded.
    """
    if TRITON_WITHHELD is None or substitution.get("verdict") != "NOT-COMPARABLE":
        return False
    if substitution.get("lost_slots"):
        return False
    absorbed = sorted(set(substitution.get("absorbed_array_slots") or []))
    driven = set(substitution.get("driven_slots") or [])
    if not absorbed or not set(absorbed) <= driven:
        return False
    expected_drop = substitution.get("expected_drop_declared")
    drop = substitution.get("launch_drop_per_step")
    if expected_drop is None or drop is None:
        return False
    counted = expected_drop - sum(baseline_launches_per_step(slot, spec)
                                  for slot in absorbed)
    substitution["expected_drop_tritonless"] = counted
    if not (abs(drop - counted) < 1e-9 and substitution.get("counts_agree")):
        substitution["why_on_this_leg"] = (
            f"the veto leg left {absorbed} on the array path, so the counted drop is "
            f"{counted} ({expected_drop} per the drive row minus the baseline "
            f"launches of those slots); the fused leg dropped {drop} with "
            f"counts_agree={substitution.get('counts_agree')!r}")
        return False
    substitution["strict_verdict"] = substitution["verdict"]
    substitution["strict_why"] = substitution.get("why")
    substitution["verdict"] = TRITONLESS_SUBSTITUTION_VERDICT
    substitution["expected_drop_per_step"] = counted
    substitution["why"] = (
        f"the counted-launch saving is {counted}: {expected_drop} per the drive "
        f"row's units minus the baseline launches of {absorbed}, which the "
        "Triton-less veto leg served on the array path because no CUDA single "
        "serves them and this process has no second table; the uncounted saving is "
        "those slots' array-path elementwise kernels, which no witness in this gate "
        "measures")
    return True

#: Set once by :func:`main`. Module-level because the leg builders below are called
#: from a dozen places and threading a backend through every one of them would be a
#: dozen chances to pass the wrong one.
BACKEND = "triton"

#: THE SENTINEL MEANING "WHATEVER THIS LEG PREFERS". :func:`_fuse_env` takes an
#: explicit ``prefer`` from the three arbitration controls, which each pin a
#: direction on purpose; every other call site is a DRIVE row and must follow the
#: leg. A sentinel rather than ``None`` because ``None`` is itself a direction --
#: it is what ``cuda_default_precedence`` sets -- so the two cannot share a spelling.
LEG_PREFERENCE = "<leg>"

#: Set once by :func:`main` from ``--table-preference``, and it is WHAT SEPARATES
#: THE TWO NVIDIA LEGS. ``"cuda"`` is the ``cuda_shipped`` campaign: the switch
#: names this table, the incumbent yields, and every count the run licenses is a
#: BY-PREFERENCE number. ``None`` is ``cuda_default_precedence``: the switch is
#: unset, the release-gated Triton table composes first and holds every slot both
#: admit, and what dispatches is what a user who sets nothing actually gets. The
#: record publishes both, side by side and named; until this argument existed the
#: second leg could not be run at all, so the by-default number in
#: ``recut_driver_dispatch_record.py`` would have been an assertion.
TABLE_PREFERENCE: Optional[str] = "cuda"

#: THE HAND-CUDA LAUNCH WITNESS, set once by :func:`main` when one installs.
#: Module-level for the reason ``BACKEND`` is: ``step_leg`` is reached from five
#: call sites and threading a second counter through each is five chances to pass
#: ``None``. It stays ``None`` on a Triton or Metal run, and every site that reads
#: it treats absence as "this run has no CUDA witness" rather than as zero launches.
#:
#: IT EXISTED AND WAS DISCARDED. ``main`` built a ``CudaLaunchCounter`` and
#: installed it, and the variable died on the next line -- so a CUDA run's evidence
#: was taken entirely from the TRITON counter, which reads zero when only hand-CUDA
#: kernels launch. The gate therefore reported VACUOUS-PASS ("no Triton kernel
#: launched at all") on a case whose record showed four CUDA slots correctly driven.
CUDA_COUNTER: Optional[Any] = None


def drive_table(backend: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The per-case expectations for the backend this run is driving."""
    return DRIVE_CUDA if (backend or BACKEND) == "cuda" else DRIVE


def arbitration_verdict(record: Dict[str, Any], expected_table: str) -> str:
    """Which of the three arbitration answers this one-step record shows.

    THREE DIRECTIONS, AND EACH IS A DIFFERENT CLAIM:

    * ``INCUMBENT-YIELDED`` — the preference leg: the effective order puts this
      table first, the driven slots carry ITS labels, and the incumbent's kernels
      read zero on them. That is what licenses a by-preference dispatch number.
    * ``ARBITRATION-DEFAULT-HELD`` — the default leg: the preference is unset, every
      seam both tables admit went to the incumbent, and the record NAMES the unit
      that lost with the slots it lost. That is what licenses the by-default number.
    * ``PREFERENCE-REFUSED-BY-NAME`` — a preference naming a table this host cannot
      run is refused by NAME rather than answered with a different table.
    """
    decision = record.get("decision")
    arbitration = record.get("arbitration") or {}
    if decision == "refused":
        reason = record.get("refused_because") or ""
        if "names no candidate table" in reason:
            return "PREFERENCE-REFUSED-BY-NAME"
        return "REFUSED"
    order = list(arbitration.get("effective_order") or [])
    if order[:1] == [expected_table] and expected_table == "cuda":
        return "INCUMBENT-YIELDED"
    refused = (arbitration.get("refused") or {}).get("cuda") or {}
    if order[:1] == ["triton"] and refused:
        return "ARBITRATION-DEFAULT-HELD"
    if order[:1] == ["triton"]:
        return "ARBITRATION-DEFAULT-HELD-UNCONTESTED"
    return "UNCLASSIFIED"


def ladder_for(total: int) -> List[int]:
    points = [p for p in LADDER_HEAD if p <= total]
    step = LADDER_HEAD[-1] * 4
    while step < total:
        points.append(step)
        step *= 4
    points.append(total)
    return sorted(set(p for p in points if p > 0))


# ---------------------------------------------------------------------------
# Leg mechanics
# ---------------------------------------------------------------------------

def _fuse_env(arms: List[str],
              reached_by: str = "opt-in",
              prefer: Optional[str] = LEG_PREFERENCE) -> Dict[str, Optional[str]]:
    """The environment a fused leg runs under, and WHICH ROUTE reaches the arm.

    TWO ROUTES, AND THE SHIPPED ONE IS ``release``. Until 2026-08-29 there was only
    one: ``MEEP_GPU_FUSE_ARMS`` named the arms, because no arm was released and the
    switch was the only way to reach a fused product at all. The release changed
    what ships — with the switch UNSET, clause (8) admits
    ``fastpath.RELEASED_FUSED_ARMS`` inside ``FUSED_RELEASE_ENVELOPE`` — and a gate
    that kept driving the opt-in would be measuring a path no user takes. So an
    in-envelope case now runs with the switch UNSET and the release does the
    admitting; the folded cases, which the envelope does not cover, keep the opt-in,
    because there the switch really is the only route.

    THE ROUTE IS ASSERTED, NOT ASSUMED: :func:`release_route_leg` reads the record
    back and requires ``fusion.value`` to be ``None`` and ``opted_in`` empty on a
    ``release`` leg, so a leg that silently fell back to the opt-in cannot pass as
    evidence about the shipped path.
    """
    # THE TABLE PREFERENCE IS PART OF THE LEG, and it is set only on the CUDA
    # campaign. Under the shipped precedence the release-gated Triton table composes
    # FIRST and holds every slot both admit, so a CUDA leg run without the switch
    # would measure the incumbent — which is exactly what the
    # ``cuda_default_precedence`` leg is for, and why it is the ONE leg that passes
    # ``prefer=None`` deliberately.
    if prefer == LEG_PREFERENCE:
        prefer = TABLE_PREFERENCE
    preference = {"MEEP_GPU_BACKEND_PREFERENCE": prefer} if BACKEND == "cuda" else {}
    if reached_by == "release":
        return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": None, **preference}
    return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
            "MEEP_GPU_FUSE_ARMS": ",".join(arms) if arms else None, **preference}


ARRAY_ENV: Dict[str, Optional[str]] = {"MEEP_GPU_DISPATCH": "1",
                                       "MEEP_GPU_FUSED": "0",
                                       "MEEP_GPU_FUSE_ARMS": None}
def unfused_env() -> Dict[str, Optional[str]]:
    """The substitution baseline: every certified sub-step arm, NOTHING fused.

    THE VETO TOKEN IS READ FROM THE PACKAGE, not spelled ``"0"`` here, so a gate
    cannot go on asking for a baseline the product stopped offering — the failure
    this leg had on 2026-08-29 was exactly a stale idea of what a switch value
    meant, and a literal would let the same failure recur silently.

    A FUNCTION RATHER THAN A CONSTANT because ``meep_gpu`` is imported LATE
    everywhere in this gate and in ``gate_dispatch_end_to_end``: a module-scope
    import would pull MEEP in before the harness has decided the subnormal policy,
    which is the one ordering rung 8b's evidence depends on.
    """
    from meep_gpu import fastpath as fp  # noqa: PLC0415 - late by design, see above

    return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
            "MEEP_GPU_FUSE_ARMS": fp.FUSE_ARMS_VETO}


def step_leg(case: str, leg: Dict[str, Any], counter: Any, num_steps: int) -> None:
    """Advance one leg and keep the PER-CHUNK launch delta, not just the total.

    The per-chunk delta is the substitution proof's instrument and the reason this
    does not simply call the end-to-end gate's ``step_leg``: a total accumulated
    from the first chunk also contains the plan-time warm pass's zero-grid launches
    and every compile the first step paid for, and a launches-per-step computed
    over that is the fusion's ratio plus a constant nobody can subtract afterwards.
    Deltas are taken per chunk for the second reason the precedent names too: the
    legs are stepped ALTERNATELY, so a delta from one early origin also counts the
    other legs' launches.
    """
    e2e._apply_env(leg)
    driver = leg["driver"]
    before = counter.snapshot()
    cuda_before = CUDA_COUNTER.snapshot() if CUDA_COUNTER is not None else None
    started = time.time()
    driver.run(num_steps=num_steps)
    elapsed = time.time() - started
    chunk = e2e.TritonLaunchCounter.delta(before, counter.snapshot())
    cuda_chunk = (e2e.CudaLaunchCounter.delta(cuda_before, CUDA_COUNTER.snapshot())
                  if CUDA_COUNTER is not None else None)
    running = leg.setdefault("triton_launches", {"total": 0, "hook_calls": 0,
                                                 "zero_grid_launches": 0,
                                                 "by_kernel": {}})
    running["total"] += chunk["total"]
    running["hook_calls"] += chunk["hook_calls"]
    running["zero_grid_launches"] += chunk["zero_grid_launches"]
    for name, count in chunk["by_kernel"].items():
        running["by_kernel"][name] = running["by_kernel"].get(name, 0) + count
    if cuda_chunk is not None:
        cuda_running = leg.setdefault("cuda_launches",
                                      {"total": 0, "memo_calls": 0,
                                       "zero_grid_launches": 0, "by_kernel": {}})
        cuda_running["total"] += cuda_chunk["total"]
        cuda_running["memo_calls"] += cuda_chunk["memo_calls"]
        cuda_running["zero_grid_launches"] += cuda_chunk["zero_grid_launches"]
        for name, count in cuda_chunk["by_kernel"].items():
            cuda_running["by_kernel"][name] = (cuda_running["by_kernel"].get(name, 0)
                                               + count)
    leg.setdefault("chunks", []).append({
        "steps": int(num_steps),
        "launches": chunk["total"], "hook_calls": chunk["hook_calls"],
        "zero_grid_launches": chunk["zero_grid_launches"],
        "launches_per_step": (chunk["total"] / num_steps) if num_steps else None,
        "hook_calls_per_step": (chunk["hook_calls"] / num_steps) if num_steps else None,
        "by_kernel": chunk["by_kernel"],
        # THE SECOND TABLE'S CHUNK, recorded on every run and consulted on a CUDA
        # one. Both tables can launch in the same step -- the Triton table serves
        # fills the CUDA table does not carry -- so these are two measurements of
        # one run and not two spellings of one measurement.
        "cuda_launches": cuda_chunk["total"] if cuda_chunk else None,
        "cuda_memo_calls": cuda_chunk["memo_calls"] if cuda_chunk else None,
        "cuda_zero_grid_launches": (cuda_chunk["zero_grid_launches"]
                                    if cuda_chunk else None),
        "cuda_launches_per_step": ((cuda_chunk["total"] / num_steps)
                                   if cuda_chunk and num_steps else None),
        "cuda_memo_calls_per_step": ((cuda_chunk["memo_calls"] / num_steps)
                                     if cuda_chunk and num_steps else None),
        "cuda_counts_agree": cuda_chunk["counts_agree"] if cuda_chunk else None,
        "cuda_by_kernel": cuda_chunk["by_kernel"] if cuda_chunk else None,
    })
    # THE MEMO WITNESS IS INSTALLED AFTER THE FIRST WARM, which is the ordering its
    # own docstring requires: it wraps what is ALREADY in the compile cache, and
    # before the first chunk that store is empty. Idempotent per kernel.
    if CUDA_COUNTER is not None and len(leg["chunks"]) == 1:
        CUDA_COUNTER.wrap_memo_store()
    leg["active_step_path"] = driver.active_step_path
    leg["plan"] = e2e._plan_summary(driver.fast_path_report())
    leg["report"] = dict(driver.fast_path_report() or {})
    leg["steps"] = int(driver.step_count)
    leg["wall_s"] = round(leg.get("wall_s", 0.0) + elapsed, 3)
    say(f"{case}/{leg['label']}: path={leg['active_step_path']} steps={leg['steps']} "
        f"+{num_steps} launches+{chunk['total']} "
        f"({chunk['total'] / num_steps if num_steps else 0:.2f}/step"
        + (f", cuda+{cuda_chunk['total']}" if cuda_chunk else "")
        + f", {elapsed:.1f} s)")


def steady_state_rate(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Launches per complete step, over every chunk AFTER the first.

    The first chunk carries the configuration freeze — the plan-time warm pass
    launches every kernel once over an EMPTY grid on purpose — and a rate computed
    over it is not the run's rate. Every later chunk is steady state.
    """
    chunks = leg.get("chunks") or []
    tail = chunks[1:]
    steps = sum(c["steps"] for c in tail)
    if not steps:
        return {"measured": False, "why": "the leg ran a single chunk"}
    triton = sum(c["launches"] for c in tail)
    hooks = sum(c["hook_calls"] for c in tail)
    cuda = sum((c.get("cuda_launches") or 0) for c in tail)
    memo = sum((c.get("cuda_memo_calls") or 0) for c in tail)
    # WHICH TABLE'S LAUNCHES THIS RUN IS ABOUT. Both are recorded; the substitution
    # proof compares the one whose products the run drives, because on a CUDA run
    # the Triton counter reads a constant that has nothing to do with the fusion
    # under test -- and a drop computed from it is a drop in someone else's kernels.
    # EVERY KERNEL LAUNCH THE SEAM MADE, WHICHEVER TABLE MADE IT. The substitution
    # claim is about the SEAM's launch count -- how many kernels one step costs with
    # the fusion and without it -- and on the hand-CUDA campaign the two legs need
    # not be served by the same table: until 2026-09-17 that table's single-sub-step
    # arms carried no launch (``cuda_kernels.arms.CudaSlotPlan`` was a plan, not a
    # launch), so with fused arms vetoed the merge gave every slot to the incumbent
    # and the baseline was a TRITON run; since then the two certified real-PML
    # singles launch and the merge adopts them as SEAMS, so a real-PML row's baseline
    # is this table's own (``unfused_baseline`` on the drive row), while every other
    # row's still is not. Counting only one table's witness produced a NEGATIVE drop --
    # measured pml_2d: fused 0 Triton + 4800 CUDA, unfused 9604 Triton + 0 CUDA --
    # and the gate reported NO-DROP on a leg that had halved the launch count.
    # Summing both witnesses gives 4.0/step unfused against 2.0/step fused, which is
    # the two fused pairs, and reduces to the old number exactly on a Triton run
    # where the CUDA counter reads zero throughout.
    #
    # THE SECOND WITNESS IS SUMMED THE SAME WAY, over the steady-state tail only:
    # the CUDA memo witness is installed after the first chunk populates the compile
    # cache, so it misses each kernel's first call and agrees exactly from chunk two.
    primary = triton + cuda
    secondary = hooks + memo
    return {"measured": True, "steps": steps,
            "tables": [name for name, count in (("triton", triton), ("cuda", cuda))
                       if count],
            "launches": primary, "hook_calls": secondary,
            "launches_per_step": primary / steps,
            "hook_calls_per_step": secondary / steps,
            "triton_launches": triton, "triton_hook_calls": hooks,
            "triton_launches_per_step": triton / steps,
            "cuda_launches": cuda, "cuda_memo_calls": memo,
            "cuda_launches_per_step": cuda / steps,
            "per_chunk": [((c["launches_per_step"] or 0)
                           + (c.get("cuda_launches_per_step") or 0))
                          for c in tail]}


def fusion_block(leg: Dict[str, Any]) -> Dict[str, Any]:
    """What the record says the fusion DID, read off the artifact rather than assumed."""
    report = leg.get("report") or {}
    fusion = dict(report.get("fusion") or {})
    slots = report.get("slots") or {}
    driven = dict(fusion.get("driven") or {})
    counters = report.get("launch_counters") or {}
    return {
        "opted_in": fusion.get("opted_in"),
        "driven": driven,
        "driven_slot_count": len(driven),
        "dispatched_slots": [n for n, e in slots.items() if e.get("state") == "dispatched"],
        "arms": {n: e.get("arm") for n, e in slots.items()
                 if e.get("state") == "dispatched"},
        "counters": counters,
        "families": {arm: (report.get("families") or {}).get(arm, {}).get("gate")
                     for arm in set(driven.values())},
        "refused_because": report.get("refused_because"),
        "decision": report.get("decision"),
        # THE COMPOSER'S OWN SEAM REFUSALS, copied so the artifact says WHY a product
        # did not install: the 2026-09-17_allpaths VACUOUS-PASS on
        # folded_complex_offdiag_2d had to be diagnosed off-device because this block
        # recorded the outcome (two slots, not four) and not the reason.
        "not_installed": fusion.get("not_installed"),
    }


def expected_substitution(spec: Dict[str, Any]) -> Dict[str, int]:
    """How many slots a case's fused units occupy, and how many launches that saves.

    ONE FUNCTION BECAUSE THE TWO NUMBERS ARE THE SAME FACT. A two-slot pair occupies
    two slots and saves one launch per step; a three-slot weld occupies three and
    saves one or two, according to whether its trailing group launches at all --
    see ``tail_in_leading`` below. Both the non-vacuity clause and the
    substitution proof read the shape, and when only one of them knew about triples
    a correctly composed plan failed one clause and passed the other -- which is the
    worst way for a gate to be wrong, because the disagreement looks like evidence.

    ``pairs`` counts UNITS, triple included; ``triple`` says one of them is the
    three-slot weld; ``tail_in_leading`` says that weld is the component-major
    shape whose trailing group launches nothing. All three are declared per case in
    the drive table rather than inferred from what dispatched: a clause that reads
    its own expectation off the measurement cannot fail.
    """
    pairs = int(spec["pairs"])
    triples = 1 if spec.get("triple") else 0
    # POLES, AND WHY THE DROP IS NOT ALWAYS ONE PER UNIT. The comment below was
    # written on ONE-POLE cases and is the P = 1 instance of a wider rule. The
    # unfused baseline for a DISPERSIVE CUDA case is the TRITON table (this table's
    # launchable singles are the two real-PML families, and neither admits a
    # susceptibility), and
    # Triton advances the ADE ONCE PER POLE PER COMPONENT -- 3P launches a step --
    # while the CUDA weld's trailing group loops the poles INSIDE one kernel and
    # launches 3 a step whatever P is. So the weld removes 3(P - 1) launches beyond
    # the one its unit saves.
    #
    # MEASURED on the GPU host 2026-09-14, over every dispersive case of the run, and
    # the fit is exact on all six rather than on the ones it was derived from:
    # dispersive_2d P=1 7 -> 5 (drop 2), folded_dispersive_2d P=1 9 -> 7 (2),
    # material_dispersion_0d P=2 9 -> 5 (4), dispersive5_2d P=5 19 -> 5 (14),
    # absorber_1d P=5 18 -> 5 (13), dispersive6_2d P=6 22 -> 5 (17).
    #
    # ``poles`` IS DECLARED PER CASE, for this function's own stated reason: a
    # clause that reads its own expectation off the measurement cannot fail. A row
    # that declares the wrong P still fails here, because the measured drop will
    # not match, and a weld that stopped collapsing its poles would fail too.
    poles = int(spec.get("poles", 1))
    # A TRIPLE OCCUPIES THREE SLOTS AND SAVES ONE LAUNCH, NOT TWO, and the
    # difference was measured rather than reasoned. The three-slot weld runs TWO
    # device groups: a leading one that welds ``step_D`` and ``update_E`` into a
    # single kernel, and a trailing one that advances ``update_P`` ONCE PER
    # COMPONENT. Measured on the GPU host 2026-09-11, dispersive_2d at 2200 steps:
    # ``dispersive_fused_electric_pair_pml_real`` 2200 launches (1/step) and
    # ``three_slot_polarization_real`` 6600 (3/step), against the unfused leg's
    # ``constitutive_step_dispersive`` 1/step, ``pml_curl_step`` 2/step and
    # ``ade_update_p`` 3/step. The trailing group is a one-for-one replacement of
    # the array ADE consult; only the leading weld removes a launch. So the saving
    # is one per UNIT, pair and triple alike.
    #
    # A TRIPLE WHOSE TRAILING GROUP LAUNCHES NOTHING SAVES ONE MORE, and the
    # paragraph above describes only the SEPARATE-TAIL shape. Written out, against
    # an unfused baseline of 3P + 2 (one curl, one constitutive, 3P ADE):
    #
    #   separate tail    leading welds step_D+update_E -> 1/step, trailing advances
    #                    update_P once per component -> 3/step.  4 replace 3P + 2,
    #                    so the drop is 3P - 2 -- which is what the expression below
    #                    already yields at one unit.
    #   component-major  leading welds ALL THREE sub-steps per component -> 3/step,
    #                    trailing launches NOTHING: complex_no_pml_three_slot_
    #                    dispersive_weld's ``launch_trailing`` returns
    #                    ``{"launched": False, "launches": 0}`` and says the
    #                    polarizations were advanced inside the leading group.
    #                    3 replace 3P + 2, so the drop is 3P - 1, exactly one more.
    #
    # MEASURED on the GPU host 2026-09-14, complex_no_pml_3d over 600 dispatches: the
    # unfused leg ran 6.0 launches a step (the step_B curl both legs share, plus
    # step_D, update_E and 3 ADE) against the fused leg's 4.0 (that same step_B curl
    # plus 3 welded), a drop of 2.0 with ``hook_drop_per_step`` 2.0 agreeing and the
    # slot sets identical -- where this function said 1 and the proof read
    # DROPPED-BUT-NOT-EXACT on a correct result.
    #
    # DECLARED, NOT DETECTED, for this function's stated reason. There is no honest
    # witness in the record for "the tail launched nothing": ``plan_launches`` reads
    # 600 on ``update_P`` for this very weld, because ``_TripleHalfPlan.run``
    # increments it whether or not its launcher reached the device. What keeps the
    # declaration non-vacuous is the same thing that keeps ``poles`` honest -- a row
    # that declares the wrong shape still fails here, because the measured drop will
    # not match it.
    tail_in_leading = 1 if (triples and spec.get("tail_in_leading")) else 0
    return {"slots": 2 * pairs + triples,
            "drop_per_step": pairs + 3 * (poles - 1) + tail_in_leading}


def fused_leg_is_real(leg: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Every clause below is a FAILURE if false. Non-vacuity, said four ways."""
    failures: List[str] = []
    block = fusion_block(leg)
    driven = block["driven"]
    counters = block["counters"]

    # WHICH TABLE FILLED THE DISPATCHED SLOTS. Read once, near the top, because the
    # shape clauses below and the launch witnesses further down both turn on it.
    composition = dict((leg.get("report") or {}).get("composition") or {})
    served_by = list(composition.get("tables_dispatched") or [])

    if leg["active_step_path"] != "fused":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if not driven:
        failures.append("the record names no driven fused arm")
    # A PAIR TAKES TWO SLOTS. One slot carrying a fused label is a fused arm that
    # did NOT absorb its partner, which is a composition nothing here measured.
    #
    # A TRIPLE TAKES THREE, AND THE ARITHMETIC USED TO SAY OTHERWISE. The CUDA
    # table's three-slot dispersive weld holds ``step_D``/``update_E``/``update_P``
    # in ONE unit, so a case carrying it drives 2*pairs + 1 slots. Counting it as a
    # pair made the clause fire on a correctly composed plan -- the case reported
    # "5 fused slots, expected 4" and the leg failed on the shape it was written to
    # measure. ``triple`` is declared per case in the drive table because it is a
    # property of the PRODUCT, not something to infer from the driven set (inferring
    # it from what dispatched would make the clause unable to fail).
    # THE SHAPE BELONGS TO THE TABLE THE DRIVE ENTRY DESCRIBES, so it is asserted
    # only on a leg that table served. ``pairs`` and ``triple`` say how the CUDA
    # composition occupies the seam; on ``cuda_default_precedence`` the shipped
    # precedence hands the slots to the incumbent, whose composition is legitimately
    # a different shape -- measured 2026-09-11 on dispersive_2d, where Triton's two
    # pairs fill 4 slots against this entry's 5, and on folded_dispersive_2d, where
    # it fills 2. Asserting the absent table's shape there failed a leg for
    # measuring exactly what it exists to measure. The count is still RECORDED on
    # every leg, so the by-default composition is readable rather than inferred.
    expected_slots = expected_substitution(spec)["slots"]
    own_table_served = (BACKEND != "cuda") or ("cuda" in served_by) or not served_by
    if own_table_served and len(driven) != expected_slots:
        failures.append(f"the record names {len(driven)} fused slots, expected "
                        f"{expected_slots} ({spec['pairs']} unit(s) x 2"
                        + (" + 1 for the three-slot weld" if spec.get("triple")
                           else "") + ")")
    for slot, arm in sorted(driven.items()):
        entry = counters.get(slot) or {}
        if not entry.get("dispatches"):
            failures.append(f"{slot} carries {arm!r} but its consult ran "
                            f"{entry.get('dispatches')} times")
    # PROGRAMS on the LEADING slot only: the absorbed slot holds a sentinel or the
    # repair, neither of which spells a launch grid, and None there is unknown
    # rather than zero (fastpath.launch_counters says so in as many words).
    leading = [s for s in driven if s in ("step_B", "step_D")]
    for slot in leading:
        programs = (counters.get(slot) or {}).get("programs_per_dispatch")
        if programs == 0:
            failures.append(f"{slot} launched over an EMPTY grid: 0 programs")
        if programs is None:
            failures.append(f"{slot} spells no launch grid, so 'a kernel computed' "
                            "cannot be told from 'a kernel launched'")
    # THE LAUNCH WITNESSES OF THE TABLE THIS RUN DRIVES. Two of them, because one
    # counter that cannot be wrong is worth less than two that must agree.
    #
    # ASKING THE WRONG TABLE IS NOT A CONSERVATIVE MISTAKE. On a CUDA run the
    # Triton counter reads zero when the hand-CUDA kernels launch perfectly, and
    # this clause turned a correct dispatch into VACUOUS-PASS; the mirror error
    # would be worse, since a Triton run reads zero on the CUDA witness and the
    # clause would pass anything.
    #
    # AND "THE TABLE THIS RUN DRIVES" IS THE ONE THE MERGE FILLED THE SLOTS FROM,
    # not the one the preference names. The ``cuda_default_precedence`` leg exists to
    # measure what dispatches when the user sets NOTHING, and under the shipped
    # precedence that is the release-gated Triton table holding every slot both admit
    # -- which is the leg's RESULT, not its failure. Measured 2026-09-11: all ten
    # dispatching cases ran fused at 2.0 launches/step against the unfused leg's 4.0,
    # an EXACT substitution, and the leg reported VACUOUS-PASS on every one of them
    # because the clauses below asked a CUDA witness that correctly read zero.
    if BACKEND == "cuda" and "cuda" in served_by:
        cuda = leg.get("cuda_launches") or {}
        # THE RECORD'S OWN TABLE CLAIM, CHECKED AGAINST THE MERGE'S PER-SLOT ANSWER.
        # ``composition.table`` names the PRIMARY table, which under
        # MEEP_GPU_BACKEND_PREFERENCE=cuda is "cuda" on every plan -- including one
        # where the CUDA table offered nothing launchable and Triton filled every
        # dispatched slot. Measured on pml_3d, 2026-09-11: that key read "cuda"
        # beside a ``tables_dispatched`` of ``["triton"]`` and three Triton kernels
        # as the only launches. The launch clauses below catch the substance; this
        # one names the mis-statement, so an artifact carrying it cannot be read as
        # a CUDA dispatch by anything downstream that trusts the claim.
        claimed = composition.get("table")
        ran = served_by
        if claimed and ran and claimed not in ran:
            failures.append(f"the record's composition claims table {claimed!r} "
                            f"while the merge filled the dispatched slots from "
                            f"{ran}: the claim is about the primary table and not "
                            "about what ran")
        if not (CUDA_COUNTER is not None and CUDA_COUNTER.snapshot()["installed"]):
            failures.append("no hand-CUDA launch witness was installed, so 'a "
                            "kernel ran' cannot be told from 'a kernel was planned'")
        if not cuda.get("total"):
            failures.append("no hand-CUDA kernel launched at all "
                            "(compile_cache.get_or_compile proxy)")
        if not cuda.get("memo_calls"):
            failures.append("the compile-cache memo witness counted nothing, so "
                            "the proxy total has no second witness to agree with")
        else:
            # AGREEMENT IS ASKED OF THE STEADY STATE, NOT THE ACCUMULATED TOTAL.
            # The memo witness wraps what is ALREADY in the compile cache, so it
            # cannot be installed until the first chunk has populated it; every
            # kernel's FIRST call is therefore seen by the proxy and not by the memo,
            # and the two totals differ by exactly the number of distinct kernels.
            # Measured 2026-09-11 on pml_2d: proxy 4800, memo 4798, two kernels --
            # reported as a disagreement, which is a true statement about the totals
            # and a false one about the instruments. Every chunk after the first has
            # both witnesses on the path and must agree exactly.
            tail = (leg.get("chunks") or [])[1:]
            disagreeing = [i + 2 for i, c in enumerate(tail)
                           if c.get("cuda_counts_agree") is False]
            if disagreeing:
                failures.append(
                    f"the two CUDA witnesses disagree on steady-state chunk(s) "
                    f"{disagreeing}: one of them is not on the path the launches "
                    f"took")
            if not tail:
                failures.append("the leg ran a single chunk, so no steady-state "
                                "chunk exists on which the two CUDA witnesses "
                                "could be required to agree")
    else:
        if BACKEND == "cuda":
            # A CUDA RUN THE INCUMBENT SERVED. The clauses are the incumbent's, and
            # WHICH TABLE SERVED IS RECORDED, because that is the whole content of
            # the by-default number: a reader must never have to infer it from the
            # backend the run was launched under.
            failures.extend(
                [] if served_by else
                ["the record names no dispatching table at all, so 'which table "
                 "served' -- the only thing this leg measures -- is unreadable"])
        if not leg["triton_launches"]["total"]:
            failures.append("no Triton kernel launched at all "
                            "(JITFunction.run counter)")
        if not leg["triton_launches"]["hook_calls"]:
            failures.append("no Triton launcher hook fired")
    # A FOLDED CASE THAT DOES NOT RUN A FILL IN-LAUNCH IS VACUOUS, and this clause is
    # the reason the two folded cases are in the set at all. What the 2026-09-02
    # consults changed is that ``fill_B``/``fill_D`` can be SERVED FROM A KERNEL;
    # rung (6b) refuses a fold whose fills are on the array path while its curls are
    # not, so a "folded case dispatched" that left both fills on the host would be
    # measuring a composition the ladder is supposed to refuse — and would look
    # identical, byte for byte, to the one it is supposed to allow. The consult must
    # have RUN (``dispatches``) and the slot must be dispatched, both read off the
    # artifact rather than off the plan this file built.
    if spec.get("fills"):
        dispatched = set(block["dispatched_slots"])
        for slot in spec["fills"]:
            if slot not in dispatched:
                failures.append(f"{slot} is not dispatched, so the fold ran its "
                                f"ghost fill on the array path while its curls ran "
                                f"on kernels")
                continue
            entry = counters.get(slot) or {}
            if not entry.get("dispatches"):
                failures.append(f"{slot} is dispatched but its consult ran "
                                f"{entry.get('dispatches')} times")
        far = leg.get("report", {}).get("launch_counters") or {}
        block["far_fill_consults"] = {
            name: (far.get(name) or {}).get("dispatches")
            for name in ("fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D")}
    return {"real": not failures, "failures": failures,
            "slots_driven": len(driven), "slots_expected_of_this_table": expected_slots,
            # WHICH TABLE FILLED THE DISPATCHED SLOTS, on every leg. The
            # by-preference and by-default numbers differ by exactly this, and a
            # record that leaves it to be inferred from the backend the run was
            # launched under is a record that can be read as either.
            "served_by_table": served_by, **block}


#: The four slots a two-slot fused pair or a certified single seam may hold; the
#: ``unfused_baseline`` clause of the substitution proof is a statement about these.
SEAM_SLOTS = ("step_B", "update_H", "step_D", "update_E")


def substitution_proof(fused: Dict[str, Any], unfused: Dict[str, Any],
                       spec: Dict[str, Any]) -> Dict[str, Any]:
    """One launch fewer per pair per step, counted twice over.

    THE CLAIM FUSION MAKES, and the only one no byte comparison can reach: two
    sub-steps became one launch. Both counts are taken at Triton's own entry
    points, independently of anything this package's bookkeeping says, and the
    comparison is against a THIRD leg that dispatched the same slots with fusion
    off — not against the array path, which launches nothing and would make any
    number look like a saving.

    THE SLOT SETS MUST MATCH before the arithmetic means anything. If the unfused
    leg dispatched a different set, the delta mixes the fusion with a coverage
    difference and the claim is recorded as not-comparable rather than asserted.

    AND THE BASELINE MUST ACTUALLY BE UNFUSED, read out of ITS OWN RECORD rather
    than inferred from the environment it was handed. That is not defensive
    padding: on 2026-08-29 the environment said "no arms named" and the record
    said ``driven: {step_B: fused pair B, ...}``, because the release had changed
    what an unset switch means. The delta was then a true 0.0 between two
    identical legs, and only the record could say why.
    """
    a, b = steady_state_rate(fused), steady_state_rate(unfused)
    # THE EXPECTED DROP IS ONE PER FUSED UNIT OF THE COMPOSITION THAT RAN. Normally
    # that is the drive entry's ``pairs``, which describes this table's composition.
    # On a leg the INCUMBENT served the entry is about an absent table: measured
    # 2026-09-11 on folded_dispersive_2d under the unset preference, Triton holds ONE
    # released fused arm on that shape and saved exactly 1 launch per step against
    # this entry's 2, and the proof read DROPPED-BUT-NOT-EXACT on a correct result.
    # Counting the distinct fused labels the leg actually drove keeps the clause
    # exact and non-vacuous -- a leg that fused nothing has no units and never
    # reaches here, and one that fused N units must still save exactly N.
    fused_block = fusion_block(fused)
    units_driven = len({label for label in (fused_block["driven"] or {}).values()
                        if label})
    own_table_ran = (BACKEND != "cuda"
                     or "cuda" in list(((fused.get("report") or {})
                                        .get("composition") or {})
                                       .get("tables_dispatched") or []))
    expected_drop = (expected_substitution(spec)["drop_per_step"] if own_table_ran
                     else units_driven)
    out: Dict[str, Any] = {"fused": a, "unfused": b,
                           "expected_drop_read_from": (
                               "the drive entry" if own_table_ran
                               else f"the {units_driven} fused unit(s) this leg drove"),
                           "expected_drop_per_step": expected_drop,
                           # WHICH TABLE'S KERNELS EACH LEG RAN, disclosed because
                           # on this seam they need not be the same one. See the
                           # docstring's fourth paragraph.
                           "fused_tables": a.get("tables"),
                           "unfused_tables": b.get("tables")}
    baseline = dict((unfused.get("report") or {}).get("fusion") or {})
    out["baseline_vetoed_fusion"] = baseline.get("vetoed")
    out["baseline_driven"] = dict(baseline.get("driven") or {})
    # THE SUBSTITUTION IS TABLE-AGNOSTIC AND STAYS THAT WAY. It counts every kernel
    # launch the seam made, whichever table made it, so on
    # ``cuda_default_precedence`` it measures the INCUMBENT's saving -- which is a
    # real measurement and part of the by-default answer, not a gap. Scoping it to
    # the run's own table was tried on 2026-09-11 and was wrong in the direction that
    # matters: it turned pml_2d's correct 4.0 -> 2.0 drop into NOT-APPLICABLE and
    # then failed the case at the ladder's substitution rung. A case where NOTHING
    # fused never reaches here -- the by-default verdict above returns first.
    if not spec["pairs"]:
        # A CASE THAT IS HERE TO BE REFUSED SUBSTITUTES NOTHING, and reporting a
        # zero drop as EXACT would put a green word on a measurement that was never
        # taken. Measured 2026-08-29: the first campaign did exactly that on
        # folded_2d.
        out["verdict"] = "NOT-APPLICABLE"
        out["why"] = "this case expects a refusal, so no launch is substituted"
        return out
    if not (a.get("measured") and b.get("measured")):
        out["verdict"] = "NOT-MEASURED"
        return out
    # THE BASELINE'S TABLE IS DECLARED WHERE THE ROW KNOWS IT. Since the merge adopts
    # certified launchable single SEAMS of the hand-CUDA table, a real-PML row's veto
    # leg under MEEP_GPU_BACKEND_PREFERENCE=cuda is served by that table's own
    # singles, and the fusion number is measured against the kernels the product
    # replaces rather than against another table's. A row that declares
    # ``unfused_baseline`` is checked against the veto leg's own record on every leg
    # where this table's products ran; a row that does not declare it keeps the
    # incumbent baseline and claims nothing about which table served it.
    # THE CLAIM IS ABOUT THE FOUR SEAM SLOTS, read off the veto leg's dispatched arms
    # (a ``cuda:`` label is that table's, by the merge's own namespacing), and not off
    # the launch-count table list: on folded_2d the two mirror fills are Triton's on
    # every leg, so a table list would read ['cuda', 'triton'] on a seam that is
    # entirely this table's.
    declared_baseline = spec.get("unfused_baseline")
    if own_table_ran and declared_baseline:
        seam_arms = {slot: arm for slot, arm in fusion_block(unfused)["arms"].items()
                     if slot in SEAM_SLOTS}
        served_by = {slot: ("cuda" if str(arm).startswith("cuda:") else "incumbent")
                     for slot, arm in seam_arms.items()}
        out["unfused_seam_arms"] = seam_arms
        if (set(seam_arms) != set(SEAM_SLOTS)
                or set(served_by.values()) != {declared_baseline}):
            out["verdict"] = "BASELINE-TABLE-MISMATCH"
            out["why"] = (f"the drive row declares its unfused baseline is the "
                          f"{declared_baseline!r} table and the veto leg's own record "
                          f"shows the seam slots served as {seam_arms}: the fusion "
                          "number would be measured against another table's kernels")
            return out
    fused_slots = sorted(fusion_block(fused)["dispatched_slots"])
    unfused_slots = sorted(fusion_block(unfused)["dispatched_slots"])
    out["fused_dispatched_slots"] = fused_slots
    out["unfused_dispatched_slots"] = unfused_slots
    out["same_slot_set"] = fused_slots == unfused_slots
    # A SLOT THE BASELINE LEFT ON THE ARRAY PATH AND THE FUSED UNIT ABSORBED. Measured
    # 2026-09-17 on complex_no_pml_offdiag: the weld held step_D/update_E, the veto
    # leg held step_D alone (the only Triton candidate for update_E is PENDING and
    # the CUDA single carries no resolver), both legs ran 2.0 launches a step, and
    # the strict slot-set clause read NOT-COMPARABLE on a correct result. The delta
    # there MIXES -1 from the fusion with +1 from absorbing a slot the baseline never
    # launched a kernel for, and both records are here to un-mix it: an absorbed slot
    # must lie inside a unit the fused leg drove, must be DECLARED on the drive row
    # (``baseline_array_slots``) so the row goes red the day the baseline serves it,
    # and lowers the COUNTED expectation by one per slot. The uncounted saving --
    # that slot's array-path elementwise kernels -- is named in the verdict rather
    # than credited, because no witness in this gate measures it.
    absorbed = sorted(set(fused_slots) - set(unfused_slots))
    lost = sorted(set(unfused_slots) - set(fused_slots))
    declared_absorbed = sorted(spec.get("baseline_array_slots") or [])
    driven_slots = set(fused_block["driven"] or {})
    out["absorbed_array_slots"] = absorbed
    # Recorded for :func:`tritonless_baseline_only`, which reads the proof's own
    # facts rather than re-deriving them from its prose.
    out["lost_slots"] = lost
    out["driven_slots"] = sorted(driven_slots)
    out["declared_baseline_array_slots"] = declared_absorbed
    out["expected_drop_declared"] = expected_drop
    drop = b["launches_per_step"] - a["launches_per_step"]
    hook_drop = b["hook_calls_per_step"] - a["hook_calls_per_step"]
    out["launch_drop_per_step"] = drop
    out["hook_drop_per_step"] = hook_drop
    out["counts_agree"] = abs(drop - hook_drop) < 1e-9
    if out["baseline_driven"] or out["baseline_vetoed_fusion"] is not True:
        # A BASELINE THAT FUSED IS NOT A BASELINE. Reported as its own verdict and
        # not folded into NO-DROP, because the two have opposite fixes: NO-DROP
        # says the fusion saved nothing, this says nothing was compared.
        out["verdict"] = "BASELINE-FUSED"
        out["why"] = ("the unfused leg's own record shows "
                      f"vetoed={out['baseline_vetoed_fusion']!r} and driven="
                      f"{out['baseline_driven']}, so the delta is between two "
                      "fused legs and measures nothing about substitution")
    elif lost or not set(absorbed) <= driven_slots:
        outside = sorted(set(absorbed) - driven_slots)
        out["verdict"] = "NOT-COMPARABLE"
        out["why"] = ("the two legs dispatched different slot sets, so the launch "
                      "delta mixes fusion with a coverage difference"
                      + (f": the veto leg dispatched {lost} and the fused leg did not"
                         if lost else "")
                      + (f": the fused leg dispatched {outside} outside any unit it "
                         "drove" if outside else ""))
    elif absorbed != declared_absorbed:
        out["verdict"] = "NOT-COMPARABLE"
        out["why"] = (f"the fused unit absorbed {absorbed}, which the veto leg left "
                      f"on the array path, and the drive row declares "
                      f"baseline_array_slots={declared_absorbed}; an absorbed slot is "
                      "declared, not detected, so the row fails the day the baseline "
                      "serves it")
    else:
        counted = expected_drop - len(absorbed)
        out["expected_drop_per_step"] = counted
        if abs(drop - counted) < 1e-9 and out["counts_agree"]:
            if absorbed:
                out["verdict"] = "EXACT-ABSORBED-ARRAY-SLOT"
                out["why"] = (f"the counted-launch saving is {counted}: "
                              f"{expected_drop} per the drive row's units minus "
                              f"{len(absorbed)} for {absorbed}, which the veto leg "
                              "served on the array path and launched no kernel for; "
                              "the uncounted saving is that slot's array-path "
                              "elementwise kernels, which no witness in this gate "
                              "measures")
            else:
                out["verdict"] = "EXACT"
        elif drop > 0:
            out["verdict"] = "DROPPED-BUT-NOT-EXACT"
        else:
            out["verdict"] = "NO-DROP"
    return out


#: The substitution verdicts a dispatching case may carry. ``EXACT`` is one launch
#: fewer per unit per step against a baseline that launched a kernel in every slot
#: the unit holds; the second is the same proof on a row whose drive entry DECLARES a
#: slot the baseline left on the array path, and it is spelled differently so a
#: counted drop of 0 never reads as a plain pass.
EXACT_SUBSTITUTION_VERDICTS = ("EXACT", "EXACT-ABSORBED-ARRAY-SLOT")


# ---------------------------------------------------------------------------
# The armed controls
# ---------------------------------------------------------------------------

#: The slots whose array arm is the CONSTITUTIVE sub-step — the only ones the
#: withheld-consult mechanism can put an array call on top of. ``update_P``'s array
#: arm advances the polarization and is not a constitutive recomputation at all,
#: which is why a slot holding a device group is left running rather than withheld.
_CONSTITUTIVE_SLOTS = ("update_E", "update_H")


def _slot_launches(plan: Any) -> bool:
    """Does the plan sitting in this slot perform a DEVICE LAUNCH of its own?

    Asked of the object, not of its class name. Every launching plan in the tree
    keeps its own integer ``launches`` counter and increments it in ``run``
    (``cuda_kernels/fused_pairs.CudaFusedPairPlan``, ``CudaFusedTriplePlan``,
    ``_TripleHalfPlan``); every sentinel and every repair wrapper is ``__slots__``-ed
    without one (``NoopPlan``, ``TrailingRepairPlan``, ``LeadingRepairPlan``,
    ``metal_kernels.launch.SyncedPlan``). So the counter IS the structural
    distinction, and a product installed with a shape this gate has not seen is
    classified by what it does rather than by a name someone remembered to add here.

    ``SyncedPlan`` forwards unknown attributes to its ``inner``, so a Metal-wrapped
    launcher answers True through it. That is the right answer and it changes
    nothing: a wrapped launcher only ever sits in a slot the ``absorbed_by is
    _unwrapped_inner`` test already excluded.
    """
    return isinstance(getattr(plan, "launches", None), int)


def _launches_of(plan: Any) -> Optional[int]:
    """That counter's value, or ``None`` where there is none to read."""
    value = getattr(plan, "launches", None)
    return value if isinstance(value, int) else None


def _withhold_absorbed_consult():
    """Force every ABSORBED consult to answer False. Returns (restore, state).

    THE ARMED NULL. With the second consult answering False the driver runs the
    array ``update_H``/``update_E`` ON TOP of the constitutive half the fused
    launch already computed. ``stepping._apply_constitutive_pml`` ACCUMULATES —
    ``field += kps*fw; field -= kms*fw_previous`` — so that is not a redundant
    recomputation but a wrong answer wherever the PML is graded, which is what
    ``FastPathPlan.dispatch``'s ``_committed`` clause exists to prevent. If this
    control does NOT diverge, the byte comparison cannot see whether the fused
    kernel's second half ran, and every clean result above it is unfalsifiable.

    "ABSORBED" IS NOT "NAMES AN ABSORBER", and the difference decides what this
    control measures. THREE plan shapes carry ``absorbed_by``:

    * ``launch.NoopPlan`` (:1673) — the sentinel in the absorbed slot of a clean
      pair. Launches nothing; ``absorbed_by`` is the pair that did the work.
    * ``deposit_repair.TrailingRepairPlan`` (:698) — the same slot when the seam
      carries a deposit. Runs the repair, not the constitutive half.
    * ``deposit_repair.LeadingRepairPlan`` (:668) — the LEADING slot of a
      repair-carrying pair, where ``absorbed_by is self.inner`` and ``run`` calls
      ``self.inner.run``. IT LAUNCHES.

    Withholding the third suppresses the launch itself, which pushes the run apart
    for a second reason and makes a divergence ambiguous about which mechanism
    produced it. Measured 2026-08-29: the first campaign withheld ``step_D`` on
    ``pml_2d`` and ``dispersive_2d`` for exactly that reason. So the leading repair
    slot is EXCLUDED here and recorded under ``skipped_leading`` — the discriminator
    is structural (``absorbed_by is`` the plan's own ``inner``) rather than a slot
    name, so a pair installed the other way round is still classified correctly.

    A FOURTH SHAPE LAUNCHES AND THE ``absorbed_by is inner`` TEST CANNOT SEE IT, and
    that is what this function got wrong until 2026-09-14. A three-slot weld is ONE
    product with TWO device groups (``cuda_kernels/fused_pairs.CudaFusedTriplePlan``),
    and ``_install_fused_triple`` puts ``triple.trailing`` — a ``_TripleHalfPlan``,
    which LAUNCHES group 2 — straight into ``update_P``. Its ``absorbed_by`` is the
    TRIPLE and it has no ``inner`` at all, so the leading test reads False and the
    slot was withheld like a sentinel: the second group never launched, and the
    "fused" leg was the array leg wearing a fused label. The same reading suppresses
    GROUP ONE on a triple whose seam carries no deposit, where ``plans[curl_name]``
    is ``triple.leading`` bare.

    MEASURED, on the 2026-09-14 CUDA shipped leg: ``absorber_1d`` and
    ``material_dispersion_0d`` — the two cases driving
    ``cuda:no-absorber three-slot dispersive weld`` — were the ONLY two of the
    twenty-one dispatching cases reporting NULL-DID-NOT-DIVERGE. Every other case
    diverged because its layer is ACTIVE and the array ``update_E`` accumulates on
    top; the PML triples (``dispersive_2d``, ``dispersive5_2d``,
    ``folded_dispersive_2d``) diverged for that reason WITH THEIR SECOND GROUP
    EQUALLY SUPPRESSED, which is a pass on a composition the control does not
    describe. On the two no-absorber cases the layer is inactive, the array
    ``update_E`` is a pure overwrite (``deposit_repair._apply_plain``), and with
    group 2 suppressed as well the leg collapsed exactly onto the array leg.

    SO THE DISCRIMINATOR IS "DOES THIS SLOT LAUNCH", asked of the plan rather than of
    its class name: every launching plan in the tree carries its own integer
    ``launches`` counter (``CudaFusedPairPlan``, ``CudaFusedTriplePlan``,
    ``_TripleHalfPlan``) and no sentinel or repair wrapper defines one —
    ``NoopPlan.__slots__`` is ``("absorbed_by", "sub_step")``,
    ``TrailingRepairPlan.__slots__`` is ``("slot", "absorbed_by", "_leading",
    "_fields", "_pml")``. Such a slot is recorded under ``skipped_launching``, kept
    SEPARATE from ``skipped_leading`` so a reader can see the trailing group was left
    running rather than infer it.

    Patched at the CLASS, not the instance: ``FastPathPlan`` is a frozen
    dataclass. The leg that runs under it is stepped on its own, never
    interleaved, so the patch cannot reach another leg.
    """
    from meep_gpu import fastpath as fp

    real = fp.FastPathPlan.dispatch
    state = {"withheld": 0, "slots": {}, "skipped_leading": {},
             "skipped_launching": {}}

    def _unwrapped_inner(plan: Any) -> Any:
        """A plan's own launcher, with any residency wrapper taken off.

        THE IDENTITY THIS CONTROL DISCRIMINATES ON IS BREAKABLE BY A WRAPPER, and on
        the Metal route it is broken. ``deposit_repair`` and ``withdraw_hoist`` set
        ``absorbed_by = inner`` at construction to mark the LEADING slot -- the one
        that names its own launcher -- and ``metal_kernels.launch.synced`` then
        replaces ``plan.inner`` with a ``SyncedPlan`` around it WITHOUT updating
        ``absorbed_by``. The test then reads False on a leading slot (its
        ``absorbed_by`` is the raw launcher, its ``inner`` the wrapper) and True on a
        plain wrapped pair (``SyncedPlan`` sets ``absorbed_by`` to its own inner), so
        the two branches swap.
        
        WHAT THAT COST, measured 2026-09-11 on the first Metal route campaign: the
        control withheld the LEADING consult, no kernel launched at all, and the
        "fused" leg was the array leg -- so cylindrical, whose only pair is
        step_D/update_E, compared two identical array runs and reported
        NULL-DID-NOT-DIVERGE, while the cases that did diverge diverged off their
        magnetic pair alone. Unwrapping restores the intended discrimination without
        touching ``launch.synced``, which is pinned by 39 of the 44 Metal welds.
        """
        inner = getattr(plan, "inner", None)
        seen = 0
        while inner is not None and seen < _WRAPPER_CHAIN_LIMIT:
            nested = getattr(inner, "inner", None)
            if nested is None or type(inner).__name__ != "SyncedPlan":
                return inner
            inner, seen = nested, seen + 1
        return inner

    def patched(self, slot, fields):  # noqa: ANN001
        if slot in getattr(self, "slots", ()):
            plan = self.step_plan.plans.get(slot)
            absorbed_by = getattr(plan, "absorbed_by", None)
            if absorbed_by is not None:
                if absorbed_by is _unwrapped_inner(plan):
                    # The leading repair slot: it names its own launcher.
                    state["skipped_leading"][slot] = (
                        state["skipped_leading"].get(slot, 0) + 1)
                elif _slot_launches(plan):
                    # A device group of its own sits here — a triple's trailing
                    # half, or a bare triple half in a seam with no deposit.
                    # Withholding it does not let an array sub-step run ON TOP of a
                    # launch; it DELETES the launch, which is the opposite
                    # experiment.
                    state["skipped_launching"][slot] = (
                        state["skipped_launching"].get(slot, 0) + 1)
                else:
                    state["withheld"] += 1
                    state["slots"][slot] = state["slots"].get(slot, 0) + 1
                    return False
        return real(self, slot, fields)

    fp.FastPathPlan.dispatch = patched

    def restore() -> None:
        fp.FastPathPlan.dispatch = real

    return restore, state


#: The value the mutated in-seam pass writes into the wall plane. A wipe writes
#: zero and nothing else, so any nonzero constant is a value the pass the fused
#: kernel performed in-launch cannot have produced.
#: How many residency wrappers this gate will unwrap before it stops looking. A
#: bound rather than a while-True because a cycle here would hang the control rather
#: than fail it, and one wrapper is all any shipped route installs.
_WRAPPER_CHAIN_LIMIT = 8

SEAM_MUTATION_VALUE = 0.5


def _mutate_in_seam_pass(side: str):
    """Make ``zero_metal_{side}`` NON-IDEMPOTENT. Returns (restore, state).

    WHY THIS IS THE RIGHT MUTATION. Byte identity on a metallically walled grid
    rests on a composition no gate had run: the fused kernel performs the wall
    wipe IN-LAUNCH (``FusedPairPlan.zero_metal``) and the driver then runs the
    ARRAY wipe on top of the launch, unconditionally, because
    ``driver.step`` puts it outside every consult. The claim "that re-run changes
    nothing" is a NULL. Replacing the wipe with a write the launch cannot have
    produced turns the null into a positive: on the ARRAY leg the mutated plane
    is read by the array ``update_H``/``update_E`` that follows, and on the FUSED
    leg that read already happened inside the launch, so the two must part.

    A control that stayed identical under this mutation would mean the comparison
    is blind to what the in-seam pass does between the two consults, which is
    exactly the composition the fold rung (6b) refuses on.
    """
    import meep_gpu.driver as drv

    name = f"zero_metal_{side}"
    real = getattr(drv, name)
    targets = ("Bx", "By", "Bz") if side == "B" else ("Dx", "Dy", "Dz")
    state = {"calls": 0, "written": []}

    def mutated(fields):  # noqa: ANN001
        real(fields)
        state["calls"] += 1
        for component in targets:
            array = getattr(fields, component, None)
            if array is None:
                continue
            # THE MUTANT IS A HOST WRITE, and under a held Metal residency a host write
            # to a sealed array raises instead of landing; on the 2026-09-23 preflight
            # every held case's in-seam control read ERROR for exactly that. Declaring
            # it makes the mutation land the way a real host pass would -- synced down,
            # written, synced back up -- so the control still measures what it exists
            # to measure: that the run DIVERGES.
            from meep_gpu import host_writes  # noqa: PLC0415 - late, as elsewhere
            host_writes.acquire(array)[0, :, :] = SEAM_MUTATION_VALUE
            if component not in state["written"]:
                state["written"].append(component)

    setattr(drv, name, mutated)

    def restore() -> None:
        setattr(drv, name, real)

    return restore, state


def _pair_of_lifted(case: str, arms: List[str], gpu_id: int,
                    res: Optional[int], counter: Any,
                    label_a: str = "probe",
                    reached_by: str = "opt-in") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # THE CONTROLS TAKE THE SAME ROUTE AS THE CASE. An armed control that reached
    # its fused launch through the opt-in while the case reached it through the
    # release would be arming a different composition than the one it is a control
    # for -- the same pair, but admitted by a different clause.
    builder = CASES[case]
    a = e2e.run_leg(case, builder, label_a, _fuse_env(arms, reached_by),
                    counter, gpu_id, res)
    b = e2e.run_leg(case, builder, "array", ARRAY_ENV, counter, gpu_id, res)
    return a, b


def _slot_plans(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen step plan's slot mapping for this leg, or ``{}`` if there is none.

    The same object ``_withhold_absorbed_consult``'s patch reads, reached the same
    way, so the two cannot disagree about what sits in a slot.
    """
    plan = getattr(leg.get("driver"), "_fast_path", None)
    return dict(getattr(getattr(plan, "step_plan", None), "plans", None) or {})


def _launching_slots_ran(leg: Dict[str, Any], skipped: Dict[str, int],
                         steps: int) -> Dict[str, Any]:
    """Did every device group this control LEFT RUNNING actually launch? Measured.

    THE POSITIVE HALF OF THE 2026-09-14 REPAIR. Once a trailing device group is no
    longer withheld, the row owes the reader proof that it ran — otherwise "we left
    it running" is an assertion about the patch rather than about the leg. The
    product's OWN counter answers it: ``_TripleHalfPlan.run`` increments
    ``launches`` on every call, the leg is lifted fresh for this control and stepped
    exactly ``steps`` times, so a per-step group must show at least one launch per
    consult it was reached on.

    Every failure is a STRING rather than an exception: a control that raises loses
    the twelve steps it already paid for, and an unreadable plan is itself a finding.
    """
    out: Dict[str, Any] = {"slots": sorted(skipped), "deltas": {},
                           "consults": dict(skipped), "steps": steps,
                           "failures": []}
    if not skipped:
        return out
    plans = _slot_plans(leg)
    for slot, consults in sorted(skipped.items()):
        plan = plans.get(slot)
        launches = _launches_of(plan)
        out["deltas"][slot] = launches
        if plan is None:
            out["failures"].append(
                f"{slot} was left running as a device group but the frozen plan "
                f"carries nothing in that slot, so nothing can say whether it ran")
        elif launches is None:
            out["failures"].append(
                f"{slot} was left running as a device group and the plan sitting "
                f"there spells no launch counter, so 'the group ran' cannot be told "
                f"from 'the consult was reached'")
        elif launches < consults:
            out["failures"].append(
                f"{slot} was consulted {consults} times over {steps} steps and its "
                f"device group launched {launches} times: the fused product's "
                f"second half did not run on every step this control left it free "
                f"to")
    return out


def _constitutive_is_an_overwrite(leg: Dict[str, Any]) -> Optional[bool]:
    """Is this leg on the plain constitutive branch? ``None`` when it cannot be read.

    ``stepping._pml_is_active`` is the fork the engine itself takes, so it is the
    fork this gate asks about rather than a re-derivation from the case's
    declarations: a layer that is present but inactive reads False there and takes
    the pure-overwrite arm, which is exactly the configuration ``absorber_1d``
    carries (an ``mp.Absorber`` is a conductivity, not a split-field PML).

    FAIL CLOSED. ``None`` — an unreadable driver or a raising predicate — is not
    ``False`` and not ``True``: the caller grants no NOT-APPLICABLE on it, because a
    control silenced on ignorance is the failure this whole file is built against.
    """
    from meep_gpu import stepping as _stepping  # noqa: PLC0415 - late, as elsewhere

    driver = leg.get("driver")
    if driver is None:
        return None
    try:
        return not _stepping._pml_is_active(driver.pml)  # noqa: SLF001
    except Exception:  # noqa: BLE001 - an unreadable layer is "unknown", not "plain"
        return None


def withheld_control(case: str, spec: Dict[str, Any], gpu_id: int,
                     res: Optional[int], counter: Any) -> Dict[str, Any]:
    """The absorbed consult forced False. MUST diverge — WHERE THE MECHANISM EXISTS.

    THE MECHANISM IS AN ACCUMULATION, and it is worth saying which branch carries it
    because on the other branch this control cannot be armed at all.
    ``stepping.update_E`` forks on ``_pml_is_active``:

    * ACTIVE — the split-field arm, ``field += kps*fw; field -= kms*fw_previous``
      (``_apply_constitutive_pml``). A second application is a WRONG ANSWER, so
      letting the array call run on top of the kernel's own constitutive half MUST
      part the two legs, and a run where it does not is a run whose byte comparison
      cannot see whether the kernel's second half ran.
    * INACTIVE — the plain arm, ``E = constitutive``, a pure overwrite
      (``stepping.py:1019-1022``, and ``deposit_repair._apply_plain`` inverts exactly
      that). A second application is the IDENTITY, whatever the kernel did.
      ``update_H`` on the same branch returns without touching ``H`` at all
      (``stepping.py:944-945``), which is the identity twice over.

    MEASURED off-device on a NumPy lift, 2026-09-14, applying the array sub-step a
    SECOND TIME IN ITS OWN SLOT over 12 driver steps and comparing every array on
    ``Fields`` and the PML::

        case                      pml_is_active   update_E twice   update_P twice
        absorber_1d                     False        0 of 65          13 of 65
        material_dispersion_0d          False        0 of 29           5 of 29
        dispersive_2d                   True        12 of 41           0 of 41
        pml_1d                          True         8 of 32           0 of 32
        dispersive5_2d                  True        22 of 77          22 of 77

    So on an inactive layer this control is NOT-APPLICABLE rather than failing: the
    comparison it makes is between a state and the same state by construction, and
    calling that NULL-DID-NOT-DIVERGE reports a blind instrument as a defect in the
    product. IT IS DERIVED FROM THE LAYER AND THE WITHHELD SET, never from the case
    name or the arm label, so a case that acquires an active layer is scored again.

    AND THE N/A IS NOT A FREE PASS: it is granted only beside the positive
    measurement that replaces it. ``update_P`` holds a DEVICE GROUP on a three-slot
    weld, this control now leaves it running (see
    :func:`_withhold_absorbed_consult`), and the row asserts that group's own
    ``launches`` counter actually advanced over the control's twelve steps. A
    trailing group that did not launch is a FAILING verdict here, which is the
    question "did the fused kernel's second half run" asked directly instead of
    through a comparison that cannot answer it.
    """
    say(f"{case}: ARMED NULL — running {CONTROL_STEPS} steps with the ABSORBED "
        "consult forced False")
    a, b = _pair_of_lifted(case, spec["arms"], gpu_id, res, counter, "withheld",
                           spec.get("reached_by", "opt-in"))
    pre = e2e.compare_state(e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"]))
    row: Dict[str, Any] = {"control": "absorbed-consult-withheld",
                           "precondition_identical": pre["identical"]}
    if not pre["identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        a["driver"].close()
        b["driver"].close()
        return row
    restore, state = _withhold_absorbed_consult()
    try:
        step_leg(case, a, counter, CONTROL_STEPS)
    finally:
        restore()
    step_leg(case, b, counter, CONTROL_STEPS)
    block = fusion_block(a)
    launching = _launching_slots_ran(a, state["skipped_launching"], CONTROL_STEPS)
    constitutive_only = bool(state["slots"]) and all(
        slot in _CONSTITUTIVE_SLOTS for slot in state["slots"])
    overwrite_branch = _constitutive_is_an_overwrite(a)
    verdict = e2e.compare_state(e2e.collect_state(a["driver"]),
                                e2e.collect_state(b["driver"]))
    row.update({
        "steps": CONTROL_STEPS,
        "consults_withheld": state["withheld"],
        "withheld_slots": state["slots"],
        # The leading repair slots this control deliberately LEFT ALONE. Recorded
        # rather than merely skipped: it is the reader's evidence that the pairs
        # here carry a deposit repair at all, and which seam carries it.
        "leading_repair_slots_left_running": state["skipped_leading"],
        # The slots holding a DEVICE GROUP of their own, likewise left running, and
        # what that group's own counter did over these steps. A three-slot weld's
        # trailing half lives here; on a two-slot pair the mapping is empty.
        "launching_slots_left_running": state["skipped_launching"],
        "launching_slots": launching,
        "every_withheld_slot_is_constitutive": constitutive_only,
        "constitutive_is_a_pure_overwrite": overwrite_branch,
        "fused_arms_driven": block["driven"],
        "diverged": not verdict["identical"],
        "arrays_differing": verdict["arrays_differing"],
        "arrays_compared": verdict["arrays"],
        "first_differences": verdict["differences"][:6],
    })
    if not state["withheld"]:
        row["verdict"] = "NOT-ARMED"
        row["why"] = ("no absorbed consult was reached, so nothing was withheld and "
                      "this control measured nothing")
    elif launching["failures"]:
        # THE SLOT THIS CONTROL STOPPED WITHHOLDING MUST HAVE RUN. Leaving a device
        # group running is only honest while the row can show it ran; a trailing
        # group whose counter stood still is the very thing the withheld comparison
        # used to be asked about, and it is answered here directly.
        row["verdict"] = "LAUNCHING-SLOT-DID-NOT-LAUNCH"
        row["why"] = "; ".join(launching["failures"])
    elif not verdict["identical"]:
        row["verdict"] = "DIVERGED-AS-REQUIRED"
    elif overwrite_branch and constitutive_only:
        # THE MECHANISM DOES NOT EXIST ON THIS BRANCH. See the docstring's table:
        # a second array constitutive on an inactive layer changed 0 of 65 arrays on
        # absorber_1d and 0 of 29 on material_dispersion_0d, measured with no kernel
        # anywhere near the run. A comparison that is the identity by construction
        # cannot be evidence either way, and scoring it as a failure would report
        # the instrument as a defect in the product.
        row["verdict"] = "NOT-APPLICABLE"
        carries = ["the case's in-seam-pass mutation, which diverges only if the "
                   "leading group's constitutive half ran IN-LAUNCH"]
        if launching["slots"]:
            carries.insert(0, f"the device groups left running "
                              f"{launching['slots']}, whose own launch counters "
                              f"read {launching['deltas']} over {CONTROL_STEPS} "
                              f"steps")
        row["why"] = (
            f"the layer is INACTIVE, so the array arm of every withheld slot "
            f"{sorted(state['slots'])} is a pure overwrite (update_E, "
            f"stepping.py:1019-1022) or a return without a write (update_H, "
            f"stepping.py:944-945); applying it a second time is the identity "
            f"whatever the kernel did, so this comparison has no armed direction "
            f"here. What carries the claim instead: " + "; ".join(carries))
    else:
        row["verdict"] = "NULL-DID-NOT-DIVERGE"
        row["why"] = ("the array constitutive call ran ON TOP of the fused launch's "
                      "own constitutive half and the answer did not change, so this "
                      "comparison cannot see whether the fused kernel's second half "
                      "ran at all")
    say(f"{case}: ARMED NULL -> {row['verdict']} "
        f"({row.get('consults_withheld')} consults withheld, "
        f"{row.get('arrays_differing')} arrays differ)")
    a["driver"].close()
    b["driver"].close()
    return row


#: Which driver slots each seam's fused pair occupies. A pair "spans" a seam when it
#: holds one of that seam's slots -- the slots the in-seam wipe runs between.
_SEAM_SLOTS: Dict[str, Tuple[str, ...]] = {
    "B": ("step_B", "update_H"),
    "D": ("step_D", "update_E", "update_P"),
}


def _side_is_fused(driven: Dict[str, str], side: str) -> bool:
    """Does a fused product span this seam on this leg? Read off the driven slots.

    Read from the RECORD rather than from the drive entry, because the entry
    describes THIS table's composition and the leg may have been served by the
    other one.
    """
    return any(driven.get(slot) for slot in _SEAM_SLOTS.get(side, ()))


def seam_mutation_control(case: str, spec: Dict[str, Any], side: str, gpu_id: int,
                          res: Optional[int], counter: Any) -> Dict[str, Any]:
    """``zero_metal_{side}`` made non-idempotent. MUST diverge."""
    say(f"{case}: SEAM MUTATION on zero_metal_{side} — {CONTROL_STEPS} steps")
    restore, state = _mutate_in_seam_pass(side)
    row: Dict[str, Any] = {"control": f"in-seam-pass-mutated-{side}",
                           "mutation": f"zero_metal_{side} writes "
                                       f"{SEAM_MUTATION_VALUE} into the wall plane "
                                       f"instead of leaving the wipe idempotent"}
    try:
        a, b = _pair_of_lifted(case, spec["arms"], gpu_id, res, counter, "mutated",
                               spec.get("reached_by", "opt-in"))
        pre = e2e.compare_state(e2e.collect_state(a["driver"]),
                                e2e.collect_state(b["driver"]))
        row["precondition_identical"] = pre["identical"]
        if not pre["identical"]:
            row["verdict"] = "HARNESS-FAILURE"
            a["driver"].close()
            b["driver"].close()
            return row
        step_leg(case, a, counter, CONTROL_STEPS)
        step_leg(case, b, counter, CONTROL_STEPS)
        verdict = e2e.compare_state(e2e.collect_state(a["driver"]),
                                    e2e.collect_state(b["driver"]))
        block = fusion_block(a)
        row.update({
            "steps": CONTROL_STEPS,
            "pass_calls": state["calls"],
            "components_written": state["written"],
            "fused_arms_driven": block["driven"],
            "diverged": not verdict["identical"],
            "arrays_differing": verdict["arrays_differing"],
            "first_differences": verdict["differences"][:6],
        })
        if not state["calls"]:
            row["verdict"] = "NOT-ARMED"
            row["why"] = f"zero_metal_{side} was never called"
        elif not block["driven"]:
            row["verdict"] = "NOT-ARMED"
            row["why"] = "the leg did not fuse, so the mutation had no seam to sit in"
        elif not _side_is_fused(block["driven"], side):
            # THE GUARD IS PER SIDE, BECAUSE THE MUTATION IS. ``zero_metal_D`` only
            # sits in a seam a fused ELECTRIC product spans; on a leg whose D seam
            # runs separate certified arms, both legs run the same mutated wipe
            # followed by the same array sub-step and the mutation cancels. That is a
            # true statement about the composition, not evidence that the comparator
            # is blind. Measured 2026-09-11 on the cuda_default_precedence leg:
            # folded_dispersive_2d fuses only its B seam under the shipped
            # precedence, and asking "did anything fuse" instead of "did THIS side
            # fuse" failed the leg on its own correct result. The Metal route gate
            # carries the same guard for the same reason.
            row["verdict"] = "NOT-APPLICABLE"
            row["why"] = (f"this leg fuses {sorted(set(block['driven'].values()))} "
                          f"and no fused pair spans the {side} seam, so the mutated "
                          f"wipe has no in-launch twin to be compared against")
        elif verdict["identical"]:
            row["verdict"] = "NULL-DID-NOT-DIVERGE"
            row["why"] = ("the in-seam pass was changed to write a value the fused "
                          "launch cannot have produced and the two legs still "
                          "matched, so the byte comparison is blind to what runs "
                          "between the two consults")
        else:
            row["verdict"] = "DIVERGED-AS-REQUIRED"
        a["driver"].close()
        b["driver"].close()
    finally:
        restore()
    say(f"{case}: SEAM MUTATION {side} -> {row.get('verdict')} "
        f"({row.get('arrays_differing')} arrays differ)")
    return row


def _one_step_report(case: str, env: Dict[str, Optional[str]], gpu_id: int,
                     res: Optional[int]) -> Tuple[Dict[str, Any], str]:
    """Lift ``case`` under ``env``, take ONE driver step, return its record + path.

    One step is the whole instrument for a leg that reads a DECISION: the plan is
    frozen on the first step and the artifact is complete from then on, so a longer
    run would cost device time and measure nothing further. Legs that compare BYTES
    are the ones that need the full budget, and they are elsewhere.
    """
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    builder = CASES[case]
    sim, _monitors, _until = builder(mp) if res is None else builder(mp, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=e2e.PREFER_GPU, gpu_id=gpu_id)
    driver.step()
    report = dict(driver.fast_path_report() or {})
    path = driver.active_step_path
    driver.close()
    return report, path


def arbitration_summary(controls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Every arbitration verdict a case produced, de-duplicated and sorted.

    THE SHAPE IS A LIST BECAUSE THE MEASUREMENT IS THREE ANSWERS, NOT ONE.
    :func:`arbitration_leg` runs three one-step reads per dispatching case and each
    answers a different question: which table composes first when nothing is
    preferred, whether the incumbent yields when this one is, and whether an
    unrunnable preference is refused BY NAME. ``recut_driver_dispatch_record.py``
    requires the first two by name before it will publish a by-preference number,
    so a summary that reports one verdict per case is not a smaller record -- it is
    a record missing the two statements the number is conditioned on.
    """
    verdicts: Dict[str, List[str]] = {}
    for entry in controls:
        if not (entry.get("control") or "").startswith("arbitration"):
            continue
        case = entry.get("case")
        if case is None:
            continue
        seen = verdicts.setdefault(case, [])
        verdict = entry.get("verdict")
        if verdict not in seen:
            seen.append(verdict)
    return {case: sorted(seen, key=lambda v: (v is None, v))
            for case, seen in verdicts.items()}


def arbitration_leg(case: str, spec: Dict[str, Any], gpu_id: int,
                    res: Optional[int]) -> List[Dict[str, Any]]:
    """The three one-step reads that turn the arbitration into a MEASUREMENT.

    Every CUDA dispatch number this campaign licenses is conditioned on WHICH TABLE
    composed first, and until this leg existed that condition was a sentence in a
    constant's docstring. Three directions, three controls, and the release requires
    all three per dispatching case:

    * DEFAULT (``MEEP_GPU_BACKEND_PREFERENCE`` unset) — the plan must still
      dispatch, every seam both tables admit must go to the INCUMBENT, and
      ``arbitration.refused.cuda`` must NAME the unit that lost with the slots it
      lost. The hand-CUDA counters must read zero over the step.
    * PREFERRED (``=cuda``) — the effective order must put this table first, the
      driven slots must carry ITS labels, and the incumbent's kernels must read zero
      on them. That is what licenses the by-preference number.
    * REJECTED (``=metal``, a table no CuPy host can run) — the plan must be REFUSED
      BY NAME, naming the switch and the token, and the step path must be the array
      path. A refusal that arrived as "internal error" would mean the rung raised
      where the ladder's contract says it answers.

    ON THE ``cuda_alone`` LEG THERE IS ONE TABLE, so the first two reads ask a
    different question with the same instruments: with the preference unset AND set
    to ``cuda``, the record must show the hand-CUDA table composing alone because
    Triton is not importable (:func:`sole_table_verdict`). The rejected read is
    unchanged -- a preference naming a table the host cannot run is refused by name
    whether or not Triton is present.
    """
    if BACKEND != "cuda" or spec.get("expect") != "dispatch":
        return [{"control": "arbitration", "verdict": "NOT-APPLICABLE",
                 "why": ("the arbitration controls measure which of TWO NVIDIA "
                         "tables composed first; this run drives one table")}]
    rows: List[Dict[str, Any]] = []

    default, default_path = _one_step_report(
        case, _fuse_env(spec["arms"], spec.get("reached_by", "release"), prefer=None),
        gpu_id, res)
    refused_units = ((default.get("arbitration") or {}).get("refused") or {}).get(
        "cuda") or {}
    rows.append({
        "control": "arbitration-default-precedence",
        "effective_order": list((default.get("arbitration") or {})
                                .get("effective_order") or []),
        "step_path": default_path,
        "cuda_units_refused": dict(refused_units),
        "driven": dict((default.get("fusion") or {}).get("driven") or {}),
        "triton_table": dict((default.get("tables") or {}).get("triton") or {}),
        "verdict": (sole_table_verdict(default) if TRITON_WITHHELD is not None
                    else arbitration_verdict(default, "triton")),
    })

    preferred, preferred_path = _one_step_report(
        case, _fuse_env(spec["arms"], spec.get("reached_by", "release"),
                        prefer="cuda"),
        gpu_id, res)
    driven = dict((preferred.get("fusion") or {}).get("driven") or {})
    namespaced = [label for label in driven.values() if label.startswith("cuda:")]
    rows.append({
        "control": "arbitration-cuda-preferred",
        "effective_order": list((preferred.get("arbitration") or {})
                                .get("effective_order") or []),
        "step_path": preferred_path,
        "driven": driven,
        "namespaced_labels": sorted(set(namespaced)),
        "tables_dispatched": list((preferred.get("composition") or {})
                                  .get("tables_dispatched") or []),
        # EVERY SLOT THE RECORD CALLS DISPATCHED NAMES THE TABLE THAT FILLED IT.
        # Both tables launch real kernels over the same arrays, so a step credited
        # to the wrong one is invisible to every byte comparison.
        "backends": {slot: entry.get("backend")
                     for slot, entry in (preferred.get("slots") or {}).items()
                     if entry.get("state") == "dispatched"},
        "verdict": (sole_table_verdict(preferred) if TRITON_WITHHELD is not None
                    else "INCUMBENT-YIELDED"
                    if namespaced and arbitration_verdict(preferred, "cuda")
                    == "INCUMBENT-YIELDED" else "NO-CUDA-UNIT-ADOPTED"),
    })

    rejected, rejected_path = _one_step_report(
        case, dict(_fuse_env(spec["arms"], spec.get("reached_by", "release"),
                             prefer="metal")),
        gpu_id, res)
    rows.append({
        "control": "arbitration-preference-rejected",
        "step_path": rejected_path,
        "refused_because": rejected.get("refused_because"),
        "refusing_rung": classify_refusal(rejected.get("refused_because") or ""),
        "verdict": ("PREFERENCE-REFUSED-BY-NAME"
                    if (arbitration_verdict(rejected, "cuda")
                        == "PREFERENCE-REFUSED-BY-NAME"
                        and rejected_path == "array")
                    else "NOT-REFUSED-BY-NAME"),
    })
    for row in rows:
        say(f"{case}: {row['control']} -> {row['verdict']}")
    return rows


def release_route_leg(case: str, spec: Dict[str, Any], gpu_id: int,
                      res: Optional[int]) -> Dict[str, Any]:
    """The SHIPPED route, asserted: the switch UNSET, and the release does the admitting.

    WHY THIS LEG EXISTS AT ALL. Every fused launch this gate has ever measured was
    reached through ``MEEP_GPU_FUSE_ARMS``. What ships is the other route — the
    switch unset, ``RELEASED_FUSED_ARMS`` admitting inside
    ``FUSED_RELEASE_ENVELOPE`` — and a gate whose evidence is all about the opt-in
    would license a path no user takes. The main case legs now run the shipped route
    (``reached_by: release``); this control reads the record BACK and requires it to
    say so, because "the switch was unset in the environment" and "the release is
    what admitted the arm" are different claims and only the second is the licence.

    REFUSES A SILENT FALLBACK. If a future edit made the main leg set the switch
    again, every byte comparison would still pass and the gate would go on calling
    the result evidence about the shipped path. The three assertions below are what
    stop that: the switch's value must be absent from the record, ``opted_in`` must
    be empty, and ``released_here`` must cover every arm the composer drove.
    """
    if spec.get("reached_by") != "release":
        return {"leg": "release-route", "verdict": "NOT-APPLICABLE",
                "why": (f"{case} is outside FUSED_RELEASE_ENVELOPE and is reached "
                        f"by {spec.get('reached_by')!r}; there is no release to "
                        "assert here")}
    report, path = _one_step_report(case, _fuse_env(spec["arms"], "release"),
                                    gpu_id, res)
    fusion = dict(report.get("fusion") or {})
    driven = sorted(set((fusion.get("driven") or {}).values()))
    released = list(fusion.get("released_here") or [])
    row = {
        "leg": "release-route",
        "switch_value_in_the_record": fusion.get("value"),
        "opted_in": list(fusion.get("opted_in") or []),
        "released_here": released,
        "outside_the_released_envelope": list(
            fusion.get("outside_the_released_envelope") or []),
        "driven": dict(fusion.get("driven") or {}),
        "driver_route_gate": fusion.get("driver_route_gate"),
        "decision": report.get("decision"),
        "active_step_path": path,
        "the_switch_was_not_used": fusion.get("value") is None
                                   and not (fusion.get("opted_in") or []),
        "the_release_covers_every_driven_arm": bool(driven)
                                               and set(driven) <= set(released),
    }
    row["verdict"] = ("RELEASE-ADMITTED"
                      if (row["the_switch_was_not_used"]
                          and row["the_release_covers_every_driven_arm"]
                          and report.get("decision") == "dispatched"
                          and path == "fused")
                      else "RELEASE-DID-NOT-ADMIT")
    say(f"{case}: release-route leg -> {row['verdict']} :: switch="
        f"{row['switch_value_in_the_record']!r} released={released} driven={driven}")
    return row


#: The case :func:`envelope_leg` drives, and why it is this one.
#:
#: THE TRITON ROW MOVED 2026-09-13, FOR THE SECOND TIME AND FOR THE SAME REASON THE
#: FIRST MOVE HAPPENED. It named ``pml_3d_diagonal``, whose only envelope refusal
#: was ``dimensions=3`` on the two ordinary pairs; this round widens that row to
#: frozenset({1, 2, 3}) and promotes the case to a DRIVE row, so the release ADMITS
#: it and it can no longer witness the release declining anything. A witness has to
#: be a shape the release ROW refuses while the arm's PREDICATE admits it — that
#: split is what lets both directions run on one case — and a case the release
#: admits satisfies neither direction.
#:
#: ``folded_dispersive5_2d`` IS THE REPLACEMENT, on axis ``susceptibilities``. A
#: fold at five poles: ``fused pair D (folded)`` pins that axis at 0 already and
#: ``fused pair B (folded)`` gains a pin this round, so the release refuses both,
#: BY NAME and on the named axis, while the composer's own predicates admit the B
#: arm — measured on a NumPy lift 2026-09-13, where its every refusal is the array
#: module. Nothing else on this table is admitted there either, and each refusal
#: names its own axis rather than falling out by accident: the unfolded pairs and
#: the dispersive pair refuse on ``folded``, the complex pairs on the absent mirror,
#: the cylindrical pair on ``cylindrical``. The folded DISPERSIVE product, which is
#: what the composer gives the D seam on a dispersive fold, was not released at all
#: when that measurement was taken; the re-attribution round released it on
#: ``folded_dispersive_2d`` and pinned its ``susceptibilities`` row at ONE pole
#: expressly so that this case keeps its witness — re-measured on a NumPy lift of
#: the builder with all three rows in place, ``released_fused_arms`` is still empty
#: here and the third refusal reads ``susceptibilities=5, not 1``. Widening that
#: row to five is what a five-pole ROUTE case would cost, and it is why no such
#: case is in this table.
#:
#: THE OPT-IN NAMES THE MAGNETIC ARM AND THAT IS FORCED RATHER THAN CHOSEN: on a
#: dispersive fold the D seam goes to the folded dispersive product, so an opt-in
#: naming ``fused pair D (folded)`` would leave the named arm undriven for a reason
#: that has nothing to do with the envelope. ``ENVELOPE_ROWS['cuda']`` withholds its
#: magnetic pair for the same reason. WHAT DIRECTION 2 STILL OWES, and it is the one
#: thing no off-device measurement settles: that the named arm is DRIVEN on the leg,
#: that ``fused pair D (folded)`` is not, and that the plan dispatches anyway. If it
#: fails there, report it rather than falling back to a weaker witness — a release
#: that cannot be shown refusing is worse than a lever that does not land.
#:
#: THE ENVELOPE CONTROL IS PER TABLE, because "a shape the release does not cover"
#: is a statement about a PARTICULAR release and the two tables cover different
#: ones. Reading the Triton row on a CUDA run measured nothing: ``pml_3d_diagonal``
#: is a 3-D grid the Triton arms decline on ``dimensions``, and it is exactly a case
#: the CUDA arms DO drive -- so direction 1 asked the envelope to refuse a
#: configuration it correctly admits, and the control reported
#: ENVELOPE-DID-NOT-HOLD on a working envelope (measured 2026-09-11). The same
#: per-table reading is why the CUDA table admitting three arms on
#: ``folded_dispersive5_2d`` does not touch the Triton direction 1: ``released_here``
#: is per table, and a Triton leg with no backend preference dispatches no other
#: table's arms. The precedent is in the artifact — the standing Triton envelope row
#: on ``pml_3d_diagonal`` reads ENVELOPE-HOLDS with ``released_here []`` on a case
#: where ``CUDA_RELEASED_FUSED_ARMS`` lists both ordinary CUDA pairs.
#:
#: WHAT THIS CONTROL LOOKS LIKE ON A MAC, so the next reader does not re-derive it:
#: ``ENVELOPE-DID-NOT-HOLD``, with ``released_here`` naming two METAL arms. Measured
#: here 2026-09-13 under ``--smoke``. Neither half is about the witness — the Metal
#: table also released the folded B/H pair at five poles this round, and on a host
#: whose only candidate table is Metal that is the release the report reads; and a
#: NumPy smoke run fuses nothing, so direction 2 cannot hold there whatever the
#: case. This control's measurement is the Triton leg's, on a host where Metal is
#: not a candidate. The Metal route gate keeps its own envelope (``no_pml_2d`` on
#: ``pml_active``) and is untouched by this row.
#:
#: THE CUDA ROW USES ``pml_3d`` AND THE AXIS IS ``off_diagonal_epsilon``. That case
#: builds a sphere, MEEP's subpixel averaging writes off-diagonal chi1inv rows, and
#: ``cuda:fused magnetic pair``'s own release axes pin that axis False -- so with the
#: switch unset nothing of this table is released there, which is direction 1. The
#: opt-in still reaches the magnetic pair (its predicate carries no off-diagonal
#: clause on the H side) while the electric pair's does refuse, which is the partial
#: admission direction 2 needs on the same grid.
ENVELOPE_ROWS: Dict[str, Dict[str, Any]] = {
    # RE-POINTED 2026-09-13 off ``pml_3d_diagonal``, which this round's dimensions
    # widening turns into a DRIVE case. See the note above for why the replacement is
    # a fold at five poles and why the opt-in names the MAGNETIC arm.
    "triton": {"case": "folded_dispersive5_2d",
               "withhold": ["fused pair B (folded)"],
               "pair_arms": ("fused pair B (folded)", "fused pair D (folded)"),
               "axis": "susceptibilities"},
    # RE-POINTED 2026-09-13: the sphere dispatches the magnetic pair now (its
    # off-diagonal admission became a per-arm corner), so it can no longer witness the
    # envelope refusing. ``nonlinear_3d`` is the witness instead: the pair's predicate
    # admits a nonlinear grid (the nonlinear cell is welded) while its release row pins
    # ``nonlinearity`` False -- the split both directions need, on a case that is
    # deliberately NOT a DRIVE row. It is 3-D and not 1-D because the Triton table
    # released ``fused pair B (nonlinear)`` at 1-D the same day, and the shipped
    # precedence dispatches THAT here on the CUDA legs too (measured in the first
    # preflight: direction 1 failed on the other table's release).
    "cuda": {"case": "nonlinear_3d", "withhold": ["cuda:fused magnetic pair"],
             "pair_arms": ("cuda:fused magnetic pair", "cuda:fused electric pair"),
             "axis": "nonlinearity"},
}

#: Set from :data:`ENVELOPE_ROWS` by :func:`main`, once the backend is known.
#:
#: THESE TWO DEFAULTS MOVE WITH THE ROW ABOVE, and forgetting them is a silent
#: failure rather than a loud one: ``main`` overwrites both from
#: :data:`ENVELOPE_ROWS`, but any caller that is NOT ``main`` — a test importing this
#: module, a smoke harness — reads what is written here and would drive the wrong
#: case without saying so. Edit all three in one change.
ENVELOPE_CASE = "folded_dispersive5_2d"
ENVELOPE_WITHHOLD = ["fused pair B (folded)"]   # leaves the folded D pair un-admitted


def envelope_leg(gpu_id: int, res: Optional[int]) -> Dict[str, Any]:
    """The ENVELOPE, both directions, on a shape the release does not cover.

    DIRECTION 1 — the envelope refuses the REQUEST. With the switch unset on the
    backend's own witness shape, ``released_fused_arms`` is empty, so the composer
    is asked with ``fuse=False`` and nothing fuses; the record must NAME the axis
    :data:`ENVELOPE_ROWS` declares — ``susceptibilities`` on the Triton row since
    2026-09-13, ``nonlinearity`` on the CUDA one. Without this the envelope is a
    null: no device run has ever shown it declining anything, and a predicate that
    has only ever said yes is not known to be able to say no.

    THE AXIS IS READ FROM BOTH HALVES OF THE RELEASE, since 2026-09-02. The axes
    that separate arms — ``dimensions`` first, and now every axis this round moved —
    live in each arm's own ``FUSED_RELEASE_ARM_AXES`` row rather than in the shared
    ``FUSED_RELEASE_ENVELOPE``, because the folded pairs and the ordinary ones are
    released on different shapes, so a leg that read only
    ``outside_the_released_envelope`` would find it EMPTY here and report the
    envelope as broken while it was working. The record publishes the per-arm half
    as ``outside_this_arms_own_cases`` and this leg reads both.

    DIRECTION 2 — WHAT AN UN-ADMITTED PRODUCT DOES NOW, and it is not what it did.
    This direction used to assert that a partial opt-in left the other arm neither
    opted into nor released, so clause (8) refused the WHOLE plan by name. That
    stopped being reachable on 2026-09-02 and the change is the point rather than a
    casualty: the composer is now handed the admitted LABEL SET
    (``plan_step``'s ``fuse_labels``), so a product this run cannot admit is never
    installed, its seam keeps the separate certified arms, and the plan DISPATCHES.
    The measured failure that forced it is in this campaign's own case list — a
    conductive 2-D grid went entirely to the array path once an un-releasable
    product was wired, because a fused label occupies both slots of its seam and
    rejecting it rejected the arms underneath.

    So this direction now measures THAT, on the same witness shape, with the same
    partial opt-in: the named arm is driven, the unnamed one is NOT, the plan
    dispatches anyway, and the un-offered seam is dispatched as separate arms. On
    the Triton witness the un-offered seam is a dispersive fold's, where the folded
    DISPERSIVE product is what the composer selects and what it is not offered, so
    the record should show that product ``offered to run`` rather than driven.
    Clause (8) has not been retired — it is a backstop with no live path, which
    ``direction_3`` states as an invariant over both legs and
    ``test_dispatch_contract`` still drives positively with a stubbed composer.

    Both directions must hold. Either alone is satisfiable by a broken predicate:
    an envelope that refuses everything passes (1), and one that refuses nothing
    passes (2) whenever the opt-in is partial.
    """
    # THE ENVELOPE IS A STATEMENT ABOUT THE RELEASE, NOT ABOUT THE PRECEDENCE, so
    # both directions pin the preference to this table on every leg. Left to follow
    # the leg, the control ran unpreferred on ``cuda_default_precedence`` and
    # direction 2's opt-in could not reach the arm it withholds -- measuring the
    # merge instead of the envelope, on the one leg where those differ.
    envelope_prefer = "cuda" if BACKEND == "cuda" else LEG_PREFERENCE
    unset_report, unset_path = _one_step_report(
        ENVELOPE_CASE, _fuse_env([], "release", prefer=envelope_prefer), gpu_id, res)
    unset_fusion = dict(unset_report.get("fusion") or {})
    shared = list(unset_fusion.get("outside_the_released_envelope") or [])
    # THE PER-ARM HALF IS PER TABLE ON A TWO-TABLE RECORD, and reading the flat key
    # on a CUDA run found it EMPTY: the flat key carries the TRITON arms' reasons
    # and the second table's go under ``outside_this_arms_own_cases_by_table``.
    # Direction 1 then failed for want of a reason that was in the artifact one key
    # over, on a run where nothing had in fact fused.
    per_arm = dict(unset_fusion.get("outside_this_arms_own_cases") or {})
    if BACKEND == "cuda":
        by_table = dict(unset_fusion.get("outside_this_arms_own_cases_by_table")
                        or {})
        per_arm = dict(by_table.get("cuda") or by_table.get(BACKEND) or per_arm)
    reasons = shared + [f"{arm}: {why}" for arm, whys in sorted(per_arm.items())
                        for why in whys]

    opt_report, opt_path = _one_step_report(
        ENVELOPE_CASE, _fuse_env(ENVELOPE_WITHHOLD, "opt-in",
                                 prefer=envelope_prefer), gpu_id, res)
    opt_fusion = dict(opt_report.get("fusion") or {})
    opt_driven = sorted(set((opt_fusion.get("driven") or {}).values()))
    opt_slots = {name: entry.get("arm")
                 for name, entry in (opt_report.get("slots") or {}).items()
                 if entry.get("state") == "dispatched"}
    opt_reasons = dict(opt_fusion.get("not_installed") or {})
    row_spec = ENVELOPE_ROWS[BACKEND if BACKEND in ENVELOPE_ROWS else "triton"]
    missing = [arm for arm in row_spec["pair_arms"]
               if arm not in ENVELOPE_WITHHOLD]
    offer_reasons = [f"{product}: {text}" for product, text
                     in sorted(opt_reasons.items())
                     if "offered to run" in str(text)]

    row: Dict[str, Any] = {
        "leg": "envelope",
        "case": ENVELOPE_CASE,
        "switch_unset": {
            "released_here": list(unset_fusion.get("released_here") or []),
            "outside_the_released_envelope": shared,
            "outside_this_arms_own_cases": per_arm,
            "driven": dict(unset_fusion.get("driven") or {}),
            "decision": unset_report.get("decision"),
            "active_step_path": unset_path,
            "names_the_axis": any(row_spec["axis"] in reason for reason in reasons),
            "axis_required": row_spec["axis"],
            "nothing_fused": not (unset_fusion.get("driven") or {}),
        },
        "opted_into_a_subset": {
            "opted_in": ENVELOPE_WITHHOLD,
            "not_admitted": missing,
            "admitted": list(opt_fusion.get("admitted") or []),
            "driven": opt_driven,
            "decision": opt_report.get("decision"),
            "active_step_path": opt_path,
            "dispatched_slots": opt_slots,
            "the_named_arm_was_driven": set(opt_driven) == set(ENVELOPE_WITHHOLD),
            "the_unnamed_arm_was_not": not (set(missing) & set(opt_driven)),
            # The un-offered seam must be SERVED, not emptied: a plan that merely
            # refused to fuse ``step_B``/``update_H`` and left them on the array
            # path would satisfy the two clauses above and still be the regression
            # this direction exists to measure.
            "the_unoffered_seam_kept_its_arms": all(
                opt_slots.get(slot) for slot in ("step_B", "update_H")),
            "the_composer_named_the_offer": bool(offer_reasons),
            "offer_reasons": offer_reasons[:2],
            "refused_because": opt_report.get("refused_because") or "",
        },
    }
    one = (row["switch_unset"]["names_the_axis"]
           and row["switch_unset"]["nothing_fused"]
           and not row["switch_unset"]["released_here"])
    subset = row["opted_into_a_subset"]
    two = (opt_report.get("decision") == "dispatched"
           and opt_path == "fused"
           and subset["the_named_arm_was_driven"]
           and subset["the_unnamed_arm_was_not"]
           and subset["the_unoffered_seam_kept_its_arms"]
           and subset["the_composer_named_the_offer"])
    # CLAUSE (8) IS A BACKSTOP AND THE ARTIFACT SAYS SO. Not "it never fires" as a
    # hope: on both legs of this control, every DRIVEN fused label must be one the
    # ladder admitted, which is the property that makes the clause unreachable from
    # the shipped composer. A future composer that installs past the offer breaks
    # this and the leg reports it.
    three = True
    backstop = []
    for name, report in (("switch_unset", unset_report),
                         ("opted_into_a_subset", opt_report)):
        fusion = dict(report.get("fusion") or {})
        driven = set((fusion.get("driven") or {}).values())
        admitted = set(fusion.get("admitted") or ())
        backstop.append({"leg": name, "driven": sorted(driven),
                         "admitted": sorted(admitted),
                         "driven_is_a_subset_of_admitted": driven <= admitted})
        three = three and driven <= admitted
    row["clause_8_backstop"] = backstop
    row["direction_1_envelope_refuses_the_request"] = one
    row["direction_2_the_offer_keeps_the_unadmitted_product_out"] = two
    row["direction_3_no_fused_label_reaches_clause_8_unadmitted"] = three
    row["verdict"] = ("ENVELOPE-HOLDS" if (one and two and three)
                      else "ENVELOPE-DID-NOT-HOLD")
    say(f"{ENVELOPE_CASE}: envelope leg -> {row['verdict']} :: unset "
        f"reasons={reasons[:1]} | subset driven={opt_driven} "
        f"path={opt_path} offer_named={subset['the_composer_named_the_offer']}")
    return row


def stamp_flux_applicability(observables: Dict[str, Any],
                             monitors_declared: int) -> Dict[str, Any]:
    """Say whether the flux comparison is a MEASUREMENT, in the block itself.

    ``e2e.compare_spectra`` over two empty mappings returns ``identical True`` with
    ``monitors 0``, which is the honest answer to "did the spectra differ" and a
    VACUOUS answer to "do the observables agree". The first used to be conjoined
    into the case verdict as though it were the second, so a case whose builder
    returns no monitor contributed a green clause for a comparison nobody made.

    ``applicable`` IS DERIVED FROM THE BUILDER'S RETURN, which ``run_leg`` carries,
    so it is a property of the case and not a key a future round can set to quiet a
    clause. ``migration_agrees`` is the guard on that derivation: the declared count
    must equal the count that actually migrated onto the driver, or a monitor built
    and dropped in the lift would arrive here indistinguishable from a case that
    declares none -- and would silence a comparison that should have run.
    """
    observables["declared_by_the_builder"] = monitors_declared
    observables["applicable"] = monitors_declared > 0
    observables["migration_agrees"] = (
        monitors_declared == observables.get("monitors"))
    return observables


def flux_agrees(observables: Dict[str, Any]) -> bool:
    """The flux conjunct of the case verdict: True where there is nothing to ask.

    A case with no monitor contributes no clause rather than a passing one; what
    carries it is the byte comparison over every array on ``Fields`` and the PML,
    which is the same instrument on every case here.
    """
    if not observables.get("applicable"):
        return True
    return bool(observables.get("identical"))


def case_verdict_for_second_consult(site: Dict[str, Any]) -> Optional[str]:
    """The case-level word for a second-consult outcome, or ``None`` to pass through.

    TWO FAILING WORDS, BECAUSE THEY ARE TWO FINDINGS. ``SECOND-CONSULT-FAILURE`` is
    about the half-step: its consults composed differently from the step's, which is
    a defect in a kernel, a plan or the containment rule.
    ``SECOND-CONSULT-HARNESS-FAILURE`` is about the legs handed to it: they were not
    byte-identical going in, so nothing the site measured is attributable to it. One
    word for both sent a reader to the seam for a run whose seam was clean.

    ``NOT-APPLICABLE`` PASSES THROUGH LIKE A PASS, and it has to: the site declines
    a case that carries no monitor (see :func:`second_consult_site`), and a word
    that fell through to the ``return`` below would fail the case on a measurement
    the case's own cell makes impossible to take. It is NOT the same as ``PASS`` in
    the row — the row says which it was — but it decides the case verdict the same
    way, by deciding nothing.
    """
    verdict = site.get("verdict")
    if verdict in ("PASS", "NOT-APPLICABLE"):
        return None
    if verdict == "HARNESS-FAILURE":
        return "SECOND-CONSULT-HARNESS-FAILURE"
    return "SECOND-CONSULT-FAILURE"


def second_consult_site(case: str, legs: Dict[str, Dict[str, Any]],
                        counter: Any) -> Dict[str, Any]:
    """Exercise ``synchronize_magnetic_fields`` on every leg and compare.

    THE SEAM HAS SEVEN CONSULTS, NOT FIVE. ``synchronize_magnetic_fields`` runs
    ``step_B`` and ``update_H`` again off a plan it deliberately does NOT re-freeze
    (driver.py reads ``None if self._fast_path_stale else self._fast_path``), with
    the same three unconsulted in-seam passes between them. Any flux or energy
    monitor reaches it. A gate that drove only ``step()`` has not driven the seam,
    and for a fused pair it is the place the ``_committed`` bookkeeping is most
    exposed: the leading consult commits the absorbed slot and the absorbed consult
    must discard that commitment inside the SAME call, or the next step's
    ``update_H`` dispatches off a stale one.

    A CASE WITH NO MONITOR IS NOT-APPLICABLE RATHER THAN A PASS, and the Metal route
    gate has kept that record for ``cylindrical`` since 2026-09-12
    (``gate_dispatch_metal_route.second_consult_site``). "Any flux or energy monitor
    reaches it" is the sentence this site's whole claim rests on: a case with no
    monitor never reaches the half-step in a real run, and the driver's own
    ``synchronize_magnetic_fields`` can still be called by hand — so the choice is
    between declining and putting a green word on a measurement the configuration
    never takes.

    THE FACT IS READ OFF THE BUILDER, not off the drive table. ``run_leg`` carries
    the monitor list the case's builder RETURNED, so the applicability is a property
    of the case rather than a per-case key someone can set to silence an
    inconvenient clause; a builder that starts returning a monitor is scored again
    with no edit here. Fail closed: the decline needs every leg to DECLARE a list and
    every declared list to be empty, so a leg dict without the key measures as before.

    IT IS ALSO WHY ``material_dispersion_0d`` READ HARNESS-FAILURE on the 2026-09-14
    CUDA leg. Its cell is ``(1, 1, 1)``: every curl is zero, so the magnetic
    half-step genuinely cannot change ``B`` or ``H`` and the site's own non-vacuity
    clause — written for the 2026-09-11 aliasing ``collect_state``, where the same
    reading was a defect — fired on a true property of the cell. Declining on the
    monitor fact answers that before the clause is reached, and keeps the clause
    armed everywhere it can still be wrong.
    """
    out: Dict[str, Any] = {"site": "synchronize_magnetic_fields"}
    declared = {label: leg.get("monitors") for label, leg in legs.items()}
    out["monitors_declared"] = {label: (None if value is None else len(value))
                                for label, value in declared.items()}
    if legs and all(value is not None and len(value) == 0
                    for value in declared.values()):
        out["verdict"] = "NOT-APPLICABLE"
        out["why"] = (f"{case}'s builder returns NO MONITORS, so nothing in a real "
                      "run reaches synchronize_magnetic_fields and there is no "
                      "observable for the comparison after it to be about; the "
                      "clauses below are declined rather than scored")
        say(f"{case}: second consult site -> NOT-APPLICABLE (the builder returns "
            f"no monitors)")
        return out
    before: Dict[str, Any] = {}
    after: Dict[str, Any] = {}
    launches: Dict[str, Any] = {}
    for label, leg in legs.items():
        driver = leg["driver"]
        before[label] = e2e.collect_state(driver)
        # THE SITE OWNS ONLY WHAT IT CHANGED, and until 2026-09-11 the row could not
        # say so. Every comparison below is between states taken AFTER a leg's own
        # half-step, so a divergence the legs carried INTO the site reads as a
        # divergence the site produced -- and the artifact carries no field that
        # tells the two apart. The combination is diagnosable, but only by
        # arithmetic on three other fields: with every ``restore_returns_the_state``
        # True, ``restored_fused_vs_array`` is exactly ``before[fused] ==
        # before[array]``, so a row reporting all-True restores beside a False
        # restored comparison is reporting a divergence that PREDATES the half-step,
        # whatever its ``synchronized_state`` says. That is a deduction a reader has
        # to make; this field is the measurement.
        #
        # The snapshots are taken in the loop's own order, so the first leg's
        # entry state is read before ANY leg has synchronized: a half-step that
        # wrote through another driver's arrays -- the failure the ``fields is not
        # self.fields`` identity guard exists for -- lands here as a later leg
        # entering unequal, not as a mystery in the comparison at the end.
        e2e._apply_env(leg)
        snap = counter.snapshot()
        cuda_snap = (CUDA_COUNTER.snapshot() if CUDA_COUNTER is not None else None)
        driver.synchronize_magnetic_fields()
        triton_delta = e2e.TritonLaunchCounter.delta(snap, counter.snapshot())
        cuda_delta = (e2e.CudaLaunchCounter.delta(cuda_snap, CUDA_COUNTER.snapshot())
                      if CUDA_COUNTER is not None else None)
        # BOTH TABLES' DELTAS ARE KEPT AND THE RUN'S OWN TABLE DECIDES THE CLAUSES
        # below. The array leg must launch nothing at this site; asked of the wrong
        # table that requirement is met by a leg that launched a kernel per step.
        launches[label] = dict(triton_delta)
        launches[label]["triton_total"] = triton_delta["total"]
        launches[label]["cuda"] = cuda_delta
        if BACKEND == "cuda":
            launches[label]["total"] = (cuda_delta or {}).get("total", 0)
        after[label] = e2e.collect_state(driver)
    out["launches_during_sync"] = launches
    # THE HALF-STEP'S OWN CONSULT NAME, and what the plan answered it.
    # ``synchronize_magnetic_fields`` asks for ``fastpath.SYNC_UPDATE_H_PASS``
    # rather than ``update_H``, so a product spanning into the electric half can
    # decline the site; a fused pair inside the magnetic half must NOT be declined,
    # or this run took the array constitutive while ``step`` took the kernel and the
    # composition above is not the one the record describes. Read off the plan
    # rather than asserted: the count is the evidence either way.
    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415 - late, as above
    out["sync_pass_name"] = _fastpath.SYNC_UPDATE_H_PASS
    out["sync_refusals"] = {}
    for label, leg in legs.items():
        plan = getattr(leg["driver"], "_fast_path", None)
        out["sync_refusals"][label] = dict(getattr(plan, "sync_refusals", {}) or {})
    out["no_pair_was_declined_at_the_second_site"] = not any(
        out["sync_refusals"].values())
    out["state_entering_the_site"] = {
        "fused_vs_array": e2e.compare_state(before["fused"],
                                            before["array"])["identical"],
        "unfused_vs_array": e2e.compare_state(before["unfused"],
                                              before["array"])["identical"],
    }
    out["the_legs_entered_the_site_identical"] = all(
        out["state_entering_the_site"].values())
    out["synchronized_state"] = {
        "fused_vs_array": e2e.compare_state(after["fused"], after["array"])["identical"],
        "unfused_vs_array": e2e.compare_state(after["unfused"],
                                              after["array"])["identical"],
        "sync_changed_the_state": {
            label: not e2e.compare_state(before[label], after[label])["identical"]
            for label in legs},
    }
    restored: Dict[str, Any] = {}
    for label, leg in legs.items():
        leg["driver"].restore_magnetic_fields()
        restored[label] = e2e.collect_state(leg["driver"])
    out["restore_returns_the_state"] = {
        label: e2e.compare_state(before[label], restored[label])["identical"]
        for label in legs}
    out["restored_fused_vs_array"] = e2e.compare_state(
        restored["fused"], restored["array"])["identical"]
    # The substitution shows up here too: a fused B pair is ONE launch where the
    # unfused leg takes two, in a call that runs the magnetic half and nothing else.
    fused_total = launches["fused"]["total"]
    unfused_total = launches["unfused"]["total"]
    out["launch_drop_in_sync"] = unfused_total - fused_total
    out["array_leg_launched"] = launches["array"]["total"]
    # A SITE CANNOT BE BLAMED FOR A STATE IT INHERITED, and the two answers get
    # different words. ``HARNESS-FAILURE`` is the verdict the case-level
    # precondition already uses for exactly this shape -- legs that are not
    # comparable, so no downstream comparison is about the thing under test -- and
    # the half-step deserves the same, because a reader who sees ``FAIL`` here goes
    # looking for a defect in ``step_B``, the fills or the constitutive and there is
    # nothing there to find.
    if not out["the_legs_entered_the_site_identical"]:
        out["verdict"] = "HARNESS-FAILURE"
        out["why"] = ("the legs were NOT byte-identical entering the half-step, so "
                      "every comparison below inherits a divergence that "
                      "synchronize_magnetic_fields did not produce; the ladder "
                      "above is where it entered")
    elif not all(out["synchronized_state"]["sync_changed_the_state"].values()):
        # THE SITE'S OWN NON-VACUITY, and it has been false before. A half-step that
        # advanced B and H cannot leave the state unchanged, so a leg reporting no
        # change is a leg whose snapshots are not snapshots -- which is exactly what
        # an aliasing ``collect_state`` produced on every NumPy lift until
        # 2026-09-11. Every clause below is trivially satisfied in that state: the
        # legs match because each was compared with itself.
        out["verdict"] = "HARNESS-FAILURE"
        out["why"] = ("a leg reports that the magnetic half-step changed nothing, "
                      "which the half-step cannot do; the comparisons below are "
                      "between a state and itself and measure nothing: "
                      f"{out['synchronized_state']['sync_changed_the_state']}")
    else:
        out["verdict"] = (
            "PASS" if (out["synchronized_state"]["fused_vs_array"]
                       and out["restored_fused_vs_array"]
                       and launches["array"]["total"] == 0
                       and out["no_pair_was_declined_at_the_second_site"])
            else "FAIL")
    say(f"{case}: second consult site -> {out['verdict']} "
        f"(sync launches fused={fused_total} unfused={unfused_total} "
        f"array={launches['array']['total']}; entered identical="
        f"{out['the_legs_entered_the_site_identical']})")
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(case: str, gpu_id: int, counter: Any, res: Optional[int],
             max_steps: Optional[int]) -> Dict[str, Any]:
    spec = drive_table()[case]
    builder = CASES[case]
    row: Dict[str, Any] = {
        "case": case, "intent": CASE_INTENT[case], "why_this_case": spec["why"],
        "arms_requested": spec["arms"], "pairs_expected": spec["pairs"],
        "expect": spec["expect"],
        "needs_probe": spec.get("needs_probe"),
        "probe_path": (os.environ.get(spec["needs_probe"])
                       if spec.get("needs_probe") else None),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # A CASE WHOSE LICENCE THIS LEG DOES NOT CARRY IS NOT THIS LEG'S EVIDENCE.
    # The CUDA table's complex arms take their expansion licence from the unified
    # probe record, named by an environment variable that ONE leg exports
    # (``cuda_shipped_expansion_probe``; ``CUDA_RELEASED_ARM_LEGS`` says so by arm).
    # On the other four legs the arm refuses by name, the case falls to the array
    # path, and the leg recorded DID-NOT-FUSE -- a FAILING verdict saying "this
    # product did not dispatch" where the honest statement is "this leg was never
    # the one that could dispatch it". Four of five legs would fail on two cases
    # that only the fifth was ever meant to drive.
    #
    # THE TEST IS THE ENVIRONMENT, NOT THE LEG'S NAME, so a probe leg that forgot
    # to export the variable is still a failure rather than a skip.
    needs = spec.get("needs_probe")
    if needs and not os.environ.get(needs):
        row["verdict"] = "PASS-NOT-THIS-LEG"
        row["why"] = (f"this case's expansion licence comes from the record "
                      f"{needs} names, and this leg exports no such path; the arms "
                      f"{spec['arms']} cannot be admitted here and the campaign "
                      f"drives them on {spec.get('leg', 'the probe leg')}")
        say(f"{case}: {needs} unset -> PASS-NOT-THIS-LEG (the probe leg drives it)")
        return row

    say(f"{case}: lifting three legs (fused / unfused / array)")
    fused = e2e.run_leg(case, builder, "fused",
                        _fuse_env(spec["arms"], spec.get("reached_by", "opt-in")),
                        counter, gpu_id, res)
    unfused = e2e.run_leg(case, builder, "unfused", unfused_env(), counter,
                          gpu_id, res)
    array = e2e.run_leg(case, builder, "array", ARRAY_ENV, counter, gpu_id, res)
    legs = {"fused": fused, "unfused": unfused, "array": array}
    row["grid_shape"] = fused["grid_shape"]

    states = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
    pre_fa = e2e.compare_state(states["fused"], states["array"])
    pre_ua = e2e.compare_state(states["unfused"], states["array"])
    row["precondition"] = {"fused_vs_array": pre_fa["identical"],
                           "unfused_vs_array": pre_ua["identical"],
                           "arrays": pre_fa["arrays"],
                           "words": pre_fa["words_compared"]}
    if not (pre_fa["identical"] and pre_ua["identical"]):
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = ("the legs were lifted to DIFFERENT initial states, so no "
                      "downstream comparison is about the fused route")
        for leg in legs.values():
            leg["driver"].close()
        return row
    say(f"{case}: precondition OK — {pre_fa['arrays']} arrays, "
        f"{pre_fa['words_compared']} words identical on all three legs")

    dt = float(fused["driver"].grid.dt)
    total = max(1, int(round(fused["until"] / dt)))
    if max_steps is not None:
        total = min(total, max_steps)
    ladder = ladder_for(total)
    row["total_steps"] = total
    row["checkpoints_planned"] = ladder
    row["checkpoints"] = []
    row["first_divergent_checkpoint"] = None

    taken = 0
    for point in ladder:
        for leg in (fused, unfused, array):
            step_leg(case, leg, counter, point - taken)
        taken = point
        snap = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
        fa = e2e.compare_state(snap["fused"], snap["array"])
        ua = e2e.compare_state(snap["unfused"], snap["array"])
        entry = {"steps": point,
                 "fused_vs_array_identical": fa["identical"],
                 "unfused_vs_array_identical": ua["identical"],
                 "arrays": fa["arrays"], "words": fa["words_compared"],
                 "fused_arrays_differing": fa["arrays_differing"],
                 "unfused_arrays_differing": ua["arrays_differing"],
                 "differences": fa["differences"][:6]}
        row["checkpoints"].append(entry)
        say(f"{case}: checkpoint {point}/{total} -> fused"
            f"{'==' if fa['identical'] else '!='}array, unfused"
            f"{'==' if ua['identical'] else '!='}array")
        if not fa["identical"] and row["first_divergent_checkpoint"] is None:
            row["first_divergent_checkpoint"] = point
            row["first_divergence"] = entry

    row["steps_taken"] = {label: leg["steps"] for label, leg in legs.items()}
    row["wall_s"] = {label: leg["wall_s"] for label, leg in legs.items()}
    finals = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
    spectra = {label: e2e.read_spectra(leg["driver"]) for label, leg in legs.items()}
    row["final_state_fused_vs_array"] = e2e.compare_state(finals["fused"],
                                                          finals["array"])
    row["final_state_unfused_vs_array"] = e2e.compare_state(finals["unfused"],
                                                            finals["array"])
    row["observables"] = e2e.compare_spectra(spectra["fused"], spectra["array"])
    # THE FLUX COMPARISON OF A CASE WITH NO MONITOR IS NOT A PASS. ``compare_spectra``
    # over two empty mappings returns ``identical True`` with ``monitors 0``, and that
    # word used to enter ``agree`` beside the byte comparison as though a spectrum had
    # been compared. It had not: the case's builder returns no monitors, which on a
    # single-cell grid is a property of the cell rather than an omission -- there is
    # nothing to put a flux plane on. So the block is marked NOT-APPLICABLE, and the
    # clause is DROPPED from ``agree`` rather than left to pass vacuously.
    #
    # DERIVED FROM THE BUILDER AND CROSS-CHECKED AGAINST THE MIGRATION. The declared
    # count is what the case returned; ``observables['monitors']`` is what actually
    # migrated onto the driver. Equal is the only healthy reading: a monitor built and
    # not migrated would otherwise be indistinguishable here from a case that declares
    # none, and it would silence a comparison that should have run.
    stamp_flux_applicability(row["observables"],
                             len(fused.get("monitors") or ()))
    row["comparator_control"] = e2e.comparator_negative_control(finals["fused"])
    row["array_leg_control"] = e2e.killswitch_leg_is_clean(array)
    # THE SHARED CONTROL ASKS BOTH WITNESSES: ``e2e.killswitch_leg_is_clean`` fails a
    # kill-switch leg that launched a hand-CUDA kernel as well as one that launched a
    # Triton kernel, so the CUDA clause is not repeated here. The CUDA total is
    # still stamped on the row, so a reader of a CUDA artifact sees the count the
    # control was judged on rather than inferring it from ``clean``.
    if BACKEND == "cuda":
        cuda = array.get("cuda_launches") or {}
        row["array_leg_control"]["cuda_launches"] = cuda.get("total")
    row["legs"] = {label: {"plan": leg["plan"],
                           "fusion": fusion_block(leg),
                           "triton_launches": leg["triton_launches"],
                           # BOTH TABLES ON EVERY LEG, so a reader of the artifact
                           # can see which one launched without re-deriving it from
                           # the backend field.
                           "cuda_launches": leg.get("cuda_launches"),
                           "chunks": leg["chunks"],
                           "steady_state": steady_state_rate(leg)}
                   for label, leg in legs.items()}
    row["substitution"] = substitution_proof(fused, unfused, spec)
    row["evidence"] = fused_leg_is_real(fused, spec)
    row["second_consult_site"] = second_consult_site(case, legs, counter)
    row["policy"] = e2e._policy_stamp()

    for leg in legs.values():
        leg["driver"].close()

    # ------------------------------------------------------------------ verdict
    # THE FLUX CLAUSE ONLY WHERE THERE IS A FLUX. ``observables['identical']`` is
    # True over two empty spectra mappings, so on a case with no monitor it used to
    # contribute a green word to ``agree`` for a comparison never made. It is
    # conjoined only when the case declares a monitor; the byte comparison over every
    # array on ``Fields`` and the PML is what carries a no-monitor case, and the row
    # records ``observables.applicable`` so the reader sees which.
    agree = (row["final_state_fused_vs_array"]["identical"]
             and flux_agrees(row["observables"])
             and row["first_divergent_checkpoint"] is None)
    control = row["comparator_control"]
    dispatched = fused["active_step_path"] == "fused" and bool(
        row["evidence"]["driven"])

    if UNCERTIFIED_POLICY_INSTALLED is not None and spec["expect"] == "dispatch":
        # THE SEAM SUPPORTS ONE POLICY, AND THIS IS THE LEG THAT SAYS SO. The
        # harness installed a policy the certified families were not cut under, so
        # the required outcome inverts: rung 8b must refuse BY NAME — naming both
        # policies — nothing may fuse, and the run must still be byte-identical to
        # the array path, because a refusal that also changed the answer would be
        # two failures wearing one word.
        #
        # A folded case is deliberately NOT judged here. Rungs (6)/(6b) answer
        # BEFORE 8b, so a folded configuration is refused for its own reason on
        # this leg too and its existing branch already records WHICH rung did it.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        named = all(needle in reason for needle in POLICY_REFUSAL_NEEDLES)
        row["policy_refusal"] = {
            "harness_installed": UNCERTIFIED_POLICY_INSTALLED,
            "refused_because": reason,
            "names_the_policy_rung": named,
            "names_the_installed_policy": (
                repr(UNCERTIFIED_POLICY_INSTALLED) in reason),
            "did_not_dispatch": not dispatched,
            "byte_identical_to_array": agree,
        }
        if dispatched:
            row["verdict"] = "DISPATCHED-UNDER-AN-UNCERTIFIED-POLICY"
            row["why"] = (f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r} "
                          "and a fused pair dispatched anyway; every certified "
                          "family was cut under a different policy")
        elif not named:
            row["verdict"] = "POLICY-REFUSAL-MISSING"
            row["why"] = (f"nothing dispatched, but the refusal does not name the "
                          f"subnormal policy: {reason!r}")
        elif not agree:
            row["verdict"] = "DIVERGENCE"
            row["why"] = ("the policy rung refused as required and the run still "
                          "differed from the array path")
        else:
            row["verdict"] = "PASS-POLICY-REFUSED"
        return row

    if spec["expect"] == "no-fused-arm":
        # THE CLAIM THAT SURVIVED THE FILL CONSULTS, and it is stronger than the one
        # it replaces. This case used to be ``refused-folded``: nothing on a folded
        # grid was allowed to dispatch, because rung (6) refused any plan that filled
        # ``fill_B``/``fill_D`` and rung (6b) refused any fold whose fills were not
        # both covered. The consults retired the first and the folded release
        # discharged the second for the REAL fold. What is left for a folded COMPLEX
        # grid is not "the plan is refused" — measured 2026-09-02 on the GPU host, with
        # the complex expansion licence offered the two complex fill arms are covered,
        # rung (6b) passes, and six of seven slots dispatch as SEPARATE certified
        # arms. Without the licence the same case refuses at (6b). Both are correct
        # and neither is what the old expectation asserted.
        #
        # SO THE ASSERTION IS THE ONE THAT MATTERS ON EITHER LEG: no FUSED product
        # may be driven here — every folded complex fused label is unreleased, and
        # since the label offer the composer is not even asked for one — and the run
        # must be byte-identical to the array path whichever way it goes. When the
        # plan IS refused the rung is still classified and recorded, so a refusal
        # that moves between rungs is still visible rather than absorbed.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        driven = ((report.get("fusion") or {}).get("driven") or {})
        block = fusion_block(fused)
        row["no_fused_arm"] = {
            "refused_because": reason,
            "refusing_rung": classify_refusal(reason) if reason else None,
            "the_plan_was_refused": bool(reason),
            "fused_arms_driven": dict(driven),
            "dispatched_slots": block["dispatched_slots"],
            # WHICH ARM, AND WHICH TABLE, SERVED EACH DISPATCHED SLOT, on every leg:
            # a row that carries evidence for separate certified arms is read off
            # these two maps, never inferred from the leg it ran on.
            "dispatched_arms": dict(block["arms"]),
            "dispatched_backends": {
                slot: ((report.get("slots") or {}).get(slot) or {}).get("backend")
                for slot in block["dispatched_slots"]},
            "unfused_leg_dispatched": unfused["active_step_path"] == "fused",
            "byte_identical_to_array": agree,
        }
        # THE SINGLES A ROW EXISTS FOR, when it names them (``singles``). A
        # no-fused-arm row that is the route case of a CUDA SINGLE would otherwise
        # pass whether or not that single ever ran -- byte identity to the array path
        # holds just as well when the slot stayed on it. So on every leg where the
        # hand-CUDA table composes FIRST or ALONE, and the harness installed the
        # certified policy (the flush leg's refusal at 8b is its own answer), each
        # named slot must have been dispatched under the named label with its consult
        # run, a hand-CUDA kernel must have launched, and the comparator must have
        # seen its planted word. The default-precedence leg is exempt: the incumbent
        # composes first there and may hold every slot.
        required = dict(spec.get("singles") or {})
        on_this_leg = bool(required) and BACKEND == "cuda" and (
            UNCERTIFIED_POLICY_INSTALLED is None
            and (TABLE_PREFERENCE == "cuda" or TRITON_WITHHELD is not None))
        missing: List[str] = []
        if on_this_leg:
            for slot, label in sorted(required.items()):
                arm = block["arms"].get(slot)
                runs = (block["counters"].get(slot) or {}).get("dispatches")
                if arm != label:
                    missing.append(f"{slot} dispatched {arm!r}, not {label!r}")
                elif not runs:
                    missing.append(f"{slot} carries {label!r} but its consult ran "
                                   f"{runs!r} times")
            if not (fused.get("cuda_launches") or {}).get("total"):
                missing.append("no hand-CUDA kernel launched on the fused leg")
            if not (control.get("armed") and control.get("comparator_saw_it")
                    and control.get("words_mismatched") == 1):
                missing.append("the comparator did not see its planted word, so the "
                               "byte identity is not evidence about these arms")
        row["no_fused_arm"]["singles"] = {"required": required,
                                          "required_on_this_leg": on_this_leg,
                                          "failures": missing}
        if driven:
            row["verdict"] = "FUSED-WITHOUT-A-RELEASE"
            row["why"] = (f"a fused product was driven here ({sorted(set(driven.values()))}); "
                          "this row expects none, because no fused label is released "
                          "on its shape and the composer is not offered one")
        elif reason and classify_refusal(reason) is None:
            row["verdict"] = "REFUSAL-CHANGED"
            row["why"] = (f"the plan was refused and the refusal matches no rung "
                          f"this gate knows: {reason!r}")
        elif not agree:
            row["verdict"] = "DIVERGENCE"
        elif missing:
            row["verdict"] = "SINGLES-NOT-DISPATCHED"
            row["why"] = "; ".join(missing)
        else:
            row["verdict"] = "PASS-NO-FUSED-ARM"
        return row

    if spec["expect"] == "refused-folded":
        # THE REFUSAL THAT MUST STAND, MEASURED IN BOTH DIRECTIONS. Nothing about a
        # folded configuration is allowed to dispatch under this gate — that is the
        # first clause, and it is the one that would fire if a later phase retired a
        # rung without a gate. The second clause is which RUNG answered: the plan
        # reaches several that all refuse the same population, and a refusal that
        # moved from one to another is a change in what is refused, so the rung is
        # NAMED in the artifact rather than absorbed by a text match that happens to
        # still pass. An unrecognised rung fails.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        rung = classify_refusal(reason)
        driven = ((report.get("fusion") or {}).get("driven") or {})
        row["fold_refusal"] = {
            "refused_because": reason,
            "refusing_rung": rung,
            "clause_8_admitted_the_arm": dict(driven),
            "did_not_dispatch": fused["active_step_path"] == "array",
            "unfused_leg_dispatched": unfused["active_step_path"] == "fused",
        }
        if dispatched:
            row["verdict"] = "REFUSAL-RETIRED-WITHOUT-A-GATE"
            row["why"] = ("a folded configuration dispatched a fused pair; this "
                          "gate does not license that and the rung must stand")
        elif rung is None:
            row["verdict"] = "REFUSAL-CHANGED"
            row["why"] = (f"the refusal matches no rung this gate knows: {reason!r}")
        elif not agree:
            row["verdict"] = "DIVERGENCE"
        else:
            row["verdict"] = "PASS-REFUSAL-STANDS"
        return row

    if not row["observables"]["migration_agrees"]:
        # A MONITOR BUILT AND NOT MIGRATED WOULD SILENCE THE FLUX CLAUSE. The
        # applicability above is read off the builder's return; this is the check
        # that the driver received what the builder returned, so "this case declares
        # no monitor" can never be reached by a monitor that was dropped in the lift.
        row["verdict"] = "MONITOR-MIGRATION-MISMATCH"
        row["why"] = (f"the builder returned "
                      f"{row['observables']['declared_by_the_builder']} monitor(s) "
                      f"and {row['observables']['monitors']} reached the driver, so "
                      "the flux comparison is neither the case's nor absent")
    elif not agree:
        row["verdict"] = "DIVERGENCE"
        if (row["final_state_fused_vs_array"]["identical"]
                and row["observables"]["identical"]):
            row["why"] = ("the FINAL state and the observables agree, but the fused "
                          f"and array paths differed at "
                          f"{row['first_divergent_checkpoint']} steps and the "
                          "difference was later absorbed; they computed different bits")
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
    elif not dispatched and TRITON_WITHHELD is not None:
        # THE TRITON-LESS HOST HAS NO INCUMBENT, SO "DID NOT FUSE" IS A REGRESSION
        # HERE. On the default-precedence leg a case expecting 'dispatch' that fused
        # nothing can mean the incumbent table held the seam; on this leg there is no
        # incumbent to hold it. Every 'dispatch' row of DRIVE_CUDA names this table's
        # own fused products, and a row whose licence this leg does not carry has
        # already returned PASS-NOT-THIS-LEG above -- so no case legitimately reaches
        # this line, and it fails exactly as DID-NOT-FUSE fails on a by-preference
        # leg. A refusal at a rung this gate does not recognise fails as
        # REFUSAL-CHANGED, the guard the no-fused-arm branch carries. WHY is still
        # recorded off the record, so a failing Triton-less row names its rung.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        row["alone"] = {
            "refused_because": reason,
            "refusing_rung": classify_refusal(reason) if reason else None,
            "dispatched_slots": fusion_block(fused)["dispatched_slots"],
            "served_by_table": list((report.get("composition") or {})
                                    .get("tables_dispatched") or []),
        }
        if reason and classify_refusal(reason) is None:
            row["verdict"] = "REFUSAL-CHANGED"
            row["why"] = (f"with Triton withheld the plan was refused and the "
                          f"refusal matches no rung this gate knows: {reason!r}")
        else:
            row["verdict"] = "DID-NOT-FUSE"
            row["why"] = (
                "with Triton withheld and the preference unset no fused product was "
                "driven on a case that expects one: "
                + (f"the plan was refused ({reason})" if reason else
                   f"the seam ran {row['alone']['served_by_table'] or 'no table'}'s "
                   f"separate certified arms on {row['alone']['dispatched_slots']}"))
    elif not dispatched and BACKEND == "cuda" and TABLE_PREFERENCE is None:
        # THE LEG THAT MEASURES WHAT A USER WHO SETS NOTHING GETS, and on some shapes
        # the answer is "no fused product at all". Under the shipped precedence the
        # release-gated Triton table composes first and a PENDING primary arm refuses
        # the whole plan at rung 7a rather than yielding its slots
        # (``YIELD_PENDING_PRIMARY_SLOTS`` ships False, with its reason recorded), so
        # a shape Triton declines does NOT fall through to this table by default.
        # Measured 2026-09-11 on pml_3d_diagonal and special_kz_2d: four Triton
        # launches per step from separate certified arms and no fused label anywhere.
        # That is this leg's RESULT -- it is exactly the number the record publishes
        # beside the by-preference one -- and scoring it DID-NOT-FUSE failed the leg
        # for measuring what it exists to measure. Guarded on the preference being
        # UNSET, so a by-preference leg that fuses nothing still fails.
        row["verdict"] = "PASS-BY-DEFAULT-NOT-FUSED"
        row["why"] = (
            "with the backend preference unset no fused product was admitted on "
            f"this shape: the seam ran {sorted(set((row.get('evidence') or {}).get('served_by_table') or [])) or 'no dispatching table'}'s "
            "separate certified arms, which is the by-default answer this leg "
            "publishes beside the by-preference one")
    elif not dispatched:
        row["verdict"] = "DID-NOT-FUSE"
        row["why"] = ((fused.get("report") or {}).get("refused_because")
                      or "the fused route was asked for and did not dispatch")
    elif not row["evidence"]["real"]:
        row["verdict"] = "VACUOUS-PASS"
        row["why"] = "; ".join(row["evidence"]["failures"])
    elif not row["array_leg_control"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
    elif (row["substitution"]["verdict"] not in EXACT_SUBSTITUTION_VERDICTS
          and not tritonless_baseline_only(row["substitution"], spec)):
        # EXACT OR NOTHING, and the enumeration used to be the other way round: the
        # chain listed the failing verdicts, so ``NOT-COMPARABLE`` and
        # ``DROPPED-BUT-NOT-EXACT`` — a mixed slot set, and a drop of the wrong size
        # — fell through to PASS-FUSED. Every case reaching this line expects a
        # substitution (``pairs`` > 0; the refusal cases returned above), so the
        # claim in this file's docstring is "exactly one launch fewer per pair per
        # step" and the verdict now says so directly.
        row["verdict"] = "NO-SUBSTITUTION"
        row["why"] = (row["substitution"].get("why")
                      or "the fused leg did not launch exactly one kernel fewer per "
                         "pair per step than the unfused leg, so the byte identity "
                         "is about two spellings of the same launches")
    elif case_verdict_for_second_consult(row["second_consult_site"]):
        row["verdict"] = case_verdict_for_second_consult(row["second_consult_site"])
        row["why"] = (row["second_consult_site"].get("why")
                      or "the magnetic half-step's own consults did not compose the "
                         "way the step's did")
    else:
        row["verdict"] = "PASS-FUSED"
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: ``PASS-NOT-THIS-LEG`` is a SKIP and not a statement about the product: the case
#: declares an expansion-probe record this leg does not export, so no arm of it
#: could be admitted here at all. It keeps four legs from failing on the two cases
#: the fifth drives, and it cannot mask a real refusal, because the probe leg DOES
#: export the variable and therefore never reaches it.
PASS_VERDICTS = ("PASS-FUSED", "PASS-REFUSAL-STANDS", "PASS-POLICY-REFUSED",
                 "PASS-NO-FUSED-ARM", "PASS-NOT-THIS-LEG",
                 "PASS-BY-DEFAULT-NOT-FUSED")
#: THE THREE ARBITRATION VERDICTS JOIN THE PASS SET, and they are separate names
#: rather than a shared "OK" because each licenses a different number: the
#: by-preference count rests on INCUMBENT-YIELDED, the by-default count on
#: ARBITRATION-DEFAULT-HELD, and the refusal-by-name control on
#: PREFERENCE-REFUSED-BY-NAME. A run missing one of them has not measured what the
#: record it feeds says it measured.
CONTROL_PASS = ("DIVERGED-AS-REQUIRED", "NOT-APPLICABLE", "REFUSED-BY-NAME",
                "INCUMBENT-YIELDED", "ARBITRATION-DEFAULT-HELD",
                "ARBITRATION-DEFAULT-HELD-UNCONTESTED",
                "PREFERENCE-REFUSED-BY-NAME",
                "RELEASE-ADMITTED", "ENVELOPE-HOLDS",
                # The cuda_alone leg's two arbitration reads (sole_table_verdict).
                "SOLE-TABLE-COMPOSED")


#: THE BRACKET'S OWN BYTES, added to the curated ``provenance.source_sha256`` block
#: that ``gate_dispatch_end_to_end._provenance`` builds. That list digests the seam,
#: both NVIDIA tables and the CUDA composer, but not ``deposit_repair.py`` -- the module
#: whose ``LeadingRepairPlan`` / ``TrailingRepairPlan`` wrap every bracketed seam's
#: launch on all three tables. Read 2026-09-19 off the three 2026-09-17 ``allpaths``
#: campaigns (14 legs): every leg names the bracket only in the top-level
#: ``imported_source_sha256`` (``meep_gpu/deposit_repair.py`` at ``c1d55704``), and the
#: block ``recut_driver_dispatch_record.py`` compares -- 39 keys on each fused-route
#: leg -- does not carry it. So the byte-parity evidence behind ``served_in_dispatch``
#: on a bracketed seam was recorded against bytes no curated block named.
#: ``gate_driver_route_fused.py:983`` already digests the same file.
#:
#: PACKAGE-RELATIVE, the one spelling this map uses (the "ONE SPELLING PER MAP"
#: comment in ``gate_dispatch_end_to_end._provenance``). ``recut_driver_dispatch_record
#: ._leg_key`` strips ``meep_gpu/`` before it looks a bound path up, so a record that
#: binds ``meep_gpu/deposit_repair.py`` resolves to this key.
#:
#: ``cuda_kernels/fused_pairs.py`` IS NOT LISTED: the end-to-end list already digests
#: it under that key, and all 9 fused-route legs above carry it (``283722e7``); the 5
#: Metal legs record ``cuda_kernels/fused_pairs.py`` in neither block.
#:
#: RECORDED, NOT BOUND. The recut binds the key set of the record it cuts
#: (``bound = sorted(record["source_sha256"])``, ``recut_driver_dispatch_record.py:310``)
#: and ignores every other leg key, so this addition changes no record and makes no recut
#: refuse. Whether a ``driver_dispatch`` record should bind the bracket is that record's
#: release decision; binding it needs the key in the record AND a campaign that ran with
#: this block, because no earlier leg carries it here.
BRACKET_SOURCES = ("deposit_repair.py",)


def _provenance(gpu_id: int) -> Dict[str, Any]:
    """The end-to-end gate's provenance, plus :data:`BRACKET_SOURCES` in its digest map.

    Host-only: reads files and versions and launches nothing, so it answers off-device.
    A key the end-to-end list already digested keeps that digest; a DIFFERENT digest for
    it raises, because the file was rewritten between the two reads and a leg recording
    either value would describe a program it may not have run.
    """
    record = e2e._provenance(gpu_id)  # noqa: SLF001
    digests = record["source_sha256"]
    package = os.path.join(_API, "meep_gpu")
    for name in BRACKET_SOURCES:
        with open(os.path.join(package, name), "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        recorded = digests.setdefault(name, digest)
        if recorded != digest:
            raise RuntimeError(
                f"{name} read {recorded[:16]}... then {digest[:16]}... inside one "
                f"provenance call: the file changed while the gate recorded it")
    return record


# ---------------------------------------------------------------------------
# A route leg starts only on a card the table under test certifies
# ---------------------------------------------------------------------------

def card_capability(gpu_id: int) -> Optional[str]:
    """Device ``gpu_id``'s compute capability, spelled as the ledgers key it. Never raises.

    ``None`` when it cannot be read. The device is the one the legs lift onto
    (``lift_simulation(gpu_id=...)``), read the way ``fastpath._device_identity``
    reads it.
    """
    try:
        import cupy  # noqa: PLC0415
        from meep_gpu import fastpath  # noqa: PLC0415

        properties = cupy.cuda.runtime.getDeviceProperties(gpu_id)
        return fastpath._normalized_capability(  # noqa: SLF001
            (properties["major"], properties["minor"]))
    except Exception:  # noqa: BLE001 - an unreadable card is refused, not a crash
        return None


def installed_triton_version() -> Optional[str]:
    """The importable Triton's version, as ``fastpath`` records it, or ``None``."""
    try:
        import triton  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no Triton is refused on the Triton table
        return None
    return str(getattr(triton, "__version__", "unknown"))


def uncertified_card_refusal(table: str, capability: Optional[str],
                             triton_version: Optional[str] = None) -> Optional[str]:
    """Why a route leg of ``table`` may not start on this card, or ``None``.

    A supported NVIDIA card that is not certified dispatches by default, uncertified,
    and ``recut_driver_dispatch_record.py`` refuses every such row only after the
    campaign has spent its GPU time. So a leg starts only when the card's compute
    capability is in ``fastpath.capability_admission(table)["admitted"]`` and, on the
    Triton table, whose identity is the device and the Triton version together, when
    the installed Triton is in ``fastpath.validated_triton_versions()``. A capability
    or a version that could not be read is not certified. ``--smoke`` lifts the NumPy
    reference and is never asked.

    The end-to-end licence gate is not guarded here: it refuses every ``MEEP_GPU_*``
    variable and reads no admission, so the admission check before it stays a step of
    the round.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    tail = ("; a supported card that is not certified dispatches by default, "
            "uncertified, and recut_driver_dispatch_record.py refuses every such row. "
            "Bind the card's family gates first (docs/development/certification.md, "
            "steps 1 to 3), then start the route campaign")
    if capability is None:
        return (f"REFUSING: this route leg drives the {table} table and the card's "
                f"compute capability could not be read; a card that cannot be judged "
                f"is not certified" + tail)
    admitted = fastpath.capability_admission(table)["admitted"]
    if not admitted or capability not in admitted:
        return (f"REFUSING: this route leg drives the {table} table on compute "
                f"capability {capability}, which that table's records do not certify "
                f"(capability_admission({table!r}) admits "
                f"{list(admitted) if admitted is not None else 'nothing readable'})"
                + tail)
    if table == "triton":
        validated = fastpath.validated_triton_versions()
        if triton_version not in validated:
            return (f"REFUSING: this route leg drives the triton table with Triton "
                    f"{triton_version or 'not importable'}, which that table's records "
                    f"do not certify (validated_triton_versions() is "
                    f"{list(validated)})" + tail)
    return None


def uncertified_switch_refusal(environ: Mapping[str, str]) -> Optional[str]:
    """Why a route leg may not start under an exported uncertified switch, or ``None``.

    :func:`uncertified_card_refusal` judges the table under test alone, and no leg
    sets ``fastpath.UNCERTIFIED_SWITCH``, so a value the invoking shell exports
    reaches every leg. ``1`` lifts rung 4e, so a table that only supports the card
    composes, uncertified, beside the certified table under test; ``0`` restricts the
    kernels to certified identities, which is not the default a user who sets nothing
    gets; any other value is refused at rung 3b and the leg takes the array path. A
    route leg therefore runs with the switch unset, as the round scripts leave it.
    ``--smoke`` lifts the NumPy reference and is never asked.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    value = environ.get(fastpath.UNCERTIFIED_SWITCH)
    if value is None:
        return None
    return (f"REFUSING: {fastpath.UNCERTIFIED_SWITCH}={value!r} is set; a route leg "
            "runs with it unset, so that it measures the default and no table the "
            "card is not certified for composes beside the table under test. Unset "
            "it, then start the leg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--backend", default="triton", choices=("triton", "cuda"),
                        help="which kernel table to drive through the seam. "
                             "'triton' is byte-for-byte what this gate always ran; "
                             "'cuda' swaps the per-case expectations for "
                             "DRIVE_CUDA, sets MEEP_GPU_BACKEND_PREFERENCE on the "
                             "dispatching legs, and installs the hand-CUDA launch "
                             "witnesses beside the Triton one.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap the ladder; the full run is the default")
    parser.add_argument("--install-policy", default=None,
                        help="install this float32 subnormal policy before "
                             "anything else. UNSET IS THE GATE RUN: it leaves the "
                             "install to fastpath rung 8b, which is precisely the "
                             "evidence DISPATCH_BY_DEFAULT is waiting on.")
    parser.add_argument("--table-preference", default="cuda",
                        choices=("cuda", "unset"),
                        help="CUDA ONLY, and it names the leg. 'cuda' sets the "
                             "backend-preference switch and measures the "
                             "BY-PREFERENCE number (the cuda_shipped family of "
                             "legs); 'unset' leaves it alone and measures what "
                             "dispatches under the shipped precedence, which is "
                             "the cuda_default_precedence leg and the only source "
                             "of the by-default number.")
    parser.add_argument("--without-triton", action="store_true",
                        help="CUDA ONLY, and it names the leg: make Triton "
                             "unimportable in this process before anything imports "
                             "it (sys.modules['triton'] = None), so rung 4 drops "
                             "the Triton table by name and the hand-CUDA table "
                             "composes alone -- the cuda_alone leg, which runs with "
                             "--table-preference unset and is the only leg this "
                             "flag may be passed to.")
    parser.add_argument("--no-controls", action="store_true",
                        help="skip the armed controls (diagnostic only; a run "
                             "made with this is not a gate run)")
    parser.add_argument("--smoke", action="store_true",
                        help="NOT A GATE RUN. Lift the NumPy reference "
                             "(prefer_gpu=False) and tolerate a host with no "
                             "Triton, so the harness's own plumbing can be shaken "
                             "out on a laptop. The reference consults no kernel "
                             "table, so nothing about the fused route is measured "
                             "and the artifact says so. Without it the gate refuses "
                             "a host with no CUDA device.")
    arguments = parser.parse_args()

    global BACKEND  # noqa: PLW0603 - the run axis, set once before any leg builds
    BACKEND = arguments.backend

    # THE LEG NAME AND THE SWITCH MUST AGREE, checked here rather than trusted to a
    # campaign script. ``recut_driver_dispatch_record.py`` writes a ``legs`` map
    # into the published record saying that ``cuda_shipped`` ran with the
    # preference set and ``cuda_default_precedence`` ran without it; nothing
    # downstream re-reads the environment, so a leg run under the wrong switch
    # would publish a by-preference measurement AS the by-default number and every
    # artifact would look right. The directory leaf is the leg's name -- that is how
    # the recut discovers legs at all -- so the two are comparable right here.
    global TABLE_PREFERENCE  # noqa: PLW0603 - the leg axis, set once, same reason
    TABLE_PREFERENCE = None if arguments.table_preference == "unset" else "cuda"
    leg_name = os.path.basename(os.path.normpath(arguments.out))
    if arguments.backend != "cuda" and TABLE_PREFERENCE is None:
        print("REFUSING: --table-preference unset is a CUDA leg; this run drives "
              f"{arguments.backend}, where there is no second NVIDIA table to "
              "yield to and the switch means nothing", file=sys.stderr)
        return 2
    if arguments.backend == "cuda" and leg_name.startswith("cuda_"):
        wants_unset = leg_name in UNSET_PREFERENCE_LEGS
        if wants_unset != (TABLE_PREFERENCE is None):
            print(f"REFUSING: {leg_name!r} is "
                  + ("a leg that measures what a user who sets nothing gets and "
                     "must run with --table-preference unset"
                     if wants_unset else
                     "a by-preference leg and must run with --table-preference "
                     "cuda")
                  + f"; this run has --table-preference "
                    f"{arguments.table_preference!r}", file=sys.stderr)
            return 2
    # THE FLAG AND THE LEG NAME MUST AGREE, for the reason the preference must: the
    # directory leaf is how the recut discovers a leg, and a cuda_alone artifact cut
    # with Triton importable -- or a by-preference leg cut without it -- would publish
    # one host's composition under the other's name with every byte comparison green.
    if bool(arguments.without_triton) != (arguments.backend == "cuda"
                                          and leg_name == TRITON_ABSENT_LEG):
        print(f"REFUSING: --without-triton is {bool(arguments.without_triton)} on "
              f"leg {leg_name!r} (backend {arguments.backend!r}); it is required on "
              f"the {TRITON_ABSENT_LEG!r} leg of the cuda campaign and refused on "
              "every other", file=sys.stderr)
        return 2
    if arguments.without_triton:
        # FIRST, before anything below can import Triton -- the launch counter's
        # install does, and so does the ladder's rung 4.
        global TRITON_WITHHELD  # noqa: PLW0603 - the leg axis, set once
        try:
            TRITON_WITHHELD = withhold_triton()
        except RuntimeError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 3
    # AN NVIDIA GATE, REFUSED OFF NVIDIA before its leg directory exists: on an Apple
    # host prefer_gpu=True resolves the Metal table, which this gate does not grade.
    refusal = (None if arguments.smoke
               else e2e.nvidia_host_refusal("gate_dispatch_fused_route.py"))
    # AND ONLY ON A CARD THE TABLE UNDER TEST CERTIFIES, before the leg directory
    # exists (:func:`uncertified_card_refusal`). The Triton version is read on the
    # Triton table only: the ``cuda_alone`` leg has withheld Triton by now.
    if refusal is None and not arguments.smoke:
        refusal = uncertified_card_refusal(
            arguments.backend, card_capability(arguments.gpu_id),
            installed_triton_version() if arguments.backend == "triton" else None)
    # AND WITH THE UNCERTIFIED SWITCH UNSET (:func:`uncertified_switch_refusal`): the
    # check above admits the table under test alone, and an exported ``=1`` would
    # let the other table compose beside it uncertified.
    if refusal is None and not arguments.smoke:
        refusal = uncertified_switch_refusal(os.environ)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    os.makedirs(arguments.out, exist_ok=True)

    # A GATE MUST NOT WRITE INTO ANOTHER RUN'S ARTIFACT. ``cases.jsonl`` and
    # ``controls.jsonl`` are opened in APPEND mode, because progress reporting wants every row
    # on disk the moment it lands; the cost of that is that pointing --out at a
    # directory a previous run already wrote silently INTERLEAVES two runs' rows
    # in one file. ``summary.json`` is then rewritten from THIS process's in-memory
    # list, so the two files disagree and only the JSONL carries the contamination
    # -- the half a reader is most likely to trust and least likely to re-derive.
    #
    # MEASURED 2026-08-29, on this gate: a campaign relaunched into the previous
    # attempt's directory appended a second run's pml_2d and conductive_2d rows
    # under the first's, on a different GPU, with a different gate sha. The run was
    # killed and restarted into a fresh directory. Refusing is the fix; the
    # alternative -- truncating -- destroys the earlier run instead.
    existing = sorted(name for name in ("cases.jsonl", "controls.jsonl",
                                        "summary.json", "gate.json")
                      if os.path.exists(os.path.join(arguments.out, name)))
    if existing:
        print(f"REFUSING: {arguments.out} already holds a previous run's "
              f"{', '.join(existing)}. This gate appends its rows, so writing here "
              f"would interleave two runs in one artifact. Point --out at a fresh "
              f"directory.", file=sys.stderr)
        return 5

    e2e._PROGRESS_PATH = os.path.join(arguments.out, "progress.log")
    e2e.PREFER_GPU = not arguments.smoke
    rows_path = os.path.join(arguments.out, "cases.jsonl")

    drive = drive_table(arguments.backend)
    names = (list(drive) if arguments.cases == "all"
             else [n.strip() for n in arguments.cases.split(",") if n.strip()])
    unknown = [n for n in names if n not in drive]
    if unknown:
        print(f"unknown cases: {unknown}; known: {sorted(drive)}", file=sys.stderr)
        return 2

    if TRITON_WITHHELD is not None:
        # NO TRITON WITNESS IS POSSIBLE AND NONE IS NEEDED: the process cannot import
        # Triton, which withhold_triton verified, so its launch count is zero by
        # construction and the stand-in says so rather than pretending to watch.
        counter: Any = AbsentTritonCounter()
        say(f"Triton withheld for this leg ({TRITON_WITHHELD['mechanism']}; import "
            f"now raises {TRITON_WITHHELD['import_error']}); an installed Triton "
            f"{TRITON_WITHHELD['installed_triton_version']!r} is "
            f"{'hidden' if TRITON_WITHHELD['installed_triton_hidden'] else 'absent'}"
            "; the hand-CUDA witnesses carry this leg")
    else:
        counter = e2e.TritonLaunchCounter()
        try:
            counter.install()
            say("Triton launch counter installed on JITFunction.run and "
                "CompiledKernel.launch_enter_hook")
        except Exception as exc:  # noqa: BLE001
            if not arguments.smoke:
                say(f"REFUSING: the Triton launch counter could not be installed "
                    f"({exc!r}); without it a fallback that matched could not be "
                    "told from a dispatch")
                return 3
            say(f"smoke run: no Triton launch counter ({exc!r})")

    # THE HAND-CUDA WITNESSES, INSTALLED BEFORE ANY LIFT. Both are package-seam
    # counters (see ``e2e.CudaLaunchCounter``) and neither reads anything
    # ``fastpath`` owns, so a defect in the dispatcher's own bookkeeping cannot make
    # them agree by construction. The array leg must read ZERO on both, and the two
    # must agree with each other on every dispatching leg.
    global ENVELOPE_CASE, ENVELOPE_WITHHOLD  # noqa: PLW0603 - per-table control
    _envelope = ENVELOPE_ROWS.get(arguments.backend) or ENVELOPE_ROWS["triton"]
    ENVELOPE_CASE = _envelope["case"]
    ENVELOPE_WITHHOLD = list(_envelope["withhold"])

    global CUDA_COUNTER  # noqa: PLW0603 - the run's second witness, set once
    if arguments.backend == "cuda":
        cuda_counter = e2e.CudaLaunchCounter()
        cuda_counter.install()
        # THE LINE THIS BLOCK WAS MISSING. Until it was added the counter was built,
        # installed, reported -- and then the local name died at the end of this
        # block, so every clause downstream read the TRITON counter and a run in
        # which only hand-CUDA kernels launched was scored as having launched
        # nothing. The comment above already said the array leg must read zero on
        # both witnesses; nothing implemented that sentence.
        CUDA_COUNTER = cuda_counter
        snapshot = cuda_counter.snapshot()
        if not snapshot["installed"] and not arguments.smoke:
            say(f"REFUSING: no hand-CUDA launch witness could be installed "
                f"({snapshot['witnesses']}); without one a fallback that matched "
                "could not be told from a dispatch")
            return 3
        say(f"hand-CUDA launch witnesses: {snapshot['witnesses']}")

    provenance = _provenance(arguments.gpu_id)
    provenance["gate"] = "dispatch_fused_route"
    provenance["backend"] = arguments.backend
    provenance["campaign_legs"] = list(CAMPAIGN_LEGS[arguments.backend])
    if arguments.backend == "cuda":
        from meep_gpu import fastpath as _fp_arb  # noqa: PLC0415

        provenance["backend_precedence"] = list(_fp_arb.BACKEND_PRECEDENCE)
        provenance["backend_preference_switch"] = _fp_arb.BACKEND_PREFERENCE_SWITCH
        # TWO KEYS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND USED TO BE ONE.
        # ``leg_table_preference`` is what every DRIVE row of this leg sets, and it
        # is the field that says which of the two NVIDIA legs this artifact is.
        # ``ambient_backend_preference`` is what the invoking shell happened to
        # export, which the drive rows override and which is recorded only so a run
        # launched inside a stray export is readable after the fact. Reading the
        # ambient value alone -- which is what this block did -- reported "unset" on
        # every leg including the by-preference one.
        provenance["leg_table_preference"] = TABLE_PREFERENCE
        provenance["ambient_backend_preference"] = os.environ.get(
            _fp_arb.BACKEND_PREFERENCE_SWITCH)
        # HOW THIS LEG'S HOST WAS MADE TRITON-LESS, or None on every leg that ran
        # with whatever Triton the host has.
        provenance["triton_withheld"] = TRITON_WITHHELD
    provenance["smoke_run_measures_nothing_about_the_fused_route"] = bool(arguments.smoke)
    provenance["install_policy_requested"] = arguments.install_policy
    provenance["harness_installs_nothing"] = arguments.install_policy is None
    if arguments.install_policy:
        # MEEP FIRST, and not as a formality. ``install_host_policy`` drives the
        # host FPU through MEEP's ``set_zero_subnormals`` and calls a request it
        # cannot verify UNATTAINABLE when MEEP is not imported here — so a 'flush'
        # leg run without this import installs a policy on two executors and warns
        # about the third, which is the split the whole rung exists to refuse.
        import meep  # noqa: PLC0415,F401
        import meep_gpu  # noqa: PLC0415
        try:
            provenance["installed_policy"] = dict(
                meep_gpu.install_subnormal_policy(arguments.install_policy))
            say(f"installed subnormal policy {arguments.install_policy!r}")
        except BaseException as exc:  # noqa: BLE001
            say(f"REFUSING: install_subnormal_policy raised {exc!r}")
            return 4
        # WHICH LEG THIS IS, decided against fastpath's own constant rather than a
        # literal here, so a change to the certified policy moves this leg with it
        # instead of leaving a gate asserting a policy nobody certifies any more.
        from meep_gpu import fastpath as _fp  # noqa: PLC0415
        certified = _fp.CERTIFICATION_SUBNORMAL_POLICY
        provenance["certification_policy"] = certified
        if arguments.install_policy != certified:
            global UNCERTIFIED_POLICY_INSTALLED  # noqa: PLW0603
            UNCERTIFIED_POLICY_INSTALLED = arguments.install_policy
            say(f"the harness installed {arguments.install_policy!r} and every "
                f"dispatchable family was certified under {certified!r}: this leg "
                "REQUIRES rung 8b to refuse by name on every dispatch case, with "
                "the run still byte-identical to the array path")
    provenance["uncertified_policy_installed"] = UNCERTIFIED_POLICY_INSTALLED
    provenance["policy_at_start"] = e2e._policy_stamp()
    with open(os.path.join(arguments.out, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, default=str)
    say(f"host={provenance['host']} device={provenance['device']} "
        f"triton={provenance['triton']} meep={provenance['meep']} "
        f"policy={provenance['policy_at_start']}")

    rows: List[Dict[str, Any]] = []
    controls: List[Dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        say(f"=== case {index}/{len(names)}: {name} ===")
        started = time.time()
        try:
            row = run_case(name, arguments.gpu_id, counter, arguments.resolution,
                           arguments.max_steps)
        except BaseException as exc:  # noqa: BLE001 - a raising case is a recorded row
            import traceback  # noqa: PLC0415
            row = {"case": name, "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:2000],
                   "traceback": traceback.format_exc()[-4000:]}
        row["case_wall_s"] = round(time.time() - started, 2)
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:  # progress reporting, incremental
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
        say(f"=== case {index}/{len(names)}: {name} -> {row.get('verdict')} "
            f"({row['case_wall_s']} s) ===")

        # THE CONTROLS FOLLOW THE CASE. A case skipped because this leg carries no
        # probe for it has no fused launch to arm anything against, and the armed
        # controls would each record NOT-ARMED -- which the release reads as a
        # control that failed.
        if (arguments.no_controls or drive[name]["expect"] != "dispatch"
                or row.get("verdict") in ("PASS-NOT-THIS-LEG",
                                          "PASS-BY-DEFAULT-NOT-FUSED")):
            # A CASE THAT FUSED NOTHING BY DEFAULT HAS NO SEAM TO ARM AGAINST. The
            # withheld consult needs an absorbed consult to withhold and the in-seam
            # mutation needs an in-launch twin to be compared against; with no fused
            # product both record NOT-ARMED, which the release reads as a control
            # that failed. Measured 2026-09-11 on pml_3d_diagonal and special_kz_2d.
            continue

        def record(entry: Dict[str, Any], case_name: str = name) -> None:  # noqa: E306
            """Append one control row and FLUSH IT, one row per control.

            Written as each control lands rather than in a per-case block, for the
            two reasons progress reporting gives. A block write loses every control a crash
            interrupts; and the block this replaced took a FIXED ``controls[-4:]``
            slice against a per-case count that varies with ``seam`` (four rows for
            ``pml_2d``, three for ``conductive_2d``), so a three-control case
            re-wrote the previous case's last row into ``controls.jsonl``. The
            in-memory ``controls`` list was right and the file was not, which is the
            worse half of that failure: the duplicate is only visible to a reader of
            the incremental artifact.
            """
            entry["case"] = case_name
            controls.append(entry)
            with open(os.path.join(arguments.out, "controls.jsonl"), "a",
                      encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
                handle.flush()

        if UNCERTIFIED_POLICY_INSTALLED is not None:
            # THE ARMED CONTROLS CANNOT ARM ON THIS LEG, and the honest record says
            # that rather than reporting three NOT-ARMED rows as failures. Each of
            # them needs a fused launch to sit beside — an absorbed consult to
            # withhold, an in-seam pass between two consults — and rung 8b refused
            # the plan whole, so there is no seam to instrument. What this leg
            # measures is the refusal itself, which is the case verdict above.
            record({"control": "armed-controls",
                    "verdict": "NOT-APPLICABLE",
                    "why": (f"the harness installed "
                            f"{UNCERTIFIED_POLICY_INSTALLED!r}, so rung 8b refused "
                            "the plan and no fused launch exists to withhold or to "
                            "put a mutated in-seam pass beside")})
            continue

        if arguments.backend == "cuda":
            # THE ARBITRATION CONTROLS RUN FIRST, because the two below are about a
            # composition and these are about WHICH composition: a withheld-consult
            # control on a step the incumbent actually served would be arming the
            # wrong table's seam.
            try:
                for entry in arbitration_leg(name, drive[name], arguments.gpu_id,
                                             arguments.resolution):
                    record(entry)
            except BaseException as exc:  # noqa: BLE001 - a raising control is a row
                record({"control": "arbitration", "verdict": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}"[:2000]})

        for maker in (
            lambda: withheld_control(name, drive[name], arguments.gpu_id,
                                     arguments.resolution, counter),
            lambda: release_route_leg(name, drive[name], arguments.gpu_id,
                                      arguments.resolution),
        ):
            try:
                entry = maker()
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": "raised", "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
            record(entry)
        for side in drive[name]["seam"]:
            try:
                entry = seam_mutation_control(name, drive[name], side,
                                              arguments.gpu_id,
                                              arguments.resolution, counter)
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": f"in-seam-pass-mutated-{side}",
                         "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
            record(entry)

    # THE ENVELOPE, ONCE PER RUN AND NOT PER CASE. It is a property of the RELEASE,
    # not of any case here: it drives its own out-of-envelope shape (pml_3d) and
    # asks the two questions no in-envelope case can reach — that the envelope
    # declines a request, and that clause (8) still refuses by name what neither the
    # switch nor the release admits. Running it per case would repeat one
    # measurement five times and say nothing more.
    if not arguments.no_controls:
        if UNCERTIFIED_POLICY_INSTALLED is not None:
            entry = {"control": "envelope", "verdict": "NOT-APPLICABLE",
                     "why": (f"the harness installed "
                             f"{UNCERTIFIED_POLICY_INSTALLED!r}, so rung 8b refuses "
                             "before clause (8) is reached and neither direction of "
                             "the envelope is observable on this leg")}
        else:
            try:
                entry = envelope_leg(arguments.gpu_id, arguments.resolution)
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": "envelope", "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
        entry.setdefault("case", ENVELOPE_CASE)
        entry.setdefault("control", "envelope")
        controls.append(entry)
        with open(os.path.join(arguments.out, "controls.jsonl"), "a",
                  encoding="utf-8") as handle:   # progress reporting: flushed as it lands
            handle.write(json.dumps(entry, default=str) + "\n")
            handle.flush()

    failures = [r["case"] for r in rows if r.get("verdict") not in PASS_VERDICTS]
    control_failures = [f"{c.get('case')}:{c.get('control') or c.get('leg')}"
                        for c in controls if c.get("verdict") not in CONTROL_PASS]
    dispatched = [r["case"] for r in rows if r.get("verdict") == "PASS-FUSED"]
    policy_refused = [r["case"] for r in rows
                      if r.get("verdict") == "PASS-POLICY-REFUSED"]
    # WHAT THIS LEG HAD TO SHOW, which is not the same question on both legs. A
    # normal leg is released by a fused pair reaching the seam; the uncertified-
    # policy leg is released by rung 8b refusing every dispatch case BY NAME while
    # the answer stays byte-identical, and requiring a dispatch there would ask a
    # leg to prove the thing it exists to show cannot happen.
    demonstrated = policy_refused if UNCERTIFIED_POLICY_INSTALLED else dispatched
    released = bool(demonstrated) and not failures and not control_failures
    reasons: List[str] = []
    if failures:
        reasons.append(f"cases not passing: {failures}")
    if control_failures:
        reasons.append(f"controls not passing: {control_failures}")
    if not demonstrated and UNCERTIFIED_POLICY_INSTALLED:
        reasons.append(f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r} and "
                       "no case was refused by name at the subnormal-policy rung, "
                       "so nothing about the seam's single-policy rule was measured")
    elif not demonstrated:
        reasons.append("no case dispatched a fused pair through the driver seam, "
                       "so nothing about the fused route was measured")
    # A LEG THAT MEASURES ZERO MUST NOT RELEASE. `control_failures` catches a
    # control that ran and failed; it cannot catch one that never ran, and the two
    # controls below are the ones the 2026-08-29 release rests on — the shipped
    # ROUTE and the ENVELOPE. Requiring them by verdict rather than by count means a
    # future edit that drops either from the loop turns this red instead of quietly
    # releasing on the evidence that remains.
    if not arguments.no_controls and UNCERTIFIED_POLICY_INSTALLED is None:
        verdicts = [c.get("verdict") for c in controls]
        if "ENVELOPE-HOLDS" not in verdicts:
            reasons.append("the envelope leg did not run or did not hold, so "
                           "nothing measured that FUSED_RELEASE_ENVELOPE can "
                           "decline a configuration or that clause (8) still "
                           "refuses an unadmitted arm by name")
            released = False
        if "RELEASE-ADMITTED" not in verdicts:
            reasons.append("no case was shown reaching its fused arm through "
                           "RELEASED_FUSED_ARMS with MEEP_GPU_FUSE_ARMS unset, so "
                           "this run is not evidence about the SHIPPED route")
            released = False
    if arguments.no_controls:
        reasons.append("--no-controls: the armed nulls did not run, so this is a "
                       "diagnostic run and not a gate run")
        released = False
    if arguments.smoke:
        reasons.append("--smoke: lifted as the NumPy reference (prefer_gpu=False), "
                       "which consults no kernel table, so nothing about the fused "
                       "route was measured")
        released = False

    summary = {
        "gate": "dispatch_fused_route",
        "purpose": ("drive a cross-sub-step fused pair through the driver's own "
                    "consults and byte-compare the whole run against the array path"),
        "provenance": provenance,
        "verdicts": {r["case"]: r.get("verdict") for r in rows},
        "first_divergent_checkpoint": {r["case"]: r.get("first_divergent_checkpoint")
                                       for r in rows},
        "substitution": {r["case"]: (r.get("substitution") or {}).get("verdict")
                         for r in rows},
        "arms_driven": {r["case"]: (r.get("evidence") or {}).get("driven")
                        for r in rows},
        # THE ARBITRATION VERDICT PER CASE, which is what
        # ``recut_driver_dispatch_record.py --backend cuda`` refuses without. A
        # by-preference dispatch number is only as good as the leg that showed the
        # incumbent yielding, and a by-default number only as good as the leg that
        # showed it holding; neither is an assertion this gate is allowed to make.
        # EVERY ARBITRATION ANSWER PER CASE, NOT WHICHEVER RAN LAST. Three controls
        # run on each dispatching CUDA case -- default precedence, cuda preferred,
        # preference rejected -- and a case-keyed SCALAR kept only the third. That
        # is the rejected control, so the two verdicts the published number is
        # conditioned on (``INCUMBENT-YIELDED`` from the preferred control and
        # ``ARBITRATION-DEFAULT-HELD`` from the default one) never reached the
        # summary at all, and ``recut_driver_dispatch_record.py --backend cuda``
        # would have refused a campaign that measured both of them correctly.
        "arbitration": arbitration_summary(controls),
        "backend": arguments.backend,
        # NON-NULL ON THE cuda_alone LEG ONLY: how Triton was withheld, so a reader
        # of ``gate.json`` alone can tell a Triton-less composition from the others.
        "triton_withheld": TRITON_WITHHELD,
        # WHICH ROUTE ADMITTED THE ARM, per case, because it decides what the run is
        # evidence ABOUT: "release" is the shipped path (MEEP_GPU_FUSE_ARMS unset,
        # RELEASED_FUSED_ARMS admitting inside FUSED_RELEASE_ENVELOPE) and "opt-in"
        # is the switch, which is how a shape outside the envelope is reached at all.
        "reached_by": {name: drive[name].get("reached_by", "opt-in")
                       for name in (r["case"] for r in rows) if name in drive},
        "controls": {f"{c.get('case')}:{c.get('control') or c.get('leg')}":
                     c.get("verdict") for c in controls},
        "dispatched_cases": dispatched,
        "policy_refused_cases": policy_refused,
        "uncertified_policy_installed": UNCERTIFIED_POLICY_INSTALLED,
        "failures": failures,
        "control_failures": control_failures,
        "release": {"released": released, "reasons": reasons},
        "what_this_licenses": (
            "the arms listed under arms_driven, on the exact configurations listed "
            "under verdicts, with the policy recorded in provenance. It licenses "
            "NOTHING about an arm it did not drive, a shape it did not run, or the "
            "value of DISPATCH_BY_DEFAULT."),
        "rows": rows,
        "control_rows": controls,
    }
    gate_provenance.stamp(summary)
    with open(os.path.join(arguments.out, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    with open(os.path.join(arguments.out, "gate.json"), "w",
              encoding="utf-8") as handle:
        json.dump({k: v for k, v in summary.items()
                   if k not in ("rows", "control_rows")}, handle, indent=1,
                  default=str)
    say(f"SUMMARY {json.dumps(summary['verdicts'])}")
    say(f"CONTROLS {json.dumps(summary['controls'])}")
    say(f"SUBSTITUTION {json.dumps(summary['substitution'])}")
    say(f"RELEASE released={released} reasons={reasons}")
    return 0 if released else 1


if __name__ == "__main__":
    raise SystemExit(main())
