"""
Field-array storage for the MEEP-compatible FDTD engine.

``Fields`` owns the Yee-staggered field state of one simulation. It allocates
the primary displacement (D) and magnetic-induction (B) arrays that the curl
sub-steps advance, the material arrays (``eps``, ``inv_eps``) that close the
constitutive relation, and — only once PML is switched on — stored E/H plus the
twelve PML auxiliaries whose update formulas accumulate field history. Inputs
are a ``Grid`` (which supplies the array module, the stored shape, and the
mirror-symmetry flags) and a complex-field flag; outputs are raw component
arrays consumed by the step kernels, the sources, and the monitors, plus the
readback entry points (``to_cell_center`` / ``to_meep_array``), which hand the
arrays to :mod:`~.readback`: MEEP's ``get_array()`` word for word, its layout,
its sum order and its connection rules, so results can be compared against CPU
MEEP without realignment.

Dispersive materials add a second reason to store E, independent of PML. Each
``PolarizationState`` in ``polarizations`` advances its own P at the END of the
step, so ``D * inv_eps`` stops being the E of the step just finished the moment
any susceptibility is present. MEEP allocates a stored E on exactly that
condition (update_eh.cpp:166-171: ``if (s->chi1inv[ec][d_ec] || have_f_minus_p
|| dsigw != NO_DIRECTION)``), which splits the allocation in two here:
``enable_field_storage`` gives E alone, ``enable_pml_storage`` adds H and the
twelve auxiliaries on top. H stays derived from B without PML, because
``update_H`` writes nothing there and a stored H would sit at zero forever.

With polarizations live, two derived quantities are named rather than inlined:
``displacement_minus_polarization`` is MEEP's ``f_minus_p`` = D - sum P, and
``drive_field`` is MEEP's ``w`` — the field that DRIVES the polarization, which
is ``(D - sum P) * inv_eps`` and NOT the stored E. Those differ only inside an
absorber, where the stored E has already taken the PML accumulation, so getting
it wrong is exactly zero-error outside the layer and smoothly wrong inside it.

``eps``/``inv_eps`` hold eps_infinity and 1/eps_infinity — the INSTANTANEOUS
response — once susceptibilities are present. Nothing about the arrays changes
for a non-dispersive run.

An instantaneous nonlinearity (``mp.Medium(chi2=..., chi3=...)``) is a THIRD,
independent reason to store E, and the one with the quietest failure mode: with
chi2/chi3 present, E is no longer ``D * inv_eps`` at all, so a ``get_E`` that
recomputed it would hand back the LINEAR field — smooth, plausible, and missing
every harmonic the run exists to produce. ``set_nonlinear_volumes`` therefore
carries the same ``enable_field_storage`` obligation a susceptibility does. The
nonlinearity itself allocates nothing per step: it is a pointwise Pade factor on
the constitutive product (:func:`~.stepping.calc_nonlinear_u`), with no auxiliary
field, no ODE and no history, and it is unrelated to the dispersion ADE.

All quantities are in MEEP natural units (c = 1, frequency = 1/wavelength);
host applications perform any external unit conversion. Real runs store
float32, complex runs complex64;
``eps``/``inv_eps`` are always float32, so a real run halves the FIELD storage
(:meth:`Fields.field_bytes_per_cell`) but not quite the total. The array module
comes from ``grid.xp`` — this module never imports cupy.

MEEP uses the D/B formulation:
- D and B are PRIMARY fields (updated by curl)
- E = D / epsilon (update_eh in step.cpp)
- H = B / mu = B (for non-magnetic materials, mu = 1)

Field component indices (meep.hpp component enum):
- Ex=0, Ey=1, Ez=2 (D-field derived)
- Hx=3, Hy=4, Hz=5 (B-field derived)
- Dx=6, Dy=7, Dz=8 (primary)
- Bx=9, By=10, Bz=11 (primary)

Memory model: E and H are not stored persistently. Without PML, E = D*inv_eps
and H = B are served on demand by get_E()/get_H(); with PML they must be stored
because the PML constitutive update accumulates,

    f[i] += (kap+sig)*fw_new - (kap-sig)*fw_prev

where f[i] is the field from the previous step, not a fresh D/eps evaluation.
That is 8 arrays without PML versus 26 with it.

Mirror symmetry phases are DERIVED here, not tabulated: :func:`mirror_parity`
reproduces MEEP's ``symmetry::phase_shift`` for a mirror about any of the three
axes and either declared phase, so an odd mirror and a Z mirror need no new
table and cannot drift from the even X/Y one. ``SYMMETRY_PHASES`` is that
derivation evaluated for the even X/Y case and is kept only because other
modules read it.

MEEP source references: meep.hpp (fields class, f[][] arrays, component enum),
vec.cpp (mirror(), symmetry::transform, symmetry::phase_shift),
fields_chunk.cpp (allocation), structure.cpp (chi1inv).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from . import part_products

if TYPE_CHECKING:  # Import-time independence: Grid is only needed for annotations.
    from .dispersion import PolarizationState
    from .grid import Grid
    from .readback import HAliasRule


AXIS_NAMES = ("x", "y", "z")  # Field-array axis order, matching Grid.


def _slab(axis: int, index: Any) -> Tuple[Any, ...]:  # Index tuple selecting along one axis only.
    return (slice(None),) * axis + (index,)



# The twelve stored components, grouped by how they transform. E and D are true
# vectors; H and B are pseudovectors, which is the whole of the difference in
# MEEP's phase_shift.
VECTOR_COMPONENTS = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz")
PSEUDOVECTOR_COMPONENTS = ("Hx", "Hy", "Hz", "Bx", "By", "Bz")
FIELD_COMPONENTS = VECTOR_COMPONENTS + PSEUDOVECTOR_COMPONENTS


def mirror_parity(component: str, axis: int, phase: int = 1) -> int:
    """Sign a component picks up under one mirror plane — MEEP's ``symmetry::phase_shift``.

    THE ONE DERIVATION. Everything mirror-parity in this engine — the ghost cell
    the curl reads across the plane, the quadrant unfolding, the source-parity
    check — resolves through here, so an odd mirror or a Z mirror adds no second
    table to keep in step with the first.

    ``f(S x) = mirror_parity(c, axis, phase) * f(x)``, where S reflects ``axis``.

    MEEP vec.cpp. ``mirror(axis, gv)`` sets ``S[axis].flipped`` and leaves every
    other direction alone, and ``mp.Mirror(direction, phase)`` multiplies the
    result by ``phase``, which lands in ``symmetry::ph``. ``transform(d, 1)`` is
    then ``{d, flipped: d == axis, phase: ph}``, and ``phase_shift(c, 1)``
    reads::

        phase = transform(dir(c), 1).phase          # = ph
        flip  = transform(dir(c), 1).flipped        # = dir(c) == axis
        if (is_magnetic(c) || is_B(c)) {            # pseudovector correction
          FOR_DIRECTIONS(d) {
            if (transform(d, 1).flipped) flip = !flip;
            shift = (transform(d,1).d - d + 6) % 3; ...
          }
          if (have_one && have_two) flip = !flip;
        }
        return flip ? -phase : phase;

    A single mirror flips exactly ONE direction and permutes none, so the loop
    toggles ``flip`` once and the have_one/have_two handedness term never fires.
    That leaves two rules and a factor:

    * a true vector (E, D) is ODD about the plane it is normal to, even otherwise;
    * a pseudovector (H, B) is exactly the opposite — EVEN about its own axis;
    * ``phase`` multiplies both, which is what makes ``mp.Mirror(d, phase=-1)``
      the same fold with every sign inverted rather than a different one.

    The odd case is not a curiosity: a current along the mirror's own normal —
    an ``Ex`` dipole on the ``x = 0`` plane — is odd under an even mirror and can
    only be folded by declaring ``phase=-1``, which is precisely why MEEP exposes
    the argument.

    Args:
        component: One of the twelve stored component names.
        axis: 0 (X), 1 (Y), or 2 (Z) — the direction NORMAL to the plane.
        phase: The plane's declared phase, +1 (even mirror) or -1 (odd).

    Returns:
        +1 (symmetric) or -1 (antisymmetric) about that plane.

    Raises:
        ValueError: for an unknown component, an axis outside 0-2, or a phase
            that is not +/-1.
    """
    if component not in FIELD_COMPONENTS:
        raise ValueError(
            f"unknown field component {component!r}; expected one of {list(FIELD_COMPONENTS)}"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"mirror axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
    if phase not in (1, -1):
        raise ValueError(
            f"mirror phase must be +1 (even) or -1 (odd), got {phase!r}"
        )
    own_axis = AXIS_NAMES.index(component[1])  # 'Ex' -> x; the component's Yee direction.
    flipped = (own_axis == axis) != (component in PSEUDOVECTOR_COMPONENTS)
    return int(phase) * (-1 if flipped else 1)


# The even X/Y projection of `mirror_parity`, as (phase_x, phase_y) per component:
# exactly the table this module used to hardcode, now computed from the one
# derivation so the two cannot disagree.
#
# NOTHING ON THE LIVE PATH READS IT. Every consumer — the ghost-cell fill, the
# source-parity rule, the DFT/flux unfolding — resolves `mirror_parity` at the
# plane's own declared phase, which is what makes an odd plane and a Z plane work
# on CuPy exactly as they work on NumPy: the shared `grid.xp` array path is the
# GPU path. The only readers left are the tests that pin this projection against
# the derivation; the fused-kernel fill helpers that read it were deleted with the
# rest of that file's in-file fallback (fused-plan Phase 0). The retained kernel
# wrappers in `cuda_kernels/step_curl_kernels.py` still take the SAME even
# projection through `get_symmetry_phase`'s default phase — which is why
# `fastpath.plan_fast_path` keeps every folded run on the array path until the
# wrappers thread the declared phase (fused-plan Phase 2).
#
# Keep it that way: this table cannot express phase = -1 or a Z plane, so wiring
# the kernels in without generalizing them first would fold those cases with the
# wrong sign. Call `mirror_parity` for anything new.
SYMMETRY_PHASES: Dict[str, Tuple[int, int]] = {
    component: (mirror_parity(component, 0), mirror_parity(component, 1))
    for component in FIELD_COMPONENTS
}

# Yee offsets in units of dx/2 (meep/vec.hpp iyee_shift).
# 1 = already at a half-integer (cell-center) position along that axis, no
# interpolation needed; 0 = at an integer position, interpolate to reach the
# cell center. Every component has at least one zero, so cell-centering always
# produces a fresh array rather than aliasing stored state.
IYEE_SHIFTS: Dict[str, Tuple[int, int, int]] = {
    'Ex': (1, 0, 0), 'Ey': (0, 1, 0), 'Ez': (0, 0, 1),
    'Dx': (1, 0, 0), 'Dy': (0, 1, 0), 'Dz': (0, 0, 1),
    'Hx': (0, 1, 1), 'Hy': (1, 0, 1), 'Hz': (1, 1, 0),
    'Bx': (0, 1, 1), 'By': (1, 0, 1), 'Bz': (1, 1, 0),
}

# The components a conductivity can be installed on, in MEEP's FOR_D_AND_B order
# (structure.cpp:377-379). D and B, never E or H: MEEP's set_conductivity redirects an
# electric or magnetic component onto its D/B partner and multiplies by chi1inv, and
# the absorber path never takes that branch.
CONDUCTIVE_COMPONENTS: Tuple[str, ...] = ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz')

# Components that carry a BFAST IIR state. MEEP allocates ``f_bfast[cc][cmp]`` inside
# ``step_db``, which runs over B_stuff and D_stuff only (step_db.cpp:76-79), so E and H
# never have one — the pass is on the curl sub-step, not the constitutive one.
BFAST_COMPONENTS: Tuple[str, ...] = ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz')


def require_even_xy_planes_only(grid) -> None:  # Refuse a plane SYMMETRY_PHASES cannot express.
    """Refuse any mirror plane the even-X/Y projection above cannot represent.

    The rule that belongs with the table, so a consumer of ``SYMMETRY_PHASES`` cannot
    quietly serve a plane it has no entry for. ``SYMMETRY_PHASES`` is
    :func:`mirror_parity` frozen at ``phase = +1`` on X and Y: read through it, a Z
    plane is invisible and an odd plane comes out with every sign inverted, and in
    both cases the run completes and returns a full, plausible field.

    No production code calls this since the fused-kernel fallback that did was
    deleted (fused-plan Phase 0); it stands as the CONTRACT for the symmetry slices —
    the retained kernel wrappers still bake the same even projection through
    ``get_symmetry_phase``'s default, so ``fastpath.plan_fast_path`` must keep folded
    runs on the array path until they thread the declared phase. The live CuPy path
    is the shared ``grid.xp`` array path, which resolves each plane's own phase
    through :func:`mirror_parity` and is general on all three axes. Generalize the
    kernels the same way when wiring them in (fused-plan Phase 2), and delete this.

    Raises:
        NotImplementedError: naming every plane the table cannot carry.
    """
    unsupported = []
    for axis, name in enumerate(("X", "Y", "Z")):
        reader = getattr(grid, "mirror_phase", None)
        phase = reader(axis) if callable(reader) else (
            1 if getattr(grid, ("sym_x", "sym_y", "sym_z")[axis], False) else None
        )
        if phase is None:
            continue
        if axis == 2 or phase != 1:
            unsupported.append(f"Mirror({name!r}, phase={phase:+d})")
    if unsupported:
        raise NotImplementedError(
            f"This path carries mirror symmetry through fields.SYMMETRY_PHASES, the even X/Y "
            f"projection of mirror_parity, which cannot express {', '.join(unsupported)}: a Z "
            f"plane is ignored and an odd plane is folded with every sign inverted, both "
            f"silently. Use the shared grid.xp array path, which resolves the parity per plane, "
            f"or generalize this one to mirror_parity(component, axis, phase) first."
        )


def get_symmetry_phase(component: str, direction: str, phase: int = 1) -> int:  # Named-axis mirror parity.
    """:func:`mirror_parity` addressed by axis NAME rather than index.

    The spelling the monitors, the CUDA kernels and the tests use. ``direction``
    is 'x', 'y' or 'z'; ``phase`` is the plane's declared phase and defaults to
    MEEP's own +1, so every existing call site keeps the even-mirror parity it
    asked for.

    Args:
        component: Field component name (Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz, Bx, By, Bz)
        direction: 'x', 'y', or 'z' — the direction normal to the mirror plane.
        phase: +1 for an even mirror, -1 for an odd one.

    Returns:
        +1 (even/symmetric) or -1 (odd/antisymmetric)

    Raises:
        ValueError: If the component, the direction, or the phase is not recognized.
    """
    if direction not in AXIS_NAMES:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
    return mirror_parity(component, AXIS_NAMES.index(direction), phase)


class StepScratch:
    """Reusable full-volume buffers for the step loop's temporaries.

    THE MEASUREMENT THIS EXISTS FOR. On this machine a float32 elementwise pass over
    a 640x640 array costs 132.7 us when it allocates its result and 52.7 us when it
    writes into an existing one — 60% of the cost is the allocation, because above
    roughly a megabyte the allocator hands back freshly mmap'd pages and the
    first-touch page faults dominate the arithmetic. The array step path materializes
    on the order of a hundred such temporaries per timestep, so its throughput PEAKS
    near 100k cells and DECLINES above it (measured 56 -> 32 Mcell-steps/s from 100k
    to 3.7M cells) while MEEP's rises monotonically. Reusing the buffers removed a
    measured 29% of a 2-D PML step and 22% of a 3-D one, bit-identically.

    The same buffers matter more on a device than on the host: there each temporary is
    a memory-pool allocation, an extra kernel, and an extra full read-and-write of
    device memory.

    THE INVARIANT CALLERS MUST HOLD. A buffer is owned by its ``tag`` for as long as
    the caller holds it, and two live temporaries must never share one. Tags are
    therefore assigned per ROLE, and a tag is reused across two roles only where their
    lifetimes provably cannot overlap — ``previous`` serves both the split-field PML
    recurrence (inside the curl sub-step) and the constitutive accumulation (inside
    the E/H sub-step), which never run at the same time. Everything a monitor, a
    source or the driver may hold past the step is allocated normally; this pool is
    for values consumed within the call that produced them.

    Buffers are UNINITIALIZED on the way out. Every user overwrites the whole volume
    before reading it, and a zeroing pass would give back a third of what the pool
    saves.
    """

    def __init__(self, xp: Any):
        self.xp = xp
        self._slots: Dict[Tuple[Any, ...], Any] = {}
        self._constants: Dict[Tuple[Any, ...], Any] = {}

    def take(self, tag: str, shape: Tuple[int, ...], dtype: Any) -> Any:  # Uninitialized buffer.
        """Return this tag's buffer at ``shape``/``dtype``, allocating it once.

        ``shape`` must be a tuple of ints and ``dtype`` a dtype object — what
        ``array.shape`` and ``array.dtype`` already are. The lookup runs ~30 times a
        timestep and normalizing the key here cost more than it protected: at 25.6k
        cells a step is 0.68 ms, so a few microseconds of tuple rebuilding per call is
        a measurable tax on exactly the small grids that have no allocation to save.
        """
        key = (tag, shape, dtype)
        slot = self._slots.get(key)
        if slot is None:
            slot = self.xp.empty(shape, dtype=dtype)
            self._slots[key] = slot
        return slot

    def like(self, tag: str, array: Any) -> Any:  # Uninitialized buffer shaped like `array`.
        return self.take(tag, array.shape, array.dtype)

    def copy_of(self, tag: str, array: Any) -> Any:  # This tag's buffer, holding `array`'s values.
        buffer = self.take(tag, array.shape, array.dtype)
        self.xp.copyto(buffer, array)
        return buffer

    def constant(self, key: Tuple[Any, ...], build: Any) -> Any:  # Memoized run-invariant array.
        """Return a RUN-INVARIANT derived array, building it at most once.

        Separate from :meth:`take` because the VALUE is reused, not merely the
        storage: a radial weight vector, a divisor, an index table. ``key`` must
        therefore name every input the value depends on — a key that forgets one
        serves a stale array that is smooth, plausible and wrong, which is the
        failure mode this whole engine is built to avoid.

        The values here are axis-length vectors and index tables, not volumes, so
        the cache is bounded by the shapes one run touches.
        """
        try:
            return self._constants[key]
        except KeyError:
            value = build()
            self._constants[key] = value
            return value

    def clear(self) -> None:  # Release every buffer (the driver's close path).
        self._slots.clear()
        self._constants.clear()

    def bytes_held(self) -> int:  # Total pooled bytes, for the driver's memory report.
        return sum(int(slot.nbytes) for slot in self._slots.values())


@dataclass
class Fields:
    """
    MEEP-compatible field storage with on-demand E/H.

    Translation of meep::fields from meep.hpp. MEEP stores fields as
    f[component][chunk][index]; this keeps one dense 3-D array per component.

    Primary fields (updated by curl):
        Dx, Dy, Dz: Electric displacement
        Bx, By, Bz: Magnetic induction

    Derived fields (computed on demand while PML is inactive):
        Ex, Ey, Ez: E = D / epsilon (via get_E())
        Hx, Hy, Hz: H = B / mu (via get_H())

    Once PML is active, E and H are stored because the PML constitutive update
    accumulates field history.

    Material:
        eps: Permittivity (epsilon_r)
        inv_eps: 1/epsilon, for E = D * inv_eps without a divide

    force_complex_fields mirrors MEEP's Simulation flag: when True the field
    arrays are complex64, which is required for a nonzero Bloch ``k_point`` and for
    sources carrying a spatially varying complex amplitude (a converging beam's
    spherical phase, say).

    False — MEEP's own default — stores float32 and halves every array counted by
    :meth:`field_bytes_per_cell`. It is not a reduced-accuracy mode: Maxwell's
    equations, the constitutive relations and the PML recurrence all carry real
    coefficients here, so a real run reproduces the REAL PART of the complex one to
    the last bit (measured 0.0 across plain, PML, dispersive and mirror-folded runs,
    which is also what CPU MEEP's own two modes give each other). What is genuinely
    lost is the imaginary plane: a Bloch phase has nowhere to live, and a complex
    source amplitude would be silently re-read as a time shift, so both are refused
    rather than dropped. Monitors are unaffected — a DFT accumulator stays complex64
    over real fields, exactly as MEEP's does (dft.cpp:294-305).
    """
    grid: "Grid"
    force_complex_fields: bool = False

    # The components MEEP allocated no array for (:meth:`set_absent_components`). A
    # class attribute, not a dataclass field: the empty default is every Fields not
    # built by a lift, and only a lift that found an absent component sets its own.
    absent_components = ()

    # Primary D-field (meep.hpp: f[Dx], f[Dy], f[Dz])
    Dx: Any = field(default=None, repr=False)
    Dy: Any = field(default=None, repr=False)
    Dz: Any = field(default=None, repr=False)

    # Primary B-field (meep.hpp: f[Bx], f[By], f[Bz])
    Bx: Any = field(default=None, repr=False)
    By: Any = field(default=None, repr=False)
    Bz: Any = field(default=None, repr=False)

    # Derived E-field (E = D / eps) — allocated only when PML is active.
    # While _pml_active is False these stay None and get_E() computes on the fly.
    Ex: Any = field(default=None, repr=False)
    Ey: Any = field(default=None, repr=False)
    Ez: Any = field(default=None, repr=False)

    # Derived H-field (H = B / mu = B for non-magnetic) — allocated only when
    # PML is active; otherwise get_H() returns the B array itself.
    Hx: Any = field(default=None, repr=False)
    Hy: Any = field(default=None, repr=False)
    Hz: Any = field(default=None, repr=False)

    # PML auxiliary fields f_u (meep.hpp: f_u[cc][cmp] for split-field PML).
    # MEEP step_generic.cpp: fu holds the intermediate result for the dsigu PML
    # direction, needed when two PML directions are active (3D) in step_curl.
    fu_Bx: Any = field(default=None, repr=False)
    fu_By: Any = field(default=None, repr=False)
    fu_Bz: Any = field(default=None, repr=False)
    fu_Dx: Any = field(default=None, repr=False)
    fu_Dy: Any = field(default=None, repr=False)
    fu_Dz: Any = field(default=None, repr=False)

    # Conductivity + PML auxiliary fields f_cond (meep.hpp: f_cond[cc][cmp]).
    # They integrate the conductive curl before the two split-field PML stages.
    # Allocated only when both mechanisms are active, and only for the components
    # that actually carry a conductivity — MEEP's f_cond is indexed by component.
    f_cond_Dx: Any = field(default=None, repr=False)
    f_cond_Dy: Any = field(default=None, repr=False)
    f_cond_Dz: Any = field(default=None, repr=False)
    f_cond_Bx: Any = field(default=None, repr=False)
    f_cond_By: Any = field(default=None, repr=False)
    f_cond_Bz: Any = field(default=None, repr=False)

    # BFAST auxiliary fields f_bfast (meep.hpp:1469, ``f_bfast[cc][cmp]``). One per
    # D and B component, holding the state of the marginally stable IIR
    # ``F_n = S_n - F_{n-1}`` whose output ``F_n - F_{n-1}`` is the Tustin (bilinear)
    # time derivative of the interpolated cross product ``k x E`` / ``k x H``.
    # Allocated whenever ``grid.bfast_active``, INDEPENDENT of PML and of
    # conductivity — MEEP's own allocation test is ``use_bfast && !f_bfast[cc][cmp]``
    # (step_db.cpp:76-79), which names neither — so a BFAST run without a PML keeps
    # its term instead of silently losing it.
    f_bfast_Dx: Any = field(default=None, repr=False)
    f_bfast_Dy: Any = field(default=None, repr=False)
    f_bfast_Dz: Any = field(default=None, repr=False)
    f_bfast_Bx: Any = field(default=None, repr=False)
    f_bfast_By: Any = field(default=None, repr=False)
    f_bfast_Bz: Any = field(default=None, repr=False)

    # PML auxiliary fields f_w (meep.hpp: f_w[cc][cmp]).
    # MEEP step_generic.cpp step_update_EDHB: fw holds the constitutive result
    # for the dsigw PML direction (the component's OWN direction, MEEP's d_c),
    # applied by update_eh as the third PML direction's absorption.
    f_w_Ex: Any = field(default=None, repr=False)
    f_w_Ey: Any = field(default=None, repr=False)
    f_w_Ez: Any = field(default=None, repr=False)
    f_w_Hx: Any = field(default=None, repr=False)
    f_w_Hy: Any = field(default=None, repr=False)
    f_w_Hz: Any = field(default=None, repr=False)

    # Material (meep.hpp: s->chi1inv for inverse permittivity)
    eps: Any = field(default=None, repr=False)
    inv_eps: Any = field(default=None, repr=False)
    _eps_components: Dict[str, Any] = field(default_factory=dict, repr=False)
    _inv_eps_components: Dict[str, Any] = field(default_factory=dict, repr=False)

    # MEEP's conductivity volumes (structure.cpp: s->conductivity[c][d]), keyed by the
    # D or B component they belong to and point-sampled at that component's OWN Yee
    # position, never subpixel-averaged (structure.cpp:868-895, where `multby` is 0 for
    # a D/B component so the raw sigma is stored). A component absent from these dicts
    # is lossless and takes the untouched arithmetic path, which is what keeps a
    # conductivity-free run's step_D and step_B bit-identical.
    #
    # ONE SHARED VOLUME IS NOT ENOUGH for a graded sigma. MEEP samples Dx, Dy and Dz at
    # three different points; emulating that with a single array by snapping the sample
    # position costs 1.51e-02 (node-snapped) to 1.07e-01 (centre-snapped) relative Linf
    # on a 2-D mp.Absorber case, against a parity band of 3e-07..6e-06. A UNIFORM sigma
    # is the special case where registration cannot matter, and `set_d_conductivity`
    # still installs one array object under all three D keys for it.
    #
    # `condfac` = 1 - sigma*dt/2, MEEP's float32 `(1 - dt2 * cnd[i])` (step_generic.cpp:88),
    # and `condinv` = 1/(1 + sigma*dt/2), formed in double and rounded once
    # (structure.cpp:699); the SAME dt and the same expression on both sides —
    # step_generic.cpp's step_curl is one function for D and B alike, with no half-step
    # offset between them.
    _conductivity: Dict[str, Any] = field(default_factory=dict, repr=False)
    _condfac: Dict[str, Any] = field(default_factory=dict, repr=False)
    _condinv: Dict[str, Any] = field(default_factory=dict, repr=False)

    # MEEP's INSTANTANEOUS nonlinear susceptibilities (structure.cpp: s->chi2[c],
    # s->chi3[c]), keyed by E component. Nothing here is an auxiliary field: chi2/chi3
    # carry no history and no ODE, they only rescale the constitutive product where E
    # is recovered from D. A component absent from these dicts is linear and takes the
    # untouched arithmetic path, which is what makes zero nonlinearity byte-identical.
    # MEEP keeps the pair together — set_chi3 allocates a zero chi2 and vice versa
    # (structure.cpp:815-826 and 851-862) — so both dicts always carry the same keys.
    _chi2_components: Dict[str, Any] = field(default_factory=dict, repr=False)
    _chi3_components: Dict[str, Any] = field(default_factory=dict, repr=False)

    # One PolarizationState per susceptibility (MEEP: fields_chunk::pol[E_stuff]).
    polarizations: List["PolarizationState"] = field(default_factory=list, repr=False)

    # True once PML storage is allocated and E/H are stored rather than derived.
    _pml_active: bool = field(default=False, repr=False)

    # True once the run's first PML ``update_H`` has started H and ``f_w_H`` as copies
    # of B on the cells whose chunk carries H's own layer, as MEEP's lazy allocation
    # does (``stepping.separate_h_from_b``; update_eh.cpp:165-177). Until then MEEP's
    # H is B's storage. A state seeded mid-run carries True; ``FdtdDriver.reset`` clears
    # it with the fields (``Fields.reset`` zeroes arrays only).
    h_separated: bool = field(default=False, repr=False)

    # True once E is stored rather than recomputed from D on demand. Implied by PML,
    # and independently by any susceptibility (MEEP update_eh.cpp:166-171).
    _stored_E: bool = field(default=False, repr=False)

    # Off-diagonal chi1inv rows: {row component: {partner component: float32 volume}}.
    # MEEP's s->chi1inv[ec][d] for d != component_direction(ec); empty when the
    # tensor is diagonal, which keeps every existing path byte-identical.
    _chi1inv_offdiagonal: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    # Shared buffer for D - sum P; one per grid, reused across the three components
    # because each is consumed before the next is formed.
    _fmp_scratch: Any = field(default=None, repr=False)

    # Per-component buffers for the same quantity, allocated only by a nonlinear run.
    # The nonlinearity's Yee average needs all three D - sum P volumes ALIVE AT ONCE
    # (MEEP's ``dmp[dc][cmp]``, ``dmp[dc_1][cmp]``, ``dmp[dc_2][cmp]`` in
    # update_eh.cpp:143-146), which the single shared buffer above cannot serve.
    _fmp_scratch_by_component: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Reusable buffers for the step loop's full-volume temporaries — the curl
    # accumulator, the two shifted stencil operands, and the PML recurrences'
    # previous-value copies. See :class:`StepScratch` for the measurement and for
    # the tag-lifetime rule callers have to hold.
    scratch: Any = field(default=None, repr=False)

    def __post_init__(self):  # Zero-initialize field storage (MEEP: fields::initialize_field).
        if self.scratch is None:
            self.scratch = StepScratch(self.grid.xp)
        self._allocate_fields()
        self._init_material()

    def _field_dtype(self):  # Storage dtype for field arrays: complex64 or float32.
        xp = self.grid.xp
        return xp.complex64 if self.force_complex_fields else xp.float32

    def _allocate_fields(self):
        """
        Allocate field tensors initialized to zero.

        MEEP reference: fields_chunk::alloc_f() in fields_chunk.cpp

        E, H, and the PML auxiliaries are deliberately not allocated here: they
        are allocated lazily by enable_pml_storage() when PML is switched on, or
        computed on the fly by get_E()/get_H() when it is not.
        """
        xp = self.grid.xp
        shape = self.grid.shape
        dtype = self._field_dtype()

        # Primary fields (D, B) — always allocated
        self.Dx = xp.zeros(shape, dtype=dtype)
        self.Dy = xp.zeros(shape, dtype=dtype)
        self.Dz = xp.zeros(shape, dtype=dtype)

        self.Bx = xp.zeros(shape, dtype=dtype)
        self.By = xp.zeros(shape, dtype=dtype)
        self.Bz = xp.zeros(shape, dtype=dtype)

        # Derived fields (E, H) — None until enable_pml_storage()
        self.Ex = None
        self.Ey = None
        self.Ez = None
        self.Hx = None
        self.Hy = None
        self.Hz = None

        # PML auxiliary fields f_u — allocated lazily by enable_pml_storage()
        self.fu_Bx = None
        self.fu_By = None
        self.fu_Bz = None
        self.fu_Dx = None
        self.fu_Dy = None
        self.fu_Dz = None

        # Conductivity + PML auxiliary f_cond — allocated only by the combination.
        self.f_cond_Dx = None
        self.f_cond_Dy = None
        self.f_cond_Dz = None
        self.f_cond_Bx = None
        self.f_cond_By = None
        self.f_cond_Bz = None

        # PML auxiliary fields f_w — allocated lazily by enable_pml_storage()
        self.f_w_Ex = None
        self.f_w_Ey = None
        self.f_w_Ez = None
        self.f_w_Hx = None
        self.f_w_Hy = None
        self.f_w_Hz = None

        self._ensure_bfast_storage()

    def _ensure_bfast_storage(self) -> None:
        """Allocate the six ``f_bfast`` IIR states when BFAST is on, and only then.

        Unconditional on PML and on conductivity, because MEEP's test is
        ``use_bfast && !f_bfast[cc][cmp]`` alone (step_db.cpp:76-79) and it sits
        BESIDE, not inside, the ``f_cond`` and ``f_u`` allocations. Complex storage
        gets ONE complex array per component rather than two real ones: the
        recursion ``F_n = S_n - F_{n-1}`` has real coefficients, so the real and
        imaginary parts advance independently and MEEP's two ``cmp`` rows are the
        two halves of the same complex array.
        """
        if not self.grid.bfast_active:
            return
        xp = self.grid.xp
        shape = self.grid.shape
        dtype = self._field_dtype()
        for component in BFAST_COMPONENTS:
            name = "f_bfast_" + component
            if getattr(self, name, None) is None:
                setattr(self, name, xp.zeros(shape, dtype=dtype))

    def enable_field_storage(self):
        """
        Allocate the stored E arrays, without switching on PML mode.

        MEEP update_eh.cpp:166-171 allocates ``f[ec]`` when the component has a
        chi1inv, when ``f_minus_p`` exists (a polarization or an integrated source),
        or when the component's own direction absorbs. This engine skips the first
        of those deliberately — a non-dispersive E is served on demand as
        ``D * inv_eps`` and costs no array — so the trigger here is a susceptibility,
        and :meth:`enable_pml_storage` layers the absorber's needs on top.

        H is deliberately NOT allocated: ``update_H`` writes nothing without PML, so
        a stored H would sit at zero for the whole run while ``get_H`` returned it.

        Idempotent, and a one-way switch: it changes get_E() from compute-on-demand
        to return-the-stored-array.
        """
        if self.Ex is None:
            xp = self.grid.xp
            dtype = self._field_dtype()
            self.Ex = xp.zeros(self.grid.shape, dtype=dtype)
            self.Ey = xp.zeros(self.grid.shape, dtype=dtype)
            self.Ez = xp.zeros(self.grid.shape, dtype=dtype)
        self._stored_E = True

    def enable_pml_storage(self):
        """
        Allocate E, H, and the PML auxiliary fields for PML mode.

        Called by the step_curl entry points when PML is active, so that the
        accumulation formula required by the PML constitutive update has field
        history to accumulate into. Idempotent: repeated calls keep the existing
        arrays and their contents.

        This is a one-way mode switch — it also changes get_E()/get_H() from
        compute-on-demand to return-the-stored-array.
        """
        if self._pml_active:
            self._ensure_conductive_pml_storage()
            return  # Already allocated; do not discard accumulated state.

        xp = self.grid.xp
        shape = self.grid.shape
        dtype = self._field_dtype()

        # E may already exist because a susceptibility asked for it; re-allocating
        # would discard a live polarization's drive history.
        self.enable_field_storage()
        self.Hx = xp.zeros(shape, dtype=dtype)
        self.Hy = xp.zeros(shape, dtype=dtype)
        self.Hz = xp.zeros(shape, dtype=dtype)

        # PML auxiliary fields f_u (for curl PML)
        self.fu_Bx = xp.zeros(shape, dtype=dtype)
        self.fu_By = xp.zeros(shape, dtype=dtype)
        self.fu_Bz = xp.zeros(shape, dtype=dtype)
        self.fu_Dx = xp.zeros(shape, dtype=dtype)
        self.fu_Dy = xp.zeros(shape, dtype=dtype)
        self.fu_Dz = xp.zeros(shape, dtype=dtype)

        # PML auxiliary fields f_w (for constitutive PML)
        self.f_w_Ex = xp.zeros(shape, dtype=dtype)
        self.f_w_Ey = xp.zeros(shape, dtype=dtype)
        self.f_w_Ez = xp.zeros(shape, dtype=dtype)
        self.f_w_Hx = xp.zeros(shape, dtype=dtype)
        self.f_w_Hy = xp.zeros(shape, dtype=dtype)
        self.f_w_Hz = xp.zeros(shape, dtype=dtype)

        self._pml_active = True
        self._ensure_conductive_pml_storage()

    def _ensure_conductive_pml_storage(self) -> None:
        """Allocate an ``f_cond`` history per conductive component, for conductivity + PML.

        Per component, not per side: MEEP's ``f_cond[c][cmp]`` is indexed by component
        and exists exactly where ``s->conductivity[c][d]`` does, so a run whose only
        loss is a ``D_conductivity`` allocates three and an ``mp.Absorber`` — electric
        AND magnetic (python/simulation.py:316) — allocates six.
        """
        if not self._pml_active:
            return
        xp = self.grid.xp
        shape = self.grid.shape
        dtype = self._field_dtype()
        for component in CONDUCTIVE_COMPONENTS:
            name = "f_cond_" + component
            if component in self._conductivity:
                if getattr(self, name) is None:
                    setattr(self, name, xp.zeros(shape, dtype=dtype))
            else:
                setattr(self, name, None)

    def set_d_conductivity(self, conductivity: Any):
        """Install MEEP's D conductivity and derive its two step coefficients.

        ``conductivity`` is sigma_D in MEEP units — the quantity
        ``mp.Medium(D_conductivity=...)`` takes, which enters Maxwell's equations as
        ``sigma_D * D`` rather than the textbook ``sigma * E`` and so differs from the
        usual electric conductivity by a factor of epsilon
        (doc/docs/Materials.md). Passing ``None`` removes it.

        Accepts either ONE volume — installed under all three D components, which is
        MEEP's answer whenever sigma is uniform and the only place registration cannot
        matter — or a mapping from component name (``'Dx'``, ``'Dy'``, ``'Dz'``) to its
        own volume, which is what a GRADED sigma such as ``mp.Absorber``'s needs
        because MEEP point-samples each component at its own Yee position.

        The two coefficients are structure.cpp:693-706 and step_generic.cpp:97::

            condfac = 1 - sigma_D*dt/2      condinv = 1/(1 + sigma_D*dt/2)
            D <- (condfac*D - dtdx*curl) * condinv

        Both are stored float32 like every other material array, each formed as MEEP
        forms it: ``condfac`` in float32 from ``dt2 = float32(dt) * 0.5``
        (step_generic.cpp:88), ``condinv`` in double from the float32 sigma word and
        the double dt, rounded once (structure.cpp:699). A conductivity that is
        identically zero is stored as ``None`` rather than as a volume of zeros, so
        the plain ``step_D`` path stays bit-identical rather than merely equal to
        within a multiply by 1.0.

        With a PML, an ``f_cond`` history is allocated per conductive component. It
        implements step_generic.cpp's conductive stage ahead of the two split-field
        recurrences; neither the conductivity-only nor PML-only allocation changes.
        """
        self._set_conductivity_side(("Dx", "Dy", "Dz"), conductivity, "D")

    def set_b_conductivity(self, conductivity: Any):
        """Install MEEP's B conductivity — the magnetic half of an ``mp.Absorber``.

        The exact mirror of :meth:`set_d_conductivity`, and deliberately a separate
        entry point rather than a flag: the two are independent in MEEP
        (``FOR_D_AND_B(c) { if (mat.has_conductivity(c)) ... }``, structure.cpp:377-379)
        and a ``mp.Medium(B_conductivity=...)`` sets only this one.

        Dropping this half of an absorber is not a rounding error. Measured against
        pristine MEEP 1.33.0 on the reconstruction that is otherwise bit-identical:
        electric-only lands 4.08e-03 relative Linf in 1-D and 6.46e-01 in 2-D.
        """
        self._set_conductivity_side(("Bx", "By", "Bz"), conductivity, "B")

    def _set_conductivity_side(self, components: Tuple[str, ...], conductivity: Any,
                               label: str) -> None:
        """Install (or clear) one side's three conductivity volumes and their coefficients."""
        xp = self.grid.xp
        for component in components:
            self._conductivity.pop(component, None)
            self._condfac.pop(component, None)
            self._condinv.pop(component, None)
        if conductivity is not None:
            if isinstance(conductivity, dict):
                supplied = dict(conductivity)
                unknown = set(supplied) - set(components)
                if unknown:
                    raise ValueError(
                        f"{label} conductivity mapping carries {sorted(unknown)}, which are not "
                        f"{label}-side components; the {label} setter installs {list(components)} "
                        f"only, and the other side has its own entry point."
                    )
            else:
                supplied = {component: conductivity for component in components}
            # step_generic.cpp:88 forms `realnum dt2 = dt * 0.5` from the realnum dt the
            # step receives; halving is exact, so this is float32(dt) / 2.
            dt2 = xp.float32(xp.float32(self.grid.dt) * xp.float32(0.5))
            dt = float(self.grid.dt)
            for component, value in supplied.items():
                if value is None:
                    continue
                values = xp.ascontiguousarray(xp.asarray(value, dtype=xp.float32))
                if tuple(values.shape) != tuple(self.grid.shape):
                    raise ValueError(
                        f"{label} conductivity shape {tuple(values.shape)} for {component} does "
                        f"not match the grid shape {tuple(self.grid.shape)}."
                    )
                self._conductivity[component] = values
                # condfac is MEEP's per-cell float32 `(1 - dt2 * cnd[i])` (step_generic.cpp:88-90).
                self._condfac[component] = (xp.float32(1.0) - values * dt2).astype(xp.float32)
                # condinv is formed in double and rounded once (structure.cpp:699):
                # `1 / (1 + conductivity[i] * dt * 0.5)`, the float word promoted and dt
                # the structure's double, left to right. The float32 spelling rounds
                # twice and differs at sigma = 0.5, 0.3 and 0.25 with dt = 0.05.
                wide = values.astype(xp.float64)
                self._condinv[component] = (1.0 / (1.0 + (wide * dt) * 0.5)).astype(xp.float32)
        for component in components:
            if component not in self._conductivity:
                setattr(self, "f_cond_" + component, None)
        self._ensure_conductive_pml_storage()

    def conductivity_for(self, component: str) -> Any:  # MEEP's s->conductivity[c][d]; None if lossless.
        """The stored sigma volume for one D or B component, or None."""
        return self._conductivity.get(component)

    def condfac_for(self, component: str) -> Any:  # 1 - sigma*dt/2; None if the component is lossless.
        return self._condfac.get(component)

    def condinv_for(self, component: str) -> Any:  # 1/(1 + sigma*dt/2); None if the component is lossless.
        return self._condinv.get(component)

    def set_nonlinear_volumes(
        self,
        chi2_by_component: Dict[str, Any],
        chi3_by_component: Dict[str, Any],
    ) -> None:
        """Install MEEP's instantaneous chi2/chi3 per E component.

        This is ``structure_chunk::set_chi2`` / ``set_chi3`` (structure.cpp:794-864)
        with MEEP's two rules kept intact:

        * **Both or neither, per component.** MEEP's comment is explicit — "currently,
          our update_e_from_d routine requires that chi2 be present if chi3 is, and
          vice versa" (structure.cpp:815-816) — because ``step_update_EDHB`` branches
          on ``chi3`` alone and then dereferences ``chi2[i]``. Whichever of the two the
          caller supplied, the partner is stored as an explicit zero.
        * **Identically zero is no nonlinearity at all.** MEEP deletes both arrays when
          the pair is trivial (structure.cpp:822-826), which is the ONLY reason a
          zero-chi3 run reproduces the linear engine bit for bit rather than to within
          a multiply by ``1.0/1.0``. A component whose chi2 and chi3 are both zero
          everywhere is dropped here for the same reason.

        Each value is a float (uniform, allocation-free) or a float32 volume in this
        grid's shape. Both mappings must be keyed by ``Ex``/``Ey``/``Ez``; a missing
        key is 0.0. Validation of the caller's spelling belongs upstream — this is the
        storage-level installer, and it checks only what it stores.
        """
        expected = ("Ex", "Ey", "Ez")
        for label, mapping in (("chi2", chi2_by_component), ("chi3", chi3_by_component)):
            unknown = sorted(set(mapping) - set(expected))
            if unknown:
                raise ValueError(
                    f"{label} must be keyed by the electric components {expected}, got unexpected "
                    f"{unknown}. Magnetic (H-side) nonlinearity is not implemented on this engine."
                )
        chi2: Dict[str, Any] = {}
        chi3: Dict[str, Any] = {}
        for component in expected:
            second = self._coerce_nonlinear_value(chi2_by_component.get(component, 0.0),
                                                  f"chi2[{component!r}]")
            third = self._coerce_nonlinear_value(chi3_by_component.get(component, 0.0),
                                                 f"chi3[{component!r}]")
            if self.is_trivial_chi(second) and self.is_trivial_chi(third):
                continue  # MEEP deletes the trivial pair; the linear path stays bit-identical.
            self._require_chi2_respects_the_mirror_planes(component, second)
            chi2[component] = second
            chi3[component] = third
        self._chi2_components = chi2
        self._chi3_components = chi3

    def _require_chi2_respects_the_mirror_planes(self, component: str, chi2: Any) -> None:
        """Refuse a chi2 on a component the fold forces to be ODD about a mirror plane.

        A chi2 medium IS NOT CENTROSYMMETRIC — that is the textbook reason chi2
        vanishes in a centrosymmetric crystal, and it is a statement about this
        engine's mirror planes too. The nonlinear factor carries
        ``c2 = D_c * chi2 * chi1inv^2``, which is LINEAR in D and therefore has D's
        parity, while ``c3`` carries ``Dsqr`` and is even whatever D does. So the
        recovered ``E = chi1inv * u * D`` keeps D's parity under chi3 and loses it
        under chi2 the moment the fold declares D_c odd: the folded half is then
        reconstructed from a symmetry the material does not have.

        Measured on the fold-equivalence case: with chi3 alone a folded run
        reproduces the full-domain run BIT FOR BIT on every axis and both phases;
        adding a chi2 to the component the plane makes odd breaks it by 4.1e-02 in
        one step, and nothing about the resulting field looks wrong.

        The rule is the parity itself and not the axis, so it covers an odd mirror
        correctly: an odd plane makes its OWN component even (chi2 fine there) and the
        two transverse ones odd (chi2 refused). MEEP performs no such check and folds
        anyway.
        """
        if self.is_trivial_chi(chi2):
            return
        for axis in range(3):
            phase = self.grid.mirror_phase(axis)
            if phase is None or mirror_parity(component, axis, phase) == 1:
                continue
            plane = "even" if phase == 1 else "odd"
            raise ValueError(
                f"chi2 on {component} is not compatible with the {plane} mirror plane on the "
                f"{AXIS_NAMES[axis]} axis: that plane forces {component} to be ODD about it, while "
                f"a chi2 term contributes D^2, which is even. A chi2 medium is not "
                f"centrosymmetric, so this mirror is not a symmetry of the material and the folded "
                f"half would be reconstructed from a symmetry the run does not have (measured: a "
                f"folded chi3 run reproduces the full-domain run bit for bit, the same run with "
                f"chi2 breaks by 4.1e-02 in one step). Run the full domain (symmetry=()), or keep "
                f"chi2 on the components this plane leaves even. chi3 is unaffected — it is even "
                f"under a mirror, which is why chi3 exists in centrosymmetric media and chi2 "
                f"does not."
            )

    def _coerce_nonlinear_value(self, value: Any, label: str) -> Any:
        """Validate one chi2/chi3 entry: a finite float, or a finite float32 volume.

        Sign is deliberately unconstrained. A negative chi3 is self-DEfocusing and a
        negative chi2 is a reversed Pockels coefficient; both are ordinary materials,
        unlike a negative sigma or conductivity, which are gain and are refused.
        """
        xp = self.grid.xp
        shape = getattr(value, "shape", None)
        if shape is None or shape == ():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite, got {value!r}.")
            return number
        values = xp.ascontiguousarray(xp.asarray(value, dtype=xp.float32))
        if tuple(values.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"{label} shape {tuple(values.shape)} does not match the grid shape "
                f"{tuple(self.grid.shape)}."
            )
        if not bool(xp.all(xp.isfinite(values))):
            raise ValueError(f"{label} must be finite in every cell.")
        return values

    @staticmethod
    def is_trivial_chi(value: Any) -> bool:  # MEEP's `trivial` flag (structure.cpp:810-813).
        """Is this chi2/chi3 entry identically zero, scalar or volume?

        The single definition of MEEP's per-component ``trivial`` test, shared by the
        installer here and by the driver's before-the-first-step guard, so the two
        cannot disagree about which calls install a nonlinearity and which are no-ops.
        """
        if getattr(value, "shape", None):
            return not bool(value.any())
        return float(value) == 0.0

    @property
    def has_nonlinearity(self) -> bool:  # Does any component carry a chi2 or chi3?
        return bool(self._chi2_components)

    @property
    def nonlinear_components(self) -> Tuple[str, ...]:  # The E components with a live nonlinearity.
        return tuple(component for component in ("Ex", "Ey", "Ez") if component in self._chi2_components)

    def is_nonlinear(self, component: str) -> bool:  # Does this component take the Pade correction?
        return component in self._chi2_components

    def chi2_for(self, component: str) -> Any:  # MEEP's s->chi2[ec]; 0.0 where the component is linear.
        return self._chi2_components.get(component, 0.0)

    def chi3_for(self, component: str) -> Any:  # MEEP's s->chi3[ec]; 0.0 where the component is linear.
        return self._chi3_components.get(component, 0.0)

    @property
    def has_conductivity(self) -> bool:  # Is a D conductivity installed on any component?
        return any(component in self._conductivity for component in ("Dx", "Dy", "Dz"))

    @property
    def has_magnetic_conductivity(self) -> bool:  # Is a B conductivity installed on any component?
        return any(component in self._conductivity for component in ("Bx", "By", "Bz"))

    @property
    def conductive_components(self) -> Tuple[str, ...]:  # The D/B components carrying a sigma.
        return tuple(c for c in CONDUCTIVE_COMPONENTS if c in self._conductivity)

    @property
    def has_polarizations(self) -> bool:  # Does any susceptibility carry a live P?
        return any(state.driven() for state in self.polarizations)

    @property
    def stores_E(self) -> bool:  # Is E a stored array rather than recomputed from D?
        """True when ``get_E`` returns storage — the invariant ``update_E`` must honour.

        Keying the constitutive sub-step on THIS rather than on
        ``has_polarizations`` is what stops the two disagreeing: a stored E that
        update_E declined to write would be returned as a volume of zeros by every
        monitor and every curl, which is a silent, total, entirely quiet failure.
        """
        return self._stored_E

    def get_E(self, component: str) -> Any:
        """
        Get an E-field component: E = D / eps.

        With E stored — PML active, a susceptibility present, an instantaneous
        nonlinearity present, or any combination — this returns the stored array (a
        live reference). Otherwise it computes E = D * inv_eps into a freshly
        allocated array, so the caller may mutate the result freely.

        The stored branch is not an optimisation: once a polarization advances at the
        end of the step, ``D * inv_eps`` is no longer the E of the step just
        finished, and recomputing it here would hand every monitor and every curl a
        field one polarization out of date. Under chi2/chi3 it is not even the same
        constitutive relation — the Pade factor is missing — so the recomputed field
        would be the linear one, harmonics and all removed.

        Args:
            component: 'Ex', 'Ey', or 'Ez'

        Returns:
            E-field array (stored reference when stored, freshly computed if not)
        """
        if self._stored_E:
            return getattr(self, component)  # Stored E: PML and/or dispersion.

        D_name = 'D' + component[1]  # Ex -> Dx
        D = getattr(self, D_name)
        if component in self.absent_components:
            # MEEP holds no array for it and reads it as 0.0 (monitor.cpp:160-164), so
            # +0 everywhere: D * inv_eps would be -0 wherever inv_eps is negative.
            return self.grid.xp.zeros(D.shape, dtype=D.dtype)
        return part_products.scale(self.inverse_epsilon_for(component), D)  # Per part, as MEEP.

    def set_absent_components(self, names: Any) -> None:  # MEEP's unallocated components.
        """Record the components MEEP allocated no field array for.

        MEEP allocates a field component only when something requires it:
        ``fields::_require_component`` (fields.cpp:552-572) allocates every component
        ``is_like`` the required one (fields.cpp:471-474), which in a 2-D cell is its
        own polarization only (TM = Ez, Dz, Hx, Hy, Bx, By against TE, fields.cpp:458-469)
        unless ``is_aniso2d`` couples the two, and ``gv.has_field`` keeps four
        components in 1-D. A component nothing required has ``f[c] == NULL``: the
        constitutive update skips it (``if (f[ec][cmp])``, update_eh.cpp:150), and the
        reader answers 0.0 for it (``fields_chunk::get_field``, monitor.cpp:160-164).

        Every array here is allocated and stepped regardless. With no source on them
        and nothing coupling them in, those components only ever hold zeros, so the
        difference is the sign of a zero. The curls and the absorber forms keep an
        all-zero neighbourhood at +0 (every coefficient they multiply by is positive,
        and ``+0 - (-0)`` is +0); the constitutive product with a negative inverse
        permittivity does not (MEEP's perfect-conductor cells hold chi1inv = -1e-20,
        and ``-1e-20 * (+0)`` is -0), and neither do the mirror fills on a folded grid,
        which write a D plane as ``parity * image``, -0 with an odd parity.
        ``update_E`` therefore leaves an absent E component untouched, at its initial
        +0, and :meth:`get_E` serves zeros for it, as MEEP does. A dispatched kernel
        that steps it anyway is cleared after its launch
        (``fastpath.AbsentComponentClear``). ``names`` is any iterable of component names; an empty one
        restores the default (every component stepped).
        """
        names = set(names)
        chosen = tuple(name for name in FIELD_COMPONENTS if name in names)
        unknown = sorted(names - set(FIELD_COMPONENTS))
        if unknown:
            raise ValueError(f"set_absent_components: {unknown} are not field components "
                             f"(the twelve are {FIELD_COMPONENTS}).")
        if chosen:
            self.absent_components = chosen
        else:
            self.__dict__.pop("absent_components", None)

    def component_factors(self, name: str) -> Tuple[Any, ...]:
        """The one or two whole-grid arrays whose POINTWISE PRODUCT is this component.

        One array where the component is stored; ``(D, inv_eps)`` on the DERIVED branch
        of :meth:`get_E`, where ``get_E`` would form the product over the whole grid
        for a caller that only wants a plane of it.

        THE MEASUREMENT. With no PML and no susceptibility nothing stores E, so every
        monitor read of an E component allocated and computed a full-volume
        ``D * inv_eps`` and kept one plane. On a 640x640 cell with one five-frequency
        flux plane: 1.298 ms per monitor per step, 52.6% of a bare step, against
        0.562 ms and 7.3% for the same monitor on a run that stores E. It is paid by
        exactly the runs that have the least work to hide it behind — periodic band
        structures, and the normalization leg of the two-run idiom.

        Handing back the FACTORS lets a caller slice or gather each one and multiply
        the blocks. ``(D * inv_eps)[block] == D[block] * inv_eps[block]`` cell for
        cell, so what comes out is bit-identical to indexing the product; the caller
        just never pays for the cells it discards. A caller must MULTIPLY rather than
        assume one array — the tuple's length is a property of the run's storage, not
        of the component name.
        """
        if name in ("Ex", "Ey", "Ez") and not self._stored_E:
            displacement = getattr(self, "D" + name[1])
            if name in self.absent_components:
                # MEEP's 0.0 for a component it holds no array for (get_E), as a
                # read-only view, so a caller indexing one plane allocates nothing.
                xp = self.grid.xp
                return (xp.broadcast_to(xp.zeros((), dtype=displacement.dtype), displacement.shape),)
            inverse = self.inverse_epsilon_for(name)
            if getattr(inverse, "shape", None) == displacement.shape:
                return (displacement, inverse)
        return (self.get_component(name),)

    def sliced_component(self, name: str, index: Any) -> Any:
        """One indexed block of a component, without materializing the whole volume.

        The contiguous case of :meth:`component_factors` — see there for why the
        factors are indexed instead of their product.
        """
        factors = self.component_factors(name)
        block = factors[0][index]
        for factor in factors[1:]:
            block = block * factor[index]
        return block

    def displacement_minus_polarization(self, component: str) -> Any:
        """MEEP's ``f_minus_p`` for one E component: D - sum_n P_n.

        Returns the D array ITSELF (aliased, not copied) when no susceptibility
        drives this component — MEEP does the same, feeding ``f[dc]`` straight through
        when ``f_minus_p`` was never allocated (update_eh.cpp:143-146) — so the caller
        must not mutate the result. When something does drive it, the difference is
        formed in a shared scratch buffer.

        Aliasing D is what makes the zero-strength degenerate case reduce byte for
        byte to the non-dispersive engine: ``D - 0.0`` is bit-exact for every finite
        value, but not forming the subtraction at all is exact by construction.

        Args:
            component: 'Ex', 'Ey', or 'Ez'
        """
        displacement = getattr(self, 'D' + component[1])
        contributors = [state for state in self.polarizations if state.drives(component)]
        if not contributors:
            return displacement
        if self._fmp_scratch is None:
            self._fmp_scratch = self.grid.xp.zeros(self.grid.shape, dtype=self._field_dtype())
        scratch = self._fmp_scratch
        scratch[...] = displacement
        for state in contributors:
            state.subtract_into(component, scratch)
        return scratch

    def displacement_minus_polarization_volumes(self) -> Dict[str, Any]:
        """All three ``D - sum P`` volumes at once — MEEP's ``dmp`` table.

        :meth:`displacement_minus_polarization` hands back a SHARED scratch buffer, on
        the invariant that each component is consumed before the next is formed. The
        instantaneous nonlinearity breaks that invariant: MEEP's ``Dsqr`` averages the
        two TRANSVERSE components into the position of the one being updated
        (step_generic.cpp:599), so Ez's update reads Dx and Dy while its own volume is
        live. This allocates one buffer per driven component instead, and only a
        nonlinear run ever pays for them.

        Undriven components are still ALIASED to their D array, exactly as MEEP passes
        ``f[dc]`` straight through when ``f_minus_p`` was never allocated
        (update_eh.cpp:143-146), so a nonlinear non-dispersive run allocates nothing
        here at all. The caller must not mutate the result.
        """
        volumes: Dict[str, Any] = {}
        for component in ("Ex", "Ey", "Ez"):
            displacement = getattr(self, "D" + component[1])
            contributors = [state for state in self.polarizations if state.drives(component)]
            if not contributors:
                volumes[component] = displacement
                continue
            scratch = self._fmp_scratch_by_component.get(component)
            if scratch is None:
                scratch = self.grid.xp.zeros(self.grid.shape, dtype=self._field_dtype())
                self._fmp_scratch_by_component[component] = scratch
            scratch[...] = displacement
            for state in contributors:
                state.subtract_into(component, scratch)
            volumes[component] = scratch
        return volumes

    def drive_field(self, component: str) -> Any:
        """MEEP's ``w``: the field that drives the polarization, ``(D - sum P) * inv_eps``.

        MEEP update_pols.cpp:44 — ``w[c][cmp] = f_w[c][cmp] ? f_w[c][cmp] :
        f[c][cmp]`` — and step_generic.cpp's ``step_update_EDHB``, which writes
        ``fw[i] = gs*us`` (the constitutive product) BEFORE the absorbing
        accumulation overwrites ``f``. So:

        * inside a PML the stored E has already had ``+= (kap+sig)*fw - (kap-sig)*fw_prev``
          applied and is NOT the constitutive product any more; ``f_w`` is;
        * with no PML the stored E is exactly the constitutive product.

        Driving P from the stored E instead is the single most likely silent wrong
        answer in dispersion: the two agree exactly outside the absorber, so every
        no-PML case passes and only the PML case is wrong — smoothly, plausibly, and
        in the region a transmission spectrum is normalised against.

        With PML the choice is MEEP's, PER CHUNK: ``f_w`` exists only in a chunk that
        carries the component's own (dsigw) layer, so ``w`` is ``f_w`` on those cells
        and the stored E everywhere else — which there IS the constitutive product,
        because ``update_E`` writes ``E = (D - P) * chi1inv`` outside the layer's
        chunks and leaves ``f_w`` untouched. The region plan ``update_E`` dispatched on
        (:attr:`region_plan`) says which cells are which; a component split between
        the two is composed into a pooled buffer, one per component, read before the
        next is formed.

        Args:
            component: 'Ex', 'Ey', or 'Ez'
        """
        if not self._pml_active:
            return getattr(self, component)
        plan = self.region_plan
        if plan is None:
            raise RuntimeError(
                f"drive_field({component!r}) under a PML needs the region plan the PML "
                f"sub-steps dispatch on, and none is installed: MEEP's w is f_w only in "
                f"the chunks that carry {component}'s own layer. The driver installs it "
                f"with the layer; stepping Fields directly installs it at the first "
                f"update_E.")
        mode = plan.spec("constitutive", component).mode
        stored = getattr(self, component)
        if mode == "outside_only":
            return stored
        aux = getattr(self, 'f_w_' + component)
        if mode == "inside_only":
            return aux
        xp = self.grid.xp
        drive = (xp.empty_like(stored) if self.scratch is None
                 else self.scratch.like("drive_" + component, stored))
        xp.copyto(drive, stored)
        xp.copyto(drive, aux, where=plan.mask("constitutive", component, xp))
        return drive

    @property
    def region_plan(self) -> Any:
        """The :class:`~.chunk_map.RegionPlan` the PML sub-steps dispatch on, or None.

        Derived, not state: it is rebuilt from the chunk division and the layer
        (``FdtdDriver`` installs it with the absorber; ``stepping`` installs the
        one-rank plan for Fields stepped directly), so a capsule does not carry it.
        """
        return self.__dict__.get("_region_plan")

    def install_region_plan(self, plan: Any) -> None:  # The driver's (or stepping's) plan.
        self.__dict__["_region_plan"] = plan

    @property
    def chunk_layout(self) -> Any:
        """The lifted :class:`~.chunk_map.ChunkLayout` the driver installed, or None.

        Derived, not state, like :attr:`region_plan`: ``FdtdDriver`` installs the
        division :meth:`~.driver.FdtdDriver.set_chunk_layout` gave it, with or without
        an absorbing layer, and None while it steps the one-rank default. ``update_E``
        reads off-diagonal presence from it (``stepping._chunk_layout``); with None it
        falls back to the region plan's one-rank division under a layer, and to the
        one chunk of an unlayered one-rank grid otherwise.
        """
        return self.__dict__.get("_installed_chunk_layout")

    def install_chunk_layout(self, layout: Any) -> None:  # The driver's lifted division, or None.
        self.__dict__["_installed_chunk_layout"] = layout

    def offdiag_presence(self, component: str, partner: str, layout: Any) -> Any:
        """Where MEEP adds row ``component``'s ``partner`` OFFDIAG term: None (everywhere) or a mask.

        MEEP passes NULL for an off-diagonal ``chi1inv`` array in every chunk that
        deleted it (:meth:`~.chunk_map.ChunkLayout.offdiag_kept`; update_eh.cpp:191-196),
        and for every array on the ``r = 0`` row of a cylindrical grid, which
        ``update_eh`` steps a second time with the off-diagonals NULL
        (update_eh.cpp:197-208; vec.hpp:1100-1104 keeps that row out of the first
        pass). Returns None when the term is added on every cell, otherwise a bool
        volume on this grid's array module, broadcastable to the field. ``layout``
        None is a one-chunk division, which keeps every array :attr:`has_offdiagonal_epsilon`
        holds (an identically zero one is dropped at install). Cached per (row,
        partner) on the layout's and the coefficient's identity; derived, not state.
        """
        import numpy  # noqa: PLC0415 - host masks; the field arrays stay on grid.xp

        coefficient = self._chi1inv_offdiagonal[component][partner]
        cache = self.__dict__.setdefault("_offdiag_presence", {})
        hit = cache.get((component, partner))
        if hit is not None and hit[0] is layout and hit[1] is coefficient:
            return hit[2]
        iyee = IYEE_SHIFTS[component]
        host = None if layout is None else layout.offdiag_presence(coefficient, iyee)
        grid = self.grid
        if getattr(grid, "cylindrical", False) and iyee[0] == 0 and int(grid.origin_doubled(0)) == 0:
            # Stored row 0 of a shift-0 component sits on the axis, r = 0.
            if host is None:
                host = numpy.ones(tuple(grid.shape), dtype=bool)
            host = numpy.array(host, dtype=bool, copy=True)
            host[0, :, :] = False
        mask = None if host is None else grid.xp.asarray(numpy.ascontiguousarray(host))
        cache[(component, partner)] = (layout, coefficient, mask)
        return mask

    def get_H(self, component: str) -> Any:
        """
        Get an H-field component: H = B / mu = B (mu = 1, non-magnetic).

        With PML active this returns the stored H array.

        WITHOUT PML THE RETURN VALUE IS THE B ARRAY ITSELF, NOT A COPY. mu = 1
        makes H numerically identical to B, so no array is allocated — but any
        in-place mutation of the returned array corrupts the primary B field and
        therefore the whole simulation. Copy before writing, or write to the B
        component explicitly and intentionally.

        Args:
            component: 'Hx', 'Hy', or 'Hz'

        Returns:
            H-field array (stored reference if PML active, the B array if not)
        """
        if self._pml_active:
            return getattr(self, component)  # PML active: stored H-field.

        B_name = 'B' + component[1]  # Hx -> Bx
        return getattr(self, B_name)

    def _init_material(self, eps_background: float = 1.0):
        """
        Initialize the material arrays.

        MEEP reference: structure::set_chi1inv() in structure.cpp — MEEP stores
        chi1inv = 1/epsilon so the constitutive update is a multiply.

        With susceptibilities present these hold eps_infinity and 1/eps_infinity —
        the instantaneous response — not eps(omega). Passing a material's STATIC
        permittivity here while also adding susceptibilities gives a medium of
        roughly doubled index: physical-looking and completely wrong.
        """
        xp = self.grid.xp
        shape = self.grid.shape

        self.eps = xp.full(shape, eps_background, dtype=xp.float32)
        self.inv_eps = xp.full(shape, 1.0 / eps_background, dtype=xp.float32)
        self._eps_components = {component: self.eps for component in ("Ex", "Ey", "Ez")}
        self._inv_eps_components = {
            component: self.inv_eps for component in ("Ex", "Ey", "Ez")
        }

    def set_epsilon_volumes(
        self,
        epsilon_by_component: Dict[str, Any],
        inverse_by_component: Dict[str, Any],
        chi1inv_offdiagonal: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Install diagonal/per-component epsilon arrays, optionally with a full row.

        Distinct arrays represent diagonal anisotropy or an isotropic geometry
        sampled at each E component's own Yee position. Aliased arrays retain
        the historical isotropic storage cost and arithmetic path.

        ``chi1inv_offdiagonal`` is MEEP's ``chi1inv[c][d]`` for ``d`` not the
        component's own direction: the outer key is the row component, the inner
        key the PARTNER component whose D the entry couples in, and the value the
        raw inverse-tensor entry — NOT a permittivity, and free to be negative or
        zero. MEEP registers these entries at the row component's Yee site minus
        half a cell along its own axis, the integer node
        (``structure_chunk::set_chi1inv``, anisotropic_averaging.cpp:248-257,
        ``here - shift1``); the arrays here are indexed by the row component's own
        loop exactly as MEEP's are, so the stencil in ``stepping.update_E``
        consumes them slot-for-slot and the registration never needs restating.

        Identically-zero rows are DROPPED, so a caller that passes explicit zeros
        reduces byte for byte to the diagonal engine — the same rule
        :meth:`set_nonlinear_volumes` applies to a zero chi2/chi3.

        A folded (mirror) axis takes any surviving row UNCHANGED: the stencil never ghosts
        the coefficient (the partner-axis shift touches the field before the multiply; the
        own-axis shift goes UP, away from the fold), so reconstructing neighbours with the
        FIELD's parity alone is the whole story — for an EVEN coefficient live ON the plane
        as much as an odd one. Measured 8.3e-13..4.7e-12 (:meth:`_validated_offdiagonal_rows`).

        Installing any row switches E to stored mode: the row product must be
        formed once per step in ``update_E`` (all three D volumes alive at once),
        not per ``get_E`` call.
        """
        expected = {"Ex", "Ey", "Ez"}
        if set(epsilon_by_component) != expected or set(inverse_by_component) != expected:
            raise ValueError("epsilon component maps must contain exactly Ex, Ey, and Ez.")
        surviving = self._validated_offdiagonal_rows(chi1inv_offdiagonal)
        self._eps_components = dict(epsilon_by_component)
        self._inv_eps_components = dict(inverse_by_component)
        self._chi1inv_offdiagonal = surviving
        if surviving:
            self.enable_field_storage()
        # Preserve the long-standing scalar attributes as the Ez view for code
        # that only needs a representative material array. Constitutive updates
        # always use the component-aware accessors below.
        self.eps = self._eps_components["Ez"]
        self.inv_eps = self._inv_eps_components["Ez"]

    def _validated_offdiagonal_rows(
        self, chi1inv_offdiagonal: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize the off-diagonal rows; empty dict when none survive."""
        if not chi1inv_offdiagonal:
            return {}
        xp = self.grid.xp
        surviving: Dict[str, Dict[str, Any]] = {}
        for row, partners in chi1inv_offdiagonal.items():
            if row not in ("Ex", "Ey", "Ez"):
                raise ValueError(
                    f"chi1inv_offdiagonal rows must be keyed Ex/Ey/Ez, got {row!r}."
                )
            for partner, values in (partners or {}).items():
                if partner not in ("Ex", "Ey", "Ez"):
                    raise ValueError(
                        f"chi1inv_offdiagonal partners must be Ex/Ey/Ez, got {partner!r}."
                    )
                if partner == row:
                    raise ValueError(
                        f"chi1inv_offdiagonal[{row!r}][{partner!r}] names the row's own "
                        f"component; the diagonal entry belongs in the epsilon maps, not here."
                    )
                array = xp.asarray(values)
                if array.dtype.kind == "c":
                    raise ValueError(
                        f"chi1inv_offdiagonal[{row!r}][{partner!r}] is complex; the "
                        f"inverse-permittivity tensor of a lossless medium is real."
                    )
                if tuple(array.shape) != tuple(self.grid.shape):
                    raise ValueError(
                        f"chi1inv_offdiagonal[{row!r}][{partner!r}] has shape "
                        f"{tuple(array.shape)}; the grid stores {tuple(self.grid.shape)}."
                    )
                array = array.astype(xp.float32, copy=False)
                if not bool(xp.all(xp.isfinite(array))):
                    raise ValueError(
                        f"chi1inv_offdiagonal[{row!r}][{partner!r}] contains non-finite "
                        f"values; every tensor entry must be finite."
                    )
                if not bool(xp.any(array)):
                    continue  # Identically zero: the diagonal engine, byte for byte.
                surviving.setdefault(row, {})[partner] = array
        # A mirror-folded grid takes the rows unchanged: driven fold-equivalence
        # against the full domain measures 8.3e-13 (odd coefficient) to 4.7e-12
        # (even + odd together) — the stencil never ghosts the coefficient, so
        # the field-parity ghosts the shift helpers build are the whole story.
        # See stepping._mask_metallic_wall_coupling for the measurement.
        return surviving

    @property
    def has_offdiagonal_epsilon(self) -> bool:
        """Whether any off-diagonal chi1inv row survived installation."""
        return bool(self._chi1inv_offdiagonal)

    def chi1inv_offdiagonal_for(self, component: str) -> Dict[str, Any]:
        """This row's off-diagonal entries, keyed by partner component; may be empty."""
        return self._chi1inv_offdiagonal.get(component, {})

    def set_isotropic_epsilon_volume(self, epsilon: Any, inverse: Any) -> None:
        """Install one aliased epsilon/inverse pair for all E components."""
        self.set_epsilon_volumes(
            {component: epsilon for component in ("Ex", "Ey", "Ez")},
            {component: inverse for component in ("Ex", "Ey", "Ez")},
        )

    def epsilon_for(self, component: str) -> Any:
        """Permittivity sampled for one electric component."""
        try:
            return self._eps_components[component]
        except KeyError as exc:
            raise ValueError(
                f"epsilon component must be Ex, Ey, or Ez, got {component!r}."
            ) from exc

    def inverse_epsilon_for(self, component: str) -> Any:
        """Inverse permittivity used by one electric constitutive update."""
        try:
            return self._inv_eps_components[component]
        except KeyError as exc:
            raise ValueError(
                f"epsilon component must be Ex, Ey, or Ez, got {component!r}."
            ) from exc

    @property
    def has_component_epsilon(self) -> bool:
        """Whether the three E components do not all alias one material array."""
        return len({id(value) for value in self._eps_components.values()}) > 1

    def set_background_eps(self, eps: float):  # Overwrite the whole volume with a uniform permittivity.
        xp = self.grid.xp
        epsilon = xp.full(self.grid.shape, eps, dtype=xp.float32)
        inverse = xp.full(self.grid.shape, 1.0 / eps, dtype=xp.float32)
        self.set_isotropic_epsilon_volume(epsilon, inverse)

    def add_sphere(self, center: Tuple[float, float, float],
                   radius: float, eps: float):
        """
        Add a hard-edged dielectric sphere (no subpixel smoothing).

        MEEP reference: geometric_object handling in structure.cpp
        """
        X, Y, Z = self.grid.meshgrid()
        cx, cy, cz = center

        r_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        mask = r_sq <= radius**2

        for component in ("Ex", "Ey", "Ez"):
            self._eps_components[component][mask] = eps
            self._inv_eps_components[component][mask] = 1.0 / eps

    def reset(self):  # Zero every allocated field array, keeping the material arrays.
        """Return every field and auxiliary to zero, leaving the material arrays alone.

        The polarizations' P AND P_prev both go: the ADE is second order, so a re-run
        that kept a stale P_prev would start from a history it never lived through and
        produce a wrong but entirely plausible field.
        """
        for array in [self.Dx, self.Dy, self.Dz,
                      self.Bx, self.By, self.Bz]:
            array.fill(0)

        # E exists without PML too once a susceptibility is present, so it is zeroed on
        # its own flag; H and the twelve auxiliaries are PML-only.
        for array in [self.Ex, self.Ey, self.Ez,
                      self.Hx, self.Hy, self.Hz,
                      self.fu_Bx, self.fu_By, self.fu_Bz,
                      self.fu_Dx, self.fu_Dy, self.fu_Dz,
                      self.f_cond_Dx, self.f_cond_Dy, self.f_cond_Dz,
                      self.f_cond_Bx, self.f_cond_By, self.f_cond_Bz,
                      self.f_bfast_Dx, self.f_bfast_Dy, self.f_bfast_Dz,
                      self.f_bfast_Bx, self.f_bfast_By, self.f_bfast_Bz,
                      self.f_w_Ex, self.f_w_Ey, self.f_w_Ez,
                      self.f_w_Hx, self.f_w_Hy, self.f_w_Hz,
                      self._fmp_scratch,
                      *self._fmp_scratch_by_component.values()]:
            if array is not None:
                array.fill(0)

        for state in self.polarizations:
            state.reset()

    def get_component(self, name: str) -> Any:
        """
        Get a field array by component name.

        E and H components route through get_E()/get_H() so they work in both
        the stored (PML) and computed (non-PML) modes.

        Args:
            name: Component name (Ex, Ey, Ez, Dx, Dy, Dz, Hx, Hy, Hz, Bx, By, Bz)

        Returns:
            Field array (computed on demand for E/H when PML is not active)

        Raises:
            ValueError: If the component name is not one of the twelve components.
        """
        if name in ('Ex', 'Ey', 'Ez'):
            return self.get_E(name)
        if name in ('Hx', 'Hy', 'Hz'):
            return self.get_H(name)
        if name in ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz'):
            return getattr(self, name)
        valid_names = 'Bx, By, Bz, Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz'
        raise ValueError(f"Invalid component name '{name}'. Valid names: {valid_names}")

    def to_cell_center(self, component: str, alias: Optional["HAliasRule"] = None) -> Any:
        """The component at every stored cell centre, as MEEP's ``get_array`` forms each value.

        Cell centre ``i`` on each axis sits at doubled ``origin + 2*i + 1``, the stored
        layout (halved on a folded axis, no periodic duplicate). Each value is MEEP's
        centred sum ``0.25 * (((a + b) + c) + d)`` over the component's chunk words
        (array_slice.cpp:413-415), stored with weight 1 and phase ``(1, +0)``: the word
        MEEP's ``get_array`` holds at that point. :mod:`~.readback` is the
        transcription; a host NumPy array, float32 or complex64 as the storage is.

        ``alias`` is the run's :class:`~.readback.HAliasRule`: a B component with H
        stored needs it, because a not-owned B word in a chunk where MEEP holds H as B's
        alias is the owner's H. ``FdtdDriver.get_field`` passes the driver's own.

        Raises:
            ValueError: an unknown component, or a B component read with H stored and
                no ``alias``.
        """
        from .readback import cell_center  # noqa: PLC0415 - readback imports this module

        if component not in FIELD_COMPONENTS:
            self.get_component(component)  # the named refusal
        return cell_center(self, component, alias)

    def to_meep_array(self, component: str, alias: Optional["HAliasRule"] = None,
                      center: Any = None, size: Any = None) -> Any:
        """MEEP's ``sim.get_array(component=c)`` (or with ``center``/``size``), word for word.

        The whole cell by default: ``fields::total_volume()``, every centred point MEEP
        visits, the full domain on a folded axis, the periodic duplicate plane where MEEP
        has one (phased by ``pow(eikna, -1)`` on a Bloch axis), MEEP's own rank and axis
        order (:meth:`meep_array_axes`). ``center``/``size`` select a volume as
        ``mp.Volume`` does, with MEEP's interpolation weights and collapse on an empty
        direction. :func:`~.readback.get_array` is the transcription; ``alias`` as for
        :meth:`to_cell_center`. A host NumPy array.
        """
        from .readback import get_array  # noqa: PLC0415 - readback imports this module

        if component not in FIELD_COMPONENTS:
            self.get_component(component)  # the named refusal
        return get_array(self, component, alias, center=center, size=size)

    def meep_array_axes(self) -> Tuple[int, ...]:
        """The axes ``sim.get_array`` reports, in x, y, z order — MEEP's ``n > 1`` filter.

        MEEP builds an array slice's shape one direction at a time and keeps only the
        entries longer than a single point (array_slice.cpp
        ``get_array_slice_dimensions``: ``if (n > 1) dims[rank++] = n;``), so its
        result has FEWER AXES rather than singleton ones. Three configurations reach
        that filter and each was measured against MEEP:

        * a reduced-dimension axis, which is not one of MEEP's directions at all — a
          2-D ``get_array`` is ``(nx[+1], ny[+1])`` and a 1-D one is ``(nz[+1],)``;
        * a one-cell axis of a genuinely 3-D run, which MEEP builds for
          ``cell_size.x = 0`` at ``dimensions=3`` — measured ``(21, 21)`` for a
          0.1 x 2 x 2 cell at resolution 10, not ``(2, 21, 21)``, because the periodic
          duplicate coincides with the plane it duplicates;
        * a mirror-folded one-cell axis, which cannot occur (a fold needs an even
          count of at least two) and is covered by the same rule anyway.

        Published rather than kept private because ``from_meep`` has to map an axis
        NUMBER onto a POSITION in the returned array before it can slice it — apply
        the x correction at position 0 of a 1-D result and it lands on z.
        In CYLINDRICAL mode the order is (z, r), not (r, z): MEEP's Dcyl strides
        make Z the fastest direction (vec.cpp yucky_directions), and
        ``get_array_slice_dimensions`` walks the directions in that internal
        order, so ``sim.get_array`` on a (r=2, z=4) cell at resolution 10 is
        measured (40, 20). Neither axis gains a duplicate plane — the r axis
        ends at the axis and the PEC wall, z is metallic by default and Bloch
        under a k_point, none of which are the unfolded-periodic case.

        The ``n > 1`` filter applies to Dcyl too, and this returned a hardcoded
        ``(2, 0)`` until it was measured otherwise: a cell declared
        ``mp.Vector3(sr, 0, 0)`` — the spelling of every Dcyl example with no z
        extent, ``ring-cyl.py`` and ``perturbation_theory.py`` among them — has ONE
        z cell, and MEEP reports ``(nr,)``. Measured ``(760,)`` on ring-cyl.py's own
        cell at resolution 20, where the two-axis reading predicted ``(1, 760)`` and
        refused the lift.
        """
        order = (2, 0) if getattr(self.grid, "cylindrical", False) else (0, 1, 2)
        return tuple(axis for axis in order if self._meep_array_points(axis) > 1)

    def _meep_array_points(self, axis: int) -> int:  # Points MEEP reports along one axis.
        """MEEP's ``n`` for one axis, before its ``n > 1`` filter.

        A folded axis reports its full count; an unfolded one its count plus the periodic
        duplicate, ``n_full + 1`` (a metallic axis reports ``n_full``, which cannot change
        the ``n > 1`` answer: both exceed 1 for the same axes). A one-cell axis reports 1,
        whatever its boundary: MEEP collapses it (:func:`~.readback.get_array`).
        """
        count = self.grid.shape_full[axis]
        if count == 1:
            return 1
        return count if self.grid.is_mirrored(axis) else count + 1

    def field_bytes_per_cell(self) -> int:
        """Per-cell cost of the DTYPE-CARRYING arrays alone — the part real mode halves.

        Every array counted here is float32 in a real run and complex64 in a complex
        one, so this number is exactly halved by ``force_complex_fields=False``:
        the D/B pair, the stored E, the PML H and its twelve auxiliaries, the six
        BFAST IIR states, the ``D - sum P`` scratch, and each susceptibility's
        P/P_prev/scratch.

        It is reported separately from :meth:`bytes_per_cell` because the TOTAL is not
        halved and saying otherwise would over-promise: ``eps``, ``inv_eps`` and the
        three conductivity volumes are float32 in both modes by construction, so a
        plain run goes 56 -> 32 B/cell (a 1.75x saving, not 2x) while its field arrays
        go 48 -> 24 exactly. A caller sizing a GPU job needs the honest total; a test
        pinning the halving needs this.
        """
        element = 8 if self.force_complex_fields else 4
        arrays = 6  # D and B, always.
        if self._stored_E:
            arrays += 3
        if self._pml_active:
            arrays += 3 + 6 + 6  # H, the six f_u, the six f_w.
            arrays += len(self._conductivity)  # f_cond, one per conductive component.
        if self.grid.bfast_active:
            arrays += 6  # f_bfast, one per D and B component (step_db.cpp:76-79).
        if self._fmp_scratch is not None:
            arrays += 1
        arrays += len(self._fmp_scratch_by_component)  # A nonlinear dispersive run's per-component D - P.
        total = arrays * element
        for state in self.polarizations:
            total += state.bytes_per_cell(element)
        return total

    def bytes_per_cell(self) -> int:
        """Storage this container costs per grid cell, susceptibilities included.

        A six-term metal fit (MEEP's Ag/Au are a Drude plus four or five Lorentzians)
        costs 240-360 B/cell against a 56 B/cell non-PML baseline, so a budget that
        counts only the field arrays under-reports such a run by six times — which is
        the difference between a job that dispatches to GPU and one that OOMs after
        the setup has been paid for.

        The material arrays are float32 in a real run and in a complex one alike, so
        this total shrinks by less than half when ``force_complex_fields`` is False;
        :meth:`field_bytes_per_cell` is the part that halves exactly.

        WHAT THIS DELIBERATELY EXCLUDES: the step loop's pooled temporaries
        (:class:`StepScratch`, read with ``fields.scratch.bytes_held()``). They are
        allocated on the first STEP rather than here, so counting them would make this
        number change under a caller who only built the container; and they are
        bounded and small next to the state — five element-sized volumes on the
        ordinary PML path (two stencil shifts, two curl accumulators and one previous
        value), three more on the conductive-PML path, and two on a cylindrical grid.
        A GPU budget wants both numbers added.
        """
        total = self.field_bytes_per_cell()
        material_arrays = {
            id(value)
            for value in (
                *self._eps_components.values(),
                *self._inv_eps_components.values(),
            )
        }
        total += len(material_arrays) * 4  # Material arrays are always float32.
        conductivity_arrays = {
            id(value)
            for value in (
                *self._conductivity.values(), *self._condfac.values(), *self._condinv.values(),
            )
        }
        # A uniform sigma installs ONE array object under all three D keys, so the set
        # collapses to three; a graded absorber's six components cost eighteen.
        total += len(conductivity_arrays) * 4
        nonlinear_arrays = {
            id(value)
            for value in (*self._chi2_components.values(), *self._chi3_components.values())
            if getattr(value, "shape", None)
        }
        total += len(nonlinear_arrays) * 4  # chi2/chi3 volumes; a uniform value costs none.
        return total

    def __repr__(self) -> str:
        mb = self.grid.total_cells * self.bytes_per_cell() / 1e6
        dtype_str = "complex64" if self.force_complex_fields else "float32"
        pml_str = ", pml_storage=True" if self._pml_active else ""
        pol_str = f", susceptibilities={len(self.polarizations)}" if self.polarizations else ""
        cond_str = f", conductivity={self.conductive_components}" if self._conductivity else ""
        chi_str = f", nonlinear={self.nonlinear_components}" if self.has_nonlinearity else ""
        return (
            f"Fields(shape={self.grid.shape}, dtype={dtype_str}, memory={mb:.1f}MB"
            f"{pml_str}{pol_str}{cond_str}{chi_str})"
        )
