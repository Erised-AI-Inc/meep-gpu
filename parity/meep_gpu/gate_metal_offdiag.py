"""Byte gate for the Metal TENSOR (off-diagonal epsilon) ``update_E`` kernel.

THE CLAIM THIS GATE MAKES, and the only one: **byte-identity to ``stepping.py`` on
this host, subject to a declared and CHECKED subnormal-free precondition.** Not a
stated tolerance. Leg 5 is that check and it is REQUIRED to fire on the scaled
control — a precondition never demonstrated to fire is decorative.

THE ORACLE IS IN PROCESS. ``stepping.update_E(fields, pml)`` and the Metal launch
run in ONE process against ONE seed, on real ``Grid``/``Fields``/``PML`` objects
carrying rows installed through the PUBLIC installer
(``Fields.set_epsilon_volumes``). The in-file transcription stays, in a reduced
role: it is the mutation substrate and the only way to exercise a deliberately
wrong configuration. LEG 0 pins it against ``stepping.py`` over multiple full
cycles BEFORE any Metal kernel is trusted, which is what stops "bit-identical"
from meaning the kernel reproduced whatever the harness wrote twice.

WHAT THIS FAMILY ADDS OVER THE CERTIFIED CONSTITUTIVE ONE, and therefore what this
gate is actually about: the row product reads NEIGHBOURS through a transverse Yee
average. It is the first Metal constitutive stencil that is not element-wise. The
one genuinely new arithmetic element is THE COEFFICIENT MULTIPLY SITTING BETWEEN
THE TWO SHIFTS, and the leg that matters most is the mutation that hoists it.

THE ALLCLOSE-BLINDNESS CONTROL IS CARRIED HERE (leg 6, ``m1`` on UNIFORM rows at
2**-16 amplitude). It is the sharpest justification of byte gating in this project:
on uniform rows the hoist is ALGEBRAICALLY EQUAL and differs only through
distributivity rounding (~1 ulp relative, far below ``rtol=1e-5``), while scaling
the state by an exact power of two commutes with rounding — so the byte mismatches
SURVIVE the scaling while every absolute delta drops below ``atol=1e-8``. The case
must FAIL bytes and PASS ``allclose``. The measured wrong turns are recorded beside
it: ``m1`` on VARYING rows cannot serve, because a first-order registration defect
is O(1) relative and ``allclose`` correctly catches it at any amplitude (``rtol``
is scale-free), and the 0.25 distribution cannot serve because it is a true null.

THE TEN LEGS:

  0  transcription  the in-gate NumPy reference vs ``stepping.py``, full cycles
  0b execution      proof the bytes came off the GPU: mirror residency, the entry
                    point's TYPE, the launch counter, a pre-launch negative control,
                    and the CPU-bound fallback measured as a loud refusal
  1  reference      the Metal kernel vs ``stepping.py``, real engine objects
  2  synthetic      shapes x boundaries x row forms x inv-eps forms x amplitudes
  3  seam           partial rows: uncoupled components match the CERTIFIED plain
                    kernel while the coupled one differs (non-vacuous)
  4  signed_zero    +-0 lattice seeding, census-floored per seeding
  5  precondition   the subnormal census, and the control that makes it fire
  6  mutations      armed, launch-counted, three-valued, with the blindness control
  7  refusals       every named refusal, including the STALE-DOCSTRING guard
  8  engine         plans from the engine's own objects + the disjointness sweep,
                    and the WIRING CONTRACT: the arm is wired as of tranche 2, so
                    the leg asserts plan_step selects it BY LABEL for update_E
                    while the plain E-side arm still refuses (see leg 8's own
                    docstring for why re-pointing that assertion is a strictly
                    stronger check and not a relaxation)

CASE DISCIPLINE, inherited and non-negotiable: uint32 compares never ``allclose``;
a non-power-of-two Courant in every sweep; every case proves the step MOVED STATE
**and** that its reference bytes DIFFER from a coefficient-dropped diagonal control
(so no case passes with the coupling dark); a CANCELLATION amplitude class where
``gs*us ~ -coupling``, without which the row-sum association is not byte-visible in
f32 — MEASURED and floored, in both directions, because its first shape was inert
(see :data:`AMPLITUDES`); ALL EIGHT boundary triples, because the original four
were all x/y-symmetric and an axis confusion survives in the other four; MANDATORY
spatially varying row volumes plus both three-distinct and three-alias
inverse-epsilon forms; predicted nulls recorded WITH their derivations AND with a
sensitivity control that must fire; armed mutations launch-counted with DISARMED /
NEEDLE-MISSED fail paths; and a refusals leg that also records the DELIBERATE
ADMISSIONS with a byte measurement, so "correctly admitted" is distinguishable
from "never asked".

NOTE ON ``dtdx``: it does not enter this sub-step's arithmetic at all. The
non-power-of-two-Courant mandate therefore lands in the REAL-LAYER COEFFICIENT
TABLES here — the ``kps_a_h``/``kms_a_h`` vectors a Courant of 0.35 produces — and
the artifact says so rather than sweeping a dead parameter and calling it coverage.

STATED WEAKNESS, recorded because it is a real difference from the Triton
certification: there is NO PTX-EQUIVALENT AUDIT on this executor.
``compile_shader`` exposes no disassembly, so a "refuse the compile when the
generated code violates the policy" guard has no Metal analogue and the mutation
legs plus this gate are the only arbiters.

Usage::

    MEEP_GPU_SUBNORMAL_POLICY=flush python -u \\
      parity/meep_gpu/gate_metal_offdiag.py \\
      --out parity/meep_gpu/results/metal_offdiag_2026-08-15/gate.json
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep (MEEP's set_zero_subnormals is a no-op
# under #if HAVE_IMMINTRIN_H, so match_meep measures "keep"). The gate requests
# flush EXPLICITLY, before anything resolves a policy, and stamps the resolution
# into the artifact — the claim is only as good as the precondition it was
# certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import IYEE_SHIFTS, Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    coverage as metal_coverage,
    device,
    launch as metal_launch,
    offdiag_update_e as offdiag,
    preconditions,
    shaders,
    subnormal,
    templates,
)
from meep_gpu.pml import PML  # noqa: E402

from metal_gate_kit import (  # noqa: E402
    Counter,
    MutationHarness,
    argument_parser,
    assert_census_floor,
    assert_moved,
    cannot_certify,
    differing,
    environment_stamp,
    log,
    needle,
    predicted_null,
    provenance,
    run_legs,
    save,
    state_digest,
    summarize,
    wanted_legs,
)


def allclose_passes(left: Any, right: Any, rtol: float = 1e-5,
                    atol: float = 1e-8) -> bool:
    """Would ``allclose`` have accepted this pair?

    Local to this gate and used for exactly ONE purpose — the blindness control,
    where a case must FAIL bytes while PASSING allclose. It is deliberately NOT in
    the shared kit: a tolerance helper sitting beside `differing` in a module every
    family imports is an invitation to reach for it as a verdict, which would make
    these gates the thing they exist to replace.
    """
    return bool(np.allclose(np.asarray(left, dtype=np.float64),
                            np.asarray(right, dtype=np.float64),
                            rtol=rtol, atol=atol))


def assert_state_moved(before: Dict[str, Any], after: Dict[str, Any],
                       names: Sequence[str], label: str) -> int:
    """The step MOVED STATE, with the count returned for the artifact row."""
    moved = sum(differing(before[name], after[name]) for name in names)
    assert_moved(moved, label)
    return moved


def assert_differs_from_control(measured: Dict[str, Any], control: Dict[str, Any],
                                names: Sequence[str], label: str,
                                what: str) -> int:
    """The FEATURE WAS ON. Returns how many words separate it from the control.

    Without this a case passes just as happily with the coupling dark: a coupling of
    exactly zero reproduces the diagonal engine byte for byte, and a gate that only
    compared kernel against oracle would call that a pass FOR THE COUPLING.
    """
    delta = sum(differing(measured[name], control[name]) for name in names)
    assert delta > 0, (
        f"{label}: the reference bytes are IDENTICAL to the {what} control, so this "
        f"case would pass with the feature dark. Nothing about {what} is measured "
        f"here")
    return delta

PERIODIC, METALLIC = templates.PERIODIC, templates.METALLIC

#: The volumes a full state carries.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: What ``update_E`` writes and what every comparison therefore covers. THE
#: AUXILIARIES ARE STATE: a kernel that is right for one launch and wrong forever
#: after only diverges once ``f_w_*`` is compared, and omitting them is how that
#: defect hides.
COMPARED: Tuple[str, ...] = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: Non-power-of-two FIRST. dtdx does not enter this sub-step, so 0.35 buys its
#: coverage through the REAL-LAYER COEFFICIENT TABLES rather than through a
#: multiplier in the kernel — see the module docstring.
COURANTS: Tuple[float, ...] = (0.35, 0.5)

#: ALL EIGHT boundary triples, and the last four are a hole this round closed.
#:
#: The first four were the whole matrix: (0,0,0), (1,1,0), (1,1,1), (0,0,1). Every
#: one of them is SYMMETRIC in x and y, so a defect that treated one axis as
#: another — the m7 class, the wall mask applied to the wrong axis of the wrong
#: component — could still be caught only where the two axes it confused happened
#: to carry the same boundary. The four appended triples are exactly the
#: asymmetric ones: a single metallic x, a single metallic y, and the two mixed
#: pairs. Measured clean on all eight (2026-08-15, 0 differing words each), but by
#: a probe rather than by this gate, which is the definition of a hole.
#:
#: INDEX ORDER IS LOAD-BEARING: legs 3/4/5/7 index this tuple (``CONFIGS[1]`` is
#: the metallic-xy workhorse, ``CONFIGS[1:3]`` the two mostly-metallic ones), so
#: the new entries are APPENDED rather than interleaved.
CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any], ...] = (
    ("periodic_all", (1.2, 1.0, 0.9), "periodic"),
    # The metallic axis in BOTH roles: x and y are a coupled component's transverse
    # (partner) axis AND another component's own axis, so one config exercises the
    # partner-axis near ghost, the own-axis far ghost and the wall mask at once.
    ("metallic_xy", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic")),
    ("metallic_all", (1.1, 1.0, 0.9), "metallic"),
    ("metallic_z", (1.2, 1.0, 0.9), ("periodic", "periodic", "metallic")),
    ("metallic_x", (1.2, 1.0, 0.9), ("metallic", "periodic", "periodic")),
    ("metallic_y", (1.2, 1.0, 0.9), ("periodic", "metallic", "periodic")),
    ("metallic_xz", (1.2, 1.0, 0.9), ("metallic", "periodic", "metallic")),
    ("metallic_yz", (1.2, 1.0, 0.9), ("periodic", "metallic", "metallic")),
)

#: Row forms. ``uniform_full`` is what the blindness control needs (the hoist is
#: algebraically equal there); ``varying_full`` is MANDATORY and is what makes the
#: REGISTRATION — not merely the rounding — byte-visible.
#:
#: THE LAST TWO ARE A VALUE-CLASS AXIS ON THE COEFFICIENTS, and they were missing.
#: Every form above draws the rows from one narrow normal distribution, so the
#: whole sweep exercised a single exponent decade and no signed zero ever reached
#: the coefficient side — while the coefficient is the operand this family adds
#: over the certified constitutive kernel, and it is the one whose value class
#: decides whether ``pair * coefficient`` and the two shifts round the same way on
#: both executors. ``signed_zero_rows`` stamps a +-0 lattice into the coefficients
#: (the installer keeps them: it drops only IDENTICALLY zero rows); ``wide_exponent``
#: spreads them over 2**+-20, which puts the row product and the diagonal product
#: dozens of decades apart inside one grid. Both are floored for non-vacuity by
#: :func:`_row_value_class` rather than trusted to have been constructed.
ROW_FORMS: Tuple[str, ...] = ("uniform_full", "varying_full", "single_ex_ey",
                              "single_ez_ex", "signed_varying",
                              "signed_zero_rows", "wide_exponent")

#: The uniform tensor's off-diagonal entries, sign included: an inverse
#: permittivity tensor's off-diagonals are free to be negative and typically are.
UNIFORM_ROWS: Dict[Tuple[str, str], float] = {
    ("Ex", "Ey"): -0.031, ("Ex", "Ez"): 0.017,
    ("Ey", "Ez"): -0.022, ("Ey", "Ex"): 0.041,
    ("Ez", "Ex"): 0.013, ("Ez", "Ey"): -0.028,
}

SEED = 20260815


# ---------------------------------------------------------------------------
# Building real engine objects
# ---------------------------------------------------------------------------

def make_rows(form: str, shape, rng, scale: float = 1.0
              ) -> Dict[str, Dict[str, Any]]:
    """One row-coefficient set in ``fields.set_epsilon_volumes`` shape.

    Every form is a VOLUME, never a scalar: this family carries no scalar arm and
    the predicate refuses one by name.
    """
    def uniform(value: float) -> Any:
        return np.full(shape, np.float32(value * scale), dtype=np.float32)

    def varying(value: float) -> Any:
        return (np.float32(value * scale)
                * (1.0 + 0.4 * rng.standard_normal(shape))).astype(np.float32)

    def signed_zero(value: float) -> Any:
        """A varying row with a +-0 LATTICE stamped through it.

        The installer drops only an IDENTICALLY zero row (fields.py:1302-1303), so
        a row that is zero on a lattice and live elsewhere survives — which is what
        makes the signed-zero class constructible on the COEFFICIENT side at all.
        """
        out = varying(value)
        out.reshape(-1)[::13] = np.float32(-0.0)
        out.reshape(-1)[5::19] = np.float32(0.0)
        return out

    def wide(value: float) -> Any:
        """A varying row spread over 2**+-20 — the exponent-range class.

        An exact power of two per cell, so the spread is a pure exponent change and
        the mantissas stay the ones ``varying`` drew: the class isolates dynamic
        range from a change of distribution.
        """
        exponents = rng.integers(-20, 21, size=shape)
        return (varying(value) * np.float32(2.0) ** exponents).astype(np.float32)

    if form == "uniform_full":
        return {
            "Ex": {"Ey": uniform(UNIFORM_ROWS[("Ex", "Ey")]),
                   "Ez": uniform(UNIFORM_ROWS[("Ex", "Ez")])},
            "Ey": {"Ez": uniform(UNIFORM_ROWS[("Ey", "Ez")]),
                   "Ex": uniform(UNIFORM_ROWS[("Ey", "Ex")])},
            "Ez": {"Ex": uniform(UNIFORM_ROWS[("Ez", "Ex")]),
                   "Ey": uniform(UNIFORM_ROWS[("Ez", "Ey")])}}
    if form in ("signed_zero_rows", "wide_exponent"):
        draw = signed_zero if form == "signed_zero_rows" else wide
        return {"Ex": {"Ey": draw(UNIFORM_ROWS[("Ex", "Ey")]),
                       "Ez": draw(UNIFORM_ROWS[("Ex", "Ez")])},
                "Ey": {"Ez": draw(UNIFORM_ROWS[("Ey", "Ez")]),
                       "Ex": draw(UNIFORM_ROWS[("Ey", "Ex")])},
                "Ez": {"Ex": draw(UNIFORM_ROWS[("Ez", "Ex")]),
                       "Ey": draw(UNIFORM_ROWS[("Ez", "Ey")])}}
    if form == "varying_full":
        return {"Ex": {"Ey": varying(UNIFORM_ROWS[("Ex", "Ey")]),
                       "Ez": varying(UNIFORM_ROWS[("Ex", "Ez")])},
                "Ey": {"Ez": varying(UNIFORM_ROWS[("Ey", "Ez")]),
                       "Ex": varying(UNIFORM_ROWS[("Ey", "Ex")])},
                "Ez": {"Ex": varying(UNIFORM_ROWS[("Ez", "Ex")]),
                       "Ey": varying(UNIFORM_ROWS[("Ez", "Ey")])}}
    if form == "single_ex_ey":
        return {"Ex": {"Ey": varying(UNIFORM_ROWS[("Ex", "Ey")])}}
    if form == "single_ez_ex":
        return {"Ez": {"Ex": varying(UNIFORM_ROWS[("Ez", "Ex")])}}
    if form == "signed_varying":
        return {"Ex": {"Ey": (0.08 * rng.standard_normal(shape)).astype(np.float32)},
                "Ez": {"Ey": (0.06 * rng.standard_normal(shape)).astype(np.float32)}}
    raise ValueError(f"unknown row form {form!r}")


def build(cell, boundaries, courant: float, seed: int, form: str = "varying_full",
          scale: float = 1.0, signed_zeros: bool = False,
          aliased_inverse_epsilon: bool = False, rows: Any = None,
          resolution: float = 10.0) -> Tuple[Any, Any, Any]:
    """A real Grid/Fields/PML triple with rows installed through the PUBLIC installer.

    ``aliased_inverse_epsilon`` is the isotropic install (one array handed back as
    all three components) against the three-distinct-volume form; both are swept,
    because the plan's aliasing refusal must admit aliased INPUTS while refusing an
    input that aliases an OUTPUT, and a sweep over only one form measures half of
    that.

    ``signed_zeros`` seeds a +-0 LATTICE into every volume. MEEP keeps float32
    subnormals on arm64, so the signed-zero class is constructible on the HOST side
    here — unlike on a flushing x86 host.
    """
    grid = Grid(resolution=resolution, cell_size=cell, boundaries=boundaries,
                dimensions=3, courant=courant, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    count = int(np.prod(shape))
    index = np.arange(count, dtype=np.float32).reshape(shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    rng = np.random.default_rng(seed)
    if aliased_inverse_epsilon:
        eps_map = {c: epsilon for c in ("Ex", "Ey", "Ez")}
        inv_map = {c: inverse for c in ("Ex", "Ey", "Ez")}
    else:
        eps_map, inv_map = {}, {}
        for offset, component in enumerate(("Ex", "Ey", "Ez")):
            local = (epsilon * np.float32(1.0 + 0.05 * offset)).astype(np.float32)
            eps_map[component] = local
            inv_map[component] = (np.float32(1.0) / local).astype(np.float32)
    installed = make_rows(form, shape, rng, scale=1.0) if rows is None else rows
    fields.set_epsilon_volumes(eps_map, inv_map, installed)
    pml = PML(grid=grid,
              thickness=tuple((2, 2) if shape[a] >= 6 else (0, 0) for a in range(3)))
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(shape) * (0.37 * scale)).astype(np.float32)
        if signed_zeros:
            host.reshape(-1)[::17] = np.float32(-0.0)
            host.reshape(-1)[7::23] = np.float32(0.0)
        array[...] = host
    return grid, fields, pml


def _bare_rig(grid, complex_fields: bool = False) -> Tuple[Any, Any]:
    """A minimal Fields/PML pair on an already-built grid, rows installed.

    :func:`build` constructs its own Cartesian grid; the refusal cases below need
    grids ``build`` cannot make — a beta run, a BFAST run, a cylindrical one — so
    the grid arrives ready and this fills in the rest. The absorber skips any axis
    too thin to hold the layer, which is what a reduced axis on those grids needs.
    """
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    fields.enable_pml_storage()
    shape = grid.shape
    rng = np.random.default_rng(SEED)
    fields.set_epsilon_volumes(
        {c: np.full(shape, np.float32(2.0), dtype=np.float32)
         for c in ("Ex", "Ey", "Ez")},
        {c: np.full(shape, np.float32(0.5), dtype=np.float32)
         for c in ("Ex", "Ey", "Ez")},
        {"Ex": {"Ey": (0.03 * rng.standard_normal(shape)).astype(np.float32)}})
    return fields, PML(grid=grid,
                       thickness=tuple((2, 2) if shape[a] >= 6 else (0, 0)
                                       for a in range(3)))


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


class _Overridden:
    """One engine object with a few attributes replaced.

    The refusals leg has to CONSTRUCT each unimplemented feature and CALL the
    predicate — reading the clause is not evidence that it fires — and several of
    those features have no installer to reach them through: a scalar inverse
    epsilon, an ``xp`` that is not NumPy, ``stores_E`` false on a run whose rows
    force it true. This is how such a configuration is presented to the predicate
    without teaching the engine to build one.
    """

    def __init__(self, inner: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_inner"), name)


class _PlantedRows(_Overridden):
    """A ``Fields`` whose off-diagonal rows bypassed ``set_epsilon_volumes``.

    The installer normalizes every malformed row away — it casts to float32,
    refuses complex and a shape mismatch, refuses non-finite values and drops an
    identically zero row (fields.py:1262-1310) — so ``_row_reasons``' clauses are
    unreachable through it. They exist for a row that arrives another way, and the
    package's own rule is that such a row must be refused BY NAME rather than
    stepped. This is the only way to hand the predicate one.
    """

    def __init__(self, inner: Any, rows: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(inner,
                         chi1inv_offdiagonal_for=lambda c: rows.get(c, {}),
                         has_offdiagonal_epsilon=True)


class _NoDtype:
    """Volume-shaped, answers no dtype — the fail-closed case, measured."""

    shape = (12, 10, 9)


def boundary_codes(grid, pml) -> Tuple[int, int, int]:
    kinds = stepping._boundary_kinds(grid, pml)
    return tuple(METALLIC if kind == "metallic" else PERIODIC for kind in kinds)


def constitutive_coefficients(pml, suffix: str = "_h") -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kps", "kms")}


# ---------------------------------------------------------------------------
# The in-gate transcription — the MUTATION SUBSTRATE, pinned by leg 0
# ---------------------------------------------------------------------------

def _shifted(array: Any, axis: int, up: bool, code: int) -> Any:
    """``stepping._shift_up`` / ``_shift_down`` for one axis, as a gather.

    ``np.roll`` is a pure gather and preserves every bit including signed zeros;
    the metallic face is then overwritten with an exact ``+0.0``, which is what
    the kernel's ternary delivers.
    """
    out = np.roll(array, -1 if up else 1, axis=axis)
    if code == METALLIC:
        index: List[Any] = [slice(None)] * 3
        index[axis] = array.shape[axis] - 1 if up else 0
        out[tuple(index)] = np.float32(0.0)
    return out


def reference_offdiag(targets: Sequence[Any], aux: Sequence[Any],
                      sources: Sequence[Any], inverse_epsilon: Sequence[Any],
                      rows: Sequence[Optional[Any]], coefficients: Dict[str, Any],
                      codes: Sequence[int], walls: Sequence[int],
                      hoist: bool = False, same_direction: bool = False,
                      distribute_quarter: bool = False,
                      drop_wall_mask: bool = False,
                      overapply_wall_mask: bool = False,
                      commute_row_sum: bool = False,
                      store_before_load: bool = False) -> None:
    """``stepping.update_E``'s off-diagonal branch, transcribed elementwise, in place.

    The keyword flags are the HOST-SIDE mutations: they let the mutation leg plant a
    defect in the ORACLE and confirm the harness would have caught it, which is what
    separates "the kernel matches the reference" from "the reference is right".
    Every flag defaults to the transcribed behaviour.
    """
    for component, (_name, _source, own_axis) in enumerate(offdiag.E_TERMS):
        gs = sources[component]
        us = inverse_epsilon[component]
        total = None
        for offset in (0, 1):
            coefficient = rows[2 * component + offset]
            if coefficient is None:
                continue
            partner_axis = offdiag.TRANSVERSE_PARTNERS[component][offset]
            values = sources[partner_axis]
            # g[i] + g[i-sx]: half a cell DOWN the partner's own axis.
            pair = values + _shifted(values, partner_axis,
                                     up=bool(same_direction),
                                     code=codes[partner_axis])
            if hoist:
                # THE DEFECT: a plain four-point average times u[i]. Same algebra
                # only for a uniform coefficient, and bitwise different even then.
                four = pair + _shifted(pair, own_axis, up=True, code=codes[own_axis])
                term = 0.25 * (four * coefficient)
            else:
                product = pair * coefficient
                shifted = _shifted(product, own_axis, up=True, code=codes[own_axis])
                if distribute_quarter:
                    term = 0.25 * product + 0.25 * shifted
                else:
                    term = 0.25 * (product + shifted)
            total = term if total is None else total + term
        if total is not None:
            if not drop_wall_mask:
                axes = (range(3) if overapply_wall_mask
                        else offdiag.WALL_MASK_AXES[component])
                for axis in axes:
                    if walls[axis]:
                        index: List[Any] = [slice(None)] * 3
                        index[axis] = 0
                        total[tuple(index)] = 0
            constitutive = (total + gs * us) if commute_row_sum else (gs * us + total)
        else:
            constitutive = gs * us
        axis_name = "xyz"[own_axis]
        kps = coefficients["kps_" + axis_name]
        kms = coefficients["kms_" + axis_name]
        if store_before_load:
            aux[component][...] = constitutive
            previous = aux[component].copy()
        else:
            previous = aux[component].copy()
            aux[component][...] = constitutive
        targets[component] += kps * aux[component]
        targets[component] -= kms * previous


def reference_state(state: Dict[str, Any], pml, rows: Sequence[Optional[Any]],
                    inverse_epsilon: Sequence[Any], codes, walls,
                    **flags: Any) -> Dict[str, Any]:
    """Run the transcription over a copy of one state and return the result."""
    arrays = {name: np.array(state[name], copy=True) for name in state}
    reference_offdiag(
        [arrays[t[0]] for t in offdiag.E_TERMS],
        [arrays["f_w_" + t[0]] for t in offdiag.E_TERMS],
        [arrays[t[1]] for t in offdiag.E_TERMS],
        inverse_epsilon, rows, constitutive_coefficients(pml), codes, walls,
        **flags)
    return arrays


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def run_on_device(state: Dict[str, Any], pml, rows_map: Dict[str, Dict[str, Any]],
                  inverse_epsilon: Dict[str, Any], codes, walls,
                  source: Optional[str] = None,
                  contract: str = shaders.CONTRACT_OFF,
                  suffix: str = "_h",
                  counter: Optional[List[Counter]] = None) -> Dict[str, Any]:
    """One launch through the SHIPPED plan object, returning the host results.

    Everything goes through ``plan_offdiag_constitutive_from_arrays`` ->
    ``KernelPlan.run`` so the bytes this gate certifies are the bytes the engine
    route would launch. ``source`` is the mutation seam; dropping it is not a
    silent slowdown but a silent DISARMING.
    """
    targets = tuple(term[0] for term in offdiag.E_TERMS)
    arrays: Dict[str, Any] = {}
    for term in offdiag.E_TERMS:
        arrays[term[0]] = np.array(state[term[0]], copy=True)
        arrays["f_w_" + term[0]] = np.array(state["f_w_" + term[0]], copy=True)
        arrays[term[1]] = np.array(state[term[1]], copy=True)
    for name in targets:
        arrays["inv_eps_" + name] = inverse_epsilon[name]
    flat = {key: np.asarray(value).reshape(-1)
            for key, value in constitutive_coefficients(pml, suffix).items()}

    residency = device.Residency()
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF:
        row_values = tuple((rows_map.get(row) or {}).get(partner)
                           for row, partner in offdiag.ROW_SLOTS)
        row_mask = tuple(int(value is not None) for value in row_values)
        text = source if source is not None else offdiag.offdiag_source(
            row_mask, codes, walls, contract)
        function = device.compile_source(text).offdiag_constitutive_step
        if counter is not None:
            function = Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = offdiag.plan_offdiag_constitutive_from_arrays(
        arrays, flat, rows_map, codes, walls, residency, functions=functions)
    plan.run()
    residency.sync_out()
    assert plan.launches == 1, (
        "the plan reports zero launches: a byte comparison against a plan that "
        "never ran is trivially satisfied")
    return arrays


def run_certified_constitutive(state: Dict[str, Any], pml,
                               inverse_epsilon: Dict[str, Any]) -> Dict[str, Any]:
    """The CERTIFIED plain E-side constitutive kernel on the same state.

    Two roles, and both are load-bearing. As the COUPLING-DROPPED CONTROL it is
    what makes every case non-vacuous: a coupling of exactly zero reproduces this
    byte for byte, so a case whose reference equals this control is measuring
    nothing about the coupling. As the SEAM ORACLE (leg 3) it is what the
    uncoupled components of a partial-row build must match exactly.
    """
    targets = ("Ex", "Ey", "Ez")
    arrays = {name: np.array(state[name], copy=True)
              for name in targets + ("f_w_Ex", "f_w_Ey", "f_w_Ez",
                                     "Dx", "Dy", "Dz")}
    for name in targets:
        arrays["inv_eps_" + name] = inverse_epsilon[name]
    flat = {key: np.asarray(value).reshape(-1)
            for key, value in constitutive_coefficients(pml, "_h").items()}
    residency = device.Residency()
    plan = metal_launch.plan_constitutive_from_arrays("E", arrays, flat, residency)
    plan.run()
    residency.sync_out()
    return arrays


def rows_as_slots(rows_map: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Any], ...]:
    return tuple((rows_map.get(row) or {}).get(partner)
                 for row, partner in offdiag.ROW_SLOTS)


def installed_rows(fields) -> Dict[str, Dict[str, Any]]:
    return {row: dict(fields.chi1inv_offdiagonal_for(row))
            for row in ("Ex", "Ey", "Ez")}


def inverse_map(fields) -> Dict[str, Any]:
    return {name: fields.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez")}


# ---------------------------------------------------------------------------
# LEG 0 — the transcription, pinned against stepping.py over full cycles
# ---------------------------------------------------------------------------

def leg_transcription(payload: Dict[str, Any], out: str, cycles: int = 3) -> None:
    """The in-gate reference vs ``stepping.py`` itself, over multiple full cycles.

    THIS LEG COMES FIRST AND NOTHING ELSE IS TRUSTED WITHOUT IT. The reference is
    the mutation substrate: leg 6 plants defects in it to confirm the comparison
    would catch them. A substrate that is not the array path makes every one of
    those measurements a statement about the harness.

    Multiple cycles, not one: ``f_w_*`` is STATE, so a transcription that is right
    for one call and wrong for the next only diverges on the second.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, courant, SEED)
            codes = boundary_codes(grid, pml)
            walls = offdiag.wall_mask_axes(grid)
            slots = rows_as_slots(installed_rows(fields))
            inverse = [fields.inverse_epsilon_for(n) for n in ("Ex", "Ey", "Ez")]
            state = snapshot(fields)
            mirror = {key: np.array(value, copy=True) for key, value in state.items()}
            worst = 0
            moved_total = 0
            for cycle in range(cycles):
                before = {key: np.array(value, copy=True)
                          for key, value in mirror.items()}
                stepping.update_E(fields, pml)
                oracle = snapshot(fields)
                reference_offdiag(
                    [mirror[t[0]] for t in offdiag.E_TERMS],
                    [mirror["f_w_" + t[0]] for t in offdiag.E_TERMS],
                    [mirror[t[1]] for t in offdiag.E_TERMS],
                    inverse, slots, constitutive_coefficients(pml), codes, walls)
                worst = max(worst, sum(differing(mirror[n], oracle[n])
                                       for n in COMPARED))
                moved_total += assert_state_moved(before, oracle, COMPARED,
                                                  f"{name}/c{cycle}")
                # Keep the two states in lockstep for the next cycle.
                for key in mirror:
                    mirror[key][...] = oracle[key]
            row = {"case": name, "courant": courant, "cycles": cycles,
                   "bc": list(codes), "walls": list(walls),
                   "row_mask": [int(v is not None) for v in slots],
                   "differing_words": worst, "moved_words": moved_total,
                   "digest": state_digest({n: mirror[n] for n in COMPARED})}
            rows.append(row)
            log(f"[leg0 transcription] {name:<13} courant={courant} "
                f"cycles={cycles} differing={worst} moved={moved_total} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["transcription"] = rows
            save(payload, out)
            assert worst == 0, (
                f"{name}/courant={courant}: the in-gate transcription differs from "
                f"stepping.py by {worst} words. Every mutation measurement below "
                f"would be a statement about this harness, not about the kernel")
    # --- AND THE OTHER DIRECTION: is the comparison SENSITIVE? -----------------
    # A transcription that agrees with stepping.py proves the harness reproduces the
    # array path; it does not prove the COMPARISON would notice if it stopped. So
    # each host-side defect flag is planted in the reference and required to move
    # bytes. Without this, leg 0's zero could equally mean "the comparison is
    # blind", and every mutation measurement downstream inherits that doubt.
    sensitivity: List[Dict[str, Any]] = []
    grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, SEED)
    codes = boundary_codes(grid, pml)
    walls = offdiag.wall_mask_axes(grid)
    slots = rows_as_slots(installed_rows(fields))
    inverse = [fields.inverse_epsilon_for(n) for n in ("Ex", "Ey", "Ez")]
    base = snapshot(fields)
    stepping.update_E(fields, pml)
    oracle = snapshot(fields)
    for flag, why in (
            ("hoist", "the coefficient hoisted out from between the two shifts"),
            ("same_direction", "both shifts taken the same way — the half-cell "
                               "registration error"),
            ("drop_wall_mask", "the metallic wall-coupling mask dropped"),
            ("overapply_wall_mask", "the mask applied to the component's OWN axis"),
            ("store_before_load", "f_w written before prev is read")):
        broken = reference_state(base, pml, slots, inverse, codes, walls,
                                 **{flag: True})
        delta = sum(differing(broken[n], oracle[n]) for n in COMPARED)
        row = {"host_defect": flag, "why": why, "differing_words": delta}
        sensitivity.append(row)
        log(f"[leg0 sensitivity] {flag:<22} differing={delta} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["transcription_sensitivity"] = sensitivity
        save(payload, out)
        assert delta > 0, (
            f"planting {flag!r} in the in-gate reference moved NO bytes against "
            f"stepping.py. Leg 0's agreement would then be evidence that the "
            f"comparison is blind, not that the transcription is right")

    # The two RECORDED NULLS, on the host side, with the same derivations the
    # kernel-side mutation leg carries. Measured here as well because a null that
    # holds only in one implementation is a coincidence rather than a derivation.
    for flag, why in (
            ("distribute_quarter", "0.25 is an exact power of two, so scaling by "
                                   "it commutes with round-to-nearest"),
            ("commute_row_sum", "f32 addition is bitwise commutative")):
        null = reference_state(base, pml, slots, inverse, codes, walls,
                               **{flag: True})
        delta = sum(differing(null[n], oracle[n]) for n in COMPARED)
        row = predicted_null({"host_defect": flag, "differing_words": delta}, why)
        sensitivity.append(row)
        log(f"[leg0 sensitivity] {flag:<22} differing={delta} (NULL, expected 0) "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["transcription_sensitivity"] = sensitivity
        save(payload, out)
        assert delta == 0, (
            f"{flag!r} moved {delta} words on the HOST side but is a recorded "
            f"null: either the derivation is wrong or NumPy is not doing what it "
            f"is claimed to do here — both are findings, neither is a pass")

    payload["counts"]["transcription"] = len(rows) + len(sensitivity)
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG 0b — proof the bytes came off the GPU
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """That the bytes this gate compares came off the GPU, measured five ways.

    THE FALSE-PASS THIS CLOSES, and it was open on this family. Every other device
    leg here reads host arrays after a launch and compares them to ``stepping.py``.
    If the launch had silently not happened, or had happened on the host, those
    arrays would still be *some* numbers. The family carried only
    ``plan.launches == 1`` against that, and a launch count is the weakest of the
    available proofs: ``KernelPlan.run`` increments BEFORE it calls, so the counter
    is satisfied by an entry point that does nothing at all. The standard applied
    here is the one the BFAST tranche established, transplanted rather than
    reinvented, because a family-by-family standard is how a track ends up with a
    weakest link nobody named.

    1. RESIDENCY. Every mirror the ENGINE-route plan binds is reported with its
       ``torch`` device string, and every one must be ``mps``.
    2. THE ENTRY POINT'S IDENTITY. The object the launch path calls must be a
       ``torch._C._mps_MetalKernel`` — a compiled handle from
       ``torch.mps.compile_shader``. No Python callable in this tree can satisfy
       that, so "a host function stood in for the kernel" is excluded BY TYPE. This
       is the check that catches the strongest form of the fallback: a Python
       shim that forwards to the real kernel and therefore produces every correct
       number, which residency alone cannot distinguish.
    3. THE COUNTER counts LAUNCHES, not plans: 0 before, 1 after one run, 2 after
       two.
    4. THE NEGATIVE CONTROL, which is what makes 1-3 mean anything: the host arrays
       are compared to the oracle BEFORE the launch and must DIFFER, so "the plan
       was built and never run" cannot pass this gate's comparison.
    5. THE FALLBACK ITSELF, measured rather than argued away: the same shipped plan
       builder against a CPU-resident registry must RAISE. The exact message is
       recorded because it is a fact about this torch build, not a law.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in (CONFIGS[0], CONFIGS[1]):
        grid, fields, pml = build(cell, boundaries, 0.35, SEED,
                                  form="varying_full")
        oracle_grid, oracle_fields, oracle_pml = build(cell, boundaries, 0.35, SEED,
                                                       form="varying_full")
        stepping.update_E(oracle_fields, oracle_pml)
        after = snapshot(oracle_fields)

        residency = device.Residency()
        plan = offdiag.plan_offdiag_constitutive(fields, pml, residency)
        assert plan is not None, offdiag.offdiag_constitutive_coverage(
            fields, pml, residency).reasons

        devices = {mirror: str(residency.tensor(mirror).device)
                   for mirror in residency.names}
        host_resident = sorted(m for m, d in devices.items()
                               if not d.startswith("mps"))
        entry = plan._functions[shaders.CONTRACT_OFF]
        entry_type = f"{type(entry).__module__}.{type(entry).__name__}"

        assert plan.launches == 0, "a freshly built plan reports a launch"
        before_launch = snapshot(fields)
        unlaunched = sum(differing(before_launch[n], after[n]) for n in COMPARED)

        plan.run()
        residency.sync_out()
        launched_once = plan.launches
        got = snapshot(fields)
        bad = sum(differing(got[n], after[n]) for n in COMPARED)
        moved_words = sum(differing(got[n], before_launch[n]) for n in COMPARED)

        plan.run()
        residency.sync_out()
        launched_twice = plan.launches

        cpu_error: Optional[str] = None
        cpu_residency = device.Residency(device="cpu")
        cpu_grid, cpu_fields, cpu_pml = build(cell, boundaries, 0.35, SEED,
                                              form="varying_full")
        cpu_plan = offdiag.plan_offdiag_constitutive(cpu_fields, cpu_pml,
                                                     cpu_residency)
        assert cpu_plan is not None, (
            "the CPU-resident registry was refused at PLAN time, so this leg never "
            "reached the question it exists to ask")
        try:
            cpu_plan.run()
        except Exception as exc:  # noqa: BLE001 - the refusal is the measurement
            cpu_error = f"{type(exc).__name__}: {exc}"

        row = {"case": name, "mirror_count": len(devices),
               "mirror_devices": sorted(set(devices.values())),
               "host_resident_mirrors": host_resident,
               "entry_point_type": entry_type,
               "launches_after_one_run": launched_once,
               "launches_after_two_runs": launched_twice,
               "unlaunched_differing": unlaunched,
               "moved": moved_words,
               "cpu_binding_error": cpu_error,
               "differing_words": bad}
        rows.append(row)
        log(f"[leg0b execution] {name:<13} mirrors={len(devices)} "
            f"all_mps={not host_resident} entry={entry_type} "
            f"launches={launched_once}->{launched_twice} "
            f"unlaunched_differing={unlaunched} moved={moved_words} "
            f"cpu_binding={'REFUSED' if cpu_error else 'ACCEPTED (!!)'} "
            f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["execution"] = rows
        payload["counts"]["execution"] = len(rows)
        save(payload, out)

        assert devices, "VACUOUS: the plan bound no mirrors at all"
        assert not host_resident, (
            f"the plan bound HOST-resident mirrors {host_resident}: the bytes this "
            f"gate compares did not all come off the GPU")
        assert entry_type == "torch._C._mps_MetalKernel", (
            f"the launch path calls a {entry_type}, not a compiled Metal kernel "
            f"handle; the executor this gate certified is not Metal")
        assert launched_once == 1 and launched_twice == 2, (
            f"the launch counter reports {launched_once}/{launched_twice} for one "
            f"and two runs; it is not counting launches")
        assert unlaunched > 0, (
            f"VACUOUS negative control on {name}: the host arrays already matched "
            f"the oracle BEFORE the launch, so this leg cannot tell a kernel that "
            f"ran from one that did not")
        assert_moved(moved_words, f"execution/{name}")
        assert cpu_error is not None and "CPU tensor" in cpu_error, (
            f"a CPU-resident mirror bound to the Metal entry point did NOT raise "
            f"(got {cpu_error!r}): a silent host fallback is reachable on this "
            f"executor, and every device leg's residency must then be asserted per "
            f"launch rather than per plan")
        assert not bad, row


# ---------------------------------------------------------------------------
# LEG 1 — the shader vs stepping.py, real engine objects
# ---------------------------------------------------------------------------

def leg_reference(payload: Dict[str, Any], out: str) -> None:
    """The shipped kernel against ``stepping.py``, through the ENGINE route.

    Rows are installed with the PUBLIC installer, the plan is built by
    ``plan_offdiag_constitutive`` from the ``Fields``/``PML`` objects themselves,
    and every case carries both non-vacuity checks: the step MOVED STATE, and the
    oracle DIFFERS from the coupling-dropped diagonal control.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    compared = 0
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            for form in ("varying_full", "uniform_full", "single_ex_ey"):
                for aliased in (False, True):
                    grid, fields, pml = build(cell, boundaries, courant, SEED,
                                              form=form,
                                              aliased_inverse_epsilon=aliased)
                    before = snapshot(fields)
                    stepping.update_E(fields, pml)
                    after = snapshot(fields)
                    moved = assert_state_moved(before, after, COMPARED,
                                               f"{name}/{form}")
                    control = run_certified_constitutive(before, pml,
                                                         inverse_map(fields))
                    coupling = assert_differs_from_control(
                        after, control, COMPARED, f"{name}/{form}",
                        "coefficient-dropped diagonal")

                    restore(fields, before)
                    residency = device.Residency()
                    plan = offdiag.plan_offdiag_constitutive(fields, pml, residency)
                    assert plan is not None, offdiag.offdiag_constitutive_coverage(
                        fields, pml, residency).reasons
                    plan.run()
                    residency.sync_out()
                    assert plan.launches == 1
                    bad = sum(differing(getattr(fields, n), after[n])
                              for n in COMPARED)
                    stale = residency.verify()
                    compared += 1
                    row = {"case": name, "courant": courant, "row_form": form,
                           "aliased_inverse_epsilon": aliased,
                           "row_mask": list(plan.row_mask),
                           "bc": list(plan.boundary_codes),
                           "walls": list(plan.wall_axes),
                           "differing_words": bad, "moved_words": moved,
                           "coupling_delta_words": coupling,
                           "stale_mirrors": stale}
                    rows.append(row)
                    log(f"[leg1 reference] {name:<13} courant={courant} "
                        f"{form:<14} aliased={int(aliased)} rows={plan.row_mask} "
                        f"differing={bad} moved={moved} coupling={coupling} "
                        f"({time.time() - started:.1f}s)")
                    payload["legs"]["reference"] = rows
                    save(payload, out)
                    assert bad == 0, row
                    assert not stale, (
                        f"{name}: mirrors {stale!r} disagree with the host after "
                        f"the sync; a stale mirror is a smooth, plausible, wrong "
                        f"field rather than an error")
    payload["counts"]["reference"] = compared


# ---------------------------------------------------------------------------
# LEG 2 — the sweep
# ---------------------------------------------------------------------------

#: Shapes: the ordinary one, a NON-POWER-OF-TWO one, and a REDUCED axis. On an
#: n = 1 axis the wrap returns the same plane and the partner pair is ``2*g`` —
#: NOT the curl's exact zero — matching MEEP's stride(d)=0 double-read, and that
#: is the case a shape sweep exists to pin.
SWEEP_CELLS: Tuple[Tuple[str, Tuple[float, float, float], float], ...] = (
    ("cube", (1.2, 1.0, 0.9), 10.0),
    ("odd", (1.3, 1.7, 1.1), 10.0),
    ("reduced_x", (0.1, 1.0, 0.9), 10.0),
)

#: Amplitude classes. The CANCELLATION class puts ``gs*us`` and the coupling in
#: near-exact opposition, which is the only regime where the row-sum association
#: is byte-visible in f32; without it a whole class of grouping defects is
#: unobservable and the gate would not know.
#:
#: NAMES ONLY, AND THAT IS THE HONEST SHAPE. This was a (name, factor) pair whose
#: factor was bound to ``_factor`` and never applied to anything, so the artifact's
#: ``amplitude`` column read like a state-scale sweep that did not exist. The
#: cancellation class is produced entirely by :func:`_cancelling_inverse_epsilon`;
#: the state scale is swept by leg 5 and by the blindness control, which are where
#: a scale factor belongs.
#:
#: AND THE CLASS ITSELF WAS VACUOUS UNTIL 2026-08-15. Its first shape solved
#: ``u = -(gs*us)/g_partner`` on the reading that the four-point transverse sum is
#: ``4*g_partner[i]`` — true for a smooth field, FALSE for the random seeding
#: :func:`build` uses, where the three other samples are independent draws.
#: Measured before the fix: the median ``|src| / |gs*us|`` was 1.000000 on the
#: cancellation arm and 1.000000 on the normal arm, with 2.6-3.3% of cells below
#: 0.1 — no cancellation at all, on 49 of the sweep's 98 rows. It also collapsed
#: the row mask from (1,1,1,1,1,1) to (1,0,1,0,1,0), so half the sweep quietly ran
#: three single-row components. Both are fixed below, and
#: :func:`_cancellation_class` now MEASURES what was achieved so the label cannot
#: drift away from the class again.
AMPLITUDES: Tuple[str, ...] = ("normal", "cancellation")


def _row_totals(state: Dict[str, Any], slots: Sequence[Optional[Any]],
                codes: Sequence[int], walls: Sequence[int]) -> Dict[str, Any]:
    """The coupling total per component, from the transcription LEG 0 pins.

    The same expression tree ``reference_offdiag`` walks, wall mask included, but
    returning ``total`` before the row sum instead of stepping the state — which is
    what :func:`_cancelling_inverse_epsilon` needs to solve against. ``None`` for a
    component with no surviving row.
    """
    totals: Dict[str, Any] = {}
    for component, (name, _source, own_axis) in enumerate(offdiag.E_TERMS):
        total = None
        for offset in (0, 1):
            coefficient = slots[2 * component + offset]
            if coefficient is None:
                continue
            partner_axis = offdiag.TRANSVERSE_PARTNERS[component][offset]
            values = state[offdiag.E_TERMS[partner_axis][1]]
            pair = values + _shifted(values, partner_axis, False,
                                     codes[partner_axis])
            product = pair * coefficient
            term = 0.25 * (product
                           + _shifted(product, own_axis, True, codes[own_axis]))
            total = term if total is None else total + term
        if total is not None:
            for axis in offdiag.WALL_MASK_AXES[component]:
                if walls[axis]:
                    index: List[Any] = [slice(None)] * 3
                    index[axis] = 0
                    total[tuple(index)] = 0
        totals[name] = total
    return totals


def _cancelling_inverse_epsilon(state: Dict[str, Any], fields,
                                slots: Sequence[Optional[Any]],
                                codes: Sequence[int],
                                walls: Sequence[int]) -> Dict[str, Any]:
    """An inverse epsilon that makes ``gs*us`` cancel the coupling POINTWISE.

    SOLVE ON THE DIAGONAL SIDE, NOT THE COEFFICIENT SIDE — that is the whole
    correction. ``total`` is a STENCIL in the coefficient (``u[i]`` multiplies the
    near pair and ``u[i+s]`` the far one), so no pointwise choice of ``u`` puts the
    row sum anywhere in particular. ``us`` is a plain pointwise factor, so
    ``us = -total/gs`` makes ``gs*us + total`` cancel to the last bits, cell by
    cell, with the rows left exactly as installed — which keeps all six live
    instead of collapsing them to three.

    ``set_epsilon_volumes`` stores the inverse map with no normalization
    (fields.py:1251-1252), so an unphysical inverse epsilon is installable, which
    is what makes the class constructible at all. It IS unphysical — |us| reaches
    about 6 against a physical 0.6 — and that is stated rather than hidden: the
    class exists to make a FLOATING-POINT regime reachable, not to model a
    material.

    Cells where the solve is unusable keep the physical value: ``|gs|`` too small
    to divide by, a non-finite quotient, or a magnitude past 1e3. Those are the
    minority the floor accounts for — measured 0.83-0.92 of cells below a 1e-6
    residual ratio on the workhorse case.
    """
    totals = _row_totals(state, slots, codes, walls)
    out: Dict[str, Any] = {}
    for _component, (name, source, _axis) in enumerate(offdiag.E_TERMS):
        base = np.ascontiguousarray(fields.inverse_epsilon_for(name),
                                    dtype=np.float32)
        total = totals[name]
        if total is None:
            out[name] = base
            continue
        gs = state[source]
        with np.errstate(divide="ignore", invalid="ignore"):
            solved = (-total / gs).astype(np.float32)
        usable = ((np.abs(gs) > np.float32(1e-3)) & np.isfinite(solved)
                  & (np.abs(solved) < np.float32(1e3)))
        out[name] = np.ascontiguousarray(np.where(usable, solved, base),
                                         dtype=np.float32)
    return out


def _cancellation_class(state: Dict[str, Any], after: Dict[str, Any],
                        inverse: Dict[str, Any],
                        slots: Sequence[Optional[Any]]) -> Dict[str, Any]:
    """How much cancellation the case ACTUALLY achieved. Measured, never assumed.

    ``f_w`` holds the constitutive product this sub-step formed, so
    ``|f_w| / |gs*us|`` is the residual after the row sum — 1 where the coupling is
    a perturbation and 0 where it cancelled the diagonal product. Reported per
    coupled component and reduced with ``max``, because a form with one surviving
    row cancels one component and leaves the other two alone.
    """
    fractions: List[float] = []
    for component, (name, source, _axis) in enumerate(offdiag.E_TERMS):
        if slots[2 * component] is None and slots[2 * component + 1] is None:
            continue
        gs = np.asarray(state[source], dtype=np.float64)
        us = np.asarray(inverse[name], dtype=np.float64)
        diagonal = np.abs(gs * us)
        constitutive = np.abs(np.asarray(after["f_w_" + name], dtype=np.float64))
        live = diagonal > 0
        if not np.any(live):
            continue
        fractions.append(float(np.mean(constitutive[live] / diagonal[live] < 1e-6)))
    return {"coupled_components": len(fractions),
            "frac_cancelled_below_1e-6": max(fractions) if fractions else 0.0}


def _row_value_class(rows_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """What value class the INSTALLED coefficients actually carry.

    Measured from the rows the installer kept, never from the form name, and
    measured AFTER the cancellation replacement so the row reports what the case
    really ran on. Two numbers, and the leg floors both across the sweep:

    * ``negative_zero_words`` — how many coefficient words are exactly ``-0.0``.
      The installer drops only an IDENTICALLY zero row, so a lattice of signed
      zeros inside a live row survives; a count of zero across the whole sweep
      would mean the signed-zero coefficient class was named and never built.
    * ``exponent_span`` — the number of distinct binary exponents among the
      nonzero coefficients, which is what makes ``wide_exponent`` a class rather
      than a label.
    """
    negative = 0
    exponents: set = set()
    for partners in rows_map.values():
        for volume in partners.values():
            flat = np.ascontiguousarray(volume, dtype=np.float32).reshape(-1)
            negative += int(np.count_nonzero(flat.view(np.uint32)
                                             == np.uint32(0x80000000)))
            live = flat[flat != 0]
            if live.size:
                exponents.update(np.unique(
                    np.frexp(live.astype(np.float64))[1]).tolist())
    return {"negative_zero_words": negative, "exponent_span": len(exponents)}


def leg_synthetic(payload: Dict[str, Any], out: str) -> None:
    """Shapes x boundaries x row forms x inverse-epsilon forms x amplitudes x guard.

    The GUARD arm is the contraction one: ``contract="fast"`` removes the file-scope
    directive, and the gate records whether the bytes moved. It is recorded as an
    OUTCOME rather than asserted in one direction, because whether contraction is
    observable depends on the expression tree — but a run where it is observable
    nowhere would mean the guard is doing nothing, so the leg asserts it fired at
    least once across the sweep.

    THE ROW VALUE CLASS IS AN AXIS HERE AND IS FLOORED. Every other form draws the
    coefficients from one narrow normal distribution, so before ``signed_zero_rows``
    and ``wide_exponent`` this sweep exercised a single exponent decade on the one
    operand this family adds over the certified constitutive kernel, and never a
    signed zero on it at all. :func:`_row_value_class` measures what was installed
    and the leg refuses to pass if either class stayed empty — a class that is named
    and never constructed is worse than one that is absent, because the artifact
    claims it.
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    started = time.time()
    guard_observable = 0
    compared = 0
    worst_negative_zeros = 0
    worst_exponent_span = 0
    best_cancellation = 0.0
    best_normal_cancellation = 0.0

    def skip(shape_name: str, config_name: str, form: str, amplitude: str,
             why: str) -> None:
        """A SKIP WITH ITS REASON. Two of these used to be bare ``continue``s.

        The engine's METALLIC-one-cell refusal was always recorded; the
        cancellation-install failure and the no-surviving-row case were not, and a
        silently dropped case is indistinguishable from one that ran. If the
        cancellation install had failed everywhere, all 49 of its rows would have
        vanished and the artifact would still have listed the amplitude axis.
        """
        skipped.append({"shape": shape_name, "case": config_name,
                        "row_form": form, "amplitude": amplitude,
                        "skipped_because": why})
        payload["legs"]["synthetic_skipped"] = skipped
        save(payload, out)
        log(f"[leg2 synthetic] SKIP {shape_name}/{config_name}/{form}/"
            f"{amplitude}: {why}")

    for shape_name, cell, resolution in SWEEP_CELLS:
        for config_name, _cell, boundaries in CONFIGS:
            for form in ROW_FORMS:
                for amplitude in AMPLITUDES:
                    courant = COURANTS[compared % len(COURANTS)]
                    try:
                        grid, fields, pml = build(
                            cell, boundaries, courant, SEED + compared, form=form,
                            aliased_inverse_epsilon=(compared % 2 == 0),
                            resolution=resolution)
                    except ValueError as exc:
                        # The engine refuses a METALLIC one-cell axis by name
                        # (grid.py:909): MEEP overrides a unit direction to
                        # periodic whatever the k_point says, so the combination
                        # this sweep would build has no MEEP counterpart to be
                        # identical to.
                        skip(shape_name, config_name, form, amplitude,
                             str(exc).split(".")[0])
                        continue
                    codes = boundary_codes(grid, pml)
                    walls = offdiag.wall_mask_axes(grid)
                    rows_map = installed_rows(fields)
                    if not any(value is not None
                               for value in rows_as_slots(rows_map)):
                        skip(shape_name, config_name, form, amplitude,
                             "no off-diagonal row survived installation, so this "
                             "configuration belongs to the certified plain kernel")
                        continue
                    if amplitude == "cancellation":
                        # THE ROWS ARE LEFT EXACTLY AS INSTALLED and the DIAGONAL
                        # side is solved instead — see _cancelling_inverse_epsilon
                        # for why the coefficient side cannot produce this class.
                        try:
                            fields.set_epsilon_volumes(
                                {c: fields._eps_components[c]
                                 for c in ("Ex", "Ey", "Ez")},
                                _cancelling_inverse_epsilon(
                                    snapshot(fields), fields,
                                    rows_as_slots(rows_map), codes, walls),
                                rows_map)
                        except ValueError as exc:
                            skip(shape_name, config_name, form, amplitude,
                                 f"the cancelling inverse epsilon was refused at "
                                 f"install: {exc}")
                            continue
                        rows_map = installed_rows(fields)
                    value_class = _row_value_class(rows_map)
                    worst_negative_zeros = max(worst_negative_zeros,
                                               value_class["negative_zero_words"])
                    worst_exponent_span = max(worst_exponent_span,
                                              value_class["exponent_span"])
                    inverse = inverse_map(fields)

                    before = snapshot(fields)
                    stepping.update_E(fields, pml)
                    after = snapshot(fields)
                    label = f"{shape_name}/{config_name}/{form}/{amplitude}"
                    moved = assert_state_moved(before, after, COMPARED, label)
                    control = run_certified_constitutive(before, pml, inverse)
                    coupling = assert_differs_from_control(
                        after, control, COMPARED, label,
                        "coefficient-dropped diagonal")

                    got = run_on_device(before, pml, rows_map, inverse, codes, walls)
                    bad = sum(differing(got[n], after[n]) for n in COMPARED)

                    fast = run_on_device(before, pml, rows_map, inverse, codes,
                                         walls, contract=shaders.CONTRACT_FAST)
                    fast_delta = sum(differing(fast[n], after[n]) for n in COMPARED)
                    guard_observable += int(fast_delta > 0)

                    cancellation = _cancellation_class(before, after, inverse,
                                                       rows_as_slots(rows_map))
                    achieved = cancellation["frac_cancelled_below_1e-6"]
                    if amplitude == "cancellation":
                        best_cancellation = max(best_cancellation, achieved)
                    else:
                        best_normal_cancellation = max(best_normal_cancellation,
                                                       achieved)

                    compared += 1
                    row = {"shape": shape_name, "grid": list(grid.shape),
                           "case": config_name, "row_form": form,
                           "amplitude": amplitude, "courant": courant,
                           "bc": list(codes), "walls": list(walls),
                           "differing_words": bad, "moved_words": moved,
                           "coupling_delta_words": coupling,
                           "contract_fast_delta_words": fast_delta,
                           "row_value_class": value_class,
                           "cancellation_class": cancellation}
                    rows.append(row)
                    log(f"[leg2 synthetic] {label:<44} n={tuple(grid.shape)} "
                        f"differing={bad} moved={moved} coupling={coupling} "
                        f"fast_delta={fast_delta} "
                        f"rows(-0={value_class['negative_zero_words']},"
                        f"exps={value_class['exponent_span']}) "
                        f"cancelled={achieved:.3f} "
                        f"({time.time() - started:.1f}s)")
                    payload["legs"]["synthetic"] = rows
                    save(payload, out)
                    assert bad == 0, row
    payload["counts"]["synthetic"] = compared
    payload["legs"]["synthetic_skipped"] = skipped
    payload["legs"]["synthetic_guard_observable_cases"] = guard_observable
    payload["legs"]["synthetic_row_value_class"] = {
        "worst_negative_zero_words": worst_negative_zeros,
        "worst_exponent_span": worst_exponent_span}
    payload["legs"]["synthetic_cancellation_class"] = {
        "best_frac_cancelled_cancellation_arm": best_cancellation,
        "best_frac_cancelled_normal_arm": best_normal_cancellation,
        "measured": ("|f_w| / |gs*us| below 1e-6, per coupled component, reduced "
                     "with max. The pair is what makes the class DISCRIMINATING "
                     "rather than merely present: the normal arm must not reach "
                     "it")}
    save(payload, out)
    assert compared > 0, "the sweep ran no cases"
    # THE VALUE-CLASS FLOORS. A named class that was never constructed makes the
    # artifact claim coverage it does not have, which is worse than not sweeping
    # the class at all.
    assert_census_floor(worst_negative_zeros,
                        "signed-zero ROW COEFFICIENTS across the sweep", floor=8)
    assert worst_exponent_span >= 20, (
        f"the widest installed coefficient exponent span was "
        f"{worst_exponent_span} binary exponents. The wide_exponent form exists to "
        f"put the row product and the diagonal product decades apart inside one "
        f"grid, and a span this narrow means it collapsed to the ordinary class")
    assert guard_observable > 0, (
        "the contraction guard changed NO bytes anywhere in the sweep. Either the "
        "variant selector is not selecting or this family's expression tree offers "
        "the compiler nothing to contract — both are findings, and a gate that "
        "recorded neither would be certifying the pragma's presence rather than "
        "its effect")
    # THE CANCELLATION FLOOR, and its DISCRIMINATION control. The first shape of
    # this class was measurably inert (median |src|/|gs*us| = 1.000000 on both
    # arms), which is exactly the failure a named-but-unmeasured class produces:
    # the artifact claimed a regime the sweep never entered. Both directions are
    # asserted, because a floor alone would also pass if EVERY case cancelled — in
    # which case "cancellation" would name the sweep, not a class within it.
    assert best_cancellation >= 0.5, (
        f"the cancellation arm reached only {best_cancellation:.3f} of cells with "
        f"|f_w| under 1e-6 of |gs*us|. The class exists to make the row-sum "
        f"ASSOCIATION byte-visible in f32, which needs the diagonal product and "
        f"the coupling in near-exact opposition; a sweep that labels the class "
        f"without entering it claims coverage it does not have")
    assert best_normal_cancellation < 0.05, (
        f"the NORMAL arm reached {best_normal_cancellation:.3f} cancelled cells, "
        f"so the two amplitude classes are not distinguishable and the axis is a "
        f"label rather than a variable")


# ---------------------------------------------------------------------------
# LEG 3 — the compiled seam
# ---------------------------------------------------------------------------

def leg_seam(payload: Dict[str, Any], out: str) -> None:
    """Partial rows: the UNCOUPLED components must match the CERTIFIED plain kernel.

    Two halves and the second is what makes it non-vacuous:

    1. every component with NO surviving row is byte-identical to the certified
       ``constitutive_step`` E arm — that is docstring point 6, and it is what
       licenses the disjointness seam with ``constitutive_coverage(side='E')``;
    2. the COUPLED component DIFFERS from it. Without that the leg would pass just
       as well on a kernel whose coupling was dead everywhere.

    The rows-all-dead build is not emitted at all: :func:`offdiag.offdiag_source`
    REFUSES it by name, because that configuration is the certified plain kernel's
    and emitting a second kernel for it would overlap the two families. The refusal
    is asserted here rather than worked around.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    refused = False
    try:
        offdiag.offdiag_source((0, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    except ValueError:
        refused = True
    assert refused, (
        "an all-dead row mask emitted a source. That configuration is the certified "
        "plain kernel's, and a second kernel for it would make the two families "
        "overlap on exactly the runs the install-time zero-row drop separates")

    partial: Tuple[Tuple[str, Dict[str, Tuple[str, ...]]], ...] = (
        ("only_ex", {"Ex": ("Ey",)}),
        ("only_ez", {"Ez": ("Ex", "Ey")}),
        ("ex_and_ey", {"Ex": ("Ez",), "Ey": ("Ex",)}),
    )
    # ALL EIGHT boundary triples: an uncoupled component must reduce to the
    # certified kernel on an asymmetric wall set as much as on a symmetric one,
    # and the four asymmetric triples are exactly where a per-component wall-mask
    # axis error survives.
    for name, cell, boundaries in CONFIGS:
        for label, live in partial:
            grid, fields, pml = build(cell, boundaries, 0.35, SEED)
            shape = grid.shape
            rng = np.random.default_rng(SEED + 3)
            full = make_rows("varying_full", shape, rng)
            trimmed = {row: {p: full[row][p] for p in partners}
                       for row, partners in live.items()}
            fields.set_epsilon_volumes(
                {c: fields._eps_components[c] for c in ("Ex", "Ey", "Ez")},
                {c: fields._inv_eps_components[c] for c in ("Ex", "Ey", "Ez")},
                trimmed)
            codes = boundary_codes(grid, pml)
            walls = offdiag.wall_mask_axes(grid)
            inverse = inverse_map(fields)
            before = snapshot(fields)
            stepping.update_E(fields, pml)
            after = snapshot(fields)
            assert_state_moved(before, after, COMPARED, f"{name}/{label}")

            got = run_on_device(before, pml, installed_rows(fields), inverse,
                                codes, walls)
            plain = run_certified_constitutive(before, pml, inverse)
            bad = sum(differing(got[n], after[n]) for n in COMPARED)

            coupled = set(live)
            uncoupled = [t[0] for t in offdiag.E_TERMS if t[0] not in coupled]
            uncoupled_delta = sum(
                differing(got[n], plain[n]) + differing(got["f_w_" + n],
                                                        plain["f_w_" + n])
                for n in uncoupled)
            coupled_delta = sum(
                differing(got[n], plain[n]) + differing(got["f_w_" + n],
                                                        plain["f_w_" + n])
                for n in sorted(coupled))
            row = {"case": name, "partial": label,
                   "coupled": sorted(coupled), "uncoupled": uncoupled,
                   "differing_words": bad,
                   "uncoupled_vs_certified_words": uncoupled_delta,
                   "coupled_vs_certified_words": coupled_delta}
            rows.append(row)
            log(f"[leg3 seam] {name:<13} {label:<11} differing={bad} "
                f"uncoupled_delta={uncoupled_delta} coupled_delta={coupled_delta} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["seam"] = rows
            save(payload, out)
            assert bad == 0, row
            assert uncoupled_delta == 0, (
                f"{name}/{label}: an UNCOUPLED component differs from the certified "
                f"plain kernel by {uncoupled_delta} words. The 'no surviving row' "
                f"arm is supposed to be that kernel's body verbatim, and the "
                f"disjointness seam rests on it")
            assert coupled_delta > 0, (
                f"{name}/{label}: the COUPLED components are byte-identical to the "
                f"plain kernel, so this case would pass with the coupling dark")
    payload["counts"]["seam"] = len(rows)
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG 4 — signed zeros
# ---------------------------------------------------------------------------

def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """A +-0 lattice through the whole state, with a census FLOOR per seeding.

    ZERO-INIT IS A FIXED POINT of this sub-step — every product is zero, the row
    sum is zero, ``f_w`` takes zero and the recurrence adds zero — so a leg that
    seeded zeros and compared zeros would agree with itself and certify nothing.
    The seeding is a LATTICE inside a physically-seeded state, and the floor is what
    makes the class real: a census of zero is VACUOUS, not passed.

    The signed-zero OUTPUT class is where the interest is. Metal's flushed-zero
    SIGN is per-op (measured: ``x + zero`` loses the sign of a flushed negative
    subnormal while ``x * two`` keeps it), and the wall mask writes exact zeros
    into the coupling — so this leg is where a sign convention difference would
    show up if there were one.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS[1:3]:
        for form in ("varying_full", "uniform_full"):
            grid, fields, pml = build(cell, boundaries, 0.35, SEED, form=form,
                                      signed_zeros=True)
            codes = boundary_codes(grid, pml)
            walls = offdiag.wall_mask_axes(grid)
            inverse = inverse_map(fields)
            rows_map = installed_rows(fields)
            before = snapshot(fields)

            # A census of ZERO is VACUOUS, not passed: the floor is what makes the
            # +-0 lattice a real class rather than a label on an empty one, and it
            # is checked PER SEEDING because zero-init is a fixed point of this
            # sub-step and would have moved nothing at all.
            #
            # THE D VOLUMES ARE FLOORED TOO, and they were not. The floor used to
            # cover ``COMPARED`` alone — E and f_w, which enter this sub-step only
            # as ``f`` and ``prev`` in the element-wise PML tail. The volumes the
            # COUPLING STENCIL READS are Dx/Dy/Dz, at the home cell, at the
            # partner-axis neighbour, at the own-axis neighbour and at the corner;
            # a signed-zero handling difference in the row product could only show
            # up through them. Flooring only the outputs left this leg certifying
            # the -0 OPERAND class for the one part of the sub-step that does not
            # have a stencil.
            seeded_names = ("Dx", "Dy", "Dz") + COMPARED
            seeded = {n: preconditions.signed_zero_census(before[n])
                      for n in seeded_names}
            for n in seeded_names:
                assert_census_floor(seeded[n]["negative_zero"],
                                    f"{name}/{form} seeded -0 in {n}", floor=8)

            stepping.update_E(fields, pml)
            after = snapshot(fields)
            assert_state_moved(before, after, COMPARED, f"{name}/{form}")
            produced = {n: preconditions.signed_zero_census(after[n])
                        for n in COMPARED}

            got = run_on_device(before, pml, rows_map, inverse, codes, walls)
            bad = sum(differing(got[n], after[n]) for n in COMPARED)
            out_negative = sum(produced[n]["negative_zero"] for n in COMPARED)
            row = {"case": name, "row_form": form, "differing_words": bad,
                   "seeded_negative_zeros": {k: v["negative_zero"]
                                             for k, v in seeded.items()},
                   "output_negative_zeros": out_negative,
                   "output_class_note": (
                       "MEASURED, and recorded rather than asserted. The INPUT "
                       "class is floored and met; the OUTPUT -0 count is zero for "
                       "this family and that is a derivation rather than a gap. "
                       "The certified CURL produces exact -0 outputs because its "
                       "ownership mask writes zeros that then run through the "
                       "split-field recurrence; this sub-step's wall mask writes "
                       "+0.0 into the coupling and the PML tail then adds "
                       "kps*src and subtracts kms*prev, so an exact -0 survives to "
                       "a stored output only if both terms vanish with the right "
                       "signs, which the physical seeding does not produce. The "
                       "leg therefore certifies that seeded -0 OPERANDS are "
                       "handled identically, which is the class it can construct, "
                       "and says so instead of reporting a zero census as a pass")}
            rows.append(row)
            log(f"[leg4 signed_zero] {name:<13} {form:<14} differing={bad} "
                f"out_negative_zeros={out_negative} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            payload["counts"]["signed_zero"] = len(rows)
            save(payload, out)
            assert bad == 0, row


# ---------------------------------------------------------------------------
# LEG 5 — the subnormal precondition
# ---------------------------------------------------------------------------

#: (label, state scale, must the census stay clean)
SUBNORMAL_SCALES: Tuple[Tuple[str, float, bool], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-20, True),
    ("subnormal_band", 1e-38, False),
)


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The census that BOUNDS the claim, and its own non-vacuity proof.

    Operands, RESULTS and INTERMEDIATES, censused separately through
    :class:`preconditions.SubnormalWindow`. The intermediates matter here and are
    reconstructed on the host to be censused at all: the row product forms
    ``pair``, ``product`` and ``term`` inside the kernel, and a value can be in the
    band there while every operand and every result is an ordinary number.

    THE REACHABILITY QUESTION FOR THIS FAMILY, answered rather than inherited: the
    coupling multiplies a FIELD by a COEFFICIENT, never a field by a field, so it is
    in the same magnitude class as the certified constitutive product and there is
    no ``E^3``-style cube to drive an intermediate three decades below its
    operands. That is why this family does NOT invoke
    :func:`preconditions.nonlinear_intermediate_reasons` — the plan-time bound the
    chi2/chi3 family needs — and the census below is what turns that reading into a
    measurement.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, scale, expect_clean in SUBNORMAL_SCALES:
        grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, SEED,
                                  form="varying_full", scale=scale)
        codes = boundary_codes(grid, pml)
        walls = offdiag.wall_mask_axes(grid)
        inverse = inverse_map(fields)
        rows_map = installed_rows(fields)
        slots = rows_as_slots(rows_map)
        before = snapshot(fields)
        stepping.update_E(fields, pml)
        after = snapshot(fields)

        # THE FLOORS ARE THIS FAMILY'S, declared rather than defaulted. Every
        # censused volume must contribute at least one grid's worth of words, and
        # the INTERMEDIATE floor is non-zero because this family HAS intermediates
        # that never reach a stored array — `pair`, `product` and `term`. A window
        # that censused only what was stored would report clean without ever having
        # looked at the values the row product actually forms.
        window = preconditions.SubnormalWindow(
            first_step=0, last_step=0,
            per_array_words=int(np.prod(grid.shape)),
            per_intermediate_words=int(np.prod(grid.shape)))
        for name in ("Dx", "Dy", "Dz") + COMPARED:
            window.observe(f"operand:{name}", before[name], step=0)
        for name in COMPARED:
            window.observe(f"result:{name}", after[name], step=0)
        # EVERY intermediate the kernel forms, not the three that were easy to
        # name. The first shape of this leg censused `pair`, `product` and `term`
        # and called that "INTERMEDIATES" — while the shader also forms `gs*us`,
        # the row sum `total`, and the two tail products `kps*src` and `kms*prev`,
        # none of which was looked at. `total` is the one that can reach the band
        # while both of its addends are ordinary numbers, because it is a SUM and
        # the sweep deliberately builds a cancelling regime; measured at physical
        # scale here, |total| bottoms at 1.0e-06 with a zero census, so this is a
        # claim brought back in line with its measurement rather than a divergence
        # repaired.
        totals = _row_totals(before, slots, codes, walls)
        for component, (name, source, own_axis) in enumerate(offdiag.E_TERMS):
            for offset in (0, 1):
                coefficient = slots[2 * component + offset]
                if coefficient is None:
                    continue
                partner_axis = offdiag.TRANSVERSE_PARTNERS[component][offset]
                values = before[offdiag.E_TERMS[partner_axis][1]]
                # RECONSTRUCTED ON THE HOST, same expression and same operand
                # order as the shader. Weaker than reading the device's registers,
                # which is not possible here, and strictly stronger than censusing
                # only the stored arrays.
                pair = values + _shifted(values, partner_axis, False,
                                         codes[partner_axis])
                product = pair * coefficient
                term = 0.25 * (product
                               + _shifted(product, own_axis, True, codes[own_axis]))
                tag = f"{name}:{offset}"
                window.observe_intermediate(f"pair:{tag}", pair, step=0)
                window.observe_intermediate(f"product:{tag}", product, step=0)
                window.observe_intermediate(f"term:{tag}", term, step=0)
            diagonal = before[source] * inverse[name]
            window.observe_intermediate(f"diagonal:{name}", diagonal, step=0)
            if totals[name] is not None:
                window.observe_intermediate(f"total:{name}", totals[name], step=0)
            axis_name = "xyz"[own_axis]
            coefficients = constitutive_coefficients(pml)
            window.observe_intermediate(
                f"kps_times_src:{name}",
                coefficients["kps_" + axis_name] * after["f_w_" + name], step=0)
            window.observe_intermediate(
                f"kms_times_prev:{name}",
                coefficients["kms_" + axis_name] * before["f_w_" + name], step=0)

        got = run_on_device(before, pml, rows_map, inverse, codes, walls)
        bad = sum(differing(got[n], after[n]) for n in COMPARED)
        report = window.report()
        row = {"scale": label, "factor": scale, "differing_words": bad,
               "census": report,
               "gate_verdict": ("REFUSED (precondition)" if not report["clean"]
                                else ("identical" if not bad else "DIFFERS"))}
        rows.append(row)
        log(f"[leg5 precondition] {label:<14} subnormal_words="
            f"{report['subnormal_words']}/{report['observed_words']} "
            f"clean={report['clean']} vacuous={report['vacuous']} "
            f"differing={bad} ({time.time() - started:.1f}s)")
        payload["legs"]["precondition"] = rows
        save(payload, out)
        if expect_clean:
            preconditions.assert_clean_or_refuse(window, f"offdiag/{label}")
            assert bad == 0, row
        else:
            preconditions.demonstrate_firing(window, f"offdiag/{label}")
            assert bad > 0, (
                f"{label}: the control was expected to DIVERGE and did not, so the "
                f"cliff this claim rests on was not reproduced on this family")

    payload["counts"]["precondition"] = len(rows)


# ---------------------------------------------------------------------------
# LEG 6 — mutations
# ---------------------------------------------------------------------------

def _mutations() -> Dict[str, Dict[str, Any]]:
    """The planted defects, in shipped-SOURCE terms.

    ``must_catch`` is three-valued. The two ``False`` entries are RECORDED NULLS
    with derivations, not omissions — see the module docstring and each ``why``.
    """
    return {
        "m1_hoist_coefficient_out_of_the_shifts": {
            "must_catch": True,
            "why": ("the ONE new arithmetic element of this family: u[i] must "
                    "multiply the near pair and u[i+s] the far pair. Hoisting u[i] "
                    "onto a four-point average is the same ALGEBRA only for a "
                    "uniform coefficient, and MEEP registers the entry at the "
                    "integer node (anisotropic_averaging.cpp:248-257)"),
            "apply": lambda s: needle(
                s,
                "    float term_00 = 0.25f * ((near_00 * unear_00)"
                " + (far_00 * ufar_00));",
                "    float term_00 = 0.25f * ((near_00 + far_00) * unear_00);"),
        },
        "m2_both_shifts_same_direction": {
            "must_catch": True,
            "why": ("the partner shift goes DOWN and the own-axis shift UP; taking "
                    "both the same way is the half-cell registration error"),
            "apply": lambda s: needle(s, "        + (dvy ? g1[i * nyz + dj * nzi + k]"
                                         " : 0.0f);",
                                     "        + (uvy ? g1[i * nyz + uj * nzi + k]"
                                     " : 0.0f);"),
        },
        "m3_mispair_coefficient_slots": {
            "must_catch": True,
            "why": ("the cycle order binds coefficients to partners (X->Y->Z, "
                    "stepping.py:1235-1237); swapping which buffer a term reads is "
                    "the defect the slot table exists to prevent"),
            "apply": lambda s: needle(s, "    float unear_00 = u01[ii];",
                                      "    float unear_00 = u02[ii];"),
        },
        "m4_far_coefficient_at_the_near_node": {
            "must_catch": True,
            "why": ("u[i+s] is what multiplies the far pair; reading u[i] there is "
                    "the hoist's half — a coefficient sampled half a cell off"),
            "apply": lambda s: needle(
                s, "    float ufar_00 = uvx ? u01[ui * nyz + j * nzi + k] : 0.0f;",
                "    float ufar_00 = u01[ii];"),
        },
        "m5_drop_the_corner_sample": {
            "must_catch": True,
            "why": ("the far pair is g[i+s] + g[(i+s)-sx]; dropping the corner "
                    "leaves a one-sided average that is smooth and wrong"),
            "apply": lambda s: needle(
                s,
                "        + ((uvx && dvy) ? g1[ui * nyz + dj * nzi + k] : 0.0f);",
                "        + 0.0f;"),
        },
        "m6_drop_the_wall_mask": {
            "must_catch": True,
            "why": ("the coupling reads LIVE partner volumes beside a metallic "
                    "wall whose tangential E is never stepped (measured 2.6e-02 "
                    "unmasked against a 2.0e-07 floor masked)"),
            "apply": lambda s: needle(s, "    total0 = at_y ? 0.0f : total0;",
                                      "    // wall mask dropped on component 0/y"),
            "codes": (METALLIC, METALLIC, PERIODIC),
        },
        "m7_overapply_the_wall_mask": {
            "must_catch": True,
            "why": ("only the axes where the component's Yee shift is 0 are masked; "
                    "masking its OWN axis zeroes a plane MEEP steps"),
            "apply": lambda s: needle(
                s, "    total0 = at_y ? 0.0f : total0;",
                "    total0 = at_y ? 0.0f : total0;\n"
                "    total0 = at_x ? 0.0f : total0;"),
            "codes": (METALLIC, METALLIC, PERIODIC),
        },
        "m8_store_before_load": {
            "must_catch": True,
            "why": ("`prev` must be read BEFORE f_w is written (stepping.py:2137); "
                    "reversed it is wrong only where kms != 0, i.e. INSIDE THE PML "
                    "ONLY, and looks like a slightly worse absorber"),
            "apply": lambda s: needle(s, "    float prev0 = w0[ii];",
                                      "    w0[ii] = 0.0f;\n"
                                      "    float prev0 = w0[ii];"),
        },
        "m9_contract_on": {
            "must_catch": True,
            "why": ("removes the one compile option; the recurrence's "
                    "multiply-adds contract into fmas"),
            "apply": lambda s: needle(s, shaders.contraction_pragma("off"),
                                      shaders.contraction_pragma("fast")),
        },
        "m10_flatten_the_pml_accumulation": {
            "must_catch": True,
            "why": ("`((f + kps*src) - kms*prev)` flattened to "
                    "`f + (kps*src - kms*prev)` is a different float32 number"),
            "apply": lambda s: needle(
                s, "    a0 = a0 + kp_0 * src0;\n    a0 = a0 - km_0 * prev0;",
                "    a0 = a0 + (kp_0 * src0 - km_0 * prev0);"),
        },
        # --- THE INDEX AND AXIS ARMS, added 2026-08-15 ----------------------
        # Six assembled flat-index expressions and a per-component coefficient
        # index were held by a unit test's independent spelling and by nothing in
        # this gate. Task "arm the axis-mapping and index-decomposition mutations"
        # established these as required for the sibling constitutive family; this
        # one never got them. Every measured count below is from this host.
        "m14_index_decode_k_from_ny": {
            "must_catch": True,
            "why": ("the flat decode is (i,j,k) of a C-contiguous (nx,ny,nz); "
                    "taking k modulo ny reads a transposed volume. Measured 5,565 "
                    "differing words"),
            "apply": lambda s: needle(s, "    int k   = ii % nzi;",
                                      "    int k   = ii % nyi;"),
        },
        "m15_stride_table_nyz_from_nx": {
            "must_catch": True,
            "why": ("`_STRIDE`/`_index` assemble all six neighbour indices from "
                    "one axis stride; nyz built from nx makes every one of them a "
                    "different cell. Measured 5,706 differing words"),
            "apply": lambda s: needle(s, "    int nyz = nyi * nzi;",
                                      "    int nyz = nxi * nzi;"),
        },
        "m16_pml_coefficient_on_the_wrong_axis": {
            "must_catch": True,
            "why": ("the constitutive tail reads the coefficient at the "
                    "COMPONENT'S OWN axis (MEEP's dsigw, stepping.py:1015 via "
                    "E_CONSTITUTIVE_TERMS); indexing kps_x by j is the absorber "
                    "profile rotated a quarter turn. Measured 468 differing words"),
            "apply": lambda s: needle(
                s, "    float kp_0 = kp0[i], km_0 = km0[i];",
                "    float kp_0 = kp0[j], km_0 = km0[j];"),
        },
        "m17_wall_mask_on_the_high_face": {
            "must_catch": True,
            "why": ("`_mask_metallic_wall_coupling` zeroes FACE 0 only "
                    "(stepping.py:1283, `term[_face(axis, 0)] = 0`) — the low wall "
                    "is stored and the high wall is the shift-up zero ghost. "
                    "Measured 1,476 differing words"),
            "apply": lambda s: needle(
                s, "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
                "    bool at_x = (i == nxi-1), at_y = (j == nyi-1),"
                " at_z = (k == nzi-1);"),
        },
        "m18_periodic_wrap_clamped": {
            "must_catch": True,
            "why": ("a periodic near ghost WRAPS to the far plane "
                    "(stepping._shift_down:1818); clamping to row 0 duplicates the "
                    "boundary cell and is a plausible-looking Neumann condition. "
                    "Measured 436 differing words"),
            "apply": lambda s: needle(s, "    dk = (dk < 0) ? (nzi - 1) : dk;",
                                      "    dk = (dk < 0) ? 0 : dk;"),
        },
        "m19_prev_from_the_wrong_auxiliary": {
            "must_catch": True,
            "why": ("each component's tail subtracts kms times ITS OWN previous "
                    "f_w (stepping.py:2137-2143); crossing the three auxiliaries "
                    "mixes polarizations inside the absorber. Measured 1,080 "
                    "differing words"),
            "apply": lambda s: needle(s, "    float prev0 = w0[ii];",
                                      "    float prev0 = w1[ii];"),
        },
        "m20_null_partner_axis_metallic_near_ghost_wrapped": {
            "must_catch": False,
            "why": ("RECORDED NULL, AND NOW MEASURED — it used to be recorded as "
                    "'NOT MEASURED BY MUTATION'. The partner-axis metallic NEAR "
                    "ghost's entire support is the partner-axis face-0 plane, and "
                    "the wall-coupling mask zeroes the whole total on that plane "
                    "BEFORE the row sum (a metallic partner axis is always a "
                    "Yee-shift-0 transverse axis of the component, so the mask "
                    "covers exactly it). Wrapping the ghost instead of zeroing it "
                    "therefore moves 0 words. Its sensitivity control is m21: the "
                    "SAME change on the own-axis far ghost, which the mask does "
                    "NOT cover, moves 162"),
            "apply": lambda s: needle(
                s, "    dvy = (dj >= 0);\n    uvy = (uj < nyi);",
                "    dj = (dj < 0) ? (nyi - 1) : dj;\n    uvy = (uj < nyi);"),
        },
        "m21_own_axis_metallic_far_ghost_wrapped": {
            "must_catch": True,
            "why": ("m20's SENSITIVITY CONTROL. The own-axis far ghost is the one "
                    "the wall mask does not cover — the mask is face 0 and this is "
                    "the top plane — so the same wrap-instead-of-zero is visible "
                    "here. Without this pair m20's zero would be evidence that the "
                    "harness is blind to a ghost change rather than that the mask "
                    "hides one. Measured 162 differing words"),
            "apply": lambda s: needle(
                s, "    dvx = (di >= 0);\n    uvx = (ui < nxi);",
                "    dvx = (di >= 0);\n    ui = (ui == nxi) ? 0 : ui;"),
        },
        "m11_null_distribute_the_quarter": {
            "must_catch": False,
            "why": ("RECORDED NULL with a derivation: 0.25 is an exact power of "
                    "two, so scaling by it commutes with round-to-nearest (the "
                    "rounding grid scales exactly) and 0.25*(A+B) == 0.25*A + "
                    "0.25*B bitwise wherever nothing is subnormal. The transcribed "
                    "association is fidelity, not a pinned grouping"),
            "apply": lambda s: needle(
                s,
                "    float term_00 = 0.25f * ((near_00 * unear_00)"
                " + (far_00 * ufar_00));",
                "    float term_00 = (0.25f * (near_00 * unear_00))"
                " + (0.25f * (far_00 * ufar_00));"),
        },
        "m12_null_commute_the_row_sum": {
            "must_catch": False,
            "why": ("RECORDED NULL: f32 addition is bitwise commutative, so "
                    "`total + gs*us` equals `gs*us + total`. The diagonal-first "
                    "order is the array path's (stepping.py:1007-1008) and is kept "
                    "for fidelity, not because it is observable"),
            "apply": lambda s: needle(
                s, "    float src0 = (gs0 * us0) + total0;",
                "    float src0 = total0 + (gs0 * us0);"),
        },
    }


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Every mutation ARMED, LAUNCH-COUNTED and classified three ways.

    Then THE BLINDNESS CONTROL, which is the reason this family's gate exists in
    this shape: ``m1`` on UNIFORM rows at 2**-16 amplitude must FAIL bytes and PASS
    ``allclose``. The two measured wrong turns are recorded beside it so a later
    round does not re-derive them.
    """
    harness = MutationHarness(payload, out, key="mutations")
    started = harness.started
    default_codes = (METALLIC, METALLIC, PERIODIC)
    for label, entry in _mutations().items():
        codes = entry.get("codes", default_codes)
        boundaries = tuple("metallic" if c == METALLIC else "periodic"
                           for c in codes)
        grid, fields, pml = build((1.2, 1.0, 0.9), boundaries, 0.35, SEED,
                                  form="varying_full")
        walls = offdiag.wall_mask_axes(grid)
        rows_map = installed_rows(fields)
        inverse = inverse_map(fields)
        base = snapshot(fields)
        stepping.update_E(fields, pml)
        oracle = snapshot(fields)
        row_mask = tuple(int(v is not None) for v in rows_as_slots(rows_map))
        shipped = offdiag.offdiag_source(row_mask, codes, walls)

        counters: List[Counter] = []
        missed = False
        caught = ran = 0
        try:
            mutated = entry["apply"](shipped)
        except LookupError:
            missed = True
            mutated = None
        if mutated is not None:
            got = run_on_device(base, pml, rows_map, inverse, codes, walls,
                                source=mutated, counter=counters)
            ran = 1
            caught = int(any(differing(got[n], oracle[n]) for n in COMPARED))
        launches = sum(counter.launches for counter in counters)
        harness.record(label,
                       harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       extra={"bc": list(codes)})

    # --- HOST mutation: the Yee sub-lattice, chosen by the plan and not the kernel
    grid, fields, pml = build((1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
                              0.35, SEED, form="varying_full")
    codes = boundary_codes(grid, pml)
    walls = offdiag.wall_mask_axes(grid)
    rows_map = installed_rows(fields)
    inverse = inverse_map(fields)
    base = snapshot(fields)
    stepping.update_E(fields, pml)
    oracle = snapshot(fields)
    swapped = run_on_device(base, pml, rows_map, inverse, codes, walls, suffix="")
    caught = int(any(differing(swapped[n], oracle[n]) for n in COMPARED))
    harness.record("m13_half_integer_suffix_swap", f"CAUGHT {caught}/1", 1,
                   caught, 1, True,
                   "the constitutive coefficients taken from the INTEGER Yee "
                   "sub-lattice instead of the half-integer one (stepping.py:1015) "
                   "— a half-cell error in the absorber profile, converged, smooth "
                   "and wrong")

    # --- THE ALLCLOSE-BLINDNESS CONTROL -------------------------------------
    # Uniform rows, state scaled by an EXACT power of two. On uniform rows the m1
    # hoist is algebraically equal and differs only through distributivity
    # rounding; the exact-power-of-two scaling commutes with round-to-nearest, so
    # the byte mismatches SURVIVE while every absolute delta drops under atol.
    scale = 2.0 ** -16
    grid, fields, pml = build((1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
                              0.35, SEED, form="uniform_full", scale=scale)
    codes = boundary_codes(grid, pml)
    walls = offdiag.wall_mask_axes(grid)
    rows_map = installed_rows(fields)
    inverse = inverse_map(fields)
    base = snapshot(fields)
    stepping.update_E(fields, pml)
    oracle = snapshot(fields)
    row_mask = tuple(int(v is not None) for v in rows_as_slots(rows_map))
    shipped = offdiag.offdiag_source(row_mask, codes, walls)
    hoisted = _mutations()["m1_hoist_coefficient_out_of_the_shifts"]["apply"](shipped)
    counters = []
    got = run_on_device(base, pml, rows_map, inverse, codes, walls,
                        source=hoisted, counter=counters)
    byte_delta = sum(differing(got[n], oracle[n]) for n in COMPARED)
    allclose_ok = all(allclose_passes(got[n], oracle[n]) for n in COMPARED)
    # And the recorded wrong turn: the same hoist on VARYING rows, which allclose
    # CORRECTLY catches at any amplitude because rtol is scale-free.
    grid_v, fields_v, pml_v = build((1.2, 1.0, 0.9),
                                    ("metallic", "metallic", "periodic"),
                                    0.35, SEED, form="varying_full", scale=scale)
    codes_v = boundary_codes(grid_v, pml_v)
    walls_v = offdiag.wall_mask_axes(grid_v)
    rows_v = installed_rows(fields_v)
    inverse_v = inverse_map(fields_v)
    base_v = snapshot(fields_v)
    stepping.update_E(fields_v, pml_v)
    oracle_v = snapshot(fields_v)
    mask_v = tuple(int(v is not None) for v in rows_as_slots(rows_v))
    hoisted_v = _mutations()["m1_hoist_coefficient_out_of_the_shifts"]["apply"](
        offdiag.offdiag_source(mask_v, codes_v, walls_v))
    got_v = run_on_device(base_v, pml_v, rows_v, inverse_v, codes_v, walls_v,
                          source=hoisted_v)
    varying_allclose_ok = all(allclose_passes(got_v[n], oracle_v[n])
                              for n in COMPARED)

    row = {"mutation": "blindness_control_m1_uniform_rows_at_2**-16",
           "verdict": f"BYTES {byte_delta} / ALLCLOSE {'PASS' if allclose_ok else 'FAIL'}",
           "launches": sum(c.launches for c in counters),
           "state_scale": scale,
           "byte_differing_words": byte_delta,
           "allclose_passes": allclose_ok,
           "recorded_wrong_turn_varying_rows_allclose_passes": varying_allclose_ok,
           "why": ("THE SHARPEST JUSTIFICATION OF BYTE GATING IN THIS PROJECT. On "
                   "UNIFORM rows the hoist is algebraically equal and differs only "
                   "through distributivity rounding (~1 ulp relative, far under "
                   "rtol=1e-5); scaling the state by an exact power of two commutes "
                   "with round-to-nearest so the byte mismatches survive while "
                   "every absolute delta falls under atol=1e-8. The measured wrong "
                   "turn is recorded beside it: the SAME hoist on VARYING rows is a "
                   "first-order registration defect, O(1) relative, which allclose "
                   "correctly catches at any amplitude because rtol is scale-free")}
    harness.rows.append(row)
    log(f"[leg6 mutations] blindness_control bytes={byte_delta} "
        f"allclose_passes={allclose_ok} varying_allclose_passes="
        f"{varying_allclose_ok} ({time.time() - started:.1f}s)")
    payload["legs"]["mutations"] = harness.rows
    save(payload, out)
    assert byte_delta > 0, (
        "the blindness control's hoist changed NO bytes on uniform rows; the "
        "control cannot demonstrate what byte gating buys if the bytes agree")
    assert allclose_ok, (
        f"the blindness control FAILED allclose ({byte_delta} words differ), so it "
        f"is not a blindness control at all — allclose caught this defect and the "
        f"amplitude or the row form is wrong")
    assert not varying_allclose_ok, (
        "the recorded wrong turn PASSED allclose on varying rows, which contradicts "
        "the derivation: a first-order registration defect is O(1) relative and "
        "rtol is scale-free, so allclose must catch it")
    payload["counts"]["mutations"] = len(harness.rows)
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG 7 — refusals, by name
# ---------------------------------------------------------------------------

def leg_refusals(payload: Dict[str, Any], out: str) -> None:
    """Every configuration this family refuses, with the clause that refuses it.

    Includes the STALE-DOCSTRING GUARD: ``stepping.py:1219-1220`` and
    ``fields.py:1237-1241`` both claim the installer refuses folded rows, and
    ``_validated_offdiagonal_rows`` demonstrably installs them. This leg SHOWS the
    predicate refusing a folded run that ``stepping`` steps, so the refusal is
    recorded as THIS KERNEL FAMILY'S rather than mistaken for the engine's.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    def record(name: str, covered: bool, reasons, expectation: bool,
               note: str = "") -> None:
        row = {"case": name, "covered": covered, "reasons": list(reasons),
               "expected_covered": expectation}
        if note:
            row["note"] = note
        rows.append(row)
        log(f"[leg7 refusals] {name:<34} covered={covered} "
            f"reasons={len(reasons)} ({time.time() - started:.1f}s)")
        payload["legs"]["refusals"] = rows
        save(payload, out)
        assert covered == expectation, row
        if not covered:
            assert reasons, f"{name} refused with NO reason; a nameless refusal " \
                            f"is not auditable"

    # The admitted baseline, so the refusals below are not all this leg measures.
    grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, SEED)
    residency = device.Residency()
    verdict_ok = offdiag.offdiag_constitutive_coverage(fields, pml, residency)
    record("admitted_baseline", verdict_ok.covered, verdict_ok.reasons, True)

    # No residency declared: the Metal-only clause.
    naked = offdiag.offdiag_constitutive_coverage(fields, pml, None)
    record("no_residency_declared", naked.covered, naked.reasons, False)

    # A ZERO-ROW run: the install-time drop makes it the certified plain kernel's,
    # and the two predicates must be DISJOINT rather than merely ordered.
    _g, plain_fields, plain_pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, SEED,
                                        rows={})
    plain_res = device.Residency()
    zero_rows = offdiag.offdiag_constitutive_coverage(plain_fields, plain_pml,
                                                      plain_res)
    record("zero_rows_belong_to_the_plain_kernel", zero_rows.covered,
           zero_rows.reasons, False,
           note="fields.py:1302-1303 drops identically-zero rows at install")
    other = metal_coverage.constitutive_coverage(plain_fields, plain_pml, "E",
                                                 plain_res)
    row = {"case": "disjointness_zero_rows", "offdiag_covered": zero_rows.covered,
           "plain_covered": other.covered}
    rows.append(row)
    payload["legs"]["refusals"] = rows
    save(payload, out)
    assert not (zero_rows.covered and other.covered), (
        "a zero-row run is co-admitted by BOTH the offdiag and the plain "
        "constitutive predicate; the two families would overlap on exactly the "
        "runs the install-time drop separates")

    # No absorber: the family implements the split-field path only.
    inert = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
    no_pml = offdiag.offdiag_constitutive_coverage(fields, inert, residency)
    record("inactive_absorber", no_pml.covered, no_pml.reasons, False)

    # Complex storage.
    complex_grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9),
                        boundaries="periodic", dimensions=3, courant=0.35,
                        k_point=(0.0, 0.0, 0.0), xp=np)
    complex_fields = Fields(grid=complex_grid, force_complex_fields=True)
    complex_fields.enable_pml_storage()
    complex_pml = PML(grid=complex_grid,
                      thickness=tuple((2, 2) for _ in range(3)))
    complex_verdict = offdiag.offdiag_constitutive_coverage(
        complex_fields, complex_pml, device.Residency())
    record("complex_storage", complex_verdict.covered, complex_verdict.reasons,
           False)

    # THE STALE-DOCSTRING GUARD. A folded grid: the install does NOT refuse the
    # rows (whatever stepping.py:1219-1220 and fields.py:1237-1241 claim), and
    # stepping steps the configuration — but this family refuses it, and the
    # refusal is the FAMILY's.
    folded_grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9),
                       boundaries="periodic", dimensions=3, courant=0.35,
                       k_point=(0.0, 0.0, 0.0), xp=np, symmetry=("x",))
    folded = Fields(grid=folded_grid, force_complex_fields=False)
    folded.enable_pml_storage()
    shape = folded_grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    folded.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")},
                               make_rows("varying_full", shape, rng))
    # The folded axis takes the HIGH face only: cell 0 is the mirror plane, which
    # is a boundary condition rather than an absorber (pml.py:408 refuses the pair).
    folded_pml = PML(grid=folded_grid,
                     thickness={"x": {"high": 2}, "y": (2, 2), "z": (2, 2)})
    folded_verdict = offdiag.offdiag_constitutive_coverage(
        folded, folded_pml, device.Residency())
    installed_on_fold = bool(folded.has_offdiagonal_epsilon)
    stepped_on_fold = True
    try:
        stepping.update_E(folded, folded_pml)
    except Exception as exc:  # noqa: BLE001 - the point is whether it steps at all
        stepped_on_fold = False
        payload["legs"]["fold_step_error"] = repr(exc)
    record("folded_axis_stale_docstring_guard", folded_verdict.covered,
           folded_verdict.reasons, False,
           note=(f"rows installed on the folded grid: {installed_on_fold}; "
                 f"stepping.update_E ran: {stepped_on_fold}. stepping.py:1219-1220 "
                 f"and fields.py:1237-1241 claim the installer refuses this "
                 f"combination; _validated_offdiagonal_rows (fields.py:1262-1310) "
                 f"demonstrably does not. The refusal recorded here is THIS KERNEL "
                 f"FAMILY'S, not the engine's"))
    assert installed_on_fold, (
        "the installer refused folded rows, which would mean the stale-docstring "
        "hazard has been fixed underneath this gate — a finding, and this leg's "
        "note must be rewritten rather than left claiming otherwise")

    # ------------------------------------------------------------------
    # EVERY REMAINING UNIMPLEMENTED FEATURE, CONSTRUCTED AND CALLED.
    #
    # The five cases above were the whole leg, and the other nine features this
    # family does not carry — dispersion, chi2/chi3, beta, BFAST, cylindrical, a
    # non-NumPy array module, a scalar inverse epsilon, unstored E, and the five
    # malformed row forms — were refused only by READING `_grid_reasons` and
    # `_row_reasons`. A clause that is read is not a clause that fires: this
    # project has found two real over-coverage defects exactly by constructing the
    # configuration instead. Each case below also names the clause it expects, so
    # a run that refused for an unrelated reason (an accidental shape error, say)
    # fails rather than passing as coverage of the feature.
    # ------------------------------------------------------------------
    def refuse(label: str, fields_like: Any, pml_like: Any, needle: str,
               note: str = "") -> None:
        try:
            verdict = offdiag.offdiag_constitutive_coverage(
                fields_like, pml_like, device.Residency())
        except Exception as exc:  # noqa: BLE001 - a raise is the finding
            raise AssertionError(
                f"{label}: the predicate RAISED {exc!r} instead of returning a "
                f"verdict. plan_offdiag_constitutive promises None-means-refused "
                f"and the arm table treats a raising predicate as a refusal; one "
                f"that does both reports a single configuration two ways "
                f"depending on which caller reached it first") from exc
        record(label, verdict.covered, verdict.reasons, False, note)
        assert any(needle in reason for reason in verdict.reasons), (
            f"{label} was refused, but no reason mentions {needle!r}: "
            f"{list(verdict.reasons)}. A refusal for an unrelated reason is not "
            f"coverage of this feature")

    base_cell, base_bounds = CONFIGS[1][1], CONFIGS[1][2]

    # A registered susceptibility: the source becomes D - sum P in scratch buffers.
    grid_d, fields_d, pml_d = build(base_cell, base_bounds, 0.35, SEED)
    fields_d.polarizations.append(PolarizationState(
        Susceptibility(kind="lorentzian", frequency=1.0, gamma=0.0), 0.5, grid_d,
        np.float32))
    refuse("dispersion_registered", fields_d, pml_d, "susceptibility is registered",
           note=f"driven={fields_d.polarizations[0].driven()}")

    # chi2/chi3: the Pade factor REPLACES the constitutive product.
    _g, fields_n, pml_n = build(base_cell, base_bounds, 0.35, SEED)
    fields_n.set_nonlinear_volumes({c: 0.1 for c in ("Ex", "Ey", "Ez")},
                                   {c: 0.05 for c in ("Ex", "Ey", "Ez")})
    refuse("chi2_chi3_installed", fields_n, pml_n, "chi2/chi3 is installed",
           note=f"has_nonlinearity={fields_n.has_nonlinearity}")

    # beta: MEEP's out-of-plane 2-D wavevector, an added coupling.
    beta_grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.0), courant=0.35,
                     boundaries="periodic", dimensions=2, k_point=(0.0, 0.0, 0.0),
                     beta=0.4, xp=np)
    beta_fields, beta_pml = _bare_rig(beta_grid, complex_fields=True)
    refuse("beta_nonzero", beta_fields, beta_pml, "beta=0.4 is nonzero",
           note="MEEP fields::beta, the analytic exp(i*2*pi*beta*z) dependence")

    # BFAST: a second additive term on every curl.
    bfast_grid = Grid(resolution=10.0, cell_size=base_cell, courant=0.35,
                      boundaries="periodic", dimensions=3, k_point=(0.0, 0.0, 0.0),
                      bfast_scaled_k=(0.2, 0.0, 0.0), xp=np)
    bfast_fields, bfast_pml = _bare_rig(bfast_grid)
    refuse("bfast_active", bfast_fields, bfast_pml, "BFAST is active",
           note=f"bfast_active={bfast_grid.bfast_active}")

    # Cylindrical: the axial extent moves the coefficient index.
    cyl_grid = Grid(resolution=10.0, cell_size=(1.2, 0.0, 0.9), courant=0.35,
                    boundaries="periodic", dimensions=2, k_point=(0.0, 0.0, 0.0),
                    cylindrical=True, m=1, xp=np)
    cyl_fields, cyl_pml = _bare_rig(cyl_grid)
    refuse("cylindrical_axis", cyl_fields, cyl_pml, "cylindrical (Dcyl)")

    # A non-NumPy array module: the residency layer mirrors HOST arrays.
    class _NotNumpy:
        __name__ = "cupy"

    plain_grid, plain_f, plain_p = build(base_cell, base_bounds, 0.35, SEED)
    refuse("array_module_is_not_numpy",
           _Overridden(plain_f, grid=_Overridden(plain_grid, xp=_NotNumpy())),
           plain_p, "not numpy")

    # A scalar inverse epsilon: this kernel binds three pointers and has no
    # scalar arm.
    refuse("scalar_inverse_epsilon",
           _Overridden(plain_f, inverse_epsilon_for=lambda c: np.float32(0.5)),
           plain_p, "is a scalar, not a volume")

    # E recomputed from D: there is no stored array for the tail to accumulate into.
    refuse("E_is_not_stored", _Overridden(plain_f, stores_E=False), plain_p,
           "recomputed from D")

    # The five malformed row forms, PLANTED PAST THE INSTALLER, plus the
    # fail-closed case that used to raise an IndexError out of the predicate.
    shape = tuple(plain_grid.shape)
    # THE LOOP VARIABLE IS NOT `needle`: that is the KIT'S mutation helper, and
    # shadowing it leaves the function unbound for the rest of this leg — a
    # NameError waiting for whoever next plants a defect here.
    for label, value, clause in (
            ("planted_row_float64",
             np.full(shape, 0.03, dtype=np.float64), "is not float32"),
            ("planted_row_complex64",
             (np.full(shape, 0.03) + 0.01j).astype(np.complex64), "is complex"),
            ("planted_row_wrong_shape",
             np.full(shape[:2] + (shape[2] - 1,), np.float32(0.03),
                     dtype=np.float32), "!= grid shape"),
            ("planted_row_f_contiguous",
             np.asfortranarray(np.full(shape, np.float32(0.03), dtype=np.float32)),
             "not C-contiguous"),
            ("planted_row_scalar", np.float32(0.03), "is not a volume"),
            ("planted_row_without_dtype", _NoDtype(), "exposes no dtype")):
        refuse(label, _PlantedRows(plain_f, {"Ex": {"Ey": value}}), plain_p, clause)

    # ------------------------------------------------------------------
    # THE CLAUSES NO CASE HAD EVER CONSTRUCTED. Reading a clause is not evidence
    # that it fires, and these seven were refused only by reading.
    # ------------------------------------------------------------------
    bloch_grid = Grid(resolution=10.0, cell_size=base_cell, boundaries="periodic",
                      dimensions=3, courant=0.35, k_point=(0.3, 0.0, 0.0), xp=np)
    bloch_fields, bloch_pml = _bare_rig(bloch_grid)
    refuse("nonzero_k_point", bloch_fields, bloch_pml, "is not exactly zero",
           note="a Bloch phase multiplies one wrapped plane and needs complex "
                "storage; the wrap here is phase-free by construction")

    class _NoMirror:
        """A residency object that cannot register a volume."""

    naked_residency = offdiag.offdiag_constitutive_coverage(plain_f, plain_p,
                                                            _NoMirror())
    record("residency_without_mirror", naked_residency.covered,
           naked_residency.reasons, False)
    assert any("does not expose mirror()" in reason
               for reason in naked_residency.reasons), naked_residency.reasons

    refuse("fields_without_a_grid", _Overridden(plain_f, grid=None), plain_p,
           "carries no grid")
    refuse("unallocated_auxiliary", _Overridden(plain_f, f_w_Ey=None), plain_p,
           "f_w_Ey is not allocated")

    # THE PUBLIC-INSTALLER ROUTE, and it is not the same case as the planted one.
    # `_validated_offdiagonal_rows` casts with astype(..., copy=False), whose
    # default order="K" PRESERVES Fortran order (fields.py:1296), and
    # `set_epsilon_volumes` stores the inverse map with no normalization at all
    # (fields.py:1251-1252). So a caller reaches BOTH of these refusals through the
    # front door; the docstrings that said every malformed row is unreachable
    # through the installer were measurably wrong about contiguity.
    fortran_grid = Grid(resolution=10.0, cell_size=base_cell, boundaries="periodic",
                        dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fortran_shape = fortran_grid.shape
    fortran_eps = np.full(fortran_shape, np.float32(2.0), dtype=np.float32)
    fortran_inv = np.full(fortran_shape, np.float32(0.5), dtype=np.float32)
    fortran_row = np.asfortranarray(
        (0.03 * np.random.default_rng(SEED + 11).standard_normal(fortran_shape)
         ).astype(np.float32))
    fortran_fields = Fields(grid=fortran_grid, force_complex_fields=False)
    fortran_fields.enable_pml_storage()
    fortran_fields.set_epsilon_volumes(
        {c: fortran_eps for c in ("Ex", "Ey", "Ez")},
        {c: fortran_inv for c in ("Ex", "Ey", "Ez")},
        {"Ex": {"Ey": fortran_row}})
    kept = fortran_fields.chi1inv_offdiagonal_for("Ex").get("Ey")
    assert kept is not None and not kept.flags["C_CONTIGUOUS"], (
        "the installer normalized the Fortran-ordered row after all, so this "
        "case no longer measures the front-door route and the note must be "
        "rewritten rather than left claiming otherwise")
    fortran_pml = PML(grid=fortran_grid, thickness=tuple((2, 2) for _ in range(3)))
    refuse("installed_row_is_f_contiguous", fortran_fields, fortran_pml,
           "not C-contiguous",
           note="reached through the PUBLIC installer, not planted past it: "
                "astype(..., copy=False) keeps Fortran order")

    fortran_inverse = Fields(grid=fortran_grid, force_complex_fields=False)
    fortran_inverse.enable_pml_storage()
    fortran_inverse.set_epsilon_volumes(
        {c: fortran_eps for c in ("Ex", "Ey", "Ez")},
        {c: np.asfortranarray(fortran_inv) for c in ("Ex", "Ey", "Ez")},
        {"Ex": {"Ey": np.ascontiguousarray(fortran_row)}})
    refuse("installed_inverse_epsilon_is_f_contiguous", fortran_inverse,
           fortran_pml, "not C-contiguous",
           note="set_epsilon_volumes stores the inverse map verbatim")

    refuse("pml_coefficient_vector_is_the_wrong_length",
           plain_f, _Overridden(plain_p, kps_x_h=np.zeros(3, dtype=np.float32)),
           "entries, axis x has",
           note="the coefficient vectors are indexed by the component's OWN axis; "
                "a short vector is an out-of-bounds read, not a crash")

    # THE ALIAS CLAUSES, CONSTRUCTED. The output one was refused from the start;
    # the SOURCE one is new and is the second real over-coverage defect this
    # project has found by constructing a configuration rather than reading a
    # clause. Its damage is invisible to this gate's own comparison — see
    # `offdiag._alias_reasons` for the two-step measurement.
    for label, rows_map, needle_text in (
            ("row_aliases_an_output", {"Ex": {"Ey": "Ex"}}, "aliases output Ex"),
            ("row_aliases_a_source", {"Ex": {"Ey": "Dy"}}, "aliases source Dy")):
        _g, alias_fields, alias_pml = build(base_cell, base_bounds, 0.35, SEED)
        alias_fields.set_epsilon_volumes(
            {c: alias_fields._eps_components[c] for c in ("Ex", "Ey", "Ez")},
            {c: alias_fields._inv_eps_components[c] for c in ("Ex", "Ey", "Ez")},
            {row: {partner: getattr(alias_fields, target)
                   for partner, target in partners.items()}
             for row, partners in rows_map.items()})
        refuse(label, alias_fields, alias_pml, needle_text)
        assert offdiag.plan_offdiag_constitutive(
            alias_fields, alias_pml, device.Residency()) is None, (
            f"{label}: the predicate refused but the builder did not return None")

    _g, inv_alias_fields, inv_alias_pml = build(base_cell, base_bounds, 0.35, SEED)
    inv_alias_fields.set_epsilon_volumes(
        {c: inv_alias_fields._eps_components[c] for c in ("Ex", "Ey", "Ez")},
        {"Ex": inv_alias_fields.Dx,
         "Ey": inv_alias_fields._inv_eps_components["Ey"],
         "Ez": inv_alias_fields._inv_eps_components["Ez"]},
        installed_rows(inv_alias_fields))
    refuse("inverse_epsilon_aliases_a_source", inv_alias_fields, inv_alias_pml,
           "aliases source Dx")

    # ------------------------------------------------------------------
    # THE DELIBERATE ADMISSIONS. A refusals leg that only records refusals cannot
    # tell "correctly admitted" from "never asked": conductivity is NOT a clause
    # here, deliberately, on the reading that it changes the CURL sub-steps only
    # (`condfac` appears in stepping.py exclusively inside `_apply_curl`,
    # :504-537). That reading is now MEASURED for this family rather than cited.
    # ------------------------------------------------------------------
    def admit(label: str, install, note: str) -> None:
        _grid, admitted, admitted_pml = build(base_cell, base_bounds, 0.35, SEED,
                                              form="varying_full")
        install(admitted)
        residency_here = device.Residency()
        verdict = offdiag.offdiag_constitutive_coverage(admitted, admitted_pml,
                                                        residency_here)
        before = snapshot(admitted)
        stepping.update_E(admitted, admitted_pml)
        after = snapshot(admitted)
        moved = assert_state_moved(before, after, COMPARED, label)
        restore(admitted, before)
        plan = offdiag.plan_offdiag_constitutive(admitted, admitted_pml,
                                                 residency_here)
        assert plan is not None, verdict.reasons
        plan.run()
        residency_here.sync_out()
        bad = sum(differing(getattr(admitted, n), after[n]) for n in COMPARED)
        row = {"case": label, "covered": verdict.covered,
               "reasons": list(verdict.reasons), "expected_covered": True,
               "differing_words": bad, "moved_words": moved, "note": note}
        rows.append(row)
        log(f"[leg7 admitted] {label:<34} covered={verdict.covered} "
            f"differing={bad} moved={moved} ({time.time() - started:.1f}s)")
        payload["legs"]["refusals"] = rows
        save(payload, out)
        assert verdict.covered, verdict.reasons
        assert bad == 0, row

    conductivity = (0.3 * np.abs(np.random.default_rng(SEED + 5).standard_normal(
        tuple(plain_grid.shape)))).astype(np.float32)
    admit("admitted_d_conductivity",
          lambda f: f.set_d_conductivity(conductivity),
          "an electric conductivity enters step_D's recurrence and nothing in "
          "update_E; admitted on purpose and byte-identical when admitted")
    admit("admitted_b_conductivity",
          lambda f: f.set_b_conductivity(conductivity),
          "the magnetic half of an mp.Absorber; same argument, other curl")

    payload["counts"]["refusals"] = len(rows)
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG 8 — the engine route and the disjointness sweep
# ---------------------------------------------------------------------------

def leg_engine(payload: Dict[str, Any], out: str) -> None:
    """Plans from the engine's own objects, and no two arms co-admitting.

    THE ARM IS REGISTERED ``wired=True`` AS OF TRANCHE 2, and this leg asserts
    THAT. It used to assert the opposite, and the flip is a supersession recorded
    rather than a relaxation: the tranche-1 assertion's stated premise was "a wired
    arm would let ``plan_step`` select a family WHOSE COMPOSITION HAS NOT BEEN
    MEASURED", and that premise is now false by measurement. The whole-step gate
    (``results/metal_whole_step_2026-08-16/whole_step.json``) drives 15 cases x 12
    cycles, 20,200,320 uint32 comparisons, ZERO divergences, with all 23 arms
    wired, and its case ``offdiag_real_one_row`` is this family being SELECTED by
    ``plan_step`` for ``update_E`` (``{"step_B": "PML", "step_D": "PML",
    "update_E": "offdiag", "update_H": "ordinary"}``) and reproducing the array
    path for the whole step. Composition measured is exactly the condition the old
    assertion named as the thing it was protecting.

    THE REPLACEMENT IS STRICTLY STRONGER, which is the only shape in which a gate
    assertion may be re-pointed. "``plan_step`` must select NOTHING" is satisfied
    by a planner that has silently lost the family; the assertions below require
    that it select the OFFDIAG ARM BY LABEL, so a planner that dropped the arm, or
    one that let the PLAIN constitutive arm win the slot, now FAILS here where the
    old form would have passed. The disjointness seam — the plain E-side predicate
    must still REFUSE an off-diagonal run — is unchanged and still the byte-relevant
    invariant: losing it means two products on one slot.
    """
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()

    specs = [s for s in arms.registered(offdiag.SLOT) if s.family == offdiag.FAMILY]
    assert len(specs) == 1, f"expected one offdiag arm, found {specs!r}"
    assert specs[0].wired, (
        "the offdiag arm is registered UNWIRED. Tranche 2 wired it and the "
        "whole-step gate measured the composition; an unwired arm here means the "
        "family was silently dropped out of dispatch")
    arm_label = specs[0].label
    payload["arm_registration"] = {
        "family": specs[0].family, "slot": specs[0].slot, "label": arm_label,
        "wired": bool(specs[0].wired),
        "supersession": (
            "tranche 1 asserted wired=False; tranche 2 wired the arm after the "
            "whole-step gate measured the composition (15 cases x 12 cycles, "
            "20200320 uint32 comparisons, 0 divergences, case "
            "'offdiag_real_one_row' selects update_E=offdiag). This gate asserts "
            "the CURRENT contract and requires selection BY LABEL, which is "
            "stricter than the retired 'selects nothing'"),
        "evidence": "parity/meep_gpu/results/metal_whole_step_2026-08-16/whole_step.json",
    }

    for name, cell, boundaries in CONFIGS:
        for form in ("varying_full", "single_ez_ex"):
            grid, fields, pml = build(cell, boundaries, 0.35, SEED, form=form)
            residency = device.Residency()
            plan = offdiag.plan_offdiag_constitutive(fields, pml, residency)
            assert plan is not None
            step = metal_launch.plan_step(fields, pml, residency=residency,
                                          sources=())
            offdiag_verdict = offdiag.offdiag_constitutive_coverage(
                fields, pml, residency)
            plain_verdict = metal_coverage.constitutive_coverage(
                fields, pml, "E", residency)
            row = {"case": name, "row_form": form,
                   "offdiag_covered": offdiag_verdict.covered,
                   "plain_E_covered": plain_verdict.covered,
                   "plan_step_selected_update_E": step.selected.get("update_E"),
                   "plan_step_replaces": list(step.replaces),
                   "row_mask": list(plan.row_mask)}
            rows.append(row)
            log(f"[leg8 engine] {name:<13} {form:<14} offdiag={offdiag_verdict.covered} "
                f"plain_E={plain_verdict.covered} "
                f"selected_E={step.selected.get('update_E')!r} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["engine"] = rows
            save(payload, out)
            assert offdiag_verdict.covered, offdiag_verdict.reasons
            assert not plain_verdict.covered, (
                f"{name}/{form}: the PLAIN E-side predicate admits an off-diagonal "
                f"run. That refusal is the DISJOINTNESS SEAM between the two "
                f"families and losing it means two products on one slot")
            assert step.selected.get("update_E") == arm_label, (
                f"{name}/{form}: plan_step selected "
                f"{step.selected.get('update_E')!r} for update_E on an "
                f"off-diagonal run, expected {arm_label!r}. The arm is wired and "
                f"its predicate admits, so the offdiag product must own this slot; "
                f"None means the family was dropped out of dispatch and any other "
                f"label means a second family took a slot this one admits")
    payload["counts"]["engine"] = len(rows)
    save(payload, out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("transcription", leg_transcription),
    ("execution", leg_execution),
    ("reference", leg_reference),
    ("synthetic", leg_synthetic),
    ("seam", leg_seam),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("refusals", leg_refusals),
    ("engine", leg_engine),
)


def main() -> int:
    parser = argument_parser(__doc__ or "")
    arguments = parser.parse_args()
    started = time.time()
    payload: Dict[str, Any] = {"environment": environment_stamp(), "legs": {},
                               "counts": {}}

    refusals: List[str] = []
    if not payload["environment"].get("mps_available"):
        refusals.append("no MPS device is available on this host; the claim was "
                        "not measured")
    policy = subnormal.mps_policy_report()
    payload["subnormal_policy"] = policy
    if not policy["admitted"]:
        refusals.extend(policy["reasons"])
    if refusals:
        return cannot_certify(payload, arguments.out, refusals)

    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # This family's OWN source enumeration, recorded HERE rather than in
    # fingerprints.json: the byte gate binds its own provenance record inside its
    # results directory, so the verdict and the sources it is a verdict about
    # travel together. (The arm is WIRED as of tranche 2; this record is kept
    # family-local regardless, because a shared side file is exactly what an
    # earlier probe overwrote.)
    launched = {
        f"offdiag_constitutive_step/rows-{''.join(str(b) for b in mask)}"
        f"/bc-{''.join(str(c) for c in codes)}"
        f"/walls-{''.join(str(w) for w in walls)}/contract-{mode}":
            offdiag.offdiag_source(mask, codes, walls, mode)
        for mask in ((1, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1))
        for codes in ((0, 0, 0), (1, 1, 0), (1, 1, 1))
        for walls in ((0, 0, 0), (1, 1, 0), (1, 1, 1))
        for mode in shaders.CONTRACT_MODES
    }
    payload["provenance"] = provenance(out_dir, {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "fields.py": os.path.join(API_ROOT, "meep_gpu", "fields.py"),
        "metal_kernels/offdiag_update_e.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "offdiag_update_e.py"),
        "metal_kernels/templates.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
        "metal_kernels/shaders.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "shaders.py"),
        "metal_kernels/device.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
        "metal_kernels/plans.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "plans.py"),
        "metal_kernels/preconditions.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "preconditions.py"),
        "metal_kernels/coverage.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "coverage.py"),
        # ADDED when the arm went wired: leg 8 calls `launch.plan_step` and
        # asserts WHICH family wins `update_E`, so the planner and the arm table
        # are now load-bearing inputs to this gate's verdict. They were absent
        # from this list while the arm was unwired and the leg asserted only that
        # nothing was selected — a provenance gap that would have let the planner
        # or the registration change under a green gate without the hashes moving.
        "metal_kernels/launch.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "launch.py"),
        "metal_kernels/arms.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "arms.py"),
        "metal_kernels/registry.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "registry.py"),
        "gate_metal_offdiag.py": os.path.abspath(__file__),
        "metal_gate_kit.py": os.path.join(HERE, "metal_gate_kit.py"),
    }, kernel_sources=launched, name="gate_provenance.json")
    # AND IN THE GATE'S OWN ARTIFACT, not only in a side file. The per-source
    # hashes are what says WHICH specialisations this run launched, and a side file
    # in a shared directory is exactly what the probe overwrote; carrying them in
    # gate.json means the verdict and the thing it is a verdict about travel
    # together.
    payload["launched_kernel_sha256"] = {
        label: shaders.source_sha256(source)
        for label, source in sorted(launched.items())}
    payload["corpus_digest"] = {mode: offdiag.corpus_digest(mode)
                                for mode in shaders.CONTRACT_MODES}
    payload["iyee_shifts"] = {c: list(IYEE_SHIFTS[c]) for c in ("Ex", "Ey", "Ez")}
    payload["courant_note"] = (
        "dtdx does NOT enter this sub-step's arithmetic at all. The "
        "non-power-of-two-Courant mandate therefore lands in the REAL-LAYER "
        "COEFFICIENT TABLES (kps_a_h / kms_a_h cut at courant=0.35), and this "
        "artifact says so rather than sweeping a dead parameter and calling it "
        "coverage")
    payload["predicted_nulls"] = [
        {"name": "m11_null_distribute_the_quarter",
         "reason": "0.25 is an exact power of two, so scaling by it commutes with "
                   "round-to-nearest away from underflow: 0.25*(A+B) == 0.25*A + "
                   "0.25*B bitwise. The transcribed association is fidelity, not a "
                   "pinned grouping. MEASURED by the mutation leg as must_catch "
                   "False"},
        {"name": "m12_null_commute_the_row_sum",
         "reason": "f32 addition is bitwise commutative. MEASURED by the mutation "
                   "leg as must_catch False"},
        {"name": "partner_axis_metallic_NEAR_zero_ghost",
         "reason": "MEASURED BY MUTATION as of 2026-08-15 — this entry used to say "
                   "'NOT MEASURED BY MUTATION, recorded rather than claimed', and "
                   "it was measurable all along. The near ghost's entire support "
                   "is the partner-axis face-0 plane, which the wall-coupling mask "
                   "zeroes BEFORE the row sum: a metallic partner axis is always a "
                   "Yee-shift-0 transverse axis of the component, so the mask "
                   "covers exactly that plane. m20 wraps the ghost instead of "
                   "zeroing it and moves 0 words; its sensitivity control m21 "
                   "makes the same change on the OWN-axis far ghost, which the "
                   "mask does not cover, and moves 162 — so the zero is the mask "
                   "hiding a ghost rather than the harness being blind to one. "
                   "Mirrors, where the mask abstains, are refused by this "
                   "family's predicate"},
        {"name": "dtdx_sweep",
         "reason": "NOT SWEPT, and deliberately: dtdx does not enter this "
                   "sub-step. See courant_note"},
    ]
    save(payload, arguments.out)

    log(f"[env] torch={payload['environment']['torch']} "
        f"frontend={payload['environment']['metal_frontend']} "
        f"policy={policy['resolved']}")

    certified = True
    try:
        ran = run_legs(LEGS, payload, arguments.out, wanted_legs(arguments.legs))
    except AssertionError as exc:
        certified = False
        ran = list(payload["legs"])
        payload["failure"] = str(exc)
        log(f"[FAILED] {exc}")

    return summarize(
        payload, arguments.out,
        claim=("byte-identity to stepping.py on this host, under a CHECKED "
               "subnormal-free precondition (leg 5) — not a stated tolerance"),
        scope=("the off-diagonal (tensor) chi1inv electric constitutive sub-step "
               "under an active split-field PML: real float32 storage, periodic "
               "and metallic ghost rules, volume inverse epsilon (three distinct "
               "and three aliased), one to six surviving rows; NO fold, no "
               "cylindrical axis, no Bloch phase, no beta, no BFAST, no chi2/chi3, "
               "no dispersion, no scalar inverse epsilon"),
        stated_weakness=("no PTX-equivalent audit exists on this executor: "
                         "compile_shader exposes no disassembly, so a 'refuse the "
                         "compile when the generated code violates the policy' "
                         "guard has no Metal analogue and the mutation legs plus "
                         "this gate are the only arbiters"),
        started=started, legs_run=ran,
        compared=sum(payload["counts"].values()), certified=certified,
        extra={"dispatch": (
            "WIRED as of tranche 2: the arm is registered wired=True and leg 8 "
            "asserts plan_step selects it BY LABEL for update_E on every "
            "off-diagonal configuration, while the plain E-side arm still "
            "refuses. Composition was measured by the whole-step gate "
            "(results/metal_whole_step_2026-08-16/whole_step.json: 15 cases x 12 "
            "cycles, 20200320 uint32 comparisons, 0 divergences, case "
            "'offdiag_real_one_row'). meep_gpu.fastpath.plan_fast_path is a "
            "separate question and is NOT measured here"),
            "arm_registration": payload.get("arm_registration")})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
