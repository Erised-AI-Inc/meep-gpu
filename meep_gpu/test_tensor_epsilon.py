"""Off-diagonal chi1inv (tensor permittivity) tests: hand stencil, degenerate byte
identity, CPU-MEEP oracles, and refusals.

The constitutive update with a full inverse-permittivity row is MEEP's
``step_update_EDHB`` (step_generic.cpp:566-720)::

    E_c[i] = D_c[i]*u[i] + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)
    OFFDIAG(u, g, sx) = 0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])

with ``s`` the stride along the component's OWN axis, ``s1``/``s2`` the strides
along the two partners in cycle order (X->Y->Z), and the off-diagonal row entries
registered at the component's Yee site MINUS a half cell along its own axis — the
integer node — by ``structure_chunk::set_chi1inv`` (anisotropic_averaging.cpp:
248-257, ``here - shift1``). That registration is what makes MEEP's "stable
averaging" pairing exact: each two-point partner average sits at the same node as
the coefficient that multiplies it.

The uniform-tensor oracles isolate the stencil arithmetic (no geometry, no
registration ambiguity); the hand-stencil test pins the exact index pattern
against an independently written loop; the degenerate test holds the diagonal
engine byte-identical when the feature is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from .driver import FdtdDriver
from .grid import Mirror
from . import stepping

_HAS_MEEP = importlib.util.find_spec("meep") is not None

# The uniform tensor every oracle run shares: symmetric, positive definite
# (eigenvalues 1.83, 2.42, 3.25), every off-diagonal entry nonzero so all six
# OFFDIAG terms carry weight.
_EPS_TENSOR = np.array([
    [2.0, 0.35, 0.20],
    [0.35, 2.5, 0.15],
    [0.20, 0.15, 3.0],
], dtype=np.float64)

_CASE = {
    "cell_size": (1.1, 1.3, 1.5),
    "resolution": 10,
    "frequency": 1.0,
    "until": 1.5,
    "sources": (
        ("Ex", (0.15, -0.25, -0.35), 0.7),
        ("Ey", (-0.15, 0.25, -0.15), 0.5),
        ("Ez", (0.05, 0.05, -0.25), 1.0),
    ),
    "eps_diag": [_EPS_TENSOR[0, 0], _EPS_TENSOR[1, 1], _EPS_TENSOR[2, 2]],
    "eps_offdiag": [_EPS_TENSOR[0, 1], _EPS_TENSOR[0, 2], _EPS_TENSOR[1, 2]],
    "chi3": 1e-3,
}

# variant -> extra mp.Simulation arguments the oracle applies. "periodic" is the
# k = 0 wrap the diagonal-epsilon oracle uses; "metallic" is MEEP's default PEC
# box; "bloch" carries a phase across the wrap the avg4 neighbour reads cross;
# "pml" exercises the f_w path the absorbing accumulation takes the product
# through; "chi3" pins the Pade-factor composition of MEEP's MOST GENERAL CASE.
_VARIANTS = ("periodic", "metallic", "bloch", "pml", "chi3")

_ORACLE_SCRIPT = r'''
"""CPU-MEEP oracle for a uniform FULL-TENSOR dielectric, five boundary variants."""
import json
import sys

import meep as mp
import numpy as np

case = json.loads(sys.argv[1])
output_dir = sys.argv[2]

def medium(chi3=0.0):
    return mp.Medium(
        epsilon_diag=mp.Vector3(*case["eps_diag"]),
        epsilon_offdiag=mp.Vector3(*case["eps_offdiag"]),
        E_chi3_diag=mp.Vector3(chi3, chi3, chi3),
    )

def build(variant):
    kwargs = dict(
        cell_size=mp.Vector3(*case["cell_size"]),
        resolution=case["resolution"],
        default_material=medium(case["chi3"] if variant == "chi3" else 0.0),
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=case["frequency"]),
                component=getattr(mp, component),
                center=mp.Vector3(*center),
                amplitude=amplitude,
            )
            for component, center, amplitude in case["sources"]
        ],
        force_complex_fields=True,
        k_point=mp.Vector3(),
    )
    if variant == "metallic":
        kwargs["k_point"] = False
    elif variant == "bloch":
        kwargs["k_point"] = mp.Vector3(0.3, 0.0, 0.0)
    elif variant == "pml":
        kwargs["boundary_layers"] = [mp.PML(0.4)]
    return mp.Simulation(**kwargs)

for variant in json.loads(sys.argv[3]):
    simulation = build(variant)
    simulation.run(until=case["until"])
    np.savez(
        output_dir + "/" + variant + ".npz",
        **{
            component.lower(): np.asarray(
                simulation.get_array(component=getattr(mp, component), cmplx=True)
            )
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
    )
    print(variant, "done", flush=True)
'''


def _chi1inv_rows():
    """The inverse tensor, as the driver installs it: per-row diagonal 'epsilon'
    (the reciprocal of the chi1inv diagonal entry) plus the raw off-diagonal
    chi1inv entries, exactly the quantities MEEP's get_chi1inv reports."""
    inverse = np.linalg.inv(_EPS_TENSOR)
    diagonal = {name: 1.0 / inverse[i, i] for i, name in enumerate(("Ex", "Ey", "Ez"))}
    names = ("Ex", "Ey", "Ez")
    offdiagonal = {
        row: {
            partner: inverse[i, j]
            for j, partner in enumerate(names) if j != i
        }
        for i, row in enumerate(names)
    }
    return diagonal, offdiagonal


def _uniform(driver, value):
    return np.full(driver.shape, value, dtype=np.float32)


def _install_tensor(driver, include_offdiagonal=True, scale=1.0):
    diagonal, offdiagonal = _chi1inv_rows()
    volumes = {name: _uniform(driver, value) for name, value in diagonal.items()}
    if not include_offdiagonal:
        driver.set_epsilon_components(volumes)
        return
    rows = {
        row: {partner: _uniform(driver, value * scale)
              for partner, value in partners.items()}
        for row, partners in offdiagonal.items()
    }
    driver.set_epsilon_components(volumes, chi1inv_offdiagonal=rows)


def _run_engine(variant, include_offdiagonal=True):
    kwargs = dict(
        cell_size=_CASE["cell_size"],
        resolution=_CASE["resolution"],
        force_complex_fields=True,
    )
    if variant == "metallic":
        kwargs["boundaries"] = "metallic"
    elif variant == "bloch":
        kwargs["k_point"] = (0.3, 0.0, 0.0)
    driver = FdtdDriver(**kwargs)
    _install_tensor(driver, include_offdiagonal=include_offdiagonal)
    if variant == "chi3":
        driver.set_chi3(float(_CASE["chi3"]))
    if variant == "pml":
        driver.setup_pml(4)  # 0.4 length units at resolution 10, matching mp.PML(0.4).
    for component, center, amplitude in _CASE["sources"]:
        driver.add_source({
            "component": component,
            "frequency": _CASE["frequency"],
            "center": center,
            "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude,
        })
    driver.run(until=_CASE["until"])
    fields = {
        component.lower(): driver.get_field(component)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    driver.close()
    return fields


def _relative_l2(candidate, reference):
    denominator = np.linalg.norm(reference.ravel())
    assert denominator > 0.0, "CPU-MEEP reference field is degenerate"
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


@pytest.fixture(scope="module")
def tensor_meep_oracle(tmp_path_factory):
    if not _HAS_MEEP:
        pytest.skip("CPU MEEP is not installed")
    directory = tmp_path_factory.mktemp("tensor_epsilon_oracle")
    script_path = directory / "oracle.py"
    script_path.write_text(_ORACLE_SCRIPT, encoding="utf-8")
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(_CASE), str(directory),
         json.dumps(list(_VARIANTS))],
        capture_output=True, text=True, env=environment, timeout=600,
    )
    assert completed.returncode == 0, (
        f"CPU-MEEP tensor-epsilon oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    oracle = {}
    for variant in _VARIANTS:
        with np.load(directory / f"{variant}.npz") as archive:
            oracle[variant] = {name: np.asarray(archive[name]) for name in archive.files}
    return oracle


def _reference_view(variant, volume):
    """MEEP's full-volume get_array layout vs the driver's owned cells.

    Periodic runs carry the duplicate low plane (driver cell i = MEEP i+1);
    a metallic run returns exactly the owned N cells.
    """
    if variant == "metallic":
        return volume
    return volume[1:, 1:, 1:]


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
@pytest.mark.parametrize("variant", _VARIANTS)
def test_uniform_tensor_epsilon_matches_cpu_meep(tensor_meep_oracle, variant):
    """All six complex fields match MEEP under a uniform FULL-TENSOR epsilon.

    A uniform tensor has no registration question — every slot holds the same
    row — so this isolates the stencil arithmetic: the 0.25 stable average, the
    partner-axis pairing, the own-axis pairing, the boundary ghosts each variant
    exercises, and (chi3) the Pade-factor composition of MEEP's MOST GENERAL
    CASE. Measured on landing, worst component per variant: periodic 1.88e-07,
    metallic 2.03e-07, bloch 1.86e-07, pml 4.62e-07, chi3 2.07e-07 — the
    engine's established floors for each feature class, so the tensor terms cost
    nothing. The diagonal-only control sits at 2.68e-01, six orders away. The
    metallic variant is also the pin for the wall mask: unmasked, the coupling
    term writes partner-D values into the never-stepped tangential wall plane
    and the same comparison reads 2.6e-02.
    """
    result = _run_engine(variant)
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        reference = _reference_view(variant, tensor_meep_oracle[variant][component])
        candidate = result[component]
        assert candidate.shape == reference.shape, (
            f"{variant}/{component}: shape {candidate.shape} vs MEEP {reference.shape}"
        )
        error = _relative_l2(candidate, reference)
        assert error < 5e-6, f"{variant}/{component} complex relative L2 {error:.3e}"


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_dropping_the_offdiagonal_is_not_a_small_error(tensor_meep_oracle):
    """The diagonal-only run of the same tensor is FAR from the oracle.

    This is the run the old refusal existed to prevent — same diagonal, coupling
    thrown away — and it is the control that proves the parity test above can
    tell the feature from its absence.
    """
    result = _run_engine("periodic", include_offdiagonal=False)
    reference = _reference_view("periodic", tensor_meep_oracle["periodic"]["ex"])
    difference = _relative_l2(result["ex"], reference)
    assert difference > 0.05, (
        f"diagonal-only Ex is only {difference:.3e} from the full-tensor oracle; "
        f"the parity test cannot see the off-diagonal terms at all."
    )


def _hand_stencil(D, u_diag, offdiag_rows, component):
    """MEEP's step_update_EDHB row product, written as an independent index loop.

    Periodic k = 0 indices throughout (every wrap is a plain modulo). This is
    deliberately NOT a rearrangement of the engine's vectorized expression — the
    strides, the pairing and the 0.25 come straight from step_generic.cpp:583.
    """
    axes = ("x", "y", "z")
    own = axes.index(component[1].lower())
    n = D[component[1].lower()].shape
    result = np.zeros(n, dtype=np.complex128)
    g = D[component[1].lower()]
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                value = g[i, j, k] * u_diag[i, j, k]
                for offset in (1, 2):
                    partner_axis = (own + offset) % 3
                    partner = axes[partner_axis]
                    u_row = offdiag_rows.get("E" + partner)
                    if u_row is None:
                        continue
                    gp = D[partner]

                    def at(di, dj, dk, arr):
                        return arr[(i + di) % n[0], (j + dj) % n[1], (k + dk) % n[2]]

                    down = [0, 0, 0]
                    down[partner_axis] = -1
                    up = [0, 0, 0]
                    up[own] = 1
                    pair_low = at(0, 0, 0, gp) + at(*down, gp)
                    pair_high = (at(*up, gp)
                                 + at(up[0] + down[0], up[1] + down[1], up[2] + down[2], gp))
                    value += 0.25 * (pair_low * at(0, 0, 0, u_row)
                                     + pair_high * at(*up, u_row))
                result[i, j, k] = value
    return result


def test_offdiagonal_stencil_matches_an_independent_hand_loop():
    """One constitutive update equals MEEP's formula written out index by index.

    Spatially varying D AND spatially varying off-diagonal rows, so a stencil
    that samples the coefficient at one site instead of two (the unstable form),
    pairs the partner shift upward instead of downward, or drops the 0.25 cannot
    match. Agreement bar 1e-6 relative (float32 storage against a float64 loop).
    """
    rng = np.random.default_rng(20260803)
    driver = FdtdDriver(cell_size=(0.4, 0.5, 0.6), resolution=10,
                        force_complex_fields=True)
    shape = driver.shape
    diagonal = {name: _uniform(driver, value)
                for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
    rows = {}
    for row, partners in (("Ex", ("Ey", "Ez")), ("Ey", ("Ez", "Ex")), ("Ez", ("Ex", "Ey"))):
        rows[row] = {p: rng.uniform(-0.2, 0.2, size=shape).astype(np.float32)
                     for p in partners}
    driver.set_epsilon_components(diagonal, chi1inv_offdiagonal=rows)
    fields = driver.fields
    D = {}
    for axis in ("x", "y", "z"):
        pattern = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        getattr(fields, "D" + axis)[...] = pattern.astype(np.complex64)
        D[axis] = np.asarray(getattr(fields, "D" + axis)).astype(np.complex128)

    stepping.update_E(fields, None)

    for component in ("Ex", "Ey", "Ez"):
        expected = _hand_stencil(
            D,
            np.asarray(fields.inverse_epsilon_for(component), dtype=np.float64),
            {p: np.asarray(a, dtype=np.float64)
             for p, a in rows[component].items()},
            component,
        )
        stored = np.asarray(fields.get_E(component))
        error = _relative_l2(stored, expected)
        assert error < 1e-6, f"{component}: engine vs hand loop {error:.3e}"
    driver.close()


def test_offdiagonal_contribution_is_linear_in_the_coefficient():
    """E(2u_off) - E(0) is exactly twice E(u_off) - E(0), pointwise.

    The off-diagonal term is linear in the coefficient by construction; a scale
    factor hidden anywhere in the stencil (a dropped 0.25, a double-counted
    plane) breaks this exact relation before it breaks any oracle bound.
    """
    def run(scale):
        driver = FdtdDriver(cell_size=(0.4, 0.4, 0.4), resolution=10,
                            force_complex_fields=True)
        if scale == 0.0:
            _install_tensor(driver, include_offdiagonal=False)
        else:
            _install_tensor(driver, scale=scale)
        rng = np.random.default_rng(7)
        for axis in ("x", "y", "z"):
            pattern = rng.standard_normal(driver.shape) + 1j * rng.standard_normal(driver.shape)
            getattr(driver.fields, "D" + axis)[...] = pattern.astype(np.complex64)
        if not driver.fields.stores_E:
            driver.fields.enable_field_storage()
        stepping.update_E(driver.fields, None)
        result = {c: np.asarray(driver.fields.get_E(c)).copy() for c in ("Ex", "Ey", "Ez")}
        driver.close()
        return result

    base = run(0.0)
    single = run(1.0)
    double = run(2.0)
    for component in ("Ex", "Ey", "Ez"):
        lhs = double[component] - base[component]
        rhs = 2.0 * (single[component] - base[component])
        scale = np.abs(rhs).max()
        assert scale > 0.0, f"{component}: the off-diagonal contribution is identically zero"
        assert np.abs(lhs - rhs).max() < 1e-5 * scale, (
            f"{component}: doubling the coefficient did not double the contribution"
        )


def test_zero_offdiagonal_rows_are_dropped_and_change_nothing():
    """Explicit all-zero rows reduce to the diagonal engine, byte for byte."""
    def run(with_zero_rows):
        driver = FdtdDriver(cell_size=(0.4, 0.5, 0.4), resolution=10,
                            force_complex_fields=True)
        volumes = {name: _uniform(driver, value)
                   for name, value in (("Ex", 2.0), ("Ey", 3.0), ("Ez", 4.0))}
        if with_zero_rows:
            zeros = {row: {partner: np.zeros(driver.shape, dtype=np.float32)
                           for partner in ("Ex", "Ey", "Ez") if partner != row}
                     for row in ("Ex", "Ey", "Ez")}
            driver.set_epsilon_components(volumes, chi1inv_offdiagonal=zeros)
        else:
            driver.set_epsilon_components(volumes)
        assert driver.fields.has_offdiagonal_epsilon is False
        driver.add_source({"component": "Ez", "frequency": 1.0,
                           "center": (0.05, 0.05, -0.05), "size": (0.0, 0.0, 0.0)})
        driver.run(until=1.0)
        fields = {c: np.asarray(driver.get_field(c)).tobytes()
                  for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        driver.close()
        return fields

    without = run(False)
    with_rows = run(True)
    for component, payload in without.items():
        assert with_rows[component] == payload, (
            f"{component}: an all-zero off-diagonal row changed the run; the degenerate "
            f"case must reduce to the diagonal engine exactly."
        )


def test_offdiagonal_rows_are_validated():
    """Wrong shape, non-finite values and complex rows raise; none installs."""
    driver = FdtdDriver(cell_size=(0.4, 0.4, 0.4), resolution=10,
                        force_complex_fields=True)
    volumes = {name: _uniform(driver, 2.0) for name in ("Ex", "Ey", "Ez")}

    bad_shape = {"Ex": {"Ey": np.zeros((2, 2, 2), dtype=np.float32)}}
    with pytest.raises(ValueError, match="shape"):
        driver.set_epsilon_components(volumes, chi1inv_offdiagonal=bad_shape)

    not_finite = {"Ex": {"Ey": np.full(driver.shape, np.nan, dtype=np.float32)}}
    with pytest.raises(ValueError, match="finite"):
        driver.set_epsilon_components(volumes, chi1inv_offdiagonal=not_finite)

    complex_row = {"Ex": {"Ey": np.full(driver.shape, 0.1 + 0.1j, dtype=np.complex64)}}
    with pytest.raises(ValueError, match="real"):
        driver.set_epsilon_components(volumes, chi1inv_offdiagonal=complex_row)

    self_coupling = {"Ex": {"Ex": _uniform(driver, 0.1)}}
    with pytest.raises(ValueError, match="diagonal"):
        driver.set_epsilon_components(volumes, chi1inv_offdiagonal=self_coupling)
    driver.close()


def _node_coordinates(driver):
    """The all-integer NODE positions every off-diagonal slot is registered at.

    MEEP samples each off-diagonal entry at ``here - shift1`` — the all-even
    ivec (anisotropic_averaging.cpp:248-257) — so the profiles a symmetry
    argument depends on must be built there. Building them at the component's
    own Yee coordinates offsets the coefficient by half a cell AS CONSUMED,
    which silently breaks its odd symmetry: measured 1.9e-03 of apparent fold
    error that was entirely probe misregistration.
    """
    grid = driver.grid
    axes = [grid.axis_origin(axis) + np.arange(count) * grid.dx
            for axis, count in enumerate(driver.shape)]
    return np.meshgrid(*axes, indexing="ij")


@pytest.mark.parametrize("kind", ["even", "odd", "both"])
def test_tensor_fold_equivalence_is_exact(kind):
    """A mirror-folded tensor run reproduces the full domain to storage precision.

    Both coefficient parities under Mirror(X), separately and together:

    * ``even`` — chi_yz (neither index on the fold axis), uniform-symmetric,
      NONZERO ON THE PLANE. This is the case the fold-plane mask would destroy:
      zeroing the coupling there the way the metallic rule does costs 2.0e-02.
    * ``odd`` — chi_xy (one index on the fold axis), an odd smooth function of x
      vanishing on the plane, as any structure symmetric under the fold must.
    * ``both`` together.

    Measured on landing: 3.1e-12 / 8.3e-13 / 4.7e-12 — float32 storage noise.
    The folded driver's rows are the stored-half WINDOW of the full driver's
    arrays, so the comparison carries bitwise-identical coefficients and
    measures fold physics alone. The stencil never ghosts the coefficient (the
    partner-axis shift touches the field before the multiply; the own-axis
    shift goes UP, away from the fold), which is why no coefficient-parity
    bookkeeping is needed anywhere.
    """
    cell = (2.0, 4.0, 2.0)
    primaries = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")

    def build_rows(driver):
        X, Y, _ = _node_coordinates(driver)
        even = (0.10 * np.exp(-(X ** 2 + 0.5 * Y ** 2))).astype(np.float32)
        odd = (0.12 * np.tanh(2.0 * X) * np.exp(-(X ** 2 + Y ** 2))).astype(np.float32)
        rows = {}
        if kind in ("even", "both"):
            rows.setdefault("Ey", {})["Ez"] = even
            rows.setdefault("Ez", {})["Ey"] = even
        if kind in ("odd", "both"):
            rows.setdefault("Ex", {})["Ey"] = odd
            rows.setdefault("Ey", {})["Ex"] = odd
        return rows

    def make(symmetry, rows):
        driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                            symmetry=(Mirror("X"),) if symmetry else ())
        driver.set_epsilon_components(
            {name: _uniform(driver, value)
             for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
            chi1inv_offdiagonal=rows,
        )
        driver.add_source({"component": "Ez", "frequency": 1.0, "source_type": "gaussian",
                           "fwidth": 0.8, "center": (0.0, 0.0, 0.0),
                           "size": (0.0, 0.0, 0.0)})
        return driver

    full = make(False, None)
    full_rows = build_rows(full)
    full.close()
    full = make(False, full_rows)
    centre = full.shape[0] // 2
    window = (slice(centre - 1, None),)
    # The folded grid stores the +L/2 far plane as a real cell (one row more
    # than the full-domain window reaches), so the coefficients are built on
    # the folded driver's own coordinates: bitwise-identical to the full
    # arrays on every shared row — the formulas see the same inputs — and the
    # far row gets its true x = +L/2 values, which a wrap-copy of row 0 would
    # sign-flip for the odd-parity coefficient.
    folded_probe = make(True, None)
    folded_rows = build_rows(folded_probe)
    folded_probe.close()
    folded = make(True, folded_rows)
    assert folded.fields.has_offdiagonal_epsilon, "the folded install dropped the rows"
    full.run(num_steps=12)
    folded.run(num_steps=12)
    scale = max(float(np.max(np.abs(full.get_field(n, cell_centered=False))))
                for n in primaries)
    assert scale > 0.0
    # The stored far-plane row has no full-domain counterpart in the window;
    # its dynamics are pinned by the fold's own far-plane tests, so the
    # equivalence here compares the shared rows and trims it.
    worst = max(
        float(np.max(np.abs(folded.get_field(name, cell_centered=False)[:-1]
                            - full.get_field(name, cell_centered=False)[window])))
        for name in primaries + ("Ex", "Ey", "Ez")
    )
    full.close(); folded.close()
    assert worst / scale < 1e-9, (
        f"{kind}: the folded tensor run diverges from the full domain by "
        f"{worst / scale:.3e}; the fold must be exact (measured 8.3e-13..4.7e-12)."
    )
