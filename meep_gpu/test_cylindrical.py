"""Cylindrical-coordinates (MEEP Dcyl) tests — the P1 ladder of
the design notes (fdtd-cylindrical-plan).

Built test-first, one increment at a time: each test lands alongside the engine
piece it pins, so the tree stays green throughout the port. The MEEP facts each
test transcribes are cited at the formula (step_db.cpp line numbers from the
tree the plan was read against).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from . import stepping
from .driver import FdtdDriver
from .grid import AXIS, METALLIC, Grid, Mirror

_HAS_MEEP = importlib.util.find_spec("meep") is not None

_M0_CASE = {
    "r": 2.0, "z": 4.0, "resolution": 10, "until": 3.0,
    "frequency": 1.0, "fwidth": 0.5, "center": (0.75, 0.0, -0.55),
    # Transmitted-flux oracle (P3): a z-normal plane reaching from the axis out
    # to r=1.2, decimation pinned to 1 on both sides — MEEP's automatic factor
    # is a *sampling* choice, and on a run this short the two Riemann sums
    # legitimately differ (measured 1.36-1.41x with MEEP left on automatic).
    "flux": {
        "fcen": 1.0, "df": 0.4, "nfreq": 3, "fwidth": 0.4, "until": 8.0,
        "center": (0.4, 0.0, -0.55),
        "plane_center": (0.6, 0.0, 0.8), "plane_size": (1.2, 0.0, 0.0),
    },
}

_M0_ORACLE = r'''
"""CPU-MEEP oracle: cylindrical m in (0, 1), vacuum, metallic walls, off-axis Ez source."""
import json, sys
import meep as mp
import numpy as np

case = json.loads(sys.argv[1])
out = {}
PML = [mp.PML(0.5, direction=mp.R, side=mp.High), mp.PML(0.5, direction=mp.Z)]
AXIS = (0.0, 0.0, -0.55)
for tag, m, layers, until, cmplx, src, where in (
        ("m0_", 0, [], case["until"], True, mp.Ez, case["center"]),
        ("m1_", 1, [], case["until"], True, mp.Ez, case["center"]),
        ("m2_", 2, [], case["until"], True, mp.Ez, case["center"]),
        ("m0p_", 0, PML, 2 * case["until"], True, mp.Ez, case["center"]),
        ("m1p_", 1, PML, 2 * case["until"], True, mp.Ez, case["center"]),
        ("m0r_", 0, [], case["until"], False, mp.Ez, case["center"]),
        ("a_er1_", 1, [], case["until"], True, mp.Er, AXIS),
        ("a_ez0_", 0, [], case["until"], True, mp.Ez, AXIS),
        ("a_ep1_", 1, [], case["until"], True, mp.Ep, AXIS)):
    sim = mp.Simulation(
        cell_size=mp.Vector3(case["r"], 0, case["z"]), resolution=case["resolution"],
        dimensions=mp.CYLINDRICAL, m=m, force_complex_fields=cmplx,
        boundary_layers=layers,
        sources=[mp.Source(mp.GaussianSource(frequency=case["frequency"], fwidth=case["fwidth"]),
                           component=src, center=mp.Vector3(*where))])
    sim.run(until=until)
    for name, comp in (("er", mp.Er), ("ep", mp.Ep), ("ez", mp.Ez),
                       ("hr", mp.Hr), ("hp", mp.Hp), ("hz", mp.Hz)):
        out[tag + name] = np.asarray(sim.get_array(component=comp, cmplx=cmplx))
    out[tag + "hp_snap"] = np.asarray(sim.get_array(component=mp.Hp, cmplx=cmplx, snap=True))
    sim.reset_meep()
flux_case = case["flux"]
for tag, m in (("f0_", 0), ("f1_", 1)):
    sim = mp.Simulation(
        cell_size=mp.Vector3(case["r"], 0, case["z"]), resolution=case["resolution"],
        dimensions=mp.CYLINDRICAL, m=m, force_complex_fields=True,
        boundary_layers=PML,
        sources=[mp.Source(mp.GaussianSource(frequency=flux_case["fcen"],
                                             fwidth=flux_case["fwidth"]),
                           component=mp.Ez, center=mp.Vector3(*flux_case["center"]))])
    flux = sim.add_flux(flux_case["fcen"], flux_case["df"], flux_case["nfreq"],
                        mp.FluxRegion(center=mp.Vector3(*flux_case["plane_center"]),
                                      size=mp.Vector3(*flux_case["plane_size"])),
                        decimation_factor=1)
    sim.run(until=flux_case["until"])
    out[tag + "flux"] = np.asarray(mp.get_fluxes(flux))
    sim.reset_meep()
out["meep_version"] = np.array(mp.__version__)
np.savez(sys.argv[2], **out)
'''


@pytest.fixture(scope="module")
def cylindrical_m0_oracle(tmp_path_factory):
    if not _HAS_MEEP:
        pytest.skip("CPU MEEP is not installed")
    directory = tmp_path_factory.mktemp("cylindrical_m0_oracle")
    script = directory / "oracle.py"
    script.write_text(_M0_ORACLE, encoding="utf-8")
    output = directory / "fields.npz"
    environment = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    completed = subprocess.run(
        [sys.executable, str(script), json.dumps(_M0_CASE), str(output)],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert completed.returncode == 0 and output.exists(), (
        f"cylindrical m=0 oracle failed (exit {completed.returncode}).\n"
        f"stderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _run_m_engine(m, pml=False, complex_fields=True, component="Ez", center=None):
    driver = FdtdDriver(cell_size=(_M0_CASE["r"], 0.0, _M0_CASE["z"]),
                        resolution=_M0_CASE["resolution"], cylindrical=True, m=m,
                        force_complex_fields=complex_fields, boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    if pml:
        driver.setup_pml({"x": (0, 5), "z": (5, 5)})
    driver.add_source({"component": component, "frequency": _M0_CASE["frequency"],
                       "source_type": "gaussian", "fwidth": _M0_CASE["fwidth"],
                       "center": _M0_CASE["center"] if center is None else center,
                       "size": (0.0, 0.0, 0.0)})
    driver.run(until=(2 if pml else 1) * _M0_CASE["until"])
    fields = {
        "er": np.asarray(driver.get_field("Ex")).squeeze(),
        "ep": np.asarray(driver.get_field("Ey")).squeeze(),
        "ez": np.asarray(driver.get_field("Ez")).squeeze(),
        "hr": np.asarray(driver.get_field("Hx")).squeeze(),
        "hp": np.asarray(driver.get_field("Hy")).squeeze(),
        "hz": np.asarray(driver.get_field("Hz")).squeeze(),
        "hp_raw": np.asarray(driver.get_field("Hy", cell_centered=False)).squeeze(),
    }
    driver.close()
    return fields


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_cylindrical_m0_matches_cpu_meep(cylindrical_m0_oracle):
    """The m=0 cylindrical stepper reproduces CPU MEEP — the P1 exit measurement.

    An off-axis Ez point source in vacuum with MEEP's default metallic walls (a
    Dcyl run has no k_point machinery; z terminates PEC, r_max likewise, the axis
    by the m=0 rules). Every live component at the engine floor, measured on
    landing: Er **1.59e-07**, Ez **1.11e-07**, Hp **1.66e-07** cell-centred, and
    Hp RAW vs MEEP's ``snap=True`` **1.66e-07** — Hp is Dcyl's centred component
    (eps_component, vec.cpp:314-323), so that last comparison pins the
    registration with no interpolation in the loop at all. The Ez comparison is
    also the pin for the readback's far-wall rule: the periodic wrap the
    Cartesian metallic axes get away with (their wrapped row 0 is the zero-held
    wall) read 1.27e-01 here, because cylindrical row 0 is the LIVE axis row.
    Ep stays identically zero: at m=0 the (Ep, Hr, Hz) family is decoupled from
    an Ez source, and 0 + anything-real returns exact 0 in this engine.
    """
    ours = _run_m_engine(0)
    reference = {name: cylindrical_m0_oracle["m0_" + name].T
                 for name in ("er", "ez", "hp")}
    assert ours["er"].shape == reference["er"].shape
    for name in ("er", "ez", "hp"):
        error = _rel(ours[name], reference[name])
        assert error < 5e-6, f"m=0 {name} cell-centred rel L2 {error:.3e}"
    hp_raw = _rel(ours["hp_raw"], cylindrical_m0_oracle["m0_hp_snap"].T)
    assert hp_raw < 5e-6, f"Hp raw-vs-snap rel L2 {hp_raw:.3e}"
    assert float(np.abs(ours["ep"]).max()) == 0.0, (
        "Ep must stay exactly zero at m=0 under an Ez source (decoupled family)"
    )


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_cylindrical_m1_matches_cpu_meep(cylindrical_m0_oracle):
    """The m=1 stepper — i*m/r couplings and |m|=1 axis rules — reproduces CPU MEEP.

    The same source at m=1 couples all six components. Measured on landing:
    Er **1.25e-07**, Ep **1.37e-07**, Ez **9.0e-08**, Hr **1.30e-07**,
    Hp **1.60e-07** — the engine floor across every component with a live
    reference. Hz is the discrete-cancellation pin rather than an L2 row:
    MEEP's (1/r)d(r*Ep)/dr and (i*m/r)*Er cancel in its Bz update to float32
    roundoff (**5.5e-11** against an Ez scale of 1.1e-02), and this engine's do
    too (**5.3e-11**) — but only after the prefix sum's far wall was given
    MEEP's wall row instead of the generic zero ghost. That defect was born at
    the LAST TWO radial rows at step 3 (rows 0-17 exactly zero error), grew
    inward to 4.9e-01 by 60 steps, and never touched m=0, where Ep is
    identically zero — the exact shape of a silent wrong answer this suite
    exists to keep out.
    """
    ours = _run_m_engine(1)
    reference = {name: cylindrical_m0_oracle["m1_" + name].T
                 for name in ("er", "ep", "ez", "hr", "hp")}
    for name in ("er", "ep", "ez", "hr", "hp"):
        error = _rel(ours[name], reference[name])
        assert error < 5e-6, f"m=1 {name} cell-centred rel L2 {error:.3e}"
    ez_scale = float(np.abs(ours["ez"]).max())
    hz_ratio = float(np.abs(ours["hz"]).max()) / ez_scale
    assert hz_ratio < 1e-6, (
        f"Hz/Ez = {hz_ratio:.3e}: the discrete Bz cancellation (MEEP holds Hz at "
        f"roundoff, ~5e-09 of the field scale) is broken."
    )


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_cylindrical_m2_matches_cpu_meep(cylindrical_m0_oracle):
    """|m| > 1 — the near-axis zeroing hack — reproduces CPU MEEP, VERSION-AWARE.

    At |m| > 1 MEEP's default ``zero_fields_near_cylorigin`` holds field rows
    within |m| pixels of the axis at zero (its own stability hack,
    step_db.cpp:398-434) — and upstream FIXED the component set in #3164
    (593a4b42, first released in 1.33.0): the 1.29-era code zeroed only Dp/Dz
    and Br, the fix zeroes all six components plus their fu/f_cond auxiliaries.
    This engine follows the fix, and since 2026-08-04 the suite oracle IS a
    post-fix MEEP (pristine 1.33.0 single precision in the reference env), so the
    PRIMARY branch is the engine-floor assertion; the pre-1.33 branch is
    vestigial, kept for a frozen 1.29 regression oracle (the retired binary,
    archived 2026-08-04):

    * oracle >= 1.33 (post-#3164) — the live suite oracle, the primary branch:
      the engine floor. Measured on adopting the fix and re-measured unchanged
      at the 2026-08-06 re-baselining: Er 1.76e-07 / Ep 1.23e-07 / Ez 7.8e-08 /
      Hr 1.13e-07 / Hp 2.04e-07 against pristine 1.33.
    * oracle < 1.33 — VESTIGIAL (no such binary remains installed): the KNOWN
      bug-vs-fix delta, present and in band. Measured: Er 8.977e-02 /
      Ez 3.02e-02 / Hp 1.68e-01 — identical to every digit with
      pristine-1.33-vs-1.29 on the same config, which is the statement that the
      engine sits exactly ON the 1.33 reference rather than merely near it.
      Asserting the band (not the floor) keeps the fix's presence pinned under
      such an oracle: if the engine regressed to the 1.29 behaviour, the delta
      would VANISH and this branch would fail.

    Hz stays at float roundoff in both engines regardless (the discrete Bz
    cancellation the m=1 test pins).
    """
    ours = _run_m_engine(2)
    version = tuple(int(part) for part in
                    str(cylindrical_m0_oracle["meep_version"]).split(".")[:2])
    post_fix_oracle = version >= (1, 33)
    for name in ("er", "ep", "ez", "hr", "hp"):
        reference = cylindrical_m0_oracle["m2_" + name].T
        error = _rel(ours[name], reference)
        if post_fix_oracle:  # PRIMARY: the live suite oracle (1.33+).
            assert error < 5e-6, f"m=2 {name} vs post-#3164 oracle rel L2 {error:.3e}"
        elif name in ("er", "ez", "hp"):  # VESTIGIAL: a frozen pre-#3164 oracle.
            assert 1e-2 < error < 5e-1, (
                f"m=2 {name} vs pre-#3164 oracle rel L2 {error:.3e} is outside the known "
                f"bug-vs-fix band (measured 3.0e-02..1.7e-01); either the engine regressed "
                f"to the 1.29 zeroing (delta vanishes) or something else moved")
    hz_ratio = float(np.abs(ours["hz"]).max()) / float(np.abs(ours["ez"]).max())
    assert hz_ratio < 1e-6, f"m=2 Hz/Ez = {hz_ratio:.3e}: Bz cancellation broken"


def test_cylindrical_m2_zeroing_covers_all_six_components():
    """The |m| > 1 near-axis rows are zero in ALL SIX field components — the #3164 set.

    Oracle-free on purpose: the m=2 oracle test's pre-#3164 branch asserts a
    loose bug-vs-fix delta band that a PARTIAL revert (dropping one component
    from the zeroing set) does not leave — measured: the
    `cylindrical_m2_zeroing_missing_dr` mutation survived the band. This test
    kills that class directly: after real driven steps at m=2, rows [0:|m|] of
    Dx/Dy/Dz/Bx/By/Bz must be EXACTLY zero (the constraint is a hard overwrite,
    not a decay), while the interior rows carry live field (the liveness check
    that stops an all-zero run from passing vacuously).
    """
    driver = FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True, m=2,
                        force_complex_fields=True, boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.add_source({"component": "Ez", "frequency": 1.0, "source_type": "gaussian",
                       "fwidth": 0.5, "center": (0.75, 0.0, -0.55), "size": (0.0, 0.0, 0.0)})
    driver.run(until=1.0)
    rows = slice(0, 2)
    live = 0.0
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        volume = np.asarray(getattr(driver.fields, name))
        near_axis = float(np.abs(volume[rows]).max())
        assert near_axis == 0.0, (
            f"m=2 {name} rows [0:2] carry {near_axis:.3e}; the post-#3164 "
            f"zero_fields_near_cylorigin set zeroes all six components there")
        live = max(live, float(np.abs(volume[2:]).max()))
    driver.close()
    assert live > 0.0, "no field anywhere: the run was vacuous and asserts nothing"


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_accurate_fields_near_cylorigin_matches_cpu_meep_and_is_not_the_default():
    """MEEP's alternative |m| >= 2 near-axis branch, measured rather than transcribed.

    ``accurate_fields_near_cylorigin`` is the INVERSE of MEEP's internal
    ``zero_fields_near_cylorigin`` (python/simulation.py:2483) and it changes exactly
    one thing, for |m| >= 2 only: instead of holding every r row within ``|m|`` pixels
    of the axis at zero, it holds the ``r = 0`` row alone — an ordinary axis boundary
    condition and nothing else. MEEP's own comment says this "probably maintains
    2nd-order accuracy" at the cost of stability: it needs Courant <~ 1/(|m| + 0.5),
    which ``Grid`` REFUSES above rather than warning about, because a cylindrical
    instability is a smooth growing mode and not a crash.

    WHY THIS IS MEASURED AND NOT READ. The read-only MEEP checkout available here is
    v1.31.0-31-g0fdfd587, which predates upstream #3164 ("Fix bug in
    zero_fields_near_cylorigin for |m| >= 2", 593a4b42, first released in 1.33.0) — and
    in that tree BOTH branches zero only Dp/Dz on the D side and Br on the B side. The
    oracle this engine is measured against is 1.33.0, so the component set was settled
    by a discriminating experiment on the oracle itself rather than by transcription
    from the wrong version. Three candidate branches, one cell (3x3 Dcyl, resolution
    20, m = 2, PML 0.5, an off-axis Er Gaussian, Courant = 1/(|m| + 0.6) which is
    test_pml_cyl's own choice, 600 steps), against pristine CPU MEEP 1.33.0:

        all six components, r = 0 row only    7.5444e-07   <- MEEP's
        pre-#3164 set (Dp/Dz, Br), r = 0 row  6.0673e-03
        the DEFAULT |m| rows, flag set        2.0345e-02

    So #3164 widened the component set in BOTH branches, and the accurate branch is the
    same six components on a one-row slice. The two rejected candidates are kept in the
    assertions below as controls, because each is a complete, smooth, full-signal run
    with the same step count — nothing but the parity number separates them.
    """
    import meep as mp
    from .from_meep import gpu_compatibility, run_on_gpu

    m, resolution, steps = 2, 20, 600
    courant = 1.0 / (abs(m) + 0.6)  # test_pml_cyl.py's own choice, and inside the bound.
    until = (steps - 0.5) * (courant / resolution)

    def build():
        sim = mp.Simulation(
            cell_size=mp.Vector3(3.0, 0, 3.0), resolution=resolution,
            dimensions=mp.CYLINDRICAL, m=m, boundary_layers=[mp.PML(0.5)],
            accurate_fields_near_cylorigin=True,
            sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Er,
                               center=mp.Vector3(0.9, 0, 0))])
        sim.Courant = courant
        return sim

    verdict = gpu_compatibility(build())
    assert verdict.supported, f"the accurate branch is still refused: {verdict.reasons}"

    def measure():
        result = run_on_gpu(build(), until=until, prefer_gpu=False)
        reference = build()
        reference.run(until=until)
        ours, theirs = [], []
        for name, component in (("Ex", mp.Er), ("Ey", mp.Ep), ("Ez", mp.Ez),
                                ("Hx", mp.Hr), ("Hy", mp.Hp), ("Hz", mp.Hz)):
            mine = np.asarray(result.get_array(name), dtype=np.complex128)
            reference_array = np.asarray(
                reference.get_array(component=component, cmplx=True), dtype=np.complex128)
            assert mine.shape == reference_array.shape, (name, mine.shape)
            ours.append(mine.ravel())
            theirs.append(reference_array.ravel())
        result.close()
        ours, theirs = np.concatenate(ours), np.concatenate(theirs)
        matched = abs(reference.meep_time() - steps * reference.fields.dt) < 1e-9
        return (float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)),
                float(np.linalg.norm(theirs)), matched)

    error, signal, matched = measure()
    assert signal > 1.0, f"the oracle run carries no field ({signal:.3e}): nothing is pinned"
    assert matched, "the two runs did not step the same span"
    assert error < 5e-6, (
        f"accurate_fields_near_cylorigin at m={m} is {error:.4e} from CPU MEEP 1.33.0 "
        f"(measured 7.5444e-07 on landing, inside the band its three default-branch "
        f"siblings occupy: 6.3778e-06 / 1.3783e-06 / 1.5676e-06)")

    # CONTROL 1: the DEFAULT zeroing left in place while the flag is set — the wrong
    # answer this branch exists to avoid, and one that completes and looks healthy.
    original = stepping._cylindrical_axis_rows
    stepping._cylindrical_axis_rows = lambda grid: slice(0, abs(grid.m))
    try:
        default_error, _, default_matched = measure()
    finally:
        stepping._cylindrical_axis_rows = original
    assert default_matched and default_error > 1e-3, (
        f"keeping the DEFAULT |m|-row zeroing while accurate_fields_near_cylorigin is "
        f"set measured {default_error:.4e}; it must disagree (7.5444e-07 correct against "
        f"2.0345e-02 measured for this control), or this test is not about the branch.")


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_accurate_fields_near_cylorigin_refuses_an_unstable_courant():
    """The bound is enforced, not documented — an unstable Dcyl run does not crash.

    Without the default's zero rows the near-axis update is stable only for
    Courant <~ 1/(|m| + 0.5) (MEEP's own comment on the branch). Above it the field
    grows smoothly and the run returns something field-shaped, so the Grid refuses at
    construction and the converter's gate names the same bound before anything is
    built. The flag must also be an exact NO-OP for m = 0 and |m| = 1, which
    ``step_db.cpp`` reaches through entirely separate branches.
    """
    with pytest.raises(ValueError, match=r"1/\(\|m\| \+ 0.5\)"):
        FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True, m=3,
                   courant=0.5, accurate_fields_near_cylorigin=True,
                   force_complex_fields=True)
    for m in (0, 1):
        # No bound at all below |m| = 2, because the flag reaches no branch there.
        driver = FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True,
                            m=m, courant=0.5, accurate_fields_near_cylorigin=True,
                            force_complex_fields=True)
        assert driver.grid.accurate_fields_near_cylorigin
        driver.close()


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
@pytest.mark.parametrize("m", [0, 1])
def test_cylindrical_pml_matches_cpu_meep(cylindrical_m0_oracle, m):
    """Cylindrical PML — r high face + both z faces — at the engine floor.

    MEEP's cylindrical PML is the ordinary graded sigma (no radial stretch);
    the r layer is high-side only because the axis is not a boundary. Measured
    on landing at twice the vacuum runtime (so absorbed-and-returned energy is
    in the comparison): m=0 Er 3.0e-07 / Ez 1.7e-07 / Hp 3.6e-07; m=1 all five
    live components 2.2e-07..3.8e-07.

    m=1 here is the pin for two defects found while landing it, both recorded
    in task history and the mutation battery: the |m|=1 axis-rule increments
    must ride the PML recurrences with their own ladder axes (plain post-adds
    measured Ep 4.6e-01), and ``PML.axis_wraps`` must defer to the grid's one
    definition — a restated ``not is_mirrored`` wrap-graded the axis row's
    integer position 0 with the outer wall's peak sigma (kps[0] = 2.295), which
    nothing at m=0 reads (Hp's ladder axis is phi) and the m=1 stored Hr read
    at 2.3x and growing.
    """
    ours = _run_m_engine(m, pml=True)
    live = ("er", "ez", "hp") if m == 0 else ("er", "ep", "ez", "hr", "hp")
    for name in live:
        reference = cylindrical_m0_oracle[f"m{m}p_" + name].T
        error = _rel(ours[name], reference)
        assert error < 5e-6, f"m={m}+PML {name} cell-centred rel L2 {error:.3e}"


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_cylindrical_m0_real_fields_match_cpu_meep(cylindrical_m0_oracle):
    """m=0 runs in REAL float32 storage and still lands on MEEP — P3.

    MEEP's own storage rule (simulation.py:2517-2522, fields.cpp:824): real
    fields unless force_complex_fields, or cylindrical with m != 0. Every m=0
    term this engine steps — the r-derivative prefix, the axis rules, the PML
    ladders — is real arithmetic (the i*m/r coupling carries the only i, and
    its factor is m), so m=0 honours the request rather than silently promoting.
    Measured on landing against a real-field MEEP run: Er 2.3e-07, Ez 1.7e-07,
    Hp 1.5e-07, in float32 throughout.
    """
    ours = _run_m_engine(0, complex_fields=False)
    for name in ("er", "ez", "hp"):
        assert ours[name].dtype == np.float32, (
            f"m=0 real run stored {name} as {ours[name].dtype}; the request was float32")
        reference = cylindrical_m0_oracle["m0r_" + name].T
        error = _rel(ours[name], reference)
        assert error < 5e-6, f"m=0 real-fields {name} rel L2 {error:.3e}"


def test_cylindrical_nonzero_m_refuses_real_fields_by_name():
    """m != 0 with real storage refuses, naming the i*m/r coupling.

    MEEP promotes the same request silently (fields ARE complex at m != 0;
    change_m aborts only when re-assigning m onto a real-field run). This
    engine takes an explicit flag, so the honest answer to an impossible
    request is the refusal, not a silent override — the lift translates
    MEEP's (m != 0, force_complex_fields=False) default to complex itself.
    """
    with pytest.raises(ValueError, match="complex fields"):
        FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True, m=1,
                   force_complex_fields=False, boundaries={"z": "metallic"})


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
@pytest.mark.parametrize("tag,m,component,live", [
    ("a_er1_", 1, "Ex", ("er", "ep", "ez", "hr", "hp", "hz")),
    ("a_ez0_", 0, "Ez", ("er", "ez", "hp")),
    ("a_ep1_", 1, "Ey", ("er", "ep", "ez", "hr", "hp", "hz")),
])
def test_cylindrical_axis_sources_match_cpu_meep(cylindrical_m0_oracle, tag, m, component, live):
    """Sources placed AT r=0 inject what MEEP injects — the P3 axis-source pin.

    The rule under test is MEEP's little_owned_corner (vec.cpp:432-436): in Dcyl
    with the r origin at 0, an integer-r component's owned corner claws back
    from doubled 2 to 0 — the axis row of Ez/Ep IS owned, stepped by the per-m
    axis rules. Transcribing only the generic little_owned_corner0 clipped an
    axis source on those components to NOTHING: the run completed and every
    component came back identically zero (rel L2 exactly 1.0) — the shape of
    silent no-op this suite exists to catch. Er is r-half-shifted, so its
    below-axis ladder rung folds through the r -> -r image machinery instead;
    it was measured right before the fix and pins that fold. On landing: Er@axis
    m=1 all six components 0.8-1.7e-07; Ez@axis m=0 Er/Ez/Hp 1.3-2.3e-07;
    Ep@axis m=1 all six 0.9-1.6e-07.
    """
    ours = _run_m_engine(m, component=component, center=(0.0, 0.0, -0.55))
    for name in live:
        reference = cylindrical_m0_oracle[tag + name].T
        norm = float(np.linalg.norm(reference))
        if norm < 1e-30:
            continue
        error = _rel(ours[name], reference)
        assert error < 5e-6, f"{tag}{name} rel L2 {error:.3e}"


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
@pytest.mark.parametrize("m", [0, 1])
def test_cylindrical_flux_matches_cpu_meep(cylindrical_m0_oracle, m):
    """Transmitted flux through a z-normal plane matches MEEP's ``add_flux`` — P3.

    The cylindrical area element: MEEP weights every flux sample by the
    circumference of its own ring — loop_in_chunks.cpp:505-515 sets
    ``dV = dV0 + dV1*iloopR`` with the 2*pi*r factor evaluated at the loop's
    starting centered-grid point and ``dV1 = dV0_cartesian * 2*pi*inva``
    advancing it per r row. Without that factor in the monitor's weight ladder
    this plane (r in [0, 1.2]) integrated plain dr and measured 0.47-0.49x
    MEEP; with it, both m at the engine floor on landing: m=0 rel err
    [8.5e-9, 1.0e-7, 6.9e-8] across the three frequencies, m=1
    [1.1e-8, 5.8e-9, 1.1e-7]. m=1 is the pin for the second component pair of
    cylindrical Sz (Er*conj(Hp) - Ep*conj(Hr), dft.cpp:625-629): an Ez source
    at m=0 leaves Ep and Hr identically zero, so only m=1 weighs that pair's
    -1. Decimation is pinned to 1 on both sides — see the case comment.
    """
    flux_case = _M0_CASE["flux"]
    driver = FdtdDriver(cell_size=(_M0_CASE["r"], 0.0, _M0_CASE["z"]),
                        resolution=_M0_CASE["resolution"], cylindrical=True, m=m,
                        force_complex_fields=True, boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.setup_pml({"x": (0, 5), "z": (5, 5)})
    driver.add_source({"component": "Ez", "frequency": flux_case["fcen"],
                       "source_type": "gaussian", "fwidth": flux_case["fwidth"],
                       "center": flux_case["center"], "size": (0.0, 0.0, 0.0)})
    monitor = driver.add_flux_monitor(
        fcen=flux_case["fcen"], df=flux_case["df"], nfreq=flux_case["nfreq"],
        center=flux_case["plane_center"], size=flux_case["plane_size"])
    driver.run(until=flux_case["until"])
    ours = np.asarray(monitor.get_flux_spectrum())
    driver.close()
    reference = cylindrical_m0_oracle[f"f{m}_flux"]
    for index, (mine, theirs) in enumerate(zip(ours, reference)):
        error = abs(mine - theirs) / abs(theirs)
        assert error < 5e-6, (
            f"m={m} flux frequency {index}: ours {mine:.8e} meep {theirs:.8e} "
            f"rel err {error:.3e}")


def test_cylindrical_flux_weights_integrate_ring_area():
    """The flux weight ladder integrates 2*pi*r dr — exactly, where exactness holds.

    Two planes, no MEEP in the loop. An interior plane (r in [0.3, 1.1]) must
    integrate f=1 to the annulus area pi*(1.1^2 - 0.3^2) EXACTLY: the taper
    weights integrate linear functions exactly and 2*pi*r is linear in r, so
    any deviation is a defect, not quadrature. An axis-touching plane
    (r in [0, 1.2]) is NOT the exact ring area: MEEP's chunk intersection
    drops the ladder's below-axis point but keeps the s1 taper on the first
    owned row, leaving a +pi*dr^2/8 excess over pi*1.2^2 — measured
    4.527820 against MEEP's own ``integrate_field_function`` on this exact
    grid (analytic 4.523893), and the clipped-ladder model reproduces it to
    1e-5. Matching MEEP's clip, not the textbook integral, is the contract.
    """
    driver = FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True,
                        m=0, force_complex_fields=True, boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    interior = driver.add_flux_monitor(fcen=1.0, df=0.4, nfreq=1,
                                       center=(0.7, 0.0, 0.8), size=(0.8, 0.0, 0.0))
    touching = driver.add_flux_monitor(fcen=1.0, df=0.4, nfreq=1,
                                       center=(0.6, 0.0, 0.8), size=(1.2, 0.0, 0.0))
    results = {}
    for name, monitor in (("interior", interior), ("touching", touching)):
        weights = np.asarray(monitor._weights)
        results[name] = float(weights.sum() * monitor._measure)
    driver.close()
    exact = np.pi * (1.1**2 - 0.3**2)
    assert abs(results["interior"] - exact) < 1e-5, (
        f"interior plane integrates {results['interior']:.6f}, annulus area {exact:.6f}")
    assert abs(results["touching"] - 4.527820) < 1e-4, (
        f"axis-touching plane integrates {results['touching']:.6f}, "
        f"MEEP's clipped-ladder value 4.527820")


def test_cylindrical_grid_declares_meeps_dcyl_conventions():
    """The cylindrical Grid states MEEP's volcyl facts as declarations.

    r maps to axis 0 with its origin at EXACTLY 0 (volcyl io=(0,0);
    center_origin shifts z only, vec.cpp:722-730), phi is the one-cell invariant
    axis (the exp(i*m*phi) dependence is analytic), z stays centred. The r
    boundary pair is (AXIS, METALLIC) — no boundary at the axis (has_boundary
    Dcyl, vec.cpp:465-473), a PEC wall at r_max — and r neither wraps nor
    counts as a metallic axis for the wall-zeroing machinery (the axis row
    belongs to the per-m rules).
    """
    grid = Grid(resolution=10, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=0)
    assert grid.shape == (20, 1, 40)
    assert grid.dimensions == 2  # MEEP's own normalization of CYLINDRICAL.
    assert grid.invariant_axes == (False, True, False)
    assert grid.boundaries[0] == (AXIS, METALLIC)
    assert grid.axis_origin(0) == 0.0 and grid.axis_origin(2) == -2.0
    assert grid.is_axis(0) and not grid.is_axis(1) and not grid.is_axis(2)
    assert [grid.axis_wraps(a) for a in range(3)] == [False, True, True]
    assert grid.metallic_axes == (False, False, False)


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_cylindrical_script_lifts_through_run_on_gpu():
    """A whole cylindrical MEEP script — geometry, PML, flux — lifts as written. P4.

    The corpus-facing exit test: ``run_on_gpu`` on a Dcyl script with a
    dielectric ring (eps=4, MEEP's own subpixel smoothing), an m=1 off-axis
    source, r-high + z PML, and a migrated ``add_flux`` plane. Everything the
    lift had to learn is on this one path: mp.R as a PML direction, Er/Ep/Ez as
    source components, the (nr, nz) grid with no duplicate planes, iveccyl
    permittivity sampling (get_array_metadata refuses Dcyl, so the epsilon
    window is registration-derived), and MEEP's (z, r) get_array order.
    Measured on landing: fields 1.9-3.7e-07, migrated flux vs mp.get_fluxes at
    the same 1e-7 class (vacuum probe: [8.5e-9, 1.0e-7, 6.9e-8]).
    """
    import meep as mp
    from .from_meep import run_on_gpu

    geometry = [mp.Block(center=mp.Vector3(0.7, 0, 0.0), size=mp.Vector3(0.4, mp.inf, 0.6),
                         material=mp.Medium(epsilon=4.0))]

    def build():
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, 0, 4.0), resolution=10, dimensions=mp.CYLINDRICAL,
            m=1, force_complex_fields=True, geometry=geometry,
            boundary_layers=[mp.PML(0.5, direction=mp.R, side=mp.High),
                             mp.PML(0.5, direction=mp.Z)],
            sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.4),
                               component=mp.Ez, center=mp.Vector3(0.4, 0, -0.55))])
        flux = sim.add_flux(1.0, 0.4, 3,
                            mp.FluxRegion(center=mp.Vector3(0.6, 0, 0.8),
                                          size=mp.Vector3(1.2, 0, 0)),
                            decimation_factor=1)
        return sim, flux

    sim, flux = build()
    result = run_on_gpu(sim, until=6.0, prefer_gpu=False)
    try:
        ours = {name: np.asarray(result.get_array(engine))
                for name, engine in (("Ez", "Ez"), ("Er", "Ex"), ("Hp", "Hy"))}
        our_flux = np.asarray(result.get_flux_spectrum(flux))
    finally:
        result.close()

    reference_sim, reference_flux = build()
    reference_sim.run(until=6.0)
    for name, mine in ours.items():
        theirs = np.asarray(reference_sim.get_array(component=getattr(mp, name), cmplx=True))
        assert mine.shape == theirs.shape, (
            f"{name}: lifted get_array is {mine.shape}, MEEP's Dcyl (z, r) array is "
            f"{theirs.shape}")
        error = _rel(mine, theirs)
        assert error < 5e-6, f"lifted {name} rel L2 {error:.3e}"
    theirs_flux = np.asarray(mp.get_fluxes(reference_flux))
    for index, (mine, theirs) in enumerate(zip(our_flux, theirs_flux)):
        error = abs(mine - theirs) / abs(theirs)
        assert error < 5e-6, (
            f"migrated flux frequency {index}: ours {mine:.8e} meep {theirs:.8e} "
            f"rel err {error:.3e}")


def test_cylindrical_near2far_radiates_rings_not_the_cartesian_dyadic():
    """A cylindrical near field goes through the ring Green's function, not the point one.

    MEEP's cylindrical near2far integrates the Green's function over each source
    RING numerically (greencyl, near2far.cpp:275-349); the Cartesian dyadic with
    any scalar reweighting is a different — confidently wrong — far field,
    measured 5.9 relative against CPU MEEP on the same near data. The monitor now
    implements the ring integral; the whole transformation, its parity against
    ``sim.get_farfield`` and the controls that would catch the Cartesian answer
    live in ``test_cylindrical_near2far.py``. What is asserted here is only that
    a cylindrical grid selects the ring path at all.
    """
    from .dft import Near2FarMonitor, _cylinder_regions

    grid = Grid(resolution=10, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=0)
    monitor = Near2FarMonitor(grid, 1.0, _cylinder_regions(1.2, 1.0, 0.8), closed=True)
    assert monitor.cylindrical and monitor.greencyl_tol == 1e-3
    assert monitor._radiated_stack.__self__ is monitor
    positions, currents, magnetic = monitor._surface_currents()
    point = np.array([[6.0, 0.0, 9.0]])
    # Nothing has been accumulated, so both paths return zeros — the statement is
    # that the dispatch picked the ring integral, which the far point's (r, 0, z)
    # spelling is enough to show: a Cartesian read would accept a nonzero phi.
    assert monitor._greencyl(positions, currents, magnetic, point).shape == (1, 1, 6)
    with pytest.raises(ValueError, match="phi = 0 half-plane"):
        monitor.farfield((6.0, 1.0, 9.0))


def test_cylindrical_grid_refusals_are_by_name():
    """Each constraint the mode cannot serve raises naming the constraint."""
    build = dict(resolution=10, cell_size=(2.0, 0.0, 4.0), cylindrical=True)
    with pytest.raises(ValueError, match="single-valued"):
        Grid(m=0.5, **build)
    with pytest.raises(ValueError, match="phi extent"):
        Grid(resolution=10, cell_size=(2.0, 1.0, 4.0), cylindrical=True)
    with pytest.raises(ValueError, match="mirror symmetry"):
        Grid(symmetry=(Mirror("Z"),), **build)
    with pytest.raises(ValueError, match="Z-only"):
        Grid(k_point=(0.1, 0.0, 0.0), **build)
    with pytest.raises(ValueError, match="boundary pair"):
        Grid(boundaries={"x": "metallic"}, **build)
    with pytest.raises(ValueError, match="azimuthal"):
        Grid(resolution=10, cell_size=(2.0, 2.0, 2.0), m=1)


def test_rderiv_prefix_sum_matches_meeps_explicit_loop():
    """The vectorized prefix sum equals step_db.cpp:99-119 written out row by row.

    MEEP's loop, transcribed verbatim (ir0 folded in as the code does)::

        f_rderiv_int[0, iz] = 0
        f_rderiv_int[ir, iz] = f_rderiv_int[ir-1, iz]
            + (1/((ir+ir0)-0.5)) * (f_p[ir,iz]*(ir+ir0) - f_p[ir-1,iz]*((ir-1)+ir0))

    Random complex Fp over a (nr, 1, nz) volume, both ir0 = 0.5 (a cell touching
    the axis, Fp with r-shift 1 — the Dz/Hp case) and ir0 = 3.5 (a cell whose
    origin sits off the axis). Agreement bar 1e-6 relative in float32 — the
    cumsum's summation order differs from the fused loop only in rounding.
    """
    rng = np.random.default_rng(20260803)
    for ir0 in (0.5, 3.5):
        f_p = (rng.standard_normal((7, 1, 5))
               + 1j * rng.standard_normal((7, 1, 5))).astype(np.complex64)
        expected = np.zeros_like(f_p, dtype=np.complex128)
        for ir in range(1, f_p.shape[0]):
            expected[ir] = expected[ir - 1] + (
                f_p[ir] * (ir + ir0) - f_p[ir - 1] * ((ir - 1) + ir0)
            ) / ((ir + ir0) - 0.5)
        result = stepping.cylindrical_rderiv_prefix(np, f_p, ir0)
        scale = float(np.abs(expected).max())
        assert scale > 0.0
        worst = float(np.abs(result - expected).max()) / scale
        assert worst < 1e-6, f"ir0={ir0}: prefix sum diverges from MEEP's loop by {worst:.3e}"
        assert result.dtype == f_p.dtype, "the prefix sum must keep the field dtype"
