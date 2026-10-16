"""The CuPy array path stores the NumPy array path's words after every sub-step.

The NVIDIA kernel tables are certified against the CuPy array path on their own device,
while the array-path oracle and MEEP identity are measured on the NumPy array path
(``docs/design/floating-point.md``, "What "bit-identical" is identity with"). This module
is the link between the two, on a CUDA host. Two drivers hold the same seeded state, one
on NumPy and one on CuPy (``prefer_gpu=True``; the suites pin ``MEEP_GPU_DISPATCH=0``,
so it steps the CuPy array path). Each takes the real ``FdtdDriver.step()``. The sub-steps
``driver.py`` runs (``step_B``, each source's ``inject``, the fills and wall clears,
MEEP's start of H, ``update_H``, ``step_D``, the ``f_minus_p`` brackets, ``update_E``,
``update_P``) are wrapped where the driver looks them up, and after each one every array
the capsule pins (``capsule.state_arrays``: the fields, the PML auxiliaries and
coefficient tables, the polarizations, the material and source tables) is compared as
32-bit words, the sign of zero included.

The cases are built without MEEP, and their state is written as float32 bit patterns:
normal words with planted +0, -0 and subnormal words, and a family of states near the
bottom of the float32 range, where most operations of every sub-step meet a subnormal.
They cover real and complex storage; the layer forms on both region routes (NumPy
selects on the innermost resolved axis, CuPy steps views on every axis); a division that
is not separable (both take the mask route); conductivity; stored-current sources, one
of them integrated; mirror planes; the off-diagonal row; dispersion; chi2 and chi3;
beta; BFAST; and Bloch phases.

THE POLICY. ``keep`` is installed on the host and on CuPy before anything compiles; both
NVIDIA tables are certified under it. Under ``flush`` the host's only lever on x86 is
MEEP's ``set_zero_subnormals`` and CuPy needs ``CUPY_ACCELERATORS=''`` before it is
imported, so the flush leg is measured outside the suite.

THE CYLINDRICAL ROWS. The radial derivative of Bz and Dz is the difference of MEEP's
per-chunk radial sum (``stepping.cylindrical_rderiv_difference``), which NumPy forms as a
``cumsum`` per chunk segment, a sequential accumulation, and every other array module as
a row loop, so CuPy's parallel scan is never used and the two round alike.

Needs CuPy and a CUDA device: ``requires_resource("cupy")``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy
import pytest

from . import capsule
from . import driver as driver_module
from . import sources as sources_module
from .backends import cupy_available
from .chunk_map import ChunkLayout, grid_frames
from .dispersion import DRUDE, LORENTZIAN, Susceptibility
from .driver import FdtdDriver
from .grid import Mirror

pytestmark = pytest.mark.requires_resource("cupy")

#: The sub-step functions ``driver.step`` calls by the names ``driver.py`` imports.
SUB_STEPS = ("step_B", "step_D", "update_H", "update_E", "update_P",
             "fill_symmetry_bc_B", "fill_symmetry_bc_D", "zero_metal_B", "zero_metal_D",
             "fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D", "separate_h_from_b",
             "f_minus_p_begin", "f_minus_p_end")
STEPS = 3
E_NAMES = ("Ex", "Ey", "Ez")

#: Every case is resolution 8 with Courant 0.5 (dt = 2**-4, dx = 2**-3).
CASES: Dict[str, Dict[str, Any]] = {
    "real_pml_3d": dict(cell=(1.5, 1.25, 1.0), real=True, pml={"x": (2, 3), "z": (0, 2)}),
    "real_pml_2d_sources": dict(
        cell=(2.0, 2.0, 0.0), dims=2, real=True, pml={"x": 3, "y": 3},
        sources=(("Ez", (-0.8125, 0.0625, 0.0), (0.0, 0.0, 0.0), False),
                 ("Ex", (0.0, 0.25, 0.0), (0.75, 0.0, 0.0), False),
                 ("Hz", (0.1875, -0.1875, 0.0), (0.0, 0.0, 0.0), False),
                 ("Ey", (0.125, 0.0, 0.0), (0.0, 0.0, 0.0), True))),
    "real_conductive_pml_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, real=True, pml={"y": 2},
                                   d_conductivity=True, b_conductivity=True),
    "real_folded_pml_3d": dict(cell=(1.5, 1.25, 1.0), real=True, symmetry=(("X", 1), ("Y", -1)),
                               pml={"x": (0, 2), "y": (0, 2), "z": 2}),
    "real_offdiag_pml_3d": dict(cell=(1.5, 1.25, 1.25), real=True, pml={"x": 2, "y": 2},
                                offdiagonal=True),
    "real_dispersive_pml_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, real=True, pml={"y": 2},
                                   susceptibilities=("lorentz", "drude")),
    "real_nonlinear_pml_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, real=True, pml={"y": 2},
                                  nonlinear=True),
    "real_nonseparable_pml_3d": dict(cell=(1.5, 1.5, 1.5), real=True, pml=2, nonseparable=True),
    "complex_bloch_pml_2d": dict(cell=(2.0, 1.5, 0.0), dims=2, k=(0.2, 0.0, 0.0),
                                 pml={"y": (1.5, 2.5)}),
    "complex_offdiag_pml_3d": dict(cell=(1.5, 1.25, 1.25), k=(0.0, 0.0, 0.25),
                                   pml={"x": 2, "y": 2}, offdiagonal=True),
    "complex_sources_pml_2d": dict(
        cell=(2.0, 2.0, 0.0), dims=2, k=(0.2, 0.0, 0.0), pml={"y": 3},
        sources=(("Ez", (-0.8125, 0.0625, 0.0), (0.0, 0.0, 0.0), False),
                 ("Hz", (0.1875, -0.1875, 0.0), (0.0, 0.0, 0.0), False),
                 ("Ey", (0.125, 0.0, 0.0), (0.0, 0.0, 0.0), True))),
    "complex_conductive_dispersive_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, k=(0.15, 0.0, 0.0),
                                             pml={"y": 2}, d_conductivity=True,
                                             susceptibilities=("lorentz",)),
    "complex_beta_pml_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, beta=0.375, pml={"x": 2, "y": 2}),
    "complex_bfast_pml_2d": dict(cell=(1.5, 1.25, 0.0), dims=2, bfast=(0.25, 0.125, 0.0),
                                 pml={"y": 2}),
    "tiny_real_pml_3d": dict(cell=(1.5, 1.25, 1.0), real=True, tiny=True,
                             pml={"x": (2, 3), "z": (0, 2)}),
    "tiny_real_sources_dispersive_2d": dict(
        cell=(2.0, 2.0, 0.0), dims=2, real=True, tiny=True, pml={"y": 3},
        susceptibilities=("lorentz",),
        sources=(("Ez", (-0.8125, 0.0625, 0.0), (0.0, 0.0, 0.0), False),
                 ("Ey", (0.125, 0.0, 0.0), (0.0, 0.0, 0.0), True))),
    "tiny_real_offdiag_conductive_3d": dict(cell=(1.5, 1.25, 1.25), real=True, tiny=True,
                                            pml={"x": 2}, offdiagonal=True, d_conductivity=True),
    "tiny_complex_bloch_pml_2d": dict(cell=(2.0, 1.5, 0.0), dims=2, tiny=True, k=(0.2, 0.0, 0.0),
                                      pml={"y": (1.5, 2.5)}),
    "cylindrical_real_m0_pml": dict(cell=(2.0, 0.0, 1.5), real=True, cylindrical=True, m=0,
                                    pml={"x": (0, 2), "z": 2}),
    "cylindrical_complex_m1_pml": dict(cell=(2.0, 0.0, 1.5), cylindrical=True, m=1,
                                       pml={"x": (0, 2), "z": 2}),
}


def _bits(seed: Tuple[str, ...], shape: Tuple[int, ...], tiny: bool) -> numpy.ndarray:
    """A float32 state volume as bit patterns, with planted +0, -0 and subnormal words.

    Normal words have exponents 2**-6 to 2**1 (``tiny``: 2**-126 to 2**-119); about 2 %
    (``tiny``: 5 %) are each +0 and -0, and 1 % (``tiny``: 20 %) are subnormal.
    """
    generator = numpy.random.Generator(numpy.random.PCG64(list(map(ord, "/".join(seed)))))
    words = generator.integers(0, 2 ** 32, size=int(numpy.prod(shape)), dtype=numpy.uint64)
    mantissa = words & numpy.uint64(0x7FFFFF)
    low = 1 if tiny else 121
    exponent = numpy.uint64(low) + ((words >> numpy.uint64(23)) & numpy.uint64(7))
    sign = (words >> numpy.uint64(31)) & numpy.uint64(1)
    bits = (sign << numpy.uint64(31)) | (exponent << numpy.uint64(23)) | mantissa
    selector = (words >> numpy.uint64(32)) & numpy.uint64(0xFF)
    subnormal, positive, negative = (51, 64, 77) if tiny else (3, 8, 13)
    bits = numpy.where(selector < subnormal, (sign << numpy.uint64(31)) | (mantissa | numpy.uint64(1)), bits)
    bits = numpy.where((selector >= subnormal) & (selector < positive), numpy.uint64(0), bits)
    bits = numpy.where((selector >= positive) & (selector < negative), numpy.uint64(0x80000000), bits)
    return bits.astype(numpy.uint32).view(numpy.float32).reshape(shape)


def _volume(seed: Tuple[str, ...], shape: Tuple[int, ...], low: float, high: float,
            zero_fraction: float = 0.0) -> numpy.ndarray:
    generator = numpy.random.Generator(numpy.random.PCG64(list(map(ord, "/".join(seed)))))
    values = generator.uniform(low, high, size=shape).astype(numpy.float32)
    if zero_fraction:
        values[generator.uniform(size=shape) < zero_fraction] = numpy.float32(0.0)
    return values


def _current(index: int):
    """A source waveform with no libm call, the same function for both drivers."""
    def current(time: float) -> complex:
        return complex(0.5 + 0.25 * index - 0.125 * time, 0.0625 * time - 0.25)
    return current


def _stored_volumes(fields: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        for prefix in ("", "fu_", "f_cond_", "f_bfast_", "f_w_"):
            array = getattr(fields, prefix + component, None)
            if array is not None:
                out[prefix + component] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in state.P.items():
            out[f"P[{index}].{component}"] = array
        for component, array in state.P_prev.items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def _nonseparable(driver: Any) -> ChunkLayout:
    """The one-rank division with the x layer chunks one cell deeper in the high-y boxes only."""
    base = ChunkLayout.one_rank(driver.grid, driver.pml)
    xs = sorted({lo[0] for lo, _ in base.boxes} | {hi[0] for _, hi in base.boxes})
    ys = sorted({lo[1] for lo, _ in base.boxes} | {hi[1] for _, hi in base.boxes})
    remap = {xs[1]: xs[1] + 2, xs[2]: xs[2] - 2}
    boxes = [((lo, hi) if lo[1] < ys[-2] else
              ((remap.get(lo[0], lo[0]), lo[1], lo[2]), (remap.get(hi[0], hi[0]), hi[1], hi[2])))
             for lo, hi in base.boxes]
    return ChunkLayout(boxes, base.pml_flags, grid_frames(driver.grid), faces=base.faces, source="record")


def _build(name: str, prefer_gpu: bool) -> FdtdDriver:
    spec = CASES[name]
    driver = FdtdDriver(
        cell_size=spec["cell"], resolution=8.0, courant=0.5,
        force_complex_fields=not spec.get("real", False),
        symmetry=tuple(Mirror(axis, phase) for axis, phase in spec.get("symmetry", ())),
        k_point=spec.get("k", (0.0, 0.0, 0.0)), dimensions=spec.get("dims", 3),
        cylindrical=spec.get("cylindrical", False), m=spec.get("m", 0),
        beta=spec.get("beta", 0.0), bfast_scaled_k=spec.get("bfast", (0.0, 0.0, 0.0)),
        prefer_gpu=prefer_gpu)
    shape = tuple(int(n) for n in driver.grid.shape)
    epsilon = {c: _volume((name, "eps", c), shape, 1.0, 3.0) for c in E_NAMES}
    offdiagonal = None
    if spec.get("offdiagonal"):
        offdiagonal = {row: {partner: _volume((name, "chi1inv", row, partner), shape, -0.0625,
                                              0.0625, zero_fraction=0.4)
                             for partner in E_NAMES if partner != row} for row in E_NAMES}
    driver.set_epsilon_components(epsilon, chi1inv_offdiagonal=offdiagonal)
    if spec.get("d_conductivity"):
        driver.set_conductivity({c: _volume((name, "sigma_d", c), shape, 0.0, 2.0, 0.3)
                                 for c in ("Dx", "Dy", "Dz")})
    if spec.get("b_conductivity"):
        driver.set_b_conductivity({c: _volume((name, "sigma_b", c), shape, 0.0, 2.0, 0.3)
                                   for c in ("Bx", "By", "Bz")})
    for index, kind in enumerate(spec.get("susceptibilities", ())):
        term = Susceptibility(frequency=(3.0 if kind == "lorentz" else 2.0) / (2.0 * numpy.pi),
                              gamma=0.0625 * (index + 1), kind=LORENTZIAN if kind == "lorentz" else DRUDE)
        driver.add_susceptibility(term, {c: _volume((name, "pol", str(index), c), shape, 0.0, 1.0, 0.3)
                                         for c in E_NAMES})
    if spec.get("nonlinear"):
        driver.set_chi2({c: _volume((name, "chi2", c), shape, 0.0, 0.03125, 0.3) for c in E_NAMES})
        driver.set_chi3({c: _volume((name, "chi3", c), shape, 0.0, 0.03125, 0.3) for c in E_NAMES})
    driver.setup_pml(spec["pml"])
    if spec.get("nonseparable"):
        driver.set_chunk_layout(_nonseparable(driver))
    for index, (component, center, size, integrated) in enumerate(spec.get("sources", ())):
        driver.add_source({"component": component, "center": center, "size": size,
                           "source_type": "custom", "src_func": _current(index),
                           "is_integrated": integrated})
    for key, array in _stored_volumes(driver.fields).items():
        host_shape = tuple(int(n) for n in array.shape)
        host = numpy.empty(host_shape, dtype=array.dtype)
        if host.dtype.kind == "c":
            host.real[...] = _bits((name, key, "re"), host_shape, spec.get("tiny", False))
            host.imag[...] = _bits((name, key, "im"), host_shape, spec.get("tiny", False))
        else:
            host[...] = _bits((name, key), host_shape, spec.get("tiny", False))
        array[...] = driver.xp.asarray(host)
    return driver


def _words(array: numpy.ndarray) -> numpy.ndarray:
    flat = numpy.ascontiguousarray(array).reshape(-1)
    if flat.dtype.itemsize % 4:
        return flat.view(numpy.uint8)
    return flat.view(numpy.uint32)


def _differing(left: Dict[str, numpy.ndarray], right: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    out: Dict[str, Any] = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        assert (a.dtype, a.shape) == (b.dtype, b.shape), key
        count = int(numpy.count_nonzero(_words(a) != _words(b)))
        if count:
            out[key] = count
    return out


class _Recorder:
    def __init__(self) -> None:
        self.driver: Any = None
        self.snapshots: List[Tuple[str, Dict[str, numpy.ndarray]]] = []
        self.depth = 0

    def wrap(self, label: str, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                self.depth -= 1
            if self.depth == 0 and self.driver is not None:
                name = label if label != "inject" else f"inject[{args[0].field_type}]"
                self.snapshots.append((name, capsule.state_arrays(self.driver)))
            return result
        return wrapped


@pytest.fixture(scope="module")
def keep_on_cupy():
    """CuPy and a CUDA device, with the keep policy on the host and on CuPy."""
    if not cupy_available():
        pytest.skip("CuPy is not installed or no CUDA device is visible")
    from . import subnormal_policy  # noqa: PLC0415

    before = os.environ.get("CUPY_CACHE_DIR")
    pointed = subnormal_policy.point_cupy_cache_at_keep_policy()
    assert not pointed["provenance"]["reasons"], pointed
    stamp = subnormal_policy.install_subnormal_policy("keep", executors=("host", "cupy"))
    yield stamp
    subnormal_policy.uninstall_subnormal_policy()
    if before is None:
        os.environ.pop("CUPY_CACHE_DIR", None)
    else:
        os.environ["CUPY_CACHE_DIR"] = before


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_sub_step_stores_the_numpy_words(name, keep_on_cupy, monkeypatch):
    reference, candidate = _build(name, False), _build(name, True)
    try:
        assert reference.gpu is None and candidate.gpu == "cuda"
        assert not _differing(capsule.state_arrays(reference), capsule.state_arrays(candidate))
        recorder = _Recorder()
        for sub_step in SUB_STEPS:
            monkeypatch.setattr(driver_module, sub_step,
                                recorder.wrap(sub_step, getattr(driver_module, sub_step)))
        for value in list(vars(sources_module).values()):
            if isinstance(value, type) and "inject" in value.__dict__:
                monkeypatch.setattr(value, "inject", recorder.wrap("inject", value.__dict__["inject"]))
        compared = 0
        for step in range(1, STEPS + 1):
            runs = []
            for driver in (reference, candidate):
                recorder.driver, recorder.snapshots = driver, []
                driver.step()
                runs.append(recorder.snapshots)
            recorder.driver = None
            assert [label for label, _ in runs[0]] == [label for label, _ in runs[1]]
            for (label, left), (_, right) in zip(*runs):
                differing = _differing(left, right)
                assert not differing, (
                    f"{name}: step {step}, after {label}: {sum(differing.values())} words differ "
                    f"between the NumPy and the CuPy array path: {differing}")
                compared += 1
        assert compared >= STEPS * 8, f"{name}: only {compared} sub-steps were compared"
        # What was compared is the CuPy ARRAY path: the candidate's freeze consulted no
        # kernel (dispatch is pinned off in the suites), so every slot stepped stepping.py.
        assert candidate.active_step_path == "array"
        report = candidate.fast_path_report()
        assert report is not None and report.get("decision") == "refused", report
        plan = candidate.fields.region_plan
        if plan is not None:
            assert plan.backend == "cupy"
            assert "select" not in {spec.mode for spec in plan.specs.values()}
    finally:
        reference.close()
        candidate.close()
