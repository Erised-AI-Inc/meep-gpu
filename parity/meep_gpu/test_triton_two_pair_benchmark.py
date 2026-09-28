"""Laptop tests for the two-pair Triton performance harness.

The benchmark itself requires CUDA. These tests pin the state/coefficient product,
repeat policy and incremental-artifact contract without importing CuPy or Triton.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE / "probe_triton_two_pair_benchmark.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("triton_two_pair_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_product_covers_both_pairs_and_inverse_epsilon():
    probe = load_probe()
    shape = (5, 4, 3)
    state = probe.host_state(shape)

    assert tuple(state) == probe.ALL_STATE_NAMES
    assert len(state) == 27
    assert len({id(value) for value in state.values()}) == 27
    for value in state.values():
        assert value.shape == shape
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
    for name in probe.INVERSE_EPSILON_NAMES:
        assert np.all(state[name] > 0)
    assert probe.host_state(shape)["Dx"].tobytes() == state["Dx"].tobytes()


def test_coefficient_product_keeps_the_four_yee_families_separate():
    probe = load_probe()
    shape = (7, 5, 2)
    coefficients = probe.host_coefficients(shape)

    assert tuple(coefficients) == (
        "B_curl_half", "H_const_integer", "D_curl_integer", "E_const_half")
    assert all(tuple(family) == (
        "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
        for name, family in coefficients.items() if "curl" in name)
    assert all(tuple(family) == (
        "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")
        for name, family in coefficients.items() if "const" in name)
    for family in coefficients.values():
        for key, value in family.items():
            assert value.shape == (shape["xyz".index(key[-1])],)
            assert value.dtype == np.float32
            assert value.flags.c_contiguous

    # A shared table would make a half-cell registration swap invisible.
    assert coefficients["B_curl_half"]["kms_x"].tobytes() != (
        coefficients["D_curl_integer"]["kms_x"].tobytes())
    assert coefficients["H_const_integer"]["kms_x"].tobytes() != (
        coefficients["E_const_half"]["kms_x"].tobytes())


def test_repeat_policy_is_bounded_and_reaches_the_target_when_possible():
    probe = load_probe()
    assert probe.choose_repeats(0.01, 0.12, 500) == 12
    assert probe.choose_repeats(1.0, 0.12, 500) == 3
    assert probe.choose_repeats(1e-6, 0.12, 500) == 500
    assert probe.choose_repeats(float("nan"), 0.12, 500) == 3


def test_restore_uses_the_explicit_host_to_device_copy_route():
    probe = load_probe()
    host = probe.host_state((2, 2, 1))

    class DeviceArray:
        def __init__(self):
            self.received = None

        def set(self, value):
            self.received = value

    state = {name: DeviceArray() for name in probe.ALL_STATE_NAMES}
    probe.restore_state(state, host)
    for name in probe.ALL_STATE_NAMES:
        assert state[name].received is host[name]


def test_metallic_control_runs_the_real_intervening_wall_writes():
    probe = load_probe()
    state = {name: np.ones((3, 4, 2), dtype=np.float32)
             for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}

    assert probe.zero_metal_arrays(state, (1, 1, 0), "B") == 2
    assert np.all(state["Bx"][0, :, :] == 0)
    assert np.all(state["By"][:, 0, :] == 0)
    assert np.all(state["Bz"] == 1)

    assert probe.zero_metal_arrays(state, (1, 1, 0), "D") == 4
    assert np.all(state["Dx"][:, 0, :] == 0)
    assert np.all(state["Dy"][0, :, :] == 0)
    assert np.all(state["Dz"][0, :, :] == 0)
    assert np.all(state["Dz"][:, 0, :] == 0)
    assert probe.unfused_launch_count((0, 0, 0)) == 4
    assert probe.unfused_launch_count((1, 1, 0)) == 10


def test_harness_records_incrementally_and_checks_bits_outside_timing():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "left[name].view(cp.uint32) != right[name].view(cp.uint32)" in source
    assert "atomic_save(results, out_path)" in source
    assert "window % 2 == 0" in source
    assert "state diverged after timing window" in source
    assert '"launches": {"unfused": unfused_launch_count(boundaries), "fused": 2}' in source


def test_fair_tuning_includes_2d_and_3d_and_tunes_every_control_launch():
    probe = load_probe()
    assert (2048, 2048, 1) in probe.FAIR_TUNING_SHAPES
    assert any(shape[2] > 1 for shape in probe.FAIR_TUNING_SHAPES)
    assert probe.UNFUSED_STAGES == ("step_B", "update_H", "step_D", "update_E")
    assert probe.WARPS == (1, 2, 4, 8)
    assert probe.FUSED_POLICY_WARPS == (1, 1)
    assert probe.UNFUSED_POLICY_WARPS == (8, 1, 2, 1)

    source = SCRIPT.read_text(encoding="utf-8")
    assert "changed output bits" in source
    assert 'results["tuning_choices"].append' in source
    assert 'unfused_warps=control_warps' in source
    assert '"all_unfused_warp_outputs_bit_identical"' in source
    assert '"policy_sweep"' in source
    assert '"policy_worst_ratio"' in source
