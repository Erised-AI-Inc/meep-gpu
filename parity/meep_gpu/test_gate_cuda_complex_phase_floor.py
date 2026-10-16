"""Host checks of the complex CUDA gate's phase floor (no device).

The floor exists so that a case labelled phased cannot pass while measuring the
unphased kernel. It used to call every case with a nonzero k-point phased. MEEP's
connection rule (``part_products.connect_parts``) makes a phase within
``CONNECT_TOLERANCE`` of +1 a copy, and the Brillouin-edge k-point on the gate's 8-cell
axis wraps to exactly that: eikna = exp(2*pi*i*0.5*8) = +1. The array path and the kernel
both copy there, every byte is identical, and the label-based floor failed on 96 of 736
phase-labelled curl cases. The floor now grades by the connection the array path
applies, read without the kernel module, and keeps its teeth: a kernel encoding that
disagrees with the array path on any axis fails it, as do a phase-labelled k-point
that never connects a plane and a Brillouin edge that never negates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_gate():
    path = HERE / "gate_cuda_complex.py"
    spec = importlib.util.spec_from_file_location("gate_cuda_complex_phase_floor_host", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def _record(gate, grid, sub_step, k_label, shape_label):
    """The phase fields of one case record, as ``one_case`` derives them."""
    backward = gate.complex_emitter.KERNELS[sub_step][1]
    flags, _values = gate.host_phase_arguments(grid, backward)
    kinds = gate.array_path_connection(grid, backward)
    return {"sub_step": sub_step, "k_label": k_label,
            "shape": [int(n) for n in shape_label.split("x")],
            "phase_is_live": any(int(f) != gate.complex_emitter.CONNECT_NONE for f in flags),
            "array_connection": list(kinds),
            "array_connects_a_plane": any(k in ("negate", "product") for k in kinds),
            "connection_agrees": ([int(f) for f in flags]
                                  == [gate.ARRAY_CONNECTION_KIND[k] for k in kinds])}


@pytest.fixture(scope="module")
def product_records(gate):
    """Every viable curl configuration of the full product, on the host."""
    rng = np.random.default_rng(0)
    records = []
    for sub_step in gate.CURL_SUB_STEPS:
        for cell, shape_label in gate.SHAPES:
            for triple in gate.BOUNDARY_TRIPLES:
                for k_point, k_label in gate.K_POINTS:
                    if gate.config_viable(cell, triple, k_point) is not None:
                        continue
                    _fields, _layer, grid = gate.build(np, cell, triple, k_point, 0.5,
                                                       False, rng)
                    records.append(_record(gate, grid, sub_step, k_label, shape_label))
    return records


def test_the_full_product_passes_the_floor_and_copies_only_the_8_cell_edge(gate,
                                                                           product_records):
    floor = gate.phase_floor(product_records)
    assert floor["passed"], floor
    assert floor["configurations_that_copy"] == [("8x8x8", "kedge")]
    assert floor["phase_labelled_cases_that_copy"] > 0
    assert floor["phased_curl_cases"] == (floor["phase_labelled_curl_cases"]
                                          - floor["phase_labelled_cases_that_copy"])


def test_the_edge_negates_on_the_odd_and_one_cell_axes(gate, product_records):
    edge = {tuple(r["shape"]): r["array_connection"][0]
            for r in product_records if r["k_label"] == "kedge"}
    assert edge == {(8, 8, 8): "copy", (13, 17, 11): "negate", (1, 16, 16): "negate"}


def _case(gate, k_label, kinds, flags, shape=(13, 17, 11)):
    return {"sub_step": "step_B", "k_label": k_label, "shape": list(shape),
            "phase_is_live": any(f != gate.complex_emitter.CONNECT_NONE for f in flags),
            "array_connection": list(kinds),
            "array_connects_a_plane": any(k in ("negate", "product") for k in kinds),
            "connection_agrees": list(flags) == [gate.ARRAY_CONNECTION_KIND[k]
                                                 for k in kinds]}


def _healthy(gate):
    none, neg, ph = (gate.complex_emitter.CONNECT_NONE, gate.complex_emitter.CONNECT_NEGATE,
                     gate.complex_emitter.CONNECT_PHASE)
    u = "unconnected"
    return [
        _case(gate, "k0", (u, u, u), (none, none, none)),
        _case(gate, "kx", ("product", u, u), (ph, none, none)),
        _case(gate, "kedge", ("negate", u, u), (neg, none, none)),
        _case(gate, "kedge", ("copy", u, u), (none, none, none), shape=(8, 8, 8)),
    ]


def test_a_copy_by_meeps_rule_is_not_a_dead_phased_case(gate):
    floor = gate.phase_floor(_healthy(gate))
    assert floor["passed"], floor
    assert floor["configurations_that_copy"] == [("8x8x8", "kedge")]


def test_a_kernel_that_copies_where_the_array_path_negates_fails(gate):
    cases = _healthy(gate)
    cases[2] = _case(gate, "kedge", ("negate", "unconnected", "unconnected"),
                     (gate.complex_emitter.CONNECT_NONE,) * 3)
    floor = gate.phase_floor(cases)
    assert not floor["passed"]
    assert not floor["clauses"]["every_case_connects_as_the_array_path"]
    assert not floor["clauses"]["every_phased_case_is_live_in_the_kernel"]


def test_a_kernel_that_multiplies_where_the_array_path_negates_fails(gate):
    cases = _healthy(gate)
    cases[2] = _case(gate, "kedge", ("negate", "unconnected", "unconnected"),
                     (gate.complex_emitter.CONNECT_PHASE, gate.complex_emitter.CONNECT_NONE,
                      gate.complex_emitter.CONNECT_NONE))
    floor = gate.phase_floor(cases)
    assert not floor["passed"]
    assert not floor["clauses"]["every_case_connects_as_the_array_path"]


def test_a_phase_label_that_never_connects_fails(gate):
    cases = _healthy(gate)
    cases[1] = _case(gate, "kx", ("copy", "unconnected", "unconnected"),
                     (gate.complex_emitter.CONNECT_NONE,) * 3)
    floor = gate.phase_floor(cases)
    assert not floor["passed"]
    assert not floor["clauses"]["every_phased_k_point_connects_a_plane_somewhere"]


def test_an_edge_that_never_negates_fails(gate):
    cases = [c for c in _healthy(gate) if not (c["k_label"] == "kedge"
                                               and c["array_connection"][0] == "negate")]
    floor = gate.phase_floor(cases)
    assert not floor["passed"]
    assert not floor["clauses"]["the_brillouin_edge_negates_somewhere"]
