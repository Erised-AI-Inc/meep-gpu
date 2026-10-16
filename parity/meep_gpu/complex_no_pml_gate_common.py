"""What the two complex no-absorber curl gates share: cases, states, records, mutations.

``gate_metal_complex_no_pml_curl.py`` (the lossless curl) and
``gate_metal_complex_no_pml_conductive.py`` (the conductive one) step one kernel body,
:mod:`meep_gpu.metal_kernels.complex_no_pml_conductive`'s, beside the same oracle,
``stepping.step_B`` / ``step_D``. This module holds the parts of those gates that
must not drift apart:

* **the cases** (:data:`CASES`): 3-D, 2-D and 1-D grids, a 3-D grid that stores one
  cell on a live axis, unphased, phased (``CONNECT_PHASE``) and Brillouin-edge
  (``CONNECT_NEGATE``) axes, and metallic walls. Partner presence follows each grid's
  invariance and reaches the kernel through the chunk record only;
* **the record each case binds** (:func:`case_record`): the composer's one-rank
  division (``gate_chunk_records.layer_record``), and its every-axis-present twin for
  the ``presence_word_ignored`` row (:func:`all_axes_present`);
* **the planted state** (:func:`fill`): ``f = -0`` per part on the curl targets and
  constant sources, so every stencil reads equal neighbours and the absent-partner
  forms differ from their reversed spellings only in the sign of a zero;
* **the mutations** (:func:`curl_mutations`), each needle required exactly once in the
  source it is armed on (:func:`needle`), with its case and its expectation: the
  previous pairwise bracket (KERNELS.md §7.1, inverted from the old
  ``curl_grouping_flattened``), the full bracket over an absent partner, the
  absent-first-partner form with the sign of ``dtdx`` kept or its difference reversed,
  the plane-0 connection dropped or a negation spelled as a product by the bound
  placeholder, and the presence word ignored. Each absent-arm mutation has a
  declared-null twin on a grid where that arm never runs.

Host-only NumPy: no torch import here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gate_chunk_records import layer_record
from meep_gpu import stepping
from meep_gpu.chunk_map import RECORD_PRESENCE_WORD
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import complex_fields
from meep_gpu.pml import PML

#: label -> (cell size, dimensions, Bloch k, boundaries). Every axis of every case is
#: periodic unless named; ``k`` on a 1.0-long axis at 0.5 is the Brillouin edge.
CASES: Dict[str, Tuple[Tuple[float, float, float], int, Tuple[float, float, float],
                       Tuple[str, str, str]]] = {
    "two_axis_bloch": ((1.0, 0.875, 0.75), 3, (0.2, -0.125, 0.0),
                       ("periodic", "periodic", "periodic")),
    "zero_phase_control": ((1.0, 0.875, 0.75), 3, (0.0, 0.0, 0.0),
                           ("periodic", "periodic", "periodic")),
    "metallic_xz_bloch_y": ((1.0, 0.875, 0.75), 3, (0.0, 0.125, 0.0),
                            ("metallic", "periodic", "metallic")),
    "collapsed_y": ((1.25, 0.0625, 0.75), 3, (0.2, 0.0, 0.0),
                    ("periodic", "periodic", "periodic")),
    "brillouin_edge_x": ((1.0, 0.875, 0.75), 3, (0.5, 0.0, 0.0),
                         ("periodic", "periodic", "periodic")),
    "planar_2d_bloch": ((1.0, 0.875, 0.0), 2, (0.2, -0.125, 0.0),
                        ("periodic", "periodic", "periodic")),
    "planar_2d_edge_y": ((1.0, 1.0, 0.0), 2, (0.0, 0.5, 0.0),
                         ("periodic", "periodic", "periodic")),
    "planar_2d_metallic_x": ((1.0, 0.875, 0.0), 2, (0.0, 0.3, 0.0),
                             ("metallic", "periodic", "periodic")),
    "line_1d_z": ((0.0, 0.0, 2.0), 1, (0.0, 0.0, 0.3),
                  ("periodic", "periodic", "periodic")),
    "planar_2d_unphased": ((1.0, 0.875, 0.0), 2, (0.0, 0.0, 0.0),
                           ("periodic", "periodic", "periodic")),
}


def words(value: Any) -> np.ndarray:
    """View all float32 words without changing complex layout."""
    array = np.ascontiguousarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.complex64)):
        raise TypeError(f"expected float32 or complex64, got {array.dtype}")
    return array.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def negative_zeros(value: Any) -> int:
    flat = np.ascontiguousarray(value).reshape(-1).view(np.float32)
    return int(np.count_nonzero((flat == 0) & np.signbit(flat)))


def grid_for(case: str) -> Grid:
    cell, dimensions, phase, boundaries = CASES[case]
    return Grid(resolution=8.0, cell_size=tuple(cell), courant=0.3,
                boundaries=tuple(boundaries), k_point=tuple(phase), xp=np,
                dimensions=dimensions)


def build(case: str, *, seed: int, planted: bool = False,
          install: Optional[Callable[[Fields], None]] = None,
          names: Iterable[str] = ()) -> Tuple[Fields, PML]:
    """A live complex no-absorber state on one case's grid.

    ``install`` runs before the fill (a conductivity allocates ``f_cond_*``);
    ``names`` are filled beside the twelve field components.
    """
    grid = grid_for(case)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    fields.enable_field_storage()
    if install is not None:
        install(fields)
    fill(fields, seed=seed, planted=planted, names=names)
    return fields, PML(grid=grid, thickness=0)


def fill(fields: Fields, *, seed: int, planted: bool, names: Iterable[str] = ()) -> None:
    """Random words on every live volume, or the planted signed-zero state."""
    rng = np.random.default_rng(seed)
    shape = fields.grid.shape
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", *tuple(names)):
        array = getattr(fields, name, None)
        if array is None:
            continue
        if planted and name[0] in "BD":
            array.real[...] = -0.0
            array.imag[...] = -0.0
        elif planted and name[0] in "EH":
            array[...] = np.complex64(0.25 - 0.125j)
        else:
            array[...] = (rng.uniform(-0.3, 0.3, shape).astype(np.float32)
                          + 1j * rng.uniform(-0.3, 0.3, shape).astype(np.float32))


def case_record(fields: Fields, pml: PML) -> np.ndarray:
    """The record the composer hands this run's plans: the one-rank division."""
    return layer_record(fields.grid, pml)


def all_axes_present(record: Any) -> np.ndarray:
    """``record`` with every partner-presence bit set: the kernel then steps the
    both-partner bracket on an invariant axis."""
    out = np.array(record, copy=True)
    out[RECORD_PRESENCE_WORD] = 7
    return out


def b_specialization(case: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """(boundary codes, step_B connection kinds) of one case, as the plan derives them."""
    grid = grid_for(case)
    resolved = stepping._boundary_kinds(grid, None)
    codes = tuple(1 if kind == "metallic" else 0 for kind in resolved)
    kinds, _values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, resolved), backward=False)
    return codes, kinds


def needle(text: str, old: str, new: str) -> str:
    """Replace ``old`` by ``new``, refusing unless ``old`` occurs exactly once."""
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"a mutation needle matches {count} times, not once: {old!r}")
    return text.replace(old, new)


#: One mutation row: (label, case, transform of the step_B source, expectation,
#: planted state?, every-axis-present record?). Expectation "caught" requires a
#: differing word; "null" requires none (the mutated arm does not run on that case).
MutationRow = Tuple[str, str, Callable[[str], str], str, bool, bool]


def curl_mutations() -> List[MutationRow]:
    """The stencil, connection and presence rows both curl gates arm."""
    both = "c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z)"
    pairwise = "c_mul_coefficient_left(dtdx, (c_y - c_dy) + (b_dz - b_z))"
    absent_second = ": c_mul_coefficient_left(dtdx, c_y - c_dy))"
    full_bracket = ": c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z))"
    absent_first = "c_mul_coefficient_left(-dtdx, c_x - c_dx)"

    def previous_grouping(text: str) -> str:
        text = needle(text, both, pairwise)
        text = needle(text, "c_mul_coefficient_left(dtdx, ((a_z - a_dz) + c_dx) - c_x)",
                      "c_mul_coefficient_left(dtdx, (a_z - a_dz) + (c_dx - c_x))")
        return needle(text, "c_mul_coefficient_left(dtdx, ((b_x - b_dx) + a_dy) - a_y)",
                      "c_mul_coefficient_left(dtdx, (b_x - b_dx) + (a_dy - a_y))")

    def full_bracket_absent(text: str) -> str:
        return needle(text, absent_second, full_bracket)

    def keeps_sign(text: str) -> str:
        return needle(text, absent_first, "c_mul_coefficient_left(dtdx, c_x - c_dx)")

    def reversed_difference(text: str) -> str:
        return needle(text, absent_first, "c_mul_coefficient_left(dtdx, c_dx - c_x)")

    def connection_dropped(text: str) -> str:
        return needle(text, "        c_dx = c_mul(c, px);\n        b_dx = c_mul(b, px);\n",
                      "        // dropped: the x connection of the unshifted partners\n")

    def negation_as_product(text: str) -> str:
        text = needle(text, "        c_dy = -c;\n", "        c_dy = c_mul(c, py);\n")
        return needle(text, "        a_dy = -a;\n", "        a_dy = c_mul(a, py);\n")

    def unchanged(text: str) -> str:
        return text

    return [
        ("previous_curl_grouping", "two_axis_bloch", previous_grouping, "caught",
         False, False),
        ("full_bracket_over_an_absent_partner", "planar_2d_bloch", full_bracket_absent,
         "caught", False, False),
        ("full_bracket_over_an_absent_partner_on_a_resolved_grid", "two_axis_bloch",
         full_bracket_absent, "null", False, False),
        ("absent_first_partner_keeps_the_sign_of_dtdx", "planar_2d_bloch", keeps_sign,
         "caught", False, False),
        ("absent_first_partner_keeps_the_sign_of_dtdx_on_a_resolved_grid",
         "two_axis_bloch", keeps_sign, "null", False, False),
        ("absent_first_partner_as_the_reversed_difference", "planar_2d_unphased",
         reversed_difference, "caught", True, False),
        ("bloch_x_connection_dropped", "two_axis_bloch", connection_dropped, "caught",
         False, False),
        ("negation_spelled_as_a_product_by_the_placeholder", "planar_2d_edge_y",
         negation_as_product, "caught", False, False),
        ("presence_word_ignored", "planar_2d_bloch", unchanged, "caught", False, True),
    ]


#: The planted rows the shipped kernel must reproduce word for word: (label, case).
PLANTED_CASES: Tuple[Tuple[str, str], ...] = (
    ("planted_3d_unphased", "zero_phase_control"),
    ("planted_3d_bloch", "two_axis_bloch"),
    ("planted_3d_metallic", "metallic_xz_bloch_y"),
    ("planted_2d_unphased", "planar_2d_unphased"),
    ("planted_2d_edge", "planar_2d_edge_y"),
    ("planted_1d", "line_1d_z"),
)


def provenance_cases() -> Mapping[str, Any]:
    return {label: {"cell_size": list(cell), "dimensions": dimensions,
                    "k_point": list(phase), "boundaries": list(boundaries)}
            for label, (cell, dimensions, phase, boundaries) in CASES.items()}
