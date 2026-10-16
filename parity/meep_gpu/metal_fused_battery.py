"""The per-chunk form battery for the Metal fused pairs that LIFT the base bodies.

KERNELS.md §7.2 asks every family gate for a must-catch battery against the per-chunk
forms: the select removed, the predicate keyed on coefficient identity instead of chunk
membership, a bound off by one, the two Yee sub-lattices swapped, the dsig and dsigu
flags swapped, an auxiliary stored on a plain cell, and the two plain forms spelled
through the accumulating ones (caught only where a -0 reaches them). Two rows are
declared nulls: an auxiliary LOAD outside its guard with the store still guarded
changes no word.

THE ROWS ARE THE BASE GATE'S OWN, NOT A SECOND SPELLING. ``fused_magnetic_pair``,
``fused_electric_pair`` and ``nonlinear_fused_magnetic_pair`` splice the certified
curl and constitutive bodies of :mod:`meep_gpu.metal_kernels.shaders` verbatim (the
constitutive half renamed pointer by pointer, never in a line these rows touch), so
the needles :mod:`gate_metal_pml` arms on the base kernels resolve in the fused
sources too. Each transform refuses a needle it cannot find, so a lift that stopped
carrying a line fails here by name rather than arming nothing.

``b8`` is armed on the physical band, where it differs on the dsig-only cells whose
dsig coefficients are not 1; ``b13`` differs only on a -0 source and runs on the
+-0 lattice (:mod:`metal_value_classes`), which seeds one on every other cell.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import gate_metal_pml as base_gate
import metal_value_classes as values
from gate_chunk_records import layer_record

#: The curl rows, in the base gate's order.
CURL_ROWS: Tuple[str, ...] = (
    "b1_select_removed",
    "b2_predicate_on_coefficient_identity",
    "b5_dsig_dsigu_swapped",
    "b6_auxiliary_stored_on_a_plain_cell",
    "b7_auxiliary_load_unconditional",
    "b8_plain_curl_through_the_recurrence",
)

#: The constitutive rows.
CONSTITUTIVE_ROWS: Tuple[str, ...] = (
    "b9_constitutive_select_removed",
    "b10_constitutive_predicate_on_coefficient_identity",
    "b11_f_w_stored_on_a_plain_cell",
    "b12_constitutive_load_unconditional",
    "b13_plain_constitutive_through_the_accumulation",
)

#: The rows a physical-band state cannot reach: they differ only on a -0 word.
LATTICE_ROWS: Tuple[str, ...] = ("b13_plain_constitutive_through_the_accumulation",)


def source_rows(source: str, curl: bool = True, constitutive: bool = True
                ) -> Dict[str, Dict[str, Any]]:
    """Every source row of the battery applied to one fused source.

    ``{name: {"source", "must_catch", "why", "value_class"}}``. ``must_catch`` is the
    base gate's: True (must diverge) or False (a declared null, must not).
    """
    names = (CURL_ROWS if curl else ()) + (CONSTITUTIVE_ROWS if constitutive else ())
    rows: Dict[str, Dict[str, Any]] = {}
    for name in names:
        table = (base_gate.CURL_MUTATIONS if name in CURL_ROWS
                 else base_gate.CONSTITUTIVE_MUTATIONS)
        entry = table[name]
        rows[name] = {
            "source": entry["apply"](source),
            "must_catch": entry["must_catch"],
            "why": entry["why"],
            "value_class": (values.PM_ZERO_LATTICE if name in LATTICE_ROWS
                            else values.UNIFORM),
        }
    return rows


#: The record rows: the kernel is right and the record it is handed is not.
RECORD_ROWS: Mapping[str, Tuple[Callable[[Any], Any], str]] = base_gate.RECORD_MUTATIONS


def record_for(grid: Any, pml: Any, mutate: Callable[[Any], Any] = None) -> Any:
    """The one-rank record of the case, optionally mutated."""
    record = layer_record(grid, pml)
    return record if mutate is None else mutate(record)


def form_census(grid: Any, pml: Any) -> Dict[str, Dict[str, int]]:
    """Cells per (component, form) on this case (DESIGN.md D13's coverage rule)."""
    return base_gate._form_counts(layer_record(grid, pml), grid.shape)


def empty_form_classes(census: Mapping[str, Mapping[str, int]],
                       components: Sequence[str]) -> Tuple[str, ...]:
    """The (component, form) classes of ``components`` this case never steps."""
    return tuple(f"{component}/{form}" for component in components
                 for form, count in census.get(component, {}).items() if not count)


def scored(must_catch: Any, caught: bool, launched: bool) -> bool:
    """A row passes when it launched and its outcome is the declared one."""
    if not launched:
        return False
    if must_catch is None:
        return True
    return bool(caught) == bool(must_catch)
