"""The Bloch connection is resolved on any host, so its predicate clause is about the run.

``bloch_connection`` holds ``bloch_phase_arguments`` and ``connection_of``, which lived in
the CuPy-scope ``complex_pml_kernels``. Two fused products' predicates resolve the Bloch
tables before they admit a run (``complex_offdiag_update_e.bloch_connection_table``):
``complex_no_pml_offdiag_fused_electric_pair`` and
``folded_complex_offdiag_fused_electric_pair``. On a host without CuPy that clause raised
``ModuleNotFoundError: No module named 'cupy'`` and returned it as a refusal, which the
coverage census (asking with the backend clause satisfied at the input) scored as a
refusal of the run: 5 cells on the 1.0 CUDA census, 2 of them rows the products admit.

Pinned here:

1. in a fresh interpreter in which CuPy cannot be imported, both tables resolve and both
   products, asked through a grid whose array module answers ``cupy`` (the census's
   factoring), admit a configuration each serves -- among them the configuration of
   ``TestHoleyWvgBands.test_fields_at_kx``, a phased folded run -- and CuPy was never
   imported;
2. the clause still refuses by name when the table cannot be resolved (a planted raise);
3. the launcher binds the same function the predicates resolve.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from . import bloch_connection
from . import complex_no_pml_offdiag_fused_electric_pair as no_pml_pair
from . import folded_complex_offdiag_fused_electric_pair as folded_pair

#: ``(id, product module, predicate, fixture)``. A fixture is ``module:function`` and
#: returns ``(fields, pml)``, or a driver whose fields and layer are read.
PRODUCTS = (
    ("no_pml", "complex_no_pml_offdiag_fused_electric_pair",
     "covers_complex_no_pml_offdiag_fused_electric_pair",
     "meep_gpu.cuda_kernels.test_complex_offdiag_stencil_welds:_no_pml_complex_fixture"),
    ("folded", "folded_complex_offdiag_fused_electric_pair",
     "covers_folded_complex_offdiag_fused_electric_pair",
     "meep_gpu.cuda_kernels.test_complex_offdiag_stencil_welds:_folded_complex_fixture"),
    ("holey_wvg_row", "folded_complex_offdiag_fused_electric_pair",
     "covers_folded_complex_offdiag_fused_electric_pair",
     "meep_gpu.test_composer_run_division:_row_driver"),
)

#: The clause's own refusal prefix, read from both predicates.
BLOCH_CLAUSE = "the Bloch table this launch binds once for both halves: "


class CupyNamed:
    """An array module reporting ``cupy`` as its name and nothing else."""

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    __name__ = "cupy"

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class CupyNamedGrid:
    """A grid with ``xp`` reporting ``cupy``, every other read forwarded: the census's
    factoring of the backend clause (``cuda_predicate_battery._GridWithCupyBackend``)."""

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", CupyNamed(grid.xp))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), name)


def build_fixture(spec: str):
    """``(fields, pml)`` from a ``module:function`` fixture."""
    import importlib  # noqa: PLC0415

    module_name, function = spec.split(":")
    made = getattr(importlib.import_module(module_name), function)()
    if isinstance(made, tuple):
        return made
    return made.fields, made.pml


_CHILD = textwrap.dedent('''
    import json, sys
    sys.modules["cupy"] = None  # any import of CuPy now raises ImportError
    from meep_gpu.cuda_kernels import complex_offdiag_update_e as offdiag
    from meep_gpu.cuda_kernels import {module} as product
    from meep_gpu.cuda_kernels.test_bloch_connection import CupyNamedGrid, build_fixture

    fields, pml = build_fixture({fixture!r})
    table = offdiag.bloch_connection_table(fields.grid)
    covered, reason = product.{predicate}(fields, pml, CupyNamedGrid(fields.grid), ())
    print(json.dumps({{"table": [list(part) for part in table], "covered": covered,
                      "reason": reason, "cupy": sys.modules.get("cupy") is not None,
                      "has_bloch": bool(fields.grid.has_bloch)}}))
''')


@pytest.mark.parametrize("label,module,predicate,fixture", PRODUCTS,
                         ids=[product[0] for product in PRODUCTS])
def test_each_product_admits_on_a_host_without_cupy(label, module, predicate, fixture):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    done = subprocess.run(
        [sys.executable, "-c", _CHILD.format(module=module, predicate=predicate,
                                             fixture=fixture)],
        capture_output=True, text=True, env=env, cwd=root, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["cupy"] is False, "CuPy was imported"
    assert len(found["table"]) == 4, found["table"]
    assert found["covered"] is True, (
        f"{label}: asked with the backend clause satisfied, on a host without CuPy, the "
        f"product refused {found['reason']!r}")
    if label != "folded":
        # A phase on x (and y on the no-PML cell): the table is not the trivial one.
        assert found["has_bloch"] is True
        assert any(kind != 0 for kind in found["table"][0]), found["table"]


@pytest.mark.parametrize("label,module,predicate,fixture", PRODUCTS,
                         ids=[product[0] for product in PRODUCTS])
def test_an_unresolvable_table_is_still_refused_by_name(label, module, predicate,
                                                        fixture, monkeypatch):
    family = {"complex_no_pml_offdiag_fused_electric_pair": no_pml_pair,
              "folded_complex_offdiag_fused_electric_pair": folded_pair}[module]
    fields, pml = build_fixture(fixture)

    def planted(grid: Any, backward: bool):
        raise ValueError("planted: no connection table for this grid")

    monkeypatch.setattr(bloch_connection, "bloch_phase_arguments", planted)
    covered, reason = getattr(family, predicate)(fields, pml, CupyNamedGrid(fields.grid), ())
    assert covered is False
    assert reason == f"{BLOCH_CLAUSE}ValueError: planted: no connection table for this grid"


def test_the_launcher_binds_the_function_the_predicates_resolve():
    from .test_complex_chunk_forms import load_with_numpy_as_cupy  # noqa: PLC0415

    module = load_with_numpy_as_cupy("complex_pml_kernels", "_bloch_connection_launcher")
    assert module.bloch_phase_arguments is bloch_connection.bloch_phase_arguments
    assert not hasattr(module, "connection_of"), (
        "connection_of lives in bloch_connection only; the launcher binds its kinds "
        "through bloch_phase_arguments")
