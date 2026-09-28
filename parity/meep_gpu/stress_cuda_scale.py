"""SCALE stress leg for the certified hand-CUDA families: the corpus's real shapes.

=============================================================================
WHAT THIS IS FOR, AND WHAT IT IS NOT
=============================================================================

The 26 gates under ``parity/meep_gpu/`` establish byte identity against
``stepping``'s array path, per sub-step, at one launch and at
``MULTI_STEP_BUDGET = 60``, under both float32 subnormal policies, over uniform
and subnormal-band value classes. Every one of them does it on a fixture the
size of a postage stamp: ``(13,11,9)``, ``(32,32,32)``, ``(64,48,1)``,
``(17,5,3)``.

THE CORPUS DOES NOT LOOK LIKE THAT. Measured off
``results/cuda_predicate_coverage_2026-08-20_closed/`` (186 rows, the record the
759/759 coverage claim is cut from):

    11,245,000 cells   grating2d_triangular_lattice.py   100 x 173 x 650
       704,969 cells   TestLDOS.test_ldos_3D              89 x  89 x  89
       490,000 cells   gaussian-beam.py / oblique-source.py  700 x 700 x 1
       404,800 cells   stochastic_emitter_reciprocity.py 220 x 1840 x 1
        22,500 cells   THE MEDIAN                        150 x 150 x 1

The largest admitted row is 8,738x the largest gate fixture. This leg runs the
same kernels at the shapes the corpus actually has, and it asks the four
questions a postage stamp cannot:

1. THE INT32 INDEX BOUND. Four predicates carry the clause
   ``if cells >= 2**31: return False, "... exceeds the kernel's int32 index
   range"`` (coverage.py:1214-1215, :1597-1601, :1797-1801, :2147-2148) and the
   complex arm carries it HALVED (``2 * cells >= 2 ** 31``, :2878-2880). Nothing
   has ever exercised those clauses near their bound. This leg BISECTS for the
   threshold and reports the measured value beside the transcribed one.
2. GRID-DEPENDENT INDEX DECOMPOSITION. ``k = idx % nz; j = (idx/nz) % ny;
   i = idx/(ny*nz)`` (step_curl_kernels.py:231-233, constitutive_kernels.py:
   304-306) is exact in ``int`` only while the products fit. 220x1840x1 stresses
   a different corner of that than 89^3 does, and both are checked EXHAUSTIVELY
   over the whole index range rather than sampled.
3. LAUNCH GEOMETRY. ``blocks = (nx*ny*nz + threads - 1) // threads`` with
   ``threads = 256`` (constitutive_kernels.py:373, :471; step_curl_kernels.py:
   1905 and the nine ``threads = 256`` launchers beside it). Correct at 1,287
   cells is not evidence at 11,245,000: the block count, the top lane's ``int
   idx``, and the guard's own ``nx * ny * nz`` are all recomputed and checked.
4. MEMORY. The largest row is order a gigabyte of float32 volumes before any
   auxiliary. Every case predicts its footprint from a MEASURED words-per-cell
   and SKIPS WITH A REASON rather than dying in the allocator, and every device
   run records peak device memory so a future reader knows what the leg needs.

IT IS NOT A CERTIFICATION. It changes no predicate, no kernel and no record. It
adds one axis -- shape -- to legs that already exist, and its NumPy arm compiles
nothing and certifies nothing (see below).

=============================================================================
WHY THE SUBJECT IS TWO FAMILIES AND NOT TWENTY-TWO
=============================================================================

The two families that carry the corpus's LARGEST admitted rows are also the two
whose kernels are certified: ``covers_real_pml_curl`` (step_B/step_D, 249
admitted slots, top row 11,245,000 cells) and ``covers_real_pml_constitutive``
(update_H/update_E, 209 slots, same top row). Both already have a gate with a
transcribed NumPy arm, and this file IMPORTS THOSE ARMS rather than writing a
third spelling of the same device tree -- the sibling gates' rule, for the same
reason: a harness that owns its own transcription of a kernel can drift from the
kernel and report the drift as a kernel property.

Every other family's largest admitted row is smaller, and the ones that are not
(``cuda_offdiag`` at 11,245,000, ``cuda_ade``/``cuda_dispersive`` at 404,800,
``cuda_complex`` at 350,000) have no NumPy arm to borrow. They are named in
``not_established`` rather than silently omitted.

=============================================================================
WHAT MAKES THIS LEG ABLE TO FAIL
=============================================================================

* NON-VACUITY PER CHECKPOINT, not once at the end. ``oracle_moved`` is measured
  against the frozen state AND against the previous checkpoint at every
  checkpoint. A recurrence that has settled is refused, because from there a
  deliberately wrong reference still reports IDENTICAL.
* CHECKPOINTS, so a divergence has a LOCATION. Both paths run from one frozen
  state and are compared by sha256 over their uint32 words at every checkpoint;
  the first differing checkpoint is then re-run and compared word-for-word with
  ``probe.bit_compare`` so the record carries the ULP gap and the component, not
  just a boolean.
* PARTIAL RESULTS AS THEY LAND. One JSON line per checkpoint into
  ``<out>.checkpoints.jsonl``, flushed, plus an atomic rewrite of the artifact
  after every case (the progress-reporting rule).
* SEEDS FROM A DIGEST, never ``hash()``. ``hash()`` of a tuple containing a
  string is salted by PYTHONHASHSEED, so a hash-seeded case cannot be replayed
  from its own record.
* PLANTED DEFECTS WHOSE VERDICT IS SHAPE-DEPENDENT, and that is the whole
  argument of this file. ``guard_truncated_to_int16`` is what
  ``if (idx >= (short)(nx*ny*nz)) return;`` would do: at the gate fixtures'
  1,287 cells it is a NO-OP and MUST come back NULL; at 22,500 cells and above
  it MUST be CAUGHT. A leg that reported the same verdict at both sizes would
  not be measuring scale. The expectation is computed from the shape and
  compared with the measurement, so a defect that fails to bite where the
  arithmetic says it must is a harness failure, not a pass.
* A SIZE-INDEPENDENT CONTROL. The sibling gates' own sub-lattice mutation is
  carried at every shape and must be CAUGHT everywhere, so "caught at 11.2M" is
  not resting on a comparator that only wakes up when the array is big.

=============================================================================
RUNNING IT
=============================================================================

Laptop (no CUDA), the leg that is validated before a device slot is spent::

    PYTHONPATH=$(pwd) python -u \\
        parity/meep_gpu/stress_cuda_scale.py --backend numpy \\
        --out /tmp/scale_numpy/stress.json

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=0 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u parity/meep_gpu/stress_cuda_scale.py --backend cuda \\
        --subnormal-policy keep --out $OUT/keep/stress.json

``--product reduced`` is the shrunk form: the three largest shapes only, one
sub-step per family, one value class, one policy.

THE NumPy ARM COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the
NVRTC contraction guard exists for and it does not exercise the subnormal policy.
What it settles is whether the fixture, the predicate, the index arithmetic, the
launch geometry, the memory budget and the comparator all hold at corpus scale,
and whether this harness can fail -- all before a contended GPU is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_folded_curl as gate_curl  # noqa: E402
import gate_cuda_folded_constitutive as gate_const  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

#: The shim every gate uses: NumPy behind CuPy's ``__name__``. Imported, not
#: re-declared, so the three tracks' laptop legs stay commensurable.
_NumpyWearingCupysName = gate_const._NumpyWearingCupysName

SEED = 20260820

#: The int32 clause's constant, transcribed from coverage.py:1214. The bisection
#: below MEASURES the threshold and this is what it is compared against; it is
#: never used to decide anything.
TRANSCRIBED_INT32_CELL_BOUND = 2 ** 31

#: The complex arm's halved constant, transcribed from coverage.py:2878. The
#: kernels address ``2*idx`` and ``2*idx+1`` in ``int``, so ``2*ncells`` is what
#: must stay below 2**31.
TRANSCRIBED_INT32_WORD_BOUND = 2 ** 31

#: CUDA's gridDim.x ceiling. A 1-D launch is the only geometry either certified
#: launcher uses, so y and z (65535 each) are not reachable and are not checked.
CUDA_MAX_GRID_DIM_X = 2 ** 31 - 1

#: Where the corpus record is read from. ``_closed`` is the chain the 759/759
#: claim is cut from; ``_final`` is its predecessor and is accepted as a fallback
#: so this file still runs in a checkout that has only the older tree.
CORPUS_DIRS: Tuple[str, ...] = (
    "results/cuda_predicate_coverage_2026-08-20_closed",
    "results/cuda_predicate_coverage_2026-08-20_final",
)
CORPUS_FILES: Tuple[str, ...] = ("examples.jsonl", "tests.jsonl",
                                 "tests_param_matched.jsonl")

#: 0.5 is exactly representable in float32 and 0.35 is not; only the second can
#: distinguish a contracted expression from an uncontracted one. Scale is this
#: file's axis, so the sweep runs at the inexact courant and the exact one is a
#: control on the largest shape only -- multiplying every 11-million-cell case by
#: two courants buys a repetition, not a question.
INEXACT_COURANT = 0.35
EXACT_COURANT = 0.5

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: The fixture the gates use, carried as a control so the shape-dependent defect
#: verdicts have a small end to be null at. ``(13,11,9)`` is
#: ``test_step_curl_pml_real``'s and it is 1,287 cells.
GATE_FIXTURE_SHAPE: Tuple[int, int, int] = (13, 11, 9)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------
#
# Each family is the SIBLING GATE'S OWN entry points, bound here. Nothing in this
# file transcribes a kernel: ``run_kernel_numpy`` and ``run_kernel_cuda`` are the
# gates', ``build`` and ``seed_state`` are the gates', and the oracle is
# ``stepping`` itself. What this file owns is the SHAPE the fixture is built at.

def _curl_runner_args(sub_step: str, fields, layer, grid) -> Tuple[Any, ...]:
    """``(tables, codes, dtdx)`` for the curl launcher, resolved as a dispatch would.

    ``boundary_codes_for(grid, LICENSED_SUBSTITUTION)`` asks the SHIPPED resolver
    ``coverage.real_curl_boundary_codes`` -- the same function the predicate
    consults -- so a code split that drifted between predicate and harness cannot
    hide here.
    """
    codes, _kinds = gate_curl.boundary_codes_for(grid, gate_curl.LICENSED_SUBSTITUTION)
    return (gate_curl.tables_for(sub_step, layer), codes, float(grid.dt / grid.dx))


def _const_runner_args(side: str, fields, layer, grid) -> Tuple[Any, ...]:
    return (gate_const.tables_for(side, layer),)


FAMILIES: Dict[str, Dict[str, Any]] = {
    "curl": {
        "gate": gate_curl,
        "record_key": "cuda_curl",
        "sub_steps": ("step_B", "step_D"),
        "predicate_name": "cuda_kernels.coverage.covers_real_pml_curl",
        # The record's sub-step key IS this gate's argument. Spelled anyway, so
        # the two families are read the same way and neither is the special case.
        "to_gate": lambda s: s,
        "predicate": lambda f, p, g, s: coverage.covers_real_pml_curl(f, p, g, s),
        "oracle": lambda f, p, s: (stepping.step_B(f, p) if s == "step_B"
                                   else stepping.step_D(f, p)),
        "runner_args": _curl_runner_args,
        "kernel_module": "meep_gpu/cuda_kernels/step_curl_kernels.py",
        "threads_symbol": "_REAL_PML_THREADS",
        "needs_epsilon": lambda s: False,
        "sub_lattice_mutation": "swap_curl_sublattice",
    },
    "constitutive": {
        "gate": gate_const,
        "record_key": "cuda_constitutive",
        "sub_steps": ("update_H", "update_E"),
        "predicate_name": "cuda_kernels.coverage.covers_real_pml_constitutive",
        # The record says ``update_H``/``update_E``; the gate and the predicate
        # both take ``H``/``E``. Converted in one place rather than at each call
        # site, because a call site that forgot would ask the gate for a side it
        # has never heard of and the KeyError would arrive three frames away.
        "to_gate": lambda s: s.split("_", 1)[1],
        # The predicate takes the SIDE, so the record key is converted here and
        # not at four call sites.
        "predicate": lambda f, p, g, s: coverage.covers_real_pml_constitutive(
            f, p, g, s.split("_", 1)[1]),
        "oracle": lambda f, p, s: (stepping.update_H(f, p) if s == "update_H"
                                   else stepping.update_E(f, p)),
        "runner_args": _const_runner_args,
        "kernel_module": "meep_gpu/cuda_kernels/constitutive_kernels.py",
        "threads_symbol": "_CONSTITUTIVE_THREADS",
        # ``update_E``'s source is ``D * inv_eps``; against a table of ones
        # ``drop_inverse_epsilon`` is bit-identical, which is why the gate draws
        # three independent volumes away from 1.0 and why this leg calls it.
        "needs_epsilon": lambda s: s == "update_E",
        "sub_lattice_mutation": "swap_constitutive_sublattice",
    },
}

#: The sub-step each family's mutation legs are scored on when the product is
#: reduced. Chosen rather than sliced: ``update_E`` is the side with the inverse
#: epsilon and ``step_B`` the half-integer sub-lattice, so a reduced run still
#: carries one of each structure.
REDUCED_SUB_STEP: Dict[str, str] = {"curl": "step_B", "constitutive": "update_E"}


# ---------------------------------------------------------------------------
# The shipped launch geometry, read off the shipped source TEXT
# ---------------------------------------------------------------------------

def shipped_launch_facts(relative_path: str, threads_symbol: str) -> Dict[str, Any]:
    """Block size, guard expression and index form, regexed from the kernel module.

    READ AS TEXT, NOT IMPORTED. ``constitutive_kernels`` and ``step_curl_kernels``
    both ``import cupy`` at module scope (constitutive_kernels.py:143), so on the
    laptop -- which is where this leg is validated -- importing them is not
    available. The three facts this leg needs about the launch are lexical, and a
    regex over the shipped bytes is a MEASUREMENT of them rather than a
    transcription that can go stale in a docstring.

    If any of the three stops matching, the launch-geometry leg refuses instead of
    checking a geometry the module no longer has.
    """
    path = os.path.join(_REPO_API, relative_path)
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    threads = sorted({int(v) for v in re.findall(
        r"^\s*(?:threads|_CONSTITUTIVE_THREADS|_REAL_PML_THREADS) = (\d+)\s*$",
        source, re.M)})
    # THE MODULE'S OWN BLOCK-SIZE SYMBOL MUST BE PRESENT BY NAME. Without this the
    # regex above would happily report ``[256]`` from a bare ``threads = 256``
    # inside some other launcher while the certified pair's own constant had been
    # renamed or deleted, and the leg would check a geometry belonging to a
    # different kernel.
    named = re.search(rf"^{re.escape(threads_symbol)} = (\d+)\s*$", source, re.M)
    guards = sorted(set(re.findall(r"if \(idx >= ([^)]+)\) return;", source)))
    idx_forms = sorted(set(re.findall(r"int idx = ([^;]+);", source)))
    decomposition = sorted(set(re.findall(
        r"int i = (idx / \(ny \* nz\));", source)))
    return {
        "module": relative_path,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "threads_literals": threads,
        "threads_symbol": threads_symbol,
        "threads_symbol_value": int(named.group(1)) if named else None,
        "guard_expressions": guards,
        "flat_index_forms": idx_forms,
        "slow_axis_decompositions": decomposition,
        "usable": (threads == [256] and guards == ["nx * ny * nz"]
                   and idx_forms == ["blockIdx.x * blockDim.x + threadIdx.x"]
                   and named is not None and int(named.group(1)) == 256),
    }


def launch_geometry(cells: int, threads: int) -> Dict[str, Any]:
    """``blocks = ceil(cells/threads)`` and every int it makes the device form.

    THE THREE THINGS A POSTAGE STAMP CANNOT SHOW, all of them here:

    * ``blocks`` against CUDA's gridDim.x ceiling. 11,245,000 cells is 43,926
      blocks -- comfortably inside 2**31-1, and comfortably OUTSIDE the 65,535
      that a 16-bit grid dimension would allow, which is exactly why a launcher
      that ever narrowed the grid dimension would be invisible at 1,287 cells.
    * the top lane's ``int idx``, which is ``blocks*threads - 1`` and not
      ``cells - 1``: the guard runs AFTER the index is formed
      (step_curl_kernels.py:228-229), so the largest int the kernel constructs is
      the one the LAST BLOCK's last lane makes, past the end of the grid.
    * ``lanes_beyond_guard``, the tail the guard discards. Zero is not the healthy
      value -- it means ``cells`` is an exact multiple of 256 and this case cannot
      distinguish a guard that is off by a block.
    """
    blocks = (cells + threads - 1) // threads
    top_lane_index = blocks * threads - 1
    return {
        "cells": int(cells),
        "threads_per_block": int(threads),
        "blocks": int(blocks),
        "top_lane_flat_index": int(top_lane_index),
        "lanes_beyond_guard": int(blocks * threads - cells),
        "covers_every_cell": bool(blocks * threads >= cells),
        "no_wholly_idle_block": bool((blocks - 1) * threads < cells),
        "blocks_within_cuda_grid_dim_x": bool(blocks <= CUDA_MAX_GRID_DIM_X),
        "top_lane_index_within_int32": bool(top_lane_index < 2 ** 31),
        "guard_product_within_int32": bool(cells < 2 ** 31),
        # How much room the corpus has before a 16-bit grid dimension would stop
        # covering it. Recorded because it is the honest reading of the
        # ``grid_dim_x_capped_at_65535`` defect coming back NULL everywhere.
        "sixteen_bit_grid_dim_cover": int(65535 * threads),
        "fraction_of_sixteen_bit_cover": float(cells) / float(65535 * threads),
    }


# ---------------------------------------------------------------------------
# The index arithmetic, checked exhaustively rather than sampled
# ---------------------------------------------------------------------------

def index_expression_bounds(shape: Tuple[int, int, int], threads: int,
                            words_per_cell: int = 1) -> Dict[str, Any]:
    """Every int the certified kernels form, at its maximum, in exact int64.

    Transcribed from the two shipped bodies and named by the expression, so a
    reader can check each against the source rather than against this docstring:

    ``idx``            ``blockIdx.x * blockDim.x + threadIdx.x``  (:228)
    ``nx * ny * nz``   the guard                                   (:229)
    ``sx = ny * nz``   the slow stride                             (:231-233)
    ``idx + sx``       ``shift_up``'s neighbour, worst case        (prelude)
    ``idx + (na-1)*sx````shift_dn``'s periodic wrap-back           (prelude)
    ``2*idx + 1``      the complex arm's word offset               (coverage:2872)

    ``idx + sx`` IS LISTED AT ITS SYNTACTIC WORST CASE, ``cells - 1 + ny*nz``,
    not at the largest value it can actually reach. The branch guarantees
    ``ia + 1 < na`` before forming it, so the reachable maximum is ``cells - 1``;
    the syntactic one is carried because it is what the compiler is entitled to
    evaluate and because a bound that is only met by the reachable set is a bound
    that a later edit can break silently.
    """
    nx, ny, nz = (int(n) for n in shape)
    cells = nx * ny * nz
    blocks = (cells + threads - 1) // threads
    expressions = {
        "flat_index_top_lane": blocks * threads - 1,
        "guard_product_nx_ny_nz": cells,
        "slow_stride_ny_nz": ny * nz,
        "medium_stride_nz": nz,
        "shift_up_neighbour_worst_case": (cells - 1) + ny * nz,
        "shift_dn_wrap_back_worst_case": (cells - 1) + (nx - 1) * ny * nz,
        "complex_word_offset_2idx_plus_1": 2 * (cells - 1) + 1,
    }
    over = {name: value for name, value in expressions.items() if value >= 2 ** 31}
    return {
        "expressions": {k: int(v) for k, v in expressions.items()},
        "exceeding_int32": {k: int(v) for k, v in over.items()},
        "all_within_int32": not over,
        # The complex arm's expression is reported for every shape but is only a
        # REFUSAL for a complex run; a real-field row is not refused by it and
        # saying otherwise would read as a coverage loss that does not exist.
        "complex_word_offset_within_int32": bool(
            expressions["complex_word_offset_2idx_plus_1"] < 2 ** 31),
        "words_per_cell": int(words_per_cell),
    }


def decomposition_is_exact(shape: Tuple[int, int, int],
                           chunk: int = 4_194_304) -> Dict[str, Any]:
    """``k = idx%nz; j = (idx/nz)%ny; i = idx/(ny*nz)`` in int32, over EVERY index.

    EXHAUSTIVE, NOT SAMPLED, and chunked so an 11-million-cell grid costs a
    bounded working set rather than a 90 MB int64 temporary per term. Two things
    are checked at every index:

    * the int32 evaluation equals the int64 truth, term by term;
    * the round trip ``(i*ny + j)*nz + k`` reproduces ``idx``, which is the
      property the kernel actually depends on -- a decomposition can agree with
      int64 on each term and still fail to be a bijection if a product wrapped.

    C's ``/`` and ``%`` truncate toward zero and NumPy's floor toward minus
    infinity; they agree for non-negative operands, and ``idx`` is non-negative
    on every lane that passes the guard, so the NumPy evaluation is the C one
    here. A shape whose ``nx*ny*nz`` did not fit in int32 would break that, and
    such a shape is refused by the predicate before it reaches this function --
    which is the clause :func:`bisect_predicate_cell_bound` exercises.
    """
    nx, ny, nz = (int(n) for n in shape)
    cells = nx * ny * nz
    if cells >= 2 ** 31:
        return {"checked": 0, "exact": False,
                "why": f"{cells} cells cannot be decomposed in int32 at all"}
    plane = np.int32(ny * nz)
    nz32, ny32 = np.int32(nz), np.int32(ny)
    mismatches = 0
    checked = 0
    for start in range(0, cells, chunk):
        stop = min(start + chunk, cells)
        idx32 = np.arange(start, stop, dtype=np.int32)
        idx64 = idx32.astype(np.int64)
        k32 = idx32 % nz32
        j32 = (idx32 // nz32) % ny32
        i32 = idx32 // plane
        truth_k = idx64 % nz
        truth_j = (idx64 // nz) % ny
        truth_i = idx64 // (ny * nz)
        bad = ((k32 != truth_k) | (j32 != truth_j) | (i32 != truth_i))
        round_trip = (i32.astype(np.int64) * ny + j32) * nz + k32
        bad |= (round_trip != idx64)
        mismatches += int(np.count_nonzero(bad))
        checked += int(idx32.size)
    return {"checked": checked, "mismatches": mismatches,
            "exact": mismatches == 0,
            "note": "every index, not a sample; int32 evaluation vs int64 truth"}


# ---------------------------------------------------------------------------
# The corpus record: which shapes, and what configuration each one is
# ---------------------------------------------------------------------------

def corpus_dir(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in CORPUS_DIRS:
        path = os.path.join(_HERE, candidate)
        if os.path.isdir(path):
            return path
    raise SystemExit(
        "no corpus coverage record found; pass --corpus-dir explicitly. Tried: "
        + ", ".join(CORPUS_DIRS))


def read_corpus(directory: str) -> List[Dict[str, Any]]:
    """Every measured row, from whichever of the three ladders the tree carries."""
    rows: List[Dict[str, Any]] = []
    for name in CORPUS_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_ladder"] = name
                rows.append(row)
    if not rows:
        raise SystemExit(f"corpus record {directory} carried no rows")
    return rows


def corpus_shape_census(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The size distribution the sweep is selecting against, measured not quoted."""
    cells = sorted(int(r["grid_cells"]) for r in rows if r.get("grid_cells"))
    if not cells:
        return {"rows": 0}
    return {
        "rows": len(cells),
        "min_cells": cells[0],
        "median_cells": int(cells[len(cells) // 2]),
        "max_cells": cells[-1],
        "gate_fixture_cells": int(np.prod(GATE_FIXTURE_SHAPE)),
        "max_over_gate_fixture": float(cells[-1]) / float(np.prod(GATE_FIXTURE_SHAPE)),
        "rows_above_int16": sum(1 for c in cells if c >= 2 ** 15),
        "rows_above_sixteen_bit_block_cover": sum(1 for c in cells
                                                  if c > 65535 * 256),
    }


def admitted_slots(rows: List[Dict[str, Any]], record_key: str,
                   sub_step: str) -> List[Dict[str, Any]]:
    """Rows the record says this family admits at this sub-step, modulo the backend.

    ``covered_modulo_backend`` and not ``covered``: the record was measured on a
    laptop, so every row's raw verdict is "backend is not CuPy". The
    modulo-backend column is the one the 759/759 claim is counted from.
    """
    out = []
    for row in rows:
        block = row.get(record_key)
        if not isinstance(block, dict):
            continue
        slot = block.get(sub_step)
        if not isinstance(slot, dict) or not slot.get("covered_modulo_backend"):
            continue
        if not row.get("configuration") or not row.get("grid_shape"):
            continue
        out.append(row)
    return out


def spec_from_configuration(row: Dict[str, Any], label: str,
                            phase: int = 1) -> Optional[Dict[str, Any]]:
    """A gate-shaped fixture spec that reproduces this corpus row's STORED shape.

    ``grid_shape`` in the record is ``grid.shape`` -- the STORED extent, already
    halved on any mirrored axis -- and that is the extent the kernel indexes, so
    it is the thing this leg has to reproduce. The full ``cell_size`` that
    produces it is SOLVED FOR rather than assumed: ``Grid``'s fold gives
    ``stored = n - n//2 + 1`` on a mirrored axis, but the resolved termination
    moves the answer by one (measured: a stored 402 comes from a full 802 under a
    metallic Y and from a full 801 under a periodic one), so the solver searches a
    small window and the case is REFUSED if no full extent reproduces the record's
    shape. A case that silently ran a neighbouring shape would be reporting a
    number about a grid the corpus does not contain.

    THE MIRROR PHASE IS NOT IN THE RECORD. ``configuration`` carries ``mirrored``
    but not the phase, so +1 is a fixture choice and is declared as one; it enters
    the ghost fill, which is a separate driver pass from every sub-step here.
    """
    configuration = row["configuration"]
    shape = tuple(int(n) for n in row["grid_shape"])
    if len(shape) != 3:
        return None
    if configuration.get("cylindrical") or any(configuration.get("is_axis", (0, 0, 0))):
        return None  # both predicates refuse a Dcyl grid; not this leg's family
    mirrored = [bool(v) for v in configuration["mirrored"]]
    metallic = [bool(v) for v in configuration["metallic"]]
    boundaries = tuple("metallic" if metallic[a] else "periodic" for a in range(3))
    axes = "".join("XYZ"[a] for a in range(3) if mirrored[a])
    cell = _solve_cell(shape, mirrored, boundaries, axes, phase)
    if cell is None:
        return None
    return {
        "label": label,
        "axes": axes,
        "phase": phase,
        "boundaries": boundaries,
        "cell": cell,
        "target_shape": shape,
    }


def _solve_cell(shape: Tuple[int, int, int], mirrored: List[bool],
                boundaries: Tuple[str, str, str], axes: str,
                phase: int) -> Optional[Tuple[float, float, float]]:
    """Full cell extents whose built ``Grid.shape`` equals ``shape`` exactly."""
    candidates: List[List[int]] = []
    for axis in range(3):
        want = shape[axis]
        if not mirrored[axis]:
            candidates.append([want])
        else:
            base = 2 * (want - 1)
            candidates.append([n for n in range(max(1, base - 3), base + 4)])
    planes = tuple(Mirror(name, phase) for name in axes)
    for nx in candidates[0]:
        for ny in candidates[1]:
            for nz in candidates[2]:
                cell = (float(nx), float(ny), float(nz))
                try:
                    grid = Grid(resolution=1.0, cell_size=cell,
                                boundaries=boundaries, symmetry=planes,
                                xp=np, courant=INEXACT_COURANT)
                except Exception:  # noqa: BLE001 - an unbuildable candidate is not a match
                    continue
                if tuple(int(n) for n in grid.shape) == shape:
                    return cell
    return None


def select_subjects(rows: List[Dict[str, Any]], family: str, sub_step: str,
                    top_n: int) -> List[Dict[str, Any]]:
    """The largest admitted shapes for this slot, plus the median and the gate fixture.

    THE SMALL END IS NOT DECORATION. Every shape-dependent defect in this file has
    a threshold, and a sweep that only ran the big shapes could report "CAUGHT"
    without ever showing that the same defect is INVISIBLE at the size the
    existing gates run -- which is the entire claim this leg exists to make.
    """
    record_key = FAMILIES[family]["record_key"]
    admitted = admitted_slots(rows, record_key, sub_step)
    admitted.sort(key=lambda r: -int(r["grid_cells"]))

    chosen: List[Dict[str, Any]] = []
    seen: set = set()
    for row in admitted:
        key = (tuple(row["grid_shape"]),
               tuple(row["configuration"]["mirrored"]),
               tuple(row["configuration"]["metallic"]))
        if key in seen:
            continue
        spec = spec_from_configuration(row, f"corpus_max_{len(chosen) + 1}")
        if spec is None:
            continue
        seen.add(key)
        spec["label"] = f"{row['row']}"
        chosen.append({"row": row["row"], "ladder": row["_ladder"],
                       "cells": int(row["grid_cells"]), "spec": spec,
                       "provenance": "corpus_largest"})
        if len(chosen) >= top_n:
            break

    # THE ASPECT-RATIO CORNERS, chosen by shape rather than by size. The
    # decomposition ``k = idx%nz; j = (idx/nz)%ny; i = idx/(ny*nz)`` is a
    # different arithmetic on a long thin grid than on a cube even at the same
    # cell count: 220x1840x1 puts almost the whole index in ``j`` with ``nz = 1``,
    # so ``idx/nz`` is the identity and ``ny`` carries the range, while 89^3
    # spreads it across all three terms. A sweep ranked purely by cell count
    # misses that -- measured: at ``--shapes-per-family 3`` the curl family's
    # selection stops one row above 220x1840x1 -- so the two corners are selected
    # by name.
    #
    # THE CORNERS ARE SEARCHED AMONG THE LARGE ROWS ONLY -- the admitted rows at
    # or above the 75th percentile of cell count. Searched over the whole set, the
    # widest single axis in the curl family is ``3rd-harm-1d.py`` at (1,1,2500):
    # a 2,500-cell 1-D run whose aspect is extreme and whose SIZE is a fifth of
    # the gate fixtures' largest. That is the wrong corner -- the question is what
    # an extreme aspect does at scale, not what it does at 2,500 cells -- and the
    # percentile floor is what keeps the two properties on the same case.
    admitted_cells = sorted(int(r["grid_cells"]) for r in admitted)
    floor = (admitted_cells[int(0.75 * (len(admitted_cells) - 1))]
             if admitted_cells else 0)
    large = [r for r in admitted if int(r["grid_cells"]) >= floor]
    for label, key in (("corpus_widest_axis",
                        lambda r: max(int(n) for n in r["grid_shape"])),
                       ("corpus_extreme_aspect",
                        lambda r: (max(int(n) for n in r["grid_shape"])
                                   / max(1, min(int(n) for n in r["grid_shape"]))))):
        for row in sorted(large, key=key, reverse=True):
            shape_key = (tuple(row["grid_shape"]),
                         tuple(row["configuration"]["mirrored"]),
                         tuple(row["configuration"]["metallic"]))
            if shape_key in seen:
                continue
            spec = spec_from_configuration(row, row["row"])
            if spec is None:
                continue
            seen.add(shape_key)
            chosen.append({"row": row["row"], "ladder": row["_ladder"],
                           "cells": int(row["grid_cells"]), "spec": spec,
                           "provenance": label})
            break

    # The median row, as the corpus's centre of mass.
    cells = sorted(int(r["grid_cells"]) for r in admitted)
    if cells:
        median = cells[len(cells) // 2]
        for row in admitted:
            if int(row["grid_cells"]) != median:
                continue
            shape_key = (tuple(row["grid_shape"]),
                         tuple(row["configuration"]["mirrored"]),
                         tuple(row["configuration"]["metallic"]))
            if shape_key in seen:
                continue
            spec = spec_from_configuration(row, row["row"])
            if spec is None:
                continue
            seen.add(shape_key)
            chosen.append({"row": row["row"], "ladder": row["_ladder"],
                           "cells": int(row["grid_cells"]), "spec": spec,
                           "provenance": "corpus_median"})
            break

    # The gate fixture, so the defects' small end is the size the record was cut
    # at rather than merely a small number.
    chosen.append({
        "row": "gate_fixture_control",
        "ladder": "not_a_corpus_row",
        "cells": int(np.prod(GATE_FIXTURE_SHAPE)),
        "provenance": "gate_fixture",
        "spec": {"label": "gate_fixture_control", "axes": "", "phase": 1,
                 "boundaries": ("periodic", "periodic", "periodic"),
                 "cell": tuple(float(n) for n in GATE_FIXTURE_SHAPE),
                 "target_shape": GATE_FIXTURE_SHAPE},
    })
    return chosen


# ---------------------------------------------------------------------------
# Memory: predicted before allocation, measured after
# ---------------------------------------------------------------------------

def measure_words_per_cell(family: str, sub_step: str) -> Dict[str, Any]:
    """How many float32 words per cell this family's fixture allocates.

    MEASURED on an 8x8x8 build of the same family rather than counted off a list
    of array names, so an array added to ``Fields`` later is included without
    anyone remembering to add it here. The number is then multiplied by the case's
    cell count to PREDICT the footprint, which is what makes "skip with a reason"
    possible instead of "die in the allocator".
    """
    gate = FAMILIES[family]["gate"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    spec = {"label": "probe", "axes": "", "phase": 1,
            "boundaries": ("periodic", "periodic", "periodic"),
            "cell": (8.0, 8.0, 8.0)}
    fields, layer, grid = gate.build(_NumpyWearingCupysName(), spec, INEXACT_COURANT)
    cells = int(np.prod(grid.shape))
    total = 0
    counted = 0
    for name in dir(fields):
        if name.startswith("__"):
            continue
        try:
            value = getattr(fields, name)
        except Exception:  # noqa: BLE001 - a property that raises is not an array
            continue
        if isinstance(value, np.ndarray) and value.size >= cells:
            total += int(value.nbytes)
            counted += 1
    words = total / 4.0 / cells
    state = len(gate.state_names(gate_arg))
    return {
        "measured_on_shape": [8, 8, 8],
        "field_arrays_counted": counted,
        "field_words_per_cell": words,
        "frozen_snapshot_words_per_cell": float(state),
        # The oracle's rolling checkpoint copy is the OUTPUT set only.
        "checkpoint_copy_words_per_cell": float(len(gate.outputs(gate_arg))),
        "total_words_per_cell": words + state + len(gate.outputs(gate_arg)),
    }


def predicted_bytes(cells: int, words_per_cell: float) -> int:
    return int(math.ceil(cells * words_per_cell * 4))


def host_peak_rss_bytes() -> int:
    """Peak resident set of THIS process, in bytes.

    Recorded per case because the predicted footprint is the FIXTURE's -- the
    arrays ``Fields``/``PML`` hold plus the frozen snapshot -- and the array
    path's own temporaries are on top of it. Measured on the laptop: an
    11,245,000-cell curl case predicts 1.84 GB of fixture and peaks at 2.70 GB
    resident, so a budget set from the prediction alone is about 1.5x optimistic.
    The number is recorded rather than folded into the prediction, because the
    transient factor is a property of the ARRAY PATH's temporaries and a device
    run does not pay it in the same place.

    ``ru_maxrss`` is bytes on Darwin and kilobytes on Linux; both are handled so
    the laptop leg and the device leg report the same unit.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def device_memory_snapshot() -> Optional[Dict[str, Any]]:
    """Peak and current device memory, or None on the NumPy backend."""
    if cp is None:
        return None
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "mempool_used_bytes": int(pool.used_bytes()),
        "mempool_total_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


# ---------------------------------------------------------------------------
# The planted defects: thresholds, so their verdict depends on the SHAPE
# ---------------------------------------------------------------------------

def _int16_truncate(value: int) -> int:
    """``(short)value`` -- two's-complement truncation to 16 bits."""
    return ((value + 2 ** 15) % 2 ** 16) - 2 ** 15


def defect_threshold(name: str, cells: int, threads: int) -> Optional[int]:
    """The flat index past which a defective kernel writes nothing.

    Each of the three is a NARROWING of one int in the launch path, and each has a
    threshold that is a function of the SHAPE. That is the point: at the gate
    fixtures' 1,287 cells all three thresholds sit past the end of the grid and
    every one of them is a no-op, which is why 26 gates could not have seen any of
    them.

    ``guard_truncated_to_int16``   ``if (idx >= (short)(nx*ny*nz)) return;``
    ``grid_dim_x_capped_at_65535`` a 16-bit gridDim.x: 65,535 blocks of 256
    ``flat_index_wraps_at_int32``  the clause the predicates refuse past 2**31
    """
    if name == "guard_truncated_to_int16":
        truncated = _int16_truncate(cells)
        return max(0, truncated)
    if name == "grid_dim_x_capped_at_65535":
        return 65535 * threads
    if name == "flat_index_wraps_at_int32":
        return 2 ** 31
    raise KeyError(f"unknown scale defect {name!r}")


SCALE_DEFECTS: Tuple[str, ...] = (
    "guard_truncated_to_int16",
    "grid_dim_x_capped_at_65535",
    "flat_index_wraps_at_int32",
)


def apply_partial_cover(xp, fields, before: Dict[str, np.ndarray],
                        names: Sequence[str], threshold: int) -> int:
    """Restore every output word at flat index >= ``threshold`` to its pre-launch value.

    THIS IS THE DEFECT, EXACTLY. A lane that returns early before touching memory
    leaves the word it owns exactly as the previous launch left it, and that is
    what this does -- from the state snapshotted immediately before the launch, so
    a multi-launch leg accumulates the defect the way a real one would rather than
    reverting to the frozen state each time.

    It is applied to the KERNEL PASS ONLY. The oracle is ``stepping``, untouched.
    """
    changed = 0
    for name in names:
        array = getattr(fields, name)
        flat = array.reshape(-1)
        if threshold >= flat.size:
            continue
        original = before[name].reshape(-1)
        # ``before`` is a HOST copy on both backends (``to_host``), so the tail
        # goes back through ``xp.asarray``: assigning a NumPy slice straight into
        # a CuPy view is the one line in this file that would work on the laptop
        # and refuse on the device.
        flat[threshold:] = xp.asarray(np.ascontiguousarray(original[threshold:]))
        changed += int(flat.size - threshold)
    return changed


# ---------------------------------------------------------------------------
# Digests, so a checkpoint costs a hash and not a second copy of the volume
# ---------------------------------------------------------------------------

def word_digest(array: Any) -> str:
    """sha256 over the raw uint32 words of one array.

    NEVER ``allclose`` and never a float comparison: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately. The digest is over the same uint32 view
    ``probe.bit_compare`` compares, so a checkpoint that agrees here agrees there.
    """
    host = to_host(array)
    contiguous = np.ascontiguousarray(host, dtype=np.float32).ravel()
    return hashlib.sha256(contiguous.view(np.uint32).tobytes()).hexdigest()


def state_digests(fields, names: Sequence[str]) -> Dict[str, str]:
    return {name: word_digest(getattr(fields, name)) for name in names}


def moved_fraction(before: Dict[str, np.ndarray], fields,
                   names: Sequence[str]) -> float:
    """Fraction of output WORDS that differ from ``before``, compared as uint32."""
    moved = total = 0
    for name in names:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(to_host(getattr(fields, name)),
                                 dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_rng(family: str, sub_step: str, label: str, courant: float,
             value_class: str) -> np.random.Generator:
    """A replayable seed. NOT ``hash()``.

    ``hash()`` of a tuple containing a string is salted by PYTHONHASHSEED --
    measured on this project, one key gave four distinct seeds in four
    interpreters -- so a hash-seeded case cannot be replayed from its own record.
    sha256 of the same key is the same number in every process forever.
    """
    key = f"{family}|{sub_step}|{label}|{courant}|{value_class}"
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"))


def one_case(backend: str, family: str, sub_step: str, subject: Dict[str, Any],
             courant: float, value_class: str, launches: int,
             checkpoint_every: int, memory_budget_bytes: int,
             words_per_cell: Dict[str, Any],
             defect: Optional[str] = None,
             host_mutation: Optional[str] = None,
             emit: Optional[Callable[[Dict[str, Any]], None]] = None
             ) -> Dict[str, Any]:
    """One frozen state at one corpus shape, run twice, compared at every checkpoint.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects,
    and the kernel arm is the SIBLING GATE'S runner. There is no transcription in
    this file to drift.

    ONE ``Fields`` OBJECT, RUN TWICE FROM THE SAME FROZEN STATE. Two objects would
    mean two epsilon draws unless copied across, and at 11 million cells it would
    also mean a second gigabyte. The oracle pass records a digest per checkpoint;
    the kernel pass records the same digests; the first checkpoint whose digests
    differ is then LOCALIZED with a full ``bit_compare`` so the record carries the
    component and the ULP gap and not only a boolean.
    """
    started = time.time()
    gate = FAMILIES[family]["gate"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    spec = subject["spec"]
    threads = subject["threads"]
    cells = int(np.prod(spec["target_shape"]))

    case: Dict[str, Any] = {
        "family": family, "sub_step": sub_step, "backend": backend,
        "row": subject["row"], "ladder": subject["ladder"],
        "provenance": subject["provenance"],
        "target_shape": [int(n) for n in spec["target_shape"]],
        "cells": cells,
        "courant": courant, "value_class": value_class,
        "defect": defect, "host_mutation": host_mutation,
        "launches": launches, "checkpoint_every": checkpoint_every,
        "fold_axes": spec["axes"], "boundaries": list(spec["boundaries"]),
        "mirror_phase_is_a_fixture_choice": spec["phase"],
    }

    # THE MEMORY GATE COMES FIRST, before anything is allocated. A case that
    # cannot fit is a SKIP WITH A REASON; dying in the allocator would take the
    # artifact and every case after it down with it.
    predicted = predicted_bytes(cells, words_per_cell["total_words_per_cell"])
    case["memory"] = {
        "predicted_bytes": predicted,
        "budget_bytes": int(memory_budget_bytes),
        "words_per_cell": words_per_cell,
    }
    if predicted > memory_budget_bytes:
        case["skipped"] = (
            f"predicted footprint {predicted / 1e9:.2f} GB exceeds the "
            f"{memory_budget_bytes / 1e9:.2f} GB budget; raise --memory-budget-gb "
            f"or run this shape on a device with the room")
        case["seconds"] = time.time() - started
        return case

    # THE LAUNCH GEOMETRY AND THE INDEX ARITHMETIC, both before the fixture: they
    # are properties of the shape, they cost nothing, and a shape that fails
    # either one should be recorded as failing it rather than as an allocation.
    case["launch_geometry"] = launch_geometry(cells, threads)
    case["index_expressions"] = index_expression_bounds(
        tuple(spec["target_shape"]), threads)

    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = case_rng(family, sub_step, spec["label"], courant, value_class)

    fields, layer, grid = gate.build(xp, spec, courant)
    built = tuple(int(n) for n in grid.shape)
    if built != tuple(spec["target_shape"]):
        case["skipped"] = (f"the fixture built {built}, not the corpus row's "
                           f"{tuple(spec['target_shape'])}; a case that ran a "
                           f"neighbouring shape would report a number about a "
                           f"grid the corpus does not contain")
        case["seconds"] = time.time() - started
        return case

    if FAMILIES[family]["needs_epsilon"](sub_step):
        gate.install_epsilon(fields, grid, rng)
    host = gate.seed_state(fields, grid, gate_arg, value_class, rng)
    case["operand_census"] = operand_census(host)
    case["structure"] = gate.structure_facts(grid)
    if backend == "cuda":
        case["memory"]["device_after_build"] = device_memory_snapshot()

    covered, reason = FAMILIES[family]["predicate"](fields, layer, grid, sub_step)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason,
                               "predicate": FAMILIES[family]["predicate_name"]}
    # THE PREDICATE GATES THIS LEG, unlike the gates that exist to change one.
    # This file is not asking whether a refusal is deserved; it is asking whether
    # an ADMITTED configuration survives its own size. A case the predicate
    # refuses for anything but the backend clause is not this leg's business.
    if not covered and not _is_backend_clause(reason):
        case["skipped"] = f"the predicate refuses this configuration: {reason}"
        case["seconds"] = time.time() - started
        return case

    case["decomposition"] = decomposition_is_exact(tuple(spec["target_shape"]))

    outputs = gate.outputs(gate_arg)
    frozen = gate.snapshot(fields, gate_arg)
    runner_args = FAMILIES[family]["runner_args"](gate_arg, fields, layer, grid)
    if host_mutation is not None:
        tables = gate.host_mutated_tables(host_mutation, gate_arg, layer, grid)
        runner_args = (tables,) + tuple(runner_args[1:])

    checkpoints = _checkpoint_schedule(launches, checkpoint_every)
    case["checkpoint_launches"] = checkpoints

    # ---- Pass 1: the oracle -------------------------------------------------
    oracle = FAMILIES[family]["oracle"]
    previous = {name: frozen[name].copy() for name in outputs}
    oracle_records: List[Dict[str, Any]] = []
    for launch in range(1, launches + 1):
        oracle(fields, layer, sub_step)
        gate.advance_sources(fields, gate_arg)
        if launch not in checkpoints:
            continue
        record = {
            "launch": launch,
            "moved_since_frozen": moved_fraction(frozen, fields, outputs),
            "moved_since_previous_checkpoint": moved_fraction(previous, fields,
                                                              outputs),
            "digests": state_digests(fields, outputs),
        }
        oracle_records.append(record)
        previous = {name: to_host(getattr(fields, name)).copy() for name in outputs}
        if emit is not None:
            emit({"case": _case_key(case), "pass": "oracle", **record})
    case["oracle_checkpoints"] = oracle_records

    # NON-VACUITY, PER CHECKPOINT. A zero-initialised constitutive leaves every
    # word +0.0 forever and a deliberately wrong reference still reports
    # IDENTICAL; half of one earlier gate's cases could not fail for exactly this
    # reason. A recurrence that has SETTLED is the same failure one launch later,
    # which is why the previous-checkpoint delta is a floor too and not a note.
    dead = [r["launch"] for r in oracle_records if r["moved_since_frozen"] == 0.0]
    settled = [r["launch"] for r in oracle_records
               if r["moved_since_previous_checkpoint"] == 0.0]
    if dead or settled:
        case["skipped"] = (
            f"the array path moved no output word at checkpoint(s) "
            f"{dead or settled}; a case that moved nothing certifies nothing")
        case["non_vacuity"] = {"dead_checkpoints": dead,
                               "settled_checkpoints": settled}
        case["seconds"] = time.time() - started
        return case
    case["non_vacuity"] = {
        "dead_checkpoints": [], "settled_checkpoints": [],
        "min_moved_since_frozen": min(r["moved_since_frozen"]
                                      for r in oracle_records),
        "min_moved_since_previous": min(r["moved_since_previous_checkpoint"]
                                        for r in oracle_records),
    }

    # ---- Pass 2: the kernel, from the SAME frozen state ---------------------
    gate.restore(fields, frozen)
    runner = (gate.run_kernel_cuda if backend == "cuda" else gate.run_kernel_numpy)
    threshold = defect_threshold(defect, cells, threads) if defect else None
    case["defect_threshold"] = threshold
    kernel_records: List[Dict[str, Any]] = []
    defect_words = 0
    for launch in range(1, launches + 1):
        pre = ({name: to_host(getattr(fields, name)).copy() for name in outputs}
               if defect is not None else None)
        runner(gate_arg, fields, *runner_args)
        if defect is not None:
            defect_words += apply_partial_cover(xp, fields, pre, outputs, threshold)
        gate.advance_sources(fields, gate_arg)
        if launch not in checkpoints:
            continue
        record = {"launch": launch,
                  "moved_since_frozen": moved_fraction(frozen, fields, outputs),
                  "digests": state_digests(fields, outputs)}
        kernel_records.append(record)
        if emit is not None:
            emit({"case": _case_key(case), "pass": "kernel", **record})
    case["kernel_checkpoints"] = kernel_records
    if defect is not None:
        case["defect_words_reverted"] = defect_words
        # A defect that reverted NOTHING is a defect that was never armed at this
        # shape. That is a legitimate outcome -- it is what a threshold past the
        # end of the grid means -- but it must be recorded as "not armed" rather
        # than allowed to read as "the kernel survived it".
        case["defect_armed"] = defect_words > 0

    # ---- The comparison, checkpoint by checkpoint ---------------------------
    per_checkpoint = []
    first_divergence = None
    for oracle_record, kernel_record in zip(oracle_records, kernel_records):
        differing = sorted(name for name in outputs
                           if oracle_record["digests"][name]
                           != kernel_record["digests"][name])
        entry = {"launch": oracle_record["launch"],
                 "identical": not differing,
                 "differing_components": differing}
        per_checkpoint.append(entry)
        if differing and first_divergence is None:
            first_divergence = oracle_record["launch"]
    case["per_checkpoint"] = per_checkpoint
    case["bit_identical"] = all(entry["identical"] for entry in per_checkpoint)
    case["first_divergent_launch"] = first_divergence

    # WHERE, not just WHETHER. A long run that reports one verdict at the end
    # cannot say where it diverged, and where is the whole value -- so the first
    # differing checkpoint is re-run on both paths and compared word for word.
    if first_divergence is not None:
        case["localized"] = _localize(gate, family, sub_step, fields, layer,
                                      frozen, runner, runner_args, outputs,
                                      first_divergence, defect, threshold, xp)

    if backend == "cuda":
        case["memory"]["device_peak"] = device_memory_snapshot()
    case["memory"]["host_peak_rss_bytes"] = host_peak_rss_bytes()
    case["memory"]["host_peak_over_predicted"] = (
        case["memory"]["host_peak_rss_bytes"] / predicted if predicted else None)
    case["seconds"] = time.time() - started
    return case


def _case_key(case: Dict[str, Any]) -> str:
    return (f"{case['family']}/{case['sub_step']}/{case['row']}/"
            f"{'x'.join(str(n) for n in case['target_shape'])}/"
            f"{case['value_class']}/c{case['courant']}"
            + (f"/defect={case['defect']}" if case.get("defect") else "")
            + (f"/mut={case['host_mutation']}" if case.get("host_mutation") else ""))


def _checkpoint_schedule(launches: int, every: int) -> List[int]:
    """Launch 1 always, then every ``every``, and the last launch always.

    LAUNCH 1 IS A CHECKPOINT because it is the single-launch verdict every gate
    already reports, and a stress leg that could not reproduce it would be
    measuring something else. The LAST launch is always one because a schedule
    that stopped short would leave the deepest state uncompared.
    """
    marks = {1, launches}
    marks.update(range(every, launches + 1, every))
    return sorted(m for m in marks if 1 <= m <= launches)


def _localize(gate, family: str, sub_step: str, fields, layer,
              frozen: Dict[str, np.ndarray], runner, runner_args,
              outputs: Sequence[str], launch: int, defect: Optional[str],
              threshold: Optional[int], xp) -> Dict[str, Any]:
    """Re-run both paths to ``launch`` and compare word for word."""
    oracle = FAMILIES[family]["oracle"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    gate.restore(fields, frozen)
    for _ in range(launch):
        oracle(fields, layer, sub_step)
        gate.advance_sources(fields, gate_arg)
    reference = {name: to_host(getattr(fields, name)).copy() for name in outputs}

    gate.restore(fields, frozen)
    for _ in range(launch):
        pre = ({name: to_host(getattr(fields, name)).copy() for name in outputs}
               if defect is not None else None)
        runner(gate_arg, fields, *runner_args)
        if defect is not None:
            apply_partial_cover(xp, fields, pre, outputs, threshold)
        gate.advance_sources(fields, gate_arg)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs}
    out = combine(parts)
    out["at_launch"] = launch
    return out


def _is_backend_clause(reason: Any) -> bool:
    return isinstance(reason, str) and "backend is not CuPy" in reason


# ---------------------------------------------------------------------------
# The int32 bound, bisected against the predicate itself
# ---------------------------------------------------------------------------

class _ShapeSubstituted:
    """A real object, answering one question differently: its shape.

    WHY A PROXY AND NOT A FABRICATION. The int32 clause refuses a cell count no
    machine can allocate -- 2**31 float32 words is 8.6 GB PER ARRAY and the
    fixture holds two dozen -- so the clause cannot be reached by building the
    grid it describes. Every OTHER clause the predicate evaluates is answered here
    by a REAL ``Grid``/``Fields``/``PML`` built at a small shape; only the extent
    is substituted. That keeps the instrument one substitution away from the
    thing it measures instead of a hand-written stand-in for all of it.

    WHAT MAKES IT CREDIBLE IS THE SOUNDNESS LEG, not this docstring:
    :func:`standin_soundness` sets the substituted shape EQUAL to the real shape
    and requires the proxy's verdict to equal the real objects' verdict, at every
    shape the sweep ran. A disagreement anywhere marks the bound leg UNSOUND and
    its numbers are withheld rather than reported.
    """

    def __init__(self, target: Any, shape: Tuple[int, int, int]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_shape", tuple(int(n) for n in shape))

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_target"), item)


class _ShapedArray:
    """A real array wearing a different ``shape``.

    ``dtype``, ``flags`` and everything else are the REAL array's; ``shape``,
    ``size`` and ``ndim`` are derived from the substituted extent, because those
    three are the only questions the predicates ask that a substituted shape can
    answer differently. ``size`` is derived rather than passed through for the
    reason ``_coefficient_vector_problem`` exists (coverage.py:772): it compares
    ``vector.size`` against ``shape[axis]`` BEFORE it compares the broadcast
    shape, so a proxy that reported the real size would be refused for a length
    reason at every substituted extent and the bisection would never reach the
    int32 clause.
    """

    def __init__(self, array: Any, shape: Tuple[int, ...]):
        self._array = array
        self.shape = tuple(int(n) for n in shape)

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def flags(self):
        return self._array.flags

    @property
    def size(self) -> int:
        product = 1
        for n in self.shape:
            product *= int(n)
        return product

    @property
    def ndim(self) -> int:
        return len(self.shape)


class _GridProxy(_ShapeSubstituted):
    @property
    def shape(self):
        return object.__getattribute__(self, "_shape")

    def stored_cells(self, axis: int) -> int:
        return object.__getattribute__(self, "_shape")[axis]

    def owned_cells(self, axis: int) -> int:
        # The real grid's stored-past-owned relationship, preserved: an axis the
        # real grid stores one slot past its owned window keeps that slot here,
        # because the fold clause reads exactly that difference.
        target = object.__getattribute__(self, "_target")
        delta = int(target.stored_cells(axis)) - int(target.owned_cells(axis))
        return object.__getattribute__(self, "_shape")[axis] - delta


#: Methods the predicates call on ``fields`` that RETURN a volume:
#: ``inverse_epsilon_for`` (coverage.py:1614) and ``condfac_for`` (:1169). A proxy
#: that wrapped only attributes would hand back a real-shaped volume from these
#: and be refused for a shape reason before the extent clause was ever reached.
_VOLUME_RETURNING_METHODS: Tuple[str, ...] = ("inverse_epsilon_for", "condfac_for")


class _FieldsProxy(_ShapeSubstituted):
    def __getattr__(self, item):
        value = getattr(object.__getattribute__(self, "_target"), item)
        shape = object.__getattribute__(self, "_shape")
        if item in _VOLUME_RETURNING_METHODS and callable(value):
            def wrapped(*args, _inner=value, _shape=shape, **kwargs):
                result = _inner(*args, **kwargs)
                if result is None:
                    return None
                return _ShapedArray(result, _shape)
            return wrapped
        if hasattr(value, "dtype") and getattr(value, "ndim", None) == 3:
            return _ShapedArray(value, shape)
        return value


#: A PML coefficient vector's axis, read off its NAME. ``PML`` spells them
#: ``kms_x``, ``kps_y_h``, ``sinv_z`` and so on, so the axis is in the attribute.
_PML_VECTOR_AXIS = re.compile(r"_(?P<axis>[xyz])(?:_h)?$")


class _PmlProxy(_ShapeSubstituted):
    def __getattr__(self, item):
        value = getattr(object.__getattribute__(self, "_target"), item)
        shape = object.__getattribute__(self, "_shape")
        if hasattr(value, "dtype") and getattr(value, "ndim", None) == 3:
            # THE AXIS COMES FROM THE NAME, NOT FROM THE BROADCAST SHAPE. Reading
            # it off the shape -- "the first axis whose extent is not 1" -- is
            # correct on a fixture whose every axis is thicker than one cell and
            # WRONG on a 2-D corpus row: a z vector on a grid with nz = 1 is
            # ``(1,1,1)``, every axis reads as 1, the fallback picks axis 0, and
            # the proxy hands back an x-length z vector. Measured: it made the
            # identity soundness arm fail at (150,150,1) and (160,91,1) and pass
            # everywhere else, which is exactly the shape class the corpus is
            # mostly made of.
            match = _PML_VECTOR_AXIS.search(item)
            if match is None:
                return _ShapedArray(value, shape)
            axis = "xyz".index(match.group("axis"))
            new = [1, 1, 1]
            new[axis] = shape[axis]
            return _ShapedArray(value, tuple(new))
        return value


def _proxied(fields, layer, grid, shape: Tuple[int, int, int]):
    return (_FieldsProxy(fields, shape), _PmlProxy(layer, shape),
            _GridProxy(grid, shape))


def standin_soundness(family: str, sub_step: str, fields, layer, grid,
                      shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """Proxy verdict vs real verdict, at the SAME shape. Must agree."""
    predicate = FAMILIES[family]["predicate"]
    real = predicate(fields, layer, grid, sub_step)
    pf, pp, pg = _proxied(fields, layer, grid, shape)
    proxied = predicate(pf, pp, pg, sub_step)
    return {
        "shape": [int(n) for n in shape],
        "real": {"covered": bool(real[0]), "reason": real[1]},
        "proxied": {"covered": bool(proxied[0]), "reason": proxied[1]},
        "sound": bool(real[0]) == bool(proxied[0]) and real[1] == proxied[1],
    }


def bisect_predicate_cell_bound(family: str, sub_step: str, fields, layer,
                                grid) -> Dict[str, Any]:
    """The first cell count at which the predicate flips, MEASURED not quoted.

    A ``(1, 1, n)`` extent is used so the search moves one number: the shape's
    product is ``n``, so a bisection on ``n`` is a bisection on the cell count and
    nothing else moves with it. The transcribed clause says the flip is at exactly
    2**31; this reports where it actually is, plus the two neighbouring cases
    spelled out -- one just inside, one just outside -- because a threshold is a
    claim about a pair and a bisection alone does not show the pair.
    """
    predicate = FAMILIES[family]["predicate"]

    def admits(n: int) -> Tuple[bool, Any]:
        pf, pp, pg = _proxied(fields, layer, grid, (1, 1, n))
        covered, reason = predicate(pf, pp, pg, sub_step)
        return bool(covered), reason

    low_ok, low_reason = admits(1)
    high_ok, high_reason = admits(2 ** 32)
    if not low_ok or high_ok:
        return {"bisected": False,
                "why": ("the bracket does not contain a flip: (1,1,1) covered="
                        f"{low_ok} ({low_reason}); (1,1,2**32) covered={high_ok} "
                        f"({high_reason})")}
    low, high = 1, 2 ** 32
    while high - low > 1:
        middle = (low + high) // 2
        if admits(middle)[0]:
            low = middle
        else:
            high = middle
    inside_ok, inside_reason = admits(low)
    outside_ok, outside_reason = admits(high)
    return {
        "bisected": True,
        "largest_admitted_cells": int(low),
        "smallest_refused_cells": int(high),
        "measured_bound": int(high),
        "transcribed_bound": TRANSCRIBED_INT32_CELL_BOUND,
        "agrees_with_transcription": int(high) == TRANSCRIBED_INT32_CELL_BOUND,
        "just_inside": {"cells": int(low), "covered": inside_ok,
                        "reason": inside_reason},
        "just_outside": {"cells": int(high), "covered": outside_ok,
                         "reason": outside_reason},
        "refusal_names_int32": bool(isinstance(outside_reason, str)
                                    and "int32 index range" in outside_reason),
    }


def complex_word_bound_leg(courant: float) -> Dict[str, Any]:
    """The HALVED bound, on the clause both complex predicates share.

    ``coverage._complex_grid_refusal`` (:2872-2880) is called DIRECTLY rather than
    through ``covers_real_pml_complex_curl``, and the record says so: the public
    predicates gate on an expansion license loaded from a probe record, which is a
    different question from the one this leg asks. What is measured here is the
    clause itself -- whether a complex fixture's admission flips at ``2*cells ==
    2**31`` and not at ``cells == 2**31`` -- which is the halving, and it is the
    only place in this file where a private function is the subject.
    """
    xp = _NumpyWearingCupysName()
    rng = np.random.default_rng(SEED)
    fields, layer, grid = gate_cuda_complex_fixture(xp, courant, rng)

    def refusal(n: int) -> Any:
        pf, pp, pg = _proxied(fields, layer, grid, (1, 1, n))
        return coverage._complex_grid_refusal(pf, pp, pg)  # noqa: SLF001

    baseline = refusal(int(np.prod(grid.shape)))
    if baseline is not None:
        return {"measured": False,
                "why": f"the complex fixture is itself refused: {baseline}"}
    low, high = 1, 2 ** 32
    if refusal(low) is not None or refusal(high) is None:
        return {"measured": False, "why": "the bracket does not contain a flip"}
    while high - low > 1:
        middle = (low + high) // 2
        if refusal(middle) is None:
            low = middle
        else:
            high = middle
    return {
        "measured": True,
        "clause": "coverage._complex_grid_refusal (called directly)",
        "largest_admitted_cells": int(low),
        "smallest_refused_cells": int(high),
        "measured_word_bound": int(2 * high),
        "transcribed_word_bound": TRANSCRIBED_INT32_WORD_BOUND,
        "is_halved_against_the_real_path": int(high) * 2 == TRANSCRIBED_INT32_WORD_BOUND,
        "just_outside_reason": refusal(high),
    }


def gate_cuda_complex_fixture(xp, courant: float, rng):
    """A small complex-storage fixture, built the way the complex gate builds one.

    ``force_complex_fields=True`` always -- that is the complex-storage-at-k=0
    class and it is what makes the word-offset bound the live one.
    """
    grid = Grid(resolution=1.0, cell_size=(12.0, 10.0, 8.0),
                boundaries=("periodic", "periodic", "periodic"),
                xp=xp, courant=courant, k_point=(0.0, 0.0, 0.0))
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    return fields, layer, grid


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def checkpoint_emitter(path: str) -> Callable[[Dict[str, Any]], None]:
    """One flushed JSON line per checkpoint, appended as it lands.

    The progress-reporting rule, and the reason it matters more than usual here: a stress run
    is long by definition and one whose only signal is an exit code cannot be told
    from a hang. An interrupted run keeps every checkpoint up to the failure and
    the file is readable with ``tail`` on whichever machine owns the job.
    """
    def emit(record: Dict[str, Any]) -> None:
        record["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
    return emit


def build_plan(rows: List[Dict[str, Any]], product: str, top_n: int,
               threads_by_family: Dict[str, int]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for family, spec in FAMILIES.items():
        sub_steps = (spec["sub_steps"] if product == "full"
                     else (REDUCED_SUB_STEP[family],))
        classes = VALUE_CLASSES if product == "full" else ("uniform",)
        for sub_step in sub_steps:
            subjects = select_subjects(rows, family, sub_step, top_n)
            for subject in subjects:
                subject["threads"] = threads_by_family[family]
                for value_class in classes:
                    plan.append({"family": family, "sub_step": sub_step,
                                 "subject": subject, "courant": INEXACT_COURANT,
                                 "value_class": value_class})
            # THE EXACT COURANT, on the largest shape only. Multiplying every
            # 11-million-cell case by a second courant buys a repetition of an
            # answer, not a second question -- but a sweep that never ran the
            # exactly-representable one could not say the shape result is
            # independent of it.
            if subjects and product == "full":
                plan.append({"family": family, "sub_step": sub_step,
                             "subject": subjects[0], "courant": EXACT_COURANT,
                             "value_class": "uniform"})
    return plan


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              plan: List[Dict[str, Any]], launches: int, checkpoint_every: int,
              memory_budget: int, words: Dict[Tuple[str, str], Dict[str, Any]],
              emit) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, entry in enumerate(plan, start=1):
        case = one_case(backend, entry["family"], entry["sub_step"],
                        entry["subject"], entry["courant"], entry["value_class"],
                        launches, checkpoint_every, memory_budget,
                        words[(entry["family"], entry["sub_step"])], emit=emit)
        cases.append(case)
        results["sweep"] = cases
        save(results, out_path)
        shape = "x".join(str(n) for n in case["target_shape"])
        if case.get("skipped"):
            log(f"[sweep] {index}/{len(plan)} {entry['family']}/{entry['sub_step']} "
                f"{case['row']} {shape} SKIPPED: {case['skipped'][:80]} "
                f"({case['seconds']:.1f} s)")
            continue
        log(f"[sweep] {index}/{len(plan)} {entry['family']}/{entry['sub_step']} "
            f"{case['row']} {shape} cells={case['cells']:,} "
            f"blocks={case['launch_geometry']['blocks']:,} "
            f"{'IDENTICAL' if case['bit_identical'] else 'DIVERGED@' + str(case['first_divergent_launch'])} "
            f"moved={case['non_vacuity']['min_moved_since_frozen']:.3f} "
            f"mem={case['memory']['predicted_bytes'] / 1e9:.2f}GB "
            f"({case['seconds']:.1f} s)")
    return cases


def run_defects(results: Dict[str, Any], out_path: str, backend: str,
                plan: List[Dict[str, Any]], launches: int, checkpoint_every: int,
                memory_budget: int, words: Dict[Tuple[str, str], Dict[str, Any]],
                emit) -> Dict[str, Any]:
    """Every planted defect at every selected shape, scored against its ARITHMETIC.

    THE EXPECTATION IS COMPUTED, NOT ASSERTED BY HAND. Each defect has a threshold
    that is a function of the shape, so "must be caught here and must be null
    there" falls out of the shape rather than out of a list somebody maintained.
    A defect whose measured verdict disagrees with its threshold is a HARNESS
    failure and is reported as one -- which is the only way a leg that has never
    been seen to fail becomes a leg that measures something.
    """
    # One case per (family, sub_step, shape), at the inexact courant and the
    # uniform class: a defect leg answers "can this see this at this size", and
    # multiplying it by the value classes buys repetitions of that answer.
    seen: set = set()
    legs: List[Dict[str, Any]] = []
    for entry in plan:
        key = (entry["family"], entry["sub_step"],
               tuple(entry["subject"]["spec"]["target_shape"]))
        if key in seen or entry["value_class"] != "uniform":
            continue
        if entry["courant"] != INEXACT_COURANT:
            continue
        seen.add(key)
        legs.append(entry)

    out: Dict[str, Any] = {}
    for defect in SCALE_DEFECTS:
        scored: List[Dict[str, Any]] = []
        for entry in legs:
            case = one_case(backend, entry["family"], entry["sub_step"],
                            entry["subject"], entry["courant"], "uniform",
                            launches, checkpoint_every, memory_budget,
                            words[(entry["family"], entry["sub_step"])],
                            defect=defect, emit=emit)
            if case.get("skipped"):
                scored.append(case)
                continue
            cells = case["cells"]
            threshold = case["defect_threshold"]
            case["expected_to_bite"] = bool(threshold is not None
                                            and threshold < cells)
            # HOW MUCH OF THE VOLUME THE DEFECT LEAVES ALONE, and it matters.
            # A "total" cover -- threshold 0, nothing written at all -- is caught
            # by any comparator that looks at one word, so a defect leg made
            # entirely of total covers would not show that the comparison reaches
            # the whole volume. A "partial" cover leaves the head of the grid
            # correct and corrupts only the tail, which is what a narrow guard
            # actually does whenever the truncation lands positive: measured,
            # 490,000 cells truncates to 31,248, so gaussian-beam.py's case
            # corrupts 93.6%% of the volume and leaves the first 6.4%% exact.
            case["defect_cover"] = (
                "none" if not case["expected_to_bite"]
                else "total" if threshold == 0 else "partial")
            case["defect_fraction_reverted"] = (
                0.0 if not case["expected_to_bite"]
                else float(cells - threshold) / float(cells))
            case["caught"] = not case["bit_identical"]
            case["agrees_with_arithmetic"] = (case["caught"]
                                              == case["expected_to_bite"])
            scored.append(case)
            log(f"[defect] {defect} {entry['family']}/{entry['sub_step']} "
                f"{case['row']} cells={cells:,} threshold={threshold:,} "
                f"cover={case['defect_cover']} "
                f"expected={'BITE' if case['expected_to_bite'] else 'NULL'} "
                f"measured={'CAUGHT' if case['caught'] else 'NULL'} "
                f"{'OK' if case['agrees_with_arithmetic'] else 'HARNESS FAILURE'}")
        live = [c for c in scored if not c.get("skipped")]
        bite = [c for c in live if c["expected_to_bite"]]
        null = [c for c in live if not c["expected_to_bite"]]
        out[defect] = {
            "ran": len(live),
            "expected_to_bite": len(bite),
            "caught_where_expected": sum(1 for c in bite if c["caught"]),
            "expected_null": len(null),
            "null_confirmed": sum(1 for c in null if not c["caught"]),
            "disagreements": [_case_key(c) for c in live
                              if not c["agrees_with_arithmetic"]],
            "covers": sorted({c["defect_cover"] for c in live}),
            "partial_cover_cases": [
                {"case": _case_key(c), "cells": c["cells"],
                 "threshold": c["defect_threshold"],
                 "fraction_reverted": c["defect_fraction_reverted"]}
                for c in live if c["defect_cover"] == "partial"],
            "verdict": ("AGREES WITH ARITHMETIC"
                        if live and all(c["agrees_with_arithmetic"] for c in live)
                        else "NO LEGS" if not live else "HARNESS FAILURE"),
            "cases": scored,
        }
        results["scale_defects"] = out
        save(results, out_path)
    return out


def run_size_independent_control(results: Dict[str, Any], out_path: str,
                                 backend: str, plan: List[Dict[str, Any]],
                                 launches: int, checkpoint_every: int,
                                 memory_budget: int,
                                 words: Dict[Tuple[str, str], Dict[str, Any]],
                                 emit) -> Dict[str, Any]:
    """The sibling gates' own sub-lattice mutation, at every shape. MUST BE CAUGHT.

    WITHOUT THIS THE DEFECT LEG PROVES LESS THAN IT LOOKS. A shape-dependent
    defect that is caught at 11 million cells and null at 1,287 is consistent with
    a comparator that only wakes up when the array is large. A defect that is
    size-INDEPENDENT and is caught at both ends separates the two readings, and it
    is the gates' own mutation rather than a new one so it cannot be a mutation
    tuned to this harness.
    """
    seen: set = set()
    out: Dict[str, Any] = {}
    legs: List[Dict[str, Any]] = []
    for entry in plan:
        key = (entry["family"], entry["sub_step"],
               tuple(entry["subject"]["spec"]["target_shape"]))
        if key in seen or entry["value_class"] != "uniform":
            continue
        if entry["courant"] != INEXACT_COURANT:
            continue
        seen.add(key)
        legs.append(entry)

    scored: List[Dict[str, Any]] = []
    for entry in legs:
        mutation = FAMILIES[entry["family"]]["sub_lattice_mutation"]
        case = one_case(backend, entry["family"], entry["sub_step"],
                        entry["subject"], entry["courant"], "uniform",
                        launches, checkpoint_every, memory_budget,
                        words[(entry["family"], entry["sub_step"])],
                        host_mutation=mutation, emit=emit)
        if not case.get("skipped"):
            case["caught"] = not case["bit_identical"]
            log(f"[control] {mutation} {entry['family']}/{entry['sub_step']} "
                f"{case['row']} cells={case['cells']:,} "
                f"{'CAUGHT' if case['caught'] else 'UNCAUGHT'}")
        scored.append(case)
    live = [c for c in scored if not c.get("skipped")]
    caught = sum(1 for c in live if c["caught"])
    out = {
        "mutation": "the gates' own sub-lattice swap, per family",
        "ran": len(live), "caught": caught,
        "verdict": ("NO LEGS" if not live else "CAUGHT" if caught == len(live)
                    else "PARTIAL" if caught else "UNCAUGHT"),
        # The smallest and largest shape it was caught at, because a control whose
        # legs all landed at one size is not a size-independent control.
        "caught_at_cells": sorted({c["cells"] for c in live if c["caught"]}),
        "cases": scored,
    }
    results["size_independent_control"] = out
    save(results, out_path)
    return out


def _bound_probe_fixture(family: str, sub_step: str, cell: Tuple[float, float, float],
                         courant: float, boundaries=("periodic",) * 3, axes: str = ""):
    """A small REAL fixture, seeded, for the substitution legs to stand on."""
    gate = FAMILIES[family]["gate"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    spec = {"label": "bound_probe", "axes": axes, "phase": 1,
            "boundaries": tuple(boundaries), "cell": tuple(cell)}
    xp = _NumpyWearingCupysName()
    rng = np.random.default_rng(SEED)
    fields, layer, grid = gate.build(xp, spec, courant)
    if FAMILIES[family]["needs_epsilon"](sub_step):
        gate.install_epsilon(fields, grid, rng)
    gate.seed_state(fields, grid, gate_arg, "uniform", rng)
    return fields, layer, grid


#: The cross-substitution pairs. Each is ``(cell_A, cell_B)``: the proxy stands on
#: a REAL fixture built at A, wears B's extent, and its verdict must equal a REAL
#: fixture built at B. IDENTITY SUBSTITUTION ALONE IS NOT ENOUGH -- a proxy that
#: ignored the substitution entirely would pass it 100%.
CROSS_SUBSTITUTION_PAIRS: Tuple[Tuple[Tuple[float, float, float],
                                      Tuple[float, float, float]], ...] = (
    ((12.0, 10.0, 8.0), (13.0, 11.0, 9.0)),   # the gate fixture's own extent
    ((12.0, 10.0, 8.0), (32.0, 32.0, 32.0)),  # cubic: every axis the same length
    ((12.0, 10.0, 8.0), (64.0, 48.0, 1.0)),   # a one-thick axis
    ((13.0, 11.0, 9.0), (12.0, 10.0, 8.0)),   # and the substitution run backwards
    # STANDING ON a one-thick axis, not merely wearing one. This pair is the one
    # that catches a proxy which infers a coefficient vector's axis from its
    # broadcast shape: on a grid with nz = 1 the z vector is (1,1,1) and carries
    # no axis at all. It was added after exactly that defect was measured.
    ((64.0, 48.0, 1.0), (12.0, 10.0, 8.0)),
)


def run_bound_legs(results: Dict[str, Any], out_path: str,
                   sweep: List[Dict[str, Any]], courant: float,
                   memory_budget: int,
                   words: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """The int32 clause, bisected -- after the stand-in is shown to be transparent.

    THE ORDER IS THE ARGUMENT. A bisection on a proxy is a claim about the proxy
    unless the proxy is first shown to answer as the real objects do, so two
    soundness arms run before any bound is reported:

    * IDENTITY, at every shape the sweep actually scored. The proxy wears the
      shape it already has, and its verdict must equal the real objects'. This is
      what shows the wrappers -- the substituted ``size``, the wrapped
      ``inverse_epsilon_for``, the derived ``owned_cells`` -- are transparent
      across folds, metallic terminations and one-thick axes, i.e. across the
      shape variety the corpus has and the bound leg extrapolates from.
    * CROSS, on four ``A -> B`` pairs. The proxy stands on a real fixture built at
      A, wears B's extent, and must agree with a REAL fixture built at B. Identity
      alone would be passed by a proxy that ignored the substitution entirely;
      this arm is the one that says the substitution is CORRECT and not merely
      inert, and it is what licenses reading a verdict at an extent no machine can
      allocate.
    """
    out: Dict[str, Any] = {"identity_soundness": [], "cross_soundness": [],
                           "bisections": {}}

    # ---- arm 1: identity, at every scored shape ---------------------------
    # DEDUPED BY (family, sub_step, shape): the sweep runs each shape under two
    # value classes and the proxy's transparency is a property of the SHAPE, so a
    # second draw of the same grid would rebuild a gigabyte to re-ask a question
    # that has already been answered.
    seen_shapes: set = set()
    for case in sweep:
        if case.get("skipped"):
            continue
        family, sub_step = case["family"], case["sub_step"]
        shape_key = (family, sub_step, tuple(case["target_shape"]))
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        predicted = predicted_bytes(case["cells"],
                                    words[(family, sub_step)]["total_words_per_cell"])
        if predicted > memory_budget:
            out["identity_soundness"].append(
                {"family": family, "sub_step": sub_step, "row": case["row"],
                 "shape": case["target_shape"], "sound": None,
                 "why": "not rebuilt: predicted footprint exceeds the budget"})
            continue
        gate = FAMILIES[family]["gate"]
        gate_arg = FAMILIES[family]["to_gate"](sub_step)
        xp = _NumpyWearingCupysName()
        rng = case_rng(family, sub_step, case["row"], case["courant"],
                       case["value_class"])
        rebuilt_spec = {"label": case["row"], "axes": case["fold_axes"],
                        "phase": case["mirror_phase_is_a_fixture_choice"],
                        "boundaries": tuple(case["boundaries"]),
                        "cell": tuple(float(c) for c in _cell_for(case)),
                        "target_shape": tuple(case["target_shape"])}
        try:
            fields, layer, grid = gate.build(xp, rebuilt_spec, case["courant"])
        except Exception as exc:  # noqa: BLE001
            out["identity_soundness"].append(
                {"family": family, "sub_step": sub_step, "row": case["row"],
                 "shape": case["target_shape"], "sound": None,
                 "why": f"not rebuilt: {type(exc).__name__}: {exc}"})
            continue
        if FAMILIES[family]["needs_epsilon"](sub_step):
            gate.install_epsilon(fields, grid, rng)
        gate.seed_state(fields, grid, gate_arg, case["value_class"], rng)
        entry = standin_soundness(family, sub_step, fields, layer, grid,
                                  tuple(grid.shape))
        entry.update({"family": family, "sub_step": sub_step, "row": case["row"],
                      "cells": case["cells"]})
        out["identity_soundness"].append(entry)
        del fields, layer, grid
    results["int32_bound"] = out
    save(results, out_path)
    unsound = [e for e in out["identity_soundness"] if e.get("sound") is False]
    log(f"[bound] identity soundness: "
        f"{sum(1 for e in out['identity_soundness'] if e.get('sound'))}"
        f"/{len(out['identity_soundness'])} sound, {len(unsound)} disagreements")

    # ---- arm 2: cross substitution ----------------------------------------
    for family, spec in FAMILIES.items():
        predicate = spec["predicate"]
        for sub_step in spec["sub_steps"]:
            for cell_a, cell_b in CROSS_SUBSTITUTION_PAIRS:
                fa, la, ga = _bound_probe_fixture(family, sub_step, cell_a, courant)
                fb, lb, gb = _bound_probe_fixture(family, sub_step, cell_b, courant)
                real_b = predicate(fb, lb, gb, sub_step)
                pf, pp, pg = _proxied(fa, la, ga, tuple(gb.shape))
                proxied = predicate(pf, pp, pg, sub_step)
                out["cross_soundness"].append({
                    "family": family, "sub_step": sub_step,
                    "stood_on": [int(n) for n in ga.shape],
                    "wore": [int(n) for n in gb.shape],
                    "real_at_worn_shape": {"covered": bool(real_b[0]),
                                           "reason": real_b[1]},
                    "proxied": {"covered": bool(proxied[0]), "reason": proxied[1]},
                    "sound": (bool(real_b[0]) == bool(proxied[0])
                              and real_b[1] == proxied[1]),
                })
    cross_bad = [e for e in out["cross_soundness"] if not e["sound"]]
    log(f"[bound] cross soundness: "
        f"{len(out['cross_soundness']) - len(cross_bad)}/{len(out['cross_soundness'])} "
        f"sound")
    results["int32_bound"] = out
    save(results, out_path)

    # ---- the bisection, only reported if both arms held --------------------
    out["soundness_holds"] = not unsound and not cross_bad
    for family, spec in FAMILIES.items():
        for sub_step in spec["sub_steps"]:
            fields, layer, grid = _bound_probe_fixture(
                family, sub_step, (12.0, 10.0, 8.0), courant)
            out["bisections"][f"{family}/{sub_step}"] = \
                bisect_predicate_cell_bound(family, sub_step, fields, layer, grid)
            leg = out["bisections"][f"{family}/{sub_step}"]
            if leg.get("bisected"):
                log(f"[bound] {family}/{sub_step}: "
                    f"admitted<={leg['largest_admitted_cells']:,} "
                    f"refused>={leg['smallest_refused_cells']:,} "
                    f"agrees_with_transcription={leg['agrees_with_transcription']}")
            else:
                log(f"[bound] {family}/{sub_step}: NOT BISECTED: {leg.get('why')}")
            results["int32_bound"] = out
            save(results, out_path)

    out["complex_halved_bound"] = complex_word_bound_leg(courant)
    log(f"[bound] complex halved word bound: "
        f"{out['complex_halved_bound'].get('measured_word_bound')} "
        f"(halved={out['complex_halved_bound'].get('is_halved_against_the_real_path')})")
    results["int32_bound"] = out
    save(results, out_path)
    return out


def _cell_for(case: Dict[str, Any]) -> Tuple[float, float, float]:
    """The full cell extent that reproduces a scored case's stored shape."""
    shape = tuple(int(n) for n in case["target_shape"])
    mirrored = [bool(v) for v in case["structure"]["mirrored"]]
    cell = _solve_cell(shape, mirrored, tuple(case["boundaries"]),
                       case["fold_axes"], case["mirror_phase_is_a_fixture_choice"])
    if cell is None:
        raise ValueError(f"no full extent reproduces {shape}")
    return cell


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    sweep = results.get("sweep", [])
    scored = [c for c in sweep if not c.get("skipped")]
    identical = [c for c in scored if c["bit_identical"]]

    if not scored:
        reasons.append("no case was scored at all")
    if len(identical) != len(scored):
        diverged = [(_case_key(c), c["first_divergent_launch"])
                    for c in scored if not c["bit_identical"]]
        reasons.append(f"divergence on {len(scored) - len(identical)} of "
                       f"{len(scored)} cases: {diverged[:4]}")

    # THE ASPECT CORNERS MUST HAVE BEEN SCORED, not merely planned. A scale claim
    # that rests only on the largest grids is a claim about cell count; the
    # decomposition question is about the SHAPE, and a run whose corner cases all
    # skipped would carry no evidence on it.
    provenances = {c.get("provenance") for c in scored}
    for corner in ("corpus_widest_axis", "corpus_extreme_aspect"):
        if corner not in provenances:
            reasons.append(f"no {corner} case was scored; the index-decomposition "
                           f"question is about shape, not cell count")

    cells_scored = sorted({c["cells"] for c in scored})
    largest = max(cells_scored) if cells_scored else 0
    corpus_max = results.get("corpus_census", {}).get("max_cells", 0)
    if largest < corpus_max:
        reasons.append(f"the largest scored case is {largest:,} cells but the "
                       f"corpus reaches {corpus_max:,}; a scale claim must reach "
                       f"the size it names")

    for case in scored:
        if not case["launch_geometry"]["covers_every_cell"]:
            reasons.append(f"{_case_key(case)}: the launch grid does not cover "
                           f"every cell")
        if not case["launch_geometry"]["top_lane_index_within_int32"]:
            reasons.append(f"{_case_key(case)}: the top lane's flat index leaves "
                           f"int32")
        if not case["index_expressions"]["all_within_int32"]:
            reasons.append(f"{_case_key(case)}: "
                           f"{case['index_expressions']['exceeding_int32']}")
        if not case["decomposition"]["exact"]:
            reasons.append(f"{_case_key(case)}: the index decomposition is not "
                           f"exact in int32")

    for defect, leg in results.get("scale_defects", {}).items():
        if leg["verdict"] != "AGREES WITH ARITHMETIC":
            reasons.append(f"scale defect {defect} is {leg['verdict']} "
                           f"({leg['disagreements'][:3]})")

    control = results.get("size_independent_control", {})
    if control and control.get("verdict") != "CAUGHT":
        reasons.append(f"the size-independent control is {control.get('verdict')} "
                       f"({control.get('caught')}/{control.get('ran')})")

    bound = results.get("int32_bound", {})
    unsound = [e for e in bound.get("identity_soundness", []) if e.get("sound") is False]
    cross_bad = [e for e in bound.get("cross_soundness", []) if not e.get("sound")]
    if unsound:
        reasons.append(f"the shape stand-in disagreed with the real objects at "
                       f"{len(unsound)} scored shape(s); the bound leg is UNSOUND")
    if cross_bad:
        reasons.append(f"the shape stand-in disagreed with a REAL fixture at the "
                       f"substituted extent on {len(cross_bad)} pair(s); the "
                       f"substitution is not transparent and the bound leg is "
                       f"UNSOUND")
    if bound and not bound.get("cross_soundness"):
        reasons.append("no cross-substitution leg ran; identity substitution "
                       "alone would be passed by a proxy that ignored the "
                       "substitution entirely")
    for key, leg in bound.get("bisections", {}).items():
        if not leg.get("bisected"):
            reasons.append(f"the int32 bound could not be bisected for {key}: "
                           f"{leg.get('why')}")
        elif not leg.get("agrees_with_transcription"):
            reasons.append(f"the measured int32 bound for {key} is "
                           f"{leg.get('measured_bound')}, not the transcribed "
                           f"{TRANSCRIBED_INT32_CELL_BOUND}")

    # THE DEFECT LEG'S OWN NON-VACUITY. A run in which no planted defect was ever
    # expected to bite has not shown the harness can fail; it has shown the
    # harness was never asked to.
    bit_expected = sum(leg.get("expected_to_bite", 0)
                       for leg in results.get("scale_defects", {}).values())
    if results.get("scale_defects") and bit_expected == 0:
        reasons.append("no planted defect was expected to bite at any scored "
                       "shape; this run did not demonstrate that the leg can fail")
    # AND AT LEAST ONE OF THEM MUST HAVE BEEN A PARTIAL COVER. A battery whose
    # every biting leg wrote nothing at all is caught by a comparator that looks
    # at one word; a partial cover is the only arm that shows the comparison
    # reaches the tail of an eleven-million-cell volume.
    partial = sum(len(leg.get("partial_cover_cases", []))
                  for leg in results.get("scale_defects", {}).values())
    if results.get("scale_defects") and bit_expected and not partial:
        reasons.append("every planted defect that bit was a TOTAL cover; no leg "
                       "showed the comparison reaching the tail of a volume whose "
                       "head was left exact")

    # NOT `released`. A STRESS HARNESS IS NOT A GATE, and the distinction is not
    # cosmetic: gate_provenance.stamp() copies a `released` field into
    # canonical_verdict, which is what read_verdict() — and therefore every weld
    # test in this package — treats as "this run licensed something". MEASURED
    # 2026-08-20: on `--backend numpy`, with no device in the process at all, this
    # function returned released=true and the stamped artifact carried
    # canonical_verdict {released: true}. A laptop transcription check was
    # indistinguishable, to the shared reader, from a device gate that had passed.
    #
    # The sibling harnesses spell the outcome `findings` and say "not a gate" in as
    # many words. This one now agrees with them. What a stress run produces is a
    # list of things worth looking at; what licenses a kernel is a gate under
    # parity/meep_gpu/gate_cuda_*.py, and only that.
    return {
        "findings": reasons,
        "reasons": reasons,
        "scored_cases": len(scored),
        "bit_identical_cases": len(identical),
        "cells_scored": cells_scored,
        "shapes_scored": sorted({tuple(c["target_shape"]) for c in scored}),
        "provenances_scored": sorted(p for p in provenances if p),
        "largest_scored_cells": largest,
        "corpus_max_cells": corpus_max,
        "reaches_corpus_max": bool(largest >= corpus_max) if corpus_max else False,
        "defects_expected_to_bite": bit_expected,
        "defect_partial_cover_legs": partial,
        "claim": (
            "the certified hand-CUDA curl and constitutive kernels reproduce "
            "stepping's array path bit-for-bit at the corpus's real shapes, up to "
            f"{largest:,} cells, at every checkpoint of a "
            f"{sweep[0]['launches'] if sweep else 0}-launch run, with the launch "
            "geometry, the int32 index expressions and the index decomposition "
            "checked exhaustively at each shape"),
        "does_not_claim": [
            "any device result: the NumPy arm compiles nothing and launches "
            "nothing, and no verdict here is about NVRTC, the subnormal policy or "
            "a real GPU",
            "a long horizon: this leg's launch budget is short by design and the "
            "step-67 class of divergence is dimension A's, not this one's",
            "the other twenty families: only covers_real_pml_curl and "
            "covers_real_pml_constitutive have a transcribed NumPy arm to borrow, "
            "and cuda_offdiag reaches the same 11,245,000-cell row with none",
            "bit identity AT the int32 bound: 2**31 cells cannot be allocated, so "
            "the bound is exercised as a PREDICATE flip on a shape-substituted "
            "stand-in and not as a launch",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--corpus-dir", default=None,
                        help="the predicate-coverage record to draw shapes from")
    parser.add_argument("--shapes-per-family", type=int, default=3,
                        help="how many of the largest admitted shapes per slot")
    parser.add_argument("--launches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--memory-budget-gb", type=float, default=6.0,
                        help=("predicted footprint above which a case is SKIPPED "
                              "WITH A REASON rather than allowed to OOM"))
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--skip-defects", action="store_true")
    parser.add_argument("--skip-bound", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    checkpoint_path = args.out + ".checkpoints.jsonl"
    emit = checkpoint_emitter(checkpoint_path)

    results: Dict[str, Any] = {
        "leg": "stress_cuda_scale",
        "dimension": "B / SCALE",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("do the certified hand-CUDA kernels still reproduce the "
                     "array path at the shapes the corpus actually has, and does "
                     "the int32 clause that refuses the rest sit where it says?"),
        "checkpoint_stream": checkpoint_path,
    }

    directory = corpus_dir(args.corpus_dir)
    rows = read_corpus(directory)
    results["corpus_record"] = os.path.relpath(directory, _REPO_API)
    results["corpus_census"] = corpus_shape_census(rows)

    # The shipped launch facts, regexed off the shipped bytes. A module whose
    # geometry no longer matches makes this leg REFUSE rather than check a
    # geometry the module does not have.
    launch_facts = {family: shipped_launch_facts(spec["kernel_module"],
                                                 spec["threads_symbol"])
                    for family, spec in FAMILIES.items()}
    results["shipped_launch_facts"] = launch_facts
    unusable = [f for f, facts in launch_facts.items() if not facts["usable"]]
    if unusable:
        results["status"] = (f"refused: the shipped launch geometry no longer "
                             f"matches what this leg checks, for {unusable}")
        save(results, args.out)
        log(f"[fatal] {results['status']}")
        return 2
    threads_by_family = {f: facts["threads_literals"][0]
                         for f, facts in launch_facts.items()}

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        observer = probe.install_nvrtc_binary_observer()
        results["nvrtc_observer"] = observer
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {
            "python": sys.version.split()[0],
            "numpy_version": np.__version__,
            "note": "NumPy backend: compiles nothing, launches nothing, certifies nothing",
        }
    save(results, args.out)

    words = {}
    for family, spec in FAMILIES.items():
        for sub_step in spec["sub_steps"]:
            words[(family, sub_step)] = measure_words_per_cell(family, sub_step)
    results["words_per_cell"] = {f"{k[0]}/{k[1]}": v for k, v in words.items()}

    plan = build_plan(rows, args.product, args.shapes_per_family, threads_by_family)
    results["plan"] = [
        {"family": e["family"], "sub_step": e["sub_step"], "row": e["subject"]["row"],
         "shape": [int(n) for n in e["subject"]["spec"]["target_shape"]],
         "cells": e["subject"]["cells"], "courant": e["courant"],
         "value_class": e["value_class"], "provenance": e["subject"]["provenance"]}
        for e in plan]
    save(results, args.out)
    log(f"[plan] {len(plan)} cases across {len(FAMILIES)} families; "
        f"shapes {sorted({e['subject']['cells'] for e in plan})}")

    budget = int(args.memory_budget_gb * 1e9)
    sweep = run_sweep(results, args.out, args.backend, plan, args.launches,
                      args.checkpoint_every, budget, words, emit)

    if not args.skip_defects:
        run_defects(results, args.out, args.backend, plan, args.launches,
                    args.checkpoint_every, budget, words, emit)
        run_size_independent_control(results, args.out, args.backend, plan,
                                     args.launches, args.checkpoint_every,
                                     budget, words, emit)
    if not args.skip_bound:
        run_bound_legs(results, args.out, sweep, INEXACT_COURANT, budget, words)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[findings] {len(verdict['findings'])} "
        f"scored={verdict['scored_cases']} identical={verdict['bit_identical_cases']} "
        f"largest={verdict['largest_scored_cells']:,} cells")
    for reason in verdict["reasons"]:
        log(f"[finding]   - {reason}")
    # EXIT 0 EVEN WITH FINDINGS, matching the sibling harnesses. A finding is
    # something to read, not a failed gate; a non-zero exit here would make a
    # stress run look like a merge-bar failure to any caller that only reads the
    # status. Non-zero is reserved for the harness being unable to MEASURE — which
    # main() raises for separately.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
