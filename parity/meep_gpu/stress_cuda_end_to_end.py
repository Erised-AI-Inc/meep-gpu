"""END-TO-END STRESS: a REAL corpus script, stepped long, with the CUDA kernels
substituted at every admitted slot, compared as words at checkpoints.

=============================================================================
WHAT THIS MEASURES THAT THE 26 GATES DO NOT
=============================================================================

Every gate under ``parity/meep_gpu/`` builds its own synthetic ``Grid`` /
``Fields`` / ``PML`` triple, freezes it, launches ONE sub-step (or 60 of the same
sub-step) and compares. That establishes the arithmetic. It does not establish
the arithmetic IN THE ORDER AND STATE A RUN ACTUALLY PRODUCES, and the two are
different claims:

* a real run **ramps a source**, so the operands sweep from exactly zero through
  the subnormal band into the normal range and back down over thousands of steps,
  rather than being drawn once from a fixture's distribution;
* a real run carries **material that varies in space** — MEEP's own rasterized,
  subpixel-smoothed permittivity, not ``rng.uniform(1.2, 3.4)``;
* a real run interleaves the sub-steps with **the driver's own boundary work**
  (``fill_symmetry_bc_B``/``_D``, ``zero_metal_B``/``_D``,
  ``fill_folded_far_ghosts_B``/``_D``), with **source injection and withdrawal**,
  and with **monitors that read the fields mid-run**;
* a real run reaches **step counts the fixtures never do**.

So this leg lifts a corpus row twice into two ``FdtdDriver`` objects, points ONE
of them at the kernels, steps both through the driver's own ``step()``, and
compares the FULL field inventory as uint32 words at checkpoints.

=============================================================================
THE SUBSTITUTION SEAM, AND WHY IT IS NOT A PACKAGE EDIT
=============================================================================

``meep_gpu`` must not import ``meep_gpu.cuda_kernels`` — ``test_package_boundary``
fails the build on a module-level import — and ``fastpath.plan_fast_path`` carries
no hand-CUDA branch at all (``coverage.py`` header: "THIS IS NOT WIRED TO
DISPATCH"). There is therefore no dispatch consult a hand-CUDA plan can be handed.

The seam used here is the one the Triton whole-step leg already uses and
documents (``probe_fused_kernel_bit_identity.install_triton_substeps``, :3278):
the DRIVER MODULE's globals. ``driver.py`` imports ``step_B``, ``step_D``,
``update_H``, ``update_E``, ``update_P`` by name (driver.py:191-195) and calls
them through those names (driver.py:3281-3306), so rebinding the module
attributes from outside redirects the step without editing one byte under
``meep_gpu/``. Nothing in the package is modified, and the undo is returned.

**SCOPED TO ONE ``Fields``.** The module globals are shared by every driver in the
process, so an unscoped patch would send the ORACLE's steps through the kernels
too. The probe measured what that looks like: "the fused state was stepped twice
per timestep and every array diverged at step 1, which reads exactly like a broken
kernel and is not one" (:3287-3291). Every wrapper here therefore checks
``fields is owner`` first and delegates otherwise.

=============================================================================
WHICH SLOT GETS WHICH KERNEL, AND WHO DECIDES
=============================================================================

The SHIPPED predicate decides, per sub-step, per row — never this file:

  ``step_B`` / ``step_D``  -> ``coverage.covers_real_pml_curl(f, pml, grid, ss)``
  ``update_H`` / ``update_E`` -> ``coverage.covers_real_pml_constitutive(...)``
  ``update_E`` (off-diagonal) -> ``coverage.covers_real_pml_offdiag_constitutive``
  ``update_P``  -> NO CUDA KERNEL AND NO CUDA PREDICATE. Never substituted, and
                   the record says so rather than leaving a silent hole.

The two E-side predicates are measured DISJOINT over the corpus (coverage report,
"DISJOINTNESS, measured: rows admitted by BOTH E-side predicates: 0"); this
harness asks both and refuses to run a row where both answer yes, because a
harness that picked one would be choosing where the record should measure.

A slot the predicate refuses falls through to the array path, and the row's record
carries the refusal reason. A row where NOTHING is admitted is still run: it is the
inert control that shows the wrapper itself is transparent.

=============================================================================
THE TWO BACKENDS
=============================================================================

``--backend cuda`` calls the SHIPPED launch wrappers
(``step_curl_kernels._step_{B,D}_fused_pml_real``,
``constitutive_kernels.update_fused_pml_real``,
``offdiag_constitutive_kernels`` through the offdiag gate's ``KernelBackend``) and
synchronizes. That leg CERTIFIES.

``--backend numpy`` calls the NumPy transcriptions the family gates already own
and already validated — ``gate_cuda_folded_curl.run_kernel_numpy`` (which is the
device tree's grouping, its three boundary codes and its mask set, transcribed) and
``gate_cuda_folded_constitutive.run_kernel_numpy``. IT COMPILES NOTHING AND
CERTIFIES NOTHING. What it settles is: does the substitution seam work on a real
lifted driver, does the predicate route the slots a real corpus row actually
presents, does a long real run stay identical to the array path through the
transcription, and CAN THIS HARNESS FAIL. All of that on a laptop, before a
contended device slot is spent.

=============================================================================
WHAT MAKES A PASS NON-VACUOUS
=============================================================================

Three floors, taken PER CHECKPOINT rather than once at the end, because a run that
is identical because nothing happened is the failure mode this project has hit
repeatedly (a zero-initialized constitutive leaves every word ``+0.0`` forever, and
a deliberately wrong reference still reports IDENTICAL):

* ``moved_words_since_last_checkpoint`` on the ORACLE's inventory. Zero means the
  interval measured nothing, and the checkpoint is marked ``vacuous`` — it does not
  count towards the row's identical-checkpoint tally.
* ``dispatches_since_last_checkpoint`` per substituted slot. A slot that did not
  fire in the interval cannot be evidence about that slot in that interval.
* ``slots_substituted`` for the row as a whole. Zero means the row is an inert
  control and is reported as one; it is never counted as kernel evidence.

=============================================================================
THE PLANTED DEFECTS — a stress leg nobody has seen fail measures nothing
=============================================================================

``--defect`` mutates the SUBSTITUTED leg only (never the oracle) and every one
must be CAUGHT; ``--defect none`` is the null control and must be NULL CONFIRMED.
See :data:`DEFECTS` for what each is and which real defect it stands for.

=============================================================================
SEEDS
=============================================================================

This leg draws no random numbers: the corpus row IS the fixture, and its state is
whatever MEEP's own geometry rasterization and the script's own sources produce.
:func:`case_seed` exists for the one place a future arm might need one (a perturbed
restart) and is digest-derived rather than ``hash()``-derived, because ``hash()``
of a tuple containing strings is salted by ``PYTHONHASHSEED`` and a case seeded
from it cannot be replayed from its own record.

=============================================================================
RUNNING IT
=============================================================================

Laptop (no CUDA), the whole chosen row set::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/stress_cuda_end_to_end.py --backend numpy \\
        --steps 600 --checkpoint-every 50 --out /tmp/stress_e2e

One planted defect, to show the leg flips::

    ... --backend numpy --defect swap_constitutive_kps_kms --steps 200 \\
        --checkpoint-every 25 --out /tmp/stress_e2e_defect

Device (the GPU host, ONE verified-idle GPU, pinned)::

    CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=. python -u \\
        parity/meep_gpu/stress_cuda_end_to_end.py --backend cuda \\
        --steps 5000 --checkpoint-every 100 --out results/stress_e2e_<stamp>

Progress reporting: the parent prints one flushed line per row; the CHILD appends one JSONL
row per CHECKPOINT as it lands and rewrites its per-row artifact atomically after
every checkpoint, so an interrupted run keeps everything up to the failure and
``tail`` on the machine that owns the job is the whole status check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402

SEED = 20260820

#: MEEP's corpus. The same two roots the 2026-08-09 lift records and the
#: 2026-08-20 predicate census used.
#:
#: RESOLVED, NOT HARDCODED. These were absolute paths under a specific laptop home
#: directory with no flag and no fallback, which made the leg unrunnable on the
#: device host it exists to run on — the very machine whose MEEP checkout lives
#: somewhere else entirely. The device leg is supposed to be a flag flip; a path
#: that only resolves on the machine that cannot run it is not one.
#:
#: Order: --examples-dir / --tests-dir, then MEEP_PYTHON_DIR, then the checkout
#: beside this repo, then the laptop default. Whichever wins is RECORDED in the
#: artifact, because "which corpus did this measure" is part of the measurement.
_MEEP_ROOTS = (
    os.environ.get("MEEP_PYTHON_DIR"),
    os.path.join(os.path.expanduser("~"), "Documents", "meep", "python"),
    os.path.join(os.path.expanduser("~"), "meep", "python"),
)


def _resolve_corpus_root() -> str:
    """The first MEEP python/ directory that actually holds examples/ and tests/."""
    for root in _MEEP_ROOTS:
        if not root:
            continue
        if (os.path.isdir(os.path.join(root, "examples"))
                and os.path.isdir(os.path.join(root, "tests"))):
            return root
    raise RuntimeError(
        "no MEEP python/ directory found with both examples/ and tests/. Tried "
        f"{[r for r in _MEEP_ROOTS if r]}. Pass --examples-dir and --tests-dir, or "
        f"set MEEP_PYTHON_DIR.")


_CORPUS_ROOT = None          # resolved lazily so importing this module never raises
EXAMPLES_DIR = None
TESTS_DIR = None


def resolve_corpus(examples_dir=None, tests_dir=None):
    """Bind the corpus roots, honouring explicit arguments first."""
    global _CORPUS_ROOT, EXAMPLES_DIR, TESTS_DIR
    if examples_dir and tests_dir:
        EXAMPLES_DIR, TESTS_DIR = examples_dir, tests_dir
        _CORPUS_ROOT = os.path.dirname(os.path.abspath(examples_dir))
    else:
        _CORPUS_ROOT = _resolve_corpus_root()
        EXAMPLES_DIR = examples_dir or os.path.join(_CORPUS_ROOT, "examples")
        TESTS_DIR = tests_dir or os.path.join(_CORPUS_ROOT, "tests")
    return EXAMPLES_DIR, TESTS_DIR

#: The subject: every file whose bytes decide what this leg measures. Same scope
#: rule as ``measure_predicate_coverage.subject_digest`` — the WHOLE kernel
#: directory, not the subset imported, so drift in a transitively-reached module
#: still shows.
SUBJECT_GLOB = "meep_gpu/cuda_kernels"
SUBJECT_EXTRA = ("meep_gpu/stepping.py", "meep_gpu/fields.py", "meep_gpu/pml.py",
                 "meep_gpu/grid.py", "meep_gpu/driver.py", "meep_gpu/from_meep.py",
                 "parity/meep_gpu/gate_cuda_folded_curl.py",
                 "parity/meep_gpu/gate_cuda_folded_constitutive.py",
                 "parity/meep_gpu/gate_cuda_offdiag.py",
                 "parity/meep_gpu/stress_cuda_end_to_end.py")


# ---------------------------------------------------------------------------
# THE ROWS
# ---------------------------------------------------------------------------
#
# SEVEN, NOT 186, AND CHOSEN RATHER THAN SLICED. Each carries the reason it is in
# the set; between them they present every boundary code the real-field curl pair
# ships, both constitutive sides, the off-diagonal E-side family, 1-D/2-D/3-D
# storage, a source that ramps and a source that is integrated, and one row where
# the predicate admits nothing at all.
#
# The facts quoted per row (shape, boundary kinds, admitted slots) are READ OFF
# the 2026-08-20 closed predicate census
# (results/cuda_predicate_coverage_2026-08-20_closed/{examples,tests}.jsonl), not
# guessed — and this harness re-asks the predicate at run time anyway and records
# the answer it got, so a row whose census line has gone stale is visible as a
# mismatch rather than silently carried.

ROWS: Tuple[Dict[str, Any], ...] = (
    {
        "label": "straight_waveguide",
        "leg": "examples",
        "script": "straight-waveguide.py",
        "census": {"shape": [160, 80, 1], "kinds": ["metallic", "metallic", "periodic"],
                   "admits": ["step_B", "step_D", "update_H", "update_E"]},
        "why": "THE BASE CERTIFIED FAMILY, unfolded. Both curl boundary codes that "
               "existed before 2026-08-20 (BC_METALLIC on x and y, BC_PERIODIC on the "
               "invariant z) and both constitutive sides, on a 2-D run with a "
               "continuous source and a graded rasterized epsilon.",
    },
    {
        "label": "integrated_source",
        "leg": "tests",
        "module": "test_integrated_source.py",
        "case": "TestIntegratedSource.test_integrated_source",
        "census": {"shape": [120, 120, 1], "kinds": ["periodic", "periodic", "periodic"],
                   "admits": ["step_B", "step_D", "update_H", "update_E"]},
        "why": "THE RUN-ORDER FEATURE NO FIXTURE HAS. An integrated source WITHDRAWS "
               "its standing offset immediately before the curl sub-step and reinjects "
               "the whole fresh dipole after it (driver.py:3277-3280, :3335) — so the "
               "substituted kernel is handed a D/B array that host code mutated between "
               "this step and the last. Also the all-periodic arm: every axis "
               "BC_PERIODIC, no wall mask anywhere.",
    },
    {
        "label": "folded_user_material",
        "leg": "tests",
        "module": "test_user_defined_material.py",
        "case": "TestUserMaterials.test_epsilon_func",
        "census": {"shape": [51, 51, 1], "kinds": ["mirror", "mirror", "periodic"],
                   "admits": ["step_B", "step_D", "update_H", "update_E"]},
        "why": "A FOLD ON TWO AXES AT ONCE, on a real row. MEASURED: both folds here "
               "are METALLIC-terminated and resolve to BC_METALLIC, so this row is the "
               "folded-metallic arm and `folded_periodic_metasurface` is the "
               "folded-periodic one. What it adds over an unfolded row is the "
               "INTERLEAVING: the fold's repairs (fill_symmetry_bc_B/_D, "
               "fill_folded_far_ghosts_B/_D) are driver-level passes that run BETWEEN "
               "the substituted sub-steps, which is exactly what a single-sub-step gate "
               "cannot see.",
    },
    {
        "label": "folded_periodic_metasurface",
        "leg": "examples",
        "script": "metasurface_lens.py",
        "census": {"shape": [390, 10, 1], "kinds": ["periodic", "mirror", "periodic"],
                   "admits": ["step_B", "step_D", "update_H", "update_E"]},
        "why": "THE ONLY ARM THAT DRIVES BC_MIRROR_PERIODIC — the code added to the "
               "shipped kernels on 2026-08-20 and the whole subject of that re-cut. "
               "MEASURED on this host: the shipped resolver returns codes (0, 2, 0) "
               "here, while BOTH mirror rows below resolve their folds to BC_METALLIC. "
               "Without this row the newest boundary arm is never launched by an "
               "end-to-end leg at all, and `mirror_as_metallic` — the defect that IS "
               "the pre-2026-08-20 kernel — would be inert on every row and score a "
               "meaningless 0/0.",
    },
    {
        "label": "three_d_fold",
        "leg": "tests",
        "module": "test_dump_load.py",
        "case": "TestLoadDump.test_load_dump_fields_3d",
        "census": {"shape": [39, 18, 19], "kinds": ["metallic", "mirror", "mirror"],
                   "admits": ["step_B", "step_D", "update_H", "update_E"]},
        "why": "THE ONLY GENUINELY 3-D ARM in the cheap band, and it mixes a metallic "
               "wall with TWO folds. Three live axes is where the kernel's "
               "i = idx/(ny*nz), j = (idx/nz)%ny, k = idx%nz decomposition can be "
               "wrong without being wrong on a 2-D row.",
    },
    {
        "label": "offdiag_full_mask",
        "leg": "tests",
        "module": "test_adjoint_solver.py",
        "case": "TestAdjointSolver.test_offdiagonal",
        "census": {"shape": [150, 150, 1], "kinds": ["metallic", "metallic", "periodic"],
                   "admits": ["step_B", "step_D", "update_H"],
                   "offdiag_update_E": True, "row_mask": [1, 1, 1, 1, 1, 1]},
        "why": "THE OFF-DIAGONAL FAMILY, at the ONLY row of the corpus that drives the "
               "FULL row mask 111111 (the census's other 15 offdiag rows all drive "
               "100100). It is also the row that exercises the E-side DISJOINTNESS in "
               "situ: covers_real_pml_constitutive refuses update_E here and "
               "covers_real_pml_offdiag_constitutive admits it, so the harness must "
               "route one sub-step to a different family's kernel mid-step.",
    },
    {
        "label": "inert_absorber_1d",
        "leg": "tests",
        "module": "test_absorber_1d.py",
        "case": "TestAbsorber.test_absorber",
        "census": {"shape": [1, 1, 400], "kinds": ["periodic", "periodic", "metallic"],
                   "admits": []},
        "why": "THE INERT CONTROL. No active PML layer (an absorber, not a PML) and five "
               "polarizations, so every predicate refuses and the wrappers delegate on "
               "every slot for every step. A divergence here is the HARNESS, not a "
               "kernel — the same role `replace_nothing` plays in the Triton whole-step "
               "leg, where that shape of harness defect was actually found.",
    },
)

ROWS_BY_LABEL = {row["label"]: row for row in ROWS}

#: The sub-steps the driver consults, in ``FdtdDriver.step`` order
#: (driver.py:3281-3306).
SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E", "update_P")

#: The full field inventory, transcribed from ``Fields.reset`` (fields.py:1382-1401)
#: — the engine's own list of "every field array a step may have written". The
#: polarizations' P and P_prev are appended per state at read time, as ``reset``
#: does. Comparing a subset is how a harness certifies a kernel that gets one array
#: right and another wrong: the sibling track measured a dropped auxiliary store at
#: 120/120 UNCAUGHT when the auxiliary was not compared.
INVENTORY_ARRAYS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "Bx", "By", "Bz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
    "f_cond_Bx", "f_cond_By", "f_cond_Bz",
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def case_seed(label: str, *parts: Any) -> int:
    """A replayable per-case seed, derived from a DIGEST rather than ``hash()``.

    ``hash()`` of a tuple containing a string is salted by ``PYTHONHASHSEED``:
    measured, one key gave four distinct seeds in four interpreters, so a case
    seeded that way cannot be replayed from its own record.
    """
    key = "|".join([label, *(str(part) for part in parts)])
    return SEED + int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------
# Progress (the progress-reporting rule)
# ---------------------------------------------------------------------------

_PROGRESS_PATH: Optional[str] = None


def say(message: str) -> None:
    """One flushed line, to stdout and to the progress file if one is configured."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _PROGRESS_PATH:
        with open(_PROGRESS_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic rewrite, provenance-stamped. Never a partial file on the disk."""
    gate_provenance.stamp(payload)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()


# ---------------------------------------------------------------------------
# The backend shim: NumPy behind CuPy's name
# ---------------------------------------------------------------------------

class _CupyNamed:
    """The lifted grid's real array module, reporting ``cupy`` as its name.

    Nothing else is overridden — ``float32`` in particular stays the wrapped
    module's own object, which is what keeps the predicate's dtype clauses meaning
    what they mean. Same shim the 26 gates and the predicate census use, so the
    laptop legs are commensurable.
    """

    __name__ = "cupy"

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _GridWithCupyBackend:
    """The lifted grid with ``xp`` replaced by :class:`_CupyNamed` and nothing else."""

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", _CupyNamed(getattr(grid, "xp", None)))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), name)


#: Every grid read the two curl/constitutive predicates and the boundary resolver
#: take, as read out of the shipped source. The proxy is probed on exactly these,
#: per row, so a proxy that silently answered a grid question differently would be
#: visible instead of quietly moving the admission.
GRID_READS: Tuple[Tuple[str, Optional[Tuple[Any, ...]]], ...] = (
    ("has_symmetry", ()), ("cylindrical", None), ("shape", None),
    ("has_bloch", None), ("bfast_active", None), ("beta", None),
    ("is_mirrored", (0,)), ("is_mirrored", (1,)), ("is_mirrored", (2,)),
    ("is_axis", (0,)), ("is_axis", (1,)), ("is_axis", (2,)),
    ("is_metallic", (0,)), ("is_metallic", (1,)), ("is_metallic", (2,)),
)


def shim_soundness(grid: Any, proxy: Any) -> Dict[str, Any]:
    """Measure that the proxy changed the backend NAME and nothing else."""
    real_xp = getattr(grid, "xp", None)
    record: Dict[str, Any] = {
        "real_xp_name": getattr(real_xp, "__name__", None),
        "proxy_xp_name": getattr(proxy.xp, "__name__", None),
        "float32_is_identical": proxy.xp.float32 is getattr(real_xp, "float32", object()),
        "disagreements": [],
    }
    for name, arguments in GRID_READS:
        def read(target: Any) -> Any:
            attribute = getattr(target, name, "<<absent>>")
            if arguments is None or not callable(attribute):
                return attribute
            return attribute(*arguments)
        try:
            mine, theirs = read(proxy), read(grid)
            agree = bool(mine == theirs)
        except BaseException as exc:  # noqa: BLE001
            mine, theirs, agree = f"RAISED {type(exc).__name__}", None, False
        if not agree:
            record["disagreements"].append(
                {"read": f"{name}{arguments if arguments else ''}",
                 "proxy": repr(mine)[:120], "grid": repr(theirs)[:120]})
    record["sound"] = (record["proxy_xp_name"] == "cupy"
                       and record["float32_is_identical"]
                       and not record["disagreements"])
    return record


def to_host(array: Any) -> np.ndarray:
    """A NumPy copy of a device or host array, without importing CuPy to find out."""
    if array is None:
        return None
    getter = getattr(array, "get", None)
    if callable(getter) and type(array).__module__.startswith("cupy"):
        return getter()
    return np.asarray(array)


# ---------------------------------------------------------------------------
# The inventory and the word comparison
# ---------------------------------------------------------------------------

def inventory(fields: Any) -> Dict[str, np.ndarray]:
    """Host copies of every allocated field array, plus each pole's P and P_prev."""
    state: Dict[str, np.ndarray] = {}
    for name in INVENTORY_ARRAYS:
        array = getattr(fields, name, None)
        if array is not None:
            state[name] = to_host(array).copy()
    for index, pole in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component in pole.driven():
            state[f"P{index}_{component}"] = to_host(pole.P[component]).copy()
            state[f"Pprev{index}_{component}"] = to_host(pole.P_prev[component]).copy()
    return state


def words(array: np.ndarray) -> np.ndarray:
    """The array's raw storage as uint32 words.

    NEVER ``allclose``: ``-0.0 == 0.0`` and ``NaN != NaN`` both lie, and a real run
    puts signed zeros and (under the ``keep`` policy) subnormals into the operands
    on its own. ``complex64`` views as two words per element, which is the storage.
    """
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.float32:
        return contiguous.ravel().view(np.uint32)
    if contiguous.dtype == np.complex64:
        return contiguous.ravel().view(np.uint32)
    if contiguous.dtype == np.float64:
        return contiguous.ravel().view(np.uint64)
    return contiguous.ravel().view(np.uint8)


def compare_inventories(reference: Dict[str, np.ndarray],
                        produced: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per-array differing word counts, and the worst element, as bytes."""
    missing = sorted(set(reference) ^ set(produced))
    differing: Dict[str, Dict[str, Any]] = {}
    total = same = 0
    for name in sorted(set(reference) & set(produced)):
        a, b = words(reference[name]), words(produced[name])
        total += int(a.size)
        if a.shape != b.shape:
            differing[name] = {"shape_mismatch": [list(reference[name].shape),
                                                  list(produced[name].shape)]}
            continue
        mask = a != b
        count = int(np.count_nonzero(mask))
        same += int(a.size) - count
        if count:
            index = int(np.argmax(mask))
            differing[name] = {
                "differing_words": count,
                "total_words": int(a.size),
                "first_index": index,
                "reference_word": int(a[index]),
                "produced_word": int(b[index]),
            }
    return {
        "bit_identical": not differing and not missing,
        "inventory_mismatch": missing,
        "total_words": total,
        "differing_words": sum(entry.get("differing_words", 0)
                               for entry in differing.values()),
        "differing_arrays": sorted(differing),
        "detail": differing,
    }


def restore_inventory(fields: Any, state: Dict[str, np.ndarray], xp: Any) -> None:
    """Write a snapshot back into the live arrays, in place, byte for byte."""
    for name in INVENTORY_ARRAYS:
        array = getattr(fields, name, None)
        if array is not None and name in state:
            array[...] = xp.asarray(state[name])
    for index, pole in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component in pole.driven():
            key = f"P{index}_{component}"
            if key in state:
                pole.P[component][...] = xp.asarray(state[key])
            key = f"Pprev{index}_{component}"
            if key in state:
                pole.P_prev[component][...] = xp.asarray(state[key])


def operand_census(state: Dict[str, np.ndarray]) -> Dict[str, int]:
    """What VALUE CLASSES the real run is actually presenting, right now.

    The gates draw their operands from a declared value class — ``uniform`` or
    ``subnormal_band`` — and sweep both because the float32 subnormal policy only
    has a subject if the numbers reach the band. A REAL run does not get to
    declare: a source that ramps from zero sweeps its own fields up through the
    band and an absorber grades them back down, and nobody has measured whether a
    corpus row's fields ever enter it.

    Counted on the ORACLE's inventory as raw words, so a signed zero is
    distinguishable from a zero (``-0.0 == 0.0`` in every comparison that is not a
    byte comparison).
    """
    total = zeros = negative_zeros = subnormals = nans = infinities = 0
    for name in sorted(state):
        array = np.ascontiguousarray(state[name])
        if array.dtype == np.complex64:
            array = array.view(np.float32)
        elif array.dtype != np.float32:
            continue
        raw = array.ravel().view(np.uint32)
        exponent = raw & np.uint32(0x7F800000)
        mantissa = raw & np.uint32(0x007FFFFF)
        zero = (exponent == 0) & (mantissa == 0)
        total += int(raw.size)
        zeros += int(np.count_nonzero(zero))
        negative_zeros += int(np.count_nonzero(zero & (raw == np.uint32(0x80000000))))
        subnormals += int(np.count_nonzero((exponent == 0) & (mantissa != 0)))
        full = exponent == np.uint32(0x7F800000)
        nans += int(np.count_nonzero(full & (mantissa != 0)))
        infinities += int(np.count_nonzero(full & (mantissa == 0)))
    return {"float32_words": total, "zeros": zeros, "negative_zeros": negative_zeros,
            "subnormals": subnormals, "nans": nans, "infinities": infinities}


def moved_words(before: Dict[str, np.ndarray],
                after: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """How many words the ORACLE changed over an interval. The non-vacuity floor.

    Zero-init is a fixed point of every one of these recurrences: an interval in
    which the array path changed nothing certifies nothing, and a deliberately
    wrong kernel would still report IDENTICAL over it.
    """
    moved = total = 0
    live: List[str] = []
    for name in sorted(set(before) & set(after)):
        a, b = words(before[name]), words(after[name])
        if a.shape != b.shape:
            continue
        count = int(np.count_nonzero(a != b))
        total += int(a.size)
        moved += count
        if count:
            live.append(name)
    return {"moved_words": moved, "total_words": total,
            "moved_fraction": (moved / total) if total else 0.0,
            "arrays_that_moved": live}


# ---------------------------------------------------------------------------
# THE PLANTED DEFECTS
# ---------------------------------------------------------------------------
#
# Applied to the SUBSTITUTED leg only, never to the oracle. Each names the real
# defect it stands for. A defect leg that does not diverge is a finding about this
# harness, not a clean bill for the kernel.

DEFECTS: Dict[str, str] = {
    "none":
        "THE NULL CONTROL. The wrappers are installed on all five slots and route "
        "exactly as the predicate says. Must be NULL CONFIRMED (identical at every "
        "non-vacuous checkpoint). A leg whose null control diverges is measuring "
        "the harness.",
    "swap_constitutive_kps_kms":
        "(kap+sig) and (kap-sig) exchanged in the constitutive tables: the sign of "
        "the absorption reversed on the auxiliary term. A wrong answer with no "
        "error, and bit-identical OUTSIDE the layer where kps == kms == 1, so it "
        "can only be caught inside the absorber — which is a claim about whether a "
        "real row's fields ever reach the absorber at all.",
    "swap_curl_sublattice":
        "step_B handed the INTEGER coefficient set and step_D the HALF-INTEGER one. "
        "A half-cell error in the absorber profile: converged, smooth and wrong. "
        "The kernel file's own note calls confusing the two the highest-consequence "
        "error in it.",
    "drop_fu_store":
        "The curl's split-field auxiliary is restored after the substituted launch, "
        "i.e. the store is dropped. CORRECT FOR EXACTLY ONE LAUNCH and wrong "
        "forever after — the defect class a single-comparison gate is blind to and "
        "a checkpointed long run is built for.",
    "mirror_as_metallic":
        "A folded PERIODIC axis handed BC_METALLIC instead of BC_MIRROR_PERIODIC — "
        "literally the pre-2026-08-20 kernel. It differs on ONE PLANE of one axis, "
        "so it is also the test of whether this harness can LOCALIZE rather than "
        "merely detect. Inert on a row with no folded periodic axis, and the record "
        "says which rows it could bite on.",
    "drop_update_H":
        "The covered sub-step is dropped from the step entirely: the wrapper claims "
        "the slot and does nothing. The 'coverage is a set' mis-plumbing (Triton "
        "spec section G.m10). Must diverge at the first checkpoint.",
}


# ---------------------------------------------------------------------------
# THE SUBSTITUTION
# ---------------------------------------------------------------------------

class Substitution:
    """Point the driver module's five sub-step names at the kernels, scoped to one Fields.

    The plan is frozen ONCE, at install, exactly as ``FdtdDriver.step`` freezes its
    own fast-path plan at the first step (driver.py:3256-3262): the coefficient
    tables and the boundary codes are derived once and reused, so a step cannot run
    half on tables built from one configuration and half on another.
    """

    def __init__(self, backend: str, fields: Any, pml: Any, grid: Any,
                 defect: str = "none") -> None:
        from meep_gpu.cuda_kernels import coverage  # noqa: PLC0415
        import gate_cuda_folded_constitutive as gcons  # noqa: PLC0415
        import gate_cuda_folded_curl as gcurl  # noqa: PLC0415

        self.backend = backend
        self.fields = fields
        self.pml = pml
        self.grid = grid
        self.defect = defect
        self.coverage = coverage
        self.gcurl = gcurl
        self.gcons = gcons
        self.dispatches: Dict[str, int] = {slot: 0 for slot in SLOTS}
        self.fallbacks: Dict[str, int] = {slot: 0 for slot in SLOTS}
        self.foreign_calls = 0
        self.errors: List[str] = []
        self._undo: Optional[Callable[[], None]] = None
        self._originals: Dict[str, Callable[..., Any]] = {}

        self.proxy = grid if getattr(getattr(grid, "xp", None), "__name__", "") == "cupy" \
            else _GridWithCupyBackend(grid)
        self.shim = shim_soundness(grid, self.proxy)
        self.dtdx = float(grid.dt / grid.dx)

        self.admission: Dict[str, Dict[str, Any]] = {}
        self.runners: Dict[str, Callable[[Any], None]] = {}
        self._plan()

    # -- planning -----------------------------------------------------------

    def _ask(self, name: str, call: Callable[[], Tuple[Any, Any]]) -> Dict[str, Any]:
        try:
            covered, reason = call()
        except BaseException as exc:  # noqa: BLE001 - a raise is a refusal WITH a reason
            return {"admitted": False, "reason": f"PREDICATE RAISED "
                                                 f"{type(exc).__name__}: {exc}"[:400],
                    "raised": True, "predicate": name}
        return {"admitted": bool(covered), "reason": str(reason)[:400],
                "raised": False, "predicate": name}

    def _plan(self) -> None:
        coverage, gcurl, gcons = self.coverage, self.gcurl, self.gcons
        fields, pml, proxy = self.fields, self.pml, self.proxy

        # --- the curl pair -------------------------------------------------
        codes: Optional[Tuple[int, int, int]] = None
        kinds: Optional[Tuple[str, ...]] = None
        for sub_step in ("step_B", "step_D"):
            verdict = self._ask(
                "coverage.covers_real_pml_curl",
                lambda s=sub_step: coverage.covers_real_pml_curl(fields, pml, proxy, s))
            self.admission[sub_step] = verdict
            if not verdict["admitted"]:
                continue
            if codes is None:
                arm = ("mirror_as_metallic" if self.defect == "mirror_as_metallic"
                       else gcurl.LICENSED_SUBSTITUTION)
                codes, kinds = gcurl.boundary_codes_for(proxy, arm)
                self.codes, self.kinds, self.code_arm = codes, kinds, arm
            tables = (gcurl.host_mutated_tables("swap_curl_sublattice", sub_step, pml,
                                                proxy)
                      if self.defect == "swap_curl_sublattice"
                      else gcurl.tables_for(sub_step, pml))
            self.runners[sub_step] = self._curl_runner(sub_step, tables, codes)

        # --- the constitutive pair, and the off-diagonal E ------------------
        for slot, side in (("update_H", "H"), ("update_E", "E")):
            verdict = self._ask(
                "coverage.covers_real_pml_constitutive",
                lambda s=side: coverage.covers_real_pml_constitutive(
                    fields, pml, proxy, s))
            self.admission[slot] = verdict
            if not verdict["admitted"]:
                continue
            tables = (gcons.host_mutated_tables("swap_kps_kms", side, pml, proxy)
                      if self.defect == "swap_constitutive_kps_kms"
                      else gcons.tables_for(side, pml))
            self.runners[slot] = self._constitutive_runner(side, tables)

        offdiag = self._ask(
            "coverage.covers_real_pml_offdiag_constitutive",
            lambda: coverage.covers_real_pml_offdiag_constitutive(fields, pml, proxy))
        self.admission["update_E_offdiag"] = offdiag
        if offdiag["admitted"]:
            if "update_E" in self.runners:
                # The two E-side predicates are measured DISJOINT over all 186 rows.
                # A row where both admit is a finding about the PREDICATES, and this
                # harness must not silently pick one.
                raise RuntimeError(
                    "BOTH E-side predicates admit this row; the census measured that "
                    "set empty (rows admitted by BOTH E-side predicates: 0). Refusing "
                    "to choose a family on the engine's behalf.")
            self.runners["update_E"] = self._offdiag_runner()

        # ``update_P`` is never substituted: there is no CUDA kernel and no CUDA
        # predicate for it. Recorded as a refusal with that reason rather than left
        # as a silent hole in the slot list.
        self.admission["update_P"] = {
            "admitted": False, "raised": False, "predicate": None,
            "reason": "no CUDA kernel and no CUDA predicate for update_P"}

        # THE DEFECT MAY CLAIM A SLOT THE PREDICATE REFUSED, and that is deliberate:
        # "the covered sub-step was dropped from the step entirely" has to fire whether
        # or not the slot is in the plan, or it aims at the refusal branch only and
        # never applies at all. The sibling track measured exactly that — an m10 leg
        # reporting 10/10 identical for a mutation it never applied — so the slots the
        # DEFECT installed are recorded separately from the ones the predicate admitted.
        self.defect_installed: List[str] = []
        if self.defect == "drop_update_H":
            self.runners["update_H"] = self._noop_runner()
            self.defect_installed.append("update_H")

    # -- the four runners ---------------------------------------------------

    def _curl_runner(self, sub_step: str, tables: Dict[str, Any],
                     codes: Tuple[int, int, int]) -> Callable[[Any], None]:
        gcurl = self.gcurl
        runner = (gcurl.run_kernel_cuda if self.backend == "cuda"
                  else gcurl.run_kernel_numpy)
        drop_fu = self.defect == "drop_fu_store"
        aux = tuple(gcurl.SUB_STEP_ARRAYS[sub_step]["aux"])

        def run(fields: Any) -> None:
            if drop_fu:
                held = {name: getattr(fields, name).copy() for name in aux}
            runner(sub_step, fields, tables, codes, self.dtdx)
            if drop_fu:
                for name, value in held.items():
                    getattr(fields, name)[...] = value
        return run

    def _constitutive_runner(self, side: str,
                             tables: Dict[str, Any]) -> Callable[[Any], None]:
        gcons = self.gcons
        runner = (gcons.run_kernel_cuda if self.backend == "cuda"
                  else gcons.run_kernel_numpy)

        def run(fields: Any) -> None:
            runner(side, fields, tables)
        return run

    def _offdiag_runner(self) -> Callable[[Any], None]:
        """``update_E`` through the off-diagonal family's own launcher.

        The row volumes, the row mask and the wall flags are read with the SHIPPED
        helpers (``coverage.offdiag_row_volumes`` / ``.offdiag_row_mask`` /
        ``.offdiag_wall_mask_flags``) — the single place the slot binding is derived
        from the engine's own rows, so the predicate, the launcher and the emitter
        cannot disagree about which coefficient volume pairs with which partner.
        """
        import gate_cuda_offdiag as goff  # noqa: PLC0415

        coverage = self.coverage
        fields, pml, proxy = self.fields, self.pml, self.proxy
        mask = coverage.offdiag_row_mask(fields)
        rows: Dict[str, Dict[str, Any]] = {}
        for row, partner in coverage.OFFDIAG_ROW_SLOTS:
            volume = (fields.chi1inv_offdiagonal_for(row) or {}).get(partner)
            if volume is not None:
                rows.setdefault(row, {})[partner] = volume
        # The sub-lattice is asked of the shipped helper rather than transcribed:
        # getting it backwards is a half-cell error in the absorber, not a crash.
        if not coverage.constitutive_sub_lattice("E"):
            raise RuntimeError(
                "coverage.constitutive_sub_lattice('E') is False, but the off-diagonal "
                "gate binds the HALF-INTEGER tables for update_E; the two have drifted "
                "and this harness will not guess which is right.")
        tables = goff.half_integer_tables(pml)
        codes = tuple(coverage.BC_CODES[kind]
                      for kind in coverage.real_pml_boundary_kinds(proxy))
        walls = coverage.offdiag_wall_mask_flags(proxy)
        backend = goff.build_backend("cuda" if self.backend == "cuda" else "numpy")
        self.offdiag_facts = {"row_mask": [int(v) for v in mask],
                              "codes": [int(c) for c in codes],
                              "walls": [int(w) for w in walls],
                              "backend": type(backend).__name__}

        def run(target: Any) -> None:
            backend.launch(target, pml, rows, mask, tables, codes, walls)
        return run

    def _noop_runner(self) -> Callable[[Any], None]:
        def run(fields: Any) -> None:
            return None
        return run

    # -- install / undo -----------------------------------------------------

    @property
    def substituted_slots(self) -> Tuple[str, ...]:
        return tuple(slot for slot in SLOTS if slot in self.runners)

    def install(self) -> None:
        from meep_gpu import driver as driver_module  # noqa: PLC0415

        originals = {slot: getattr(driver_module, slot) for slot in SLOTS}
        self._originals = originals
        owner = self.fields

        def make(slot: str) -> Callable[..., Any]:
            def wrapper(fields: Any, pml: Any = None) -> Any:
                # SCOPED. The module globals are shared by every driver in the
                # process; without this check the ORACLE steps through the kernels
                # too and every array diverges at step 1 for a reason that is not
                # the kernel's.
                if fields is not owner:
                    self.foreign_calls += 1
                    return originals[slot](fields, pml)
                runner = self.runners.get(slot)
                if runner is None:
                    self.fallbacks[slot] += 1
                    return originals[slot](fields, pml)
                self.dispatches[slot] += 1
                runner(fields)
                return None
            return wrapper

        for slot in SLOTS:
            setattr(driver_module, slot, make(slot))

        def undo() -> None:
            for slot, function in originals.items():
                setattr(driver_module, slot, function)
        self._undo = undo

    def uninstall(self) -> None:
        if self._undo is not None:
            self._undo()
            self._undo = None

    def armability(self, pristine: "Substitution") -> Dict[str, Any]:
        """IS THIS DEFECT EVEN VISIBLE ON THIS ROW, from this row's own live state?

        THE FLOOR THIS EXISTS FOR, and it was measured before it was written. The
        constitutive kps/kms swap came back UNCAUGHT on ``metasurface_lens.py`` over
        8000 steps, which reads exactly like a harness that cannot fail. It is not:
        the sub-step's component ``c`` reads AXIS ``c``'s coefficient vector, that
        row's only live components are ``Ez``/``By``, and its ``z`` and ``y`` vectors
        are the identity while the absorber lives on ``x`` (measured: max|kps-kms| is
        0.254 on x and exactly 0 on y and z; ``Dx``, ``Dy``, ``Bx``, ``Bz`` are
        identically 0.0 at every step). The mutation is REAL and lands on a table
        nothing on that row reads.

        "UNCAUGHT" and "could not have been caught" are different verdicts and only
        one of them is about the harness, so they are separated HERE rather than in a
        reader's head. From the live state, each substituted slot is run once with the
        pristine tables and once with the mutated ones, on the same frozen state, and
        the two outputs are compared as words. Identical means the mutation changed
        nothing from THIS state.

        **IT IS A PROPERTY OF THE STATE, NOT ONLY OF THE ROW, AND THAT WAS MEASURED
        THE HARD WAY.** The first version of this probe ran ONCE, at the first
        checkpoint, on the stated reasoning that the answer is fixed by which
        coefficient vectors the live components read. That reasoning is wrong and the
        harness's own record falsified it inside one run: ``swap_curl_sublattice`` on
        ``metasurface_lens.py`` probed INERT at step 50 and the leg then DIVERGED at
        step 100 — the field had not yet reached the mutated rows at the earlier
        probe. A single early probe therefore manufactures exactly the false "could
        not have been caught" it exists to prevent. It is taken at EVERY checkpoint,
        and a row counts as armed if the mutation ever changed a word.

        The run is not perturbed: the state is snapshotted before and restored after,
        so the stepping continues from exactly the bytes it would have.
        """
        xp = self.grid.xp
        out: Dict[str, Any] = {"slots": {}}
        frozen = inventory(self.fields)
        for slot in self.substituted_slots:
            mutated_runner = self.runners.get(slot)
            pristine_runner = pristine.runners.get(slot)
            if mutated_runner is None:
                continue
            if pristine_runner is None:
                # The defect claimed a slot the predicate refused; the pristine tree
                # runs the array path there, which is the honest comparison.
                # ``self``'s originals, not ``pristine``'s: the pristine plan is built
                # but never installed, so it captured none.
                pristine_runner = lambda fields, s=slot: (  # noqa: E731
                    self.array_path(s)(fields, self.pml))
            restore_inventory(self.fields, frozen, xp)
            pristine_runner(self.fields)
            self.synchronize()
            reference = inventory(self.fields)
            restore_inventory(self.fields, frozen, xp)
            mutated_runner(self.fields)
            self.synchronize()
            produced = inventory(self.fields)
            verdict = compare_inventories(reference, produced)
            out["slots"][slot] = {
                "mutation_changes_output": not verdict["bit_identical"],
                "differing_words": verdict["differing_words"],
                "differing_arrays": verdict["differing_arrays"],
            }
        restore_inventory(self.fields, frozen, xp)
        out["armed"] = any(entry["mutation_changes_output"]
                           for entry in out["slots"].values())
        return out

    def array_path(self, slot: str) -> Callable[..., Any]:
        """The ENGINE's own function for a slot, for the armability probe's control.

        Taken from the originals captured at install, never re-read off the driver
        module: by probe time that name is this harness's own wrapper, and reading it
        back would make the control call the mutated tree.
        """
        return self._originals[slot]

    def synchronize(self) -> None:
        if self.backend != "cuda":
            return
        import cupy as cp  # noqa: PLC0415
        cp.cuda.runtime.deviceSynchronize()

    def report(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "backend": self.backend,
            "defect": self.defect,
            "dtdx": self.dtdx,
            "admission": self.admission,
            "substituted_slots": list(self.substituted_slots),
            "slots_substituted": len(self.substituted_slots),
            "defect_installed_slots": list(getattr(self, "defect_installed", ())),
            "slots_admitted_by_predicate": [
                slot for slot in self.substituted_slots
                if slot not in getattr(self, "defect_installed", ())],
            "dispatches": dict(self.dispatches),
            "fallbacks": dict(self.fallbacks),
            "foreign_calls": self.foreign_calls,
            "shim": self.shim,
        }
        if hasattr(self, "codes"):
            record["boundary_codes"] = [int(c) for c in self.codes]
            record["boundary_kinds"] = list(self.kinds)
            record["boundary_code_arm"] = self.code_arm
        if hasattr(self, "offdiag_facts"):
            record["offdiag"] = self.offdiag_facts
        return record


# ---------------------------------------------------------------------------
# THE CHILD: one row, lifted twice, stepped in lockstep
# ---------------------------------------------------------------------------

def capture_row(row: Dict[str, Any], case_timeout: float):
    """``(record, sim, restore)`` for one corpus row, through the SURVEY's own capture.

    The examples leg reuses ``sweep_corpus_lift_parity.capture_simulation`` and the
    tests leg the survey's ``_run_case``, so the object stepped here is the object
    the 2026-08-09 lift record and the 2026-08-20 predicate census scored. The tests
    leg runs EVERY case in the module in enumeration order and keeps the wanted one:
    several MEEP test modules depend on an earlier case having run (``test_dump_load``
    writes then reads), so running only the wanted subset would be a different
    experiment.
    """
    if row["leg"] == "examples":
        from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
            capture_simulation,
        )
        record, sim, restore = capture_simulation(
            os.path.join(EXAMPLES_DIR, row["script"]))
        return record, sim, restore

    from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

    namespace = harness.build_child_namespace()
    module_path = os.path.join(TESTS_DIR, row["module"])
    module = namespace["_import_module"](module_path)
    wanted = row["case"]
    for class_name, method_name in namespace["_enumerate_cases"](module):
        record, sim, restore, _case = namespace["_run_case"](
            module, module_path, class_name, method_name, case_timeout)
        restore()
        if f"{class_name}.{method_name}" == wanted:
            return record, sim, (lambda: None)
    raise RuntimeError(f"case {wanted!r} not found in {row['module']}")


def lift_twice(sim, prefer_gpu: bool):
    """Two drivers from ONE declared simulation, and why it is not one lift copied.

    ``lift_simulation`` calls ``sim.init_sim()``, and a second lift of an already
    initialized simulation is REFUSED by name (from_meep.py:4989: "this simulation
    has already been initialized ... everything a caller can do to an initialized
    MEEP simulation is INVISIBLE to this converter"). ``sim.reset_meep()`` is the
    engine's own answer — it rebuilds from the declared arguments, which ARE
    readable — so both drivers come from the same declaration rather than from a
    copy of one driver's arrays.

    Byte-equality of the two is a PRECONDITION checked by the caller, never assumed:
    two separately-lifted drivers can differ in their inputs silently, and a
    comparison of two runs that never shared a starting state measures nothing.
    """
    import meep_gpu  # noqa: PLC0415

    reference = meep_gpu.lift_simulation(sim, prefer_gpu=prefer_gpu)
    sim.reset_meep()
    substituted = meep_gpu.lift_simulation(sim, prefer_gpu=prefer_gpu)
    return reference, substituted


def run_row(row: Dict[str, Any], backend: str, steps: int, every: int,
            defect: str, out_dir: str, case_timeout: float) -> Dict[str, Any]:
    """Lift, plan, install, step both drivers in lockstep, checkpoint, compare."""
    started = time.time()
    label = row["label"]
    per_row_path = os.path.join(out_dir, "per_row", f"{label}.json")
    checkpoints_path = os.path.join(out_dir, "checkpoints.jsonl")
    record: Dict[str, Any] = {
        "label": label, "leg": row["leg"], "why": row["why"], "census": row["census"],
        "backend": backend, "defect": defect, "steps_requested": steps,
        "checkpoint_every": every, "seed": case_seed(label, backend, defect, steps),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoints": [],
    }
    save(record, per_row_path)

    say(f"[{label}] capturing")
    capture, sim, restore = capture_row(row, case_timeout)
    restore()
    record["capture"] = {"has_simulation": capture.get("has_simulation"),
                         "accepted": capture.get("accepted")}
    if sim is None:
        record["error"] = "no mp.Simulation to lift"
        save(record, per_row_path)
        return record

    say(f"[{label}] lifting twice (prefer_gpu={backend == 'cuda'})")
    reference, substituted = lift_twice(sim, prefer_gpu=(backend == "cuda"))
    record["grid"] = {
        "shape": [int(v) for v in reference.shape],
        "cells": int(math.prod(int(v) for v in reference.shape)),
        "dt": float(reference.grid.dt), "dx": float(reference.grid.dx),
        "xp": getattr(reference.grid.xp, "__name__", None),
        "pml_active": bool(reference.pml is not None
                           and getattr(reference.pml, "is_active", False)),
        "n_sources": len(getattr(reference, "_sources", ()) or ()),
        "n_polarizations": len(getattr(reference.fields, "polarizations", ()) or ()),
        "force_complex_fields": bool(getattr(reference.fields,
                                             "force_complex_fields", False)),
    }

    # THE PRECONDITION. Checked, never assumed: a comparison of two runs that never
    # shared a starting state measures nothing, and a failure here is a HARNESS
    # failure, not a divergence.
    initial = compare_inventories(inventory(reference.fields),
                                  inventory(substituted.fields))
    record["seeded_identically"] = initial["bit_identical"]
    record["seeded_identically_detail"] = {
        k: v for k, v in initial.items() if k != "detail"}
    if not initial["bit_identical"]:
        record["error"] = ("HARNESS FAILURE: the two lifted drivers do not start "
                           "byte-equal")
        record["seconds"] = time.time() - started
        save(record, per_row_path)
        return record

    plan = Substitution(backend, substituted.fields, substituted.pml,
                        substituted.grid, defect=defect)
    plan.install()
    record["plan"] = plan.report()
    say(f"[{label}] slots={plan.substituted_slots or '(none — inert control)'} "
        f"shape={record['grid']['shape']} cells={record['grid']['cells']}")
    save(record, per_row_path)

    last_reference = inventory(reference.fields)
    last_dispatches = dict(plan.dispatches)
    pristine_plan: Optional[Substitution] = None
    identical = vacuous = 0
    first_divergence: Optional[int] = None
    try:
        for step in range(1, steps + 1):
            reference.step()
            substituted.step()
            if step % every and step != steps:
                continue
            plan.synchronize()
            reference_state = inventory(reference.fields)
            produced = inventory(substituted.fields)
            verdict = compare_inventories(reference_state, produced)
            movement = moved_words(last_reference, reference_state)
            interval = {slot: plan.dispatches[slot] - last_dispatches[slot]
                        for slot in SLOTS}
            last_reference, last_dispatches = reference_state, dict(plan.dispatches)
            silent = [slot for slot in plan.substituted_slots if interval[slot] == 0]
            entry = {
                "label": label, "backend": backend, "defect": defect, "step": step,
                "bit_identical": verdict["bit_identical"],
                "differing_words": verdict["differing_words"],
                "total_words": verdict["total_words"],
                "differing_arrays": verdict["differing_arrays"],
                "inventory_mismatch": verdict["inventory_mismatch"],
                "oracle_moved_words": movement["moved_words"],
                "oracle_moved_fraction": movement["moved_fraction"],
                "oracle_arrays_that_moved": len(movement["arrays_that_moved"]),
                "oracle_operand_census": operand_census(reference_state),
                "dispatches_in_interval": interval,
                "slots_silent_in_interval": silent,
                # A checkpoint the array path did not move over, or over which a
                # substituted slot never fired, cannot distinguish a correct kernel
                # from a wrong one. It is reported and NOT counted.
                "vacuous": movement["moved_words"] == 0 or bool(silent),
                "elapsed_s": round(time.time() - started, 2),
            }
            if not verdict["bit_identical"]:
                entry["worst"] = dict(list(verdict["detail"].items())[:6])
            # THE ARMABILITY FLOOR, taken at EVERY checkpoint. Not once: the answer
            # moves with the state (see Substitution.armability), and an early-only
            # probe manufactures the false "could not have been caught" the floor
            # exists to prevent.
            if defect != "none":
                if pristine_plan is None:
                    pristine_plan = Substitution(backend, substituted.fields,
                                                 substituted.pml, substituted.grid,
                                                 defect="none")
                try:
                    armability = plan.armability(pristine_plan)
                except BaseException as exc:  # noqa: BLE001
                    armability = {"error": f"{type(exc).__name__}: {exc}"[:400],
                                  "armed": None}
                entry["defect_armability"] = armability
                entry["defect_armed"] = armability.get("armed")
                record.setdefault("defect_armability_by_step", {})[str(step)] = \
                    armability
                if armability.get("armed"):
                    record["defect_armed"] = True
                    record.setdefault("defect_armed_first_step", step)
                elif armability.get("armed") is None:
                    record.setdefault("defect_armed", None)
                else:
                    record.setdefault("defect_armed", False)
            record["checkpoints"].append(entry)
            append_jsonl(checkpoints_path, entry)
            if entry["vacuous"]:
                vacuous += 1
            elif verdict["bit_identical"]:
                identical += 1
            if not verdict["bit_identical"] and first_divergence is None:
                first_divergence = step
            say(f"[{label}] step {step}/{steps} "
                f"{'IDENTICAL' if verdict['bit_identical'] else 'DIVERGED'} "
                f"diff={verdict['differing_words']}/{verdict['total_words']} "
                f"arrays={verdict['differing_arrays'][:4]} "
                f"moved={movement['moved_fraction']:.4f}"
                f"{' VACUOUS' if entry['vacuous'] else ''} "
                f"({entry['elapsed_s']} s)")
            save(record, per_row_path)
    except BaseException as exc:  # noqa: BLE001 - a raise mid-run is a result
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        plan.uninstall()

    record["plan"] = plan.report()
    record["identical_checkpoints"] = identical
    record["vacuous_checkpoints"] = vacuous
    record["first_divergence_step"] = first_divergence
    record["diverged"] = first_divergence is not None
    record["steps_run"] = min(steps, record["checkpoints"][-1]["step"]
                              if record["checkpoints"] else 0)
    record["seconds"] = round(time.time() - started, 2)
    save(record, per_row_path)
    return record


# ---------------------------------------------------------------------------
# THE PARENT
# ---------------------------------------------------------------------------

def subject_digest(api_root: str) -> Dict[str, Any]:
    """sha256 of every subject file, plus one digest over the whole manifest."""
    entries: Dict[str, str] = {}
    directory = os.path.join(api_root, SUBJECT_GLOB)
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                entries[os.path.relpath(path, api_root)] = hashlib.sha256(
                    open(path, "rb").read()).hexdigest()
    for name in SUBJECT_EXTRA:
        path = os.path.join(api_root, name)
        if os.path.exists(path):
            entries[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest = "".join(f"{k}  {v}\n" for k, v in sorted(entries.items()))
    return {"files": entries, "file_count": len(entries),
            "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest()}


def verdict(rows: Sequence[Dict[str, Any]], defect: str) -> Dict[str, Any]:
    """What the run establishes, and what it refuses to call established.

    A defect leg is scored CAUGHT/UNCAUGHT and the null leg NULL CONFIRMED, and the
    two are scored by DIFFERENT rules on purpose: a battery of only-must-be-caught
    legs scores identically whether the comparator works or has degenerated into
    failing everything.
    """
    def admitted(row: Dict[str, Any]) -> int:
        # BY THE PREDICATE, not by the defect. A defect that claims a refused slot
        # would otherwise promote the inert control into an evidential row and take
        # the one leg that can only be measuring the harness out of the record.
        plan = row.get("plan", {})
        if "slots_admitted_by_predicate" in plan:
            return len(plan["slots_admitted_by_predicate"])
        return plan.get("slots_substituted", 0)

    scored = [row for row in rows if not row.get("error")]
    evidential = [row for row in scored if admitted(row) > 0]
    inert = [row for row in scored if admitted(row) == 0]
    non_vacuous = sum(row.get("identical_checkpoints", 0) for row in evidential)
    diverged = [row["label"] for row in evidential if row.get("diverged")]
    inert_diverged = [row["label"] for row in inert if row.get("diverged")]
    out: Dict[str, Any] = {
        "defect": defect,
        "rows_run": len(rows),
        "rows_with_error": [row["label"] for row in rows if row.get("error")],
        "rows_with_kernels": [row["label"] for row in evidential],
        "rows_inert": [row["label"] for row in inert],
        "non_vacuous_identical_checkpoints": non_vacuous,
        "vacuous_checkpoints": sum(row.get("vacuous_checkpoints", 0) for row in scored),
        "rows_diverged": diverged,
        "inert_rows_diverged": inert_diverged,
    }
    if defect == "none":
        out["null_confirmed"] = (not diverged and not inert_diverged
                                 and not out["rows_with_error"]
                                 and non_vacuous > 0)
        out["release"] = bool(out["null_confirmed"])
    else:
        # THREE OUTCOMES, NOT TWO. A row where the mutation provably changed no
        # output word from that row's own live state is INERT for this defect, and
        # scoring it UNCAUGHT would charge the harness for a property of the row.
        armed = [row for row in evidential if row.get("defect_armed") is not False]
        out["caught_rows"] = [row["label"] for row in armed if row.get("diverged")]
        out["uncaught_rows"] = [row["label"] for row in armed
                                if not row.get("diverged")]
        out["defect_inert_rows"] = [row["label"] for row in evidential
                                    if row.get("defect_armed") is False]
        out["armability_unmeasured_rows"] = [row["label"] for row in evidential
                                             if row.get("defect_armed") is None]
        out["caught"] = bool(out["caught_rows"])
        # THE STRONGER STATEMENT, and the one worth quoting: every row on which the
        # mutation could bite, bit.
        out["caught_on_every_armed_row"] = bool(armed) and not out["uncaught_rows"]
        out["first_divergence_steps"] = {
            row["label"]: row.get("first_divergence_step") for row in evidential}
        out["defect_armed_by_row"] = {row["label"]: row.get("defect_armed")
                                      for row in evidential}
        out["defect_armed_first_step"] = {
            row["label"]: row.get("defect_armed_first_step") for row in evidential}
        # A defect leg never releases anything: it is the evidence that the leg CAN
        # fail, not evidence about the kernels.
        out["release"] = False
        out["inert_rows_must_not_diverge"] = not inert_diverged
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _PROGRESS_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    parser.add_argument("--rows", default="all",
                        help="comma-separated row labels, or 'all'")
    parser.add_argument("--examples-dir", default=None,
                        help="MEEP python/examples; defaults to MEEP_PYTHON_DIR, "
                             "then a checkout beside $HOME")
    parser.add_argument("--tests-dir", default=None,
                        help="MEEP python/tests; same resolution order")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--defect", choices=tuple(DEFECTS), default="none")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="per-row wall clock ceiling in the parent")
    parser.add_argument("--case-timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-row", default=None, help="child mode; not for hand use")
    args = parser.parse_args(argv)
    # Bind the corpus BEFORE anything reads EXAMPLES_DIR/TESTS_DIR.
    resolve_corpus(args.examples_dir, args.tests_dir)

    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "per_row"), exist_ok=True)
    _PROGRESS_PATH = os.path.join(out_dir, "progress.log")

    if args.child_row:
        record = run_row(ROWS_BY_LABEL[args.child_row], args.backend, args.steps,
                         args.checkpoint_every, args.defect, out_dir,
                         args.case_timeout)
        return 0 if not record.get("error") else 1

    labels = (tuple(row["label"] for row in ROWS) if args.rows == "all"
              else tuple(part.strip() for part in args.rows.split(",") if part.strip()))
    unknown = [label for label in labels if label not in ROWS_BY_LABEL]
    if unknown:
        print(f"unknown row labels: {unknown}; known: {sorted(ROWS_BY_LABEL)}",
              flush=True)
        return 2

    results: Dict[str, Any] = {
        "what": "end-to-end stress: real corpus rows, long horizons, CUDA kernels "
                "substituted at every admitted slot, full inventory compared as "
                "uint32 words at checkpoints",
        "certifies": args.backend == "cuda",
        "backend": args.backend,
        "defect": args.defect,
        "defect_means": DEFECTS[args.defect],
        "steps": args.steps,
        "checkpoint_every": args.checkpoint_every,
        "rows_requested": list(labels),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "subject_before": subject_digest(_API),
        "rows": [],
    }
    artifact = os.path.join(out_dir, "stress.json")
    save(results, artifact)

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = _API + os.pathsep + environment.get("PYTHONPATH", "")

    workdir = os.path.join(out_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    # The examples corpus reads data files relative to its own directory; the sweep
    # solves this by symlinking every non-script entry into the working directory.
    for entry in sorted(os.listdir(EXAMPLES_DIR)):
        if entry.endswith((".py", ".ipynb")):
            continue
        link = os.path.join(workdir, entry)
        if not os.path.exists(link):
            try:
                os.symlink(os.path.join(EXAMPLES_DIR, entry), link)
            except OSError:
                pass

    say(f"stress: backend={args.backend} defect={args.defect} steps={args.steps} "
        f"every={args.checkpoint_every} rows={len(labels)}")
    for index, label in enumerate(labels, start=1):
        per_row_path = os.path.join(out_dir, "per_row", f"{label}.json")
        if args.resume and os.path.exists(per_row_path):
            say(f"  {index}/{len(labels)} {label}: resumed from disk")
            results["rows"].append(json.load(open(per_row_path, encoding="utf-8")))
            save(results, artifact)
            continue
        command = [sys.executable, "-u", os.path.abspath(__file__),
                   "--child-row", label, "--backend", args.backend,
                   "--steps", str(args.steps),
                   "--checkpoint-every", str(args.checkpoint_every),
                   "--defect", args.defect, "--out", out_dir,
                   "--case-timeout", str(args.case_timeout)]
        started = time.time()
        status = "ok"
        try:
            completed = subprocess.run(command, cwd=workdir, env=environment,
                                       timeout=args.timeout, check=False,
                                       stdout=None, stderr=subprocess.PIPE)
            stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
        except subprocess.TimeoutExpired as expired:
            status = "timeout"
            stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                           if expired.stderr else "")
        if os.path.exists(per_row_path):
            row_record = json.load(open(per_row_path, encoding="utf-8"))
        else:
            row_record = {"label": label,
                          "error": ("child died" if status == "ok" else "child timeout"),
                          "stderr_tail": stderr_text[-2000:]}
        if row_record.get("error") and stderr_text:
            row_record.setdefault("stderr_tail", stderr_text[-2000:])
        results["rows"].append(row_record)
        say(f"  {index}/{len(labels)} {label}: "
            f"slots={row_record.get('plan', {}).get('slots_substituted')} "
            f"identical={row_record.get('identical_checkpoints')} "
            f"vacuous={row_record.get('vacuous_checkpoints')} "
            f"first_divergence={row_record.get('first_divergence_step')} "
            f"{'ERROR ' + str(row_record.get('error'))[:120] if row_record.get('error') else ''}"
            f" ({time.time() - started:.1f} s)")
        save(results, artifact)

    results["subject_after"] = subject_digest(_API)
    results["subject_stable"] = (results["subject_after"]["manifest_sha256"]
                                 == results["subject_before"]["manifest_sha256"])
    results["verdict"] = verdict(results["rows"], args.defect)
    results["not_established"] = [
        "NOTHING ON A DEVICE unless backend=cuda: the NumPy leg runs the family "
        "gates' TRANSCRIPTIONS, compiles nothing and certifies nothing.",
        "Determinism: each row is ONE comparison against the oracle. A race that "
        "resolves the same way twice is invisible here.",
        "update_P, and every family with no CUDA predicate, is never substituted.",
        "Only the rows in ROWS. 6 of 186; the census's other arms are untouched.",
    ]
    save(results, artifact)
    say(f"verdict: {json.dumps(results['verdict'], default=str)}")
    say(f"subject_stable={results['subject_stable']} -> {artifact}")
    if args.defect == "none":
        return 0 if results["verdict"].get("null_confirmed") else 1
    return 0 if results["verdict"].get("caught") else 1


if __name__ == "__main__":
    raise SystemExit(main())
