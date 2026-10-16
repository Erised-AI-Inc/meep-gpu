"""The shared machinery of the two Triton cylindrical H->D device gates.

``gate_triton_cylindrical_real_fused_hd_pair.py`` (3 real m = 0 rows) and
``gate_triton_cylindrical_fused_hd_pair.py`` (16 complex rows) measure the SAME
product shape -- ``update_H`` into launch-local scratch (launch 1), then the certified
cylindrical ``step_D`` plan over the scratch ``H``: its column-serial radial scan
(launch 2) and its curl (launch 3) -- on two storage classes. Every leg below is
written once against a :class:`FamilySpec` the gate instantiates, and is the H->D
template gate's leg (``gate_triton_fused_hd_pair.py``) with two the cylindrical seam
owes:

``purity_ledger``     on the array path: how many ``Hy`` words ``update_H`` moves over
                      the rows the radial scan reads -- the floor that makes the
                      scan-reads-the-pre-launch-Hy mutation a claim about the product
                      rather than about a quiet fixture.
``ghost_observability`` read off the product's and the certified curl's own text: the
                      scan and the curl are the certified plan's (this product lifts no
                      curl or scan text), the radial difference's backward tap is
                      guarded, and the ownership mask zeroes the target that tap feeds
                      at the axis row -- the reason no edit to what the r near ghost
                      serves can be byte-visible here.

The 0.9.2 product fused the pre-``cumsum`` increment into launch 1 and kept the
array module's ``cumsum``; the legs that measured that stage (``increment_stage``,
``scan_order``) left with it.

Every product row also carries the cylindrical floors the CUDA cylindrical gate
defines: the radial difference the scan hands the curl is live (nonzero somewhere) and
the axis row moves.

THE TWO POLICIES NEED NOT BE SYMMETRIC. Where a family's predicate refuses a policy,
the composer selects nothing on its seam and the two composition references are
UNREACHABLE under it. What that record then measures is the product's arithmetic built
from arrays, against the array path and the two certified singles built the same way;
the verdict names the unreachable references rather than scoring them, and
``composer_reachable_under_this_policy`` is RECORDED, never a release clause a family
could pass under a policy its predicate refuses by construction.

THE EVIDENCE STANDARD is the template's: uint32 words per COMPLETE driver step over
every stored volume, never ``allclose``; every zero beside a control that moves words;
N of D with D named. Progress reporting: one flushed line per case, every row appended and fsynced
as it lands, the lift leg writing one JSON per corpus row and a progress log the child
appends to.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import linecache
import math
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402
import h_to_d_seam  # noqa: E402
import hd_seam_restore  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.subnormal_policy import SubnormalPolicyUnattainable  # noqa: E402
from meep_gpu.fastpath import (  # noqa: E402
    SYNC_PASS_OWNERS, SYNC_PATH_SLOTS, SYNC_UPDATE_H_PASS,
)
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402

#: The reference engines, in the order the record reports them (the template's).
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "purity_ledger", "ghost_observability"),
    "device": ("refusal", "arbitration", "product", "seed_scale", "launch_structure",
               "sync", "withdraw", "byte_neutral", "mutation", "disarm", "lift"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census the lift leg reads its corpus rows from, and the seam record whose
#: ``arms.triton`` names each row's two arms.
CENSUS = "predicate_coverage_triton_2026-09-04_cylm0"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

STEPS = 60
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: The outcome of a mutant the subnormal policy in force refuses to COMPILE: its
#: PTX is not uniform under that policy. Neither a word count nor a pass.
REFUSED_BY_THE_POLICY = "REFUSED BY THE POLICY"
LIFT_CLEAN_STEP_FLOOR = 8
SEED_SCALE_BITS = 80
EXIT_INCOMPLETE = 75

#: Kernel launches one run of a certified cylindrical ``step_D`` (or ``step_B``) plan
#: performs: its column-serial radial scan, then its curl (or fused curl) kernel.
#: The certified plans declare no ``kernel_launches_per_run``; this is read off their
#: ``run`` (``self.scan.run(guard)`` then ``kernel[...]``), and the independent kernel
#: counter checks it.
CYLINDRICAL_PLAN_KERNEL_LAUNCHES = 2


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The family adapter
# ---------------------------------------------------------------------------

class FamilySpec:
    """What one gate tells the kit about its product. Subclassed per gate."""

    #: The gate name, the family module, storage class and the cell's two arms.
    gate: str = ""
    family: Any = None
    complex_storage: bool = False
    cell_arms: Tuple[str, str] = ("", "")
    #: Every module whose bytes the verdict depends on (repo-relative).
    sources: Tuple[str, ...] = ()
    #: The synthetic fixtures ``(name, keywords)`` and which one the mutations use.
    cases: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    mutation_case: str = ""
    #: The launch-1 kernel's function name in the family module (the mutation seam).
    kernel_name: str = ""
    #: ``tag -> spec`` mutation table and the one byte-neutral edit.
    mutations: Dict[str, Dict[str, Any]] = {}
    byte_neutral: Dict[str, Any] = {}
    fixture_resolution: float = 10.0

    # -- fixtures ------------------------------------------------------------
    def driver_keywords(self, keywords: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def pml_spec(self, keywords: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    # -- plans -----------------------------------------------------------------
    def build_weld(self, driver: Any, kernel: Any = None, curl_kernel: Any = None,
                   forced: bool = False, scan_kernel: Any = None) -> Any:
        raise NotImplementedError

    def singles(self, driver: Any, forced: bool = False) -> Tuple[Any, Any]:
        raise NotImplementedError

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        raise NotImplementedError

    def default_kernels(self, owner: Any) -> Dict[str, Any]:
        """``{attribute path: default kernel}`` for every kernel ``owner`` launches; a
        dotted path (``"scan._kernel"``) names a kernel slot on a nested object."""
        raise NotImplementedError

    def kernel_launches_of(self, owner: Any) -> int:
        """Kernel launches ONE run of ``owner`` performs, by declaration."""
        declared = getattr(owner, "kernel_launches_per_run", None)
        if declared is not None:
            return int(declared)
        if getattr(owner, "scan", None) is not None:
            return CYLINDRICAL_PLAN_KERNEL_LAUNCHES
        return 1

    def kernel_of(self, module: Any) -> Any:
        return getattr(module, self.kernel_name)

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def half_integer_rebind(self, plan: Any, pml: Any) -> Any:
        raise NotImplementedError

    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        """(findings, measures) for the family's own lift and spelling pins."""
        raise NotImplementedError

    def arbitration_incumbents(self) -> Tuple[str, str]:
        """The labels the SHIPPED composer must put on step_B/update_H and step_D/update_E."""
        raise NotImplementedError

    def product_row(self) -> Dict[str, str]:
        """The CERTIFIED_FUSED_PRODUCTS row the arbitration leg injects in process."""
        raise NotImplementedError

    def policy_reaches_composer(self, policy: Optional[str]) -> Tuple[bool, str]:
        """Can the composer install this family's arms under ``policy``?"""
        return True, ""

    def composition_refusal_needle(self) -> str:
        return ""

    # -- the seam the host mutations reach --------------------------------------
    def word_view(self, array: Any) -> Any:
        """The pointer view of one rotating volume, as the plan binds it."""
        return array

    def host_launcher(self, host: str, driver: Any) -> Optional[Callable[[Any], Any]]:
        """A family-specific host mutation by name, or ``None`` when unknown."""
        return None

    def ghost_observability(self) -> Tuple[List[str], Dict[str, Any]]:
        """(findings, measures) read off the certified curl's own text."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Words, volumes, comparison  (the template's, verbatim)
# ---------------------------------------------------------------------------

def _module_of(array: Any) -> Any:
    if type(array).__module__.split(".")[0] == "cupy":  # pragma: no cover - device
        import cupy  # noqa: PLC0415

        return cupy
    return np


def _host(array: Any) -> np.ndarray:
    xp = _module_of(array)
    if xp is not np:  # pragma: no cover - device path
        return np.ascontiguousarray(xp.asnumpy(array))
    return np.ascontiguousarray(array)


def native_words(array: Any) -> Any:
    xp = _module_of(array)
    contiguous = xp.ascontiguousarray(array)
    if contiguous.dtype == xp.complex64:
        return contiguous.reshape(-1).view(xp.uint32)
    return xp.ascontiguousarray(
        contiguous.astype(xp.float32, copy=False)).reshape(-1).view(xp.uint32)


def words(array: Any) -> np.ndarray:
    contiguous = _host(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    return np.frombuffer(contiguous.astype(np.float32, copy=False).tobytes(),
                         dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = native_words(left), native_words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(_module_of(a).count_nonzero(a != b))


_RESET_VOLUMES: Optional[Tuple[str, ...]] = None


def reset_declared_volumes() -> Tuple[str, ...]:
    global _RESET_VOLUMES  # noqa: PLW0603
    if _RESET_VOLUMES is not None:
        return _RESET_VOLUMES
    text = (API_ROOT / "meep_gpu" / "fields.py").read_text(encoding="utf-8")
    match = re.search(r"\n    def reset\(self\).*?(?=\n    def )", text, re.S)
    if match is None:
        raise SystemExit("cannot find Fields.reset in fields.py; refusing to guess "
                         "the list of stored volumes")
    names = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", match.group(0)))
    _RESET_VOLUMES = tuple(sorted(names - {"polarizations",
                                           "_fmp_scratch_by_component"}))
    return _RESET_VOLUMES


def stored_volumes(fields: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in reset_declared_volumes():
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for component, array in (getattr(fields, "_fmp_scratch_by_component", None)
                             or {}).items():
        if array is not None:
            out[f"_fmp_scratch_by_component[{component}]"] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def capture(driver: Any) -> Dict[str, Any]:
    return {
        "volumes": {name: _module_of(array).array(array, copy=True)
                    for name, array in stored_volumes(driver.fields).items()},
        "dipoles": [getattr(source, "_applied_dipole", None)
                    for source in getattr(driver, "_sources", ())],
        "step_count": int(driver.step_count),
        # MEEP's start of H is state a step advances: the run's first update_H starts
        # H from B and every later one does not.
        "h_separated": bool(getattr(driver.fields, "h_separated", True)),
    }


def restore(driver: Any, snapshot: Mapping[str, Any]) -> List[str]:
    lazily: List[str] = []
    for name, array in stored_volumes(driver.fields).items():
        saved = snapshot["volumes"].get(name)
        if saved is None:
            array.fill(0)
            lazily.append(name)
        else:
            array[...] = saved
    for source, dipole in zip(getattr(driver, "_sources", ()), snapshot["dipoles"]):
        if dipole is not None:
            source._applied_dipole = dipole  # noqa: SLF001
    driver.step_count = snapshot["step_count"]
    if "h_separated" in snapshot:
        driver.fields.h_separated = bool(snapshot["h_separated"])
    return lazily


def _is_private(name: str) -> bool:
    return name.startswith("_")


def compare_snapshots(reference: Mapping[str, Any], other: Mapping[str, Any],
                      include_private: bool = False) -> Dict[str, int]:
    a, b = reference["volumes"], other["volumes"]
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a)
            if (include_private or not _is_private(name))
            and (n := differing(a[name], b[name]))}


AUTOPSY_WORDS = 12


def _classify(word: int) -> str:
    if not word & 0x7FFFFFFF:
        return "zero"
    if not word & 0x7F800000:
        return "subnormal"
    return "normal"


def autopsy(reference: Mapping[str, Any], other: Mapping[str, Any],
            difference: Mapping[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in sorted(difference):
        left, right = words(reference["volumes"][name]), words(other["volumes"][name])
        if left.shape != right.shape:
            out[name] = {"shape_differs": [int(left.size), int(right.size)]}
            continue
        where = np.flatnonzero(left != right)
        classes: Dict[str, int] = {}
        for a_word, b_word in zip(left[where].tolist(), right[where].tolist()):
            key = f"{_classify(a_word)}->{_classify(b_word)}"
            classes[key] = classes.get(key, 0) + 1
        transcript = []
        for index in where[:AUTOPSY_WORDS].tolist():
            a_word, b_word = int(left[index]), int(right[index])
            a_value = float(np.frombuffer(np.uint32(a_word).tobytes(), dtype=np.float32)[0])
            b_value = float(np.frombuffer(np.uint32(b_word).tobytes(), dtype=np.float32)[0])
            transcript.append({
                "index": int(index),
                "reference_word": f"0x{a_word:08x}", "other_word": f"0x{b_word:08x}",
                "reference": a_value, "other": b_value,
                "reference_class": _classify(a_word), "other_class": _classify(b_word),
                "word_distance": abs(a_word - b_word),
                "same_sign": (a_word >> 31) == (b_word >> 31)})
        out[name] = {"differing_words": int(where.size), "words_in_volume": int(left.size),
                     "classes": dict(sorted(classes.items())), "transcript": transcript}
    return out


def subnormal_census(snapshot: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, array in snapshot["volumes"].items():
        w = native_words(array)
        xp = _module_of(w)
        banded = int(xp.count_nonzero(
            ((w & xp.uint32(0x7F800000)) == 0) & ((w & xp.uint32(0x007FFFFF)) != 0)))
        if banded:
            out[name] = banded
    return out


# ---------------------------------------------------------------------------
# The synthetic fixture: a REAL cylindrical driver
# ---------------------------------------------------------------------------

EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}


def build_driver(spec: FamilySpec, keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS, amplitude: float = 0.37,
                 prefer_gpu: bool = True) -> Any:
    """One seeded ``FdtdDriver`` on a Dcyl grid of the case's shape.

    ``resolution = 10`` with ``cell_size = (nr/10, 0, nz/10)`` lands on ``(nr, 1, nz)``
    exactly (``meep_cell_count`` is ``int(size*a + 0.5)``). z is METALLIC unless the
    case says periodic (both terminations are in the corpus; the real family admits
    metallic only). The PML sits on the high r face and, when z is walled, on both z
    faces -- the corpus arrangement.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    nr, nz = keywords["shape"]
    resolution = float(spec.fixture_resolution)
    driver = FdtdDriver(
        cell_size=(nr / resolution, 0.0, nz / resolution), resolution=resolution,
        courant=float(keywords.get("courant", 0.35)),
        force_complex_fields=bool(spec.complex_storage),
        boundaries={"z": str(keywords.get("z_boundary", "metallic"))},
        cylindrical=True, m=int(keywords.get("m", 0)),
        accurate_fields_near_cylorigin=bool(keywords.get("accurate", False)),
        prefer_gpu=prefer_gpu, gpu_id=0)
    if tuple(int(n) for n in driver.grid.shape) != (int(nr), 1, int(nz)):
        raise RuntimeError(f"the grid built {tuple(driver.grid.shape)} for {(nr, 1, nz)}")
    driver.setup_pml(spec.pml_spec(keywords))
    shape = tuple(driver.grid.shape)
    xp = driver.xp
    driver.fields.set_epsilon_volumes(
        {name: xp.asarray(np.full(shape, np.float32(value), np.float32))
         for name, value in EPSILON.items()},
        {name: xp.asarray(np.full(shape, np.float32(1.0 / value), np.float32))
         for name, value in EPSILON.items()})
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        if array.dtype.name == "complex64":
            host = ((float(amplitude) * rng.standard_normal(array.shape)).astype(np.float32)
                    * scale).astype(np.complex64)
            host.imag = ((float(amplitude) * rng.standard_normal(array.shape))
                         .astype(np.float32) * scale)
            array[...] = _module_of(array).asarray(host)
        else:
            host = (float(amplitude) * rng.standard_normal(array.shape)).astype(np.float32) * scale
            array[...] = _module_of(array).asarray(host.astype(np.float32))
    # A SEEDED STATE IS A MID-RUN ONE unless the case says otherwise: H was started
    # from B long ago, so no arrangement starts it again. A first-update case
    # (``h_started`` False) keeps the flag False, and ``capture``/``restore`` carry it,
    # so every arrangement's step 1 is the run's first update_H -- MEEP's start of H
    # (``stepping.separate_h_from_b``), measured on every arrangement.
    driver.fields.h_separated = bool(keywords.get("h_started", True))
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


def pin_array_path(driver: Any) -> None:
    driver._fast_path = None  # noqa: SLF001
    driver._fast_path_stale = False  # noqa: SLF001


def install(driver: Any, shim: Optional["Shim"]) -> None:
    driver._fast_path = shim  # noqa: SLF001
    driver._fast_path_stale = False  # noqa: SLF001


# ---------------------------------------------------------------------------
# The dispatch shim and the launch witnesses  (the template's, with two kernels)
# ---------------------------------------------------------------------------

class ABSORBED:
    """A slot a fused launch already performed: answer True, run nothing."""


class CountingPlan:
    __slots__ = ("inner", "absorbed_by", "plan_runs")

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self.plan_runs = 0

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.plan_runs += 1
        self.inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def declaring(plan: Any) -> Any:
    inner = getattr(plan, "absorbed_by", None)
    if inner is not None and inner is not plan:
        return declaring(inner)
    return plan


class Shim:
    """A ``FastPathPlan``-shaped object that runs this gate's plans inside the driver."""

    __slots__ = ("fields", "plans", "dispatched", "declined", "absorbed",
                 "sync_refusals", "sync_answered", "sync_hazard", "counting_plans")

    def __init__(self, fields: Any, plans: Mapping[str, Any],
                 sync_hazard: bool = False) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.counting_plans: List["CountingPlan"] = []
        claimed: List[int] = []

        def wrap(plan: Any) -> Any:
            owner = declaring(plan)
            if id(owner) in claimed:
                return plan
            claimed.append(id(owner))
            wrapper = CountingPlan(plan)
            self.counting_plans.append(wrapper)
            return wrapper

        for slot, plan in list(self.plans.items()):
            if plan is ABSORBED:
                continue
            if isinstance(plan, (list, tuple)):
                self.plans[slot] = [wrap(entry) for entry in plan]
            else:
                self.plans[slot] = wrap(plan)
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.absorbed: Dict[str, int] = {}
        self.sync_refusals = 0
        self.sync_answered = 0
        self.sync_hazard = sync_hazard

    def span_of(self, slot: str) -> Tuple[str, ...]:
        plan = self.plans.get(slot)
        if plan is None or plan is ABSORBED:
            return (slot,)
        owner = declaring(plan)
        return tuple(getattr(owner, "replaces_sub_steps", None)
                     or getattr(owner, "replaces", None) or (slot,))

    def dispatch(self, slot: str, fields: Any) -> bool:
        owner = SYNC_PASS_OWNERS.get(slot)
        if owner is not None:
            if owner not in self.plans:
                self.declined[slot] = self.declined.get(slot, 0) + 1
                return False
            outside = tuple(name for name in self.span_of(owner)
                            if name not in SYNC_PATH_SLOTS)
            if outside and not self.sync_hazard:
                self.sync_refusals += 1
                return False
            self.sync_answered += 1
            slot = owner
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        if plan is ABSORBED:
            self.absorbed[slot] = self.absorbed.get(slot, 0) + 1
            return True
        if slot == "update_P" and isinstance(plan, (list, tuple)):
            drive_field = fields.drive_field
            for entry in plan:
                entry.run(drive_field)
        else:
            plan.run()
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    @property
    def launches(self) -> int:
        return sum(wrapper.plan_runs for wrapper in self.counting_plans)


class CountingKernel:
    """An independent launch counter around one Triton kernel object."""

    __slots__ = ("kernel", "calls")

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, grid: Any) -> Any:
        launcher = self.kernel[grid]

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return launcher(*args, **kwargs)

        return call

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)


def _slot_owner(owner: Any, path: str) -> Tuple[Any, str]:
    """The object holding a dotted kernel slot, and the slot's last name."""
    *parents, leaf = path.split(".")
    holder = owner
    for name in parents:
        if name.isdigit():
            items = tuple(holder) if holder is not None else ()
            holder = items[int(name)] if int(name) < len(items) else None
        else:
            holder = getattr(holder, name, None)
        if holder is None:
            return None, leaf
    return holder, leaf


def count_kernels(spec: FamilySpec, owner: Any) -> List[CountingKernel]:
    """Wrap EVERY kernel a plan launches, or decline by name.

    A slot path may be dotted (``"scan._kernel"``): the certified cylindrical plans
    launch their radial scan through a nested ``RadialScan``'s own slot, and a weld
    holds two such plans.
    """
    defaults = spec.default_kernels(owner)
    if not defaults:
        return []
    wrappers: List[CountingKernel] = []
    for path, default in defaults.items():
        holder, leaf = _slot_owner(owner, path)
        if holder is None or not hasattr(holder, leaf):
            return []
        resolved = getattr(holder, leaf, None) or default
        if resolved is None:
            return []
        wrapper = CountingKernel(resolved)
        setattr(holder, leaf, wrapper)
        wrappers.append(wrapper)
    return wrappers


# ---------------------------------------------------------------------------
# The engines, on ONE driver, in lockstep from one seed
# ---------------------------------------------------------------------------

def settle_rotation(plan: Any, fields: Any, originals: Mapping[str, Any]) -> int:
    settled = 0
    for name in getattr(plan, "rotated_names", ()):
        current = getattr(fields, name)
        original = originals[name]
        if current is not original:
            original[...] = current
            setattr(fields, name, original)
            plan.rotated[name] = current
            settled += 1
    return settled


def rotating_owners(shim: Optional[Shim]) -> Tuple[Any, ...]:
    if shim is None:
        return ()
    owners: List[Any] = []
    seen: List[int] = []
    for plan in shim.plans.values():
        if plan is ABSORBED:
            continue
        for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
            owner = declaring(entry)
            if id(owner) in seen or not getattr(owner, "rotated_names", ()):
                continue
            seen.append(id(owner))
            owners.append(owner)
    return tuple(owners)


def rotating_originals(fields: Any, owners: Sequence[Any]) -> Dict[str, Any]:
    return {name: getattr(fields, name)
            for owner in owners for name in owner.rotated_names}


class Arrangement:
    __slots__ = ("name", "shim", "counters", "uncounted", "rotating", "originals",
                 "selected", "reasons", "on_step_done", "wipe_private", "unreachable")

    def __init__(self, name: str, shim: Optional[Shim],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None,
                 wipe_private: bool = False, unreachable: Optional[str] = None) -> None:
        self.name = name
        self.shim = shim
        self.wipe_private = bool(wipe_private)
        self.counters: List[CountingKernel] = []
        self.uncounted: List[str] = []
        self.rotating = rotating_owners(shim)
        self.originals = ({} if shim is None
                          else rotating_originals(shim.fields, self.rotating))
        self.selected = dict(selected or {})
        self.reasons = dict(reasons or {})
        self.on_step_done: Optional[Callable[[], None]] = None
        #: Set when this arrangement could not be built under the run's policy; it
        #: is then not stepped and not compared, and the record says why.
        self.unreachable = unreachable

    def count_launches(self, spec: FamilySpec) -> None:
        if self.shim is None:
            return
        wrapped: List[int] = []
        for slot, plan in self.shim.plans.items():
            if plan is ABSORBED:
                continue
            for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
                owner = declaring(entry)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                counters = count_kernels(spec, owner)
                if counters:
                    self.counters.extend(counters)
                else:
                    self.uncounted.append(f"{slot}:{type(owner).__name__}")

    def settle(self, fields: Any) -> int:
        return sum(settle_rotation(plan, fields, self.originals)
                   for plan in self.rotating)

    def owners(self) -> List[Any]:
        out: List[Any] = []
        seen: List[int] = []
        for plan in (self.shim.plans.values() if self.shim else ()):
            if plan is ABSORBED:
                continue
            for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
                owner = declaring(entry)
                if id(owner) not in seen:
                    seen.append(id(owner))
                    out.append(owner)
        return out

    def launches(self) -> Dict[str, Any]:
        return {"plans": 0 if self.shim is None else self.shim.launches,
                "kernels": sum(counter.calls for counter in self.counters),
                "uncounted_plans": list(self.uncounted)}


def _composed(driver: Any, fuse: bool) -> Any:
    return triton_launch.plan_step(
        driver.fields, driver.pml,
        sources=tuple(getattr(driver, "_sources", ())), fuse=fuse)


def arrangement_singles(spec: FamilySpec, driver: Any, forced: bool = False) -> Arrangement:
    fields = driver.fields
    constitutive, curl = spec.singles(driver, forced=forced)
    if constitutive is None or curl is None:
        raise RuntimeError("a certified single was refused on a fixture this gate "
                           "expects it to admit")
    shim = Shim(fields, {"update_H": constitutive, "step_D": curl})
    return Arrangement("singles", shim,
                       selected={"update_H": spec.cell_arms[0], "step_D": spec.cell_arms[1]})


def arrangement_composition(spec: FamilySpec, driver: Any, fuse: bool) -> Arrangement:
    plan = _composed(driver, fuse)
    plans = {slot: plan.plans[slot] for slot in triton_launch.STEP_ORDER
             if slot in plan.plans}
    if plan.polarization_plans:
        plans["update_P"] = list(plan.polarization_plans)
    name = "composition_today" if fuse else "unfused"
    reasons = {key: list(value) for key, value in plan.reasons.items()}
    # THE COMPOSER SELECTED NOTHING FOR THIS SEAM: under a policy the family's arms
    # are not certified for, the composition IS the array path. Recorded as
    # unreachable rather than compared, so a trivial agreement of the array path with
    # itself is never read as a composition's.
    unreachable = None
    if not any(slot in plans for slot in spec.family.REPLACES):
        needle = spec.composition_refusal_needle()
        named = [r for value in reasons.values() for r in value if needle and needle in r]
        unreachable = (f"the composer selected no arm on {spec.family.REPLACES}: "
                       + (named[0] if named else "no reason names this family's policy"))
    return Arrangement(name, Shim(driver.fields, plans), selected=plan.selected,
                       reasons={key: value for key, value in reasons.items()
                                if "fused" in key or "cylindrical" in key},
                       unreachable=unreachable)


def arrangement_weld(spec: FamilySpec, driver: Any, kernel: Any = None,
                     curl_kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True, name: str = "weld",
                     forced: bool = False, scan_kernel: Any = None) -> Arrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed."""
    family = spec.family
    sources = tuple(getattr(driver, "_sources", ()))
    plan = spec.build_weld(driver, kernel=kernel, curl_kernel=curl_kernel, forced=forced,
                           scan_kernel=scan_kernel)
    occupant: Any = plan if launcher is None else launcher(plan)
    if hoist:
        occupant = withdraw_hoist.LeadingWithdrawPlan(
            occupant, driver.fields, sources, span=family.REPLACES)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        base = _composed(driver, fuse=False)
        for slot in triton_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = base.plans[slot]
        if base.polarization_plans:
            plans["update_P"] = list(base.polarization_plans)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return Arrangement(name, shim,
                       selected={"update_H": f"{family.FAMILY} (forced)",
                                 "step_D": f"{family.FAMILY} (forced)"})


def drive(driver: Any, arrangements: Mapping[str, Arrangement], steps: int,
          reference: str = "array",
          per_step_hook: Optional[Callable[[int, str, Any], Any]] = None,
          progress: Optional[Callable[[str], None]] = None,
          stop_on_divergence: bool = True,
          stop_when_banded: bool = False,
          pairs: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Step every REACHABLE arrangement in lockstep from one seed; compare per step."""
    active = {name: a for name, a in arrangements.items() if a.unreachable is None}
    seed = capture(driver)
    state: Dict[str, Dict[str, Any]] = {
        name: {"volumes": {key: value.copy() for key, value in seed["volumes"].items()},
               "dipoles": list(seed["dipoles"]), "step_count": seed["step_count"],
               # Every arrangement starts from the seed's start-of-H flag, not from
               # whatever the previous arrangement's step left on the shared Fields.
               "h_separated": seed["h_separated"]}
        for name in active}
    per_step: List[Dict[str, Any]] = []
    hooks: Dict[str, List[Any]] = {name: [] for name in active}
    first_divergence: Optional[int] = None
    first_banded: Optional[int] = None
    moved_words = 0
    words_compared = 0
    for step in range(1, steps + 1):
        for name, arrangement in active.items():
            restore(driver, state[name])
            if arrangement.wipe_private:
                for volume_name, array in stored_volumes(driver.fields).items():
                    if _is_private(volume_name):
                        array.fill(0)
            install(driver, arrangement.shim)
            driver.step()
            arrangement.settle(driver.fields)
            if per_step_hook is not None:
                hooks[name].append(per_step_hook(step, name, driver))
            state[name] = capture(driver)
        pin_array_path(driver)
        banded_now = subnormal_census(state[reference])
        if banded_now and first_banded is None:
            first_banded = step
        if stop_when_banded and banded_now:
            if progress is not None:
                progress(f"step {step}/{steps} STOP: the reference state entered the "
                         f"float32 denormal band")
            break
        row: Dict[str, Any] = {"step": step}
        if step == 1:
            moved_words = sum(differing(seed["volumes"][k], state[reference]["volumes"][k])
                              for k in seed["volumes"])
        for name in active:
            if name == reference:
                continue
            difference = compare_snapshots(state[reference], state[name])
            private = {key: value for key, value in compare_snapshots(
                state[reference], state[name], include_private=True).items()
                if _is_private(key)}
            row[name] = {"differing_volumes": difference,
                         "differing_words": int(sum(difference.values())),
                         "private_volumes_that_differ": private}
            if difference and not row.get("autopsy"):
                row["autopsy"] = {name: autopsy(state[reference], state[name], difference)}
        for left, right in pairs:
            if left not in active or right not in active:
                continue
            difference = compare_snapshots(state[left], state[right])
            row.setdefault("pairs", {})[f"{left}|{right}"] = {
                "differing_volumes": difference,
                "differing_words": int(sum(difference.values()))}
        words_compared += sum(
            int(np.prod(array.shape)) * (2 if array.dtype.name == "complex64" else 1)
            for array in state[reference]["volumes"].values()
        ) * max(len(active) - 1, 1)
        per_step.append(row)
        diverged = any(isinstance(row.get(name), dict)
                       and row[name]["differing_words"] for name in active
                       if name != reference)
        if diverged and first_divergence is None:
            first_divergence = step
        if progress is not None:
            progress(f"step {step}/{steps} "
                     + " ".join(f"{name}={row[name]['differing_words']}"
                                for name in active if name != reference))
        if diverged and stop_on_divergence:
            break
    banded = subnormal_census(state[reference])
    return {
        # The reference engine's FINAL state, for a caller that measures a floor on
        # the state the run reached rather than on the seed; popped before the record
        # is written.
        "_reference_final_state": state[reference],
        "first_banded_step": first_banded,
        "steps_compared": len(per_step),
        "steps_requested": steps,
        "bit_identical": first_divergence is None,
        "first_divergence_step": first_divergence,
        "words_compared": int(words_compared),
        "reference_moved_words_step_1": int(moved_words),
        "per_step": per_step,
        "hooks": {name: value for name, value in hooks.items()
                  if any(entry is not None for entry in value)},
        "reference_subnormal_words": banded,
        "launches": {name: arrangement.launches()
                     for name, arrangement in active.items()},
        "selected": {name: arrangement.selected for name, arrangement in arrangements.items()},
        "unreachable": {name: arrangement.unreachable
                        for name, arrangement in arrangements.items()
                        if arrangement.unreachable},
    }


def all_arrangements(spec: FamilySpec, driver: Any, kernel: Any = None,
                     curl_kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     count: bool = False, forced: bool = False) -> Dict[str, Arrangement]:
    built: Dict[str, Arrangement] = {
        "array": Arrangement("array", None),
        "singles": arrangement_singles(spec, driver, forced=forced),
        "composition_today": arrangement_composition(spec, driver, fuse=True),
        "unfused": arrangement_composition(spec, driver, fuse=False),
        "weld": arrangement_weld(spec, driver, kernel=kernel, curl_kernel=curl_kernel,
                                 launcher=launcher, forced=forced),
        "weld_seam_only": arrangement_weld(spec, driver, kernel=kernel,
                                           curl_kernel=curl_kernel, launcher=launcher,
                                           rest_unfused=False, name="weld_seam_only",
                                           forced=forced),
    }
    if count:
        for arrangement in built.values():
            arrangement.count_launches(spec)
    return built


def cylindrical_floors(driver: Any, spec: FamilySpec) -> Dict[str, Any]:
    """The CUDA cylindrical gate's two floors on THIS fixture, on the array path.

    ``radial_difference_is_live``: the radial difference ``stepping.step_D`` hands the
    Dz curl (``cylindrical_rderiv_difference`` over the current ``Hy``, MEEP's per-chunk
    sum) is nonzero somewhere -- an all-zero difference makes every scan defect
    invisible. ``axis_row_is_live``: one array-path ``step_D`` moves the axis row of the
    targets the m = 0 / |m| = 1 rules write. Both are measured on a copy of the state
    so nothing here advances the run.
    """
    fields = driver.fields
    snapshot = capture(driver)
    try:
        hy = fields.Hy
        radial = stepping.cylindrical_rderiv_difference(
            driver.xp, hy, spec.family.PREFIX_IR0,
            stepping._cylindrical_scan_starts(fields, driver.pml, "Dz"),  # noqa: SLF001
            hy.shape[0], backward=True, scratch=fields.scratch)
        host = _host(radial)
        host = host.view(np.float32) if host.dtype == np.complex64 else host.astype(np.float32)
        live_words = int(np.count_nonzero(host))
        before = {name: words(getattr(fields, name)[0]).copy() for name in ("Dy", "Dz")}
        stepping.step_D(fields, driver.pml)
        after = {name: words(getattr(fields, name)[0]) for name in ("Dy", "Dz")}
        moved = {name: int(np.count_nonzero(before[name] != after[name])) for name in before}
    finally:
        restore(driver, snapshot)
    return {
        "radial_rows": int(host.shape[0]),
        "radial_difference_nonzero_words": live_words,
        "radial_difference_is_live": live_words > 0,
        "axis_row_words_moved": moved,
        "axis_row_is_live": any(moved.values()),
    }


def run_product(spec: FamilySpec, driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                require_full_budget: bool = True, clean_floor: int = 1,
                forced: bool = False, floors_at: str = "seed") -> Dict[str, Any]:
    """The four-reference identity on ONE driver. The core measurement.

    ``floors_at`` says WHERE the two cylindrical floors are measured. On the synthetic
    fixture the seed is a random state, so ``seed`` is right and BOTH floors are
    required. A lifted corpus row starts from the engine's zeros and is driven by its
    own sources, so at the seed the radial difference is identically zero and no axis
    word moves (measured 2026-09-07: two rows bit-identical over 60 steps and failed on
    the floors alone); there the floors are measured on the array path's FINAL state,
    the difference is required live, and the axis row's liveness is RECORDED -- whether
    the corpus row's own field has reached the axis by the end of the budget is a
    fact about the row, not about the product.
    """
    started = time.time()
    if floors_at not in ("seed", "end"):
        raise ValueError(f"floors_at must be 'seed' or 'end', not {floors_at!r}")
    floors = cylindrical_floors(driver, spec) if floors_at == "seed" else None
    arrangements = all_arrangements(spec, driver, count=True, forced=forced)
    PAIRS = (("weld_seam_only", "singles"), ("weld", "unfused"),
             ("weld", "composition_today"))
    result = drive(driver, arrangements, steps, progress=progress,
                   stop_when_banded=False, stop_on_divergence=False, pairs=PAIRS)
    final_state = result.pop("_reference_final_state")
    if floors is None:
        seed_state = capture(driver)
        restore(driver, final_state)
        pin_array_path(driver)
        try:
            floors = cylindrical_floors(driver, spec)
        finally:
            restore(driver, seed_state)
            pin_array_path(driver)
    floors["measured_at"] = floors_at
    floors_hold = floors["radial_difference_is_live"] and (floors["axis_row_is_live"]
                                                           or floors_at == "end")
    active = [name for name, a in arrangements.items() if a.unreachable is None]
    references = [name for name in active if name != "array"]

    def agrees(name: str) -> bool:
        return all(not row.get(name, {}).get("differing_words")
                   for row in result["per_step"])

    per_reference = {name: agrees(name) for name in references}
    disagree_with_array = sorted(name for name in references if not per_reference[name])

    def pair_agrees(key: str) -> Optional[bool]:
        if not any(key in row.get("pairs", {}) for row in result["per_step"]):
            return None
        return all(not row.get("pairs", {}).get(key, {}).get("differing_words")
                   for row in result["per_step"])

    pair_agreement = {f"{a}|{b}": pair_agrees(f"{a}|{b}") for a, b in PAIRS}
    seam_claim = {
        "the_weld_at_its_seam_equals_the_two_certified_singles":
            pair_agreement["weld_seam_only|singles"],
        "the_weld_equals_the_unfused_composition": pair_agreement["weld|unfused"],
        "the_weld_equals_the_composition_installed_today":
            pair_agreement["weld|composition_today"],
    }
    reachable_claims = {key: value for key, value in seam_claim.items() if value is not None}
    result.update({
        "passed": bool(reachable_claims and all(reachable_claims.values())
                       and result["reference_moved_words_step_1"] > 0
                       and result["steps_compared"] >= clean_floor
                       and (not require_full_budget or result["steps_compared"] == steps)
                       and result["bit_identical"]
                       and floors_hold),
        "seam_claim": seam_claim,
        "seam_claims_reachable": sorted(reachable_claims),
        "cylindrical_floors": floors,
        "agreement_with_the_array_path": per_reference,
        "pairwise_agreement": pair_agreement,
        "references_that_disagree_with_the_array_path": disagree_with_array,
        "the_seam_alone_agrees_with_the_array_path": per_reference.get("weld_seam_only"),
        "weld_agrees_with_the_certified_singles": pair_agreement["weld_seam_only|singles"],
        "the_certified_singles_agree_with_the_array_path": bool(per_reference.get("singles")),
        "built_from_arrays": bool(forced),
        "seconds": round(time.time() - started, 2),
    })
    return result


# ---------------------------------------------------------------------------
# HOST LEGS
# ---------------------------------------------------------------------------

def leg_driver_order(spec: FamilySpec) -> Dict[str, Any]:
    family = spec.family
    findings: List[str] = []
    text = (API_ROOT / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    body = text.split("    def step(self", 1)[1].split("\n    def ", 1)[0]
    lines = [line.strip() for line in body.splitlines()]
    consults = [line for line in lines if 'fast.dispatch("' in line]
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0] for line in consults]
    try:
        first = order.index("update_H")
    except ValueError:
        findings.append("driver.step no longer consults update_H")
        first = -1
    if first >= 0 and order[first + 1:first + 2] != ["step_D"]:
        findings.append(f"the consult after update_H is {order[first + 1:first + 2]}, "
                        f"not ['step_D']")
    between = body.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    statements = [line for line in statements if "update_H(self.fields" not in line]
    # The magnetic f_minus_p restore, which writes B alone and which step_D does not
    # read (withdraw_hoist.SEAM_PASS); an integrated source stands no withdraw here.
    # statements[0] and [-1] are the tails of the two consult lines themselves.
    if tuple(statements[1:-1]) != h_to_d_seam.EXPECTED_BETWEEN[1:]:
        findings.append(f"between the two consults stands {statements[1:-1]}, not the magnetic "
                        f"f_minus_p restore alone")
    if any("withdraw" in line for line in statements):
        findings.append("a withdraw stands between the two consults")
    if any("inject" in line for line in statements):
        findings.append("an injection stands between the two consults")
    if family.REPLACES != withdraw_hoist.SEAM_SPAN:
        findings.append(f"REPLACES {family.REPLACES} is not withdraw_hoist.SEAM_SPAN")
    if family.SEAM != h_to_d_seam.SEAM:
        findings.append(f"the product's SEAM {family.SEAM!r} is not the boards' "
                        f"{h_to_d_seam.SEAM!r}")
    if "step_D" in SYNC_PATH_SLOTS:
        findings.append("step_D is inside SYNC_PATH_SLOTS")
    if SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS) != "update_H":
        findings.append("the sync channel no longer owns update_H by name")
    if family.CARRIES_DEPOSIT_REPAIR or family.HOISTS_THE_WITHDRAW or family.INSTALLABLE:
        findings.append("a declaration flag is True that this round's verdicts say is False")
    # THE RADIAL DIFFERENCE'S PLACE IN step_D, read off stepping.py: MEEP's per-chunk
    # sum of the POST-update_H magnetic snapshot's Hy at ir0 = 0.5, over the Dz chunk
    # starts, fed to the Dz term only -- what the certified plan's scan computes.
    stepping_text = (API_ROOT / "meep_gpu" / "stepping.py").read_text(encoding="utf-8")
    step_d = stepping_text.split("def step_D(", 1)[1].split("\ndef ", 1)[0]
    for needle, what in (
            ('radial = cylindrical_rderiv_difference(\n            grid.xp, '
             'magnetic["Hy"], 0.5,', "sums magnetic['Hy'] at ir0 = 0.5"),
            ('_cylindrical_scan_starts(fields, pml if pml_active else None, "Dz")',
             "keys the sum on Dz's chunk starts"),
            ('if cylindrical and term.target == "Dz":', "feeds the Dz term only")):
        if needle not in step_d:
            findings.append(f"stepping.step_D no longer {what}")
    if float(family.PREFIX_IR0) != 0.5 or family.PREFIX_COMPONENT != "Hy":
        findings.append("the product's radial-sum declaration disagrees with stepping.step_D")
    return {"passed": not findings, "findings": findings,
            "consult_order": order,
            "statements_between_the_two_consults": statements,
            "replaces": list(family.REPLACES),
            "radial_sum": {"component": family.PREFIX_COMPONENT,
                           "ir0": family.PREFIX_IR0},
            "launches_per_run": family.LAUNCHES_PER_RUN,
            "kernel_launches_per_run": family.KERNEL_LAUNCHES_PER_RUN,
            "installable": family.INSTALLABLE,
            "installable_reason": family.INSTALLABLE_REASON}


def leg_transcription(spec: FamilySpec) -> Dict[str, Any]:
    findings, measures = spec.transcription()
    source = Path(spec.family.__file__).read_text(encoding="utf-8")
    if not source.isascii():
        findings.append("the family module is not ASCII")
    unresolved = []
    for tag, mutation in spec.mutations.items():
        if mutation["target"] != "kernel":
            continue
        hits = source.count(mutation["old"])
        if hits != 1:
            unresolved.append(f"{tag}: {hits} matches")
    if spec.byte_neutral and source.count(spec.byte_neutral["old"]) != 1:
        unresolved.append(f"{spec.byte_neutral['tag']}: byte-neutral needle")
    if unresolved:
        findings.append("mutation needles that do not resolve exactly once: "
                        + "; ".join(unresolved))
    measures["mutation_needles"] = len([1 for s in spec.mutations.values()
                                        if s["target"] == "kernel"])
    return {"passed": not findings, "findings": findings, "measures": measures}


def purity_ledger(spec: FamilySpec, fields: Any, pml: Any) -> Dict[str, Any]:
    """How many ``Hy`` words ``update_H`` MOVES over the rows the radial scan reads.

    THE FLOOR FOR THE SCAN'S ONE ORDERING FACT. The certified scan in launch 2 sums the
    SCRATCH ``Hy`` launch 1 wrote; if ``update_H`` moved nothing there, a scan of the
    PRE-LAUNCH ``Hy`` would agree with it by accident and the scan-reads-the-pre-launch-
    Hy mutation would be a null about the fixture rather than the product.
    """
    histories = ("f_w_Hx", "f_w_Hy", "f_w_Hz")
    before = _host(fields.Hy).copy()
    before_w = {name: _host(getattr(fields, name)).copy() for name in histories}
    stepping.update_H(fields, pml)
    after = _host(fields.Hy)
    moved = words(before) != words(after)
    # The history moves only where the component's OWN axis carries a layer (the
    # auxiliary contract): r for Hx, z for Hz, and never phi, so f_w_Hy stays at rest.
    moved_w = {name: int(np.count_nonzero(words(before_w[name])
                                          != words(_host(getattr(fields, name)))))
               for name in histories}
    return {"cells": int(before.size),
            "scan_source_words": int(moved.size),
            "scan_source_words_update_H_moved": int(np.count_nonzero(moved)),
            "f_w_H_words_moved": moved_w,
            "f_w_H_words_moved_total": int(sum(moved_w.values())),
            "meets_floor": int(np.count_nonzero(moved)) > 0}


def leg_purity_ledger(spec: FamilySpec) -> Dict[str, Any]:
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for name, keywords in spec.cases:
        driver = build_driver(spec, dict(keywords), seed=11, prefer_gpu=False)
        try:
            rows[name] = purity_ledger(spec, driver.fields, driver.pml)
        finally:
            driver.close()
        if not rows[name]["meets_floor"]:
            findings.append(f"{name}: update_H moves no Hy word the scan reads")
        if rows[name]["f_w_H_words_moved_total"] == 0:
            findings.append(f"{name}: update_H moves no f_w_H word")
        if rows[name]["f_w_H_words_moved"]["f_w_Hy"]:
            findings.append(f"{name}: update_H moved f_w_Hy, whose own axis phi carries "
                            f"no layer; the auxiliary contract leaves it at rest")
    return {"passed": not findings, "findings": findings, "fixtures": rows,
            "what_it_measures": (
                "of the Hy words the radial scan reads (every stored row), how many one "
                "array-path update_H moves on the fixture's seeded state; the floor that "
                "makes the scan-reads-the-pre-launch-Hy mutation a claim about the "
                "product rather than about a quiet fixture")}


def leg_ghost_observability(spec: FamilySpec) -> Dict[str, Any]:
    """Read off the product's and the certified curl's own text, never restated: the
    scan and the curl are the certified plan's, the radial difference's backward tap is
    guarded, and the ownership mask zeroes the target that tap feeds at the axis row."""
    findings, measures = spec.ghost_observability()
    return {"passed": not findings, "findings": findings, **measures}


# ---------------------------------------------------------------------------
# DEVICE LEGS: refusal and arbitration
# ---------------------------------------------------------------------------

class GridView:
    def __init__(self, grid: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_grid"), name)


class FieldsView:
    def __init__(self, fields: Any, **overrides: Any) -> None:
        self._fields = fields
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._fields, name)


def stand_in_source(component: str = "Ez", integrated: bool = True,
                    field_type: str = "D", points: int = 3) -> Any:
    """A source class carrying an integrated withdraw of its own, which the predicate's
    withdraw clause refuses. No engine source carries one since an integrated source
    entered through ``f_minus_p`` (``withdraw_hoist``)."""

    class _Stand:
        def withdraw(self, fields):  # never called: a predicate reads a source
            raise AssertionError("a coverage predicate called a source's withdraw")

    stand = _Stand()
    stand.component = component
    stand.field_type = field_type
    stand.is_integrated = integrated
    stand._n_source_points = points
    stand._applied_dipole = 0j
    return stand


def leg_refusal(spec: FamilySpec, policy: Optional[str] = None) -> Dict[str, Any]:
    """Every configuration outside the cell is refused BY NAME, with its reason.

    Under a policy the family's arms are not certified for, a case that must be
    ADMITTED is instead refused by the certification clause -- and by NOTHING ELSE:
    every reason on it must carry the policy needle, so a structural refusal cannot
    hide behind the policy one. The named refusals are checked as under any policy.
    """
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    reachable, why = spec.policy_reaches_composer(policy)
    policy_needle = spec.composition_refusal_needle()
    driver = build_driver(spec, dict(spec.cases[0][1]), seed=1)
    try:
        for case in spec.refusal_cases(driver):
            verdict = spec.coverage(case["fields"], case["pml"], case["sources"])
            reasons = list(verdict.reasons)
            row = {"case": case["name"], "covered": verdict.covered, "reasons": reasons}
            if case["must_refuse"] and verdict.covered:
                findings.append(f"{case['name']}: admitted, and must be refused")
            if not case["must_refuse"]:
                if reachable and not verdict.covered:
                    findings.append(f"{case['name']}: refused, and must be admitted ({reasons})")
                if not reachable:
                    # Refused by the policy clause ALONE, or the case has a structural
                    # refusal the policy is masking.
                    structural = [r for r in reasons if policy_needle not in r]
                    row["refused_by_the_policy_alone"] = bool(reasons) and not structural
                    if verdict.covered:
                        findings.append(f"{case['name']}: admitted under a policy the "
                                        f"family's arms are not certified for")
                    elif structural:
                        findings.append(f"{case['name']}: refused by something other than "
                                        f"the policy clause under {policy!r}: {structural}")
            needle = case.get("needle")
            if needle and not any(needle in reason for reason in reasons):
                findings.append(f"{case['name']}: no refusal names {needle!r}; got {reasons}")
            if case.get("first_half") and reasons and not reasons[0].startswith(case["first_half"]):
                findings.append(f"{case['name']}: the FIRST refusal is {reasons[0]!r}; "
                                f"the driver reaches {case['first_half']!r} first")
            rows.append(row)
    finally:
        driver.close()
    return {"passed": not findings, "findings": findings, "cases": rows,
            "policy_reaches_composer": reachable,
            "why_the_policy_does_not_reach_the_composer": why or None}


def inject_product_row(spec: FamilySpec) -> Callable[[], None]:
    products = triton_launch.CERTIFIED_FUSED_PRODUCTS
    arms = triton_launch.CERTIFIED_FUSED_PAIR_ARMS
    name = spec.family.FAMILY
    row = spec.product_row()
    products[name] = dict(row)
    arms[name] = spec.family.ARMS
    original_modules = triton_launch._certified_fused_product_modules  # noqa: SLF001

    def modules():
        table = original_modules()
        table[row["module"]] = spec.family
        return table

    triton_launch._certified_fused_product_modules = modules  # noqa: SLF001

    def undo() -> None:
        products.pop(name, None)
        arms.pop(name, None)
        triton_launch._certified_fused_product_modules = original_modules  # noqa: SLF001

    return undo


def leg_arbitration(spec: FamilySpec, policy: Optional[str]) -> Dict[str, Any]:
    """Who holds ``update_H`` and ``step_D`` on a row this product admits, measured."""
    findings: List[str] = []
    family = spec.family
    driver = build_driver(spec, dict(spec.cases[0][1]), seed=3)
    try:
        fields, pml = driver.fields, driver.pml
        reachable, why = spec.policy_reaches_composer(policy)
        verdict = spec.coverage(fields, pml, ())
        shipped = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
        shipped_selected = dict(shipped.selected)
        out: Dict[str, Any] = {"predicate_admits_the_fixture": verdict.covered,
                               "predicate_reasons": list(verdict.reasons),
                               "policy_reaches_composer": reachable,
                               "shipped_selected": shipped_selected}
        if not reachable:
            # UNDER THIS POLICY THE FAMILY'S ARMS ARE NOT CERTIFIED and the composer
            # must select nothing on the seam; the predicate must refuse by that name.
            needle = spec.composition_refusal_needle()
            if verdict.covered:
                findings.append("the predicate admits the fixture under a policy the "
                                "family's arms are not certified for")
            elif needle and not any(needle in r for r in verdict.reasons):
                findings.append(f"no refusal names the certification clause {needle!r}")
            if any(slot in shipped_selected for slot in family.REPLACES):
                findings.append(f"the composer selected {shipped_selected} under a policy "
                                f"the arms are not certified for")
            out["why_unreachable"] = why
            out["composer_refusals"] = {key: list(value) for key, value in shipped.reasons.items()
                                        if "cylindrical" in key}
            return {"passed": not findings, "findings": findings, **out}
        if not verdict.covered:
            findings.append(f"the arbitration fixture is not one this product admits: "
                            f"{verdict.reasons}")
        holder_b, holder_d = spec.arbitration_incumbents()
        if shipped_selected.get("update_H") != holder_b or shipped_selected.get("step_B") != holder_b:
            findings.append(f"the shipped composer holds step_B/update_H as "
                            f"{shipped_selected.get('step_B')!r}/{shipped_selected.get('update_H')!r}, "
                            f"not {holder_b!r}")
        if shipped_selected.get("step_D") != holder_d or shipped_selected.get("update_E") != holder_d:
            findings.append(f"the shipped composer holds step_D/update_E as "
                            f"{shipped_selected.get('step_D')!r}/{shipped_selected.get('update_E')!r}, "
                            f"not {holder_d!r}")
        routed = family.FAMILY in triton_launch.CERTIFIED_FUSED_PRODUCTS
        if routed:
            # THE SHIPPED TABLES CARRY THIS PRODUCT (routed 2026-09-07): the composer
            # consulted the shipped row in the plan above, so the arbitration is
            # measured against THAT row and nothing is injected. The row the composer
            # holds is required to be the family's own declaration, and the arm pair
            # the family's ARMS, so a table edited away from the module fails here
            # rather than being measured under a different name.
            row_source = "shipped"
            shipped_row = dict(triton_launch.CERTIFIED_FUSED_PRODUCTS[family.FAMILY])
            if shipped_row != dict(spec.product_row()):
                findings.append(f"the shipped CERTIFIED_FUSED_PRODUCTS row {shipped_row} "
                                f"is not the family's own {spec.product_row()}")
            if triton_launch.CERTIFIED_FUSED_PAIR_ARMS.get(family.FAMILY) != tuple(family.ARMS):
                findings.append(f"the shipped CERTIFIED_FUSED_PAIR_ARMS row "
                                f"{triton_launch.CERTIFIED_FUSED_PAIR_ARMS.get(family.FAMILY)!r} "
                                f"is not the family's ARMS {tuple(family.ARMS)!r}")
            injected_selected = dict(shipped_selected)
            refusals = [reason for key, value in shipped.reasons.items()
                        if family.FAMILY in key for reason in value]
            absorb_refusal = triton_launch._pair_may_absorb(  # noqa: SLF001
                injected_selected, "update_H", "step_D", family.ARMS)
            uninstallable = triton_launch._declared_uninstallable(  # noqa: SLF001
                {shipped_row["module"]: family}, family.FAMILY, shipped_row)
        else:
            row_source = "injected"
            undo = inject_product_row(spec)
            try:
                injected = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
                injected_selected = dict(injected.selected)
                refusals = [reason for key, value in injected.reasons.items()
                            if family.FAMILY in key for reason in value]
                absorb_refusal = triton_launch._pair_may_absorb(  # noqa: SLF001
                    injected_selected, "update_H", "step_D", family.ARMS)
                uninstallable = triton_launch._declared_uninstallable(  # noqa: SLF001
                    {spec.product_row()["module"]: family}, family.FAMILY,
                    triton_launch.CERTIFIED_FUSED_PRODUCTS[family.FAMILY])
            finally:
                undo()
        if injected_selected.get("update_H") != holder_b:
            findings.append(f"with the row injected update_H moved to "
                            f"{injected_selected.get('update_H')!r}")
        if not refusals:
            findings.append("the composer records no refusal naming this product")
        if absorb_refusal is None:
            findings.append("_pair_may_absorb does NOT refuse this product against the "
                            "live selected table")
        if uninstallable is None or "INSTALLABLE = False" not in uninstallable:
            findings.append("_declared_uninstallable does not refuse this product by the flag")
        seams = triton_launch.CERTIFIED_FUSED_PAIR_SEAMS
        if seams.get("update_H") != ("step_D", withdraw_hoist.SEAM):
            findings.append(f"the composer's H->D seam row is {seams.get('update_H')!r}")
        if list(seams)[-1] != "update_H":
            findings.append("the H->D seam row is not the last row of the seam table")
        pairs_installed = sum(
            1 for pair in ({shipped_selected.get(a) for a in ("step_B", "update_H")},
                           {shipped_selected.get(a) for a in ("step_D", "update_E")})
            if len(pair) == 1 and None not in pair and "fused pair" in str(next(iter(pair))))
        out.update({
            "row_source": row_source,
            "selected_with_the_row_injected": injected_selected,
            "composer_refusals": refusals,
            "pair_may_absorb_refusal": absorb_refusal,
            "declared_uninstallable_refusal": uninstallable,
            "neighbouring_pairs_installed_today": pairs_installed,
            "composer_launch_algebra": {
                "rule": "launches = 4 - (installed pairs) over step_B..update_E",
                "today": 4 - pairs_installed,
                "with_this_product": 4 - 1,
                "verdict_by_the_rule": ("LOSS" if pairs_installed == 2 else
                                        "TIE" if pairs_installed == 1 else "GAIN")},
            "cylindrical_device_launch_algebra": {
                "what_it_counts": "device launches on the D-side seam per step per row, "
                                  "measured by launch_structure rather than assumed here",
                "d_side_seam_today": 1 + CYLINDRICAL_PLAN_KERNEL_LAUNCHES,
                "d_side_seam_with_this_product": family.LAUNCHES_PER_RUN,
                "each_neighbouring_pair_saves": 1,
                "net_saving_displacing_both_neighbours":
                    (1 + CYLINDRICAL_PLAN_KERNEL_LAUNCHES) - family.LAUNCHES_PER_RUN
                    - pairs_installed},
        })
    finally:
        driver.close()
    return {"passed": not findings, "findings": findings, **out}


# ---------------------------------------------------------------------------
# DEVICE LEGS: the product
# ---------------------------------------------------------------------------

def leg_product(spec: FamilySpec, steps: int, seed: int, forced: bool,
                progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, keywords in spec.cases:
        driver = build_driver(spec, dict(keywords), seed=seed)
        try:
            result = run_product(spec, driver, steps, forced=forced, progress=(
                (lambda message, case=name: progress(f"{case} {message}")) if progress else None))
        finally:
            driver.close()
        result.update({"leg": "product", "case": name, "keywords": dict(keywords)})
        rows.append(result)
    return rows


def leg_seed_scale(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for bits in (0, 40, SEED_SCALE_BITS):
        driver = build_driver(spec, dict(spec.cases[0][1]), seed=seed, scale_bits=bits)
        try:
            result = run_product(spec, driver, steps, require_full_budget=True, forced=forced)
        finally:
            driver.close()
        rows[f"2^{bits}"] = {"bit_identical": result["bit_identical"],
                             "first_divergence_step": result["first_divergence_step"],
                             "steps_compared": result["steps_compared"],
                             "reference_subnormal_words": result["reference_subnormal_words"]}
    if not rows[f"2^{SEED_SCALE_BITS}"]["bit_identical"]:
        findings.append(f"the shipped scale 2^{SEED_SCALE_BITS} is not identical")
    return {"passed": not findings, "findings": findings, "scales": rows}


def leg_launch_structure(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    """Launches per step at the seam AND over the whole step, two witnesses.

    The kernel counter wraps every Triton kernel a plan can launch (the weld's launch 1
    and both certified step_D plans' scan and curl; a certified cylindrical plan's scan
    and curl); the plan counter counts ``run()`` calls. Every arrangement's kernel count
    must equal its plans' declared launches per run times their runs
    (:meth:`FamilySpec.kernel_launches_of`), and the weld's per-step count at the seam is
    reported beside the singles' -- three against three.
    """
    findings: List[str] = []
    family = spec.family
    driver = build_driver(spec, dict(dict(spec.cases)[spec.mutation_case]), seed=seed)
    try:
        arrangements = all_arrangements(spec, driver, count=True, forced=forced)
        result = drive(driver, arrangements, min(steps, 8), stop_on_divergence=False)
        result.pop("_reference_final_state", None)
        launches = result["launches"]
        expected_kernels: Dict[str, int] = {}
        seam_kernels: Dict[str, int] = {}
        for name, arrangement in arrangements.items():
            if arrangement.unreachable or arrangement.shim is None:
                continue
            expected = 0
            seam = 0
            seam_owners = {id(declaring(arrangement.shim.plans[slot]))
                           for slot in family.REPLACES
                           if arrangement.shim.plans.get(slot) not in (None, ABSORBED)}
            for wrapper in arrangement.shim.counting_plans:
                owner = declaring(wrapper.inner)
                count = wrapper.plan_runs * spec.kernel_launches_of(owner)
                expected += count
                if id(owner) in seam_owners:
                    seam += count
            expected_kernels[name] = expected
            seam_kernels[name] = seam
    finally:
        driver.close()
    steps_run = result["steps_compared"]
    for name, counts in launches.items():
        if name == "array":
            continue
        if counts["uncounted_plans"]:
            findings.append(f"{name}: the kernel counter cannot see {counts['uncounted_plans']}")
        elif counts["kernels"] != expected_kernels.get(name):
            findings.append(f"{name}: the kernel counter saw {counts['kernels']} launches and "
                            f"the plans' declared launches per run predict "
                            f"{expected_kernels.get(name)}")
    if steps_run and seam_kernels.get("weld") != family.LAUNCHES_PER_RUN * steps_run:
        findings.append(f"the weld launched {seam_kernels.get('weld')} kernels at the seam "
                        f"over {steps_run} steps, not LAUNCHES_PER_RUN = "
                        f"{family.LAUNCHES_PER_RUN} per step")
    per_step = {name: (None if not steps_run else {
        "kernels_per_step": counts["kernels"] / steps_run,
        "seam_kernels_per_step": seam_kernels.get(name, 0) / steps_run})
        for name, counts in launches.items()}
    return {
        "passed": not findings, "findings": findings, "steps": steps_run,
        "launches_over_the_run": launches,
        "expected_kernel_launches_from_declarations": expected_kernels,
        "seam_kernel_launches": seam_kernels,
        "device_launches_per_step": per_step,
        "d_side_seam_per_step": {
            "singles": 1 + CYLINDRICAL_PLAN_KERNEL_LAUNCHES,
            "weld": family.LAUNCHES_PER_RUN,
            "what_is_measured": "kernel launches (independent wrappers on every kernel "
                                "slot, the certified plans' nested scan slot included) "
                                "and plan runs; nothing is read off source"},
        "the_honest_reading": (
            "under MEEP's per-chunk radial sum the weld performs the D-side seam in the "
            "same number of kernel launches as the two certified singles it replaces "
            "(update_H, the scan, the curl); it is kept for its form alignment, not for "
            "a launch saving, and the composer's own rule refuses it (arbitration leg)"),
    }


def leg_sync(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    findings: List[str] = []
    out: Dict[str, Any] = {}

    def hook(step: int, name: str, driver: Any) -> Any:
        if step not in SYNC_STEPS:
            return None
        driver.synchronize_magnetic_fields()
        driver.restore_magnetic_fields()
        return {"step": step, "synchronized": True}

    for armed in (False, True):
        driver = build_driver(spec, dict(dict(spec.cases)[spec.mutation_case]), seed=seed)
        try:
            arrangements = {
                "array": Arrangement("array", None),
                "weld": arrangement_weld(spec, driver, sync_hazard=armed, forced=forced,
                                         name="weld_sync_armed" if armed else "weld"),
            }
            result = drive(driver, arrangements, min(steps, 10), per_step_hook=hook,
                           stop_on_divergence=False)
            result.pop("_reference_final_state", None)
            shim = arrangements["weld"].shim
            out["armed" if armed else "declining"] = {
                "bit_identical": result["bit_identical"],
                "first_divergence_step": result["first_divergence_step"],
                "sync_refusals": shim.sync_refusals, "sync_answered": shim.sync_answered,
                "differing_volumes_at_the_end": (
                    result["per_step"][-1].get("weld_sync_armed")
                    or result["per_step"][-1].get("weld", {})).get("differing_volumes", {}),
                "steps_compared": result["steps_compared"]}
        finally:
            driver.close()
    declining, armed_row = out["declining"], out["armed"]
    if not declining["bit_identical"]:
        findings.append("the DECLINING arm is not identical to the array path")
    if declining["sync_refusals"] == 0:
        findings.append("the declining arm never refused the sync consult; the leg is disarmed")
    if armed_row["bit_identical"]:
        findings.append("the ARMED arm did NOT diverge")
    else:
        electric = {name for name in armed_row["differing_volumes_at_the_end"]
                    if name.startswith(("D", "fu_D", "f_cond_D"))}
        if not electric:
            findings.append("the armed arm diverged but not in D/fu_D/f_cond_D")
        armed_row["electric_volumes_that_diverged"] = sorted(electric)
    return {"passed": not findings, "findings": findings, "arms": out, "sync_steps": list(SYNC_STEPS)}


def volume_source(keywords: Mapping[str, Any], component: str = "Ez",
                  integrated: bool = True, resolution: float = 10.0) -> Dict[str, Any]:
    """One integrated electric source in the driver's own schema, inside the cell."""
    nr, nz = keywords["shape"]
    return {"source_type": "continuous", "component": component, "frequency": 0.6,
            "start_time": 0.0, "amplitude": 1.0,
            "center": (0.4 * nr / resolution, 0.0, 0.5 * nz / resolution),
            "is_integrated": integrated}


def leg_withdraw(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    """The seam's one pass on a device: :func:`hd_seam_restore.seam_restore_leg`.

    Since entry 25 what stands between the two consults is the magnetic ``f_minus_p``
    restore, not an electric withdraw. The Dcyl driver carries an integrated electric
    (``Ez``) AND an integrated magnetic (``Hz``) source inside the cell and clear of the
    absorbers; the weld arrangement is force-installed, stepped in lockstep with its own
    array-path reference (its rotation settled after every step, as :func:`drive`
    settles it) and required identical, and the weld with the restore moved before the
    launch and with no magnetic ``f_minus_p`` must each diverge. THE SEED IS NOT SCALED
    HERE (2^0): the ``f_minus_p`` words are the sources' amplitude.

    ``forced`` is the run's own: under a policy the family's arms are not certified for,
    the predicate must refuse by that clause and no other (never by a withdraw) and the
    weld is built from arrays, as every other leg here builds it then.
    """
    keywords = dict(dict(spec.cases)[spec.mutation_case])
    nr, nz = keywords["shape"]
    resolution = spec.fixture_resolution
    sources = (volume_source(keywords, resolution=resolution),
               {**volume_source(keywords, component="Hz", resolution=resolution),
                "frequency": 1.1,
                "center": (0.6 * nr / resolution, 0.0, 0.35 * nz / resolution)})
    built: List[Any] = []
    welds: Dict[int, Arrangement] = {}

    def build() -> Any:
        driver = build_driver(spec, keywords, seed=seed, sources=sources, scale_bits=0)
        built.append(driver)
        return driver

    def install_weld(driver: Any) -> Callable[[], int]:
        arrangement = arrangement_weld(spec, driver, forced=forced)
        welds[id(driver)] = arrangement
        install(driver, arrangement.shim)
        # The weld's own launches: the update_H slot it holds (the shim counts every
        # slot's plan, and the other slots carry the unfused singles).
        return lambda: int(arrangement.shim.dispatched.get("update_H", 0))

    def coverage(driver: Any) -> Tuple[bool, str]:
        verdict = spec.coverage(driver.fields, driver.pml,
                                tuple(getattr(driver, "_sources", ())))
        reasons = list(verdict.reasons)
        if not forced:
            return bool(verdict.covered), "; ".join(reasons)
        by_the_policy = bool(not verdict.covered and reasons and not any(
            "withdraw" in reason for reason in reasons))
        return by_the_policy, ("refused by the policy's certification clause alone: "
                               + "; ".join(reasons))

    try:
        record = hd_seam_restore.seam_restore_leg(
            build=build, install_weld=install_weld, coverage=coverage,
            state_of=lambda driver: capture(driver)["volumes"],
            compare=lambda left, right: compare_snapshots({"volumes": left},
                                                          {"volumes": right}),
            synchronize=lambda: None, steps=steps, spec_label=spec.mutation_case,
            settle=lambda driver: welds[id(driver)].settle(driver.fields))
    finally:
        for driver in built:
            with contextlib.suppress(Exception):
                driver.close()
    record["weld_built_from_arrays"] = bool(forced)
    return record


def mutated_family(spec: FamilySpec, tag: str, old: str, new: str) -> Any:
    """Import a COPY of the family module with one source edit applied."""
    source_path = Path(spec.family.__file__)
    text = source_path.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits != 1:
        raise AssertionError(f"{tag}: the mutation needle matches {hits} times, not once")
    mutated_text = text.replace(old, new, 1)
    if mutated_text == text:
        raise AssertionError(f"{tag}: the mutation changed nothing")
    name = f"{spec.family.__name__}__mut_{tag}"
    filename = str(source_path) + f"#{tag}"
    linecache.cache[filename] = (len(mutated_text), None, mutated_text.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "meep_gpu.triton_kernels"
    sys.modules[name] = module
    exec(compile(mutated_text, filename, "exec"), module.__dict__)  # noqa: S102
    return module


class RotationSkipped:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:
        writes, reads = self.inner._resolve()  # noqa: SLF001
        self.inner._launch(writes, reads, guard)  # noqa: SLF001
        self.inner.launches += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class FusionOn:
    """The plan launched with floating-point contraction ENABLED -- a host mutation."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:  # noqa: ARG002
        self.inner.run(guard=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class CurlReadsPreLaunchH:
    """Launches 2 and 3 run over the PRE-LAUNCH H instead of this run's scratch.

    THE ORDER DEFECT, driven through the shipped ``_launch``: launch 1 still writes the
    scratch, and the certified step_D plan the weld picks is swapped for the one built
    over the OTHER allocation -- the pre-launch H -- for exactly one run, so the scan
    sums and the curl differences the state the seam's first half has not yet produced.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self.swaps = 0

    def run(self, guard: Optional[bool] = None) -> None:
        plan = self.inner
        writes, reads = plan._resolve()  # noqa: SLF001
        wrong = plan._curl_plan_for(reads)  # noqa: SLF001
        right = plan._curl_plan_for(writes)  # noqa: SLF001
        if wrong is right:
            raise AssertionError("the pre-launch and scratch H resolve to one plan")
        plans = plan._curl_plans  # noqa: SLF001
        key = next(key for key, value in plans.items() if value is right)
        plans[key] = wrong
        try:
            plan.run(guard)
        finally:
            plans[key] = right
        self.swaps += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class ScanReadsPreLaunchHy:
    """Launch 2's scan sums the PRE-LAUNCH Hy; launch 3's curl still reads the scratch H.

    The certified step_D plan over this run's scratch takes, for one run, the OTHER
    plan's ``RadialScan`` -- the one bound to the pre-launch Hy -- so the difference the
    curl reads is the previous step's field's while every other operand is this step's.
    The purity ledger is the floor that makes this a claim about the product.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:
        plan = self.inner
        writes, reads = plan._resolve()  # noqa: SLF001
        right = plan._curl_plan_for(writes)  # noqa: SLF001
        wrong = plan._curl_plan_for(reads)  # noqa: SLF001
        saved = right.scan
        right.scan = wrong.scan
        try:
            plan.run(guard)
        finally:
            right.scan = saved

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class ScanSkipped:
    """Launch 2 not performed: the curl reads whatever the scan's output last held.

    The certified plan's own ``RadialScan`` is replaced, for one run, by a stand-in
    whose ``run`` does nothing, so the difference the curl reads is the previous
    step's (or the zero-initialised buffer on step 1).
    """

    class _NoScan:
        def __init__(self, scan: Any) -> None:
            self._scan = scan
            self.out = scan.out

        def run(self, guard: Optional[bool] = None) -> None:  # noqa: ARG002
            return None

        def __getattr__(self, name: str) -> Any:
            return getattr(self._scan, name)

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:
        plan = self.inner
        writes, _reads = plan._resolve()  # noqa: SLF001
        target = plan._curl_plan_for(writes)  # noqa: SLF001
        saved = target.scan
        target.scan = self._NoScan(saved)
        try:
            plan.run(guard)
        finally:
            target.scan = saved

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class Ir0Ladder:
    """The scan's two invariant row vectors rebuilt at ANOTHER ``ir0`` -- the B side's
    0.0 ladder bound where the D side's 0.5 belongs. Built by the array path's own
    ``_cylindrical_rderiv_rows`` so the only thing that moves is ``ir0``."""

    def __init__(self, inner: Any, ir0: float) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self.ir0 = float(ir0)

    def run(self, guard: Optional[bool] = None) -> None:
        plan = self.inner
        writes, _reads = plan._resolve()  # noqa: SLF001
        scan = plan._curl_plan_for(writes).scan  # noqa: SLF001
        saved = (scan._w, scan._rinv)  # noqa: SLF001
        xp = _module_of(saved[0])
        weights, rinv = stepping._cylindrical_rderiv_rows(  # noqa: SLF001
            np, int(scan.rows_src), self.ir0, np.dtype(np.float32))
        scan._w = xp.asarray(np.ascontiguousarray(weights.reshape(-1)))  # noqa: SLF001
        scan._rinv = xp.asarray(np.ascontiguousarray(rinv.reshape(-1)))  # noqa: SLF001
        try:
            plan.run(guard)
        finally:
            scan._w, scan._rinv = saved  # noqa: SLF001

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def leg_byte_neutral(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    findings: List[str] = []
    edit = spec.byte_neutral
    driver = build_driver(spec, dict(dict(spec.cases)[spec.mutation_case]), seed=seed)
    try:
        module = mutated_family(spec, edit["tag"], edit["old"], edit["new"])
        kernel = spec.kernel_of(module)
        arrangements = {"array": Arrangement("array", None),
                        "weld": arrangement_weld(spec, driver, kernel=kernel, forced=forced)}
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
        result.pop("_reference_final_state", None)
    finally:
        driver.close()
    if not result["bit_identical"]:
        findings.append(f"the byte-neutral control DIVERGED at step {result['first_divergence_step']}")
    return {"passed": not findings, "findings": findings, "case": spec.mutation_case,
            "tag": edit["tag"], "why": edit["why"],
            "bit_identical": result["bit_identical"], "steps_compared": result["steps_compared"]}


def leg_disarm(spec: FamilySpec, steps: int, seed: int, forced: bool) -> Dict[str, Any]:
    findings: List[str] = []
    driver = build_driver(spec, dict(dict(spec.cases)[spec.mutation_case]), seed=seed)
    try:
        arrangements = {"array": Arrangement("array", None),
                        "weld": arrangement_weld(spec, driver, kernel=None, forced=forced)}
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
        result.pop("_reference_final_state", None)
    finally:
        driver.close()
    if not result["bit_identical"]:
        findings.append(f"the DISARMED harness diverged at step {result['first_divergence_step']}")
    return {"passed": not findings, "findings": findings, "case": spec.mutation_case,
            "bit_identical": result["bit_identical"], "steps_compared": result["steps_compared"]}


def leg_mutation(spec: FamilySpec, steps: int, seed: int, policy: Optional[str],
                 forced: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag, mutation in spec.mutations.items():
        expected = mutation.get("expected_override", mutation["expected"])
        if isinstance(expected, dict):
            expected = expected.get(policy or "keep", expected.get("keep"))
        case = mutation.get("case", spec.mutation_case)
        driver = build_driver(spec, dict(dict(spec.cases)[case]), seed=seed)
        row: Dict[str, Any] = {"leg": "mutation", "mutation": tag, "case": case,
                               "target": mutation["target"], "expected": expected,
                               "why": mutation["why"]}
        try:
            kernel = None
            launcher = None
            if mutation["target"] == "kernel":
                module = mutated_family(spec, tag, mutation["old"], mutation["new"])
                kernel = spec.kernel_of(module)
            elif mutation["target"] == "host":
                host = mutation["host"]
                if host == "rotation_skipped":
                    launcher = RotationSkipped
                elif host == "fusion_on":
                    launcher = FusionOn
                elif host == "half_integer":
                    def launcher(plan: Any, pml=driver.pml) -> Any:
                        return spec.half_integer_rebind(plan, pml)
                elif host == "curl_reads_pre_launch_H":
                    launcher = CurlReadsPreLaunchH
                elif host == "scan_reads_pre_launch_Hy":
                    launcher = ScanReadsPreLaunchHy
                elif host == "scan_skipped":
                    launcher = ScanSkipped
                elif host == "ir0_zero_ladder":
                    def launcher(plan: Any) -> Any:
                        return Ir0Ladder(plan, 0.0)
                elif host == "withdraw_after_step_D":
                    row.update({"outcome": "MEASURED IN THE withdraw LEG", "passed": True})
                    rows.append(row)
                    continue
                else:
                    launcher = spec.host_launcher(host, driver)
                    if launcher is None:
                        raise AssertionError(f"{tag}: unknown host mutation {host!r}")
            arrangements = {"array": Arrangement("array", None),
                            "weld": arrangement_weld(spec, driver, kernel=kernel,
                                                     launcher=launcher, forced=forced)}
            result = drive(driver, arrangements, steps, stop_on_divergence=True)
            result.pop("_reference_final_state", None)
            outcome = "NULL" if result["bit_identical"] else "CAUGHT"
            row.update({"outcome": outcome,
                        "first_divergence_step": result["first_divergence_step"],
                        "steps_compared": result["steps_compared"],
                        "passed": outcome == expected})
        except SubnormalPolicyUnattainable as refused:
            # The mutated spelling cannot be compiled uniformly under the policy in
            # force. For an ARMED mutation that is a catch before the first launch; for
            # a predicted null it is what the expectation table must say by policy.
            row.update({"outcome": REFUSED_BY_THE_POLICY, "refusal": str(refused)[:400],
                        "passed": expected in ("CAUGHT", REFUSED_BY_THE_POLICY)})
        except Exception as error:  # noqa: BLE001
            row.update({"outcome": "REFUSED TO BUILD", "passed": False,
                        "error": f"{type(error).__name__}: {error}"[:600]})
        finally:
            driver.close()
        rows.append(row)
        log(f"    mutation {tag}: {row.get('outcome')} (expected {expected})")
    return rows


# ---------------------------------------------------------------------------
# The corpus lift
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(spec: FamilySpec, results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    facts_by_label: Dict[str, dict] = {}
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            facts_by_label[f"{record.get('leg', leg)}:{record['row']}"] = record
    rows: List[dict] = []
    for label, seam_row in seam_rows.items():
        arms = (seam_row.get("arms") or {}).get("triton") or {}
        if (arms.get("update_H"), arms.get("step_D")) != tuple(spec.cell_arms):
            continue
        record = facts_by_label.get(label, {})
        rows.append({
            "label": label, "leg": seam_row.get("leg"), "row": seam_row["row"],
            # THE CASE THE HARNESS ENUMERATES. A parameterised test row is named by
            # the census's matcher (`_0_0_0`) but enumerated by the shim (`__idx0`);
            # the matched census record carries the shim's name, and the child must be
            # asked for THAT one or it finds no Simulation and the row is silently
            # filed unliftable (measured 2026-09-07: 3 of 3 real rows, 0 lifted).
            "harness_case": (record.get("shim_row") or record.get("case")
                             or seam_row["row"]),
            "module": record.get("module"),
            "grid_cells": record.get("grid_cells"), "grid_shape": record.get("grid_shape"),
            "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {}).get("withdraw_in_seam")),
            "arms": {"update_H": arms.get("update_H"), "step_D": arms.get("step_D")},
        })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(spec.cell_arms),
             "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CYLHD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CYLHD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def lift_child(spec: FamilySpec, leg: str, script: Optional[str], module_path: Optional[str],
               case: Optional[str], out_json: str, steps: int, max_cells: Optional[int],
               policy: Optional[str] = None) -> int:
    """One corpus row: capture the simulation, lift it TO THE DEVICE, drive it, with the
    subnormal policy installed IN THIS PROCESS before the first device compile."""
    record: Dict[str, Any] = {"leg": leg, "row": case or Path(script or "").name}
    started = time.time()
    try:
        import cupy  # noqa: PLC0415
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

        if leg == "examples":
            from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415

            captured, sim, restore_sim = capture_simulation(script)
            record.update({key: value for key, value in captured.items() if key != "row"})
            restore_sim()
        else:
            from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

            namespace = harness.build_child_namespace()
            module = namespace["_import_module"](module_path)
            sim = None
            for class_name, method_name in namespace["_enumerate_cases"](module):
                captured, sim_candidate, restore_case, _case = namespace["_run_case"](
                    module, module_path, class_name, method_name, 900.0)
                restore_case()
                if f"{class_name}.{method_name}" == case:
                    record.update({key: value for key, value in captured.items()
                                   if key != "row"})
                    sim = sim_candidate
                    break
        if sim is None:
            record.update(measured=False, unliftable_on_this_host=True,
                          note="no mp.Simulation to lift on this host's MEEP build")
            _write_child(out_json, record)
            return 0
        if policy is not None:
            subnormal_policy.install_subnormal_policy(policy, cupy=cupy, strict=True)
        record["host_flushing_at_lift"] = bool(backends.subnormals_flushed())
        _child_progress("lifting to the device")
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
        try:
            record[f"{spec.gate}_gate"] = evaluate_row(spec, driver, steps, max_cells, policy)
            record["measured"] = True
        finally:
            with contextlib.suppress(Exception):
                driver.close()
        record["subnormal_policy"] = subnormal_policy.policy_stamp()
    except BaseException as error:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        record.update(measured=False, child_error=f"{type(error).__name__}: {error}"[:800],
                      child_traceback=traceback.format_exc()[-2500:])
    _write_child(out_json, record)
    return 0


def _write_child(out_json: str, record: Dict[str, Any]) -> None:
    gate_provenance.stamp(record)
    Path(out_json).write_text(json.dumps(record, default=str), encoding="utf-8")
    _child_progress(f"measured={record.get('measured')}")


def evaluate_row(spec: FamilySpec, driver: Any, steps: int, max_cells: Optional[int],
                 policy: Optional[str]) -> Dict[str, Any]:
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": spec.family.FAMILY, "steps_requested": steps, "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = spec.coverage(driver.fields, driver.pml, sources)
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    composed = _composed(driver, fuse=True)
    block["composer_selected"] = dict(composed.selected)
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    block["grid_shape"] = [int(v) for v in driver.grid.shape]
    block["m"] = int(getattr(driver.grid, "m", 0))
    if not verdict.covered:
        reachable, why = spec.policy_reaches_composer(policy)
        needle = spec.composition_refusal_needle()
        by_policy = (not reachable and needle
                     and any(needle in r for r in verdict.reasons))
        block.update(driven=False,
                     why_not_driven=("refused by the certification-policy clause"
                                     if by_policy else "the predicate refused this row"),
                     refused_by_the_certification_policy=bool(by_policy))
        return block
    if max_cells and cells > int(max_cells):
        block.update(driven=False, why_not_driven=f"{cells} cells exceeds the cap {max_cells}")
        return block
    seed = capture(driver)
    control = drive(driver, {"array": Arrangement("array", None),
                             "array_repeat": Arrangement("array_repeat", None, wipe_private=True)},
                    min(steps, 4), stop_on_divergence=True)
    control.pop("_reference_final_state", None)
    restore(driver, seed)
    pin_array_path(driver)
    block["array_path_repeats_itself"] = bool(control["bit_identical"])
    block["determinism_control_steps"] = control["steps_compared"]
    if not control["bit_identical"]:
        block.update(driven=False,
                     why_not_driven=(f"NOT DETERMINISTIC: two array-path runs from one "
                                     f"captured seed disagree at step "
                                     f"{control['first_divergence_step']}"))
        return block
    block["driven"] = True
    # THE DRIVER'S OWN CHUNK DIVISION for every plan built on this lifted row, the one the
    # array path steps with (hd_seam_restore.plans_bind_the_driver_chunk_layout).
    with hd_seam_restore.plans_bind_the_driver_chunk_layout(driver):
        block.update(run_product(spec, driver, steps, progress=_child_progress,
                                 require_full_budget=True, clean_floor=1, floors_at="end"))
    return block


def leg_lift(spec: FamilySpec, gate_path: Path, out_dir: Path, steps: int,
             max_cells: Optional[int], timeout: float, resume: bool,
             only: Optional[Sequence[str]], interpreter: str,
             policy: Optional[str] = None) -> Dict[str, Any]:
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(spec, HERE / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; set "
                           f"MEEP_GPU_CORPUS_ROOT to the checkout the census was cut over")}
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_CYLHD_GATE_PROGRESS": str(progress_log)})
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim: set = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)
    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["MEEP_GPU_CYLHD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim else str(API_ROOT))
        started = time.time()
        if not (resume and record_path.exists()):
            command = [interpreter, "-u", str(gate_path), "--lift-child",
                       "--lift-child-leg", row["leg"], "--lift-child-out", str(record_path),
                       "--steps", str(steps)]
            if policy is not None:
                command += ["--subnormal-policy", policy]
            if max_cells is not None:
                command += ["--lift-max-cells", str(max_cells)]
            if row["leg"] == "examples":
                command += ["--lift-child-script", str(examples_dir / row["row"])]
            else:
                if not row["module"]:
                    measured.append({**row, "measured": False, "note": "no module recorded"})
                    log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module")
                    continue
                command += ["--lift-child-module", str(tests_dir / row["module"]),
                            "--lift-child-case", row["harness_case"]]
            log(f"lift {index}/{len(rows)} {row['label']} start (cells {row.get('grid_cells')})")
            stderr_text = ""
            note = "child died"
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                note = "child timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get(f"{spec.gate}_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    key = f"{spec.gate}_gate"
    blocks = {r["label"]: (r.get(key) or {}) for r in measured}
    unliftable = {record["label"]: str(record.get("error") or record.get("note") or "")
                  for record in measured if record.get("unliftable_on_this_host")}
    # A ROW THE ENGINE'S OWN LIFTER REFUSES BY NAME (MeepSimulationNotLiftable) is
    # nobody's to step either: named and counted out with the rows above.
    refused_by_the_lifter = {record["label"]: reason for record in measured
                             if record["label"] not in unliftable
                             and not (record.get(key) or {})
                             and (reason := hd_seam_restore.lifter_refusal(record))}
    unliftable.update({label: f"refused by the engine's lifter: {reason}"
                       for label, reason in refused_by_the_lifter.items()})
    blocks = {label: block for label, block in blocks.items() if label not in unliftable}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items() if b.get("predicate_admits") is False)
    refused_by_policy = sorted(label for label in refused
                               if blocks[label].get("refused_by_the_certification_policy"))
    # THE WITHDRAW REFUSALS, DERIVED FROM EACH ROW'S OWN SOURCES (an electric source
    # whose withdraw does work; none since entry 25), never from the predicate's answer;
    # the certification clause's refusals are scored by the ledger below.
    accounting = hd_seam_restore.lift_withdraw_accounting(
        blocks, facts["rows_with_a_standing_withdraw"],
        tolerated=lambda label, block: label in refused_by_policy)
    refused_by_the_withdraw = accounting["refused_by_a_standing_withdraw"]
    expected_refused = accounting["expected_refused_from_the_row_sources"]
    unmeasured = sorted(label for label, b in blocks.items() if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    non_deterministic = {label: blocks[label].get("why_not_driven") for label in admitted
                         if blocks[label].get("array_path_repeats_itself") is False}
    not_driven = {label: blocks[label].get("why_not_driven") for label in admitted
                  if not blocks[label].get("driven") and label not in non_deterministic}
    diverged = sorted(label for label in driven
                      if not all(v for v in (blocks[label].get("seam_claim") or {}).values()
                                 if v is not None))
    diverged_against_the_array_path = sorted(label for label in driven
                                             if not blocks[label].get("bit_identical"))
    reachable, why_unreachable = spec.policy_reaches_composer(policy)
    # THE REFUSAL LEDGER: every refused row is either a standing withdraw (expected from
    # the row's own sources; none since entry 25, so the seam record's withdraw_in_seam
    # rows are admitted) or, under a policy the family's arms are not certified for, the
    # certification clause -- and under such a policy EVERY liftable row must be refused
    # by it, or a row slipped through a clause that should have held.
    refusals_accounted = (
        set(refused) == (set(refused_by_the_withdraw) | set(refused_by_policy))
        and accounting["passed"]
        and set(expected_refused) <= set(refused)
        and (reachable or set(refused_by_policy) == set(blocks)))

    def _pair_holds(table: Mapping[str, Any], first: str, second: str) -> bool:
        held = table.get(first)
        return bool(held) and held == table.get(second) and "fused pair" in str(held)

    arbitration: Dict[str, Any] = {"loss": [], "tie": [], "gain": [], "unknown": []}
    for label in driven:
        table = blocks[label].get("composer_selected") or {}
        if not table:
            arbitration["unknown"].append(label)
            continue
        neighbours = (_pair_holds(table, "step_B", "update_H") + _pair_holds(table, "step_D", "update_E"))
        arbitration[("gain", "tie", "loss")[neighbours]].append(label)
    expected_stamp = ({"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}.get(policy)
                      if policy else None)
    stamps = {rec["label"]: (rec.get("subnormal_policy") or {})
              for rec in measured if rec["label"] in driven}
    child_policy_mismatches = {
        label: {"policy": stamp.get("policy"), "installed": stamp.get("installed"),
                "unattained": stamp.get("unattained"), "expected": expected_stamp}
        for label, stamp in stamps.items()
        if policy is not None and (stamp.get("policy") != expected_stamp
                                   or not stamp.get("installed") or stamp.get("unattained"))}
    child_compiles = {
        "rows_with_a_stamp": len(stamps),
        "nvrtc_calls": int(sum(int(s.get("nvrtc_calls") or 0) for s in stamps.values())),
        "ftz_removed": int(sum(int(s.get("ftz_removed") or 0) for s in stamps.values())),
        "triton_llir_compiles": int(sum(int((s.get("triton") or {}).get("llir_compiles") or 0)
                                        for s in stamps.values())),
        "triton_ptx_audited": int(sum(int((s.get("triton") or {}).get("ptx_audited") or 0)
                                      for s in stamps.values())),
        "triton_ptx_with_ftz": int(sum(int((s.get("triton") or {}).get("ptx_with_ftz") or 0)
                                       for s in stamps.values())),
        "triton_ptx_violations": int(sum(int((s.get("triton") or {}).get("ptx_violations") or 0)
                                         for s in stamps.values())),
    }
    children_ran_under_the_policy = (policy is None or not driven or (
        len(stamps) == len(driven) and not child_policy_mismatches))
    # A LEG THAT LIFTED NOTHING OF A NON-EMPTY CELL DID NOT MEASURE. Every unliftable
    # row is named with MEEP's own error, but a cell whose EVERY row is unliftable is
    # the promoted-harness-path shape (a child that never finds a Simulation) and must
    # fail rather than pass by exclusion.
    lifted_something = bool(blocks) or facts["rows_in_cell"] == 0 or bool(only)
    return {
        "passed": bool(measured and not unmeasured and lifted_something
                       and (bool(only) or len(admitted) + len(refused) + len(unliftable)
                            == facts["rows_in_cell"])
                       and len(driven) + len(non_deterministic) == len(admitted)
                       and refusals_accounted
                       and not not_driven and not diverged
                       and passed_rows == driven
                       and set(driven) | set(non_deterministic) == set(admitted)
                       and children_ran_under_the_policy
                       # UNDER A POLICY THE COMPOSER REACHES, ROWS MUST BE DRIVEN: a leg
                       # that drove nothing where it could have is a pass by not executing.
                       and (bool(driven) or not reachable or not admitted)),
        "facts": facts,
        "rows_in_cell": facts["rows_in_cell"],
        "this_host_lifted_at_least_one_row_of_the_cell": lifted_something,
        "harness_case_per_row": {row["label"]: row["harness_case"] for row in rows},
        "rows_unliftable_on_this_host": unliftable,
        "rows_this_host_could_lift": len(blocks),
        "the_denominator": (f"{len(blocks)} of the cell's {facts['rows_in_cell']} rows; "
                            f"{len(unliftable)} NAMED in rows_unliftable_on_this_host"),
        "rows_measured": len(measured), "rows_unmeasured": unmeasured,
        "admitted": len(admitted), "admitted_rows": admitted,
        "refused": refused,
        "refused_by_the_standing_withdraw": refused_by_the_withdraw,
        "refused_by_the_certification_policy": refused_by_policy,
        "refused_expected_from_the_row_sources": expected_refused,
        "refusal_accounting": accounting,
        "rows_refused_by_the_lifter": refused_by_the_lifter,
        "policy_reaches_composer": reachable,
        "why_the_policy_does_not_reach_the_composer": why_unreachable or None,
        "driven": len(driven), "driven_rows": driven, "not_driven": not_driven,
        "rows_refused_as_non_deterministic": non_deterministic,
        "rows_bit_identical": len(passed_rows),
        "rows_that_diverged": diverged,
        "rows_where_ANY_arrangement_disagrees_with_the_array_path": diverged_against_the_array_path,
        "arbitration_over_the_driven_rows": {k: sorted(v) for k, v in arbitration.items()},
        "arbitration_counts": {k: len(v) for k, v in arbitration.items()},
        "child_subnormal_policy": {
            "requested": policy, "expected_stamp": expected_stamp,
            "every_driven_child_installed_and_attained_it": children_ran_under_the_policy,
            "mismatches": child_policy_mismatches,
            "compiles_summed_over_the_children": child_compiles},
        "seam_claim_per_row": {label: blocks[label].get("seam_claim") for label in driven},
        "cylindrical_floors_per_row": {label: blocks[label].get("cylindrical_floors")
                                       for label in driven},
        "first_banded_step_per_row": {label: blocks[label].get("first_banded_step")
                                      for label in driven},
        "complete_driver_steps_compared": int(sum(blocks[label].get("steps_compared") or 0
                                                  for label in driven)),
        "words_compared": int(sum(blocks[label].get("words_compared", 0) for label in driven)),
        "modules_lifted_with_the_parameterized_shim": sorted(needs_shim),
        "steps": steps, "rows_jsonl": str(lift_dir / "rows.jsonl"),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(spec: FamilySpec) -> Dict[str, str]:
    return {name: sha256(API_ROOT / name) for name in spec.sources if (API_ROOT / name).is_file()}


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def parse_legs(value: str) -> Tuple[str, ...]:
    if value in ("all", ""):
        return ALL_LEGS
    if value in LEG_GROUPS:
        return LEG_GROUPS[value]
    wanted = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in wanted if name not in ALL_LEGS]
    if unknown:
        raise SystemExit(f"unknown leg(s) {unknown}; known: {ALL_LEGS}")
    return wanted


def verdict(spec: FamilySpec, rows: Sequence[Dict[str, Any]], legs: Sequence[str],
            policy_attained: bool, composer_reachable: bool) -> Dict[str, Any]:
    by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_leg.setdefault(row["leg"], []).append(row)
    clauses: Dict[str, Any] = {}
    for leg in legs:
        entries = by_leg.get(leg, [])
        clauses[f"{leg}_ran"] = bool(entries)
        clauses[f"{leg}_passes"] = bool(entries) and all(entry.get("passed") for entry in entries)
    if "product" in legs:
        products = by_leg.get("product", ())
        clauses["product_ran_every_case"] = ({row["case"] for row in products}
                                             == {name for name, _ in spec.cases})
        clauses["every_product_case_moved_the_reference"] = all(
            row.get("reference_moved_words_step_1", 0) > 0 for row in products)
        clauses["every_product_case_ran_the_full_budget"] = all(
            row.get("steps_compared") == row.get("steps_requested") for row in products)
        # EVERY REFERENCE THE POLICY REACHES WAS CARRIED, and every one it does not
        # reach is NAMED with its reason: under a policy the family's arms are not
        # certified for, the two composition references are the array path and are
        # recorded as unreachable rather than compared against themselves.
        clauses["every_product_case_carried_every_reachable_reference"] = all(
            set(row.get("launches", {})) | set(row.get("unreachable", {})) == set(MODES)
            for row in products)
        clauses["every_unreachable_reference_is_named_with_its_reason"] = all(
            all(bool(reason) for reason in (row.get("unreachable") or {}).values())
            for row in products)
        clauses["the_composition_references_were_carried"] = (
            (not composer_reachable) or all(
                set(row.get("launches", {})) == set(MODES) for row in products))
        clauses["every_product_case_met_the_cylindrical_floors"] = all(
            (row.get("cylindrical_floors") or {}).get("radial_difference_is_live")
            and (row.get("cylindrical_floors") or {}).get("axis_row_is_live") for row in products)
    if "mutation" in legs:
        armed = [row for row in by_leg.get("mutation", ()) if row.get("expected") == "CAUGHT"]
        clauses["every_armed_mutation_is_caught"] = bool(armed) and all(
            row.get("outcome") in ("CAUGHT", "MEASURED IN THE withdraw LEG",
                                   REFUSED_BY_THE_POLICY)
            for row in armed)
        clauses["every_mutation_refused_by_the_policy_names_its_refusal"] = all(
            row.get("refusal") for row in by_leg.get("mutation", ())
            if row.get("outcome") == REFUSED_BY_THE_POLICY)
        clauses["the_scan_ordering_plant_is_CAUGHT"] = any(
            row["mutation"] == "m_scan_reads_the_pre_launch_Hy"
            and row.get("outcome") == "CAUGHT"
            for row in by_leg.get("mutation", ()))
        clauses["every_predicted_null_is_recorded_with_its_reason"] = all(
            row.get("why") for row in by_leg.get("mutation", ()) if row.get("expected") == "NULL")
    clauses["the_policy_installed_is_the_one_requested"] = bool(policy_attained)
    clauses["every_leg_requested_ran"] = set(legs) <= set(by_leg)
    clauses["the_whole_gate_ran"] = set(legs) == set(ALL_LEGS)
    return {"clauses": clauses, "released": all(clauses.values()),
            # RECORDED, NOT A CLAUSE: a keep-certified complex family cannot reach the
            # composer under flush by construction, and the flush record's claim is the
            # forced-arm arithmetic against the array path, which the clauses above
            # score. A reader who wants the composition claim reads this flag first.
            "composer_reachable_under_this_policy": bool(composer_reachable),
            "what_released_means_when_the_composer_is_unreachable": (
                None if composer_reachable else
                "the product's arithmetic (the arm FORCED past the composer) "
                "is byte-identical to the array path and to the two certified singles "
                "built the same way over the full budget under this policy; the two "
                "composition references and the lifted rows are REFUSED by the "
                "certification clause and are named, not scored"),
            "legs_requested": list(legs), "legs_that_ran": sorted(by_leg)}


def main(spec: FamilySpec, gate_path: Path, argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=spec.gate)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--legs", default="all")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--subnormal-policy", default=None, choices=(None, "keep", "flush"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=3600.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None)
    parser.add_argument("--lift-interpreter", default=sys.executable)
    parser.add_argument("--lift-child", action="store_true")
    parser.add_argument("--lift-child-leg", default="examples")
    parser.add_argument("--lift-child-script", default=None)
    parser.add_argument("--lift-child-module", default=None)
    parser.add_argument("--lift-child-case", default=None)
    parser.add_argument("--lift-child-out", default=None)
    args = parser.parse_args(argv)

    if args.lift_child:
        return lift_child(spec, args.lift_child_leg, args.lift_child_script,
                          args.lift_child_module, args.lift_child_case, args.lift_child_out,
                          args.steps, args.lift_max_cells, policy=args.subnormal_policy)

    legs = parse_legs(args.legs)
    if args.no_device:
        legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])
    started = time.time()
    rows: List[Dict[str, Any]] = []
    family = spec.family
    payload: Dict[str, Any] = {
        "gate": spec.gate, "family": family.FAMILY, "cell_arms": list(spec.cell_arms),
        "seed": args.seed, "steps": args.steps, "legs_requested": list(legs),
        "numpy_version": np.__version__,
        "subnormal_policy_requested": args.subnormal_policy,
        "cases": [name for name, _ in spec.cases],
        "source_sha256": source_hashes(spec),
        "installable": family.INSTALLABLE,
        "installable_reason": family.INSTALLABLE_REASON,
        "launches_per_run": family.LAUNCHES_PER_RUN,
        "kernel_launches_per_run": family.KERNEL_LAUNCHES_PER_RUN,
        "what_served_means_on_a_board": (
            "PREDICATE ADMISSION, by the boards' own definition. This product is in no "
            "released-arm envelope, the dispatcher never plans it, and the composer is not "
            "offered it; a standalone gate measures its bytes and its composition"),
        "standalone_gate_note": (
            "this product is not registered in the composer's tables; the composition the "
            "composer installs today on its rows is the two neighbouring cylindrical pairs, "
            "which this gate drives as its composition_today reference. The composition "
            "with this product INSTALLED BY THE COMPOSER awaits wiring and is not measured "
            "here -- the weld arrangements force-install it"),
        "rows": rows,
    }
    policy_attained = True
    composer_reachable, why = spec.policy_reaches_composer(args.subnormal_policy)
    payload["composer_reachable_under_this_policy"] = {"reachable": composer_reachable, "why": why}
    forced = not composer_reachable
    payload["built_from_arrays"] = forced
    if not args.no_device:
        triton_launch.require_triton()
        import cupy  # noqa: PLC0415

        if args.import_meep_for_host_policy:
            import probe_fused_kernel_bit_identity as bit_identity  # noqa: PLC0415

            payload["meep_host_import"] = bit_identity.import_meep_for_host_policy()
        from meep_gpu import subnormal_policy  # noqa: PLC0415

        subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cupy, strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        policy_attained = bool(payload["subnormal_policy"].get("attained", True))
        import triton  # noqa: PLC0415

        payload.update({
            "cupy_version": cupy.__version__, "triton_version": triton.__version__,
            "device": str(cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())["name"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")})
        log(f"subnormal policy installed: {payload['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}); composer reachable: {composer_reachable}"
            + (f" ({why})" if why else ""))

    out_path = Path(args.out).resolve() if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    stream = (out_path.with_suffix(".rows.jsonl").open("w", encoding="utf-8") if out_path else None)

    def record(leg: str, result: Any) -> None:
        entries = result if isinstance(result, list) else [result]
        for entry in entries:
            entry.setdefault("leg", leg)
            entry["elapsed_s"] = round(time.time() - started, 1)
            rows.append(entry)
            if stream is not None:
                emit(stream, entry)
            log(f"{leg:20s} {entry.get('case', entry.get('mutation', '')):40s} "
                f"{'PASS' if entry.get('passed') else 'FAIL'} ({entry['elapsed_s']} s)")
            for finding in entry.get("findings", []):
                log(f"    ! {finding}")
        if out_path:
            payload["release"] = verdict(spec, rows, legs, policy_attained, composer_reachable)
            payload["device_status"] = ("RUN" if any(row["leg"] in LEG_GROUPS["device"]
                                                     for row in rows) else "NOT_RUN")
            payload["passed"] = bool(payload["release"]["released"])
            payload["seconds"] = round(time.time() - started, 1)
            gate_provenance.stamp(payload)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                encoding="utf-8")

    progress = lambda m: log(f"    {m}")  # noqa: E731
    steps, seed, policy = args.steps, args.seed, args.subnormal_policy
    try:
        if "driver_order" in legs:
            record("driver_order", leg_driver_order(spec))
        if "transcription" in legs:
            record("transcription", leg_transcription(spec))
        if "purity_ledger" in legs:
            record("purity_ledger", leg_purity_ledger(spec))
        if "ghost_observability" in legs:
            record("ghost_observability", leg_ghost_observability(spec))
        if "refusal" in legs:
            record("refusal", leg_refusal(spec, policy))
        if "arbitration" in legs:
            record("arbitration", leg_arbitration(spec, policy))
        if "product" in legs:
            record("product", leg_product(spec, steps, seed, forced, progress=progress))
        if "seed_scale" in legs:
            record("seed_scale", leg_seed_scale(spec, steps, seed, forced))
        if "launch_structure" in legs:
            record("launch_structure", leg_launch_structure(spec, steps, seed, forced))
        if "sync" in legs:
            record("sync", leg_sync(spec, steps, seed, forced))
        if "withdraw" in legs:
            record("withdraw", leg_withdraw(spec, steps, seed, forced))
        if "byte_neutral" in legs:
            record("byte_neutral", leg_byte_neutral(spec, steps, seed, forced))
        if "mutation" in legs:
            record("mutation", leg_mutation(spec, steps, seed, policy, forced))
        if "disarm" in legs:
            record("disarm", leg_disarm(spec, steps, seed, forced))
        if "lift" in legs:
            record("lift", leg_lift(spec, gate_path, out_path.parent if out_path else Path.cwd(),
                                    args.lift_steps or steps, args.lift_max_cells,
                                    args.lift_timeout, args.lift_resume,
                                    (args.lift_only.split(",") if args.lift_only else None),
                                    args.lift_interpreter, policy=policy))
    finally:
        if stream is not None:
            stream.close()

    payload["release"] = verdict(spec, rows, legs, policy_attained, composer_reachable)
    payload["device_status"] = ("RUN" if any(row["leg"] in LEG_GROUPS["device"] for row in rows)
                                else "NOT_RUN")
    payload["passed"] = bool(payload["release"]["released"])
    payload["seconds"] = round(time.time() - started, 1)
    if not args.no_device:
        # THE STAMP AT EXIT, beside the one taken at install: the executor counters (the
        # Triton compiles audited for `.ftz`, the NVRTC strips) are zero at install by
        # construction and only mean something once this process has compiled.
        from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

        payload["subnormal_policy_at_exit"] = _policy.policy_stamp()
    gate_provenance.stamp(payload)
    if out_path:
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
        log(f"wrote {out_path}")
    log(f"RELEASED={payload['release']['released']} ({payload['seconds']} s)")
    for name, value in sorted(payload["release"]["clauses"].items()):
        if not value:
            log(f"  clause FAILED: {name}")
    if not payload["release"]["released"]:
        return 1 if set(legs) == set(ALL_LEGS) else EXIT_INCOMPLETE
    return 0
