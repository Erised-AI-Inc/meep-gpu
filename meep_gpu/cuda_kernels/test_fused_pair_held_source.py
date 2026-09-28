"""The fused products' plan-held source and resolve-once, exercised where there is no device.

WHY THIS FILE EXISTS. Until 2026-09-19 every CUDA fused-product plan called its
launcher without the launcher's ``kernel=`` door, so the launcher's
``(kernel or _get_kernel())`` ran on every launch and each getter re-spliced its
whole certified source before keying the memo (laptop, stub CuPy: 274.5 us per
magnetic-pair launch, 385.7 us per electric, 4601 us for the conductive/BFAST H->D
weld), and ``CudaFusedPairPlan.run`` re-resolved arguments that never change.
``fused_pairs._held_kernel`` now emits once per plan and shape and asks the memo on
every launch; ``fused_pairs._resolved_once`` holds a pure resolver's answer.

WHAT IS PINNED, and each is a way the change could be silently wrong:

1. THE KEY IS THE GETTER'S. After the product's own ``_get_kernel`` has compiled
   (what ``warm()`` does at plan time), a plan's launches add NO compile-log entry
   and hand the launcher the very object the getter returned. A copied name, flag
   or option tuple that drifted would split the memo, and on a device every first
   launch would recompile.
2. ONE EMISSION PER PLAN AND SHAPE, and none at plan build.
3. ``get_or_compile`` IS ASKED ON EVERY LAUNCH, through the module attribute, so the
   route gate's two witnesses -- the proxy installed after import and the memo wrap
   installed after the first launch -- both see every held launch and agree
   (``parity/meep_gpu/gate_dispatch_end_to_end.py:1049-1140``).
4. THE POLICY STAYS ON THE PER-LAUNCH PATH: a token change after the first launch
   is a memo miss compiled from the held text.
5. RESOLVE-ONCE IS OPT-IN AND NARROW: the builders that hold are exactly the pure
   ones, and the off-diagonal weld and the special_kz H->D weld -- whose resolvers
   hand out scratch the launch rotates -- still resolve on every launch, while their
   TEXT is held like everyone else's.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. ``cupy`` is a stub whose ``RawKernel``
records its construction; every emitter is a fake returning a string that depends
only on its arguments, so the getter and the plan emit equal text exactly when they
are asked the same question -- which is the question being tested.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types
from typing import Any, Dict, List

import pytest

from . import compile_cache, fused_pairs

# EVERY MODULE A BUILDER OR CLOSURE IMPORTS LAZILY IS IMPORTED HERE, before any stub
# ``cupy`` exists. A module first imported while the stub sat in ``sys.modules`` would
# bind it at scope for the rest of the session; the fixture below asserts none was.
from . import (  # noqa: F401
    bfast_fused_electric_pair, bfast_fused_magnetic_pair,
    complex_beta_fused_electric_pair, complex_beta_fused_magnetic_pair, complex_emitter,
    complex_folded_fused_electric_pair, complex_folded_fused_magnetic_pair,
    complex_fused_electric_pair, complex_fused_magnetic_pair,
    conductive_bfast_fused_hd_pair, conductive_fused_electric_pair, coverage,
    cylindrical_fused_electric_pair, cylindrical_fused_hd_pair,
    cylindrical_fused_magnetic_pair, cylindrical_prefix, cylindrical_real_fused_electric_pair,
    cylindrical_real_fused_hd_pair, cylindrical_real_fused_magnetic_pair,
    dispersive_fused_electric_pair, fused_electric_pair, fused_hd_pair,
    fused_magnetic_pair, in_seam_coverage, no_pml_complex_fused_electric_pair,
    no_pml_dispersive_fused_electric_pair, no_pml_three_slot_dispersive_weld,
    complex_no_pml_offdiag_fused_electric_pair, complex_offdiag_stencil_weld,
    complex_offdiag_update_e, folded_complex_offdiag_fused_electric_pair,
    folded_offdiag_fused_electric_pair, folded_offdiag_kernels,
    offdiag_emitter, offdiag_fused_electric_pair, offdiag_stencil_weld, registry,
    special_kz_fused_electric_pair, special_kz_fused_hd_pair,
    special_kz_fused_magnetic_pair, three_slot_dispersive_weld,
)

PACKAGE = "meep_gpu.cuda_kernels"
LAUNCHES = 4


# ---------------------------------------------------------------------------
# The stand-ins
# ---------------------------------------------------------------------------

class _StubRawKernel:
    """``cupy.RawKernel`` as far as the memo can tell: constructed, then called."""

    constructed = 0

    def __init__(self, code: str, name: str, options=()) -> None:
        type(self).constructed += 1
        self.code, self.name, self.options = code, name, tuple(options)
        self.calls = 0

    def __call__(self, grid, block, args, **kwargs):
        self.calls += 1

    def compile(self):
        return None


class _Arguments(dict):
    """A resolved-argument dict that answers every key a closure reads.

    The closures only FORWARD what they resolve, so a sentinel per key is enough for
    every entry but the few that select a kernel (the arm, the expansion, the curl
    arm, the H->D state), which each product's row sets explicitly.
    """

    def __missing__(self, key):
        return f"<{key}>"


GRID = types.SimpleNamespace(dt=0.5, dx=1.0)
FIELDS = types.SimpleNamespace(grid=GRID, Dx=types.SimpleNamespace(shape=(4, 4, 4)))
ROW_MASK = (1, 0, 0, 0, 0, 0)


def _context() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        fields=FIELDS, pml="<pml>", grid=GRID, sources=(),
        license_for=lambda _name: {"arm": 1}, subnormal_policy=None)


class _Recorder:
    """The product's launcher (or ``run_``), recording the kernel it was handed.

    It CALLS that kernel, once, the way the real launcher does, so a counting proxy
    or a wrapped memo value sees the launch.
    """

    def __init__(self) -> None:
        self.kernels: List[Any] = []
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args, **kwargs):
        kernel = kwargs.get("kernel")
        self.kernels.append(kernel)
        self.calls.append({"args": args, "kwargs": kwargs})
        kernel((1,), (1,), ())
        return {"launched": True, "blocks": 1, "threads": 1}


def _fake_emitter(counter: Dict[str, int], label: str):
    def emit(*args):
        counter[label] = counter.get(label, 0) + 1
        return f"// {label}{args!r}"
    return emit


# ---------------------------------------------------------------------------
# The covered products, one row each. ``getter`` is what the product's own
# ``_get_kernel`` is called with for the SAME launch the row drives.
# ---------------------------------------------------------------------------

_CYL_PREFIX = ("cylindrical_prefix",)
PRODUCTS: Dict[str, Dict[str, Any]] = {
    "fused_magnetic_pair": dict(
        builder="_fused_magnetic_pair_plan", launcher="launch_fused_magnetic_pair",
        emitter="fused_magnetic_pair_source", getter=()),
    "fused_electric_pair": dict(
        builder="_fused_electric_pair_plan", launcher="launch_fused_electric_pair",
        emitter="fused_electric_pair_source", getter=()),
    "bfast_fused_magnetic_pair": dict(
        builder="_bfast_fused_magnetic_pair_plan",
        launcher="launch_bfast_fused_magnetic_pair",
        emitter="bfast_fused_magnetic_pair_source", getter=()),
    "bfast_fused_electric_pair": dict(
        builder="_bfast_fused_electric_pair_plan",
        launcher="launch_bfast_fused_electric_pair",
        emitter="bfast_fused_electric_pair_source", getter=()),
    "special_kz_fused_magnetic_pair": dict(
        builder="_special_kz_fused_magnetic_pair_plan",
        launcher="launch_special_kz_fused_magnetic_pair",
        emitter="special_kz_fused_magnetic_pair_source", getter=()),
    "special_kz_fused_electric_pair": dict(
        builder="_special_kz_fused_electric_pair_plan",
        launcher="launch_special_kz_fused_electric_pair",
        emitter="special_kz_fused_electric_pair_source", getter=()),
    "cylindrical_real_fused_magnetic_pair": dict(
        builder="_cylindrical_real_fused_magnetic_pair_plan",
        launcher="launch_cylindrical_real_fused_magnetic_pair",
        emitter="cylindrical_real_fused_magnetic_pair_source", getter=(),
        patches=_CYL_PREFIX),
    "cylindrical_real_fused_electric_pair": dict(
        builder="_cylindrical_real_fused_electric_pair_plan",
        launcher="launch_cylindrical_real_fused_electric_pair",
        emitter="cylindrical_real_fused_electric_pair_source", getter=(),
        patches=_CYL_PREFIX),
    "fused_hd_pair": dict(
        builder="_fused_hd_pair_plan", launcher="launch_fused_hd_pair",
        emitter="kernel_source", getter=(), patches=("scratch_rotation",),
        prime=("fused_hd_pair_tables", "fused_hd_pair_codes", "fused_hd_pair_scratch")),
    "special_kz_fused_hd_pair": dict(
        builder="_special_kz_fused_hd_pair_plan",
        launcher="launch_special_kz_fused_hd_pair",
        emitter="kernel_source", getter=(), patches=("scratch_rotation",)),
    "cylindrical_real_fused_hd_pair": dict(
        builder="_cylindrical_real_fused_hd_pair_plan",
        launcher="run_cylindrical_real_fused_hd_pair",
        emitter="kernel_source", getter=(), arguments={"state": {}}),
    "complex_fused_magnetic_pair": dict(
        builder="_complex_fused_magnetic_pair_plan",
        launcher="launch_complex_fused_magnetic_pair",
        emitter="complex_fused_magnetic_pair_source", getter=("FMA_V1",),
        arguments={"arm": "FMA_V1"}),
    "complex_fused_electric_pair": dict(
        builder="_complex_fused_electric_pair_plan",
        launcher="launch_complex_fused_electric_pair",
        emitter="complex_fused_electric_pair_source", getter=("FMA_V1",),
        arguments={"arm": "FMA_V1"}),
    "complex_folded_fused_magnetic_pair": dict(
        builder="_complex_folded_fused_magnetic_pair_plan",
        launcher="launch_complex_folded_fused_magnetic_pair",
        emitter="complex_folded_fused_magnetic_pair_source", getter=("NAIVE",),
        arguments={"arm": "NAIVE"}),
    "complex_folded_fused_electric_pair": dict(
        builder="_complex_folded_fused_electric_pair_plan",
        launcher="launch_complex_folded_fused_electric_pair",
        emitter="complex_folded_fused_electric_pair_source", getter=("NAIVE",),
        arguments={"arm": "NAIVE"}),
    "complex_beta_fused_magnetic_pair": dict(
        builder="_complex_beta_fused_magnetic_pair_plan",
        launcher="launch_complex_beta_fused_magnetic_pair",
        emitter="complex_beta_fused_magnetic_pair_source", getter=(1,),
        arguments={"arm": 1}),
    "complex_beta_fused_electric_pair": dict(
        builder="_complex_beta_fused_electric_pair_plan",
        launcher="launch_complex_beta_fused_electric_pair",
        emitter="complex_beta_fused_electric_pair_source", getter=(1,),
        arguments={"arm": 1}),
    "cylindrical_fused_magnetic_pair": dict(
        builder="_cylindrical_fused_magnetic_pair_plan",
        launcher="launch_cylindrical_fused_magnetic_pair",
        emitter="cylindrical_fused_magnetic_pair_source", getter=("FMA_V1",),
        arguments={"arm": "FMA_V1"}, patches=_CYL_PREFIX),
    "cylindrical_fused_electric_pair": dict(
        builder="_cylindrical_fused_electric_pair_plan",
        launcher="launch_cylindrical_fused_electric_pair",
        emitter="cylindrical_fused_electric_pair_source", getter=("FMA_V1",),
        arguments={"arm": "FMA_V1"}, patches=_CYL_PREFIX),
    "conductive_fused_electric_pair": dict(
        builder="_conductive_fused_electric_pair_plan",
        launcher="launch_conductive_fused_electric_pair",
        emitter="conductive_fused_electric_pair_source", getter=((True, False, True),),
        patches=("conductive_bindings_3",)),
    "dispersive_fused_electric_pair": dict(
        builder="_dispersive_fused_electric_pair_plan",
        launcher="launch_dispersive_fused_electric_pair",
        emitter="dispersive_fused_electric_pair_source", getter=((1, 0, 2),),
        patches=("pole_bindings",)),
    "no_pml_dispersive_fused_electric_pair": dict(
        builder="_no_pml_dispersive_fused_electric_pair_plan",
        launcher="launch_no_pml_dispersive_fused_electric_pair",
        emitter="no_pml_dispersive_fused_electric_pair_source",
        getter=((False, True, False), (1, 0, 2)),
        patches=("pole_bindings", "conductive_bindings_2")),
    "three_slot_dispersive_weld": dict(
        builder="_three_slot_dispersive_weld_plan", group="leading",
        module="dispersive_fused_electric_pair",
        launcher="launch_dispersive_fused_electric_pair",
        emitter="dispersive_fused_electric_pair_source", getter=((1, 0, 2),),
        patches=("pole_bindings",)),
    "no_pml_three_slot_dispersive_weld": dict(
        builder="_no_pml_three_slot_dispersive_weld_plan", group="leading",
        module="no_pml_dispersive_fused_electric_pair",
        launcher="launch_no_pml_dispersive_fused_electric_pair",
        emitter="no_pml_dispersive_fused_electric_pair_source",
        getter=((False, True, False), (1, 0, 2)),
        patches=("pole_bindings", "conductive_bindings_2")),
    "conductive_bfast_fused_hd_pair": dict(
        builder="_conductive_bfast_fused_hd_pair_plan",
        launcher="run_conductive_bfast_fused_hd_pair",
        emitter="current_source", getter=("conductive", (True, False, False)),
        arguments={"state": {"variant": "conductive", "cond": (True, False, False)}}),
    "conductive_bfast_fused_hd_pair[bfast]": dict(
        builder="_conductive_bfast_fused_hd_pair_plan",
        module="conductive_bfast_fused_hd_pair",
        launcher="run_conductive_bfast_fused_hd_pair",
        emitter="current_source", getter=("bfast", None),
        arguments={"state": {"variant": "bfast", "cond": None}}),
    "cylindrical_fused_hd_pair": dict(
        builder="_cylindrical_fused_hd_pair_plan",
        launcher="run_cylindrical_fused_hd_pair",
        emitter="kernel_source", getter=(1,), arguments={"state": {"arm": 1}}),
    "no_pml_complex_fused_electric_pair": dict(
        builder="_no_pml_complex_fused_electric_pair_plan",
        launcher="launch_no_pml_complex_fused_electric_pair",
        emitter="no_pml_complex_fused_electric_pair_source", getter=("FMA_V1", True),
        arguments={"expansion": "FMA_V1", "curl_arm": "conductive"}),
    "offdiag_fused_electric_pair": dict(
        builder="_offdiag_fused_electric_pair_plan",
        launcher="launch_offdiag_fused_electric_pair",
        emitter="kernel_source", getter=(ROW_MASK,),
        arguments={"row_mask": ROW_MASK}, patches=("scratch_rotation",)),
    "folded_offdiag_fused_electric_pair": dict(
        builder="_folded_offdiag_fused_electric_pair_plan",
        launcher="launch_folded_offdiag_fused_electric_pair",
        emitter="kernel_source", getter=(ROW_MASK,),
        arguments={"row_mask": ROW_MASK},
        patches=("scratch_rotation", "validate")),
    "folded_complex_offdiag_fused_electric_pair": dict(
        builder="_folded_complex_offdiag_fused_electric_pair_plan",
        launcher="launch_folded_complex_offdiag_fused_electric_pair",
        emitter="kernel_source", getter=(ROW_MASK, "FMA_V1"),
        arguments={"row_mask": ROW_MASK, "arm": "FMA_V1"},
        patches=("scratch_rotation", "validate")),
    "complex_no_pml_offdiag_fused_electric_pair": dict(
        builder="_complex_no_pml_offdiag_fused_electric_pair_plan",
        launcher="launch_complex_no_pml_offdiag_fused_electric_pair",
        emitter="kernel_source", getter=(ROW_MASK, 0),
        arguments={"row_mask": ROW_MASK, "arm": 0},
        patches=("scratch_rotation", "validate")),
    "no_pml_complex_fused_electric_pair[plain]": dict(
        builder="_no_pml_complex_fused_electric_pair_plan",
        module="no_pml_complex_fused_electric_pair",
        launcher="launch_no_pml_complex_fused_electric_pair",
        emitter="no_pml_complex_fused_electric_pair_source", getter=(0, False),
        arguments={"expansion": 0, "curl_arm": "plain"}),
}


def _module(row_name: str) -> types.ModuleType:
    row = PRODUCTS[row_name]
    return sys.modules[f"{PACKAGE}.{row.get('module', row_name)}"]


@pytest.fixture
def device(monkeypatch):
    """A stub ``cupy`` everywhere a getter can reach for one, and a clean memo.

    Restores the memo and the log it found, and REFUSES to finish if a package module
    was first imported while the stub stood: that module would have bound the stub
    at scope for every later test in the session.
    """
    before = set(sys.modules)
    saved_memo = dict(compile_cache._compiled_kernels)  # noqa: SLF001
    saved_log = list(compile_cache._compile_log)  # noqa: SLF001
    stub = types.ModuleType("cupy")
    stub.RawKernel = _StubRawKernel
    monkeypatch.setitem(sys.modules, "cupy", stub)
    compile_cache.clear_kernel_cache()
    compile_cache.clear_compile_log()
    yield stub
    compile_cache.clear_kernel_cache()
    compile_cache.clear_compile_log()
    compile_cache._compiled_kernels.update(saved_memo)  # noqa: SLF001
    compile_cache._compile_log.extend(saved_log)  # noqa: SLF001
    leaked = sorted(name for name in set(sys.modules) - before
                    if name.startswith("meep_gpu"))
    assert not leaked, (
        f"{leaked} were first imported while the stub cupy stood; pre-import them at "
        f"the top of this file")


def _arm(row_name: str, monkeypatch, stub) -> Dict[str, Any]:
    """Patch one product for a device-free drive; returns the emission counter."""
    row = PRODUCTS[row_name]
    family = _module(row_name)
    counter: Dict[str, int] = {}
    monkeypatch.setattr(family, row["emitter"], _fake_emitter(counter, row["emitter"]))
    if hasattr(family, "cp"):
        monkeypatch.setattr(family, "cp", stub)
    # THE PACKAGE MEMO, AS A CUPY HOST BINDS IT. ``complex_fused_magnetic_pair`` and
    # ``complex_fused_electric_pair`` import ``complex_pml_kernels`` (CuPy at scope)
    # in the same ``try`` as ``compile_cache``, so on THIS host the block falls to
    # its by-path branch and the getter keys a PRIVATE copy of the memo
    # (``cuda_kernels_compile_cache``). Where CuPy imports -- the only place a plan
    # launches -- the in-package branch runs and the getter's memo is the package
    # module ``fused_pairs`` asks. Pointing the attribute there reproduces that host.
    if (getattr(family, "compile_cache", compile_cache) is not compile_cache):
        monkeypatch.setattr(family, "compile_cache", compile_cache)
    recorder = _Recorder()
    monkeypatch.setattr(family, row["launcher"], recorder)
    for patch in row.get("patches", ()):
        if patch == "cylindrical_prefix":
            monkeypatch.setattr(cylindrical_prefix, "cylindrical_prefix",
                                lambda *a, **k: "<prefix>")
        elif patch == "scratch_rotation":
            rotations = iter(range(10 ** 6))
            monkeypatch.setattr(family, "assert_scratch_is_disjoint",
                                lambda *a, **k: None)
            monkeypatch.setattr(family, "rotate_into_fields",
                                lambda fields, scratch: f"<rotated {next(rotations)}>")
        elif patch == "validate":
            monkeypatch.setattr(family, "_validate_launch_arguments",
                                lambda *a, **k: None)
        elif patch == "conductive_bindings_3":
            monkeypatch.setattr(family, "conductive_bindings",
                                lambda fields: (("c",) * 6, ("h",) * 3,
                                                (True, False, True)))
        elif patch == "conductive_bindings_2":
            monkeypatch.setattr(family, "conductive_bindings",
                                lambda fields: (("c",) * 6, (False, True, False)))
        elif patch == "pole_bindings":
            monkeypatch.setattr(family, "pole_bindings",
                                lambda fields: (("p",) * 3, [1, 0, 2]))
            monkeypatch.setattr(family, "dispersive_kernels", types.SimpleNamespace(
                normalized_pole_counts=lambda c: tuple(int(v) for v in c)))
        else:  # pragma: no cover - a typo in the table
            raise AssertionError(f"no such patch {patch!r}")
    for name in row.get("prime", ()):
        monkeypatch.setattr(family, name, lambda *a, _n=name, **k: f"<{_n}>")
    if row_name == "special_kz_fused_hd_pair":
        for name in ("special_kz_fused_hd_pair_tables", "special_kz_fused_hd_pair_codes",
                     "beta_scalars", "special_kz_fused_hd_pair_scratch"):
            monkeypatch.setattr(family, name, lambda *a, _n=name, **k: f"<{_n}>")
    return {"counter": counter, "recorder": recorder, "family": family}


def _plan(row_name: str):
    row = PRODUCTS[row_name]
    plan = getattr(fused_pairs, row["builder"])(_context())
    if row.get("prime") or row_name == "special_kz_fused_hd_pair":
        # These two closures read state their resolver allocates, so the drive goes
        # through the resolver first -- with every allocation a sentinel.
        plan.resolve_launch_args(_context())
    return plan


def _launch(row_name: str, plan) -> None:
    row = PRODUCTS[row_name]
    if row.get("group") == "leading":
        plan.launch_leading(FIELDS, _Arguments(row.get("arguments", {})))
    elif row_name == "special_kz_fused_hd_pair":
        plan.launch_kernel(FIELDS, plan.resolve_launch_args(_context()))
    else:
        plan.launch_kernel(FIELDS, _Arguments(row.get("arguments", {})))


# ---------------------------------------------------------------------------
# 1-2. The key is the getter's; one emission per plan; none at build
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row_name", sorted(PRODUCTS))
def test_a_held_launch_is_served_the_entry_the_getter_compiled(row_name, device,
                                                               monkeypatch):
    armed = _arm(row_name, monkeypatch, device)
    family, counter = armed["family"], armed["counter"]
    warmed = family._get_kernel(*PRODUCTS[row_name]["getter"])  # noqa: SLF001
    assert len(compile_cache.compile_log()) == 1
    compile_cache.clear_compile_log()
    counter.clear()

    plan = _plan(row_name)
    assert counter == {}, "a device-free plan build must emit nothing"
    for _ in range(LAUNCHES):
        _launch(row_name, plan)

    assert compile_cache.compile_log() == (), (
        f"{row_name}: the held key missed the entry its own getter compiled -- the "
        f"name, the storage flag or the options were copied wrong; log "
        f"{compile_cache.compile_log()}")
    assert all(kernel is warmed for kernel in armed["recorder"].kernels), row_name
    assert len(armed["recorder"].kernels) == LAUNCHES
    assert counter == {PRODUCTS[row_name]["emitter"]: 1}, (
        f"{row_name}: {counter} emissions over {LAUNCHES} launches, not 1")


def test_every_builder_that_holds_a_source_has_a_row_here():
    """A new held builder must be added to :data:`PRODUCTS`, or its key is untested."""
    tree = ast.parse(pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8"))
    helpers = {"_held_kernel", "_held_arm_kernel", "_held_dispersive_kernel",
               "_held_no_pml_dispersive_kernel", "_held_offdiag_kernel"}
    holding = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name in helpers:
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in helpers):
                holding.add(node.name)
    assert holding == {row["builder"] for row in PRODUCTS.values()}


def test_a_second_shape_is_a_second_emission_never_a_stale_one(device, monkeypatch):
    """The dispersive arity is read per launch: a new arity gets its own text."""
    armed = _arm("dispersive_fused_electric_pair", monkeypatch, device)
    family = armed["family"]
    plan = _plan("dispersive_fused_electric_pair")
    arities = iter([[1, 0, 2], [1, 0, 2], [2, 2, 2], [1, 0, 2]])
    monkeypatch.setattr(family, "pole_bindings",
                        lambda fields: (("p",) * 3, next(arities)))
    for _ in range(4):
        _launch("dispersive_fused_electric_pair", plan)
    assert armed["counter"] == {"dispersive_fused_electric_pair_source": 2}
    kernels = armed["recorder"].kernels
    assert kernels[0] is kernels[1] is kernels[3]
    assert kernels[2] is not kernels[0]
    assert "(2, 2, 2)" in kernels[2].code and "(1, 0, 2)" in kernels[0].code


# ---------------------------------------------------------------------------
# 3-4. The witnesses see every launch; the policy stays per launch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row_name", sorted(PRODUCTS))
def test_both_gate_witnesses_see_every_held_launch_and_agree(row_name, device,
                                                             monkeypatch):
    """The gate's proxy is installed AFTER import and its memo wrap AFTER launch 1.

    Holding the kernel object would read zero on both for every launch past the
    first, and the route gate would call the witnesses disagreeing.
    """
    _arm(row_name, monkeypatch, device)
    plan = _plan(row_name)
    proxy = {"asked": 0, "launched": 0}
    real = compile_cache.get_or_compile

    class _Counting:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __call__(self, *args, **kwargs):
            proxy["launched"] += 1
            return self._inner(*args, **kwargs)

    def counted(*args, **kwargs):
        proxy["asked"] += 1
        return _Counting(real(*args, **kwargs))

    monkeypatch.setattr(compile_cache, "get_or_compile", counted)
    _launch(row_name, plan)

    memo = {"launched": 0}

    class _Wrapped:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __call__(self, *args, **kwargs):
            memo["launched"] += 1
            return self._inner(*args, **kwargs)

    store = compile_cache._compiled_kernels  # noqa: SLF001
    for key, kernel in list(store.items()):
        store[key] = _Wrapped(kernel)
    for _ in range(LAUNCHES - 1):
        _launch(row_name, plan)

    assert proxy == {"asked": LAUNCHES, "launched": LAUNCHES}, (row_name, proxy)
    assert memo["launched"] == LAUNCHES - 1, (row_name, memo)


def test_a_policy_change_after_the_first_launch_recompiles_the_held_text(
        device, monkeypatch):
    armed = _arm("fused_magnetic_pair", monkeypatch, device)
    plan = _plan("fused_magnetic_pair")
    _launch("fused_magnetic_pair", plan)
    assert len(compile_cache.compile_log()) == 1
    monkeypatch.setattr(compile_cache, "compile_policy_token", lambda: "flush|later")
    _launch("fused_magnetic_pair", plan)
    log = compile_cache.compile_log()
    assert len(log) == 2 and log[1]["policy_token"] == "flush|later"
    assert log[0]["source_sha256"] == log[1]["source_sha256"]
    assert armed["counter"] == {"fused_magnetic_pair_source": 1}, (
        "the recompile under the new policy must come from the HELD text")
    first, second = armed["recorder"].kernels
    assert first is not second


# ---------------------------------------------------------------------------
# 5. Resolve-once: opt-in, and only where nothing rotates
# ---------------------------------------------------------------------------

#: The builders whose resolver reads only the frozen configuration.
RESOLVED_ONCE = {
    "_fused_magnetic_pair_plan", "_fused_electric_pair_plan",
    "_bfast_fused_magnetic_pair_plan", "_bfast_fused_electric_pair_plan",
    "_special_kz_fused_magnetic_pair_plan", "_special_kz_fused_electric_pair_plan",
    "_cylindrical_real_fused_magnetic_pair_plan",
    "_cylindrical_real_fused_electric_pair_plan",
    "_complex_fused_magnetic_pair_plan", "_complex_fused_electric_pair_plan",
    "_complex_folded_fused_magnetic_pair_plan", "_complex_folded_fused_electric_pair_plan",
    "_complex_beta_fused_magnetic_pair_plan", "_complex_beta_fused_electric_pair_plan",
    "_cylindrical_fused_magnetic_pair_plan", "_cylindrical_fused_electric_pair_plan",
    "_conductive_fused_electric_pair_plan", "_dispersive_fused_electric_pair_plan",
    "_no_pml_dispersive_fused_electric_pair_plan", "_three_slot_dispersive_weld_plan",
    "_no_pml_three_slot_dispersive_weld_plan",
}

#: Resolvers that hand out state the launch rotates, or read the fields: never held.
NEVER_RESOLVED_ONCE = {
    "_offdiag_fused_electric_pair_plan", "_folded_offdiag_fused_electric_pair_plan",
    "_folded_complex_offdiag_fused_electric_pair_plan",
    "_complex_no_pml_offdiag_fused_electric_pair_plan",
    "_dispersive_offdiag_fused_polarization_pair_plan", "_special_kz_fused_hd_pair_plan",
    "_no_pml_complex_fused_electric_pair_plan",
}


def test_resolve_once_is_opted_into_by_exactly_the_pure_builders():
    tree = ast.parse(pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8"))
    opted = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == "_resolved_once"):
                opted.add(node.name)
    assert opted == RESOLVED_ONCE
    assert not opted & NEVER_RESOLVED_ONCE


def test_run_itself_still_asks_the_resolver_on_every_launch():
    """Resolve-once lives in the builders, so it cannot reach a rotating resolver."""
    asked = []
    plan = fused_pairs.CudaFusedPairPlan(
        family="f", label="l", kernel_label="k", replaces=(),
        slots=("step_D", "update_E"), context=types.SimpleNamespace(fields=None),
        resolve_launch_args=lambda ctx: asked.append(1) or {"n": len(asked)},
        launch=lambda fields, arguments: {"blocks": arguments["n"]})
    for _ in range(3):
        plan.run()
    assert len(asked) == 3 and plan.launch_grid == (3,)


def test_a_pure_resolver_is_read_once_per_plan(device, monkeypatch):
    armed = _arm("bfast_fused_magnetic_pair", monkeypatch, device)
    family = armed["family"]
    reads: Dict[str, int] = {}

    def counting(name, value):
        def read(*args, **kwargs):
            reads[name] = reads.get(name, 0) + 1
            return value
        return read

    monkeypatch.setattr(coverage, "real_curl_boundary_codes",
                        counting("codes", ((0, 0, 0), None)))
    monkeypatch.setattr(in_seam_coverage, "zero_metal_axes", counting("walls", (0, 0, 0)))
    monkeypatch.setattr(family, "bfast_fused_magnetic_pair_tables",
                        counting("tables", "<tables>"))
    monkeypatch.setattr(family, "bfast_scalars", counting("coefficients", "<c>"))
    plan = _plan("bfast_fused_magnetic_pair")
    for _ in range(LAUNCHES):
        plan.run()
    assert reads == {"codes": 1, "walls": 1, "tables": 1, "coefficients": 1}
    assert plan.launches == LAUNCHES
    handed = [call["args"][1] for call in armed["recorder"].calls]
    assert handed == ["<tables>"] * LAUNCHES


def test_the_off_diagonal_weld_still_resolves_every_launch(device, monkeypatch):
    """Its resolver reads the row volumes fresh and hands out the scratch the launch
    rotates; held, the second launch would bind the FIRST launch's scratch. Its TEXT
    is held all the same -- one emission across the three launches."""
    family = offdiag_fused_electric_pair
    emitted: Dict[str, int] = {}
    monkeypatch.setattr(family, "kernel_source", _fake_emitter(emitted, "kernel_source"))
    reads = {"rows": 0}
    handed: List[Any] = []
    rotations = iter(range(10 ** 6))

    def rows(fields):
        reads["rows"] += 1
        return ["<row>"]

    monkeypatch.setattr(coverage, "offdiag_row_volumes", rows)
    monkeypatch.setattr(coverage, "offdiag_row_mask", lambda fields: (1, 0, 0))
    monkeypatch.setattr(coverage, "offdiag_wall_mask_flags", lambda grid: (0, 0, 0))
    monkeypatch.setattr(offdiag_emitter, "normalized_row_mask", lambda mask: mask)
    monkeypatch.setattr(offdiag_stencil_weld, "fill_plan", lambda grid: "<fills>")
    monkeypatch.setattr(family, "offdiag_fused_electric_pair_tables", lambda pml: "<t>")
    monkeypatch.setattr(family, "offdiag_fused_electric_pair_scratch",
                        lambda fields: "<scratch 0>")
    monkeypatch.setattr(family, "offdiag_fused_electric_pair_codes",
                        lambda grid, pml: "<codes>")
    monkeypatch.setattr(family, "assert_scratch_is_disjoint", lambda *a, **k: None)
    monkeypatch.setattr(family, "rotate_into_fields",
                        lambda fields, scratch: f"<rotated {next(rotations)}>")

    def launcher(fields, scratch, *args, **kwargs):
        handed.append(scratch)
        return {"launched": True, "blocks": 1}

    monkeypatch.setattr(family, "launch_offdiag_fused_electric_pair", launcher)
    plan = fused_pairs._offdiag_fused_electric_pair_plan(_context())  # noqa: SLF001
    for _ in range(3):
        plan.run()
    assert reads["rows"] == 3
    assert handed == ["<scratch 0>", "<rotated 0>", "<rotated 1>"]
    assert emitted == {"kernel_source": 1}


def test_the_special_kz_hd_weld_holds_its_text_but_not_its_resolution(device,
                                                                      monkeypatch):
    """``resolve`` returns ``dict(held)`` including the scratch the launch rotates."""
    armed = _arm("special_kz_fused_hd_pair", monkeypatch, device)
    plan = fused_pairs._special_kz_fused_hd_pair_plan(_context())  # noqa: SLF001
    for _ in range(3):
        plan.run()
    handed = [call["args"][1] for call in armed["recorder"].calls]
    assert handed == ["<special_kz_fused_hd_pair_scratch>", "<rotated 0>",
                      "<rotated 1>"]
    assert armed["counter"] == {"kernel_source": 1}
    resolved = plan.resolve_launch_args(_context())
    assert "sources" not in resolved, (
        "the held text must not leak into the dict the resolver copies")
