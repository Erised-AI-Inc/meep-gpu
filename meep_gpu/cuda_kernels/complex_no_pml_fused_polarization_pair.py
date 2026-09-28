"""The COMPLEX no-absorber hand-CUDA E->P fused pair: ``update_E`` -> ``update_P``, per component.

THE BOARD'S CELL: ``E_to_P (cuda_complex_no_pml/complex no-PML stored E,
cuda_complex_no_pml/complex no-PML ADE)`` -- 4 seam-instances of demand, all 4
clearing the source seam on the driver fact alone (the driver advances the
polarizations immediately after ``update_E``, driver.py:3313 then :3315, with
NOTHING between the two consults). The cell sat UNDETERMINED until the
2026-09-02 residue round's fitness entry determined it POINTWISE (fusion-residue
audit §1.5 stage 1, the ``fused_update_P_no_pml_complex[_uniform]`` label); this
module is stage 2 -- the product that serves it, on the SHIPPED
:mod:`.fused_polarization_pair` shape: the per-component split, the host
rotation between component launches, and the register drive hand-off. The
configuration is PROVEN servable by both siblings: Triton and Metal each serve
these same TestLoadDump rows with their complex fused ADE chains.

WHAT ONE COMPONENT'S LAUNCH PERFORMS. The certified complex stored-E update for
THAT component (``update_E_no_pml_complex_stored``'s own ``cf_store(f{c}, idx,
mul_field_left(minus_poles(...), inv_eps_{c}[idx]))`` line, value-named so the
result stays in a register) followed per-thread by the certified complex ADE
recurrences of that component's driving states
(``update_P_no_pml_complex[_uniform]``'s own ``ade_step`` line, per-pole
renamed, with THE SEAM: ``cf_load(drive, idx)`` becomes the register the E half
just stored -- ``fields.drive_field(c)`` IS the stored E without an absorber
(fields.py:1140-1163), and a float32 word pair stored to global and reloaded is
the identity on the bits). 3 launches replace ``1 + sum over states of
len(driven())``, with the HOST rotating between component launches exactly as
the certified launcher rotates (dispersion.py:689-691, run once per driving
state after each component launch returns).

WHY THE SPLIT IS BIT-EXACT AGAINST THE DRIVER ORDER: the real pair's own
argument, unchanged -- the complex stored-E update reads P at ITS OWN component
only (``minus_poles`` subtracts the component's own bank), so component c's
launch reads nothing any earlier component's launch or rotation wrote, and the
P recurrence for (state, c) reads that state's own ``P[c]``/``P_prev[c]`` and
writes that state's own ``_scratch`` -- disjoint across states, untouched by
the E half.

THE ALIASING RULE IS THE REAL PAIR'S: the pole bank is bound ONCE -- the E
half's ``minus_poles`` and the P half's ``p_now`` load both read through the
same ``a{i}`` parameter -- and the spare bank slots are bound to the source
pointer and never read, exactly as the certified stored-E launcher binds them
(two const restrict pointers that alias carry no modified object, the certified
precedent).

NOTHING HERE IS DISPATCH; :mod:`.fused_pairs` can plan it opt-in only.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from . import complex_no_pml_kernels as _family
    from .coverage import ade_sigma_is_volume
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.machinery as _machinery
    import importlib.util as _importlib_util
    import os as _os
    import sys as _sys

    def _load(stem):
        """By-path loader with a synthetic package, so the loaded siblings'
        own relative imports resolve too."""
        here = _os.path.dirname(_os.path.abspath(__file__))
        package = "cuda_kernels_bypath"
        if package not in _sys.modules:
            spec = _machinery.ModuleSpec(package, None, is_package=True)
            shim = _importlib_util.module_from_spec(spec)
            shim.__path__ = [here]
            _sys.modules[package] = shim
        name = f"{package}.{stem}"
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _importlib_util.spec_from_file_location(
            name, _os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _family = _load("complex_no_pml_kernels")
    ade_sigma_is_volume = _load("coverage").ade_sigma_is_volume
    compile_cache = _load("compile_cache")

try:
    from .complex_pml_kernels import word_view
except ImportError:  # a host with no CuPy: only the launch is unavailable
    word_view = None


# =============================================================================
# THE PARTITION -- EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================

#: Byte-identical with a gate verdict AND a record behind it.
#: TWO RECORDS STAND BEHIND THIS ONE NAME AND THE LATER ONE IS THE LIVE BYTES'.
#: ``cuda_complex_no_pml_fused_polarization_pair_2026-09-02`` certified the
#: weld whose drive-identity clause could only be established through
#: ``array.data.ptr``; ``_drive_identity_problem`` now settles the same fact by
#: OBJECT identity first, so the backend-free census can read it, and the weld
#: was re-gated in its own right as
#: ``cuda_complex_no_pml_fused_polarization_pair_2026-09-02b``. The DEVICE TEXT
#: is untouched by that edit -- it is a predicate clause, not a kernel line --
#: and both runs measure the same emitted source; the second block exists
#: because the module's bytes moved and a verdict is about bytes.
#:
#: EACH RUN: both legs RELEASED under BOTH float32 subnormal policies on
#: the GPU host, 8/8 cases identical to BOTH references per complete driver step --
#: the pole volumes included and compared BY SLOT -- with the launch count 3
#: per seam step against the certified route's 4 or 7, the seam null
#: (``reload_drive_from_global``) INERT as declared, the wrong-drive control
#: diverging, and all 14 mutation legs and all 4 rotation legs passing (their
#: ``launcher_copy_NULL`` required and measured inert).
CERTIFIED_KERNELS = (
    "fused_polarization_pair_no_pml_complex",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: FALSE, BY DRIVER CONSTRUCTION rather than by measurement: the driver injects
#: NOTHING between update_E (driver.py:3313) and update_P (:3315) -- no
#: injection slot, no fill, no wall pass -- so there is no deposit for a
#: bracket to carry and ``fused_pairs.FUSED_PAIR_SEAMS`` records the seam name
#: ``None``. The same fact, from the same lines, that the two real E->P
#: products declare.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_complex_no_pml_fused_polarization_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_polarization_pair_no_pml_complex"

#: The driver passes the THREE per-component launches jointly perform
#: (driver.py:3313, :3315). No in-between pass exists to carry or refuse.
REPLACES = ("update_E", "update_P")

#: The sub-step slot the planner holds this on -- the seam-LEADING slot.
SLOT = "update_E"

#: Every line of certified device text this file did not lift verbatim.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    cf_store(f0, idx, mul_field_left(\n"
             "        minus_poles(g0, a0, ..., idx, np0),\n"
             "        inv_eps_0[idx]));",
     "became": "    cf ev = mul_field_left(minus_poles(...), inv_eps_0[idx]);\n"
               "    cf_store(f0, idx, ev);",
     "why": "same expression tree, same store; the value is additionally NAMED "
            "so the P half can read it from a register. A float32 word pair "
            "stored to global and reloaded is the identity on the bits."},
    {"line": "    cf_store(p_out, idx, ade_step(\n"
             "        cf_load(p_now, idx), cf_load(p_prev, idx), "
             "cf_load(drive, idx),\n        sigma[idx], c_now, c_prev, c_drive));",
     "became": "    cf_store(p_out_i, idx, ade_step(\n"
               "        cf_load(a_i, idx), cf_load(p_prev_i, idx), ev,\n"
               "        sigma_i[idx], c_now_i, c_prev_i, c_drive_i));",
     "why": "per-pole renames plus THE SEAM: the drive load becomes the "
            "register the E half just stored (fields.drive_field(c) IS the "
            "stored E without an absorber), and p_now becomes the bank slot "
            "the E half's minus_poles already reads through -- the pole buffer "
            "bound ONCE, the real pair's aliasing rule."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "MAX_POLES", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "component_specs",
    "covers_complex_no_pml_fused_polarization_pair", "device_sources",
    "fused_polarization_pair_no_pml_complex_source",
    "launch_complex_no_pml_fused_polarization_component",
    "run_complex_no_pml_fused_polarization_pair",
]

#: The certified stored-E template's compiled bank width, inherited.
MAX_POLES = _family.MAX_POLES

#: (stored E, displacement, bank stem, count name) per component, transcribed
#: from the certified ``_STORED_E_TEMPLATE``'s own three store lines.
_COMPONENTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("Ex", "Dx", "a", "np0"),
    ("Ey", "Dy", "b", "np1"),
    ("Ez", "Dz", "c", "np2"),
)


# =============================================================================
# THE LIFT
# =============================================================================

def _certified_store_line(component_index: int, arm) -> str:
    """The certified stored-E line for ONE component, lifted verbatim.

    Read from the certified emission itself (``kernel_source('update_E', arm)``)
    so a gate mutation of the family's template reaches this weld, and asserted
    to appear exactly once.
    """
    target, _displacement, stem, count_name = _COMPONENTS[component_index]
    f = f"f{component_index}"
    g = f"g{component_index}"
    banks = ", ".join(f"{stem}{slot}" for slot in range(MAX_POLES))
    line = (f"    cf_store({f}, idx, mul_field_left(\n"
            f"        minus_poles({g}, {banks}, idx, {count_name}),\n"
            f"        inv_eps_{component_index}[idx]));\n")
    emission = _family.kernel_source("update_E", arm)
    if emission.count(line) != 1:
        raise AssertionError(
            f"the certified stored-E emission carries {target}'s store line "
            f"{emission.count(line)} times, not once; this weld LIFTS that line "
            f"rather than retyping it and cannot splice around its absence")
    return line


def _certified_recurrence_line(sigma_is_volume: bool, arm) -> str:
    """The certified complex ADE store, lifted verbatim from the emission."""
    key = "update_P" if sigma_is_volume else "update_P_uniform"
    sigma_read = "sigma[idx]" if sigma_is_volume else "sigma"
    line = ("    cf_store(p_out, idx, ade_step(\n"
            "        cf_load(p_now, idx), cf_load(p_prev, idx), "
            "cf_load(drive, idx),\n"
            f"        {sigma_read}, c_now, c_prev, c_drive));\n")
    emission = _family.kernel_source(key, arm)
    if emission.count(line) != 1:
        raise AssertionError(
            f"the certified {key} emission carries the recurrence store "
            f"{emission.count(line)} times, not once; this weld LIFTS that line "
            f"rather than retyping it and cannot splice around its absence")
    return line


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Substitute ``old`` exactly once, or raise naming what was being lifted."""
    if text.count(old) != 1:
        raise AssertionError(
            f"the certified text carries {text.count(old)} occurrences of "
            f"{what}, not 1; this family LIFTS it rather than retyping it")
    return text.replace(old, new, 1)


def _prelude(arm) -> str:
    """The certified family prelude for one arm -- the cf helpers, minus_poles
    and ade_step, exactly as the certified kernels compile them."""
    emission = _family.kernel_source("update_E", arm)
    marker = 'extern "C" __global__ void '
    if emission.count(marker) != 1:
        raise AssertionError(
            "the certified stored-E emission no longer carries one global entry "
            "point; the prelude cannot be identified")
    return emission.split(marker, 1)[0]


def _signature(component_index: int, count: int,
               sigma_kinds: Sequence[bool]) -> str:
    """The fused signature for one component, one specialization.

    The ONLY hand-written device text in this module, and it is a signature: no
    arithmetic lives here. Each pole-bank pointer appears EXACTLY ONCE; both
    halves read through it.
    """
    target, _displacement, stem, count_name = _COMPONENTS[component_index]
    f = f"f{component_index}"
    g = f"g{component_index}"
    lines: List[str] = [f'extern "C" __global__ void {KERNEL_NAME}(']
    lines.append(f"    // {target}'s word view, the E half's target.")
    lines.append(f"    float* __restrict__ {f},")
    lines.append(f"    // {target}'s displacement, read-only here.")
    lines.append(f"    const float* __restrict__ {g},")
    lines.append("    // NOT __restrict__, the certified template's own "
                 "declaration.")
    lines.append(f"    const float* inv_eps_{component_index},")
    lines.append(f"    // THE POLE BANK, BOUND ONCE ({MAX_POLES} slots, the "
                 f"certified width);")
    lines.append("    // spare slots are bound to the source pointer and never "
                 "read.")
    for slot in range(MAX_POLES):
        lines.append(f"    const float* __restrict__ {stem}{slot},")
    for i in range(count):
        sigma = (f"const float* __restrict__ sigma_{i}," if sigma_kinds[i]
                 else f"float sigma_{i},")
        lines.append(f"    float* __restrict__ p_out_{i}, "
                     f"const float* __restrict__ p_prev_{i}, {sigma}")
        lines.append(f"    float c_now_{i}, float c_prev_{i}, "
                     f"float c_drive_{i},")
    lines.append(f"    int {count_name}, int n_elem")
    lines.append(") {")
    return "\n".join(lines) + "\n"


def fused_polarization_pair_no_pml_complex_source(
        arm, component_index: int, count: int,
        sigma_kinds: Sequence[bool]) -> str:
    """The whole fused kernel for one arm, one component, one specialization."""
    if not 0 <= int(component_index) <= 2:
        raise ValueError(f"component_index must be 0..2, got {component_index!r}")
    count = int(count)
    if not 0 <= count <= MAX_POLES:
        raise ValueError(
            f"count must be 0..{MAX_POLES} (the certified bank width), got "
            f"{count}")
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    if len(kinds) != count:
        raise ValueError(
            f"sigma_kinds carries {len(kinds)} flags for {count} poles; one "
            f"compile-time sigma form per driving state, in registration order")
    target, _displacement, stem, _count_name = _COMPONENTS[component_index]
    f = f"f{component_index}"

    # THE E HALF: the certified store line, value-named. Same expression tree,
    # same store; the register is what the P half reads.
    store = _certified_store_line(component_index, arm)
    store = _replace_once(
        store, f"    cf_store({f}, idx, mul_field_left(",
        "    cf ev = mul_field_left(", "the certified store's opening")
    store = _replace_once(
        store, "[idx]));", f"[idx]);\n    cf_store({f}, idx, ev);",
        "the certified store's closing")

    poles: List[str] = []
    for i in range(count):
        line = _certified_recurrence_line(kinds[i], arm)
        line = _replace_once(line, "cf_store(p_out, idx,",
                             f"cf_store(p_out_{i}, idx,",
                             "the recurrence's store")
        line = _replace_once(line, "cf_load(p_now, idx)",
                             f"cf_load({stem}{i}, idx)",
                             "the recurrence's P^n load (the bank slot)")
        line = _replace_once(line, "cf_load(p_prev, idx)",
                             f"cf_load(p_prev_{i}, idx)",
                             "the recurrence's P^(n-1) load")
        line = _replace_once(
            line, "cf_load(drive, idx)", "ev",
            "the certified drive load (THE SEAM: the register the E half "
            "just stored)")
        if kinds[i]:
            line = _replace_once(line, "sigma[idx]", f"sigma_{i}[idx]",
                                 "the volume sigma load")
        else:
            line = _replace_once(line, "        sigma,", f"        sigma_{i},",
                                 "the uniform sigma bind")
        line = _replace_once(line, "c_now, c_prev, c_drive",
                             f"c_now_{i}, c_prev_{i}, c_drive_{i}",
                             "the recurrence coefficients")
        poles.append(
            f"\n    // pole {i}: dispersion.PolarizationState.update, the "
            f"certified\n    // complex recurrence under this launch's per-pole "
            f"names.\n" + line)

    source = "".join((
        _prelude(arm),
        _signature(component_index, count, kinds),
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx >= n_elem) return;\n"
        "\n"
        "    // THE E HALF: the certified stored-E update for this component,\n"
        "    // value-named. stepping.update_P binds fields.drive_field(c) as\n"
        "    // the drive -- the STORED E without an absorber (fields.py:1140-\n"
        "    // 1163) -- and a float32 word pair stored and reloaded is the\n"
        "    // identity on the bits, so the register IS the certified load.\n",
        store,
        "".join(poles),
        "}\n",
    ))
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Representative sources for a digest: both arms, all components, the
    corpus-relevant counts and both sigma forms."""
    out: Dict[str, str] = {}
    for arm_name in ("NAIVE", "FMA_V1"):
        for component in (0, 1, 2):
            for count in (0, 1, 2):
                for kinds in sorted({tuple([True] * count),
                                     tuple([False] * count)}):
                    key = (f"{KERNEL_NAME}|arm{arm_name}|c{component}|n{count}|"
                           f"{''.join('v' if kind else 'u' for kind in kinds)}")
                    out[key] = fused_polarization_pair_no_pml_complex_source(
                        arm_name, component, count, kinds)
    return out


_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, component_index: int, count: int,
                sigma_kinds: Sequence[bool]):
    """Compile one specialization, memoized on (name, options, policy, source)."""
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicate and the emitter need no device and still run")
    code = fused_polarization_pair_no_pml_complex_source(
        arm, component_index, count, sigma_kinds)
    kinds = "".join("v" if bool(kind) else "u" for kind in sigma_kinds)
    key = compile_cache.kernel_cache_key(
        f"{KERNEL_NAME}_c{component_index}_n{count}_{kinds}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


# =============================================================================
# THE LAUNCH
# =============================================================================

def _driving_states(fields: Any, target: str) -> Tuple[Any, ...]:
    """The states driving one component, in ``fields.polarizations`` order --
    the real pair's own resolver, restated against the same engine facts."""
    return tuple(state for state in tuple(getattr(fields, "polarizations", ()) or ())
                 if callable(getattr(state, "drives", None)) and state.drives(target))


def component_specs(fields: Any) -> Dict[str, Dict[str, Any]]:
    """Per component: the driving states, pole count and sigma kinds, RIGHT NOW.
    Resolved per call and never cached: the rotation moves the buffers between
    component launches."""
    specs: Dict[str, Dict[str, Any]] = {}
    for index, (target, _displacement, _stem, _count) in enumerate(_COMPONENTS):
        states = _driving_states(fields, target)
        specs[target] = {
            "index": index,
            "states": states,
            "count": len(states),
            "kinds": tuple(bool(ade_sigma_is_volume(state, target))
                           for state in states),
        }
    return specs


def _base_address(array: Any) -> Optional[int]:
    data = getattr(array, "data", None)
    pointer = getattr(data, "ptr", None)
    return int(pointer) if pointer is not None else None


def _drive_identity_problem(fields: Any, target: str) -> Optional[str]:
    """Why the register hand-off would not be the certified drive; None if it is."""
    drive = getattr(fields, "drive_field", None)
    if not callable(drive):
        return ("fields.drive_field is not callable; the drive identity the "
                "register hand-off rests on cannot be checked")
    try:
        named = drive(target)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"fields.drive_field({target!r}) raised {type(exc).__name__}: {exc}"
    expected = getattr(fields, target, None)
    if expected is None:
        return f"fields.{target} is not allocated"
    # TWO WAYS TO ESTABLISH THE SAME FACT, AND THE OBJECT ONE COMES FIRST.
    # The claim is that ``update_P`` reads the array ``update_E`` just wrote;
    # ``named is expected`` settles it outright, on any array module, and a
    # base-pointer match settles the VIEW case CuPy can produce (a different
    # object over the same allocation) which object identity alone would miss.
    #
    # THE POINTER TEST ALONE WAS A CENSUS DEFECT, measured 2026-09-02 and worth
    # recording where the clause is: ``_base_address`` reads ``array.data.ptr``,
    # which is a CuPy attribute -- a NumPy ``ndarray.data`` is a memoryview and
    # has none -- so on the BACKEND-FREE census every row refused here with a
    # reason about the drive, whatever the drive actually was. The board then
    # reported this product's whole cell (4 corpus rows, all four clearing the
    # seam) as served by nothing, which is a wrong number about a cell that has
    # a shipped, certified product on it. The identity is unchanged; what
    # changed is that it can now be established off-device too.
    if named is expected:
        return None
    named_address = _base_address(named)
    if named_address is not None and named_address == _base_address(expected):
        return None
    return (f"fields.drive_field({target!r}) is neither fields.{target} nor a "
            f"view over its allocation: the update_P half would read a drive "
            f"the update_E half of this launch did not write, and the register "
            f"hand-off would be a different number, not a faster one")


def assert_disjoint_bindings(fields: Any, target: str,
                             states: Sequence[Any]) -> int:
    """Every ``__restrict__`` pointer of ONE component launch distinct.

    CHECKED PER LAUNCH, AFTER THE PREVIOUS COMPONENT'S ROTATION. The spare bank
    slots deliberately alias the source pointer (both read-only, the certified
    stored-E launcher's own binding) and are excluded here.
    """
    named: List[Tuple[str, Any]] = [
        (target, getattr(fields, target)),
        ("D" + target[1], getattr(fields, "D" + target[1]))]
    for i, state in enumerate(states):
        named.append((f"bank_{i}", state.P[target]))
        named.append((f"p_prev_{i}", state.P_prev[target]))
        named.append((f"p_out_{i}", state._scratch))
        if ade_sigma_is_volume(state, target):
            named.append((f"sigma_{i}", state.sigma[target]))
    seen: Dict[int, str] = {}
    for label, array in named:
        address = _base_address(array)
        if address is None:
            raise ValueError(
                f"{label} exposes no readable base address; the restrict "
                f"promises in the fused signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]} in the {target} launch; every "
                f"live pointer in this signature is __restrict__, and the weld's "
                f"whole shape exists to bind each allocation once")
        seen[address] = label
    return len(seen)


def launch_complex_no_pml_fused_polarization_component(
        fields: Any, arm, target: str, states: Sequence[Any],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """ONE component's fused launch. The caller rotates afterwards."""
    spec = {t: i for i, (t, _d, _s, _c) in enumerate(_COMPONENTS)}
    if target not in spec:
        raise ValueError(f"target must be one of {sorted(spec)}, got {target!r}")
    component_index = spec[target]
    count = len(states)
    if count > MAX_POLES:
        raise ValueError(
            f"{count} driving states exceed the certified bank width "
            f"{MAX_POLES}; the certified stored-E template refuses the same")
    kinds = tuple(bool(ade_sigma_is_volume(state, target)) for state in states)
    problem = _drive_identity_problem(fields, target)
    if problem is not None:
        raise ValueError(problem)
    assert_disjoint_bindings(fields, target, states)
    if word_view is None:
        raise RuntimeError(
            "complex_pml_kernels.word_view is not importable on this host (it "
            "imports CuPy at module scope); a launch needs it")

    stored = getattr(fields, target)
    n_elem = int(stored.size)
    source_words = word_view(getattr(fields, "D" + target[1]))
    arguments: List[Any] = [word_view(stored), source_words,
                            fields.inverse_epsilon_for(target)]
    for i in range(MAX_POLES):
        if i < count:
            arguments.append(word_view(states[i].P[target]))
        else:
            # A SPARE SLOT, bound to the source pointer and never read -- the
            # certified stored-E launcher's own binding.
            arguments.append(source_words)
    for i, state in enumerate(states):
        scratch = state._scratch
        if int(scratch.size) != n_elem:
            raise ValueError(
                f"state {i}'s scratch holds {int(scratch.size)} cells and the "
                f"launch walks {n_elem}; the certified update_P guard this weld "
                f"drops was that size, and dropping it is licensed only where "
                f"the two are one number")
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple this "
                f"kernel bakes as scalars")
        arguments.append(word_view(scratch))
        arguments.append(word_view(state.P_prev[target]))
        sigma = state.sigma[target]
        arguments.append(sigma if kinds[i] else np.float32(sigma))
        arguments.extend(np.float32(value) for value in coefficients)
    arguments.append(np.int32(count))
    arguments.append(np.int32(n_elem))

    blocks = (n_elem + _FUSED_THREADS - 1) // _FUSED_THREADS
    (kernel or _get_kernel(arm, component_index, count, kinds))(
        (blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"component": target, "poles": count, "kinds": list(kinds),
            "blocks": blocks, "threads": _FUSED_THREADS, "elements": n_elem}


def run_complex_no_pml_fused_polarization_pair(
        fields: Any, pml: Any, arm, *,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in THREE launches, host rotation between.

    THE ROTATION IS TRANSCRIBED from dispersion.py:689-691, run once per
    driving state after each component launch RETURNS -- the real pair's own
    sequence, which is what makes the buffer sequence identical to the
    certified per-(state, component) launcher's.
    """
    launches = 0
    recurrences = 0
    per_component: List[Dict[str, Any]] = []
    for target, _displacement, _stem, _count in _COMPONENTS:
        spec = component_specs(fields)[target]
        states = spec["states"]
        result = launch_complex_no_pml_fused_polarization_component(
            fields, arm, target, states, kernel)
        launches += 1
        recurrences += len(states)
        per_component.append(result)
        for state in states:
            # dispersion.py:689-691, verbatim, once the launch API returned.
            p = state.P[target]
            p_prev = state.P_prev[target]
            scratch = state._scratch
            state.P[target] = scratch
            state.P_prev[target] = p
            state._scratch = p_prev
    return {"launched": True, "launches": launches,
            "recurrences": recurrences, "replaces": REPLACES,
            "kernel": KERNEL_NAME, "components": per_component}


# =============================================================================
# COVERAGE
# =============================================================================

def _seam_clauses(fields: Any) -> Optional[str]:
    """The clauses this weld ADDS over its two halves; None when they hold."""
    stored = getattr(fields, "Ex", None)
    if stored is None:
        return "fields.Ex is not allocated"
    n_elem = int(getattr(stored, "size", 0))
    for target, _displacement, _stem, _count in _COMPONENTS:
        states = _driving_states(fields, target)
        if states:
            problem = _drive_identity_problem(fields, target)
            if problem is not None:
                return problem
        if len(states) > MAX_POLES:
            return (f"{len(states)} states drive {target}, past the certified "
                    f"bank width {MAX_POLES}")
        for index, state in enumerate(states):
            for label, array in ((f"P[{target!r}]", state.P.get(target)),
                                 (f"P_prev[{target!r}]",
                                  state.P_prev.get(target)),
                                 ("_scratch", getattr(state, "_scratch", None))):
                size = int(getattr(array, "size", -1)) if array is not None else -1
                if size != n_elem:
                    return (f"polarization {index} {label} holds {size} cells "
                            f"and the fused launch walks {n_elem}; this weld "
                            f"drops the certified update_P guard and may only "
                            f"do so where the two bounds are one number")
    return None


def covers_complex_no_pml_fused_polarization_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May three per-component launches span ``update_E`` -> ``update_P``?

    A CONJUNCTION over the ``cuda_complex_no_pml`` family's own two certified
    predicates -- ``covers_complex_no_pml_stored_e`` and
    ``covers_complex_no_pml_ade_update_p`` -- plus the seam clauses the weld
    adds (the drive identity behind the register hand-off, the extent identity
    behind the dropped guard, the certified bank width).

    ``sources`` IS ACCEPTED AND INERT -- a driver fact, not a leniency: nothing
    is injected between driver.py:3313 and :3315 on any run. Disjoint from the
    two REAL E->P products on storage alone (their halves refuse complex64 by
    name; these require it).
    """
    covered, reason = _family.covers_complex_no_pml_stored_e(
        fields, pml, grid, license, subnormal_policy)
    if not covered:
        return False, f"update_E half: {reason}"
    covered, reason = _family.covers_complex_no_pml_ade_update_p(
        fields, pml, grid, license, subnormal_policy)
    if not covered:
        return False, f"update_P half: {reason}"
    problem = _seam_clauses(fields)
    if problem is not None:
        return False, problem
    return True, "covered"
