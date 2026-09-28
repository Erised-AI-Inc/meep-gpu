"""Shared machinery of the four SCRATCH-OUTPUT offdiag D/E welds.

THE DESIGN, measured before it was built
(``parity/meep_gpu/results/metal_scratch_weld_closed_form_2026-09-01/probe.json``,
ten fixtures, two complete steps each, byte-identical, every reachable null
diverging): the off-diagonal constitutive arm is a STENCIL over the curl arm's
D output, so the weld that fuses them must never mutate D in place. Instead

* the curl half writes ``D_new``/``fu_new`` to LAUNCH-LOCAL SCRATCH buffers and
  the pre-launch ``D``/``fu`` stay readable for the whole dispatch;
* the constitutive half takes its own cell's displacement from registers and
  RE-DERIVES every foreign stencil tap from pre-launch state through the same
  inline function (:func:`step_cell_function`) — at most three foreign cells per
  live term, and NOTHING WRITTEN IS EVER READ;
* the in-seam passes are carried per cell by a CLOSED FORM
  (:func:`d_final_function`): a fill destination redirects to its image source
  (near: stored 0 -> ``MIRROR_SOURCE_INDEX`` with ``mirror_parity``; far: stored
  ``last`` -> the runtime reflect row), the wall clear sits INSIDE the far parity
  and OUTSIDE the near one (the driver's order; flipping it moves signed-zero
  words — measured, 2 words on the odd-parity walled fixture), and an unfolded
  family's form degenerates to ``clear ? +0 : v``;
* the LAUNCHER ROTATES the D/fu buffers afterwards: the plan owns twin host
  arrays, resolves the current role of each physical buffer immediately before
  every launch by host identity — :meth:`.device.Residency.tensor_for_host`, the
  seam built for exactly this — and swaps the ENGINE's references only after the
  launch succeeds. That is :mod:`.ade_update_p`'s certified choreography on the
  D seam, where the array path itself never rotates, so the plan owns both
  buffers outright rather than following an engine rotation.

WHAT IS LIFTED AND WHAT IS NEW. The curl arithmetic inside
:func:`step_cell_function` is the certified curl emitter's own body — cut at the
decode-prologue anchor, re-headed as an inline function of ``(i, j, k)``, stores
replaced by a struct return, and asserted line for line
(:func:`lift_curl_tail`). The constitutive text is the certified offdiag
emitter's own output with the displacement loads substituted for register /
re-derivation reads through exact needles (each asserted to match exactly once).
The NEW text — the only text these families can be blamed for — is the closed
form in :func:`d_final_function`, whose arithmetic is the probe's, and whose
parity spellings are the fills' own measured ones (``-x`` real,
``c_mul((+/-1, +0), z)`` complex, plain copy for +1 real).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..fields import IYEE_SHIFTS, mirror_parity
from .device import Residency

__all__ = [
    "AXES", "COORDS", "D_TARGETS", "ScratchWeldPairPlan", "d_final_function",
    "displacement_needles", "lift_curl_tail", "needle", "plan_scratch_weld",
    "seam_lines", "step_cell_function", "substitute_loads", "weld_body",
]

AXES = "xyz"
COORDS = ("i", "j", "k")
D_TARGETS = ("Dx", "Dy", "Dz")

#: The decode-prologue anchor every certified curl body ends its index derivation
#: with. The lift cuts AFTER this line and re-derives ``ii`` from the function's
#: own (i, j, k) parameters — integer arithmetic, so no bit can move.
DECODE_END = "    int i   = plane / nyi;\n"


def needle(text: str, old: str, new: str, count: int = 1) -> str:
    """Replace with an exact occurrence count, or refuse.

    The same discipline every gate applies to a mutation: a substitution that
    silently found nothing (or two of something) emits a kernel reading the wrong
    volume, which is a smooth wrong answer rather than a crash.
    """
    hits = text.count(old)
    if hits != count:
        raise AssertionError(
            f"the weld needle {old!r} matches {hits} times, expected {count}")
    return text.replace(old, new)


def lift_curl_tail(source: str, *, store_edits: Sequence[Tuple[str, str]],
                   decode_end: str = DECODE_END,
                   guard: str = "if (idx >= n_elem) { return; }") -> str:
    """The certified curl body BELOW the decode prologue, its stores rewritten.

    ``source`` is the certified emitter's own output. Everything through
    ``decode_end`` is dropped — the helper takes i/j/k as parameters and rebuilds
    ``nyz``/``ii`` from them — and the ``n_elem`` guard must have sat above the
    cut, so the helper is total on every stored cell and the dispatch keeps the
    guard.

    ``decode_end`` IS A PER-FAMILY ANCHOR and not a constant, because the real and
    complex emitters end their prologues differently: the real curl's last decode
    line is ``int i = plane / nyi;`` with ``nyz`` above it, and the complex one
    puts ``int nxi = int(nx);`` and ``int nyz = nyi * nzi;`` BELOW that line. Cut
    at the wrong one and the helper redeclares ``nyz`` (a compile error) or reads
    an ``nx`` it has no parameter for.

    ``store_edits`` is ``(old, new)`` per store line, each applied exactly once:
    ``("...f0[ii] = v0...", "")`` where the certified line computed the value into
    a register first and only stored it, and ``("f0[ii] = f0[ii] - curl0;",
    "float2 v0 = f0[ii] - curl0;")`` where the certified line was the computation.
    THE RIGHT-HAND SIDE IS NEVER TOUCHED by either form: what moves is the
    destination, from a volume the fused kernel must not write to a register the
    struct returns.
    """
    if guard not in source.split(decode_end, 1)[0]:
        raise AssertionError("the certified curl source no longer carries its "
                             "guard above the decode prologue")
    if decode_end not in source:
        raise AssertionError(
            f"the certified curl source no longer carries the decode anchor "
            f"{decode_end.strip()!r}; the lift has nowhere to cut")
    tail = source.split(decode_end, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError("the certified curl source does not end with '}'")
    tail = tail[: -len("}\n")]
    for old, new in store_edits:
        tail = needle(tail, old, new)
    return tail


def step_cell_function(name: str, lifted_tail: str, *, value_type: str,
                       pointer_parameters: Sequence[Tuple[str, str]],
                       scalar_parameters: Sequence[Tuple[str, str]],
                       returns: Sequence[str]) -> str:
    """One inline function computing the raw stepped cell from PRE-LAUNCH state.

    The body is the lifted certified curl tail; the header re-derives the flat
    index from the (i, j, k) parameters. ONE code path serves the thread's own
    cell and every foreign re-derivation, so the two cannot disagree.

    ``returns`` names the registers the struct carries (``v0..v2`` always,
    ``n0..n2`` where the family stores a split-field auxiliary).
    """
    fields = "".join(f" {value_type} {reg};" for reg in returns)
    params = ", ".join(
        [f"int {coord}" for coord in COORDS]
        + ["int nxi", "int nyi", "int nzi"]
        + [f"{kind} {label}" for kind, label in pointer_parameters]
        + [f"{kind} {label}" for kind, label in scalar_parameters])
    tail = "\n".join("    " + line if line.strip() else line
                     for line in lifted_tail.splitlines())
    out = [
        f"struct {name}_result {{{fields} }};",
        f"static inline {name}_result {name}({params}) {{",
        "    int nyz = nyi * nzi;",
        "    int ii = i * nyz + j * nzi + k;",
        tail,
        f"    {name}_result out;",
        "".join(f" out.{reg} = {reg};" for reg in returns).strip(),
        "    return out;",
        "}",
    ]
    return "\n".join(out)


def _parity_sign(component: str, axis: int, phase: int) -> int:
    weight = int(mirror_parity(component, axis, int(phase)))
    if weight not in (1, -1):
        raise ValueError(f"mirror parity must be +/-1, got {weight!r}")
    return weight


def _apply_parity(value: str, weight: int, *, complex_storage: bool,
                  guard: str) -> Optional[str]:
    """One parity application line, in the fills' own measured spelling.

    Real: ``-x`` for -1 and NO LINE for +1 (a plain copy is the identity; a
    runtime weight flushes subnormals on this backend — symmetry._parity_spelling).
    Complex: the FULL product ``c_mul((w, +0), z)`` at BOTH signs — the folded
    complex fill's own arithmetic (folded_complex.mirror_parity_coefficients:
    the coefficient words are (+/-1.0, +0.0) exactly, and +1 is NOT elided
    because the fill launches the product there too).
    """
    if not complex_storage:
        if weight == 1:
            return None
        return f"    {value} = {guard} ? -{value} : {value};"
    word = f"float2({float(weight):.1f}f, 0.0f)"
    return f"    {value} = {guard} ? c_mul({word}, {value}) : {value};"


def d_final_function(component_index: int, *, step_name: str,
                     call_arguments: str, value_type: str,
                     folded_axes: Mapping[int, int],
                     far_axes: Mapping[int, int],
                     zero_metal: Sequence[bool],
                     complex_storage: bool,
                     own_parameters: str = "",
                     mirror_source_index: int = 2) -> str:
    """The closed form for one component: fills and clear carried per cell.

    ``folded_axes`` maps folded axis -> declared phase; ``far_axes`` the subset
    with a stored far slot (folded PERIODIC) -> the same phase. ``zero_metal`` is
    the walled-axis triple.

    ``own_parameters`` are parameters THIS function takes and does NOT forward to
    ``step_name`` — the runtime reflect rows, which ride in ``Params`` and are the
    one input the far arm needs and the curl does not. They are kept a separate
    argument rather than appended to ``call_arguments`` because the inner call's
    actuals are rebuilt from THAT string's labels: a reflect row spliced in there
    would be passed on to a ``step_cell`` that has no parameter for it, which is a
    compile error on a good day and an argument shift on a bad one.

    The emitted composition is the probe's:

        far?  ->  parity_far x [ clear ? +0 : near(y) ]      (mask INSIDE)
        else  ->  clear ? +0 : near(x)                        (mask OUTSIDE near)

    with ``near`` the multi-axis redirect carrying the parity PRODUCT — one
    line per axis, so a corner composes exactly as the sequential fills do.
    """
    component = D_TARGETS[component_index]
    iyee = IYEE_SHIFTS[component]
    own_axis = component_index
    zero = "float2(0.0f, 0.0f)" if complex_storage else "0.0f"
    lines: List[str] = [
        f"static inline {value_type} d_final_{component_index}("
        f"int i, int j, int k, int nxi, int nyi, int nzi{call_arguments}"
        f"{own_parameters}) {{"]

    near_axes = [axis for axis in sorted(folded_axes) if iyee[axis] == 0]
    far_here = own_axis in far_axes and iyee[own_axis] == 1

    if far_here:
        extent = ("nxi", "nyi", "nzi")[own_axis]
        reflect = f"reflect_{AXES[own_axis]}"
        lines.append(f"    bool was_far = ({COORDS[own_axis]} == {extent} - 1);")
        lines.append(f"    {COORDS[own_axis]} = was_far ? {reflect} : "
                     f"{COORDS[own_axis]};")

    redirect_guards: List[Tuple[int, str]] = []
    for axis in near_axes:
        guard = f"near_{AXES[axis]}"
        lines.append(f"    bool {guard} = ({COORDS[axis]} == 0);")
        lines.append(f"    {COORDS[axis]} = {guard} ? {mirror_source_index} : "
                     f"{COORDS[axis]};")
        redirect_guards.append((axis, guard))

    # The pointer pass-through is rebuilt from the parameter labels alone, and
    # from `call_arguments` ONLY — `own_parameters` is this function's and stops
    # here, which is what keeps the inner call's arity the curl's.
    forwarded = _forward_labels(call_arguments)
    lines.append(f"    {value_type} value = {step_name}({', '.join(COORDS)}, "
                 f"nxi, nyi, nzi{forwarded}).v{component_index};")

    for axis, guard in redirect_guards:
        line = _apply_parity("value", _parity_sign(component, axis,
                                                   folded_axes[axis]),
                            complex_storage=complex_storage, guard=guard)
        if line is not None:
            lines.append(line)

    clear_guards = [f"({COORDS[axis]} == 0)"
                    for axis in range(3)
                    if bool(zero_metal[axis]) and iyee[axis] == 0]
    if clear_guards:
        lines.append(f"    value = ({' || '.join(clear_guards)}) ? {zero} : "
                     f"value;")

    if far_here:
        line = _apply_parity("value", _parity_sign(component, own_axis,
                                                   far_axes[own_axis]),
                            complex_storage=complex_storage, guard="was_far")
        if line is not None:
            lines.append(line)

    lines.append("    return value;")
    lines.append("}")
    return "\n".join(lines)


def _forward_labels(call_arguments: str) -> str:
    """``", kind name, kind name"`` -> ``", name, name"`` for the inner call."""
    out: List[str] = []
    for piece in call_arguments.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(piece.split()[-1].lstrip("*"))
    return ("", ", " + ", ".join(out))[bool(out)]


def substitute_loads(text: str, substitutions: Sequence[Tuple[str, str, int]]) -> str:
    """Apply the (old, new, count) needles that turn loads into re-derivations.

    A ``count`` of ``-1`` means AT LEAST ONE and however many the emitter wrote:
    ``g0[ii]`` appears once as the component's own displacement and once more per
    live term that takes component 0 as a partner, and the number is a property of
    the row mask rather than of this function. Zero occurrences is still refused —
    a needle that matched nothing has left a displacement read standing.
    """
    for old, new, count in substitutions:
        if count < 0:
            hits = text.count(old)
            if hits < 1:
                raise AssertionError(
                    f"the weld needle {old!r} matched nothing; a displacement read "
                    f"the fused kernel cannot serve would be left standing")
            text = text.replace(old, new)
            continue
        text = needle(text, old, new, count)
    return text


def displacement_needles(row_mask: Sequence[int], *, e_terms: Sequence[Any],
                         transverse_partners: Sequence[Sequence[int]],
                         neighbour_names: Mapping[int, Tuple[str, str]],
                         index: Callable[[Dict[int, str]], str],
                         register: str, call: str) -> List[Tuple[str, str, int]]:
    """Every displacement read the certified constitutive body makes, redirected.

    THE NEEDLES ARE BUILT FROM THE EMITTER'S OWN TABLES — ``index`` is that
    emitter's ``_index``, ``neighbour_names`` its per-axis ``(down, up)`` index
    variable names, and ``transverse_partners``/``e_terms`` its partner tables —
    so a change to how the certified kernel spells a neighbour index reaches this
    substitution without a second edit, and a needle that stopped matching RAISES
    instead of quietly leaving a read of the pre-launch displacement in a fused
    kernel.

    ``neighbour_names`` is a NORMALISED ``{axis: (down, up)}`` and not any one
    emitter's raw table, because the four certified emitters do not agree on that
    table's shape: the real ones key it by axis LETTER with five members, the
    complex no-PML one by axis INDEX with eight. Each family narrows its own to
    this pair, which is the only part this function uses.

    ``register`` formats the OWN-cell replacement (the value this thread already
    computed) and ``call`` the foreign one, taking ``(partner, coordinates)``.
    Both the plain and the folded emitters compose their four taps from the same
    four index shapes — home, partner-axis down, own-axis up, and the corner —
    which is why one function serves all four families.
    """
    out: List[Tuple[str, str, int]] = [
        (f"g{p}[ii]", register.format(p), -1) for p in range(3)]
    seen = set()
    for component in range(3):
        own = e_terms[component][2]
        own_up = neighbour_names[own][1]
        for offset, partner in enumerate(transverse_partners[component]):
            if not row_mask[2 * component + offset]:
                continue
            partner_down = neighbour_names[partner][0]
            for coordinates in ({partner: partner_down}, {own: own_up},
                                {own: own_up, partner: partner_down}):
                old = f"g{partner}[{index(coordinates)}]"
                if old in seen:
                    continue
                seen.add(old)
                names = [coordinates.get(axis, COORDS[axis]) for axis in range(3)]
                out.append((old, call.format(partner, ", ".join(names)), -1))
    return out


def weld_body(constitutive_source: str, *, decode_end: str, seam: str,
              needles: Sequence[Tuple[str, str, int]],
              anchor: str = "uint idx [[thread_position_in_grid]])\n{\n") -> str:
    """The fused kernel body: the certified constitutive one, seam spliced in.

    The certified source's own body is lifted whole (signature and closing brace
    off), the SEAM is inserted immediately after its decode prologue — which is
    where every index the seam needs first exists and before any displacement is
    read — and every displacement read below is redirected by ``needles``.

    THE FINAL CHECK IS THE ONE THAT MATTERS. After the substitution no ``gN[``
    may survive anywhere in the body: in the fused signature those three
    pointers are the MAGNETIC field, so one missed read is a smooth, plausible,
    entirely wrong answer rather than a compile error.
    """
    if anchor not in constitutive_source:
        raise AssertionError(
            "the certified constitutive source no longer carries the body anchor")
    body = constitutive_source.split(anchor, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified constitutive source does not end with '}'")
    body = body[: -len("}\n")]
    if decode_end not in body:
        raise AssertionError(
            f"the certified constitutive body no longer carries {decode_end.strip()!r}; "
            f"the seam has nowhere to splice")
    head, tail = body.split(decode_end, 1)
    tail = substitute_loads(tail, needles)
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded constitutive half still reads g{target}, which in the "
                f"fused signature is the MAGNETIC field and not a displacement")
    return head + decode_end + seam + tail


def seam_lines(*, value_type: str, forward: str, split_field: bool,
               d_final_arguments: str = "") -> str:
    """The scratch stores and the three final displacements, as kernel text.

    ORDER IS LOAD-BEARING AND IS THE DRIVER'S. ``step_cell`` at this thread's own
    cell gives the RAW stepped auxiliary, which the fills and the wall clear never
    touch (``stepping._fill_symmetry_ghost_cells`` and ``._zero_metal`` both walk
    ``D_CURL_TERMS`` / ``D_COMPONENTS``, never ``fu_*``), so the split-field
    scratch takes it unmodified; the three displacements take the CLOSED FORM,
    which is the same raw value with the seam's three passes resolved per cell.
    """
    lines = [
        "\n    // --- THE SCRATCH WELD: step_D and the three in-seam passes ----",
        "    // Nothing written below is read: the constitutive half re-derives",
        "    // every foreign tap from PRE-LAUNCH state through the same",
        "    // `step_cell`, and the launcher rotates the buffers afterwards.",
    ]
    if split_field:
        lines += [
            f"    step_cell_result own = step_cell({forward});",
            "    n0s[ii] = own.n0; n1s[ii] = own.n1; n2s[ii] = own.n2;",
        ]
    lines += [f"    {value_type} dfin{c} = d_final_{c}({forward}"
              f"{d_final_arguments});" for c in range(3)]
    lines.append("    d0s[ii] = dfin0; d1s[ii] = dfin1; d2s[ii] = dfin2;\n")
    return "\n".join(lines) + "\n"


class ScratchWeldPairPlan:
    """ONE dispatch spanning the D seam, with the post-launch buffer rotation.

    NOT a :class:`.plans.KernelPlan`: that base freezes one argument tuple and
    forbids overriding ``run``, while this seam's whole point is that the D/fu
    argument slots ALTERNATE between two physical buffers. The pattern is
    :class:`.ade_update_p.MetalAdeUpdatePPlan`'s — persistent physical mirrors,
    live role resolution by host identity immediately before every launch, and
    the reference rotation applied only after the launch succeeds — with one
    difference the docstring owns: the array path never rotates D, so the twin
    buffers belong to the PLAN, and the rotation swaps the ENGINE's attribute
    references (``fields.Dx`` <-> the plan's twin) so every later pass, sync and
    read-back sees the freshly written buffer under the engine's own name.

    ``rotated`` maps attribute name -> the twin host array the plan owns.

    THE SIGNATURE ORDER IS THE ARGUMENT ORDER AND IS NOT NEGOTIABLE: every
    rotating WRITE (the scratch) first, in ``rotated_names`` order, then every
    rotating READ (the pre-launch buffer) in the same order, then
    ``static_args`` — the non-rotating pointers and the packed ``Params``
    record. :meth:`run` concatenates exactly those three groups, so a kernel
    that interleaved a static pointer between two rotating ones would bind the
    wrong buffer to every argument after it. The four families' templates are
    all written to that layout for this reason.
    """

    __slots__ = ("residency", "volumes", "fields", "rotated", "rotated_names",
                 "static_args", "launches", "launches_per_run", "_functions",
                 "shape", "codes", "zero_metal", "row_mask", "replaces_sub_steps",
                 "repair_paths", "family")

    performs_device_work = True

    def __init__(self, family: str, residency: Residency, fields: Any,
                 rotated: Mapping[str, Any], functions: Mapping[str, Any],
                 rotated_names: Sequence[str],
                 static_args: Sequence[Any], volumes: Sequence[str],
                 shape: Sequence[int], codes: Sequence[int],
                 zero_metal: Sequence[int], row_mask: Sequence[int],
                 replaces: Sequence[str]) -> None:
        self.family = family
        self.residency = residency
        self.fields = fields
        self.rotated = dict(rotated)  # name -> twin host array (plan-owned)
        self.rotated_names = tuple(rotated_names)
        if set(self.rotated_names) != set(self.rotated):
            raise ValueError(
                f"the rotation order {self.rotated_names} does not name the same "
                f"volumes as the twin table {tuple(sorted(self.rotated))}; the "
                f"launch binds by that order and a mismatch is a wrong buffer")
        self.static_args = tuple(static_args)
        self._functions = dict(functions)
        self.volumes = tuple(dict.fromkeys(volumes))
        self.shape = tuple(int(n) for n in shape)
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(int(flag) for flag in zero_metal)
        self.row_mask = tuple(int(flag) for flag in row_mask)
        self.replaces_sub_steps = tuple(replaces)
        self.repair_paths = ()
        self.launches = 0
        self.launches_per_run = 1

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted(self._functions))

    @property
    def runs(self) -> int:
        return self.launches

    def _resolve(self) -> Tuple[List[Any], List[Any]]:
        """(scratch write tensors, pre-launch read tensors), by CURRENT role."""
        writes: List[Any] = []
        reads: List[Any] = []
        for name in self.rotated_names:
            current = getattr(self.fields, name)
            read = self.residency.tensor_for_host(current)
            write = self.residency.tensor_for_host(self.rotated[name])
            if read is None or write is None:
                raise RuntimeError(
                    f"the {self.family} weld lost the device mirror of {name}; "
                    f"a launch against an unresolved buffer would step garbage")
            if read is write:
                raise RuntimeError(
                    f"the {self.family} weld resolved {name} and its scratch to "
                    f"ONE tensor; the whole design is that nothing written is "
                    f"read, so an aliased pair is refused rather than launched")
            reads.append(read)
            writes.append(write)
        return writes, reads

    def run(self, contract: Optional[str] = None) -> None:
        from .shaders import CONTRACT_OFF  # noqa: PLC0415

        mode = CONTRACT_OFF if contract is None else contract
        function = self._functions.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} variant (it was built with "
                f"{self.variants})")
        writes, reads = self._resolve()
        args: List[Any] = list(writes) + list(reads)
        args.extend(self.static_args)
        self.launches += 1
        function(*args)
        # THE ROTATION — only after the launch call returned. The engine's
        # references move to the freshly written buffers; the pre-launch buffers
        # become the plan's twins and will be next launch's scratch.
        for name in self.rotated_names:
            current = getattr(self.fields, name)
            setattr(self.fields, name, self.rotated[name])
            self.rotated[name] = current

    def describe(self) -> str:
        return (f"{type(self).__name__}(family={self.family!r}, "
                f"shape={self.shape}, codes={self.codes}, "
                f"row_mask={self.row_mask}, variants={self.variants})")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.describe()


# ---------------------------------------------------------------------------
# The plan builder — twins, mirrors, packs and the rotation, in one place
# ---------------------------------------------------------------------------

def scratch_twin(residency: Residency, name: str, host: Any,
                 dtype: Any = None) -> Tuple[Any, Any]:
    """The plan-owned twin of one rotating volume, and its device mirror.

    A FRESH ZERO ARRAY OF THE SAME DTYPE AND SHAPE, registered under a name of
    its own. Zeroing rather than copying is deliberate and is measurable: the
    kernel writes EVERY stored cell of the twin on every launch (the dispatch is
    sized from it and the closed form is total), so a leftover value can only
    surface if that totality breaks — and then it surfaces as a zero plane, which
    a byte comparison catches, rather than as a plausible stale field.

    ``dtype`` is passed through to :meth:`.Residency.mirror`, and a complex
    family MUST pass ``numpy.complex64``: the mirror's default resolution is
    float32, which on a complex host array is a silent cast that discards the
    imaginary part rather than an error.
    """
    import numpy  # noqa: PLC0415

    twin = numpy.zeros_like(numpy.ascontiguousarray(host))
    return twin, residency.mirror(f"weld_scratch:{name}", twin, dtype=dtype)


def plan_scratch_weld(family: str, residency: Residency, fields: Any, *,
                      rotated_names: Sequence[str], static_args: Sequence[Any],
                      functions: Mapping[str, Any], volumes: Sequence[str],
                      shape: Sequence[int], codes: Sequence[int],
                      zero_metal: Sequence[int], row_mask: Sequence[int],
                      replaces: Sequence[str],
                      twins: Mapping[str, Any]) -> "ScratchWeldPairPlan":
    """Assemble the plan once every mirror and pack the family needs exists.

    Kept here rather than in each family module because the ROTATION is the part
    that is easy to get subtly wrong — a twin registered under a name another
    plan already holds, a rotation order that does not match the signature — and
    one home for it is one place to check.
    """
    return ScratchWeldPairPlan(
        family, residency, fields, dict(twins), dict(functions),
        tuple(rotated_names), tuple(static_args), tuple(volumes), tuple(shape),
        tuple(codes), tuple(zero_metal), tuple(row_mask), tuple(replaces))
