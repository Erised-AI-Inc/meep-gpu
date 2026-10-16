"""The own-cell hoist of the certified real-PML singles, and the one inverse every lifter reads.

WHAT THE HOIST IS -- THE REGISTER VIEW. A hoisted single loads every word its thread
reads at ``idx`` into registers in one block after the index decode, calls its CERTIFIED
helper UNCHANGED on those registers (``constitutive_apply(&h_x, &w_x, 0, src_x, ...)``,
``pml_apply(&b_x, &fu_x, 0, curl, ...)``: the helper's ``f``/``fw`` are one-element views
of the registers, ``idx`` is 0), and writes the registers back to global memory after the
last component, in the certified store order. Every load is issued before the first store,
which is how the Triton singles issue them. Nothing about the arithmetic moves: the helper
text is byte-identical and live, so every source mutation keyed on it still lands where it
did, and the right-hand sides, their grouping and the store order are the certified ones.
The inverse below refuses unless the helper reads each own-cell word before it stores it
(so a load issued ahead of the helper reads the same word) and stores them in the
write-back's order.

WHY THIS FILE EXISTS. The fused products LIFT the certified singles' statements --
``constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i]);`` and friends -- and
their own device text keeps that form. They read the certified text through
:func:`unhoisted_kernel_code`, which returns the shipped statement form BYTE FOR BYTE, so a
hoist in a single moves no product's device text.

THE INVERSE IS ANCHORED, NOT A POST-PASS: one load block right after the decode whose
statements are exactly the expected ones in order, one write-back block right before the
closing brace, one call line per component, each exactly once, each refusing by name.

AN EMITTED SINGLE: THE OFF-DIAGONAL ``update_E`` (``update_E_pml_real_offdiag``,
'cuda:off-diagonal'). Its text is not a constant to spell by hand: ``offdiag_emitter``
emits one certified statement-form text per row mask, and the lifters, the NumPy
evaluator and the emitter's corpus digest all read that text. :func:`hoisted_kernel_code`
is the forward of the inverse -- the same three anchors, the spelling of the two singles
Round B shipped -- and ``offdiag_emitter.offdiag_launch_source`` hands its result to
NVRTC. It returns nothing the inverse does not map back to the certified text byte for
byte, so the off-diagonal single keeps the singles' contract: what every lifter reads is
the inverse of what compiles. Its table row is ``update_E_pml_real``'s: the same helper
(``offdiag_emitter.PRELUDE``'s ``constitutive_apply`` is welded to the certified one), the
same prefixes, registers and call form. The folded and dispersive off-diagonal singles are
NOT in the table: they have emitters of their own (``folded_offdiag_kernels``,
``dispersive_offdiag_update_e``) and entry points of their own, and the rule refuses them
by name.

THE FORM IS CHOSEN PER STRUCTURE CHUNK (bit_parity_0_10, 0.10). Each certified helper takes
the cell's membership flag as its last parameter (``dsigw`` for ``constitutive_apply``,
``dsigu`` for ``pml_apply``) and reads and writes its auxiliary only where the flag holds;
``constitutive_apply`` reads the old field only there too. The register view keeps that
contract: every word the helper reads only under the flag is LOADED under it
(``float w_x = in_i_x ? f_w_Hx[idx] : 0.0f;``), and the auxiliary is STORED under it
(``if (in_i_x) f_w_Hx[idx] = w_x;``). The flags are the kernel's chunk-record block
(``bounds_block.bounds_block``), which sits between the index decode and the load block,
and each row names the flag its components read. A text whose helper takes no flag is not
stepped to the per-chunk forms, and the rule refuses it by name rather than guarding its
loads on a flag it does not define.

CUPY-FREE BY CONSTRUCTION: ``conductive_bfast_fused_hd_pair`` reads the certified text by
parsing ``constitutive_kernels.py`` rather than importing it, so the inverse lives where
both kinds of lifter can reach it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = ["KERNELS", "load_block", "writeback_block", "unhoisted_kernel_code",
           "hoisted_kernel_code", "assert_helper_reads_before_it_stores"]

_DECODE_END = "    int i = idx / (ny * nz);\n"
_AXES = ("x", "y", "z")
_COORDINATE = {"x": "i", "y": "j", "z": "k"}

#: kernel -> (helper, field prefix, aux prefix, field register, aux register,
#:            source (register, certified inline read) or None, call indent,
#:            the membership flag each component x, y, z passes the helper)
#:
#: The flag is the region plan's: a constitutive component reads its own axis and a curl
#: component its dsigu axis, each on the component's own Yee sub-lattice there
#: (``bounds_block.constitutive_entries`` / ``curl_entries``).
KERNELS: Dict[str, Tuple] = {
    "update_H_pml_real": ("constitutive_apply", "H", "f_w_H", "h", "w", ("src", "B"), "    ",
                          ("in_i_x", "in_i_y", "in_i_z")),
    "update_E_pml_real": ("constitutive_apply", "E", "f_w_E", "e", "w", None, "    ",
                          ("in_h_x", "in_h_y", "in_h_z")),
    "step_B_pml_real": ("pml_apply", "B", "fu_B", "b", "fu", None, "        ",
                        ("in_h_z", "in_h_x", "in_h_y")),
    "step_D_pml_real": ("pml_apply", "D", "fu_D", "d", "fu", None, "        ",
                        ("in_i_z", "in_i_x", "in_i_y")),
    # Emitted per row mask and hoisted at the launch door by hoisted_kernel_code: the row
    # of update_E_pml_real, whose helper, prefixes and call form it shares.
    "update_E_pml_real_offdiag": ("constitutive_apply", "E", "f_w_E", "e", "w", None, "    ",
                                  ("in_h_x", "in_h_y", "in_h_z")),
}

#: Which of the two own-cell words each helper reads ONLY under its membership flag, and so
#: which the register view loads under it. ``constitutive_apply`` reads neither the
#: auxiliary nor the old field on a plain cell (``f = src``); ``pml_apply`` reads the
#: auxiliary only on the recurrence's cells and the field on every cell.
_GUARDED_LOADS = {"constitutive_apply": ("aux", "field"), "pml_apply": ("aux",)}

#: The comments the forward spelling writes, per helper -- the spelling of the two
#: register-view singles in ``constitutive_kernels.py``. Comments only: the inverse skips
#: comment lines in both blocks.
_LOAD_COMMENT = {
    "constitutive_apply": (
        "    // THE OWN-CELL HOIST (register view): every word this thread reads at idx,\n"
        "    // loaded before the first store. The certified constitutive_apply steps the\n"
        "    // registers (f = &field, fw = &aux, idx = 0); the write-back stores them.\n"
        "    // The auxiliary and the old field are read only where the form reads them.\n"),
    "pml_apply": (
        "    // THE OWN-CELL HOIST (register view): the two words this thread's pml_apply\n"
        "    // reads at idx, loaded before the first store. The certified pml_apply steps\n"
        "    // the registers (f = &field, fu = &fu, idx = 0); the write-back stores them.\n"
        "    // The auxiliary is read only where the recurrence reads it.\n"),
}
_WRITEBACK_COMMENT = ("    // The write-back, in the certified store order "
                      "(auxiliary, then field).\n")

#: The chunk-record block between the decode and the load block: its comment lines and its
#: ``const int`` lines, then one blank line (``bounds_block.bounds_block``).
_BOUNDS_BLOCK = re.compile(r"\n((?:    //[^\n]*\n)*(?:    const int [^\n]*;\n)+)\n")


def _guarded(helper: str, word: str) -> bool:
    return word in _GUARDED_LOADS[helper]


def load_block(kernel: str) -> List[str]:
    """The hoisted loads, in order: the auxiliary then the field, per axis; then the sources.

    A word the helper reads only under its membership flag is loaded under it.
    """
    helper, field, aux, freg, areg, source, _indent, flags = KERNELS[kernel]
    lines = []
    for a, flag in zip(_AXES, flags):
        for word, register, array in (("aux", areg, aux), ("field", freg, field)):
            read = f"{array}{a}[idx]"
            if _guarded(helper, word):
                read = f"{flag} ? {read} : 0.0f"
            lines.append(f"    float {register}_{a} = {read};")
    if source is not None:
        for a in _AXES:
            lines.append(f"    float {source[0]}_{a} = {source[1]}{a}[idx];")
    return lines


def writeback_block(kernel: str) -> List[str]:
    """The write-back, in the certified store order: auxiliary then field, per axis.

    The auxiliary is stored only under the membership flag, as the helper stores it.
    """
    _helper, field, aux, freg, areg, _source, _indent, flags = KERNELS[kernel]
    lines = []
    for a, flag in zip(_AXES, flags):
        lines.append(f"    if ({flag}) {aux}{a}[idx] = {areg}_{a};")
        lines.append(f"    {field}{a}[idx] = {freg}_{a};")
    return lines


def _helper_text(code: str, helper: str) -> str:
    signature = f"__device__ __forceinline__ void {helper}(\n"
    if code.count(signature) != 1:
        raise AssertionError(
            f"the certified text declares {helper} {code.count(signature)} times, not once")
    text = code[code.index(signature):]
    return text[: text.index("\n}\n") + len("\n}\n")]


def assert_helper_reads_before_it_stores(code: str, helper: str) -> None:
    """The hazard half of the argument, read off the certified helper itself.

    The hoist issues the helper's two own-cell words ahead of the helper; that reads the
    same words only if the certified helper reads each BEFORE (or in the same statement
    as) its store to it, and the write-back reproduces the helper's store order only if
    the helper stores the auxiliary first. The register view guards loads and the
    auxiliary store on the cell's membership flag, so the helper must take that flag as
    its last parameter and store the auxiliary only under it. Refuses by name otherwise.
    """
    text = _helper_text(code, helper)
    params = re.search(r"\(\n(.*?)\n\) \{\n", text, re.S).group(1)
    pointers = re.findall(r"float\* __restrict__ (\w+)", params)
    if len(pointers) != 2:
        raise AssertionError(f"{helper} no longer takes exactly two pointers: {pointers!r}")
    last = [part.strip() for part in params.split(",")][-1]
    membership = re.fullmatch(r"int (\w+)", last)
    if membership is None:
        raise AssertionError(
            f"{helper} takes no membership flag (its last parameter is {last!r}): the text "
            f"steps one form on every cell, not MEEP's per-chunk forms, and the register "
            f"view would guard its loads on a flag it does not define")
    flag = membership.group(1)
    field, aux = pointers
    body = text.split(") {\n", 1)[1]
    raw = [line.split("//", 1)[0].strip() for line in body.splitlines()]
    raw = [s for s in raw if s and s != "}"]
    statements = [re.sub(rf"^if \({flag}\) ", "", s) for s in raw]
    stored = [re.match(r"^(\w+)\[idx\] =", s).group(1) for s in statements
              if re.match(rf"^({field}|{aux})\[idx\] =", s)]
    if stored != [aux, field]:
        raise AssertionError(
            f"{helper} stores {stored!r}, not [{aux!r}, {field!r}]; the write-back order "
            f"would not be the certified one")
    if not any(s.startswith(f"if ({flag}) {aux}[idx] =") for s in raw):
        raise AssertionError(
            f"{helper} does not store {aux}[idx] under {flag}; the write-back guards that "
            f"store and would not be the certified one")
    for word in (aux, field):
        store = next(n for n, s in enumerate(statements) if s.startswith(f"{word}[idx] ="))
        reads = [n for n, s in enumerate(statements)
                 if re.search(rf"(?<!\w){word}\[idx\]",
                              s.split("=", 1)[1] if s.startswith(f"{word}[idx] =") else s)]
        if not reads or min(reads) > store:
            raise AssertionError(
                f"{helper} no longer reads {word}[idx] before it stores it; a load issued "
                f"ahead of the helper would not be the certified value")


def _bounds_end(tail: str, name: str) -> int:
    """Where the chunk-record block after the decode ends (one past its blank line)."""
    if tail.count(_DECODE_END) != 1:
        raise AssertionError(f"{name} decodes i {tail.count(_DECODE_END)} times, not once; "
                             f"the load block has no anchor")
    after = tail.index(_DECODE_END) + len(_DECODE_END)
    match = _BOUNDS_BLOCK.match(tail, after)
    if match is None:
        raise AssertionError(
            f"{name} carries no chunk-record block directly after its decode; its membership "
            f"flags are undefined and the load block has no anchor")
    return match.end()


def _check_call_flags(lines: List[str], kernel: str, name: str) -> None:
    flags = KERNELS[kernel][7]
    for line, flag in zip(lines, flags):
        if not line.endswith(f", {flag});"):
            raise AssertionError(
                f"{name}: {line.strip()!r} does not pass its membership flag {flag} last")


def unhoisted_kernel_code(code: str, kernel: str, name: str) -> str:
    """The hoisted single's text in the shipped statement form, byte for byte.

    ``code`` is the certified kernel string NVRTC compiles for the single. Returns it with
    the load block and the write-back block removed and every register-view call restored
    to the certified form. The chunk-record block stays. Every step is exactly once and
    refuses by name.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}, got {kernel!r}")
    helper, field, aux, freg, areg, source, indent, _flags = KERNELS[kernel]
    assert_helper_reads_before_it_stores(code, helper)
    marker = f'extern "C" __global__ void {kernel}('
    if code.count(marker) != 1:
        raise AssertionError(f"{name} does not declare {kernel} exactly once")
    head, tail = code.split(marker, 1)
    # 1. the load block: comment lines + exactly the expected loads + a blank line, right
    #    after the chunk-record block
    after = _bounds_end(tail, name)
    block = re.compile(r"((?:    //[^\n]*\n)*)((?:    float \w+ = [^\n]*;\n)+)\n")
    match = block.match(tail, after)
    if match is None or match.group(2).splitlines() != load_block(kernel):
        raise AssertionError(
            f"{name} carries no load block of exactly {load_block(kernel)!r} directly after "
            f"its chunk-record block; the lifters' inverse has nothing to remove")
    tail = tail[:after] + tail[match.end():]
    # 2. the write-back block: a blank line, its comment, exactly the expected stores,
    #    and the closing brace
    tail_end = re.compile(r"\n((?:    //[^\n]*\n)*)((?:    [^\n]*\w+\[idx\] = \w+;\n)+)}\n$")
    match = tail_end.search(tail)
    if match is None or match.group(2).splitlines() != writeback_block(kernel):
        raise AssertionError(
            f"{name} does not end with the write-back {writeback_block(kernel)!r}; the "
            f"certified store order has no anchor")
    tail = tail[:match.start()] + "}\n"
    # 3. one register-view call per component, restored to the certified form
    restored_lines = []
    for a in _AXES:
        hoisted = f"{indent}{helper}(&{freg}_{a}, &{areg}_{a}, 0, "
        lines = [line for line in tail.splitlines() if line.startswith(hoisted)]
        if len(lines) != 1:
            raise AssertionError(
                f"{name} carries {len(lines)} lines starting {hoisted!r}, not one; the "
                f"lifters' inverse restores exactly one statement per component")
        restored = lines[0].replace(hoisted, f"{indent}{helper}({field}{a}, {aux}{a}, idx, ", 1)
        if source is not None:
            register = f"{helper}({field}{a}, {aux}{a}, idx, {source[0]}_{a}, "
            if restored.count(register) != 1:
                raise AssertionError(
                    f"{name}'s {field}{a} statement does not take {source[0]}_{a} as its "
                    f"source: {lines[0]!r}")
            restored = restored.replace(f"idx, {source[0]}_{a}, ", f"idx, {source[1]}{a}[idx], ", 1)
        restored_lines.append(restored)
        tail = tail.replace(lines[0], restored, 1)
    _check_call_flags(restored_lines, kernel, name)
    registers = [f"{areg}_", f"{freg}_"] + ([f"{source[0]}_"] if source else [])
    for a in _AXES:
        for r in registers:
            if re.search(rf"(?<![\w.]){r}{a}\b", tail):
                raise AssertionError(
                    f"{name} still names the hoisted register {r}{a} after the inverse")
    return head + marker + tail


def hoisted_kernel_code(code: str, kernel: str, name: str) -> str:
    """The register view of a certified statement-form text: the forward of the inverse.

    For a single whose certified text is EMITTED rather than spelt as a constant (the
    off-diagonal ``update_E``, one text per row mask) this is how the hoisted text is
    issued. Three anchored edits and nothing else, each exactly once and each refusing by
    name: the load block after the chunk-record block, one register-view call per
    component, the write-back before the closing brace. The result is returned only if
    :func:`unhoisted_kernel_code` -- which runs the hazard check on the helper first --
    maps it back to ``code`` byte for byte.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}, got {kernel!r}")
    helper, field, aux, freg, areg, source, indent, _flags = KERNELS[kernel]
    assert_helper_reads_before_it_stores(code, helper)
    marker = f'extern "C" __global__ void {kernel}('
    if code.count(marker) != 1:
        raise AssertionError(f"{name} does not declare {kernel} exactly once")
    head, tail = code.split(marker, 1)
    if not tail.endswith("\n}\n"):
        raise AssertionError(f"{name}: {kernel} is not the last thing in its text; the "
                             f"write-back has no anchor")
    # 1. the load block, right after the chunk-record block
    after = _bounds_end(tail, name)
    tail = (tail[:after] + _LOAD_COMMENT[helper]
            + "".join(line + "\n" for line in load_block(kernel)) + "\n" + tail[after:])
    # 2. one certified call per component, rewritten to the register view
    certified_lines = []
    for a in _AXES:
        certified = f"{indent}{helper}({field}{a}, {aux}{a}, idx, "
        lines = [line for line in tail.splitlines() if line.startswith(certified)]
        if len(lines) != 1 or tail.count(lines[0] + "\n") != 1:
            raise AssertionError(
                f"{name} carries {len(lines)} lines starting {certified!r}, not one; the "
                f"hoist rewrites exactly one statement per component")
        certified_lines.append(lines[0])
        hoisted = lines[0].replace(certified, f"{indent}{helper}(&{freg}_{a}, &{areg}_{a}, 0, ", 1)
        if source is not None:
            read = f"0, {source[1]}{a}[idx], "
            if hoisted.count(read) != 1:
                raise AssertionError(
                    f"{name}'s {field}{a} statement does not read its source as "
                    f"{source[1]}{a}[idx]: {lines[0]!r}")
            hoisted = hoisted.replace(read, f"0, {source[0]}_{a}, ", 1)
        tail = tail.replace(lines[0] + "\n", hoisted + "\n", 1)
    _check_call_flags(certified_lines, kernel, name)
    # 3. the write-back, before the closing brace
    tail = (tail[:-len("}\n")] + "\n" + _WRITEBACK_COMMENT
            + "".join(line + "\n" for line in writeback_block(kernel)) + "}\n")
    issued = head + marker + tail
    if unhoisted_kernel_code(issued, kernel, name) != code:
        raise AssertionError(
            f"{name}: the inverse does not map the hoisted text back to the certified text "
            f"byte for byte, so the lifters would not read what compiles; nothing is issued")
    return issued
