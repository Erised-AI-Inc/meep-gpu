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

CUPY-FREE BY CONSTRUCTION: ``conductive_bfast_fused_hd_pair`` reads the certified text by
parsing ``constitutive_kernels.py`` rather than importing it, so the inverse lives where
both kinds of lifter can reach it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = ["KERNELS", "load_block", "writeback_block", "unhoisted_kernel_code",
           "assert_helper_reads_before_it_stores"]

_DECODE_END = "    int i = idx / (ny * nz);\n"
_AXES = ("x", "y", "z")
_COORDINATE = {"x": "i", "y": "j", "z": "k"}

#: kernel -> (helper, field prefix, aux prefix, field register, aux register,
#:            source (register, certified inline read) or None, call indent)
KERNELS: Dict[str, Tuple] = {
    "update_H_pml_real": ("constitutive_apply", "H", "f_w_H", "h", "w", ("src", "B"), "    "),
    "update_E_pml_real": ("constitutive_apply", "E", "f_w_E", "e", "w", None, "    "),
    "step_B_pml_real": ("pml_apply", "B", "fu_B", "b", "fu", None, "        "),
    "step_D_pml_real": ("pml_apply", "D", "fu_D", "d", "fu", None, "        "),
}


def load_block(kernel: str) -> List[str]:
    """The hoisted loads, in order: the auxiliary then the field, per axis; then the sources."""
    _helper, field, aux, freg, areg, source, _indent = KERNELS[kernel]
    lines = []
    for a in _AXES:
        lines.append(f"    float {areg}_{a} = {aux}{a}[idx];")
        lines.append(f"    float {freg}_{a} = {field}{a}[idx];")
    if source is not None:
        for a in _AXES:
            lines.append(f"    float {source[0]}_{a} = {source[1]}{a}[idx];")
    return lines


def writeback_block(kernel: str) -> List[str]:
    """The write-back, in the certified store order: auxiliary then field, per axis."""
    _helper, field, aux, freg, areg, _source, _indent = KERNELS[kernel]
    lines = []
    for a in _AXES:
        lines.append(f"    {aux}{a}[idx] = {areg}_{a};")
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
    the helper stores the auxiliary first. Refuses by name otherwise.
    """
    text = _helper_text(code, helper)
    params = re.search(r"\(\n(.*?)\n\) \{\n", text, re.S).group(1)
    pointers = re.findall(r"float\* __restrict__ (\w+)", params)
    if len(pointers) != 2:
        raise AssertionError(f"{helper} no longer takes exactly two pointers: {pointers!r}")
    field, aux = pointers
    body = text.split(") {\n", 1)[1]
    statements = [line.split("//", 1)[0].strip() for line in body.splitlines()]
    statements = [s for s in statements if s and s != "}"]
    stored = [re.match(r"^(\w+)\[idx\] =", s).group(1) for s in statements
              if re.match(rf"^({field}|{aux})\[idx\] =", s)]
    if stored != [aux, field]:
        raise AssertionError(
            f"{helper} stores {stored!r}, not [{aux!r}, {field!r}]; the write-back order "
            f"would not be the certified one")
    for word in (aux, field):
        store = next(n for n, s in enumerate(statements) if s.startswith(f"{word}[idx] ="))
        reads = [n for n, s in enumerate(statements)
                 if re.search(rf"(?<!\w){word}\[idx\]",
                              s.split("=", 1)[1] if s.startswith(f"{word}[idx] =") else s)]
        if not reads or min(reads) > store:
            raise AssertionError(
                f"{helper} no longer reads {word}[idx] before it stores it; a load issued "
                f"ahead of the helper would not be the certified value")


def unhoisted_kernel_code(code: str, kernel: str, name: str) -> str:
    """The hoisted single's text in the shipped statement form, byte for byte.

    ``code`` is the certified kernel string NVRTC compiles for the single. Returns it with
    the load block and the write-back block removed and every register-view call restored
    to the certified form. Every step is exactly once and refuses by name.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}, got {kernel!r}")
    helper, field, aux, freg, areg, source, indent = KERNELS[kernel]
    assert_helper_reads_before_it_stores(code, helper)
    marker = f'extern "C" __global__ void {kernel}('
    if code.count(marker) != 1:
        raise AssertionError(f"{name} does not declare {kernel} exactly once")
    head, tail = code.split(marker, 1)
    if tail.count(_DECODE_END) != 1:
        raise AssertionError(f"{name} decodes i {tail.count(_DECODE_END)} times, not once; "
                             f"the load block has no anchor")
    # 1. the load block: comment lines + exactly the expected loads + a blank line
    block = re.compile(r"\n((?:    //[^\n]*\n)*)((?:    float \w+ = \w+\[idx\];\n)+)")
    after = tail.index(_DECODE_END) + len(_DECODE_END)
    match = block.match(tail, after)
    if match is None or match.group(2).splitlines() != load_block(kernel):
        raise AssertionError(
            f"{name} carries no load block of exactly {load_block(kernel)!r} directly after "
            f"its decode; the lifters' inverse has nothing to remove")
    tail = tail[:after] + tail[match.end():]
    # 2. the write-back block: a blank line, its comment, exactly the expected stores,
    #    and the closing brace
    tail_end = re.compile(r"\n((?:    //[^\n]*\n)*)((?:    \w+\[idx\] = \w+;\n)+)}\n$")
    match = tail_end.search(tail)
    if match is None or match.group(2).splitlines() != writeback_block(kernel):
        raise AssertionError(
            f"{name} does not end with the write-back {writeback_block(kernel)!r}; the "
            f"certified store order has no anchor")
    tail = tail[:match.start()] + "}\n"
    # 3. one register-view call per component, restored to the certified form
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
        tail = tail.replace(lines[0], restored, 1)
    registers = [f"{areg}_", f"{freg}_"] + ([f"{source[0]}_"] if source else [])
    for a in _AXES:
        for r in registers:
            if re.search(rf"(?<![\w.]){r}{a}\b", tail):
                raise AssertionError(
                    f"{name} still names the hoisted register {r}{a} after the inverse")
    return head + marker + tail
