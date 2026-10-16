#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fused multiply-add census of a MEEP build: what makes it an identity reference.

    python tools/fma_census.py IMAGE [IMAGE ...] [--out census.json] [--require-none]
    python tools/fma_census.py --python PYTHON [--out census.json] [--require-none]
    python tools/fma_census.py --disassembly FILE[.gz] --format elf-x86_64 [--label NAME]

A MEEP build is an identity reference only if its machine code rounds every
floating-point operation on its own: no multiply and add fused into one rounding.
The compile flags do not establish that. clang contracts ``a*b+c`` written in one
expression unless told ``-ffp-contract=off``; gcc's default is
``-ffp-contract=fast``, and a gcc build emits no fused instruction only because the
baseline x86-64 instruction set has none. So the census reads the machine code.

For every image it disassembles the text (``otool -tvV`` for Mach-O; ``objdump -d
-C --no-show-raw-insn`` for ELF), splits it into functions, and counts the fused
instructions:

* arm64: ``fmadd``, ``fmsub``, ``fnmadd``, ``fnmsub`` (scalar), ``fmla``, ``fmls``
  (vector, every arrangement and the by-element forms), the widening ``fmlal``,
  ``fmlsl`` and the SVE forms;
* x86-64: ``vfmadd*``, ``vfmsub*``, ``vfnmadd*``, ``vfnmsub*``, ``vfmaddsub*``,
  ``vfmsubadd*`` (FMA3 with its 132/213/231 operand orders, AVX-512, FMA4) and the
  AVX512-FP16 complex forms ``vfmaddc*``, ``vfcmaddc*``.

Two functions are exempt: the compiler-rt complex-division builtins ``__divsc3``
and ``__divdc3`` (``___divsc3`` and ``___divdc3`` in Mach-O). The toolchain links
them statically from its own runtime archive, which was compiled when the
toolchain was built, so no flag of the MEEP build reaches them; the field updates
do not call them. Their counts are reported, never judged.

The census also reports, per function, MEEP's field-update kernels: the curl,
B/D and E/H updates, their stride-1 variants, the susceptibility updates and the
source step (``KERNELS`` below). A census that finds none of them in an image
named ``libmeep`` says so, since it then did not read the code it is meant to judge.

``--python PYTHON`` censuses the MEEP images that a process of that interpreter
loads on ``import meep`` (read from the loader, not from the files an
environment happens to hold). ``--disassembly`` reads a disassembly made on
another host (for example ``objdump -d -C --no-show-raw-insn libmeep.so | gzip``
on a Linux reference host), so a census can be judged where the binary is not.

Exit status: 0 when the census ran and, under ``--require-none``, found no fused
instruction outside the exempt builtins; 2 when ``--require-none`` was given and
it found some; 3 when the census could not run (no disassembler, an image it
cannot read, a format it does not know, no MEEP image loaded).

Standard library only; this file is not part of the installed package.
``tools/check_install.py --require-identity-reference`` and
``parity/meep_gpu/timing_records.py meep-build-record`` load it by path.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

SCHEMA = "meep-gpu/fma-census/1"

#: Fused instructions, by architecture, matched on the mnemonic (arm64: the part
#: before an arrangement suffix such as ``.4s``, which ``otool`` attaches).
ARM64_FUSED = frozenset({
    "fmadd", "fmsub", "fnmadd", "fnmsub",            # scalar
    "fmla", "fmls",                                  # vector, by vector or element
    "fmlal", "fmlal2", "fmlsl", "fmlsl2",            # widening (half to single)
    "fmad", "fmsb", "fnmad", "fnmsb", "fnmla", "fnmls",  # SVE
})
X86_FUSED = re.compile(r"^(vfn?m(add|sub)(sub|add)?(132|213|231)?[ps][sdh]"
                       r"|vfc?maddc[ps]h)$")  # the last: AVX512-FP16 complex FMA

#: The compiler-rt complex-division builtins, by their C names.
EXEMPT = ("__divsc3", "__divdc3")

#: MEEP's field-update kernels, on demangled names (the set whose instruction
#: sequences are compared across builds: environments/locks/README.md, "The
#: field-update kernels across builds").
KERNELS = re.compile(
    r"^meep::(step_(curl|beta|bfast|update_EDHB)(_stride1)?"
    r"|\w+_susceptibility::update_P"
    r"|fields_chunk::(step_db|update_eh|update_pols|step_source))\(")

FORMATS = ("mach-o-arm64", "mach-o-x86_64", "elf-x86_64", "elf-arm64")

#: Run in the interpreter under ``--python``: import MEEP, then list the MEEP images
#: the process loaded.
LOADED_PROBE = r"""
import ctypes, json, os, sys
import meep
loaded = []
try:
    with open("/proc/self/maps") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 6 and parts[5].startswith("/"):
                loaded.append(parts[5])
except OSError:
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    for index in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(index)
        if name:
            loaded.append(name.decode("utf-8", "replace"))
images = sorted({os.path.realpath(p) for p in loaded
                 if os.path.basename(p).startswith(("libmeep", "_meep"))})
print("LOADED " + json.dumps({"images": images, "meep": meep.__version__,
                              "meep_file": meep.__file__}))
"""


class CensusError(Exception):
    """The census could not run on an input."""


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_format(path: str) -> str:
    """One of FORMATS, read from the file's header. Raises CensusError otherwise."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError as exc:
        raise CensusError(f"{path}: {exc}") from None
    if head[:4] == b"\x7fELF":
        if head[4] != 2 or head[5] != 1:
            raise CensusError(f"{path}: not a 64-bit little-endian ELF file")
        machine = struct.unpack_from("<H", head, 18)[0]
        known = {62: "elf-x86_64", 183: "elf-arm64"}
        if machine not in known:
            raise CensusError(f"{path}: ELF machine {machine} is not x86-64 or arm64")
        return known[machine]
    if len(head) >= 8 and struct.unpack_from("<I", head, 0)[0] == 0xFEEDFACF:
        cpu = struct.unpack_from("<I", head, 4)[0]
        known = {0x0100000C: "mach-o-arm64", 0x01000007: "mach-o-x86_64"}
        if cpu not in known:
            raise CensusError(f"{path}: Mach-O CPU type {cpu:#x} is not arm64 or x86-64")
        return known[cpu]
    if len(head) >= 4 and struct.unpack_from(">I", head, 0)[0] in (0xCAFEBABE, 0xCAFEBABF):
        machine = os.uname().machine if hasattr(os, "uname") else ""
        return "mach-o-arm64" if machine == "arm64" else "mach-o-x86_64"
    raise CensusError(f"{path}: neither an ELF nor a Mach-O file")


def disassembler(fmt: str):
    """``(argv prefix, label)`` of the tool that disassembles ``fmt`` here."""
    if fmt.startswith("mach-o"):
        tool = shutil.which("otool") or ("/usr/bin/otool" if os.path.exists("/usr/bin/otool")
                                          else None)
        if tool is None:
            raise CensusError("no otool on this host (install the Xcode Command Line Tools)")
        return [tool, "-tvV"], "otool -tvV"
    tool = shutil.which("objdump")
    if tool is None:
        raise CensusError("no objdump on this host (install binutils)")
    return [tool, "-d", "-C", "--no-show-raw-insn"], "objdump -d -C --no-show-raw-insn"


def disassemble(path: str, fmt: str):
    """``(text, how)``: the disassembly of ``path``."""
    argv, how = disassembler(fmt)
    try:
        done = subprocess.run(argv + [path], capture_output=True, text=True, check=False,
                              errors="replace", timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CensusError(f"{how} {path}: {exc}") from None
    if done.returncode != 0 or not done.stdout.strip():
        raise CensusError(f"{how} {path} exited {done.returncode}: "
                          f"{(done.stderr or '').strip()[:300]}")
    return done.stdout, how


_MACHO_HEADER = re.compile(r"^(\S+):$")
_MACHO_INSN = re.compile(r"^[0-9a-f]+\t(\S+)")
_ELF_HEADER = re.compile(r"^[0-9a-f]+ <(.+)>:$")
_ELF_INSN = re.compile(r"^\s*[0-9a-f]+:\s*(.*)$")
_RAW_BYTES = re.compile(r"^(?:[0-9a-f]{2}(?: |$))+\s*")


def functions_of(text: str, fmt: str):
    """``OrderedDict{function name: [mnemonic, ...]}``.

    Mach-O (``otool -tvV``): a function starts at a line ``<symbol>:``, and an
    instruction line is ``<address>\\t<mnemonic>\\t<operands>``. ELF (``objdump
    -d``, GNU or LLVM): a function starts at ``<address> <<symbol>>:``, and an
    instruction line is ``<address>: [raw bytes] <mnemonic> <operands>``. A name
    that repeats (static functions of the same name in two translation units) keeps
    each occurrence, numbered ``#2``, ``#3``.
    """
    funcs: "collections.OrderedDict[str, list]" = collections.OrderedDict()
    current = None
    macho = fmt.startswith("mach-o")
    for line in text.splitlines():
        if macho:
            header = _MACHO_HEADER.match(line)
            if header and not line.startswith(("(", "/")):
                name = header.group(1)
            else:
                name = None
        else:
            header = _ELF_HEADER.match(line)
            name = header.group(1) if header else None
        if name is not None:
            base, number = name, 1
            while name in funcs:
                number += 1
                name = f"{base}#{number}"
            funcs[name] = []
            current = name
            continue
        if current is None:
            continue
        if macho:
            insn = _MACHO_INSN.match(line)
            if insn:
                funcs[current].append(insn.group(1))
            continue
        insn = _ELF_INSN.match(line)
        if not insn:
            continue
        rest = _RAW_BYTES.sub("", insn.group(1).strip())
        token = rest.split(None, 1)[0] if rest else ""
        if token and not token.startswith(("(", "<")):
            funcs[current].append(token)
    return funcs


def is_fused(mnemonic: str, fmt: str) -> bool:
    if fmt.endswith("arm64"):
        return mnemonic.split(".", 1)[0].lower() in ARM64_FUSED
    return bool(X86_FUSED.match(mnemonic.lower()))


def c_name(name: str, fmt: str) -> str:
    """The C-level name: Mach-O prefixes every symbol with one underscore."""
    base = name.split("#", 1)[0]
    if fmt.startswith("mach-o") and base.startswith("_"):
        return base[1:]
    return base


def demangle(names, fmt: str) -> dict:
    """``{name: demangled}``; names that are not Itanium-mangled map to themselves."""
    mangled = [n for n in names if c_name(n, fmt).startswith("_Z")]
    out = {n: c_name(n, fmt) for n in names}
    tool = shutil.which("c++filt") or ("/usr/bin/c++filt"
                                       if os.path.exists("/usr/bin/c++filt") else None)
    if not mangled or tool is None:
        return out
    try:
        done = subprocess.run([tool], input="\n".join(c_name(n, fmt) for n in mangled),
                              capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return out
    lines = done.stdout.splitlines()
    if len(lines) == len(mangled):
        for name, plain in zip(mangled, lines):
            suffix = name[len(name.split("#", 1)[0]):]
            out[name] = plain + suffix
    return out


def census_of_functions(funcs, fmt: str) -> dict:
    """The census of one image's functions (no file access)."""
    per = {}
    for name, ops in funcs.items():
        fused = collections.Counter(op for op in ops if is_fused(op, fmt))
        per[name] = (len(ops), fused)
    exempt = {c: 0 for c in EXEMPT}
    elsewhere = {}
    for name, (_, fused) in per.items():
        total = sum(fused.values())
        if not total:
            continue
        if c_name(name, fmt) in EXEMPT:
            exempt[c_name(name, fmt)] += total
        else:
            elsewhere[name] = fused
    names = demangle(list(per), fmt)
    kernels = {}
    for name, (count, fused) in per.items():
        plain = names[name]
        if KERNELS.match(plain):
            kernels[plain] = {"instructions": count, "fused": sum(fused.values()),
                              "by_op": dict(sorted(fused.items()))}
    fused_total = sum(sum(f.values()) for _, f in per.values())
    ranked = sorted(elsewhere.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    by_op = collections.Counter()
    for _, fused in elsewhere.items():
        by_op.update(fused)
    return {
        "functions": len(per),
        "instructions": sum(count for count, _ in per.values()),
        "fused_total": fused_total,
        "fused_in_exempt": exempt,
        "fused_elsewhere": sum(sum(f.values()) for f in elsewhere.values()),
        "functions_with_fused_elsewhere": len(elsewhere),
        "fused_elsewhere_by_op": dict(sorted(by_op.items())),
        "fused_elsewhere_top": [{"function": names[name], "fused": sum(f.values()),
                                 "by_op": dict(sorted(f.items()))}
                                for name, f in ranked[:40]],
        "kernels_found": len(kernels),
        "kernels_fused": sum(entry["fused"] for entry in kernels.values()),
        "kernels": dict(sorted(kernels.items())),
    }


def census_image(path: str) -> dict:
    """The census of one image file, with its identity."""
    fmt = image_format(path)
    text, how = disassemble(path, fmt)
    entry = {"path": os.path.realpath(path), "sha256": sha256_file(path), "format": fmt,
             "disassembler": how}
    entry.update(census_of_functions(functions_of(text, fmt), fmt))
    return entry


def census_disassembly(path: str, fmt: str, label: str = None) -> dict:
    """The census of a disassembly file (plain or gzip) made elsewhere."""
    if fmt not in FORMATS:
        raise CensusError(f"--format must be one of {', '.join(FORMATS)}, not {fmt!r}")
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        raise CensusError(f"{path}: {exc}") from None
    entry = {"path": label or os.path.realpath(path), "sha256": None, "format": fmt,
             "disassembler": f"read from {os.path.basename(path)} "
                             f"(sha256 {sha256_file(path)})"}
    funcs = functions_of(text, fmt)
    if not funcs:
        raise CensusError(f"{path}: no function found in the disassembly")
    entry.update(census_of_functions(funcs, fmt))
    return entry


def loaded_images(python: str):
    """``(images, facts)``: the MEEP images ``python`` loads on ``import meep``."""
    try:
        done = subprocess.run([python, "-s", "-c", LOADED_PROBE], capture_output=True,
                              text=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CensusError(f"{python}: {exc}") from None
    for line in done.stdout.splitlines():
        if line.startswith("LOADED "):
            facts = json.loads(line[len("LOADED "):])
            if not facts.get("images"):
                raise CensusError(f"{python} imported MEEP but loaded no libmeep/_meep image")
            return facts["images"], facts
    raise CensusError(f"{python} could not import meep: "
                      f"{(done.stderr or done.stdout).strip()[-300:]}")


def verdict(images) -> dict:
    """The census verdict over its images."""
    fused = sum(image["fused_elsewhere"] for image in images)
    functions = sum(image["functions_with_fused_elsewhere"] for image in images)
    exempt = sum(sum(image["fused_in_exempt"].values()) for image in images)
    missing_kernels = [image["path"] for image in images
                       if os.path.basename(image["path"]).startswith("libmeep")
                       and image["kernels_found"] == 0]
    if fused:
        text = (f"{fused} fused multiply-add instruction(s) in {functions} function(s) "
                f"outside the compiler-rt division builtins, over {len(images)} image(s): "
                "this build is not an identity reference")
    else:
        text = (f"no fused multiply-add instruction outside the compiler-rt division "
                f"builtins in {len(images)} of {len(images)} image(s) "
                f"({exempt} inside them, exempt)")
    return {"fused_elsewhere": fused, "functions_with_fused_elsewhere": functions,
            "fused_in_exempt": exempt, "libmeep_without_kernels": missing_kernels,
            "none_outside_exempt": fused == 0 and not missing_kernels, "text": text}


def census(paths=(), python=None, disassemblies=(), fmt=None, label=None,
           progress=say) -> dict:
    """The census record of every input. Raises CensusError when one cannot be read.

    ``progress`` receives one line per input and one per result (``None``: silent).
    """
    progress = progress or (lambda text: None)
    record = {"schema": SCHEMA, "tool": {"path": os.path.realpath(__file__),
                                         "sha256": sha256_file(__file__)},
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "images": []}
    inputs = list(paths)
    if python:
        images, facts = loaded_images(python)
        record["loaded_by"] = {"python": python, **facts}
        inputs += images
    for path in inputs:
        progress(f"census of {path}")
        entry = census_image(path)
        record["images"].append(entry)
        progress(f"  {entry['format']}: {entry['functions']} functions; fused "
            f"{entry['fused_total']} in all, {sum(entry['fused_in_exempt'].values())} in "
            f"the exempt builtins, {entry['fused_elsewhere']} elsewhere in "
            f"{entry['functions_with_fused_elsewhere']} functions; field-update kernels "
            f"{entry['kernels_found']} found, {entry['kernels_fused']} fused")
    for path in disassemblies:
        progress(f"census of the disassembly {path} ({fmt})")
        entry = census_disassembly(path, fmt, label)
        record["images"].append(entry)
        progress(f"  {entry['functions']} functions; fused {entry['fused_total']} in all, "
            f"{entry['fused_elsewhere']} outside the exempt builtins; field-update kernels "
            f"{entry['kernels_found']} found, {entry['kernels_fused']} fused")
    if not record["images"]:
        raise CensusError("nothing to census: give images, --python or --disassembly")
    record["verdict"] = verdict(record["images"])
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="*", help="Mach-O or ELF images to census")
    parser.add_argument("--python", help="census the MEEP images this interpreter loads")
    parser.add_argument("--disassembly", action="append", default=[],
                        help="a disassembly file (plain or .gz) made on another host")
    parser.add_argument("--format", choices=FORMATS, dest="fmt",
                        help="the architecture of --disassembly")
    parser.add_argument("--label", help="the image a --disassembly was made from")
    parser.add_argument("--out", help="write the census record (JSON) here")
    parser.add_argument("--require-none", action="store_true", dest="require_none",
                        help="exit 2 unless no fused instruction lies outside the "
                             "exempt builtins")
    arguments = parser.parse_args(argv)
    if arguments.disassembly and not arguments.fmt:
        parser.error("--disassembly needs --format")
    try:
        record = census(arguments.images, arguments.python, arguments.disassembly,
                        arguments.fmt, arguments.label)
    except CensusError as exc:
        print(f"CENSUS NOT RUN: {exc}", flush=True)
        return 3
    if arguments.out:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
        with open(arguments.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
            handle.write("\n")
    result = record["verdict"]
    for path in result["libmeep_without_kernels"]:
        print(f"NOTE: {path} holds none of MEEP's field-update kernels by name, so the "
              "census did not read the code it judges", flush=True)
    print(f"FMA census: {result['text']}.", flush=True)
    if arguments.require_none and not result["none_outside_exempt"]:
        print("REFUSING: --require-none was given and the census is not clean.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
