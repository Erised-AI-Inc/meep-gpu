"""The identity reference: the census, the build record, the digest smoke and the check.

A MEEP is an identity reference when its machine code fuses no multiply-add
(``tools/fma_census.py``), its build record holds that census with the images' sha256
and the host's math library (``timing_records.py meep-build-record``), and its own
words on four cases reproduce the recorded digests (``meep_identity_smoke.py``);
``tools/check_install.py --require-identity-reference`` judges a host's build. These
tests are pure: no MEEP, no GPU, no compiler, no network.

* the census on disassembly text in the three formats it reads (``otool -tvV``, GNU
  ``objdump`` with and without raw bytes, LLVM ``objdump``): what counts as fused on
  each architecture, the exempt builtins, repeated names, the kernel table, the
  verdict, the file-format reader and the command line on a gzip disassembly;
* the build record's contraction field and ``--require-identity`` as exit statuses;
* the build scripts: the census at step 8 (judged for a reference, recorded for a
  timing control), the record at step 9, the digest check at step 10, the identity
  verdicts at step 11, the identity flags of each platform and the --timing-control
  argument, the Linux ``-march`` guard run against stand-in compilers, and the native
  wrapper's role;
* the digest smoke's words, digests, planted flip and record, the committed record's
  own mint claims, and its refusals;
* the check's judge over synthetic facts (every point that can fail, and only those),
  the child against a stand-in MEEP, and the check stopping on the identity reference
  before any other step runs.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import pathlib
import stat
import struct
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "parity" / "meep_gpu"
SCRIPTS = {
    "osx-arm64": HARNESS / "build_meep_133_macos.sh",
    "linux-64": HARNESS / "build_meep_133_linux.sh",
}
NATIVE_SCRIPT = HARNESS / "build_meep_133_native.sh"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census_tool = _load("fma_census_under_test", ROOT / "tools" / "fma_census.py")
check_install = _load("check_install_identity_under_test", ROOT / "tools" / "check_install.py")
reference_tool = _load("reference_environment_identity_under_test",
                       ROOT / "tools" / "compare_reference_environment.py")
records = _load("timing_records_under_test", HARNESS / "timing_records.py")
smoke = _load("meep_identity_smoke_under_test", HARNESS / "meep_identity_smoke.py")


# --------------------------------------------------------------------------
# The census
# --------------------------------------------------------------------------

OTOOL = """/opt/env/lib/libmeep.37.dylib:
(__TEXT,__text) section
__ZN4meep9step_curlEPfNS_9componentEPKfS3_llS3_S3_RKNS_11grid_volumeEdNS_9directionES3_S3_S3_S3_S3_S3_S3_:
0000000000001000\tfmul\ts0, s1, s2
0000000000001004\tfmadd\ts0, s1, s2, s3
0000000000001008\tfmla.4s\tv0, v1, v2
000000000000100c\tret
___divsc3:
0000000000002000\tfmadd\ts0, s1, s2, s3
0000000000002004\tfnmsub\ts1, s2, s3, s4
__ZN4meep6helperEv:
0000000000003000\tfmov\ts0, #1.00000000
0000000000003004\tfmaxnm\ts0, s0, s1
_ltmp0:
0000000000004000\tfmls.2d\tv0, v1, v2[0]
_ltmp0:
0000000000004100\tfmsub\td0, d1, d2, d3
"""

GNU_RAW = """
libmeep.so.37.0.0:     file format elf64-x86-64


Disassembly of section .text:

0000000000001130 <meep::step_curl(float*, meep::component, float const*, float const*, long, long, float const*, float const*, meep::grid_volume const&, double, meep::direction, float const*, float const*, float const*, float const*, float const*, float const*, float const*)>:
    1130:\tc4 e2 71 a9 c2       \tvfmadd213ss %xmm2,%xmm1,%xmm0
    1135:\tf3 0f 59 c1          \tmulss  %xmm1,%xmm0
    1139:\tc3                   \tret
    113a:\t66 0f 1f 44 00 00    \tnopw   0x0(%rax,%rax,1)

0000000000001140 <__divsc3>:
    1140:\tc4 e2 79 b9 c1       \tvfmadd231ss %xmm1,%xmm0,%xmm0
    1145:\tc3                   \tret

0000000000001150 <g>:
    1150:\t48 b8 00 00 00 00 00 \tmovabs $0x0,%rax
    1157:\t00 00 00
    115a:\tc4 e2 69 ac c1       \tvfnmadd213ps %xmm1,%xmm2,%xmm0
    115f:\tc3                   \tret
"""

GNU_PLAIN = """
0000000000001130 <meep::fields_chunk::update_eh(meep::field_type, bool)>:
    1130:\tmulss  %xmm1,%xmm0
    1134:\taddss  %xmm2,%xmm0
    1138:\tret

0000000000001140 <__divdc3>:
    1140:\tdivsd  %xmm1,%xmm0
    1144:\tret
"""

LLVM = """
t_O2_haswell.o:\tfile format elf64-x86-64

Disassembly of section .text:

0000000000000000 <f>:
       0:      \tvfmadd213ss\t%xmm2, %xmm1, %xmm0 # xmm0 = (xmm1 * xmm0) + xmm2
       5:      \tretq
0000000000000010 <g>:
      10:      \ttestl\t%edx, %edx
      12:      \tvfmadd213ps\t%ymm2, %ymm1, %ymm0
"""


def plain_names(monkeypatch, table):
    monkeypatch.setattr(census_tool, "demangle",
                        lambda names, fmt: {n: table.get(n, census_tool.c_name(n, fmt))
                                            for n in names})


@pytest.mark.parametrize("op, fused", [
    ("fmadd", True), ("fmsub", True), ("fnmadd", True), ("fnmsub", True),
    ("fmla", True), ("fmla.4s", True), ("fmla.2d", True), ("fmla.s", True),
    ("fmls.4s", True), ("fmlal2", True), ("fmad", True),
    ("fmul", False), ("fmul.4s", False), ("fnmul", False), ("fmov", False),
    ("fmaxnm", False), ("fminnm.2d", False), ("fadd", False)])
def test_what_counts_as_fused_on_arm64(op, fused):
    assert census_tool.is_fused(op, "mach-o-arm64") is fused
    assert census_tool.is_fused(op, "elf-arm64") is fused


@pytest.mark.parametrize("op, fused", [
    ("vfmadd231ss", True), ("vfmadd132pd", True), ("vfmadd213ps", True),
    ("vfmsub231sd", True), ("vfnmadd213ps", True), ("vfnmsub132ss", True),
    ("vfmaddsub132pd", True), ("vfmsubadd231ps", True), ("vfmaddss", True),
    ("vfmadd231sh", True), ("vfmaddcph", True), ("vfcmaddcsh", True),
    ("vmulss", False), ("mulss", False), ("addss", False), ("fmul", False),
    ("vfmulcph", False), ("vfixupimmps", False), ("vfpclassps", False)])
def test_what_counts_as_fused_on_x86_64(op, fused):
    assert census_tool.is_fused(op, "elf-x86_64") is fused


def test_otool_text_splits_into_functions_and_keeps_repeated_names():
    funcs = census_tool.functions_of(OTOOL, "mach-o-arm64")
    assert list(funcs)[0].startswith("__ZN4meep9step_curl")
    assert "/opt/env/lib/libmeep.37.dylib" not in funcs
    assert funcs["___divsc3"] == ["fmadd", "fnmsub"]
    assert funcs["_ltmp0"] == ["fmls.2d"] and funcs["_ltmp0#2"] == ["fmsub"]


def test_the_census_of_otool_text_exempts_the_division_builtins(monkeypatch):
    curl = ("meep::step_curl(float*, meep::component, float const*, float const*, long, "
            "long, float const*, float const*, meep::grid_volume const&, double, "
            "meep::direction, float const*, float const*, float const*, float const*, "
            "float const*, float const*, float const*)")
    plain_names(monkeypatch, {
        "__ZN4meep9step_curlEPfNS_9componentEPKfS3_llS3_S3_RKNS_11grid_volumeEdNS_9directionES3_S3_S3_S3_S3_S3_S3_": curl,
        "__ZN4meep6helperEv": "meep::helper()"})
    result = census_tool.census_of_functions(
        census_tool.functions_of(OTOOL, "mach-o-arm64"), "mach-o-arm64")
    assert result["functions"] == 5
    assert result["fused_total"] == 6
    assert result["fused_in_exempt"] == {"__divsc3": 2, "__divdc3": 0}
    assert result["fused_elsewhere"] == 4
    assert result["functions_with_fused_elsewhere"] == 3
    assert result["fused_elsewhere_by_op"] == {"fmadd": 1, "fmla.4s": 1, "fmls.2d": 1,
                                               "fmsub": 1}
    assert result["kernels_found"] == 1
    assert result["kernels"][curl] == {"instructions": 4, "fused": 2,
                                       "by_op": {"fmadd": 1, "fmla.4s": 1}}


@pytest.mark.parametrize("text, fused_elsewhere, exempt, kernels", [
    (GNU_RAW, 2, 1, 1), (GNU_PLAIN, 0, 0, 1), (LLVM, 2, 0, 0)])
def test_the_census_reads_gnu_and_llvm_objdump_text(text, fused_elsewhere, exempt, kernels):
    funcs = census_tool.functions_of(text, "elf-x86_64")
    result = census_tool.census_of_functions(funcs, "elf-x86_64")
    assert result["fused_elsewhere"] == fused_elsewhere
    assert sum(result["fused_in_exempt"].values()) == exempt
    assert result["kernels_found"] == kernels
    if text is GNU_RAW:
        # Raw bytes are not mistaken for mnemonics, and a continuation line of bytes
        # adds no instruction.
        assert funcs["g"] == ["movabs", "vfnmadd213ps", "ret"]


def test_the_verdict_is_clean_only_without_fused_ops_and_with_the_kernels_read():
    clean = {"path": "/x/libmeep.so", "fused_elsewhere": 0, "functions_with_fused_elsewhere": 0,
             "fused_in_exempt": {"__divsc3": 9, "__divdc3": 9}, "kernels_found": 16}
    module = dict(clean, path="/x/_meep.so", fused_in_exempt={"__divsc3": 0, "__divdc3": 0},
                  kernels_found=0)
    verdict = census_tool.verdict([clean, module])
    assert verdict["none_outside_exempt"] is True and verdict["fused_in_exempt"] == 18
    assert "in 2 of 2 image(s)" in verdict["text"]
    dirty = dict(clean, fused_elsewhere=3, functions_with_fused_elsewhere=2)
    assert census_tool.verdict([dirty, module])["none_outside_exempt"] is False
    blind = dict(clean, kernels_found=0)
    result = census_tool.verdict([blind])
    assert result["none_outside_exempt"] is False
    assert result["libmeep_without_kernels"] == ["/x/libmeep.so"]


def _write(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    return path


def test_the_image_format_is_read_from_the_header(tmp_path):
    elf = bytearray(64)
    elf[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", elf, 18, 62)
    assert census_tool.image_format(str(_write(tmp_path / "x86.so", bytes(elf)))) == "elf-x86_64"
    struct.pack_into("<H", elf, 18, 183)
    assert census_tool.image_format(str(_write(tmp_path / "arm.so", bytes(elf)))) == "elf-arm64"
    macho = struct.pack("<II", 0xFEEDFACF, 0x0100000C) + bytes(24)
    assert census_tool.image_format(str(_write(tmp_path / "m.dylib", macho))) == "mach-o-arm64"
    for name, data in (("text.txt", b"not an image"), ("elf32.so", b"\x7fELF\x01\x01" + bytes(58))):
        with pytest.raises(census_tool.CensusError):
            census_tool.image_format(str(_write(tmp_path / name, data)))
    with pytest.raises(census_tool.CensusError):
        census_tool.image_format(str(tmp_path / "absent.so"))


def run_census(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "fma_census.py"), *arguments],
                          capture_output=True, text=True, timeout=120, check=False)


def test_the_command_line_judges_a_disassembly_made_elsewhere(tmp_path):
    dirty = tmp_path / "dirty.dis.gz"
    with gzip.open(dirty, "wt") as handle:
        handle.write(GNU_RAW)
    clean = tmp_path / "clean.dis"
    clean.write_text(GNU_PLAIN, encoding="utf-8")
    out = tmp_path / "census.json"
    done = run_census("--disassembly", str(dirty), "--format", "elf-x86_64",
                      "--label", "libmeep.so.37.0.0", "--out", str(out), "--require-none")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "REFUSING" in done.stdout
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema"] == census_tool.SCHEMA
    assert record["images"][0]["path"] == "libmeep.so.37.0.0"
    assert record["verdict"]["fused_elsewhere"] == 2
    done = run_census("--disassembly", str(clean), "--format", "elf-x86_64", "--require-none")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no fused multiply-add instruction outside" in done.stdout
    assert run_census("--disassembly", str(dirty), "--format", "elf-x86_64").returncode == 0
    assert run_census(str(tmp_path / "absent.so")).returncode == 3
    assert run_census("--disassembly", str(clean)).returncode == 2  # argparse: no --format


# --------------------------------------------------------------------------
# The build record
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cflags, cxxflags, expected", [
    ("-O2 -ffp-contract=off -fno-fast-math", "-O2 -ffp-contract=off -fno-fast-math", "off"),
    ("-g -O2", "-g -O2 -fopenmp", "compiler default"),
    ("-O2 -ffp-contract=on -ffp-contract=off", "-ffp-contract=off", "off"),
    ("-O2 -ffp-contract=off", "-g -O2", "mixed"),
    ("-Ofast", "-O3 -ffast-math", "fast"),
    ("-O3 -mcpu=native", "-O3 -mcpu=native", "compiler default")])
def test_the_record_names_the_contraction_the_flags_ask_for(cflags, cxxflags, expected):
    assert records.fp_contract({"CFLAGS": cflags, "CXXFLAGS": cxxflags}) == expected


def test_the_record_does_not_guess_a_contraction_it_could_not_read():
    assert records.fp_contract({"CXXFLAGS": "-O2"}) is None


def _record(clean=True, libm=True):
    return {"probe": {"meep": "1.33.0", "single_precision": True},
            "fma_census": {"verdict": {"none_outside_exempt": clean, "text": "t"}},
            "libm": {"kind": "libsystem_m", "uuid": "U"} if libm else None}


@pytest.mark.parametrize("clean, libm, status", [
    (True, True, 0), (False, True, 4), (True, False, 4)])
def test_require_identity_refuses_a_dirty_census_or_an_unread_libm(
        tmp_path, monkeypatch, clean, libm, status):
    monkeypatch.setattr(records, "meep_build_record", lambda *a, **k: _record(clean, libm))
    out = tmp_path / "build_record.json"
    code = records.main(["meep-build-record", "--python", sys.executable, "--require-single",
                         "--require-identity", "--out", str(out)])
    assert code == status
    assert json.loads(out.read_text(encoding="utf-8"))["fma_census"]["verdict"][
        "none_outside_exempt"] is clean
    assert records.main(["meep-build-record", "--python", sys.executable,
                         "--out", str(out)]) == 0


def test_a_recorder_that_stops_does_not_meet_a_requirement(tmp_path, monkeypatch):
    def broken(*arguments, **keywords):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(records, "meep_build_record", broken)
    out = str(tmp_path / "r.json")
    assert records.main(["meep-build-record", "--python", "p", "--require-identity",
                         "--out", out]) == 4
    assert records.main(["meep-build-record", "--python", "p", "--out", out]) == 0


# --------------------------------------------------------------------------
# The build scripts
# --------------------------------------------------------------------------

def step(text: str, start: str, end: str) -> str:
    return text[text.index(start):text.index(end, text.index(start))]


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_step_8_judges_the_census_of_a_reference_and_records_a_timing_controls(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    eight = step(text, 'say "8.', 'say "9.')
    census = eight[eight.index("# MANDATORY: the fused multiply-add census"):]
    assert eight.index("compare_reference_environment.py") < eight.index("fma_census.py")
    timing, reference = census.split("else", 1)
    assert '"$REPO_ROOT/tools/fma_census.py" --python "$ENV_PREFIX/bin/python"' in timing
    assert 'if [ "$ROLE" = timing-control ]' in timing and "--require-none" not in timing
    assert "--require-none" in reference.split("fi", 1)[0]
    assert '--out "$RECORD_DIR/fma_census.json"' in timing and \
        '--out "$RECORD_DIR/fma_census.json"' in reference


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_steps_9_and_10_record_the_build_and_check_its_words(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    nine = step(text, 'say "9.', 'say "10.')
    reference, control = nine.split("else", 1)
    assert 'if [ "$ROLE" = reference ]' in reference
    assert "RECORD_OPTIONS=(--require-identity)" in reference
    assert 'helper compiler-facts' in control.split("fi", 1)[0]
    assert 'RECORD_OPTIONS=(--extra-json "$RECORD_DIR/native.json")' in control
    assert "meep-build-record" in nine and '--require-single "${RECORD_OPTIONS[@]}"' in nine
    assert '--out "$RECORD_DIR/build_record.json"' in nine
    ten = step(text, 'say "10.', 'say "DONE"')
    run = ten.index('meep_identity_smoke.py" \\\n    --mode keep')
    assert ten.index('"$ROLE" = timing-control') < ten.index("else") < run


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_step_11_judges_both_modes_with_check_install(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    eleven = step(text, 'say "11.', 'say "12.')
    assert eleven.index('"$ROLE" = timing-control') < eleven.index("for MODE in keep flush")
    assert 'OMP_NUM_THREADS=1 helper identity "$REPO_ROOT/tools/check_install.py" flush' in eleven
    assert 'helper identity "$REPO_ROOT/tools/check_install.py" keep' in eleven
    assert 'echo "$STATUS" > "$RECORD_DIR/identity_$MODE.status"' in eleven
    pending = '[ "$MODE" = flush ] && [ "$STATUS" = 3 ]' in eleven
    # Apple silicon reaches flush through the FZ bit; until check_install drives it, its
    # named refusal alone is recorded as pending there, and nowhere else is anything but a
    # pass
    assert pending is (platform_tag == "osx-arm64"), platform_tag


RECORD_HELPER = _load("build_meep_133_record_for_identity", HARNESS / "build_meep_133_record.py")


def test_check_install_offers_the_identity_verdict_the_build_takes_alone():
    check_install = _load("check_install_for_the_build_test", ROOT / "tools" / "check_install.py")
    assert callable(check_install.identity_check)


@pytest.mark.parametrize("mode, reasons, status", [
    ("keep", None, 0),
    ("flush", ["MEEP's subnormal switch has no effect on this host, so it has no flush "
               "reference (only keep; Apple silicon is such a host)"], 3),
    ("flush", ["the census of the loaded images found 3 fused multiply-add instructions",
               "MEEP's subnormal switch has no effect on this host, so it has no flush "
               "reference (only keep; Apple silicon is such a host)"], 1),
    ("keep", ["its build record pins no math library"], 1),
])
def test_the_builds_identity_verdict_is_pass_pending_or_fail(tmp_path, mode, reasons, status):
    fake = tmp_path / "check_install.py"
    failure = ("None" if reasons is None else
               "Failure(" + repr(f"--require-identity-reference {mode} was given and "
                                 + "; and ".join(reasons) + ".") + ")")
    fake.write_text("class Failure:\n    def __init__(self, reason):\n        self.reason = reason\n"
                    "def identity_check(arguments):\n"
                    f"    return ['   judged {mode}'], {failure}\n", encoding="utf-8")
    got, lines = RECORD_HELPER.identity_verdict(str(fake), mode)
    assert got == status
    assert lines[0] == f"MEEP identity reference (--require-identity-reference {mode})"
    assert lines[-1].startswith("PASSED" if status == 0 else "FAILED")


def run_bash(program: str, env=None):
    return subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False, env=env)


def plan_of(platform_tag: str, *arguments: str):
    """The flags, role and default prefix name a platform script plans for ``arguments``."""
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    block = step(text, "IDENTITY_FLAGS=", "ENV_NAME=")
    quoted = " ".join("'" + argument.replace("'", "'\\''") + "'" for argument in arguments)
    program = "\n".join(["set -euo pipefail", f"set -- {quoted}", block,
                         'echo "ROLE=$ROLE|FLAGS=$FLAGS|NAME=$DEFAULT_NAME"'])
    return run_bash(program, env={key: value for key, value in os.environ.items()
                                  if key not in ("CFLAGS", "CXXFLAGS")})


@pytest.mark.parametrize("platform_tag, flags", [
    ("osx-arm64", "-O2 -ffp-contract=off -fno-fast-math"),
    ("linux-64", "-O2 -ffp-contract=off -fno-fast-math -march=x86-64 -mtune=generic")])
def test_a_reference_compiles_without_contraction_and_a_timing_control_as_told(platform_tag,
                                                                              flags):
    done = plan_of(platform_tag)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"ROLE=reference|FLAGS={flags}|NAME=meep-gpu-ref"
    done = plan_of(platform_tag, "--timing-control", "-O3 -mcpu=native")
    assert done.stdout.strip() == "ROLE=timing-control|FLAGS=-O3 -mcpu=native|NAME=meep-gpu-ref-native"
    for arguments in (("--timing-control",), ("--timing-control", ""), ("--native",)):
        done = plan_of(platform_tag, *arguments)
        assert done.returncode == 2 and "REFUSING" in done.stdout, arguments


def march_guard(tmp_path, march, role="reference"):
    # Like gcc, the stand-in writes on after the -march line (well past a pipe's buffer)
    # and fails when its reader has gone: a reader that stops early stops the build.
    compiler = tmp_path / "x86_64-conda-linux-gnu-c++"
    compiler.write_text("#!/bin/sh\nprintf 'The following options are target specific:\\n'\n"
                        "printf '  -march=                     \\t\\t%s\\n' "
                        f"'{march}'\n"
                        "i=0\nwhile [ $i -lt 4000 ]; do printf '  -m%078d\\n' $i || exit 2; "
                        "i=$((i + 1)); done\n", encoding="utf-8")
    compiler.chmod(compiler.stat().st_mode | stat.S_IEXEC)
    text = SCRIPTS["linux-64"].read_text(encoding="utf-8")
    block = step(text, "# A reference has no fused multiply-add because x86-64 has none",
                 "# NO separate build/ directory")
    program = "\n".join(["set -euo pipefail", f'ROLE="{role}"', f'CCX="{compiler}"',
                         f'CXXX="{compiler}"', block])
    return run_bash(program, env={key: value for key, value in os.environ.items()
                                  if key not in ("CFLAGS", "CXXFLAGS")})


def test_the_linux_script_refuses_a_march_with_fused_multiply_add(tmp_path):
    done = march_guard(tmp_path, "x86-64")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("-march resolves to x86-64") == 2
    done = march_guard(tmp_path, "haswell")
    assert done.returncode == 1
    assert "REFUSING: a reference is compiled for -march=x86-64" in done.stdout
    assert march_guard(tmp_path, "haswell", role="timing-control").returncode == 0


def test_the_native_build_is_a_timing_control():
    text = NATIVE_SCRIPT.read_text(encoding="utf-8")
    assert 'bash "$DELEGATE" --timing-control "$FLAGS"' in text
    assert "MEEP_BUILD_ROLE" not in text


@pytest.mark.parametrize("script", [*SCRIPTS.values(), NATIVE_SCRIPT])
def test_the_build_scripts_parse(script):
    done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------
# The digest smoke
# --------------------------------------------------------------------------

def test_a_lattice_is_three_stride_two_runs():
    axes = [[-199, -197, -195], [-2, 0], [0]]
    compact = smoke.compact_lattice(axes)
    assert compact == [[-199, 3], [-2, 2], [0, 1]]
    assert smoke.expand_lattice(compact) == tuple(axes)
    with pytest.raises(ValueError):
        smoke.compact_lattice([[0, 1, 3], [0], [0]])
    with pytest.raises(ValueError):
        smoke.expand_lattice([[0, 2], [0, 0], [0, 1]])


def test_the_digest_is_of_little_endian_float32_words():
    words = np.array([[1.0, -0.0], [2.5e-39, np.float32(3.0)]], dtype=np.float32)
    import hashlib

    assert smoke.digest(words) == hashlib.sha256(words.astype("<f4").tobytes()).hexdigest()
    assert smoke.digest(words.astype(">f4")) == smoke.digest(words)
    assert smoke.digest(np.zeros(2, np.float32)) != smoke.digest(np.array([0.0, -0.0], np.float32))


def test_the_planted_flip_moves_one_word_by_one_ulp():
    words = np.array([0.0, 0.0, 1.5, -2.0], dtype=np.float32)
    flipped = smoke.flip_one_word(words)
    changed = np.flatnonzero(flipped.view(np.uint32) != words.view(np.uint32))
    assert changed.tolist() == [2]
    assert int(flipped.view(np.uint32)[2]) ^ int(words.view(np.uint32)[2]) == 1
    assert smoke.digest(flipped) != smoke.digest(words)
    assert smoke.flip_one_word(np.zeros(3, np.float32)).view(np.uint32).tolist() == [1, 0, 0]


def test_the_word_compare_separates_values_from_the_sign_of_zero():
    ours = np.array([0.0, -0.0, 1.0, 2.0, np.nan], dtype=np.float32)
    theirs = np.array([0.0, 0.0, 1.0, 3.0, np.nan], dtype=np.float32)
    assert smoke.word_compare(ours, theirs) == {"words": 5, "equal": 3, "value_differ": 1,
                                                "sign_of_zero_only": 1}


def test_the_committed_record_holds_its_own_mint_claims():
    record = smoke.load_record(smoke.RECORD)
    assert record["schema"] == smoke.SCHEMA and record["mode"] == "keep"
    assert sorted(record["cases"]) == sorted(smoke.CASES)
    shapes = {"a": (200, 200, 1), "b": (200, 200, 1), "c": (39, 18, 19), "c_prime": (39, 18, 19)}
    for case, entry in record["cases"].items():
        assert tuple(entry["steps"]) == smoke.SCHEDULE[case]
        assert tuple(axis[1] for axis in entry["lattice"]["Ex"]) == shapes[case]
        for step_digests in entry["digests"].values():
            assert sorted(step_digests) == sorted(smoke.FIELD_ORDER)
            assert all(len(value) == 64 and int(value, 16) >= 0
                       for value in step_digests.values())
        minted = record["minted"]["evidence"]["cases"][case]
        cells = np.prod(shapes[case])
        assert minted["words_compared"] == cells * 12 * len(entry["steps"])
        assert minted["words_equal"] == minted["words_compared"]
        assert minted["value_differ"] == minted["sign_of_zero_only"] == 0
        assert minted["second_process_digests_equal"] == minted["digests"] == 12 * len(entry["steps"])
        assert minted["planted_flip_caught"] is True
        assert sorted(minted["dumps"], key=int) == [str(s) for s in entry["steps"]]
    assert record["minted"]["meep"] == {"version": "1.33.0", "single_precision": True}
    assert record["minted"]["subnormals"]["mode"] == "keep"


def test_the_record_carries_no_home_path():
    text = pathlib.Path(smoke.RECORD).read_text(encoding="utf-8")
    assert not [part for part in ("Users", "home", "private", "tmp")
                if "/" + part + "/" in text]


def test_a_record_of_another_schema_or_with_missing_digests_is_refused(tmp_path):
    record = json.loads(pathlib.Path(smoke.RECORD).read_text(encoding="utf-8"))
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(record, schema="meep-gpu/meep-identity-smoke/0")),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="is not a"):
        smoke.load_record(str(other))
    del record["cases"]["a"]["digests"]["200"]
    short = tmp_path / "short.json"
    short.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover its steps"):
        smoke.load_record(str(short))


def test_the_digest_compare_names_the_first_difference():
    expected = {"0": {name: "a" for name in smoke.FIELD_ORDER},
                "20": {name: "b" for name in smoke.FIELD_ORDER}}
    found = json.loads(json.dumps(expected))
    assert smoke.compare_digests(expected, found) == (24, 24, None)
    found["20"]["Hy"] = "c"
    assert smoke.compare_digests(expected, found) == (23, 24, "step 20 Hy")


def test_a_flush_run_is_refused_by_a_keep_record(capsys):
    assert smoke.main(["--mode", "flush"]) == 3
    assert "no flush set" in capsys.readouterr().out


def evidence_tree(tmp_path, tamper=False):
    leg = tmp_path / "a" / "coff" / "pair20"
    (leg / "dumps").mkdir(parents=True)
    words = np.arange(6, dtype="<f4").reshape(2, 3, 1)
    data = words.tobytes()
    (leg / "dumps" / "meep_raw_0001.bin").write_bytes(data[:-1] + b"\x00" if tamper else data)
    import hashlib

    index = {"case": "a", "build": "coff",
             "lattice": {name: [[0, 2], [0, 2, 4], [0]] for name in smoke.FIELD_ORDER},
             "dumps": [{"file": "dumps/meep_raw_0001.bin", "sha256": hashlib.sha256(data).hexdigest(),
                        "arrays": {"Ex": {"offset": 0, "shape": [2, 3, 1], "dtype": "<f4"}}}]}
    (leg / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return words


def test_evidence_is_read_only_through_its_index_digest(tmp_path):
    words = evidence_tree(tmp_path)
    dumps = smoke.evidence_dumps(str(tmp_path), "a", "coff")
    assert sorted(dumps) == [1]
    assert smoke.evidence_lattice(str(tmp_path), "a", "coff")["Hz"] == [[0, 2], [0, 3], [0, 1]]
    arrays = smoke.read_evidence(*dumps[1])
    assert np.array_equal(arrays["Ex"], words)
    evidence_tree(tmp_path / "bad", tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        smoke.read_evidence(*smoke.evidence_dumps(str(tmp_path / "bad"), "a", "coff")[1])


def test_a_committed_path_is_kept_below_the_archive():
    assert smoke.portable("/x/y/parity/meep_gpu/results/probe/run/dumps") == "results/probe/run/dumps"
    assert smoke.portable("/x/y/dumps") == "dumps"


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------

LIBMEEP = "/opt/env/lib/libmeep.37.dylib"
MODULE = "/opt/env/lib/python3.12/site-packages/meep/_meep.37.so"
IMAGES = {LIBMEEP: "1" * 64, MODULE: "2" * 64}
LIBM = {"kind": "libsystem_m", "uuid": "42E15F5C-02B8-3B56-9889-5B59CADD45EB",
        "macos": "26.2", "os_build": "25C56"}
EXPECTED_MEEP = {"version": "1.33.0", "single_precision": True, "mpi": True}


def census_of(images, fused=0):
    return {"images": [{"sha256": sha} for sha in sorted(images.values())],
            "verdict": {"none_outside_exempt": fused == 0, "fused_elsewhere": fused,
                        "text": f"{fused} fused"}}


def payload(mode="keep", measured=None, threads=None, libm=LIBM, meep=None):
    return {"mode": mode, "meep": meep or dict(EXPECTED_MEEP, file="/opt/env/meep/__init__.py"),
            "images": dict(IMAGES), "libm": dict(libm) if libm else libm,
            "subnormals": {"mode": measured or mode}, "subnormals_after_import": {"mode": "keep"},
            "omp_num_threads": threads}


def record(fused=0, libm=LIBM, images=IMAGES):
    return {"sha256": dict(images), "fma_census": census_of(images, fused),
            "fp_contract": "off", "libm": dict(libm) if libm else None}


def judge(mode, facts, rec=None, census=None):
    rec = record() if rec is None else rec
    census = census_of(IMAGES) if census is None else census
    return check_install.identity_reference_section(
        mode, facts, "/opt/env/share/meep-gpu-build/build_record.json", rec, census,
        EXPECTED_MEEP, reference_tool.judge_meep)


def test_a_reference_in_keep_passes_every_point():
    lines, reasons = judge("keep", payload())
    assert reasons == []
    assert any("2 of 2 sha256 as recorded" in line for line in lines)
    assert any("as the build record pins" in line for line in lines)


def test_flush_passes_only_where_it_is_measured_at_one_thread():
    assert judge("flush", payload("flush", threads="1"))[1] == []
    reasons = judge("flush", payload("flush", measured="keep", threads=None))[1]
    assert any("no flush reference" in reason for reason in reasons)
    assert any("OMP_NUM_THREADS=1" in reason for reason in reasons)
    assert len(reasons) == 2


@pytest.mark.parametrize("change, words", [
    ("no record", ["has no build record"]),
    ("other images", ["does not describe the MEEP images", "census in its build record is of other images"]),
    ("no census in record", ["carries no fused multiply-add census"]),
    ("dirty record", ["build record's census found 7"]),
    ("dirty here", ["census of the loaded images found 5"]),
    ("keep not measured", ["subnormal handling is flush"]),
    ("other libm", ["not the one its build record pins"]),
    ("no pinned libm", ["pins no math library"]),
    ("unread libm", ["could not be identified"]),
    ("double precision", ["its MEEP is 1.33.0, double precision"]),
])
def test_each_point_that_does_not_hold_is_named(change, words):
    facts, rec, census = payload(), record(), None
    if change == "no record":
        rec = "/opt/env/share/meep-gpu-build/build_record.json does not exist"
    elif change == "other images":
        rec = record(images={LIBMEEP: "3" * 64, MODULE: "2" * 64})
    elif change == "no census in record":
        rec = dict(record(), fma_census=None)
    elif change == "dirty record":
        rec = record(fused=7)
    elif change == "dirty here":
        census = census_of(IMAGES, fused=5)
    elif change == "keep not measured":
        facts = payload(measured="flush")
    elif change == "other libm":
        facts = payload(libm=dict(LIBM, uuid="0" * 8))
    elif change == "no pinned libm":
        rec = record(libm=None)
    elif change == "unread libm":
        facts = payload(libm={"kind": "libsystem_m", "unread": "no"})
    else:
        facts = payload(meep=dict(EXPECTED_MEEP, single_precision=False))
    reasons = judge("keep", facts, rec, census)[1]
    assert len(reasons) == len(words), reasons
    assert all(expected in reason for expected, reason in zip(words, reasons)), reasons


def test_a_census_that_could_not_run_here_leaves_the_record_to_judge():
    lines, reasons = check_install.identity_reference_section(
        "keep", payload(), "/r", record(), "CensusError: no objdump on this host",
        EXPECTED_MEEP, reference_tool.judge_meep)
    assert reasons == []
    assert any("not re-run: CensusError" in line for line in lines)


def test_a_meep_that_does_not_import_is_the_one_reason():
    lines, reasons = check_install.identity_reference_section(
        "keep", {"error": "ModuleNotFoundError: No module named 'meep'"})
    assert reasons == ["MEEP could not be imported (ModuleNotFoundError: No module named 'meep')"]


@pytest.mark.parametrize("identity, key", [
    (LIBM, ("libsystem_m", LIBM["uuid"])),
    ({"kind": "glibc libm", "glibc": "glibc 2.31", "sha256": "s", "cpu_features": ["avx2", "fma"],
      "glibc_tunables": None}, ("glibc libm", "glibc 2.31", "s", ("avx2", "fma"), None)),
    ({"kind": "libsystem_m", "unread": "x"}, None), (None, None)])
def test_the_math_library_is_compared_by_what_selects_its_code(identity, key):
    assert check_install.libm_key(identity) == key


def test_the_build_record_is_looked_for_beside_the_loaded_libmeep():
    assert check_install.find_build_record(IMAGES) == os.path.join(
        "/opt/env", "share", "meep-gpu-build", "build_record.json")
    assert check_install.find_build_record(IMAGES, "/elsewhere/r.json") == "/elsewhere/r.json"
    assert check_install.find_build_record({MODULE: "2"}) is None


def test_the_subnormal_measurement_names_its_four_reads():
    state = check_install.subnormal_state()
    reads = [state[key] for key in ("float32_flush_to_zero", "float32_denormals_are_zero",
                                    "float64_flush_to_zero", "float64_denormals_are_zero")]
    assert state["mode"] == ("keep" if not any(reads) else "flush" if all(reads) else "mixed")


def test_the_child_sets_meeps_switch_for_the_mode_before_it_measures(monkeypatch, capsys):
    calls = []
    fake = types.SimpleNamespace(
        __version__="1.33.0", __file__="/opt/env/meep/__init__.py",
        is_single_precision=lambda: True, with_mpi=lambda: True,
        verbosity=lambda level: None, set_zero_subnormals=lambda on: calls.append(on))
    monkeypatch.setitem(sys.modules, "meep", fake)
    monkeypatch.setattr(check_install, "libm_identity", lambda: dict(LIBM))
    for mode, switch in (("keep", False), ("flush", True)):
        assert check_install.child_identity(mode) == 0
        line = [text for text in capsys.readouterr().out.splitlines()
                if text.startswith(check_install.MARK)][-1]
        facts = json.loads(line[len(check_install.MARK):])
        assert facts["mode"] == mode and facts["meep"]["version"] == "1.33.0"
        assert facts["libm"] == LIBM and "subnormals" in facts
        assert calls[-1] is switch
        assert facts["meep"]["gdsii"] is None  # this stand-in has no with_libGDSII


@pytest.mark.parametrize("libgdsii", [True, False])
def test_the_identity_child_reads_meeps_libgdsii_fact(monkeypatch, capsys, libgdsii):
    """--require-reference and the identity check judge libGDSII only when the facts carry it:
    the child reads MEEP's own with_libGDSII(), so a MEEP built without it is not passed."""
    fake = types.SimpleNamespace(
        __version__="1.33.0", __file__="/opt/env/meep/__init__.py",
        is_single_precision=lambda: True, with_mpi=lambda: True,
        with_libGDSII=lambda: libgdsii,
        verbosity=lambda level: None, set_zero_subnormals=lambda on: None)
    monkeypatch.setitem(sys.modules, "meep", fake)
    monkeypatch.setattr(check_install, "libm_identity", lambda: dict(LIBM))
    assert check_install.child_identity("keep") == 0
    line = [text for text in capsys.readouterr().out.splitlines()
            if text.startswith(check_install.MARK)][-1]
    facts = json.loads(line[len(check_install.MARK):])
    assert facts["meep"]["gdsii"] is libgdsii


@pytest.mark.parametrize("libgdsii", [True, False, None])
def test_the_facts_child_reads_meeps_libgdsii_fact(monkeypatch, capsys, libgdsii):
    """Step 2's facts child, whose MEEP facts step 7 judges against the reference, carries the
    same key: MEEP's own with_libGDSII(), or None from a MEEP that does not report it. The
    stand-in stops the name probe (it is no package), which the child records and goes on."""
    fake = types.SimpleNamespace(
        __version__="1.33.0", __file__="/opt/env/meep/__init__.py",
        is_single_precision=lambda: True, with_mpi=lambda: True,
        verbosity=lambda level: None)
    if libgdsii is not None:
        fake.with_libGDSII = lambda: libgdsii
    monkeypatch.setitem(sys.modules, "meep", fake)
    assert check_install.child_facts() == 0
    line = [text for text in capsys.readouterr().out.splitlines()
            if text.startswith(check_install.MARK)][-1]
    facts = json.loads(line[len(check_install.MARK):])
    assert facts["meep"]["gdsii"] is libgdsii


def test_the_check_stops_on_the_identity_reference_before_any_other_step(monkeypatch, capsys):
    kinds = []

    def run_child(kind, arguments, extra=()):
        kinds.append((kind, tuple(extra)))
        return {"error": "ModuleNotFoundError: No module named 'meep'"}, None

    monkeypatch.setattr(check_install, "run_child", run_child)
    arguments = types.SimpleNamespace(require_identity_reference="keep", gpu_id=0,
                                      verbose=False, identity_build_record=None,
                                      require_reference=False, require_gpu=False)
    with pytest.raises(check_install.Failure) as caught:
        check_install.check(arguments)
    assert kinds == [("identity", ("--identity-mode", "keep"))]
    assert caught.value.section == check_install.SECTION_IDENTITY
    assert caught.value.reason.startswith("--require-identity-reference keep was given and ")
    assert "MEEP identity reference (--require-identity-reference keep)" in capsys.readouterr().out


def test_check_install_accepts_the_identity_options():
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "check_install.py"), "--help"],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0
    assert "--require-identity-reference" in done.stdout and "--identity-build-record" in done.stdout


def test_install_md_has_the_section_an_identity_failure_names():
    text = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    assert f"### {check_install.SECTION_IDENTITY}\n" in text
