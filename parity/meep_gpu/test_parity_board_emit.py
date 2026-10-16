"""The board's emitted-text witnesses: each table's emit path, against planted pairs.

Every witness of columns (a) and (b) is driven here through the REAL emit path
(``parity_board_emit.py``, one child process per table, no device library importable)
over a planted package under ``tmp_path``: one module spelling MEEP's forms behind the
chunk record, and the same module in the grouping the package shipped through 0.9.2.
The calibration on BASE fa59c2f is pinned on emitted texts of that commit
(``parity_board/base_calibration``); the whole-BASE run is the lane's measurement.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_parity_board as board  # noqa: E402
import parity_board_emit as emit  # noqa: E402
import parity_board_text as witness  # noqa: E402

ROOT = board.REPO_ROOT
CALIBRATION = ROOT / board.BOARD_DIR / "base_calibration"


# ---------------------------------------------------------------------------
# A planted package, emitted through the real child process
# ---------------------------------------------------------------------------


def _planted(tmp_path: Path, table: str, files: Mapping[str, str],
             modules: Mapping[str, Mapping[str, Any]] = None,
             kinds: Optional[Mapping[str, str]] = None) -> Path:
    """A tree holding a ``meep_gpu`` package with ``files`` in the table's directory, the
    real recipes' canonical domains, and one row per planted module."""
    root = tmp_path / "tree"
    package = root / "meep_gpu" / f"{table}_kernels"
    package.mkdir(parents=True)
    (root / "meep_gpu" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    if table == "metal":
        (package / "templates.py").write_text(
            "PERIODIC = 0\nMETALLIC = 1\nCONTRACT_OFF = 'off'\n")
        (package / "symmetry.py").write_text(
            "CODE_MIRROR_METALLIC = 2\nCODE_MIRROR_PERIODIC = 3\n")
    for name, text in files.items():
        (package / name).write_text(text)
    harness = root / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    recipes = json.loads((ROOT / emit.RECIPES_FILE).read_text())
    recipes["modules"] = {t: {} for t in board.TABLES}
    recipes["modules"][table] = dict(modules or {})
    (harness / Path(emit.RECIPES_FILE).name).write_text(json.dumps(recipes))
    rows = [{"module": Path(name).stem, "step": "X1", "lane": "KX1",
             "kind": (kinds or {}).get(Path(name).stem, "kernel"),
             "path": f"meep_gpu/{table}_kernels/{name}"}
            for name in files if not name.startswith("_")]
    (harness / Path(emit.ROWS_FILE).name).write_text(
        json.dumps({"tables": {t: {"modules": rows if t == table else []}
                               for t in board.TABLES}}))
    return root


def _emit(root: Path, table: str) -> Dict[str, Any]:
    return emit.emitted_witnesses(root, tables=(table,))[table]


def _cells(table: str, record: Mapping[str, Any], kind: str = "kernel") -> Dict[str, Any]:
    return board.emitted_text_columns(table, {"kind": kind}, record)


# --- CUDA -----------------------------------------------------------------------------

_CUDA_MEEP = r'''
import cupy as cp

CERTIFIED_KERNELS = ("update_H_planted",)
HEAD = """
__device__ __forceinline__ void constitutive_apply(
    float* f, float* fw, int idx, float src, float kps, float kms, int dsigw) {
    float value = src;
    if (dsigw) { float prev = fw[idx]; fw[idx] = src; value = f[idx] + ((kps * src) - (kms * prev)); }
    f[idx] = value;
}
"""
KERNEL = """
extern "C" __global__ void update_H_planted(float* Hx, float* f_w_Hx, const float* Bx,
    const float* kps_x, const float* kms_x, const int* cb, float dtdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx;
    const int cbi_x_a0 = cb[0], cbi_x_a1 = cb[1], cbi_x_b0 = cb[2], cbi_x_b1 = cb[3];
    const int in_i_x = (i >= cbi_x_a0 && i < cbi_x_a1) || (i >= cbi_x_b0 && i < cbi_x_b1);
    float curl = dtdx * (((Bx[i + 1] - Bx[i]) + Bx[i + 2]) - Bx[i + 3]);
    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx] - curl, kps_x[i], kms_x[i], in_i_x);
}
"""


def _get_kernel(name):
    return cp.RawKernel(HEAD + KERNEL, name)
'''

_CUDA_092 = r'''
import cupy as cp

CERTIFIED_KERNELS = ("update_H_planted",)
HEAD = """
__device__ __forceinline__ void constitutive_apply(
    float* f, float* fw, int idx, float src, float kps, float kms) {
    float prev = fw[idx]; fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}
"""
KERNEL = """
extern "C" __global__ void update_H_planted(float* Hx, float* f_w_Hx, const float* Bx,
    const float* kps_x, const float* kms_x, float dtdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx;
    float curl = dtdx * ((Bx[i + 1] - Bx[i]) + (Bx[i + 2] - Bx[i + 3]));
    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx] - curl, kps_x[i], kms_x[i]);
}
"""


def _get_kernel(name):
    return cp.RawKernel(HEAD + KERNEL, name)
'''


def test_cuda_witness_reads_the_source_the_compiler_is_handed(tmp_path):
    root = _planted(tmp_path, "cuda", {"aligned.py": _CUDA_MEEP, "shipped.py": _CUDA_092})
    emitted = _emit(root, "cuda")
    for name, state in (("aligned", "landed"), ("shipped", "gap")):
        record = emitted[name]
        assert record["witness"] == "emitted" and record["route"] == "_get_kernel", record
        assert record["units"][0]["label"].endswith("RawKernel:update_H_planted")
        cells = _cells("cuda", record)
        assert cells["b_select"]["state"] == state, (name, cells["b_select"])
        assert cells["a_form"]["state"] == state, (name, cells["a_form"])
    assert "0.9.2 grouping survives" in _cells("cuda", emitted["shipped"])["a_form"]["why"]


def test_a_lift_whose_call_site_predates_the_helper_does_not_compile(tmp_path):
    """The helper takes the membership flag; a lifter's call site that was not re-cut
    passes one argument fewer: NVRTC would refuse the text, and the board says so."""
    stale = _CUDA_MEEP.replace("kms_x[i], in_i_x);", "kms_x[i]);")
    root = _planted(tmp_path, "cuda", {"stale.py": stale})
    cells = _cells("cuda", _emit(root, "cuda")["stale"])
    for column in ("b_select", "a_form"):
        assert cells[column]["state"] == "gap"
        assert "does not compile: calls constitutive_apply with 6 arguments; the text " \
               "defines it with 7" in cells[column]["why"], cells[column]


def test_an_emitter_that_raises_on_this_tree_is_refused_by_name(tmp_path):
    broken = '''
def device_sources():
    raise AssertionError("the certified prelude no longer carries the anchor")
'''
    root = _planted(tmp_path, "cuda", {"broken.py": broken})
    record = _emit(root, "cuda")["broken"]
    assert record["witness"] == "refused"
    assert "device_sources() raises on this tree: AssertionError: the certified prelude" \
        in record["why"]
    cells = _cells("cuda", record)
    assert cells["b_select"]["state"] == cells["a_form"]["state"] == "refused"


def test_a_module_with_no_emit_route_falls_back_to_its_source_text(tmp_path):
    transform = 'def transform(text):\n    return text.replace("a", "b")\n'
    root = _planted(tmp_path, "cuda", {"transform.py": transform},
                    modules={"transform": {"route": "none", "reason": "a text transform"}})
    record = _emit(root, "cuda")["transform"]
    assert record == {"recipe": True, "witness": "source", "why": "a text transform"}
    cells = board.witness_columns("cuda", {"kind": "kernel"}, transform, record)
    assert cells["a_form"]["witness"].startswith(board.SOURCE_TEXT)
    assert "no emit route: a text transform" in cells["a_form"]["witness"]


# --- Metal ----------------------------------------------------------------------------

_METAL_MEEP = r'''
def planted_source(codes, backward):
    return """
struct ChunkBounds { int cbh_x_a0; int cbh_x_a1; };
kernel void planted_curl(device float* f0 [[buffer(0)]], device float* u0 [[buffer(1)]],
                         constant ChunkBounds& cb [[buffer(20)]],
                         uint ii [[thread_position_in_grid]]) {
    const int cbh_x_a0 = cb.cbh_x_a0;
    const int cbh_x_a1 = cb.cbh_x_a1;
    const bool in_h_x = ((i) >= cbh_x_a0 && (i) < cbh_x_a1);
    bool ds0 = in_h_x, du0 = in_h_x;
    float curl0 = dtdx * (((g1s - g1) + g2) - g2s);
    float p0 = du0 ? u0[ii] : 0.0f;
    float c0 = f0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = du0 ? ((((c0 * km_z) + n0) - p0) * si_z)
             : (ds0 ? (((c0 * km_y) - curl0) * si_y) : (c0 - curl0));
    if (du0) { u0[ii] = n0; }
    f0[ii] = v0;
}
"""
'''

_METAL_092 = r'''
def planted_source(codes, backward):
    return """
kernel void planted_curl(device float* f0 [[buffer(0)]], device float* u0 [[buffer(1)]],
                         uint ii [[thread_position_in_grid]]) {
    float curl0 = dtdx * ((g1s - g1) + (g2 - g2s));
    float p0 = u0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;
    u0[ii] = n0;
    f0[ii] = v0;
}
"""
'''


def test_metal_witness_reads_the_spliced_shader(tmp_path):
    root = _planted(tmp_path, "metal", {"aligned.py": _METAL_MEEP, "shipped.py": _METAL_092})
    emitted = _emit(root, "metal")
    for name, state in (("aligned", "landed"), ("shipped", "gap")):
        record = emitted[name]
        assert record["witness"] == "emitted", record
        assert record["route"] == "emitters: planted_source"
        # the canonical sweep: both half-steps, the metallic and the periodic triple
        assert len({u["label"] for u in record["units"]}) >= 1
        cells = _cells("metal", record)
        assert cells["b_select"]["state"] == state, (name, cells["b_select"])
        assert cells["a_form"]["state"] == state, (name, cells["a_form"])


def test_a_record_read_beside_an_unselected_recurrence_is_no_select(tmp_path):
    """A lifter that splices the base's record decode beside a curl of its own that runs
    the recurrence at every cell reads the record and selects nothing."""
    mixed = _METAL_MEEP.replace(
        "float v0 = du0 ? ((((c0 * km_z) + n0) - p0) * si_z)\n"
        "             : (ds0 ? (((c0 * km_y) - curl0) * si_y) : (c0 - curl0));",
        "float v0 = (((c0 * km_z) + n0) - p0) * si_z;")
    assert mixed != _METAL_MEEP
    root = _planted(tmp_path, "metal", {"mixed.py": mixed})
    cells = _cells("metal", _emit(root, "metal")["mixed"])
    assert cells["b_select"]["state"] == "gap", cells["b_select"]
    assert "reads the chunk record but steps" in cells["b_select"]["why"]
    assert "outside every per-cell select (recurrence" in cells["b_select"]["why"]


def test_a_record_read_through_an_unbound_cb_does_not_compile(tmp_path):
    """A lifter that carries the record as a member of its parameter block and reads it as
    if ``cb`` were in scope reads no record at all: the compiler refuses the text ("use of
    undeclared identifier 'cb'"), and so does a text that names ``ChunkBounds`` without
    defining it. The record-read token alone is not the witness."""
    member = _METAL_MEEP.replace(
        "struct ChunkBounds { int cbh_x_a0; int cbh_x_a1; };",
        "struct ChunkBounds { int cbh_x_a0; int cbh_x_a1; };\n"
        "struct Params { uint nx; ChunkBounds cb; };").replace(
        "constant ChunkBounds& cb [[buffer(20)]],", "constant Params& prm [[buffer(20)]],")
    undefined = _METAL_MEEP.replace("struct ChunkBounds { int cbh_x_a0; int cbh_x_a1; };\n", "")
    bound = member.replace("    const int cbh_x_a0 = cb.cbh_x_a0;",
                           "    constant ChunkBounds& cb = prm.cb;\n"
                           "    const int cbh_x_a0 = cb.cbh_x_a0;")
    assert len({member, undefined, bound, _METAL_MEEP}) == 4
    root = _planted(tmp_path, "metal", {"member.py": member, "undefined.py": undefined,
                                        "bound.py": bound})
    emitted = _emit(root, "metal")
    cells = _cells("metal", emitted["member"])
    for column in ("b_select", "a_form"):
        assert cells[column]["state"] == "gap", cells[column]
        assert "does not compile: reads the chunk record through cb in planted_curl, which " \
               "binds no cb" in cells[column]["why"], cells[column]
    cells = _cells("metal", emitted["undefined"])
    assert cells["b_select"]["state"] == "gap"
    assert "names the type ChunkBounds, which the text never defines" in cells["b_select"]["why"]
    cells = _cells("metal", emitted["bound"])
    assert cells["b_select"]["state"] == cells["a_form"]["state"] == "landed", cells


def test_the_record_binding_witness_reads_parameters_locals_and_scopes():
    """``cb`` is bound by a parameter (a CUDA pointer, a Metal reference with its
    attribute) or a local; a helper that takes it binds it for itself only."""
    kernel = """
__device__ inline float pick(const int* __restrict__ cb, int i) { return (float)cb[i]; }
extern "C" {
__global__ void k(float* f, const int* __restrict__ cb) { f[0] = (float)cb[3] + pick(cb, 1); }
}
"""
    assert emit.record_binding_problems("cuda", kernel) == []
    unbound = kernel.replace("float* f, const int* __restrict__ cb)", "float* f)").replace(
        "pick(cb, 1)", "1.0f")
    problems = emit.record_binding_problems("cuda", unbound)
    assert problems and "cb in k, which binds no cb" in problems[0], problems
    assert emit.record_binding_problems("triton", "tl.load(cb + 3)") == []


def test_the_emitting_process_reaches_no_device_library(tmp_path):
    reaching = "import torch\n\ndef planted_source(codes, backward):\n    return 'x'\n"
    root = _planted(tmp_path, "metal", {"reaching.py": reaching})
    record = _emit(root, "metal")["reaching"]
    assert record["witness"] == "refused"
    assert "does not import with no device" in record["why"]
    assert "torch is not importable while emitting device text" in record["why"]


# --- Triton ---------------------------------------------------------------------------

_TRITON_FORMS = '''
import triton
import triton.language as tl


@triton.jit
def chunk_in(i, a0, a1):
    return (i >= a0) & (i < a1)


@triton.jit
def record_in(cb, base, i):
    return chunk_in(i, tl.load(cb + base), tl.load(cb + base + 1))


@triton.jit
def curl_form(f, fu, curl, kms, sinv, kms_u, sinv_u, dsigu):
    n = ((fu * kms) - curl) * sinv
    recurrence = (((f * kms_u) + n) - fu) * sinv_u
    dsig_only = ((f * kms) - curl) * sinv
    return tl.where(dsigu, recurrence, dsig_only), n
'''

_TRITON_MEEP = '''
import triton
import triton.language as tl

from .forms import curl_form, record_in


@triton.jit
def planted_step(f_ptr, u_ptr, cb, g1s, g1, g2, g2s, dtdx, km, si, km_u, si_u):
    i = tl.arange(0, 128)
    du = record_in(cb, 0, i)
    curl = dtdx * (((g1s - g1) + g2) - g2s)
    f = tl.load(f_ptr + i)
    p = tl.load(u_ptr + i, mask=du, other=0.0)
    value, n = curl_form(f, p, curl, km, si, km_u, si_u, du)
    tl.store(u_ptr + i, n, mask=du)
    tl.store(f_ptr + i, value)
'''

_TRITON_092 = '''
import triton
import triton.language as tl


@triton.jit
def planted_step(f_ptr, u_ptr, g1s, g1, g2, g2s, dtdx, km, si, km_u, si_u):
    i = tl.arange(0, 128)
    curl = dtdx * ((g1s - g1) + (g2 - g2s))
    f = tl.load(f_ptr + i)
    p = tl.load(u_ptr + i)
    n = ((p * km) - curl) * si
    value = (((f * km_u) + n) - p) * si_u
    tl.store(u_ptr + i, n)
    tl.store(f_ptr + i, value)
'''


def test_triton_witness_inlines_the_jit_helpers_a_kernel_calls(tmp_path):
    root = _planted(tmp_path, "triton", {"forms.py": _TRITON_FORMS, "aligned.py": _TRITON_MEEP,
                                         "shipped.py": _TRITON_092})
    emitted = _emit(root, "triton")
    aligned = emitted["aligned"]
    assert aligned["route"] == "@jit programs, helpers inlined"
    label = aligned["units"][0]["label"]
    assert "forms.curl_form" in label and "forms.record_in" in label and \
        "forms.chunk_in" in label, label
    for name, state in (("aligned", "landed"), ("shipped", "gap")):
        cells = _cells("triton", emitted[name])
        assert cells["b_select"]["state"] == state, (name, cells["b_select"])
        assert cells["a_form"]["state"] == state, (name, cells["a_form"])
    # the kernel's OWN text reads no record: only the inlined helper does
    own = "\n".join(ast.unparse(f) for f in witness.own_text(_TRITON_MEEP).jit_functions)
    assert not any(re.search(p, own) for p in emit.EMITTED_RECORD_READ["triton"])


def test_a_declared_launch_must_be_named_by_the_module(tmp_path):
    admitting = '"""Admission only: launches :func:`.forms.curl_form`."""\nX = 1\n'
    silent = "X = 1\n"
    root = _planted(tmp_path, "triton", {"forms.py": _TRITON_FORMS, "admitting.py": admitting,
                                         "silent.py": silent},
                    modules={"admitting": {"launches": ["forms.curl_form"]},
                             "silent": {"launches": ["forms.curl_form"]}})
    emitted = _emit(root, "triton")
    assert emitted["admitting"]["witness"] == "emitted"
    assert emitted["admitting"]["route"] == "launches: forms.curl_form"
    assert emitted["silent"]["witness"] == "refused"
    assert "its own text never names" in emitted["silent"]["why"]


# ---------------------------------------------------------------------------
# The select and arity witnesses on their own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, selected, unselected", [
    ("float v = du ? ((((c * km_z) + n) - p) * si_z) : (c - curl);", 1, 0),
    ("if (dw) { a = f[i] + ((kp_0 * s) - (km_0 * p)); }", 1, 0),
    ("if (dw) a = f[i] + ((kp_0 * s) - (km_0 * p));", 1, 0),
    ("float v = (((c * km_z) + n) - p) * si_z;", 0, 1),
    ("float a = f[idx] + kps * src;", 0, 1),
    ("for (int t = 0; t < 3; t++) { v = (((c * km_z) + n) - p) * si_z; }", 0, 1),
])
def test_the_select_witness_in_c_text(text, selected, unselected):
    sites = witness.select_sites_c("metal", text)
    assert (sites.selected, sites.unselected) == (selected, unselected), sites


def test_the_select_witness_follows_a_name_into_tl_where():
    tree = ast.parse(_TRITON_FORMS.replace("import triton\nimport triton.language as tl\n", ""))
    sites = witness.select_sites_python([n for n in tree.body if isinstance(n, ast.FunctionDef)])
    assert (sites.selected, sites.unselected) == (1, 0)
    tree = ast.parse(_TRITON_092)
    sites = witness.select_sites_python([n for n in tree.body if isinstance(n, ast.FunctionDef)])
    assert (sites.selected, sites.unselected) == (0, 1)


def test_the_arity_witness_honours_overloads_and_defaults():
    text = ("inline float2 c_mul(float2 a, float2 b) { return a; }\n"
            "inline float2 c_mul(float a, float2 b) { return b; }\n"
            "inline float s(float a, float b = 1.0f) { return a; }\n"
            "kernel void k() { x = c_mul(p, q); y = s(r); z = s(r, t); }\n")
    assert emit.helper_arity_problems(text) == []
    assert emit.helper_arity_problems(text.replace("s(r, t)", "s(r, t, u)")) == [
        "calls s with 3 arguments; the text defines it with 1-2"]


# ---------------------------------------------------------------------------
# The calibration on BASE fa59c2f
# ---------------------------------------------------------------------------


def _calibration_texts():
    manifest = json.loads((CALIBRATION / "MANIFEST.json").read_text())
    assert manifest["base"] == "fa59c2f33c478c7667a173d699b50b8980098471"
    return manifest["texts"]


@pytest.mark.parametrize("entry", _calibration_texts(), ids=lambda e: e["file"])
def test_base_reads_no_record_and_no_meep_order_site(entry):
    """BASE predates the chunk record and MEEP's groupings: on its emitted text every
    witness must read nothing landed, and must still find the 0.9.2 grouping (a witness
    that sees nothing at all would pass this vacuously)."""
    text = (CALIBRATION / entry["file"]).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == entry["sha256"]
    if entry["table"] == "triton":
        functions = [emit._strip_docstring(n) for n in ast.parse(text).body
                     if isinstance(n, ast.FunctionDef)]
        judged = emit.judge_python("triton", functions, True)
    else:
        judged = emit.judge_c(entry["table"], text)
    forms = judged["forms"]
    assert judged["reads"] == [], judged["reads"]
    assert forms["new_curl"] == forms["new_constitutive"] == 0, forms
    assert forms["old_curl"] + forms["old_constitutive"] > 0, forms
    assert judged["arity"] == []
    assert judged["markers"], "the BASE text spells a PML form"


def test_the_landed_roots_read_landed_through_their_emitters():
    """The roots of the three tables (lanes KC1, KT1, KM1) read landed on their EMITTED
    text: a witness that stopped seeing them would read every later lane as a gap."""
    for table, module in (("cuda", "step_curl_kernels"), ("cuda", "constitutive_kernels"),
                          ("triton", "kernels"), ("metal", "shaders")):
        record = emit.emitted_witnesses(ROOT, tables=(table,), modules=(module,))[table][module]
        cells = _cells(table, record)
        assert cells["b_select"]["state"] == "landed", (module, cells["b_select"])
        assert cells["a_form"]["state"] == "landed", (module, cells["a_form"])


def test_the_emitter_and_its_recipes_are_the_boards_own_files():
    for name in ("parity/meep_gpu/parity_board_emit.py", "parity/meep_gpu/parity_board_emit.json",
                 "parity/meep_gpu/record_dev_evidence.py",
                 "parity/meep_gpu/test_parity_board_emit.py"):
        assert name in board.OWN_FILES, name
        assert (ROOT / name).is_file(), name
    recipes = emit.load_recipes(ROOT)
    rows = board.load_rows(ROOT)
    for table in board.TABLES:
        names = {row["module"] for row in rows["tables"][table]["modules"]}
        assert set(recipes["modules"][table]) <= names, (table, set(recipes["modules"][table]) - names)
