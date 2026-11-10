"""What may keep a weld alive on each backend, and what must never.

The rule is uniform across the three tracks: an edit keeps a weld only if the DEVICE
SOURCE is provably unchanged. What differs is how each backend spells "device source", and
those differences are structural, not stylistic -- see :mod:`meep_gpu.device_identity`.

The weighting here matches the risk. A false "unchanged" releases a verdict about bytes
nobody ran; a false "changed" only costs a re-run. So every track gets one case proving it
admits a host-side edit, and a battery proving it refuses anything reaching the device.
"""

from __future__ import annotations

import hashlib
import importlib
import pathlib

import pytest

from .device_identity import (CUDA, METAL, NONE, TRITON, cuda_kernel_digests,
                              device_digests, metal_shader_digests,
                              triton_kernel_digests)

HERE = pathlib.Path(__file__).resolve().parent

TRITON_SRC = '''
import triton
import triton.language as tl

COEFFICIENTS = (0.5, 0.25)
BLOCK_DEFAULT = 128

@triton.jit
def curl(f0, n_elem, BLOCK: tl.constexpr):
    """Doc."""
    idx = tl.arange(0, BLOCK)
    live = idx < n_elem
    v = tl.load(f0 + idx, mask=live) * COEFFICIENTS[0]
    tl.store(f0 + idx, v, mask=live)

def host_helper(a, b):
    """Not a kernel."""
    return a + b
'''


def test_a_triton_kernel_is_found_and_a_host_helper_is_not():
    digests = triton_kernel_digests(TRITON_SRC)
    assert set(digests) == {"curl"}, digests


def test_a_host_side_edit_leaves_the_triton_kernel_identity_alone():
    """The whole point: this is what the byte rule could not distinguish."""
    edited = TRITON_SRC.replace("return a + b", "return b + a").replace(
        '"""Not a kernel."""', '"""Rewritten helper docstring,\n    over two lines."""')
    assert triton_kernel_digests(edited) == triton_kernel_digests(TRITON_SRC)


#: Edits that reach the compiled Triton program. Each must move the kernel digest.
TRITON_LIVE = {
    "kernel body operator": ("* COEFFICIENTS[0]", "/ COEFFICIENTS[0]"),
    "kernel mask condition": ("idx < n_elem", "idx <= n_elem"),
    "kernel argument order": ("def curl(f0, n_elem,", "def curl(n_elem, f0,"),
    # The one a body-only digest would miss. The constant is compiled into the program;
    # the function body is not textually different at all.
    "a module constant the kernel reads": ("COEFFICIENTS = (0.5, 0.25)",
                                           "COEFFICIENTS = (0.5, 0.125)"),
    "a constexpr default": ("BLOCK: tl.constexpr", "BLOCK: tl.constexpr = 64"),
}


@pytest.mark.parametrize("name", sorted(TRITON_LIVE))
def test_anything_reaching_the_triton_program_moves_its_digest(name):
    old, new = TRITON_LIVE[name]
    assert old in TRITON_SRC, name
    assert triton_kernel_digests(TRITON_SRC.replace(old, new)) != \
        triton_kernel_digests(TRITON_SRC), (
        f"{name} reaches the compiled kernel and the digest did not move; a weld "
        f"released on this basis would describe a program nobody ran")


def test_a_constant_the_kernel_does_not_read_is_not_captured():
    """The capture is what the kernel READS, not everything in the module.

    Otherwise every unrelated module-level edit would drift every kernel, which is the
    bluntness this replaces.
    """
    edited = TRITON_SRC.replace("BLOCK_DEFAULT = 128", "BLOCK_DEFAULT = 256")
    assert triton_kernel_digests(edited) == triton_kernel_digests(TRITON_SRC)


def test_the_cuda_extractor_agrees_with_the_certification_reader():
    """Two readers of the same bytes must not disagree.

    ``cuda_kernels/test_certification_record.device_sources`` is the one the shipped
    certification is checked against. If this module's reader ever diverged, one of the
    two would be releasing welds the other refuses.
    """
    source = (HERE / "cuda_kernels" / "step_curl_kernels.py").read_text(encoding="utf-8")
    reference = importlib.import_module(
        "meep_gpu.cuda_kernels.test_certification_record")
    theirs = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for name, text in reference.device_sources(source).items()}
    mine = cuda_kernel_digests(source)
    assert set(theirs) <= set(mine), sorted(set(theirs) - set(mine))
    for name, digest in theirs.items():
        assert mine[name] == digest, name


def test_the_cuda_extractor_resolves_concatenated_kernels():
    """The shipped kernels are ``PRELUDE + r'...'``, which a literal-only reader misses.

    The point is the concatenation, not the count. Until 2026-08-26 this module carried
    fourteen device strings and the count was the cheap way to say "the extractor
    resolved the folded ones too". Twelve of those were the INHERITED kernels and were
    deleted; the two that remain are exactly the ones spelled as ``PRELUDE + r'...'``,
    so the property under test is now asserted directly instead of through a number:
    both certified kernels resolve, and the prelude they share resolves with them.
    """
    source = (HERE / "cuda_kernels" / "step_curl_kernels.py").read_text(encoding="utf-8")
    digests = cuda_kernel_digests(source)
    assert set(digests) == {"_REAL_PML_PRELUDE",
                            "_step_B_pml_real_kernel_code",
                            "_step_D_pml_real_kernel_code"}, sorted(digests)


def test_metal_returns_none_rather_than_something_weaker():
    """FAIL CLOSED. A module with no corpus_digest has no establishable device identity,
    and the caller must keep the strict byte rule rather than accept an approximation."""
    assert metal_shader_digests(HERE / "metal_kernels" / "arms.py") is None


def test_metal_reads_the_modules_own_corpus_digest():
    got = metal_shader_digests(HERE / "metal_kernels" / "offdiag_update_e.py")
    assert got and got.get("corpus"), got


def _metal_emitters(module):
    """The public ``*_source`` emitters, by the rule ``metal_shader_digests`` applies."""
    return sorted(
        name for name in dir(module)
        if name.endswith("_source") and not name.startswith(("_", "refuted"))
        and callable(getattr(module, name, None))
        and getattr(getattr(module, name), "__module__", "") == module.__name__)


def test_a_derived_metal_digest_covers_every_public_emitter_or_none():
    """All of a module's emitters, or no device digest at all.

    A digest over some of them stays equal across an edit to another's shader text, so a
    weld pinned on it would keep its PASS on bytes no gate compiled.
    """
    covered = 0
    for path in sorted((HERE / "metal_kernels").glob("*.py")):
        if path.stem.startswith("test_") or path.stem == "__init__":
            continue
        try:
            module = importlib.import_module(f"meep_gpu.metal_kernels.{path.stem}")
        except Exception:      # noqa: BLE001 - such a module has no device digest either
            assert metal_shader_digests(path) is None, path.stem
            continue
        emitters = _metal_emitters(module)
        if callable(getattr(module, "corpus_digest", None)) or not emitters:
            continue
        got = metal_shader_digests(path)
        assert got is None or sorted(got) == emitters, (path.stem, sorted(got or {}), emitters)
        covered += bool(got)
    assert covered, "no Metal module has a derived device digest; this test would be inert"


@pytest.mark.parametrize("stem,emitter", [
    ("shaders", "constitutive_source"),
    ("complex_fields", "bloch_constitutive_source"),
    ("symmetry", "mirror_ghost_fill_source"),
    ("folded_complex", "folded_mirror_fill_complex_source"),
    ("cylindrical_complex_fused_hd_pair", "curl_source"),
])
def test_the_metal_digest_sweeps_the_constitutive_fill_and_curl_emitters(stem, emitter):
    """Each of these takes a parameter outside the shared sweep and needs a declared one."""
    got = metal_shader_digests(HERE / "metal_kernels" / f"{stem}.py")
    assert got and emitter in got, (stem, sorted(got or {}))


def test_the_declared_metal_sweeps_match_the_emitters_own_tables():
    from .metal_kernels import complex_fields, corpus, folded_complex, shaders, symmetry
    assert corpus._SIDES == tuple(shaders._CONSTITUTIVE_PRODUCTS)
    assert corpus._SIDES == tuple(complex_fields._CONSTITUTIVE_PRODUCTS)
    assert corpus._FILL_PASSES == tuple(symmetry.FILL_PASSES) == tuple(folded_complex.FILL_PASSES)


@pytest.mark.parametrize("stem,template,old,new", [
    pytest.param("complex_fields", "_CONSTITUTIVE_TEMPLATE", "a0 = f0[ii] + (",
                 "a0 = f0[ii] - (", id="complex_fields"),
    pytest.param("folded_complex", "_FILL_TEMPLATE", "if (idx >= n_plane)",
                 "if (idx > n_plane)", id="folded_complex"),
])
def test_a_planted_code_edit_in_a_metal_template_moves_the_digest(monkeypatch, stem, template,
                                                                   old, new):
    """A one-token code edit in a template only the constitutive or fill emitter reads."""
    module = importlib.import_module(f"meep_gpu.metal_kernels.{stem}")
    path = HERE / "metal_kernels" / f"{stem}.py"
    before = metal_shader_digests(path)
    text = getattr(module, template)
    assert text.count(old) == 1, (stem, template, old)
    monkeypatch.setattr(module, template, text.replace(old, new))
    after = metal_shader_digests(path)
    assert before and after and before != after, (stem, before, after)


def test_an_emitter_the_sweep_cannot_reach_leaves_the_module_without_a_digest(monkeypatch):
    from .metal_kernels import complex_fields

    def planted_source(undeclared_parameter):
        return "kernel void planted() {}"

    planted_source.__module__ = complex_fields.__name__
    path = HERE / "metal_kernels" / "complex_fields.py"
    assert metal_shader_digests(path)
    monkeypatch.setattr(complex_fields, "planted_source", planted_source, raising=False)
    assert metal_shader_digests(path) is None


@pytest.mark.parametrize("relative,expected", [
    ("triton_kernels/kernels.py", TRITON),
    ("cuda_kernels/step_curl_kernels.py", CUDA),
    ("metal_kernels/offdiag_update_e.py", METAL),
    ("triton_kernels/launch.py", NONE),
    ("driver.py", NONE),
])
def test_each_file_is_routed_to_its_own_backend(relative, expected):
    kind, digests = device_digests(HERE / relative)
    assert kind == expected, (relative, kind)
    assert bool(digests) == (expected != NONE)


def test_a_file_with_no_device_source_reports_none_not_an_empty_success():
    """``launch.py`` and ``driver.py`` carry no kernel, so they get the STRICTER rule.

    That is the correct treatment, not a fallback: a file with no device source can only
    reach the device through host logic, so any code change in it matters.
    """
    for relative in ("triton_kernels/launch.py", "driver.py"):
        kind, digests = device_digests(HERE / relative)
        assert kind == NONE and digests == {}
