"""The CUDA fusion board's POINTWISE/STENCIL detector, checked against its own controls.

``build_cuda_fusion_matrix.py`` measures every fitness verdict with one function,
``analyze_kernel_reads``, and refuses to report any of them unless its four
``DETECTOR_CONTROLS`` reproduce. These tests pin that the controls reproduce on this
tree, that each control can fail (a detector blind to the cylindrical curl's imported
prelude, a curl planted to read only its own cell, a detector that answers STENCIL
always, and a neighbour read planted in the pointwise control are each refused by
name), and that the two complex Bloch constitutive labels are measured on the source
the shipped launcher compiles.

Host-only: no GPU, no CuPy, no MEEP. The planted cases run on copies in a temporary
directory; the package is only read.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_cuda_fusion_matrix as B  # noqa: E402

CONTROL_MODULES = sorted({module for module, _expected, _why in B.DETECTOR_CONTROLS}
                         | {"step_curl_kernels.py"})
BLOCH = ("fused_update_H_pml_complex_bloch", "fused_update_E_pml_complex_bloch")


def _copy_controls(tmp_path: pathlib.Path) -> pathlib.Path:
    """The control modules and the prelude the cylindrical one imports, copied."""
    root = tmp_path / "cuda_kernels"
    root.mkdir()
    for name in CONTROL_MODULES:
        shutil.copy(B.CUDA_KERNELS / name, root / name)
    return root


def test_every_detector_control_reproduces_on_this_tree():
    rows = B.run_detector_controls()
    assert [(r["module"], r["measured"]) for r in rows] == [
        (module, expected) for module, expected, _why in B.DETECTOR_CONTROLS]


def test_the_cylindrical_curl_reads_its_neighbours_through_the_imported_prelude():
    got = B.analyze_kernel_reads({"module": "cylindrical_kernels.py", "mode": "module"})
    assert got["verdict"] == "STENCIL", got
    reads = {(e["where"].split("(")[0], e["read"])
             for e in got["reads_off_this_thread_cell"]}
    assert ("shift_up", "g[idx + stride]") in reads, reads
    assert ("shift_dn", "g[idx - stride]") in reads, reads
    tree = B.ast.parse((B.CUDA_KERNELS / "cylindrical_kernels.py").read_text())
    imported = B._imported_preludes(tree)
    assert len(imported) == 1 and "shift_up" in imported[0][1], imported


def test_a_detector_blind_to_imported_preludes_fails_the_cylindrical_control(monkeypatch):
    monkeypatch.setattr(B, "_imported_preludes", lambda tree: [])
    got = B.analyze_kernel_reads({"module": "cylindrical_kernels.py", "mode": "module"})
    assert got["verdict"] == "POINTWISE", got
    with pytest.raises(SystemExit, match="cylindrical_kernels.py must measure STENCIL"):
        B.run_detector_controls()


def test_a_cylindrical_curl_planted_to_read_its_own_cell_measures_pointwise(
        tmp_path, monkeypatch):
    root = _copy_controls(tmp_path)
    path = root / "cylindrical_kernels.py"
    text = path.read_text()
    planted, count = re.subn(r"shift_(?:up|dn)\((\w+), idx, [^)]*\)", r"\1[idx]", text)
    assert count == 8, count
    path.write_text(planted)
    monkeypatch.setattr(B, "CUDA_KERNELS", root)
    got = B.analyze_kernel_reads({"module": "cylindrical_kernels.py", "mode": "module"})
    assert got["verdict"] == "POINTWISE", got
    with pytest.raises(SystemExit, match="cylindrical_kernels.py must measure STENCIL"):
        B.run_detector_controls()


def test_a_detector_that_answers_stencil_always_fails_the_pointwise_control(monkeypatch):
    monkeypatch.setattr(B, "analyze_kernel_reads", lambda spec: {"verdict": "STENCIL"})
    with pytest.raises(SystemExit, match="constitutive_kernels.py must measure POINTWISE"):
        B.run_detector_controls()


def test_a_neighbour_read_planted_in_the_pointwise_control_is_refused(tmp_path, monkeypatch):
    root = _copy_controls(tmp_path)
    path = root / "constitutive_kernels.py"
    text = path.read_text()
    assert text.count("float src_x = Bx[idx];") == 1
    path.write_text(text.replace("float src_x = Bx[idx];", "float src_x = Bx[idx + 1];"))
    monkeypatch.setattr(B, "CUDA_KERNELS", root)
    got = B.analyze_kernel_reads({"module": "constitutive_kernels.py", "mode": "module"})
    assert got["verdict"] == "STENCIL", got
    with pytest.raises(SystemExit, match="constitutive_kernels.py must measure POINTWISE"):
        B.run_detector_controls()


@pytest.mark.parametrize("label", BLOCH)
def test_the_bloch_constitutive_labels_are_measured_on_the_emitted_source(label):
    from meep_gpu.cuda_kernels import complex_emitter

    spec = B.FITNESS_SOURCE_BY_LABEL[label]
    assert (spec["mode"], spec["import"], spec["function"]) == (
        "emitted", "meep_gpu.cuda_kernels.complex_emitter", "complex_source")
    sub_step = "update_H" if "_H_" in label else "update_E"
    assert spec["variants"] == ((sub_step,),)
    source = complex_emitter.complex_source(sub_step)
    assert not re.findall(r"__(?:BOUNDS|FLAG\d|SCALE|NAME|EPS_PARAMS)__", source)
    got = B.analyze_kernel_reads(dict(spec))
    assert got["verdict"] == "POINTWISE", got
    assert got["emitted_kernel"] == complex_emitter.KERNELS[sub_step][0]


@pytest.mark.parametrize("label", BLOCH)
def test_a_neighbour_read_planted_in_the_emitted_bloch_body_is_seen(label, monkeypatch):
    from meep_gpu.cuda_kernels import complex_emitter

    shipped = complex_emitter.complex_source

    def planted(sub_step):
        text = shipped(sub_step)
        assert text.count("cf_load(g0, idx)") == 1
        return text.replace("cf_load(g0, idx)", "cf_load(g0, idx + 1)")

    monkeypatch.setattr(complex_emitter, "complex_source", planted)
    got = B.analyze_kernel_reads(dict(B.FITNESS_SOURCE_BY_LABEL[label]))
    assert got["verdict"] == "STENCIL", got
