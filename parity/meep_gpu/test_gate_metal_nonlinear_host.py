"""Host checks of the Metal nonlinear (Pade) gate: its needles, its record mutations,
its ``dsqr_band`` class.

The gate's legs launch on an Apple GPU; three things it carries do not need one, and
they keep the device legs meaningful between device runs:

* **every source mutation lands**: each needle occurs exactly once in the shipped
  source of every boundary configuration it is armed on, so no leg is disarmed by a
  re-spelled kernel. The package's previous forms are must-catch entries (KERNELS.md
  §7.1): the paired corners, the two-step accumulation and the float32 ``Dsqr``;
* **every record mutation moves the record and stays a record**: the planted chunk
  record differs from the shipped one and still passes the plan's own check, so the
  leg launches it rather than refusing it;
* **the ``dsqr_band`` class discriminates**: on its state the float32 ``Dsqr`` moves
  ``src`` words that the array path's binary64 form does not, so the
  ``dsqr_in_float32`` mutation has something to catch.

The gate sets its subnormal policy in the environment when it is imported, so every
check runs it in a child process with torch made unimportable: the host legs are
torch-free, and this session's environment is untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: Put first in the child: torch cannot be imported, so a host check that reached for
#: it fails here by name rather than passing on a host that happens to have it.
_NO_TORCH = textwrap.dedent('''
    import importlib.abc, sys
    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not importable in this host check")
            return None
    sys.meta_path.insert(0, _NoTorch())
    sys.path[:0] = [{here!r}, {root!r}]
''').format(here=HERE, root=API_ROOT)


def _child(body: str) -> dict:
    """Run ``body`` after the torch block; it prints one JSON object last."""
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _NO_TORCH + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_every_source_mutation_lands_on_its_armed_sources():
    summary = _child('''
        import json
        import gate_metal_nonlinear as gate
        from meep_gpu.metal_kernels import nonlinear_update_e as nonlinear

        rows = {}
        for label, old, new, must, armed, scalar, value_class, _why in gate.MUTATIONS:
            counts = {}
            for boundary_label, boundaries in gate.BOUNDARIES:
                components = (("Ez",) if value_class == "dsqr_band"
                              else ("Ex", "Ey", "Ez"))
                fields, pml = gate.build(boundaries, components, scalar, value_class)
                codes = tuple(nonlinear.BOUNDARY_CODES[kind] for kind in
                              nonlinear._boundary_kinds(fields.grid, pml))
                live = nonlinear.nonlinear_components(fields)
                second, third = nonlinear.chi_volume_arms(fields)
                source = nonlinear.nonlinear_source(live, second, third, codes)
                counts[boundary_label] = source.count(old)
            rows[label] = {"must": must, "armed": list(armed), "counts": counts,
                           "new_differs": new != old}
        print(json.dumps(rows))
    ''')
    for label, row in summary.items():
        assert row["new_differs"], label
        for boundary in row["armed"]:
            assert row["counts"][boundary] == 1, (label, row)
    inverted = ("four_corner_pair_association", "previous_two_step_accumulation",
                "dsqr_in_float32")
    for label in inverted:
        assert summary[label]["must"] is True, label
    assert summary["aux_load_unconditional"]["must"] is False
    assert summary["dsqr_add_operands_commuted"]["must"] is False


def test_every_record_mutation_moves_the_record_and_stays_a_record():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_nonlinear as gate
        from meep_gpu.metal_kernels import chunk_bounds

        rows = {}
        for label, mutate, _why in gate.RECORD_MUTATIONS:
            for boundary_label, boundaries in gate.BOUNDARIES:
                fields, pml = gate.build(boundaries, ("Ex", "Ey", "Ez"), False, "random")
                shipped = np.array(gate.record_of(fields, pml), dtype=np.int32)
                planted = mutate(shipped.copy())
                chunk_bounds.plan_words(planted, fields.grid.shape, "host check")
                rows[f"{label}/{boundary_label}"] = int(np.count_nonzero(planted != shipped))
        print(json.dumps(rows))
    ''')
    assert len(summary) == 6, summary
    assert all(moved > 0 for moved in summary.values()), summary


#: The child's answer when this host cannot be driven to flush: the reference of the
#: ``dsqr_band`` class runs inside ``gate.flushed_host()``, which drives the FPU through
#: ``subnormal_policy._drive_host_fpu``: an already-imported MEEP's
#: ``set_zero_subnormals``, else a host lever such as Darwin's
#: ``FE_DFL_DISABLE_DENORMS_ENV``. The child imports no MEEP, so a host with no such
#: lever cannot flush there. Measured in the child, which then exits before the gate
#: is imported.
_FLUSH_UNATTAINABLE = "host FPU cannot be driven to flush"


def test_the_dsqr_band_class_separates_the_two_forms():
    summary = _child('''
        import json
        from meep_gpu import backends, subnormal_policy

        before = backends.subnormals_flushed()
        subnormal_policy._drive_host_fpu(True)
        attainable = backends.subnormals_flushed()
        subnormal_policy._drive_host_fpu(before)
        if not attainable:
            print(json.dumps({"unattainable": %r}))
            raise SystemExit(0)

        import numpy as np
        import gate_metal_nonlinear as gate
        from meep_gpu import stepping
        from meep_gpu.metal_kernels import nonlinear_update_e as nonlinear

        rows = {}
        for boundary_label, boundaries in gate.BOUNDARIES:
            fields, pml = gate.build(boundaries, ("Ez",), False, "dsqr_band", seed=79)
            magnitude = gate.measured_magnitude(fields, "Ez", 3)
            with gate.flushed_host():
                displacement = stepping._nonlinear_displacement(fields, pml)
                g1s, g2s = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
                gs, us = fields.Dz, fields.inverse_epsilon_for("Ez")
                chi2, chi3 = fields.chi2_for("Ez"), fields.chi3_for("Ez")
                exact = stepping._nonlinear_dsqr(gs, (g1s, g2s), stepping._WHOLE)
                single = gs * gs + np.float32(0.0625) * (g1s * g1s + g2s * g2s)
                src = [(gs * us) * stepping.calc_nonlinear_u(d, gs, us, chi2, chi3)
                       for d in (exact, single)]
            rows[boundary_label] = {
                "magnitude": magnitude,
                "dsqr_split": int(np.count_nonzero(exact.view(np.uint32)
                                                   != single.view(np.uint32))),
                "src_split": int(np.count_nonzero(src[0].view(np.uint32)
                                                  != src[1].view(np.uint32))),
                "cells": int(gs.size)}
        print(json.dumps(rows))
    ''' % _FLUSH_UNATTAINABLE)
    if summary.get("unattainable") == _FLUSH_UNATTAINABLE:
        from conftest import requires_resource_skip  # noqa: PLC0415 - the root conftest

        requires_resource_skip("host_fpu_mode", _FLUSH_UNATTAINABLE)
    for boundary, row in summary.items():
        assert row["magnitude"] <= 1e29, (boundary, row)
        assert row["dsqr_split"] > row["cells"] // 4, (boundary, row)
        assert row["src_split"] > 0, (boundary, row)
