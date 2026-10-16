"""Host checks of three Metal complex ``update_E`` gates: every mutant reaches a launch.

A source-mutation leg is armed only if the body it compiles is the one the plan
launches. Each of these gates hands the ENGINE builder a function map:

* ``gate_metal_complex_dispersive_update_e`` and ``gate_metal_complex_no_pml_stored_e``
  key their maps as the plan reads them. The builder compiles the shipped body for
  any key the gate's map does not hold, so a map keyed in another shape (an element
  the plans no longer carry) leaves every mutant off the launch path: each must-catch
  row reads 0 words and the gate fails on the device, with nothing on the host to say
  why;
* ``gate_metal_complex_no_pml_offdiag_update_e`` reads each mutated case's own
  specialisation off the engine's plan, then plants its needles in that text.

So, with the compiler replaced by a recorder and the device backend reported present,
each check builds the gate's mutation case through the engine builder with the gate's
own map, runs the plan once on host arrays, and requires: the builder compiled
nothing of its own, the mutated body launched where the gate planted it, and every
other launch ran a shipped body. No device is reached and no arithmetic is checked
here; the device gates own that.

The gates set their subnormal policy in the environment when imported, so every check
runs in a child process with torch made unimportable, as the other
``test_gate_metal_*_host.py`` files do.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: Put first in the child: torch cannot be imported, the compiler is a recorder, the
#: record and presence uploads stay on the host, and the device backend reads present.
_PRELUDE = textwrap.dedent('''
    import importlib.abc, sys
    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not importable in this host check")
            return None
    sys.meta_path.insert(0, _NoTorch())
    sys.path[:0] = [{here!r}, {root!r}]

    import json
    import numpy as np
    from meep_gpu.metal_kernels import chunk_bounds as _chunk_bounds
    from meep_gpu.metal_kernels import complex_fields as _complex_fields
    from meep_gpu.metal_kernels import complex_no_pml_conductive as _conductive
    from meep_gpu.metal_kernels import offdiag_update_e as _offdiag
    from meep_gpu.metal_kernels import shaders

    _chunk_bounds.tensor = lambda record, device="mps": np.asarray(_chunk_bounds.words(record))
    _offdiag.presence_tensor = lambda words, device="mps": np.asarray(
        tuple(int(word) for word in words), np.int32)
    for _module in (_complex_fields, _conductive):
        _module._metal_backend_reasons = lambda grid: []

    class Body:
        """A compiled kernel stand-in: the source it was built from, and its launches."""
        def __init__(self, source):
            self.source, self.calls = source, 0
        def __call__(self, *arguments):
            self.calls += 1

    class Library:
        """What ``compile_source`` returns: every kernel name resolves to one Body."""
        def __init__(self, source):
            self._body = Body(source)
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return self._body

    class HostResidency:
        device = "cpu"
        def __init__(self):
            self.arrays = {{}}
        def mirror(self, name, host, constant=False, dtype=None):
            prior = self.arrays.get(name)
            if prior is not None and prior is not host:
                raise ValueError(f"{{name}} rebound")
            self.arrays[name] = host
            return host

    def refuse(*arguments, **keywords):
        raise AssertionError(
            f"the builder compiled a body the gate's map does not hold: {{arguments!r}}")

    def launched(bodies, plan):
        """Run the plan once; per body that launched, (shipped?, launches)."""
        before = {{id(body): body.calls for body in bodies}}
        plan.run()
        return [(body, body.calls - before[id(body)]) for body in bodies
                if body.calls != before[id(body)]]
''').format(here=HERE, root=API_ROOT)


def _child(body: str) -> dict:
    """Run ``body`` after the prelude; it prints one JSON object last."""
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_every_dispersive_mutant_is_the_body_its_component_launches():
    summary = _child('''
        import gate_metal_complex_dispersive_update_e as gate
        from gate_chunk_records import layer_record
        from meep_gpu.dispersion import LORENTZIAN
        from meep_gpu.metal_kernels import complex_dispersive_update_e as family

        family._metal_backend_reasons = lambda grid: []
        family.compile_complex_dispersive_e = refuse
        gate.compile_source = Library
        shipped = {family.complex_dispersive_e_source(count, axis)
                   for count in range(family.MAX_POLES + 1) for axis in range(3)}
        rows = {}
        for label, (expected, functions) in gate.mutants().items():
            # The gate's mutation case (main(): shape, poles, kind; the seed is free).
            fields, pml = gate.build(shape=(9, 7, 5), poles=2, kind=LORENTZIAN, seed=1)
            plan = family.plan_metal_complex_dispersive_e(
                fields, pml, HostResidency(), functions=functions,
                chunk_bounds=layer_record(fields.grid, pml))
            ran = launched(set(functions.values()), plan)
            rows[label] = {
                "expected": expected,
                "mutant_launches": sum(n for body, n in ran if body.source not in shipped),
                "shipped_launches": sum(n for body, n in ran if body.source in shipped),
                "ex_poles": len(family._poles(fields)["Ex"]),
                "launches": plan.launches,
            }
        print(json.dumps(rows))
    ''')
    assert len(summary) == 8, summary
    for label, row in summary.items():
        # The gate plants its defect in Ex's two-pole body; Ey and Ez run shipped ones.
        assert row["ex_poles"] == 2, (label, row)
        assert row["mutant_launches"] == 1, (label, row)
        assert row["shipped_launches"] == row["launches"] - 1 == 2, (label, row)


def test_every_stored_e_mutant_is_the_body_its_two_pole_components_launch():
    summary = _child('''
        import gate_metal_complex_no_pml_stored_e as gate
        from meep_gpu.dispersion import LORENTZIAN
        from meep_gpu.metal_kernels import complex_no_pml_stored_e as family

        family._metal_backend_reasons = lambda grid: []
        family.compile_complex_stored_e = refuse
        gate.compile_source = Library
        shipped = {family.complex_stored_e_source(count)
                   for count in range(family.MAX_POLES + 1)}
        rows = {}
        for label, functions in gate.mutation_functions().items():
            # The gate's mutation case (main(): shape, poles, kind, phase, components).
            fields, pml = gate.build(shape=(9, 7, 5), poles=2, kind=LORENTZIAN, seed=1,
                                     phase=(0.2, -0.1, 0.15), components=("Ex", "Ez"))
            plan = family.plan_metal_complex_stored_e(
                fields, pml, HostResidency(), functions=functions)
            ran = launched(set(functions.values()), plan)
            rows[label] = {
                "mutant_launches": sum(n for body, n in ran if body.source not in shipped),
                "shipped_launches": sum(n for body, n in ran if body.source in shipped),
                "two_pole_components": sum(
                    len(states) == 2 for states in family._poles(fields).values()),
                "launches": plan.launches,
            }
        print(json.dumps(rows))
    ''')
    assert set(summary) == {"second_pole_dropped", "ordered_subtraction_reversed",
                            "inverse_epsilon_omitted"}, summary
    for label, row in summary.items():
        # The gate replaces the two-pole body: Ex and Ez launch it, Ey the zero-pole one.
        assert row["two_pole_components"] == 2, (label, row)
        assert row["mutant_launches"] == 2, (label, row)
        assert row["shipped_launches"] == row["launches"] - 2 == 1, (label, row)


def test_every_offdiag_mutant_is_planted_on_the_plans_own_specialisation():
    summary = _child('''
        import gate_metal_complex_no_pml_offdiag_update_e as gate
        from gate_chunk_records import layer_record
        from meep_gpu.metal_kernels import complex_no_pml_offdiag_update_e as family

        gate.compile_source = Library
        gate.Residency = HostResidency  # _case_source builds its plan on a fresh one
        # The gate's two mutation cases (main(): seeds and specs).
        specs = {
            "phased": dict(seed=37100, cell_size=(1.125, 0.875, 0.75),
                           phase=(0.2, -0.125, 0.0), boundaries=("periodic",) * 3),
            "negated": dict(seed=37150, cell_size=(1.125, 0.875, 0.75),
                            phase=(1.0 / 2.25, 0.125, 0.0), boundaries=("periodic",) * 3),
        }
        bases = {state: gate._case_source(**spec) for state, spec in specs.items()}
        mutations = gate._mutants(bases["phased"], bases["negated"])
        rows = {}
        for label, (state, functions) in mutations.items():
            spec = dict(specs[state])
            fields, pml = gate._build(**spec)
            plan = family.plan_metal_complex_no_pml_offdiag(
                fields, pml, HostResidency(), functions=functions,
                chunk_bounds=layer_record(fields.grid, None))
            ran = launched(set(functions.values()), plan)
            own = family.complex_no_pml_offdiag_source(
                plan.row_mask, plan.codes, plan.walls, plan.phased)
            rows[label] = {
                "state": state,
                "base_is_the_plans_own": bases[state] == own,
                "mutated": all(body.source != bases[state] for body, _ in ran),
                "launches": sum(n for _, n in ran),
            }
        print(json.dumps(rows))
    ''')
    assert len(summary) == 6, summary
    assert sorted(row["state"] for row in summary.values()) == ["negated"] + ["phased"] * 5
    for label, row in summary.items():
        assert row["base_is_the_plans_own"], (label, row)
        assert row["mutated"] and row["launches"] == 1, (label, row)
