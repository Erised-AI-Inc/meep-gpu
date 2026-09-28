# Development

Start with
[CONTRIBUTING.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/CONTRIBUTING.md)
at the root of the repository: what belongs in the package, how to set up and
run the tests, and the rules particular to it. Coding agents and other automated
contributors read
[AGENTS.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/AGENTS.md), which
states the same rules as commands. Before editing a file under
<code>meep_gpu/</code> or <code>parity/meep_gpu/</code>, check whether a
certification ledger pins it. Some ledgers name a file by path and some by file
name alone, so search for the file name; then run the three weld contracts
before and after the edit and compare the drift counts they report (N of D):

```bash
grep -lE '"([^"]*/)?driver\.py"' meep_gpu/*_kernels/*.json
python -m pytest meep_gpu/test_metal_weld_contract.py meep_gpu/test_triton_weld_contract.py meep_gpu/test_cuda_weld_contract.py -q
```

A file any ledger names is pinned, and editing it, comments included, is a
certification event ([Running the certification harness](certification.md)).

The repository holds two trees. <code>meep_gpu/</code> is the package, with its
tests beside the code. <code>parity/meep_gpu/</code> is the certification and
benchmark harness: the scripts that put the kernels against the package's own
array path and against MEEP. Using the package needs nothing from the harness.

The package is designed so its numerical core can be reviewed on its own:

- production package modules import no application framework and no HDF5
  library;
- MEEP is imported only inside lift functions that need it;
- CuPy is resolved lazily, so the NumPy reference path remains importable and
  testable on a machine without CUDA;
- compiled-kernel tables are optional and fail-closed;
- the Metal table may launch hand-written shaders through PyTorch/MPS without
  making Torch an array backend; and
- any future general PyTorch array route must remain a separate backend, not an
  implicit dependency of the MEEP or CuPy paths.

The structural package-boundary tests enforce these rules. Do not move
application job, route, or persistence types into the package.

## Development priorities

1. Preserve a value-level test and a negative control for every new numerical
   feature.
2. Keep CPU-MEEP validation separate from GPU-conformance validation.
3. Add a new feature to the backend-neutral array path before implementing it in
   any of the three kernel tables.
4. Release an arm only through the full chain — a per-family device gate with a
   null control, a ledger entry bound to the bytes that executed, an entry in
   that table's released-arms map admitted by both the shared envelope and the
   arm's own axes row, and a route gate proving byte identity against the array
   path through the driver's own consults with a launch-count substitution
   proof. A benchmark is not part of that chain.
5. State the changed contract and its measured floor in the test module's
   docstring.

## Pages in this section

- [Execution backends and kernel tables](execution-backends.md): the
  layers a lifted run can execute on. Read it before choosing an implementation
  boundary.
- [The three kernel tables](kernel-tables.md): what releases an arm,
  the fused unit, and the switches. Read it before modifying a kernel, a
  certification record, or a dispatch decision.
- [Running the tests](testing.md): the two suites, how a test declares
  a missing resource, and which failures are expected in this release.
- [Running the certification harness](certification.md): how a gate
  and a campaign are run, where their records go, and how a ledger is re-cut.
- [Capability coverage checklist](../guides/capability-coverage.md): the lift
  surface area by area, and how to update it with a compatibility change.

A change to the hand-written CUDA table is only observable on a CuPy engine
under <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>, or on a host with no
validated Triton, because the Triton table composes first and holds every
contested slot.
