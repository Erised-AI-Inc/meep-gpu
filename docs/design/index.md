# Design

This section describes properties of the package that a user relies on without
seeing them.

- [Floating-point contract](floating-point.md): the precision the
  engine steps in, the rules the kernels are written to, the subnormal policy
  on each executor, and what "bit-identical" is identity with.

The architecture of the execution layers is in
[Execution backends and kernel tables](../development/execution-backends.md).
