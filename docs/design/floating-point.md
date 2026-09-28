# Floating-point contract

A run that dispatches compiled kernels returns the same bits as the same run on
the package's own array path. This page says what makes that hold, where it
stops, and what it is not: it is not identity with MEEP.

## The short form

| Property | Rule |
|---|---|
| Precision | Single precision. Real runs store float32 fields, complex runs complex64 |
| Association order | Every kernel evaluates its expressions in the grouping the array path uses |
| Multiply-add contraction | Off, on every executor |
| Subnormal numbers | One policy for the whole process, installed on every executor or refused |
| What "bit-identical" means | Kernel results equal the package's own array path on the same host, under the subnormal policy the kernel table is certified for |
| What it does not mean | Identity with MEEP, or identity between an NVIDIA host and an Apple host |

## Single precision

The engine steps in single precision. A real run stores float32 field arrays and
a complex run stores complex64; material arrays are float32, and DFT
accumulators are complex64. Complex storage is used where the problem needs it:
a Bloch wavevector, a cylindrical <code>m</code> other than 0, or a complex β.

MEEP is double precision unless it is configured with
<code>--enable-single</code>, and every MEEP build on conda-forge is double
precision. The agreement figures below were measured against a single-precision
MEEP build. Against a double-precision one (conda-forge MEEP 1.33.0, one Apple
host, 2026-09-28), the NumPy reference differed by 1.5 × 10^-6 of the final
field on the installation check's simulation, and by 5.2 × 10^-6, 7.6 × 10^-6
and 5.1 × 10^-4 on the three examples that compare with MEEP: 4 simulations on
1 host, listed under
[Precision in INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#precision).
A lift from a double-precision MEEP prints one line on standard error saying
so.

## Fixed association order

Floating-point addition is not associative, so a kernel that groups a sum
differently from the array path returns different bits. Each kernel is written
to the array path's grouping, term by term. Two examples:

- The curl stencil is grouped as <code>dtdx * ((sf - f) + (s - ss))</code>. Left
  to right, <code>sf - f + s - ss</code> associates as
  <code>((sf - f) + s) - ss</code>, which is a different float32 number.
- The dispersive polarization update is left-associated across the array path's
  two in-place additions:
  <code>((p * c_now) + (c_prev * q)) + (c_drive * (sigma * w))</code>, with the
  inner product <code>sigma * w</code> formed first.

A compiler is free to rewrite an expression it is given, so the grouping is held
by measurement and not by construction: each kernel family has a gate that
compares its output with the array path bit for bit, and mutations of the
grouping that the gate must catch. This is why a change of compiler version is
treated as a correctness event: a kernel table is refused by name on a toolchain
version other than the certified one.

## No multiply-add contraction

A fused multiply-add computes <code>a * b + c</code> with one rounding where the
array path performs two. A compiler that contracts a product and a sum into one
changes the result. Contraction is turned off on every executor:

| Executor | How contraction is turned off |
|---|---|
| Hand-written CUDA kernels | Every kernel module compiles with <code>--fmad=false</code> |
| Triton kernels | Every launch passes <code>enable_fp_fusion=False</code>, which governs both the compiler's contraction pass and the assembler's option. The setting is spelled once in the package, and a test fails the build on a second spelling |
| Metal kernels | Every shader source carries a file-scope directive that turns contraction off. It is spelled once in the package, and a test fails the build on a second spelling |

The setting is load-bearing, not a precaution. With contraction left on:

| Family | Measurement | Status |
|---|---|---|
| Dispersive polarization update, hand-written CUDA | The unguarded control is identical to the array path on 0 of 242 comparisons | recorded in the kernel module's source |
| Broadband fixed-angle curl, hand-written CUDA | 24 of 24 comparable legs diverge | recorded in the kernel module's source |
| Curl, Triton | The result diverges from the array path by up to 113,238 units in the last place | recorded in the kernel module's source |
| Folded-grid β tail, real fields, Metal | 44 of 1,728 words differ from the array path; 0 of 1,728 with contraction off | recorded in the kernel module's source |

These figures are read from the kernel modules, where each family records the
control its gate ran. The gate record behind each figure is re-cut by the
certification round on the published files.

## The subnormal policy

Subnormal numbers are the float32 values below about 1.2 × 10^-38. A processor
can keep them, as IEEE 754 specifies, or flush them to zero, which is faster on
some hardware. The fields of a decaying source pass through this range, so two
executors that disagree on it produce different bits.

Three policies can be requested:

| Policy | Meaning |
|---|---|
| <code>keep</code> | IEEE 754: subnormal results are kept |
| <code>flush</code> | Subnormal results are flushed to zero |
| <code>match_meep</code> | The default request: whatever MEEP itself does on this host. It is answered by measuring the host, not from a table of platforms. MEEP asks for flushing when it initializes; that request takes effect on x86 and has no effect on Apple silicon, so the answer is <code>flush</code> on an x86 host and <code>keep</code> on an Apple host |

In one process, several independent things decide whether a subnormal survives
an operation. The package drives all of them from one decision:

| Executor | Under <code>flush</code> | Under <code>keep</code> |
|---|---|---|
| Host processor (NumPy array path) | MEEP's own switch, or the platform's floating-point environment where MEEP's switch has no effect; then the result is measured | The same levers in the other direction; then the result is measured |
| CuPy device binaries | CuPy's own compile option already flushes. Its built-in reduction routines keep, so the policy is refused unless <code>CUPY_ACCELERATORS=''</code> was set before CuPy was imported | The flushing option is removed from every compile, and the compile cache is pointed at a directory of its own |
| Triton device binaries | A per-function attribute is injected at compile time, and the generated device code is audited | A compiler flag stops the device math library from flushing |
| Metal device (Apple) | Native: the device flushes | Not attainable: no lever exists on this device |

Each kernel table declares the policy its certification was cut under, and
dispatch installs that policy on every executor or refuses by name:

| Kernel table | Certified under | Installed by dispatch |
|---|---|---|
| Triton | <code>keep</code> and <code>flush</code> | <code>keep</code> |
| Hand-written CUDA | <code>keep</code> and <code>flush</code> | <code>keep</code> |
| Metal | <code>flush</code> only | <code>flush</code> |

Consequences a user can observe:

- **The policy is process-wide.** A run that dispatches installs the policy for
  the whole process, including the sub-steps that stay on the array path. On the
  Apple host measured, it was still installed after the result was closed, so a
  CPU MEEP run made afterwards in the same process runs under it.
- **On an x86 NVIDIA host a dispatched run keeps subnormals where MEEP flushes
  them.** The run's arithmetic is IEEE 754 throughout; MEEP's on the same host is
  not. The dispatch record states that the run's own resolution was overridden.
- **The NumPy reference installs nothing.** A <code>prefer_gpu=False</code> run
  takes the host as it finds it.
- **CuPy's compiler is guarded on a flushing host.** MEEP switches subnormal
  flushing on when it initializes on x86, and some CuPy kernels then fail to
  compile. A GPU request on a CUDA host suspends the flushing for each CuPy
  compile; where that guard cannot be installed it raises
  <code>RuntimeError</code> and names the remedy: call
  <code>meep.set_zero_subnormals(False)</code> before the first GPU kernel
  compiles.
- **Two exceptions are named rather than hidden.** Under <code>flush</code>, the
  host does not flush a negation, because negation is a sign flip and not an
  arithmetic operation, while the device does. And CuPy's built-in reductions
  keep subnormals under <code>flush</code> unless they were disabled before
  import.

| Measurement | Result | Status |
|---|---|---|
| Uniform policy on every executor | 7 of 7 cases and 80 of 80 steps bit-identical | recorded in the policy module's source |
| One executor on a different policy | 92 of 2,880 floats differ, across 22 arrays, at step 1 | recorded in the policy module's source |
| Audit of generated Triton device code | 26 kernels compiled twice; 11,736 float32 arithmetic instructions; every one carries the requested policy | recorded in the policy module's source |
| Cost of the policy in speed | Within 0.41 % on 11 of 11 rows above 200,000 cells, hand-written CUDA kernels, per sub-step, 2026-08-22 | measured; per sub-step, not a whole-step figure |

The policy functions are part of the public API; see
[Subnormal policy](../reference/public-api.md#subnormal-policy).

## What "bit-identical" is identity with

Kernel results are bit-identical to **the package's own array path on the same
host**, under the subnormal policy the kernel table is certified for.

| Host | Kernel tables | The array path that identity is measured against |
|---|---|---|
| NVIDIA | Triton, hand-written CUDA | The CuPy array path on the same device |
| Apple | Metal | The NumPy array path on the host CPU |

The comparison is of whole runs: checkpoints, the final state and the monitor
results, with launch counts proving that kernels ran. The counts, with their
dates, are in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md#correctness-under-dispatch).

It is not any of the following:

- **It is not identity with MEEP.** MEEP is a different program, and on an x86
  host it runs under a different subnormal policy from a dispatched run.
- **It is not identity between an NVIDIA host and an Apple host.** The two are
  certified under different subnormal policies, and no comparison between them
  is claimed.
- **It is not a statement about an uncertified device or toolchain.** Those are
  refused by name and take the array path.

What the property gives a user: on a certified host, a dispatched run and the
array path run under the subnormal policy the dispatched run reported give the
same result, bit for bit. Turning dispatch off with <code>MEEP_GPU_DISPATCH=0</code>
alone leaves the host's own policy in place, so a run whose values pass through
the float32 subnormal range can then differ in the last bits.

## Agreement with MEEP is measured separately

Agreement with MEEP is a tolerance, not an identity. It was measured on the
package's NumPy array path against stock MEEP 1.33.0 in single precision, on the
simulations MEEP's own scripts construct. The metric is the relative L2
difference of the complex fields over the whole volume, all six components.

| Comparison | Result | Status |
|---|---|---|
| Example scripts | 58 of 60 accepted simulations agree within 2.373 × 10^-7 to 1.585 × 10^-5 | measured 2026-08-09 |
| The 2 of 60 outside that band | One uses a random source and is not comparable. One was compared at a reduced resolution and reads 1.092 × 10^-4 there, and 5.19 × 10^-7 at a higher resolution | measured 2026-08-09 |
| Test methods | 130 of 134 accepted simulations agree within 3.123 × 10^-7 to 9.893 × 10^-6 | measured 2026-08-09 |
| MEEP's own test assertions, recomputed from this engine's fields | 366 of 366 assertions that pass on CPU MEEP are reproduced, with a band of 2: one harmonic-inversion case is not deterministic for MEEP either | measured 2026-08-09 |
| Largest grid compared with MEEP | 338,688 cells | measured |
| Simulations stepped through a kernel table and compared with MEEP | 0 of 194 | not measured |
| Comparison with MEEP at the sizes of [the timed case](../guides/will-it-help.md) | none | not measured |

These are dated statements about the engine of 2026-08-09; the stepping code has
changed since, and a re-run on the published files is owed. 58 of the 60 example
comparisons and 134 of the 134 test comparisons ran under a cap of 120,000
cells.

The correctness of a dispatched run therefore rests on two links measured at
different sizes: the kernels equal the array path, exactly, at up to 11,239,424
cells; and the array path agrees with MEEP, within the band above, at up to
338,688 cells.
