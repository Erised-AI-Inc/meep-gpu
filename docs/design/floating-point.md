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
| What it does not mean | Identity with MEEP, identity between an NVIDIA host and an Apple host, or a measured property of a device or environment outside the certified ones |

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
to the array path's grouping, term by term. Three examples:

- The curl stencil is grouped as MEEP's C groups it, left to right:
  <code>dtdx * (((sf - f) + s) - ss)</code>. The pairwise grouping
  <code>(sf - f) + (s - ss)</code> is a different float32 number. Where a curl
  partner's derivative axis is invariant (z at 2-D, phi under cylindrical
  coordinates), the stencil takes MEEP's form for the absent partner,
  <code>dtdx * (sf - f)</code> or <code>(-dtdx) * (ss - s)</code>, not the full
  bracket over a zero difference: <code>(a + s) - s</code> rounds.
- The PML constitutive update forms MEEP's right-hand side first:
  <code>f + ((kps * fw) - (kms * fw_prev))</code>.

  The kernel tables are re-transcribed to these two groupings and re-certified
  against them; until then their gates and host transcription tests differ from
  the array path by contract and are listed in
  <code>tools/ci/pending_certification.txt</code>.
- The PML form is the structure chunk's, as MEEP picks it (step_db.cpp:56-59,
  update_eh.cpp:162-163). A curl runs the split-field recurrence on the cells whose
  chunk carries its dsigu layer and <code>((kms * f) - curl) * sinv</code> on the
  field alone everywhere else, which is MEEP's plain <code>f - curl</code> wherever
  the dsig coefficients are exactly <code>1.0f</code> (checked when the region plan is
  built, refused by name otherwise). A constitutive runs the accumulation on the cells
  whose chunk carries its own layer and <code>f = src</code> elsewhere. Both are
  output selects, never a coefficient that makes one form imitate the other: a
  coefficient of 1 and 0 differs from the select on every signed zero. An auxiliary
  (<code>fu</code>, <code>f_w</code>, <code>f_cond</code>) is read and written only on
  its own form's cells. The kernel tables still run the earlier whole-grid forms and
  are listed the same way until their lanes re-transcribe these.
- A conductivity under a PML takes MEEP's conductive forms only in the structure
  chunks that keep it. MEEP deletes a chunk's conductivity array when every sample it
  takes is zero (structure.cpp:883-900): the chunk's owned samples plus one more,
  below them on the integer lattice and above them on the half-integer one
  (<code>LOOP_OVER_VOL</code>, vec.hpp:170-172). Such a chunk steps the lossless PML
  forms, which differ from the conductive ones on its dsig cells:
  <code>((kms * f) - curl) * sinv</code> is not
  <code>((kms * f) + ((0 - curl) - 0)) * sinv</code> on a signed zero, and the
  conductive form's <code>f_cond</code> stage rounds. In a chunk that keeps it, the
  <code>f_cond</code> increment is formed first and added as one term,
  <code>((kms * x) + (fcnd' - fcnd)) * sinv</code> (step_generic.cpp:167, :209);
  <code>((kms * x) + fcnd') - fcnd</code> is a different float32 number. Its two
  coefficients are formed as MEEP forms them: <code>condfac = 1 - dt2 * sigma</code> in
  float32 with <code>dt2 = float32(dt) * 0.5</code> (step_generic.cpp:88), and
  <code>condinv = 1 / (1 + sigma * dt * 0.5)</code> in double from the float32 sigma
  word, rounded once (structure.cpp:699); the float32 spelling rounds twice and
  differs. The kernel tables' conductive families still run the earlier forms and
  are listed in <code>tools/ci/pending_certification.txt</code> until their lanes
  re-transcribe these.
- The dispersive polarization update is MEEP's isotropic expression,
  <code>gamma1inv * (((p * (2 - od)) - (gamma1 * pp)) + (od * (sigma * w)))</code>
  (susceptibility.cpp:251-256), on the four float32 constants MEEP forms
  (below, "The dispersive polarization update is MEEP's arithmetic"). The three-term
  form <code>((p * c_now) + (c_prev * q)) + (c_drive * (sigma * w))</code> with its
  constants folded in double is the same recurrence in exact arithmetic and a different
  float32 number on most cells. The hand-written CUDA table's polarization kernels take
  MEEP's expression and its per-chunk drive; the Triton and Metal tables' polarization
  families still run the folded form and are listed the same way until their lanes
  re-transcribe it.

A compiler is free to rewrite an expression it is given, so the grouping is held
by measurement and not by construction: each kernel family has a gate that
compares its output with the array path bit for bit, and mutations of the
grouping that the gate must catch. This is why a change of compiler version is
treated as a correctness event. The two NVIDIA tables run on a toolchain
version other than the certified one only inside the supported range (Triton
3.1), and record such a run as not certified; outside it they are refused by
name. The Metal table runs on any GPU architecture, PyTorch and Metal frontend
(every Apple GPU is supported), and records a run outside the certified
environment as not certified. <code>MEEP_GPU_HARDWARE=certified</code> refuses
either by name instead.

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
control its gate ran. The gates behind them ran again in the 1.0 certification
round, on this release's bytes, and released.

## Fast math on the Metal table

With <code>PYTORCH_MPS_FAST_MATH=1</code>, PyTorch compiles every Metal source in
fast-math mode, <code>torch.mps.compile_shader</code> included, which is the path
every Metal kernel of the package is compiled through. The package reads the
variable and never sets it.

| Measurement | Result | Status |
|---|---|---|
| One float32 divide and one square root, compiled through <code>torch.mps.compile_shader</code> and run on the same 1,048,576 inputs with the variable unset, <code>1</code> and <code>0</code> | With <code>1</code>, 274,523 of 1,048,576 divide words and 327,338 of 1,048,576 square-root words differ from the unset run; with <code>0</code>, 0 of each | measured 2026-10-02 on one Apple M1 Max, macOS 26.2, PyTorch 2.10.0 (<code>parity/meep_gpu/probe_metal_fast_math.py</code>) |

The probe compiles two kernels of its own, not a kernel of the package. No Metal
weld ran with the variable set, so the certified environment has it unset or
<code>0</code>. A run with any other value is recorded as not certified, or refused
by name under <code>MEEP_GPU_HARDWARE=certified</code>.

## Double precision on the Metal table

The Apple GPU has no float64, and three of MEEP's statements use a double on single-precision
fields: the source deposit's subtraction, the off-diagonal row's <code>0.25</code> sum and
the nonlinear <code>Dsqr</code>'s <code>0.0625</code> term (the sections below). The Metal
table forms them in software binary64 (<code>metal_kernels/softfp64.py</code>, owner ruling
R7): the value is carried as its 64-bit pattern and each operation (widening, a product
with a power-of-two literal, an addition or subtraction, the store's narrowing) is integer
arithmetic rounded to nearest even, so each result is the word the host's float64 unit
returns. Subnormals, infinities and NaNs follow that unit too. Under the
<code>flush</code> policy, the only one a Metal kernel runs under, an input subnormal reads
as a zero of its sign, and a result below the smallest normal is a zero of its sign,
decided before rounding (arm64's rule; x86 decides after rounding, so the two differ on
the band <code>[2^-126 (1 - 2^-25), 2^-126)</code>).

Measured on one Apple M1 Max against NumPy float64 on the same host, with the FPU keeping
and then flushing (bit_parity_0_10 WP SF64): 0 differing words in each of 4,000,000 triples
per operand class and mode, over the off-diagonal row and the <code>Dsqr</code> form
(seven classes: random exponents, deep cancellation, double ties, float ties, signed zeros,
subnormals and the flush band) and the single operations (random bit patterns including
NaNs and infinities, deposits, the band); each of 16 planted defects in the helpers is
caught. The source deposit was the first statement to use it; the off-diagonal
<code>update_E</code> row of the Metal tensor families (the plain, folded and complex
folded singles and the three stencil welds that lift them) uses it too, so that row
steps on the GPU rather than on the array path, and so does the nonlinear
<code>Dsqr</code> (<code>metal_kernels/nonlinear_update_e.py</code>).

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
| Host processor (NumPy array path) | MEEP's own switch when MEEP is imported; without it, the platform lever in the table below; then the result is measured | The same levers in the other direction; then the result is measured |
| CuPy device binaries | CuPy's own compile option already flushes. Its built-in reduction routines keep, so the policy is refused unless <code>CUPY_ACCELERATORS=''</code> was set before CuPy was imported | The flushing option is removed from every compile, and the compile cache is pointed at a directory of its own |
| Triton device binaries | A per-function attribute is injected at compile time, and the generated device code is audited | A compiler flag stops the device math library from flushing |
| Metal device (Apple) | Native: the device flushes | Not attainable: no lever exists on this device |

The host processor has one lever per kind of process. MEEP's own switch comes
first whenever MEEP is imported, because it is the mechanism MEEP's own runs use;
the package's levers are for a process that has not imported MEEP, such as the
replay of the MEEP-identity fixtures, which blocks MEEP on purpose, or a user with no
MEEP at all:

| Process | Lever | Which threads |
|---|---|---|
| MEEP imported, x86 | MEEP's <code>set_zero_subnormals</code>: the x86 FTZ and DAZ bits (MXCSR <code>0x8040</code>) | Every OpenMP thread |
| MEEP imported, Apple silicon | MEEP's switch has no effect there, so the Darwin environment object below | The calling thread |
| No MEEP, Apple silicon (Darwin) | <code>fesetenv</code> on <code>_FE_DFL_DISABLE_DENORMS_ENV</code> (flush) or <code>_FE_DFL_ENV</code> (keep): FPCR.FZ | The calling thread |
| No MEEP, x86-64 Linux with glibc | <code>fegetenv</code>, then MXCSR FTZ and DAZ set (flush) or cleared (keep) in glibc's <code>fenv_t</code>, then <code>fesetenv</code>; every other MXCSR bit is kept | The calling thread, and threads it creates afterwards (measured on the Linux reference host) |
| No MEEP, any other system | None. <code>keep</code> is what the processor does by default; <code>flush</code> is refused by name, naming the system | |

Each lever is judged by measuring the processor afterwards, and a lever whose
measurement disagrees puts the previous environment back before the refusal. A
process without MEEP that is already in the requested state is left alone: a
<code>keep</code> install on an x86 host that keeps writes nothing. The
x86-64 lever writes the MXCSR word at the offset glibc's header gives
(<code>offsetof(fenv_t, __mxcsr)</code> is 28, <code>sizeof(fenv_t)</code> is 32 on
glibc 2.31), runs only where the C library is glibc on x86-64, and writes nothing
when the word it reads back has a reserved bit set. The calling thread is the one
that steps the array path: NumPy's elementwise operations run on it, and the MEEP flush
reference is defined at <code>OMP_NUM_THREADS=1</code>. A thread that was already
running keeps its own state.

Each kernel table declares the policy its certification was cut under, and
dispatch installs that policy on every executor or refuses by name:

| Kernel table | Certified under | Installed by dispatch | Kept when the caller installed it |
|---|---|---|---|
| Triton | <code>keep</code> and <code>flush</code> | <code>keep</code> | <code>keep</code>; <code>flush</code> per arm, on evidence |
| Hand-written CUDA | <code>keep</code> and <code>flush</code> | <code>keep</code> | <code>keep</code>; <code>flush</code> per arm, on evidence |
| Metal | <code>flush</code> only | <code>flush</code> | <code>flush</code> |

Consequences a user can observe:

- **The policy is process-wide.** A run that dispatches installs the policy for
  the whole process, including the sub-steps that stay on the array path. On the
  Apple host measured, it was still installed after the result was closed, so a
  CPU MEEP run made afterwards in the same process runs under it.
- **On an x86 NVIDIA host a dispatched run keeps subnormals where MEEP flushes
  them.** The run's arithmetic is IEEE 754 throughout; MEEP's on the same host is
  not. The dispatch record states that the run's own resolution was overridden.
- **A caller can run the NVIDIA kernels under <code>flush</code>, arm by arm.**
  Install <code>flush</code> before the first kernel compiles (with
  <code>CUPY_ACCELERATORS=''</code> set before CuPy is imported; for the host half,
  MEEP imported, or on x86-64 Linux with glibc the package's own lever). Dispatch keeps that policy instead of refusing, but only
  when every arm the plan dispatches has a run on this GPU's compute capability
  that released under <code>flush</code>: the <code>certified_policies</code>
  field of the arm's weld record, which the certification round's rebind tools
  write from released flush legs. One arm without that evidence refuses the whole
  plan by name, and the run takes the array path under the caller's policy.
- **The NumPy reference installs nothing.** A <code>prefer_gpu=False</code> run
  takes the host as it finds it.
- **A policy installed after CuPy has compiled still reaches those kernels.**
  Dispatch installs its policy at the configuration freeze, after the lift has
  compiled the array path's CuPy kernels. The install drops every compiled CuPy
  kernel the process holds: the cached functions, and the kernels each ufunc and
  elementwise kernel keeps on itself. Under <code>keep</code> it also adds a policy
  token to CuPy's cache key, so a dropped kernel is rebuilt rather than read back
  from a binary compiled before the install, even when the cache directory was
  already the policy's own. One kind of kernel is out of reach: a
  <code>RawKernel</code> the caller compiled before the install keeps the policy it
  was compiled under, so install the policy before compiling one.
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
| A float32 product compiled before a <code>keep</code> install | Returns 0 before the install and 9.99995 × 10^-41 after it, through a ufunc, the <code>*</code> operator and an elementwise kernel, in a plain and in a pre-set keep cache directory (3 of 3 each); a lifted 2-D PML driver holds 48 compiled kernels before the drop and 0 after it | measured on one RTX A6000, 2026-10-09; the regression test is in the policy module's test file |
| The x86-64 glibc host lever, no MEEP imported | <code>flush</code> and <code>keep</code> each attained and measured in all four probes (flush-to-zero and denormals-are-zero, float32 and float64); MXCSR <code>0x1f80</code> at entry, FTZ and DAZ set and cleared, the other control bits unchanged; a thread created after the install flushes, one already running keeps | measured on the Linux reference host (glibc 2.31), 2026-10-10; the tests are in the policy module's test file |
| The policy installed by dispatch, after the lift | The CuPy array path matches the policy installed before the lift in 306 of 306 arrays and 11 of 11 flux spectra over the nine end-to-end cases, and the route campaign's cases that pass with the policy installed first pass with it installed by dispatch: 0 of 12,459,564 final-state words differ from the array path on the Triton table, 0 of 10,927,936 on the hand-written CUDA table | measured on one RTX A6000, 2026-10-09 |

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

On an NVIDIA host the CuPy array path is in turn measured against the NumPy array path,
word for word after every sub-step
([The CuPy array path stores the NumPy array path's words](#the-cupy-array-path-stores-the-numpy-array-paths-words)).

The array path a table was certified against is itself recorded: every ledger
carries the fingerprint of the words a fixed fixture set produces on the NumPy array
path, and a change to the array path's arithmetic fails the weld contracts until the
tables are certified again
([The array path the ledgers were certified against](../development/certification.md#the-array-path-the-ledgers-were-certified-against)).

It is not any of the following:

- **It is not identity with MEEP.** MEEP is a different program, and on an x86
  host it runs under a different subnormal policy from a dispatched run.
- **It is not identity between an NVIDIA host and an Apple host.** The two are
  certified under different subnormal policies, and no comparison between them
  is claimed.
- **It is not a statement about an uncertified device, toolchain or environment.**
  On the NVIDIA tables one outside the supported range (compute capability 7.0
  to 9.0, Triton 3.1) is refused by name and takes the array path. On a
  supported NVIDIA device or toolchain outside the certified one, and on an
  Apple GPU outside the certified environment, which are supported and not
  certified, the kernels run by default: the run's record says
  <code>certified: False</code>, one
  <code>meep_gpu: NOTE</code> line says the kernels are not certified there, and no
  gate has measured the property on that environment.
  <code>MEEP_GPU_HARDWARE=certified</code> refuses it by name to the array path
  instead.

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

## The readback is MEEP's arithmetic

The readers, <code>GpuRunResult.get_array</code> (whole cell or a
<code>center</code>/<code>size</code> volume),
<code>FdtdDriver.get_field(component, cell_centered=True)</code> and
<code>FdtdDriver.get_field_point</code>, form every value as MEEP 1.33.0 forms
it, from the stored words of the cells MEEP owns (<code>meep_gpu/readback.py</code>):

- The centred sum is MEEP's: <code>0.25 * (((a + b) + c) + d)</code> in float32 with
  the <code>0.25</code> in double, <code>((a + b) + a) + b</code> for one averaged
  axis, with the axes in MEEP's direction order (z before r on a cylindrical grid).
- A sample past a face is rebuilt from an owned word by MEEP's connection rules:
  a copy, a negation, a <code>complex&lt;float&gt;</code> phase product, or the
  <code>0.0</code> of a metal plane. The package's own ghost storage is never read.
- In a chunk where MEEP keeps H as an alias of B (no layer on the component's own
  axis, no integrated magnetic source), a sample past that chunk's face is the
  owner's H word. This decides one plane of B per low layer. After the run's first
  step, a chunk with H of its own still holds zero there: MEEP's first B boundary
  pass runs on connections built before H was allocated.
- The store multiplies the double sum by the weight and by MEEP's double phase
  (a mirror parity, and <code>pow(eikna, -1)</code> on a Bloch duplicate plane), and
  rounds once per part. The power is the host C++ runtime's: libc++'s
  <code>exp(n * log(z))</code> on Darwin, libstdc++'s repeated squaring and
  <code>complex(1) /</code> the power through libgcc's <code>__divdc3</code> elsewhere
  (<code>readback.cpow_int</code>; the libstdc++ form matches a probe compiled by the
  Linux reference's own toolchain on 21,000 of 21,000 random unit-modulus cases at
  <code>n</code> = -3..3, the libc++ form on 6,472).
- The point reader is <code>grid_volume::interpolate</code> and
  <code>fields::get_field</code>: MEEP's weights and their corrections, a sum in
  double in MEEP's order, <code>0.0</code> for a sample no chunk owns.

Measured on MEEP's own words (injected into the package's arrays) and on the
phase-0 prototype's stepped words, against MEEP 1.33.0 single precision built
without contraction on Apple silicon, all twelve components: 0 differing words of
every whole-cell array, stored centre, volume and point read, on 15 Cartesian cases
(folded, odd-count, periodic, one-cell, integrated-source) and 4 cylindrical ones
(bit_parity_0_10 WP AP2). A Bloch axis was not in that count until the package
stored MEEP's owned window (next section but one); with it, the readers' words on
seven stepped Bloch and complex cases are counted there.

## The source deposit is MEEP's double-precision subtraction

MEEP's <code>step_source</code> forms <code>A = sv.current(j) * dt</code> (times
<code>double(cndinv[i])</code> under a conductivity) in <code>complex&lt;double&gt;</code>
and subtracts <code>real(A)</code> from the stored float word: one subtraction in double,
one rounding back to float (step.cpp:295-317). The array path does the same
(<code>sources._deposit</code>, through <code>host_writes.subtract_double</code>):

- the deposition amplitudes are <code>complex128</code>, MEEP's <code>amp[j]</code>;
- <code>real(A)</code> is formed per part as separate float64 operations in MEEP's order,
  <code>(amp_re * c_re) - (amp_im * c_im)</code>, then <code>* dt</code>, then
  <code>* condinv</code>, and the imaginary word of complex storage takes
  <code>(amp_re * c_im) + (amp_im * c_re)</code>; no complex multiply runs, so no host
  can contract it;
- the stored word is widened, <code>real(A)</code> subtracted once and the result rounded
  once, at every listed point;
- a silent table returns before forming anything only where MEEP's subtraction leaves
  every word: the current bitwise <code>(+0, +0)</code> (never <code>== 0</code>, which a
  <code>-0</code> part passes), <code>dt</code> finite and positive, every amplitude finite,
  <code>condinv</code> finite with its sign bit clear at the table's cells, and no point
  forming a <code>-0</code> (in the real plane a point whose <code>amp_re</code> has its sign
  bit set and <code>amp_im</code> does not, in the imaginary plane one where both have it).
  Subtracting <code>+0</code> leaves every float32 word, both zeros included; under flush the
  widening would read a subnormal word as a zero of its sign, and a flushing run stores
  none. Every other silent table still subtracts, as MEEP's loop does
  (<code>(-0) - (-0) = +0</code>), and an array the door refuses (not C-contiguous) is
  still refused. The predicate is static per table: it is read once from host copies and
  cached under weak references to the table's arrays and the <code>condinv</code> volume, so
  a silent step forms nothing, reads no device word and keeps nothing alive. A threshold on
  the current and a source's <code>last_time()</code> are never the predicate: a tail
  current of 1e-100 is not zero, and its deposit gives a stored zero the sign of
  <code>-real(A)</code>;
- the <code>condinv</code> scale is the deposited component's own, for an electric source
  (D) and a magnetic one (B), and is absent for an integrated source;
- under a held Metal residency the residency's own door forms the difference on the
  device, from the device's words, in software binary64
  ([double precision on the Metal table](#double-precision-on-the-metal-table)), in the
  host's subnormal mode measured at the call, so the words are the array path's and none
  is read back to the host.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon, with the array path keeping subnormals (bit_parity_0_10 WP AP5): on the four
lifted simulations of the bit-identity probe, every stored word of the twelve components
on every cell MEEP owns or stores as a wall equals MEEP's, sign of zero included, at
every step 0 to 20 and at every checkpoint to Table 2's N; and so does every such word
of six lifts at alternative chunk divisions (<code>num_chunks</code> 4, 11, 14 and 53,
two <code>chunk_layout</code> partitions; two of them are not separable and step on the
mask route). The only differing words are ghost cells of the folded grids, where MEEP's
reader returns <code>-0</code> for a stored <code>+0</code>. The default suite holds the
four cases to MEEP's digests from capsules, with no MEEP (Tier A,
[MEEP identity](../development/testing.md#meep-identity)).

What these cases do and do not reach:

- They deposit through the Gaussian electric point source and the magnetic volume source
  on real storage, with no conductivity. The continuous and extended classes and the
  integrated path are held to MEEP by the sources family's cases (next section); complex
  storage and the <code>condinv</code> scale are pinned against a scalar transcription of
  step.cpp (<code>test_source_deposit.py</code>), not yet against MEEP.
- A change in the last bit of <code>real(A)</code> rarely reaches a stored float word: the
  three ordering defects of that test's battery (<code>dt</code> applied before the
  product, <code>condinv</code> before <code>dt</code>, a fused complex product) move 0
  stored words there and are caught only on the double operands. MEEP's order is
  established by reading step.cpp and pinned by that operand compare, not by the
  agreement of stored words.

## Sources: MEEP's currents, MEEP's deposit list, MEEP's <code>f_minus_p</code>

Three things decide a source's words besides the deposit's own arithmetic, and the
array path takes each from MEEP:

- **The current.** A lifted run forms every <code>dipole(t)</code> and
  <code>current(t, dt)</code> through the simulation's own <code>src_time</code> object
  (<code>sources._MeepSrcTimeBinding</code>). A transcription cannot reproduce MEEP's
  carrier on every host: the C++ build folds a sine and cosine of one argument into one
  libm entry (<code>__sincos_stret</code> on Darwin, <code>sincos</code> or
  <code>cexp</code> in glibc), and libc++ and libgcc divide complex numbers differently.
  Measured on Linux (bit_parity_0_10 phase 0, P0-d3), Python's <code>math.cos</code> /
  <code>math.sin</code> miss 11 of 8,833 of MEEP's current and dipole rows where glibc's
  own entries miss none. The package's own waveforms serve a driver built without MEEP,
  with no claim of identity.
- **The deposit list.** MEEP adds a source as one piece per (lattice shift, chunk)
  (loop_in_chunks.cpp:390-531, sources.cpp:243-312) and merges a piece into an earlier
  entry of the chunk's list with the same component, the same shared
  <code>src_time</code> and the same index vector, summing the amplitudes in double
  (fields.cpp:576-585, sources.cpp:179-188); the pair deposits once and rounds once
  (<code>sources.bind_src_vols</code>). Two pieces that share a cell without sharing their
  index vector stay two entries: at the seam of a request spanning a periodic axis the
  cell is reached through the direct pass's edge weight and through the image of the
  request's other end, and is deposited twice and rounds twice, on the D deposit and in
  <code>f_minus_p</code> alike. Summing them and depositing once left the seam row 1 ulp
  from MEEP on a mirror-folded periodic axis and on an unfolded one (bit_parity_0_10 WP
  R4: 9 corpus rows of the folded case; WP RG: the unfolded and Bloch-phased rows).
- **The placement.** <code>sources._source_geometry</code> is MEEP's
  <code>add_volume_source</code> and <code>loop_in_chunks</code>: the shift odometer, each
  chunk's owned-corner clip, the boundary weights kept in the source frame and the
  narrow-range minima; the image location <code>(0.5 * is + i) * inva + shift * (0.5 *
  inva)</code>; an axis the run's dimension does not carry as one point of weight 1,
  sampled at 0 (<code>vec2py</code>); every owned point listed, amplitude zero or not;
  the real scalings per part and the complex products separately rounded. An
  eigenmode source samples its mode re-centred on the source volume
  (<code>center -= where.center()</code>, mpb.cpp:910), at each site's offset.
- **The integrated dipole.** MEEP subtracts it from <code>f_minus_p</code>, a copy of D
  (B), per point in double, in list order, and computes E (H) from the copy; D and B
  never hold it (update_eh.cpp:66-140). The driver writes <code>f_minus_p</code>'s words at
  the integrated cells around the constitutive consult and restores them after it
  (<code>sources.f_minus_p_begin</code> / <code>f_minus_p_end</code>), imaging them into
  ghost and wall cells with its own boundary passes, so a pointwise update at a ghost
  reads the image of <code>f_minus_p</code> and every ghost then returns to MEEP's D.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon (bit_parity_0_10 WP AP6g): the 14 cases of the phase-0 sources family, every
step 0 to 20 and the checkpoints to step 400, 0 differing words of 35,481,600 stored
words compared, sign of zero included, with a planted one-ulp flip caught on each
every-step leg. Three further cases reach what the family did not: integrated electric
and magnetic sources beside a mirror plane (0 differing words), a merge across a
four-chunk division (0), and integrated sources with no PML.

What this does not reach:

- A run with no PML stores no H, so B holds <code>f_minus_p</code>'s words through
  <code>step_D</code> (which reads H as B) and every D, E and B word steps as MEEP's; the
  H words at an integrated magnetic source's own cells read B's word where MEEP stores
  <code>B - dipole</code> (2 words per step on the measured case, no other word).
- With a live polarization MEEP subtracts P before the dipole and the package after it.
- A kernel table that owns <code>update_H</code> starts H from the B it reads; on the
  first step that is B with the dipole subtracted at an integrated point inside H's own
  layer, where MEEP starts from B.

## The nonlinear update is MEEP's arithmetic

MEEP's chi2/chi3 constitutive update (<code>step_update_EDHB</code>,
step_generic.cpp:542-655) is transcribed operation for operation by the array path
(<code>stepping.update_E</code>):

- each transverse sum associates left to right,
  <code>((g[i] + g[i+s]) + g[i-s1]) + g[i+(s-s1)]</code>, one float32 rounding per
  addition;
- <code>Dsqr = gs*gs + 0.0625*(g1s*g1s + g2s*g2s)</code> takes its two float32 terms,
  scales and adds them in double (<code>0.0625</code> is a double literal in MEEP) and
  rounds once, at <code>calc_nonlinear_u</code>'s <code>realnum</code> parameter; against
  the all-float32 form this differs only where the exact <code>0.0625*t</code> is below
  float32's normal range;
- the Padé factor and the row product are float32, in MEEP's grouping;
- in a run with mirror symmetry, E and its <code>f_w</code> at each ghost cell are the
  parity image of the owned cell, written after the update, as MEEP's boundary pass writes
  them (step.cpp:105-121). MEEP never evaluates the update at a ghost, and the nonlinear
  <code>Dsqr</code> reads neighbouring cells, so evaluating it there is not the image.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon, with the array path keeping subnormals (bit_parity_0_10 WP AP6h): on five lifted
nonlinear simulations (2-D and 3-D, chi2 and chi3, PML and metallic walls, 1 to 27
chunks, one folded on two mirror planes), every step 0 to 20 (0 to 100 on the folded one)
and the checkpoints to Table 2's N and to step 1000, 0 of 37,055,472 stored words differ in
value, and none on a cell MEEP owns or stores as a wall. The folded case's 12,609
differing words are signed zeros on ghost cells. The left-to-right association is what
reaches the stored words: the paired association moved 45,586 to 2,056,001 of them per
schedule past step 20 on the four cases with a transverse partner, and none at any step to
20 (the fifth case is a TM run, whose <code>Dsqr</code> is <code>gs*gs</code> alone), and
on the folded case the ghost image is
needed as well (without it, ghost words leave MEEP's from step 68 and owned words from
step 90). The double <code>Dsqr</code> moved none on any case;
it is MEEP's expression and is pinned by an operand compare, not by stored words.

What these cases do not reach: complex storage, an off-diagonal permittivity together with
a nonlinearity, a dispersive material together with a nonlinearity, and a folded periodic
axis (a nonlinear run there is refused once its far face is live). The off-diagonal case
carries one known difference, read from MEEP's source and not measured: where a chunk holds
exactly one off-diagonal entry for a component (MEEP's 2x2 branch,
step_generic.cpp:616-625 and :723-730), MEEP's <code>Dsqr</code> is
<code>gs*gs + 0.0625*(g1s*g1s)</code>, the sum of that entry's partner alone, while the
array path adds both transverse sums. Matching it needs the per-chunk presence of the
off-diagonal entries.

## Complex storage is stepped part by part

MEEP stores a complex field as two <code>realnum</code> arrays and forms no product of
two complex numbers while it steps. A real coefficient multiplies each part
(<code>(kap - sig) * f[c][cmp][i]</code>, <code>chi1inv * dmp[cmp][i]</code>; the
<code>DOCMP</code> loops, step_db.cpp:47, update_eh.cpp:149). A Bloch connection is
<code>complex&lt;realnum&gt;(thephase) * word</code>, which the contraction-off build forms
as <code>(a*c) - (b*d)</code> and <code>(a*d) + (b*c)</code> with each operation rounded
(step.cpp:183-192), or a copy or a unary minus when the phase is within 1e-13 of +1 or
-1 (boundaries.cpp:359-366).

NumPy's and CuPy's complex multiply is neither. It contracts to a fused multiply-add on
a host that has one, and there is no float-times-complex loop, so a real coefficient
times complex64 storage runs a complex multiply against <code>(k + 0j)</code>, whose
zero cross terms can flip the sign of a zero part. The array path therefore forms every
product on complex storage in <code>meep_gpu/part_products.py</code>:

- <code>scale</code>: a <code>realnum</code> coefficient times each part; a scalar is
  rounded to the part dtype first, and a wider array or a complex coefficient is refused;
- <code>quarter_turn</code>: <code>+-i</code> times a real coefficient, as an exchange of
  parts and a negation (the special-kz beta and cylindrical <code>i*m/r</code> couplings);
- <code>connect</code>: MEEP's connection rule, the phase rounded to the part dtype;
- <code>divide</code> and <code>from_parts</code>, so that no quotient or assembly is a
  complex operation either.

Real storage takes the plain ufunc in the same operand order, so a real run's words are
the ones it stored before.

**A Bloch-periodic axis stores MEEP's owned window.** MEEP owns
<code>(lc, lc + 2N]</code> of a component whose Yee shift on the axis is 0 and fills
<code>lc</code> from <code>lc + 2N</code> by <code>conj(eikna)</code>
(boundaries.cpp:203-230). The package stores the same points
(<code>Grid.stored_window_low</code>): stored cell 0 of such a component is the owned top
point <code>lc + 2N</code>. The stencil's shifts are then plain rolls, and the reads that
leave the window are the same-index reads across the half cell on plane 0:
<code>conj(eikna)</code> times stored cell 0 below the window (B's forward differences)
and <code>eikna</code> times it above (D's backward differences), formed as MEEP's
connection. The readers, the DFT gathers and the source placement map positions through
the same window (<code>Grid.wrap_count</code>). At <code>k = 0</code>, and for a
component whose Yee shift on the axis is 1, nothing moves.

A runtime probe, <code>parity/meep_gpu/complex_product_interceptor.py</code>, counts every
multiplicative ufunc that reaches a complex operand while a NumPy-reference driver steps;
it finds none on the configurations its test steps, and catches a planted one. DFT, flux
and energy reductions are outside it, as they are outside the identity claim.

Measured against MEEP 1.33.0 single precision built without contraction, with the array
path keeping subnormals (bit_parity_0_10 WP AP6a), on the nine complex-family cases of
phase 0 at every dump step (30 or 31 steps):

- k0, kedge, k1, k2, k3 and f0 store MEEP's value in every word, on Apple silicon and on
  x86-64 Linux, and the two hosts' tables are equal row for row, signed-zero counts
  included. Every owned word of k3 also carries MEEP's sign of zero: 0 of 1,382,400
  differ. The only signed-zero differences are ghost cells, 196 to 1,012 words per case;
- bz, bzp and bf need special-kz beta or BFAST as MEEP's separate passes after the
  curl; with those passes folded into the curl they differed by exactly the counts
  phase 0 measured, and with the passes separate (below) they store MEEP's value in
  every word;
- the readers on k0, kedge, k1, k2, k3, f0 and a three-axis Bloch case, in lockstep with
  MEEP, give 0 value-differing words of every whole-cell array, stored centre and point
  read at every scheduled step to 200 or 400. A <code>center</code>/<code>size</code>
  volume that crosses a periodic face comes back in another shape than MEEP's on four
  of these cases (and on a real one at <code>k = 0</code>), identically before this
  change: that is a reader item, not the window's.

The hand-written CUDA table's complex families transcribe this arithmetic: one float32
product per part, MEEP's unfused connection on plane 0 of the operand read at the cell,
and no fused multiply-add (bit_parity_0_10 lane KC6). The Metal complex families do too
(lane KM5), and so do the Triton complex kernels (the chunk record, in
docs/development/kernel-tables.md). No table's complex families take an expansion
record: the licence chain was retired in one change.

## The off-diagonal constitutive row is MEEP's double-precision sum, per chunk

Where the inverse permittivity has off-diagonal entries, MEEP's
<code>step_update_EDHB</code> forms the row
<code>gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)</code>, with
<code>OFFDIAG(u, g, sx) = 0.25 * ((g[i] + g[i - sx]) * u[i] + (g[i + s] + g[(i + s) - sx]) * u[i + s])</code>
(step_generic.cpp:580-581, :611 and :719). The pair sums, the products and their sum are
float operations, but <code>0.25</code> is a double literal, so each term is a double, the
row is summed left to right in double and the store rounds it once. The array path does
the same (<code>stepping._offdiagonal_terms</code>, rounded by <code>update_E</code>):

- <code>gs * us</code> is one float product, widened; each term's float sum is scaled by
  <code>0.25</code> in double (exact) and added in MEEP's <code>cycle_direction</code>
  order, the y partner of Ex before its z partner;
- complex storage forms the row per part with the real coefficients, as MEEP's
  <code>DOCMP</code> loop does, never through a complex multiply;
- a nonlinear component multiplies the double row by its Pade factor in double and rounds
  once (the most general case, :590-601);
- a term is added only on the cells whose chunk keeps the partner's array. MEEP deletes an
  off-diagonal array in every chunk where it is zero over every sample the chunk allocates
  (anisotropic_averaging.cpp:245-292, one sample past the cells the chunk owns along each
  axis) and passes NULL there (update_eh.cpp:191-196), so the term is absent rather than a
  zero; the two differ only when the row so far is <code>-0</code>. The verdict comes from
  the run's chunk division (<code>ChunkLayout.offdiag_kept</code>), and the cylindrical
  <code>r = 0</code> row takes no off-diagonal term (update_eh.cpp:197-208).

The diagonal entry is the caller's reciprocal rounded once: <code>FdtdDriver.set_epsilon</code>
and <code>set_epsilon_components</code> store <code>chi1inv = float(1 / eps)</code> formed
from the caller's double, as MEEP stores the inverse it forms in double, not
<code>float(1 / float(eps))</code>, which moves 17 % of float <code>chi1inv</code> words by
an ulp (342,605 of 2,000,000 drawn uniformly on [1/16, 1]). The lift hands
<code>1 / chi1inv</code> in double, so the installed word is MEEP's own; a float
permittivity keeps its reciprocal, because a float quotient rounded through double is the
correctly rounded float quotient.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon, with the array path keeping subnormals (bit_parity_0_10 WP AP6e): on the eight
off-diagonal lifts of phase 0 (2-D and 3-D, with and without a layer, a uniform rotated
medium, an anisotropic block crossing into one PML chunk, a cylinder and a sphere whose
smoothing writes the off-diagonal entries; one to 27 chunks), the array path is identical
to MEEP at every compared step: every stored word of the twelve components equals MEEP's,
sign of zero included, at every step 0 to 20, at every checkpoint to Table 2's N and at
every checkpoint to step 1000 (0 of 43,557,888 compared words differ; before this change
15,980,934 differ). The installed diagonal and off-diagonal words equal the ones MEEP
reports on every cell except cells on the first or last stored plane where MEEP's reader
answers its vacuum default (a diagonal word of 1 and off-diagonal words of 0; 64 to 191
cells per row, on the two uniform lifts); and the per-chunk verdict equals
MEEP's rule applied to MEEP's own words over its own chunk volumes, in 48 of 48 (row,
partner) cases.

What these cases do and do not reach:

- The per-chunk presence decides only signed zeros, and none of the eight lifts reaches a
  <code>-0</code> row in a chunk that dropped an array: forcing every term present leaves
  their words unchanged. The rule is MEEP's as read from its source, pinned by a
  constructed case (<code>test_offdiag_row.py</code>) and by the verdict comparison above,
  not by stored words.
- Complex storage, a nonlinear component with an off-diagonal row, cylindrical
  coordinates, mirror symmetry, a layer on a wrapping axis and a susceptibility with an
  off-diagonal row are pinned against a scalar transcription or not at all; none is yet
  compared with MEEP.

## A field component MEEP never allocated is not stepped

MEEP allocates a field component only when something requires it.
<code>fields::_require_component</code> (fields.cpp:552-572) allocates every component
<code>is_like</code> the required one (fields.cpp:471-474): in a 2-D cell that is the required
polarization only (TM is Ez, Dz, Hx, Hy, Bx and By, fields.cpp:458-469) unless the material
couples the two or <code>beta</code> is set, and in 1-D <code>gv.has_field</code> keeps four
components. A component nothing required has no array: <code>update_eh</code> skips it
(<code>if (f[ec][cmp])</code>, update_eh.cpp:150), and the point reader answers 0.0 for it
(monitor.cpp:160-164).

The array path keeps all twelve arrays. On a component MEEP never allocated they hold
only zeros, and only the sign of a zero can differ: the curls and the absorber forms keep
an all-<code>+0</code> neighbourhood at <code>+0</code>, but the constitutive product with a
negative inverse permittivity does not. MEEP's perfect conductor (<code>mp.metal</code>)
stores <code>chi1inv = -1e-20</code>, and <code>-1e-20 * (+0)</code> is <code>-0</code>.
So the lift records the components <code>fields::have_component</code> answers false for
(<code>FdtdDriver.set_absent_components</code>), <code>update_E</code> leaves those E
components untouched at their initial <code>+0</code>, and <code>Fields.get_E</code> serves
zeros for them. A source added later on such a component takes its whole class back out of
the set, as <code>require_component</code> does.

Measured against MEEP 1.33.0 single precision built without contraction (bit_parity_0_10
lane REF): on metal-cavity-ldos.py, a 2-D cavity with an Ez source in an
<code>mp.metal</code> shell, Ex and Ey read <code>-0</code> on every shell cell from step 1
(31,950 of 7,477,200 compared stored words over the run, every one a sign of zero) before
this change and equal MEEP's at every compared step after it.

No kernel table reads the set: every table's E update forms <code>chi1inv * u</code> on all
three components, which differs from the array path only in the sign of a zero (in E, or in
<code>f_w</code> inside the absorber; the accumulated <code>f</code> stays <code>+0</code>), and
only on two sets of cells of an absent component: where its <code>chi1inv</code> carries the
sign bit, and, on a folded grid, on the planes the post-injection fills write for its D. The
array path still steps the absent component's D, and <code>fill_symmetry_bc_D</code> (cell 0
of a mirrored axis the component does not own) and <code>fill_folded_far_ghosts_D</code> (the
far ghost slot of a folded periodic axis) write those planes as <code>parity * image</code>;
with an odd parity that is <code>-1 * (+0) = -0</code>, so <code>u</code> is <code>-0</code>
there in both paths and the stepped product is <code>-0</code> with no conductor anywhere.
So the plan clears those words instead of changing a kernel: when a plan that dispatches
<code>update_E</code> is frozen, <code>fastpath._finish</code> builds one clear per absent E
component with such cells (<code>fastpath.AbsentComponentClear</code>: the flat indices of
the sign-bit cells and of every fill plane, the latter read from the fills' own rule in
<code>deposit_repair</code>, and the component's E and <code>f_w</code> arrays), and after every
<code>update_E</code> consult the plan answers it writes <code>+0</code> there through
<code>host_writes.zero_cells</code>, one device launch per component under a held Metal
residency. Every word it writes is one the array path holds at <code>+0</code>, so the set may
be wider than the words a step turns, never narrower. An unfolded run with no sign bit in an
absent component's inverse permittivity (every unfolded 2-D run without a perfect conductor or
a negative permittivity) gets no clear and pays nothing per step; a folded run pays one plane
per fill; a run that derives E (no absorber, no susceptibility) stores no E word to clear. A
clear that cannot be written exactly refuses the plan by name. The held-residency reader
door (<code>dft._raw_factors</code>) serves an absent derived component as zeros, as
<code>Fields.component_factors</code> does.

Measured with <code>parity/meep_gpu/gate_dispatch_absent_components.py</code> (bit_parity_0_10
lane XAC; perfect-conductor cells under an absent polarization, the dispatched run against
the array path word for word): without the clear the absent words differ on the hand-CUDA,
Triton and Metal tables alike (on the metal-cavity-ldos.py cell, 129 of 104,960 stored words
at the last checkpoint, every one a <code>-0</code> on a shell cell); with it, 0 differing words
on every case each table dispatches. The fill planes were measured on the shipped Metal route
against the array path (bit_parity_0_10 lane FXHD, lifted corpus rows): without them,
<code>finite_grating.py</code> read 725 of 3,949,800 stored words differing per step, all
<code>f_w_Ey</code> on the far ghost plane, and <code>TestGetPoint.test_get_point</code> 124 of
92,256 (62 <code>f_w_Ex</code>, 62 <code>f_w_Ey</code>, the two far planes); with them, 0 on both
at every compared step.

## The folded increments are MEEP's separate passes

Special-kz beta and BFAST add an increment to the curl's target that MEEP computes in a
pass of its own: per component, <code>STEP_CURL</code> and then <code>STEP_BFAST</code>
(step_db.cpp:124-142), and after every component the beta pass over the in-plane
components (:161-175). Each pass adds its <code>raw</code> increment through an additive
ladder (step_generic.cpp:255-533) keyed on the chunk's two curl layers: <code>f +=
raw*siginv</code> where the component's dsigu layer is absent, <code>fu += (df =
raw*siginv); f += siginvu*df</code> where it is present, with <code>raw*cndinv</code> in
place of <code>raw</code> under a conductivity. The array path runs the same passes in the
same order (<code>stepping._apply_folded_increment</code>): folding the increment into the
curl is the same sum in exact arithmetic and a different float32 word, and two additions
into one word do not commute in rounding, so the order on every cell is curl form, BFAST
ladder, beta ladder. (In a 2-D run, the only one with beta, the two passes add nonzero
increments to disjoint components, so there the order can reach signed zeros only.)
BFAST's state takes MEEP's two roundings, <code>F = S - F_prev</code>
stored and then <code>raw = F - F_prev</code>; an absent partner is not read; and in
MEEP's plain one-partner branch the state is <code>k1*(g1s + g1)</code> with no
<code>- F</code>, a line that reaches signed zeros only, because the surviving k of a
one-partner component is zero.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon, with the array path keeping subnormals (bit_parity_0_10 WP AP6b), at every dump
step (0 to 20 and the checkpoints to 400): the nine complex-family cases above and three
more, special-kz beta together with BFAST (<code>bzf</code>), special-kz beta in real
storage (<code>bzr</code>) and BFAST with a nonzero k on the invariant axis
(<code>bfz</code>, which multiplies only partners MEEP does not read, so it stores
<code>bf</code>'s words), store MEEP's value in every word. The only differing words are
the ghost-cell signed zeros listed above. Not measured: conductivity in a folded pass, a
folded pass beside a mirror plane or a metallic wall, and BFAST beside a Bloch phase.

## The dispersive polarization update is MEEP's arithmetic

MEEP's <code>lorentzian_susceptibility::update_P</code> (susceptibility.cpp:188-256) steps
every Lorentzian and Drude term with four <code>realnum</code> constants and, where sigma is
diagonal, the isotropic expression
<code>p = gamma1inv * ((p * (2 - omega0dtsqr_denom) - gamma1 * pp) + omega0dtsqr * (s * w))</code>.
The array path forms the same words (<code>PolarizationState.update</code>):

- the constants are <code>Susceptibility.realnum_constants(dt)</code>. <code>omega_0</code>,
  <code>gamma</code> and <code>dt</code> enter as float32, since MEEP's constructor and
  <code>update_P</code> take <code>realnum</code> (meep.hpp:248, susceptibility.cpp:188-189);
  <code>omega2pi = 2 * pi * omega_0</code> and <code>g2pi = gamma * 2 * pi</code> are formed in
  double, because <code>pi</code> is a double (meep.hpp:53), and stored as float32;
  <code>omega0dtsqr = omega2pi * omega2pi * dt * dt</code>,
  <code>gamma1inv = 1 / (1 + g2pi * dt / 2)</code>, <code>gamma1 = 1 - g2pi * dt / 2</code> and
  <code>2 - omega0dtsqr_denom</code> are float32 operations, left to right (:191-195). A Drude
  term's <code>2 - omega0dtsqr_denom</code> is exactly 2 and its drive keeps
  <code>omega0dtsqr</code>. No libm call is made, so the words are the same on every host;
- the update is that expression with that grouping, every operation float32, and a uniform
  sigma rounded to float32 once, as MEEP stores <code>s[i]</code>
  (anisotropic_averaging.cpp:379, :396);
- complex storage takes every product part by part (<code>part_products.scale</code>), as
  MEEP's <code>DOCMP2</code> loop steps the two parts as two real passes;
- the drive is MEEP's <code>w = f_w ? f_w : f</code> per structure chunk (update_pols.cpp:44):
  <code>Fields.drive_field</code> returns <code>f_w</code> on the cells whose chunk carries the
  component's own layer and the stored E elsewhere.

Measured against MEEP 1.33.0 single precision built without contraction, on Apple
silicon, with the array path keeping subnormals (bit_parity_0_10 WP AP6f): on phase 0's
eight dispersive lifts (a Lorentzian, a Drude term, both on one medium, a lossless
Lorentzian; TM and TE sources; 2-D and 3-D; uniform media and a block whose sigma the lift
looks up in the declared geometry; 9 or 27 chunks, a layer on every face), the array path
is identical to MEEP at every compared step: every stored word of the twelve components
equals MEEP's, sign of zero included, at every step 0 to 20 and at the checkpoints 0, 20,
40, 80, ..., 400 (0 of 15,453,504 compared words differ; before this change 3,056,502
differ). The constants and one update are also pinned against MEEP's source compiled
without contraction (<code>test_dispersion.py</code>).

What these cases do and do not reach:

- Complex storage is pinned against two real passes on its parts, not compared with MEEP.
- MEEP's anisotropic branches (a sigma with off-diagonal entries, and their
  <code>s[i] != 0</code> guard) are not implemented on this engine.
- A susceptibility together with mirror symmetry, cylindrical coordinates, a nonlinear
  component, a conductivity or an off-diagonal <code>chi1inv</code> row is not yet compared
  with MEEP.
- The Triton and Metal tables' polarization kernels still transcribe the folded form. Their
  coverage checks find no <code>(c_now, c_prev, c_drive)</code> triple on the polarization
  and refuse the <code>update_P</code> product by name, so a dispatched run steps the
  polarization on the array path and its other slots dispatch as before (MEASURED on an RTX
  A6000 under both NVIDIA tables before the CUDA table took MEEP's expression). The
  hand-written CUDA table's polarization kernels take MEEP's four words and per-chunk drive;
  their records are re-cut in the certification round. On an Apple GPU the held-residency guard then refuses the whole
  configuration to the array path (MEASURED: 0 differing stored words against the NumPy
  reference, with and without a layer, real and complex storage).

## The lifted susceptibility sigma is MEEP's float32 word

The polarization update multiplies the drive by MEEP's stored <code>s[i]</code>, so a sigma
word one ulp off moves a P word wherever the product's rounding sits that close to a tie. In
the normal range that is rare enough to hide; in the subnormal range, where every product
rounds onto a grid of <code>2**-149</code>, it shows as soon as a field arrives. The lift
installs sigma by one of two routes (<code>driver.sigma_lift</code>): the lookup in the
declared geometry, or the inversion of MEEP's <code>chi1inv</code> at several frequencies.
The lookup installs MEEP's float32 word, the declared double rounded once, as
<code>sigma_row</code> hands it to <code>add_susceptibility</code>
(anisotropic_averaging.cpp:330-396, meepgeom.cpp:1654-1713); the inversion agrees to about
1e-6 of each term's sigma and is not exact.

The lookup reproduces libctl's geometry search, so that it takes the inversion's place on
every cell it can:

- the objects in reverse declaration order, each as its copies over
  <code>LOOP_PERIODIC</code>'s shifts in that loop's order, the unshifted copy among them
  (libctl geom.c:478-519, :1579-1600);
- the shifts are MEEP's lattice vectors, the size of the grid MEEP rounded the cell to
  (<code>user_volume.n{x,y,z}() / resolution</code>, meepgeom.cpp:1956-1985), not the
  declared cell size, over <code>LOOP_PERIODIC</code>'s three-dimensional case in every
  MEEP dimension (<code>make_geom_epsilon</code> calls <code>geom_initialize</code>, which
  sets libctl's <code>dimensions</code> to 3, geom.c:228-238; MEEP's own
  <code>set_dimensions</code> sets MEEP's <code>dim</code>), so a 1-D cell's copies lie along
  z and a cylindrical cell's over r and z;
- each copy is gated on its bounding box before the object test (<code>tree_search</code>,
  geom.c:1651-1667), with the box formed as <code>geom_get_bounding_box</code> forms it for a
  block, an ellipsoid and a sphere (geom.c:1008-1140). A point one ulp outside a copy's box
  is outside the copy, even where the object test on the point less the shift rounds onto
  the object's face. A cylinder, cone or wedge takes a looser box that only spares object
  tests, and other kinds take no gate.

Every installed point is still checked against MEEP's own permittivity at several
frequencies, and a cell that fails goes to the inversion. Measured (bit_parity_0_10 lane
REF): on stochastic_emitter_reciprocity.py (a 9.2-wide periodic cell gridded 994 pixels at
resolution 108) the lookup used to decline on the rounded lattice and the inversion's words
differed from MEEP's in five of the six Ag terms, which an Ez word in the subnormal range
showed at step 1,387; on test_dump_load.py's 3-D cell (an aSi block across a periodic face of
a cell gridded 35/15 wide) it declined on the box gate. Both now lift through the lookup
(worst disagreement 2.8e-07 and 2.0e-08 of the check's 1e-4) and step without a differing
word.

## The cylindrical radial sum and the i*m/r pass are MEEP's, per chunk

A cylindrical (Dcyl) run steps two forms of its own (step_db.cpp:86-294), and the array path
now takes both as MEEP writes them (<code>stepping.step_B</code>, <code>stepping.step_D</code>):

- **The radial derivative of the Z components** is the difference of
  <code>f_rderiv_int</code>, a running sum MEEP builds in every structure chunk over that
  chunk's own rows: <code>F[0] = 0</code>, <code>F[ir] = F[ir-1] + rinv *
  (f_p[ir]*(ir+ir0) - f_p[ir-1]*((ir-1)+ir0))</code> in <code>realnum</code>, left to
  right, with <code>rinv = 1.0/((ir+ir0) - 0.5)</code> formed in double and stored as
  <code>realnum</code>. The sum restarts at every chunk, so the row where an r layer's chunk
  begins holds two values (the lower chunk's sum and the upper chunk's zero) and a cell
  reads the sum of the chunk that owns it
  (<code>stepping.cylindrical_rderiv_difference</code>, the chunk starts from
  <code>ChunkLayout.boundaries</code>). <code>rinv</code> is multiplied, not divided by.
  One sum over the whole r extent, or a division by <code>(ir+ir0) - 0.5</code>, is the same
  derivative and another float32 word.
- **The i*m/r coupling** is a pass of its own after every component's curl (step_db.cpp:177-294),
  through the additive ladder the folded increments use: <code>rinv = the_m/r</code> is a
  <code>realnum</code> quotient of <code>the_m</code>, an integer times the double
  <code>Courant</code> rounded once, by the doubled ILOC r coordinate, and each part adds
  <code>rinv</code> times the partner's OTHER part, a quarter turn of the parts
  (<code>stepping._cylindrical_imr_increment</code>). An R target's dsig is phi, which never
  carries a layer, so it takes <code>f += raw</code> or, in a z-layer chunk, <code>fu +=
  raw; f += siginvu*raw</code>; a Z target takes <code>f += raw</code> or, in an r-layer
  chunk, <code>f += raw*siginv</code>.
- **The axis rules' products with Courant are double.** m = 0: <code>Dz += g*(Courant*4)</code>
  with the product in double and rounded once; |m| = 1: <code>(sd*Courant)*(bracket)</code>,
  the bracket in <code>realnum</code> (<code>stepping._realnum_double_product</code>). A
  float32 factor is the same word only where Courant is a float32 word (0.5 is; 0.3 is not).

Measured against MEEP 1.33.0 single precision built without contraction, on Apple silicon,
with the array path keeping subnormals (bit_parity_0_10 WP AP6c), at every step 0 to 300 and
the checkpoints to 2,000: the four cylindrical cases of phase 0 (m = 0 real and m = 1
complex, each with and without a layer on the r-high and both z faces, whose division cuts
r at row 14) and the plain m = 0 and layered m = 1 cases at Courant 0.3 store MEEP's word
in every one of 35,827,200 compared words, signed zeros included, and the readers return
MEEP's words over them. The same on Linux x86-64. Not measured: a conductivity on a cylindrical grid, a
division whose r cuts differ along z (the owner-volume branch), and an r origin off the axis
(the package places r at the axis).

The hand-written CUDA table's cylindrical kernels take the same forms: the per-chunk sum is
a column-serial scan kernel (one lane per column of stored words, the rows in order, each
add rounded once; a parallel scan associates the sum differently and stores another word
on most rows), and the curls, the i*m/r pass and the axis rules are transcribed from
<code>stepping</code>. Their records are re-cut in the certification round.

## The CuPy array path stores the NumPy array path's words

The NVIDIA tables are certified against the CuPy array path on their own device; the
array-path oracle and MEEP identity are measured on the NumPy array path. The two array
paths are one implementation over two array modules, and that is not enough to make them
store the same words: a CuPy expression compiles its own kernel, CuPy's scans and
reductions are parallel, and NVRTC contracts a multiply-add where it finds one. What makes
them agree is measured, not assumed:

- every CuPy ufunc the array path calls compiles a kernel of one operation, so nothing
  contracts; the one multi-operation body, the complex multiply, is not used: every product
  on complex storage is formed part by part
  ([Complex storage is stepped part by part](#complex-storage-is-stepped-part-by-part));
- the layer forms are pointwise, so CuPy's region plan, which steps views on every axis,
  stores the words of NumPy's, which selects on the innermost resolved axis;
- the source deposit is separate float64 operations, rounded once.

<code>meep_gpu/test_cupy_array_path.py</code> pins it on a CUDA host: two drivers hold one
seeded state, one on NumPy and one on CuPy with dispatch off, each takes
<code>FdtdDriver.step</code>, and after every sub-step the driver runs every array the
capsule pins is compared as 32-bit words, the sign of zero included. It runs 20 cases under
the keep policy.

Measured on one RTX A6000 (x86-64 Linux, CuPy 13.5.1, NumPy 2.2.6; bit_parity_0_10 WP
AP7), under keep and under flush, every CuPy driver refusing every kernel slot:

| Substrate | Steps | Differing / compared words, keep | flush |
|---|---:|---:|---:|
| 59 lifted cases: complex 12, conductive 6, dispersive 3, forced chunk divisions 4 (2 not separable), nonlinear 5, off-diagonal 8, sources 17, the 4 Tier A capsules | 100 each | 0 / 25,639,152,898 | 0 / 25,639,152,898 |
| 19 array-path-oracle fixtures | 12 each | 0 / 45,903,888 | 0 / 45,903,888 |
| 11 complex-storage hand-built states (Bloch phases, beta, BFAST, conductivity, off-diagonal, dispersive, nonlinear, mirror planes, sources) | 12 each | 0 / 40,400,760 | 0 / 40,400,760 |
| 10 states near the bottom of the float32 range | 12 each | 0 / 25,089,876 | 0 / 25,089,876 |
| A fresh lift with <code>prefer_gpu=True</code> against a fresh NumPy lift, 15 cases | 0 | 0 / 6,266,843 | 0 / 6,266,843 |

Every sub-step of every step was compared: 97,603 snapshots on the lifted cases per
policy. The comparison sees the policy: with the host keeping and CuPy flushing, the
states near the bottom of the range differ in 2,443,455 of 6,272,469 words, 10 of 10
cases, and the normal-range states in 4 of 19 and 2 of 11 cases. Under keep, 27 of the 59
lifted runs carry subnormal words. No flushed row shows the one place the two executors
are known to part under flush, the negation of a subnormal word
(<code>EXECUTOR_OP_EXCEPTIONS</code>).

**The cylindrical radial sum is sequential on both.** The radial derivative of Bz and Dz
is the difference of MEEP's per-chunk sum
(<code>stepping.cylindrical_rderiv_difference</code>; "The cylindrical radial sum and
the i*m/r pass are MEEP's, per chunk" above): NumPy forms it as a <code>cumsum</code>
per chunk segment, which accumulates row after row, and every other array module as a
row loop, so CuPy's parallel scan, which rounds differently, is never used. Four lifted
cylindrical cases (m = 0 and 1, with and without a layer, 100 steps) differ in 0 of
125,291,400 words and the cylindrical oracle fixture in 0 of 1,073,340 (12 steps), under
either policy, and the test's two cylindrical rows pass. Before the per-chunk sum, when
the prefix was <code>xp.cumsum</code> along r, the same cases differed in 44,821,841
and 227,299 words from the first <code>step_B</code> on.

Not measured here: the hand-written and Triton kernels against either array path (that is
each table's own certification), and any NVIDIA device other than this one.

