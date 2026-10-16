# The three kernel tables

## Current status

The array path is the numerical oracle and remains the fallback for every
configuration. Compiled kernels replace it slot by slot, inside
<code>driver.step()</code>, by default wherever a released arm covers the
configuration on a host its kernel table admits (a supported NVIDIA host,
compute capability 7.0 to 9.0 with Triton 3.1, or any Apple GPU, every one of
which is supported; outside the certified identity, labelled uncertified);
<code>MEEP_GPU_DISPATCH=0</code> turns
dispatch off. Three tables dispatch through the driver's own consults:

| Table | What exists | Runtime status |
|---|---|---|
| Triton | CuPy-pointer interop, per-sub-step coverage predicates, compiled kernels, fingerprinted gate records with a <code>driver_dispatch</code> block, and cross-sub-step fused products | Released and dispatching: 30 arms over 64 arm-case rows; board 334 of 597 seam-instances reachable in dispatch. The first table on a CuPy engine. |
| Hand-written CUDA | <code>cp.RawKernel</code> sources behind the same fail-closed seam, its own certification ledger, and its own board | Released and dispatching: 23 arms over 68 arm-case rows; board 352 of 597, by preference. With a validated Triton it composes second, so it serves nothing there unless <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> puts it first. Without one it composes alone, a composition run on one device for the installation check's simulation, one example and one route leg with Triton withheld from the process (19 of 39 cases byte-identical, 2026-09-29), and not timed. |
| Metal | Hand-written shaders compiled through <code>torch.mps.compile_shader</code> over persistent MPS mirrors of NumPy-owned fields, with a plan-time residency bracket, held on the device by default | Released and dispatching for a <code>prefer_gpu=True</code> driver on an Apple GPU (a NumPy engine with an MPS device): 29 arms over 72 arm-case rows; board 327 of 597, a correctness and reachability number. For the residency modes see [Execution backends](execution-backends.md#residency-is-the-seam). |

The array path is still the reference the arithmetic is proved against, and a
compiled kernel is a performance specialization rather than a second source of
physics. For the architecture and each table's own boundary, read
[Execution backends and kernel tables](execution-backends.md). For the
user-facing switches, boards and measurements, read
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

## What releases an arm

A benchmark is not in this chain. A fused arm reaches a user run only after all
four of the following. Dispatch is on by default, so once all four hold, a run
that does not set <code>MEEP_GPU_DISPATCH=0</code> gets the arm:

1. **A per-family device gate**, paired with a null control that must be able to
   fail, establishing the family's arithmetic on real hardware under the
   certified subnormal policy.
2. **A ledger weld** binding that certification to the bytes that executed —
   <code>ARM_CERTIFICATION</code> for the Triton table,
   <code>CUDA_ARM_CERTIFICATION</code> for the hand-written table, and the
   Metal table's own registry.
3. **An entry in that table's released-arms map**, restricted by both the shared
   release envelope and that arm's own axes row. Both halves must admit the
   configuration; either one refuses it alone.
4. **A route gate** driving the arm through the driver's own consults,
   byte-identical against the array path over whole runs, with a launch-count
   substitution proof showing the kernel actually replaced the sub-steps.

The predicate is fail-closed throughout. A new feature begins on the array path,
and a missing kernel can never become a reason a supported run is refused.

The admitted set is handed down to the planner explicitly rather than inferred
from a label's spelling. An un-admitted product is therefore never installed, and
its seam keeps the plans it had.

### Single kernels on the hand-written table

The hand-written table also serves a few **single** sub-step kernels, but never
one alone. The merge adopts them only as complete units, and each needs two
things: a launch it can resolve, and a certification row pointing at the byte
gate that certified its entry point. The units are:

- a seam pair, curl plus constitutive: the real-PML curl with the ordinary update,
  or the real-PML curl with one of the three off-diagonal electric updates;
- the symmetry-fill pair. It is adopted on both fill slots or
  on neither, and refused by name when another table holds the other fill slot.
  Its near and far passes run at the driver's two fill consults, with the wall
  clear between them left on the array path.

These singles let the hand-written table serve a mirror-folded run and an
off-diagonal D/E seam without a second table, which matters on a host with no
validated Triton. That composition has been run on one device for the installation check's simulation and one example, and by the
route campaign's leg with Triton withheld from the process (2026-09-29): 19 of 39 cases byte-identical to the array path with only
CUDA kernels launched, 18 complex cases skipped and 2 with no fused arm. It has not been timed.

### The chunk division on the hand-written table

MEEP picks the PML form of each curl and constitutive update per structure chunk, so
a kernel needs the run's chunk division as data. The hand-written table reads it once
per composition: the division handed to its composer, or, when none is handed in,
the lifted division the run's fields carry, else the one-rank division of the grid
and PML. The dispatch planner hands in the driver's own division
(<code>FdtdDriver.chunk_layout</code>), the same object it hands the Triton composer:
a lifted run's, or the one-rank division a driver built without MEEP steps with. A
caller that names no division, such as the coverage census, reads the same division
on a lifted run, so the census judges the division the run steps with.

The division's <code>int32[25]</code> record (<code>chunk_map.chunk_bounds_record</code>)
is uploaded once per composition, and every plan of that composition, single or
fused, binds the same device vector.

The table's root kernels read it: the real-field PML curl pair
(<code>step_B_pml_real</code>, <code>step_D_pml_real</code>), the constitutive pair
(<code>update_H_pml_real</code>, <code>update_E_pml_real</code>) and the nonlinear
<code>update_E_pml_real_nonlinear</code>. Each takes the record as its last argument,
<code>cb</code>, decodes the entries it reads once per thread with the block
<code>cuda_kernels/bounds_block.py</code> emits, and steps the forms the array path
steps from the same division (<code>chunk_map.RegionPlan</code>): the split-field
recurrence on the cells whose chunk carries the component's dsigu layer and
<code>((f * kms) - curl) * sinv</code> elsewhere; the dsigw accumulation in its chunks
and <code>f = src</code> elsewhere; each auxiliary read and written only on its own
form's cells; the stencil in MEEP's grouping, with an absent partner per the
record's presence word. The no-absorber curl (<code>step_B_no_pml_real</code>,
<code>step_B_no_pml_real_derived</code>, <code>step_D_no_pml_real</code>) carries the
same stencil and takes the record too; with no absorber no chunk carries a layer, so it
reads the presence word alone. The three no-absorber conductive curls are built from
its text and take the record with it. The nonlinear <code>update_E</code> adds its
four-point transverse sums left to right and forms <code>Dsqr</code> in double, rounded
once, as <code>stepping._nonlinear_transverse_sums</code> and <code>_nonlinear_dsqr</code>
do. The real fused products derive from those texts and step the same forms: the
B&rarr;H and D&rarr;E pairs decode the record once over both halves and key every carried
ghost on its destination's membership row, and the H&rarr;D weld evaluates update_H's
membership at each recomputed cell. The conductive PML pair derives from the root curl
text too and chooses each conductive component's form from its dsig and dsigu membership
flags; it and the conductive D&rarr;E pair built on it take the record. The BFAST and
beta families (the singles, their fused pairs and the H&rarr;D welds, real and complex)
are their root texts with MEEP's folded pass inserted after each component's absorber:
the BFAST state step or the beta increment, with MEEP's unowned cells at
<code>+0</code>, added through the pass's own additive ladder on the dsigu chunks'
cells, as <code>stepping._apply_folded_increment</code> does.

The dispersive families are re-derived. The polarization update
(<code>update_P_pml_real</code>, <code>update_P_pml_real_uniform</code>) is MEEP's
recurrence, <code>gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s *
w)))</code>, on the four float32 words <code>Susceptibility.realnum_constants</code>
gives, and its drive is MEEP's per chunk: it binds <code>f_w_&lt;c&gt;</code> and the
stored <code>E&lt;c&gt;</code> and selects between them from the record (the stored E for
both without a layer). The dispersive <code>update_E</code>, the E&rarr;P pair, the
three-slot weld, the dispersive D&rarr;E pair and the off-diagonal dispersive single and
E&rarr;P weld take the record and step the root helper's two forms; the E&rarr;P register
hand-off is the drive on both forms, since <code>constitutive_apply</code> stores the
source into <code>f_w</code> on one and into E on the other. The complex no-absorber
<code>update_P</code> and its two fused products take the recurrence one part at a time,
as MEEP's <code>DOCMP2</code> loop does, and the no-absorber dispersive pair and
three-slot weld bind the record for their curl's presence word.
The record carries no conductivity: which chunks keep a component's conductivity
(<code>RegionPlan.conductive_chunks</code>) is a per-chunk fact of the conductivity
volume and the division, and the array path's conductive curl under a PML reads it
from the plan. The conductive PML kernels read the same plan
(<code>conductive_kernels.conductive_keeps</code>): a component every chunk keeps the
conductivity of compiles the conductive forms in, and otherwise the plan's per-cell
<code>conductive_mask</code> is bound beside the record.

The off-diagonal <code>update_E</code> singles (unfolded and folded) and the two stencil
welds built on them are re-derived onto the root texts. Their constitutive tail is the
root helper; their row is MEEP's, with each term's float32 sum scaled and added in
double and the row rounded once; and a term is added only where the cell's chunk keeps
the row's <code>chi1inv</code> array. The kernel reads that per cell from one 27-bit
presence word per live row, indexed by the chunk labels the record gives the cell along
each axis on the component's own sub-lattice
(<code>cuda_kernels/offdiag_presence.py</code> builds the words from the array path's
own presence and refuses by name a record that does not reproduce it).

The complex families (<code>complex_emitter</code> and the kernels and fused products
that lift it: the complex PML singles, the folded complex curl and its two ghost-carry
emitters, the no-absorber complex curls, the complex and folded-complex fused pairs, the
complex H&rarr;D weld, the complex off-diagonal <code>update_E</code> and the PML arm of
its two stencil welds) are re-derived onto the same forms, per part. A real coefficient
times a field is one float32 product per part, so the arm names emit one body; a phased
axis wraps plainly and connects the operand read at the cell itself on plane 0, as
<code>stepping._bloch_ghost_operand</code> does (a copy, a per-part negation, or MEEP's
unfused product with the rounded phase); every kernel decodes the record and selects its
forms per cell. The no-absorber complex lifters are re-derived onto the no-absorber
curl's text: the complex no-absorber D&rarr;E pair, the complex no-absorber three-slot
dispersive weld and the no-absorber arm of the complex off-diagonal stencil weld bind
the record for the curl's partner-presence word and take the D curl's connection table
(<code>eikna</code>, the word above the stored window); the stencil weld's no-absorber
arm binds the same chunk-form pack as its folded arm (the record, the live rows'
presence words and both connection tables).

The cylindrical families (the real m = 0 curl pair <code>cylindrical_kernels</code>, the
complex curl pair <code>cylindrical_complex_kernels</code> at every m class, their fused
B&rarr;H and D&rarr;E pairs and the two cylindrical H&rarr;D welds) are re-derived onto the
same forms. The radial derivative of Bz and Dz is the difference of MEEP's per-chunk
<code>f_rderiv_int</code>, computed by a column-serial scan kernel
(<code>cylindrical_prefix.cyl_radial_difference</code>: one lane per column of stored
words, each part of a complex column, the rows in order, the sum restarted at every
chunk start the array path reads, <code>stepping._cylindrical_scan_starts</code>) before the
curl launch; a parallel scan would associate the sum differently. The i*m/r coupling is
a pass of its own through the folded increments' ladder, and the axis rules' products
with Courant are formed in double and rounded once, as
<code>stepping</code> does. The H&rarr;D welds take three launches (update_H into scratch,
the scan, the curl), as many as the D side's own, and stay uninstallable.

Two refusals send a configuration to the array path by name:

- **Every arm and fused product** refuses a division that no record describes: one
  that is not separable along an axis, or a line with more than two runs. The array
  path steps such a division on its mask route.
- **The off-diagonal families** (every axis) and **the cylindrical curls** (along r)
  index MEEP's chunks, not only the membership runs. They also refuse a division
  whose record does not name those chunks one to one. That happens when a chunk cut
  falls inside a run or inside the interior. It also happens when there is a layer
  on a wrapping axis: the stored cell 0 joins the low run while the high chunk owns
  it.

The division read, and any refusal, is in the dispatch record's
<code>tables.cuda.chunk_division</code>.

## The fused unit

The dependency-local pair is the smallest fused unit:

1. <code>step_B → zero_metal_B → update_H</code>; then
2. <code>step_D → zero_metal_D → update_E</code>.

Released products already exceed the pair. The hand-written table ships
three-slot dispersive welds, off-diagonal stencil welds, a folded complex
off-diagonal stencil weld, and a no-absorber three-slot dispersive weld.

**Sources are carried across the seam, not refused.** The deposit repair saves
the stateful arrays at the deposit points before the first slot's launch; the
driver injects, fills and clears as usual; the second slot then recomputes the
constitutive result there in the same operand order. Source-bearing seams are
route witnesses that dispatch today. Dispersion is served by a dedicated
dispersive fused pair rather than by weakening the ordinary pair's predicate.

That bracket is where the two-pair plan's fixed per-step host cost lives — a
seam-swap ablation showed the cost follows the source's seam and that neither
kernel is slow. The bracket is restricted to the component the source's
injection actually writes, bit-identically; the measurement is in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

Combining both pairs into one ordinary Triton launch is still not a simple
extension: D's curl needs the newly computed H field, so every program needs a
grid-wide barrier between the halves, which independent Triton programs cannot
provide inside a normal launch. A one-launch whole step would need a separate
persistent or cooperative design; concatenating the two kernels would be
incorrect.

## Two switches that look alike and are not

<code>MEEP_GPU_FUSED=0</code> is the highest-precedence kill switch: it refuses
the **whole** plan and sends every slot to the array path. It is read when the
plan is built, so set it before the run starts; changing it part-way through
does nothing until the configuration re-freezes.

<code>MEEP_GPU_FUSE_ARMS=0</code> is the arm-level veto: it admits no fused arm
and **leaves the rest of dispatch running**. That is the configuration that
isolates a fusion difference from a sub-step difference, and the one that lets a
launch count show what fusing saves. The route gate depends on this veto:
without it the unfused-dispatch baseline is unreachable and the substitution
proof would measure a drop of zero.

The full switch list, including the subnormal policy and the
<code>CUPY_CACHE_DIR</code> rule that goes with it, is in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

## Reading a dispatch decision

<code>driver.active_step_path</code> reports <code>"array"</code> or
<code>"fused"</code>, where <code>"fused"</code> means at least one sub-step will
launch a kernel; read it after a step, because the plan is built at the
configuration freeze inside <code>step()</code>.
<code>driver.fast_path_report()</code> is this driver's own freeze: the path, the
per-slot outcome with its reason, and the certification each dispatched family
rides on. <code>MEEP_GPU_DISPATCH_LOG</code> appends the same record per freeze.

A benchmark script must refuse to report a compiled result when the driver did
not execute one.

## The chunk record

MEEP selects every PML form per structure chunk, so a kernel that keys its forms on
the chunk division reads the division as data: the record
<code>chunk_map.chunk_bounds_record</code> builds from the run's
<code>ChunkLayout</code>, 25 <code>int32</code> words (the
<code>[a0, a1) ∪ [b0, b1)</code> membership runs per Yee sub-lattice and axis, then
a partner-presence word). It is bound once per plan, as one pointer argument named
<code>cb</code>, rather than as integer arguments or source text: Triton
specialises an integer argument equal to 1 or divisible by 16, and bounds in the
source would put one compiled binary per grid in the cache.

On the Triton table:

- **Every PML plan binds it.** The curl, constitutive and fused-pair plans of
  <code>launch.py</code>, the dispersive <code>update_E</code> plan and the
  scratch-weld plan base refuse to exist without one
  (<code>launch.bind_chunk_bounds</code>, which also refuses, by name, a record whose
  runs do not fit the plan's extent). The three <code>launch.py</code> plans and the
  three folded plans (the folded curl and the two folded fused pairs) pass it to
  their kernels as <code>cb</code>, directly after their last pointer argument.
- **The engine route takes it from one place**, <code>launch.chunk_bounds_for</code>:
  the division handed to <code>plan_step(..., chunk_layout=...)</code>, or, when none
  is handed over, the lifted division the fields carry, else the one-rank division of
  the grid and layer. The dispatch
  planner hands over the driver's own division (<code>FdtdDriver.chunk_layout</code>),
  so a dispatched plan is always judged against the run's division. A division
  handed over is judged first, for the whole plan
  (<code>coverage.chunk_layout_reasons</code>): one that is not separable, and on a
  cylindrical or off-diagonal run one that is not a <code>{low, interior, high}</code>
  product, leaves every slot on the array path with the reason named.
- **The harness route has no default.** <code>plan_from_arrays</code> and its
  siblings require <code>chunk_bounds</code>; the gates state which division their
  reference steps with (<code>parity/meep_gpu/gate_chunk_records.py</code>).

On the Metal table:

- **The base kernels read it.** The real-field PML curl and constitutive kernels
  (<code>metal_kernels/shaders.py</code>) and the folded curl
  (<code>metal_kernels/symmetry.py</code>) bind the record as
  <code>constant ChunkBounds&amp; cb</code>, their last buffer, and select every cell's
  PML form from it, as the array path's region plan does. A family that carries its
  scalars in a <code>Params</code> struct embeds the same 100-byte struct as a member
  and aliases it with <code>chunk_bounds.params_alias()</code>.
- **Every base plan binds one, with no default**, through
  <code>chunk_bounds.plan_words</code>, which also refuses a record whose presence word
  disagrees with the plan's shape. The engine route takes the composer's record
  (<code>launch.chunk_division</code>); the harness route requires
  <code>chunk_bounds</code>.
- **The same-cell fused pairs bind it.** <code>fused_magnetic_pair</code>,
  <code>fused_electric_pair</code> and <code>nonlinear_fused_magnetic_pair</code> lift
  both base bodies, so they step the base forms; their <code>Params</code> carries the
  record and one alias line before the lifted text binds it.
  <code>fused_dispersive_pair</code> takes its curl lines from the base curl's emitted
  source and binds the record as its last buffer; the PML arm of
  <code>fused_ade_chain</code> carries it in <code>Params</code>. Both select their E half
  per chunk, and the ADE chain's drive is the register that is <code>f_w</code> on the
  accumulating cells and E on the plain ones.
- **The H&rarr;D welds bind it too** (<code>fused_hd_pair</code>,
  <code>folded_fused_hd_pair</code>): their <code>Params</code> embeds the record and
  <code>h_cell</code> takes it as a parameter, so each recompute selects its form at the
  recomputed cell.
- **The fold-carry pairs bind it** (<code>folded_fused_pair</code>,
  <code>folded_fused_magnetic_pair</code>, <code>folded_fused_dispersive_pair</code>): the
  record rides in their <code>Params</code>, their curl and constitutive halves are lifted
  from the certified emitters, and a ghost the fill moves along the component's own axis
  evaluates its constitutive membership at its destination.
- **The complex families bind it** (<code>complex_fields</code>, <code>folded_complex</code>,
  <code>complex_conductive_pml</code> and the complex and folded-complex fused pairs):
  their singles bind the record last, their pairs embed it in <code>Params</code>, and
  their products are part-wise, as <code>part_products</code> forms them.
- **The beta and BFAST families bind it** (<code>special_kz</code>,
  <code>bfast_curl</code>, <code>folded_beta</code> and the products that lift them): their
  curls are the certified curls with MEEP's separate pass inserted after the per-chunk
  forms (<code>metal_kernels/folded_pass.py</code>), the singles bind the record last and
  the pairs embed it in <code>Params</code>. The BFAST pass selects its one-partner forms
  on the record's presence word, so a nonzero k on a declared-invariant axis is admitted,
  as on the Triton table: it multiplies only a partner MEEP does not read.
- **The nonlinear <code>update_E</code> binds it**
  (<code>metal_kernels/nonlinear_update_e.py</code>): the record last, the base
  constitutive tail per chunk, the four-corner transverse sums left to right, and
  <code>Dsqr</code>'s <code>0.0625</code> scale and the addition after it in software
  binary64 (<code>metal_kernels/softfp64.py</code>, ruling R7), rounded once, as
  <code>stepping._nonlinear_dsqr</code> forms them. Its harness builder requires
  <code>chunk_bounds</code>.
- **The cylindrical families bind it and step MEEP's cylindrical forms**
  (<code>cylindrical_real</code>, <code>cylindrical_complex</code>,
  <code>cylindrical_complex_scan</code> and their six fused products). The radial
  derivative of Bz and Dz is the difference of two partial sums over each chunk's rows,
  restarted at +0 on the chunk's first row with <code>rinv</code> multiplied, as
  <code>stepping.cylindrical_rderiv_difference</code> forms it: a column-serial kernel
  on real storage and in the complex H&rarr;D weld, the array path's own call on the other
  complex plans; the curl reads it. The i*m/r coupling is applied after the forms and
  before the stores, a quarter turn per part subtracted through MEEP's additive
  ladder. The axis rules form MEEP's
  double product with Courant in float32, which is the same word only when Courant is a
  float32 word, so any other Courant is refused by name
  (<code>cylindrical_real.courant_reasons</code>).
- **The other families that lift a base body do not bind it yet**; their sources do not
  compile and <code>plan_step</code> refuses their slots at plan build, leaving the singles
  to serve.
- **The conductive PML curl** (<code>metal_kernels/conductive_pml.py</code>) binds it,
  then one 27-bit conductivity presence word per target last, and selects MEEP's four
  conductive forms per chunk on the dsig and dsigu memberships where the cell's chunk
  keeps the conductivity (<code>ChunkLayout.conductive_chunks</code>), the two lossless
  forms where it does not; its two fused products carry both in <code>Params</code>.
  The words are built from the array path's own per-cell presence and indexed by the
  off-diagonal families' chunk code; a division they cannot carry is refused to the
  array path by name. The complex conductive PML curl
  (<code>metal_kernels/complex_conductive_pml.py</code>) steps the same four forms part
  by part, binds no words, and refuses a partial presence, its H/E companion with it.
- **The off-diagonal families bind it and form MEEP's double row in software
  binary64.** The off-diagonal <code>update_E</code> singles
  (<code>offdiag_update_e</code>, its folded and complex folded twins) bind the
  record, then one 27-bit presence word per row slot last, and step the base
  constitutive kernel's per-chunk tail; the three PML scratch welds that lift them
  (<code>offdiag_fused_electric_pair</code> and its folded and folded-complex twins)
  carry both in <code>Params</code> and pass the record to <code>step_cell</code>, so
  each re-derived cell selects its curl form where it is. The row is the array path's:
  <code>gs*us</code> in float32, widened; each live term's float32 pair sum
  wall-masked, scaled by the double <code>0.25</code> and added where the cell's chunk
  keeps the row (the word's bit at the cell's chunk code, read off the record on the
  component's own sub-lattice per axis); the row rounded once, every double operation
  through <code>metal_kernels/softfp64.py</code> (ruling R7). The words are built from
  the array path's own presence and refused by name where the record does not
  reproduce it (<code>offdiag_update_e.presence_words</code>). The complex folded row
  connects a Bloch phase on the same-index operand's plane 0, as the array path does.
  The folded off-diagonal dispersive product
  (<code>folded_offdiag_dispersive_update_e</code>) assembles the same row with the
  same emitter over the ordered <code>D - P0 - ...</code> gathers, its presence words
  riding in its <code>Params</code> before the record, and the complex no-absorber
  off-diagonal <code>update_E</code> (<code>complex_no_pml_offdiag_update_e</code>)
  forms it per part with the complex folded row's connection, binding the record and
  the words last; the stencil weld that lifts it carries both in <code>Params</code>.
  All five arms are wired.
- **The dispersive families bind it, and step MEEP's polarization arithmetic.** The
  pointwise dispersive <code>update_E</code> singles (<code>dispersive_update_e</code>, its
  folded and complex twins) bind the record last and step the base constitutive
  kernel's per-chunk tail on <code>(D - P0 - ...) * inv_eps</code>. The
  <code>update_P</code> recurrence (<code>ade_update_p</code>, and the ADE half of
  <code>fused_ade_chain</code> and <code>complex_fused_ade_chain</code>) is MEEP's
  isotropic update on its four float32 constants
  (<code>PolarizationState.constants</code>), complex storage as two real passes, and
  under an absorber <code>ade_update_p</code> selects its drive per chunk on the record,
  <code>w = f_w</code> on the component's own-layer cells and the stored E elsewhere,
  instead of asking <code>Fields.drive_field</code> for a pointer. The folded
  off-diagonal dispersive product (<code>folded_offdiag_dispersive_update_e</code>)
  carries the record in a new <code>Params</code> and steps the same tail; its row is
  the off-diagonal families' software-binary64 one (above).
- **The no-absorber curl reads its presence word only.** The real-field no-PML curl
  (<code>metal_kernels/no_pml_curl.py</code>), its conductive twin
  (<code>metal_kernels/no_pml_conductive.py</code>, which imports the same statements and
  bounds block and keeps its three conductive tail statements) and the two D-&gt;E pairs
  that lift them (<code>no_pml_fused_electric_pair.py</code>,
  <code>no_pml_conductive_fused_electric_pair.py</code>) bind the record last and take
  each curl's stencil form from the partner-presence bits, with the PML curl's own
  statements; no chunk of such a run carries a layer, so no membership entry is read.
- **The complex no-absorber curls read it the same way.** The complex no-PML curl body
  (<code>metal_kernels/complex_no_pml_conductive.py</code>, shared by the lossless
  <code>complex_no_pml_curl.py</code>) binds the record last and steps the complex PML
  curl's own stencil and plane-0 Bloch connection; the complex conductive D-&gt;E pair
  lifts that curl from the certified emitter's output and carries the record in
  <code>Params</code>. The complex no-absorber off-diagonal <code>update_E</code>
  (<code>complex_no_pml_offdiag_update_e.py</code>) steps the off-diagonal families'
  software-binary64 row per part (above); with no absorber its only form is
  <code>E = src</code>, and the record serves its presence labels.
- **MEEP's start of H is carried by the composition, not by a kernel.** No Metal
  kernel starts H and <code>f_w</code> from B on a run's first <code>update_H</code>
  (MEEP's <code>update_eh.cpp:165-177</code>, the array path's
  <code>stepping.separate_h_from_b</code>), so <code>launch.plan_step</code> puts
  <code>launch.StartOfHPlan</code> around whatever serves that slot on a run frozen
  before its first step: the start on the array path before a launch that steps
  <code>update_H</code> there (the singles), and <code>stepping.update_H</code> re-stepped
  after the slot's own work where a fused magnetic pair launched at <code>step_B</code>,
  once a run, as the hand-CUDA launchers carry it. <code>launch.h_start_reasons</code>
  bounds which runs a single takes from step 1 and is no longer what makes that step
  exact: a +0 B times a layer's negative <code>kms</code> is -0, which the start keeps
  in H. An integrated magnetic source that reaches the start's cells refuses the serving
  product by name, since MEEP starts H before its <code>f_minus_p</code> words.

The four <code>kernels.py</code> entry points those plans launch
(<code>pml_curl_step</code>, <code>constitutive_step</code>,
<code>fused_curl_constitutive_B</code> and <code>fused_curl_constitutive_D</code>)
declare <code>cb</code> in that position and read it. Their forms are the shared
<code>@triton.jit</code> helpers of <code>triton_kernels/forms.py</code>, which
inline into each kernel: the curl takes the split-field recurrence on the cells
whose chunk carries the component's dsigu layer and the dsig-only form elsewhere,
the constitutive update takes MEEP's accumulation on its own layer's cells and
<code>f = src</code> elsewhere, the curl expression follows the record's
partner-presence word, and <code>fu</code> and <code>f_w</code> are read and
written only on their layer's cells. The H-side plans also carry MEEP's start of H
(<code>start_h</code>) on the run's first <code>update_H</code>, the BFAST run's
constitutive plan (<code>bfast_curl.plan_bfast_run_constitutive</code>) included.

The folded real family reads the record the same way, through the same helpers:
<code>symmetry.pml_curl_step_folded</code> and the curl halves of
<code>folded_fused_pair.folded_fused_curl_constitutive_D</code> and
<code>folded_fused_magnetic_pair.folded_fused_curl_constitutive_B</code> are the root
curl with the fold's ghost rule and ownership masks, and their constitutive halves are
the root constitutive with the source in a register. A carried ghost cell keys its
constitutive form where it is, on the record line of the component's own axis read at
the destination's row (<code>_carry_ghost_E</code> / <code>_carry_ghost</code> take the
destination's flag); the record is ghost-adjusted, so a ghost row reads its image
row's line. The folded <code>update_H</code> plan and the folded magnetic pair carry
MEEP's start of H, the pair at its ghost cells too. On the engine route the folded
builders bind the run's division (<code>plan_step(..., chunk_layout=...)</code>); the
harness builders (<code>plan_folded_from_arrays</code>,
<code>plan_folded_fused_pair_from_arrays</code> and
<code>plan_folded_fused_magnetic_pair_from_arrays</code>) require one.

The nonlinear <code>update_E</code> (<code>triton_kernels/nonlinear_update_e.py</code>)
binds the record the same way, as <code>cb</code> after its six coefficient
pointers, and takes the same constitutive forms through
<code>forms.constitutive_form</code>: the accumulation on the cells whose chunk
carries the component's own layer, <code>E = src</code> elsewhere, and
<code>f_w</code> read and written only on the accumulation's cells. Its source
arithmetic is the array path's: the four-corner transverse sums associate left to
right, and Dsqr is formed in double and rounded once. The nonlinear run's other
slots (the spine curl and <code>update_H</code> arms, and the fused B/H pair) are
the root plans, bound to the run's division and, on <code>update_H</code>, to the
fields MEEP's start of H is recorded on.

The conductive curl (<code>conductivity.conductive_pml_curl_step</code>) and the
conductive fused electric pair read the same record, on each component's own
sub-lattice, for its dsig and its dsigu layer, and select MEEP's conductive forms per
structure chunk. Their third test, whether the cell's chunk keeps the component's
conductivity, is a per-chunk fact the record cannot carry: a component whose chunks do
not all keep it binds the division's conductive-chunk volume
(<code>ChunkLayout.conductive_mask</code>, one <code>uint8</code> per cell, through
<code>conductivity.conductive_chunk_volumes</code>), and a component whose chunks all
keep it binds none and reads none. Where a chunk keeps none, the lossless forms of
<code>forms.curl_form</code> step the cell, as the array path steps it.

The off-diagonal <code>update_E</code> kernels (<code>offdiag_update_e.py</code>,
<code>folded_offdiag_update_e.py</code>) and the two scratch-output welds that carry
them (<code>offdiag_fused_electric_pair.py</code>,
<code>folded_offdiag_fused_electric_pair.py</code>) read the record the same way, and
form MEEP's row: <code>gs*us</code> widened to double, each surviving term's float32
sum wall-masked and scaled by MEEP's double <code>0.25</code>, the terms added in
double, the row rounded once (<code>offdiag_update_e._row_add</code>). MEEP drops an
off-diagonal term in every chunk that deleted its array, so these plans bind a second
<code>int32</code> vector, <code>op</code>
(<code>offdiag_update_e.presence_record</code>): one 27-bit mask per row slot over the
<code>{low, interior, high}</code> product, then one segment line per sub-lattice and
axis read off the division's owner volume. A slot no chunk drops compiles to no read;
the engine route packs the array path's own presence
(<code>offdiag_update_e.presence_for</code>) and the harness route requires
<code>presence</code> (<code>offdiag_update_e.ALL_PRESENT</code> states every term
present). Each weld's twin of each <code>fu_D*</code> takes the pre-launch word
outside the recurrence's cells, so the rotation keeps the auxiliary contract; the
folded weld's curl half is the folded curl's body, lifted at the same anchors.

The complex kernels (<code>complex_fields.bloch_pml_curl_step</code> and
<code>bloch_constitutive_step</code>, the two complex fused pairs,
<code>complex_offdiag_update_e.py</code>, and <code>folded_complex.py</code>'s curl and
fill with the two folded complex fused pairs) read the record the same way and step
each stored part with the real kernels' forms: a real coefficient times each part, one
rounding each, with no multiply-add. The Bloch connection is MEEP's: the shifted
operand is a plain roll of the stored window, and plane 0 of the unshifted operand
takes the not-owned word, a copy within <code>part_products.CONNECT_TOLERANCE</code> of
+1, a per-part negation within it of -1 and otherwise the unfused product with the
phase's float32 parts (<code>complex_fields.connection_arguments</code>,
<code>_connect</code>). The mirror fill multiplies each part by the host-rounded parity
word. No expansion arm remains: the plans record none and the predicates ask for no
licence, since the licence chain was removed in one change.

The fold passes run where MEEP runs them, after the curl
(<code>special_kz.beta_pml_curl_step</code> and <code>beta_bloch_pml_curl_step</code>,
<code>bfast_curl.bfast_pml_curl_step</code>, <code>folded_complex.py</code>'s two beta
curls, and the ten fused pairs that carry a pass). The beta pass follows every
component's curl form: its increment is the host-rounded word times the centre
partner, negated by <code>* -1.0</code> (on complex storage a quarter turn of the
partner's parts by one real word, so no complex product and no multiply-add), an exact
<code>+0</code> on every cell the curl's ownership mask drops, and
<code>forms.fold_ladder</code> subtracts it: <code>f - (sinv*q)</code> outside the
component's dsigu chunk, <code>fu - (sinv*q)</code> and
<code>f - (sinv_u*(sinv*q))</code> inside it. The BFAST pass runs per component after
its curl through <code>forms.bfast_advance</code>: the state takes MEEP's two roundings
(<code>F = S - F_prev</code> stored, then <code>F - F_prev</code>), an absent partner
is not read, MEEP's plain branch (a chunk carrying neither the dsig nor the dsigu
layer) drops <code>- F</code>, and the state keeps its word on unowned cells. A nonzero
k on a declared-invariant axis is admitted, because it multiplies a partner MEEP does
not read.

The dispersive and ADE kernels (<code>kernels.ade_update_p</code>, the pole-aware
<code>update_E</code> of <code>dispersive_update_e.py</code> and
<code>folded_offdiag_dispersive_update_e.py</code>, the two dispersive D/E pairs, the
one-susceptibility ADE state, the four E-to-P chains, <code>complex_ade.py</code> and
<code>no_pml_ade.py</code>) read the record the same way. Each E half selects the
constitutive form on the component's own dsigw line of the half-integer sub-lattice,
with <code>f_w</code> read and written on the accumulation's cells only, and each
recurrence is MEEP's <code>update_P</code> on the four <code>realnum</code> words
<code>PolarizationState.constants</code> holds, taken as float32 scalars:
<code>gamma1inv * (((p * two_minus_od) - (gamma1 * pp)) + (omega0dtsqr * (s * w)))</code>,
per part on complex storage. The drive is MEEP's per-chunk <code>w</code>: a plan that
reads it from memory reads <code>Fields.drive_field</code>'s composed words, and a chain
hands the recurrence the E half's <code>src</code>, which is <code>f_w</code> on the
layer's chunks and the stored field everywhere else.
<code>coverage.polarization_constant_reasons</code> is the one clause every Triton ADE
predicate uses to refuse four words that are missing, non-finite or not float32 words.

The cylindrical kernels (<code>cylindrical_triton.py</code>,
<code>cylindrical_complex.py</code> and the four cylindrical fused pairs,
<code>cylindrical_real_fused_{magnetic,electric}_pair.py</code> and
<code>cylindrical_fused_{magnetic,electric}_pair.py</code>) read the record the same way and
take the array path's Dcyl forms. Bz's and Dz's curl is <code>dtdx</code> times MEEP's
per-chunk radial difference (<code>stepping.cylindrical_rderiv_difference</code>), which a
column-serial scan kernel, <code>cylindrical_triton.cyl_radial_difference</code>, computes
before the curl: one lane per (phi, z) column, and per stored part on complex storage,
sums each chunk's rows in order from <code>+0</code> with <code>rinv</code> multiplied, and
writes the difference a target row reads from its own chunk's sum. A parallel scan
associates the sum differently, so nothing replaces the loop. The plan binds the run's
chunk start per r row (<code>cylindrical_triton.radial_scan_starts</code>, the array path's
<code>_cylindrical_scan_starts</code>) and refuses by name a division whose r cuts differ
along z or that is not the division the record describes. The other curls take
<code>forms.curl_stencil</code> with the presence word (phi is invariant). On complex
storage the i*m/r coupling is a pass of its own after the curl's form, in register: the
quarter turn of the partner by a host-built float32 row <code>the_m / r</code>, through
MEEP's additive ladder. The axis rules' products with Courant (the m = 0
<code>Dz</code> add and the |m| = 1 axis-row increments) are formed in double from a bound
<code>float64</code> vector and rounded once. The complex curl has no multiply-add. The
cylindrical H-to-D welds (<code>cylindrical_fused_hd_pair.py</code>,
<code>cylindrical_real_fused_hd_pair.py</code>) step <code>update_H</code> into a
scratch twin (the Cartesian weld's constitutive block, statement for statement) and then
run the certified curl plan, scan and curl, over the scratch H: three launches against the
singles' three, so they save no launch and are not installed.

The Triton H-to-D welds (none dispatched) carry their bases' forms. Each is the base
step's own text with its magnetic reads redirected to registers and recomputes:
<code>fused_hd_pair._h_cell</code> (and the complex
<code>complex_fused_hd_pair._h_cell_complex</code>, per part) is the root constitutive
lifted to a cell, so it reads the record's dsigw lines at the cell it is handed and
carries MEEP's start of H (<code>start_h</code>, not specialized on); the curl half is the
base curl's text, forms, passes and connection included. The scratch history the
rotation publishes is written at every cell: B where the component's own layer
accumulates, the pre-launch word elsewhere.

The no-absorber curls read the record too. <code>no_pml.plain_curl_step</code>,
<code>no_pml_conductive.conductive_plain_curl_step</code> and the curl half of
<code>no_pml_fused_electric_pair.no_pml_fused_curl_constitutive_D</code> are one
transcription of the curl; each declares <code>cb</code> directly after its last pointer
argument and reads the record's partner-presence word alone, through
<code>forms.curl_stencil</code>, because a run with no absorber has no membership run on
any axis. Which partners exist is the run's (<code>Grid.is_invariant</code>), not the
stored extent's: a 3-D run that stores one cell on an axis still takes both partners
there. Their plans require the record on the harness route; on the engine route every
division of a no-absorber run binds the same words as its one-rank record, so the
certified-product composer, which builds the fused pair without the run's division,
binds that one. The stored-E <code>update_E</code> and the null constitutive plans bind
no record: their one form is <code>f = src</code>, or nothing.

The complex no-absorber curls are the complex curl. The conductive curl
(<code>complex_no_pml_conductive.complex_conductive_no_pml_curl_step</code>) is
<code>complex_fields.bloch_pml_curl_step</code> statement for statement through the
ownership mask, followed by the three-pass conductive tail on each part; it reads the
record's presence word alone, and the conductive D->E pair
(<code>complex_conductive_fused_pair.py</code>) lifts its statements. The lossless curl
(<code>complex_no_pml_curl.py</code>) launches the complex curl itself under the
absorber-free record and unit columns, so its dsig-only form, exact on unit
coefficients, is MEEP's plain form on every cell and its placeholder auxiliary is
never read or written. The complex stored-E <code>update_E</code> binds no record.

## What must be measured

Benchmark a slice against the array path and CPU MEEP on the same physics, both
before it is released and again once it is. Include periodic-core and realistic
PML/monitor workloads, then add the material and boundary features the slice
claims to cover. Report one-rank and well-utilized CPU comparisons separately,
and state which kernel table served.

Do not publish projected kernel speedups. Performance claims follow only from a
measurement and are bounded by the limits stated with it; the measurements
that support a user-facing comparison are in
[Will it help?](../guides/will-it-help.md).
