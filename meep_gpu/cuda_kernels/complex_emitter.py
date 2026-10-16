"""The COMPLEX-storage PML device sources -- no CuPy.

WHAT THIS FAMILY IS. Complex64 field storage under a real (split-field) PML: the
four sub-steps ``step_B`` / ``step_D`` / ``update_H`` / ``update_E`` when the run
carries ``force_complex_fields`` or a nonzero ``k_point``. ``coverage.covers_real_pml_curl``
and ``coverage.covers_real_pml_constitutive`` refuse this configuration by name
("complex64 storage: the recurrence is the same but the storage is not"); this module is
the kernel, and ``coverage.covers_real_pml_complex_curl`` /
``coverage.covers_real_pml_complex_constitutive`` are the written refusal for the
configurations it must NOT serve.

WHY AN EMITTER RATHER THAN FOUR STRING CONSTANTS. Every complex kernel on this table
lifts its helpers from here -- the fused pairs, the folds, the fill carries, the H->D
weld, the off-diagonal row and the no-absorber family -- so the device code is a
function that can be imported and called, and checked, on a machine with no GPU. It is
a stdlib-only sibling of ``coverage.py``, ``compile_cache.py`` and ``offdiag_emitter.py``,
on this package's rule that the parts whose failure mode is a SILENT WRONG ANSWER are the
parts that must be exercisable on a laptop.

NOTHING HERE IMPORTS CUPY OR NUMPY, and nothing here imports ``coverage`` or
``bounds_block`` either: this family's tables are its own, and the chunk-record decode is
carried as a literal that ``test_complex_pml.py`` pins equal to
``bounds_block.bounds_block`` (the one spelling of that decode on this table).

=============================================================================
THE REFERENCE: THE ARRAY PATH, PART BY PART
=============================================================================

The array path is ``stepping.py`` (with ``part_products.py`` for every product on complex
storage), and every line below carries its citation. MEEP stores a complex field as two
``realnum`` arrays and steps each part with real operands (the ``DOCMP`` loops,
step_db.cpp:47, update_eh.cpp:149), so the kernel is the real kernel of the same sub-step
with THREE changes and only three:

1. every field access is a float32 WORD PAIR at ``2*idx`` / ``2*idx + 1`` (complex64 is
   bit-layout (re, im) interleaved, so a C-contiguous complex volume IS a C-contiguous
   float32 volume of shape ``(nx, ny, nz, 2)``);
2. every real-coefficient product is formed ONCE PER PART -- ``c * z.re`` and
   ``c * z.im``, ``part_products.scale``'s ``multiply(coeff, values.real)`` and
   ``multiply(coeff, values.imag)``; complex add and subtract are per part already;
3. on a Bloch-phased axis the operand read at the cell ITSELF takes MEEP's not-owned word
   on plane 0 (``stepping._bloch_ghost_operand``; see THE STORED WINDOW below), formed by
   ``part_products.connect``'s rule: a copy, a per-part negation, or
   ``complex<realnum>(thephase) * word`` with every product and sum rounded on its own.

There is no complex x complex product on the stepping path and none here
(``parity/meep_gpu/complex_product_interceptor.py`` checks the array path's half of that).
In particular no product is FUSED: every real-coefficient product is formed part by
part, as MEEP does and as the array path does, so the emitter has one body.

=============================================================================
THE FORMS, PER STRUCTURE CHUNK
=============================================================================

MEEP chooses the PML form of each curl and constitutive update per structure chunk
(``dsig``/``dsigu``, step_db.cpp:56-59; ``dsigw``, update_eh.cpp:162-163), and so does the
array path (``stepping._apply_curl_regions`` / ``_apply_constitutive_regions``, from the
run's ``chunk_map.RegionPlan``). Each kernel reads the run's chunk record ``cb``
(``chunk_map.chunk_bounds_record``, ``int32[25]``) and decodes it once after its index
decode, exactly as the real root kernels do (``step_curl_kernels``,
``constitutive_kernels``):

* the curl reads each component's dsigu axis on its own Yee sub-lattice -- the B curl the
  half-integer record, the D curl the integer one -- and the partner-presence word;
* ``update_H`` reads each component's own axis on the integer record, ``update_E`` on the
  half-integer one.

The forms are the real root's, per part:

* curl, dsigu cell: the split-field recurrence, ``stepping._apply_pml_update`` in its
  in-place order (``fu *= kms; fu -= curl; fu *= sinv; f *= kms_u; f += fu;
  f -= fu_previous; f *= sinv_u``);
* curl, every other cell: the dsig-only form ``((f * kms) - curl) * sinv``
  (``stepping._apply_dsig_only_update``), which IS MEEP's plain ``f - curl`` wherever the
  dsig coefficients are exactly 1 (the region plan's precondition,
  ``chunk_map.check_unit_coefficients``). ``fu`` is read and written only on the
  recurrence's cells (the auxiliary contract, KERNELS.md section 1, C2);
* constitutive, dsigw cell: ``f + ((kps * src) - (kms * prev))``, the right-hand side
  formed first (``stepping._apply_constitutive_pml``, MEEP step_generic.cpp:689, :697),
  with ``prev`` loaded before ``fw`` takes ``src``;
* constitutive, every other cell: ``f = src`` (H = B, E = D * inv_eps), an output select
  that reads neither ``fw`` nor the old field (C4: a ``kps = 1, kms = 0`` accumulation is
  wrong on every -0).

THE STENCIL is MEEP's left-to-right ``((g1s - g1) + g2) - g2s`` per part
(``stepping._curl_from_operands``, step_generic.cpp:104), with the forms for a partner
absent from the step plan (``stepping.curl_presence``): ``dtdx * (g1s - g1)`` with ``g2``
absent, ``(-dtdx) * (g2s - g2)`` with ``g1`` absent, and an exact zero with neither.

=============================================================================
THE STORED WINDOW: WHERE THE BLOCH PHASE GOES
=============================================================================

On an axis MEEP makes Bloch-periodic it OWNS, for a component with Yee shift 0 there, the
points ``lc + 2 .. lc + 2N`` and holds ``lc`` as a not-owned word filled from the owned
``lc + 2N`` (boundaries.cpp:203-230, step.cpp:171-221). The engine stores that window:
stored cell 0 of a shift-0 component on a phased axis IS the owned top point
(``Grid.stored_window_low``). So:

* a read ONE STORED CELL OVER is always an owned word, and both shift helpers are plain
  wraps there -- no phase (``stepping._shift_up`` / ``_shift_down``);
* the reads that leave the window are the ones at the SAME stored index across the half
  cell, on plane 0 (``stepping._bloch_ghost_operand``): B's forward curl at stored cell 0
  reads its partner's not-owned word BELOW the window, ``conj(eikna) * g[0]``; D's
  backward curl at stored cell 0 reads the not-owned word ABOVE it, ``eikna * g[0]``.

:func:`cwindow` is that read (its connection is ``cconnect``, which the fused products and
the off-diagonal row also apply to a value they hold in a register): it takes the axis's
CONNECTION KIND and the rounded phase parts, both resolved on the host from ``part_products.connect_parts``' rule
(``complex_pml_kernels.bloch_phase_arguments``): kind 0 for no connection (an unphased
axis, or a phase within ``part_products.CONNECT_TOLERANCE`` of +1, which is a copy), kind
:data:`CONNECT_PHASE` for the unfused product, kind :data:`CONNECT_NEGATE` for a phase
within the tolerance of -1, a per-part negation. The host passes ``conj(eikna)`` to the B
curl and ``eikna`` to the D curl. No tolerance test runs on the device.

=============================================================================
PLATFORM REQUIREMENTS
=============================================================================

* NO FMA CONTRACTION. ``--fmad=false`` is CORRECTNESS, not tuning: the recurrence has
  contraction candidates at ``(fu*kms) - curl`` and ``(f*kms_u) + fu_new``, the
  constitutive tail at ``(kps*src) - (kms*prev)``, and the connection at both of its sums.
  Nothing in this family fuses on purpose.
* NEGATION IS ``* -1.0f``: exact on every word (a sign flip, signed zero included) and the
  same spelling as the sibling transcriptions; ``test_complex_pml.py`` pins the absence of
  unary minus on any float path.
* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style rule:
  ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
  ``open(..., 'w')`` (compiler.py:368), so the bytes go through the interpreter's LOCALE
  encoding -- ASCII under C/POSIX, which is what a non-interactive shell on the validation
  host gets. Two em-dashes in a comment killed a sibling kernel at its first launch on
  2026-08-15.
* NO DIVISION anywhere: the reciprocal the recurrence needs is ``sinv``, computed
  host-side by ``PML``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

#: The connection kinds a phased axis binds (``ph_*``), from ``part_products.connect_parts``:
#: none (unphased, or a copy within ``CONNECT_TOLERANCE`` of +1), the unfused product, and a
#: per-part negation (within the tolerance of -1). The device text names the last two.
CONNECT_NONE = 0
CONNECT_PHASE = 1
CONNECT_NEGATE = 2

#: The four kernels this family emits, keyed by the sub-step they replace. The value is
#: ``(kernel name, is the D/backward side, does the source carry inv_eps)`` -- the two
#: structural axes, written out so a launcher and a test read the same table.
KERNELS: Dict[str, Tuple[str, bool, bool]] = {
    "step_B": ("step_B_pml_complex_bloch", False, False),
    "step_D": ("step_D_pml_complex_bloch", True, False),
    "update_H": ("update_H_pml_complex_bloch", False, False),
    "update_E": ("update_E_pml_complex_bloch", False, True),
}

#: The two curl sub-steps and the two constitutive sides, as the launcher and the
#: predicates name them.
CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
CONSTITUTIVE_SIDES: Tuple[str, ...] = ("H", "E")

#: Which Yee sub-lattice each sub-step's coefficient vectors come from.
#: ``stepping._curl_coefficients``: the B curl reads HALF-INTEGER positions and the D curl
#: INTEGER ones. ``stepping._constitutive_coefficients``: ``update_H`` reads INTEGER and
#: ``update_E`` HALF-INTEGER. Swapped, it is a half-cell error in the absorber profile --
#: converged, smooth and wrong -- which is why the gate carries it as a host mutation. The
#: chunk record is read on the same sub-lattice as the coefficients (bounds_block's rule).
HALF_INTEGER: Dict[str, bool] = {
    "step_B": True, "step_D": False, "H": False, "E": True}

#: The membership flag each curl target passes to ``pml_apply``: its dsigu axis on its own
#: Yee sub-lattice (``bounds_block.curl_entries``). Bx/By/Bz take z/x/y; the B targets sit on
#: the half-integer lattice along their dsigu axis and the D targets on the integer one.
CURL_FLAGS: Dict[bool, Tuple[str, str, str]] = {
    False: ("in_h_z", "in_h_x", "in_h_y"),   # step_B
    True: ("in_i_z", "in_i_x", "in_i_y"),    # step_D
}

#: The membership flag each constitutive component passes to ``constitutive_apply``: its
#: own axis on its own sub-lattice (``bounds_block.constitutive_entries``).
CONSTITUTIVE_FLAGS: Dict[str, Tuple[str, str, str]] = {
    "H": ("in_i_x", "in_i_y", "in_i_z"),
    "E": ("in_h_x", "in_h_y", "in_h_z"),
}

# ---------------------------------------------------------------------------
# The chunk-record decode, as the literal bounds_block emits (pinned by the tests)
# ---------------------------------------------------------------------------

_BOUNDS_COMMENT = (
    "    // THE CHUNK RECORD (chunk_map.chunk_bounds_record): row r of this lattice and axis\n"
    "    // lies in a chunk that carries the axis's layer iff (a0 <= r < a1) || (b0 <= r < b1).\n")

_BOUNDS_LATTICE: Dict[bool, str] = {
    False: (
        "    const int cbi_x_a0 = cb[0], cbi_x_a1 = cb[1], cbi_x_b0 = cb[2], cbi_x_b1 = cb[3];\n"
        "    const int in_i_x = (i >= cbi_x_a0 && i < cbi_x_a1) || (i >= cbi_x_b0 && i < cbi_x_b1);\n"
        "    const int cbi_y_a0 = cb[4], cbi_y_a1 = cb[5], cbi_y_b0 = cb[6], cbi_y_b1 = cb[7];\n"
        "    const int in_i_y = (j >= cbi_y_a0 && j < cbi_y_a1) || (j >= cbi_y_b0 && j < cbi_y_b1);\n"
        "    const int cbi_z_a0 = cb[8], cbi_z_a1 = cb[9], cbi_z_b0 = cb[10], cbi_z_b1 = cb[11];\n"
        "    const int in_i_z = (k >= cbi_z_a0 && k < cbi_z_a1) || (k >= cbi_z_b0 && k < cbi_z_b1);\n"),
    True: (
        "    const int cbh_x_a0 = cb[12], cbh_x_a1 = cb[13], cbh_x_b0 = cb[14], cbh_x_b1 = cb[15];\n"
        "    const int in_h_x = (i >= cbh_x_a0 && i < cbh_x_a1) || (i >= cbh_x_b0 && i < cbh_x_b1);\n"
        "    const int cbh_y_a0 = cb[16], cbh_y_a1 = cb[17], cbh_y_b0 = cb[18], cbh_y_b1 = cb[19];\n"
        "    const int in_h_y = (j >= cbh_y_a0 && j < cbh_y_a1) || (j >= cbh_y_b0 && j < cbh_y_b1);\n"
        "    const int cbh_z_a0 = cb[20], cbh_z_a1 = cb[21], cbh_z_b0 = cb[22], cbh_z_b1 = cb[23];\n"
        "    const int in_h_z = (k >= cbh_z_a0 && k < cbh_z_a1) || (k >= cbh_z_b0 && k < cbh_z_b1);\n"),
}

_BOUNDS_PRESENCE = (
    "    const int cb_presence = cb[24];\n"
    "    const int pr_x = (cb_presence >> 0) & 1;\n"
    "    const int pr_y = (cb_presence >> 1) & 1;\n"
    "    const int pr_z = (cb_presence >> 2) & 1;\n")


def curl_bounds_block(sub_step: str) -> str:
    """The curl's chunk-record block: its lattice on all three axes, and the presence bits.

    Every component of a curl reads its dsigu axis on the same lattice, and the three dsigu
    axes are x, y and z once each, so the block is the whole lattice
    (``bounds_block.bounds_block(curl_entries(...), (0, 1, 2))``).
    """
    return (_BOUNDS_COMMENT + _BOUNDS_LATTICE[HALF_INTEGER[sub_step]] + _BOUNDS_PRESENCE
            + "\n")


def constitutive_bounds_block(side: str) -> str:
    """The constitutive side's chunk-record block: its lattice on all three axes."""
    return _BOUNDS_COMMENT + _BOUNDS_LATTICE[HALF_INTEGER[side]] + "\n"


def presence_bounds_block() -> str:
    """The partner-presence block alone (``bounds_block.bounds_block((), (0, 1, 2))``).

    For a no-absorber curl: no chunk carries a layer, so there is no membership to read,
    and the record's presence word still chooses MEEP's stencil form per component.
    """
    return _BOUNDS_COMMENT + _BOUNDS_PRESENCE + "\n"


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: Word-pair addressing, complex add/sub and the zero -- everything that is not a product.
_HEAD = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1
#define CONNECT_PHASE 1
#define CONNECT_NEGATE 2

// complex64 as a float32 WORD PAIR. A C-contiguous complex volume is a
// C-contiguous float32 volume with a doubled last axis, so complex cell w is
// words 2*w and 2*w+1. Every index below is a COMPLEX CELL index; the doubling
// happens here and nowhere else.
typedef struct { float re; float im; } cf;

__device__ __forceinline__ cf cf_load(const float* g, int idx) {
    cf z;
    z.re = g[2 * idx];
    z.im = g[2 * idx + 1];
    return z;
}

__device__ __forceinline__ void cf_store(float* g, int idx, cf z) {
    // Two word stores replace the array path's single complex store. Non-semantic:
    // the re and im planes of one cell do not alias.
    g[2 * idx] = z.re;
    g[2 * idx + 1] = z.im;
}

// Complex add and subtract are component-wise: MEEP's DOCMP loops step each part
// with real operands, and NumPy's complex add is per part.
__device__ __forceinline__ cf cf_add(cf a, cf b) {
    cf o;
    o.re = a.re + b.re;
    o.im = a.im + b.im;
    return o;
}

__device__ __forceinline__ cf cf_sub(cf a, cf b) {
    cf o;
    o.re = a.re - b.re;
    o.im = a.im - b.im;
    return o;
}

// The array path assigns the INTEGER 0 to a complex64 array, which is the word
// pair (+0.0f, +0.0f) and not a sign-carrying zero.
__device__ __forceinline__ cf cf_zero() {
    cf o;
    o.re = 0.0f;
    o.im = 0.0f;
    return o;
}

// A per-part negation, NumPy's negative: a sign flip on every word, signed zero
// included (MEEP's CONNECT_NEGATE, part_products.connect_parts).
__device__ __forceinline__ cf cf_neg(cf a) {
    cf o;
    o.re = a.re * -1.0f;
    o.im = a.im * -1.0f;
    return o;
}
'''

#: The products, part by part (module docstring).
_PART_PRODUCTS = r'''
// ---- PART BY PART ----------------------------------------------------------
// A real coefficient times a field is ONE float32 product per part
// (part_products.scale: multiply(coeff, values.real), multiply(coeff,
// values.imag); MEEP's DOCMP loops, step_db.cpp:47, update_eh.cpp:149). The two
// helpers below compute the same words -- IEEE multiplication is commutative,
// signed zero included -- and keep two names so a reader can match each call to
// the array path's own operand order.

// fu *= kms and its family (stepping._apply_pml_update), D * inv_eps.
__device__ __forceinline__ cf mul_field_left(cf z, float c) {
    cf o;
    o.re = z.re * c;
    o.im = z.im * c;
    return o;
}

// kps * fw and kms * fw_previous (stepping._apply_constitutive_pml), and the
// scalar-left scale(dtdx, total) of the curl.
__device__ __forceinline__ cf mul_coefficient_left(float c, cf z) {
    cf o;
    o.re = c * z.re;
    o.im = c * z.im;
    return o;
}

// complex<realnum>(thephase) * word, MEEP's CONNECT_PHASE (step.cpp:183-192;
// part_products.connect_parts): the phase rounded to float parts on the host,
// then four products and two sums, each rounded on its own -- libc++'s
// operator* under the contraction-off build. p is the phase, g the stored word.
__device__ __forceinline__ cf rotate_field_left(cf g, cf p) {
    cf o;
    o.re = (p.re * g.re) - (p.im * g.im);
    o.im = (p.re * g.im) + (p.im * g.re);
    return o;
}
'''

#: The ghost rules, the stored window, the stencil and the two update forms. They call the
#: product helpers, so they follow that block in the emitted source.
_TAIL = r'''
// stepping._shift_up, PERIODIC and METALLIC branches only: MIRROR and CYL_AXIS
// are refused by the predicate. The periodic far face is a PLAIN wrap, Bloch
// phase or not: the shifted operand's stored cell 0 is MEEP's owned top point
// wherever the axis carries a phase (the stored window), so the word one cell
// up from the top stored cell is that owned word. An invariant axis (na == 1)
// takes the wrap branch with idx - ia*stride == idx, which is what xp.roll
// computes there.
__device__ __forceinline__ cf cshift_up(
    const float* g, int idx, int ia, int na, int stride, int bc
) {
    if (ia + 1 < na) return cf_load(g, idx + stride);
    if (bc == BC_METALLIC) return cf_zero();
    return cf_load(g, idx - ia * stride);
}

// stepping._shift_down: the mirror of cshift_up, a plain wrap on a periodic
// axis whatever its phase.
__device__ __forceinline__ cf cshift_dn(
    const float* g, int idx, int ia, int na, int stride, int bc
) {
    if (ia > 0) return cf_load(g, idx - stride);
    if (bc == BC_METALLIC) return cf_zero();
    return cf_load(g, idx + (na - 1) * stride);
}

// stepping._bloch_ghost_operand's connection, applied to a value: on plane 0 of a
// phased axis (ia == 0) the word read at the cell itself is MEEP's not-owned word
// across the window edge -- below it for the B curl (the host passes
// conj(eikna)), above it for the D curl (eikna) -- connected by
// part_products.connect's rule, which the host resolved to ph: 0 none (unphased,
// or a copy), CONNECT_PHASE the unfused product, CONNECT_NEGATE a per-part
// negation. Everywhere else the value is returned as it is.
__device__ __forceinline__ cf cconnect(cf w, int ia, int ph, cf phase) {
    if (ia != 0 || ph == 0) return w;
    if (ph == CONNECT_NEGATE) return cf_neg(w);
    return rotate_field_left(w, phase);
}

// The operand read at the cell itself, from storage: the stored word, connected
// across the window edge on plane 0 of a phased axis. The stored field is never
// written.
__device__ __forceinline__ cf cwindow(
    const float* g, int idx, int ia, int ph, cf phase
) {
    return cconnect(cf_load(g, idx), ia, ph, phase);
}

// stepping._curl_from_operands: MEEP's step_curl stencil, per part, rounded op
// by op.
//   both partners present: dtdx * (((g1s - g1) + g2) - g2s)   (step_generic.cpp:104)
//   g2 absent:             dtdx * (g1s - g1)                  (:108)
//   g1 absent:             (-dtdx) * (g2s - g2)               (:72-76, partners swapped)
//   neither:               (+0, +0) (MEEP steps no such component)
// A partner is absent when its derivative axis is invariant (pr1/pr2 from the
// record's presence word). The full bracket over an absent partner is not the
// same number: (a + g2) - g2 rounds where the absent form returns a.
__device__ __forceinline__ cf curl_stencil(
    float dtdx, cf sf, cf f1, cf f2, cf ss, int pr1, int pr2
) {
    if (pr1 && pr2) return mul_coefficient_left(dtdx, cf_sub(cf_add(cf_sub(sf, f1), f2), ss));
    if (pr1) return mul_coefficient_left(dtdx, cf_sub(sf, f1));
    if (pr2) return mul_coefficient_left(dtdx * -1.0f, cf_sub(ss, f2));
    return cf_zero();
}

// MEEP's two non-conductive curl forms, chosen per structure chunk
// (step_db.cpp:56-59; stepping._apply_curl_regions), per part:
//   dsigu (the cell's chunk carries the dsigu layer): the split-field recurrence,
//     stepping._apply_pml_update in its in-place order
//       fu *= kms; fu -= curl; fu *= sinv;
//       f *= kms_u; f += fu; f -= fu_previous; f *= sinv_u
//   otherwise: the dsig-only form on the field alone, stepping._apply_dsig_only_update
//       f *= kms; f -= curl; f *= sinv
//     which is MEEP's plain f - curl wherever the dsig coefficients are exactly 1
//     (the region plan's precondition, chunk_map.check_unit_coefficients).
// fu is read and written only on the recurrence's cells (the auxiliary contract).
// The fu stage precedes the one load of f, and the conditional operator evaluates
// only the arm it selects.
__device__ __forceinline__ void pml_apply(
    float* f, float* fu, int idx, cf curl,
    float kms, float sinv, float kms_u, float sinv_u, int dsigu
) {
    cf fprev = dsigu ? cf_load(fu, idx) : cf_zero();
    cf fu_new = mul_field_left(cf_sub(mul_field_left(fprev, kms), curl), sinv);
    if (dsigu) cf_store(fu, idx, fu_new);
    cf fcur = cf_load(f, idx);
    cf value = dsigu ? mul_field_left(cf_sub(cf_add(mul_field_left(fcur, kms_u), fu_new), fprev), sinv_u)
                     : mul_field_left(cf_sub(mul_field_left(fcur, kms), curl), sinv);
    cf_store(f, idx, value);
}

// stepping._apply_constitutive_regions: MEEP's two constitutive forms, chosen
// per structure chunk (update_eh.cpp:162-170), per part.
//   dsigw (the cell's chunk carries the component's own layer): the accumulation
//   of stepping._apply_constitutive_pml, MEEP's step_update_EDHB with dsigw active
//       realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
//       fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
//       f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
//   with the right-hand side formed first: f + ((kps*src) - (kms*prev));
//   otherwise: f = src (H = B, E = D*inv_eps), and fw is not touched.
// `prev` is loaded before the fw store, and the plain form reads neither fw nor
// the old f.
__device__ __forceinline__ void constitutive_apply(
    float* f, float* fw, int idx, cf src, float kps, float kms, int dsigw
) {
    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();
    if (dsigw) cf_store(fw, idx, src);
    cf updated = dsigw ? cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), mul_coefficient_left(kms, prev)))
                       : src;
    cf_store(f, idx, updated);
}
'''

#: The curl kernels. ``__SIDE__`` is B or D, ``__SHIFT__`` the ghost helper, ``__BOUNDS__``
#: the chunk-record block, and ``__FLAG<t>__`` target t's dsigu membership flag. The three
#: per-component blocks differ only in their stencil and their mask, so they are written out
#: rather than macro-ed: the term table IS the physics and a reader has to be able to check
#: it against ``stepping.B_CURL_TERMS`` by eye.
_CURL_TEMPLATE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    const int* __restrict__ cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Strides in COMPLEX CELLS. cf_load does the word doubling.
    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

__BOUNDS__    cf px; px.re = pxr; px.im = pxi;
    cf py; py.re = pyr; py.im = pyi;
    cf pz; pz.re = pzr; pz.im = pzi;

    // Target 0: curl_x = d(g2)/dy - d(g1)/dz; dsig=y, dsigu=z; mask __AXES0__.
    {
        cf f_1 = cwindow(g2, idx, j, ph_y, py);
        cf sf = __SHIFT__(g2, idx, j, ny, sy, bc_y);
        cf f_2 = cwindow(g1, idx, k, ph_z, pz);
        cf ss = __SHIFT__(g1, idx, k, nz, sz, bc_z);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_y, pr_z);
__MASK0__
        pml_apply(f0, u0, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], __FLAG0__);
    }

    // Target 1: curl_y = d(g0)/dz - d(g2)/dx; dsig=z, dsigu=x; mask __AXES1__.
    {
        cf f_1 = cwindow(g0, idx, k, ph_z, pz);
        cf sf = __SHIFT__(g0, idx, k, nz, sz, bc_z);
        cf f_2 = cwindow(g2, idx, i, ph_x, px);
        cf ss = __SHIFT__(g2, idx, i, nx, sx, bc_x);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_z, pr_x);
__MASK1__
        pml_apply(f1, u1, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], __FLAG1__);
    }

    // Target 2: curl_z = d(g1)/dx - d(g0)/dy; dsig=x, dsigu=y; mask __AXES2__.
    {
        cf f_1 = cwindow(g1, idx, i, ph_x, px);
        cf sf = __SHIFT__(g1, idx, i, nx, sx, bc_x);
        cf f_2 = cwindow(g0, idx, j, ph_y, py);
        cf ss = __SHIFT__(g0, idx, j, ny, sy, bc_y);
        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_x, pr_y);
__MASK2__
        pml_apply(f2, u2, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j], __FLAG2__);
    }
}
'''

#: The constitutive kernels. ``__SCALE__`` is the E side's D*inv_eps product, empty on the H
#: side where the source is B itself (mu = 1 is already baked into ``H_CONSTITUTIVE_TERMS``;
#: ``stepping.update_H`` passes ``getattr(fields, source)``).
_CONSTITUTIVE_TEMPLATE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
__EPS_PARAMS__    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z,
    const int* __restrict__ cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

__BOUNDS__    // NO ghost rule, NO ownership mask and NO PHASE, and all three are
    // transcribed rather than forgotten: _apply_constitutive_regions writes every
    // cell of the volume and reads no neighbour, and the constitutive sub-step never
    // sees the wrap. Anyone porting the curl kernel's shape into this one will reach
    // for all three. None belongs here.
    //
    // Component 0 takes its coefficient pair and its membership flag from axis x, 1
    // from y, 2 from z -- MEEP's dsigw, the absorption a component accumulates along
    // the direction it points in (H_/E_CONSTITUTIVE_TERMS). It is NOT the dsig/dsigu
    // cycle the curl uses, and confusing the two is a smooth, converged, entirely
    // wrong absorber.
    cf s0 = cf_load(g0, idx);
    cf s1 = cf_load(g1, idx);
    cf s2 = cf_load(g2, idx);
__SCALE__
    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i], __FLAG0__);
    constitutive_apply(f1, w1, idx, s1, kps_y[j], kms_y[j], __FLAG1__);
    constitutive_apply(f2, w2, idx, s2, kps_z[k], kms_z[k], __FLAG2__);
}
'''

#: The E side's extra parameters. NOT ``__restrict__``, deliberately: an isotropic run hands
#: the same device pointer three times (``Fields.inverse_epsilon_for``), and restrict on
#: mutually aliasing arguments is a promise the caller cannot keep. They are read-only, so
#: nothing is lost but the promise. They are float32 VOLUMES indexed by the COMPLEX CELL
#: index -- one real coefficient per cell, shared by both parts -- and NEVER word-doubled:
#: ``inv_eps`` stays float32 under complex storage.
_EPS_PARAMS = ("    const float* inv_eps_0, const float* inv_eps_1,\n"
               "    const float* inv_eps_2,\n")

#: The E side's source product, per part (``stepping.update_E``:
#: ``part_products.scale(fields.inverse_epsilon_for(component), source)``).
_SCALE_BODY = """    s0 = mul_field_left(s0, inv_eps_0[idx]);
    s1 = mul_field_left(s1, inv_eps_1[idx]);
    s2 = mul_field_left(s2, inv_eps_2[idx]);
"""

#: The ownership mask, per target and per side. ``stepping._mask_non_owned_cells`` zeroes
#: cell 0 of every axis on which the target's Yee shift is 0 and which is metallic; a mirror
#: and the cylindrical axis are refused by the predicate, so METALLIC is the only live
#: branch. ``fields.IYEE_SHIFTS``: Bx (0,1,1), By (1,0,1), Bz (1,1,0); Dx (1,0,0),
#: Dy (0,1,0), Dz (0,0,1). So each B target masks on ONE axis and each D target on TWO --
#: which is the asymmetry a reader should check by eye.
_MASK_AXES: Dict[bool, Tuple[Tuple[str, ...], ...]] = {
    False: (("x",), ("y",), ("z",)),            # step_B: Bx, By, Bz
    True: (("y", "z"), ("x", "z"), ("x", "y")),  # step_D: Dx, Dy, Dz
}

_AXIS_INDEX = {"x": "i", "y": "j", "z": "k"}


def _mask_lines(axes: Sequence[str]) -> str:
    """The ``curl = cf_zero()`` guards for one target, one per masked axis."""
    return "\n".join(
        f"        if (bc_{axis} == BC_METALLIC && {_AXIS_INDEX[axis]} == 0) "
        f"curl = cf_zero();" for axis in axes)


def prelude() -> str:
    """The shared helpers every kernel of this family opens with."""
    return _HEAD + _PART_PRODUCTS + _TAIL


def complex_source(sub_step: str) -> str:
    """The device source for one sub-step.

    ``sub_step`` is a key of :data:`KERNELS`: ``step_B``, ``step_D``, ``update_H`` or
    ``update_E``.
    """
    if sub_step not in KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(KERNELS)}, got {sub_step!r}")
    name, backward, scaled = KERNELS[sub_step]
    head = prelude()
    if sub_step in CURL_SUB_STEPS:
        body = _CURL_TEMPLATE.replace("__NAME__", name)
        body = body.replace("__SHIFT__", "cshift_dn" if backward else "cshift_up")
        body = body.replace("__BOUNDS__", curl_bounds_block(sub_step))
        for target, axes in enumerate(_MASK_AXES[backward]):
            # The AXES placeholder is a comment and the MASK placeholder is code;
            # they are two names because one substitution serving both would splice
            # a multi-line guard into the middle of a // comment and silently
            # comment out the first guard.
            body = body.replace(f"__AXES{target}__", "+".join(axes))
            body = body.replace(f"__MASK{target}__", _mask_lines(axes))
            body = body.replace(f"__FLAG{target}__", CURL_FLAGS[backward][target])
        return head + body
    side = "E" if scaled else "H"
    body = _CONSTITUTIVE_TEMPLATE.replace("__NAME__", name)
    body = body.replace("__EPS_PARAMS__", _EPS_PARAMS if scaled else "")
    body = body.replace("__SCALE__", _SCALE_BODY if scaled else "")
    body = body.replace("__BOUNDS__", constitutive_bounds_block(side))
    for target in range(3):
        body = body.replace(f"__FLAG{target}__", CONSTITUTIVE_FLAGS[side][target])
    return head + body


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Four sub-steps, four sources; a single changed character anywhere in this module
    moves this value. The per-source digests of
    what a gate actually launches belong in that gate's artifact, which is where a
    certification record reads them from.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for sub_step in sorted(KERNELS):
        digest.update(sub_step.encode("ascii"))
        digest.update(complex_source(sub_step).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: str):
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
