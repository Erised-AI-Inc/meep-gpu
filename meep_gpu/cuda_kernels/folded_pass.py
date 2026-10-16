"""MEEP's folded increments as passes after the curl: the one CUDA spelling.

WHAT A FOLDED PASS IS. Two families add a term to the B and D curls that MEEP does
not form inside the curl: BFAST (``grid.bfast_active``) and special_kz
(``grid.beta != 0``). MEEP's ``step_db`` (step_db.cpp:47-175) runs, per component,
``STEP_CURL`` and then ``STEP_BFAST`` on the same component (:124-142), and after every
component has been through both, the special-kz beta pass over the x and y components
(:161-175). Each pass adds its increment through its OWN additive ladder
(step_generic.cpp:255-333 beta, :335-533 BFAST), keyed on the chunk's dsig and dsigu
layers exactly as the curl of the same component keys (``stepping``'s
``_apply_folded_increment``, ``_bfast_term``, ``_special_kz_beta_term``)::

    outside the dsigu chunks   f -= sinv * inc
    inside them                fu -= (df = sinv * inc); f -= sinv_u * df

where ``inc`` is ``-raw`` with MEEP's unowned cells at ``+0``. Every stage SUBTRACTS:
``x - (+0)`` is ``x`` for every word, ``-0`` included, so a cell MEEP's loop never
visits keeps its word, while adding a masked ``+0`` would turn a stored ``-0`` into
``+0``. Two additions into one word do not commute in rounding, so the order on every
cell is the curl form, then the BFAST ladder, then the beta ladder.

WHY A PASS AFTER ``pml_apply`` AND NOT A TERM INSIDE THE CURL. Until 0.10 these kernels
added the increment to ``curl`` before the masks, so it rode the curl's own
``((f * kms) - curl) * sinv`` recurrence; the array path does not, and on the complex
lane's ``bf`` case the old folding differs in 120,035 of 278,318 words
(bit_parity_0_10 AP6b). The pass helper below runs after the certified ``pml_apply``
call of the same component, on the words ``pml_apply`` just stored: a float32 stored and
reloaded is the identity on the bits, so the two roundings of MEEP's two passes are the
two roundings here. The certified ``pml_apply`` is not edited.

THE BFAST STATE (``stepping._bfast_term``). With both partners present::

    F_prev = F[i];  F[i] = (k1*(g1s + g1) - k2*(g2s + g2)) - F_prev;  raw = F[i] - F_prev

so the state is stored (one rounding) and the increment is a SECOND rounding of its
difference from the previous state. With one partner MEEP reads only that partner (an
absent partner is a NULL pointer, step_generic.cpp:342-346), and in its PLAIN branch --
no layer on the dsig or dsigu axis in the chunk, and no conductivity -- the state line is
``F[i] = k1*(g1s + g1)`` with no ``- F[i]`` (:373-379). With neither partner MEEP steps
no such component: no increment, no state word. The state is written on MEEP's owned
cells only, and the increment is ``+0`` elsewhere.

THE BETA INCREMENT (``stepping._special_kz_beta_term``). Real storage: ``-(c * g)`` with
``c = float32(sign * 2*pi*beta*dt)``. Complex storage: ``-quarter_turn(c, g)``, the
quarter turn ``i*c*(x + iy) = -(c*y) + i*(c*x)`` formed part by part
(``part_products.quarter_turn``), negated -- two real products and two sign flips, never a
complex product. ``g`` is the partner's STORED word at the cell itself (the component
snapshot, no Bloch connection).

THE FUSED PRODUCTS derive ``fold_apply_reg`` / ``cfold_apply_reg`` from the helpers below
by the same four anchored rewrites the fused pairs apply to ``pml_apply``
(``fused_magnetic_pair``: the signature returns the value, the parameter list gains
``owned``, the ownership guard sits above the one load of the field, and the closing
store also returns the value). The ``fu`` stage stays above the guard, as in
``pml_apply_reg``: the curl writes ``fu`` at every recurrence cell and no fill touches it.

STDLIB ONLY: the emitters run on a host with no CuPy, where the coverage census runs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BFAST_PASS_HELPER", "COMPLEX_FOLD_HELPERS", "REAL_FOLD_HELPERS",
    "BETA_TARGETS", "bfast_insert", "beta_insert", "complex_beta_insert",
    "cfold_apply_reg_source", "curl_call", "fold_apply_reg_source",
    "capture_fold_calls", "insert_after_call", "ownership_lines", "rewrite_helper",
    "with_fold_reg",
]

#: The BFAST state step (real storage). Appended, after :data:`REAL_FOLD_HELPERS`, to the
#: BFAST family's copy of ``step_curl_kernels._REAL_PML_PRELUDE``; the special_kz family
#: carries the ladder only.
BFAST_PASS_HELPER = r'''
// stepping._bfast_term (MEEP step_db.cpp:129-142, step_generic.cpp:335-533): the BFAST
// state step on one component, after its curl, over the curl's own four operands.
//   both partners:  F = (k1*(g1s + g1) - k2*(g2s + g2)) - F_prev
//   one partner:    F = k*(gs + g) - F_prev, and in MEEP's plain branch (no dsig or
//                   dsigu layer in the chunk, no conductivity) F = k*(gs + g)
//   neither:        no state word and no increment (MEEP steps no such component)
// An absent partner is not read (MEEP's NULL pointer). The state is stored on MEEP's
// owned cells only and the increment returned is -(F - F_prev), the pass's second
// rounding, or +0 on a cell MEEP's loop does not visit.
__device__ __forceinline__ float bfast_pass(
    float* __restrict__ F, int idx, float sf, float f1, float ss, float f2,
    float k1, float k2, int pr1, int pr2, int plain, int owned
) {
    if (!pr1 && !pr2) return 0.0f;
    float fprev = F[idx];
    float single = pr1 ? (k1 * (sf + f1)) : (k2 * (ss + f2));
    float updated = (pr1 && pr2) ? (((k1 * (sf + f1)) - (k2 * (ss + f2))) - fprev)
                                 : (plain ? single : (single - fprev));
    if (!owned) return 0.0f;
    F[idx] = updated;
    return -(updated - fprev);
}
'''

#: The real folded ladder. Appended to a family's copy of
#: ``step_curl_kernels._REAL_PML_PRELUDE`` (never written into it: every lifter of the
#: root anchors on that text).
REAL_FOLD_HELPERS = r'''
// stepping._apply_folded_increment: MEEP's additive ladder for one folded pass
// (step_generic.cpp:255-333 beta, :335-533 BFAST), after the curl of the same
// component and keyed on the same chunk flags. `inc` is -raw, MEEP's unowned cells at
// +0, and every stage SUBTRACTS it (x - (+0) is x, -0 included):
//   outside the dsigu chunks: f -= sinv * inc
//   inside them:              fu -= (df = sinv * inc); f -= sinv_u * df
// sinv is exactly 1 outside the dsig chunks (the region plan's precondition), so the
// first form is MEEP's plain f += raw there. fu is read and written only on the
// recurrence's cells (the auxiliary contract).
__device__ __forceinline__ void fold_apply(
    float* __restrict__ f, float* __restrict__ fu, int idx, float inc,
    float sinv, float sinv_u, int dsigu
) {
    float df = sinv * inc;
    if (dsigu) fu[idx] = fu[idx] - df;
    float fcur = f[idx];
    float value = dsigu ? (fcur - (sinv_u * df)) : (fcur - df);
    f[idx] = value;
}
'''

#: The complex helpers, over ``complex_emitter``'s word pair ``cf`` and its part-wise
#: helpers. Appended to a complex family's prelude.
COMPLEX_FOLD_HELPERS = r'''
// stepping._special_kz_beta_term under complex storage: -quarter_turn(c, g), with
// quarter_turn(c, g) = -(c*g.im) + i*(c*g.re) formed part by part
// (part_products.quarter_turn) and then negated: two real products and two sign flips,
// never a complex product. -(-(x)) is x on every word, so the real part is c*g.im.
// c is the host-rounded float32 coefficient of this call site (the D side's is
// negated in double before the round, stepping's `coefficient if magnetic else
// -coefficient`). g is the partner's STORED word at the cell itself.
__device__ __forceinline__ cf beta_increment(float c, cf g) {
    cf o;
    o.re = c * g.im;
    o.im = -(c * g.re);
    return o;
}

// stepping._apply_folded_increment per part: MEEP's additive ladder for one folded
// pass, after the curl of the same component and keyed on the same chunk flags. `inc`
// is -raw, MEEP's unowned cells at (+0, +0), and every stage SUBTRACTS it:
//   outside the dsigu chunks: f -= sinv * inc
//   inside them:              fu -= (df = sinv * inc); f -= sinv_u * df
__device__ __forceinline__ void cfold_apply(
    float* f, float* fu, int idx, cf inc,
    float sinv, float sinv_u, int dsigu
) {
    cf df = mul_coefficient_left(sinv, inc);
    if (dsigu) cf_store(fu, idx, cf_sub(cf_load(fu, idx), df));
    cf fcur = cf_load(f, idx);
    cf value = dsigu ? cf_sub(fcur, mul_coefficient_left(sinv_u, df)) : cf_sub(fcur, df);
    cf_store(f, idx, value);
}
'''

#: The beta pass reaches the x and y components only (MEEP's ``cc`` runs over d_c in
#: {X, Y}, step_db.cpp:161-175): target 0 takes the SECOND source's stored word at the
#: cell, target 1 the FIRST source's (stepping.B_CURL_TERMS / D_CURL_TERMS against the
#: call sites: Bx <- Ey, By <- Ex, Dx <- Hy, Dy <- Hx).
BETA_TARGETS: Tuple[int, ...] = (0, 1)


# =============================================================================
# The fused products' value-returning, ownership-guarded twins
# =============================================================================

def _function_text(prelude: str, signature: str, what: str) -> str:
    if prelude.count(signature) != 1:
        raise AssertionError(
            f"the prelude declares {what} {prelude.count(signature)} times, not once; the "
            f"fused twin has nothing to be derived from")
    text = prelude[prelude.index(signature):]
    end = text.find("\n}\n")
    if end < 0:
        raise AssertionError(f"{what} has no closing brace in the prelude")
    return text[: end + len("\n}\n")]


def _derive(helper: str, edits: Sequence[Tuple[str, str, str]], what: str) -> str:
    derived = helper
    for old, new, anchor in edits:
        if derived.count(old) != 1:
            raise AssertionError(
                f"{what} no longer carries {anchor} ({old!r}) exactly once (found "
                f"{derived.count(old)}); the fused twin cannot be derived from it")
        derived = derived.replace(old, new, 1)
    return derived


_FOLD_SIGNATURE = "__device__ __forceinline__ void fold_apply(\n"
_FOLD_PARAMETERS = "    float sinv, float sinv_u, int dsigu\n)"
_FOLD_FCUR = "    float fcur = f[idx];\n"
_FOLD_STORE = "    f[idx] = value;\n"

_CFOLD_SIGNATURE = "__device__ __forceinline__ void cfold_apply(\n"
_CFOLD_FCUR = "    cf fcur = cf_load(f, idx);\n"
_CFOLD_STORE = "    cf_store(f, idx, value);\n"

_GUARD_COMMENT = (
    "    // THE OWNERSHIP GUARD, the fused pairs' own (pml_apply_reg): on a cell a fill\n"
    "    // images the SOURCE thread writes f, so this thread neither reads nor writes\n"
    "    // it. The fu stage above stays unguarded: the curl writes fu at every\n"
    "    // recurrence cell and no fill touches it.\n")


def _reg_edits(signature: str, reg_signature: str, fcur: str, store: str,
               guard_return: str, owned: bool) -> List[Tuple[str, str, str]]:
    edits = [(signature, reg_signature, "its signature"),
             (store, store + "    return value;\n", "its closing store")]
    if owned:
        edits[1:1] = [
            (_FOLD_PARAMETERS, "    float sinv, float sinv_u, int dsigu, int owned\n)",
             "its parameter list"),
            (fcur, _GUARD_COMMENT + guard_return + fcur, "its one load of the field")]
    return edits


def fold_apply_reg_source(helpers: str = REAL_FOLD_HELPERS, owned: bool = True) -> str:
    """``fold_apply`` returning the field's word, behind the ownership guard.

    Anchored rewrites of the helper's own text and no other: the signature returns
    ``float`` and is renamed ``fold_apply_reg``, and the closing store also returns the
    value; with ``owned`` (a pair that carries fills) the parameter list gains ``owned``
    and the guard sits directly above the one load of the field. Raises by name if any
    anchor moved.
    """
    helper = _function_text(helpers, _FOLD_SIGNATURE, "fold_apply")
    return _derive(helper, _reg_edits(
        _FOLD_SIGNATURE, "__device__ __forceinline__ float fold_apply_reg(\n",
        _FOLD_FCUR, _FOLD_STORE, "    if (!owned) return 0.0f;\n", owned), "fold_apply")


def cfold_apply_reg_source(helpers: str = COMPLEX_FOLD_HELPERS, owned: bool = True) -> str:
    """``cfold_apply`` returning the field's word pair, behind the ownership guard."""
    helper = _function_text(helpers, _CFOLD_SIGNATURE, "cfold_apply")
    return _derive(helper, _reg_edits(
        _CFOLD_SIGNATURE, "__device__ __forceinline__ cf cfold_apply_reg(\n",
        _CFOLD_FCUR, _CFOLD_STORE, "    if (!owned) return cf_zero();\n", owned),
        "cfold_apply")


def rewrite_helper(prelude: str, signature: str, edits: Sequence[Tuple[str, str, str]],
                   what: str) -> str:
    """``prelude`` with the one helper that opens on ``signature`` rewritten in place.

    Each edit is ``(old, new, anchor name)`` and must match exactly once WITHIN that
    helper's text, so an anchor another helper of the same prelude also carries (the
    closing ``f[idx] = value;`` of ``pml_apply`` and ``fold_apply``) is never touched.
    """
    helper = _function_text(prelude, signature, what)
    if prelude.count(helper) != 1:
        raise AssertionError(f"the prelude carries {what}'s text {prelude.count(helper)} "
                             f"times, not once; it cannot be rewritten in place")
    return prelude.replace(helper, _derive(helper, edits, what), 1)


def with_fold_reg(prelude: str, complex_storage: bool = False, owned: bool = True) -> str:
    """``prelude`` with the folded ladder replaced IN PLACE by its value-returning twin.

    The twin is derived from the helper's own text (:func:`fold_apply_reg_source`), and
    the plain helper leaves the prelude: no call to it is left in a fused body, and a dead
    copy of a helper is where a first-site mutation needle lands vacuously.
    """
    if complex_storage:
        signature, what = _CFOLD_SIGNATURE, "cfold_apply"
        twin = cfold_apply_reg_source(_function_text(prelude, signature, what), owned)
    else:
        signature, what = _FOLD_SIGNATURE, "fold_apply"
        twin = fold_apply_reg_source(_function_text(prelude, signature, what), owned)
    helper = _function_text(prelude, signature, what)
    if prelude.count(helper) != 1:
        raise AssertionError(f"the prelude carries {what}'s text {prelude.count(helper)} "
                             f"times, not once")
    return prelude.replace(helper, twin, 1)


# =============================================================================
# Body transforms: what a family inserts after the certified pml_apply call
# =============================================================================

_MASK_LINE = re.compile(r"^(\s*)if \((.+)\) curl = (?:0\.0f|cf_zero\(\));$")


def curl_call(body: str, prefix: str, what: str) -> Tuple[int, List[str]]:
    """``(line index, argument list)`` of the one line of ``body`` starting ``prefix``.

    ``prefix`` is the call's leading text (``"        pml_apply(Bx, fu_Bx, idx, "``).
    Refuses by name unless exactly one line matches and it closes on that line, so a
    family never inserts after an arbitrary statement.
    """
    lines = body.split("\n")
    hits = [n for n, line in enumerate(lines) if line.startswith(prefix)]
    if len(hits) != 1:
        raise AssertionError(
            f"{what}: {len(hits)} lines start {prefix!r}, not 1; the folded pass has no "
            f"anchor")
    line = lines[hits[0]].strip()
    match = re.fullmatch(r"(?:\w+ = )?\w+\((.*)\);", line)
    if match is None:
        raise AssertionError(f"{what}: {line!r} does not close on its own line")
    return hits[0], [part.strip() for part in match.group(1).split(",")]


def ownership_lines(body: str, prefix: str, what: str, flag: str = "owned_cell",
                    indent: str = "        ") -> List[str]:
    """The curl's own mask lines of one component, re-spelled as an ownership flag.

    The lines immediately above the component's ``pml_apply`` call that zero its curl
    (``if (...) curl = 0.0f;``) are MEEP's not-owned test (``stepping._mask_non_owned_cells``,
    the same function masks the folded increment). Each becomes ``if (...) flag = 0;``
    under ``int flag = 1;`` -- read off the certified text, never retyped.
    """
    index, _arguments = curl_call(body, prefix, what)
    lines = body.split("\n")
    tests: List[str] = []
    position = index - 1
    while position >= 0:
        match = _MASK_LINE.match(lines[position])
        if match is None:
            break
        tests.insert(0, match.group(2))
        position -= 1
    if not tests:
        raise AssertionError(
            f"{what}: no curl mask line sits directly above {prefix.strip()!r}; the "
            f"ownership flag would admit every cell")
    return ([f"{indent}int {flag} = 1;"]
            + [f"{indent}if ({test}) {flag} = 0;" for test in tests])


def insert_after_call(body: str, prefix: str, lines: Sequence[str], what: str) -> str:
    """``body`` with ``lines`` inserted directly after the one line starting ``prefix``."""
    index, _arguments = curl_call(body, prefix, what)
    out = body.split("\n")
    out[index + 1:index + 1] = list(lines)
    return "\n".join(out)


def _stencil_partners(body: str, prefix: str, what: str) -> Tuple[str, str]:
    """The two presence arguments of the component's ``curl_stencil`` call."""
    index, _arguments = curl_call(body, prefix, what)
    lines = body.split("\n")
    for position in range(index - 1, -1, -1):
        match = re.search(r"curl_stencil\((.*)\);$", lines[position])
        if match:
            arguments = [part.strip() for part in match.group(1).split(",")]
            if len(arguments) != 7:
                raise AssertionError(f"{what}: curl_stencil takes {len(arguments)} arguments")
            return arguments[5], arguments[6]
        if lines[position].strip() == "{":
            break
    raise AssertionError(f"{what}: no curl_stencil call above {prefix.strip()!r}")


def _dsig_flag(dsigu_flag: str, sinv: str) -> str:
    """The dsig axis's membership flag, on the dsigu flag's lattice: ``sinv_y[j]`` with
    ``in_h_z`` gives ``in_h_y`` (a curl component's dsig and dsigu axes lie on one
    sub-lattice: the B curl's both half-integer, the D curl's both integer)."""
    match = re.fullmatch(r"sinv_([xyz])\[[ijk]\]", sinv)
    lattice = re.fullmatch(r"in_([hi])_[xyz]", dsigu_flag)
    if match is None or lattice is None:
        raise AssertionError(f"cannot read the dsig flag off {sinv!r} and {dsigu_flag!r}")
    return f"in_{lattice.group(1)}_{match.group(1)}"


def bfast_insert(body: str, prefix: str, target: str, state: str, k1: str, k2: str,
                 what: str, fold: str = "fold_apply", plain_extra: str = "") -> List[str]:
    """The BFAST pass of one component, inserted after its ``pml_apply`` call.

    ``prefix`` is the call's leading text; ``state`` the component's ``f_bfast`` pointer;
    ``k1``/``k2`` its host-gated scalars. The operands are the curl's own four registers
    (``sf``, ``f1``, ``ss``, ``f2``) and the presence bits are the curl stencil's own.
    ``plain_extra`` conjoins a family's further plain-branch clause (the conductive weld's
    "and no conductivity").
    """
    _index, arguments = curl_call(body, prefix, what)
    if len(arguments) != 9:
        raise AssertionError(f"{what}: the curl call takes {len(arguments)} arguments, not 9")
    field, aux, sinv, sinv_u, dsigu = (arguments[0], arguments[1], arguments[5],
                                       arguments[7], arguments[8])
    pr1, pr2 = _stencil_partners(body, prefix, what)
    plain = f"!{_dsig_flag(dsigu, sinv)} && !{dsigu}" + plain_extra
    return (["        // STEP_BFAST (step_db.cpp:124-142; stepping._bfast_term): MEEP's own pass",
             "        // on this component after its curl, over the curl's four operands,",
             "        // through its own ladder (stepping._apply_folded_increment)."]
            + ownership_lines(body, prefix, what)
            + [f"        float inc_bfast = bfast_pass({state}, idx, sf, f1, ss, f2, {k1}, "
               f"{k2}, {pr1}, {pr2}, {plain}, owned_cell);",
               f"        {fold}({field}, {aux}, idx, inc_bfast, {sinv}, {sinv_u}, {dsigu});"])


def beta_insert(body: str, prefix: str, partner: str, coefficient: str, what: str,
                fold: str = "fold_apply") -> List[str]:
    """The real special_kz beta pass of one component, after its ``pml_apply`` call.

    ``partner`` is the register holding the partner's STORED word at the cell (the curl's
    unshifted centre load), ``coefficient`` the host-rounded scalar of this call site.
    """
    _index, arguments = curl_call(body, prefix, what)
    if len(arguments) != 9:
        raise AssertionError(f"{what}: the curl call takes {len(arguments)} arguments, not 9")
    field, aux, sinv, sinv_u, dsigu = (arguments[0], arguments[1], arguments[5],
                                       arguments[7], arguments[8])
    return (["        // The special_kz beta pass (step_db.cpp:161-175;",
             "        // stepping._special_kz_beta_term): -(c * g), +0 where MEEP's loop does",
             "        // not visit, through its own ladder after this component's curl."]
            + ownership_lines(body, prefix, what)
            + [f"        float inc_beta = owned_cell ? (-({coefficient} * {partner})) : 0.0f;",
               f"        {fold}({field}, {aux}, idx, inc_beta, {sinv}, {sinv_u}, {dsigu});"])


def complex_beta_insert(body: str, prefix: str, partner: str, coefficient: str,
                        what: str, fold: str = "cfold_apply") -> List[str]:
    """The complex special_kz beta pass of one component, after its ``pml_apply`` call.

    ``partner`` is the C expression of the partner's STORED word pair at the cell
    (``cf_load(g1, idx)``): the curl's centre registers are read through the Bloch
    connection, and the beta pass reads the component snapshot.
    """
    _index, arguments = curl_call(body, prefix, what)
    if len(arguments) != 9:
        raise AssertionError(f"{what}: the curl call takes {len(arguments)} arguments, not 9")
    field, aux, sinv, sinv_u, dsigu = (arguments[0], arguments[1], arguments[5],
                                       arguments[7], arguments[8])
    return (["        // The special_kz beta pass (step_db.cpp:161-175;",
             "        // stepping._special_kz_beta_term): -quarter_turn(c, g) part by part,",
             "        // (+0, +0) where MEEP's loop does not visit, through its own ladder",
             "        // after this component's curl."]
            + ownership_lines(body, prefix, what)
            + [f"        cf inc_beta = owned_cell ? beta_increment({coefficient}, {partner}) "
               f": cf_zero();",
               f"        {fold}({field}, {aux}, idx, inc_beta, {sinv}, {sinv_u}, {dsigu});"])


def capture_fold_calls(body: str, calls: Dict[str, Tuple[str, Optional[str]]], fold: str,
                       fold_reg: str, what: str) -> str:
    """Each family fold call on a captured target re-spelled to update its register.

    ``calls`` maps the call's leading arguments (``"Bx, fu_Bx, "``) to ``(register,
    ownership flag)``: ``fold_apply(Bx, fu_Bx, idx, ...);`` becomes ``b_x =
    fold_apply_reg(Bx, fu_Bx, idx, ..., own_x);`` (no flag appended where the flag is
    None: a pair that carries no fill owns every cell). A listed call that is absent is
    skipped (the beta pass reaches two of three targets); one that occurs twice is
    refused, and no uncaptured fold call may survive.
    """
    for head, (register, flag) in calls.items():
        old = f"{fold}({head}idx, "
        lines = [line for line in body.split("\n") if old in line]
        if not lines:
            continue
        if len(lines) != 1:
            raise AssertionError(
                f"{what}: {len(lines)} lines call {old.strip()!r}, not 1; the register "
                f"capture has no anchor")
        line = lines[0]
        if not line.rstrip().endswith(");"):
            raise AssertionError(f"{what}: {line.strip()!r} does not close on its line")
        indent = line[: len(line) - len(line.lstrip())]
        call = line.strip()[:-2].replace(f"{fold}(", f"{fold_reg}(", 1)
        tail = f", {flag});" if flag is not None else ");"
        body = body.replace(line, f"{indent}{register} = {call}{tail}", 1)
    if re.search(rf"(?<!\w){re.escape(fold)}\(", body):
        raise AssertionError(
            f"{what}: a {fold} call survived the capture; its value would be stored and "
            f"never reach the carried register")
    return body
