"""Source mutations of the Triton fold passes, shared by the fold family's gates and probes.

The special-kz beta pass and the BFAST pass are MEEP's own passes after the curl
(step_db.cpp:124-175): each component's curl form, then its BFAST pass, then -- after
every component -- the beta pass, each adding its increment through the additive ladder
(``triton_kernels/forms.py``: ``fold_ladder``, ``bfast_advance``). Every kernel that
carries a pass splices ONE ``if HAS_BETA:`` / ``if HAS_BFAST:`` block after its curl forms,
character for character the same block in the curl and in every lifter, so the rewrites
that plant a defect in that block are the same text everywhere. This module is their one
spelling; each gate or probe picks the entries its kernel carries, scores each on a case
that enters the branch it rewrites, and declares the expectation here.

Each entry is ``(id, why, expectation, rewrite, needs)``:

* ``rewrite(source) -> (mutated source, hits)``; ``hits == 0`` means NOT ARMED;
* ``expectation`` is ``"caught"`` (some stored word must differ from the oracle) or
  ``"null"`` (a DECLARED null: no stored word can differ, with the reason in ``why``);
* ``needs`` names what the scoring case must carry for the rewritten branch to run:
  ``"wall"`` (a metallic axis, so the ownership mask fires), ``"layer"`` (a PML layer on
  the dsigu axis of target 1, which is x for both curl families), ``"one_partner"`` (a
  component one of whose partners differentiates along an invariant axis: a 2-D grid),
  ``"invariant_k"`` (BFAST with a nonzero k on a declared-invariant axis),
  ``"mirror_periodic"`` (a folded PERIODIC axis), or nothing.

THE INVERSIONS (results/bit_parity_0_10/design/KERNELS.md section 7.1). The previous
kernels carried the folded term INSIDE the curl, before the forms (the array path's
previous placement); the beta pass and the BFAST pass now follow the forms, as MEEP runs
them. ``*_inside_the_curl`` plants the previous placement back, and must be caught. The
BFAST state's previous one-rounding advance ``total - 2*state`` (AP6b) is planted back by
``f_one_rounding_advance``, and the previous whole-pass on a one-partner component (the
absent partner's product kept with a zeroed coefficient) by ``f_absent_partner_read``.

Host-only text; nothing here imports the package. The shared pieces a re-cut harness
needs beside the rewrites live here too: :func:`body_and_insert` (the transcription
legs' AST split), :data:`MUTANT_IMPORTS` (a mutant's prelude), :func:`table` and
:func:`check_needs`.
"""

from __future__ import annotations

import ast
import re as _re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

Rewrite = Callable[[str], Tuple[str, int]]
Entry = Tuple[str, str, str, Rewrite, Tuple[str, ...]]

#: The prelude a compiled mutant of a fold kernel needs: the shared ``@triton.jit`` forms
#: the shipped bodies call by their imported names, the record's two word offsets, and the
#: boundary codes. A mutant differs from the shipped text only by its rewrite.
MUTANT_IMPORTS = (
    "from meep_gpu.triton_kernels.forms import (axis_present, bfast_advance, "
    "constitutive_form, curl_form, curl_stencil, fold_ladder, record_in)\n"
    "from meep_gpu.triton_kernels.kernels import RECORD_HALF_INTEGER, "
    "RECORD_PRESENCE\n")

#: Every ``needs`` word an entry may carry (the module docstring says what each means).
NEEDS = ("wall", "layer", "one_partner", "invariant_k", "mirror_periodic")


def body_and_insert(path: str, name: str, flag: str) -> Tuple[List[str], List[str]]:
    """``(body lines, insert lines)`` of one shipped kernel, read from the FILE.

    The insert is the kernel's top-level ``if <flag>:`` statement (the fold pass every
    carrier splices in after its curl forms); the body is every other top-level statement
    after the docstring. Each is the exact source text split into stripped, comment-free
    code lines -- never ``ast.unparse``d, so the parenthesisation is what is compared. A
    transcription leg checks that the body IS the root kernel's body and that the insert
    IS the curl's own insert.
    """
    text = open(path, encoding="utf-8").read()
    node = next((found for found in ast.walk(ast.parse(text))
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    statements = list(node.body)
    if (isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements = statements[1:]

    def lines(nodes: Iterable[ast.stmt]) -> List[str]:
        out: List[str] = []
        for statement in nodes:
            for raw in (ast.get_source_segment(text, statement) or "").splitlines():
                code = raw.split("#", 1)[0].strip()
                if code:
                    out.append(code)
        return out

    insert = [s for s in statements
              if isinstance(s, ast.If) and getattr(s.test, "id", "") == flag]
    body = [s for s in statements if s not in insert]
    return lines(body), lines(insert)


def insert_neighbours(path: str, name: str, flag: str) -> Tuple[str, str]:
    """The first code line of the top-level statements just before and after the insert."""
    text = open(path, encoding="utf-8").read()
    node = next(found for found in ast.walk(ast.parse(text))
                if isinstance(found, ast.FunctionDef) and found.name == name)
    statements = list(node.body)
    position = next(index for index, s in enumerate(statements)
                    if isinstance(s, ast.If) and getattr(s.test, "id", "") == flag)

    def first_line(statement: ast.stmt) -> str:
        segment = ast.get_source_segment(text, statement) or ""
        return segment.splitlines()[0].split("#", 1)[0].strip()

    return first_line(statements[position - 1]), first_line(statements[position + 1])


def table(entries: Sequence[Entry]) -> Tuple[Tuple[Tuple[str, str, str, Rewrite], ...],
                                             Dict[str, Tuple[str, ...]]]:
    """The 4-tuple table the gates and probes iterate, and each entry's ``needs``."""
    return (tuple((name, why, expectation, rewrite)
                  for name, why, expectation, rewrite, _needs in entries),
            {name: tuple(needs) for name, _why, _expectation, _rewrite, needs in entries})


def declare(entries: Sequence[Entry],
            expectations: Dict[str, Tuple[str, str]]) -> List[Entry]:
    """``entries`` with the named ones' expectation and reason replaced.

    A carrier declares here what the shared table cannot know: a mutation that is a
    STRUCTURAL null on this carrier (a later pass of the same launch overwrites every
    word it changes, say) is declared ``"null"`` with the reason. Naming an entry the
    table does not carry raises, so a declaration cannot outlive its mutation.
    """
    names = {entry[0] for entry in entries}
    unknown = sorted(set(expectations) - names)
    if unknown:
        raise AssertionError(f"declared expectations for unknown mutations {unknown}")
    out: List[Entry] = []
    for name, why, expectation, rewrite, needs in entries:
        if name in expectations:
            expectation, why = expectations[name]
        out.append((name, why, expectation, rewrite, needs))
    return out


def cases_for(needs: Dict[str, Tuple[str, ...]], carries: Dict[Any, Tuple[str, ...]],
              default: Any) -> Dict[str, Any]:
    """Per entry, the case it is scored on: ``default`` if that case carries every need,
    else the first case (in ``carries`` order) that does. Raises when none does, so a
    shared entry never lands on a grid that cannot enter the branch it rewrites."""
    out: Dict[str, Any] = {}
    for name, wanted in needs.items():
        if set(wanted) <= set(carries[default]):
            continue
        chosen = next((key for key, have in carries.items() if set(wanted) <= set(have)),
                      None)
        if chosen is None:
            raise AssertionError(f"mutation {name} needs {sorted(wanted)} and no case "
                                 f"carries them all")
        out[name] = chosen
    return out


def check_needs(name: str, needs: Sequence[str], carried: Iterable[str],
                case_name: str) -> None:
    """Raise unless the scoring case carries everything the rewritten branch needs.

    A mutation scored on a case that never enters the branch it rewrites reports
    UNCAUGHT while measuring nothing; so does one whose ``needs`` names a word this
    module does not define.
    """
    have = set(carried)
    for need in needs:
        if need not in NEEDS:
            raise AssertionError(f"mutation {name} declares an unknown need {need!r}")
        if need not in have:
            raise AssertionError(
                f"mutation {name} needs {need!r}, and case {case_name!r} carries only "
                f"{sorted(have)}: the rewritten lines would be a DEAD BRANCH and the leg "
                f"would report 'uncaught' while measuring nothing")


def replace_all(source: str, before: str, after: str) -> Tuple[str, int]:
    """Every occurrence of ``before``; the hit count is the number replaced."""
    hits = source.count(before)
    return source.replace(before, after), hits


def rewrite_block(source: str, before: Sequence[str],
                  after: Sequence[str]) -> Tuple[str, int]:
    """Replace a multi-line fragment, matching on the significant statements only.

    Blank lines and comments are skipped when matching and dropped from the replaced
    span; the replacement keeps its own relative indentation, re-prefixed with the
    matched span's base (the shared probes' ``_rewrite_block``).
    """
    lines = source.splitlines()
    significant = [index for index, line in enumerate(lines)
                   if line.strip() and not line.strip().startswith("#")]
    needle = [fragment.strip() for fragment in before]
    hits = 0
    cursor = 0
    while cursor <= len(significant) - len(needle):
        span = significant[cursor:cursor + len(needle)]
        if [lines[index].strip() for index in span] == needle:
            first, last = span[0], span[-1]
            indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
            lines[first:last + 1] = [(indent + fragment) if fragment.strip() else fragment
                                     for fragment in after]
            hits += 1
            significant = [index for index, line in enumerate(lines)
                           if line.strip() and not line.strip().startswith("#")]
            cursor = significant.index(first) + len(after)
        else:
            cursor += 1
    return "\n".join(lines), hits


def _chain(*rewrites: Rewrite) -> Rewrite:
    """Apply rewrites in order; armed only if EVERY one hit."""
    def apply(source: str) -> Tuple[str, int]:
        hits = []
        for rewrite in rewrites:
            source, count = rewrite(source)
            hits.append(count)
        return source, (min(hits) if hits else 0)
    return apply


def _lines_matching(prefix: str, replacement: Callable[[str], str]) -> Rewrite:
    """Rewrite every line whose stripped text starts with ``prefix``."""
    def apply(source: str) -> Tuple[str, int]:
        out, hits = [], 0
        for line in source.splitlines():
            if line.strip().startswith(prefix):
                indent = line[:len(line) - len(line.lstrip())]
                out.append(indent + replacement(line.strip()))
                hits += 1
            else:
                out.append(line)
        return "\n".join(out), hits
    return apply


# ---------------------------------------------------------------------------
# The special-kz beta pass
# ---------------------------------------------------------------------------

def beta_mutations(complex_storage: bool, folded: bool,
                   curl_anchor: str) -> List[Entry]:
    """The beta pass's mutations for one kernel family.

    ``curl_anchor`` is the kernel's last curl-stencil line (target 2's), after which
    ``*_inside_the_curl`` plants the previous placement of the term.
    """
    if complex_storage:
        q0 = ("q0_re = beta_plus * b_im", "q0_im = (beta_plus * b_re) * -1.0")
        q1 = ("q1_re = beta_minus * a_im", "q1_im = (beta_minus * a_re) * -1.0")
        ladder1 = ("v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)",
                   "v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)")
        # The previous placement adds the increment to the curl the forms subtract:
        # curl + (c*y, -(c*x)) per part, the real term's curl - (c*g) on each part.
        inside = ["if HAS_BETA:",
                  "    curl0_re = curl0_re + (beta_plus * b_im)",
                  "    curl0_im = curl0_im - (beta_plus * b_re)",
                  "    curl1_re = curl1_re + (beta_minus * a_im)",
                  "    curl1_im = curl1_im - (beta_minus * a_re)"]
        sign_old, sign_new = "q0_re = beta_plus * b_im", "q0_re = (beta_plus * b_im) * -1.0"
        dtdx_old = "q1_re = beta_minus * a_im"
        dtdx_new = "q1_re = dtdx * (beta_minus * a_im)"
        swap = (["q0_re = beta_plus * b_im", "q0_im = (beta_plus * b_re) * -1.0",
                 "q1_re = beta_minus * a_im", "q1_im = (beta_minus * a_re) * -1.0"],
                ["q0_re = beta_plus * a_im", "q0_im = (beta_plus * a_re) * -1.0",
                 "q1_re = beta_minus * b_im", "q1_im = (beta_minus * b_re) * -1.0"])
        mask_prefix = ("q0_re = tl.where(", "q0_im = tl.where(", "q1_re = tl.where(",
                       "q1_im = tl.where(")
        aux = ("v1_re, n1_re = fold_ladder(", "v1_re, _n1_re = fold_ladder(")
        quarter = (["q0_re = beta_plus * b_im", "q0_im = (beta_plus * b_re) * -1.0"],
                   ["q0_re = (beta_plus * b_im) * -1.0", "q0_im = beta_plus * b_re"])
    else:
        q0 = ("q0 = (beta_plus * b) * -1.0",)
        q1 = ("q1 = (beta_minus * a) * -1.0",)
        ladder1 = ("v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)",)
        inside = ["if HAS_BETA:",
                  "    curl0 = curl0 - (beta_plus * b)",
                  "    curl1 = curl1 - (beta_minus * a)"]
        sign_old, sign_new = "q0 = (beta_plus * b) * -1.0", "q0 = (beta_plus * b) * 1.0"
        dtdx_old = "q1 = (beta_minus * a) * -1.0"
        dtdx_new = "q1 = (dtdx * (beta_minus * a)) * -1.0"
        swap = (["q0 = (beta_plus * b) * -1.0", "q1 = (beta_minus * a) * -1.0"],
                ["q0 = (beta_plus * a) * -1.0", "q1 = (beta_minus * b) * -1.0"])
        mask_prefix = ("q0 = tl.where(", "q1 = tl.where(")
        aux = ("v1, n1 = fold_ladder(", "v1, _n1 = fold_ladder(")
        quarter = None
    del q0, q1

    def unmask(source: str) -> Tuple[str, int]:
        total = 0
        for prefix in mask_prefix:
            name = prefix.split(" = ")[0]
            source, hits = _lines_matching(
                prefix, lambda line, name=name: f"{name} = {name}")(source)
            total += hits
        return source, total

    def ladder_without_dsigu(source: str) -> Tuple[str, int]:
        total = 0
        for line in ladder1:
            source, hits = replace_all(source, line,
                                       line.replace(", du1)", ", live & (idx < 0))"))
            total += hits
        return source, total

    def ladder_axes_swapped(source: str) -> Tuple[str, int]:
        total = 0
        for line in ladder1:
            source, hits = replace_all(source, line,
                                       line.replace("si_z, si_x, du1", "si_x, si_z, du1"))
            total += hits
        return source, total

    entries: List[Entry] = [
        ("b_pass_dropped", "no beta pass at all: the ordinary pair on a beta run",
         "caught", lambda s: replace_all(s, "if HAS_BETA:", "if HAS_BETA and False:"), ()),
        ("b_partners_swapped",
         "target 0 takes the first source's centre and target 1 the second's",
         "caught", lambda s: rewrite_block(s, swap[0], swap[1]), ()),
        ("b_sign_flipped", "the increment not negated: MEEP's f - raw instead of f + raw",
         "caught", lambda s: replace_all(s, sign_old, sign_new), ()),
        ("b_scaled_by_dtdx", "the analytic derivative multiplied by dtdx",
         "caught", lambda s: replace_all(s, dtdx_old, dtdx_new), ()),
        ("b_pass_inside_the_curl",
         "THE INVERSION: the term added to the curl before the forms, the previous "
         "placement; MEEP's pass follows the forms through its own ladder",
         "caught", _chain(
             lambda s: replace_all(s, "if HAS_BETA:", "if HAS_BETA and False:"),
             lambda s: rewrite_block(s, [curl_anchor], [curl_anchor] + inside)),
         ("layer",)),
        ("b_increment_unmasked",
         "the increment not masked on the cells the curl's mask drops",
         "caught", unmask, ("wall", "layer")),
        ("b_ladder_without_dsigu",
         "target 1's ladder takes the outside form on its dsigu chunk too",
         "caught", ladder_without_dsigu, ("layer",)),
        ("b_ladder_axes_swapped",
         "target 1's ladder scales by the dsigu axis's sinv first and the dsig axis's "
         "second", "caught", ladder_axes_swapped, ("layer",)),
        ("b_auxiliary_not_advanced",
         "the pass leaves fu at its post-curl word on the dsigu chunk",
         "caught", lambda s: replace_all(s, aux[0], aux[1]), ("layer",)),
    ]
    if quarter is not None:
        entries.append(
            ("b_quarter_turn_reversed",
             "the quarter turn by -i where it is by +i: the parts exchanged with the "
             "negation on the other one", "caught",
             lambda s: rewrite_block(s, quarter[0], quarter[1]), ()))
    if folded:
        # Both targets' top-plane lines: which target a fold masks there depends on the
        # side (B masks target 0 on a folded PERIODIC y, D target 1), so one spelling
        # serves both carriers of a side and the case's single fold arms it.
        names = ("q0_re", "q1_re") if complex_storage else ("q0", "q1")

        def top_plane_unmasked(source: str) -> Tuple[str, int]:
            total = 0
            for name in names:
                source, hits = _lines_matching(
                    f"{name} = tl.where(last_", lambda line, name=name: f"{name} = {name}"
                )(source)
                total += hits
            return source, total

        entries.append(
            ("b_top_plane_mask_dropped",
             "the increment not masked on the top plane of a folded PERIODIC axis",
             "caught", top_plane_unmasked, ("mirror_periodic",)))
    return entries


# ---------------------------------------------------------------------------
# The BFAST pass
# ---------------------------------------------------------------------------

def bfast_mutations() -> List[Entry]:
    """The BFAST pass's mutations (real storage; the curl and both fused pairs)."""
    advance2 = "bf2, q2 = bfast_advance(st2, b_x, b, a_y, a, k1_2, k2_2, px, py, ds2 | du2)"
    advance0 = "bf0, q0 = bfast_advance(st0, c_y, c, b_z, b, k1_0, k2_0, py, pz, ds0 | du0)"
    ladder1 = "v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)"

    def unmask_increment(source: str) -> Tuple[str, int]:
        total = 0
        for name in ("q0", "q1", "q2"):
            source, hits = _lines_matching(f"{name} = tl.where(",
                                           lambda line, name=name: f"{name} = {name}")(source)
            total += hits
        return source, total

    def unmask_state(source: str) -> Tuple[str, int]:
        total = 0
        for name in ("bf0", "bf1", "bf2"):
            source, hits = _lines_matching(f"{name} = tl.where(",
                                           lambda line, name=name: f"{name} = {name}")(source)
            total += hits
        return source, total

    return [
        ("f_pass_dropped", "no BFAST pass at all: the ordinary kernel on a BFAST run",
         "caught", lambda s: replace_all(s, "if HAS_BFAST:", "if HAS_BFAST and False:"), ()),
        ("f_one_rounding_advance",
         "THE INVERSION (AP6b): target 2's increment from the previous one-rounding "
         "advance total - 2*state instead of MEEP's F = S - F_prev, raw = F - F_prev",
         "caught", lambda s: rewrite_block(
             s, [advance2],
             [advance2,
              "q2 = (((k1_2 * (b_x + b)) - (k2_2 * (a_y + a))) - (2.0 * st2)) * -1.0"]),
         ()),
        ("f_plain_branch_keeps_the_state",
         "the plain one-partner branch spelled single - F as on a layered chunk; MEEP's "
         "plain line drops the - F", "caught",
         lambda s: replace_all(s, advance0, advance0.replace("ds0 | du0", "live")),
         ("one_partner",)),
        ("f_absent_partner_read",
         "THE INVERSION: target 0 forms both partners' products whatever the presence, "
         "the absent one's with its zeroed coefficient; MEEP does not read it",
         "caught", lambda s: replace_all(s, advance0,
                                         advance0.replace("py, pz,", "live, live,")),
         ("invariant_k",)),
        ("f_k_pair_swapped", "target 2's k1 and k2 exchanged", "caught",
         lambda s: replace_all(s, advance2, advance2.replace("k1_2, k2_2", "k2_2, k1_2")),
         ()),
        ("f_increment_unmasked",
         "the increment not masked on the cells the curl's mask drops", "caught",
         unmask_increment, ("wall", "layer")),
        ("f_state_stored_on_unowned_cells",
         "the new state stored on the cells MEEP's owned loop does not visit", "caught",
         unmask_state, ("wall",)),
        ("f_state_store_dropped", "target 1's state never written back", "caught",
         lambda s: replace_all(s, "tl.store(s1 + idx, bf1, mask=live)",
                               "tl.store(s1 + idx, bf1, mask=live & (idx < 0))"), ()),
        ("f_ladder_without_dsigu",
         "target 1's ladder takes the outside form on its dsigu chunk too", "caught",
         lambda s: replace_all(s, ladder1, ladder1.replace(", du1)", ", live & (idx < 0))")),
         ("layer",)),
        ("f_ladder_axes_swapped",
         "target 1's ladder scales by the dsigu axis's sinv first", "caught",
         lambda s: replace_all(s, ladder1, ladder1.replace("si_z, si_x", "si_x, si_z")),
         ("layer",)),
        ("f_auxiliary_not_advanced",
         "the pass leaves fu at its post-curl word on the dsigu chunk", "caught",
         lambda s: replace_all(s, "v1, n1 = fold_ladder(", "v1, _n1 = fold_ladder("),
         ("layer",)),
    ]


# ---------------------------------------------------------------------------
# The forms every carrier copies (the curl half and the constitutive half)
# ---------------------------------------------------------------------------
#
# Every fold kernel is a root body (``kernels.pml_curl_step``, the two root fused
# pairs, ``complex_fields``' curl, the complex and folded pairs) character for
# character, plus its pass. The root bodies' form lines are therefore the same text in
# every carrier, up to the storage's part suffix (``curl2`` / ``curl2_re``) and the
# fold's ownership mask (``live`` / ``own0``); the rewrites below match by pattern, not
# by one spelling, so one table serves every carrier. KERNELS.md section 7.1 (the
# previous grouping planted back: ``a1``, ``a3``) and section 7.2 (the chunk-form
# battery: ``b*``) say what each one plants.


_STENCIL = _re.compile(
    r"^(?P<lhs>curl2(?:_re|_im)?) = curl_stencil\((?P<g1s>\w+), (?P<g1>\w+), "
    r"(?P<g2>\w+), (?P<g2s>\w+), dtdx, (?P<first>\w+), (?P<second>\w+)\)$")
_DSIGU = _re.compile(r"^(?P<lhs>du[012]) = record_in\(cb, lattice \+ \d, \w\)$")


def _regex_lines(pattern: "_re.Pattern[str]", replacement: Callable[["_re.Match[str]"], str]
                 ) -> Rewrite:
    """Rewrite every line whose code (comment stripped) matches ``pattern`` in full."""
    def apply(source: str) -> Tuple[str, int]:
        out, hits = [], 0
        for line in source.splitlines():
            match = pattern.match(line.split("#", 1)[0].strip())
            if match:
                indent = line[:len(line) - len(line.lstrip())]
                out.append(indent + replacement(match))
                hits += 1
            else:
                out.append(line)
        return "\n".join(out), hits
    return apply


def _first_call(source: str, name: str) -> Tuple[int, int]:
    """``(start, end)`` of the first ``name(...)`` call (balanced), or ``(-1, -1)``."""
    start = source.find(name + "(")
    while start > 0 and (source[start - 1].isalnum() or source[start - 1] == "_"):
        start = source.find(name + "(", start + 1)
    if start < 0:
        return -1, -1
    depth = 0
    for index in range(start + len(name), len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    return -1, -1


def _previous_constitutive_grouping(source: str) -> Tuple[str, int]:
    """The first ``constitutive_form(f, prev, src, kps, kms, dw)`` call, re-spelled as
    the previous package grouping ``(f + kps*src) - kms*prev`` behind the same select."""
    start, end = _first_call(source, "constitutive_form")
    if start < 0:
        return source, 0
    call = ast.parse(" ".join(source[start:end].split()), mode="eval").body
    if not isinstance(call, ast.Call) or len(call.args) != 6:
        return source, 0
    f, prev, src, kps, kms, dw = (ast.unparse(argument) for argument in call.args)
    return (source[:start]
            + f"tl.where({dw}, ({f} + {kps} * {src}) - {kms} * {prev}, {src})"
            + source[end:]), 1


def _dw_line(index: str) -> "_re.Pattern[str]":
    return _re.compile(rf"^(?P<lhs>dw{index}) = record_in\(cb, (?P<base>[^,]+), "
                       rf"(?P<coordinate>\w)\)$")


def _lattice_swapped(match: "_re.Match[str]") -> str:
    base = match.group("base").replace(" ", "")
    if base.startswith("RECORD_HALF_INTEGER+"):
        swapped = base[len("RECORD_HALF_INTEGER+"):]
    else:
        swapped = "RECORD_HALF_INTEGER + " + base.split("+")[-1]
    return f"{match.group('lhs')} = record_in(cb, {swapped}, {match.group('coordinate')})"


def form_mutations(constitutive: bool = True) -> List[Entry]:
    """The root forms' inversions and chunk-form battery for one carrier.

    ``constitutive`` is False for a curl alone (the beta and BFAST curls, the folded
    beta curls). Every entry needs ``"layer"``: the forms differ only across a chunk
    boundary, so a case with no PML layer scores nothing.
    """
    def stencil_pairwise(match: "_re.Match[str]") -> str:
        return (f"{match.group('lhs')} = dtdx * (({match.group('g1s')} - "
                f"{match.group('g1')}) + ({match.group('g2')} - {match.group('g2s')}))")

    def dsigu_everywhere(match: "_re.Match[str]") -> str:
        return f"{match.group('lhs')} = live"

    def store_unmasked(source: str) -> Tuple[str, int]:
        return _lines_matching_with(
            source, lambda line: line.startswith("tl.store(u2 ") and line.endswith(
                " & du2)"), lambda line: line[:-len(" & du2)")] + ")")

    def load_unmasked(source: str) -> Tuple[str, int]:
        return _lines_matching_with(
            source, lambda line: "= tl.load(u2 " in line and "mask=live & du2," in line,
            lambda line: line.replace("mask=live & du2,", "mask=live,"))

    entries: List[Entry] = [
        ("a1_pairwise_stencil",
         "THE INVERSION (KERNELS.md 7.1): target 2's curl in the previous package "
         "grouping (g1s - g1) + (g2 - g2s) instead of MEEP's ((g1s - g1) + g2) - g2s",
         "caught", _regex_lines(_STENCIL, stencil_pairwise), ()),
        ("b1_whole_grid_recurrence",
         "the curl forms' select removed: the recurrence, and its fu traffic, on every "
         "cell", "caught", _regex_lines(_DSIGU, dsigu_everywhere), ("layer",)),
        ("b2_coefficient_predicate",
         "target 1's form keyed on coefficient identity (kms != 1), not chunk "
         "membership: the two differ on a run's row whose coefficient is exactly 1 "
         "(a sigma = 0 row inside the chunk), which a folded axis's high-face-only "
         "run need not have, so a folded carrier scores it on a grid whose x axis "
         "is unfolded", "caught",
         lambda s: replace_all(s, "du1 = record_in(cb, lattice + 0, i)",
                               "du1 = tl.load(kmx + i, mask=live, other=1.0) != 1.0"),
         ("layer",)),
        ("b3_membership_one_row_off", "target 1's dsigu line read one row off",
         "caught", lambda s: replace_all(s, "du1 = record_in(cb, lattice + 0, i)",
                                         "du1 = record_in(cb, lattice + 0, i + 1)"),
         ("layer",)),
        ("b5_dsig_dsigu_swapped",
         "target 2's forms take the dsigu axis's coefficients where the dsig axis's "
         "belong", "caught",
         lambda s: replace_all(s, "km_x, si_x, km_y, si_y, du2)",
                               "km_y, si_y, km_x, si_x, du2)"), ("layer",)),
        ("b6_fu_store_unmasked",
         "fu written on a cell outside its dsigu layer (the auxiliary contract)",
         "caught", store_unmasked, ("layer",)),
        ("n1_fu_load_unmasked",
         "DECLARED NULL: the auxiliary read on every cell feeds only the discarded form",
         "null", load_unmasked, ("layer",)),
    ]
    if constitutive:
        def fw_store_unmasked(source: str) -> Tuple[str, int]:
            done = {"hit": 0}

            def first(line: str) -> bool:
                if done["hit"] or not line.startswith("tl.store(w0 "):
                    return False
                if " & dw0)" not in line:
                    return False
                done["hit"] = 1
                return True
            return _lines_matching_with(source, first,
                                        lambda line: line.replace(" & dw0)", ")"))

        entries += [
            ("a3_previous_constitutive_grouping",
             "THE INVERSION (KERNELS.md 7.1): component 0's accumulation in the "
             "previous package grouping (f + kps*src) - kms*prev instead of MEEP's "
             "f + ((kps*src) - (kms*prev))", "caught",
             _previous_constitutive_grouping, ("layer",)),
            ("b4_constitutive_membership_one_row_off",
             "component 1's dsigw line read one row off", "caught",
             _regex_lines(_dw_line("1"), lambda m: (
                 f"{m.group('lhs')} = record_in(cb, {m.group('base')}, "
                 f"{m.group('coordinate')} + 1)")), ("layer",)),
            ("b10_constitutive_lattice_swapped",
             "the constitutive half keyed on the other sub-lattice's lines", "caught",
             _chain(_regex_lines(_dw_line("0"), _lattice_swapped),
                    _regex_lines(_dw_line("1"), _lattice_swapped),
                    _regex_lines(_dw_line("2"), _lattice_swapped)), ("layer",)),
            ("b7_fw_store_unmasked",
             "f_w written on a cell outside its dsigw layer (the auxiliary contract)",
             "caught", fw_store_unmasked, ("layer",)),
        ]
    return entries


def _lines_matching_with(source: str, predicate: Callable[[str], bool],
                         replacement: Callable[[str], str]) -> Tuple[str, int]:
    """Rewrite every line whose stripped text satisfies ``predicate``."""
    out, hits = [], 0
    for line in source.splitlines():
        stripped = line.strip()
        if predicate(stripped):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(indent + replacement(stripped))
            hits += 1
        else:
            out.append(line)
    return "\n".join(out), hits


__all__ = ["Entry", "MUTANT_IMPORTS", "NEEDS", "beta_mutations", "bfast_mutations",
           "body_and_insert", "cases_for", "check_needs", "declare", "form_mutations", "insert_neighbours",
           "replace_all", "rewrite_block", "table"]
