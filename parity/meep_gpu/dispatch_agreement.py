"""FLOOR 8 — every dispatch total a board publishes equals the measurement.

WHY THIS IS ITS OWN MODULE AND NOT PART OF ``fusion_taxonomy``. It began there, beside
FLOOR 7, which is where it belongs conceptually — and that placement cost a Metal board
cut. Ten Metal H->D gate artifacts pin ``parity/meep_gpu/fusion_taxonomy.py`` in their
import closure, so ADDING a function to that module drifted every one of them and
``build_fusion_matrix.py`` refused the cut by name: "the bytes this cut credits are NOT
the bytes the gates certified". The floor is a BOARD-side check that no gate consults, so
the honest fix is not to re-run ten MPS gates for a change that provably cannot reach a
kernel verdict, and not to invent a declared-drift exemption on a board that has none —
it is to put the code where it does not sit in a gate's closure. Moving it back restored
``fusion_taxonomy.py`` to ``257e4e10c93b``, byte-for-byte the digest those ten artifacts
recorded.

THE SPLIT IS ALSO THE RIGHT ONE ON ITS OWN TERMS. ``fusion_taxonomy`` is the SERVED-axis
taxonomy: buckets, the instance ledger, and FLOOR 7 comparing every published served total
to that ledger. ``served`` and ``served_in_dispatch`` are different numbers with different
authorities — a served total is DERIVED from the instance ledger, a dispatch total is
MEASURED off the tree by ``dispatch_reachability.served_in_dispatch`` — so the two floors
answering to two authorities in two modules is clearer than one module holding both.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["DISPATCH_CONDITION_CONTAINERS", "DISPATCH_HEADLINE_KEYS",
           "dispatch_agreement"]


#: THE SAME FLOOR FOR THE OTHER AXIS. ``served`` and ``served_in_dispatch`` are
#: DIFFERENT NUMBERS -- one licenses a predicate verdict, the other a driver consult --
#: and :func:`headline_agreement` was only ever told about the first. That gap was not
#: hypothetical: measured 2026-09-11 on ``fusion_matrix_metal_2026-09-11_dispatch``,
#: the Metal board published ``served_by_a_fused_product_in_dispatch: 0`` at its top
#: level -- a LITERAL left behind when the table was unreleased -- beside its own
#: ``headline.served_in_dispatch_measurement.served_in_dispatch`` of 176, in one
#: artifact, with a ``why_dispatch_is_zero`` string sending the reader to a
#: ``why_zero`` that was ``None`` and an ``arms`` table that was not empty. Nothing
#: refused the cut, because every floor the boards had was about the served axis.
#:
#: A NAME ADDED TO A BOARD'S HEADLINE FOR ITS DISPATCH TOTAL GOES HERE IN THE SAME
#: CHANGE, for the reason stated at :data:`SERVED_HEADLINE_KEYS`: the walk cannot see
#: a key it was not told about.
DISPATCH_HEADLINE_KEYS: Sequence[str] = (
    "served_in_dispatch",                      # dispatch_reachability's own name
    "served_by_a_fused_product_in_dispatch",   # the Metal board's top level
)

#: SUBTREES WHOSE DISPATCH NUMBER IS A DIFFERENT CONDITION, exempt BY NAME rather
#: than by a predicate over values.
#:
#: ``by_default_precedence`` is the CUDA board's by-DEFAULT answer: what dispatches
#: when the user sets no ``MEEP_GPU_BACKEND_PREFERENCE``, measured by the
#: ``cuda_default_precedence`` leg and published beside the by-preference headline. It
#: reads 0 while the headline reads 209 and BOTH are correct -- they are two different
#: questions, the same way ``served_three_seam_ledger`` is a different fact from
#: ``served_by_a_fused_product`` rather than a second spelling of it. An equality floor
#: that did not know this would refuse a correct CUDA board, and the wrong repair would
#: be to delete the by-default number, which is the one thing in that artifact a user
#: choosing no preference actually needs.
DISPATCH_CONDITION_CONTAINERS: Sequence[str] = (
    "by_default_precedence",
)


def _walk_dispatch(node: Any, path: str, found: List[Any],
                   exempt: Optional[str] = None) -> None:
    """Every dispatch-headline occurrence, with the condition subtrees marked.

    ``exempt`` carries the name of the enclosing condition container once the walk
    has entered one, so the caller can report those occurrences WITHOUT comparing
    them -- a reader of the block can still see they were found and why they were
    not checked, which is what stops the exemption from being invisible.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            inner = exempt or (key if key in DISPATCH_CONDITION_CONTAINERS else None)
            if key in DISPATCH_HEADLINE_KEYS:
                found.append((here, value, inner))
            _walk_dispatch(value, here, found, inner)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk_dispatch(value, f"{path}[{index}]", found, exempt)


def dispatch_agreement(result: Dict[str, Any], measured: Dict[str, Any],
                       backend: str) -> Dict[str, Any]:
    """FLOOR 8 -- every dispatch total the artifact publishes equals the measurement.

    The dispatch sibling of :func:`headline_agreement`, and it compares against a
    different authority for a reason. A served total is DERIVED from the instance
    ledger, so the ledger is what it must equal. A dispatch total is MEASURED off the
    tree by ``dispatch_reachability.served_in_dispatch`` -- which walks
    ``fastpath.py``'s parse tree, joins each served product to an arm label and asks
    the release table whether that arm dispatches on that row -- so the measurement
    block IS the authority, and what this floor refuses is an artifact that publishes
    any other number for the same question.

    WHY IT IS NOT ENOUGH TO "JUST READ THE MEASUREMENT". Every board already does,
    somewhere; the defect this closes is a SECOND publication of the same number that
    stopped tracking it. A hardcoded 0 beside a measured 176 is not a stale comment,
    it is the headline a reader quotes, and it survived because no floor on the
    dispatch axis existed at all.

    Occurrences inside a :data:`DISPATCH_CONDITION_CONTAINERS` subtree are LISTED and
    not compared -- see that constant for why the CUDA board's by-default 0 beside a
    by-preference 209 is two correct answers rather than a disagreement.
    """
    if "served_in_dispatch" not in measured:
        raise SystemExit(
            f"{backend}: the dispatch measurement block carries no "
            f"'served_in_dispatch', so there is nothing for the artifact's dispatch "
            f"headline to be checked against.")
    expected = int(measured["served_in_dispatch"])
    found: List[Any] = []
    _walk_dispatch(result, "", found)
    if not found:
        raise SystemExit(
            f"{backend}: the artifact publishes none of "
            f"{list(DISPATCH_HEADLINE_KEYS)}. A dispatch total under a name this "
            f"floor was not told about is a headline nothing checks; add the key to "
            f"fusion_taxonomy.DISPATCH_HEADLINE_KEYS.")
    checked = [(path, value) for path, value, exempt in found if exempt is None]
    conditional = [{"path": path, "value": value, "condition": exempt}
                   for path, value, exempt in found if exempt is not None]
    if not checked:
        raise SystemExit(
            f"{backend}: every dispatch total in the artifact sits inside a declared "
            f"condition container {list(DISPATCH_CONDITION_CONTAINERS)}, so the "
            f"unconditional headline is missing and this floor checked nothing.")
    disagreeing = [(path, value) for path, value in checked if value != expected]
    if disagreeing:
        shown = ", ".join(f"{path}={value!r}" for path, value in disagreeing[:6])
        raise SystemExit(
            f"{backend}: dispatch_reachability measured {expected} served instances "
            f"as DISPATCHING and the artifact publishes a different dispatch total at "
            f"{len(disagreeing)} path(s): {shown}. A dispatch headline is READ from "
            f"the measurement, never typed beside it -- the Metal board carried a "
            f"hardcoded 0 next to its own measured 176 until 2026-09-11, and this is "
            f"the split this floor refuses. If the number is meant to be a different "
            f"CONDITION, put it under a container named in "
            f"fusion_taxonomy.DISPATCH_CONDITION_CONTAINERS with its reason.")
    return {
        "served_in_dispatch": expected,
        "derived_from": ("dispatch_reachability.served_in_dispatch(<backend>), "
                         "walked off fastpath.py's parse tree and the release table"),
        "paths_checked": [path for path, _value in checked],
        "conditional_totals_not_compared": conditional,
        "is_an_upper_bound": bool(measured.get("is_an_upper_bound")),
        "agree": True,
    }
