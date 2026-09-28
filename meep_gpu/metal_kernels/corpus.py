"""Publish a family's emitted-shader corpus as one digest, from its emitter alone.

WHY. A weld binds a gate's verdict to the sha256 of the Python files it imported, which
cannot tell a docstring fix from a changed coefficient. `meep_gpu/device_identity.py`
lifts that to the CUDA track's rule -- an edit keeps a weld only if the DEVICE SOURCE is
unchanged -- and for Metal the device source is the emitted shader text. It reads a
module's ``corpus_digest()`` and, finding none, refuses rather than guessing.

Measured 2026-08-22: three of the forty-one Metal kernel modules published one, so 2 of 42
welds could be pinned by device source and the other 40 sat on the weaker
comments-only tier. The three that did each hand-rolled a ``specialisations()`` product
over the same small vocabulary. This is that vocabulary, factored out, so a module
publishes a corpus by naming its emitter instead of re-deriving an enumeration.

THE DOMAINS ARE CANONICAL, NOT EXHAUSTIVE, and the distinction is the whole safety
argument. The enumeration is a FINGERPRINT over a fixed, declared sweep -- a changed
character anywhere in the emitted text moves the digest -- not a claim that every
specialisation in it is reachable, gated, or certified. What a gate certified is the
subset it launched, recorded per-source in its own artifact. ``offdiag_update_e``
(:func:`~.offdiag_update_e.specialisations`) says the same thing about its own sweep.

FAIL CLOSED. A parameter this table does not know is a refusal, not a default: guessing a
domain would silently narrow the sweep and make the digest agree across a change it never
covered. Add the parameter here, deliberately, or leave the module unpublished.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

from . import templates

__all__ = ["DOMAINS", "OVERRIDES", "corpus_digest_for", "UnknownEmitterParameter"]

_CODES: Tuple[int, ...] = (templates.PERIODIC, templates.METALLIC)
_TRIPLES: Tuple[Tuple[int, int, int], ...] = tuple(
    itertools.product(_CODES, repeat=3))
_FLAG_TRIPLES: Tuple[Tuple[int, int, int], ...] = tuple(
    itertools.product((0, 1), repeat=3))
_BOOL_TRIPLES: Tuple[Tuple[bool, bool, bool], ...] = tuple(
    itertools.product((False, True), repeat=3))


class UnknownEmitterParameter(TypeError):
    """The emitter takes a parameter with no declared domain."""


#: Parameter name -> the canonical values swept for it. Keep this table small and
#: deliberate; every entry widens what a digest covers for every module that uses it.
DOMAINS: Dict[str, Tuple[Any, ...]] = {
    "codes": _TRIPLES,                 # per-axis boundary code
    "walls": _TRIPLES,
    "backward": (False, True),         # which half-step the curl is
    "phased": _FLAG_TRIPLES,           # per-axis Bloch phase present
    "phases": _FLAG_TRIPLES,
    "zero_metal": _BOOL_TRIPLES,       # per-axis inline wall clear
    "conductive": _BOOL_TRIPLES,
    "has_beta": (False, True),
    "expansion": ("FMA_V1", "NAIVE"),  # the two measured complex-multiply arms
    "contract": (templates.CONTRACT_OFF,),
}


def _domain(name: str, overrides: Dict[str, Sequence[Any]]) -> Tuple[Any, ...]:
    if name in overrides:
        return tuple(overrides[name])
    if name in DOMAINS:
        return DOMAINS[name]
    raise UnknownEmitterParameter(
        f"no declared domain for emitter parameter {name!r}: add it to "
        f"metal_kernels.corpus.DOMAINS deliberately, or pass it in `overrides`. "
        f"Guessing would narrow the sweep silently, which is the failure this refuses.")


def corpus_digest_for(emitter: Callable[..., str],
                      **overrides: Sequence[Any]) -> Dict[str, Any]:
    """One sha256 over every source ``emitter`` produces on the canonical sweep.

    Returns ``{"count", "sha256", "parameters"}``. The parameters are recorded so a
    reader can see what the digest covered without reading this module -- a digest whose
    sweep is invisible is a number nobody can reproduce.

    Emission must be deterministic and free of any device; both hold for the Metal
    emitters, which are pure string builders (measured: two calls of
    ``complex_fields.bloch_constitutive_source`` return byte-identical 4,461-char text).
    """
    signature = inspect.signature(emitter)
    names = [name for name, p in signature.parameters.items()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    domains = [_domain(name, overrides) for name in names]
    digest = hashlib.sha256()
    count = refused = 0
    for combination in itertools.product(*domains):
        try:
            source = emitter(*combination)
        except ValueError:
            # THE EMITTER'S OWN STRUCTURAL REFUSAL, and it belongs in the corpus as an
            # absence rather than a crash: a Bloch phase on a metallic axis is not a
            # source this family can emit (complex_fields.py:411 refuses it, as
            # stepping._bloch_phases does). Counted, not swallowed -- a module whose
            # refusal count jumps has had its domain of validity moved, and the count
            # sits beside the digest so that is visible without reading this file.
            refused += 1
            continue
        digest.update(source.encode("utf-8"))
        count += 1
    if not count:
        raise ValueError(
            f"{getattr(emitter, '__name__', emitter)!r} emitted nothing over the "
            f"canonical sweep ({refused} combinations refused); a digest over an empty "
            f"corpus would certify nothing while looking like a measurement")
    return {"count": count, "refused": refused, "sha256": digest.hexdigest(),
            "parameters": {n: len(d) for n, d in zip(names, domains)}}


#: DELIBERATELY NOT PUBLISHED: ``complex_conductive_fused_pair``. Its emitter takes
#: ``pole_counts`` as a THREE-TUPLE, one entry per E component, non-negative and with no
#: cap stated anywhere in that module (``complex_conductive_fused_pair.py:369-374``). The
#: full product over even a modest per-entry range is six figures of emissions, so
#: publishing it means choosing a canonical subset -- and a subset chosen here, with no
#: measurement saying which pole configurations the shader text actually varies over,
#: is precisely the silent narrowing this file refuses. It stays on the comments-only
#: tier until something establishes that domain.

#: PER-MODULE WIDENINGS, declared rather than inferred. The default ``codes`` domain is
#: periodic and metallic only; a folded family exists to carry the mirror fills and
#: refuses an unfolded grid outright ("no axis is folded", folded_fused_pair.py), so every
#: combination of the default sweep is refused and its corpus would be empty. Widening the
#: default instead would silently change what every other module's digest covers, so the
#: widening is named per module and visible here.
_FOLDED_CODES = tuple(
    itertools.product((templates.PERIODIC, templates.METALLIC, 3), repeat=3))

#: The dispersive pair's own bounds check (``fused_dispersive_pair.py:316-319``) states
#: both domains outright: ``axis`` must be 0, 1 or 2, and ``pole_count`` must lie in
#: ``[0, MAX_POLES]``. Taken from the emitter's refusal rather than chosen here, so the
#: sweep cannot be narrower than what the emitter itself accepts.
from .complex_dispersive_update_e import MAX_POLES as _MAX_POLES  # noqa: E402

#: The cylindrical pair's two closed axes, from the module's own ``specialisations()``
#: (``cylindrical_fused_magnetic_pair.py:558-561``): z periodic or metallic, and the two
#: certified m arms. ``zero_metal`` stays the default triple -- the emitter REFUSES a wall
#: on r or phi (`:479-481`), so those combinations are counted as refusals rather than
#: quietly dropped, which is how the count beside the digest stays readable.
def _cylindrical_arms() -> Tuple[Any, ...]:
    from . import cylindrical_complex  # noqa: PLC0415 - avoids an import cycle
    return (cylindrical_complex.M_ONE, cylindrical_complex.M_MANY)


OVERRIDES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "folded_fused_pair": {"codes": _FOLDED_CODES},
    "folded_fused_magnetic_pair": {"codes": _FOLDED_CODES},
    "fused_dispersive_pair": {"axis": (0, 1, 2),
                              "pole_count": tuple(range(_MAX_POLES + 1))},
    "cylindrical_fused_magnetic_pair": {"bcz": _CODES, "m_arm": _cylindrical_arms()},
    "folded_complex_fused_magnetic_pair": {"codes": _FOLDED_CODES},
    # ADDED 2026-08-27 with the family. Without it `corpus_digest_for` refuses
    # outright -- all 1024 combinations of the CANONICAL sweep emit nothing,
    # because this emitter takes the folded quadruple `folded_axis_kinds`
    # resolves and refuses a code that is not a fold. Same override, same
    # reason, as the folded complex parent one line above.
    "folded_beta_complex_fused_magnetic_pair": {"codes": _FOLDED_CODES},
    "folded_complex": {"codes": _FOLDED_CODES},
    "folded_beta": {"codes": _FOLDED_CODES},
    "folded_offdiag_update_e": {"codes": _FOLDED_CODES},
    "complex_folded_offdiag_update_e": {"codes": _FOLDED_CODES},
}
