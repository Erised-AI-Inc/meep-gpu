"""The launchable-plan base every Metal family subclasses.

WHAT A PLAN IS. Built once for a (fields, pml, sub-step) triple and launched per
timestep. Everything resolvable ahead of the loop is resolved at build time: the
coverage verdict, the boundary specialisation, the Yee sub-lattice pairing, the
compiled shader functions, the flattened coefficient views and the device mirrors.
NOTHING ON THE LAUNCH PATH ALLOCATES, COMPILES OR COPIES — the argument tuple is
built once and :meth:`KernelPlan.run` unpacks one tuple and calls one function.

WHY THIS BASE EXISTS. ``PmlCurlPlan`` and ``ConstitutivePlan`` duplicated their
``__slots__``, their variant selector and their ``__repr__`` almost verbatim while
the package carried one family. With a tree of families that duplication becomes
the place a contract quietly drifts, so exactly the shared part is here and each
family contributes only its ``_args`` layout. NOTHING IS ADDED TO THE LAUNCH PATH:
:meth:`run` is still one dict lookup and one call.

THE VARIANT SELECTOR IS A CONTRACT, NOT A CONVENIENCE. ``contract`` is not for
callers — it exists so a byte gate can MEASURE the contraction guard's effect
(identical with, non-identical without). ``None`` takes the pinned mode. A mode the
plan was NOT built with RAISES rather than silently launching the pinned one,
because a guard argument that quietly did nothing would make the gate's most
important leg vacuous. Note the polarity is the opposite of Triton's launch
keyword: ``enable_fp_fusion=True`` (contraction on, guard removed) corresponds to
``contract="fast"`` here, because on Metal the directive is a property of the
SOURCE and each mode is a different compiled kernel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .shaders import CONTRACT_OFF


class KernelPlan:
    """A launchable, allocation-free Metal sub-step.

    Subclasses set :attr:`_args` (the exact positional tuple the shader entry point
    takes) and, by convention, name themselves in :meth:`describe`. They must NOT
    override :meth:`run`: one launch path is what makes "the bytes the gate
    certifies are the bytes the engine would launch" true across a family tree.
    """

    __slots__ = ("_args", "_functions", "launches")

    #: Subclasses list the attributes ``__repr__`` reports, in order.
    REPR_FIELDS: Tuple[str, ...] = ()

    #: HOW MANY KERNEL LAUNCHES ONE ``run`` OWES. One for this base, by
    #: construction: :meth:`run` unpacks one tuple and calls one function. It is
    #: DECLARED rather than left implicit because the whole-step gate asserts the
    #: exact per-cycle launch count on every filled slot — that assertion is what
    #: stops a slot passing by not executing — and a plan whose ``run`` dispatches
    #: twice (``cylindrical_real``'s scan-then-curl) would otherwise either fail a
    #: correct composition or force the gate to special-case it by class name.
    #: A subclass that adds a dispatch overrides this in the same change.
    launches_per_run: int = 1

    def __init__(self, functions: Dict[str, Any], args: Sequence[Any]) -> None:
        self._functions = dict(functions)
        self._args = tuple(args)
        # Launch counting is not diagnostics: a leg that certifies "the plan
        # reproduced the array path" by comparing bytes is trivially satisfied by a
        # plan that was never invoked, so every gate asserts this moved.
        self.launches = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted(self._functions))

    def run(self, contract: Optional[str] = None) -> None:
        """Launch the sub-step against the mirrors. In place."""
        mode = CONTRACT_OFF if contract is None else contract
        function = self._functions.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} "
                f"rather than launching the pinned one")
        self.launches += 1
        function(*self._args)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}"
                          for name in self.REPR_FIELDS)
        joiner = ", " if parts else ""
        return f"{type(self).__name__}({parts}{joiner}variants={self.variants})"

    def __repr__(self) -> str:
        return self.describe()
