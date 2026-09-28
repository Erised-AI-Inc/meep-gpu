"""A local stand-in for the `parameterized` package, present ONLY on this leg's path.

The package is not installed in this environment and the harness installs none, so
the six MEEP test modules that decorate with ``@parameterized.parameterized.expand``
cannot be imported — 24 of the 134 engine-accepted rows, and they are exactly the
special_kz / Bloch-k / mirror rows the newly certified families exist for. Skipping
them would understate precisely what this round measures.

WHAT IS REPRODUCED, AND WHAT IS NOT.

Reproduced: the semantics that select a run — one generated method per tuple in the
decorator's list, in list order, called as ``func(self, *tuple)``. Every module in
scope uses exactly that one form (``@parameterized.parameterized.expand([...])``,
verified by grep across all six).

NOT reproduced, deliberately: the upstream NAME. ``parameterized.expand`` builds
``{func}_{index}_{safe_first_param}``; re-deriving that string would be a guess, and a
guess is not allowed to decide which recorded row a measurement belongs to. The methods
here are named ``{func}__idx{N}`` — unmistakably this file's naming — and each carries
its parameter tuple on ``__parameterized_args__``.

IDENTITY IS THEREFORE MEASURED, NOT NAMED. The analysis matches each generated case to
a row of the 2026-08-09 lift record by exact equality of the ``_facts`` block —
cell size, resolution, Courant, k_point, source and boundary kinds, symmetry kinds,
geometry and DFT counts — computed by the survey harness's own ``_facts`` function on
both sides. A case whose facts match no recorded row, or match more than one, is
reported unmatched rather than assigned.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Iterable


def _tuple(args: Any) -> tuple:
    if isinstance(args, (tuple, list)):
        return tuple(args)
    return (args,)


def expand(input_list: Iterable[Any], *_args: Any, **_kwargs: Any) -> Callable:
    """Inject one method per parameter tuple into the calling class body."""
    cases = [_tuple(entry) for entry in input_list]

    def decorator(func: Callable) -> None:
        namespace = sys._getframe(1).f_locals  # noqa: SLF001 - the class body, as upstream
        for index, args in enumerate(cases):
            def make(bound: tuple = args, target: Callable = func) -> Callable:
                def method(self):  # noqa: ANN001, ANN202
                    return target(self, *bound)
                return method

            method = make()
            method.__name__ = f"{func.__name__}__idx{index}"
            method.__qualname__ = method.__name__
            method.__doc__ = func.__doc__
            method.__parameterized_args__ = args
            method.__parameterized_index__ = index
            method.__parameterized_source__ = func.__name__
            namespace[method.__name__] = method
        # The undecorated original must not remain a runnable test: upstream replaces it
        # with a non-callable placeholder and unittest's loader skips it for the same
        # reason (``callable(getattr(cls, name))`` is False).
        return None

    return decorator


class _Parameterized:
    """The ``parameterized.parameterized`` namespace the tests reach through."""

    expand = staticmethod(expand)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "only parameterized.parameterized.expand is stood in for; this module "
            "reproduces no other entry point rather than guessing at one")


parameterized = _Parameterized


class param(tuple):  # noqa: N801 - upstream spelling
    def __new__(cls, *args: Any, **kwargs: Any):
        if kwargs:
            raise NotImplementedError(
                "param(**kwargs) is not stood in for; no module in scope uses it")
        return tuple.__new__(cls, args)


class parameterized_class:  # noqa: N801 - upstream spelling
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "parameterized_class is not stood in for; no module in scope uses it")
