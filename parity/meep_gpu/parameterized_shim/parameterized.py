"""Narrow local stand-in for the MEEP corpus's ``parameterized.expand`` usage.

The corpus environment does not install the third-party ``parameterized`` package.
Six upstream MEEP test modules use only ``@parameterized.parameterized.expand``;
without this deliberately small compatibility module those cases cannot even be
constructed, which removes the Bloch, symmetry, and cylindrical rows this corpus is
meant to census.  This is a *harness-only* import path, supplied to child processes
whose source explicitly imports ``parameterized``.  It is not a package dependency
and does not emulate unobserved API forms.

Generated method names are intentionally not upstream-like.  The runner binds their
identity to the historical lift ledger by an exact facts block, never by a guessed
name.  Each generated method carries its source method, ordinal, and argument tuple
so that matching is inspectable in the saved row.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Iterable


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def expand(cases: Iterable[Any], *_args: Any, **_kwargs: Any) -> Callable[..., None]:
    """Install one method per supplied tuple in the surrounding class body."""
    values = [_as_tuple(case) for case in cases]

    def decorate(function: Callable[..., Any]) -> None:
        namespace = sys._getframe(1).f_locals  # noqa: SLF001 - class-body injection
        for index, arguments in enumerate(values):
            def build(bound: tuple[Any, ...] = arguments,
                      target: Callable[..., Any] = function) -> Callable[..., Any]:
                def generated(self: Any) -> Any:
                    return target(self, *bound)
                return generated

            generated = build()
            generated.__name__ = f"{function.__name__}__mps_corpus_{index}"
            generated.__qualname__ = generated.__name__
            generated.__doc__ = function.__doc__
            generated.__parameterized_source__ = function.__name__
            generated.__parameterized_index__ = index
            generated.__parameterized_args__ = arguments
            namespace[generated.__name__] = generated
        # Upstream replaces the decorated original.  Leaving it callable would run an
        # unparameterized method and measure a row the upstream suite never has.
        return None

    return decorate


class _Parameterized:
    expand = staticmethod(expand)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "the corpus shim implements only parameterized.parameterized.expand")


parameterized = _Parameterized


class param(tuple):  # noqa: N801 - upstream spelling
    def __new__(cls, *args: Any, **kwargs: Any) -> "param":
        if kwargs:
            raise NotImplementedError("the corpus shim does not emulate param(**kwargs)")
        return tuple.__new__(cls, args)


class parameterized_class:  # noqa: N801 - upstream spelling
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("the corpus shim does not emulate parameterized_class")
