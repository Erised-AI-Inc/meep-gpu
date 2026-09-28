"""Minimal stand-in for the ``parameterized`` package, for surveying MEEP's tests.

Ten of MEEP's test modules decorate a test method with
``@parameterized.parameterized.expand([...])``; without the package those modules
do not import, and a survey that reports them as "dependency missing" loses ten
modules' worth of oracles for a decorator with no numerical content whatsoever.

This provides exactly the API those modules use — ``parameterized.expand`` over a
list of argument tuples — and nothing else. It is on ``sys.path`` only for the
children of ``survey_meep_tests.py``, which records ``parameterized_shim`` on every
row produced with it, and it is never installed into any environment.

Semantics matched to the real package for the forms MEEP uses:

* each entry is a tuple of positional arguments, or a bare value (wrapped in a
  1-tuple), which becomes the arguments of one generated test method;
* generated names are ``<original>_<index>[_<safe first arg>]``, so the survey's
  case names line up with what ``pytest`` would print;
* the original undecorated method is removed from the class, exactly as the real
  ``expand`` does, so ``TestLoader.getTestCaseNames`` sees only the expansions.
"""

from __future__ import annotations

import functools
import re


def to_safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_")


class parameterized:  # noqa: N801 - the upstream package spells it lowercase.
    """Namespace carrying :meth:`expand`; the real package's shape is the same."""

    @classmethod
    def expand(cls, input_list, name_func=None, doc_func=None, skip_on_empty=False, **legacy):
        def decorator(func):
            frame_locals = _caller_locals()
            for index, entry in enumerate(input_list):
                args = tuple(entry) if isinstance(entry, (tuple, list)) else (entry,)
                name = _expanded_name(func.__name__, index, args)

                def make(bound_args):
                    @functools.wraps(func)
                    def wrapper(self, *extra, **kwargs):
                        return func(self, *bound_args, *extra, **kwargs)
                    return wrapper

                generated = make(args)
                generated.__name__ = name
                frame_locals[name] = generated
            frame_locals[func.__name__] = None  # Removed by the metaclass sweep below.
            return _Removed(func.__name__)
        return decorator


class _Removed:
    """Placeholder left where the undecorated method was.

    ``TestLoader.getTestCaseNames`` only collects callables, so a non-callable
    attribute is invisible to it — the same net effect as the real package's
    ``delattr``, without needing a metaclass.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only.
        return f"<parameterized-expanded {self.name}>"


def _expanded_name(base: str, index: int, args: tuple) -> str:
    suffix = ""
    if args and isinstance(args[0], (str, int, float, bool)):
        suffix = "_" + to_safe_name(args[0])
    return f"{base}_{index}{suffix}"


def _caller_locals():
    """The class body's namespace, where the generated methods must be written."""
    import inspect

    frame = inspect.currentframe()
    # _caller_locals -> decorator -> the class body being executed. A class body's
    # f_locals is a real, writable mapping in CPython (a function's is not), which
    # is how the upstream package injects its generated methods too.
    for _ in range(2):
        frame = frame.f_back
    return frame.f_locals


param = tuple  # Unused by MEEP's tests; present so `from parameterized import param` works.
