"""Per-gate mutation needle tables for the shared gate harness.

A NEEDLE is one rewrite of a kernel's device text: the defect a gate's source-mutation
leg installs, applied as ``transform(source) -> (mutated source, sites hit)``. The
shared harness (``probe_fused_kernel_bit_identity.py``) spells the curl and
constitutive defects once, as its SHARED DEFAULT (``probe.SOURCE_MUTATIONS``), and
until 0.10 every CUDA gate that borrowed one read it straight out of that default. A
re-spelling there therefore moved every borrowing gate at once, whichever kernel table
it measured.

This module gives each gate its OWN table. :func:`gate_table` builds it from the names
the gate arms: each name resolves to the gate's own needle when the gate supplies one
(``own=``), and otherwise to the shared default's entry, borrowed by identity. So:

* a lane that re-spells a defect for its own kernels adds ONE ``own=`` entry in its
  own gate file, and no other gate's bytes move;
* the shared default is re-spelled only once :func:`NeedleTable.borrowed` shows that
  no gate still borrowing it measures a kernel the new spelling would not match. The
  0.10 inversion of the MEEP-grouping mutations (``bit_parity_0_10/DESIGN.md`` D12,
  KERNELS §7.1) went that way: each lane spelled its gates' defects on its kernels'
  new text, and the shared default took the real-PML pair's spellings last, when no
  table borrowed the old ones, so every gate whose kernels carry that text borrows
  them again;
* ``parity/meep_gpu/test_mutation_anchors.py`` checks every table host-side, against
  the recorded anchor counts in ``gate_needle_anchors.json``, on every test run.

A table is a read-only mapping from name to transform, so a gate indexes it, iterates
it and composes it (``dict(table)``) exactly as it did the shared dictionary. A name
the gate does not arm raises ``KeyError`` naming it, as the dictionary did.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

Transform = Callable[[str], Tuple[str, int]]


@dataclass(frozen=True)
class Needle:
    """Replace every occurrence of ``find`` with ``replace``; report how many there were.

    ``regex`` selects ``re.subn`` over ``str.replace``. The site count is
    ``source.count(find)`` or ``len(re.findall(find, source))`` respectively -- the
    count every hand-written transform in the harness reported -- so a needle and the
    function it replaced return the same pair on every source. ``why`` is the
    defect's account: what it breaks, and whether it must be caught.
    """

    find: str
    replace: str
    regex: bool = False
    why: str = ""

    def __call__(self, source: str) -> Tuple[str, int]:
        if self.regex:
            return (re.subn(self.find, self.replace, source)[0],
                    len(re.findall(self.find, source)))
        return source.replace(self.find, self.replace), source.count(self.find)

    def spec(self) -> Dict[str, Any]:
        """The rewrite itself, without the prose: what a record pins."""
        return {"find": self.find, "replace": self.replace, "regex": self.regex}

    def digest(self) -> str:
        """sha256 of :meth:`spec` as canonical JSON."""
        text = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform_spec(transform: Transform) -> Dict[str, Any]:
    """A table entry's identity for a record: a needle's digest, or a function's name.

    A gate may own an entry that is a function rather than a :class:`Needle` (a
    rewrite that moves a block, or one that matches a pattern no single substitution
    expresses). It is named by its qualified name, which is what changes when it is
    replaced.
    """
    if isinstance(transform, Needle):
        return {"needle_sha256": transform.digest()}
    module = getattr(transform, "__module__", None) or "<unknown>"
    name = getattr(transform, "__qualname__", None) or repr(transform)
    return {"function": f"{module}.{name}"}


class NeedleTable(Mapping[str, Transform]):
    """One gate's needles: every name it arms, each owned by the gate or borrowed.

    Built by :func:`gate_table`; read-only. ``borrowed`` and ``owned`` partition the
    names, so a reader of the table (or of the anchor record) sees which defects this
    gate spells for itself and which it still takes from the shared default.
    """

    def __init__(self, gate: str, entries: Mapping[str, Transform],
                 borrowed: Sequence[str]) -> None:
        self._gate = gate
        self._entries: Dict[str, Transform] = dict(entries)
        self._borrowed = tuple(name for name in self._entries if name in set(borrowed))

    @property
    def gate(self) -> str:
        return self._gate

    @property
    def borrowed(self) -> Tuple[str, ...]:
        """Names whose entry IS the shared default's object."""
        return self._borrowed

    @property
    def owned(self) -> Tuple[str, ...]:
        """Names whose entry this gate supplies itself."""
        return tuple(name for name in self._entries if name not in self._borrowed)

    def __getitem__(self, name: str) -> Transform:
        return self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (f"NeedleTable({self._gate!r}: {len(self._entries)} needles, "
                f"{len(self._borrowed)} borrowed)")


def gate_table(gate: str, shared: Mapping[str, Transform], names: Sequence[str],
               own: Optional[Mapping[str, Transform]] = None) -> NeedleTable:
    """The table of ``gate``: every name in ``names``, in that order.

    A name in ``own`` takes the gate's entry; any other name is borrowed from
    ``shared`` by identity. Refused, by name: a name neither supplies, a name armed
    twice, an owned entry the gate never arms (it would be dead), and an entry that is
    not callable.
    """
    own = dict(own or {})
    order = list(names)
    repeated = sorted({name for name in order if order.count(name) > 1})
    if repeated:
        raise ValueError(f"{gate}: needles armed more than once: {repeated}")
    dead = sorted(set(own) - set(order))
    if dead:
        raise ValueError(f"{gate}: owned needles that the gate never arms: {dead}")
    missing = sorted(name for name in order if name not in own and name not in shared)
    if missing:
        raise KeyError(f"{gate}: needles neither owned nor in the shared default: "
                       f"{missing}")
    entries: Dict[str, Transform] = {}
    borrowed = []
    for name in order:
        if name in own:
            entries[name] = own[name]
        else:
            entries[name] = shared[name]
            borrowed.append(name)
        if not callable(entries[name]):
            raise TypeError(f"{gate}: needle {name!r} is not callable")
    return NeedleTable(gate, entries, borrowed)
