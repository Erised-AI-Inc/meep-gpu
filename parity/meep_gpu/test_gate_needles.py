"""``gate_needles``: what a needle does to a source, and what a gate's table may hold.

Host-only. The tables are the seam a lane re-spells one gate's mutation through
without moving another gate's bytes, so the clauses here are the ones that keep that
seam honest: a needle returns exactly what the hand-written transform it replaced
returned, a borrowed entry IS the shared default's object, and a table refuses by
name every way of arming nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate_needles  # noqa: E402
from gate_needles import Needle  # noqa: E402

SOURCE = ("    float a = f[idx] + kps * src;\n"
          "    f[idx] = a - kms * prev;\n"
          "    float b = D_x[idx] * inv_eps_Ex[idx];\n"
          "    float c = D_y[idx] * inv_eps_Ey[idx];\n"
          "    f[idx] = a - kms * prev;\n")


def test_a_literal_needle_is_str_replace_and_str_count():
    needle = Needle(find="    f[idx] = a - kms * prev;\n", replace="")
    assert needle(SOURCE) == (SOURCE.replace(needle.find, ""), 2)


def test_a_regex_needle_is_subn_with_the_findall_count():
    pattern = r"(D_([xyz])\[idx\]) \* inv_eps_E\2\[idx\]"
    needle = Needle(find=pattern, replace=r"\1", regex=True)
    assert needle(SOURCE) == (re.subn(pattern, r"\1", SOURCE)[0],
                              len(re.findall(pattern, SOURCE)))
    assert needle(SOURCE)[1] == 2
    assert "inv_eps" not in needle(SOURCE)[0]


def test_a_needle_that_matches_nothing_returns_the_source_and_zero():
    needle = Needle(find="never present", replace="x")
    assert needle(SOURCE) == (SOURCE, 0)


def test_the_digest_pins_the_rewrite_and_not_the_prose():
    one = Needle(find="a", replace="b", why="first account")
    two = Needle(find="a", replace="b", why="a different account")
    assert one.digest() == two.digest()
    assert one.digest() != Needle(find="a", replace="c").digest()
    assert one.digest() != Needle(find="a", replace="b", regex=True).digest()


def test_a_borrowed_entry_is_the_shared_object_and_an_owned_one_is_the_gates():
    shared = {"first": Needle(find="a", replace="b"),
              "second": Needle(find="c", replace="d")}
    own = Needle(find="c", replace="e")
    table = gate_needles.gate_table("gate_x", shared, ("first", "second"),
                                    own={"second": own})
    assert table["first"] is shared["first"]
    assert table["second"] is own
    assert table.borrowed == ("first",)
    assert table.owned == ("second",)
    assert list(table) == ["first", "second"]
    assert dict(table) == {"first": shared["first"], "second": own}
    with pytest.raises(KeyError):
        table["third"]  # a name the gate does not arm


def test_a_table_refuses_every_way_of_arming_nothing():
    shared = {"first": Needle(find="a", replace="b")}
    with pytest.raises(KeyError, match="neither owned nor in the shared default"):
        gate_needles.gate_table("gate_x", shared, ("first", "absent"))
    with pytest.raises(ValueError, match="never arms"):
        gate_needles.gate_table("gate_x", shared, ("first",),
                                own={"dead": Needle(find="x", replace="y")})
    with pytest.raises(ValueError, match="more than once"):
        gate_needles.gate_table("gate_x", shared, ("first", "first"))
    with pytest.raises(TypeError, match="not callable"):
        gate_needles.gate_table("gate_x", shared, ("first", "own"), own={"own": "text"})


def test_a_function_entry_is_identified_by_its_qualified_name():
    def rewrite(source):
        return source, 0

    spec = gate_needles.transform_spec(rewrite)
    assert spec == {"function": f"{__name__}.{rewrite.__qualname__}"}
    assert gate_needles.transform_spec(Needle(find="a", replace="b")) == {
        "needle_sha256": Needle(find="a", replace="b").digest()}
