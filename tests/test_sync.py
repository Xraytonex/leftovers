"""Hand-written differential cases for execute_sync: each must match stock graphql-core
exactly (data and errors in order) and run on the compiled path, or fall back where expected."""

from __future__ import annotations

import pytest
from graphql import (
    GraphQLArgument,
    GraphQLBoolean,
    GraphQLField,
    GraphQLFloat,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLString,
    execute,
    execute_sync,
    parse,
)

from leftovers import STATS, CompiledExecutor
from tests.fixtures import Custom, O, run_sync, sync_schema

CASES = [
    ("all fields", '{ people { id name age tags pets { name type } school { id name } greet(who: "bob") } }',
     None, True),
    ("alias/fragment/typename/vars",
     "query Q($w: String!) { people { pid: id ...F nm: name name __typename "
     "school { __typename ... on School { name } } greet(who: $w) } } fragment F on Person { age pets { name } }",
     {"w": "amy"}, True),
    ("bubble to item", "{ people { id nn_name } }", None, True),
    ("bubble to list", "{ strict { id nn_name } }", None, True),
    ("bubble to root", "{ must { id nn_name } }", None, True),
    ("non-null list item", "{ people { strict_pets { name } } }", None, True),
    ("fallback: variable @skip", "query ($s: Boolean!) { people { id name @skip(if: $s) } }", {"s": True}, False),
    ("repeat (cache hit)", "{ people { id name } }", None, True),
]


@pytest.mark.parametrize(("name", "query", "variables", "expect_compiled"), CASES, ids=[c[0] for c in CASES])
def test_cases(name, query, variables, expect_compiled) -> None:
    stock, compiled, used = run_sync(sync_schema, query, variables)
    assert compiled == stock
    assert used == expect_compiled


def _nested_schema(n: int) -> GraphQLSchema:
    """n objects with four scalars and five nested objects each, on default resolvers."""

    def obj(name: str, fields: dict) -> GraphQLObjectType:
        return GraphQLObjectType(name, lambda: {k: GraphQLField(v) for k, v in fields.items()})

    pair = {"id": GraphQLInt, "name": GraphQLString}
    item = obj("Item", {
        "id": GraphQLInt, "name": GraphQLString, "lastname": GraphQLString, "age": GraphQLInt,
        "address": obj("Address", {"street": GraphQLString, "number": GraphQLInt}),
        "job": obj("Job", {"id": GraphQLInt, "org_name": GraphQLString}),
        "partner": obj("Partner", pair), "school": obj("School", pair),
        "pets": GraphQLList(obj("Pet", {"name": GraphQLString, "type": GraphQLString})),
    })
    rows = [O(id=i, name=f"name{i}", lastname=f"last{i}", age=20 + i % 50,
              address=O(street=f"street {i}", number=i % 300), job=O(id=i * 7, org_name=f"org{i % 97}"),
              partner=O(id=i + 1, name=f"partner{i}"), school=O(id=i % 40, name=f"school{i % 40}"),
              pets=[O(name=f"pet{i}a", type="dog"), O(name=f"pet{i}b", type="cat")]) for i in range(n)]
    return GraphQLSchema(GraphQLObjectType("Query", {"items": GraphQLField(GraphQLList(item),
                                                                           resolve=lambda *_: rows)}))


def test_nested_objects_workload() -> None:
    query = ("{ items { id name lastname age address { street number } job { id org_name } "
             "partner { id name } pets { name type } school { id name } } }")
    stock, compiled, used = run_sync(_nested_schema(500), query)
    assert compiled == stock
    assert used


def test_argument_values_are_fresh_per_item() -> None:
    def y(src, info, xs):  # mutates its list argument
        xs.append(9)
        return list(xs)

    obj = GraphQLObjectType("O", {"y": GraphQLField(GraphQLList(GraphQLInt),
                                                    {"xs": GraphQLArgument(GraphQLList(GraphQLInt))}, resolve=y)})
    schema = GraphQLSchema(GraphQLObjectType("Query", {"c": GraphQLField(GraphQLList(obj),
                                                                         resolve=lambda *_: [1, 2, 3])}))
    stock, compiled, used = run_sync(schema, "{ c { y(xs: [0]) } }")
    assert compiled == stock
    assert used


def test_arguments_not_coerced_for_unreached_fields() -> None:
    calls = {"n": 0}

    def parse_literal(node, _vars=None):
        calls["n"] += 1
        return int(node.value)

    scalar = GraphQLScalarType("S", parse_value=lambda v: v, parse_literal=parse_literal)
    obj = GraphQLObjectType("P", {"f": GraphQLField(GraphQLInt, {"a": GraphQLArgument(scalar)},
                                                    resolve=lambda *_, **__: 1)})
    schema = GraphQLSchema(GraphQLObjectType("Query", {"o": GraphQLField(GraphQLList(obj), resolve=lambda *_: [])}))
    doc = parse("{ o { f(a: 1) } }")
    execute_sync(schema, doc)
    stock_calls, calls["n"] = calls["n"], 0
    execute_sync(schema, doc, executor_class=CompiledExecutor)
    assert calls["n"] == stock_calls


def test_cache_hits_when_query_is_reparsed() -> None:
    """Servers parse the query text on every request; the compile cache keys on the text."""
    schema = GraphQLSchema(GraphQLObjectType("Query", {"n": GraphQLField(GraphQLList(GraphQLInt),
                                                                         resolve=lambda *_: [1, 2])}))
    compiles, runs = STATS["compiles"], STATS["compiled_runs"]
    for _ in range(200):
        execute_sync(schema, parse("{ n }"), executor_class=CompiledExecutor)
    assert STATS["compiles"] - compiles == 1
    assert STATS["compiled_runs"] - runs == 200


def test_leaf_list_fast_path() -> None:
    inf = float("inf")
    data = {
        "s": [["a", "b"], [], ["a", None], ["a", 1], ("a", "b"), ["a", True], "abc", [Custom()]],
        "s_nn": [["a", "b"], ["a", None], []],
        "i": [[1, 2], [1, 2**31], [-(2**31) - 1], [1, None], [True, 2], [1.0], []],
        "i_nn": [[1, 2], [1, None]],
        "f": [[1.5, 2.0], [inf], [float("nan")], [1, 2.5], [None, 1.5], [True]],
        "b": [[True, False], [True, None], [1, 0], []],
    }
    item = {"s": GraphQLString, "s_nn": GraphQLNonNull(GraphQLString), "i": GraphQLInt,
            "i_nn": GraphQLNonNull(GraphQLInt), "f": GraphQLFloat, "b": GraphQLBoolean}
    rows = [{k: vals[n % len(vals)] for k, vals in data.items()} for n in range(8)]
    row = GraphQLObjectType("Row", {k: GraphQLField(GraphQLList(t)) for k, t in item.items()})
    schema = GraphQLSchema(GraphQLObjectType("Query", {"rows": GraphQLField(GraphQLList(row),
                                                                            resolve=lambda *_: rows)}))
    stock, compiled, used = run_sync(schema, "{ rows { s s_nn i i_nn f b } }")
    assert compiled == stock
    assert used


def test_sync_mode_has_no_awaitable_checks() -> None:
    """execute_sync stays on the sync path: no is_awaitable calls at all."""
    calls = []

    def counting(value):
        calls.append(value)
        return False

    doc = parse("{ people { id name } }")
    result = execute(sync_schema, doc, executor_class=CompiledExecutor, is_awaitable=counting)
    assert not hasattr(result, "__await__")
    assert calls  # aware mode: a custom is_awaitable is honoured
    calls.clear()
    execute_sync(sync_schema, doc, executor_class=CompiledExecutor)
    assert calls == []
