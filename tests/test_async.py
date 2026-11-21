"""Hand-written differential cases for async execute: each must match stock graphql-core
exactly (data and errors in order) and run on the compiled path."""

from __future__ import annotations

import asyncio

import pytest
from graphql import (
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    execute,
    parse,
)

import leftovers
from leftovers import CompiledExecutor
from tests.fixtures import later, make_async_schema, run_async, settle

CASES = [
    ("sync resolvers via async entry", "{ people { id name age tags pets { name type } school { id name } } }"),
    ("async root, sync below", "{ apeople { id name age pets { name } school { name } } }"),
    ("async per-item greet (args)", '{ people { id greet(who: "amy") } }'),
    ("async nullable raising", "{ people { id a_raise } }"),
    ("async non-null None -> item", "{ apeople { id a_nn } }"),
    ("async non-null in [T!] -> list", "{ astrict { id a_nn } }"),
    ("async root non-null -> data null", "{ amust { id nn_name } }"),
    ("dataloader per item", "{ people { id a_school { id name } } }"),
    ("awaitable list items", "{ people { id coro_pets { name } } }"),
    ("awaitable [Pet!] item None", "{ people { id strict_coros { name } } }"),
    ("callable attr -> awaitable", "{ people { id attr_coro } }"),
    ("mixed: async root + loader + sync errors",
     "{ apeople { id nn_name age a_raise a_school { name } pets { name } } }"),
]


def check(schema: GraphQLSchema, query: str, root=None) -> None:
    stock, compiled, used = asyncio.run(run_async(schema, query, root))
    assert compiled == stock
    assert used


@pytest.mark.parametrize(("name", "query"), CASES, ids=[c[0] for c in CASES])
def test_cases(name, query) -> None:
    check(make_async_schema(), query)


def test_async_is_type_of() -> None:
    async def never(value, info):  # stock awaits an async is_type_of
        return False

    t = GraphQLObjectType("T", {"x": GraphQLField(GraphQLInt, resolve=lambda s, i: 1)}, is_type_of=never)
    check(GraphQLSchema(GraphQLObjectType("Query", {"t": GraphQLField(t, resolve=lambda s, i: object())})),
          "{ t { x } }")


async def agen():
    for i in range(3):
        yield i


class Obj:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def test_async_iterables() -> None:
    schema = GraphQLSchema(GraphQLObjectType("Query", {
        "l": GraphQLField(GraphQLList(GraphQLInt), resolve=lambda s, i: agen()),
        "d": GraphQLField(GraphQLList(GraphQLInt)),  # default resolver on a fresh root each run
    }))
    check(schema, "{ l }")
    check(schema, "{ d }", lambda: Obj(d=agen()))


async def _run_both(schema: GraphQLSchema, query: str, probe) -> list:
    """Run stock, then compiled, and return `probe()` after each run has settled."""
    out = []
    for cls in (None, CompiledExecutor):
        await settle(execute_kw(schema, query, cls))
        for _ in range(3):
            await asyncio.sleep(0)
        out.append(probe())
    return out


def execute_kw(schema: GraphQLSchema, query: str, cls):
    return execute(schema, parse(query), **({"executor_class": cls} if cls else {}))


def test_discarded_pending_work_still_runs() -> None:
    """A sibling's non-null failure discards pending work; stock still lets it finish."""
    ran: list[str] = []

    async def side(src, info):
        await asyncio.sleep(0)
        ran.append("side")
        return 1

    obj = GraphQLObjectType("O", {"x": GraphQLField(GraphQLInt, resolve=side),
                                  "y": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=lambda s, i: None)})
    schema = GraphQLSchema(GraphQLObjectType("Query", {"o": GraphQLField(obj, resolve=lambda s, i: {})}))

    def probe():
        effects = list(ran)
        ran.clear()
        return effects

    stock, compiled = asyncio.run(_run_both(schema, "{ o { x y } }", probe))
    assert stock == ["side"]
    assert compiled == stock


def test_foreign_close_is_not_called() -> None:
    closed: list[bool] = []

    class Handle:  # an awaitable with its own close(): stock never calls it
        def __await__(self):
            return later(1).__await__()

        def close(self):
            closed.append(True)

    obj = GraphQLObjectType("O2", {"x": GraphQLField(GraphQLInt, resolve=lambda s, i: Handle()),
                                   "y": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=lambda s, i: None)})
    schema = GraphQLSchema(GraphQLObjectType("Query", {"o": GraphQLField(obj, resolve=lambda s, i: {})}))

    def probe():
        count = len(closed)
        closed.clear()
        return count

    stock, compiled = asyncio.run(_run_both(schema, "{ o { x y } }", probe))
    assert stock == 0
    assert compiled == stock


def _chain_schema(resolver) -> GraphQLSchema:
    node = GraphQLObjectType("Node", lambda: {
        "val": GraphQLField(GraphQLInt),
        "child": GraphQLField(node, resolve=resolver),
        "kids": GraphQLField(GraphQLList(node), resolve=lambda s, i: [resolver(s, i), resolver(s, i)]),
    })
    return GraphQLSchema(GraphQLObjectType("Query", {
        "root": GraphQLField(node, resolve=lambda s, i: {"level": 0, "val": 0})}))


def _chain_query(depth: int) -> str:
    return "{ root { val " + "child { val kids { val } " * depth + "}" * depth + " } }"


async def child_async(src, info):
    await asyncio.sleep(0)
    level = src["level"] + 1
    return None if level == 7 else {"level": level, "val": level}  # a null deep down


def child_sync(src, info):
    return {"level": src["level"] + 1, "val": src["level"] + 1}


@pytest.mark.parametrize("resolver", [child_async, child_sync], ids=["async", "sync"])
def test_deep_nesting_of_custom_resolvers(resolver) -> None:
    """Past CONT_DEPTH_MAX nested custom resolvers, the awaited subtree goes to stock."""
    check(_chain_schema(resolver), _chain_query(9))


def test_generated_code_grows_linearly_with_depth() -> None:
    """Each nested custom resolver is emitted inline and as a continuation; without the
    depth cap the code would double per level."""
    schema = _chain_schema(child_sync)
    sizes = {}
    for depth in (9, 15):
        leftovers.clear_cache()
        asyncio.run(settle(execute_kw(schema, _chain_query(depth), CompiledExecutor)))
        sizes[depth] = leftovers.STATS["last_lines"]
    assert sizes[15] < 2 * sizes[9]


def test_deep_list_nesting_compiles() -> None:
    """Python allows 20 nested blocks per function; deep completions move into helpers."""
    depth = 30
    node = GraphQLObjectType("Deep", lambda: {"v": GraphQLField(GraphQLInt),
                                              "next": GraphQLField(GraphQLList(node))})

    def chain(level: int):
        return {"v": level, "next": [chain(level + 1)] if level < depth else []}

    schema = GraphQLSchema(GraphQLObjectType("Query", {"d": GraphQLField(GraphQLList(node),
                                                                         resolve=lambda s, i: [chain(0)])}))
    query = "{ d { v " + "next { v " * depth + "}" * depth + " } }"
    check(schema, query)


def test_discarded_pending_list_keeps_its_error_handling() -> None:
    """A sibling's synchronous non-null failure discards a still-pending list. Stock has
    already wrapped that list in its field's error handler, so an error raised inside it
    is still recorded (here before the root records the failure, while the root waits for
    the cancelled `deep` sibling)."""

    def boom(src, info):
        raise ValueError("boom")

    async def resolve_later(src, info):
        await asyncio.sleep(0)
        return {}

    async def deep_child(src, info):
        for _ in range(3):
            await asyncio.sleep(0)
        return {}

    item = GraphQLObjectType("Item", {"x": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=boom)})
    obj = GraphQLObjectType("Obj", {
        "items": GraphQLField(GraphQLList(GraphQLNonNull(item)), resolve=lambda s, i: [later({})]),
        "bad": GraphQLField(GraphQLNonNull(GraphQLInt), resolve=lambda s, i: "x7"),
    })
    deep = GraphQLObjectType("D", lambda: {"a": GraphQLField(deep, resolve=deep_child),
                                           "b": GraphQLField(deep, resolve=deep_child),
                                           "v": GraphQLField(GraphQLInt, resolve=lambda s, i: 1)})
    schema = GraphQLSchema(GraphQLObjectType("Query", {
        "o": GraphQLField(GraphQLNonNull(obj), resolve=resolve_later),
        "deep": GraphQLField(deep, resolve=deep_child),
    }))
    stock, compiled, used = asyncio.run(run_async(schema, "{ o { items { x } bad } deep { a { v } b { v } } }"))
    assert '"boom"' in stock[1]
    assert compiled == stock
    assert used
