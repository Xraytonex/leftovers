"""Schemas and fault-injected data shared by the hand-written differential tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    execute_sync,
    parse,
)

from leftovers import STATS, CompiledExecutor


class O:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class RaisingAge(O):
    @property
    def age(self):
        raise ValueError("age exploded")


class RaisingPetName(O):
    @property
    def name(self):
        raise KeyError("pet name missing")


class Custom:
    def __str__(self) -> str:
        return "custom-str"


def pet(n: int) -> O:
    return O(name=f"pet{n}", type="dog")


async def later(value: Any) -> Any:
    await asyncio.sleep(0)
    return value


async def boom(msg: str) -> Any:
    await asyncio.sleep(0)
    raise ValueError(msg)


school = O(id=1, name="MIT")
people: list[Any] = [
    O(id=0, name="ok", nn_name="a", age=30, tags=["x", "y"], pets=[pet(0)], school=school),
    {"id": 1, "name": "dict", "nn_name": "b", "age": 31, "tags": [], "pets": [], "school": school},
    RaisingAge(id=2, name="raises", nn_name="c", tags=None, pets=None, school=None),
    O(id=3, name=123, nn_name="d", age="abc", tags=None, pets=[], school=school),
    O(id=4, name="float age", nn_name="e", age=3.5, tags=["a", 1, True, None], pets=[], school=school),
    O(id=5, name=True, nn_name="f", age=2**40, tags=None, pets=[], school=school),
    O(id=6, name="callable school", nn_name="g", age=1, tags=None, pets=5,
      school=lambda info: O(id=9, name=f"via {info.field_name}")),
    O(id=7, name="str pets", nn_name="h", age=ValueError("boom"), tags=None, pets="abc", school=None),
    O(id=8, name="null nn_name", nn_name=None, age=8, tags=None, pets=[], school=school),
    O(id=9, name="bad pets", nn_name="i", age=9, tags=None,
      pets=[RaisingPetName(type="cat"), None, pet(9)], school=school),
    O(id=10, name="mapping pets", nn_name="j", age=10, tags=None, pets={"a": 1}, school=school),
    O(id=None, name="null id", nn_name="k", age=11, tags=None, pets=[], school=school),
    O(id=12, name="tuple pets", nn_name="l", age=12, tags=("t",), pets=(pet(12), pet(13)), school=None),
    O(id=13, name=object(), nn_name="m", age=13, tags=None, pets=[], school=school),
    O(id=14, name=Custom(), nn_name="n", age=14, tags=None, pets=[], school=school),
]
for p in people:
    strict = [pet(1), None] if getattr(p, "id", None) == 0 else [pet(2)]
    # a default-resolver attribute that is callable and returns an awaitable (stock awaits it)
    attr_coro = lambda info: later("from attr")  # noqa: E731
    if isinstance(p, dict):
        p["strict_pets"], p["attr_coro"] = [pet(3)], attr_coro
    else:
        p.strict_pets, p.attr_coro = strict, attr_coro

Pet = GraphQLObjectType("Pet", lambda: {"name": GraphQLField(GraphQLString), "type": GraphQLField(GraphQLString)})
School = GraphQLObjectType("School", lambda: {"id": GraphQLField(GraphQLInt), "name": GraphQLField(GraphQLString)})


def greet(src: Any, info: Any, who: str) -> str:
    name = src["name"] if isinstance(src, dict) else src.name
    return f"hi {who} from {name}"


Person = GraphQLObjectType(
    "Person",
    lambda: {
        "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
        "name": GraphQLField(GraphQLString),
        "nn_name": GraphQLField(GraphQLNonNull(GraphQLString)),
        "age": GraphQLField(GraphQLInt),
        "tags": GraphQLField(GraphQLList(GraphQLString)),
        "pets": GraphQLField(GraphQLList(Pet)),
        "strict_pets": GraphQLField(GraphQLList(GraphQLNonNull(Pet))),
        "school": GraphQLField(School),
        "greet": GraphQLField(GraphQLString, {"who": GraphQLArgument(GraphQLNonNull(GraphQLString))}, resolve=greet),
    },
)
sync_schema = GraphQLSchema(
    GraphQLObjectType(
        "Query",
        {
            "people": GraphQLField(GraphQLList(Person), resolve=lambda r, i: people),
            "strict": GraphQLField(GraphQLList(GraphQLNonNull(Person)), resolve=lambda r, i: people),
            "must": GraphQLField(GraphQLNonNull(Person), resolve=lambda r, i: people[8]),
        },
    )
)


class Loader:
    """DataLoader-style batching: keys from one loop tick become one batch call."""

    def __init__(self, batch: Any) -> None:
        self.batch, self.queue = batch, []
        self.loop = asyncio.get_running_loop()

    def load(self, key: Any) -> asyncio.Future:
        fut = self.loop.create_future()
        if not self.queue:
            self.loop.call_soon(self.dispatch)
        self.queue.append((key, fut))
        return fut

    def dispatch(self) -> None:
        queue, self.queue = self.queue, []
        for (_, fut), value in zip(queue, self.batch([k for k, _ in queue]), strict=True):
            fut.set_result(value)


def _pid(src: Any) -> Any:
    return src["id"] if isinstance(src, dict) else src.id


def make_async_schema() -> GraphQLSchema:
    loader: dict[str, Loader | None] = {"l": None}

    def school_loader() -> Loader:
        current = loader["l"]
        if current is None or current.loop is not asyncio.get_running_loop():
            current = loader["l"] = Loader(lambda keys: [O(id=k, name=f"school{k}") for k in keys])
        return current

    async def a_greet(src, info, who):
        await asyncio.sleep(0)
        name = src["name"] if isinstance(src, dict) else src.name
        return f"hi {who} from {name}"

    def a_nn(src, info):  # non-null field, async; None for person 8 -> bubbles to the item
        pid = _pid(src)
        return later(None if pid == 8 else f"nn{pid}")

    def a_raise(src, info):  # nullable field, async; raises for even ids
        pid = _pid(src)
        return boom(f"async boom {pid}") if pid is not None and pid % 2 == 0 else later(pid)

    def a_school(src, info):  # DataLoader-style per-item async object
        pid = _pid(src)
        return school_loader().load(pid if pid is not None else -1)

    def pets_of_coros(src, info):  # list whose items are awaitables
        return [later(pet(1)), later(None), later(pet(2))]

    def strict_coros(src, info):  # [Pet!] with a None item -> bubbles to the list field
        return [later(pet(1)), later(None)]

    async_person = GraphQLObjectType(
        "Person",
        lambda: {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
            "name": GraphQLField(GraphQLString),
            "nn_name": GraphQLField(GraphQLNonNull(GraphQLString)),
            "age": GraphQLField(GraphQLInt),
            "tags": GraphQLField(GraphQLList(GraphQLString)),
            "pets": GraphQLField(GraphQLList(Pet)),
            "school": GraphQLField(School),
            "greet": GraphQLField(GraphQLString, {"who": GraphQLArgument(GraphQLNonNull(GraphQLString))},
                                  resolve=a_greet),
            "a_nn": GraphQLField(GraphQLNonNull(GraphQLString), resolve=a_nn),
            "a_raise": GraphQLField(GraphQLInt, resolve=a_raise),
            "a_school": GraphQLField(School, resolve=a_school),
            "coro_pets": GraphQLField(GraphQLList(Pet), resolve=pets_of_coros),
            "strict_coros": GraphQLField(GraphQLList(GraphQLNonNull(Pet)), resolve=strict_coros),
            "attr_coro": GraphQLField(GraphQLString),  # default resolver; value is a callable
        },
    )

    async def root_people(r, i):
        await asyncio.sleep(0)
        return people

    async def root_must(r, i):
        await asyncio.sleep(0)
        return people[8]  # nn_name None -> bubbles through non-null `amust` to data: null

    return GraphQLSchema(GraphQLObjectType(
        "Query",
        {
            "people": GraphQLField(GraphQLList(async_person), resolve=lambda r, i: people),
            "apeople": GraphQLField(GraphQLList(async_person), resolve=root_people),
            "astrict": GraphQLField(GraphQLList(GraphQLNonNull(async_person)), resolve=root_people),
            "amust": GraphQLField(GraphQLNonNull(async_person), resolve=root_must),
        },
    ))


def as_json(result: Any) -> tuple[str, str]:
    """`data` and the formatted `errors`, in order, as JSON text."""
    return json.dumps(result.data), json.dumps([e.formatted for e in (result.errors or [])])


def run_sync(schema: GraphQLSchema, query: str, variables: dict | None = None) -> tuple[Any, Any, bool]:
    """(stock, compiled, whether the compiled path ran) for one execute_sync of `query`."""
    doc = parse(query)
    stock = execute_sync(schema, doc, variable_values=variables)
    before = STATS["compiled_runs"]
    compiled = execute_sync(schema, doc, variable_values=variables, executor_class=CompiledExecutor)
    return as_json(stock), as_json(compiled), STATS["compiled_runs"] > before


async def settle(result: Any) -> Any:
    return await result if hasattr(result, "__await__") else result


async def run_async(schema: GraphQLSchema, query: str, root: Any = None) -> tuple[Any, Any, bool]:
    """(stock, compiled, whether the compiled path ran) for one async execute of `query`.
    `root` is a factory, so each side gets a fresh root value."""
    doc = parse(query)
    stock = await settle(execute(schema, doc, root() if root else None))
    before = STATS["compiled_runs"]
    compiled = await settle(execute(schema, doc, root() if root else None, executor_class=CompiledExecutor))
    return as_json(stock), as_json(compiled), STATS["compiled_runs"] > before
