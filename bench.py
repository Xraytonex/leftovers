"""Benchmark: stock graphql-core against leftovers on synthetic data.

    python bench.py [--items 5000] [--repeats 9]

Workload: a list of N objects, each with four scalars, four nested objects and a list of
two more, read by graphql-core's default resolver. Two runs: execute_sync, and async
execute with an async root resolver. Each timing is the median of --repeats runs after
two warm-up runs, with gc.collect() before every run and stock and leftovers alternating.
The script first checks that both return identical JSON (data and errors).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

from graphql import (
    GraphQLField,
    GraphQLInt,
    GraphQLList,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    execute,
    execute_sync,
    parse,
)

from leftovers import STATS, CompiledExecutor

QUERY = """{ items {
    id name lastname age
    address { street number }
    job { id org_name }
    partner { id name }
    pets { name type }
    school { id name }
} }"""


class Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def make_rows(n: int) -> list[Obj]:
    return [
        Obj(id=i, name=f"name{i}", lastname=f"last{i}", age=20 + i % 50,
            address=Obj(street=f"street {i}", number=i % 300),
            job=Obj(id=i * 7, org_name=f"org{i % 97}"),
            partner=Obj(id=i + 1, name=f"partner{i}"),
            pets=[Obj(name=f"pet{i}a", type="dog"), Obj(name=f"pet{i}b", type="cat")],
            school=Obj(id=i % 40, name=f"school{i % 40}"))
        for i in range(n)
    ]


def make_schema(rows: list[Obj], async_root: bool) -> GraphQLSchema:
    def obj(name: str, fields: dict) -> GraphQLObjectType:
        return GraphQLObjectType(name, lambda: {k: GraphQLField(v) for k, v in fields.items()})

    pair = {"id": GraphQLInt, "name": GraphQLString}
    item = obj("Item", {
        "id": GraphQLInt, "name": GraphQLString, "lastname": GraphQLString, "age": GraphQLInt,
        "address": obj("Address", {"street": GraphQLString, "number": GraphQLInt}),
        "job": obj("Job", {"id": GraphQLInt, "org_name": GraphQLString}),
        "partner": obj("Partner", pair),
        "pets": GraphQLList(obj("Pet", {"name": GraphQLString, "type": GraphQLString})),
        "school": obj("School", pair),
    })

    async def items_async(_root: Any, _info: Any) -> list[Obj]:
        await asyncio.sleep(0)
        return rows

    resolve = items_async if async_root else (lambda _root, _info: rows)
    return GraphQLSchema(GraphQLObjectType("Query", {"items": GraphQLField(GraphQLList(item), resolve=resolve)}))


def as_json(result: Any) -> str:
    return json.dumps([result.data, [e.formatted for e in result.errors or []]])


def measure(runs: dict[str, Callable[[], Any]], repeats: int) -> dict[str, float]:
    """Median seconds per run for each callable, alternating between them."""
    for fn in runs.values():
        fn()
        fn()
    times: dict[str, list[float]] = {name: [] for name in runs}
    for _ in range(repeats):
        for name, fn in runs.items():
            gc.collect()
            t0 = time.perf_counter()
            fn()
            times[name].append(time.perf_counter() - t0)
    return {name: statistics.median(ts) for name, ts in times.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    rows = make_rows(args.items)
    doc = parse(QUERY)
    sync_schema, async_schema = make_schema(rows, False), make_schema(rows, True)
    loop = asyncio.new_event_loop()

    def run_async(cls: Any) -> Any:
        kwargs = {"executor_class": cls} if cls else {}
        return loop.run_until_complete(execute(async_schema, doc, **kwargs))

    setups = {
        "execute_sync": (lambda: execute_sync(sync_schema, doc),
                         lambda: execute_sync(sync_schema, doc, executor_class=CompiledExecutor)),
        "execute, async root": (lambda: run_async(None), lambda: run_async(CompiledExecutor)),
    }

    print(f"leftovers {version('leftovers')}, graphql-core {version('graphql-core')}, "
          f"Python {platform.python_version()} on {platform.machine()}")
    print(f"{args.items:,} items; median of {args.repeats} runs\n")
    print(f"{'':<22}{'stock':>10}{'leftovers':>12}{'ratio':>8}  identical")
    for label, (stock, compiled) in setups.items():
        before = STATS["compiled_runs"]
        identical = as_json(stock()) == as_json(compiled())
        if STATS["compiled_runs"] == before:
            raise SystemExit(f"{label}: leftovers fell back to the stock executor")
        fallbacks = STATS["fallback_runs"]
        med = measure({"stock": stock, "leftovers": compiled}, args.repeats)
        if STATS["fallback_runs"] != fallbacks:
            raise SystemExit(f"{label}: leftovers fell back to the stock executor while timing")
        print(f"{label:<22}{med['stock'] * 1e3:>8.1f}ms{med['leftovers'] * 1e3:>10.1f}ms"
              f"{med['stock'] / med['leftovers']:>7.1f}x  {'yes' if identical else 'NO'}")
    loop.close()


if __name__ == "__main__":
    main()
