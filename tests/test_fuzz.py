"""Hypothesis differential fuzzing: leftovers must return exactly what stock graphql-core returns.

Compared: `data` (JSON) and `errors` (message, path, locations) in order. Sync worlds run
execute_sync; async worlds run execute and await. Faults are independent per position, so
most failing examples have several (multi-fault). More examples: LEFTOVERS_FUZZ_EXAMPLES=5000.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import json
import os
import random
import warnings

import pytest
from graphql import execute, execute_sync, parse
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import leftovers
from leftovers import STATS, CompiledExecutor
from tests.fuzz_world import World

EXAMPLES = int(os.environ.get("LEFTOVERS_FUZZ_EXAMPLES", "200"))
SETTINGS = settings(max_examples=EXAMPLES, deadline=None, derandomize=False,
                    suppress_health_check=[HealthCheck.too_slow])
seeds = st.integers(min_value=0, max_value=2**31 - 1)
fault_rates = st.sampled_from([0.0, 0.1, 0.25, 0.5])


def normalize(result) -> tuple:
    errors = [(e.message, e.path, [(loc.line, loc.column) for loc in e.locations or []])
              for e in result.errors or []]
    return json.dumps(result.data), errors


def compare(stock, compiled, query: str, used: bool) -> None:
    # The only allowed fallback: the operation needed more than MAX_LINES of generated
    # code (a budget by design); results must still match.
    over_budget = STATS["last_lines"] > leftovers.MAX_LINES
    assert used or over_budget, f"compiled path not used for {query}"
    s, c = normalize(stock), normalize(compiled)
    assert s[0] == c[0], f"data differs for {query}\nstock     {s[0]}\nleftovers {c[0]}"
    assert s[1] == c[1], f"errors differ for {query}\nstock     {s[1]}\nleftovers {c[1]}"


@pytest.mark.fuzz
@SETTINGS
@given(seed=seeds, fault_rate=fault_rates)
def test_fuzz_sync(seed: int, fault_rate: float) -> None:
    world = World(seed, fault_rate, async_rate=0.0)
    query = world.query(random.Random(seed + 1))
    doc = parse(query)
    stock = execute_sync(world.schema, doc, world.root())
    before = STATS["compiled_runs"]
    compiled = execute_sync(world.schema, doc, world.root(), executor_class=CompiledExecutor)
    compare(stock, compiled, query, STATS["compiled_runs"] > before)


async def run_async(world: World, doc, cls):
    result = execute(world.schema, doc, world.root(), **({"executor_class": cls} if cls else {}))
    if inspect.isawaitable(result):
        result = await result
    for _ in range(4):  # let background settling finish, as a server's loop would
        await asyncio.sleep(0)
    return result


@pytest.mark.fuzz
@SETTINGS
@given(seed=seeds, fault_rate=fault_rates, async_rate=st.sampled_from([0.2, 0.5, 0.9]))
def test_fuzz_async(seed: int, fault_rate: float, async_rate: float) -> None:
    world = World(seed, fault_rate, async_rate)
    query = world.query(random.Random(seed + 1))
    doc = parse(query)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # both sides may orphan coroutines alike
        stock = asyncio.run(run_async(world, doc, None))
        before = STATS["compiled_runs"]
        compiled = asyncio.run(run_async(world, doc, CompiledExecutor))
        gc.collect()  # finalize orphaned coroutines here, not at interpreter shutdown
    compare(stock, compiled, query, STATS["compiled_runs"] > before)
