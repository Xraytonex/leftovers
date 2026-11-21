"""Compile cache and the executor classes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from graphql.execution.execute import assume_not_awaitable
from graphql.execution.executor import Executor
from graphql.execution.executor_throwing_on_incremental import (
    ExecutorThrowingOnIncremental,
)
from graphql.type import GraphQLObjectType

from ._codegen import Unsupported, compile_operation
from ._runtime import FALLBACK, STATS

CACHE_MAX = 512
_CACHE: OrderedDict[tuple, tuple[Any, Any, Any, Any]] = OrderedDict()


def clear_cache() -> None:
    """Drop every compiled operation (e.g. after mutating a schema, see the contract)."""
    _CACHE.clear()


def _cache_key(ex: Executor, root_type: GraphQLObjectType, aware: bool) -> tuple:
    """Key on the query *text* (servers re-parse every request, so AST identity would
    miss every time) plus everything codegen depends on. Contract: the schema is not
    mutated after its first execution (resolvers, is_type_of and coercers are baked in)."""
    op = ex.operation
    loc = op.loc
    if loc is not None and loc.source is not None:
        doc = ("src", loc.source.body, op.name.value if op.name else None)
    else:
        doc = ("ast", id(op))
    return (id(ex.schema), id(root_type), id(ex.field_resolver), aware, doc)


def get_compiled(ex: Executor, root_type: GraphQLObjectType, aware: bool):
    key = _cache_key(ex, root_type, aware)
    hit = _CACHE.get(key)
    # identity checks on objects the entry keeps alive, so a recycled id() never matches
    if (hit is not None and hit[0] is ex.schema and hit[3] is ex.field_resolver
            and (key[4][0] == "src" or hit[1] is ex.operation)):
        _CACHE.move_to_end(key)
        return hit[2]
    try:
        fn, _code = compile_operation(ex, root_type, aware)
        STATS["compiles"] += 1
    except (Unsupported, SyntaxError, RecursionError):
        fn = None
        STATS["unsupported"] += 1
    _CACHE[key] = (ex.schema, ex.operation, fn, ex.field_resolver)
    if len(_CACHE) > CACHE_MAX:  # bounded LRU: no unbounded growth
        _CACHE.popitem(last=False)
    return fn


def generated_source(ex: Executor, root_type: GraphQLObjectType, aware: bool = False) -> str:
    return compile_operation(ex, root_type, aware)[1]


class CompiledMixin:
    """Root-execution override; put it in front of any graphql-core 3.3 Executor subclass
    (e.g. Strawberry's StrawberryGraphQLCoreExecutionContext, an IncrementalExecutor)."""

    def execute_root_grouped_field_set(  # noqa: PLR0913
        self, root_type, root_value, grouped_field_set, serially, position_context
    ):
        mm = self.middleware_manager
        if not serially and position_context is None and (mm is None or not mm.middlewares):
            aware = self.is_awaitable is not assume_not_awaitable
            fn = get_compiled(self, root_type, aware)
            if fn is not None:
                STATS["compiled_runs"] += 1  # counted before the call: fn may raise (root null)
                result = fn(self, root_value)
                if result is not FALLBACK:
                    return result
                STATS["compiled_runs"] -= 1
        STATS["fallback_runs"] += 1
        return super().execute_root_grouped_field_set(
            root_type, root_value, grouped_field_set, serially, position_context
        )


class CompiledExecutor(CompiledMixin, ExecutorThrowingOnIncremental):
    """Drop-in for graphql-core's default executor class."""
