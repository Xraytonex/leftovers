"""leftovers: a compiling executor for graphql-core 3.3 with the same output as stock.

Usage: execute_sync(schema, document, ..., executor_class=CompiledExecutor), or
execute(...) for async. For another graphql-core 3.3 Executor subclass, put CompiledMixin
in front of it.

For each (schema, operation, mode) it emits one straight-line Python function that
resolves and completes the whole query, with loops and nested objects inlined. Paths are
kept as lazy tuples and turned into graphql-core `Path` objects only when an error or a
resolver needs them.

Two modes, picked per request from the executor's `is_awaitable`:
- sync (`assume_not_awaitable`): no awaitable checks at all.
- aware (anything else, e.g. async `execute`): it mirrors stock's async structure
  exactly. Wherever stock calls `is_awaitable`, so does the compiled code (except on
  values whose exact type is already known: str, int, float, bool, list). A pending
  position is an `_A`. As in stock, each object with pending fields becomes one awaitable
  that awaits a single field directly or gathers several with gather_with_cancel
  (execute_fields), each list likewise (complete_iterable_value), and a pending object or
  list under a field or item is wrapped in that position's own error handler
  (await_completed). An awaitable from a resolver is awaited and its subtree continues in
  compiled code (a continuation, = complete_awaitable_value); any other awaitable goes to
  stock's own completion at that path. A synchronous failure hands everything it orphans
  to stock's settle_in_background, as stock does.

Anything the generator cannot prove equivalent falls back to the stock executor:
middleware, abstract types, @defer/@stream, variable-dependent @skip/@include,
introspection roots, fragment arguments, mutations (serial execution).
"""

from ._codegen import CONT_DEPTH_MAX, MAX_LINES
from ._runtime import STATS
from .executor import CompiledExecutor, CompiledMixin, clear_cache, generated_source

__all__ = [
    "CONT_DEPTH_MAX",
    "MAX_LINES",
    "STATS",
    "CompiledExecutor",
    "CompiledMixin",
    "clear_cache",
    "generated_source",
]
__version__ = "0.1.0"
