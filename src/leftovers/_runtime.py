"""Runtime helpers called by the generated code (sync and awaitable-aware modes)."""

from __future__ import annotations

from typing import Any

from graphql import located_error
from graphql.execution.executor import Executor, collect_iterator_awaitables
from graphql.pyutils import Path, Undefined, gather_with_cancel
from graphql.type import GraphQLObjectType

FALLBACK = object()
STATS = {"compiled_runs": 0, "fallback_runs": 0, "compiles": 0, "unsupported": 0, "last_lines": 0}


def _mk(lp: tuple | None) -> Path | None:
    return None if lp is None else Path(_mk(lp[0]), lp[1], lp[2])


def _handle(ex: Executor, raw: Exception, nodes: list | None, lp: tuple, non_null: bool) -> None:
    """Mirror of Executor.handle_field_error, with the path built lazily."""
    path = _mk(lp)
    error = located_error(raw, nodes, path.as_list())
    if ex.error_propagation and non_null:
        raise error
    ex.collected_errors.add(error, path)


def _info(ex: Executor, field_def: Any, nodes: list, parent: GraphQLObjectType, lp: tuple):
    return ex.build_resolve_info(field_def, nodes, parent, _mk(lp))


def _leaf(return_type: Any, value: Any) -> Any:
    """complete_value for a nullable leaf: exception, null/undefined, then coercion."""
    if isinstance(value, Exception):
        raise value
    if value is None or value is Undefined:
        return None
    return Executor.complete_leaf_value(return_type, value)


class _A:
    """A pending position value (aware mode only).

    `wrapped`: the position's own error handling is inside `aw`. Resolver awaitables and
    stock's awaitable completions handle their own errors; a pending object or list, or
    a stock complete_value handoff, gets `_wrap` at its field or list item. Only the value
    a continuation returns stays unwrapped (`_cont` handles its errors). `done`: absorbed
    by its parent or handed to settle_in_background. Every _A is registered in `pend`,
    so code can detect new ones with len(pend)."""

    __slots__ = ("aw", "done", "wrapped")

    def __init__(self, aw: Any, pend: list, wrapped: bool) -> None:
        self.aw, self.wrapped, self.done = aw, wrapped, False
        pend.append(self)


def _drop_since(ex: Executor, pend: list, n: int) -> None:
    """A synchronous failure discards what was created since `n`: settle whatever is
    still loose in the background, like stock's execute_fields/complete_iterable_value
    excepts (it still runs with a loop running; coroutines are closed without one)."""
    loose = []
    for i in range(n, len(pend)):
        a = pend[i]
        if not a.done:
            a.done = True
            loose.append(a.aw)
    if loose:
        ex.settle_in_background(loose)


def _drain(ex: Executor, iterator: Any) -> None:
    """Stock's complete_iterable_value except: settle awaitables still in the iterator."""
    rest = collect_iterator_awaitables(iterator, ex.is_awaitable)
    if rest:
        ex.settle_in_background(rest)


async def _await_completed(ex: Executor, completed: Any, rtype: Any, fdl: Any, path: Path) -> Any:
    """Stock's await_completed (execute_field / complete_list_item_value)."""
    try:
        return await completed
    except Exception as error:
        ex.handle_field_error(error, rtype, fdl, path)
        return None


def _wrap(ex: Executor, a: _A, rtype: Any, fdl: Any, lp: tuple) -> None:
    """Give a pending object or list its position's error handling right away, as stock's
    execute_field and complete_list_item_value do, so it still applies when a synchronous
    failure discards the position before its parent absorbs it."""
    a.aw = _await_completed(ex, a.aw, rtype, fdl, _mk(lp))
    a.wrapped = True


async def _gather_into(container: Any, keys: list, aws: list) -> Any:
    """Stock's get_results / get_completed_results."""
    if len(aws) == 1:  # one awaitable: awaited directly, as stock does
        container[keys[0]] = await aws[0]
    else:
        for key, value in zip(keys, await gather_with_cancel(*aws), strict=True):
            container[key] = value
    return container


def _obj_pend(d: dict, pend: list) -> Any:
    """Tail of stock's execute_fields: an object with pending fields becomes one awaitable."""
    keys, aws = [], []
    for key, val in d.items():
        if type(val) is _A:  # wrapped: every field value _A carries its own error handling
            val.done = True
            keys.append(key)
            aws.append(val.aw)
    return _A(_gather_into(d, keys, aws), pend, False) if keys else d


def _list_pend(lst: list, pend: list) -> Any:
    """Tail of stock's complete_iterable_value for a list with pending items."""
    keys, aws = [], []
    for i, val in enumerate(lst):
        if type(val) is _A:  # wrapped, like field values
            val.done = True
            keys.append(i)
            aws.append(val.aw)
    return _A(_gather_into(lst, keys, aws), pend, False) if keys else lst


def _stock(ex: Executor, pend: list, inner: Any, fdl: Any, info: Any, path: Path, value: Any) -> Any:
    """Complete one position with stock's complete_value (async is_type_of, async
    iterables); an awaitable result is an unwrapped pending position."""
    completed = ex.complete_value(inner, fdl, info, path, value, None)
    return _A(completed, pend, False) if ex.is_awaitable(completed) else completed


async def _cont(ex: Executor, aw: Any, flp: tuple, k_fn: Any, nodes: list, non_null: bool) -> Any:
    """Stock's complete_awaitable_value, with the subtree completed in compiled code."""
    pend: list = []
    synced = False
    try:
        value = await ex.with_abort_signal(aw)
        out = k_fn(ex, value, flp, pend)
        synced = True
        if type(out) is _A:
            out.done = True
            out = await out.aw
    except Exception as error:
        if not synced:
            _drop_since(ex, pend, 0)
        _handle(ex, error, nodes, flp, non_null)
        return None
    return out
