"""A random GraphQL world for differential fuzzing: schema, query and lazily produced data.

Every value is a pure function of (seed, position), produced fresh on each access, so stock
and leftovers see the same data and the same faults, and every run gets fresh coroutines. Faults
are drawn independently per position (multi-fault): null, raising resolver or attribute,
returned Exception, wrong-typed or out-of-range value, non-iterable list, Exception or
awaitable list items, and, in async worlds, awaitables with 0-2 ticks of delay that may
raise, plus async iterables.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from typing import Any

from graphql import (
    GraphQLBoolean,
    GraphQLField,
    GraphQLFloat,
    GraphQLInt,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)

SCALARS = {
    "Int": (GraphQLInt, lambda r: r.randint(-3, 3)),
    "String": (GraphQLString, lambda r: r.choice(["a", "bb", ""])),
    "Float": (GraphQLFloat, lambda r: r.choice([0.5, -2.0, 3.25])),
    "Boolean": (GraphQLBoolean, lambda r: r.random() < 0.5),
}
class Opaque:
    """A value no scalar accepts, with a stable repr (error messages must not contain ids)."""

    def __repr__(self) -> str:
        return "<opaque>"


WRONG = [lambda r: Opaque(), lambda r: "x7", lambda r: 2**40, lambda r: float("nan"), lambda r: 1.5,
         lambda r: True, lambda r: {"k": 1}, lambda r: [1, 2]]


class Raise:
    """Marker: producing this value raises instead."""

    def __init__(self, msg: str) -> None:
        self.msg = msg


async def later(ticks: int, value: Any) -> Any:
    for _ in range(ticks):
        await asyncio.sleep(0)
    if isinstance(value, Raise):
        raise ValueError(value.msg)
    return value


async def agen(items: list) -> Any:
    for item in items:
        await asyncio.sleep(0)
        yield item


class World:
    def __init__(self, seed: int, fault_rate: float, async_rate: float) -> None:
        r = random.Random(seed)
        self.seed, self.fault_rate, self.async_rate = seed, fault_rate, async_rate
        names = [f"T{i}" for i in range(r.randint(1, 3))]
        self.spec = {n: [(f"f{j}", self.rand_type(r, names, 0), r.random() < 0.35)
                         for j in range(r.randint(1, 4))] for n in names}
        self.types = {n: GraphQLObjectType(n, self._fields(n)) for n in names}
        roots = {}
        for k in range(r.randint(1, 2)):
            t = self.rand_type(r, names, 0, root=True)
            roots[f"q{k}"] = (t, k == 0)  # q0 custom resolver, q1 default resolver on the root
        self.roots = roots
        self.schema = GraphQLSchema(GraphQLObjectType("Query", {
            name: GraphQLField(self.build(t), resolve=self.resolver(t) if custom else None)
            for name, (t, custom) in roots.items()}))

    # -- types
    def rand_type(self, r: random.Random, names: list, depth: int, root: bool = False) -> tuple:
        base = ("O", r.choice(names)) if root or r.random() < 0.4 else ("S", r.choice(list(SCALARS)))
        t = base
        if depth < 2 and r.random() < 0.35:
            t = ("L", ("N", t) if r.random() < 0.4 else t)
        if r.random() < 0.4:
            t = ("N", t)
        return t

    def build(self, t: tuple) -> Any:
        kind, arg = t
        if kind == "S":
            return SCALARS[arg][0]
        if kind == "O":
            return self.types[arg]
        if kind == "L":
            return GraphQLList(self.build(arg))
        return GraphQLNonNull(self.build(arg))

    def _fields(self, name: str):
        return lambda: {fname: GraphQLField(self.build(t), resolve=self.resolver(t) if custom else None)
                        for fname, t, custom in self.spec[name]}

    # -- queries
    def query(self, r: random.Random) -> str:
        parts = []
        for name, (t, _custom) in self.roots.items():
            parts.append(f"{name}{self.selection(t, r, 0)}")
            if r.random() < 0.2:
                parts.append(f"al_{name}: {name}{self.selection(t, r, 0)}")
        return "{ " + " ".join(parts) + " }"

    def selection(self, t: tuple, r: random.Random, depth: int) -> str:
        while t[0] in ("N", "L"):
            t = t[1]
        if t[0] == "S":
            return ""
        fields = self.spec[t[1]]
        picked = [f for f in fields if r.random() < 0.7] or fields[:1]
        out = []
        for fname, ft, _custom in picked:
            leaf = ft
            while leaf[0] in ("N", "L"):
                leaf = leaf[1]
            if leaf[0] == "O" and depth >= 3:
                continue
            out.append(f"{fname}{self.selection(ft, r, depth + 1)}")
            if r.random() < 0.15:
                out.append(f"a_{fname}: {fname}{self.selection(ft, r, depth + 1)}")
        if not out or r.random() < 0.15:
            out.append("__typename")
        return " { " + " ".join(out) + " }"

    # -- data
    def rng(self, path: list) -> random.Random:
        return random.Random(f"{self.seed}:{'/'.join(map(str, path))}")

    def produce(self, t: tuple, path: list, allow_async: bool = True) -> Any:
        """The value a resolver/attribute/list item yields at `path` (fresh each call)."""
        r = self.rng(path)
        if r.random() < self.fault_rate:
            v = self.fault(t, path, r)
        else:
            v = self.normal(t, path, r)
        if allow_async and r.random() < self.async_rate:
            return later(r.choice([0, 0, 1, 2]), v)
        return Raise(v.msg) if isinstance(v, Raise) else v

    def normal(self, t: tuple, path: list, r: random.Random) -> Any:
        kind, arg = t
        if kind == "N":
            return self.normal(arg, path, r)
        if kind == "S":
            return SCALARS[arg][1](r)
        if kind == "O":
            return Node(self, arg, path) if r.random() < 0.7 else DictNode(self, arg, path)
        items = [self.item(arg, [*path, i]) for i in range(r.choice([0, 1, 2, 3]))]
        if self.async_rate and r.random() < 0.1:
            return agen(items)
        return tuple(items) if r.random() < 0.1 else items

    def fault(self, t: tuple, path: list, r: random.Random) -> Any:
        choice = r.random()
        if choice < 0.3:
            return None
        if choice < 0.5:
            return Raise(f"boom {'/'.join(map(str, path))}")
        if choice < 0.6:
            return ValueError(f"returned {'/'.join(map(str, path))}")
        if t[0] == "L" or (t[0] == "N" and t[1][0] == "L"):
            return r.choice([5, "abc", [None, ValueError("item")]])
        return r.choice(WRONG)(r)

    def item(self, t: tuple, path: list) -> Any:
        """A list item: a raising position becomes an Exception item (stock raises it)."""
        v = self.produce(t, path)
        return ValueError(v.msg) if isinstance(v, Raise) else v

    def resolver(self, t: tuple):
        def resolve(src: Any, info: Any) -> Any:
            v = self.produce(t, ["R", *info.path.as_list()])
            if isinstance(v, Raise):
                raise ValueError(v.msg)
            return v
        return resolve

    def root(self) -> Any:
        return Node(self, None, [])


class Node:
    """An object position read by the default resolver through attributes."""

    def __init__(self, world: World, tname: str | None, path: list) -> None:
        self._w, self._t, self._p = world, tname, path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        w, path = self._w, [*self._p, name]
        t = w.roots[name][0] if self._t is None else dict((f, ft) for f, ft, _c in w.spec[self._t])[name]
        v = w.produce(t, path)
        if isinstance(v, Raise):
            raise ValueError(v.msg)
        if w.rng([*path, "callable"]).random() < 0.1:
            return lambda info, **kwargs: v  # the default resolver calls callables
        return v


class DictNode(Mapping):
    """An object position read by the default resolver through Mapping.get."""

    def __init__(self, world: World, tname: str, path: list) -> None:
        self._node = Node(world, tname, path)
        self._keys = [f for f, _t, _c in world.spec[tname]]

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self._node, key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
