"""Ariadne integration: graphql_sync, async graphql and the ASGI app, stock vs leftovers."""

from __future__ import annotations

import asyncio
import json

import pytest

ariadne = pytest.importorskip("ariadne")
httpx = pytest.importorskip("httpx")
from ariadne import ObjectType, QueryType, make_executable_schema  # noqa: E402
from ariadne.asgi import GraphQL  # noqa: E402

from leftovers import STATS, CompiledExecutor  # noqa: E402

SDL = """
type Query { books: [Book!]! slow: [Book] }
type Book { id: Int! title: String price: Float tags: [String!] author: Author }
type Author { name: String! books: [Book!] }
"""
BOOKS = [{"id": i, "title": None if i == 3 else f"t{i}", "price": i * 1.5, "tags": ["a", "b"][: i % 3],
          "author": {"name": None if i == 4 else f"a{i}", "books": []}} for i in range(6)]
QUERY = "{ books { id title price tags author { name } } slow { id author { name } } }"


def schema():
    query, book = QueryType(), ObjectType("Book")
    query.set_field("books", lambda *_: BOOKS)

    async def slow(*_):
        await asyncio.sleep(0)
        return BOOKS[:3]

    async def author(obj, info):  # async per-item resolver
        await asyncio.sleep(0)
        return obj["author"]

    query.set_field("slow", slow)
    book.set_field("author", author)
    return make_executable_schema(SDL, query, book)


def test_graphql_sync_identical() -> None:
    query = QueryType()
    query.set_field("books", lambda *_: BOOKS)
    sync_schema = make_executable_schema(SDL, query)  # default resolvers below the root
    q = "{ books { id title price tags author { name books { id } } } }"
    before = STATS["compiled_runs"]
    stock = ariadne.graphql_sync(sync_schema, {"query": q})[1]
    compiled = ariadne.graphql_sync(sync_schema, {"query": q}, execution_context_class=CompiledExecutor)[1]
    assert json.dumps(compiled) == json.dumps(stock)
    assert stock.get("errors")
    assert STATS["compiled_runs"] > before


def test_async_graphql_identical() -> None:
    s = schema()

    async def run(cls):
        kw = {"execution_context_class": cls} if cls else {}
        return (await ariadne.graphql(s, {"query": QUERY}, **kw))[1]

    before = STATS["compiled_runs"]
    stock, compiled = asyncio.run(run(None)), asyncio.run(run(CompiledExecutor))
    assert json.dumps(compiled) == json.dumps(stock)
    assert stock.get("errors")  # the fixture has nulls in non-null fields
    assert STATS["compiled_runs"] > before


def test_asgi_identical() -> None:
    s = schema()

    async def post(cls):
        kw = {"execution_context_class": cls} if cls else {}
        app = GraphQL(s, **kw)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            resp = await client.post("/", json={"query": QUERY})
            return resp.status_code, resp.content

    assert asyncio.run(post(CompiledExecutor)) == asyncio.run(post(None))
