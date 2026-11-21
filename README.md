# leftovers

An executor for graphql-core 3.3 that compiles each query once into plain Python and reheats it on every request.

## Install

leftovers is not on PyPI. It needs Python 3.10 or later and graphql-core 3.3. Install it from a checkout of this repository:
```bash
pip install .
```

## Usage

With graphql-core, pass `executor_class=CompiledExecutor`. `execute_sync` and `execute` take the same argument.
```python
import asyncio

from graphql import build_schema, graphql, graphql_sync
from leftovers import CompiledExecutor

schema = build_schema("type Query { books: [Book!]! } type Book { title: String! }")
root = {"books": [{"title": "Dune"}, {"title": "Emma"}]}
query = "{ books { title } }"

result = graphql_sync(schema, query, root, executor_class=CompiledExecutor)
result = asyncio.run(graphql(schema, query, root, executor_class=CompiledExecutor))
```

With Ariadne, pass `execution_context_class=CompiledExecutor`.
```python
import asyncio

from ariadne import QueryType, graphql, graphql_sync, make_executable_schema
from ariadne.asgi import GraphQL
from leftovers import CompiledExecutor

query = QueryType()
query.set_field("books", lambda *_: [{"title": "Dune"}, {"title": "Emma"}])
schema = make_executable_schema("type Query { books: [Book!]! } type Book { title: String! }", query)
data = {"query": "{ books { title } }"}

success, result = graphql_sync(schema, data, execution_context_class=CompiledExecutor)
success, result = asyncio.run(graphql(schema, data, execution_context_class=CompiledExecutor))
app = GraphQL(schema, execution_context_class=CompiledExecutor)
```

## What to know

- Only graphql-core 3.3 is supported. Graphene uses graphql-core 3.2, so it isn't supported yet.
- Some operations run entirely on the stock executor: mutations, operations with middleware, any selection of a field of interface or union type, `@defer` and `@stream`, `@skip` or `@include` with a variable condition, `__schema` and `__type`, fragment arguments, and operations that would need more than 20,000 lines of generated code. `leftovers.STATS` counts compiled runs and fallbacks.
- Compiled code is cached by schema and query text (the last 512 operations), with the schema's resolvers built in. Don't change a schema after its first execution, or call `leftovers.clear_cache()` after you do.
- The output is the same as stock graphql-core's: the same data, and the same errors in the same order. The tests check this against stock, sync and async: hand-written cases cover arguments, variables, fragments and aliases, and Hypothesis differential fuzzing runs random schemas and queries with several faults per query (nulls, raised and returned exceptions, wrong types, delayed awaitables).

## Benchmark

`python bench.py` on an Apple M3 Pro, Python 3.13, graphql-core 3.3.2: a list of 5,000 objects with nested objects and lists, on default resolvers, took 218 ms with stock `execute_sync` and 18 ms with leftovers (11.9x), and 252 ms against 27 ms (9.4x) through `execute` with an async root resolver.

## License

MIT. See [LICENSE](LICENSE).
