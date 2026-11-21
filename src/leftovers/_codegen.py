"""The code generator: one straight-line Python function per (schema, operation, mode)."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from graphql import GraphQLError
from graphql.execution.collect_fields import collect_fields
from graphql.execution.executor import Executor, default_field_resolver, invalid_return_type_error
from graphql.execution.values import get_argument_values
from graphql.language import DirectiveNode, VariableNode, Visitor, visit
from graphql.pyutils import Undefined, is_async_iterable, is_iterable
from graphql.type import (
    GraphQLObjectType,
    is_leaf_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
)
from graphql.type.introspection import TypeNameMetaFieldDef
from graphql.type.scalars import (
    GRAPHQL_MAX_INT,
    GRAPHQL_MIN_INT,
    serialize_boolean,
    serialize_float,
    serialize_int,
    serialize_string,
)

from ._runtime import (
    FALLBACK,
    STATS,
    _A,
    _cont,
    _drain,
    _drop_since,
    _handle,
    _info,
    _leaf,
    _list_pend,
    _mk,
    _obj_pend,
    _stock,
    _wrap,
)
from ._strawberry import _strawberry_basic

CONT_DEPTH_MAX = 4  # a field with this many custom-resolver ancestors gets no continuation
MAX_LINES = 20_000  # generated-code budget; a bigger operation runs on the stock executor
SPLIT_INDENT = 14  # deeper object/list completions go into helper functions (Python allows
#                    only 20 statically nested blocks per function)


class Unsupported(Exception):
    """Raised at compile time for anything the generator does not handle."""


# ---------------------------------------------------------------- compile-time checks
class _DirectiveScan(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.bad = False

    def enter_directive(self, node: DirectiveNode, *_args: Any) -> None:
        name = node.name.value
        if name in ("defer", "stream"):
            self.bad = True
        elif name in ("skip", "include") and any(
            isinstance(arg.value, VariableNode) for arg in node.arguments
        ):
            self.bad = True  # field set would depend on per-request variables


def _document_supported(ex: Executor) -> bool:
    scan = _DirectiveScan()
    visit(ex.operation, scan)
    for fragment in ex.fragment_definitions.values():
        visit(fragment, scan)
    return not scan.bad


def _composite(rtype: Any) -> bool:
    inner = rtype.of_type if is_non_null_type(rtype) else rtype
    return is_list_type(inner) or is_object_type(inner)


# ---------------------------------------------------------------- code generator
class _Gen:
    def __init__(self, ex: Executor, aware: bool) -> None:
        self.ex = ex
        self.aware = aware
        self.schema = ex.schema
        self.lines: list[str] = []
        self.extra: list[str] = []  # continuation functions (aware mode)
        self.conts: dict[tuple, str] = {}  # one continuation per query position, shared by copies
        self.static_path: list[str] = []  # response names from the root to the current field
        self.custom_depth = 0  # custom-resolver fields on the current query path (incl. this one)
        self.ns: dict[str, Any] = {
            "Mapping": Mapping,
            "Undefined": Undefined,
            "GraphQLError": GraphQLError,
            "is_iterable": is_iterable,
            "is_async_iterable": is_async_iterable,
            "invalid_return_type_error": invalid_return_type_error,
            "gav": get_argument_values,
            "_handle": _handle,
            "_info": _info,
            "_leaf": _leaf,
            "_mk": _mk,
            "_A": _A,
            "_stock": _stock,
            "_cont": _cont,
            "_obj_pend": _obj_pend,
            "_list_pend": _list_pend,
            "_drop_since": _drop_since,
            "_drain": _drain,
            "_wrap": _wrap,
            "_isfinite": isfinite,
            "FALLBACK": FALLBACK,
        }
        self.n = 0

    def name(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def const(self, prefix: str, value: Any) -> str:
        key = self.name(prefix)
        self.ns[key] = value
        return key

    def emit(self, ind: int, line: str) -> None:
        self.lines.append("    " * ind + line)

    # -- object: resolve each field in order, then build the dict in field order
    def object_body(self, otype: GraphQLObjectType, gfs: Any, src: str, lp: str, ind: int) -> str:
        m = self.name("m")
        self.emit(ind, f"{m} = isinstance({src}, Mapping)")
        n0 = None
        if self.aware:
            n0 = self.name("n")
            self.emit(ind, f"{n0} = len(pend)")
        pairs = []
        for response_name, fdl in gfs.items():
            r = self.name("r")
            self.field(otype, response_name, fdl, src, m, lp, r, ind)
            pairs.append(f"{response_name!r}: {r}")
        d = self.name("d")
        self.emit(ind, f"{d} = {{{', '.join(pairs)}}}")
        if n0:
            self.emit(ind, f"if len(pend) != {n0}: {d} = _obj_pend({d}, pend)")
        return d

    def field(self, otype, response_name, fdl, src, m, lp, r, ind) -> Any:  # noqa: PLR0913
        node = fdl[0].node
        fname = node.name.value
        if fname in ("__schema", "__type"):
            raise Unsupported("introspection root")
        if any(fd.fragment_variable_values is not None for fd in fdl):
            raise Unsupported("fragment arguments")
        field_def = self.schema.get_field(otype, fname)
        if field_def is None:
            raise Unsupported(f"unknown field {fname}")
        rtype = field_def.type
        non_null = is_non_null_type(rtype)
        N = self.const("N", [fd.node for fd in fdl])
        FD = self.const("FD", field_def)
        PT = self.const("PT", otype)
        FDL = self.const("FDL", fdl)
        flp = f"({lp}, {response_name!r}, {otype.name!r})"
        resolve = field_def.resolve or self.ex.field_resolver
        self.static_path.append(response_name)
        args = ""
        arg_line = None
        if field_def.args or node.arguments:
            # Coerced per field execution, inside the field's try, exactly like stock:
            # each item gets fresh (possibly mutable) argument values and coercion
            # errors land on that field.
            A = self.name("A")
            NODE = self.const("NODE", node)
            FVV = self.const("FVV", fdl[0].fragment_variable_values)
            arg_line = f"{A} = gav({FD}, {NODE}, ex.variable_values, {FVV}, ex.hide_suggestions)"
            args = f", **{A}"
        info = f"_info(ex, {FD}, {N}, {PT}, {flp})"
        ctx = (f"{otype}.{fname}", N, info, fdl, FDL, flp)  # label, nodes, info, details, consts
        v = self.name("v")
        e = self.name("e")
        nf = None
        if self.aware and _composite(rtype):  # a failure below may orphan pending work
            nf = self.name("nf")
            self.emit(ind, f"{nf} = len(pend)")
        self.emit(ind, "try:")
        if arg_line:
            self.emit(ind + 1, arg_line)
        call = None
        if field_def is TypeNameMetaFieldDef:
            self.emit(ind + 1, f"{r} = {otype.name!r}")
        else:
            custom = False
            if resolve is default_field_resolver:
                self.emit(ind + 1, f"{v} = {src}.get({fname!r}) if {m} else getattr({src}, {fname!r}, None)")
                call = f"{info}{args}"
            elif (basic := _strawberry_basic(resolve)) is not None:
                # Strawberry's basic field == field.default_resolver(source, python_name):
                # no Info, no arguments, no callable invocation, AttributeError on a miss.
                getter, attr = basic
                if getter is getattr:
                    self.emit(ind + 1, f"{v} = getattr({src}, {attr!r})")
                else:
                    self.emit(ind + 1, f"{v} = {self.const('DR', getter)}({src}, {attr!r})")
            else:
                custom = True
                RS = self.const("RS", resolve)
                self.emit(ind + 1, f"{v} = {RS}({src}, {info}{args})")
            if custom:
                self.custom_depth += 1
            if custom and self.aware:
                K = self.continuation(otype, rtype, ctx)
                self.emit(ind + 1, f"if is_aw({v}):")
                if K is not None:
                    self.emit(ind + 2, f"{r} = _A(_cont(ex, {v}, {flp}, {K}, {N}, {non_null}), pend, True)")
                else:  # past CONT_DEPTH_MAX: stock completes this subtree after the await
                    self.emit(ind + 2, f"{r} = _A(ex.complete_awaitable_value({self.const('RT', rtype)}, "
                                       f"{FDL}, {info}, _mk({flp}), {v}, None), pend, True)")
                self.emit(ind + 1, "else:")
                self.complete(rtype, v, r, ctx, flp, ind + 2, call, field_level=True, aw_checked=True)
            else:
                self.complete(rtype, v, r, ctx, flp, ind + 1, call, field_level=True)
            if custom:
                self.custom_depth -= 1
        self.emit(ind, f"except Exception as {e}:")
        if nf:
            self.emit(ind + 1, f"if len(pend) != {nf}: _drop_since(ex, pend, {nf})")
        self.emit(ind + 1, f"{r} = _handle(ex, {e}, {N}, {flp}, {non_null})")
        self.static_path.pop()
        return rtype

    def continuation(self, otype: Any, rtype: Any, ctx: tuple) -> str | None:
        """`def k(ex, v, flp, pend)` completing this field's subtree after an await.

        The subtree is also emitted inline for the non-awaitable case, so without
        sharing, every nested custom resolver would double the code. One continuation per
        query position (shared by every copy of that position) makes growth quadratic,
        and a field with CONT_DEPTH_MAX or more custom-resolver ancestors gets none (None):
        its awaited subtree goes to stock, so deeper levels only add linear code."""
        if self.custom_depth - 1 >= CONT_DEPTH_MAX:
            return None
        key = (otype.name, tuple(self.static_path), ctx[3][0].node.name.value)
        if key in self.conts:
            return self.conts[key]
        K = self.conts[key] = self.name("k")
        saved = self.lines
        self.lines = []
        out = self.name("o")
        self.emit(0, f"def {K}(ex, {out}_v, flp, pend):")
        self.emit(1, "is_aw = ex.is_awaitable")
        k_ctx = (*ctx[:2], ctx[2].replace(ctx[5], "flp"), *ctx[3:5], "flp")
        self.complete(rtype, f"{out}_v", out, k_ctx, "flp", 1, None, field_level=True, region_root=True)
        self.emit(1, f"return {out}")
        self.extra.extend(self.lines)
        self.lines = saved
        return K

    # -- complete_value, specialised by type; `call` = default-resolver callable handling
    def complete(self, rtype, v, out, ctx, lp, ind, call, field_level,  # noqa: PLR0913
                 region_root=False, aw_checked=False) -> None:
        non_null = is_non_null_type(rtype)
        inner = rtype.of_type if non_null else rtype
        null_msg = repr(f"Cannot return null for non-nullable field {ctx[0]}.")
        # Stock checks is_awaitable on a field's resolved value (execute_field) and on
        # each list item (complete_iterable_value); a continuation's value is already
        # awaited.
        check_aw = self.aware and not region_root and not aw_checked
        handoff = None
        if check_aw and field_level:
            handoff = (f"_A(ex.complete_awaitable_value({self.const('RT', rtype)}, {ctx[4]}, "
                       f"{ctx[2]}, _mk({lp}), {v}, None), pend, True)")
        elif check_aw:
            handoff = (f"_A(ex.complete_awaitable_list_item_value({v}, {self.const('RT', rtype)}, "
                       f"{ctx[4]}, {ctx[2]}, _mk({lp}), None), pend, True)")
        if is_leaf_type(inner):
            fast = self.leaf_fast_path(inner, v)
            body_ind = ind
            if fast:
                self.emit(ind, f"if {fast}:")
                self.emit(ind + 1, f"{out} = {v}")
                self.emit(ind, "else:")
                body_ind = ind + 1
            if call:
                self.emit(body_ind, f"if callable({v}): {v} = {v}({call})")
            RT = self.const("RT", inner)
            if check_aw:
                self.emit(body_ind, f"if {v} is not None and is_aw({v}):")
                self.emit(body_ind + 1, f"{out} = {handoff}")
                self.emit(body_ind, "else:")
                self.emit(body_ind + 1, f"{out} = _leaf({RT}, {v})")
            else:
                self.emit(body_ind, f"{out} = _leaf({RT}, {v})")
        else:
            if call:
                self.emit(ind, f"if callable({v}): {v} = {v}({call})")
            body_ind = ind
            if check_aw:
                guard = f"type({v}) is not list and " if is_list_type(inner) else ""
                self.emit(ind, f"if {v} is not None and {guard}is_aw({v}):")
                self.emit(ind + 1, f"{out} = {handoff}")
                self.emit(ind, "else:")
                body_ind = ind + 1
            self.emit(body_ind, f"if isinstance({v}, Exception): raise {v}")
            self.emit(body_ind, f"if {v} is None or {v} is Undefined:")
            self.emit(body_ind + 1, f"{out} = None")
            self.emit(body_ind, "else:")
            if is_list_type(inner):
                list_ind = body_ind + 1
                if self.aware:  # stock checks async iterables first (complete_list_value)
                    self.emit(list_ind, f"if type({v}) is not list and is_async_iterable({v}):")
                    self.stock_here(list_ind + 1, out, inner, ctx, lp, v)
                    self.emit(list_ind, "else:")
                    list_ind += 1
                self.composite(self.list_value, inner, v, out, ctx, lp, list_ind)
            elif is_object_type(inner):
                if self.aware and inner.is_type_of:  # may be async: stock completes this object
                    self.stock_here(body_ind + 1, out, inner, ctx, lp, v)
                else:
                    self.composite(self.object_value, inner, v, out, ctx, lp, body_ind + 1)
            else:
                raise Unsupported(f"abstract type {inner}")
            if self.aware and not region_root:
                # a pending object or list gets its position's error handling at once,
                # as in stock's execute_field / complete_list_item_value
                self.emit(ind, f"if type({out}) is _A and not {out}.wrapped: "
                               f"_wrap(ex, {out}, {self.const('RT', rtype)}, {ctx[4]}, {lp})")
        if non_null:
            # a pending value is never None, so this check is skipped for it naturally
            self.emit(ind, f"if {out} is None: raise TypeError({null_msg})")

    def stock_here(self, ind, out, inner, ctx, lp, v) -> None:  # noqa: PLR0913
        self.emit(ind, f"{out} = _stock(ex, pend, {self.const('RT', inner)}, {ctx[4]}, {ctx[2]}, _mk({lp}), {v})")

    def leaf_fast_path(self, leaf_type: Any, v: str) -> str | None:
        """Exact-type checks under which the built-in coercer returns the value unchanged."""
        coerce = leaf_type.coerce_output_value
        if coerce is serialize_string:
            return f"type({v}) is str"
        if coerce is serialize_int:
            return f"type({v}) is int and {GRAPHQL_MIN_INT} <= {v} <= {GRAPHQL_MAX_INT}"
        if coerce is serialize_float:
            return f"type({v}) is float and _isfinite({v})"
        if coerce is serialize_boolean:
            return f"type({v}) is bool"
        return None

    def leaf_list_fast_path(self, item_type: Any, v: str) -> str | None:
        """Whole-list check under which stock's completion returns the same values: an
        exact `list` whose items all have the exact type the built-in coercer passes
        through unchanged (None too where items are nullable; ints in range; floats
        finite). One C-level pass instead of a per-item loop."""
        non_null = is_non_null_type(item_type)
        leaf = item_type.of_type if non_null else item_type
        if not is_leaf_type(leaf):
            return None
        exact = {serialize_string: str, serialize_int: int, serialize_float: float,
                 serialize_boolean: bool}.get(leaf.coerce_output_value)
        if exact is None:
            return None
        allowed = {exact} if non_null or exact in (int, float) else {exact, type(None)}
        TS = self.const("TS", frozenset(allowed))
        cond = f"type({v}) is list and set(map(type, {v})) <= {TS}"
        if exact is int:
            cond += f" and (not {v} or ({GRAPHQL_MIN_INT} <= min({v}) and max({v}) <= {GRAPHQL_MAX_INT}))"
        elif exact is float:
            cond += f" and all(map(_isfinite, {v}))"
        return cond

    def composite(self, gen, ctype, v, out, ctx, lp, ind) -> None:  # noqa: PLR0913
        """Object/list completion, inline or, past SPLIT_INDENT, in a helper function
        `h(ex, v, lp, flp[, pend])` that starts again at the top indentation level."""
        if ind < SPLIT_INDENT:
            gen(ctype, v, out, ctx, lp, ind)
            return
        H = self.name("h")
        saved = self.lines
        self.lines = []
        params = "ex, v, lp, flp, pend" if self.aware else "ex, v, lp, flp"
        self.emit(0, f"def {H}({params}):")
        if self.aware:
            self.emit(1, "is_aw = ex.is_awaitable")
        h_ctx = (*ctx[:2], ctx[2].replace(ctx[5], "flp"), *ctx[3:5], "flp")
        gen(ctype, "v", "o", h_ctx, "lp", 1)
        self.emit(1, "return o")
        self.extra.extend(self.lines)
        self.lines = saved
        args = f"ex, {v}, {lp}, {ctx[5]}, pend" if self.aware else f"ex, {v}, {lp}, {ctx[5]}"
        self.emit(ind, f"{out} = {H}({args})")

    def object_value(self, otype, v, out, ctx, lp, ind) -> None:  # noqa: PLR0913
        if otype.is_type_of:
            IT = self.const("IT", otype.is_type_of)
            OT = self.const("OT", otype)
            self.emit(
                ind,
                f"if not {IT}({v}, {ctx[2]}): raise invalid_return_type_error({OT}, {v}, {ctx[1]})",
            )
        gfs, new_defer_usages, _forbidden = self.ex.collect_subfields(otype, ctx[3])
        if new_defer_usages:
            raise Unsupported("defer")
        d = self.object_body(otype, gfs, v, lp, ind)
        self.emit(ind, f"{out} = {d}")

    def list_value(self, ltype, v, out, ctx, lp, ind) -> None:  # noqa: PLR0913
        item_type = ltype.of_type
        fast = self.leaf_list_fast_path(item_type, v)
        if fast:
            self.emit(ind, f"if {fast}:")
            self.emit(ind + 1, f"{out} = {v}[:]")
            self.emit(ind, "else:")
            ind += 1
        msg = repr(f"Expected Iterable, but did not find one for field '{ctx[0]}'.")
        self.emit(ind, f"if not (type({v}) is list or is_iterable({v})): raise GraphQLError({msg})")
        lst, ap, i, it, o, e = (self.name(p) for p in ("L", "ap", "i", "it", "o", "e"))
        self.emit(ind, f"{lst} = []")
        self.emit(ind, f"{ap} = {lst}.append")
        ilp = f"({lp}, {i}, None)"
        item_non_null = is_non_null_type(item_type)
        body = ind
        nl = ni = None
        if self.aware:
            nl, itr = self.name("nl"), self.name("I")
            self.emit(ind, f"{nl} = len(pend)")
            self.emit(ind, f"{itr} = iter({v})")
            self.emit(ind, "try:")
            body = ind + 1
            self.emit(body, f"for {i}, {it} in enumerate({itr}):")
            if _composite(item_type):
                ni = self.name("ni")
                self.emit(body + 1, f"{ni} = len(pend)")
        else:
            self.emit(body, f"for {i}, {it} in enumerate({v}):")
        self.emit(body + 1, "try:")
        self.complete(item_type, it, o, ctx, ilp, body + 2, None, field_level=False)
        self.emit(body + 2, f"{ap}({o})")
        self.emit(body + 1, f"except Exception as {e}:")
        if ni:
            self.emit(body + 2, f"if len(pend) != {ni}: _drop_since(ex, pend, {ni})")
        self.emit(body + 2, f"{ap}(_handle(ex, {e}, {ctx[1]}, {ilp}, {item_non_null}))")
        if self.aware:
            self.emit(ind, "except BaseException:")
            self.emit(ind + 1, f"_drain(ex, {itr})")
            self.emit(ind + 1, "raise")
            self.emit(ind, f"if len(pend) != {nl}: {lst} = _list_pend({lst}, pend)")
        self.emit(ind, f"{out} = {lst}")


def compile_operation(ex: Executor, root_type: GraphQLObjectType, aware: bool):
    if not _document_supported(ex):
        raise Unsupported("variable-dependent or incremental directives")
    gen = _Gen(ex, aware)
    gfs, new_defer_usages, _ = collect_fields(
        ex.schema, ex.fragments, ex.variable_values, root_type, ex.operation, ex.hide_suggestions
    )
    if new_defer_usages:
        raise Unsupported("defer")
    body = gen.object_body(root_type, gfs, "root", "None", 2 if aware else 1)
    head = ["def __compiled(ex, root):"]
    if aware:  # a sync failure at the root (data: null) orphans what it started
        head += ["    pend = []", "    is_aw = ex.is_awaitable", "    try:"]
        tail = ["    except BaseException:", "        _drop_since(ex, pend, 0)", "        raise",
                f"    return {body}.aw if type({body}) is _A else {body}"]
    else:
        tail = [f"    return {body}"]
    src = [*gen.extra, *head, *gen.lines, *tail]
    STATS["last_lines"] = len(src)
    if len(src) > MAX_LINES:
        raise Unsupported(f"generated code too large ({len(src)} lines)")
    code = "\n".join(src)
    namespace = dict(gen.ns)
    exec(compile(code, "<graphql-compiled>", "exec"), namespace)  # noqa: S102 - generated from schema+AST
    return namespace["__compiled"], code
