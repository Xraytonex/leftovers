"""Recognition of Strawberry's basic-field resolver (optional; Strawberry not required)."""

from __future__ import annotations

from typing import Any


def _strawberry_basic(resolve: Any) -> tuple[Any, str] | None:
    """(default_resolver, python_name) if `resolve` is Strawberry's basic-field resolver.

    Strawberry wraps a field that has no resolver, extensions or exception handlers in
    `_get_basic_result`, which returns `field.get_result(source, None, [], {})`, i.e.
    `field.default_resolver(source, field.python_name)`. Anything else stays a call."""
    if not getattr(resolve, "_is_default", False) or getattr(resolve, "__name__", "") != "_get_basic_result":
        return None
    try:
        from strawberry.types.field import StrawberryField
    except ImportError:
        return None
    for cell in getattr(resolve, "__closure__", None) or ():
        f = cell.cell_contents
        if (
            isinstance(f, StrawberryField)
            and f.is_basic_field
            and type(f).get_result is StrawberryField.get_result
        ):
            return f.default_resolver, f.python_name
    return None
