"""Generate OpenAI-format tool schemas from Python callables.

Uses ``inspect`` and ``typing`` to produce JSON Schema from function
signatures and docstrings — no pydantic dependency required.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Any, Callable, get_type_hints


def callable_to_openai_tool(func: Callable) -> dict:
    """Convert a Python callable to an OpenAI function-tool schema.

    Reads type hints for parameter types and the docstring for
    descriptions (Google-style ``Args:`` section).

    Supported types: ``str``, ``int``, ``float``, ``bool``,
    ``list[T]``, ``dict``, ``Optional[T]``.  Unknown types fall
    back to ``{"type": "string"}``.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    param_docs = _parse_param_docs(doc)
    func_desc = _parse_func_description(doc)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        hint = hints.get(name, str)
        prop = _python_type_to_json_schema(hint)

        if name in param_docs:
            prop["description"] = param_docs[name]

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                **({"required": required} if required else {}),
            },
        },
    }


# ------------------------------------------------------------------
# Type mapping
# ------------------------------------------------------------------

def _python_type_to_json_schema(hint: Any) -> dict:
    """Convert a Python type hint to a JSON Schema dict."""
    origin = getattr(hint, "__origin__", None)

    if hint is str:
        return {"type": "string"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is bool:
        return {"type": "boolean"}

    if origin is list:
        args = getattr(hint, "__args__", None)
        items = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    if origin is typing.Union:
        # Optional[X] = Union[X, None]
        non_none = [a for a in hint.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"type": "string"}

    # Fallback
    return {"type": "string"}


# ------------------------------------------------------------------
# Docstring parsing (Google style)
# ------------------------------------------------------------------

_ARGS_SECTION_RE = re.compile(r"^\s*Args:\s*$", re.MULTILINE)
_PARAM_RE = re.compile(r"^\s{2,}(\w+)\s*(?:\(.+?\))?\s*:\s*(.+)")


def _parse_func_description(docstring: str) -> str:
    """Extract the summary line(s) before ``Args:``."""
    match = _ARGS_SECTION_RE.search(docstring)
    desc = docstring[: match.start()].strip() if match else docstring.strip()
    # Collapse into single line
    return " ".join(desc.split())


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style ``Args:`` block."""
    match = _ARGS_SECTION_RE.search(docstring)
    if not match:
        return {}

    result: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in docstring[match.end() :].splitlines():
        # Stop at next section header (Returns:, Raises:, etc.)
        stripped = line.strip()
        if stripped and not stripped.startswith(" ") and stripped.endswith(":") and not _PARAM_RE.match(line):
            break

        pm = _PARAM_RE.match(line)
        if pm:
            # Flush previous
            if current_name:
                result[current_name] = " ".join(current_lines).strip()
            current_name = pm.group(1)
            current_lines = [pm.group(2).strip()]
        elif current_name and stripped:
            current_lines.append(stripped)

    # Flush last
    if current_name:
        result[current_name] = " ".join(current_lines).strip()

    return result
