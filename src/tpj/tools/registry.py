"""Per-task tool registry: maps tool names to their specs and implementations and calls them with a timeout."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from tpj.tools.spec import ToolSpec


class ToolError(Exception):
    """A tool could not be called: unknown name or bad / missing arguments."""


class ToolRegistry:
    """Tool specs plus the callables that implement them for one task."""

    def __init__(self, specs: list[ToolSpec], impls: dict[str, Callable[..., Any]], timeout: float = 30.0):
        self.specs = {spec.name: spec for spec in specs}
        self.impls = impls
        self.timeout = timeout

    def spec(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def has(self, name: str) -> bool:
        return name in self.impls

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """Call tool ``name`` with keyword ``args``. Exceptions raised by the tool itself propagate."""
        impl = self.impls.get(name)
        if impl is None:
            raise ToolError(f"unknown tool '{name}'")
        try:
            if inspect.iscoroutinefunction(impl):
                return await asyncio.wait_for(impl(**args), self.timeout)
            # sync tools run in a worker thread so a slow tool doesn't block the event loop
            return await asyncio.wait_for(asyncio.to_thread(impl, **args), self.timeout)
        except TypeError as exc:  # bad / missing arguments
            raise ToolError(f"{name}: {exc}") from exc
