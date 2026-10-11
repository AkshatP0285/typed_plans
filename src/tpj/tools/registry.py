from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from tpj.tools.spec import ToolSpec


class ToolError(Exception):
    pass


class ToolRegistry:
    """Tool specs plus the callables that implement them for one task."""

    def __init__(self, specs: list[ToolSpec], impls: dict[str, Callable[..., Any]], timeout: float = 30.0):
        self.specs = {s.name: s for s in specs}
        self.impls = impls
        self.timeout = timeout

    def spec(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def has(self, name: str) -> bool:
        return name in self.impls

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        fn = self.impls.get(name)
        if fn is None:
            raise ToolError(f"unknown tool '{name}'")
        try:
            if inspect.iscoroutinefunction(fn):
                return await asyncio.wait_for(fn(**args), self.timeout)
            return await asyncio.wait_for(asyncio.to_thread(fn, **args), self.timeout)
        except TypeError as e:  # bad / missing arguments
            raise ToolError(f"{name}: {e}") from e
