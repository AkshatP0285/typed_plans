from __future__ import annotations

from abc import ABC, abstractmethod

from tpj.benchmarks.base import AgentResult, Task
from tpj.tools.registry import ToolRegistry


class Agent(ABC):
    @abstractmethod
    async def run(self, task: Task, registry: ToolRegistry) -> AgentResult: ...
