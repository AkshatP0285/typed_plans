"""Plan formats compared in RQ1.

Every format tells the planner how to write a plan and parses the reply back
into the common :class:`~tpj.ir.schema.Plan`, so everything downstream
(execution, verification, metrics) is identical across formats.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tpj.ir.schema import Plan


class PlanParseError(Exception):
    pass


class PlanFormat(ABC):
    name: str = "base"

    @abstractmethod
    def instructions(self) -> str:
        """Format rules inserted into the planner prompt."""

    @abstractmethod
    def render(self, plan: Plan) -> str:
        """Render a plan in this format (used for few-shot examples and fine-tuning targets)."""

    def json_schema(self) -> dict[str, Any] | None:
        """Schema for constrained decoding, or ``None`` for free-form text."""
        return None

    @abstractmethod
    async def parse(self, text: str, query: str) -> Plan:
        """Turn planner output into a Plan. Raises :class:`PlanParseError`."""
