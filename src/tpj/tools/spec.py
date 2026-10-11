"""Tool description shared by all benchmarks (name, JSON-schema parameters, return type, safety)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """One tool as the agent sees it.

    ``parameters`` is a JSON schema for the arguments. ``returns`` is a type name such as
    ``"list[Flight]"`` and ``output_schema`` an optional JSON schema of the output. ``safety`` is
    ``"danger"`` for tools that change the world (bookings, payments, ...).
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    returns: str = "Any"
    output_schema: dict[str, Any] | None = None
    safety: Literal["safe", "danger"] = "safe"

    def to_openai(self) -> dict[str, Any]:
        """OpenAI function-calling format (used by native tool-calling agents such as ReAct)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def to_prompt(self) -> dict[str, Any]:
        """Compact dict for the tool list in planner/judge prompts; return info only when it is known."""
        prompt_entry: dict[str, Any] = {"name": self.name, "description": self.description,
                                        "parameters": self.parameters}
        if self.returns != "Any":
            prompt_entry["returns"] = self.returns
        if self.output_schema:
            prompt_entry["output_schema"] = self.output_schema
        return prompt_entry
