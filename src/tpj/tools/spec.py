from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    returns: str = "Any"
    output_schema: dict[str, Any] | None = None
    safety: Literal["safe", "danger"] = "safe"

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def to_prompt(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "description": self.description, "parameters": self.parameters}
        if self.returns != "Any":
            d["returns"] = self.returns
        if self.output_schema:
            d["output_schema"] = self.output_schema
        return d
