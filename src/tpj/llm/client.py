"""OpenAI-compatible chat client used for every generative model.

The same class talks to vLLM servers (local Qwen, LoRA adapters) and hosted
APIs (OpenAI, DashScope, ...). Responses are cached on disk keyed by the full
request, so re-running an unchanged experiment costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from tpj.eval.trace import emit

StructuredMode = Literal["response_format", "guided_json", "none"]


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        structured_output: StructuredMode = "response_format",
        extra_body: dict[str, Any] | None = None,
        cache_dir: str | None = ".cache/llm",
        max_concurrency: int = 16,
        max_retries: int = 3,
        timeout: float = 120.0,
        name: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        api_key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.name = name or model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.structured_output = structured_output
        self.extra_body = dict(extra_body or {})
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(cache_dir)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """``role`` labels the call in the trace (planner / judge / gate / ...)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        extra_body = dict(self.extra_body)
        if json_schema is not None:
            if self.structured_output == "response_format":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": json_schema},
                }
            elif self.structured_output == "guided_json":
                extra_body["guided_json"] = json_schema
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body

        key = hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()
        if self._cache is not None and key in self._cache:
            res: LLMResult = self._cache[key]
            res = LLMResult(**{**res.__dict__, "cached": True})
            self._record(role, res, latency=0.0, messages=messages)
            return res

        t0 = time.time()
        async with self._sem:
            resp = await self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        res = LLMResult(
            text=msg.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            tool_calls=tool_calls,
        )
        if self._cache is not None:
            self._cache[key] = res
        self._record(role, res, latency=time.time() - t0, messages=messages)
        return res

    def _record(self, role: str, res: LLMResult, latency: float, messages: list[dict[str, Any]]) -> None:
        # Raw prompt + response go into the trace itself (not just the disk cache, which is
        # keyed by request hash and not organised per-task) so a task's trace.jsonl is
        # self-contained for re-scoring under different criteria later without re-running
        # any model - e.g. a free_text run's planner output (before conversion) and its
        # converter output (after) are both here, independently inspectable.
        emit(
            "llm_call",
            role=role,
            model=self.name,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            cached=res.cached,
            latency=latency,
            prompt=messages,
            response=res.text,
        )


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    # Qwen3-style reasoning blocks
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])
