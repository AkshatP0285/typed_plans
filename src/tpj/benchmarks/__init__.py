"""Benchmark adapters. Import concrete adapters from their modules (they pull optional deps)."""
from tpj.benchmarks.base import AgentResult, Benchmark, CallRecord, Task

__all__ = ["AgentResult", "Benchmark", "CallRecord", "Task"]
