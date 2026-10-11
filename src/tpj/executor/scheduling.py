"""Which ready steps run together."""
from __future__ import annotations


class WaveScheduler:
    """Run every ready step at once (dependency-aware waves)."""

    name = "waves"

    def __init__(self, max_parallel: int | None = None):
        self.max_parallel = max_parallel

    def select(self, ready: list[str]) -> list[str]:
        return ready[: self.max_parallel] if self.max_parallel else ready


class SequentialScheduler(WaveScheduler):
    """One step at a time (ablation: verification after every call)."""

    name = "sequential"

    def __init__(self) -> None:
        super().__init__(max_parallel=1)
