"""Cache of verified step outputs.

The key is the tool name plus fully resolved arguments. A repaired upstream
step that produces a different output therefore changes its children's keys
automatically, while untouched branches hit the cache. Only outputs that passed
the gate (or that the judge cleared) are stored.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(tool: str, args: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()


class NodeCache:
    enabled = True

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def fresh(self) -> "NodeCache":
        return type(self)()

    def get(self, key: str) -> tuple[bool, Any]:
        if key in self._store:
            return True, self._store[key]
        return False, None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value


class NoCache(NodeCache):
    enabled = False

    def get(self, key: str) -> tuple[bool, Any]:
        return False, None

    def put(self, key: str, value: Any) -> None:
        pass
