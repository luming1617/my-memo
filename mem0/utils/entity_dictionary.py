"""Scope-aware entity dictionary cache for the query-side entity path.

Design doc 4.3 layer 1: load entity surface + aliases per scope from the
entity store, match queries by exact name (CJK substring / Latin whole-word
boundary so Apple does not match Pineapple), cache with TTL, and invalidate
on entity writes. Language-agnostic and threshold-free; hits score sim=1.0.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, List

_CJK_RE = re.compile(r"[一-鿿]")


class EntityDictionary:
    def __init__(self, loader: Callable[[dict], list], ttl_seconds: float = 300.0, clock: Callable[[], float] = time.monotonic):
        self._loader = loader
        self._ttl = ttl_seconds
        self._clock = clock
        self._cache = {}
        self._lock = threading.Lock()

    def invalidate(self):
        with self._lock:
            self._cache.clear()

    def _entries(self, search_filters: dict):
        key = tuple(sorted((k, str(v)) for k, v in search_filters.items()))
        now = self._clock()
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
        rows = self._loader(search_filters)
        entries = []
        for row in rows:
            payload = getattr(row, "payload", None) or {}
            surface = payload.get("data")
            if not isinstance(surface, str) or not surface.strip():
                continue
            names = [surface] + [a for a in (payload.get("aliases") or []) if isinstance(a, str) and a.strip()]
            entries.append((row, names))
        with self._lock:
            self._cache[key] = (now + self._ttl, entries)
        return entries

    @staticmethod
    def _name_matches(query_lower: str, name: str) -> bool:
        name_lower = name.lower()
        if not name_lower:
            return False
        if _CJK_RE.search(name_lower):
            return name_lower in query_lower
        # Latin whole-word boundary: prevents Apple matching Pineapple.
        pattern = r"(?<![a-z0-9])" + re.escape(name_lower) + r"(?![a-z0-9])"
        return re.search(pattern, query_lower) is not None

    def match(self, query: str, search_filters: dict) -> List:
        query_lower = query.lower()
        hits = []
        for row, names in self._entries(search_filters):
            if any(self._name_matches(query_lower, name) for name in names):
                hits.append(row)
        return hits