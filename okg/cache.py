"""Per-turn score cache for serving. [D45]

Before this, every `assess` call re-scored EVERY earlier user turn, twice (adapter on and off): a 20-turn
conversation cost ~400 guard passes over its life. Now each user turn is scored once:

* key = chained SHA-256 over the messages up to and including that turn (+ the model version), so an
  edited or regenerated history gets a new key automatically and identical prefixes are shared;
* values are small dicts (guard probabilities, text-level side scores);
* bounded LRU in process memory. The interface (get / put / __contains__) is what a shared store such as
  Redis would need, if several workers serve one conversation.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


def chain_keys(messages: list[dict], namespace: str = "") -> list[str]:
    """keys[i] identifies messages[:i+1] (O(total length), computed once per call)."""
    h = hashlib.sha256(namespace.encode("utf-8"))
    keys = []
    for m in messages:
        h.update(json.dumps([m["role"], m["content"]], ensure_ascii=False).encode("utf-8"))
        h.update(b"\x1e")
        keys.append(h.copy().hexdigest()[:32])
    return keys


def text_key(namespace: str, text: str) -> str:
    return namespace + ":" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:32]


class LRUCache:
    def __init__(self, maxsize: int = 200000):
        self.maxsize = int(maxsize)
        self._d: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"entries": len(self._d), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else None}
