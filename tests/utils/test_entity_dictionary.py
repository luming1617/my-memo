"""S7 acceptance: entity dictionary matching rules (design doc 4.3 layer 1)."""

from types import SimpleNamespace

from mem0.utils.entity_dictionary import EntityDictionary


def _row(surface, aliases=None, linked=None):
    return SimpleNamespace(
        id=f"e-{surface}",
        payload={"data": surface, "aliases": aliases or [], "linked_memory_ids": linked or ["m1"]},
    )


class TestMatchingRules:
    def setup_method(self):
        self.rows = [
            _row("苹果公司", ["Apple", "苹果"]),
            _row("Pineapple", []),
            _row("iPhone 16 Pro", []),
        ]
        self.loads = []

        def loader(filters):
            self.loads.append(filters)
            return self.rows

        self.dict = EntityDictionary(loader=loader)

    def test_cjk_substring(self):
        hits = self.dict.match("苹果公司最近股价如何", {"user_id": "u1"})
        assert [h.payload["data"] for h in hits] == ["苹果公司"]

    def test_latin_alias_whole_word(self):
        hits = self.dict.match("What is Apple stock price?", {"user_id": "u1"})
        assert [h.payload["data"] for h in hits] == ["苹果公司"]

    def test_apple_does_not_match_pineapple(self):
        hits = self.dict.match("Apple pie recipe", {"user_id": "u1"})
        assert all(h.payload["data"] != "Pineapple" for h in hits)

    def test_pineapple_query_matches_pineapple(self):
        hits = self.dict.match("pineapple plants", {"user_id": "u1"})
        assert [h.payload["data"] for h in hits] == ["Pineapple"]

    def test_case_insensitive(self):
        assert self.dict.match("APPLE", {"user_id": "u1"})
        assert self.dict.match("iphone 16 pro 价格", {"user_id": "u1"})

    def test_no_match(self):
        assert self.dict.match("完全无关的查询", {"user_id": "u1"}) == []


class TestCache:
    def test_ttl_and_invalidation(self):
        now = [0.0]
        loads = []

        def loader(filters):
            loads.append(filters)
            return [_row("甲公司")]

        d = EntityDictionary(loader=loader, ttl_seconds=100.0, clock=lambda: now[0])
        d.match("甲公司", {"user_id": "u1"})
        d.match("甲公司", {"user_id": "u1"})
        assert len(loads) == 1  # cached

        now[0] = 101.0
        d.match("甲公司", {"user_id": "u1"})
        assert len(loads) == 2  # TTL expired

        d.match("甲公司", {"user_id": "u1"})
        d.invalidate()
        d.match("甲公司", {"user_id": "u1"})
        assert len(loads) == 3  # write invalidation

    def test_scope_keys_isolated(self):
        loads = []

        def loader(filters):
            loads.append(filters)
            return []

        d = EntityDictionary(loader=loader)
        d.match("x", {"user_id": "u1"})
        d.match("x", {"user_id": "u2"})
        d.match("x", {"user_id": "u1"})
        assert len(loads) == 2