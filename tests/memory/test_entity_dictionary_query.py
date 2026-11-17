"""S7 acceptance: query-side two-layer entity boosts (design doc 4.3 / 8.2 S7).

Dictionary-first + whole-query vector fallback, scope filtering preserved,
write invalidation, legacy spaCy path untouched, multi-tenant isolation.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory

MESSAGES = [{"role": "user", "content": "我在苹果公司工作。"}]
EXTRACTION = {
    "memory": [
        {
            "id": "0",
            "text": "用户在苹果公司工作",
            "attributed_to": "user",
            "entities": [{"surface": "苹果公司", "aliases": ["Apple"], "type": "org"}],
        }
    ]
}


def _entity_row(surface="苹果公司", aliases=None, linked=None, user="u1"):
    return SimpleNamespace(
        id="entity-1",
        payload={
            "data": surface,
            "aliases": aliases if aliases is not None else ["Apple"],
            "entity_type": "PROPER",
            "linked_memory_ids": linked if linked is not None else ["m1"],
            "user_id": user,
        },
    )


def _setup(mocker, *, entity_rows=None, vector_matches=None, entity_source="llm"):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    embedder = mocker.MagicMock()
    embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
    embedder.return_value.config.model = "test-model"
    embedder.return_value.revision = None
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", embedder)

    main_store = mocker.MagicMock()
    main_store.search.return_value = [SimpleNamespace(id="m1", score=0.6, payload={"data": "用户在苹果公司工作"})]
    main_store.list.return_value = ([], None)
    main_store.keyword_search.return_value = None
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = (entity_rows if entity_rows is not None else [_entity_row()], None)
    entity_store.search.return_value = vector_matches or []
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    llm_factory = mocker.MagicMock()
    llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
    mocker.patch("mem0.utils.factory.LlmFactory.create", llm_factory)
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store, entity_store


class TestDictionaryBoosts:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store = _setup(mocker)
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def test_chinese_entity_query_gets_boost(self, memory):
        results = memory._search_vector_store("苹果公司股价", {"user_id": "u1"}, limit=5, explain=True)
        details = results[0]["score_details"]
        assert details["entity_boost"] == pytest.approx(0.5)
        assert details["max_possible_score"] == 1.5

    def test_alias_cross_language_boost(self, memory):
        results = memory._search_vector_store("Apple stock", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == pytest.approx(0.5)

    def test_pineapple_not_boosted_by_apple(self, mocker):
        main_store, _ = _setup(mocker, entity_rows=[_entity_row(surface="Pineapple", aliases=[], linked=["m2"])])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        results = memory._search_vector_store("Apple stock", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == 0.0

    def test_scope_filters_passed_to_dictionary(self, memory):
        memory._search_vector_store("苹果公司", {"user_id": "u1", "agent_id": "a1"}, limit=5)
        filters = self.entity_store.list.call_args.kwargs["filters"]
        assert filters == {"user_id": "u1", "agent_id": "a1"}

    def test_no_spacy_on_query_side_in_llm_mode(self, memory, mocker):
        spacy_spy = mocker.patch("mem0.memory.main.extract_entities", side_effect=AssertionError("spaCy must not run"))
        memory._search_vector_store("苹果公司", {"user_id": "u1"}, limit=5)
        spacy_spy.assert_not_called()

    def test_write_invalidates_cache(self, memory):
        memory._search_vector_store("苹果公司", {"user_id": "u1"}, limit=5)
        loads_after_first = self.entity_store.list.call_count
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        memory._search_vector_store("苹果公司", {"user_id": "u1"}, limit=5)
        assert self.entity_store.list.call_count > loads_after_first + 1


class TestVectorFallback:
    def test_fallback_boost_uses_similarity(self, mocker):
        match = SimpleNamespace(id="entity-1", score=0.7, payload={"linked_memory_ids": ["m1"], "data": "苹果公司"})
        main_store, _ = _setup(mocker, entity_rows=[], vector_matches=[match])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        results = memory._search_vector_store("某公司", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == pytest.approx(0.7 * 0.5)

    def test_below_threshold_no_boost(self, mocker):
        match = SimpleNamespace(id="entity-1", score=0.3, payload={"linked_memory_ids": ["m1"], "data": "x"})
        main_store, _ = _setup(mocker, entity_rows=[], vector_matches=[match])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        results = memory._search_vector_store("q", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == 0.0

    def test_dictionary_hit_takes_max(self, mocker):
        match = SimpleNamespace(id="entity-1", score=0.7, payload={"linked_memory_ids": ["m1"], "data": "苹果公司"})
        main_store, _ = _setup(mocker, vector_matches=[match])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        results = memory._search_vector_store("苹果公司", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == pytest.approx(0.5)


class TestLegacyPath:
    def test_spacy_mode_uses_legacy_boosts(self, mocker):
        main_store, entity_store = _setup(mocker, entity_source="spacy")
        spacy_spy = mocker.patch("mem0.memory.main.extract_entities", return_value=[("PROPER", "Acme")])
        legacy_spy = mocker.patch.object(Memory, "_compute_entity_boosts", return_value={"m1": 0.4})
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("Acme", {"user_id": "u1"}, limit=5, explain=True)
        spacy_spy.assert_called_once()
        legacy_spy.assert_called_once()
        assert results[0]["score_details"]["entity_boost"] == 0.4


class TestAsyncDictionaryBoosts:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store = _setup(mocker)
        memory = AsyncMemory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    @pytest.mark.asyncio
    async def test_chinese_query_boost(self, memory):
        results = await memory._search_vector_store("苹果公司股价", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_vector_fallback(self, mocker):
        match = SimpleNamespace(id="entity-1", score=0.8, payload={"linked_memory_ids": ["m1"], "data": "x"})
        main_store, _ = _setup(mocker, entity_rows=[], vector_matches=[match])
        memory = AsyncMemory(MemoryConfig())
        memory.config.entity_source = "llm"
        results = await memory._search_vector_store("q", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["entity_boost"] == pytest.approx(0.4)


class TestMultiTenantIsolationRealQdrant:
    def test_boosts_stay_in_scope(self, mocker, tmp_path):
        import hashlib
        import math
        import re

        mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
        mocker.patch("mem0.memory.main.capture_event")

        def _embed(text, action=None):
            text = text.lower()
            tokens = re.findall(r"[a-z0-9]+", text)
            cjk = re.findall(r"[一-鿿]", text)
            tokens += cjk + [a + b for a, b in zip(cjk, cjk[1:])]
            vec = [0.0] * 64
            for t in tokens:
                vec[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            return [v / norm for v in vec] if norm else vec

        embedder = mocker.MagicMock()
        embedder.return_value.embed.side_effect = _embed
        embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [_embed(t) for t in texts]
        embedder.return_value.config.model = "test-model"
        embedder.return_value.revision = None
        mocker.patch("mem0.utils.factory.EmbedderFactory.create", embedder)
        llm_factory = mocker.MagicMock()
        llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
        mocker.patch("mem0.utils.factory.LlmFactory.create", llm_factory)
        mocker.patch("mem0.memory.main.SQLiteManager", mocker.MagicMock())

        from mem0.configs.vector_stores.qdrant import QdrantConfig

        config = MemoryConfig()
        config.entity_source = "llm"
        config.vector_store.config = QdrantConfig(
            collection_name="s7_tenant", embedding_model_dims=64, path=str(tmp_path / "qdrant")
        )
        config.history_db_path = str(tmp_path / "history.db")
        memory = Memory(config)
        memory.db.get_last_messages = MagicMock(return_value=[])

        r1 = memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        r2 = memory._add_to_vector_store(MESSAGES, {"user_id": "u2"}, {"user_id": "u2"}, infer=True)
        u1_ids = {r["id"] for r in r1}
        u2_ids = {r["id"] for r in r2}

        results = memory._search_vector_store("苹果公司", {"user_id": "u1"}, limit=10, explain=True)
        assert results, "scoped search must return the tenant's memory"
        for item in results:
            assert item["id"] in u1_ids
            assert item["id"] not in u2_ids
            if item["score_details"]["entity_boost"] > 0:
                assert item["id"] in u1_ids
        memory.close()