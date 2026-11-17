"""S4 acceptance tests: update entity relinking (design doc 5.1 / 8.2 S4).

- entity_source=llm: update relinks via whole-text vector fallback against
  existing entity records (payload-only append, no re-embed of entities,
  no new entity creation, no spaCy/LLM on this path).
- The precomputed update embedding is reused (no extra embed call).
- Profile.entity_vector_threshold gates the fallback.
- entity_source=spacy: legacy spaCy relink path unchanged.
- Chinese memories keep entity links across updates (P3 regression).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory

ADD_MESSAGES = [{"role": "user", "content": "我在苹果公司工作。"}]
EXTRACTION = {
    "memory": [
        {
            "id": "0",
            "text": "用户在苹果公司工作",
            "attributed_to": "user",
            "lang": "zh",
            "entities": [{"surface": "苹果公司", "aliases": ["Apple"], "type": "org"}],
        }
    ]
}


def _setup_mocks(mocker, *, entity_matches=None):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    mock_embedder = mocker.MagicMock()
    mock_embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
    mock_embedder.return_value.config.model = "test-model"
    mock_embedder.return_value.revision = None
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    main_store = mocker.MagicMock()
    main_store.search.return_value = []
    main_store.list.return_value = ([], None)
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    entity_store.search.return_value = entity_matches or []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    mock_llm_factory = mocker.MagicMock()
    mock_llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm_factory)

    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store, entity_store, mock_embedder.return_value


def _entity_match(score=0.8, linked=None):
    return SimpleNamespace(
        id="entity-1",
        score=score,
        payload={
            "data": "苹果公司",
            "aliases": ["Apple"],
            "entity_type": "PROPER",
            "linked_memory_ids": linked if linked is not None else [],
            "user_id": "u1",
        },
    )


class TestSyncUpdateRelink:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store, self.embedder = _setup_mocks(mocker, entity_matches=[_entity_match()])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        self.main_store.get.return_value = SimpleNamespace(
            id="m1", payload={"data": "用户在苹果公司工作", "user_id": "u1"}
        )
        self.main_store.update.return_value = True
        return memory

    def test_vector_fallback_appends_link_payload_only(self, memory):
        memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        self.entity_store.update.assert_called_once()
        kwargs = self.entity_store.update.call_args.kwargs
        assert kwargs["vector_id"] == "entity-1"
        assert kwargs["vector"] is None
        assert "m1" in kwargs["payload"]["linked_memory_ids"]

    def test_no_new_entities_created(self, memory):
        memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        self.entity_store.insert.assert_not_called()

    def test_no_spacy_on_llm_update_path(self, memory, mocker):
        spacy_spy = mocker.patch("mem0.memory.main.extract_entities", side_effect=AssertionError("spaCy must not run"))
        memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        spacy_spy.assert_not_called()

    def test_precomputed_embedding_reused(self, memory):
        calls_before = self.embedder.embed.call_count
        memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        assert self.embedder.embed.call_count == calls_before

    def test_below_threshold_not_linked(self, mocker):
        main_store, entity_store, _ = _setup_mocks(mocker, entity_matches=[_entity_match(score=0.3)])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        main_store.get.return_value = SimpleNamespace(id="m1", payload={"data": "旧文本", "user_id": "u1"})
        main_store.update.return_value = True
        memory._update_memory("m1", "完全无关的新文本", {"完全无关的新文本": [0.9, 0.9, 0.9]}, None)
        entity_store.update.assert_not_called()

    def test_already_linked_not_duplicated(self, memory):
        self.entity_store.search.return_value = [_entity_match(linked=["m1"])]
        memory._update_memory("m1", "用户在苹果公司升职了", {"用户在苹果公司升职了": [0.1, 0.2, 0.3]}, None)
        self.entity_store.update.assert_not_called()

    def test_negation_update_keeps_link(self, memory):
        memory._update_memory("m1", "用户不再喜欢苹果产品了", {"用户不再喜欢苹果产品了": [0.1, 0.2, 0.3]}, None)
        self.entity_store.update.assert_called_once()
        assert "m1" in self.entity_store.update.call_args.kwargs["payload"]["linked_memory_ids"]

    def test_metadata_only_update_no_relink(self, memory):
        memory._update_memory("m1", None, {}, {"note": "x"})
        self.entity_store.search.assert_not_called()
        self.entity_store.update.assert_not_called()


class TestSpacyLegacyRelink:
    def test_spacy_mode_unchanged(self, mocker):
        main_store, entity_store, _ = _setup_mocks(mocker)
        spacy_spy = mocker.patch("mem0.memory.main.extract_entities", return_value=[("PROPER", "Acme")])
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        main_store.get.return_value = SimpleNamespace(id="m1", payload={"data": "old", "user_id": "u1"})
        main_store.update.return_value = True
        memory._update_memory("m1", "new text about Acme", {"new text about Acme": [0.1, 0.2, 0.3]}, None)
        spacy_spy.assert_called_once_with("new text about Acme")
        assert entity_store.insert.called  # legacy path may create entities


class TestAsyncUpdateRelink:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store, self.embedder = _setup_mocks(mocker, entity_matches=[_entity_match()])
        memory = AsyncMemory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        self.main_store.get.return_value = SimpleNamespace(
            id="m1", payload={"data": "用户在苹果公司工作", "user_id": "u1"}
        )
        self.main_store.update.return_value = True
        return memory

    @pytest.mark.asyncio
    async def test_vector_fallback_appends_link(self, memory):
        await memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        self.entity_store.update.assert_called_once()
        kwargs = self.entity_store.update.call_args.kwargs
        assert kwargs["vector"] is None
        assert "m1" in kwargs["payload"]["linked_memory_ids"]
        self.entity_store.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_precomputed_embedding_reused(self, memory):
        calls_before = self.embedder.embed.call_count
        await memory._update_memory("m1", "用户从苹果公司离职了", {"用户从苹果公司离职了": [0.1, 0.2, 0.3]}, None)
        assert self.embedder.embed.call_count == calls_before


class TestChineseUpdateKeepsLinksRealQdrant:
    def test_p3_regression(self, mocker, tmp_path):
        """Chinese memory update must not silently drop entity links (P3).

        Uses the deterministic content-sensitive mock embedder so the
        updated text still matches the stored Chinese entity vector.
        """
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

        mock_embedder = mocker.MagicMock()
        mock_embedder.return_value.embed.side_effect = _embed
        mock_embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [_embed(t) for t in texts]
        mock_embedder.return_value.config.model = "test-model"
        mock_embedder.return_value.revision = None
        mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)
        mock_llm_factory = mocker.MagicMock()
        mock_llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
        mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm_factory)
        mocker.patch("mem0.memory.main.SQLiteManager", mocker.MagicMock())

        from mem0.configs.vector_stores.qdrant import QdrantConfig

        config = MemoryConfig()
        config.entity_source = "llm"
        config.vector_store.config = QdrantConfig(
            collection_name="s4_test", embedding_model_dims=64, path=str(tmp_path / "qdrant")
        )
        config.history_db_path = str(tmp_path / "history.db")
        memory = Memory(config)
        memory.db.get_last_messages = MagicMock(return_value=[])

        result = memory._add_to_vector_store(ADD_MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        memory_id = result[0]["id"]
        entity_rows, _ = memory.entity_store.list(filters={"user_id": "u1"}, top_k=100)
        assert any(memory_id in r.payload["linked_memory_ids"] for r in entity_rows)

        memory.update(memory_id, text="用户最近从苹果公司离职，打算休息一段时间")

        entity_rows, _ = memory.entity_store.list(filters={"user_id": "u1"}, top_k=100)
        assert any(memory_id in r.payload["linked_memory_ids"] for r in entity_rows), (
            "Chinese memory lost its entity link after update (P3 regression)"
        )
        memory.close()
