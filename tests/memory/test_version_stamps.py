"""S2 acceptance tests: payload version stamps + startup validation + profile threshold.

Design doc 6.4 / 8.2 S2:
- Every write path stamps embedder_id / sparse_encoder_id / schema_version.
- Startup samples existing payloads; a stamp mismatch blocks writes (reads OK).
- Legacy payloads without stamps are not blocked (migration is out of scope).
- The semantic threshold defaults to the resolved ModelProfile value; the
  default profile keeps the historical 0.1 so behavior is unchanged.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory, _resolve_model_profile
from mem0.utils.model_profiles import PAYLOAD_SCHEMA_VERSION

MESSAGES = [{"role": "user", "content": "I love pour-over coffee."}]
EXTRACTION = {"memory": [{"id": "0", "text": "The user loves pour-over coffee", "attributed_to": "user"}]}


def _setup_mocks(mocker, *, listed_rows=None):
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
    main_store.list.return_value = (listed_rows or [], None)
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    mock_llm_factory = mocker.MagicMock()
    mock_llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION)
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm_factory)

    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store


def _stamped_payloads(main_store):
    payloads = []
    for call in main_store.insert.call_args_list:
        payloads.extend(call.kwargs["payloads"])
    return payloads


class TestWritePathStamps:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store = _setup_mocks(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def _assert_stamps(self, payload):
        assert payload["embedder_id"] == "openai:test-model"
        assert payload["sparse_encoder_id"] == "bm25"
        assert payload["schema_version"] == PAYLOAD_SCHEMA_VERSION

    def test_infer_add_stamps(self, memory):
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payloads = _stamped_payloads(self.main_store)
        assert payloads and all("embedder_id" in p for p in payloads)
        self._assert_stamps(payloads[0])

    def test_direct_write_stamps(self, memory):
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=False)
        self._assert_stamps(_stamped_payloads(self.main_store)[0])

    def test_update_refreshes_stamps(self, memory):
        self.main_store.get.return_value = SimpleNamespace(
            id="m1", payload={"data": "old text", "user_id": "u1", "embedder_id": "openai:old-model"}
        )
        self.main_store.update.return_value = True
        memory._update_memory("m1", "new text", {"new text": [0.1, 0.2, 0.3]}, None)
        payload = self.main_store.update.call_args.kwargs["payload"]
        self._assert_stamps(payload)

    def test_metadata_only_update_refreshes_stamps(self, memory):
        self.main_store.get.return_value = SimpleNamespace(
            id="m1", payload={"data": "same text", "user_id": "u1", "embedder_id": "openai:old-model"}
        )
        self.main_store.update.return_value = True
        memory._update_memory("m1", None, {}, {"note": "x"})
        payload = self.main_store.update.call_args.kwargs["payload"]
        self._assert_stamps(payload)


class TestAsyncWritePathStamps:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store = _setup_mocks(mocker)
        memory = AsyncMemory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    @pytest.mark.asyncio
    async def test_infer_add_stamps(self, memory):
        await memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payloads = _stamped_payloads(self.main_store)
        assert payloads
        assert payloads[0]["embedder_id"] == "openai:test-model"
        assert payloads[0]["schema_version"] == PAYLOAD_SCHEMA_VERSION

    @pytest.mark.asyncio
    async def test_update_refreshes_stamps(self, memory):
        self.main_store.get.return_value = SimpleNamespace(
            id="m1", payload={"data": "old text", "user_id": "u1", "embedder_id": "openai:old-model"}
        )
        self.main_store.update.return_value = True
        await memory._update_memory("m1", "new text", {"new text": [0.1, 0.2, 0.3]}, None)
        payload = self.main_store.update.call_args.kwargs["payload"]
        assert payload["embedder_id"] == "openai:test-model"


class TestStartupValidation:
    def _row(self, stamps):
        return SimpleNamespace(id="m1", payload={"data": "legacy", **stamps})

    def test_mismatch_blocks_writes(self, mocker):
        rows = [
            self._row(
                {
                    "embedder_id": "openai:other-model",
                    "sparse_encoder_id": "bm25",
                    "schema_version": PAYLOAD_SCHEMA_VERSION,
                }
            )
        ]
        main_store = _setup_mocks(mocker, listed_rows=rows)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="collection"):
            memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        with pytest.raises(ValueError, match="collection"):
            memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=False)
        main_store.insert.assert_not_called()

    def test_mismatch_still_allows_reads(self, mocker):
        rows = [self._row({"embedder_id": "openai:other-model"})]
        main_store = _setup_mocks(mocker, listed_rows=rows)
        main_store.search.return_value = [SimpleNamespace(id="m1", score=0.9, payload={"data": "legacy"})]
        main_store.keyword_search.return_value = None
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("coffee", {"user_id": "u1"}, limit=5)
        assert [r["id"] for r in results] == ["m1"]

    def test_matching_stamps_allow_writes(self, mocker):
        rows = [
            self._row(
                {
                    "embedder_id": "openai:test-model",
                    "sparse_encoder_id": "bm25",
                    "schema_version": PAYLOAD_SCHEMA_VERSION,
                }
            )
        ]
        main_store = _setup_mocks(mocker, listed_rows=rows)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert main_store.insert.called

    def test_legacy_unstamped_payloads_allow_writes(self, mocker):
        main_store = _setup_mocks(mocker, listed_rows=[self._row({})])
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert main_store.insert.called

    def test_sparse_encoder_mismatch_blocks(self, mocker):
        rows = [
            self._row(
                {
                    "embedder_id": "openai:test-model",
                    "sparse_encoder_id": "learned",
                    "schema_version": PAYLOAD_SCHEMA_VERSION,
                }
            )
        ]
        _setup_mocks(mocker, listed_rows=rows)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        with pytest.raises(ValueError):
            memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)

    def test_async_mismatch_blocks_writes(self, mocker):
        rows = [self._row({"embedder_id": "openai:other-model"})]
        main_store = _setup_mocks(mocker, listed_rows=rows)
        memory = AsyncMemory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        with pytest.raises(ValueError):
            import asyncio

            asyncio.run(memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True))
        main_store.insert.assert_not_called()


class TestProfileThresholdWiring:
    def _candidates(self, main_store):
        main_store.search.return_value = [
            SimpleNamespace(id="low", score=0.2, payload={"data": "low scoring memory"}),
            SimpleNamespace(id="high", score=0.6, payload={"data": "high scoring memory"}),
        ]
        main_store.keyword_search.return_value = None

    def test_default_threshold_unchanged(self, mocker):
        main_store = _setup_mocks(mocker)
        self._candidates(main_store)
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("coffee", {"user_id": "u1"}, limit=5)
        assert {r["id"] for r in results} == {"low", "high"}

    def test_profile_threshold_applies(self, mocker):
        main_store = _setup_mocks(mocker)
        self._candidates(main_store)
        memory = Memory(MemoryConfig())
        memory.config.profiles = {"test-model": {"semantic_threshold": 0.5}}
        memory._model_profile = _resolve_model_profile(memory.config, memory.embedding_model)
        results = memory._search_vector_store("coffee", {"user_id": "u1"}, limit=5)
        assert [r["id"] for r in results] == ["high"]

    def test_explicit_threshold_overrides_profile(self, mocker):
        main_store = _setup_mocks(mocker)
        self._candidates(main_store)
        memory = Memory(MemoryConfig())
        memory.config.profiles = {"test-model": {"semantic_threshold": 0.5}}
        memory._model_profile = _resolve_model_profile(memory.config, memory.embedding_model)
        results = memory._search_vector_store("coffee", {"user_id": "u1"}, limit=5, threshold=0.1)
        assert {r["id"] for r in results} == {"low", "high"}
