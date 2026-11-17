"""S3 acceptance tests: LLM retrieval metadata + derived entity index.

Design doc 3.1 / 3.5 / 8.2 S3:
- include_metadata injects the metadata section only when enabled; the
  default prompt stays byte-identical.
- lang / entities+aliases / keywords / linked_memory_ids land in payload
  (links resolved from the anti-hallucination integer mapping).
- entity_source=llm switches Phase 7 to payload entities with spaCy
  fallback for missing/malformed output; spacy default path unchanged.
- Entity records carry aliases; cross-language merge via alias exact match.
- Derived entity collection has no sparse slot and is rebuildable from
  payloads with zero LLM calls.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig
from mem0.configs.prompts import generate_additive_extraction_prompt
from mem0.memory.main import AsyncMemory, Memory, _validate_llm_entities
from mem0.memory.utils import parse_messages

ZH_MESSAGES = [{"role": "user", "content": "我在苹果公司工作，负责 iPhone 16 Pro 产品线。"}]
LLM_ENTITIES = [
    {"surface": "苹果公司", "aliases": ["Apple", "苹果"], "type": "org"},
    {"surface": "iPhone 16 Pro", "aliases": [], "type": "brand"},
]
EXTRACTION = {
    "memory": [
        {
            "id": "0",
            "text": "用户在苹果公司工作，负责 iPhone 16 Pro 产品线",
            "attributed_to": "user",
            "lang": "zh",
            "entities": LLM_ENTITIES,
            "keywords": ["iPhone 16 Pro"],
            "linked_memory_ids": ["0"],
        }
    ]
}


def _setup_mocks(mocker, *, extraction=None, existing_results=None):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    mock_embedder = mocker.MagicMock()
    mock_embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
    mock_embedder.return_value.config.model = "test-model"
    mock_embedder.return_value.revision = None
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    main_store = mocker.MagicMock()
    main_store.search.return_value = existing_results or []
    main_store.list.return_value = ([], None)
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    mock_llm_factory = mocker.MagicMock()
    mock_llm_factory.return_value.generate_response.return_value = json.dumps(
        extraction if extraction is not None else EXTRACTION, ensure_ascii=False
    )
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm_factory)

    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store, entity_store, mock_llm_factory.return_value


def _payloads(store):
    out = []
    for call in store.insert.call_args_list:
        out.extend(call.kwargs["payloads"])
    return out


class TestValidateLlmEntities:
    def test_type_mapping(self):
        raw = [
            {"surface": "A", "type": "person"},
            {"surface": "B", "type": "topic"},
            {"surface": "C", "type": "concept"},
            {"surface": "D", "type": "identifier"},
            {"surface": "E", "type": "weird-unknown"},
        ]
        result = _validate_llm_entities(raw)
        assert [e["type"] for e in result] == ["PROPER", "TOPIC", "TOPIC", "IDENTIFIER", "PROPER"]

    def test_missing_or_malformed_returns_none(self):
        assert _validate_llm_entities(None) is None
        assert _validate_llm_entities("not a list") is None
        assert _validate_llm_entities([{"surface": ""}]) is None
        assert _validate_llm_entities([{"aliases": []}]) is None
        assert _validate_llm_entities(["string item"]) is None
        assert _validate_llm_entities([{"surface": "X", "aliases": "not a list"}]) is None

    def test_empty_list_is_valid(self):
        assert _validate_llm_entities([]) == []

    def test_dedup_by_normalized_surface(self):
        raw = [{"surface": "Apple Inc"}, {"surface": "apple  inc"}, {"surface": "Apple"}]
        result = _validate_llm_entities(raw)
        assert [e["surface"] for e in result] == ["Apple Inc", "Apple"]


class TestPromptMetadataSection:
    def test_default_prompt_byte_identical(self):
        kwargs = dict(
            existing_memories=[],
            new_messages=parse_messages(ZH_MESSAGES),
            last_k_messages=[],
            custom_instructions=None,
        )
        assert generate_additive_extraction_prompt(**kwargs) == generate_additive_extraction_prompt(
            **kwargs, use_input_language=False, include_metadata=False
        )

    def test_metadata_section_injected(self):
        prompt = generate_additive_extraction_prompt(new_messages="x", include_metadata=True)
        assert "## Retrieval Metadata" in prompt
        assert "苹果公司" in prompt and "Apple" in prompt
        assert '"lang"' in prompt and '"entities"' in prompt and '"keywords"' in prompt

    def test_wiring_llm_entity_source_injects_without_multilingual(self, mocker):
        _, _, llm = _setup_mocks(mocker)
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        prompt = llm.generate_response.call_args.kwargs["messages"][1]["content"]
        assert "## Retrieval Metadata" in prompt
        assert "## Language Requirement" not in prompt

    def test_wiring_default_off(self, mocker):
        _, _, llm = _setup_mocks(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        prompt = llm.generate_response.call_args.kwargs["messages"][1]["content"]
        assert "## Retrieval Metadata" not in prompt


class TestConfig:
    def test_default_entity_source(self):
        assert MemoryConfig().entity_source == "spacy"

    def test_invalid_entity_source_rejected(self):
        with pytest.raises(ValidationError):
            MemoryConfig(entity_source="bert")


class TestPayloadMetadata:
    @pytest.fixture
    def memory(self, mocker):
        existing = [SimpleNamespace(id="existing-uuid", score=0.9, payload={"data": "旧记忆", "hash": "h"})]
        self.main_store, self.entity_store, self.llm = _setup_mocks(mocker, existing_results=existing)
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.config.multilingual = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def test_metadata_lands_in_payload(self, memory):
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = _payloads(self.main_store)[0]
        assert payload["lang"] == "zh"
        assert payload["keywords"] == ["iPhone 16 Pro"]
        assert payload["entities"] == [
            {"surface": "苹果公司", "aliases": ["Apple", "苹果"], "type": "PROPER"},
            {"surface": "iPhone 16 Pro", "aliases": [], "type": "PROPER"},
        ]

    def test_links_resolved_to_real_uuids(self, memory):
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = _payloads(self.main_store)[0]
        assert payload["linked_memory_ids"] == ["existing-uuid"]

    def test_unknown_link_ids_dropped(self, memory):
        extraction = json.loads(json.dumps(EXTRACTION))
        extraction["memory"][0]["linked_memory_ids"] = ["0", "42"]
        self.llm.generate_response.return_value = json.dumps(extraction, ensure_ascii=False)
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = _payloads(self.main_store)[0]
        assert payload["linked_memory_ids"] == ["existing-uuid"]

    def test_spacy_default_payload_has_no_entities(self, mocker):
        main_store, _, _ = _setup_mocks(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = _payloads(main_store)[0]
        assert "entities" not in payload
        assert "lang" not in payload


class TestPhase7LlmSource:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store, _ = _setup_mocks(mocker)
        self.batch_spy = mocker.patch(
            "mem0.memory.main.extract_entities_batch", side_effect=AssertionError("spaCy batch must not run")
        )
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def test_llm_entities_used_not_spacy(self, memory):
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        self.batch_spy.assert_not_called()
        payloads = _payloads(self.entity_store)
        by_text = {p["data"]: p for p in payloads}
        assert by_text["苹果公司"]["aliases"] == ["Apple", "苹果"]
        assert by_text["苹果公司"]["entity_type"] == "PROPER"
        assert by_text["iPhone 16 Pro"]["entity_type"] == "PROPER"

    def test_fallback_to_spacy_when_entities_missing(self, mocker):
        extraction = {"memory": [{"id": "0", "text": "用户在苹果公司工作", "attributed_to": "user"}]}
        main_store, entity_store, _ = _setup_mocks(mocker, extraction=extraction)
        single_spy = mocker.patch("mem0.memory.main.extract_entities", return_value=[("PROPER", "Fallback Co")])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        single_spy.assert_called()
        payload = _payloads(main_store)[0]
        assert payload["entities"] == [{"surface": "Fallback Co", "aliases": [], "type": "PROPER"}]
        assert _payloads(entity_store)[0]["data"] == "Fallback Co"

    def test_fallback_on_malformed_entities(self, mocker):
        extraction = {
            "memory": [
                {"id": "0", "text": "用户在苹果公司工作", "attributed_to": "user", "entities": [{"surface": ""}]}
            ]
        }
        main_store, _, _ = _setup_mocks(mocker, extraction=extraction)
        mocker.patch("mem0.memory.main.extract_entities", return_value=[("PROPER", "Fallback Co")])
        memory = Memory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert _payloads(main_store)[0]["entities"][0]["surface"] == "Fallback Co"

    def test_alias_exact_match_merges_cross_language(self, memory):
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        first_payloads = _payloads(self.entity_store)
        assert len(first_payloads) == 2
        existing_row = SimpleNamespace(
            id="entity-1",
            payload={
                "data": "苹果公司",
                "aliases": ["Apple", "苹果"],
                "entity_type": "PROPER",
                "linked_memory_ids": ["m1"],
                "user_id": "u1",
            },
        )
        self.entity_store.list.return_value = ([existing_row], None)
        self.entity_store.insert.reset_mock()
        self.entity_store.update.reset_mock()

        extraction = {
            "memory": [
                {
                    "id": "0",
                    "text": "Apple released a new phone",
                    "attributed_to": "user",
                    "lang": "en",
                    "entities": [{"surface": "Apple", "aliases": [], "type": "org"}],
                }
            ]
        }
        self.llm = memory.llm
        self.llm.generate_response.return_value = json.dumps(extraction)
        memory._add_to_vector_store(
            [{"role": "user", "content": "Apple released a new phone."}], {}, {"user_id": "u1"}, infer=True
        )
        self.entity_store.insert.assert_not_called()
        updated = self.entity_store.update.call_args.kwargs["payload"]
        assert set(updated["linked_memory_ids"]) >= {"m1"}
        assert len(updated["linked_memory_ids"]) == 2


class TestSpacyLegacyPath:
    def test_spacy_batch_still_used_by_default(self, mocker):
        main_store, entity_store, _ = _setup_mocks(mocker)
        batch_spy = mocker.patch("mem0.memory.main.extract_entities_batch", return_value=[[("PROPER", "Acme")]])
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        batch_spy.assert_called_once()
        assert _payloads(entity_store)[0]["data"] == "Acme"


class TestAsyncS3:
    @pytest.fixture
    def memory(self, mocker):
        self.main_store, self.entity_store, _ = _setup_mocks(mocker)
        memory = AsyncMemory(MemoryConfig())
        memory.config.entity_source = "llm"
        memory.config.multilingual = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    @pytest.mark.asyncio
    async def test_payload_metadata_and_llm_entities(self, memory):
        await memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = _payloads(self.main_store)[0]
        assert payload["lang"] == "zh"
        assert payload["entities"][0]["surface"] == "苹果公司"
        entity_payloads = _payloads(self.entity_store)
        assert {p["data"] for p in entity_payloads} == {"苹果公司", "iPhone 16 Pro"}

    @pytest.mark.asyncio
    async def test_rebuild_zero_llm(self, memory):
        await memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        llm_calls = memory.llm.generate_response.call_count
        stored_rows = [
            SimpleNamespace(id=p_id, payload=payload)
            for call in self.main_store.insert.call_args_list
            for p_id, payload in zip(call.kwargs["ids"], call.kwargs["payloads"])
        ]
        self.main_store.list.return_value = (stored_rows, None)
        self.entity_store.reset = MagicMock()
        result = await memory.rebuild_entity_index(filters={"user_id": "u1"})
        assert result["rebuilt_memories"] == 1
        assert result["entity_links"] == 2
        assert memory.llm.generate_response.call_count == llm_calls


class TestDerivedIndexRealQdrant:
    def test_no_sparse_slot_and_rebuild(self, mocker, tmp_path):
        mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
        mocker.patch("mem0.memory.main.capture_event")
        mock_embedder = mocker.MagicMock()

        def _embed(text, action=None):
            import hashlib

            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [((h >> (i * 8)) % 256) / 256.0 + 0.01 for i in range(3)]

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
            collection_name="s3_test", embedding_model_dims=3, path=str(tmp_path / "qdrant")
        )
        config.history_db_path = str(tmp_path / "history.db")
        memory = Memory(config)
        memory.db.get_last_messages = MagicMock(return_value=[])

        memory._add_to_vector_store(ZH_MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)

        entity_store = memory.entity_store
        info = entity_store.client.get_collection(entity_store.collection_name)
        assert not info.config.params.sparse_vectors

        llm_calls = memory.llm.generate_response.call_count
        result = memory.rebuild_entity_index(filters={"user_id": "u1"})
        assert result["rebuilt_memories"] == 1
        assert result["entity_links"] == 2
        assert memory.llm.generate_response.call_count == llm_calls
        rows, _ = entity_store.list(filters={"user_id": "u1"}, top_k=100)
        assert {r.payload["data"] for r in rows} == {"苹果公司", "iPhone 16 Pro"}
        memory.close()
