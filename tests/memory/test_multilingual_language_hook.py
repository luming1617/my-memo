"""S1 acceptance tests: multilingual language hook (design doc 8.2 S1).

Invariants under test:
- Default config keeps the extraction prompt byte-identical to the
  pre-change builder output (use_input_language defaults to False).
- multilingual=True injects the existing Language Requirement section,
  for both Memory and AsyncMemory.
- Chinese extraction output is stored verbatim (language fidelity is the
  LLM's job once the hook is wired; the real-LLM quality check runs through
  scripts/multilingual_eval with a real config).
"""

import json
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.configs.prompts import generate_additive_extraction_prompt
from mem0.memory.main import AsyncMemory, Memory
from mem0.memory.utils import parse_messages

ZH_MESSAGES = [{"role": "user", "content": "我叫张伟，最喜欢喝手冲咖啡，尤其是耶加雪菲。"}]
ZH_EXTRACTION = {"memory": [{"id": "0", "text": "张伟最喜欢喝手冲咖啡，尤其是耶加雪菲", "attributed_to": "user"}]}
EN_MESSAGES = [{"role": "user", "content": "My name is Sarah and I love pour-over coffee."}]


def _setup_mocks(mocker):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    mock_embedder = mocker.MagicMock()
    mock_embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    main_store = mocker.MagicMock()
    main_store.search.return_value = []
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    mock_llm_factory = mocker.MagicMock()
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm_factory)

    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return mock_llm_factory.return_value, main_store


def _captured_user_prompt(mock_llm):
    kwargs = mock_llm.generate_response.call_args.kwargs
    return kwargs["messages"][1]["content"]


def _expected_prompt(messages, *, use_input_language=False, include_metadata=False):
    return generate_additive_extraction_prompt(
        existing_memories=[],
        new_messages=parse_messages(messages),
        last_k_messages=[],
        custom_instructions=None,
        use_input_language=use_input_language,
        include_metadata=include_metadata,
    )


class TestMultilingualConfig:
    def test_default_is_false(self):
        assert MemoryConfig().multilingual is False

    def test_explicit_true(self):
        assert MemoryConfig(multilingual=True).multilingual is True

    def test_from_config_dict(self, mocker):
        _setup_mocks(mocker)
        assert Memory.from_config({"multilingual": True}).config.multilingual is True


class TestSyncLanguageHook:
    @pytest.fixture
    def memory(self, mocker):
        self.mock_llm, self.main_store = _setup_mocks(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        self.mock_llm.generate_response.return_value = json.dumps(ZH_EXTRACTION, ensure_ascii=False)
        return memory

    def test_prompt_byte_identical_when_off(self, memory):
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        captured = _captured_user_prompt(self.mock_llm)
        assert captured == _expected_prompt(ZH_MESSAGES, use_input_language=False)
        assert "## Language Requirement" not in captured

    def test_prompt_byte_identical_when_off_english(self, memory):
        memory._add_to_vector_store(EN_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        captured = _captured_user_prompt(self.mock_llm)
        assert captured == _expected_prompt(EN_MESSAGES, use_input_language=False)

    def test_language_section_injected_when_on(self, memory):
        memory.config.multilingual = True
        memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        captured = _captured_user_prompt(self.mock_llm)
        assert "## Language Requirement" in captured
        assert captured == _expected_prompt(ZH_MESSAGES, use_input_language=True, include_metadata=True)

    def test_chinese_memory_stored_verbatim(self, memory):
        memory.config.multilingual = True
        result = memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert len(result) == 1
        assert result[0]["memory"] == ZH_EXTRACTION["memory"][0]["text"]
        payload = self.main_store.insert.call_args.kwargs["payloads"][0]
        assert payload["data"] == ZH_EXTRACTION["memory"][0]["text"]


class TestAsyncLanguageHook:
    @pytest.fixture
    def memory(self, mocker):
        self.mock_llm, self.main_store = _setup_mocks(mocker)
        memory = AsyncMemory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        self.mock_llm.generate_response.return_value = json.dumps(ZH_EXTRACTION, ensure_ascii=False)
        return memory

    @pytest.mark.asyncio
    async def test_prompt_byte_identical_when_off(self, memory):
        await memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        captured = _captured_user_prompt(self.mock_llm)
        assert captured == _expected_prompt(ZH_MESSAGES, use_input_language=False)
        assert "## Language Requirement" not in captured

    @pytest.mark.asyncio
    async def test_language_section_injected_when_on(self, memory):
        memory.config.multilingual = True
        await memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        captured = _captured_user_prompt(self.mock_llm)
        assert captured == _expected_prompt(ZH_MESSAGES, use_input_language=True, include_metadata=True)

    @pytest.mark.asyncio
    async def test_chinese_memory_stored_verbatim(self, memory):
        memory.config.multilingual = True
        result = await memory._add_to_vector_store(ZH_MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert result[0]["memory"] == ZH_EXTRACTION["memory"][0]["text"]
