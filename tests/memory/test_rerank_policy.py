"""S9 acceptance: rerank scope + conditional triggers (design doc 4.6 / 8.2 S9).

- rerank=True stays explicit; only the top rerank_top_n (default 20) are sent.
- conditional trigger: language mismatch (payload.lang) or low semantic score.
- Not triggered -> results identical to no-rerank behavior.
- Reranker failure falls back to original results.
"""

from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory


def _items(n, lang=None, semantic=0.8):
    out = []
    for i in range(n):
        item = {"id": f"m{i}", "memory": f"memory {i}", "score": semantic}
        if lang:
            item["metadata"] = {"lang": lang}
        item["score_details"] = {"semantic_score": semantic}
        out.append(item)
    return out


def _setup(mocker):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mocker.MagicMock())
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", mocker.MagicMock())
    mocker.patch("mem0.utils.factory.LlmFactory.create", mocker.MagicMock())
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())


@pytest.fixture
def memory(mocker):
    _setup(mocker)
    mem = Memory(MemoryConfig())
    mem.reranker = MagicMock()
    mem.reranker.rerank.side_effect = lambda q, docs, k: list(reversed(docs))[:k]
    return mem


class TestRerankScope:
    def test_top_20_truncation(self, memory, mocker):
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(30))
        memory.search("coffee", filters={"user_id": "u1"}, rerank=True, top_k=30)
        docs = memory.reranker.rerank.call_args.args[1]
        assert len(docs) == 20

    def test_custom_top_n(self, memory, mocker):
        memory.config.rerank_top_n = 5
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(30))
        memory.search("coffee", filters={"user_id": "u1"}, rerank=True, top_k=30)
        docs = memory.reranker.rerank.call_args.args[1]
        assert len(docs) == 5

    def test_failure_falls_back(self, memory, mocker):
        original = _items(3)
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        memory.reranker.rerank.side_effect = RuntimeError("reranker down")
        result = memory.search("coffee", filters={"user_id": "u1"}, rerank=True)
        assert result["results"] == original

    def test_rerank_false_untouched(self, memory, mocker):
        original = _items(3)
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        result = memory.search("coffee", filters={"user_id": "u1"}, rerank=False)
        memory.reranker.rerank.assert_not_called()
        assert result["results"] == original


class TestConditionalTrigger:
    def test_lang_mismatch_triggers(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(3, lang="en"))
        memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_called_once()

    def test_lang_match_skips(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        original = _items(3, lang="zh")
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        result = memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_not_called()
        assert result["results"] == original

    def test_missing_lang_does_not_trigger(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        original = _items(3)
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        result = memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_not_called()
        assert result["results"] == original

    def test_low_semantic_triggers(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        memory.config.rerank_semantic_threshold = 0.4
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(3, semantic=0.2))
        memory.search("coffee", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_called_once()

    def test_high_semantic_skips(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        memory.config.rerank_semantic_threshold = 0.4
        original = _items(3, semantic=0.9)
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        result = memory.search("coffee", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_not_called()
        assert result["results"] == original

    def test_always_default_reranks_on_match(self, memory, mocker):
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(3, lang="zh"))
        memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_called_once()


class TestAsyncRerank:
    @pytest.fixture
    def memory(self, mocker):
        _setup(mocker)
        mem = AsyncMemory(MemoryConfig())
        mem.reranker = MagicMock()
        mem.reranker.rerank.side_effect = lambda q, docs, k: list(reversed(docs))[:k]
        return mem

    @pytest.mark.asyncio
    async def test_top_20_truncation(self, memory, mocker):
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(30))
        await memory.search("coffee", filters={"user_id": "u1"}, rerank=True, top_k=30)
        docs = memory.reranker.rerank.call_args.args[1]
        assert len(docs) == 20

    @pytest.mark.asyncio
    async def test_conditional_skip(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        original = _items(3, lang="zh")
        mocker.patch.object(memory, "_search_vector_store", return_value=original)
        result = await memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_not_called()
        assert result["results"] == original

    @pytest.mark.asyncio
    async def test_conditional_lang_mismatch(self, memory, mocker):
        memory.config.rerank_trigger = "conditional"
        mocker.patch.object(memory, "_search_vector_store", return_value=_items(3, lang="en"))
        await memory.search("咖啡", filters={"user_id": "u1"}, rerank=True)
        memory.reranker.rerank.assert_called_once()
