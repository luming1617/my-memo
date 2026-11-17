"""S8 acceptance: lossless near-duplicate tagging + read folding (3.3 / 4.5).

- Tagging never skips writes; threshold from ModelProfile.
- Folding representative = newest updated_at, cluster rank = max score,
  applied before top_k truncation; explain carries cluster info.
- Negation / number-substitution pairs below threshold are NOT folded.
- near_duplicate=False keeps legacy behavior.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory, _fold_near_duplicates

MESSAGES = [{"role": "user", "content": "我喜欢喝咖啡。"}]
EXTRACTION = {"memory": [{"id": "0", "text": "用户喜欢喝咖啡", "attributed_to": "user"}]}


def _setup(mocker, *, batch_matches=None, top1=None):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    embedder = mocker.MagicMock()
    embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    embedder.return_value.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
    embedder.return_value.config.model = "test-model"
    embedder.return_value.revision = None
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", embedder)

    main_store = mocker.MagicMock()
    main_store.search.return_value = top1 or []
    main_store.search_batch.return_value = batch_matches or [[]]
    main_store.list.return_value = ([], None)
    main_store.keyword_search.return_value = None
    entity_store = mocker.MagicMock()
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    llm_factory = mocker.MagicMock()
    llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
    mocker.patch("mem0.utils.factory.LlmFactory.create", llm_factory)
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store


def _match(score, mid="existing-1"):
    return [SimpleNamespace(id=mid, score=score, payload={"data": "用户喜欢喝咖啡"})]


class TestTagging:
    @pytest.fixture
    def memory(self, mocker):
        self.store = _setup(mocker, batch_matches=[_match(0.97)])
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def test_tagged_but_still_inserted(self, memory):
        result = memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        assert len(result) == 1  # lossless: not skipped
        payload = self.store.insert.call_args.kwargs["payloads"][0]
        assert payload["near_duplicate_of"] == "existing-1"
        assert payload["near_duplicate_similarity"] == pytest.approx(0.97)

    def test_below_threshold_not_tagged(self, mocker):
        store = _setup(mocker, batch_matches=[_match(0.90)])
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert "near_duplicate_of" not in payload

    def test_disabled_no_tag(self, mocker):
        store = _setup(mocker, batch_matches=[_match(0.99)])
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert "near_duplicate_of" not in payload

    def test_direct_write_tagged(self, mocker):
        store = _setup(mocker, top1=_match(0.98))
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=False)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert payload["near_duplicate_of"] == "existing-1"

    def test_profile_threshold_override(self, mocker):
        store = _setup(mocker, batch_matches=[_match(0.90)])
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.config.profiles = {"test-model": {"near_duplicate_threshold": 0.85}}
        from mem0.memory.main import _resolve_model_profile

        memory._model_profile = _resolve_model_profile(memory.config, memory.embedding_model)
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert payload["near_duplicate_of"] == "existing-1"


class TestFolding:
    def _candidates(self, store):
        store.search.return_value = [
            SimpleNamespace(
                id="old",
                score=0.9,
                payload={
                    "data": "用户喜欢喝咖啡",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "near_duplicate_of": "new",
                    "near_duplicate_similarity": 0.97,
                },
            ),
            SimpleNamespace(
                id="new",
                score=0.5,
                payload={"data": "用户不再喝咖啡了", "updated_at": "2025-01-01T00:00:00+00:00"},
            ),
            SimpleNamespace(
                id="other",
                score=0.8,
                payload={"data": "用户周末去徒步", "updated_at": "2025-02-01T00:00:00+00:00"},
            ),
        ]

    def test_representative_newest_rank_max(self, mocker):
        store = _setup(mocker)
        self._candidates(store)
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        results = memory._search_vector_store("咖啡", {"user_id": "u1"}, limit=5, explain=True)
        by_id = {r["id"]: r for r in results}
        assert "old" not in by_id
        assert by_id["new"]["score"] == pytest.approx(0.9)
        cluster = by_id["new"]["score_details"]["near_duplicate_cluster"]
        assert cluster["representative"] == "new"
        assert set(cluster["members"]) == {"old", "new"}
        assert cluster["folded_count"] == 1

    def test_fold_before_truncation_no_squeeze(self, mocker):
        store = _setup(mocker)
        self._candidates(store)
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        results = memory._search_vector_store("咖啡", {"user_id": "u1"}, limit=2)
        assert [r["id"] for r in results] == ["new", "other"]

    def test_folding_disabled_returns_both(self, mocker):
        store = _setup(mocker)
        self._candidates(store)
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("咖啡", {"user_id": "u1"}, limit=5)
        assert {r["id"] for r in results} == {"old", "new", "other"}

    def test_chain_clustering(self):
        scored = [
            {"id": "a", "score": 0.9, "payload": {"near_duplicate_of": "b", "updated_at": "2024-01-01"}},
            {"id": "b", "score": 0.8, "payload": {"near_duplicate_of": "c", "updated_at": "2024-06-01"}},
            {"id": "c", "score": 0.7, "payload": {"updated_at": "2025-01-01"}},
        ]
        folded = _fold_near_duplicates(scored, explain=True)
        assert len(folded) == 1
        assert folded[0]["id"] == "c"
        assert folded[0]["score"] == 0.9
        assert folded[0]["score_details"]["near_duplicate_cluster"]["folded_count"] == 2


class TestNegationNumberPairsNotFolded:
    def test_negation_pair_below_threshold(self, mocker):
        store = _setup(mocker, batch_matches=[_match(0.57)])
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert "near_duplicate_of" not in payload

    def test_number_pair_below_threshold(self, mocker):
        store = _setup(mocker, batch_matches=[_match(0.93)])
        memory = Memory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        payload = store.insert.call_args.kwargs["payloads"][0]
        assert "near_duplicate_of" not in payload


class TestAsyncS8:
    @pytest.fixture
    def memory(self, mocker):
        self.store = _setup(mocker, batch_matches=[_match(0.97)])
        memory = AsyncMemory(MemoryConfig())
        memory.config.near_duplicate = True
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    @pytest.mark.asyncio
    async def test_tagged_but_inserted(self, memory):
        result = await memory._add_to_vector_store(MESSAGES, {"user_id": "u1"}, {"user_id": "u1"}, infer=True)
        assert len(result) == 1
        payload = self.store.insert.call_args.kwargs["payloads"][0]
        assert payload["near_duplicate_of"] == "existing-1"

    @pytest.mark.asyncio
    async def test_folding(self, memory):
        self.store.search.return_value = [
            SimpleNamespace(
                id="old", score=0.9, payload={"data": "x", "updated_at": "2024-01-01", "near_duplicate_of": "new"}
            ),
            SimpleNamespace(id="new", score=0.5, payload={"data": "y", "updated_at": "2025-01-01"}),
        ]
        results = await memory._search_vector_store("q", {"user_id": "u1"}, limit=5)
        assert [r["id"] for r in results] == ["new"]
        assert results[0]["score"] == pytest.approx(0.9)


class TestRealQdrantFold:
    def test_negation_pair_folds_with_lowered_threshold(self, mocker, tmp_path):
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
        mocker.patch("mem0.utils.factory.LlmFactory.create", llm_factory)
        mocker.patch("mem0.memory.main.SQLiteManager", mocker.MagicMock())

        from mem0.configs.vector_stores.qdrant import QdrantConfig

        config = MemoryConfig()
        config.near_duplicate = True
        config.profiles = {"test-model": {"near_duplicate_threshold": 0.5}}
        config.vector_store.config = QdrantConfig(
            collection_name="s8_test", embedding_model_dims=64, path=str(tmp_path / "qdrant")
        )
        config.history_db_path = str(tmp_path / "history.db")
        memory = Memory(config)
        memory.db.get_last_messages = MagicMock(return_value=[])

        llm_factory.return_value.generate_response.return_value = json.dumps(
            {"memory": [{"id": "0", "text": "张伟喜欢喝咖啡", "attributed_to": "user"}]}, ensure_ascii=False
        )
        r1 = memory._add_to_vector_store(
            [{"role": "user", "content": "我喜欢喝咖啡"}], {"user_id": "u1"}, {"user_id": "u1"}, infer=True
        )
        llm_factory.return_value.generate_response.return_value = json.dumps(
            {"memory": [{"id": "0", "text": "张伟不再喝咖啡了", "attributed_to": "user"}]}, ensure_ascii=False
        )
        r2 = memory._add_to_vector_store(
            [{"role": "user", "content": "我戒咖啡了"}], {"user_id": "u1"}, {"user_id": "u1"}, infer=True
        )
        assert len(r1) == 1 and len(r2) == 1  # both stored (lossless)

        tagged = memory.get(r2[0]["id"])
        assert tagged["metadata"].get("near_duplicate_of") == r1[0]["id"]

        results = memory._search_vector_store("咖啡", {"user_id": "u1"}, limit=5, explain=True)
        assert len(results) == 1
        assert results[0]["id"] == r2[0]["id"]  # newest fact surfaces
        assert results[0]["score_details"]["near_duplicate_cluster"]["folded_count"] == 1
        memory.close()
