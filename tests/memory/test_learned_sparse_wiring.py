"""S5 acceptance: Memory-layer dual-vector wiring + dual-output provider.

Covers explicit sparse insert/update/search, coverage normalization with
zero floor, original-text storage in learned mode, and byte-identical
legacy bm25 behavior (lemmatized query + sigmoid).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.memory.main import AsyncMemory, Memory
from mem0.utils.factory import EmbedderFactory

MESSAGES = [{"role": "user", "content": "我买了 iPhone 16 Pro 256G。"}]
EXTRACTION = {"memory": [{"id": "0", "text": "用户买了 iPhone 16 Pro 256G", "attributed_to": "user"}]}
SPARSE = {"indices": [3, 11], "values": [0.6, 0.4]}


class FakeDualEmbedder:
    """Class-level dual interface (contract detection is class-based)."""

    def __init__(self):
        self.config = SimpleNamespace(model="fake-dual")
        self.revision = "rev-1"
        self.embed_calls = []

    def embed(self, text, memory_action=None):
        self.embed_calls.append(text)
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts, memory_action="add"):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_batch_dual(self, texts, memory_action="add"):
        return [[0.1, 0.2, 0.3] for _ in texts], [dict(SPARSE) for _ in texts]


def _setup(mocker, *, sparse_encoder="learned", dual=True):
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")

    embedder = FakeDualEmbedder() if dual else MagicMock()
    if not dual:
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        embedder.embed_batch.side_effect = lambda texts, action="add": [[0.1, 0.2, 0.3] for _ in texts]
        embedder.config.model = "test-model"
        embedder.revision = None
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", return_value=embedder)

    main_store = MagicMock()
    main_store.supports_sparse_vectors = True
    main_store.sparse_encoder = sparse_encoder
    main_store.search.return_value = []
    main_store.list.return_value = ([], None)
    main_store.keyword_search.return_value = None
    entity_store = MagicMock()
    entity_store.supports_sparse_vectors = False
    entity_store.list.return_value = ([], None)
    entity_store.search_batch.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[main_store, entity_store])

    llm_factory = mocker.MagicMock()
    llm_factory.return_value.generate_response.return_value = json.dumps(EXTRACTION, ensure_ascii=False)
    mocker.patch("mem0.utils.factory.LlmFactory.create", llm_factory)
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    return main_store, embedder


class TestLearnedWriteWiring:
    @pytest.fixture
    def memory(self, mocker):
        self.store, self.embedder = _setup(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def test_infer_add_passes_sparse_vectors(self, memory):
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        kwargs = self.store.insert.call_args.kwargs
        assert kwargs["sparse_vectors"] == [SPARSE]

    def test_learned_mode_stores_original_text(self, memory):
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        payload = self.store.insert.call_args.kwargs["payloads"][0]
        assert payload["text_lemmatized"] == EXTRACTION["memory"][0]["text"]

    def test_direct_write_passes_sparse(self, memory):
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=False)
        assert self.store.insert.call_args.kwargs["sparse_vectors"] == [SPARSE]

    def test_update_passes_sparse_vector(self, memory):
        self.store.get.return_value = SimpleNamespace(id="m1", payload={"data": "old", "user_id": "u1"})
        self.store.update.return_value = True
        memory.update("m1", text="新文本")
        assert self.store.update.call_args.kwargs["sparse_vector"] == SPARSE

    def test_non_dual_embedder_no_sparse(self, mocker):
        store, _ = _setup(mocker, dual=False)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert "sparse_vectors" not in store.insert.call_args.kwargs


class TestLearnedSearchWiring:
    @pytest.fixture
    def memory(self, mocker):
        self.store, self.embedder = _setup(mocker)
        memory = Memory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    def _candidates(self):
        self.store.search.return_value = [
            SimpleNamespace(id="m1", score=0.6, payload={"data": "用户买了 iPhone 16 Pro 256G"})
        ]

    def test_keyword_search_receives_query_sparse(self, memory):
        self._candidates()
        memory._search_vector_store("iPhone 16 Pro", {"user_id": "u1"}, limit=5)
        kwargs = self.store.keyword_search.call_args.kwargs
        assert kwargs["sparse_vector"] == SPARSE
        assert kwargs["query"] == "iPhone 16 Pro"

    def test_coverage_normalization_applied(self, memory):
        self._candidates()
        # raw inner product 0.5; query self score = 0.6^2 + 0.4^2 = 0.52
        self.store.keyword_search.return_value = [SimpleNamespace(id="m1", score=0.5, payload={})]
        results = memory._search_vector_store("iPhone", {"user_id": "u1"}, limit=5, explain=True)
        details = results[0]["score_details"]
        assert details["bm25_score"] == pytest.approx(min(0.5 / 0.52, 1.0), abs=1e-6)
        assert details["max_possible_score"] == 2.0

    def test_zero_floor_zeroes_weak_signal(self, memory):
        self._candidates()
        self.store.keyword_search.return_value = [SimpleNamespace(id="m1", score=0.01, payload={})]
        results = memory._search_vector_store("iPhone", {"user_id": "u1"}, limit=5, explain=True)
        details = results[0]["score_details"]
        assert details["bm25_score"] == 0.0
        assert details["max_possible_score"] == 1.0  # lexical path absent -> denominator adapts

    def test_bm25_legacy_path_unchanged(self, mocker):
        store, _ = _setup(mocker, sparse_encoder="bm25")
        store.search.return_value = [SimpleNamespace(id="m1", score=0.6, payload={"data": "coffee"})]
        store.keyword_search.return_value = [SimpleNamespace(id="m1", score=8.0, payload={})]
        lemmatize_spy = mocker.patch("mem0.memory.main.lemmatize_for_bm25", return_value="coffee lemma")
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("I like coffee", {"user_id": "u1"}, limit=5, explain=True)
        kwargs = store.keyword_search.call_args.kwargs
        assert kwargs["query"] == "coffee lemma"
        assert "sparse_vector" not in kwargs
        lemmatize_spy.assert_called()
        # sigmoid normalization, not coverage
        from mem0.utils.scoring import get_bm25_params, normalize_bm25

        midpoint, steepness = get_bm25_params("I like coffee", lemmatized="coffee lemma")
        assert results[0]["score_details"]["bm25_score"] == pytest.approx(normalize_bm25(8.0, midpoint, steepness))

    def test_learned_store_non_dual_embedder_path_absent(self, mocker):
        store, _ = _setup(mocker, dual=False)
        store.search.return_value = [SimpleNamespace(id="m1", score=0.6, payload={"data": "x"})]
        store.keyword_search.return_value = None
        memory = Memory(MemoryConfig())
        results = memory._search_vector_store("q", {"user_id": "u1"}, limit=5, explain=True)
        assert "sparse_vector" not in store.keyword_search.call_args.kwargs
        assert results[0]["score_details"]["max_possible_score"] == 1.0


class TestAsyncLearnedWiring:
    @pytest.fixture
    def memory(self, mocker):
        self.store, self.embedder = _setup(mocker)
        memory = AsyncMemory(MemoryConfig())
        memory.db.get_last_messages = MagicMock(return_value=[])
        return memory

    @pytest.mark.asyncio
    async def test_add_passes_sparse(self, memory):
        await memory._add_to_vector_store(MESSAGES, {}, {"user_id": "u1"}, infer=True)
        assert self.store.insert.call_args.kwargs["sparse_vectors"] == [SPARSE]

    @pytest.mark.asyncio
    async def test_search_coverage_normalization(self, memory):
        self.store.search.return_value = [SimpleNamespace(id="m1", score=0.6, payload={"data": "x"})]
        self.store.keyword_search.return_value = [SimpleNamespace(id="m1", score=0.5, payload={})]
        results = await memory._search_vector_store("q", {"user_id": "u1"}, limit=5, explain=True)
        assert results[0]["score_details"]["bm25_score"] == pytest.approx(min(0.5 / 0.52, 1.0), abs=1e-6)

    @pytest.mark.asyncio
    async def test_update_passes_sparse(self, memory):
        self.store.get.return_value = SimpleNamespace(id="m1", payload={"data": "old", "user_id": "u1"})
        self.store.update.return_value = True
        await memory.update("m1", text="新文本")
        assert self.store.update.call_args.kwargs["sparse_vector"] == SPARSE


class TestDualOutputProvider:
    def test_config_whitelist_accepts_provider(self):
        assert EmbedderConfig(provider="dual_output", config={"model": "bge-m3"}).provider == "dual_output"

    def test_factory_creates_provider(self, mocker):
        response = mocker.MagicMock()
        response.json.return_value = {
            "dense": [[0.1, 0.2]],
            "sparse": [{"indices": [1], "values": [0.5]}],
            "model_id": "tei:bge-m3",
            "revision": "r1",
        }
        response.raise_for_status.return_value = None
        mocker.patch("mem0.utils.embedding_service.httpx.post", return_value=response)
        embedder = EmbedderFactory.create(
            "dual_output", {"model": "bge-m3", "service_url": "http://embed.local", "embedding_dims": 2}, None
        )
        assert embedder.embed("你好", "add") == [0.1, 0.2]
        dense, sparse = embedder.embed_batch_dual(["你好"], "search")
        assert dense == [[0.1, 0.2]]
        assert sparse == [{"indices": [1], "values": [0.5]}]
        assert embedder.revision == "r1"

    def test_local_fallback_requires_flagembedding(self, mocker):
        embedder = EmbedderFactory.create("dual_output", {"model": "BAAI/bge-m3"}, None)
        mocker.patch.dict("sys.modules", {"FlagEmbedding": None})
        with pytest.raises(ImportError, match="FlagEmbedding"):
            embedder.embed("你好", "add")
