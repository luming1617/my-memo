"""S5 acceptance: Qdrant learned-sparse (lexical) slot full chain.

Real embedded Qdrant (no server needed). Verifies slot creation without
IDF modifier, explicit sparse insert/search/update, capability bits, and
that the legacy bm25 slot keeps its IDF modifier.
"""

import pytest
from qdrant_client.models import Modifier

from mem0.vector_stores.qdrant import Qdrant


@pytest.fixture
def learned_store(tmp_path):
    store = Qdrant(
        collection_name="learned_test",
        embedding_model_dims=4,
        path=str(tmp_path / "qdrant"),
        sparse_encoder="learned",
    )
    yield store


SPARSE_A = {"indices": [1, 7], "values": [0.8, 0.3]}
SPARSE_B = {"indices": [2, 9], "values": [0.5, 0.5]}
SPARSE_QUERY = {"indices": [1, 7], "values": [1.0, 1.0]}


class TestLexicalSlot:
    def test_slot_created_without_modifier(self, learned_store):
        info = learned_store.client.get_collection("learned_test")
        sparse_cfg = info.config.params.sparse_vectors
        assert "lexical" in sparse_cfg
        assert "bm25" not in sparse_cfg
        assert sparse_cfg["lexical"].modifier is None
        assert learned_store.supports_sparse_vectors is True
        assert learned_store.sparse_encoder == "learned"

    def test_bm25_legacy_slot_keeps_idf(self, tmp_path):
        store = Qdrant(collection_name="bm25_test", embedding_model_dims=4, path=str(tmp_path / "qdrant_bm25"))
        info = store.client.get_collection("bm25_test")
        assert info.config.params.sparse_vectors["bm25"].modifier == Modifier.IDF
        assert store.supports_sparse_vectors is True

    def test_insert_and_keyword_search(self, learned_store):
        learned_store.insert(
            vectors=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            payloads=[{"data": "甲公司"}, {"data": "乙公司"}],
            ids=[1, 2],
            sparse_vectors=[SPARSE_A, SPARSE_B],
        )
        hits = learned_store.keyword_search(query="甲", top_k=5, sparse_vector=SPARSE_QUERY)
        assert hits is not None
        assert hits[0].id == 1
        # inner product: 0.8*1.0 + 0.3*1.0 = 1.1
        assert hits[0].score == pytest.approx(1.1, abs=1e-6)

    def test_keyword_search_without_sparse_vector_returns_none(self, learned_store):
        assert learned_store.keyword_search(query="甲", top_k=5) is None

    def test_insert_without_sparse_still_works(self, learned_store):
        learned_store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "x"}], ids=[1])
        assert learned_store.get(1) is not None

    def test_update_full_branch_refreshes_lexical(self, learned_store):
        learned_store.insert(
            vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "old"}], ids=[1], sparse_vectors=[SPARSE_A]
        )
        learned_store.update(
            vector_id=1,
            vector=[0.4, 0.3, 0.2, 0.1],
            payload={"data": "new"},
            sparse_vector=SPARSE_B,
        )
        hits = learned_store.keyword_search(query="q", top_k=5, sparse_vector={"indices": [2, 9], "values": [1.0, 1.0]})
        assert hits and hits[0].id == 1
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        old_hits = learned_store.keyword_search(query="q", top_k=5, sparse_vector=SPARSE_QUERY)
        assert not old_hits or old_hits[0].score == pytest.approx(0.0, abs=1e-6)

    def test_update_payload_only_does_not_touch_sparse(self, learned_store):
        learned_store.insert(
            vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "old"}], ids=[1], sparse_vectors=[SPARSE_A]
        )
        learned_store.update(vector_id=1, vector=None, payload={"data": "new", "linked_memory_ids": ["m"]})
        hits = learned_store.keyword_search(query="q", top_k=5, sparse_vector=SPARSE_QUERY)
        assert hits and hits[0].score == pytest.approx(1.1, abs=1e-6)

    def test_existing_collection_slot_detection(self, tmp_path):
        path = str(tmp_path / "qdrant_reopen")
        Qdrant(collection_name="reopen", embedding_model_dims=4, path=path, sparse_encoder="learned")
        reopened = Qdrant(collection_name="reopen", embedding_model_dims=4, path=path, sparse_encoder="learned")
        assert reopened._has_lexical_slot is True
        assert reopened.supports_sparse_vectors is True

    def test_learned_on_legacy_collection_absent_path(self, tmp_path):
        path = str(tmp_path / "qdrant_legacy")
        Qdrant(collection_name="legacy", embedding_model_dims=4, path=path)  # bm25 collection
        learned = Qdrant(collection_name="legacy", embedding_model_dims=4, path=path, sparse_encoder="learned")
        assert learned._has_lexical_slot is False
        assert learned.supports_sparse_vectors is False
        assert learned.keyword_search(query="q", sparse_vector=SPARSE_QUERY) is None

    def test_none_mode_no_capability(self, tmp_path):
        store = Qdrant(
            collection_name="none_test",
            embedding_model_dims=4,
            path=str(tmp_path / "qdrant_none"),
            sparse_encoder="none",
        )
        assert store.supports_sparse_vectors is False
        info = store.client.get_collection("none_test")
        assert not info.config.params.sparse_vectors
