"""S6 acceptance: pgvector sparsevec adapter (design doc 6.3 / 8.2 S6).

Mocked-cursor SQL verification (no live PostgreSQL in this environment;
live integration runs when a pgvector >= 0.7.0 database is available).
Covers: sparsevec column + HNSW(ip) index, insert/update sparse writes,
<#> query with sign flip, top-1000 non-zero truncation, 1-based indices,
capability bits, and unchanged legacy full-text path.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from mem0.vector_stores.pgvector import SPARSEVEC_MAX_NNZ, PGVector, _sparsevec_literal


def _render(stmt):
    try:
        return stmt.as_string(None)
    except Exception:
        return str(stmt)


@pytest.fixture
def pg_learned():
    with patch("mem0.vector_stores.pgvector.PSYCOPG_VERSION", 3), patch("mem0.vector_stores.pgvector.ConnectionPool"):
        store = PGVector(
            dbname="db",
            collection_name="ml_test",
            embedding_model_dims=3,
            user="u",
            password="p",
            host="h",
            port=5432,
            diskann=False,
            hnsw=True,
            sparse_encoder="learned",
        )
        cursor = MagicMock()
        cursor.rowcount = 1
        store._get_cursor = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=cursor), __exit__=MagicMock(return_value=False))
        )
        store._collection_ensured = True
        yield store, cursor


class TestSparsevecLiteral:
    def test_one_based_and_format(self):
        literal = _sparsevec_literal({"indices": [0, 4], "values": [0.5, 0.25]})
        assert literal == "{1:0.5,5:0.25}/5"

    def test_truncation_keeps_top_weights(self):
        indices = list(range(1500))
        values = [float(i) for i in range(1500)]
        literal = _sparsevec_literal({"indices": indices, "values": values})
        body = literal.split("}/")[0].strip("{")
        entries = body.split(",")
        assert len(entries) == SPARSEVEC_MAX_NNZ
        kept = {int(e.split(":")[0]) for e in entries}
        # highest weights are indices 500..1499 -> 1-based 501..1500
        assert kept == set(range(501, 1501))
        assert literal.endswith("/1500")

    def test_empty(self):
        assert _sparsevec_literal({"indices": [], "values": []}) == "{}/1"


class TestLearnedSchema:
    def test_create_col_sparsevec_and_hnsw_ip(self, pg_learned):
        store, cursor = pg_learned
        store._collection_ensured = False
        store.list_cols = MagicMock(return_value=[])
        store.create_col()
        statements = [_render(c.args[0]) for c in cursor.execute.call_args_list]
        joined = "\n".join(statements)
        assert "sparse_vector sparsevec" in joined
        assert "sparsevec_ip_ops" in joined
        assert "gin(to_tsvector" not in joined

    def test_capability_bits(self, pg_learned):
        store, _ = pg_learned
        assert store.supports_sparse_vectors is True
        assert store.sparse_encoder == "learned"

    def test_legacy_bm25_schema_unchanged(self):
        with (
            patch("mem0.vector_stores.pgvector.PSYCOPG_VERSION", 3),
            patch("mem0.vector_stores.pgvector.ConnectionPool"),
        ):
            store = PGVector(
                dbname="db",
                collection_name="legacy",
                embedding_model_dims=3,
                user="u",
                password="p",
                host="h",
                port=5432,
                diskann=False,
                hnsw=True,
            )
            cursor = MagicMock()
            store._get_cursor = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(return_value=cursor), __exit__=MagicMock(return_value=False))
            )
            store.list_cols = MagicMock(return_value=[])
            store.create_col()
            joined = "\n".join(_render(c.args[0]) for c in cursor.execute.call_args_list)
            assert "gin(to_tsvector" in joined
            assert "sparsevec" not in joined
            assert store.supports_sparse_vectors is False


class TestLearnedInsert:
    def test_insert_writes_sparse_column(self, pg_learned):
        store, cursor = pg_learned
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        store.insert(
            vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            payloads=[{"data": "a"}, {"data": "b"}],
            ids=ids,
            sparse_vectors=[{"indices": [0], "values": [0.8]}, None],
        )
        stmt = _render(cursor.executemany.call_args.args[0])
        assert "sparse_vector" in stmt and "%s::sparsevec" in stmt
        data = cursor.executemany.call_args.args[1]
        assert data[0][3] == "{1:0.8}/1"
        assert data[1][3] is None

    def test_insert_without_sparse_writes_null(self, pg_learned):
        store, cursor = pg_learned
        store.insert(vectors=[[0.1, 0.2, 0.3]], payloads=[{"data": "a"}], ids=[str(uuid.uuid4())])
        data = cursor.executemany.call_args.args[1]
        assert data[0][3] is None


class TestLearnedKeywordSearch:
    def test_inner_product_sign_flip(self, pg_learned):
        store, cursor = pg_learned
        # The SQL negates <#> (negative inner product); the DB therefore
        # returns a positive score which the adapter passes through.
        cursor.fetchall.return_value = [("id-1", 1.1, {"data": "a"})]
        results = store.keyword_search(query="q", top_k=5, sparse_vector={"indices": [0, 4], "values": [1.0, 1.0]})
        stmt = _render(cursor.execute.call_args.args[0])
        assert "<#>" in stmt
        assert "-(sparse_vector" in stmt
        params = cursor.execute.call_args.args[1]
        assert params[0] == "{1:1.0,5:1.0}/5"
        assert results[0].score == pytest.approx(1.1)

    def test_without_sparse_vector_returns_none(self, pg_learned):
        store, _ = pg_learned
        assert store.keyword_search(query="q") is None

    def test_filters_applied(self, pg_learned):
        store, cursor = pg_learned
        cursor.fetchall.return_value = []
        store.keyword_search(query="q", sparse_vector={"indices": [0], "values": [1.0]}, filters={"user_id": "u1"})
        stmt = _render(cursor.execute.call_args.args[0])
        assert "payload->>%s = %s" in stmt
        params = cursor.execute.call_args.args[1]
        assert "user_id" in params and "u1" in params

    def test_none_mode_returns_none(self):
        with (
            patch("mem0.vector_stores.pgvector.PSYCOPG_VERSION", 3),
            patch("mem0.vector_stores.pgvector.ConnectionPool"),
        ):
            store = PGVector(
                dbname="db",
                collection_name="none",
                embedding_model_dims=3,
                user="u",
                password="p",
                host="h",
                port=5432,
                diskann=False,
                hnsw=False,
                sparse_encoder="none",
            )
            store._collection_ensured = True
            assert store.keyword_search(query="q", sparse_vector={"indices": [0], "values": [1.0]}) is None
            assert store.supports_sparse_vectors is False


class TestLearnedUpdate:
    def test_update_refreshes_sparse_column(self, pg_learned):
        store, cursor = pg_learned
        store.update(
            vector_id="id-1",
            vector=[0.1, 0.2, 0.3],
            payload={"data": "new"},
            sparse_vector={"indices": [2], "values": [0.9]},
        )
        statements = [_render(c.args[0]) for c in cursor.execute.call_args_list]
        assert any("SET sparse_vector = %s::sparsevec" in s for s in statements)
        sparse_call = next(c for c in cursor.execute.call_args_list if "sparse_vector" in _render(c.args[0]))
        assert sparse_call.args[1][0] == "{3:0.9}/3"

    def test_payload_only_update_no_sparse_write(self, pg_learned):
        store, cursor = pg_learned
        store.update(vector_id="id-1", vector=None, payload={"data": "new"})
        statements = [_render(c.args[0]) for c in cursor.execute.call_args_list]
        assert not any("sparse_vector" in s for s in statements)
