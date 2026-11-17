"""S5 acceptance: EmbeddingService contract + coverage normalization."""

import pytest

from mem0.utils.embedding_service import DualEmbeddingResult, EmbeddingService
from mem0.utils.scoring import normalize_sparse_coverage


class TestEmbeddingServiceRemote:
    def _mock_post(self, mocker, payload):
        response = mocker.MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return mocker.patch("mem0.utils.embedding_service.httpx.post", return_value=response)

    def test_request_and_response_contract(self, mocker):
        post = self._mock_post(
            mocker,
            {
                "dense": [[0.1, 0.2], [0.3, 0.4]],
                "sparse": [{"indices": [1, 5], "values": [0.7, 0.2]}, None],
                "model_id": "tei:bge-m3",
                "revision": "abc123",
            },
        )
        service = EmbeddingService(service_url="http://embed.local/", api_key="k", model="bge-m3")
        result = service.embed_batch(["你好", "hello"])
        post.assert_called_once()
        args, kwargs = post.call_args
        assert args[0] == "http://embed.local/embed"
        assert kwargs["json"] == {"texts": ["你好", "hello"]}
        assert kwargs["headers"]["Authorization"] == "Bearer k"
        assert result.dense == [[0.1, 0.2], [0.3, 0.4]]
        assert result.sparse[0] == {"indices": [1, 5], "values": [0.7, 0.2]}
        assert result.sparse[1] is None
        assert result.model_id == "tei:bge-m3"
        assert result.revision == "abc123"

    def test_count_mismatch_raises(self, mocker):
        self._mock_post(mocker, {"dense": [[0.1]], "sparse": [None]})
        service = EmbeddingService(service_url="http://embed.local")
        with pytest.raises(ValueError):
            service.embed_batch(["a", "b"])

    def test_empty_batch(self):
        service = EmbeddingService(service_url="http://embed.local", model="m", revision="r")
        result = service.embed_batch([])
        assert result.dense == [] and result.model_id == "m" and result.revision == "r"


class TestEmbeddingServiceLocal:
    def test_local_encoder_fallback(self):
        calls = []

        def encoder(texts):
            calls.append(texts)
            return DualEmbeddingResult(
                dense=[[1.0] for _ in texts],
                sparse=[{"indices": [0], "values": [1.0]} for _ in texts],
                model_id="local:bge-m3",
            )

        service = EmbeddingService(local_encoder=encoder)
        result = service.embed_batch(["x"])
        assert calls == [["x"]]
        assert result.model_id == "local:bge-m3"

    def test_requires_url_or_encoder(self):
        with pytest.raises(ValueError):
            EmbeddingService()


class TestCoverageNormalization:
    def test_bounded_and_absolute(self):
        assert normalize_sparse_coverage(0.5, 1.0, 0.1) == 0.5
        assert normalize_sparse_coverage(2.0, 1.0, 0.1) == 1.0

    def test_zero_floor(self):
        assert normalize_sparse_coverage(0.05, 1.0, 0.1) == 0.0
        assert normalize_sparse_coverage(0.1, 1.0, 0.1) == 0.1

    def test_zero_self_score(self):
        assert normalize_sparse_coverage(0.5, 0.0, 0.1) == 0.0

    def test_length_independent(self):
        # Same coverage ratio for short and long queries (no length buckets)
        assert normalize_sparse_coverage(1.0, 4.0, 0.1) == normalize_sparse_coverage(4.0, 16.0, 0.1)
