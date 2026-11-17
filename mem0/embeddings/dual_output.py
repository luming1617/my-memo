"""Dual-output embedding provider: dense + learned sparse in one forward pass.

Wraps EmbeddingService (remote HTTP first-class, in-process FlagEmbedding
fallback for development). ``embed``/``embed_batch`` keep the dense-only
EmbeddingBase view so existing callers work unchanged; ``embed_batch_dual``
exposes both outputs for the Memory layer's explicit sparse write path.
"""

from __future__ import annotations

from typing import Literal, Optional

from mem0.embeddings.base import EmbeddingBase
from mem0.utils.embedding_service import DualEmbeddingResult, EmbeddingService


class DualOutputEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.revision = getattr(self.config, "revision", None)
        self.model_id = None
        self._local_model = None
        service_url = getattr(self.config, "service_url", None)
        self._service = EmbeddingService(
            service_url=service_url,
            model=self.config.model,
            api_key=self.config.api_key,
            revision=self.revision,
            local_encoder=None if service_url else self._local_encode,
        )

    def _local_encode(self, texts) -> DualEmbeddingResult:
        if self._local_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as e:
                raise ImportError(
                    "In-process dual-output embedding requires FlagEmbedding. "
                    'Install it with: pip install "mem0ai[dual]"'
                ) from e
            self._local_model = BGEM3FlagModel(self.config.model or "BAAI/bge-m3", use_fp16=False)
        model_name = self.config.model or "BAAI/bge-m3"
        out = self._local_model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense = [vec.tolist() for vec in out["dense_vecs"]]
        sparse = []
        for weights in out["lexical_weights"]:
            indices = [int(i) for i in weights.keys()]
            sparse.append({"indices": indices, "values": [float(weights[i]) for i in indices]})
        return DualEmbeddingResult(
            dense=dense, sparse=sparse, model_id=f"local:{model_name}", revision=self.revision
        )

    def embed_batch_dual(self, texts, memory_action: Optional[Literal["add", "search", "update"]] = "add"):
        result = self._service.embed_batch(list(texts))
        self.model_id = result.model_id
        self.revision = result.revision
        return result.dense, result.sparse

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        dense, _ = self.embed_batch_dual([text], memory_action)
        return dense[0]

    def embed_batch(self, texts, memory_action="add"):
        dense, _ = self.embed_batch_dual(texts, memory_action)
        return dense