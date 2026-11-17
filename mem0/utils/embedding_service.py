"""EmbeddingService: batch dual-output (dense + learned sparse) embedding contract.

Design doc 2.3 / 3.2. One forward pass yields both vector types plus model
identity. Remote HTTP batch services are the first-class path (the model
process does not scale with API workers); an in-process encoder can be
injected as a development/single-machine fallback.

Remote protocol (a thin FastAPI/TEI/vLLM wrapper implements it):
    POST {service_url}/embed
    request:  {"texts": ["...", ...]}
    response: {"dense": [[float, ...], ...],
               "sparse": [{"indices": [int, ...], "values": [float, ...]}, ...],
               "model_id": "provider:model", "revision": "..." | null}
``sparse`` entries may be null when the service cannot encode a text.

Supply chain: weights are pinned by revision; services must load from
fixed local paths and never fetch weights from the public internet at
runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DualEmbeddingResult:
    dense: List[List[float]]
    sparse: List[Optional[dict]] = field(default_factory=list)
    model_id: str = ""
    revision: Optional[str] = None


class EmbeddingService:
    """Batch dual-output client. Exactly one of service_url / local_encoder."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        revision: Optional[str] = None,
        timeout: float = 60.0,
        local_encoder: Optional[Callable[[List[str]], DualEmbeddingResult]] = None,
    ):
        if not service_url and local_encoder is None:
            raise ValueError("EmbeddingService requires service_url (remote) or local_encoder (in-process fallback)")
        self.service_url = service_url.rstrip("/") if service_url else None
        self.model = model
        self.api_key = api_key
        self.revision = revision
        self.timeout = timeout
        self.local_encoder = local_encoder

    def embed_batch(self, texts: List[str]) -> DualEmbeddingResult:
        if not texts:
            return DualEmbeddingResult(dense=[], sparse=[], model_id=self.model or "", revision=self.revision)
        if self.local_encoder is not None:
            result = self.local_encoder(texts)
            if len(result.dense) != len(texts):
                raise ValueError(f"local encoder returned {len(result.dense)} vectors for {len(texts)} texts")
            return result
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(
            f"{self.service_url}/embed",
            json={"texts": texts},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        dense = data["dense"]
        if len(dense) != len(texts):
            raise ValueError(f"embedding service returned {len(dense)} vectors for {len(texts)} texts")
        sparse = data.get("sparse") or [None] * len(texts)
        if len(sparse) != len(texts):
            raise ValueError(f"embedding service returned {len(sparse)} sparse vectors for {len(texts)} texts")
        return DualEmbeddingResult(
            dense=dense,
            sparse=sparse,
            model_id=data.get("model_id") or self.model or "",
            revision=data.get("revision") or self.revision,
        )