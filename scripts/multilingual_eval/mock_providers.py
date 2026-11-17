"""Deterministic mock providers for the multilingual evaluation harness.

These mock the *providers* (LLM / embedding service), never the Memory
pipeline under test. They make baseline runs reproducible without network
access; they measure pipeline behavior, not model quality. A real-service
baseline is run by passing a real MemoryConfig dict via run_eval.py --config.
"""

from __future__ import annotations

import hashlib
import math
import re

from mem0.embeddings.base import EmbeddingBase

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def new_messages_key(prompt: str) -> str:
    """Key canned responses by the exact '## New Messages' section content."""
    match = re.search(r"## New Messages\n(.*?)\n\n## ", prompt, re.DOTALL)
    section = match.group(1) if match else prompt
    return hashlib.sha1(section.encode("utf-8")).hexdigest()


class EvalMockLLM:
    """Returns canned extraction JSON registered by the runner.

    Unregistered prompts return an empty extraction and are counted so the
    runner can fail loudly instead of silently scoring an empty pipeline.
    """

    responses: dict = {}
    misses: list = []

    def __init__(self, config=None):
        self.config = config

    def generate_response(self, messages, response_format=None, **kwargs):
        user_prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        key = new_messages_key(user_prompt)
        if key in self.responses:
            return self.responses[key]
        self.misses.append(user_prompt[:200])
        return '{"memory": []}'


class EvalMockEmbedding(EmbeddingBase):
    """Deterministic hashed bag-of-tokens embedding (L2-normalized).

    Tokens: lowercase ASCII words + CJK unigrams and bigrams, so shared
    Chinese substrings raise cosine similarity while unrelated texts stay
    near zero. Cross-lingual pairs share no tokens and score ~0, which is
    exactly the weakness a real multilingual model is expected to fix.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.dims = (getattr(config, "embedding_dims", None) or 256) if config else 256

    def _tokens(self, text: str):
        text = text.lower()
        tokens = _WORD_RE.findall(text)
        cjk = _CJK_RE.findall(text)
        tokens.extend(cjk)
        tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
        return tokens

    def embed(self, text, memory_action=None):
        vec = [0.0] * self.dims
        for token in self._tokens(text):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch_dual(self, texts, memory_action="add"):
        """Deterministic learned-sparse mock: hashed token indices, count weights."""
        dense = [self.embed(t, memory_action) for t in texts]
        sparse = []
        for text in texts:
            counts = {}
            for token in self._tokens(text):
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            sparse.append({"indices": sorted(counts), "values": [counts[i] for i in sorted(counts)]})
        return dense, sparse
