"""ModelProfile registry: per-model retrieval thresholds and normalization params.

Values are measured once per model by evaluation and shipped with releases
(design doc 7.2); deployments do not calibrate per instance. Config-level
overrides are available through ``MemoryConfig.profiles`` keyed by model id.

The default entry reproduces the historical production constants so that
existing configurations behave identically. ``near_duplicate_threshold`` and
``sparse_zero_floor`` are provisional defaults pending dedicated measurement
(translation/negation/number-substitution pairs and no-hit queries); they are
deliberately conservative (few tags, weak sparse signals zeroed).
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel

PAYLOAD_SCHEMA_VERSION = "1"


class ModelProfile(BaseModel):
    semantic_threshold: float = 0.1
    entity_match_threshold: float = 0.95
    entity_vector_threshold: float = 0.5
    near_duplicate_threshold: float = 0.95
    sparse_zero_floor: float = 0.1


DEFAULT_PROFILE = ModelProfile()

_MODEL_PROFILES: Dict[str, ModelProfile] = {
    "text-embedding-3-small": ModelProfile(),
    "text-embedding-3-large": ModelProfile(),
}


def resolve_profile(model_id: Optional[str], overrides: Optional[Dict[str, dict]] = None) -> ModelProfile:
    base = _MODEL_PROFILES.get(model_id or "", DEFAULT_PROFILE)
    override = (overrides or {}).get(model_id or "")
    if not override:
        return base
    return base.model_copy(update=override)
