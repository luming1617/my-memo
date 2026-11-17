"""S2 acceptance tests: ModelProfile registry (design doc 7.2 / 8.2 S2)."""

from mem0.configs.base import MemoryConfig
from mem0.utils.model_profiles import (
    PAYLOAD_SCHEMA_VERSION,
    ModelProfile,
    resolve_profile,
)


class TestModelProfileDefaults:
    def test_current_production_values_preserved(self):
        profile = resolve_profile("text-embedding-3-small")
        assert profile.semantic_threshold == 0.1
        assert profile.entity_match_threshold == 0.95
        assert profile.entity_vector_threshold == 0.5

    def test_unknown_model_falls_back_to_defaults(self):
        assert resolve_profile("some-unregistered-model") == resolve_profile("text-embedding-3-small")

    def test_none_model_falls_back_to_defaults(self):
        assert resolve_profile(None) == ModelProfile()

    def test_schema_version_constant(self):
        assert isinstance(PAYLOAD_SCHEMA_VERSION, str) and PAYLOAD_SCHEMA_VERSION


class TestProfileOverrides:
    def test_partial_override_merges_over_base(self):
        profile = resolve_profile(
            "text-embedding-3-small",
            overrides={"text-embedding-3-small": {"semantic_threshold": 0.35}},
        )
        assert profile.semantic_threshold == 0.35
        assert profile.entity_match_threshold == 0.95

    def test_override_for_other_model_ignored(self):
        profile = resolve_profile(
            "text-embedding-3-small",
            overrides={"other-model": {"semantic_threshold": 0.9}},
        )
        assert profile.semantic_threshold == 0.1

    def test_override_unknown_model(self):
        profile = resolve_profile(
            "bge-m3",
            overrides={"bge-m3": {"semantic_threshold": 0.2, "near_duplicate_threshold": 0.88}},
        )
        assert profile.semantic_threshold == 0.2
        assert profile.near_duplicate_threshold == 0.88

    def test_memory_config_profiles_field(self):
        config = MemoryConfig(profiles={"bge-m3": {"semantic_threshold": 0.2}})
        assert config.profiles["bge-m3"]["semantic_threshold"] == 0.2
        assert MemoryConfig().profiles is None
