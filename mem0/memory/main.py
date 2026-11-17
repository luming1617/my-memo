import asyncio
import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import time
import uuid
import warnings
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
from mem0.exceptions import LLMError, VectorStoreError
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.notices import (
    PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS,
    detect_decay_usage_from_delete,
    detect_decay_usage_from_delete_all,
    detect_scale_threshold_from_add_result,
    detect_scale_threshold_from_top_k,
    detect_temporal_usage_from_metadata,
    detect_temporal_usage_from_search,
    display_decay_usage_notice,
    display_decay_usage_notice_async,
    display_first_run_notice,
    display_first_run_notice_async,
    display_performance_slow_query_notice,
    display_performance_slow_query_notice_async,
    display_scale_threshold_notice,
    display_scale_threshold_notice_async,
    display_temporal_usage_notice,
    display_temporal_usage_notice_async,
    get_decay_feature_error_message,
    get_decay_feature_error_message_async,
    get_temporal_feature_error_message,
    get_temporal_feature_error_message_async,
)
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.entity_dictionary import EntityDictionary
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.model_profiles import PAYLOAD_SCHEMA_VERSION, resolve_profile
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    normalize_sparse_coverage,
    score_and_rank,
)
from mem0.vector_stores.base import VectorStoreBase

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


def _vector_store_list_rows(listed):
    if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list):
        return listed[0]
    if isinstance(listed, (list, tuple)):
        return listed
    return []


def _embedding_model_id(config, embedding_model) -> str:
    model = getattr(getattr(embedding_model, "config", None), "model", None) or (config.embedder.config or {}).get(
        "model"
    ) or "default"
    revision = getattr(embedding_model, "revision", None) or (config.embedder.config or {}).get("revision")
    return f"{config.embedder.provider}:{model}" + (f"@{revision}" if revision else "")


def _resolve_model_profile(config, embedding_model):
    model = getattr(getattr(embedding_model, "config", None), "model", None) or (config.embedder.config or {}).get(
        "model"
    )
    return resolve_profile(model, config.profiles)


def _build_version_stamps(config, embedding_model) -> Dict[str, str]:
    vs_config = config.vector_store.config
    if isinstance(vs_config, dict):
        sparse_encoder_id = vs_config.get("sparse_encoder") or "bm25"
    else:
        sparse_encoder_id = getattr(vs_config, "sparse_encoder", None) or "bm25"
    return {
        "embedder_id": _embedding_model_id(config, embedding_model),
        "sparse_encoder_id": sparse_encoder_id,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
    }


def _check_version_stamps(vector_store, stamps: Dict[str, str]) -> Optional[str]:
    """Sample existing payloads; return a write-block reason on stamp mismatch.

    Legacy payloads without stamps are allowed (migration is out of scope).
    Sampling failures never block writes — the guard only acts on evidence.
    """
    try:
        rows = _vector_store_list_rows(vector_store.list(filters=None, top_k=5))
    except Exception as e:
        logger.warning(f"Version-stamp sampling failed, skipping startup validation: {e}")
        return None
    for row in rows:
        payload = getattr(row, "payload", None) or {}
        for key, value in stamps.items():
            existing = payload.get(key)
            if existing is not None and existing != value:
                return (
                    f"Collection was written with {key}={existing!r} but the current config uses {value!r}. "
                    "Writes are blocked to avoid mixing incompatible vector spaces. "
                    "Use a new collection for the new configuration (existing data is not migrated)."
                )
    return None


_LLM_ENTITY_TYPE_MAP = {
    "person": "PROPER",
    "org": "PROPER",
    "location": "PROPER",
    "brand": "PROPER",
    "work": "PROPER",
    "topic": "TOPIC",
    "concept": "TOPIC",
    "identifier": "IDENTIFIER",
}


def _validate_llm_entities(raw) -> Optional[list]:
    """Normalize LLM-output entities to [{surface, aliases, type}].

    Returns None when the field is missing or malformed so the caller can
    fall back to spaCy for that memory. Unknown types coerce to PROPER;
    surfaces are deduplicated by normalized text.
    """
    if not isinstance(raw, list):
        return None
    out = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            return None
        surface = item.get("surface")
        if not isinstance(surface, str) or not surface.strip():
            return None
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            return None
        aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
        entity_type = _LLM_ENTITY_TYPE_MAP.get(str(item.get("type", "")).lower(), "PROPER")
        key = " ".join(surface.strip().lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append({"surface": surface.strip(), "aliases": aliases, "type": entity_type})
    return out


def _sparse_capable(vector_store) -> bool:
    # Strict identity: the capability must be explicitly declared True by the
    # adapter, not merely truthy (guards against duck-typed accidents).
    return getattr(vector_store, "supports_sparse_vectors", False) is True


def _dual_embedder(embedding_model) -> bool:
    # The dual-output method must be part of the provider's class interface.
    return callable(getattr(type(embedding_model), "embed_batch_dual", None))


def _learned_sparse_mode(vector_store) -> bool:
    return getattr(vector_store, "sparse_encoder", None) == "learned"


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / len(text)


def _should_rerank_conditional(config, query: str, candidates: list) -> bool:
    """Zero-cost rerank triggers: query/candidate language mismatch (payload.lang)
    or low best semantic score. Missing lang data never triggers the lang rule."""
    query_is_cjk = _cjk_ratio(query) > 0.2
    langs = [
        (c.get("metadata") or {}).get("lang")
        for c in candidates
        if isinstance((c.get("metadata") or {}).get("lang"), str)
    ]
    if langs:
        dominant = max(set(langs), key=langs.count)
        if dominant.lower().startswith("zh") != query_is_cjk:
            return True
    threshold = config.rerank_semantic_threshold
    if threshold is not None and candidates:
        best = max(
            (c.get("score_details") or {}).get("semantic_score") or c.get("score") or 0.0 for c in candidates
        )
        if best < threshold:
            return True
    return False


def _fold_near_duplicates(scored_results: list, explain: bool = False) -> list:
    """Cluster results by payload.near_duplicate_of (union-find).

    Representative = newest updated_at (fact updates surface); cluster rank =
    max member score. Applied before top_k truncation so duplicates do not
    squeeze out distinct memories. History is never deleted (lossless).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for result in scored_results:
        target = (result.get("payload") or {}).get("near_duplicate_of")
        if target:
            ra, rb = find(result["id"]), find(str(target))
            if ra != rb:
                parent[ra] = rb

    clusters = {}
    for result in scored_results:
        clusters.setdefault(find(result["id"]), []).append(result)

    folded = []
    for members in clusters.values():
        representative = max(members, key=lambda r: (r.get("payload") or {}).get("updated_at") or "")
        rep = dict(representative)
        rep["score"] = max(r["score"] for r in members)
        if explain and len(members) > 1:
            details = dict(rep.get("score_details") or {})
            details["near_duplicate_cluster"] = {
                "representative": rep["id"],
                "members": [r["id"] for r in members],
                "folded_count": len(members) - 1,
            }
            rep["score_details"] = details
        folded.append(rep)

    folded.sort(key=lambda r: r["score"], reverse=True)
    return folded


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key",
    "secret_key",
    "private_key",
    "access_key",
    "password",
    "credentials",
    "credential",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "session_token",
    "client_secret",
    "auth_client_secret",
    "azure_client_secret",
    "service_account_json",
    "aws_session_token",
})

# Suffixes that indicate a field likely holds a secret value.
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})
DELETE_ALL_BATCH_SIZE = 1000

# Tenant-scoping fields that caller-supplied metadata must never set, on either the
# creation or the update path (issues #4490, #6277, #6655).
_IDENTITY_KEYS = ENTITY_PARAMS | {"actor_id"}


def _strip_identity_keys(
    metadata: Dict[str, Any],
    existing_payload: Dict[str, Any],
    *,
    context: str = "update()",
) -> Dict[str, Any]:
    """Drop identity keys from caller metadata; scope is set by the entity params, not metadata.

    On the update path `existing_payload` carries the memory's current scope, so
    re-sending an identical value is silently accepted; only a changed value warns.
    On the creation path there is no prior payload, so pass an empty dict and every
    identity key present in `metadata` is dropped with a warning.
    """
    clean = {}
    for key, value in metadata.items():
        if key not in _IDENTITY_KEYS:
            clean[key] = value
        elif value != existing_payload.get(key):
            logger.warning(f"{context}: ignoring metadata['{key}'] - identity fields cannot be set through metadata")
    return clean


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value: Optional[Any], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Coerces non-string values (e.g. integer ids) to str
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    if value is None:
        return None
    # Callers commonly pass integer ids (e.g. a database primary key). Coerce
    # to str at this single validation point so scoping stays consistent across
    # add/search/get_all/delete_all instead of crashing on `.strip()`.
    if not isinstance(value, str):
        value = str(value)
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    if any(c.isspace() for c in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _validate_and_trim_search_query(query: str) -> str:
    """
    Validates and normalizes a search query before embedding/vector search.

    Raises:
        ValueError: If query is not a string or is empty/whitespace-only.
    """
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`. Identity
       scope is set from the entity params only; identity keys in `input_metadata` are
       dropped, so freeform metadata cannot place a memory into an unrequested scope.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    # Identity scope is set below from the entity params only. Stripping the keys here
    # stops caller metadata from placing a memory into a scope the caller did not pass,
    # which the re-pins below cannot prevent for a param that was left unset (issue #6655).
    base_metadata_template = (
        _strip_identity_keys(deepcopy(input_metadata), {}, context="add()") if input_metadata else {}
    )
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    # Validate and trim entity IDs
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _escape_scope_value(val: Any) -> str:
    """Escape the structural delimiters of the session scope key."""
    return str(val).replace("%", "%25").replace("&", "%26").replace("=", "%3D")


def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={_escape_scope_value(val)}")
    return "&".join(parts)


def _entity_collection_name(provider: str, collection_name: str) -> str:
    separator = "-" if provider == "s3_vectors" else "_"
    return f"{collection_name}{separator}entities"


def _normalize_expiration_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("expiration_date must be a valid date in YYYY-MM-DD format.") from exc
    raise ValueError("expiration_date must be a date string in YYYY-MM-DD format.")


def _payload_is_expired(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    expiration_date = payload.get("expiration_date")
    if not expiration_date:
        return False
    try:
        return date.fromisoformat(str(expiration_date)) < datetime.now(timezone.utc).date()
    except ValueError:
        return False


setup_config()
logger = logging.getLogger(__name__)

_UNSET = object()
_PROJECT_UPDATE_UNSUPPORTED_ERROR = "Project updates are not supported by the OSS Memory SDK."


class _OSSProject:
    def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(get_decay_feature_error_message("sync", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class _AsyncOSSProject:
    async def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(await get_decay_feature_error_message_async("async", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        self._model_profile = _resolve_model_profile(self.config, self.embedding_model)
        self._version_stamps = _build_version_stamps(self.config, self.embedding_model)
        self._write_block_reason = _check_version_stamps(self.vector_store, self._version_stamps)
        if self._write_block_reason:
            logger.error(self._write_block_reason)

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Entity store is initialized lazily on first use
        self._entity_store = None
        self._entity_dictionary = None

        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        capture_event("mem0.init", self, {"sync_type": "sync"})

    def _guard_write(self):
        if self._write_block_reason:
            raise ValueError(self._write_block_reason)

    def _embed_texts_dual(self, texts, action):
        """Batch embed returning (dense_list, sparse_list_or_None)."""
        if _sparse_capable(self.vector_store) and _dual_embedder(self.embedding_model):
            return self.embedding_model.embed_batch_dual(texts, action)
        return self.embedding_model.embed_batch(texts, action), None

    def _embed_text_dual(self, text, action):
        """Single-text embed returning (dense, sparse_or_None)."""
        if _sparse_capable(self.vector_store) and _dual_embedder(self.embedding_model):
            dense, sparse = self.embedding_model.embed_batch_dual([text], action)
            return dense[0], sparse[0]
        return self.embedding_model.embed(text, action), None

    def _text_for_sparse(self, text):
        # Learned sparse encodes the original text; English lemmatization is
        # only meaningful for the legacy BM25 path.
        if _learned_sparse_mode(self.vector_store):
            return text
        return lemmatize_for_bm25(text)

    def _near_duplicate_tag(self, embedding, filters):
        """Top-1 ANN against the scoped store; tag only, never skip (lossless)."""
        if not self.config.near_duplicate:
            return {}
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            matches = self.vector_store.search(query="", vectors=embedding, top_k=1, filters=search_filters)
            if matches and matches[0].score >= self._model_profile.near_duplicate_threshold:
                return {
                    "near_duplicate_of": str(matches[0].id),
                    "near_duplicate_similarity": float(matches[0].score),
                }
        except Exception as e:
            logger.debug(f"Near-duplicate check failed: {e}")
        return {}

    @property
    def project(self):
        return _OSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            # Set collection name on the cloned config
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # Derived entity index: no sparse slot (rebuildable from payloads)
            if hasattr(entity_config, 'sparse_encoder'):
                entity_config.sparse_encoder = "none"
            elif isinstance(entity_config, dict) and 'sparse_encoder' in entity_config:
                entity_config['sparse_encoder'] = "none"
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    @property
    def entity_dictionary(self):
        """Lazily initialized scope-aware entity dictionary cache."""
        if self._entity_dictionary is None:
            self._entity_dictionary = EntityDictionary(loader=self._load_entity_rows)
        return self._entity_dictionary

    def _load_entity_rows(self, search_filters):
        listed = self.entity_store.list(filters=search_filters, top_k=10000)
        return _vector_store_list_rows(listed)

    def _invalidate_entity_dictionary(self):
        if self._entity_dictionary is not None:
            self._entity_dictionary.invalidate()

    @staticmethod
    def _normalize_entity_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _existing_entities_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.entity_store.list(filters=filters, top_k=10000)
        except Exception as e:
            logger.debug(f"Exact entity lookup failed, falling back to semantic dedup: {e}")
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self._normalize_entity_text(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
            for alias in payload.get("aliases") or []:
                if isinstance(alias, str):
                    alias_key = self._normalize_entity_text(alias)
                    if alias_key and alias_key not in rows_by_text:
                        rows_by_text[alias_key] = row
        return rows_by_text

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters, aliases=None):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            exact_match = self._existing_entities_by_text(search_filters).get(self._normalize_entity_text(entity_text))

            existing = []
            if exact_match is None:
                existing = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=1,
                    filters=search_filters,
                )

            semantic_match = (
                existing[0]
                if existing and existing[0].score >= self._model_profile.entity_match_threshold
                else None
            )
            match = exact_match or semantic_match
            if match:
                # Update existing entity's linked_memory_ids and aliases
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                changed = False
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    changed = True
                if aliases:
                    merged = set(payload.get("aliases") or []) | set(aliases)
                    merged.discard(payload.get("data"))
                    if merged != set(payload.get("aliases") or []):
                        payload["aliases"] = sorted(merged)
                        changed = True
                if changed:
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "aliases": sorted(set(aliases or [])),
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
            self._invalidate_entity_dictionary()
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters, delete_empty=True):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record
            (or keep it with an empty list when delete_empty=False, so the
            update relink fallback can still find and re-append to it).
          - otherwise re-embed the entity text and update the payload
            (the vector store's update() requires a vector).

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        if delete_empty:
                            try:
                                self.entity_store.delete(vector_id=row.id)
                            except Exception as e:
                                logger.debug(f"Entity delete failed for id={row.id}: {e}")
                        else:
                            try:
                                self.entity_store.update(
                                    vector_id=row.id,
                                    vector=None,
                                    payload={**payload, "linked_memory_ids": []},
                                )
                            except Exception as e:
                                logger.debug(f"Entity empty-link update failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
            self._invalidate_entity_dictionary()
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters, embedding=None):
        """Link `memory_id` to entities for `text`, scoped to `filters`.

        entity_source=llm: match existing entity records only (whole-text
        vector fallback; the dictionary layer joins in S7). No spaCy/LLM and
        no new entity creation on this path.
        entity_source=spacy: legacy per-entity search-then-update-or-insert
        via `_upsert_entity`. Non-fatal on any failure.
        """
        if self.config.entity_source == "llm":
            self._relink_entities_by_vector(memory_id, text, filters, embedding)
            return
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = self._normalize_entity_text(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    def _relink_entities_by_vector(self, memory_id, text, filters, embedding=None):
        """Whole-text vector fallback: append memory_id to matching entity
        records (payload-only, entities are not re-embedded). Gated by
        Profile.entity_vector_threshold. Non-fatal."""
        try:
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            if embedding is None:
                embedding = self.embedding_model.embed(text, "update")
            matches = self.entity_store.search(
                query=text, vectors=embedding, top_k=10, filters=search_filters
            )
            for match in matches:
                if match.score < self._model_profile.entity_vector_threshold:
                    continue
                payload = match.payload or {}
                linked = payload.get("linked_memory_ids", [])
                if not isinstance(linked, list) or memory_id in linked:
                    continue
                payload["linked_memory_ids"] = linked + [memory_id]
                try:
                    self.entity_store.update(vector_id=match.id, vector=None, payload=payload)
                except Exception as e:
                    logger.debug(f"Entity relink update failed for id={match.id}: {e}")
        except Exception as e:
            logger.warning(f"Entity vector relink failed for memory_id={memory_id}: {e}")

    def rebuild_entity_index(self, filters=None):
        """Rebuild the derived entity index from memory payloads (zero LLM calls).

        payload.entities is the source of truth; memories without it fall
        back to spaCy extraction. The entity collection is reset and
        recreated from the current derived configuration (no sparse slot).
        """
        search_filters = {
            k: v for k, v in (filters or {}).items() if k in ("user_id", "agent_id", "run_id") and v
        }
        store = self.entity_store
        try:
            store.reset()
        except Exception as e:
            logger.warning(f"Entity index reset failed during rebuild: {e}")
        rows = _vector_store_list_rows(self.vector_store.list(filters=search_filters or None, top_k=10000))
        entity_links = 0
        for row in rows:
            payload = getattr(row, "payload", None) or {}
            memory_id = str(row.id)
            entities = payload.get("entities")
            if entities is None:
                entities = [
                    {"surface": surface, "aliases": [], "type": entity_type}
                    for entity_type, surface in extract_entities(payload.get("data", ""))
                ]
            for ent in entities:
                self._upsert_entity(
                    ent["surface"], ent["type"], memory_id, search_filters, aliases=ent.get("aliases")
                )
                entity_links += 1
        self._invalidate_entity_dictionary()
        return {"rebuilt_memories": len(rows), "entity_links": entity_links}

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            timestamp (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            expiration_date (Any, optional): Date in YYYY-MM-DD format. Expired memories are hidden
                from search and get_all unless show_expired is True.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.

        Note:
            `search()` and `get_all()` scope queries via `filters={"user_id": "...", "agent_id": "...", "run_id": "..."}` —
            they reject top-level `user_id`/`agent_id`/`run_id` arguments. `add()` accepts them top-level, but passing
            the same arguments to `search()`/`get_all()` raises a `ValueError`; use the `filters` form there instead.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """
        if timestamp is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "add", "timestamp"))

        normalized_expiration_date = _normalize_expiration_date(expiration_date)
        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        if normalized_expiration_date is not None:
            processed_metadata["expiration_date"] = normalized_expiration_date

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # TODO: Remove procedural-memory support in a future breaking release.
        # Remove memory_type and its helpers from Memory and AsyncMemory together.
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            scale_threshold_notice = detect_scale_threshold_from_add_result(self, results)
            if temporal_usage_notice:
                display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
            else:
                display_first_run_notice(self, "sync", "add")
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        scale_threshold_notice = detect_scale_threshold_from_add_result(self, vector_store_result)
        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "add")
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        self._guard_write()
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings, msg_sparse = self._embed_text_dual(msg_content, "add")
                mem_id = self._create_memory(
                    msg_content, {msg_content: msg_embeddings}, per_msg_meta, existing_sparse={msg_content: msg_sparse}
                )

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
            use_input_language=self.config.multilingual,
            include_metadata=self.config.multilingual or self.config.entity_source == "llm",
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            # Re-raise so callers can implement provider fallback / retry.
            # The original silent ``return []`` made upstream callers unable to
            # distinguish "LLM unavailable" (429/5xx/timeout) from "LLM
            # extracted no facts" -- both surfaced as an empty list.
            logger.error(f"LLM extraction failed: {e}")
            raise LLMError(f"LLM extraction failed: {e}") from e

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            extracted_memories = []

        if not extracted_memories:
            # Save messages even if nothing extracted
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        sparse_map = {}
        try:
            mem_embeddings_list, sparse_list = self._embed_texts_dual(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
            if sparse_list is not None:
                sparse_map = dict(zip(mem_texts, sparse_list))
        except Exception:
            # Fallback: embed individually
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build set of existing hashes for dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []  # (memory_id, text, embedding, payload)
        seen_hashes = set()  # dedup within the current batch
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = self._text_for_sparse(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]
            if self.config.multilingual or self.config.entity_source == "llm":
                if isinstance(mem.get("lang"), str) and mem["lang"].strip():
                    mem_metadata["lang"] = mem["lang"].strip()
                keywords = mem.get("keywords")
                if isinstance(keywords, list):
                    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
                    if keywords:
                        mem_metadata["keywords"] = keywords
                links = mem.get("linked_memory_ids")
                if isinstance(links, list):
                    resolved = [uuid_mapping[str(link)] for link in links if str(link) in uuid_mapping]
                    if resolved:
                        mem_metadata["linked_memory_ids"] = resolved
            if self.config.entity_source == "llm":
                validated_entities = _validate_llm_entities(mem.get("entities"))
                if validated_entities is None:
                    validated_entities = [
                        {"surface": surface, "aliases": [], "type": entity_type}
                        for entity_type, surface in extract_entities(text)
                    ]
                mem_metadata["entities"] = validated_entities
            mem_metadata.update(self._version_stamps)

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 5.5: Lossless near-duplicate tagging (full-store top-1 ANN,
        # reusing Phase 3 vectors; tagged records are still inserted).
        if self.config.near_duplicate:
            try:
                try:
                    match_lists = self.vector_store.search_batch(
                        queries=[r[1] for r in records],
                        vectors_list=[r[2] for r in records],
                        top_k=1,
                        filters=search_filters,
                    )
                except Exception:
                    match_lists = [
                        self.vector_store.search(query=r[1], vectors=r[2], top_k=1, filters=search_filters)
                        for r in records
                    ]
                for rec, matches in zip(records, match_lists):
                    if matches and matches[0].score >= self._model_profile.near_duplicate_threshold:
                        rec[3]["near_duplicate_of"] = str(matches[0].id)
                        rec[3]["near_duplicate_similarity"] = float(matches[0].score)
            except Exception as e:
                logger.debug(f"Near-duplicate tagging failed: {e}")

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        # Only records confirmed to be stored make it into history, entity
        # links, and the returned results — a record the store rejected must
        # never be reported back as a successful ADD.
        persisted_records = []
        insert_kwargs = {}
        if sparse_map and _sparse_capable(self.vector_store):
            insert_kwargs["sparse_vectors"] = [sparse_map.get(r[1]) for r in records]
        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
                **insert_kwargs,
            )
            persisted_records = records
        except Exception:
            # Fallback: insert one by one
            for rec in records:
                try:
                    rec_kwargs = (
                        {"sparse_vectors": [sparse_map.get(rec[1])]}
                        if sparse_map and _sparse_capable(self.vector_store)
                        else {}
                    )
                    self.vector_store.insert(vectors=[rec[2]], ids=[rec[0]], payloads=[rec[3]], **rec_kwargs)
                    persisted_records.append(rec)
                except Exception as e:
                    logger.error(f"Failed to insert memory {rec[0]}: {e}")

        if not persisted_records:
            self.db.save_messages(messages, session_scope)
            raise VectorStoreError(
                f"Failed to insert any of the {len(records)} extracted memories into the vector store"
            )

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in persisted_records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            # Fallback: add one by one
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in persisted_records]
            if self.config.entity_source == "llm":
                all_entities = [
                    [(e["type"], e["surface"], e.get("aliases") or []) for e in r[3].get("entities", [])]
                    for r in persisted_records
                ]
            else:
                all_entities = [[(t, s, []) for t, s in ents] for ents in extract_entities_batch(all_texts)]

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> [entity_type, entity_text, set of memory_ids, aliases]
            for idx, (memory_id, text, embedding, payload) in enumerate(persisted_records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text, aliases in entities:
                    key = self._normalize_entity_text(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                        global_entities[key][3] = sorted(set(global_entities[key][3]) | set(aliases))
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}, sorted(set(aliases))]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)


                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                # Filter out entities with failed embeddings
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    exact_matches = self._existing_entities_by_text(search_filters)

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids, aliases = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        exact_match = exact_matches.get(key)

                        semantic_match = (
                            matches[0]
                            if matches and matches[0].score >= self._model_profile.entity_match_threshold
                            else None
                        )
                        match = exact_match or semantic_match
                        if match:
                            # Update existing entity
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            if aliases:
                                merged_aliases = set(payload.get("aliases") or []) | set(aliases)
                                merged_aliases.discard(payload.get("data"))
                                payload["aliases"] = sorted(merged_aliases)
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "aliases": aliases,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
            self._invalidate_entity_dictionary()
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in persisted_records
        ]

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            display_first_run_notice(self, "sync", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        display_first_run_notice(self, "sync", "get")
        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        fetch_limit = limit if show_expired else max(limit * 4, 60)
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        all_memories_result = self._get_all_from_vector_store(effective_filters, fetch_limit, show_expired, limit)

        if scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "get_all", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "get_all")
        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit, show_expired=False, output_limit=None):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            if not show_expired and _payload_is_expired(mem.payload):
                continue
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)
            if output_limit is not None and len(formatted_memories) >= output_limit:
                break

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to the
                embedding model profile's semantic_threshold (0.1 for the default profile).
            rerank (bool, optional): Whether to rerank results. Defaults to False.
            explain (bool, optional): Whether to include score_details for each result. Defaults to False.
            reference_date (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        if reference_date is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "search", "reference_date"))

        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        search_start = time.perf_counter()
        original_memories = self._search_vector_store(
            query, effective_filters, limit, threshold, explain=explain, show_expired=show_expired
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # Apply reranking if enabled and reranker is available (top-N scoped,
        # optional conditional trigger; failure falls back to original results)
        if rerank and self.reranker and original_memories:
            candidates = original_memories[: self.config.rerank_top_n]
            should_rerank = self.config.rerank_trigger != "conditional" or _should_rerank_conditional(
                self.config, query, candidates
            )
            if should_rerank:
                try:
                    reranked_memories = self.reranker.rerank(query, candidates, limit)
                    original_memories = (reranked_memories + original_memories[len(candidates):])[:limit]
                except Exception as e:
                    logger.warning(f"Reranking failed, using original results: {e}")

        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            display_performance_slow_query_notice(
                self,
                "sync",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            display_first_run_notice(self, "sync", "search")
        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold=None, explain=False, show_expired=False):
        # Resolve the semantic threshold from the model profile when not given explicitly
        if threshold is None:
            threshold = self._model_profile.semantic_threshold

        # Step 1: Preprocess query
        query_lemmatized = self._text_for_sparse(query)
        use_dictionary = self.config.entity_source == "llm" or self.config.multilingual
        query_entities = [] if use_dictionary else extract_entities(query)

        # Step 2: Embed query (one forward pass yields dense + sparse when available)
        embeddings, query_sparse = self._embed_text_dual(query, "search")

        # Step 3: Semantic search (over-fetch for scoring pool)
        internal_limit = max(limit * 4, 60)
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        learned_mode = _learned_sparse_mode(self.vector_store) and query_sparse is not None
        if learned_mode:
            keyword_results = self.vector_store.keyword_search(
                query=query, top_k=internal_limit, filters=filters, sparse_vector=query_sparse
            )
        else:
            keyword_results = self.vector_store.keyword_search(
                query=query_lemmatized, top_k=internal_limit, filters=filters
            )

        # Step 5: Compute normalized keyword scores from keyword results
        bm25_scores = {}
        if keyword_results is not None and learned_mode:
            query_self_score = sum(v * v for v in query_sparse["values"])
            zero_floor = self._model_profile.sparse_zero_floor
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    normalized = normalize_sparse_coverage(raw_score, query_self_score, zero_floor)
                    if normalized > 0:
                        bm25_scores[mem_id] = normalized
        elif keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if use_dictionary:
            entity_boosts = self._compute_entity_boosts_multilingual(query, filters, embeddings)
        elif query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            payload = mem.payload if hasattr(mem, 'payload') else {}
            if not show_expired and _payload_is_expired(payload):
                continue
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Step 8: Score and rank (fold before top_k truncation when enabled)
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=len(candidates) if self.config.near_duplicate else limit,
            explain=explain,
        )
        if self.config.near_duplicate:
            scored_results = _fold_near_duplicates(scored_results, explain=explain)[:limit]

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("data"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self._normalize_entity_text(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = self.embedding_model.embed_batch(entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            entity_store = self.entity_store

            def _search_entity(entity_text, embedding):
                return entity_store.search(
                    query=entity_text, vectors=embedding, top_k=500, filters=search_filters
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(_search_entity, text, emb): text
                    for text, emb in zip(entity_texts, embeddings)
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        matches = future.result()
                    except Exception as e:
                        logger.warning("Entity boost search failed for one entity: %s", e)
                        continue

                    for match in matches:
                        similarity = match.score if hasattr(match, 'score') else 0.0
                        if similarity < 0.5:
                            continue

                        payload = match.payload if hasattr(match, 'payload') else {}
                        linked_memory_ids = payload.get("linked_memory_ids", [])
                        if not isinstance(linked_memory_ids, list):
                            continue

                        num_linked = max(len(linked_memory_ids), 1)
                        memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                        boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                        for memory_id in linked_memory_ids:
                            if memory_id:
                                memory_key = str(memory_id)
                                memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def _entity_vector_matches(self, embedding, filters, top_k=500):
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        return self.entity_store.search(query="", vectors=embedding, top_k=top_k, filters=search_filters)

    def _compute_entity_boosts_multilingual(self, query, filters, query_embedding):
        """Two-layer entity boosts: alias dictionary first, whole-query vector
        fallback second (design doc 4.3). Boost formula identical to the legacy
        path; dictionary hits score sim=1.0, fallback hits score their cosine.
        """
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        def _apply(linked_memory_ids, similarity):
            if not isinstance(linked_memory_ids, list):
                return
            num_linked = max(len(linked_memory_ids), 1)
            memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
            boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
            for memory_id in linked_memory_ids:
                if memory_id:
                    memory_key = str(memory_id)
                    memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        try:
            for row in self.entity_dictionary.match(query, search_filters):
                payload = getattr(row, "payload", None) or {}
                _apply(payload.get("linked_memory_ids", []), 1.0)
        except Exception as e:
            logger.warning(f"Entity dictionary match failed: {e}")

        try:
            for match in self._entity_vector_matches(query_embedding, filters):
                similarity = match.score if hasattr(match, "score") else 0.0
                if similarity < self._model_profile.entity_vector_threshold:
                    continue
                payload = match.payload if hasattr(match, "payload") else {}
                _apply(payload.get("linked_memory_ids", []), similarity)
        except Exception as e:
            logger.warning(f"Entity vector fallback failed: {e}")

        return memory_boosts

    def update(
        self,
        memory_id,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_date: Any = _UNSET,
        data: Optional[str] = None,
    ):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            text (str, optional): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.
                ``user_id``/``agent_id``/``run_id``/``actor_id`` are ignored here - they are
                immutable after creation.
            expiration_date (Any, optional): Date in YYYY-MM-DD format, or None to clear it.
            data (str, optional): Deprecated alias for ``text``. Will be removed in the next
                major release; use ``text`` instead.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", text="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        if data is not None:
            logger.warning(
                "The `data` argument to update() is deprecated and will be removed in the "
                "next major release. Use `text` instead."
            )
            if text is None:
                text = data

        if text is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of text, metadata, or expiration_date must be provided.")

        update_metadata = deepcopy(metadata) if metadata is not None else None
        if expiration_date is not _UNSET:
            update_metadata = update_metadata or {}
            update_metadata["expiration_date"] = _normalize_expiration_date(expiration_date)

        existing_embeddings = {}
        existing_sparse = {}
        if text is not None:
            dense, sparse = self._embed_text_dual(text, "update")
            existing_embeddings[text] = dense
            existing_sparse[text] = sparse

        sparse_arg = existing_sparse if any(v is not None for v in existing_sparse.values()) else None
        if sparse_arg is not None:
            self._update_memory(memory_id, text, existing_embeddings, update_metadata, existing_sparse=sparse_arg)
        else:
            self._update_memory(memory_id, text, existing_embeddings, update_metadata)
        display_first_run_notice(self, "sync", "update")
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete")
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # Keep listing after each batch is deleted. Most vector stores cap
        # list() at 100 results by default, which silently truncates deletes.
        deleted_count = 0
        seen_batches = set()
        while True:
            memories = self.vector_store.list(
                filters=filters, top_k=DELETE_ALL_BATCH_SIZE
            )[0]
            if not memories:
                break
            batch_ids = tuple(sorted(str(memory.id) for memory in memories))
            if batch_ids in seen_batches:
                logger.warning("Stopping delete_all after a repeated memory batch")
                break
            seen_batches.add(batch_ids)
            for memory in memories:
                self._delete_memory(memory.id)
            deleted_count += len(memories)

        logger.info(f"Deleted {deleted_count} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(deleted_count)
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete_all", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete_all")
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        history = self.db.get_history(memory_id)
        display_first_run_notice(self, "sync", "history")
        return history

    def _create_memory(self, data, existing_embeddings, metadata=None, existing_sparse=None):
        logger.debug(f"Creating memory with {data=}")
        self._guard_write()
        sparse_vector = (existing_sparse or {}).get(data)
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings, sparse_vector = self._embed_text_dual(data, "add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = self._text_for_sparse(data)
        new_metadata.update(self._version_stamps)
        new_metadata.update(self._near_duplicate_tag(embeddings, new_metadata))

        insert_kwargs = (
            {"sparse_vectors": [sparse_vector]}
            if sparse_vector is not None and _sparse_capable(self.vector_store)
            else {}
        )
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
            **insert_kwargs,
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if not procedural_memory:
            raise ValueError(
                "The LLM returned no content for the procedural memory summary. "
                "The model may have declined the request or returned an empty response."
            )

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None, existing_sparse=None):
        logger.info(f"Updating memory with {data=}")
        self._guard_write()

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            # Backing-store failure, not a bad memory_id: re-raise the original so the REST layer maps it to 5xx, not 4xx.
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")
        if data is None:
            data = prev_value
        if not isinstance(data, str):
            raise ValueError(f"Memory with id {memory_id} does not have text content to update")
        text_changed = data != prev_value

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(_strip_identity_keys(metadata, existing_memory.payload))

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = self._text_for_sparse(data)
        new_metadata.update(self._version_stamps)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
            sparse_vector = (existing_sparse or {}).get(data)
        else:
            embeddings, sparse_vector = self._embed_text_dual(data, "update")

        update_kwargs = (
            {"sparse_vector": sparse_vector}
            if sparse_vector is not None and _sparse_capable(self.vector_store)
            else {}
        )
        updated = self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
            **update_kwargs,
        )
        if updated is False:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        if text_changed:
            self._remove_memory_from_entity_store(memory_id, session_filters, delete_empty=False)
            self._link_entities_for_memory(memory_id, data, session_filters, embedding=embeddings)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        self.db.reset()
        self.db.close()
        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "sync"})
        display_first_run_notice(self, "sync", "reset")

    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions
        self._entity_store = None
        self._entity_dictionary = None

        self._model_profile = _resolve_model_profile(self.config, self.embedding_model)
        self._version_stamps = _build_version_stamps(self.config, self.embedding_model)
        self._write_block_reason = _check_version_stamps(self.vector_store, self._version_stamps)
        if self._write_block_reason:
            logger.error(self._write_block_reason)

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        if MEM0_TELEMETRY:
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            telemetry_config.collection_name = "mem0migrations"
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config.path, exist_ok=True)
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        capture_event("mem0.init", self, {"sync_type": "async"})

    def _guard_write(self):
        if self._write_block_reason:
            raise ValueError(self._write_block_reason)

    def _embed_texts_dual(self, texts, action):
        """Batch embed returning (dense_list, sparse_list_or_None)."""
        if _sparse_capable(self.vector_store) and _dual_embedder(self.embedding_model):
            return self.embedding_model.embed_batch_dual(texts, action)
        return self.embedding_model.embed_batch(texts, action), None

    def _embed_text_dual(self, text, action):
        """Single-text embed returning (dense, sparse_or_None)."""
        if _sparse_capable(self.vector_store) and _dual_embedder(self.embedding_model):
            dense, sparse = self.embedding_model.embed_batch_dual([text], action)
            return dense[0], sparse[0]
        return self.embedding_model.embed(text, action), None

    def _text_for_sparse(self, text):
        if _learned_sparse_mode(self.vector_store):
            return text
        return lemmatize_for_bm25(text)

    async def _near_duplicate_tag_async(self, embedding, filters):
        """Async variant of `_near_duplicate_tag`."""
        if not self.config.near_duplicate:
            return {}
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            matches = await asyncio.to_thread(
                self.vector_store.search, query="", vectors=embedding, top_k=1, filters=search_filters
            )
            if matches and matches[0].score >= self._model_profile.near_duplicate_threshold:
                return {
                    "near_duplicate_of": str(matches[0].id),
                    "near_duplicate_similarity": float(matches[0].score),
                }
        except Exception as e:
            logger.debug(f"Near-duplicate check failed (async): {e}")
        return {}

    @property
    def project(self):
        return _AsyncOSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # Derived entity index: no sparse slot (rebuildable from payloads)
            if hasattr(entity_config, 'sparse_encoder'):
                entity_config.sparse_encoder = "none"
            elif isinstance(entity_config, dict) and 'sparse_encoder' in entity_config:
                entity_config['sparse_encoder'] = "none"
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    @property
    def entity_dictionary(self):
        """Lazily initialized scope-aware entity dictionary cache."""
        if self._entity_dictionary is None:
            self._entity_dictionary = EntityDictionary(loader=self._load_entity_rows)
        return self._entity_dictionary

    def _load_entity_rows(self, search_filters):
        listed = self.entity_store.list(filters=search_filters, top_k=10000)
        return _vector_store_list_rows(listed)

    def _invalidate_entity_dictionary(self):
        if self._entity_dictionary is not None:
            self._entity_dictionary.invalidate()

    @staticmethod
    def _normalize_entity_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _existing_entities_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.entity_store.list(filters=filters, top_k=10000)
        except Exception as e:
            logger.debug(f"Exact entity lookup failed, falling back to semantic dedup: {e}")
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self._normalize_entity_text(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
            for alias in payload.get("aliases") or []:
                if isinstance(alias, str):
                    alias_key = self._normalize_entity_text(alias)
                    if alias_key and alias_key not in rows_by_text:
                        rows_by_text[alias_key] = row
        return rows_by_text

    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters, aliases=None):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        try:
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            exact_match = (
                await asyncio.to_thread(self._existing_entities_by_text, search_filters)
            ).get(self._normalize_entity_text(entity_text))

            existing = []
            if exact_match is None:
                existing = await asyncio.to_thread(
                    self.entity_store.search,
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=1,
                    filters=search_filters,
                )

            semantic_match = (
                existing[0]
                if existing and existing[0].score >= self._model_profile.entity_match_threshold
                else None
            )
            match = exact_match or semantic_match
            if match:
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                changed = False
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    changed = True
                if aliases:
                    merged = set(payload.get("aliases") or []) | set(aliases)
                    merged.discard(payload.get("data"))
                    if merged != set(payload.get("aliases") or []):
                        payload["aliases"] = sorted(merged)
                        changed = True
                if changed:
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "aliases": sorted(set(aliases or [])),
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                await asyncio.to_thread(
                    self.entity_store.insert,
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
            await asyncio.to_thread(self._invalidate_entity_dictionary)
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    async def _bulk_clear_entity_store(self, filters):
        """Delete all entity records matching the given scope filters.

        Used by delete_all to avoid the race condition that occurs when
        concurrent _delete_memory coroutines each try to read-modify-write
        the same entity rows' linked_memory_ids lists.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                except Exception as e:
                    logger.debug(f"Bulk entity delete failed for id={row.id}: {e}")
            await asyncio.to_thread(self._invalidate_entity_dictionary)
        except Exception as e:
            logger.warning(f"Bulk entity store cleanup failed: {e}")

    async def _remove_memory_from_entity_store(self, memory_id, filters, delete_empty=True):
        """Async variant of `Memory._remove_memory_from_entity_store`."""
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        if delete_empty:
                            try:
                                await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                            except Exception as e:
                                logger.debug(f"Entity delete failed for id={row.id} (async): {e}")
                        else:
                            try:
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=row.id,
                                    vector=None,
                                    payload={**payload, "linked_memory_ids": []},
                                )
                            except Exception as e:
                                logger.debug(f"Entity empty-link update failed for id={row.id} (async): {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup (async)")
                            continue
                        try:
                            vec = await asyncio.to_thread(self.embedding_model.embed, entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}' (async): {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            await asyncio.to_thread(
                                self.entity_store.update,
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id} (async): {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error (async): {e}")
            await asyncio.to_thread(self._invalidate_entity_dictionary)
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id} (async): {e}")

    async def _link_entities_for_memory(self, memory_id, text, filters, embedding=None):
        """Async variant of `Memory._link_entities_for_memory`."""
        if self.config.entity_source == "llm":
            await self._relink_entities_by_vector_async(memory_id, text, filters, embedding)
            return
        try:
            entities = await asyncio.to_thread(extract_entities, text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = self._normalize_entity_text(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    async def _relink_entities_by_vector_async(self, memory_id, text, filters, embedding=None):
        """Async variant of `Memory._relink_entities_by_vector`."""
        try:
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            if embedding is None:
                embedding = await asyncio.to_thread(self.embedding_model.embed, text, "update")
            matches = await asyncio.to_thread(
                self.entity_store.search, query=text, vectors=embedding, top_k=10, filters=search_filters
            )
            for match in matches:
                if match.score < self._model_profile.entity_vector_threshold:
                    continue
                payload = match.payload or {}
                linked = payload.get("linked_memory_ids", [])
                if not isinstance(linked, list) or memory_id in linked:
                    continue
                payload["linked_memory_ids"] = linked + [memory_id]
                try:
                    await asyncio.to_thread(
                        self.entity_store.update, vector_id=match.id, vector=None, payload=payload
                    )
                except Exception as e:
                    logger.debug(f"Entity relink update failed for id={match.id} (async): {e}")
        except Exception as e:
            logger.warning(f"Entity vector relink failed for memory_id={memory_id} (async): {e}")

    async def rebuild_entity_index(self, filters=None):
        """Async variant of `Memory.rebuild_entity_index` (zero LLM calls)."""
        search_filters = {
            k: v for k, v in (filters or {}).items() if k in ("user_id", "agent_id", "run_id") and v
        }
        store = self.entity_store
        try:
            await asyncio.to_thread(store.reset)
        except Exception as e:
            logger.warning(f"Entity index reset failed during rebuild (async): {e}")
        rows = _vector_store_list_rows(
            await asyncio.to_thread(self.vector_store.list, filters=search_filters or None, top_k=10000)
        )
        entity_links = 0
        for row in rows:
            payload = getattr(row, "payload", None) or {}
            memory_id = str(row.id)
            entities = payload.get("entities")
            if entities is None:
                entities = [
                    {"surface": surface, "aliases": [], "type": entity_type}
                    for entity_type, surface in await asyncio.to_thread(extract_entities, payload.get("data", ""))
                ]
            for ent in entities:
                await self._upsert_entity_async(
                    ent["surface"], ent["type"], memory_id, search_filters, aliases=ent.get("aliases")
                )
                entity_links += 1
        self._invalidate_entity_dictionary()
        return {"rebuilt_memories": len(rows), "entity_links": entity_links}

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            timestamp (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            expiration_date (Any, optional): Date in YYYY-MM-DD format. Expired memories are hidden
                from search and get_all unless show_expired is True.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.

        Note:
            `search()` and `get_all()` scope queries via `filters={"user_id": "...", "agent_id": "...", "run_id": "..."}` —
            they reject top-level `user_id`/`agent_id`/`run_id` arguments. `add()` accepts them top-level, but passing
            the same arguments to `search()`/`get_all()` raises a `ValueError`; use the `filters` form there instead.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        if timestamp is not None:
            raise ValueError(await get_temporal_feature_error_message_async("async", "add", "timestamp"))

        normalized_expiration_date = _normalize_expiration_date(expiration_date)
        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )
        if normalized_expiration_date is not None:
            processed_metadata["expiration_date"] = normalized_expiration_date

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # TODO: Remove procedural-memory support in a future breaking release.
        # Remove memory_type and its helpers from Memory and AsyncMemory together.
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, results)
            if temporal_usage_notice:
                await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
            else:
                await display_first_run_notice_async(self, "async", "add")
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = await self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, vector_store_result)
        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "add")
        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        prompt: Optional[str] = None,
    ):
        self._guard_write()
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings, msg_sparse = await asyncio.to_thread(self._embed_text_dual, msg_content, "add")
                mem_id = await self._create_memory(
                    msg_content, {msg_content: msg_embeddings}, per_msg_meta, existing_sparse={msg_content: msg_sparse}
                )

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(effective_filters)
        last_messages = await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
            use_input_language=self.config.multilingual,
            include_metadata=self.config.multilingual or self.config.entity_source == "llm",
        )

        try:
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            # Re-raise so callers can implement provider fallback / retry
            # (see sync counterpart for rationale).
            logger.error(f"LLM extraction failed (async): {e}")
            raise LLMError(f"LLM extraction failed: {e}") from e

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response (async): {e}")
            extracted_memories = []

        if not extracted_memories:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        sparse_map = {}
        try:
            mem_embeddings_list, sparse_list = await asyncio.to_thread(self._embed_texts_dual, mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
            if sparse_list is not None:
                sparse_map = dict(zip(mem_texts, sparse_list))
        except Exception:
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []
        seen_hashes = set()
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = self._text_for_sparse(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]
            if self.config.multilingual or self.config.entity_source == "llm":
                if isinstance(mem.get("lang"), str) and mem["lang"].strip():
                    mem_metadata["lang"] = mem["lang"].strip()
                keywords = mem.get("keywords")
                if isinstance(keywords, list):
                    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
                    if keywords:
                        mem_metadata["keywords"] = keywords
                links = mem.get("linked_memory_ids")
                if isinstance(links, list):
                    resolved = [uuid_mapping[str(link)] for link in links if str(link) in uuid_mapping]
                    if resolved:
                        mem_metadata["linked_memory_ids"] = resolved
            if self.config.entity_source == "llm":
                validated_entities = _validate_llm_entities(mem.get("entities"))
                if validated_entities is None:
                    validated_entities = [
                        {"surface": surface, "aliases": [], "type": entity_type}
                        for entity_type, surface in extract_entities(text)
                    ]
                mem_metadata["entities"] = validated_entities
            mem_metadata.update(self._version_stamps)

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 5.5: Lossless near-duplicate tagging (async)
        if self.config.near_duplicate:
            try:
                try:
                    match_lists = await asyncio.to_thread(
                        self.vector_store.search_batch,
                        queries=[r[1] for r in records],
                        vectors_list=[r[2] for r in records],
                        top_k=1,
                        filters=search_filters,
                    )
                except Exception:
                    match_lists = []
                    for r in records:
                        match_lists.append(
                            await asyncio.to_thread(
                                self.vector_store.search, query=r[1], vectors=r[2], top_k=1, filters=search_filters
                            )
                        )
                for rec, matches in zip(records, match_lists):
                    if matches and matches[0].score >= self._model_profile.near_duplicate_threshold:
                        rec[3]["near_duplicate_of"] = str(matches[0].id)
                        rec[3]["near_duplicate_similarity"] = float(matches[0].score)
            except Exception as e:
                logger.debug(f"Near-duplicate tagging failed (async): {e}")

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        # Only records confirmed to be stored make it into history, entity
        # links, and the returned results — a record the store rejected must
        # never be reported back as a successful ADD.
        persisted_records = []
        insert_kwargs = {}
        if sparse_map and _sparse_capable(self.vector_store):
            insert_kwargs["sparse_vectors"] = [sparse_map.get(r[1]) for r in records]
        try:
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
                **insert_kwargs,
            )
            persisted_records = records
        except Exception:
            for rec in records:
                try:
                    rec_kwargs = (
                        {"sparse_vectors": [sparse_map.get(rec[1])]}
                        if sparse_map and _sparse_capable(self.vector_store)
                        else {}
                    )
                    await asyncio.to_thread(
                        self.vector_store.insert, vectors=[rec[2]], ids=[rec[0]], payloads=[rec[3]], **rec_kwargs
                    )
                    persisted_records.append(rec)
                except Exception as e:
                    logger.error(f"Failed to insert memory {rec[0]} (async): {e}")

        if not persisted_records:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            raise VectorStoreError(
                f"Failed to insert any of the {len(records)} extracted memories into the vector store"
            )

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in persisted_records
        ]
        try:
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        except Exception:
            for hr in history_records:
                try:
                    await asyncio.to_thread(
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in persisted_records]
            if self.config.entity_source == "llm":
                all_entities = [
                    [(e["type"], e["surface"], e.get("aliases") or []) for e in r[3].get("entities", [])]
                    for r in persisted_records
                ]
            else:
                all_entities = [
                    [(t, s, []) for t, s in ents]
                    for ents in await asyncio.to_thread(extract_entities_batch, all_texts)
                ]

            # 7a: Global dedup
            global_entities = {}
            for idx, (memory_id, text, embedding, payload) in enumerate(persisted_records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text, aliases in entities:
                    key = self._normalize_entity_text(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                        global_entities[key][3] = sorted(set(global_entities[key][3]) | set(aliases))
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}, sorted(set(aliases))]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                except Exception:
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    exact_matches = await asyncio.to_thread(self._existing_entities_by_text, search_filters)

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = await asyncio.to_thread(
                        self.entity_store.search_batch,
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids, aliases = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        exact_match = exact_matches.get(key)

                        semantic_match = (
                            matches[0]
                            if matches and matches[0].score >= self._model_profile.entity_match_threshold
                            else None
                        )
                        match = exact_match or semantic_match
                        if match:
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            if aliases:
                                merged_aliases = set(payload.get("aliases") or []) | set(aliases)
                                merged_aliases.discard(payload.get("data"))
                                payload["aliases"] = sorted(merged_aliases)
                            try:
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        else:
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "aliases": aliases,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    if to_insert_vectors:
                        try:
                            await asyncio.to_thread(
                                self.entity_store.insert,
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed (async): {e}")
            await asyncio.to_thread(self._invalidate_entity_dictionary)
        except Exception as e:
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in persisted_records
        ]

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            await display_first_run_notice_async(self, "async", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        await display_first_run_notice_async(self, "async", "get")
        return result_item

    async def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        fetch_limit = limit if show_expired else max(limit * 4, 60)
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        all_memories_result = await self._get_all_from_vector_store(effective_filters, fetch_limit, show_expired, limit)

        if scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "get_all", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "get_all")
        return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit, show_expired=False, output_limit=None):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            if not show_expired and _payload_is_expired(mem.payload):
                continue
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)
            if output_limit is not None and len(formatted_memories) >= output_limit:
                break

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to the
                embedding model profile's semantic_threshold (0.1 for the default profile).
            rerank (bool, optional): Whether to rerank results. Defaults to False.
            explain (bool, optional): Whether to include score_details for each result. Defaults to False.
            reference_date (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        if reference_date is not None:
            raise ValueError(
                await get_temporal_feature_error_message_async("async", "search", "reference_date")
            )

        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        search_start = time.perf_counter()
        original_memories = await self._search_vector_store(
            query, effective_filters, limit, threshold, explain=explain, show_expired=show_expired
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # Apply reranking if enabled and reranker is available (top-N scoped,
        # optional conditional trigger; failure falls back to original results)
        if rerank and self.reranker and original_memories:
            candidates = original_memories[: self.config.rerank_top_n]
            should_rerank = self.config.rerank_trigger != "conditional" or _should_rerank_conditional(
                self.config, query, candidates
            )
            if should_rerank:
                try:
                    # Run reranking in thread pool to avoid blocking async loop
                    reranked_memories = await asyncio.to_thread(self.reranker.rerank, query, candidates, limit)
                    original_memories = (reranked_memories + original_memories[len(candidates):])[:limit]
                except Exception as e:
                    logger.warning(f"Reranking failed, using original results: {e}")

        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            await display_performance_slow_query_notice_async(
                self,
                "async",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            await display_first_run_notice_async(self, "async", "search")
        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold=None, explain=False, show_expired=False):
        # Resolve the semantic threshold from the model profile when not given explicitly
        if threshold is None:
            threshold = self._model_profile.semantic_threshold

        # Step 1: Preprocess query (CPU-bound)
        query_lemmatized = await asyncio.to_thread(self._text_for_sparse, query)
        use_dictionary = self.config.entity_source == "llm" or self.config.multilingual
        query_entities = [] if use_dictionary else await asyncio.to_thread(extract_entities, query)

        # Step 2: Embed query (one forward pass yields dense + sparse when available)
        embeddings, query_sparse = await asyncio.to_thread(self._embed_text_dual, query, "search")

        # Step 3: Semantic search (over-fetch)
        internal_limit = max(limit * 4, 60)
        semantic_results = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        learned_mode = _learned_sparse_mode(self.vector_store) and query_sparse is not None
        if learned_mode:
            keyword_results = await asyncio.to_thread(
                self.vector_store.keyword_search,
                query=query, top_k=internal_limit, filters=filters, sparse_vector=query_sparse,
            )
        else:
            keyword_results = await asyncio.to_thread(
                self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
            )

        # Step 5: Compute normalized keyword scores
        bm25_scores = {}
        if keyword_results is not None and learned_mode:
            query_self_score = sum(v * v for v in query_sparse["values"])
            zero_floor = self._model_profile.sparse_zero_floor
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    normalized = normalize_sparse_coverage(raw_score, query_self_score, zero_floor)
                    if normalized > 0:
                        bm25_scores[mem_id] = normalized
        elif keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if use_dictionary:
            entity_boosts = await self._compute_entity_boosts_multilingual_async(query, filters, embeddings)
        elif query_entities:
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            payload = mem.payload if hasattr(mem, 'payload') else {}
            if not show_expired and _payload_is_expired(payload):
                continue
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Step 8: Score and rank (fold before top_k truncation when enabled)
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=len(candidates) if self.config.near_duplicate else limit,
            explain=explain,
        )
        if self.config.near_duplicate:
            scored_results = _fold_near_duplicates(scored_results, explain=explain)[:limit]

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}
            if not payload.get("data"):
                continue

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self._normalize_entity_text(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            sem = asyncio.Semaphore(4)

            async def _search_entity(entity_text, embedding):
                async with sem:
                    return await asyncio.to_thread(
                        self.entity_store.search,
                        query=entity_text,
                        vectors=embedding,
                        top_k=500,
                        filters=search_filters,
                    )

            results = await asyncio.gather(
                *(_search_entity(text, emb) for text, emb in zip(entity_texts, embeddings)),
                return_exceptions=True,
            )

            for matches in results:
                if isinstance(matches, BaseException):
                    logger.warning("Entity boost search failed for one entity: %s", matches)
                    continue

                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    async def _entity_vector_matches_async(self, embedding, filters, top_k=500):
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        return await asyncio.to_thread(
            self.entity_store.search, query="", vectors=embedding, top_k=top_k, filters=search_filters
        )

    async def _compute_entity_boosts_multilingual_async(self, query, filters, query_embedding):
        """Async variant of `_compute_entity_boosts_multilingual`."""
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        def _apply(linked_memory_ids, similarity):
            if not isinstance(linked_memory_ids, list):
                return
            num_linked = max(len(linked_memory_ids), 1)
            memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
            boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
            for memory_id in linked_memory_ids:
                if memory_id:
                    memory_key = str(memory_id)
                    memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        try:
            rows = await asyncio.to_thread(self.entity_dictionary.match, query, search_filters)
            for row in rows:
                payload = getattr(row, "payload", None) or {}
                _apply(payload.get("linked_memory_ids", []), 1.0)
        except Exception as e:
            logger.warning(f"Entity dictionary match failed (async): {e}")

        try:
            matches = await self._entity_vector_matches_async(query_embedding, filters)
            for match in matches:
                similarity = match.score if hasattr(match, "score") else 0.0
                if similarity < self._model_profile.entity_vector_threshold:
                    continue
                payload = match.payload if hasattr(match, "payload") else {}
                _apply(payload.get("linked_memory_ids", []), similarity)
        except Exception as e:
            logger.warning(f"Entity vector fallback failed (async): {e}")

        return memory_boosts

    async def update(
        self,
        memory_id,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_date: Any = _UNSET,
        data: Optional[str] = None,
    ):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            text (str, optional): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.
                ``user_id``/``agent_id``/``run_id``/``actor_id`` are ignored here - they are
                immutable after creation.
            expiration_date (Any, optional): Date in YYYY-MM-DD format, or None to clear it.
            data (str, optional): Deprecated alias for ``text``. Will be removed in the next
                major release; use ``text`` instead.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", text="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        if data is not None:
            logger.warning(
                "The `data` argument to update() is deprecated and will be removed in the "
                "next major release. Use `text` instead."
            )
            if text is None:
                text = data

        if text is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of text, metadata, or expiration_date must be provided.")

        update_metadata = deepcopy(metadata) if metadata is not None else None
        if expiration_date is not _UNSET:
            update_metadata = update_metadata or {}
            update_metadata["expiration_date"] = _normalize_expiration_date(expiration_date)

        existing_embeddings = {}
        existing_sparse = {}
        if text is not None:
            embeddings, sparse = await asyncio.to_thread(self._embed_text_dual, text, "update")
            existing_embeddings[text] = embeddings
            existing_sparse[text] = sparse

        sparse_arg = existing_sparse if any(v is not None for v in existing_sparse.values()) else None
        if sparse_arg is not None:
            await self._update_memory(memory_id, text, existing_embeddings, update_metadata, existing_sparse=sparse_arg)
        else:
            await self._update_memory(memory_id, text, existing_embeddings, update_metadata)
        await display_first_run_notice_async(self, "async", "update")
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        await self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete")
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        deleted_count = 0
        errors = []
        seen_batches = set()
        while True:
            memories = await asyncio.to_thread(
                self.vector_store.list,
                filters=filters,
                top_k=DELETE_ALL_BATCH_SIZE,
            )
            batch = memories[0] if memories else []
            if not batch:
                break
            batch_ids = tuple(sorted(str(memory.id) for memory in batch))
            if batch_ids in seen_batches:
                logger.warning("Stopping delete_all after a repeated memory batch")
                break
            seen_batches.add(batch_ids)
            delete_tasks = [
                self._delete_memory(memory.id, skip_entity_cleanup=True)
                for memory in batch
            ]
            results = await asyncio.gather(*delete_tasks, return_exceptions=True)
            batch_errors = [
                result for result in results if isinstance(result, BaseException)
            ]
            errors.extend(batch_errors)
            deleted_count += len(results) - len(batch_errors)

        if self._entity_store is not None:
            await self._bulk_clear_entity_store(filters)

        if errors:
            logger.warning("Failed to delete %d memories", len(errors))
            for err in errors:
                logger.warning("Delete error: %s", err)

        logger.info(f"Deleted {deleted_count} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(deleted_count)
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete_all", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete_all")
        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        history = await asyncio.to_thread(self.db.get_history, memory_id)
        await display_first_run_notice_async(self, "async", "history")
        return history

    async def _create_memory(self, data, existing_embeddings, metadata=None, existing_sparse=None):
        logger.debug(f"Creating memory with {data=}")
        self._guard_write()
        sparse_vector = (existing_sparse or {}).get(data)
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings, sparse_vector = await asyncio.to_thread(self._embed_text_dual, data, "add")

        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = self._text_for_sparse(data)
        new_metadata.update(self._version_stamps)
        new_metadata.update(await self._near_duplicate_tag_async(embeddings, new_metadata))

        insert_kwargs = (
            {"sparse_vectors": [sparse_vector]}
            if sparse_vector is not None and _sparse_capable(self.vector_store)
            else {}
        )
        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
            **insert_kwargs,
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                # langchain-core is only needed to adapt messages for a custom
                # LangChain LLM. The default path uses self.llm and must not
                # require the optional dependency, mirroring the sync version.
                try:
                    from langchain_core.messages.utils import (
                        convert_to_messages,  # type: ignore
                    )
                except ImportError as e:
                    raise ImportError(
                        "langchain-core is required to pass a custom LLM to procedural memory. "
                        "Install it with 'pip install langchain-core'."
                    ) from e

                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = remove_code_blocks(response.content)
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)

        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if not procedural_memory:
            raise ValueError(
                "The LLM returned no content for the procedural memory summary. "
                "The model may have declined the request or returned an empty response."
            )

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None, existing_sparse=None):
        logger.info(f"Updating memory with {data=}")
        self._guard_write()

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            # Backing-store failure, not a bad memory_id: re-raise the original so the REST layer maps it to 5xx, not 4xx.
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")
        if data is None:
            data = prev_value
        if not isinstance(data, str):
            raise ValueError(f"Memory with id {memory_id} does not have text content to update")
        text_changed = data != prev_value

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(_strip_identity_keys(metadata, existing_memory.payload))

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = self._text_for_sparse(data)
        new_metadata.update(self._version_stamps)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
            sparse_vector = (existing_sparse or {}).get(data)
        else:
            embeddings, sparse_vector = await asyncio.to_thread(self._embed_text_dual, data, "update")

        update_kwargs = (
            {"sparse_vector": sparse_vector}
            if sparse_vector is not None and _sparse_capable(self.vector_store)
            else {}
        )
        updated = await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
            **update_kwargs,
        )
        if updated is False:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        if text_changed:
            await self._remove_memory_from_entity_store(memory_id, session_filters, delete_empty=False)
            await self._link_entities_for_memory(memory_id, data, session_filters, embedding=embeddings)

        return memory_id

    async def _delete_memory(self, memory_id, existing_memory=None, skip_entity_cleanup=False):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        if not skip_entity_cleanup:
            await self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        await asyncio.to_thread(self.db.reset)
        await asyncio.to_thread(self.db.close)
        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        if self._entity_store is not None:
            try:
                await asyncio.to_thread(self._entity_store.reset)
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "async"})
        await display_first_run_notice_async(self, "async", "reset")

    def close(self):
        """Release resources held by this AsyncMemory instance."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
