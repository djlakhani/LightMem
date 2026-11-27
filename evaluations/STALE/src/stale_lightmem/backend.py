from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .config import EvalConfig
from .dataset import iter_turns
from .qwen import assert_no_thinking, wrap_openai_client_no_thinking


class MemoryBackend(Protocol):
    def reset(self) -> None: ...
    def ingest(self, scenario: dict[str, Any], view: str) -> dict[str, Any]: ...
    def consolidate(self) -> dict[str, Any]: ...
    def snapshot(self) -> list[dict[str, Any]]: ...
    def retrieve(
        self, query: str, limit: int
    ) -> tuple[list[str], list[dict[str, Any]]]: ...
    def future_candidates(
        self,
        anchor: dict[str, Any],
        similarity_limit: int,
        recency_limit: int,
    ) -> dict[str, Any]: ...
    def statistics(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class LightMemBackend:
    """Full LightMem pipeline backed by an embedded, isolated Qdrant collection."""

    def __init__(self, config: EvalConfig, run_dir: Path):
        from lightmem.memory.lightmem import LightMemory

        self.config = config
        qdrant_path = run_dir / "scratch" / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        log_dir = run_dir / "logs" / "lightmem"
        log_dir.mkdir(parents=True, exist_ok=True)

        lm = config.lightmem
        qwen = config.qwen
        lightmem_config = {
            "pre_compress": True,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": lm.compression_model,
                        "device_map": "cuda",
                        "use_llmlingua2": True,
                    },
                    "compress_config": {
                        "instruction": "",
                        "rate": lm.compression_rate,
                        "target_token": -1,
                    },
                },
            },
            "topic_segment": True,
            "precomp_topic_shared": True,
            "topic_segmenter": {"model_name": "llmlingua-2"},
            "messages_use": lm.messages_use,
            "metadata_generate": True,
            "text_summary": True,
            # OpenaiManager is intentional: unlike the current VllmManager it
            # implements LightMemory._call_update_llm for offline consolidation.
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": qwen.model,
                    "api_key": qwen.api_key,
                    "openai_base_url": qwen.base_url,
                    "max_tokens": qwen.extraction_max_tokens,
                    "temperature": qwen.temperature,
                    "top_p": qwen.top_p,
                    "seed": qwen.seed,
                },
            },
            "extract_threshold": 0.1,
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": lm.embedding_model,
                    "embedding_dims": lm.embedding_dimensions,
                    "model_kwargs": {"device": "cpu"},
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": "stale_scenario",
                    "embedding_model_dims": lm.embedding_dimensions,
                    "path": str(qdrant_path),
                    "on_disk": True,
                },
            },
            "update": "offline",
            "logging": {
                "level": "INFO",
                "file_enabled": True,
                "log_dir": str(log_dir),
                "console_enabled": True,
                "file_level": "DEBUG",
            },
            "extraction_mode": lm.extraction_mode,
        }

        # Prevent an unrelated shell setting from rerouting this local client.
        openrouter_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            self.memory = LightMemory.from_config(lightmem_config)
        finally:
            if openrouter_key is not None:
                os.environ["OPENROUTER_API_KEY"] = openrouter_key
        self.memory.manager.client = wrap_openai_client_no_thinking(
            self.memory.manager.client
        )

    def reset(self) -> None:
        self.memory.embedding_retriever.reset()
        self.memory.senmem_buffer_manager.buffer.clear()
        self.memory.senmem_buffer_manager.big_buffer.clear()
        self.memory.senmem_buffer_manager.token_count = 0
        self.memory.shortmem_buffer_manager.buffer.clear()
        self.memory.shortmem_buffer_manager.token_count = 0
        for key in self.memory.token_stats:
            self.memory.token_stats[key] = 0
        if hasattr(self.memory.text_embedder, "total_calls"):
            self.memory.text_embedder.total_calls = 0
        if hasattr(self.memory.text_embedder, "total_tokens"):
            self.memory.text_embedder.total_tokens = 0
        if self.snapshot():
            raise RuntimeError(
                "Qdrant isolation failure: collection is not empty after reset"
            )

    def ingest(self, scenario: dict[str, Any], view: str) -> dict[str, Any]:
        started = time.perf_counter()
        extraction_calls: list[dict[str, Any]] = []
        for pair, is_final in iter_turns(scenario, view):
            result = self.memory.add_memory(
                messages=pair,
                force_segment=is_final,
                force_extract=is_final,
            )
            for output in result.get("add_output_prompt", []):
                assert_no_thinking(str(output))
            if result.get("api_call_nums", 0):
                extraction_calls.append(copy.deepcopy(result))
        memories = self.snapshot()
        if not memories:
            raise RuntimeError("LightMem produced no memories for the scenario")
        return {
            "elapsed_seconds": time.perf_counter() - started,
            "extraction_calls": extraction_calls,
            "memory_count": len(memories),
        }

    def consolidate(self) -> dict[str, Any]:
        started = time.perf_counter()
        lm = self.config.lightmem
        self.memory.construct_update_queue_all_entries(
            top_k=lm.update_queue_top_k,
            keep_top_n=lm.update_queue_keep_n,
        )
        self.memory.offline_update_all_entries(
            score_threshold=lm.update_score_threshold,
        )
        return {"elapsed_seconds": time.perf_counter() - started}

    def snapshot(self) -> list[dict[str, Any]]:
        points = self.memory.embedding_retriever.get_all(
            with_vectors=False,
            with_payload=True,
        )
        return sorted(
            (
                {"id": str(point["id"]), "payload": point.get("payload", {})}
                for point in points
            ),
            key=lambda item: item["id"],
        )

    def retrieve(
        self, query: str, limit: int
    ) -> tuple[list[str], list[dict[str, Any]]]:
        query_vector = self.memory.text_embedder.embed(query)
        results = self.memory.embedding_retriever.search(
            query_vector=query_vector,
            limit=limit,
            return_full=True,
        )
        traces: list[dict[str, Any]] = []
        formatted: list[str] = []
        for result in results:
            payload = result.get("payload", {})
            traces.append(
                {
                    "id": str(result.get("id")),
                    "score": result.get("score"),
                    "payload": payload,
                }
            )
            formatted.append(
                f"{payload.get('time_stamp', '')} {payload.get('weekday', '')} "
                f"{payload.get('memory', '')}".strip()
            )
        return formatted, traces

    def future_candidates(
        self,
        anchor: dict[str, Any],
        similarity_limit: int,
        recency_limit: int,
    ) -> dict[str, Any]:
        points = self.memory.embedding_retriever.get_all(
            with_vectors=True,
            with_payload=True,
        )
        return build_future_candidates(
            anchor,
            points,
            similarity_limit=similarity_limit,
            recency_limit=recency_limit,
        )

    def statistics(self) -> dict[str, Any]:
        result = copy.deepcopy(self.memory.get_token_statistics())
        result["attributes"] = copy.deepcopy(self.memory.get_attribute_statistics())
        return result

    def close(self) -> None:
        client = getattr(self.memory.embedding_retriever, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def diff_memories(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    before_map = {item["id"]: item.get("payload", {}) for item in before}
    after_map = {item["id"]: item.get("payload", {}) for item in after}
    deleted = sorted(set(before_map) - set(after_map))
    added = sorted(set(after_map) - set(before_map))
    updated = sorted(
        key
        for key in set(before_map) & set(after_map)
        if before_map[key].get("memory") != after_map[key].get("memory")
    )
    unchanged = sorted(
        key for key in set(before_map) & set(after_map) if key not in updated
    )
    return {
        "before_count": len(before),
        "after_count": len(after),
        "deleted_ids": deleted,
        "added_ids": added,
        "updated_ids": updated,
        "unchanged_ids": unchanged,
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Future-candidate vectors are missing or incompatible")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Future-candidate vectors must have nonzero norm")
    return dot / (left_norm * right_norm)


def build_future_candidates(
    anchor: dict[str, Any],
    points: list[dict[str, Any]],
    *,
    similarity_limit: int = 8,
    recency_limit: int = 4,
) -> dict[str, Any]:
    """Build a deterministic future-state candidate set for one anchor."""
    if similarity_limit < 0 or recency_limit < 0:
        raise ValueError("Future-candidate limits cannot be negative")

    anchor_id = str(anchor.get("id"))
    anchor_payload = anchor.get("payload", {})
    anchor_timestamp = anchor_payload.get("float_time_stamp")
    anchor_attributes = anchor_payload.get("attributes", [])
    if not isinstance(anchor_timestamp, (int, float)):
        return {
            "anchor_id": anchor_id,
            "eligible_count": 0,
            "eligible_memories": [],
            "candidates": [],
            "skip_reason": "missing_anchor_timestamp",
        }
    if not isinstance(anchor_attributes, list) or not anchor_attributes:
        return {
            "anchor_id": anchor_id,
            "eligible_count": 0,
            "eligible_memories": [],
            "candidates": [],
            "skip_reason": "missing_anchor_attributes",
        }

    point_by_id = {str(point.get("id")): point for point in points}
    anchor_point = point_by_id.get(anchor_id)
    if anchor_point is None:
        raise RuntimeError(f"Anchor {anchor_id} is missing from the memory index")
    anchor_vector = anchor_point.get("vector")
    if not isinstance(anchor_vector, list):
        raise RuntimeError(f"Anchor {anchor_id} has no stored embedding vector")

    eligible: list[dict[str, Any]] = []
    for point in points:
        point_id = str(point.get("id"))
        if point_id == anchor_id:
            continue
        payload = point.get("payload", {})
        timestamp = payload.get("float_time_stamp")
        attributes = payload.get("attributes", [])
        if not isinstance(timestamp, (int, float)) or timestamp <= anchor_timestamp:
            continue
        if not isinstance(attributes, list):
            continue
        matched_attributes = [
            attribute for attribute in anchor_attributes if attribute in set(attributes)
        ]
        if not matched_attributes:
            continue
        vector = point.get("vector")
        if not isinstance(vector, list):
            continue
        eligible.append(
            {
                "id": point_id,
                "score": _cosine_similarity(anchor_vector, vector),
                "matched_attributes": matched_attributes,
                "payload": payload,
            }
        )

    similarity_ranked = sorted(
        eligible,
        key=lambda item: (
            -item["score"],
            -float(item["payload"]["float_time_stamp"]),
            item["id"],
        ),
    )
    similarity_selected = similarity_ranked[:similarity_limit]
    similarity_ids = {item["id"] for item in similarity_selected}
    recency_selected = sorted(
        (item for item in eligible if item["id"] not in similarity_ids),
        key=lambda item: (
            -float(item["payload"]["float_time_stamp"]),
            -item["score"],
            item["id"],
        ),
    )[:recency_limit]

    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(similarity_selected, start=1):
        candidates.append(
            {**item, "candidate_source": "similarity", "source_rank": rank}
        )
    for rank, item in enumerate(recency_selected, start=1):
        candidates.append({**item, "candidate_source": "recency", "source_rank": rank})
    return {
        "anchor_id": anchor_id,
        "eligible_count": len(eligible),
        "eligible_memories": similarity_ranked,
        "candidates": candidates,
        "skip_reason": None,
    }
