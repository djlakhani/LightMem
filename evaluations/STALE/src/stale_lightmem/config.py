from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    volume_root: str = "/workspace/lightmem-stale"
    run_root: str = "/workspace/lightmem-stale/runs"
    cache_root: str = "/workspace/lightmem-stale/cache"


@dataclass(frozen=True)
class QwenConfig:
    model: str = "Qwen/Qwen3-4B"
    revision: str | None = None
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    thinking: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    extraction_max_tokens: int = 2048
    future_filter_max_tokens: int = 512
    answer_max_tokens: int = 512


@dataclass(frozen=True)
class LightMemConfig:
    compression_model: str = (
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    )
    compression_revision: str | None = None
    compression_rate: float = 0.8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    embedding_dimensions: int = 384
    messages_use: str = "user_only"
    extraction_mode: str = "flat"
    update_queue_top_k: int = 20
    update_queue_keep_n: int = 10
    update_score_threshold: float = 0.8
    retrieval_limit: int = 20
    future_similarity_limit: int = 8
    future_recency_limit: int = 4
    future_update_limit: int = 3
    future_filter_concurrency: int = 4


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "gpt-6-luna"
    concurrency: int = 1
    max_retries: int = 5
    reasoning_enabled: bool = True


@dataclass(frozen=True)
class S3Config:
    enabled: bool = True
    bucket: str = "stale-qwen-evals-589498925064-us-east-1-an"
    region: str = "us-east-1"
    prefix: str = "lightmem/stale/qwen3-4b"


@dataclass(frozen=True)
class EvalConfig:
    schema_version: int
    scenario_path: str
    scenario_sha256: str
    context_view: str = "truncated"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    lightmem: LightMemConfig = field(default_factory=LightMemConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    s3: S3Config = field(default_factory=S3Config)
    source_path: str = field(default="", compare=False)

    @property
    def digest(self) -> str:
        data = asdict(self)
        data.pop("source_path", None)
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def resolved_scenario_path(self) -> Path:
        source_dir = Path(self.source_path).resolve().parent
        return (source_dir / os.path.expandvars(self.scenario_path)).resolve()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: str | Path) -> EvalConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = _expand(yaml.safe_load(handle))

    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation config schema_version; expected 1")
    if raw.get("context_view") not in {"truncated", "evidence_only"}:
        raise ValueError("context_view must be 'truncated' or 'evidence_only'")

    qwen = QwenConfig(**raw.pop("qwen", {}))
    if qwen.thinking:
        raise ValueError("Qwen thinking must remain disabled for this benchmark")
    if qwen.future_filter_max_tokens < 1:
        raise ValueError("qwen.future_filter_max_tokens must be positive")

    paths = PathsConfig(**raw.pop("paths", {}))
    lightmem = LightMemConfig(**raw.pop("lightmem", {}))
    for field_name in (
        "future_similarity_limit",
        "future_recency_limit",
        "future_update_limit",
        "future_filter_concurrency",
    ):
        if getattr(lightmem, field_name) < 1:
            raise ValueError(f"lightmem.{field_name} must be positive")
    judge = JudgeConfig(**raw.pop("judge", {}))
    s3 = S3Config(**raw.pop("s3", {}))
    config = EvalConfig(
        **raw,
        paths=paths,
        qwen=qwen,
        lightmem=lightmem,
        judge=judge,
        s3=s3,
        source_path=str(source),
    )
    return config
