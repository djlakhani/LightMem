from dataclasses import replace
from pathlib import Path

from stale_lightmem.config import PathsConfig, load_config
from stale_lightmem.qwen import GenerationResult
from stale_lightmem.runner import run_generation

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, config, run_dir):
        self.consolidated = False

    def reset(self):
        self.consolidated = False

    def ingest(self, scenario, view):
        return {"elapsed_seconds": 0.1, "extraction_calls": [], "memory_count": 1}

    def consolidate(self):
        self.consolidated = True
        return {"elapsed_seconds": 0.1}

    def snapshot(self):
        memory = "new state" if self.consolidated else "old state"
        return [{"id": "memory", "payload": {"memory": memory}}]

    def retrieve(self, query, limit):
        trace = [{"id": "memory", "score": 1.0, "payload": {"memory": "new state"}}]
        return ["2025-01-01 Wed new state"], trace

    def statistics(self):
        return {"summary": {"total_llm_tokens": 10, "total_embedding_calls": 1}}

    def close(self):
        return None


class FakeQwen:
    def generate(self, messages, max_tokens):
        return GenerationResult(
            content="answer using the new state",
            elapsed_seconds=0.1,
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            attempts=1,
        )


def test_cpu_fake_generation_pipeline(tmp_path: Path):
    config = load_config(ROOT / "configs" / "qwen3_4b_full.yaml")
    config = replace(
        config,
        paths=PathsConfig(
            volume_root=str(tmp_path),
            run_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
        ),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = run_generation(
        config,
        run_dir,
        limit=1,
        upload=False,
        backend_factory=FakeBackend,
        qwen_client=FakeQwen(),
    )
    assert len(records) == 1
    assert set(records[0]["answers"]) == {"dim1", "dim2", "dim3"}
    assert records[0]["memory_diff"]["updated_ids"] == ["memory"]
    assert (run_dir / "scenario_records.jsonl").exists()
