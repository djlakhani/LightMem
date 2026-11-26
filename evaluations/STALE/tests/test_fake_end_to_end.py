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
        trace = [
            {
                "id": f"memory-{index}",
                "score": 1.0 - index / 10,
                "payload": {
                    "time_stamp": f"2025-01-0{index + 1}",
                    "attributes": [
                        "Residence & housing",
                        "Goals & commitments",
                    ],
                    "memory": f"new state {index}",
                },
            }
            for index in range(5)
        ]
        return [f"2025-01-0{i + 1} new state {i}" for i in range(5)], trace

    def statistics(self):
        return {"summary": {"total_llm_tokens": 10, "total_embedding_calls": 1}}

    def close(self):
        return None


class FakeQwen:
    def generate(self, messages, max_tokens, response_format=None):
        content = (
            '{"memory_ids":["memory-3","memory-1","memory-0","memory-2"]}'
            if response_format
            else "answer using the new state"
        )
        return GenerationResult(
            content=content,
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
    first_answer = records[0]["answers"]["dim1"]
    assert first_answer["evidence_selection"]["memory_ids"] == [
        "memory-3",
        "memory-1",
        "memory-0",
        "memory-2",
    ]
    final_prompt = first_answer["prompt"][1]["content"]
    assert "Memory ID: memory-3" in final_prompt
    assert "Residence & housing" in final_prompt
    assert "memory-4" not in final_prompt
    assert (run_dir / "scenario_records.jsonl").exists()
