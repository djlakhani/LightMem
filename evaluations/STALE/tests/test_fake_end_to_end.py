from dataclasses import replace
from pathlib import Path

from stale_lightmem.config import (
    DependencyResolutionConfig,
    PathsConfig,
    load_config,
)
from stale_lightmem.qwen import GenerationResult
from stale_lightmem.runner import run_generation
from lightmem.memory.taxonomy import label_metadata

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
                "score": 1.0 / index,
                "payload": {
                    "memory": f"state {index}",
                    "time_stamp": f"2025-01-0{index}T10:00:00.000",
                    "float_time_stamp": float(index),
                    "category_labels": [
                        "residence_housing",
                        "goals_commitments",
                    ],
                },
            }
            for index in range(1, 6)
        ]
        return [f"2025-01-0{i} state {i}" for i in range(1, 6)], trace

    def future_candidates(self, anchor, semantic_limit, recent_limit):
        anchor_id = anchor["id"]
        labels = ["residence_housing", "goals_commitments"]
        return [
            {
                "id": f"future-{anchor_id}",
                "score": 0.9,
                "payload": {
                    "memory": f"later state for {anchor_id}",
                    "time_stamp": "2025-02-01T10:00:00.000",
                    "float_time_stamp": 100.0,
                    "category_labels": labels,
                },
                "category_labels": label_metadata(labels),
                "matching_labels": label_metadata(labels),
                "selection_source": "semantic",
            }
        ]

    def statistics(self):
        return {"summary": {"total_llm_tokens": 10, "total_embedding_calls": 1}}

    def close(self):
        return None


class FakeQwen:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, max_tokens, response_format=None):
        import json

        self.calls += 1
        stage = (self.calls - 1) % 3
        if stage == 0:
            content = json.dumps(
                {"anchor_ids": [f"memory-{index}" for index in range(1, 5)]}
            )
        elif stage == 1:
            payload = json.loads(messages[1]["content"])
            content = json.dumps(
                {
                    "anchor_resolutions": {
                        group["anchor"]["id"]: {
                            "candidate_verdicts": {
                                candidate["id"]: "updates"
                                for candidate in group[
                                    "future_states_within_matching_categories"
                                ]
                            },
                            "selected_changes": [
                                {
                                    "memory_id": group[
                                        "future_states_within_matching_categories"
                                    ][0]["id"],
                                    "relation": "updates",
                                    "why": "The later state directly replaces the earlier state.",
                                }
                            ],
                            "resolution_summary": (
                                "The later user state replaces the earlier anchor state."
                            ),
                        }
                        for group in payload["anchor_groups"]
                    }
                }
            )
        else:
            content = "answer using the resolved newer state"
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
    qwen = FakeQwen()
    records = run_generation(
        config,
        run_dir,
        limit=1,
        upload=False,
        backend_factory=FakeBackend,
        qwen_client=qwen,
    )
    assert len(records) == 1
    assert set(records[0]["answers"]) == {"dim1", "dim2", "dim3"}
    assert records[0]["memory_diff"]["updated_ids"] == ["memory"]
    assert qwen.calls == 9
    assert records[0]["answers"]["dim1"]["anchor_selection"]["anchor_ids"] == [
        "memory-1",
        "memory-2",
        "memory-3",
        "memory-4",
    ]
    assert (run_dir / "scenario_records.jsonl").exists()


def test_dependency_resolution_can_be_disabled(tmp_path: Path):
    config = load_config(ROOT / "configs" / "qwen3_4b_full.yaml")
    config = replace(
        config,
        paths=PathsConfig(
            volume_root=str(tmp_path),
            run_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
        ),
        dependency_resolution=DependencyResolutionConfig(enabled=False),
    )
    run_dir = tmp_path / "disabled-run"
    run_dir.mkdir()

    class PlainFakeQwen:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, max_tokens, response_format=None):
            self.calls += 1
            return GenerationResult(
                content="plain baseline answer",
                elapsed_seconds=0.1,
                usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                attempts=1,
            )

    qwen = PlainFakeQwen()
    records = run_generation(
        config,
        run_dir,
        limit=1,
        upload=False,
        backend_factory=FakeBackend,
        qwen_client=qwen,
    )
    assert qwen.calls == 3
    assert records[0]["answers"]["dim1"]["answer"] == "plain baseline answer"
    assert "anchor_selection" not in records[0]["answers"]["dim1"]
