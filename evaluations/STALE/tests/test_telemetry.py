import json
from dataclasses import replace
from pathlib import Path

from stale_lightmem.artifacts import atomic_write_json, checkpoint_name
from stale_lightmem.config import PathsConfig, load_config
from stale_lightmem.runner import canonical_hash
from stale_lightmem.telemetry import (
    generate_telemetry_report,
    parse_semantic_labels,
    run_telemetry_analysis,
    state_presence,
)

ROOT = Path(__file__).resolve().parents[1]


def _memory(memory_id, text, attributes=None):
    return {
        "id": memory_id,
        "payload": {
            "memory": text,
            "time_stamp": "2025-01-01",
            "attributes": attributes or ["Residence & housing", "Goals & commitments"],
        },
    }


def test_semantic_labels_support_single_and_multi_memory_sets():
    memories = [_memory("old-a", "old part"), _memory("old-b", "other part")]
    content = json.dumps(
        {
            "old": {
                "decisions": [
                    {"id": "old-a", "matches": False, "reasoning": "partial"},
                    {"id": "old-b", "matches": False, "reasoning": "partial"},
                ],
                "support_sets": [
                    {
                        "memory_ids": ["old-a", "old-b"],
                        "reasoning": "the pair captures the old state",
                    }
                ],
            },
            "new": {
                "decisions": [
                    {"id": "old-a", "matches": True, "reasoning": "new state"},
                    {"id": "old-b", "matches": False, "reasoning": "not new"},
                ],
                "support_sets": [],
            },
        }
    )
    labels = parse_semantic_labels(content, memories)
    assert labels["new"]["support_sets"][0]["memory_ids"] == ["old-a"]
    assert state_presence(["old-a"], labels["old"])["present"] is False
    assert state_presence(["old-a", "old-b"], labels["old"])["present"] is True
    assert state_presence(["old-a"], labels["new"])["present"] is True


def test_telemetry_checkpoint_and_report_use_full_memory_inventory(tmp_path):
    config = load_config(ROOT / "configs" / "qwen3_4b_full.yaml")
    config = replace(
        config,
        paths=PathsConfig(
            volume_root=str(tmp_path),
            run_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
        ),
    )
    scenario = json.loads((ROOT / "scenarios.json").read_text())["data"][0]
    scenario_hash = canonical_hash(scenario)
    run_dir = tmp_path / "run"
    generation_dir = run_dir / "checkpoints" / "generation"
    judge_dir = run_dir / "checkpoints" / "judge"
    generation_dir.mkdir(parents=True)
    judge_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "manifest.json",
        {"config_hash": config.digest, "run_id": "run", "status": "running"},
    )
    memories = [
        _memory("old", scenario["M_old"]),
        _memory("new", scenario["M_new"]),
        _memory("other", "unrelated memory"),
    ]
    answers = {}
    for dim_key in ("dim1", "dim2", "dim3"):
        answers[dim_key] = {
            "query": scenario["probing_queries"][f"{dim_key}_query"],
            "retrieval": memories,
            "evidence_selection": {
                "selected_memories": memories[:2],
                "generation": {
                    "method": "llm",
                    "llm_attempts": 1,
                    "fallback_used": False,
                    "fallback_reason": None,
                },
            },
            "future_update_analysis": {"anchors": []},
            "prompt": [{"role": "user", "content": "exact final prompt"}],
            "answer": "new state answer",
            "generation": {"attempts": 1},
        }
    record = {
        "status": "complete",
        "config_hash": config.digest,
        "scenario_hash": scenario_hash,
        "index": 1,
        "uid": scenario["uid"],
        "type": scenario["type"],
        "type_name": scenario["type_name"],
        "context_view": "truncated",
        "post_update_memories": memories,
        "memory_diff": {"before_count": 3, "after_count": 3},
        "answers": answers,
        "token_statistics": {
            "attributes": {
                "candidate_facts": 3,
                "llm_valid_pairs": 2,
                "embedding_fallback_attempts": 1,
                "embedding_fallback_successes": 1,
                "embedding_fallback_failures": 0,
                "attribute_rejections": 0,
            }
        },
    }
    atomic_write_json(generation_dir / checkpoint_name(1, scenario["uid"]), record)
    atomic_write_json(
        judge_dir / checkpoint_name(1, scenario["uid"]),
        {
            "status": "complete",
            "uid": scenario["uid"],
            "evaluation": {
                f"{dim_key}_eval": {"pass": True, "reasoning": "correct"}
                for dim_key in ("dim1", "dim2", "dim3")
            },
        },
    )

    class FakeSemanticJudge:
        def label(self, received_scenario, received_memories):
            assert received_scenario["uid"] == scenario["uid"]
            assert [item["id"] for item in received_memories] == [
                "old",
                "new",
                "other",
            ]
            decisions = {
                "old": [
                    {
                        "id": item["id"],
                        "matches": item["id"] == "old",
                        "reasoning": "old",
                    }
                    for item in received_memories
                ],
                "new": [
                    {
                        "id": item["id"],
                        "matches": item["id"] == "new",
                        "reasoning": "new",
                    }
                    for item in received_memories
                ],
            }
            return {
                "result": {
                    state: {
                        "decisions": decisions[state],
                        "support_sets": [{"memory_ids": [state], "reasoning": state}],
                        "independent_matching_ids": [state],
                        "participating_ids": [state],
                    }
                    for state in ("old", "new")
                },
                "attempts": 1,
                "usage": {"total_tokens": 10},
                "prompt": {},
            }

    telemetry = run_telemetry_analysis(
        config, run_dir, upload=False, judge=FakeSemanticJudge()
    )
    assert telemetry[0]["questions"]["dim1"]["stages"][0]["old"]["present"]
    (run_dir / "report.md").write_text(
        "# LightMem STALE Evaluation\n\nshort report\n", encoding="utf-8"
    )
    summary = generate_telemetry_report(
        config, run_dir, allow_partial=False, upload=False
    )
    assert summary["status"] == "complete"
    report = (run_dir / "memory_filtering_telemetry_report.md").read_text()
    assert "Top-20 embedding retrieval" in report
    assert "Final prompt contains M_old + M_new: **True**" in report
    assert "embedding_fallback_successes" in report
    assert (run_dir / "report.md").read_text() == report
    assert (
        (run_dir / "accuracy_report.md")
        .read_text()
        .startswith("# LightMem STALE Evaluation")
    )
