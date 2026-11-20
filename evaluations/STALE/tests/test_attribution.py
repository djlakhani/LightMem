import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from stale_lightmem.attribution import (
    build_citation_prompt,
    citation_metrics,
    discover_state_candidates,
    parse_citation_response,
)
from stale_lightmem.attribution_runner import (
    generate_attribution_report,
    run_attribution_judging,
    run_cited_generation,
    run_labeling,
)
from stale_lightmem.config import PathsConfig, load_config
from stale_lightmem.qwen import GenerationResult


def _retrieval():
    return [
        {
            "id": f"id-{index}",
            "score": 1.0 / index,
            "payload": {
                "time_stamp": f"2025-01-0{index}T10:00:00.000",
                "weekday": "Mon",
                "memory": f"memory {index}",
            },
        }
        for index in range(1, 6)
    ]


def test_citation_prompt_exposes_ids_without_ground_truth():
    prompt = build_citation_prompt("What changed?", _retrieval(), "SR")
    rendered = json.dumps(prompt)
    assert "MEMORY_ID=id-1" in rendered
    assert "exactly 4 memory IDs" in rendered
    assert "M_old" not in rendered
    assert "M_new" not in rendered


def test_parse_citation_response_requires_four_valid_unique_ids():
    content = json.dumps(
        {
            "answer": "The state changed.",
            "memory_ids_used": ["id-1", "id-2", "id-3", "id-4"],
        }
    )
    parsed = parse_citation_response(content, {f"id-{i}" for i in range(1, 6)})
    assert parsed["memory_ids_used"] == ["id-1", "id-2", "id-3", "id-4"]

    with pytest.raises(ValueError, match="not in retrieval"):
        parse_citation_response(
            content.replace('"id-4"', '"invented"'),
            {f"id-{i}" for i in range(1, 6)},
        )
    with pytest.raises(ValueError, match="unique"):
        parse_citation_response(
            content.replace('"id-4"', '"id-3"'),
            {f"id-{i}" for i in range(1, 6)},
        )


def test_citation_metrics_distinguish_old_new_and_other():
    metrics = citation_metrics(
        ["id-1", "id-2", "id-3", "id-4"],
        _retrieval(),
        old_ids={"id-1", "not-retrieved-old"},
        new_ids={"id-2", "id-5"},
    )
    assert metrics["evidence_pattern"] == "both"
    assert metrics["selected_old_ids"] == ["id-1"]
    assert metrics["selected_new_ids"] == ["id-2"]
    assert metrics["selected_neither_ids"] == ["id-3", "id-4"]
    assert metrics["old_ids_available_in_retrieval"] == ["id-1"]
    assert metrics["new_ids_available_in_retrieval"] == ["id-2", "id-5"]


def test_discover_state_candidates_follows_extractor_source_id():
    scenario = {
        "views": {
            "truncated": {
                "sessions": [
                    {
                        "timestamp": "2025-01-01 10:00",
                        "messages": [
                            {
                                "role": "user",
                                "content": "I moved from Boston to Seattle yesterday.",
                            },
                            {"role": "assistant", "content": "Understood."},
                        ],
                    }
                ]
            }
        }
    }
    fact = "User moved from Boston to Seattle yesterday."
    record = {
        "context_view": "truncated",
        "ingest": {
            "extraction_calls": [
                {
                    "add_input_prompt": [
                        [
                            {"role": "system", "content": "extract"},
                            {
                                "role": "user",
                                "content": (
                                    "--- Topic 1 ---\n"
                                    "[2025-01-01T10:00:00.000, Wed] 0.user: "
                                    "I moved from Boston to Seattle yesterday."
                                ),
                            },
                        ]
                    ],
                    "add_output_prompt": [
                        json.dumps({"data": [{"source_id": 0, "fact": fact}]})
                    ],
                }
            ]
        },
        "pre_update_memories": [
            {"id": "new-state", "payload": {"memory": fact, "time_stamp": "x"}}
        ],
        "post_update_memories": [
            {"id": "new-state", "payload": {"memory": fact, "time_stamp": "x"}}
        ],
    }
    result = discover_state_candidates(
        "I moved from Boston to Seattle yesterday.", scenario, record
    )
    assert result["status"] == "candidates_found"
    assert [item["id"] for item in result["eligible_candidates"]] == ["new-state"]


def test_existing_artifact_has_only_one_missing_state_extraction():
    artifact = (
        Path(__file__).resolve().parents[4]
        / "evaluation-artifacts/qwen3-4b-stale-lightmem-raw/stale/qwen3-4b"
        / "20261118T050518Z-161fcbb3-4fa9f354/scenario_records.jsonl"
    )
    if not artifact.is_file():
        pytest.skip("Downloaded evaluation artifact is not present")
    scenarios_path = Path(__file__).resolve().parents[1] / "scenarios.json"
    scenarios = {
        item["uid"]: item for item in json.loads(scenarios_path.read_text())["data"]
    }
    records = [json.loads(line) for line in artifact.read_text().splitlines() if line]
    missing = []
    for record in records:
        scenario = scenarios[record["uid"]]
        for key in ("M_old", "M_new"):
            result = discover_state_candidates(scenario[key], scenario, record)
            if result["status"] != "candidates_found":
                missing.append((record["index"], key, result["status"]))
    assert missing == [(44, "M_new", "missing_from_extraction_input")]


def test_fake_attribution_pipeline_uses_existing_retrievals(tmp_path):
    source = (
        Path(__file__).resolve().parents[4]
        / "evaluation-artifacts/qwen3-4b-stale-lightmem-raw/stale/qwen3-4b"
        / "20261118T050518Z-161fcbb3-4fa9f354"
    )
    if not (source / "scenario_records.jsonl").is_file():
        pytest.skip("Downloaded evaluation artifact is not present")
    config_path = Path(__file__).resolve().parents[1] / "configs/qwen3_4b_full.yaml"
    config = load_config(config_path)
    config = replace(
        config,
        paths=PathsConfig(
            volume_root=str(tmp_path),
            run_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
        ),
    )
    run_dir = tmp_path / "attribution-run"
    run_dir.mkdir()

    class FakeJudge:
        def label(self, scenario, discoveries):
            result = {}
            for key in ("old", "new"):
                candidates = discoveries[key]["eligible_candidates"]
                decisions = [
                    {
                        "id": item["id"],
                        "matches": index == 0,
                        "reasoning": "selected first source-attributed fact",
                    }
                    for index, item in enumerate(candidates)
                ]
                result[key] = {
                    "decisions": decisions,
                    "matching_ids": [
                        item["id"] for item in decisions if item["matches"]
                    ],
                    "excluded_ids": [
                        item["id"] for item in decisions if not item["matches"]
                    ],
                }
            return {
                "result": result,
                "usage": {"total_tokens": 1},
                "prompt": {},
            }

        def evaluate(self, scenario, answers, ground_truth):
            result = {}
            for dim_key in ("dim1", "dim2", "dim3"):
                metrics = answers[dim_key]["citation_metrics"]
                result[f"{dim_key}_eval"] = {
                    "answer_pass": True,
                    "answer_reasoning": "fake pass",
                    "old_evidence": {
                        "used": metrics["used_old_evidence"],
                        "reasoning": "exact overlap",
                    },
                    "new_evidence": {
                        "used": metrics["used_new_evidence"],
                        "reasoning": "exact overlap",
                    },
                    "citation_assessment": "fake assessment",
                }
            return {
                "result": result,
                "usage": {"total_tokens": 1},
                "prompt": {},
            }

    class FakeQwen:
        def generate(self, messages, **kwargs):
            ids = re.findall(r"\[MEMORY_ID=([^\]]+)\]", messages[1]["content"])
            content = json.dumps(
                {
                    "answer": "The newer state should control.",
                    "memory_ids_used": ids[:4],
                }
            )
            return GenerationResult(
                content=content,
                elapsed_seconds=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                attempts=1,
            )

    labels = run_labeling(
        config,
        source,
        run_dir,
        limit=1,
        upload=False,
        judge=FakeJudge(),
    )
    generated = run_cited_generation(
        config, source, run_dir, upload=False, qwen=FakeQwen()
    )
    judged = run_attribution_judging(
        config, source, run_dir, upload=False, judge=FakeJudge()
    )
    summary = generate_attribution_report(config, source, run_dir, allow_partial=True)

    assert len(labels) == len(generated) == len(judged) == 1
    assert len(generated[0]["answers"]["dim1"]["memory_ids_used"]) == 4
    assert summary["judged_questions"] == 3
    assert (run_dir / "report.md").is_file()
