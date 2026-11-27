from stale_lightmem.report import build_summary


def _judgment(index: int):
    scenario_type = "T1" if index < 25 else "T2"
    return {
        "uid": str(index),
        "type": scenario_type,
        "evaluation": {
            "dim1_eval": {"pass": True, "reasoning": "ok"},
            "dim2_eval": {"pass": index % 2 == 0, "reasoning": "ok"},
            "dim3_eval": {"pass": False, "reasoning": "ok"},
        },
    }


def test_summary_has_all_expected_denominators():
    summary = build_summary([_judgment(index) for index in range(50)])
    assert summary["status"] == "complete"
    assert summary["accuracy"]["overall"]["total"] == 150
    assert summary["accuracy"]["T1"]["total"] == 75
    assert summary["accuracy"]["T2"]["total"] == 75
    assert summary["accuracy"]["SR"]["total"] == 50
    assert summary["accuracy"]["T1×IPA"]["total"] == 25
    assert summary["accuracy"]["T2×PR"]["total"] == 25


def test_summary_tracks_evidence_selection_time_and_tokens():
    records = [
        {
            "answers": {
                "dim1": {
                    "evidence_selection": {
                        "generation": {
                            "method": "embedding_rank_fallback",
                            "fallback_used": True,
                            "elapsed_seconds": 0.25,
                            "usage": {"total_tokens": 12},
                        }
                    },
                    "future_update_analysis": {
                        "candidate_search_elapsed_seconds": 0.1,
                        "anchors": [
                            {
                                "candidate_search": {"candidates": [{"id": "future"}]},
                                "selected_updates": [],
                                "filter_result": {
                                    "method": "no_updates_fallback",
                                    "fallback_used": True,
                                    "llm_attempts": 2,
                                    "elapsed_seconds": 0.2,
                                    "usage": {"total_tokens": 7},
                                },
                            }
                        ],
                    },
                    "generation": {
                        "elapsed_seconds": 0.5,
                        "usage": {"total_tokens": 20},
                    },
                }
            }
        }
    ]
    summary = build_summary([], records)
    assert summary["timing"]["evidence_selection_seconds"] == 0.25
    assert summary["usage"]["evidence_selection_qwen_tokens"] == 12
    assert summary["usage"]["answer_qwen_tokens"] == 20
    assert summary["timing"]["future_candidate_search_seconds"] == 0.1
    assert summary["timing"]["future_filter_seconds"] == 0.2
    assert summary["usage"]["future_filter_qwen_tokens"] == 7
    assert summary["evidence_selection"] == {
        "total": 1,
        "llm": 0,
        "embedding_rank_fallback": 1,
    }
    assert summary["future_updates"]["anchors"] == 1
    assert summary["future_updates"]["with_candidates"] == 1
    assert summary["future_updates"]["no_updates_fallback"] == 1
