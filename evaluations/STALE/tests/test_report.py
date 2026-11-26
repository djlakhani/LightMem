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
                            "elapsed_seconds": 0.25,
                            "usage": {"total_tokens": 12},
                        }
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
