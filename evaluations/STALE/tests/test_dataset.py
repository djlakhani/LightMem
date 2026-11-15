import copy
from pathlib import Path

from stale_lightmem.config import load_config
from stale_lightmem.dataset import iter_turns, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_dataset_contract():
    config = load_config(ROOT / "configs" / "qwen3_4b_full.yaml")
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    assert len(dataset.scenarios) == 50
    assert sum(item["type"] == "T1" for item in dataset.scenarios) == 25
    assert sum(item["type"] == "T2" for item in dataset.scenarios) == 25
    assert len({item["uid"] for item in dataset.scenarios}) == 50


def test_iter_turns_copies_and_timestamps_messages():
    config = load_config(ROOT / "configs" / "qwen3_4b_full.yaml")
    scenario = load_dataset(
        config.resolved_scenario_path, config.scenario_sha256
    ).scenarios[0]
    original = copy.deepcopy(scenario)
    turns = list(iter_turns(scenario, "evidence_only"))
    assert turns
    assert sum(is_final for _, is_final in turns) == 1
    assert turns[-1][1] is True
    assert all("time_stamp" in message for pair, _ in turns for message in pair)
    assert scenario == original
