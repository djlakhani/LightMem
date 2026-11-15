from pathlib import Path

from stale_lightmem.artifacts import atomic_write_json, completed_checkpoint
from stale_lightmem.backend import diff_memories
from stale_lightmem.config import S3Config
from stale_lightmem.judge import parse_judgment
from stale_lightmem.s3 import S3ArtifactStore, sha256_file


def test_parse_judgment_validates_all_dimensions():
    parsed = parse_judgment(
        '{"dim1_eval":{"reasoning":"a","pass":true},'
        '"dim2_eval":{"reasoning":"b","pass":false},'
        '"dim3_eval":{"reasoning":"c","pass":true}}'
    )
    assert parsed["dim2_eval"]["pass"] is False


def test_completed_checkpoint_requires_matching_hashes(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    atomic_write_json(
        path,
        {"status": "complete", "config_hash": "config", "scenario_hash": "scenario"},
    )
    assert completed_checkpoint(path, "config", "scenario") is not None
    assert completed_checkpoint(path, "other", "scenario") is None


def test_memory_diff_ignores_update_queue_metadata():
    before = [
        {"id": "a", "payload": {"memory": "same"}},
        {"id": "b", "payload": {"memory": "old"}},
    ]
    after = [
        {"id": "a", "payload": {"memory": "same", "update_queue": [{"id": "x"}]}},
        {"id": "b", "payload": {"memory": "new"}},
    ]
    result = diff_memories(before, after)
    assert result["updated_ids"] == ["b"]
    assert result["unchanged_ids"] == ["a"]


def test_s3_upload_uses_immutable_run_prefix_and_checksum(tmp_path: Path):
    source = tmp_path / "artifact.json"
    source.write_text("{}\n", encoding="utf-8")

    class FakeS3:
        def __init__(self):
            self.upload = None

        def upload_file(self, filename, bucket, key, ExtraArgs):
            self.upload = (filename, bucket, key, ExtraArgs)

        def head_object(self, Bucket, Key):
            return {"Metadata": {"sha256": sha256_file(source)}}

    store = object.__new__(S3ArtifactStore)
    store.config = S3Config(bucket="bucket", prefix="prefix")
    store.run_id = "run-id"
    store.client = FakeS3()
    key = store.upload_file(source, "checkpoints/item.json")
    assert key == "prefix/run-id/checkpoints/item.json"
    assert store.client.upload[1] == "bucket"
    assert store.client.upload[3]["Metadata"]["sha256"] == sha256_file(source)
