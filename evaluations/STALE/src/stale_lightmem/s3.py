from __future__ import annotations

import hashlib
from pathlib import Path

from .config import S3Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class S3ArtifactStore:
    def __init__(self, config: S3Config, run_id: str):
        import boto3

        self.config = config
        self.run_id = run_id
        self.client = boto3.client("s3", region_name=config.region)

    @property
    def run_prefix(self) -> str:
        return f"{self.config.prefix.strip('/')}/{self.run_id}"

    def preflight(self) -> None:
        self.client.head_bucket(Bucket=self.config.bucket)

    def upload_file(self, path: Path, relative_path: str | None = None) -> str:
        relative = relative_path or path.name
        key = f"{self.run_prefix}/{relative.lstrip('/')}"
        checksum = sha256_file(path)
        self.client.upload_file(
            str(path),
            self.config.bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": checksum}},
        )
        head = self.client.head_object(Bucket=self.config.bucket, Key=key)
        if head.get("Metadata", {}).get("sha256") != checksum:
            raise RuntimeError(
                f"S3 checksum metadata verification failed for s3://{self.config.bucket}/{key}"
            )
        return key

    def upload_run(self, run_dir: Path) -> list[str]:
        uploaded: list[str] = []
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or "scratch" in path.relative_to(run_dir).parts:
                continue
            uploaded.append(self.upload_file(path, str(path.relative_to(run_dir))))
        return uploaded

    def download_run(self, destination: Path) -> int:
        destination.mkdir(parents=True, exist_ok=True)
        paginator = self.client.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(
            Bucket=self.config.bucket, Prefix=f"{self.run_prefix}/"
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                relative = key[len(self.run_prefix) + 1 :]
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.config.bucket, key, str(target))
                count += 1
        return count
