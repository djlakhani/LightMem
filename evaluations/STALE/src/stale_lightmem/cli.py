from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runner import (
    create_run_dir,
    generate_report,
    preflight,
    run_generation,
    run_judging,
    upload_run,
)
from .s3 import S3ArtifactStore

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qwen3_4b_full.yaml"


def _run_dir(args: argparse.Namespace, config, create: bool = False) -> Path:
    if args.run_dir:
        path = Path(args.run_dir).resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    if create:
        return create_run_dir(config, getattr(args, "run_id", None))
    raise ValueError("--run-dir is required for this command")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LightMem on fixed STALE scenarios"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("preflight")
    pre.add_argument("--skip-s3", action="store_true")
    pre.add_argument("--skip-gpu", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--run-dir")
    run.add_argument("--run-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--no-upload", action="store_true")

    judge = subparsers.add_parser("judge")
    judge.add_argument("--run-dir", required=True)
    judge.add_argument("--no-upload", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--allow-partial", action="store_true")

    upload = subparsers.add_parser("upload")
    upload.add_argument("--run-dir", required=True)

    all_cmd = subparsers.add_parser("all")
    all_cmd.add_argument("--run-dir")
    all_cmd.add_argument("--run-id")
    all_cmd.add_argument("--limit", type=int)
    all_cmd.add_argument("--no-upload", action="store_true")
    all_cmd.add_argument("--allow-partial", action="store_true")
    all_cmd.add_argument("--skip-preflight", action="store_true")

    hydrate = subparsers.add_parser("hydrate")
    hydrate.add_argument("--run-id", required=True)
    hydrate.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        print(
            json.dumps(
                preflight(
                    config, check_s3=not args.skip_s3, check_gpu=not args.skip_gpu
                ),
                indent=2,
            )
        )
        return
    if args.command == "run":
        run_dir = _run_dir(args, config, create=True)
        records = run_generation(
            config, run_dir, limit=args.limit, upload=not args.no_upload
        )
        print(
            json.dumps({"run_dir": str(run_dir), "completed": len(records)}, indent=2)
        )
        return
    if args.command == "judge":
        run_dir = _run_dir(args, config)
        records = run_judging(config, run_dir, upload=not args.no_upload)
        print(json.dumps({"run_dir": str(run_dir), "judged": len(records)}, indent=2))
        return
    if args.command == "report":
        summary = generate_report(
            config, _run_dir(args, config), allow_partial=args.allow_partial
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "upload":
        uploaded = upload_run(config, _run_dir(args, config))
        print(json.dumps({"uploaded": len(uploaded)}, indent=2))
        return
    if args.command == "hydrate":
        count = S3ArtifactStore(config.s3, args.run_id).download_run(Path(args.run_dir))
        print(json.dumps({"downloaded": count, "run_dir": args.run_dir}, indent=2))
        return
    if args.command == "all":
        run_dir = _run_dir(args, config, create=True)
        if not args.skip_preflight:
            preflight(config, check_s3=not args.no_upload)
        run_generation(config, run_dir, limit=args.limit, upload=not args.no_upload)
        run_judging(config, run_dir, upload=not args.no_upload)
        summary = generate_report(
            config, run_dir, allow_partial=args.allow_partial or bool(args.limit)
        )
        if not args.no_upload:
            upload_run(config, run_dir)
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
