from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import read_json
from .attribution_runner import (
    create_attribution_run_dir,
    generate_attribution_report,
    preflight_attribution,
    resolve_source_run,
    run_attribution_judging,
    run_cited_generation,
    run_labeling,
    upload_attribution_run,
)
from .config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qwen3_4b_full.yaml"


def _source_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-run-dir")
    group.add_argument("--source-run-id")


def _resolve_source(config, args: argparse.Namespace) -> Path:
    return resolve_source_run(
        config,
        source_run_dir=getattr(args, "source_run_dir", None),
        source_run_id=getattr(args, "source_run_id", None),
    )


def _run_dir(config, source: Path, args: argparse.Namespace, create: bool) -> Path:
    requested = getattr(args, "run_dir", None)
    if requested:
        path = Path(requested).expanduser().resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    if not create:
        raise ValueError("--run-dir is required")
    return create_attribution_run_dir(
        config,
        read_json(source / "manifest.json"),
        getattr(args, "run_id", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the STALE retrieved-memory evidence-attribution experiment"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--skip-s3", action="store_true")

    hydrate = commands.add_parser("hydrate-source")
    hydrate.add_argument("--source-run-id", required=True)

    label = commands.add_parser("label")
    _source_args(label)
    label.add_argument("--run-dir")
    label.add_argument("--run-id")
    label.add_argument("--limit", type=int)
    label.add_argument("--no-upload", action="store_true")

    generate = commands.add_parser("generate")
    _source_args(generate)
    generate.add_argument("--run-dir", required=True)
    generate.add_argument("--no-upload", action="store_true")

    judge = commands.add_parser("judge")
    _source_args(judge)
    judge.add_argument("--run-dir", required=True)
    judge.add_argument("--no-upload", action="store_true")

    report = commands.add_parser("report")
    _source_args(report)
    report.add_argument("--run-dir", required=True)
    report.add_argument("--allow-partial", action="store_true")

    upload = commands.add_parser("upload")
    upload.add_argument("--run-dir", required=True)

    all_command = commands.add_parser("all")
    _source_args(all_command)
    all_command.add_argument("--run-dir")
    all_command.add_argument("--run-id")
    all_command.add_argument("--limit", type=int)
    all_command.add_argument("--no-upload", action="store_true")
    all_command.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "preflight":
        print(
            json.dumps(
                preflight_attribution(config, check_s3=not args.skip_s3), indent=2
            )
        )
        return
    if args.command == "hydrate-source":
        source = resolve_source_run(config, source_run_id=args.source_run_id)
        print(json.dumps({"source_run_dir": str(source)}, indent=2))
        return
    if args.command == "upload":
        uploaded = upload_attribution_run(config, Path(args.run_dir).resolve())
        print(json.dumps({"uploaded": len(uploaded)}, indent=2))
        return

    source = _resolve_source(config, args)
    if args.command == "label":
        run_dir = _run_dir(config, source, args, create=True)
        records = run_labeling(
            config,
            source,
            run_dir,
            limit=args.limit,
            upload=not args.no_upload,
        )
        print(json.dumps({"run_dir": str(run_dir), "labeled": len(records)}, indent=2))
        return

    run_dir = _run_dir(config, source, args, create=args.command == "all")
    if args.command == "generate":
        records = run_cited_generation(
            config, source, run_dir, upload=not args.no_upload
        )
        print(
            json.dumps({"run_dir": str(run_dir), "generated": len(records)}, indent=2)
        )
        return
    if args.command == "judge":
        records = run_attribution_judging(
            config, source, run_dir, upload=not args.no_upload
        )
        print(json.dumps({"run_dir": str(run_dir), "judged": len(records)}, indent=2))
        return
    if args.command == "report":
        summary = generate_attribution_report(
            config,
            source,
            run_dir,
            allow_partial=args.allow_partial,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "all":
        if not args.skip_preflight:
            preflight_attribution(config, check_s3=not args.no_upload)
        run_labeling(
            config,
            source,
            run_dir,
            limit=args.limit,
            upload=not args.no_upload,
        )
        run_cited_generation(config, source, run_dir, upload=not args.no_upload)
        run_attribution_judging(config, source, run_dir, upload=not args.no_upload)
        summary = generate_attribution_report(
            config,
            source,
            run_dir,
            allow_partial=bool(args.limit),
        )
        uploaded = []
        if not args.no_upload:
            uploaded = upload_attribution_run(config, run_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "uploaded": len(uploaded),
                    "summary": summary,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
