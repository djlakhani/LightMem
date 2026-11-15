# LightMem on STALE

This directory contains a reproducible evaluation harness for running the full
LightMem pipeline on the 50 fixed scenarios in `scenarios.json`.

The default evaluation uses the distractor-heavy `truncated` view, Qwen3-4B for
all memory-system and user-facing generation, embedded Qdrant for vector storage,
and `gpt-6-luna` for independent judging. Qwen thinking is explicitly disabled
and checked on every memory-system request and user-facing answer.

## RunPod quick start

The expected pod has one RTX A6000 and a persistent volume mounted at
`/workspace`. The pod must expose `OPENAI_API_KEY`; AWS credentials are expected
to come from its existing role or environment.

```bash
cd LightMem/evaluations/STALE
scripts/bootstrap_runpod.sh
scripts/run_runpod.sh
```

Persistent caches and results are stored beneath `/workspace/lightmem-stale`.
The bootstrap script downloads exact, pinned model revisions into
`/workspace/lightmem-stale/models`.
Completed checkpoints and final artifacts are uploaded under:

```text
s3://stale-qwen-evals-589498925064-us-east-1-an/lightmem/stale/qwen3-4b/<run-id>/
```

No S3 deletion operation is implemented.

## Staged commands

Start the local Qwen server in one terminal:

```bash
scripts/start_qwen.sh
```

Then use the CLI from another terminal:

```bash
uv run stale-lightmem --config configs/qwen3_4b_full.yaml preflight
uv run stale-lightmem --config configs/qwen3_4b_full.yaml run --no-upload --limit 1
uv run stale-lightmem --config configs/qwen3_4b_full.yaml judge --run-dir /workspace/lightmem-stale/runs/<run-id>
uv run stale-lightmem --config configs/qwen3_4b_full.yaml report --run-dir /workspace/lightmem-stale/runs/<run-id>
uv run stale-lightmem --config configs/qwen3_4b_full.yaml upload --run-dir /workspace/lightmem-stale/runs/<run-id>
```

`run` and `judge` are resumable. A checkpoint is reused only when its scenario
and configuration hashes match. Use `hydrate` to restore a run from S3 before
resuming it on a different pod.

## Qdrant behavior

Qdrant runs in embedded mode through `qdrant-client`; there is no extra Docker
service. Extracted memory text, timestamps, and other payload metadata are stored
alongside 384-dimensional vectors. The collection is reset between scenarios.
The database itself is scratch state and is not uploaded, while pre/post memory
snapshots and every scored retrieval are retained in the run artifacts.

## Outputs

- `manifest.json`: versions, model revisions, hashes, hardware, and configuration.
- `scenario_records.jsonl`: pre/post memories, retrieval traces, prompts, answers,
  time, and token usage.
- `judgments.jsonl`: all judge decisions and usage.
- `summary.json` and `report.md`: overall, T1/T2, SR/PR/IPA, and intersection
  accuracies.
- `checkpoints/`: atomic per-scenario generation and judge checkpoints.

Run CPU-only tests with `uv run pytest`.

