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

The server defaults to a 16,384-token context window. Override it with
`QWEN_MAX_MODEL_LEN` only if the GPU has enough KV-cache capacity; changing the
value requires restarting the server. The evaluation preflight includes a
production-shaped dependency-resolution smoke (four anchors with twelve future
candidates each), verifies that the running server actually reports a 16,384-token
context, and fails with restart instructions if an old 8,192-token process is
still serving. Run it after every server restart and before a long run.

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

## Category-aware dependency resolution

The full profile assigns every extracted memory exactly two labels from the
fixed 20-label taxonomy. The labels must come from different parent groups:
Living & logistics, Work & finance, Health & personal, and Social & other.
Labeling is part of the existing extraction request rather than a separate LLM
call.

Label formatting variants are canonicalized locally. If the model still
returns invalid labels after one repair request, the existing embedding model
selects a valid cross-group pair instead of aborting the scenario. The payload
records `taxonomy_label_status`, the original labels, and the validation error
so fallback assignments remain auditable.

For each STALE question, the answer pipeline:

1. Retrieves the normal top 20 memories and asks Qwen to select four anchors.
2. For each anchor, retrieves later memories sharing a leaf label, retaining the
   top eight by semantic similarity plus four additional most-recent memories.
3. Requires Qwen to classify every anchor/candidate pair, select at most three
   updates, invalidations, or append operations, and summarize the current state.
4. Gives the four resolved anchor packages to a separate final-answer request.

The anchor-selection prompt intentionally omits category labels. All prompts,
candidate pools, pairwise verdicts, selected changes, reasons, timing, and token
usage are retained in `scenario_records.jsonl`.

## Outputs

- `manifest.json`: versions, model revisions, hashes, hardware, and configuration.
- `scenario_records.jsonl`: pre/post memories, retrieval traces, prompts, answers,
  time, and token usage.
- `judgments.jsonl`: all judge decisions and usage.
- `summary.json` and `report.md`: overall, T1/T2, SR/PR/IPA, and intersection
  accuracies.
- `checkpoints/`: atomic per-scenario generation and judge checkpoints.

Run CPU-only tests with `uv run pytest`.

