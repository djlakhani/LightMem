# LightMem on STALE

This directory contains a reproducible evaluation harness for running the full
LightMem pipeline on the 50 fixed scenarios in `scenarios.json`.

The default evaluation uses the distractor-heavy `truncated` view, Qwen3-4B for
all memory-system and user-facing generation, embedded Qdrant for vector storage,
and `gpt-6-luna` for independent judging. Qwen thinking is explicitly disabled
and checked on every memory-system request and user-facing answer.

User-facing generation uses two Qwen stages. The first stage receives the full
top-20 retrieval trace and returns the four most relevant memory IDs as strict
JSON. The second stage receives only those selected memories, including their
IDs, timestamps, attribute labels, and text, and produces the final answer.

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

## Retrieved-memory evidence-attribution experiment

This second-stage experiment consumes an existing completed LightMem run. It
does **not** rebuild memories or rerun retrieval. For every SR, PR, and IPA
question it gives Qwen3-4B the exact stored top-20 retrieval trace and requires
a JSON answer plus exactly four ranked memory IDs.

The cited IDs are a self-reported attribution signal, not causal proof of model
reliance. A causal extension would rerun the same questions after masking or
removing selected memories and measure how the answers change.

Before generation, `gpt-6-luna` narrows the memory chunks attributed to the
M_old and M_new source messages, retaining only chunks that actually express
the corresponding state. UUID intersection is then computed deterministically;
the judge grades answer correctness and explains the verified old/new evidence
use. This avoids asking an LLM to perform error-prone UUID matching.

Run a one-scenario smoke test over the existing source run:

```bash
scripts/run_evidence_attribution.sh \
  20261118T050518Z-161fcbb3-4fa9f354 \
  --limit 1 \
  --no-upload
```

Run all 50 scenarios and upload the result:

```bash
scripts/run_evidence_attribution.sh \
  20261118T050518Z-161fcbb3-4fa9f354
```

The script starts Qwen automatically when port 8000 is not already healthy.
It hydrates the source run from the existing S3 prefix when necessary. New
artifacts are uploaded without deletion under:

```text
s3://stale-qwen-evals-589498925064-us-east-1-an/lightmem/stale/qwen3-4b/evidence-attribution/<run-id>/
```

The attribution run contains `evidence_labels.jsonl`, `cited_answers.jsonl`,
`attribution_judgments.jsonl`, resumable per-stage checkpoints, `summary.json`,
and a detailed `report.md`. The report shows source-attributed candidate chunks,
the narrowed M_old/M_new ID sets, all 20 retrieved memories, Qwen's four cited
IDs, exact prompts, answers, deterministic overlaps, and judge reasoning.

For staged operation, use `uv run stale-evidence-attribution --help` and the
`label`, `generate`, `judge`, `report`, and `upload` subcommands.

