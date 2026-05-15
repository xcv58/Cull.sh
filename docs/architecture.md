# Cull.sh Architecture

## Purpose

`Cull.sh` is a CLI-first photo culling tool for macOS that combines:

- fast local analysis for obvious rejects
- vision-model scoring for semantic and compositional judgment
- non-destructive Lightroom-compatible output through `.xmp` sidecars

The architecture is intentionally provider-agnostic so the same pipeline can use:

- local Ollama models for development
- Anthropic for cloud semantic scoring
- OpenAI for cloud semantic scoring

## Product Principles

- Never modify or delete RAW files.
- Avoid expensive vision calls for frames that are obviously bad.
- Preserve existing `.xmp` data when possible and update only culling-related fields.
- Keep the CLI predictable, resumable, and auditable.
- Separate orchestration from extraction, scoring, and output writing.
- Prefer deterministic guided CLI setup over a free-form preflight chat with a model.

## High-Level Pipeline

### Phase 0: Discovery

1. Scan the target directory recursively.
2. Select supported RAW extensions such as `.arw`, `.cr2`, `.cr3`, `.nef`, `.raf`, `.dng`, and `.orf`.
3. Build an internal work item for each RAW file.
4. Assign provisional scene groups using parent folder, file ordering, modification-time gaps, and numeric filename continuity.

### Phase 1: Local Filtering

1. Extract an embedded JPEG preview from each RAW.
2. Decode the preview into an image matrix.
3. Compute fast quality metrics:
   - sharpness via Laplacian variance
   - edge strength via Tenengrad
   - brightness mean
   - contrast standard deviation
   - optional portrait hints via local face / eye detection
4. Reject images only when multiple technical sharpness metrics fall below threshold, instead of relying on a single blur scalar.
5. Compute a local rank score per scene and suppress near-duplicates before vision scoring.
6. Preserve scene grouping and local metrics in run artifacts for later burst ranking.

### Phase 2: Semantic Vision Scoring

1. Batch only phase-1 survivors after scene-relative reranking.
2. Send preview images plus the user prompt to a provider backend.
3. Require structured output with:
   - filename
   - rating
   - label
   - keep / reject decision
   - reasoning summary for logs only
4. Validate model output against a schema before applying it.

### Phase 3: Sidecar Writing

1. Compute the `.xmp` sidecar path for each RAW file.
2. If a sidecar exists, merge changes instead of replacing the file wholesale.
3. Update Lightroom-relevant fields only:
   - `xmp:Rating`
   - `xmp:Label`
4. Preserve unrelated metadata.

### Phase 4: Run Artifacts

Store run metadata in `runs/<timestamp>/`:

- `manifest.jsonl`: one line per image
- `config.json`: resolved CLI and backend configuration
- optional cached previews for debugging
- optional evaluation exports

Runs should be persisted incrementally so a long vision phase does not leave the user blind:

- write `config.json` at startup
- rewrite `manifest.jsonl` after extraction
- rewrite `manifest.jsonl` after local filtering
- rewrite `manifest.jsonl` after every completed vision batch
- write sidecars as decisions become available when not in dry-run mode

## Modules

### `cull_sh.cli`

User-facing command definitions with Typer.

Responsibilities:

- parse arguments
- resolve prompt strategy from explicit prompt, genre preset, or guided setup
- construct `PipelineConfig`
- run the pipeline
- render console output
- show progress for long-running phases

### `cull_sh.prompting`

Prompt-resolution helpers.

Responsibilities:

- genre preset definitions
- generated prompt templates
- guided setup support
- prompt selection metadata

### `cull_sh.config`

Application configuration and defaults.

Responsibilities:

- pipeline settings
- backend selection
- worker counts
- thresholds

### `cull_sh.models`

Core dataclasses and enums shared across the package.

Responsibilities:

- RAW work items
- preview payloads
- local metrics
- final decisions
- batch request / response structures

### `cull_sh.scanner`

Filesystem discovery.

Responsibilities:

- recursive RAW discovery
- extension filtering
- path normalization

### `cull_sh.extractors`

Preview extraction interfaces and implementations.

Responsibilities:

- `ExifToolPreviewExtractor`
- `SipsPreviewExtractor`
- fallback selection
- future in-memory extraction support

Notes:

- `exiftool` should be the primary extractor for proprietary RAW files.
- `sips` is a useful fallback but should not be treated as equivalent for all formats.

### `cull_sh.quality`

Fast local quality metrics.

Responsibilities:

- preview decoding
- multi-metric sharpness scoring
- brightness / contrast heuristics
- portrait-aware face and eye hints
- local thresholds
- future ranking helpers

### `cull_sh.grouping`

Provisional scene and burst grouping.

Responsibilities:

- assign stable `scene_id` values before local analysis
- group by folder, timestamp gaps, and filename continuity
- provide grouping hooks for later duplicate suppression and per-scene ranking

### `cull_sh.backends.base`

Provider-agnostic vision backend contract.

Responsibilities:

- batch scoring interface
- structured result contract

### `cull_sh.backends.ollama`

Local development backend via Ollama.

Responsibilities:

- call Ollama `chat` API
- send preview images
- request structured JSON output

### `cull_sh.pipeline`

Application orchestrator.

Responsibilities:

- coordinate discovery, extraction, quality checks, backend scoring, and output writing
- isolate phase transitions
- support dry runs and resume later

### `cull_sh.xmp`

Lightroom-compatible sidecar writing.

Responsibilities:

- create or merge XMP sidecars
- map internal decision fields to Lightroom fields

## Data Model

### Work Item

One RAW file and its derived state during a run.

Suggested fields:

- raw path
- xmp path
- preview source
- preview bytes or preview cache path
- scene id
- scene index
- local quality metrics
- local decision
- backend result
- final decision

### Final Decision

Normalized result written to output:

- rating: integer from `-1` to `5`
- label: optional Lightroom color label
- keep: boolean
- source: `local` or backend provider
- notes: optional audit text

## Concurrency Strategy

### Extraction

Extraction is mostly subprocess and file I/O bound.

Approach:

- use a bounded `ThreadPoolExecutor` or process pool wrapper for extractor subprocesses
- avoid unbounded fan-out
- limit concurrency by CPU count and disk behavior

### Blur Scoring

Local quality scoring is CPU-bound image analysis.

Approach:

- use `ProcessPoolExecutor`
- keep OpenCV worker functions top-level and pickle-friendly
- set `cv2.setNumThreads(1)` inside workers to avoid oversubscription
- compute multiple metrics in a single decode pass to avoid duplicate image decoding work

### Vision Scoring

Vision scoring is network-bound or local inference bound.

Approach:

- batch per provider limits
- bound request concurrency separately from extraction workers
- retry malformed structured responses

## Backend Contract

Every vision backend should implement a consistent interface:

- input: batch of preview-bearing work items + prompt
- output: validated culling decisions keyed by filename or work item id

The pipeline must not depend on provider-specific response formats.

## Configuration Surface

Planned CLI flags:

- `--path`
- `--prompt`
- `--genre`
- `--prefer`
- `--interactive`
- `--limit`
- `--provider`
- `--model`
- `--backend-url`
- `--batch-size`
- `--extract-workers`
- `--score-workers`
- `--min-blur-score`
- `--extensions`
- `--lightroom-auto-edit`
- `lightroom-adaptive-color`
- `--dry-run`
- `--resume`

## Error Handling

- extractor failure should not abort the whole run
- malformed backend output should be retried or marked for manual review
- sidecar write failures should be reported with file-level detail
- partial runs should still produce manifest records
- long-running phases should expose progress so users can distinguish slow work from a hung run

## Prompt Strategy

`Cull.sh` should not require users to hand-write a prompt for every shoot.

Recommended order:

1. Use an explicit `--prompt` when the user wants full control.
2. Otherwise use a genre preset plus optional preference.
3. In interactive sessions, ask the user for genre and preference with a short guided flow.
4. Use `auto` only as a broad generic preset until true genre inference is implemented.

## Build Order

### Milestone 1

- repo scaffold
- CLI
- RAW discovery
- config and data models

### Milestone 2

- `exiftool` preview extraction
- local blur scoring
- manifest logging

### Milestone 3

- Ollama backend
- structured result validation
- prompt formatting and batching

### Milestone 4

- XMP merge and write support
- dry-run and resume support

### Milestone 5

- cloud backends
- burst grouping
- evaluation and prompt tuning tooling
