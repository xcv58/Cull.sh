# Cull.sh Architecture

## Purpose

Cull.sh is a local-first, CLI-driven photo workflow. It owns discovery,
technical analysis, scene grouping, candidate ranking, semantic culling,
human-auditable manifests, and optional AI edit recipes. Renderers remain
separate tools.

The current production split is:

- Cull.sh decides what to review and records why.
- Ollama supplies structured semantic culling and edit suggestions.
- RapidRAW renders approved edit recipes and exports final files.
- Lightroom-compatible XMP remains an optional interoperability path.

## Safety Invariants

- Never delete, rename, or overwrite source photographs.
- Dry run is the default for culling and XMP edit commands.
- Merge existing XMP rather than replacing unrelated metadata.
- Keep TOPIQ out of the hard quality-reject vote.
- Fail fast when required TOPIQ scores or Qwen production responses are unavailable.
- Stage RapidRAW work in a new isolated directory.
- Require a manifest-matched approval file before final RapidRAW export, unless
  the caller explicitly passes `--approve-all`.
- Persist enough provenance to reproduce or audit every decision.

## Culling Pipeline

### 1. Discovery and grouping

`cull_sh.scanner` recursively discovers supported RAW and optional JPEG files,
ignores AppleDouble `._*` companions, and builds `RawAsset` records.
`cull_sh.grouping` assigns provisional scene groups from folder, timestamps,
and filename continuity.

### 2. Preview extraction and local analysis

`cull_sh.extractors` uses ExifTool as the primary proprietary-RAW preview
extractor. Local analysis computes Laplacian, Tenengrad, brightness, contrast,
optional face/eye hints, learned IQA support metrics, and a scene-relative local
rank. Multiple weak technical signals are required for a hard local reject.

### 3. TOPIQ candidate ranking

Standalone TOPIQ-NR runs on every preview when production ranking is enabled.
Folder-relative local and TOPIQ percentiles are blended at 75% and 25%,
respectively. TOPIQ affects candidate ordering only; it cannot independently
reject a photograph. Missing TOPIQ coverage fails before metadata persistence.

The shadow artifact records the strongest local/TOPIQ disagreements without
changing decisions. `--no-topiq-ranking` restores local-only ordering;
`--no-topiq-shadow` disables only the review artifact.

### 4. Semantic vision scoring

The provider-agnostic backend receives only viable, scene-ranked previews and a
resolved prompt. Responses are schema-validated and normalized into ratings,
labels, pick/review/reject buckets, and audit text. Qwen 27B is the shared
production semantic model for culling and editing. It runs with thinking enabled,
one attempt, no fallback model, and aborts the run after the first failed cohort.
The deterministic local gate and TOPIQ-assisted ranking remain the primary culling
structure; Qwen supplies the final semantic triage rather than acting as a
standalone selector. Ollama generation is capped at 2,048 tokens with
`num_predict`; this leaves bounded space for thinking plus the final JSON response.

### 5. Persistence

Run configuration and `manifest.jsonl` are written incrementally. With
`--no-dry-run`, Cull.sh merges Lightroom-compatible culling state into RAW XMP
or JPEG embedded metadata. Source image bytes are never rewritten for RAWs.

## AI Develop and RapidRAW Pipeline

### 1. Suggest

`suggest-edits` selects culled, non-rejected RAWs and asks
`orcarouter/Qwen3.8-27B-Uncensored` for a bounded RapidRAW recipe. The executable
surface includes global tone, white-balance/color, presence, detail/noise,
vignette, crop, and safely cropped rotation controls. The model may also preserve
unbounded editing ideas as explicit `additional_edits`; these notes are never
represented as rendered changes. Editing uses one image per request, is
fail-fast with one attempt and no fallback, and freezes recipe plus
model/prompt provenance in `edit-suggestions.jsonl`.
The transport schema requires an explicit value for every executable control,
while persisted recipes retain backward-compatible neutral defaults.

### 2. Stage

`rapidraw-stage` creates a new directory, copies each source RAW, translates the
recipe into a colocated `.rrdata` sidecar, and writes an immutable
`rapidraw-manifest.json` plus an approval template. Duplicate basenames are
disambiguated. Scalar bounds, rotation-with-crop, crop geometry, and a
minimum retained crop area are validated before any copy is rendered. Spatial
operations such as masks and healing remain explicit future-work intents until
a mask-generation stage can supply real geometry or mask pixels.

### 3. Preview and approve

The bounded validation stage receives the neutral and rendered pair plus actual
decoded dimensions and measured outer-edge facts. This prevents aspect-ratio
padding introduced by vision preprocessing from being reported as image
letterboxing. The validator may accept, reject, or make one bounded refinement;
the human preference recorded by the review page remains authoritative.

`rapidraw-preview` invokes RapidRAW's headless CLI on the staged copies and
builds a private `review.html` with actual before/after renders. Choices are
stored in the browser locally and downloaded as `rapidraw-approvals.json`.

### 4. Export

`rapidraw-export` verifies the approval file's SHA-256 stage binding and exports
only approved records. It keeps EXIF metadata by default, fails on an
unrecognized pre-existing output, and records per-file completion in
`rapidraw-export.json` so a corrected run can resume without repeating successful
exports.

## Model Policy

- Culling: retain the validated local/TOPIQ structure with Qwen 27B semantic triage.
- Candidate rank: local percentile 75%, TOPIQ-NR percentile 25%.
- Hard reject: deterministic multi-signal technical gate; no TOPIQ vote.
- Production model: Qwen 27B with thinking enabled, one attempt, fail-fast, no fallback.
- Gemma: historical benchmark support only; not a production dependency.
- Facet: benchmark/reference source only; do not depend on its complete pipeline.

## Primary Modules

- `cull_sh.cli`: commands, validation, and user-facing completion summaries.
- `cull_sh.pipeline`: culling phase orchestration and persistence boundaries.
- `cull_sh.quality`: local and TOPIQ scoring.
- `cull_sh.ranking`: scene-relative ranking and TOPIQ percentile blend.
- `cull_sh.backends`: structured provider contracts and Ollama implementation.
- `cull_sh.benchmark`, `edit_benchmark`, `vlm_benchmark`, `culling_blind`:
  frozen evaluations and hidden-identity human review.
- `cull_sh.shadow`: capped local/TOPIQ disagreement review.
- `cull_sh.rapidraw`: isolated staging, real-render review, approval enforcement,
  export provenance, and resume state.
- `cull_sh.xmp`: Lightroom-compatible merge-safe metadata writes.

## Artifact Contract

Every material run lives under a timestamped run directory or an explicitly
chosen RapidRAW stage. Artifacts are local and gitignored. A run is complete
only when its manifest exists, expected record counts match, required sidecars
or exports exist, and the command exits successfully. A prospective benchmark
must retain the original pre-review run unchanged and later compare it with the
final human XMP through `benchmark --shadow-run`.
