# Cull.sh

`Cull.sh` is a natural-language-driven CLI for photo culling.

The intended workflow is:

1. Discover proprietary RAW files and, by default, JPEG files.
2. Extract preview images without modifying source photos.
3. Run fast local quality checks such as blur detection.
4. Send only viable candidates to a vision model backend.
5. Write Lightroom-compatible culling metadata with ratings and labels.
6. Generate fail-fast local AI develop suggestions for selected RAWs.
7. Review and render approved edits through RapidRAW, or optionally write XMP.

The repository is scaffolded around a provider-agnostic backend interface so local models such as Ollama can be used for development, while Anthropic or OpenAI can be added later without changing the pipeline shape.

## Layout

- `main.py`: thin entrypoint
- `cull_sh/`: application package
- `docs/architecture.md`: architecture plan and build order
- `requirements.txt`: bootstrap dependencies

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --help
python main.py doctor
```

If you plan to use the Ollama backend, make sure the local API is running on
`http://localhost:11434`. `python main.py doctor` now probes that endpoint directly.

## Current Status

This scaffold includes:

- CLI entrypoint with Typer
- provider-agnostic backend interface
- Ollama backend implementation
- RAW and JPEG discovery
- provisional scene grouping
- multi-metric local preview analysis
- TOPIQ NR assisted candidate ranking with a thumbnail disagreement review
- portrait-aware local face / eye hints
- scene-relative candidate reranking
- near-duplicate suppression before vision scoring
- run manifests under `runs/<timestamp>/manifest.jsonl`
- XMP sidecar writing with merge support for existing RAW sidecars
- embedded JPEG culling metadata
- optional Lightroom sidecar edits for RAWs
- shared Qwen culling and develop suggestions with thinking enabled and no fallback model
- approval-gated RapidRAW staging, preview rendering, and final export
- architecture plan for the full app

## Current Workflow

```bash
python main.py doctor
python main.py benchmark --path /path/to/human-reviewed/raws --facet-db /path/to/facet.db
python main.py benchmark --path /path/to/newly-reviewed/raws --shadow-run runs/<original-cull-run>
python main.py blind-culling-test --frozen-run runs/<prospective-cull-run>
python main.py blind-culling-sample --folder-count 12 --photos-per-folder 1 --exclude-folder "2026-05-06 Kuala Lumpur"
python main.py culling-ground-truth-test --folder-count 13 --total-photos 30 --exclude-run runs/culling-blind-tests/<earlier-run>
python main.py score-blind-culling --run-dir runs/culling-blind-tests/<timestamp> --choices ~/Downloads/blind-culling-choices.csv
python main.py score-culling-ground-truth --run-dir runs/culling-ground-truth-tests/<timestamp> --choices ~/Downloads/culling-ground-truth-choices.csv
python main.py cull --path /path/to/raws --prompt "Keep the sharpest wildlife photos"
python main.py cull --path /path/to/raws --genre portrait
python main.py cull --path /path/to/raws --genre street --prefer "interesting gestures and layering"
python main.py cull --path /path/to/raws --genre flowers --limit 24
python main.py cull --path /path/to/raws --prompt "Keep the sharpest wildlife photos" --no-dry-run
python main.py cull --path /path/to/raws --genre event --lightroom-auto-edit --no-dry-run
python main.py lightroom-edit --path "/path/to/culled raws" --no-dry-run
python main.py lightroom-adaptive-color --path "/path/to/culled raws"
python main.py lightroom-jpeg-auto --path "/path/to/culled raws"
python main.py suggest-edits --path "/path/to/culled raws"
python main.py suggest-edits --path "/path/to/culled raws" --no-dry-run
python main.py rapidraw-stage --suggestions runs/<timestamp>/edit-suggestions.jsonl --output /path/to/new-stage
python main.py rapidraw-preview --stage /path/to/new-stage
python main.py rapidraw-export --stage /path/to/new-stage --approvals ~/Downloads/rapidraw-approvals.json
python main.py repair-sidecars
python main.py repair-sidecars --run-dir runs/20260411-220756-823242
```

Both `cull` and `suggest-edits` use the local
`orcarouter/Qwen3.8-27B-Uncensored` model by default, with thinking enabled.
Production model calls are fail-fast: each command makes one attempt, uses no
fallback model, and stops on the first failed cohort so the underlying local-model
problem can be fixed. Both culling and editing also send a bounded Ollama `num_predict` value
(`--backend-max-output-tokens`, default 2048) so thinking has room to finish while
a malformed structured response
cannot generate indefinitely while keeping the HTTP connection active.

`benchmark` reads existing Lightroom XMP picks, ratings, and rejects as human
ground truth without modifying photos or sidecars. It compares the current
MUSIQ/NIMA/local-quality signals with any cached Facet signals, writes a
resumable current-signal cache, and reports global AUC, within-burst ranking,
false-reject safety, and the behavior of the current local reject gate under
`runs/<timestamp>/`.

`blind-culling-test` compares Gemma 4 and Qwen 27B on the semantic candidates
already locked by a prospective dry run. It reuses the frozen prompt, reads no
XMP, writes no photo metadata, randomizes the per-photo A/B assignment, and
keeps the answer key outside the generated HTML. Complete the review page,
download `blind-culling-choices.csv`, and reveal the result only with
`score-blind-culling`.

`blind-culling-sample` builds a seeded, reproducible sample across incomplete
top-level photo folders without reading XMP. Completed `DONE`/`EXPORTED`
folders, hidden files, and explicitly excluded folders are skipped. Use one
photo per folder for maximum subject diversity, or multiple photos with
`--min-sequence-gap` to prevent adjacent filename runs.

For model-selection decisions, prefer `culling-ground-truth-test`. Its review
page shows only each photograph and asks the reviewer to independently assign
reject, review, or pick. Model decisions, explanations, and identities remain
hidden until scoring, avoiding explanation-quality and A/B anchoring bias.
`--total-photos` is balanced across the requested folder count, and repeated
`--exclude-run` options prevent reuse of earlier blind-test photos.

Regular `cull` runs enable `--topiq-ranking` by default. Folder-relative TOPIQ
and local-score percentiles are blended with a 25% TOPIQ weight for candidate
ordering; TOPIQ does not participate in the hard quality-reject gate. Scores
are saved to the manifest and `--topiq-shadow` controls a capped disagreement
review artifact. Use `--no-topiq-ranking` to restore local-only ordering and
`--no-topiq-shadow` to skip the review page without disabling ranking.

For a prospective benchmark, keep the original cull run, finish reviewing the
folder normally in Lightroom, then pass that run to `benchmark --shadow-run`.
The benchmark reuses the frozen pre-review TOPIQ/current scores and decisions,
compares them with the final XMP, and does not rescore the folder.

If you omit `--prompt`, the CLI can generate one from a genre preset:

- `auto`: broad cross-genre prompt
- `mixed`
- `portrait`
- `street`
- `flowers`
- `wildlife`
- `landscape`
- `event`
- `product`

In an interactive terminal, omitting `--prompt` starts a guided setup so the app can
ask for genre and any extra preference. This is intentionally deterministic and
cheaper than adding a separate model-driven “conversation” step before every run.

In `--dry-run` mode, the pipeline still:

- discovers RAW and JPEG files
- extracts previews
- computes blur scores
- scores surviving images with the configured backend
- writes run artifacts under `runs/`
- shows progress during extraction, local scoring, and vision scoring

With `--no-dry-run`, it writes or updates `.xmp` sidecars next to each RAW and
embedded XMP culling metadata inside JPEG files.

`--lightroom-auto-edit` extends culling sidecar writes for RAW photos with
safe Lightroom sidecar edits:

- lens profile corrections enabled

Cull.sh does not invent Lightroom's per-image Adaptive Color AI payload. The
validated path is to let Lightroom apply that profile through the UI or a preset
automation pass, because the visible Adaptive Color result requires Lightroom's
generated `crs:AILook` data. The verifier counts Lightroom `.acr` payloads for
sidecar RAW files and embedded DNG XMP when Lightroom writes the AI payload back
into a DNG.

For folders containing DNG files, Lightroom may show an Adaptive Profile refresh
warning after metadata is written back into the DNG. The generated
`lightroom-adaptive-color` Computer Use handoff includes a DNG-only cleanup:
filter the folder to DNGs, select the visible DNG set, run
`Photo > Update AI Settings`, and wait for Lightroom to finish the batch. Repeat
that DNG-only update after any post-Lightroom sidecar/DNG metadata repair.

By default, Lightroom edit stages process all RAWs, including rejected photos,
so the whole folder has the same baseline treatment if you later rescue a reject.
Use `--lightroom-edit-scope kept` on `cull`, `lightroom-edit`, or
`lightroom-adaptive-color` to restore the older kept-only behavior.

JPEG culling is enabled by default on `cull`. Same-stem RAW+JPEG pairs mirror
the RAW cull decision by default so paired JPEGs do not consume extra vision
scoring unless you pass `--score-paired-jpegs`. Use `--raw-only` when you want
the older RAW-only culling behavior.

For folders that have already been culled, use `lightroom-edit` to apply the
same sidecar edit to every RAW:

```bash
python main.py lightroom-edit --path "/path/to/culled-raws" --dry-run
python main.py lightroom-edit --path "/path/to/culled-raws" --no-dry-run
```

Then use `lightroom-adaptive-color` to prepare the Computer Use UI stage for
Adaptive Color:

```bash
python main.py lightroom-adaptive-color --path "/path/to/culled-raws"
```

This writes `lightroom-adaptive-color.md` and `.json` under a new `runs/`
directory. The checklist tells Computer Use to open Adobe Lightroom, filter to
the in-scope RAW photos, verify the visible count, select all visible photos, and
choose `Adaptive Color` directly from Lightroom's Profile dropdown. This profile
step must happen inside Lightroom so Lightroom can generate per-image
`crs:AILook` data. When DNGs are in scope, the checklist also tells Computer Use
to filter to the DNG subset and run `Photo > Update AI Settings` after the paste
so Lightroom refreshes the embedded Adaptive Profile render data.

JPEG Lightroom edits are separate because Adaptive Color/Profile is a RAW/DNG
feature. Use `lightroom-jpeg-auto` after culling to generate a Computer Use
handoff that filters to JPG/JPEG files, selects the in-scope JPEG set, and runs
`Photo > Apply Auto Settings` only:

```bash
python main.py lightroom-jpeg-auto --path "/path/to/culled-raws"
python main.py lightroom-jpeg-auto --path "/path/to/culled-raws" --lightroom-edit-scope kept
```

## AI Develop Edits

`suggest-edits` is an explicit opt-in stage. Regular culling does not run local
AI develop edits. It asks the configured vision model for natural global and
composition adjustments and freezes the complete RapidRAW recipes in
`edit-suggestions.jsonl`. It is a dry run by default; the Lightroom-compatible
subset is only written to source XMP sidecars when you pass `--no-dry-run`.
Rejected RAW files and RAW files without existing sidecars are skipped by
default, and existing culling state is preserved.

```bash
python main.py suggest-edits --path "/path/to/culled-raws"            # dry run: show suggestions only
python main.py suggest-edits --path "/path/to/culled-raws" --no-dry-run
python main.py suggest-edits --path "/path/to/culled-raws" --prefer "warm, punchy look"
python main.py suggest-edits --path "/path/to/culled-raws" --batch-size 1
python main.py suggest-edits --path "/path/to/raws" --include-unculled
```

The executable recipe covers exposure/brightness, contrast, highlights,
shadows, whites, blacks, temperature, tint, vibrance, saturation, clarity,
dehaze, structure, sharpening, luminance/color noise reduction, vignette,
crop, and rotation. Crop and rotation evaluation are enabled by default;
nonzero rotation requires crop bounds that remove RapidRAW's rotated black
edges. Useful ideas outside that executable surface—such as HSL,
curves, color grading, masks, healing, or lens work—are retained as explicit
`additional_edits` instead of being silently dropped or falsely reported as
applied.

Editing defaults to one image per model request to avoid cross-image leakage.
Every executable field is marked required in the structured model schema, so
Qwen must explicitly decide each expanded control instead of silently omitting
newer fields. Stored legacy recipes remain readable with neutral defaults.
The selected Qwen model runs with thinking enabled and is intentionally
fail-fast: Cull.sh does not use a fallback model or retry failed photographs
individually. Gemma remains supported by historical benchmark commands, but is
not a production dependency.

## RapidRAW Develop Workflow

This is an available experimental workflow, not the current production
finishing path. The current supported handoff is Lightroom-compatible XMP,
followed by manual finishing and JPEG export in Lightroom. RapidRAW development
is paused after its rendered results did not establish Lightroom parity; the
commands below remain documented so completed experiments stay reproducible.

The RapidRAW path has three explicit gates:

1. `rapidraw-stage` copies suggested RAWs into a new isolated directory, writes
   `.rrdata`, records the Cull.sh commit and suggestion provenance, and refuses
   to reuse an existing stage. Original RAW/XMP files are not modified.
2. `rapidraw-preview` runs the installed headless exporter and creates
   `review.html` with real before/after renders. Review choices stay local and
   download as `rapidraw-approvals.json`.
3. `rapidraw-export` verifies that the approval file matches the immutable
   stage manifest, exports only approved photographs, retains metadata by
   default, and records resumable completion in `rapidraw-export.json`.

```bash
python main.py rapidraw-stage \
  --suggestions runs/<timestamp>/edit-suggestions.jsonl \
  --output /path/to/new-stage
python main.py rapidraw-preview --stage /path/to/new-stage
python main.py rapidraw-export \
  --stage /path/to/new-stage \
  --approvals ~/Downloads/rapidraw-approvals.json
```

`rapidraw-export --approve-all` is available only as an explicit bypass for a
controlled pilot. Final export otherwise fails without a matching approval
file. Generated review and output files live inside the stage, never beside the
original photographs.

### Rendered edit feedback pilot

`rapidraw-feedback-pilot` tests a bounded render-and-review loop on picks that
were added by a frozen machine cull but were not picks in the pre-application
human XMP backup:

```bash
python main.py rapidraw-feedback-pilot \
  --path "/path/to/photos" \
  --cull-run runs/<frozen-cull-run> \
  --human-baseline "/path/to/pre-application-xmp-backup" \
  --output "/path/to/new-or-resumable-feedback-stage" \
  --include-existing-picks \
  --with-crop
```

By default the pilot retains its historical machine-added-picks-only scope.
`--include-existing-picks` selects the union of protected picks in the baseline
and new picks in the frozen cull manifest. It fails rather than silently omit a
protected pick that is absent from the manifest. For a clean automated-versus-
human experiment, use `--frozen-machine-picks` instead: every frozen manifest
pick is included and all pre-existing flags are ignored, preventing prior human
judgment from leaking into the automated arm. The two flags are mutually
exclusive.

For each selected RAW, RapidRAW first creates a neutral baseline render. Qwen
suggests a structured recipe from that renderer-consistent baseline, RapidRAW
renders the recipe, and an independent Qwen pass compares the real before/after
pair. Suggestions and rendered pairs default to one photo per request. The
review can accept, revert, or make one bounded refinement; it cannot iterate
indefinitely. Unrendered `additional_edits` never count as visible improvement
but remain available for later mask or advanced-edit stages.
The reviewer receives decoded image dimensions and measured outer-edge facts so
vision-model letterboxing outside a cropped image is not mistaken for black
pixels in the photograph. It also receives tonal measurements and native-pixel
detail patches. A final check inspects the exact accepted/reverted/refined pixels;
non-accept blocks unattended delivery without another automatic refinement.
This is a model quality gate, not a guarantee of human preference.
The generated `review.html` shows baseline, first edit, and final candidate side
by side and records a human preference locally.

The stage is resumable and never writes the source RAW/XMP files. Normalized
crop suggestions are converted to the full-resolution pixel coordinates used by
RapidRAW's headless export path, and implausibly small crop renders fail fast.
`--baseline-mode camera-midtones-v1` opts into an experimental camera-JPEG-guided
brightness calibration; neutral remains the default. New stages use schema 4
and `delivery-pixels-v2`; historical stages are not silently upgraded.

For an explicitly unattended experiment, export every completed Qwen-validated
recipe into a separate delivery folder:

```bash
python main.py rapidraw-feedback-export \
  --stage "/path/to/completed-feedback-stage" \
  --output "/path/to/delivery-jpegs" \
  --quality 95 \
  --keep-metadata \
  --unattended
```

The delivery folder contains JPEGs only. Export provenance and resumable state
remain in `feedback-export.json` inside the isolated stage. The exporter refuses
incomplete validation records, a source-photo folder as its destination, changed
completed JPEGs, or a non-empty output folder that it does not own. The explicit
`--unattended` acknowledgement is required because Qwen's rendered-edit verdict
replaces human approval in this mode. RapidRAW 1.6.1 retains supported EXIF such
as camera and capture time, but its current `--keep-metadata` path does not retain
GPS coordinates.

See [unattended quality validation](docs/unattended-quality-validation.md) for
renderer tests, baseline experiments, and the separate final-album selection
policy that resolves assisted triage reviews into unattended decisions.

`--limit` now applies after whole-folder scene grouping, so `--limit 24` means
"process the first 24 scenes" rather than "stop after 24 files".

`--batch-size` now controls the maximum images per same-scene cohort sent to the
vision backend for comparative scoring.

Longer runs now persist progress incrementally:

- local rejections can write sidecars before vision scoring starts
- each vision batch rewrites `manifest.jsonl`
- each completed vision batch writes sidecars immediately when `--no-dry-run` is enabled

## Lightroom Flags

Cull.sh writes Lightroom-visible culling state in three forms:

- `xmpDM:Pick=1` for kept images and `xmpDM:Pick=-1` for rejects
- `xmp:Label="Red"` for local blur rejects and `xmp:Label="Yellow"` for vision rejects
- `xmp:Rating="-1"` for rejects and `1..5` for kept images

If you need to repair sidecars from an older run without rescoring, use:

```bash
python main.py repair-sidecars
```

By default this replays the latest run manifest under `runs/`.
