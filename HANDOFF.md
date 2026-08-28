# Cull.sh Handoff

## Current development iteration — August 28

- Branch: `codex/sentosa-unattended-quality`, based on `codex/topiq-qwen-rapidraw`.
- Regression suite: 160 passed, including four subtests; installed-app control
  tests also passed. The photographic quality evaluation is still running.
- Renderer semantics, final-pixel validation, experimental calibrated baselines,
  straightening evidence, cautious local rejects, and post-selection duplicate
  handling are implemented. See [quality validation](docs/unattended-quality-validation.md).
- Frozen Sentosa source/human/first-machine artifacts remain untouched.
- Live progress is in `runs/sentosa-quality-v2/pilot-v2/status.json`; the full
  389-photo selection stage is `runs/sentosa-quality-v2/album`.
- Keep Qwen thinking/one-attempt/no-fallback and existing TOPIQ ranking unchanged.
- Do not call Sentosa tuning an independent benchmark or claim Lightroom parity.

The foundation notes below describe the prior branch and historical results.

## Repository state

- Foundation branch: `feat/ai-develop-edits`
- Foundation draft PR: <https://github.com/xcv58/Cull.sh/pull/2>
- Follow-up branch: `codex/topiq-qwen-rapidraw`
- Follow-up base: `feat/ai-develop-edits`
- Test suite: 128 passed

The follow-up branch contains the original three focused commits plus the
TOPIQ/Qwen migration, rendered-feedback pilot, and expanded RapidRAW recipe work:

1. `364e3ed` — model benchmarks, shadow evaluation, TOPIQ ranking, and
   AppleDouble scanning exclusion.
2. `51e8dd9` — approval-gated RapidRAW stage, visual preview, final export,
   provenance, resume behavior, CLI, tests, and architecture documentation.
3. `cceb6c8` — bounded Ollama structured generation via `num_predict`, exposed
   configuration, run provenance, and regression tests.

## Production decisions

- Do not migrate to the complete Facet pipeline.
- Blend standalone TOPIQ-NR into candidate ranking at 25%; keep the existing
  local percentile at 75%.
- TOPIQ does not participate in hard rejects.
- Use `orcarouter/Qwen3.8-27B-Uncensored` for production semantic culling and
  edit suggestions with one attempt, no fallback, and bounded generation.
  Gemma results remain historical benchmark evidence only.
- Use one image per edit-suggestion request and one rendered pair per feedback
  request to prevent cross-image association leakage.
- Require every executable edit field in Qwen's transport schema, and give the
  validator actual decoded dimensions plus measured outer-edge facts so display
  padding is not confused with pixels. Keep human review authoritative.
- The executable RapidRAW recipe includes global tone, relative white balance,
  presence, detail/noise, vignette, crop, and rotation paired with bounds that
  remove rotated black edges. Preserve
  unsupported but useful HSL/curve/mask/healing/lens ideas as explicit
  `additional_edits`; never report them as rendered.
- Use RapidRAW as the primary automated renderer/exporter. Lightroom-compatible
  XMP remains optional interoperability.

## RapidRAW workflow

The installed local build is detected at
`~/Applications/RapidRAW.app/Contents/MacOS/RapidRAW`, version 1.6.1.

```bash
python main.py suggest-edits --path "/path/to/culled-raws"
python main.py rapidraw-stage \
  --suggestions runs/<timestamp>/edit-suggestions.jsonl \
  --output /path/to/new-stage
python main.py rapidraw-preview --stage /path/to/new-stage
python main.py rapidraw-export \
  --stage /path/to/new-stage \
  --approvals ~/Downloads/rapidraw-approvals.json
```

The production smoke test completed stage, real RapidRAW preview, and final
export for one Sony ARW in `/private/tmp/cull-rapidraw-smoke.p9aFPc`. The JPEG
was 9504 x 6336 and retained camera make, model, and capture timestamp. Original
RAW/XMP files were untouched.

## Prospective shadow baseline

Folder: `/Volumes/Sandisk 4T/RAW Photos/2026-05-06 Kuala Lumpur`

Frozen successful run: `runs/20260821-145704-238706`

- 60 RAW files, 6 scenes
- dry run, RAW-only, no MUSIQ/NIMA
- TOPIQ ranking enabled at 25%
- batch size 1
- 47 local rejects
- 12 Gemma reviews
- 0 picks
- 1 bounded vision failure: `DSC07689.ARW`
- 10 TOPIQ review thumbnails across 2 scene disagreements
- 0 metadata records written
- 0 Lightroom edit records written

No RAW, XMP, or ACR file under the source folder had a modification time newer
than the run's `config.json` after completion. The exact artifact hashes are
recorded in `docs/prospective-validation.md`.

An earlier superseded run, `runs/20260821-144041-340315`, was interrupted after
one semantic item because a malformed Gemma response continued streaming for
13 minutes. It exposed the missing generation ceiling and led to commit
`cceb6c8`. Keep it only as an operational diagnostic; do not use it as the
prospective baseline.

## Next human gate

Review the Kuala Lumpur folder normally and finalize its XMP picks/rejects.
Do not replace or modify the frozen successful run. After human review:

```bash
python main.py benchmark \
  --path "/Volumes/Sandisk 4T/RAW Photos/2026-05-06 Kuala Lumpur" \
  --shadow-run runs/20260821-145704-238706
```

That comparison is the remaining prospective validation. The current result is
not yet a quality score because final human labels do not exist.

## Known observations

- The 47/60 deterministic local-reject count is high and should be examined
  against the future human review before changing thresholds.
- `DSC07689.ARW` exhausted three bounded malformed-response attempts. It remains
  explicit in the frozen manifest rather than silently falling back.
- RapidRAW final export requires a manifest-matched approval file by default.
  `--approve-all` is an explicit pilot bypass only.
