# Cull.sh Handoff

## Current workflow decision — September 7

- Keep Cull.sh for culling, ranking, duplicate handling, and Lightroom-compatible
  metadata. Continue to use Qwen for semantic culling and TOPIQ only at its
  existing ranking weight.
- Use the [canonical new-folder workflow](docs/new-folder-workflow.md): Cull.sh
  writes merge-safe culling metadata and lens corrections, Lightroom applies
  Adaptive Color to all RAWs in one batch, and Lightroom batch-exports only the
  selected photographs. This is a native batch operation, not per-photo edit
  automation. Do not invest further in per-photo Lightroom UI automation unless
  the user explicitly reopens it.
- Pause RapidRAW editing/export development. The implementation and completed
  experiments remain as development evidence, but RapidRAW is not the current
  production finishing path and its results did not establish Lightroom parity.
- The ignored twelve-photo Lightroom pilot is intentionally unfinished: it has
  isolated RAW copies and lens-correction sidecars, but no verified Adaptive
  Color payloads and no JPEG exports. Nothing in that pilot needs to be resumed
  or committed.
- The full reviewed checkpoint is backed up on the forward-only branch
  `origin/codex/sentosa-unattended-quality`. The staged integration procedure is
  recorded in [the main integration plan](docs/main-integration-plan.md).
- Remote `main` contains the AI-suggestion foundation (PR #2), the required CI
  gate (PR #4), TOPIQ/benchmark/initial RapidRAW tooling (PR #5), and the Qwen
  culling/rendered-validation layer (PR #3), and the final Sentosa
  quality/correction layer (PR #6). Every staged integration layer is merged,
  and each gated layer plus the final post-merge `main` commit passed exact-head
  CI.
- `main` requires an up-to-date `pytest` check through a pull request, enforces
  the rule for admins, requires resolved conversations, and disallows force
  pushes and branch deletion. Repository-local `push.default=simple` overrides
  the user's global `matching` setting; still use explicit refspecs for every
  integration push.

## Current development iteration — August 28

- Branch: `codex/sentosa-unattended-quality`, updated through protected `main`
  without rebasing or rewriting its recovery history.
- Regression suite: 171 passed plus four subtests; installed-app control
  tests also passed. The twelve-photo pilot finished: ten passed and two failed
  final quality checks for darkening (DSC03797 and DSC03806). No delivery export
  was authorized by that original pilot.
- Renderer semantics, final-pixel validation, experimental calibrated baselines,
  straightening evidence, cautious local rejects, and post-selection duplicate
  handling are implemented. See [quality validation](docs/unattended-quality-validation.md).
- Frozen Sentosa source/human/first-machine artifacts remain untouched.
- The parent `runs/sentosa-quality-v2/pilot-v2` is preserved. User-authorized
  two-photo correction is complete in `pilot-v2-correction-1/status.json`, using
  the explicit `cull_sh.feedback_correction` workflow. Ten passed records are
  copied unchanged; only two failed photos received a correction and new final
  check. Both passed. Twelve verified sRGB delivery JPEGs (292.1 MiB total) are
  in `runs/sentosa-quality-v2/pilot-v2-correction-1/delivery`. The job exited at
  17:32 EDT. The old shutdown heartbeat remains paused.
  The full 389-photo selection stage remains prepared with zero decisions in
  `runs/sentosa-quality-v2/album`; no automatic album continuation is enabled.
- Earlier runtime recovery preserved six completed reviews after a worker reset and a
  reproduced oversized-request error. Resume this job with `--context-tokens
  65536`; bounded overviews leave full-resolution renders and native detail
  patches intact. Prior failures and runtime transitions are recorded in the run.
- Keep Qwen thinking/one-attempt/no-fallback and existing TOPIQ ranking unchanged.
- Do not call Sentosa tuning an independent benchmark or claim Lightroom parity.

The foundation notes below describe the prior branch and historical results.

## Historical repository state

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
- RapidRAW editing/export is paused and is not the current production finishing
  path. Lightroom-compatible XMP remains the supported handoff from Cull.sh;
  finishing and JPEG export are manual in Lightroom.

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
