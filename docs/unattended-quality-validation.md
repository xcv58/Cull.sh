# Unattended quality validation

This is a development iteration following the Sentosa human/machine comparison,
not a new independent benchmark. Keep the first frozen cull, its 28 delivered
JPEGs, original RAW/XMP/ACR files, and human exports unchanged. Neither human pick
labels nor human JPEGs enter the new model requests or baseline calibration.

## What changed

- **Renderer semantics:** Qwen receives RapidRAW-specific exposure/brightness,
  relative temperature/tint, and vignette instructions. Positive vignette mixes
  toward white; negative darkens. No sign reversal is applied in the adapter.
- **Bounded validation:** baseline → initial recipe → render → accept/revert/one
  refinement → final render → final delivery check. The last check cannot trigger
  another refinement. Non-accept blocks delivery and remains failed on resume.
  Reviews include decoded dimensions, tone/edge measurements, and native-pixel
  corner/center patches. Measurements are evidence, not hard aesthetic thresholds.
- **Baseline experiment:** `camera-midtones-v1` brackets brightness at 0 and
  ±0.5/1.0/1.5 toward the embedded camera JPEG's tonal percentiles, penalizing new
  near-white pixels. It does not use human edits, alter exposure, force night
  scenes bright, or reproduce Lightroom Adaptive Color. Neutral remains the CLI
  default; the development pilot explicitly opts into calibration.
- **Absolute recipes:** supplied starting values are retained for unchanged
  controls. Zero is not a no-op for a nonzero starting value. Rejecting a proposed
  edit restores the actual starting recipe, including calibration.
- **Straightening:** long-line slope/consensus is supplied as advisory evidence;
  perspective lines are not automatically leveled. Rotation retains safe crop
  bounds. Installed-renderer tests check direction and absence of black fill.
- **Unattended selection:** a separate final-album policy considers every frozen
  frame, including previous reviews/local rejects, without a fixed pick quota.
  It does not inherit assisted triage's “pick sparingly, otherwise review” policy.
  Scores are reused only after matching the RAW checksum ledger. Conservative
  adjacent lookalikes are suppressed after selection, and must reference a
  surviving final pick. The remaining selected photographs still need editing
  and successful delivery checks; selection does not itself prove delivery.
- **Local quality gate:** both primary sharpness measures and the configured
  number of corroborating weak signals are required. Missing optional metrics
  cannot reduce that requirement. TOPIQ remains ranking-only at the existing
  weight; it does not hard-reject photos.
- **Delivery integrity:** final-pixel/recipe hashes, renderer binary and relevant
  preferences are checked. Delivery renders must match inspected geometry and
  pixels within JPEG tolerance, then receive an sRGB ICC tag without
  recompression. Failed candidates remain in the isolated stage, not delivery.

Schema 4 uses `delivery-pixels-v2`. Older feedback experiments are frozen and
must not be silently upgraded to this validation policy. Resume using identical
inputs/settings; changed recipes, pixels, policies or settings require a new
stage or an explicit investigation. Completed model work is checkpointed.

## Commands

Installed-renderer control probe (uses isolated synthetic images):

```bash
.venv/bin/python -m cull_sh.render_probe \
  --source /path/to/isolated-raws --output runs/new-control-probe \
  --binary /Users/yihong/Applications/RapidRAW.app/Contents/MacOS/RapidRAW \
  --controls-only
```

Add `--stems DSC00001 DSC00002 DSC00003` without `--controls-only` to compare
basic, AgX, perceptual brightness, and camera previews for RAWs. A new output
directory is required so previous probes remain intact.

Prepare the final selection pass without calling Qwen:

```bash
.venv/bin/python -m cull_sh.album_selection \
  --frozen-run /path/to/frozen-cull --source /path/to/isolated-raws \
  --output runs/new-album --raw-checksums /path/to/raw-sha256.txt \
  --prepare-only
```

Remove `--prepare-only` to run/resume contextual selection. The checksum ledger
uses SHA-256 followed by a filename/path (standard `shasum -a 256` format).
The completed `manifest.jsonl` is published only after every frame has a final
outcome; consumers reject incomplete or altered album manifests.

Run an explicit editing pilot, then optionally the full selection pass:

```bash
.venv/bin/python -m cull_sh.quality_pilot \
  --source /path/to/isolated-raws --cull-run /path/to/frozen-cull \
  --output runs/new-edit-pilot \
  --binary /Users/yihong/Applications/RapidRAW.app/Contents/MacOS/RapidRAW \
  --stems DSC00001 DSC00002 DSC00003 \
  --album-output runs/new-album --raw-checksums /path/to/raw-sha256.txt
```

This command writes `stage/feedback-manifest.json`, a review page, and—only after
all final checks pass—`delivery/*.jpg`. Optional album selection runs sequentially
after successful pilot delivery, avoiding concurrent Qwen requests. It does not
automatically edit/export the newly selected full album. Qwen keeps thinking
enabled, one attempt, 4096 output tokens and a 600-second per-request timeout;
there is no model fallback. A failure stops the job and leaves resumable records.

### Explicit runtime recovery

`--context-tokens 65536` caps Ollama's allocation for this job only, without
changing its global settings, model or output budget. The general backend default
remains unspecified. Overviews sent to the model are bounded to 4 megapixels,
2560 pixels per edge and 4 MiB encoded JPEG; small images and native-pixel detail
sheets pass through unchanged. Full-resolution stage and delivery files are not
resized. The model is told that overview dimensions differ from source dimensions.

When resuming saved model work with a changed context or transport policy, supply
`--runtime-change-reason 'specific diagnosed reason'` (the older alias
`--context-change-reason` is also accepted). The manifest records the transition
and retained work counts; new review/check records identify the runtime settings.
This is an explicit operational recovery, not an automatic retry or model fallback.
Model, prompt, thinking, renderer and recipe changes remain separately guarded.
Album context/transport settings may change after preparation only while it has
no model decisions; once decisions exist they are fixed for that album stage.

The August 28 recovery reproduced HTTP 413 on DSC03966: two 60 MP renders totaled
approximately 120 MiB before base64. Bounded overviews plus the native patches
reduced the image payload from about 160 MiB to 3.64 MiB. A fresh worker at 65536
context also reduced the logged context allocation from 16533 to 4245 MiB.
The original worker's connection reset is confirmed, but its precise exit cause
is not established by the available macOS diagnostics. See the ignored run-local
`pilot-v2/recovery-20260828` directory for preserved failure records and evidence.

## Sentosa development run, August 28

### Explicit correction after a failed final quality check

The completed `pilot-v2` has twelve final renders/checks: ten passed, while
DSC03797 and DSC03806 failed for excessive daylight darkening. Calibrated
brightness was reduced from 0.5 to 0.3 and from 1.0 to 0.5 respectively. In the
latter case, the first recipe reset brightness to zero, and the ordinary
half-stop refinement limit could not restore 1.0 in one step. This is development
evidence of imperfect recipe reasoning and bounded repair, not a renderer crash.

`cull_sh.feedback_correction` supports a separately authorized development
experiment. It checks parent/source hashes and renderer provenance, copies the
stage into a separate tree, and preserves all passed recipes, pixels and final
checks. Only failed records get a new ordinary correction request. The old
failed record and parent manifest hash remain in the new manifest. Final checks
use the original prompt, without the correction guidance or prior diagnosis.
There is no automatic recursive correction; a failed new check remains failed
on resume, and blocks the entire pilot export as before.

```bash
.venv/bin/python -m cull_sh.feedback_correction \
  --parent-stage runs/sentosa-quality-v2/pilot-v2/stage \
  --output runs/sentosa-quality-v2/pilot-v2-correction-1 \
  --binary /Users/yihong/Applications/RapidRAW.app/Contents/MacOS/RapidRAW \
  --context-tokens 65536 --reason 'Explicitly authorized diagnosis and correction'
```

The reason is part of the resume identity; use the exact original command when
resuming an interrupted correction. Do not launch this example over an existing
experiment with a different reason. This command never starts album selection.
It does not change the default production prompt or remove refinement limits.
Improvement on these selected failures cannot be reported as an unbiased
benchmark or as a fully unattended success.

- Work branch: `codex/sentosa-unattended-quality`.
- Inputs: 389 hash-verified isolated RAWs from the original blind experiment.
- Full selection prepared as 90 chronological cohorts in
  `runs/sentosa-quality-v2/album`.
- Controls: real installed RapidRAW 1.6.1 passed vignette, temperature/tint,
  rotation direction, and safe-edge checks. See `control-probe/probe-report.json`.
- Three-photo starting-render probe: AgX alone did not resolve the dark baseline;
  camera-referenced midtone calibration is being tested, not declared superior.
- First pilot (`pilot/`) was stopped after two initial responses exposed a
  conflicting zero/no-change instruction. Its records are retained as diagnostic
  evidence, not reused as validated recipes. It produced no delivery JPEGs.
- Corrected pilot: `runs/sentosa-quality-v2/pilot-v2`; inspect `status.json` and
  `stage/feedback-manifest.json` for current progress, not this static document.

Evaluate daylight readability, casts, unwanted white corners, highlight texture,
night mood, and geometry on the fixed pilot. Then assess full-album coverage and
repetition, comparing to human choices only after its decisions are frozen.
Improvement on Sentosa is development evidence. A future untouched folder is
still required for an unbiased prospective assessment. Lens-profile parity,
Adaptive Color equivalents, and dependable local masks are not established by
these changes; unsupported suggestions remain explicitly unrendered notes.
