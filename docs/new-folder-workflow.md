# New Folder Production Workflow

This is the default production runbook for a new photo folder. It keeps
culling and metadata changes auditable, gives every RAW the same Lightroom
baseline, and exports only the final picks.

The production contract is:

1. Cull.sh evaluates the folder in dry-run mode and freezes its decisions.
2. After the run is verified, Cull.sh replays those frozen decisions into XMP.
3. Cull.sh enables Lightroom lens corrections for every RAW.
4. Lightroom applies Adaptive Color to every RAW in one batch.
5. Lightroom exports only the picked photographs as JPEGs in one batch.

This is a batch workflow. It does not use per-photo AI develop suggestions or
per-photo Lightroom UI automation.

## 1. Preflight and protect existing work

Before running a command that can write metadata:

- Confirm the exact source folder and whether it contains RAWs, DNGs, JPEGs,
  existing XMP sidecars, or Lightroom `.acr` payloads.
- Ignore output directories, hidden files, and AppleDouble `._*` companions.
- Back up existing XMP sidecars and embedded-DNG metadata when the folder has
  already been touched by a person or another application.
- Run the dependency check:

```bash
.venv/bin/python main.py doctor
```

Do not start a metadata-writing phase if ExifTool, Ollama, Qwen, or required
TOPIQ support is unavailable.

## 2. Run and verify culling without writing metadata

Choose the genre and optional preference after inspecting the folder. A broad
starting command is:

```bash
.venv/bin/python main.py cull \
  --path "/path/to/new-folder" \
  --genre auto \
  --no-interactive
```

Dry run is the default. The command may write reproducibility and review
artifacts under `runs/`, but it does not write culling metadata beside the
photos. Production culling uses Qwen with thinking enabled, one attempt, no
fallback, and fail-fast behavior. TOPIQ contributes 25% of candidate ranking
and does not participate in the hard reject gate.

Before applying the decisions, verify:

- discovered assets and manifest coverage match the intended folder;
- every expected cohort completed and no model request failed;
- pick, review, and reject counts are plausible;
- TOPIQ coverage is complete and the disagreement artifact is available;
- source image bytes and existing sidecars have not changed.

If the run is incomplete or suspicious, stop and diagnose it. Do not write a
partially successful run into the photo folder.

## 3. Apply the frozen culling decisions

Replay the verified manifest instead of running model inference again:

```bash
.venv/bin/python main.py repair-sidecars \
  --run-dir runs/<timestamp> \
  --preserve-existing-picks \
  --no-dry-run
```

`--preserve-existing-picks` leaves a photo already marked as a pick completely
unchanged. This makes a new run additive when a folder contains earlier human
selections. Inspect the command summary and confirm that its expected and
written counts agree.

## 4. Apply the RAW baseline to the whole folder

Ensure every RAW, including rejects, receives merge-safe lens-profile settings:

```bash
.venv/bin/python main.py lightroom-edit \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all \
  --no-dry-run
```

This stage enables Lightroom lens corrections only. It preserves unrelated
existing XMP fields and does not invent Lightroom's Adaptive Color payload.
Verify the discovered count, edit-candidate count, written-sidecar count,
lens-enabled count, and error count before opening Lightroom.

## 5. Apply Adaptive Color once as a Lightroom batch

Generate the count-aware handoff:

```bash
.venv/bin/python main.py lightroom-adaptive-color \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all
```

Then use Adobe Lightroom's Local view:

1. Open the exact folder, clear filters, and include subfolders only when they
   are part of the intended scope.
2. Show all in-scope RAWs, including rejected photos, and verify that the
   visible count matches the generated handoff.
3. On one seed RAW, choose **Adaptive Color** from the Profile control.
4. Copy Edit Settings, clear all settings, and select only the profile/treatment
   setting. Do not copy exposure, white balance, curves, masks, crop, geometry,
   detail, or lens corrections.
5. Return to Grid, select all visible RAWs, and paste once.
6. Wait for Lightroom's AI update to finish. Inspect at least the first,
   middle, and last photograph to confirm that the batch reached the full set.
7. If the folder contains DNGs, filter to the DNG subset and run
   **Photo > Update AI Settings** after the paste.

Validate Lightroom's written state after the batch:

```bash
.venv/bin/python main.py lightroom-adaptive-color \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all \
  --no-handoff \
  --assert-complete
```

Stop if the visible count or Adaptive Color verification does not match. Fix
the batch scope or Lightroom writeback before exporting.

## 6. Export the final picks

In Lightroom, filter to the Cull.sh picks/kept photographs and export that set
to JPEG in one batch. Do not export the rejected photos merely because they
received the same baseline edit.

The intended result is simple: all RAWs retain a consistent lens-correction and
Adaptive Color baseline, while only the selected photographs become delivery
JPEGs.

## JPEG-only exception

Adaptive Color/Profile is a RAW/DNG feature. When standalone JPEGs are in
scope, prepare a separate JPEG batch with:

```bash
.venv/bin/python main.py lightroom-jpeg-auto \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all
```

That Lightroom handoff uses **Photo > Apply Auto Settings** for JPEGs. It is
not part of the normal RAW Adaptive Color batch and should not be mixed into
the RAW baseline procedure.

## Failure and recovery rules

- If Qwen fails, stop and repair the local-model problem. Do not fall back to a
  second model or silently continue.
- If manifest coverage or counts do not match, do not write metadata.
- If Lightroom's visible count differs from the handoff, do not paste edits.
- If Adaptive Color verification fails, do not export until the missing batch
  is understood.
- Once a dry run is frozen and verified, use `repair-sidecars`; do not spend
  hours rerunning the entire culling pipeline merely to apply its decisions.
- Do not use per-photo Qwen editing, RapidRAW production rendering, or a fixed
  exposure/white-balance/crop/mask recipe as part of this default workflow.
