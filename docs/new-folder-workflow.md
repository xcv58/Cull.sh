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

This is a batch workflow. The Local view steps below were checked in Adobe
Lightroom desktop 10.0 with Sony ARW, DJI DNG, and standalone JPEG files.

## 1. Preflight and protect existing work

Before running a command that can write metadata:

- Confirm the exact source folder and whether it contains RAWs, DNGs, JPEGs,
  existing XMP sidecars, or Lightroom `.acr` payloads.
- Cull.sh skips hidden files, AppleDouble companions, and subdirectories
  named `output`. In Lightroom, work in individual source folders if an
  included delivery folder already contains photos.
- Back up existing XMP sidecars and embedded-DNG metadata when the folder has
  already been touched by a person or another application.
- Run the dependency check:

```bash
.venv/bin/python main.py doctor
```

Do not write culling decisions before the ExifTool, Ollama, Qwen, and TOPIQ
checks pass. A listed Ollama model can have missing weights. If Doctor cannot
read the configured model, restore it before culling. If applying the
Lightroom baseline first, back up the DNG and JPEG originals; after the later
culling metadata write, verify that the Lightroom edits still read correctly.

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

If Lightroom already saved edits inside the DNGs, use
`repair-sidecars --embed-dng-metadata` with the same verified manifest and
keep a backup of the edited DNGs before replay. Lightroom 10 reads an external
XMP as the active settings for a DNG when one is present; a new cull-only XMP
can hide its embedded profile. The embedded replay only writes culling tags
and refuses DNGs that already have an external XMP. Reopen Lightroom and
verify both the flag and the profile before exporting. Even a flag-only
embedded write can leave Lightroom's Adaptive Color render stale.

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

For a DNG that already has embedded Lightroom edits, use Lightroom's Optics
panel to correct any missing lens setting. Do not create a new lens-only XMP
next to that DNG.

## 5. Apply Adaptive Color once as a Lightroom batch

Generate the count-aware handoff:

```bash
.venv/bin/python main.py lightroom-adaptive-color \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all
```

Then use Adobe Lightroom's Local view. In Lightroom 10, Photos and Filtering
are separate left-side panels. Shortcuts below assume the default Lightroom
key map; use the named controls if the workspace uses a different map.

1. Open the exact folder, clear filters, and include subfolders only when they
   are part of the intended scope.
2. Open **Filtering > Extension**. Choose the in-scope RAW extensions, such
   as `arw` and `dng`, leaving the Flag filter clear. Check that the grid count
   matches the handoff. Leave JPEGs for the separate pass below.
3. Open one RAW in Detail. In **Edit > Profile**, choose **Adaptive Color**.
4. Open **Choose Edit Settings to Copy** (`Shift+Cmd+C` on Mac). In **Copy
   Settings**, choose **Select > None**, check only **Profile: Adaptive Color**,
   visually verify the selection, and click **Copy**.
5. Return to Square Grid (`G`), select all visible RAWs, and verify that the
   selected count matches the visible count. Paste once (`Cmd+V` on Mac).
   Lightroom may show a confirmation with the affected count for a large batch.
6. Let the paste and AI update finish. Inspect sample photos in Detail. On the
   tested build the same batch wrote active Adaptive Color to ARW XMP plus ACR
   sidecars and to embedded DNG metadata.
7. If a DNG reports stale or missing AI settings, isolate the DNGs with the
   Extension filter, check the count, and use **Photo > Update AI Settings**.
   Recheck the embedded settings after this and after any metadata repair.

Validate Lightroom's written state after the batch:

```bash
.venv/bin/python main.py lightroom-adaptive-color \
  --path "/path/to/new-folder" \
  --lightroom-edit-scope all \
  --no-handoff \
  --assert-complete
```

Stop if the visible count, Adaptive Color, or lens count does not match. An
ACR file on its own can represent other AI edits; Cull.sh also checks for the
active Adaptive Color reference. Fix scope or Lightroom writeback before
exporting.

Lightroom 10 can paste Adaptive Color using a previously loaded edit state,
even if the lens sidecars were written on disk before the paste. If lens
corrections are missing after the batch, generate a fresh handoff for the
affected paths. In **Edit > Optics**, enable **Lens Corrections** on a
matching camera/lens seed, then use **Copy Settings > Select > None** and
check only **Optics > Enable Lens Corrections**. Paste to the affected photos
with that lens, separately for each camera/lens combination. Recheck both
the lens and Adaptive Color counts before exporting.

## 6. Export the final picks

Clear Extension filters and turn on **Filtering > Flag > Show picked photos**.
Compare the grid count with the frozen picks (plus any preserved picks). In
Square Grid, select the result and use **File > Export** (or `Shift+E` with the
default shortcuts). Choose JPG, Full Size, sRGB, and the requested quality
(90% is a starting point). Export to a separate delivery folder and check the
output names and count against the picks. If Lightroom opens **AI Update
Required**, leave **Update affected photos** checked and let the refresh
finish before choosing the export location.

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

Use **Filtering > Extension** to show just `jpg`/`jpeg`, verify the selected
count in Square Grid, and choose **Photo > Apply Auto Settings** (or
`Shift+A` with the default shortcuts). In Local view Lightroom writes the
edits as metadata in the JPEG, so keep a backup of the original file.

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
