# Handoff — AI develop edits + gemma4:12b default

**Branch:** `feat/ai-develop-edits` (off `main` / `origin/main`)
**Feature commit:** `22af454` — *Add AI develop-edit suggestions and default to gemma4:12b*
**Status:** implemented, unit-tested (60/60 pass), and validated end-to-end on a real 66-RAW folder. Not pushed; no PR yet.

---

## What was done

### 1. Git state reconciled
The repo had a detached HEAD and a local `main` with an unrelated history to `origin/main`.
- Archived the original 29-commit development history as tag **`archive/dev-history`** (browsable via `git log archive/dev-history`).
- Reset local `main` to `origin/main` (the canonical, feature-complete code) and re-attached HEAD; `main` tracks `origin/main`.

### 2. Default vision model → `gemma4:12b`
Changed the default in `cull_sh/config.py` (`BackendConfig.model`) and the `cull` / `suggest-edits` CLI `--model` option. Confirmed `gemma4:12b` is vision-capable (11.9B, larger than the previous `gemma4:latest` 8.0B default). Override per-run with `--model`.

### 3. New feature: `suggest-edits` command
AI-suggested **global develop edits** written as standard, reversible Camera Raw (`crs:`) settings into the XMP sidecar next to each **kept** RAW. Rejected RAWs are skipped; existing cull state (rating/label/pick) is preserved (merge, not overwrite).

Sliders suggested (intentionally minimal prototype): **exposure, contrast, highlights, shadows, vibrance** (+ free-text `summary`). **No crop / rotation / local edits** — see "Findings".

Usage:
```bash
python main.py suggest-edits --path "/path/to/culled-raws"             # dry run: show suggestions only
python main.py suggest-edits --path "/path/to/culled-raws" --no-dry-run
python main.py suggest-edits --path "/path/to/culled-raws" --prefer "warm, punchy look"
```
Suggestions are logged to `runs/<ts>/edit-suggestions.jsonl`.

---

## Files changed (commit `22af454`)

| File | Change |
|---|---|
| `cull_sh/models.py` | new `EditSuggestion` dataclass |
| `cull_sh/backends/base.py` | `suggest_edits` added to the `VisionBackend` contract |
| `cull_sh/backends/ollama.py` | `suggest_edits` + `OllamaEditPayload`/`OllamaBatchEditPayload` (clamped Pydantic schema) + `_parse_edit_payload`; **refactored the retry loop into shared `_chat_structured`** (no behavior change to scoring) |
| `cull_sh/xmp.py` | `write_develop_sidecar` / `apply_develop_settings` (writes `crs:Exposure2012`, `Contrast2012`, `Highlights2012`, `Shadows2012`, `Vibrance`, `HasSettings`) |
| `cull_sh/cli.py` | `suggest-edits` command + helpers; default model `gemma4:12b` |
| `cull_sh/config.py` | default model `gemma4:12b` |
| `tests/test_xmp.py` | develop-sidecar write + cull-state-preservation tests |
| `tests/test_ollama_backend.py` | `suggest_edits` ordering + edit-payload parsing tests |
| `README.md` | "AI Develop Edits" section + status/workflow updates |

---

## Verification

- **Unit:** `python -m pytest tests/ -q` → **60 passed**.
- **XMP output:** exiftool reads the standard Camera Raw tags back (`Exposure 2012`, `Contrast 2012`, …) → Lightroom-compatible.
- **Merge safety:** writing develop settings preserves existing cull `Rating`/`Pick`/`Label`.
- **End-to-end** on a copy of `2026-05-20 tainan` (66 RAW, trimmed to RAW-only):
  - `cull --no-dry-run`: **10 pick, 38 review, 11 reject, 7 failed**, 59 sidecars written.
  - `suggest-edits --no-dry-run`: **51 sidecars edited, 4 failed**.
  - **FastRawViewer** visually confirms cull ratings/labels render from the sidecars (picks = ★★★★★, rejects = red "Reject", review = none).

---

## Findings / known issues

1. **gemma4:12b JSON reliability ≈ 89%.** ~11% of cohorts failed with malformed JSON ("extra data" after the object) or hallucinated filenames (7 cull + 4 edit cohort failures). The pipeline handles these gracefully (marks failed, continues, writes the rest) — not a code bug, a model-quality limit.
2. **Edit suggestions are very uniform** (≈ +5 contrast, −10/−15 highlights, +10 shadows, +5 vibrance across nearly all photos; empty `summary`). Amplified by `temperature: 0`, a generic prompt, visually similar skyline frames, and schema defaults. The model is not differentiating much per-image.
3. **No crop suggestions — by design.** The request sends a strict JSON `format` schema with only the 5 global fields, so structured output makes crop impossible regardless of the model. (Free-form prompts to the same model *do* return crops.) Crop is feasible to add via `crs:HasCrop`/`crs:CropTop/Left/Bottom/Right`/`crs:CropAngle` if wanted.

## Environment notes (for the visual-render step)

- **Lightroom Classic is NOT installed** on this machine (only an uninstaller stub). Only the **cloud "Adobe Lightroom" (CC)** is present (`com.adobe.lightroomCC`, v9.4).
- Whether the CC app reads our local `.xmp` sidecars (ratings/labels/develop) is **unresolved** — the maintainer reports it loaded color/reject/pick from local files previously; this was not confirmed in-session. The original source folder did contain Adobe-written `.acr`/`.xmp`.
- **FastRawViewer** browses sidecars (shows cull labels) but its **trial is expired**, so it won't render the main RAW preview (can't show the applied edits).
- No CLI raw renderers (darktable/rawtherapee/dcraw) installed.

---

## Suggested next steps

- **Decide:** push branch / open PR.
- **Improve edit quality:** require a per-image `summary`, nudge `temperature` up slightly, and/or strengthen the prompt to force per-image reasoning.
- **Optional:** add opt-in **crop** suggestions (`--with-crop`) — extend schema with `has_crop` + normalized edges + `crop_angle`, clamp/validate the rectangle, write `crs:Crop*` only when confident.
- **Robustness:** more tolerant JSON parsing (extract first object) and/or per-image retry fallback to cut the ~11% cohort failure rate.
- **Resolve the render path:** confirm whether the CC app reads sidecars, or install Lightroom Classic / activate FastRawViewer for an authoritative visual of the develop edits.

## Test artifacts (gitignored, local only)

- Test folder: `/Users/xcv58/Pictures/Photos/2026-05-20 tainan-test` (66 RAW copy, originals untouched).
- Run dirs under `runs/` — cull: `20260627-231730-545857`, suggest-edits: `20260628-003835-539668`.
