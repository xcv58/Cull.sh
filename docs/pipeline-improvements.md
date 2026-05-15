# Pipeline Improvements

This document captures the next meaningful upgrades for Cull.sh beyond the current
Laplacian-plus-VLM baseline.

## Implemented Foundations

The current codebase now includes:

- provisional scene grouping before local filtering
- multi-metric local quality analysis:
  - Laplacian variance
  - Tenengrad
  - brightness mean
  - contrast standard deviation
- portrait-aware local hints through face and eye detection
- scene-relative local ranking scores
- near-duplicate suppression before the vision call
- manifest persistence for scene IDs and local metric detail

These changes make later ranking and genre-specific logic easier to add without
rewriting the core pipeline again.

## Near-Term Upgrades

### 1. Better Scene-Relative Ranking

Current issue:

- the pipeline now ranks and suppresses within scenes, but still uses simple
  heuristics rather than a learned comparative model

Recommended next step:

- improve the local rank score beyond blur / contrast heuristics
- compare scene members directly when enough candidates survive
- optionally let the vision backend receive a small scene set instead of only
  independent single-image calls

### 2. Better No-Reference Quality Metrics

Current issue:

- Laplacian and Tenengrad are fast but still limited

Recommended candidates:

- CPBD for perceptual blur estimation
- BRISQUE for broader no-reference image quality
- PyIQA-backed metrics such as MUSIQ or NIMA for learned quality scoring

Suggested rollout:

1. add CPBD or BRISQUE as an optional local metric
2. record it in manifests only at first
3. compare against actual keep/reject outcomes before making it part of the gate

### 3. Portrait-Specific Local Ranking

Current issue:

- portrait prompts ask for eye sharpness and expression quality, but local analysis
  only provides weak hints so far

Recommended next step:

- use face count and eye count to influence local ranking
- treat missing eyes as a warning signal, not an automatic reject
- add face-size heuristics so tiny or off-angle detections do not dominate

### 4. Better Duplicate Suppression

Current issue:

- the current duplicate suppression relies on a simple perceptual hash

Recommended next step:

- compare dHash with alternative preview embeddings
- calibrate thresholds on real burst sequences
- decide when duplicates should be auto-rejected versus only deprioritized

## Product Patterns Worth Borrowing

The current market pattern is consistent:

- do local technical filtering first
- identify duplicates and bursts early
- use face-specific logic for portrait and event work
- prefer relative ranking inside a scene over absolute image scores
- keep the Lightroom workflow non-destructive and low-friction

Cull.sh should continue moving in that direction.
