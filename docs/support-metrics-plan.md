# Support Metrics Rollout Plan

This plan adds more local image-quality signals without turning Cull.sh into an
opaque scoring system. The goal is:

- reject only clearly bad photos automatically
- preserve useful borderline frames for `review`
- keep enough trace data to debug every local decision later

## Phase 1: Instrumentation First

Add new support metrics and record them for every file:

- `BRISQUE` for no-reference technical quality
- `CPBD` for perceptual blur

Requirements:

- expose tunable thresholds and enable/disable flags on the CLI
- store raw scores in the manifest
- store threshold checks and vote outcomes in the manifest
- preserve whether each metric was only observed or actively used for rejection

Expected output per file:

- raw local scores
- threshold values
- weak/ok vote per metric
- local trace explaining why the file was rejected or kept for review

## Phase 2: Conservative Gate Integration

Integrate the new metrics conservatively:

- keep current Laplacian + Tenengrad as primary local blur checks
- keep `MUSIQ` and `NIMA` as learned support signals
- add `BRISQUE` and `CPBD` as optional support votes
- default to recording the new metrics before relying on them heavily

Reject policy:

- primary blur gate must still indicate weakness
- support metrics contribute configurable weak votes
- duplicate suppression and scene safeguards still take precedence where needed

This keeps the reject logic debuggable and prevents a single new metric from
silently dominating the outcome.

## Phase 3: Decision Trace Coverage

Every file should answer these questions from the manifest alone:

- what local scores were computed?
- which thresholds were used?
- which metrics voted weak?
- did duplicate suppression fire?
- did scene-level reranking fire?
- did the scene safeguard restore the file?
- was the file sent to vision?
- what was the final decision and why?

This trace is required for future threshold tuning.

## Phase 4: Bounded Validation

Run a bounded dry run on an existing sample RAW folder to validate:

- score computation works in worker processes
- manifests contain the new metrics and traces
- no unexpected crashes from third-party metric libraries
- support-metric import failures are surfaced cleanly

If a bug appears, fix it immediately before doing the full-folder pass.

## Phase 5: Full-Folder Production Pass

Once bounded validation is clean:

- run the full sample folder with sidecar writes enabled
- verify the run finishes without failures
- if it fails, fix the bug and rerun the full folder again

Success criteria:

- full run completes with `0` failures
- manifest includes the new score fields and decision traces
- Lightroom-compatible sidecars are updated for the full folder

## Phase 6: Post-Run Evaluation

After the full run succeeds:

- compare new reject/review/pick counts to the previous production run
- inspect a few known edge cases
- decide whether `BRISQUE` or `CPBD` should stay observational or become active
  reject votes by default

This phase is evaluation only. Threshold changes should follow evidence from the
trace data, not guesswork.
