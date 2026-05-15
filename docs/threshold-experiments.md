# Threshold Experiments

This document records concrete local-quality thresholds and experiment flags used
for full-folder production reruns.

## Baseline Support-Metric Trace Run

Run artifacts:

- `runs/20260412-204203-368526`

Command:

```bash
.venv/bin/python main.py cull \
  --path '/path/to/sample-raw-folder' \
  --genre mixed \
  --prefer 'strong light, interesting shapes, clean lines, and distinct subjects' \
  --score-workers 2 \
  --no-interactive \
  --no-dry-run
```

Local thresholds:

- `min_blur_score = 110.0`
- `min_tenengrad_score = 45.0`
- `min_musiq_score = 50.0`
- `min_nima_score = 4.6`
- `max_brisque_score = 55.0`
- `min_cpbd_score = 0.3`
- `local_reject_required_support_votes = 2`

Support metric gate flags:

- `use_brisque_for_reject = false`
- `use_cpbd_for_reject = false`

Interpretation:

- `MUSIQ` and `NIMA` participate in the local reject consensus
- `BRISQUE` and `CPBD` are recorded and traced, but do not contribute weak votes

## Full Rerun With Support Metrics Active

Purpose:

- test whether `BRISQUE` and `CPBD` should actively contribute to local reject
- preserve the exact parameter set used for the rerun

Command:

```bash
.venv/bin/python main.py cull \
  --path '/path/to/sample-raw-folder' \
  --genre mixed \
  --prefer 'strong light, interesting shapes, clean lines, and distinct subjects' \
  --score-workers 2 \
  --use-brisque-for-reject \
  --use-cpbd-for-reject \
  --no-interactive \
  --no-dry-run
```

Local thresholds:

- `min_blur_score = 110.0`
- `min_tenengrad_score = 45.0`
- `min_musiq_score = 50.0`
- `min_nima_score = 4.6`
- `max_brisque_score = 55.0`
- `min_cpbd_score = 0.3`
- `local_reject_required_support_votes = 2`

Support metric gate flags:

- `use_brisque_for_reject = true`
- `use_cpbd_for_reject = true`

Expected local reject rule:

- primary blur gate must still be weak:
  - Laplacian weak
  - Tenengrad weak
- and at least `2` available support votes must also be weak across:
  - `MUSIQ`
  - `NIMA`
  - `BRISQUE`
  - `CPBD`

Results:

- run artifacts: `runs/20260412-222016-922899`
- extraction: `0:00:12`
- local quality: `0:12:01`
- vision scoring: `0:26:08`
- locally rejected: `154`
- review: `132`
- pick: `35`
- sent to vision: `167`
- sidecars written: `321`
- failures: `0`
