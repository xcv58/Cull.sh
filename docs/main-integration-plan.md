# Safe main integration plan

## Goal and production contract

Move the complete tracked Cull.sh history to `main` without changing the current
production contract: Cull.sh performs culling, ranking, duplicate handling, and
writes Lightroom-compatible metadata. Lightroom finishing and JPEG export remain
manual. AI editing and RapidRAW commands remain explicit, experimental, and
opt-in. Gitignored run data, RAW/JPEG files, model weights, and machine-specific
state never enter Git.

`codex/sentosa-unattended-quality` is the forward-only recovery checkpoint. Do
not force-push or rewrite it. Integration uses normal merge commits and exact
head checks so the original commits and validated boundaries remain recoverable.

## Integration status — September 7, 2026

1. **AI suggestion foundation — PR #2**
   - Merged at `1f97bcb`.
   - The exact checkout passed CLI import and 72 tests locally.
2. **Required CI gate — PR #4**
   - Merged as `3e71aaf`.
   - Added `.github/workflows/tests.yml` with immutable action SHAs.
   - Exact post-merge CI passed.
3. **TOPIQ, benchmarks, and initial RapidRAW experiment — PR #5**
   - Original feature boundary: `e61912d`; up-to-date PR head: `bbb2f3e`.
   - Merged as `485928c`; exact PR and post-merge CI passed.
   - Contract: TOPIQ affects ordering only at 25 percent and never casts a hard
     reject vote; RapidRAW remains behind explicit stage/preview/export commands.
4. **Qwen culling, blind review, and rendered-validation development — PR #3**
   - Original feature boundary: `dafef4e`; corrected PR head: `f215bc2`.
   - Merged as `3a03035`; exact PR and post-merge CI passed.
   - The CI gate caught and fixed a clean-checkout default-directory bug and an
     ANSI-sensitive assertion before merge.
   - Contract: Qwen uses one attempt, thinking enabled, bounded output, and no
     fallback; existing human picks survive replay; the unrelated local 12B
     model is not removed.
5. **Sentosa quality/correction checkpoint and final workflow policy**
   - Feature range begins after `dafef4e` and is carried by
     `codex/sentosa-unattended-quality`.
   - This is the final integration PR. It adds album selection, render
     diagnostics, bounded quality-pilot checkpoints, and isolated correction.
   - Contract: no full-album processing starts implicitly; source RAW/XMP and
     frozen outputs remain untouched; production documentation keeps Lightroom
     finishing manual and RapidRAW paused.

`main` is protected: pull requests are required, `pytest` must pass against an
up-to-date head, admin enforcement is enabled, conversations must be resolved,
and force-pushes and deletion are disabled. Required approvals remain zero so a
single-owner repository cannot deadlock itself.

## Gate for every PR head

- Worktree contains no unexpected tracked or untracked files.
- `git diff --check <base>...HEAD` passes.
- No RAW/JPEG/model/run artifacts or credentials are present in the Git diff.
- `python main.py --help` imports successfully in a clean environment.
- `python -m pytest tests -q` passes on the exact head SHA in CI.
- Dry-run regression coverage confirms no source photo or sidecar bytes are
  written without an explicit mutation command.
- Any command that can write XMP, stage RAWs, invoke a renderer, or export JPEGs
  remains behind an explicit command or `--no-dry-run`/approval gate.
- Reviewer checks user-visible defaults in `README.md`, `config.py`, and CLI
  help rather than inferring behavior from tests alone.

If a layer fails a gate, fix it on that layer, merge current `main` into the
layer without rewriting history, and validate the new exact head. Never bypass
the required check with administrator privileges.

## Push-safety recovery

On September 6, a plain push inherited the user's global
`push.default=matching` setting. It fast-forwarded `main` through the first layer
and expanded the remote PR #3 branch through `dafef4e`. No commit was lost.
Repository-local `push.default=simple` now prevents another multi-branch push;
all integration pushes still use explicit source and destination refs.

## Final verification

After the fifth merge, verify `origin/main` contains the final reviewed head,
wait for the exact post-merge CI run, and test a fresh detached checkout with:

- `python main.py --help`
- `python -m pytest tests -q`
- `git diff --check`
- dry-run/source-safety regression tests

Do not run a live model, write photo metadata, invoke Lightroom automation, or
start full-album processing merely to prove the Git merge. The final `main` SHA
and CI URL belong in the integration report and PR record; adding them to the
same commit would create a self-referential verification loop.
