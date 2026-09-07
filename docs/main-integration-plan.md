# Safe main integration plan

## Goal

Move the complete tracked Cull.sh history to `main` without changing the current
production contract: Cull.sh performs culling and writes Lightroom-compatible
metadata; Lightroom finishing/export remains manual. Experimental AI editing and
RapidRAW commands stay explicit and opt-in. Gitignored run data, RAW/JPEG files,
model weights, and machine-specific state never enter Git.

The immutable backup branch is `codex/sentosa-unattended-quality`. Do not force
push or rewrite it. The existing history is linear, so integration can use
stacked PRs and preserve bisectable commits.

On September 6, a plain push inherited the user's global
`push.default=matching` setting. It fast-forwarded `main` through the first
layer and expanded the remote PR #3 branch through `dafef4e`. No commit was lost
and the checkpoint branch was published correctly. Repository-local
`push.default=simple` now prevents another multi-branch push. Use an explicit
source and destination ref for every integration push anyway.

## Required gate zero: CI on main

Before merging any additional feature code, land `.github/workflows/tests.yml`
alone from `codex/main-ci-gate` (PR #4), based on current `origin/main`. Require
its test job for later PRs.
If repository settings do not allow a required check, still wait for a successful
run on every exact PR head before merging. Do not treat a run from another SHA as
evidence for the current head.

## Integration train

Merge one layer at a time. After each merge, update the next PR's base to `main`,
resolve only real conflicts, rerun the full test workflow, and perform the listed
contract check. Use normal merge commits rather than squash or force-push so the
validated boundaries and original history remain recoverable.

1. **AI suggestion foundation — PR #2, already on main**
   - `main` fast-forwarded to `1f97bcb`; GitHub records PR #2 as merged.
   - The exact `1f97bcb` checkout passed CLI import and all 72 tests locally.
     Do not rewrite it merely to change the merge mechanism.
   - Contract: regular `cull` behavior remains separate from the explicit
     `suggest-edits` command; dry run remains the default; XMP writes remain
     opt-in.
2. **TOPIQ, benchmark, and initial RapidRAW experiment — recover original PR #3 boundary**
   - Range: `1f97bcb..e61912d`
   - After PR #4 is green and merged, publish a new immutable branch at
     `e61912d` with an explicit refspec and open it against `main`.
   - Leave PR #3 unmerged; its head now points to the later `dafef4e` boundary.
   - Contract: TOPIQ affects ordering only at 25 percent and never casts a hard
     reject vote; RapidRAW requires its explicit stage/preview/export commands.
3. **Qwen culling, blind review, and rendered-validation development**
   - Range: `e61912d..dafef4e`.
   - After layer two merges, change PR #3's base to `main`, retitle it for this
     actual layer, and review the newly narrowed diff. Do not move its head.
   - Contract: Qwen uses one attempt, thinking enabled, bounded output, and no
     fallback; existing human picks survive replay; blind-test answer keys remain
     separated from review pages.
4. **Sentosa quality/correction checkpoint and final workflow policy**
   - Range: `dafef4e..codex/sentosa-unattended-quality`.
   - Open the already-published checkpoint branch against the layer-three branch.
   - Contract: album selection and all editing/export commands remain explicit;
     no local experiment is described as an independent benchmark; production
     documentation keeps Lightroom finishing manual and RapidRAW paused.

Do not merge PR #3 or open layer four against `main` prematurely: their diffs
would include unmerged parent work and obscure the actual review boundary.

## Gate for every PR head

- Worktree contains no unexpected tracked or untracked files.
- `git diff --check <base>...HEAD` passes.
- No RAW/JPEG/model/run artifacts or credentials are present in the Git diff.
- `python main.py --help` imports successfully in a clean environment.
- `python -m pytest tests -q` passes on the exact head SHA in CI.
- A read-only fixture smoke test confirms `cull` dry-run writes no source photo
  or sidecar bytes.
- Any command that can write XMP, stage RAWs, invoke a renderer, or export JPEGs
  remains behind an explicit command or `--no-dry-run`/approval gate.
- Reviewer checks the user-visible defaults in `README.md`, `config.py`, and CLI
  help rather than inferring behavior from tests alone.

If a layer fails a gate, fix it on that layer and revalidate all descendants by
rebasing them onto the repaired commit without rewriting the immutable checkpoint
branch. Create a new corrected descendant branch instead.

## Final main verification

After the fourth merge, verify `origin/main` contains the final reviewed SHA,
run the complete test suite once more from a fresh checkout, and repeat a dry-run
cull on a copied fixture. Do not run a live model, write photo metadata, or start
full-album processing merely to prove the Git merge. Record the final `main` SHA
and CI URL in `HANDOFF.md` only after those checks are green.
