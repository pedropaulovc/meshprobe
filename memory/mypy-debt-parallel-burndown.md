---
name: mypy-debt-parallel-burndown
description: CI only ever ran `mypy src`, never `tests/` — 243 test-side mypy errors accumulated silently over the whole project history; burned to zero via 6 parallel worktree-isolated agents, one PR per group
metadata:
  type: feedback
---

`.github/workflows/ci.yml` and `publish.yml` both ran `uv run mypy src` only,
never `mypy src tests`. Result: 243 mypy errors (enum-literal-vs-string
mismatches, unnarrowed discriminated-union attribute access, `Optional` used
without narrowing, heterogeneous-dict-unpack errors) accumulated across 20 test
files over the entire ~276-commit history, completely unnoticed by CI.

**How it was confirmed non-regression:** checked out a clean `origin/main`
worktree and reran `mypy src tests` there — identical error count/files,
proving it predated any in-flight branch. Then bisected ~24 sample points
across full git history (`git log --reverse` sampled every N commits, mypy run
against each via a scratch worktree reusing the already-`uv sync`'d venv's
`.venv/bin/mypy` directly) — error count grew roughly monotonically from the
project's first few commits to the present, tracking file-count growth, i.e.
genuine slow accumulation from an uncovered lint gap, not one bad commit (one
real ~25-error step existed at a specific older commit range — the
`view-rotate --frame camera` + background-compositing work, commits
`550ab70`..`dd77533` — cited separately when the user pushed back with
"likely recent regression").

**Why the parallel-agent split worked well:** the user said "burn down debt to
zero; use parallel agents if possible." Split 243 errors across 20 files into 6
roughly-balanced groups by per-file error count (`mypy ... | grep ': error:' |
sed 's/:[0-9]*: error:.*//' | sort | uniq -c | sort -rn` to get the breakdown),
then launched 6 parallel `Agent` calls with `subagent_type: general-purpose` and
`isolation: 'worktree'` — worktree isolation was necessary (not optional)
because each agent runs its own `git commit`/`git push`, and concurrent git
operations in a shared working tree/index would race even though the edited
files themselves were disjoint. Each agent: fixed only its assigned file(s) as
genuine type narrowing (never `type: ignore` except as an explicit last resort
with justification), verified with `mypy src <its files>` (scoped, so it
doesn't see noise from the other groups' still-broken files), ran ruff
format/check and the relevant pytest files to confirm zero behavior change,
committed, pushed its own branch, and opened its own **draft** PR.

**Result (PRs #86-#91):** 0 `type: ignore` suppressions across all 6 groups, 0
suspected production (`src/`) bugs uncovered — every error was a genuine
test-side typing gap. One agent found and fixed a latent cross-platform test
bug as a side effect (a sandbox path-mount test compared against `Path(...)`
when the real code always returns `PurePosixPath` for container-side paths —
would have silently passed only on POSIX hosts). Total wall-clock was bounded
by the slowest single group (the ~55-error blender-import integration-test
file, which also had to run a real ~6 min pytest suite against a real Blender
binary) rather than the sum of all groups' work.

**Follow-up required for this to actually prevent regrowth:** after all fix
PRs land, `ci.yml`/`publish.yml`'s `mypy src` must become `mypy src tests` —
otherwise the exact same silent-accumulation failure mode recurs immediately
for the next feature that adds an untyped test.
