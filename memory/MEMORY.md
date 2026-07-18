# Memory index

- [mypy debt parallel burndown](mypy-debt-parallel-burndown.md) — CI only ran `mypy src`; 243 test-side errors accumulated silently; fixed via 6 parallel worktree-isolated agents, one PR each, then gate `mypy src tests` in CI to stop regrowth
- [Private corpus location](private-corpus-location.md) — held-out `private-vN` eval corpus lives only in the base (non-worktree) checkout's `.corpora/`, needed by `eval migrate` to repin the private tier
