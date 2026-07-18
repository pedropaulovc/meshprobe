---
name: private-corpus-location
description: the held-out private-vN eval corpus lives only in the base (non-worktree) checkout's .corpora/, not in any feature worktree or the public generate/curated-generate pipeline
metadata:
  type: reference
---

The qualification harness's "private" tier (`evals/manifests/private/private.json`)
is NOT reproducible via the documented public `eval generate` /
`eval curated-generate` / `eval merge` / `eval pin` pipeline in README.md — that
pipeline only builds the public procedural + curated corpora. The private tier
is instead produced via `eval migrate`:

```
uv run meshprobe eval migrate .corpora/private-v7 .corpora \
  --version private-v8 --opaque-family opaque_family_v8
```

which requires an existing `private-v7` corpus directory as input — it migrates
forward, it doesn't generate from scratch. That source corpus is genuinely
held-out/private data (README: "private evaluator data stay outside the
sandbox"), so it won't exist in a fresh feature-branch worktree
(`meshprobe-wt-*`). It DOES exist locally on this machine in the base repo
checkout's `.corpora/` (the non-worktree checkout, e.g. `.corpora/private-v7`)
— locate it via `locate`/`find` across the filesystem when a Codex
review finding claimed the private manifest's runtime pin was stale. Also
present there: `private-v6`, `private-v8`. Transient temp directories from
past test/PR runs have also held copies — not durable, don't rely on those.

**How to apply:** before concluding "I can't reproduce/fix the private tier,
the data isn't available," check the base (non-worktree) checkout's
`.corpora/` first — don't assume private/held-out data means literally
inaccessible on this machine. See also [[mypy-debt-parallel-burndown]] for the
broader mypy-debt campaign this was discovered during.
