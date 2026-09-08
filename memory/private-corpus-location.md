---
name: private-corpus-location
description: held-out private evaluation data lives only in the base checkout's .corpora/, with separate identity-preserving migration and identity-changing replacement paths
metadata:
  type: reference
---

The documented public `eval generate` / `eval curated-generate` / `eval merge`
pipeline builds only the public corpus. It cannot recover the held-out private tier in
`evals/manifests/private/private.json`.

## Preserve historical identity when source data exists

If a valid retained `private-v7` corpus exists, migrate it to `private-v8` without changing model
or episode identities:

```sh
uv run meshprobe eval migrate .corpora/private-v7 .corpora \
  --version private-v8 --opaque-family opaque_family_v8
```

This preserves the historical `private-v8` identity, not the committed `private-v20`
replacement manifest. Use it only when retained v7 source material has actually been recovered;
then regenerate a matching private-v8 pin deliberately. Historical `private-v6` / `private-v7`
payloads are not present on this machine after the 2026-09 data loss; never claim they are without
checking the base checkout's `.corpora/`.

## Replace lost private material

The original private-v8 payload was lost in 2026-09. The retained
`meshprobe-private/generate.py` generator rebuilt a valid replacement at
`.corpora/private-v20`. This replacement is **not** the historical corpus: its
`corpus_manifest_sha256` and `generator_sha256` in `evals/manifests/private/private.json`
identify new data. Never hand-edit those fields or reuse the v8 label for a replacement.

Run from the base checkout, after confirming neither the destination nor its staging directory
contains material worth preserving. Replace the `source` placeholder with the retained generator
path before running the snippet. The module-global override changes only the new corpus's
versioned output path and manifest; it does not modify the retained private generator source.

```python
import importlib.util
from pathlib import Path

source = Path("/path/to/meshprobe-private/generate.py")  # replace with the retained generator path
output_root = Path(".corpora").resolve()
version = "private-v20"
if (output_root / version).exists() or (output_root / f".{version}.building").exists():
    raise RuntimeError("refusing to replace existing corpus material")

spec = importlib.util.spec_from_file_location("meshprobe_private_rebuild", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.CORPUS_VERSION = version
print(module.build(output_root))
```

Then regenerate, never patch, the private tier manifest:

```sh
uv run meshprobe eval pin .corpora/private-v20 evals/manifests/private \
  --private --blender /path/to/blender
```

`build()` and `pin` both validate the schema-3 corpus. The 2026-09 private-v20 replacement
validated 640 models and 2,560 episodes; `validate_tier_manifest` plus
`validate_runtime_pin` confirmed all new pins before commit.

## How to apply

Before treating a private pin as irreproducible, check the base (non-worktree) checkout's
`.corpora/`. Feature worktrees intentionally do not contain this ignored data. If retained v7 data
is available, migrate it. If it is absent, use the replacement path only with explicit
authorization, record the changed identity in the changelog, and repin every private manifest
from the validated replacement.
