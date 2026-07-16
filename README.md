# MeshProbe

MeshProbe gives AI agents a small, typed interface for inspecting 3D assemblies
without editing the source model. Agents can identify components, move a camera,
switch between orthographic and perspective projection, control lens and lighting,
hide occluders, mark parts, and request evidence renders or contact sheets.

MeshProbe starts a persistent, factory-clean Blender worker. It imports GLB, glTF, OBJ,
or STL files and returns a validated scene manifest without changing the source file.
It can render the current inspection state with Eevee or CUDA-backed Cycles, produce
private component and highlight masks plus a Depth/Normal EXR for evaluators, rank
line-of-sight occluders, and build a focused 3×3 contact sheet with perspective context
and six orthographic detail views.

## The Playwright CLI model, for 3D

If you have used Playwright CLI, MeshProbe's interaction model should feel familiar.
Playwright opens a named browser session, applies small operations such as navigation and
clicks, and keeps a compact page snapshot that an agent can query. MeshProbe does the same
for a 3D model: it opens a named inspection session, applies camera, lighting, display, and
marking operations, and writes compact state under `.meshprobe`.

That makes MeshProbe useful when an agent needs to investigate a model over several turns.
The model is imported once. Later commands reuse the same Blender worker, and a daemon
restart reconstructs the session from its last acknowledged checkpoint. The agent usually
reads `components.yml` and `state.yml` with `rg` or `yq`; the complete manifest remains in
`scene.json` for `jq` queries.

```bash
uv run meshprobe --session gearbox open models/gearbox.glb
uv run meshprobe --session gearbox find '**/idler*'
yq '.components[] | select(.name | test("idler"; "i"))' \
  .meshprobe/sessions/gearbox/components.yml
uv run meshprobe --session gearbox display c7 --mode isolated
uv run meshprobe --session gearbox illumination-set raking_left
uv run meshprobe --session gearbox render-sheet c7
```

Normal output is a short receipt containing the result and state paths. Add `--json` or
`--yaml` for a machine-readable receipt, or `--raw` when the full operation result is actually
needed. `meshprobe schema --kind state` gives a compact field map for every durable file;
add `--full` for formal JSON Schemas, or put global `--yaml` before `schema` for YAML output.

## Install

MeshProbe uses Python 3.12 or newer, `uv`, and Blender 5.2 LTS on Linux and Windows.
The qualification harness uses Bubblewrap on Linux. On Ubuntu 24.04 or newer, load the
distro's AppArmor profile. It grants Bubblewrap the namespace permissions needed by the
harness and keeps user-namespace hardening enabled:

```bash
sudo apt-get install apparmor-profiles bubblewrap
sudo cp \
  /usr/share/apparmor/extra-profiles/bwrap-userns-restrict \
  /etc/apparmor.d/bwrap-userns-restrict
sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict
```

Windows needs no additional sandbox package. MeshProbe creates a fresh AppContainer
profile for each episode and puts the agent in a constrained Job Object. Install the
pinned Blender release from an elevated PowerShell prompt:

```powershell
choco install blender --version=5.2.0 --yes
```

Install the locked Python environment and run the test suite:

```bash
uv sync
uv run meshprobe schema
uv run pytest
```

## Sessions and lifecycle commands

MeshProbe stores session data in the current project's `.meshprobe` directory by default.
Use `--workspace` to choose another project root and `-s/--session` to switch sessions.

The lifecycle commands have deliberately different failure and persistence semantics:

| Command | Renderer shutdown | Checkpoint retained | Daemon |
| --- | --- | --- | --- |
| `close` | Graceful, selected session | Yes | Keeps running |
| `close --all` | Graceful, every session | Yes | Stops gracefully |
| `kill` | Immediate, selected session | Last acknowledged state | Keeps running |
| `kill --all` | Immediate process-tree stop | Last acknowledged state | Force-stops |
| `delete-data` | Requires a stopped daemon | Deletes all `.meshprobe` data | Stays stopped |

Use `close` at the end of ordinary work. Use `kill` to test recovery or escape a wedged
renderer. Neither command deletes the session: the next inspection operation reopens the
source, verifies its hash, and replays the checkpoint. `delete-data` is the explicit cleanup
operation.

Each session contains:

```text
.meshprobe/sessions/<name>/
  metadata.json       # source, status, worker PID
  scene.json          # complete scene manifest
  components.yml      # deterministic c1, c2, ... reference index
  state.yml           # camera, lighting, sparse display/mark overrides
  checkpoint.json     # acknowledged replay log
  events.jsonl        # accepted and rejected operations
  results/ snapshots/ artifacts/
```

The CLI and MCP adapter share the same Pydantic operation contracts. Blender remains a
separate factory-clean worker, and source files stay untouched. Renders, manifests, traces,
and normalized cache entries are derived artifacts.

MCP clients get the same discriminated command schema and result envelope through one
`meshprobe` tool. Start the stdio server with:

```bash
uv run meshprobe-mcp
```

## Qualification harness

The evaluation harness runs each agent without network access, with a read-only assigned
model, a writable artifact directory, and evaluator-owned traces and render passes. It
uses Bubblewrap on Linux and AppContainer plus Job Object limits on Windows. It supports
a JSONL command-line protocol and an MCP stdio protocol. Both adapters use the same
prompt, command schema, budgets, broker, and answer contract.

Build the released 512-model procedural corpus and the 160-model curated track, then
combine and pin them:

```bash
uv run meshprobe eval generate .corpora --version procedural-v6
uv run meshprobe eval curated-generate \
  evals/curated/catalog.json .cache/meshprobe-curated .corpora \
  --build-version curated-v2 --corpus-version curated-tasks-v6
uv run meshprobe eval merge .corpora \
  .corpora/procedural-v6 .corpora/curated-tasks-v6 \
  --version qualification-v6
uv run meshprobe eval pin \
  .corpora/qualification-v6 .corpora/manifests-v6
```

The resulting release corpus has 672 models, 2,528 episodes, and 672 full-stack
investigations. The nested smoke, pull-request, nightly, and release manifests committed
under `evals/manifests/public/` pin every episode, model, corpus, MeshProbe version,
Blender version, importer, and render engine. The curated catalog pins the source commit,
download hash, topology hash, license, and attribution for 20 CC0 assets; the build
creates eight controlled variants of each three-source inspection assembly.

Run a pinned tier with either agent transport:

```bash
uv run meshprobe eval run-tier \
  .corpora/qualification-v6 evals/manifests/public/smoke.json .runs \
  --adapter cli \
  --blender /path/to/blender \
  --agent-command-json '["/path/to/agent"]'
```

The CLI agent reads one episode envelope, sends `tool_call` lines, and finishes with a
structured `submission`. With `--adapter mcp`, the agent speaks MCP as a client over its
standard streams and calls the `meshprobe` and `submit` tools. Runs checkpoint after each
episode and publish JSON plus Markdown reports split by operation, task family,
difficulty, projection, focal-length band, illumination, model source, and failure gate.
On Linux the episode envelope uses `/workspace/input` and `/workspace/artifacts` paths;
on Windows it uses the assigned absolute paths granted to the AppContainer.

To test the packaged wheel instead of the checkout, run the pinned smoke tier through a
clean uv environment:

```bash
uv run python tools/clean_install_smoke.py \
  .corpora/qualification-v6 evals/manifests/public/smoke.json \
  .runs/clean-install-smoke
```

The command records the wheel hash, runtime versions, manifest hash, and qualification
report hash in `clean-install-report.json`.

To migrate existing local datasets without regenerating geometry or task identities:

```bash
uv run meshprobe eval migrate .corpora/procedural-v5 .corpora --version procedural-v6
uv run meshprobe eval migrate .corpora/curated-tasks-v5 .corpora \
  --version curated-tasks-v6
uv run meshprobe eval migrate .corpora/qualification-v5 .corpora \
  --version qualification-v6
uv run meshprobe eval migrate .corpora/private-v6 .corpora \
  --version private-v7 --opaque-family opaque_family_v7
```

`eval audit-migration` verifies that model hashes, episode IDs, evaluator truth, and the
full-investigation operation contract survived the migration. Schema-v1 corpora are rejected
by the qualification runtime instead of being translated silently.

Every qualification agent runs in the platform sandbox and retains its exact prompt, live
protocol stream, and stderr under `episodes/<episode>/evaluator/agent-run/`. The same audit
artifacts are retained by the ergonomics pilot.

The separate CLI ergonomics pilot pairs 12 basic and 12 intermediate episodes across Claude
Opus and Codex Luna (`gpt-5.6-luna`). It records exposed command trajectories and token
accounting; it does not claim access to hidden chain of thought and does not replace release
qualification. The harness silently enforces a 768,000-token ceiling on each attempt in
addition to its wall-clock limit. On Linux, agents run inside Bubblewrap with only the
assigned public model, a clean MeshProbe install, credentials, and their writable attempt
directory mounted. Blender is mounted read-only as a runtime dependency. The repository and
private evaluator data stay outside the sandbox:

```bash
uv run meshprobe eval ergonomics \
  .corpora/qualification-v6 .runs/ergonomics-v1
```

Live provider streams and stderr are saved beneath
`.runs/ergonomics-v1/attempts/<episode>/<agent>/attempt-<n>/`; the exact prompt is saved there
as `prompt.txt`.
The ergonomics pilot currently runs on Linux; Windows support will use AppContainer isolation
and will not fall back to an unsandboxed agent.

The approved session and migration design is in
[docs/session-cli-plan.md](docs/session-cli-plan.md). The broader qualification design remains
in [docs/plan.md](docs/plan.md).

## License

Apache-2.0.
