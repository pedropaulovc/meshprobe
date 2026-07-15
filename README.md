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

## Try the protocol

MeshProbe uses Python 3.12 or newer and `uv`:

```bash
uv sync
uv run meshprobe schema
uv run pytest
```

Validate a scene manifest or find components in one:

```bash
uv run meshprobe open assembly.glb > scene.json
uv run meshprobe validate-manifest scene.json
uv run meshprobe find scene.json 'assembly/**/idler*' --kind glob
```

The CLI and MCP adapter use the same Pydantic contracts. Blender runs as a
separate factory-clean worker and receives line-delimited JSON commands. Source files
remain untouched; renders, manifests, traces, and normalized cache entries are derived
artifacts.

For a live inspection, put the public commands in a JSONL file. The first command must
open the scene; every later command uses the same Blender session:

```json
{"request_id":"open","op":"scene.open","source_path":"/models/assembly.glb"}
{"request_id":"parts","op":"component.find","selector":{"kind":"glob","pattern":"**/idler*"}}
{"request_id":"light","op":"illumination.set","illumination":{"preset":"raking_left"}}
{"request_id":"render","op":"render.image","output_path":"evidence.png"}
```

```bash
uv run meshprobe run inspection.jsonl
```

MCP clients get the same discriminated command schema and result envelope through one
`meshprobe` tool. Start the stdio server with:

```bash
uv run meshprobe-mcp
```

## Qualification harness

The evaluation harness runs each agent in a Bubblewrap sandbox with no network, a
read-only assigned model, a writable artifact directory, and evaluator-owned traces and
render passes. It supports a JSONL command-line protocol and an MCP stdio protocol. Both
adapters use the same prompt, command schema, budgets, broker, and answer contract.

Build the released 512-model procedural corpus and the 160-model curated track, then
combine and pin them:

```bash
uv run meshprobe eval generate .corpora --version procedural-v1
uv run meshprobe eval curated-generate \
  evals/curated/catalog.json .cache/meshprobe-curated .corpora
uv run meshprobe eval merge .corpora \
  .corpora/procedural-v1 .corpora/curated-tasks-v1 \
  --version qualification-v1
uv run meshprobe eval pin \
  .corpora/qualification-v1 .corpora/manifests
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
  .corpora/qualification-v1 evals/manifests/public/smoke.json .runs \
  --adapter cli \
  --agent-command-json '["/path/to/agent"]'
```

The CLI agent reads one episode envelope, sends `tool_call` lines, and finishes with a
structured `submission`. With `--adapter mcp`, the agent speaks MCP as a client over its
standard streams and calls the `meshprobe` and `submit` tools. Runs checkpoint after each
episode and publish JSON plus Markdown reports split by operation, task family,
difficulty, projection, focal-length band, illumination, model source, and failure gate.

The approved implementation and evaluation design is in [docs/plan.md](docs/plan.md).

## License

Apache-2.0.
