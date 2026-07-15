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

The approved implementation and evaluation design is in [docs/plan.md](docs/plan.md).

## License

Apache-2.0.
