# MeshProbe

MeshProbe gives AI agents a small, typed interface for inspecting 3D assemblies
without editing the source model. Agents can identify components, move a camera,
switch between orthographic and perspective projection, control lens and lighting,
hide occluders, mark parts, and request evidence renders or contact sheets.

The project is in early development. MeshProbe can start a persistent factory-clean
Blender worker, import GLB, glTF, OBJ, or STL files, and return a validated scene
manifest without changing the source file. It can render the current inspection state
with Eevee or CUDA-backed Cycles, emit private component/highlight masks and a
Depth/Normal EXR for evaluators, rank line-of-sight occluders, and build a focused 3×3
contact sheet with perspective context and six orthographic detail views.

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

The CLI and future MCP adapter use the same Pydantic contracts. Blender runs as a
separate factory-clean worker and receives line-delimited JSON commands. Source files
remain untouched; renders, manifests, traces, and normalized cache entries are derived
artifacts.

The approved implementation and evaluation design is in [docs/plan.md](docs/plan.md).

## License

Apache-2.0.
