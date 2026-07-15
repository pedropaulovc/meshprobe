# MeshProbe

MeshProbe gives AI agents a small, typed interface for inspecting 3D assemblies
without editing the source model. Agents can identify components, move a camera,
switch between orthographic and perspective projection, control lens and lighting,
hide occluders, mark parts, and request evidence renders or contact sheets.

The project is in early development. The current slice implements the public command
schema, scene manifests, deterministic component selectors, visual session state,
camera and illumination validation, and a CLI for exercising those contracts. Blender
rendering is the next slice.

## Try the protocol

MeshProbe uses Python 3.12 or newer and `uv`:

```bash
uv sync
uv run meshprobe schema
uv run pytest
```

Validate a scene manifest or find components in one:

```bash
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

