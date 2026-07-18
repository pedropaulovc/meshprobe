# Changelog

All notable changes to MeshProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-18

### Added

- A plausibility warning on `scene.open` when the imported root bounds span more than 50 m or
  less than 1 mm, the classic signature of a millimeter/meter unit mistake, plus a `--unit-scale`
  CLI option to correct a wrong-unit asset without a re-export cycle.
- Blender 4.2 support for software-compatible rendering, with an honest unknown-graphics status
  where that Blender version cannot report a platform. Hardware-required work remains rejected
  unless hardware is actually observed.
- Automatic default camera framing on open and reset so the model fills the render without
  clipping. The session retains its chosen framing aspect ratio through movement, recovery, and
  daemon upgrades.
- `view.frame` and `view-frame` for fitting an absolute perspective or orthographic view to
  selected components.
- Fast `screen_edges` inspection rendering, now the default style, plus `shaded_edges` for
  configurable CAD-style edges.
- `--background-srgb` for exact display-referred backgrounds; component name and derived orbit
  angle fields in durable session state; glob selectors for focus and occlusion operations.
- `meshprobe --version`, `python -m meshprobe`, per-command schema lookup, and Windows Blender
  auto-discovery.
- Warnings for effectively empty render frames and for a render resolution whose aspect ratio
  differs from the camera framing.

### Changed

- Default render resolution is now 2576 × 2576 for inspection output.
- `view-orbit --projection-json` is optional; a bare orbit inherits the active projection.
- `view-rotate` defaults to the source frame, and the default highlight color is deep pink.
- Root-level CLI options can appear after a subcommand.
- `neutral_studio` is brighter for scenes without source lighting.

### Fixed

- View framing and camera diagnostics now use the correct projection geometry for perspective and
  orthographic cameras, including non-square renders and clip-plane refreshes after motion.
- Invalid raw camera aspect values and rejected camera commands leave existing session state
  unchanged.
- Evaluator depth and normal passes no longer pay for Freestyle, and foreground masks correctly
  sample textured alpha cutouts.
- Render-style fallbacks and durable checkpoints retain the worker-resolved style and legacy
  source-camera baseline across recovery.
- GPU probing fails clearly on unsupported Blender versions instead of calling unavailable APIs.

## [0.3.0] - 2026-07-17

### Added

- Relative camera translation in world and camera frames through `view.move` and the
  `view-move` CLI command, with unit-aware deltas and replayable movement receipts.
- Basis-relative camera rotation in source and world frames through `view.rotate` and the
  `view-rotate` CLI command, with orthonormal basis validation and explicit visual-angle receipts.
- Perspective depth-of-field controls for aperture, exact focus distance, and component-based
  focus, with the resolved focus distance recorded in render receipts.
- A `shaded_edges` CAD render style with configurable line color, width, crease angle, and edge
  types including silhouettes, borders, contours, creases, and material boundaries.
- Custom sRGB highlight colors for marked components.
- CLI support for custom illumination, visible-background overrides, and their resolved state.
- Source and world coordinate-frame metadata for GLTF scenes, including source-to-world
  transforms and frame-tagged camera poses, bounds, and receipts.
- Explicit `hardware_required` and `software_allowed` graphics policies, plus render receipts and
  session status fields describing the graphics vendor, renderer, backend, Blender device, and
  detected hardware class.
- A metadata-only `component.occlusion` query and `occlusion` CLI command for the current camera,
  reporting visible and occluded sample counts, visibility fraction, blocker identities and hit
  counts, and the evaluated camera without generating image artifacts.

### Changed

- Component arguments now accept exact display names throughout the CLI. Ambiguous names report
  matching component paths, while receipts retain stable IDs alongside readable names and paths.
- State-changing operations now return compact operation-local deltas, state hashes, and durable
  result paths instead of repeating the full session snapshot. Full state remains available from
  `session.snapshot`.
- Contact-sheet occlusion evidence now names blocker components, frames focused comparison panels
  around the target, reuses the same camera before and after blocker removal, recomputes visibility
  after every removal, and records which generated camera produced the measurements.
- Visible background color is independent from ambient illumination, so backdrop overrides no
  longer change preset lighting or material brightness.
- Render styles, camera operations, depth-of-field state, coordinate frames, and graphics details
  are persisted in versioned session and evidence manifests for deterministic replay and recovery.
- Public and private qualification corpora are migrated to evaluation schema 3 and repinned for
  MeshProbe 0.3.0 as `qualification-v7` and `private-v8` without changing model identities,
  episode identities, or evaluator truth.
- After CI passes, releases now validate the workflows, changelog, wheel, and source distribution;
  tag the tested `main` commit; publish to PyPI with retry-safe duplicate checks; and create the
  GitHub Release only after publishing succeeds.

### Fixed

- `meshprobe.__version__` is derived from installed distribution metadata so it cannot drift from
  the package version used by PyPI and runtime qualification pins.
- WSL2 EEVEE rendering now detects and uses Mesa D3D12 on the selected NVIDIA adapter instead of
  silently labeling llvmpipe software rendering as hardware graphics.
- Graphics policy now propagates to internal visibility and contact-sheet renders.
- GLTF camera operations now honor the source coordinate system instead of exposing Blender's
  converted axes without a frame label.
- Camera, render-style, illumination, and evaluation mutations are validated before state changes
  and remain atomic when rendering or artifact validation fails.
- Checkpoint upgrades preserve camera rotation and render-style state, and failed upgrades stop the
  recovery worker instead of leaving an inconsistent background process.

## [0.2.0] - 2026-07-16

### Added

- Read-only inspection sessions for GLB, glTF, OBJ, and STL models, backed by a persistent
  Blender worker.
- Camera, projection, lighting, visibility, isolation, marking, and component query operations.
- Evidence renders, evaluator-only render passes, and focused contact sheets.
- Durable CLI sessions with checkpoints, event logs, crash recovery, and compact YAML state.
- An MCP server with the same typed operation contracts as the CLI.
- An isolated qualification harness with procedural and curated corpora, pinned public and
  private tiers, clean-install verification, and JSON and Markdown reports.
- Linux and Windows support with Bubblewrap and AppContainer sandboxing.
- PyPI releases through GitHub Actions and OIDC trusted publishing.

[Unreleased]: https://github.com/pedropaulovc/meshprobe/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/pedropaulovc/meshprobe/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pedropaulovc/meshprobe/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pedropaulovc/meshprobe/releases/tag/v0.2.0
