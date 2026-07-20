# Changelog

All notable changes to MeshProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-07-20

### Fixed

- Plain CLI help no longer includes Markdown syntax, so it remains readable and line-greppable.
- `render-image` now uses the session camera's framing aspect by default, while explicitly supplied
  dimensions still request an intentional reframe.
- `render-image --timeout` supports legitimate long CPU renders; a timeout now cancels and restores
  the Blender worker rather than retrying the render and leaving the session busy.
- Runtime failures from `render-image` now print a plain `Render failed:` message with exit code 1
  instead of argument-usage formatting.
- Windows worker exits now include hexadecimal exception codes and a conditional NVIDIA driver
  reset/TDR diagnostic for `0xC0000409`.

### Changed

- Public qualification-v8 and held-out private-v8 manifests now pin the MeshProbe 1.2.1 runtime.

## [1.2.0] - 2026-07-19

### Added

- `meshprobe help` now supports the cmdhelp v0.1 surface: concise text help, Markdown and JSON
  command references, `llm` as a Markdown alias, capability discovery, scoped subcommands, and
  depth-limited command trees.
- Agent-facing help is generated from the live CLI tree and includes arguments, flags, enum values,
  examples, stdin/stdout behavior, exit codes, session context, and links to command result schemas.
- Public qualification-v8 and held-out private-v8 manifests now pin the MeshProbe 1.2.0 runtime
  and package hash.

## [1.1.0] - 2026-07-18

### Added

- Incremental isolation controls: an isolated component set can now be replaced, extended, or
  reduced without rebuilding the selection from scratch.
- Orbit-sweep contact sheets for comparing bounded camera combinations, and optional side-by-side
  reference-image comparisons with deterministic placement metadata.

### Changed

- An explicit `view-orbit --aspect-ratio` now updates inherited perspective sensor geometry, so
  the requested framing is reflected by the actual camera rather than diagnostics alone.

### Fixed

- Compact JSON receipts from `open` now include every scene-import warning with its stable code,
  including generated illumination, flattened hierarchy, and reconstructed camera lenses.
- Compact JSON receipts from `find` now include each matched component's ref, ID, name, and path.

### Changed

- Public qualification-v8 and held-out private-v8 manifests now pin the MeshProbe 1.1.0 runtime.

## [1.0.0] - 2026-07-18

### Added

- `--render` on `open`, `display`, `mark`, `illumination-set`, every `view-*` command,
  and `reset`, so agents can mutate visual state and inspect the resulting frame in one CLI
  invocation while retaining separate journal entries and machine-readable receipts.
- Durable `undo [COUNT]` for visual state changes. Queries and renders do not consume history,
  reset starts a new undo boundary, and undo remains atomic across renderer and daemon failures.

### Changed

- Opt-in renders preserve the session's framing aspect ratio within the default 2576-pixel
  render budget and report partial success clearly when the mutation succeeds but rendering fails.
- CLI help now lists illumination presets, explains every mark mode, and documents the focused
  contact sheet's context, blocker-removal, and six-axis panels.
- Durable `checkpoint.json` state now uses schema 3: compatibility baselines live in
  `replay_prefix`, while `accepted_commands` contains only undoable mutations since the last reset.
- Opaque-scene foreground checks use a uniform material override instead of rebuilding materials
  for every component, while transparent and mixed-culling scenes retain exact material handling.
- Public qualification-v8 and held-out private-v8 manifests now pin the MeshProbe 1.0.0 package
  and importer identities.

### Fixed

- Labeled components are now composited as clip-aware annotations, so surrounding geometry cannot
  hide them and depth of field cannot blur them. Labels remain excluded from scene lighting and
  evaluator Depth/Normal evidence in both Eevee and Cycles.

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
  selected components; `--focus` on other view commands remains a diagnostic and does not move
  the camera.
- `view-rotate --frame camera` for rotations relative to the active camera basis.
- Fast `screen_edges` inspection rendering, now the default style, plus `shaded_edges` for
  configurable CAD-style edges.
- `--background-srgb` for exact display-referred backgrounds; component name and derived orbit
  angle fields in durable session state; glob selectors for `display`, `mark`, `render-sheet`,
  `focus`, and `occlusion`.
- `meshprobe --version`, `python -m meshprobe`, per-command schema lookup, and Windows Blender
  auto-discovery.
- Result-envelope schemas in `meshprobe schema` and CLI help, so automation can validate command
  responses without inferring their wrapper shape.
- Warnings for effectively empty render frames and for a render resolution whose aspect ratio
  differs from the camera framing.

### Changed

- Default render resolution is now 2576 × 2576 for inspection output.
- `view-orbit --projection-json` is optional; a bare orbit inherits the active projection, while
  `--focal-length` and `--ortho-scale` provide concise projection overrides.
- `view-rotate` defaults to the source frame, and the default highlight color is deep pink.
- Root-level CLI options can appear after a subcommand.
- `neutral_studio` is brighter for scenes without source lighting.
- A `find` command that matches nothing reports an explicit zero match count in its default
  receipt rather than looking like an unavailable result.

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
- The GPU-adapter compatibility warning is emitted once per worker, rather than once for every
  command in an affected session.
- Public and private qualification manifests are repinned to the 0.4.0 runtime.

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

[Unreleased]: https://github.com/pedropaulovc/meshprobe/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/pedropaulovc/meshprobe/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/pedropaulovc/meshprobe/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/pedropaulovc/meshprobe/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/pedropaulovc/meshprobe/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/pedropaulovc/meshprobe/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pedropaulovc/meshprobe/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pedropaulovc/meshprobe/releases/tag/v0.2.0
