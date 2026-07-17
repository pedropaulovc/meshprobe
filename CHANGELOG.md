# Changelog

All notable changes to MeshProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Releases now create a version tag from `pyproject.toml`, publish to PyPI, and create the
  GitHub Release only after publishing succeeds.

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

[Unreleased]: https://github.com/pedropaulovc/meshprobe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pedropaulovc/meshprobe/releases/tag/v0.2.0
