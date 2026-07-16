# Implementation status

This matrix maps the requirements in `docs/plan.md` to executable evidence. A row
is complete only when the public contract, implementation, and a relevant test all
exist. Counts and test names refer to the current branch.

| Requirement | Status | Evidence or remaining work |
| --- | --- | --- |
| Python 3.12+, uv, typed CLI and MCP service | Complete | `pyproject.toml`, `protocol.py`, `service.py`, `cli.py`, `mcp_server.py`; adapter contract tests |
| Factory-clean persistent Blender worker | Complete | `BlenderController.start`; real Blender integration tests |
| GLB primary; glTF, OBJ, STL capability reporting | Complete | importer integration tests, explicit unit and hierarchy warnings |
| Source assets remain immutable | Complete | bundle snapshots around import/render/sheets plus OS read-only safety test |
| Content-addressed normalized geometry | Complete | `artifacts.py`, `scene.export_normalized`, cache manifest attached to `SceneManifest` |
| Atomic publication, complete eviction, abandoned-write invisibility | Complete | `test_artifacts.py`, including concurrent publishers and interrupted writes |
| Stable hierarchy, transforms, bounds, materials, warnings | Complete | `SceneManifest` validators and Blender import tests |
| Missing texture, flattened hierarchy, animation, and procedural capability warnings | Complete | scene capability fields and integration fixtures |
| Component find and inspect | Complete | shared service contract and selector tests |
| Six-axis absolute pose and orbit conversion | Complete | camera property tests and Blender state integration |
| Perspective, focal length, sensor fit, orthographic scale | Complete | protocol models, property tests, real Blender checks |
| Basis, FOV, frustum, target depth, focus projected bounds | Complete | `CameraDiagnostics`; pure geometry and Blender integration tests |
| Area, point, spot, sun, RGB, and color temperature | Complete | illumination contracts and Blender application tests |
| Named illumination presets and luminance diagnostics | Complete | preset integration and render manifest tests |
| Content-addressed HDR/EXR environment maps | Complete | cache verification, EXR Blender integration, path-independent state hash test |
| Hide, show, isolate, ghost, select, highlight, and label | Complete | visual state and reset integration test |
| Eevee/Cycles color renders and evaluator-only passes | Complete | color, EXR, component-ID, and highlighted pass tests |
| Worker crash recovery and accepted-state replay | Complete | import and state replay integration tests |
| Focused 3x3 contact sheet | Complete | nine manifested panels, component-mask visibility, iterative blocker peeling, stop metrics, callouts, and legend |
| Custom sheet recipes | Complete | per-panel absolute/orbit cameras, projection, focal length, illumination, fixed-pose studies, and dolly zoom |
| Bounded request budgets | Complete | broker reserves call, render, pixel, and worst-case output bytes before dispatch; overruns are discarded and tested |
| Procedural and curated eval corpus | Complete | 672 public models and 2,528 public episodes; private manifest remains separately generated |
| Deterministic semantic, state, evidence, safety, and budget gates | Complete | grouped predicates must hold in one accepted state-changing snapshot and an accepted render with the identical state hash; private masks enforce target, absence, highlight, contrast, projection, lens, and illumination combinations |
| Metamorphic variants and channel ablations | Complete | all declared variants have relation tests; every required operation, state predicate, grouped render, and evidence combination has a failing ablation |
| Isolated Linux and Windows runners | Complete | namespaces/cgroups on Linux, AppContainer/Job Object on Windows; both CI paths execute Blender |
| Golden image and exact mask regression suite | Complete | `test_render_goldens.py` enforces bounded Delta E and edge drift plus exact semantic component masks |
| Truncated protocol and render-crash recovery tests | Complete | controller rejects truncated worker output; real Blender render-phase crash restarts, replays accepted state, and preserves the source |
| Release-scale threshold proof | Missing | Run a declared agent adapter on the pinned public and private release tiers and publish reports |
| Clean-install smoke report | Complete | CI builds and installs wheels in clean Linux and Windows uv environments; `evals/reports/clean-install-v4/` records a 40/40 pinned smoke run from the built wheel with runtime and artifact hashes |

The completion gate is the conjunction of every row above being complete, the
full Linux and Windows suites passing on the same commit, and the pinned release
evaluation meeting every threshold in `docs/plan.md`.
