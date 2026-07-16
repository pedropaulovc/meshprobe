# Stateful CLI and evaluation migration plan

## Goal

MeshProbe should behave like Playwright CLI for 3D inspection: an agent opens a named
session once, applies small operations to it, and queries compact durable state instead of
reloading a scene or consuming a full response on every turn. MeshProbe owns this state in a
project-local `.meshprobe` directory. Agents can inspect it efficiently with `rg`, `jq`, and
`yq`.

## CLI surface

The flat command set is:

- `open`, `snapshot`, `find`, and `inspect`
- `view-set`, `view-orbit`, and `illumination-set`
- `display`, `mark`, `render-image`, `render-sheet`, and `reset`
- `list`, `close [--all]`, `kill [--all]`, and `delete-data`

Global options select the named session and workspace and control output:
`-s/--session`, `--workspace`, `--json`, and `--raw`.

`close` gracefully checkpoints and stops the selected renderer. `close --all` does this for
every session and stops the daemon. `kill` immediately terminates the selected renderer and
leaves its last acknowledged checkpoint available for recovery. `kill --all` forcibly stops
the daemon and its renderer process tree. `delete-data` removes `.meshprobe` only after the
daemon is stopped.

Components receive deterministic short references (`c1`, `c2`, and so on) in sorted path
order. Commands accept a short reference, a full stable component ID, or an exact component
path.

## Durable state

Each workspace has one authenticated loopback daemon. Each active session has one Blender
worker. The filesystem contract is:

```text
.meshprobe/
  daemon.json
  daemon.log
  sessions/<name>/
    metadata.json
    scene.json
    components.yml
    state.yml
    checkpoint.json
    events.jsonl
    results/
    snapshots/
    artifacts/
```

`components.yml` is a compact static reference/path/name index. `state.yml` contains exact
camera and illumination values plus component display/mark defaults and sparse overrides.
`scene.json` retains the complete manifest for `jq`. `events.jsonl` records accepted and
rejected operations. Normal stdout is a small receipt pointing to these files; `--json`
prints that receipt as JSON and `--raw` prints the operation result.

The daemon uses newline-delimited JSON over loopback with a per-workspace token, startup
locking, and stale-process recovery. The durable checkpoint stores controller-normalized
accepted state commands. A restarted daemon reopens the source, verifies its hash, and
replays only acknowledged commands. `MeshProbeClient.execute` is the shared client boundary;
CLI and MCP use it while qualification evaluation can inject an in-process `SessionManager`.

## Evaluation dataset migration

Runtime and evaluation protocol version 2 replaces `scene.describe` with
`session.snapshot`. `EVALUATED_OPERATIONS` explicitly excludes administrative lifecycle
commands.

Publish these versioned corpora without changing geometry or task identity:

- `procedural-v6`
- `curated-tasks-v6`, still based on curated geometry build `curated-v2`
- `qualification-v6`, with the same model and episode counts
- `private-v7`, with the same counts and a production `opaque_family_v7` builder that
  codifies the current private-family construction

The migration audit must prove that model IDs, model hashes, episode IDs, and truth payloads
are unchanged; only the schema/protocol version and operation rename may differ. Full
investigations may contain evaluated operations only. Public and private tier manifests are
repinned to protocol 2 and the final package hash. Existing corpora and reports remain
available. A clean-install v6 report is committed, while full public/private qualification is
deferred.

Version 1 evaluation inputs are rejected clearly rather than translated at runtime.

## CLI ergonomics pilot

The diagnostic pilot is separate from qualification and runs through `meshprobe eval
ergonomics`. It uses 24 paired public episodes: 12 basic and 12 intermediate, deterministically
stratified. Claude and Codex receive the same minimal prompt, a fresh workspace/session for
each attempt, alternating run order, and sequential Blender access. Every attempt is
checkpointed.

The required agent commands are:

```text
claude -p --model opus --output-format stream-json --verbose \
  --no-session-persistence --safe-mode --allowedTools=Bash
codex exec --model gpt-5.6-luna --json --ephemeral --ignore-user-config --ignore-rules
```

Preflight checks versions, authentication, and model availability with no model fallback.
It also opens and closes a real public model inside the final sandbox layout, proves the
assigned model is read-only, and proves the repository path is absent.
Run a four-pair canary before the remaining twenty. Infrastructure or provider failures may
be retried once; semantic failures are not retried. The harness silently enforces a
256,000-token ceiling per attempt from the live provider usage stream, alongside the episode
wall limit; prompts do not disclose the ceiling to the evaluated agent.
The process sandbox also caps any single file at 1 GiB. Episode evidence budgets remain
separate because Blender's internal startup files are not submitted evidence. The pilot
allows at most 512 processes and threads so Codex and Blender can coexist without removing
the runaway-process guard. The per-process CPU ceiling scales with host core count; the wall
deadline remains authoritative for multithreaded Blender renders.

On Linux, each agent runs inside Bubblewrap with its own PID and mount namespaces. The
sandbox shares the host network for provider access but exposes only the assigned public
model, the agent credential file, a clean wheel-installed MeshProbe runtime, and the writable
attempt directory. The selected Blender distribution is mounted read-only as a runtime
dependency. The repository, corpus root, private truth, and prior attempts are not mounted.
Codex's inner sandbox is disabled because `workspace-write` blocks the loopback session
daemon; the outer Bubblewrap boundary remains authoritative. Raw provider streams and stderr
remain readable on the host at
`attempts/<episode>/<agent>/attempt-<n>/{stream.jsonl,stderr.log}` while a run is in progress.
The exact evaluated prompt is persisted beside them as `prompt.txt` and referenced from the
attempt record in the checkpoint report.

Windows ergonomics runs remain disabled until the pilot supports the same boundary through
the existing AppContainer launcher, including provider network access, credential-file
projection, and a writable attempt directory. There is no unsandboxed fallback.

These audit controls apply to qualification too. Every external CLI, MCP, reference, and
ergonomics agent uses the platform sandbox and writes an exact `prompt.txt`, live
`stream.jsonl`, and live `stderr.log`. Qualification stores them under each episode's
`evaluator/agent-run`; ergonomics stores them directly under the numbered attempt. Private
truth and evaluator passes remain outside the agent mount namespace.

Metrics cover correctness and evidence gates, time and tokens to open, help and invalid
calls, retries, reference use, `rg`/`jq`/`yq` use, bytes read, full raw reads, redundant calls,
operations, renders, elapsed time, receipt comprehension, and final-answer quality. Claude
token accounting includes input, output, cache creation, and cache reads. Codex accounting
includes input, cached input, output, and reasoning-output tokens. Analysis is limited to
exposed reasoning summaries and command trajectories; it does not claim access to hidden
chain of thought.

Raw attempts live under ignored `.runs/ergonomics-v1`. A concise JSON and Markdown report is
committed under `evals/reports/ergonomics-v1`. The pilot may inform CLI fixes, but it is not a
release qualification result.

## Delivery and verification

Verification includes unit and daemon integration tests, a real Blender crash/recovery test,
CLI/MCP parity, explicit rejection of v1 eval inputs, migration audits, formatting, lint,
strict mypy, full pytest and coverage, a clean wheel install, and CI. The PR becomes ready at
code complete so review and verification overlap. It can merge after CI and reviews are
clean; the deferred full qualification run is not a merge gate.
