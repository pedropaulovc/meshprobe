"""Paired diagnostic evaluation of agent-facing CLI ergonomics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from meshprobe.client import MeshProbeClient
from meshprobe.evals.schemas import Difficulty, EpisodeGroundTruth, EpisodeSpec
from meshprobe.workspace import atomic_json, atomic_text


class ErgonomicsAgent(StrEnum):
    CLAUDE = "claude-opus"
    CODEX = "codex-luna"


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning_output: int = 0
    cache_creation: int = 0
    cache_read: int = 0


class ErgonomicsMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_gate: bool
    evidence_gate: bool
    structured_output: bool
    elapsed_seconds: float
    time_to_open_seconds: float | None = None
    tokens: TokenUsage
    commands: tuple[str, ...]
    help_calls: int
    invalid_calls: int
    retries: int
    short_ref_uses: int
    rg_calls: int
    jq_calls: int
    yq_calls: int
    bytes_read: int
    raw_reads: int
    full_file_reads: int
    redundant_calls: int
    meshprobe_operations: int
    renders: int
    receipt_path_uses: int


class ErgonomicsAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    episode_id: str
    difficulty: Difficulty
    agent: ErgonomicsAgent
    attempt: int
    command: tuple[str, ...]
    return_code: int
    provider_error: str | None = None
    final: dict[str, Any] | None = None
    metrics: ErgonomicsMetrics
    raw_stream_path: str


@dataclass(frozen=True)
class ErgonomicsRun:
    root: Path
    attempts: tuple[ErgonomicsAttempt, ...]
    report_path: Path
    report_markdown_path: Path


AGENT_COMMANDS: dict[ErgonomicsAgent, tuple[str, ...]] = {
    ErgonomicsAgent.CLAUDE: (
        "claude",
        "-p",
        "--model",
        "opus",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--safe-mode",
    ),
    ErgonomicsAgent.CODEX: (
        "codex",
        "exec",
        "--model",
        "luna",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
    ),
}


def select_paired_episodes(corpus_root: Path, *, per_difficulty: int = 12) -> tuple[str, ...]:
    manifest = _read_json(corpus_root / "public" / "manifest.json")
    by_difficulty: dict[Difficulty, list[EpisodeSpec]] = {
        Difficulty.BASIC: [],
        Difficulty.INTERMEDIATE: [],
    }
    for episode_id in manifest["episodes"]:
        spec = EpisodeSpec.model_validate_json(
            (corpus_root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        if spec.difficulty in by_difficulty:
            by_difficulty[spec.difficulty].append(spec)
    selected: list[str] = []
    for difficulty in (Difficulty.BASIC, Difficulty.INTERMEDIATE):
        ranked = sorted(
            by_difficulty[difficulty],
            key=lambda spec: (
                _family_occurrence(by_difficulty[difficulty], spec),
                _rank(spec.episode_id),
            ),
        )
        chosen = _round_robin_families(ranked, per_difficulty)
        if len(chosen) != per_difficulty:
            raise ValueError(f"corpus has fewer than {per_difficulty} {difficulty} episodes")
        selected.extend(spec.episode_id for spec in chosen)
    return tuple(selected)


def preflight_agents() -> dict[str, str]:
    checks = {
        "claude_version": ("claude", "--version"),
        "claude_auth": ("claude", "auth", "status"),
        "codex_version": ("codex", "--version"),
        "codex_auth": ("codex", "login", "status"),
    }
    results: dict[str, str] = {}
    for name, command in checks.items():
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed ({completed.returncode}): {output}")
        results[name] = output
    return results


def run_ergonomics_pilot(
    corpus_root: Path,
    output_root: Path,
    *,
    per_difficulty: int = 12,
    canary_pairs: int = 4,
) -> ErgonomicsRun:
    corpus_root = corpus_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = preflight_agents()
    atomic_json(output_root / "preflight.json", preflight)
    episodes = select_paired_episodes(corpus_root, per_difficulty=per_difficulty)
    attempts = _load_attempts(output_root)
    for index, episode_id in enumerate(episodes):
        order = (
            (ErgonomicsAgent.CLAUDE, ErgonomicsAgent.CODEX)
            if index % 2 == 0
            else (ErgonomicsAgent.CODEX, ErgonomicsAgent.CLAUDE)
        )
        for agent in order:
            key = (episode_id, agent)
            if key in {(attempt.episode_id, attempt.agent) for attempt in attempts}:
                continue
            attempt = _run_with_retry(corpus_root, output_root, episode_id, agent)
            attempts.append(attempt)
            _checkpoint(output_root, attempts, preflight)
        if index + 1 == canary_pairs:
            canary = attempts[: canary_pairs * 2]
            if sum(attempt.metrics.structured_output for attempt in canary) < len(canary) // 2:
                raise RuntimeError(
                    "ergonomics canary stopped: fewer than half of attempts "
                    "produced structured output"
                )
    report_path = _checkpoint(output_root, attempts, preflight)
    return ErgonomicsRun(
        root=output_root,
        attempts=tuple(attempts),
        report_path=report_path,
        report_markdown_path=report_path.with_suffix(".md"),
    )


def _run_with_retry(
    corpus_root: Path,
    output_root: Path,
    episode_id: str,
    agent: ErgonomicsAgent,
) -> ErgonomicsAttempt:
    first = _run_attempt(corpus_root, output_root, episode_id, agent, attempt=1)
    if first.provider_error is None:
        return first
    second = _run_attempt(corpus_root, output_root, episode_id, agent, attempt=2)
    metrics = second.metrics.model_copy(update={"retries": 1})
    return second.model_copy(update={"metrics": metrics})


def _run_attempt(
    corpus_root: Path,
    output_root: Path,
    episode_id: str,
    agent: ErgonomicsAgent,
    *,
    attempt: int,
) -> ErgonomicsAttempt:
    spec = EpisodeSpec.model_validate_json(
        (corpus_root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
    )
    truth = EpisodeGroundTruth.model_validate_json(
        (corpus_root / "private" / "ground_truth" / f"{episode_id}.json").read_text(
            encoding="utf-8"
        )
    )
    root = output_root / "attempts" / episode_id / agent / f"attempt-{attempt}"
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(spec, corpus_root / "public" / "models" / spec.model_file)
    command = (*AGENT_COMMANDS[agent], prompt)
    raw_path = root / "stream.jsonl"
    stderr_path = root / "stderr.log"
    started = time.monotonic()
    provider_error: str | None = None
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with (
        raw_path.open("w", encoding="utf-8") as raw_stream,
        stderr_path.open("w", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.Popen(
            command,
            cwd=workspace,
            stdout=raw_stream,
            stderr=stderr_stream,
            text=True,
            env={**os.environ, "NO_COLOR": "1"},
            start_new_session=os.name != "nt",
            creationflags=creation_flags,
        )
        try:
            return_code = process.wait(timeout=spec.budgets.wall_seconds)
        except subprocess.TimeoutExpired:
            return_code = 124
            provider_error = f"timeout after {spec.budgets.wall_seconds:g} seconds"
            _terminate_agent(process)
        except BaseException:
            _terminate_agent(process)
            raise
        finally:
            _stop_workspace(workspace)
    elapsed = time.monotonic() - started
    raw = raw_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    if return_code != 0 and provider_error is None:
        provider_error = (stderr or raw)[-2000:] or f"agent exited {return_code}"
    events = _json_events(raw)
    final = _extract_final(raw, events)
    commands = tuple(_extract_commands(events))
    tokens = _token_usage(agent, events)
    answer_gate = final is not None and final.get("answer") == truth.answer.model_dump(mode="json")
    evidence = final.get("evidence_manifest_paths", []) if final else []
    evidence_gate = not truth.evidence_requirements or bool(evidence)
    metrics = _metrics(
        answer_gate=answer_gate,
        evidence_gate=evidence_gate,
        final=final,
        elapsed=elapsed,
        tokens=tokens,
        commands=commands,
        raw=raw,
        workspace=workspace,
    )
    return ErgonomicsAttempt(
        episode_id=episode_id,
        difficulty=spec.difficulty,
        agent=agent,
        attempt=attempt,
        command=command[:-1],
        return_code=return_code,
        provider_error=provider_error,
        final=final,
        metrics=metrics,
        raw_stream_path=str(raw_path),
    )


def _stop_workspace(workspace: Path) -> None:
    client = MeshProbeClient(workspace)
    with suppress(OSError, RuntimeError, TimeoutError, ValueError):
        client.kill_all()


def _terminate_agent(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _prompt(spec: EpisodeSpec, model_path: Path) -> str:
    schema = {
        "answer": {
            "status": "answered | indeterminate",
            "values": {},
            "conflicting_component_ids": [],
            "missing_capabilities": [],
        },
        "evidence_manifest_paths": [],
    }
    return (
        "Use the meshprobe CLI to inspect the read-only 3D model. Work only in the current "
        "directory. Start by opening the model in a named session, use compact receipts and "
        ".meshprobe state files, and gather evidence when the task requires it. "
        f"Model: {model_path}. Task: {spec.prompt} "
        "End with exactly one line beginning MESHPROBE_RESULT= followed by JSON matching: "
        f"{json.dumps(schema, separators=(',', ':'))}"
    )


def _metrics(
    *,
    answer_gate: bool,
    evidence_gate: bool,
    final: dict[str, Any] | None,
    elapsed: float,
    tokens: TokenUsage,
    commands: tuple[str, ...],
    raw: str,
    workspace: Path,
) -> ErgonomicsMetrics:
    joined = "\n".join(commands)
    meshprobe = [command for command in commands if re.search(r"\bmeshprobe\b", command)]
    first_open = next(
        (index for index, command in enumerate(commands) if " open " in command), None
    )
    normalized = [re.sub(r"\s+", " ", command.strip()) for command in meshprobe]
    return ErgonomicsMetrics(
        answer_gate=answer_gate,
        evidence_gate=evidence_gate,
        structured_output=final is not None,
        elapsed_seconds=elapsed,
        time_to_open_seconds=(
            elapsed * (first_open + 1) / len(commands)
            if first_open is not None and commands
            else None
        ),
        tokens=tokens,
        commands=commands,
        help_calls=len(re.findall(r"(?:--help|\bhelp\b)", joined)),
        invalid_calls=_count_nonzero_exits(raw),
        retries=0,
        short_ref_uses=len(re.findall(r"\bc\d+\b", joined)),
        rg_calls=len(re.findall(r"(?:^|[;&|]\s*)rg\b", joined, re.MULTILINE)),
        jq_calls=len(re.findall(r"(?:^|[;&|]\s*)jq\b", joined, re.MULTILINE)),
        yq_calls=len(re.findall(r"(?:^|[;&|]\s*)yq\b", joined, re.MULTILINE)),
        bytes_read=_bytes_read(commands, workspace),
        raw_reads=len(re.findall(r"meshprobe[^\n]*--raw", joined)),
        full_file_reads=len(re.findall(r"\b(?:cat|less|head|tail)\s+[^|;&]+", joined)),
        redundant_calls=max(0, len(normalized) - len(set(normalized))),
        meshprobe_operations=len(meshprobe),
        renders=sum(
            "render-image" in command or "render-sheet" in command for command in meshprobe
        ),
        receipt_path_uses=len(re.findall(r"\.meshprobe/[^\s'\"]+", joined)),
    )


def _checkpoint(
    output_root: Path,
    attempts: list[ErgonomicsAttempt],
    preflight: dict[str, str],
) -> Path:
    report = {
        "schema_version": 1,
        "pilot": "ergonomics-v1",
        "qualification": False,
        "hidden_chain_of_thought_analyzed": False,
        "preflight": preflight,
        "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        "summary": _summary(attempts),
    }
    path = output_root / "ergonomics-report.json"
    atomic_json(path, report)
    atomic_text(path.with_suffix(".md"), ergonomics_report_markdown(report))
    return path


def _summary(attempts: list[ErgonomicsAttempt]) -> dict[str, Any]:
    by_agent: dict[str, dict[str, float | int]] = {}
    for agent in ErgonomicsAgent:
        selected = [attempt for attempt in attempts if attempt.agent is agent]
        by_agent[agent] = {
            "attempts": len(selected),
            "answer_passes": sum(item.metrics.answer_gate for item in selected),
            "evidence_passes": sum(item.metrics.evidence_gate for item in selected),
            "input_tokens": sum(item.metrics.tokens.input for item in selected),
            "output_tokens": sum(item.metrics.tokens.output for item in selected),
            "reasoning_output_tokens": sum(
                item.metrics.tokens.reasoning_output for item in selected
            ),
            "elapsed_seconds": sum(item.metrics.elapsed_seconds for item in selected),
            "meshprobe_operations": sum(item.metrics.meshprobe_operations for item in selected),
            "help_calls": sum(item.metrics.help_calls for item in selected),
            "invalid_calls": sum(item.metrics.invalid_calls for item in selected),
            "retries": sum(item.metrics.retries for item in selected),
            "short_ref_uses": sum(item.metrics.short_ref_uses for item in selected),
            "rg_calls": sum(item.metrics.rg_calls for item in selected),
            "jq_calls": sum(item.metrics.jq_calls for item in selected),
            "yq_calls": sum(item.metrics.yq_calls for item in selected),
            "bytes_read": sum(item.metrics.bytes_read for item in selected),
            "raw_reads": sum(item.metrics.raw_reads for item in selected),
            "full_file_reads": sum(item.metrics.full_file_reads for item in selected),
            "redundant_calls": sum(item.metrics.redundant_calls for item in selected),
            "renders": sum(item.metrics.renders for item in selected),
            "receipt_path_uses": sum(item.metrics.receipt_path_uses for item in selected),
        }
    return by_agent


def ergonomics_report_markdown(report: dict[str, Any]) -> str:
    """Render the concise, non-qualification pilot report."""

    summary = report.get("summary", {})
    lines = [
        "# MeshProbe CLI ergonomics pilot",
        "",
        "This diagnostic compares Claude Opus and Codex Luna on paired basic and "
        "intermediate episodes. It is not a release qualification result.",
        "",
        "| Agent | Attempts | Answer passes | Evidence passes | Input tokens | "
        "Output tokens | Reasoning tokens | CLI operations | Invalid calls | Elapsed (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for agent in ErgonomicsAgent:
        values = summary.get(agent, {})
        lines.append(
            f"| {agent} | {values.get('attempts', 0)} | {values.get('answer_passes', 0)} | "
            f"{values.get('evidence_passes', 0)} | {values.get('input_tokens', 0)} | "
            f"{values.get('output_tokens', 0)} | "
            f"{values.get('reasoning_output_tokens', 0)} | "
            f"{values.get('meshprobe_operations', 0)} | {values.get('invalid_calls', 0)} | "
            f"{float(values.get('elapsed_seconds', 0)):.1f} |"
        )
    lines.extend(
        (
            "",
            "The analysis uses exposed reasoning summaries and command trajectories only. "
            "Hidden chain of thought was not collected or analyzed.",
            "",
        )
    )
    return "\n".join(lines)


def _json_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _extract_final(raw: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [raw]
    candidates.extend(_event_strings(events))
    pattern = re.compile(r"MESHPROBE_RESULT=(\{.*\})")
    for candidate in reversed(candidates):
        match = pattern.search(candidate)
        if match is None:
            continue
        try:
            result = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return cast(dict[str, Any], result)
    return None


def _event_strings(events: list[dict[str, Any]]) -> list[str]:
    strings: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)
            return
        if isinstance(value, str):
            strings.append(value)

    visit(events)
    return strings


def _extract_commands(events: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if isinstance(value, str) and key in {"command", "cmd"}:
            commands.append(value)

    visit(events)
    return commands


def _token_usage(agent: ErgonomicsAgent, events: list[dict[str, Any]]) -> TokenUsage:
    values = {
        "input": 0,
        "cached_input": 0,
        "output": 0,
        "reasoning_output": 0,
        "cache_creation": 0,
        "cache_read": 0,
    }
    aliases = {
        "input_tokens": "input",
        "cached_input_tokens": "cached_input",
        "output_tokens": "output",
        "reasoning_output_tokens": "reasoning_output",
        "cache_creation_input_tokens": "cache_creation",
        "cache_read_input_tokens": "cache_read",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                target = aliases.get(key)
                if target is not None and isinstance(child, int):
                    values[target] = max(values[target], child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(events)
    if agent is ErgonomicsAgent.CODEX and values["cached_input"] > values["input"]:
        values["input"] = values["cached_input"]
    return TokenUsage(**values)


def _count_nonzero_exits(raw: str) -> int:
    return len(re.findall(r'"(?:exit_code|exitCode)"\s*:\s*[1-9]\d*', raw))


def _bytes_read(commands: tuple[str, ...], workspace: Path) -> int:
    paths: set[Path] = set()
    for command in commands:
        for match in re.findall(r"(?:^|\s)(\.meshprobe/[^\s'\"|;&]+)", command):
            path = workspace / match
            if path.is_file():
                paths.add(path)
    return sum(path.stat().st_size for path in paths)


def _load_attempts(output_root: Path) -> list[ErgonomicsAttempt]:
    report = output_root / "ergonomics-report.json"
    if not report.is_file():
        return []
    payload = _read_json(report)
    return [ErgonomicsAttempt.model_validate(item) for item in payload.get("attempts", [])]


def _round_robin_families(specs: list[EpisodeSpec], limit: int) -> list[EpisodeSpec]:
    buckets: dict[str, list[EpisodeSpec]] = {}
    for spec in specs:
        buckets.setdefault(spec.family.value, []).append(spec)
    selected: list[EpisodeSpec] = []
    while len(selected) < limit and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family]:
                selected.append(buckets[family].pop(0))
            if len(selected) == limit:
                break
    return selected


def _family_occurrence(specs: list[EpisodeSpec], selected: EpisodeSpec) -> int:
    return sum(spec.family is selected.family for spec in specs)


def _rank(episode_id: str) -> str:
    return hashlib.sha256(f"meshprobe-ergonomics-v1\0{episode_id}".encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
