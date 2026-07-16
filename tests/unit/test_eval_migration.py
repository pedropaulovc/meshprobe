from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import meshprobe.evals.ergonomics as ergonomics
from meshprobe.evals.ergonomics import ergonomics_report_markdown, select_paired_episodes
from meshprobe.evals.factory import build_corpus, validate_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.migration import (
    codify_opaque_family,
    migrate_corpus_v2,
    restore_schema_v1_source,
)
from meshprobe.evals.schemas import (
    Difficulty,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    ModelSource,
    TaskFamily,
)


def test_v2_migration_preserves_model_and_episode_identity(tmp_path: Path) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v5",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    restore_schema_v1_source(source.root, corpus_version="procedural-v5")
    with pytest.raises(ValueError, match="migrate the corpus to schema version 2"):
        validate_corpus(source.root)

    migrated, audit = migrate_corpus_v2(
        source.root,
        tmp_path,
        corpus_version="procedural-v6",
    )

    assert migrated.model_count == 1
    assert migrated.episode_count == 4
    assert audit.source_version == "procedural-v5"
    assert audit.destination_version == "procedural-v6"
    assert audit.model_hashes_unchanged
    assert audit.episode_ids_unchanged
    assert audit.truth_payloads_unchanged

    existing, repeated_audit = migrate_corpus_v2(
        source.root,
        tmp_path,
        corpus_version="procedural-v6",
    )
    assert existing.root == migrated.root
    assert repeated_audit == audit


def test_private_opaque_family_construction_is_a_production_step(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="private-v7",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(32, 33),
        model_source=ModelSource.PRIVATE,
    )

    codify_opaque_family(corpus.root)
    validated = validate_corpus(corpus.root)
    truth = EpisodeGroundTruth.model_validate_json(
        next((corpus.root / "private" / "ground_truth").glob("*.json")).read_text(encoding="utf-8")
    )

    assert validated.manifest.generator_families == ("hidden_clip", "opaque_family_v7")
    assert truth.generator_family == "opaque_family_v7"


def test_ergonomics_selection_pairs_basic_and_intermediate_episodes(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="selection-v2",
        families=(GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.STAMPED_ARROW),
        seeds=(0,),
        model_source=ModelSource.CURATED,
    )
    selected = select_paired_episodes(corpus.root, per_difficulty=2)
    specs = [
        EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        for episode_id in selected
    ]

    assert len(selected) == 4
    assert [spec.difficulty for spec in specs].count(Difficulty.BASIC) == 2
    assert [spec.difficulty for spec in specs].count(Difficulty.INTERMEDIATE) == 2
    assert all(
        spec.episode_class is EpisodeClass.POSITIVE and spec.model_source is ModelSource.CURATED
        for spec in specs
        if spec.difficulty is Difficulty.BASIC
    )
    assert all(
        spec.episode_class is not EpisodeClass.NEGATIVE
        and spec.family is not TaskFamily.NEGATIVE_AMBIGUOUS
        for spec in specs
        if spec.difficulty is Difficulty.INTERMEDIATE
    )
    prompt = ergonomics._prompt(specs[0], corpus.root / "public" / "models" / specs[0].model_file)
    assert "token limit" not in prompt.casefold()
    assert "problem_understanding" in prompt
    assert "tool_sufficiency" in prompt
    assert "target_component_id" in prompt


def test_ergonomics_selection_rejects_hidden_role_and_ambiguous_tasks(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="eligibility-v2",
        families=(GeneratorFamily.AMBIGUOUS_TWIN,),
        seeds=(0,),
    )
    specs = [
        EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (corpus.root / "public" / "episodes").glob("*.json")
    ]

    assert not any(ergonomics._ergonomics_eligible(spec) for spec in specs)


def test_ergonomics_attempt_persists_exact_prompt(tmp_path: Path) -> None:
    prompt = "inspect this assigned model exactly"
    path = tmp_path / "prompt.txt"

    persisted = ergonomics.prepare_attempt_files(tmp_path, prompt)

    assert persisted.prompt == path
    assert path.read_text(encoding="utf-8") == prompt


def test_ergonomics_markdown_is_explicitly_diagnostic() -> None:
    report = {
        "summary": {
            "claude-opus": {
                "attempts": 2,
                "answer_passes": 1,
                "evidence_passes": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_output_tokens": 0,
                "meshprobe_operations": 5,
                "invalid_calls": 1,
                "elapsed_seconds": 12.5,
            }
        }
    }

    markdown = ergonomics_report_markdown(report)

    assert "| claude-opus | 2 | 1 | 2 | 100 | 20 | 0 | 5 | 1 | 12.5 |" in markdown
    assert "not a release qualification result" in markdown
    assert "Hidden chain of thought was not collected" in markdown


@pytest.mark.skipif(os.name == "nt", reason="ergonomics pilot requires Bubblewrap")
def test_ergonomics_agent_command_uses_bubblewrap_workspace_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(
        ergonomics,
        "_agent_read_only_mounts",
        lambda agent, cli_runtime, input_root: (
            (runtime, ergonomics.ERGONOMICS_RUNTIME),
            (input_root, ergonomics.ERGONOMICS_INPUT),
        ),
    )

    command = ergonomics._sandboxed_agent_command(
        ("/usr/bin/python3", "-c", "print('isolated')"),
        agent=ergonomics.ErgonomicsAgent.CODEX,
        root=tmp_path / "attempt",
        cli_runtime=runtime,
        wall_seconds=10,
        output_bytes=1024,
    )

    assert "bwrap" in " ".join(command)
    assert "--share-net" in command
    assert "/workspace/artifacts" in command
    assert str(ergonomics.ERGONOMICS_RUNTIME) in command
    assert f"--fsize={ergonomics.ERGONOMICS_PROCESS_OUTPUT_LIMIT}:" in " ".join(command)
    cpu = next(argument for argument in command if argument.startswith("--cpu="))
    assert int(cpu.removeprefix("--cpu=").split(":", 1)[0]) >= 600
    nproc = next(argument for argument in command if argument.startswith("--nproc="))
    assert int(nproc.removeprefix("--nproc=").split(":", 1)[0]) >= 512


def test_ergonomics_mounts_complete_blender_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "blender-5.2"
    runtime.mkdir()
    executable = runtime / "blender"
    executable.write_bytes(b"blender")
    shim = tmp_path / "bin" / "blender"
    shim.parent.mkdir()
    shim.symlink_to(executable)
    monkeypatch.setattr(ergonomics.shutil, "which", lambda name: str(shim))

    assert ergonomics._blender_runtime() == runtime


def test_ergonomics_runtime_preflight_checks_mount_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    models = corpus / "public" / "models"
    models.mkdir(parents=True)
    (models / "model.glb").write_bytes(b"model")
    observed: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        ergonomics,
        "_sandboxed_agent_command",
        lambda command, **kwargs: observed.append(command) or ("/bin/true",),
    )

    assert ergonomics._preflight_runtime_boundary(corpus, tmp_path / "run", tmp_path) == (
        "available"
    )
    script = observed[0][-1]
    assert "runtime-proof open" in script
    assert "runtime-proof/state.yml" in script
    assert "pyproject.toml" in script
    assert ".write-probe" in script
    assert "python3 -c" in script
    assert "Path(sys.prefix)" in script
    assert "yaml.safe_load" in script
    assert "Image.open" in script
    assert ".python-batch-probe.json" in script


def test_ergonomics_runtime_installs_shell_discovery_helper(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)

    ergonomics._install_runtime_helpers(runtime)

    helper = runtime / "bin" / "which"
    assert helper.stat().st_mode & 0o111
    assert 'command -v "$name"' in helper.read_text(encoding="utf-8")


def test_ergonomics_time_to_open_uses_acknowledged_event(tmp_path: Path) -> None:
    started = datetime(2026, 7, 16, 18, 0, tzinfo=UTC)
    events = tmp_path / ".meshprobe" / "sessions" / "review" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "at": (started + timedelta(seconds=4.25)).isoformat(),
                "op": "scene.open",
                "status": "accepted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert ergonomics._time_to_open_seconds(tmp_path, started) == 4.25


def test_ergonomics_retries_only_no_progress_provider_failures(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"type":"rate_limit_event"}\n', encoding="utf-8")
    (tmp_path / "stderr.log").write_text("", encoding="utf-8")

    no_progress = SimpleNamespace(
        provider_error="timeout after 180 seconds",
        metrics=SimpleNamespace(meshprobe_operations=0),
        raw_stream_path=str(stream),
    )
    semantic_overrun = SimpleNamespace(
        provider_error="timeout after 180 seconds",
        metrics=SimpleNamespace(meshprobe_operations=4),
        raw_stream_path=str(stream),
    )

    assert ergonomics._retryable_provider_failure(no_progress)
    assert not ergonomics._retryable_provider_failure(semantic_overrun)


def test_ergonomics_live_token_monitor_deduplicates_claude_messages(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl"
    first = {
        "type": "assistant",
        "message": {
            "id": "msg-1",
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
                "output_tokens": 10,
            },
        },
    }
    second = {
        "type": "assistant",
        "message": {
            "id": "msg-2",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 300,
                "output_tokens": 15,
            },
        },
    }
    stream.write_text(
        "\n".join((json.dumps(first), json.dumps(first), json.dumps(second))) + "\n",
        encoding="utf-8",
    )
    monitor = ergonomics._LiveTokenMonitor(ergonomics.ErgonomicsAgent.CLAUDE, stream)

    assert monitor.total() == 650
    assert monitor.usage.input == 625
    assert monitor.usage.uncached_input == 5
    assert monitor.usage.output == 25
    assert monitor.total() == 650


def test_ergonomics_claude_final_usage_overrides_incremental_messages() -> None:
    events = [
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "usage": {
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 200,
                    "output_tokens": 10,
                },
            },
        },
        {
            "type": "result",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 120,
                "cache_read_input_tokens": 250,
                "output_tokens": 40,
            },
        },
    ]

    usage = ergonomics._token_usage(ergonomics.ErgonomicsAgent.CLAUDE, events)

    assert usage.input == 373
    assert usage.uncached_input == 3
    assert usage.cache_creation == 120
    assert usage.cache_read == 250
    assert usage.output == 40


def test_ergonomics_extracts_each_provider_command_once() -> None:
    codex_events = [
        {
            "type": "item.started",
            "item": {"type": "command_execution", "command": "meshprobe list"},
        },
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "meshprobe list"},
        },
    ]
    claude_events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "meshprobe snapshot"},
                    }
                ]
            },
        }
    ]

    assert ergonomics._extract_commands(ergonomics.ErgonomicsAgent.CODEX, codex_events) == [
        "meshprobe list"
    ]
    assert ergonomics._extract_commands(ergonomics.ErgonomicsAgent.CLAUDE, claude_events) == [
        "meshprobe snapshot"
    ]


def test_ergonomics_counts_executables_not_meshprobe_state_paths() -> None:
    command = (
        "/bin/bash -lc 'command -v meshprobe; meshprobe --help; "
        "cat .meshprobe/state.yml; ./.meshprobe-runtime/bin/meshprobe -s demo open model.glb "
        "&& timeout 120 meshprobe -s demo render-sheet c1'"
    )

    assert ergonomics._meshprobe_calls(command) == (
        "--help",
        "-s demo open model.glb",
        "-s demo render-sheet c1",
    )


def test_ergonomics_counts_actual_session_events(tmp_path: Path) -> None:
    events = tmp_path / ".meshprobe" / "sessions" / "demo" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            (
                '{"op":"scene.open","status":"accepted"}',
                "not-json",
                '{"op":"component.find","status":"accepted"}',
                '{"op":"session.close","status":"accepted"}',
            )
        ),
        encoding="utf-8",
    )

    assert ergonomics._session_operations(tmp_path) == (
        "scene.open",
        "component.find",
        "session.close",
    )


def test_ergonomics_preflight_probes_exact_models(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        output = (
            "MESHPROBE_PREFLIGHT_OK"
            if command[-1] == ergonomics.MODEL_PREFLIGHT_PROMPT
            else "authenticated"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(ergonomics.subprocess, "run", run)

    result = ergonomics.preflight_agents()

    assert result["claude_model"] == "available"
    assert result["codex_model"] == "available"
    assert commands[-2][0:4] == ("claude", "-p", "--model", "opus")
    assert "--allowedTools=Bash,Read" in commands[-2]
    assert commands[-1][0:5] == (
        "codex",
        "exec",
        "--model",
        "gpt-5.6-luna",
        "--json",
    )
    assert "danger-full-access" in commands[-1]


def test_ergonomics_preflight_surfaces_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        if command[:4] == ("codex", "exec", "--model", "gpt-5.6-luna"):
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="The 'gpt-5.6-luna' model is not supported with a ChatGPT account.",
            )
        output = (
            "MESHPROBE_PREFLIGHT_OK"
            if command[-1] == ergonomics.MODEL_PREFLIGHT_PROMPT
            else "authenticated"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(ergonomics.subprocess, "run", run)

    with pytest.raises(RuntimeError, match=r"luna.*not supported"):
        ergonomics.preflight_agents()
