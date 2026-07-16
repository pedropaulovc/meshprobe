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
from meshprobe.evals.schemas import Difficulty, EpisodeGroundTruth, EpisodeSpec, ModelSource


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
    prompt = ergonomics._prompt(specs[0], corpus.root / "public" / "models" / specs[0].model_file)
    assert "token limit" not in prompt.casefold()


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
    assert monitor.usage.input == 5
    assert monitor.usage.output == 25
    assert monitor.total() == 650


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
    assert "--allowedTools=Bash" in commands[-2]
    assert commands[-1][0:5] == (
        "codex",
        "exec",
        "--model",
        "gpt-5.6-luna",
        "--json",
    )


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
