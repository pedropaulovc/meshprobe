from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import meshprobe.evals.ergonomics as ergonomics
from meshprobe.evals.ergonomics import ergonomics_report_markdown, select_paired_episodes
from meshprobe.evals.factory import build_corpus, validate_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.harness.attempts import prepare_attempt_files
from meshprobe.evals.migration import (
    audit_migration,
    codify_opaque_family,
    migrate_corpus_v3,
    restore_schema_v1_source,
)
from meshprobe.evals.schemas import (
    Difficulty,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    ModelSource,
    Operation,
    StructuredAnswer,
    TaskFamily,
)
from meshprobe.sources import sha256_file


def set_schema_v2_contract(
    corpus_root: Path,
    *,
    spec_missing: tuple[str, ...],
    truth_missing: tuple[str, ...] | None = None,
) -> None:
    truth_missing = spec_missing if truth_missing is None else truth_missing
    episode_root = corpus_root / "public" / "episodes"
    for path in episode_root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        required = payload.get("required_operations")
        if isinstance(required, list):
            payload["required_operations"] = [
                operation for operation in required if operation not in spec_missing
            ]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (corpus_root / "private" / "ground_truth").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        required = payload.get("required_operations")
        if isinstance(required, list):
            payload["required_operations"] = [
                operation for operation in required if operation not in truth_missing
            ]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = corpus_root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest["episode_sha256"] = {
        episode_id: sha256_file(episode_root / f"{episode_id}.json")
        for episode_id in manifest["episodes"]
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_v3_migration_preserves_model_and_episode_identity(tmp_path: Path) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v5",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    restore_schema_v1_source(source.root, corpus_version="procedural-v5")
    with pytest.raises(ValueError, match="migrate the corpus to schema version 3"):
        validate_corpus(source.root)

    output_root = tmp_path / "brand-new" / "migrations"
    migrated, audit = migrate_corpus_v3(
        source.root,
        output_root,
        corpus_version="procedural-v6",
    )

    assert migrated.model_count == 1
    assert migrated.episode_count == 4
    assert audit.source_version == "procedural-v5"
    assert audit.destination_version == "procedural-v6"
    assert audit.model_hashes_unchanged
    assert audit.episode_ids_unchanged
    assert audit.truth_payloads_unchanged
    full_spec = next(
        EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (migrated.root / "public" / "episodes").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["family"] == "full_investigation"
    )
    full_truth = EpisodeGroundTruth.model_validate_json(
        (migrated.root / "private" / "ground_truth" / f"{full_spec.episode_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert Operation.VIEW_ROTATE in full_spec.required_operations
    assert Operation.COMPONENT_OCCLUSION in full_spec.required_operations
    assert full_truth.required_operations == full_spec.required_operations

    existing, repeated_audit = migrate_corpus_v3(
        source.root,
        output_root,
        corpus_version="procedural-v6",
    )
    assert existing.root == migrated.root
    assert repeated_audit == audit


@pytest.mark.parametrize(
    "missing_operations",
    [
        ("component.occlusion",),
        ("component.occlusion", "view.rotate"),
        ("component.occlusion", "view.move", "view.rotate"),
    ],
)
def test_v3_migration_accepts_explicit_schema_v2_source(
    tmp_path: Path,
    missing_operations: tuple[str, ...],
) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    set_schema_v2_contract(source.root, spec_missing=missing_operations)

    migrated, audit = migrate_corpus_v3(
        source.root,
        tmp_path / "migrated",
        corpus_version="procedural-v3",
    )

    assert audit.full_investigations_valid
    assert migrated.manifest.schema_version == 3
    full_spec = next(
        EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (migrated.root / "public" / "episodes").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["family"] == "full_investigation"
    )
    assert "component.occlusion" in full_spec.required_operations
    assert "view.move" in full_spec.required_operations
    assert "view.rotate" in full_spec.required_operations


@pytest.mark.parametrize(
    ("spec_missing", "truth_missing", "message"),
    [
        (
            ("component.occlusion", "view.move"),
            None,
            "not valid for schema 2",
        ),
        (
            ("component.occlusion", "component.find"),
            None,
            "not valid for schema 2",
        ),
        (
            ("component.occlusion",),
            ("component.occlusion", "view.rotate"),
            "operation sets do not match",
        ),
    ],
)
def test_v3_migration_rejects_invalid_schema_v2_operation_contracts(
    tmp_path: Path,
    spec_missing: tuple[str, ...],
    truth_missing: tuple[str, ...] | None,
    message: str,
) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    set_schema_v2_contract(
        source.root,
        spec_missing=spec_missing,
        truth_missing=truth_missing,
    )

    with pytest.raises(ValueError, match=message):
        migrate_corpus_v3(
            source.root,
            tmp_path / "migrated",
            corpus_version="procedural-v3",
        )


def test_v3_migration_rejects_payload_schema_mismatch(tmp_path: Path) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    set_schema_v2_contract(
        source.root,
        spec_missing=("component.occlusion",),
    )
    episode_path = next((source.root / "public" / "episodes").glob("*.json"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["schema_version"] = 1
    episode_path.write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema does not match"):
        migrate_corpus_v3(
            source.root,
            tmp_path / "migrated",
            corpus_version="procedural-v3",
        )


def test_v3_migration_preserves_historical_operation_set_on_non_full_episode(
    tmp_path: Path,
) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    set_schema_v2_contract(
        source.root,
        spec_missing=("component.occlusion", "view.move", "view.rotate"),
    )
    episode_root = source.root / "public" / "episodes"
    full_path = next(
        path
        for path in episode_root.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["family"] == "full_investigation"
    )
    historical_operations = json.loads(full_path.read_text(encoding="utf-8"))["required_operations"]
    non_full_path = next(
        path
        for path in episode_root.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["family"] != "full_investigation"
    )
    non_full = json.loads(non_full_path.read_text(encoding="utf-8"))
    non_full["required_operations"] = historical_operations
    non_full_path.write_text(
        json.dumps(non_full, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    truth_path = source.root / "private" / "ground_truth" / non_full_path.name
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["required_operations"] = historical_operations
    truth_path.write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = source.root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode_sha256"][non_full_path.stem] = sha256_file(non_full_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    migrated, _ = migrate_corpus_v3(
        source.root,
        tmp_path / "migrated",
        corpus_version="procedural-v3",
    )

    migrated_spec = json.loads(
        (migrated.root / "public" / "episodes" / non_full_path.name).read_text(encoding="utf-8")
    )
    migrated_truth = json.loads(
        (migrated.root / "private" / "ground_truth" / non_full_path.name).read_text(
            encoding="utf-8"
        )
    )
    assert migrated_spec["required_operations"] == historical_operations
    assert migrated_truth["required_operations"] == historical_operations


def test_v3_migration_audit_rejects_incomplete_private_operation_contract(
    tmp_path: Path,
) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    restore_schema_v1_source(source.root, corpus_version="procedural-v2")
    migrated, _ = migrate_corpus_v3(
        source.root,
        tmp_path / "migrated",
        corpus_version="procedural-v3",
    )
    full_spec_path = next(
        path
        for path in (migrated.root / "public" / "episodes").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["family"] == "full_investigation"
    )
    full_truth_path = migrated.root / "private" / "ground_truth" / full_spec_path.name
    truth = json.loads(full_truth_path.read_text(encoding="utf-8"))
    truth["required_operations"].remove("component.occlusion")
    full_truth_path.write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid full-investigation operation set"):
        audit_migration(source.root, migrated.root)


@pytest.mark.parametrize("tamper", ["schema", "episode_hash"])
def test_v3_migration_audit_validates_destination_corpus(
    tmp_path: Path,
    tamper: str,
) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    restore_schema_v1_source(source.root, corpus_version="procedural-v2")
    migrated, _ = migrate_corpus_v3(
        source.root,
        tmp_path / "migrated",
        corpus_version="procedural-v3",
    )
    manifest_path = migrated.root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "schema":
        manifest["schema_version"] = 2
    else:
        first_episode = manifest["episodes"][0]
        manifest["episode_sha256"][first_episode] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected = "migrate the corpus" if tamper == "schema" else "episode hash mismatch"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        audit_migration(source.root, migrated.root)


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
    for path in (corpus.root / "public" / "episodes").glob("*.json"):
        spec = EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        if spec.difficulty is not Difficulty.BASIC:
            continue
        neutral_file = f"{Path(spec.model_file).stem}--rigid_transform.glb"
        path.write_text(
            spec.model_copy(update={"model_file": neutral_file}).model_dump_json(indent=2),
            encoding="utf-8",
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


def test_ergonomics_basic_selection_uses_neutral_curated_variants(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="neutral-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
        model_source=ModelSource.CURATED,
    )
    specs = [
        EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (corpus.root / "public" / "episodes").glob("*.json")
    ]
    basic = next(spec for spec in specs if spec.difficulty is Difficulty.BASIC)

    assert ergonomics._ergonomics_eligible(
        basic.model_copy(update={"model_file": "assembly--rigid_transform.glb"})
    )
    assert not ergonomics._ergonomics_eligible(
        basic.model_copy(update={"model_file": "assembly--extra_occluder.glb"})
    )


def test_ergonomics_answer_gate_applies_public_defaults() -> None:
    truth = StructuredAnswer.model_validate(
        {
            "status": "answered",
            "values": {"target_component_id": "cmp_target"},
        }
    )
    final = {
        "answer": {
            "status": "answered",
            "values": {"target_component_id": "cmp_target"},
        }
    }

    assert ergonomics._answer_matches(final, truth)


def test_ergonomics_attempt_persists_exact_prompt(tmp_path: Path) -> None:
    prompt = "inspect this assigned model exactly"
    path = tmp_path / "prompt.txt"

    persisted = prepare_attempt_files(tmp_path, prompt)

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
    monkeypatch.setattr(shutil, "which", lambda name: str(shim))

    assert ergonomics._blender_runtime() == runtime


@pytest.mark.skipif(os.name == "nt", reason="ergonomics runtime uses Bubblewrap paths")
def test_ergonomics_runtime_preflight_checks_mount_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    models = corpus / "public" / "models"
    models.mkdir(parents=True)
    (models / "model.glb").write_bytes(b"model")
    observed: list[tuple[str, ...]] = []

    def fake_sandboxed_agent_command(command: tuple[str, ...], **kwargs: object) -> tuple[str, ...]:
        observed.append(command)
        return ("/bin/true",)

    monkeypatch.setattr(
        ergonomics,
        "_sandboxed_agent_command",
        fake_sandboxed_agent_command,
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


@pytest.mark.skipif(os.name == "nt", reason="shell discovery helper uses POSIX executable bits")
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


def _fake_ergonomics_attempt(
    *, provider_error: str, meshprobe_operations: int, raw_stream_path: str
) -> ergonomics.ErgonomicsAttempt:
    return ergonomics.ErgonomicsAttempt(
        episode_id="episode",
        difficulty=Difficulty.BASIC,
        agent=ergonomics.ErgonomicsAgent.CLAUDE,
        attempt=1,
        token_limit=1000,
        command=("claude",),
        return_code=1,
        provider_error=provider_error,
        metrics=ergonomics.ErgonomicsMetrics(
            answer_gate=False,
            evidence_gate=False,
            structured_output=False,
            elapsed_seconds=0.0,
            tokens=ergonomics.TokenUsage(),
            commands=(),
            help_calls=0,
            invalid_calls=0,
            retries=0,
            short_ref_uses=0,
            rg_calls=0,
            jq_calls=0,
            yq_calls=0,
            bytes_read=0,
            raw_reads=0,
            full_file_reads=0,
            redundant_calls=0,
            meshprobe_operations=meshprobe_operations,
            renders=0,
            receipt_path_uses=0,
        ),
        prompt_path="prompt.txt",
        raw_stream_path=raw_stream_path,
    )


def test_ergonomics_retries_only_no_progress_provider_failures(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"type":"rate_limit_event"}\n', encoding="utf-8")
    (tmp_path / "stderr.log").write_text("", encoding="utf-8")

    no_progress = _fake_ergonomics_attempt(
        provider_error="timeout after 180 seconds",
        meshprobe_operations=0,
        raw_stream_path=str(stream),
    )
    semantic_overrun = _fake_ergonomics_attempt(
        provider_error="timeout after 180 seconds",
        meshprobe_operations=4,
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


def test_ergonomics_counts_failed_commands_for_both_provider_streams() -> None:
    raw = "\n".join(
        (
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "exit_code": 2},
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "is_error": True},
                            {"type": "tool_result", "is_error": False},
                        ]
                    },
                }
            ),
        )
    )

    assert ergonomics._count_nonzero_exits(raw) == 2


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

    monkeypatch.setattr(subprocess, "run", run)

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

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match=r"luna.*not supported"):
        ergonomics.preflight_agents()
