"""Checkpointed, atomic publication of the seeded procedural corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from meshprobe.evals.generators import (
    GeneratedEpisode,
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.leakage import validate_public_directory, validate_public_episode
from meshprobe.evals.schemas import (
    CorpusManifest,
    CorpusTier,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    Operation,
    TaskFamily,
)
from meshprobe.sources import sha256_file


@dataclass(frozen=True)
class CorpusBuild:
    root: Path
    manifest: CorpusManifest
    model_count: int
    episode_count: int
    full_investigation_count: int


def build_corpus(
    output_root: Path,
    *,
    corpus_version: str = "procedural-v1",
    families: Iterable[GeneratorFamily] = tuple(GeneratorFamily),
    seeds: Iterable[int] = range(32),
) -> CorpusBuild:
    selected_families = tuple(families)
    selected_seeds = tuple(seeds)
    if not selected_families or not selected_seeds:
        raise ValueError("corpus generation requires at least one family and seed")
    if len(set(selected_families)) != len(selected_families):
        raise ValueError("corpus families must be unique")
    if len(set(selected_seeds)) != len(selected_seeds) or any(seed < 0 for seed in selected_seeds):
        raise ValueError("corpus seeds must be unique non-negative integers")

    generator_sha256 = generator_source_sha256()
    destination = output_root.expanduser().resolve() / corpus_version
    if destination.exists():
        existing = validate_corpus(destination)
        if existing.manifest.generator_sha256 != generator_sha256:
            raise RuntimeError(
                "corpus version already exists with a different generator; choose a new version"
            )
        requested_families = tuple(family.value for family in selected_families)
        if existing.manifest.generator_families != requested_families:
            raise RuntimeError("corpus version already exists with different generator families")
        if existing.manifest.generator_seeds != selected_seeds:
            raise RuntimeError("corpus version already exists with different generator seeds")
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root.resolve() / f".{corpus_version}.building"
    _prepare_staging(staging, generator_sha256)
    public = staging / "public"
    private = staging / "private"
    models = public / "models"
    public_episodes = public / "episodes"
    private_truth = private / "ground_truth"
    for path in (models, public_episodes, private_truth):
        path.mkdir(parents=True, exist_ok=True)

    completed = _read_checkpoint(staging, generator_sha256)
    for family in selected_families:
        for seed in selected_seeds:
            key = f"{family}:{seed}"
            if key in completed:
                continue
            published = publish_model(build_model(family, seed), models)
            episodes = generate_episodes(published)
            for episode in episodes:
                validate_public_episode(episode.spec, episode.ground_truth)
                _write_episode(public_episodes, private_truth, episode)
            completed.add(key)
            _write_checkpoint(staging, generator_sha256, completed)

    episode_paths = sorted(public_episodes.glob("*.json"))
    episode_ids = tuple(path.stem for path in episode_paths)
    episode_hashes = {path.stem: sha256_file(path) for path in episode_paths}
    model_hashes = {path.name: sha256_file(path) for path in sorted(models.glob("*.glb"))}
    expected_models = len(selected_families) * len(selected_seeds)
    expected_episodes = expected_models * 4
    if (
        len(tuple(models.glob("*.glb"))) != expected_models
        or len(episode_paths) != expected_episodes
    ):
        raise RuntimeError("procedural corpus publication count does not match its declared inputs")
    if set(selected_families) == set(GeneratorFamily) and len(selected_seeds) >= 8:
        _validate_operation_class_coverage(episode_paths)
    manifest = CorpusManifest(
        corpus_version=corpus_version,
        tier=CorpusTier.RELEASE,
        generator_sha256=generator_sha256,
        generator_families=tuple(family.value for family in selected_families),
        generator_seeds=selected_seeds,
        model_sha256=model_hashes,
        episodes=episode_ids,
        episode_sha256=episode_hashes,
    )
    _write_json(public / "manifest.json", manifest.model_dump(mode="json"))
    validate_public_directory(public, private)
    (staging / "checkpoint.json").unlink()
    os.replace(staging, destination)
    return validate_corpus(destination)


def validate_corpus(root: Path) -> CorpusBuild:
    public = root / "public"
    private = root / "private"
    manifest = CorpusManifest.model_validate_json(
        (public / "manifest.json").read_text(encoding="utf-8")
    )
    public_episode_ids = {path.stem for path in (public / "episodes").glob("*.json")}
    private_episode_ids = {path.stem for path in (private / "ground_truth").glob("*.json")}
    expected_episode_ids = set(manifest.episodes)
    if public_episode_ids != expected_episode_ids:
        raise RuntimeError("public episode files do not exactly match the manifest")
    if private_episode_ids != expected_episode_ids:
        raise RuntimeError("private ground-truth files do not exactly match the manifest")
    model_hashes: dict[str, str] = {}
    for episode_id, expected_hash in manifest.episode_sha256.items():
        path = public / "episodes" / f"{episode_id}.json"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"published episode hash mismatch: {episode_id}")
        truth_path = private / "ground_truth" / f"{episode_id}.json"
        if not truth_path.is_file():
            raise RuntimeError(f"published episode has no private ground truth: {episode_id}")
        spec = EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        truth = EpisodeGroundTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
        validate_public_episode(spec, truth)
        if spec.required_operations != truth.required_operations:
            raise RuntimeError(f"operation contract mismatch: {episode_id}")
        model_path = public / "models" / spec.model_file
        if spec.model_file not in model_hashes:
            if not model_path.is_file():
                raise RuntimeError(f"published episode model is missing: {spec.model_file}")
            model_hashes[spec.model_file] = sha256_file(model_path)
        if model_hashes[spec.model_file] != spec.model_sha256:
            raise RuntimeError(f"published model hash mismatch: {spec.model_file}")
    validate_public_directory(public, private)
    model_count = len(tuple((public / "models").glob("*.glb")))
    if model_hashes != manifest.model_sha256:
        raise RuntimeError("model files and hashes do not exactly match the manifest")
    if set(model_hashes) != {path.name for path in (public / "models").glob("*.glb")}:
        raise RuntimeError("model files are not referenced exactly by published episodes")
    if model_count == 512 and len(manifest.episodes) >= 2_048:
        _validate_operation_class_coverage(sorted((public / "episodes").glob("*.json")))
    full_count = 0
    for path in (public / "episodes").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        full_count += payload["family"] == TaskFamily.FULL_INVESTIGATION
    return CorpusBuild(
        root=root,
        manifest=manifest,
        model_count=model_count,
        episode_count=len(manifest.episodes),
        full_investigation_count=full_count,
    )


def generator_source_sha256() -> str:
    directory = Path(__file__).parent
    sources = tuple(
        directory / name
        for name in (
            "factory.py",
            "generators.py",
            "geometry.py",
            "leakage.py",
            "schemas.py",
            "variants.py",
        )
    )
    digest = hashlib.sha256(b"meshprobe-eval-generator-v1\0")
    for path in sources:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _validate_operation_class_coverage(episode_paths: list[Path]) -> None:
    coverage: dict[Operation, set[EpisodeClass]] = {operation: set() for operation in Operation}
    for path in episode_paths:
        spec = EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for operation in spec.required_operations:
            coverage[operation].add(spec.episode_class)
    expected = set(EpisodeClass)
    missing = {
        operation: sorted(str(item) for item in expected - classes)
        for operation, classes in coverage.items()
        if classes != expected
    }
    if missing:
        raise RuntimeError(f"operation class coverage is incomplete: {missing}")


def _prepare_staging(staging: Path, generator_sha256: str) -> None:
    if not staging.exists():
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set())
        return
    checkpoint = staging / "checkpoint.json"
    if not checkpoint.is_file():
        shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set())
        return
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("generator_sha256") != generator_sha256:
        shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set())


def _read_checkpoint(staging: Path, generator_sha256: str) -> set[str]:
    payload = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
    if payload.get("generator_sha256") != generator_sha256:
        raise RuntimeError("staging checkpoint belongs to a different generator revision")
    completed = payload.get("completed", [])
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise RuntimeError("staging checkpoint has an invalid completed-model list")
    return set(completed)


def _write_checkpoint(staging: Path, generator_sha256: str, completed: set[str]) -> None:
    _write_json(
        staging / "checkpoint.json",
        {"generator_sha256": generator_sha256, "completed": sorted(completed)},
    )


def _write_episode(public: Path, private: Path, episode: GeneratedEpisode) -> None:
    _write_json(public / f"{episode.spec.episode_id}.json", episode.spec.model_dump(mode="json"))
    _write_json(
        private / f"{episode.ground_truth.episode_id}.json",
        episode.ground_truth.model_dump(mode="json"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(serialized, encoding="utf-8")
    os.replace(pending, path)
