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
from meshprobe.evals.schemas import CorpusManifest, CorpusTier, TaskFamily
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

    destination = output_root.expanduser().resolve() / corpus_version
    if destination.exists():
        return validate_corpus(destination)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root.resolve() / f".{corpus_version}.building"
    generator_sha256 = generator_source_sha256()
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
    expected_models = len(selected_families) * len(selected_seeds)
    expected_episodes = expected_models * 4
    if (
        len(tuple(models.glob("*.glb"))) != expected_models
        or len(episode_paths) != expected_episodes
    ):
        raise RuntimeError("procedural corpus publication count does not match its declared inputs")
    manifest = CorpusManifest(
        corpus_version=corpus_version,
        tier=CorpusTier.RELEASE,
        generator_sha256=generator_sha256,
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
    for episode_id, expected_hash in manifest.episode_sha256.items():
        path = public / "episodes" / f"{episode_id}.json"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"published episode hash mismatch: {episode_id}")
        truth_path = private / "ground_truth" / f"{episode_id}.json"
        if not truth_path.is_file():
            raise RuntimeError(f"published episode has no private ground truth: {episode_id}")
    validate_public_directory(public, private)
    model_count = len(tuple((public / "models").glob("*.glb")))
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
