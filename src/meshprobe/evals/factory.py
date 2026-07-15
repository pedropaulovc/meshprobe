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
    PUBLIC_GENERATOR_FAMILIES,
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
    ModelSource,
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
    families: Iterable[GeneratorFamily] = PUBLIC_GENERATOR_FAMILIES,
    seeds: Iterable[int] = range(32),
    model_source: ModelSource = ModelSource.PROCEDURAL,
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
    recipe_sha256 = _recipe_sha256(
        generator_sha256,
        selected_families,
        selected_seeds,
        model_source,
    )
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
        if existing.manifest.model_sources != (model_source,):
            raise RuntimeError("corpus version already exists with a different model source")
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root.resolve() / f".{corpus_version}.building"
    _prepare_staging(staging, generator_sha256, recipe_sha256)
    public = staging / "public"
    private = staging / "private"
    models = public / "models"
    public_episodes = public / "episodes"
    private_truth = private / "ground_truth"
    for path in (models, public_episodes, private_truth):
        path.mkdir(parents=True, exist_ok=True)

    completed = _read_checkpoint(staging, generator_sha256, recipe_sha256)
    for family in selected_families:
        for seed in selected_seeds:
            key = f"{family}:{seed}"
            if key in completed:
                continue
            published = publish_model(build_model(family, seed), models)
            episodes = generate_episodes(published, model_source=model_source)
            for episode in episodes:
                validate_public_episode(episode.spec, episode.ground_truth)
                _write_episode(public_episodes, private_truth, episode)
            completed.add(key)
            _write_checkpoint(staging, generator_sha256, completed, recipe_sha256)

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
    if set(selected_families) == set(PUBLIC_GENERATOR_FAMILIES) and len(selected_seeds) >= 8:
        _validate_operation_class_coverage(episode_paths)
    manifest = CorpusManifest(
        corpus_version=corpus_version,
        tier=CorpusTier.RELEASE,
        generator_sha256=generator_sha256,
        model_sources=(model_source,),
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
    model_families: dict[str, str] = {}
    model_sources: set[ModelSource] = set()
    for episode_id, expected_hash in manifest.episode_sha256.items():
        path = public / "episodes" / f"{episode_id}.json"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"published episode hash mismatch: {episode_id}")
        truth_path = private / "ground_truth" / f"{episode_id}.json"
        if not truth_path.is_file():
            raise RuntimeError(f"published episode has no private ground truth: {episode_id}")
        spec = EpisodeSpec.model_validate_json(path.read_text(encoding="utf-8"))
        model_sources.add(spec.model_source)
        truth = EpisodeGroundTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
        previous_family = model_families.setdefault(spec.model_file, truth.generator_family)
        if previous_family != truth.generator_family:
            raise RuntimeError(f"model episodes disagree on generator family: {spec.model_file}")
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
    if model_sources != set(manifest.model_sources):
        raise RuntimeError("episode model sources do not match the corpus manifest")
    if set(model_families.values()) != set(manifest.generator_families):
        raise RuntimeError("private episode families do not match the corpus manifest")
    if set(model_hashes) != {path.name for path in (public / "models").glob("*.glb")}:
        raise RuntimeError("model files are not referenced exactly by published episodes")
    procedural_families = {family.value for family in PUBLIC_GENERATOR_FAMILIES}
    if (
        procedural_families.issubset(manifest.generator_families)
        and len(manifest.episodes) >= 2_048
    ):
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


def merge_corpora(
    corpus_roots: Iterable[Path],
    output_root: Path,
    *,
    corpus_version: str = "qualification-v1",
) -> CorpusBuild:
    """Atomically combine independently validated corpora without changing artifacts."""

    corpora = tuple(
        validate_corpus(path.expanduser().resolve(strict=True)) for path in corpus_roots
    )
    if len(corpora) < 2:
        raise ValueError("corpus merge requires at least two validated inputs")
    composite_hash = _composite_generator_sha256(corpora)
    destination = output_root.expanduser().resolve() / corpus_version
    if destination.exists():
        existing = validate_corpus(destination)
        if existing.manifest.generator_sha256 != composite_hash:
            raise RuntimeError("merged corpus version belongs to different input manifests")
        return existing
    staging = output_root.expanduser().resolve() / f".{corpus_version}.building"
    if staging.exists():
        shutil.rmtree(staging)
    public = staging / "public"
    private = staging / "private"
    models = public / "models"
    episodes = public / "episodes"
    ground_truth = private / "ground_truth"
    for path in (models, episodes, ground_truth):
        path.mkdir(parents=True, exist_ok=True)

    model_hashes: dict[str, str] = {}
    episode_hashes: dict[str, str] = {}
    episode_file_hashes: dict[str, str] = {}
    families: list[str] = []
    seeds: list[int] = []
    model_sources: list[ModelSource] = []
    for corpus in corpora:
        for model_source in corpus.manifest.model_sources:
            if model_source not in model_sources:
                model_sources.append(model_source)
        for family in corpus.manifest.generator_families:
            if family not in families:
                families.append(family)
        for seed in corpus.manifest.generator_seeds:
            if seed not in seeds:
                seeds.append(seed)
        _merge_tree(
            corpus.root / "public" / "models",
            models,
            corpus.manifest.model_sha256,
            model_hashes,
        )
        _merge_tree(
            corpus.root / "public" / "episodes",
            episodes,
            {
                f"{episode_id}.json": digest
                for episode_id, digest in corpus.manifest.episode_sha256.items()
            },
            episode_file_hashes,
        )
        for episode_id, digest in corpus.manifest.episode_sha256.items():
            previous = episode_hashes.setdefault(episode_id, digest)
            if previous != digest:
                raise RuntimeError(f"episode collision has different content: {episode_id}")
        _merge_private_truth(corpus, ground_truth)

    public_episode_paths = sorted(episodes.glob("*.json"))
    manifest = CorpusManifest(
        corpus_version=corpus_version,
        tier=CorpusTier.RELEASE,
        generator_sha256=composite_hash,
        model_sources=tuple(model_sources),
        generator_families=tuple(families),
        generator_seeds=tuple(seeds),
        model_sha256=model_hashes,
        episodes=tuple(path.stem for path in public_episode_paths),
        episode_sha256=episode_hashes,
    )
    _write_json(public / "manifest.json", manifest.model_dump(mode="json"))
    validate_public_directory(public, private)
    os.replace(staging, destination)
    return validate_corpus(destination)


def _composite_generator_sha256(corpora: tuple[CorpusBuild, ...]) -> str:
    digest = hashlib.sha256(b"meshprobe-composite-corpus-v1\0")
    for corpus in sorted(corpora, key=lambda item: item.manifest.corpus_version):
        manifest_path = corpus.root / "public" / "manifest.json"
        digest.update(corpus.manifest.corpus_version.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(manifest_path)))
    return digest.hexdigest()


def _merge_tree(
    source_root: Path,
    destination_root: Path,
    declared_hashes: dict[str, str],
    accumulated_hashes: dict[str, str],
) -> None:
    for filename, digest in declared_hashes.items():
        source = source_root / filename
        destination = destination_root / filename
        previous = accumulated_hashes.setdefault(filename, digest)
        if previous != digest:
            raise RuntimeError(f"artifact collision has different content: {filename}")
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _merge_private_truth(corpus: CorpusBuild, destination: Path) -> None:
    for episode_id in corpus.manifest.episodes:
        source = corpus.root / "private" / "ground_truth" / f"{episode_id}.json"
        target = destination / source.name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"ground truth collision has different content: {episode_id}")
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def generator_source_sha256() -> str:
    directory = Path(__file__).parent
    sources = (
        *(
            directory / name
            for name in (
                "factory.py",
                "generators.py",
                "geometry.py",
                "leakage.py",
                "schemas.py",
                "variants.py",
            )
        ),
        directory.parent / "blender" / "worker.py",
        directory.parent / "identity.py",
        directory.parent / "models.py",
    )
    digest = hashlib.sha256(b"meshprobe-eval-generator-v1\0")
    for path in sources:
        digest.update(path.relative_to(directory.parent).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _recipe_sha256(
    generator_sha256: str,
    families: tuple[GeneratorFamily, ...],
    seeds: tuple[int, ...],
    model_source: ModelSource,
) -> str:
    payload = {
        "generator_sha256": generator_sha256,
        "families": [family.value for family in families],
        "seeds": seeds,
        "model_source": model_source.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _prepare_staging(
    staging: Path,
    generator_sha256: str,
    recipe_sha256: str | None = None,
) -> None:
    recipe = recipe_sha256 or generator_sha256
    if not staging.exists():
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set(), recipe)
        return
    checkpoint = staging / "checkpoint.json"
    if not checkpoint.is_file():
        shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set(), recipe)
        return
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (
        payload.get("generator_sha256") != generator_sha256
        or payload.get("recipe_sha256", payload.get("generator_sha256")) != recipe
    ):
        shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_checkpoint(staging, generator_sha256, set(), recipe)


def _read_checkpoint(
    staging: Path,
    generator_sha256: str,
    recipe_sha256: str | None = None,
) -> set[str]:
    recipe = recipe_sha256 or generator_sha256
    payload = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
    if payload.get("generator_sha256") != generator_sha256:
        raise RuntimeError("staging checkpoint belongs to a different generator revision")
    if payload.get("recipe_sha256", payload.get("generator_sha256")) != recipe:
        raise RuntimeError("staging checkpoint belongs to a different corpus recipe")
    completed = payload.get("completed", [])
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise RuntimeError("staging checkpoint has an invalid completed-model list")
    return set(completed)


def _write_checkpoint(
    staging: Path,
    generator_sha256: str,
    completed: set[str],
    recipe_sha256: str | None = None,
) -> None:
    _write_json(
        staging / "checkpoint.json",
        {
            "generator_sha256": generator_sha256,
            "recipe_sha256": recipe_sha256 or generator_sha256,
            "completed": sorted(completed),
        },
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
