"""Deterministic, hash-pinned qualification tier manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from meshprobe.evals.factory import CorpusBuild, validate_corpus
from meshprobe.evals.generators import PUBLIC_GENERATOR_FAMILIES
from meshprobe.evals.schemas import (
    CorpusTier,
    EpisodeSpec,
    ModelSource,
    PassThresholds,
    RuntimePin,
    TaskFamily,
    TierManifest,
)
from meshprobe.sources import sha256_file

STANDARD_TIER_SIZES: Mapping[CorpusTier, int] = {
    CorpusTier.SMOKE: 40,
    CorpusTier.PULL_REQUEST: 200,
    CorpusTier.NIGHTLY: 800,
}

DEFAULT_THRESHOLDS = PassThresholds(
    overall=0.8,
    full_stack=0.7,
    per_operation=0.75,
)


def current_runtime_pin(blender: str = "blender") -> RuntimePin:
    """Record exact package, importer, protocol, and Blender versions."""

    completed = subprocess.run(
        (blender, "--version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    if not first_line.startswith("Blender "):
        raise RuntimeError(f"unexpected Blender version output: {first_line}")
    worker = Path(__file__).parents[1] / "blender" / "worker.py"
    return RuntimePin(
        meshprobe_version=importlib.metadata.version("meshprobe"),
        blender_version=first_line.removeprefix("Blender "),
        importer_sha256=sha256_file(worker),
        render_engines=("eevee", "cycles"),
    )


def validate_runtime_pin(expected: RuntimePin, actual: RuntimePin | None = None) -> RuntimePin:
    """Fail before a run when package, protocol, Blender, importer, or engines drift."""

    observed = actual or current_runtime_pin()
    if observed != expected:
        differences = {
            field: {"expected": getattr(expected, field), "actual": getattr(observed, field)}
            for field in RuntimePin.model_fields
            if getattr(expected, field) != getattr(observed, field)
        }
        raise RuntimeError(f"qualification runtime does not match its pin: {differences}")
    return observed


def pin_standard_tiers(
    corpus_root: Path,
    output_root: Path,
    *,
    runtime: RuntimePin,
    thresholds: PassThresholds = DEFAULT_THRESHOLDS,
) -> tuple[TierManifest, ...]:
    """Publish nested 40/200/800/all release manifests from one corpus."""

    corpus = validate_corpus(corpus_root.expanduser().resolve(strict=True))
    if corpus.episode_count < 2_000:
        raise ValueError("release qualification requires at least 2000 episodes")
    specs = _load_specs(corpus)
    ordered = _coverage_order(specs.values())
    manifests = [
        _tier_manifest(corpus, specs, ordered[:size], tier, runtime, thresholds)
        for tier, size in STANDARD_TIER_SIZES.items()
    ]
    manifests.append(
        _tier_manifest(corpus, specs, ordered, CorpusTier.RELEASE, runtime, thresholds)
    )
    _publish_manifests(output_root, manifests)
    return tuple(manifests)


def pin_private_tier(
    corpus_root: Path,
    output_root: Path,
    *,
    runtime: RuntimePin,
    thresholds: PassThresholds = DEFAULT_THRESHOLDS,
) -> TierManifest:
    """Pin a held-out corpus with at least 500 full-stack investigations."""

    corpus = validate_corpus(corpus_root.expanduser().resolve(strict=True))
    specs = _load_specs(corpus)
    sources = {spec.model_source for spec in specs.values()}
    if sources != {ModelSource.PRIVATE}:
        raise ValueError("private qualification accepts only evaluator-private models")
    public_families = {family.value for family in PUBLIC_GENERATOR_FAMILIES}
    held_out_families = set(corpus.manifest.generator_families) - public_families
    if not public_families.issubset(corpus.manifest.generator_families) or not held_out_families:
        raise ValueError("private qualification requires released and opaque held-out families")
    if any(seed < 32 for seed in corpus.manifest.generator_seeds):
        raise ValueError("private qualification seeds must not overlap released seeds 0 through 31")
    full_stack = sum(spec.family is TaskFamily.FULL_INVESTIGATION for spec in specs.values())
    if full_stack < 500:
        raise ValueError("private qualification requires at least 500 full-stack episodes")
    ordered = _coverage_order(specs.values())
    manifest = _tier_manifest(
        corpus,
        specs,
        ordered,
        CorpusTier.PRIVATE,
        runtime,
        thresholds,
    )
    _publish_manifests(output_root, [manifest])
    return manifest


def validate_tier_manifest(path: Path, corpus_root: Path) -> TierManifest:
    corpus = validate_corpus(corpus_root.expanduser().resolve(strict=True))
    manifest = TierManifest.model_validate_json(path.read_text(encoding="utf-8"))
    corpus_manifest = corpus.root / "public" / "manifest.json"
    if manifest.corpus_manifest_sha256 != sha256_file(corpus_manifest):
        raise RuntimeError("tier points to a different corpus manifest")
    if manifest.generator_sha256 != corpus.manifest.generator_sha256:
        raise RuntimeError("tier points to a different generator revision")
    if not set(manifest.episodes).issubset(corpus.manifest.episodes):
        raise RuntimeError("tier contains episodes outside its corpus")
    expected_episode_hashes = {
        episode_id: corpus.manifest.episode_sha256[episode_id] for episode_id in manifest.episodes
    }
    if manifest.episode_sha256 != expected_episode_hashes:
        raise RuntimeError("tier episode hashes do not match its corpus")
    specs = _load_specs(corpus)
    model_files = {specs[episode_id].model_file for episode_id in manifest.episodes}
    expected_models = {
        model_file: corpus.manifest.model_sha256[model_file] for model_file in model_files
    }
    if manifest.model_sha256 != expected_models:
        raise RuntimeError("tier model hashes do not match its corpus")
    return manifest


def _load_specs(corpus: CorpusBuild) -> dict[str, EpisodeSpec]:
    return {
        episode_id: EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        for episode_id in corpus.manifest.episodes
    }


def _coverage_order(specs: Iterable[EpisodeSpec]) -> tuple[str, ...]:
    candidates = tuple(specs)
    dimensions = {spec.episode_id: _dimensions(spec) for spec in candidates}
    uncovered = set().union(*dimensions.values())
    ranked = sorted(candidates, key=lambda spec: _rank(spec.episode_id))
    selected: list[EpisodeSpec] = []
    remaining = list(ranked)
    while uncovered:
        best = max(
            remaining,
            key=lambda spec: (
                len(dimensions[spec.episode_id] & uncovered),
                -_rank(spec.episode_id),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered -= dimensions[best.episode_id]
    selected.extend(sorted(remaining, key=lambda spec: _rank(spec.episode_id)))
    return tuple(spec.episode_id for spec in selected)


def _dimensions(spec: EpisodeSpec) -> set[str]:
    values = {
        f"family:{spec.family}",
        f"class:{spec.episode_class}",
        f"difficulty:{spec.difficulty}",
        f"source:{spec.model_source}",
    }
    values.update(f"operation:{operation}" for operation in spec.required_operations)
    return values


def _rank(episode_id: str) -> int:
    digest = hashlib.sha256(f"meshprobe-tier-order-v1\0{episode_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def _tier_manifest(
    corpus: CorpusBuild,
    specs: Mapping[str, EpisodeSpec],
    episodes: tuple[str, ...],
    tier: CorpusTier,
    runtime: RuntimePin,
    thresholds: PassThresholds,
) -> TierManifest:
    model_files = {specs[episode_id].model_file for episode_id in episodes}
    return TierManifest(
        tier=tier,
        corpus_version=corpus.manifest.corpus_version,
        corpus_manifest_sha256=sha256_file(corpus.root / "public" / "manifest.json"),
        generator_sha256=corpus.manifest.generator_sha256,
        runtime=runtime,
        thresholds=thresholds,
        episodes=episodes,
        episode_sha256={
            episode_id: corpus.manifest.episode_sha256[episode_id] for episode_id in episodes
        },
        model_sha256={
            model_file: corpus.manifest.model_sha256[model_file]
            for model_file in sorted(model_files)
        },
    )


def _publish_manifests(output_root: Path, manifests: Iterable[TierManifest]) -> None:
    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        path = output / f"{manifest.tier}.json"
        serialized = manifest.model_dump_json(indent=2) + "\n"
        pending = path.with_name(f".{path.name}.pending")
        pending.write_text(serialized, encoding="utf-8")
        os.replace(pending, path)
