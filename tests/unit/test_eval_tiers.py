from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from meshprobe.evals.factory import build_corpus, validate_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.schemas import (
    CorpusManifest,
    CorpusTier,
    EpisodeGroundTruth,
    EpisodeSpec,
    ModelSource,
    RuntimePin,
    TierManifest,
)
from meshprobe.evals.tiers import (
    _package_sha256,
    current_runtime_pin,
    pin_private_tier,
    pin_standard_tiers,
    validate_runtime_pin,
    validate_tier_manifest,
)


def runtime_pin() -> RuntimePin:
    return RuntimePin(
        meshprobe_version="0.1.0.dev0",
        meshprobe_sha256="b" * 64,
        blender_version="4.5.3 LTS",
        importer_sha256="a" * 64,
        render_engines=("eevee", "cycles"),
    )


def add_opaque_family(corpus_root: Path) -> None:
    episode_root = corpus_root / "public" / "episodes"
    truth_root = corpus_root / "private" / "ground_truth"
    first_spec = EpisodeSpec.model_validate_json(
        next(iter(sorted(episode_root.glob("*.json")))).read_text(encoding="utf-8")
    )
    for episode_path in sorted(episode_root.glob("*.json")):
        spec = EpisodeSpec.model_validate_json(episode_path.read_text(encoding="utf-8"))
        if spec.model_file != first_spec.model_file:
            continue
        truth_path = truth_root / f"{spec.episode_id}.json"
        truth = EpisodeGroundTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
        truth_path.write_text(
            truth.model_copy(update={"generator_family": "opaque_family_v2"}).model_dump_json(),
            encoding="utf-8",
        )
    manifest_path = corpus_root / "public" / "manifest.json"
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    updated = manifest.model_copy(
        update={"generator_families": (*manifest.generator_families, "opaque_family_v2")}
    )
    manifest_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")


def test_shipped_tier_manifests_use_current_schema() -> None:
    manifest_root = Path(__file__).parents[2] / "evals" / "manifests"
    manifest_paths = tuple(sorted(manifest_root.rglob("*.json")))

    assert manifest_paths
    assert all(
        TierManifest.model_validate_json(path.read_text(encoding="utf-8")).schema_version == 3
        for path in manifest_paths
    )


def test_private_family_claim_must_be_backed_by_episode_truth(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="manifest-only-opaque-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(32,),
        model_source=ModelSource.PRIVATE,
    )
    manifest_path = corpus.root / "public" / "manifest.json"
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        manifest.model_copy(
            update={"generator_families": (*manifest.generator_families, "opaque_family_v2")}
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="private episode families"):
        validate_corpus(corpus.root)


def test_standard_tiers_are_nested_exact_and_hash_validated(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path / "corpus", corpus_version="tiers-v1")
    manifests = pin_standard_tiers(
        corpus.root,
        tmp_path / "manifests",
        runtime=runtime_pin(),
    )
    by_tier = {manifest.tier: manifest for manifest in manifests}

    assert len(by_tier[CorpusTier.SMOKE].episodes) == 40
    assert len(by_tier[CorpusTier.PULL_REQUEST].episodes) == 200
    assert len(by_tier[CorpusTier.NIGHTLY].episodes) == 800
    assert len(by_tier[CorpusTier.RELEASE].episodes) == 2_048
    assert set(by_tier[CorpusTier.SMOKE].episodes).issubset(
        by_tier[CorpusTier.PULL_REQUEST].episodes
    )
    assert set(by_tier[CorpusTier.PULL_REQUEST].episodes).issubset(
        by_tier[CorpusTier.NIGHTLY].episodes
    )
    assert (
        validate_tier_manifest(
            tmp_path / "manifests" / "smoke.json",
            corpus.root,
        )
        == by_tier[CorpusTier.SMOKE]
    )
    valid_smoke = by_tier[CorpusTier.SMOKE]
    invalid_path = tmp_path / "invalid-tier.json"
    cases = (
        (valid_smoke.model_copy(update={"corpus_manifest_sha256": "0" * 64}), "corpus"),
        (valid_smoke.model_copy(update={"generator_sha256": "0" * 64}), "generator"),
        (
            valid_smoke.model_copy(
                update={
                    "episodes": (*valid_smoke.episodes, "outside_ep"),
                    "episode_sha256": valid_smoke.episode_sha256 | {"outside_ep": "0" * 64},
                }
            ),
            "outside",
        ),
        (
            valid_smoke.model_copy(
                update={
                    "episode_sha256": valid_smoke.episode_sha256
                    | {valid_smoke.episodes[0]: "0" * 64}
                }
            ),
            "episode hashes",
        ),
        (
            valid_smoke.model_copy(
                update={
                    "model_sha256": valid_smoke.model_sha256
                    | {next(iter(valid_smoke.model_sha256)): "0" * 64}
                }
            ),
            "model hashes",
        ),
    )
    for invalid, message in cases:
        invalid_path.write_text(invalid.model_dump_json(), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            validate_tier_manifest(invalid_path, corpus.root)

    private_corpus = build_corpus(
        tmp_path / "private-corpus",
        corpus_version="private-v1",
        seeds=range(32, 64),
        model_source=ModelSource.PRIVATE,
    )
    add_opaque_family(private_corpus.root)
    private = pin_private_tier(
        private_corpus.root,
        tmp_path / "private-manifest",
        runtime=runtime_pin(),
    )
    assert private.tier is CorpusTier.PRIVATE
    assert len(private.episodes) == 2_048
    assert len(private.model_sha256) == 512


def test_tier_minimums_reject_tiny_corpora(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="tiny-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    with pytest.raises(ValueError, match="at least 2000"):
        pin_standard_tiers(corpus.root, tmp_path / "standard", runtime=runtime_pin())
    with pytest.raises(ValueError, match="evaluator-private"):
        pin_private_tier(corpus.root, tmp_path / "wrong-source", runtime=runtime_pin())
    public_only = build_corpus(
        tmp_path / "wrong-family",
        corpus_version="wrong-family-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(32,),
        model_source=ModelSource.PRIVATE,
    )
    with pytest.raises(ValueError, match="opaque held-out"):
        pin_private_tier(public_only.root, tmp_path / "wrong-family-pin", runtime=runtime_pin())
    overlapping = build_corpus(
        tmp_path / "overlapping",
        corpus_version="overlapping-v1",
        seeds=(0, 1),
        model_source=ModelSource.PRIVATE,
    )
    add_opaque_family(overlapping.root)
    with pytest.raises(ValueError, match="must not overlap"):
        pin_private_tier(overlapping.root, tmp_path / "overlapping-pin", runtime=runtime_pin())
    with pytest.raises(ValueError, match="at least 500 full-stack"):
        private_corpus = build_corpus(
            tmp_path / "private-corpus",
            corpus_version="tiny-private-v1",
            seeds=(32, 33),
            model_source=ModelSource.PRIVATE,
        )
        add_opaque_family(private_corpus.root)
        pin_private_tier(private_corpus.root, tmp_path / "private", runtime=runtime_pin())


def test_runtime_pin_rejects_any_runtime_drift() -> None:
    expected = runtime_pin()
    assert validate_runtime_pin(expected, expected) == expected
    drifted = expected.model_copy(update={"blender_version": "5.2.0 LTS"})
    with pytest.raises(RuntimeError, match="blender_version"):
        validate_runtime_pin(expected, drifted)


def test_current_runtime_pin_reads_exact_package_blender_and_importer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "meshprobe.evals.tiers.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "Blender 5.2.0 LTS\nother\n", ""
        ),
    )
    monkeypatch.setattr("meshprobe.evals.tiers.importlib.metadata.version", lambda _: "1.2.3")

    pin = current_runtime_pin("custom-blender")

    assert pin.meshprobe_version == "1.2.3"
    assert len(pin.meshprobe_sha256) == 64
    assert pin.blender_version == "5.2.0 LTS"
    assert len(pin.importer_sha256) == 64


def test_package_hash_is_path_independent_and_changes_with_python_source(tmp_path: Path) -> None:
    first = tmp_path / "first" / "meshprobe"
    second = tmp_path / "second" / "meshprobe"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
        (root / "nested" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        (root / "ignored.txt").write_text("not executable source\n", encoding="utf-8")

    original = _package_sha256(first)
    assert _package_sha256(second) == original

    (second / "nested" / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert _package_sha256(second) != original


def test_current_runtime_pin_rejects_unexpected_version_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "meshprobe.evals.tiers.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "unknown\n", ""),
    )
    with pytest.raises(RuntimeError, match="unexpected Blender"):
        current_runtime_pin()
