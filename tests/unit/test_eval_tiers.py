from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.schemas import CorpusTier, RuntimePin
from meshprobe.evals.tiers import (
    pin_private_tier,
    pin_standard_tiers,
    validate_tier_manifest,
)


def runtime_pin() -> RuntimePin:
    return RuntimePin(
        meshprobe_version="0.1.0.dev0",
        blender_version="4.5.3 LTS",
        importer_sha256="a" * 64,
        render_engines=("eevee", "cycles"),
    )


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

    private = pin_private_tier(
        corpus.root,
        tmp_path / "private-manifest",
        runtime=runtime_pin(),
    )
    assert private.tier is CorpusTier.PRIVATE
    assert len(private.episodes) == 2_048


def test_tier_minimums_reject_tiny_corpora(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="tiny-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    with pytest.raises(ValueError, match="at least 2000"):
        pin_standard_tiers(corpus.root, tmp_path / "standard", runtime=runtime_pin())
    with pytest.raises(ValueError, match="at least 500 full-stack"):
        pin_private_tier(corpus.root, tmp_path / "private", runtime=runtime_pin())
