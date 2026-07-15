from __future__ import annotations

from meshprobe.models import SceneManifest
from meshprobe.selectors import (
    ComponentIndex,
    ComponentSelector,
    SelectorKind,
    stable_component_id,
)


def test_stable_component_id_depends_on_source_and_path() -> None:
    first = stable_component_id("a" * 64, "root/idler")
    assert first == stable_component_id("a" * 64, "root/idler")
    assert first != stable_component_id("b" * 64, "root/idler")
    assert first != stable_component_id("a" * 64, "root/idler-2")


def test_exact_name_returns_all_candidates(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    matches = index.find(ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern="idler"))
    assert [match.path for match in matches] == ["assembly/drive/idler"]


def test_glob_matches_hierarchy(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    matches = index.find(ComponentSelector(kind=SelectorKind.GLOB, pattern="assembly/**/i*"))
    assert [match.display_name for match in matches] == ["idler"]


def test_regex_matches_path(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    matches = index.find(ComponentSelector(kind=SelectorKind.REGEX, pattern=r"cover/.+"))
    assert [match.display_name for match in matches] == ["retaining clip"]


def test_missing_id_is_explicit(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    try:
        index.by_id("missing")
    except ValueError as error:
        assert str(error) == "unknown component id: missing"
    else:
        raise AssertionError("missing component id was accepted")
