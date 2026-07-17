from __future__ import annotations

from meshprobe.models import SceneManifest
from meshprobe.selectors import (
    ComponentIndex,
    ComponentSelector,
    SelectorKind,
    is_glob_pattern,
    normalize_glob,
    path_glob_match,
    stable_component_id,
)


def test_is_glob_pattern_detects_metacharacters() -> None:
    assert is_glob_pattern("**/idler")
    assert is_glob_pattern("assembly/*")
    assert is_glob_pattern("gear?")
    assert is_glob_pattern("[abc]lip")
    assert not is_glob_pattern("assembly/drive/idler")
    assert not is_glob_pattern("c2")


def test_normalize_glob_anchors_bare_patterns() -> None:
    assert normalize_glob("idler") == "**/idler"
    assert normalize_glob("i*") == "**/i*"
    assert normalize_glob("assembly/**/idler") == "assembly/**/idler"


def test_stable_component_id_depends_on_source_and_path() -> None:
    first = stable_component_id("a" * 64, "root/idler")
    assert first == stable_component_id("a" * 64, "root/idler")
    assert first != stable_component_id("b" * 64, "root/idler")
    assert first != stable_component_id("a" * 64, "root/idler-2")


def test_exact_name_returns_all_candidates(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    matches = index.find(ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern="idler"))
    assert [match.path for match in matches] == ["assembly/drive/idler"]


def test_selector_results_are_sorted_by_stable_path(scene_manifest: SceneManifest) -> None:
    reordered = scene_manifest.model_copy(update={"components": scene_manifest.components[::-1]})
    matches = ComponentIndex(reordered).find(
        ComponentSelector(kind=SelectorKind.GLOB, pattern="assembly/**")
    )
    assert [match.path for match in matches] == sorted(match.path for match in matches)


def test_glob_matches_hierarchy(scene_manifest: SceneManifest) -> None:
    index = ComponentIndex(scene_manifest)
    matches = index.find(ComponentSelector(kind=SelectorKind.GLOB, pattern="assembly/**/i*"))
    assert [match.display_name for match in matches] == ["idler"]


def test_glob_star_does_not_cross_path_segments() -> None:
    assert path_glob_match("assembly/drive/idler", "assembly/*/idler")
    assert not path_glob_match("assembly/drive/gears/idler", "assembly/*/idler")
    assert path_glob_match("assembly/drive/gears/idler", "assembly/**/idler")
    assert path_glob_match("assembly/idler", "assembly/**/idler")


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
