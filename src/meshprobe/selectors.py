"""Stable component identities and deterministic selector behavior."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from meshprobe.identity import stable_component_id
from meshprobe.models import Component, SceneManifest

__all__ = ["ComponentIndex", "ComponentSelector", "SelectorKind", "stable_component_id"]


class SelectorKind(StrEnum):
    EXACT_NAME = "exact_name"
    EXACT_PATH = "exact_path"
    GLOB = "glob"
    REGEX = "regex"


class ComponentSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SelectorKind
    pattern: Annotated[str, StringConstraints(min_length=1, max_length=512)]


class ComponentIndex:
    """A deterministic, renderer-independent index over a scene manifest."""

    def __init__(self, manifest: SceneManifest) -> None:
        self._components = tuple(sorted(manifest.components, key=lambda component: component.path))
        self._by_id = {component.id: component for component in manifest.components}

    def by_id(self, component_id: str) -> Component:
        try:
            return self._by_id[component_id]
        except KeyError as error:
            raise ValueError(f"unknown component id: {component_id}") from error

    def find(self, selector: ComponentSelector) -> tuple[Component, ...]:
        predicate = self._predicate(selector)
        return tuple(component for component in self._components if predicate(component))

    @staticmethod
    def _predicate(selector: ComponentSelector) -> Callable[[Component], bool]:
        if selector.kind is SelectorKind.EXACT_NAME:
            return lambda component: component.display_name == selector.pattern

        if selector.kind is SelectorKind.EXACT_PATH:
            return lambda component: component.path == selector.pattern

        if selector.kind is SelectorKind.GLOB:
            return lambda component: path_glob_match(component.path, selector.pattern)

        try:
            expression = re.compile(selector.pattern)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error
        return lambda component: expression.search(component.path) is not None


def path_glob_match(path: str, pattern: str) -> bool:
    """Match component paths with segment-local `*` and recursive `**`."""

    path_parts = path.split("/")
    pattern_parts = pattern.split("/")
    memo: dict[tuple[int, int], bool] = {}

    def matches(path_index: int, pattern_index: int) -> bool:
        key = (path_index, pattern_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
            memo[key] = result
            return result
        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            result = matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )
            memo[key] = result
            return result
        result = path_index < len(path_parts) and fnmatch.fnmatchcase(
            path_parts[path_index], pattern_part
        )
        result = result and matches(path_index + 1, pattern_index + 1)
        memo[key] = result
        return result

    return matches(0, 0)
