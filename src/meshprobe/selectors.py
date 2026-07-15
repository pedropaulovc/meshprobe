"""Stable component identities and deterministic selector behavior."""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from meshprobe.models import Component, SceneManifest


class SelectorKind(StrEnum):
    EXACT_NAME = "exact_name"
    EXACT_PATH = "exact_path"
    GLOB = "glob"
    REGEX = "regex"


class ComponentSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SelectorKind
    pattern: Annotated[str, StringConstraints(min_length=1, max_length=512)]


def stable_component_id(source_sha256: str, instance_path: str) -> str:
    """Derive an ID from immutable source identity and the hierarchy path."""

    digest = hashlib.sha256(f"{source_sha256}\0{instance_path}".encode()).hexdigest()
    return f"cmp_{digest[:24]}"


class ComponentIndex:
    """A deterministic, renderer-independent index over a scene manifest."""

    def __init__(self, manifest: SceneManifest) -> None:
        self._components = manifest.components
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
            return lambda component: fnmatch.fnmatchcase(component.path, selector.pattern)

        try:
            expression = re.compile(selector.pattern)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error
        return lambda component: expression.search(component.path) is not None
