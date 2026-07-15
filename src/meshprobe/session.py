"""In-memory visual state with reset and replay-friendly hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from meshprobe.models import (
    Camera,
    DisplayMode,
    Illumination,
    MarkMode,
    SceneManifest,
)


class ComponentVisualState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display: DisplayMode = DisplayMode.SHOWN
    mark: MarkMode = MarkMode.UNMARKED


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera: Camera
    illumination: Illumination
    components: dict[str, ComponentVisualState]
    state_sha256: str


class InspectionSession:
    """Own visual overrides without mutating imported scene data."""

    def __init__(self, manifest: SceneManifest) -> None:
        self.manifest = manifest
        self._known_ids = frozenset(component.id for component in manifest.components)
        self.reset()

    def reset(self) -> SessionSnapshot:
        self._camera = self.manifest.imported_camera
        self._illumination = self.manifest.imported_illumination
        self._components = {
            component_id: ComponentVisualState() for component_id in sorted(self._known_ids)
        }
        return self.snapshot()

    def set_camera(self, camera: Camera) -> SessionSnapshot:
        self._camera = camera
        return self.snapshot()

    def set_illumination(self, illumination: Illumination) -> SessionSnapshot:
        self._illumination = illumination
        return self.snapshot()

    def display(self, component_ids: Iterable[str], mode: DisplayMode) -> SessionSnapshot:
        selected = self._validated_ids(component_ids)
        if mode is DisplayMode.ISOLATED:
            self._components = {
                component_id: state.model_copy(
                    update={
                        "display": (
                            DisplayMode.ISOLATED if component_id in selected else DisplayMode.HIDDEN
                        )
                    }
                )
                for component_id, state in self._components.items()
            }
            return self.snapshot()

        for component_id in selected:
            self._components[component_id] = self._components[component_id].model_copy(
                update={"display": mode}
            )
        return self.snapshot()

    def mark(self, component_ids: Iterable[str], mode: MarkMode) -> SessionSnapshot:
        selected = self._validated_ids(component_ids)
        for component_id in selected:
            self._components[component_id] = self._components[component_id].model_copy(
                update={"mark": mode}
            )
        return self.snapshot()

    def snapshot(self) -> SessionSnapshot:
        payload = {
            "camera": self._camera.model_dump(mode="json"),
            "illumination": self._illumination.model_dump(mode="json"),
            "components": {
                component_id: state.model_dump(mode="json")
                for component_id, state in sorted(self._components.items())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return SessionSnapshot(
            camera=self._camera,
            illumination=self._illumination,
            components=dict(self._components),
            state_sha256=hashlib.sha256(canonical).hexdigest(),
        )

    def _validated_ids(self, component_ids: Iterable[str]) -> frozenset[str]:
        selected = frozenset(component_ids)
        if not selected:
            raise ValueError("at least one component id is required")
        unknown = selected - self._known_ids
        if unknown:
            raise ValueError(f"unknown component ids: {sorted(unknown)}")
        return selected
