"""Read-only 3D model inspection primitives."""

from importlib.metadata import version as distribution_version

from meshprobe.controller import BlenderController
from meshprobe.models import SceneManifest
from meshprobe.session import InspectionSession

__all__ = ["BlenderController", "InspectionSession", "SceneManifest"]
__version__ = distribution_version("meshprobe")
