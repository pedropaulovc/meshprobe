from __future__ import annotations

from pathlib import Path

from meshprobe.evals.generators import GeneratorFamily, build_model, publish_model
from meshprobe.evals.reference_agent import infer_arrow_direction


def test_reference_agent_reads_arrow_geometry_without_role_metadata(tmp_path: Path) -> None:
    observed: set[str] = set()
    for seed in range(16):
        generated = build_model(GeneratorFamily.STAMPED_ARROW, seed)
        published = publish_model(generated, tmp_path)
        arrow_name = next(
            component.name for component in generated.scene.components if component.role == "arrow"
        )
        expected = str(generated.semantic_answers["arrow_direction"])
        observed.add(expected)
        assert infer_arrow_direction(published.path, arrow_name) == expected

    assert observed == {"left", "right", "up", "down"}
