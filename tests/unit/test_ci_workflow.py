from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
PR_GROUP_EXPRESSION = (
    "ci-${{ github.event_name == 'pull_request' && "
    "format('pr-{0}', github.event.pull_request.number) || "
    "format('run-{0}', github.run_id) }}"
)


def _concurrency(event_name: str, pull_request: int | None, run_id: int) -> tuple[str, bool]:
    if event_name == "pull_request":
        assert pull_request is not None
        return f"ci-pr-{pull_request}", True
    return f"ci-run-{run_id}", False


def test_ci_workflow_uses_the_validated_concurrency_expressions() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert workflow["concurrency"] == {
        "group": PR_GROUP_EXPRESSION,
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


@pytest.mark.parametrize(
    ("event_name", "pull_request", "run_id", "expected_group", "expected_cancel"),
    [
        ("pull_request", 193, 1001, "ci-pr-193", True),
        ("pull_request", 193, 1002, "ci-pr-193", True),
        ("pull_request", 194, 1003, "ci-pr-194", True),
        ("push", None, 2001, "ci-run-2001", False),
        ("push", None, 2002, "ci-run-2002", False),
    ],
)
def test_ci_concurrency_event_matrix(
    event_name: str,
    pull_request: int | None,
    run_id: int,
    expected_group: str,
    expected_cancel: bool,
) -> None:
    assert _concurrency(event_name, pull_request, run_id) == (
        expected_group,
        expected_cancel,
    )
