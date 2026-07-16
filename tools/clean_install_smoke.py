"""Build, install, and qualify the MeshProbe wheel in an isolated uv environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = _arguments()
    repo_root = Path(__file__).parents[1].resolve(strict=True)
    corpus_root = args.corpus_root.expanduser().resolve(strict=True)
    tier_manifest = args.tier_manifest.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"clean-install output already exists: {output_root}")
    output_root.mkdir(parents=True)

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required for the clean-install qualification")
    wheel_root = output_root / "wheel"
    wheel_root.mkdir()
    _run((uv, "build", "--wheel", "--out-dir", str(wheel_root)), cwd=repo_root)
    wheels = tuple(wheel_root.glob("meshprobe-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one MeshProbe wheel, found {len(wheels)}")
    wheel = wheels[0]

    environment_root = output_root / "environment"
    _run((uv, "venv", "--python", "3.12", str(environment_root)), cwd=repo_root)
    python = _environment_executable(environment_root, "python")
    _run((uv, "pip", "install", "--python", str(python), str(wheel)), cwd=repo_root)
    meshprobe = _environment_executable(environment_root, "meshprobe")
    _run((str(meshprobe), "--help"), cwd=repo_root)

    tier = _json_object(tier_manifest)
    tier_name = tier.get("tier")
    if not isinstance(tier_name, str):
        raise SystemExit("tier manifest does not declare a string tier")
    qualification_root = output_root / "qualification"
    command = (
        str(meshprobe),
        "eval",
        "run-tier",
        str(corpus_root),
        str(tier_manifest),
        str(qualification_root),
        "--adapter",
        "reference",
        "--blender",
        args.blender,
        "--workers",
        str(args.workers),
    )
    _run(command, cwd=repo_root)

    qualification_report = qualification_root / tier_name / "qualification-report.json"
    qualification = _json_object(qualification_report)
    threshold_failures = qualification.get("threshold_failures")
    if threshold_failures != []:
        raise SystemExit("clean-install qualification did not meet its declared thresholds")
    overall = qualification.get("overall")
    if not isinstance(overall, dict) or not isinstance(overall.get("episodes"), int):
        raise SystemExit("clean-install qualification report has no overall episode count")
    report = {
        "schema_version": 1,
        "status": "passed",
        "platform": platform.platform(),
        "python": _version((str(python), "--version")),
        "uv": _version((uv, "--version")),
        "blender": _version((args.blender, "--version")),
        "wheel": {
            "filename": wheel.name,
            "sha256": _sha256(wheel),
        },
        "tier": tier_name,
        "tier_manifest_sha256": _sha256(tier_manifest),
        "corpus_manifest_sha256": tier.get("corpus_manifest_sha256"),
        "qualification_report": str(qualification_report.relative_to(output_root)),
        "qualification_report_sha256": _sha256(qualification_report),
        "episodes": overall["episodes"],
        "passed_thresholds": True,
    }
    report_path = output_root / "clean-install-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": "passed"}, sort_keys=True))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("tier_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 65))
    return parser.parse_args()


def _environment_executable(root: Path, name: str) -> Path:
    if sys.platform == "win32":
        suffix = ".exe"
        return root / "Scripts" / f"{name}{suffix}"
    return root / "bin" / name


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _version(command: tuple[str, ...]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0]


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
