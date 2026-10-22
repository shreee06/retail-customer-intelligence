"""Path helpers shared by pipeline stages."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_relative(path: Path) -> str:
    """Return a POSIX path relative to the project root for use in reports.

    Paths outside the project (e.g. temporary test folders) fall back to the
    file name so reports never embed absolute, machine-specific locations.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name
