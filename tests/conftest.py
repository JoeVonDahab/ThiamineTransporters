import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference"
LEGACY_ROOT = ROOT / "data" / "legacy"


def _have(path: Path) -> bool:
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


requires_reference = pytest.mark.skipif(
    not _have(REFERENCE_DIR), reason="run `triplevar reference` to create data/reference"
)
requires_legacy = pytest.mark.skipif(
    not (LEGACY_ROOT / "final_dataset").exists(), reason="legacy artifacts not linked at data/legacy"
)


@pytest.fixture(scope="session")
def slc19a3_reference():
    from triplevar.reference import load_reference

    return load_reference("SLC19A3", REFERENCE_DIR)


@pytest.fixture(scope="session")
def slc19a2_reference():
    from triplevar.reference import load_reference

    return load_reference("SLC19A2", REFERENCE_DIR)
