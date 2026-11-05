"""Test configuration: markers, network opt-in and shared fixtures.

Markers (registered in pyproject.toml, enforced with --strict-markers):

* ``integration``: needs heavy optional dependencies, model downloads or
  external binaries. Such tests call ``tests.support.require_module`` /
  ``require_tool`` (or the fixtures below) and are skipped with a clear reason
  when something is missing. CI deselects them with ``-m "not integration"``.
* ``network``: needs internet access. Skipped unless ``--run-network`` is
  given. A test that downloads model weights is ``integration`` and
  ``network``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from provenance_kit.tools import ExternalTools, Tool
from tests.support import SEED, find_tools, require_tool


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (they need internet access)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs internet access: pass --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def rng() -> np.random.Generator:
    """A NumPy generator with the suite's fixed seed."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def external_tools() -> ExternalTools:
    """The ExternalTools configuration for this test run."""
    return find_tools()


@pytest.fixture
def ffmpeg_path() -> Path:
    """The ffmpeg executable, or skip."""
    return require_tool(Tool.FFMPEG)


@pytest.fixture
def ffprobe_path() -> Path:
    """The ffprobe executable, or skip."""
    return require_tool(Tool.FFPROBE)


@pytest.fixture
def fpcalc_path() -> Path:
    """The fpcalc executable, or skip."""
    return require_tool(Tool.FPCALC)
