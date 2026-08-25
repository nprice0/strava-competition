"""Import smoke tests for every module under ``strava_competition.tools``.

Guards against import-time crashes in CLI tools (e.g. the historical
``defusedxml.ElementTree.register_namespace`` failure) that would otherwise
only surface when a tool is actually run.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import strava_competition.tools as tools_pkg


def _iter_tool_module_names() -> list[str]:
    """Walk the tools package and return every importable module name."""
    return [
        module_info.name
        for module_info in pkgutil.walk_packages(
            tools_pkg.__path__, prefix=f"{tools_pkg.__name__}."
        )
    ]


_MODULE_NAMES = _iter_tool_module_names()


def test_tools_package_is_not_empty() -> None:
    assert _MODULE_NAMES, "expected at least one module under strava_competition.tools"


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_tool_module_imports(module_name: str) -> None:
    """Every tools module must import without side effects or crashes."""
    importlib.import_module(module_name)
