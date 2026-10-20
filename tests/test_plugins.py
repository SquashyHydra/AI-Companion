"""Tests for plugin auto-discovery."""

import pytest

from companion.tools.plugins import load_plugins
from companion.tools.registry import default_registry


@pytest.fixture(autouse=True)
def _restore_registry():
    before = {t.name for t in default_registry.tools}
    yield
    for name in {t.name for t in default_registry.tools} - before:
        default_registry.unregister(name)


def test_missing_folder_means_no_plugins(tmp_path):
    report = load_plugins(tmp_path / "absent")
    assert report.loaded == []
    assert report.failed == {}


def test_plugin_tools_are_registered(tmp_path):
    (tmp_path / "hello_plugin.py").write_text(
        "from companion.tools.registry import default_registry\n"
        "\n"
        "@default_registry.tool\n"
        "def plugin_hello() -> str:\n"
        '    """Say hello from a plugin."""\n'
        '    return "hi from plugin"\n',
        encoding="utf-8",
    )

    report = load_plugins(tmp_path)

    assert report.loaded == ["hello_plugin.py"]
    assert default_registry.execute("plugin_hello", "{}") == "hi from plugin"


def test_broken_plugin_is_skipped_and_its_partial_tools_removed(tmp_path):
    (tmp_path / "a_broken.py").write_text(
        "from companion.tools.registry import default_registry\n"
        "\n"
        "@default_registry.tool\n"
        "def plugin_partial() -> str:\n"
        '    """Registered before the failure."""\n'
        '    return "x"\n'
        "\n"
        "raise RuntimeError('boom at import')\n",
        encoding="utf-8",
    )
    (tmp_path / "b_good.py").write_text(
        "from companion.tools.registry import default_registry\n"
        "\n"
        "@default_registry.tool\n"
        "def plugin_good() -> str:\n"
        '    """Works fine."""\n'
        '    return "ok"\n',
        encoding="utf-8",
    )

    report = load_plugins(tmp_path)

    assert report.loaded == ["b_good.py"]
    assert "a_broken.py" in report.failed
    assert "boom at import" in report.failed["a_broken.py"]
    names = {t.name for t in default_registry.tools}
    assert "plugin_partial" not in names
    assert "plugin_good" in names


def test_files_starting_with_underscore_are_skipped(tmp_path):
    (tmp_path / "_helpers.py").write_text("raise RuntimeError('should not import')\n", encoding="utf-8")

    report = load_plugins(tmp_path)

    assert report.loaded == []
    assert report.failed == {}
