"""Configuration for testing the example scripts."""
from pathlib import Path
import runpy
import pytest


def pytest_collect_file(file_path: Path, parent):
    """Pytest hook.

    Create a collector for the given path, or None if not relevant.
    The new node needs to have the specified parent as parent.
    """
    p = Path(file_path)
    if p.suffix == ".py" and "example" in p.name:
        return Script.from_parent(parent, path=p, name=p.name)


class Script(pytest.File):
    """Script files collected by pytest."""

    def collect(self):
        """Collect the script as its own item."""
        yield ScriptItem.from_parent(self, name=self.name)


class ScriptItem(pytest.Item):
    """Item script collected by pytest."""

    def runtest(self) -> None:
        """Run the script as a test."""
        runpy.run_path(str(self.path))

    def repr_failure(self, excinfo):
        """Return only the error traceback of the script."""
        excinfo.traceback = excinfo.traceback.cut(path=self.path)
        return super().repr_failure(excinfo)
