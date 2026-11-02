"""The worked example in `wise_dynamics/basilisk/README.md` runs as written."""

import doctest
from pathlib import Path

README = Path(__file__).parent.parent / "wise_dynamics" / "basilisk" / "README.md"
EXAMPLE_STATEMENTS = 18


def test_readme_example_runs():
    result = doctest.testfile(str(README), module_relative=False)
    assert result.failed == 0
    assert result.attempted == EXAMPLE_STATEMENTS
