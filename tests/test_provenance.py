"""`distribution_version`: the version string a run file records for an installed distribution."""

import json
from importlib.metadata import PackageNotFoundError

import pytest

import wise_dynamics.provenance
from wise_dynamics import distribution_version


class _FakeDistribution:
    def __init__(self, direct_url: str | None) -> None:
        self.version = "1.2.3"
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (None, "1.2.3"),
        (json.dumps({"url": "file:///src", "dir_info": {}}), "1.2.3"),
        (
            json.dumps({"vcs_info": {"vcs": "git", "commit_id": "0123456789abcdef0123"}}),
            "1.2.3+0123456789ab",
        ),
    ],
)
def test_distribution_version_appends_a_recorded_commit(monkeypatch, direct_url, expected):
    monkeypatch.setattr(
        wise_dynamics.provenance, "distribution", lambda _: _FakeDistribution(direct_url)
    )
    assert distribution_version("anything") == expected


def test_distribution_version_raises_on_a_malformed_direct_url(monkeypatch):
    monkeypatch.setattr(wise_dynamics.provenance, "distribution", lambda _: _FakeDistribution("{"))
    with pytest.raises(json.JSONDecodeError):
        distribution_version("anything")


def test_distribution_version_raises_for_a_missing_distribution():
    with pytest.raises(PackageNotFoundError):
        distribution_version("wise-dynamics-absent-distribution")
