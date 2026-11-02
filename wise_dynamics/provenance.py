"""Installed versions, for the provenance attributes of a run file."""

import json
from importlib.metadata import distribution

__all__ = ["distribution_version"]


def distribution_version(name: str) -> str:
    """The installed version of a distribution, with the commit it was installed from.

    Args:
        name: The distribution name, for example `"wise-dynamics"` or `"bsk"`.

    Returns:
        The version from the distribution's metadata. When the install recorded a version-control
        commit in its `direct_url.json`, `+` and the commit's first 12 characters are appended,
        for example `"0.1.0+04150b23305d"`.

    Raises:
        importlib.metadata.PackageNotFoundError: `name` is not installed.
        json.JSONDecodeError: The install's `direct_url.json` is malformed.
    """
    installed = distribution(name)
    direct_url = installed.read_text("direct_url.json")
    commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
    return f"{installed.version}+{commit[:12]}" if commit else installed.version
