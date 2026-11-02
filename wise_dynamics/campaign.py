"""Monte Carlo campaigns: reproducible cases of a named plant, each written to one run file.

This module loads with numpy, h5py and the standard library. `Campaign.plant` names a plant in
`wise_dynamics.basilisk.PLANTS`, and that registry, with Basilisk, is imported on first use of the
plant class.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from wise_dynamics.dispersion import apply_draws
from wise_dynamics.run_file import Dataset, write_run

if TYPE_CHECKING:
    from wise_dynamics.basilisk.plant import Plant

__all__ = ["Campaign", "Case", "distribution_version"]


@dataclass(frozen=True)
class Campaign:
    """A set of `runs` cases of one plant, all seeded from one campaign seed.

    Attributes:
        plant: The plant's name in `wise_dynamics.basilisk.PLANTS`, for example
            `"imu_spacecraft"`.
        seed: The campaign seed, a non-negative integer. Case `k` of a campaign is the same
            whatever `runs` is, so a campaign can be extended or a single case rerun.
        runs: The number of cases, at least 1.
        duration_s: Simulated duration of each case, in seconds, greater than zero.
        out_dir: The directory run files are written to. Stored as an absolute path. Created on
            the first write.

    Raises:
        ValueError: `seed` is negative, `runs` is below 1, or `duration_s` is not positive.
    """

    plant: str
    seed: int
    runs: int
    duration_s: float
    out_dir: Path

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"campaign seed must be non-negative, got {self.seed}")
        if self.runs < 1:
            raise ValueError(f"a campaign needs at least one run, got {self.runs}")
        if not self.duration_s > 0:
            raise ValueError(f"case duration must be positive, got {self.duration_s} s")
        object.__setattr__(self, "out_dir", Path(self.out_dir).absolute())

    @property
    def plant_type(self) -> type["Plant[Any]"]:
        """The plant class `plant` names.

        Raises:
            ValueError: `plant` is absent from `wise_dynamics.basilisk.PLANTS`; the message
                lists the names it holds.
        """
        from wise_dynamics.basilisk.plants import PLANTS

        try:
            return PLANTS[self.plant]
        except KeyError:
            raise ValueError(
                f"unknown plant {self.plant!r}; known plants: {', '.join(sorted(PLANTS))}"
            ) from None

    def case(self, index: int, draws: Mapping[str, Any] | None = None) -> "Case":
        """Case `index` of this campaign.

        Args:
            index: The case number, `0 <= index < runs`.
            draws: Configuration field name to the value this case uses in place of the
                nominal one. `None` selects `standard_draws(index)`.

        Returns:
            The `Case`.

        Raises:
            ValueError: `index` is outside the campaign, or a draw names a field the plant's
                configuration lacks.
        """
        return Case(self, index, self.standard_draws(index) if draws is None else draws)

    def draws_rng(self, index: int) -> np.random.Generator:
        """A fresh generator on case `index`'s draws stream, for a scenario drawing its own
        dispersion.

        Case `index` takes its random streams from
        `numpy.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)`: the first stream draws
        the dispersed configuration, the second seeds the plant's sensor noise.

        Raises:
            ValueError: `index` is outside the campaign.
        """
        return np.random.default_rng(self._streams(index)[0])

    def noise_seed(self, index: int) -> int:
        """The integer seed of case `index`'s sensor-noise generator, from its second stream.

        Raises:
            ValueError: `index` is outside the campaign.
        """
        return int(self._streams(index)[1].generate_state(1)[0])

    def standard_draws(self, index: int) -> dict[str, np.ndarray]:
        """The plant's standard dispersion drawn from case `index`'s draws stream.

        Raises:
            ValueError: `index` is outside the campaign.
        """
        plant_type = self.plant_type
        return plant_type.standard_dispersion.draw(plant_type.nominal(), self.draws_rng(index))

    def _streams(self, index: int) -> list[np.random.SeedSequence]:
        if not 0 <= index < self.runs:
            raise ValueError(f"case {index} is outside a campaign of {self.runs} runs")
        return np.random.SeedSequence(self.seed, spawn_key=(index,)).spawn(2)


@dataclass(frozen=True, eq=False)
class Case:
    """One case of a campaign: its draws, its plant and its run file.

    Build one with `Campaign.case`.

    Attributes:
        campaign: The campaign this case belongs to.
        index: The case number, `0 <= index < campaign.runs`.
        draws: Configuration field name to the value this case uses in place of the nominal
            one, stored as read-only float arrays.
        config: The plant's nominal configuration with `draws` applied.
        noise_seed: The integer seed of this case's sensor-noise generator.

    Raises:
        ValueError: `index` is outside the campaign, or a draw names a field the plant's
            configuration lacks.
    """

    campaign: Campaign
    index: int
    draws: Mapping[str, np.ndarray]

    config: Any = field(init=False)
    noise_seed: int = field(init=False)

    def __post_init__(self) -> None:
        frozen = {name: np.asarray(value, dtype=float) for name, value in self.draws.items()}
        object.__setattr__(self, "draws", MappingProxyType(frozen))
        config = apply_draws(self.campaign.plant_type.nominal(), frozen)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "noise_seed", self.campaign.noise_seed(self.index))

    def build_plant(self) -> "Plant[Any]":
        """A new plant built from `config`, its sensor noise seeded with `noise_seed`."""
        return self.campaign.plant_type(self.config, noise_seed=self.noise_seed)

    @property
    def run_path(self) -> Path:
        """This case's run file, `<out_dir>/run_<index, zero-padded to 4>.h5`."""
        return self.campaign.out_dir / f"run_{self.index:04d}.h5"

    def root_attrs(self) -> dict[str, Any]:
        """The root attributes this case's run file carries.

        Returns:
            `case`, `seed` (the campaign seed), `created_utc` (ISO 8601, UTC, to the second),
            `versions_wise_dynamics`, the plant class's `provenance()` entries (`producer`,
            `versions_bsk`), and one `draw_<name>` float array per entry of `draws`.
        """
        attrs: dict[str, Any] = {
            "case": self.index,
            "seed": self.campaign.seed,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "versions_wise_dynamics": distribution_version("wise-dynamics"),
            **self.campaign.plant_type.provenance(),
        }
        for name, value in self.draws.items():
            attrs[f"draw_{name}"] = value
        return attrs

    def write(
        self, datasets: Mapping[str, Dataset], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        """Write this case's run file at `run_path`.

        Args:
            datasets: Dataset name to `Dataset`, as `wise_dynamics.write_run` takes them.
            extra_attrs: Further root attributes, for example the version of the flight
                software that ran, as `versions_<distribution>`.

        Returns:
            `run_path`.

        Raises:
            ValueError: An entry of `extra_attrs` names one of `root_attrs()`'s keys.
            FileExistsError: `run_path` already exists.
        """
        attrs = self.root_attrs()
        extra_attrs = extra_attrs or {}
        reserved = sorted(set(attrs) & set(extra_attrs))
        if reserved:
            raise ValueError(f"extra_attrs may not set the reserved root attrs {reserved}")
        return write_run(self.run_path, datasets, {**attrs, **extra_attrs})


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
