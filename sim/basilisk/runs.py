"""Writes one HDF5 run file: root attrs plus a set of float64 datasets, each carrying a string
`units` attribute. The caller names every dataset and attr; this module writes exactly what it
is given."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def write_run(
    path: Path,
    arrays: dict[str, tuple[np.ndarray, str]],
    attrs: dict[str, Any],
) -> None:
    """Writes `path` as one HDF5 file: `attrs` at the root (e.g. `case`, `seed`, `producer`,
    `created_utc`, `versions_*`, `draw_<name>`), and one float64 dataset per `arrays` entry
    (dataset name -> `(data, units)`), each carrying its `units` string as its own attribute.
    Creates `path`'s parent directory if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in attrs.items():
            f.attrs[name] = value

        for dataset_name, (data, units) in arrays.items():
            dataset = f.create_dataset(dataset_name, data=np.asarray(data, dtype=np.float64))
            dataset.attrs["units"] = units
