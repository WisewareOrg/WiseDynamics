"""WiseDynamics' explicit Runge-Kutta integrator, compiled against the exact Basilisk build it links.

The integrator, in `_runge_kutta_native`, does the arithmetic of Basilisk's `svIntegratorRungeKutta`
in the same order, so a run is bit-identical to one with Basilisk's integrator, without allocating per
step. It reads Basilisk's C++ classes directly, so importing this module refuses a Basilisk whose
source revision is not the one the library was built against, and a library whose compile-time
layout facts are not the ones Basilisk records: either mismatch would corrupt memory, not raise.
"""

import ctypes
import importlib.util
from dataclasses import dataclass
from typing import Any

import Basilisk
from Basilisk.simulation import spacecraft  # noqa: F401 - loads the libdynamicsLib.so the library links

__all__ = ["Tableau", "install"]


def _load() -> ctypes.CDLL:
    """Load the compiled integrator and check it was built against the Basilisk that is installed.

    Returns:
        The library.

    Raises:
        ImportError: The library is missing, was built against another Basilisk source revision, or
            with another layout than Basilisk's.
    """
    spec = importlib.util.find_spec(f"{__package__}._runge_kutta_native")
    if spec is None or spec.origin is None:
        raise ImportError("the compiled integrator _runge_kutta_native is not installed; reinstall wise-dynamics")
    library = ctypes.CDLL(spec.origin)
    library.wd_install_runge_kutta.restype = ctypes.c_void_p
    library.wd_install_runge_kutta.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.POINTER(ctypes.c_double)] * 3
    for name in ("wd_basilisk_revision", "wd_abi_architecture", "wd_abi_eigen_version"):
        getattr(library, name).restype = ctypes.c_char_p
    info: Any = Basilisk.getBuildInfo()
    built = library.wd_basilisk_revision().decode()
    installed = info["artifact"]["sourceRevision"]
    if built != installed:
        raise ImportError(
            f"the compiled integrator was built against Basilisk {built}, but Basilisk {installed} is installed;"
            " reinstall wise-dynamics against the bsk it pins"
        )
    abi = info["abi"]
    facts = {
        "target.architecture": (library.wd_abi_architecture().decode(), abi["target"]["architecture"]),
        "cxx.standardLibrary.useCxx11Abi": (
            library.wd_abi_use_cxx11_abi(),
            abi["cxx"]["standardLibrary"]["useCxx11Abi"],
        ),
        "dependencies.eigen.version": (
            library.wd_abi_eigen_version().decode(),
            abi["dependencies"]["eigen"]["version"],
        ),
        "dependencies.eigen.maxAlignBytes": (
            library.wd_abi_eigen_max_align_bytes(),
            abi["dependencies"]["eigen"]["maxAlignBytes"],
        ),
    }
    differing = {name: pair for name, pair in facts.items() if pair[0] != pair[1]}
    if differing:
        raise ImportError(
            "the compiled integrator was built with another layout than Basilisk's, as (built, Basilisk):"
            f" {differing}; reinstall wise-dynamics without flags that change them, such as -march or"
            " -D_GLIBCXX_USE_CXX11_ABI"
        )
    return library


_LIBRARY = _load()


@dataclass(frozen=True)
class Tableau:
    """The Butcher tableau of an explicit Runge-Kutta method.

    Attributes:
        a: The stage coefficients, one row per stage, each as long as `b`.
        b: The weights.
        c: The nodes.
    """

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check the tableau is an explicit method of at least one stage, every array its length.

        Raises:
            ValueError: It is not.
        """
        stages = len(self.b)
        if stages == 0 or len(self.c) != stages or len(self.a) != stages:
            raise ValueError(f"a tableau has as many nodes and rows of a as weights, at least one: {self}")
        for i, row in enumerate(self.a):
            if len(row) != stages or any(row[j] != 0 for j in range(i, stages)):
                raise ValueError(f"row {i} of a is not {stages} long and zero from column {i} on: {self}")


def install(craft: Any, tableau: Tableau) -> int:
    """Give a Basilisk spacecraft the compiled integrator of `tableau`, which the spacecraft owns.

    Args:
        craft: The Basilisk `Spacecraft`.
        tableau: The method.

    Returns:
        The integrator's address, which the spacecraft's `integrator` now holds; the spacecraft owns
        the integrator, so the address keeps nothing alive.

    Raises:
        RuntimeError: The spacecraft did not take the integrator, as one whose integration is synced to
            another's does not, or it could not be built.
    """
    stages = len(tableau.b)
    doubles = ctypes.c_double * (stages * stages)
    a = doubles(*(x for row in tableau.a for x in row))
    b = (ctypes.c_double * stages)(*tableau.b)
    c = (ctypes.c_double * stages)(*tableau.c)
    address = _LIBRARY.wd_install_runge_kutta(int(craft.this), stages, a, b, c)
    if not address or int(craft.integrator.this) != address:
        raise RuntimeError(f"spacecraft {craft.ModelTag!r} did not take the compiled integrator")
    return address
