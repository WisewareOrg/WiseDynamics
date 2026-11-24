"""WiseDynamics' explicit Runge-Kutta integrator, compiled against the exact Basilisk build it links.

The integrator, in `_runge_kutta_native`, does the arithmetic of Basilisk's `svIntegratorRungeKutta`
in the same order, so a run is bit-identical to one with Basilisk's integrator, without allocating per
step. It reads Basilisk's C++ classes directly, so importing this module refuses a Basilisk whose
source revision is not the one the library was built against, and a library whose build facts are
not the ones Basilisk records: a layout mismatch would corrupt memory, not raise, and another
instruction set would change the arithmetic.
"""

import ctypes
import functools
import importlib.util
import json
import operator
from dataclasses import dataclass
from typing import Any

import Basilisk
from Basilisk.simulation import spacecraft  # loads the libdynamicsLib.so the library links

__all__ = ["Tableau", "install", "integrates_here"]


def _load() -> ctypes.CDLL:
    """Load the compiled integrator and check it was built against the Basilisk that is installed.

    Returns:
        The library.

    Raises:
        ImportError: The library is missing, was built against another Basilisk source revision, or
            with other build facts than Basilisk's.
    """
    spec = importlib.util.find_spec(f"{__package__}._runge_kutta_native")
    if spec is None or spec.origin is None:
        raise ImportError("the compiled integrator _runge_kutta_native is not installed; reinstall wise-dynamics")
    library = ctypes.CDLL(spec.origin)
    library.wd_install_runge_kutta.restype = ctypes.c_void_p
    library.wd_install_runge_kutta.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.POINTER(ctypes.c_double)] * 3
    library.wd_basilisk_revision.restype = ctypes.c_char_p
    library.wd_integrates_here.restype = ctypes.c_bool
    library.wd_integrates_here.argtypes = [ctypes.c_void_p]
    library.wd_abi.restype = ctypes.c_char_p
    info: Any = Basilisk.getBuildInfo()
    built = library.wd_basilisk_revision().decode()
    installed = info["artifact"]["sourceRevision"]
    if built != installed:
        raise ImportError(
            f"the compiled integrator was built against Basilisk {built}, but Basilisk {installed} is installed;"
            " reinstall wise-dynamics against the bsk it pins"
        )
    recorded = info["abi"]
    facts = {
        name: (built_fact, functools.reduce(operator.getitem, name.split("."), recorded))
        for name, built_fact in json.loads(library.wd_abi()).items()
    }
    differing = {name: pair for name, pair in facts.items() if pair[0] != pair[1]}
    if differing:
        raise ImportError(
            "the compiled integrator was built with other build facts than Basilisk's, as (built, Basilisk):"
            f" {differing}; reinstall wise-dynamics without CXXFLAGS that change them, such as -mavx or"
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


def _check_spacecraft(craft: Any) -> None:
    """Refuse anything but a Basilisk `Spacecraft`, whose address the library reads as its `DynamicObject`.

    Args:
        craft: What is passed as the spacecraft.

    Raises:
        TypeError: `craft` is not a Basilisk `Spacecraft`.
    """
    if not isinstance(craft, spacecraft.Spacecraft):
        raise TypeError(f"the compiled integrator takes a Basilisk Spacecraft, not {type(craft).__name__}")


def install(craft: Any, tableau: Tableau) -> None:
    """Give a Basilisk spacecraft the compiled integrator of `tableau`, which the spacecraft owns.

    Args:
        craft: The Basilisk `Spacecraft`.
        tableau: The method.

    Raises:
        TypeError: `craft` is not a Basilisk `Spacecraft`.
        RuntimeError: The spacecraft did not take the integrator, as one whose integration is synced to
            another's does not, or it could not be built.
    """
    _check_spacecraft(craft)
    stages = len(tableau.b)
    a = (ctypes.c_double * (stages * stages))(*(x for row in tableau.a for x in row))
    b = (ctypes.c_double * stages)(*tableau.b)
    c = (ctypes.c_double * stages)(*tableau.c)
    if not _LIBRARY.wd_install_runge_kutta(int(craft.this), stages, a, b, c):
        raise RuntimeError(f"spacecraft {craft.ModelTag!r} did not take the compiled integrator")


def integrates_here(craft: Any) -> bool:
    """Whether a Basilisk spacecraft integrates with the compiled integrator.

    Args:
        craft: The Basilisk `Spacecraft`.

    Returns:
        True when its integrator is the compiled one.

    Raises:
        TypeError: `craft` is not a Basilisk `Spacecraft`.
    """
    _check_spacecraft(craft)
    return _LIBRARY.wd_integrates_here(int(craft.this))
