"""The rule the scenario and vehicle registries share: one subclass of a base per module."""

import importlib
import pkgutil
import types

__all__ = ["subclasses_by_module"]


def subclasses_by_module[T](package: types.ModuleType, base: type[T]) -> dict[str, type[T]]:
    """Map each module of a package to the one subclass of `base` it defines.

    Args:
        package: The package whose modules are imported and searched.
        base: The class whose subclasses are registered.

    Returns:
        The subclass of each module, keyed by the module's name.

    Raises:
        TypeError: A module defines no subclass of `base`, or more than one. Classes a module
            imports do not count.
    """
    found: dict[str, type[T]] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        defined = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, base)
            and value.__module__ == module.__name__
        ]
        if len(defined) != 1:
            raise TypeError(
                f"{base.__name__.lower()} module {module.__name__} defines {len(defined)} "
                f"{base.__name__} subclasses; it must define exactly one"
            )
        found[info.name] = defined[0]
    return found
