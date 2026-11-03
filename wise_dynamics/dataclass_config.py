"""The shape of any dataclass instance, which every plant configuration is."""

from typing import Any, ClassVar, Protocol

__all__ = ["DataclassConfig"]


class DataclassConfig(Protocol):
    """Any dataclass instance: the shape of a plant configuration."""

    __dataclass_fields__: ClassVar[dict[str, Any]]
