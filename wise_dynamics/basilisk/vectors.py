"""3-vectors and 3x3 matrices as configuration tuples, and Basilisk's column form."""

__all__ = ["Matrix3", "Vector3", "column"]

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]


def column(vector: Vector3) -> list[list[float]]:
    """A 3-vector in Basilisk's column form, `[[x], [y], [z]]`."""
    return [[float(component)] for component in vector]
