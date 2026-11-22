"""A deliberate type error, to prove CI's type check fails."""


def seeded() -> int:
    """Return a string where the signature promises an int.

    Returns:
        A string, which the signature does not allow.
    """
    return "not an int"
