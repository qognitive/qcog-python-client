DEFAULT_QCOG_VERSION = "0.0.101"


def numeric_version(version: str) -> list[int]:
    """Reformulate a string M.N.F version for test comparison.

    Parameters
    ----------
    version : str
        expected to be of the form M.N.F

    Return
    ------
    list[int]
        a list of 3 int that can pythonically compared

    """
    numbers = version.split(".")
    if len(numbers) != 3:
        raise ValueError(f"Invalid version number {version}")

    return [int(w) for w in numbers]
