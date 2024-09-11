"""Utility functions for the PyTorch client."""

import io
import os
import re
from typing import Callable

from qcog_python_client.qcog.pytorch.types import FilePath, QFile

default_rules = {
    re.compile(r".*__pycache__.*"),
    re.compile(r".*\.git.*"),
}


def exclude(file_path: str, rules: set[re.Pattern[str]] | None = None) -> bool:
    """Check against a set of regexes rules.

    Parameters
    ----------
    file_path : str
        The path to the file.

    rules : set[re.Pattern[str]] | None
        The set of regex rules to check against. Defaults to `default_rules`.
        if None is passed.

    """
    rules = rules or default_rules
    for pattern in rules:
        if re.match(pattern, file_path):
            return True

    return False


def get_folder_structure(
    file_path: str, *, filter: Callable[[FilePath], bool] | None = None
) -> dict[FilePath, QFile]:
    """Return the folder structure as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the folder.

    filter : Callable[[FilePath], bool] | None
        The filter function to apply to the folder structure.
        to exclude some paths

    Returns
    -------
    dict[FilePath, QFile]
        The folder structure as a dictionary.

    """
    folder_items = os.listdir(file_path)

    retval: dict[FilePath, QFile] = {}
    for item in folder_items:
        item_path = os.path.join(file_path, item)

        if filter and filter(item_path):
            continue

        # Check if the item is a file
        if os.path.isfile(item_path):
            with open(item_path, "rb") as f:
                retval[item_path] = QFile.model_validate(
                    {
                        "path": item_path,
                        "filename": item,
                        "content": io.BytesIO(f.read()),
                    }
                )
        elif os.path.isdir(item_path):
            retval.update(get_folder_structure(item_path))

        else:
            raise ValueError(f"Item {item_path} is neither a file nor a directory.")

    return retval
