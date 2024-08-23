import os
from pathlib import Path

from qcog_python_client import monitor
from qcog_python_client.qcog.pytorch.validate.shared import (
    FileToValidate,
    ValidateCommand,
)


def copy_monitor_package(monitor_package_destination: str):
    """Copy the monitor package to the training directory."""
    pass


def setup_monitor_import(
    file: FileToValidate,
    command: ValidateCommand,
) -> None:
    # We need to add the monitoring package from the qcog_package into
    # the training directory and update the import on the file in order
    # to point to the new location.

    # First thing we need to check if the package has already been copied
    # We prepend the package with a `_` to avoid conflicts with the
    # user's package.

    monitor_package_name = "_monitor_"
    monitor_package_position = Path(os.path.abspath(monitor.__file__)).parent

    folder_path = os.path.basename(file.path)

    # The directory structure passed in the command.
    directory = command.directory

    print("Directory: ", directory)

    # Write the new content to the file
    # Parse the AST tree to a string see: https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
    # file.content = ast.unparse(ast_tree).encode("utf-8")
