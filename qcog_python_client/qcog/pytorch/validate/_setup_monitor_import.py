import ast
import os
import shutil
from pathlib import Path

from qcog_python_client import monitor
from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog.pytorch.validate.shared import FileToValidate


def copy_monitor_package(monitor_package_destination: str):
    """Copy the monitor package to the training directory."""
    pass


def setup_monitor_import(file: FileToValidate) -> None:
    # We need to add the monitoring package from the qcog_package into
    # the training directory and update the import on the file in order
    # to point to the new location.

    # First thing we need to check if the package has already been copied
    # We prepend the package with a `_` to avoid conflicts with the
    # user's package.

    monitor_package_name = "_monitor_"
    monitor_package_position = Path(os.path.abspath(monitor.__file__)).parent

    folder_path = os.path.basename(file.path)

    monitor_package_destination = os.path.join(folder_path, monitor_package_name)

    # Check if the package has already been copied
    if not os.path.exists(monitor_package_destination):
        # Copy the package if it hasn't been copied yet
        logger.info(
            f"Installing the monitor package from {monitor_package_position} to {monitor_package_destination}"
        )

        shutil.copytree(monitor_package_position, monitor_package_destination)

    # Now we can update the import on the file.
    ast_tree = ast.parse(file.content)

    for node in ast.walk(ast_tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "qcog_python_client":
                # The new module is `monitor_package_name`
                node.module = monitor_package_name

    # Write the new content to the file
    # Parse the AST tree to a string see: https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
    file.content = ast.unparse(ast_tree).encode("utf-8")

    print("Updated file content: ", file.content)
