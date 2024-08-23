import ast
import copy
import io
import os
from pathlib import Path

from qcog_python_client import monitor
from qcog_python_client.qcog.pytorch import utils
from qcog_python_client.qcog.pytorch.handler import Handler
from qcog_python_client.qcog.pytorch.types import (
    Directory,
    QFile,
    ValidateCommand,
)

MONITOR_PACKAGE_NAME = "_monitor_"


def setup_monitor_import(
    self: Handler[ValidateCommand],
    file: QFile,
    directory: Directory,
) -> Directory:
    # We need to add the monitoring package from the qcog_package into
    # the training directory and update the import on the file in order
    # to point to the new location.

    # First thing we need to check if the package has already been copied
    # We prepend the package with a `_` to avoid conflicts with the
    # user's package.

    monitor_package = Path(os.path.abspath(monitor.__file__)).parent

    package_content = utils.get_folder_structure(
        str(monitor_package),
        filter=utils.exclude,  # Apply exclusion rules
    )

    # Now we want to copy the package to the training directory.
    # This "copy" is only happening in memory, we are not writing.
    # The `folder` is defined by the `keys` of the dictionary.
    # We need to change the keys and the `path` of the files in the
    # package_content dictionary in order to match the new location
    # defined by the keys of the `directory` dictionary.

    # The `root` of the folder is defined as the parent folder
    # of the file to validate.

    # We can use that to re-construct the new path of the files
    # in the package_content dictionary.

    root = Path(file.path).parent

    for file_path, file_ in package_content.items():
        # Get the relative path of the file
        relative_path = os.path.relpath(file_path, monitor_package)

        # prepend the relative path of the content of the package
        # with the package name, that, in this case is `_monitor_`
        # to avoid conflicts with the user's package.
        # Et voilà, we have the new path of the file in the training
        new_path = os.path.join(root, MONITOR_PACKAGE_NAME, relative_path)

        # Update the path of the file
        file_.path = new_path
        file_.pkg_name = MONITOR_PACKAGE_NAME

        # Update the directory
        directory[new_path] = file_

    # Now we need to update the import in the file that has the
    # import of the monitor package. The file is the same file
    # at the address of the file to validate.

    # Generate the ast from the content
    ast_tree = ast.parse(file.content.getvalue())

    # Now we need to find the import statement
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "qcog_python_client":
                # For now there is only the package monitor
                # to import, but in case in the future we
                # have more packages that are importable
                # from the python client into the training
                # script, then we need to check that the
                # names only contain the package monitor.
                if len(node.names) > 1:
                    raise ValueError(
                        "Only one import is allowed from the qcog_python_client package."  # noqa: E501
                    )

                package_name = node.names[0].name

                if package_name != "monitor":
                    raise ValueError(
                        "The only package that can be imported is monitor."
                    )

                # Now we need to update the import statement
                # to point to the new package
                node.module = MONITOR_PACKAGE_NAME

                # Now re-write the content of the file
                # starting from the modified AST tree
                # Parse the AST tree to a string see: https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
                file.content = io.BytesIO(ast.unparse(ast_tree).encode())

    # Now that the file has been updated, we need to update the corresponding
    # file in the directory
    directory[file.path] = file

    return copy.deepcopy(directory)
