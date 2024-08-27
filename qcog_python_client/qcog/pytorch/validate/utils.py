"""Utility functions for validating the input data."""

import ast
import distutils
import distutils.sysconfig
import importlib
import io
import os
import sys

from qcog_python_client.qcog.pytorch.types import Directory, QFile


def validate_directory(dir: dict) -> Directory:
    """Validate the directory."""
    return {k: QFile(**v) for k, v in dir.items()}


def get_third_party_imports(source_code: io.BytesIO, package_path: str) -> set[str]:
    """Get all third-party packages imported in a Python module.

    Parameters
    ----------
    source_code : io.BytesIO
        The source code of the module.
    package_path : str
        The path of the package to which the module belongs.

    Returns
    -------
    A set of third-party packages imported by the module.

    """
    # Parse the source code
    tree = ast.parse(source_code.getvalue())

    # Find all import statements
    imports: set[str] = set()
    for node in ast.walk(tree):
        # Import nodes can be of type ast.Import or ast.ImportFrom
        # as the import statement can be of the form `import module`
        # or `from module import submodule`
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    # Identify third-party packages
    third_party_packages = set()

    # Get the path of the standard library.
    # All the modules that are OS dependent are on this path
    python_sys_lib = distutils.sysconfig.get_python_lib(
        plat_specific=True, standard_lib=True
    )
    print("** python_sys_lib", python_sys_lib)

    for imp_ in imports:
        # Split the package name to handle submodules
        base_package = imp_.split(".")[0]

        print(" - Base package is ", base_package)

        # Check if it's a package that belongs to the current package
        # So it's part of the customer project
        if is_package_module(os.path.join(package_path, base_package)):
            print(" - Is package module")
            continue

        spec = importlib.util.find_spec(base_package)
        print(" - Spec is ", spec)
        if spec is None:
            continue

        is_builtin = (
            spec.origin == "built-in" or
            spec.origin == "frozen" or
            str(spec.origin).startswith(python_sys_lib) or
            base_package in
            sys.builtin_module_names
        )
        if is_builtin:
            print(" - Is builtin")
            continue

        third_party_packages.add(base_package)
    return third_party_packages


def is_package_module(module_path: str) -> bool:
    """Check if a Python module exists in the specified path."""
    # Check if the file exists

    module_path = module_path if module_path.endswith(".py") else module_path + ".py"
    return os.path.isfile(module_path)
