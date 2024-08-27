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
    python_sys_lib = distutils.sysconfig.get_python_lib(standard_lib=True)
    print("** python_sys_lib", python_sys_lib)

    for imp_ in imports:
        # Split the package name to handle submodules
        base_package = imp_.split(".")[0]

        print("-----> Package ---> ", base_package)

        # Check if it's a package that belongs to the current package
        if is_package_module(os.path.join(package_path, base_package)):
            continue

        spec = importlib.util.find_spec(base_package)

        if spec is None or spec.origin is None:
            continue

        # Built in package
        if spec.origin == 'built-in':
            continue

        print("*** Import Path ", spec.origin)
        # If the spec.origin of the module matches the path of the standard library
        # or the module is a built-in module, then it is not a third-party

        if (
            base_package not in sys.builtin_module_names
            and spec.origin != "built-in"
            and python_sys_lib not in spec.origin
        ):
            third_party_packages.add(base_package)
    return third_party_packages


def is_package_module(module_path: str) -> bool:
    """Check if a Python module exists in the specified path."""
    # Check if the file exists

    module_path = module_path if module_path.endswith(".py") else module_path + ".py"
    return os.path.isfile(module_path)
