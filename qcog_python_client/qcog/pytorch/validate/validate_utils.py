"""Utility functions for validating the model package."""

import ast
import os
import pkgutil
import sys
from functools import lru_cache

from qcog_python_client.log import qcoglogger as logger


@lru_cache
def get_stdlib_modules() -> set[str]:
    """Get a set of all standard library modules."""
    stdlib_modules: set[str] = set()
    for importer, modname, ispkg in pkgutil.iter_modules():
        # Exclude packages
        stdlib_modules.add(modname)

    return stdlib_modules


def get_third_party_imports(module_path: str) -> set[str]:
    """Get all third-party packages imported in a Python module.

    Parameters
    ----------
    module_path : str
        The absolute path to the module file (e.g., /path/to/module.py).

    Returns
    -------
    A set of third-party packages imported by the module.

    """
    # Check if the file exists
    if not os.path.isfile(module_path):
        logger.warning(f"Module file not found: {module_path}")
        return set()

    # Read the module's source code
    with open(module_path, "r") as file:
        source_code = file.read()

    # Parse the source code
    tree = ast.parse(source_code)

    # Find all import statements
    imports: set[str] = set()
    for node in ast.walk(tree):
        # Import nodes can be of type ast.Import or ast.ImportFrom
        # as the import statement can be of the form `import module`
        # or `from module import submodule`
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    # Get list of standard library modules
    stdlib_modules = get_stdlib_modules()

    # Identify third-party packages
    third_party_packages = set()
    for imp in imports:
        # Split the package name to handle submodules
        base_package = imp.split(".")[0]
        if (
            base_package not in stdlib_modules
            and base_package not in sys.builtin_module_names
        ):
            third_party_packages.add(base_package)

    return third_party_packages


def is_package_module(module_path: str) -> bool:
    """Check if a Python module exists in the specified path."""
    # Check if the file exists

    module_path = module_path if module_path.endswith(".py") else module_path + ".py"
    return os.path.isfile(module_path)
