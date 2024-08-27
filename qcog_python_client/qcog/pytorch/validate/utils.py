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
        standard_lib=True, plat_specific=True
    )

    for imp_ in imports:
        # Split the package name to handle submodules
        base_package = imp_.split(".")[0]
        print("\n------> Base Package: ", base_package)
        # Check if it's a package that belongs to the current package
        if is_package_module(os.path.join(package_path, base_package)):
            continue

        if base_package in sys.builtin_module_names:
            continue

        spec = importlib.util.find_spec(base_package)

        if spec is None or spec.origin is None:
            continue

        if spec.origin == "frozen" or spec.origin == "built-in":
            continue

        # At this point we might have 2 cases:
        # os specific modules or third-party modules
        # os specific modules are in `python_sys_lib`
        # third-party modules are usually in
        # `<python_sys_lib>/site-packages` or `<python_sys_lib>/dist-packages`.
        # But depending on how python interpreter calls the script
        # os specific modules could be located in a system
        # specific path like `/usr/lib/python3.XX/`

        common_path = os.path.commonpath([spec.origin, python_sys_lib])
        print("\n------> Module Path: ", spec.origin)
        print("------> Python Sys Lib: ", python_sys_lib)
        print("------> Common path: ", common_path)

        if infolder(spec.origin, python_sys_lib):
            # Check if the module is directly inside the folder
            # or if it's a subpackage
            continue

        print(f"------> Third Party Package: {base_package}")
        third_party_packages.add(base_package)
    return third_party_packages


def is_package_module(module_path: str) -> bool:
    """Check if a Python module exists in the specified path."""
    # Check if the file exists

    module_path = module_path if module_path.endswith(".py") else module_path + ".py"
    return os.path.isfile(module_path)


def infolder(module_path: str, folder_path: str) -> bool:
    """Check if a module is in a folder."""
    # Folder Path should be a substring of the module path
    if not module_path.startswith(folder_path):
        return False

    # Check if the file that the module is pointing at is a `__init__.py` file.
    # If it is then it's a package and the previous segment of the path
    # is the package name, otherwise the module is the package name.

    package_name = None
    filename = os.path.basename(module_path)
    if filename == "__init__.py":
        package_name = os.path.basename(os.path.dirname(module_path))
    else:
        package_name = os.path.basename(module_path)

    print("Package Name: ", package_name)
    # The module is in the folder if the `package_name` is right after
    # the `folder_path` in the `module_path`
    # so something like `<folder_path>/<package_name>` should be
    # the a substring of the `module_path`
    module_subpath_candidate = os.path.join(folder_path, package_name)

    return module_path.startswith(module_subpath_candidate)

