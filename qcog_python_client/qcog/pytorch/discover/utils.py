"""Shared utilities."""

import os


def pkg_name(package_path: str) -> str:
    """From the package path, get the package name."""
    return os.path.basename(package_path)
