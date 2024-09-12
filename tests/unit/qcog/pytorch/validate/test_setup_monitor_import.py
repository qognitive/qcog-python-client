import io
from unittest.mock import Mock

import pytest
from anyio import Path

from qcog_python_client.qcog.pytorch.handler import Handler
from qcog_python_client.qcog.pytorch.types import QFile


@pytest.fixture
def mock_handler():
    return Mock(spec=Handler)


@pytest.fixture
def monitor_package_folder_path():
    return "/package_path/to/monitor_package"


@pytest.fixture
def training_package_folder_path():
    return "/package_path/to/training_package"


@pytest.fixture
def mock_relevant_file(training_package_folder_path):
    """Mock the relevant file that has a monitor import statement."""
    # Sample file content with import statement
    sample_content = b"""
from qcog_python_client import monitor

def dummy_function():
    pass
        """
    return QFile(
        filename="train.py",
        path=f"{training_package_folder_path}/train.py",
        content=io.BytesIO(sample_content),
        pkg_name="training_package",
    )


@pytest.fixture
def mock_training_package(training_package_folder_path, mock_relevant_file):
    """Mock a training package with a train.py file that contains the import statement."""  # noqa: E501
    return {
        f"{training_package_folder_path}/__init__.py": QFile(
            filename="__init__.py",
            path=f"{training_package_folder_path}/__init__.py",
            content=io.BytesIO(b""),
            pkg_name="training_package",
        ),
        f"{training_package_folder_path}/train.py": mock_relevant_file,
    }


@pytest.fixture
def mock_monitor_package(monitor_package_folder_path):
    """Mock a monitor package that is located in another path."""
    return {
        f"{monitor_package_folder_path}/__init__.py": QFile(
            filename="__init__.py",
            path=f"{monitor_package_folder_path}/__init__.py",
            content=io.BytesIO(b""),
            pkg_name="monitor_package",
        ),
        f"{monitor_package_folder_path}/monitor.py": QFile(
            filename="monitor.py",
            path=f"{monitor_package_folder_path}/monitor.py",
            content=io.BytesIO(b"def dummy_function(): \n\tpass"),
            pkg_name="monitor_package",
        ),
    }


def test_setup_monitor_import_directory_update(
    mock_handler,
    mock_monitor_package,
    mock_relevant_file,
    mock_training_package,
    monitor_package_folder_path,
    training_package_folder_path,
):
    from qcog_python_client.qcog.pytorch.validate._setup_monitor_import import (
        setup_monitor_import,
    )

    # In this test we want to make sure that, no matter where the monitor
    # package is located, the files are correctly copied into the directory
    # and the path of the files is correctly moved inside the directory.

    # We are overriding the two functions to get the monitor package folder
    # and the monitor package content in order to return the mock values

    # We assume that we will find the same files that are inside the mocked
    # monitor package, inside the directory within a _monitor_ folder.

    updated_directory = setup_monitor_import(
        mock_handler,
        mock_relevant_file,
        mock_training_package,
        monitor_package_folder_path=monitor_package_folder_path,
        folder_content_getter=lambda folder_path: mock_monitor_package,
    )

    monitor_files = [
        (path, f) for path, f in updated_directory.items() if "monitor" in path
    ]

    # The file moved are the same as the mocked monitor package
    assert len(monitor_files) == len(mock_monitor_package)

    # All the files moved have the same path as the keys
    assert any(path == f.path for path, f in monitor_files)

    # The base path of the moved files is the same as
    # the base path of the training package
    monitor_file_paths = [str(Path(path).parent) for path, _ in monitor_files]
    assert any(
        path == training_package_folder_path + "/_monitor_"
        for path in monitor_file_paths
    )

    # The relevant file import has been updated to point to the new location
    relevant_file = updated_directory.get(mock_relevant_file.path)

    assert relevant_file is not None

    relevant_file_content = relevant_file.content.read()
    assert b"import _monitor_ as monitor" in relevant_file_content

    # Make sure the old import is not there anymore
    assert b"from qcog_python_client import monitor" not in relevant_file_content


def test_update_import_exceptions_multiple_files_imported_from_qcog_python_client(
    mock_handler,
    mock_monitor_package,
    mock_relevant_file,
    mock_training_package,
    monitor_package_folder_path,
    training_package_folder_path,
):
    from qcog_python_client.qcog.pytorch.validate._setup_monitor_import import (
        setup_monitor_import,
    )

    file_with_multiple_imports = b"""
from qcog_python_client import monitor, other_module

def dummy_function():
    pass
"""
    file = QFile(
        filename="train.py",
        path=f"{training_package_folder_path}/train.py",
        content=io.BytesIO(file_with_multiple_imports),
        pkg_name="training_package",
    )

    # Overrider the file with one that has multiple imports from the qcog_python_client
    mock_training_package[file.path] = file

    with pytest.raises(ValueError) as exc_info:
        setup_monitor_import(
            mock_handler,
            file,
            mock_training_package,
            monitor_package_folder_path=monitor_package_folder_path,
            folder_content_getter=lambda folder_path: mock_monitor_package,
        )

    exc_info == "Only one import is allowed from the qcog_python_client package."
