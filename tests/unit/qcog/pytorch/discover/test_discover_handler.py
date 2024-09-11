# tests/pythorch_model/test_discoverhandler.py
import io
import os

import pytest

from qcog_python_client.qcog.pytorch.discover import DiscoverCommand, DiscoverHandler
from qcog_python_client.qcog.pytorch.discover.discoverhandler import (
    _is_model_module,
    _is_service_import_module,
)
from qcog_python_client.qcog.pytorch.types import QFile
from qcog_python_client.qcog.pytorch.validate.validatehandler import ValidateCommand


@pytest.fixture
def mock_model_dir(tmp_path):
    # Create a temporary directory structure
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Create mock files
    model_file = model_dir / "model.py"
    model_file.write_text("print('This is a model file')")

    monitor_file = model_dir / "monitor.py"
    monitor_file.write_text("from qcog_python_client import monitor")

    other_file = model_dir / "other.py"
    other_file.write_text("print('This is another file')")

    return model_dir


@pytest.fixture
def discover_handler():
    return DiscoverHandler()


@pytest.mark.asyncio
async def test_handle(mock_model_dir, discover_handler):
    payload = DiscoverCommand(model_name="test_model", model_path=str(mock_model_dir))
    result = await discover_handler.handle(payload)

    assert isinstance(result, ValidateCommand)

    relevant_file_ids = result.relevant_files.keys()
    dir_file_names = {f.filename for f in result.directory.values()}
    assert "model_module" in relevant_file_ids
    assert "monitor_service_import_module" in relevant_file_ids
    assert "model.py" in dir_file_names
    assert "monitor.py" in dir_file_names
    assert "other.py" in dir_file_names


@pytest.mark.asyncio
async def test_maybe_model_module_positive(mock_model_dir, discover_handler):
    model_file_path = os.path.join(mock_model_dir, "model.py")
    with open(model_file_path, "rb") as f:  # noqa: ASYNC230
        file_content = io.BytesIO(f.read())

    f = QFile(
        filename="model.py",
        path=model_file_path,
        content=file_content,
        pkg_name=None,
    )
    discover_handler.model_path = mock_model_dir
    result = await _is_model_module(discover_handler, f)
    assert result is True


@pytest.mark.asyncio
async def test_maybe_model_module_negative(mock_model_dir, discover_handler):
    model_file_path = os.path.join(mock_model_dir, "monitor.py")
    with open(model_file_path, "rb") as f:  # noqa: ASYNC230
        file_content = io.BytesIO(f.read())

    f = QFile(
        filename="monitor.py",
        path=model_file_path,
        content=file_content,
        pkg_name=None,
    )

    discover_handler.model_path = mock_model_dir
    result = await _is_model_module(discover_handler, f)
    assert result is False


@pytest.mark.asyncio
async def test_maybe_monitor_service_import_module(mock_model_dir, discover_handler):
    monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    with open(monitor_file_path, "rb") as f:  # noqa: ASYNC230
        file_content = io.BytesIO(f.read())

    f = QFile(
        filename="monitor.py",
        path=monitor_file_path,
        content=file_content,
        pkg_name=None,
    )

    discover_handler.model_path = mock_model_dir
    assert await _is_service_import_module(discover_handler, f)


@pytest.mark.asyncio
async def test_maybe_monitor_service_import_module_wrong_import(
    mock_model_dir, discover_handler
):
    # monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    # with open(monitor_file_path, "rb") as f:  # noqa: ASYNC230
    #     file_content = io.BytesIO(f.read())

    monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    file_content = io.BytesIO(b"from qcog_python_client import monitor, extra")

    f = QFile(
        filename="monitor.py",
        path=monitor_file_path,
        content=file_content,
        pkg_name=None,
    )

    discover_handler.model_path = mock_model_dir
    with pytest.raises(ValueError) as exc_info:
        await _is_service_import_module(discover_handler, f)
    assert (
        "You cannot import anything from qcog_python_client other than monitor."
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_maybe_monitor_service_import_module_with_alias(
    mock_model_dir, discover_handler
):
    # monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    # with open(monitor_file_path, "rb") as f:  # noqa: ASYNC230
    #     file_content = io.BytesIO(f.read())

    monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    file_content = io.BytesIO(b"from qcog_python_client import monitor as mon")

    f = QFile(
        filename="monitor.py",
        path=monitor_file_path,
        content=file_content,
        pkg_name=None,
    )

    discover_handler.model_path = mock_model_dir
    assert await _is_service_import_module(discover_handler, f)


@pytest.mark.asyncio
async def test_revert(discover_handler):
    discover_handler.model_name = "test_model"
    discover_handler.model_path = "/path/to/model"
    discover_handler.relevant_files = {"model_module": {}}

    await discover_handler.revert()

    assert not hasattr(discover_handler, "model_name")
    assert not hasattr(discover_handler, "model_path")
    assert not hasattr(discover_handler, "relevant_files")
