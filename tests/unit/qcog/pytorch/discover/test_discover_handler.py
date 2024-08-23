# tests/pythorch_model/test_discoverhandler.py
import io
import os

import pytest

from qcog_python_client.qcog.pytorch.discover import DiscoverCommand, DiscoverHandler
from qcog_python_client.qcog.pytorch.discover.discoverhandler import (
    _maybe_model_module,
    _maybe_monitor_service_import_module,
)
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
    assert "model_module" in discover_handler.relevant_files
    assert "monitor_service_import_module" in discover_handler.relevant_files
    assert "model.py" in discover_handler.directory
    assert "monitor.py" in discover_handler.directory
    assert "other.py" in discover_handler.directory


@pytest.mark.asyncio
async def test_maybe_model_module(mock_model_dir, discover_handler):
    model_file_path = os.path.join(mock_model_dir, "model.py")
    with open(model_file_path, "rb") as f:  # noqa: ASYNC230
        file_content = io.BytesIO(f.read())

    discover_handler.model_path = mock_model_dir
    result = await _maybe_model_module(discover_handler, model_file_path, file_content)
    assert result is not None
    assert result["path"] == model_file_path


@pytest.mark.asyncio
async def test_maybe_monitor_service_import_module(mock_model_dir, discover_handler):
    monitor_file_path = os.path.join(mock_model_dir, "monitor.py")
    with open(monitor_file_path, "rb") as f:  # noqa: ASYNC230
        file_content = io.BytesIO(f.read())
    discover_handler.model_path = mock_model_dir
    result = await _maybe_monitor_service_import_module(
        discover_handler, monitor_file_path, file_content
    )
    assert result is not None
    assert result["path"] == monitor_file_path


@pytest.mark.asyncio
async def test_revert(discover_handler):
    discover_handler.model_name = "test_model"
    discover_handler.model_path = "/path/to/model"
    discover_handler.relevant_files = {"model_module": {}}

    await discover_handler.revert()

    assert not hasattr(discover_handler, "model_name")
    assert not hasattr(discover_handler, "model_path")
    assert not hasattr(discover_handler, "relevant_files")
