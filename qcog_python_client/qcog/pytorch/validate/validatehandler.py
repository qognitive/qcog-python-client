"""Validate the model module."""

from __future__ import annotations

from typing import Callable

from qcog_python_client.qcog.pytorch.handler import Command, Handler
from qcog_python_client.qcog.pytorch.types import (
    Directory,
    QFile,
    ValidateCommand,
)
from qcog_python_client.qcog.pytorch.upload.uploadhandler import UploadCommand
from qcog_python_client.qcog.pytorch.validate._setup_monitor_import import (
    setup_monitor_import,
)
from qcog_python_client.qcog.pytorch.validate._validate_model_module import (
    validate_model_module,
)


class ValidateHandler(Handler):
    """Validate the model module."""

    commands = (Command.validate,)
    attempts = 1
    directory: Directory

    validate_map: dict[
        str, Callable[[ValidateHandler, QFile, Directory], Directory]
    ] = {
        "model_module": validate_model_module,
        "monitor_service_import_module": setup_monitor_import,
    }

    async def handle(self, payload: ValidateCommand) -> UploadCommand:
        """Handle the validation."""
        self.directory = payload.directory

        # `directory` will go through a series of validations
        # based on the `validate_map` keys.
        for key, validate_fn in self.validate_map.items():
            relevant_file = payload.relevant_files.get(key)

            if not relevant_file:
                raise FileNotFoundError(
                    f"File {key} not found in the relevant files. Keys: {payload.relevant_files.keys()}"  # noqa: E501
                )

            parsed = QFile.model_validate(relevant_file)
            self.directory = validate_fn(self, parsed, self.directory)

        raise NotImplementedError("Not implemented yet")

        return UploadCommand(
            upload_folder=payload.model_path, model_name=payload.model_name
        )

    async def revert(self) -> None:
        """Revert the changes."""
        pass
