"""Validate the model module."""

from __future__ import annotations

import os
from typing import Callable

from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog.pytorch.handler import Command, Handler
from qcog_python_client.qcog.pytorch.types import (
    Directory,
    QFile,
    RelevantFileId,
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
        RelevantFileId, Callable[[ValidateHandler, QFile, Directory], Directory]
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

        verify_directory(self.directory)

        return UploadCommand(
            upload_folder=payload.model_path,
            model_name=payload.model_name,
            directory=self.directory,
        )

    async def revert(self) -> None:
        """Revert the changes."""
        pass


def verify_directory(d: Directory):
    """Verify the directory."""
    for file_path, file in d.items():
        if file_path != file.path:
            raise ValueError(f"File path mismatch: {file_path} != {file.path}")

        if file.filename != os.path.basename(file_path):
            raise ValueError(
                f"File path is not a basename: {file_path} - {file.filename}"
            )

        # Make sure the file content is not empty
        if not file.content.read():
            logger.warning(f"File content is empty: {file_path}")

        file.content.seek(0)
