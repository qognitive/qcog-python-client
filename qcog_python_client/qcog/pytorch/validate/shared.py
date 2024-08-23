"""Shared models for the validation service."""

from pydantic import BaseModel

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command


class FileToValidate(BaseModel):
    """File to validate."""

    path: str
    content: bytes
    pkg_name: str


class ValidateCommand(BoundedCommand):
    """Validate command."""

    model_name: str
    model_path: str
    relevant_files: dict
    directory: dict
    command: Command = Command.validate
