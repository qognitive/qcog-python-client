"""Types shared between multiple handlers."""

from __future__ import annotations

import io
from typing import Literal, TypeAlias

from pydantic import BaseModel

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command


class DiscoverCommand(BoundedCommand):
    """Payload to dispatch a discover command."""

    model_name: str
    model_path: str
    command: Command = Command.discover

    """Shared Types."""


class ValidateCommand(BoundedCommand):
    """Validate command."""

    model_name: str
    model_path: str
    relevant_files: RelevantFiles
    directory: Directory
    command: Command = Command.validate


class QFile(BaseModel):
    """File object."""

    filename: str
    path: str
    content: io.BytesIO
    pkg_name: str | None = None

    model_config = {"arbitrary_types_allowed": True}


FilePath: TypeAlias = str

RelevantFileId: TypeAlias = Literal["model_module", "monitor_service_import_module"]

Directory: TypeAlias = dict[FilePath, QFile]
RelevantFiles: TypeAlias = dict[RelevantFileId, QFile]

