"""Discover the module and the model."""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, TypedDict, TypeAlias
import os

from anyio import open_file

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler
from qcog_python_client.qcog.pytorch.validate.validatehandler import ValidateCommand


class DiscoverCommand(BoundedCommand):
    """Payload to dispatch a discover command."""

    model_name: str
    model_path: str
    command: Command = Command.discover


def pkg_name(package_path: str) -> str:
    """From the package path, get the package name."""
    return os.path.basename(package_path)


class _RelevantFile(TypedDict):
    path: str
    content: str
    pkg_name: str


async def _maybe_model_module(
    self: DiscoverHandler, file_path: str
) -> _RelevantFile | None:
    """Check if the file is the model module."""
    if file_path == self.model_module_name:
        item_path = os.path.join(self.model_path, file_path)

        async with await open_file(item_path, "rb") as file:
            encoded_content = await file.read()
            return {
                "path": item_path,
                "content": encoded_content,
                "pkg_name": pkg_name(self.model_path),
            }
    return None


async def _maybe_monitor_importing_module(
    self: DiscoverHandler, file_path: str
) -> _RelevantFile | None:
    """Check if the file is importing the monitor service."""
    item_path = os.path.join(self.model_path, file_path)

    async with await open_file(item_path, "r") as file:
        content = await file.read()
        if "from qcog_python_client import monitor" in content:
            return {
                "path": item_path,
                "content": content,
                "pkg_name": pkg_name(self.model_path),
            }

    return None


RelevantFileId: TypeAlias = str
FilePath: TypeAlias = str
MaybeIsRelevantFile: TypeAlias = Callable[
    [Handler, FilePath], Awaitable[_RelevantFile | None]
]

relevant_files_map: dict[RelevantFileId, MaybeIsRelevantFile] = {
    "model_module": _maybe_model_module,
    "monitor_service_import": _maybe_monitor_importing_module,
}


async def maybe_relevant_file(
    self: DiscoverHandler,
    file_path: str,
) -> dict[RelevantFileId, _RelevantFile] | None:
    """Check if the file is relevant."""
    for relevant_file_id, maybe_relevant_file in relevant_files_map.items():
        relevant_file = await maybe_relevant_file(self, file_path)
        if relevant_file:
            return {relevant_file_id: relevant_file}

    return None


class DiscoverHandler(Handler):
    """Discover the folder and the model.

    Saves all the relevant files in a dictionary called
    relevant_files.

    For now the only relevant file is the model.py file.
    Eventually we will add also `requirements.txt` in case
    the user wants to install other dependencies.
    """

    model_module_name = "model.py"  # The name of the model module
    monitor_service_import = "from qcog_python_client import monitor"
    retries = 0
    commands = (Command.discover,)
    relevant_files: dict

    async def handle(self, payload: DiscoverCommand) -> ValidateCommand:
        """Handle the discovery of a custom model.

        Parameters
        ----------
        payload : DiscoverCommand
            The payload to discover
            model_name : str
                The name of the model to be used for the current model
            model_path : str
                Where to find the folder containing the model

        """
        self.model_name = payload.model_name
        # Get the absolute path of the model module
        self.model_path = os.path.abspath(payload.model_path)

        # Check if the model module exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model module not found at {payload.model_path}")

        # Check if the folder contains the model module
        content = os.listdir(self.model_path)

        # Initialize the relevant files dictionary
        self.relevant_files = {}

        async def process_file(file_path: str):
            return await maybe_relevant_file(self, file_path)

        processed: list[dict[RelevantFileId, _RelevantFile]] = await asyncio.gather(
            map(process_file, content)
        )

        self.relevant_files = {
            fid: rel for file in processed for fid, rel in file.items()
        }

        # if file == self.model_module_name:
        #     item_path = os.path.join(self.model_path, file)

        #     async with await open_file(item_path, "rb") as file:
        #         encoded_content = await file.read()
        #         self.relevant_files.update(
        #             {
        #                 "model_module": {
        #                     "path": item_path,
        #                     "content": encoded_content,
        #                     "pkg_name": pkg_name(self.model_path),
        #                 }
        #             }
        #         )

        # Once the discovery has been completed,
        # Issue a validate command that will be executed next

        print("Relevant files: ", self.relevant_files)
        return ValidateCommand(
            relevant_files=self.relevant_files,
            model_name=self.model_name,
            model_path=self.model_path,
        )

    async def revert(self) -> None:
        """Revert the changes."""
        # Unset the attributes
        delattr(self, "model_name")
        delattr(self, "model_path")
        delattr(self, "relevant_files")
