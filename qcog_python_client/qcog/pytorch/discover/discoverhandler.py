"""Discover the module and the model.

Finds the folder, convert the folder into a dictionary
Create a `relevant_files` dictionary that contains the
relevant files for the model. that will be validated later.
"""

from __future__ import annotations

import ast
import asyncio
import io
import os
from typing import (
    IO,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    TypeAlias,
    TypedDict,
)

from anyio import open_file

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler
from qcog_python_client.qcog.pytorch.validate.validatehandler import ValidateCommand


class DiscoverCommand(BoundedCommand):
    """Payload to dispatch a discover command."""

    model_name: str
    model_path: str
    command: Command = Command.discover


class _File(TypedDict):
    path: str
    content: IO[bytes]
    pkg_name: str


RelevantFileId: TypeAlias = str
FilePath: TypeAlias = str
FileContent: TypeAlias = io.BytesIO
MaybeIsRelevantFile: TypeAlias = Callable[
    [Handler, FilePath, FileContent], Coroutine[Any, Any, _File | None]
]


def pkg_name(package_path: str) -> str:
    """From the package path, get the package name."""
    return os.path.basename(package_path)


async def _maybe_model_module(
    self: DiscoverHandler, file_path: FilePath, file_content: io.BytesIO
) -> _File | None:
    """Check if the file is the model module."""
    module_name = os.path.basename(file_path)
    if module_name == self.model_module_name:
        return {
            "path": file_path,
            "content": file_content,
            "pkg_name": pkg_name(self.model_path),
        }
    return None


async def _maybe_monitor_service_import_module(
    self: DiscoverHandler, file_path: str, file_content: io.BytesIO
) -> _File | None:
    """Check if the file is importing the monitor service."""
    # Make sure the item is not a folder. If so, exit
    if os.path.isdir(file_path):
        return None

    tree = ast.parse(file_content.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # We assume that the only way that the monitor service
            # will be imported is like
            # from qcog_python_client import monitor or eventually
            # from qcog_python_client import monitor as <alias>.
            if node.module == "qcog_python_client" and any(
                a.name == "monitor" for a in node.names
            ):
                return {
                    "path": file_path,
                    "content": file_content,
                    "pkg_name": pkg_name(self.model_path),
                }
    return None


relevant_files_map: dict[RelevantFileId, MaybeIsRelevantFile] = {
    "model_module": _maybe_model_module,
    "monitor_service_import_module": _maybe_monitor_service_import_module,
}


async def maybe_relevant_file(
    self: DiscoverHandler,
    file_path: FilePath,
    file_content: io.BytesIO,
) -> dict[RelevantFileId, _File]:
    """Check if the file is relevant."""
    retval: dict[RelevantFileId, _File] = {}

    for relevant_file_id, _maybe_relevant_file_fn in relevant_files_map.items():
        relevant_file = await _maybe_relevant_file_fn(self, file_path, file_content)
        if relevant_file:
            retval.update({relevant_file_id: relevant_file})

    return retval


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

        # Load all the folder in memory
        self.directory: dict[FilePath, _File] = {}
        pkg_name_ = pkg_name(self.model_path)

        for item in content:
            item_path = os.path.join(self.model_path, item)
            # Avoid folders
            if os.path.isdir(item_path):
                continue

            async with await open_file(item_path, "rb") as file:
                encoded_content = await file.read()
                self.directory[item] = {
                    "path": item_path,
                    "content": encoded_content,
                    "pkg_name": pkg_name_,
                }

        # Figure it out which files are relevant.
        # Relevant files have a specific id that
        # is specified in the `relevant_files_map`
        # dictionary as the key of that dictionary.

        # Initialize the relevant files dictionary
        self.relevant_files = {}

        def process_file(
            f: tuple[FilePath, FileContent],
        ) -> Awaitable[dict[RelevantFileId, _File]]:
            return maybe_relevant_file(self, *f)

        # Tranform the directory dictionary into a list
        # of tuple (file_path, file_content) to be passed
        # to the filter function `process_file`

        dir_content = list(
            map(lambda x: (x[0], io.BytesIO(x[1]["content"])), self.directory.items())
        )

        # Process the files in parallel gathering the results
        # from the coroutines returned by the `process_file`
        # function. Some of the files might not be relevant.
        # `lambda f: f is not None` will filter out those.

        processed: Iterable[dict[RelevantFileId, _File]] = filter(
            lambda f: f is not None,  # Filter out the files that are not relevant
            await asyncio.gather(
                *map(process_file, dir_content)
            ),  # Process the files in parallel
        )

        # Index the relevant files on the relevantFileId
        self.relevant_files = {
            fid: rel for file in processed for fid, rel in file.items()
        }

        return ValidateCommand(
            relevant_files=self.relevant_files,
            directory=self.directory,
            model_name=self.model_name,
            model_path=self.model_path,
        )

    async def revert(self) -> None:
        """Revert the changes."""
        # Unset the attributes
        delattr(self, "model_name")
        delattr(self, "model_path")
        delattr(self, "relevant_files")
