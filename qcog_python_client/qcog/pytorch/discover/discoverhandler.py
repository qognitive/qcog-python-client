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
    Iterable,
)

from anyio import open_file

from qcog_python_client.qcog.pytorch import utils
from qcog_python_client.qcog.pytorch.discover.types import MaybeIsRelevantFile
from qcog_python_client.qcog.pytorch.discover.utils import pkg_name
from qcog_python_client.qcog.pytorch.handler import Command, Handler
from qcog_python_client.qcog.pytorch.types import (
    Directory,
    DiscoverCommand,
    QFile,
    RelevantFileId,
    RelevantFiles,
    ValidateCommand,
)


async def _maybe_model_module(self: DiscoverHandler, file: QFile) -> QFile | None:
    """Check if the file is the model module."""
    module_name = os.path.basename(file.path)
    if module_name == self.model_module_name:
        return file
    return None


async def _maybe_monitor_service_import_module(
    self: DiscoverHandler, file: QFile
) -> QFile | None:
    """Check if the file is importing the monitor service."""
    # Make sure the item is not a folder. If so, exit
    if os.path.isdir(file.path):
        return None

    tree = ast.parse(file.content.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # We assume that the only way that the monitor service
            # will be imported is like
            # from qcog_python_client import monitor or eventually
            # from qcog_python_client import monitor as <alias>.
            if node.module == "qcog_python_client" and any(
                a.name == "monitor" for a in node.names
            ):
                return file
    return None


relevant_files_map: dict[RelevantFileId, MaybeIsRelevantFile] = {
    "model_module": _maybe_model_module,  # type: ignore
    "monitor_service_import_module": _maybe_monitor_service_import_module,  # type: ignore
}


async def maybe_relevant_file(
    self: DiscoverHandler,
    file: QFile,
) -> dict[RelevantFileId, QFile]:
    """Check if the file is relevant."""
    retval: dict[RelevantFileId, QFile] = {}

    for relevant_file_id, _maybe_relevant_file_fn in relevant_files_map.items():
        relevant_file = await _maybe_relevant_file_fn(self, file)
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
    relevant_files: RelevantFiles

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
        self.directory: Directory = {}
        pkg_name_ = pkg_name(self.model_path)

        for item in content:
            item_path = os.path.join(self.model_path, item)
            # filter by exclusion rules
            if utils.exclude(item_path):
                continue
            # Avoid folders
            if os.path.isdir(item_path):
                continue

            async with await open_file(item_path, "rb") as file:
                self.directory[item_path] = QFile.model_validate(
                    {
                        "path": item_path,
                        "filename": item,
                        "content": io.BytesIO(await file.read()),
                        "pkg_name": pkg_name_,
                    }
                )

        # Figure it out which files are relevant.
        # Relevant files have a specific id that
        # is specified in the `relevant_files_map`
        # dictionary as the key of that dictionary.

        # Initialize the relevant files dictionary
        self.relevant_files = {}

        # Process the files in parallel gathering the results
        # from the coroutines returned by the `process_file`
        # function. Some of the files might not be relevant.
        # `lambda f: f is not None` will filter out those.

        processed: Iterable[dict[RelevantFileId, QFile]] = filter(
            lambda f: f is not None,  # Filter out the files that are not relevant
            await asyncio.gather(
                *map(lambda f: maybe_relevant_file(self, f), self.directory.values())
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
