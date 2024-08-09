"""Discover the module and the model."""

import asyncio
import base64
import os
from concurrent import futures
from dataclasses import dataclass

from qcog_python_client.qcog.pytorch._validate import ValidateCommand
from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler


@dataclass
class DiscoverCommand(BoundedCommand):
    model_name: str
    model_path: str
    command: Command = Command.discover


class DiscoverHandler(Handler):
    """Discover the folder and the model.

    Saves all the relevant files in a dictionary called
    relevant_files.

    For now the only relevant file is the model.py file.
    Eventually we will add also `requirements.txt` in case
    the user wants to install other dependencies.
    """

    model_module_name = "model.py"
    retries = 0
    command = Command.discover
    relevant_files: dict

    async def handle(self, payload: DiscoverCommand) -> ValidateCommand:
        self.model_name = payload.model_name
        self.model_path = payload.model_path

        # Get the absolute path of the model module
        abs_path = os.path.abspath(payload.model_path)

        # Check if the model module exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model module not found at {payload.model_path}")

        # Check if the folder contains the model module
        content = os.listdir(abs_path)

        # Initialize the relevant files dictionary
        self.relevant_files = {}
        executor = futures.ThreadPoolExecutor(max_workers=1)

        # NOTE: Eventually we can parallelize this operation.
        for item in content:
            if item == self.model_module_name:
                item_path = os.path.join(abs_path, item)
                _, encoded_content = await read_async(executor, item_path)
                self.relevant_files.update({item_path: encoded_content})

        # Once the discovery has been completed,
        # Issue a validate command that will be executed next
        return ValidateCommand(
            relevant_files=self.relevant_files,
        )

    async def revert(self) -> None:
        # Unset the attributes
        delattr(self, "model_name")
        delattr(self, "model_path")
        delattr(self, "relevant_files")


async def read_async(
    executor: futures.ThreadPoolExecutor, file_path: str
) -> tuple[str, str]:
    loop = asyncio.get_running_loop()
    io_wrapper = await loop.run_in_executor(executor, open, file_path, "r")
    try:
        encoded = base64.b64encode(io_wrapper.read().encode()).decode()
        return file_path, encoded
    finally:
        io_wrapper.close()
