"""Discover the module and the model."""

import asyncio
import base64
import os
from concurrent import futures

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler
from qcog_python_client.qcog.pytorch.validate._validate import ValidateCommand


class DiscoverCommand(BoundedCommand):
    model_name: str
    model_path: str
    command: Command = Command.discover


def pkg_name(package_path: str) -> str:
    """From the package path, get the package name."""
    return os.path.basename(package_path)


class DiscoverHandler(Handler):
    """Discover the folder and the model.

    Saves all the relevant files in a dictionary called
    relevant_files.

    For now the only relevant file is the model.py file.
    Eventually we will add also `requirements.txt` in case
    the user wants to install other dependencies.
    """

    model_module_name = "model.py"  # The name of the model module
    retries = 0
    commands = (Command.discover,)
    relevant_files: dict

    async def handle(self, payload: DiscoverCommand) -> ValidateCommand:
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
        executor = futures.ThreadPoolExecutor(max_workers=1)

        # NOTE: Eventually we can parallelize this operation.
        for item in content:
            if item == self.model_module_name:
                item_path = os.path.join(self.model_path, item)
                _, encoded_content = await read_async(executor, item_path)
                self.relevant_files.update(
                    {
                        "model_module": {
                            "path": item_path,
                            "content": encoded_content,
                            "pkg_name": pkg_name(self.model_path),
                        }
                    }
                )

        # Once the discovery has been completed,
        # Issue a validate command that will be executed next
        return ValidateCommand(
            relevant_files=self.relevant_files,
            model_name=self.model_name,
            model_path=self.model_path,
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
