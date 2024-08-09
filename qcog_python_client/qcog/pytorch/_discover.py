"""Discover the module and the model."""

import os
from dataclasses import dataclass

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler


@dataclass
class DiscoverPayload(BoundedCommand):
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

    def handle(self, payload: DiscoverPayload):
        self.model_name = payload.model_name
        self.model_path = payload.model_path

        # Get the absolute path of the model module
        abs_path = os.path.abspath(payload.model_path)

        # Check if the model module exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model module not found at {payload.model_path}")

        # Check if the folder contains the model module
        content = os.listdir(abs_path)

        print(content)

    def revert(self):
        pass
