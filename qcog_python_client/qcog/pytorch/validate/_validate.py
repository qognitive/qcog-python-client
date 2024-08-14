from typing import Any, Callable

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler
from qcog_python_client.qcog.pytorch.upload._upload import UploadCommand
from qcog_python_client.qcog.pytorch.validate._validate_module import (
    FileToValidate,
    validate_model_module,
)


class ValidateCommand(BoundedCommand):
    model_name: str
    model_path: str
    relevant_files: dict
    command: Command = Command.validate


class ValidateHandler(Handler):
    commands = (Command.validate,)
    attempts = 1

    validate_map: dict[str, Callable[[FileToValidate], Any]] = {
        "model_module": validate_model_module
    }

    async def handle(self, payload: ValidateCommand) -> UploadCommand:
        validated: list = []

        for key, validate_fn in self.validate_map.items():
            file = payload.relevant_files.get(key)

            if not file:
                raise FileNotFoundError(f"File {key} not found in the relevant files.")

            parsed = FileToValidate.model_validate(file)
            validated.append(validate_fn(parsed))

        return UploadCommand(
            upload_folder=payload.model_path, model_name=payload.model_name
        )

    async def revert(self) -> None:
        pass
