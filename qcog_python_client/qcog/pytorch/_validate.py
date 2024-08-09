from dataclasses import dataclass

from qcog_python_client.qcog.pytorch._upload import UploadCommand
from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler


@dataclass
class ValidateCommand(BoundedCommand):
    relevant_files: dict
    command: Command = Command.validate


class ValidateHandler(Handler):
    command: Command = Command.validate

    async def handle(self, payload: ValidateCommand) -> UploadCommand:
        print("-- Executing Validate Handler --")
        print("Validating relevant files: ", payload.relevant_files)
        # Once validation has been completed,
        # Issue an upload command
        return UploadCommand()

    async def revert(self) -> None:
        pass
