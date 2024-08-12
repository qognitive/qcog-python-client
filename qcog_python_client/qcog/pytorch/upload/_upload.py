from dataclasses import dataclass

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


@dataclass
class UploadCommand(BoundedCommand):
    command: Command = Command.upload


class UploadHandler(Handler[UploadCommand]):
    command: Command = Command.upload

    async def handle(self, payload: UploadCommand) -> None:
        print("-- Executing Upload Handler --")
        return None

    async def revert(self) -> None:
        pass
