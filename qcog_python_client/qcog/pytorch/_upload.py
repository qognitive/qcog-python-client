from dataclasses import dataclass

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


@dataclass
class UploadPayload(BoundedCommand):
    command: Command = Command.upload


class UploadHandler(Handler):
    command: Command = Command.upload

    def handle(self, payload: UploadPayload):
        print("-- Executing Upload Handler --")
        return None

    def revert(self):
        pass
