from dataclasses import dataclass

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


@dataclass
class UploadPayload(BoundedCommand):
    command: Command = Command.discover


class UploadHandler(Handler):
    def handle(self, payload: UploadPayload):
        pass

    def revert(self):
        pass
