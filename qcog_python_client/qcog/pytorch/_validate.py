from dataclasses import dataclass

from qcog_python_client.qcog.pytorch._upload import UploadPayload
from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler


@dataclass
class ValidatePayload(BoundedCommand):
    relevant_files: dict
    command: Command = Command.validate


class ValidateHandler(Handler):
    command: Command = Command.validate

    def handle(self, payload: ValidatePayload):
        print("-- Executing Validate Handler --")
        print("Validating relevant files: ", payload.relevant_files)
        # Once validation has been completed,
        # Issue the next command that is the upload
        return UploadPayload()

    def revert(self):
        pass
