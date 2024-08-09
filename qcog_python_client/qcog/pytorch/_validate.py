from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Command, Handler


class ValidatePayload(BoundedCommand):
    command: Command = Command.validate

class ValidateHandler(Handler):
    def handle(self, payload: ValidatePayload):
        pass

    def revert(self):
        pass
