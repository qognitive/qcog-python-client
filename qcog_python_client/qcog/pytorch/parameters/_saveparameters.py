from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)
from qcog_python_client.schema.common import PytorchTrainingParameters


class SaveParametersCommand(BoundedCommand):
    command: Command = Command.save_parameters
    parameters: PytorchTrainingParameters


class SaveParametersHandler(Handler[SaveParametersCommand]):
    commands = (Command.save_parameters,)
    attempts = 1

    async def handle(self, payload: SaveParametersCommand) -> None:
        post_request = self.get_tool("post_request")
        self.parameters_response = await post_request(
            "training_parameters",
            {"model": "pytorch", "parameters": payload.parameters.model_dump()},
        )

    async def revert(self) -> None:
        pass
