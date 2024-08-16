from typing import Callable

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


class SaveParametersCommand(BoundedCommand):
    command: Command = Command.save_parameters
    parameters: dict


class SaveParametersHandler(Handler[SaveParametersCommand]):
    commands = (Command.save_parameters,)
    attempts = 1

    async def handle(self, payload: SaveParametersCommand) -> None:
        # Check if a dynamic model to validate the parameters
        # has been set on the context
        validate: Callable[[dict], None] | None = self.context.get("params_validator")

        if validate:
            validate(payload.parameters)

        post_request = self.get_tool("post_request")
        self.parameters_response = await post_request(
            "training_parameters",
            {"model": "pytorch", "parameters": payload.parameters},
        )

    async def revert(self) -> None:
        pass
