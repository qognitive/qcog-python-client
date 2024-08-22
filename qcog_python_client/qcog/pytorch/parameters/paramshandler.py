"""Handle the saving of the parameters."""

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


class SaveParametersCommand(BoundedCommand):
    """Payload to dispatch a save parameters command."""

    command: Command = Command.save_parameters
    parameters: dict


class SaveParametersHandler(Handler[SaveParametersCommand]):
    """Save the parameters."""

    commands = (Command.save_parameters,)
    attempts = 1

    async def handle(self, payload: SaveParametersCommand) -> None:
        """Handle the saving of the parameters.

        Parameters
        ----------
        payload : SaveParametersCommand
            The payload to save the parameters
            parameters : dict
                The parameters to be saved

        """
        post_request = self.get_tool("post_request")
        self.parameters_response = await post_request(
            "training_parameters",
            {"model": "pytorch", "parameters": payload.parameters},
        )

    async def revert(self) -> None:
        """Revert the changes."""
        pass
