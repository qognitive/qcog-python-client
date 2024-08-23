"""Handle the training of the model."""

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


class TrainCommand(BoundedCommand):
    """Payload to dispatch a train command."""

    command: Command = Command.train
    model_guid: str
    dataset_guid: str
    training_parameters_guid: str


class TrainHandler(Handler[TrainCommand]):
    """Train the model."""

    commands = (Command.train,)
    attempts = 1

    async def handle(self, payload: TrainCommand) -> None:
        """Handle the training.

        Parameters
        ----------
        payload : TrainCommand
            The payload to train the model
            model_guid : str
                The model that will be trained
            dataset_guid : str
                The dataset to be used for training
            training_parameters_guid : str
                The training parameters to be used for training

        """
        post_request = self.get_tool("post_request")
        self.trained_model = await post_request(
            f"pytorch_model/{payload.model_guid}/trained_model",
            {
                "dataset_guid": payload.dataset_guid,
                "training_parameters_guid": payload.training_parameters_guid,
            },
        )

    async def revert(self) -> None:
        """Revert the changes."""
        pass
