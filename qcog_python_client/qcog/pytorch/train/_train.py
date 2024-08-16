from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


class TrainCommand(BoundedCommand):
    command: Command = Command.train
    model_guid: str
    dataset_guid: str
    training_parameters_guid: str


class TrainHandler(Handler[TrainCommand]):
    commands = (Command.train,)
    attempts = 1

    async def handle(self, payload: TrainCommand) -> None:
        post_request = self.get_tool("post_request")
        self.trained_model = await post_request(
            f"pytorch_model/{payload.model_guid}/trained_model",
            {
                "dataset_guid": payload.dataset_guid,
                "training_parameters_guid": payload.training_parameters_guid,
            },
        )

    async def revert(self) -> None:
        pass
