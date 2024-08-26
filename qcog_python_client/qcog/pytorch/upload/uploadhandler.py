"""Handler for uploading the model to the server."""

import aiohttp

from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog.pytorch.handler import (
    Command,
    Handler,
)
from qcog_python_client.qcog.pytorch.types import UploadCommand
from qcog_python_client.qcog.pytorch.upload.utils import compress_folder


class UploadHandler(Handler[UploadCommand]):
    """Upload the model to the server.

    It sets the created_model attribute on the handler.

    The model has the following attributes:
    training_parameters_guid: str
        The parameters associated with that model

    dataset_guid: str
        The dataset associated with that model

    qcog_version: str
        The version of the model. In case of a pytorch model,
        the version will be `pytorch-{model_name}`

    project_guid: str
        The project guid

    experiment_name: str
        The experiment name returned by the server.
        The shape is `training-pytorch-{model_name}`

    run_guid: str
        The run guid returned by the server. It's associated with
        the experiment guid if no train run has been executed yet.

    status: str
        Is set to `unknown` if no train run has been executed yet.

    """

    commands = (Command.upload,)
    attempts = 1
    data: aiohttp.FormData

    async def handle(self, payload: UploadCommand) -> None:
        """Handle the upload."""
        folder_path = payload.upload_folder
        # Compress the folder
        tar_gzip_folder = compress_folder(payload.directory, folder_path)
        # Retrieve the multipart request tool
        post_multipart = self.get_tool("post_multipart")

        self.data = aiohttp.FormData()
        self.data.add_field(
            "model",
            tar_gzip_folder,
            filename=f"model-{payload.model_name}.tar.gz",
            content_type="application/gzip",
        )

        assert self.data is not None
        logger.info(f"Uploading model {payload.model_name} to the server")
        logger.info(f"Type of data: {type(self.data)}")

        response = await post_multipart(
            f"pytorch_model/?model_name={payload.model_name}",
            self.data,
        )

        self.created_model = response

    async def revert(self) -> None:
        """Revert the changes."""
        pass
