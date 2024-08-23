"""Handler for uploading the model to the server."""

import io
import os
import tarfile

import aiohttp

from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


def compress_folder(folder_path: str) -> io.BytesIO:
    """Compress a folder."""
    # We define the arcname as the basename of the folder
    # In this way we avoid the full path in the tar.gz file
    arcname = os.path.basename(folder_path)

    # What to exclude (__pycache__, .git, etc)
    def filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return (
            None
            if any(name in tarinfo.name for name in {".git", "__pycache__"})
            else tarinfo
        )

    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(folder_path, arcname=arcname, filter=filter)
        logger.info("Compressed targzip with memebers: ", tar.getnames())

    buffer.seek(0)

    return buffer


class UploadCommand(BoundedCommand):
    """Payload to dispatch an upload command."""

    upload_folder: str
    model_name: str
    command: Command = Command.upload


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
        tar_gzip_folder = compress_folder(folder_path)
        # Retrieve the multipart request tool
        post_multipart = self.get_tool("post_multipart")

        self.data = aiohttp.FormData()

        self.data.add_field(
            "model",
            tar_gzip_folder,
            filename=f"model-{payload.model_name}.tar.gz",
            content_type="application/gzip",
        )

        response = await post_multipart(
            f"pytorch_model/?model_name={payload.model_name}",
            self.data,
        )

        self.created_model = response

    async def revert(self) -> None:
        """Revert the changes."""
        delattr(self, "data")
