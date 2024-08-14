import gzip
import io
import os
import tarfile

import aiohttp

from qcog_python_client.qcog.pytorch.handler import (
    BoundedCommand,
    Command,
    Handler,
)


def compress_folder(folder_path: str) -> bytes:
    """Compress a folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                tar.add(os.path.join(root, file))
    buffer.seek(0)
    return gzip.compress(buffer.read())


class UploadCommand(BoundedCommand):
    upload_folder: str
    model_name: str
    command: Command = Command.upload


class UploadHandler(Handler[UploadCommand]):
    commands = (Command.upload,)
    attempts = 1

    async def handle(self, payload: UploadCommand) -> None:
        folder_path = payload.upload_folder
        # Compress the folder
        gzip_folder = compress_folder(folder_path)
        # Retrieve the multipart request tool
        post_multipart = self.get_tool("post_multipart")
        # Retrieve dataset_guid, training_parameters_guid
        # from the context
        dataset_guid = self.context.get("dataset_guid")
        training_parameters_guid = self.context.get("training_parameters_guid")

        if not dataset_guid:
            raise ValueError("Dataset guid not found in the context.")

        if not training_parameters_guid:
            raise ValueError("Training parameters guid not found in the context.")

        data = aiohttp.FormData()

        data.add_field(
            "file",
            gzip_folder,
            filename=f"model-{payload.model_name}.tar.gz",
            content_type="application/gzip",
        )

        response = await post_multipart(
            f"pytorch_model/?dataset_guid={dataset_guid}&training_parameters_guid={training_parameters_guid}",
            data,
        )

        print("Response:", response)

    async def revert(self) -> None:
        pass
