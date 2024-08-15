import io
import os
import tarfile

import aiohttp

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
        if "__pycache__" in tarinfo.name:
            return None
        if ".git" in tarinfo.name:
            return None
        return tarinfo

    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(folder_path, arcname=arcname, filter=filter)
        print("Compressed targzip with memebers: ", tar.getnames())

    buffer.seek(0)

    return buffer


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
        tar_gzip_folder = compress_folder(folder_path)
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
            "model",
            tar_gzip_folder,
            filename=f"model-{payload.model_name}.tar.gz",
            content_type="application/gzip",
        )

        response = await post_multipart(
            f"pytorch_model/?dataset_guid={dataset_guid}&training_parameters_guid={training_parameters_guid}&model_name={payload.model_name}",
            data,
        )

        self.created_model = response

    async def revert(self) -> None:
        pass
