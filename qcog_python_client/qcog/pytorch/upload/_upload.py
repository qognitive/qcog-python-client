import gzip
import io
import os
import tarfile
from dataclasses import dataclass

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


@dataclass
class UploadCommand(BoundedCommand):
    upload_folder: str
    model_name: str
    command: Command = Command.upload


class UploadHandler(Handler[UploadCommand]):
    command: Command = Command.upload
    attempts = 1

    async def handle(self, payload: UploadCommand) -> None:
        folder_path = payload.upload_folder
        # Compress the folder
        gzip_folder = compress_folder(folder_path)
        # Retrieve the multipart request tool
        post_multipart = self.get_tool("post_multipart")

        data = aiohttp.FormData()
        data.add_field(
            "file",
            gzip_folder,
            filename=f"model-{payload.model_name}.tar.gz",
            content_type="application/gzip",
        )

        response = await post_multipart(
            "pytorch_model",
            data,
        )

        print("---- Response ----")
        print(response)

    async def revert(self) -> None:
        pass
