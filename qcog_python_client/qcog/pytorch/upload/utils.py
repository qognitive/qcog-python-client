"""Utility functions for uploading files to the QCoG platform."""

import io
import os
import tarfile

from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog.pytorch.types import Directory


def compress_folder(directory: Directory, folder_path: str) -> io.BytesIO:
    """Compress a folder from a in-memory directory.

    Parameters
    ----------
    directory : Directory
        The directory to compress, where the key is the path of the file
        and the value is a QFile object.

    folder_path : str
        The path of the folder to compress.

    Returns
    -------
    io.BytesIO
        The compressed folder as a BytesIO object representing a tarfile.

    """
    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for qfile in directory.values():
            # Get a relative path from the arcname
            rel_path = os.path.relpath(qfile.path, folder_path)
            logger.info(f"Adding {rel_path} to training package...")

            # Create TarInfo object and set the size
            tarfinfo = tarfile.TarInfo(name=rel_path)
            tarfinfo.size = len(qfile.content.getvalue())

            # Add the file to the tar archive
            tar.addfile(tarfinfo, fileobj=io.BytesIO(qfile.content.getvalue()))

    buffer.seek(0)

    return buffer
