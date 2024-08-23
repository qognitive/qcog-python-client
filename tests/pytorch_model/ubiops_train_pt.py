"""Wrapper around a customer defined training function.
Local development version for local testing.
"""

import base64
import gzip
import io
import json
import os
import re
import sys
from typing import Any, Callable, Literal, TypedDict

import pandas as pd
import torch

train_fn: Callable[..., dict] | None = None

# LOG PWD

print("-------> PWD: ", os.getcwd())

# LOG MODULE PATH

print("-------> MODULE PATH: ", __file__)

sys.path.append(os.getcwd())

# Try to import the customer function
try:
    from model import train as _train_fn

    train_fn = _train_fn
except ImportError as e:
    raise ImportError(
        "Failed to import the train function from the model.py file: ", e
    ) from e


def decode_base64(encoded_string: str) -> str:
    """Decode into original string str type.

    Parameters
    ----------
    encoded_string: str
        encoded base64 string

    Returns
    -------
    str: decoded string
    """
    base64_bytes: bytes = encoded_string.encode("ascii")
    decoded_bytes: bytes = base64.b64decode(base64_bytes)
    return decoded_bytes.decode("ascii")


Compression = Literal["gzip"]


class DataFramePayload(TypedDict):
    blob: str
    indexing: list[int]


def base642dataframe(encoded_string: str) -> pd.DataFrame:
    """Decode a base64 string and parse as DataFrame.

    Parameters
    ----------
    encoded_string: str
        base64 encoded string

    Returns
    -------
    pd.DataFrame: parsed csv dataframe
    """
    decoded_string: str = decode_base64(encoded_string)
    raw_dict: dict = json.loads(decoded_string)

    payload: DataFramePayload = DataFramePayload(
        blob=raw_dict["blob"],
        indexing=raw_dict["indexing"],
    )
    s = io.StringIO(payload["blob"])
    return pd.read_csv(s, index_col=payload["indexing"])


def base64Location2dataframe(  # noqa
    location: str, *, compression: Compression | None = None
) -> pd.DataFrame:
    """Return from a file location, a pandas dataframe.

    Parameters
    ----------
    location: str
        the location of the file to be read

    compression: 'gzip' | None
        specify if a compression has been applied to the file

    Returns
    -------
    pd.DataFrame: parsed csv dataframe
    """
    encoded_string: str | bytes | None = None

    import boto3

    s3_client = boto3.client("s3")
    key = "/".join(location.split("/")[3:])

    # Location has two parts that are interesting to us:
    # 1. The bucket name
    # 2. The key of the file in the bucket
    # The url is classic s3 url format: s3://bucket-name/key
    # We need to extract the bucket name and the key

    pattern = r"s3://(?P<bucket>[^/]+)/(?P<key>.+)"
    match = re.match(pattern, location)

    if not match:
        raise ValueError("Invalid s3 url format")

    bucket, key = match.group("bucket"), match.group("key")

    print("=============================")
    print("Bucket: ", bucket)
    print("Key: ", key)
    print("=============================")

    file_object = io.BytesIO()
    s3_client.download_fileobj(bucket, key, file_object)
    file_object.seek(0)

    # If the file is compressed, open it with the right method
    if compression == "gzip":
        with gzip.GzipFile(fileobj=file_object, mode="r") as f:
            encoded_string = f.read()

    # Otherwise assume it's just a base64 encoded string
    else:
        with open(location, "r") as f:
            encoded_string = f.read()

    # Decompress the file if it's a bytes object and gzip compression is specified
    if isinstance(encoded_string, bytes):
        encoded_string = encoded_string.decode("ascii")

    return base642dataframe(encoded_string)


# UbiOps training function
def train(
    training_data: str, parameters: dict, context: dict, base_directory: Any
) -> dict:
    """Train function adapter for UbiOps."""
    print("Context: ", context)
    if train_fn is None:
        raise ValueError("No train function found")

    training_data_location = training_data

    print("DEBUG: Training with parameters: ", parameters)
    print("Training data Location: ", training_data_location)
    print("Running Training Function...")

    data = base64Location2dataframe(training_data_location, compression="gzip")
    result = train_fn(data, **parameters["parameters"])

    print("Training Function Completed...")

    model: torch.nn.Module = result.get("model")
    metrics: dict = result.get("metrics", {})

    # Save the model
    artifact_filename = parameters.get("artifact_filename", "model.pt")
    torch.save(model, artifact_filename)

    return {
        "artifact": artifact_filename,
        "metadata": {},
        "metrics": {"run_id": context.get("run_id"), **metrics},
        "additional_output_files": [],
    }
