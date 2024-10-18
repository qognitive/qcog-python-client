import base64
import csv
import gzip
import io
import json
from typing import TypedDict

import pandas as pd


class DataFramePayload(TypedDict):
    """DataFrame payload."""

    blob: str
    indexing: list[int]


def decode_base64(encoded_string: str) -> str:
    """Decode base64 string.

    From a base64 encoded str type, decode into original
    string str type

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


def base642dataframe(encoded_string: str) -> pd.DataFrame:
    """Decode base64 string and parse as csv dataframe.

    From a base64 encoded str type, decode into original
    string str type and parse as csv dataframe using io

    Parameters
    ----------
    encoded_string: str
        encoded base64 string

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


def encode_base64(data: pd.DataFrame) -> str:
    """Base64 encode a pandas dataframe.

    Take a normal pandas dataframe and encode as
    base64 "string" of csv export

    Parameters
    ----------
    data: pd.DataFrame
        dataframe to encode

    Returns
    -------
    str: encoded base64 string

    """
    indexing: list[int] = list(range(data.index.nlevels))

    raw_string: str = data.to_csv()
    payload: DataFramePayload = DataFramePayload(
        blob=raw_string,
        indexing=indexing,
    )
    raw_bytes: bytes = json.dumps(payload).encode("ascii")
    base64_bytes = base64.b64encode(raw_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


def compress_data(data: pd.DataFrame) -> io.BytesIO:
    """Compress a pandas dataframe.

    Take a normal pandas dataframe and compress as
    gzip "string" of csv export
    It adds the indexing of the columns as a csv.
    It uses a separator "---" to separate the csv and the indexing.

    Parameters
    ----------
    data: pd.DataFrame
        dataframe to compress

    Returns
    -------
    str: compressed gzip string

    """
    indexing: list[int] = list(range(data.index.nlevels))

    # Transform the indexing into a csv
    indexing_csv = io.StringIO()

    writer = csv.writer(indexing_csv)
    writer.writerow(indexing)
    indexing_csv.seek(0)

    data_csv: str = data.to_csv()
    buffer = io.BytesIO()

    with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
        f.write(data_csv.encode())
        f.write(b"\n---\n")
        f.write(indexing_csv.getvalue().encode())

    buffer.seek(0)

    return buffer
