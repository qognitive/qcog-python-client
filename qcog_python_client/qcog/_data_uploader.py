"""Subclass to upload data.

This is a separate class because it will be heavily modified in order
to support multi part uploads or other types of uploads.
"""

from pandas.core.api import DataFrame as DataFrame

from qcog_python_client.qcog._base64utils import encode_base64
from qcog_python_client.qcog._interfaces import ABCDataClient, ABCRequestClient
from qcog_python_client.schema import DatasetPayload


class DataClient(ABCDataClient):
    """Data Client Uploader.

    Current implementation that relies on a classic http post request.
    """

    def __init__(self, http_client: ABCRequestClient) -> None:
        self.http_client = http_client

    async def upload_data(self, data: DataFrame) -> dict:
        data_payload = DatasetPayload(
            format="dataframe",
            source="client",
            data=encode_base64(data),
            project_guid=None,
        ).model_dump()

        return await self.http_client.post("dataset", data_payload)
