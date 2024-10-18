"""Subclass to upload data.

This is a separate class because it will be heavily modified in order
to support multi part uploads or other types of uploads.
"""

import aiohttp
from pandas.core.api import DataFrame as DataFrame

from qcog_python_client.qcog._base64utils import compress_data, encode_base64
from qcog_python_client.qcog._interfaces import IDataClient, IRequestClient
from qcog_python_client.schema import DatasetPayload


class DataClient(IDataClient):
    """Data Client Uploader.

    Current implementation that relies on a classic http post request.
    """

    def __init__(self, http_client: IRequestClient) -> None:
        self.http_client = http_client

    async def upload_data(self, data: DataFrame) -> dict:
        data_payload = DatasetPayload(
            format="dataframe",
            source="client",
            data=encode_base64(data),
            project_guid=None,
        ).model_dump()

        return await self.http_client.post("dataset", data_payload)

    async def stream_data(
        self,
        data: DataFrame,
        *,
        dataset_id: str,
        encoding: str = "gzip",
    ) -> dict:
        """Stream data to the server.

        This method will stream the data to the server in chunks.

        Parameters
        ----------
        data : DataFrame
            The data to stream to the server.
        dataset_id : str
            The ID of the dataset to stream the data to.
            This should be unique for each Dataset.
        encoding : str
            The encoding of the data.

        """
        headers = self.http_client.headers
        base_url = self.http_client.base_url
        url = f"{base_url}/dataset/upload?dataset_id={dataset_id}&format=dataframe&source=client&encoding={encoding}"  # noqa: E501

        # Zip gzip the data
        zip_data = compress_data(data)

        form = aiohttp.FormData()
        form.add_field(
            "file", zip_data, filename="data.csv", content_type="application/gzip"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=form) as response:
                    response.raise_for_status()
                    data_: dict = await response.json()
                    return data_
        except Exception as e:
            raise e
