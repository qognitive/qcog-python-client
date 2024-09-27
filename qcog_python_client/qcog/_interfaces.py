from abc import ABC, abstractmethod
from typing import Any, overload

import aiohttp
import pandas as pd


class IRequestClient(ABC):
    """Interface for a request client."""

    @abstractmethod
    async def get(self, url: str) -> dict:
        """Execute a get request. Returns a single object."""
        ...

    @abstractmethod
    async def get_many(self, url: str) -> list[dict]:
        """Execute a get request. Returns a list of objects."""
        ...

    @overload
    async def post(self, url: str, data: dict) -> dict: ...

    @overload
    async def post(
        self,
        url: str,
        data: aiohttp.FormData,
    ) -> dict: ...

    @abstractmethod
    async def post(
        self,
        url: str,
        data: dict | aiohttp.FormData,
    ) -> dict:
        """Execute a post request."""
        ...

    @property
    def headers(self) -> dict:
        """Get the headers."""
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        """Get the base url of the request."""
        raise NotImplementedError


class IDataClient(ABC):
    """Interface for a data client."""

    @abstractmethod
    async def upload_data(self, data: pd.DataFrame) -> dict:
        """Upload a dataframe."""
        ...

    @abstractmethod
    async def stream_data(self, data: Any, *, dataset_id: str) -> dict:
        """Download a dataframe."""
        ...
