from abc import ABC, abstractmethod
from typing import Literal, overload

import aiohttp
import pandas as pd


class ABCRequestClient(ABC):
    """Interface for a request client."""

    @abstractmethod
    async def get(self, url: str) -> dict:
        """Execute a get request."""
        ...

    @overload
    async def post(
        self, url: str, data: dict, *, content_type: Literal["json"] = "json"
    ) -> dict: ...

    @overload
    async def post(
        self,
        url: str,
        data: aiohttp.FormData,
        *,
        content_type: Literal["data"] = "data",
    ) -> dict: ...

    @abstractmethod
    async def post(
        self,
        url: str,
        data: dict | aiohttp.FormData,
        *,
        content_type: Literal["json", "data"] = "json",
    ) -> dict:
        """Execute a post request."""
        ...


class ABCDataClient(ABC):
    """Interface for a data client."""

    @abstractmethod
    async def upload_data(self, data: pd.DataFrame) -> dict:
        """Upload a dataframe."""
        ...
