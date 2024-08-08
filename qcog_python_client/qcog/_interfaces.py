from abc import ABC, abstractmethod

import pandas as pd


class ABCRequestClient(ABC):
    """Interface for a request client."""

    @abstractmethod
    async def get(self, url: str) -> dict:
        """Execute a get request."""
        ...

    @abstractmethod
    async def post(self, url: str, data: dict) -> dict:
        """Execute a post request."""
        ...


class ABCDataClient(ABC):
    """Interface for a data client."""

    @abstractmethod
    async def upload_data(self, data: pd.DataFrame) -> dict:
        """Upload a dataframe."""
        ...
