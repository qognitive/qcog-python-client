"""Define the interface for a monitoring service."""

from abc import ABC, abstractmethod
from typing import Any


class Monitor(ABC):
    """Define the interface for a monitoring service."""

    @abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Define the initialization method for the monitoring service."""
        ...

    @abstractmethod
    def log(self, *args: Any, **kwargs: Any) -> Any:
        """Define the logging method for the monitoring service."""
        ...

    @abstractmethod
    def close(self) -> Any:
        """Define the close method for the monitoring service."""
        ...
