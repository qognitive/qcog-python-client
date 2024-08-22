"""Define the interface for a monitoring service."""

from abc import ABC, abstractmethod


class Monitor(ABC):
    """Define the interface for a monitoring service."""

    @abstractmethod
    def init(self, *args, **kwargs):
        """Define the initialization method for the monitoring service."""
        ...

    @abstractmethod
    def log(self, *args, **kwargs):
        """Define the logging method for the monitoring service."""
        ...

    @abstractmethod
    def close(self):
        """Define the close method for the monitoring service."""
        ...
