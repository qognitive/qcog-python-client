"""Defines the interface for the chain of responsibility pattern."""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar


class BoundedCommand(Protocol):
    """Command type."""

    command: Command


CommandPayloadType = TypeVar("CommandPayloadType", bound=BoundedCommand)


class Command(enum.Enum):
    """Command enum."""

    discover = "discover"
    validate = "validate"
    upload = "upload"


class Handler(ABC, Generic[CommandPayloadType]):
    """Interface for the chain of responsibility pattern."""

    next: Handler | None
    executed: bool
    retries: int = 3
    retry_after: int = 3
    command: Command

    @abstractmethod
    def handle(self, payload: CommandPayloadType) -> CommandPayloadType:
        """Handle the data."""
        ...

    @abstractmethod
    def revert(self) -> None:
        """Revert the changes."""
        ...

    def set_next(self, next_component: Handler) -> Handler:
        """Set the next component."""
        self.next = next_component
        return next_component

    def handle_or_next(self, payload: CommandPayloadType) -> CommandPayloadType:
        """Handle the data or pass it to the next handler."""
        # If the handler matches the command in the payload
        # Attempt the execution of the command
        if self.command == payload.command:
            for i in range(self.retries):
                try:
                    result = self.handle(payload)
                    # If the command was successfully executed
                    # Set the executed flag to True
                    self.executed = True
                    # If there is a next handler
                    # Pass the payload to the next handler
                    if self.next:
                        return self.next.handle_or_next(result)

                except Exception as e:
                    # Try to handle the exception and retry the command
                    print(f"Exception at {self.command}: {e}")
                    print(f"Retrying {self.command}...")
                    print(f"Attempt {i+1}/{self.retries}")
                    self.revert()
                    # Try executing the command again
                    self.handle(payload)
                    time.sleep(self.retry_after)

        if self.next:
            return self.next.handle_or_next(payload)
