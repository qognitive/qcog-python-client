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

    head: Handler  # Reference to the head of the chain
    next: Handler | None = None
    attempts: int = 3
    retry_after: int = 3
    command: Command

    @abstractmethod
    def handle(self, payload: CommandPayloadType) -> CommandPayloadType | None:
        """Handle the data."""
        ...

    @abstractmethod
    def revert(self) -> None:
        """Revert the changes."""
        ...

    def set_next(self, next_component: Handler, head: Handler) -> Handler:
        """Set the next component."""
        self.next = next_component
        self.head = head
        return next_component

    def dispatch(self, payload: CommandPayloadType) -> Handler:
        """Dispatch the payload through the chain."""
        # If the handler matches the command in the payload
        # Attempt the execution of the command
        if self.command == payload.command:
            for i in range(self.attempts):
                try:
                    next_command = self.handle(payload)
                    # If another command has been issued by the handler
                    # dispatch
                    if next_command:
                        self.head.dispatch(next_command)

                    # Otherwise return the handler
                    return self
                except Exception as e:
                    print(f"Error executing command {self.command}: {e}")
                    # Try to handle the exception and retry the command
                    self.revert()
                    # Try executing the command again
                    self.handle(payload)
                    time.sleep(self.retry_after)

        # Otherwise dispatch to the next handler
        if self.next:
            return self.next.dispatch(payload)

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.command})"
