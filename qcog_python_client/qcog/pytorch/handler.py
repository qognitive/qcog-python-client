"""Defines the interface for the chain of responsibility pattern.

Commands get dispatched through the chain of handlers. Each handler
checks if it can handle the command. If it can, it executes the command.
If the handler returns a new command, it gets dispatched again though the chain.

Each handler implements an automatic retry mechanism.
The number of attempts and the time to wait can be configured for each handler.

In case of error, the `revert` method is called to undo any changes made by the handler.
"""

from __future__ import annotations

import asyncio
import enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Generic, TypeAlias, TypeVar

from pydantic import BaseModel


class BoundedCommand(BaseModel):
    """Command type."""

    command: Command
    dispatch_next: bool = False


CommandPayloadType = TypeVar("CommandPayloadType", bound=BoundedCommand)
ToolName: TypeAlias = str
ToolFn: TypeAlias = Callable[..., Coroutine[Any, Any, dict]]


class Command(enum.Enum):
    """Command enum."""

    save_parameters = "save_parameters"
    discover = "discover"
    validate = "validate"
    upload = "upload"
    revert_all = "revert_all"
    train = "train"


class Handler(ABC, Generic[CommandPayloadType]):
    """Interface for the chain of responsibility pattern."""

    head: Handler  # Reference to the head of the chain
    next: Handler | None = None
    attempts: int = 3
    retry_after: int = 3
    commands: tuple[Command]
    get_tool: Callable[[ToolName], ToolFn]
    context: dict

    @abstractmethod
    async def handle(self, payload: CommandPayloadType) -> CommandPayloadType | None:
        """Handle the data."""
        ...

    @abstractmethod
    async def revert(self) -> None:
        """Revert the changes."""
        ...

    def set_next(self, next_component: Handler, head: Handler) -> Handler:
        """Set the next component."""
        self.next = next_component
        self.head = head
        return next_component

    async def dispatch(self, payload: CommandPayloadType) -> Handler:
        """Dispatch the payload through the chain."""
        # If the command is a `revert_all` command
        # and the handler has already been executed
        # we want to revert the state of the handler
        # and pass the command to the next handler
        if payload.command == Command.revert_all:
            await self.revert()
            if self.next:
                return await self.next.dispatch(payload)
            return self

        # If the handler matches the command in the payload
        # Attempt the execution of the command
        if payload.command in self.commands:
            for i in range(self.attempts):
                try:
                    return await execute_and_dispatch_next(self, payload)
                except Exception as e:
                    # Try again based on the number of attempts.
                    # 1 - revert the state of the handler
                    # 2 - wait for the specified time
                    # 3 - try again
                    print(f"Attempt {i}, error: {e}")
                    await self.revert()
                    await asyncio.sleep(self.retry_after)
                    return await execute_and_dispatch_next(self, payload)

        # If there is a next handler, dispatch the payload to the next handler
        if self.next:
            return await self.next.dispatch(payload)

        raise AttributeError(f"Command {payload.command} not found in the chain")

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.commands})"


async def execute_and_dispatch_next(
    handler: Handler, command: CommandPayloadType
) -> Handler:
    """Execute the command and dispatch the next command.

    If no next command is found, return the current handler.
    """
    next_command = await handler.handle(command)

    # If the returned payload is a command
    # And the command is set to dispatch the next command
    # And there is a next handler, then dispatch the next command
    if next_command and command.dispatch_next and handler.next:
        # If the command is set to `dispatch_next`,
        # continue the chain and set the next command
        # to be dispatched next
        next_command.dispatch_next = True
        return await handler.next.dispatch(next_command)

    # Otherwise, return the current handler
    return handler
