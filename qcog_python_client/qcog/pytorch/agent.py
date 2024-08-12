"""Class to handle all the PyTorch operations."""

from typing import Any, Callable, Coroutine

from qcog_python_client.qcog.pytorch.discover._discover import (
    DiscoverCommand,
    DiscoverHandler,
)
from qcog_python_client.qcog.pytorch.handler import Handler, ToolFn
from qcog_python_client.qcog.pytorch.upload._upload import UploadHandler
from qcog_python_client.qcog.pytorch.validate._validate import ValidateHandler


class PyTorchAgent:
    """PyTorch Agent."""

    def __init__(self) -> None:
        """Initialize the PyTorch Agent."""
        # Tools should be async functions that return a dictionary
        self.tools: dict[str, Callable[..., Coroutine[Any, Any, dict]]] = {}
        self._chain: Handler | None = None

    @property
    def chain(self) -> Handler:
        """Get the chain of responsibility."""
        if not self._chain:
            raise ValueError("Chain not initialized. Call the `init` method first.")
        return self._chain

    def init(self) -> None:
        """Initialize the PyTorch Agent.

        This method needs to be called after registering the tools.
        """
        # Chain of Responsibility
        self._chain = self._init(
            DiscoverHandler(),
            ValidateHandler(),
            UploadHandler(),
        )

    def _get_tool(self, tool_name: str) -> ToolFn:
        """Get a tool from the tools dictionary."""
        tool = self.tools.get(tool_name)

        if not tool:
            raise ValueError(
                f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}"  # noqa
            )

        return tool

    def _init(self, *chain: Handler) -> Handler:
        """Initialize the Responsibility Chain."""
        if len(chain) == 1:
            raise ValueError("Chain must have at least 2 handlers.")
        head = chain[0]

        for i in range(len(chain) - 1):
            current_handler = chain[i]
            next_handler = chain[i + 1]
            current_handler.set_next(next_handler, head)

        # Register the get_tool function
        for handler in chain:
            handler.get_tool = self._get_tool

        return head

    async def upload(self, model_path: str, model_name: str) -> None:
        """Upload the model to the server."""
        # Init Command will dispatch a Discover Command
        await self.chain.dispatch(
            payload=DiscoverCommand(model_name=model_name, model_path=model_path)
        )

    async def train(self, data: Any) -> dict:
        """Train the model."""
        raise NotImplementedError()

    async def inference(self, data: Any, model_name: str) -> Any:
        """Run inference."""
        raise NotImplementedError()

    def register_tool(
        self,
        tool_name: str,
        fn: ToolFn,
    ) -> None:
        """Register a tool to be used by the handlers in the chain."""
        self.tools[tool_name] = fn
