"""Class to handle all the PyTorch operations."""

from typing import Any, Callable, Coroutine, cast

from qcog_python_client.qcog.pytorch.discover._discover import (
    DiscoverCommand,
    DiscoverHandler,
)
from qcog_python_client.qcog.pytorch.handler import Handler, ToolFn
from qcog_python_client.qcog.pytorch.parameters._saveparameters import (
    SaveParametersCommand,
    SaveParametersHandler,
)
from qcog_python_client.qcog.pytorch.train._train import TrainCommand, TrainHandler
from qcog_python_client.qcog.pytorch.upload._upload import UploadHandler
from qcog_python_client.qcog.pytorch.validate._validate import ValidateHandler
from qcog_python_client.schema.common import PytorchTrainingParameters


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

    def _get_tool(self, tool_name: str) -> ToolFn:
        """Get a tool from the tools dictionary."""
        tool = self.tools.get(tool_name)

        if not tool:
            raise ValueError(
                f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}"  # noqa
            )

        return tool

    def set_context(self, key: str, value: Any) -> None:
        """Set a value on the chain context."""
        self.chain.context[key] = value

    def init(self, *chain: Handler) -> Handler:
        """Initialize the Responsibility Chain."""
        if len(chain) == 1:
            raise ValueError("Chain must have at least 2 handlers.")
        head = chain[0]

        context: dict = {}

        for i in range(len(chain) - 1):
            current_handler = chain[i]
            next_handler = chain[i + 1]
            current_handler.set_next(next_handler, head)

        # Register the get_tool function
        for handler in chain:
            handler.get_tool = self._get_tool
            handler.context = context

        self._chain = head
        return head

    async def upload_model(self, model_path: str, model_name: str) -> dict:
        """Upload the model to the server."""
        # Init Command will dispatch a Discover Command
        handler = await self.chain.dispatch(
            payload=DiscoverCommand(
                model_name=model_name,
                model_path=model_path,
                dispatch_next=True,  # Will follow the whole chain
            )
        )
        upload_handler = cast(UploadHandler, handler)
        # We dont wanna expose the chain or the handlers outside of the PytorchAgent
        return upload_handler.created_model

    async def train_model(self,
        model_guid: str,
        *,
        dataset_guid: str,
        training_parameters: dict,
    ) -> dict:
        """Train the model."""
        # Upload training command
        save_parameters_handler: SaveParametersHandler = await self.chain.dispatch(
            payload=SaveParametersCommand(
                parameters=training_parameters
            )
        )

        training_parameters_guid = save_parameters_handler.parameters_response['guid']

        train_handler = await self.chain.dispatch(
            payload=TrainCommand(
                model_guid=model_guid,
                dataset_guid=dataset_guid,
                training_parameters_guid=training_parameters_guid,
            )
        )

        handler = cast(TrainHandler, train_handler)
        return handler.trained_model

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

    @classmethod
    def create_agent(
        cls,
    ) -> "PyTorchAgent":
        """Create a PyTorch Agent with a http client for making requests."""
        agent = cls()
        agent.init(
            DiscoverHandler(),
            ValidateHandler(),
            UploadHandler(),
            SaveParametersHandler(),
            TrainHandler(),
        )
        return agent
