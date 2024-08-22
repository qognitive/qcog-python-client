"""Class to handle all the PyTorch operations."""

from __future__ import annotations

from typing import Any, cast

from qcog_python_client.qcog.pytorch.discover.discoverhandler import (
    DiscoverCommand,
    DiscoverHandler,
)
from qcog_python_client.qcog.pytorch.handler import Handler, ToolFn, ToolName
from qcog_python_client.qcog.pytorch.parameters.paramshandler import (
    SaveParametersCommand,
    SaveParametersHandler,
)
from qcog_python_client.qcog.pytorch.train.trainhandler import (
    TrainCommand,
    TrainHandler,
)
from qcog_python_client.qcog.pytorch.upload.uploadhandler import UploadHandler
from qcog_python_client.qcog.pytorch.validate.validatehandler import ValidateHandler


class PyTorchAgent:
    """PyTorch Agent.

    This is the Client of the chain of responsibility pattern.
    It offers a simple API to interact with the chain and it abstracts
    all the handlers and tools from the user.

    Some of the methods can be used to provide a context to the chain,
    like the `set_context` method or `register_tool` in a completely
    agnostic and unopinionated way.

    Once the agent is instantiated using `create_agent` class method,
    The chain is initialized on the `_chain` attribute.

    When dispatching a command, the chain will be traversed and the
    handlers will be executed in order.

    Once a handler responsible for the command is found, the `handle`
    method will be called on the handler.

    `handle` can return another command to be dispatched or `None`.

    The chain will continue if the command has the `dispatch_next` attribute
    set to `True`.

    If no command is returned or `dispatch_next` is `False`, the chain will
    stop and the `dispatch` method will return the last handler executed.
    """

    def __init__(self) -> None:
        """Initialize the PyTorch Agent."""
        self._chain: Handler | None = None

    @property
    def chain(self) -> Handler:
        """Get the chain of responsibility."""
        if not self._chain:
            raise AttributeError("Chain not initialized. Call the `init` method first.")
        return self._chain

    ########################################
    # Private Methods
    ########################################

    def _init(self, *chain: Handler) -> Handler:
        """Chain initialization method.

        It creates the context and the tools dictionaries and sets the
        reference on the handlers.

        It also calls the `set_next` method on each handler to set the
        reference to the next handler in the chain and the head of the chain.

        This method should not be called directly but being part of an
        initialization factory method like `create_agent`.

        Parameters
        ----------
        chain : Handler
            The chain of handlers to be initialized.

        """
        if len(chain) < 1:
            raise ValueError("Chain must have at least 2 handlers.")
        head = chain[0]

        context: dict = {}
        tools: dict[ToolName, ToolFn] = {}

        for i in range(len(chain) - 1):
            current_handler = chain[i]
            next_handler = chain[i + 1]
            current_handler.set_next(next_handler, head)

        # Pass context and tools reference to the handlers
        for handler in chain:
            handler._context = context
            handler._tools = tools

        self._chain = head
        return head

    ########################################
    # Public Methods
    ########################################

    def register_tool(
        self,
        tool_name: ToolName,
        fn: ToolFn,
    ) -> None:
        """Register a tool to be used by the handlers in the chain.

        Tools are injected into the handlers and can be used to perform
        operations that are not directly related to the chain of responsibility.

        For example, a tool can be used to make HTTP requests, read files, etc.

        Tools should be async functions that return the result of the operation
        in a dictionary format.

        Parameters
        ----------
        tool_name : ToolName
            The name of the tool as a string
        fn : ToolFn
            The function that will be called by the handlers
            as a Callable[..., Awaitable[dict]].

        """
        if self.chain._tools is None:
            raise AttributeError("Chain not initialized. Call the `init` method first.")
        self.chain._tools[tool_name] = fn

    async def upload_model(self, model_path: str, model_name: str) -> dict:
        """Upload the model to the server.

        Parameters
        ----------
        model_path : str
            The path to the model module
        model_name : str
            The name of the model

        Returns
        -------
        dict
            The uploaded model

        """
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

    async def train_model(
        self,
        model_guid: str,
        *,
        dataset_guid: str,
        training_parameters: dict,
    ) -> dict:
        """Train the model.

        Start a training session for a the model with guid `model_guid`

        Parameters
        ----------
        model_guid : str
            The guid of the model to train

        dataset_guid : str
            The guid of the dataset to use for training

        training_parameters : dict
            The training parameters

        Returns
        -------
        dict
            The trained model

        """
        # Upload training command

        handler = await self.chain.dispatch(
            payload=SaveParametersCommand(parameters=training_parameters)
        )
        save_parameters_handler = cast(SaveParametersHandler, handler)

        training_parameters_guid = save_parameters_handler.parameters_response["guid"]

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

    @classmethod
    def create_agent(
        cls,
    ) -> PyTorchAgent:
        """Create a PyTorch Agent.

        Initialize the chain of responsibility with the handlers.
        """
        agent = cls()
        agent._init(
            DiscoverHandler(),
            ValidateHandler(),
            UploadHandler(),
            SaveParametersHandler(),
            TrainHandler(),
        )
        return agent
