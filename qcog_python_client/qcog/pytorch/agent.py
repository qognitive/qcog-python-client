"""Class to handle all the PyTorch operations."""

from typing import Any

from qcog_python_client.qcog.pytorch._discover import DiscoverCommand, DiscoverHandler
from qcog_python_client.qcog.pytorch._upload import UploadHandler
from qcog_python_client.qcog.pytorch._validate import ValidateHandler
from qcog_python_client.qcog.pytorch.handler import Handler


class PyTorchAgent:
    """PyTorch Agent."""

    def __init__(self) -> None:
        """Initialize the PyTorch Agent."""
        self.chain = self._init(DiscoverHandler(), ValidateHandler(), UploadHandler())

    def _init(self, *chain: Handler) -> Handler:
        """Initialize the Responsibility Chain."""
        if len(chain) == 1:
            raise ValueError("Chain must have at least 2 handlers.")
        head = chain[0]

        for i in range(len(chain) - 1):
            current_handler = chain[i]
            next_handler = chain[i + 1]
            current_handler.set_next(next_handler, head)

        return head

    async def init(self, model_path: str, model_name: str) -> None:
        """Start the PyTorch Agent."""
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

