"""Class to handle all the PyTorch operations."""

from typing import Any

from qcog_python_client.qcog.pytorch._discover import DiscoverHandler, DiscoverPayload
from qcog_python_client.qcog.pytorch._upload import UploadHandler
from qcog_python_client.qcog.pytorch._validate import ValidateHandler
from qcog_python_client.qcog.pytorch.handler import Handler


class PyTorchAgent:
    """PyTorch Agent."""

    def __init__(self):
        """Initialize the PyTorch Agent."""
        self.chain = self._init(DiscoverHandler(), ValidateHandler(), UploadHandler())

    def _init(self, *chain: Handler) -> Handler:
        """Initialize the Responsibility Chain."""
        for i in range(len(chain) - 1):
            chain[i].set_next(chain[i + 1])

        return chain[0]

    def init(self, model_path: str, model_name: str) -> None:
        """Start the PyTorch Agent."""
        self.chain.handle(payload=DiscoverPayload(
            model_name=model_name,
            model_path=model_path
        ))

    def train(self, data: Any) -> dict:
        """Train the model."""
        pass

    def inference(self, data: Any, model_name: str) -> Any:
        """Run inference."""
        pass
