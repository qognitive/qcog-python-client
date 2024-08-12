import os

import aiohttp
import pytest

from qcog_python_client.qcog._httpclient import RequestClient
from qcog_python_client.qcog.pytorch.agent import PyTorchAgent


@pytest.mark.asyncio
async def test_pytorch_agent_discovery():
    """Test basic discovery of PyTorchAgent"""

    model_path = "tests/pytorch_model"
    model_name = "test_model_00"
    # Register custom tools for the agent.
    # The following `tools` will be available
    # inside the handlers of the agent
    request_client = RequestClient(
        token=os.getenv("API_TOKEN"),
        hostname="localhost",
        port=8000,
    )
    async def post_multipart(url: str, data: aiohttp.FormData) -> dict:
        return await request_client.post(
            url,
            data,
            content_type="data",
        )

    agent = PyTorchAgent.create_agent(
        post_request=request_client.post,
        get_request=request_client.get,
        post_multipart=post_multipart,
    )

    await agent.upload(model_path, model_name)
