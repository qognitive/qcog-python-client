import pytest

from qcog_python_client.qcog.pytorch.agent import PyTorchAgent


@pytest.mark.asyncio
async def test_pytorch_agent_discovery():
    """Test basic discovery of PyTorchAgent"""

    agent = PyTorchAgent()
    await agent.init("tests/pytorch_model", "test_model_00")
