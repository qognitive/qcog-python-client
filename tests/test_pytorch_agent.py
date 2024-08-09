from qcog_python_client.qcog.pytorch.agent import PyTorchAgent


def test_pytorch_agent_discovery():
    agent = PyTorchAgent()
    agent.init("tests/pytorch_model", "test_model_00")
