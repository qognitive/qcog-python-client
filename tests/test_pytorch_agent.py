import os

import pytest

from qcog_python_client.qcog import AsyncQcogClient
from tests.datasets import get_wbc_data

df_train, df_test, df_target = get_wbc_data()


@pytest.mark.asyncio
async def test_pytorch_workflow():
    client = await AsyncQcogClient.create(
        token=os.getenv("API_TOKEN"),
        hostname="localhost",
        port=8000,
    )

    client = await client.pytorch(
        model_name="test-model-05",
        model_path="tests/pytorch_model",
    )
    client = await client.data(df_train)

    await client.train_pytorch(
        {
            "batch_size": 32,
            "epochs": 10,
        }
    )
    print(await client.status())
    await client.wait_for_training(poll_time=10)
    print(client.metrics)
