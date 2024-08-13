"""In order to run the tests, you need to have a valid API_TOKEN.

**Run train test**
```bash
export API_TOKEN=<your_api_token>
pytest tests/test_example_1.py -s -vv -k test_train_one

Once the test is executed you should get the id of the trained model.

**Run inference test**
```bash
export API_TOKEN=<your_api_token>
export TRAINED_MODEL=<trained_model_id>
pytest tests/test_example_1.py -s -vv -k test_inference_one
```
"""

import os
import sys

import pytest

from qcog_python_client import AsyncQcogClient, QcogClient
from qcog_python_client.schema import (
    GradOptimizationParameters,
    LOBPCGFastStateParameters,
)
from qcog_python_client.schema.generated_schema.models import TrainingStatus
from tests.datasets import get_wbc_data

API_TOKEN = os.getenv("API_TOKEN")
TRAINED_MODEL = os.getenv("TRAINED_MODEL")

df_train, df_test, df_target = get_wbc_data()


@pytest.mark.skip(reason="Run this test manually")
def test_train_one():
    """Run training test."""
    if API_TOKEN is None:
        raise ValueError("API_TOKEN not found in environment variables")

    qcml = QcogClient.create(token=API_TOKEN, hostname="localhost", port=8000)

    # If a model is already trained and the guid is specified,
    # then load the model

    if TRAINED_MODEL is not None:
        qcml = qcml.preloaded_model(TRAINED_MODEL)
        # Check the status of the train
        status = qcml.status()

        if status != TrainingStatus.completed:
            raise ValueError(f"Model is not trained yet. Status: {status}")

        else:
            print(f"Model {TRAINED_MODEL} is already trained. Skipping training.")
            return

    qcml.data(df_train)

    qcml = qcml.ensemble(
        operators=df_train.columns.tolist(),
        dim=64,
        num_axes=64,
    )

    weight_optimization = GradOptimizationParameters(
        iterations=10,
        learning_rate=1e-3,
    )

    get_states_extra = LOBPCGFastStateParameters(
        iterations=10,
    )

    qcml = qcml.train(
        batch_size=10,
        num_passes=5,
        weight_optimization=weight_optimization,
        get_states_extra=get_states_extra,
    )

    print("*** WAITING FOR TRAINING ***")
    qcml.wait_for_training(poll_time=10)
    print("Progress: ", qcml.progress())
    print("----- MODEL TRAINED -----")
    print(qcml.trained_model["guid"])
    print("----- LOSS -----")
    print(qcml.get_loss())


@pytest.mark.skip(reason="Run this test manually")
async def test_train_one_async():
    """Run training test."""
    if API_TOKEN is None:
        raise ValueError("API_TOKEN not found in environment variables")

    qcml = await AsyncQcogClient.create(
        token=API_TOKEN, hostname="localhost", port=8000
    )

    await qcml.data(df_train)

    qcml = qcml.ensemble(
        operators=df_train.columns.tolist(),
        dim=64,
        num_axes=64,
    )

    weight_optimization = GradOptimizationParameters(
        iterations=10,
        learning_rate=1e-3,
    )

    get_states_extra = LOBPCGFastStateParameters(
        iterations=10,
    )

    qcml = await qcml.train(
        batch_size=10,
        num_passes=5,
        weight_optimization=weight_optimization,
        get_states_extra=get_states_extra,
    )

    print("*** WAITING FOR TRAINING ***")
    await qcml.wait_for_training(poll_time=10)
    print("Progress: ", await qcml.progress())
    print("----- MODEL TRAINED -----")
    print(qcml.trained_model["guid"])
    print("----- LOSS -----")
    print(await qcml.get_loss())


@pytest.mark.skip(reason="Run this test manually")
def test_inference_one():
    """Run inference test."""
    qcml = QcogClient.create(token=API_TOKEN, hostname="localhost", port=8000)

    if TRAINED_MODEL is None:
        raise ValueError("TRAINED_MODEL not found in environment variables")

    qcml.preloaded_model(TRAINED_MODEL)

    result_df = qcml.inference(
        data=df_test,
        parameters={
            "state_parameters": LOBPCGFastStateParameters(
                learning_rate_axes=1e-3,
                iterations=10,
            )
        },
    )

    num_correct = result_df.idxmax(axis=1) == df_target.idxmax(axis=1)

    print(f"Accuracy: {num_correct.sum() / len(num_correct)}")


@pytest.mark.skip(reason="Run this test manually")
async def test_inference_one_async():
    """Run inference test."""
    qcml = await AsyncQcogClient.create(
        token=API_TOKEN,
        hostname="localhost",
        port=8000,
    )

    if TRAINED_MODEL is None:
        raise ValueError("TRAINED_MODEL not found in environment variables")

    await qcml.preloaded_model(TRAINED_MODEL)

    result_df = await qcml.inference(
        data=df_test,
        parameters={
            "state_parameters": LOBPCGFastStateParameters(
                learning_rate_axes=1e-3,
                iterations=10,
            )
        },
    )

    num_correct = result_df.idxmax(axis=1) == df_target.idxmax(axis=1)

    print(f"Accuracy: {num_correct.sum() / len(num_correct)}")


if __name__ == "__main__":
    import asyncio

    cmd = sys.argv[1]

    if cmd == "train":
        test_train_one()

    elif cmd == "train_async":
        asyncio.run(test_train_one_async())

    elif cmd == "inference_async":
        asyncio.run(test_inference_one_async())

    elif cmd == "inference":
        test_inference_one()
